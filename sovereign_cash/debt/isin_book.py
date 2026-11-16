"""Cash interest from the live ISIN book, not a residual coupon.

Coupons: ``issuance/cache/ops_*.parquet``. The UK tape leaves ``coupon`` empty; it is parsed from
``sec_desc_long`` with ``uk_flows.dmo.parse_coupon``. Junk coupons outside 0–20 are dropped.

Bills: same tape where the rows exist (US, DE, FR, IT, UK, JP — UK bills have a maturity and no ISIN).
Each live bill carries ``issue_tenor_yrs`` from its security description / issue→maturity so rolls
are priced at the book's own tenors (4w–52w), not a flat 3m. A bill pays no coupon.

Existing bonds keep their coupon. This module does not price new issuance and does not touch the curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "issuance" / "cache"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "uk_flows") not in sys.path:
    sys.path.insert(0, str(REPO / "uk_flows"))


def _ops() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(CACHE.glob("ops_*.parquet"))]
    ops = pd.concat(frames, ignore_index=True)
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["maturity_dt"] = pd.to_datetime(ops["maturity_dt"])
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    ops["coupon"] = pd.to_numeric(ops["coupon"], errors="coerce")
    return ops


def _coupon(ops: pd.DataFrame) -> pd.Series:
    """Coupon in percent. A valid tap (0–20) wins. Otherwise the gilt name. Still clamped to 0–20:
    AFT writes 20361.25 for a 1.25% May-2036 OAT, and the name parser then adds the year."""
    from sovereign_cash._vendor import parse_coupon

    c = ops["coupon"].where((ops["coupon"] >= 0) & (ops["coupon"] <= 20))
    # a later bad tap must not wipe an earlier good one
    good = c.groupby([ops["country"], ops["isin"].astype("string")]).transform("median")
    # A floater's description is a tenor ("Note 2-Year"). The name parser reads that as a coupon.
    parsed = ops["sec_desc_long"].map(parse_coupon)
    parsed = parsed.where((parsed >= 0) & (parsed <= 20))
    parsed = parsed.where(ops["sec_type"] != "FRN")
    return good.fillna(parsed)


def outstanding(asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per live instrument. Face in local million. Bills keyed by maturity when ISIN is missing."""
    ops = _ops()
    asof = pd.Timestamp(asof or ops["op_dt"].max())
    ops = ops[(ops["op_dt"] <= asof) & (ops["maturity_dt"] > asof) & ops["amount_issued"].notna()].copy()
    ops["coupon_pct"] = _coupon(ops)
    ops["key"] = ops["isin"].astype("string")
    missing = ops["key"].isna() | ops["key"].isin(["<NA>", "None", ""])
    ops.loc[missing, "key"] = (
        ops.loc[missing, "country"].astype(str) + "|bill|" + ops.loc[missing, "maturity_dt"].dt.strftime("%Y-%m-%d")
        + "|" + ops.loc[missing].groupby(["country", "maturity_dt"]).cumcount().astype(str)
    )
    # Buybacks are negative taps. Net them before the type is taken from the issued rows.
    issued = ops[ops["amount_issued"] > 0].copy()
    g = ops.groupby(["country", "key"], as_index=False).agg(face_mn=("amount_issued", "sum"),
                                                            maturity_dt=("maturity_dt", "max"))
    typ = issued.groupby(["country", "key"])["sec_type"].agg(lambda s: s.mode().iloc[0] if len(s) else "Nominal")
    cpn = issued.groupby(["country", "key"])["coupon_pct"].median()
    g = g.merge(typ.rename("sec_type"), on=["country", "key"], how="left")
    g = g.merge(cpn.rename("coupon_pct"), on=["country", "key"], how="left")
    # Bill original life from issue→maturity (sec_desc_long is "Bill 13-Week", "BTF 52w", "TDB 1y"…).
    # Face-weighted across taps of the same key so rolls price at the book's tenors, not a flat 3m.
    if "issue_dt" in issued.columns:
        iss = pd.to_datetime(issued["issue_dt"], errors="coerce").fillna(issued["op_dt"])
    else:
        iss = issued["op_dt"]
    issued["_issue_tenor"] = (issued["maturity_dt"] - iss).dt.days / 365.25
    bill_tap = issued[(issued["sec_type"] == "Bill") & issued["_issue_tenor"].gt(0)]
    if len(bill_tap):
        def _wavg(s: pd.Series) -> float:
            w = bill_tap.loc[s.index, "amount_issued"].to_numpy(float)
            v = s.to_numpy(float)
            return float(np.average(v, weights=w)) if w.sum() > 0 else float(np.nanmean(v))

        tenor = bill_tap.groupby(["country", "key"])["_issue_tenor"].agg(_wavg)
        desc = bill_tap.groupby(["country", "key"])["sec_desc_long"].agg(
            lambda s: s.mode().iloc[0] if len(s) and s.mode().size else (s.iloc[0] if len(s) else "")
        )
        g = g.merge(tenor.rename("issue_tenor_yrs"), on=["country", "key"], how="left")
        g = g.merge(desc.rename("sec_desc_long"), on=["country", "key"], how="left")
    else:
        g["issue_tenor_yrs"] = np.nan
        g["sec_desc_long"] = pd.NA
    g = g[g["face_mn"] > 1.0]
    g["resid_yrs"] = (g["maturity_dt"] - asof).dt.days / 365.25
    g["asof"] = asof
    g["coupon_source"] = np.where(g["sec_type"].eq("Bill"), "discount",
                                  np.where(g["coupon_pct"].notna(), "ops or name", "missing"))
    return g


def legacy_cash(y0: int) -> pd.DataFrame:
    """Coupon cash and redemptions, local bn, of the coupon stock outstanding at end-``y0``.

    Face nets buybacks. ``coupon_bn`` pays the coupon
    in full if the bond is alive all year and half if it matures that year. ``face_bn`` is the end-``y0``
    stock, repeated on every row so a caller can scale redemptions onto the debt identity.
    """
    book = outstanding(pd.Timestamp(f"{y0}-12-31"))
    coup = book[book["sec_type"] != "Bill"]
    rows = []
    for cc, g in coup.groupby("country"):
        cpn = g["coupon_pct"].fillna(0.0) / 100.0
        face = g["face_mn"] / 1000.0
        my = g["maturity_dt"].dt.year
        face0 = float(face.sum())
        for y in range(y0, y0 + 8):
            alive = my > y
            dies = my == y
            rows.append({
                "cc": str(cc).lower(), "year": y,
                "coupon_bn": float((cpn * face * (alive.astype(float) + 0.5 * dies.astype(float))).sum()),
                "redeem_bn": float(face[dies].sum()),
                "face_bn": face0,
            })
    return pd.DataFrame(rows)


def locked_interest(book: pd.DataFrame, year: int, bill_rate: float) -> pd.DataFrame:
    """Interest in local bn during ``year`` from instruments already outstanding.

    A coupon bond pays its coupon if it is alive at mid-year, half if it matures during the year.
    Bills are discount instruments. A constant stock rolled inside the year costs ``bill_rate`` times
    the stock, whatever the average life — the life only sets how fast a rate change lands, and every
    bill in this tape matures inside a year, so a full year is entirely at ``bill_rate``. The life is
    reported so the ladder is visible. ``bill_rate`` is a decimal.
    """
    rows = []
    for cc, g in book.groupby("country"):
        coup = g[g["sec_type"] != "Bill"]
        alive = coup["maturity_dt"].dt.year > year
        dies = coup["maturity_dt"].dt.year == year
        cpn = coup["coupon_pct"].fillna(0.0) / 100.0
        interest = float((cpn * coup["face_mn"] * (alive.astype(float) + 0.5 * dies.astype(float))).sum()) / 1000.0
        missing = int(coup["coupon_pct"].isna().sum())
        bills = g[g["sec_type"] == "Bill"]
        if bills.empty:
            b_int, b_face, wam, b_note = 0.0, 0.0, np.nan, "no bill rows in the tape"
        else:
            b_face = float(bills["face_mn"].sum()) / 1000.0
            wam = float(np.average(bills["resid_yrs"].clip(lower=1 / 365), weights=bills["face_mn"]))
            b_int = b_face * bill_rate
            b_note = f"maturity ladder, WAM {wam:.2f}y, stock rolled at bill_rate"
        rows.append({"cc": cc.lower(), "year": year, "coupon_interest_bn": interest,
                     "bill_interest_bn": b_int, "interest_bn": interest + b_int,
                     "bill_face_bn": b_face, "bill_wam": wam, "coupons_missing": missing,
                     "bill_note": b_note})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    book = outstanding()
    asof = book["asof"].iloc[0]
    print("asof", asof.date())
    face = book.groupby(["country", "sec_type"]).face_mn.sum().unstack(fill_value=0) / 1000.0
    print(face.round(1))
    coup = book[book.sec_type.ne("Bill")]
    print("coupons missing", int(coup.coupon_pct.isna().sum()), "of", len(coup))
    rates = {"US": 0.0394, "UK": 0.0384, "DE": 0.0264, "FR": 0.0264, "IT": 0.0264, "JP": 0.0156}
    frames = []
    for y in (asof.year, asof.year + 1, asof.year + 2, asof.year + 3):
        one = []
        for cc, g in book.groupby("country"):
            one.append(locked_interest(g, y, rates[cc]))
        frames.append(pd.concat(one, ignore_index=True))
    out = pd.concat(frames, ignore_index=True)
    print(out.pivot(index="year", columns="cc", values="interest_bn").round(1))
    y1 = int(out.year.min()) + 1
    print(out[out.year == y1][["cc", "coupon_interest_bn", "bill_interest_bn", "bill_face_bn",
                               "bill_wam", "coupons_missing", "bill_note"]].round(2).to_string(index=False))
