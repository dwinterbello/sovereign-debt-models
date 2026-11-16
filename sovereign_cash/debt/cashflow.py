"""Marketable interest from the netted ISIN book.

Fixed coupons stay on the bond that carries them. Floaters reset each year to the bill
yield plus the spread implied by the tape. Bills with a maturity pay the remaining-life
yield until they mature and, after they roll, the forward yield of that bill's own issue
tenor (4w–52w from the security description / issue→maturity), not a flat 3m. Japan's MoF
stock is scaled onto the TDB maturity ladder when the tape has bill rows.

Linker uplift accrues with inflation on the indexed principal. The gap between this
marketable bill and the y0 interest outturn is non-marketable (loans, postal savings,
definition differences). It is a level, and a rate shock does not move it.

Central-bank holdings: Japan has BoJ face by issue (`issuance/cache/boj_holdings` →
`official_isin_flow`). Coupons paid to the BoJ are remitted back to MoF, so the
general-government peer path nets them (US: debt held by the public). EA PSPP/PEPP
use a QP ISIN allocation for float/DV01; that book is not yet wired into interest.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.debt.isin_book import outstanding

HERE_DATA = Path(__file__).resolve().parents[1] / "data" / "debt"
REPO = Path(__file__).resolve().parents[2]
OFFICIAL_ISIN = REPO / "issuance" / "cache" / "official_isin_flow.parquet"
# Countries whose official_isin_flow rows are ISIN-exact holdings (not QP).
CB_NET_CC = frozenset({"jp"})


@lru_cache(maxsize=4)
def book(y0: int) -> pd.DataFrame:
    return outstanding(pd.Timestamp(f"{y0}-12-31"))


def _index_ratios(y0: int) -> dict[str, float]:
    """Indexed principal / par face. UK from the DMO uplift rebuild. US and France from the
    issuer's linker stock, which already includes uplift."""
    out = {cc: 1.0 for cc in ("us", "uk", "de", "fr", "it", "jp")}
    u = pd.read_parquet(HERE_DATA / "uk_uplift.parquet")
    nom = u[(u["var"] == "il_nominal_bn") & (u["year"] == y0)]
    up = u[(u["var"] == "il_uplift_bn") & (u["year"] == y0)]
    if len(nom) and len(up) and float(nom["value"].iloc[0]) > 0:
        out["uk"] = float(nom["value"].iloc[0] + up["value"].iloc[0]) / float(nom["value"].iloc[0])
    off = pd.read_parquet(HERE_DATA / "official_stock.parquet")
    b = book(y0)
    for cc, code in (("us", "US"), ("fr", "FR")):
        row = off[(off["country"] == cc) & (off["var"] == "linker_bn")].sort_values("year")
        tape = b[(b["country"] == code) & (b["sec_type"] == "Linker")]["face_mn"].sum() / 1000.0
        if len(row) and tape > 0:
            out[cc] = float(row["value"].iloc[-1]) / tape
    from sovereign_cash.debt.index_ratio import country_ratios
    extra = country_ratios(b, pd.Timestamp(f"{y0}-12-31"), fallback=True)
    for cc in ("de", "it", "jp"):
        if extra.get(cc, 1.0) > 1.0:
            out[cc] = extra[cc]
    return out


def _weight(mat: pd.Series, year: int) -> pd.Series:
    y = mat.dt.year
    return (y > year).astype(float) + 0.5 * (y == year).astype(float)


def _official_face(cc: str, asof: pd.Timestamp) -> pd.DataFrame:
    """ISIN face held by the domestic central bank at the latest month ≤ ``asof``.

    Empty if the official ISIN cache is missing or has no rows for ``cc``.
    """
    if not OFFICIAL_ISIN.exists():
        return pd.DataFrame(columns=["isin", "official_stock_mn", "maturity_dt"])
    h = pd.read_parquet(OFFICIAL_ISIN)
    h = h[h["country"].astype(str).str.lower().eq(cc.lower())].copy()
    if h.empty:
        return pd.DataFrame(columns=["isin", "official_stock_mn", "maturity_dt"])
    h["month"] = pd.to_datetime(h["month"])
    cut = h.loc[h["month"] <= asof, "month"].max()
    if not pd.notna(cut):
        cut = h["month"].max()
    sub = h[h["month"].eq(cut)].copy()
    face = pd.to_numeric(sub["official_stock_mn"], errors="coerce").fillna(0.0)
    sub = sub.loc[face > 0, ["isin", "official_stock_mn", "maturity_dt"]].copy()
    sub["official_stock_mn"] = face.loc[sub.index].astype(float)
    sub["maturity_dt"] = pd.to_datetime(sub["maturity_dt"], errors="coerce")
    return sub


def cb_coupon_path(cc: str, y0: int, years: list[int]) -> pd.Series:
    """Coupon cash (local bn) on end-``y0`` central-bank holdings, aged by maturity.

    Face is capped at the coupon-book amount for that ISIN. QT runoff is maturity only
    (no reinvestment onto the remittance path — BoJ still buys, but new purchases are
    small under the published plan and would raise remittances, lowering net further).
    Returns an empty series when ``cc`` is not on the ISIN-exact net list or the cache
    is missing.
    """
    if cc.lower() not in CB_NET_CC:
        return pd.Series(dtype=float)
    asof = pd.Timestamp(f"{y0}-12-31")
    off = _official_face(cc, asof)
    if off.empty:
        return pd.Series(dtype=float)
    b = book(y0)
    code = cc.upper()
    coup = b[(b["country"] == code) & (b["sec_type"] != "Bill")][
        ["key", "face_mn", "coupon_pct", "maturity_dt"]
    ].rename(columns={"key": "isin"})
    m = off.merge(coup, on="isin", how="inner", suffixes=("_off", "_bk"))
    if m.empty:
        return pd.Series(dtype=float)
    mat = m["maturity_dt_bk"] if "maturity_dt_bk" in m.columns else m["maturity_dt"]
    if "maturity_dt_off" in m.columns:
        mat = mat.fillna(m["maturity_dt_off"])
    mat = pd.to_datetime(mat, errors="coerce")
    cpn = m["coupon_pct"].fillna(0.0).to_numpy(float) / 100.0
    face = np.minimum(
        m["official_stock_mn"].to_numpy(float),
        m["face_mn"].to_numpy(float),
    )
    out = {}
    for y in years:
        w = _weight(mat, int(y)).to_numpy(float)
        out[int(y)] = float((cpn * face / 1000.0 * w).sum())
    return pd.Series(out, dtype=float).sort_index()


def components(y0: int, years: list[int], infl_fn) -> pd.DataFrame:
    """One row per country × year.

    ``infl_fn(cc, year)`` is the decimal accrual rate for linker principal that year.
    """
    b = book(y0)
    ir0 = _index_ratios(y0)
    cb = {cc: cb_coupon_path(cc, y0, years) for cc in CB_NET_CC}
    rows = []
    for cc_up, g in b.groupby("country"):
        cc = str(cc_up).lower()
        ir = ir0.get(cc, 1.0)
        bills = g[g["sec_type"] == "Bill"]
        frn = g[g["sec_type"] == "FRN"]
        link = g[g["sec_type"] == "Linker"]
        fixed = g[g["sec_type"].isin(["Nominal", "Linker"])]
        # a stored FRN coupon at or below a 2% short rate is the spread; above that it is a full coupon
        spr = frn["coupon_pct"]
        known = frn[spr.notna() & (frn["face_mn"] > 0)]
        # A missing floater coupon is a missing discount margin, not a 1% spread.
        frn_spread = float(np.average(np.minimum(known["coupon_pct"], 1.5) / 100.0, weights=known["face_mn"])) if len(known) else 0.0
        indexed = ir
        level = 1.0
        cb_cc = cb.get(cc, pd.Series(dtype=float))
        for y in years:
            w_fix = _weight(fixed["maturity_dt"], y) if len(fixed) else None
            if len(fixed):
                cpn = fixed["coupon_pct"].fillna(0.0) / 100.0
                scale = np.where(fixed["sec_type"].eq("Linker"), indexed * level, 1.0)
                coupon = float((cpn * scale * fixed["face_mn"] / 1000.0 * w_fix).sum())
            else:
                coupon = 0.0
            w_frn = _weight(frn["maturity_dt"], y) if len(frn) else pd.Series(dtype=float)
            frn_face = float((frn["face_mn"] / 1000.0 * w_frn).sum()) if len(frn) else 0.0
            w_lk = _weight(link["maturity_dt"], y) if len(link) else pd.Series(dtype=float)
            lk_face = float((link["face_mn"] / 1000.0 * w_lk).sum()) if len(link) else 0.0
            inf = float(infl_fn(cc, y))
            uplift = lk_face * indexed * level * inf
            redeem = 0.0
            coup_bonds = g[g["sec_type"] != "Bill"]
            if len(coup_bonds):
                redeem = float(coup_bonds.loc[coup_bonds["maturity_dt"].dt.year == y, "face_mn"].sum()) / 1000.0
            rows.append({
                "cc": cc, "year": y,
                "coupon_bn": coupon,
                "cb_coupon_bn": float(cb_cc.get(int(y), 0.0)) if len(cb_cc) else 0.0,
                "uplift_bn": uplift,
                "frn_face_bn": frn_face,
                "frn_spread": frn_spread,
                "linker_face_bn": lk_face,
                "index_ratio": indexed * level,
                "bill_face_bn": float(bills["face_mn"].sum()) / 1000.0 if len(bills) else 0.0,
                "redeem_bn": redeem,
                "bill_rows": int(len(bills)),
            })
            level *= 1.0 + inf
    return pd.DataFrame(rows)


def _bill_issue_tenor(bills: pd.DataFrame) -> np.ndarray:
    """Original issue life in years for each live bill row.

    Prefer ``issue_tenor_yrs`` from the outstanding book (issue→maturity / sec_desc). Fall
    back to residual life capped at 1y, then 0.25 only if nothing else is known.
    """
    n = len(bills)
    if n == 0:
        return np.asarray([], dtype=float)
    if "issue_tenor_yrs" in bills.columns:
        t = pd.to_numeric(bills["issue_tenor_yrs"], errors="coerce").to_numpy(float)
    else:
        t = np.full(n, np.nan)
    if "resid_yrs" in bills.columns:
        resid = pd.to_numeric(bills["resid_yrs"], errors="coerce").to_numpy(float)
    else:
        resid = np.full(n, np.nan)
    out = np.where(np.isfinite(t) & (t > 0), t, np.minimum(np.where(np.isfinite(resid), resid, 0.25), 1.0))
    return np.clip(out, 1.0 / 365.0, 1.0)


def bill_ladder(bills: pd.DataFrame, year: int, y0: int, rate) -> float:
    """Interest, local bn, on the bill rows.

    ``rate(start, tenor)`` is the decimal yield of a bill issued ``start`` years after
    end-y0 with life ``tenor`` years. Until each bill matures it earns the yield of its
    remaining life read at end-y0. After it rolls, the replacement earns the forward yield
    of that bill's own issue tenor (from the security description), not a flat 3m.
    """
    if bills.empty:
        return 0.0
    t0 = year - (y0 + 1)
    t1 = t0 + 1.0
    asof = pd.Timestamp(f"{y0}-12-31")
    resid = (bills["maturity_dt"] - asof).dt.days / 365.25
    face = bills["face_mn"].to_numpy(float)
    resid = resid.to_numpy(float)
    tenor = _bill_issue_tenor(bills)
    total = 0.0
    for f, r, ten in zip(face, resid, tenor):
        r = max(float(r), 1.0 / 365.0)
        locked = float(rate(0.0, min(r, 1.0)))
        a, b = max(t0, 0.0), min(t1, r)
        if b > a:
            total += f * locked * (b - a)
        c, d = max(t0, r), t1
        if d > c:
            total += f * float(rate(c, float(ten))) * (d - c)
    return total / 1000.0


def attach_japan_bills(frame: pd.DataFrame, bills_bn: float) -> pd.DataFrame:
    """MoF short-term + financing-bill stock onto the Japan rows.

    When the ops tape has TDB maturities, keep ``bill_rows`` so the maturity ladder prices
    the stock; only the face is replaced by the MoF total (tape is ~85% of MoF). With no
    tape rows the charge stays a flat 3m on the whole stock.
    """
    if not bills_bn or not np.isfinite(bills_bn):
        return frame
    m = frame["cc"].eq("jp")
    frame.loc[m, "bill_face_bn"] = float(bills_bn)
    return frame
