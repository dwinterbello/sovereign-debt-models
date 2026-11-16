"""US coupon-size scenarios on free-float absorption.

The base is the published forward: Aug-2026 refunding sizes held, buybacks at the last quarter's
pace, SOMA rollover on top of announced sizes. Each scenario perturbs that forward analytically:

  flow   + extra face per product x that forward row's DV01 per dollar (same bucket, same yield)
         + (buyback scale - 1) x the forward buyback DV01
  stock  + cumulative extra issuance DV01 (a 12-month horizon, so aging inside it is ignored)

Absorption is then 12m private flow over private stock per sleeve, exactly as `imbalance_panel`.

Sizes come from the financing gap, not from a guess at a coupon path. Dealers put FY27-28 at
$1.45tn more than Aug-2026 coupon sizes fund (TBAC minutes, 5 Aug 2026). Treasury's own refunding
charts carry three ways to fund it: Coupons Constant (bills take it), Prorated (bill share held at
22.2%, coupons take 77.8%) and Bills Constant (coupons take all). Increases phase in over three
refunding quarters, as in Aug-2023 to Feb-2024, and the full monthly increment is the one that
funds the scenario's share of the gap by Sep 2028.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from issuance.float_rv import _onto_stock, buyback_forward
from issuance.forward import _us_identity
from issuance.imbalance import SLEEVE_OF, SLEEVES
from issuance.paths import CACHE

GAP_BN = 1450.0
GAP_END = pd.Timestamp("2028-09-01")
BILL_SHARE = 0.222
# Monthly size increase per term, Jul-2023 to Apr-2024, $bn (10y/20y/30y on reopening months).
LAST_CYCLE = {"2y": 27, "3y": 18, "5y": 27, "7y": 9, "10y": 7, "20y": 1, "30y": 4, "frn": 6}
FRONT = {k: LAST_CYCLE[k] for k in ("2y", "3y", "5y", "7y", "frn")}

SCENARIOS = [
    {"key": "base", "name": "Base: Aug-26 sizes held", "start": None, "gap_share": 0.0, "mix": None, "buyback": 1.0,
     "why": "Treasury guidance holds sizes for at least the next several quarters; Coupons Constant funds the gap with bills."},
    {"key": "front", "name": "Front-end first, May-27", "start": "2027-05-01", "gap_share": 1 - BILL_SHARE, "mix": FRONT,
     "buyback": 1.0, "why": "Dealers: increases sometime in 2027, guidance changed a few quarters ahead. Market expects 2y-7y first. Prorated funding."},
    {"key": "cycle", "name": "2023-24 mix, May-27", "start": "2027-05-01", "gap_share": 1 - BILL_SHARE, "mix": LAST_CYCLE,
     "buyback": 1.0, "why": "Repeat the last increase cycle's term mix (5% to 20y and 30y). Prorated funding."},
    {"key": "prorata", "name": "Pro rata, Feb-27", "start": "2027-02-01", "gap_share": 1.0, "mix": "prorata",
     "buyback": 1.0, "why": "Earliest plausible start (guidance changed 4 Nov). Every nominal term and FRN up the same percent. Bills Constant."},
    {"key": "nobb", "name": "Buybacks stop", "start": None, "gap_share": 0.0, "mix": None, "buyback": 0.0,
     "why": "Sizes held; liquidity-support and cash-management buybacks halted from Oct-26."},
    {"key": "stress", "name": "Stress: pro rata Feb-27, no buybacks", "start": "2027-02-01", "gap_share": 1.0,
     "mix": "prorata", "buyback": 0.0, "why": "Upper bound on long-end supply inside the horizon."},
]


EARLIEST_START = pd.Timestamp("2027-02-01")  # Aug-2026 guidance: sizes held for at least several quarters
REFUNDING_MONTHS = (2, 5, 8, 11)


def bridge_scenario() -> dict | None:
    """Gap and start from the fiscal path instead of the dealer survey (`us/fiscal/supply_bridge`).

    Gap: net bills that fund FY27-28 with Aug-2026 coupon sizes held (Coupons Constant).
    Start: first refunding on or after EARLIEST_START at which the next twelve months'
    bill funding share (net bills / net marketable) exceeds Treasury's Prorated 22.2%.
    """
    from issuance.fiscal_bridge import BRIDGE

    meta = BRIDGE.with_suffix(".json")
    if not BRIDGE.exists() or not meta.exists():
        return None
    gap = json.loads(meta.read_text()).get("gap_fy27_28", {})
    if not gap.get("complete"):
        return None
    b = pd.read_parquet(BRIDGE).loc["unchanged"]
    b.index = pd.to_datetime(b.index)
    nb = b["net_bills_total_bn"].rolling(12).sum().shift(-11)
    nc = b["net_coupons_bn"].rolling(12).sum().shift(-11)
    share = nb / (nb + nc)
    ok = share[(share.index >= EARLIEST_START) & share.index.month.isin(REFUNDING_MONTHS) & (share > BILL_SHARE)]
    if ok.empty:
        return None
    start = ok.index[0]
    g = float(gap["net_bills_bn"])
    return {"key": "bridge", "name": f"Fiscal bridge: ${g / 1000:.2f}tn gap, {start:%b-%y}", "start": str(start.date()),
            "gap_share": 1 - BILL_SHARE, "mix": LAST_CYCLE, "buyback": 1.0, "gap_bn": g,
            "why": f"Gap = FY27-28 net bills on the fiscal path with coupons held (dealers: ${GAP_BN / 1000:.2f}tn). "
                   f"Start = first refunding where the next 12m bill share exceeds {BILL_SHARE:.1%} "
                   f"({float(ok.iloc[0]):.1%}). 2023-24 mix, Prorated."}


def phase_weight(months: pd.DatetimeIndex, start: pd.Timestamp) -> np.ndarray:
    """1/3, 2/3, then 1 per refunding quarter from the start month; 0 before it."""
    gap = (months.year - start.year) * 12 + (months.month - start.month)
    w = np.minimum(gap // 3 + 1, 3) / 3.0
    return np.where(gap >= 0, w, 0.0)


def full_increment_bn(start: str | None, gap_share: float, gap_bn: float = GAP_BN) -> float:
    """Monthly increase at full phase-in that funds gap_share of the FY27-28 gap by GAP_END."""
    if start is None or gap_share <= 0:
        return 0.0
    s = pd.Timestamp(start)
    months = pd.date_range(s, GAP_END, freq="MS")
    return gap_share * gap_bn / float(phase_weight(months, s).sum())


def _us_forward() -> pd.DataFrame:
    p = pd.read_parquet(CACHE / "forecast_priced.parquet")
    p = p[(p["country"] == "US") & (p["gross_mn"].fillna(0) > 0)].copy()
    p["month"] = pd.to_datetime(p["op_dt"]).dt.to_period("M").dt.to_timestamp()
    p["tok"] = [_us_identity(i)[0] for i in p["isin"]]
    p["kind"] = [_us_identity(i)[1] for i in p["isin"]]
    # Reopenings carry the line's CUSIP, not a term label: name them by residual.
    reopen = (p["kind"] == "") & (p["sec_type"].astype(str) == "Nominal")
    yrs = p["resid_mat_yrs"].astype(float)
    p.loc[reopen, "tok"] = np.select([yrs >= 25, yrs >= 15, yrs >= 8], ["30y", "20y", "10y"], "")[reopen.to_numpy()]
    p.loc[reopen & (p["tok"] != ""), "kind"] = "Nominal"
    p["dv01_per_mn"] = p["dv01_usd_mn"] / p["gross_mn"]
    return p[p["kind"].isin(["Nominal", "FRN"])]


def extra_flow(sc: dict, fwd: pd.DataFrame) -> pd.DataFrame:
    """Extra US private DV01 by issue bucket and month from the scenario's size increases."""
    cols = ["country", "month", "tenor_bucket", "d_dv01", "d_face_mn"]
    x = full_increment_bn(sc["start"], sc["gap_share"], sc.get("gap_bn", GAP_BN))
    if x == 0 or fwd.empty:
        return pd.DataFrame(columns=cols)
    f = fwd.copy()
    months = pd.DatetimeIndex(f["month"])
    f["w"] = phase_weight(months, pd.Timestamp(sc["start"]))
    if sc["mix"] == "prorata":
        f["share"] = f["gross_mn"] / f.groupby("month")["gross_mn"].transform("sum")
    else:
        mix = pd.Series(sc["mix"], dtype=float)
        mix = mix / mix.sum()
        n = f.groupby(["month", "tok"])["gross_mn"].transform("sum")
        f["share"] = f["tok"].map(mix).fillna(0.0) * f["gross_mn"] / n
    f["d_face_mn"] = x * 1000.0 * f["w"] * f["share"]
    f["d_dv01"] = f["d_face_mn"] * f["dv01_per_mn"]
    return f.assign(country="US")[cols]


def scenario_panel() -> tuple[pd.DataFrame, list[dict]]:
    fl = pd.read_parquet(CACHE / "float_share_fc.parquet")
    fl = fl[fl["country"] == "US"].copy()
    fl["month"] = pd.to_datetime(fl["month"])
    stock_keys = fl[["country", "tenor_bucket"]].drop_duplicates()
    base = fl.groupby(["month", "tenor_bucket"], as_index=False).agg(
        flow=("flow_dv01", "sum"), stock=("private_stock_dv01", "sum"), phase=("phase", "first"))
    fwd = _us_forward()
    ten = pd.read_parquet(CACHE / "forecast_tenor.parquet")
    ten["month"] = pd.to_datetime(ten["month"])
    bb = buyback_forward(ten)
    bb = _onto_stock(bb, stock_keys, ["buyback_dv01_usd"]) if not bb.empty else bb
    rows, meta = [], []
    br = bridge_scenario()
    for sc in SCENARIOS + ([br] if br else []):
        add = extra_flow(sc, fwd)
        d = pd.DataFrame(columns=["month", "tenor_bucket", "d_dv01"])
        if not add.empty:
            d = _onto_stock(add, stock_keys, ["d_dv01", "d_face_mn"])[["month", "tenor_bucket", "d_dv01"]]
        g = base.merge(d, on=["month", "tenor_bucket"], how="left")
        g["d_dv01"] = pd.to_numeric(g["d_dv01"], errors="coerce").fillna(0.0)
        g = g.sort_values("month")
        g["d_stock"] = g.groupby("tenor_bucket")["d_dv01"].cumsum()
        g["d_bb"] = 0.0
        if not bb.empty and sc["buyback"] != 1.0:
            b = bb[["month", "tenor_bucket", "buyback_dv01_usd"]]
            g = g.merge(b, on=["month", "tenor_bucket"], how="left")
            g["d_bb"] = (sc["buyback"] - 1.0) * pd.to_numeric(g["buyback_dv01_usd"], errors="coerce").fillna(0.0)
            g = g.drop(columns="buyback_dv01_usd")
        g["flow"] = g["flow"] + g["d_dv01"] + g["d_bb"]
        g["stock"] = g["stock"] + g["d_stock"]
        g["sleeve"] = g["tenor_bucket"].map(SLEEVE_OF)
        g = g[g["sleeve"].notna()]
        phase = g.groupby("month")["phase"].first()
        tot = g.groupby("month")[["flow", "stock"]].sum().assign(sleeve="All")
        parts = [tot] + [g[g["sleeve"] == s].groupby("month")[["flow", "stock"]].sum().assign(sleeve=s)
                         for s in SLEEVES if s != "All"]
        for p in parts:
            idx = pd.date_range(p.index.min(), p.index.max(), freq="MS")
            p = p.reindex(idx)
            flow12 = p["flow"].fillna(0.0).rolling(12, min_periods=12).sum()
            absorb = flow12 / p["stock"].where(p["stock"] > 0)
            rows.append(pd.DataFrame({"scenario": sc["key"], "sleeve": p["sleeve"].ffill().bfill().to_numpy(),
                                      "month": idx, "absorb": absorb.to_numpy(),
                                      "phase": phase.reindex(idx).fillna("history").to_numpy()}))
        long_b = [b for b, s in SLEEVE_OF.items() if s == "Long"]
        horizon = add[add["month"] <= fl["month"].max()] if not add.empty else add
        meta.append({
            "key": sc["key"], "name": sc["name"], "start": sc["start"] or "",
            "gap_share": sc["gap_share"], "buyback": sc["buyback"], "why": sc["why"],
            "mix": "pro rata to current sizes" if sc["mix"] == "prorata" else
                   (", ".join(f"{k} {v}" for k, v in sc["mix"].items()) if sc["mix"] else "none"),
            "gap_bn": round(sc.get("gap_bn", GAP_BN), 1),
            "gap_source": "fiscal bridge" if sc["key"] == "bridge" else "dealer survey (TBAC minutes, 5 Aug 2026)",
            "full_bn_month": round(full_increment_bn(sc["start"], sc["gap_share"], sc.get("gap_bn", GAP_BN)), 1),
            "horizon_face_bn": round(float(horizon["d_face_mn"].sum()) / 1000.0, 1) if not horizon.empty else 0.0,
            "horizon_long_dv01": round(float(horizon.loc[horizon["tenor_bucket"].isin(long_b), "d_dv01"].sum()), 1)
                                 if not horizon.empty else 0.0,
        })
    out = pd.concat(rows, ignore_index=True)
    return out[(out["month"] >= "2005-01-01") & out["absorb"].notna()], meta


def write_scenarios() -> pd.DataFrame:
    out, meta = scenario_panel()
    out.to_parquet(CACHE / "rv_scenarios.parquet", index=False)
    (CACHE / "rv_scenarios_meta.json").write_text(json.dumps(meta, indent=1))
    end = out[out["sleeve"] == "Long"].sort_values("month").groupby("scenario").tail(1)
    print("  scenarios US Long absorb at horizon", end.set_index("scenario")["absorb"].round(4).to_dict())
    return out
