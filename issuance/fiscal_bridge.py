"""US forward bills, Fed bill take and buybacks from the fiscal supply bridge.

`us/fiscal/supply_bridge.py` carries bills the fiscal path needs (steered to the
refunding TGA targets), the Fed's reserve-management purchase pace and buybacks.
The issuance forward used to copy each of these from the same month a year earlier,
chaining past twelve months, and set forward buybacks to zero. Here:

  bill gross (DV01, face, duration) = year-ago value × public-issued bill stock ratio
      (gross bill issuance rolls the public-issued stock; at a constant term mix its
      monthly DV01 is ≈ stock / 12 × 1bp, so it scales with the stock, not with net bills)
  official bill DV01 = trailing 3-month mean (the purchase pace is held constant)
  buyback DV01 = each tenor's last-quarter monthly average (`float_rv.buyback_forward`,
      Treasury's own refunding assumption)

Without the bridge files the ratios are 1 and the year-ago copy stands.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FISCAL = ROOT / "us" / "fiscal" / "derived"
BRIDGE = FISCAL / "supply_bridge.parquet"
TIDY = FISCAL / "auctions_tidy.parquet"
OFFICIAL_BILL_MONTHS = 3
BILL_CLS = ("bill", "cmb")


def public_bill_stock_history(months: pd.DatetimeIndex, tidy: pd.DataFrame | None = None) -> pd.Series:
    """Public-issued bill par outstanding at each month end ($bn), from auction results."""
    t = pd.read_parquet(TIDY) if tidy is None else tidy
    b = t[t["cls"].isin(BILL_CLS) & (t["status"] == "awarded")]
    par = pd.to_numeric(b["public_par_bn"], errors="coerce").fillna(0.0).to_numpy()
    iss, mat = b["issue_date"].to_numpy(), b["maturity_date"].to_numpy()
    out = {}
    for m in months:
        e = (pd.Timestamp(m) + pd.offsets.MonthEnd(0)).to_datetime64()
        out[pd.Timestamp(m)] = float(par[(iss <= e) & (mat > e)].sum())
    return pd.Series(out, name="public_bill_stock_bn")


def bill_stock_ratio(fc_months: pd.DatetimeIndex) -> dict[pd.Timestamp, float]:
    """S(t) / S(t−12) per forecast month: bridge stock forward, auction-result stock behind."""
    if not BRIDGE.exists() or not TIDY.exists() or len(fc_months) == 0:
        return {}
    br = pd.read_parquet(BRIDGE).loc["unchanged"]["public_bill_stock_bn"]
    br.index = pd.to_datetime(br.index)
    first = br.index.min()
    lags = pd.DatetimeIndex(sorted({m - pd.DateOffset(months=12) for m in fc_months}))
    hist = public_bill_stock_history(lags[lags < first])
    # Both legs are tidy public par (the bridge starts from the same auction-result stock).
    stock = pd.concat([hist, br]).sort_index()
    stock = stock[~stock.index.duplicated(keep="last")]
    out = {}
    for m in fc_months:
        a, b = stock.get(pd.Timestamp(m)), stock.get(pd.Timestamp(m) - pd.DateOffset(months=12))
        if a is not None and b is not None and b > 0 and pd.notna(a):
            out[pd.Timestamp(m)] = float(a / b)
    return out


def scaled_last_year(hist: pd.DataFrame, extra: pd.DataFrame, col: str, ratio: dict) -> pd.Series:
    """Year-ago value × stock ratio, chaining through forecast months past a year out (US only)."""
    known = {(c, pd.Timestamp(m)): v for c, m, v in zip(hist["country"], hist["month"], hist[col]) if pd.notna(v)}
    out = {}
    for i, (c, m) in extra[["country", "month"]].sort_values("month").iterrows():
        m = pd.Timestamp(m)
        v = known.get((c, m - pd.DateOffset(months=12)))
        if v is None:
            continue
        v = float(v) * (ratio.get(m, 1.0) if c == "US" else 1.0)
        known[(c, m)] = v
        out[i] = v
    return pd.Series(out, dtype=float)


def official_bill_forward(hist: pd.DataFrame, extra: pd.DataFrame) -> pd.Series:
    """US: trailing-quarter mean of official bill DV01 for every forecast month."""
    h = hist[hist["country"] == "US"].sort_values("month")["official_bill_dv01_usd"].dropna()
    v = float(h.tail(OFFICIAL_BILL_MONTHS).mean()) if len(h) else 0.0
    us = extra["country"] == "US"
    return pd.Series(v, index=extra.index[us])


def buyback_monthly(fc_tenor: pd.DataFrame) -> pd.DataFrame:
    from issuance.float_rv import buyback_forward

    bb = buyback_forward(fc_tenor)
    if bb.empty:
        return pd.DataFrame(columns=["country", "month", "buyback_dv01_usd"])
    bb["month"] = pd.to_datetime(bb["month"])
    return bb.groupby(["country", "month"], as_index=False)["buyback_dv01_usd"].sum()


def apply_us(supply: pd.DataFrame, extra: pd.DataFrame, fc_tenor: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Overwrite US bill / official bill / buyback forward columns on `extra` (forecast months)."""
    extra = extra.copy()
    us = extra["country"] == "US"
    fc_m = pd.DatetimeIndex(sorted(pd.to_datetime(extra.loc[us, "month"]).unique()))
    ratio = bill_stock_ratio(fc_m)
    if us.any():
        s = scaled_last_year(supply, extra[us], "bill_dv01_usd", ratio)
        extra.loc[s.index, "bill_dv01_usd"] = s
        o = official_bill_forward(supply, extra)
        extra.loc[o.index, "official_bill_dv01_usd"] = o
    bb = buyback_monthly(fc_tenor)
    extra = extra.drop(columns=["buyback_dv01_usd"], errors="ignore").merge(bb, on=["country", "month"], how="left")
    extra["buyback_dv01_usd"] = extra["buyback_dv01_usd"].fillna(0.0)
    info = {"bill_stock_ratio": {str(k.date()): round(v, 4) for k, v in ratio.items()},
            "official_bill_months": OFFICIAL_BILL_MONTHS, "bridge": BRIDGE.exists()}
    return extra, info
