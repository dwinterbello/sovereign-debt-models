"""Convert face issuance into USD DV01 using residual maturity, coupon and yield."""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.fetch_fx import fetch_fx, fx_on
from issuance.paths import CACHE
from issuance.tenor import BUCKET_MID

FREQ = 2  # semi-annual, gilt / UST / bund convention for duration


def macaulay_duration(years: float, coupon: float, yld: float, freq: int = FREQ) -> float:
    """Macaulay duration in years for a fixed coupon bond (face 100)."""
    if years is None or not np.isfinite(years) or years <= 0:
        return 0.0
    if years < 1.0 and (coupon is None or not np.isfinite(coupon) or coupon == 0):
        return float(years)
    times, pv = _cash_flows(years, coupon, yld, freq)
    price = float(pv.sum())
    if price == 0:
        return float(years)
    return float(np.sum(times * pv) / price)


def _cash_flows(years: float, coupon: float, yld: float, freq: int = FREQ) -> tuple[np.ndarray, np.ndarray]:
    """Coupon dates counted back from maturity, so the first period is the stub, and their PVs."""
    y = 0.03 if yld is None or not np.isfinite(yld) else float(yld) / 100.0
    c = 0.0 if coupon is None or not np.isfinite(coupon) else float(coupon) / 100.0
    n = max(1, int(np.ceil(years * freq - 1e-9)))
    times = years - (n - np.arange(1, n + 1)) / freq
    cfs = np.full(n, 100.0 * c / freq)
    cfs[-1] += 100.0
    return times, cfs / (1.0 + y / freq) ** (freq * times)


def modified_duration(years: float, coupon: float, yld: float, freq: int = FREQ) -> float:
    mac = macaulay_duration(years, coupon, yld, freq)
    y = 0.03 if yld is None or not np.isfinite(yld) else float(yld) / 100.0
    return mac / (1.0 + y / freq)


def dirty_price(years: float, coupon: float, yld: float, freq: int = FREQ) -> float:
    if years is None or not np.isfinite(years) or years <= 0:
        return 100.0
    return float(_cash_flows(years, coupon, yld, freq)[1].sum())


def dv01_local_mn(amount_mn: float, years: float, coupon: float, yld: float, price: float | None) -> float:
    """Local-currency millions per basis point for `amount_mn` face."""
    if amount_mn is None or not np.isfinite(amount_mn) or amount_mn == 0:
        return 0.0
    px = price if price is not None and np.isfinite(price) and price > 10 else dirty_price(years, coupon, yld)
    dmod = modified_duration(years, coupon, yld)
    return float(amount_mn) * (px / 100.0) * dmod / 10_000.0


def attach_dv01(flow: pd.DataFrame, fx: pd.DataFrame | None = None) -> pd.DataFrame:
    if fx is None:
        fx = fetch_fx(force=False)
    out = flow.reset_index(drop=True)
    if out.empty:
        out["dv01_local_mn"] = pd.Series(dtype=float)
        out["dv01_usd_mn"] = pd.Series(dtype=float)
        return out
    years = out.get("resid_mat_yrs")
    if years is None or years.isna().all():
        years = out["tenor_bucket"].map(BUCKET_MID)
    out["_years"] = pd.to_numeric(years, errors="coerce").fillna(out["tenor_bucket"].map(BUCKET_MID))
    if "sec_type" in out.columns:
        frn = out["sec_type"].astype(str).str.upper().eq("FRN")
        out.loc[frn, "_years"] = 0.25
    coupon = pd.to_numeric(out["coupon"], errors="coerce") if "coupon" in out.columns else pd.Series(np.nan, index=out.index)
    # TIPS / ILB auction yields are real. A missing yield is a par real bond
    # at the real coupon, not a 3% nominal assumption.
    if "sec_type" in out.columns:
        linker = out["sec_type"].astype(str).eq("Linker")
    else:
        linker = pd.Series(False, index=out.index)
    y_col = "yield_avg" if "yield_avg" in out.columns else ("yield" if "yield" in out.columns else None)
    yld = pd.to_numeric(out[y_col], errors="coerce") if y_col else pd.Series(np.nan, index=out.index)
    yld = yld.where(~(linker & yld.isna() & coupon.notna()), coupon)
    # A missing coupon is a par bond at the yield, not a zero-coupon.
    coupon = coupon.where(coupon.notna(), yld)
    p_col = "price_avg" if "price_avg" in out.columns else ("price" if "price" in out.columns else None)
    price = pd.to_numeric(out[p_col], errors="coerce") if p_col else pd.Series(np.nan, index=out.index)
    a_col = "gross_mn" if "gross_mn" in out.columns else "amount_issued"
    amount = pd.to_numeric(out[a_col], errors="coerce")
    date_col = "month" if "month" in out.columns else "op_dt"
    dates = pd.to_datetime(out[date_col], errors="coerce")
    ccy = out["ccy"] if "ccy" in out.columns else pd.Series("USD", index=out.index)
    fx_panel = fx.copy()
    fx_panel["date"] = pd.to_datetime(fx_panel["date"])
    fx_panel = fx_panel.sort_values("date")
    spots = []
    for code, idx in ccy.groupby(ccy).groups.items():
        col = {"GBP": "GBPUSD", "EUR": "EURUSD", "USD": "USDUSD", "JPY": "JPYUSD",
               "SEK": "SEKUSD"}.get(str(code))
        fallback = {"GBP": 1.27, "EUR": 1.08, "USD": 1.0, "JPY": 0.0067, "SEK": 0.095}.get(str(code), 1.0)
        if col is None or col not in fx_panel.columns:
            spots.append(pd.Series(fallback, index=idx))
            continue
        left = pd.DataFrame({"date": dates.loc[idx], "_i": list(idx)}).sort_values("date")
        merged = pd.merge_asof(left, fx_panel[["date", col]].dropna(), on="date", direction="backward")
        merged[col] = merged[col].fillna(fallback)
        spots.append(merged.set_index("_i")[col])
    spot = pd.concat(spots).reindex(out.index).fillna(1.0)
    local = np.array([
        dv01_local_mn(
            0.0 if pd.isna(amount.iat[i]) else float(amount.iat[i]),
            0.0 if pd.isna(out["_years"].iat[i]) else float(out["_years"].iat[i]),
            0.0 if pd.isna(coupon.iat[i]) else float(coupon.iat[i]),
            3.0 if pd.isna(yld.iat[i]) else float(yld.iat[i]),
            None if pd.isna(price.iat[i]) else float(price.iat[i]),
        )
        for i in range(len(out))
    ])
    from issuance.index_ratio import ratios_for

    local = local * ratios_for(out, "op_dt" if "op_dt" in out.columns else date_col)
    out["dv01_local_mn"] = local
    out["dv01_usd_mn"] = local * spot.to_numpy()
    out.drop(columns=["_years"], inplace=True)
    return out


def write_dv01(isin_flow: pd.DataFrame, tenor_flow: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fx = fetch_fx(force=False)
    isin = attach_dv01(isin_flow, fx)
    bb_rows = isin_flow["buyback_mn"].fillna(0) != 0
    isin["buyback_dv01_usd"] = 0.0
    if bb_rows.any():
        bb = attach_dv01(isin_flow[bb_rows].assign(gross_mn=isin_flow.loc[bb_rows, "buyback_mn"]), fx)
        isin.loc[bb_rows.to_numpy(), "buyback_dv01_usd"] = bb["dv01_usd_mn"].to_numpy()
    tenor = attach_dv01(tenor_flow, fx)
    # Tenor rows carry no coupon or yield. Their DV01 is the sum of the priced ISIN rows.
    keys = ["country", "month", "sec_type", "tenor_bucket"]
    priced = isin.assign(tenor_bucket=isin["tenor_bucket"].astype(str)).groupby(keys, dropna=False)[
        ["dv01_local_mn", "dv01_usd_mn", "buyback_dv01_usd"]].sum()
    tenor["buyback_dv01_usd"] = 0.0
    t = tenor.assign(_b=tenor["tenor_bucket"].astype(str))
    hit = t.set_index(["country", "month", "sec_type", "_b"]).index.map(
        lambda k: priced.index.get_loc(k) if k in priced.index else -1)
    hit = np.asarray(hit)
    ok = hit >= 0
    for col in ("dv01_local_mn", "dv01_usd_mn", "buyback_dv01_usd"):
        vals = tenor[col].to_numpy(dtype=float).copy()
        vals[ok] = priced[col].to_numpy()[hit[ok]]
        tenor[col] = vals
    isin.to_parquet(CACHE / "dv01_isin.parquet", index=False)
    tenor.to_parquet(CACHE / "dv01_tenor.parquet", index=False)
    return isin, tenor
