"""One pricing convention for every DV01 the float metric divides: coupon at the market yield.

Flow (new issues), stock (everything outstanding) and official holdings are all priced as a
semi-annual bond at its own coupon and the month's market yield for its country, tenor and
type. Before this, flow was a zero-coupon at 3%, stock was par at its own coupon and official
holdings were par at 3%, so the ratio between them moved with the convention, not the market.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from issuance.tenor import BUCKET_MID, bucket_from_years

_FRAC = {"¼": 0.25, "½": 0.5, "¾": 0.75, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875}
_CPN = re.compile(r"^\s*(\d+)?\s*([¼½¾⅛⅜⅝⅞])?\s*(?:(\d+)\s*/\s*(\d+))?\s*%")


def coupon_from_name(name: object) -> float:
    """'4¼% Treasury Gilt 2049' → 4.25; '0 1/8% Index-linked …' → 0.125."""
    m = _CPN.match(str(name or ""))
    if not m or not any(m.groups()):
        return float("nan")
    whole, frac, num, den = m.groups()
    out = float(whole or 0)
    if frac:
        out += _FRAC[frac]
    if num and den and float(den) > 0:
        out += float(num) / float(den)
    return out


def clean_ops(ops: pd.DataFrame) -> pd.DataFrame:
    """Coupons that are not coupons are dropped, gilt coupons come from the name, and a missing
    yield takes the same line's nearest print."""
    if ops is None or ops.empty or "coupon" not in ops.columns:
        return ops
    out = ops.copy()
    cpn = pd.to_numeric(out["coupon"], errors="coerce").astype(float)
    cpn = cpn.where((cpn >= 0) & (cpn <= 20))
    if "sec_desc_long" in out.columns:
        missing = cpn.isna() & out["country"].astype(str).eq("UK")
        if missing.any():
            cpn.loc[missing] = out.loc[missing, "sec_desc_long"].map(coupon_from_name)
    out["coupon"] = cpn
    if "yield" in out.columns and "isin" in out.columns:
        y = pd.to_numeric(out["yield"], errors="coerce").astype(float)
        out["yield"] = y.where((y > -3) & (y < 25))
        out = _fill_yield(out)
    return out


FILL_DAYS = 92


def _fill_yield(out: pd.DataFrame) -> pd.DataFrame:
    """A missing yield takes the same line's print within FILL_DAYS, else the month's market curve.
    An old line's issue yield is not today's market (JGB liquidity-enhancement taps of 2019 lines
    would otherwise price at 0.3%)."""
    miss = out["yield"].isna() & out["op_dt"].notna() & out["isin"].notna()
    if not miss.any():
        return out
    have = out.loc[out["yield"].notna() & out["op_dt"].notna() & out["isin"].notna(), ["isin", "op_dt", "yield"]].copy()
    have["isin"] = have["isin"].astype(str)
    want = out.loc[miss, ["isin", "op_dt"]].copy()
    want["isin"] = want["isin"].astype(str)
    want["_i"] = want.index
    got = pd.merge_asof(want.sort_values("op_dt"), have.sort_values("op_dt"), on="op_dt", by="isin",
                        direction="nearest", tolerance=pd.Timedelta(days=FILL_DAYS))
    out.loc[got["_i"].to_numpy(), "yield"] = got["yield"].to_numpy()
    miss = out["yield"].isna() & out["op_dt"].notna() & out["maturity_dt"].notna()
    if miss.any() and "sec_type" in out.columns:
        curve = market_curve(out)
        sub = out.loc[miss]
        years = (pd.to_datetime(sub["maturity_dt"]) - pd.to_datetime(sub["op_dt"])).dt.days / 365.25
        for c, idx in sub.groupby(sub["country"].astype(str)).groups.items():
            if not (curve["country"] == c).any():
                continue
            out.loc[idx, "yield"] = curve_yield(curve, c, sub.loc[idx, "sec_type"].astype(str).eq("Linker").to_numpy(),
                                                sub.loc[idx, "op_dt"].to_numpy(), years.loc[idx].to_numpy())
    return out


def price(years, coupon, yld) -> np.ndarray:
    """Dirty price per 100 at a coupon date, semi-annual."""
    years = np.clip(np.asarray(years, dtype=float), 0.0, 60.0)
    c = np.asarray(coupon, dtype=float) / 100.0
    y = np.asarray(yld, dtype=float) / 200.0
    n = np.maximum(np.round(years * 2.0), 1.0)
    small = np.abs(y) < 1e-9
    ys = np.where(small, 1.0, y)
    v = (1.0 + ys) ** -n
    annuity = np.where(small, n, (1.0 - v) / ys)
    return np.where(small, 100.0 + 100.0 * c * n / 2.0, 100.0 * c / 2.0 * annuity + 100.0 * v)


def dv01_mn(face, years, coupon, yld, fx=1.0) -> np.ndarray:
    """USD million per bp for `face` million at the coupon and yield."""
    face = np.asarray(face, dtype=float)
    years = np.asarray(years, dtype=float)
    yld = np.asarray(yld, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    coupon = np.where(np.isfinite(coupon), coupon, yld)
    bump = (price(years, coupon, yld - 0.01) - price(years, coupon, yld + 0.01)) / 2.0
    return face * bump / 100.0 * np.asarray(fx, dtype=float)


def market_curve(ops: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Median auction yield by country, type (nominal/linker), month and tenor bucket.

    Months with no auction in a bucket carry the last print; a bucket that has never printed
    in a month is interpolated across the buckets that did. Linkers fall back to nominal − 2.
    """
    g = ops[(~ops["sec_type"].astype(str).isin(["Bill", "Foreign", "FRN"])) & ops["yield"].notna()
            & ops["maturity_dt"].notna() & (ops["amount_issued"] > 0)].copy()
    if "op_type" in g.columns:
        g = g[g["op_type"].astype(str).ne("Liquidity Enhancement")]
    g["linker"] = g["sec_type"].astype(str).eq("Linker")
    g["month"] = pd.to_datetime(g["op_dt"]).dt.to_period("M").dt.to_timestamp()
    years = (pd.to_datetime(g["maturity_dt"]) - pd.to_datetime(g["op_dt"])).dt.days / 365.25
    g["tenor_bucket"] = [bucket_from_years(v) for v in years]
    med = g.groupby(["country", "linker", "month", "tenor_bucket"])["yield"].median()
    buckets = [b for b in BUCKET_MID if b != "unknown"]
    mids = np.array([BUCKET_MID[b] for b in buckets], dtype=float)
    rows = []
    for (country, linker), part in med.groupby(level=["country", "linker"]):
        wide = part.droplevel(["country", "linker"]).unstack("tenor_bucket").reindex(columns=buckets)
        stop = max(wide.index.max(), end) if end is not None else wide.index.max()
        wide = wide.reindex(pd.date_range(wide.index.min(), stop, freq="MS")).ffill(limit=12)
        vals = np.array(wide.to_numpy(dtype=float), copy=True)
        for i in range(len(vals)):
            ok = np.isfinite(vals[i])
            if ok.sum() >= 2:
                vals[i] = np.interp(mids, mids[ok], vals[i][ok])
            elif ok.sum() == 1:
                vals[i] = vals[i][ok][0]
        wide = pd.DataFrame(vals, index=wide.index, columns=buckets).ffill().bfill()
        long = wide.stack().rename("yld").reset_index()
        long.columns = ["month", "tenor_bucket", "yld"]
        long["country"], long["linker"] = country, bool(linker)
        rows.append(long)
    curve = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["month", "tenor_bucket", "yld", "country", "linker"])
    have = set(zip(curve["country"], curve["linker"]))
    extra = []
    for country in curve["country"].unique():
        if (country, True) not in have:
            nom = curve[(curve["country"] == country) & ~curve["linker"]].copy()
            nom["linker"], nom["yld"] = True, nom["yld"] - 2.0
            extra.append(nom)
    if extra:
        curve = pd.concat([curve, *extra], ignore_index=True)
    return curve


def curve_yield(curve: pd.DataFrame, country: str, linker, month, years) -> np.ndarray:
    """Market yield for each (linker, month, years) of one country; last month carried forward."""
    linker = np.broadcast_to(np.asarray(linker, dtype=bool), np.shape(years))
    month = pd.to_datetime(pd.Series(np.broadcast_to(np.asarray(month, dtype="datetime64[ns]"), np.shape(years))))
    month = month.dt.to_period("M").dt.to_timestamp()
    bucket = [bucket_from_years(v) for v in np.asarray(years, dtype=float)]
    c = curve[curve["country"] == country]
    if c.empty:
        return np.full(len(bucket), 3.0)
    last = c["month"].max()
    first = c["month"].min()
    key = pd.DataFrame({"linker": linker, "month": month.clip(first, last).to_numpy(), "tenor_bucket": bucket})
    got = key.merge(c[["linker", "month", "tenor_bucket", "yld"]], on=["linker", "month", "tenor_bucket"], how="left")
    return got["yld"].fillna(3.0).to_numpy(dtype=float)
