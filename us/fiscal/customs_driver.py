"""Customs duties = effective tariff rate × goods imports.

``dep_customs`` in the flow model is otherwise trailing-mean × calendar shape.
That level lags every tariff step by a year. Here the level is the identity
CBO uses: trailing effective rate × import path. Imports forward are a constant
share of nominal GDP (last GDP print held flat).

Rate and share at an origin use only months whose import print has already
landed (BOPGIMP lags ~6 weeks). Calendar shape still multiplies the daily
level so Periodic Monthly Statement spikes stay in the path.

  PYTHONPATH=. .venv/bin/python -c "from customs_driver import snapshot; print(snapshot())"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

FRED_GOODS_IMPORTS = "BOPGIMP"  # monthly, $mn, Census/BOP goods imports
FRED_NOMINAL_GDP = "GDP"  # quarterly SAAR, $bn
# Trailing 3 months: reacts to tariff steps; 12m lagged the 2025 ramp behind the
# 250-day mean. PRE 2022–23 and the step window both prefer 3m over 6m/12m.
RATE_MONTHS = 3
SHARE_MONTHS = 3


def _monthly_customs(actual_mn: pd.Series) -> pd.Series:
    s = actual_mn.dropna().sort_index()
    return s.resample("ME").sum().rename("customs_mn")


def _monthly_imports(imports_mn: pd.Series) -> pd.Series:
    """Stamp each BOPGIMP observation at month-end so it aligns with DTS months."""
    s = imports_mn.dropna().sort_index()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s[~s.index.duplicated(keep="last")].rename("imports_mn")


def last_complete_import_month(imports_mn: pd.Series, origin: pd.Timestamp) -> pd.Timestamp | None:
    """Latest month-end whose import print is known by ``origin`` (series already published)."""
    m = _monthly_imports(imports_mn)
    m = m[m.index <= pd.Timestamp(origin)]
    return None if m.empty else pd.Timestamp(m.index.max())


def effective_rate(
    customs_mn: pd.Series,
    imports_mn: pd.Series,
    origin: pd.Timestamp,
    *,
    months: int = RATE_MONTHS,
) -> float:
    """Trailing ``months`` of customs / imports ending at the last complete import month ≤ origin."""
    end = last_complete_import_month(imports_mn, origin)
    if end is None:
        return float("nan")
    c = _monthly_customs(customs_mn)
    i = _monthly_imports(imports_mn)
    both = pd.concat([c, i], axis=1, sort=True).dropna()
    both = both[both.index <= end].tail(months)
    if len(both) < max(3, months // 2) or both["imports_mn"].sum() <= 0:
        return float("nan")
    return float(both["customs_mn"].sum() / both["imports_mn"].sum())


def import_gdp_share(
    imports_mn: pd.Series,
    gdp_saar_bn: pd.Series,
    origin: pd.Timestamp,
    *,
    months: int = SHARE_MONTHS,
) -> float:
    """Trailing mean of (12 × monthly imports / GDP SAAR), last complete import month ≤ origin."""
    from tga_deficit import interpolate_to_daily

    end = last_complete_import_month(imports_mn, origin)
    if end is None:
        return float("nan")
    i = _monthly_imports(imports_mn)
    i = i[i.index <= end].tail(months)
    if i.empty:
        return float("nan")
    idx = pd.date_range(i.index.min() - pd.Timedelta(days=100), i.index.max(), freq="D")
    gdp = interpolate_to_daily(gdp_saar_bn.dropna(), idx)
    saar = (i * 12.0 / 1000.0)  # $bn SAAR
    g = gdp.reindex(i.index).ffill()
    share = (saar / g).replace([np.inf, -np.inf], np.nan).dropna()
    if share.empty:
        return float("nan")
    return float(share.mean())


# DTS customs prints on Fed business days only; the panel mean is a business-day
# mean. Spreading the annual import SAAR over 365 calendar days understates the
# level by ~365/252.
BUSINESS_DAYS_PER_YEAR = 252


def imports_daily_mn(
    share: float,
    gdp_saar_bn: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Business-day goods-import pace ($mn) = share × GDP SAAR / 252."""
    from tga_deficit import interpolate_to_daily

    if not np.isfinite(share) or share <= 0:
        return pd.Series(np.nan, index=dates)
    gdp = interpolate_to_daily(gdp_saar_bn.dropna(), pd.DatetimeIndex(dates))
    return (share * gdp / BUSINESS_DAYS_PER_YEAR * 1000.0).rename("imports_daily_mn")


def load_drivers(*, refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    from tga_deficit import fetch_fred

    imports = fetch_fred(FRED_GOODS_IMPORTS, refresh=refresh)
    gdp = fetch_fred(FRED_NOMINAL_GDP, refresh=refresh)
    return imports, gdp


def customs_forecast(
    actual_mn: pd.Series,
    origin: pd.Timestamp,
    dates: pd.DatetimeIndex,
    *,
    shape: np.ndarray | None = None,
    imports_mn: pd.Series | None = None,
    gdp_saar_bn: pd.Series | None = None,
) -> np.ndarray:
    """Daily customs forecast ($mn) after ``origin``.

    Level = effective rate × import pace; ``shape`` (mean ≈ 1) scales the calendar.
    Missing drivers fall back to the trailing 250-business-day mean.
    """
    if imports_mn is None or gdp_saar_bn is None:
        imports_mn, gdp_saar_bn = load_drivers()
    origin = pd.Timestamp(origin)
    rate = effective_rate(actual_mn, imports_mn, origin)
    share = import_gdp_share(imports_mn, gdp_saar_bn, origin)
    hist = actual_mn[actual_mn.index <= origin].dropna()
    fallback = float(hist.tail(250).mean()) if len(hist) else 0.0
    if not np.isfinite(rate) or not np.isfinite(share) or rate < 0:
        level = np.full(len(dates), fallback)
    else:
        imp = imports_daily_mn(share, gdp_saar_bn, pd.DatetimeIndex(dates))
        level = (rate * imp).to_numpy(dtype=float)
        level = np.where(np.isfinite(level), level, fallback)
    if shape is None:
        return level
    return level * np.asarray(shape, dtype=float)


def snapshot(origin: pd.Timestamp | None = None) -> dict:
    """Live rate / share / implied daily level for the desk."""
    panel = pd.read_parquet(DERIVED / "dts_bucket_daily.parquet")["dep_customs"]
    origin = pd.Timestamp(origin) if origin is not None else panel.dropna().index.max()
    imports, gdp = load_drivers()
    rate = effective_rate(panel, imports, origin)
    share = import_gdp_share(imports, gdp, origin)
    end = last_complete_import_month(imports, origin)
    daily = float("nan") if not np.isfinite(rate) or not np.isfinite(share) else float(
        rate * share * float(gdp.dropna().iloc[-1]) / BUSINESS_DAYS_PER_YEAR * 1000.0
    )
    return {
        "origin": str(pd.Timestamp(origin).date()),
        "imports_through": None if end is None else str(end.date()),
        "effective_rate": rate,
        "import_gdp_share": share,
        "implied_daily_mn": daily,
        "trailing_mean_mn": float(panel[panel.index <= origin].tail(250).mean()),
    }
