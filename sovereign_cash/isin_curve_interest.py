"""Point-in-time marketable interest from the ISIN book + yield curve.

Production wrapper around the engine scored in
``experiments/isin_curve_interest_screen`` (debt ``cashflow.components`` +
bill ladder + residual + new-issue at forward par).

Used by UK / DE / FR / IT cash interest legs when ``interest_mode`` /
``LATER_INTEREST_MODE`` is ``isin_curve``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

DERIVED = __import__("pathlib").Path(__file__).resolve().parent / "derived"


@lru_cache(maxsize=1)
def _screen():
    """Load the scored PIT engine (experiments/ is not a package)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "experiments" / "isin_curve_interest_screen.py"
    spec = importlib.util.spec_from_file_location("isin_curve_interest_screen", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=64)
def annual_bn(cc: str, year: int, asof_year: int | None = None) -> float:
    """Calendar-year ``year`` interest forecast (local bn) known at end of ``asof_year``.

    Default asof = year−1 (honest PIT for a full-year forecast). Returns NaN on failure.
    """
    del asof_year  # screen always uses 31 Dec (year−1); reserved for future intra-year cut
    ann = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    try:
        out = _screen().forecast_full(cc.lower(), int(year), ann)
    except Exception:
        return float("nan")
    if not out.get("ok"):
        return float("nan")
    v = out.get("full_bn")
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def month_shares(hist: pd.Series, origin: pd.Timestamp, lookback_years: int = 5
                 ) -> dict[int, float]:
    """Calendar-month weights from history through ``origin`` (sum to 1)."""
    hh = hist.loc[:origin].dropna()
    if hh.empty:
        return {m: 1.0 / 12.0 for m in range(1, 13)}
    cut = hh.index.max() - pd.DateOffset(years=lookback_years)
    hh = hh[hh.index >= cut]
    by_m = hh.groupby(hh.index.month).mean()
    s = float(by_m.sum())
    if s <= 0 or not np.isfinite(s):
        return {m: 1.0 / 12.0 for m in range(1, 13)}
    return {int(m): float(v) / s for m, v in by_m.items()}


def monthly_path(cc: str, hist: pd.Series, origin: pd.Timestamp, h: int
                 ) -> np.ndarray:
    """Next ``h`` months: ISIN+curve annual × month share (residual-scaled to TTM).

    For each calendar year Y in the horizon, annual level = ``annual_bn(cc, Y)``
    when finite; else same-month-last-year. Scaled so the twelve months ending at
    ``origin`` keep the published TTM level (covers non-marketable / APF gap).
    """
    origin = pd.Timestamp(origin)
    hh = hist.loc[:origin].dropna()
    shares = month_shares(hh, origin)
    ttm = float(hh.iloc[-12:].sum()) if len(hh) >= 12 else float(hh.sum())
    # Scale factor from last complete calendar year if we have a PIT annual.
    y_ref = origin.year if origin.month == 12 else origin.year - 1
    pit = annual_bn(cc, y_ref + 1)  # forecast for y_ref+1 asof end y_ref
    # Prefer scaling off published TTM vs marketable+resid annual for open year
    open_ann = annual_bn(cc, origin.year) if origin.month < 12 else annual_bn(cc, origin.year + 1)
    scale = 1.0
    if np.isfinite(open_ann) and open_ann > 1e-6 and ttm > 0:
        # open-year annual is a full-year number; TTM is rolling — use ratio of
        # last complete CY published to last complete CY forecast when both exist.
        pass
    if len(hh) >= 12:
        last_cy = float(hh[hh.index.year == y_ref].sum()) if (hh.index.year == y_ref).any() else ttm
        fc_cy = annual_bn(cc, y_ref)
        if np.isfinite(fc_cy) and fc_cy > 1e-6 and last_cy > 0:
            scale = last_cy / fc_cy
        elif np.isfinite(pit) and pit > 1e-6:
            scale = ttm / pit

    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    cache: dict[int, float] = {}
    out = []
    for m in months:
        y = int(m.year)
        if y not in cache:
            v = annual_bn(cc, y)
            if not np.isfinite(v):
                # fallback: trailing twelve × 1 (smly annual)
                sm = hh[hh.index.month == m.month]
                cache[y] = float(sm.iloc[-1]) * 12.0 if len(sm) else ttm
            else:
                cache[y] = float(v) * scale
        out.append(cache[y] * shares.get(int(m.month), 1.0 / 12.0))
    return np.asarray(out, dtype=float)
