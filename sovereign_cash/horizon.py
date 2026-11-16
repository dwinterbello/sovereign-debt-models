"""Coupon-programme horizon: past the last gross-issuance month the financing
identity is not a forecast (gross=0, redemptions continue, buffer dumps).

Cash requirement itself may still be soft-pinned / plan-shaped; only the
bond / bill / buffer path is truncated for display.
"""
from __future__ import annotations

import pandas as pd

GROSS_FLOOR = 0.5  # local bn; ignore noise zeros


def programme_end(df: pd.DataFrame, gross_col: str = "bond_gross_bn") -> pd.Timestamp | None:
    if df is None or df.empty or gross_col not in df.columns or "is_forecast" not in df.columns:
        return None
    d = df.copy()
    d["month"] = pd.to_datetime(d["month"])
    g = d.loc[d["is_forecast"].astype(bool) & (d[gross_col].fillna(0) > GROSS_FLOOR), "month"]
    return pd.Timestamp(g.max()) if len(g) else None


def clip_frame(df: pd.DataFrame, gross_col: str = "bond_gross_bn") -> pd.DataFrame:
    """Drop rows after the last gross-issuance month. No-op if no programme end."""
    end = programme_end(df, gross_col=gross_col)
    if end is None:
        return df
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"])
    return out.loc[out["month"] <= end].reset_index(drop=True)


def clip_series(s: pd.Series, end: pd.Timestamp | None) -> pd.Series:
    if end is None or s is None or len(s) == 0:
        return s
    idx = pd.to_datetime(s.index)
    return s.loc[idx <= end]
