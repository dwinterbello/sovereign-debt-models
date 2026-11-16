"""Harmonized residual-maturity buckets used on the dashboard and in the QE allocator."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Residual years at the operation date. Bills sit in 0-1. Linkers keep the same
# residual buckets; sec_type splits them later.
BUCKETS = (
    ("0-1y", 0.0, 1.0),
    ("1-3y", 1.0, 3.0),
    ("3-7y", 3.0, 7.0),
    ("7-10y", 7.0, 10.0),
    ("10-15y", 10.0, 15.0),
    ("15-20y", 15.0, 20.0),
    ("20-30y", 20.0, 30.0),
    ("30y+", 30.0, np.inf),
)

BUCKET_ORDER = [name for name, _, _ in BUCKETS]
BUCKET_MID = {
    "0-1y": 0.5,
    "1-3y": 2.0,
    "3-7y": 5.0,
    "7-10y": 8.5,
    "10-15y": 12.5,
    "15-20y": 17.5,
    "20-30y": 25.0,
    "30y+": 35.0,
}

# Static issue-tenor labels used by the country scrapers ("10y", "30yL", "3m").
_STATIC_YEARS = {
    "1m": 1 / 12,
    "3m": 0.25,
    "6m": 0.5,
    "9m": 0.75,
    "12m": 1.0,
    "1y": 1.0,
    "2y": 2.0,
    "3y": 3.0,
    "4y": 4.0,
    "5y": 5.0,
    "6y": 6.0,
    "7y": 7.0,
    "8y": 8.0,
    "10y": 10.0,
    "15y": 15.0,
    "20y": 20.0,
    "25y": 25.0,
    "30y": 30.0,
    "40y": 40.0,
    "50y": 50.0,
}


def years_from_label(label: object) -> float | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    text = str(label).strip().replace(" ", "")
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return None
    text = text.replace("L", "").replace("l", "")
    if text in _STATIC_YEARS:
        return _STATIC_YEARS[text]
    return None


def bucket_from_years(years: object) -> str:
    if years is None or (isinstance(years, float) and np.isnan(years)):
        return "unknown"
    try:
        value = float(years)
    except (TypeError, ValueError):
        return "unknown"
    if value < 0:
        return "unknown"
    for name, lo, hi in BUCKETS:
        if lo <= value < hi:
            return name
    return "30y+"


def attach_tenor(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill resid_mat_yrs / tenor_bucket from dates or static labels."""
    out = frame.copy()
    if "resid_mat_yrs" not in out.columns:
        out["resid_mat_yrs"] = np.nan
    out["resid_mat_yrs"] = pd.to_numeric(out["resid_mat_yrs"], errors="coerce")
    missing = out["resid_mat_yrs"].isna()
    if "op_dt" in out.columns and "maturity_dt" in out.columns:
        op = pd.to_datetime(out["op_dt"], errors="coerce")
        mat = pd.to_datetime(out["maturity_dt"], errors="coerce")
        computed = (mat - op).dt.days / 365.25
        out.loc[missing, "resid_mat_yrs"] = computed[missing]
        missing = out["resid_mat_yrs"].isna()
    for col in ("sec_desc_dyn", "sec_desc"):
        if col in out.columns and missing.any():
            mapped = pd.to_numeric(out.loc[missing, col].map(years_from_label), errors="coerce")
            out.loc[missing, "resid_mat_yrs"] = mapped.to_numpy()
            missing = out["resid_mat_yrs"].isna()
    out["tenor_bucket"] = out["resid_mat_yrs"].map(bucket_from_years)
    return out
