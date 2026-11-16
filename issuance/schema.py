"""Canonical operations schema shared by every country adapter."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

OPS_COLUMNS = (
    "country",
    "ccy",
    "op_dt",
    "issue_dt",
    "maturity_dt",
    "isin",
    "cusip",
    "sec_desc_long",
    "amount_issued",
    "official_take_mn",
    "amount_bid",
    "price",
    "yield",
    "coupon",
    "btc",
    "sec_type",
    "op_type",
    "sec_desc",
    "sec_desc_dyn",
    "resid_mat_yrs",
    "tenor_bucket",
    "is_forecast",
    "source",
)

SEC_TYPES = ("Bill", "Nominal", "Linker", "FRN")
OP_TYPES = (
    "Auction",
    "Syndication",
    "Tender",
    "TAP",
    "Buyback",
    "Exchange",
    "Own Holdings",
    "Market Adjustment",
    "Reopening",
    "Liquidity Enhancement",
    "Other",
)

COUNTRY_CCY = {
    "UK": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "IT": "EUR",
    "ES": "EUR",
    "US": "USD",
    "JP": "JPY",
    "SE": "SEK",
}

COUNTRY_ISIN_PREFIX = {
    "UK": "GB",
    "DE": "DE",
    "FR": "FR",
    "IT": "IT",
    "ES": "ES",
    "US": "US",
    "JP": "JP",
    "SE": "SE",
}


def empty_ops() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OPS_COLUMNS))


def coerce_ops(frame: pd.DataFrame, country: str) -> pd.DataFrame:
    """Force a country operations table onto the canonical schema."""
    out = empty_ops() if frame is None or frame.empty else frame.copy()
    for col in OPS_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["country"] = country
    out["ccy"] = COUNTRY_CCY[country]
    for col in ("op_dt", "issue_dt", "maturity_dt"):
        out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
    for col in ("amount_issued", "official_take_mn", "amount_bid", "price", "yield", "coupon", "btc", "resid_mat_yrs"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["isin"] = out["isin"].astype("string").str.strip().str.upper()
    out["cusip"] = out["cusip"].astype("string").str.strip().str.upper()
    if out["resid_mat_yrs"].isna().any() and out["op_dt"].notna().any():
        missing = out["resid_mat_yrs"].isna() & out["op_dt"].notna() & out["maturity_dt"].notna()
        out.loc[missing, "resid_mat_yrs"] = (
            (out.loc[missing, "maturity_dt"] - out.loc[missing, "op_dt"]).dt.days / 365.25
        )
    out["is_forecast"] = out["is_forecast"].fillna(False).astype(bool)
    out["sec_type"] = out["sec_type"].astype("string")
    out["op_type"] = out["op_type"].astype("string")
    return out[list(OPS_COLUMNS)].reset_index(drop=True)


def concat_ops(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return empty_ops()
    return pd.concat(parts, ignore_index=True)
