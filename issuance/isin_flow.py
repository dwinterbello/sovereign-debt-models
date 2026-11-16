"""Group operations into ISIN-level and tenor-level monthly flows."""

from __future__ import annotations

import pandas as pd

from issuance.paths import CACHE
from issuance.schema import concat_ops
from issuance.tenor import attach_tenor, BUCKET_ORDER


def load_ops() -> pd.DataFrame:
    parts = []
    for country in ("uk", "de", "fr", "it", "es", "us", "jp", "se"):
        path = CACHE / f"ops_{country}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    from issuance.fetch_es import tag_syndications as tag_es_syndications
    from issuance.fetch_it import supplement_it
    from issuance.fetch_uk import tag_creations
    from issuance.pricing import clean_ops

    return clean_ops(attach_tenor(supplement_it(tag_es_syndications(tag_creations(concat_ops(parts))))))


# DFA Own Holdings are issuer inventory, not market supply. They stay in the
# ops file for stock/WAM; they do not enter gross issuance or DV01.
NON_MARKET_OP = frozenset({"Own Holdings"})
# Riksgälden outstanding snapshot rows sit beside auction taps in ops_se.
STOCK_SOURCE = "outstanding CH1073"


def market_ops(ops: pd.DataFrame) -> pd.DataFrame:
    if ops is None or ops.empty or "op_type" not in ops.columns:
        return ops
    out = ops[~ops["op_type"].astype(str).isin(NON_MARKET_OP)].copy()
    if "source" in out.columns:
        out = out[~out["source"].astype(str).str.contains(STOCK_SOURCE, na=False)]
    if "sec_type" in out.columns:
        out = out[~out["sec_type"].astype(str).isin(["Foreign"])]
    # Buybacks with no maturity cannot be assigned to an ISIN stock.
    # They stay on the ops file; reconcile reports the unallocated sum.
    if "maturity_dt" in out.columns:
        out = out[out["maturity_dt"].notna()]
    return out


def _public(g: pd.DataFrame) -> pd.DataFrame:
    """Face sold to the market: a Fed rollover add-on is in the stock, not in supply."""
    g = g.copy()
    if "official_take_mn" not in g.columns:
        g["official_take_mn"] = 0.0
    take = pd.to_numeric(g["official_take_mn"], errors="coerce").fillna(0.0)
    amt = pd.to_numeric(g["amount_issued"], errors="coerce")
    g["official_take_mn"] = take.where(amt > 0, 0.0)
    g["amount_issued"] = amt.where(amt <= 0, (amt - g["official_take_mn"]).clip(lower=0.0))
    return g


def by_isin(ops: pd.DataFrame) -> pd.DataFrame:
    g = _public(market_ops(ops))
    ids = g["isin"].astype(str)
    # JGBs and T-Discount Bills carry MoF issue numbers (JP10y#383, JPTDB#1413), not ISINs.
    jgb = g["country"].astype(str).eq("JP") & ids.str.fullmatch(r"JP[A-Z]*(?:\d+y)?#\d+")
    g = g[g["isin"].notna() & ((ids.str.len() >= 10) | jgb)].copy()
    g["month"] = pd.to_datetime(g["op_dt"]).dt.to_period("M").dt.to_timestamp()
    out = g.groupby(["country", "ccy", "isin", "month", "sec_type", "tenor_bucket"], dropna=False).agg(
        gross_mn=("amount_issued", lambda s: s[s > 0].sum()),
        buyback_mn=("amount_issued", lambda s: s[s < 0].sum()),
        n_ops=("amount_issued", "count"),
        official_take_mn=("official_take_mn", "sum"),
        coupon=("coupon", "last"),
        yield_avg=("yield", "mean"),
        price_avg=("price", "mean"),
        maturity_dt=("maturity_dt", "last"),
        resid_mat_yrs=("resid_mat_yrs", "mean"),
        sec_desc_long=("sec_desc_long", "last"),
    ).reset_index()
    out["net_mn"] = out["gross_mn"] + out["buyback_mn"]
    return out.sort_values(["country", "isin", "month"])


def by_tenor(ops: pd.DataFrame) -> pd.DataFrame:
    g = _public(market_ops(ops))
    g["month"] = pd.to_datetime(g["op_dt"]).dt.to_period("M").dt.to_timestamp()
    out = g.groupby(["country", "ccy", "month", "sec_type", "tenor_bucket"], dropna=False).agg(
        gross_mn=("amount_issued", lambda s: s[s > 0].sum()),
        buyback_mn=("amount_issued", lambda s: s[s < 0].sum()),
        n_ops=("amount_issued", "count"),
        official_take_mn=("official_take_mn", "sum"),
        n_isin=("isin", "nunique"),
        resid_mat_yrs=("resid_mat_yrs", "mean"),
    ).reset_index()
    out["net_mn"] = out["gross_mn"] + out["buyback_mn"]
    out["tenor_bucket"] = pd.Categorical(out["tenor_bucket"], categories=BUCKET_ORDER + ["unknown"], ordered=True)
    return out.sort_values(["country", "month", "tenor_bucket"])


def write_flows(ops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    isin = by_isin(ops)
    tenor = by_tenor(ops)
    isin.to_parquet(CACHE / "flow_isin.parquet", index=False)
    tenor.to_parquet(CACHE / "flow_tenor.parquet", index=False)
    return isin, tenor
