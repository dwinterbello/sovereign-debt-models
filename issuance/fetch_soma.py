"""NY Fed SOMA Treasury holdings (CUSIP → ISIN). Monthly snapshots."""

from __future__ import annotations

import pandas as pd

from issuance.http import FetchError, session
from issuance.paths import CACHE
from issuance.tenor import attach_tenor, bucket_from_years

MONTHLY = "https://markets.newyorkfed.org/api/soma/tsy/get/monthly.json"
ASOF = "https://markets.newyorkfed.org/api/soma/tsy/get/asof/{date}.json"
SUMMARY = "https://markets.newyorkfed.org/api/soma/summary.json"

SEC_TYPE = {
    "Bills": "Bill",
    "NotesBonds": "Nominal",
    "Notes": "Nominal",
    "Bonds": "Nominal",
    "TIPS": "Linker",
    "FRN": "FRN",
    "FRNs": "FRN",
}


def _isin(cusip: object) -> object:
    text = str(cusip).strip().upper().replace(" ", "")
    if len(text) < 8 or text in {"NAN", "NONE", "NULL", ""}:
        return pd.NA
    return "US" + text[:9]


def _pages(force: bool = False) -> pd.DataFrame:
    cached = CACHE / "soma_holdings_raw.parquet"
    if cached.exists() and not force:
        return pd.read_parquet(cached)
    resp = session().get(MONTHLY, timeout=180)
    if resp.status_code != 200 or not resp.content:
        raise FetchError(f"SOMA monthly {resp.status_code}")
    payload = resp.json()
    rows = (payload.get("soma") or {}).get("holdings") or []
    if not rows:
        raise FetchError("SOMA monthly returned no holdings")
    raw = pd.DataFrame(rows)
    raw.to_parquet(cached, index=False)
    return raw


def fetch_soma(force: bool = False) -> pd.DataFrame:
    raw = _pages(force=force)
    out = pd.DataFrame()
    out["asof"] = pd.to_datetime(raw.get("asOfDate"), errors="coerce")
    out["month"] = out["asof"].dt.to_period("M").dt.to_timestamp()
    out["cusip"] = raw.get("cusip")
    out["isin"] = out["cusip"].map(_isin)
    out["maturity_dt"] = pd.to_datetime(raw.get("maturityDate"), errors="coerce")
    out["coupon"] = pd.to_numeric(raw.get("coupon"), errors="coerce")
    par = pd.to_numeric(raw.get("parValue"), errors="coerce")
    out["official_stock_mn"] = par / 1e6
    stype = raw.get("securityType", pd.Series(dtype=str)).astype(str)
    out["sec_type"] = stype.map(lambda x: SEC_TYPE.get(x, "Nominal"))
    out["sec_desc_long"] = stype
    out["resid_mat_yrs"] = (out["maturity_dt"] - out["asof"]).dt.days / 365.25
    out["tenor_bucket"] = out["resid_mat_yrs"].map(bucket_from_years)
    out["country"] = "US"
    out["program"] = "SOMA"
    out = out[out["asof"].notna() & out["isin"].notna() & (out["official_stock_mn"].fillna(0) > 0)]
    # one snapshot per calendar month (last as-of in the month)
    out = out.sort_values(["month", "asof"])
    last_asof = out.groupby("month")["asof"].transform("max")
    out = out[out["asof"] == last_asof]
    out = attach_tenor(out)
    out.to_parquet(CACHE / "soma_holdings.parquet", index=False)
    print(f"  SOMA holdings: {len(out)} rows, {out['month'].nunique()} months, "
          f"{out['asof'].min().date()} → {out['asof'].max().date()}")
    return out


def soma_flow(holdings: pd.DataFrame | None = None, bills_only: bool = False) -> pd.DataFrame:
    """CUSIP stock → monthly official flow. First sample month is 0, not a dump."""
    if holdings is None:
        path = CACHE / "soma_holdings.parquet"
        if not path.exists():
            return pd.DataFrame()
        holdings = pd.read_parquet(path)
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    h = holdings.copy()
    # Bills stay on the holdings file. Coupon official / DV01 is notes, bonds, TIPS, FRN.
    if "sec_type" in h.columns:
        is_bill = h["sec_type"].astype(str) == "Bill"
        h = h[is_bill] if bills_only else h[~is_bill]
    if h.empty:
        return pd.DataFrame()
    h["month"] = pd.to_datetime(h["month"])
    wide = h.pivot_table(
        index="month", columns="isin", values="official_stock_mn", aggfunc="sum"
    ).sort_index().fillna(0.0)
    flow = wide.diff()
    flow.iloc[0] = 0.0
    long = flow.stack().rename("official_mn").reset_index()
    long = long[long["official_mn"].abs() > 1e-8]
    meta = h.sort_values("month").groupby(["month", "isin"]).tail(1)
    keep = [
        c for c in (
            "month", "isin", "tenor_bucket", "sec_type", "maturity_dt",
            "resid_mat_yrs", "coupon", "official_stock_mn",
        ) if c in meta.columns
    ]
    out = long.merge(meta[keep], on=["month", "isin"], how="left")
    last = h.sort_values("month").groupby("isin").tail(1)
    last_keep = [c for c in keep if c != "month"]
    miss = out["tenor_bucket"].isna() if "tenor_bucket" in out.columns else out["isin"].notna()
    if miss.any():
        filled = out.loc[miss, ["month", "isin"]].merge(last[last_keep], on="isin", how="left")
        for col in last_keep:
            if col == "isin" or col not in out.columns:
                continue
            out.loc[miss, col] = filled[col].to_numpy()
    # A flow is bucketed as supply is, by residual inside the month, not at the month-end
    # snapshot: a new 30y is 30.0y at auction and 29.96y at month-end, and the add-on netted out
    # of the 30y+ supply bucket must come out of the same official bucket.
    if "maturity_dt" in out.columns:
        start = pd.to_datetime(out["month"])
        out["resid_mat_yrs"] = (pd.to_datetime(out["maturity_dt"]) - start).dt.days / 365.25
        out["tenor_bucket"] = out["resid_mat_yrs"].map(bucket_from_years)
    out["country"] = "US"
    out["program"] = "SOMA"
    return out
