"""Norges Bank raw data → ``sovereign_cash/data/no/`` (SDMX REST, data.norges-bank.no, no key).

LIQUIDITY_STATISTICS   business-daily liquidity effects, accumulated from 1 January (UNIT_MEASURE=ALE),
                       NOK mn, 2008→: government accounts (Tab1 line 1) and its specification (Tab4:
                       taxes, VAT, NAV, local authorities, oil tax, SDFI, health, other), government
                       securities (Tab2: bond / bill issuance, maturities, repos), FX purchases for the
                       GPFG, financing of the transfer to the state, notes and coins, Norges Bank lending.
                       Published Mon/Thu, two business days' lag.
LIQUIDITY_FORECAST     structural / total liquidity, realised (STAT_*) and Norges Bank's daily forecast
                       path to year-end (FCAST_*), one vintage per publication (FORECAST_DATE), 2014→.
SHORT_RATES / IR       NOWA, policy rate.
ANN_FX_SPU             monthly announcement of NB's daily FX transactions for the GPFG (buy/sell, NOK mn/day).
GOVT_PRIMARY_MARKET    per-ISIN auctions / syndications / own-stock changes with outstanding volume,
                       own holdings and maturity (for the forward redemption schedule).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "no"
API = "https://data.norges-bank.no/api/data"
URLS = {
    "liquidity_statistics": f"{API}/LIQUIDITY_STATISTICS/?format=csv&startPeriod=2000-01-01&locale=en",
    "liquidity_forecast": f"{API}/LIQUIDITY_FORECAST/?format=csv&startPeriod=2000-01-01&locale=en",
    "nowa": f"{API}/SHORT_RATES/B.NOWA.ON.R?format=csv&startPeriod=2011-01-01&locale=en",
    "policy_rate": f"{API}/IR/B.KPRA.SD.R?format=csv&startPeriod=2011-01-01&locale=en",
    "fx_spu": f"{API}/ANN_FX_SPU/?format=csv&startPeriod=2000-01-01&locale=en",
    "govt_primary": f"{API}/GOVT_PRIMARY_MARKET/?format=csv&startPeriod=2000-01-01&locale=en",
}


def _get(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";", dtype=str)
    df["value"] = pd.to_numeric(df["OBS_VALUE"].str.replace(",", "", regex=False), errors="coerce")
    return df


def fetch_all() -> dict[str, int]:
    RAW.mkdir(parents=True, exist_ok=True)
    n = {}
    for name, url in URLS.items():
        df = _get(url)
        keep = [c for c in df.columns if c == c.upper() or c == "value"]
        df[keep].to_parquet(RAW / f"{name}.parquet", index=False)
        n[name] = len(df)
    (RAW / "fetch_log.json").write_text(json.dumps({"fetched": pd.Timestamp.now(tz="UTC").isoformat(), **n}))
    return n


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW / f"{name}.parquet")


if __name__ == "__main__":
    print(fetch_all())
