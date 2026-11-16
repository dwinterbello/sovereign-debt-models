"""Daily USD crosses. FRED first; ECB SDW cross rates if FRED is dark."""

from __future__ import annotations

import io

import pandas as pd
import requests

from issuance.http import get_bytes
from issuance.paths import CACHE

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ECB = "https://data-api.ecb.europa.eu/service/data/EXR/D.{ccy}.EUR.SP00.A?format=csvdata"
SERIES = {"GBPUSD": "DEXUSUK", "EURUSD": "DEXUSEU"}


def _series_from_csv(payload: bytes, date_hints=("TIME_PERIOD", "DATE", "observation_date"),
                     val_hints=("OBS_VALUE", "OBSVALUE")) -> pd.Series:
    df = pd.read_csv(io.BytesIO(payload))
    date_hints_u = {h.upper() for h in date_hints}
    val_hints_u = {h.upper() for h in val_hints}
    date_col = next((c for c in df.columns if c.upper() in date_hints_u), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "DATE" in c.upper() or "TIME_PERIOD" in c.upper()), df.columns[0])
    val_col = next((c for c in df.columns if c.upper() in val_hints_u), None)
    if val_col is None:
        val_col = next((c for c in df.columns if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().any()), df.columns[1])
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col], errors="coerce")
    return s.dropna()


def _fred(code: str) -> pd.Series:
    # FRED tarpits custom / browser user agents (get_bytes sends one); the requests default is served.
    r = requests.get(FRED.format(code), timeout=(10, 60))
    r.raise_for_status()
    return _series_from_csv(r.content)


def _ecb_per_eur(ccy: str) -> pd.Series:
    payload = get_bytes(ECB.format(ccy=ccy), timeout=60, retries=2)
    return _series_from_csv(payload)


def _from_ecb() -> dict[str, pd.Series]:
    usd_per_eur = _ecb_per_eur("USD")  # USD per 1 EUR
    gbp_per_eur = _ecb_per_eur("GBP")  # GBP per 1 EUR
    jpy_per_eur = _ecb_per_eur("JPY")  # JPY per 1 EUR
    sek_per_eur = _ecb_per_eur("SEK")  # SEK per 1 EUR
    aligned = pd.concat(
        {"EURUSD": usd_per_eur, "GBP_EUR": gbp_per_eur, "JPY_EUR": jpy_per_eur, "SEK_EUR": sek_per_eur},
        axis=1,
    )
    aligned = aligned.dropna(subset=["EURUSD", "GBP_EUR"])
    aligned["GBPUSD"] = aligned["EURUSD"] / aligned["GBP_EUR"]
    aligned["JPYUSD"] = aligned["EURUSD"] / aligned["JPY_EUR"]
    aligned["SEKUSD"] = aligned["EURUSD"] / aligned["SEK_EUR"]
    return {
        "EURUSD": aligned["EURUSD"], "GBPUSD": aligned["GBPUSD"],
        "JPYUSD": aligned["JPYUSD"], "SEKUSD": aligned["SEKUSD"],
    }


def fetch_fx(force: bool = False) -> pd.DataFrame:
    dest = CACHE / "fx_daily.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)
    parts: dict[str, pd.Series] = {}
    try:
        parts = _from_ecb()
        if not parts or any(s.empty for s in parts.values()):
            raise RuntimeError("ECB FX series empty")
        print("  FX from ECB SDW (USD/EUR, GBP/EUR → GBPUSD)")
    except Exception as exc:
        print(f"  ECB unavailable ({exc}); trying FRED")
        parts = {name: _fred(code) for name, code in SERIES.items()}
        parts["JPYUSD"] = 1.0 / _fred("DEXJPUS")
        parts["SEKUSD"] = 1.0 / _fred("DEXSDUS")  # FRED: SEK per USD
    fx = pd.concat(parts, axis=1).sort_index()
    fx["USDUSD"] = 1.0
    fx = fx.reset_index()
    date_col = fx.columns[0]
    fx = fx.rename(columns={date_col: "date"})
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    if fx[["EURUSD", "GBPUSD"]].dropna(how="all").empty:
        raise RuntimeError("FX panel empty after FRED/ECB fetch")
    fx.to_parquet(dest, index=False)
    return fx


def fx_on(fx: pd.DataFrame, date, ccy: str) -> float:
    col = {"GBP": "GBPUSD", "EUR": "EURUSD", "USD": "USDUSD", "JPY": "JPYUSD", "SEK": "SEKUSD"}[ccy]
    panel = fx.set_index("date")[col].sort_index()
    ts = pd.Timestamp(date)
    if ts in panel.index and pd.notna(panel.loc[ts]):
        return float(panel.loc[ts])
    prior = panel.loc[:ts].dropna()
    if prior.empty:
        return float(panel.dropna().iloc[0])
    return float(prior.iloc[-1])
