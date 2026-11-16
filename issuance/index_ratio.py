"""Linker index ratios, so a linker's DV01 is on its inflation-uplifted principal.

Auction and holdings amounts for linkers are nominal (unindexed) face. The cash, and the real-rate
DV01, scale with the index ratio: reference index at the date / reference index at the line's base
date, both three months lagged. UK: the DMO's monthly RPI (within 1% of the DMO's published ratios
on every live linker). US: CPI-U NSA (FRED CPIAUCNS). Euro area: HICP all items (ECB, spliced onto
Eurostat's ECOICOP 2 index from Jan 2026); the OAT€i/BTP€i reference is ex tobacco and BTP Italia's
is Italian FOI ex tobacco, both within ~1% of it. Japan: nationwide CPI ex fresh food (the JGBi
reference), Statistics Bureau 2020 base chained onto the 2025 base; before Jan 2020 it is OECD all
items rescaled at the splice. Months past the last print carry the trailing-year rate.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE
from issuance.uk_remit import INDEX_RATIOS

RPI_XML = INDEX_RATIOS.parent / "rpi.xml"
DATA = Path(__file__).resolve().parent / "data"
LAG_MONTHS = 3


def _extend(level: pd.Series) -> pd.Series:
    level = level.dropna().sort_index()
    step = (level.iloc[-1] / level.iloc[-13]) ** (1 / 12)
    ext = pd.date_range(level.index[-1] + pd.DateOffset(months=1), periods=48, freq="MS")
    return pd.concat([level, pd.Series(level.iloc[-1] * step ** np.arange(1, 49), index=ext)])


def _csv(name: str) -> pd.Series:
    d = pd.read_csv(DATA / name, parse_dates=["month"])
    return _extend(d.set_index("month")["index"].astype(float))


@lru_cache(maxsize=1)
def _uk_rpi() -> pd.Series:
    vals = {}
    for el in ET.parse(RPI_XML).getroot().iter("DIM_DATE"):
        s = el.find("SILO_RPI_SERIES")
        if s is not None and s.get("RPI_SERIES_JAN_87"):
            vals[pd.Timestamp(el.get("RPI_DATE")[:10])] = float(s.get("RPI_SERIES_JAN_87"))
    return _extend(pd.Series(vals))


@lru_cache(maxsize=1)
def _levels() -> dict[str, pd.Series]:
    ea = _csv("ea_hicp.csv")
    out = {"US": _csv("us_cpiu_nsa.csv"), "DE": ea, "FR": ea, "IT": ea, "JP": _csv("jp_cpi_core.csv")}
    if RPI_XML.exists():
        out["UK"] = _uk_rpi()
    return out


@lru_cache(maxsize=1)
def base_dates() -> dict[tuple[str, str], pd.Timestamp]:
    """First tape date of every linker line, by country and ISIN."""
    out = {}
    for c in ("uk", "de", "fr", "it", "es", "us", "jp"):
        path = CACHE / f"ops_{c}.parquet"
        if not path.exists():
            continue
        o = pd.read_parquet(path, columns=["isin", "op_dt", "sec_type"])
        o = o[o["sec_type"].astype(str).eq("Linker") & o["isin"].notna()]
        for isin, day in o.groupby(o["isin"].astype(str))["op_dt"].min().items():
            out[(c.upper(), isin)] = pd.Timestamp(day)
    return out


def _level(series: pd.Series, day: pd.Timestamp) -> float:
    m = (pd.Timestamp(day) - pd.DateOffset(months=LAG_MONTHS)).to_period("M").to_timestamp()
    m = min(max(m, series.index[0]), series.index[-1])
    return float(series.asof(m))


def index_ratio(country: str, isin: str, day) -> float:
    """1.0 for a line not on the tape (a new issue starts at par)."""
    base = base_dates().get((str(country), str(isin)))
    series = _levels().get(str(country))
    if base is None or series is None or pd.isna(day) or pd.Timestamp(day) <= base:
        return 1.0
    return _level(series, pd.Timestamp(day)) / _level(series, base)


def ratios_for(frame: pd.DataFrame, date_col: str) -> np.ndarray:
    """Index ratio per row; 1.0 for non-linkers or rows without a country and ISIN."""
    out = np.ones(len(frame))
    if not {"country", "isin", "sec_type"} <= set(frame.columns) or date_col not in frame.columns:
        return out
    linker = frame["sec_type"].astype(str).eq("Linker").to_numpy()
    if not linker.any():
        return out
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    cache: dict = {}
    for i in np.flatnonzero(linker):
        key = (str(frame["country"].iat[i]), str(frame["isin"].iat[i]), dates.iat[i])
        if key not in cache:
            cache[key] = index_ratio(*key)
        out[i] = cache[key]
    return out
