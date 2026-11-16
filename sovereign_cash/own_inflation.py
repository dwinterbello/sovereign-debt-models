"""House CPI paths, read from fixed input files.

`annual_cpi` reads derived/own_inflation_paths.json; `_history` reads the published history in
derived/own_inflation_history_frozen.json; `refresh()` leaves both as they are.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

SC = Path(__file__).resolve().parent
ROOT = SC.parent
DERIVED = SC / "derived" / "own_inflation_paths.json"
HIST = SC / "derived" / "own_inflation_history_frozen.json"
H_MAX_DEFAULT = 52
END_YEAR = 2030

SOURCE = {
    "us": ("g4", "us", "US house CPI"),
    "uk": ("g4", "uk", "UK house CPI"),
    "jp": ("g4", "jp", "JP house CPI"),
    "de": ("ea", "de", "DE house HICP"),
    "fr": ("ea", "fr", "FR house HICP"),
    "it": ("ea", "it", "IT house HICP"),
    "es": ("ea", "es", "ES house HICP"),
    "se": ("nordic", "policy", "SE house CPIF"),
    "no": ("nordic", "headline", "NO house CPI"),
}


def _frozen_history(cc: str) -> pd.Series:
    node = json.loads(HIST.read_text()).get(cc) if HIST.exists() else None
    if not node:
        return pd.Series(dtype=float)
    return pd.Series({pd.Timestamp(k): v for k, v in node["data"].items()}, dtype=float).sort_index()


_H = json.loads(HIST.read_text()) if HIST.exists() else {}
HISTORY = {cc: (v["label"], (lambda c=cc: _frozen_history(c))) for cc, v in _H.items()}


def _history(cc: str) -> pd.Series | None:
    return HISTORY[cc][1]() if cc in HISTORY else None


def _me(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp(how="end").normalize()
def _annual(actual: pd.Series, forecast: pd.Series) -> dict[int, float]:
    """Calendar-year mean of monthly YoY, published months first. A year needs at least ten of its twelve
    months (the Oct/Nov-2025 US prints lost to the shutdown are interpolated)."""
    m = pd.concat([actual, forecast[~forecast.index.isin(actual.index)]]).sort_index()
    m = m[m.index.year <= END_YEAR]
    out = {}
    for y, g in m.groupby(m.index.year):
        if y < pd.Timestamp.today().year - 1 or len(g) < 10:
            continue
        full = g.reindex(pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="ME")).interpolate(limit_direction="both")
        out[int(y)] = float(full.mean())
    return out


def build_country(cc: str, h_max: int = H_MAX_DEFAULT) -> dict:
    raise NotImplementedError("house CPI paths are fixed inputs: see derived/own_inflation_paths.json")


def refresh(countries=tuple(SOURCE), h_max: int = H_MAX_DEFAULT, out: Path = DERIVED) -> dict:
    """Keep the paths as they are."""
    res = json.loads(out.read_text()) if out.exists() else {}
    res.setdefault("_meta", {})["errors"] = {}
    return res


@lru_cache(maxsize=16)
def annual_cpi(cc: str) -> pd.Series:
    if not DERIVED.exists():
        return pd.Series(dtype=float)
    node = json.loads(DERIVED.read_text()).get(cc)
    if not node or not node.get("annual"):
        return pd.Series(dtype=float)
    s = pd.Series({int(y): float(v) for y, v in node["annual"].items()}, dtype=float).sort_index()
    s.attrs["source"] = node.get("source", "")
    s.attrs["asof"] = node.get("asof", "")
    return s


def source_label(cc: str) -> str:
    s = annual_cpi(cc)
    if s.empty:
        return "IMF WEO CPI (no house path)"
    live = SOURCE.get(cc, (None, None, None))[2] if cc in SOURCE else None
    src = live or s.attrs.get("source") or "house CPI"
    return f"{src} to {int(s.index.max())}, IMF WEO CPI beyond"
