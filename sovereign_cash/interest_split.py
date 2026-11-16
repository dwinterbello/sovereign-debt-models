"""Carve debt interest out of monthly spending so rates can drive it.

The published monthly books (FR SMB, IT BdI T1, DE BMF) do not split interest.
Interest sits inside spending and was wrongly driven by CPI. This module takes
the central-government interest path, spreads each year across months with that
year's spending calendar, and returns:

  interest_bn   monthly carved interest
  primary_bn    spending − interest (floored at 0)

France: AFT *charge budgétaire de la dette* overlays forecast years (not the
debt-book projection). Rate scenarios for FR use AFT's published +100bp
triangle (1y / 5y / 9y). IT/DE keep the ISIN-book ``debt_repricing`` shock.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = Path(__file__).resolve().parent / "derived"
FR_AUTH = Path(__file__).resolve().parent / "data" / "fr" / "authority.json"
SPEND_COL = {"fr": "depenses", "it": "cur_exp", "de": "expenditure"}


def annual_interest(cc: str) -> pd.Series:
    """Year → interest bn. Outturn years use realised; forecast years prefer authority."""
    d = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    d = d[d["cc"] == cc].sort_values("year")
    out = {}
    for _, r in d.iterrows():
        y = int(r["year"])
        if bool(r["forecast"]) and np.isfinite(r.get("interest_authority_bn", np.nan)):
            out[y] = float(r["interest_authority_bn"])
        else:
            out[y] = float(r["interest_bn"])
    s = pd.Series(out, dtype=float)
    if cc == "fr" and FR_AUTH.exists():
        charge = json.loads(FR_AUTH.read_text())["debt_charge"]["years"]
        for y, row in charge.items():
            s.loc[int(y)] = float(row["bn"])
    return s


def annual_effective_rate(cc: str) -> pd.Series:
    """Interest / debt × 100, same year basis as ``annual_interest``."""
    d = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    d = d[d["cc"] == cc].sort_values("year")
    out = {}
    for _, r in d.iterrows():
        y = int(r["year"])
        debt = float(r["debt_bn"])
        if debt <= 0:
            continue
        if bool(r["forecast"]) and np.isfinite(r.get("interest_authority_bn", np.nan)):
            i = float(r["interest_authority_bn"])
        else:
            i = float(r["interest_bn"])
        out[y] = 100.0 * i / debt
    return pd.Series(out, dtype=float)


def rates_monthly(cc: str, through: pd.Timestamp | None = None) -> pd.Series:
    """Monthly effective-rate level (pp), authority path past the last outturn year."""
    ann = annual_effective_rate(cc)
    if ann.empty:
        return pd.Series(dtype=float)
    end = through or pd.Timestamp(f"{int(ann.index.max())}-12-01")
    start = pd.Timestamp(f"{int(ann.index.min())}-01-01")
    months = pd.date_range(start, end, freq="MS")
    vals = []
    for m in months:
        y = int(m.year)
        if y in ann.index:
            vals.append(float(ann.loc[y]))
        elif len(ann.loc[:y]):
            vals.append(float(ann.loc[:y].iloc[-1]))
        else:
            vals.append(float(ann.iloc[0]))
    return pd.Series(vals, index=months, name="rates")


def _year_share(spend: pd.Series, y: int, months: pd.Index) -> float:
    """Share of a year's spending that falls in ``months``, from up to three earlier full years.

    A part-year (the current year, or a series that starts mid-year) gets this share of the
    annual interest, not all of it. Equal months if there is no full year to learn from.
    """
    pos = spend.clip(lower=0)
    shares = []
    for py in range(y - 1, y - 4, -1):
        full = pos.loc[f"{py}-01-01":f"{py}-12-01"]
        if len(full) == 12 and float(full.sum()) > 0:
            shares.append(float(full[full.index.month.isin(months)].sum() / full.sum()))
    return float(np.mean(shares)) if shares else len(months) / 12.0


def split(cc: str, spend: pd.Series) -> pd.DataFrame:
    """Monthly ``interest`` and ``primary`` from annual interest × spending shares.

    A year with fewer than twelve months of spending (the year in progress) is given only
    its months' share of the annual interest (``_year_share``). Until Nov 2026 it was given
    the whole year's, which put twelve months of interest into Jan-Aug 2026.
    """
    spend = spend.dropna().sort_index()
    ann = annual_interest(cc)
    interest = pd.Series(0.0, index=spend.index, dtype=float)

    def fill(y: int, total: float) -> None:
        chunk = spend.loc[f"{y}-01-01":f"{y}-12-01"]
        if chunk.empty:
            return
        if len(chunk) < 12:
            total = total * _year_share(spend, y, chunk.index.month)
        s = float(chunk.clip(lower=0).sum())
        if s <= 0:  # equal month fill when spending is signed / empty
            interest.loc[chunk.index] = float(total) / len(chunk)
        else:
            interest.loc[chunk.index] = float(total) * chunk.clip(lower=0) / s

    for y, total in ann.items():
        fill(int(y), float(total))
    # years in the spend index past the interest table: hold last annual total
    if len(ann):
        last_y, last_i = int(ann.index.max()), float(ann.iloc[-1])
        for y in sorted({m.year for m in spend.index if m.year > last_y}):
            fill(y, last_i)
    primary = (spend - interest).clip(lower=0)
    # if interest > spend in a month, absorb into primary=0 and keep interest
    return pd.DataFrame({"interest": interest, "primary": primary, "spend": spend})


def _aft_rate_triangle(shock_bp: float = 100.0, onset: pd.Timestamp | None = None) -> dict[int, float]:
    """AFT permanent +100bp by calendar year, from ``onset`` (default: next month).

    The k-th twelve months after the shock cost f(k) a year, f straight-line through
    (0, 0) and AFT's published points (1y 3.2, 5y 23.5, 9y 33.5) and flat after; each
    month carries f(k)/12. Until Nov 2026 year 1 was pinned to calendar 2026 whatever
    the start date, which put AFT a year ahead of our own book.
    """
    raw = json.loads(FR_AUTH.read_text())["rate_shock"]["triangle"]
    pts = sorted((int(k), float(v)) for k, v in raw.items())
    hs, vs = [0.0] + [h for h, _ in pts], [0.0] + [v for _, v in pts]
    if onset is None:
        onset = pd.Timestamp.today().normalize().replace(day=1) + pd.offsets.MonthBegin(1)
    scale = shock_bp / 100.0
    out = {}
    for y in range(onset.year, onset.year + 10):
        tot = 0.0
        for m in pd.date_range(f"{y}-01-01", periods=12, freq="MS"):
            t = (m.year - onset.year) * 12 + (m.month - onset.month) + 1
            if t > 0:
                tot += float(np.interp(int(np.ceil(t / 12.0)), hs, vs)) / 12.0
        out[y] = tot * scale
    return out


def rates_shock_delta(cc: str, shock_bp: float = 100.0) -> dict[int, float]:
    """Δ interest bn by calendar year from a +100bp permanent parallel shift.

    FR: AFT published triangle. IT/DE: ``debt_repricing`` baseline/current vs shock.
    """
    if cc == "fr" and FR_AUTH.exists():
        return _aft_rate_triangle(shock_bp)
    path = DERIVED / "debt_repricing.parquet"
    if not path.exists():
        return {}
    d = pd.read_parquet(path)
    d = d[(d["cc"] == cc) & (d["scenario"] == "baseline")]
    cur = d[d["case"] == "current"].set_index("year")["interest_bn"]
    shk = d[d["case"] == "shock"].set_index("year")["interest_bn"]
    common = cur.index.intersection(shk.index)
    scale = shock_bp / 100.0
    return {int(y): float(shk.loc[y] - cur.loc[y]) * scale for y in common}
