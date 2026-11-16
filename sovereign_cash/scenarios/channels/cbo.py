"""United States: CBO rules of thumb (April 2026), federal deficit by fiscal year.

CBO publishes sustained shocks only: each variable 0.1pp different in every year
from January 2026. They map to the dashboard's growth mode; a one-off level
shift is not published, so US level mode is unavailable rather than invented.

  ngdp   slower productivity growth, sign flipped (+ = faster), real and nominal
         GDP move together; CBO lets interest rates fall with growth, so the
         interest line moves too
  cpi    'inflation and interest rates' minus 'interest rates': the inflation
         part alone (prices and wages 0.1pp faster, rates unchanged)
  rates  all rates 0.1pp higher; per bp = per 0.1pp / 10

Timing: CBO's shocks start in January 2026; the dashboard's start next month.
Each CBO year is moved later by that gap (linear between fiscal years, zero
before the shock), so both rules and the ISIN book start on the same date.

CBO's 'debt service' lines (interest on the extra borrowing) are dropped: that
effect is the dashboard's own interest-on-extra-borrowing switch everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels.common import CCY, N_YEARS, NAMES, ts

ROT = Path(__file__).resolve().parents[2] / "data" / "us" / "cbo_rules_of_thumb.json"
LINES = [{"id": "receipts", "label": "Revenues", "sign": -1},
         {"id": "primary", "label": "Outlays ex net interest", "sign": 1},
         {"id": "interest", "label": "Net interest", "sign": 1}]
LINE_MAP = {
    "revenues": "receipts",
    "outlays_mandatory": "primary", "outlays_discretionary": "primary",
    "net_interest_rates": "interest", "net_interest_inflation": "interest",
}  # net_interest_debt_service: replaced by the engine's feedback switch


def _shifted(vals: list[float], all_years: list[int], year: int, lag_years: float) -> float:
    """CBO's value for fiscal ``year`` if the shock had started ``lag_years`` later (0 before it starts)."""
    t = year - lag_years
    lo = int(np.floor(t))
    w = t - lo

    def at(y: int) -> float:
        return float(vals[all_years.index(y)]) if y in all_years else 0.0

    return (1.0 - w) * at(lo) + w * at(lo + 1)


def _lines(sc: dict, years: list[int], all_years: list[int], lag_years: float) -> dict[str, list[float]]:
    out = {k: [0.0] * len(years) for k in ("receipts", "primary", "interest")}
    for name, vals in sc["lines"].items():
        line = LINE_MAP.get(name)
        if line is None:
            continue
        for i, y in enumerate(years):
            out[line][i] += _shifted(vals, all_years, y, lag_years)
    return out


def build(book, onset: pd.Timestamp) -> dict:
    from sovereign_cash.debt import sources as S

    rot = json.loads(ROT.read_text())
    all_years = [int(y) for y in rot["years"]]
    cbo_start = pd.Timestamp(rot["shock_start"] + "-01")
    lag = max(0.0, ((onset.year - cbo_start.year) * 12 + onset.month - cbo_start.month) / 12.0)
    first = onset.year + 1 if onset.month >= 10 else onset.year  # the fiscal year the shock lands in
    years = list(range(first, first + N_YEARS))
    sc = rot["scenarios"]
    step = float(rot["shock_size_pp"])

    prod = _lines(sc["productivity"], years, all_years, lag)
    rates = _lines(sc["rates"], years, all_years, lag)
    both = _lines(sc["inflation_and_rates"], years, all_years, lag)
    infl = {k: [b - r for b, r in zip(both[k], rates[k])] for k in both}

    channels = []
    for line in ("receipts", "primary", "interest"):
        g = [-v / step for v in prod[line]]  # CBO shock is slower growth
        if any(g):
            channels.append({
                "id": f"us_ngdp_{line}", "type": "linear", "driver": "ngdp", "line": line,
                "per_unit": {"growth": g},
                "method": "CBO rules of thumb: productivity growth 0.1pp/yr, sign flipped, per pp, moved to our start date",
                "badge": "CBO rule of thumb", "source": rot["source"], "url": rot["url"]})
        c = [v / step for v in infl[line]]
        if any(c):
            channels.append({
                "id": f"us_cpi_{line}", "type": "linear", "driver": "cpi", "line": line,
                "per_unit": {"growth": c},
                "method": "CBO: 'inflation and interest rates' less 'interest rates', per pp, moved to our start date",
                "badge": "CBO rule of thumb", "source": rot["source"], "url": rot["url"]})
        r = [v / (step * 100.0) for v in rates[line]]
        if any(r):
            channels.append({
                "id": f"us_rates_{line}", "type": "linear", "driver": "rates", "line": line,
                "method_group": "authority", "per_unit": {"level": r, "growth": r},
                "method": "CBO: all interest rates 0.1pp higher every year, per bp, moved to our start date",
                "badge": "CBO rule of thumb", "source": rot["source"], "url": rot["url"]})

    default_methods = {"rates": "book"}
    if book is not None:
        weights = [{y - 1: 0.25, y: 0.75} for y in years]  # Oct-Sep fiscal year
        ch = book.channel("us", [], weights=weights,
                          note="; calendar years weighted onto October-September; excludes Fed remittances")
        if ch is not None:
            channels.append(ch)
            channels += book.curve_channels("us", [], weights=weights)

    cbo = S.cbo()
    piv = cbo.pivot_table(index="year", columns="var", values="value")
    total = [-float(piv.at[y, "deficit_bn"]) for y in years]           # CBO stores the balance
    interest = [float(piv.at[y, "interest_net_bn"]) for y in years]
    receipts = [float(piv.at[y, "revenue_bn"]) for y in years]
    primary = [r - float(piv.at[y, "primary_deficit_bn"]) for r, y in zip(receipts, years)]
    return {
        "cc": "us", "name": NAMES["us"], "ccy": CCY["us"], "unit": "USD bn",
        "measure": "Federal deficit (CBO)", "period_kind": "fiscal year (October-September)",
        "periods": [f"FY{y}" for y in years], "origin": ts(onset - pd.offsets.MonthBegin(1)), "onset": ts(onset),
        "lines": LINES,
        "baseline": {"total": total, "lines": {"receipts": receipts, "primary": primary, "interest": interest},
                     "source": str(cbo["vintage"].iloc[-1]) + " baseline"},
        "channels": channels, "default_methods": default_methods,
        "modes": {"ngdp": ["growth"], "cpi": ["growth"]},
        "model": {"family": "CBO rules of thumb", "cbo_shock_start": rot["shock_start"],
                  "lag_years": lag, "note": "CBO years moved later to the dashboard's start; debt service dropped "
                                            "(the interest-on-extra-borrowing switch replaces it)"},
        **({"feedback": {"rate_pct": [r] * len(years), "source": "debt snapshot marginal new-issue rate"}}
           if book is not None and (r := book.feedback_rate("us")) is not None else {}),
    }
