"""Shared labels and timing helpers for the channel builders."""
from __future__ import annotations

import pandas as pd

NAMES = {"fr": "France", "de": "Germany", "it": "Italy", "uk": "United Kingdom", "us": "United States",
         "es": "Spain", "se": "Sweden", "no": "Norway", "jp": "Japan"}
CCY = {"fr": "EUR", "de": "EUR", "it": "EUR", "es": "EUR", "uk": "GBP", "us": "USD",
       "se": "SEK", "no": "NOK", "jp": "JPY"}
LINE_LABELS = {
    "recettes": "Receipts (recettes)", "revenue": "Revenue", "tax_rev": "Tax revenue",
    "other_rev": "Other revenue", "primary": "Primary spending", "interest": "Interest",
    "speciaux": "Special accounts", "cap_exp": "Capital spending", "treasury_bal": "Treasury balance",
    "receipts": "Receipts", "petroleum": "Petroleum cash flows",
}
N_YEARS = 4  # current year + 3, as the published unit-shock tables


def period_weights(label: str) -> dict[int, float]:
    """Calendar-year weights of one period label: ``FY2027`` (US, Oct-Sep), ``2026-27`` (UK, Apr-Mar), ``2026``."""
    if label.startswith("FY"):
        y = int(label[2:])
        return {y - 1: 0.25, y: 0.75}
    if "-" in label:
        y = int(label[:4])
        return {y: 0.75, y + 1: 0.25}
    return {int(label): 1.0}


def ts(x) -> str:
    return str(pd.Timestamp(x).date())


def tau(month: pd.Timestamp, onset: pd.Timestamp, lag: int = 0) -> int:
    """Months the shock has been in force by ``month`` (1 in the onset month), after ``lag``."""
    n = (month.year - onset.year) * 12 + (month.month - onset.month) + 1 - lag
    return max(0, int(n))


def period_months(start: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, periods=12, freq="MS"))


def calendar_periods(years: list[int]) -> list[list[pd.Timestamp]]:
    return [period_months(pd.Timestamp(f"{y}-01-01")) for y in years]


def uk_fy_periods(first_fy: int, n: int) -> list[list[pd.Timestamp]]:
    """UK fiscal years April-March; ``first_fy`` 2026 = 2026-27."""
    return [period_months(pd.Timestamp(f"{y}-04-01")) for y in range(first_fy, first_fy + n)]


def us_fy_periods(first_fy: int, n: int) -> list[list[pd.Timestamp]]:
    """US fiscal years October-September; ``first_fy`` 2027 = Oct 2026 - Sep 2027."""
    return [period_months(pd.Timestamp(f"{y - 1}-10-01")) for y in range(first_fy, first_fy + n)]


def tau_months(periods: list[list[pd.Timestamp]], onset: pd.Timestamp, lag: int = 0) -> list[list[int]]:
    return [[tau(m, onset, lag) for m in months] for months in periods]


def period_end_tau(periods: list[list[pd.Timestamp]], onset: pd.Timestamp, lag: int = 0) -> list[int]:
    """Months in force by the end of each period (the shock-year index the matrix type uses)."""
    return [tau(months[-1], onset, lag) for months in periods]


def exposure(periods: list[list[pd.Timestamp]], onset: pd.Timestamp) -> list[float]:
    """Share of each period's months on or after the onset."""
    return [sum(m >= onset for m in months) / len(months) for months in periods]


def default_onset() -> pd.Timestamp:
    """Shocks start next month: history is never shocked."""
    return (pd.Timestamp.today().normalize().replace(day=1) + pd.offsets.MonthBegin(1))
