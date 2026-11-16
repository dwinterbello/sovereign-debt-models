"""AFT text helpers: French maturity strings and auction-file unit scaling."""

from __future__ import annotations

import re

import pandas as pd

_FR_MONTHS = {
    "janvier": 1, "jan": 1,
    "février": 2, "fevrier": 2, "feb": 2, "february": 2,
    "mars": 3, "mar": 3, "march": 3,
    "avril": 4, "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "juin": 6, "jun": 6, "june": 6,
    "juillet": 7, "jul": 7, "july": 7,
    "août": 8, "aout": 8, "aug": 8, "august": 8,
    "septembre": 9, "sep": 9, "sept": 9, "september": 9,
    "octobre": 10, "oct": 10, "october": 10,
    "novembre": 11, "nov": 11, "november": 11,
    "décembre": 12, "decembre": 12, "dec": 12, "december": 12,
}

_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-zéûôà]+)\s+(\d{4})", re.IGNORECASE)
_SLASH_DATE_RE = re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{4})")
_COUPON_RE = re.compile(r"(\d+[,.]?\d*)\s*%")


def parse_french_maturity(text: object) -> pd.Timestamp:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return pd.NaT
    raw = str(text)
    match = _DATE_RE.search(raw)
    if match:
        month = _FR_MONTHS.get(match.group(2).lower())
        if month is not None:
            try:
                return pd.Timestamp(int(match.group(3)), month, int(match.group(1)))
            except ValueError:
                return pd.NaT
    slash = _SLASH_DATE_RE.search(raw)
    if slash:
        day, month, year = (int(slash.group(1)), int(slash.group(2)), int(slash.group(3)))
        if month > 12 and day <= 12:
            day, month = month, day
        try:
            return pd.Timestamp(year, month, day)
        except ValueError:
            return pd.NaT
    return pd.NaT


def parse_coupon(text: object) -> float:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return float("nan")
    match = _COUPON_RE.search(str(text).replace(" ", ""))
    if not match:
        match = _COUPON_RE.search(str(text))
    if not match:
        return float("nan")
    return float(match.group(1).replace(",", "."))


def scale_yield(series: pd.Series) -> pd.Series:
    """AFT stores yields as 0.0381; canonical schema is percent."""
    out = pd.to_numeric(series, errors="coerce")
    sample = out.dropna().abs()
    if len(sample) and float(sample.median()) < 0.5:
        return out * 100.0
    return out


def scale_price(series: pd.Series) -> pd.Series:
    """AFT stores prices as 1.0155; canonical schema is per 100."""
    out = pd.to_numeric(series, errors="coerce")
    sample = out.dropna().abs()
    if len(sample) and float(sample.median()) < 5:
        return out * 100.0
    return out
