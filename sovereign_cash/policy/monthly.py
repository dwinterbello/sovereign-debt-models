"""Spread legislated policy onto monthly cash-requirement paths.

The scenario pack already keeps a no-policy baseline and a ``with_policy`` total.
Cash boards that run the same no-policy component path for outer years
(``scenarios.channels.sarimax.later_years``) were dropping the policy line, so
France 2027+ on the financing board ignored LPM / surcharge run-off etc.

Rule: add the overlay only on months after the last outturn. Years that are
soft-pinned to an authority solde which already embeds those measures (FR BUM
α=1 for the year in progress) are left alone — callers pass only the months that
come from the no-policy model.
"""
from __future__ import annotations

import pandas as pd

from sovereign_cash.policy import overlay


def period_year(cc: str, month: pd.Timestamp) -> int:
    """Model period year for a month (UK = FY starting April; else calendar)."""
    ts = pd.Timestamp(month)
    if cc == "uk":
        return int(ts.year if ts.month >= 4 else ts.year - 1)
    if cc == "us":
        # US federal FY starts October.
        return int(ts.year + 1 if ts.month >= 10 else ts.year)
    return int(ts.year)


def base_year(cc: str, origin: pd.Timestamp) -> int:
    """Last complete period at ``origin`` (the outturn the model carries forward)."""
    ts = pd.Timestamp(origin)
    py = period_year(cc, ts)
    # If origin is inside a period, that period is still open — base is the prior one.
    if cc == "uk":
        return py if ts.month == 3 else py - 1
    if cc == "us":
        return py if ts.month == 9 else py - 1
    return py if ts.month == 12 else py - 1


def _growth(cc: str, years: list[int], base: int) -> dict[str, pd.Series] | None:
    try:
        from sovereign_cash.scenarios.channels import own_annual as OA
    except Exception:  # noqa: BLE001
        return None
    if cc not in getattr(OA, "COUNTRIES", ()):
        return None
    span = list(range(base, max(years) + 1))
    y, p = OA.levels(cc, span)
    spend = y if OA.method(cc, "primary") == "persist_gdp" else p
    return {"tax": y, "spend": spend}


def annual_effects(cc: str, years: list[int], base: int,
                   first_share: float = 1.0) -> dict[int, float] | None:
    """Period-year → effect on borrowing (+ adds), local bn. None if no measures."""
    if cc not in overlay.COUNTRIES or not years:
        return None
    pol = overlay.build(cc, list(years), base, first_share, growth=_growth(cc, years, base))
    if pol is None:
        return None
    return {int(y): float(v) for y, v in zip(years, pol["total"])}


def add_to_monthly(cc: str, s: pd.Series, origin: pd.Timestamp,
                   *, years: list[int] | None = None) -> pd.Series:
    """Return ``s`` with legislated policy spread across forecast months after ``origin``.

    Each period year's overlay total is split equally over the forecast months that
    fall in that period (so the year sums to the overlay, including a scaled first
    open year when ``first_share`` < 1).
    """
    if cc not in overlay.COUNTRIES or s is None or len(s) == 0:
        return s
    out = s.astype(float).copy()
    origin = pd.Timestamp(origin)
    forecast = out.index[out.index > origin]
    if not len(forecast):
        return out
    by = {}
    for m in forecast:
        by.setdefault(period_year(cc, m), []).append(m)
    yrs = years or sorted(by)
    base = base_year(cc, origin)
    # Open period at origin: scale its annual by the share of months still ahead.
    open_py = period_year(cc, origin)
    if open_py in by:
        n_left = len(by[open_py])
        first_share = n_left / 12.0
    else:
        first_share = 1.0
    eff = annual_effects(cc, yrs, base, first_share=first_share)
    if not eff:
        return out
    for y, months in by.items():
        v = eff.get(int(y), 0.0)
        if abs(v) < 1e-12 or not months:
            continue
        add = v / len(months)
        for m in months:
            out.loc[m] = float(out.loc[m]) + add
    return out
