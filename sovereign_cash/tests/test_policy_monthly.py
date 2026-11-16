"""Legislated policy must land on monthly cash paths, not only the scenario pack."""
from __future__ import annotations

import pandas as pd
import pytest

from sovereign_cash.policy import monthly as M
from sovereign_cash.scenarios.channels.sarimax import later_years


def test_add_to_monthly_fr_2027_moves_the_year():
    months = pd.date_range("2027-01-01", periods=12, freq="MS")
    bare = pd.Series(10.0, index=months)
    out = M.add_to_monthly("fr", bare, pd.Timestamp("2026-09-01"))
    assert float(out.sum()) == pytest.approx(120.0 + float(out.sum() - bare.sum()), abs=1e-9)
    assert abs(float(out.sum()) - 120.0) > 5.0  # LPM / surcharge package


def test_later_years_fr_includes_policy():
    months = pd.date_range("2026-10-01", periods=27, freq="MS")
    s = later_years("fr", months, open_year=2026)
    assert len(s) >= 12
    y27 = s[s.index.year == 2027]
    assert len(y27) == 12
    # With policy the year is above the old no-policy grown fantasy (~139).
    assert float(y27.sum()) > 150.0
