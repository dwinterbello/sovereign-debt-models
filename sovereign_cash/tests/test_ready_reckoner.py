"""The ready reckoner is the published coefficient, including the lag and the debt-interest triangle."""
from __future__ import annotations

import pytest

from sovereign_cash.uk.ready_reckoner import psnb, run, unit


def _m(name: str, year: str, line: str, fy: str) -> float:
    d = run({name: {year: 1.0}})
    hit = d[(d["line"] == line) & (d["effect_fy"] == fy) & (d["determinant"] == name)]
    return float(hit["gbp_m"].sum())


def test_earnings_hits_paye_and_nics_in_the_same_year():
    assert _m("Earnings level", "2025-26", "PAYE income tax", "2025-26") == pytest.approx(3470.377956, abs=1e-3)
    assert _m("Earnings level", "2025-26", "NICs", "2025-26") == pytest.approx(2615.387625, abs=1e-3)
    assert _m("Earnings level", "2025-26", "PAYE income tax", "2024-25") == 0


def test_mixed_income_is_lagged_one_year():
    assert _m("Mixed income", "2024-25", "SA income tax", "2024-25") == 0
    assert _m("Mixed income", "2024-25", "SA income tax", "2025-26") == pytest.approx(372.909747, abs=1e-3)


def test_gilt_rate_shock_persists_through_the_triangle():
    assert _m("Long-term interest rates", "2025-26", "Debt interest", "2025-26") == pytest.approx(1466.785848, abs=1e-3)
    assert _m("Long-term interest rates", "2025-26", "Debt interest", "2026-27") == pytest.approx(2708.497428, abs=1e-3)
    # the financing loop is the interest total, in £bn, times the CGNCR row, summed over shock years
    loop = run({"Long-term interest rates": {"2025-26": 1.0}})
    loop = loop[loop["determinant"] == "CGNCR financing loop"]
    assert float(loop.loc[loop["effect_fy"] == "2025-26", "gbp_m"].sum()) == pytest.approx(175.338, abs=0.01)
    assert float(psnb(run(unit("Long-term interest rates", ("2025-26",))))["2029-30"]) > 0


def test_unknown_determinant_is_rejected():
    with pytest.raises(KeyError):
        run({"not a determinant": {"2026-27": 1.0}})
