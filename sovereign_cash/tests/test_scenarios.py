"""Authority-driver scenarios: unit shocks move cash; baseline uses AMECO/WEO."""
from __future__ import annotations

import numpy as np
import pytest

from sovereign_cash import scenarios as SX
from sovereign_cash.macro_authority import annual_authority, driver_path


@pytest.fixture(scope="module")
def it_sc():
    return SX.run("it")


def test_authority_has_forward_years():
    ngdp, cpi, src = annual_authority("it")
    assert 2026 in ngdp.index or 2027 in ngdp.index
    assert 2026 in cpi.index
    assert "AMECO" in src["ngdp"] or "DFP" in src["ngdp"]
    assert "WEO" in src["cpi"]


def test_driver_path_shock_only_future():
    import pandas as pd
    origin = pd.Timestamp("2024-06-01")
    base = driver_path("fr", origin, 18)
    sh = driver_path("fr", origin, 18, shock={"ngdp": 1.0})
    # lagged series: compare future tails after lag
    b, s = base["ngdp_yoy"].dropna(), sh["ngdp_yoy"].dropna()
    common = b.index.intersection(s.index)
    # at least some months differ by ~1pp
    assert float((s.loc[common] - b.loc[common]).abs().max()) >= 0.5


def test_it_scenario_ok(it_sc):
    assert it_sc["ok"]
    assert "gdp" in {s["key"] for s in it_sc["shocks"]}
    assert "cpi" in {s["key"] for s in it_sc["shocks"]}


def test_gdp_shock_moves_cash(it_sc):
    gdp = next(s for s in it_sc["shocks"] if s["key"] == "gdp")
    deltas = [v for v in gdp["delta"].values() if np.isfinite(v)]
    assert deltas
    # receipt elasticity: higher NGDP → lower cash requirement (deficit)
    assert any(abs(v) > 0.05 for v in deltas)


def test_table_html_renders(it_sc):
    html = SX.table_html("it", it_sc)
    assert "Δ cash requirement" in html
    assert "data-sx=" in html


def test_de_scenarios_include_rates():
    p = SX.run("de")
    assert p["ok"]
    keys = {s["key"] for s in p["shocks"]}
    assert keys >= {"gdp", "cpi", "rates"}
    rates = next(s for s in p["shocks"] if s["key"] == "rates")
    assert any(v > 0 for v in rates["delta"].values())
