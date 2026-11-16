"""France authority = DG Trésor RESF + PLF solde + AFT charge / triangle — not AMECO/WEO."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sovereign_cash.interest_split import annual_interest, rates_shock_delta
from sovereign_cash.macro_authority import annual_authority

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data" / "fr" / "plan.json"
AUTH = ROOT / "data" / "fr" / "authority.json"


def test_fr_macro_is_resf_not_ameco():
    ngdp, cpi, src = annual_authority("fr")
    assert "RESF" in src["cpi"] or "Trésor" in src["cpi"]
    # RESF PLF 2027 vintage
    assert cpi.loc[2026] == pytest.approx(2.1)
    assert cpi.loc[2027] == pytest.approx(1.8)
    assert ngdp.loc[2026] == pytest.approx(0.5 + 2.1)
    assert ngdp.loc[2027] == pytest.approx(1.0 + 1.8)


def test_fr_plan_deficit_is_plf_solde_not_stale_lfi():
    """PLF 2027 revises 2026 to 137.2 and sets 2027 déficit à financer at 156.4."""
    years = json.loads(PLAN.read_text())["years"]
    assert years["2026"]["deficit_to_finance"] == pytest.approx(137.2)
    assert years["2027"]["deficit_to_finance"] == pytest.approx(156.4)
    auth = json.loads(AUTH.read_text())["state_deficit"]["years"]
    assert auth["2026"]["bn"] == pytest.approx(137.2)
    assert auth["2027"]["bn"] == pytest.approx(156.4)
    # AFT besoin identity: besoin ≈ deficit + amortisation + other needs
    y27 = years["2027"]
    other = y27["besoin"] - y27["amortisation"] - y27["deficit_to_finance"]
    assert abs(other) < 20  # residual treasury needs / resources, not a missing deficit


def test_fr_interest_is_aft_charge():
    i = annual_interest("fr")
    assert i.loc[2026] == pytest.approx(62.6)
    assert i.loc[2027] == pytest.approx(72.9)


def test_fr_rates_shock_matches_aft_year1():
    import pandas as pd

    from sovereign_cash.interest_split import _aft_rate_triangle
    # a shock starting in January: the k-th calendar year is the triangle's year k
    d = _aft_rate_triangle(100.0, pd.Timestamp("2027-01-01"))
    assert d[2027] == pytest.approx(3.2)
    assert d[2031] == pytest.approx(23.5)
    # starting in November: two months of year 1 land in the first calendar year
    e = _aft_rate_triangle(100.0, pd.Timestamp("2026-11-01"))
    assert e[2026] == pytest.approx(3.2 * 2 / 12)
    assert rates_shock_delta("fr", 50.0)  # the public entry point still answers
