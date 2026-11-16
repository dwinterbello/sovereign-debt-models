"""UK IT+NIC frozen-threshold fiscal-drag elasticity (OBR EFO Oct 2026 para 4.14)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels import own_annual as OA


def test_eps_switches_after_prior_law_freeze():
    assert OA.itnic_eps(2020) == 1.0  # pre-freeze: unit (rule ≈ GDP share when earnings track NGDP)
    assert OA.itnic_eps(2021) == OA.EPS_FROZEN
    assert OA.itnic_eps(2027) == OA.EPS_FROZEN
    assert OA.itnic_eps(2028) == 1.0  # Budget 2025 extension is policy, not here
    assert abs(OA.EPS_FROZEN - 1.3) < 1e-12
    assert abs(OA.EPS_INDEXED - 1.3 * 4.5 / 7.0) < 1e-12  # OBR para 4.14 ratio, documented


def test_fiscal_drag_beats_unit_when_earnings_rise_under_freeze():
    """With frozen ε>1 and flat GDP, rising earnings raise IT+NIC receipts above the unit share."""
    T = 2025
    years = [2026, 2027]
    Y = pd.Series({2025: 1000.0, 2026: 1000.0, 2027: 1000.0})
    E = pd.Series({2025: 1.0, 2026: 1.01, 2027: 1.01 ** 2})
    got = OA.receipts_fiscal_drag(400.0, T, Y, E, years)
    unit = [400.0, 400.0]  # GDP flat → unit share flat
    assert got[0] > unit[0]
    assert got[1] > got[0]


def test_backtest_scores_and_ships_fiscal_drag_for_uk():
    r = OA.backtest("uk")
    sc = r["lines"]["receipts"]["scores"]
    assert "fiscal_drag" in sc
    assert all(np.isfinite(sc["fiscal_drag"][f"h{k}"]) for k in (1, 2, 3))
    assert r["lines"]["receipts"]["chosen"] == "fiscal_drag"
    meta = r["lines"]["receipts"]["fiscal_drag"]
    assert meta["eps_frozen"] == 1.3
    assert meta["freeze_last_prior_law"] == 2027
    assert meta.get("shipped", "").startswith("structural")
    assert OA.method("uk", "receipts") == "fiscal_drag"


def test_forecast_applies_drag_and_skips_extension_years():
    fc, T, info = OA.forecast("uk", [2026, 2027, 2028, 2029])
    assert info["methods"]["receipts"] == "fiscal_drag"
    assert info["fiscal_drag"]["eps_by_year"]["2026-27"] == OA.EPS_FROZEN
    assert info["fiscal_drag"]["eps_by_year"]["2027-28"] == OA.EPS_FROZEN
    assert info["fiscal_drag"]["eps_by_year"]["2028-29"] == 1.0
    assert info["fiscal_drag"]["eps_by_year"]["2029-30"] == 1.0
    # vs the same rule with thresholds indexed (ε = 1): the freeze adds receipts in the freeze years. (Against the
    # plain GDP share it need not: nominal GDP runs on our CPI path, earnings on the OBR's.)
    h = OA.history("uk")
    yrs = [2026, 2027, 2028, 2029]
    Y, _ = OA.levels("uk", list(h.index) + yrs)
    E = OA.earnings_index(list(h.index) + yrs)
    drag = OA.receipts_fiscal_drag(float(h.at[T, "receipts"]), T, Y, E, yrs)
    eps, OA.EPS_FROZEN = OA.EPS_FROZEN, 1.0
    try:
        flat = OA.receipts_fiscal_drag(float(h.at[T, "receipts"]), T, Y, E, yrs)
    finally:
        OA.EPS_FROZEN = eps
    for i in (0, 1):
        assert drag[i] > flat[i]
