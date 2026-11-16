"""Sweden bottom-up NBR baseline: identity, PRE gate, no-lookahead, production lock."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.se import bottom_up as BU
from sovereign_cash.se import build as B
from sovereign_cash.se import model as M

MONTHLY = contract.DERIVED / "se_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "se_cash_summary.json"
BU_SCORE = BU.OUT_SCORE


@pytest.fixture(scope="module")
def panels():
    d = B.load()
    comps = BU.components_bn(d["nbr"])
    y = comps["budget_balance"]
    return d, comps, y


def test_ch1089_identity(panels):
    _, comps, _ = panels
    assert BU.identity_resid_bn(comps) < 0.002  # ≤ 1 mn SEK → 0.001 bn


def test_select_and_anchored_pre_gate(panels):
    d, comps, y = panels
    selected = BU.select_component_params(comps)
    assert set(selected) == set(BU.COMPONENTS)
    params = BU.params_tuple(selected)
    assert selected["nbr_primary"][0] == "smly"
    bt = M.backtest(y)
    chosen, _ = M.select(bt)
    rgm = d["rg"].set_index("month")["rg_forecast_bn"].dropna()
    abt = M.rg_anchor_backtest(y, rgm, chosen)
    method, _ = M.select_allocation(abt)
    an = BU.score_anchored(y, comps, params, chosen, method, rgm)
    ship, reason = BU.ship_decision(an.get("PRE", {}))
    assert ship is False
    assert "does not beat" in reason or "incomplete" in reason
    pre = an["PRE"]
    assert BU.mean_mae(pre, "bottom_up") >= (1.0 - BU.EDGE) * BU.mean_mae(pre, "production") - 1e-9


def test_unanchored_beats_production_on_pre(panels):
    """Component skill is real; it is not the live ship gate."""
    d, comps, y = panels
    params = BU.params_tuple(BU.select_component_params(comps))
    chosen, _ = M.select(M.backtest(y))
    un = BU.score_unanchored(y, comps, params, chosen)
    bu = BU.mean_mae(un["PRE"], "bottom_up")
    prod = BU.mean_mae(un["PRE"], "production")
    assert bu < prod
    worse = sum(1 for h in BU.SCORE_H if un["PRE"][h]["bottom_up"] > un["PRE"][h]["production"])
    assert worse <= 1


def test_no_lookahead(panels):
    _, comps, _ = panels
    params = BU.params_tuple(BU.select_component_params(comps))
    for asof in (pd.Timestamp("2015-06-01"), pd.Timestamp("2018-03-01"),
                 comps["nbr"].index.max() - pd.DateOffset(months=2)):
        assert BU.no_lookahead_ok(comps, params, asof)


@pytest.mark.skipif(not MONTHLY.exists() or not SUMMARY.exists(), reason="run sovereign_cash.se.run first")
def test_derived_production_unchanged_when_no_ship():
    s = json.loads(SUMMARY.read_text())
    assert "bottom_up" in s
    assert s["bottom_up"]["ship"] is False
    df = pd.read_parquet(MONTHLY)
    assert "cash_requirement_bottomup_bn" in df.columns
    h = df[~df["is_forecast"]]
    np.testing.assert_allclose(h["cash_requirement_bottomup_bn"], h["cash_requirement_bn"], atol=1e-9)
    f = df[df["is_forecast"]]
    assert f["cash_requirement_bottomup_bn"].notna().all()
    # live path still RG-anchored seasonal (budget_balance_model_bn present and cash ≠ pure bottom-up in general)
    assert "budget_balance_model_bn" in f.columns
    assert BU_SCORE.exists()
    assert any("Bottom-up CH1089" in n and "NO-SHIP" in n for n in s["notes"])
    assert all("_bn" not in n for n in s["notes"] if "Bottom-up" in n)
