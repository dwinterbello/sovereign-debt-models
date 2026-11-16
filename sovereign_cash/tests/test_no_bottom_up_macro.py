"""Norway Tab4×macro bottom-up (v2): PRE gate, NB pin, no-lookahead."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.no import bottom_up_macro as BU
from sovereign_cash.no import build as B
from sovereign_cash.no import fetch

MONTHLY = contract.DERIVED / "no_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "no_cash_summary.json"
BU_SCORE = BU.OUT_SCORE


@pytest.fixture(scope="module")
def panels():
    fl = B.daily_flows(fetch.read("liquidity_statistics"))
    vin = B.vintages(fetch.read("liquidity_forecast"))
    m = BU.monthly_tab4(fl)
    drivers = BU.macro_drivers()
    rules = BU.select_use_macro(m, drivers)
    return fl, vin, m, drivers, rules


def test_tab4_identity(panels):
    _, _, m, _, _ = panels
    gap = (m[B.TAB4].sum(axis=1) - m["gov_accounts"]).abs()
    assert float(gap.max()) < 0.01


def test_petroleum_unit_and_mainland_rules(panels):
    _, _, _, _, rules = panels
    assert rules["oil_tax"] == "unit_mult"
    assert rules["sdfi"] == "unit_mult"
    assert set(rules) == set(B.TAB4)
    assert all(r in ("smly", "unit_mult", "add_expanding") for r in rules.values())


def test_pre_gate_and_category_table(panels):
    fl, vin, m, drivers, rules = panels
    pre = BU.score_beyond_nb(m, vin, drivers, rules, "PRE")
    assert set(pre) == set(BU.HORIZONS)
    ship, reason = BU.ship_decision(pre)
    assert ship, reason
    bu = BU.mean_mae(pre, "bottom_up")
    prod = BU.mean_mae(pre, "production_smly")
    assert bu < (1.0 - BU.EDGE) * prod
    assert 0 < BU.MAINLAND_RESID_WEIGHT <= 1.0
    cats = BU.category_table(m, drivers, rules)
    assert len(cats) == len(B.TAB4)
    assert all("driver" in r and "rule" in r for r in cats)
    assert reason


def test_no_lookahead(panels):
    _, _, m, drivers, rules = panels
    for asof in (pd.Timestamp("2018-06-01"), pd.Timestamp("2024-03-01"),
                 m.index.max() - pd.DateOffset(months=2)):
        assert BU.no_lookahead_ok(m, drivers, asof, rules)


def test_hybrid_ties_nb_inside_path(panels):
    fl, vin, _, _, _ = panels
    S = B.structural_realised(fetch.read("liquidity_forecast"))
    hist = B.history(fl)
    last_obs = fl.index.max()
    if last_obs < last_obs + pd.offsets.BMonthEnd(0):
        pm = last_obs.to_period("M").to_timestamp()
        base = B.partial_month(fl, S, pm)
    else:
        base = hist
    fw, meta = B.forward(fl, base, vin, S, fetch.read("fx_spu"), fetch.read("govt_primary"))
    fw2, report = BU.attach_bottomup(fw, fl, vin, meta)
    nb = fw2[~fw2["scenario"]]
    np.testing.assert_allclose(nb["cash_requirement_bottomup_bn"], nb["cash_requirement_bn"], atol=1e-9)
    assert "version" in report and report["version"] == "v2_macro"


@pytest.mark.skipif(not BU_SCORE.exists(), reason="run bottom_up_macro / no.run first")
def test_score_json_schema():
    s = json.loads(BU_SCORE.read_text())
    assert "category_driver_table" in s
    assert "pre_mean_mae" in s and "oos_mean_mae" in s
    assert "rejects" in s
