"""Norway bottom-up cash_requirement baseline: PRE gate, no-lookahead, production lock."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.no import bottom_up as BU
from sovereign_cash.no import bottom_up_macro as BUM
from sovereign_cash.no import build as B
from sovereign_cash.no import fetch

MONTHLY = contract.DERIVED / "no_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "no_cash_summary.json"


@pytest.fixture(scope="module")
def panels():
    fl = B.daily_flows(fetch.read("liquidity_statistics"))
    vin = B.vintages(fetch.read("liquidity_forecast"))
    m = BU.monthly_tab4(fl)
    return fl, vin, m


def test_tab4_sums_to_gov_accounts(panels):
    _, _, m = panels
    gap = (m[B.TAB4].sum(axis=1) - m["gov_accounts"]).abs()
    assert float(gap.max()) < 0.01


def test_select_rules_and_pre_gate(panels):
    fl, vin, m = panels
    rules = BU.select_rules(m, vin)
    assert set(rules) == set(B.TAB4)
    assert all(r in BU.RULE_NAMES for r in rules.values())
    pre = BU.score_beyond_nb(m, vin, rules, "PRE")
    assert set(pre) == set(BU.HORIZONS)
    ship, reason = BU.ship_decision(pre)
    # smly is the beyond-NB ceiling on PRE (January-only sample)
    assert ship is False
    assert "does not beat" in reason or "incomplete" in reason
    assert BU.mean_mae(pre, "bottom_up") >= (1.0 - BU.EDGE) * BU.mean_mae(pre, "production_smly") - 1e-9


def test_no_lookahead(panels):
    _, vin, m = panels
    rules = BU.select_rules(m, vin)
    for asof in (pd.Timestamp("2018-06-01"), pd.Timestamp("2024-03-01"), m.index.max() - pd.DateOffset(months=2)):
        assert BU.no_lookahead_ok(m, asof, rules)


def test_hybrid_ties_nb_inside_path(panels):
    fl, vin, _ = panels
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
    assert report["ship"] is False


@pytest.mark.skipif(not MONTHLY.exists() or not SUMMARY.exists(), reason="run sovereign_cash.no.run first")
def test_derived_nb_pin_and_ship_path():
    """v1 Tab4 stays NO-SHIP; live build uses v2 macro (may SHIP beyond-NB)."""
    s = json.loads(SUMMARY.read_text())
    assert "bottom_up" in s
    df = pd.read_parquet(MONTHLY)
    assert "cash_requirement_bottomup_bn" in df.columns
    f = df[df["is_forecast"]]
    nb = f[~f["scenario"]]
    sc = f[f["scenario"]]
    assert nb["fc_source"].str.startswith("norges_bank").all()
    np.testing.assert_allclose(nb["cash_requirement_bottomup_bn"], nb["cash_requirement_bn"])
    assert sc["cash_requirement_bottomup_bn"].notna().all()
    if s["bottom_up"].get("ship"):
        assert s["bottom_up"].get("version") == "v2_macro"
        assert (sc["fc_source"] == "bottom_up_macro").all()
        np.testing.assert_allclose(sc["cash_requirement_bn"], sc["cash_requirement_bottomup_bn"])
    else:
        assert sc["fc_source"].str.startswith("same_month").all()
    assert BUM.OUT_SCORE.exists()
