"""UK bottom-up CGNCR baseline: identity, no-lookahead, PRE ship gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.uk import bottom_up as BU
from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

MONTHLY = contract.DERIVED / "uk_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "uk_cash_summary.json"

_COMP_KEYS = ("cg_current_receipts", "cg_current_exp", "cg_capital_exp", "ps_interest_paid", "cgncr_ex")


def _ons_ready() -> bool:
    ons = src.ons_monthly()
    return all(k in ons.columns for k in _COMP_KEYS)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete; run uk.run --refresh")
def test_component_identity_holds():
    c = BU.components()
    np.testing.assert_allclose(c["cgncr"], c["bridge"] + c["resid"], atol=1e-9)
    np.testing.assert_allclose(c["primary"] + c["interest"], c["curr"], atol=1e-9)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_forecast_no_lookahead():
    c = BU.components()
    origin = c.index[-13]
    base = BU.forecast(c, origin)
    poisoned = c.copy()
    poisoned.iloc[-12:] = 1e6
    np.testing.assert_allclose(base, BU.forecast(poisoned, origin))


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_ship_rule_uses_pre_only():
    """OOS improvement must not flip the gate when PRE loses."""
    bt = BU.backtest()
    pre_bu = BU.mean_mae(bt, "PRE", BU.BOTTOMUP)
    pre_prod = BU.mean_mae(bt, "PRE", BU.PRODUCTION)
    # Fabricate: PRE worse, OOS much better → must NO-SHIP
    fake = bt.copy()
    m = (fake["window"] == "PRE") & (fake["model"] == BU.BOTTOMUP)
    fake.loc[m, "abs_err"] = fake.loc[m, "abs_err"] + 50.0
    d = BU.ship_rule(fake)
    assert d["ship"] is False
    # Real decision is PRE-consistent with means
    real = BU.ship_rule(bt)
    bu_mean = float(np.mean([pre_bu[h] for h in M.HORIZONS]))
    prod_mean = float(np.mean([pre_prod[h] for h in M.HORIZONS]))
    edge = (prod_mean - bu_mean) / prod_mean
    worse = sum(1 for h in M.HORIZONS if pre_bu[h] > pre_prod[h])
    assert real["ship"] == (edge > M.EDGE_BAND and worse <= 1)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_locked_spec_beats_production_on_pre_or_documents_ceiling():
    d = BU.evaluate()
    assert d["spec"] == BU.SPEC
    assert set(d["bu_pre"]) >= set(M.HORIZONS)
    assert set(d["prod_pre"]) >= set(M.HORIZONS)
    # Gate math (v1 is rejected for live even if gate_pass): ship live only if SHIP and gate
    worse = d["worse_horizons"]
    gate = d.get("gate_pass", d["ship"])
    if gate:
        assert d["edge"] > M.EDGE_BAND
        assert len(worse) <= 1
    else:
        assert d["edge"] <= M.EDGE_BAND or len(worse) > 1
    assert d["ship"] is False  # v1 seasonal rejected; macro v2 is the candidate


@pytest.mark.skipif(not MONTHLY.exists() or not SUMMARY.exists(), reason="run uk.run first")
def test_bottomup_column_present_and_bills_still_close():
    df = pd.read_parquet(MONTHLY).set_index("month")
    s = json.loads(SUMMARY.read_text())
    fc = df[df["is_forecast"]]
    assert "cash_requirement_bottomup_bn" in fc.columns
    assert fc["cash_requirement_bottomup_bn"].notna().all()
    e = s["forward"]["end_march"]
    fy = int(e["fy"][:4])
    mar = pd.Timestamp(fy + 1, 3, 1)
    # Bills solved to remit end-March regardless of CGNCR source
    assert fc.loc[mar, "dmo_net_cash_bn"] == pytest.approx(e["net_cash_plan_end_march_bn"], abs=2e-3)
    ship = s["forward"].get("cgncr_bottomup_ship")
    if ship:
        np.testing.assert_allclose(fc["cash_requirement_bn"], fc["cash_requirement_bottomup_bn"], atol=1e-9)
        assert e["cgncr_source_in_fy"] == "bottom_up"
    else:
        # Production unchanged vs hybrid of model/plan columns
        hybrid = np.where(fc["cgncr_from_plan"], fc["cash_requirement_fc_official_bn"],
                          fc["cash_requirement_model_bn"])
        np.testing.assert_allclose(fc["cash_requirement_bn"], hybrid, atol=1e-9)
