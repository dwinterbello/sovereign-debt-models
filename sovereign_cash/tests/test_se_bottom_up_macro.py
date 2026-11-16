"""Sweden v2 macro bottom-up: identity, PRE gate, no-lookahead, production lock."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.se import bottom_up_macro as BUM
from sovereign_cash.se import build as B
from sovereign_cash.se import model as M

MONTHLY = contract.DERIVED / "se_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "se_cash_summary.json"
BU_SCORE = BUM.OUT_SCORE


@pytest.fixture(scope="module")
def panels():
    d = B.load()
    scb = BUM.fetch_scb_budget()
    cats = BUM.categories(d, scb)
    y = cats["budget_balance"]
    return d, cats, y, scb


def test_scb_ch1089_identity(panels):
    _, cats, _, _ = panels
    assert BUM.identity_resid_bn(cats) < 0.1  # SCB vs CH1089: one month 76 mn, mean ~7 mn


def test_macro_anchored_pre_gate_no_ship(panels):
    d, cats, y, _ = panels
    bt = M.backtest(y)
    chosen, _ = M.select(bt)
    rgm = d["rg"].set_index("month")["rg_forecast_bn"].dropna()
    abt = M.rg_anchor_backtest(y, rgm, chosen)
    method, _ = M.select_allocation(abt)
    report = BUM.build_report(d, y, chosen, method, write=False)
    assert report["ship"] is False
    assert "does not beat" in report["ship_reason"] or "incomplete" in report["ship_reason"]
    pre = report["pre_mean_mae_anchored"]
    assert pre["bottom_up"] >= (1.0 - BUM.EDGE) * pre["production"] - 1e-9
    assert "tax_rev" in report["category_drivers"]
    assert "interest" in report["category_drivers"]
    assert report["category_drivers"]["net_lending"].get("driver", "").startswith("None")


def test_no_lookahead(panels):
    d, _, _, _ = panels
    for asof in (pd.Timestamp("2016-06-01"), pd.Timestamp("2018-03-01"),
                 d["nbr"]["month"].max() - pd.DateOffset(months=2)):
        assert BUM.no_lookahead_ok(d, asof)


def test_authority_budgetprop_present():
    auth = BUM.load_authority()
    assert "budgetprop" in auth
    assert "2026" in auth["budgetprop"]["years"]
    assert "tax_rev_bn" in auth["budgetprop"]["years"]["2026"]
    assert "ki" in auth
    assert auth["ki"]["years"]["2026"]["budget_balance_bn"] < 0
    assert "statskontoret" in auth


@pytest.mark.skipif(not MONTHLY.exists() or not SUMMARY.exists(), reason="run sovereign_cash.se.run first")
def test_derived_macro_column_when_built():
    s = json.loads(SUMMARY.read_text())
    if "bottom_up_macro" not in s:
        pytest.skip("summary predates macro wire")
    assert s["bottom_up_macro"]["ship"] is False
    df = pd.read_parquet(MONTHLY)
    assert "cash_requirement_bottomup_macro_bn" in df.columns
    h = df[~df["is_forecast"]]
    np.testing.assert_allclose(
        h["cash_requirement_bottomup_macro_bn"], h["cash_requirement_bn"], atol=1e-9)
