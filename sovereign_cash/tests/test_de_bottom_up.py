"""Germany bottom-up cash-requirement baseline — PRE gate, identity, no-lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.de import bottom_up as BU
from sovereign_cash.core import cash_framework as C
from sovereign_cash.de import model as M


@pytest.fixture(scope="module")
def verdict():
    BU.ship_verdict.cache_clear()
    return BU.ship_verdict()


@pytest.fixture(scope="module")
def mae():
    return BU.mae_table()


def test_ship_rule_matches_pre_numbers(verdict, mae):
    pre = mae[mae["window"] == "PRE"].set_index("h")
    assert set(pre.index) >= set(C.HORIZONS)
    bu = float(pre.loc[list(C.HORIZONS), "bottomup"].mean())
    prod = float(pre.loc[list(C.HORIZONS), "production"].mean())
    assert bu == pytest.approx(verdict["pre_mean_bottomup"], rel=1e-6)
    assert prod == pytest.approx(verdict["pre_mean_production"], rel=1e-6)
    worse = sum(1 for h in C.HORIZONS if pre.at[h, "bottomup"] > pre.at[h, "production"] + 1e-12)
    edge = (prod - bu) / prod
    expect = edge >= BU.REL_EDGE and worse <= 1
    assert verdict["ship"] is expect


def test_bottomup_beats_planhc_on_pre(mae):
    pre = mae[mae["window"] == "PRE"].set_index("h")
    assert float(pre["bottomup"].mean()) < float(pre["planhc"].mean()) * (1 - BU.REL_EDGE)


def test_forecast_no_lookahead():
    exp, rev, cash, mb, maps = BU._flows()
    origin = pd.Timestamp("2018-06-01")
    assert origin in cash.index
    fc1 = BU.forecast_core(origin, 12, exp, rev, mb, maps)
    assert fc1 is not None
    exp2, rev2 = exp.copy(), rev.copy()
    exp2.loc["2019-01-01":] = exp2.loc["2019-01-01":] + 50.0
    rev2.loc["2019-01-01":] = rev2.loc["2019-01-01":] + 40.0
    fc2 = BU.forecast_core(origin, 12, exp2, rev2, mb, maps)
    assert fc2 is not None
    # months whose smly base is ≤ origin must match; open-year close uses only ≤ origin
    open_m = [m for m in fc1.index if m.year == origin.year]
    assert np.allclose(fc1.loc[open_m], fc2.loc[open_m])


def test_apply_identity_and_column(verdict):
    h, meta = M.build_history()
    cash = meta["cash_full"].dropna()
    hc = M.haircuts(cash, meta["core"], h["other_fin_gap_bn"])
    f, info = M.forward(h, cash, "planhc_3", "gap0", meta["core"], hc)
    assert "cash_requirement_bottomup_bn" in f.columns
    assert "cash_requirement_core_bottomup_bn" in f.columns
    assert info["bottomup_ship"] is verdict["ship"]
    resid = (f["buffer_change_bn"]
             - (-f["cash_requirement_bn"] + f["net_bonds_bn"] + f["net_bills_bn"] + f["other_fin_bn"]))
    assert resid.abs().max() < 1e-6
    # observed months stay on actuals
    obs = f[f["cash_observed"]]
    for m in obs.index:
        assert obs.at[m, "cash_requirement_core_bn"] == pytest.approx(float(cash[m]), abs=1e-9)
    if verdict["ship"]:
        fut = f[~f["cash_observed"]]
        assert np.allclose(fut["cash_requirement_core_bn"], fut["cash_requirement_core_bottomup_bn"])
        assert np.allclose(fut["cash_requirement_bn"], fut["cash_requirement_bottomup_bn"])


def test_production_unchanged_when_ship_forced_off():
    """ship=False keeps the incoming live core; bottom-up is alternate only."""
    h, meta = M.build_history()
    cash = meta["cash_full"].dropna()
    hc = M.haircuts(cash, meta["core"], h["other_fin_gap_bn"])
    f, _ = M.forward(h, cash, "planhc_3", "gap0", meta["core"], hc)
    # restore a frozen live core snapshot then re-apply with ship off
    live_core = f["cash_requirement_core_bn"].copy()
    live_tot = f["cash_requirement_bn"].copy()
    out, inf = BU.apply_to_forward(f, cash, meta["core"], cash.index.max(), ship=False)
    assert inf["bottomup_ship"] is False
    assert np.allclose(out["cash_requirement_core_bn"], live_core)
    assert np.allclose(out["cash_requirement_bn"], live_tot)
    assert "cash_requirement_bottomup_bn" in out.columns
