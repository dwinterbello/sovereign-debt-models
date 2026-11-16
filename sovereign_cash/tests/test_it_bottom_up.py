"""Italy bottom-up fabbisogno baseline — PRE gate, identity, no-lookahead, production lock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.it import bottom_up as BU
from sovereign_cash.it import model as M
from sovereign_cash.it import run as R


@pytest.fixture(scope="module")
def bt():
    return BU.score()


@pytest.fixture(scope="module")
def verdict(bt):
    return BU.ship_rule(bt)


@pytest.fixture(scope="module")
def mae(bt):
    return BU.mae_table(bt)


def test_ship_rule_matches_pre_and_is_no_ship(verdict, mae):
    pre = mae[mae["window"] == "PRE"].set_index("h")
    assert set(pre.index) >= set(M.HORIZONS)
    bu = float(pre.loc[list(M.HORIZONS), "bottomup"].mean())
    prod = float(pre.loc[list(M.HORIZONS), "production"].mean())
    assert bu == pytest.approx(verdict["pre_mean_bottomup"], rel=1e-6)
    assert prod == pytest.approx(verdict["pre_mean_production"], rel=1e-6)
    worse = sum(1 for h in M.HORIZONS if pre.at[h, "bottomup"] > pre.at[h, "production"] + 1e-12)
    edge = (prod - bu) / prod
    oos_ok = verdict["oos_ok"]
    expect = edge >= BU.REL_EDGE and worse <= 1 and oos_ok
    assert verdict["ship"] is expect
    # Ceiling: production plan×smly (smly on PRE) is not beaten by 5%.
    assert verdict["ship"] is False
    assert bu >= (1.0 - BU.REL_EDGE) * prod - 1e-9


def test_oos_reported(verdict, mae):
    oos = mae[mae["window"] == "OOS"]
    assert not oos.empty
    assert verdict["oos_mean_bottomup"] == pytest.approx(
        float(oos.set_index("h").loc[list(M.HORIZONS), "bottomup"].mean()), rel=1e-6)


def test_forecast_no_lookahead():
    frame = BU._frame_bn()
    x = BU._mef_rt_mn()
    origin = pd.Timestamp("2018-06-01")
    assert origin in frame.index and origin in x.index
    fc1 = BU.forecast(origin, 12, x, frame, plans_bn={}, soft_pin=False)
    assert fc1 is not None
    frame2 = frame.copy()
    x2 = x.copy()
    future = origin + pd.DateOffset(months=1)
    frame2.loc[future:] = frame2.loc[future:] + 5.0
    x2.loc[future:] = x2.loc[future:] + 5_000.0
    fc2 = BU.forecast(origin, 12, x2, frame2, plans_bn={}, soft_pin=False)
    assert fc2 is not None
    assert np.allclose(fc1.to_numpy(), fc2.to_numpy(), rtol=0, atol=1e-9)


def test_identity_closes_on_components():
    frame = BU._frame_bn()
    ident = BU._identity_bn(frame)
    rebuilt = (frame["primary"] + frame["interest"] + frame["cap_exp"]
               - frame["tax_rev"] - frame["other_rev"] - frame["treasury_bal"])
    assert (ident - rebuilt).abs().max() < 1e-9


def test_apply_column_and_production_lock(verdict):
    # Live path uses macro_v2 (SHIP); v1 remains NO-SHIP for comparison only.
    from sovereign_cash.it import bottom_up_macro as BUm
    inp = R.load_inputs(refresh=False)
    hist, asof = R.history(inp)
    x = inp["mef"]["fabb_mn"].loc["2010-01":asof].asfreq("MS")
    bt, sc, chosen = R.run_backtest(inp["mef"]["fabb_rt_mn"].loc["2010-01":asof].asfreq("MS"))
    fwd, info = R.forward(inp, hist, asof, x, chosen, bt, sc)
    assert "cash_requirement_bottomup_bn" in fwd.columns
    # FORCE_LIVE is off: macro_v2 reaches the board only when its own gate passes
    assert info.get("bottomup_engine") == "macro_v2"
    if not BUm.FORCE_LIVE and not info["bottomup_ship"]:
        assert not fwd["cash_requirement_bn"].equals(fwd["cash_requirement_bottomup_bn"])
    assert verdict["ship"] is False  # v1 still NO-SHIP
    live = fwd["cash_requirement_bn"].copy()
    out, inf = BU.apply_to_forward(fwd, x, asof, ship=False)
    assert inf["bottomup_ship"] is False
    assert np.allclose(out["cash_requirement_bn"], live)
    # Explicit v1 ship=False must not overwrite; macro ship set live == bottomup for the year in progress
    # (later years run on the component model, as the Shock scenarios page)
    open_yr = fwd.index.year == asof.year
    if info["bottomup_ship"]:
        assert np.allclose(fwd.loc[open_yr, "cash_requirement_bn"], fwd.loc[open_yr, "cash_requirement_bottomup_bn"])
    resid = (out["buffer_change_bn"]
             - (-out["cash_requirement_bn"] + out["net_bonds_bn"]
                + out["net_bills_bn"] + out["other_fin_bn"]))
    assert resid.abs().max() < 1e-6
    assert BUm.SHIP is True


def test_soft_plan_pin_moves_toward_plan():
    idx = pd.date_range("2026-09-01", periods=4, freq="MS")
    fc = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx)
    x = pd.Series(5.0, index=pd.date_range("2026-01-01", "2026-08-01", freq="MS"))
    # remaining plan for 2026 = 100 − 40 = 60 over 4 months → scale 60/40 = 1.5 at weight 1
    pinned = BU.soft_plan_pin(fc, x, pd.Timestamp("2026-08-01"), {2026: 100.0}, weight=1.0)
    assert pinned.sum() == pytest.approx(60.0)
    soft = BU.soft_plan_pin(fc, x, pd.Timestamp("2026-08-01"), {2026: 100.0}, weight=0.5)
    assert soft.sum() == pytest.approx(0.5 * 60.0 + 0.5 * 40.0)
