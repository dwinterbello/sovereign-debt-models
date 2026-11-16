"""Monthly SARIMAX component path: MoM rejected; IT level+exog clears PRE."""
from __future__ import annotations

import pandas as pd
import pytest

from sovereign_cash import monthly_components as mc


@pytest.fixture(scope="module")
def fr_mom():
    return mc.mae(mc.backtest("fr", step=6, mode="mom", with_exog=False), "fr")


@pytest.fixture(scope="module")
def it_level():
    return mc.mae(mc.backtest("it", step=6, mode="level", with_exog=True), "it")


def test_fr_mom_loses_pre_h12(fr_mom):
    pre = fr_mom[(fr_mom["window"] == "PRE") & (fr_mom["h"] == 12)]
    assert not pre.empty
    r = pre.iloc[0]
    assert r["sarimax"] > r["smly"]


def test_it_level_exog_beats_smly_pre_h12(it_level):
    pre = it_level[(it_level["window"] == "PRE") & (it_level["h"] == 12)]
    assert not pre.empty
    r = pre.iloc[0]
    assert r["sarimax"] < r["smly"] - 1.0  # clear PRE win (~27 vs ~42 at step=6)


def test_live_path_it_returns_twelve_months():
    y = mc.components("it")
    origin = y.index[max(48, len(y) - 24)]
    path = mc.live_path("it", origin, 12, mode="level", with_exog=True)
    assert path is not None
    assert len(path) == 12
    assert (path.index[1] - path.index[0]).days in range(28, 32)


def test_label_covers_sx():
    from sovereign_cash.it import model as IM
    assert "SARIMAX" in IM.label("sx_level_exog")
