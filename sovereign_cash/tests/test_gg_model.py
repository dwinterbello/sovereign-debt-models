"""GG deficit model — identity, PRE peer gate, live path shape."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sovereign_cash.debt import gg_model as GG

DATA = Path(__file__).resolve().parents[1] / "data" / "debt"
HAS = (DATA / "ameco.parquet").exists() and (DATA / "weo.parquet").exists()


@pytest.mark.skipif(not HAS, reason="AMECO/WEO cache missing")
def test_history_identity():
    ameco, weo = GG._load()
    hist = GG.history("fr", ameco, weo)
    assert len(hist) > 20
    # primary = balance + interest; structural = primary − cycle
    assert (hist["primary"] - (hist["balance"] + hist["interest"])).abs().max() < 1e-9
    assert (hist["structural_primary"] - (hist["primary"] - hist["cycle"])).abs().max() < 1e-9


@pytest.mark.skipif(not HAS, reason="AMECO/WEO cache missing")
def test_forecast_holds_structural():
    ameco, weo = GG._load()
    hist = GG.history("fr", ameco, weo)
    W = GG._pivot(weo, "fr")
    origin = 2022
    fc = GG.forecast("fr", hist.loc[:origin], W, origin, h=3)
    assert len(fc) == 3
    sp0 = float(hist.loc[origin, "structural_primary"])
    assert (fc["structural_primary"] - sp0).abs().max() < 1e-9
    # balance identity
    assert (fc["balance"] - (fc["structural_primary"] + fc["cycle"] - fc["interest"])).abs().max() < 1e-9


@pytest.mark.skipif(not HAS, reason="AMECO/WEO cache missing")
def test_run_writes_and_scores():
    s = GG.run(write=True)
    assert set(s["countries"]) >= set(GG.COUNTRIES)
    out = Path(__file__).resolve().parents[1] / "derived"
    assert (out / "gg_forecast.parquet").exists()
    assert (out / "gg_model_summary.json").exists()
    live = pd.read_parquet(out / "gg_forecast.parquet")
    assert set(live["country"].unique()) >= set(GG.COUNTRIES)
    # FR live should have a 2027 row near the OE neighbourhood if origin ≤ 2025
    fr = live[live["country"] == "fr"]
    assert not fr.empty
    # summary JSON round-trip
    j = json.loads((out / "gg_model_summary.json").read_text())
    assert "fr" in j["countries"]
    assert j["countries"]["fr"]["mae"] is not None
