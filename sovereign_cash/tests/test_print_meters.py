"""Accrued print meters — UK PSNB + ES B.9. Offline on local caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt.note import print_meters as P

UK_PROFILE = Path(__file__).resolve().parents[1] / "data" / "uk" / "obr_psnb_profile_ytd.json"


def test_uk_psnb_sign_and_fytd_matches_bulletin_order():
    y = P.uk_psnb_monthly()
    assert y.loc["2026-09-01"] == pytest.approx(24.52, abs=0.05)
    ytd = float(y.loc["2026-04-01":"2026-09-01"].sum())
    assert ytd == pytest.approx(97.8, abs=0.2)


def test_uk_profile_stub_rescales_ytd_to_obr_bulletin():
    assert UK_PROFILE.exists()
    stub = json.loads(UK_PROFILE.read_text())
    path = P.uk_plan_monthly_path(pd.Timestamp("2026-09-01"))
    done = path.loc[:"2026-09-01"]
    assert float(done.sum()) == pytest.approx(float(stub["ytd_profile_bn"]), abs=0.05)
    assert float(path.sum()) == pytest.approx(float(stub["annual_psnb_bn"]), abs=0.05)


def test_uk_ship_gate_documented():
    score = P.score_uk_h1()
    if score.empty:
        pytest.skip("PSNB history too thin for PRE score")
    ship = P.uk_ship_model(score)
    assert ship["model"] in ("seas_add_3", "profile")
    assert "mae_profile" in ship
    # Live state must expose official gap from stub when through matches.
    live = P.uk_live_state()
    assert live["through"] == "2026-09"
    assert live["official_gap_bn"] == pytest.approx(6.0, abs=0.05)


def test_es_b9_deficit_positive_and_seas_scores():
    y = P.es_b9_deficit_monthly()
    assert y.min() > -50  # mostly deficit months; allow surplus months
    assert y.loc["2026-07-01"] == pytest.approx(22.143, abs=0.05)
    score = P.score_es_h1()
    if score.empty:
        pytest.skip("ES IGAE too thin")
    ship = P.es_ship_model(score)
    assert ship["model"] in ("seas_add_3", "smly")
    live = P.es_live_state()
    assert live["ytd_bn"] > 0
    assert live["next_bn"] is not None


def test_spike_fr_it_de_rejects_monthly_gg():
    findings = P.spike_fr_it_de()
    assert findings["fr"]["usable_monthly_gg"] is False
    assert findings["it"]["usable_monthly_gg"] is False
    assert findings["de"]["usable_monthly_gg"] is False
    assert P.SPIKE_PATH.exists()


def test_render_writes_png():
    png, cap, prose = P.render(out=P.NOTE / "_test_print_meters.png")
    assert png.exists() and png.stat().st_size > 10_000
    assert "Exhibit 1b" in cap
    assert "PSNB" in prose and "B.9" in prose
    png.unlink(missing_ok=True)
