"""Italy v2 macro bottom-up — PRE gate, identity, no-lookahead, SHIP lock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.it import bottom_up_macro as BU
from sovereign_cash.it import model as M
from sovereign_cash.it import run as R

AUTH = Path(__file__).resolve().parents[1] / "data" / "it" / "authority.json"


@pytest.fixture(scope="module")
def bt():
    return BU.score(step=1)


@pytest.fixture(scope="module")
def verdict(bt):
    return BU.ship_rule(bt)


@pytest.fixture(scope="module")
def mae(bt):
    return BU.mae_table(bt)


def test_authority_json_has_dfp_categories():
    raw = json.loads(AUTH.read_text())
    assert "categories_cash_central" in raw
    years = raw["categories_cash_central"]["years"]
    assert "2026" in years and "tributari" in years["2026"]
    assert years["2026"]["interessi_passivi"] > 0
    assert raw["elasticities"]["tax_rev"]["driver"] == "ngdp"
    assert "clip" in raw["elasticities"]["tax_rev"]
    assert raw["elasticities"]["tax_rev"]["beta"] == pytest.approx(BU.BETA_TAX)
    assert raw["elasticities"]["primary"]["beta"] == pytest.approx(BU.BETA_PRIMARY)
    assert raw["elasticities"]["treasury_bal"]["scale"] == pytest.approx(BU.TREAS_SCALE)
    assert raw["elasticities"]["interest"]["driver"] is None
    assert raw.get("score_note", {}).get("ship") is True
    ids = {v["id"] for v in raw["vintages"]}
    assert ids >= {"DEF_2021", "DEF_2024", "DFP_2025", "DFP_2026"}
    def21 = next(v for v in raw["vintages"] if v["id"] == "DEF_2021")
    assert "2022" in def21["years"] and def21["years"]["2022"]["tributari"] > 1e5
    assert (AUTH.parent / "plan" / "DEF21-Sez-II.pdf").exists()
    assert (AUTH.parent / "plan" / "Tavole_DFP-2025.xlsx").exists()
    assert (AUTH.parent / "tax" / "Base_dati_entrate_cassa_agosto2026.xlsx").exists()
    inv = raw["frontier_inventory_2026_10_20"]
    assert "IRPEF" in inv["tax_split"]["found"]
    assert "PRE −11%" in inv["interest"]["block"]
    inv_b = raw["frontier_inventory_2026_10_20b"]
    assert "NO-SHIP change" in inv_b["verdict"]
    assert BU.DFP_SHARE is False
    assert "finanze_tax_cassa" in raw
    assert "IRPEF" in raw["finanze_tax_cassa"]["note"]
    na = raw["national_accounts_central"]["years"]
    assert "2023" in na and "imposte_dirette" in na["2026"]
    assert "capital_cash_sublines" in raw
    assert raw["categories_cash_central"]["map_to_bdi"]["treasury_bal"] is None


def test_dfp_vintages_point_in_time():
    # Before DEF 2021 publication: no 2022 path
    assert BU._dfp_row_at(pd.Timestamp("2021-03-01"), 2022) is None
    # After DEF 2021: 2022 known; DFP 2026 not yet
    row = BU._dfp_row_at(pd.Timestamp("2021-05-01"), 2022)
    assert row is not None and abs(float(row["tributari"]) - 426412) < 1
    # After DEF 2024: prefer DEF 2024's 2025 over nothing
    row25 = BU._dfp_row_at(pd.Timestamp("2024-05-01"), 2025)
    assert row25 is not None and abs(float(row25["tributari"]) - 513561) < 1
    # After DFP 2025: 2025 path updates to DFP 2025 levels
    row25b = BU._dfp_row_at(pd.Timestamp("2025-05-01"), 2025)
    assert row25b is not None and abs(float(row25b["tributari"]) - 529184) < 1


def test_ship_clears_gate(verdict, mae):
    pre = mae[mae["window"] == "PRE"].set_index("h")
    assert set(pre.index) >= set(M.HORIZONS)
    bu = float(pre.loc[list(M.HORIZONS), "bottomup"].mean())
    prod = float(pre.loc[list(M.HORIZONS), "production"].mean())
    assert bu == pytest.approx(verdict["pre_mean_bottomup"], rel=1e-6)
    assert prod == pytest.approx(verdict["pre_mean_production"], rel=1e-6)
    assert BU.SHIP is True
    assert BU.FORCE_LIVE is False
    # isin_curve interest: PRE often fails REL_EDGE — ship must follow the gate, never FORCE_LIVE.
    assert verdict["ship"] is bool(verdict["clears_gate"])
    edge = (prod - bu) / prod
    oos_bu = verdict["oos_mean_bottomup"]
    oos_pr = verdict["oos_mean_production"]
    if verdict["clears_gate"]:
        assert edge >= BU.REL_EDGE
        assert verdict["oos_ok"] is True
        assert oos_bu <= (1.0 + BU.OOS_BAND) * oos_pr + 1e-12
    else:
        assert verdict["ship"] is False


def test_ngdp_clip_is_pre_support(verdict):
    cap = BU._pre_ngdp_support()
    assert cap == pytest.approx(verdict["ngdp_clip"], rel=1e-9)
    assert 0.01 < cap < 0.05  # PRE |NGDP YoY| support ~2.5pp


def test_oos_reported(verdict, mae):
    oos = mae[mae["window"] == "OOS"]
    assert not oos.empty
    assert verdict["oos_mean_bottomup"] == pytest.approx(
        float(oos.set_index("h").loc[list(M.HORIZONS), "bottomup"].mean()), rel=1e-6)


def test_forecast_no_lookahead():
    frame = BU._frame_bn()
    x = BU._mef_rt_mn()
    origin = pd.Timestamp("2018-06-01")
    fc1 = BU.forecast(origin, 12, x, frame)
    assert fc1 is not None
    frame2 = frame.copy()
    x2 = x.copy()
    future = origin + pd.DateOffset(months=1)
    frame2.loc[future:] = frame2.loc[future:] + 5.0
    x2.loc[future:] = x2.loc[future:] + 5_000.0
    fc2 = BU.forecast(origin, 12, x2, frame2)
    assert fc2 is not None
    assert np.allclose(fc1.to_numpy(), fc2.to_numpy(), rtol=0, atol=1e-9)


def test_identity_closes_on_components():
    frame = BU._frame_bn()
    ident = BU._identity_bn(frame)
    rebuilt = (frame["primary"] + frame["interest"] + frame["cap_exp"]
               - frame["tax_rev"] - frame["other_rev"] - frame["treasury_bal"])
    assert (ident - rebuilt).abs().max() < 1e-9


def test_category_table_names_drivers(verdict):
    tab = verdict["category_table"]
    assert tab
    by = {r["category"]: r for r in tab}
    assert by["tax_rev"]["driver"] == "ngdp"
    assert by["primary"]["driver"] == "cpi"
    assert by["interest"]["driver"] is None  # smly; sx rejected under PRE-scaled treasury
    assert by["treasury_bal"]["driver"] is None
    assert verdict["treas_scale"] == pytest.approx(BU.TREAS_SCALE)
    assert verdict["beta_tax"] == pytest.approx(BU.BETA_TAX)
    assert verdict["oos_mean_bottomup"] < verdict["oos_mean_production"]  # shipped OOS cut


def test_apply_replaces_live_when_ship():
    inp = R.load_inputs(refresh=False)
    hist, asof = R.history(inp)
    x = inp["mef"]["fabb_mn"].loc["2010-01":asof].asfreq("MS")
    bt, sc, chosen = R.run_backtest(
        inp["mef"]["fabb_rt_mn"].loc["2010-01":asof].asfreq("MS"))
    fwd, info = R.forward(inp, hist, asof, x, chosen, bt, sc)
    assert "cash_requirement_bottomup_bn" in fwd.columns
    assert info.get("bottomup_engine") == "macro_v2"
    open_yr = fwd.index.year == asof.year
    if info.get("bottomup_ship"):
        # live = bottom-up for the year in progress; later years are the component model
        assert np.allclose(fwd.loc[open_yr, "cash_requirement_bn"],
                           fwd.loc[open_yr, "cash_requirement_bottomup_bn"])
        assert info["bottomup_live_months"] >= 3
        fy = fwd.loc["2026-10-01":"2026-12-01"]
        assert not np.allclose(fy["cash_requirement_bn"], fy["cash_requirement_plan_bn"])
    else:
        # NO-SHIP: production path stays; BU column is research-only
        assert info["bottomup_ship"] is False
    later = ~open_yr
    assert (fwd.loc[later, "cash_requirement_ext_source"] == "component model").all()
    # Explicit ship=False must not overwrite an already-set production column
    prod = fwd.copy()
    prod["cash_requirement_bn"] = 99.0
    out, inf = BU.apply_to_forward(prod, x, asof, ship=False)
    assert inf["bottomup_ship"] is False
    assert np.allclose(out["cash_requirement_bn"], 99.0)
