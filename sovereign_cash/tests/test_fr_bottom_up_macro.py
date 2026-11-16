"""France v2 PLF×RESF category baseline — identity, no-lookahead, ship gate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import bottom_up as BU
from sovereign_cash.fr import bottom_up_macro as BUM
from sovereign_cash.fr import fetch as F
from sovereign_cash.yoy_components import SPEC, components, deficit

CACHED = (F.DATA / "insee_smb.xml").exists()
AUTH = Path(__file__).resolve().parents[1] / "data" / "fr" / "authority.json"


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_v1_ship_off_v2_gated():
    assert BU.SHIP is False
    assert BUM.SHIP is False or BUM.ship_decision()["ship"] is True


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_authority_has_plf_categories():
    cats = json.loads(AUTH.read_text())["categories"]["years"]
    assert "2026" in cats and "2027" in cats
    for y in ("2025", "2026", "2027"):
        assert cats[y]["ir"] is not None and cats[y]["tva"] is not None
        assert cats[y]["credits_ministeriels"] is not None
    elast = json.loads(AUTH.read_text())["elasticities"]
    assert elast["ir"]["driver"] == "masse_salariale"
    assert elast["tva"]["driver"] == "ngdp"
    knobs = elast["ship_knobs"]
    assert knobs["share_k"] == BUM.SHARE_K == 5
    assert BUM.SOFT_PIN_ALPHA == 1.0  # PLF solde pin when known (PRE-flat, OOS-better)
    assert knobs["soft_pin_alpha"] == BUM.SOFT_PIN_ALPHA
    assert knobs["int_smly_w"] == BUM.INT_SMLY_W == 1.0
    assert set(knobs["grow_keys"]) == set(BUM.GROW_KEYS) == {"remboursements_locaux"}
    assert knobs["grow_same_year"] is True and BUM.GROW_SAME_YEAR is True
    assert knobs["use_ols"] is False and BUM.USE_OLS is False
    # 2023–24 exécution fills; 2024 charge = AFT exécution
    assert cats["2023"]["credits_ministeriels"] == 336.1
    assert cats["2023"]["psr"] == 68.1
    assert cats["2024"]["credits_ministeriels"] == 327.0
    assert cats["2024"]["psr"] == 67.8
    assert cats["2024"]["charge"] == 50.1
    assert cats["2024"]["solde_budgets_annexes"] == 0.4
    debt = json.loads(AUTH.read_text())["debt_charge"]["years"]
    assert debt["2024"]["bn"] == 50.1
    # 2022 documented but not pinned (OOS regress)
    pub = json.loads(AUTH.read_text())["categories"].get("published_execution", {})
    assert "2022" in pub.get("years", {})
    assert "2022" not in cats


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_forecast_identity_and_length():
    frame = components("fr")
    origin = frame.dropna().index.max() - pd.DateOffset(months=14)
    # identity holds on components before soft pin (pin shifts cash only)
    fc = BUM.forecast(origin, 12, frame, soft_pin_alpha=0.0)
    assert fc is not None and len(fc) == 12
    rebuilt = deficit("fr", {n: fc[n] for n, _ in SPEC["fr"]})
    assert (rebuilt - fc["cash_requirement"]).abs().max() < 1e-9


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_soft_pin_moves_open_plf_year():
    frame = components("fr")
    origin = pd.Timestamp("2025-11-01")
    raw = BUM.forecast(origin, 12, frame, soft_pin_alpha=0.0)
    pinned = BUM.forecast(origin, 12, frame, soft_pin_alpha=1.0)
    assert raw is not None and pinned is not None
    assert abs(float(raw["cash_requirement"].sum()) - float(pinned["cash_requirement"].sum())) > 0.5


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_no_lookahead_plf_categories():
    """PLF 2026 categories unused before October 2025."""
    frame = components("fr")
    early = pd.Timestamp("2025-06-01")
    late = pd.Timestamp("2025-11-01")
    c_early = BUM.category_annuals(early, 2026, frame)
    c_late = BUM.category_annuals(late, 2026, frame)
    assert c_early is not None and c_late is not None
    auth = json.loads(AUTH.read_text())["categories"]["years"]["2026"]
    # late origin pins IR to PLF 96.8; early must not
    assert c_late["ir"] == pytest.approx(float(auth["ir"]), rel=1e-6)
    assert abs(c_early["ir"] - float(auth["ir"])) > 0.5


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_ship_decision_sets_flag():
    bt = BUM.backtest(step=1)
    d = BUM.ship_decision(bt)
    assert d["pre_bu"] is not None and d["pre_prod"] is not None
    assert BUM.SHIP is bool(d["ship"])
    if d["ship"]:
        assert d["pre_bu"] < d["pre_prod"] * (1.0 - BUM.REL_EDGE)
        assert d["worse_h"] <= 1
        assert d["oos_bu"] is not None and d["oos_bu"] < d["oos_prod"]


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_category_audit_names_drivers():
    tab = BUM.category_audit()
    assert "ir" in set(tab["category"])
    assert tab.loc[tab["category"] == "ir", "driver"].iloc[0] == "masse_salariale"
    assert tab.loc[tab["category"] == "tva", "driver"].iloc[0] == "ngdp"
    remb = tab.loc[tab["category"] == "remboursements_locaux", "rule"].iloc[0]
    assert "same-year" in remb
    nf = tab.loc[tab["category"] == "non_fiscal", "driver"].iloc[0]
    assert nf is None or (isinstance(nf, float) and pd.isna(nf))


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_2024_plf_pin_when_known():
    """Complete 2024 row pins after Oct-2023 (no SMB structural fallback)."""
    frame = components("fr")
    late = pd.Timestamp("2023-11-01")
    c = BUM.category_annuals(late, 2024, frame)
    auth = json.loads(AUTH.read_text())["categories"]["years"]["2024"]
    assert c is not None
    assert c["credits_ministeriels"] == pytest.approx(float(auth["credits_ministeriels"]))
    assert c["cas_pensions"] == pytest.approx(float(auth["cas_pensions"]))


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_macro_ship_false_leaves_seas(monkeypatch):
    from sovereign_cash.fr import model as M

    monkeypatch.setattr(BUM, "SHIP", False)
    monkeypatch.setattr(BU, "SHIP", False)
    h, meta = M.build_history()
    cash = meta["cash_full"].dropna()
    f, info = M.forward(h, cash, "seas_mult_3", "gap0", meta["debt"], meta["mlt"], meta["ops"])
    assert info["bottom_up_ship"] is False
    assert info.get("bottom_up_macro_ship") is False
