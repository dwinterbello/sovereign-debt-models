"""France bottom-up cash-requirement baseline — identity, PRE ship gate, no-lookahead."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import bottom_up as BU
from sovereign_cash.fr import fetch as F
from sovereign_cash.yoy_components import SPEC, components, deficit

CACHED = (F.DATA / "insee_smb.xml").exists()
AUTH = Path(__file__).resolve().parents[1] / "data" / "fr" / "authority.json"


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_components_rebuild_smb_deficit():
    frame = components("fr")
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]})
    smb = F.load_smb()
    cash = -C.monthly_from_ytd(smb["solde_ytd"])
    err = (y.reindex(cash.index) - cash).dropna().abs()
    assert err.mean() < 0.05


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_forecast_identity_and_length():
    frame = components("fr")
    origin = frame.dropna().index.max() - pd.DateOffset(months=14)
    fc = BU.forecast(origin, 12, frame)
    assert fc is not None and len(fc) == 12
    rebuilt = deficit("fr", {n: fc[n] for n, _ in SPEC["fr"]})
    assert (rebuilt - fc["cash_requirement"]).abs().max() < 1e-9


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_no_lookahead_charge_vintage():
    """AFT charge for year Y is unused before September Y−1."""
    frame = components("fr")
    # origin before Sep-2025 must not read 2026 charge (62.6)
    early = pd.Timestamp("2025-06-01")
    late = pd.Timestamp("2025-10-01")
    i_early = BU._interest_path(frame["interest"], early, 18)
    i_late = BU._interest_path(frame["interest"], late, 18)
    assert i_early is not None and i_late is not None
    m = pd.Timestamp("2026-06-01")
    # late origin sees charge 62.6 × June share; early uses history growth
    charge = json.loads(AUTH.read_text())["debt_charge"]["years"]["2026"]["bn"]
    sh = BU._month_share(frame["interest"].dropna().loc[:late], late)
    assert sh is not None
    assert i_late[m] == pytest.approx(charge * float(sh.loc[6]), rel=1e-6)
    assert abs(i_early[m] - charge * float(sh.loc[6])) > 0.05


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_v1_rejected_ship_false():
    """v1 seas_mult + AFT charge is rejected; live gate stays off (see bottom_up_macro)."""
    assert BU.SHIP is False
    bt = BU.backtest(step=1)
    d = BU.ship_decision(bt)
    assert d["pre_bu"] is not None and d["pre_prod"] is not None
    assert d["oos_bu"] is not None and d["oos_prod"] is not None


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_ship_false_leaves_seas_as_live_shape(monkeypatch):
    """Lock: if both v1 and v2 SHIP are off, forward cash shape stays on seas_mult."""
    from sovereign_cash.fr import bottom_up_macro as BUM
    from sovereign_cash.fr import model as M

    monkeypatch.setattr(BU, "SHIP", False)
    monkeypatch.setattr(BUM, "SHIP", False)
    h, meta = M.build_history()
    cash = meta["cash_full"].dropna()
    f, info = M.forward(h, cash, "seas_mult_3", "gap0", meta["debt"], meta["mlt"], meta["ops"])
    assert info["bottom_up_ship"] is False
    last = cash.index.max()
    fut = f.loc[f.index > last]
    assert "cash_requirement_bottomup_bn" in fut.columns
    assert fut["cash_requirement_seas_bn"].notna().all()


@pytest.mark.skipif(not CACHED, reason="INSEE SMB cache missing")
def test_authority_inventory_lists_aft_and_plf():
    rows = BU.authority_series_used()
    blob = json.dumps(rows)
    assert "charge" in blob.lower() or "AFT" in blob
    assert "déficit" in blob.lower() or "deficit" in blob.lower()
