"""UK bottom-up macro (OBR/RR): identity, no-lookahead, PRE ship gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.uk import bottom_up_macro as BU
from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

AUTH = Path(__file__).resolve().parents[1] / "data" / "uk" / "authority.json"
_COMP_KEYS = ("cg_current_receipts", "cg_current_exp", "cg_capital_exp", "ps_interest_paid",
              "cgncr_ex", "cg_taxes_receivable")
_FIN_KEYS = tuple(BU.FIN_PARTS)


def _ons_ready() -> bool:
    ons = src.ons_monthly()
    return all(k in ons.columns for k in _COMP_KEYS) and all(k in ons.columns for k in _FIN_KEYS)


def test_authority_json_has_obr_categories():
    assert AUTH.exists()
    a = BU.authority()
    cats = a.get("categories_bn", {})
    for k in ("income_tax_bn", "nics_bn", "vat_bn", "corp_tax_bn", "debt_interest_bn", "welfare_bn"):
        assert k in cats
        assert "2025" in cats[k] or 2025 in cats[k] or "2024" in cats[k]


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_component_identity_holds():
    c = BU.components()
    np.testing.assert_allclose(c["cgncr"], c["bridge"] + c["resid"], atol=1e-9)
    np.testing.assert_allclose(c["it"] + c["nics"] + c["vat"] + c["ct"] + c["other_tax"],
                               c["taxes"], atol=1e-9)
    np.testing.assert_allclose(c["welfare"] + c["rdel_proxy"], c["primary"], atol=1e-9)
    np.testing.assert_allclose(
        c["resid"], c[list(BU.FIN_PARTS)].sum(axis=1) + c["resid_gap"], atol=1e-9)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_fin_parts_explain_most_of_residual():
    c = BU.components().dropna(subset=["resid", "resid_gap"])
    mapped = c[list(BU.FIN_PARTS)].sum(axis=1)
    r2 = 1.0 - float(((c["resid"] - mapped) ** 2).sum()
                     / ((c["resid"] - c["resid"].mean()) ** 2).sum())
    assert r2 > 0.80
    assert float(c["resid_gap"].abs().mean()) < float(c["resid"].abs().mean()) * 0.45


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_forecast_no_lookahead():
    c = BU.components()
    origin = c.index[-13]
    base = BU.forecast(c, origin)
    poisoned = c.copy()
    poisoned.iloc[-12:] = 1e6
    np.testing.assert_allclose(base, BU.forecast(poisoned, origin))


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_rr_elasticities_are_order_one():
    e = BU._rr_elas()
    assert 0.5 < e["it_awe"] < 2.5
    assert 0.5 < e["nics_awe"] < 2.5
    assert 0.3 < e["vat_cons"] < 1.5
    assert e["di_gilt_frac"] > 0
    tri = BU._di_triangle()
    assert tri["gilt"].get(0, 0) > 0
    assert tri["bank_rate"].get(0, 0) > 0


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_ship_rule_pre_only_and_consistent():
    d = BU.evaluate()
    assert set(d["bu_pre"]) >= set(M.HORIZONS)
    assert set(d["prod_pre"]) >= set(M.HORIZONS)
    # FORCE_LIVE is off: ship follows the PRE gate (and the module SHIP flag).
    if d["gate_pass"]:
        assert d["ship"] == BU.SHIP
        assert d["edge"] > M.EDGE_BAND
        assert len(d["worse_horizons"]) <= 1
    else:
        assert d["ship"] is False
    assert BU.FORCE_LIVE is False
    audit = d["category_audit"]
    assert "driver" in audit.columns
    assert audit["driver"].notna().all()
    # unmapped residual must not silently claim a fake macro driver
    resid = audit.loc[audit["category"] == "resid_gap"].iloc[0]
    assert "None" in resid["driver"] or "no monthly" in resid["rule"].lower() or "calendar" in resid["rule"].lower() or "seasonal" in resid["rule"].lower()
    assert d["residual_share_pre"] < 0.20


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_v1_also_forced_no_ship():
    from sovereign_cash.uk import bottom_up as V1
    assert V1.SHIP is False


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_equity_and_bank_rate_drivers_present():
    drv = BU._drivers()
    assert "equity" in drv and drv["equity"].dropna().shape[0] > 100
    assert "bank_rate_dpp" in drv and drv["bank_rate_dpp"].dropna().shape[0] > 100
    assert "gdp" in drv and drv["gdp"].dropna().shape[0] > 100


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_shipped_spec_keys_and_authority_audit():
    for k in ("welfare_cpi", "rdel_cpi", "rdel_gdp", "nontax_gdp", "other_tax_retail",
              "other_tax_equity", "cap_retail", "cap_gdp", "soft_pin_alpha", "resid_sa3",
              "fin_recv_pay", "fin_recv_pay_elas", "fin_other_fin", "fin_other_fin_br",
              "fin_acq_sec", "fin_gilt_adj", "fin_apf_xfer", "fin_net_lend", "fin_resid_gap",
              "sa_equity_nudge", "sa_equity_persist", "sa_equity_beta", "sa_equity_feb_w"):
        assert k in BU.SPEC
    assert BU.SPEC["resid_sa3"] == 0.0  # calendar for unmapped resid / default fin parts
    assert BU.SPEC["cap_gdp"] > 0
    assert BU.SPEC["rdel_gdp"] > 0
    assert BU.SPEC["soft_pin_alpha"] == 0.12  # light remit CGNCR pin (PRE-gated)
    assert BU.SPEC["fin_recv_pay"] == "grow_retail"
    assert BU.SPEC["fin_recv_pay_elas"] == 0.0
    assert BU.SPEC["fin_other_fin"] == "sa3_add_br"
    assert BU.SPEC["fin_other_fin_br"] > 0
    assert BU.SPEC["fin_acq_sec"] == "trail12"
    assert BU.SPEC["sa_equity_nudge"] is True
    assert 0 < BU.SPEC["sa_equity_persist"] < 1
    # authority.json audit may lag SPEC; gate is evaluate()
    # NO-SHIP: the monthly CGNCR scorer fails this SPEC (uk_cash_ship_autopsy.md); run.py ships only on evaluate()
    assert BU.SHIP is False and BU.FORCE_LIVE is False


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_fin_part_rules_differ_from_sa3():
    c = BU.components()
    origin = c.index[-25]
    sa3_recv = BU._part_path(c["recv_pay"], origin, 12, 1.0)
    grow_recv = BU._fin_part_path(c["recv_pay"], origin, 12, "recv_pay", BU.SPEC, BU._drivers())
    assert not np.allclose(sa3_recv, grow_recv)
    sa3_acq = BU._part_path(c["acq_sec"], origin, 12, 1.0)
    trail_acq = BU._fin_part_path(c["acq_sec"], origin, 12, "acq_sec", BU.SPEC, BU._drivers())
    assert not np.allclose(sa3_acq, trail_acq)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_resid_sa3_differs_from_calendar():
    c = BU.components()
    origin = c.index[-25]
    cal = BU._resid_path(c["resid"], origin, 12, 0.0)
    sa3 = BU._resid_path(c["resid"], origin, 12, 1.0)
    assert not np.allclose(cal, sa3)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_fin_part_paths_sum_to_resid():
    c = BU.components()
    origin = c.index[-25]
    paths = BU.category_paths(c, origin, h=12)
    recon = paths["resid_gap"].copy()
    for k in BU.FIN_PARTS:
        recon = recon + paths[k]
    np.testing.assert_allclose(recon, paths["resid"], atol=1e-9)


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_sa_equity_nudge_moves_jan_off_calendar_carry():
    """Jan-2026 CGT spike must not reprint one-for-one into Jan-2027."""
    c = BU.components()
    origin = pd.Timestamp("2026-09-01")
    assert origin in c.index or origin <= c.index.max()
    on = BU.category_paths(c, origin, h=6)
    off = BU.category_paths(c, origin, h=6, spec={**BU.SPEC, "sa_equity_nudge": False})
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    i_jan = list(months).index(pd.Timestamp("2027-01-01"))
    i_feb = list(months).index(pd.Timestamp("2027-02-01"))
    # Nudge cuts the calendar carry of the Jan-26 other_tax/IT spike.
    assert on["other_tax"][i_jan] < off["other_tax"][i_jan] - 1.0
    assert on["it"][i_jan] < off["it"][i_jan] - 0.5
    # February moves in the same direction (spill).
    assert on["other_tax"][i_feb] < off["other_tax"][i_feb]
    # Disposal-FY equity for Jan-2027 is known by Sep-2026 (ends Mar-2026).
    r = BU.disposal_fy_equity_return(2027, asof=origin)
    assert np.isfinite(r) and r > 0
    # No lookahead: asof before Mar-2025 must not see Mar-2026.
    r_early = BU.disposal_fy_equity_return(2027, asof=pd.Timestamp("2025-06-01"))
    assert r_early != r


@pytest.mark.skipif(not _ons_ready(), reason="ONS component cache incomplete")
def test_sa_equity_nudge_raises_jan_cgncr_vs_blind_carry():
    c = BU.components()
    origin = pd.Timestamp("2026-09-01")
    on = BU.forecast(c, origin, h=6)
    off = BU.forecast(c, origin, h=6, spec={**BU.SPEC, "sa_equity_nudge": False})
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    i_jan = list(months).index(pd.Timestamp("2027-01-01"))
    # Less SA receipts → higher (less negative) CGNCR in January.
    assert on[i_jan] > off[i_jan] + 2.0
