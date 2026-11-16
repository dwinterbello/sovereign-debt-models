"""Exhibit 1c must plot cash models, not authority soft-pins — esp. France."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt.note.fytd_deficit import (
    _fy_short,
    _plan_hline_lab,
    _prior_plan_bn,
    _soft_pinned_to_plan,
    build_panel,
    model_remainder,
    next_fy_model_bn,
    next_fy_model_flows,
)


def test_france_outlook_is_bum_a0_not_plf_pin():
    fr = next(p for p in build_panel() if p.cc == "fr")
    rem = model_remainder(fr)
    assert rem is not None and len(rem) >= 1
    # Guard is path-identity vs α=1, not YE≈plan: with YTD past the revised solde,
    # α=0 H2 can land within a few tenths of €bn of plan without being pinned.
    assert not _soft_pinned_to_plan(fr, rem)
    from sovereign_cash.fr import bottom_up_macro as BUM
    live = fr.cum_path.dropna()
    pinned = BUM.path(live.index.max(), int(len(rem)), soft_pin_alpha=1.0).reindex(rem.index)
    assert float((rem.astype(float) - pinned.astype(float)).abs().sum()) > 0.05


def test_france_next_fy_includes_legislated_policy():
    """Next-FY = BUM α=0 + policy overlay — not the bare grown path (~€139) or a PLF pin."""
    from sovereign_cash.fr import bottom_up_macro as BUM
    from sovereign_cash.policy.monthly import add_to_monthly

    fr = next(p for p in build_panel() if p.cc == "fr")
    live = fr.cum_path.dropna()
    origin = live.index.max()
    y0 = fr.fy_origin + pd.DateOffset(months=12)
    months = pd.date_range(y0, periods=12, freq="MS")
    end = y0 + pd.DateOffset(months=11)
    n = int((end.to_period("M") - origin.to_period("M")).n)
    bare = BUM.path(origin, n, soft_pin_alpha=0.0).reindex(months)
    with_pol = add_to_monthly("fr", bare, origin)
    bn = next_fy_model_bn(fr)
    assert bn is not None
    assert bn == pytest.approx(float(with_pol.sum()), abs=0.05)
    # Policy must move the year (LPM / surcharge) — not a no-op.
    assert abs(float(with_pol.sum()) - float(bare.sum())) > 1.0
    # And must not sit on the old unpinned fantasy far below the PLF.
    assert bn > 150.0


def test_italy_outlook_uses_bottomup_column_when_present():
    it = next(p for p in build_panel() if p.cc == "it")
    rem = model_remainder(it)
    assert rem is not None
    # Soft-pin to DFP would hit plan exactly; BU − RRF may sit near it but
    # the remainder series must be finite and cover the open FY months.
    assert rem.notna().all()
    assert abs(float(rem.sum())) < 1e6


def test_france_prior_plan_is_lfi_not_plf():
    fr = next(p for p in build_panel() if p.cc == "fr")
    prior = _prior_plan_bn(fr)
    assert prior is not None
    bn, lab = prior
    assert lab == "prev plan"
    assert bn == 124.4
    assert abs(bn - float(fr.plan)) > 1.0  # PLF 137.2 vs LFI 124.4


def test_plan_hline_labels_include_fy_and_plan():
    assert _fy_short("2026") == "2026"
    assert _fy_short("FY2027") == "FY27"
    assert _fy_short("FY2026-27") == "FY26-27"
    assert _plan_hline_lab("2026", 4.5) == "2026 plan: 4.5"
    assert _plan_hline_lab("FY2027", 6.1) == "FY27 plan: 6.1"
    assert _plan_hline_lab("2026", 4.1, kind="prev plan") == "2026 prev plan: 4.1"
    assert _plan_hline_lab("2026", 4.32, 4.39) == "2026 plan: 4.32"


def test_next_fy_model_flows_match_bn_where_present():
    for p in build_panel():
        flows = next_fy_model_flows(p)
        bn = next_fy_model_bn(p)
        if flows is None:
            assert bn is None
            continue
        assert len(flows) == 12
        assert flows.notna().all()
        assert abs(float(flows.sum()) - float(bn)) < 1e-6
    # Core + thin (ES/JP) must ship a next-FY path.
    have = {p.cc for p in build_panel() if next_fy_model_flows(p) is not None}
    assert {"us", "uk", "de", "fr", "it", "se", "es", "jp"} <= have


def test_uk_next_fy_uses_seasonal_not_collapsed_bottomup():
    uk = next(p for p in build_panel() if p.cc == "uk")
    flows = next_fy_model_flows(uk)
    assert flows is not None and len(flows) == 12
    # Live BU next FY ~£48bn; seasonal / remit ~£136–145bn.
    assert float(flows.sum()) > 100.0
    # January must stay a large receipt month (self-assessment), not ~flat.
    jan = flows[flows.index.month == 1]
    assert len(jan) == 1 and float(jan.iloc[0]) < -10.0
