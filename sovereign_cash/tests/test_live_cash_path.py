"""Every financing-hub surface names the live cash path, never a stale seasonal tag as live."""
from __future__ import annotations

import re

import pytest

from sovereign_cash import investor as iv


def _summ(cc: str, ship: bool) -> dict:
    return {"uk": {"forward": {"cgncr_model": "seasonal_add3", "cgncr_bottomup_ship": ship}},
            "de": {"forward": {"model": "planhc_3", "bottomup_macro_ship": ship}},
            "fr": {"forward": {"model": "seas_mult_3", "bottomup_macro_ship": ship}},
            "it": {"forward": {"model": "smly", "bottomup_ship": ship}},
            "se": {"bottom_up": {"ship": ship, "production_model": "seas_k5"}},
            "no": {"bottom_up": {"ship": ship}}}[cc]


@pytest.mark.parametrize("cc", ["uk", "de", "fr", "it", "se", "no"])
def test_seasonal_role_follows_ship_flag(cc):
    on, off = _summ(cc, True), _summ(cc, False)
    assert iv.bu_ship(cc, on) and not iv.bu_ship(cc, off)
    assert "not live" in iv.seasonal_role(cc, on)
    assert iv.seasonal_role(cc, off).startswith("live")
    assert iv.seasonal_tag(cc, on) in iv.seasonal_label(cc, on)


@pytest.mark.xfail(reason="work in progress in the 10 Nov local snapshot: investor.live_cash_path segments/wording and methodology.page.cash_gate_table were never finished", strict=False)
@pytest.mark.parametrize("cc", iv.ORDER)
def test_live_cash_path_never_claims_stale_seasonal(cc):
    p = iv.live_cash_path(cc)
    if not p:
        pytest.skip(f"no derived data for {cc}")
    s = iv.summary(cc) if cc != "us" else {}
    keys = {g["key"] for g in p["segments"]}
    assert p["bu_ship_flag"] == iv.bu_ship(cc, s)
    if cc == "us":
        assert keys == {"dts"} and p["seasonal_tag"] is None and not p["shipped_bu"]
        return
    tag = p["seasonal_tag"]
    if p["bu_ship_flag"]:
        assert not p["seasonal_live"] and "not live" in (p["seasonal_role"] or "not live")
        assert p["bu_tier"] is None or not p["shipped_bu"]
        if cc in ("de", "fr"):
            # the open year is bottom-up; the seasonal tag may appear only as the fallback it was scored against
            assert "seasonal" not in keys and "bu" in keys
            if tag:
                for m in re.finditer(re.escape(tag), p["basis"] + " " + p["oos"]):
                    assert "fallback" in (p["basis"] + " " + p["oos"])[max(0, m.start() - 30):m.start()]
    else:
        assert not p["shipped_bu"]
        if cc in ("uk", "it", "se"):
            assert p["bu_detail"], f"{cc}: research BU should be described when not live"
    if cc == "uk":
        assert not p["bu_ship_flag"] and "model" in keys
    if cc == "se":
        assert keys == {"rg"} and not p["bu_ship_flag"]


@pytest.mark.parametrize("cc", ["uk", "it", "se"])
def test_bu_research_edge_sign(cc):
    p = iv.live_cash_path(cc)
    if not p or not p.get("bu_detail"):
        pytest.skip("no research BU")
    m = re.search(r"PRE mean MAE ([\d.]+) vs live ([\d.]+) \(BU \d+% (better|worse)\)", p["bu_detail"])
    assert m, p["bu_detail"]
    bu, live, word = float(m[1]), float(m[2]), m[3]
    assert word == ("better" if bu < live else "worse")


@pytest.mark.xfail(reason="work in progress in the 10 Nov local snapshot: investor.live_cash_path segments/wording and methodology.page.cash_gate_table were never finished", strict=False)
def test_overview_and_methodology_do_not_present_fallback_as_live():
    from sovereign_cash.methodology.page import cash_gate_table
    tbl = cash_gate_table()
    for cc in ("de", "fr"):
        s = iv.summary(cc)
        if iv.bu_ship(cc, s):
            assert f"{iv.seasonal_tag(cc, s)} (seasonal fallback / scoreboard, not live" in tbl
    assert "Production model" not in tbl
