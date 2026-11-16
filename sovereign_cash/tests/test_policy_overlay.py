"""Legislated-policy overlay: sources parse and reconcile, the netting rule, and the pack's two lines."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from sovereign_cash.policy import overlay
from sovereign_cash.policy.uk_tables import TOL_M

DATA = overlay.DATA


@pytest.mark.parametrize("key", ["budget_2025", "budget_2026"])
def test_uk_tables_reconcile_to_the_published_total(key):
    d = json.loads((DATA / "uk" / f"{key}.json").read_text())
    assert d["reconciles"]
    assert len(d["measures"]) > 50
    for a, b in zip(d["sum_of_measures_m"], d["total_m"]):
        assert abs(a - b) <= TOL_M
    assert [m["n"] for m in d["measures"]] == list(range(1, len(d["measures"]) + 1))


def test_us_groups_add_up_to_cbo_total():
    d = json.loads((DATA / "us" / "pl119_21.json").read_text())
    ms = [m for m in overlay.raw("us") if "119-21" in m["event"]]
    for i, y in enumerate(d["years"]):
        assert sum(m["by_year"][y] for m in ms) == pytest.approx(d["totals"]["deficit"][i] / 1000.0, abs=1e-9)


def test_us_secure_america_is_on_top_of_obbba():
    saa = json.loads((DATA / "us" / "pl119_98.json").read_text())
    ms = overlay.raw("us")
    assert any("119-98" in m["event"] for m in ms)
    got = next(m for m in ms if "119-98" in m["event"])
    assert got["by_year"][2027] == pytest.approx(saa["outlays"][saa["years"].index(2027)] / 1000.0)
    assert sum(m["by_year"].get(2027, 0.0) for m in ms) == pytest.approx(
        sum(m["by_year"].get(2027, 0.0) for m in ms if "119-21" in m["event"]) + got["by_year"][2027])


def _flat(years, v=1.0):
    return pd.Series({y: v * (1.03 ** (y - years[0])) for y in years})


def test_base_year_measure_counts_only_off_its_share(monkeypatch):
    # a measure worth 10 in the base year that grows with its driver adds nothing; a later one counts in full
    years = list(range(2025, 2030))
    x = _flat(years)
    ms = [{"label": "in base", "name": "a", "head": "tax", "event": "e1",
           "by_year": {y: 10.0 * float(x[y] / x[2025]) for y in years}},
          {"label": "later", "name": "b", "head": "spend", "event": "e2", "by_year": {2027: 4.0, 2028: 5.0}}]
    monkeypatch.setattr(overlay, "raw", lambda cc: ms)
    o = overlay.build("uk", [2026, 2027, 2028, 2029], 2025, 0.5, growth={"tax": x, "spend": x})
    assert o["total"] == pytest.approx([0.0, 4.0, 5.0, 5.0])
    assert sum(g["values"][2] for g in o["groups"]) == pytest.approx(o["total"][2])
    # on the lines: receipts fall when a tax measure adds to borrowing, spending rises with a spending one
    assert o["by_line"]["primary"] == pytest.approx([0.0, 4.0, 5.0, 5.0])
    assert o["by_line"]["receipts"] == pytest.approx([0.0] * 4)


def test_first_year_scaled_only_for_events_before_it(monkeypatch):
    ms = [{"label": "old", "name": "a", "head": "tax", "event": "e1", "by_year": {2025: 0.0, 2026: 12.0},
           "scale_first": True},
          {"label": "new", "name": "b", "head": "tax", "event": "e2", "by_year": {2026: 6.0}, "scale_first": False}]
    monkeypatch.setattr(overlay, "raw", lambda cc: ms)
    o = overlay.build("uk", [2026, 2027], 2025, 7 / 12)
    assert o["total"][0] == pytest.approx(12.0 * 7 / 12 + 6.0)


@pytest.mark.parametrize("cc", ["uk", "us", "de"])
def test_pack_draws_both_lines(cc):
    path = overlay.DATA.parents[1] / "derived" / "scenario_pack.json"
    if not path.exists():
        pytest.skip("no pack built")
    c = json.loads(path.read_text())["countries"].get(cc)
    if not c or "policy" not in c:
        pytest.skip("pack built before the overlay")
    p = c["policy"]
    assert c["baseline"]["with_policy"] == pytest.approx([a + b for a, b in zip(c["baseline"]["total"], p["total"])])
    assert [sum(g["values"][i] for g in p["groups"]) for i in range(len(c["periods"]))] == pytest.approx(p["total"])
    signs = {ln["id"]: ln["sign"] for ln in c["lines"]}
    assert [sum(signs[k] * v[i] for k, v in p["by_line"].items()) for i in range(len(c["periods"]))] == \
        pytest.approx(p["total"])
    assert p["sources"] and all(s["url"].startswith("https://") for s in p["sources"])


def test_every_country_is_cross_checked():
    d = json.loads((DATA / "checks.json").read_text())
    assert {r["cc"] for r in d["rows"]} == set(overlay.COUNTRIES)
    assert all(r["url"].startswith("https://") and r["verdict"] for r in d["rows"])


def test_uk_tax_decisions_add_up_to_the_obr_totals():
    assert max(overlay.obr_check().values()) < 0.15  # £bn; the OBR rounds to 0.1


@pytest.mark.parametrize("cc", ["uk", "us", "es", "se", "no", "jp"])
def test_authority_bridge_adds_up(cc):
    path = overlay.DATA.parents[1] / "derived" / "scenario_pack.json"
    if not path.exists():
        pytest.skip("no pack built")
    b = (json.loads(path.read_text()).get("authority_bridge") or {}).get(cc)
    if not b:
        pytest.skip("pack built before the bridge")
    for r in b["rows"]:
        parts = sum(v["forecast"] + v["definitions"] for v in r["lines"].values())
        assert r["authority"] + parts == pytest.approx(r["ours"], abs=1e-6)
