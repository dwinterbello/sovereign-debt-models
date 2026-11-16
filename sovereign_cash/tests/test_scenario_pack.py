"""The built scenario pack is well formed and its own checks pass."""
from __future__ import annotations

import json

import pytest

from sovereign_cash.scenarios import engine
from sovereign_cash.scenarios.pack import PACK_PATH

pytestmark = pytest.mark.skipif(not PACK_PATH.exists(), reason="run sovereign_cash.scenarios.pack first")
TYPES = {"linear", "grid", "elasticity", "elasticity_monthly", "matrix", "beta_monthly"}


@pytest.fixture(scope="module")
def pack() -> dict:
    return json.loads(PACK_PATH.read_text())


def test_every_country_built_without_errors(pack):
    assert pack["schema_version"] == engine.SCHEMA_VERSION
    assert pack["errors"] == {}
    assert set(pack["countries"]) == {"us", "uk", "de", "fr", "it", "es", "se", "no", "jp"}


def test_channels_are_well_formed(pack):
    for c in pack["countries"].values():
        n = len(c["periods"])
        lines = {ln["id"] for ln in c["lines"]}
        assert len(c["baseline"]["total"]) == n
        for ch in c["channels"]:
            assert ch["type"] in TYPES, ch["id"]
            assert ch["line"] in lines, ch["id"]
            assert ch.get("method") and ch.get("source") and ch.get("badge"), ch["id"]
            if ch["type"] == "linear":
                assert all(len(v) == n for v in ch["per_unit"].values()), ch["id"]
            if ch["type"] == "grid":
                assert all(len(r) == n for r in ch["values"]) and 0.0 in ch["knots"], ch["id"]
                assert ch["values"][ch["knots"].index(0.0)] == [0.0] * n, ch["id"]
            if ch["type"] == "elasticity":
                assert len(ch["base"]) == n and all(len(t) == 12 for t in ch["tau_months"]), ch["id"]
            if ch["type"] == "elasticity_monthly":
                assert len(ch["base"]) == len(ch["tau"]) == len(c["month_period"]), ch["id"]


def test_identity_closes_on_the_baseline(pack):
    for cc, c in pack["countries"].items():
        bl = c["baseline"].get("lines")
        if not bl:
            continue
        for i in range(len(c["periods"])):
            s = sum(ln["sign"] * bl[ln["id"]][i] for ln in c["lines"] if ln["id"] in bl)
            if set(bl) == {ln["id"] for ln in c["lines"]}:
                assert s == pytest.approx(c["baseline"]["total"][i], abs=1e-6), (cc, i)


def test_the_engine_reproduces_the_full_models(pack):
    for cc, r in pack["checks"].items():
        assert "error" not in r, (cc, r)
        if cc in ("_rates_interpolation", "_curve_linearity"):
            # straight lines between grid points / linear in the two anchors: within 1% of the interest bill
            assert all(x["pct_of_interest"] < 1.0 for x in r.values()), (cc, r)
            continue
        if cc == "uk":
            assert all(x["max_abs_err_bn"] < 1e-6 for x in r.values()), r
        else:
            assert all(e < 1e-6 for e in r["vs_published_table_bn"].values()), (cc, r)
            assert all(x["max_abs_err"] < 1e-6 for x in r["vs_full_model_bn"]), (cc, r)


def test_presets_evaluate_everywhere(pack):
    for p in pack["presets"]:
        for cc, c in pack["countries"].items():
            s = json.loads(json.dumps(p["shocks"]))
            for d in ("ngdp", "cpi"):
                if d in s and s[d].get("mode") not in c.get("modes", {}).get(d, ["level", "growth"]):
                    s[d]["mode"] = c["modes"][d][0]
            out = engine.evaluate(pack, cc, s)
            assert all(abs(v) < 1e7 for v in out["scenario"]), (p["id"], cc)


def test_rates_rise_raises_borrowing_everywhere(pack):
    for cc in pack["countries"]:
        d = engine.evaluate(pack, cc, {"rates": {"size": 100.0}})["delta"]
        assert sum(d) > 0, (cc, d)


def test_level_spec_responses_persist_and_compound(pack):
    """France and Italy: a one-off level shift persists; growth compounds (the point of the respecification)."""
    for cc in ("fr", "it"):
        if pack["countries"][cc].get("spec") != "ratio":
            continue
        lvl = engine.evaluate(pack, cc, {"ngdp": {"size": 1.0, "mode": "level"}})["delta_by_driver"]["ngdp"]
        gro = engine.evaluate(pack, cc, {"ngdp": {"size": 1.0, "mode": "growth"}})["delta_by_driver"]["ngdp"]
        assert all(v < 0 for v in lvl[1:]), (cc, lvl)          # higher GDP lowers borrowing every year
        assert abs(gro[-1]) > 2.0 * abs(gro[1]), (cc, gro)     # and builds up when growth is faster


def test_curve_anchors_are_hit(pack):
    import numpy as np

    from sovereign_cash.scenarios import curve_pca
    cache = curve_pca.load()
    for f in cache["markets"].values():
        if "error" in f:
            continue
        sh = curve_pca.shape(f, 120.0, -40.0)
        assert float(sh(np.array(0.25))) == pytest.approx(120.0, abs=1e-6)
        assert float(sh(np.array(10.0))) == pytest.approx(-40.0, abs=1e-6)


def _next_label(p: str) -> str:
    if p.startswith("FY"):
        return f"FY{int(p[2:]) + 1}"
    if "-" in p:
        y = int(p[:4]) + 1
        return f"{y}-{str(y + 1)[2:]}"
    return str(int(p) + 1)


def test_history_runs_up_to_the_forecast_on_the_same_lines(pack):
    """Outturn is display only, but it must sit right before the forecast and on its lines."""
    for cc, c in pack["countries"].items():
        h = c.get("history")
        assert h, f"{cc}: no history"
        ps = h["periods"]
        assert _next_label(ps[-1]) == c["periods"][0], cc
        assert all(_next_label(a) == b for a, b in zip(ps, ps[1:])), cc
        assert len(h["total"]) == len(ps) and sum(v is not None for v in h["total"]) >= len(ps) - 1, cc
        ids = {ln["id"] for ln in c["lines"]}
        assert set(h["lines"]) <= ids, cc
        for k, v in h["lines"].items():
            assert len(v) == len(ps), (cc, k)
        if set(h["lines"]) == ids:  # full identity: lines add up to the total
            signs = {ln["id"]: ln["sign"] for ln in c["lines"]}
            for i, t in enumerate(h["total"]):
                if t is not None and all(h["lines"][k][i] is not None for k in ids):
                    assert sum(signs[k] * h["lines"][k][i] for k in ids) == pytest.approx(t, abs=1e-3), (cc, i)
        stub = h.get("stub")
        if stub:
            assert set(stub["lines"]) == ids, cc
            first = c["periods"][0].replace("FY", "")[:4]
            assert stub["months"][0][:4] in (first, str(int(first) - 1)), cc


@pytest.mark.parametrize("cc,board_has_policy", (("fr", True), ("de", True), ("it", False)))
def test_boards_after_the_open_year_are_our_model(pack, cc, board_has_policy):
    """After the year in progress each cash board runs on our component path: it equals the scenario baseline.

    Since f0623e676 the France and Germany boards take those years from ``sarimax.later_years``, which adds the
    legislated-policy overlay (``policy.monthly``), so they equal the pack's ``with_policy`` line; the Italy board
    calls ``our_path`` directly (no overlay; Ex1c adds it), so it equals the no-policy ``total``."""
    c = pack["countries"].get(cc)
    if not c or not c["baseline"].get("board_full_year"):
        pytest.skip("no board")
    want = c["baseline"]["with_policy"] if board_has_policy else c["baseline"]["total"]
    for i in range(1, len(c["periods"])):
        assert want[i] == pytest.approx(c["baseline"]["board_full_year"][i], abs=0.05), (cc, i)


def test_no_authority_total_in_the_baseline(pack):
    """Baselines are our models: no line is a bridge to a plan."""
    for c in pack["countries"].values():
        assert not any(ln.get("bridge") for ln in c["lines"])
