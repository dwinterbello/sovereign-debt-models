"""The browser engine is the Python engine: random packs, random shocks, same numbers.

Also the invariants a reviewer would check by hand. Pure stdlib + node; no model fits.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from sovereign_cash.scenarios.engine import PackError, evaluate, level_gap

CLI = Path(__file__).resolve().parent / "js" / "engine_cli.mjs"
NODE = shutil.which("node")
TOL = 1e-9


def _pack(seed: int) -> dict:
    r = random.Random(seed)
    n_per = r.randint(2, 6)
    months = n_per * 12
    month_period = [m // 12 for m in range(months)]
    onset = r.randint(0, 6)
    tau = [max(0, m - onset + 1) for m in range(months)]
    lines = [{"id": "receipts", "sign": -1}, {"id": "primary", "sign": 1}, {"id": "interest", "sign": 1}]
    ch = [
        {"id": "ngdp_rec", "type": "beta_monthly", "driver": "ngdp", "line": "receipts",
         "beta": r.uniform(0.05, 2.0), "tau": tau},
        {"id": "cpi_prim", "type": "beta_monthly", "driver": "cpi", "line": "primary",
         "beta": r.uniform(0.05, 2.0), "tau": tau},
        {"id": "brent_rec", "type": "linear", "driver": "brent", "line": "receipts",
         "per_unit": {"level": [r.uniform(-1, 1) for _ in range(n_per)],
                      "growth": [r.uniform(-1, 1) for _ in range(n_per)]}},
        {"id": "cpi_spend_el", "type": "elasticity", "driver": "cpi", "line": "primary",
         "elasticity": r.uniform(0.1, 1.2), "base": [r.uniform(50, 400) for _ in range(n_per)],
         "tau_months": [[max(0, 12 * p + m + 1 - onset) for m in range(12)] for p in range(n_per)]},
        {"id": "cpi_rate_mx", "type": "matrix", "driver": "cpi", "line": "primary", "kind": "rate",
         "tau": [max(0, 12 * (j + 1) - onset) for j in range(n_per)],
         "exposure": [r.uniform(0.3, 1.0)] + [1.0] * (n_per - 1),
         "coef": [[r.uniform(0, 3) if p >= j else 0.0 for p in range(n_per)] for j in range(n_per)]},
        {"id": "ngdp_rec_lvl", "type": "elasticity_monthly", "driver": "ngdp", "line": "receipts",
         "elasticity": r.uniform(0.6, 1.4), "base": [r.uniform(10, 60) for _ in range(months)],
         "tau": [max(0, m - onset - 1) for m in range(months)]},
        {"id": "rates_persist_mx", "type": "matrix", "driver": "rates", "line": "interest", "kind": "persistent",
         "method_group": "authority",
         "tau": [max(0, 12 * (j + 1) - onset) for j in range(n_per)],
         "coef": [[r.uniform(0, 0.03) if p >= j else 0.0 for p in range(n_per)] for j in range(n_per)]},
        {"id": "ngdp_level_mx", "type": "matrix", "driver": "ngdp", "line": "receipts", "kind": "level",
         "tau": [max(0, 12 * (j + 1) - onset) for j in range(n_per)],
         "coef": [[r.uniform(0, 3) if p >= j else 0.0 for p in range(n_per)] for j in range(n_per)]},
        {"id": "rates_book", "type": "grid", "driver": "rates", "line": "interest", "method_group": "book",
         "knots": [-100.0, 0.0, 100.0, 200.0],
         "values": [[r.uniform(-9, -1) for _ in range(n_per)], [0.0] * n_per,
                    [r.uniform(1, 9) for _ in range(n_per)], [r.uniform(9, 20) for _ in range(n_per)]]},
        {"id": "rates_auth", "type": "linear", "driver": "rates", "line": "interest", "method_group": "authority",
         "per_unit": {"level": [r.uniform(0.01, 0.2) for _ in range(n_per)],
                      "growth": [r.uniform(0.01, 0.2) for _ in range(n_per)]}},
    ]
    return {"schema_version": 1, "countries": {"xx": {
        "periods": [str(2026 + i) for i in range(n_per)],
        "month_period": month_period,
        "lines": lines,
        "channels": ch,
        "default_methods": {"rates": r.choice(["book", "authority"])},
        "baseline": {"total": [r.uniform(-50, 200) for _ in range(n_per)]},
        "feedback": {"rate_pct": [r.uniform(0.5, 5.0) for _ in range(n_per)]},
    }}}


def _shocks(r: random.Random) -> dict:
    s = {}
    for d in ("ngdp", "cpi", "brent"):
        if r.random() < 0.8:
            s[d] = {"size": r.uniform(-4, 4), "mode": r.choice(["level", "growth"])}
    if r.random() < 0.8:
        s["rates"] = {"size": r.uniform(-250, 350)}
    if r.random() < 0.5:
        s["methods"] = {"rates": r.choice(["book", "authority"])}
    if r.random() < 0.5:
        s["scale"] = {r.choice(["ngdp_rec", "cpi_spend_el", "rates_book"]): r.uniform(0, 2)}
    s["feedback"] = r.random() < 0.5
    return s


def _close(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(a)))
    return a == b


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_python_and_browser_engines_agree_on_200_random_cases():
    for seed in range(20):
        pack = _pack(seed)
        r = random.Random(1000 + seed)
        cases = [{"cc": "xx", "shocks": _shocks(r)} for _ in range(10)]
        js = json.loads(subprocess.run([NODE, str(CLI)], input=json.dumps({"pack": pack, "cases": cases}),
                                       capture_output=True, text=True, check=True).stdout)
        for case, got in zip(cases, js):
            assert got["ok"], got
            want = evaluate(pack, case["cc"], case["shocks"])
            assert _close(want, got["result"]), (seed, case)


def test_zero_shock_returns_the_baseline_exactly():
    pack = _pack(1)
    out = evaluate(pack, "xx", {})
    assert out["scenario"] == out["baseline"]
    assert all(v == 0.0 for v in out["delta"])


def test_linear_channels_scale_linearly():
    pack = _pack(2)
    off = {"cpi_spend_el": 0.0, "ngdp_rec_lvl": 0.0}  # constant-elasticity channels are not linear
    one = evaluate(pack, "xx", {"ngdp": {"size": 1.0}, "cpi": {"size": 1.0}, "scale": off})["delta"]
    three = evaluate(pack, "xx", {"ngdp": {"size": 3.0}, "cpi": {"size": 3.0}, "scale": off})["delta"]
    assert all(abs(3 * a - b) < 1e-9 for a, b in zip(one, three))


def test_growth_equals_level_after_exactly_twelve_months_and_compounds_after():
    assert level_gap(2.0, "growth", 12) == pytest.approx(2.0)
    assert level_gap(2.0, "growth", 6) < 2.0 < level_gap(2.0, "growth", 24)
    assert level_gap(2.0, "growth", 24) == pytest.approx(100 * (1.02 ** 2 - 1))
    assert level_gap(2.0, "level", 0) == 0.0


def test_higher_ngdp_lowers_borrowing_through_receipts():
    pack = _pack(3)
    d = evaluate(pack, "xx", {"ngdp": {"size": 1.0}})["delta_by_driver"]["ngdp"]
    assert all(v <= 0 for v in d) and any(v < 0 for v in d)


def test_drivers_add_up_to_the_total_without_feedback():
    pack = _pack(4)
    out = evaluate(pack, "xx", {"ngdp": {"size": 1.5}, "rates": {"size": 80.0}, "brent": {"size": -10.0}})
    summed = [sum(v[i] for v in out["delta_by_driver"].values()) for i in range(len(out["periods"]))]
    assert summed == pytest.approx(out["delta"])


def test_only_the_chosen_rates_method_counts():
    pack = _pack(5)
    book = evaluate(pack, "xx", {"rates": {"size": 100.0}, "methods": {"rates": "book"}})["delta"]
    auth = evaluate(pack, "xx", {"rates": {"size": 100.0}, "methods": {"rates": "authority"}})["delta"]
    grid = next(c for c in pack["countries"]["xx"]["channels"] if c["id"] == "rates_book")["values"][2]
    assert book == pytest.approx(grid)
    assert auth != pytest.approx(book)


def test_feedback_adds_interest_on_extra_debt_and_vanishes_at_zero_rate():
    pack = _pack(6)
    s = {"rates": {"size": 100.0}, "feedback": True}
    on = evaluate(pack, "xx", s)
    assert all(f > 0 for f in on["feedback"][1:])
    pack["countries"]["xx"]["feedback"]["rate_pct"] = [-1.0] * len(on["periods"])  # -1% + 100bp = 0
    off = evaluate(pack, "xx", s)
    assert all(abs(f) < 1e-12 for f in off["feedback"])


def test_bad_inputs_are_refused():
    pack = _pack(7)
    with pytest.raises(PackError):
        evaluate(pack, "xx", {"ngdp": {"size": 1.0, "mode": "sideways"}})
    pack["schema_version"] = 99
    with pytest.raises(PackError):
        evaluate(pack, "xx", {})


def test_elasticity_channel_is_epsilon_times_base_times_gap():
    pack = _pack(8)
    ch = next(c for c in pack["countries"]["xx"]["channels"] if c["id"] == "cpi_spend_el")
    out = evaluate(pack, "xx", {"cpi": {"size": 2.0, "mode": "growth"}})["delta_by_line"]["primary"]
    beta_part = evaluate(pack, "xx", {"cpi": {"size": 2.0, "mode": "growth"},
                                      "scale": {"cpi_spend_el": 0.0}})["delta_by_line"]["primary"]
    want = [b * sum((1 + level_gap(2.0, "growth", t) / 100) ** ch["elasticity"] - 1 for t in ts) / 12
            for b, ts in zip(ch["base"], ch["tau_months"])]
    assert [a - b for a, b in zip(out, beta_part)] == pytest.approx(want)


def test_scale_zero_switches_a_channel_off_and_out_of_range_is_refused():
    pack = _pack(9)
    on = evaluate(pack, "xx", {"ngdp": {"size": 1.0}})["delta"]
    off = evaluate(pack, "xx", {"ngdp": {"size": 1.0},
                                "scale": {"ngdp_rec": 0.0, "ngdp_level_mx": 0.0, "ngdp_rec_lvl": 0.0}})["delta"]
    assert any(abs(v) > 0 for v in on) and all(v == 0.0 for v in off)
    with pytest.raises(PackError):
        evaluate(pack, "xx", {"ngdp": {"size": 1.0}, "scale": {"ngdp_rec": 9.0}})


def test_matrix_rate_determinant_hits_once_in_level_mode_and_every_year_in_growth_mode():
    ch = {"id": "m", "type": "matrix", "driver": "cpi", "line": "x", "kind": "rate",
          "tau": [3, 15, 27], "coef": [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]}
    from sovereign_cash.scenarios.engine import channel_delta
    assert channel_delta(ch, {"cpi": {"size": 2.0, "mode": "level"}}, 3, None) == pytest.approx([2.0, 1.0, 0.0])
    assert channel_delta(ch, {"cpi": {"size": 2.0, "mode": "growth"}}, 3, None) == pytest.approx([2.0, 3.0, 3.0])
    ch["kind"] = "level"
    lvl = channel_delta(ch, {"cpi": {"size": 2.0, "mode": "growth"}}, 3, None)
    g = [level_gap(2.0, "growth", t) for t in (3, 15, 27)]
    assert lvl == pytest.approx([g[0], 0.5 * g[0] + g[1], 0.5 * g[1] + g[2]])


def test_persistent_rate_shock_is_the_same_in_either_mode():
    from sovereign_cash.scenarios.engine import channel_delta
    ch = {"id": "p", "type": "matrix", "driver": "rates", "line": "x", "kind": "persistent",
          "tau": [3, 15, 27], "coef": [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]}
    a = channel_delta(ch, {"rates": {"size": 100.0, "mode": "level"}}, 3, None)
    b = channel_delta(ch, {"rates": {"size": 100.0, "mode": "growth"}}, 3, None)
    assert a == b == pytest.approx([100.0, 150.0, 150.0])


def test_monthly_elasticity_is_exact_for_constant_elasticity():
    from sovereign_cash.scenarios.engine import channel_delta
    ch = {"id": "e", "type": "elasticity_monthly", "driver": "ngdp", "line": "x", "elasticity": 0.8,
          "base": [10.0] * 24, "tau": [max(0, m - 2) for m in range(24)]}
    mp = [m // 12 for m in range(24)]
    got = channel_delta(ch, {"ngdp": {"size": 3.0, "mode": "level"}}, 2, mp)
    one = 10.0 * (1.03 ** 0.8 - 1)
    assert got == pytest.approx([one * 9, one * 12])  # months 0-2 are inside the lag (tau 0)
    unit = dict(ch, elasticity=1.0)
    assert channel_delta(unit, {"ngdp": {"size": 3.0, "mode": "level"}}, 2, mp)[1] == pytest.approx(12 * 0.3)
