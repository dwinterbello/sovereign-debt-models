"""Build the scenario pack: each country's baseline and its exact response to each shock.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.scenarios.pack [--only fr uk] [--no-check]

The pack is the single input of the shock dashboard and the Methodology page.
Every number in it comes from a fitted model or a published authority table;
``engine.evaluate`` only combines them. ``check`` re-runs the real models for
sample shocks and records how far the engine is from them (shown on the page).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios import engine, history
from sovereign_cash.scenarios.channels import annual, cbo, obr, own_annual, sarimax
from sovereign_cash.scenarios.channels.common import default_onset, period_weights

log = logging.getLogger(__name__)

SC = Path(__file__).resolve().parents[1]
REPO = SC.parent
PACK_PATH = SC / "derived" / "scenario_pack.json"
ORDER = ("us", "uk", "de", "fr", "it", "es", "se", "no", "jp")
DRIVERS = {
    "ngdp": {"label": "Nominal GDP", "unit": "%", "range": [-5, 5], "step": 0.25, "modes": True},
    "cpi": {"label": "Inflation", "unit": "pp", "range": [-3, 6], "step": 0.25, "modes": True},
    "rates": {"label": "Bond yields, parallel", "unit": "bp", "range": [-200, 300], "step": 25, "modes": False},
    "r3m": {"label": "3-month yield", "unit": "bp", "range": [-200, 300], "step": 25, "modes": False},
    "r10y": {"label": "10-year yield", "unit": "bp", "range": [-200, 300], "step": 25, "modes": False},
    "brent": {"label": "Brent oil price", "unit": "%", "range": [-50, 50], "step": 5, "modes": False},
}
PRESETS = [
    {"id": "base", "label": "Baseline", "shocks": {}},
    {"id": "pub_gdp", "label": "Published table: NGDP +1pp", "shocks": {"ngdp": {"size": 1.0, "mode": "growth"}}},
    {"id": "pub_cpi", "label": "Published table: CPI +1pp", "shocks": {"cpi": {"size": 1.0, "mode": "growth"}}},
    {"id": "pub_rates", "label": "Published table: yields +100bp", "shocks": {"rates": {"size": 100.0}}},
    {"id": "stagflation", "label": "Stagflation", "shocks": {
        "ngdp": {"size": -1.0, "mode": "growth"}, "cpi": {"size": 2.0, "mode": "growth"},
        "rates": {"size": 150.0}, "feedback": True}},
    {"id": "recession", "label": "Recession with easing", "shocks": {
        "ngdp": {"size": -2.0, "mode": "growth"}, "cpi": {"size": -0.5, "mode": "growth"},
        "rates": {"size": -100.0}, "brent": {"size": -30.0}, "feedback": True}},
    {"id": "selloff", "label": "Bond sell-off", "shocks": {"rates": {"size": 200.0}, "feedback": True}},
    {"id": "bear_steep", "label": "Bear steepener", "shocks": {"r3m": {"size": 25.0}, "r10y": {"size": 150.0},
                                                               "feedback": True}},
    {"id": "bull_flat", "label": "Cuts, long end anchored", "shocks": {"r3m": {"size": -150.0},
                                                                       "r10y": {"size": -25.0}}},
]


def _model_selection() -> dict | None:
    """The FR/DE/IT level-spec backtest behind ``monthly_components.LEVEL_SPEC`` (for the Methodology page)."""
    p = SC / "derived" / "level_spec_backtest.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    from sovereign_cash.monthly_components import LEVEL_SPEC, METHOD
    out = {"results": d, "in_use": LEVEL_SPEC, "methods": {f"{c}/{n}": m for (c, n), m in METHOD.items()}}
    my = SC / "derived" / "multi_year_backtest.json"
    if my.exists():
        raw = json.loads(my.read_text())
        out["multi_year"] = {cc: r["summary"] for cc, r in raw.items()}
    oa = own_annual.BACKTEST_PATH
    if oa.exists():
        out["own_annual"] = json.loads(oa.read_text())
        out["multi_year_rows"] = {cc: [[x["origin"], x["year"], x["h"], x["spec"], round(x["err_total"], 2)]
                                       for x in r["rows"] if x["spec"] in ("final", "current", "naive")]
                                  for cc, r in raw.items()}
    return out


def _curve_summary() -> dict | None:
    from sovereign_cash.scenarios import curve_pca

    if not curve_pca.CACHE.exists():
        return None
    c = curve_pca.load()
    out = {"source_for": c["source_for"], "markets": {}}
    for m, f in c["markets"].items():
        if "error" in f:
            out["markets"][m] = {"error": f["error"]}
            continue
        w = curve_pca.weights(f)
        out["markets"][m] = {"sample": f["sample"], "explained": f["explained"], "grid": f["grid"],
                             "w3m": w[:, 0].tolist(), "w10y": w[:, 1].tolist(),
                             "two_pc_3m": curve_pca.shape_two_pc(f, 100.0, 0.0).tolist()}
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        return "unknown"


def builders(onset: pd.Timestamp) -> dict[str, Callable]:
    out: dict[str, Callable] = {cc: (lambda b, cc=cc: sarimax.build(cc, b)) for cc in sarimax.COUNTRIES}
    # our own annual component model everywhere else; the published rules (OBR reckoner, CBO rules of thumb) stay
    # as the 'authority' method, and the official baselines are shown beside ours for comparison only
    out["uk"] = lambda b: own_annual.build("uk", b, onset, published=obr.build(b, onset))
    out["us"] = lambda b: own_annual.build("us", b, onset, published=cbo.build(b, onset))
    for cc in annual.COUNTRIES:
        out[cc] = lambda b, cc=cc: own_annual.build(cc, b, onset, published=annual.build(cc, b, onset),
                                                    published_channels=False)
    return out


def _gdp(book, c: dict) -> list[float] | None:
    """Nominal GDP per period (local bn), fiscal years weighted from calendar years."""
    g = book.annual(c["cc"], "gdp_bn")
    out = []
    for p in c["periods"]:
        w = period_weights(p)
        if not all(y in g.index and np.isfinite(g[y]) for y in w):
            return None
        out.append(float(sum(g[y] * k for y, k in w.items())))
    return out


def _period_year(cc: str, label: str) -> int:
    return int(label[2:]) if label.startswith("FY") else int(label[:4])


def _policy(c: dict) -> dict | None:
    """Legislated policy on top of our forecast (sovereign_cash.policy.overlay): a fixed line, never shocked."""
    from sovereign_cash.policy import overlay

    cc = c["cc"]
    if cc not in overlay.COUNTRIES:
        return None
    years = [_period_year(cc, p) for p in c["periods"]]
    origin = (c.get("model") or {}).get("origin_period")  # the last outturn period (year as int)
    own = cc in own_annual.COUNTRIES
    if origin is not None and own:
        base = origin if isinstance(origin, int) else _period_year(cc, str(origin))
    else:
        base = years[0]
    span = list(range(base, years[-1] + 1))
    y, p = own_annual.levels(cc, span)
    spend = y if own and own_annual.method(cc, "primary") == "persist_gdp" else p
    first_share = c["month_period"].count(0) / 12.0 if c.get("month_period") else 1.0
    pol = overlay.build(cc, years, base, first_share if base == years[0] - 1 else 1.0,
                        growth={"tax": y, "spend": spend})
    if pol is None:
        return None
    pol["base_period_label"] = own_annual.label(cc, base) if own else c["periods"][0]
    c["baseline"]["with_policy"] = [a + b for a, b in zip(c["baseline"]["total"], pol["total"])]
    return pol


def build(only: list[str] | None = None, *, with_book: bool = True, book=None) -> dict:
    from sovereign_cash.scenarios.channels.debt_book import DebtBook

    onset = default_onset()
    if book is None and with_book:
        book = DebtBook()
    countries, errors = {}, {}
    fns = builders(onset)
    for cc in ORDER:
        if only and cc not in only:
            continue
        try:
            countries[cc] = fns[cc](book)
            if book is not None:
                countries[cc]["gdp"] = _gdp(book, countries[cc])
            countries[cc]["history"] = history.build(cc, countries[cc], book)
            pol = _policy(countries[cc])
            if pol is not None:
                countries[cc]["policy"] = pol
        except Exception as exc:  # noqa: BLE001 - one country must not sink the pack
            log.exception("pack: %s failed", cc)
            errors[cc] = f"{type(exc).__name__}: {exc}"
    out = {
        "model_selection": _model_selection(),
        "curve_factors": _curve_summary(),
        "schema_version": engine.SCHEMA_VERSION,
        "built_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_vintage": _git_sha(),
        "debt_book_asof": None if book is None else book.asof,
        "drivers": DRIVERS,
        "presets": PRESETS,
        "countries": countries,
        "errors": errors,
    }
    try:
        from sovereign_cash.scenarios import bridge

        out["authority_bridge"] = bridge.build(out)
    except Exception as exc:  # noqa: BLE001 - a display table must not sink the pack
        log.exception("pack: authority bridge failed")
        errors["bridge"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------- checks against the real models
def check(pack: dict, book=None) -> dict:
    """Engine vs the full models, recorded in the pack and shown in the trust strip."""
    out = {}
    for cc in pack["countries"]:
        try:
            if cc in sarimax.COUNTRIES:
                out[cc] = _check_sarimax(pack, cc)
            elif cc == "uk":
                out[cc] = _check_uk(pack)
        except Exception as exc:  # noqa: BLE001
            out[cc] = {"error": f"{type(exc).__name__}: {exc}"}
    if book is not None:
        try:
            out["_rates_interpolation"] = _check_rates(pack, book)
        except Exception as exc:  # noqa: BLE001
            out["_rates_interpolation"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            out["_curve_linearity"] = _check_curve(pack, book)
        except Exception as exc:  # noqa: BLE001
            out["_curve_linearity"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _check_curve(pack: dict, book, combos=((150.0, -50.0), (-100.0, 200.0), (300.0, 300.0))) -> dict:
    """The book re-run on curve shapes it was not built from vs the engine's linear combination."""
    from sovereign_cash.scenarios import curve_pca
    from sovereign_cash.scenarios.channels.debt_book import DebtBook

    cache = curve_pca.load()
    res = {}
    for cc, c in pack["countries"].items():
        chans = [x for x in c["channels"] if x["driver"] in ("r3m", "r10y") and x["id"].endswith("_book")]
        if len(chans) != 2:
            continue
        fac, _ = curve_pca.factors_for(cc, cache)
        worst = 0.0
        for x3, x10 in combos:
            run = book.inputs.project(curve_pca.shape(fac, x3, x10))
            probe = DebtBook.__new__(DebtBook)
            probe.runs = {0.0: book.runs[0.0], 1.0: run}
            probe.grid = (0.0, 1.0)
            want = probe.channel(cc, [int(p) for p in c["periods"]] if c["periods"][0].isdigit() else [],
                                 weights=_weights(c))["values"][1]
            got = [sum(engine.channel_delta(ch, {"r3m": {"size": x3}, "r10y": {"size": x10}},
                                            len(c["periods"]), None)[i] for ch in chans)
                   for i in range(len(c["periods"]))]
            worst = max(worst, _max_err(got, want))
        base_i = abs(sum(c["baseline"]["lines"].get("interest", [1.0]))) or 1.0
        res[cc] = {"max_abs_err_bn": worst, "pct_of_interest": 100.0 * worst / base_i}
    return res


def _check_rates(pack: dict, book, between=(50.0, 150.0, 250.0)) -> dict:
    """The ISIN book re-run between grid points vs the engine's straight-line interpolation."""
    from sovereign_cash.scenarios.channels.debt_book import DebtBook

    res = {}
    extra = {b: book.inputs.project(b) for b in between}
    for cc, c in pack["countries"].items():
        ch = next((x for x in c["channels"] if x["id"] == f"{cc}_rates_book"), None)
        if ch is None:
            continue
        worst = 0.0
        for b, run in extra.items():
            probe = DebtBook.__new__(DebtBook)
            probe.runs = {0.0: book.runs[0.0], b: run}
            probe.grid = (0.0, b)
            want = probe.channel(cc, [int(p) for p in c["periods"]] if c["periods"][0].isdigit() else [],
                                 weights=_weights(c))["values"][1]
            got = engine.channel_delta(ch, {"rates": {"size": b}}, len(c["periods"]), None)
            worst = max(worst, _max_err(got, want))
        base_i = abs(sum(c["baseline"]["lines"].get("interest", [1.0]))) or 1.0
        res[cc] = {"max_abs_err_bn": worst, "pct_of_interest": 100.0 * worst / base_i}
    return res


def _weights(c: dict) -> list[dict[int, float]] | None:
    p = c["periods"][0]
    if p.startswith("FY"):
        return [{int(x[2:]) - 1: 0.25, int(x[2:]): 0.75} for x in c["periods"]]
    if "-" in p:
        return [{int(x[:4]): 0.75, int(x[:4]) + 1: 0.25} for x in c["periods"]]
    return None


def _max_err(a, b) -> float:
    return float(max(abs(x - y) for x, y in zip(a, b)))


def _check_sarimax(pack: dict, cc: str) -> dict:
    from sovereign_cash import monthly_components as M
    from sovereign_cash.scenarios import unit_shocks as U

    c = pack["countries"][cc]
    origin = pd.Timestamp(c["origin"])
    years = [int(p) for p in c["periods"]]
    last = pd.Timestamp(f"{years[-1]}-12-01")
    h = max(12, (last.year - origin.year) * 12 + (12 - origin.month) + 1)
    res = {"vs_published_table_bn": {}, "vs_full_model_bn": []}
    pub = U.run(cc, years=years)
    for row in pub.get("shocks", []):
        key = {"gdp": "ngdp", "cpi": "cpi", "rates": "rates"}[row["key"]]
        if key == "rates":
            continue  # book: _check_rates re-runs today's; AFT: same function on both sides
        s = {key: {"size": 1.0, "mode": "growth"}}
        got = engine.evaluate(pack, cc, s)["delta"]
        res["vs_published_table_bn"][key] = _max_err(got, [row["delta"][str(y)] for y in years])
    spec = M.spec_for(cc)
    base = M.live_path(cc, origin, h, spec=spec)
    rng = np.random.default_rng(7)
    for i in range(4):
        g, p = float(rng.uniform(-3, 3)), float(rng.uniform(-2, 3))
        mode = "growth" if i % 2 == 0 or spec == "yoy" else "level"
        shock = {"ngdp": g, "hicp": p} if mode == "growth" else {"ngdp_level": g, "hicp_level": p}
        full = M.live_path(cc, origin, h, shock=shock, spec=spec)
        if full is None or base is None:
            continue
        want = [float((full - base)[full.index.year == y].sum()) for y in years]
        got = engine.evaluate(pack, cc, {"ngdp": {"size": g, "mode": mode},
                                         "cpi": {"size": p, "mode": mode}})["delta"]
        res["vs_full_model_bn"].append({"ngdp_pp": round(g, 3), "cpi_pp": round(p, 3), "mode": mode,
                                        "max_abs_err": _max_err(got, want)})
    return res


def _check_uk(pack: dict) -> dict:
    """Engine (growth mode, full exposure) vs ready_reckoner.run for the same shock."""
    from sovereign_cash.scenarios.channels.obr import BUNDLES
    from sovereign_cash.uk import ready_reckoner as R

    c = json.loads(json.dumps(pack["countries"]["uk"]))
    for ch in c["channels"]:
        if ch["type"] == "matrix":
            ch["exposure"] = [1.0] * len(ch["tau"])
            ch["tau"] = [12 * (j + 1) for j in range(len(ch["tau"]))]
    p2 = {**pack, "countries": {"uk": c}}
    fys = c["periods"]
    out = {}
    for driver in ("ngdp", "cpi", "rates"):
        size = 100.0 if driver == "rates" else 1.0
        shocks = {}
        for det, kind, units in BUNDLES[driver]:
            if det not in R.determinants():
                continue
            # growth mode: level determinants carry the compounded gap, rate ones the growth
            shocks[det] = {fy: units * (engine.level_gap(size, "growth", 12 * (j + 1)) if kind == "level" else size)
                           for j, fy in enumerate(fys)}
        eff = R.run(shocks)
        eff = eff[eff["determinant"] != "CGNCR financing loop"]
        want = R.psnb(eff).reindex(fys).fillna(0.0).tolist()
        s = {driver: {"size": size, "mode": "growth"},
             "methods": {"rates": "authority", "ngdp": "authority", "cpi": "authority"}}
        got = engine.evaluate(p2, "uk", s)["delta"]
        out[driver] = {"max_abs_err_bn": _max_err(got, want)}
    return out


RECORD_PATH = SC / "derived" / "forecast_vintages.jsonl"


def record(pack: dict, path: Path = RECORD_PATH) -> dict:
    """Archive today's baselines (one record per build date, the latest wins) and summarise the archive.

    This is the real-time track record: each year's forecasts are scored against outturn when the year closes.
    """
    day = pack["built_at"][:10]
    rec = {"date": day, "built_at": pack["built_at"], "model_vintage": pack.get("model_vintage"), "countries": {}}
    for cc, c in pack["countries"].items():
        stub = (c.get("history") or {}).get("stub") or {}
        tot = [float(v) for v in c["baseline"]["total"]]
        if stub:
            tot[0] += float(stub.get("total", 0.0))
        rec["countries"][cc] = {"periods": c["periods"], "baseline_full_period": [round(v, 3) for v in tot],
                                "source": c["baseline"].get("source", "")}
        if c.get("policy"):
            rec["countries"][cc]["with_policy_full_period"] = [
                round(v + p, 3) for v, p in zip(tot, c["policy"]["total"])]
    old = []
    if path.exists():
        old = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    keep = [r for r in old if r.get("date") != day] + [rec]
    keep.sort(key=lambda r: r["date"])
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in keep))
    first_periods = [p for c in pack["countries"].values() for p in c["periods"][:1]]
    return {"n_builds": len(keep), "since": keep[0]["date"],
            "first_scorable": min(first_periods) if first_periods else None}


def write(pack: dict, path: Path = PACK_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path == PACK_PATH:
        pack["forecast_record"] = record(pack)
    path.write_text(json.dumps(pack, indent=1, allow_nan=False))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-book", action="store_true", help="skip the ISIN-book rates grid")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", type=Path, default=PACK_PATH)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    book = None
    if not a.no_book:
        from sovereign_cash.scenarios.channels.debt_book import DebtBook
        book = DebtBook()
    pack = build(a.only, with_book=not a.no_book, book=book)
    if not a.no_check:
        pack["checks"] = check(pack, book)
    p = write(pack, a.out)
    print(f"wrote {p} countries={list(pack['countries'])} errors={pack['errors']}")
    print(json.dumps(pack.get("checks", {}), indent=1))
    return 0 if not pack["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
