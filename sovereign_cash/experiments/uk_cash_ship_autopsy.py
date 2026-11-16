"""UK monthly cash bottom-up ship autopsy.

Re-scores ``uk/bottom_up_macro`` under interest_mode ∈ {isin_curve, rr} × soft_pin_alpha
∈ {0.12, 0} against the production hybrid on the module's own PRE/OOS windows, audits
the interest leaf and the per-category error, and compares with the numbers the live
``derived/uk_cash_summary.json`` carries.

Writes ``uk_cash_ship_autopsy.json`` + ``uk_cash_ship_autopsy.md`` next to this file.
No production code is touched.

    PYTHONPATH=. .venv/bin/python sovereign_cash/experiments/uk_cash_ship_autopsy.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.uk import bottom_up_macro as BU
from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "uk_cash_ship_autopsy.json"
OUT_MD = HERE / "uk_cash_ship_autopsy.md"
SUMMARY = HERE.parent / "derived" / "uk_cash_summary.json"

# Sign of each category inside CGNCR = spend − receipts + fin parts.
SIGN = {
    "welfare": 1, "rdel_proxy": 1, "interest": 1, "cap": 1,
    "it": -1, "nics": -1, "vat": -1, "ct": -1, "other_tax": -1, "nontax": -1,
    "recv_pay": 1, "gilt_adj": 1, "other_fin": 1, "acq_sec": 1, "apf_xfer": 1, "net_lend": 1,
    "resid_gap": 1,
}
CATS = list(SIGN)

VARIANTS = {
    "isin_a12": {"interest_mode": "isin_curve", "soft_pin_alpha": 0.12},
    "rr_a12": {"interest_mode": "rr", "soft_pin_alpha": 0.12},
    "isin_a0": {"interest_mode": "isin_curve", "soft_pin_alpha": 0.0},
    "rr_a0": {"interest_mode": "rr", "soft_pin_alpha": 0.0},
    # Exact Oct-27 story spec (pre-commit 80f0b77c: no SA nudge, RR interest) on today's data.
    "rr_a12_nosa": {"interest_mode": "rr", "soft_pin_alpha": 0.12, "sa_equity_nudge": False},
    "rr_a0_nosa": {"interest_mode": "rr", "soft_pin_alpha": 0.0, "sa_equity_nudge": False},
}


def _r(x, n=3):
    if x is None:
        return None
    if isinstance(x, dict):
        return {str(k): _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return x


def gate(bt: pd.DataFrame) -> dict:
    d = BU.ship_rule(bt)
    keep = ("ship", "edge", "worse_horizons", "bu_pre", "prod_pre", "bu_pre_mean",
            "prod_pre_mean", "bu_oos", "prod_oos", "bu_oos_mean", "prod_oos_mean", "cg_method")
    out = {k: d[k] for k in keep}
    out["oos_edge"] = ((d["prod_oos_mean"] - d["bu_oos_mean"]) / d["prod_oos_mean"]
                       if d.get("prod_oos_mean") else float("nan"))
    # signed bias
    for w in ("PRE", "OOS"):
        sub = bt[(bt["window"] == w) & (bt["model"] == BU.BOTTOMUP)].dropna(subset=["abs_err"])
        out[f"bu_{w.lower()}_bias"] = {int(h): float((g["fc"] - g["actual"]).mean())
                                       for h, g in sub.groupby("h")}
        n = {int(h): int(len(g)) for h, g in sub.groupby("h")}
        out[f"n_{w.lower()}"] = n
    return out


def per_year(bt: pd.DataFrame, h: int, window: str) -> dict:
    sub = bt[(bt["window"] == window) & (bt["h"] == h)
             & bt["model"].isin([BU.BOTTOMUP, BU.PRODUCTION])].dropna(subset=["abs_err"])
    piv = sub.pivot_table(index="origin", columns="model", values="abs_err")
    piv["year"] = piv.index.year
    g = piv.groupby("year")[[BU.BOTTOMUP, BU.PRODUCTION]].mean()
    g["n"] = piv.groupby("year").size()
    return {int(y): {"bu": float(r[BU.BOTTOMUP]), "prod": float(r[BU.PRODUCTION]), "n": int(r["n"])}
            for y, r in g.iterrows()}


def category_errors(comp: pd.DataFrame, spec: dict, window: str) -> dict:
    """Per-category monthly MAE (h=1..12 pooled) and signed 12m-cumulative contribution to the
    CGNCR identity error, on the origins the backtest scores in ``window`` at h=12."""
    if window == "PRE":
        lo, hi = M.PRE
    else:
        lo, hi = M.OOS_START, comp.index.max()
    full = {**BU.SPEC, **spec}
    mae = {k: [] for k in CATS}
    cum = {k: [] for k in CATS}
    h1 = {k: [] for k in CATS}
    n = 0
    for o in comp.index[(comp.index >= lo) & (comp.index <= hi)]:
        fut = pd.date_range(o + pd.offsets.MonthBegin(1), periods=12, freq="MS")
        if fut[-1] not in comp.index:
            continue
        if window == "PRE" and fut[-1] > M.PRE[1] + pd.DateOffset(months=12):
            continue
        # Covid-free requirement as in M._window (inputs and targets)
        if (fut[-1] >= M.COVID_EST[0] and o < M.COVID_EST[1]) or \
           (o >= M.COVID_EST[0] and o - pd.DateOffset(months=M.INPUT_MONTHS - 1) <= M.COVID_EST[1]):
            continue
        paths = BU.category_paths(comp, o, h=12, spec=full)
        n += 1
        for k in CATS:
            act = comp.loc[fut, k].to_numpy(dtype=float)
            fc = np.asarray(paths[k], dtype=float)
            mae[k].append(float(np.nanmean(np.abs(fc - act))))
            h1[k].append(float(abs(fc[0] - act[0])))
            cum[k].append(SIGN[k] * float(np.nansum(fc - act)))
    rows = []
    for k in CATS:
        c = np.array(cum[k])
        rows.append({
            "category": k, "sign": SIGN[k],
            "monthly_mae": float(np.mean(mae[k])),
            "h1_mae": float(np.mean(h1[k])),
            "cum12_abs_contrib": float(np.mean(np.abs(c))),
            "cum12_signed_bias": float(np.mean(c)),
        })
    df = pd.DataFrame(rows).sort_values("cum12_abs_contrib", ascending=False)
    tot = float(df["monthly_mae"].sum())
    df["share_monthly_mae"] = df["monthly_mae"] / tot
    return {"n_origins": n, "rows": df.to_dict(orient="records")}


def interest_leaf(comp: pd.DataFrame, window: str) -> dict:
    """Interest leaf alone: isin_curve vs rr vs calendar (smly), monthly MAE by horizon class
    and 12m-cum MAE / bias on the same origins."""
    from sovereign_cash import isin_curve_interest as IC

    if window == "PRE":
        lo, hi = M.PRE
    else:
        lo, hi = M.OOS_START, comp.index.max()
    s = comp["interest"]
    drv = BU._drivers()
    res = {"isin_curve": [], "rr": [], "cal": []}
    cum = {"isin_curve": [], "rr": [], "cal": []}
    byh = {k: {h: [] for h in (1, 3, 6, 12)} for k in res}
    per_origin = []
    for o in comp.index[(comp.index >= lo) & (comp.index <= hi)]:
        fut = pd.date_range(o + pd.offsets.MonthBegin(1), periods=12, freq="MS")
        if fut[-1] not in comp.index:
            continue
        if (fut[-1] >= M.COVID_EST[0] and o < M.COVID_EST[1]) or \
           (o >= M.COVID_EST[0] and o - pd.DateOffset(months=M.INPUT_MONTHS - 1) <= M.COVID_EST[1]):
            continue
        act = s.loc[fut].to_numpy(dtype=float)
        gilt = BU._yoy_at(drv.get("gilt_dpp"), o)
        br = BU._yoy_at(drv.get("bank_rate_dpp"), o)
        paths = {
            "rr": BU._interest_triangle(s, o, 12, gilt, br),
            "isin_curve": IC.monthly_path("uk", s, o, 12),
            "cal": BU._grow(s, o, 12, 0.0, 0.0),
        }
        row = {"origin": str(o.date()), "actual12": float(act.sum())}
        for k, p in paths.items():
            p = np.asarray(p, dtype=float)
            res[k].append(float(np.mean(np.abs(p - act))))
            cum[k].append(float(p.sum() - act.sum()))
            for h in (1, 3, 6, 12):
                byh[k][h].append(float(abs(p[:h].sum() - act[:h].sum())))
            row[f"{k}_12m"] = float(p.sum())
        per_origin.append(row)
    out = {"n_origins": len(per_origin)}
    for k in res:
        out[k] = {
            "monthly_mae": float(np.mean(res[k])),
            "cum_mae_by_h": {h: float(np.mean(v)) for h, v in byh[k].items()},
            "cum12_bias": float(np.mean(cum[k])),
            "cum12_mae": float(np.mean(np.abs(cum[k]))),
        }
    # year cut on 12m cum error for isin vs rr
    po = pd.DataFrame(per_origin)
    po["year"] = pd.to_datetime(po["origin"]).dt.year
    yc = {}
    for y, g in po.groupby("year"):
        yc[int(y)] = {
            "isin_cum12_err": float((g["isin_curve_12m"] - g["actual12"]).mean()),
            "rr_cum12_err": float((g["rr_12m"] - g["actual12"]).mean()),
            "cal_cum12_err": float((g["cal_12m"] - g["actual12"]).mean()),
            "actual12": float(g["actual12"].mean()),
            "n": int(len(g)),
        }
    out["by_origin_year"] = yc
    return out


def isin_engine_facts() -> dict:
    """Years with a finite PIT annual, and the information-date mismatch of ``monthly_path``."""
    from sovereign_cash import isin_curve_interest as IC

    ann = {y: IC.annual_bn("uk", y) for y in range(2010, 2027)}
    return {
        "annual_bn_by_year": ann,
        "first_finite_year": min((y for y, v in ann.items() if np.isfinite(v)), default=None),
        "notes": [
            "annual_bn(uk, Y) is the calendar-year-Y forecast made asof 31 Dec Y-1 (published I0 for "
            "Y-1 required): for an origin in Jan..Nov of Y-1 it is lookahead; for months of the "
            "origin's own year it is up to 11 months stale (no in-year JW2P print enters).",
            "Years with NaN annual fall back to same-month-last-year × 12 × month share, which "
            "with JW2P's seasonal profile is a distorted calendar carry (PRE targets 2012-2015).",
            "Scale = last complete CY actual / annual_bn(last CY): last year's forecast error is "
            "carried as a multiplicative correction onto this year's level.",
        ],
    }


def production_facts(comp: pd.DataFrame, vint: pd.DataFrame) -> dict:
    x = comp["cgncr"]
    bt_base = M.backtest_cgncr(x, vint)
    t = M.score_table(bt_base)
    cg_method = M.choose(t, M.CGNCR_MODELS)
    paired = M.official_paired(bt_base, cg_method)
    pm = M.paired_mae(paired, "PRE", cg_method)
    pp = M.paired_mae(paired, "PRE", M.OFFICIAL)
    beats = {h: M.model_beats(pm, pp, h) for h in M.HORIZONS}
    return {
        "cg_method": cg_method,
        "paired_pre_model": pm, "paired_pre_official": pp,
        "model_beats_official_at_h": beats,
        "pre_mae_all_models": {m: M.table_mae(t, "PRE", m) for m in M.CGNCR_MODELS + M.CGNCR_BENCH},
        "oos_mae_all_models": {m: M.table_mae(t, "OOS", m) for m in M.CGNCR_MODELS + M.CGNCR_BENCH},
        "cgncr_last_month": str(x.dropna().index.max().date()),
        "n_remit_vintages": int(len(vint)),
        "last_vintage_date": str(pd.to_datetime(vint["vintage_date"]).max().date()),
    }


def live_summary() -> dict:
    if not SUMMARY.exists():
        return {}
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    f = d.get("forward", {})
    keep = {k: f.get(k) for k in f if k.startswith("cgncr_bottomup") and k != "cgncr_bottomup_spec"}
    keep["asof"] = d.get("asof")
    keep["spec_interest_mode"] = (f.get("cgncr_bottomup_spec") or {}).get("interest_mode")
    keep["spec_alpha"] = (f.get("cgncr_bottomup_spec") or {}).get("soft_pin_alpha")
    keep["skill_note"] = next((s for s in d.get("skill_notes", []) if "Bottom-up" in s), None)
    keep["file_mtime"] = str(pd.Timestamp(SUMMARY.stat().st_mtime, unit="s").round("s"))
    return keep


def main() -> None:
    t0 = time.time()
    comp = BU.components()
    vint = src.remit_vintages()
    out: dict = {
        "scored_at": str(pd.Timestamp.now().round("s")),
        "module_flags": {"SHIP": BU.SHIP, "FORCE_LIVE": BU.FORCE_LIVE,
                         "INTEREST_ISIN_CURVE": BU.INTEREST_ISIN_CURVE,
                         "SPEC_interest_mode": BU.SPEC["interest_mode"],
                         "SPEC_alpha": BU.SPEC["soft_pin_alpha"], "EDGE_BAND": M.EDGE_BAND},
        "windows": {"PRE_origins": [str(M.PRE[0].date()), str(M.PRE[1].date())],
                    "OOS_origins_from": str(M.OOS_START.date()),
                    "covid_excluded": [str(M.COVID_EST[0].date()), str(M.COVID_EST[1].date())]},
        "live_summary": live_summary(),
        "production": production_facts(comp, vint),
    }
    gates = {}
    bts = {}
    for name, sp in VARIANTS.items():
        t = time.time()
        bt = BU.backtest(comp=comp, vint=vint, spec=sp)
        bts[name] = bt
        gates[name] = gate(bt)
        gates[name]["spec_delta"] = sp
        gates[name]["per_year_pre_h12"] = per_year(bt, 12, "PRE")
        gates[name]["per_year_oos_h12"] = per_year(bt, 12, "OOS")
        gates[name]["secs"] = round(time.time() - t, 1)
        print(f"{name}: PRE bu {gates[name]['bu_pre_mean']:.3f} prod {gates[name]['prod_pre_mean']:.3f} "
              f"edge {100 * gates[name]['edge']:.1f}% worse {gates[name]['worse_horizons']} "
              f"OOS bu {gates[name]['bu_oos_mean']:.3f} prod {gates[name]['prod_oos_mean']:.3f} "
              f"[{gates[name]['secs']}s]", flush=True)
    out["gates"] = gates
    # Also the production-only series from the first bt (identical across variants).
    out["gate_fresh_vs_live"] = {
        "fresh_isin_a12": {k: gates["isin_a12"][k] for k in ("bu_pre", "prod_pre", "edge", "ship",
                                                            "worse_horizons", "bu_oos", "prod_oos")},
        "live": out["live_summary"],
    }
    # Paired per-origin delta isin vs rr on PRE h=12 (same origins)
    a = bts["isin_a12"]; b = bts["rr_a12"]
    pa = a[(a["window"] == "PRE") & (a["model"] == BU.BOTTOMUP)].set_index(["origin", "h"])["abs_err"]
    pb = b[(b["window"] == "PRE") & (b["model"] == BU.BOTTOMUP)].set_index(["origin", "h"])["abs_err"]
    d = (pa - pb).dropna()
    out["isin_minus_rr_pre_abs_err"] = {
        int(h): {"mean": float(g.mean()), "share_origins_isin_worse": float((g > 0).mean()), "n": int(len(g))}
        for h, g in d.groupby(level="h")}

    print("category audit ...", flush=True)
    out["category"] = {
        "isin_pre": category_errors(comp, VARIANTS["isin_a12"], "PRE"),
        "rr_pre": category_errors(comp, VARIANTS["rr_a12"], "PRE"),
        "isin_oos": category_errors(comp, VARIANTS["isin_a12"], "OOS"),
        "rr_oos": category_errors(comp, VARIANTS["rr_a12"], "OOS"),
    }
    print("interest leaf ...", flush=True)
    out["interest_leaf"] = {"PRE": interest_leaf(comp, "PRE"), "OOS": interest_leaf(comp, "OOS")}
    out["isin_engine"] = isin_engine_facts()
    out["secs_total"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(_r(out), indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT_JSON} in {out['secs_total']}s")


if __name__ == "__main__":
    main()
