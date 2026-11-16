"""Nominal GDP / price LEVEL instead of YoY growth in the FR/DE/IT component models.

    PYTHONPATH=. .venv/bin/python sovereign_cash/experiments/level_spec_backtest.py

Walk-forward, every 3rd month, realised drivers for every spec alike (as the
production backtest), scored on the cumulative cash requirement at h = 1/3/6/12.
All specs are scored on the origins they share. Windows as monthly_components.mae:
PRE = windows ending by Dec-2019 (Dec-2021 for Italy), OOS = origins from Jan-2022.

Gate (decided before running): a level spec replaces "yoy" if its mean PRE MAE
across horizons is no more than 5% above yoy's and it is not worse at more than
one horizon by over 10%. The level specs are preferred on economics (receipts
proportional to the GDP level), so the gate asks only that fit is not given up.
Among passing specs, the lower PRE mean wins. OOS is reported, never selected on.
Writes derived/level_spec_backtest.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash import monthly_components as M
from sovereign_cash.yoy_components import SPEC

OUT = Path(__file__).resolve().parents[1] / "derived" / "level_spec_backtest.json"
CCS = ("fr", "de", "it")
TOL_MEAN, TOL_H = 0.05, 0.10


def elasticities(cc: str) -> dict:
    """Full-sample fits at the latest origin: beta, s.e., t vs 0 and vs 1 (loglevel)."""
    frame = M.components(cc)
    y = M.deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()
    origin = y.index.max()
    lv = M.level_drivers(cc)
    out = {}
    for name, key in SPEC[cc]:
        if key not in M.LEVEL_KEY:
            continue
        x = lv[M.LEVEL_KEY[key]].shift(M.MACRO_LAG)
        hist = frame[name].dropna().loc[:origin]
        months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=12, freq="MS")
        xx = x.reindex(hist.index.union(months)).ffill()
        f = M._fit(np.log(hist[hist > 0]), origin, 12, np.log(xx).rename(x.name)) if (hist > 0).all() else None
        if f is None or f.beta is None:
            out[name] = None
            continue
        out[name] = {"driver": key, "beta": f.beta, "se": f.beta_se,
                     "t_vs_0": f.beta / f.beta_se if f.beta_se else None,
                     "t_vs_1": (f.beta - 1.0) / f.beta_se if f.beta_se else None,
                     "nobs": f.nobs}
    return out


def run(cc: str) -> dict:
    bts = {s: M.backtest(cc, step=3, spec=s) for s in M.SPECS}
    keys = [set(map(tuple, b[["origin", "h"]].astype(str).to_numpy())) for b in bts.values() if not b.empty]
    common = set.intersection(*keys) if keys else set()
    scores = {}
    for s, bt in bts.items():
        if bt.empty:
            scores[s] = None
            continue
        k = bt[["origin", "h"]].astype(str).apply(tuple, axis=1)
        sub = bt[k.isin(common)]
        m = M.mae(sub, cc)
        scores[s] = {w: {int(r["h"]): float(r["sarimax"]) for _, r in g.iterrows()}
                     for w, g in m.groupby("window")}
        scores[s]["n_origins"] = int(sub["origin"].nunique())
    base = scores["yoy"]["PRE"]
    verdict = {}
    for s in ("loglevel", "ratio"):
        if not scores.get(s) or "PRE" not in scores[s]:
            verdict[s] = {"pass": False, "why": "no PRE origins"}
            continue
        pre = scores[s]["PRE"]
        hs = sorted(set(pre) & set(base))
        rel = np.mean([pre[h] for h in hs]) / np.mean([base[h] for h in hs]) - 1.0
        worse = [h for h in hs if pre[h] > base[h] * (1 + TOL_H)]
        verdict[s] = {"pre_mean_vs_yoy": float(rel), "worse_by_10pct_at": worse,
                      "pass": bool(rel <= TOL_MEAN and len(worse) <= 1)}
    passing = [s for s, v in verdict.items() if v["pass"]]
    if passing:
        mean = {s: np.mean(list(scores[s]["PRE"].values())) for s in passing}
        chosen = min(passing, key=mean.get)
    else:
        chosen = "yoy"
    return {"scores": scores, "verdict": verdict, "chosen": chosen, "elasticities": elasticities(cc)}


def main() -> int:
    res = {}
    for cc in CCS:
        try:
            res[cc] = run(cc)
        except Exception as exc:  # noqa: BLE001
            res[cc] = {"error": f"{type(exc).__name__}: {exc}"}
        print(cc, json.dumps(res[cc], indent=1, default=float)[:4000], flush=True)
    res["_gate"] = {"tol_mean": TOL_MEAN, "tol_h": TOL_H, "rule": __doc__.split("Gate")[1].split("Writes")[0].strip()}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
