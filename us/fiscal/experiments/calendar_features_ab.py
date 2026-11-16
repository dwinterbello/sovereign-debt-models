"""A/B of flow-model calendar features, PRE-gated.

Variants: the pre-fix unrolled Wednesday (`old`), the SSA holiday roll (`ssa`),
and the roll plus IRS filing-season opening (`filing`) and/or a biweekly federal
payroll phase (`biweek`) on every bucket; `prod` is the roll plus biweek on
salaries only. Scores per-bucket daily MAE (monthly origins) and the
cumulative net operating flow at weekly origins.

  PYTHONPATH=. .venv/bin/python us/fiscal/experiments/calendar_features_ab.py [--variants old ssa ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

SALARIES = {"wd_salaries": ("biweek",)}
VARIANTS = {
    "old": (False, set(), {}),
    "ssa": (True, set(), {}),
    "ssa_filing": (True, {"filing_bd"}, {}),
    "ssa_biweek": (True, {"biweek"}, {}),
    "ssa_both": (True, {"filing_bd", "biweek"}, {}),
    "prod": (True, set(), SALARIES),
    "payroll": (True, set(), {"wd_salaries": ("payroll_bd",)}),
}


def run(name: str) -> dict:
    import flow_model as fm

    fm.SSA_ROLL, fm.FEATURES_ON, fm.BUCKET_EXTRA = VARIANTS[name][0], set(VARIANTS[name][1]), dict(VARIANTS[name][2])
    buckets = fm.daily_bucket_score(write=False)
    agg = fm.score(fm.backtest(verbose=False, save=False))
    return {"variant": name, "buckets": buckets.reset_index().to_dict(orient="records"),
            "agg": agg.to_dict(orient="records")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=len(args.variants)) as ex:
        res = list(ex.map(run, args.variants))
    out = HERE / "derived" / "calendar_features_ab.json"
    prev = {r["variant"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    prev.update({r["variant"]: r for r in res})
    res = list(prev.values())
    out.write_text(json.dumps(res, indent=1, default=float))

    import pandas as pd

    agg = pd.concat([pd.DataFrame(r["agg"]).assign(variant=r["variant"]) for r in res])
    print(agg.pivot_table(index=["window", "h"], columns="variant", values="model").round(2).to_string())
    b = pd.concat([pd.DataFrame(r["buckets"]).assign(variant=r["variant"]) for r in res])
    for w in ("PRE", "OOS"):
        print(f"\nper-bucket daily MAE {w} ($bn)")
        print(b.pivot_table(index="bucket", columns="variant", values=f"model_{w}").round(3).to_string())


if __name__ == "__main__":
    main()
