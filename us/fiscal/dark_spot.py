"""The repo "dark spot": primary-dealer uncleared bilateral Treasury repo, and what clearing costs.

FR 2004 (from Jan-2022) splits dealer financing by settlement venue:
  UB    uncleared bilateral (general + specified): the segment no public rate covers
  CB    cleared bilateral via FICC (DVP), of which CBSP sponsored
  GCF   FICC general collateral finance
  TRI   tri-party (incl. sponsored tri-party)
Reverse repo (PDSIRRA) is dealers lending cash against Treasuries (to hedge funds,
mostly); repo (PDSORA) is dealers borrowing cash. Overnight/continuing plus term,
Treasuries ex-TIPS plus TIPS.

The SEC clearing mandate moves eligible repo into FICC (repo by 30-Jun-2027 after
the 2025 extension). Sizing:
  margin     FICC initial margin on cleared repo the bilateral book mostly does not
             post today (OFR's 2022 NCCBR pilot: most hedge-fund repo at zero haircut):
             IM = rate × migrating reverse repo
  rate cost  IM funded at the borrower's spread over the cash it earns at FICC:
             Δrepo ≈ IM rate × funding spread (bp)
  netting    cleared repo and reverse repo net on dealer balance sheets: up to
             min(UB repo, UB reverse repo) of gross leverage-exposure relief

  PYTHONPATH=. .venv/bin/python us/fiscal/dark_spot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DERIVED = HERE / "derived"
NYFED = HERE.parents[1] / "flow" / "output" / "nyfed.parquet"
VENUES = {"UB": ("UBG", "UBS"), "CB": ("CBG", "CBS"), "CBSP": ("CBSP",), "GCF": ("GCF",),
          "TRI": ("TRI", "TRIG", "TRISP")}
COLLATERAL = ("UTSET", "UTST")
TERMS = ("", "TAL30", "TAG30")
SIDES = {"rev": "PDSIRRA", "repo": "PDSORA"}
IM_RATES = (0.01, 0.02, 0.04, 0.06)
FUND_SPREADS_BP = (50.0, 100.0, 150.0)
MIGRATION = (0.5, 1.0)  # share of the uncleared reverse repo book that moves


def venue_panel() -> pd.DataFrame:
    n = pd.read_parquet(NYFED, columns=["date", "keyid", "value"])
    n["date"] = pd.to_datetime(n["date"])
    piv = n.pivot_table(index="date", columns="keyid", values="value", aggfunc="last")
    out = pd.DataFrame(index=piv.index)
    for side, root in SIDES.items():
        for v, codes in VENUES.items():
            keys = [f"{root}-{c}{col}{t}" for c in codes for col in COLLATERAL for t in TERMS]
            keys = [k for k in keys if k in piv.columns]
            out[f"{side}_{v}"] = piv[keys].sum(axis=1, min_count=1) / 1000.0 if keys else np.nan
    out = out.loc["2022-01-01":].dropna(subset=["rev_UB"])
    for side in SIDES:
        out[f"{side}_sum"] = out[[f"{side}_{v}" for v in VENUES]].sum(axis=1)
        out[f"{side}_UB_share"] = out[f"{side}_UB"] / out[f"{side}_sum"]
    return out


def sizing(ub_rev_bn: float) -> list[dict]:
    rows = []
    for mig in MIGRATION:
        for im in IM_RATES:
            for s in FUND_SPREADS_BP:
                rows.append({"migration": mig, "im_rate": im, "fund_spread_bp": s,
                             "margin_bn": mig * ub_rev_bn * im, "repo_cost_bp": im * s})
    return rows


def run(verbose: bool = True) -> dict:
    p = venue_panel()
    p.to_parquet(DERIVED / "dark_spot_venues.parquet")
    last = p.iloc[-1]
    yr = p[p.index <= p.index[-1] - pd.DateOffset(years=1)]
    ago = yr.iloc[-1] if len(yr) else p.iloc[0]
    out = {
        "asof": str(p.index[-1].date()),
        "levels_bn": {k: float(last[k]) for k in p.columns if not k.endswith("share")},
        "ub_share": {s: float(last[f"{s}_UB_share"]) for s in SIDES},
        "ub_share_1y_ago": {s: float(ago[f"{s}_UB_share"]) for s in SIDES},
        "ub_share_first": {s: float(p[f"{s}_UB_share"].iloc[0]) for s in SIDES},
        "netting_capacity_bn": float(min(last["rev_UB"], last["repo_UB"])),
        "sizing": sizing(float(last["rev_UB"])),
        "central": {"im_rate": 0.02, "fund_spread_bp": 100.0, "migration": 1.0,
                    "margin_bn": float(last["rev_UB"]) * 0.02, "repo_cost_bp": 2.0},
    }
    (DERIVED / "dark_spot.json").write_text(json.dumps(out, indent=1))
    if verbose:
        q = p.resample("QE").last()
        print(q[[c for c in p.columns if c.startswith("rev_")]].round(0).to_string())
        print(q[[c for c in p.columns if c.startswith("repo_")]].round(0).to_string())
        print(json.dumps({k: v for k, v in out.items() if k != "sizing"}, indent=1, default=float))
    return out


if __name__ == "__main__":
    run()
