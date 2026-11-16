"""Norway bottom-up cash_requirement baseline (Tab4 components).

Identity (Norges Bank LIQUIDITY_STATISTICS Tab4 → Tab1 government accounts):

    cash_requirement = gov_accounts
                     = customs_excise + vat + tax_net + nav + local_authorities
                       + oil_tax + health + sdfi + other_net

Live production inside Norges Bank's structural path is already an NB residual
(not a seasonal deficit). Beyond that path it is same-month-last-year (smly).
This module builds a Tab4 component path and a hybrid:

    cash_requirement_bottomup_bn = NB residual when NB covers the month,
                                   else Σ component forecasts

Ship rule (``_SHARED.md``): promote hybrid to ``cash_requirement_bn`` only if
mean PRE MAE over ``HORIZONS`` on **beyond-NB** target months beats production
smly by ≥ ``EDGE`` relative and is not worse at more than one horizon.
Beyond-NB PRE targets are only Januarys (NB publishes through year-end), so the
gate is thin; current data → NO-SHIP (smly is the ceiling).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sovereign_cash import contract
from sovereign_cash.no import build as B

OUT_SCORE = contract.DERIVED / "no_bottomup_score.json"
HORIZONS = (1, 3, 6, 12)
EDGE = 0.05  # relative PRE improvement required to ship
RULE_NAMES = ("smly", "mean3y", "cal5", "pace_dev", "petro_yoy")


def _get(m: pd.DataFrame, col: str, t: pd.Timestamp, asof: pd.Timestamp) -> float:
    if t not in m.index or t > asof:
        return np.nan
    return float(m[col].loc[t])


def rule_smly(m: pd.DataFrame, col: str, month: pd.Timestamp, asof: pd.Timestamp) -> float:
    return _get(m, col, month - pd.DateOffset(months=12), asof)


def rule_mean3y(m: pd.DataFrame, col: str, month: pd.Timestamp, asof: pd.Timestamp) -> float:
    xs = [_get(m, col, month - pd.DateOffset(months=12 * j), asof) for j in (1, 2, 3)]
    xs = [x for x in xs if pd.notna(x)]
    return float(np.mean(xs)) if xs else np.nan


def rule_cal5(m: pd.DataFrame, col: str, month: pd.Timestamp, asof: pd.Timestamp) -> float:
    hist = m.loc[:asof, col]
    same = hist[hist.index.month == month.month]
    if len(same) >= 3:
        return float(same.tail(5).mean())
    return rule_smly(m, col, month, asof)


def rule_pace_dev(m: pd.DataFrame, col: str, month: pd.Timestamp, asof: pd.Timestamp) -> float:
    hist = m.loc[:asof, col]
    if len(hist) < 24:
        return rule_smly(m, col, month, asof)
    pace = float(hist.tail(12).mean())
    ly = month - pd.DateOffset(months=12)
    if ly not in hist.index:
        return pace
    return float(pace + (hist.loc[ly] - hist.loc[:ly].tail(12).mean()))


def rule_petro_yoy(m: pd.DataFrame, col: str, month: pd.Timestamp, asof: pd.Timestamp) -> float:
    base = rule_smly(m, col, month, asof)
    if pd.isna(base):
        return np.nan
    r0 = m.loc[asof - pd.DateOffset(months=2) : asof, col]
    r1 = m.loc[asof - pd.DateOffset(months=14) : asof - pd.DateOffset(months=12), col]
    if len(r0) < 1 or len(r1) < 1 or abs(float(r1.mean())) < 1.0:
        return base
    return float(base * float(r0.mean()) / float(r1.mean()))


RULES: dict[str, Callable[[pd.DataFrame, str, pd.Timestamp, pd.Timestamp], float]] = {
    "smly": rule_smly,
    "mean3y": rule_mean3y,
    "cal5": rule_cal5,
    "pace_dev": rule_pace_dev,
    "petro_yoy": rule_petro_yoy,
}


def monthly_tab4(fl: pd.DataFrame) -> pd.DataFrame:
    m = fl.resample("MS").sum()
    for c in B.LINES.values():
        if c not in m.columns:
            m[c] = 0.0
    return m


def nb_covers(vin: dict, tgt: pd.Timestamp) -> bool:
    cand = [v for v in sorted(vin) if v < tgt]
    if not cand:
        return False
    end = tgt + pd.offsets.MonthEnd(0)
    return end in vin[cand[-1]][2].index


def forecast_component(m: pd.DataFrame, col: str, month: pd.Timestamp, asof: pd.Timestamp,
                       rule: str) -> float:
    return RULES[rule](m, col, month, asof)


def forecast_cash(m: pd.DataFrame, month: pd.Timestamp, asof: pd.Timestamp,
                  rule_map: dict[str, str]) -> float:
    return float(sum(forecast_component(m, c, month, asof, rule_map[c]) for c in B.TAB4))


def select_rules(m: pd.DataFrame, vin: dict) -> dict[str, str]:
    """Per-component rule with lowest pooled PRE MAE on beyond-NB PRE target months."""
    err: dict[str, dict[str, list[float]]] = {c: {r: [] for r in RULE_NAMES} for c in B.TAB4}
    origins = pd.date_range(B.PRE[0] - pd.DateOffset(months=12), B.PRE[1], freq="MS")
    for o in origins:
        for h in HORIZONS:
            tgt = o + pd.DateOffset(months=h)
            if not (B.PRE[0] <= tgt <= B.PRE[1]) or tgt not in m.index:
                continue
            if nb_covers(vin, tgt):
                continue
            for c in B.TAB4:
                act = float(m[c].loc[tgt])
                for r in RULE_NAMES:
                    p = forecast_component(m, c, tgt, o, r)
                    if pd.notna(p):
                        err[c][r].append(abs(p - act))
    out = {}
    for c in B.TAB4:
        maes = {r: float(np.mean(err[c][r])) for r in RULE_NAMES if err[c][r]}
        out[c] = min(maes, key=maes.get) if maes else "smly"
    return out


def score_beyond_nb(m: pd.DataFrame, vin: dict, rule_map: dict[str, str],
                    window: str) -> dict[int, dict]:
    """Point MAE of bottom-up vs smly on beyond-NB months; PRE uses PRE targets, OOS uses OOS targets."""
    if window == "PRE":
        lo, hi = B.PRE
    else:
        lo, hi = B.OOS_START, m.index.max()
    by_h = {h: {"bu": [], "smly": []} for h in HORIZONS}
    origins = pd.date_range(lo - pd.DateOffset(months=12), hi, freq="MS")
    for o in origins:
        for h in HORIZONS:
            tgt = o + pd.DateOffset(months=h)
            if not (lo <= tgt <= hi) or tgt not in m.index:
                continue
            if nb_covers(vin, tgt):
                continue
            act = float(m["gov_accounts"].loc[tgt])
            bu = forecast_cash(m, tgt, o, rule_map)
            sm = rule_smly(m, "gov_accounts", tgt, o)
            if pd.isna(bu) or pd.isna(sm):
                continue
            by_h[h]["bu"].append(abs(bu - act))
            by_h[h]["smly"].append(abs(sm - act))
    out = {}
    for h in HORIZONS:
        if not by_h[h]["bu"]:
            continue
        out[h] = {
            "bottom_up": round(float(np.mean(by_h[h]["bu"])), 2),
            "production_smly": round(float(np.mean(by_h[h]["smly"])), 2),
            "n": int(len(by_h[h]["bu"])),
        }
    return out


def mean_mae(score: dict[int, dict], key: str) -> float:
    vals = [score[h][key] for h in HORIZONS if h in score]
    return float(np.mean(vals)) if vals else np.nan


def ship_decision(pre: dict[int, dict]) -> tuple[bool, str]:
    if not pre or len(pre) < len(HORIZONS):
        return False, "incomplete PRE score table"
    bu = mean_mae(pre, "bottom_up")
    prod = mean_mae(pre, "production_smly")
    if not (bu < (1.0 - EDGE) * prod):
        return False, (f"PRE mean MAE bottom-up {bu:.2f} does not beat production smly {prod:.2f} "
                       f"by ≥{EDGE:.0%}")
    worse = sum(1 for h in HORIZONS if h in pre and pre[h]["bottom_up"] > pre[h]["production_smly"])
    if worse > 1:
        return False, f"worse than production at {worse} horizons (max 1 allowed)"
    return True, "PRE gate passed"


def component_mae_table(m: pd.DataFrame, vin: dict) -> dict[str, dict[str, float]]:
    err: dict[str, dict[str, list[float]]] = {c: {r: [] for r in RULE_NAMES} for c in B.TAB4}
    origins = pd.date_range(B.PRE[0] - pd.DateOffset(months=12), B.PRE[1], freq="MS")
    for o in origins:
        for h in HORIZONS:
            tgt = o + pd.DateOffset(months=h)
            if not (B.PRE[0] <= tgt <= B.PRE[1]) or tgt not in m.index or nb_covers(vin, tgt):
                continue
            for c in B.TAB4:
                act = float(m[c].loc[tgt])
                for r in RULE_NAMES:
                    p = forecast_component(m, c, tgt, o, r)
                    if pd.notna(p):
                        err[c][r].append(abs(p - act))
    return {c: {r: round(float(np.mean(xs)), 2) for r, xs in err[c].items() if xs} for c in B.TAB4}


def attach_bottomup(fw: pd.DataFrame, fl: pd.DataFrame, vin: dict, meta: dict,
                    rule_map: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Add ``cash_requirement_bottomup_bn`` on forecast rows: NB residual when covered, else Tab4 sum."""
    m = monthly_tab4(fl)
    if rule_map is None:
        rule_map = select_rules(m, vin)
    asof = fw.index.min() - pd.DateOffset(months=1)
    # last complete / partial history month available to the forward builder
    last_hist = asof
    bu = []
    for month in fw.index:
        if not bool(fw.loc[month, "scenario"]) and str(fw.loc[month, "fc_source"]).startswith("norges"):
            bu.append(float(fw.loc[month, "cash_requirement_bn"]))
        else:
            bu.append(forecast_cash(m, month, last_hist, rule_map))
    out = fw.copy()
    out["cash_requirement_bottomup_bn"] = bu
    pre = score_beyond_nb(m, vin, rule_map, "PRE")
    oos = score_beyond_nb(m, vin, rule_map, "OOS")
    ship, reason = ship_decision(pre)
    report = {
        "ship": ship,
        "ship_reason": reason,
        "edge": EDGE,
        "horizons": list(HORIZONS),
        "rule_map": rule_map,
        "component_mae_pre_beyond_nb": component_mae_table(m, vin),
        "pre_beyond_nb": {str(h): pre[h] for h in sorted(pre)},
        "oos_beyond_nb": {str(h): oos[h] for h in sorted(oos)},
        "pre_mean_mae": {
            "bottom_up": round(mean_mae(pre, "bottom_up"), 2),
            "production_smly": round(mean_mae(pre, "production_smly"), 2),
        },
        "oos_mean_mae": {
            "bottom_up": round(mean_mae(oos, "bottom_up"), 2),
            "production_smly": round(mean_mae(oos, "production_smly"), 2),
        },
        "rejects": [
            "uniform cal5 / mean3y / pace_dev on all Tab4: worse PRE mean than smly",
            "petro_yoy on oil_tax/sdfi + cal5 on rest: worse PRE mean than smly",
            "per-component PRE argmin map: PRE mean improves <5% vs smly and loses at h=3/6",
            "beating NB inside its published window: disallowed (lookahead / overlap)",
        ],
        "authority_series": [
            {"name": "Norges Bank LIQUIDITY_STATISTICS Tab4 (govt accounts specification)",
             "use": "component history for bottom-up"},
            {"name": "Norges Bank LIQUIDITY_STATISTICS Tab1 linje01 government accounts",
             "use": "headline cash_requirement actual"},
            {"name": "Norges Bank LIQUIDITY_FORECAST FCAST_S / STAT_S vintages",
             "use": "NB coverage mask + live residual inside path",
             "vintage": meta.get("vintage")},
            {"name": "Norges Bank FX transactions for the GPFG (monthly)",
             "use": "financing path only (unchanged vs production)"},
            {"name": "Norges Bank government securities primary market + debt_calendar.json",
             "use": "NGB / T-bill financing (unchanged vs production)"},
        ],
        "note": ("Beyond-NB PRE targets are January only (NB path ends 31 Dec). "
                 "Hybrid ties production inside the NB window by construction."),
    }
    return out, report


def apply_ship(fw: pd.DataFrame, report: dict) -> pd.DataFrame:
    """If ship, replace live cash_requirement with bottom-up and rebuild buffer identity."""
    out = fw.copy()
    if not report["ship"]:
        return out
    out["cash_requirement_bn"] = out["cash_requirement_bottomup_bn"]
    out["buffer_change_bn"] = (-out["cash_requirement_bn"] + out["net_bonds_bn"]
                               + out["net_bills_bn"] + out["other_fin_bn"])
    out["gov_liquidity_effect_bn"] = -out["buffer_change_bn"]
    out.loc[out["scenario"], "fc_source"] = "bottom_up_tab4"
    return out


def run_report(fl: pd.DataFrame, vin: dict, fw: pd.DataFrame, meta: dict,
               write: bool = True) -> dict:
    fw2, report = attach_bottomup(fw, fl, vin, meta)
    fw3 = apply_ship(fw2, report)
    # copy bottom-up column back for the caller even when not shipped
    fw["cash_requirement_bottomup_bn"] = fw2["cash_requirement_bottomup_bn"]
    if report["ship"]:
        for c in ("cash_requirement_bn", "buffer_change_bn", "gov_liquidity_effect_bn", "fc_source"):
            fw[c] = fw3[c]
    if write:
        contract.DERIVED.mkdir(parents=True, exist_ok=True)
        OUT_SCORE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def no_lookahead_ok(m: pd.DataFrame, asof: pd.Timestamp, rule_map: dict[str, str]) -> bool:
    """Forecast month asof+1 must equal a call with the frame truncated at asof."""
    tgt = asof + pd.DateOffset(months=1)
    if tgt not in m.index:
        return True
    full = forecast_cash(m, tgt, asof, rule_map)
    trunc = forecast_cash(m.loc[:asof], tgt, asof, rule_map)
    return abs(full - trunc) < 1e-9
