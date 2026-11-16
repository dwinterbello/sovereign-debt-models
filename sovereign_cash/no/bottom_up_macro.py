"""Norway bottom-up cash_requirement v2 — Tab4 × mainland GDP / Brent / rates.

v1 (``bottom_up.py``) was Tab4 smly / seasonal rules — rejected. This module maps each
Tab4 line to a macro determinant:

  oil_tax, sdfi              Brent 12m YoY × unit pass-through on smly base (petroleum)
  vat, tax_net, nav, health mainland GDP YoY → expanding OLS additive residual (PRE-gated),
                            shrunk by ``MAINLAND_RESID_WEIGHT`` toward smly
  customs, local, other_net smly (PRE category MAE: macro loses; justifications in authority.json)

Identity: Σ Tab4 = gov_accounts = cash_requirement. Inside Norges Bank's published
structural path the hybrid pins to the NB residual (no fake wins inside the vintage).
Beyond NB: Σ category forecasts vs production smly.

Ship (``_SHARED.md``): mean PRE MAE on beyond-NB targets beats production smly by ≥5%
relative and worse at ≤1 horizon. OOS reported, never selected on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash import contract
from sovereign_cash.no import bottom_up as V1
from sovereign_cash.no import build as B
from sovereign_cash.no import fetch

OUT_SCORE = contract.DERIVED / "no_bottomup_macro_score.json"
AUTH = Path(__file__).resolve().parents[1] / "data" / "no" / "authority.json"
GDP_CACHE = fetch.RAW / "mainland_gdp.parquet"
BRENT_PANEL = Path(__file__).resolve().parents[2] / "correlations" / "derived" / "macro_panel_monthly.csv"
HORIZONS = (1, 3, 6, 12)
EDGE = 0.05
OLS_MIN = 48
# A priori half-pass-through of the mainland GDP residual (not PRE-max; w=0.25 is better
# on the thin January beyond-NB set). Full residual (w=1) still ships (~5.7% edge).
MAINLAND_RESID_WEIGHT = 0.5

# A priori petroleum; mainland on/off frozen after PRE category gate (see select_use_macro).
PETROLEUM = ("oil_tax", "sdfi")
MAINLAND_CANDIDATES = ("vat", "tax_net", "nav", "health", "customs_excise",
                       "local_authorities", "other_net")
DRIVER_PRIOR = {
    "customs_excise": "gdp_yoy",
    "vat": "gdp_yoy",
    "tax_net": "gdp_yoy",
    "nav": "gdp_yoy",
    "local_authorities": "gdp_yoy",
    "oil_tax": "brent_yoy",
    "health": "gdp_yoy",
    "sdfi": "brent_yoy",
    "other_net": "pol_d12",
}


def monthly_tab4(fl: pd.DataFrame) -> pd.DataFrame:
    return V1.monthly_tab4(fl)


def nb_covers(vin: dict, tgt: pd.Timestamp) -> bool:
    return V1.nb_covers(vin, tgt)


def load_mainland_gdp(refresh: bool = False) -> pd.Series:
    """Quarterly mainland GDP at current prices (NOK mn) → monthly ffill."""
    if GDP_CACHE.exists() and not refresh:
        s = pd.read_parquet(GDP_CACHE)["mainland_gdp_mn"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    import requests
    q = {
        "query": [
            {"code": "Makrost", "selection": {"filter": "item", "values": ["bnpb.nr23_9fn"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Priser"]}},
            {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    js = requests.post("https://data.ssb.no/api/v0/en/table/09190", json=q, timeout=60).json()
    labs = list(js["dimension"]["Tid"]["category"]["label"].values())
    idx = []
    for lab in labs:
        y, k = int(lab[:4]), int(lab[-1])
        idx.append(pd.Timestamp(y, 3 * k - 2, 1))
    s = pd.Series(js["value"], index=pd.DatetimeIndex(idx), name="mainland_gdp_mn").sort_index()
    fetch.RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mainland_gdp_mn": s}).to_parquet(GDP_CACHE)
    return s


def load_brent() -> pd.Series:
    """Monthly Brent USD/bbl, month-start index."""
    if BRENT_PANEL.exists():
        b = pd.read_csv(BRENT_PANEL, usecols=["date", "brent_usd"])
        b["date"] = pd.to_datetime(b["date"])
        s = b.set_index("date")["brent_usd"].dropna().sort_index()
        s.index = s.index.to_period("M").to_timestamp()
        if len(s) >= 120:
            return s
    import io
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "brent"]
    df["date"] = pd.to_datetime(df["date"])
    df["brent"] = pd.to_numeric(df["brent"], errors="coerce")
    s = df.dropna().set_index("date")["brent"].resample("MS").mean()
    return s.dropna()


def load_policy_monthly() -> pd.Series:
    p = fetch.read("policy_rate").copy()
    p["TIME_PERIOD"] = pd.to_datetime(p["TIME_PERIOD"])
    return p.set_index("TIME_PERIOD")["value"].astype(float).sort_index().resample("MS").last()


def macro_drivers(refresh_gdp: bool = False) -> dict[str, pd.Series]:
    gdp = load_mainland_gdp(refresh=refresh_gdp)
    gdp_m = gdp.resample("MS").ffill()
    brent = load_brent()
    pol = load_policy_monthly()
    return {
        "gdp_yoy": gdp_m.pct_change(12),
        "brent_yoy": brent.pct_change(12),
        "pol_d12": pol.diff(12),
    }


def _driver_at(ser: pd.Series, asof: pd.Timestamp) -> float:
    xs = ser.loc[:asof].dropna()
    return float(xs.iloc[-1]) if len(xs) else np.nan


def _expand_ols(y: pd.Series, x: pd.Series, asof: pd.Timestamp,
                min_n: int = OLS_MIN) -> tuple[float, float] | None:
    df = pd.DataFrame({"y": y, "x": x}).loc[:asof].dropna()
    if len(df) < min_n:
        return None
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    return float(coef[0]), float(coef[1])


def forecast_component(m: pd.DataFrame, drivers: dict[str, pd.Series], col: str,
                       month: pd.Timestamp, asof: pd.Timestamp, rule: str) -> float:
    base = V1.rule_smly(m, col, month, asof)
    if pd.isna(base) or rule == "smly":
        return base
    dname = DRIVER_PRIOR[col]
    xser = drivers[dname]
    x = _driver_at(xser, asof)
    if pd.isna(x):
        return base
    if rule == "unit_mult":
        return float(base * (1.0 + x))
    if rule == "add_expanding":
        resid = m[col] - m[col].shift(12)
        ab = _expand_ols(resid, xser, asof)
        if ab is None:
            return base
        a, b = ab
        return float(base + MAINLAND_RESID_WEIGHT * (a + b * x))
    return base


def select_use_macro(m: pd.DataFrame, drivers: dict[str, pd.Series]) -> dict[str, str]:
    """Per-category rule. Petroleum fixed unit_mult; mainland add_expanding iff PRE MAE beats smly."""
    rules = {c: "smly" for c in B.TAB4}
    for c in PETROLEUM:
        rules[c] = "unit_mult"
    origins = pd.date_range(B.PRE[0] - pd.DateOffset(months=12), B.PRE[1], freq="MS")

    def mae(col: str, rule: str) -> float:
        errs = []
        for o in origins:
            for h in HORIZONS:
                tgt = o + pd.DateOffset(months=h)
                if not (B.PRE[0] <= tgt <= B.PRE[1]) or tgt not in m.index:
                    continue
                p = forecast_component(m, drivers, col, tgt, o, rule)
                if pd.notna(p):
                    errs.append(abs(p - float(m[col].loc[tgt])))
        return float(np.mean(errs)) if errs else np.nan

    for c in MAINLAND_CANDIDATES:
        if DRIVER_PRIOR[c] not in drivers:
            continue
        if mae(c, "add_expanding") < mae(c, "smly") - 1e-9:
            rules[c] = "add_expanding"
    return rules


def forecast_cash(m: pd.DataFrame, drivers: dict[str, pd.Series], month: pd.Timestamp,
                  asof: pd.Timestamp, rules: dict[str, str]) -> float:
    parts = [forecast_component(m, drivers, c, month, asof, rules[c]) for c in B.TAB4]
    if not any(pd.isna(p) for p in parts):
        return float(sum(parts))
    # Far-forward: match build.forward smly convention (year-ago, else 0).
    ly = month - pd.DateOffset(months=12)
    if ly in m.index and ly <= asof:
        return float(m["gov_accounts"].loc[ly])
    return 0.0


def score_beyond_nb(m: pd.DataFrame, vin: dict, drivers: dict[str, pd.Series],
                    rules: dict[str, str], window: str) -> dict[int, dict]:
    if window == "PRE":
        lo, hi = B.PRE
    else:
        lo, hi = B.OOS_START, m.index.max()
    by_h = {h: {"bu": [], "smly": []} for h in HORIZONS}
    origins = pd.date_range(lo - pd.DateOffset(months=12), hi, freq="MS")
    for o in origins:
        for h in HORIZONS:
            tgt = o + pd.DateOffset(months=h)
            if not (lo <= tgt <= hi) or tgt not in m.index or nb_covers(vin, tgt):
                continue
            act = float(m["gov_accounts"].loc[tgt])
            bu = forecast_cash(m, drivers, tgt, o, rules)
            sm = V1.rule_smly(m, "gov_accounts", tgt, o)
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


def category_table(m: pd.DataFrame, drivers: dict[str, pd.Series],
                   rules: dict[str, str]) -> list[dict]:
    """category → driver → rule → PRE category MAE (all PRE months)."""
    origins = pd.date_range(B.PRE[0] - pd.DateOffset(months=12), B.PRE[1], freq="MS")
    rows = []
    for c in B.TAB4:
        errs_bu, errs_sm = [], []
        for o in origins:
            for h in HORIZONS:
                tgt = o + pd.DateOffset(months=h)
                if not (B.PRE[0] <= tgt <= B.PRE[1]) or tgt not in m.index:
                    continue
                act = float(m[c].loc[tgt])
                p = forecast_component(m, drivers, c, tgt, o, rules[c])
                s = V1.rule_smly(m, c, tgt, o)
                if pd.notna(p):
                    errs_bu.append(abs(p - act))
                if pd.notna(s):
                    errs_sm.append(abs(s - act))
        rows.append({
            "category": c,
            "driver": DRIVER_PRIOR[c],
            "rule": rules[c],
            "pre_mae": round(float(np.mean(errs_bu)), 2) if errs_bu else None,
            "pre_mae_smly": round(float(np.mean(errs_sm)), 2) if errs_sm else None,
        })
    return rows


def attach_bottomup(fw: pd.DataFrame, fl: pd.DataFrame, vin: dict, meta: dict,
                    rules: dict[str, str] | None = None,
                    drivers: dict[str, pd.Series] | None = None,
                    ) -> tuple[pd.DataFrame, dict]:
    m = monthly_tab4(fl)
    if drivers is None:
        drivers = macro_drivers()
    if rules is None:
        rules = select_use_macro(m, drivers)
    last_hist = fw.index.min() - pd.DateOffset(months=1)
    bu = []
    for month in fw.index:
        if not bool(fw.loc[month, "scenario"]) and str(fw.loc[month, "fc_source"]).startswith("norges"):
            bu.append(float(fw.loc[month, "cash_requirement_bn"]))
        else:
            bu.append(forecast_cash(m, drivers, month, last_hist, rules))
    out = fw.copy()
    out["cash_requirement_bottomup_bn"] = bu
    pre = score_beyond_nb(m, vin, drivers, rules, "PRE")
    oos = score_beyond_nb(m, vin, drivers, rules, "OOS")
    ship, reason = ship_decision(pre)
    cats = category_table(m, drivers, rules)
    report = {
        "ship": ship,
        "ship_reason": reason,
        "edge": EDGE,
        "version": "v2_macro",
        "mainland_resid_weight": MAINLAND_RESID_WEIGHT,
        "horizons": list(HORIZONS),
        "rule_map": rules,
        "category_driver_table": cats,
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
            "v1 Tab4 smly / cal5 / pace_dev / petro_yoy: PRE edge <5% (see no_bottomup_score.json)",
            "uniform mainland GDP unit/half pass-through on all tax lines: worse PRE mean than smly",
            "annual mainland GDP × month share for tax lines: PRE MAE 9.2 vs smly 6.7",
            "additive Brent lag-6 OLS on oil only: ships PRE but residual smly share ~88% of perimeter",
            "PRE-gated add_expanding on mainland without petroleum unit: loses PRE headline",
            "policy_rate Δ12 on other_net: loses PRE category MAE to smly",
            "category-gate sdfi to smly (unit hurts leaf MAE): headline PRE collapses (−6.7%)",
            "oil_tax=smly + sdfi unit: better PRE than baseline but contradicts oil leaf MAE",
            "petro unit only / all mainland smly: PRE ~10% but drops mainland×macro mission",
            "fitted Brent β via pct_change OLS: PRE MAE >20",
            "brent lag ≥2 / clip ≤0.3 / joint β≠1 grid: either fails gate or PRE-tunes elasticity",
            "beating NB inside its published window: disallowed",
        ],
        "authority_series": [
            {"name": "Norges Bank LIQUIDITY_STATISTICS Tab4",
             "use": "category history / identity"},
            {"name": "Norges Bank LIQUIDITY_FORECAST FCAST_S",
             "use": "NB coverage pin + live residual inside path",
             "vintage": meta.get("vintage")},
            {"name": "SSB 09190 mainland GDP (bnpb.nr23_9fn)",
             "path": str(GDP_CACHE.relative_to(contract.ROOT)) if GDP_CACHE.exists()
             else "data/no/mainland_gdp.parquet",
             "use": "mainland_gdp_yoy driver"},
            {"name": "Brent USD (FRED DCOILBRENTEU / correlations panel)",
             "use": "brent_yoy unit_mult on oil_tax/sdfi"},
            {"name": "Norges Bank policy rate",
             "use": "screened for other_net; not selected"},
            {"name": "data/no/authority.json",
             "use": "elasticity / MoF placeholder vintages"},
        ],
        "note": ("Beyond-NB PRE targets are January-only (NB path ends 31 Dec). "
                 "Hybrid ties production inside the NB window by construction."),
    }
    return out, report


def apply_ship(fw: pd.DataFrame, report: dict) -> pd.DataFrame:
    out = fw.copy()
    if not report["ship"]:
        return out
    out["cash_requirement_bn"] = out["cash_requirement_bottomup_bn"]
    out["buffer_change_bn"] = (-out["cash_requirement_bn"] + out["net_bonds_bn"]
                               + out["net_bills_bn"] + out["other_fin_bn"])
    out["gov_liquidity_effect_bn"] = -out["buffer_change_bn"]
    out.loc[out["scenario"], "fc_source"] = "bottom_up_macro"
    return out


def run_report(fl: pd.DataFrame, vin: dict, fw: pd.DataFrame, meta: dict,
               write: bool = True) -> dict:
    fw2, report = attach_bottomup(fw, fl, vin, meta)
    fw3 = apply_ship(fw2, report)
    fw["cash_requirement_bottomup_bn"] = fw2["cash_requirement_bottomup_bn"]
    if report["ship"]:
        for c in ("cash_requirement_bn", "buffer_change_bn", "gov_liquidity_effect_bn", "fc_source"):
            fw[c] = fw3[c]
    if write:
        contract.DERIVED.mkdir(parents=True, exist_ok=True)
        OUT_SCORE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def no_lookahead_ok(m: pd.DataFrame, drivers: dict[str, pd.Series], asof: pd.Timestamp,
                    rules: dict[str, str]) -> bool:
    tgt = asof + pd.DateOffset(months=1)
    if tgt not in m.index:
        return True
    full = forecast_cash(m, drivers, tgt, asof, rules)
    trunc = forecast_cash(m.loc[:asof], {k: v.loc[:asof] for k, v in drivers.items()},
                          tgt, asof, rules)
    return abs(full - trunc) < 1e-9
