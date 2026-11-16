"""Sweden bottom-up NBR baseline from CH1089 components.

Identity (Riksgälden CH1089, NBR sign + = borrowing):

    nbr = nbr_primary + nbr_net_lending + nbr_interest
    budget_balance = −nbr
    cash_requirement = nbr

Live production = PRE-selected seasonal on the budget balance, then rest-of-year
anchored to Riksgälden's annual total (``anchor_to_riksgalden``). This module
forecasts the three published components with independent PRE-gated seasonal
rules, sums them, and exposes ``cash_requirement_bottomup_bn``.

Ship rule (``_SHARED.md`` + ``PROMPT_SE.md``): promote to live only if the
**RG-anchored** bottom-up shape beats the **RG-anchored** production shape on
mean PRE MAE over ``SCORE_H`` by ≥ ``EDGE`` relative and is not worse at more
than one horizon. Unanchored component skill is reported; it does not alone
authorize a live switch (the desk path is seasonal + authority annual).

**v1 status:** SHIP=False (anchored PRE fails). Macro category path lives in
``bottom_up_macro.py`` (v2); this module is retained for the seasonal baseline score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash import contract
from sovereign_cash.se import build as B
from sovereign_cash.se import model as M

OUT_SCORE = contract.DERIVED / "se_bottomup_score.json"
AUTH = Path(__file__).resolve().parents[1] / "data" / "se" / "authority.json"
COMPONENTS = ("nbr_primary", "nbr_net_lending", "nbr_interest")
EDGE = 0.05
SCORE_H = M.SCORE_H
ORIGIN_START = pd.Timestamp("2010-12-01")
ANCHOR_ORIGIN_MONTHS = (6, 7, 8, 9)


def load_authority() -> dict:
    return json.loads(AUTH.read_text()) if AUTH.exists() else {}


def components_bn(nbr: pd.DataFrame) -> dict[str, pd.Series]:
    """CH1089 monthly frame → NBR-sign series in bn SEK."""
    idx = nbr.set_index("month") if "month" in nbr.columns else nbr
    out = {c: idx[c] / 1000.0 for c in COMPONENTS}
    out["nbr"] = idx["nbr"] / 1000.0
    out["budget_balance"] = idx["budget_balance"] / 1000.0
    return out


def identity_resid_bn(comps: dict[str, pd.Series]) -> float:
    s = comps["nbr_primary"] + comps["nbr_net_lending"] + comps["nbr_interest"]
    return float((comps["nbr"] - s).abs().max())


def _comp_mae(series: pd.Series, k: int, a: float, f: str) -> float:
    rows = []
    for o in [o for o in series.index if o >= ORIGIN_START and o < series.index[-1]]:
        fc = M.forecast(series, o, k, a, f, hmax=12)
        for t, v in fc.items():
            if t in series.index:
                h = (t.year - o.year) * 12 + t.month - o.month
                rows.append((t, h, abs(v - series[t])))
    if not rows:
        return np.inf
    bt = pd.DataFrame(rows, columns=["target", "h", "ae"])
    bt["win"] = M.window(bt["target"])
    s = bt[(bt["win"] == "PRE") & bt["h"].isin(SCORE_H)]
    if s.empty:
        return np.inf
    return float(s.groupby("h")["ae"].mean().mean())


def select_component_params(comps: dict[str, pd.Series]) -> dict[str, tuple[str, int, float, str]]:
    """Per-component seasonal rule with lowest mean PRE MAE over SCORE_H."""
    out = {}
    for c in COMPONENTS:
        best_name, best_p, best_mae = "smly", (1, 0.0, "mean"), np.inf
        for name, k, a, f in M.variants():
            mae = _comp_mae(comps[c], k, a, f)
            if mae < best_mae:
                best_name, best_p, best_mae = name, (k, a, f), mae
        k, a, f = best_p
        out[c] = (best_name, k, a, f)
    return out


def params_tuple(selected: dict[str, tuple[str, int, float, str]]) -> dict[str, tuple[int, float, str]]:
    return {c: (k, a, f) for c, (_, k, a, f) in selected.items()}


def forecast_nbr(comps: dict[str, pd.Series], params: dict[str, tuple[int, float, str]],
                 origin: pd.Timestamp, hmax: int = 12) -> pd.Series:
    s = None
    for c in COMPONENTS:
        part = M.forecast(comps[c], origin, *params[c], hmax=hmax)
        s = part if s is None else s.add(part, fill_value=np.nan)
    return s


def forecast_balance(comps: dict[str, pd.Series], params: dict[str, tuple[int, float, str]],
                     origin: pd.Timestamp, hmax: int = 12) -> pd.Series:
    return -forecast_nbr(comps, params, origin, hmax=hmax)


def component_mae_table(comps: dict[str, pd.Series],
                        selected: dict[str, tuple[str, int, float, str]] | None = None) -> dict:
    """PRE component MAE for smly and the PRE-selected rule (plus one shared seasonal)."""
    shared = {
        "smly": (1, 0.0, "mean"),
        "seas_k5_a0.5_mean": (5, 0.5, "mean"),
    }
    tab = {}
    for c in COMPONENTS:
        row = {lab: round(_comp_mae(comps[c], *p), 3) for lab, p in shared.items()}
        if selected:
            name, k, a, f = selected[c]
            row["chosen"] = round(_comp_mae(comps[c], k, a, f), 3)
            row["chosen_name"] = name
        tab[c] = row
    return tab


def _headline_bt(y: pd.Series, fc_fn) -> pd.DataFrame:
    rows = []
    for o in [o for o in y.index if o >= ORIGIN_START and o < y.index[-1]]:
        fc = fc_fn(o, 12)
        for t, v in fc.items():
            if t in y.index:
                h = (t.year - o.year) * 12 + t.month - o.month
                rows.append((o, t, h, float(v), float(y[t])))
    bt = pd.DataFrame(rows, columns=["origin", "target", "h", "fc", "actual"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    bt["win"] = M.window(bt["target"])
    return bt


def score_unanchored(y: pd.Series, comps: dict[str, pd.Series],
                     params: dict[str, tuple[int, float, str]],
                     prod_name: str) -> dict[str, dict]:
    """Unanchored budget-balance MAE: bottom-up vs production seasonal."""
    k, a, f = M.choose_params(prod_name)
    bt_bu = _headline_bt(y, lambda o, h: forecast_balance(comps, params, o, hmax=h))
    bt_pr = _headline_bt(y, lambda o, h: M.forecast(y, o, k, a, f, hmax=h))
    out = {}
    for win in ("PRE", "OOS"):
        rows = {}
        for h in SCORE_H:
            bu = bt_bu[(bt_bu["win"] == win) & (bt_bu["h"] == h)]["ae"]
            pr = bt_pr[(bt_pr["win"] == win) & (bt_pr["h"] == h)]["ae"]
            if bu.empty or pr.empty:
                continue
            rows[h] = {
                "bottom_up": round(float(bu.mean()), 3),
                "production": round(float(pr.mean()), 3),
                "n": int(len(bu)),
            }
        out[win] = rows
    return out


def _rg_anchor_bt(y: pd.Series, rgm: pd.Series, raw_fn, method: str) -> pd.DataFrame:
    rows = []
    for o in y.index:
        if o.month not in ANCHOR_ORIGIN_MONTHS:
            continue
        ms = pd.date_range(o + pd.DateOffset(months=1), pd.Timestamp(o.year, 12, 1), freq="MS")
        if not all(m in rgm.index and m in y.index for m in ms):
            continue
        raw = raw_fn(o, len(ms)).reindex(ms)
        tot = float(rgm.reindex(ms).sum())
        al = M.allocate(raw, tot, method, y, o)
        for t in ms:
            h = (t.year - o.year) * 12 + t.month - o.month
            rows.append((o, t, h, float(al[t]), float(y[t])))
    bt = pd.DataFrame(rows, columns=["origin", "target", "h", "fc", "actual"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    bt["win"] = M.window(bt["target"])
    return bt


def score_anchored(y: pd.Series, comps: dict[str, pd.Series],
                   params: dict[str, tuple[int, float, str]],
                   prod_name: str, method: str, rgm: pd.Series) -> dict[str, dict]:
    """Live-path gate: both shapes anchored to the same Riksgälden rest-of-year total."""
    k, a, f = M.choose_params(prod_name)
    bt_bu = _rg_anchor_bt(y, rgm, lambda o, h: forecast_balance(comps, params, o, hmax=h), method)
    bt_pr = _rg_anchor_bt(y, rgm, lambda o, h: M.forecast(y, o, k, a, f, hmax=h), method)
    out = {}
    for win in ("PRE", "OOS"):
        rows = {}
        for h in SCORE_H:
            bu = bt_bu[(bt_bu["win"] == win) & (bt_bu["h"] == h)]["ae"]
            pr = bt_pr[(bt_pr["win"] == win) & (bt_pr["h"] == h)]["ae"]
            if bu.empty and pr.empty:
                continue
            rows[h] = {
                "bottom_up": round(float(bu.mean()), 3) if len(bu) else None,
                "production": round(float(pr.mean()), 3) if len(pr) else None,
                "n": int(max(len(bu), len(pr))),
            }
        out[win] = rows
    return out


def mean_mae(score: dict[int, dict], key: str) -> float:
    vals = [score[h][key] for h in SCORE_H if h in score and score[h].get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def ship_decision(pre_anchored: dict[int, dict]) -> tuple[bool, str]:
    """Gate on RG-anchored PRE (the live path), not unanchored component skill."""
    if not pre_anchored or len(pre_anchored) < 2:
        return False, "incomplete PRE anchored score table"
    bu = mean_mae(pre_anchored, "bottom_up")
    prod = mean_mae(pre_anchored, "production")
    if not np.isfinite(bu) or not np.isfinite(prod):
        return False, "non-finite PRE anchored MAE"
    if not (bu < (1.0 - EDGE) * prod):
        return False, (f"PRE anchored mean MAE bottom-up {bu:.3f} does not beat production "
                       f"{prod:.3f} by ≥{EDGE:.0%}")
    worse = sum(1 for h in SCORE_H if h in pre_anchored
                and pre_anchored[h].get("bottom_up") is not None
                and pre_anchored[h].get("production") is not None
                and pre_anchored[h]["bottom_up"] > pre_anchored[h]["production"])
    if worse > 1:
        return False, f"worse than production at {worse} anchored horizons (max 1 allowed)"
    return True, "PRE anchored gate passed"


def forward_bottomup(comps: dict[str, pd.Series], params: dict[str, tuple[int, float, str]],
                     y: pd.Series, origin: pd.Timestamp, months: pd.DatetimeIndex,
                     method: str, carry: float, annual: dict | None = None) -> pd.Series:
    """Unanchored component path; optionally RG-anchored (same helper as production)."""
    hmax = int((months.max().year - origin.year) * 12 + months.max().month - origin.month)
    raw = forecast_balance(comps, params, origin, hmax=max(hmax, 12))
    raw = raw.reindex(months)
    return B.anchor_to_riksgalden(raw, y, origin, method, carry, annual)


def attach_column(df: pd.DataFrame, comps: dict[str, pd.Series],
                  params: dict[str, tuple[int, float, str]], y: pd.Series) -> pd.Series:
    """History = outcome NBR; forecast = unanchored bottom-up NBR (alternate column)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    hist = ~df["is_forecast"]
    out.loc[hist] = df.loc[hist, "cash_requirement_bn"].to_numpy()
    origin = y.index[-1]
    fut = df.loc[df["is_forecast"], "month"]
    if len(fut):
        raw = forecast_nbr(comps, params, origin, hmax=len(fut) + 2)
        out.loc[df["is_forecast"]] = raw.reindex(fut).to_numpy()
    return out


def apply_ship(df: pd.DataFrame, report: dict, y: pd.Series, comps: dict[str, pd.Series],
               params: dict[str, tuple[int, float, str]], method: str, carry: float) -> pd.DataFrame:
    """If ship, replace live forecast cash_requirement with RG-anchored bottom-up shape."""
    out = df.copy()
    if not report["ship"]:
        return out
    origin = y.index[-1]
    fut = out["is_forecast"]
    months = pd.DatetimeIndex(out.loc[fut, "month"])
    bal = forward_bottomup(comps, params, y, origin, months, method, carry)
    out.loc[fut, "budget_balance_bn"] = bal.reindex(months).to_numpy()
    out.loc[fut, "cash_requirement_bn"] = (-bal.reindex(months)).to_numpy()
    out.loc[fut, "buffer_change_bn"] = (
        -out.loc[fut, "cash_requirement_bn"]
        + out.loc[fut, "net_bonds_bn"]
        + out.loc[fut, "net_bills_bn"]
        + out.loc[fut, "other_fin_bn"]
    ).to_numpy()
    return out


def build_report(d: dict, y: pd.Series, chosen: str, method: str,
                 write: bool = True,
                 selected: dict[str, tuple[str, int, float, str]] | None = None) -> dict:
    comps = components_bn(d["nbr"])
    if selected is None:
        selected = select_component_params(comps)
    params = params_tuple(selected)
    rgm = d["rg"].set_index("month")["rg_forecast_bn"].dropna()
    un = score_unanchored(y, comps, params, chosen)
    an = score_anchored(y, comps, params, chosen, method, rgm)
    ship, reason = ship_decision(an.get("PRE", {}))
    auth = load_authority()
    report = {
        "ship": ship,
        "ship_reason": reason,
        "edge": EDGE,
        "horizons": list(SCORE_H),
        "production_model": chosen,
        "allocation_method": method,
        "component_params": {c: {"name": n, "k": k, "a": a, "f": f}
                             for c, (n, k, a, f) in selected.items()},
        "component_mae_pre": component_mae_table(comps, selected),
        "identity_resid_bn": round(identity_resid_bn(comps), 6),
        "pre_unanchored": {str(h): un["PRE"][h] for h in sorted(un.get("PRE", {}))},
        "oos_unanchored": {str(h): un["OOS"][h] for h in sorted(un.get("OOS", {}))},
        "pre_anchored": {str(h): an["PRE"][h] for h in sorted(an.get("PRE", {}))},
        "oos_anchored": {str(h): an["OOS"][h] for h in sorted(an.get("OOS", {}))},
        "pre_mean_mae_unanchored": {
            "bottom_up": round(mean_mae(un.get("PRE", {}), "bottom_up"), 3),
            "production": round(mean_mae(un.get("PRE", {}), "production"), 3),
        },
        "oos_mean_mae_unanchored": {
            "bottom_up": round(mean_mae(un.get("OOS", {}), "bottom_up"), 3),
            "production": round(mean_mae(un.get("OOS", {}), "production"), 3),
        },
        "pre_mean_mae_anchored": {
            "bottom_up": round(mean_mae(an.get("PRE", {}), "bottom_up"), 3),
            "production": round(mean_mae(an.get("PRE", {}), "production"), 3),
        },
        "oos_mean_mae_anchored": {
            "bottom_up": round(mean_mae(an.get("OOS", {}), "bottom_up"), 3),
            "production": round(mean_mae(an.get("OOS", {}), "production"), 3),
        },
        "rejects": [
            "same seasonal params on all three components: identical to headline seasonal (no gain)",
            "interest prior-year annual pin × seasonal: PRE unanchored 10.31 vs plain 10.10",
            "RG-anchored bottom-up shape (uniform): PRE mean loses to production anchored shape",
            "monthly MoM/YoY SARIMAX on NBR: already rejected repo-wide for SE-class seasonals",
        ],
        "authority_series": [
            {"name": "Riksgälden CH1089 Primärt saldo / Nettoutlåning / Räntor",
             "use": "bottom-up component history", "path": "data/se/nbr_monthly.csv"},
            {"name": "Riksgälden Statsupplåning annual budget balance / NBR",
             "use": "live-path annual pin (unchanged when NO-SHIP)",
             "years": auth.get("nbr_annual_balance", {}).get("years")},
            {"name": "Riksgälden monthly outcome forecast (MFN press releases)",
             "use": "RG rest-of-year total for anchored PRE gate"},
            {"name": "Riksgälden Statsupplåning borrowing plan (SGB / bills / liq)",
             "use": "financing path only (unchanged)"},
        ],
        "note": ("Unanchored bottom-up beats unanchored production on PRE (~10% relative) via "
                 "smly primary + median lending + level-shifted interest; the live path is "
                 "RG-anchored and that gate fails, so the live cash-requirement path is unchanged."),
    }
    if write:
        contract.DERIVED.mkdir(parents=True, exist_ok=True)
        OUT_SCORE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def run_report(d: dict, y: pd.Series, chosen: str, method: str, carry: float,
               df: pd.DataFrame, write: bool = True) -> tuple[pd.DataFrame, dict]:
    comps = components_bn(d["nbr"])
    selected = select_component_params(comps)
    params = params_tuple(selected)
    report = build_report(d, y, chosen, method, write=write, selected=selected)
    out = df.copy()
    out["cash_requirement_bottomup_bn"] = attach_column(out, comps, params, y)
    out = apply_ship(out, report, y, comps, params, method, carry)
    return out, report


def no_lookahead_ok(comps: dict[str, pd.Series], params: dict[str, tuple[int, float, str]],
                    asof: pd.Timestamp) -> bool:
    tgt = asof + pd.DateOffset(months=1)
    if tgt not in comps["nbr"].index:
        return True
    full = float(forecast_nbr(comps, params, asof, hmax=1).get(tgt, np.nan))
    trunc = {c: s.loc[:asof] for c, s in comps.items()}
    cut = float(forecast_nbr(trunc, params, asof, hmax=1).get(tgt, np.nan))
    return abs(full - cut) < 1e-9
