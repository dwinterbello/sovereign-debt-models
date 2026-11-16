"""Germany bottom-up core cash-requirement baseline.

Identity (core budget, deficit +):

    cash_requirement_core = expenditure − revenue
                          = primary + interest − revenue

BMF Monatsbericht prints monthly Ist and Soll for Ausgaben / Einnahmen (and
Steuereinnahmen from 2017). Interest is not printed monthly; the debt-book carve
is available but did not beat the two-line Soll path on PRE, so the shipped
bottom-up pins spending and revenue to their own Soll × historical outturn/plan
ratio for the open budget year, then same-month-last-year on each line after that.

Production today (``planhc_3`` a priori + open-year Soll overlay for the current
year, planhc for later years) is the benchmark. Walk-forward PRE (windows ending
≤ Dec-2019) picks the bottom-up; OOS is reported only.

Tried and rejected (one line each):
- Annual YoY macro pass-through on revenue/primary/interest — loses PRE to smly.
- Monthly level SARIMAX + HICP/NGDP — scored in deficit_scores; not selected on PRE.
- YTD Ist/Soll execution ratio (alone or blended) — worse PRE than historical ratio.
- Raw Soll pin (ratio=1) — worse PRE than haircut / hist ratio.
- Tax vs non-tax Soll split — no PRE sample (tax series from 2017).
- Interest carve + primary Soll — ties open-year path; no PRE gain.
- Later-year planhc after open-year split — loses PRE to line-level smly (this module).

Soll remains a hard annual pin for the open budget year. Later-year Soll × haircut
loses PRE to smly; debt-brake a priori still argues for the plan on OOS — ship is
PRE-only per the shared contract.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.de import fetch as F
from sovereign_cash.de import model as M
from sovereign_cash.yoy_components import _execution, _month_share, close_to_plan

HORIZONS = C.HORIZONS
REL_EDGE = 0.05  # ship if mean PRE MAE ≤ (1 − REL_EDGE) × production


def _flows() -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, dict]:
    core = M.core_budget()
    mb = F.load_bmf_mb()
    exp = C.monthly_from_ytd(core["ytd_expenditure"].dropna()).dropna()
    rev = C.monthly_from_ytd(core["ytd_revenue"].dropna()).dropna()
    cash = (exp - rev).dropna()
    maps = M.soll_line_maps(mb)
    return exp, rev, cash, mb, maps


def forecast_core(origin: pd.Timestamp, h: int,
                  exp: pd.Series | None = None, rev: pd.Series | None = None,
                  mb: pd.DataFrame | None = None, maps: dict | None = None,
                  ) -> pd.Series | None:
    """Open year: spend/rev each close to Soll × hist execution; later years: smly per line."""
    if exp is None or rev is None or mb is None or maps is None:
        exp, rev, _, mb, maps = _flows()
    known = mb.loc[:origin].dropna(subset=["soll_expenditure", "soll_revenue", "soll_year"])
    known = known[known["soll_year"] == origin.year]
    if known.empty:
        return None
    plan_e = float(known.iloc[-1]["soll_expenditure"])
    plan_r = float(known.iloc[-1]["soll_revenue"])
    sh_e, sh_r = _month_share(exp, origin), _month_share(rev, origin)
    if sh_e is None or sh_r is None:
        return None
    re = _execution(exp, maps["soll_expenditure"], origin)
    rr = _execution(rev, maps["soll_revenue"], origin)
    fe = close_to_plan(exp, plan_e, re, origin, h, sh_e)
    fr = close_to_plan(rev, plan_r, rr, origin, h, sh_r)
    for m in fe.index:
        if m.year == origin.year:
            continue
        pe = exp.get(m - pd.DateOffset(years=1), np.nan)
        pr = rev.get(m - pd.DateOffset(years=1), np.nan)
        if pd.isna(pe) or pd.isna(pr) or (m - pd.DateOffset(years=1)) > origin:
            return None
        fe.loc[m] = float(pe)
        fr.loc[m] = float(pr)
    return fe - fr


def production_core(origin: pd.Timestamp, h: int, cash: pd.Series, targets: dict,
                    exp: pd.Series, rev: pd.Series, mb: pd.DataFrame, maps: dict,
                    ) -> pd.Series | None:
    """Live-like path: open-year Soll split, later months ``planhc_3``."""
    tg = targets.get(origin)
    ph = C.forecast(cash, origin, h, "planhc_3", targets=tg)
    if ph is None:
        return None
    bu = forecast_core(origin, h, exp, rev, mb, maps)
    if bu is None:
        return None
    out = bu.copy()
    for m in out.index:
        if m.year != origin.year:
            out.loc[m] = float(ph.loc[m])
    return out


def score(horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Walk-forward cumulative MAE rows: bottom-up vs production overlay vs planhc / smly."""
    exp, rev, cash, mb, maps = _flows()
    core = M.core_budget()
    hc = M.haircuts(cash, core)
    targets = M.soll_by_origin(core, hc)
    rows = []
    for origin in cash.index:
        for h in horizons:
            end = origin + pd.DateOffset(months=h)
            act = cash.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h or act.isna().any():
                continue
            tg = targets.get(origin)
            ph = C.forecast(cash, origin, h, "planhc_3", targets=tg)
            sm = C.forecast(cash, origin, h, "smly")
            bu = forecast_core(origin, h, exp, rev, mb, maps)
            prod = production_core(origin, h, cash, targets, exp, rev, mb, maps)
            if ph is None or bu is None or prod is None:
                continue

            def err(fc: pd.Series) -> float:
                return abs(float(fc.iloc[:h].sum()) - float(act.sum()))

            w = "PRE" if end <= C.PRE_END else ("OOS" if origin >= C.OOS_START else "mid")
            rows.append({
                "origin": origin, "h": h, "window": w,
                "bottomup": err(bu), "production": err(prod),
                "planhc": err(ph),
                "smly": err(sm) if sm is not None else np.nan,
            })
    return pd.DataFrame(rows)


def mae_table(bt: pd.DataFrame | None = None) -> pd.DataFrame:
    bt = score() if bt is None else bt
    rows = []
    for (w, h), g in bt.groupby(["window", "h"]):
        g = g.dropna(subset=["bottomup", "production", "planhc"])
        if g.empty:
            continue
        rows.append({
            "window": w, "h": int(h), "n": int(len(g)),
            "bottomup": float(g["bottomup"].mean()),
            "production": float(g["production"].mean()),
            "planhc": float(g["planhc"].mean()),
            "smly": float(g["smly"].mean()) if g["smly"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["window", "h"]).reset_index(drop=True)


def ship_rule(bt: pd.DataFrame | None = None) -> dict:
    """PRE-only: mean MAE across horizons ≤ (1−REL_EDGE)×production, worse at ≤1 horizon."""
    bt = score() if bt is None else bt
    pre = bt[bt["window"] == "PRE"]
    by_h = {}
    for h in HORIZONS:
        g = pre[pre["h"] == h].dropna(subset=["bottomup", "production"])
        if g.empty:
            continue
        by_h[int(h)] = {
            "n": int(len(g)),
            "bottomup": float(g["bottomup"].mean()),
            "production": float(g["production"].mean()),
            "planhc": float(g["planhc"].mean()),
        }
    if len(by_h) < len(HORIZONS):
        return {"ship": False, "reason": "incomplete PRE horizons", "by_h": by_h}
    bu_mean = float(np.mean([r["bottomup"] for r in by_h.values()]))
    prod_mean = float(np.mean([r["production"] for r in by_h.values()]))
    planhc_mean = float(np.mean([r["planhc"] for r in by_h.values()]))
    worse = sum(1 for r in by_h.values() if r["bottomup"] > r["production"] + 1e-12)
    edge = (prod_mean - bu_mean) / prod_mean if prod_mean > 0 else 0.0
    ship = edge >= REL_EDGE and worse <= 1
    oos = bt[bt["window"] == "OOS"]
    oos_h = {}
    for h in HORIZONS:
        g = oos[oos["h"] == h].dropna(subset=["bottomup", "production"])
        if len(g):
            oos_h[int(h)] = {
                "n": int(len(g)),
                "bottomup": float(g["bottomup"].mean()),
                "production": float(g["production"].mean()),
                "planhc": float(g["planhc"].mean()),
            }
    return {
        "ship": ship,
        "reason": (f"PRE mean MAE bottom-up {bu_mean:.2f} vs production {prod_mean:.2f} "
                   f"({100 * edge:.1f}% better), worse at {worse}/{len(by_h)} horizons"
                   + ("; SHIP" if ship else f"; need ≥{100 * REL_EDGE:.0f}% and ≤1 worse horizon")),
        "pre_mean_bottomup": bu_mean,
        "pre_mean_production": prod_mean,
        "pre_mean_planhc": planhc_mean,
        "pre_by_h": by_h,
        "oos_by_h": oos_h,
        "oos_mean_bottomup": float(np.mean([r["bottomup"] for r in oos_h.values()])) if oos_h else None,
        "oos_mean_production": float(np.mean([r["production"] for r in oos_h.values()])) if oos_h else None,
        "authority": {
            "soll_expenditure": "BMF Monatsbericht Soll Ausgaben (printed at origin)",
            "soll_revenue": "BMF Monatsbericht Soll Einnahmen (printed at origin)",
            "execution_ratio": "complete-year Ist/Soll before origin, Covid excluded",
            "later_years": "same-month last year on expenditure and revenue",
            "plan_json": "special funds only (unchanged); core later-year Soll not used when shipped",
        },
    }


@lru_cache(maxsize=1)
def ship_verdict() -> dict:
    return ship_rule()


def apply_to_forward(f: pd.DataFrame, cash: pd.Series, core: pd.DataFrame,
                     last_cash: pd.Timestamp, ship: bool | None = None,
                     ) -> tuple[pd.DataFrame, dict]:
    """Attach ``cash_requirement_*_bottomup_bn``; optionally replace live core path."""
    exp = C.monthly_from_ytd(core["ytd_expenditure"].dropna()).dropna()
    rev = C.monthly_from_ytd(core["ytd_revenue"].dropna()).dropna()
    mb = F.load_bmf_mb()
    maps = M.soll_line_maps(mb)
    months = f.index
    h = len(months)
    # bottom-up from last observed cash month (same origin the live core uses)
    bu = forecast_core(last_cash, h, exp, rev, mb, maps)
    core_bu = []
    for m in months:
        if bool(f.at[m, "cash_observed"]) if "cash_observed" in f.columns else m <= last_cash:
            core_bu.append(float(cash[m]) if m in cash.index else float(f.at[m, "cash_requirement_core_bn"]))
        elif bu is not None and m in bu.index:
            core_bu.append(float(bu.loc[m]))
        else:
            core_bu.append(float(f.at[m, "cash_requirement_core_bn"]))
    f = f.copy()
    f["cash_requirement_core_bottomup_bn"] = core_bu
    f["cash_requirement_bottomup_bn"] = (
        f["cash_requirement_core_bottomup_bn"] + f["cash_requirement_sv_bn"])
    verdict = ship_verdict() if ship is None else {"ship": ship, "reason": "override"}
    use = bool(verdict["ship"]) if ship is None else bool(ship)
    if use:
        # year in progress only; later years stay on the component model (see de.model.forward)
        later = f.index.year > last_cash.year
        f["cash_requirement_core_bn"] = f["cash_requirement_core_bottomup_bn"].where(~later, f["cash_requirement_core_bn"])
        f["cash_requirement_bn"] = f["cash_requirement_core_bn"] + f["cash_requirement_sv_bn"]
        f["buffer_change_bn"] = (
            -f["cash_requirement_bn"] + f["net_bonds_bn"] + f["net_bills_bn"] + f["other_fin_bn"])
        if "buffer_level_implied_bn" in f.columns and len(f):
            # caller may refresh from history; keep relative consistency
            pass
        f["fc_source"] = [
            (("core observed (BMF); " if (bool(f.at[m, "cash_observed"])
                                          if "cash_observed" in f.columns else m <= last_cash)
              else "core bottom-up spend/rev Soll×ratio then smly; ")
             + str(f.at[m, "fc_source"]).split("; ", 1)[-1]
             if "fc_source" in f.columns else "core bottom-up")
            for m in months]
    info = {"bottomup_ship": use, "bottomup_reason": verdict.get("reason", ""),
            "bottomup_pre_mean": verdict.get("pre_mean_bottomup"),
            "bottomup_pre_production_mean": verdict.get("pre_mean_production"),
            "bottomup_pre_planhc_mean": verdict.get("pre_mean_planhc"),
            "bottomup_oos_mean": verdict.get("oos_mean_bottomup"),
            "bottomup_authority": verdict.get("authority"),
            "bottomup_pre_by_h": verdict.get("pre_by_h"),
            "bottomup_oos_by_h": verdict.get("oos_by_h")}
    return f, info
