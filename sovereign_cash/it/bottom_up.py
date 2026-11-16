"""Italy bottom-up fabbisogno baseline (BdI T1 components + MEF residual).

Identity (EUR bn, BdI Tavola 1 formation → state-sector cash need):

    cash ≈ primary + interest + cap_exp − tax_rev − other_rev − treasury_bal

MEF's printed fabbisogno and the BdI identity differ by ~€1.2bn/month MAE, so the
path also forecasts the residual (MEF − identity) with same-month-last-year and adds
it back. That keeps the series on the MEF print the live board scores.

Component rules (chosen on PRE component MAE, then re-checked on headline PRE):

  interest     seasonal SARIMAX (1,0,0)×(0,1,1,12) + lag-2 effective rate
  tax / other / primary / cap / treasury_bal     same-month-last-year
  residual     same-month-last-year

Open DFP years take a **soft** plan pin (50% blend of the component path toward the
remaining DFP annual total). Hard close loses PRE; soft pin is for the alternate
column only.

Production benchmark = plan × smly shape for years in ``events.PLAN``, else smly.
PRE (origins 2015–2021, COVID screened) does not clear a ≥5% relative edge; OOS is
reported only. Live ``cash_requirement_bn`` stays plan×smly.

Tried and rejected (one line each):
- Aggregate MoM/YoY SARIMAX — loses PRE (existing).
- Full-basket ``sx_level_exog`` — PRE win historically voided; OOS collapses.
- Annual component GDP/CPI pass-through — loses PRE to smly.
- All-component smly identity without residual — loses to aggregate smly (perimeter gap).
- Tax SARIMAX + NGDP alone — PRE worse; OOS better (not selectable).
- YTD tax/spend pace ratios — PRE blow-up.
- Calendar median / trim / debias overlays — lose PRE.
- Hard DFP close on component shape — loses PRE to smly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from sovereign_cash.it import events as ev
from sovereign_cash.it import model as M
from sovereign_cash.it import sources as src
from sovereign_cash.yoy_components import SPEC, components, deficit

BN = 1000.0
HORIZONS = M.HORIZONS
REL_EDGE = 0.05
OOS_BAND = 0.10  # ship also needs OOS mean ≤ (1 + OOS_BAND) × production
SOFT_PLAN_W = 0.5  # weight on plan scale vs component path inside DFP years
COMPS = [n for n, _ in SPEC["it"]]


def _mef_rt_mn() -> pd.Series:
    rel = src.fetch_mef_fabbisogno(refresh=False)
    return src.mef_monthly(rel)["fabb_rt_mn"].loc["2010-01":].asfreq("MS")


def _frame_bn() -> pd.DataFrame:
    return components("it")


def _identity_bn(frame: pd.DataFrame) -> pd.Series:
    return deficit("it", {n: frame[n] for n, _ in SPEC["it"]})


def _smly(y: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    vals = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        if prev not in y.index or pd.isna(y.loc[prev]):
            return None
        vals.append(float(y.loc[prev]))
    return pd.Series(vals, index=months)


@lru_cache(maxsize=256)
def _interest_sx_cached(origin_key: str, h: int) -> tuple[float, ...] | None:
    from sovereign_cash.monthly_components import _exog_col, fit_forecast, monthly_drivers

    frame = _frame_bn()
    y_mn = (frame["interest"] * BN).dropna().asfreq("MS")
    origin = pd.Timestamp(origin_key)
    drivers = monthly_drivers("it")
    ex = _exog_col("rates", drivers)
    fc = fit_forecast(y_mn, origin, h, ex, kind="level")
    if fc is None:
        sm = _smly(frame["interest"].dropna(), origin, h)
        return None if sm is None else tuple(float(v) for v in sm.to_numpy())
    return tuple(float(v) / BN for v in fc.to_numpy())


def _interest_sx(y_bn: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    # Cache key ignores y_bn identity (always the carved IT interest line).
    del y_bn
    raw = _interest_sx_cached(pd.Timestamp(origin).strftime("%Y-%m-%d"), h)
    if raw is None:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pd.Series(raw, index=months)


def forecast_components(origin: pd.Timestamp, h: int,
                        frame: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Monthly component forecasts (EUR bn) through the identity, no residual yet."""
    frame = _frame_bn() if frame is None else frame
    cols: dict[str, pd.Series] = {}
    for name, _ in SPEC["it"]:
        y = frame[name].dropna()
        if name == "interest":
            fc = _interest_sx(y, origin, h)
        else:
            fc = _smly(y, origin, h)
        if fc is None:
            return None
        cols[name] = fc
    out = pd.DataFrame(cols)
    out["cash_identity"] = deficit("it", {k: out[k] for k, _ in SPEC["it"]})
    return out


def _residual_smly(x_bn: pd.Series, ident_bn: pd.Series, origin: pd.Timestamp,
                   h: int) -> pd.Series | None:
    resid = (x_bn - ident_bn).reindex(x_bn.index).fillna(0.0).asfreq("MS")
    return _smly(resid, origin, h)


def soft_plan_pin(fc: pd.Series, x_bn: pd.Series, origin: pd.Timestamp,
                  plans_bn: dict[int, float], weight: float = SOFT_PLAN_W) -> pd.Series:
    """Blend component path toward remaining DFP total inside covered calendar years."""
    out = fc.astype(float).copy()
    by_year: dict[int, list[pd.Timestamp]] = {}
    for m in out.index:
        by_year.setdefault(int(m.year), []).append(m)
    for yr, months in by_year.items():
        if yr not in plans_bn:
            continue
        ytd = float(x_bn[(x_bn.index.year == yr) & (x_bn.index <= origin)].sum()) if origin.year == yr else 0.0
        target_rem = float(plans_bn[yr]) - ytd
        cur = float(out.loc[months].sum())
        if abs(cur) < 1e-9 or abs(target_rem) < 1e-9:
            continue
        scale = weight * (target_rem / cur) + (1.0 - weight)
        out.loc[months] = out.loc[months] * scale
    return out


def forecast(origin: pd.Timestamp, h: int = 12,
             x_mn: pd.Series | None = None, frame: pd.DataFrame | None = None,
             plans_bn: dict[int, float] | None = None,
             soft_pin: bool = True) -> pd.Series | None:
    """Bottom-up monthly fabbisogno path (EUR bn), optional soft DFP pin."""
    x_mn = _mef_rt_mn() if x_mn is None else x_mn
    frame = _frame_bn() if frame is None else frame
    x_bn = x_mn / BN
    comps = forecast_components(origin, h, frame)
    if comps is None:
        return None
    ident = _identity_bn(frame).reindex(x_bn.index)
    resid = _residual_smly(x_bn, ident, origin, h)
    if resid is None:
        return None
    path = comps["cash_identity"] + resid
    if soft_pin:
        plans = plans_bn if plans_bn is not None else {y: v / BN for y, v in ev.plan_mn().items()}
        path = soft_plan_pin(path, x_bn.loc[:origin], origin, plans)
    return path


def production_path(origin: pd.Timestamp, h: int, x_mn: pd.Series,
                    plans_bn: dict[int, float]) -> pd.Series | None:
    """Live-like path: smly, hard-anchored to DFP for covered years (EUR bn)."""
    x_bn = (x_mn / BN).asfreq("MS")
    if origin not in x_bn.index:
        return None
    o = x_bn.index.get_loc(origin)
    raw = M.forecast(x_mn.asfreq("MS"), o, "smly", horizon=h)
    if np.isnan(raw).any():
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fc = pd.Series(raw / BN, index=months)
    if not plans_bn:
        return fc
    return M.anchor_to_plan(fc, x_bn.loc[:origin], origin, plans_bn)


def score(x_mn: pd.Series | None = None, frame: pd.DataFrame | None = None,
          step: int = 1) -> pd.DataFrame:
    """Walk-forward cumulative absolute errors (EUR bn) for bottom-up vs production."""
    x_mn = _mef_rt_mn() if x_mn is None else x_mn.asfreq("MS")
    frame = _frame_bn() if frame is None else frame
    plans_bn = {y: v / BN for y, v in ev.plan_mn().items()}
    x_bn = x_mn / BN
    last = min(x_bn.dropna().index.max(), frame.dropna(how="any").index.max())
    rows = []
    for i, origin in enumerate(x_mn.index):
        if step > 1 and i % step:
            continue
        if origin < pd.Timestamp("2013-01-01") or origin > last - pd.DateOffset(months=12):
            continue
        if origin not in frame.index:
            continue
        # Score the component path without the live soft DFP pin (PRE has no DFP years;
        # pinning open OOS years into 2026–27 is a live-path choice, not a selection knob).
        bu = forecast(origin, 12, x_mn, frame, plans_bn, soft_pin=False)
        prod = production_path(origin, 12, x_mn, plans_bn)
        sm = production_path(origin, 12, x_mn, {})
        if bu is None or prod is None or sm is None:
            continue
        for h in HORIZONS:
            end = origin + pd.DateOffset(months=h)
            if end > last:
                continue
            act = x_bn.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h or act.isna().any():
                continue
            start = origin - pd.DateOffset(months=11)
            touches = bool(start <= M.EXCL[1] and end >= M.EXCL[0])
            rows.append({
                "origin": origin, "h": int(h),
                "bottomup": abs(float(bu.iloc[:h].sum()) - float(act.sum())),
                "production": abs(float(prod.iloc[:h].sum()) - float(act.sum())),
                "smly": abs(float(sm.iloc[:h].sum()) - float(act.sum())),
                "touches_excl": touches,
            })
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    bt["window"] = M.window_of(bt)
    return bt


def mae_table(bt: pd.DataFrame | None = None) -> pd.DataFrame:
    bt = score() if bt is None else bt
    rows = []
    for (w, h), g in bt.groupby(["window", "h"]):
        if w == "":
            continue
        g = g.dropna(subset=["bottomup", "production"])
        if g.empty:
            continue
        rows.append({
            "window": w, "h": int(h), "n": int(len(g)),
            "bottomup": float(g["bottomup"].mean()),
            "production": float(g["production"].mean()),
            "smly": float(g["smly"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["window", "h"]).reset_index(drop=True)


def ship_rule(bt: pd.DataFrame | None = None) -> dict:
    """PRE ≥5% better than production, ≤1 worse horizon; OOS not >+10% relative."""
    bt = score() if bt is None else bt
    pre = bt[bt["window"] == "PRE"]
    by_h: dict[int, dict] = {}
    for h in HORIZONS:
        g = pre[pre["h"] == h].dropna(subset=["bottomup", "production"])
        if g.empty:
            continue
        by_h[int(h)] = {
            "n": int(len(g)),
            "bottomup": float(g["bottomup"].mean()),
            "production": float(g["production"].mean()),
            "smly": float(g["smly"].mean()),
        }
    if len(by_h) < len(HORIZONS):
        return {"ship": False, "reason": "incomplete PRE horizons", "pre_by_h": by_h}
    bu_mean = float(np.mean([r["bottomup"] for r in by_h.values()]))
    prod_mean = float(np.mean([r["production"] for r in by_h.values()]))
    worse = sum(1 for r in by_h.values() if r["bottomup"] > r["production"] + 1e-12)
    edge = (prod_mean - bu_mean) / prod_mean if prod_mean > 0 else 0.0
    oos = bt[bt["window"] == "OOS"]
    oos_h: dict[int, dict] = {}
    for h in HORIZONS:
        g = oos[oos["h"] == h].dropna(subset=["bottomup", "production"])
        if len(g):
            oos_h[int(h)] = {
                "n": int(len(g)),
                "bottomup": float(g["bottomup"].mean()),
                "production": float(g["production"].mean()),
                "smly": float(g["smly"].mean()),
            }
    oos_bu = float(np.mean([r["bottomup"] for r in oos_h.values()])) if oos_h else None
    oos_prod = float(np.mean([r["production"] for r in oos_h.values()])) if oos_h else None
    oos_ok = (oos_bu is None or oos_prod is None
              or oos_bu <= (1.0 + OOS_BAND) * oos_prod + 1e-12)
    ship = edge >= REL_EDGE and worse <= 1 and oos_ok
    reason = (f"PRE mean MAE bottom-up {bu_mean:.2f} vs production {prod_mean:.2f} "
              f"({100 * edge:.1f}% better), worse at {worse}/{len(by_h)} horizons"
              + (f"; OOS {oos_bu:.2f} vs {oos_prod:.2f}" if oos_bu is not None else "")
              + ("; SHIP" if ship else
                 f"; need ≥{100 * REL_EDGE:.0f}% PRE edge, ≤1 worse horizon, "
                 f"OOS ≤+{100 * OOS_BAND:.0f}%"))
    return {
        "ship": ship,
        "reason": reason,
        "pre_mean_bottomup": bu_mean,
        "pre_mean_production": prod_mean,
        "pre_by_h": by_h,
        "oos_by_h": oos_h,
        "oos_mean_bottomup": oos_bu,
        "oos_mean_production": oos_prod,
        "oos_ok": oos_ok,
        "authority": {
            "dfp_fabbisogno": f"DFP {ev.PLAN_VINTAGE} Tav. II.1.2.5 %GDP × Tav. II.1.2.2 NGDP (soft pin)",
            "bdi_t1": "Banca d'Italia Finanza pubblica Tavola 1 (tax/other/cur/cap/treasury)",
            "interest_carve": "debt-book annual interest × BdI current-spend month shares",
            "rates": "effective interest / debt (authority on forecast years)",
            "mef_fabbisogno": "MEF provisional state-sector fabbisogno (score target + residual)",
            "rrf": "RRF grant share from Council Implementing Decision (forward subtract)",
        },
        "rejects": [
            "MoM/YoY SARIMAX — PRE fail",
            "sx_level_exog full basket — OOS disaster",
            "annual GDP/CPI component pass-through — loses PRE to smly",
            "component smly identity without residual — perimeter gap vs MEF",
            "tax SARIMAX+NGDP — PRE worse (OOS better; not selectable)",
            "YTD pace ratios — PRE blow-up",
            "hard DFP close on component shape — loses PRE",
        ],
    }


@lru_cache(maxsize=1)
def ship_verdict() -> dict:
    return ship_rule()


def apply_to_forward(f: pd.DataFrame, x_mn: pd.Series, asof: pd.Timestamp,
                     plans_bn: dict[int, float] | None = None,
                     ship: bool | None = None,
                     rrf_grant_bn: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach ``cash_requirement_bottomup_bn``; replace live path only when ship=True."""
    plans = plans_bn if plans_bn is not None else {y: v / BN for y, v in ev.plan_mn().items()}
    months = f.index
    h = len(months)
    bu = forecast(asof, h, x_mn.loc[:asof].asfreq("MS"), plans_bn=plans, soft_pin=True)
    if bu is None:
        path = f["cash_requirement_bn"].astype(float).copy()
    else:
        path = bu.reindex(months).astype(float)
        if rrf_grant_bn is not None:
            # Grants lower the printed fabbisogno; strip known forward instalments.
            path = path - rrf_grant_bn.reindex(months).fillna(0.0)
        path = path.fillna(f["cash_requirement_bn"])
    f = f.copy()
    f["cash_requirement_bottomup_bn"] = path.to_numpy()
    verdict = ship_verdict() if ship is None else {"ship": bool(ship), "reason": "override"}
    use = bool(verdict["ship"]) if ship is None else bool(ship)
    info = {
        "bottomup_ship": use,
        "bottomup_reason": verdict.get("reason", ""),
        "bottomup_pre_mean": verdict.get("pre_mean_bottomup"),
        "bottomup_pre_production_mean": verdict.get("pre_mean_production"),
        "bottomup_oos_mean": verdict.get("oos_mean_bottomup"),
        "bottomup_oos_production_mean": verdict.get("oos_mean_production"),
        "bottomup_authority": verdict.get("authority"),
        "bottomup_pre_by_h": verdict.get("pre_by_h"),
        "bottomup_oos_by_h": verdict.get("oos_by_h"),
        "bottomup_rejects": verdict.get("rejects"),
    }
    if use:
        f["cash_requirement_bn"] = f["cash_requirement_bottomup_bn"]
    return f, info
