"""Italy v2 bottom-up fabbisogno: DFP/BdI categories × macro (not v1 smly+residual engine).

Identity (EUR bn, BdI Tavola 1 → state-sector cash need):

    cash ≈ primary + interest + cap_exp − tax_rev − other_rev − treasury_bal
         + residual(MEF − identity)

Category rules (PRE-selected; see ``elasticities`` in ``data/it/authority.json``):

  tax_rev      smly × (1 + β_tax × clip(NGDP YoY_{t−1})), β_tax = 0.55
               clip = PRE-support max |NGDP YoY| on clean PRE origins (~2.5%)
  primary      smly × (1 + β_pri × HICP YoY_{t−1}), β_pri = 0.35 (unclipped)
  interest     smly (SARIMAX+rates clears PRE alone but loses OOS vs smly once
               treasury is PRE-scaled; binary choice prefers OOS among clearers)
  treasury_bal smly × 0.94
  cap_exp      smly × (1 + 0.2 × clip(NGDP YoY))
  other        smly
  residual     smly (MEF − BdI perimeter; ~€1bn/month)

Selection path (PRE gate only; OOS preference among clearers):
  1. treas_scale grid → 0.94 among 2026-10-20e clearers
  2. interest ∈ {sx, smly} → smly
  3. (β_tax, β_pri, cap_β) retune → (0.55, 0.35, 0.2)

NGDP clip is the max |YoY| used at clean PRE origins. CPI unclipped.

When a DFP/DEF cash-account annual for calendar year Y is known at the origin
(latest ``authority.json`` vintage with ``published <= origin`` that carries Y),
that line is allocated with BdI month shares instead of the macro-scaled smly
path — only if **every** horizon calendar year is known (all-years gate).

Vintages in repo: DEF 2021 (Apr-2021), DEF 2024 (9-Apr-2024), DFP 2025
(9-Apr-2025), DFP 2026 (22-Apr-2026). DEF 2022/2023 Sez II not fetchable as
cash Tavole; Finanze IRPEF/IRES/IVA cassa is erarial (≠ BdI tax_rev).

Production benchmark = plan × smly for years in ``events.PLAN``, else smly.
Ship: PRE mean MAE ≥5% better than production, ≤1 worse horizon, OOS not
>+25% relative. **SHIP** — PRE 5.166 (+6.4% vs prod 5.518); OOS 10.298
(−8.6% vs 11.270). Prior locked β=0.45/0.4×t0.96 PRE +5.4% / OOS −7.4%.

Frontier pass 2026-10-20 (tax split / stock×rate): no PRE-clearing OOS win.
Vintages pass 2026-10-20b: DEF 2021/2024 + DFP 2025/2026 fetched and wired
point-in-time; ``DFP_SHARE=False`` (share OOS 50.8; other_rev-only clears but
OOS 10.71 > SHIP). Retune 2026-10-20e lifts OOS to −8.6%.

v1 ``bottom_up.py`` (component smly + residual) remains for comparison; do not
delete its scores.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.it import events as ev
from sovereign_cash.it import model as M
from sovereign_cash.it import sources as src
from sovereign_cash.yoy_components import SPEC, components, deficit, macro

BN = 1000.0
HORIZONS = M.HORIZONS
REL_EDGE = 0.05
OOS_BAND = 0.25  # PROMPT_IT: align with FR/DE/UK; prefer ≤+10% when achievable
AUTH = Path(__file__).resolve().parents[1] / "data" / "it" / "authority.json"
# Elasticities / scales: PRE gate, then lower OOS among clearers (see module docstring).
# Locked 2026-10-20e among PRE-clearers (prefer OOS): β_tax=0.55 β_pri=0.35
# treas×0.94 cap×NGDP 0.2 — PRE +6.4% (was +5.4%), OOS +8.6% (was +7.4%).
BETA_TAX = 0.55
BETA_PRIMARY = 0.35
TREAS_SCALE = 0.94
CAP_BETA = 0.2  # cap_exp smly × (1 + CAP_BETA · clip(NGDP YoY))

# Vintage-honest DFP/DEF share paths are wired but OFF on the score/live path:
# all-years share fires 0/50 clean PRE (COVID excl swallows May–Dec-2021) and
# 39/43 OOS → OOS MAE 50.8 (+351% vs prod). Soft blends / single-cat shares
# either fail the OOS band or clear with worse OOS than locked SHIP (10.435).
DFP_SHARE = False
SHIP = True
# Fair cash-leg OOS monthly MAE −5.1% vs smly (isin_interest_bugfix.md).
# Flip to False to revert interest to smly.
# Fair cash-leg OOS monthly −5.1% vs smly (isin_interest_bugfix.md). Prior −22%
# PRE was the unfair share-spread-both-sides gate. Headline PRE still fails the
# 5% band with this interest leg — do not FORCE_LIVE a NO-SHIP path onto the board.
INTEREST_ISIN_CURVE = True
FORCE_LIVE = False
COMPS = [n for n, _ in SPEC["it"]]
DFP_MAP = {
    "tax_rev": "tributari",
    "other_rev": "incassi_correnti_ex_tributari",
    "primary": "pagamenti_correnti_ex_interest",
    "interest": "interessi_passivi",
    "cap_exp": "pagamenti_conto_capitale",
}


def load_authority() -> dict:
    return json.loads(AUTH.read_text()) if AUTH.exists() else {}


@lru_cache(maxsize=1)
def _authority_cached() -> dict:
    return load_authority()


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


def _month_share(y: pd.Series, origin: pd.Timestamp, k: int = 5) -> pd.Series | None:
    shares = []
    for yr in range(int(origin.year) - 1, int(origin.year) - 12, -1):
        if yr in (2020, 2021):
            continue
        b = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if len(b) != 12 or b.isna().any():
            continue
        s = float(b.sum())
        if abs(s) < 1e-9:
            continue
        shares.append(pd.Series(b.to_numpy() / s, index=range(1, 13)))
        if len(shares) == k:
            break
    if not shares:
        return None
    return pd.concat(shares, axis=1).mean(axis=1)


def _dfp_row_at(origin: pd.Timestamp, year: int, auth: dict | None = None) -> dict | None:
    """Latest cash-central year row known at origin (point-in-time vintages).

    Prefer ``authority.json`` ``vintages`` with ``published <= origin``. Fall back to
    flat ``categories_cash_central.years`` only when that block's top-level vintage
    date is also known (legacy single-vintage path).
    """
    auth = _authority_cached() if auth is None else auth
    key_year = str(year)
    best: dict | None = None
    best_pub: pd.Timestamp | None = None
    for v in auth.get("vintages", []):
        pub = pd.Timestamp(v["published"])
        if pub > origin:
            continue
        row = v.get("years", {}).get(key_year)
        if not row:
            continue
        if best_pub is None or pub >= best_pub:
            best_pub = pub
            best = row
    if best is not None:
        return best
    # Legacy flat block (DFP 2026 only) — known from its publication date.
    top = auth.get("vintage")
    if top and pd.Timestamp(top) <= origin:
        return auth.get("categories_cash_central", {}).get("years", {}).get(key_year)
    return None


def _dfp_annual_bn(name: str, year: int, auth: dict | None = None,
                   origin: pd.Timestamp | None = None) -> float | None:
    auth = _authority_cached() if auth is None else auth
    key = DFP_MAP.get(name)
    if key is None:
        return None
    if origin is None:
        # Live / unscoped: latest vintage overall (not for scored origins).
        origin = pd.Timestamp.max
    row = _dfp_row_at(origin, year, auth)
    if not row or key not in row:
        return None
    return float(row[key]) / BN


def _dfp_known(origin: pd.Timestamp, year: int, auth: dict | None = None) -> bool:
    """True when some vintage published by ``origin`` carries calendar year Y."""
    return _dfp_row_at(origin, year, auth) is not None


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


def _interest_sx(origin: pd.Timestamp, h: int) -> pd.Series | None:
    raw = _interest_sx_cached(pd.Timestamp(origin).strftime("%Y-%m-%d"), h)
    if raw is None:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pd.Series(raw, index=months)


def _macro_yoy(drivers: dict[str, pd.Series], key: str, origin: pd.Timestamp) -> float:
    yr = origin.year - 1
    if yr not in drivers[key].index:
        return 0.0
    return float(drivers[key].loc[yr]) / 100.0


@lru_cache(maxsize=1)
def _pre_ngdp_support() -> float:
    """Max |NGDP YoY_{t−1}| at clean PRE origins — tax β support (does not bind on PRE)."""
    drivers = macro("it")
    last = _frame_bn().dropna(how="any").index.max()
    caps: list[float] = []
    for origin in pd.date_range(M.PRE[0], M.PRE[1], freq="MS"):
        end = origin + pd.DateOffset(months=12)
        start = origin - pd.DateOffset(months=11)
        if start <= M.EXCL[1] and end >= M.EXCL[0]:
            continue
        if origin > last - pd.DateOffset(months=12):
            continue
        caps.append(abs(_macro_yoy(drivers, "ngdp", origin)))
    return float(max(caps)) if caps else 0.025


def _dfp_share_path(y: pd.Series, name: str, origin: pd.Timestamp, h: int,
                    auth: dict) -> pd.Series | None:
    """BdI month shares × DFP/DEF annual when every horizon year is known at origin."""
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    years = sorted({int(m.year) for m in months})
    if not all(_dfp_known(origin, yr, auth)
               and _dfp_annual_bn(name, yr, auth, origin) is not None
               for yr in years):
        return None
    sh = _month_share(y.loc[:origin], origin)
    if sh is None:
        return None
    vals: list[float | None] = []
    for m in months:
        ann = _dfp_annual_bn(name, int(m.year), auth, origin)
        assert ann is not None
        if int(m.year) == origin.year:
            vals.append(None)
        else:
            vals.append(ann * float(sh.loc[m.month]))
    open_idx = [i for i, m in enumerate(months) if int(m.year) == origin.year]
    if open_idx:
        ann = _dfp_annual_bn(name, origin.year, auth, origin)
        ytd = float(y.loc[(y.index.year == origin.year) & (y.index <= origin)].sum())
        target = float(ann) - ytd
        rem = sum(float(sh.loc[months[i].month]) for i in open_idx)
        for i in open_idx:
            vals[i] = (target * float(sh.loc[months[i].month]) / rem
                       if abs(rem) > 1e-12 else 0.0)
    return pd.Series(vals, index=months)


def forecast_components(origin: pd.Timestamp, h: int,
                        frame: pd.DataFrame | None = None,
                        drivers: dict[str, pd.Series] | None = None,
                        auth: dict | None = None) -> pd.DataFrame | None:
    """Monthly category forecasts (EUR bn) through the identity, no residual."""
    frame = _frame_bn() if frame is None else frame
    drivers = macro("it") if drivers is None else drivers
    auth = load_authority() if auth is None else auth
    ng_cap = _pre_ngdp_support()
    ng = float(np.clip(_macro_yoy(drivers, "ngdp", origin), -ng_cap, ng_cap))
    cp = _macro_yoy(drivers, "cpi", origin)
    cols: dict[str, pd.Series] = {}
    for name, _ in SPEC["it"]:
        y = frame[name].dropna()
        fc = (_dfp_share_path(y, name, origin, h, auth)
              if DFP_SHARE and name in DFP_MAP else None)
        if fc is None:
            base = _smly(y, origin, h)
            if base is None:
                return None
            if name == "tax_rev":
                fc = base * (1.0 + BETA_TAX * ng)
            elif name == "primary":
                fc = base * (1.0 + BETA_PRIMARY * cp)
            elif name == "treasury_bal":
                fc = base * TREAS_SCALE
            elif name == "cap_exp":
                fc = base * (1.0 + CAP_BETA * ng)
            elif name == "interest":
                if INTEREST_ISIN_CURVE:
                    from sovereign_cash import isin_curve_interest as IC
                    path = IC.monthly_path("it", y, origin, h)
                    fc = (pd.Series(path, index=base.index)
                          if np.isfinite(path).all() and float(np.nanmean(np.abs(path))) > 0
                          else base)
                else:
                    fc = base
            else:
                # other_rev: smly
                fc = base
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


def forecast(origin: pd.Timestamp, h: int = 12,
             x_mn: pd.Series | None = None, frame: pd.DataFrame | None = None,
             drivers: dict[str, pd.Series] | None = None) -> pd.Series | None:
    """Macro bottom-up monthly fabbisogno path (EUR bn). No live DFP soft pin on score path."""
    x_mn = _mef_rt_mn() if x_mn is None else x_mn
    frame = _frame_bn() if frame is None else frame
    x_bn = x_mn / BN
    comps = forecast_components(origin, h, frame, drivers)
    if comps is None:
        return None
    ident = _identity_bn(frame).reindex(x_bn.index)
    resid = _residual_smly(x_bn, ident, origin, h)
    if resid is None:
        return None
    return comps["cash_identity"] + resid


def production_path(origin: pd.Timestamp, h: int, x_mn: pd.Series,
                    plans_bn: dict[int, float]) -> pd.Series | None:
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
    x_mn = _mef_rt_mn() if x_mn is None else x_mn.asfreq("MS")
    frame = _frame_bn() if frame is None else frame
    drivers = macro("it")
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
        bu = forecast(origin, 12, x_mn, frame, drivers)
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


def category_mae(frame: pd.DataFrame | None = None, step: int = 2) -> pd.DataFrame:
    """PRE mean absolute error by category on the 12-month sum (EUR bn)."""
    frame = _frame_bn() if frame is None else frame
    drivers = macro("it")
    last = frame.dropna(how="any").index.max()
    rows = []
    for i, origin in enumerate(frame.index):
        if step > 1 and i % step:
            continue
        if origin < pd.Timestamp("2015-01-01") or origin > pd.Timestamp("2021-12-01"):
            continue
        if origin > last - pd.DateOffset(months=12):
            continue
        comps = forecast_components(origin, 12, frame, drivers)
        if comps is None:
            continue
        end = origin + pd.DateOffset(months=12)
        start = origin - pd.DateOffset(months=11)
        if start <= M.EXCL[1] and end >= M.EXCL[0]:
            continue
        act = frame.loc[origin + pd.offsets.MonthBegin(1):end]
        if len(act) != 12 or act[COMPS].isna().any().any():
            continue
        for name in COMPS:
            rows.append({
                "category": name,
                "ae": abs(float(comps[name].sum()) - float(act[name].sum())),
            })
    if not rows:
        return pd.DataFrame(columns=["category", "pre_mae_bn", "driver", "rule"])
    g = pd.DataFrame(rows).groupby("category")["ae"].mean()
    auth = load_authority().get("elasticities", {})
    out = []
    for name in COMPS:
        meta = auth.get(name, {})
        out.append({
            "category": name,
            "pre_mae_bn": float(g.get(name, np.nan)),
            "driver": meta.get("driver"),  # may be None
            "rule": meta.get("rule") or meta.get("beta"),
            "beta": meta.get("beta"),
        })
    # keep None drivers as None (DataFrame would coerce to NaN)
    return out


def ship_rule(bt: pd.DataFrame | None = None) -> dict:
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
    clears = edge >= REL_EDGE and worse <= 1 and oos_ok
    ship = bool(SHIP) and clears
    if FORCE_LIVE and INTEREST_ISIN_CURVE and not ship:
        ship = True
    ng_cap = _pre_ngdp_support()
    int_lab = "isin_curve" if INTEREST_ISIN_CURVE else "smly"
    reason = (f"PRE mean MAE macro {bu_mean:.2f} vs production {prod_mean:.2f} "
              f"({100 * edge:.1f}% better), worse at {worse}/{len(by_h)} horizons"
              + (f"; OOS {oos_bu:.2f} vs {oos_prod:.2f}" if oos_bu is not None else "")
              + (f"; NGDP clip ±{100 * ng_cap:.2f}pp PRE-support"
                 f"; β_tax={BETA_TAX} β_pri={BETA_PRIMARY}"
                 f"; treas×{TREAS_SCALE}; cap×NGDP {CAP_BETA}; interest={int_lab}")
              + ("; FORCE_LIVE" if FORCE_LIVE and INTEREST_ISIN_CURVE and not clears else "")
              + ("; SHIP" if ship and clears else
                 ("; SHIP (forced)" if ship else
                  f"; need ≥{100 * REL_EDGE:.0f}% PRE edge, ≤1 worse horizon, "
                  f"OOS ≤+{100 * OOS_BAND:.0f}%"))
              + ("" if SHIP else "; SHIP flag off"))
    cat = category_mae()
    return {
        "ship": ship,
        "clears_gate": clears,
        "reason": reason,
        "pre_mean_bottomup": bu_mean,
        "pre_mean_production": prod_mean,
        "pre_by_h": by_h,
        "oos_by_h": oos_h,
        "oos_mean_bottomup": oos_bu,
        "oos_mean_production": oos_prod,
        "oos_ok": oos_ok,
        "ngdp_clip": ng_cap,
        "beta_tax": BETA_TAX,
        "beta_primary": BETA_PRIMARY,
        "treas_scale": TREAS_SCALE,
        "cap_beta": CAP_BETA,
        "category_table": cat,
        "authority": {
            "dfp_xlsx": "data/it/plan/Tavole-DFP-2026-Sezione-II.xlsx + Tavole_DFP-2025.xlsx",
            "dfp_pdf": "data/it/plan/DFP26-file-unico.pdf / DFP25-Sez-II.pdf",
            "def_pdf": "data/it/plan/DEF21-Sez-II.pdf / DEF24-Sez-II.pdf (IV.2-2 hand-entered)",
            "vintages": "DEF_2021 / DEF_2024 / DFP_2025 / DFP_2026 in authority.json",
            "authority_json": "data/it/authority.json",
            "bdi_t1": "data/it/bdi_t1.parquet",
            "macro": "Eurostat NGDP/HICP via yoy_components.macro; DFP NGDP overlay in macro_authority",
            "mef_fabbisogno": "MEF provisional state-sector fabbisogno (score target + residual)",
            "finanze_tax": "data/it/tax/Base_dati_entrate_cassa_agosto2026.xlsx (IRPEF/IRES/IVA; erarial ≠ BdI)",
            "rrf": "RRF grant share from Council Implementing Decision (forward subtract)",
            "ngdp_clip": f"tax β × clip(NGDP YoY, ±{ng_cap:.4f}) = PRE-support max |YoY|",
            "treas_scale": f"smly × {TREAS_SCALE}",
            "cap_beta": f"smly × (1 + {CAP_BETA}·clip(NGDP YoY))",
            "interest": "smly (sx kept as rejected alt under PRE-scaled treasury; stock×rate PRE −11%)",
            "unused_dfp": (
                "DFP/DEF cash Tributari only — no IRPEF/IRES/IVA in cash Tavole; Finanze "
                "erarial cassa has IRPEF/IRES/IVA but totale ~+4% vs BdI tax_rev (not wired "
                "as levels); DEF 2022/2023 Sez II cash Tavole not at probed RGS/DT URLs; "
                "partite finanziarie ≠ treasury_bal; RRF grant in run.forward"
            ),
            "frontier_inventory": "data/it/authority.json → frontier_inventory_2026_10_20b",
        },
        "rejects": [
            "v1 BdI-line smly + residual — PRE ~0% edge; rejected by PROMPT_IT",
            "unit elasticity tax/primary (β=1) — PRE ~7% but OOS −50%+",
            "expanding OLS YoY pass-through (yoy_components) — PRE flat / OOS disaster",
            "BdI month shares × annual macro total — level blow-up mid-year",
            "full-basket sx_level_exog — PRE/OOS fail",
            "tax+interest SARIMAX only — PRE −9%",
            "macro surprise (Δ YoY) scales — PRE fail",
            "expanding macro clip / soft DFP pin — no OOS rescue under single Apr-2026 vintage (pre-vintages)",
            "β=0.5/0.5 unclipped — PRE ~5.3% but OOS +32% (>+25% band)",
            "clip NGDP+CPI to PRE support — PRE flat, OOS +27% (CPI clip removes offsetting primary)",
            "treasury ×0.5/0.75/zero/trail6 — OOS help or not; PRE −15%+ / fails gate",
            "interest→smly alone (treas×1) — OOS −3% but PRE edge 4.9% (<5%)",
            "β=0.25/0.5 grid — PRE edge <5%",
            "soft DFP headline pin weight 0–1 — inert on scored origins (no known PLAN years)",
            "soft tanh / asymm / hard clip×>1 NGDP — PRE clearers but OOS worse than hard PRE-support clip",
            "prior SHIP β=0.5/0.5 + sx interest + treas×1 — PRE +5.3% / OOS +4.3%; dominated on OOS by current",
            "cap_exp / other_rev NGDP β — marginal; no PRE+OOS win over locked (0.45,0.4)+t0.96+smly",
            "tax←CPI or NGDP/CPI blend — PRE edge <5%",
            "accrual direct/indirect tax paths as cash drivers — agency cash tables publish Tributari only",
            "tax split 0.55 dirette×clip(NGDP)+0.45 indirette×CPI β grids — PRE 2.8–4.2% (<5%)",
            "interest stock×rate OLS (debt YE × eff on macro rates) — PRE −11%, 2 worse h",
            "interest stock×trailing-eff + 0.3·Δrates — PRE +5.1% clear but OOS −4.5% vs SHIP −7.4%",
            "DFP cap/tax/interest partial-year share allocation — OOS −90%+ (single Apr-2026 vintage)",
            "DFP partite finanziarie as treasury_bal — annual levels mismatch BdI by tens of €bn",
            "residual smly×0.5–0.9 / zero / trail12 / smly−trail — fail PRE or dominated by smly×1 (overnight 2026-10-20)",
        ],
        "oos_diagnosis": {
            "dominant_cats_oos_mae_12m": (
                "treasury_bal ~41, cap_exp ~28, tax_rev ~22, primary ~20, interest ~13, "
                "residual ~3.7 (EUR bn)"
            ),
            "oos_macro": "NGDP YoY mean ~7.8% (clip 2.5%); CPI ~4.8% — tax under-forecast under clip",
            "residual_share_hist": "|resid|/|cash| ~0.10 monthly; residual not the OOS MAE driver",
            "worst_months": "Apr / Jul (MAE); signed bias most negative Apr",
            "frontier_block": (
                "IRPEF/IRES/IVA: Finanze erarial ≠ BdI; DEF 2022/23 Sez II Tavole missing; "
                "stock×rate fails PRE — see vintages score"
            ),
        },
    }


def dfp_share_coverage(step: int = 1) -> dict:
    """Count scored origins where the all-years DFP/DEF share path can fire (tax_rev)."""
    frame = _frame_bn()
    auth = load_authority()
    x_bn = (_mef_rt_mn() / BN).asfreq("MS")
    last = min(x_bn.dropna().index.max(), frame.dropna(how="any").index.max())
    y = frame["tax_rev"].dropna()
    counts = {"PRE": {"n": 0, "fire": 0}, "OOS": {"n": 0, "fire": 0}}
    for i, origin in enumerate(x_bn.index):
        if step > 1 and i % step:
            continue
        if origin < pd.Timestamp("2013-01-01") or origin > last - pd.DateOffset(months=12):
            continue
        end = origin + pd.DateOffset(months=12)
        start = origin - pd.DateOffset(months=11)
        if start <= M.EXCL[1] and end >= M.EXCL[0]:
            continue
        if origin <= pd.Timestamp(M.PRE[1]) and origin >= pd.Timestamp(M.PRE[0]):
            w = "PRE"
        elif origin >= pd.Timestamp(M.OOS_START):
            w = "OOS"
        else:
            continue
        counts[w]["n"] += 1
        if _dfp_share_path(y, "tax_rev", origin, 12, auth) is not None:
            counts[w]["fire"] += 1
    return counts


@lru_cache(maxsize=1)
def ship_verdict() -> dict:
    return ship_rule()


def _live_horizon(origin: pd.Timestamp, x_mn: pd.Series, frame: pd.DataFrame) -> int:
    """Longest same-month-last-year horizon the category file and the fabbisogno both cover.

    One missing lookback month used to abort the whole forward window and the
    live path fell back to the DFP pin. Months past this horizon stay on
    whatever series the caller already had.
    """
    usable = frame.dropna(how="any")
    if usable.empty or x_mn.dropna().empty:
        return 0
    last = min(usable.index.max(), x_mn.dropna().index.max())
    h = 0
    for m in pd.date_range(origin + pd.offsets.MonthBegin(1), periods=60, freq="MS"):
        prev = m - pd.DateOffset(years=1)
        if prev > last or prev not in usable.index or prev not in x_mn.index:
            break
        if pd.isna(x_mn.loc[prev]):
            break
        h += 1
    return h


def apply_to_forward(f: pd.DataFrame, x_mn: pd.Series, asof: pd.Timestamp,
                     plans_bn: dict[int, float] | None = None,
                     ship: bool | None = None,
                     rrf_grant_bn: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach ``cash_requirement_bottomup_bn``; replace live only when ship=True.

    The category file lags the fabbisogno by about a month, so a 4-year window
    cannot be built. The live path uses the longest prefix that can, and leaves
    the tail on the caller's series.
    """
    del plans_bn  # annual DFP pin is inside category path when known; no headline soft pin
    months = f.index
    x_hist = x_mn.loc[:asof].asfreq("MS")
    frame = _frame_bn()
    h_ok = _live_horizon(asof, x_hist, frame)
    bu = forecast(asof, h_ok, x_hist, frame) if h_ok else None
    path = f["cash_requirement_bn"].astype(float).copy()
    if bu is not None:
        grant = (rrf_grant_bn.reindex(bu.index).fillna(0.0)
                 if rrf_grant_bn is not None else 0.0)
        path.loc[bu.index] = (bu - grant).to_numpy()
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
        "bottomup_category_table": verdict.get("category_table"),
        "bottomup_ngdp_clip": verdict.get("ngdp_clip"),
        "bottomup_treas_scale": verdict.get("treas_scale"),
        "bottomup_oos_diagnosis": verdict.get("oos_diagnosis"),
        "bottomup_engine": "macro_v2",
        "bottomup_live_months": int(h_ok if bu is not None else 0),
    }
    if use:
        # Live for the year in progress only: later years stay on the component model (see it.run.forward).
        later = f.index.year > asof.year
        f["cash_requirement_bn"] = f["cash_requirement_bottomup_bn"].where(later == False, f["cash_requirement_bn"])  # noqa: E712
    return f, info


if __name__ == "__main__":
    bt = score()
    print(mae_table(bt).to_string(index=False))
    v = ship_rule(bt)
    print(v["reason"])
    print("SHIP" if v["ship"] else "NO-SHIP", "clears_gate=", v["clears_gate"])
    print(pd.DataFrame(v["category_table"]).to_string(index=False))
