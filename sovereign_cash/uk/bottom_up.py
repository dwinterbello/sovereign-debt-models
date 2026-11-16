"""Bottom-up CGNCR(ex) baseline from ONS published components.

Identity (monthly, £bn; ONS PUSF)

    CGNCR(ex) = current_exp + capital_exp − current_receipts + residual

where residual absorbs financial transactions, timing, and the M98R perimeter
(ex NRAM / B&B / Network Rail) versus the CG current/capital accounts.

Debt interest is carved from current expenditure (JW2P as the PS interest paid
proxy) and forecast separately. Component rules are locked on PRE origins only
against the live hybrid (remit vintage × profile vs seasonal model at EDGE_BAND).

Live ``cash_requirement_bn`` switches to this path only when mean PRE MAE beats
production by more than ``EDGE_BAND`` and is worse at ≤1 scored horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

# Locked 2026-10-20 on PRE mean cumulative MAE vs production hybrid.
# REJECTED as v1 seasonal main engine — see bottom_up_macro.py. Do not promote.
SHIP = False
SPEC = {
    "rec": "seasonal_add3",
    "primary": "seasonal_add3",
    "interest": "smly",
    "cap": "seasonal_add3",
    "resid": "smly",
}

AUTHORITY = (
    ("ONS PUSF ANBV", "cg_current_receipts", "CG total current receipts"),
    ("ONS PUSF MF6S", "cg_current_exp", "CG total current expenditure"),
    ("ONS PUSF MF6T", "cg_capital_exp", "CG total capital expenditure"),
    ("ONS PUSF JW2P", "ps_interest_paid", "PS interest & dividends paid (debt-interest proxy)"),
    ("ONS PUSF M98R", "cgncr_ex", "CGNCR ex NRAM, B&B, Network Rail (target / residual)"),
    ("DMO remit vintages", "remit_vintages", "production benchmark / ship gate only"),
)

BOTTOMUP = "bottom_up"
PRODUCTION = "production"


def components() -> pd.DataFrame:
    """Monthly component frame (£bn). Residual is exact on overlapping history."""
    ons = src.ons_monthly()
    need = ["cg_current_receipts", "cg_current_exp", "cg_capital_exp", "ps_interest_paid", "cgncr_ex"]
    missing = [k for k in need if k not in ons.columns]
    if missing:
        raise FileNotFoundError(
            f"ONS component series missing from cache: {missing}. "
            "Run: PYTHONPATH=. .venv/bin/python -m sovereign_cash.uk.run --refresh")
    df = pd.DataFrame({
        "rec": ons["cg_current_receipts"],
        "curr": ons["cg_current_exp"],
        "cap": ons["cg_capital_exp"],
        "interest": ons["ps_interest_paid"],
        "cgncr": ons["cgncr_ex"],
    }).dropna()
    df["primary"] = df["curr"] - df["interest"]
    df["bridge"] = df["curr"] + df["cap"] - df["rec"]
    df["resid"] = df["cgncr"] - df["bridge"]
    df.index.name = "month"
    return df


def forecast(comp: pd.DataFrame, origin: pd.Timestamp, h: int = M.FC_MONTHS,
             spec: dict[str, str] | None = None) -> np.ndarray:
    """Monthly CGNCR(ex) for origin+1..origin+h from components through ``origin`` only."""
    spec = spec or SPEC
    n = h
    rec = M.cgncr_forecast(comp["rec"], origin, spec["rec"], h=n)
    prim = M.cgncr_forecast(comp["primary"], origin, spec["primary"], h=n)
    inte = M.cgncr_forecast(comp["interest"], origin, spec["interest"], h=n)
    cap = M.cgncr_forecast(comp["cap"], origin, spec["cap"], h=n)
    resid = M.cgncr_forecast(comp["resid"], origin, spec["resid"], h=n)
    return prim + inte + cap - rec + resid


def production_forecast(x: pd.Series, origin: pd.Timestamp, vint: pd.DataFrame,
                        cg_method: str, pm: dict[int, float], pp: dict[int, float],
                        h: int = M.FC_MONTHS) -> np.ndarray:
    """Live hybrid: remit vintage × profile unless the seasonal model clears EDGE_BAND."""
    cg_m = M.cgncr_forecast(x, origin, cg_method, h=h)
    cg_p = M.official_vintage_forecast(x, origin, vint, h=h)
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fy = M.fy_of(origin) if origin.month != 3 else M.fy_of(origin) + 1
    in_fy = np.array([M.fy_of(m) == fy for m in months])
    months_left = int(in_fy.sum())
    src_in = "model" if (not pp or M.model_beats(pm, pp, months_left)) else "plan"
    src_after = "model" if (not pp or M.model_beats(pm, pp, M.FC_MONTHS)) else "plan"
    use_plan = np.where(in_fy, src_in == "plan", src_after == "plan") & np.isfinite(cg_p)
    return np.where(use_plan, cg_p, cg_m)


def backtest(comp: pd.DataFrame | None = None, vint: pd.DataFrame | None = None,
             spec: dict[str, str] | None = None) -> pd.DataFrame:
    """Walk-forward cumulative h-month sums for bottom-up, production hybrid, official, chosen."""
    comp = components() if comp is None else comp
    vint = src.remit_vintages() if vint is None else vint
    spec = spec or SPEC
    x = comp["cgncr"]
    bt_base = M.backtest_cgncr(x, vint)
    t = M.score_table(bt_base)
    cg_method = M.choose(t, M.CGNCR_MODELS)
    paired = M.official_paired(bt_base, cg_method)
    pm = M.paired_mae(paired, "PRE", cg_method)
    pp = M.paired_mae(paired, "PRE", M.OFFICIAL)

    rows = []
    origins = x.index[x.index >= pd.Timestamp("2008-01-01")]
    for o in origins:
        fut = x.loc[o + pd.offsets.MonthBegin(1):].iloc[:M.FC_MONTHS]
        if fut.empty:
            continue
        n = len(fut)
        paths = {
            BOTTOMUP: forecast(comp, o, h=n, spec=spec),
            PRODUCTION: production_forecast(x, o, vint, cg_method, pm, pp, h=n),
            M.OFFICIAL: M.official_vintage_forecast(x, o, vint, h=n),
            cg_method: M.cgncr_forecast(x, o, cg_method, h=n),
            "smly": M.cgncr_forecast(x, o, "smly", h=n),
        }
        for meth, fc in paths.items():
            fc = np.asarray(fc, dtype=float)
            for h in M.HORIZONS:
                if n < h:
                    continue
                f = fc[:h]
                rows.append({
                    "origin": o, "h": h, "model": meth,
                    "fc": float(f.sum()) if np.isfinite(f).all() else np.nan,
                    "actual": float(fut.iloc[:h].sum()),
                    "target_end": fut.index[h - 1],
                })
    bt = pd.DataFrame(rows)
    bt["abs_err"] = (bt["fc"] - bt["actual"]).abs()
    bt["window"] = M._window(bt)
    bt.attrs["cg_method"] = cg_method
    return bt


def mean_mae(bt: pd.DataFrame, window: str, model: str) -> dict[int, float]:
    sub = bt[(bt["window"] == window) & (bt["model"] == model)].dropna(subset=["abs_err"])
    return {int(h): float(g["abs_err"].mean()) for h, g in sub.groupby("h")}


def ship_rule(bt: pd.DataFrame, band: float = M.EDGE_BAND) -> dict:
    """PRE-only ship decision vs the production hybrid."""
    bu = mean_mae(bt, "PRE", BOTTOMUP)
    prod = mean_mae(bt, "PRE", PRODUCTION)
    if not bu or not prod:
        return {"ship": False, "reason": "empty PRE scores", "bu_pre": bu, "prod_pre": prod}
    hs = [h for h in M.HORIZONS if h in bu and h in prod]
    bu_mean = float(np.mean([bu[h] for h in hs]))
    prod_mean = float(np.mean([prod[h] for h in hs]))
    worse = [h for h in hs if bu[h] > prod[h] + 1e-12]
    edge = (prod_mean - bu_mean) / prod_mean if prod_mean > 0 else float("nan")
    ship = bool(np.isfinite(edge) and edge > band and len(worse) <= 1)
    oos_bu = mean_mae(bt, "OOS", BOTTOMUP)
    oos_prod = mean_mae(bt, "OOS", PRODUCTION)
    return {
        "ship": ship,
        "band": band,
        "edge": edge,
        "worse_horizons": worse,
        "bu_pre": bu,
        "prod_pre": prod,
        "bu_pre_mean": bu_mean,
        "prod_pre_mean": prod_mean,
        "bu_oos": oos_bu,
        "prod_oos": oos_prod,
        "bu_oos_mean": float(np.mean([oos_bu[h] for h in M.HORIZONS if h in oos_bu])) if oos_bu else float("nan"),
        "prod_oos_mean": float(np.mean([oos_prod[h] for h in M.HORIZONS if h in oos_prod])) if oos_prod else float("nan"),
        "spec": dict(SPEC),
        "cg_method": bt.attrs.get("cg_method"),
        "reason": ("PRE mean MAE beats production by more than EDGE_BAND"
                   if ship else "PRE edge inside band or worse at >1 horizon"),
    }


def evaluate(comp: pd.DataFrame | None = None, vint: pd.DataFrame | None = None) -> dict:
    bt = backtest(comp=comp, vint=vint)
    decision = ship_rule(bt)
    decision["bt"] = bt
    decision["gate_pass"] = bool(decision["ship"])
    if not SHIP:
        decision["ship"] = False
        decision["reason"] = "v1 seasonal bottom-up rejected; see bottom_up_macro.py"
    return decision
