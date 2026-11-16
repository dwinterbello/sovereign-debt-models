"""Augment FR seasonals with macro (UK-style), do not replace them.

Base = seas_mult_3 / smly on SMB components.
Overlay = calendar month × (1 + Σ ε · driver_yoy/100), or additive residual
  cash_seas + β · (driver − train mean) from expanding OLS on past h=12 errors.

Drivers: FR/EZ/DE share prices (FRED), HICP YoY. CPI hits primary (indexed spend);
equity hits recettes (IS / capital tax channel). Gate vs seas_mult_3 (≥5% PRE, worse_h≤1).

Does not mutate SHIP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.monthly_components import monthly_drivers
from sovereign_cash.yoy_components import SPEC, components, deficit

ROOT = Path(__file__).resolve().parents[1]
PRE_END = pd.Timestamp("2019-12-01")
REL_EDGE = 0.05
HORIZONS = C.HORIZONS
STEP = 3
MIN_OLS = 24


def _fred_yoy(name: str) -> pd.Series:
    raw = json.loads((ROOT / "data" / "fr" / name).read_text())
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in raw["observations"]})
    s = s.sort_index()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return (s.pct_change(12) * 100.0).rename(name.replace(".json", ""))


def drivers() -> dict[str, pd.Series]:
    d = monthly_drivers("fr")
    out = {
        "hicp": d["hicp_yoy"].rename("hicp"),
        "ngdp": d["ngdp_yoy"].rename("ngdp"),
        "eq_fr": _fred_yoy("equity_fred.json").rename("eq_fr"),
        "eq_ez": _fred_yoy("equity_ez_fred.json").rename("eq_ez"),
        "eq_de": _fred_yoy("equity_de_fred.json").rename("eq_de"),
    }
    # pan-EU: mean of available FR/EZ/DE at each date
    eq = pd.concat([out["eq_fr"], out["eq_ez"], out["eq_de"]], axis=1)
    out["eq_eu"] = eq.mean(axis=1).rename("eq_eu")
    return out


def _yoy_at(s: pd.Series, origin: pd.Timestamp, lag: int) -> float:
    """Point-in-time YoY known at origin (lag months)."""
    t = origin - pd.DateOffset(months=lag)
    if t not in s.index or pd.isna(s.loc[t]):
        sub = s.loc[:origin].dropna()
        if sub.empty:
            return 0.0
        return float(sub.iloc[-1 - lag]) if len(sub) > lag else float(sub.iloc[-1])
    return float(s.loc[t])


def _seas_comp(y: pd.Series, origin: pd.Timestamp, h: int, model: str = "seas_mult_3") -> pd.Series | None:
    return C.forecast(y, origin, h, model)


def forecast_scale(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                   drv: dict[str, pd.Series],
                   eps_rec: float, eps_pri: float,
                   rec_key: str | None, pri_key: str | None,
                   lag: int, base: str = "seas_mult_3") -> pd.Series | None:
    """Component seasonal × (1 + ε · yoy/100); deficit identity."""
    cols = {}
    for name, _ in SPEC["fr"]:
        y = frame[name].dropna()
        fc = _seas_comp(y, origin, h, base)
        if fc is None:
            fc = _seas_comp(y, origin, h, "smly")
        if fc is None:
            return None
        if name == "recettes" and rec_key and eps_rec:
            g = eps_rec * _yoy_at(drv[rec_key], origin, lag) / 100.0
            fc = fc * (1.0 + g)
        elif name == "primary" and pri_key and eps_pri:
            g = eps_pri * _yoy_at(drv[pri_key], origin, lag) / 100.0
            fc = fc * (1.0 + g)
        cols[name] = fc
    return deficit("fr", cols)


def forecast_resid(origin: pd.Timestamp, h: int, y: pd.Series,
                   drv: dict[str, pd.Series], key: str, lag: int,
                   base: str = "seas_mult_3") -> pd.Series | None:
    """seas + expanding OLS residual on lagged driver (like Nordic skew overlay)."""
    seas = C.forecast(y, origin, h, base)
    if seas is None:
        return None
    # build training: past origins with realised h=12 error vs seas, on driver at that origin
    hist = y.dropna()
    xs, es = [], []
    # walk every month back with full year realised
    for o in hist.index:
        if o >= origin:
            break
        end = o + pd.DateOffset(months=12)
        if end > origin:  # only fully known errors
            continue
        act = hist.loc[o + pd.offsets.MonthBegin(1):end]
        if len(act) != 12:
            continue
        s = C.forecast(hist, o, 12, base)
        if s is None:
            continue
        xs.append(_yoy_at(drv[key], o, lag))
        es.append(float(act.sum() - s.sum()))
    if len(xs) < MIN_OLS:
        return seas  # no overlay yet
    X = np.asarray(xs, float)
    E = np.asarray(es, float)
    Xc = X - X.mean()
    if float(Xc.var()) < 1e-8:
        return seas
    beta = float(np.dot(Xc, E - E.mean()) / np.dot(Xc, Xc))
    # cap overlay ±15bn on the 12m sum, spread evenly
    x0 = _yoy_at(drv[key], origin, lag) - float(X.mean())
    add = float(np.clip(beta * x0, -15.0, 15.0)) / h
    return seas + add


def walk(specs: list[dict], step: int = STEP) -> dict[str, dict]:
    frame = components("fr")
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    drv = drivers()
    last = y.index.max()
    out = {}
    for spec in specs:
        name = spec["name"]
        rows = []
        for i, origin in enumerate(y.index):
            if i % step:
                continue
            end = origin + pd.DateOffset(months=12)
            if end > last or len(y.loc[:origin]) < 48:
                continue
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != 12 or act.isna().any():
                continue
            kind = spec["kind"]
            if kind == "scale":
                fc = forecast_scale(
                    origin, 12, frame, drv,
                    spec.get("eps_rec", 0.0), spec.get("eps_pri", 0.0),
                    spec.get("rec_key"), spec.get("pri_key"),
                    spec.get("lag", 2), spec.get("base", "seas_mult_3"),
                )
            elif kind == "resid":
                fc = forecast_resid(
                    origin, 12, y, drv, spec["key"], spec.get("lag", 2),
                    spec.get("base", "seas_mult_3"),
                )
            elif kind == "scale_both":
                # recettes×eq and primary×cpi in one
                fc = forecast_scale(
                    origin, 12, frame, drv,
                    spec["eps_rec"], spec["eps_pri"],
                    spec["rec_key"], spec["pri_key"],
                    spec.get("lag", 2),
                )
            else:
                raise ValueError(kind)
            seas = C.forecast(y, origin, 12, "seas_mult_3")
            if fc is None or seas is None:
                continue
            for h in HORIZONS:
                rows.append({
                    "origin": origin, "h": int(h),
                    "model": float(fc.iloc[:h].sum()),
                    "seas": float(seas.iloc[:h].sum()),
                    "actual": float(act.iloc[:h].sum()),
                })
        bt = pd.DataFrame(rows)
        if bt.empty:
            out[name] = {"ship": False, "reason": "empty", "pre": None}
            print(name, "empty")
            continue
        ends = [o + pd.DateOffset(months=int(h)) for o, h in zip(bt["origin"], bt["h"])]
        bt["window"] = np.where(pd.to_datetime(ends) <= PRE_END, "PRE",
                                np.where(bt["origin"] >= pd.Timestamp("2022-01-01"), "OOS", "mid"))
        pre = bt[bt["window"] == "PRE"]
        oos = bt[bt["window"] == "OOS"]
        def _mae(g):
            return g.groupby("h").apply(
                lambda x: pd.Series({
                    "model": (x["model"] - x["actual"]).abs().mean(),
                    "seas": (x["seas"] - x["actual"]).abs().mean(),
                }), include_groups=False)
        pt = _mae(pre)
        bu = float(pt["model"].mean()) if len(pt) else None
        se = float(pt["seas"].mean()) if len(pt) else None
        worse = int((pt["model"] > pt["seas"]).sum()) if len(pt) else 99
        ok = bu is not None and se and (bu <= se * (1 - REL_EDGE)) and worse <= 1
        oos_m = float(_mae(oos)["model"].mean()) if len(oos) else None
        oos_s = float(_mae(oos)["seas"].mean()) if len(oos) else None
        rel = (se - bu) / se if se else None
        g = {
            "ship": ok, "pre": bu, "seas_pre": se, "rel": rel, "worse_h": worse,
            "oos": oos_m, "seas_oos": oos_s,
            "reason": (f"PRE {bu:.2f} vs seas {se:.2f} ({rel:+.1%}, worse_h={worse})"
                       if bu is not None else "n/a"),
        }
        out[name] = g
        print(f"{name}: {g['reason']}  OOS={oos_m}")
    return out


def main() -> None:
    specs: list[dict] = []
    # --- CPI on primary (indexed-spend channel): Voies-style ε grid ---
    for eps in (0.25, 0.5, 0.75, 1.0, 1.5):
        for lag in (0, 1, 2, 3, 6, 12):
            specs.append({
                "name": f"scale|pri×cpi ε={eps} lag{lag}",
                "kind": "scale", "eps_pri": eps, "pri_key": "hicp",
                "eps_rec": 0.0, "rec_key": None, "lag": lag,
            })
    # --- equity on recettes (IS / capital): FR / EZ / DE / EU ---
    for eq in ("eq_fr", "eq_ez", "eq_de", "eq_eu"):
        for eps in (0.25, 0.5, 0.7, 1.0):
            for lag in (3, 6, 12):  # corporate tax lags
                specs.append({
                    "name": f"scale|rec×{eq} ε={eps} lag{lag}",
                    "kind": "scale", "eps_rec": eps, "rec_key": eq,
                    "eps_pri": 0.0, "pri_key": None, "lag": lag,
                })
    # --- joint: CPI spend + EU equity receipts (UK other_tax style) ---
    for eps_c, eps_e, lag in (
        (0.5, 0.5, 6), (1.0, 0.5, 6), (0.5, 0.7, 6), (1.0, 0.7, 12),
        (0.75, 0.5, 3), (1.0, 0.25, 6),
    ):
        specs.append({
            "name": f"joint|pri×cpi{eps_c}+rec×eq_eu{eps_e}|lag{lag}",
            "kind": "scale_both",
            "eps_pri": eps_c, "pri_key": "hicp",
            "eps_rec": eps_e, "rec_key": "eq_eu", "lag": lag,
        })
    # --- residual overlays (seas + β·driver) ---
    for key in ("hicp", "eq_eu", "eq_fr", "ngdp"):
        for lag in (1, 2, 3, 6, 12):
            specs.append({
                "name": f"resid|{key}|lag{lag}",
                "kind": "resid", "key": key, "lag": lag,
            })

    print(f"n_specs={len(specs)}")
    res = walk(specs)
    ships = {k: v for k, v in res.items() if v.get("ship")}
    print("\n=== SHIP ===")
    if not ships:
        print("none")
    else:
        for k, v in sorted(ships.items(), key=lambda kv: kv[1]["pre"]):
            print(k, v)
    ranked = sorted(
        ((k, v) for k, v in res.items() if v.get("pre") is not None),
        key=lambda kv: kv[1]["pre"],
    )
    print("\n=== best 20 PRE ===")
    for k, v in ranked[:20]:
        mark = "SHIP" if v.get("ship") else ""
        print(f"{v['pre']:.3f} rel={v.get('rel'):+.1%} w={v.get('worse_h')} OOS={v.get('oos')} {mark} {k}")
    # CPI-only best
    cpi = [(k, v) for k, v in ranked if "pri×cpi" in k or (k.startswith("resid|hicp"))]
    print("\n=== best CPI-linked ===")
    for k, v in cpi[:8]:
        print(f"{v['pre']:.3f} rel={v.get('rel'):+.1%} {k}")


if __name__ == "__main__":
    main()
