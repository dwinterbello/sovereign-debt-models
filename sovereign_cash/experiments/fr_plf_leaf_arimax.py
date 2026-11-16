"""FR cash: transform sweep + PLF-leaf monthly SARIMAX with desk equity / CPI.

Does not mutate SHIP. Gate = mean PRE MAE ≥5% better than seas_mult_3, worse_h ≤1.

Why synthetic leaves: INSEE SMB only publishes aggregate recettes / dépenses. PLF IR/IS/TVA
are annual. Monthly leaf_t = recettes_t × structural RFN share (latest PLF known at origin).
Leaf-specific exogs can still diverge forecasts; recombined via the SMB identity.

Transforms: level / mom / diff / yoy / qoq. Drivers: Eurostat HICP YoY (desk CPI proxy),
FRED SPASTT01FRM661N equity YoY, NGDP YoY. Equity→IS is the headline leaf trial.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import bottom_up_macro as BUM
from sovereign_cash.monthly_components import (
    MACRO_LAG, MIN_HIST, ORDER, SEASONAL, monthly_drivers, _smly_path,
)
from sovereign_cash.yoy_components import SPEC, components, deficit

ROOT = Path(__file__).resolve().parents[1]
EQUITY = ROOT / "data" / "fr" / "equity_fred.json"
PRE_END = pd.Timestamp("2019-12-01")
REL_EDGE = 0.05
HORIZONS = C.HORIZONS
STEP = 6  # search; winners re-checkable at step=3


def _equity_yoy() -> pd.Series:
    raw = json.loads(EQUITY.read_text())
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in raw["observations"]})
    s = s.sort_index()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return (s.pct_change(12) * 100.0).rename("equity_yoy")


def _transform(y: pd.Series, kind: str) -> pd.Series:
    if kind == "mom":
        return y.pct_change().replace([np.inf, -np.inf], np.nan)
    if kind == "qoq":
        # three-month % change on month-end levels (cash is monthly; QoQ ≈ pct_change(3))
        return y.pct_change(3).replace([np.inf, -np.inf], np.nan)
    if kind == "diff":
        return y.diff()
    if kind == "yoy":
        return y.pct_change(12).replace([np.inf, -np.inf], np.nan)
    return y


def _stitch(fc: pd.Series, y: pd.Series, origin: pd.Timestamp, kind: str) -> pd.Series | None:
    out = {}
    level = float(y.loc[origin])
    for m in fc.index:
        v = float(fc.loc[m])
        if kind == "mom":
            if level <= 0 and v <= -1:
                return None
            level = level * (1.0 + v)
        elif kind == "qoq":
            # invert 3m growth as one-month geometric step; refuse non-positive bases
            if level <= 0 or (1.0 + v) <= 0:
                return None
            level = level * float((1.0 + v) ** (1.0 / 3.0))
        elif kind == "diff":
            level = level + v
        elif kind == "yoy":
            base = y.get(m - pd.DateOffset(years=1), np.nan)
            if pd.isna(base):
                return None
            level = float(base) * (1.0 + v)
        else:
            level = v
        out[m] = level
    return pd.Series(out)


def _sarimax(z: pd.Series, origin: pd.Timestamp, h: int,
             exog: pd.Series | None) -> pd.Series | None:
    hist = z.loc[:origin].dropna()
    if len(hist) < MIN_HIST:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    x_hist = x_fut = None
    if exog is not None:
        x = exog.reindex(hist.index.union(months))
        x_hist = pd.DataFrame({exog.name: x.reindex(hist.index)})
        ok = x_hist.dropna().index
        if len(ok) < MIN_HIST:
            return None
        hist = hist.loc[ok]
        x_hist = x_hist.loc[ok]
        fut = x.reindex(months)
        if fut.isna().any():
            last = float(x.loc[:origin].dropna().iloc[-1])
            fut = fut.fillna(last)
        x_fut = pd.DataFrame({exog.name: fut})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                hist, exog=x_hist, order=ORDER, seasonal_order=SEASONAL,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=80)
            fc = res.forecast(h, exog=x_fut)
    except Exception:
        return None
    return pd.Series(np.asarray(fc), index=months)


def fit_forecast(y: pd.Series, origin: pd.Timestamp, h: int,
                 exog: pd.Series | None = None, kind: str = "level") -> pd.Series | None:
    z = _transform(y, kind)
    fc = _sarimax(z, origin, h, exog)
    if fc is None:
        return None
    return _stitch(fc, y, origin, kind)


def _lag(s: pd.Series, k: int) -> pd.Series:
    return s.shift(k).rename(s.name)


def driver_panel() -> dict[str, pd.Series]:
    d = monthly_drivers("fr")
    eq = _equity_yoy()
    return {
        "hicp_yoy": d["hicp_yoy"],
        "ngdp_yoy": d["ngdp_yoy"],
        "rates": d.get("rates", pd.Series(dtype=float)),
        "equity_yoy": eq,
    }


def _exog(name: str | None, drivers: dict[str, pd.Series], lag: int = MACRO_LAG) -> pd.Series | None:
    if name is None:
        return None
    if name not in drivers:
        return None
    return _lag(drivers[name], lag).rename(name)


# --- A: aggregate SMB transform × driver sweep ---------------------------------

def forecast_agg(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                 drivers: dict[str, pd.Series], mode: str,
                 recettes_exog: str | None, primary_exog: str | None,
                 lag: int = MACRO_LAG) -> pd.DataFrame | None:
    cols = {}
    for name, key in SPEC["fr"]:
        y = frame[name].dropna()
        if name == "recettes":
            ex = _exog(recettes_exog, drivers, lag)
        elif name == "primary":
            ex = _exog(primary_exog, drivers, lag)
        elif name == "interest":
            ex = _exog("rates" if key else None, drivers, lag)
        else:
            ex = None
        kind = mode
        if mode in ("mom", "qoq", "yoy") and name not in ("recettes", "primary"):
            kind = "diff"
        fc = fit_forecast(y, origin, h, ex, kind=kind)
        if fc is None and name == "interest":
            fc = _smly_path(y, origin, h)
        if fc is None:
            return None
        cols[name] = fc
    out = pd.DataFrame(cols)
    out["cash_requirement"] = deficit("fr", {k: out[k] for k, _ in SPEC["fr"]})
    return out


# --- B: PLF leaf monthly SARIMAX ------------------------------------------------

TAX = list(BUM.TAX_KEYS)  # ir, is, tva, accises, autres_fiscales
LEAF_DRIVER = {
    "ir": "ngdp_yoy",          # masse salariale proxy
    "is": "equity_yoy",        # desk equity → IS
    "tva": "ngdp_yoy",
    "accises": "hicp_yoy",
    "autres_fiscales": "equity_yoy",
    "non_fiscal": None,
    "credits_ministeriels": "hicp_yoy",
    "cas_pensions": "hicp_yoy",
    "remboursements_locaux": "hicp_yoy",
    "psr": "ngdp_yoy",
}


def _monthly_leaves(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame | None:
    """Allocate monthly recettes/primary into PLF leaves via structural shares at origin."""
    cats = BUM._cat_table()
    tax_sh = BUM._structural_tax_shares(cats, origin)
    spend_sh = BUM._structural_spend_shares(cats, origin)
    nf_share = 0.06
    for y in sorted((yy for yy in cats if BUM._plf_known(origin, yy) or yy < origin.year), reverse=True):
        row = cats[y]
        rfn, nf = row.get("recettes_fiscales_nettes"), row.get("non_fiscal")
        if rfn is not None and nf is not None and float(rfn) + float(nf) > 0:
            nf_share = float(nf) / (float(rfn) + float(nf))
            break
    rec = frame["recettes"].dropna().loc[:origin]
    pri = frame["primary"].dropna().loc[:origin]
    inte = frame["interest"].dropna().loc[:origin]
    spe = frame["speciaux"].dropna().loc[:origin]
    if rec.empty or pri.empty:
        return None
    idx = rec.index.intersection(pri.index)
    nf = rec.loc[idx] * nf_share
    rfn = rec.loc[idx] - nf
    out = {k: rfn * tax_sh[k] for k in TAX}
    out["non_fiscal"] = nf
    out.update({k: pri.loc[idx] * spend_sh[k] for k in BUM.SPEND_KEYS})
    out["charge"] = inte.reindex(idx)
    out["speciaux"] = spe.reindex(idx)
    return pd.DataFrame(out)


def forecast_leaves(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                    drivers: dict[str, pd.Series], mode: str,
                    lag: int = MACRO_LAG,
                    leaf_driver: dict[str, str | None] | None = None) -> pd.DataFrame | None:
    leaves = _monthly_leaves(frame, origin)
    if leaves is None:
        return None
    leaf_driver = leaf_driver or LEAF_DRIVER
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fc_leaves = {}
    for col in list(TAX) + ["non_fiscal"] + list(BUM.SPEND_KEYS):
        y = leaves[col].dropna()
        dname = leaf_driver.get(col)
        ex = _exog(dname, drivers, lag) if dname else None
        # percent transforms only on strictly-positive tax / spend leaves
        pct_ok = col in ("ir", "is", "tva", "accises", "autres_fiscales",
                         "credits_ministeriels", "cas_pensions", "psr")
        kind = mode if (mode in ("level", "diff") or pct_ok) else "diff"
        if mode in ("mom", "qoq", "yoy") and not pct_ok:
            kind = "diff"
        fc = fit_forecast(y, origin, h, ex, kind=kind)
        if fc is None:
            return None
        fc_leaves[col] = fc
    # charge / speciaux: smly (interest carve)
    for name, src in (("charge", "interest"), ("speciaux", "speciaux")):
        y = frame[src].dropna()
        fc = _smly_path(y, origin, h)
        if fc is None:
            fc = fit_forecast(y, origin, h, None, kind="diff")
        if fc is None:
            return None
        fc_leaves[name if name != "charge" else "charge"] = fc
    recettes = sum(fc_leaves[k] for k in TAX) + fc_leaves["non_fiscal"]
    primary = sum(fc_leaves[k] for k in BUM.SPEND_KEYS)
    out = pd.DataFrame({
        "recettes": recettes,
        "primary": primary,
        "interest": fc_leaves["charge"],
        "speciaux": fc_leaves["speciaux"],
    }, index=months)
    out["cash_requirement"] = deficit("fr", {k: out[k] for k, _ in SPEC["fr"]})
    return out


def _score_rows(rows: list[dict]) -> pd.DataFrame:
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    ends = [o + pd.DateOffset(months=int(h)) for o, h in zip(bt["origin"], bt["h"])]
    bt["window"] = np.where(pd.to_datetime(ends) <= PRE_END, "PRE",
                            np.where(bt["origin"] >= pd.Timestamp("2022-01-01"), "OOS", "mid"))
    return bt


def mae_table(bt: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    if bt.empty or "window" not in bt.columns:
        return pd.DataFrame(columns=["window", "h", "n", "model", "seas"])
    rows = []
    for (w, h), g in bt.groupby(["window", "h"]):
        rows.append({
            "window": w, "h": int(h), "n": len(g),
            "model": float((g[col] - g["actual"]).abs().mean()),
            "seas": float((g["seas"] - g["actual"]).abs().mean()),
        })
    return pd.DataFrame(rows)


def gate(tab: pd.DataFrame) -> dict:
    pre = tab[tab["window"] == "PRE"].set_index("h")
    if pre.empty or not set(HORIZONS).issubset(pre.index):
        return {"ship": False, "reason": "PRE incomplete", "pre": None, "seas": None}
    bu = float(pre.loc[list(HORIZONS), "model"].mean())
    seas = float(pre.loc[list(HORIZONS), "seas"].mean())
    worse = int((pre.loc[list(HORIZONS), "model"] > pre.loc[list(HORIZONS), "seas"]).sum())
    ok = (bu <= seas * (1 - REL_EDGE)) and worse <= 1
    oos = tab[tab["window"] == "OOS"]
    oos_m = float(oos.groupby("h")["model"].mean().mean()) if len(oos) else None
    oos_s = float(oos.groupby("h")["seas"].mean().mean()) if len(oos) else None
    return {
        "ship": ok,
        "pre": bu, "seas_pre": seas, "rel": (seas - bu) / seas if seas else None,
        "worse_h": worse, "oos": oos_m, "seas_oos": oos_s,
        "reason": (f"PRE {bu:.2f} vs seas {seas:.2f} ({(seas-bu)/seas:+.1%}, worse_h={worse})"
                   if ok else f"PRE fail {bu:.2f} vs seas {seas:.2f} "
                             f"({(seas-bu)/seas if seas else 0:+.1%}, worse_h={worse})"),
    }


def walk_agg(specs: list[dict], step: int = STEP) -> dict[str, dict]:
    frame = components("fr")
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    drivers = driver_panel()
    last = y.index.max()
    out = {}
    for spec in specs:
        name = spec["name"]
        rows = []
        for i, origin in enumerate(y.index):
            if i % step:
                continue
            end = origin + pd.DateOffset(months=12)
            if end > last or len(y.loc[:origin]) < MIN_HIST:
                continue
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != 12 or act.isna().any():
                continue
            fc = forecast_agg(origin, 12, frame, drivers, spec["mode"],
                              spec.get("recettes_exog"), spec.get("primary_exog"),
                              lag=spec.get("lag", MACRO_LAG))
            seas = C.forecast(y, origin, 12, "seas_mult_3")
            if fc is None or seas is None:
                continue
            for h in HORIZONS:
                rows.append({
                    "origin": origin, "h": int(h),
                    "model": float(fc["cash_requirement"].iloc[:h].sum()),
                    "seas": float(seas.iloc[:h].sum()),
                    "actual": float(act.iloc[:h].sum()),
                })
        bt = _score_rows(rows)
        tab = mae_table(bt)
        g = gate(tab)
        g["n_pre"] = int(len(bt[bt["window"] == "PRE"])) if len(bt) else 0
        out[name] = g
        print(f"AGG {name}: {g['reason']}  OOS={g.get('oos')}")
    return out


def walk_leaves(specs: list[dict], step: int = STEP) -> dict[str, dict]:
    frame = components("fr")
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    drivers = driver_panel()
    last = y.index.max()
    out = {}
    for spec in specs:
        name = spec["name"]
        ld = dict(LEAF_DRIVER)
        ld.update(spec.get("leaf_driver", {}))
        rows = []
        for i, origin in enumerate(y.index):
            if i % step:
                continue
            end = origin + pd.DateOffset(months=12)
            if end > last or len(y.loc[:origin]) < MIN_HIST:
                continue
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != 12 or act.isna().any():
                continue
            fc = forecast_leaves(origin, 12, frame, drivers, spec["mode"],
                                 lag=spec.get("lag", MACRO_LAG), leaf_driver=ld)
            seas = C.forecast(y, origin, 12, "seas_mult_3")
            if fc is None or seas is None:
                continue
            for h in HORIZONS:
                rows.append({
                    "origin": origin, "h": int(h),
                    "model": float(fc["cash_requirement"].iloc[:h].sum()),
                    "seas": float(seas.iloc[:h].sum()),
                    "actual": float(act.iloc[:h].sum()),
                })
        bt = _score_rows(rows)
        tab = mae_table(bt)
        g = gate(tab)
        g["n_pre"] = int(len(bt[bt["window"] == "PRE"])) if len(bt) else 0
        out[name] = g
        print(f"LEAF {name}: {g['reason']}  OOS={g.get('oos')}")
    return out


def main() -> None:
    assert EQUITY.exists(), f"missing {EQUITY}"
    # Full transform×driver grid already scored once (mom/diff/yoy blow up; level ties/loses).
    # This pass: qoq + equity-lag screen + leaf ARIMAX.
    agg_specs = []
    for mode in ("qoq",):
        for rex, pex, tag in (
            ("ngdp_yoy", "hicp_yoy", "ngdp+hicp"),
            ("equity_yoy", "hicp_yoy", "equity+hicp"),
            (None, None, "noexog"),
        ):
            agg_specs.append({
                "name": f"{mode}|{tag}|lag2",
                "mode": mode, "recettes_exog": rex, "primary_exog": pex, "lag": 2,
            })
    for lag in (0, 1, 3, 6, 12):
        agg_specs.append({
            "name": f"level|equity+hicp|lag{lag}",
            "mode": "level", "recettes_exog": "equity_yoy", "primary_exog": "hicp_yoy", "lag": lag,
        })
    # reference: shipped-class level noexog / ngdp
    agg_specs.extend([
        {"name": "level|ngdp+hicp|lag2", "mode": "level",
         "recettes_exog": "ngdp_yoy", "primary_exog": "hicp_yoy", "lag": 2},
        {"name": "level|noexog", "mode": "level",
         "recettes_exog": None, "primary_exog": None, "lag": 2},
    ])

    print("=== A: aggregate SMB SARIMAX transform × drivers ===")
    agg = walk_agg(agg_specs)

    leaf_specs = []
    for mode in ("level", "mom", "qoq", "yoy"):
        leaf_specs.append({"name": f"plf_{mode}|eq→IS+cpi|lag2", "mode": mode, "lag": 2})
        leaf_specs.append({
            "name": f"plf_{mode}|eq→IS,tva+cpi→rest|lag2", "mode": mode, "lag": 2,
            "leaf_driver": {"tva": "hicp_yoy", "ir": "hicp_yoy"},
        })
        leaf_specs.append({
            "name": f"plf_{mode}|all_equity_tax|lag2", "mode": mode, "lag": 2,
            "leaf_driver": {k: "equity_yoy" for k in TAX},
        })
        leaf_specs.append({
            "name": f"plf_{mode}|all_cpi|lag2", "mode": mode, "lag": 2,
            "leaf_driver": {k: "hicp_yoy" for k in list(TAX) + list(BUM.SPEND_KEYS)},
        })
    for lag in (0, 1, 3, 6):
        leaf_specs.append({
            "name": f"plf_level|eq→IS|lag{lag}", "mode": "level", "lag": lag,
        })

    print("=== B: PLF-leaf monthly SARIMAX (synthetic shares × leaf exog) ===")
    leaf = walk_leaves(leaf_specs)

    ships = {**{f"agg/{k}": v for k, v in agg.items() if v.get("ship")},
             **{f"leaf/{k}": v for k, v in leaf.items() if v.get("ship")}}
    print("\n=== SHIP clearers ===")
    if not ships:
        print("none")
    else:
        for k, v in sorted(ships.items(), key=lambda kv: kv[1]["pre"] or 99):
            print(k, v)

    # dump ranked PRE
    allr = {**{f"agg/{k}": v for k, v in agg.items()},
            **{f"leaf/{k}": v for k, v in leaf.items()}}
    ranked = sorted(
        ((k, v) for k, v in allr.items() if v.get("pre") is not None),
        key=lambda kv: kv[1]["pre"],
    )
    print("\n=== PRE rank (top 15) ===")
    for k, v in ranked[:15]:
        print(f"{v['pre']:.2f}  rel={v.get('rel')}  worse={v.get('worse_h')}  OOS={v.get('oos')}  {k}")
    print("\n=== PRE rank (worst 5) ===")
    for k, v in ranked[-5:]:
        print(f"{v['pre']:.2f}  {k}  {v['reason']}")


if __name__ == "__main__":
    main()
