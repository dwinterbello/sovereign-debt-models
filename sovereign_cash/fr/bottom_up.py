"""France bottom-up cash-requirement baseline (SMB components + AFT charge).

Rebuilds the State deficit from published monthly pieces:

  cash = primary + interest − recettes − speciaux

Rules (PRE-gated; see ``score`` / ``ship_decision``):

  recettes, primary   seas_mult_3 on each line's own history
  interest            AFT charge budgétaire × 3y month share when the charge
                      vintage is known at the origin; else last complete-year
                      carved interest × share, grown one year by the last
                      observed YoY for later calendar years
  speciaux            trailing-60m month-of-year mean (mean-reverting, no macro)

Live path still closes open years to the PLF/LFI ``déficit à financer`` via
``anchor_to_plan`` (same pin as production). The bottom-up supplies the month
shape. Ship only when mean PRE MAE beats production ``seas_mult_3`` by ≥5%
relative and is not worse at more than one horizon.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.yoy_components import SPEC, components, deficit

AUTH = Path(__file__).resolve().parent.parent / "data" / "fr" / "authority.json"
REL_EDGE = 0.05
SPECIAUX_MONTHS = 60
CHARGE_SHARE_K = 3
# AFT / LFI charge for year Y is treated as known from September Y−1 (budget autumn).
CHARGE_KNOWN_MONTH = 9

# v1 rejected as lazy (seas_mult + AFT charge). v2 is ``fr.bottom_up_macro``.
SHIP = False


def _smly(y: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    vals = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        if prev not in y.index or pd.isna(y.loc[prev]):
            return None
        vals.append(float(y.loc[prev]))
    return pd.Series(vals, index=months)


def _month_share(y: pd.Series, origin: pd.Timestamp, k: int = CHARGE_SHARE_K) -> pd.Series | None:
    shares = []
    for yr in range(int(origin.year) - 1, int(origin.year) - 12, -1):
        b = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        s = float(b.sum()) if len(b) == 12 and b.notna().all() else 0.0
        if s <= 0:
            continue
        shares.append(pd.Series(b.to_numpy() / s, index=range(1, 13)))
        if len(shares) == k:
            break
    if not shares:
        return None
    return pd.concat(shares, axis=1).mean(axis=1)


def _moy_mean(y: pd.Series, origin: pd.Timestamp, h: int, n: int = SPECIAUX_MONTHS) -> pd.Series:
    hist = y.dropna().loc[:origin].iloc[-n:]
    g = hist.groupby(hist.index.month).mean()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pd.Series([float(g.get(m.month, 0.0)) for m in months], index=months)


def _charge_table() -> dict[int, float]:
    if not AUTH.exists():
        return {}
    raw = json.loads(AUTH.read_text())["debt_charge"]["years"]
    return {int(y): float(r["bn"]) for y, r in raw.items()}


def _interest_path(interest: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    """AFT charge × month share when known; else last complete year (+ growth)."""
    hist = interest.dropna().loc[:origin]
    sh = _month_share(hist, origin)
    if sh is None:
        return _smly(hist, origin, h)
    tot0 = None
    for y2 in range(int(origin.year) - 1, int(origin.year) - 10, -1):
        b2 = hist.loc[f"{y2}-01-01":f"{y2}-12-01"]
        if len(b2) == 12 and b2.notna().all() and float(b2.sum()) > 0:
            tot0 = float(b2.sum())
            break
    if tot0 is None:
        return _smly(hist, origin, h)
    g = 0.0
    for y2 in range(int(origin.year) - 1, int(origin.year) - 6, -1):
        b2 = hist.loc[f"{y2}-01-01":f"{y2}-12-01"]
        b1 = hist.loc[f"{y2 - 1}-01-01":f"{y2 - 1}-12-01"]
        if len(b2) == 12 and len(b1) == 12 and float(b1.sum()) > 0:
            g = float(b2.sum()) / float(b1.sum()) - 1.0
            break
    charge = _charge_table()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = {}
    for m in months:
        yy = int(m.year)
        known = origin >= pd.Timestamp(yy - 1, CHARGE_KNOWN_MONTH, 1)
        if yy in charge and known:
            tot = charge[yy]
        elif yy <= origin.year:
            tot = tot0
        else:
            tot = tot0 * ((1.0 + g) ** (yy - origin.year))
        out[m] = tot * float(sh.loc[m.month])
    return pd.Series(out)


def forecast(origin: pd.Timestamp, h: int = 12,
             frame: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Monthly component paths and cash_requirement through origin+1..origin+h."""
    frame = components("fr") if frame is None else frame
    r = C.forecast(frame["recettes"].dropna(), origin, h, "seas_mult_3")
    p = C.forecast(frame["primary"].dropna(), origin, h, "seas_mult_3")
    i = _interest_path(frame["interest"], origin, h)
    s = _moy_mean(frame["speciaux"], origin, h)
    if r is None or p is None or i is None:
        return None
    parts = {"recettes": r, "primary": p, "interest": i, "speciaux": s}
    out = pd.DataFrame(parts)
    out["cash_requirement"] = deficit("fr", parts)
    return out


def path(origin: pd.Timestamp, h: int = 12,
         frame: pd.DataFrame | None = None) -> pd.Series | None:
    fc = forecast(origin, h, frame=frame)
    return None if fc is None else fc["cash_requirement"]


def production_path(y: pd.Series, origin: pd.Timestamp, h: int = 12) -> pd.Series | None:
    """Unanchored production shape (``seas_mult_3``) — PRE benchmark when no PLF vintage."""
    return C.forecast(y, origin, h, "seas_mult_3")


def backtest(frame: pd.DataFrame | None = None, step: int = 1) -> pd.DataFrame:
    """Walk-forward cumulative MAE rows: bottom-up vs production seas_mult_3."""
    frame = components("fr") if frame is None else frame
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    rows = []
    for i, origin in enumerate(y.index):
        if i % step:
            continue
        if origin.year < 2012:
            continue
        bu = path(origin, 12, frame)
        prod = production_path(y, origin, 12)
        if bu is None or prod is None:
            continue
        for h in C.HORIZONS:
            end = origin + pd.DateOffset(months=h)
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h or act.isna().any():
                continue
            rows.append({
                "origin": origin, "h": int(h),
                "bottomup": float(bu.iloc[:h].sum()),
                "production": float(prod.iloc[:h].sum()),
                "actual": float(act.sum()),
            })
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    ends = [o + pd.DateOffset(months=int(h)) for o, h in zip(bt["origin"], bt["h"])]
    bt["window"] = np.where(pd.to_datetime(ends) <= C.PRE_END, "PRE",
                            np.where(bt["origin"] >= C.OOS_START, "OOS", "mid"))
    return bt


def mae_table(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt
    rows = []
    for (w, h), g in bt.groupby(["window", "h"]):
        rows.append({
            "window": w, "h": int(h), "n": len(g),
            "bottomup": float((g["bottomup"] - g["actual"]).abs().mean()),
            "production": float((g["production"] - g["actual"]).abs().mean()),
        })
    return pd.DataFrame(rows)


def mean_mae(bt: pd.DataFrame, window: str, col: str) -> float | None:
    tab = mae_table(bt)
    g = tab[tab["window"] == window]
    if g.empty or set(g["h"]) != set(C.HORIZONS):
        return None
    return float(g.set_index("h").loc[list(C.HORIZONS), col].mean())


def ship_decision(bt: pd.DataFrame | None = None) -> dict:
    """PRE gate: ≥5% relative vs production, worse at ≤1 horizon."""
    bt = backtest() if bt is None else bt
    tab = mae_table(bt)
    pre = tab[tab["window"] == "PRE"].set_index("h")
    if pre.empty or set(pre.index) != set(C.HORIZONS):
        return {"ship": False, "reason": "PRE incomplete", "pre_bu": None, "pre_prod": None}
    bu = float(pre.loc[list(C.HORIZONS), "bottomup"].mean())
    prod = float(pre.loc[list(C.HORIZONS), "production"].mean())
    worse = int((pre["bottomup"] > pre["production"] + 1e-9).sum())
    rel = (prod - bu) / prod if prod else 0.0
    ok = bu < prod * (1.0 - REL_EDGE) and worse <= 1
    oos = tab[tab["window"] == "OOS"].set_index("h")
    oos_bu = float(oos.loc[list(C.HORIZONS), "bottomup"].mean()) if set(oos.index) >= set(C.HORIZONS) else None
    oos_prod = float(oos.loc[list(C.HORIZONS), "production"].mean()) if set(oos.index) >= set(C.HORIZONS) else None
    return {
        "ship": ok,
        "reason": (f"PRE bu {bu:.2f} vs prod {prod:.2f} ({rel:+.1%} rel, worse_h={worse})"
                   if ok else f"PRE fail: bu {bu:.2f} vs prod {prod:.2f} ({rel:+.1%}, worse_h={worse})"),
        "pre_bu": bu, "pre_prod": prod, "rel": rel, "worse_h": worse,
        "oos_bu": oos_bu, "oos_prod": oos_prod,
        "by_h": tab.to_dict(orient="records"),
    }


def authority_series_used() -> list[dict]:
    """Hand-maintained authority inputs this baseline reads."""
    auth = json.loads(AUTH.read_text()) if AUTH.exists() else {}
    charge = auth.get("debt_charge", {}).get("years", {})
    deficit = auth.get("state_deficit", {}).get("years", {})
    return [
        {"series": "AFT charge budgétaire de la dette (programme 117)",
         "path": "data/fr/authority.json → debt_charge",
         "years": {y: {"bn": r["bn"], "vintage": r.get("vintage")} for y, r in charge.items()},
         "use": "open-year interest annual pin × month share"},
        {"series": "PLF/LFI déficit à financer (solde budgétaire hors FMI)",
         "path": "data/fr/authority.json → state_deficit / plan.json → deficit_to_finance",
         "years": {y: {"bn": r["bn"], "vintage": r.get("vintage")} for y, r in deficit.items()},
         "use": "live annual close via anchor_to_plan (not in unanchored PRE score)"},
        {"series": "SMB recettes / dépenses / comptes spéciaux (INSEE BDM)",
         "path": "fr/fetch.load_smb → yoy_components.components",
         "use": "component history; interest carved via interest_split"},
    ]
