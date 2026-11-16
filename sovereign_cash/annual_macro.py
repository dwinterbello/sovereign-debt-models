"""Annual cash-requirement level × seasonal month shape.

Same split as the US fiscal flow model: the level is the fiscal object, the
calendar only allocates it.

The level is built bottom-up. Each subcomponent has its own annual regression
on the lag-1 driver in ``SPEC`` (receipts / tax → nominal GDP, current spending
→ CPI). Components with no driver repeat last year. The cash requirement is the
identity of those annual totals, then ``target_k`` spreads it across months.

Slope rules scored: ``floor`` (b≥0, intuitive), ``free``, ``unit`` (b=1, a=0).
On France the fitted GDP/CPI slopes are negative, so ``floor`` is a pure drift;
``unit`` is the least-bad component rule and still loses PRE to seasonals. The
chooser keeps the seasonal path until a component rule wins. Published plans
(LFI, DFP) still override the open year when present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.yoy_components import SKIP_YEARS, SPEC, components, deficit, macro

MIN_YEARS = 5
HORIZON = 12
SHAPE_K = 3
# how the annual YoY of a driven component is taken from its macro
SLOPE_RULES = ("floor", "free", "unit")


def annual(y: pd.Series) -> pd.Series:
    """Complete calendar-year totals. Incomplete years are dropped."""
    rows = {}
    for yr, g in y.groupby(y.index.year):
        if len(g) == 12 and g.notna().all():
            rows[int(yr)] = float(g.sum())
    return pd.Series(rows).sort_index()


def quarterly(y: pd.Series) -> pd.Series:
    """Complete calendar-quarter totals, stamped at quarter end (Mar/Jun/Sep/Dec)."""
    q = y.resample("QE").sum()
    counts = y.resample("QE").count()
    q = q[counts == 3]
    return q.dropna()


def _pairs(levels: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
           keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    """Annual YoY of ``levels`` on lag-1 macros named in ``keys``. Covid years out."""
    last = int(origin.year) - 1
    ys, xs = [], []
    for yr in levels.index:
        if yr > last or yr in SKIP_YEARS or (yr - 1) not in levels.index:
            continue
        row = []
        ok = True
        for k in keys:
            if (yr - 1) not in drivers[k].index:
                ok = False
                break
            row.append(float(drivers[k].loc[yr - 1]) / 100.0)
        if not ok:
            continue
        base = float(levels.loc[yr - 1])
        if abs(base) < 1e-9:
            continue
        ys.append(float(levels.loc[yr]) / base - 1.0)
        xs.append(row)
    if len(ys) < MIN_YEARS:
        return None
    return np.asarray(ys), np.asarray(xs)


def fit_level(levels: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
              keys: tuple[str, ...], rule: str = "floor") -> tuple[float, np.ndarray] | None:
    """YoY = a + B·macro. ``floor`` forces b≥0; ``unit`` is a=0, b=1; ``free`` is plain OLS."""
    if rule == "unit":
        return 0.0, np.ones(len(keys))
    packed = _pairs(levels, drivers, origin, keys)
    if packed is None:
        return None
    ys, xs = packed
    X = np.column_stack([np.ones(len(xs)), xs])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    slopes = coef[1:].astype(float)
    if rule == "floor":
        slopes = np.maximum(slopes, 0.0)
        intercept = float(np.mean(ys) - slopes @ np.mean(xs, axis=0))
    else:
        intercept = float(coef[0])
    return intercept, slopes


def _growth(levels: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
            keys: tuple[str, ...], rule: str = "floor") -> float:
    """One-year growth factor from last complete year's macro. 1.0 if the fit is thin."""
    packed = fit_level(levels, drivers, origin, keys, rule=rule)
    yr = int(origin.year) - 1
    if packed is None or any(yr not in drivers[k].index for k in keys):
        return 1.0
    a, b = packed
    x = np.array([float(drivers[k].loc[yr]) / 100.0 for k in keys])
    return 1.0 + float(a + b @ x)


def predict_year(levels: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
                 year: int, keys: tuple[str, ...], rule: str = "floor") -> float | None:
    """Full-year total for ``year`` from the last complete year, compounded with one growth.

    Macro is always the year before the origin (no AMECO lookahead into year Y). An open
    year and the next one share that same growth factor.
    """
    complete = [y for y in levels.index if y < origin.year]
    if not complete:
        return None
    base_yr = max(complete)
    level = float(levels.loc[base_yr])
    g = _growth(levels, drivers, origin, keys, rule=rule)
    return level * (g ** (year - base_yr))


def _identity(cc: str, parts: dict[str, float]) -> float:
    if cc == "fr":
        return parts["primary"] + parts["interest"] - parts["recettes"] - parts["speciaux"]
    if cc == "de":
        return parts["primary"] + parts["interest"] - parts["revenue"]
    return (parts["primary"] + parts["interest"] + parts["cap_exp"] - parts["tax_rev"]
            - parts["other_rev"] - parts["treasury_bal"])


def component_year(cc: str, frame: pd.DataFrame, drivers: dict[str, pd.Series],
                   origin: pd.Timestamp, year: int, rule: str = "floor") -> float | None:
    """Annual deficit from annual component levels, each on its own driver."""
    parts = {}
    for name, key in SPEC[cc]:
        ann = annual(frame[name].dropna())
        complete = [y for y in ann.index if y < origin.year]
        if not complete:
            return None
        if key is None:
            parts[name] = float(ann.loc[max(complete)])
            continue
        pred = predict_year(ann, drivers, origin, year, (key,), rule=rule)
        if pred is None:
            return None
        parts[name] = pred
    return _identity(cc, parts)


def component_coefs(cc: str, origin: pd.Timestamp, rule: str = "floor"
                    ) -> dict[str, tuple[float, float] | None]:
    """Per-component (intercept, slope) at ``origin`` for inspection."""
    frame = components(cc)
    drivers = macro(cc)
    out = {}
    for name, key in SPEC[cc]:
        if key is None:
            out[name] = None
            continue
        ann = annual(frame[name].dropna())
        packed = fit_level(ann, drivers, origin, (key,), rule=rule)
        out[name] = (float(packed[0]), float(packed[1][0])) if packed else None
    return out


def targets_asof(y: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
                 keys: tuple[str, ...] | None, cc: str | None = None,
                 frame: pd.DataFrame | None = None, rule: str = "floor"
                 ) -> dict[int, float] | None:
    """Calendar-year totals known at the origin for the open year and the next one.

    Pass ``cc`` + ``frame`` for the component identity. Pass ``keys`` for a single
    regression on the aggregate cash requirement (research only).
    """
    years = {int(origin.year), int(origin.year) + 1}
    out = {}
    if cc and frame is not None:
        for yr in years:
            v = component_year(cc, frame, drivers, origin, yr, rule=rule)
            if v is None:
                return None
            out[yr] = v
        return out
    levels = annual(y.loc[:origin])
    if levels.empty:
        return None
    for yr in years:
        if keys is not None:
            v = predict_year(levels, drivers, origin, yr, keys, rule=rule)
        else:
            complete = [y for y in levels.index if y < origin.year]
            v = float(levels.loc[max(complete)]) if complete else None
        if v is None:
            return None
        out[yr] = v
    return out


def component_book(cc: str, rule: str = "unit") -> dict[pd.Timestamp, dict[int, float]]:
    """Per-origin annual deficit targets from the component regressions."""
    frame = components(cc)
    y = deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()
    drivers = macro(cc)
    out = {}
    for origin in y.index:
        tg = targets_asof(y, drivers, origin, None, cc=cc, frame=frame, rule=rule)
        if tg:
            out[origin] = tg
    return out


def forecast_monthly(y: pd.Series, origin: pd.Timestamp, h: int, targets: dict[int, float],
                     k: int = SHAPE_K) -> pd.Series | None:
    """Spread annual targets with the additive seasonal profile (``target_k``)."""
    return C.forecast(y, origin, h, f"target_{k}", targets=targets)


def score(cc: str, specs: dict[str, tuple[str, ...] | str] | None = None) -> pd.DataFrame:
    """Walk-forward cumulative MAE. Specs: macro key tuples, ``smly``, ``seas``, ``comp``."""
    frame = components(cc)
    y = deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()
    drivers = macro(cc)
    specs = specs or {
        "smly": "smly",
        "seas": "seas",
        "ngdp": ("ngdp",),
        "ngdp_cpi": ("ngdp", "cpi"),
        "cpi": ("cpi",),
        "comp": "comp",
        "last": "last",
    }
    rows = []
    for origin in y.index:
        end = origin + pd.DateOffset(months=HORIZON)
        act = y.loc[origin + pd.offsets.MonthBegin(1):end]
        if len(act) != HORIZON or act.isna().any():
            continue
        rec = {"origin": origin, "end": end, "actual": float(act.sum())}
        for name, spec in specs.items():
            if spec == "smly":
                fc = C.forecast(y, origin, HORIZON, "smly")
            elif spec == "seas":
                fc = C.forecast(y, origin, HORIZON, "seas_mult_3")
                if fc is None:
                    fc = C.forecast(y, origin, HORIZON, "seas_add_3")
            elif spec == "last":
                tg = targets_asof(y, drivers, origin, None)
                fc = forecast_monthly(y, origin, HORIZON, tg) if tg else None
            elif spec == "comp":
                tg = targets_asof(y, drivers, origin, None, cc=cc, frame=frame)
                fc = forecast_monthly(y, origin, HORIZON, tg) if tg else None
            else:
                tg = targets_asof(y, drivers, origin, spec)
                fc = forecast_monthly(y, origin, HORIZON, tg) if tg else None
            if fc is None or fc.isna().any():
                continue
            rec[name] = float(fc.sum())
        if len(rec) > 3:
            pre_end = pd.Timestamp("2021-12-01") if cc == "it" else pd.Timestamp("2019-12-01")
            rec["window"] = ("PRE" if end <= pre_end
                             else "OOS" if origin >= pd.Timestamp("2022-01-01") else "mid")
            rows.append(rec)
    return pd.DataFrame(rows)


def mae(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt
    cols = [c for c in bt.columns if c not in ("origin", "end", "actual", "window")]
    out = []
    for window, g in bt.groupby("window"):
        rec = {"window": window, "n": len(g)}
        for c in cols:
            rec[c] = float((g[c] - g["actual"]).abs().mean())
        out.append(rec)
    return pd.DataFrame(out)


def targets_book(y: pd.Series, drivers: dict[str, pd.Series],
                 keys: tuple[str, ...] = ("ngdp",)) -> dict[pd.Timestamp, dict[int, float]]:
    """Per-origin annual targets for ``annual_*`` models in ``core.cash_framework.backtest``."""
    out = {}
    for origin in y.dropna().index:
        tg = targets_asof(y, drivers, origin, keys)
        if tg:
            out[origin] = tg
    return out


def cash_series(cc: str) -> pd.Series:
    frame = components(cc)
    return deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()


def backtest_level(y: pd.Series, drivers: dict[str, pd.Series],
                   keys: tuple[str, ...] = ("ngdp",), name: str = "annual_ngdp_3",
                   horizons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Walk-forward rows: origin, h, model, fc, actual (same units as ``y``)."""
    y = y.dropna().sort_index()
    book = targets_book(y, drivers, keys)
    hs = horizons or tuple(C.HORIZONS)
    rows = []
    last = y.index.max()
    for origin, tg in book.items():
        fc = forecast_monthly(y, origin, max(hs), tg)
        if fc is None:
            continue
        for h in hs:
            end = origin + pd.DateOffset(months=h)
            if end > last:
                continue
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h or act.isna().any():
                continue
            rows.append({"origin": origin, "h": h, "model": name,
                         "fc": float(fc.iloc[:h].sum()), "actual": float(act.sum())})
    return pd.DataFrame(rows)


def forecast_path(y: pd.Series, origin: pd.Timestamp, h: int, drivers: dict[str, pd.Series],
                  keys: tuple[str, ...] = ("ngdp",)) -> pd.Series | None:
    tg = targets_asof(y, drivers, origin, keys)
    return forecast_monthly(y, origin, h, tg) if tg else None


def ytd_targets(y: pd.Series, origin: pd.Timestamp, k: int = SHAPE_K,
                floor: float = 1.0) -> dict[int, float] | None:
    """Open-year total = YTD / seasonal share elapsed; next year = last complete year.

    Same US-style level × shape split, with the level taken from within-year execution
    rather than a macro regression (the deficit's GDP slope floors to zero).
    """
    levels = annual(y.loc[:origin])
    complete = [yr for yr in levels.index if yr < origin.year]
    if not complete:
        return None
    last = float(levels.loc[max(complete)])
    if origin.month == 12:
        return {int(origin.year): float(y.loc[f"{origin.year}-01-01":origin].sum()),
                int(origin.year) + 1: last}
    share = C.seasonal_share(y, origin, k, floor)
    if share is None:
        return None
    w = float(share.loc[:origin.month].sum())
    if w <= 0.05:
        return None
    ytd = float(y.loc[f"{origin.year}-01-01":origin].sum())
    return {int(origin.year): ytd / w, int(origin.year) + 1: last}


def ytd_book(y: pd.Series) -> dict[pd.Timestamp, dict[int, float]]:
    out = {}
    for origin in y.dropna().index:
        tg = ytd_targets(y, origin)
        if tg:
            out[origin] = tg
    return out


def forecast_ytd(y: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    tg = ytd_targets(y, origin)
    return forecast_monthly(y, origin, h, tg) if tg else None


def score_quarters(cc: str) -> pd.DataFrame:
    """Same idea on quarterly totals: next-four-quarter sum vs last year / macro."""
    frame = components(cc)
    y = deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()
    drivers = macro(cc)
    q = quarterly(y)
    rows = []
    for origin in q.index:
        # origin is quarter-end; forecast the next four complete quarters
        future = q.loc[origin + pd.offsets.QuarterEnd(1):origin + pd.offsets.QuarterEnd(4)]
        if len(future) != 4:
            continue
        # need four quarters a year ago
        past = q.loc[origin - pd.offsets.QuarterEnd(3):origin]
        if len(past) != 4:
            continue
        smly = float(q.loc[origin - pd.DateOffset(years=1) + pd.offsets.QuarterEnd(1):
                           origin - pd.DateOffset(years=1) + pd.offsets.QuarterEnd(4)].sum())
        if not np.isfinite(smly):
            continue
        # annual-style: scale last four quarters by fitted calendar-year YoY
        levels = annual(y.loc[:origin])
        packed = fit_level(levels, drivers, origin, ("ngdp", "cpi"))
        scale = 1.0
        if packed is not None and (origin.year - 1) in drivers["ngdp"].index:
            a, b = packed
            x = np.array([float(drivers["ngdp"].loc[origin.year - 1]) / 100.0,
                          float(drivers["cpi"].loc[origin.year - 1]) / 100.0])
            scale = 1.0 + float(a + b @ x)
        rec = {"origin": origin, "actual": float(future.sum()), "smly": smly,
               "macro": float(past.sum()) * scale}
        pre_end = pd.Timestamp("2021-12-01") if cc == "it" else pd.Timestamp("2019-12-01")
        rec["window"] = ("PRE" if future.index.max() <= pre_end
                         else "OOS" if origin >= pd.Timestamp("2022-01-01") else "mid")
        rows.append(rec)
    return mae(pd.DataFrame(rows)) if rows else pd.DataFrame()
