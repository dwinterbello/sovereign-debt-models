"""Sweden: seasonal budget-balance forecast, walk-forward scoring, 12-month forward path.

Target: central-government budget balance y_m (bn SEK, surplus +) = − net borrowing requirement.

Candidate models, all forecasts of y_{o+h} from data through origin o (h = 1..12):

    smly                ŷ = y_{t−12}                                              (naive benchmark)
    seas_k{K}_a{a}_{f}  ŷ = f_j(y_{t−12j}) + a · (L_o − f_j(L_{o−12j})),  j = 1..K

L_o is the trailing 12-month mean of y at the origin, f is mean or median. The month profile is
additive (the balance changes sign within the year, so a multiplicative profile is meaningless).
a = 0 is a K-year same-month average; a = 1 shifts it by the change in the trailing level.

Selection uses PRE target months only (2012-01..2019-12), mean MAE over h = 1..12. OOS
(2022-01→) is reported, never used to choose. Riksgälden's own published monthly forecast
(press releases) is a free benchmark, scored on the months where it exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRE = (pd.Timestamp("2012-01-01"), pd.Timestamp("2019-12-01"))
OOS_START = pd.Timestamp("2022-01-01")
HMAX = 12
SCORE_H = (1, 3, 6, 12)
GRID_K = (1, 2, 3, 5)
GRID_A = (0.0, 0.5, 1.0)
GRID_F = ("mean", "median")


def variants() -> list[tuple[str, int, float, str]]:
    out = [("smly", 1, 0.0, "mean")]
    for k in GRID_K:
        for a in GRID_A:
            for f in GRID_F:
                if k == 1 and f == "median":
                    continue
                out.append((f"seas_k{k}_a{a:g}_{f}", k, a, f))
    return out


def forecast(y: pd.Series, origin: pd.Timestamp, k: int, a: float, f: str, hmax: int = HMAX) -> pd.Series:
    """Forecasts for origin+1..origin+hmax from y (monthly, month-start index) through origin."""
    agg = np.nanmean if f == "mean" else np.nanmedian
    hist = y.loc[:origin]
    lvl = hist.rolling(12, min_periods=12).mean()
    pos = {m: i for i, m in enumerate(hist.index)}
    io = pos[origin]
    vals = hist.to_numpy()
    lv = lvl.to_numpy()
    out = {}
    for h in range(1, hmax + 1):
        t = origin + pd.DateOffset(months=h)
        js = [j for j in range(1, k + 12) if h - 12 * j <= 0][:k]
        same = [vals[io + h - 12 * j] for j in js if io + h - 12 * j >= 0]
        if not same:
            continue
        base = agg(same)
        if a:
            past = [lv[io - 12 * j] for j in range(1, k + 1) if io - 12 * j >= 0]
            adj = a * (lv[io] - agg(past)) if past and np.isfinite(lv[io]) else 0.0
            base = base + (adj if np.isfinite(adj) else 0.0)
        out[t] = float(base)
    return pd.Series(out, dtype=float)


def backtest(y: pd.Series, start: pd.Timestamp = pd.Timestamp("2010-12-01")) -> pd.DataFrame:
    rows = []
    origins = [o for o in y.index if o >= start and o < y.index[-1]]
    for name, k, a, f in variants():
        for o in origins:
            fc = forecast(y, o, k, a, f)
            for t, v in fc.items():
                if t in y.index:
                    rows.append((name, o, t, (t.year - o.year) * 12 + t.month - o.month, v, y[t]))
    bt = pd.DataFrame(rows, columns=["model", "origin", "target", "h", "fc", "actual"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    return bt


def window(t: pd.Series) -> pd.Series:
    w = pd.Series("", index=t.index)
    w[(t >= PRE[0]) & (t <= PRE[1])] = "PRE"
    w[t >= OOS_START] = "OOS"
    return w


def select(bt: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    b = bt.assign(win=window(bt["target"]))
    tab = b[b["win"] != ""].groupby(["model", "win"])["ae"].mean().unstack()
    return str(tab["PRE"].idxmin()), tab.sort_values("PRE")


def score_table(bt: pd.DataFrame, chosen: str, rg: pd.DataFrame) -> list[dict]:
    """Model vs same-month-last-year (all months) and vs Riksgälden's forecast (months it exists)."""
    b = bt[bt["model"].isin([chosen, "smly"])].assign(win=window(bt["target"]))
    b = b[b["win"] != ""]
    rgm = rg.set_index("month")["rg_forecast_bn"].dropna()
    out = []
    for win in ("PRE", "OOS"):
        for h in SCORE_H:
            s = b[(b["win"] == win) & (b["h"] == h)]
            m = s[s["model"] == chosen].set_index("target")
            n = s[s["model"] == "smly"].set_index("target")
            common = m.index.intersection(rgm.index)
            rg_ae = (rgm.reindex(common) - m.loc[common, "actual"]).abs()
            out.append({
                "window": win, "h": h, "model": round(float(m["ae"].mean()), 2),
                "benchmark": round(float(n["ae"].mean()), 2), "benchmark_name": "same month last year",
                "n": int(len(m)),
                "riksgalden": round(float(rg_ae.mean()), 2) if len(common) else None,
                "model_on_rg_months": round(float(m.loc[common, "ae"].mean()), 2) if len(common) else None,
                "n_rg": int(len(common)),
                "model_name": chosen, "unit": "bn SEK, MAE of monthly budget balance",
            })
    return out


def choose_params(name: str) -> tuple[int, float, str]:
    for n, k, a, f in variants():
        if n == name:
            return k, a, f
    raise KeyError(name)


# --------------------------------------------------------------------------------------
# Anchoring: spread (annual total − months already known − model months) across the model months.

ALLOC_METHODS = ("uniform", "abs_share", "err_share", "hist_share")


def month_dispersion(y: pd.Series, origin: pd.Timestamp, k: int = 10) -> pd.Series:
    """Mean |y_m − y_{m−12}| by calendar month over the last k years through origin."""
    h = y.loc[:origin]
    d = (h - h.shift(12)).dropna()
    d = d[d.index > origin - pd.DateOffset(years=k)]
    return d.abs().groupby(d.index.month).mean()


def hist_shares(y: pd.Series, origin: pd.Timestamp, months: pd.DatetimeIndex, k: int = 10) -> pd.Series:
    """Pooled share of each month in the same set of calendar months over the last k complete years."""
    num = pd.Series(0.0, index=months)
    den = 0.0
    for j in range(1, k + 1):
        past = [m - pd.DateOffset(years=j) for m in months]
        if past[-1] > origin or past[0] not in y.index:
            continue
        v = y.reindex(past).to_numpy()
        if np.isnan(v).any():
            continue
        num += v
        den += v.sum()
    return num / den if den else pd.Series(np.nan, index=months)


def allocate(raw: pd.Series, total: float, method: str, y: pd.Series | None = None,
             origin: pd.Timestamp | None = None) -> pd.Series:
    """Model months ``raw`` (one calendar year's remaining months) rescaled to sum to ``total``.

    uniform    raw + gap / n                      (equal additive shift)
    abs_share  raw + gap · |raw| / Σ|raw|         (gap lands where the model has the large flows;
                                                   sign-safe when months change sign)
    err_share  raw + gap · d_m / Σd               (d_m = trailing mean |y_m − y_{m−12}|: gap lands in
                                                   the months that historically move most)
    hist_share total · pooled historical share of each month in the same months, falling back to
               abs_share when the pooled denominator is small or opposite in sign to ``total``
    """
    gap = total - float(raw.sum())
    if method == "uniform":
        return raw + gap / len(raw)
    if method == "err_share" and y is not None and origin is not None:
        w = month_dispersion(y, origin).reindex(raw.index.month).to_numpy()
        if np.isfinite(w).all() and w.sum() > 0:
            return raw + gap * pd.Series(w / w.sum(), index=raw.index)
    if method == "hist_share" and y is not None and origin is not None:
        s = hist_shares(y, origin, raw.index)
        if s.notna().all() and float(s.abs().max()) <= 3.0:
            return total * s
    w = raw.abs()
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(raw), index=raw.index)
    return raw + gap * w


def allocation_backtest(y: pd.Series, chosen: str, start: pd.Timestamp = pd.Timestamp("2011-12-01")) -> pd.DataFrame:
    """Shape-only score: anchor each rest-of-year (and the following calendar year) to the *realised*
    annual total, so the only thing that differs between methods is where the gap lands."""
    k, a, f = choose_params(chosen)
    rows = []
    for o in [o for o in y.index if o >= start and o.month < 12]:
        yr_end = pd.Timestamp(o.year + 1, 12, 1)
        if yr_end > y.index[-1]:
            break
        raw = forecast(y, o, k, a, f, hmax=12 + 12 - o.month)
        for yr in (o.year, o.year + 1):
            m = raw[raw.index.year == yr]
            done = float(y[(y.index.year == yr) & (y.index <= o)].sum())
            total = float(y[y.index.year == yr].sum()) - done
            for meth in ALLOC_METHODS:
                al = allocate(m, total, meth, y, o)
                for t, v in al.items():
                    rows.append((meth, o, t, (t.year - o.year) * 12 + t.month - o.month, v, y[t]))
    bt = pd.DataFrame(rows, columns=["method", "origin", "target", "h", "fc", "actual"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    return bt


def rg_anchor_backtest(y: pd.Series, rg: pd.Series, chosen: str, origin_months=(6, 7, 8, 9)) -> pd.DataFrame:
    """The shipped setting: rest-of-year anchored to a Riksgälden total. Riksgälden's annual vintages are
    not archived, so the total is the sum of its press-release forecasts for the remaining months (each
    from the latest report before that month, so up to one report later than the origin). Scored against
    outcomes (``ae``) and against Riksgälden's own monthly path (``ae_rg``)."""
    k, a, f = choose_params(chosen)
    rows = []
    for o in y.index:
        if o.month not in origin_months:
            continue
        ms = pd.date_range(o + pd.DateOffset(months=1), pd.Timestamp(o.year, 12, 1), freq="MS")
        if not all(m in rg.index and m in y.index for m in ms):
            continue
        raw = forecast(y, o, k, a, f, hmax=len(ms))
        tot = float(rg.reindex(ms).sum())
        for meth in ALLOC_METHODS:
            al = allocate(raw, tot, meth, y, o)
            for t in ms:
                rows.append((meth, o, t, (t.year - o.year) * 12 + t.month - o.month, al[t], y[t], rg[t]))
    bt = pd.DataFrame(rows, columns=["method", "origin", "target", "h", "fc", "actual", "rg"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    bt["ae_rg"] = (bt["fc"] - bt["rg"]).abs()
    return bt


def select_allocation(abt: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Lowest PRE MAE vs outcomes."""
    b = abt.assign(win=window(abt["target"]))
    b = b[b["win"] != ""]
    tab = b.groupby(["method", "win"])["ae"].mean().unstack()
    return str(tab["PRE"].idxmin()), tab.sort_values("PRE")


def describe(name: str) -> str:
    """Plain-English name of a model variant."""
    if name == "smly":
        return "same month last year"
    k, a, f = choose_params(name)
    shift = {0.0: "no level shift", 0.5: "half level shift", 1.0: "full level shift"}.get(a, f"level shift {a:g}")
    return f"{k}-year same-month {f}, {shift}"
