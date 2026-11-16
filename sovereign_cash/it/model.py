"""Fabbisogno forecasts, walk-forward scoring, and the forward Treasury-liquidity path.

Forecast of the monthly state-sector cash requirement x (EUR mn, fabbisogno > 0), origin o, h = 1..12:

  zero          0                                             (other-financing drift benchmark)
  smly          x[o+h−12]                                     same month last year
  flat          mean(x[o−11..o])                              trailing 12-month mean
  add_K         L_o + mean_k (x[o+h−12k] − L_{o−12k})         level-adjusted same month, K years
  mul_K         L_o × mean_k (x[o+h−12k] / L_{o−12k})         multiplicative month-of-year profile

L_t is the trailing 12-month mean ending at t. COVID screen, identical for every model: months in
EXCL (Mar-2020 → Feb-2021) never enter a forecast or a score. add_K / mul_K skip any older year whose
month or level window touches EXCL (up to two older years replace it); a backtest cell (origin, h)
is scored only if neither its own inputs x[o−11..o] (the origin level, and every smly input) nor its
targets x[o+1..o+h] touch EXCL. Scoring is on the cumulative requirement Σ x[o+1..o+h] at
h = 1, 3, 6, 12 from monthly origins: PRE = origins 2015–2021, OOS = origins 2022→. The model is
chosen on PRE mean MAE across horizons only; OOS is reported, never used to choose.

Forward path: L_{t} = L_{t−1} − x̂_t + net bonds_t + net bills_t + other_t. The BOT stock is held at
its current level plus its average calendar-month change (trailing five years, demeaned), so liquidity
absorbs the financing surplus or gap; BOT is raised above that only to keep liquidity at the floor (P10
of the trailing 36 month-ends). Liquidity above the cap (P90) is reported as a surplus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)
PRE = ("2015-01-01", "2021-12-01")
OOS_START = "2022-01-01"
EXCL = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-02-01"))
BAND_MONTHS = 36
BAND_Q = (0.10, 0.90)


def label(m: str) -> str:
    """Plain-English model name for page text."""
    if m in ("smly", "flat", "zero"):
        return {"smly": "last-year repeat", "flat": "trailing 12m mean", "zero": "zero"}[m]
    if m == "sx_level_exog":
        return "component seasonal SARIMAX, lines as shares of the NGDP / HICP level"
    if m.startswith("annual_comp_unit"):
        return "component annual (GDP/CPI pass-through) × seasonal shape"
    if m.startswith("annual_comp_"):
        return "component annual (floored slopes) × seasonal shape"
    if m.startswith("annual_"):
        return "annual level × seasonal shape"
    kind, k = m.split("_", 1)
    return f"level-adjusted seasonal, {k}y" if kind == "add" else f"multiplicative seasonal, {k}y"


def _level(x: pd.Series, t: int) -> float:
    w = x.iloc[t - 11:t + 1]
    return float(w.mean()) if t >= 11 and w.notna().all() else np.nan


def in_excl(d: pd.Timestamp) -> bool:
    return EXCL[0] <= d <= EXCL[1]


def _touches_excl(x: pd.Series, *idx: int) -> bool:
    return any(0 <= i < len(x) and in_excl(x.index[i]) for i in idx)


def forecast(x: pd.Series, o: int, model: str, horizon: int = 12) -> np.ndarray:
    """Monthly forecasts x̂[o+1..o+horizon] from data through position `o` (inclusive)."""
    out = np.full(horizon, np.nan)
    if model == "zero":
        return np.zeros(horizon)
    lo = _level(x, o)
    if model == "flat":
        out[:] = lo
        return out
    if model == "smly":
        for h in range(1, horizon + 1):
            j = o + h - 12
            while j > o:
                j -= 12
            out[h - 1] = float(x.iloc[j]) if 0 <= j <= o and pd.notna(x.iloc[j]) else np.nan
        return out
    kind, rest = model.split("_", 1)
    k_years = int(rest)
    n = len(x)
    for h in range(1, horizon + 1):
        vals = []
        for k in range(1, k_years + 3):
            if len(vals) == k_years:
                break
            j, lo_k = o + h - 12 * k, o - 12 * k
            if lo_k - 11 < 0:
                break
            if not (0 <= j < n and 0 <= lo_k < n):
                continue
            if _touches_excl(x, j, *range(lo_k - 11, lo_k + 1)):
                continue
            lk = _level(x, lo_k)
            if not (np.isfinite(lk) and pd.notna(x.iloc[j])):
                continue
            if kind == "add":
                vals.append(x.iloc[j] - lk)
            elif abs(lk) > 1e-6:
                vals.append(x.iloc[j] / lk)
        if vals and np.isfinite(lo):
            out[h - 1] = lo + np.mean(vals) if kind == "add" else lo * np.mean(vals)
    return out


CANDIDATES = ("smly", "flat", "add_1", "add_2", "add_3", "add_5", "mul_3", "mul_5")


def backtest(x: pd.Series, models: tuple[str, ...] = CANDIDATES, start: str = "2013-01-01") -> pd.DataFrame:
    """Long table: origin, h, model, forecast and actual cumulative requirement (EUR mn).

    ``touches_excl`` flags cells whose inputs x[o−11..o] or targets x[o+1..o+h] touch EXCL.
    """
    x = x.asfreq("MS")
    rows = []
    for o in range(len(x)):
        if x.index[o] < pd.Timestamp(start):
            continue
        fcs = {m: forecast(x, o, m) for m in models}
        for h in HORIZONS:
            if o + h >= len(x):
                continue
            act = x.iloc[o + 1:o + h + 1]
            if act.isna().any():
                continue
            touches = _touches_excl(x, *range(o - 11, o + h + 1))
            for m, f in fcs.items():
                rows.append({"origin": x.index[o], "h": h, "model": m, "fc": float(np.sum(f[:h])),
                             "actual": float(act.sum()), "touches_excl": touches})
    return pd.DataFrame(rows)


def window_of(bt: pd.DataFrame) -> pd.Series:
    clean = ~bt["touches_excl"]
    pre = (bt["origin"] >= PRE[0]) & (bt["origin"] <= PRE[1]) & clean
    oos = (bt["origin"] >= OOS_START) & clean
    return pd.Series(np.where(pre, "PRE", np.where(oos, "OOS", "")), index=bt.index)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE (EUR bn) by window × h × model, with n."""
    b = bt.assign(window=window_of(bt), ae=(bt["fc"] - bt["actual"]).abs() / 1000.0)
    b = b[b["window"] != ""].dropna(subset=["fc"])
    return b.groupby(["window", "h", "model"]).agg(mae_bn=("ae", "mean"), n=("ae", "size")).reset_index()


def choose(sc: pd.DataFrame) -> str:
    """Lowest PRE mean MAE across horizons (models that forecast every PRE origin only)."""
    pre = sc[sc["window"] == "PRE"]
    full = pre.groupby("h")["n"].transform("max")
    pre = pre[pre["n"] == full]
    tab = pre.pivot(index="model", columns="h", values="mae_bn").dropna()
    return str(tab.mean(axis=1).idxmin())


def anchor_to_plan(fc: pd.Series, x: pd.Series, origin: pd.Timestamp, plans: dict[int, float]) -> pd.Series:
    """Shift the monthly forecast so each calendar year sums to the plan total (EUR, same units as x)."""
    from sovereign_cash.core.cash_framework import anchor_to_plan as _anchor
    return _anchor(fc, x, origin, plans)


def account_band(liq_bn: pd.Series, asof: pd.Timestamp) -> tuple[float, float]:
    w = liq_bn.loc[:asof].tail(BAND_MONTHS)
    return float(w.quantile(BAND_Q[0])), float(w.quantile(BAND_Q[1]))


def bot_seasonal_net(stock_bn: pd.Series, asof: pd.Timestamp, years: int = 5) -> pd.Series:
    """Average month-on-month BOT stock change by calendar month over the trailing ``years``, demeaned
    so twelve months net to zero (EUR bn, index 1..12)."""
    d = stock_bn.loc[:asof].dropna().diff().dropna()
    d = d[d.index > asof - pd.DateOffset(years=years)]
    m = d.groupby(d.index.month).mean().reindex(range(1, 13)).fillna(0.0)
    return m - m.mean()


def hold_bills(level0: float, flows: np.ndarray, floor: float, existing_red: np.ndarray,
               mix: dict[int, float], target_net: np.ndarray, known_gross: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BOT book rolled forward with the stock held on ``target_net``: returns (net, gross, redemptions).

    Redemptions in month i = existing BOT maturing + new bills issued at j < i with tenor i − j (tenor
    shares ``mix``). Gross = redemptions + target net, raised further only when liquidity would fall
    below ``floor``; never negative. ``known_gross`` (NaN = free) pins months whose auctions settled.
    Liquidity absorbs everything else, so a financing surplus shows up as liquidity above the band.
    """
    n = len(flows)
    pinned = np.zeros(n, dtype=bool) if known_gross is None else np.isfinite(known_gross)
    red = np.asarray(existing_red, dtype=float).copy()
    gross, net = np.zeros(n), np.zeros(n)
    lvl = level0
    for i, f in enumerate(flows):
        nxt = lvl + f
        if pinned[i]:
            gross[i] = known_gross[i]
        else:
            want = float(target_net[i]) + max(floor - (nxt + float(target_net[i])), 0.0)
            gross[i] = max(red[i] + want, 0.0)
        net[i] = gross[i] - red[i]
        for ten, share in mix.items():
            if i + int(ten) < n:
                red[i + int(ten)] += gross[i] * float(share)
        lvl = nxt + net[i]
    return net, gross, red
