"""Calendar model of daily TGA operating flows, walk-forward scored.

Each Table II bucket b is forecast as

    ŷ_b(d | origin o) = level_b(o) × shape_b(calendar(d))

level_b(o) is the trailing 250-business-day mean known at the origin; shape_b
is a gradient-boosted map from calendar features to the flow/level ratio,
refit monthly on data before the origin. Calendar features encode what is
known in advance: business-day position in the month, the two weekend
conventions payers use (1st/3rd/15th paid the preceding vs following
business day), Social Security Wednesdays, weekend/holiday gaps for tax
accrual, and business-day distance to each tax deadline. Deadlines come from
`TAX_DEADLINES`, so announced shifts (Jul-2020, May-2021) are honest
forecast-time information.

When growth is concentrated in calendar spikes (Medicare, VA, OPM, HHS, …)
the mean×shape product lifts quiet days with the spike mean. Those buckets
can take the `spike` choice: shape ≥ SPIKE_SHAPE_THR keeps mean × shape;
quiet days pin to the trailing median. (Scaling quiet days by shape/quiet_ref
lost PRE on HHS-other / OPM — the calendar already separates spike days, so
quiet residual shape is noise.) Candidates: 2019→2025 top-decile mean grew
≥ SPIKE_GAP above the bottom-half mean; PRE picks among flat / model / spike.

Interest is not modelled: it is the coupon schedule from `auction_ladder`
scaled by its trailing realised ratio. Customs duties optionally replace the
trailing mean with effective tariff rate × goods imports (`customs_driver`),
imports held as a constant share of nominal GDP; PRE picks among flat /
calendar / customs. SSA optionally splits Benefits (Wed cohorts) from SSI
(1st-of-month): each part gets its own mean×shape with the spike pin; PRE
picks among flat / model / ssa_split.

Scored against two benchmarks on the cumulative operating balance over the
next h business days: a trailing-20-day mean (flat) and same-business-day-
of-month last year rescaled by level growth. Windows: PRE 2022-01→2023-12,
OOS 2024-01→ latest. A horizon is "won" only if the model beats both
benchmarks on both windows.

  PYTHONPATH=. .venv/bin/python us/fiscal/flow_model.py --backtest
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Many small HGB fits: OpenMP oversubscription makes each ~30x slower on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from auction_ladder import fed_holidays

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

LEVEL_WINDOW = 250
TRAIN_LEVEL_LAG = 15  # train on level known ~3 weeks earlier, as at forecast time
HORIZON = 30  # business days forecast from each origin
HORIZONS = (1, 5, 10, 20)
TRAIN_START = pd.Timestamp("2013-10-01")
# Stimulus payments and moved deadlines dominate; drop from fitting, keep for scoring.
TRAIN_EXCLUDE = (pd.Timestamp("2020-03-15"), pd.Timestamp("2021-06-30"))
WINDOWS = {"PRE": ("2022-01-01", "2023-12-31"), "OOS": ("2024-01-01", "2099-12-31")}
WINSOR_Q = 0.995
# Spike-level rule: quiet days (shape < thr) → trailing median; spikes → mean × shape.
SPIKE_SHAPE_THR = 2.0
SPIKE_GAP = 0.15  # min (top-decile growth − bottom-half growth) to candidacy
SPIKE_GROWTH_YEARS = (2019, 2025)

# Individual return deadline, by year, where it moved from Apr-15 (rolled to business day).
RETURN_DEADLINE_OVERRIDE = {2020: "2020-07-15", 2021: "2021-05-17"}
# Corporate / estimated deadlines moved (2020: Apr & Jun → Jul-15).
ESTIMATE_OVERRIDE = {"2020-04-15": "2020-07-15", "2020-06-15": "2020-07-15"}

_FED_HOL = fed_holidays()


def bdays(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, freq="C", holidays=_FED_HOL)


def _roll(d: pd.Timestamp, bd: pd.DatetimeIndex, prec: bool) -> pd.Timestamp:
    i = bd.searchsorted(d)
    if i < len(bd) and bd[i] == d:
        return d
    return bd[i - 1] if prec else bd[min(i, len(bd) - 1)]


def tax_deadlines(years: range) -> dict[str, list[pd.Timestamp]]:
    """Deadline dates (unrolled) by type for the given calendar years."""
    out: dict[str, list[pd.Timestamp]] = {"corp": [], "est": [], "ret": []}
    for y in years:
        for m in (4, 6, 9, 12):
            d = f"{y}-{m:02d}-15"
            out["corp"].append(pd.Timestamp(ESTIMATE_OVERRIDE.get(d, d)))
        for m in (1, 4, 6, 9):
            d = f"{y}-{m:02d}-15"
            out["est"].append(pd.Timestamp(ESTIMATE_OVERRIDE.get(d, d)))
        out["ret"].append(pd.Timestamp(RETURN_DEADLINE_OVERRIDE.get(y, f"{y}-04-15")))
    return out


TAX_DEADLINES = tax_deadlines(range(2005, 2036))

# IRS individual filing-season opening (IRS newsroom). Default: Monday on/after 23 Jan.
FILING_OPEN_OVERRIDE = {2013: "2013-01-30", 2014: "2014-01-31", 2015: "2015-01-20", 2016: "2016-01-19",
                        2021: "2021-02-12"}
PAYROLL_ANCHOR = pd.Timestamp("2013-01-04")  # federal civilian payday: biweekly Friday
OPTIONAL_FEATURES = ("filing_bd", "biweek", "payroll_bd")
FEATURES_ON: set[str] = set()
# Per-bucket extras, kept only where they cut PRE daily MAE by >5% (calendar_features_ab.py):
# payroll_bd on salaries −88% PRE (biweek −83%); on every bucket the phase loses PRE in
# aggregate. filing_bd loses PRE.
BUCKET_EXTRA: dict[str, tuple[str, ...]] = {"wd_salaries": ("payroll_bd",)}
SSA_ROLL = True


def filing_open(year: int) -> pd.Timestamp:
    if year in FILING_OPEN_OVERRIDE:
        return pd.Timestamp(FILING_OPEN_OVERRIDE[year])
    d = pd.Timestamp(year=year, month=1, day=23)
    return d + pd.Timedelta(days=(0 - d.dayofweek) % 7)


def ssa_wednesdays(bd: pd.DatetimeIndex) -> pd.Series:
    """n-th Wednesday of the month (SSA pays 2nd/3rd/4th by birth date), moved to the
    preceding business day when the Wednesday is a holiday (SSA payment calendar rule)."""
    out: dict[pd.Timestamp, int] = {}
    for p in pd.period_range(bd[0], bd[-1], freq="M"):
        weds = pd.date_range(p.start_time, p.end_time, freq="W-WED")
        for n, w in enumerate(weds, start=1):
            out[_roll(w, bd, True)] = n
    return pd.Series(out)


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast-time-known calendar features for business dates."""
    dates = pd.DatetimeIndex(dates)
    bd = bdays(dates.min() - pd.Timedelta(days=60), dates.max() + pd.Timedelta(days=60))
    pos = bd.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("calendar_features expects Fed business days")
    f = pd.DataFrame(index=dates)
    f["month"] = dates.month
    f["dom"] = dates.day
    f["dow"] = dates.dayofweek
    ym = bd.to_period("M")
    s = pd.Series(np.arange(len(bd)), index=bd)
    first = s.groupby(ym).transform("min")
    last = s.groupby(ym).transform("max")
    f["bdom"] = (s - first + 1).to_numpy()[pos]
    f["bdom_rev"] = (last - s + 1).to_numpy()[pos]
    f["gap_prev"] = np.r_[1, np.diff(bd.asi8) // 86_400_000_000_000][pos]
    f["gap_next"] = np.r_[np.diff(bd.asi8) // 86_400_000_000_000, 1][pos]
    if SSA_ROLL:
        f["wed_n"] = ssa_wednesdays(bd).reindex(dates).fillna(0).astype(int).to_numpy()
    else:
        f["wed_n"] = np.where(dates.dayofweek == 2, (dates.day - 1) // 7 + 1, 0)
    opens = sorted({bd.searchsorted(filing_open(y)) for y in range(dates.min().year - 1, dates.max().year + 2)})
    j = np.searchsorted(opens, pos, side="right") - 1
    since = pos - np.asarray(opens)[np.clip(j, 0, None)]
    f["filing_bd"] = np.where((j >= 0) & (since <= 45), since, 99)
    f["biweek"] = ((dates - PAYROLL_ANCHOR).days % 14).to_numpy()
    # Paydays on a holiday are paid the preceding business day.
    paydays = pd.date_range(PAYROLL_ANCHOR, bd[-1], freq="14D")
    pay_pos = np.array(sorted({bd.searchsorted(_roll(d, bd, True)) for d in paydays if d >= bd[0]}))
    k = np.searchsorted(pay_pos, pos, side="right") - 1
    f["payroll_bd"] = np.where(k >= 0, pos - pay_pos[np.clip(k, 0, None)], 99) if pay_pos.size else 99

    # Pay-date flags for anchors 1/3/15 under both conventions.
    months = pd.period_range(dates.min() - pd.offsets.MonthBegin(2), dates.max() + pd.offsets.MonthBegin(2), freq="M")
    for anchor in (1, 3, 15):
        prec = {_roll(pd.Timestamp(year=p.year, month=p.month, day=anchor), bd, True) for p in months}
        foll = {_roll(pd.Timestamp(year=p.year, month=p.month, day=anchor), bd, False) for p in months}
        f[f"pay{anchor}_prec"] = dates.isin(list(prec)).astype(int)
        f[f"pay{anchor}_foll"] = dates.isin(list(foll)).astype(int)

    for kind, lst in TAX_DEADLINES.items():
        rolled = np.array(sorted({bd.searchsorted(_roll(d, bd, False)) for d in lst
                                  if bd[0] <= d <= bd[-1]}))
        if rolled.size == 0:
            f[f"dist_{kind}"] = 99
            continue
        j = np.searchsorted(rolled, pos)
        nxt = rolled[np.clip(j, 0, len(rolled) - 1)] - pos
        prv = rolled[np.clip(j - 1, 0, len(rolled) - 1)] - pos
        dist = np.where(np.abs(prv) < np.abs(nxt), prv, nxt)
        f[f"dist_{kind}"] = np.where(np.abs(dist) <= 6, dist, 99)
    return f


# --------------------------------------------------------------------------------------
# Model


def _level(y: pd.Series, lag: int) -> pd.Series:
    return y.rolling(LEVEL_WINDOW, min_periods=120).mean().shift(lag)


def spike_growth_gap(y: pd.Series, y0: int = SPIKE_GROWTH_YEARS[0],
                     y1: int = SPIKE_GROWTH_YEARS[1]) -> float | None:
    """Top-decile mean growth minus bottom-half mean growth, y0→y1.

    Large positive = the level rise is concentrated on spike days (Medicare-style);
    near zero / negative = the base grew with the mean (SSA, withheld).
    """
    def _means(yr: int) -> tuple[float, float] | None:
        s = y[y.index.year == yr].dropna()
        if len(s) < 100:
            return None
        top = float(s[s >= s.quantile(0.9)].mean())
        bot = float(s[s <= s.quantile(0.5)].mean())
        if top <= 1e-6 or bot <= 1e-6:
            return None
        return top, bot

    a, b = _means(y0), _means(y1)
    if a is None or b is None:
        return None
    return (b[0] / a[0] - 1.0) - (b[1] / a[1] - 1.0)


def is_spike_candidate(y: pd.Series) -> bool:
    gap = spike_growth_gap(y)
    if gap is None:
        return False
    # Also require the top decile itself rose (not just base collapse).
    s0 = y[y.index.year == SPIKE_GROWTH_YEARS[0]].dropna()
    s1 = y[y.index.year == SPIKE_GROWTH_YEARS[1]].dropna()
    if len(s0) < 100 or len(s1) < 100:
        return False
    top0 = float(s0[s0 >= s0.quantile(0.9)].mean())
    top1 = float(s1[s1 >= s1.quantile(0.9)].mean())
    if top0 <= 1e-6 or (top1 / top0 - 1.0) < 0.10:
        return False
    return gap >= SPIKE_GAP


def spike_level_forecast(shape: np.ndarray, mean_lvl: float, med_lvl: float,
                         quiet_ref: float | None = None) -> np.ndarray:
    """Mean×shape on spike days; trailing median on quiet days.

    ``quiet_ref`` is accepted for call-site compatibility and ignored — quiet
    days pin flat to the median (shape residual below the threshold is noise).
    """
    quiet = shape < SPIKE_SHAPE_THR
    out = mean_lvl * shape
    out = np.where(quiet, med_lvl, out)
    return np.maximum(out, 0.0)


def load_train_starts() -> dict[str, pd.Timestamp]:
    path = DERIVED / "dts_train_starts.json"
    if not path.exists():
        return {}
    return {k: pd.Timestamp(v) for k, v in json.loads(path.read_text()).items()}


class BucketModel:
    def __init__(self, name: str, start: pd.Timestamp | None = None):
        self.name = name
        self.start = max(TRAIN_START, start) if start is not None else TRAIN_START
        self.model: HistGradientBoostingRegressor | None = None
        self.cap = np.inf
        self.extra = set(FEATURES_ON) | set(BUCKET_EXTRA.get(name, ()))

    def _x(self, feats: pd.DataFrame) -> np.ndarray:
        cols = [c for c in feats.columns if c not in OPTIONAL_FEATURES or c in self.extra]
        return feats[cols].to_numpy(dtype=float)

    def fit(self, y: pd.Series, feats: pd.DataFrame) -> "BucketModel":
        y = y[y.index >= self.start]
        lvl = _level(y, TRAIN_LEVEL_LAG)
        r = (y / lvl).replace([np.inf, -np.inf], np.nan)
        ok = r.notna()
        ok &= ~((y.index >= TRAIN_EXCLUDE[0]) & (y.index <= TRAIN_EXCLUDE[1]))
        ok &= lvl.abs() > 1e-6
        if ok.sum() < 250:
            self.model = None
            return self
        rr = r[ok]
        self.cap = float(rr.abs().quantile(WINSOR_Q))
        rr = rr.clip(-self.cap, self.cap)
        self.model = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=12,
            l2_regularization=1.0, random_state=0,
        )
        self.model.fit(self._x(feats.loc[rr.index]), rr.to_numpy())
        return self

    def shape(self, feats: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.ones(len(feats))
        return np.clip(self.model.predict(self._x(feats)), -self.cap, self.cap)


def interest_forecast(coupon_bn: pd.Series, actual_mn: pd.Series, origin: pd.Timestamp, dates) -> np.ndarray:
    """Interest = coupon schedule × trailing realised ratio + trailing off-coupon-day mean ($mn)."""
    c = coupon_bn.reindex(actual_mn.index).fillna(0.0) * 1000.0
    hist = actual_mn[actual_mn.index <= origin].tail(LEVEL_WINDOW)
    ch = c.reindex(hist.index)
    on = ch > 50
    ratio = hist[on].sum() / ch[on].sum() if ch[on].sum() > 0 else 1.0
    off = hist[~on].mean() if (~on).any() else 0.0
    fut = coupon_bn.reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy() * 1000.0
    return np.where(fut > 50, fut * ratio, off)


def load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    import dts_flows

    panel = pd.read_parquet(DERIVED / "dts_bucket_daily.parquet")
    coupon = pd.read_parquet(DERIVED / "coupon_schedule.parquet")["coupon_bn"]
    panel = panel[[c for c in dts_flows.OP_BUCKETS if c in panel.columns]]
    return panel, coupon


def load_ssa_parts() -> pd.DataFrame:
    """Daily Benefits / SSI $mn. Rebuilds from the Table II cache if the parquet is missing."""
    path = DERIVED / "dts_ssa_parts.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import dts_flows

    panel, _ = load_inputs()
    parts = dts_flows.ssa_parts_panel(dts_flows.fetch_raw(refresh=False, verbose=False), panel.index)
    DERIVED.mkdir(parents=True, exist_ok=True)
    parts.to_parquet(path)
    return parts


def ssa_split_forecast(parts: pd.DataFrame, origin: pd.Timestamp, dates: pd.DatetimeIndex,
                       starts: dict[str, pd.Timestamp] | None = None) -> np.ndarray:
    """Sum of Benefits and SSI calendar forecasts (each with the spike quiet-day pin)."""
    starts = load_train_starts() if starts is None else starts
    start = starts.get("wd_ssa")
    fdates = calendar_features(dates)
    out = np.zeros(len(dates), dtype=float)
    for col in ("benefits", "ssi"):
        y = parts[col]
        hist = y[y.index <= origin]
        m = BucketModel("wd_ssa", start).fit(hist, calendar_features(hist.index))
        sh = m.shape(fdates)
        mean_lvl = float(hist.tail(LEVEL_WINDOW).mean()) if len(hist) else 0.0
        med_lvl = float(hist.tail(LEVEL_WINDOW).median()) if len(hist) else 0.0
        # Both parts are spike calendars (Wed / 1st); pin quiet days to the part median.
        out += spike_level_forecast(sh, mean_lvl, med_lvl)
    return out


def fit_all(panel: pd.DataFrame, feats: pd.DataFrame, origin: pd.Timestamp) -> dict[str, BucketModel]:
    hist = panel[panel.index <= origin]
    starts = load_train_starts()
    return {
        b: BucketModel(b, starts.get(b)).fit(hist[b], feats.loc[hist.index])
        for b in panel.columns if b != "wd_interest"
    }


CHOICE_PATH = DERIVED / "flow_model_choice.json"
BUCKET_SCORE_PATH = DERIVED / "flow_bucket_score.json"


def load_choice() -> dict[str, str]:
    """Per-bucket 'model', 'flat', 'spike', 'customs', or 'ssa_split' (PRE-selected)."""
    if CHOICE_PATH.exists():
        return json.loads(CHOICE_PATH.read_text()).get("choice", {})
    return {}


def forecast(models: dict[str, BucketModel], panel: pd.DataFrame, coupon: pd.Series,
             origin: pd.Timestamp, dates: pd.DatetimeIndex, choice: dict[str, str] | None = None,
             *, customs_imports: pd.Series | None = None, customs_gdp: pd.Series | None = None,
             ssa_parts: pd.DataFrame | None = None,
             ) -> pd.DataFrame:
    """Bucket forecasts ($mn) for `dates` from information at ``origin``."""
    choice = load_choice() if choice is None else choice
    fdates = calendar_features(dates)
    hist = panel[panel.index <= origin]
    out = pd.DataFrame(index=dates)
    for b in panel.columns:
        if b == "wd_interest":
            out[b] = interest_forecast(coupon, panel["wd_interest"], origin, dates)
            continue
        if b == "dep_customs" and choice.get(b) == "customs":
            from customs_driver import customs_forecast

            shape = None if b not in models else models[b].shape(fdates)
            out[b] = customs_forecast(
                panel["dep_customs"], origin, dates, shape=shape,
                imports_mn=customs_imports, gdp_saar_bn=customs_gdp,
            )
            continue
        if b == "wd_ssa" and choice.get(b) == "ssa_split":
            parts = load_ssa_parts() if ssa_parts is None else ssa_parts
            out[b] = ssa_split_forecast(parts, origin, dates)
            continue
        tail = hist[b].tail(LEVEL_WINDOW)
        mean_lvl = float(tail.mean())
        ch = choice.get(b, "model")
        if ch == "flat":
            out[b] = mean_lvl
            continue
        sh = models[b].shape(fdates)
        if ch == "spike":
            out[b] = spike_level_forecast(sh, mean_lvl, float(tail.median()))
        else:
            out[b] = mean_lvl * sh
    return out


def _net(df: pd.DataFrame) -> pd.Series:
    dep = [c for c in df.columns if c.startswith("dep_")]
    wd = [c for c in df.columns if c.startswith("wd_")]
    return df[dep].sum(axis=1) - df[wd].sum(axis=1)


# --------------------------------------------------------------------------------------
# Benchmarks


def bench_flat(panel: pd.DataFrame, origin: pd.Timestamp, dates) -> pd.Series:
    net = _net(panel[panel.index <= origin])
    return pd.Series(net.tail(20).mean(), index=dates)


def bench_last_year(panel: pd.DataFrame, origin: pd.Timestamp, dates) -> pd.Series:
    """Same month & business-day-of-month a year earlier, per bucket × level growth."""
    hist = panel[panel.index <= origin]
    f_hist = calendar_features(hist.index)
    key = pd.MultiIndex.from_arrays([hist.index.year, f_hist["month"], f_hist["bdom"]])
    lookup = hist.set_axis(key)
    lookup = lookup[~lookup.index.duplicated()]
    fd = calendar_features(pd.DatetimeIndex(dates))
    lvl_now = hist.tail(LEVEL_WINDOW).mean()
    lvl_ly = hist[hist.index <= origin - pd.DateOffset(years=1)].tail(LEVEL_WINDOW).mean()
    g = (lvl_now / lvl_ly.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0)
    rows = []
    for d, m, k in zip(pd.DatetimeIndex(dates), fd["month"], fd["bdom"]):
        idx = (d.year - 1, m, k)
        while idx not in lookup.index and idx[2] > 1:
            idx = (idx[0], idx[1], idx[2] - 1)
        rows.append(lookup.loc[idx] * g if idx in lookup.index else lvl_now)
    return _net(pd.DataFrame(rows, index=dates))


# --------------------------------------------------------------------------------------
# Walk-forward


def backtest(start: str = "2022-01-01", every: str = "W-FRI", verbose: bool = True, save: bool = True) -> pd.DataFrame:
    panel, coupon = load_inputs()
    feats = calendar_features(panel.index)
    actual = _net(panel)
    origins = [d for d in pd.date_range(start, panel.index.max(), freq=every) if d in panel.index]
    origins = [panel.index[panel.index.searchsorted(d, side="right") - 1] for d in
               pd.date_range(start, panel.index.max(), freq=every)]
    origins = sorted(set(origins))
    rows = []
    models: dict[str, BucketModel] = {}
    fitted_month = None
    for i, o in enumerate(origins):
        if fitted_month != o.to_period("M"):
            models = fit_all(panel, feats, o)
            fitted_month = o.to_period("M")
            if verbose:
                print(f"   refit at {o.date()} ({i + 1}/{len(origins)})")
        fut = panel.index[panel.index > o][:HORIZON]
        if len(fut) == 0:
            continue
        fc = forecast(models, panel, coupon, o, fut)
        m = _net(fc)
        b1 = bench_flat(panel, o, fut)
        b2 = bench_last_year(panel, o, fut)
        for h, d in enumerate(fut, start=1):
            rows.append((o, d, h, actual.loc[d], m.loc[d], b1.loc[d], b2.loc[d]))
    bt = pd.DataFrame(rows, columns=["origin", "date", "h", "actual", "model", "flat", "last_year"])
    if save:
        bt.to_parquet(DERIVED / "flow_backtest.parquet", index=False)
    return bt


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE ($bn) of the cumulative net operating flow over the first h days, by window."""
    out = []
    bt = bt.sort_values(["origin", "h"]).copy()
    for col in ("actual", "model", "flat", "last_year"):
        bt[f"cum_{col}"] = bt.groupby("origin")[col].cumsum()
    for w, (a, b) in WINDOWS.items():
        sl = bt[(bt["origin"] >= a) & (bt["origin"] <= b)]
        for h in HORIZONS:
            x = sl[sl["h"] == h]
            if x.empty:
                continue
            rec = {"window": w, "h": h, "n": len(x)}
            for col in ("model", "flat", "last_year"):
                rec[col] = float((x[f"cum_{col}"] - x["cum_actual"]).abs().mean() / 1000.0)
            rec["actual_sd"] = float(x["cum_actual"].std() / 1000.0)
            out.append(rec)
    s = pd.DataFrame(out)
    s["wins"] = (s["model"] < s["flat"]) & (s["model"] < s["last_year"])
    return s


def daily_bucket_score(start: str = "2022-01-01", write: bool = True) -> pd.DataFrame:
    """Per-bucket daily MAE ($bn) by window, model vs flat vs spike [/ssa_split], refit monthly.

    Writes the PRE-window choice to CHOICE_PATH; OOS columns are then an honest
    read of that choice. Spike candidacy is the 2019→2025 top-decile vs bottom-half
    growth gap; PRE picks among flat / model / spike for candidates, else flat /
    model. SSA also competes ``ssa_split`` (Benefits | SSI). Customs is forced to
    rate × imports (PRE has no tariff step).
    """
    from customs_driver import customs_forecast, load_drivers

    panel, coupon = load_inputs()
    feats = calendar_features(panel.index)
    imports, gdp = load_drivers()
    ssa_parts = load_ssa_parts() if "wd_ssa" in panel.columns else None
    months = pd.period_range(start, panel.index.max(), freq="M")
    errs = []
    all_model = {b: "model" for b in panel.columns}
    all_spike = {b: "spike" for b in panel.columns}
    all_ssa = {**all_model, "wd_ssa": "ssa_split"}
    for p in months:
        o = panel.index[panel.index < p.start_time]
        if len(o) == 0:
            continue
        o = o[-1]
        fut = panel.index[(panel.index >= p.start_time) & (panel.index <= p.end_time)]
        if len(fut) == 0:
            continue
        models = fit_all(panel, feats, o)
        fc = forecast(models, panel, coupon, o, fut, choice=all_model)
        fc_sp = forecast(models, panel, coupon, o, fut, choice=all_spike)
        fc_ssa = (
            forecast(models, panel, coupon, o, fut, choice=all_ssa, ssa_parts=ssa_parts)
            if ssa_parts is not None else None
        )
        lvl = panel[panel.index <= o].tail(LEVEL_WINDOW).mean()
        cust = None
        if "dep_customs" in panel.columns:
            shape = models["dep_customs"].shape(calendar_features(fut))
            cust = customs_forecast(
                panel["dep_customs"], o, fut, shape=shape,
                imports_mn=imports, gdp_saar_bn=gdp,
            )
        w = next(k for k, (a, b) in WINDOWS.items() if a <= str(p.start_time.date()) <= b)
        for b in panel.columns:
            y = panel.loc[fut, b]
            c_err = float((pd.Series(cust, index=fut) - y).abs().sum()) if b == "dep_customs" and cust is not None else float("nan")
            s_err = (
                float((fc_ssa[b] - y).abs().sum())
                if b == "wd_ssa" and fc_ssa is not None else float("nan")
            )
            errs.append((w, b, (fc[b] - y).abs().sum(), (lvl[b] - y).abs().sum(),
                         (fc_sp[b] - y).abs().sum(), c_err, s_err, y.abs().sum(), len(fut)))
    e = pd.DataFrame(errs, columns=["window", "bucket", "model", "flat", "spike", "customs",
                                    "ssa_split", "mean_abs", "n"])
    g = e.groupby(["bucket", "window"])[
        ["model", "flat", "spike", "customs", "ssa_split", "mean_abs", "n"]
    ].sum(min_count=1)
    for col in ("model", "flat", "spike", "customs", "ssa_split", "mean_abs"):
        g[col] = g[col] / g["n"] / 1000.0
    t = g.drop(columns="n").unstack("window")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    # A bucket whose composition broke after PRE has no fitted model there (tie): it
    # inherits the architecture's default and its OOS column is its only evidence.
    choice: dict[str, str] = {}
    for b in t.index:
        cands = {"flat": float(t.loc[b, "flat_PRE"]), "model": float(t.loc[b, "model_PRE"])}
        if is_spike_candidate(panel[b]):
            cands["spike"] = float(t.loc[b, "spike_PRE"])
        if b == "wd_ssa" and pd.notna(t.loc[b, "ssa_split_PRE"]):
            cands["ssa_split"] = float(t.loc[b, "ssa_split_PRE"])
        choice[b] = min(cands, key=cands.get)
    # Identities, not calendar fits: coupons for interest, rate × imports for customs.
    # PRE (2022–23) has no tariff step, so a PRE gate would always keep the trailing mean.
    choice["wd_interest"] = "model"
    choice["dep_customs"] = "customs"
    t["choice"] = pd.Series(choice)
    t["spike_candidate"] = pd.Series({b: is_spike_candidate(panel[b]) for b in t.index})
    for w in WINDOWS:
        t[f"skill_{w}"] = 1 - t[f"model_{w}"] / t[f"flat_{w}"]
    if write:
        CHOICE_PATH.write_text(json.dumps({"selected_on": WINDOWS["PRE"], "choice": choice}, indent=1))
        BUCKET_SCORE_PATH.write_text(t.reset_index().to_json(orient="records", indent=1))
    return t.sort_values("flat_OOS", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--buckets", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        bt = backtest()
        s = score(bt)
        print(s.round(2).to_string(index=False))
        (DERIVED / "flow_model_score.json").write_text(json.dumps(s.to_dict(orient="records"), indent=1))
    if args.buckets:
        print(daily_bucket_score().round(2).to_string())


if __name__ == "__main__":
    main()
