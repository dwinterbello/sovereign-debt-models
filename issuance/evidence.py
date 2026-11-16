"""Living evidence table: does each headline metric line up with the next move in price?

Recomputed on every refresh from `headline.panel()` and `market.load()`, so every number the
dashboard quotes about a metric comes from here, never from hand-typed notes.

Method (identical for every metric):
  x    percentile of the metric in its own country's history to date (expanding, min 24 months),
       after multiplying by its pressure sign, so + always means more supply pressure / less
       demand; centred at the median and scaled so 1 unit = 10 percentile points. Chosen over
       rolling z, rolling percentile, level and changes by `experiments/evidence_transforms.py`
       (most forward links); that choice was made after seeing the results.
  y    change in the target from t+lag to t+lag+h months; lag = months until the metric is
       public (`headline.lag_for`: 1 for supply and auction data, 2 for the score and pillars,
       2 for euro-area metrics built on ECB purchase data). Return targets (rx10_h) are already a
       holding-period return, so y is the excess return earned from t+lag to t+lag+h.
  ctl  the target's change over the 12 months to t (for return targets: the 12m excess return
       realised over the 12 months to t), so a metric is credited only with what it adds to the
       move price has already made (DV01 is larger when yields are low, so supply measured in
       DV01 partly encodes the price path).
  Countries are pooled after removing each one's mean; every month is used; standard errors are
  Driscoll–Kraay (overlapping changes and shocks common to all countries). β is bp of target
  per 10 percentile points of pressure, signed so + is the theory sign for that target
  (`TARGET_SIGN`; every target is +1 today: supply pressure → higher yields, steeper curve,
  wider spreads, a higher term premium, and — Greenwood & Vayanos 2014 — higher *future* excess
  returns, because the extra term premium is earned by whoever holds the duration).
  loo_min_t  smallest t with any one country left out (pooled targets only).
  p_chance   share of circular shifts of x against y (same shift for every country, 12 months or
             more, every month) whose t is at least as large: the chance of a t this big from a
             metric with the same persistence and no timing link to price. Needs ≥19 shifts
             (so p can reach 0.05); shorter samples carry no p and the check does not block.
  oos_r2     real-time pooled forecast from 2016: each month's β uses only y realised by then; R²
             against the same regression without x.
  t_bar      the in-sample bar (T_BAR); `chance()` reports how many tests ran and how many
             forward links that many null tests would produce.
  Forward link = β > 0, t ≥ t_bar, loo_min_t ≥ 1, p_chance ≤ 0.10, oos_r2 > 0. Wrong sign mirrors
  the in-sample part. β > 0 with t ≥ 2 that fails the rest is 'in-sample only'.
  r12  pooled correlation of x with the 12m change ending at t (descriptive: did price move with
       the metric, not did the metric lead it). Signed by `CO_SIGN` so + always means pressure
       rose as bonds cheapened: for excess-return targets that is a negative realised return.

Targets: y10 (10y level), rel10 (10y minus the mean of UK/DE/US/JP), s1030 (30y − 10y),
spread_de (FR, IT 10y over Bunds), asw20 (UK 20y gilt − SONIA OIS, zero-coupon), rx10_3 / rx10_12
(10y excess return over cash, 3m / 12m, `market.excess_return`), tp10 (term-premium proxy; US ACM,
UK gilt − OIS, DE/JP 10y − 2y, FR/IT 10y − Bund 2y — see `market`).
Windows: 2010+ sets the verdict; 2010–18 and 2019+ are its two halves, so a link found in one half
can be checked in the other.

Interactions (`interactions()`, Vayanos–Vila 2021; Greenwood, Hanson, Stein & Sunderam 2023):
  y = a_c + β1·x + β2·x·s + β3·s + γ·y12, s = a state's own-country expanding percentile − 0.5
  (so s runs −0.5 … +0.5 and β2 is the change in the supply slope from the lowest to the highest
  state). States: rvol (realised 10y vol, risk-bearing capacity: supply should matter more when
  vol is high) and float (1 − central-bank share of coupon DV01, Eren–Schrimpf–Xia: more
  duration held by private balance sheets, so supply should matter more; the elasticity reading —
  price-elastic private holders flatten demand — predicts the opposite, which shows as 'wrong
  sign'). Same DK / LOO / rotation (x rotated, the state stays with y) / OOS machinery on β2; OOS
  R² is against the linear model with x and s. Split: β on x alone in the top and bottom state
  terciles. 2010+ window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from issuance import market
from issuance.headline import METRICS, holder_share, lag_for, panel
from issuance.paths import CACHE

OUT = CACHE / "evidence.parquet"
OUT_INTER = CACHE / "evidence_interactions.parquet"
PCT_MIN = 24
HORIZONS = (3, 12)
T_BAR = 2.0
CO_MOVE = 0.3
MIN_OBS = 36
POOLED = 3
LOO_T = 1.0
P_CHANCE = 0.10
MIN_SHIFT = 12
SHIFT_STEP = 1
MIN_SHIFTS = 19
OOS_START = pd.Timestamp("2016-01-01")
WINDOWS = {"2010+": ("2010-01-01", None), "2010–18": ("2010-01-01", "2018-12-01"), "2019+": ("2019-01-01", None)}
TARGETS = {
    "y10": ("10y yield", None),
    "rel10": ("10y vs peers", None),
    "s1030": ("10s30s", None),
    "spread_de": ("10y vs Bund", ("FR", "IT")),
    "asw20": ("UK 20y ASW", ("UK",)),
    "rx10_3": ("10y excess return, 3m", None),
    "rx10_12": ("10y excess return, 12m", None),
    "tp10": ("10y term premium", None),
}
# 'return' targets are stored at the start month of their holding period (market.excess_return).
TARGET_KIND = {"rx10_3": "return", "rx10_12": "return"}
TARGET_HORIZONS = {"rx10_3": (3,), "rx10_12": (12,)}
# The 12m excess return realised to t is the control for every return target.
RETURN_CONTROL = "rx10_12"
# +1: the theory says more supply pressure raises the target over the horizon.
TARGET_SIGN = {t: 1 for t in TARGETS}
# Sign of r12 (co-movement with the past 12m) that means 'pressure arrived as bonds cheapened':
# a cheapening is a rise in a yield or spread but a negative realised excess return.
CO_SIGN = {"rx10_3": -1, "rx10_12": -1}
NEW_TARGETS = ("rx10_3", "rx10_12", "tp10")
NEW_METRICS = ("dur_stock_gdp", "dur_stock_gdp_long")
STATES = {
    "rvol": "10y realised vol, 3m",
    "float": "Private share of coupon DV01 (1 − central-bank share)",
}
# Months after t until the state is public (holdings data a month; euro area as its cb_share).
STATE_LAG = {"rvol": lambda c: 0, "float": lambda c: lag_for("cb_share", c)}
TERCILE = 1 / 6
INTER_WINDOW = "2010+"


def _rank_last(a: np.ndarray) -> float:
    return (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean()


def pressure_pct(v: pd.Series) -> pd.Series:
    """Percentile of each month in the country's own history to date (no look-ahead), centred
    and scaled so one unit = 10 percentile points."""
    return (v.expanding(min_periods=PCT_MIN).apply(_rank_last, raw=True) - 0.5) * 10


def pressure_z(p: pd.DataFrame | None = None) -> pd.DataFrame:
    p = panel() if p is None else p
    h = p[p["phase"] == "history"].sort_values(["metric", "country", "month"])
    out = []
    for (m, c), g in h.groupby(["metric", "country"]):
        meta = METRICS.get(m, {})
        v = g.set_index("month")["value"].asfreq("MS") * meta.get("sign", 1)
        z = pressure_pct(v)
        out.append(pd.DataFrame({"metric": m, "country": c, "month": z.index, "x": z.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna() if out else pd.DataFrame(columns=["metric", "country", "month", "x"])


def forward_changes(mkt: pd.DataFrame | None = None, lags=(1, 2)) -> pd.DataFrame:
    mkt = market.load() if mkt is None else mkt
    out = []
    ctl = {c: g.set_index("month")["value"].sort_index().asfreq("MS")
           for c, g in mkt[mkt["series"] == RETURN_CONTROL].groupby("country")}
    for tgt, (_, only) in TARGETS.items():
        t = mkt[mkt["series"] == tgt]
        if only:
            t = t[t["country"].isin(only)]
        sign = TARGET_SIGN.get(tgt, 1)
        ret = TARGET_KIND.get(tgt) == "return"
        for c, g in t.groupby("country"):
            y = g.set_index("month")["value"].sort_index().asfreq("MS")
            if ret:
                back = ctl[c].shift(12).reindex(y.index) if c in ctl else pd.Series(np.nan, index=y.index)
            else:
                back = y - y.shift(12)
            for lag in lags:
                for h in TARGET_HORIZONS.get(tgt, HORIZONS):
                    fwd = y.shift(-lag) if ret else y.shift(-(lag + h)) - y.shift(-lag)
                    out.append(pd.DataFrame({"target": tgt, "country": c, "lag": lag, "h": h, "month": fwd.index,
                                             "y": sign * fwd.to_numpy(), "y12": sign * back.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna(subset=["y", "y12"], how="all") if out else pd.DataFrame()


def _demean(q: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return q.groupby("country")[cols].transform(lambda v: v - v.mean())


def _within(a: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Remove each country's mean from every column."""
    k = int(cc.max()) + 1
    cnt = np.maximum(np.bincount(cc, minlength=k), 1)
    mu = np.stack([np.bincount(cc, weights=a[:, j], minlength=k) for j in range(a.shape[1])], axis=1) / cnt[:, None]
    return a - mu[cc]


def dk_ols(y: np.ndarray, X: np.ndarray, time: np.ndarray, maxlags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Driscoll–Kraay standard errors: Bartlett kernel over the cross-section sums of the
    scores per period, cluster small-sample correction. Same numbers as statsmodels
    `cov_type='hac-groupsum'`; X carries its own constant, `time` = period codes 0..T−1."""
    xtx = np.linalg.pinv(X.T @ X)
    b = xtx @ (X.T @ y)
    xu = X * (y - X @ b)[:, None]
    n_t = int(time.max()) + 1
    g = np.stack([np.bincount(time, weights=xu[:, j], minlength=n_t) for j in range(X.shape[1])], axis=1)
    S = g.T @ g
    for lag in range(1, maxlags + 1):
        s = g[lag:].T @ g[:-lag]
        S += (1 - lag / (maxlags + 1)) * (s + s.T)
    cov = xtx @ S @ xtx.T
    n, k = X.shape
    n_g = len(np.unique(time))
    cov *= n_g / (n_g - 1) * (n - 1) / (n - k)
    return b, b / np.sqrt(np.diag(cov))


def _core(y: np.ndarray, xs: np.ndarray, cc: np.ndarray, month: np.ndarray, maxlags: int):
    """Pooled within-country OLS of y on the columns of xs (first column is the tested one)."""
    if len(y) < MIN_OBS:
        return None
    tt = np.unique(month, return_inverse=True)[1]
    d = _within(np.c_[xs, y], cc)
    if np.std(d[:, 0]) == 0:
        return None
    return dk_ols(d[:, -1], np.c_[np.ones(len(d)), d[:, :-1]], tt, maxlags)


def _arrays(q: pd.DataFrame, cols: list[str]):
    return (q["y"].to_numpy(float), q[cols].to_numpy(float), q["country"].factorize()[0],
            q["month"].to_numpy("datetime64[ns]"))


def _ols(q: pd.DataFrame, h: int, cols: tuple[str, ...] = ("x", "y12")) -> tuple[float, float] | None:
    """Pooled slope on the first of `cols` (default x, controlling for the past 12m move), after
    removing each country's mean."""
    got = _ols_all(q, h, cols)
    return None if got is None else (got[0][0], got[1][0])


def _ols_all(q: pd.DataFrame, h: int, cols: tuple[str, ...]):
    """(β×100, t) for every column in `cols`."""
    if len(q) < MIN_OBS:
        return None
    got = _core(*_arrays(q, list(cols)), h + 1)
    return None if got is None else (got[0][1:] * 100, got[1][1:])


def _rotations(q: pd.DataFrame, h: int, cols: tuple[str, ...], k_of: int = 0, rebuild=None) -> np.ndarray:
    """t of coefficient `k_of` with x rotated against everything else by k months, the same k for
    every country. `rebuild(xs, x_rot)` recomputes columns that contain x (interaction terms)."""
    months = np.sort(q["month"].unique())
    n = len(months)
    if n - 2 * MIN_SHIFT + 1 < MIN_SHIFTS:
        return np.array([])
    y, xs, cc, mo = _arrays(q, list(cols))
    tpos = np.searchsorted(months, q["month"].to_numpy("datetime64[ns]"))
    wide = np.full((n, int(cc.max()) + 1), np.nan)
    wide[tpos, cc] = q["x"].to_numpy(float)
    ts = []
    for k in range(MIN_SHIFT, n - MIN_SHIFT + 1, SHIFT_STEP):
        xr = wide[(tpos - k) % n, cc]
        keep = np.isfinite(xr)
        z = xs[keep].copy()
        z[:, 0] = xr[keep]
        if rebuild is not None:
            z = rebuild(z)
        got = _core(y[keep], z, cc[keep], mo[keep], h + 1)
        if got is not None:
            ts.append(got[1][1 + k_of])
    return np.array(ts)


def _shift_t(q: pd.DataFrame, h: int) -> np.ndarray:
    """t-statistics with x rotated against (y, y12) by k months, the same k for every country."""
    return _rotations(q, h, ("x", "y12"))


def _oos_r2(q: pd.DataFrame, h: int, full: tuple[str, ...] = ("x", "y12"), base: tuple[str, ...] = ("y12",)) -> float:
    """Real-time pooled forecast from OOS_START: each month's β uses only rows whose y was realised
    by then. R² of the `full` regression against the `base` one (positive = the extra columns add
    to the forecast)."""
    cols = list(dict.fromkeys([*full, *base]))
    q = q.dropna(subset=[*cols, "y"]).sort_values("month")
    if q.empty:
        return np.nan
    known = (q["month"] + pd.to_timedelta((q["lag"] + h) * 31, unit="D")).to_numpy("datetime64[ns]")
    month = q["month"].to_numpy("datetime64[ns]")
    cc = q["country"].factorize()[0]
    n_c = int(cc.max()) + 1
    a = np.c_[q[cols].to_numpy(float), q["y"].to_numpy(float)]
    fi = [cols.index(c) for c in full]
    bi = [cols.index(c) for c in base]
    e_x, e_0 = [], []
    need = min(2, len(np.unique(cc)))
    for m in np.unique(month[month >= np.datetime64(OOS_START)]):
        tr = known <= m
        te = month == m
        cnt = np.bincount(cc[tr], minlength=n_c)
        if tr.sum() < MIN_OBS * 2 or (cnt > 0).sum() < need:
            continue
        te = te & (cnt[cc] > 0)
        if not te.any():
            continue
        mu = np.stack([np.bincount(cc[tr], weights=a[tr, j], minlength=n_c) for j in range(a.shape[1])], axis=1)
        mu = mu / np.maximum(cnt, 1)[:, None]
        dtr, dte = a[tr] - mu[cc[tr]], a[te] - mu[cc[te]]
        one, one_t = np.ones((len(dtr), 1)), np.ones((len(dte), 1))
        bx = np.linalg.lstsq(np.c_[one, dtr[:, fi]], dtr[:, -1], rcond=None)[0]
        b0 = np.linalg.lstsq(np.c_[one, dtr[:, bi]], dtr[:, -1], rcond=None)[0]
        e_x += list(dte[:, -1] - np.c_[one_t, dte[:, fi]] @ bx)
        e_0 += list(dte[:, -1] - np.c_[one_t, dte[:, bi]] @ b0)
    if len(e_x) < 24:
        return np.nan
    return float(1 - np.sum(np.square(e_x)) / np.sum(np.square(e_0)))


def _p_chance(ts: np.ndarray, t: float) -> float:
    if not len(ts):
        return np.nan
    return float((1 + ((ts >= t) if t >= 0 else (ts <= t)).sum()) / (1 + len(ts)))


def _fit(sub: pd.DataFrame, h: int) -> dict:
    co = sub.dropna(subset=["x", "y12"])
    r12 = np.nan
    if len(co) > 30:
        dc = _demean(co, ["x", "y12"])
        r12 = float(np.corrcoef(dc["x"], dc["y12"])[0, 1])
    q = sub.dropna(subset=["x", "y", "y12"]).sort_values("month")
    got = _ols(q, h)
    if got is None:
        return {}
    per = {c: float(np.corrcoef(g["x"], g["y"])[0, 1]) for c, g in q.groupby("country") if len(g) > 24 and g["x"].std() > 0}
    loo = [_ols(q[q["country"] != c], h) for c in q["country"].unique()] if q["country"].nunique() >= POOLED else []
    loo = [v for v in loo if v is not None]
    ts = _shift_t(q, h)
    t = got[1]
    return {
        "beta_bp": got[0],
        "t": t,
        "hits": int(sum(v > 0 for v in per.values())),
        "n_countries": int(q["country"].nunique()),
        "n_obs": len(q),
        "first": q["month"].min(),
        "last": q["month"].max(),
        "r12": r12,
        "loo_min_t": min(v[1] for v in loo) if loo else np.nan,
        "loo_max_t": max(v[1] for v in loo) if loo else np.nan,
        "p_chance": _p_chance(ts, t),
        "n_shifts": len(ts),
        "oos_r2": _oos_r2(q, h),
    }


def verdict(beta: float, t: float, loo_min_t: float = np.nan, p_chance: float = np.nan, loo_max_t: float = np.nan,
            oos_r2: float = np.nan, t_bar: float = T_BAR) -> str:
    """forward link: β > 0 (the theory sign), t ≥ t_bar, every leave-one-country-out
    t ≥ 1, circular-shift chance p ≤ 0.10, and a positive real-time out-of-sample R². Wrong sign
    mirrors the in-sample part. Missing checks (one country, short sample) do not block."""
    if not np.isfinite(t):
        return "untested"
    chance_ok = not np.isfinite(p_chance) or p_chance <= P_CHANCE
    oos_ok = not np.isfinite(oos_r2) or oos_r2 > 0
    if beta > 0 and t >= t_bar and (not np.isfinite(loo_min_t) or loo_min_t >= LOO_T) and chance_ok and oos_ok:
        return "forward link"
    if beta < 0 and t <= -t_bar and (not np.isfinite(loo_max_t) or loo_max_t <= -LOO_T) and chance_ok:
        return "wrong sign"
    if beta > 0 and t >= 2:
        return "in-sample only"
    return "no link"


def _pairs(p: pd.DataFrame | None, mkt: pd.DataFrame | None, metrics=None, targets=None):
    """(metric, target, h, frame of x joined to y) for every test, with lags per country."""
    xz = pressure_z(p)
    fw = forward_changes(mkt)
    if xz.empty or fw.empty:
        return
    for m in METRICS if metrics is None else metrics:
        xm = xz[xz["metric"] == m].copy()
        if xm.empty:
            continue
        xm["lag"] = [lag_for(m, c) for c in xm["country"]]
        lags = sorted(set(xm["lag"]))
        for tgt in TARGETS if targets is None else targets:
            for h in TARGET_HORIZONS.get(tgt, HORIZONS):
                df = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"), on=["country", "month", "lag"])
                yield m, tgt, h, lags, df


def _window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return df[(df["month"] >= start) & ((df["month"] <= end) if end else True)]


def build(p: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for m, tgt, h, lags, df in _pairs(p, mkt):
        for wname in WINDOWS:
            sub = _window(df, wname)
            got = _fit(sub, h)
            if not got:
                continue
            got["r12"] *= CO_SIGN.get(tgt, 1)
            rows.append({"metric": m, "label": METRICS[m]["label"], "target": tgt, "target_label": TARGETS[tgt][0],
                         "window": wname, "lag": "/".join(str(v) for v in lags), "horizon": h, **got,
                         "countries": " ".join(sorted(sub["country"].unique()))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["t_bar"] = T_BAR
    out["theory_sign"] = out["target"].map(TARGET_SIGN).fillna(1).astype(int)
    out["verdict"] = [verdict(b, t, lo, pc, hi, oo, tb) for b, t, lo, pc, hi, oo, tb in zip(
        out["beta_bp"], out["t"], out["loo_min_t"], out["p_chance"], out["loo_max_t"], out["oos_r2"], out["t_bar"])]
    out["computed"] = pd.Timestamp.today().normalize()
    return out


# ---------------------------------------------------------------- state interactions
def states(mkt: pd.DataFrame | None = None, holders: pd.DataFrame | None = None) -> pd.DataFrame:
    """country, month, state, value, s: s = expanding own-country percentile − 0.5 (no look-ahead)."""
    mkt = market.load() if mkt is None else mkt
    holders = holder_share() if holders is None else holders
    raw = []
    v = mkt[mkt["series"] == "rvol10"]
    raw.append(pd.DataFrame({"country": v["country"], "month": v["month"], "state": "rvol", "value": v["value"]}))
    if not holders.empty:
        hh = holders[holders["phase"] == "history"]
        raw.append(pd.DataFrame({"country": hh["country"], "month": hh["month"], "state": "float",
                                 "value": 1 - hh["cb_hold_share"]}))
    out = []
    for (st, c), g in pd.concat(raw, ignore_index=True).groupby(["state", "country"]):
        s = g.set_index("month")["value"].sort_index().asfreq("MS")
        pct = pressure_pct(s) / 10
        out.append(pd.DataFrame({"country": c, "month": s.index, "state": st, "value": s.to_numpy(), "s": pct.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna(subset=["s"]) if out else pd.DataFrame(
        columns=["country", "month", "state", "value", "s"])


def _with_state(df: pd.DataFrame, st: pd.DataFrame, state: str) -> pd.DataFrame:
    """Join the state public by the time x is (shifted by its lag beyond the metric's)."""
    s = st[st["state"] == state][["country", "month", "s"]]
    if df.empty or s.empty:
        return pd.DataFrame()
    parts = []
    for (c, lag), g in df.groupby(["country", "lag"]):
        k = max(0, STATE_LAG[state](c) - int(lag))
        sc = s[s["country"] == c].set_index("month")["s"].asfreq("MS").shift(k)
        parts.append(g.assign(s=g["month"].map(sc)))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["x", "y", "y12", "s"])
    return out.assign(xs=out["x"] * out["s"]).sort_values("month")


def _rebuild_xs(z: np.ndarray) -> np.ndarray:
    z[:, 1] = z[:, 0] * z[:, 2]
    return z


INTER_COLS = ("x", "xs", "s", "y12")


def fit_interaction(q: pd.DataFrame, h: int) -> dict:
    """β1 (x at the median state), β2 (x × s), β3 (s) with DK t, LOO and rotation on β2, OOS R²
    against the linear model, and β on x in the top / bottom state terciles."""
    got = _ols_all(q, h, INTER_COLS)
    if got is None:
        return {}
    (b1, b2, b3, _), (t1, t2, t3, _) = got
    loo = []
    if q["country"].nunique() >= POOLED:
        for c in q["country"].unique():
            g = _ols_all(q[q["country"] != c], h, INTER_COLS)
            if g is not None:
                loo.append(g[1][1])
    ts = _rotations(q, h, INTER_COLS, k_of=1, rebuild=_rebuild_xs)
    hi, lo = q[q["s"] >= TERCILE], q[q["s"] <= -TERCILE]
    bh, bl = _ols(hi, h), _ols(lo, h)
    return {
        "beta1_bp": b1, "t1": t1, "beta2_bp": b2, "t2": t2, "beta3_bp": b3, "t3": t3,
        "loo_min_t": min(loo) if loo else np.nan, "loo_max_t": max(loo) if loo else np.nan,
        "p_chance": _p_chance(ts, t2), "n_shifts": len(ts),
        "oos_r2": _oos_r2(q, h, INTER_COLS, ("x", "s", "y12")),
        "beta_hi_bp": np.nan if bh is None else bh[0], "t_hi": np.nan if bh is None else bh[1], "n_hi": len(hi),
        "beta_lo_bp": np.nan if bl is None else bl[0], "t_lo": np.nan if bl is None else bl[1], "n_lo": len(lo),
        "n_obs": len(q), "n_countries": int(q["country"].nunique()), "first": q["month"].min(), "last": q["month"].max(),
        "countries": " ".join(sorted(q["country"].unique())),
    }


def interactions(p: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None, st: pd.DataFrame | None = None,
                 metrics=None, targets=None, state_names=None) -> pd.DataFrame:
    """Supply × state for every metric × target × horizon × state over INTER_WINDOW."""
    mkt = market.load() if mkt is None else mkt
    st = states(mkt) if st is None else st
    rows = []
    for m, tgt, h, lags, df in _pairs(p, mkt, metrics, targets):
        sub = _window(df, INTER_WINDOW)
        for state in STATES if state_names is None else state_names:
            q = _with_state(sub, st, state)
            if q.empty:
                continue
            got = fit_interaction(q, h)
            if not got:
                continue
            rows.append({"metric": m, "label": METRICS[m]["label"], "target": tgt, "target_label": TARGETS[tgt][0],
                         "horizon": h, "state": state, "state_label": STATES[state], "window": INTER_WINDOW,
                         "lag": "/".join(str(v) for v in lags), **got})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["t_bar"] = T_BAR
    out["verdict"] = [verdict(b, t, lo, pc, hi, oo, T_BAR) for b, t, lo, pc, hi, oo in zip(
        out["beta2_bp"], out["t2"], out["loo_min_t"], out["p_chance"], out["loo_max_t"], out["oos_r2"])]
    out["computed"] = pd.Timestamp.today().normalize()
    return out


def chance(rows: pd.DataFrame, window: str | None = "2010+") -> dict:
    """How many tests `rows` holds and how many forward links that many null tests would give by
    the in-sample bar alone: n × P(t ≥ T_BAR) under a normal null (an upper bound — LOO, rotation
    and OOS only remove passes; tests on overlapping targets are not independent, so it is a
    count, not a distribution). `bonferroni_t` = one-sided 5% family-wise bar for n tests."""
    r = rows if window is None or "window" not in rows else rows[rows["window"] == window]
    n = len(r)
    p1 = float(1 - norm.cdf(T_BAR))
    return {"n_tests": n, "n_forward_link": int((r["verdict"] == "forward link").sum()) if n else 0,
            "n_wrong_sign": int((r["verdict"] == "wrong sign").sum()) if n else 0,
            "p_pass_null": p1, "expected_by_chance": n * p1,
            "bonferroni_t": float(norm.ppf(1 - 0.05 / n)) if n else np.nan}


def new_rows(ev: pd.DataFrame) -> pd.DataFrame:
    """Rows added by the literature upgrade: a new target or a new metric."""
    if ev.empty:
        return ev
    return ev[ev["target"].isin(NEW_TARGETS) | ev["metric"].isin(NEW_METRICS)]


def summary(ev: pd.DataFrame, window: str = "2010+", targets=None) -> pd.DataFrame:
    """One verdict per metric over `window`, read only off targets that pool three or more
    countries, so one or two markets cannot carry it: 'leads price' if one of them has a forward
    link, 'wrong sign' if one is significantly wrong-signed, 'moves with price' if the same-window
    12m correlation reaches CO_MOVE on one, else 'no link'. A metric that exists for fewer
    countries (UK tail, positioning) is judged on its own targets and `scope` names them.
    `targets` restricts the targets read (default: all)."""
    if ev.empty:
        return pd.DataFrame()
    e = ev[ev["window"] == window]
    if targets is not None:
        e = e[e["target"].isin(list(targets))]
    rows = []
    for m, g in e.groupby("metric", sort=False):
        wide = g[g["n_countries"] >= POOLED]
        use = wide if len(wide) else g
        co = use.loc[use["r12"].idxmax()] if use["r12"].notna().any() else None
        n_link = int((use["verdict"] == "forward link").sum())
        n_wrong = int((use["verdict"] == "wrong sign").sum())
        best = use.loc[use["t"].idxmin()] if n_wrong and not n_link else use.loc[use["t"].idxmax()]
        if n_link:
            overall = "leads price"
        elif n_wrong:
            overall = "wrong sign"
        elif co is not None and co["r12"] >= CO_MOVE:
            overall = "moves with price"
        else:
            overall = "no link"
        scope = "pooled" if len(wide) else " ".join(sorted(set(" ".join(g["countries"]).split()))) + " only"
        rows.append({"metric": m, "overall": overall, "scope": scope, "best_target": best["target"],
                     "best_beta_bp": best["beta_bp"], "best_t": best["t"], "n_link": n_link, "n_targets": len(use),
                     "co_target": None if co is None else co["target"], "co_r12": np.nan if co is None else co["r12"]})
    return pd.DataFrame(rows)


def write() -> pd.DataFrame:
    p, mkt = panel(), market.load()
    ev = build(p, mkt)
    if not ev.empty:
        ev.to_parquet(OUT, index=False)
    inter = interactions(p, mkt)
    if not inter.empty:
        inter.to_parquet(OUT_INTER, index=False)
    return ev


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()


def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(OUT_INTER) if OUT_INTER.exists() else pd.DataFrame()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    ev = write()
    cols = ["metric", "target", "window", "lag", "beta_bp", "t", "loo_min_t", "p_chance", "hits", "n_countries", "verdict"]
    print(ev[cols].round(2).to_string(index=False))
    for w in WINDOWS:
        print(w)
        print(summary(ev, w).round(2).to_string(index=False))
    inter = load_interactions()
    print("tests, 2010+:", chance(ev), "\nnew tests, 2010+:", chance(new_rows(ev)), "\ninteractions:", chance(inter))
