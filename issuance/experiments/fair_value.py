"""Supply fair value: does the gap between price and what supply implies close?

Two steps, both real time (month t uses only data public at t; metric lagged by its publication lag):
  1. fair value   price_t = a + b·x_t, fitted on the trailing FV_WIN months up to t
                  (country level: own history; panel: countries pooled with country means removed)
  2. reversion    price change over the next h months on residual_t, β fitted on pairs already realised

Compared against two benchmarks with the same machinery:
  rw   no change
  own  the price's own gap to its trailing FV_WIN-month mean (plain mean reversion, no supply)
A supply residual only earns its place if it beats `own`: every level series mean-reverts.

x as level, z (trailing 60m) or percentile (trailing 60m rank); metrics absorb, absorb_long,
cb_share (sign: more pressure = higher x), and absorb + cb_share jointly.

Run: PYTHONPATH=. .venv/bin/python -m issuance.experiments.fair_value
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance import market
from issuance.headline import METRICS, lag_for, panel

TARGETS = {"rel10": None, "s1030": None, "spread_de": ("FR", "IT"), "asw20": ("UK",)}
SETS = {"absorb": ["absorb"], "absorb_long": ["absorb_long"], "cb_share": ["cb_share"],
        "absorb+cb": ["absorb", "cb_share"]}
TRANSFORMS = ("level", "z", "pct")
HORIZONS = (3, 6, 12)
FV_WIN = 120
Z_WIN = 60
OOS_START = pd.Timestamp("2016-01-01")
MIN_FIT = 36


def _transform(v: pd.Series, how: str) -> pd.Series:
    if how == "level":
        return v
    if how == "z":
        r = v.rolling(Z_WIN, min_periods=24)
        return (v - r.mean()) / r.std()
    return v.rolling(Z_WIN, min_periods=24).apply(lambda a: (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean(), raw=True)


def _x_panel(how: str) -> pd.DataFrame:
    """country, month, one column per metric: transformed, signed, lagged to when it is public."""
    p = panel()
    p = p[p["phase"] == "history"]
    out = []
    for (m, c), g in p[p["metric"].isin({"absorb", "absorb_long", "cb_share"})].groupby(["metric", "country"]):
        v = g.set_index("month")["value"].sort_index().asfreq("MS") * METRICS[m].get("sign", 1)
        v = _transform(v, how).shift(lag_for(m, c))
        out.append(pd.DataFrame({"country": c, "month": v.index, "metric": m, "x": v.to_numpy()}))
    x = pd.concat(out).pivot_table(index=["country", "month"], columns="metric", values="x").reset_index()
    return x


def _y_panel() -> pd.DataFrame:
    mk = market.load()
    parts = []
    for t, only in TARGETS.items():
        s = mk[mk["series"] == t]
        if only:
            s = s[s["country"].isin(only)]
        parts.append(s.assign(target=t)[["country", "month", "target", "value"]])
    return pd.concat(parts)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)[0]


def _country_signals(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Real-time supply residual and own-mean gap for one country."""
    g = g.sort_values("month").reset_index(drop=True)
    res, own = np.full(len(g), np.nan), np.full(len(g), np.nan)
    for i in range(len(g)):
        w = g.iloc[max(0, i - FV_WIN + 1): i + 1]
        own[i] = g["y"].iat[i] - w["y"].mean() if len(w) >= MIN_FIT else np.nan
        w = w.dropna(subset=cols)
        if len(w) < MIN_FIT or g.loc[i, cols].isna().any():
            continue
        b = _ols(w[cols].to_numpy(), w["y"].to_numpy())
        res[i] = g["y"].iat[i] - (b[0] + g.loc[i, cols].to_numpy(float) @ b[1:])
    return g.assign(res=res, own=own)


def _panel_signals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pooled fair value: common slope, country intercepts, trailing FV_WIN months."""
    df = df.sort_values(["month", "country"]).reset_index(drop=True)
    months = np.sort(df["month"].unique())
    res = pd.Series(np.nan, index=df.index)
    for t in months:
        w = df[(df["month"] <= t) & (df["month"] > t - pd.DateOffset(months=FV_WIN))].dropna(subset=cols + ["y"])
        if len(w) < MIN_FIT * 2 or w["country"].nunique() < 2:
            continue
        mu = w.groupby("country")[cols + ["y"]].transform("mean")
        b = _ols((w[cols] - mu[cols]).to_numpy(), (w["y"] - mu["y"]).to_numpy())[1:]
        now = df[(df["month"] == t)].dropna(subset=cols + ["y"])
        means = w.groupby("country")[cols + ["y"]].mean()
        now = now[now["country"].isin(means.index)]
        if now.empty:
            continue
        m = means.loc[now["country"]]
        fv = m["y"].to_numpy() + (now[cols].to_numpy() - m[cols].to_numpy()) @ b
        res.loc[now.index] = now["y"].to_numpy() - fv
    return df.assign(res=res.to_numpy())


def _forward(df: pd.DataFrame, h: int) -> pd.Series:
    return df.groupby("country")["y"].transform(lambda s: s.shift(-h) - s)


def _score(df: pd.DataFrame, sig: str, h: int) -> dict:
    """In-sample pooled t (country means removed, Driscoll–Kraay) and real-time OOS R² vs rw and vs own."""
    d = df.assign(f=_forward(df, h)).dropna(subset=[sig, "f", "own"])
    d = d[d["month"] >= "2010-01-01"]
    if len(d) < MIN_FIT or d["month"].nunique() < MIN_FIT:
        return {}
    dm = d[[sig, "f"]] - d.groupby("country")[[sig, "f"]].transform("mean")
    fit = sm.OLS(dm["f"].to_numpy(), sm.add_constant(dm[[sig]].to_numpy())).fit(
        cov_type="hac-groupsum", cov_kwds={"time": d["month"].factorize(sort=True)[0], "maxlags": h + 1})
    e_s, e_o, e_rw = [], [], []
    for t in np.sort(d.loc[d["month"] >= OOS_START, "month"].unique()):
        tr = d[d["month"] <= t - pd.DateOffset(months=h)]
        te = d[d["month"] == t]
        if len(tr) < MIN_FIT:
            continue
        bs, bo = _ols(tr[[sig]].to_numpy(), tr["f"].to_numpy()), _ols(tr[["own"]].to_numpy(), tr["f"].to_numpy())
        e_s += list(te["f"] - (bs[0] + bs[1] * te[sig]))
        e_o += list(te["f"] - (bo[0] + bo[1] * te["own"]))
        e_rw += list(te["f"])
    if len(e_s) < 24:
        return {}
    sse = lambda e: float(np.sum(np.square(e)))
    return {"beta_bp": float(fit.params[1]) * 100 if abs(d["f"]).max() < 5 else float(fit.params[1]),
            "t": float(fit.tvalues[1]), "oos_vs_rw": 1 - sse(e_s) / sse(e_rw), "oos_vs_own": 1 - sse(e_s) / sse(e_o),
            "own_vs_rw": 1 - sse(e_o) / sse(e_rw), "n": len(d), "countries": d["country"].nunique()}


def run() -> pd.DataFrame:
    y = _y_panel()
    rows = []
    for how in TRANSFORMS:
        x = _x_panel(how)
        for tgt in TARGETS:
            yt = y[y["target"] == tgt][["country", "month", "value"]].rename(columns={"value": "y"})
            base = yt.merge(x, on=["country", "month"], how="left")
            for sname, cols in SETS.items():
                if not set(cols) <= set(base.columns):
                    continue
                per = [_country_signals(g[["country", "month", "y", *cols]], cols) for _, g in base.groupby("country")]
                cdf = pd.concat(per, ignore_index=True)
                specs = [("country", cdf)]
                if cdf["country"].nunique() >= 3:
                    pdf = _panel_signals(base[["country", "month", "y", *cols]], cols).merge(
                        cdf[["country", "month", "own"]], on=["country", "month"])
                    specs.append(("panel", pdf))
                for fv, df in specs:
                    for h in HORIZONS:
                        got = _score(df, "res", h)
                        if got:
                            rows.append({"target": tgt, "x": sname, "transform": how, "fair_value": fv, "h": h, **got})
    return pd.DataFrame(rows)


def by_country(tgt: str, cols: list[str], how: str, h: int) -> pd.DataFrame:
    """Same test one country at a time (country-level fair value)."""
    y = _y_panel()
    x = _x_panel(how)
    base = y[y["target"] == tgt][["country", "month", "value"]].rename(columns={"value": "y"}).merge(
        x, on=["country", "month"], how="left")
    out = []
    for c, g in base.groupby("country"):
        s = _country_signals(g[["country", "month", "y", *cols]], cols)
        got = _score(s, "res", h)
        if got:
            out.append({"country": c, **got})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    out = run()
    out.to_csv("/tmp/fair_value.csv", index=False)
    print(out.round(3).to_string(index=False))
