"""Stress test of the evidence table: does any metric survive stricter specifications?

For each metric × pooled target (10y, 10y vs peers, 10s30s), 2010+:
  base     the production regression (country-demeaned, 12m past-change control, Driscoll–Kraay)
  tfe      + month fixed effects: only cross-country differences count, common moves are removed
  dx       x replaced by its 3-month change (differenced metric)
  ctl      + past 1m and 3m change of the target (short momentum / reversal)
  oos_r2   expanding-window pooled forecast from 2016 (β re-estimated each month on data public by
           then), out-of-sample R² against the same model without x
  boot     share of 39 metric × target tests that would clear |t| ≥ 2 by chance (Bonferroni bar)

Run: PYTHONPATH=. .venv/bin/python -m issuance.experiments.evidence_audit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance import evidence as E, market
from issuance.headline import METRICS, lag_for

POOLED_TARGETS = ("y10", "rel10", "s1030")
OOS_START = pd.Timestamp("2016-01-01")
HORIZON = 3


def _frame(metric: str, target: str) -> pd.DataFrame:
    xz = E.pressure_z()
    xm = xz[xz["metric"] == metric].copy()
    if xm.empty:
        return pd.DataFrame()
    xm["lag"] = [lag_for(metric, c) for c in xm["country"]]
    mkt = market.load()
    t = mkt[mkt["series"] == target]
    rows = []
    for c, g in t.groupby("country"):
        y = g.set_index("month")["value"].sort_index().asfreq("MS")
        for lag in sorted(set(xm["lag"])):
            rows.append(pd.DataFrame({
                "country": c, "lag": lag, "month": y.index,
                "y": (y.shift(-(lag + HORIZON)) - y.shift(-lag)).to_numpy(),
                "y12": (y - y.shift(12)).to_numpy(),
                "y3": (y - y.shift(3)).to_numpy(),
                "y1": (y - y.shift(1)).to_numpy(),
                "pub": lag,
            }))
    fw = pd.concat(rows, ignore_index=True)
    df = xm.merge(fw, on=["country", "month", "lag"]).sort_values(["country", "month"])
    df["dx"] = df.groupby("country")["x"].diff(3)
    return df[df["month"] >= "2010-01-01"]


def _fit(q: pd.DataFrame, xcol: str, ctl: list[str], tfe: bool = False) -> tuple[float, float, int]:
    q = q.dropna(subset=[xcol, "y", *ctl])
    if len(q) < E.MIN_OBS or q["country"].nunique() < 3:
        return np.nan, np.nan, len(q)
    cols = [xcol, "y", *ctl]
    d = q[cols] - q.groupby("country")[cols].transform("mean")
    if tfe:
        d = d - d.groupby(q["month"]).transform("mean")
    X = sm.add_constant(d[[xcol, *ctl]].to_numpy())
    fit = sm.OLS(d["y"].to_numpy(), X).fit(
        cov_type="hac-groupsum", cov_kwds={"time": q["month"].factorize(sort=True)[0], "maxlags": HORIZON + 1})
    return float(fit.params[1]) * 100, float(fit.tvalues[1]), len(q)


def _oos_r2(q: pd.DataFrame) -> float:
    """Pooled expanding forecast; a month's regression uses only rows whose y was realised and
    whose x was public before the forecast month."""
    q = q.dropna(subset=["x", "y", "y12"]).sort_values("month")
    months = sorted(m for m in q["month"].unique() if m >= OOS_START)
    e_full, e_base = [], []
    for m in months:
        # y for origin s is known at s + pub + HORIZON months
        train = q[q["month"] + pd.to_timedelta((q["pub"] + HORIZON) * 31, unit="D") <= m]
        test = q[q["month"] == m]
        if len(train) < 60 or test.empty:
            continue
        mu = train.groupby("country")[["x", "y", "y12"]].mean()
        tr = train[["x", "y", "y12"]] - mu.loc[train["country"]].to_numpy()
        te = test[["x", "y", "y12"]] - mu.reindex(test["country"]).to_numpy()
        bf = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[["x", "y12"]]], tr["y"], rcond=None)[0]
        bb = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[["y12"]]], tr["y"], rcond=None)[0]
        pf = np.c_[np.ones(len(te)), te[["x", "y12"]]] @ bf
        pb = np.c_[np.ones(len(te)), te[["y12"]]] @ bb
        e_full += list(te["y"] - pf)
        e_base += list(te["y"] - pb)
    if not e_full:
        return np.nan
    return 1 - np.sum(np.square(e_full)) / np.sum(np.square(e_base))


def run() -> pd.DataFrame:
    rows = []
    for m in METRICS:
        for tgt in POOLED_TARGETS:
            q = _frame(m, tgt)
            if q.empty or q["country"].nunique() < 3:
                continue
            b = _fit(q, "x", ["y12"])
            f = _fit(q, "x", ["y12"], tfe=True)
            d = _fit(q, "dx", ["y12"])
            c = _fit(q, "x", ["y12", "y3", "y1"])
            rows.append({"metric": m, "target": tgt, "n": b[2],
                         "base_b": b[0], "base_t": b[1], "tfe_b": f[0], "tfe_t": f[1],
                         "dx_b": d[0], "dx_t": d[1], "ctl_b": c[0], "ctl_t": c[1],
                         "oos_r2_pct": 100 * _oos_r2(q)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    out = run()
    print(out.round(2).to_string(index=False))
    n = len(out)
    from scipy.stats import norm
    bar = norm.ppf(1 - 0.05 / (2 * n))
    print(f"\n{n} tests; Bonferroni |t| bar at 5% = {bar:.2f}")
    print("pass base |t|>=bar:", out.loc[out["base_t"].abs() >= bar, ["metric", "target"]].values.tolist())
