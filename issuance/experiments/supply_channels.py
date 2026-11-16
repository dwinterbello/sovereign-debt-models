"""Where would a supply effect show up if the 3-month forward test misses it?

Same pooled panel as the evidence table (country-demeaned, Driscoll–Kraay), 2010+, x = metric
z-scored in its own country and signed so + = more supply pressure.

  fwd_h    price change over the next h months (3, 6, 12) from publication, past 12m change control
  co12     contemporaneous: 12m change in price on 12m change in x (does supply move *with* price)
  co12_tfe co12 with month fixed effects (common global moves removed; only cross-country gaps count)

Targets add y10 − y2 (strips most of the policy-rate path, so closer to a term premium) to the
production set.

Run: PYTHONPATH=. .venv/bin/python -m issuance.experiments.supply_channels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance import evidence as E, market
from issuance.headline import METRICS, lag_for

TARGETS = ("y10", "s210", "s1030", "rel10")
HORIZONS = (3, 6, 12)


def _targets() -> pd.DataFrame:
    m = market.load()
    w = m.pivot_table(index=["country", "month"], columns="series", values="value")
    w["s210"] = w["y10"] - w["y2"]
    return w.reset_index()


def _fit(q: pd.DataFrame, xcol: str, ycol: str, ctl: list[str], maxlags: int, tfe: bool = False):
    q = q.dropna(subset=[xcol, ycol, *ctl])
    if len(q) < E.MIN_OBS or q["country"].nunique() < 3:
        return np.nan, np.nan
    cols = [xcol, ycol, *ctl]
    d = q[cols] - q.groupby("country")[cols].transform("mean")
    if tfe:
        d = d - d.groupby(q["month"]).transform("mean")
    X = sm.add_constant(d[[xcol, *ctl]].to_numpy())
    fit = sm.OLS(d[ycol].to_numpy(), X).fit(
        cov_type="hac-groupsum", cov_kwds={"time": q["month"].factorize(sort=True)[0], "maxlags": maxlags})
    return float(fit.params[1]) * 100, float(fit.tvalues[1])


def run() -> pd.DataFrame:
    xz = E.pressure_z()
    w = _targets()
    rows = []
    for metric in METRICS:
        xm = xz[xz["metric"] == metric].copy()
        if xm.empty:
            continue
        xm["lag"] = [lag_for(metric, c) for c in xm["country"]]
        for tgt in TARGETS:
            parts = []
            for c, g in w[["country", "month", tgt]].dropna().groupby("country"):
                y = g.set_index("month")[tgt].sort_index().asfreq("MS")
                for lag in sorted(set(xm.loc[xm["country"] == c, "lag"])):
                    f = pd.DataFrame({"country": c, "lag": lag, "month": y.index,
                                      "y12": (y - y.shift(12)).to_numpy()})
                    for h in HORIZONS:
                        f[f"f{h}"] = (y.shift(-(lag + h)) - y.shift(-lag)).to_numpy()
                    parts.append(f)
            if not parts:
                continue
            df = xm.merge(pd.concat(parts), on=["country", "month", "lag"]).sort_values(["country", "month"])
            df["dx12"] = df.groupby("country")["x"].diff(12)
            df = df[df["month"] >= "2010-01-01"]
            if df["country"].nunique() < 3:
                continue
            r = {"metric": metric, "target": tgt}
            for h in HORIZONS:
                r[f"fwd{h}_b"], r[f"fwd{h}_t"] = _fit(df, "x", f"f{h}", ["y12"], h + 1)
            r["co12_b"], r["co12_t"] = _fit(df, "dx12", "y12", [], 13)
            r["co12_tfe_b"], r["co12_tfe_t"] = _fit(df, "dx12", "y12", [], 13, tfe=True)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    out = run()
    print(out.round(2).to_string(index=False))
