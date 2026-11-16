"""Evidence regression under every transformation of the metric.

Same test as the production evidence table (country-demeaned pooled OLS, past 12m price change
control, Driscoll–Kraay, leave-one-country-out, rotation p, real-time OOS R² from 2016), 2010+,
horizons 3 and 12 months, pooled targets only. x, signed so + = more supply pressure:

  level   raw value (country means removed by the regression)
  z36 / z60   rolling z-score, 36 / 60 months
  pct60   rolling 60-month percentile
  pctx    expanding percentile (all history to date)
  d3 / d12    3- / 12-month change

Run: PYTHONPATH=. .venv/bin/python -m issuance.experiments.evidence_transforms
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import evidence as E
from issuance.headline import METRICS, lag_for, panel
from issuance.rv_metrics import rolling_z

TARGETS = ("y10", "rel10", "s1030")
METRIC_SET = ("priv12", "absorb", "absorb_long", "cb_share", "score", "cover", "gross12", "imbalance")


def _rank_last(a: np.ndarray) -> float:
    return (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean()


def transforms(v: pd.Series) -> dict[str, pd.Series]:
    return {
        "level": v,
        "z36": rolling_z(v, 36, 24),
        "z60": rolling_z(v, 60, 24),
        "pct60": v.rolling(60, min_periods=24).apply(_rank_last, raw=True),
        "pctx": v.expanding(min_periods=24).apply(_rank_last, raw=True),
        "d3": v.diff(3),
        "d12": v.diff(12),
    }


def x_panel() -> pd.DataFrame:
    p = panel()
    h = p[(p["phase"] == "history") & p["metric"].isin(METRIC_SET)].sort_values(["metric", "country", "month"])
    out = []
    for (m, c), g in h.groupby(["metric", "country"]):
        v = g.set_index("month")["value"].asfreq("MS") * METRICS[m].get("sign", 1)
        for name, s in transforms(v).items():
            out.append(pd.DataFrame({"metric": m, "transform": name, "country": c, "month": s.index, "x": s.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def run() -> pd.DataFrame:
    xp = x_panel()
    fw = E.forward_changes()
    rows = []
    for (m, tr), xm in xp.groupby(["metric", "transform"]):
        xm = xm.drop(columns=["metric", "transform"]).copy()
        xm["lag"] = [lag_for(m, c) for c in xm["country"]]
        for tgt in TARGETS:
            for h in E.HORIZONS:
                df = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns=["h", "target"]),
                              on=["country", "month", "lag"])
                sub = df[df["month"] >= "2010-01-01"]
                if sub["country"].nunique() < 3:
                    continue
                got = E._fit(sub, h)
                if not got:
                    continue
                v = E.verdict(got["beta_bp"], got["t"], got["loo_min_t"], got["p_chance"], got["loo_max_t"], got["oos_r2"])
                rows.append({"metric": m, "transform": tr, "target": tgt, "h": h, "t": got["t"],
                             "loo_min_t": got["loo_min_t"], "p": got["p_chance"], "oos_r2": got["oos_r2"], "verdict": v})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    out = run()
    out.to_csv("/tmp/evidence_transforms.csv", index=False)
    piv = out.pivot_table(index=["metric", "target", "h"], columns="transform", values="t").round(2)
    print(piv.to_string())
    print()
    print(out.pivot_table(index=["metric", "target", "h"], columns="transform", values="oos_r2").mul(100).round(1).to_string())
    print()
    print(out[out["verdict"].isin(["forward link", "wrong sign"])].round(3).sort_values(["verdict", "t"]).to_string(index=False))
