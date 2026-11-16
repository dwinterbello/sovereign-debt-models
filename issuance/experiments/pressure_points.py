"""Pressure points: does price behave differently when a metric sits at an extreme of its own
history, rather than in proportion to it?

Same inputs as `issuance.evidence` (expanding own-country percentile of each headline metric,
signed so high = more supply pressure; forward change in the target from when the metric is
public). Instead of one slope, compare the forward move when the metric is in its top decile
(top quintile, bottom decile) with every other month, after removing each country's mean move.

    top10   mean excess move when percentile ≥ 90, minus the mean when it is not (bp)
    lin10   what the linear fit implies for the same split: β × (mean x in top − mean x rest)
    conv    top10 / lin10 — above 1 means the extreme does more than the slope says (convex)
    p_rot   share of circular shifts of x against y (same shift for every country, ≥12 months)
            with |top10| at least as large; overlapping windows and persistence kept intact
    eps     separate episodes in the top decile (runs of consecutive months, all countries)
    q1..q5  mean excess move by quintile of the percentile (shape check)

Run:  PYTHONPATH=. .venv/bin/python -m issuance.experiments.pressure_points
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import evidence as E
from issuance.headline import METRICS, lag_for

OUT = "/tmp/pressure_points.csv"
START = "2010-01-01"
CUTS = {"top10": 4.0, "top20": 3.0}
MIN_IN = 12


def _excess(q: pd.DataFrame) -> pd.Series:
    return q["y"] - q.groupby("country")["y"].transform("mean")


def _split(q: pd.DataFrame, cut: float, low: bool = False) -> float | None:
    inn = q["x"] <= -cut if low else q["x"] >= cut
    if inn.sum() < MIN_IN or (~inn).sum() < MIN_IN:
        return None
    e = _excess(q)
    return float(e[inn].mean() - e[~inn].mean()) * 100


def _episodes(q: pd.DataFrame, cut: float) -> int:
    n = 0
    for _, g in q.sort_values("month").groupby("country"):
        inn = (g["x"] >= cut).to_numpy()
        n += int(inn[0]) + int(((inn[1:]) & (~inn[:-1])).sum()) if len(inn) else 0
    return n


def _rotation_p(q: pd.DataFrame, stat: float, cut: float, low: bool = False) -> float | None:
    months = np.sort(q["month"].unique())
    n = len(months)
    if n - 2 * E.MIN_SHIFT + 1 < E.MIN_SHIFTS:
        return None
    wide = q.pivot_table(index="month", columns="country", values="x").reindex(months)
    hits = tot = 0
    for k in range(E.MIN_SHIFT, n - E.MIN_SHIFT + 1):
        rolled = pd.DataFrame(np.roll(wide.to_numpy(), k, axis=0), index=pd.Index(months, name="month"), columns=wide.columns)
        xs = rolled.stack().rename("x").reset_index()
        sq = q.drop(columns="x").merge(xs, on=["month", "country"]).dropna(subset=["x"])
        s = _split(sq, cut, low)
        if s is None:
            continue
        tot += 1
        hits += abs(s) >= abs(stat)
    return hits / tot if tot >= E.MIN_SHIFTS else None


def run() -> pd.DataFrame:
    xz = E.pressure_z()
    fw = E.forward_changes()
    rows = []
    for m in METRICS:
        xm = xz[xz["metric"] == m].copy()
        if xm.empty:
            continue
        xm["lag"] = [lag_for(m, c) for c in xm["country"]]
        for tgt in E.TARGETS:
            for h in E.HORIZONS:
                q = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"), on=["country", "month", "lag"])
                q = q[q["month"] >= START].dropna(subset=["x", "y"])
                if len(q) < E.MIN_OBS:
                    continue
                lin = E._ols(q.dropna(subset=["y12"]), h)
                e = _excess(q)
                qb = pd.cut(q["x"], [-5.01, -3, -1, 1, 3, 5.01], labels=["q1", "q2", "q3", "q4", "q5"])
                quint = (e.groupby(qb, observed=False).mean() * 100).round(1).to_dict()
                for name, cut in CUTS.items():
                    s = _split(q, cut)
                    if s is None:
                        continue
                    inn = q["x"] >= cut
                    lin10 = lin[0] * (q.loc[inn, "x"].mean() - q.loc[~inn, "x"].mean()) if lin else np.nan
                    rows.append({"metric": m, "target": tgt, "h": h, "split": name, "diff_bp": round(s, 1),
                                 "lin_bp": round(lin10, 1), "conv": round(s / lin10, 2) if lin10 and abs(lin10) > 0.5 else np.nan,
                                 "n_in": int(inn.sum()), "eps": _episodes(q, cut),
                                 "countries": " ".join(sorted(q.loc[inn, "country"].unique())),
                                 "p_rot": _rotation_p(q, s, cut), **quint})
                s = _split(q, CUTS["top10"], low=True)
                if s is not None:
                    inn = q["x"] <= -CUTS["top10"]
                    rows.append({"metric": m, "target": tgt, "h": h, "split": "bot10", "diff_bp": round(s, 1),
                                 "n_in": int(inn.sum()), "countries": " ".join(sorted(q.loc[inn, "country"].unique())),
                                 "p_rot": _rotation_p(q, s, CUTS["top10"], low=True), **quint})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 30)
    r = run()
    sig = r[(r["p_rot"] <= 0.10)].sort_values(["split", "p_rot"])
    print(f"{len(r)} tests, {len(sig)} with rotation p ≤ 0.10 (≈{0.10 * len(r):.0f} expected by chance)")
    print(sig.to_string(index=False))
