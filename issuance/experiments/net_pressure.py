"""Net pressure: supply percentile minus demand percentile, tested like any headline metric.

Each leg is the expanding own-country percentile from `issuance.evidence` (signed so high = more
supply pressure; scaled ±5 = 0th/100th). A net measure is the mean of its legs, re-ranked in its
own history to date so it is again a uniform percentile, and then run through the same linear test
(`evidence._fit`, `evidence.verdict`) and the same extreme-decile test as `pressure_points`.

Legs (supply side: priv12, absorb, absorb_long, gross12; demand side, already sign-flipped: score,
cover, cb_share). Lag = the slower leg's publication lag.

Run:  PYTHONPATH=. .venv/bin/python -m issuance.experiments.net_pressure
"""
from __future__ import annotations

import pandas as pd

from issuance import evidence as E
from issuance.experiments import pressure_points as P
from issuance.headline import lag_for

OUT = "/tmp/net_pressure.csv"
NETS = {
    "absorb−score": ("absorb", "score"),
    "priv12−score": ("priv12", "score"),
    "absorb−cover": ("absorb", "cover"),
    "absorb−cb": ("absorb", "cb_share"),
    "absorb_long−score": ("absorb_long", "score"),
    "absorb−score−cb": ("absorb", "score", "cb_share"),
    "gross12−cover": ("gross12", "cover"),
}


def net_panel(xz: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, legs in NETS.items():
        w = xz[xz["metric"].isin(legs)].pivot_table(index=["country", "month"], columns="metric", values="x")
        if not set(legs) <= set(w.columns):
            continue
        w = w[list(legs)].dropna()
        for c, g in w.groupby(level="country"):
            v = g.mean(axis=1).droplevel("country").sort_index().asfreq("MS")
            x = E.pressure_pct(v)
            lag = max(lag_for(m, c) for m in legs)
            out.append(pd.DataFrame({"metric": name, "country": c, "month": x.index, "x": x.to_numpy(), "lag": lag}))
    return pd.concat(out, ignore_index=True).dropna(subset=["x"])


def run() -> pd.DataFrame:
    xz = E.pressure_z()
    fw = E.forward_changes()
    xn = net_panel(xz)
    rows = []
    for name in NETS:
        xm = xn[xn["metric"] == name]
        for tgt in E.TARGETS:
            for h in E.HORIZONS:
                q = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"), on=["country", "month", "lag"])
                q = q[q["month"] >= "2010-01-01"]
                got = E._fit(q, h)
                if not got:
                    continue
                v = E.verdict(got["beta_bp"], got["t"], got["loo_min_t"], got["p_chance"], got["loo_max_t"], got["oos_r2"])
                qq = q.dropna(subset=["y"])
                top = P._split(qq, 4.0)
                rows.append({"net": name, "target": tgt, "h": h, "beta_bp": round(got["beta_bp"], 2), "t": round(got["t"], 2),
                             "loo_min_t": got["loo_min_t"], "p": got["p_chance"], "oos_r2": got["oos_r2"], "verdict": v,
                             "top10_bp": None if top is None else round(top, 1),
                             "top10_p": None if top is None else P._rotation_p(qq, top, 4.0),
                             "n_top": int((qq["x"] >= 4).sum()), "eps": P._episodes(qq, 4.0),
                             "countries": " ".join(sorted(q["country"].unique()))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 300)
    r = run()
    print(r.to_string(index=False))
