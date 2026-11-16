"""Daily auction-window event study: do yields cheapen into supply and recover after it?

For every nominal coupon auction (US auctions and reopenings, UK auctions and syndications, DE and
JP auctions; FR/IT have no daily yields here) since 2010, on that country's daily par yield at the
nearest benchmark (<5y → 2y, 5–15y → 10y, >15y → 30y), in business days of that series:

  pre   y(t−1) − y(t−6)    the five days into the auction
  day   y(t) − y(t−1)      auction day (every auction here prices before the local close)
  post  y(t+5) − y(t)      the five days after

Each window outright and relative to the same tenor's mean change in the other three markets
(removes global moves). Several auctions mapping to one country-day-tenor are one event with
summed DV01. Size = log(DV01 / trailing 2-year median DV01 for that country and benchmark).

Reports mean (bp, + = cheapening), t with HAC errors in event order (5 lags), share positive,
and the slope on size. Concession trade = pre + day (short into the auction) and post (long after).

Run: PYTHONPATH=. .venv/bin/python -m issuance.experiments.auction_window
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance import market
from issuance.isin_flow import load_ops, market_ops
from issuance.paths import CACHE
from issuance.pricing import dv01_mn

COUNTRIES = ("UK", "DE", "US", "JP")
OPS = ("Auction", "Reopening", "Syndication")
DAILY = CACHE / "market_daily_yields.parquet"
PRE, POST = 5, 5


def daily_yields(refresh: bool = False) -> pd.DataFrame:
    if DAILY.exists() and not refresh:
        return pd.read_parquet(DAILY)
    rows = []
    for c, name, _, s in market._daily_sets():
        if c in market.DAILY and name in ("y2", "y3", "y10", "y30"):
            s = s[s.index >= "2009-06-01"]
            rows.append(pd.DataFrame({"country": c, "series": name, "date": s.index, "value": s.to_numpy()}))
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(DAILY, index=False)
    return d


def bench(years: float) -> str:
    return "y2" if years < 5 else ("y10" if years <= 15 else "y30")


def events() -> pd.DataFrame:
    ops = market_ops(load_ops())
    o = ops[ops["country"].isin(COUNTRIES) & ops["op_type"].astype(str).isin(OPS)
            & ops["sec_type"].astype(str).eq("Nominal") & (ops["amount_issued"] > 0)
            & (pd.to_datetime(ops["op_dt"]) >= "2008-01-01")].copy()
    o["op_dt"] = pd.to_datetime(o["op_dt"]).dt.normalize()
    yrs = (pd.to_datetime(o["maturity_dt"]) - o["op_dt"]).dt.days / 365.25
    o = o[yrs >= 1.5]
    yrs = yrs.loc[o.index]
    o["years"] = yrs
    o["bench"] = [bench(v) for v in yrs]
    o["dv01"] = dv01_mn(o["amount_issued"].to_numpy(float), yrs.to_numpy(float),
                        o["coupon"].to_numpy(float), o["yield"].fillna(3.0).to_numpy(float))
    o["synd"] = o["op_type"].astype(str).eq("Syndication")
    e = o.groupby(["country", "op_dt", "bench"], as_index=False).agg(dv01=("dv01", "sum"), synd=("synd", "max"),
                                                                    n=("dv01", "size"))
    e = e.sort_values("op_dt")
    med = []
    for (c, b), g in e.groupby(["country", "bench"]):
        s = g.set_index("op_dt")["dv01"]
        m = s.rolling("730D", closed="left").median()
        med.append(pd.Series(m.to_numpy(), index=g.index))
    e["size"] = np.log(e["dv01"] / pd.concat(med).reindex(e.index))
    return e


def windows(e: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    wide = {k: g.pivot_table(index="date", columns="country", values="value").sort_index()
            for k, g in y.groupby("series")}
    out = []
    for r in e.itertuples():
        w = wide.get(r.bench)
        if w is None or r.country not in w:
            continue
        s = w[r.country].dropna()
        pos = s.index.searchsorted(r.op_dt)
        if pos >= len(s) or s.index[pos] != r.op_dt or pos < PRE + 1 or pos + POST >= len(s):
            continue
        d = s.index
        lv = lambda k: s.iat[k]
        peers = [c for c in w.columns if c != r.country]

        def peer(a, b):
            p = w[peers].reindex([d[a], d[b]])
            return float((p.iloc[1] - p.iloc[0]).mean()) if p.notna().all().any() else np.nan

        pre, day, post = lv(pos - 1) - lv(pos - 1 - PRE), lv(pos) - lv(pos - 1), lv(pos + POST) - lv(pos)
        out.append({"country": r.country, "date": r.op_dt, "bench": r.bench, "size": r.size, "synd": r.synd,
                    "pre": pre * 100, "day": day * 100, "post": post * 100,
                    "pre_rel": (pre - peer(pos - 1 - PRE, pos - 1)) * 100,
                    "day_rel": (day - peer(pos - 1, pos)) * 100,
                    "post_rel": (post - peer(pos, pos + POST)) * 100})
    return pd.DataFrame(out)


def _mean_t(v: pd.Series) -> tuple[float, float]:
    v = v.dropna()
    if len(v) < 20:
        return np.nan, np.nan
    f = sm.OLS(v.to_numpy(), np.ones(len(v))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(f.params[0]), float(f.tvalues[0])


def _slope_t(d: pd.DataFrame, col: str) -> tuple[float, float]:
    q = d.dropna(subset=[col, "size"])
    if len(q) < 30:
        return np.nan, np.nan
    f = sm.OLS(q[col].to_numpy(), sm.add_constant(q["size"].to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(f.params[1]), float(f.tvalues[1])


def summarise(w: pd.DataFrame) -> pd.DataFrame:
    w = w.sort_values("date").assign(into=lambda d: d["pre_rel"] + d["day_rel"])
    rows = []
    periods = {"2010+": ("2010-01-01", None), "2010–18": ("2010-01-01", "2018-12-31"), "2019+": ("2019-01-01", None)}
    groups = [("all", w)] + [(c, g) for c, g in w.groupby("country")] + [(f"{c} {b}", g) for (c, b), g in w.groupby(["country", "bench"])]
    for pname, (a, b) in periods.items():
        for gname, g in groups:
            g = g[(g["date"] >= a) & ((g["date"] <= b) if b else True)]
            if len(g) < 20:
                continue
            r = {"period": pname, "group": gname, "n": len(g)}
            for col in ("pre", "day", "post", "pre_rel", "day_rel", "post_rel", "into"):
                r[col], r[col + "_t"] = _mean_t(g[col])
            r["post_rel_pos"] = float((g["post_rel"] < 0).mean())
            r["size_into"], r["size_into_t"] = _slope_t(g, "into")
            r["size_post"], r["size_post_t"] = _slope_t(g, "post_rel")
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    y = daily_yields(refresh="--refresh" in sys.argv)
    w = windows(events(), y)
    w.to_parquet("/tmp/auction_window_events.parquet", index=False)
    s = summarise(w)
    cols = ["period", "group", "n", "pre_rel", "pre_rel_t", "day_rel", "day_rel_t", "post_rel", "post_rel_t",
            "into", "into_t", "size_into", "size_into_t", "size_post", "size_post_t", "pre", "post"]
    print(s[cols].round(2).to_string(index=False))
