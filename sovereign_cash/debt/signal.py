"""Does the supply / demand gap price? Panel test on six markets, 2012 → today.

The Dalio framing says that when private buyers have to absorb more duration (deficits plus central-
bank runoff), or debt service crowds the budget, yields and term premia rise. Each candidate is a
monthly country series, known at the month it is used:

  priv_tr    private net supply, trailing 12m, % GDP (net issuance − central-bank net purchases), lag 1m
  priv_fw    the same over the *next* 12m, realised — perfect foresight; issuance plans are largely
             published a year ahead, so this is the most generous version of the claim
  cb_tr      central-bank net purchases, trailing 12m, % GDP, lag 1m
  roll       rollovers (bills + coupons inside a year), % GDP, lag 1m
  bills      bill share of the marketable stock, lag 1m
  int_rev    general-government gross interest / revenue (OECD, IMF), last completed year (from May;
             Y−2 before); general government because central-government history is not published for all six

Targets: the next 12 months' change (bp) in the 10y yield, the 10y−3m term spread and 10s30s.
Specs: country fixed effects (does a country price its own supply over time?) and two-way fixed
effects (country + month: relative value — does the country whose supply rises more than the others'
underperform them?). Month effects alone would compare persistent country levels across six
countries (Japan's rollovers are 39% of GDP, Germany's 6%), not supply news. Driscoll-
Kraay errors (12 lags). Honest p: each regressor replaced by an AR(1) with its own country
persistence and scale, 300 draws. Checks: 2012–19 vs 2020→ halves, leave one country out, and
non-overlapping September-to-September changes. A result counts only at placebo p ≤ 0.15 with the
same sign in both halves and in the non-overlapping sample — the bar used across this repo.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.signal [n_sim]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash._vendor import dk_ols

HERE = Path(__file__).resolve().parent
DERIVED = HERE.parent / "derived"
MARKET = HERE.parents[1] / "issuance" / "cache" / "market_monthly.parquet"
CC = ("us", "uk", "de", "fr", "it", "jp")
H = 12
XS = ("priv_tr", "priv_fw", "cb_tr", "roll", "bills", "int_rev")
YS = {"y10": "10y yield", "tp10": "10y − 3m term spread", "s1030": "10s30s"}
START, END = "2012-01-01", "2026-09-01"
RNG = np.random.default_rng(7)
# Sign the supply thesis predicts: more private supply / rollovers / bills / interest burden -> higher
# yields, wider term spread, steeper curve; more central-bank buying -> the opposite.
THESIS = {"priv_tr": 1, "priv_fw": 1, "cb_tr": -1, "roll": 1, "bills": 1, "int_rev": 1}


def panel() -> pd.DataFrame:
    hist = pd.read_parquet(DERIVED / "debt_history.parquet")
    ann = pd.read_parquet(DERIVED / "debt_annual.parquet")
    hist["month"] = pd.to_datetime(hist["month"])
    m = pd.read_parquet(MARKET)
    m["month"] = pd.to_datetime(m["month"])
    m = m[m["series"].isin(list(YS))].pivot_table(index=["country", "month"], columns="series", values="value")
    rows = []
    for cc in CC:
        h = hist[hist["cc"] == cc].set_index("month").sort_index()
        idx = pd.date_range(START, h.index.max(), freq="MS")
        h = h.reindex(idx)
        d = pd.DataFrame(index=idx)
        d["priv_tr"] = h["private12_pct_gdp"].shift(1)
        d["priv_fw"] = h["private12_pct_gdp"].shift(-H)
        d["cb_tr"] = h["cb12_pct_gdp"].shift(1)
        d["roll"] = h["rollover_pct_gdp"].shift(1)
        d["bills"] = 100 * h["bill_share_stock"].shift(1)
        a = ann[ann["country"] == cc].set_index("year")["gg_interest_pct_rev"]
        yr = np.where(idx.month >= 5, idx.year - 1, idx.year - 2)
        d["int_rev"] = a.reindex(yr).to_numpy()
        mk = m.loc[cc.upper()] if cc.upper() in m.index.get_level_values(0) else pd.DataFrame()
        for y in YS:
            s = mk[y].reindex(idx) * 100 if y in mk else pd.Series(np.nan, idx)
            d[f"d_{y}"] = s.shift(-H) - s
        d["hist_ok"] = ~h["is_forecast"].fillna(True).astype(bool)
        d["country"] = cc
        rows.append(d.rename_axis("month").reset_index())
    p = pd.concat(rows, ignore_index=True)
    # forward targets and perfect-foresight supply must sit inside history
    p = p[p["month"] <= pd.Timestamp(END) - pd.DateOffset(months=H)]
    return p


def _demean(q: pd.DataFrame, cols: list[str], fe: str) -> pd.DataFrame:
    q = q.copy()
    for c in cols:
        if fe in ("country", "two"):
            q[c] = q[c] - q.groupby("country")[c].transform("mean")
        if fe in ("month", "two"):
            q[c] = q[c] - q.groupby("month")[c].transform("mean")
    return q


def _fit(p: pd.DataFrame, y: str, x: str, fe: str, a: str = START, b: str = END) -> dict | None:
    q = p[(p["month"] >= a) & (p["month"] <= b)].dropna(subset=[y, x])
    if fe != "country":
        q = q[q.groupby("month")["country"].transform("count") >= 4]
    if q["country"].nunique() < 3 or len(q) < 60:
        return None
    q = _demean(q, [y, x], fe)
    return dk_ols(q, y, [x], lags=H)


def _ar1(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    if len(x) < 24:
        return 0.9, float(x.std() or 1.0), float(x.mean() if len(x) else 0.0)
    rho, c = np.polyfit(x.shift(1).dropna().to_numpy(), x.iloc[1:].to_numpy(), 1)
    e = x.iloc[1:].to_numpy() - (c + rho * x.shift(1).dropna().to_numpy())
    return float(min(rho, 0.995)), float(e.std()), float(x.mean())


def _sim(rho: float, sig: float, mu: float, n: int) -> np.ndarray:
    e = RNG.normal(0, sig, n + 200)
    s = np.zeros_like(e)
    for j in range(1, len(e)):
        s[j] = rho * s[j - 1] + e[j]
    return mu + s[200:]


def placebo(p: pd.DataFrame, y: str, x: str, fe: str, t_obs: float, n: int) -> float:
    pars = {c: _ar1(g.set_index("month")[x]) for c, g in p.groupby("country")}
    hits, done = 0, 0
    for _ in range(n):
        pp = p.copy()
        for c, (rho, sig, mu) in pars.items():
            msk = (pp["country"] == c) & pp[x].notna()
            pp.loc[msk, x] = _sim(rho, sig, mu, int(msk.sum()))
        f = _fit(pp, y, x, fe)
        if f is None:
            continue
        done += 1
        hits += abs(f["t"][x]) >= abs(t_obs)
    return hits / done if done else np.nan


def nonoverlap(p: pd.DataFrame, y: str, x: str, fe: str) -> dict | None:
    q = p[p["month"].dt.month == 9].dropna(subset=[y, x])
    if len(q) < 15:
        return None
    q = _demean(q, [y, x], fe)
    X, Y = q[x].to_numpy(), q[y].to_numpy()
    b = float(X @ Y / (X @ X))
    e = Y - b * X
    se = float(np.sqrt(e @ e / (len(Y) - 1) / (X @ X)))
    return {"beta": b, "t": b / se, "n": int(len(q))}


def oos_r2(p: pd.DataFrame, y: str, x: str, fe: str, start: str = "2018-01-01") -> float:
    """Expanding-window out-of-sample R² against the country-mean forecast.

    At each origin month t the slope and country means are estimated on rows whose 12m target ended
    by t. Two-way: the target and regressor are first demeaned across countries within each month,
    so the forecast is of relative performance.
    """
    q = p.dropna(subset=[y, x]).copy()
    if fe == "two":
        q = q[q.groupby("month")["country"].transform("count") >= 4]
        for c in (y, x):
            q[c] = q[c] - q.groupby("month")[c].transform("mean")
    sse_m = sse_b = 0.0
    for t in sorted(q.loc[q["month"] >= start, "month"].unique()):
        tr = q[q["month"] <= pd.Timestamp(t) - pd.DateOffset(months=H)]
        te = q[q["month"] == t]
        if tr["country"].nunique() < 3 or len(tr) < 60:
            continue
        mx, my = tr.groupby("country")[x].mean(), tr.groupby("country")[y].mean()
        xd = tr[x] - tr["country"].map(mx)
        yd = tr[y] - tr["country"].map(my)
        b = float(xd @ yd / (xd @ xd))
        bench = te["country"].map(my)
        pred = bench + b * (te[x] - te["country"].map(mx))
        ok = bench.notna() & pred.notna()
        sse_m += float(((te[y] - bench)[ok] ** 2).sum())
        sse_b += float(((te[y] - pred)[ok] ** 2).sum())
    return 1 - sse_b / sse_m if sse_m else np.nan


def run(n_sim: int = 300) -> dict:
    p = panel()
    out = {"tests": [], "n_sim": n_sim, "window": f"{START[:7]} → {p['month'].max():%Y-%m} (start of 12m change)"}
    for y in YS:
        yc = f"d_{y}"
        for x in XS:
            for fe in ("country", "two"):
                f = _fit(p, yc, x, fe)
                if f is None:
                    continue
                t = f["t"][x]
                r = {"y": y, "x": x, "fe": fe, "beta": f["beta"][x], "t": t, "n": f["n"], "r2": f["r2"],
                     "p_placebo": placebo(p, yc, x, fe, t, n_sim)}
                for lab, a, b in (("pre", "2012-01-01", "2019-12-01"), ("post", "2020-01-01", END)):
                    g = _fit(p, yc, x, fe, a, b)
                    r[f"beta_{lab}"] = g["beta"][x] if g else np.nan
                    r[f"t_{lab}"] = g["t"][x] if g else np.nan
                lo = [(_fit(p[p["country"] != c], yc, x, fe) or {}).get("beta", {}).get(x, np.nan) for c in CC]
                r["loco_min"], r["loco_max"] = float(np.nanmin(lo)), float(np.nanmax(lo))
                no = nonoverlap(p, yc, x, fe)
                r["beta_nonoverlap"], r["t_nonoverlap"] = (no["beta"], no["t"]) if no else (np.nan, np.nan)
                r["oos_r2"] = oos_r2(p, yc, x, fe)
                s = np.sign(r["beta"])
                r["passes"] = bool(r["p_placebo"] <= 0.15 and np.sign(r["beta_pre"]) == s
                                   and np.sign(r["beta_post"]) == s and np.sign(r["beta_nonoverlap"]) == s
                                   and np.sign(r["loco_min"]) == np.sign(r["loco_max"]) == s)
                r["thesis_sign"] = THESIS[x]
                r["supports_thesis"] = bool(r["passes"] and s == THESIS[x])
                r["against_thesis"] = bool(r["passes"] and s == -THESIS[x])
                out["tests"].append(r)
    t = pd.DataFrame(out["tests"])
    out["n_tests"] = int(len(t))
    out["n_p05"] = int((t["p_placebo"] <= 0.05).sum())
    out["n_pass"] = int(t["passes"].sum())
    out["n_support"] = int(t["supports_thesis"].sum())
    out["n_against"] = int(t["against_thesis"].sum())
    out["expected_p05_by_chance"] = round(0.05 * len(t), 1)
    (DERIVED / "debt_signal.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    o = run(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
    t = pd.DataFrame(o["tests"])
    pd.set_option("display.width", 250)
    cols = ["y", "x", "fe", "beta", "t", "p_placebo", "beta_pre", "beta_post", "beta_nonoverlap", "loco_min",
            "loco_max", "oos_r2", "n", "passes", "supports_thesis"]
    print(t[cols].round(2).to_string())
    print({k: o[k] for k in ("n_tests", "n_p05", "expected_p05_by_chance", "n_pass", "n_support", "n_against",
                             "window")})
