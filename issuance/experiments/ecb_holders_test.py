"""Do euro-area holder-sector metrics (ECB SHS/SHSS, QSA, BSI) line up with the next move in price, and does
domestic bank buying lean against spreads in stress (Altavilla–Pagano–Simonelli)?

    PYTHONPATH=. .venv/bin/python -m issuance.experiments.ecb_holders_test

Same machinery as `issuance.evidence` (no re-implementation): x = `evidence.pressure_pct` of the metric ×
its `ecb_holders.METRICS` sign; y / y12 = `evidence.forward_changes`; fit = `evidence._fit` (pooled
within-country DK OLS with the y12 control, LOO, circular-shift p, real-time OOS R² from 2016);
`evidence.verdict`; `evidence.chance`. Metrics are stamped at the month their last input became public
(`ecb_holders.metrics`) and the y window starts one month later (LAG = 1).

Only DE, FR, IT: leave-one-out drops a third of the panel, and spread_de pools FR and IT only, so LOO
does not run there (`evidence.POOLED` = 3). Rows that `_fit` cannot score (n < MIN_OBS) are kept with
verdict 'untested' and state 'short'; scored rows with no rotation (fewer than MIN_SHIFTS shifts) carry
state 'descriptive' — their p_chance is NaN and does not block, so read them as in-sample only.

(a) main     every metric × `evidence.TARGETS` × h ∈ {3, 12} × windows 2010+, 2010–18, 2019+.
(b) inter    bank-buying metrics × state, `evidence.fit_interaction`, 2010+. States: 'rvol' (from
             `evidence.states`) and 'stress' = own-country expanding percentile of the spread to Bunds
             (FR, IT; market data, public at t). β2 > 0 = bank buying is *less* of a demand signal in stress
             (x is signed so + = less buying); the nexus reading (domestic banks absorb in stress, which
             calms spreads) predicts β2 < 0 → 'wrong sign'.
(c) nexus    descriptive: corr of 12m domestic bank buying (% GDP) with the same-window 12m change in
             spread_de, in the top stress tercile vs the rest (FR, IT, 2010+).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import ecb_holders as eh
from issuance import evidence as ev
from issuance import market
from issuance.paths import CACHE

OUT = CACHE / "ecb_holders_evidence.parquet"
LAG = 1
BANK = ("bsi_bank_f3m", "bsi_bank_f12m", "ecb_bank_d4q")
INTER_TARGETS = ("spread_de", "y10")


def pressure(rows: pd.DataFrame) -> pd.DataFrame:
    """metric, country, month, x (signed by `eh.METRICS`)."""
    out = []
    for (m, c), g in rows[rows["phase"] == "history"].groupby(["metric", "country"]):
        v = g.set_index("month")["value"].sort_index().asfreq("MS") * eh.METRICS[m]["sign"]
        x = ev.pressure_pct(v)
        out.append(pd.DataFrame({"metric": m, "country": c, "month": x.index, "x": x.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def _pairs(rows: pd.DataFrame, mkt: pd.DataFrame, metrics=None, targets=None):
    xz = pressure(rows)
    fw = ev.forward_changes(mkt, lags=(LAG,))
    for m in eh.METRICS if metrics is None else metrics:
        xm = xz[xz["metric"] == m].assign(lag=LAG)
        if xm.empty:
            continue
        for tgt in ev.TARGETS if targets is None else targets:
            for h in ev.TARGET_HORIZONS.get(tgt, ev.HORIZONS):
                yield m, tgt, h, xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"),
                                          on=["country", "month", "lag"])


def main_tests(rows: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = eh.metrics() if rows is None else rows
    mkt = market.load() if mkt is None else mkt
    out = []
    for m, tgt, h, df in _pairs(rows, mkt):
        for wname in ev.WINDOWS:
            sub = ev._window(df, wname)
            base = {"test": "main", "metric": m, "label": eh.METRICS[m]["label"], "target": tgt,
                    "target_label": ev.TARGETS[tgt][0], "window": wname, "lag": str(LAG), "horizon": h,
                    "countries": " ".join(sorted(sub["country"].unique()))}
            got = ev._fit(sub, h)
            if not got:
                q = sub.dropna(subset=["x", "y", "y12"])
                out.append({**base, "n_obs": len(q), "n_countries": int(q["country"].nunique()),
                            "state": "short"})
                continue
            got["r12"] *= ev.CO_SIGN.get(tgt, 1)
            st = "descriptive" if got["n_shifts"] == 0 else None
            out.append({**base, **got, "state": st})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["t_bar"] = ev.T_BAR
    res["theory_sign"] = res["target"].map(ev.TARGET_SIGN).fillna(1).astype(int)
    res["verdict"] = [ev.verdict(b, t, lo, pc, hi, oo, tb) if np.isfinite(t) else "untested"
                      for b, t, lo, pc, hi, oo, tb in zip(res["beta_bp"], res["t"].astype(float), res["loo_min_t"],
                                                          res["p_chance"], res["loo_max_t"], res["oos_r2"],
                                                          res["t_bar"])]
    return res


def stress_frame(mkt: pd.DataFrame) -> pd.DataFrame:
    """country, month, s: own-country expanding percentile of the spread to Bunds − 0.5."""
    out = []
    t = mkt[(mkt["series"] == "spread_de") & mkt["country"].isin(["FR", "IT"])]
    for c, g in t.groupby("country"):
        v = g.set_index("month")["value"].sort_index().asfreq("MS")
        out.append(pd.DataFrame({"country": c, "month": v.index, "s": (ev.pressure_pct(v) / 10).to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def interaction_tests(rows: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = eh.metrics() if rows is None else rows
    mkt = market.load() if mkt is None else mkt
    rv = ev.states(mkt)
    rv = rv[rv["state"] == "rvol"][["country", "month", "s"]]
    states = {"rvol": (rv, ev.STATES["rvol"]), "stress": (stress_frame(mkt), "Spread to Bunds, own percentile")}
    out = []
    for m, tgt, h, df in _pairs(rows, mkt, BANK, INTER_TARGETS):
        sub = ev._window(df, ev.INTER_WINDOW)
        for state, (st, slabel) in states.items():
            q = sub.merge(st, on=["country", "month"]).dropna(subset=["x", "y", "y12", "s"])
            if q.empty:
                continue
            q = q.assign(xs=q["x"] * q["s"]).sort_values("month")
            got = ev.fit_interaction(q, h)
            if not got:
                continue
            out.append({"test": "interaction", "metric": m, "label": eh.METRICS[m]["label"], "target": tgt,
                        "target_label": ev.TARGETS[tgt][0], "horizon": h, "state": state, "state_label": slabel,
                        "window": ev.INTER_WINDOW, "lag": str(LAG), **got})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["t_bar"] = ev.T_BAR
    res["beta_bp"], res["t"] = res["beta2_bp"], res["t2"]
    res["verdict"] = [ev.verdict(b, t, lo, pc, hi, oo, ev.T_BAR) for b, t, lo, pc, hi, oo in zip(
        res["beta2_bp"], res["t2"], res["loo_min_t"], res["p_chance"], res["loo_max_t"], res["oos_r2"])]
    return res


def nexus_corr(rows: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same-window 12m bank buying vs 12m Δspread_de, by stress tercile (data month, descriptive only)."""
    b = eh.monthly_metrics() if rows is None else rows
    mkt = market.load() if mkt is None else mkt
    b = b[b["metric"] == "bsi_bank_f12m"][["country", "month", "value"]]
    sp = mkt[(mkt["series"] == "spread_de") & mkt["country"].isin(["FR", "IT"])]
    out = []
    st = stress_frame(mkt)
    for c, g in sp.groupby("country"):
        s = g.set_index("month")["value"].sort_index().asfreq("MS")
        d = pd.DataFrame({"month": s.index, "dspread": (s - s.shift(12)).to_numpy()})
        j = b[b["country"] == c].merge(d, on="month").merge(st[st["country"] == c], on=["country", "month"])
        j = j[j["month"] >= "2010-01-01"].dropna()
        for name, k in (("high stress", j["s"] >= ev.TERCILE), ("rest", j["s"] < ev.TERCILE), ("all", j["s"] > -1)):
            jj = j[k]
            out.append({"test": "nexus_corr", "country": c, "state": name, "window": "2010+",
                        "n_obs": len(jj), "r12": jj["value"].corr(jj["dspread"]) if len(jj) > 12 else np.nan,
                        "metric": "bsi_bank_f12m", "target": "spread_de", "horizon": 0})
    return pd.DataFrame(out)


def build() -> pd.DataFrame:
    rows, mkt = eh.metrics(), market.load()
    res = pd.concat([main_tests(rows, mkt), interaction_tests(rows, mkt), nexus_corr(None, mkt)], ignore_index=True)
    res["computed"] = pd.Timestamp.today().normalize()
    return res


def write() -> pd.DataFrame:
    res = build()
    if not res.empty:
        res.to_parquet(OUT, index=False)
    return res


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    res = write()
    main = res[res["test"] == "main"]
    cols = ["metric", "target", "horizon", "window", "beta_bp", "t", "loo_min_t", "p_chance", "oos_r2", "r12",
            "n_obs", "n_countries", "state", "verdict"]
    print(main[main["window"] == "2010+"][cols].round(2).to_string(index=False))
    for w in ev.WINDOWS:
        print(w, ev.chance(main[main["verdict"] != "untested"], w))
        print(main[main["window"] == w].groupby("metric")["verdict"].value_counts().unstack(fill_value=0).to_string())
    inter = res[res["test"] == "interaction"]
    icol = ["state", "metric", "target", "horizon", "beta1_bp", "t1", "beta2_bp", "t2", "loo_min_t", "p_chance",
            "oos_r2", "beta_hi_bp", "t_hi", "beta_lo_bp", "t_lo", "n_obs", "verdict"]
    print(inter[icol].round(2).to_string(index=False))
    print("interactions:", ev.chance(inter, None))
    print(res[res["test"] == "nexus_corr"][["country", "state", "n_obs", "r12"]].round(2).to_string(index=False))
    print(eh.absorption().round(1).to_string(index=False))
