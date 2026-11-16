"""Does the IMF investor base (Arslanalp & Tsuda) line up with the next move in price, and does the
elastic-holder share change how much supply moves price (Fang, Hardy & Lewis)?

    PYTHONPATH=. .venv/bin/python -m issuance.experiments.investor_base_test

Same machinery as `issuance.evidence` (no re-implementation): x = `evidence.pressure_pct` of the
metric × its `imf_investor_base.METRICS` sign; y / y12 = `evidence.forward_changes`; fit =
`evidence._fit` (pooled within-country DK OLS with the y12 control, LOO, circular-shift p, real-time
OOS R² from 2016); `evidence.verdict`; `evidence.chance`. Metrics are stamped at the month the IMF
release made them public (`imf_investor_base.public_month`), and the y window starts one month later
(lag 1, `headline.lag_for`'s default).

(a) main   every investor-base metric × `evidence.TARGETS` × h ∈ {3, 12} (return targets their own h)
           × windows 2010+, 2010–18, 2019+.
(b) inter  `evidence.fit_interaction` on supply metrics (priv12, gross12, absorb, absorb_long from
           `headline.panel()`, `evidence.pressure_z` with their headline lags) × state s = own-country
           expanding percentile − 0.5 of the elastic-holder share (dnb + fnb) and, as a check, the
           foreign nonbank share. The state is public at its stamped month, which is never after the
           supply metric's. β2 > 0 = supply moves price more when the elastic share is high ('forward
           link' under `verdict`). The demand-system reading (Koijen–Yogo; FHL's counterfactual:
           without nonbanks a 10% debt rise costs 8.4% instead of 5.8%) predicts β2 < 0, which shows
           as 'wrong sign'. 2010+ only, as `evidence.interactions`.
"""

from __future__ import annotations

import pandas as pd

from issuance import evidence as ev
from issuance import headline, market
from issuance import imf_investor_base as ib
from issuance.paths import CACHE

OUT = CACHE / "investor_base_evidence.parquet"
LAG = 1
SUPPLY = ("priv12", "gross12", "absorb", "absorb_long")
STATES = {"elastic": ("elastic_share", "Elastic-holder share (domestic + foreign nonbanks), IMF"),
          "fnb": ("fnb_share", "Foreign nonbank share, IMF")}


def pressure(rows: pd.DataFrame) -> pd.DataFrame:
    """metric, country, month, x for investor-base metrics (signed by `ib.METRICS`)."""
    out = []
    for (m, c), g in rows[rows["phase"] == "history"].groupby(["metric", "country"]):
        v = g.set_index("month")["value"].sort_index().asfreq("MS") * ib.METRICS[m]["sign"]
        x = ev.pressure_pct(v)
        out.append(pd.DataFrame({"metric": m, "country": c, "month": x.index, "x": x.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def main_tests(rows: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = ib.metrics() if rows is None else rows
    mkt = market.load() if mkt is None else mkt
    xz = pressure(rows)
    fw = ev.forward_changes(mkt, lags=(LAG,))
    out = []
    for m in ib.METRICS:
        xm = xz[xz["metric"] == m].assign(lag=LAG)
        if xm.empty:
            continue
        for tgt in ev.TARGETS:
            for h in ev.TARGET_HORIZONS.get(tgt, ev.HORIZONS):
                df = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"),
                              on=["country", "month", "lag"])
                for wname in ev.WINDOWS:
                    sub = ev._window(df, wname)
                    got = ev._fit(sub, h)
                    if not got:
                        continue
                    got["r12"] *= ev.CO_SIGN.get(tgt, 1)
                    out.append({"test": "main", "metric": m, "label": ib.METRICS[m]["label"], "target": tgt,
                                "target_label": ev.TARGETS[tgt][0], "window": wname, "lag": str(LAG), "horizon": h,
                                **got, "countries": " ".join(sorted(sub["country"].unique()))})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["t_bar"] = ev.T_BAR
    res["theory_sign"] = res["target"].map(ev.TARGET_SIGN).fillna(1).astype(int)
    res["verdict"] = [ev.verdict(b, t, lo, pc, hi, oo, tb) for b, t, lo, pc, hi, oo, tb in zip(
        res["beta_bp"], res["t"], res["loo_min_t"], res["p_chance"], res["loo_max_t"], res["oos_r2"], res["t_bar"])]
    return res


def state_frame(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """country, month, s: own-country expanding percentile − 0.5 of the public state value."""
    out = []
    for c, g in rows[rows["metric"] == metric].groupby("country"):
        v = g.set_index("month")["value"].sort_index().asfreq("MS")
        out.append(pd.DataFrame({"country": c, "month": v.index, "s": (ev.pressure_pct(v) / 10).to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def interaction_tests(rows: pd.DataFrame | None = None, hp: pd.DataFrame | None = None,
                      mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = ib.metrics() if rows is None else rows
    hp = headline.panel() if hp is None else hp
    mkt = market.load() if mkt is None else mkt
    xz = ev.pressure_z(hp[hp["metric"].isin(SUPPLY)])
    fw = ev.forward_changes(mkt)
    out = []
    for state, (sm, slabel) in STATES.items():
        st = state_frame(rows, sm)
        for m in SUPPLY:
            xm = xz[xz["metric"] == m].copy()
            if xm.empty:
                continue
            xm["lag"] = [headline.lag_for(m, c) for c in xm["country"]]
            lags = "/".join(str(v) for v in sorted(set(xm["lag"])))
            for tgt in ev.TARGETS:
                for h in ev.TARGET_HORIZONS.get(tgt, ev.HORIZONS):
                    df = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"),
                                  on=["country", "month", "lag"])
                    q = ev._window(df, ev.INTER_WINDOW).merge(st, on=["country", "month"])
                    q = q.dropna(subset=["x", "y", "y12", "s"])
                    if q.empty:
                        continue
                    q = q.assign(xs=q["x"] * q["s"]).sort_values("month")
                    got = ev.fit_interaction(q, h)
                    if not got:
                        continue
                    out.append({"test": "interaction", "metric": m, "label": headline.METRICS[m]["label"],
                                "target": tgt, "target_label": ev.TARGETS[tgt][0], "horizon": h, "state": state,
                                "state_label": slabel, "window": ev.INTER_WINDOW, "lag": lags, **got})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["t_bar"] = ev.T_BAR
    res["beta_bp"], res["t"] = res["beta2_bp"], res["t2"]
    res["verdict"] = [ev.verdict(b, t, lo, pc, hi, oo, ev.T_BAR) for b, t, lo, pc, hi, oo in zip(
        res["beta2_bp"], res["t2"], res["loo_min_t"], res["p_chance"], res["loo_max_t"], res["oos_r2"])]
    return res


def build() -> pd.DataFrame:
    rows, mkt = ib.metrics(), market.load()
    res = pd.concat([main_tests(rows, mkt), interaction_tests(rows, None, mkt)], ignore_index=True)
    res["vintage"] = ib.vintage()
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
            "n_countries", "verdict"]
    print(main[main["window"] == "2010+"][cols].round(2).to_string(index=False))
    for w in ev.WINDOWS:
        print(w, ev.chance(main, w))
        print(main[main["window"] == w].groupby("metric")["verdict"].value_counts().unstack(fill_value=0).to_string())
    inter = res[res["test"] == "interaction"]
    icol = ["state", "metric", "target", "horizon", "beta1_bp", "t1", "beta2_bp", "t2", "loo_min_t", "loo_max_t",
            "p_chance", "oos_r2", "beta_hi_bp", "beta_lo_bp", "verdict"]
    print(inter[icol].round(2).to_string(index=False))
    for s in STATES:
        print(s, ev.chance(inter[inter["state"] == s], None))
