"""Do UK holder-sector gilt metrics (ONS financial accounts, BoE Bankstats, APF) line up with the next
move in gilt prices?

    PYTHONPATH=. .venv/bin/python -m issuance.experiments.uk_holders_test

Same machinery as `issuance.evidence` (no re-implementation): x = `evidence.pressure_pct` of the metric ×
its `uk_holders.METRICS` sign; y / y12 = `evidence.forward_changes`; fit = `evidence._fit` (DK OLS with
the y12 control, circular-shift p, real-time OOS R² from 2016); `evidence.verdict`; `evidence.chance`.
Metrics are stamped at the month their last input became public (`uk_holders.metrics`) and the y window
starts one month later (LAG = 1).

UK only: there is no country to leave out, so LOO is blank and does not block. spread_de is FR/IT only
and drops out. s1030, asw20 and tp10 start in 2016 (`market`), so their 2010–18 rows are short. Rows
that `_fit` cannot score (n < MIN_OBS) are kept with verdict 'untested' and state 'short'; scored rows
with fewer than MIN_SHIFTS rotations carry state 'descriptive' (p_chance NaN, does not block).

(a) main      every metric × `evidence.TARGETS` × h ∈ {3, 12} × windows 2010+, 2010–18, 2019+.
(b) co_move   descriptive: corr of each 4q sector flow (% GDP, at its quarter end) with the same-window
              12m change in y10, s1030, s1020 (y20 − y10, 2000→) and asw20, 2010+. + = the sector bought
              while the target rose (cheapened / steepened).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import evidence as ev
from issuance import market
from issuance import uk_holders as uh
from issuance.paths import CACHE

OUT = CACHE / "uk_holders_evidence.parquet"
OUT_ABSORB = uh.ABSORB_OUT
LAG = 1
CO_METRICS = ("uk_icpf_f4q", "uk_ofi_f4q", "uk_overseas_f4q", "uk_banks_f4q", "uk_apf_f4q")
CO_TARGETS = ("y10", "s1030", "s1020", "asw20")
# Column set of `ecb_holders_evidence.parquet`, so the two load the same way.
COLUMNS = ["test", "metric", "label", "target", "target_label", "window", "lag", "horizon", "countries", "beta_bp",
           "t", "hits", "n_countries", "n_obs", "first", "last", "r12", "loo_min_t", "loo_max_t", "p_chance",
           "n_shifts", "oos_r2", "state", "t_bar", "theory_sign", "verdict", "state_label", "beta1_bp", "t1",
           "beta2_bp", "t2", "beta3_bp", "t3", "beta_hi_bp", "t_hi", "n_hi", "beta_lo_bp", "t_lo", "n_lo", "country",
           "computed"]


def pressure(rows: pd.DataFrame) -> pd.DataFrame:
    """metric, country, month, x (signed by `uh.METRICS`)."""
    out = []
    for (m, c), g in rows[rows["phase"] == "history"].groupby(["metric", "country"]):
        v = g.set_index("month")["value"].sort_index().asfreq("MS") * uh.METRICS[m]["sign"]
        x = ev.pressure_pct(v)
        out.append(pd.DataFrame({"metric": m, "country": c, "month": x.index, "x": x.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def _pairs(rows: pd.DataFrame, mkt: pd.DataFrame, metrics=None, targets=None):
    xz = pressure(rows)
    fw = ev.forward_changes(mkt, lags=(LAG,))
    fw = fw[fw["country"] == "UK"]
    for m in uh.METRICS if metrics is None else metrics:
        xm = xz[xz["metric"] == m].assign(lag=LAG)
        if xm.empty:
            continue
        for tgt in ev.TARGETS if targets is None else targets:
            f = fw[fw["target"] == tgt]
            if f.empty:
                continue
            for h in ev.TARGET_HORIZONS.get(tgt, ev.HORIZONS):
                yield m, tgt, h, xm.merge(f[f["h"] == h].drop(columns="h"), on=["country", "month", "lag"])


def main_tests(rows: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = uh.metrics() if rows is None else rows
    mkt = market.load() if mkt is None else mkt
    out = []
    for m, tgt, h, df in _pairs(rows, mkt):
        for wname in ev.WINDOWS:
            sub = ev._window(df, wname)
            base = {"test": "main", "metric": m, "label": uh.METRICS[m]["label"], "target": tgt,
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


def _uk_series(mkt: pd.DataFrame, name: str) -> pd.Series:
    if name == "s1020":
        return _uk_series(mkt, "y20") - _uk_series(mkt, "y10")
    g = mkt[(mkt["country"] == "UK") & (mkt["series"] == name)]
    return g.set_index("month")["value"].sort_index().asfreq("MS")


def co_move(qm: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None, start: str = "2010-01-01") -> pd.DataFrame:
    """Same-window 4q sector flow vs 12m change in the target at the quarter-end month (descriptive)."""
    qm = uh.quarterly_metrics() if qm is None else qm
    mkt = market.load() if mkt is None else mkt
    out = []
    for tgt in CO_TARGETS:
        y = _uk_series(mkt, tgt)
        d12 = (y - y.shift(12)).rename("d12")
        for m in CO_METRICS:
            g = qm[qm["metric"] == m]
            mo = g["quarter_end"].dt.to_period("M").dt.to_timestamp()
            j = pd.DataFrame({"month": mo.to_numpy(), "value": g["value"].to_numpy()})
            j = j.merge(pd.DataFrame({"month": d12.index, "d12": d12.to_numpy()}), on="month").dropna()
            j = j[j["month"] >= start]
            out.append({"test": "co_move", "metric": m, "label": uh.METRICS[m]["label"], "target": tgt,
                        "window": "2010+", "horizon": 0, "country": "UK", "n_obs": len(j),
                        "r12": j["value"].corr(j["d12"]) if len(j) > 12 else np.nan,
                        "first": j["month"].min() if len(j) else pd.NaT, "last": j["month"].max() if len(j) else pd.NaT})
    return pd.DataFrame(out)


def absorption() -> pd.DataFrame:
    return pd.concat([uh.absorption(), uh.absorption_monthly()], ignore_index=True)


def build() -> pd.DataFrame:
    rows, mkt = uh.metrics(), market.load()
    res = pd.concat([main_tests(rows, mkt), co_move(None, mkt)], ignore_index=True)
    res["state_label"] = np.where(res["test"] == "co_move", "same-window 12m co-movement (descriptive)", None)
    res["computed"] = pd.Timestamp.today().normalize()
    return res.reindex(columns=COLUMNS)


def write() -> pd.DataFrame:
    res = build()
    if not res.empty:
        res.to_parquet(OUT, index=False)
    ab = absorption()
    if not ab.empty:
        ab.to_parquet(OUT_ABSORB, index=False)
    return res


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 800)
    res = write()
    main = res[res["test"] == "main"]
    cols = ["metric", "target", "horizon", "window", "beta_bp", "t", "p_chance", "oos_r2", "r12", "n_obs",
            "state", "verdict"]
    print(main[main["window"] == "2010+"][cols].round(2).to_string(index=False))
    for w in ev.WINDOWS:
        print(w, ev.chance(main[main["verdict"] != "untested"], w))
        print(main[main["window"] == w].groupby("metric")["verdict"].value_counts().unstack(fill_value=0).to_string())
    print(res[res["test"] == "co_move"].pivot_table(index="metric", columns="target", values="r12").round(2).to_string())
    print(absorption().round(1).to_string(index=False))
