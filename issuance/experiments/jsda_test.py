"""Does JSDA investor-type JGB trading line up with the next move in JGB prices? Who absorbed the
super-long supply in 2023–26?

    PYTHONPATH=. .venv/bin/python -m issuance.experiments.jsda_test

Scope: Japan only, one country. Same machinery as `issuance.evidence` (no re-implementation):
x = `evidence.pressure_pct` of the metric × its `fetch_jsda.METRICS` sign (net buying is demand, so
sign −1: + x = less buying); y / y12 = `evidence.forward_changes` for JP; fit = `evidence._fit`
(DK OLS with the y12 control, circular-shift p, real-time OOS R² from 2016); `evidence.verdict`;
`evidence.chance`. With one country there is no leave-one-out (loo_min_t is NaN and does not
block) and the country fixed effect is just a constant. Metrics are stamped at the month after the
data month (JSDA publishes mid-month, `fetch_jsda.public_month`), and the y window starts one month
later still (LAG = 1), so the price window opens two months after the data month.

Targets: y10, rel10, s1030 (JP 30y − 10y), rx10_3, rx10_12, tp10 (JP: 10y − 2y), h ∈ {3, 12} (return
targets their own h), windows 2010+, 2010–18, 2019+.

Descriptive:
  absorption()  super-long (20/30/40y) net supply by calendar year — gross issuance net of buybacks
                (`flow_isin`) less redemptions (`lines`), BoJ Δholdings (`boj_holdings`), JSDA net
                purchases by investor group, and the residual (direct auction buyers, dealer
                inventory, and anything JSDA does not see).
  mof_compare() JSDA foreigners vs MoF weekly 'non-residents' net purchases of Japanese debt
                securities' summed to months: coupon JGBs vs long-term debt, bills vs short-term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import evidence as ev
from issuance import fetch_jsda as js
from issuance import market
from issuance.paths import CACHE

OUT = CACHE / "jsda_evidence.parquet"
OUT_ABSORB = CACHE / "jsda_absorption.parquet"
LAG = 1
TARGETS = ("y10", "rel10", "s1030", "rx10_3", "rx10_12", "tp10")
SL_PRODUCTS = ("20y", "30y", "40y")
GROUPS = {
    "insurers": ("insurers",),
    "trust_banks": ("trust_banks",),
    "banks": ("city_banks", "regional_banks", "regional2", "shinkin", "agri_fin"),
    "inv_trusts": ("inv_trusts",),
    "foreigners": ("foreigners",),
    "other_domestic": ("other_fin", "mutual_aid", "business_corps", "other_corps", "individuals"),
}


def pressure(rows: pd.DataFrame) -> pd.DataFrame:
    """metric, country, month, x (signed by `js.METRICS`)."""
    out = []
    for (m, c), g in rows[rows["phase"] == "history"].groupby(["metric", "country"]):
        v = g.set_index("month")["value"].sort_index().asfreq("MS") * js.METRICS[m]["sign"]
        x = ev.pressure_pct(v)
        out.append(pd.DataFrame({"metric": m, "country": c, "month": x.index, "x": x.to_numpy()}))
    return pd.concat(out, ignore_index=True).dropna()


def main_tests(rows: pd.DataFrame | None = None, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = js.metrics() if rows is None else rows
    mkt = market.load() if mkt is None else mkt
    mkt = mkt[mkt["country"] == "JP"]
    xz = pressure(rows)
    fw = ev.forward_changes(mkt, lags=(LAG,))
    out = []
    for m in js.METRICS:
        xm = xz[xz["metric"] == m].assign(lag=LAG)
        if xm.empty:
            continue
        for tgt in TARGETS:
            for h in ev.TARGET_HORIZONS.get(tgt, ev.HORIZONS):
                df = xm.merge(fw[(fw["target"] == tgt) & (fw["h"] == h)].drop(columns="h"),
                              on=["country", "month", "lag"])
                for wname in ev.WINDOWS:
                    sub = ev._window(df, wname)
                    got = ev._fit(sub, h)
                    if not got:
                        continue
                    got["r12"] *= ev.CO_SIGN.get(tgt, 1)
                    out.append({"test": "main", "metric": m, "label": js.METRICS[m]["label"], "target": tgt,
                                "target_label": ev.TARGETS[tgt][0], "window": wname, "lag": str(LAG), "horizon": h,
                                **got, "countries": " ".join(sorted(sub["country"].unique()))})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["t_bar"] = ev.T_BAR
    res["theory_sign"] = res["target"].map(ev.TARGET_SIGN).fillna(1).astype(int)
    res["verdict"] = [ev.verdict(b, t, lo, pc, hi, oo, tb) for b, t, lo, pc, hi, oo, tb in zip(
        res["beta_bp"], res["t"], res["loo_min_t"], res["p_chance"], res["loo_max_t"], res["oos_r2"], res["t_bar"])]
    res["scope"] = "JP only (single country: no LOO; country effect = constant)"
    return res


# ---------------------------------------------------------------- descriptive
def _year_end_boj(products=SL_PRODUCTS) -> pd.Series:
    b = pd.read_parquet(CACHE / "boj_holdings.parquet")
    b = b[b["product"].isin(list(products))]
    tot = b.groupby("date")["face_mn"].sum().sort_index() / 1000
    return tot.groupby(tot.index.year).last()


def absorption(p: pd.DataFrame | None = None, years=(2023, 2024, 2025, 2026)) -> pd.DataFrame:
    """Super-long (20/30/40y) net supply and who took it, ¥tn per calendar year (the last year to the
    latest JSDA month)."""
    p = js.panel() if p is None else p
    last = p["month"].max()
    f = pd.read_parquet(CACHE / "flow_isin.parquet")
    f = f[(f["country"] == "JP") & f["isin"].str.match(r"^JP(20y|30y|40y)#")]
    f["month"] = pd.to_datetime(f["month"])
    f = f[f["month"] <= last]
    gross = (f["gross_mn"] - f["buyback_mn"].fillna(0)).groupby(f["month"].dt.year).sum() / 1e6
    ln = pd.read_parquet(CACHE / "lines.parquet")
    ln = ln[(ln["country"] == "JP") & ln["isin"].str.match(r"^JP(20y|30y|40y)#")]
    mat = pd.to_datetime(ln["maturity_dt"])
    ln = ln[mat <= last]
    red = ln["stock_mn"].where(ln["stock_mn"] > 0, ln["peak_mn"]).groupby(mat[mat <= last].dt.year).sum() / 1e6
    boj = _year_end_boj()
    sl = p[p["maturity"] == "super_long"]
    rows = []
    for y in years:
        r = {"year": f"{y}" if y < last.year else f"{y} (to {last:%b})"}
        r["gross_issuance"] = float(gross.get(y, 0.0))
        r["redemptions"] = float(red.get(y, 0.0))
        r["net_supply"] = r["gross_issuance"] - r["redemptions"]
        r["boj_change"] = float(boj.get(y, np.nan) - boj.get(y - 1, np.nan)) / 1000
        sy = sl[sl["month"].dt.year == y]
        for g, invs in GROUPS.items():
            r[g] = float(sy[sy["investor"].isin(list(invs))]["net_purchase_jpy_bn"].sum()) / 1000
        r["jsda_sum"] = sum(r[g] for g in GROUPS)
        r["residual"] = r["net_supply"] - r["boj_change"] - r["jsda_sum"]
        rows.append(r)
    return pd.DataFrame(rows)


def mof_compare(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """Correlation of JSDA foreigners' net purchases with MoF non-resident net purchases (monthly and
    12m sums), 2005+. MoF weeks are assigned to the month of their week-end date."""
    p = js.panel() if p is None else p
    w = js.wide(p)
    fx = pd.DataFrame({
        "jsda_coupon": w[("foreigners", "total_jgb")] - w[("foreigners", "tbills")].fillna(0),
        "jsda_bills": w[("foreigners", "tbills")],
        "jsda_total": w[("foreigners", "total_jgb")],
    })
    mw = pd.read_parquet(CACHE / "jp_flows_weekly.parquet")
    mw["month"] = pd.to_datetime(mw["week_end"]).dt.to_period("M").dt.to_timestamp()
    mm = mw.groupby("month")[["l_lt_net", "l_st_net"]].sum()
    mm["l_debt_net"] = mm["l_lt_net"] + mm["l_st_net"]
    # drop the partial first/last MoF months
    mm = mm.iloc[1:-1]
    j = fx.join(mm, how="inner").dropna()
    rows = []
    for a, b, lab in (("jsda_coupon", "l_lt_net", "coupon JGBs vs MoF long-term debt"),
                      ("jsda_bills", "l_st_net", "T-bills vs MoF short-term debt"),
                      ("jsda_total", "l_debt_net", "all JGBs vs MoF all debt")):
        for k, kl in ((1, "monthly"), (3, "3m sum"), (12, "12m sum")):
            x, y = j[a].rolling(k).sum(), j[b].rolling(k).sum()
            ok = x.notna() & y.notna()
            for wname, lo, hi in (("2005+", "2005-01-01", None), ("2019+", "2019-01-01", None)):
                sel = ok & (j.index >= lo)
                rows.append({"pair": lab, "freq": kl, "window": wname, "n": int(sel.sum()),
                             "corr": float(np.corrcoef(x[sel], y[sel])[0, 1]),
                             "jsda_mean_bn": float(j.loc[sel, a].mean()), "mof_mean_bn": float(j.loc[sel, b].mean())})
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    rows, mkt = js.metrics(), market.load()
    res = main_tests(rows, mkt)
    res["vintage"] = js.panel()["month"].max()
    res["computed"] = pd.Timestamp.today().normalize()
    return res


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
    pd.set_option("display.max_rows", 500)
    res = write()
    cols = ["metric", "target", "horizon", "window", "beta_bp", "t", "p_chance", "n_shifts", "oos_r2", "r12",
            "n_obs", "verdict"]
    print(res[res["window"] == "2010+"][cols].round(2).to_string(index=False))
    for w in ev.WINDOWS:
        print(w, ev.chance(res, w))
        print(res[res["window"] == w].groupby("metric")["verdict"].value_counts().unstack(fill_value=0).to_string())
    print(absorption().round(2).to_string(index=False))
    print(mof_compare().round(2).to_string(index=False))
