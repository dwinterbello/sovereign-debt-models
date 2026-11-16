"""Long-end private supply (15y+) and next-12m 10s30s: the measure behind dalio-note section 5, and its audit.

Checks, per market:
  dv01      the shipped measure (`rv_supply_fc.dv01_long`).
  dv01_jpfix  the same with the forward JP rows net of BoJ long-end purchases (history nets central-bank flow by
            bucket; the forward extension takes gross tenor DV01 and nets official flow only in the total).
  face      private face at 15y+ (gross less central-bank net purchases in the same buckets), % of local GDP. DV01 per
            unit of face is ~30% higher at a 0.5% yield than at 3%, so a DV01 average carries the ZIRP years.
For each: next-12m level vs its 2012-onward mean, own-market slope of next-12m Δ10s30s (2y and realised-vol
controls, HAC), pooled and ex-JP slopes (country FE, Driscoll-Kraay), and the pre/post-2020 halves.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.long_end
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance.paths import CACHE
from issuance.rv_metrics import LONG
from sovereign_cash._vendor import dk_ols
from sovereign_cash.debt import signal as S

CC6 = ["us", "uk", "de", "fr", "it", "jp"]
HIST_END = pd.Timestamp("2026-09-01")


def _gdp() -> pd.DataFrame:
    a = pd.read_parquet(S.DERIVED / "debt_annual.parquet")[["country", "year", "gdp_usd_bn", "gdp_lcu_bn"]]
    return a


def measures() -> pd.DataFrame:
    s = pd.read_parquet(CACHE / "rv_supply_fc.parquet")[["country", "month", "dv01_long", "official_dv01_usd"]]
    s["country"] = s["country"].str.lower()
    s["month"] = pd.to_datetime(s["month"])

    # history: gross long coupon DV01 less the netted series = official long DV01; JP share of its total official DV01
    ten = pd.read_parquet(CACHE / "dv01_tenor.parquet")
    ten["month"] = pd.to_datetime(ten["month"])
    ten = ten[ten["sec_type"].astype(str) != "Bill"]
    ten["country"] = ten["country"].str.lower()
    gl = ten[ten["tenor_bucket"].isin(LONG)].groupby(["country", "month"])["dv01_usd_mn"].sum().rename("gross_long")
    j = s[s["country"] == "jp"].set_index("month").join(gl.xs("jp"), how="left")
    win = j.loc[HIST_END - pd.DateOffset(months=11):HIST_END]
    long_share = float((win["gross_long"] - win["dv01_long"]).sum() / win["official_dv01_usd"].sum())
    fwd = (s["country"] == "jp") & (s["month"] > HIST_END)
    s["dv01_jpfix"] = s["dv01_long"]
    s.loc[fwd, "dv01_jpfix"] = s.loc[fwd, "dv01_long"] - long_share * s.loc[fwd, "official_dv01_usd"]

    # face: gross 15y+ face (history + forecast) less central-bank net purchases in the same buckets
    fc = pd.read_parquet(CACHE / "dv01_tenor_fc.parquet")
    fc["country"] = fc["country"].str.lower()
    fc["month"] = pd.to_datetime(fc["month"])
    gface = fc[fc["tenor_bucket"].isin(LONG)].groupby(["country", "month"])["gross_mn"].sum().rename("gface")
    o = pd.read_parquet(CACHE / "official_isin_flow.parquet")
    o["country"] = o["country"].str.lower()
    o["month"] = pd.to_datetime(o["month"])
    oface = o[o["tenor_bucket"].isin(LONG)].groupby(["country", "month"])["official_mn"].sum().rename("oface")
    # Fed add-ons are in the SOMA change but already out of gross (public take): take them out of official too
    addon = ten[ten["tenor_bucket"].isin(LONG)].groupby(["country", "month"])["official_take_mn"].sum().rename("addon")
    f = pd.concat([gface, oface, addon], axis=1).fillna(0.0).reset_index()
    f["oface"] = f["oface"] - f["addon"]
    # forward JP: BoJ plan × trailing-12m long share of purchase face
    jo = o[(o["country"] == "jp") & (o["month"] > HIST_END - pd.DateOffset(months=12)) & (o["month"] <= HIST_END)]
    pos = jo[jo["official_mn"] > 0]
    face_share = float(pos.loc[pos["tenor_bucket"].isin(LONG), "official_mn"].sum() / pos["official_mn"].sum())
    plan = pd.read_parquet(CACHE / "boj_purchase_plan.parquet")
    plan["month"] = pd.to_datetime(plan["month"])
    pm = plan.set_index("month")["planned_mn"]
    m = (f["country"] == "jp") & (f["month"] > HIST_END)
    f.loc[m, "oface"] = [face_share * float(pm.get(x, pm.iloc[-1])) for x in f.loc[m, "month"]]
    f.loc[(f["country"] != "jp") & (f["month"] > HIST_END), "oface"] = 0.0
    f["pface"] = f["gface"] - f["oface"]

    out = s.merge(f[["country", "month", "pface"]], on=["country", "month"], how="left")
    # a month with no 15y+ auction and no central-bank flow there is zero supply, not missing
    last = fc.groupby("country")["month"].max()
    out["pface"] = out["pface"].where(out["pface"].notna() | (out["month"] > out["country"].map(last)), 0.0)
    out["year"] = out["month"].dt.year
    out = out.merge(_gdp(), on=["country", "year"], how="left").sort_values(["country", "month"])
    out["dv01"] = out["dv01_long"] / (out["gdp_usd_bn"] / 1000)
    out["dv01_jpfix"] = out["dv01_jpfix"] / (out["gdp_usd_bn"] / 1000)
    out["face"] = 100 * (out["pface"] / 1000) / out["gdp_lcu_bn"]
    for c in ("dv01", "dv01_jpfix", "face"):
        out[c + "12"] = out.groupby("country")[c].transform(lambda x: x.rolling(12, min_periods=12).sum())
        out[c + "_fw"] = out.groupby("country")[c + "12"].shift(-12)
    out.attrs.update(long_share=long_share, face_share=face_share)
    return out


def panel(m: pd.DataFrame) -> pd.DataFrame:
    p = S.panel()
    mk = pd.read_parquet(S.MARKET)
    mk["month"] = pd.to_datetime(mk["month"])
    w = mk[mk["series"].isin(["y2", "rvol10"])].pivot_table(index=["country", "month"], columns="series", values="value").reset_index()
    w["country"] = w["country"].str.lower()
    w = w.sort_values(["country", "month"])
    for c in ("y2", "rvol10"):
        w["d_" + c] = w.groupby("country")[c].shift(-12) - w[c]
    w["d_y2"] *= 100
    p = p.merge(w[["country", "month", "d_y2", "d_rvol10"]], on=["country", "month"], how="left")
    p = p.merge(m[["country", "month", "dv01_fw", "dv01_jpfix_fw", "face_fw"]], on=["country", "month"], how="left")
    return p[p["month"] >= "2012-01-01"]


def pooled(p, x, a="2012-01-01", b="2026-12-01"):
    cols = [x, "d_y2", "d_rvol10"]
    q = p[(p["month"] >= a) & (p["month"] <= b)].dropna(subset=["d_s1030", *cols])
    r = dk_ols(S._demean(q, ["d_s1030", *cols], "country"), "d_s1030", cols, lags=12)
    return float(r["beta"][x]), float(r["t"][x]), int(len(q))


def own(p, x, cc):
    z = p[p["country"] == cc].dropna(subset=["d_s1030", x, "d_y2", "d_rvol10"])
    if len(z) < 30:
        return np.nan, np.nan
    r = sm.OLS(z["d_s1030"], sm.add_constant(z[[x, "d_y2", "d_rvol10"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return float(r.params[x]), float(r.tvalues[x])


MODELS = {"all": ("priv_fw", "priv_tr", "all-tenor private net supply"), "long": ("face_fw", "face_tr", "15y+ private supply")}
EUR_CC = ("de", "fr", "it")
NO_ALL = ("jp",)  # all-tenor JP is BoJ runoff inside 10y; its own slope is ~1 (t 0.9) and its 2012→ mean is QE-distorted


def robust(p: pd.DataFrame, x: str, xtr: str) -> dict:
    """Pooled (country FE, DK), ex-JP, halves, two-way FE, worst leave-one-out, Januaries only, trailing (known) supply,
    and expanding-window OOS R² of the pooled fit against the controls-only fit."""
    cols = [x, "d_y2", "d_rvol10"]
    q = p.dropna(subset=["d_s1030", *cols]).copy()
    out = {"pooled": pooled(p, x)[:2], "ex_jp": pooled(p[p["country"] != "jp"], x)[:2],
           "pre": pooled(p, x, b="2019-12-01")[:2], "post": pooled(p, x, a="2020-01-01")[:2],
           "pre_xjp": pooled(p[p["country"] != "jp"], x, b="2019-12-01")[:2]}
    tw = S._demean(S._demean(q, ["d_s1030", *cols], "country"), ["d_s1030", *cols], "month")
    r = dk_ols(tw, "d_s1030", cols, lags=12)
    out["two_way"] = (float(r["beta"][x]), float(r["t"][x]))
    out["loo_min_t"] = min(pooled(p[p["country"] != cc], x)[1] for cc in CC6)
    jan = S._demean(q[q["month"].dt.month == 1], ["d_s1030", *cols], "country")
    r = sm.OLS(jan["d_s1030"], jan[cols]).fit(cov_type="cluster", cov_kwds={"groups": q.loc[jan.index, "month"]})
    out["jan"] = (float(r.params[x]), float(r.tvalues[x]))
    out["trailing"] = pooled(p, xtr)[:2]
    err_f, err_c = [], []
    for T in pd.date_range("2018-01-01", q["month"].max(), freq="MS"):
        tr_ = q[q["month"] <= T - pd.DateOffset(months=12)]
        te = q[q["month"] == T]
        if len(tr_) < 120 or te.empty:
            continue
        mu = tr_.groupby("country")[["d_s1030", *cols]].mean()
        d = tr_.set_index("country")[["d_s1030", *cols]] - mu
        bf = np.linalg.lstsq(d[cols].to_numpy(), d["d_s1030"].to_numpy(), rcond=None)[0]
        bc = np.linalg.lstsq(d[cols[1:]].to_numpy(), d["d_s1030"].to_numpy(), rcond=None)[0]
        t_ = te.set_index("country")
        ok = t_.index.intersection(mu.index)
        dt = t_.loc[ok, cols] - mu.loc[ok, cols]
        y = t_.loc[ok, "d_s1030"] - mu.loc[ok, "d_s1030"]
        err_f += list((y - dt.to_numpy() @ bf) ** 2)
        err_c += list((y - dt[cols[1:]].to_numpy() @ bc) ** 2)
    out["oos_r2"] = 1 - float(np.sum(err_f) / np.sum(err_c))
    return out


def drifts() -> dict:
    """Both regressions side by side: per market next-12m supply vs its 2012→ mean and the implied 12m Δ10s30s
    (pooled slope × deviation), plus EUR = DE/FR/IT weighted by 2025 USD GDP. Returns {"T": table, "R": robustness}."""
    m = measures()
    p = panel(m)
    p = p.merge(m[["country", "month", "face12"]].rename(columns={"face12": "face_tr"}), on=["country", "month"], how="left")
    h = pd.read_parquet(S.DERIVED / "debt_history.parquet")
    h["month"] = pd.to_datetime(h["month"])
    gdp = _gdp().query("year == 2025").set_index("country")["gdp_usd_bn"]
    ew = gdp[list(EUR_CC)] / gdp[list(EUR_CC)].sum()
    R = {k: robust(p, x, xtr) for k, (x, xtr, _) in MODELS.items()}
    rows = {}
    for cc in CC6:
        x = h[h["cc"] == cc].set_index("month").sort_index()
        last = x[~x["is_forecast"].astype(bool)].index.max()
        g = m[m["country"] == cc].set_index("month")["face"]
        rows[cc] = {"all_fwd": float(x["private12_pct_gdp"].get(last + pd.DateOffset(months=12))),
                    "all_mean": float(p.loc[p["country"] == cc, "priv_fw"].mean()),
                    "long_fwd": float(g.loc[HIST_END + pd.DateOffset(months=1):HIST_END + pd.DateOffset(months=12)].sum()),
                    "long_mean": float(p.loc[p["country"] == cc, "face_fw"].mean())}
    T = pd.DataFrame(rows).T
    for k in MODELS:
        T[k + "_dev"] = T[k + "_fwd"] - T[k + "_mean"]
        T[k + "_bp"] = R[k]["pooled"][0] * T[k + "_dev"]
    W = wedge(p)
    bl, bs = W["all"]["long"][0], W["all"]["short"][0]
    T["joint_long"] = bl * T["long_dev"]
    T["joint_short"] = bs * (T["all_dev"] - T["long_dev"])
    T["joint_bp"] = T["joint_long"] + T["joint_short"]
    T.loc[list(NO_ALL), ["all_bp", "joint_short", "joint_bp"]] = np.nan
    T.loc["eur"] = {c: float((T.loc[list(EUR_CC), c] * ew).sum()) for c in T.columns}
    return {"T": T, "R": R, "ew": ew, "W": W}


def wedge(p: pd.DataFrame) -> dict:
    """Long vs short supply in one fit: Δ10s30s on 15y+ supply and supply inside 15y (all-tenor net − 15y+; redemptions
    are all short at maturity), same controls and FE. The wedge is the 15y+ coefficient in a joint fit with all-tenor
    (= long − short per pp). OOS R² of the joint fit against controls only, as in `robust`."""
    q = p.assign(short_fw=p["priv_fw"] - p["face_fw"])
    cols = ["face_fw", "short_fw", "d_y2", "d_rvol10"]
    out = {}
    for lab, z in (("all", q), ("xjp", q[q["country"] != "jp"])):
        z = z.dropna(subset=["d_s1030", *cols])
        r = dk_ols(S._demean(z, ["d_s1030", *cols], "country"), "d_s1030", cols, lags=12)
        j = dk_ols(S._demean(z, ["d_s1030", "priv_fw", *cols[1:]], "country").assign(
            face_fw=S._demean(z, ["face_fw"], "country")["face_fw"]), "d_s1030", ["priv_fw", "face_fw", "d_y2", "d_rvol10"], lags=12)
        out[lab] = {"long": (float(r["beta"]["face_fw"]), float(r["t"]["face_fw"])),
                    "short": (float(r["beta"]["short_fw"]), float(r["t"]["short_fw"])),
                    "wedge_t": float(j["t"]["face_fw"]), "oos_r2": _oos(z, ["face_fw", "short_fw"])}
    return out


def _oos(q: pd.DataFrame, xs: list[str]) -> float:
    cols = [*xs, "d_y2", "d_rvol10"]
    e, ec = [], []
    for T in pd.date_range("2018-01-01", q["month"].max(), freq="MS"):
        tr = q[q["month"] <= T - pd.DateOffset(months=12)]
        te = q[q["month"] == T]
        if len(tr) < 120 or te.empty:
            continue
        mu = tr.groupby("country")[["d_s1030", *cols]].mean()
        d = tr.set_index("country")[["d_s1030", *cols]] - mu
        b = np.linalg.lstsq(d[cols].to_numpy(), d["d_s1030"].to_numpy(), rcond=None)[0]
        bc = np.linalg.lstsq(d[cols[-2:]].to_numpy(), d["d_s1030"].to_numpy(), rcond=None)[0]
        t_ = te.set_index("country")
        ok = t_.index.intersection(mu.index)
        dt = t_.loc[ok, cols] - mu.loc[ok, cols]
        y = t_.loc[ok, "d_s1030"] - mu.loc[ok, "d_s1030"]
        e += list((y - dt.to_numpy() @ b) ** 2)
        ec += list((y - dt[cols[-2:]].to_numpy() @ bc) ** 2)
    return 1 - float(np.sum(e) / np.sum(ec))


def main() -> None:
    m = measures()
    p = panel(m)
    print(f"JP long share of BoJ official DV01 (trailing 12m): {m.attrs['long_share']:.2f}; of purchase face: {m.attrs['face_share']:.2f}")
    for x in ("dv01", "dv01_jpfix", "face"):
        print(f"\n== {x} ==")
        rows = []
        for cc in CC6:
            g = m[m["country"] == cc].set_index("month")[x]
            fwd = float(g.loc[HIST_END + pd.DateOffset(months=1):HIST_END + pd.DateOffset(months=12)].sum())
            trail = float(g.loc[HIST_END - pd.DateOffset(months=11):HIST_END].sum())
            mean = float(p.loc[p["country"] == cc, x + "_fw"].mean())
            b, t = own(p, x + "_fw", cc)
            rows.append({"cc": cc, "trail": trail, "fwd": fwd, "mean": mean, "dev": fwd - mean, "own_b": b, "own_t": t,
                         "own_bp": b * (fwd - mean)})
        print(pd.DataFrame(rows).set_index("cc").round(2).to_string())
        xjp = p[p["country"] != "jp"]
        for lab, q, a, b in (("pooled", p, "2012-01-01", "2026-12-01"), ("ex-JP", xjp, "2012-01-01", "2026-12-01"),
                             ("pre-2020", p, "2012-01-01", "2019-12-01"), ("post-2020", p, "2020-01-01", "2026-12-01"),
                             ("ex-JP post-2020", xjp, "2020-01-01", "2026-12-01")):
            bb, tt, n = pooled(q, x + "_fw", a, b)
            print(f"  {lab:16s} b {bb:+.3f}  t {tt:+.2f}  n {n}")
    d = drifts()
    print("\n", d["T"].round(2).to_string())
    print("\n== wedge ==", d["W"])
    for k, r in d["R"].items():
        print(f"\n== {k} ==")
        for kk, v in r.items():
            print(f"  {kk:10s} {v}")


if __name__ == "__main__":
    main()
