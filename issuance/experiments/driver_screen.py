"""One row per country × investor × tenor. No average, no dynamic weights.

Screens that already exist and mostly fail the evidence gate (do not re-average them):
  us_demand_evidence, ecb_holders_evidence, uk_holders_evidence.

This file adds the tenor cut those files do not have:
  JP  JSDA outright net purchases, each investor × maturity sleeve
  US  Treasury auction allotment share, each investor class × int/long

ECB SHSS public API (probed 2026-10-19) publishes holder × issuer × original maturity
only as S (≤1y) / L (>1y) / T. There is no 2–10y vs >10y holder split. The S/L
transactions file is cache/ecb_holders/shss_sector_maturity_F.csv and starts 2021-Q1,
too short to gate.

  PYTHONPATH=. .venv/bin/python -m issuance.experiments.driver_screen
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance.experiments.flow_score_test import target_panel
from issuance.paths import CACHE

OUT = CACHE / "driver_screen.parquet"
JSDA_SLEEVE = {"medium": "int", "long": "long10", "super_long": "ultra", "tbills": "bills"}


def _z(s: pd.Series) -> pd.Series:
    mu = s.rolling(36, min_periods=24).mean()
    sd = s.rolling(36, min_periods=24).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-4, 4)


def _fit(df: pd.DataFrame) -> dict | None:
    sub = df[["month", "x", "f3l2"]].dropna()
    if len(sub) < 36:
        return None
    q = sub[sub["month"].dt.month % 3 == 0]
    if len(q) < 16 or q["x"].std() == 0:
        return None
    y = q["f3l2"] - q["f3l2"].mean()
    x = q["x"] - q["x"].mean()
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {
        "n": int(len(q)),
        "r": round(float(np.corrcoef(q["x"], q["f3l2"])[0, 1]), 3),
        "beta_bp": round(float(fit.params["x"]) * 100, 2),
        "t": round(float(fit.tvalues["x"]), 2),
    }


def _jsda(tp: pd.DataFrame) -> pd.DataFrame:
    js = pd.read_parquet(CACHE / "jsda_investor.parquet")
    js = js[js["investor"].astype(str).ne("total")].copy()
    js["month"] = pd.to_datetime(js["month"]).dt.to_period("M").dt.to_timestamp()
    js["sleeve"] = js["maturity"].map(JSDA_SLEEVE)
    jp = tp[(tp["country"] == "JP") & (tp["target"] == "lvl10")][["month", "f3l2"]]
    rows = []
    for (inv, sl), g in js.dropna(subset=["sleeve"]).groupby(["investor", "sleeve"]):
        flow = g.groupby("month")["net_purchase_jpy_bn"].sum().sort_index().rolling(3, min_periods=3).sum()
        df = pd.DataFrame({"month": _z(flow).index, "x": _z(flow).to_numpy()}).merge(jp, on="month")
        df = df[df["month"] >= "2010-01-01"]
        got = _fit(df)
        if got:
            rows.append({"country": "JP", "driver": inv, "sleeve": sl, "target": "lvl10",
                         "source": "JSDA OTC outright", **got})
    return pd.DataFrame(rows)


def _us_allot(tp: pd.DataFrame) -> pd.DataFrame:
    al = pd.read_parquet(CACHE / "us_allotments.parquet")
    al["term_years"] = pd.to_numeric(al["term_years"], errors="coerce")
    al = al[al["term_years"] > 2].copy()
    al["sleeve"] = np.where(al["term_years"] > 10, "long", "int")
    al["month"] = pd.to_datetime(al["issue_date"]).dt.to_period("M").dt.to_timestamp()
    rows = []
    for tgt in ("lvl10", "s1030"):
        y = tp[(tp["country"] == "US") & (tp["target"] == tgt)][["month", "f3l2"]]
        for (cls, sl), g in al.groupby(["investor_class", "sleeve"]):
            denom = al[al["sleeve"] == sl].groupby("month")["amount_bn"].sum()
            share = g.groupby("month")["amount_bn"].sum() / denom.replace(0, np.nan)
            zz = _z(share.rolling(3, min_periods=2).mean())
            df = pd.DataFrame({"month": zz.index, "x": zz.to_numpy()}).merge(y, on="month")
            df = df[df["month"] >= "2010-01-01"]
            got = _fit(df)
            if got:
                rows.append({"country": "US", "driver": cls, "sleeve": sl, "target": tgt,
                             "source": "Treasury investor-class allotments", **got})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    tp = target_panel()
    out = pd.concat([_jsda(tp), _us_allot(tp)], ignore_index=True)
    out["ship"] = (out["t"] <= -2.0)
    out.to_parquet(OUT, index=False)
    return out


if __name__ == "__main__":
    o = run()
    print(f"wrote {OUT}  {len(o)} drivers  ship {int(o['ship'].sum())}")
    print(o[o["ship"]].sort_values("t").to_string(index=False))
