"""Which supply/demand metric lines up with yields?

Targets (month mean of daily where available):
  slope   2s10s        US (FRED), DE (Bundesbank), UK (DMO spot nodes)
  s1030   10s30s       US, DE, UK
  lvl10   10y yield    US, DE, UK daily; FR, IT OECD monthly
  sprde   10y over Bund FR, IT (OECD monthly)
  asw10/20/30          UK gilt minus SONIA OIS (workbook 32, 2022+). No USD/EUR swap
                       history is cached, so asset swap is UK only.
Scored on the 12m change ending at t (same window as the metric), and the 3m and 6m change
after t. Correlations are pooled after removing each country's mean; monthly observations
overlap, so read them as ranking, not t-stats.

  PYTHONPATH=. .venv/bin/python issuance/experiments/supply_metric_test.py
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from issuance.imbalance import SLEEVE_OF
from issuance.paths import CACHE, REPO
from issuance.rv_metrics import rolling_z

G4 = REPO / "correlations" / "g4" / "cache"
WB = REPO / "projects" / "uk_long_end" / "workbook"
OECD = {"FR": "IRLTLT01FRM156N", "IT": "IRLTLT01ITM156N", "DE": "IRLTLT01DEM156N"}


def _fred(series: str) -> pd.Series:
    raw = urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "v"]
    return pd.Series(pd.to_numeric(df["v"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"])).dropna()


def _monthly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(s.index.to_period("M").to_timestamp()).mean()


def _fred_csv(name: str, col: str) -> pd.Series:
    return pd.read_csv(G4 / "fred" / f"fred_{name}.csv", index_col=0, parse_dates=True)[col]


def _uk_asw() -> pd.DataFrame:
    """Gilt nominal spot minus SONIA OIS spot, BoE fitted curves, percentage points."""
    c = pd.read_parquet(REPO / "uk_flows" / "output" / "curves.parquet")
    c = c[(c["measure"] == "spot") & (c["segment"] == "full") & c["curve"].isin(["nominal", "ois"])]
    out = {}
    for t in (5, 10, 20, 25):
        sub = c[(c["maturity_years"] - t).abs() < 1e-6].pivot_table(index="date", columns="curve", values="rate")
        if {"nominal", "ois"} <= set(sub.columns):
            s = (sub["nominal"] - sub["ois"]).dropna()
            if len(s) > 500:
                out[t] = s
    return pd.DataFrame(out)


def targets() -> pd.DataFrame:
    us2, us10 = _fred_csv("dgs2", "ust_2"), _fred_csv("dgs10", "ust_10")
    us30 = _fred_csv("dgs30", "ust_30")
    bund = pd.read_csv(G4 / "bbk" / "bbk_bund.csv", index_col=0, parse_dates=True)
    uk = pd.read_parquet(CACHE / "uk_spot_nodes.parquet").set_index("date")
    oecd = {k: _monthly(_fred(v)) for k, v in OECD.items()}
    asw = _uk_asw()
    series = {
        ("US", "slope"): us10 - us2, ("DE", "slope"): bund["bund_10"] - bund["bund_2"], ("UK", "slope"): uk["y10"] - uk["y2"],
        ("US", "s1030"): us30 - us10, ("DE", "s1030"): bund["bund_30"] - bund["bund_10"], ("UK", "s1030"): uk["y30"] - uk["y10"],
        ("US", "lvl10"): us10, ("DE", "lvl10"): bund["bund_10"], ("UK", "lvl10"): uk["y10"],
        ("FR", "lvl10"): oecd["FR"], ("IT", "lvl10"): oecd["IT"],
        ("FR", "sprde"): oecd["FR"] - oecd["DE"], ("IT", "sprde"): oecd["IT"] - oecd["DE"],
        **{("UK", f"asw{t}"): asw[t] for t in asw.columns},
    }
    out = []
    for (c, name), s in series.items():
        m = _monthly(s)
        out.append(pd.DataFrame({"country": c, "target": name, "month": m.index, "y": m.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _z(df: pd.DataFrame, col: str, window: int = 36) -> pd.Series:
    return df.groupby("country", group_keys=False)[col].apply(lambda x: rolling_z(x, window, window // 2))


def _absorb(f: pd.DataFrame, sleeve: str | None) -> pd.DataFrame:
    if sleeve:
        f = f[f["tenor_bucket"].map(SLEEVE_OF) == sleeve]
    g = f.groupby(["country", "month"], as_index=False)[["flow_dv01", "private_stock_dv01"]].sum().sort_values(["country", "month"])
    g["flow12"] = g.groupby("country")["flow_dv01"].transform(lambda x: x.rolling(12, min_periods=12).sum())
    g["absorb"] = g["flow12"] / g["private_stock_dv01"].where(g["private_stock_dv01"] > 0)
    return g[["country", "month", "absorb"]]


def metrics() -> pd.DataFrame:
    s = pd.read_parquet(CACHE / "rv_supply.parquet").sort_values(["country", "month"])
    s["month"] = pd.to_datetime(s["month"])
    s["btc_3m"] = s.groupby("country", group_keys=False)["btc"].apply(lambda x: x.rolling(3, min_periods=2).mean())
    s["cover3_z"] = _z(s, "btc_3m")
    s["off12"] = s.groupby("country")["official_dv01_usd"].transform(lambda x: x.rolling(12, min_periods=12).sum())
    s["off_share12"] = s["off12"] / s["coupon_dv01_usd_12m"].where(s["coupon_dv01_usd_12m"].abs() > 1e-9)

    f = pd.read_parquet(CACHE / "float_share.parquet")
    f["month"] = pd.to_datetime(f["month"])
    a = _absorb(f, None)
    for sl in ("Short", "Medium", "Long"):
        a = a.merge(_absorb(f, sl).rename(columns={"absorb": f"absorb_{sl.lower()}"}), on=["country", "month"], how="left")
    a = a.sort_values(["country", "month"])
    w = pd.read_parquet(CACHE / "rv_wam.parquet")[["country", "month", "wam_issuance_12m", "wam_stock"]]
    w["month"] = pd.to_datetime(w["month"])

    df = s.merge(a, on=["country", "month"], how="left").merge(w, on=["country", "month"], how="left")
    df = df.sort_values(["country", "month"]).reset_index(drop=True)
    df["wam_tilt"] = df["wam_issuance_12m"] - df["wam_stock"]
    out = pd.DataFrame({"country": df["country"], "month": df["month"]})
    out["gross12_z"] = df["coupon_dv01_usd_12m_z"]
    out["priv12_z"] = df["private_dv01_usd_12m_z"]
    out["imb_cover3"] = df["coupon_dv01_usd_12m_z"] - df["cover3_z"]
    out["cover3_neg"] = -df["cover3_z"]
    out["off_share_neg_z"] = -_z(df, "off_share12")
    out["absorb"] = df["absorb"]
    out["absorb_z"] = _z(df, "absorb")
    out["absorb_z60"] = _z(df, "absorb", 60)
    out["absorb_d12"] = df.groupby("country")["absorb"].diff(12)
    out["absorb_long_z"] = _z(df, "absorb_long")
    out["absorb_short_z"] = _z(df, "absorb_short")
    df["tilt"] = df["absorb_long"] - df["absorb_short"]
    out["absorb_tilt_z"] = _z(df, "tilt")
    out["absorb_minus_cover3"] = out["absorb_z"] - df["cover3_z"]
    out["wam_tilt_z"] = _z(df, "wam_tilt")
    return out


CANDIDATES = (
    "absorb_z", "absorb_z60", "absorb", "absorb_d12", "absorb_long_z", "absorb_short_z", "absorb_tilt_z",
    "absorb_minus_cover3", "priv12_z", "gross12_z", "imb_cover3", "cover3_neg", "off_share_neg_z", "wam_tilt_z",
)
HORIZONS = ("d12", "f3", "f6")


def _corr(sub: pd.DataFrame, m: str, h: str) -> tuple[float, int, int]:
    sub = sub[["country", m, h]].dropna()
    if len(sub) < 30:
        return np.nan, 0, 0
    dm = sub.groupby("country")[[m, h]].transform(lambda x: x - x.mean())
    pooled = float(np.corrcoef(dm[m], dm[h])[0, 1])
    per = [np.corrcoef(g[m], g[h])[0, 1] for _, g in sub.groupby("country") if len(g) > 24]
    return pooled, sum(np.sign(v) == np.sign(pooled) for v in per), len(per)


def score(start: str = "2012-01-01") -> pd.DataFrame:
    t = targets().sort_values(["country", "target", "month"])
    g = t.groupby(["country", "target"])["y"]
    t["d12"] = g.diff(12)
    t["f3"] = g.shift(-3) - t["y"]
    t["f6"] = g.shift(-6) - t["y"]
    df = metrics().merge(t, on=["country", "month"], how="inner")
    df = df[df["month"] >= start]
    rows = []
    for tgt, sub in df.groupby("target"):
        for m in CANDIDATES:
            row = {"target": tgt, "metric": m}
            for h in HORIZONS:
                r, same, n = _corr(sub, m, h)
                row[h] = round(r, 3) if pd.notna(r) else np.nan
                row[f"{h}_sign"] = f"{same}/{n}"
            rows.append(row)
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame) -> pd.DataFrame:
    """Signed mean corr. Every metric is oriented + = more supply pressure, and every target
    should then rise (higher yield, steeper, wider over Bunds, gilts cheaper to swaps). A large
    negative is a confound, not a fit, so |corr| is not the score."""
    piv = res.pivot(index="metric", columns="target", values=list(HORIZONS))
    out = pd.DataFrame(index=piv.index)
    for h in HORIZONS:
        out[h] = piv[h].mean(axis=1).round(3)
    cells = piv.to_numpy()
    out["right_sign"] = [f"{int((r > 0).sum())}/{int(np.isfinite(r).sum())}" for r in cells]
    out["signed_mean"] = piv.mean(axis=1).round(3)
    return out.sort_values("signed_mean", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    res = score()
    for tgt, sub in res.groupby("target"):
        print(f"\n== {tgt} ==")
        print(sub.drop(columns="target").sort_values("d12", key=lambda x: -x.abs()).to_string(index=False))
    print("\n== ranking: mean |corr| across targets ==")
    print(summary(res).to_string())
