"""Supply against demand, per country and DMO sleeve, history and forward.

Two measures:

  absorb_z     Free-float absorption: 12m private coupon DV01 (gross minus official) over the DV01
               of the private coupon stock, z over 36m. How much new duration the price-sensitive
               holder has to take relative to what it already holds. Headline.
  imbalance_z  12m gross coupon DV01 z minus z of the 3-month mean auction cover. Auction-desk view.

`issuance/experiments/supply_metric_test.py` scores 14 candidates against 2s10s, 10s30s, 10y,
10y over Bunds and UK gilt-OIS asset swap (5/10/20/25y). The absorption family ranks first on the
signed score; it is strongest on UK ASW (10y: 0.61 on the 12m change, 0.37 on the next 3m) and
weak or mixed on outright slope and spreads. Cover on its own has no sign.

The dashboard recomputes absorb_z from `absorb` with its z-window setting; absorb_z here is 36m.
`absorb_by_bucket` is the same ratio per residual tenor bucket.

Forward: supply from the priced auction calendar (dv01_tenor_fc), absorption from float_share_fc,
cover as a recursive 3-month mean of the last prints.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from issuance.demand import _uk_auction_frame, normalize_btc
from issuance.headline import first_open_month
from issuance.isin_flow import market_ops
from issuance.paths import CACHE
from issuance.rv_metrics import _coupon_ops, rolling_z
from issuance.tenor import bucket_from_years

Z_WINDOW, Z_MINP = 36, 18
SLEEVE_OF = {
    "0-1y": "Short", "1-3y": "Short", "3-7y": "Short",
    "7-10y": "Medium", "10-15y": "Medium",
    "15-20y": "Long", "20-30y": "Long", "30y+": "Long",
}
SLEEVES = ("All", "Short", "Medium", "Long")


def _with_all(df: pd.DataFrame, keys: list[str], vals: list[str]) -> pd.DataFrame:
    df = df[df["sleeve"].notna()]
    tot = df.groupby(keys, as_index=False)[vals].sum()
    tot["sleeve"] = "All"
    return pd.concat([df.groupby(keys + ["sleeve"], as_index=False)[vals].sum(), tot], ignore_index=True)


def _uk_cover() -> pd.DataFrame:
    raw = _uk_auction_frame()
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    year = raw["name"].astype(str).map(lambda s: (m := re.search(r"(19|20)\d{2}\s*$", s.strip())) and int(m.group(0)))
    years = (year + 0.5) - (raw["date"].dt.year + raw["date"].dt.dayofyear / 365.25)
    raw["tenor_bucket"] = years.map(lambda y: bucket_from_years(float(y)) if pd.notna(y) else None)
    return pd.DataFrame({
        "country": "UK", "op_dt": raw["date"], "tenor_bucket": raw["tenor_bucket"],
        "btc": pd.to_numeric(raw["cover"], errors="coerce"),
        "face": pd.to_numeric(raw.get("total_nominal_mn"), errors="coerce"),
    })


def auction_cover(ops: pd.DataFrame) -> pd.DataFrame:
    """Face-weighted cover per country, month and sleeve (including All)."""
    g = market_ops(_coupon_ops(ops))
    g = g[(g["amount_issued"] > 0) & (g["country"] != "UK")].copy()
    btc = normalize_btc(g["btc"]) if "btc" in g.columns else pd.Series(np.nan, index=g.index)
    if "amount_bid" in g.columns:
        cover = pd.to_numeric(g["amount_bid"], errors="coerce") / pd.to_numeric(g["amount_issued"], errors="coerce")
        btc = btc.fillna(cover.where((cover > 0.4) & (cover < 20)))
    parts = [pd.DataFrame({"country": g["country"], "op_dt": g["op_dt"], "tenor_bucket": g["tenor_bucket"],
                           "btc": btc, "face": g["amount_issued"]}), _uk_cover()]
    a = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    a = a[a["btc"].notna() & (a["face"] > 0)]
    a["month"] = pd.to_datetime(a["op_dt"]).dt.to_period("M").dt.to_timestamp()
    a["sleeve"] = a["tenor_bucket"].map(SLEEVE_OF)
    a["wb"] = a["btc"] * a["face"]
    out = _with_all(a, ["country", "month"], ["wb", "face"])
    out["btc"] = out["wb"] / out["face"]
    return out[["country", "month", "sleeve", "btc"]]


def _supply() -> pd.DataFrame:
    path = CACHE / "dv01_tenor_fc.parquet"
    t = pd.read_parquet(path if path.exists() else CACHE / "dv01_tenor.parquet")
    if "sec_type" in t.columns:
        t = t[t["sec_type"].astype(str) != "Bill"]
    t = t.copy()
    t["month"] = pd.to_datetime(t["month"])
    t["sleeve"] = t["tenor_bucket"].map(SLEEVE_OF)
    return _with_all(t, ["country", "month"], ["dv01_usd_mn"]).rename(columns={"dv01_usd_mn": "supply_dv01"})


def _absorb() -> pd.DataFrame:
    path = CACHE / "float_share_fc.parquet"
    f = pd.read_parquet(path if path.exists() else CACHE / "float_share.parquet").copy()
    f["month"] = pd.to_datetime(f["month"])
    f["sleeve"] = f["tenor_bucket"].map(SLEEVE_OF)
    return _with_all(f, ["country", "month"], ["flow_dv01", "private_stock_dv01"])


def _cover_forward(s: pd.Series, end: pd.Timestamp) -> pd.Series:
    """3-month mean of prints; past the last print each month is the mean of the three before it."""
    s = s.dropna()
    if s.empty:
        return s
    idx = pd.date_range(s.index.min(), end, freq="MS")
    vals = s.reindex(idx)
    last = s.index.max()
    out = vals.rolling(3, min_periods=1).mean()
    hist = list(vals.loc[:last].dropna().to_numpy()[-3:])
    for m in idx[idx > last]:
        nxt = float(np.mean(hist[-3:]))
        out.loc[m] = nxt
        hist.append(nxt)
    return out


def imbalance_panel(ops: pd.DataFrame) -> pd.DataFrame:
    sup, cov, ab = _supply(), auction_cover(ops), _absorb()
    tape_end = market_ops(_coupon_ops(ops)).groupby("country")["op_dt"].max().dt.to_period("M").dt.to_timestamp()
    horizon = sup.groupby("country")["month"].max()
    open_m = first_open_month()
    rows = []
    for (c, sl), g in sup.groupby(["country", "sleeve"]):
        g = g.set_index("month")["supply_dv01"].sort_index()
        # A sleeve with no auction in its last forward months still has a stock and a 12m sum.
        idx = pd.date_range(g.index.min(), horizon[c], freq="MS")
        frame = pd.DataFrame(index=idx)
        frame["supply_dv01"] = g.reindex(idx).fillna(0.0)
        frame["supply12"] = frame["supply_dv01"].rolling(12, min_periods=12).sum()
        frame["supply_z"] = rolling_z(frame["supply12"], Z_WINDOW, Z_MINP)
        cs = cov[(cov["country"] == c) & (cov["sleeve"] == sl)].set_index("month")["btc"].sort_index()
        frame["btc"] = cs.reindex(idx)
        frame["cover3"] = _cover_forward(cs, idx.max()).reindex(idx)
        frame["cover3_z"] = rolling_z(frame["cover3"], Z_WINDOW, Z_MINP)
        frame["imbalance_z"] = frame["supply_z"] - frame["cover3_z"]
        a = ab[(ab["country"] == c) & (ab["sleeve"] == sl)].set_index("month").sort_index().reindex(idx)
        flow12 = a["flow_dv01"].rolling(12, min_periods=12).sum()
        frame["absorb"] = flow12 / a["private_stock_dv01"].where(a["private_stock_dv01"] > 0)
        frame["absorb_z"] = rolling_z(frame["absorb"], Z_WINDOW, Z_MINP)
        frame["country"], frame["sleeve"] = c, sl
        end = tape_end.get(c)
        hist = frame.index <= end if end is not None else np.ones(len(frame), bool)
        if c in open_m:
            hist &= frame.index < open_m[c]
        frame["phase"] = np.where(hist, "history", "forecast")
        rows.append(frame.reset_index(names="month"))
    out = pd.concat(rows, ignore_index=True)
    return out[out["month"] >= "2005-01-01"]


def absorb_by_bucket() -> pd.DataFrame:
    """12m private coupon DV01 over private DV01 stock, per residual bucket. 0-1y takes no issuance."""
    path = CACHE / "float_share_fc.parquet"
    f = pd.read_parquet(path if path.exists() else CACHE / "float_share.parquet").copy()
    f["month"] = pd.to_datetime(f["month"])
    if "phase" not in f.columns:
        f["phase"] = "history"
    total = f.groupby(["country", "month"])["private_stock_dv01"].sum().rename("total")
    f = f[f["tenor_bucket"] != "0-1y"]
    rows = []
    for (c, b), g in f.groupby(["country", "tenor_bucket"]):
        g = g.groupby("month").agg(flow=("flow_dv01", "sum"), stock=("private_stock_dv01", "sum"), phase=("phase", "first"))
        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(idx)
        flow12 = g["flow"].fillna(0.0).rolling(12, min_periods=12).sum()
        # A bucket new issues print into but age straight out of (a 10.01y US 10y) has almost no
        # stock; below 3% of the country's private DV01 the ratio is a boundary artefact.
        tot = total.loc[c].reindex(idx)
        absorb = flow12 / g["stock"].where(g["stock"] > 0.03 * tot)
        rows.append(pd.DataFrame({"country": c, "tenor_bucket": b, "month": idx, "absorb": absorb.to_numpy(),
                                  "phase": g["phase"].fillna("history").to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    return out[(out["month"] >= "2005-01-01") & out["absorb"].notna()]


def write_imbalance(ops: pd.DataFrame) -> pd.DataFrame:
    out = imbalance_panel(ops)
    out.to_parquet(CACHE / "rv_imbalance.parquet", index=False)
    absorb_by_bucket().to_parquet(CACHE / "rv_absorb_bucket.parquet", index=False)
    last = out[(out["sleeve"] == "All") & (out["phase"] == "history")].sort_values("month").groupby("country").tail(1)
    print("  imbalance latest", last.set_index("country")[["imbalance_z", "absorb_z"]].round(2).to_dict("index"))
    return out
