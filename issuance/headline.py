"""Headline supply/demand metrics: one definition shared by the Overview/Country tabs and the
evidence table, so a number on screen and the test behind it are the same series.

Each metric carries a pressure sign: +1 when a higher value means more duration for the private
market to absorb (expected to cheapen bonds), −1 when a higher value means more demand.
`panel()` returns long rows: country, month, metric, value, phase (history | forecast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.paths import CACHE

METRICS: dict[str, dict] = {
    "priv12": {"label": "Private coupon DV01, 12m", "unit": "USD mn/bp", "sign": 1, "fmt": ".0f",
               "what": "Gross coupon DV01 issued less central-bank purchases (plus QT sales), trailing 12 months; "
                       "every month at the latest month's FX, so currency moves do not read as supply."},
    "absorb": {"label": "Free-float absorption", "unit": "% of free float, 12m", "sign": 1, "fmt": ".1f",
               "what": "12m private coupon DV01 ÷ free float (outstanding coupon DV01 less central-bank holdings)."},
    "absorb_long": {"label": "Absorption, long sleeve", "unit": "% of free float, 12m", "sign": 1, "fmt": ".1f",
                    "what": "Free-float absorption for 15y+ residual maturity."},
    "cb_share": {"label": "Central-bank share", "unit": "% of gross DV01", "sign": -1, "fmt": ".0f",
                 "what": "Central-bank net DV01 (purchases − sales) as a share of gross coupon DV01, 12m."},
    "score": {"label": "Demand score", "unit": "z", "sign": -1, "fmt": ".2f",
              "what": "Mean of the auction, foreign and official demand pillars, each z-scored."},
    "cover": {"label": "Auction cover, 3m", "unit": "x", "sign": -1, "fmt": ".2f",
              "what": "Bid-to-cover, 3-month mean, on each country's own convention."},
    "gross12": {"label": "Gross coupon DV01, 12m", "unit": "USD mn/bp", "sign": 1, "fmt": ".0f",
                "what": "All coupon issuance DV01 before central-bank flows, trailing 12 months, at the latest month's FX."},
    "imbalance": {"label": "Supply − cover imbalance", "unit": "z", "sign": 1, "fmt": ".2f",
                  "what": "z of 12m gross coupon DV01 minus z of 3m auction cover."},
    "tail": {"label": "Auction tail", "unit": "bp", "sign": 1, "fmt": ".1f",
             "what": "Stop-out yield minus pre-auction (when-issued) yield, monthly mean."},
    "pil_auction": {"label": "Pillar: auction", "unit": "z", "sign": -1, "fmt": ".2f", "what": "Auction demand pillar."},
    "pil_foreign": {"label": "Pillar: foreign", "unit": "z", "sign": -1, "fmt": ".2f", "what": "Foreign demand pillar."},
    "pil_official": {"label": "Pillar: official", "unit": "z", "sign": -1, "fmt": ".2f", "what": "Official demand pillar."},
    "pil_positioning": {"label": "Pillar: positioning", "unit": "z", "sign": -1, "fmt": ".2f",
                        "what": "Futures/dealer positioning pillar (not in the score)."},
    "dur_stock_gdp": {"label": "Private duration stock / GDP", "unit": "10y-eq % of GDP", "sign": 1, "fmt": ".1f",
                      "what": "Coupon stock held outside the central bank, in 10-year equivalents (private DV01 ÷ DV01 "
                              "of a 10y par bond at that month's 10y yield), as % of nominal GDP. GDP is annualised "
                              "quarterly, used from four months after the quarter starts (once published) and grown at "
                              "its trailing-year rate where not yet out. DV01 at each month's FX and GDP in local "
                              "currency, so currency moves do not read as supply."},
    "dur_stock_gdp_long": {"label": "Private duration stock / GDP, 15y+", "unit": "10y-eq % of GDP", "sign": 1,
                           "fmt": ".1f", "what": "Private duration stock / GDP for 15y+ residual maturity."},
}
HEADLINE = ("priv12", "absorb", "absorb_long", "cb_share", "score", "cover")
# Data month → first month the value is public: supply/auction data inside a month, the score
# (TIC, holdings, positioning) 4–8 weeks later.
LAG = {"score": 2, "pil_auction": 2, "pil_foreign": 2, "pil_official": 2, "pil_positioning": 2}
# Metrics built on central-bank purchases: the ECB publishes the country split of PSPP/PEPP
# holdings every two months, so for the euro area they are public a month later than issuance.
EA_LAG = {"priv12": 2, "cb_share": 2, "absorb": 2, "absorb_long": 2, "dur_stock_gdp": 2, "dur_stock_gdp_long": 2}
EA = ("DE", "FR", "IT")


def lag_for(metric: str, country: str) -> int:
    if country in EA and metric in EA_LAG:
        return max(EA_LAG[metric], LAG.get(metric, 1))
    return LAG.get(metric, 1)


def _read(name: str) -> pd.DataFrame:
    p = CACHE / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    if "month" in d:
        d["month"] = pd.to_datetime(d["month"])
    return d


def _rows(df: pd.DataFrame, col: str, metric: str, scale: float = 1.0) -> pd.DataFrame:
    if df.empty or col not in df:
        return pd.DataFrame()
    out = df[["country", "month", "phase"]].copy()
    out["value"] = pd.to_numeric(df[col], errors="coerce") * scale
    out["metric"] = metric
    return out.dropna(subset=["value"])


FX_COL = {"UK": "GBPUSD", "US": "USDUSD", "JP": "JPYUSD"}


def _fx_monthly() -> pd.DataFrame | None:
    """Monthly mean FX, USD per local unit."""
    path = CACHE / "fx_daily.parquet"
    if not path.exists():
        return None
    fx = pd.read_parquet(path)
    fx["month"] = pd.to_datetime(fx["date"]).dt.to_period("M").dt.to_timestamp()
    return fx.groupby("month").mean(numeric_only=True).sort_index()


def _fx_scale(s: pd.DataFrame) -> pd.Series | None:
    """Per row, latest-month FX ÷ that month's FX (USD per local unit)."""
    fx = _fx_monthly()
    if fx is None:
        return None
    month = s["month"].clip(upper=fx.index.max())
    col = s["country"].map(lambda c: FX_COL.get(c, "EURUSD"))
    then = pd.Series([fx[k].asof(m) for k, m in zip(col, month)], index=s.index)
    now = col.map(lambda k: fx[k].dropna().iloc[-1])
    return (now / then).where(then > 0, 1.0).fillna(1.0)


def _constant_fx(s: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Re-express monthly USD DV01 flows at the latest month's FX before the 12m sum. DV01 is
    converted at each auction's spot rate, so without this a yen or sterling move reads as a
    change in supply (USDJPY 110 → 150 alone cuts JP by a quarter)."""
    scale = _fx_scale(s)
    if scale is None:
        return s
    s = s.copy()
    for src, out in cols.items():
        flow = (s[src] * scale).fillna(0.0)
        vals = pd.Series(index=s.index, dtype=float)
        for c, idx in s.groupby("country").groups.items():
            # A month with no auction is a zero flow, not a missing row.
            f = pd.Series(flow.loc[idx].to_numpy(), index=s.loc[idx, "month"]).groupby(level=0).sum().asfreq("MS", fill_value=0.0)
            vals.loc[idx] = f.rolling(12, min_periods=12).sum().reindex(s.loc[idx, "month"]).to_numpy()
        s[out] = vals
    return s


LONG_BUCKETS = ("15-20y", "20-30y", "30y+")
# A quarter's GDP is used from this many months after the quarter starts (Q2 from August): every
# first estimate is out by then, so the history never uses a figure before it was published.
GDP_PUBLIC_MONTHS = 4


def gdp_monthly(months: pd.DatetimeIndex, quarterly: pd.Series) -> pd.Series:
    """Annualised nominal GDP per month from the latest quarter public by then; months whose due
    quarter is not out yet grow the last one at its trailing four-quarter rate."""
    q = quarterly.dropna().sort_index()
    out = pd.Series(np.nan, index=months)
    if q.empty:
        return out
    growth = q.iloc[-1] / q.iloc[-5] if len(q) >= 5 else 1.0
    usable = q.index + pd.DateOffset(months=GDP_PUBLIC_MONTHS)
    for m in months:
        due = (m - pd.DateOffset(months=GDP_PUBLIC_MONTHS)).to_period("Q").to_timestamp()
        ok = q[usable <= m]
        if ok.empty:
            continue
        last_q, v = ok.index[-1], ok.iloc[-1]
        n = (due.to_period("Q") - last_q.to_period("Q")).n if due > last_q else 0
        out.loc[m] = v * growth ** (n / 4)
    return out


def _stock_by_month(today: pd.Timestamp) -> pd.DataFrame:
    """Per country-month: private coupon DV01 (all, 15y+), total coupon DV01, phase."""
    f = _read("float_share_fc")
    if f.empty:
        f = _read("float_share")
    if f.empty:
        return pd.DataFrame()
    if "phase" not in f:
        f["phase"] = "history"
    f = f.assign(priv_long=f["private_stock_dv01"].where(f["tenor_bucket"].isin(LONG_BUCKETS), 0.0),
                 fc=f["phase"] == "forecast")
    g = f.groupby(["country", "month"], as_index=False).agg(
        priv=("private_stock_dv01", "sum"), priv_long=("priv_long", "sum"), stock=("stock_dv01", "sum"), fc=("fc", "any"))
    open_m = first_open_month(today)
    cur = today.to_period("M").to_timestamp()
    g["phase"] = ["forecast" if fc or m >= open_m.get(c, cur) else "history"
                  for c, m, fc in zip(g["country"], g["month"], g["fc"])]
    return g.drop(columns="fc")


def holder_share(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Central-bank share of the coupon stock, DV01-weighted: 1 − private DV01 / total DV01."""
    today = pd.Timestamp.today().normalize() if today is None else today
    g = _stock_by_month(today)
    if g.empty:
        return pd.DataFrame(columns=["country", "month", "cb_hold_share", "phase"])
    g["cb_hold_share"] = 1 - g["priv"] / g["stock"].where(g["stock"] > 0)
    return g[["country", "month", "cb_hold_share", "phase"]].dropna(subset=["cb_hold_share"])


def stock_panel(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """`dur_stock_gdp` and `dur_stock_gdp_long` rows (country, month, metric, value, phase)."""
    from issuance import market

    today = pd.Timestamp.today().normalize() if today is None else today
    cols = ["country", "month", "metric", "value", "phase"]
    g = _stock_by_month(today)
    gdp = market.load_gdp()
    mk = market.load()
    if g.empty or gdp.empty or mk.empty:
        return pd.DataFrame(columns=cols)
    scale = _fx_scale(g)
    g["scale"] = 1.0 if scale is None else scale
    fx = _fx_monthly()
    parts = []
    for c, h in g.groupby("country"):
        h = h.sort_values("month")
        months = pd.DatetimeIndex(h["month"])
        y10 = mk[(mk["country"] == c) & (mk["series"] == "y10")].set_index("month")["value"].sort_index()
        q = gdp[gdp["country"] == c].set_index("quarter")["gdp_bn"]
        if y10.empty or q.empty:
            continue
        y = y10.reindex(y10.index.union(months)).ffill().reindex(months)
        # USD bn of 10y par face per USD mn/bp of DV01: DV01 of USD 1bn face = 0.1·D USD mn/bp
        eq10 = h["priv"].to_numpy() * h["scale"].to_numpy() / (0.1 * market.mod_duration(y.to_numpy()))
        eq10_long = h["priv_long"].to_numpy() * h["scale"].to_numpy() / (0.1 * market.mod_duration(y.to_numpy()))
        k = FX_COL.get(c, "EURUSD")
        fx_now = 1.0 if fx is None or k not in fx else float(fx[k].dropna().iloc[-1])
        gdp_usd = gdp_monthly(months, q).to_numpy() * fx_now
        for name, v in (("dur_stock_gdp", eq10), ("dur_stock_gdp_long", eq10_long)):
            parts.append(pd.DataFrame({"country": c, "month": months, "metric": name,
                                       "value": 100 * v / gdp_usd, "phase": h["phase"].to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).dropna(subset=["value"])[cols]


TENORS = {"Short": "Short <7y", "Medium": "Medium 7–15y", "Long": "Long 15y+"}
TENOR_METRICS = ("priv12", "gross12", "cb_share", "absorb", "cover")


def panel_tenor(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """The headline metrics that split by residual-maturity sleeve, per sleeve (bills excluded).
    Same definitions as `panel()`; the All-tenor rows stay in `panel()` so the evidence table
    is not re-screened per sleeve."""
    from issuance.imbalance import SLEEVE_OF

    today = pd.Timestamp.today().normalize() if today is None else today
    cols = ["country", "month", "metric", "tenor", "value", "phase"]
    parts = []
    f = _read("float_share_fc")
    if f.empty:
        f = _read("float_share")
    t = _read("dv01_tenor_fc")
    if t.empty:
        t = _read("dv01_tenor")
    if not f.empty and not t.empty:
        # 0-1y is bills; the coupon sleeves start at 1y
        f = f[f["tenor_bucket"] != "0-1y"].assign(tenor=lambda d: d["tenor_bucket"].map(SLEEVE_OF))
        if "phase" not in f:
            f["phase"] = "history"
        if "sec_type" in t:
            t = t[t["sec_type"].astype(str) != "Bill"]
        t = t[t["tenor_bucket"] != "0-1y"].assign(tenor=lambda d: d["tenor_bucket"].map(SLEEVE_OF))
        keys = ["country", "month", "tenor"]
        g = f.groupby(keys, as_index=False).agg(priv=("flow_dv01", "sum"), fc=("phase", lambda p: (p == "forecast").any()))
        g = g.merge(t.groupby(keys, as_index=False).agg(gross=("dv01_usd_mn", "sum")), on=keys, how="outer")
        g = g[g["tenor"].isin(list(TENORS))]
        g[["priv", "gross"]] = g[["priv", "gross"]].fillna(0.0)
        g["fc"] = g["fc"].fillna(False).astype(bool)
        sc = _fx_scale(g)
        if sc is not None:
            g["gross"], g["priv"] = g["gross"] * sc, g["priv"] * sc
        open_m = first_open_month(today)
        cur = today.to_period("M").to_timestamp()
        for (c, t), h in g.groupby(["country", "tenor"]):
            h = h.set_index("month").sort_index()
            idx = pd.date_range(h.index.min(), h.index.max(), freq="MS")
            gross12 = h["gross"].reindex(idx, fill_value=0.0).rolling(12, min_periods=12).sum()
            priv12 = h["priv"].reindex(idx, fill_value=0.0).rolling(12, min_periods=12).sum()
            cb = 100 * (gross12 - priv12) / gross12.where(gross12.abs() > 1e-9)
            fc = h["fc"].reindex(idx, fill_value=False).astype(bool) | (idx >= open_m.get(c, cur))
            phase = pd.Series(fc, index=idx).map({True: "forecast", False: "history"})
            for name, v in (("priv12", priv12), ("gross12", gross12), ("cb_share", cb)):
                parts.append(pd.DataFrame({"country": c, "month": idx, "metric": name, "tenor": t,
                                           "value": v.to_numpy(), "phase": phase.to_numpy()}))
    imb = _read("rv_imbalance")
    if not imb.empty:
        imb = imb[imb["sleeve"].isin(list(TENORS))].copy()
        cur = today.to_period("M").to_timestamp()
        imb.loc[imb["month"] >= cur, "phase"] = "forecast"
        a = imb.assign(metric="absorb", tenor=imb["sleeve"], value=imb["absorb"] * 100)
        cv = imb[imb["phase"] == "history"]
        cv = cv.assign(metric="cover", tenor=cv["sleeve"], value=cv["cover3"])
        parts += [a[cols], cv[cols]]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    return out[cols].sort_values(["metric", "tenor", "country", "month"]).reset_index(drop=True)


def first_open_month(today: pd.Timestamp | None = None, supply_fc: pd.DataFrame | None = None) -> dict[str, pd.Timestamp]:
    """Per country, the first month that is not complete history: the current month, or an earlier
    one still holding forecast operations the tape has not caught up with."""
    today = pd.Timestamp.today().normalize() if today is None else today
    cur = today.to_period("M").to_timestamp()
    s = _read("rv_supply_fc") if supply_fc is None else supply_fc
    out: dict[str, pd.Timestamp] = {}
    if s is not None and not s.empty:
        out = {c: cur for c in s["country"].dropna().unique()}
        if "fc_remainder_dv01" in s.columns:
            rem = s[s["fc_remainder_dv01"].fillna(0.0) > 0]
            for c, m in pd.to_datetime(rem["month"]).groupby(rem["country"]).min().items():
                out[c] = min(out.get(c, cur), m)
    return out


def _complete(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """History rows for calendar months that have finished."""
    return df[df["month"] < today.to_period("M").to_timestamp()]


def panel(today: pd.Timestamp | None = None) -> pd.DataFrame:
    """A month is history only once it is over and holds no forecast operations; the current
    month and any month the tape has not caught up with are forecast for supply metrics and
    absent for metrics that have no forward path."""
    today = pd.Timestamp.today().normalize() if today is None else today
    cur = today.to_period("M").to_timestamp()
    parts = []
    hist = _read("rv_supply")
    s = _read("rv_supply_fc")
    if s.empty:
        s = hist
    if not s.empty:
        last = hist.groupby("country")["month"].max() if not hist.empty else s.groupby("country")["month"].max()
        s = s.sort_values(["country", "month"]).copy()
        rem = s["fc_remainder_dv01"].fillna(0.0) if "fc_remainder_dv01" in s else pd.Series(0.0, index=s.index)
        s["phase"] = ["forecast" if m > last.get(c, m) or m >= cur or r > 0 else "history"
                      for c, m, r in zip(s["country"], s["month"], rem)]
        s = _constant_fx(s, {"private_dv01_usd": "_priv12_cfx", "coupon_dv01_usd": "_gross12_cfx"})
        priv, gross = ("_priv12_cfx", "_gross12_cfx") if "_priv12_cfx" in s else ("private_dv01_usd_12m", "coupon_dv01_usd_12m")
        parts.append(_rows(s, priv, "priv12"))
        parts.append(_rows(s, gross, "gross12"))
        den = s["coupon_dv01_usd_12m"].where(s["coupon_dv01_usd_12m"].abs() > 1e-9)
        parts.append(_rows(s.assign(cb=s["official_dv01_usd_12m"] / den), "cb", "cb_share", 100))
        h = s[s["phase"] == "history"]
        parts.append(_rows(h, "tail_bp", "tail"))
    imb = _read("rv_imbalance")
    if not imb.empty:
        imb = imb.copy()
        open_m = set()
        if not s.empty:
            open_m = set(zip(s.loc[s["phase"] == "forecast", "country"], s.loc[s["phase"] == "forecast", "month"]))
        late = [m >= cur or (c, m) in open_m for c, m in zip(imb["country"], imb["month"])]
        imb.loc[late, "phase"] = "forecast"
        al = imb[imb["sleeve"] == "All"]
        parts.append(_rows(al, "absorb", "absorb", 100))
        parts.append(_rows(imb[imb["sleeve"] == "Long"], "absorb", "absorb_long", 100))
        parts.append(_rows(al[al["phase"] == "history"], "cover3", "cover"))
        parts.append(_rows(al[al["phase"] == "history"], "imbalance_z", "imbalance"))
    fs = _read("flow_score")
    if not fs.empty:
        fs["asof"] = pd.to_datetime(fs["asof"])
        fs = _complete(fs[fs["component"].isin(["_score", "_pillar"]) & (fs["month"] <= fs["asof"])], today)
        fs = fs.assign(phase="history", key=fs["pillar"].map(lambda p: "score" if p == "score" else f"pil_{p}"))
        for key, g in fs.groupby("key"):
            parts.append(_rows(g, "z", key))
    parts.append(stock_panel(today))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["country", "month", "metric", "value", "phase"])
    out = pd.concat(parts, ignore_index=True)
    return out[["country", "month", "metric", "value", "phase"]].sort_values(["metric", "country", "month"]).reset_index(drop=True)
