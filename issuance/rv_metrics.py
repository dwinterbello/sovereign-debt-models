"""Public vs private WAM and net-supply metrics used for curve RV.

WAM_public  — EA: published PSPP+PEPP WAM. UK: residual WAM of live APF holdings.
WAM_stock   — residual WAM of live *coupon* notional (bills excluded).
WAM_private — (S·WAM_s − H·WAM_h) / (S − H). US uses SOMA CUSIP stock.

Net private DV01 = coupon issuance DV01 − official DV01 (QT raises private supply).
DFA Own Holdings are excluded from issuance/DV01 (inventory, not market).
Curve supply: 2s10s = belly(7-15) − front(1-7);
10s30s = long(15y+) − belly (15-20y sits with the long end).
5s30s fly = front + long − 2·belly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.dv01 import attach_dv01, modified_duration
from issuance.fetch_fx import fetch_fx
from issuance.paths import CACHE
from issuance.tenor import BUCKET_MID

FRONT = ("1-3y", "3-7y")
BELLY = ("7-10y", "10-15y")
MIDLONG = ("15-20y",)
LONG = ("15-20y", "20-30y", "30y+")
ULTRA = ("30y+",)


def _months(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                         pd.Timestamp(end).to_period("M").to_timestamp(), freq="MS")


def _universe(ops: pd.DataFrame, country: str) -> pd.DataFrame:
    g = ops[(ops["country"] == country) & ops["isin"].notna()].copy()
    if g.empty:
        return g
    g["isin"] = g["isin"].astype("string").str.upper()
    u = g.groupby("isin").agg(
        issued=("amount_issued", "sum"),
        maturity_dt=("maturity_dt", "last"),
        first_op=("op_dt", "min"),
        coupon=("coupon", "last"),
    ).reset_index()
    return u[u["issued"].fillna(0) > 0]


def _live(universe: pd.DataFrame, asof) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    live = universe[(universe["first_op"] <= asof) & (pd.to_datetime(universe["maturity_dt"]) > asof)].copy()
    if live.empty:
        return live
    live["resid"] = (pd.to_datetime(live["maturity_dt"]) - asof).dt.days / 365.25
    live = live[live["resid"] > 0]
    return live


def _wam(face: pd.Series, years: pd.Series) -> float:
    w = pd.to_numeric(face, errors="coerce")
    y = pd.to_numeric(years, errors="coerce")
    mask = w.notna() & y.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((w[mask] * y[mask]).sum() / w[mask].sum())


def rolling_z(series: pd.Series, window: int = 36, min_periods: int = 18) -> pd.Series:
    """Rolling z-score. window months of history, no look-ahead."""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd.replace(0.0, np.nan)


def _coupon_ops(ops: pd.DataFrame, bills: bool = False) -> pd.DataFrame:
    if ops is None or ops.empty or "sec_type" not in ops.columns:
        return ops
    drop = ["Foreign"] if bills else ["Bill", "Foreign"]
    out = ops[~ops["sec_type"].astype(str).isin(drop)].copy()
    if bills and "isin" in out.columns:
        # UK T-bills come without an ISIN; same-maturity bills are one line.
        miss = out["isin"].isna() & out["sec_type"].astype(str).eq("Bill") & out["maturity_dt"].notna()
        out.loc[miss, "isin"] = (out.loc[miss, "country"].astype(str) + " bill "
                                 + pd.to_datetime(out.loc[miss, "maturity_dt"]).dt.strftime("%Y-%m-%d"))
    return out


def stock_wam_panel(ops: pd.DataFrame, bills: bool = False) -> pd.DataFrame:
    """Residual WAM of live issued notional. Coupon-only unless bills=True."""
    rows = []
    for country, grp in _coupon_ops(ops, bills).groupby("country"):
        uni = _universe(grp, country)
        if uni.empty:
            continue
        months = _months(max(uni["first_op"].min(), pd.Timestamp("2009-01-01")),
                         grp["op_dt"].max())
        for asof in months:
            live = _live(uni, asof)
            if live.empty:
                continue
            rows.append({
                "country": country,
                "month": asof,
                "outstanding_mn": float(live["issued"].sum()),
                "wam_stock": _wam(live["issued"], live["resid"]),
                "n_isin": int(len(live)),
            })
    return pd.DataFrame(rows)


def _apf_public_panel() -> pd.DataFrame:
    path = CACHE / "apf_operations.parquet"
    if not path.exists():
        return pd.DataFrame()
    ops = pd.read_parquet(path)
    if "maturity_dt" not in ops.columns or ops["resid_mat_yrs"].isna().all():
        from issuance.fetch_apf import _join_gilt_maturity
        from issuance.tenor import attach_tenor
        ops = attach_tenor(_join_gilt_maturity(ops))
    ops["isin"] = ops["isin"].astype("string").str.upper()
    ops["month"] = pd.to_datetime(ops["op_dt"]).dt.to_period("M").dt.to_timestamp()
    months = _months(ops["month"].min(), ops["month"].max())
    rows = []
    running: dict[str, float] = {}
    mat = ops.groupby("isin")["maturity_dt"].last().to_dict()
    ops = ops.sort_values("op_dt")
    i = 0
    recs = list(ops.itertuples(index=False))
    for asof in months:
        while i < len(recs) and pd.Timestamp(recs[i].op_dt) <= asof + pd.offsets.MonthEnd(0):
            running[str(recs[i].isin)] = running.get(str(recs[i].isin), 0.0) + float(recs[i].amount)
            i += 1
        face, yrs = [], []
        for isin, amt in running.items():
            if amt <= 0:
                continue
            mdt = mat.get(isin)
            if pd.isna(mdt) or pd.Timestamp(mdt) <= asof:
                continue
            resid = (pd.Timestamp(mdt) - asof).days / 365.25
            if resid <= 0:
                continue
            face.append(amt)
            yrs.append(resid)
        if not face:
            continue
        face_s, yrs_s = pd.Series(face), pd.Series(yrs)
        rows.append({
            "country": "UK",
            "month": asof,
            "official_mn": float(face_s.sum()),
            "wam_public": _wam(face_s, yrs_s),
            "program": "APF",
        })
    return pd.DataFrame(rows)


def _ea_public_panel() -> pd.DataFrame:
    path = CACHE / "official_flow.parquet"
    if not path.exists():
        return pd.DataFrame()
    off = pd.read_parquet(path)
    off = off[off["country"].isin(["DE", "FR", "IT"])].copy()
    if "tenor_bucket" in off.columns:
        off = off[off["tenor_bucket"].astype(str) != "non_central"]
    if off.empty or "official_stock_mn" not in off.columns:
        return pd.DataFrame()
    off["month"] = pd.to_datetime(off["month"])
    g = off.groupby(["country", "month", "program"], dropna=False).agg(
        official_mn=("official_stock_mn", "sum"),
        wam_public=("wam_yrs", "first"),
    ).reset_index()
    stock = g.pivot_table(index=["country", "month"], columns="program", values="official_mn")
    wams = g.pivot_table(index=["country", "month"], columns="program", values="wam_public")
    stock = stock.groupby(level=0).ffill()
    wams = wams.groupby(level=0).ffill()
    rows = []
    for (country, month) in stock.index:
        face = stock.loc[(country, month)].dropna()
        ww = wams.loc[(country, month)].reindex(face.index)
        if face.empty:
            continue
        rows.append({
            "country": country,
            "month": month,
            "official_mn": float(face.sum()),
            "wam_public": _wam(face, ww),
            "program": "PSPP+PEPP",
        })
    return pd.DataFrame(rows)


def _soma_public_panel(bills: bool = False) -> pd.DataFrame:
    path = CACHE / "soma_holdings.parquet"
    if not path.exists():
        return pd.DataFrame()
    h = pd.read_parquet(path)
    if h.empty:
        return pd.DataFrame()
    if "sec_type" in h.columns and not bills:
        h = h[h["sec_type"].astype(str) != "Bill"]
    rows = []
    for month, sub in h.groupby("month"):
        yrs = pd.to_numeric(sub.get("resid_mat_yrs"), errors="coerce")
        face = pd.to_numeric(sub["official_stock_mn"], errors="coerce")
        live = sub.loc[(yrs > 0) & (face > 0)]
        if live.empty:
            continue
        rows.append({
            "country": "US",
            "month": pd.Timestamp(month),
            "official_mn": float(pd.to_numeric(live["official_stock_mn"], errors="coerce").sum()),
            "wam_public": _wam(live["official_stock_mn"], live["resid_mat_yrs"]),
            "program": "SOMA",
        })
    return pd.DataFrame(rows)


def _boj_public_panel() -> pd.DataFrame:
    path = CACHE / "official_isin_flow.parquet"
    if not path.exists():
        return pd.DataFrame()
    h = pd.read_parquet(path)
    h = h[h["country"].astype(str).eq("JP")] if "country" in h.columns else h.iloc[0:0]
    if h.empty or "official_stock_mn" not in h.columns:
        return pd.DataFrame()
    rows = []
    for month, sub in h.groupby("month"):
        face = pd.to_numeric(sub["official_stock_mn"], errors="coerce")
        yrs = pd.to_numeric(sub["resid_mat_yrs"], errors="coerce")
        live = (face > 0) & (yrs > 0)
        if not live.any():
            continue
        rows.append({"country": "JP", "month": pd.Timestamp(month), "official_mn": float(face[live].sum()),
                     "wam_public": _wam(face[live], yrs[live]), "program": "BoJ"})
    return pd.DataFrame(rows)


def wam_panel(ops: pd.DataFrame, bills: bool = False) -> pd.DataFrame:
    stock = stock_wam_panel(ops, bills)
    # APF, PSPP+PEPP and the BoJ coupon book hold no bills here; SOMA does.
    pub = pd.concat([_apf_public_panel(), _ea_public_panel(), _soma_public_panel(bills), _boj_public_panel()],
                    ignore_index=True)
    if stock.empty:
        return stock
    out = stock.merge(pub[["country", "month", "official_mn", "wam_public"]],
                      on=["country", "month"], how="left")
    out = out.sort_values(["country", "month"])
    out["official_mn"] = out.groupby("country")["official_mn"].ffill()
    out["wam_public"] = out.groupby("country")["wam_public"].ffill()
    out["official_mn"] = out["official_mn"].fillna(0.0)
    out["private_mn"] = (out["outstanding_mn"] - out["official_mn"]).clip(lower=0)
    # Identity: S·Ws = H·Wh + P·Wp. No official book → private = stock.
    has_pub = out["official_mn"] > 1.0
    num = out["outstanding_mn"] * out["wam_stock"] - out["official_mn"] * out["wam_public"]
    den = out["private_mn"]
    out["wam_private"] = np.where(has_pub & (den > 0), num / den, out["wam_stock"])
    out["wam_gap"] = np.where(has_pub, out["wam_public"] - out["wam_private"], np.nan)
    out["official_share"] = np.where(out["outstanding_mn"] > 0,
                                     out["official_mn"] / out["outstanding_mn"], np.nan)
    return out


def _official_dv01(official: pd.DataFrame, fx: pd.DataFrame, bills: bool = False) -> pd.DataFrame:
    """Face official flow → local/USD DV01 via bucket-mid modified duration."""
    if official is None or official.empty:
        return pd.DataFrame()
    off = official.copy()
    if "tenor_bucket" in off.columns:
        off = off[off["tenor_bucket"].astype(str) != "non_central"]
    if "sec_type" in off.columns and not bills:
        off = off[off["sec_type"].astype(str) != "Bill"]
    if off.empty:
        return pd.DataFrame()
    off["month"] = pd.to_datetime(off["month"])
    bucket_yrs = off["tenor_bucket"].map(BUCKET_MID)
    if "resid_mat_yrs" in off.columns:
        years = pd.to_numeric(off["resid_mat_yrs"], errors="coerce").fillna(bucket_yrs)
    else:
        years = bucket_yrs
    if "wam_yrs" in off.columns:
        years = years.fillna(pd.to_numeric(off["wam_yrs"], errors="coerce"))
    from issuance.isin_flow import load_ops
    from issuance.pricing import curve_yield, market_curve

    curve = market_curve(load_ops())
    linker = off["sec_type"].astype(str).eq("Linker") if "sec_type" in off.columns else pd.Series(False, index=off.index)
    yld = np.full(len(off), np.nan)
    for country, idx in off.groupby("country").groups.items():
        pos = off.index.get_indexer(idx)
        yld[pos] = curve_yield(curve, country, linker.loc[idx].to_numpy(), off.loc[idx, "month"].to_numpy(),
                               years.loc[idx].fillna(5.0).to_numpy())
    tmp = pd.DataFrame({
        "month": off["month"],
        "ccy": off["country"].map({"UK": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR", "US": "USD", "JP": "JPY"}).fillna("EUR"),
        "gross_mn": pd.to_numeric(off["official_mn"], errors="coerce"),
        "tenor_bucket": off["tenor_bucket"],
        "resid_mat_yrs": years,
        "yield": yld,
        "coupon": pd.to_numeric(off["coupon"], errors="coerce") if "coupon" in off.columns else pd.NA,
        "sec_type": off["sec_type"] if "sec_type" in off.columns else pd.NA,
        "country": off["country"],
    })
    if "isin" in off.columns:
        tmp["isin"] = off["isin"]
    priced = attach_dv01(tmp, fx)
    off = off.reset_index(drop=True)
    off["official_dv01_usd"] = priced["dv01_usd_mn"].to_numpy()
    return off


FLOW_COLS = ("gross_dv01_usd", "coupon_dv01_usd", "bill_dv01_usd", "official_dv01_usd", "official_face_mn",
             "coupon_gross_mn", "buyback_dv01_usd", "net_coupon_dv01_usd", "private_dv01_usd", "dv01_front",
             "dv01_belly", "dv01_midlong", "dv01_long", "dv01_ultra", "dv01_2s10s", "dv01_10s30s", "dv01_5s30s")


def _monthly_grid(out: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar month between a country's first and last coupon auction, so rolling
    sums cover months, not rows. A month with no auction is zero issuance; official flow in it is kept."""
    parts = []
    for c, g in out.groupby("country"):
        live = g.loc[g["coupon_dv01_usd"].fillna(0.0) != 0, "month"]
        if live.empty:
            continue
        grid = pd.DataFrame({"month": pd.date_range(live.min(), live.max(), freq="MS")})
        g = grid.merge(g, on="month", how="left").assign(country=c)
        for col in FLOW_COLS:
            if col in g:
                g[col] = g[col].fillna(0.0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["country", "month"]) if parts else out


def supply_panel(tenor_dv01: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    fx = fetch_fx(force=False)
    ten = tenor_dv01.copy()
    ten["month"] = pd.to_datetime(ten["month"])
    coupon = ten[ten.get("sec_type", pd.Series("", index=ten.index)).astype(str) != "Bill"]
    bills = ten[ten.get("sec_type", pd.Series("", index=ten.index)).astype(str) == "Bill"]

    def _sum(df, col):
        if df.empty:
            return pd.DataFrame(columns=["country", "month", col])
        return df.groupby(["country", "month"], dropna=False)[col].sum().reset_index(name=col)

    coup_dv = _sum(coupon, "dv01_usd_mn").rename(columns={"dv01_usd_mn": "coupon_dv01_usd"})
    bill_dv = _sum(bills, "dv01_usd_mn").rename(columns={"dv01_usd_mn": "bill_dv01_usd"})
    all_dv = _sum(ten, "dv01_usd_mn").rename(columns={"dv01_usd_mn": "gross_dv01_usd"})
    coup_face = _sum(coupon, "gross_mn").rename(columns={"gross_mn": "coupon_gross_mn"}) if "gross_mn" in coupon.columns else pd.DataFrame()
    if "buyback_mn" in coupon.columns and not coupon.empty:
        bb = coupon.copy()
        bb = bb[pd.to_numeric(bb["buyback_mn"], errors="coerce").fillna(0) != 0]
        if not bb.empty:
            bb_tmp = bb.copy()
            bb_tmp["gross_mn"] = pd.to_numeric(bb_tmp["buyback_mn"], errors="coerce")
            bb_dv = attach_dv01(bb_tmp, fx)
            coup_bb = _sum(bb_dv, "dv01_usd_mn").rename(columns={"dv01_usd_mn": "buyback_dv01_usd"})
        else:
            coup_bb = pd.DataFrame(columns=["country", "month", "buyback_dv01_usd"])
    else:
        coup_bb = pd.DataFrame(columns=["country", "month", "buyback_dv01_usd"])

    off_dv = _official_dv01(official, fx)
    if not off_dv.empty:
        off_m = off_dv.groupby(["country", "month"], dropna=False).agg(
            official_dv01_usd=("official_dv01_usd", "sum"),
            official_face_mn=("official_mn", "sum"),
        ).reset_index()
    else:
        off_m = pd.DataFrame(columns=["country", "month", "official_dv01_usd", "official_face_mn"])

    # Gross is the public take (Fed add-ons removed in `isin_flow`). The SOMA holdings change
    # still contains the add-ons, so they come out of the official flow too, or they net twice.
    addon_b = pd.DataFrame(columns=["country", "month", "tenor_bucket", "addon_dv01_usd"])
    if "official_take_mn" in coupon.columns and not coupon.empty:
        c = coupon.copy()
        gross = pd.to_numeric(c["gross_mn"], errors="coerce")
        take = pd.to_numeric(c["official_take_mn"], errors="coerce").fillna(0.0)
        c["addon_dv01_usd"] = (c["dv01_usd_mn"] * take / gross.where(gross > 0)).fillna(0.0)
        addon_b = c.groupby(["country", "month", "tenor_bucket"], dropna=False, observed=True)["addon_dv01_usd"].sum().reset_index()
    addon_m = addon_b.groupby(["country", "month"], as_index=False)["addon_dv01_usd"].sum()
    if not off_m.empty:
        off_m = off_m.merge(addon_m, on=["country", "month"], how="left")
        off_m["official_dv01_usd"] = off_m["official_dv01_usd"] - off_m["addon_dv01_usd"].fillna(0.0)
        off_m = off_m.drop(columns="addon_dv01_usd")

    out = all_dv.merge(coup_dv, on=["country", "month"], how="outer")
    out = out.merge(bill_dv, on=["country", "month"], how="left")
    # Outer: central-bank purchases in a month with no auction are still flow.
    out = out.merge(off_m, on=["country", "month"], how="outer")
    if not coup_face.empty:
        out = out.merge(coup_face, on=["country", "month"], how="left")
    if not coup_bb.empty:
        out = out.merge(coup_bb, on=["country", "month"], how="left")
    for col in ("coupon_dv01_usd", "bill_dv01_usd", "official_dv01_usd",
                "official_face_mn", "coupon_gross_mn", "buyback_dv01_usd"):
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    if "buyback_dv01_usd" not in out.columns:
        out["buyback_dv01_usd"] = 0.0
    # Gross coupon DV01 already excludes DFA own-holdings. Buybacks reduce net supply.
    out["net_coupon_dv01_usd"] = out["coupon_dv01_usd"] + out["buyback_dv01_usd"]
    out["private_dv01_usd"] = out["net_coupon_dv01_usd"] - out["official_dv01_usd"]
    if "coupon_gross_mn" in out.columns:
        out["absorb_pct"] = np.where(out["coupon_gross_mn"].abs() > 1,
                                     out["official_face_mn"] / out["coupon_gross_mn"], np.nan)
    else:
        out["absorb_pct"] = np.nan

    # Curve supply from coupon tenor DV01 minus official DV01 in the same buckets.
    ten_c = coupon.copy()
    if not off_dv.empty:
        off_b = off_dv.groupby(["country", "month", "tenor_bucket"], dropna=False)["official_dv01_usd"].sum().reset_index()
        ten_c = ten_c.merge(off_b, on=["country", "month", "tenor_bucket"], how="left")
        ten_c["official_dv01_usd"] = ten_c["official_dv01_usd"].fillna(0.0)
        if not addon_b.empty:
            ab = addon_b.assign(tenor_bucket=addon_b["tenor_bucket"].astype(str))
            ten_c = ten_c.assign(_b=ten_c["tenor_bucket"].astype(str)).merge(
                ab.rename(columns={"tenor_bucket": "_b"}), on=["country", "month", "_b"], how="left").drop(columns="_b")
            ten_c["official_dv01_usd"] = ten_c["official_dv01_usd"] - ten_c["addon_dv01_usd"].fillna(0.0)
            ten_c = ten_c.drop(columns="addon_dv01_usd")
    else:
        ten_c["official_dv01_usd"] = 0.0
    ten_c["private_dv01_usd"] = ten_c["dv01_usd_mn"] - ten_c["official_dv01_usd"]

    def _bucket_sum(names):
        sub = ten_c[ten_c["tenor_bucket"].isin(names)]
        return sub.groupby(["country", "month"])["private_dv01_usd"].sum()

    curve = pd.DataFrame({
        "dv01_front": _bucket_sum(FRONT),
        "dv01_belly": _bucket_sum(BELLY),
        "dv01_midlong": _bucket_sum(MIDLONG),
        "dv01_long": _bucket_sum(LONG),
        "dv01_ultra": _bucket_sum(ULTRA),
    }).reset_index()
    out = out.merge(curve, on=["country", "month"], how="left")
    out["dv01_2s10s"] = out["dv01_belly"] - out["dv01_front"]
    out["dv01_10s30s"] = out["dv01_long"] - out["dv01_belly"]
    out["dv01_5s30s"] = out["dv01_front"] + out["dv01_long"] - 2.0 * out["dv01_belly"]
    out = _monthly_grid(out)
    out["coupon_dv01_usd_12m"] = out.groupby("country")["coupon_dv01_usd"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    out["private_dv01_usd_3m"] = out.groupby("country")["private_dv01_usd"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["private_dv01_usd_12m"] = out.groupby("country")["private_dv01_usd"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    out["official_dv01_usd_3m"] = out.groupby("country")["official_dv01_usd"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["official_dv01_usd_12m"] = out.groupby("country")["official_dv01_usd"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    out["dv01_2s10s_3m"] = out.groupby("country")["dv01_2s10s"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["dv01_2s10s_12m"] = out.groupby("country")["dv01_2s10s"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    out["dv01_10s30s_3m"] = out.groupby("country")["dv01_10s30s"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["dv01_10s30s_12m"] = out.groupby("country")["dv01_10s30s"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    out["dv01_5s30s_3m"] = out.groupby("country")["dv01_5s30s"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    de = out.loc[out["country"] == "DE", ["month", "private_dv01_usd"]].rename(
        columns={"private_dv01_usd": "de_private_dv01_usd"})
    out = out.merge(de, on="month", how="left")
    out["priv_vs_de"] = out["private_dv01_usd"] - out["de_private_dv01_usd"]
    out.loc[out["country"] == "DE", "priv_vs_de"] = 0.0
    out["priv_vs_de_3m"] = out.groupby("country")["priv_vs_de"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    out["priv_vs_de_12m"] = out.groupby("country")["priv_vs_de"].transform(lambda s: s.rolling(12, min_periods=3).sum())
    out["private_dv01_z"] = out.groupby("country", group_keys=False)["private_dv01_usd"].apply(
        lambda s: rolling_z(s, 36, 18)
    )
    out["private_dv01_usd_3m_z"] = out.groupby("country", group_keys=False)["private_dv01_usd_3m"].apply(
        lambda s: rolling_z(s, 36, 18)
    )
    out["private_dv01_usd_12m_z"] = out.groupby("country", group_keys=False)["private_dv01_usd_12m"].apply(
        lambda s: rolling_z(s, 36, 18)
    )
    out["coupon_dv01_usd_12m_z"] = out.groupby("country", group_keys=False)["coupon_dv01_usd_12m"].apply(
        lambda s: rolling_z(s, 36, 18)
    )
    from issuance.fetch_soma import soma_flow
    soma_bills = _official_dv01(soma_flow(bills_only=True), fx, bills=True)
    if not soma_bills.empty:
        ob = soma_bills.groupby(["country", "month"])["official_dv01_usd"].sum().reset_index(
            name="official_bill_dv01_usd")
        out = out.merge(ob, on=["country", "month"], how="left")
    else:
        out["official_bill_dv01_usd"] = 0.0
    out["official_bill_dv01_usd"] = out["official_bill_dv01_usd"].fillna(0.0)
    return add_bill_view(out)


def add_bill_view(out: pd.DataFrame) -> pd.DataFrame:
    """`*_ib` columns: the same supply with bills in. Curve legs stay coupon: bills sit under 1y."""
    out = out.sort_values(["country", "month"]).copy()
    grp = lambda col: out.groupby("country")[col]
    out["coupon_dv01_usd_ib"] = out["coupon_dv01_usd"] + out["bill_dv01_usd"]
    out["official_dv01_usd_ib"] = out["official_dv01_usd"] + out["official_bill_dv01_usd"]
    out["private_dv01_usd_ib"] = out["private_dv01_usd"] + out["bill_dv01_usd"] - out["official_bill_dv01_usd"]
    for col in ("coupon_dv01_usd", "private_dv01_usd", "official_dv01_usd"):
        for n in (3, 12):
            out[f"{col}_{n}m_ib"] = grp(f"{col}_ib").transform(
                lambda s, n=n: s.rolling(n, min_periods=1 if n == 3 else 3).sum())
    de = out.loc[out["country"] == "DE", ["month", "private_dv01_usd_ib"]].drop_duplicates("month").rename(
        columns={"private_dv01_usd_ib": "_de_ib"})
    out = out.drop(columns=[c for c in out.columns if c == "_de_ib"]).merge(de, on="month", how="left")
    out["priv_vs_de_ib"] = out["private_dv01_usd_ib"] - out["_de_ib"]
    out.loc[out["country"] == "DE", "priv_vs_de_ib"] = 0.0
    out = out.drop(columns=["_de_ib"])
    for n in (3, 12):
        out[f"priv_vs_de_{n}m_ib"] = out.groupby("country")["priv_vs_de_ib"].transform(
            lambda s, n=n: s.rolling(n, min_periods=1 if n == 3 else 3).sum())
    for src, name in (("private_dv01_usd_ib", "private_dv01_z_ib"),
                      ("private_dv01_usd_3m_ib", "private_dv01_usd_3m_z_ib"),
                      ("private_dv01_usd_12m_ib", "private_dv01_usd_12m_z_ib"),
                      ("coupon_dv01_usd_12m_ib", "coupon_dv01_usd_12m_z_ib")):
        out[name] = out.groupby("country", group_keys=False)[src].apply(lambda s: rolling_z(s, 36, 18))
    return out


def issuance_wam(ops: pd.DataFrame, bills: bool = False) -> pd.DataFrame:
    from issuance.isin_flow import market_ops
    g = market_ops(_coupon_ops(ops, bills))
    g = g[g["amount_issued"] > 0].copy()
    if g.empty:
        return pd.DataFrame()
    g["month"] = pd.to_datetime(g["op_dt"]).dt.to_period("M").dt.to_timestamp()
    g["dur_mn_y"] = pd.to_numeric(g["amount_issued"], errors="coerce") * pd.to_numeric(g["resid_mat_yrs"], errors="coerce")
    rows = []
    for (c, m), sub in g.groupby(["country", "month"]):
        amt = float(sub["amount_issued"].sum())
        dur = float(sub["dur_mn_y"].sum())
        rows.append({
            "country": c,
            "month": m,
            "wam_issuance": _wam(sub["amount_issued"], sub["resid_mat_yrs"]),
            "coupon_issued_mn": amt,
            "duration_issued_mn_y": dur,
        })
    out = pd.DataFrame(rows).sort_values(["country", "month"])
    if out.empty:
        return out
    parts = []
    for _, sub in out.groupby("country"):
        sub = sub.sort_values("month").copy()
        for n, col in ((3, "wam_issuance_3m"), (12, "wam_issuance_12m")):
            d = sub["duration_issued_mn_y"].rolling(n, min_periods=max(2, n // 3)).sum()
            a = sub["coupon_issued_mn"].rolling(n, min_periods=max(2, n // 3)).sum()
            sub[col] = d / a.replace(0.0, np.nan)
        sub["duration_issued_3m"] = sub["duration_issued_mn_y"].rolling(3, min_periods=1).sum()
        sub["duration_issued_12m"] = sub["duration_issued_mn_y"].rolling(12, min_periods=3).sum()
        sub["wam_issuance_z"] = rolling_z(sub["wam_issuance"], 36, 18)
        sub["wam_issuance_12m_z"] = rolling_z(sub["wam_issuance_12m"], 36, 18)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def issuance_wam_tenor(ops: pd.DataFrame, bills: bool = False) -> pd.DataFrame:
    from issuance.isin_flow import market_ops
    g = market_ops(_coupon_ops(ops, bills))
    g = g[g["amount_issued"] > 0].copy()
    if g.empty:
        return pd.DataFrame()
    g["month"] = pd.to_datetime(g["op_dt"]).dt.to_period("M").dt.to_timestamp()
    rows = []
    for (c, m, b), sub in g.groupby(["country", "month", "tenor_bucket"]):
        rows.append({
            "country": c,
            "month": m,
            "tenor_bucket": b,
            "wam_issuance": _wam(sub["amount_issued"], sub["resid_mat_yrs"]),
            "coupon_issued_mn": float(sub["amount_issued"].sum()),
        })
    return pd.DataFrame(rows)


BILL_WAM_COLS = (
    "outstanding_mn", "official_mn", "private_mn", "wam_stock", "wam_public", "wam_private",
    "wam_gap", "official_share", "wam_issuance", "wam_issuance_3m", "wam_issuance_12m",
    "wam_issuance_gap", "duration_issued_mn_y", "duration_issued_3m", "duration_issued_12m",
    "wam_issuance_z", "wam_issuance_12m_z", "coupon_issued_mn",
)


def _merge_bill_wam(wam: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    """Add `*_ib` WAM columns (bills in) and the bill-only face and duration issued."""
    if wam.empty:
        return wam
    ib = wam_panel(ops, bills=True)
    iss = issuance_wam(ops, bills=True)
    if not iss.empty:
        ib = ib.merge(iss, on=["country", "month"], how="outer")
        ib["wam_issuance_gap"] = ib["wam_issuance"] - ib["wam_stock"]
    keep = [c for c in BILL_WAM_COLS if c in ib.columns]
    ib = ib[["country", "month"] + keep].rename(columns={c: c + "_ib" for c in keep})
    out = wam.merge(ib, on=["country", "month"], how="left")
    out["bill_issued_mn"] = (out["coupon_issued_mn_ib"].fillna(0) - out["coupon_issued_mn"].fillna(0)).clip(lower=0)
    out["bill_duration_mn_y"] = (out["duration_issued_mn_y_ib"].fillna(0)
                                 - out["duration_issued_mn_y"].fillna(0)).clip(lower=0)
    return out


def write_rv(ops: pd.DataFrame, tenor_dv01: pd.DataFrame, official: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from issuance.demand import write_demand
    wam = wam_panel(ops)
    iss = issuance_wam(ops)
    if not wam.empty and not iss.empty:
        extra = [c for c in iss.columns if c not in ("country", "month")]
        wam = wam.merge(iss[["country", "month"] + extra], on=["country", "month"], how="left")
        wam["wam_issuance_gap"] = wam["wam_issuance"] - wam["wam_stock"]
    wam = _merge_bill_wam(wam, ops)
    supply = supply_panel(tenor_dv01, official)
    demand = write_demand(ops)
    if not supply.empty and not demand.empty:
        keep = [c for c in ("country", "month", "btc", "btc_int", "btc_long",
                            "tail_bp", "paof_takeup",
                            "btc_z", "btc_z_int", "btc_z_long", "tail_z",
                            "cover_z_dmo", "tail_z_dmo", "n_auctions",
                            "n_auctions_int", "n_auctions_long")
                if c in demand.columns]
        supply = supply.merge(demand[keep], on=["country", "month"], how="left")
        supply["demand_z"] = supply["btc_z"]
        # Heavy gross supply and a weak cover. Same definition in every country.
        supply["imbalance_z"] = supply["coupon_dv01_usd_12m_z"] - supply["demand_z"]
        supply["imbalance_z_ib"] = supply["coupon_dv01_usd_12m_z_ib"] - supply["demand_z"]
    tenor_wam = issuance_wam_tenor(ops)
    tenor_wam_ib = issuance_wam_tenor(ops, bills=True)
    if not tenor_wam_ib.empty:
        tenor_wam_ib.to_parquet(CACHE / "rv_wam_tenor_ib.parquet", index=False)
    wam.to_parquet(CACHE / "rv_wam.parquet", index=False)
    supply.to_parquet(CACHE / "rv_supply.parquet", index=False)
    if not tenor_wam.empty:
        tenor_wam.to_parquet(CACHE / "rv_wam_tenor.parquet", index=False)
    return {"wam": wam, "supply": supply, "demand": demand, "wam_tenor": tenor_wam}
