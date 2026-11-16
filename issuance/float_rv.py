"""Private flow DV01 as a share of private stock DV01, and the seasonal profile.

Free float is the live coupon book minus the official book (APF, SOMA, PSPP+PEPP),
bills out. The ratio is that month's private coupon DV01 divided by the DV01 of
the private stock in the same tenor bucket. Seasonality is the same DV01 by
calendar month and by week of the month (days 1–7, 8–14, 15–21, 22–28, 29–31).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.dv01 import modified_duration
from issuance.paths import CACHE
from issuance.schema import COUNTRY_CCY
from issuance.tenor import BUCKET_MID, bucket_from_years

# Keep in lockstep with schema.COUNTRY_CCY (SE → SEK added with the SE tape).
CCY = dict(COUNTRY_CCY)


def _dv01_mn(face: np.ndarray, years: np.ndarray, coupon: np.ndarray, yld: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """USD DV01, million per bp, at the coupon and yield (`issuance.pricing`)."""
    from issuance.pricing import dv01_mn

    years = np.clip(np.asarray(years, dtype=float), 0.05, 50)
    yld = np.asarray(yld, dtype=float)
    yld = np.where(np.isfinite(yld) & (yld >= -2) & (yld <= 25), yld, 3.0)
    coupon = np.asarray(coupon, dtype=float)
    coupon = np.where(np.isfinite(coupon) & (coupon >= 0) & (coupon <= 20), coupon, yld)
    return dv01_mn(face, years, coupon, yld, fx)


def _fx_on(dates: pd.Series, ccy: pd.Series) -> np.ndarray:
    # fetch_fx writes fx_daily.parquet (same panel the dashboard / headline use).
    path = CACHE / "fx_daily.parquet"
    if not path.exists() and (CACHE / "fx.parquet").exists():
        path = CACHE / "fx.parquet"
    fx = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not fx.empty:
        # Parquet may store us; ops/asof are ns — merge_asof requires identical unit.
        fx["date"] = pd.to_datetime(fx["date"]).astype("datetime64[ns]")
        fx = fx.sort_values("date")
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True)).astype("datetime64[ns]")
    ccy = pd.Series(ccy).reset_index(drop=True)
    out = np.ones(len(dates))
    fallback = {"GBP": 1.27, "EUR": 1.08, "USD": 1.0, "JPY": 0.0067, "SEK": 0.095}
    col = {"GBP": "GBPUSD", "EUR": "EURUSD", "JPY": "JPYUSD", "SEK": "SEKUSD"}
    for code, idx in ccy.groupby(ccy).groups.items():
        if code == "USD" or fx.empty or col.get(code) not in fx.columns:
            out[list(idx)] = fallback.get(code, 1.0)
            continue
        left = pd.DataFrame({"date": dates.loc[idx].to_numpy(), "_i": list(idx)}).sort_values("date")
        right = fx[["date", col[code]]].dropna().copy()
        right["date"] = right["date"].astype("datetime64[ns]")
        merged = pd.merge_asof(left, right, on="date", direction="backward")
        out[merged["_i"].to_numpy()] = merged[col[code]].fillna(fallback[code]).to_numpy()
    return out


def _stock_dv01(ops: pd.DataFrame, after: dict | None = None, curve: pd.DataFrame | None = None) -> pd.DataFrame:
    """Month-end coupon-stock DV01 by tenor, each bond at its coupon and the month's market
    yield. Official holdings are not removed yet.

    `after` maps country -> month; only months strictly after it are computed.
    """
    from issuance.pricing import curve_yield, market_curve

    if curve is None:
        curve = market_curve(ops)
    g = ops[~ops["sec_type"].astype(str).isin(["Bill", "Foreign"])].copy()
    g = g[g["isin"].notna() & g["maturity_dt"].notna() & g["op_dt"].notna()]
    g["isin"] = g["isin"].astype(str)
    uni = g.groupby(["country", "isin"], as_index=False).agg(
        issued=("amount_issued", "sum"),
        maturity_dt=("maturity_dt", "last"),
        first_op=("op_dt", "min"),
        coupon=("coupon", "last"),
        sec_type=("sec_type", "last"),
    )
    uni = uni[uni["issued"] > 1]
    rows = []
    for country, book in uni.groupby("country"):
        if country not in CCY:
            # Unknown sovereign must not abort float_share / float_share_fc for the rest.
            print(f"  float_rv: skip stock for {country} (not in COUNTRY_CCY)")
            continue
        book = book.copy()
        start = max(pd.Timestamp(book["first_op"].min()), pd.Timestamp("2012-01-01"))
        end = pd.Timestamp(g.loc[g["country"].eq(country), "op_dt"].max())
        months = pd.date_range(start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp(), freq="MS")
        if after and country in after:
            months = months[months > pd.Timestamp(after[country])]
        mat = pd.to_datetime(book["maturity_dt"])
        first = pd.to_datetime(book["first_op"])
        issued = book["issued"].to_numpy()
        coupon = book["coupon"].to_numpy(dtype=float)
        linker = book["sec_type"].astype(str).eq("Linker").to_numpy()
        for asof in months:
            live = (first <= asof) & (mat > asof)
            if not live.any():
                continue
            resid = (mat[live] - asof).dt.days.to_numpy() / 365.25
            face = issued[live.to_numpy()]
            cp = coupon[live.to_numpy()]
            buckets = [bucket_from_years(y) for y in resid]
            fx = _fx_on(pd.Series([asof] * len(resid)), pd.Series([CCY[country]] * len(resid)))
            yy = curve_yield(curve, country, linker[live.to_numpy()], asof, resid)
            dv = _dv01_mn(face, resid, cp, yy, fx)
            part = pd.DataFrame({"tenor_bucket": buckets, "stock_dv01": dv, "stock_mn": face})
            agg = part.groupby("tenor_bucket", as_index=False).sum()
            agg["country"] = country
            agg["month"] = asof
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _official_stock_dv01() -> pd.DataFrame:
    frames = []
    off_path = CACHE / "official_flow.parquet"
    if off_path.exists():
        off = pd.read_parquet(off_path)
        off = off[off["country"].isin(["DE", "FR", "IT", "ES", "JP"])]
        if "tenor_bucket" in off.columns and "official_stock_mn" in off.columns:
            off = off[off["tenor_bucket"].astype(str) != "non_central"]
            off["month"] = pd.to_datetime(off["month"])
            g = off.groupby(["country", "month", "tenor_bucket"], as_index=False)["official_stock_mn"].sum()
            frames.append(g.rename(columns={"official_stock_mn": "face"}))
    # Spain: no ISIN official book yet — park the Eurosystem PSPP+PEPP stock in a single
    # residual-maturity bucket near the published WAM (~7y) so private stock DV01 is not gross.
    ecb_path = CACHE / "ecb_qe_holdings.parquet"
    if ecb_path.exists():
        h = pd.read_parquet(ecb_path)
        h = h[h["country"].astype(str).str.upper() == "ES"].copy()
        if not h.empty:
            h["month"] = pd.to_datetime(h["month"])
            g = h.groupby("month", as_index=False).agg(
                face=("holdings_mn", "sum"), wam=("wam_yrs", "mean"))
            g["country"] = "ES"
            g["tenor_bucket"] = [bucket_from_years(float(w) if np.isfinite(w) else 7.0) for w in g["wam"]]
            frames.append(g[["country", "month", "tenor_bucket", "face"]])
    soma = CACHE / "soma_holdings.parquet"
    if soma.exists():
        h = pd.read_parquet(soma)
        if "sec_type" in h.columns:
            h = h[h["sec_type"].astype(str) != "Bill"]
        h["month"] = pd.to_datetime(h["month"])
        yrs = pd.to_numeric(h.get("resid_mat_yrs"), errors="coerce")
        h = h[(yrs > 0) & (pd.to_numeric(h["official_stock_mn"], errors="coerce") > 0)].copy()
        h["tenor_bucket"] = [bucket_from_years(y) for y in yrs.loc[h.index]]
        g = h.groupby(["month", "tenor_bucket"], as_index=False)["official_stock_mn"].sum()
        g["country"] = "US"
        frames.append(g.rename(columns={"official_stock_mn": "face"}))
    apf = CACHE / "apf_operations.parquet"
    if apf.exists():
        ops = pd.read_parquet(apf)
        if "maturity_dt" in ops.columns:
            ops["isin"] = ops["isin"].astype(str)
            ops = ops.sort_values("op_dt")
            running: dict[str, float] = {}
            mat = ops.groupby("isin")["maturity_dt"].last().to_dict()
            months = pd.date_range(ops["op_dt"].min().to_period("M").to_timestamp(),
                                   ops["op_dt"].max().to_period("M").to_timestamp(), freq="MS")
            recs = list(ops.itertuples(index=False))
            i = 0
            rows = []
            for asof in months:
                while i < len(recs) and pd.Timestamp(recs[i].op_dt) <= asof + pd.offsets.MonthEnd(0):
                    running[str(recs[i].isin)] = running.get(str(recs[i].isin), 0.0) + float(recs[i].amount)
                    i += 1
                for isin, amt in running.items():
                    mdt = mat.get(isin)
                    if amt <= 0 or pd.isna(mdt) or pd.Timestamp(mdt) <= asof:
                        continue
                    resid = (pd.Timestamp(mdt) - asof).days / 365.25
                    rows.append({"month": asof, "tenor_bucket": bucket_from_years(resid), "face": amt, "country": "UK"})
            if rows:
                frames.append(pd.DataFrame(rows).groupby(["country", "month", "tenor_bucket"], as_index=False)["face"].sum())
    if not frames:
        return pd.DataFrame(columns=["country", "month", "tenor_bucket", "official_face"])
    face = pd.concat(frames, ignore_index=True).rename(columns={"face": "official_face"})
    return face[["country", "month", "tenor_bucket", "official_face"]]


def _official_on_stock(out: pd.DataFrame) -> pd.DataFrame:
    """Official DV01 = the official share of a bucket's face times that bucket's DV01.

    Official holdings have no coupon in the sources, so they are priced as a slice of the bucket
    they sit in, on the same convention as the stock and the flow.
    """
    out = out.sort_values(["country", "tenor_bucket", "month"]).copy()
    out["official_face"] = out.groupby(["country", "tenor_bucket"])["official_face"].ffill().fillna(0.0)
    share = (out["official_face"] / out["stock_mn"].where(out["stock_mn"] > 0)).clip(0.0, 1.0).fillna(0.0)
    out["official_share"] = share
    out["official_dv01"] = out["stock_dv01"] * share
    return out


def _flow_dv01() -> pd.DataFrame:
    ten = pd.read_parquet(CACHE / "dv01_tenor.parquet")
    ten["month"] = pd.to_datetime(ten["month"])
    if "sec_type" in ten.columns:
        ten = ten[ten["sec_type"].astype(str) != "Bill"]
    if "buyback_dv01_usd" not in ten.columns:
        ten["buyback_dv01_usd"] = 0.0
    g = ten.groupby(["country", "month", "tenor_bucket"], as_index=False)[["dv01_usd_mn", "buyback_dv01_usd"]].sum()
    g = g.rename(columns={"dv01_usd_mn": "flow_dv01"})
    sup_path = CACHE / "rv_supply.parquet"
    if sup_path.exists():
        s = pd.read_parquet(sup_path)[["country", "month", "coupon_dv01_usd", "official_dv01_usd"]]
        s["month"] = pd.to_datetime(s["month"])
        s["off_share"] = s["official_dv01_usd"] / s["coupon_dv01_usd"].where(s["coupon_dv01_usd"].abs() > 1e-9)
        g = g.merge(s[["country", "month", "off_share"]], on=["country", "month"], how="left")
        g["flow_dv01"] = g["flow_dv01"] * (1.0 - g["off_share"].fillna(0.0))
        g = g.drop(columns="off_share")
    # Buybacks take duration back from the market (negative DV01).
    g["flow_dv01"] = g["flow_dv01"] + g["buyback_dv01_usd"].fillna(0.0)
    return g.drop(columns="buyback_dv01_usd")


def _onto_stock(flow: pd.DataFrame, stock: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Move flow in buckets the country's stock never holds to the nearest one it does.

    A US 30y prints at 30.0y (30y+) but ages into 20-30y within the month, so the
    stock panel has no 30y+ row and a plain merge drops the whole new issue.
    """
    held = stock.groupby("country")["tenor_bucket"].unique().to_dict()
    flow = flow.copy()

    def nearest(row) -> str:
        have = held.get(row["country"])
        if have is None or row["tenor_bucket"] in have:
            return row["tenor_bucket"]
        mid = BUCKET_MID.get(row["tenor_bucket"], 10.0)
        return min(have, key=lambda b: abs(BUCKET_MID.get(b, 10.0) - mid))

    flow["tenor_bucket"] = flow.apply(nearest, axis=1)
    return flow.groupby(["country", "month", "tenor_bucket"], as_index=False)[cols].sum()


def _season(ops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = ops[~ops["sec_type"].astype(str).isin(["Bill", "Foreign"])].copy()
    g = g[(g["amount_issued"] > 0) & g["op_dt"].notna() & (g["op_dt"] >= "2012-01-01")]
    years = pd.to_numeric(g["resid_mat_yrs"], errors="coerce")
    years = years.fillna(g["tenor_bucket"].map(BUCKET_MID))
    fx = _fx_on(g["op_dt"], g["country"].map(CCY))
    cp = pd.to_numeric(g["coupon"], errors="coerce").to_numpy()
    yy = pd.to_numeric(g["yield"], errors="coerce").to_numpy() if "yield" in g.columns else cp
    g = g.copy()
    g["dv01"] = _dv01_mn(g["amount_issued"].to_numpy(), years.to_numpy(), cp, yy, fx)
    g["month"] = g["op_dt"].dt.to_period("M").dt.to_timestamp()
    from issuance.headline import first_open_month

    open_m = first_open_month()
    cur = pd.Timestamp.today().to_period("M").to_timestamp()
    g = g[g["month"] < g["country"].map(open_m).fillna(cur)]
    day = g["op_dt"].dt.day
    g["week"] = np.where(day <= 7, 1, np.where(day <= 14, 2, np.where(day <= 21, 3, np.where(day <= 28, 4, 5))))
    g["cal_month"] = g["month"].dt.month
    # Mean monthly total over every complete month in the sample (a month with no auction in a
    # bucket counts as zero), so a month with more auctions is a heavier month.
    by_m, by_w = [], []
    for c, gc in g.groupby("country"):
        grid = pd.date_range(gc["month"].min(), gc["month"].max(), freq="MS")
        n_cal = pd.Series(grid.month).value_counts()
        m = gc.groupby(["cal_month", "tenor_bucket"], as_index=False)["dv01"].sum()
        m["dv01"] = m["dv01"] / m["cal_month"].map(n_cal)
        by_m.append(m.assign(country=c))
        w = gc.groupby(["week", "tenor_bucket"], as_index=False)["dv01"].sum()
        w["dv01"] = w["dv01"] / len(grid)
        by_w.append(w.assign(country=c))
    cols_m, cols_w = ["country", "cal_month", "tenor_bucket", "dv01"], ["country", "week", "tenor_bucket", "dv01"]
    return (pd.concat(by_m, ignore_index=True)[cols_m] if by_m else pd.DataFrame(columns=cols_m),
            pd.concat(by_w, ignore_index=True)[cols_w] if by_w else pd.DataFrame(columns=cols_w))


def write_float(ops: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from issuance.pricing import market_curve

    stock = _stock_dv01(ops, curve=market_curve(ops))
    official = _official_stock_dv01()
    flow = _onto_stock(_flow_dv01(), stock, ["flow_dv01"])
    out = _official_on_stock(stock.merge(official, on=["country", "month", "tenor_bucket"], how="left"))
    out["private_stock_dv01"] = (out["stock_dv01"] - out["official_dv01"]).clip(lower=0)
    out = out.merge(flow[["country", "month", "tenor_bucket", "flow_dv01"]], on=["country", "month", "tenor_bucket"], how="left")
    out["flow_dv01"] = out["flow_dv01"].fillna(0.0)
    out["float_share"] = np.where(out["private_stock_dv01"] > 0.05, out["flow_dv01"] / out["private_stock_dv01"], np.nan)
    out.to_parquet(CACHE / "float_share.parquet", index=False)
    # Write float_share_fc before seasonals: sovereign_cash.debt / investor require the fc
    # artifact; season parquet is optional for that board.
    latest = out.sort_values("month").groupby("country").tail(1)
    print("  float share latest", latest.groupby("country")["float_share"].sum().round(4).to_dict())
    write_float_fc(ops, out)
    month, week = _season(ops)
    base_m = month.groupby(["country", "cal_month"])["dv01"].sum().reset_index()
    grand = base_m.groupby("country")["dv01"].mean().rename("grand")
    base_m = base_m.merge(grand, on="country")
    base_m["seasonal_index"] = base_m["dv01"] / base_m["grand"]
    tenor_m = month.merge(month.groupby(["country", "tenor_bucket"])["dv01"].mean().rename("grand"), on=["country", "tenor_bucket"])
    tenor_m["seasonal_index"] = tenor_m["dv01"] / tenor_m["grand"].replace(0, np.nan)
    base_w = week.groupby(["country", "week"])["dv01"].sum().reset_index()
    base_m.to_parquet(CACHE / "season_month.parquet", index=False)
    tenor_m.to_parquet(CACHE / "season_month_tenor.parquet", index=False)
    base_w.to_parquet(CACHE / "season_week.parquet", index=False)
    return {"float": out, "season_month": base_m, "season_week": base_w}


BUYBACK_FC_MONTHS = 3


def buyback_forward(fc_tenor: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """US buybacks on the forecast: each tenor repeats its monthly average of the last quarter.

    That is Treasury's own assumption in the refunding financing estimates (liquidity support as
    in the previous calendar quarter). Other countries' buybacks are ad hoc and are not projected.
    """
    cols = ["country", "month", "tenor_bucket", "buyback_dv01_usd"]
    path = CACHE / "dv01_tenor.parquet"
    if not path.exists() or fc_tenor is None or fc_tenor.empty:
        return pd.DataFrame(columns=cols)
    t = pd.read_parquet(path)
    if "buyback_dv01_usd" not in t.columns:
        return pd.DataFrame(columns=cols)
    t = t[(t["country"] == "US") & (t["sec_type"].astype(str) != "Bill")].copy()
    t["month"] = pd.to_datetime(t["month"])
    last = t["month"].max()
    recent = t[t["month"] > last - pd.DateOffset(months=BUYBACK_FC_MONTHS)]
    per = recent.groupby(recent["tenor_bucket"].astype(str))["buyback_dv01_usd"].sum() / BUYBACK_FC_MONTHS
    months = sorted(pd.to_datetime(fc_tenor.loc[fc_tenor["country"] == "US", "month"]).unique())
    months = [m for m in months if m > last]
    rows = [{"country": "US", "month": m, "tenor_bucket": b, "buyback_dv01_usd": v * scale}
            for m in months for b, v in per.items() if v != 0]
    return pd.DataFrame(rows, columns=cols)


def write_float_fc(ops: pd.DataFrame, hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Float panel extended over the forecast months.

    Stock: the tape book plus every priced forward auction, aged month by month.
    Official stock: last bucket split, scaled by the forecast official face path.
    Private flow: forward gross by bucket less the forecast official DV01, spread
    pro rata to gross (SOMA rollover add-ons are allotted that way; EA and UK
    official flow is redemptions, near zero DV01).
    """
    if hist is None:
        hist = pd.read_parquet(CACHE / "float_share.parquet")
    hist = hist.copy()
    hist["month"] = pd.to_datetime(hist["month"])
    priced_path, tenor_path = CACHE / "forecast_priced.parquet", CACHE / "forecast_tenor.parquet"
    if not priced_path.exists() or not tenor_path.exists():
        hist.to_parquet(CACHE / "float_share_fc.parquet", index=False)
        return hist
    last = hist.groupby("country")["month"].max().to_dict()
    # A last tape month that still has forecast operations (rest of the current month, or auctions
    # the tape has not caught up with) is rebuilt as forecast: tape so far plus what is to come.
    tn = pd.read_parquet(tenor_path, columns=["country", "month"])
    tn_open = set(zip(tn["country"], pd.to_datetime(tn["month"])))
    for c, m in list(last.items()):
        if (c, m) in tn_open:
            hist = hist[~((hist["country"] == c) & (hist["month"] == m))]
            last[c] = m - pd.DateOffset(months=1)
    pr = pd.read_parquet(priced_path)
    pr = pr[pr["gross_mn"].fillna(0) > 0].copy()
    pr["op_dt"] = pd.to_datetime(pr["op_dt"])
    fwd = pd.DataFrame({
        "country": pr["country"],
        # Each forward auction is its own row: a tap adds face at the line's maturity either way.
        "isin": "fc:" + pr["isin"].astype(str) + ":" + pr.index.astype(str),
        "op_dt": pr["op_dt"],
        "maturity_dt": pr["op_dt"] + pd.to_timedelta(pr["resid_mat_yrs"].astype(float) * 365.25, unit="D"),
        "amount_issued": pr["gross_mn"].astype(float),
        "coupon": pr["coupon"] if "coupon" in pr.columns else 3.0,
        "sec_type": pr["sec_type"].astype(str) if "sec_type" in pr.columns else "Nominal",
    })
    book = pd.concat([ops[["country", "isin", "op_dt", "maturity_dt", "amount_issued", "coupon", "sec_type"]], fwd],
                     ignore_index=True)
    from issuance.pricing import market_curve

    stock = _stock_dv01(book, after=last, curve=market_curve(ops))
    if stock.empty:
        hist.to_parquet(CACHE / "float_share_fc.parquet", index=False)
        return hist

    official = hist.sort_values("month").groupby(["country", "tenor_bucket"]).tail(1)
    official = official[["country", "tenor_bucket", "official_face"]] if "official_face" in official.columns else \
        official[["country", "tenor_bucket"]].assign(official_face=0.0)
    scale = pd.DataFrame(columns=["country", "month", "scale"])
    if (CACHE / "rv_wam_fc.parquet").exists():
        w = pd.read_parquet(CACHE / "rv_wam_fc.parquet")[["country", "month", "official_mn"]]
        w["month"] = pd.to_datetime(w["month"])
        base = w[w.apply(lambda r: r["month"] <= last.get(r["country"], pd.Timestamp("1900-01-01")), axis=1)]
        base = base.sort_values("month").groupby("country")["official_mn"].last()
        w["scale"] = w["official_mn"] / w["country"].map(base)
        scale = w[["country", "month", "scale"]].drop_duplicates(["country", "month"], keep="last")
    stock = stock.merge(official, on=["country", "tenor_bucket"], how="left").merge(scale, on=["country", "month"], how="left")
    # The official book shrinks or grows with the official face path and ages with the bucket.
    stock["official_face"] = stock["official_face"].fillna(0.0) * stock["scale"].fillna(1.0).clip(lower=0)
    stock["official_share"] = (stock["official_face"] / stock["stock_mn"].where(stock["stock_mn"] > 0)).clip(0.0, 1.0).fillna(0.0)
    stock["official_dv01"] = stock["stock_dv01"] * stock["official_share"]
    stock["private_stock_dv01"] = (stock["stock_dv01"] - stock["official_dv01"]).clip(lower=0)

    ten = pd.read_parquet(tenor_path)[["country", "month", "tenor_bucket", "dv01_usd_mn"]]
    ten["month"] = pd.to_datetime(ten["month"])
    tape = pd.read_parquet(CACHE / "dv01_tenor.parquet")
    tape["month"] = pd.to_datetime(tape["month"])
    if "sec_type" in tape.columns:
        tape = tape[tape["sec_type"].astype(str) != "Bill"]
    # A forecast month can still hold auctions already on the tape (the month the forward starts in).
    tape = tape[tape.apply(lambda r: r["month"] > last.get(r["country"], pd.Timestamp("2100-01-01")), axis=1)]
    if "buyback_dv01_usd" not in tape.columns:
        tape["buyback_dv01_usd"] = 0.0
    ten = pd.concat([ten, tape[["country", "month", "tenor_bucket", "dv01_usd_mn", "buyback_dv01_usd"]],
                     buyback_forward(ten)], ignore_index=True)
    ten["dv01_usd_mn"] = ten["dv01_usd_mn"].fillna(0.0)
    ten["buyback_dv01_usd"] = ten["buyback_dv01_usd"].fillna(0.0)
    off_share = pd.DataFrame(columns=["country", "month", "off_share"])
    if (CACHE / "rv_supply_fc.parquet").exists():
        s = pd.read_parquet(CACHE / "rv_supply_fc.parquet")[["country", "month", "coupon_dv01_usd", "official_dv01_usd"]]
        s["month"] = pd.to_datetime(s["month"])
        s["off_share"] = s["official_dv01_usd"] / s["coupon_dv01_usd"].where(s["coupon_dv01_usd"].abs() > 1e-9)
        off_share = s[["country", "month", "off_share"]]
    ten = ten.merge(off_share, on=["country", "month"], how="left")
    ten["flow_dv01"] = ten["dv01_usd_mn"] * (1.0 - ten["off_share"].fillna(0.0)) + ten["buyback_dv01_usd"]
    ten = _onto_stock(ten, stock, ["flow_dv01", "dv01_usd_mn"])
    ten = ten.rename(columns={"dv01_usd_mn": "gross_dv01"})
    out = stock.merge(ten, on=["country", "month", "tenor_bucket"], how="left")
    out[["flow_dv01", "gross_dv01"]] = out[["flow_dv01", "gross_dv01"]].fillna(0.0)
    out["float_share"] = np.where(out["private_stock_dv01"] > 0.05, out["flow_dv01"] / out["private_stock_dv01"], np.nan)
    out["phase"] = "forecast"
    hist["phase"] = "history"
    both = pd.concat([hist, out.drop(columns=["scale"], errors="ignore")], ignore_index=True)
    both.to_parquet(CACHE / "float_share_fc.parquet", index=False)
    return both
