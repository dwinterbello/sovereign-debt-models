"""Open-line history, plus the forward path from issuance.forward."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from issuance.isin_flow import market_ops
from issuance.paths import CACHE
from issuance.tenor import bucket_from_years

LOOKBACK_DAYS = {"US": 140, "UK": 540, "DE": 540, "FR": 540, "IT": 540, "ES": 540, "JP": 400}
HORIZON_DAYS = 365


def _line_stats(g: pd.DataFrame) -> dict | None:
    g = g.sort_values("op_dt")
    taps = g[g["amount_issued"] > 0]
    if taps.empty or g["isin"].iloc[0] in ("", "nan"):
        return None
    stock = float(g["amount_issued"].sum())
    gaps = taps["op_dt"].diff().dt.days.dropna()
    sizes = taps["amount_issued"]
    last = taps.iloc[-1]
    named = taps[taps["op_type"].astype(str).ne("Liquidity Enhancement")] if "op_type" in taps else taps
    name_row = named.iloc[-1] if len(named) else last
    years = last["resid_mat_yrs"]
    if pd.isna(years) and pd.notna(last["maturity_dt"]):
        years = (last["maturity_dt"] - last["op_dt"]).days / 365.25
    return {
        "isin": str(g["isin"].iloc[0]),
        "name": str(name_row.get("sec_desc_long") or "")[:48],
        "maturity_dt": last["maturity_dt"],
        "tenor_bucket": bucket_from_years(float(years)) if pd.notna(years) else "unknown",
        "sec_type": str(last.get("sec_type") or ""),
        "stock_mn": stock,
        "n_taps": int(len(taps)),
        "median_gap_days": float(gaps.median()) if len(gaps) else 90.0,
        "median_size_mn": float(sizes.median()),
        "p90_size_mn": float(sizes.quantile(0.9)),
        "last_tap": taps["op_dt"].max(),
        "peak_mn": float(g["amount_issued"].cumsum().max()),
    }


def _paint_bucket(lines: pd.DataFrame, envelope: float, asof: pd.Timestamp) -> list[dict]:
    """Place median taps on the most overdue open line without crossing the cap."""
    if lines.empty or envelope <= 1:
        return []
    cap = float(lines["peak_mn"].quantile(0.9))
    cap = max(cap, float(lines["stock_mn"].max()))
    state = lines.copy()
    state["room"] = (cap - state["stock_mn"]).clip(lower=0)
    state["due"] = (asof - state["last_tap"]).dt.days
    remaining = float(envelope)
    rows = []
    for step in range(0, HORIZON_DAYS, 14):
        if remaining <= 1:
            break
        day = asof + pd.Timedelta(days=step)
        state = state.sort_values("due", ascending=False)
        for i, row in state.iterrows():
            if remaining <= 1:
                break
            gap = max(28.0, float(row["median_gap_days"]))
            if row["due"] < 0.75 * gap or row["room"] < 0.4 * row["median_size_mn"]:
                state.at[i, "due"] = row["due"] + 14
                continue
            size = min(float(row["median_size_mn"]), float(row["room"]), remaining)
            if size < 0.4 * float(row["median_size_mn"]):
                state.at[i, "due"] = row["due"] + 14
                continue
            rows.append({
                "isin": row["isin"], "tenor_bucket": row["tenor_bucket"],
                "op_dt": day, "amount_mn": size,
            })
            state.at[i, "room"] = row["room"] - size
            state.at[i, "due"] = 0.0
            remaining -= size
        else:
            continue
    return rows


US_TERMS = (2, 3, 5, 7, 10, 20, 30)


def us_term(desc: object, sec: object) -> str | None:
    """Original-term line for a US coupon: "9-Year 10-Month" reopens the 10y."""
    m = re.search(r"(\d+)-Year(?:\s+(\d+)-Month)?", str(desc or ""))
    if not m:
        return None
    years = int(m.group(1)) + int(m.group(2) or 0) / 12
    std = min(US_TERMS, key=lambda t: abs(t - years))
    if str(sec) == "FRN":
        return "US frn"
    if str(sec) == "Linker":
        return f"US TIPS {std}y"
    return f"US {std}y"


def _us_term_bucket(key: str) -> str:
    if key == "US frn":
        return "1-3y"
    return bucket_from_years(float(key.rsplit(" ", 1)[-1].rstrip("y")))


def _us_term_path(ops: pd.DataFrame, paint: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """A US note is a new CUSIP every month, so the line is the original term, tape and forward alike."""
    first = ops.sort_values("op_dt").groupby("isin").first()
    keys = {isin: us_term(r["sec_desc"] if pd.notna(r.get("sec_desc")) else r.get("sec_desc_long"), r["sec_type"])
            for isin, r in first.iterrows()}
    hist = ops[(ops["op_dt"] >= asof - pd.Timedelta(days=730)) & (ops["amount_issued"] > 0)]
    hist = hist[["isin", "op_dt", "amount_issued"]].rename(columns={"amount_issued": "amount_mn"})
    hist["phase"], hist["source"] = "history", "tape"
    parts = [hist]
    if paint is not None and not paint.empty:
        p = paint.copy()
        p["phase"] = "paint"
        parts.append(p[["isin", "op_dt", "amount_mn", "phase", "source"]])
    path = pd.concat(parts, ignore_index=True)
    path["isin"] = path["isin"].astype(str).map(lambda s: s if s.startswith("US ") else keys.get(s))
    path = path[path["isin"].notna()]
    path["tenor_bucket"] = path["isin"].map(_us_term_bucket)
    return path


JP_TERM_NAMES = {"JP 2y": "JGB 2y (every issue)", "JP LEA": "Liquidity enhancement (every issue)"}


def _jp_key(isin: str, lea: bool) -> str:
    if lea:
        return "JP LEA"
    if isin.startswith(("JP2y#", "next JP2y")):
        return "JP 2y"
    return isin


def _jp_path(ops: pd.DataFrame, paint: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """A 2y JGB is a new issue each month, sold once, so like a US note its line is the term.
    Liquidity-enhancement taps land on dozens of old issues; they are one line of their own."""
    h = ops[(ops["op_dt"] >= asof - pd.Timedelta(days=365 * 4)) & (ops["amount_issued"] > 0)].copy()
    lea = h["op_type"].astype(str).eq("Liquidity Enhancement")
    h["isin"] = [_jp_key(i, l) for i, l in zip(h["isin"], lea)]
    term = h["isin"].isin(list(JP_TERM_NAMES))
    keep = (term & (h["op_dt"] >= asof - pd.Timedelta(days=730))) | (~term & h["isin"].isin(opened["isin"]))
    hist = h[keep][["isin", "op_dt", "amount_issued", "tenor_bucket"]].rename(columns={"amount_issued": "amount_mn"})
    hist["phase"], hist["source"] = "history", "tape"
    parts = [hist]
    if paint is not None and not paint.empty:
        p = paint.copy()
        p["isin"] = [_jp_key(str(i), s == "lea") for i, s in zip(p["isin"], p["source"])]
        p = p[p["isin"].isin(list(JP_TERM_NAMES)) | p["isin"].isin(opened["isin"]) | p["isin"].str.startswith("next ")]
        p["phase"] = "paint"
        parts.append(p[["isin", "op_dt", "amount_mn", "tenor_bucket", "phase", "source"]])
    path = pd.concat(parts, ignore_index=True)
    path.loc[path["isin"].eq("JP 2y"), "tenor_bucket"] = "1-3y"
    path.loc[path["isin"].eq("JP LEA"), "tenor_bucket"] = "LEA"
    return path


BILL_CLASSES = ((45, "Bills ≤1m"), (120, "Bills 1–4m"), (200, "Bills 4–7m"), (10_000, "Bills 7–12m"))


def _bill_forward(grid: pd.DatetimeIndex, vals: list[float]) -> list[tuple[pd.Timestamp, float]]:
    """Next 52 weeks repeat the same weeks a year earlier around today's level.

    Bill stock has a calendar: US tax-receipt paydowns in April and June, quarter-end and
    refunding swings, UK cash-management steps. A flat roll erases all of it. Each analog week is
    taken relative to the straight line from a year ago to now, so last year's rhythm carries
    forward and last year's growth does not.
    """
    last = vals[-1]
    if len(vals) <= 52:
        return [(grid[-1] + pd.Timedelta(days=7 * j), last) for j in range(1, 53)]
    base = vals[-53]
    out = []
    for j in range(1, 53):
        trend = base + (last - base) * j / 52
        shape = vals[-53 + j] / trend if trend > 0 else 1.0
        out.append((grid[-1] + pd.Timedelta(days=7 * j), last * min(max(shape, 0.0), 3.0)))
    return out


def _bill_stock(country: str, ops: pd.DataFrame, asof: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outstanding bill face by original tenor, weekly for two years, then rolled flat to the horizon."""
    b = ops[(ops["sec_type"].astype(str) == "Bill") & (ops["amount_issued"] > 0)].copy()
    if b.empty:
        return pd.DataFrame(), pd.DataFrame()
    b["start"] = pd.to_datetime(b["issue_dt"]).fillna(pd.to_datetime(b["op_dt"]))
    b["end"] = pd.to_datetime(b["maturity_dt"])
    b = b[b["end"].notna() & (b["end"] > asof - pd.Timedelta(days=365 * 3))]
    days = (b["end"] - b["start"]).dt.days
    b["cls"] = pd.cut(days, [-1] + [d for d, _ in BILL_CLASSES], labels=[n for _, n in BILL_CLASSES]).astype(str)
    grid = pd.date_range(end=asof, periods=105, freq="7D")
    rows = []
    for cls, g in b.groupby("cls"):
        start, end, amt = g["start"].to_numpy(), g["end"].to_numpy(), g["amount_issued"].to_numpy()
        vals = [float(amt[(start <= np.datetime64(d)) & (end > np.datetime64(d))].sum()) for d in grid]
        rows += [{"isin": cls, "month": d, "cumulative_mn": v, "phase": "history", "source": "tape"}
                 for d, v in zip(grid, vals)]
        rows += [{"isin": cls, "month": d, "cumulative_mn": v, "phase": "paint", "source": "seasonal"}
                 for d, v in _bill_forward(grid, vals)]
    path = pd.DataFrame(rows)
    alive = path[path["phase"] == "history"].groupby("isin")["cumulative_mn"].max()
    path = path[path["isin"].isin(alive[alive > 0].index)]
    if path.empty:
        return pd.DataFrame(), pd.DataFrame()
    path["country"] = country
    path["tenor_bucket"] = "0-1y"
    path["amount_mn"] = 0.0
    lines = pd.DataFrame({"isin": path["isin"].unique()})
    lines = lines.assign(name=lines["isin"], country=country, tenor_bucket="0-1y", sec_type="Bill", open=True,
                         stock_mn=lines["isin"].map(path[path["phase"] == "history"].groupby("isin")["cumulative_mn"].last()))
    return lines, path


def _one_country(payload: tuple[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    country, ops = payload
    asof = pd.Timestamp(ops["op_dt"].max()).normalize()
    bill_lines, bill_path = _bill_stock(country, ops, asof)
    ops = ops[ops["sec_type"].astype(str) != "Bill"].copy()
    ops = ops[ops["isin"].notna() & ops["op_dt"].notna()]
    ops["isin"] = ops["isin"].astype(str)
    stats = [s for _, g in ops.groupby("isin", sort=False) if (s := _line_stats(g))]
    if not stats:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    lines = pd.DataFrame(stats)
    lines["country"] = country
    window = LOOKBACK_DAYS.get(country, 540)
    last_tap = lines["last_tap"]
    if country == "JP":
        # Liquidity-enhancement taps reopen hundreds of off-the-run JGBs; a line
        # is open only if a regular auction sold it inside the window.
        reg = ops[ops["op_type"].astype(str).eq("Auction") & (ops["amount_issued"] > 0)]
        last_tap = lines["isin"].map(reg.groupby("isin")["op_dt"].max())
    open_mask = (
        lines["maturity_dt"].notna()
        & (lines["maturity_dt"] > asof + pd.Timedelta(days=180))
        & (lines["stock_mn"] > 1)
        & (last_tap >= asof - pd.Timedelta(days=window))
        & (lines["n_taps"] >= 1)
    )
    lines["open"] = open_mask
    lines["asof"] = asof
    opened = lines[lines["open"]].copy()
    from issuance.forward import monthly_forecast, price_paint, project
    paint = project(country, ops, opened, asof)
    priced = price_paint(paint, ops, lines)
    if country == "US":
        path = _us_term_path(ops, paint, asof)
        terms = pd.DataFrame({"isin": sorted(path["isin"].unique())})
        terms = terms.assign(name=terms["isin"], country=country, open=True,
                             tenor_bucket=terms["isin"].map(_us_term_bucket),
                             sec_type=terms["isin"].map(lambda k: "Linker" if "TIPS" in k else ("FRN" if k == "US frn" else "Nominal")))
        lines = pd.concat([lines, terms], ignore_index=True)
    elif country == "JP":
        path = _jp_path(ops, paint, opened, asof)
        terms = pd.DataFrame({"isin": list(JP_TERM_NAMES), "name": list(JP_TERM_NAMES.values())})
        terms = terms.assign(country=country, open=True, sec_type="Nominal",
                             tenor_bucket=terms["isin"].map({"JP 2y": "1-3y", "JP LEA": "LEA"}))
        lines.loc[lines["isin"].str.startswith("JP2y#"), "open"] = False
        lines = pd.concat([lines, terms], ignore_index=True)
    else:
        hist_start = asof - pd.Timedelta(days=365 * 4)
        hist = ops[(ops["isin"].isin(opened["isin"])) & (ops["op_dt"] >= hist_start) & (ops["amount_issued"] > 0)]
        hist = hist[["isin", "op_dt", "amount_issued", "tenor_bucket"]].rename(columns={"amount_issued": "amount_mn"})
        hist["phase"] = "history"
        hist["source"] = "tape"
        if not paint.empty:
            paint["phase"] = "paint"
            path = pd.concat([hist, paint], ignore_index=True)
        else:
            path = hist
    if path.empty:
        return lines, path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    path["country"] = country
    spawned = path.loc[path["isin"].astype(str).str.startswith("next "), "isin"].drop_duplicates()
    if len(spawned):
        add = path[path["isin"].isin(spawned)].groupby("isin", as_index=False).agg(
            tenor_bucket=("tenor_bucket", "first"),
        )
        add["name"] = add["isin"].map(lambda s: "new " + str(s)[len("next "):])
        add["country"] = country
        add["open"] = True
        lines = pd.concat([lines, add], ignore_index=True)
    path = path.sort_values(["isin", "op_dt"])
    path["cumulative_mn"] = path.groupby("isin")["amount_mn"].cumsum()
    # One point per auction, so a tap is a step rather than a month average.
    path["month"] = pd.to_datetime(path["op_dt"])
    path = path.groupby(["country", "isin", "tenor_bucket", "month", "phase", "source"], as_index=False).agg(
        amount_mn=("amount_mn", "sum"), cumulative_mn=("cumulative_mn", "max"),
    )
    if not bill_path.empty:
        path = pd.concat([path, bill_path[path.columns]], ignore_index=True)
        lines = pd.concat([lines, bill_lines], ignore_index=True)
    month, tenor = monthly_forecast(priced)
    return lines, path, month, tenor, priced


def write_lines(ops: pd.DataFrame, workers: int | None = None) -> dict[str, pd.DataFrame]:
    ops = market_ops(ops)
    jobs = [(c, g.copy()) for c, g in ops.groupby("country", sort=False)]
    n = workers if workers is not None else min(len(jobs), os.cpu_count() or 1)
    if n <= 1 or len(jobs) == 1:
        parts = [_one_country(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n) as pool:
            parts = list(pool.map(_one_country, jobs))
    lines = pd.concat([p[0] for p in parts if not p[0].empty], ignore_index=True)
    path = pd.concat([p[1] for p in parts if not p[1].empty], ignore_index=True)
    month = pd.concat([p[2] for p in parts if not p[2].empty], ignore_index=True)
    tenor = pd.concat([p[3] for p in parts if not p[3].empty], ignore_index=True)
    lines.to_parquet(CACHE / "lines.parquet", index=False)
    path.to_parquet(CACHE / "line_path.parquet", index=False)
    month.to_parquet(CACHE / "forecast_month.parquet", index=False)
    tenor.to_parquet(CACHE / "forecast_tenor.parquet", index=False)
    priced = [p[4] for p in parts if len(p) > 4 and not p[4].empty]
    if priced:
        pd.concat(priced, ignore_index=True).to_parquet(CACHE / "forecast_priced.parquet", index=False)
    from issuance.forward import extend_published
    extend_published(month, tenor)
    from issuance.forward import reconcile
    opened = lines[lines["open"]]
    print(f"  lines: {len(opened)} open / {len(lines)} coupon ISINs")
    allp = pd.concat(priced, ignore_index=True) if priced else pd.DataFrame(columns=["country"])
    basis = {}
    for country, g in ops.groupby("country"):
        text = reconcile(country, g, allp[allp["country"].eq(country)])
        if text:
            basis[country] = text
            print("   ", text)
    (CACHE / "forward_basis.json").write_text(json.dumps(basis, indent=1, ensure_ascii=False))
    return {"lines": lines, "path": path}
