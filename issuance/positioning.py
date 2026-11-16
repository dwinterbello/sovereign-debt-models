"""Positioning and cross-border flow panels for the dashboard.

Every source becomes rows of one long table, ``pos_series.parquet``:
panel, series, label, date, value, unit. ``pos_summary.parquet`` carries
the latest print per series with a z-score and percentile against the
trailing three years. Loaders return an empty frame when their cache is
missing, so a dead source drops its panel instead of failing the stage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from issuance.paths import CACHE

ROOT = Path(__file__).resolve().parents[1]
CFTC = ROOT / "us" / "fiscal" / "derived" / "cftc_ust_positioning.parquet"
HISTORY_START = pd.Timestamp("2015-01-01")
COLS = ["panel", "series", "label", "date", "value", "unit"]


def _rows(df: pd.DataFrame, panel: str, unit: str, labels: dict[str, str], date_col: str = "date") -> pd.DataFrame:
    parts = []
    for col, label in labels.items():
        if col not in df.columns:
            continue
        s = df[[date_col, col]].dropna()
        parts.append(pd.DataFrame({
            "panel": panel, "series": col, "label": label,
            "date": pd.to_datetime(s[date_col]), "value": s[col].astype(float), "unit": unit,
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def us_cftc() -> pd.DataFrame:
    df = _read(CFTC)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df = df.reset_index() if "date" not in df.columns else df
    out = _rows(df, "us_cftc", "$mn/bp", {
        "lev_dv01_mn": "Leveraged funds", "asset_dv01_mn": "Asset managers", "dealer_dv01_mn": "Dealers",
    })
    by = _rows(df, "us_cftc_lev", "$mn/bp", {
        "lev_ust2y_dv01_mn": "TU", "lev_ust5y_dv01_mn": "FV", "lev_ust10y_dv01_mn": "TY",
        "lev_ust10y_ultra_dv01_mn": "UXY", "lev_ustbond_dv01_mn": "US", "lev_ustbond_ultra_dv01_mn": "WN",
    })
    return pd.concat([out, by], ignore_index=True)


def us_bidders() -> pd.DataFrame:
    df = _read(CACHE / "us_bidders.parquet")
    if df.empty:
        return pd.DataFrame(columns=COLS)
    parts = []
    for term in ("2y", "5y", "10y", "20y", "30y"):
        g = df[df["term"] == term].sort_values("auction_date")
        if g.empty:
            continue
        roll = pd.DataFrame({
            "date": g["auction_date"],
            f"dealer_{term}": g["dealer_share"].rolling(3, min_periods=1).mean() * 100,
            f"indirect_{term}": g["indirect_share"].rolling(3, min_periods=1).mean() * 100,
        })
        parts.append(_rows(roll, "us_dealer", "% of comp.", {f"dealer_{term}": term}))
        parts.append(_rows(roll, "us_indirect", "% of comp.", {f"indirect_{term}": term}))
    return pd.concat(parts, ignore_index=True)


def jp_weekly() -> pd.DataFrame:
    df = _read(CACHE / "jp_flows_weekly.parquet")
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df = df.sort_values("week_end")
    roll = pd.DataFrame({"date": df["week_end"]})
    for col in ("a_lt_net", "l_lt_net", "l_st_net"):
        roll[col] = df[col].rolling(4, min_periods=4).sum() / 1000
    return _rows(roll, "jp_weekly", "¥tn, 4w sum", {
        "a_lt_net": "Japan buying foreign bonds", "l_lt_net": "Foreigners buying JGBs (long)",
        "l_st_net": "Foreigners buying bills",
    })


JP_INVESTORS = {
    "life_insurers": "Life insurers", "banks": "Banks (banking a/c)", "trust_banks_trust": "Trust banks (pensions)",
    "investment_trusts": "Investment trusts", "public_sector": "Public sector",
}


def jp_investor() -> pd.DataFrame:
    df = _read(CACHE / "jp_flows_investor.parquet")
    if df.empty:
        return pd.DataFrame(columns=COLS)
    wide = df.pivot_table(index="month", columns="investor", values="net_bn_jpy").sort_index()
    roll = (wide.rolling(3, min_periods=3).sum() / 1000).reset_index().rename(columns={"month": "date"})
    return _rows(roll, "jp_investor", "¥tn, 3m sum", JP_INVESTORS)


def jp_futures() -> pd.DataFrame:
    df = _read(CACHE / "jpx_jgb_futures_investor.parquet")
    if df.empty or "investor" not in df.columns:
        return pd.DataFrame(columns=COLS)
    # The file carries options and minis, each in contracts and in yen; mixing them sums apples and yen.
    df = df[df["contract"].astype(str).eq("jgb10y") & df["unit"].astype(str).eq("yen")]
    if df.empty:
        return pd.DataFrame(columns=COLS)
    wide = df.pivot_table(index="week_end", columns="investor", values="net", aggfunc="sum").sort_index()
    roll = (wide.rolling(4, min_periods=4).sum() / 1e12).reset_index().rename(columns={"week_end": "date"})
    keep = {c: c.replace("_", " ").capitalize() for c in wide.columns
            if any(k in c for k in ("foreign", "proprietary", "trust", "insur", "bank", "individual"))}
    return _rows(roll, "jp_futures", "¥tn traded, 4w net", keep)


def boj() -> pd.DataFrame:
    df = _read(CACHE / "boj_holdings.parquet")
    if df.empty or "face_mn" not in df.columns:
        return pd.DataFrame(columns=COLS)
    tot = df.groupby("date", as_index=False)["face_mn"].sum()
    tot["boj_tn"] = tot["face_mn"] / 1e6
    return _rows(tot, "jp_boj", "¥tn face", {"boj_tn": "BoJ JGB holdings"})


XB_FLOWS = {
    ("EA", "ecb_bop_liab_govt_lt_debt"): "Euro area govt bonds",
    ("DE", "ecb_bop_liab_govt_lt_debt"): "Bunds",
    ("FR", "ecb_bop_liab_govt_lt_debt"): "OATs",
    ("IT", "ecb_bop_liab_govt_lt_debt"): "BTPs",
    ("UK", "boe_overseas_gilt_net_purchases"): "Gilts",
    ("US", "tic_net_lt_treasuries"): "Treasuries",
    ("US", "tic_net_lt_treasuries_official"): "Treasuries, official",
    ("US", "tic_net_lt_treasuries_private"): "Treasuries, private",
    ("US", "tic_net_lt_treasuries_japan"): "Treasuries, Japan",
}
XB_HOLDINGS = {
    ("US", "tic_holdings_official"): "Official",
    ("US", "tic_holdings_private"): "Private",
    ("US", "tic_holdings_japan"): "Japan",
    ("US", "tic_holdings_china"): "China",
    ("US", "tic_holdings_uk"): "UK",
}


def xborder() -> pd.DataFrame:
    df = _read(CACHE / "xborder_flows.parquet")
    if df.empty or "series" not in df.columns:
        return pd.DataFrame(columns=COLS)
    parts = []
    for (country, series, unit), g in df.groupby(["country", "series", "unit"]):
        holdings = (country, series) in XB_HOLDINGS
        label = XB_HOLDINGS.get((country, series)) or XB_FLOWS.get((country, series))
        if label is None:
            continue
        panel = "xb_holdings" if holdings else "xb_flows"
        g = g.sort_values("period")
        val = g["value"] if holdings else g["value"].rolling(3, min_periods=3).sum()
        base = "bn " + unit[3:].upper() if unit.lower().startswith("bn_") else unit
        u = base if holdings else base + ", 3m sum"
        parts.append(pd.DataFrame({
            "panel": panel, "series": f"{country}:{series}",
            "label": label,
            "date": pd.to_datetime(g["period"]), "value": val.astype(float), "unit": u,
        }).dropna(subset=["value"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


LOADERS = (us_cftc, us_bidders, jp_weekly, jp_investor, jp_futures, boj, xborder)


def build_series() -> pd.DataFrame:
    parts = []
    for fn in LOADERS:
        try:
            parts.append(fn())
        except Exception as exc:
            print(f"  positioning: {fn.__name__} failed: {exc}")
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out = out[out["date"] >= HISTORY_START]
    return out.sort_values(["panel", "series", "date"]).reset_index(drop=True)


def summarise(series: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    rows = []
    for (panel, key), g in series.groupby(["panel", "series"], sort=False):
        g = g.dropna(subset=["value"]).sort_values("date")
        if g.empty:
            continue
        last = g.iloc[-1]
        hist = g[g["date"] > last["date"] - pd.DateOffset(years=years)]["value"]
        sd = hist.std()
        rows.append({
            "panel": panel, "series": key, "label": last["label"], "unit": last["unit"],
            "date": last["date"], "value": last["value"],
            "z": (last["value"] - hist.mean()) / sd if sd and sd > 0 else None,
            "pct": (hist <= last["value"]).mean() * 100,
        })
    return pd.DataFrame(rows)


def write_positioning() -> pd.DataFrame:
    series = build_series()
    series.to_parquet(CACHE / "pos_series.parquet", index=False)
    summary = summarise(series)
    summary.to_parquet(CACHE / "pos_summary.parquet", index=False)
    return summary
