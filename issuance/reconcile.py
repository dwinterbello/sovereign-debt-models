"""Line operations up against official DMO / agency stock and issuance totals."""

from __future__ import annotations

import json

import pandas as pd

from issuance.paths import CACHE, DERIVED
from issuance.tenor import attach_tenor

# Face-value identity: last outstanding by ISIN should match official stock.
# Tolerance is in millions of local currency. UK DMO D1A is the hard check.
TOL_MN = {
    "UK": 50.0,   # D1A vs cumulated D2.1E
    "DE": 250.0,  # Finanzagentur stock vs history (own holdings / green twins)
    "FR": 250.0,
    "IT": 500.0,  # PDF coverage is incomplete historically
    "US": 1000.0, # Fiscal Data vs MSPD (when present)
}


def _outstanding(ops: pd.DataFrame) -> pd.DataFrame:
    g = ops[ops["isin"].notna()].copy()
    g = g.sort_values("op_dt")
    g["cum"] = g.groupby("isin")["amount_issued"].cumsum()
    last = g.groupby("isin").tail(1)
    return last[["country", "isin", "sec_type", "maturity_dt", "cum", "sec_desc_long"]].rename(
        columns={"cum": "ops_outstanding_mn"}
    )


def reconcile_uk(ops: pd.DataFrame) -> dict:
    stock_path = CACHE / "stock_uk.parquet"
    if not stock_path.exists():
        return {"country": "UK", "status": "no_stock", "n_matched": 0}
    stock = pd.read_parquet(stock_path)
    live = stock[stock["amount_nominal_mn"].fillna(0) > 0].copy()
    live["isin"] = live["isin"].astype(str).str.upper()
    asof = pd.to_datetime(stock["as_of"], errors="coerce").max()
    uk_ops = ops[ops["country"] == "UK"]
    if pd.notna(asof):
        uk_ops = uk_ops[pd.to_datetime(uk_ops["op_dt"]) <= asof]
    outstanding = _outstanding(uk_ops)
    outstanding["isin"] = outstanding["isin"].astype(str).str.upper()
    # Only live gilts (maturity still ahead of last op or in D1A).
    merged = live.merge(outstanding, on="isin", how="left", suffixes=("", "_ops"))
    merged["ops_outstanding_mn"] = merged["ops_outstanding_mn"].fillna(0)
    merged["gap_mn"] = merged["ops_outstanding_mn"] - merged["amount_nominal_mn"]
    merged["abs_gap"] = merged["gap_mn"].abs()
    tol = TOL_MN["UK"]
    n = len(merged)
    n_ok = int((merged["abs_gap"] <= tol).sum())
    total_stock = float(merged["amount_nominal_mn"].sum())
    total_ops = float(merged["ops_outstanding_mn"].sum())
    return {
        "country": "UK",
        "status": "ok" if n and n_ok / n >= 0.85 else "gap",
        "n_stock": n,
        "n_matched_within_tol": n_ok,
        "match_rate": round(n_ok / n, 4) if n else 0,
        "stock_mn": round(total_stock, 1),
        "ops_outstanding_mn": round(total_ops, 1),
        "gap_mn": round(total_ops - total_stock, 1),
        "gap_pct": round((total_ops - total_stock) / total_stock, 4) if total_stock else None,
        "worst_isins": merged.nlargest(8, "abs_gap")[
            ["isin", "name", "amount_nominal_mn", "ops_outstanding_mn", "gap_mn"]
        ].to_dict("records"),
    }


def annual_vs_ops(ops: pd.DataFrame, country: str) -> pd.DataFrame:
    g = ops[(ops["country"] == country) & (ops["amount_issued"] > 0) & (~ops.get("is_forecast", False))]
    g = g.copy()
    g["year"] = pd.to_datetime(g["op_dt"]).dt.year
    return g.groupby(["year", "sec_type"], dropna=False)["amount_issued"].sum().reset_index(name="ops_mn")


def fy_uk(ops: pd.DataFrame) -> pd.DataFrame:
    """UK DMO financial year (1 Apr – 31 Mar) gilt sales, conventional vs linker. Bills excluded."""
    g = ops[(ops["country"] == "UK") & (ops["sec_type"].isin(["Nominal", "Linker"]))].copy()
    g = g[g["amount_issued"] > 0]
    dt = pd.to_datetime(g["op_dt"])
    g["fy"] = dt.dt.year.where(dt.dt.month >= 4, dt.dt.year - 1)
    out = g.groupby(["fy", "sec_type"])["amount_issued"].sum().unstack(fill_value=0)
    out["total"] = out.sum(axis=1)
    return out.reset_index()


def write_reconcile(ops: pd.DataFrame) -> dict:
    ops = attach_tenor(ops)
    report = {
        "uk_stock": reconcile_uk(ops),
        "annual": {},
        "uk_fy": fy_uk(ops).to_dict("records"),
    }
    for country in ops["country"].dropna().unique():
        annual = annual_vs_ops(ops, country)
        report["annual"][country] = annual.to_dict("records")
        annual.to_parquet(CACHE / f"annual_{country.lower()}.parquet", index=False)
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / "reconcile.json").write_text(json.dumps(report, indent=2, default=str))
    # Also a compact table for the dashboard.
    rows = []
    for country, recs in report["annual"].items():
        df = pd.DataFrame(recs)
        if df.empty:
            continue
        last = df[df["year"] >= df["year"].max() - 5]
        rows.append({
            "country": country,
            "years": int(df["year"].nunique()),
            "ops_total_mn": float(df["ops_mn"].sum()),
            "last6y_mn": float(last["ops_mn"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_parquet(CACHE / "reconcile_summary.parquet", index=False)
    report["unallocated_maturity"] = _unallocated_maturity(ops)
    report["us_raw"] = _us_vs_raw(ops)
    report["us_coupon_vs_fiscal"] = us_coupon_vs_fiscal()
    from issuance import reconcile_official

    try:
        off_df, off_meta = reconcile_official.run()
        reconcile_official.write(off_df, off_meta)
        report["official_totals"] = off_meta
    except Exception as exc:  # the gate is the CLI / pytest; the refresh records and continues
        report["official_totals"] = {"status": "error", "error": repr(exc)}
    (DERIVED / "reconcile.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _unallocated_maturity(ops: pd.DataFrame) -> dict:
    """Face with no maturity cannot enter an ISIN stock. Report it; do not drop it silently from ops."""
    if ops is None or ops.empty or "maturity_dt" not in ops.columns:
        return {}
    miss = ops[ops["maturity_dt"].isna()]
    out = {}
    for country, grp in miss.groupby(miss["country"].astype(str)):
        out[str(country)] = {
            "n": int(len(grp)),
            "amount_mn": round(float(pd.to_numeric(grp["amount_issued"], errors="coerce").sum()), 1),
        }
    return out


COUPON_GAP_TOL = 0.03


def us_coupon_vs_fiscal() -> dict:
    """US forward coupon issuance (priced lines) vs the fiscal path's coupon cash, by month and quarter.

    Both sides price the same Aug-2026 sizes from different templates: issuance from the
    announced pattern, the fiscal path from year-ago settlements. Issuance cash ≈ face +
    (coupon − yield) × 100 × DV01 (accrued and the TIPS index ratio left out); the fiscal
    side is public cash at settlement, so auction-vs-settlement month slips show up monthly
    and wash out quarterly.
    """
    from issuance.fiscal_bridge import BRIDGE

    path = CACHE / "forecast_priced.parquet"
    if not path.exists() or not BRIDGE.exists():
        return {"status": "missing_inputs"}
    p = pd.read_parquet(path)
    p = p[(p["country"] == "US") & (pd.to_numeric(p["gross_mn"], errors="coerce").fillna(0) > 0)].copy()
    # US coupons settle on the 15th (3y/10y/30y, auctioned before the 13th) or the month end
    # (20y, TIPS, 2y/5y/7y, FRN), rolled to the next Federal business day.
    from pandas.tseries.holiday import USFederalHolidayCalendar

    a = pd.to_datetime(p["op_dt"])
    mid = a.dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=14)
    settle = (a + pd.offsets.MonthEnd(0)).where(a.dt.day >= 13, mid)
    cbd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    settle = settle.map(cbd.rollforward)
    p["month"] = settle.dt.to_period("M").dt.to_timestamp()
    prem = (pd.to_numeric(p["coupon"], errors="coerce") - pd.to_numeric(p["yield"], errors="coerce")).fillna(0.0)
    p["cash_bn"] = (p["gross_mn"] + prem * 100.0 * p["dv01_local_mn"].fillna(0.0)) / 1000.0
    iss = p.groupby("month")["cash_bn"].sum()
    br = pd.read_parquet(BRIDGE).loc["unchanged"]["coupon_iss_cash_bn"]
    br.index = pd.to_datetime(br.index)
    m = pd.DataFrame({"issuance_bn": iss, "fiscal_bn": br}).dropna()
    m = m[m.index > br.index.min()]  # origin month is partial
    if m.empty:
        return {"status": "no_overlap"}
    m["gap_pct"] = m["issuance_bn"] / m["fiscal_bn"] - 1.0
    q = m.groupby(m.index.to_period("Q"))[["issuance_bn", "fiscal_bn"]].sum()
    q = q[m.groupby(m.index.to_period("Q")).size() == 3]
    q["gap_pct"] = q["issuance_bn"] / q["fiscal_bn"] - 1.0
    flag_q = q[q["gap_pct"].abs() > COUPON_GAP_TOL]
    return {
        "status": "ok" if flag_q.empty else "gap",
        "tol_pct": COUPON_GAP_TOL,
        "total_issuance_bn": round(float(m["issuance_bn"].sum()), 1),
        "total_fiscal_bn": round(float(m["fiscal_bn"].sum()), 1),
        "quarters": {str(k): {c: round(float(v), 4 if c == "gap_pct" else 1) for c, v in r.items()}
                     for k, r in q.iterrows()},
        "months_flagged": {str(k.date()): round(float(v), 4) for k, v in m["gap_pct"].items()
                           if abs(v) > COUPON_GAP_TOL},
    }


def _us_vs_raw(ops: pd.DataFrame) -> dict:
    """Ops accepted face vs the Fiscal Data raw file. TOL_MN['US'] is the fail bar."""
    path = CACHE / "us_auctions_raw.parquet"
    if not path.exists():
        return {"status": "no_raw"}
    raw = pd.read_parquet(path)
    col = next((c for c in ("total_accepted", "total_accepted_amt") if c in raw.columns), None)
    if col is None:
        return {"status": "no_accepted_column", "columns": list(raw.columns)[:12]}
    accepted = float(pd.to_numeric(raw[col], errors="coerce").fillna(0).sum()) / 1e6
    us = ops[ops["country"].astype(str) == "US"]
    ops_mn = float(pd.to_numeric(us["amount_issued"], errors="coerce").fillna(0).clip(lower=0).sum())
    gap = ops_mn - accepted
    tol = TOL_MN["US"]
    return {
        "status": "ok" if abs(gap) <= tol else "gap",
        "ops_mn": round(ops_mn, 1),
        "raw_accepted_mn": round(accepted, 1),
        "gap_mn": round(gap, 1),
        "tol_mn": tol,
    }
