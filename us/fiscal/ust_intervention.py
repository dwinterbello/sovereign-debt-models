"""Liquidity vs fiscal diagnostics around UST / TGA operations.

Liquidity channel
    overnight GC: SOFR − administered floor (IORB/IOER); TGCR − SOFR (specials)
    30y OTR vs OFTR: daily CMT (OTR-tilted) vs GSW fitted off-the-run par;
                     auction stop-out vs GSW on 30y Bond sale days
    buybacks: Fiscal Data ``operation_type`` is Treasury's own label
              (Liquidity Support vs Cash Management)

Fiscal channel
    interest expense on public issues / interpolated GDP
    NIPA federal interest SAAR / GDP
    average coupon on marketable debt
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tga_deficit import (
    CACHE,
    DERIVED,
    FRED_NOMINAL_GDP,
    UA,
    _fiscaldata_pages,
    _to_float,
    fetch_fred,
    interpolate_to_daily,
)

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
NYFED_SECURED = "https://markets.newyorkfed.org/api/rates/secured/{product}/search.json"
INTEREST_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/interest_expense"
AVG_RATE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
BUYBACKS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/buybacks_operations"
AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

BESSENT_START = pd.Timestamp("2025-01-28")
YELLEN_BUYBACKS = pd.Timestamp("2024-05-29")


def fetch_gsw_par_30y(*, refresh: bool = False) -> pd.Series:
    """GSW Svensson fitted 30y par yield (off-the-run spline), coupon-equivalent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "gsw_svenpy30.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["date"])
        return df.set_index("date")["SVENPY30"].sort_index()

    r = requests.get(GSW_URL, headers=UA, timeout=120)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), skiprows=9)
    raw["Date"] = pd.to_datetime(raw["Date"])
    s = pd.to_numeric(raw["SVENPY30"], errors="coerce")
    out = pd.Series(s.to_numpy(), index=raw["Date"], name="SVENPY30").dropna().sort_index()
    pd.DataFrame({"date": out.index, "SVENPY30": out.values}).to_csv(cache, index=False)
    return out


def fetch_nyfed_secured(product: str, *, refresh: bool = False) -> pd.Series:
    """NY Fed secured overnight rate (sofr / tgcr), percent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"nyfed_{product}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["date"])
        return df.set_index("date")["value"].sort_index().rename(product.upper())

    r = requests.get(
        NYFED_SECURED.format(product=product),
        params={"startDate": "2018-04-02", "endDate": pd.Timestamp.today().strftime("%Y-%m-%d")},
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json().get("refRates") or []
    rec = [
        {"date": pd.to_datetime(x["effectiveDate"]), "value": float(x["percentRate"])}
        for x in rows
        if x.get("percentRate") is not None
    ]
    out = pd.DataFrame(rec).drop_duplicates("date").set_index("date")["value"].sort_index().rename(product.upper())
    pd.DataFrame({"date": out.index, "value": out.values}).to_csv(cache, index=False)
    return out


def administered_floor(*, refresh: bool = False) -> pd.Series:
    """IORB from Jul-2021; IOER before that if FRED still serves it."""
    iorb = fetch_fred("IORB", refresh=refresh)
    try:
        ioer = fetch_fred("IOER", refresh=refresh)
    except Exception:  # noqa: BLE001
        ioer = pd.Series(dtype=float, name="IOER")
    floor = iorb.combine_first(ioer).sort_index().rename("admin_floor")
    return floor


def parse_interest_public_bn(raw: pd.DataFrame) -> pd.Series:
    """Monthly interest expense on public issues, $bn."""
    df = raw.copy()
    cat = df["expense_catg_desc"].astype(str)
    pub = df.loc[cat.str.contains("PUBLIC ISSUES", case=False, na=False)]
    amt = pub["month_expense_amt"].map(_to_float)
    g = amt.groupby(pd.to_datetime(pub["record_date"])).sum() / 1e9
    return g.sort_index().rename("interest_public_bn")


def fetch_interest_public_bn(*, refresh: bool = False) -> pd.Series:
    raw = _fiscaldata_pages(
        INTEREST_URL,
        fields="record_date,expense_catg_desc,expense_group_desc,expense_type_desc,month_expense_amt",
        cache_name="interest_expense.csv",
        refresh=refresh,
    )
    return parse_interest_public_bn(raw)


def fetch_avg_marketable_rate(*, refresh: bool = False) -> pd.Series:
    raw = _fiscaldata_pages(
        AVG_RATE_URL,
        fields="record_date,security_type_desc,security_desc,avg_interest_rate_amt",
        cache_name="avg_interest_rates.csv",
        refresh=refresh,
    )
    m = raw.loc[raw["security_desc"].astype(str).str.strip() == "Total Marketable"].copy()
    s = m["avg_interest_rate_amt"].map(_to_float)
    out = pd.Series(s.to_numpy(), index=pd.to_datetime(m["record_date"]), name="avg_mkt_rate").dropna().sort_index()
    return out[~out.index.duplicated(keep="last")]


def parse_buybacks(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["operation_date"] = pd.to_datetime(df["operation_date"])
    df["accepted_bn"] = df["total_par_amt_accepted"].map(_to_float) / 1e9
    df["offered_bn"] = df["total_par_amt_offered"].map(_to_float) / 1e9
    df["operation_type"] = df["operation_type"].replace({"null": np.nan, None: np.nan})
    df["maturity_bucket"] = df["maturity_bucket"].replace({"null": np.nan, None: np.nan})
    return df.dropna(subset=["operation_date"])


def fetch_buybacks(*, refresh: bool = False) -> pd.DataFrame:
    raw = _fiscaldata_pages(
        BUYBACKS_URL,
        fields=(
            "operation_date,operation_type,security_type,maturity_bucket,"
            "total_par_amt_accepted,total_par_amt_offered"
        ),
        cache_name="buybacks_operations.csv",
        refresh=refresh,
        sort="operation_date",
    )
    return parse_buybacks(raw)


def fetch_ust_30y_auctions(*, refresh: bool = False) -> pd.DataFrame:
    raw = _fiscaldata_pages(
        AUCTIONS_URL,
        fields="auction_date,issue_date,cusip,security_type,security_term,high_yield,avg_med_yield,offering_amt",
        cache_name="auctions_30y.csv",
        refresh=refresh,
        sort="-auction_date",
        extra_params={"filter": "security_term:eq:30-Year,security_type:eq:Bond"},
    )
    df = raw.copy()
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df["high_yield"] = df["high_yield"].map(_to_float)
    df["avg_med_yield"] = df["avg_med_yield"].map(_to_float)
    df["offering_bn"] = df["offering_amt"].map(_to_float) / 1e9
    return df.dropna(subset=["auction_date", "high_yield"]).sort_values("auction_date")


def monthly_buybacks(buybacks: pd.DataFrame) -> pd.DataFrame:
    """Month-end sums: cash-management front-end vs liquidity-support long end."""
    b = buybacks.dropna(subset=["accepted_bn"]).copy()
    b["month"] = b["operation_date"].dt.to_period("M").dt.to_timestamp("M")
    b["cm_front_bn"] = np.where(b["operation_type"] == "Cash Management", b["accepted_bn"], 0.0)
    b["liq_20_30_bn"] = np.where(
        (b["operation_type"] == "Liquidity Support") & (b["maturity_bucket"] == "20Y to 30Y"),
        b["accepted_bn"],
        0.0,
    )
    b["liq_10_20_bn"] = np.where(
        (b["operation_type"] == "Liquidity Support") & (b["maturity_bucket"] == "10Y to 20Y"),
        b["accepted_bn"],
        0.0,
    )
    b["liq_other_bn"] = np.where(
        (b["operation_type"] == "Liquidity Support")
        & (~b["maturity_bucket"].isin(["20Y to 30Y", "10Y to 20Y"])),
        b["accepted_bn"],
        0.0,
    )
    g = b.groupby("month")[["cm_front_bn", "liq_20_30_bn", "liq_10_20_bn", "liq_other_bn"]].sum()
    return g.sort_index()


def build_diagnostic(
    *,
    gdp_q_bn: pd.Series,
    gsw_30: pd.Series,
    dgs30: pd.Series,
    sofr: pd.Series,
    tgcr: pd.Series,
    floor: pd.Series,
    rrp: pd.Series,
    interest_m_bn: pd.Series,
    nipa_interest_q_bn: pd.Series,
    avg_rate: pd.Series,
    auctions_30: pd.DataFrame,
    buybacks_m: pd.DataFrame,
) -> pd.DataFrame:
    """Daily panel of liquidity vs fiscal gauges."""
    idx = dgs30.dropna().index.union(gsw_30.dropna().index).sort_values()
    df = pd.DataFrame(index=idx)
    df["dgs30"] = dgs30.reindex(idx)
    df["gsw_par_30"] = gsw_30.reindex(idx).ffill()
    df["dgs30"] = df["dgs30"].ffill()
    # OTR richness, bp: fitted OFTR par minus CMT (CMT is actively-traded / OTR-tilted).
    # Positive = OTR/CMT rich to OFTR spline (liquidity premium).
    df["otr_richness_30_bp"] = 100.0 * (df["gsw_par_30"] - df["dgs30"])

    df["sofr"] = sofr.reindex(idx)
    df["tgcr"] = tgcr.reindex(idx)
    df["admin_floor"] = floor.reindex(idx).ffill()
    df["sofr_iorb_bp"] = 100.0 * (df["sofr"] - df["admin_floor"])
    df["sofr_tgcr_bp"] = 100.0 * (df["sofr"] - df["tgcr"])  # specials: SOFR includes specials, TGCR is GC
    df["rrp_bn"] = rrp.reindex(idx).ffill()

    gdp = interpolate_to_daily(gdp_q_bn, df.index)
    df["gdp_saar_bn"] = gdp
    nipa = interpolate_to_daily(nipa_interest_q_bn, df.index)
    df["nipa_interest_saar_bn"] = nipa
    df["nipa_interest_pct_gdp"] = 100.0 * df["nipa_interest_saar_bn"] / df["gdp_saar_bn"]

    if interest_m_bn is None or interest_m_bn.empty:
        df["interest_public_12m_bn"] = np.nan
        df["interest_public_12m_pct_gdp"] = np.nan
    else:
        int_12m = interest_m_bn.rolling(12, min_periods=12).sum()
        int_d = int_12m.reindex(int_12m.index.union(df.index)).ffill().reindex(df.index)
        df["interest_public_12m_bn"] = int_d
        df["interest_public_12m_pct_gdp"] = 100.0 * df["interest_public_12m_bn"] / df["gdp_saar_bn"]

    if avg_rate is None or avg_rate.empty:
        df["avg_mkt_rate"] = np.nan
    else:
        df["avg_mkt_rate"] = avg_rate.reindex(avg_rate.index.union(df.index)).ffill().reindex(df.index)

    if auctions_30 is None or auctions_30.empty or "auction_date" not in auctions_30.columns:
        df["ust30_auction_vs_gsw_bp"] = np.nan
    else:
        a = auctions_30.dropna(subset=["auction_date", "high_yield"]).set_index("auction_date").sort_index()
        a["gsw_par_30"] = gsw_30.reindex(a.index, method="ffill")
        a["auction_vs_gsw_bp"] = 100.0 * (a["high_yield"] - a["gsw_par_30"])
        # Drop data errors / non-comparable lots (e.g. stray 2y-style yields on a 30y CUSIP).
        a.loc[a["auction_vs_gsw_bp"].abs() > 50, "auction_vs_gsw_bp"] = np.nan
        df["ust30_auction_vs_gsw_bp"] = a["auction_vs_gsw_bp"].reindex(df.index)

    for col in ("cm_front_bn", "liq_20_30_bn", "liq_10_20_bn", "liq_other_bn"):
        if col in buybacks_m.columns:
            s = buybacks_m[col].copy()
            s.index = pd.to_datetime(s.index)
            df[col] = s.reindex(df.index)
    return df


def run(*, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gdp = fetch_fred(FRED_NOMINAL_GDP, refresh=refresh)
    gsw = fetch_gsw_par_30y(refresh=refresh)
    dgs30 = fetch_fred("DGS30", refresh=refresh)
    sofr = fetch_nyfed_secured("sofr", refresh=refresh)
    tgcr = fetch_nyfed_secured("tgcr", refresh=refresh)
    floor = administered_floor(refresh=refresh)
    rrp = fetch_fred("RRPONTSYD", refresh=refresh)
    interest = fetch_interest_public_bn(refresh=refresh)
    nipa = fetch_fred("A091RC1Q027SBEA", refresh=refresh)
    avg = fetch_avg_marketable_rate(refresh=refresh)
    auctions = fetch_ust_30y_auctions(refresh=refresh)
    buybacks = fetch_buybacks(refresh=refresh)
    buybacks_m = monthly_buybacks(buybacks)

    panel = build_diagnostic(
        gdp_q_bn=gdp,
        gsw_30=gsw,
        dgs30=dgs30,
        sofr=sofr,
        tgcr=tgcr,
        floor=floor,
        rrp=rrp,
        interest_m_bn=interest,
        nipa_interest_q_bn=nipa,
        avg_rate=avg,
        auctions_30=auctions,
        buybacks_m=buybacks_m,
    )

    DERIVED.mkdir(parents=True, exist_ok=True)
    panel.rename_axis("date").reset_index().to_csv(DERIVED / "ust_liquidity_fiscal_daily.csv", index=False)
    buybacks.to_csv(DERIVED / "buybacks_operations.csv", index=False)
    buybacks_m.rename_axis("month").reset_index().to_csv(DERIVED / "buybacks_monthly.csv", index=False)

    last = panel.dropna(subset=["sofr_iorb_bp", "nipa_interest_pct_gdp"], how="all")
    row = last.iloc[-1]
    print(
        f"Diagnostic {row.name.date()}: SOFR−floor {row.get('sofr_iorb_bp', np.nan):.1f}bp  "
        f"30y OTR richness {row.get('otr_richness_30_bp', np.nan):.1f}bp  "
        f"NIPA interest {row.get('nipa_interest_pct_gdp', np.nan):.2f}% GDP"
    )
    return panel, buybacks, buybacks_m


if __name__ == "__main__":
    run(refresh=True)
