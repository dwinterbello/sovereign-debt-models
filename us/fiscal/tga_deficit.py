"""US TGA-implied deficit, daily, as a share of interpolated nominal GDP.

TGA closing balance and debt held by the public are scraped from the Daily
Treasury Statement on Fiscal Data (fiscaldata.treasury.gov) — the official
TGA publication after the Nov 2023 move from fiscal.treasury.gov.

Cash identity over a window:
    implied deficit = Δ(debt held by the public) − ΔTGA

GDP is BEA quarterly SAAR (FRED ``GDP``), linearly interpolated onto the
calendar-daily index and held flat after the last print.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cache"
DERIVED = HERE / "derived"

FISCALDATA = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
DTS_CASH = f"{FISCALDATA}/v1/accounting/dts/operating_cash_balance"
DTS_DEBT = f"{FISCALDATA}/v1/accounting/dts/debt_subject_to_limit"
DTS_FLOWS = f"{FISCALDATA}/v1/accounting/dts/deposits_withdrawals_operating_cash"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"

# Table II line items (name changed Sep-2022 when the hyphen in "III-B" dropped).
DTS_ISSUE_CATS = (
    "Public Debt Cash Issues (Table IIIB)",
    "Public Debt Cash Issues (Table III-B)",
)
DTS_REDEEM_CATS = (
    "Public Debt Cash Redemp. (Table IIIB)",
    "Public Debt Cash Redemp. (Table III-B)",
)
DTS_INTEREST_CAT = "Interest on Treasury Securities"

UA = {"User-Agent": "us-fiscal/tga-deficit (research; local)"}
PAGE_SIZE = 10000
LOOKBACK_12M = 365
LOOKBACK_3M = 91
# FRED GDP = BEA NIPA Table 1.1.5: quarterly nominal GDP, SAAR, $bn.
# Not GDPC1 (real chained). Interpolated to calendar days; last print held flat.
FRED_NOMINAL_GDP = "GDP"

# DTS account_type values that carry the TGA close (schema changed 2021–22).
TGA_CLOSE_TYPES = (
    "Treasury General Account (TGA) Closing Balance",
    "Treasury General Account (TGA)",
    "Federal Reserve Account",
)
TGA_RANK = {name: i for i, name in enumerate(TGA_CLOSE_TYPES)}


def _to_float(value) -> float:
    if value is None or value == "" or value == "null" or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float(value)


def _fiscaldata_pages(
    url: str,
    *,
    fields: str,
    cache_name: str,
    refresh: bool = False,
    sort: str = "record_date",
    extra_params: dict | None = None,
) -> pd.DataFrame:
    """Download every page of a Fiscal Data table; cache as CSV."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / cache_name
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    rows: list[dict] = []
    page = 1
    last_err: Exception | None = None
    while True:
        params = {
            "fields": fields,
            "page[size]": PAGE_SIZE,
            "page[number]": page,
            "sort": sort,
            "format": "json",
        }
        if extra_params:
            params.update(extra_params)
        try:
            r = requests.get(url, params=params, headers=UA, timeout=90)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            break
        chunk = payload.get("data") or []
        rows.extend(chunk)
        total_pages = int(payload.get("meta", {}).get("total-pages") or 1)
        if page >= total_pages or not chunk:
            break
        page += 1

    if not rows:
        if cache.exists():
            return pd.read_csv(cache)
        raise RuntimeError(f"Fiscal Data empty: {url} ({last_err})")

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def parse_tga_close_mn(dts: pd.DataFrame) -> pd.Series:
    """TGA closing balance, millions of USD, from DTS operating-cash rows."""
    need = {"record_date", "account_type"}
    missing = need - set(dts.columns)
    if missing:
        raise ValueError(f"DTS cash table missing {missing}")
    df = dts.loc[dts["account_type"].isin(TGA_CLOSE_TYPES)].copy()
    df["_rank"] = df["account_type"].map(TGA_RANK)
    close = df["close_today_bal"].map(_to_float) if "close_today_bal" in df.columns else np.nan
    open_ = df["open_today_bal"].map(_to_float) if "open_today_bal" in df.columns else np.nan
    # From 2022-04-18, close_today_bal is null; closing level lives in open_today_bal
    # on the "TGA Closing Balance" row (Fiscal Data DTS notes).
    df["_val"] = pd.Series(close, index=df.index).where(lambda s: s.notna(), open_)
    df = df.dropna(subset=["_val"])
    df["_date"] = pd.to_datetime(df["record_date"])
    df = df.sort_values(["_date", "_rank"]).drop_duplicates("_date", keep="first")
    return pd.Series(df["_val"].to_numpy(), index=df["_date"].to_numpy(), name="tga_mn").sort_index()


def parse_debt_public_mn(dts: pd.DataFrame) -> pd.Series:
    """Debt held by the public, millions of USD, from DTS Table IIIC."""
    need = {"record_date", "debt_catg", "close_today_bal"}
    missing = need - set(dts.columns)
    if missing:
        raise ValueError(f"DTS debt table missing {missing}")
    df = dts.loc[dts["debt_catg"].astype(str).str.strip() == "Debt Held by the Public"].copy()
    df["_val"] = df["close_today_bal"].map(_to_float)
    df = df.dropna(subset=["_val"])
    df["_date"] = pd.to_datetime(df["record_date"])
    df = df.sort_values("_date").drop_duplicates("_date", keep="last")
    return pd.Series(df["_val"].to_numpy(), index=df["_date"].to_numpy(), name="debt_public_mn").sort_index()


def fetch_tga_mn(*, refresh: bool = False) -> pd.Series:
    raw = _fiscaldata_pages(
        DTS_CASH,
        fields="record_date,account_type,close_today_bal,open_today_bal",
        cache_name="dts_operating_cash_balance.csv",
        refresh=refresh,
    )
    return parse_tga_close_mn(raw)


def fetch_debt_public_mn(*, refresh: bool = False) -> pd.Series:
    raw = _fiscaldata_pages(
        DTS_DEBT,
        fields="record_date,debt_catg,close_today_bal",
        cache_name="dts_debt_subject_to_limit.csv",
        refresh=refresh,
    )
    return parse_debt_public_mn(raw)


def parse_dts_cat_mn(raw: pd.DataFrame) -> pd.Series:
    """One Table II category, $mn, summed if duplicate rows share a date."""
    df = raw.copy()
    df["_val"] = df["transaction_today_amt"].map(_to_float)
    df["_date"] = pd.to_datetime(df["record_date"])
    df = df.dropna(subset=["_val", "_date"])
    return df.groupby("_date")["_val"].sum().sort_index().rename("amt_mn")


def _fetch_dts_cats_mn(cats: tuple[str, ...], cache_stem: str, *, refresh: bool = False) -> pd.Series:
    pieces = []
    for cat in cats:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in cat).strip("_")
        raw = _fiscaldata_pages(
            DTS_FLOWS,
            fields="record_date,transaction_type,transaction_catg,transaction_today_amt",
            cache_name=f"{cache_stem}_{slug}.csv",
            refresh=refresh,
            extra_params={"filter": f"transaction_catg:eq:{cat}"},
        )
        pieces.append(parse_dts_cat_mn(raw))
    out = pd.concat(pieces).groupby(level=0).sum().sort_index()
    out.name = cache_stem
    return out


def fetch_dts_public_debt_issues_mn(*, refresh: bool = False) -> pd.Series:
    return _fetch_dts_cats_mn(DTS_ISSUE_CATS, "dts_issues", refresh=refresh)


def fetch_dts_public_debt_redeem_mn(*, refresh: bool = False) -> pd.Series:
    return _fetch_dts_cats_mn(DTS_REDEEM_CATS, "dts_redeem", refresh=refresh)


def fetch_dts_interest_mn(*, refresh: bool = False) -> pd.Series:
    return _fetch_dts_cats_mn((DTS_INTEREST_CAT,), "dts_interest", refresh=refresh)


def fetch_fred(code: str, *, refresh: bool = False) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"fred_{code.lower()}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["date"])
        return df.set_index("date")["value"].sort_index().rename(code)

    text = None
    for i in range(4):
        try:
            # FRED tarpits custom / browser user agents; the requests default is served.
            r = requests.get(FRED_CSV.format(code=code), timeout=(10, 60))
            r.raise_for_status()
            text = r.text
            break
        except requests.RequestException as exc:
            print(f"FRED {code}: {exc} (attempt {i + 1}/4)")
            time.sleep(5 * 2 ** i)
    if text is None:
        if cache.exists():
            print(f"FRED {code}: unreachable — using cached {cache.name}")
            df = pd.read_csv(cache, parse_dates=["date"])
            return df.set_index("date")["value"].sort_index().rename(code)
        raise RuntimeError(f"FRED {code} unreachable and no cache")
    df = pd.read_csv(io.StringIO(text))
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    val_col = [c for c in df.columns if c != date_col][0]
    values = pd.to_numeric(df[val_col].replace(".", pd.NA), errors="coerce")
    out = pd.Series(values.to_numpy(), index=pd.to_datetime(df[date_col]), name=code).dropna().sort_index()
    pd.DataFrame({"date": out.index, "value": out.values}).to_csv(cache, index=False)
    return out


def interpolate_to_daily(quarterly: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Time-linear interpolation of a lower-frequency series onto `daily_index`.

    After the last observation, hold the last value (no growth extrapolation).
    """
    q = quarterly.dropna().sort_index()
    if q.empty:
        return pd.Series(np.nan, index=daily_index, name=quarterly.name)
    union = q.index.union(daily_index)
    return q.reindex(union).interpolate(method="time").reindex(daily_index).ffill()


def calendar_daily(s: pd.Series) -> pd.Series:
    """Business-day stocks → calendar daily via ffill (weekend = Friday close)."""
    s = s.dropna().sort_index()
    if s.empty:
        return s
    return s.asfreq("D").ffill()


def change_over_days(s: pd.Series, days: int) -> pd.Series:
    """s_t − s_{t−days} on a calendar-daily series."""
    return s - s.shift(days)


def us_fiscal_year(ts) -> int:
    """US FY ending 30 Sep: 2026-08-20 → FY2026; 2025-10-01 → FY2026."""
    t = pd.Timestamp(ts)
    return int(t.year + 1 if t.month >= 10 else t.year)


def fy_start(ts) -> pd.Timestamp:
    fy = us_fiscal_year(ts)
    return pd.Timestamp(fy - 1, 10, 1)


def fy_day_of_year(ts) -> int:
    """Oct 1 = 1. Leap FY (Feb 29 in the FY) has 366 days."""
    t = pd.Timestamp(ts)
    return int((t - fy_start(t)).days + 1)


def fy_index(idx) -> pd.Index:
    return pd.Index([us_fiscal_year(t) for t in pd.DatetimeIndex(idx)], name="fy")


def fy_pivot(s: pd.Series) -> pd.DataFrame:
    """Align a daily series on day-of-FY; one column per US fiscal year."""
    s = s.dropna().sort_index()
    if s.empty:
        return pd.DataFrame()
    parts = []
    tmp = pd.DataFrame(
        {
            "fy": [us_fiscal_year(t) for t in s.index],
            "doy": [fy_day_of_year(t) for t in s.index],
            "v": s.to_numpy(),
        }
    )
    for fy, sl in tmp.groupby("fy"):
        parts.append(pd.Series(sl["v"].to_numpy(), index=sl["doy"].to_numpy(), name=int(fy)))
    return pd.concat(parts, axis=1).sort_index()


def fytd_cumsum(s: pd.Series) -> pd.DataFrame:
    """Within-FY cumulative sum, indexed by day-of-FY, one column per US FY."""
    s = s.dropna().sort_index()
    if s.empty:
        return pd.DataFrame()
    fy = pd.Series([us_fiscal_year(t) for t in s.index], index=s.index)
    return fy_pivot(s.groupby(fy).cumsum())


def fy_md_key(month: int, day: int) -> tuple[int, int]:
    """Sort key on the FY calendar (Oct=0 … Sep=11)."""
    order = month - 10 if month >= 10 else month + 2
    return (order, day)


def fytd_through_md(s: pd.Series, month: int, day: int) -> pd.Series:
    """FYTD sum through the same FY month-day, one value per FY."""
    s = s.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    rows = []
    for fy, sl in s.groupby([us_fiscal_year(t) for t in s.index]):
        mask = [fy_md_key(t.month, t.day) <= fy_md_key(month, day) for t in sl.index]
        rows.append((int(fy), float(sl.loc[mask].sum())))
    return pd.Series({k: v for k, v in rows}).sort_index()


def _zero_fill_bn(s_mn: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if s_mn is None or s_mn.dropna().empty:
        return pd.Series(0.0, index=index)
    return (s_mn.dropna() / 1000.0).reindex(index).fillna(0.0)


def attach_dts_flows(
    df: pd.DataFrame,
    issues_mn: pd.Series | None,
    redeem_mn: pd.Series | None,
    interest_mn: pd.Series | None,
) -> pd.DataFrame:
    """Table II cash identity: fiscal = (issues − redemptions) − ΔTGA.

    Public-debt cash issues/redemptions are financing. Residual vs TGA change
    is the DTS operating cash deficit. Subtract Table II 'Interest on Treasury
    Securities' for the primary (ex-interest) flow.
    """
    issues = _zero_fill_bn(issues_mn, df.index)
    redeem = _zero_fill_bn(redeem_mn, df.index)
    interest = _zero_fill_bn(interest_mn, df.index)
    d_tga = df["tga_bn"].diff()
    df["dts_issues_bn"] = issues
    df["dts_redeem_bn"] = redeem
    df["dts_interest_1d_bn"] = interest
    df["dts_fiscal_1d_bn"] = issues - redeem - d_tga
    df["dts_primary_1d_bn"] = df["dts_fiscal_1d_bn"] - interest
    fy = fy_index(df.index)
    for src, stem in (
        ("dts_fiscal_1d_bn", "dts_fiscal"),
        ("dts_primary_1d_bn", "dts_primary"),
        ("dts_interest_1d_bn", "dts_interest"),
        ("implied_deficit_1d_bn", "implied_deficit"),
    ):
        if src not in df.columns:
            continue
        df[f"{stem}_12m_bn"] = df[src].rolling(LOOKBACK_12M, min_periods=LOOKBACK_12M).sum()
        df[f"{stem}_12m_pct_gdp"] = 100.0 * df[f"{stem}_12m_bn"] / df["gdp_saar_bn"]
        df[f"{stem}_fytd_bn"] = df[src].groupby(fy).cumsum()
        df[f"{stem}_fytd_pct_gdp"] = 100.0 * df[f"{stem}_fytd_bn"] / df["gdp_saar_bn"]
    return df


def build_panel(
    tga_mn: pd.Series,
    debt_mn: pd.Series,
    gdp_q_bn: pd.Series,
    mts_surplus_mn: pd.Series | None = None,
    dts_issues_mn: pd.Series | None = None,
    dts_redeem_mn: pd.Series | None = None,
    dts_interest_mn: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily panel: 12m/3m TGA-implied deficit as % of interpolated GDP."""
    tga = calendar_daily(tga_mn) / 1000.0  # $bn
    tga.name = "tga_bn"
    debt = calendar_daily(debt_mn) / 1000.0
    debt.name = "debt_public_bn"
    df = pd.concat([tga, debt], axis=1).dropna()
    if df.empty:
        raise RuntimeError("No overlap between TGA and debt-held-by-public")

    gdp = interpolate_to_daily(gdp_q_bn, df.index).rename("gdp_saar_bn")
    df["gdp_saar_bn"] = gdp
    df = df.dropna(subset=["gdp_saar_bn"])

    d_debt_12m = change_over_days(df["debt_public_bn"], LOOKBACK_12M)
    d_tga_12m = change_over_days(df["tga_bn"], LOOKBACK_12M)
    df["d_debt_12m_bn"] = d_debt_12m
    df["d_tga_12m_bn"] = d_tga_12m
    df["implied_deficit_12m_bn"] = d_debt_12m - d_tga_12m
    df["implied_deficit_12m_pct_gdp"] = 100.0 * df["implied_deficit_12m_bn"] / df["gdp_saar_bn"]
    df["implied_deficit_1d_bn"] = df["debt_public_bn"].diff() - df["tga_bn"].diff()
    fy = fy_index(df.index)
    df["implied_deficit_fytd_bn"] = df["implied_deficit_1d_bn"].groupby(fy).cumsum()
    df["implied_deficit_fytd_pct_gdp"] = 100.0 * df["implied_deficit_fytd_bn"] / df["gdp_saar_bn"]
    df["d_tga_1d_bn"] = df["tga_bn"].diff()
    df["d_tga_fytd_bn"] = df["d_tga_1d_bn"].groupby(fy).cumsum()

    ann_3m = LOOKBACK_12M / LOOKBACK_3M
    d_debt_3m = change_over_days(df["debt_public_bn"], LOOKBACK_3M)
    d_tga_3m = change_over_days(df["tga_bn"], LOOKBACK_3M)
    df["implied_deficit_3m_ann_bn"] = (d_debt_3m - d_tga_3m) * ann_3m
    df["implied_deficit_3m_ann_pct_gdp"] = 100.0 * df["implied_deficit_3m_ann_bn"] / df["gdp_saar_bn"]
    df["implied_deficit_3m_ann_pct_gdp_ma20"] = df["implied_deficit_3m_ann_pct_gdp"].rolling(20, min_periods=10).mean()

    if mts_surplus_mn is not None and not mts_surplus_mn.dropna().empty:
        mts_def_bn = (-mts_surplus_mn.dropna().sort_index() / 1000.0).rename("mts_deficit_mn_as_bn")
        mts_12m = mts_def_bn.rolling(12, min_periods=12).sum()
        mts_daily = mts_12m.reindex(mts_12m.index.union(df.index)).ffill().reindex(df.index)
        df["mts_deficit_12m_bn"] = mts_daily
        df["mts_deficit_12m_pct_gdp"] = 100.0 * df["mts_deficit_12m_bn"] / df["gdp_saar_bn"]

    if dts_issues_mn is not None or dts_redeem_mn is not None or dts_interest_mn is not None:
        df = attach_dts_flows(df, dts_issues_mn, dts_redeem_mn, dts_interest_mn)
    return df


def run(*, refresh: bool = False) -> pd.DataFrame:
    tga = fetch_tga_mn(refresh=refresh)
    debt = fetch_debt_public_mn(refresh=refresh)
    gdp = fetch_fred(FRED_NOMINAL_GDP, refresh=refresh)
    mts = fetch_fred("MTSDS133FMS", refresh=refresh)
    issues = fetch_dts_public_debt_issues_mn(refresh=refresh)
    redeem = fetch_dts_public_debt_redeem_mn(refresh=refresh)
    interest = fetch_dts_interest_mn(refresh=refresh)
    panel = build_panel(tga, debt, gdp, mts, issues, redeem, interest)

    DERIVED.mkdir(parents=True, exist_ok=True)
    panel = panel.rename_axis("date")
    panel.reset_index().to_csv(DERIVED / "tga_implied_deficit_daily.csv", index=False)

    latest = panel.dropna(subset=["implied_deficit_12m_pct_gdp"]).iloc[-1]
    print(
        f"TGA-implied 12m deficit {latest.name.date()}: "
        f"{latest['implied_deficit_12m_pct_gdp']:.2f}% of GDP "
        f"(${latest['implied_deficit_12m_bn']:.0f}bn)  "
        f"DTS fiscal {latest.get('dts_fiscal_12m_pct_gdp', float('nan')):.2f}%  "
        f"primary {latest.get('dts_primary_12m_pct_gdp', float('nan')):.2f}%  "
        f"TGA ${latest['tga_bn']:.0f}bn"
    )
    return panel


if __name__ == "__main__":
    run(refresh=True)
