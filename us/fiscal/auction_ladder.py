"""Marketable financing ladder from Treasury auction records.

Cash raised on a settlement date is the public award, not `total_accepted`:
Fiscal Data's `total_accepted` includes the Fed's SOMA add-ons, which are an
exchange of maturing holdings and never touch the TGA. Public cash is

    (total_accepted − soma_accepted) × (adj_price/100 + adj_accrued/1000)

where `adj_*` carry the TIPS index ratio. A maturing CUSIP redeems what the
public holds of it: Σ public awards over every (re)opening, less buybacks.
SOMA holdings at maturity are rolled into the same day's auctions (QT ended
2025-12-01); while the cap bound, the Fed's redeemed share was a cash outflow
not modelled here, and the DTS residual shows it.

Announced-but-not-yet-auctioned rows carry `offering_amt` and no award; the
ladder uses the offering (par, price≈1 for coupons, discount for bills via the
latest same-term high rate) and flags those rows `status='announced'`.

  PYTHONPATH=. .venv/bin/python us/fiscal/auction_ladder.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cache"
DERIVED = HERE / "derived"

URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
BUYBACK_DETAIL_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/buybacks_security_details"
)
FIELDS = (
    "record_date,cusip,security_type,security_term,auction_date,issue_date,maturity_date,"
    "announcemt_date,offering_amt,total_accepted,soma_accepted,price_per100,adj_price,"
    "adj_accrued_int_per1000,accrued_int_per1000,index_ratio_on_issue_date,ref_cpi_on_issue_date,"
    "high_discnt_rate,high_investment_rate,cash_management_bill_cmb,reopening,"
    "inflation_index_security,floating_rate,est_pub_held_mat_by_type_amt,int_rate,"
    "first_int_payment_date,dated_date"
)
START = "1979-01-01"  # full history: 1990s 30y bonds still mature inside the sample
RAW_PATH = CACHE / "auctions_full.parquet"
UA = {"User-Agent": "us-fiscal/auction-ladder (research; local)"}
PAGE = 10000
REFETCH_DAYS = 45
FRN_SPREAD = 0.0015  # typical FRN discount margin over the 13w bill index


def _get(url: str, params: dict, tries: int = 4) -> dict:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * 2**i)
    raise RuntimeError(f"Fiscal Data {url.rsplit('/', 1)[-1]}: {last}")


def _pages(url: str, params: dict) -> pd.DataFrame:
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get(url, {**params, "page[size]": PAGE, "page[number]": page, "format": "json"})
        chunk = payload.get("data") or []
        rows.extend(chunk)
        if page >= int(payload.get("meta", {}).get("total-pages") or 1) or not chunk:
            break
        page += 1
    return pd.DataFrame(rows)


def fed_holidays(start: str = "1979-01-01", end: str = "2060-12-31") -> pd.DatetimeIndex:
    """Fedwire holidays: federal holidays, Sunday ones observed Monday, Saturday ones not observed.

    pandas' federal calendar moves Saturday holidays to Friday; the Fed stays
    open that Friday (DTS publishes, e.g. 2026-07-03). Juneteenth starts 2022.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar

    hol = USFederalHolidayCalendar().holidays(start=start, end=end)
    fixed = {(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)}
    sat_moved = [(d.dayofweek == 4) and ((d + pd.Timedelta(days=1)).month, (d + pd.Timedelta(days=1)).day) in fixed
                 for d in hol]
    early_juneteenth = [(d.month == 6 and d.day in (18, 19, 20, 21) and d.year < 2022) for d in hol]
    keep = ~(np.array(sat_moved) | np.array(early_juneteenth))
    return hol[keep]


def fed_bday() -> pd.offsets.CustomBusinessDay:
    return pd.offsets.CustomBusinessDay(holidays=fed_holidays())


def roll_forward(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Payments due on a non-business day settle on the next business day."""
    bd = fed_bday()
    idx = pd.DatetimeIndex(dates)
    return pd.DatetimeIndex([d if pd.isna(d) or bd.is_on_offset(d) else d + bd for d in idx])


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"null": np.nan, "": np.nan}), errors="coerce")


def fetch_auctions(*, refresh: bool = False) -> pd.DataFrame:
    """Every auction since START incl. announced ones; incremental refresh."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        old = pd.read_parquet(RAW_PATH)
        if not refresh:
            return old
        since = (pd.Timestamp.today() - pd.Timedelta(days=REFETCH_DAYS)).strftime("%Y-%m-%d")
        try:
            new = _pages(URL, {"fields": FIELDS, "filter": f"auction_date:gte:{since}", "sort": "auction_date"})
        except RuntimeError as exc:
            print(f"   {exc}; using cached auctions")
            return old
        df = pd.concat([old[old["auction_date"].astype(str) < since], new], ignore_index=True)
    else:
        df = _pages(URL, {"fields": FIELDS, "filter": f"auction_date:gte:{START}", "sort": "auction_date"})
    df = df.drop_duplicates(["cusip", "auction_date", "issue_date"], keep="last")
    df.to_parquet(RAW_PATH, index=False)
    return df


def fetch_buyback_details(*, refresh: bool = False) -> pd.DataFrame:
    """Per-CUSIP buyback awards (par), for netting out of public outstanding."""
    path = CACHE / "buybacks_security_details.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        df = _pages(BUYBACK_DETAIL_URL, {"sort": "operation_date"})
    except RuntimeError as exc:
        if path.exists():
            return pd.read_parquet(path)
        print(f"   {exc}; buybacks not netted")
        return pd.DataFrame(columns=["cusip", "operation_date", "par_amt_accepted"])
    df.to_parquet(path, index=False)
    return df


def classify_security(row_type: str, term: str, tips: str, frn: str, cmb: str) -> str:
    if str(cmb).lower() == "yes":
        return "cmb"
    if str(row_type).lower() == "bill":
        return "bill"
    if str(tips).lower() == "yes":
        return "tips"
    if str(frn).lower() == "yes":
        return "frn"
    return "coupon"


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for c in ("auction_date", "issue_date", "maturity_date", "announcemt_date"):
        df[c] = pd.to_datetime(df[c].replace("null", np.nan), errors="coerce")
    for c in (
        "offering_amt", "total_accepted", "soma_accepted", "price_per100", "adj_price",
        "adj_accrued_int_per1000", "accrued_int_per1000", "index_ratio_on_issue_date",
        "ref_cpi_on_issue_date", "high_discnt_rate", "est_pub_held_mat_by_type_amt",
    ):
        df[c] = _num(df[c]) if c in df.columns else np.nan
    df["cls"] = [
        classify_security(t, m, i, f, c)
        for t, m, i, f, c in zip(
            df["security_type"], df["security_term"], df["inflation_index_security"],
            df["floating_rate"], df["cash_management_bill_cmb"],
        )
    ]
    df["status"] = np.where(df["total_accepted"].notna(), "awarded", "announced")
    df["public_par_bn"] = (df["total_accepted"] - df["soma_accepted"].fillna(0.0)) / 1e9
    df.loc[df["status"] == "announced", "public_par_bn"] = df["offering_amt"] / 1e9
    price = df["adj_price"].fillna(df["price_per100"])
    accrued = df["adj_accrued_int_per1000"].fillna(df["accrued_int_per1000"]).fillna(0.0)
    df["cash_per_par"] = price / 100.0 + accrued / 1000.0
    # Announced bills: price from the latest awarded same-term discount rate.
    ann_bill = (df["status"] == "announced") & df["cls"].isin(["bill", "cmb"])
    if ann_bill.any():
        last_rate = (
            df[(df["status"] == "awarded") & df["cls"].isin(["bill", "cmb"])]
            .sort_values("auction_date").groupby("security_term")["high_discnt_rate"].last()
        )
        days = (df["maturity_date"] - df["issue_date"]).dt.days
        rate = df["security_term"].map(last_rate).fillna(4.0)
        df.loc[ann_bill, "cash_per_par"] = 1.0 - rate[ann_bill] / 100.0 * days[ann_bill] / 360.0
    df["cash_per_par"] = df["cash_per_par"].fillna(1.0)
    df["public_cash_bn"] = df["public_par_bn"] * df["cash_per_par"]
    return df.dropna(subset=["issue_date"])


def _tips_redemption_ratio(df: pd.DataFrame) -> pd.Series:
    """Index ratio at maturity ≈ last ratio of the CUSIP scaled by CPI drift since."""
    t = df[(df["cls"] == "tips") & df["index_ratio_on_issue_date"].notna()].sort_values("issue_date")
    if t.empty:
        return pd.Series(dtype=float)
    latest_cpi = t["ref_cpi_on_issue_date"].dropna().iloc[-1] if t["ref_cpi_on_issue_date"].notna().any() else np.nan
    last = t.groupby("cusip").last()
    drift = (latest_cpi / last["ref_cpi_on_issue_date"]).fillna(1.0)
    return (last["index_ratio_on_issue_date"] * drift).clip(lower=1.0)  # TIPS par floor


def maturities(df: pd.DataFrame, buybacks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Public cash due per CUSIP at maturity, $bn (bills/CMBs at par, TIPS ×index ratio)."""
    by = df.groupby("cusip").agg(
        maturity_date=("maturity_date", "max"),
        cls=("cls", "first"),
        public_par_bn=("public_par_bn", "sum"),
        status=("status", lambda s: "announced" if (s == "announced").any() else "awarded"),
    )
    cusip_col = next((c for c in ("cusip_nbr", "cusip") if buybacks is not None and c in buybacks.columns), None)
    if cusip_col and not buybacks.empty:
        col = next((c for c in ("par_amt_accepted", "total_par_amt_accepted") if c in buybacks.columns), None)
        if col:
            bb = _num(buybacks[col].astype(str)).groupby(buybacks[cusip_col]).sum() / 1e9
            by["public_par_bn"] = by["public_par_bn"] - bb.reindex(by.index).fillna(0.0)
    ratio = _tips_redemption_ratio(df)
    by["redeem_cash_bn"] = by["public_par_bn"] * ratio.reindex(by.index).fillna(1.0)
    by["redeem_cash_bn"] = by["redeem_cash_bn"].clip(lower=0.0)
    by["pay_date"] = roll_forward(by["maturity_date"])
    return by.reset_index()


def ladder(df: pd.DataFrame, buybacks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Settlement-date ladder: public cash issued / redeemed by class, $bn."""
    iss = df.pivot_table(index="issue_date", columns="cls", values="public_cash_bn", aggfunc="sum").fillna(0.0)
    iss.columns = [f"iss_{c}" for c in iss.columns]
    mat = maturities(df, buybacks)
    red = mat.pivot_table(index="pay_date", columns="cls", values="redeem_cash_bn", aggfunc="sum").fillna(0.0)
    red.columns = [f"red_{c}" for c in red.columns]
    out = iss.join(red, how="outer").fillna(0.0).sort_index()
    out.index.name = "date"
    out["iss_total"] = out.filter(like="iss_").sum(axis=1)
    out["red_total"] = out.filter(like="red_").sum(axis=1)
    out["net_bills"] = out.get("iss_bill", 0) + out.get("iss_cmb", 0) - out.get("red_bill", 0) - out.get("red_cmb", 0)
    out["net_coupons"] = out["iss_total"] - out["red_total"] - out["net_bills"]
    out["net_total"] = out["iss_total"] - out["red_total"]
    ann_iss = df[df["status"] == "announced"].groupby("issue_date")["public_cash_bn"].sum()
    out["announced_iss_bn"] = ann_iss.reindex(out.index).fillna(0.0)
    return out


def coupon_schedule(df: pd.DataFrame, end: str | pd.Timestamp | None = None) -> pd.Series:
    """Scheduled semiannual coupon cash to the public by pay date, $bn (fixed-rate notes/bonds/TIPS).

    Public par per CUSIP accumulates over (re)openings from each issue date and
    drops at maturity; coupons fall on the maturity day-of-month and six months
    either side, rolled to the next business day. TIPS pay rate × par × index
    ratio (last observed ratio, held). FRNs (quarterly, floating) are excluded.
    """
    fx = df[df["cls"].isin(["coupon", "tips", "frn"])].copy()
    if fx.empty:
        return pd.Series(dtype=float, name="coupon_bn")
    fx["rate"] = pd.to_numeric(fx["int_rate"], errors="coerce") / 100.0
    end = pd.Timestamp(end) if end is not None else fx["maturity_date"].max()
    ratio = _tips_redemption_ratio(df)
    bills = df[(df["security_term"] == "13-Week") & df["high_discnt_rate"].notna()].sort_values("issue_date")
    bill_rate = bills.set_index("issue_date")["high_discnt_rate"] / 100.0
    bill_rate = bill_rate[~bill_rate.index.duplicated(keep="last")]
    rows = []
    for cusip, g in fx.groupby("cusip"):
        mat = g["maturity_date"].max()
        cls = g["cls"].iloc[0]
        rate = g["rate"].dropna().iloc[0] if g["rate"].notna().any() else np.nan
        if pd.isna(mat) or (cls != "frn" and (pd.isna(rate) or rate <= 0)):
            continue
        months = 3 if cls == "frn" else 6
        per_year = 12 // months
        eom = mat == mat + pd.offsets.MonthEnd(0)
        mult = float(ratio.get(cusip, 1.0)) if cls == "tips" else 1.0
        iss = g.groupby("issue_date")["public_par_bn"].sum().sort_index().cumsum()
        first = iss.index.min()
        k, d = 0, mat
        while d > first and d >= pd.Timestamp("2012-01-01"):
            if d <= end:
                par = iss[iss.index < d]
                if not par.empty:
                    if cls == "frn":
                        known = bill_rate[bill_rate.index <= d]
                        r = (known.iloc[-1] if not known.empty else 0.0) + FRN_SPREAD
                    else:
                        r = rate
                    rows.append((d, par.iloc[-1] * r / per_year * mult))
            k += 1
            d = mat - pd.DateOffset(months=months * k)
            if eom:
                d = d + pd.offsets.MonthEnd(0)
    if not rows:
        return pd.Series(dtype=float, name="coupon_bn")
    s = pd.DataFrame(rows, columns=["date", "bn"])
    s["date"] = roll_forward(s["date"])
    return s.groupby("date")["bn"].sum().sort_index().rename("coupon_bn")


def last_known_settlement(df: pd.DataFrame) -> pd.Timestamp:
    """Latest issue date with an awarded or announced auction: the end of the known financing window."""
    return df["issue_date"].max()


def run(*, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    df = tidy(fetch_auctions(refresh=refresh))
    bb = fetch_buyback_details(refresh=refresh)
    lad = ladder(df, bb)
    DERIVED.mkdir(parents=True, exist_ok=True)
    lad.to_parquet(DERIVED / "auction_ladder.parquet")
    df.to_parquet(DERIVED / "auctions_tidy.parquet", index=False)
    coupon_schedule(df, end=pd.Timestamp.today() + pd.DateOffset(years=2)).to_frame().to_parquet(
        DERIVED / "coupon_schedule.parquet"
    )
    if verbose:
        print(f"Auction ladder: {len(df):,} auctions; known settlements through {last_known_settlement(df).date()}")
    return lad


if __name__ == "__main__":
    run(refresh=True)
