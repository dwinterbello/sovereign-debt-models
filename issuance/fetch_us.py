"""US Treasury auction results via Fiscal Data (CUSIP → ISIN)."""

from __future__ import annotations

import time

import pandas as pd
import requests

from issuance.http import FetchError, session
from issuance.paths import CACHE
from issuance.schema import coerce_ops
from issuance.tenor import attach_tenor

URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
FIELDS = (
    "record_date,cusip,security_type,security_term,auction_date,issue_date,"
    "maturity_date,offering_amt,total_accepted,bid_to_cover_ratio,"
    "price_per100,high_yield,int_rate,reopening,inflation_index_security,floating_rate,soma_accepted"
)


def _isin(cusip: object) -> object:
    text = str(cusip).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return pd.NA
    core = text.replace(" ", "")
    if len(core) >= 8:
        return "US" + core[:9]
    return pd.NA


def _sec_type(row: pd.Series) -> str:
    if str(row.get("inflation_index_security", "")).lower() in {"y", "yes", "true", "1"}:
        return "Linker"
    if str(row.get("floating_rate", "")).lower() in {"y", "yes", "true", "1"}:
        return "FRN"
    stype = str(row.get("security_type", "")).lower()
    if "bill" in stype:
        return "Bill"
    return "Nominal"


def _pages(force: bool = False) -> pd.DataFrame:
    cached = CACHE / "us_auctions_raw.parquet"
    if cached.exists() and not force:
        return pd.read_parquet(cached)
    rows = []
    page = 1
    sess = session()
    while True:
        params = {
            "fields": FIELDS,
            "page[number]": page,
            "page[size]": 10000,
            "sort": "auction_date",
        }
        resp = sess.get(URL, params=params, timeout=180)
        if resp.status_code == 400 and page == 1:
            resp = sess.get(URL, params={"page[number]": page, "page[size]": 10000, "sort": "auction_date"}, timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        if not data:
            break
        rows.extend(data)
        meta = payload.get("meta") or {}
        total_count = int(meta.get("total-count") or 0)
        total_pages = int(meta.get("total-pages") or 1)
        print(f"  US auctions page {page}/{total_pages} (+{len(data)}, have {len(rows)}/{total_count or '?'})")
        if total_count and len(rows) >= total_count:
            break
        if len(data) < int(params["page[size]"]):
            break
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.15)
    if not rows:
        raise FetchError("Fiscal Data auctions_query returned no rows")
    raw = pd.DataFrame(rows)
    raw.to_parquet(cached, index=False)
    return raw


TD_URL = "https://www.treasurydirect.gov/TA_WS/securities/auctioned"
TD_DAYS = 120


def _td_recent(days: int = TD_DAYS) -> pd.DataFrame:
    """TreasuryDirect auction results for the last ``days``, in Fiscal Data's column layout.

    Fiscal Data's auctions_query can stall for weeks (Oct-2026: no results after 15 Oct, no rows
    after 22 Oct) while TreasuryDirect, the primary source, carries every result the same day.
    total_accepted = awarded inside + outside the offering (SOMA add-ons), as Fiscal Data books it.
    Bills keep high_yield null, as Fiscal Data does."""
    resp = session().get(TD_URL, params={"format": "json", "days": int(days)}, timeout=120)
    resp.raise_for_status()
    rows = []
    for x in resp.json() or []:
        auc = str(x.get("competitiveAuctionCloseDate") or x.get("auctionDate") or "")[:10]
        inside = pd.to_numeric(x.get("totalParAwardedInsideOfOfferingAmount"), errors="coerce")
        outside = pd.to_numeric(x.get("totalParAwardedOutsideOfOfferingAmount"), errors="coerce")
        if not auc or pd.isna(inside) or inside <= 0:
            continue  # announced, not yet auctioned
        bill = str(x.get("securityType")) in ("Bill", "CMB")
        rows.append({
            "record_date": str(x.get("issueDate") or "")[:10],
            "cusip": x.get("cusip"),
            "security_type": x.get("securityType"),
            "security_term": x.get("securityTerm"),
            "auction_date": auc,
            "issue_date": str(x.get("issueDate") or "")[:10],
            "maturity_date": str(x.get("maturityDate") or "")[:10],
            "offering_amt": x.get("offeringAmount"),
            "total_accepted": str(int(inside + (0 if pd.isna(outside) else outside))),
            "bid_to_cover_ratio": x.get("bidToCoverRatio"),
            "price_per100": x.get("highPrice") or x.get("pricePer100"),
            "high_yield": "null" if bill else (x.get("highYield") or x.get("highDiscountMargin") or "null"),
            "int_rate": "null" if bill else (x.get("interestRate") or x.get("spread") or "null"),
            "reopening": x.get("reopening"),
            "inflation_index_security": x.get("inflationProtected") or "No",
            "floating_rate": x.get("floatingRate") or "No",
            "soma_accepted": x.get("somaNoncompetitiveParAwarded") or "0",
        })
    return pd.DataFrame(rows)


def _top_up(raw: pd.DataFrame) -> pd.DataFrame:
    """Fill auctions Fiscal Data has not published (or published without results) from TreasuryDirect."""
    try:
        td = _td_recent()
    except Exception as exc:  # noqa: BLE001 — Fiscal Data alone still ships
        print(f"  US TreasuryDirect top-up failed: {exc}")
        return raw
    if td.empty:
        return raw
    acc = pd.to_numeric(raw.get("total_accepted"), errors="coerce")
    have = set(zip(raw.loc[acc.notna() & (acc > 0), "cusip"], raw.loc[acc.notna() & (acc > 0), "auction_date"]))
    add = td[[k not in have for k in zip(td["cusip"], td["auction_date"])]]
    if add.empty:
        print("  US TreasuryDirect top-up: Fiscal Data is current")
        return raw
    stale = raw.set_index(["cusip", "auction_date"]).index.isin(list(zip(add["cusip"], add["auction_date"])))
    out = pd.concat([raw[~stale], add[[c for c in raw.columns if c in add.columns]]], ignore_index=True)
    print(f"  US TreasuryDirect top-up: {len(add)} auctions Fiscal Data lacks "
          f"({add['auction_date'].min()} → {add['auction_date'].max()})")
    return out.sort_values(["auction_date", "cusip"]).reset_index(drop=True)


BUYBACK_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/buybacks_security_details"


def _buybacks(force: bool = False) -> pd.DataFrame:
    """Treasury buybacks by CUSIP (liquidity support and cash management), since May 2024."""
    cached = CACHE / "us_buybacks_raw.parquet"
    if cached.exists() and not force:
        return pd.read_parquet(cached)
    rows, page, sess = [], 1, session()
    while True:
        resp = sess.get(BUYBACK_URL, params={"page[number]": page, "page[size]": 10000, "sort": "operation_date"},
                        timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        rows.extend(data)
        if not data or page >= int((payload.get("meta") or {}).get("total-pages") or 1):
            break
        page += 1
    raw = pd.DataFrame(rows)
    if not raw.empty:
        raw.to_parquet(cached, index=False)
    return raw


def _buyback_ops(raw: pd.DataFrame, auctions: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    par = pd.to_numeric(raw.get("par_amt_accepted"), errors="coerce")
    b = pd.DataFrame({
        "op_dt": pd.to_datetime(raw.get("operation_date"), errors="coerce"),
        "maturity_dt": pd.to_datetime(raw.get("maturity_date"), errors="coerce"),
        "cusip": raw.get("cusip_nbr"),
        "coupon": pd.to_numeric(raw.get("coupon_rate_pct"), errors="coerce"),
        "price": pd.to_numeric(raw.get("weighted_avg_accepted_price"), errors="coerce"),
        "amount_issued": -par / 1e6,
    })
    b = b[par.fillna(0) > 0].copy()
    b["isin"] = b["cusip"].map(_isin)
    kind = auctions.dropna(subset=["isin"]).drop_duplicates("isin", keep="last").set_index("isin")["sec_type"]
    b["sec_type"] = b["isin"].map(kind).fillna("Nominal")
    b["op_type"] = "Buyback"
    b["sec_desc_long"] = "Buyback"
    b["source"] = "Treasury Fiscal Data buybacks_security_details"
    return b


def _pages_or_cache(force: bool = False) -> pd.DataFrame:
    """Fiscal Data pages; when the API is down (503s and timeouts, Nov-2026) the last cached pull,
    so the TreasuryDirect top-up still runs instead of the whole US fetch failing."""
    try:
        return _pages(force=force)
    except (requests.RequestException, FetchError) as exc:
        cached = CACHE / "us_auctions_raw.parquet"
        if not cached.exists():
            raise
        print(f"  WARNING Fiscal Data auctions_query failed ({type(exc).__name__}); using the cached pull")
        return pd.read_parquet(cached)


def fetch_us(force: bool = False) -> pd.DataFrame:
    raw = _top_up(_pages_or_cache(force=force))
    last = pd.to_datetime(raw.loc[pd.to_numeric(raw["total_accepted"], errors="coerce") > 0, "auction_date"]).max()
    if pd.notna(last) and (pd.Timestamp.today().normalize() - last).days > 7:
        print(f"  WARNING US auction tape ends {last:%Y-%m-%d} — Fiscal Data and TreasuryDirect both stale")
    out = pd.DataFrame()
    out["op_dt"] = pd.to_datetime(raw.get("auction_date"), errors="coerce")
    out["issue_dt"] = pd.to_datetime(raw.get("issue_date"), errors="coerce")
    out["maturity_dt"] = pd.to_datetime(raw.get("maturity_date"), errors="coerce")
    out["cusip"] = raw.get("cusip")
    out["isin"] = out["cusip"].map(_isin)
    out["sec_desc_long"] = (
        raw.get("security_type", pd.Series(dtype=str)).astype(str) + " " +
        raw.get("security_term", pd.Series(dtype=str)).astype(str)
    )
    # Offering is the announcement. Fiscal Data leaves total_accepted null until
    # the auction results; substituting it invents face (today's 7y/bills, and
    # the 2001-09-11 4-week bill).
    accepted = pd.to_numeric(raw.get("total_accepted"), errors="coerce")
    out["amount_issued"] = accepted / 1e6
    # Fed rollover add-ons are on top of the offering: in the stock, not sold to the public.
    out["official_take_mn"] = pd.to_numeric(raw.get("soma_accepted"), errors="coerce") / 1e6
    out["price"] = pd.to_numeric(raw.get("price_per100"), errors="coerce")
    out["yield"] = pd.to_numeric(raw.get("high_yield"), errors="coerce")
    out["coupon"] = pd.to_numeric(raw.get("int_rate"), errors="coerce")
    out["btc"] = pd.to_numeric(raw.get("bid_to_cover_ratio"), errors="coerce")
    out["sec_desc"] = raw.get("security_term")
    out["op_type"] = raw.get("reopening").map(
        lambda x: "Reopening" if str(x).lower() in {"y", "yes", "true", "1"} else "Auction"
    )
    out["sec_type"] = raw.apply(_sec_type, axis=1)
    out["source"] = "Treasury Fiscal Data auctions_query"
    ops = coerce_ops(attach_tenor(out), "US")
    ops = ops[ops["op_dt"].notna() & (ops["amount_issued"].fillna(0) > 0)]
    try:
        bb = _buyback_ops(_buybacks(force=force), ops)
    except Exception as exc:  # noqa: BLE001
        print(f"  US buybacks: {exc}")
        bb = pd.DataFrame()
    if not bb.empty:
        ops = pd.concat([ops, coerce_ops(attach_tenor(bb), "US")], ignore_index=True)
    ops.to_parquet(CACHE / "ops_us.parquet", index=False)
    return ops
