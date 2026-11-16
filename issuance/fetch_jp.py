"""JGB and Treasury Discount Bill auctions, Ministry of Finance.

Sources, one row per auction (or per reopened issue for liquidity enhancement auctions):

    Auction_Results_for_JGBs.xls     bulk history, one sheet per product, 100 million yen.
                                     Lags by a month or so.
    Auction_Results_for_T-bills.xls  T-Discount Bills (2m/3m/6m/1y) from the Feb-2009 TB/FB
                                     unification, one sheet per fiscal year, billion yen.
    calendar/YYMMe.htm               monthly calendar; each auction links to a result page
                                     (eresulYYYYMMDD.htm) and, once NPC II settles, an "a" page
                                     carrying the same table plus NPC II. Billion yen. Bill
                                     results sit under etbill/etbillresul/ with the same file
                                     names as coupon results, so they are cached as tb_*.htm.

Amounts are million yen face. amount_issued = competitive accepted + non-competitive
(retail, old 2/5/10y) + NPC I + NPC II. 40y and GX are yield-Dutch auctions: no average,
so tail is NaN. Linker lowest prices from Oct-2017 are before the indexation coefficient.
Bills: accepted + NPC I ([A+B] in the workbook). Bills are TB and FB jointly: the issuance
plan's TB line (FY2026 ¥40.8tn, 1y + 6m) is only part of the 6m/1y gross, and the weekly 3m
bills are almost all FB (cash management), outside the plan. The JGB workbook's pre-2009 TB
sheet (no FBs) is not read, so bill history starts Feb-2009.

Identifier: isin = "JP{tenor}#{issue}" ("JP10y#383", "JPIL10y#29", "JPGX5y#4"); shared with
the BoJ holdings adapter, so keep the format. Bills: "JPTDB#{issue}" (one number series
across tenors).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from issuance.http import FetchError, get_bytes
from issuance.paths import CACHE, RAW
from issuance.schema import OPS_COLUMNS
from issuance.tenor import bucket_from_years

BASE = "https://www.mof.go.jp/english/policy/jgbs/auction/"
WORKBOOK_URL = BASE + "past_auction_results/Auction_Results_for_JGBs.xls"
RAW_JP = RAW / "jp"
WORKBOOK = RAW_JP / "Auction_Results_for_JGBs.xls"
TBILL_URL = BASE + "past_auction_results/Auction_Results_for_T-bills.xls"
TBILL_WORKBOOK = RAW_JP / "Auction_Results_for_T-bills.xls"
PAGES = RAW_JP / "mof_auction"
OUT = CACHE / "ops_jp.parquet"

EXTRA_COLUMNS = ("tail_yen", "tail_bp", "npc1_mn", "npc2_mn", "offering_mn", "product")
LEA_FROM = "2024-01"

# sheet -> (product label, isin stem, sec_type)
SHEETS = {
    "40年債": ("40y", "40y", "Nominal"),
    "30年債": ("30y", "30y", "Nominal"),
    "20年債": ("20y", "20y", "Nominal"),
    "15変動": ("15y FRN", "FRN15y", "FRN"),
    "10年債": ("10y", "10y", "Nominal"),
    "GX10年債": ("GX10y", "GX10y", "Nominal"),
    "10年物価連動": ("10y IL", "IL10y", "Linker"),
    "5年債": ("5y", "5y", "Nominal"),
    "GX5年債": ("GX5y", "GX5y", "Nominal"),
    "2年債": ("2y", "2y", "Nominal"),
    "6年債": ("6y", "6y", "Nominal"),
    "4年債": ("4y", "4y", "Nominal"),
    "割3年": ("3y Disc", "D3y", "Nominal"),
}
BILLS = ("TDB 2m", "TDB 3m", "TDB 6m", "TDB 1y")
BILL_MAX_YRS = 364 / 365.25
STEM = {label: stem for label, stem, _ in SHEETS.values()} | {b: "TDB" for b in BILLS}
SEC_TYPE = {label: kind for label, _, kind in SHEETS.values()} | {b: "Bill" for b in BILLS}


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().lower()


def _num(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = unicodedata.normalize("NFKC", str(value)).replace(",", "").replace("%", "").strip()
    return float(text) if re.fullmatch(r"[-+]?\d+(\.\d+)?", text) else np.nan


def isin_for(product: str, issue: object) -> str | None:
    n = _num(issue)
    if np.isnan(n) or product not in STEM:
        return None
    return f"JP{STEM[product]}#{int(n)}"


def product_from_security(text: str) -> str | None:
    """'30-Year' / '10-year Inflation-Indexed Bonds' / '10-Year Climate Transition' -> label."""
    t = _norm(text)
    m = re.search(r"(\d+)\s*-?\s*year", t)
    if not m:
        return None
    years = int(m.group(1))
    if "inflation" in t:
        return "10y IL"
    if "climate" in t or "gx" in t:
        return f"GX{years}y"
    if "floating" in t:
        return "15y FRN"
    label = f"{years}y"
    return label if label in STEM else None


def bill_product(text: object) -> str | None:
    """'3-month' / '1-year' / 'Treasury Discount Bills (6-month) (1411)' -> 'TDB 6m'."""
    m = re.search(r"(\d+)\s*-?\s*(month|year)", _norm(text))
    if not m:
        return None
    label = f"TDB {m.group(1)}{'m' if m.group(2) == 'month' else 'y'}"
    return label if label in BILLS else None


# ---------------------------------------------------------------- bulk workbook

def _pick(cols: list[str], *needles: str, exclude: tuple[str, ...] = ()) -> int | None:
    for i, c in enumerate(cols):
        if all(n in c for n in needles) and not any(x in c for x in exclude):
            return i
    return None


def _sheet_rows(raw: pd.DataFrame, product: str) -> pd.DataFrame:
    header_row = next(i for i in range(min(10, len(raw))) if _norm(raw.iat[i, 0]) == "issue number")
    cols = [_norm(c) for c in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:]
    body = body[body.iloc[:, 0].map(lambda v: not np.isnan(_num(v)))]
    if product == "Bill":
        return pd.DataFrame()

    def col(*needles: str, exclude: tuple[str, ...] = ()) -> pd.Series:
        i = _pick(cols, *needles, exclude=exclude)
        return body.iloc[:, i].map(_num) if i is not None else pd.Series(np.nan, index=body.index)

    def dates(*needles: str) -> pd.Series:
        i = _pick(cols, *needles)
        return pd.to_datetime(body.iloc[:, i], errors="coerce") if i is not None else pd.NaT

    hundred = 100.0
    out = pd.DataFrame(index=body.index)
    out["issue"] = body.iloc[:, 0].map(_num)
    out["op_dt"] = dates("auction date")
    out["issue_dt"] = dates("issue date")
    out["maturity_dt"] = dates("maturity date")
    out["coupon"] = col("nominal coupon")
    out["offering_mn"] = col("offering amount") * hundred
    out["amount_bid"] = col("competitive bids", exclude=("non-",)) * hundred
    accepted = col("amounts of bids accepted", exclude=("non-price",)) * hundred
    noncomp = col("non-competitive bids") * hundred
    out["npc1_mn"] = col("auction i for") * hundred
    out["npc2_mn"] = col("auction ii for") * hundred
    out["accepted_mn"] = accepted
    out["noncomp_mn"] = noncomp
    avg_px = col("weighted average price")
    avg_y = col("yield at the average price")
    low_px = col("lowest accepted price")
    low_y = col("yield at the lowest accepted price")
    hy_px = col("price at the highest accepted yield")
    hy_y = col("highest accepted yield", exclude=("price",))
    out["avg_px"], out["avg_y"] = avg_px, avg_y
    out["low_px"] = low_px.fillna(hy_px)
    out["low_y"] = low_y.fillna(hy_y)
    return out.reset_index(drop=True)


def _workbook(force: bool = False, refresh: bool = True) -> pd.DataFrame:
    stale = refresh and not WORKBOOK.exists() or refresh and (
        pd.Timestamp.now() - pd.Timestamp(WORKBOOK.stat().st_mtime, unit="s") > pd.Timedelta(days=7)
    )
    if force or stale:
        try:
            get_bytes(WORKBOOK_URL, dest=WORKBOOK, min_bytes=50_000)
        except FetchError as exc:
            if not WORKBOOK.exists():
                raise
            print(f"  JP workbook refetch failed ({exc}); using local copy")
    book = pd.read_excel(WORKBOOK, sheet_name=None, header=None)
    parts = []
    for sheet, (product, _stem, _kind) in SHEETS.items():
        if sheet not in book:
            print(f"  JP workbook: sheet {sheet} missing")
            continue
        rows = _sheet_rows(book[sheet], product)
        rows["product"] = product
        parts.append(rows)
    out = pd.concat(parts, ignore_index=True)
    out["source"] = "MoF Auction_Results_for_JGBs.xls"
    out["op_type"] = "Auction"
    return out


def _tbill_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    header_row = next(i for i in range(min(10, len(raw))) if _norm(raw.iat[i, 0]) == "issue number")
    cols = [_norm(c) for c in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:]
    body = body[body.iloc[:, 0].map(lambda v: not np.isnan(_num(v)))]

    def col(*needles: str, exclude: tuple[str, ...] = ()) -> pd.Series:
        i = _pick(cols, *needles, exclude=exclude)
        return body.iloc[:, i].map(_num) if i is not None else pd.Series(np.nan, index=body.index)

    def dates(*needles: str) -> pd.Series:
        i = _pick(cols, *needles)
        return pd.to_datetime(body.iloc[:, i], errors="coerce") if i is not None else pd.NaT

    bn = 1000.0
    out = pd.DataFrame(index=body.index)
    out["product"] = body.iloc[:, 1].map(bill_product)
    out["issue"] = body.iloc[:, 0].map(_num)
    out["op_dt"] = dates("auction date")
    out["issue_dt"] = dates("issue date")
    out["maturity_dt"] = dates("maturity date")
    out["coupon"] = 0.0
    out["amount_bid"] = col("competitive bids", exclude=("non-",)) * bn
    out["accepted_mn"] = col("amounts of bids accepted") * bn
    out["npc1_mn"] = col("auction i") * bn
    out["avg_px"] = col("average price")
    out["avg_y"] = col("yield at the average price")
    out["low_px"] = col("lowest accepted price")
    out["low_y"] = col("yield at the lowest accepted price")
    return out.reset_index(drop=True)


def _tbill_workbook(force: bool = False, refresh: bool = True) -> pd.DataFrame:
    """T-Discount Bill auctions Feb-2009 →, one sheet per fiscal year (billion yen)."""
    stale = refresh and (not TBILL_WORKBOOK.exists() or (
        pd.Timestamp.now() - pd.Timestamp(TBILL_WORKBOOK.stat().st_mtime, unit="s") > pd.Timedelta(days=7)))
    if force or stale:
        try:
            get_bytes(TBILL_URL, dest=TBILL_WORKBOOK, min_bytes=50_000)
        except FetchError as exc:
            if not TBILL_WORKBOOK.exists():
                print(f"  JP T-bill workbook unavailable ({exc}); bills from calendar pages only")
                return pd.DataFrame()
            print(f"  JP T-bill workbook refetch failed ({exc}); using local copy")
    book = pd.read_excel(TBILL_WORKBOOK, sheet_name=None, header=None)
    parts = [_tbill_sheet(raw) for name, raw in book.items() if name.strip().upper().startswith("FY")]
    out = pd.concat(parts, ignore_index=True)
    out = out[out["product"].notna()]
    out["source"] = "MoF Auction_Results_for_T-bills.xls"
    out["op_type"] = "Auction"
    return out


# ---------------------------------------------------------------- calendar pages

def _cached(url: str, name: str, refetch: bool) -> bytes | None:
    dest = PAGES / name
    if dest.exists() and dest.stat().st_size > 2000 and not refetch:
        return dest.read_bytes()
    try:
        return get_bytes(url, dest=dest, min_bytes=2000, retries=2)
    except FetchError as exc:
        if dest.exists():
            return dest.read_bytes()
        print(f"  JP page unavailable: {name} ({exc})")
        return None


def _tables(payload: bytes) -> list[list[list[str]]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "html.parser")
    out = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        out.append(rows)
    return out


def calendar_links(payload: bytes) -> list[dict]:
    """Coupon-JGB result links on one calendar page (bills and borrowings skipped)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        hrefs = [a.get("href", "") for a in tr.find_all("a")]
        res = [h for h in hrefs if re.search(r"/eresul/eresul\d{8}a?\.htm", h) and "etbill" not in h]
        if not res:
            continue
        main = next((h for h in res if re.search(r"eresul\d{8}\.htm", h)), None)
        extra = next((h for h in res if re.search(r"eresul\d{8}a\.htm", h)), None)
        issue = cells[1].get_text(" ", strip=True)
        out.append({"issue": issue, "main": main, "extra": extra,
                    "lea": "liquidity enhancement" in _norm(issue)})
    return out


def tbill_links(payload: bytes) -> list[dict]:
    """T-Discount Bill result links on one calendar page: product, issue number, result URL."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        res = next((a.get("href", "") for a in tr.find_all("a")
                    if re.search(r"etbill/etbillresul/eresul\d{8}\.htm", a.get("href", ""))), None)
        label = cells[1].get_text(" ", strip=True)
        product = bill_product(label) if "discount bill" in _norm(label) else None
        if res is None or product is None:
            continue
        issue = re.search(r"\((\d+)\)\s*$", label)
        out.append({"product": product, "issue": float(issue.group(1)) if issue else np.nan, "result": res})
    return out


def parse_tbill_result(payload: bytes, product: str) -> dict | None:
    """Header row + value row of a T-Discount Bill result page. Billion yen -> million."""
    for rows in _tables(payload):
        for i, head in enumerate(rows[:-1]):
            if not head or _norm(head[0]) != "issue number":
                continue
            rec = dict(zip([_norm(c) for c in head], rows[i + 1]))

            def get(*needles: str, exclude: tuple[str, ...] = ()) -> str | None:
                for k, v in rec.items():
                    if all(n in k for n in needles) and not any(x in k for x in exclude):
                        return v
                return None

            def day(*needles: str) -> pd.Timestamp:
                return pd.to_datetime(get(*needles), format="%m/%d/%Y", errors="coerce")

            bn = 1000.0
            return {
                "product": product,
                "issue": _num(get("issue number")),
                "op_dt": day("auction date"),
                "issue_dt": day("issue date"),
                "maturity_dt": day("maturity date"),
                "coupon": 0.0,
                "amount_bid": _num(get("competitive bids", exclude=("non-",))) * bn,
                "accepted_mn": _num(get("bids accepted")) * bn,
                "npc1_mn": _num(get("auction i")) * bn,
                "low_px": _num(get("lowest accepted price", exclude=("yield",))),
                "low_y": _num(get("yield at the lowest accepted price")),
                "avg_px": _num(get("average price", exclude=("yield",))),
                "avg_y": _num(get("yield at the average price")),
                "noncomp_mn": np.nan,
                "npc2_mn": np.nan,
            }
    return None


def _abs(href: str) -> str:
    return urljoin(BASE + "calendar/", href)


def parse_result(payload: bytes) -> dict | None:
    """Label row + value row of a coupon auction result page. Billion yen -> million."""
    for rows in _tables(payload):
        for i, head in enumerate(rows[:-1]):
            if not head or _norm(head[0]) != "security":
                continue
            vals = rows[i + 1]
            labels = [_norm(c) for c in head]
            rec = dict(zip(labels, vals))

            def get(*needles: str, exclude: tuple[str, ...] = ()) -> str | None:
                for k, v in rec.items():
                    if all(n in k for n in needles) and not any(x in k for x in exclude):
                        return v
                return None

            product = product_from_security(vals[0])
            if product is None:
                return None
            bn = 1000.0
            return {
                "product": product,
                "issue": _num(get("issue number")),
                "op_dt": pd.to_datetime(get("auction date"), format="%m/%d/%Y", errors="coerce"),
                "issue_dt": pd.to_datetime(get("issue date"), format="%m/%d/%Y", errors="coerce"),
                "maturity_dt": pd.to_datetime(get("maturity date"), format="%m/%d/%Y", errors="coerce"),
                "coupon": _num(get("nominal coupon")),
                "amount_bid": _num(get("competitive bids")) * bn,
                "accepted_mn": _num(get("amounts of bids accepted", exclude=("non-price",))) * bn,
                "low_px": _num(get("lowest accepted price", exclude=("yield",)) or get("price at the highest accepted yield")),
                "low_y": _num(get("yield at the lowest accepted price") or get("highest accepted yield", exclude=("price", "allotment"))),
                "avg_px": _num(get("weighted average price")),
                "avg_y": _num(get("yield at the average price")),
                "npc1_mn": _num(get("auction i for")) * bn,
                "npc2_mn": _num(get("auction ii for")) * bn,
                "noncomp_mn": np.nan,
            }
    return None


def parse_lea(summary: bytes | None, reopened: bytes) -> list[dict]:
    """Liquidity enhancement auction: one row per reopened issue (billion yen face)."""
    meta = {}
    if summary is not None:
        for rows in _tables(summary):
            if rows and _norm(rows[0][0]) == "auction date" and len(rows) > 1:
                rec = dict(zip([_norm(c) for c in rows[0]], rows[1]))
                meta = {
                    "op_dt": pd.to_datetime(rec.get("auction date"), format="%m/%d/%Y", errors="coerce"),
                    "issue_dt": pd.to_datetime(rec.get("issue date"), format="%m/%d/%Y", errors="coerce"),
                    "bids": _num(next((v for k, v in rec.items() if "competitive bids" in k), None)),
                    "accepted": _num(next((v for k, v in rec.items() if "bids accepted" in k), None)),
                    "spread": next((v for k, v in rec.items() if "average accepted spread" in k), None),
                }
                break
    out = []
    for rows in _tables(reopened):
        if not rows or "securities" not in _norm(rows[0][0]):
            continue
        for row in rows[1:]:
            if len(row) < 3:
                continue
            product = product_from_security(row[0])
            amt = _num(row[2])
            if product is None or np.isnan(amt):
                continue
            out.append({
                "product": product,
                "issue": _num(row[1]),
                "op_dt": meta.get("op_dt"),
                "issue_dt": meta.get("issue_dt"),
                "accepted_mn": amt * 1000.0,
                "lea_btc": (meta["bids"] / meta["accepted"]) if meta.get("accepted") else np.nan,
                "sec_desc_long": f"Liquidity Enhancement Auction, avg spread {meta.get('spread')}",
            })
    return out


def _months(start: str, end: date) -> list[str]:
    return [p.strftime("%y%m") for p in pd.period_range(start, pd.Period(end, "M"), freq="M")]


def _calendar(since: str, lea_from: str, bills_since: str | None = None) -> pd.DataFrame:
    today = date.today()
    recent = pd.Timestamp(today) - pd.Timedelta(days=21)
    bills_since = bills_since or since
    first = min(pd.Period(since, "M"), pd.Period(lea_from, "M"), pd.Period(bills_since, "M")).strftime("%Y-%m")
    rows = []
    for ym in _months(first, today):
        month = pd.Period(f"20{ym[:2]}-{ym[2:]}", "M")
        refetch = month.end_time >= recent
        page = _cached(f"{BASE}calendar/{ym}e.htm", f"cal_{ym}e.htm", refetch)
        if page is None:
            continue
        coupons = month >= pd.Period(since, "M")
        leas = month >= pd.Period(lea_from, "M")
        if month >= pd.Period(bills_since, "M"):
            for link in tbill_links(page):
                url = _abs(link["result"])
                stamp = re.search(r"(\d{8})", url)
                if stamp is None or pd.Timestamp(stamp.group(1)) > pd.Timestamp(today):
                    continue
                payload = _cached(url, "tb_" + url.rsplit("/", 1)[-1], pd.Timestamp(stamp.group(1)) >= recent)
                rec = parse_tbill_result(payload, link["product"]) if payload else None
                if rec is None or np.isnan(rec["accepted_mn"]):
                    continue
                rec["op_type"] = "Auction"
                rec["source"] = "MoF auction result page (T-bill)"
                rows.append(rec)
        for link in calendar_links(page):
            if link["lea"] and not leas:
                continue
            if not link["lea"] and not coupons:
                continue
            main_url = _abs(link["main"]) if link["main"] else None
            extra_url = _abs(link["extra"]) if link["extra"] else None
            stamp = re.search(r"(\d{8})", main_url or extra_url or "")
            fresh = stamp is not None and pd.Timestamp(stamp.group(1)) >= recent
            main = _cached(main_url, main_url.rsplit("/", 1)[-1], fresh) if main_url else None
            extra = _cached(extra_url, extra_url.rsplit("/", 1)[-1], fresh) if extra_url else None
            if link["lea"]:
                if extra is None:
                    continue
                for rec in parse_lea(main, extra):
                    rec["op_type"] = "Liquidity Enhancement"
                    rec["source"] = "MoF auction calendar (liquidity enhancement)"
                    rows.append(rec)
                continue
            rec = (parse_result(extra) if extra else None) or (parse_result(main) if main else None)
            if rec is None:
                continue
            rec["op_type"] = "Auction"
            rec["source"] = "MoF auction result page"
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- assembly

def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("npc1_mn", "npc2_mn", "noncomp_mn", "accepted_mn", "avg_px", "avg_y", "low_px", "low_y",
                "offering_mn", "amount_bid", "coupon", "lea_btc"):
        if col not in out.columns:
            out[col] = np.nan
    auction = out["op_type"] == "Auction"
    out["amount_issued"] = out["accepted_mn"] + (
        out[["noncomp_mn", "npc1_mn", "npc2_mn"]].fillna(0).sum(axis=1).where(auction, 0.0)
    )
    out["btc"] = (out["amount_bid"] / out["accepted_mn"]).where(auction, out["lea_btc"])
    out["price"] = out["avg_px"].fillna(out["low_px"])
    out["yield"] = out["avg_y"].fillna(out["low_y"])
    out["tail_yen"] = out["avg_px"] - out["low_px"]
    out["tail_bp"] = (out["low_y"] - out["avg_y"]) * 100.0
    out["isin"] = [isin_for(p, n) for p, n in zip(out["product"], out["issue"])]
    out["sec_type"] = out["product"].map(SEC_TYPE)
    desc = out["product"] + " #" + out["issue"].map(lambda n: str(int(n)) if not np.isnan(n) else "?")
    desc = desc.where(out["sec_type"].eq("Bill"), "JGB " + desc)
    if "sec_desc_long" in out.columns:
        out["sec_desc_long"] = out["sec_desc_long"].where(out["sec_desc_long"].notna(), desc)
    else:
        out["sec_desc_long"] = desc
    out["sec_desc"] = out["product"]
    return out


def _fill_static(ops: pd.DataFrame) -> pd.DataFrame:
    """LEA rows carry no coupon / maturity; take them from the issue's own auctions."""
    static = (ops[ops["op_type"] == "Auction"].dropna(subset=["isin"])
              .sort_values("op_dt").drop_duplicates("isin", keep="last")
              .set_index("isin")[["maturity_dt", "coupon"]])
    for col in ("maturity_dt", "coupon"):
        ops[col] = ops[col].where(ops[col].notna(), ops["isin"].map(static[col]))
    return ops


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in OPS_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["country"] = "JP"
    out["ccy"] = "JPY"
    for col in ("op_dt", "issue_dt", "maturity_dt"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in ("amount_issued", "official_take_mn", "amount_bid", "price", "yield", "coupon", "btc",
                "resid_mat_yrs", *EXTRA_COLUMNS[:-1]):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["isin"] = out["isin"].astype("string")
    out["cusip"] = out["cusip"].astype("string")
    out["resid_mat_yrs"] = (out["maturity_dt"] - out["op_dt"]).dt.days / 365.25
    # A 1y bill runs 365-371 days from the auction; keep every bill in the 0-1y bucket.
    bill = out["sec_type"].astype(str).eq("Bill")
    out.loc[bill, "resid_mat_yrs"] = out.loc[bill, "resid_mat_yrs"].clip(upper=BILL_MAX_YRS)
    out["tenor_bucket"] = out["resid_mat_yrs"].map(bucket_from_years)
    out["is_forecast"] = False
    out["sec_type"] = out["sec_type"].astype("string")
    out["op_type"] = out["op_type"].astype("string")
    out["product"] = out["product"].astype("string")
    return out[list(OPS_COLUMNS) + list(EXTRA_COLUMNS)].reset_index(drop=True)


def fetch_jp(force: bool = False, lea_from: str = LEA_FROM) -> pd.DataFrame:
    book = _workbook(force=force)
    bills = _tbill_workbook(force=force)
    last = book["op_dt"].max()
    since = (last - pd.Timedelta(days=31)).strftime("%Y-%m")
    bills_since = (bills["op_dt"].max() - pd.Timedelta(days=31)).strftime("%Y-%m") if not bills.empty else "2024-01"
    recent = _calendar(since, lea_from, bills_since)
    parts = [_finish(book)]
    if not bills.empty:
        parts.append(_finish(bills))
    if not recent.empty:
        parts.append(_finish(recent))
    ops = pd.concat(parts, ignore_index=True)
    ops = ops[ops["op_dt"].notna() & ops["isin"].notna() & (ops["amount_issued"].fillna(0) > 0)]
    ops = ops.drop_duplicates(["isin", "op_dt", "op_type"], keep="first")
    ops = _fill_static(ops)
    ops = _coerce(ops).sort_values(["op_dt", "product"]).reset_index(drop=True)
    ops.to_parquet(OUT, index=False)
    n_cal = int((~ops["source"].isin(["MoF Auction_Results_for_JGBs.xls", "MoF Auction_Results_for_T-bills.xls"])).sum())
    n_bill = int(ops["sec_type"].eq("Bill").sum())
    print(f"  JP: {len(ops)} rows ({n_cal} from calendar pages, {n_bill} T-bills), "
          f"{ops['op_dt'].min():%Y-%m-%d} → {ops['op_dt'].max():%Y-%m-%d}")
    return ops


if __name__ == "__main__":
    fetch_jp()
