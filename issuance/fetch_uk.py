"""UK gilt operations. DMO XML when the address is allowed; else the long-end workbook."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import pandas as pd

from issuance.http import FetchError, get_bytes, looks_blocked
from issuance.paths import CACHE, RAW, REPO
from issuance.schema import coerce_ops
from issuance.tenor import attach_tenor

DMO = "https://www.dmo.gov.uk/data/XmlDataReport"
WORKBOOK_OPS = REPO / "projects" / "uk_long_end" / "workbook" / "42_issuance_concession_ops.csv"
WORKBOOK_STOCK = REPO / "projects" / "uk_long_end" / "workbook" / "35_stock_apf_freefloat_by_isin.csv"


def _sys_path() -> None:
    assets = str(REPO / "data")
    if assets not in sys.path:
        sys.path.insert(0, assets)


def _classify_gilt(name: object) -> str:
    text = str(name).lower()
    if "index" in text or "linker" in text:
        return "Linker"
    return "Nominal"


def _classify_op(raw: object) -> str:
    text = str(raw).lower()
    if "syndic" in text:
        return "Syndication"
    if "tender" in text:
        return "Tender"
    if "reverse" in text or "buyback" in text or "cancel" in text:
        return "Buyback"
    if "conversion" in text or "switch" in text:
        return "Exchange"
    if "auction" in text or "outright" in text:
        return "Auction"
    return "Other"


def _from_workbook() -> pd.DataFrame:
    if not WORKBOOK_OPS.exists():
        raise FetchError(f"DMO blocked and workbook missing: {WORKBOOK_OPS}")
    raw = pd.read_csv(WORKBOOK_OPS)
    out = pd.DataFrame()
    out["op_dt"] = pd.to_datetime(raw["date"], errors="coerce")
    out["maturity_dt"] = pd.to_datetime(raw.get("redemption_date"), errors="coerce")
    out["isin"] = raw["isin"]
    out["sec_desc_long"] = raw["name"]
    out["amount_issued"] = pd.to_numeric(raw["nominal_mn"], errors="coerce")
    out["price"] = pd.to_numeric(raw.get("issue_price"), errors="coerce")
    out["yield"] = pd.to_numeric(raw.get("awarded_yield_pct"), errors="coerce")
    out["resid_mat_yrs"] = pd.to_numeric(raw.get("years_to_maturity"), errors="coerce")
    out["op_type"] = raw.get("mechanism", pd.Series(dtype=str)).map(_classify_op)
    linker = raw.get("linker")
    if linker is not None:
        out["sec_type"] = linker.map(lambda x: "Linker" if bool(x) else "Nominal")
    else:
        out["sec_type"] = out["sec_desc_long"].map(_classify_gilt)
    out["source"] = "uk_long_end/42_issuance_concession_ops (DMO D2.1E)"
    return out


def _write_stock_from_workbook() -> None:
    if not WORKBOOK_STOCK.exists():
        return
    raw = pd.read_csv(WORKBOOK_STOCK)
    out = pd.DataFrame()
    out["isin"] = raw["isin"].astype("string").str.strip().str.upper()
    out["name"] = raw["name"]
    out["amount_nominal_mn"] = pd.to_numeric(raw["nominal_mn"], errors="coerce")
    out["sec_type"] = raw.get("linker").map(lambda x: "Linker" if bool(x) else "Nominal") if "linker" in raw.columns else "Nominal"
    out["country"] = "UK"
    out["as_of"] = pd.Timestamp.today().normalize()
    out.to_parquet(CACHE / "stock_uk.parquet", index=False)


def _gilt_frame(payload: bytes) -> pd.DataFrame:
    """D2.1E rows are attributes on View_Gilt_Issuance_History, not child elements."""
    if looks_blocked(payload):
        raise FetchError("DMO D2.1E blocked")
    root = ET.fromstring(payload)
    rows = []
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag == "View_Gilt_Issuance_History" and node.attrib:
            rows.append(dict(node.attrib))
            continue
        kids = list(node)
        if len(kids) >= 4 and all(len(list(k)) == 0 for k in kids):
            rows.append({k.tag.split("}")[-1]: (k.text or "").strip() for k in kids})
    if not rows:
        raise FetchError("DMO D2.1E parsed empty")
    history = pd.DataFrame(rows)
    cols = {c.upper(): c for c in history.columns}

    def pick(*names: str):
        for name in names:
            if name in history.columns:
                return name
            if name.upper() in cols:
                return cols[name.upper()]
        return None

    out = pd.DataFrame()
    out["sec_desc_long"] = history[pick("INSTRUMENT_NAME", "sec_desc_long") or history.columns[0]]
    isin_col = pick("ISIN_CODE", "isin", "ISIN")
    out["isin"] = history[isin_col] if isin_col else pd.NA
    out["op_dt"] = history[pick("ACTUAL_DATE", "op_dt") or history.columns[0]]
    amt = pick("NOMINAL_ISSUED", "amount_issued")
    out["amount_issued"] = history[amt] if amt else pd.NA
    out["price"] = history[pick("ISSUE_CLEAN_PRICE", "price")] if pick("ISSUE_CLEAN_PRICE", "price") else pd.NA
    out["yield"] = history[pick("ISSUE_YIELD", "yield")] if pick("ISSUE_YIELD", "yield") else pd.NA
    op = pick("ISSUANCE_TYPE", "op_type")
    out["op_type"] = history[op].map(_classify_op) if op else "Auction"
    out["sec_type"] = out["sec_desc_long"].map(_classify_gilt)
    out["source"] = "DMO D2.1E"
    return out


def _from_xml(force: bool = False) -> pd.DataFrame:
    dest = RAW / "uk" / "D2_1E.xml"
    if dest.exists() and dest.stat().st_size > 5000 and not force:
        payload = dest.read_bytes()
    else:
        payload = get_bytes(f"{DMO}?reportCode=D2.1E", dest=dest, timeout=180)
    return _gilt_frame(payload)


def _d1a_rows() -> pd.DataFrame:
    dest = RAW / "uk" / "D1A.xml"
    if not dest.exists() or dest.stat().st_size < 1000:
        return pd.DataFrame()
    payload = dest.read_bytes()
    if looks_blocked(payload):
        return pd.DataFrame()
    root = ET.fromstring(payload)
    rows = []
    for node in root.iter():
        if node.tag.split("}")[-1] != "View_GILTS_IN_ISSUE":
            continue
        kind = node.attrib.get("INSTRUMENT_TYPE") or ""
        rows.append({
            "isin": node.attrib.get("ISIN_CODE"),
            "name": node.attrib.get("INSTRUMENT_NAME"),
            "maturity_dt": node.attrib.get("REDEMPTION_DATE"),
            "amount_nominal_mn": node.attrib.get("TOTAL_AMOUNT_IN_ISSUE"),
            "sec_type": "Linker" if "index" in kind.lower() else "Nominal",
            "as_of": node.attrib.get("CLOSE_OF_BUSINESS_DATE"),
        })
    return pd.DataFrame(rows)


def _attach_maturity(ops: pd.DataFrame) -> pd.DataFrame:
    """D2.1E has no redemption date. D1A covers the live book; the workbook covers matured names."""
    parts = []
    d1a = _d1a_rows()
    if not d1a.empty:
        d1a = d1a.copy()
        d1a["maturity_dt"] = pd.to_datetime(d1a["maturity_dt"], errors="coerce")
        parts.append(d1a[["isin", "maturity_dt"]])
    if WORKBOOK_OPS.exists():
        raw = pd.read_csv(WORKBOOK_OPS, usecols=lambda c: c in {"isin", "redemption_date"})
        parts.append(pd.DataFrame({
            "isin": raw["isin"],
            "maturity_dt": pd.to_datetime(raw["redemption_date"], errors="coerce"),
        }))
    if not parts:
        return ops
    mat = pd.concat(parts, ignore_index=True)
    mat["isin"] = mat["isin"].astype(str).str.strip().str.upper()
    mat = mat.dropna(subset=["isin", "maturity_dt"]).drop_duplicates("isin")
    out = ops.copy()
    out["isin"] = out["isin"].astype(str).str.strip().str.upper()
    out.loc[out["isin"].isin(["<NA>", "NAN", "NONE", ""]), "isin"] = pd.NA
    out = out.drop(columns=["maturity_dt"], errors="ignore")
    return out.merge(mat, on="isin", how="left")


D21A_XLS = REPO / "uk_flows" / "cache" / "dmo" / "auctions.xls"
D21A_PDF = RAW / "uk" / "D2_1A.pdf"
# D2.1A leaves PAOF blank until the facility closes; a blank this close to the data date is unknown.
PAOF_PENDING_BDAYS = 3
PAOF_MAX_SHARE = 0.25  # largest PAOF entitlement the DMO has offered
_FRACTIONS = {"¼": " 1/4", "½": " 1/2", "¾": " 3/4", "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8", "⅞": " 7/8"}


def _gilt_key(name: object) -> tuple | None:
    """(coupon, maturity year, linker): D2.1A and D2.1E spell coupons differently (4⅝ vs 4 5/8)."""
    import re

    text = str(name)
    for k, v in _FRACTIONS.items():
        text = text.replace(k, v)
    m = re.match(r"\s*(\d+)?(?:\s+(\d+)/(\d+))?\s*%", text)
    years = re.findall(r"(?:19|20)\d{2}", text)
    if not m or not years:
        return None
    coupon = float(m.group(1) or 0) + (float(m.group(2)) / float(m.group(3)) if m.group(2) else 0.0)
    return round(coupon, 4), int(years[-1]), "index" in text.lower()


def _d21a_results() -> pd.DataFrame:
    """D2.1A per auction: date, name, nominal via auction and via PAOF (NaN = not yet published)."""
    out = pd.DataFrame()
    data_dt = pd.NaT
    if D21A_XLS.exists():
        try:
            raw = pd.read_excel(D21A_XLS, header=None)
            stamp = str(raw.iloc[0, 0])
            data_dt = pd.to_datetime(stamp.split(":", 1)[-1].strip(), format="%d-%b-%Y", errors="coerce")
            body = raw.iloc[5:]
            out = pd.DataFrame({
                "date": pd.to_datetime(body[0], errors="coerce"),
                "name": body[1].astype(str).str.strip(),
                "auction_mn": pd.to_numeric(body[2], errors="coerce"),
                "paof_mn": pd.to_numeric(body[3], errors="coerce"),
            }).dropna(subset=["date", "auction_mn"])
        except Exception as exc:  # noqa: BLE001
            print(f"  D2.1A xls unreadable ({exc})")
            out = pd.DataFrame()
    if out.empty and D21A_PDF.exists():
        from issuance.demand import _parse_d21a

        p = _parse_d21a(D21A_PDF)
        if not p.empty:
            out = p.rename(columns={"nominal_auction_mn": "auction_mn", "nominal_paof_mn": "paof_mn"})
            out = out[["date", "name", "auction_mn", "paof_mn"]]
            out.loc[out["paof_mn"] == 0, "paof_mn"] = float("nan")
            data_dt = out["date"].max()
    if out.empty:
        return out
    if pd.isna(data_dt):
        data_dt = out["date"].max()
    pending = out["date"] > data_dt - pd.offsets.BDay(PAOF_PENDING_BDAYS)
    out["paof_mn"] = out["paof_mn"].where(out["paof_mn"].notna() | pending, 0.0)
    out["key"] = out["name"].map(_gilt_key)
    return out.reset_index(drop=True)


def paof_at_takeup(ops: pd.DataFrame, d21a: pd.DataFrame | None = None) -> pd.DataFrame:
    """D2.1E books the post-auction option at the full entitlement (10/15/25% of the auction) on
    many lines; D2.1A carries the take-up. Each auction's D2.1E rows on its settlement day are set
    to D2.1A auction + PAOF: a separate PAOF row takes the residual, a combined row takes the total."""
    d21a = _d21a_results() if d21a is None else d21a
    if ops is None or ops.empty or d21a is None or d21a.empty:
        return ops
    ops = ops.copy()
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    dt = pd.to_datetime(ops["op_dt"], errors="coerce", format="mixed").dt.normalize()
    cand = (ops["source"].astype(str) == "DMO D2.1E") & (ops["sec_type"].astype(str) != "Bill")
    cand &= ops["op_type"].astype(str).isin(["Auction", "Tender"]) & (ops["amount_issued"] > 0)
    keys = ops.loc[cand, "sec_desc_long"].map(_gilt_key)
    pool = pd.DataFrame({"dt": dt[cand], "key": keys, "amt": ops.loc[cand, "amount_issued"]})
    by_key = {k: g for k, g in pool.groupby("key", sort=False) if k is not None}
    # Before the facility D2.1E differs from D2.1A by non-competitive allotments, and D2.1E ties to stock.
    start = d21a.loc[d21a["paof_mn"] > 0, "date"].min()
    drop, n_fix, delta = [], 0, 0.0
    for r in d21a.itertuples(index=False):
        if r.key is None or pd.isna(r.paof_mn) or r.key not in by_key or not r.date >= start:
            continue
        g = by_key[r.key]
        g = g[(g["dt"] >= r.date) & (g["dt"] <= r.date + pd.Timedelta(days=7))]
        if g.empty:
            continue
        g = g[g["dt"] == g["dt"].min()]
        total = float(r.auction_mn) + float(r.paof_mn)
        have = float(g["amt"].sum())
        # Only an overbooked PAOF is corrected: D2.1E below D2.1A is a PAOF settled on another
        # day or D2.1E's own sub-£10mn rounding, and D1A stock ties to D2.1E there.
        excess = have - total
        if excess < 1.0 or excess > PAOF_MAX_SHARE * float(r.auction_mn) + 1.0 or len(g) > 2:
            continue
        if len(g) == 1:
            ops.loc[g.index[0], "amount_issued"] = total
        else:
            big, small = g["amt"].idxmax(), g["amt"].idxmin()
            rest = total - float(g.loc[big, "amt"])
            if rest < 0.5:
                drop.append(small)
            else:
                ops.loc[small, "amount_issued"] = rest
        ops.loc[g.index, "source"] = "DMO D2.1E (PAOF at D2.1A take-up)"
        n_fix += 1
        delta += total - have
    if n_fix:
        print(f"  UK PAOF at D2.1A take-up: {n_fix} auctions, {delta:+,.1f} mn vs D2.1E")
    return ops.drop(index=drop)


CREATION_MIN_LINES = 20


def tag_creations(ops: pd.DataFrame) -> pd.DataFrame:
    """DMO collateral creations (the whole gilt list on one day, for the DMO's own repo and
    SLS pool) are booked in D2.1E as auctions. They are in D1A stock, never sold: own holdings.
    Creations carry no price; an auction or PAOF settling the same day (22-Oct-2008 4¼% 2011) does."""
    if ops is None or ops.empty or "op_type" not in ops.columns:
        return ops
    uk = (ops["country"].astype(str) == "UK") if "country" in ops.columns else pd.Series(True, index=ops.index)
    gilt = uk & (ops["sec_type"].astype(str) != "Bill") & (pd.to_numeric(ops["amount_issued"], errors="coerce") > 0)
    gilt &= ops["op_type"].astype(str).isin(["Auction", "Outright"])
    for col in ("price", "yield"):
        if col in ops.columns:
            gilt &= pd.to_numeric(ops[col], errors="coerce").isna()
    day = pd.to_datetime(ops["op_dt"], errors="coerce", format="mixed").dt.normalize()
    width = ops[gilt].groupby(day[gilt])["isin"].transform("nunique")
    hit = width[width >= CREATION_MIN_LINES].index
    if len(hit):
        ops = ops.copy()
        ops["op_type"] = ops["op_type"].astype(object)
        ops.loc[hit, "op_type"] = "Own Holdings"
    return ops


def _d1a_residual(ops: pd.DataFrame) -> pd.DataFrame:
    """Face in D2.1E that D1A does not show. Dated to FY 2025-26, not the latest auction month.

    The DMO remit records a £0.6bn cash cancellation of the 4⅛% 2027 in that year.
    The nominal is not in D2.1E. The live stock has to equal D1A.
    """
    d1a = _d1a_rows()
    if d1a.empty or ops is None or ops.empty:
        return pd.DataFrame()
    asof = pd.to_datetime(d1a["as_of"], errors="coerce").max()
    book = ops.copy()
    book["isin"] = book["isin"].astype(str).str.strip().str.upper()
    book["op_dt"] = pd.to_datetime(book["op_dt"], errors="coerce", format="mixed")
    book = book[(book["sec_type"] != "Bill") & book["isin"].notna() & (book["op_dt"] <= asof)]
    src = book["source"].astype(str)
    book = book[~src.str.startswith("D1A residual")]
    book["amount_issued"] = pd.to_numeric(book["amount_issued"], errors="coerce").fillna(0)
    cum = book.groupby("isin", as_index=False)["amount_issued"].sum()
    stock = d1a.copy()
    stock["isin"] = stock["isin"].astype(str).str.strip().str.upper()
    stock["amount_nominal_mn"] = pd.to_numeric(stock["amount_nominal_mn"], errors="coerce")
    merged = stock.merge(cum, on="isin", how="left")
    merged["amount_issued"] = merged["amount_issued"].fillna(0)
    merged["gap"] = merged["amount_nominal_mn"] - merged["amount_issued"]
    hit = merged[merged["gap"].abs() > 1.0]
    if hit.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "op_dt": "2026-03-31",
        "isin": hit["isin"].to_numpy(),
        "sec_desc_long": hit["name"].to_numpy(),
        "amount_issued": hit["gap"].to_numpy(),
        "sec_type": hit["sec_type"].to_numpy(),
        "maturity_dt": hit["maturity_dt"].to_numpy(),
        "op_type": "Buyback",
        "source": "D1A residual vs D2.1E (FY2025-26 cash cancellation, nominal unpublished)",
    })
    print(f"  UK D1A residual: {len(out)} isin, {float(out['amount_issued'].sum()):.1f} mn")
    return out


def _write_stock_from_d1a() -> bool:
    d1a = _d1a_rows()
    if d1a.empty:
        return False
    out = d1a.copy()
    out["isin"] = out["isin"].astype("string").str.strip().str.upper()
    out["amount_nominal_mn"] = pd.to_numeric(out["amount_nominal_mn"], errors="coerce")
    out["country"] = "UK"
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    out = out.dropna(subset=["isin"])
    out.to_parquet(CACHE / "stock_uk.parquet", index=False)
    return True


def _bills_from_local_xml() -> pd.DataFrame:
    """D2.2D is nested (di/dd/sdd/far), not the flat gilt row shape."""
    dest = RAW / "uk" / "D2_2D.xml"
    if not dest.exists() or dest.stat().st_size < 5000:
        return pd.DataFrame()
    payload = dest.read_bytes()
    if looks_blocked(payload):
        return pd.DataFrame()
    root = ET.fromstring(payload)
    rows = []
    for di in root.iter():
        if di.tag.split("}")[-1] != "di":
            continue
        maturity = di.attrib.get("MATURITY_DATE")
        kind = di.attrib.get("TBILL_MATURITY_TYPE") or "T-bill"
        for dd in di:
            if dd.tag.split("}")[-1] != "dd":
                continue
            tender = dd.attrib.get("TENDER_DATE")
            for sdd in dd:
                if sdd.tag.split("}")[-1] != "sdd":
                    continue
                far = next((c for c in sdd if c.tag.split("}")[-1] == "far"), None)
                far_a = far.attrib if far is not None else {}
                rows.append({
                    "sec_desc_long": f"UK T-bill {kind}",
                    "op_dt": tender,
                    "issue_dt": sdd.attrib.get("ISSUE_DATE"),
                    "maturity_dt": maturity,
                    "amount_issued": sdd.attrib.get("SIZE_MILLIONS"),
                    "btc": far_a.get("COVER"),
                    "yield": far_a.get("AVERAGE_YIELD_PERCENT"),
                    "price": far_a.get("AVERAGE_PRICE_GBP"),
                    "sec_type": "Bill",
                    "op_type": "Tender",
                    "source": "DMO D2.2D",
                })
    return pd.DataFrame(rows)


def _bills(force: bool = False) -> pd.DataFrame:
    """D2.2D T-bills. Gilts are D2.1E. Local XML from a browser fetch wins over requests."""
    local = _bills_from_local_xml()
    if not local.empty and not force:
        return local
    try:
        _sys_path()
        from assets import gb_issuance_utils as gb
        bills = gb.scrape_treasury_bill_tender_results(use_selenium=False)
    except Exception as exc:
        print(f"  D2.2D scrape unavailable ({exc})")
        return local
    if bills is None or bills.empty:
        return local
    out = bills.copy()
    out["sec_type"] = "Bill"
    out["op_type"] = "Tender"
    out["source"] = "DMO D2.2D"
    if "sec_desc_long" not in out.columns:
        kind = out["maturity_type"] if "maturity_type" in out.columns else "T-bill"
        out["sec_desc_long"] = kind.map(lambda x: f"UK T-bill {x}") if hasattr(kind, "map") else "UK T-bill"
    return out


DMO_REPORTS = (("issuance_history", "D2_1E.xml"), ("gilts_in_issue", "D1A.xml"), ("bill_tenders", "D2_2D.xml"))


def _latest_op(payload: bytes) -> pd.Timestamp:
    try:
        return pd.to_datetime(_gilt_frame(payload)["op_dt"], errors="coerce", format="mixed").max()
    except Exception:  # noqa: BLE001
        return pd.Timestamp.min


def _sync_dmo() -> None:
    """Pull D2.1E / D1A / D2.2D through uk_flows' warmed session.

    A bare request is answered by the DMO bot filter. uk_flows keeps a cookie-carrying session and
    returns its last good copy when blocked, so the local files only ever move forward.
    """
    flows = str(REPO / "uk_flows")
    if flows not in sys.path:
        sys.path.insert(0, flows)
    try:
        import dmo
    except Exception as exc:  # noqa: BLE001
        print(f"  uk_flows.dmo unavailable ({exc})")
        return
    fetcher = dmo.Fetcher(quiet=True)
    for key, name in DMO_REPORTS:
        try:
            payload, _kind = dmo.raw(key, fetcher=fetcher, refresh=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  DMO {key}: {exc}")
            continue
        dest = RAW / "uk" / name
        if looks_blocked(payload[:4000]) or len(payload) < 5000:
            continue
        if key == "issuance_history" and dest.exists():
            # D2.1E shrinks when the DMO merges a PAOF into its auction row: judge it by its last date.
            if _latest_op(payload) < _latest_op(dest.read_bytes()):
                print(f"  DMO {key}: older than the local copy, kept local")
                continue
        elif dest.exists() and dest.stat().st_size > len(payload) * 1.02:
            print(f"  DMO {key}: {len(payload)} bytes is smaller than the local copy, kept local")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        print(f"  DMO {key}: {len(payload)} bytes → {dest.name}")


def fetch_uk(force: bool = False) -> pd.DataFrame:
    ops = None
    if force:
        from issuance.uk_remit import fetch_remit_pdf

        _sync_dmo()
        fetch_remit_pdf()
        force = False
    local_xml = RAW / "uk" / "D2_1E.xml"
    if (
        not force
        and local_xml.exists()
        and local_xml.stat().st_size > 5000
        and not looks_blocked(local_xml.read_bytes()[:4000])
    ):
        ops = _attach_maturity(_from_xml(force=False))
        print(f"  UK gilts from local D2.1E ({local_xml.stat().st_size} bytes, {len(ops)} rows)")
    if ops is None:
        try:
            _sys_path()
            from assets import gb_issuance_utils as gb
            gilts = gb.scrape_gilt_issuance_history(use_selenium=False, sleep_duration=0.2)
            if gilts is not None and not gilts.empty and "ISIN_CODE" in gilts.columns:
                g = gilts.rename(columns={
                    "INSTRUMENT_NAME": "sec_desc_long",
                    "ISIN_CODE": "isin",
                    "ACTUAL_DATE": "op_dt",
                    "ISSUANCE_TYPE": "op_type",
                    "NOMINAL_ISSUED": "amount_issued",
                    "ISSUE_CLEAN_PRICE": "price",
                    "ISSUE_YIELD": "yield",
                })
                g["sec_type"] = g["sec_desc_long"].map(_classify_gilt)
                g["op_type"] = g["op_type"].map(_classify_op)
                g["source"] = "DMO D2.1E"
                ops = g
        except Exception as exc:
            print(f"  DMO live scrape unavailable ({exc})")
    if ops is None or ops.empty:
        try:
            ops = _attach_maturity(_from_xml(force=force))
        except Exception as exc:
            if force and local_xml.exists() and local_xml.stat().st_size > 5000 and not looks_blocked(local_xml.read_bytes()[:4000]):
                print(f"  DMO XML refetch failed ({exc}); using local D2.1E")
                ops = _attach_maturity(_from_xml(force=False))
            else:
                print(f"  DMO XML unavailable ({exc}); using long-end workbook")
                ops = _from_workbook()
                _write_stock_from_workbook()
    ops = paof_at_takeup(ops)
    # D2.1E records collateral creations as Outright and omits later cancellations.
    # These three are the published Donations and Bequests cancellations.
    cancels = pd.DataFrame([
        {"op_dt": "2025-07-24", "isin": "GB00B4PTCY75", "sec_desc_long": "0 3/8% Index-linked Treasury Gilt 2062", "amount_issued": -7.8, "sec_type": "Linker", "maturity_dt": "2062-03-22"},
        {"op_dt": "2025-07-24", "isin": "GB00BL6C7720", "sec_desc_long": "4 1/8% Treasury Gilt 2027", "amount_issued": -14.0, "sec_type": "Nominal", "maturity_dt": "2027-01-29"},
        {"op_dt": "2026-03-13", "isin": "GB00BL6C7720", "sec_desc_long": "4 1/8% Treasury Gilt 2027", "amount_issued": -1.711, "sec_type": "Nominal", "maturity_dt": "2027-01-29"},
    ])
    cancels["op_type"] = "Buyback"
    cancels["source"] = "DMO donations and bequests cancellation"
    bills = _bills(force=force)
    parts = [ops, cancels]
    residual = _d1a_residual(pd.concat([ops, cancels], ignore_index=True))
    if not residual.empty:
        parts.append(residual)
    if bills is not None and not bills.empty:
        parts.append(bills)
        print(f"  UK bills: {len(bills)} rows")
    else:
        print("  UK bills: D2.2D unavailable (captcha or empty); gilts only")
    ops = pd.concat(parts, ignore_index=True)
    ops = coerce_ops(attach_tenor(ops), "UK")
    ops = ops[ops["op_dt"].notna() & (ops["amount_issued"].fillna(0) != 0)]
    ops = tag_creations(ops)
    ops.to_parquet(CACHE / "ops_uk.parquet", index=False)
    if not _write_stock_from_d1a() and not (CACHE / "stock_uk.parquet").exists():
        _write_stock_from_workbook()
    return ops
