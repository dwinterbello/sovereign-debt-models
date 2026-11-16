"""Bank of Japan JGB holdings by issue, outright purchase operations, monthly flow by issue.

Holdings: 'Japanese Government Bonds Held by the Bank of Japan' (by issue, ~3x a month,
face value, 100 million yen). Monthly zip/xls 2013-2016, xlsx from 2016. 2011-2012 are PDF.
Operations: 'Market Operations by the Bank of Japan' monthly workbook, sheet ope1.

Identifier convention (shared with the JGB auction stack): JP{tenor}#{n}, JPIL10y#n,
JPGX{5,10}y#n. Pre-2008 15y floaters are JPFRN15y#n (never auctioned again).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from issuance.http import FetchError, get_bytes, get_text
from issuance.paths import CACHE

BASE = "https://www.boj.or.jp"
MEI_INDEX = BASE + "/en/statistics/boj/other/mei/release/{year}/index.htm"
MEI_CURRENT = BASE + "/en/statistics/boj/other/mei/index.htm"
OPE_INDEX = BASE + "/en/statistics/boj/fm/ope/m_release/{year}/index.htm"
OPE_CURRENT = BASE + "/en/statistics/boj/fm/ope/index.htm"

RAW_DIR = CACHE / "boj_raw"
MEI_DIR = RAW_DIR / "mei"
OPE_DIR = RAW_DIR / "ope"

HOLDINGS = CACHE / "boj_holdings.parquet"
OPERATIONS = CACHE / "boj_operations.parquet"
FLOW = CACHE / "boj_isin_flow.parquet"

FIRST_YEAR = 2013
OKU = 100.0  # 100 million yen -> million yen

# (regex on label, product, id prefix). Order matters: climate / linker / floater before tenor.
_PRODUCTS = [
    (re.compile(r"(5[- ]?year).*climate|5年クライメート", re.I), "GX5y", "JPGX5y"),
    (re.compile(r"(10[- ]?year).*climate|10年クライメート", re.I), "GX10y", "JPGX10y"),
    (re.compile(r"inflation|物価連動", re.I), "IL10y", "JPIL10y"),
    (re.compile(r"floating|変動", re.I), "FRN15y", "JPFRN15y"),
    (re.compile(r"^\s*2[- ]?year|^\s*2年", re.I), "2y", "JP2y"),
    (re.compile(r"^\s*5[- ]?year|^\s*5年", re.I), "5y", "JP5y"),
    (re.compile(r"^\s*10[- ]?year|^\s*10年", re.I), "10y", "JP10y"),
    (re.compile(r"^\s*20[- ]?year|^\s*20年", re.I), "20y", "JP20y"),
    (re.compile(r"^\s*30[- ]?year|^\s*30年", re.I), "30y", "JP30y"),
    (re.compile(r"^\s*40[- ]?year|^\s*40年", re.I), "40y", "JP40y"),
    (re.compile(r"treasury discount|t-bill|国庫短期", re.I), "TB", "JPTB"),
]


def jgb_id(prefix: str, number: int) -> str:
    return f"{prefix}#{int(number)}"


def _product(label: object) -> tuple[str, str] | None:
    if not isinstance(label, str):
        return None
    text = label.strip()
    if not text or len(text) > 60:
        return None
    for pat, product, prefix in _PRODUCTS:
        if pat.search(text):
            return product, prefix
    return None


# ---------------------------------------------------------------- download


def _links(index_url: str, pattern: str) -> list[str]:
    html = get_text(index_url)
    found = re.findall(r'href="([^"]*' + pattern + r')"', html, flags=re.I)
    return sorted({BASE + h if h.startswith("/") else h for h in found})


def _download(urls: list[str], dest_dir: Path, force: bool) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for url in urls:
        dest = dest_dir / url.rsplit("/", 1)[-1]
        if not dest.exists() or force:
            try:
                get_bytes(url, dest=dest, retries=3)
            except FetchError as exc:
                print(f"  boj: skip {url}: {exc}")
                continue
        out.append(dest)
    return out


def download_mei(force: bool = False, first_year: int = FIRST_YEAR) -> list[Path]:
    """Holdings-by-issue files. Past years are immutable: only refetch the index for the current year."""
    this_year = pd.Timestamp.today().year
    urls: list[str] = []
    for year in range(first_year, this_year + 1):
        cached = sorted(MEI_DIR.glob(f"mei{str(year)[2:]}*"))
        if cached and year < this_year and not force:
            continue
        try:
            urls += _links(MEI_INDEX.format(year=year), r"mei\d+(?:en)?\.(?:xlsx|xls|zip)")
        except FetchError as exc:
            print(f"  boj: index {year}: {exc}")
    try:
        urls += _links(MEI_CURRENT, r"mei\d+\.(?:xlsx|xls|zip)")
    except FetchError as exc:
        print(f"  boj: current index: {exc}")
    _download(sorted(set(urls)), MEI_DIR, force)
    return sorted(p for p in MEI_DIR.iterdir() if p.suffix in {".xlsx", ".xls", ".zip"})


def download_ope(force: bool = False, first_year: int = FIRST_YEAR) -> list[Path]:
    this_year = pd.Timestamp.today().year
    urls: list[str] = []
    for year in range(first_year, this_year + 1):
        cached = sorted(OPE_DIR.glob(f"ope{str(year)[2:]}??.*"))
        if cached and year < this_year and not force:
            continue
        try:
            urls += _links(OPE_INDEX.format(year=year), r"ope\d{4}\.(?:xlsx|xls|zip)")
        except FetchError as exc:
            print(f"  boj: ope index {year}: {exc}")
    try:
        urls += _links(OPE_CURRENT, r"m_release/\d{4}/ope\d{4}\.(?:xlsx|xls|zip)")
    except FetchError as exc:
        print(f"  boj: ope current index: {exc}")
    _download(sorted(set(urls)), OPE_DIR, force)
    return sorted(p for p in OPE_DIR.iterdir() if p.suffix in {".xlsx", ".xls", ".zip"})


def _workbooks(path: Path) -> list[tuple[str, bytes]]:
    """(name, bytes) of the English workbook(s) inside a file."""
    if path.suffix != ".zip":
        return [(path.name, path.read_bytes())]
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))]
        en = [n for n in names if "en" in Path(n).stem.lower()] or names
        return [(n, zf.read(n)) for n in en]


def _read_sheets(payload: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(payload), sheet_name=None, header=None)


# ---------------------------------------------------------------- holdings


def _as_of(frame: pd.DataFrame, name: str) -> pd.Timestamp:
    """First date in the top block that is not the release date; falls back to the file name."""
    top = frame.iloc[:8]
    dates = []
    for val in top.values.ravel():
        if _is_date(val) and not pd.isna(val):
            dates.append(pd.Timestamp(val).normalize())
    m = re.search(r"mei(\d{2})(\d{2})(\d{2})?", name)
    from_name = None
    if m:
        yy, mm, dd = m.groups()
        if dd:
            from_name = pd.Timestamp(2000 + int(yy), int(mm), int(dd))
    if dates:
        # The as-of date precedes the release date; release dates sit in the right-hand column.
        return min(dates)
    if from_name is not None:
        return from_name
    raise ValueError(f"no as-of date in {name}")


_SUBTOTAL = re.compile(r"^\s*[23]\.\s*(Asset Purchase Program|Excluding the Asset)", re.I)
_ISSUE = re.compile(r"^\s*(?:#|No\.?)?\s*(\d{1,4})\s*$", re.I)


def _issue_number(val: object) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val) or val != int(val) or not 0 < val < 10000:
            return None
        return int(val)
    m = _ISSUE.match(str(val))
    return int(m.group(1)) if m else None


def _amount(val: object) -> float | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    if isinstance(val, str):
        text = val.replace(",", "").strip()
        if text in {"-", "－", "―"}:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return None
    return None


def parse_holdings_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    """Three layouts, one rule. Product labels are sticky per column. An entry is an issue
    number followed by an amount in the next column, where the issue sits either in the
    label column as '# 312' (2013-14 side-by-side blocks) or one column right of a label
    column (label on every row 2014-Jun 2019; label once per block from Jul 2019)."""
    rows = []
    nrow, ncol = frame.shape
    current: dict[int, tuple[str, str]] = {}
    for r in range(nrow):
        # Pre-QQE files (to Mar-2013) add 'Asset Purchase Program' / 'Excluding ...' splits
        # below the '1. Total' block.
        if any(isinstance(v, str) and _SUBTOTAL.match(v) for v in frame.iloc[r]):
            break
        for c in range(ncol):
            hit = _product(frame.iat[r, c])
            if hit:
                current[c] = hit
        for j in range(ncol - 1):
            cell = frame.iat[r, j]
            n = _issue_number(cell)
            if n is None:
                continue
            amt = _amount(frame.iat[r, j + 1])
            if amt is None:
                continue
            if isinstance(cell, str) and "#" in cell and j in current:
                product, prefix = current[j]
            elif not isinstance(cell, str) and (j - 1) in current:
                product, prefix = current[j - 1]
            else:
                continue
            rows.append({"product": product, "issue_number": n, "isin": jgb_id(prefix, n),
                         "face_mn": amt * OKU})
    return pd.DataFrame(rows, columns=["product", "issue_number", "isin", "face_mn"])


def parse_holdings_file(path: Path) -> pd.DataFrame:
    frames = []
    for name, payload in _workbooks(path):
        for _sheet, frame in _read_sheets(payload).items():
            body = parse_holdings_sheet(frame)
            if body.empty:
                continue
            body.insert(0, "date", _as_of(frame, path.name))
            body["source_file"] = path.name
            frames.append(body)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_holdings(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            body = parse_holdings_file(path)
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the history
            print(f"  boj: parse fail {path.name}: {exc}")
            continue
        if body.empty:
            print(f"  boj: no holdings in {path.name}")
            continue
        frames.append(body)
    if not frames:
        raise FetchError("no BoJ holdings parsed")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["date", "isin"], keep="last")
    out = out.sort_values(["date", "product", "issue_number"]).reset_index(drop=True)
    out["issue_number"] = out["issue_number"].astype("int64")
    return out[["date", "isin", "product", "issue_number", "face_mn", "source_file"]]


# ---------------------------------------------------------------- operations


_BUCKETS = [
    (re.compile(r"up to 1 year", re.I), "0-1y"),
    (re.compile(r"1 year and up to 3", re.I), "1-3y"),
    (re.compile(r"3 years and up to 5", re.I), "3-5y"),
    (re.compile(r"1 year and up to 5", re.I), "1-5y"),
    (re.compile(r"5 years and up to 10", re.I), "5-10y"),
    (re.compile(r"10 years and up to 25", re.I), "10-25y"),
    (re.compile(r"more than 25 years", re.I), "25y+"),
    (re.compile(r"more than 10 years", re.I), "10y+"),
    (re.compile(r"inflation", re.I), "linker"),
    (re.compile(r"floating", re.I), "frn"),
    (re.compile(r"climate", re.I), "gx"),
]


def _bucket(text: str) -> str:
    for pat, label in _BUCKETS:
        if pat.search(text):
            return label
    return text.strip()


def _is_date(val: object) -> bool:
    return isinstance(val, pd.Timestamp) or (hasattr(val, "year") and hasattr(val, "month")
                                              and not isinstance(val, (int, float, str)))


_OLD_INSTR = re.compile(r"^\s*Outright Purchases of JGBs \(([0-9FI\- ]{1,7})\)\s*$", re.I)
_SECTION = re.compile(r"^\s*Outright Purchases of JGBs\b", re.I)
_SPECIFIC = re.compile(r"(\d+)-year JGB #\s*(\d+)", re.I)


def _old_bucket(code: str) -> str:
    code = code.strip()
    if code.upper() == "F":
        return "frn"
    if code.upper() == "I":
        return "linker"
    m = re.match(r"^(\d+)-(\d+)?$", code)
    if not m:
        return code
    lo, hi = m.groups()
    return f"{lo}-{hi}y" if hi else f"{lo}y+"


def _num(val: object) -> float:
    out = _amount(val)
    return float("nan") if out is None else out


def parse_ope_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    """JGB outright purchase rows. Two layouts: to 2016 one table with an instrument column
    'Outright Purchases of JGBs (1-3)'; from 2017 one section per method with the residual
    maturity (or, for 2022-23 fixed-rate ops, the specific issue) as text at the right.
    Offered / bid / accepted sit in the three columns right of the exercise date; offered is
    blank for unlimited fixed-rate operations."""
    rows = []
    section = None
    for r in range(frame.shape[0]):
        vals = list(frame.iloc[r])
        texts = [v.strip() for v in vals if isinstance(v, str)]
        joined = " ".join(texts)
        old = next((m for m in (_OLD_INSTR.match(t) for t in texts) if m), None)
        if old is None:
            if any(_SECTION.match(t) for t in texts):
                section = "fixed_rate" if re.search(r"fixed[- ]rate method", joined, re.I) else "competitive"
                if re.search(r"Floating-rate Bonds and Inflation-indexed Bonds\)", joined, re.I) \
                        and not re.search(r"Excluding", joined, re.I):
                    section = "linker_frn"
                continue
            if texts and re.match(r"^(■|Outright|Purchases|Sales|Funds|Repurchase|Securities|Operations)",
                                  texts[0]) and not _SECTION.match(texts[0]):
                section = None
            if section is None:
                continue
        date_idx = [i for i, v in enumerate(vals) if _is_date(v) and not pd.isna(v)]
        if len(date_idx) < 2:
            continue
        ex = date_idx[1]
        if ex + 3 >= len(vals):
            continue
        offered, bid, accepted = (_num(vals[ex + k]) for k in (1, 2, 3))
        if pd.isna(accepted):
            continue
        offer_dt = pd.Timestamp(vals[date_idx[0]]).normalize()
        exec_dt = pd.Timestamp(vals[ex]).normalize()
        if not pd.Timedelta(0) <= exec_dt - offer_dt <= pd.Timedelta(days=14):
            # Typos in the source (e.g. 2008-12-11 for 2018-12-11): keep the offer year.
            exec_dt = exec_dt.replace(year=offer_dt.year)
            if exec_dt < offer_dt:
                exec_dt = exec_dt.replace(year=offer_dt.year + 1)
        if old is not None:
            label = old.group(0).strip()
            bucket = _old_bucket(old.group(1))
            method = "linker_frn" if bucket in {"linker", "frn"} else "competitive"
        else:
            english = [t for t in texts if re.search(r"[A-Za-z]", t)]
            label = english[-1] if english else (texts[-1] if texts else "")
            bucket = _bucket(label)
            method = section
        spec = _SPECIFIC.search(label)
        isin = jgb_id(f"JP{spec.group(1)}y", int(spec.group(2))) if spec else None
        if spec:
            bucket = "specific_issue"
        rows.append({
            "offer_date": offer_dt,
            "date": exec_dt,
            "method": method,
            "maturity_text": label,
            "bucket": bucket,
            "isin": isin,
            "offered_mn": offered * OKU,
            "bid_mn": bid * OKU,
            "accepted_mn": accepted * OKU,
        })
    return pd.DataFrame(rows)


def build_operations(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            for name, payload in _workbooks(path):
                for _sheet, frame in _read_sheets(payload).items():
                    body = parse_ope_sheet(frame)
                    if not body.empty:
                        body["source_file"] = path.name
                        frames.append(body)
        except Exception as exc:  # noqa: BLE001
            print(f"  boj: ope parse fail {path.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["offer_date", "date", "maturity_text", "offered_mn", "accepted_mn"])
    return out.sort_values(["date", "bucket"]).reset_index(drop=True)


# ---------------------------------------------------------------- purchase plan

# Monthly outright purchase amount (yen trillion, approx., incl. linkers) by calendar quarter.
# MPM statements of 31 Jul 2024 (cut ~0.4tn per quarter to ~2.9tn in Q1-2026) and
# 17 Jun 2025 (cut ~0.2tn per quarter from Q2-2026 to ~2.1tn in Q1-2027). 16 Jun 2026 kept that
# path and set ~2.0tn a month from Apr 2027 with no further interim assessment; the last quarter
# here is held in the forward. Not published as data.
PLAN_TN = {
    "2024Q3": 5.3, "2024Q4": 4.9,
    "2025Q1": 4.5, "2025Q2": 4.1, "2025Q3": 3.7, "2025Q4": 3.3,
    "2026Q1": 2.9, "2026Q2": 2.7, "2026Q3": 2.5, "2026Q4": 2.3,
    "2027Q1": 2.1, "2027Q2": 2.0,
}
PLAN = CACHE / "boj_purchase_plan.parquet"


def build_plan(ops: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for q, tn in PLAN_TN.items():
        for month in pd.period_range(pd.Period(q, "Q").asfreq("M", "start"),
                                     pd.Period(q, "Q").asfreq("M", "end"), freq="M"):
            if month < pd.Period("2024-08", "M"):
                continue
            rows.append({"month": month.to_timestamp(), "quarter": q, "planned_mn": tn * 1e6})
    plan = pd.DataFrame(rows)
    if ops is not None and not ops.empty:
        realised = ops.groupby(ops["date"].dt.to_period("M").dt.to_timestamp())["accepted_mn"].sum()
        last_full = ops["date"].max().to_period("M").to_timestamp()
        plan["realised_mn"] = plan["month"].map(realised).where(plan["month"] <= last_full)
    return plan


# ---------------------------------------------------------------- flow


def build_flow(holdings: pd.DataFrame) -> pd.DataFrame:
    """Month-end holdings by issue (last observation in the month), flow = change.

    BoJ does not sell JGBs outright; an issue that leaves the list is a redemption. The
    exit month carries flow = -prior holdings with matured=True. Issues that drop out
    only because a file is missing would be misflagged, so exits are confirmed against
    the next two observation dates."""
    h = holdings.copy()
    h["month"] = h["date"].dt.to_period("M").dt.to_timestamp()
    last_date = h.groupby("month")["date"].max().rename("snap")
    snap = h.merge(last_date, left_on=["month", "date"], right_on=["month", "snap"])
    snap = snap[["month", "isin", "product", "issue_number", "face_mn"]]
    months = sorted(snap["month"].unique())
    wide = snap.pivot_table(index="month", columns="isin", values="face_mn", aggfunc="last")
    wide = wide.reindex(months)
    meta = snap.drop_duplicates("isin", keep="last").set_index("isin")[["product", "issue_number"]]
    held = wide.fillna(0.0)
    present = wide.notna()
    flow = held.diff()
    flow.iloc[0] = float("nan")  # first snapshot is a stock, not a flow
    # Exit: present last month, absent this month and every later month.
    later_absent = (~present)[::-1].cummin()[::-1]
    exited = present.shift(1, fill_value=False) & later_absent
    out = pd.DataFrame({
        "flow_mn": flow.stack(),
        "holdings_mn": held.stack(),
        "matured": exited.stack(),
        "present": present.stack(),
    }).reset_index()
    out.columns = ["month", "isin", "flow_mn", "holdings_mn", "matured", "present"]
    out = out[out["present"] | out["matured"] | (out["flow_mn"] != 0)].drop(columns="present")
    out = out.join(meta, on="isin")
    out["month"] = pd.to_datetime(out["month"])
    out = out[["month", "isin", "product", "issue_number", "flow_mn", "holdings_mn", "matured"]]
    out["purchase_mn"] = out["flow_mn"].where(~out["matured"], 0.0)
    out["redemption_mn"] = out["flow_mn"].where(out["matured"], 0.0)
    return out.sort_values(["month", "product", "issue_number"]).reset_index(drop=True)


# ---------------------------------------------------------------- entry


def fetch_boj(force: bool = False, offline: bool = False, with_operations: bool = True) -> pd.DataFrame:
    if offline:
        mei = sorted(p for p in MEI_DIR.glob("mei*") if p.suffix in {".xlsx", ".xls", ".zip"})
        ope = sorted(p for p in OPE_DIR.glob("ope*") if p.suffix in {".xlsx", ".xls", ".zip"})
    else:
        mei = download_mei(force=force)
        ope = download_ope(force=force) if with_operations else []
    holdings = build_holdings(mei)
    holdings.to_parquet(HOLDINGS, index=False)
    flow = build_flow(holdings)
    flow.to_parquet(FLOW, index=False)
    if with_operations and ope:
        ops = build_operations(ope)
        if not ops.empty:
            ops.to_parquet(OPERATIONS, index=False)
            build_plan(ops).to_parquet(PLAN, index=False)
    totals = holdings.groupby("date")["face_mn"].sum()
    print(f"  boj: {holdings['date'].nunique()} snapshots {holdings['date'].min():%Y-%m-%d}"
          f"..{holdings['date'].max():%Y-%m-%d}; latest total {totals.iloc[-1] / 1e6:.1f}tn yen")
    return holdings


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-ops", action="store_true")
    args = ap.parse_args()
    fetch_boj(force=args.force, offline=args.offline, with_operations=not args.no_ops)
