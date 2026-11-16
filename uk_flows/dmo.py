"""UK Debt Management Office: gilt stock, issuance operations, official holdings, bill supply.

The DMO is the primary source for everything on the supply side of the gilt market, and unlike the
CFTC it publishes at instrument level with no lag. Eighteen reports are pulled here; between them
they cover the stock in issue by ISIN, every issuance operation ever conducted with its bidding
statistics, the government's own holdings, the forward issuance calendar, and the bill programme.

Two things make this awkward and both are handled below.

The site sits behind ShieldSquare bot protection. A plain `requests.get` returns a captcha page with
a 200 status, so a naive scraper caches the captcha as if it were data and silently corrupts the
archive. Every payload is therefore validated before it is written, requests are paced, and a
blocked response triggers exponential backoff rather than a retry loop.

The reports come in two flavours. Four of the most useful ones offer an XML endpoint with tagged
fields, which is stable and worth preferring. The rest come from a spreadsheet export whose CSV has
a title block, a date stamp, blank spacer rows and sometimes two header rows stacked on top of each
other, so the header has to be located rather than assumed.

    python dmo.py                       # refresh the cache, build every table
    python dmo.py --refresh             # ignore cache freshness and re-pull

Tables land in output/ as parquet plus csv.gz. Nothing here needs an API key.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import urllib.parse
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import progress

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "dmo"
OUTPUT = HERE / "output"

BASE = "https://www.dmo.gov.uk"
PACE_SECONDS = 14.0                 # gap between requests; the site rate-limits hard below ~10s
# Waits after a captcha. Deliberately short: rate-limiting from pacing too fast clears in tens of
# seconds, and an address-level block does not clear within any wait worth sitting through. Two
# minutes tells the two apart, which is all the ladder is for.
BACKOFF_SECONDS = (30, 90)
TIMEOUT = 120

AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/126.0.0.0 Safari/537.36")


class Blocked(RuntimeError):
    """The site served bot protection instead of data. Worth waiting and retrying."""


class Unavailable(RuntimeError):
    """The endpoint does not exist for this report. Retrying will never help."""


@dataclass(frozen=True)
class Report:
    key: str            # our name, and the cache filename
    code: str           # DMO report code
    label: str
    kind: str           # the export extension this report offers: xls, xlsx, csv or xml
    stale_hours: float  # re-pull once the cached copy is older than this
    role: str           # why we want it
    years: bool = False  # ask for every financial year, for reports that otherwise give the latest


# The kinds below are what each report actually offers, read off the export configuration the report's
# own page embeds. The index at /data/ links some reports only as a PDF, which reads like the data is
# unavailable; it is not. Every report in the catalogue exports, and the extension is the difference
# between the whole history and nothing: gilt auction results are a PDF on the index and 1,245 rows of
# cover, tail and average accepted price through the export endpoint.


REPORTS = (
    # ---- the stock ------------------------------------------------------------------------
    Report("gilts_in_issue", "D1A", "Gilts in issue", "xml", 20,
           "nominal in issue by ISIN, the denominator for every share"),
    # D1D, index-linked gilts in issue, is deliberately absent: D1A already carries every linker
    # with its base RPI and its uplifted amount.
    Report("redeemed", "D1C", "Redemption details of redeemed gilts", "xls", 24 * 7,
           "when each matured gilt left the stock, which is what closes the stock history"),
    Report("market_value", "D9D", "Estimated market value of gilts", "xls", 20,
           "the DMO's own market value per gilt, an independent check on the bond maths here"),
    # ---- the official sector -------------------------------------------------------------
    Report("official_holdings", "D4L", "Government holdings of gilts", "xls", 20,
           "CRND and DMO holdings by gilt, monthly since 2005: the public sector minus the Bank",
           years=True),
    Report("crnd_funds", "D6A", "Funds managed by CRND", "xls", 24 * 7,
           "the market value of the funds behind those holdings"),
    # ---- operations ----------------------------------------------------------------------
    Report("issuance_history", "D2.1E", "Gilt issuance history", "xml", 20,
           "every operation ever with its awarded yield: the supply spine and the concession"),
    Report("auctions", "D2.1A", "Outright gilt auctions", "xls", 20,
           "cover, tail and average accepted price: demand elasticity at the point of sale"),
    Report("syndications", "D10A", "Results of syndicated offerings", "csv", 20,
           "the long-end and linker supply that does not go through an auction"),
    Report("tenders", "D2.1PROF9", "Gilt tenders, outright sales", "xls", 24 * 7,
           "tenders, used for taps and small top-ups"),
    Report("other_operations", "D2.1PROF7", "Results of other gilt operations", "xls", 24 * 7,
           "conversions, switches and reverse auctions, which move stock without funding anything"),
    Report("repo_facility", "D2.1PROF5", "Use of standing and special gilt repo facility",
           "xls", 20, "the cleanest public squeeze indicator in the gilt market"),
    # ---- supply that has been announced but not yet sold ---------------------------------
    Report("calendar", "D5D", "Outright gilt issuance calendar", "xls", 20,
           "announced operations ahead of the event: this supply is known before it lands"),
    Report("future_redemptions", "D8B", "Future redemptions", "xls", 24 * 7,
           "the redemption leg of net supply, out to the longest maturity"),
    # ---- bills ---------------------------------------------------------------------------
    Report("bill_tenders", "D2.2D", "Treasury bill tender results", "xml", 20,
           "weekly bill cover and tail, the short-end analogue of auction cover"),
    Report("bills_stock", "D2.2G", "Treasury bill issuance and stock", "xls", 20,
           "bill stock, which swings the front end of net supply"),
    # ---- inflation and yields ------------------------------------------------------------
    Report("monthly_yields", "D4H", "Historical average daily conventional gilt yields",
           "xls", 24 * 7, "a long yield history, and a check on the fitted curve"),
    Report("rpi", "D4O", "RPI data", "xml", 24 * 7,
           "the index behind linker cash flows, and the linker uplift through history"),
    Report("index_ratios", "D10C", "Index ratios, three-month lag linkers", "xml", 20,
           "converts linker nominal into uplifted cash"),
)

BY_KEY = {report.key: report for report in REPORTS}


# ---------------------------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------------------------

def _load_cookie() -> str | None:
    """Optional browser cookie to pass DMO ShieldSquare when the address is warm-listed."""
    env = os.environ.get("DMO_COOKIE", "").strip()
    if env:
        return env
    path = HERE / "secrets" / "dmo_cookie.txt"
    if not path.exists():
        return None
    parts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts.append(line.split(";", 1)[0].strip())
    return "; ".join(parts) if parts else None


def _session() -> requests.Session:
    """A browser-shaped session, warmed on the data index so the bot filter issues cookies."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    })
    cookie = _load_cookie()
    if cookie:
        session.headers["Cookie"] = cookie
    try:
        session.get(f"{BASE}/data/", timeout=TIMEOUT)
    except requests.RequestException:
        pass                                    # the warm-up is a courtesy, not a requirement
    return session


EXPORT = f"{BASE}/umbraco/surface/DataExport/GetDataExport"


def _candidates(report: Report) -> tuple[tuple[str, str], ...]:
    """(kind, url) for this report, in the shape the site's own export form uses.

    Three details had to be taken from the site's JavaScript rather than guessed, and each one turns
    a silent failure into data:

    exportFormatValue wants the file extension, not the numeric format id the page's JSON lists
    beside it. An id is accepted and answered with "Unable to fulfil the report request".

    parameters is one URL-encoded string of "&Name=Value" pairs, named exactly as the report's own
    form fields are. Reports with a financial-year parameter return only the current year unless it
    is passed, so history needs "(All)".

    The path is under /umbraco/surface/, not /data/. The /data/ URLs the report index links are the
    human pages: they answer 200 with HTML, so treating them as data yields a web page silently
    rather than an error.
    """
    if report.kind == "xml":
        # Six reports are served by this older endpoint as flat XML, one row per record. Where it is
        # offered it is preferred: the spreadsheet exports of the same reports are page-formatted,
        # with a title block, merged headers and a footnote, and have to be dug out of a sheet.
        return (("xml", f"{BASE}/data/XmlDataReport?reportCode={report.code}"),)
    tail = f"?reportCode={report.code}&exportFormatValue={report.kind}"
    if report.years:
        tail += "&parameters=" + urllib.parse.quote("&Financial Year=(All)", safe="")
    return ((report.kind, f"{EXPORT}{tail}"),)


def _looks_blocked(text: str) -> bool:
    lowered = text[:4000].lower()
    return any(mark in lowered for mark in
               ("shieldsquare", "request unsuccessful", "incapsula", "captcha",
                "are you a human", "bot detection"))


def _validate(payload: bytes, report: Report, kind: str) -> None:
    """Refuse anything that is not plausibly the report, so the cache stays trustworthy."""
    text = payload.decode("utf-8", errors="replace")
    if _looks_blocked(text):
        raise Blocked(f"{report.code}: bot protection served instead of data")
    if "Unable to fulfil the report request" in text[:200]:
        # The export endpoint's way of rejecting a parameter or a format it does not offer. It comes
        # back as a 200 with an attachment header, so nothing else here would catch it.
        raise Unavailable(f"{report.code}: the export endpoint rejected this request")
    if len(payload) < 200:
        raise Unavailable(f"{report.code}: {len(payload)} bytes is too small to be the report")
    if kind in ("xls", "xlsx"):
        if not payload.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
            raise Unavailable(f"{report.code}: the {kind} export is not a workbook")
        return
    if kind == "xml":
        try:
            root = ET.fromstring(text)
        except ET.ParseError as error:
            raise Unavailable(f"{report.code}: XML did not parse ({error})") from error
        if len(list(root)) == 0:
            raise Unavailable(f"{report.code}: XML has no rows")
    else:
        if text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
            raise Unavailable(f"{report.code}: export returned a web page, not data")
        if "," not in text and "\t" not in text:
            raise Unavailable(f"{report.code}: export has no delimiter")


def _path(report: Report, kind: str) -> Path:
    return CACHE / f"{report.key}.{kind}"


def cached(report: Report) -> tuple[Path, str] | tuple[None, None]:
    """The cached copy of a report and which endpoint produced it, if there is one."""
    for kind, _ in _candidates(report):
        path = _path(report, kind)
        if path.exists():
            return path, kind
    return None, None


def _age_hours(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 3600


# When the DMO blocks an address it blocks the address, not the request: every subsequent report will
# be blocked too. Backing off through the ladder on all eighteen of them costs hours and buys nothing,
# so once this many reports have exhausted their backoffs the run gives up on the source and says so.
# Re-running is how you retry.
#
# The count lives here rather than on the fetcher because the tidiers each ask for their own report
# and would otherwise each get a fresh fetcher, a fresh count, and another turn through the ladder.
GIVE_UP_AFTER = 2
BLOCKED_REPORTS = 0


def shut_out() -> bool:
    return BLOCKED_REPORTS >= GIVE_UP_AFTER


class Fetcher:
    """Paces requests across a run and remembers when it was last blocked."""

    def __init__(self, quiet: bool = False):
        self.session = _session()
        self.last_request = 0.0
        self.quiet = quiet

    def _wait(self) -> None:
        gap = time.time() - self.last_request
        if gap < PACE_SECONDS:
            time.sleep(PACE_SECONDS - gap)

    def _once(self, report: Report, kind: str, url: str) -> bytes:
        self._wait()
        self.last_request = time.time()
        try:
            response = self.session.get(url, timeout=TIMEOUT)
        except requests.RequestException as error:
            raise Blocked(f"{report.code}: {error}") from error
        if response.status_code in (404, 400, 500):
            raise Unavailable(f"{report.code}: {kind} endpoint returned {response.status_code}")
        response.raise_for_status()
        _validate(response.content, report, kind)
        return response.content

    def get(self, report: Report) -> tuple[bytes, str]:
        """Try each endpoint. Back off only for bot protection; a missing endpoint is final."""
        global BLOCKED_REPORTS
        if shut_out():
            raise Blocked(f"{report.code}: not tried, this address is shut out of the DMO")

        dead: list[str] = []
        ladder = (0,) if BLOCKED_REPORTS else (0,) + BACKOFF_SECONDS
        for kind, url in _candidates(report):
            blocked = False
            for attempt, backoff in enumerate(ladder):
                if backoff:
                    if not self.quiet:
                        progress.write(f"    blocked on {report.code}, waiting {backoff}s "
                                       f"(attempt {attempt + 1})")
                    time.sleep(backoff)
                    self.session = _session()       # fresh cookies after a block
                try:
                    result = self._once(report, kind, url), kind
                except Unavailable as error:
                    dead.append(str(error))
                    break                           # this endpoint is gone; try the other
                except Blocked:
                    blocked = True
                    continue
                else:
                    BLOCKED_REPORTS = 0              # it is answering again
                    return result
            if blocked:
                break               # bot protection is about the address, not the endpoint
        if dead:
            raise Unavailable("; ".join(dead))

        BLOCKED_REPORTS += 1
        if shut_out() and not self.quiet:
            progress.write("    the DMO is blocking this address; giving up on it for this run. "
                           "See START_HERE.md, or fill inputs/issuance.csv")
        raise Blocked(f"{report.code}: blocked after {len(ladder) - 1} backoffs")


OFFLINE = False     # set by build(fetch=False); makes every read come from the cache alone


def raw(report_key: str, fetcher: Fetcher | None = None,
        refresh: bool = False) -> tuple[bytes, str]:
    """Cached bytes for one report, and the endpoint they came from.

    The cache is the archive. A failed pull returns the stale copy rather than nothing, and a
    rejected payload is never written, so the archive only ever holds real data.
    """
    report = BY_KEY[report_key]
    path, kind = cached(report)
    if path is not None and (OFFLINE or (not refresh
                                         and _age_hours(path) < report.stale_hours)):
        return path.read_bytes(), kind
    if OFFLINE:
        raise Unavailable(f"{report.code}: not in the cache and the network is switched off")

    fetcher = fetcher or Fetcher()
    try:
        payload, kind = fetcher.get(report)
    except (Blocked, Unavailable, requests.RequestException) as error:
        if path is not None:
            progress.write(f"    {report.code}: keeping cached copy ({error})")
            return path.read_bytes(), kind
        raise

    CACHE.mkdir(parents=True, exist_ok=True)
    target = _path(report, kind)
    target.write_bytes(payload)
    (CACHE / f"{report.key}.meta.json").write_text(json.dumps({
        "code": report.code,
        "label": report.label,
        "kind": kind,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "bytes": len(payload),
    }, indent=2))
    return payload, kind


def refresh_all(refresh: bool = False, quiet: bool = False) -> pd.DataFrame:
    """Pull every report once, pacing throughout. Returns what happened, per report."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(quiet=quiet)
    rows = []
    with progress.bar(total=len(REPORTS), label="DMO reports", unit="report",
                      enabled=not quiet) as step:
        for report in REPORTS:
            before_path, _ = cached(report)
            before = before_path.stat().st_mtime if before_path else None
            status, size, kind = "cached", 0, report.kind
            try:
                payload, kind = raw(report.key, fetcher=fetcher, refresh=refresh)
                size = len(payload)
                after_path, _ = cached(report)
                after = after_path.stat().st_mtime if after_path else None
                if before is None:
                    status = "new"
                elif after != before:
                    status = "refreshed"
            except Unavailable as error:
                progress.write(f"    {report.code} unavailable: {error}")
                status = "unavailable"
            except Exception as error:                       # noqa: BLE001 - report and continue
                progress.write(f"    {report.code} failed: {error}")
                status = "failed"
            rows.append({"key": report.key, "code": report.code, "label": report.label,
                         "kind": kind, "status": status, "bytes": size})
            step(f"{report.code} {status}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# reading: one generic reader per endpoint shape
# ---------------------------------------------------------------------------------------------

def _snake(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def _row_parent(root: ET.Element) -> ET.Element:
    """The element whose children are the rows.

    Most reports put the rows directly under the root, but the official holdings report wraps them in
    a second element of the same name. Descending through single-child wrappers finds the rows in both
    cases without hard-coding either.
    """
    element = root
    while len(list(element)) == 1 and len(list(list(element)[0])) > 0:
        only = list(element)[0]
        # Stop if this looks like a one-row report rather than a wrapper: a wrapper's child has
        # siblings-to-be, a row's child is a field.
        if len(list(only)) == 1 and not list(list(only)[0]):
            break
        element = only
    return element


def _flatten(element: ET.Element, into: dict[str, str], prefix: str = "") -> None:
    """Collect one row's fields, wherever the DMO chose to put them.

    Three conventions appear across these reports and a single row can use two of them at once:
    values as attributes of the row, values as attributes of a child, and values as the text of a
    leaf element. All three become columns. A name that collides with one already taken is qualified
    by its parent's tag rather than silently overwriting it.
    """
    def put(name: str, value: str) -> None:
        key = _snake(name)
        if key in into and into[key] != value:
            key = _snake(f"{prefix}_{name}") if prefix else key
        into.setdefault(key, value)

    for name, value in element.attrib.items():
        put(name, value)
    children = list(element)
    if not children:
        text = (element.text or "").strip()
        if text:
            put(element.tag, text)
        return
    for child in children:
        _flatten(child, into, prefix=element.tag)


def _xml_frame(payload: bytes) -> pd.DataFrame:
    """One row per record, whether the DMO used attributes, child attributes or element text."""
    root = ET.fromstring(payload.decode("utf-8", errors="replace"))
    rows = []
    for record in _row_parent(root):
        fields: dict[str, str] = {}
        _flatten(record, fields)
        if fields:
            rows.append(fields)
    return pd.DataFrame(rows)


def _decode(payload: bytes) -> str:
    """Text from a DMO export. The CSV exports are Windows-1252, so a pound sign is one byte."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _value_share(row: pd.Series) -> float:
    """What fraction of the populated cells in a row are numbers or dates rather than labels."""
    cells = [cell for cell in row if cell is not None and not pd.isna(cell)]
    if not cells:
        return 0.0
    hits = 0
    for cell in cells:
        if isinstance(cell, (int, float, pd.Timestamp, datetime)):
            hits += 1
            continue
        text = str(cell).strip()
        if re.fullmatch(r"[-+(]?[\d,]+(\.\d+)?\)?%?", text) or re.match(r"^\d{4}-\d{2}-\d{2}", text):
            hits += 1
    return hits / len(cells)


def _table_from_grid(table: pd.DataFrame) -> pd.DataFrame:
    """Find the header inside a page-formatted export and return what is under it.

    Every export is laid out for a printed page: a title block of unpredictable depth, then the
    header, then the data, then footnotes. The header is taken to be the first row that has at least
    three labelled cells, is not mostly numbers, and is followed by rows about as wide as itself.
    """
    if table.empty:
        return table

    filled = table.notna().sum(axis=1)
    values = pd.Series([_value_share(table.iloc[index]) for index in range(len(table))])

    # The header is the text row immediately above the data. Looking for the first text row instead
    # finds the title block, which is also text and also wide enough to look plausible; looking for
    # the widest row finds whichever data row happens to have no gaps.
    data = values.index[(values >= 0.5) & (filled >= 2)]
    header = None
    if len(data):
        first = int(data[0])
        above = [index for index in range(first)
                 if filled.iloc[index] >= 2 and values.iloc[index] < 0.5]
        header = above[-1] if above else (first - 1 if first else None)
    if header is None:
        header = int(filled.idxmax())

    body = table.iloc[header + 1:].copy()
    body.columns = [_snake(value) if isinstance(value, str) and value.strip() else f"column_{i}"
                    for i, value in enumerate(table.iloc[header])]
    body = body[body.notna().sum(axis=1) >= 2]      # drop blank spacers and footnote lines
    return body.reset_index(drop=True)


def _csv_frame(payload: bytes) -> pd.DataFrame:
    text = _decode(payload)
    delimiter = "\t" if text.count("\t") > text.count(",") else ","
    table = pd.read_csv(io.StringIO(text), header=None, sep=delimiter, dtype=str,
                        engine="python", on_bad_lines="skip")
    return _table_from_grid(table)


def sheet(payload: bytes, which: int = 0) -> pd.DataFrame:
    """One sheet of a workbook export, unparsed: the raw grid, headers and title block included.

    Kept separate from _sheet_frame because several of these reports are cross-tabs, where the
    header row is a row of dates and the shape has to be read before it can be reshaped.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = pd.read_excel(io.BytesIO(payload), sheet_name=None, header=None)
    return list(book.values())[which]


def _sheet_frame(payload: bytes) -> pd.DataFrame:
    return _table_from_grid(sheet(payload))


def frame(report_key: str, fetcher: Fetcher | None = None,
          refresh: bool = False) -> pd.DataFrame:
    """The report as a raw table, whichever export produced it."""
    payload, kind = raw(report_key, fetcher=fetcher, refresh=refresh)
    if kind == "xml":
        return _xml_frame(payload)
    if kind in ("xls", "xlsx"):
        return _sheet_frame(payload)
    return _csv_frame(payload)


# ---------------------------------------------------------------------------------------------
# tidying helpers
# ---------------------------------------------------------------------------------------------

VULGAR = {"¼": 0.25, "½": 0.5, "¾": 0.75, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
          "⅓": 1 / 3, "⅔": 2 / 3, "⅙": 1 / 6, "⅚": 5 / 6}


def parse_coupon(name: str) -> float | None:
    """Pull the coupon out of a gilt name: `0 3/8% Treasury Gilt 2026` is 0.375, `3¾%` is 3.75."""
    if not isinstance(name, str):
        return None
    head = name.split("%")[0].strip()
    if not head:
        return None
    total = 0.0
    seen = False
    for symbol, value in VULGAR.items():
        if symbol in head:
            total += value
            head = head.replace(symbol, " ")
            seen = True
    fraction = re.search(r"(\d+)\s*/\s*(\d+)", head)
    if fraction:
        total += int(fraction.group(1)) / int(fraction.group(2))
        head = head[:fraction.start()] + " " + head[fraction.end():]
        seen = True
    whole = re.search(r"(\d+(?:\.\d+)?)", head)
    if whole:
        total += float(whole.group(1))
        seen = True
    return round(total, 6) if seen else None


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "fiu":
        return series.astype("float64")
    cleaned = (series.astype(str)
               .str.replace(",", "", regex=False)
               .str.replace("£", "", regex=False)
               .str.replace("%", "", regex=False)
               .str.strip()
               .replace({"": None, "-": None, "n/a": None, "N/A": None, "nan": None,
                         "None": None, "..": None}))
    return pd.to_numeric(cleaned, errors="coerce")


def _dates(series: pd.Series) -> pd.Series:
    """Parse dates without guessing the convention.

    The XML endpoints emit ISO stamps and the spreadsheet exports emit UK day-first strings. Passing
    dayfirst at an ISO stamp silently swaps day and month whenever both are twelve or less, which
    corrupts a minority of rows and leaves the rest looking fine, so the shape is detected first.
    """
    if series.dtype.kind == "M":
        return series
    text = series.astype(str).str.strip()
    sample = text[text.str.len() > 5]
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")

    iso = sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if iso > 0.8:
            return pd.to_datetime(text.str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _dateish(series: pd.Series) -> bool:
    """Whether the values themselves look like dates, rather than the column merely being named one.

    A column of bare numbers must not be parsed as dates however it is labelled. Several of these
    exports put the report date in a header cell, so a column of amounts can end up named
    data_date_11_aug_2026, and a plain number parses as nanoseconds since 1970 without complaint:
    £92,580 million becomes a timestamp in January 1970, which is a plausible enough date to pass
    every check except this one.
    """
    if series.dtype.kind == "M":
        return True
    if series.dtype.kind in "fiu":
        return False
    text = series.dropna().astype(str).str.strip()
    if text.empty:
        return False
    return bool(text.str.contains(r"[-/]|\b[A-Z][a-z]{2}\b").mean() > 0.6)


def _pick(frame_in: pd.DataFrame, *keywords: str, exclude: tuple[str, ...] = ()) -> str | None:
    """First column whose name contains all keywords and none of the exclusions."""
    for column in frame_in.columns:
        name = str(column)
        if all(word in name for word in keywords) and not any(bad in name for bad in exclude):
            return column
    return None


def bucket(years: pd.Series) -> pd.Series:
    """The DMO's own maturity blocks, which is how the remit and the market both talk."""
    edges = [-float("inf"), 7, 15, 25, float("inf")]
    labels = ["short", "medium", "long", "ultra_long"]
    return pd.cut(years, bins=edges, labels=labels, right=False)


# ---------------------------------------------------------------------------------------------
# tidy: gilts in issue
# ---------------------------------------------------------------------------------------------

def gilts(fetcher: Fetcher | None = None, refresh: bool = False) -> pd.DataFrame:
    """Every gilt in issue, one row each, with nominal and uplifted amounts.

    Amounts are in £ millions nominal. For index-linked gilts the uplifted column is the amount a
    holder is actually owed, which is the one to use when measuring the size of the market; the
    nominal column is the one the DMO uses when reporting issuance against the remit.
    """
    table = frame("gilts_in_issue", fetcher=fetcher, refresh=refresh)
    out = pd.DataFrame()
    out["isin"] = table[_pick(table, "isin")].str.strip()
    out["name"] = table[_pick(table, "instrument_name")].str.strip()
    out["instrument_type"] = table[_pick(table, "instrument_type")].str.strip()
    out["dmo_bracket"] = table[_pick(table, "maturity_bracket")].str.strip()
    out["as_of"] = _dates(table[_pick(table, "close_of_business")])
    out["redemption_date"] = _dates(table[_pick(table, "redemption_date")])
    out["first_issue_date"] = _dates(table[_pick(table, "first_issue")])
    out["amount_nominal_mn"] = _numeric(table[_pick(table, "total_amount_in_issue")])
    uplift = _pick(table, "including_il_uplift")
    out["amount_uplifted_mn"] = _numeric(table[uplift]) if uplift else out["amount_nominal_mn"]
    base = _pick(table, "base_rpi")
    out["base_rpi"] = _numeric(table[base]) if base else pd.NA

    out["coupon"] = out["name"].map(parse_coupon)
    out["linker"] = out["instrument_type"].str.contains("index", case=False, na=False)
    out["years_to_maturity"] = ((out["redemption_date"] - out["as_of"]).dt.days / 365.25)
    out["bucket"] = bucket(out["years_to_maturity"])
    out["uplift_ratio"] = out["amount_uplifted_mn"] / out["amount_nominal_mn"]
    return out.sort_values(["linker", "redemption_date"]).reset_index(drop=True)


MONTH_HEADER = re.compile(r"^([A-Z][a-z]{2})-\s*(\d{4})$")
ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{10}")


def _has_isin(line: pd.Series) -> bool:
    return any(isinstance(cell, str) and ISIN.fullmatch(cell.strip()) for cell in line)


def official(fetcher: Fetcher | None = None, refresh: bool = False) -> pd.DataFrame:
    """Gilts held by the government, by gilt and by month end, since April 2005.

    This is the second half of the official sector and the half that is usually missed. The report's
    own note is explicit: it counts the Commissioners for the Reduction of the National Debt and the
    DMO, and it excludes the Bank of England. So it does not overlap the APF at all, and the two add:
    the Bank's rebuilt portfolio is the QE holding, this is everything else the state owns, and free
    float is the stock less both.

    The export is laid out for printing, one block per financial year with the months as columns, so
    it is read by finding each block's month header and melting the rows underneath it. Amounts are
    £ millions nominal.
    """
    payload, kind = raw("official_holdings", fetcher=fetcher, refresh=refresh)
    if kind == "xml":
        # The XML endpoint serves the current financial year only, and in pounds rather than millions.
        table = _xml_frame(payload)
        out = pd.DataFrame({
            "isin": table["isin"].astype(str).str.strip(),
            "name": table["instrument"].astype(str).str.strip(),
            "month_end": pd.to_datetime(table["month"], format="%b-%Y", errors="coerce"),
            "holding_nominal_mn": _numeric(table["amount"]) / 1e6,
        })
    else:
        grid = sheet(payload)
        rows: list[dict[str, object]] = []
        months: dict[int, pd.Timestamp] = {}
        for position in range(len(grid)):
            line = grid.iloc[position]
            found = {}
            for column, cell in line.items():
                if isinstance(cell, str):
                    match = MONTH_HEADER.match(re.sub(r"\s+", "", cell))
                    if match:
                        found[column] = pd.Period(f"{match.group(2)}-{match.group(1)}",
                                                  freq="M").end_time.normalize()
            if len(found) >= 2 and not _has_isin(line):
                # A new financial year block starts here. The test cannot ask for twelve months, or
                # even six: the current financial year is only as long as the year is old, and
                # rejecting it would silently drop the most recent months, which are the ones that
                # matter most.
                months = found
                continue
            if not months:
                continue
            name, isin = line.iloc[0], line.iloc[1]
            if not (isinstance(isin, str) and ISIN.fullmatch(isin.strip())):
                continue
            for column, month in months.items():
                rows.append({"isin": isin.strip(), "name": str(name).strip(),
                             "month_end": month, "holding_nominal_mn": line.get(column)})
        out = pd.DataFrame(rows)
        if not out.empty:
            out["holding_nominal_mn"] = _numeric(out["holding_nominal_mn"])
            out = out.dropna(subset=["holding_nominal_mn"])

    if out.empty:
        return out
    out = out.dropna(subset=["month_end"])
    # A gilt appears in every block once it exists, with a zero once the state has sold out of it.
    out["account"] = "crnd_and_dmo"
    return (out.sort_values(["month_end", "isin"])
            .drop_duplicates(["isin", "month_end"], keep="last")
            .reset_index(drop=True))


def auctions(fetcher: Fetcher | None = None, refresh: bool = False) -> pd.DataFrame:
    """Every outright gilt auction, with the two numbers that describe how it went.

    Cover is bids over the amount on offer and the tail is the gap in yield between the average
    accepted price and the lowest accepted price. Cover says how much demand turned up; the tail says
    how thin it was at the margin. A well-covered auction with a wide tail is a poor auction, and the
    two disagree often enough that carrying both is the point.

    Nominal amounts are £ millions. PAOF is the post-auction option facility, where gilt-edged market
    makers may take more of the same gilt at the average accepted price, so it is demand that was
    revealed after the auction cleared rather than at it.
    """
    table = frame("auctions", fetcher=fetcher, refresh=refresh)
    out = pd.DataFrame()
    out["date"] = _dates(table[_pick(table, "auction_date")])
    out["name"] = table[_pick(table, "gilt_name")].astype(str).str.strip()
    for column, target in (("via_auction", "nominal_auction_mn"),
                           ("paof", "nominal_paof_mn"),
                           ("total_nominal", "total_nominal_mn"),
                           ("cash_raised", "cash_raised_mn"),
                           ("cover", "cover"),
                           ("accepted_price", "average_accepted_price"),
                           ("yield_at_aap", "yield_at_aap"),
                           ("tail", "tail_bp")):
        found = _pick(table, column)
        out[target] = _numeric(table[found]) if found else pd.NA
    out["coupon"] = out["name"].map(parse_coupon)
    out["linker"] = out["name"].str.contains("index-linked", case=False, na=False)
    return (out.dropna(subset=["date"]).sort_values("date", ascending=False)
            .reset_index(drop=True))


def redeemed(fetcher: Fetcher | None = None, refresh: bool = False) -> pd.DataFrame:
    """Gilts that have matured, and the day each one left the stock.

    Needed to run the stock backwards. Cumulative issuance tells you what was created; without a
    redemption date every gilt ever issued would still be counted as outstanding, and the stock in
    2010 would come out roughly twice its true size.
    """
    table = frame("redeemed", fetcher=fetcher, refresh=refresh)
    out = pd.DataFrame()
    out["redemption_date"] = _dates(table[_pick(table, "redemption_date")])
    out["name"] = table[_pick(table, "gilt_name")].astype(str).str.strip()
    amount = _pick(table, "nominal_amount") or _pick(table, "outstanding")
    out["nominal_at_redemption_mn"] = _numeric(table[amount]) if amount else pd.NA
    out["coupon"] = out["name"].map(parse_coupon)
    return (out.dropna(subset=["redemption_date"])
            .sort_values("redemption_date").reset_index(drop=True))


def issuance(fetcher: Fetcher | None = None, refresh: bool = False) -> pd.DataFrame:
    """Every gilt issuance operation the DMO has conducted, from the issuance history report."""
    table = frame("issuance_history", fetcher=fetcher, refresh=refresh)
    out = table.copy()
    for column in out.columns:
        if "date" in column:
            out[column] = _dates(out[column])
        elif any(word in column for word in
                 ("amount", "nominal", "proceeds", "cash", "price", "yield", "cover", "size")):
            out[column] = _numeric(out[column])
    return out


def generic(report_key: str, fetcher: Fetcher | None = None,
            refresh: bool = False) -> pd.DataFrame:
    """Types coerced, names normalised, nothing dropped. Used where the schema is not pinned."""
    table = frame(report_key, fetcher=fetcher, refresh=refresh)
    out = table.copy()
    for column in out.columns:
        present = out[column].notna().sum()
        lowered = str(column)
        if ("date" in lowered or lowered.endswith("_day")) and _dateish(out[column]):
            parsed = _dates(out[column])
            if parsed.notna().sum() >= max(3, int(0.6 * present)):
                out[column] = parsed
                continue
        values = _numeric(out[column])
        if values.notna().sum() >= max(3, int(0.6 * present)):
            out[column] = values
            continue
        # Anything still mixed is written as text. A column holding numbers and the odd stray label
        # cannot be typed, and letting it through as a mixed object breaks the parquet write, which
        # would cost the whole table rather than one column.
        out[column] = out[column].astype("string")
    return out


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------

# Report key -> (output stem, tidier). Where a tidier is `None` the generic one is used, which
# keeps every column the DMO publishes with types coerced but no renaming.
TABLES: dict[str, tuple[str, object]] = {
    "gilts_in_issue": ("gilts", gilts),
    "redeemed": ("redeemed", redeemed),
    "official_holdings": ("official_holdings", official),
    "crnd_funds": ("crnd_funds", None),
    "issuance_history": ("issuance", issuance),
    "auctions": ("auctions", auctions),
    "syndications": ("syndications", None),
    "tenders": ("tenders", None),
    "other_operations": ("other_operations", None),
    "repo_facility": ("repo_facility", None),
    "future_redemptions": ("future_redemptions", None),
    "bill_tenders": ("bill_tenders", None),
    "bills_stock": ("bills_stock", None),
    "rpi": ("rpi", None),
    "index_ratios": ("index_ratios", None),
}


# ---------------------------------------------------------------------------------------------
# the site's own JSON feed
# ---------------------------------------------------------------------------------------------

# The data site's front page fills three tables from a small JSON API that is not in the report
# catalogue. Two of the three are worth keeping.
#
# Upcoming auctions is forward supply: the gilt, the date and the amount on offer, before it is sold.
# Nothing in the report catalogue carries it, because the issuance calendar report (D5D) rejects every
# request the export endpoint will accept.
#
# The portfolio metrics are the DMO's own totals for the whole gilt portfolio. They are not new
# information but they are an independent statement of it, which is what makes them useful: the
# uplifted nominal figure is what every stock number here is audited against.
#
# Recent auctions is the third and is not kept. It carries cover and tail, which was valuable while
# the auction report looked unreachable, but it returns five rows whatever limit is asked for and the
# auction report gives all 1,235.
FEEDS = f"{BASE}/umbraco/surface/TabularData"


def feed(name: str, fetcher: Fetcher | None = None) -> object:
    """One JSON feed, cached on disk like any report so an offline run still has the last copy."""
    path = CACHE / f"feed_{_snake(name)}.json"
    if OFFLINE or (path.exists() and _age_hours(path) < 20):
        if not path.exists():
            raise Unavailable(f"{name}: not in the cache and the network is switched off")
        return json.loads(path.read_text(encoding="utf-8"))

    fetcher = fetcher or Fetcher()
    fetcher._wait()                                 # the same pacing as every other request
    fetcher.last_request = time.time()
    try:
        response = fetcher.session.get(f"{FEEDS}/{name}", timeout=TIMEOUT)
    except requests.RequestException as error:
        if path.exists():
            progress.write(f"    feed {name}: network error, keeping cache ({error})")
            return json.loads(path.read_text(encoding="utf-8"))
        raise Unavailable(f"{name}: {error}") from error
    if response.status_code != 200:
        if path.exists():
            progress.write(f"    feed {name}: HTTP {response.status_code}, keeping cache")
            return json.loads(path.read_text(encoding="utf-8"))
        raise Unavailable(f"{name}: feed returned {response.status_code}")
    text = response.text
    if _looks_blocked(text):
        if path.exists():
            progress.write(f"    feed {name}: bot protection, keeping cache")
            return json.loads(path.read_text(encoding="utf-8"))
        raise Blocked(f"{name}: bot protection served instead of JSON")
    try:
        payload = response.json()
    except json.JSONDecodeError as error:
        if path.exists():
            progress.write(f"    feed {name}: invalid JSON, keeping cache ({error})")
            return json.loads(path.read_text(encoding="utf-8"))
        raise Unavailable(f"{name}: not JSON ({error})") from error
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def upcoming(fetcher: Fetcher | None = None) -> pd.DataFrame:
    """Auctions that have been announced but not yet held: supply you can see coming."""
    records = feed("GetUpcomingAuctions?limit=100", fetcher=fetcher)
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    table = pd.DataFrame(records)
    out = pd.DataFrame({
        "date": _dates(table["AuctionDate"]),
        "name": table["GiltName"].astype(str).str.strip(),
        "redemption_date": _dates(table["MaturityDate"]),
        "amount_on_offer_mn": _numeric(table["AmountOnOffer"]),
    })
    out["coupon"] = out["name"].map(parse_coupon)
    out["linker"] = out["name"].str.contains("index-linked", case=False, na=False)
    out["seen_on"] = pd.Timestamp.today().normalize()
    return out.sort_values("date").reset_index(drop=True)


def portfolio(fetcher: Fetcher | None = None) -> pd.DataFrame:
    """The DMO's own headline totals for the gilt portfolio, as one row."""
    record = feed("GetSalesData", fetcher=fetcher)
    if not isinstance(record, dict) or not record:
        return pd.DataFrame()
    # The two amounts are not in the same unit, which the feed does not say anywhere: the portfolio
    # totals are in thousands and the sales figure is in pounds. The site's own script is the evidence,
    # since it divides the sales figure by a further thousand before formatting both the same way.
    # Taken at face value the portfolio reads as £2.9bn rather than £2,899bn.
    names = {
        "UpliftedNominalValueGiltPortfolio": ("stock_uplifted_mn", 1e-3),
        "UpliftedNominalValueGiltPortfolioConventional": ("stock_conventional_mn", 1e-3),
        "UpliftedNominalValueGiltPortfolioIl": ("stock_linker_mn", 1e-3),
        "AverageMaturity": ("average_maturity_years", 1.0),
        "AverageGiltPortfolioYieldConventional": ("average_yield_conventional", 1.0),
        "AverageGiltPortfolioYieldIl": ("average_yield_linker", 1.0),
        "TotalGiltSalesToDate": ("sales_this_year_mn", 1e-6),
    }
    row = {target: float(record[source]) * scale
           for source, (target, scale) in names.items() if record.get(source) is not None}
    row["as_of"] = pd.to_datetime(record.get("LastUpdated"), errors="coerce", utc=True)
    if pd.notna(row["as_of"]):
        row["as_of"] = row["as_of"].tz_localize(None).normalize()
    return pd.DataFrame([row])


SNAPSHOTS = HERE / "cache" / "dmo_snapshots"
# Reports that describe today rather than history, so the only way to get a run of them is to keep
# each one as it appears. The stock is also reconstructed from the issuance history in the framework;
# these files are the direct evidence to check that reconstruction against.
ARCHIVED = ("gilts", "upcoming", "portfolio")


def _archive(table: pd.DataFrame, stem: str) -> None:
    """Keep a dated copy of the snapshot reports so a history accretes from the first run."""
    if stem not in ARCHIVED:
        return
    dated = next((column for column in ("as_of", "seen_on") if column in table.columns), None)
    if dated is None:
        return
    stamp = pd.to_datetime(table[dated]).max()
    if pd.isna(stamp):
        return
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / f"{stem}_{stamp:%Y-%m-%d}.csv"
    if not path.exists():
        table.to_csv(path, index=False)


def snapshots(stem: str) -> pd.DataFrame:
    """Every dated copy of a snapshot report that has been collected so far."""
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    files = sorted(SNAPSHOTS.glob(f"{stem}_*.csv"))
    if not files:
        return pd.DataFrame()
    pieces = [pd.read_csv(path, parse_dates=["as_of"], low_memory=False) for path in files]
    return pd.concat(pieces, ignore_index=True)


def _write(table: pd.DataFrame, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT / f"dmo_{stem}.parquet", index=False)
    table.to_csv(OUTPUT / f"dmo_{stem}.csv.gz", index=False, compression="gzip")
    _archive(table, stem)


def build(refresh: bool = False, quiet: bool = False,
          fetch: bool = True) -> dict[str, pd.DataFrame]:
    """Refresh the cache, then turn every report into a table in output/.

    A report that cannot be read is reported and skipped rather than failing the run: the DMO
    occasionally reshapes an export, and one broken report should not cost you the other sixteen.
    """
    global OFFLINE
    OFFLINE = not fetch
    OUTPUT.mkdir(parents=True, exist_ok=True)       # the status file is written before any table
    if fetch:
        status = refresh_all(refresh=refresh, quiet=quiet)
        if not quiet:
            bad = status[status["status"].isin(["failed", "unavailable"])]
            print(f"  fetched {len(status) - len(bad)}/{len(status)} reports")
            for row in bad.itertuples():
                print(f"    {row.code} {row.status}: {row.label}")
        status.to_csv(OUTPUT / "dmo_fetch_status.csv", index=False)

    built: dict[str, pd.DataFrame] = {}
    rows = []
    with progress.bar(total=len(TABLES), label="DMO tables", unit="table",
                      enabled=not quiet) as step:
        for key, (stem, tidier) in TABLES.items():
            report = BY_KEY[key]
            try:
                table = tidier() if tidier is not None else generic(key)
                if table.empty:
                    raise ValueError("no rows")
                _write(table, stem)
                built[stem] = table
                rows.append({"table": stem, "code": report.code, "rows": len(table),
                             "columns": table.shape[1], "status": "ok"})
            except Exception as error:                   # noqa: BLE001 - keep going
                progress.write(f"    {report.code} -> {stem}: {error}")
                rows.append({"table": stem, "code": report.code, "rows": 0, "columns": 0,
                             "status": str(error)[:120]})
            step(stem)

    for stem, builder in (("upcoming", upcoming), ("portfolio", portfolio)):
        try:
            table = builder()
            if table.empty:
                raise ValueError("no rows")
            _write(table, stem)
            built[stem] = table
            rows.append({"table": stem, "code": "feed", "rows": len(table),
                         "columns": table.shape[1], "status": "ok"})
        except Exception as error:                       # noqa: BLE001 - keep going
            progress.write(f"    feed -> {stem}: {error}")
            rows.append({"table": stem, "code": "feed", "rows": 0, "columns": 0,
                         "status": str(error)[:120]})

    summary = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "dmo_coverage.csv", index=False)
    if not quiet:
        print(summary.to_string(index=False))
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="ignore cache freshness and re-pull every report")
    parser.add_argument("--no-fetch", action="store_true",
                        help="build tables from the cache without touching the network")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    build(refresh=arguments.refresh, quiet=arguments.quiet, fetch=not arguments.no_fetch)


if __name__ == "__main__":
    main()
