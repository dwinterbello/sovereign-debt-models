"""Italian issuance: MEF parsed workbooks + Banca d'Italia auction zips."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from issuance.http import FetchError, get_bytes, get_text
from issuance.paths import CACHE, RAW
from issuance.schema import coerce_ops
from issuance.tenor import attach_tenor

BDI_CURRENT = "https://www.bancaditalia.it/compiti/operazioni-mef/risultati-aste/Aste_anno_corrente.zip"
BDI_HIST = "https://www.bancaditalia.it/compiti/operazioni-mef/risultati-aste/storico_aste.zip"

MEF = "https://www.dt.mef.gov.it"
STATS_PAGES = (
    "/en/debito_pubblico/dati_statistici/",
    "/it/debito_pubblico/dati_statistici/",
    "/en/debito_pubblico/emissioni_titoli_di_stato_interni/",
)

SEC_TYPE = {
    "BTP": "Nominal",
    "BOT": "Bill",
    "CTZ": "Nominal",
    "CCT": "FRN",
    "CCTEU": "FRN",
    "BTPEI": "Linker",
    "BTP€I": "Linker",
    "BTP ITALIA": "Linker",
    "BTP GREEN": "Nominal",
    "BTP VALORE": "Nominal",
    "BTP FUTURA": "Nominal",
}


def _sec_type(sec_cd: object, desc: object = None) -> str:
    key = str(sec_cd or "").strip().upper().replace(" ", "")
    mapped = {
        # CTZ is a 24m zero-coupon note that MEF and Banca d'Italia count as medium/long-term.
        "BTP": "Nominal", "BOT": "Bill", "CTZ": "Nominal",
        "CCT": "FRN", "CCTEU": "FRN", "BTPEI": "Linker",
        "BTPI": "Linker", "BTP€I": "Linker",
        "BTPITALIA": "Linker", "BTPGREEN": "Nominal",
        "BTPVALORE": "Nominal", "BTPFUTURA": "Nominal",
    }.get(key)
    if mapped:
        return mapped
    text = f"{sec_cd} {desc}".lower()
    if "italia" in text or "btpei" in text or "€i" in text or "indexed" in text:
        return "Linker"
    if "bot" in text:
        return "Bill"
    if "cct" in text:
        return "FRN"
    return "Nominal"


def _download_stat_tables(force: bool = False) -> list[Path]:
    dest_dir = RAW / "mef" / "stats"
    dest_dir.mkdir(parents=True, exist_ok=True)
    found = list(dest_dir.glob("*.xls*"))
    if found and not force:
        return found
    for page in STATS_PAGES:
        try:
            html = get_text(urljoin(MEF, page), timeout=60)
        except FetchError:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            name = href.split("?")[0].split("/")[-1]
            if not name.lower().endswith((".xlsx", ".xls", ".csv")):
                continue
            dest = dest_dir / name
            try:
                get_bytes(urljoin(MEF, href), dest=dest, timeout=120)
                found.append(dest)
            except FetchError:
                continue
    return found


def _scale_amount(series: pd.Series) -> pd.Series:
    """MEF parsers sometimes leave euros or thousands instead of millions."""
    out = pd.to_numeric(series, errors="coerce")
    euros = out.abs() >= 1e8
    thousands = (out.abs() >= 1e5) & ~euros
    out = out.mask(euros, out / 1e6)
    out = out.mask(thousands, out / 1e3)
    return out


def _coalesce(df: pd.DataFrame, *names: str) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name in names:
        if name in df.columns:
            out = out.fillna(df[name])
    return out


_MEF_SYNDICATION = re.compile(
    r"syndi|syndacate|placement|distribution|details|btp[_\s-]*(?:italia|valore|futura|pi[uù])", re.I)
_MEF_EXCHANGE = re.compile(r"exchange", re.I)


def _mef_op_type(op_type: pd.Series, source_file: pd.Series) -> pd.Series:
    """The MEF auctions workbook also holds syndications, retail placements and exchange offers.

    Its own ``op_type`` tags only some syndications ("Syndicate"); the rest are named by the
    results PDF ("Syndicate placement", "Placement result", "BTP Valore ... placement").
    """
    name = source_file.fillna("").astype(str)
    raw = op_type.fillna("Auction").astype(str)
    out = raw.mask(raw.str.lower().str.startswith("syndi"), "Syndication")
    is_auction_pdf = name.str.contains("auction", case=False)
    out = out.mask(~is_auction_pdf & name.str.contains(_MEF_SYNDICATION), "Syndication")
    out = out.mask(~is_auction_pdf & name.str.contains(_MEF_EXCHANGE), "Exchange")
    return out


def _dropped_auction_lines(parsed: Path, kept: pd.DataFrame) -> pd.DataFrame:
    """Auction lines the MEF combine step dropped as fuzzy duplicates of a different ISIN.

    ``postprocess_combined_data`` clusters rows matching on two of (maturity, sec_cd, sec_desc);
    an ISIN-less row bridged distinct lines, so a second BOT reopening or BTP€i on the same day
    was marked ``DROPPED``. The parsing log keeps each dropped row's fields. A logged line is
    restored when its (ISIN, auction date) is not already in the combined workbook; only exact
    repeats (same ISIN, date and amount, e.g. EN and IT copies of one PDF) are collapsed.
    Distinct lines restored this way: 2015 BTP€i ~EUR 2.0bn, 2020 BOT 6.0bn, 2024 BOT 8.5bn,
    2025 BTP€i 1.0bn.
    """
    log_path = parsed / "mef_parsing_log.xlsx"
    if not log_path.exists():
        return pd.DataFrame()
    log = pd.read_excel(log_path, sheet_name="auctions")
    if "duplicate_status" not in log.columns:
        return pd.DataFrame()
    log = log[log["duplicate_status"].astype(str).str.startswith("DROPPED")
              & pd.to_numeric(log["amount_allotted"], errors="coerce").notna()].copy()
    log["isin"] = log["operation_id"].astype(str).str.extract(r"^(IT[0-9A-Z]{10})", expand=False)
    log = log[log["isin"].notna()]
    log["op_dt"] = pd.to_datetime(log["op_dt"], errors="coerce")
    log["_source_file"] = log["pdf_path"].astype(str).str.split(r"[\\/]").str[-1]
    # Exchange-offer PDFs are booked from exchanges_combined; their auction-folder copies are not lines.
    log = log[~log["_source_file"].str.contains(_MEF_EXCHANGE)]
    have = set(zip(kept["isin"].astype(str).str.strip().str.upper(),
                   pd.to_datetime(kept["op_dt"], errors="coerce")))
    new = [(i, d) not in have for i, d in zip(log["isin"], log["op_dt"])]
    return log[new].drop_duplicates(["isin", "op_dt", "amount_allotted"], keep="first")


def _from_auctions(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    dropped = _dropped_auction_lines(path.parent, raw)
    if not dropped.empty:
        print(f"  IT auctions: {len(dropped)} lines restored from the parsing log")
        raw = pd.concat([raw, dropped[[c for c in dropped.columns if c in raw.columns]]], ignore_index=True)
    src = raw["_source_file"] if "_source_file" in raw.columns else pd.Series("", index=raw.index)
    out = pd.DataFrame({
        "op_dt": pd.to_datetime(_coalesce(raw, "op_dt", "auction_dt"), errors="coerce"),
        "issue_dt": pd.to_datetime(_coalesce(raw, "issue_dt", "settlement_dt"), errors="coerce"),
        "maturity_dt": pd.to_datetime(raw.get("maturity_dt"), errors="coerce"),
        "isin": raw.get("isin"),
        "sec_desc_long": _coalesce(raw, "sec_desc", "sec_cd"),
        "amount_issued": _scale_amount(_coalesce(raw, "amount_allotted", "amount_issued")),
        "amount_bid": pd.to_numeric(raw.get("amount_bid"), errors="coerce"),
        "price": pd.to_numeric(raw.get("price"), errors="coerce"),
        "yield": pd.to_numeric(_coalesce(raw, "yield_gross", "yield_wavg", "yield"), errors="coerce"),
        "coupon": pd.to_numeric(raw.get("coupon"), errors="coerce"),
        "btc": pd.to_numeric(raw.get("btc"), errors="coerce"),
        "resid_mat_yrs": pd.to_numeric(raw.get("resid_mat_yrs"), errors="coerce"),
        "op_type": _mef_op_type(raw.get("op_type", pd.Series("Auction", index=raw.index)), src),
        "sec_type": [_sec_type(c, d) for c, d in zip(raw.get("sec_cd", ""), raw.get("sec_desc", ""))],
        "source": f"MEF {path.name}",
    })
    return out


def _from_other(path: Path, default_op: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    allotted = _scale_amount(_coalesce(raw, "amount_issued", "amount_allotted", "amount_reissued", "amount_reopening"))
    bought = _scale_amount(raw.get("amount_boughtback")) if "amount_boughtback" in raw.columns else pd.Series(pd.NA, index=raw.index)
    offered = _scale_amount(raw.get("amount_offered")) if "amount_offered" in raw.columns else pd.Series(pd.NA, index=raw.index)
    if default_op == "Buyback":
        amount = -bought.abs()
        amount = amount.fillna(-allotted.abs())
        amount = amount.fillna(-offered.abs())
        op_type = "Buyback"
    else:
        amount = allotted
        # Retired legs of an exchange carry bought-back notional and no allotment.
        retire = amount.isna() & bought.notna()
        amount = amount.where(~retire, -bought.abs())
        op_type = pd.Series(default_op, index=raw.index)
        op_type = op_type.where(~retire, "Buyback")
    out = pd.DataFrame({
        "op_dt": pd.to_datetime(_coalesce(raw, "op_dt", "transaction_dt", "auction_dt"), errors="coerce"),
        "issue_dt": pd.to_datetime(_coalesce(raw, "issue_dt", "settlement_dt"), errors="coerce"),
        "maturity_dt": pd.to_datetime(raw.get("maturity_dt"), errors="coerce"),
        "isin": raw.get("isin"),
        "sec_desc_long": _coalesce(raw, "sec_desc", "sec_cd"),
        "amount_issued": amount,
        "price": pd.to_numeric(raw.get("price"), errors="coerce"),
        "yield": pd.to_numeric(_coalesce(raw, "yield", "yield_gross"), errors="coerce"),
        "coupon": pd.to_numeric(raw.get("coupon"), errors="coerce"),
        "resid_mat_yrs": pd.to_numeric(raw.get("resid_mat_yrs"), errors="coerce"),
        "op_type": op_type,
        "sec_type": [_sec_type(c, d) for c, d in zip(raw.get("sec_cd", ""), raw.get("sec_desc", ""))],
        "source": f"MEF {path.name}",
    })
    return out


def _bdi_dir() -> Path:
    dest = RAW / "mef" / "bdi"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _download_bdi(force: bool = False) -> list[Path]:
    dest_dir = _bdi_dir()
    out: list[Path] = []
    for url, name in ((BDI_CURRENT, "Aste_anno_corrente.zip"), (BDI_HIST, "storico_aste.zip")):
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 1000 and (not force or url == BDI_HIST):
            out.append(dest)
            continue
        try:
            get_bytes(url, dest=dest, timeout=120)
            out.append(dest)
        except FetchError as exc:
            print(f"  IT BdI skip {name}: {exc}")
            if dest.exists() and dest.stat().st_size > 1000:
                out.append(dest)
    return out


def _unzip_xls(path: Path) -> list[Path]:
    dest_dir = path.with_suffix("")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".zip":
        return [path]
    # The current-year file is renamed at every refresh ("al 30.9.2026"); an old copy left
    # beside it would be parsed too and double the year's auctions.
    for old in dest_dir.rglob("*"):
        if old.is_file():
            old.unlink()
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest_dir)
    return [p for p in dest_dir.rglob("*") if p.suffix.lower() in {".xls", ".xlsx"}]


def _parse_it_date(series: pd.Series) -> pd.Series:
    raw = series
    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(raw, errors="coerce")
    text = raw.astype(str)
    slash = text.str.contains(r"\d{1,2}-\d{1,2}-\d{4}", regex=True, na=False)
    dt = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if slash.any():
        dt.loc[slash] = pd.to_datetime(raw.loc[slash], errors="coerce", dayfirst=True)
    rest = ~slash
    if rest.any():
        dt.loc[rest] = pd.to_datetime(raw.loc[rest], errors="coerce")
    return dt


def _it_coupon(desc: object) -> float:
    text = str(desc or "")
    match = re.search(r"(\d+[,.]?\d*)\s*%", text)
    if not match:
        return float("nan")
    return float(match.group(1).replace(",", "."))


def parse_bdi_workbook(path: Path) -> pd.DataFrame:
    """BdI current-year .xls and storico .xlsx share the same 18-col layout."""
    raw = pd.read_excel(path, header=None)
    isin_col = raw.iloc[:, 2].astype(str)
    body = raw[isin_col.str.startswith("IT") & (isin_col.str.len() >= 12)].copy()
    if body.empty:
        return pd.DataFrame()
    assigned = pd.to_numeric(body.iloc[:, 13], errors="coerce")
    bid = pd.to_numeric(body.iloc[:, 12], errors="coerce")
    tipo = body.iloc[:, 8].astype(str)
    desc = body.iloc[:, 5]
    kind = body.iloc[:, 4].astype(str).str.strip().str.upper().str[:1]
    out = pd.DataFrame({
        "op_dt": _parse_it_date(body.iloc[:, 0]),
        "issue_dt": _parse_it_date(body.iloc[:, 1]),
        "maturity_dt": _parse_it_date(body.iloc[:, 6]),
        "isin": body.iloc[:, 2].astype(str).str.strip().str.upper(),
        "sec_desc_long": desc,
        "amount_issued": assigned,
        "amount_bid": bid,
        "price": pd.to_numeric(body.iloc[:, 14], errors="coerce"),
        "yield": pd.to_numeric(body.iloc[:, 16], errors="coerce").where(
            pd.to_numeric(body.iloc[:, 16], errors="coerce").fillna(0) != 0,
            pd.to_numeric(body.iloc[:, 15], errors="coerce"),
        ),
        "coupon": desc.map(_it_coupon),
        "btc": bid / assigned.replace(0, pd.NA),
        "op_type": "Auction",
        "sec_type": [_sec_type(t, d) for t, d in zip(tipo, desc)],
        "tranche_kind": kind.where(kind.isin(["O", "S"]), "O"),
        "source": f"BdI {path.name}",
    })
    out.loc[out["tranche_kind"].eq("S"), "source"] = f"BdI {path.name} supplementare"
    return out


def _from_bdi(force: bool = False) -> pd.DataFrame:
    zips = _download_bdi(force=force)
    parts = []
    for zpath in zips:
        for xls in _unzip_xls(zpath):
            try:
                part = parse_bdi_workbook(xls)
            except Exception as exc:  # noqa: BLE001
                print(f"  IT BdI parse skip {xls.name}: {exc}")
                continue
            if not part.empty:
                parts.append(part)
                print(f"  IT BdI {xls.name}: {len(part)} rows")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["isin", "op_dt", "amount_issued", "tranche_kind"], keep="first")


def _from_parsed() -> pd.DataFrame:
    parsed = RAW / "mef" / "parsed_data"
    if not parsed.exists():
        return pd.DataFrame()
    parts = []
    auctions = parsed / "auctions_combined.xlsx"
    if auctions.exists():
        parts.append(_from_auctions(auctions))
        print(f"  IT auctions: {len(parts[-1])} rows")
    for fname, op in (
        ("buybacks_combined.xlsx", "Buyback"),
        ("exchanges_combined.xlsx", "Exchange"),
        ("taps_combined.xlsx", "TAP"),
        ("globals_combined.xlsx", "Syndication"),
        ("mtn_combined.xlsx", "Syndication"),
    ):
        path = parsed / fname
        if path.exists():
            parts.append(_from_other(path, op))
            print(f"  IT {op}: {len(parts[-1])} rows from {fname}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _merge_it(mef: pd.DataFrame, bdi: pd.DataFrame) -> pd.DataFrame:
    """BdI is the live auction tape. Keep MEF TAP/buyback/syndication; add BdI after MEF auction vintage.

    MEF auction rows are the ordinary tranche only (equal to BdI "O" on every matched row
    2010-25); the specialists' supplementary reopening (BdI "S", ~EUR 30-45bn a year) is
    taken from BdI over the MEF window too. MEF's ``specialists_amount_allotted`` copies the
    offered amount and carries unit errors, so it is not used.
    """
    parts = []
    if mef is not None and not mef.empty:
        parts.append(mef)
    if bdi is None or bdi.empty:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if mef is None or mef.empty:
        return bdi
    auction = mef["op_type"].astype(str).isin(["Auction", "Reopening"])
    mef_max = mef.loc[auction, "op_dt"].max() if auction.any() else mef["op_dt"].max()
    after = bdi["op_dt"] > mef_max if pd.notna(mef_max) else pd.Series(True, index=bdi.index)
    supp = bdi["tranche_kind"].eq("S") if "tranche_kind" in bdi.columns else pd.Series(False, index=bdi.index)
    extra = bdi[after | supp]
    if extra.empty:
        return mef
    print(f"  IT BdI extra after {pd.Timestamp(mef_max).date()}: {int(after.sum())} rows;"
          f" supplementary within MEF window: {int((supp & ~after).sum())} rows")
    return pd.concat([mef, extra], ignore_index=True)


def _fill_maturity(ops: pd.DataFrame) -> pd.DataFrame:
    """Buybacks often omit the redemption date. The same ISIN's auction rows have it."""
    if ops.empty or "isin" not in ops.columns:
        return ops
    out = ops.copy()
    out["isin"] = out["isin"].astype("string").str.strip().str.upper()
    known = out.dropna(subset=["maturity_dt", "isin"]).drop_duplicates("isin")
    mapped = known.set_index("isin")["maturity_dt"]
    miss = out["maturity_dt"].isna() & out["isin"].notna()
    out.loc[miss, "maturity_dt"] = out.loc[miss, "isin"].map(mapped)
    # BOT lines whose only label is "3m"/"6m" and which never reappear.
    still = out["maturity_dt"].isna() & out["issue_dt"].notna()
    desc = out["sec_desc_long"].astype(str).str.strip().str.lower()
    for label, months in (("3m", 3), ("6m", 6), ("12m", 12)):
        hit = still & desc.eq(label)
        if hit.any():
            out.loc[hit, "maturity_dt"] = out.loc[hit, "issue_dt"] + pd.DateOffset(months=months)
    return out


SUPPLEMENT = Path(__file__).resolve().parent / "data" / "it_supplement.json"

# BdI posts each offering a few days before the auction: ISIN, size range, auction date.
BDI_ANNUNCI = "https://www.bancaditalia.it/compiti/operazioni-mef/annunci-operazioni/"
BDI_ANNUNCI_INDEX = BDI_ANNUNCI + "index.html"
OUT_ANN = CACHE / "it_bdi_announcements.parquet"
RAW_ANN = RAW / "mef" / "bdi" / "annunci"

_ANN_SEC = re.compile(
    r"(BTP€?I|BTPI|BTP|BOT|CCTEU|CCT|CTZ)\b[^\n]*\n\s*(IT\d{10})\s*/\s*\d+\s*\n\s*"
    r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})",
    re.I,
)
_ANN_AMOUNT = re.compile(
    r"Importo offerto\s*\(mln\.\s*€\)\s*((?:\d[\d.]*(?:/\d[\d.]*)?(?:\s+|$))+)",
    re.I,
)
_ANN_AUCTION = re.compile(r"Data asta\s*((?:\d{2}/\d{2}/\d{4}\s*)+)", re.I)
_ANN_PDF = re.compile(r"ANNUNCIO[-_]EMISSIONI?E?[^\"'>\s]*\.pdf(?:\.pdf)?", re.I)


def _it_ann_date(text: str) -> pd.Timestamp:
    d, m, y = text.strip().split("/")
    return pd.Timestamp(int(y), int(m), int(d))


def _it_ann_amount(tok: str) -> tuple[float, float, float]:
    """Italian thousands (1.500 = 1500). Range → (min, max, mid); single → all equal."""
    parts = [p.replace(".", "") for p in tok.split("/") if p]
    vals = [float(p) for p in parts]
    lo, hi = min(vals), max(vals)
    return lo, hi, (lo + hi) / 2.0


def _it_ann_sec_type(kind: str) -> str:
    k = kind.upper().replace("€", "E").replace(" ", "")
    if k in {"BOT"}:
        return "Bill"
    if k in {"BTPI", "BTPEI"}:
        return "Linker"
    if k in {"CCT", "CCTEU"}:
        return "FRN"
    return "Nominal"


def parse_bdi_announcement_text(text: str, *, source: str = "") -> pd.DataFrame:
    """Parse a BdI ANNUNCIO-EMISSIONE PDF into one row per ISIN (auction day + size range)."""
    text = (text or "").replace("\xa0", " ").replace("\r", "")
    secs = _ANN_SEC.findall(text)
    amts: list[str] = []
    for m in _ANN_AMOUNT.finditer(text):
        amts.extend(re.findall(r"\d[\d.]*(?:/\d[\d.]*)?", m.group(1)))
    dates: list[str] = []
    for m in _ANN_AUCTION.finditer(text):
        dates.extend(re.findall(r"\d{2}/\d{2}/\d{4}", m.group(1)))
    n = min(len(secs), len(amts), len(dates))
    rows = []
    for i in range(n):
        kind, isin, issue, mat = secs[i]
        lo, hi, mid = _it_ann_amount(amts[i])
        rows.append({
            "op_dt": _it_ann_date(dates[i]),
            "isin": isin.upper(),
            "issue_dt": _it_ann_date(issue),
            "maturity_dt": _it_ann_date(mat),
            "sec_type": _it_ann_sec_type(kind),
            "amount_min_mn": lo,
            "amount_max_mn": hi,
            "amount_mn": mid,
            "source": source or "BdI announcement",
        })
    return pd.DataFrame(rows)


def _bdi_annuncio_urls(html: str) -> list[str]:
    """Emission PDF hrefs from the BdI index (buybacks skipped). Keep the path as published —
    a few files are served as ``….pdf.pdf``. Index markup sometimes breaks the href across lines."""
    found = []
    flat = (html or "").replace("\n", "")
    for m in re.finditer(r'href="([^"]*ANNUNCIO[-_]EMISSIONI?E?[^"]+\.pdf(?:\.pdf)?)"', flat, re.I):
        found.append(urljoin(BDI_ANNUNCI, m.group(1).strip()))
    if not found:
        for name in _ANN_PDF.findall(html or ""):
            year = re.search(r"(20\d{2})", name)
            y = year.group(1) if year else str(pd.Timestamp.today().year)
            found.append(urljoin(BDI_ANNUNCI, f"documenti/{y}/{name}"))
    return list(dict.fromkeys(found))


def fetch_bdi_announcements(*, force: bool = False, limit: int = 20) -> pd.DataFrame:
    """Download recent BdI offering announcements → ``it_bdi_announcements.parquet``.

    Index page lists the latest emission PDFs; each has ISIN, offered range and auction date.
    Cached PDFs are always re-parsed. Network failure keeps the prior parquet.
    """
    from pypdf import PdfReader

    RAW_ANN.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    try:
        html = get_text(BDI_ANNUNCI_INDEX, timeout=60)
        urls = _bdi_annuncio_urls(html)[:limit]
    except FetchError as exc:
        print(f"  IT BdI annunci index: {exc}")
    if force or not urls:
        # Also re-parse anything already on disk when the index is unreachable.
        pass
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dest = RAW_ANN / name
        if dest.exists() and not force and dest.stat().st_size > 1000:
            continue
        try:
            get_bytes(url, dest=dest, timeout=60, min_bytes=1000)
        except FetchError as exc:
            print(f"  IT BdI annuncio skip {name}: {exc}")
    parts = []
    for path in sorted(RAW_ANN.glob("ANNUNCIO-EMISSIONE*.pdf")):
        try:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        except Exception as exc:  # noqa: BLE001
            print(f"  IT BdI annuncio parse fail {path.name}: {exc}")
            continue
        block = parse_bdi_announcement_text(text, source=f"BdI {path.name}")
        if not block.empty:
            parts.append(block)
    if not parts:
        if OUT_ANN.exists():
            return pd.read_parquet(OUT_ANN)
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["op_dt", "isin"], keep="last").sort_values(["op_dt", "isin"])
    out.to_parquet(OUT_ANN, index=False)
    print(f"  IT BdI announcements: {len(out)} lines through {out['op_dt'].max().date()}")
    return out


def supplement_it(ops: pd.DataFrame) -> pd.DataFrame:
    """Syndications and retail BTPs (Valore, Più, Italia) from MEF communiqués.

    The BdI auction file carries neither, and the MEF workbook books the 2023 BTP Valore
    with no ISIN and a wrong maturity. Idempotent: a row already on the tape is skipped.
    """
    if not SUPPLEMENT.exists():
        return ops
    sup = pd.read_json(SUPPLEMENT, dtype={"isin": str})
    for col in ("op_dt", "issue_dt", "maturity_dt"):
        sup[col] = pd.to_datetime(sup[col])
    # Rows default to syndications/retail; buybacks (negative amounts) carry op_type "Buyback".
    if "op_type" in sup.columns:
        sup["op_type"] = sup["op_type"].fillna("Syndication")
    else:
        sup["op_type"] = "Syndication"
    if ops is None or ops.empty:
        return coerce_ops(attach_tenor(sup), "IT")
    out = ops.copy()
    is_it = out["country"].astype(str).eq("IT") if "country" in out.columns else pd.Series(True, index=out.index)
    amt = pd.to_numeric(out["amount_issued"], errors="coerce")
    keep = []
    for row in sup.itertuples(index=False):
        near = is_it & ((out["op_dt"] - row.op_dt).abs() <= pd.Timedelta(days=7)) & ((amt - row.amount_issued).abs() < 1.0)
        if (near & out["isin"].astype("string").eq(row.isin).fillna(False)).any():
            continue
        out = out[~(near & out["isin"].isna())]
        amt = amt.loc[out.index]
        is_it = is_it.loc[out.index]
        keep.append(row._asdict())
    if not keep:
        return out
    add = coerce_ops(attach_tenor(pd.DataFrame(keep)), "IT")
    return pd.concat([out, add[[c for c in out.columns if c in add.columns]]], ignore_index=True)


def fetch_it(force: bool = False) -> pd.DataFrame:
    mef = _from_parsed()
    bdi = _from_bdi(force=force)
    ops = _merge_it(mef, bdi)
    if ops is None or ops.empty:
        files = _download_stat_tables(force=force)
        if not files:
            raise FetchError("IT issuance: no MEF workbooks and no BdI auction zips")
        raise FetchError("IT issuance: parsed MEF workbooks empty")
    ops = _fill_maturity(ops)
    ops = supplement_it(coerce_ops(attach_tenor(ops), "IT"))
    try:
        ops = add_bdi_buybacks(ops, fetch_bdi_buybacks(force=force))
    except Exception as exc:  # noqa: BLE001 — buybacks net the supply; tape still ships
        print(f"  IT buybacks failed: {exc}")
    ops = ops[ops["op_dt"].notna() & (ops["amount_issued"].fillna(0) != 0)]
    ops.to_parquet(CACHE / "ops_it.parquet", index=False)
    try:
        fetch_bdi_announcements(force=force)
    except Exception as exc:  # noqa: BLE001 — announcements feed the TBA; tape still ships
        print(f"  IT BdI announcements failed: {exc}")
    return ops


# ── Treasury buybacks (riacquisti in Banca d'Italia) ─────────────────────────
# The MEF buybacks workbook (buybacks_combined.xlsx) stopped at Nov-2025, so 2026 buybacks never
# reached the tape. Live path: the BdI announcements index lists each "Riacquisto" notice (ISINs,
# auction and settlement dates); the BdI press-release index links the results PDF for that day
# ("Importo acquistato", "Prezzo medio ponderato" per ISIN). Cached in it_bdi_buybacks.parquet so
# older operations survive the indexes rolling over.
BDI_COMUNICATI = "https://www.bancaditalia.it/media/comunicati/"
OUT_BUYBACKS = CACHE / "it_bdi_buybacks.parquet"
RAW_BB = RAW / "mef" / "bdi" / "riacquisti"
_MESI = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
         "settembre", "ottobre", "novembre", "dicembre")
_BB_SEC = re.compile(
    r"(BTP€?I|BTPI|BTP|CCTEU|CCT|CTZ)\s*([\d,]*)\s*%?\s*\n\s*(IT\d{10})\s*/\s*\d+\s*\n\s*"
    r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})", re.I)
_BB_SETTLE = re.compile(r"Data regolamento\s*((?:\d{2}/\d{2}/\d{4}\s*)+)", re.I)


def _it_num(tok: str) -> float:
    """Italian number: 1.754.000.000 → 1754000000; 97,101 → 97.101."""
    return float(tok.replace(".", "").replace(",", "."))


def parse_bdi_buyback_announcement(text: str, *, source: str = "") -> pd.DataFrame:
    """ANNUNCIO-RIACQUISTO PDF → one row per ISIN offered for repurchase."""
    text = (text or "").replace("\xa0", " ").replace("\r", "")
    secs = _BB_SEC.findall(text)
    dates = []
    for m in _ANN_AUCTION.finditer(text):
        dates.extend(re.findall(r"\d{2}/\d{2}/\d{4}", m.group(1)))
    settles = []
    for m in _BB_SETTLE.finditer(text):
        settles.extend(re.findall(r"\d{2}/\d{2}/\d{4}", m.group(1)))
    rows = []
    for i, (kind, cpn, isin, issue, mat) in enumerate(secs):
        if i >= len(dates):
            break
        rows.append({
            "op_dt": _it_ann_date(dates[i]),
            "issue_dt": _it_ann_date(settles[i]) if i < len(settles) else pd.NaT,
            "maturity_dt": _it_ann_date(mat),
            "isin": isin.upper(),
            "coupon": _it_num(cpn) if cpn else float("nan"),
            "sec_type": _it_ann_sec_type(kind),
            "source": source or "BdI riacquisto",
        })
    return pd.DataFrame(rows)


def parse_bdi_buyback_results(text: str, isins: list[str]) -> dict[str, tuple[float, float]]:
    """BdI results release → {isin: (amount bought EUR mn, weighted average price)}.

    Each ISIN block runs from "ISIN <code>" to the next "ISIN IT"; issuance releases carry
    "Importo assegnato" instead, so a block without "Importo acquistato" is not a buyback."""
    text = (text or "").replace("\xa0", " ")
    out = {}
    for isin in isins:
        for m in re.finditer(rf"ISIN\s+{isin}", text):
            nxt = re.search(r"ISIN\s+IT\d{10}", text[m.end():])
            block = text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]
            amt = re.search(r"Importo acquistato\s+([\d.]+)", block)
            if not amt:
                continue
            px = re.search(r"Prezzo medio ponderato\s+([\d.,]+)", block)
            out[isin] = (_it_num(amt.group(1)) / 1e6, _it_num(px.group(1)) if px else float("nan"))
            break
    return out


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def fetch_bdi_buybacks(*, force: bool = False, pages: int = 4) -> pd.DataFrame:
    """Riacquisto notices → results releases → buyback ops (negative amounts)."""
    RAW_BB.mkdir(parents=True, exist_ok=True)
    prior = pd.read_parquet(OUT_BUYBACKS) if OUT_BUYBACKS.exists() else pd.DataFrame()
    # 1. Announcements (the index holds the latest ~10 notices per page).
    for k in range(1, pages + 1):
        try:
            html = get_text(BDI_ANNUNCI_INDEX + (f"?page={k}" if k > 1 else ""), timeout=60)
        except FetchError as exc:
            print(f"  IT buybacks: annunci page {k}: {exc}")
            break
        flat = html.replace("\n", "")
        for href in re.findall(r'href="([^"]*ANNUNCIO[-_]RIACQUISTO[^"]+\.pdf(?:\.pdf)?)"', flat, re.I):
            dest = RAW_BB / href.rsplit("/", 1)[-1]
            if dest.exists() and dest.stat().st_size > 1000 and not force:
                continue
            try:
                get_bytes(urljoin(BDI_ANNUNCI, href), dest=dest, timeout=60, min_bytes=1000)
            except FetchError as exc:
                print(f"  IT buybacks: {dest.name}: {exc}")
    ann = []
    for path in sorted(RAW_BB.glob("ANNUNCIO*RIACQUISTO*.pdf")):
        try:
            block = parse_bdi_buyback_announcement(_pdf_text(path), source=f"BdI {path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  IT buybacks: parse {path.name}: {exc}")
            continue
        if not block.empty:
            ann.append(block)
    if not ann:
        return prior
    ann = pd.concat(ann, ignore_index=True).drop_duplicates(subset=["op_dt", "isin"], keep="last")
    done = set(zip(pd.to_datetime(prior["op_dt"]), prior["isin"])) if not prior.empty else set()
    todo = ann[[(pd.Timestamp(d), i) not in done for d, i in zip(ann["op_dt"], ann["isin"])]]
    if todo.empty:
        return prior
    # 2. Results: press-release links whose title names the auction day.
    links: list[tuple[str, str]] = []
    for k in range(1, pages + 3):
        try:
            html = get_text(BDI_COMUNICATI + "index.html" + (f"?page={k}" if k > 1 else ""), timeout=60)
        except FetchError as exc:
            print(f"  IT buybacks: comunicati page {k}: {exc}")
            break
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            if a["href"].lower().endswith(".pdf"):
                links.append((a.get_text(" ", strip=True), urljoin(BDI_COMUNICATI, a["href"])))
    rows = []
    for day, g in todo.groupby("op_dt"):
        day = pd.Timestamp(day)
        tag = re.compile(rf"(?<!\d){day.day}\s+{_MESI[day.month - 1]}\s+{day.year}")
        cands = list(dict.fromkeys(u for t, u in links if tag.search(t.lower())))
        got: dict[str, tuple[float, float]] = {}
        for url in cands:
            dest = RAW_BB / url.rsplit("/", 1)[-1]
            try:
                if force or not dest.exists():
                    get_bytes(url, dest=dest, timeout=60, min_bytes=1000)
                got.update(parse_bdi_buyback_results(_pdf_text(dest), list(g["isin"])))
            except Exception as exc:  # noqa: BLE001
                print(f"  IT buybacks: results {dest.name}: {exc}")
        if not got:
            print(f"  WARNING IT buyback {day:%Y-%m-%d} announced ({len(g)} lines) but no BdI results found")
            continue
        for r in g.itertuples(index=False):
            if r.isin in got:
                amt, px = got[r.isin]
                rows.append({**r._asdict(), "amount_issued": -amt, "price": px,
                             "sec_desc_long": f"buyback {r.isin}", "op_type": "Buyback",
                             "source": f"BdI aste di acquisto {day:%d/%m/%Y}"})
    if not rows:
        return prior
    new = pd.DataFrame(rows)
    out = pd.concat([prior, new], ignore_index=True) if not prior.empty else new
    out["op_dt"] = pd.to_datetime(out["op_dt"])
    out = out.drop_duplicates(subset=["op_dt", "isin"], keep="last").sort_values(["op_dt", "isin"])
    out.to_parquet(OUT_BUYBACKS, index=False)
    print(f"  IT buybacks: {len(new)} lines added, {len(out)} on file through {out['op_dt'].max():%Y-%m-%d}")
    return out


def add_bdi_buybacks(ops: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    """Append scraped buybacks the tape (or it_supplement.json) does not already hold."""
    if bb is None or bb.empty:
        return ops
    have = ops[ops["op_type"].astype(str).eq("Buyback")] if "op_type" in ops.columns else ops.iloc[0:0]
    keys = {(str(i), pd.Timestamp(d).normalize()) for i, d in zip(have["isin"], have["op_dt"])}
    add = bb[[(str(i), pd.Timestamp(d).normalize()) not in keys for i, d in zip(bb["isin"], bb["op_dt"])]]
    if add.empty:
        return ops
    add = coerce_ops(attach_tenor(add.copy()), "IT")
    return pd.concat([ops, add[[c for c in ops.columns if c in add.columns]]], ignore_index=True)
