"""Spanish Tesoro auction history and outstanding by ISIN.

    PYTHONPATH=. .venv/bin/python -m issuance.fetch_es [--force]

Sources (tesoro.es TLS chain needs verify=False):
  outstanding  Valores en circulación HTML — Letras + Bonos/Obligaciones by ISIN
  history      Wayback snapshots of the same page — matured ISINs (gone from live book)
  carv         BdE carv.zip (1988–2018) — Estado instrument master incl. matured lines
  auctions     Monthly-statistics XLSX 11 / historico_subastas — nominal issued by label
  results      Resultado últimas / anteriores HTML — auction date + maturity (+ coupon)
  bde          Banco de España BON*.pdf announcements — ISIN per bond line (settle date;
               live ficheros keep ~3y; older notices 404)
  cnmv         ANCV ISIN register for the Estado (principals + linkers; strips dropped)
  bulletins    Cached under data/raw/es/tesoro/

ISINs are not on the auction bulletin. Attachment:
  Letras   ISO check-digit formula ``ES0L0`` + YYMMDD
  Bonds    BdE PDF → results maturity → live/Wayback/CNMV book (Bono vs Obligación
           class + tenor + soft coupon; historico "tipo nominal" is often wrong)
  Brokers / Bloomberg TES05 / Reuters TESORES publish results, not a free ISIN tape.
"""

from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from issuance.http import get_bytes
from issuance.paths import CACHE, RAW
from issuance.schema import coerce_ops
from issuance.tenor import attach_tenor

TESORO = "https://www.tesoro.es"
OUTSTANDING = f"{TESORO}/en/deuda-publica/valores-del-tesoro/valores-en-circulacion"
RESULTS_LATEST = {
    "letras": f"{TESORO}/deuda-publica/subastas/resultado-ultimas-subastas/letras-del-tesoro",
    "bonos": f"{TESORO}/deuda-publica/subastas/resultado-ultimas-subastas/bonos-del-estado",
    "obligaciones": f"{TESORO}/deuda-publica/subastas/resultado-ultimas-subastas/obligaciones-del-estado",
    "linkers": f"{TESORO}/deuda-publica/subastas/resultado-ultimas-subastas/bonos-y-obligaciones-indexados",
}
RESULTS_HIST = f"{TESORO}/deuda-publica/subastas/resultados-subastas-anteriores"
# BdE primary-market notices: ISIN on each Bonos/Obligaciones line. Directory listing 403;
# filenames are BON{DDMMYYYY}.pdf / BON_OBL_{DDMMYYYY}.pdf on settle (anotación) date.
BDE_FICHEROS = (
    "https://www.bde.es/f/webbe/INF/MenuHorizontal/entidades-profesionales/"
    "operativa_gestiones/subastas/ficheros/"
)
BDE_MERCADEUDA = (
    "https://www.bde.es/f/webbe/INF/MenuHorizontal/entidades-profesionales/"
    "mercadeuda/subastas/"
)
BDE_UPCOMING = (
    "https://www.bde.es/wbe/es/punto-informacion/contenidos/"
    "mercado-primario-de-deuda/proximas-subasta/"
)
_ES_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}
BULLETINS = {
    "01_outstanding_nominal.xlsx": f"{TESORO}/documents/d/tesoro/01-xlsx",
    "04_letras.xlsx": f"{TESORO}/documents/d/tesoro/04-xlsx",
    "05_bonos.xlsx": f"{TESORO}/documents/d/tesoro/05-xlsx",
    "09_funding.xlsx": f"{TESORO}/documents/d/tesoro/09-xlsx",
    "11_auctions.xlsx": f"{TESORO}/documents/d/tesoro/11-xlsx",
    "13_net_financing.xlsx": f"{TESORO}/documents/d/tesoro/13-xlsx",
    "14_maturity_profile.xlsx": f"{TESORO}/documents/d/tesoro/14-xlsx",
    "historico_subastas.xlsx": f"{TESORO}/documents/d/tesoro/historico_subastas-xlsx",
}
RAW_ES = RAW / "es" / "tesoro"
RAW_BDE = RAW / "es" / "bde_subastas"
RAW_CNMV = RAW / "es" / "cnmv"
OUT_OPS = CACHE / "ops_es.parquet"
OUT_STOCK = CACHE / "es_outstanding.parquet"
OUT_STOCK_HIST = CACHE / "es_outstanding_history.parquet"
OUT_CARV = CACHE / "es_bde_carv.parquet"
OUT_RESULTS = CACHE / "es_auction_results.parquet"
OUT_BDE = CACHE / "es_bde_announcements.parquet"
OUT_CNMV = CACHE / "es_cnmv_isins.parquet"
OUT_CAL = CACHE / "es_tesoro_calendar.parquet"
TESORO_CAL_URL = f"{TESORO}/documents/d/tesoro/calendario-tesoro-publico-2026-ingles-12"
RAW_CAL = RAW_ES / "calendario_tesoro_publico_2026_en.pdf"
BDE_CARV_ZIP = (
    "https://repositorio.bde.es/bitstream/123456789/16988/1/carv.zip"
)
RAW_CARV = RAW / "es" / "bde_carv"
CNMV_ISIN = "https://www.cnmv.es/portal/ancv/isin.aspx?lang=es&nif=S-2826011E"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

_ISIN = re.compile(r"\b(ES[0-9A-Z]{10})\b")
# Bei/Oei before B/O — else "Bei-5a" matches kind=B and falls through to resid=10.
_AUCTION_LAB = re.compile(
    r"^(?P<kind>L|Bei|Oei|B|O)-?(?P<n>\d+(?:[.,]\d+)?)(?P<u>[amy])?$", re.I)


def _get(url: str, *, min_bytes: int = 500) -> bytes:
    return get_bytes(url, timeout=120, impersonate=True, verify=False, min_bytes=min_bytes)


def download_bulletins(force: bool = False) -> list[Path]:
    RAW_ES.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in BULLETINS.items():
        dest = RAW_ES / name
        if dest.exists() and dest.stat().st_size > 1000 and not force:
            out.append(dest)
            continue
        dest.write_bytes(_get(url, min_bytes=2000))
        out.append(dest)
    return out


def _parse_label(raw: object) -> tuple[str, float]:
    """Return (sec_type, resid_mat_yrs) from a Tesoro auction label (L-6m, B-5a, O€i-10a)."""
    text = str(raw).strip().replace(" ", "")
    if not text or text.lower() == "nan":
        return "Nominal", float("nan")
    # Syndications / green: "O-30a (s)", "O(BV)-20a (s)"
    t = re.sub(r"\((?:s|bv|s,bv|bv,s)\)", "", text, flags=re.I)
    t = t.replace("€i", "ei").replace("€I", "ei")
    # Ultra / long-end taps: "O->30a", "O->50a"
    m_arrow = re.match(r"^(?P<kind>O|B|Oei|Bei)->(?P<n>\d+(?:[.,]\d+)?)(?P<u>[amy])?$", t, re.I)
    if m_arrow:
        kind = m_arrow.group("kind").lower()
        n = float(m_arrow.group("n").replace(",", "."))
        u = (m_arrow.group("u") or "a").lower()
        years = n / 12.0 if u == "m" else n
        if kind in ("oei", "bei"):
            return "Linker", years
        return "Nominal", years
    m = _AUCTION_LAB.match(t)
    if not m:
        m = re.match(
            r"^(?P<kind>L|Bei|Oei|B|O)-?(?P<n>\d+(?:[.,]\d+)?)(?P<u>[amy])", t, re.I)

    if not m:
        if "ei" in t.lower():
            return "Linker", float("nan")
        if t.upper().startswith("L"):
            return "Bill", 1.0
        return "Nominal", float("nan")
    kind = m.group("kind").lower()
    n = float(m.group("n").replace(",", "."))
    u = (m.group("u") or ("m" if kind == "l" else "a")).lower()
    years = n / 12.0 if u == "m" else n
    if kind == "l":
        return "Bill", years
    if kind in ("oei", "bei"):
        return "Linker", years
    return "Nominal", years


def _as_date(cell0) -> pd.Timestamp | None:
    if isinstance(cell0, pd.Timestamp):
        return cell0.normalize()
    if isinstance(cell0, (np.datetime64,)):
        return pd.Timestamp(cell0).normalize()
    if isinstance(cell0, str) and re.match(r"\d{4}-\d{2}-\d{2}", cell0):
        return pd.to_datetime(cell0, errors="coerce").normalize()
    try:
        if hasattr(cell0, "year") and hasattr(cell0, "month"):
            return pd.Timestamp(cell0).normalize()
    except Exception:  # noqa: BLE001
        return None
    return None


# historico_subastas tabs whose dates are stamped one year late. The tab is right:
# shifted back a year they fall on Tesoro's Tue/Thu auction days (stamped Wed/Fri/Sat),
# and kept as stamped they double-count 2025 and leave 2022 empty.
_SHEET_YEAR_PLUS_ONE = (2022, 2023, 2024)


def _sheet_op_date(op_dt: pd.Timestamp, sheet_year: int) -> pd.Timestamp | None:
    """Auction date for a historico row on tab ``sheet_year``, or None if it does not
    belong to that tab (older corrupt tabs carry rewritten 2022 dates)."""
    dt = pd.Timestamp(op_dt).normalize()
    if dt.year == sheet_year:
        return dt
    if sheet_year in _SHEET_YEAR_PLUS_ONE and dt.year == sheet_year + 1:
        return (dt - pd.DateOffset(years=1)).normalize()
    return None


def _sheet_year_ok(op_dt: pd.Timestamp, sheet_year: int) -> bool:
    return _sheet_op_date(op_dt, sheet_year) is not None


def _auction_sheet(path: Path, sheet: str | int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    year = int(re.search(r"20\d{2}", str(sheet).strip()).group(0))
    rows = []
    last_dt = pd.NaT
    for _, r in raw.iterrows():
        dt = _as_date(r.iloc[0])
        if dt is not None and not pd.isna(dt):
            last_dt = dt
        if pd.isna(last_dt):
            continue
        op_dt = _sheet_op_date(last_dt, year)
        if op_dt is None:
            continue
        # short: cols 1,2,4,5  long: cols 7,8,10,11,12
        for label_i, bid_i, issued_i, yld_i, cpn_i in (
            (1, 2, 4, 5, None),
            (7, 8, 10, 12, 11),
        ):
            lab = r.iloc[label_i] if label_i < len(r) else None
            if lab is None or (isinstance(lab, float) and pd.isna(lab)):
                continue
            lab_s = str(lab).strip()
            if not re.match(r"^[LBO]", lab_s, re.I) and "€i" not in lab_s and "ei" not in lab_s.lower():
                continue
            issued = pd.to_numeric(r.iloc[issued_i], errors="coerce") if issued_i < len(r) else np.nan
            if pd.isna(issued) or issued <= 0:
                continue
            sec, resid = _parse_label(lab_s)
            bid = pd.to_numeric(r.iloc[bid_i], errors="coerce") if bid_i < len(r) else np.nan
            yld = pd.to_numeric(r.iloc[yld_i], errors="coerce") if yld_i < len(r) else np.nan
            cpn = (pd.to_numeric(r.iloc[cpn_i], errors="coerce")
                   if cpn_i is not None and cpn_i < len(r) else np.nan)
            mat = pd.NaT
            if pd.notna(resid):
                mat = (op_dt + pd.DateOffset(days=int(round(float(resid) * 365.25)))).normalize()
            rows.append({
                "op_dt": op_dt,
                "issue_dt": op_dt + pd.offsets.BDay(2),
                "sec_desc_long": lab_s,
                "sec_desc": lab_s,
                "amount_issued": float(issued),
                "amount_bid": float(bid) if pd.notna(bid) else np.nan,
                "yield": float(yld) if pd.notna(yld) else np.nan,
                "coupon": float(cpn) if pd.notna(cpn) else np.nan,
                "sec_type": sec,
                "op_type": "Auction",
                "resid_mat_yrs": resid,
                "maturity_dt": mat,
                "source": f"Tesoro subastas {int(op_dt.year)}",
                "is_forecast": False,
            })
    return pd.DataFrame(rows)


def parse_auctions(path: Path | None = None) -> pd.DataFrame:
    """Parse bulletin 11 (current years) plus historico_subastas when present."""
    paths = []
    if path is not None:
        paths = [path]
    else:
        for name in ("11_auctions.xlsx", "historico_subastas.xlsx"):
            p = RAW_ES / name
            if p.exists():
                paths.append(p)
    parts = []
    for p in paths:
        x = pd.ExcelFile(p)
        for sheet in x.sheet_names:
            if re.fullmatch(r"20\d{2}", str(sheet).strip()):
                parts.append(_auction_sheet(p, sheet))
    if not parts:
        return coerce_ops(pd.DataFrame(), "ES")
    out = coerce_ops(pd.concat(parts, ignore_index=True), "ES")
    # historico overlaps bulletin 11 on recent years — keep bulletin amounts (first).
    # The two books label the same line "O€i-10y" and "O€i-10a" (Dec-2025).
    key = out["sec_desc"].astype(str).str.replace(r"(\d)[ya]\b", r"\1a", regex=True)
    dup = pd.DataFrame({"op_dt": out["op_dt"], "k": key, "amt": out["amount_issued"]}).duplicated()
    return out[~dup].reset_index(drop=True)


def isin_check_digit(code11: str) -> str:
    """ISO 6166 check digit for the first 11 characters of an ISIN."""
    s = "".join(ch if ch.isdigit() else str(ord(ch.upper()) - 55) for ch in code11)
    total = 0
    for i, d in enumerate(s[::-1]):
        n = int(d) * (2 if i % 2 == 0 else 1)
        total += n // 10 + n % 10
    return str((10 - (total % 10)) % 10)


def letra_isin(maturity) -> str:
    """Tesoro Letras ISIN: ES0L0 + YYMMDD + check digit."""
    m = pd.Timestamp(maturity)
    body = f"ES0L0{m:%y%m%d}"
    return body + isin_check_digit(body)


def parse_cnmv_isin_html(html: str) -> pd.DataFrame:
    """One page of CNMV ANCV ISIN register for Dirección General del Tesoro."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        joined = " ".join(cells)
        m = _ISIN.search(joined.upper())
        if not m:
            continue
        isin = m.group(1)
        dates = re.findall(r"(\d{2}/\d{2}/\d{4})", joined)
        cfi_m = re.search(r"\b(DB[A-Z]{4})\b", joined)
        cfi = cfi_m.group(1) if cfi_m else ""
        fisn = next((c for c in cells if "ESTADO" in c.upper() or "LET" in c.upper()), "")
        # Coupon from FISN (ESTADO/5.90 OBESEGR…) or interest column
        cpn = np.nan
        fm = re.search(r"ESTADO/(\d+[.,]\d+)", fisn, re.I)
        if fm:
            cpn = float(fm.group(1).replace(",", "."))
        else:
            im = re.search(r"\b(\d+[.,]\d{4})\b", joined)
            if im and cfi.startswith("DBFT"):
                cpn = float(im.group(1).replace(",", "."))
        bill = isin.startswith("ES0L") or bool(re.search(r"\bLT\b|LET", fisn, re.I))
        linker = cfi.startswith("DBV") or bool(re.search(r"VAR OBE|INDEX|€I|\bEI\b", fisn, re.I))
        # Strip principal/coupon legs (DBZ*) are not auction lines
        if cfi.startswith("DBZ"):
            continue
        rows.append({
            "isin": isin,
            "fisn": fisn,
            "issue_dt": pd.to_datetime(dates[0], dayfirst=True, errors="coerce") if dates else pd.NaT,
            "maturity_dt": pd.to_datetime(dates[1], dayfirst=True, errors="coerce") if len(dates) > 1 else pd.NaT,
            "coupon": cpn,
            "cfi": cfi,
            "sec_type": "Bill" if bill else ("Linker" if linker else "Nominal"),
            "source": "CNMV ANCV",
        })
    return pd.DataFrame(rows)


def fetch_cnmv_isins(force: bool = False) -> pd.DataFrame:
    """Estado ISIN book from CNMV ANCV (principals + linkers + Letras; strips dropped)."""
    RAW_CNMV.mkdir(parents=True, exist_ok=True)
    cache_html = RAW_CNMV / "isin_pages"
    cache_html.mkdir(parents=True, exist_ok=True)
    parts = []
    for page in range(15):
        dest = cache_html / f"page_{page:02d}.html"
        if force or not dest.exists() or dest.stat().st_size < 2000:
            url = f"{CNMV_ISIN}&page={page}"
            dest.write_bytes(_get(url, min_bytes=2000))
            time.sleep(0.2)
        if dest.exists():
            parts.append(parse_cnmv_isin_html(
                dest.read_text(encoding="utf-8", errors="replace")))
    if not parts:
        return pd.read_parquet(OUT_CNMV) if OUT_CNMV.exists() else pd.DataFrame()
    out = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    out["issue_dt"] = pd.to_datetime(out["issue_dt"])
    out["maturity_dt"] = pd.to_datetime(out["maturity_dt"])
    out = out.drop_duplicates(subset=["isin"], keep="first").reset_index(drop=True)
    out.to_parquet(OUT_CNMV, index=False)
    return out


def parse_bde_announcement_text(text: str, *, source: str = "") -> pd.DataFrame:
    """Parse a BdE Bonos/Obligaciones announcement PDF text into ISIN rows.

    Letras notices have no ISIN (use ``letra_isin``); this parser is for BON*.pdf.
    ``op_dt`` is the resolución (auction) day; ``settle_dt`` is anotación / emisión.
    """
    settle = None
    m = re.search(
        r"Emisiones para el\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        text, re.I)
    if m:
        mon = _ES_MONTHS.get(m.group(2).lower())
        if mon:
            settle = pd.Timestamp(int(m.group(3)), mon, int(m.group(1)))
    op = None
    rm = re.search(
        r"Resoluci[oó]n\s+subasta\s+((?:\d{1,2}\.\d{1,2}\.\d{4}\s*)+)",
        text, re.I)
    if rm:
        first = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", rm.group(1))
        if first:
            op = pd.Timestamp(int(first.group(3)), int(first.group(2)), int(first.group(1)))
    # Coupon immediately above each ISIN line (columnar PDF → sequential pairs)
    pairs = re.findall(
        r"al\s+(\d+[.,]\d+)\s*%\s*ISIN:\s*(ES[0-9A-Z]{10})",
        text, re.I | re.S)
    if not pairs:
        pairs = [
            (None, isin) for isin in re.findall(r"ISIN:\s*(ES[0-9A-Z]{10})", text, re.I)]
    mats = []
    mm = re.search(
        r"Amortizaci[oó]n\s+((?:\d{1,2}\.\d{1,2}\.\d{4}\s*)+)", text, re.I)
    if mm:
        for d in re.findall(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", mm.group(1)):
            mats.append(pd.Timestamp(int(d[2]), int(d[1]), int(d[0])))
    # Linker flag from the instrument label block (Oblig. €i / indexados).
    linker_isins = set()
    for block in re.finditer(
        r"(Oblig\.?\s*€?i|indexad\w*).{0,80}?ISIN:\s*(ES[0-9A-Z]{10})",
        text, re.I | re.S):
        linker_isins.add(block.group(2).upper())
    rows = []
    for i, (cpn, isin) in enumerate(pairs):
        coupon = float(str(cpn).replace(",", ".")) if cpn is not None else np.nan
        mat = mats[i] if i < len(mats) else pd.NaT
        rows.append({
            "op_dt": op,
            "settle_dt": settle,
            "isin": isin.upper(),
            "coupon": coupon,
            "maturity_dt": mat,
            "sec_type": "Linker" if isin.upper() in linker_isins else "Nominal",
            "source": source or "BdE announcement",
        })
    return pd.DataFrame(rows)


def parse_tesoro_calendar_text(text: str) -> pd.DataFrame:
    """Bonos/Obligaciones auction dates from the Tesoro annual calendar PDF.

    Each month's ``Auctions`` row is Letras 6/12, Letras 3/9, then two Bonos columns.
    """
    rows = []
    for m in re.finditer(
        r"Auctions\s+((?:\d{2}/\d{2}/\d{4}\*?\s*)+)", text or ""):
        dates = re.findall(r"(\d{2})/(\d{2})/(\d{4})", m.group(1))
        # Skip Letras (first two); keep Bonos columns.
        for d, mo, y in dates[2:]:
            rows.append({
                "op_dt": pd.Timestamp(int(y), int(mo), int(d)),
                "product": "Bonos",
                "source": "Tesoro calendar",
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates(subset=["op_dt"]).sort_values("op_dt")
    return out.reset_index(drop=True)


def fetch_tesoro_calendar(*, force: bool = False) -> pd.DataFrame:
    """Download the Tesoro annual auction calendar → ``es_tesoro_calendar.parquet``."""
    RAW_ES.mkdir(parents=True, exist_ok=True)
    if force or not RAW_CAL.exists() or RAW_CAL.stat().st_size < 1000:
        try:
            get_bytes(TESORO_CAL_URL, dest=RAW_CAL, timeout=90, impersonate=True,
                      verify=False, min_bytes=1000)
        except Exception as exc:  # noqa: BLE001
            print(f"  ES Tesoro calendar: {exc}")
            if OUT_CAL.exists():
                return pd.read_parquet(OUT_CAL)
            return pd.DataFrame()
    if not RAW_CAL.exists():
        return pd.read_parquet(OUT_CAL) if OUT_CAL.exists() else pd.DataFrame()
    from pypdf import PdfReader

    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(RAW_CAL)).pages)
    out = parse_tesoro_calendar_text(text)
    if out.empty:
        return pd.read_parquet(OUT_CAL) if OUT_CAL.exists() else out
    out.to_parquet(OUT_CAL, index=False)
    print(f"  ES Tesoro calendar: {len(out)} Bonos auction days through {out['op_dt'].max().date()}")
    return out


def _bde_pdf_urls_from_upcoming(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(r"BON.*\.pdf$|BON_OBL.*\.pdf$", href, re.I):
            continue
        out.append(urljoin(BDE_UPCOMING, href))
    return list(dict.fromkeys(out))


def _bde_candidate_urls(settle: pd.Timestamp) -> list[str]:
    dstr = pd.Timestamp(settle).strftime("%d%m%Y")
    names = (f"BON{dstr}.pdf", f"BON_OBL_{dstr}.pdf")
    return [b + n for b in (BDE_FICHEROS, BDE_MERCADEUDA) for n in names]


def _bde_parse_pdf(dest: Path) -> pd.DataFrame:
    from pypdf import PdfReader

    try:
        text = "\n".join(
            (p.extract_text() or "") for p in PdfReader(io.BytesIO(dest.read_bytes())).pages)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if "ISIN" not in text.upper():
        return pd.DataFrame()
    return parse_bde_announcement_text(text, source=f"BdE {dest.name}")


def _bde_miss_path() -> Path:
    return RAW_BDE / "_miss_urls.json"


def _bde_load_misses() -> set[str]:
    p = _bde_miss_path()
    if not p.exists():
        return set()
    try:
        import json
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return set()


def _bde_save_misses(misses: set[str]) -> None:
    import json
    RAW_BDE.mkdir(parents=True, exist_ok=True)
    # Cap growth — keep newest ~20k probes
    keep = sorted(misses)[-20_000:]
    _bde_miss_path().write_text(json.dumps(keep), encoding="utf-8")


def fetch_bde_announcements(
    settle_dates: list | pd.Series | None = None,
    *,
    force: bool = False,
    op_dates: list | pd.Series | None = None,
    workers: int = 12,
) -> pd.DataFrame:
    """Download BdE BON*.pdf notices and return ISIN × settle rows.

    Probes settle dates (and auction day + 2..4 business days when ``op_dates``
    given). Parallel short-timeout GETs; 404s cached in ``_miss_urls.json``.
    Always re-parses every ``BON*.pdf`` already on disk.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from curl_cffi import requests as cffi

    RAW_BDE.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    try:
        urls.extend(_bde_pdf_urls_from_upcoming(
            _get(BDE_UPCOMING, min_bytes=1000).decode("utf-8", "replace")))
    except Exception:  # noqa: BLE001
        pass
    dates: set[pd.Timestamp] = set()
    if settle_dates is not None:
        for d in pd.to_datetime(pd.Series(settle_dates)).dropna():
            dates.add(pd.Timestamp(d).normalize())
    if op_dates is not None:
        for d in pd.to_datetime(pd.Series(op_dates)).dropna():
            op = pd.Timestamp(d).normalize()
            # Spanish bond settle is typically T+2 / T+3 from resolution
            for k in (2, 3, 4, 1, 5):
                dates.add((op + pd.offsets.BDay(k)).normalize())
    for d in sorted(dates):
        # FICHEROS first — MERCADEUDA mirrors recent only and doubles 404 traffic
        dstr = pd.Timestamp(d).strftime("%d%m%Y")
        for n in (f"BON{dstr}.pdf", f"BON_OBL_{dstr}.pdf"):
            urls.append(BDE_FICHEROS + n)
    seen: set[str] = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    misses = set() if force else _bde_load_misses()
    to_fetch = []
    for url in uniq:
        name = url.rstrip("/").split("/")[-1]
        dest = RAW_BDE / name
        if dest.exists() and dest.stat().st_size >= 1000 and not force:
            continue
        if url in misses and not force:
            continue
        to_fetch.append((url, dest))

    def _one(pair: tuple[str, Path]) -> tuple[str, bool]:
        url, dest = pair
        try:
            resp = cffi.get(url, impersonate="chrome", timeout=8, verify=False)
        except Exception:  # noqa: BLE001
            return url, False
        if resp.status_code != 200 or resp.content[:4] != b"%PDF":
            return url, False
        dest.write_bytes(resp.content)
        return url, True

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(_one, p) for p in to_fetch]
            for fut in as_completed(futs):
                url, ok = fut.result()
                if not ok:
                    misses.add(url)
        _bde_save_misses(misses)

    parts = []
    for dest in sorted(RAW_BDE.glob("BON*.pdf")):
        if dest.stat().st_size < 1000:
            continue
        part = _bde_parse_pdf(dest)
        if not part.empty:
            parts.append(part)
    if not parts:
        if OUT_BDE.exists() and not force:
            return pd.read_parquet(OUT_BDE)
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["settle_dt"] = pd.to_datetime(out["settle_dt"])
    out["maturity_dt"] = pd.to_datetime(out["maturity_dt"])
    if "op_dt" in out.columns:
        out["op_dt"] = pd.to_datetime(out["op_dt"])
    out = out.drop_duplicates(subset=["settle_dt", "isin"], keep="last")
    out.to_parquet(OUT_BDE, index=False)
    return out.reset_index(drop=True)


def _es_amount(raw) -> float:
    """Parse Tesoro HTML amounts (1.886,84 / 2.400,00 / 75169-as-751.69)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return float("nan")
    if isinstance(raw, (int, float, np.floating, np.integer)):
        v = float(raw)
        # pandas sometimes drops the decimal comma → 75169 means 751.69
        if v >= 10_000 and abs(v - round(v)) < 1e-9:
            return v / 100.0
        return v
    t = str(raw).strip().replace("\xa0", "").replace(" ", "")
    if not t or t in ("-", "—", "nan"):
        return float("nan")
    if re.fullmatch(r"\d+", t) and len(t) >= 5:
        return float(t) / 100.0
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    return float(t)


def _es_date(raw) -> pd.Timestamp:
    return pd.to_datetime(raw, dayfirst=True, errors="coerce")


def _coupon_from_denom(text: object) -> float:
    m = re.search(r"(\d+[.,]\d+|\d+)\s*%", str(text))
    return float(m.group(1).replace(",", ".")) if m else float("nan")


def _parse_result_kv(table: pd.DataFrame) -> dict:
    """Tall 2-column result table → field dict."""
    if table.shape[1] < 2:
        return {}
    d = {}
    for _, r in table.iterrows():
        k = str(r.iloc[0]).strip().lower()
        d[k] = r.iloc[1]
    return d


def _row_from_result_fields(fields: dict, *, tenor: str = "") -> dict | None:
    op = _es_date(fields.get("fecha subasta"))
    mat = _es_date(fields.get("fecha vencimiento"))
    issued = _es_amount(fields.get("nominal adjudicado"))
    if pd.isna(op) or pd.isna(mat) or pd.isna(issued) or issued <= 0:
        return None
    second = _es_amount(fields.get("nominal adjudicado (2ª vuelta)"))
    if pd.notna(second) and second > 0:
        issued = issued + second
    denom = fields.get("denominación") or fields.get("denominacion") or tenor
    linker = "€i" in str(denom).lower() or "ei" in str(denom).lower()
    bill = "mes" in tenor.lower() or str(denom).upper().startswith("L")
    return {
        "op_dt": op.normalize(),
        "issue_dt": _es_date(fields.get("fecha de liquidación")),
        "maturity_dt": mat.normalize(),
        "amount_issued": issued,
        "amount_bid": _es_amount(fields.get("nominal solicitado")),
        "yield": _es_amount(fields.get("tipo de interés marginal")),
        "coupon": _coupon_from_denom(denom),
        "sec_desc": str(denom).strip(),
        "sec_desc_long": f"{tenor} {denom}".strip(),
        "sec_type": "Bill" if bill else ("Linker" if linker else "Nominal"),
        "source": "Tesoro resultado subasta",
    }


def parse_latest_results_html(html: str, kind: str) -> pd.DataFrame:
    """Wide 'última subasta' table (columns = tenors, rows = fields)."""
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        return pd.DataFrame()
    t = tables[0]
    if "Plazo" not in t.columns:
        return pd.DataFrame()
    t = t.set_index("Plazo")
    rows = []
    for col in t.columns:
        fields = {str(i).strip().lower(): t.at[i, col] for i in t.index}
        fields.setdefault("denominación", fields.get("denominación", col))
        rec = _row_from_result_fields(fields, tenor=str(col))
        if rec:
            rows.append(rec)
    return pd.DataFrame(rows)


def parse_detail_results_html(html: str, title: str = "") -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    rows = []
    for tab in tables:
        if tab.shape[1] == 2 and tab.shape[0] >= 5:
            fields = _parse_result_kv(tab)
            rec = _row_from_result_fields(fields, tenor=title)
            if rec:
                rows.append(rec)
        elif tab.shape[1] > 2 and "Plazo" in list(tab.columns):
            part = parse_latest_results_html(
                "<table>" + tab.to_html(index=False) + "</table>", kind="")
            if not part.empty:
                rows.extend(part.to_dict("records"))
    return pd.DataFrame(rows)


def fetch_auction_results(force: bool = False, max_pages: int = 40) -> pd.DataFrame:
    """Scrape Tesoro result pages → one row per instrument with maturity (for ISIN attach)."""
    RAW_ES.mkdir(parents=True, exist_ok=True)
    cache_html = RAW_ES / "results"
    cache_html.mkdir(parents=True, exist_ok=True)
    parts = []
    prior = pd.read_parquet(OUT_RESULTS) if OUT_RESULTS.exists() and not force else pd.DataFrame()
    if not prior.empty:
        parts.append(prior)

    for kind, url in RESULTS_LATEST.items():
        dest = cache_html / f"latest_{kind}.html"
        if force or not dest.exists():
            dest.write_bytes(_get(url, min_bytes=2000))
            time.sleep(0.3)
        if dest.exists():
            parts.append(parse_latest_results_html(
                dest.read_text(encoding="utf-8", errors="replace"), kind))

    # Only re-parse detail HTML that is new since the last results write
    detail_mtime_cut = OUT_RESULTS.stat().st_mtime if OUT_RESULTS.exists() and not force else 0.0
    for dpath in sorted(cache_html.glob("detail_*.html")):
        if not force and dpath.stat().st_mtime <= detail_mtime_cut:
            continue
        parts.append(parse_detail_results_html(
            dpath.read_text(encoding="utf-8", errors="replace"), title=dpath.stem))

    # Paginated historical listing — walk "siguiente" from the main index only.
    # Year-window URLs often loop the same "siguiente" chain for minutes; maturity
    # for ISIN attach comes from BdE + Wayback outstanding instead.
    seen_urls: set[str] = set()
    page_url = RESULTS_HIST
    for page_n in range(max_pages):
        dest = cache_html / f"hist_index_0_{page_n}.html"
        if force or not dest.exists():
            try:
                dest.write_bytes(_get(page_url, min_bytes=2000))
            except Exception:  # noqa: BLE001
                break
            time.sleep(0.25)
        if not dest.exists():
            break
        html = dest.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        detail_links = []
        next_url = None
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            href = urljoin(RESULTS_HIST, a["href"])
            if "TesoroSubastasWebPortlet" in href and text.lower().startswith("subasta"):
                if href not in seen_urls:
                    detail_links.append((text, href))
                    seen_urls.add(href)
            if text.lower() in ("siguiente ›", "siguiente", "next") or "siguiente" in text.lower():
                next_url = href
        for title, href in detail_links:
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:80]
            dpath = cache_html / f"detail_0_{page_n}_{slug}_{abs(hash(href)) % 10**8}.html"
            if force or not dpath.exists():
                try:
                    dpath.write_bytes(_get(href, min_bytes=1000))
                except Exception:  # noqa: BLE001
                    continue
                time.sleep(0.15)
            if dpath.exists() and (force or dpath.stat().st_mtime > detail_mtime_cut):
                parts.append(parse_detail_results_html(
                    dpath.read_text(encoding="utf-8", errors="replace"), title=title))
        if not next_url or next_url == page_url:
            break
        page_url = next_url

    if not parts:
        return prior if not prior.empty else pd.DataFrame()
    out = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    out["op_dt"] = pd.to_datetime(out["op_dt"])
    out["maturity_dt"] = pd.to_datetime(out["maturity_dt"])
    out = out.drop_duplicates(subset=["op_dt", "maturity_dt", "amount_issued"], keep="last")
    out.to_parquet(OUT_RESULTS, index=False)
    return out.reset_index(drop=True)


def attach_isins(ops: pd.DataFrame, stock: pd.DataFrame,
                 results: pd.DataFrame | None = None,
                 bde: pd.DataFrame | None = None,
                 cnmv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill ``isin`` / ``maturity_dt`` on auction ops from results + BdE + CNMV + outstanding."""
    out = ops.copy()
    if out.empty:
        return out
    out["isin"] = out["isin"].astype("string")
    # Bond / linker: unique maturity → ISIN on the live book, then CNMV (covers matured)
    mat_map: dict[pd.Timestamp, str] = {}
    cpn_mat_map: dict[tuple[float, pd.Timestamp], str] = {}
    for book in (stock, cnmv if cnmv is not None else (
            pd.read_parquet(OUT_CNMV) if OUT_CNMV.exists() else pd.DataFrame())):
        if book is None or book.empty:
            continue
        live = book.dropna(subset=["maturity_dt", "isin"]).copy()
        live["maturity_dt"] = pd.to_datetime(live["maturity_dt"]).dt.normalize()
        for mat, g in live.groupby("maturity_dt"):
            mat_ts = pd.Timestamp(mat)
            if mat_ts not in mat_map:
                if len(g) == 1:
                    mat_map[mat_ts] = str(g.iloc[0]["isin"])
                else:
                    pref = g[~g["sec_type"].astype(str).eq("Linker")] if "sec_type" in g else g
                    mat_map[mat_ts] = str((pref if len(pref) else g).iloc[0]["isin"])
        if "coupon" in live.columns:
            for _, r in live.dropna(subset=["coupon"]).iterrows():
                key = (round(float(r["coupon"]), 2), pd.Timestamp(r["maturity_dt"]).normalize())
                if key not in cpn_mat_map and str(r["isin"]).startswith("ES"):
                    cpn_mat_map[key] = str(r["isin"])

    res = results if results is not None else (
        pd.read_parquet(OUT_RESULTS) if OUT_RESULTS.exists() else pd.DataFrame())
    if not res.empty:
        res = res.copy()
        res["op_dt"] = pd.to_datetime(res["op_dt"]).dt.normalize()
        res["maturity_dt"] = pd.to_datetime(res["maturity_dt"]).dt.normalize()
        res["amount_issued"] = pd.to_numeric(res["amount_issued"], errors="coerce")
        # Match bulletin row → result by auction date + amount (within €2mn)
        for i, row in out.iterrows():
            if pd.notna(row.get("isin")) and str(row.get("isin")).startswith("ES"):
                continue
            day = pd.Timestamp(row["op_dt"]).normalize()
            amt = float(row["amount_issued"]) if pd.notna(row["amount_issued"]) else np.nan
            cand = res[res["op_dt"] == day]
            if cand.empty or pd.isna(amt):
                continue
            cand = cand.assign(err=(cand["amount_issued"] - amt).abs())
            hit = cand.nsmallest(1, "err")
            if hit.empty or float(hit.iloc[0]["err"]) > 2.0:
                # try coupon match for bonds
                if pd.notna(row.get("coupon")) and not str(row.get("sec_type")) == "Bill":
                    cpn = float(row["coupon"])
                    c2 = cand[cand["coupon"].sub(cpn).abs() < 0.02]
                    if len(c2) == 1:
                        hit = c2
                    else:
                        continue
                else:
                    continue
            mat = pd.Timestamp(hit.iloc[0]["maturity_dt"]).normalize()
            out.at[i, "maturity_dt"] = mat
            if pd.isna(out.at[i, "issue_dt"]) and pd.notna(hit.iloc[0].get("issue_dt")):
                out.at[i, "issue_dt"] = pd.Timestamp(hit.iloc[0]["issue_dt"])
            if str(row.get("sec_type")) == "Bill" or str(hit.iloc[0].get("sec_type")) == "Bill":
                out.at[i, "isin"] = letra_isin(mat)
                out.at[i, "sec_type"] = "Bill"
            else:
                isin = mat_map.get(mat)
                if isin:
                    out.at[i, "isin"] = isin
                # coupon/sec_type from result if missing
                if pd.isna(row.get("coupon")) and pd.notna(hit.iloc[0].get("coupon")):
                    out.at[i, "coupon"] = float(hit.iloc[0]["coupon"])

    # Letras with maturity but no ISIN yet (formula)
    bill = out["sec_type"].astype(str).eq("Bill") & out["maturity_dt"].notna()
    need = bill & (out["isin"].isna() | ~out["isin"].astype(str).str.startswith("ES0L"))
    out.loc[need, "isin"] = out.loc[need, "maturity_dt"].map(letra_isin)

    # BdE announcement PDFs: settle + coupon → ISIN (authoritative for live reopenings)
    bde_df = bde if bde is not None else (
        pd.read_parquet(OUT_BDE) if OUT_BDE.exists() else pd.DataFrame())
    if not bde_df.empty:
        bde_df = bde_df.copy()
        bde_df["settle_dt"] = pd.to_datetime(bde_df["settle_dt"]).dt.normalize()
        bde_df["maturity_dt"] = pd.to_datetime(bde_df["maturity_dt"]).dt.normalize()
        bde_df["coupon"] = pd.to_numeric(bde_df["coupon"], errors="coerce")
        for i, row in out[~out["sec_type"].astype(str).eq("Bill")].iterrows():
            if pd.notna(row.get("isin")) and str(row["isin"]).startswith("ES"):
                continue
            settle = pd.to_datetime(row.get("issue_dt"), errors="coerce")
            if pd.isna(settle):
                settle = pd.Timestamp(row["op_dt"]) + pd.offsets.BDay(3)
            settle = pd.Timestamp(settle).normalize()
            day = bde_df[bde_df["settle_dt"] == settle]
            if day.empty:
                # bulletin issue_dt is BDay(2) guess — allow ±2 business days
                lo = settle - pd.offsets.BDay(2)
                hi = settle + pd.offsets.BDay(2)
                day = bde_df[(bde_df["settle_dt"] >= lo) & (bde_df["settle_dt"] <= hi)]
            if day.empty:
                continue
            hit = day
            if pd.notna(row.get("coupon")):
                cpn = float(row["coupon"])
                c2 = day[day["coupon"].sub(cpn).abs() < 0.02]
                if not c2.empty:
                    hit = c2
            if pd.notna(row.get("maturity_dt")) and len(hit) > 1:
                mat = pd.Timestamp(row["maturity_dt"]).normalize()
                m2 = hit[hit["maturity_dt"] == mat]
                if not m2.empty:
                    hit = m2
            if len(hit) != 1:
                # residual tenor vs amortización
                if pd.notna(row.get("resid_mat_yrs")):
                    op = pd.Timestamp(row["op_dt"])
                    hit = hit.assign(
                        err=((hit["maturity_dt"] - op).dt.days / 365.25
                             - float(row["resid_mat_yrs"])).abs())
                    hit = hit.nsmallest(1, "err")
                    if hit.empty or float(hit.iloc[0]["err"]) > 1.5:
                        continue
                else:
                    continue
            out.at[i, "isin"] = str(hit.iloc[0]["isin"])
            if pd.notna(hit.iloc[0].get("maturity_dt")):
                out.at[i, "maturity_dt"] = pd.Timestamp(hit.iloc[0]["maturity_dt"]).normalize()
            if pd.notna(hit.iloc[0].get("coupon")) and pd.isna(row.get("coupon")):
                out.at[i, "coupon"] = float(hit.iloc[0]["coupon"])
            if pd.notna(hit.iloc[0].get("settle_dt")):
                out.at[i, "issue_dt"] = pd.Timestamp(hit.iloc[0]["settle_dt"]).normalize()

    # Coupon + maturity from results/BdE → CNMV / live book
    for i, row in out[~out["sec_type"].astype(str).eq("Bill")].iterrows():
        if pd.notna(row.get("isin")) and str(row["isin"]).startswith("ES"):
            continue
        if pd.notna(row.get("coupon")) and pd.notna(row.get("maturity_dt")):
            key = (round(float(row["coupon"]), 2), pd.Timestamp(row["maturity_dt"]).normalize())
            if key in cpn_mat_map:
                out.at[i, "isin"] = cpn_mat_map[key]
                continue
        if pd.notna(row.get("maturity_dt")):
            isin = mat_map.get(pd.Timestamp(row["maturity_dt"]).normalize())
            if isin:
                out.at[i, "isin"] = isin

    # Bonds: nearest CNMV/outstanding line by coupon+tenor; historico "tipo nominal"
    # is often wrong on reopenings (e.g. 2.80 vs true 2.50) → fall back to tenor-only.
    books = []
    if stock is not None and not stock.empty:
        books.append(stock)
    cnmv_df = cnmv if cnmv is not None else (
        pd.read_parquet(OUT_CNMV) if OUT_CNMV.exists() else pd.DataFrame())
    if cnmv_df is not None and not cnmv_df.empty:
        books.append(cnmv_df)
    pool = pd.concat(books, ignore_index=True) if books else pd.DataFrame()
    if not pool.empty and "sec_type" in pool.columns:
        pool = pool[~pool["sec_type"].astype(str).eq("Bill")].copy()
    for i, row in out[~out["sec_type"].astype(str).eq("Bill")].iterrows():
        if pd.notna(row.get("isin")) and str(row["isin"]).startswith("ES"):
            continue
        if pool.empty or pd.isna(row.get("resid_mat_yrs")):
            continue
        op = pd.Timestamp(row["op_dt"])
        target_yrs = float(row["resid_mat_yrs"])
        live = pool.copy()
        # Prefer lines already (or just) issued by the auction date
        if "issue_dt" in live.columns:
            issued = live["issue_dt"].isna() | (pd.to_datetime(live["issue_dt"]) <= op + pd.Timedelta(days=14))
            if issued.any():
                live = live[issued]
        if str(row.get("sec_type")) == "Linker":
            live = live[live["sec_type"].astype(str).eq("Linker")] if "sec_type" in live else live
        else:
            live = live[~live["sec_type"].astype(str).eq("Linker")] if "sec_type" in live else live
        if live.empty:
            continue
        live = live.assign(yrs=(pd.to_datetime(live["maturity_dt"]) - op).dt.days / 365.25)
        live = live[live["yrs"] > 0.2]
        if live.empty:
            continue
        tol = 2.5 if "(s)" in str(row.get("sec_desc", "")).lower() or "bv" in str(
            row.get("sec_desc", "")).lower() else 1.25
        # Auction label class: B-* = Bono, O-* = Obligación, O€i-* = linker.
        # Prefer matching the outstanding/CNMV label letter so a 3y Bono does not
        # latch onto a nearby Obligación with a closer residual tenor.
        desc = str(row.get("sec_desc", "")).upper().replace(" ", "")
        is_linker = (
            str(row.get("sec_type")) == "Linker"
            or "€I" in desc or desc.startswith("BEI") or desc.startswith("OEI")
            or "EI-" in desc
        )
        want_bono = (not is_linker) and desc.startswith("B") and not desc.startswith("BV")
        want_obl = (not is_linker) and desc.startswith("O") and "EI" not in desc
        lab = live.get("sec_desc_long", pd.Series("", index=live.index)).fillna("").astype(str)
        fisn = live["fisn"].fillna("").astype(str) if "fisn" in live.columns else lab
        tag = (lab + " " + fisn).str.upper()
        class_pen = pd.Series(0.0, index=live.index)
        if want_bono:
            class_pen = (~tag.str.contains(r"\bB\b|BOEST|BONO", regex=True)).astype(float) * 3.0
        elif want_obl:
            class_pen = (~tag.str.contains(r"\bO\b|OBE|OBLIG", regex=True) |
                         tag.str.contains(r"€I|EI\b|BOEST", regex=True)).astype(float) * 3.0
        cpn = float(row["coupon"]) if pd.notna(row.get("coupon")) else float("nan")
        cpn_err = (
            pd.to_numeric(live["coupon"], errors="coerce").sub(cpn).abs()
            if pd.notna(cpn) else pd.Series(0.0, index=live.index)
        )
        cpn_err = cpn_err.fillna(0.5)
        # Joint score: tenor dominates; coupon breaks ties; class mismatch penalised.
        # Historico tipo often ≠ true coupon (±0.3) so coupon weight stays light.
        scored = live.assign(
            err=(live["yrs"] - target_yrs).abs() + 0.25 * cpn_err + class_pen)
        hit = scored.nsmallest(1, "err")
        if not hit.empty and float(hit.iloc[0]["err"]) < tol + 0.5:
            out.at[i, "isin"] = str(hit.iloc[0]["isin"])
            out.at[i, "maturity_dt"] = pd.Timestamp(hit.iloc[0]["maturity_dt"]).normalize()
            if pd.notna(hit.iloc[0].get("coupon")):
                out.at[i, "coupon"] = float(hit.iloc[0]["coupon"])

    # resid from real maturity when we have it
    has = out["maturity_dt"].notna() & out["op_dt"].notna()
    out.loc[has, "resid_mat_yrs"] = (
        (pd.to_datetime(out.loc[has, "maturity_dt"]) - pd.to_datetime(out.loc[has, "op_dt"])).dt.days
        / 365.25)
    return out


def parse_outstanding_html(html: str) -> pd.DataFrame:
    """Letras + Bonos/Obligaciones tables on the valores-en-circulación page."""
    tables = pd.read_html(io.StringIO(html))
    rows = []
    asof = None
    m = re.search(r"[Dd]ata as at\s+(\d{1,2}/\d{1,2}/\d{4})", html)
    if m:
        asof = _es_date(m.group(1))
    if asof is None or pd.isna(asof):
        m = re.search(r"[Dd]atos a\s+(\d{1,2}/\d{1,2}/\d{4})", html, re.I)
        if m:
            asof = _es_date(m.group(1))
    if asof is None or pd.isna(asof):
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", html)
        asof = _es_date(m.group(1)) if m else pd.Timestamp.today()
    for t in tables:
        cols = {str(c).lower(): c for c in t.columns}
        isin_c = next((cols[k] for k in cols if "isin" in k), None)
        mat_c = next((cols[k] for k in cols if "matur" in k or "vencim" in k), None)
        amt_c = next((cols[k] for k in cols if "outstanding" in k or "circulaci" in k or "importe" in k), None)
        if isin_c is None or mat_c is None or amt_c is None:
            continue
        idx_c = next((cols[k] for k in cols if "index" in k or "coef" in k), None)
        face_c = next((cols[k] for k in cols if "face" in k or "nominal" in k), None)
        for _, r in t.iterrows():
            raw_isin = str(r[isin_c])
            # Label is "ES00000128H5 O 1,30**" — match ISIN before stripping spaces
            # (removing spaces would glue the coupon letter onto the ISIN).
            m_isin = _ISIN.search(raw_isin.upper())
            if not m_isin:
                continue
            isin = m_isin.group(1)
            mat = _es_date(r[mat_c])
            stock = _es_amount(r[amt_c])
            if pd.isna(stock) or stock <= 0 or pd.isna(mat):
                continue
            linker = bool(idx_c and pd.notna(r.get(idx_c))) or "€i" in raw_isin or "ei" in raw_isin.lower()
            bill = isin.startswith("ES0L")
            face = _es_amount(r[face_c]) if face_c else np.nan
            # Index ratio ~1.0; read_html often turns "1,04764" into 104764.
            coef = np.nan
            if idx_c and pd.notna(r.get(idx_c)) and str(r.get(idx_c)).strip() not in ("", "nan"):
                raw_c = str(r[idx_c]).strip().replace("\xa0", "")
                if "," in raw_c and "." not in raw_c:
                    coef = float(raw_c.replace(",", "."))
                else:
                    coef = pd.to_numeric(raw_c.replace(",", ""), errors="coerce")
                    if pd.notna(coef) and coef > 10:
                        coef = float(coef) / 100_000.0
            cpn_m = re.search(r"(\d+[.,]\d+|\d+)\s*\*?\*?$", raw_isin.replace(isin, "").strip())
            coupon = float(cpn_m.group(1).replace(",", ".")) if cpn_m else np.nan
            rows.append({
                "country": "ES", "ccy": "EUR", "asof": pd.Timestamp(asof).normalize(),
                "isin": isin, "maturity_dt": pd.Timestamp(mat).normalize(),
                "stock_mn": float(stock),
                "face_mn": float(face) if pd.notna(face) else float(stock),
                "index_ratio": float(coef) if pd.notna(coef) else (1.0 if linker else np.nan),
                "coupon": coupon,
                "sec_type": "Bill" if bill else ("Linker" if linker else "Nominal"),
                "sec_desc_long": raw_isin.strip(),
                "source": "Tesoro valores en circulación",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["resid_mat_yrs"] = (out["maturity_dt"] - out["asof"]).dt.days / 365.25
    return out.sort_values(["sec_type", "maturity_dt"]).reset_index(drop=True)


def fetch_outstanding(force: bool = False) -> pd.DataFrame:
    dest = RAW_ES / "valores_en_circulacion.html"
    RAW_ES.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        dest.write_bytes(_get(OUTSTANDING, min_bytes=5000))
    return parse_outstanding_html(dest.read_text(encoding="utf-8", errors="replace"))


def parse_bde_carv_text(text: str, *, source: str = "") -> pd.DataFrame:
    """Parse one BdE ``carvYYYY.txt`` (características valores en circulación).

    Semicolon fields: ISIN; tipo; desc; issue; coupon; …; maturity (field 8).
    Keep Estado BON lines only (drop strips NDP/CUP, Letras, regional AUT).
    """
    rows = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(";")]
        if len(parts) < 9:
            continue
        isin = parts[0].upper()
        if not re.fullmatch(r"ES[0-9A-Z]{10}", isin):
            continue
        tipo = parts[1].upper()
        if tipo in ("NDP", "CUP", "LET", "AUT"):
            continue
        desc = parts[2]
        if "ESTADO" not in desc.upper():
            continue
        try:
            coupon = float(parts[4])
        except ValueError:
            coupon = float("nan")
        issue = _es_date(parts[3])
        mat = _es_date(parts[8])
        if pd.isna(mat):
            continue
        bill = isin.startswith("ES0L")
        # carv marks inflation legs with interest-type code ``I`` (fixed = ``F``)
        linker = bool(re.search(r";\s*I\s*;", line)) or "INDEX" in desc.upper()
        kind_letter = "€i" if linker else ("B" if "BONOS" in desc.upper() else "O")
        rows.append({
            "country": "ES", "ccy": "EUR",
            "isin": isin,
            "coupon": coupon,
            "issue_dt": issue,
            "maturity_dt": pd.Timestamp(mat).normalize(),
            "sec_type": "Bill" if bill else ("Linker" if linker else "Nominal"),
            "sec_desc_long": f"{isin} {kind_letter} {coupon:g}",
            "fisn": desc,
            "source": source or "BdE carv",
        })
    return pd.DataFrame(rows)


def fetch_bde_carv(force: bool = False) -> pd.DataFrame:
    """BdE carv.zip — Estado instrument master 1988–2018 (matured ISINs)."""
    from curl_cffi import requests as cffi

    RAW_CARV.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "es" / "bde_carv.zip"
    if force or not zip_path.exists() or zip_path.stat().st_size < 1000:
        s = cffi.Session()
        s.get("https://repositorio.bde.es/handle/123456789/16988",
              impersonate="chrome", timeout=60, verify=False)
        resp = s.get(
            BDE_CARV_ZIP, impersonate="chrome", timeout=180, verify=False,
            headers={"Referer": "https://repositorio.bde.es/handle/123456789/16988"})
        if resp.status_code != 200 or resp.content[:2] != b"PK":
            return pd.read_parquet(OUT_CARV) if OUT_CARV.exists() else pd.DataFrame()
        zip_path.write_bytes(resp.content)
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_CARV)

    parts = []
    for dest in sorted(RAW_CARV.glob("carv*.txt")):
        part = parse_bde_carv_text(
            dest.read_text(encoding="latin-1", errors="replace"),
            source=f"BdE {dest.name}")
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.read_parquet(OUT_CARV) if OUT_CARV.exists() else pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["isin"], keep="last").reset_index(drop=True)
    out.to_parquet(OUT_CARV, index=False)
    return out


def fetch_outstanding_history(force: bool = False, *, max_snaps: int = 40) -> pd.DataFrame:
    """Union of Wayback Machine snapshots of valores-en-circulación (matured ISINs).

    Live Tesoro / CNMV only list lines still outstanding. Auction history needs the
    lines that have already amortized — those appear in older snapshots.
    """
    from urllib.parse import quote

    from curl_cffi import requests as cffi

    raw_wb = RAW_ES / "wayback_outstanding"
    raw_wb.mkdir(parents=True, exist_ok=True)
    # Re-parse whatever is already on disk first (offline-friendly)
    parts: list[pd.DataFrame] = []
    for dest in sorted(raw_wb.glob("valores_*.html")):
        try:
            df = parse_outstanding_html(dest.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        ts = dest.stem.split("_")[-1][:8]
        df["asof"] = pd.to_datetime(ts, format="%Y%m%d", errors="coerce")
        parts.append(df)

    need_fetch = force or len(list(raw_wb.glob("valores_*.html"))) < 8
    if need_fetch:
        cdx = (
            f"{WAYBACK_CDX}?url="
            f"{quote('tesoro.es/deuda-publica/valores-del-tesoro/valores-en-circulacion')}"
            f"&output=json&fl=timestamp,original,statuscode&filter=statuscode:200&limit=500"
        )
        try:
            rows = cffi.get(cdx, timeout=90, impersonate="chrome").json()
        except Exception:  # noqa: BLE001
            rows = []
        stamps: list[tuple[str, str]] = []
        last_ym = ""
        for row in rows[1:] if rows else []:
            ts, orig = row[0], row[1]
            ym = ts[:6]
            if ym == last_ym:
                continue
            # ~quarterly
            if stamps and (int(ym) - int(stamps[-1][0][:6])) < 3 and ts[:4] == stamps[-1][0][:4]:
                continue
            stamps.append((ts, orig))
            last_ym = ym
            if len(stamps) >= max_snaps:
                break
        for ts, orig in stamps:
            dest = raw_wb / f"valores_{ts}.html"
            if dest.exists() and dest.stat().st_size > 2000 and not force:
                continue
            wb = f"https://web.archive.org/web/{ts}id_/{orig}"
            try:
                resp = cffi.get(wb, timeout=60, impersonate="chrome")
            except Exception:  # noqa: BLE001
                continue
            if resp.status_code != 200 or len(resp.content) < 2000:
                continue
            dest.write_bytes(resp.content)
            time.sleep(0.6)
            try:
                df = parse_outstanding_html(dest.read_text(encoding="utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                continue
            if df.empty:
                continue
            df["asof"] = pd.to_datetime(ts[:8], format="%Y%m%d", errors="coerce")
            parts.append(df)

    if not parts:
        return pd.read_parquet(OUT_STOCK_HIST) if OUT_STOCK_HIST.exists() else pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["isin"], keep="last").reset_index(drop=True)
    out.to_parquet(OUT_STOCK_HIST, index=False)
    return out


def results_after_bulletin(ops0: pd.DataFrame, results: pd.DataFrame | None) -> pd.DataFrame:
    """The Tesoro bulletin (11_auctions.xlsx) is monthly, so the tape stopped ~a month behind
    (Nov-2026: last row 15 Oct while the 5 Nov results were already scraped). Auctions on the
    results pages dated after the bulletin's last day join the tape; attach_isins then maps
    them by maturity like any bulletin row. The bulletin replaces them once it catches up."""
    if results is None or results.empty or ops0 is None or ops0.empty:
        return ops0
    res = results.copy()
    res["op_dt"] = pd.to_datetime(res["op_dt"]).dt.normalize()
    last = pd.to_datetime(ops0["op_dt"]).max()
    new = res[(res["op_dt"] > last) & pd.to_numeric(res["amount_issued"], errors="coerce").gt(0)]
    if new.empty:
        return ops0
    keep = [c for c in ("op_dt", "issue_dt", "maturity_dt", "amount_issued", "amount_bid", "coupon",
                        "sec_desc", "sec_desc_long", "sec_type") if c in new.columns]
    add = new[keep].copy()
    add["source"] = "Tesoro resultado subasta (after bulletin)"
    add = coerce_ops(add, "ES")
    print(f"  ES results after bulletin ({last:%Y-%m-%d}): {len(add)} rows to {add['op_dt'].max():%Y-%m-%d}")
    return pd.concat([ops0, add], ignore_index=True)


def fetch_es(force: bool = False) -> pd.DataFrame:
    download_bulletins(force=force)
    stock = fetch_outstanding(force=force)
    try:
        hist = fetch_outstanding_history(force=force)
    except Exception as e:  # noqa: BLE001
        print(f"ES outstanding history: {e}")
        hist = pd.read_parquet(OUT_STOCK_HIST) if OUT_STOCK_HIST.exists() else pd.DataFrame()
    try:
        carv = fetch_bde_carv(force=force)
    except Exception as e:  # noqa: BLE001
        print(f"ES BdE carv: {e}")
        carv = pd.read_parquet(OUT_CARV) if OUT_CARV.exists() else pd.DataFrame()
    # ISIN book: carv first, then Wayback/live/CNMV overwrite (better linker labels).
    isin_book = pd.concat(
        [d for d in (carv, hist, stock) if d is not None and not d.empty],
        ignore_index=True,
    )
    if not isin_book.empty:
        isin_book = isin_book.drop_duplicates(subset=["isin"], keep="last")
    try:
        results = fetch_auction_results(force=force)
    except Exception as e:  # noqa: BLE001 — bulletin ops still useful without results
        print(f"ES auction results: {e}")
        results = pd.read_parquet(OUT_RESULTS) if OUT_RESULTS.exists() else pd.DataFrame()
    ops0 = results_after_bulletin(parse_auctions(), results)
    bond = ops0[~ops0["sec_type"].astype(str).eq("Bill")] if not ops0.empty else ops0
    try:
        cnmv = fetch_cnmv_isins(force=force)
    except Exception as e:  # noqa: BLE001
        print(f"ES CNMV ISINs: {e}")
        cnmv = pd.read_parquet(OUT_CNMV) if OUT_CNMV.exists() else pd.DataFrame()
    try:
        # Parallel short-timeout probes + miss cache; 10y covers live reopenings + recent maturities.
        cut = pd.Timestamp.today().normalize() - pd.DateOffset(years=10)
        bde = fetch_bde_announcements(
            settle_dates=bond.loc[bond["op_dt"] >= cut, "issue_dt"] if not bond.empty else None,
            op_dates=bond.loc[bond["op_dt"] >= cut, "op_dt"] if not bond.empty else None,
            force=force,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ES BdE announcements: {e}")
        bde = pd.read_parquet(OUT_BDE) if OUT_BDE.exists() else pd.DataFrame()
    ops = attach_tenor(attach_isins(ops0, isin_book, results, bde, cnmv))
    CACHE.mkdir(parents=True, exist_ok=True)
    ops.to_parquet(OUT_OPS, index=False)
    if not stock.empty:
        stock.to_parquet(OUT_STOCK, index=False)
    try:
        fetch_tesoro_calendar(force=force)
    except Exception as exc:  # noqa: BLE001 — calendar feeds the TBA; tape still ships
        print(f"  ES Tesoro calendar failed: {exc}")
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ops = fetch_es(force=a.force)
    stock = pd.read_parquet(OUT_STOCK) if OUT_STOCK.exists() else pd.DataFrame()
    n_isin = int(ops["isin"].astype("string").str.startswith("ES").fillna(False).sum())
    print(f"ops_es: {len(ops)} rows, {ops['op_dt'].min().date()} → {ops['op_dt'].max().date()}, "
          f"with ISIN {n_isin}/{len(ops)}")
    print(f"outstanding: {len(stock)} lines, stock €{stock['stock_mn'].sum()/1e3:.1f}bn"
          if not stock.empty else "outstanding: empty")
    if OUT_RESULTS.exists():
        r = pd.read_parquet(OUT_RESULTS)
        print(f"results: {len(r)} instrument prints "
              f"{r['op_dt'].min().date()} → {r['op_dt'].max().date()}")
    if OUT_BDE.exists():
        b = pd.read_parquet(OUT_BDE)
        print(f"bde announcements: {len(b)} ISIN lines "
              f"{b['settle_dt'].min().date() if b['settle_dt'].notna().any() else '?'} → "
              f"{b['settle_dt'].max().date() if b['settle_dt'].notna().any() else '?'}")
    if OUT_CNMV.exists():
        c = pd.read_parquet(OUT_CNMV)
        print(f"cnmv isins: {len(c)} principal lines")
    if OUT_STOCK_HIST.exists():
        h = pd.read_parquet(OUT_STOCK_HIST)
        print(f"outstanding history (wayback): {len(h)} unique ISINs")
    if OUT_CARV.exists():
        v = pd.read_parquet(OUT_CARV)
        print(f"bde carv (1988-2018): {len(v)} Estado lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def tag_syndications(ops: pd.DataFrame) -> pd.DataFrame:
    """The Tesoro tape books its syndications as auctions (O-10a 20-Jan-2026 EUR15bn, O-30a
    24-Feb-2026 EUR7bn, O-10a 27-May-2026 EUR13bn, O(BV)-20a 8-Sep-2026 EUR4bn). Spanish bond
    auctions are on Thursdays; a single coupon line of EUR2.5bn+ sold on any other weekday is a
    syndication. The forward's syndication repeat needs the tag."""
    if ops is None or ops.empty or "country" not in ops.columns:
        return ops
    out = ops.copy()
    dt = pd.to_datetime(out["op_dt"])
    es = out["country"].astype(str).eq("ES")
    coupon = ~out["sec_type"].astype(str).isin(["Bill"])
    amt = pd.to_numeric(out["amount_issued"], errors="coerce")
    big = es & coupon & (amt >= 2500.0) & dt.dt.dayofweek.ne(3)
    n_lines = out[es & coupon].groupby(dt[es & coupon])["isin"].transform("nunique")
    single = pd.Series(False, index=out.index)
    single.loc[n_lines.index] = n_lines.eq(1)
    hit = big & single & out["op_type"].astype(str).eq("Auction")
    out.loc[hit, "op_type"] = "Syndication"
    return out
