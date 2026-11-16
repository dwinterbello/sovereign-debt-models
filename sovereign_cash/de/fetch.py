"""Germany: raw sources for the federal cash framework, cached under ``sovereign_cash/data/de/``.

finanzagentur   ``schuldenbericht_dt.xlsx`` — monthly federal debt by use (core budget, each
                Sondervermögen) and instrument (Bund/Bobl/Schatz/Bubill/...), YTD gross borrowing
                and redemptions, and the stock of ``Kassenverstärkungskredite und Geldanlagen``
                (net cash-management borrowing less money-market placements). Jan-1995 →.
bmf_mb          BMF Monatsbericht "Entwicklung des Bundeshaushalts" — core-budget YTD Ist:
                expenditure, revenue, Finanzierungssaldo, Kassenmittel, coin, reserves,
                unterjähriger Kapitalmarktsaldo, plus the year's Soll. One HTML page per issue;
                the URL scheme changes across years, so each issue is found from its landing page.
destatis        EU-Haushaltsrahmenrichtlinie monthly table: core budget and Extrahaushalte
                YTD Finanzierungssaldo (recent ~3 years only on the page).

All fetches are cache-first; ``refresh=True`` re-downloads. Network needs a Chrome TLS
fingerprint for bundesfinanzministerium.de (Radware bot wall on plain clients).
"""

from __future__ import annotations

import html as _html
import io
import json
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "de"
MB_DIR = DATA / "bmf_mb"
BMF = "https://www.bundesfinanzministerium.de/"
FA_URL = ("https://www.deutsche-finanzagentur.de/fileadmin/user_upload/Institutionelle-investoren/"
          "berichtswesen/schuldenbericht_dt.xlsx")
DESTATIS_URL = ("https://www.destatis.de/DE/Themen/Staat/Oeffentliche-Finanzen/"
                "EU-Haushaltsrahmenrichtlinie/Tabellen/bund.html")
MONTHS_DE = {"Januar": 1, "Februar": 2, "März": 3, "April": 4, "Mai": 5, "Juni": 6, "Juli": 7,
             "August": 8, "September": 9, "Oktober": 10, "November": 11, "Dezember": 12}


def _get(url: str, timeout: int = 60):
    from curl_cffi import requests as cffi
    return cffi.get(url, impersonate="chrome", timeout=timeout)


# --------------------------------------------------------------------------------------
# Finanzagentur debt statistics


FA_ROWS = {
    # stock sheet row label → column name (EUR, later scaled to bn)
    "Kredite Bundeshaushalt, Sondervermögen, Darlehensfinanzierung FMS & WSF": "total",
    "Bundeshaushalt": "core",
    "Finanzmarktstabilisierungsfonds": "sv_fms",
    "Investitions- und Tilgungsfonds": "sv_itf",
    "Wirtschaftsstabilisierungsfonds": "sv_wsf",
    "Sondervermögen Bundeswehr": "sv_bw",
    "Sondervermögen Infrastruktur und Klimaneutralität": "sv_sik",
    "Restrukturierungsfonds": "sv_rsf",
    "Bundesanleihen": "bund",
    "Bundesobligationen": "bobl",
    "Bundesschatzanweisungen": "schatz",
    "Unverzinsliche Schatzanweisungen des Bundes": "bubill",
    "Unverzinsliche Schatzanweisungen": "bubill",
    "Inflationsindexierte Bundeswertpapiere": "linker",
    "Grüne Bundeswertpapiere": "green",
    "Schuldscheindarlehen": "ssd",
    "Geldmarktgeschäfte zur Haushaltsfinanzierung": "mm_budget",
    "Sonstige Kredite und Buchschulden": "other_loans",
    "Bundeswertpapiere": "securities",
    "Kassenverstärkungskredite und Geldanlagen": "kvk_net",
    "Verbindlichkeiten aus der Kapitalindexierung inflationsindexierter Bundeswertpapiere": "linker_uplift",
    "Sonstige Bundeswertpapiere": "other_sec",
    "Zusatzemissionen des Bundes": "extra_iss",
}


def fetch_finanzagentur(refresh: bool = False) -> Path:
    dest = DATA / "schuldenbericht_dt.xlsx"
    if dest.exists() and not refresh:
        return dest
    r = _get(FA_URL, timeout=120)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise RuntimeError("Finanzagentur download is not an xlsx (blocked?)")
    DATA.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def _fa_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """One sheet → month-indexed frame (EUR bn), first occurrence of each label above 'Nachrichtlich'."""
    raw = xl.parse(sheet, header=None)
    hdr_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Datum")][0]
    dates = pd.to_datetime(raw.iloc[hdr_row, 1:], format="%d.%m.%Y", errors="coerce")
    out: dict[str, pd.Series] = {}
    section = "main"
    for i in range(hdr_row + 1, len(raw)):
        label = str(raw.iloc[i, 0]).strip()
        if label.startswith("Nachrichtlich"):
            section = "memo"
            continue
        if label == "Eigenbestand":
            section = "own"
            continue
        name = FA_ROWS.get(label)
        if label == "Umlaufvolumen" and section == "memo":
            section = "volume"
            continue
        if name is None:
            continue
        key = name if section in ("main", "memo") else f"{section}_{name}"
        if section == "memo" and name != "linker_uplift":
            continue
        if key in out:
            continue
        vals = pd.to_numeric(raw.iloc[i, 1:], errors="coerce").to_numpy() / 1e9
        out[key] = pd.Series(vals, index=dates.to_numpy())
    df = pd.DataFrame(out)
    df = df[df.index.notna()]
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    return df.sort_index()


def load_finanzagentur(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{'stock', 'gross_ytd', 'redemption_ytd'} monthly frames, EUR bn (redemptions positive)."""
    xl = pd.ExcelFile(fetch_finanzagentur(refresh))
    stock = _fa_sheet(xl, "rpgSchuldenstand")
    gross = _fa_sheet(xl, "rpgBruttokreditaufnahme")
    red = -_fa_sheet(xl, "rpgTilgungen")
    return {"stock": stock, "gross_ytd": gross, "redemption_ytd": red}


# --------------------------------------------------------------------------------------
# BMF Monatsbericht: core-budget execution


def _issue_candidates(y: int, m: int) -> list[str]:
    prev = pd.Period(f"{y}-{m:02d}", "M") - 1
    name = {v: k for k, v in MONTHS_DE.items()}[prev.month].lower().replace("ä", "ae")
    k4 = "Inhalte/Kapitel-4-Wirtschafts-und-Finanzlage"
    return [
        f"Monatsberichte/{y}/{m:02d}/{k4}/4-3-entwicklung-des-bundeshaushalts.html",
        f"Monatsberichte/{y}/{m:02d}/{k4}/4-3-entwicklung-des-bundeshaushalts-{name}-{prev.year}.html",
    ]


def _landing_candidates(y: int, m: int) -> list[str]:
    return [
        f"Monatsberichte/Ausgabe/{y}/{m:02d}/monatsbericht-{m:02d}-{y}.html",
        f"Monatsberichte/{y}/{m:02d}/Kapitel/kapitel-4-wirtschafts-und-finanzlage.html",
        f"Monatsberichte/{y}/{m:02d}/Kapitel/kapitel-3-wirtschafts-und-finanzlage.html",
        f"Monatsberichte/{y}/{m:02d}/monatsbericht-{m:02d}-{y}.html",
    ]


def _fetch_issue(y: int, m: int, pause: float = 1.0) -> str | None:
    for path in _issue_candidates(y, m):
        r = _get(BMF + path)
        time.sleep(pause)
        if r.status_code == 200 and "Finanzierungssaldo" in r.text:
            return r.text
    for path in _landing_candidates(y, m):
        r = _get(BMF + path)
        time.sleep(pause)
        if r.status_code != 200:
            continue
        links = re.findall(r'href="([^"]*entwicklung[^"]*bundeshaushalt[^"]*)"', r.text, re.I)
        for link in links:
            url = link if link.startswith("http") else BMF + link.lstrip("/")
            r2 = _get(_html.unescape(url))
            time.sleep(pause)
            if r2.status_code == 200 and "Finanzierungssaldo" in r2.text:
                return r2.text
    return None


def fetch_bmf_mb(start: str = "2014-01", end: str | None = None, refresh: bool = False,
                 verbose: bool = False) -> list[Path]:
    """Download every Monatsbericht issue page in [start, end]; missing issues are logged."""
    MB_DIR.mkdir(parents=True, exist_ok=True)
    end_p = pd.Period(end or pd.Timestamp.today().strftime("%Y-%m"), "M")
    miss_path = MB_DIR / "_missing.json"
    missing = json.loads(miss_path.read_text()) if miss_path.exists() else {}
    got = []
    for p in pd.period_range(start, end_p, freq="M"):
        dest = MB_DIR / f"{p}.html"
        if dest.exists() and not refresh:
            got.append(dest)
            continue
        # A missing issue older than three months is not retried without refresh.
        if str(p) in missing and not refresh and p < end_p - 3:
            continue
        try:
            text = _fetch_issue(p.year, p.month)
        except Exception as exc:  # noqa: BLE001 — network errors are retried next run, not marked missing
            if verbose:
                print(p, "error", exc)
            continue
        if text is None:
            missing[str(p)] = pd.Timestamp.today().strftime("%Y-%m-%d")
            if verbose:
                print(p, "missing")
            continue
        missing.pop(str(p), None)
        dest.write_text(text, encoding="utf-8")
        got.append(dest)
        if verbose:
            print(p, "ok")
    miss_path.write_text(json.dumps(missing, indent=1, sort_keys=True))
    return got


def _num(s: str) -> float | None:
    s = s.strip().replace("\u2212", "-").replace("–", "-").replace(" ", "").replace("\xa0", "")
    if s in ("", "-", "x", "."):
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


_ROW_KEYS = {
    "Ausgaben": "expenditure",
    "Einnahmen": "revenue",
    "Steuereinnahmen": "tax",
    "Finanzierungssaldo": "fin_balance",
    "Kassenmittel": "kassenmittel",
    "Münzeinnahmen": "coins",
    "Saldo der Rücklagenbewegungen": "reserves",
    "Nettokreditaufnahme": "capmkt_balance",
}


_UNIT = re.compile(r"[\s()]*(Mrd\.?)?[\s()]*€?[\s()]*|in|%|in %")


def parse_mb(text: str) -> dict | None:
    """First table of an issue page → {'month', 'soll_*', 'ytd_*', 'ist_prev_*'} in EUR bn."""
    i = text.find("<table")
    if i < 0:
        return None
    j = text.find("</table>", i)
    tab = text[i:j]
    head = re.sub(r"<[^>]+>", " ", tab[: tab.find("Ausgaben")])
    head = _html.unescape(re.sub(r"\s+", " ", head))
    mm = re.findall(r"(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*(20\d\d)", head)
    soll = re.search(r"Soll\s*(?:\d\s+)?(20\d\d)", head)
    if not mm:
        return None
    mon, yr = mm[-1]
    out: dict = {"month": f"{yr}-{MONTHS_DE[mon]:02d}", "soll_year": int(soll.group(1)) if soll else None}
    # Some years wrap rows in <td> instead of <tr>, so walk a flat cell stream.
    body = tab[max(tab.find("Ausgaben") - 400, 0):] if "Ausgaben" in tab else tab
    flat = re.sub(r"<t[hd][^>]*>", "\x01", body)
    flat = _html.unescape(re.sub(r"<[^>]+>", " ", flat))
    cells = [re.sub(r"\s+", " ", c).strip() for c in flat.split("\x01")]
    cells = [c for c in cells if c]
    ncol = 3 if soll else 2
    seen: set[str] = set()
    for n, label in enumerate(cells):
        key = next((v for k, v in _ROW_KEYS.items() if label.startswith(k)), None)
        if key is None or key in seen:
            continue
        # Value cells run to the next text cell; unit fragments ("(", "Mrd.", "€)") and footnote
        # markers sit in between, so keep the last ``ncol`` numeric-or-dash cells.
        vals = []
        for c in cells[n + 1:n + 20]:
            if _UNIT.fullmatch(c):
                continue
            if re.search(r"[A-Za-zÄÖÜäöü]", c):
                break
            if _num(c) is not None or c in ("-", "–", "x"):
                vals.append(c)
        if len(vals) < ncol:
            continue
        seen.add(key)
        nums = [_num(v) for v in vals[-ncol:]]
        if ncol == 2:
            nums = [nums[0], None, nums[1]]
        out[f"ist_prev_{key}"], out[f"soll_{key}"], out[f"ytd_{key}"] = nums
    return out if out.get("ytd_fin_balance") is not None else None


def load_bmf_mb() -> pd.DataFrame:
    """Parsed core-budget YTD rows, one per reported month (latest issue wins)."""
    rows = []
    for f in sorted(MB_DIR.glob("*.html")):
        rec = parse_mb(f.read_text(encoding="utf-8"))
        if rec:
            rec["issue"] = f.stem
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns=KASSE_RENAME)
    df["month"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    return df.sort_values(["month", "issue"]).drop_duplicates("month", keep="last").set_index("month")


# Kassenmittel is the end-of-month cash position, not a year-to-date flow.
KASSE_RENAME = {"ytd_kassenmittel": "kassenmittel_pos"}


# --------------------------------------------------------------------------------------
# BMF Monatsbericht archive PDF: SDDS "Central Government Operations – Haushalt Bund"

SDDS_PDF = ("Monatsberichte/2001-2016/Inhalte/Monatsbericht-Archiv-Downloads/2016/"
            "monatsbericht-2016-12-deutsch.pdf?__blob=publicationFile&v=1")
SDDS_COLS = ("expenditure", "revenue", "fin_balance", "kassenmittel", "coins", "capmkt_balance")


def fetch_sdds_pdf(refresh: bool = False) -> Path:
    dest = DATA / "bmf_mb_2016-12.pdf"
    if dest.exists() and not refresh:
        return dest
    r = _get(BMF + SDDS_PDF, timeout=120)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise RuntimeError("BMF archive PDF not returned")
    DATA.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def _decode_cid(tok: str) -> str:
    """The table font maps digits to cid 18..27, the thousands gap to cid 3 and minus to cid 15."""
    out = []
    for code in re.findall(r"\(cid:(\d+)\)", tok):
        c = int(code)
        if 18 <= c <= 27:
            out.append(str(c - 18))
        elif c == 15:
            out.append("-")
    return "".join(out)


def parse_sdds_pdf(path: Path) -> pd.DataFrame:
    """YTD core-budget SDDS table (EUR bn) for every month the PDF carries (2011-01 → 2016-11)."""
    import pdfplumber

    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "Special Data Dissemination" not in text or "Haushalt Bund" not in text:
                continue
            year = None
            for line in text.splitlines():
                m = re.match(r"^(20\d\d)\s+(.*)$", line)
                if m:
                    year, line = int(m.group(1)), m.group(2)
                parts = line.split(" ", 1)
                if parts[0] not in MONTHS_DE or year is None or len(parts) < 2:
                    continue
                vals = [_decode_cid(t) for t in parts[1].split()]
                vals = [v for v in vals if v not in ("", "-")]
                if len(vals) != 6:
                    continue
                rec = {"month": pd.Timestamp(year, MONTHS_DE[parts[0]], 1)}
                rec.update({f"ytd_{c}": int(v) / 1e3 for c, v in zip(SDDS_COLS, vals)})
                rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates("month").set_index("month").sort_index()
    return df.rename(columns=KASSE_RENAME)


# --------------------------------------------------------------------------------------
# Destatis EU-Haushaltsrahmenrichtlinie (core + Extrahaushalte)


def fetch_destatis(refresh: bool = False) -> pd.DataFrame:
    """Core and Extrahaushalte YTD Finanzierungssaldo, EUR bn; accumulates across downloads."""
    dest = DATA / "destatis_bund.csv"
    old = pd.read_csv(dest, parse_dates=["month"]) if dest.exists() else pd.DataFrame()
    if dest.exists() and not refresh:
        return old.set_index("month")
    r = _get(DESTATIS_URL)
    r.raise_for_status()
    t = r.text
    i = t.find("<table")
    tab = t[i:t.find("</table>", i)]
    rows, year = [], None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tab, re.S):
        cells = [re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
        cells = [c.replace("\xad", "") for c in cells if c.strip() != ""]
        if not cells:
            continue
        if re.fullmatch(r"20\d\d", cells[0]):
            year, cells = int(cells[0]), cells[1:]
        if not cells or cells[0] not in MONTHS_DE or year is None:
            continue
        nums = [_num(c) for c in cells[1:7]]
        if len(nums) < 6 or any(v is None for v in nums):
            continue
        rows.append({"month": pd.Timestamp(year, MONTHS_DE[cells[0]], 1),
                     "core_exp": nums[0] / 1e3, "core_rev": nums[1] / 1e3, "core_fin_balance": nums[2] / 1e3,
                     "extra_exp": nums[3] / 1e3, "extra_rev": nums[4] / 1e3, "extra_fin_balance": nums[5] / 1e3})
    new = pd.DataFrame(rows)
    if new.empty:
        raise RuntimeError("Destatis table parsed to nothing")
    df = pd.concat([old, new]).drop_duplicates("month", keep="last").sort_values("month")
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df.set_index("month")


def fetch_all(refresh: bool = False, verbose: bool = False) -> None:
    fetch_finanzagentur(refresh)
    fetch_destatis(refresh)
    fetch_sdds_pdf(refresh=False)
    fetch_bmf_mb(start="2017-01", refresh=False, verbose=verbose)


if __name__ == "__main__":
    import sys

    fetch_all(refresh="--refresh" in sys.argv, verbose=True)
