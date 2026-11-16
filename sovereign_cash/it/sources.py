"""Fetch and parse the Italian cash sources into tidy monthly frames (EUR mn unless stated).

MEF press releases   state-sector cash balance ('saldo del settore statale'), provisional on the
                     first working days of the next month; each release also restates the same
                     month a year earlier (final). 2010→.
MEF Tesoro           month-end Treasury liquidity: account at Banca d'Italia ('conto disponibilità',
                     from 2025 'saldo dei depositi governativi') + money-market deployment (OPTES).
                     Nov-2011→.
Banca d'Italia       'Finanza pubblica: fabbisogno e debito' monthly supplement. Tavola 1 (central
                     government borrowing requirement), Tavola 2 (general-government financing by
                     instrument, change in Treasury liquidity). Each edition carries ~24 months;
                     the latest vintage wins per month. Dec-2012→.

Raw files are cached under ``sovereign_cash/data/it``; ``refresh=False`` never touches the network
when the cache exists.
"""

from __future__ import annotations

import html as _html
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "it"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"}

DT_BASE = "https://www.dt.mef.gov.it"
LIQ_PAGE = DT_BASE + "/it/debito_pubblico/gestione_liquidita/dati_liquidita/index.html?selezione-anno={y}"
MEF_BASE = "https://www.mef.gov.it"
MEF_INDEX = MEF_BASE + "/ufficio-stampa/comunicati/{y}/index.html?page=1&pagesize=500"
BDI_BASE = "https://www.bancaditalia.it"
BDI_YEAR = BDI_BASE + "/pubblicazioni/finanza-pubblica/{y}-finanza-pubblica/index.html"

MONTHS = {"gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6, "lug": 7, "ago": 8, "set": 9,
          "ott": 10, "nov": 11, "dic": 12}
MONTH_NAMES = {"gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7,
               "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12}


# --------------------------------------------------------------------------------------
# HTTP


def _get(url: str, path: Path, refresh: bool = False, binary: bool = False, pause: float = 0.3) -> Path | None:
    """Download `url` to `path` unless cached. Returns the path, or None on failure."""
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    import requests

    try:
        r = requests.get(url, headers=UA, timeout=60)
    except requests.RequestException:
        return path if path.exists() else None
    if r.status_code != 200 or not r.content:
        return path if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    time.sleep(pause)
    return path


def _text_html(raw: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    t = _html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def _pdf_text(path: Path) -> list[str]:
    import pypdf

    return [p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages]


def it_number(tok: str) -> float:
    """'(17.067)' → 17067, '-2.391' → -2391, '..' → 0, '7,9' → 7.9, '....' → NaN."""
    t = tok.strip().strip("()").strip()
    if t in ("..", "-", "—", ""):
        return 0.0
    if t == "....":
        return float("nan")
    t = t.replace(".", "").replace(",", ".")
    return float(t)


# --------------------------------------------------------------------------------------
# Treasury liquidity (MEF Dipartimento del Tesoro)

_LIQ_ROW = re.compile(r"^\s*(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)-(\d{2})\s*[;\s]\s*(.*)$", re.I)


def parse_liquidity_text(text: str) -> pd.DataFrame:
    """Rows 'mmm-yy; account; [vincolati;] OPTES; total' → month, account_mn, mm_mn, total_mn."""
    rows = []
    for line in text.splitlines():
        m = _LIQ_ROW.match(line)
        if not m:
            continue
        nums = [x for x in re.split(r"[;\s]+", m.group(3).strip()) if x]
        vals = []
        for x in nums:
            try:
                vals.append(it_number(x))
            except ValueError:
                break
        if len(vals) < 3:
            continue
        month = pd.Timestamp(2000 + int(m.group(2)), MONTHS[m.group(1).lower()], 1)
        rows.append({"month": month, "account_mn": vals[0], "total_mn": vals[-1], "mm_mn": vals[-1] - vals[0]})
    return pd.DataFrame(rows, columns=["month", "account_mn", "mm_mn", "total_mn"])


def fetch_liquidity(refresh: bool = False) -> pd.DataFrame:
    """Month-end Treasury liquidity, EUR mn. One file per year from the dati_liquidita page."""
    d = RAW / "liquidity"
    d.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.today().year
    for y in range(2012, this_year + 1):
        idx = _get(LIQ_PAGE.format(y=y), RAW / "index" / f"liq_{y}.html", refresh=refresh and y >= this_year - 1)
        if idx is None:
            continue
        links = sorted(set(re.findall(r'href="([^"]*liquidita_del_tesoro/[^"]+\.(?:csv|pdf))"',
                                      idx.read_text(encoding="utf-8", errors="ignore"))))
        csv = [u for u in links if u.endswith(".csv")]
        for u in (csv or links):
            _get(DT_BASE + u, d / ("liq_" + Path(u).name), refresh=refresh and y == this_year)
    frames = []
    for f in sorted(d.iterdir()):
        if f.suffix == ".csv":
            text = f.read_bytes().decode("latin-1")
        elif f.suffix == ".pdf":
            if f.with_suffix(".csv").exists() or any(f.stem.rstrip("_") == c.stem for c in d.glob("*.csv")):
                continue
            text = "\n".join(_pdf_text(f))
        else:
            continue
        p = parse_liquidity_text(text)
        p["file"] = f.name
        frames.append(p)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["total_mn"])
    # A later file (the running-year CSV) supersedes an earlier one for the same month.
    out["mtime"] = out["file"].map(lambda n: (d / n).stat().st_mtime)
    out = out.sort_values(["month", "mtime"]).drop_duplicates("month", keep="last")
    return out.drop(columns="mtime").set_index("month").sort_index()


# --------------------------------------------------------------------------------------
# Government securities outstanding by type (MEF 'Composizione dei Titoli di Stato')

COMP_PAGE = DT_BASE + "/it/debito_pubblico/dati_statistici/composizione_titoli_stato/index.html?selezione-anno={y}"
SCAD_PAGE = DT_BASE + "/it/debito_pubblico/dati_statistici/titoli_scadenza_prossimi_12_mesi/"


def parse_composition_text(text: str) -> dict | None:
    """Month-end stock of BOT and of all government securities (EUR mn) from one composition file."""
    m = re.search(r"circolazione al[\s;]*(\d{1,2})[\s./_-]*([A-Za-z]+|\d{1,2})[\s./_-]*(\d{4}|\d{2})\b", text, re.I)
    if not m:
        return None
    mo = m.group(2).lower()
    mon = int(mo) if mo.isdigit() else MONTH_NAMES.get(mo, MONTHS.get(mo[:3]))
    if not mon:
        return None
    yr = int(m.group(3))
    month = pd.Timestamp(yr + 2000 if yr < 100 else yr, mon, 1)
    num = r"(\d{1,3}(?:\.\d{3})*,\d+|\d{1,3}(?:\.\d{3})+)"
    bot = re.search(r"^\s*BOT\b[;\s]+" + num, text, re.M)
    tot = re.search(r"^\s*Totale\b[;\s]+" + num, text, re.M)
    if not bot:
        return None
    return {"month": month, "bot_stock_mn": it_number(bot.group(1)),
            "total_stock_mn": it_number(tot.group(1)) if tot else float("nan")}


def fetch_composition(refresh: bool = False, start_year: int = 2012) -> pd.DataFrame:
    d = RAW / "composizione"
    d.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.today().year
    for y in range(start_year, this_year + 1):
        idx = _get(COMP_PAGE.format(y=y), RAW / "index" / f"comp_{y}.html", refresh=refresh and y >= this_year - 1)
        if idx is None:
            continue
        links = sorted(set(re.findall(r'href="([^"]*composizione_titoli_stato/[^"]+\.(?:csv|pdf))"',
                                      idx.read_text(encoding="utf-8", errors="ignore"))))
        stems = {}
        for u in links:
            stems.setdefault(Path(u).stem, []).append(u)
        for stem, us in stems.items():
            u = next((x for x in us if x.endswith(".csv")), us[0])
            _get(DT_BASE + u, d / f"{y}_{Path(u).name}", pause=0.2)
    rows = []
    for f in sorted(d.iterdir()):
        try:
            text = f.read_bytes().decode("latin-1") if f.suffix == ".csv" else "\n".join(_pdf_text(f))
        except Exception:  # noqa: BLE001
            continue
        r = parse_composition_text(text)
        if r:
            # The file name is the publication's own month label; a few files repeat the prior
            # month's header inside.
            fm = (re.search(r"al[_-](\d{1,2})[._-](\d{1,2})[._-](\d{4})", f.name)
                  or re.search(r"al[_-](\d{1,2})[._-]([a-z]+)[._-](\d{4})", f.name, re.I))
            if fm:
                mo = fm.group(2).lower()
                mon = int(mo) if mo.isdigit() else MONTH_NAMES.get(mo)
                if mon:
                    r["month"] = pd.Timestamp(int(fm.group(3)), mon, 1)
            rows.append({**r, "file": f.name})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["month", "file"]).drop_duplicates("month", keep="last").set_index("month").sort_index()


def parse_maturities_text(text: str) -> pd.DataFrame:
    """'Titoli in scadenza nei prossimi 12 mesi': month × (BOT, BTP, BTPi, BTP Italia, CCTeu, other), EUR mn."""
    asof = re.search(r"Aggiornamento al\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    rows = []
    for line in text.splitlines():
        m = _LIQ_ROW.match(line)
        if not m:
            continue
        vals = [it_number(x) for x in re.split(r"[;\s]+", m.group(3).strip()) if x]
        if len(vals) < 7:
            continue
        rows.append({"month": pd.Timestamp(2000 + int(m.group(2)), MONTHS[m.group(1).lower()], 1),
                     "bot_mn": vals[0], "btp_mn": vals[1], "btpi_mn": vals[2], "btp_italia_mn": vals[3],
                     "ccteu_mn": vals[4], "other_mn": vals[5], "total_mn": vals[6]})
    out = pd.DataFrame(rows)
    if asof is not None:
        out.attrs["asof"] = str(pd.Timestamp(int(asof.group(3)), MONTH_NAMES[asof.group(2).lower()],
                                             int(asof.group(1))).date())
    return out


def fetch_maturities(refresh: bool = False) -> pd.DataFrame:
    """Latest MEF 12-month maturity schedule (linkers at indexed capital)."""
    d = RAW / "scadenze"
    d.mkdir(parents=True, exist_ok=True)
    idx = _get(SCAD_PAGE, RAW / "index" / "scad12.html", refresh=refresh)
    if idx is not None:
        for u in set(re.findall(r'href="([^"]*titoli_in_scadenza/[^"]+\.csv)"',
                                idx.read_text(encoding="utf-8", errors="ignore"))):
            _get(DT_BASE + u, d / Path(u).name, pause=0.2)
    best = None
    for f in d.glob("*.csv"):
        p = parse_maturities_text(f.read_bytes().decode("latin-1"))
        if len(p) and (best is None or p.attrs.get("asof", "") > best.attrs.get("asof", "")):
            best = p
    return best if best is not None else pd.DataFrame()


def maturity_vintages() -> pd.DataFrame:
    """Every cached 12-month maturity schedule, long, with its ``vintage`` (as-of date)."""
    frames = []
    for f in (RAW / "scadenze").glob("*.csv"):
        p = parse_maturities_text(f.read_bytes().decode("latin-1"))
        if len(p) and p.attrs.get("asof"):
            frames.append(p.assign(vintage=pd.Timestamp(p.attrs["asof"])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------
# Banca d'Italia supplement

_NUM = r"\(?-?\d{1,3}(?:\.\d{3})*(?:,\d+)?\)?|\(?\.\.\.?\.?\)?"
_ROW_START = re.compile(r"^\s*(?:(\d{4})\s*-\s*)?(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\b\.?", re.I)
T1_COLS = ("tax_rev", "other_rev", "tot_rev", "cur_exp", "cap_exp", "tot_exp", "budget_bal", "treasury_bal",
           "treasury_fondi", "fabb_ac", "privatisations", "fabb_ac_net")
T2_COLS_11 = ("deposits", "postal", "st_securities", "mlt_securities", "mfi_loans", "eu_loans", "other_liab",
              "debt_transactions", "liq_change_neg", "liq_deploy_neg", "fabb_ap")
T2_COLS_10 = ("deposits", "postal", "st_securities", "mlt_securities", "mfi_loans", "other_liab",
              "debt_transactions", "liq_change_neg", "liq_deploy_neg", "fabb_ap")


def parse_bdi_table(text: str, ncols: tuple[int, ...]) -> list[tuple[pd.Timestamp, list[float]]]:
    """Monthly rows of a BdI supplement table. Annual rows ('2024 ....') are skipped."""
    out = []
    year = None
    for line in text.splitlines():
        line = re.sub(r"\.{3,}", " ", line)
        m = _ROW_START.match(line)
        if not m:
            continue
        if m.group(1):
            year = int(m.group(1))
        if year is None:
            continue
        rest = line[m.end():]
        toks = re.findall(_NUM, rest)
        if len(toks) not in ncols:
            continue
        try:
            vals = [it_number(t) for t in toks]
        except ValueError:
            continue
        out.append((pd.Timestamp(year, MONTHS[m.group(2).lower()], 1), vals))
    return out


def parse_bdi_pdf(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tavola 1 and Tavola 2 monthly rows from one supplement edition (EUR mn)."""
    pages = _pdf_text(path)
    t1, t2 = [], []
    for txt in pages:
        if "TCCE0100" in txt and "Formazione del fabbisogno" in txt:
            for month, v in parse_bdi_table(txt, (12,)):
                if abs(v[9] + v[6] + v[7]) <= 3:  # fabbisogno = −(budget + treasury balance)
                    t1.append({"month": month, **dict(zip(T1_COLS, v))})
        if "TCCE0125" in txt and "strumenti" in txt:
            for month, v in parse_bdi_table(txt, (10, 11)):
                cols = T2_COLS_11 if len(v) == 11 else T2_COLS_10
                r = dict(zip(cols, v))
                parts = ("deposits", "st_securities", "mlt_securities", "mfi_loans", "eu_loans", "other_liab")
                if abs(sum(r.get(c, 0.0) for c in parts) - r["debt_transactions"]) > 3:
                    continue
                if abs(r["debt_transactions"] + r["liq_change_neg"] - r["fabb_ap"]) > 3:
                    continue
                r.setdefault("eu_loans", 0.0)
                t2.append({"month": month, **r})
    return pd.DataFrame(t1), pd.DataFrame(t2)


def _bdi_editions(refresh: bool) -> list[Path]:
    d = RAW / "bdi_fpi"
    d.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.today().year
    for y in range(2015, this_year + 1):
        idx = _get(BDI_YEAR.format(y=y), RAW / "index" / f"bdi_{y}.html", refresh=refresh and y >= this_year - 1)
        if idx is None:
            continue
        links = sorted(set(re.findall(rf"(/pubblicazioni/finanza-pubblica/{y}-finanza-pubblica/[^\"]+\.pdf)",
                                      idx.read_text(encoding="utf-8", errors="ignore"))))
        if not links:
            continue

        def _key(u: str) -> tuple:
            n = Path(u).stem
            m = re.search(r"(\d{8})", n)
            if m:
                s = m.group(1)
                return (s, ) if s.startswith("20") else (s[4:] + s[2:4] + s[:2],)
            m = re.search(r"suppl_(\d+)_", n)
            return (f"{y}{int(m.group(1)):04d}",) if m else (n,)

        links = sorted(links, key=_key)
        # Each edition carries ~two years of months: the year's last edition plus its mid-year one
        # leave no gaps. The running year takes its latest only.
        pick = {links[-1]} if y == this_year else {links[-1], links[len(links) // 2]}
        for u in pick:
            _get(BDI_BASE + u, d / f"{y}_{Path(u).name}", refresh=False, pause=0.5)
    return sorted(d.glob("*.pdf"))


def fetch_bdi(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tavola 1 / Tavola 2 monthly frames; per month the edition with the latest data wins."""
    t1s, t2s = [], []
    for f in _bdi_editions(refresh):
        try:
            a, b = parse_bdi_pdf(f)
        except Exception:  # noqa: BLE001 — a malformed edition must not sink the rest
            continue
        vint = max([a["month"].max() if len(a) else pd.NaT, b["month"].max() if len(b) else pd.NaT],
                   default=pd.NaT)
        for x, acc in ((a, t1s), (b, t2s)):
            if len(x):
                acc.append(x.assign(vintage=vint, file=f.name))

    def _latest(frames):
        if not frames:
            return pd.DataFrame()
        z = pd.concat(frames, ignore_index=True).sort_values(["month", "vintage"])
        return z.drop_duplicates("month", keep="last").set_index("month").sort_index()

    return _latest(t1s), _latest(t2s)


# --------------------------------------------------------------------------------------
# MEF state-sector press releases

_SLUG = re.compile(r"fabbisogno|avanzo|saldo|settore-statale|settore_statale|disavanzo", re.I)
_SLUG_SKIP = re.compile(r"entrate|tributar|partite-iva|rapporto|studi-di-settore|privatizz|btp|bot-|asta|emission",
                        re.I)
_AMT = r"(\d{1,3}(?:\.\d{3})+|\d+(?:,\d+)?)\s*(milioni|miliardi|mln|mld)"
_VAL = re.compile(r"(fabbisogno|disavanzo|avanzo|disponibilit[àa])[^\d]{0,140}?" + _AMT, re.I)
_CMP_STRICT = re.compile(r"(?:a fronte d[^.]{0,80}?|si confronta con\s+|che si era chiuso con\s+)(?:un |una )?"
                         r"(fabbisogno|avanzo|disavanzo)"
                         r"\s+(?:di|pari a)\s+(?:circa\s+)?" + _AMT, re.I)


def _amount(num: str, unit: str) -> float:
    v = it_number(num)
    return v * 1000.0 if unit.lower().startswith(("miliard", "mld")) else v


def _sign(word: str) -> float:
    w = word.lower()
    return -1.0 if w.startswith(("avanzo", "disponib")) else 1.0


def _infer_month(name: str, year: int | None, pub: pd.Timestamp) -> pd.Timestamp:
    mo = MONTH_NAMES[name.lower()]
    if year:
        return pd.Timestamp(year, mo, 1)
    y = pub.year if mo < pub.month else pub.year - 1
    return pd.Timestamp(y, mo, 1)


def parse_mef_release(text: str) -> list[dict]:
    """State-sector balance(s) in one MEF release; fabbisogno > 0, avanzo < 0, EUR mn.

    Returns records with kind 'provisional' (the month reported), 'prior_year' (the same month a year
    earlier, restated final) or 'final' (the older 'dati di sintesi' table releases).
    """
    m = re.search(r"Comunicato Stampa N\S*\s*\d+\s+del\s+(\d{2})/(\d{2})/(\d{4})", text)
    if not m:
        m = re.search(r"Roma,?\s+(\d{2})/(\d{2})/(\d{4})", text)
    if not m:
        return []
    pub = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    body = text[m.end():]
    body = body.split("Questa sezione rappresenta")[0]
    out = []
    # Older 'dati di sintesi' tables: 'Fabbisogno (-) / Disponibilità (+) 1.803'.
    t = re.search(r"Fabbisogno\s*\(-\)\s*/\s*Disponibilit\S*\s*\(\+\)\s*(-?\d{1,3}(?:\.\d{3})*)", body)
    if t:
        mm = re.search(r"mese di (" + "|".join(MONTH_NAMES) + r")\s*(\d{4})?", text, re.I)
        if mm:
            month = _infer_month(mm.group(1), int(mm.group(2)) if mm.group(2) else None, pub)
            out.append({"pub_date": pub, "month": month, "fabb_mn": -it_number(t.group(1)), "kind": "final"})
        return out
    # The provisional monthly print is always flagged 'in via provvisoria'; releases also quote
    # year-to-date totals ('A tutto novembre ... 69.300 milioni'), so the flag anchors the number.
    # Month: the one before publication (two before at the latest), named near the flag.
    lo, hi = pub.replace(day=1) - pd.DateOffset(months=2), pub.replace(day=1) - pd.DateOffset(months=1)
    p = None
    for anchor in (r"in via provvisori\w*", r"prime valutazioni", r"stima del fabbisogno",
                   r"[Nn]el mese di (?:" + "|".join(MONTH_NAMES) + r")"):
        p = re.search(anchor, body, re.I)
        if p:
            break
    if not p:
        return out
    a = re.compile(_AMT, re.I).search(body, p.end(), p.end() + 120)
    if not a:
        return out
    kws = list(re.finditer(r"fabbisogno|disavanzo|avanzo|disponibilit[àa]", body[max(0, p.start() - 220):a.start()], re.I))
    if not kws:
        return out
    month = None
    for cand in re.finditer(r"\b(" + "|".join(MONTH_NAMES) + r")\b\s*(?:del\s+)?(\d{4})?",
                            body[max(0, p.start() - 260):a.start()], re.I):
        mo = _infer_month(cand.group(1), int(cand.group(2)) if cand.group(2) else None, pub)
        if lo <= mo <= hi:
            month = mo
    month = month if month is not None else hi
    out.append({"pub_date": pub, "month": month, "kind": "provisional",
                "fabb_mn": _sign(kws[-1].group(0)) * _amount(a.group(1), a.group(2))})
    seg = body[p.start():]
    v = re.compile(_AMT, re.I).search(seg)
    # Year-ago restatement, only in the explicit form 'a fronte di un agosto 2025 che si era chiuso con
    # un fabbisogno di 204 milioni' / 'che si confronta con un fabbisogno di 2.412 milioni'. Bare or
    # bracketed numbers ('(-19.942 milioni)') carry inconsistent signs across years and are skipped.
    sentences = re.split(r"\.\s+(?=[A-Z])", seg[v.end():], maxsplit=2)
    w = _CMP_STRICT.search(sentences[0])
    if w is None and len(sentences) > 1 and re.search(r"corrispondente mese", sentences[1]):
        w = _CMP_STRICT.search(sentences[1])
    if w:
        out.append({"pub_date": pub, "month": month - pd.DateOffset(years=1),
                    "fabb_mn": _sign(w.group(1)) * _amount(w.group(2), w.group(3)), "kind": "prior_year"})
    return out


def fetch_mef_fabbisogno(refresh: bool = False, start_year: int = 2010) -> pd.DataFrame:
    """All parsed state-sector releases (one row per release × month)."""
    d = RAW / "mef_comunicati"
    d.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.today().year
    recs = []
    for y in range(start_year, this_year + 1):
        idx = _get(MEF_INDEX.format(y=y), RAW / "index" / f"mef_{y}.html", refresh=refresh and y >= this_year - 1)
        if idx is None:
            continue
        slugs = sorted(set(re.findall(rf'href="/ufficio-stampa/comunicati/{y}/([^"/]+)/"',
                                      idx.read_text(encoding="utf-8", errors="ignore"))))
        for slug in slugs:
            if not _SLUG.search(slug) or _SLUG_SKIP.search(slug):
                continue
            p = _get(f"{MEF_BASE}/ufficio-stampa/comunicati/{y}/{slug}/", d / f"{y}_{slug[:150]}.html", pause=0.2)
            if p is None:
                continue
            for r in parse_mef_release(_text_html(p.read_text(encoding="utf-8", errors="ignore"))):
                recs.append({**r, "url": f"{MEF_BASE}/ufficio-stampa/comunicati/{y}/{slug}/"})
    return pd.DataFrame(recs, columns=["pub_date", "month", "fabb_mn", "kind", "url"])


def mef_monthly(rel: pd.DataFrame) -> pd.DataFrame:
    """Per month: first provisional print and the latest final restatement (EUR mn).

    ``fabb_mn`` (best available: final, else provisional) is the history; ``fabb_rt_mn`` (first
    provisional print, else final) is what the backtest scores against.
    """
    if rel.empty:
        return pd.DataFrame(columns=["provisional_mn", "final_mn", "fabb_mn", "fabb_rt_mn", "provisional_pub"])
    prov = rel[rel["kind"] == "provisional"].sort_values("pub_date").drop_duplicates("month", keep="first")
    fin = rel[rel["kind"].isin(["final", "prior_year"])].sort_values("pub_date").drop_duplicates("month", keep="last")
    out = pd.DataFrame({"provisional_mn": prov.set_index("month")["fabb_mn"],
                        "provisional_pub": prov.set_index("month")["pub_date"]})
    out = out.join(fin.set_index("month")["fabb_mn"].rename("final_mn"), how="outer")
    # Parse guard: MEF revisions of the provisional print are a few hundred million; a 'final' several
    # billion away is a year-to-date figure caught by the parser, not a revision.
    bad = (out["final_mn"] - out["provisional_mn"]).abs() > MAX_REVISION_MN
    out.loc[bad, "final_mn"] = float("nan")
    out.attrs["dropped_finals"] = int(bad.sum())
    out["fabb_mn"] = out["final_mn"].fillna(out["provisional_mn"])
    out["fabb_rt_mn"] = out["provisional_mn"].fillna(out["final_mn"])
    return out.sort_index()


MAX_REVISION_MN = 3000.0
