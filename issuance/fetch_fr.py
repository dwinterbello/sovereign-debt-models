"""French AFT auction / syndication history. Parses official hist_mlt / hist_btf xlsx.

Auction amounts are the history files' totals incl. non-competitive bids wherever the history has the
auction. OTC buybacks come from the AFT monthly operations recap (Jan-2024 →) as negative ``Buyback`` rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urljoin

import pandas as pd
from bs4 import BeautifulSoup

from issuance.fr_latest import parse_aft_tables
from issuance.fr_text import parse_coupon, parse_french_maturity, scale_price, scale_yield
from issuance.http import FetchError, get_bytes, get_text, session
from issuance.paths import CACHE, HERE, RAW
from issuance.schema import coerce_ops
from issuance.tenor import attach_tenor

AFT = "https://www.aft.gouv.fr"
PAGES = (
    "/en/principaux-chiffres-oat",
    "/en/btf-principaux-chiffres",
    "/fr/principaux-chiffres-oat",
    "/fr/btf-principaux-chiffres",
)
FILE_HINTS = ("hist_mlt", "hist_btf", "syndication", "historique")
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_LATEST = "www.aft.gouv.fr/en/dernieres-adjudications"
WAYBACK_SYND = (
    "https://web.archive.org/web/20260731095402id_/"
    "https://www.aft.gouv.fr/files/medias-aft/3_Dette/3.2_OATMLT/"
    "3.2.1_OAT/Principaux%20chiffres/1999-2026_historique_syndications.xlsx"
)
PRESS_SLUGS = (
    "20260108-emission-oat",
    "20260122-next-issuance-inflation-indexed-oats",
    "20260122-issuance-oats",
    "20260219-issuance-oats",
    "20260319-issuance-oats",
    "20260402-issuance-oats",
    "20260416-issuance-oats",
    "20260416-next-issuance-inflation-indexed-oats",
    "20260507-issuance-oats",
    "20260521-issuance-oats",
    "20260521-next-issuance-inflation-indexed-oats",
    "20260702-issuance-oats",
    "20260716-issuance-oats",
)
RECAP_PAGE = "/fr/recapitulatif-mensuel-operations"
# AFT medium/long-term programme net of buybacks, EUR mn (programme indicatif, kept through the year).
# Later years come from sovereign_cash/data/fr/plan.json.
AFT_NET_PROGRAMME_MN = {2024: 285_000.0, 2025: 300_000.0}


def _sec_type(name: object, auction_type: object = None) -> str:
    text = f"{name} {auction_type}".lower()
    if "oati" in text or "oat€" in text or "oatei" in text or "adju_i" in text or "synd_i" in text:
        return "Linker"
    if "btf" in text:
        return "Bill"
    return "Nominal"


def _isin_start(raw: pd.DataFrame, col: int) -> int:
    for i in range(min(8, len(raw))):
        cell = str(raw.iloc[i, col]) if raw.shape[1] > col else ""
        if cell.startswith("FR") and len(cell) >= 12:
            return i
    return 1


def _download_key_figures(force: bool = False) -> list[Path]:
    dest_dir = RAW / "aft" / "KeyFigures"
    dest_dir.mkdir(parents=True, exist_ok=True)
    found = list(dest_dir.rglob("*.xlsx")) + list((RAW / "aft").rglob("*.xlsx"))
    have_hist = any("hist_" in p.name.lower() or "syndic" in p.name.lower() for p in found)
    if have_hist and not force:
        return found
    for page in PAGES:
        try:
            html = get_text(urljoin(AFT, page), timeout=60, impersonate=True)
        except FetchError:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            name = href.split("?")[0].split("/")[-1].lower()
            if not name.endswith((".xlsx", ".xls")):
                continue
            if not any(h in name or h in href.lower() for h in FILE_HINTS):
                if "chiffre" not in page and "key-figure" not in page:
                    continue
            url = urljoin(AFT, href)
            if "btf" in name:
                dest = dest_dir / "BTF" / name
            elif "syndic" in name:
                dest = dest_dir / "OAT" / name
            else:
                dest = dest_dir / "OAT" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                get_bytes(url, dest=dest, timeout=120, impersonate=True)
                found.append(dest)
            except FetchError:
                continue
    return found


def _download_live_page() -> Path | None:
    """Today's AFT results page. It only lists the last few weeks, so one copy per day accumulates."""
    dest = _latest_dir() / f"{pd.Timestamp.now():%Y%m%d}235959_dernieres-adjudications.html"
    try:
        get_bytes(urljoin(AFT, "/en/dernieres-adjudications"), dest=dest, timeout=60,
                  min_bytes=20_000, impersonate=True)
    except FetchError as exc:
        print(f"  FR live page skip: {exc}")
        return None
    return dest


def _latest(paths: list[Path], needle: str) -> Path | None:
    hits = [p for p in paths if needle in p.name.lower()]
    if not hits:
        return None
    return max(hits, key=lambda p: p.name)


def _load_mlt(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    start = _isin_start(raw, 5)
    body = raw.iloc[start:]
    body = body[body.iloc[:, 5].astype(str).str.startswith("FR")]
    served = pd.to_numeric(body.iloc[:, 8], errors="coerce")
    total = pd.to_numeric(body.iloc[:, 11], errors="coerce") if body.shape[1] > 11 else pd.Series(pd.NA, index=body.index)
    desc = body.iloc[:, 6]
    out = pd.DataFrame({
        "op_dt": pd.to_datetime(body.iloc[:, 1], errors="coerce"),
        "issue_dt": pd.to_datetime(body.iloc[:, 2], errors="coerce"),
        "isin": body.iloc[:, 5].astype(str).str.strip(),
        "sec_desc_long": desc,
        "amount_bid": pd.to_numeric(body.iloc[:, 7], errors="coerce"),
        "amount_issued": total.fillna(served),
        "btc": pd.to_numeric(body.iloc[:, 9], errors="coerce"),
        "yield": scale_yield(body.iloc[:, 12]) if body.shape[1] > 12 else pd.NA,
        "price": scale_price(body.iloc[:, 13]) if body.shape[1] > 13 else pd.NA,
        "coupon": desc.map(parse_coupon),
        "maturity_dt": desc.map(parse_french_maturity),
        "op_type": "Auction",
        "sec_type": [_sec_type(d, t) for d, t in zip(desc, body.iloc[:, 0])],
        "source": f"AFT {path.name}",
    })
    return out


def _load_btf(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    start = _isin_start(raw, 6)
    body = raw.iloc[start:]
    body = body[body.iloc[:, 6].astype(str).str.startswith("FR")]
    served = pd.to_numeric(body.iloc[:, 8], errors="coerce")
    total = pd.to_numeric(body.iloc[:, 11], errors="coerce") if body.shape[1] > 11 else pd.Series(pd.NA, index=body.index)
    out = pd.DataFrame({
        "op_dt": pd.to_datetime(body.iloc[:, 0], errors="coerce"),
        "issue_dt": pd.to_datetime(body.iloc[:, 1], errors="coerce"),
        "maturity_dt": pd.to_datetime(body.iloc[:, 3], errors="coerce"),
        "isin": body.iloc[:, 6].astype(str).str.strip(),
        "sec_desc_long": body.iloc[:, 2].map(lambda w: f"BTF {w}w"),
        "amount_bid": pd.to_numeric(body.iloc[:, 7], errors="coerce"),
        "amount_issued": total.fillna(served),
        "btc": pd.to_numeric(body.iloc[:, 9], errors="coerce"),
        "yield": scale_yield(body.iloc[:, 12]) if body.shape[1] > 12 else pd.NA,
        "coupon": 0.0,
        "op_type": "Auction",
        "sec_type": "Bill",
        "source": f"AFT {path.name}",
    })
    return out


def _load_synd(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    start = _isin_start(raw, 5)
    body = raw.iloc[start:]
    body = body[body.iloc[:, 5].astype(str).str.startswith("FR")]
    desc = body.iloc[:, 6]
    issued = pd.to_numeric(body.iloc[:, 11], errors="coerce")
    issue_dt = pd.to_datetime(body.iloc[:, 2], errors="coerce")
    out = pd.DataFrame({
        "op_dt": issue_dt,
        "issue_dt": issue_dt,
        "isin": body.iloc[:, 5].astype(str).str.strip(),
        "sec_desc_long": desc,
        "amount_issued": issued,
        "yield": scale_yield(body.iloc[:, 12]) if body.shape[1] > 12 else pd.NA,
        "price": scale_price(body.iloc[:, 13]) if body.shape[1] > 13 else pd.NA,
        "coupon": desc.map(parse_coupon),
        "maturity_dt": desc.map(parse_french_maturity),
        "op_type": issued.map(lambda x: "Buyback" if pd.notna(x) and x < 0 else "Syndication"),
        "sec_type": [_sec_type(d, t) for d, t in zip(desc, body.iloc[:, 0])],
        "source": f"AFT {path.name}",
    })
    return out


def _wayback_id(url: str, ts: str) -> str:
    return f"https://web.archive.org/web/{ts}id_/{url}"


def _download_wayback_synd(force: bool = False) -> Path | None:
    dest = RAW / "aft" / "KeyFigures" / "OAT" / "1999-2026_historique_syndications.xlsx"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    try:
        get_bytes(WAYBACK_SYND, dest=dest, timeout=90)
        return dest
    except FetchError as exc:
        print(f"  FR Wayback synd skip: {exc}")
        return dest if dest.exists() else None


def _latest_dir() -> Path:
    dest = RAW / "aft" / "LatestAuctions"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _download_latest_html(force: bool = False) -> list[Path]:
    dest_dir = _latest_dir()
    found = list(dest_dir.glob("*.html"))
    if found and not force:
        return found
    sess = session()
    try:
        cdx = sess.get(
            WAYBACK_CDX,
            params={
                "url": WAYBACK_LATEST,
                "output": "json",
                "fl": "timestamp,original,statuscode",
                "filter": "statuscode:200",
                "from": "202512",
                "limit": 40,
            },
            timeout=45,
        )
        cdx.raise_for_status()
        rows = cdx.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  FR Wayback CDX skip: {exc}")
        return found
    for row in rows[1:]:
        ts, original = str(row[0]), str(row[1])
        dest = dest_dir / f"{ts}_dernieres-adjudications.html"
        if dest.exists() and dest.stat().st_size > 2000:
            found.append(dest)
            continue
        try:
            get_bytes(_wayback_id(original, ts), dest=dest, timeout=60)
            found.append(dest)
        except FetchError as exc:
            print(f"  FR latest {ts} skip: {exc}")
    return list({p.resolve(): p for p in found}.values())


def _download_press_html(force: bool = False) -> list[Path]:
    dest_dir = _latest_dir() / "press"
    dest_dir.mkdir(parents=True, exist_ok=True)
    found = list(dest_dir.glob("*.html"))
    # A fixed list of past communiqués; the live page covers new auctions.
    if found:
        return found
    # Tracked communiqué JSON already covers the Cloudflare gap; skip a slow CDX sweep.
    if not force and (HERE / "data" / "fr_2026_press_auctions.json").exists():
        return found
    sess = session()
    for slug in PRESS_SLUGS:
        dest = dest_dir / f"{slug}.html"
        if dest.exists() and dest.stat().st_size > 500:
            found.append(dest)
            continue
        for lang in ("en", "fr"):
            live = f"https://www.aft.gouv.fr/{lang}/publications/communiques-presse/{slug}"
            try:
                cdx = sess.get(
                    WAYBACK_CDX,
                    params={
                        "url": live,
                        "output": "json",
                        "fl": "timestamp,original,statuscode",
                        "filter": "statuscode:200",
                        "limit": 3,
                    },
                    timeout=20,
                )
                payload = cdx.json() if cdx.ok else []
            except Exception:
                payload = []
            saved = False
            for row in payload[1:]:
                try:
                    get_bytes(_wayback_id(str(row[1]), str(row[0])), dest=dest, timeout=40)
                    found.append(dest)
                    saved = True
                    break
                except FetchError:
                    continue
            if saved:
                break
    return list({p.resolve(): p for p in found}.values())


def _from_latest_html(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for path in paths:
        try:
            html = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        part = parse_aft_tables(html, source=f"AFT {path.name}")
        if not part.empty:
            parts.append(part)
            print(f"  FR latest {path.name}: {len(part)} rows")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _from_press_csv() -> pd.DataFrame:
    path = HERE / "data" / "fr_2026_press_auctions.json"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_json(path)
    if raw.empty:
        return pd.DataFrame()
    raw["op_dt"] = pd.to_datetime(raw["op_dt"], errors="coerce")
    raw["issue_dt"] = pd.to_datetime(raw.get("issue_dt"), errors="coerce")
    raw["maturity_dt"] = raw["sec_desc_long"].map(parse_french_maturity)
    raw["coupon"] = raw["sec_desc_long"].map(parse_coupon)
    raw["op_type"] = "Auction"
    raw["sec_type"] = [_sec_type(d) for d in raw["sec_desc_long"]]
    raw["source"] = raw.get("source", "AFT communiqué")
    print(f"  FR press JSON: {len(raw)} rows")
    return raw


def _recap_dir() -> Path:
    dest = RAW / "aft" / "MonthlyRecap"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _download_recaps(force: bool = False) -> list[Path]:
    """AFT 'Récapitulatif mensuel des opérations' PDFs (Jan-2024 →), saved as YYYY-MM_operations_mensuelles.pdf."""
    dest_dir = _recap_dir()
    have = sorted(dest_dir.glob("*_operations_mensuelles.pdf"))
    if have and not force:
        return have
    try:
        html = get_text(urljoin(AFT, RECAP_PAGE), timeout=60, impersonate=True)
    except FetchError as exc:
        # aft.gouv.fr sits behind a Cloudflare challenge (403 to scripts): read the page's
        # latest Wayback capture instead, and fetch each PDF from Wayback when direct fails.
        html = _wayback_latest_text(urljoin(AFT, RECAP_PAGE))
        if html is None:
            print(f"  FR recap page skip: {exc} (no Wayback capture)")
            return have
        print("  FR recap page via Wayback (live page 403)")
    for href in sorted(set(re.findall(r'href="([^"]+\.pdf)"', html))):
        href = re.sub(r"^https?://web\.archive\.org/web/\d+(?:id_)?/", "", href)
        name = unquote(href.rsplit("/", 1)[-1])
        m = re.match(r"(\d{2})(\d{2})[ _]op", name)
        if "Recap" not in unquote(href) or not m:
            continue
        dest = dest_dir / f"20{m.group(2)}-{m.group(1)}_operations_mensuelles.pdf"
        if dest.exists() and dest.stat().st_size > 2000:
            continue
        url = urljoin(AFT, href)
        try:
            get_bytes(url, dest=dest, timeout=60, min_bytes=2000, impersonate=True)
        except FetchError as exc:
            if not _wayback_latest_bytes(url, dest):
                print(f"  FR recap {name} skip: {exc} (no Wayback capture)")
    return sorted(dest_dir.glob("*_operations_mensuelles.pdf"))


def _wayback_latest_ts(url: str) -> str | None:
    try:
        cdx = session().get(WAYBACK_CDX, params={
            "url": url, "output": "json", "fl": "timestamp,statuscode",
            "filter": "statuscode:200", "limit": -1}, timeout=45)
        rows = cdx.json() if cdx.ok else []
    except Exception:  # noqa: BLE001
        return None
    return str(rows[-1][0]) if len(rows) > 1 else None


def _wayback_latest_text(url: str) -> str | None:
    ts = _wayback_latest_ts(url)
    if ts is None:
        return None
    try:
        return get_text(_wayback_id(url, ts), timeout=60)
    except FetchError:
        return None


def _wayback_latest_bytes(url: str, dest: Path) -> bool:
    ts = _wayback_latest_ts(url)
    if ts is None:
        return False
    try:
        get_bytes(_wayback_id(url, ts), dest=dest, timeout=60, min_bytes=2000)
        return True
    except FetchError:
        return False


_RECAP_BB_TOTAL = re.compile(r"rachats de gr[ée] [àa] gr[ée] pour le compte de l.[ÉE]tat\s*:\s*([\d\s.,]+?)\s*(M€|€)?\s*$",
                             re.M | re.I)
_RECAP_BB_LINE = re.compile(r"^(OAT€i|OATei|OATi|OAT)\s+(\d+[.,]\d+)\s*%\s+(\d{2}/\d{2}/\d{4})\s+(\d[\d\s.,]*?)\s*$")
_RECAP_BB_BTF = re.compile(r"^BTF\b.*\)\s+(\d[\d\s.,]*?)\s*$")
_RECAP_TOTAL_OAT = re.compile(r"^TOTAL OAT\s+(\d[\d\s.,]*?)\s*$", re.M)


def _recap_num(text: str) -> float:
    t = text.replace("\xa0", "").replace(" ", "")
    # AFT mixes "1 330,00" (decimal comma) and "41,420" (thousands comma) in the same series.
    t = t.replace(",", ".") if re.search(r",\d{2}$", t) else t.replace(",", "")
    return float(t)


def parse_recap(path: Path) -> dict:
    """One monthly recap: OTC buybacks by line (EUR mn) and the month's TOTAL OAT issued.

    The section header total is checked against the line sum (BTF lines included); ``ok`` is
    False when they differ by more than EUR 1mn.
    """
    import pdfplumber

    m = re.match(r"(\d{4})-(\d{2})", path.name)
    with pdfplumber.open(path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    return recap_from_text(text, pd.Timestamp(int(m.group(1)), int(m.group(2)), 1), path.name)


def recap_from_text(text: str, month: pd.Timestamp, name: str = "") -> dict:
    tot = _RECAP_TOTAL_OAT.search(text)
    head = _RECAP_BB_TOTAL.search(text)
    unit = 1e-6 if head and head.group(2) == "€" else 1.0
    header_mn = _recap_num(head.group(1)) * unit if head else float("nan")
    start = head.end() if head else text.find("Rachats de gr")
    stop = text.find("Pensions livr", max(start, 0))
    section = text[start:stop] if start >= 0 and stop > start else ""
    lines, btf_mn = [], 0.0
    for raw in section.splitlines():
        raw = raw.strip()
        b = _RECAP_BB_BTF.match(raw)
        if b:
            btf_mn += _recap_num(b.group(1)) * unit
            continue
        x = _RECAP_BB_LINE.match(raw)
        if not x:
            continue
        kind = x.group(1).replace("OATei", "OAT€i")
        coupon = float(x.group(2).replace(",", "."))
        lines.append({"kind": kind, "coupon": coupon, "maturity_dt": pd.to_datetime(x.group(3), format="%d/%m/%Y"),
                      "nominal_mn": _recap_num(x.group(4)) * unit,
                      "label": f"{kind} {x.group(2).replace('.', ',')}% {x.group(3)}"})
    line_sum = sum(r["nominal_mn"] for r in lines) + btf_mn
    return {"month": month, "file": name, "lines": lines, "btf_mn": btf_mn, "header_mn": header_mn,
            "total_oat_mn": _recap_num(tot.group(1)) if tot else float("nan"),
            "ok": head is not None and abs(line_sum - header_mn) <= 1.0}


def load_recaps(paths: list[Path] | None = None) -> list[dict]:
    paths = sorted(_recap_dir().glob("*_operations_mensuelles.pdf")) if paths is None else paths
    return [parse_recap(p) for p in paths]


def _buyback_ops(recaps: list[dict], ops: pd.DataFrame) -> pd.DataFrame:
    """OTC buybacks as negative ``Buyback`` rows, ISIN matched on kind × coupon × maturity from the tape.

    The recap gives no trade dates, so each month's buybacks are dated on its last business day.
    """
    known = ops[ops["isin"].astype(str).str.startswith("FR") & ops["maturity_dt"].notna()].copy()
    known["_lk"] = known["sec_type"].astype(str).eq("Linker")
    key = known.assign(_c=pd.to_numeric(known["coupon"], errors="coerce").round(3),
                       _m=pd.to_datetime(known["maturity_dt"]).dt.normalize())
    isin_of = key.drop_duplicates(["_lk", "_c", "_m"]).set_index(["_lk", "_c", "_m"])["isin"].to_dict()
    rows = []
    for rc in recaps:
        if not rc["ok"]:
            print(f"  FR recap {rc['file']}: buyback lines {sum(r['nominal_mn'] for r in rc['lines']):.1f} "
                  f"≠ header {rc['header_mn']:.1f} EUR mn — month skipped")
            continue
        day = rc["month"] + pd.offsets.BMonthEnd(0)
        for r in rc["lines"]:
            if r["nominal_mn"] <= 0:
                continue
            linker = r["kind"] != "OAT"
            isin = isin_of.get((linker, round(r["coupon"], 3), r["maturity_dt"].normalize()))
            rows.append({
                "op_dt": day, "issue_dt": day, "maturity_dt": r["maturity_dt"], "isin": isin,
                "sec_desc_long": r["label"], "amount_issued": -r["nominal_mn"], "coupon": r["coupon"],
                "op_type": "Buyback", "sec_type": "Linker" if linker else "Nominal",
                "source": f"AFT recap {rc['file']}",
            })
    out = pd.DataFrame(rows)
    if len(out):
        miss = out["isin"].isna()
        if miss.any():
            print(f"  FR buybacks: {int(miss.sum())} lines without an ISIN on the tape "
                  f"({-out.loc[miss, 'amount_issued'].sum():,.0f} EUR mn)")
        print(f"  FR buybacks: {len(out)} rows, {-out['amount_issued'].sum() / 1000:,.1f} EUR bn "
              f"({out['op_dt'].min():%Y-%m} → {out['op_dt'].max():%Y-%m})")
    return out


def _hist_totals(ops: pd.DataFrame, hist: list[pd.DataFrame]) -> pd.DataFrame:
    """Auction amounts set to the AFT history total (competitive + non-competitive) where it has the auction.

    Communiqués and the results page print the competitive amount; ONC lands two days later and is only
    in hist_mlt / hist_btf. ``attrs['onc_added_mn']`` holds the per-year uplift.
    """
    frames = [h for h in hist if h is not None and not h.empty]
    ops.attrs["onc_added_mn"] = {}
    if not frames:
        return ops
    h = pd.concat(frames, ignore_index=True)
    tot = h.groupby([h["isin"].astype(str).str.upper(), pd.to_datetime(h["op_dt"]).dt.normalize()])["amount_issued"].sum()
    auc = ops["op_type"].astype(str).eq("Auction")
    keys = pd.MultiIndex.from_arrays([ops.loc[auc, "isin"].astype(str).str.upper(),
                                      pd.to_datetime(ops.loc[auc, "op_dt"]).dt.normalize()])
    new = pd.Series(tot.reindex(keys).to_numpy(), index=ops.index[auc])
    upd = new.notna() & (new - ops.loc[auc, "amount_issued"]).abs().gt(0.5)
    diff = (new - ops.loc[auc, "amount_issued"])[upd]
    ops.loc[diff.index, "amount_issued"] = new[upd]
    yr = pd.to_datetime(ops.loc[diff.index, "op_dt"]).dt.year
    ops.attrs["onc_added_mn"] = {int(y): float(v) for y, v in diff.groupby(yr).sum().items()}
    if len(diff):
        print(f"  FR hist totals (incl. ONC): {len(diff)} auctions, "
              + ", ".join(f"{y} {v / 1000:+.2f}bn" for y, v in ops.attrs["onc_added_mn"].items()))
    return ops


def annual_vs_aft(ops: pd.DataFrame, recaps: list[dict] | None = None) -> pd.DataFrame:
    """Calendar-year OAT gross / buybacks / net on the tape vs AFT recap totals and the net programme (EUR bn)."""
    import json

    recaps = load_recaps() if recaps is None else recaps
    o = ops[ops["sec_type"].astype(str).ne("Bill")]
    yr = pd.to_datetime(o["op_dt"]).dt.year
    gross = o[o["amount_issued"] > 0].groupby(yr)["amount_issued"].sum() / 1000
    bb = -o[o["op_type"].astype(str).eq("Buyback")].groupby(yr)["amount_issued"].sum() / 1000
    rc = pd.DataFrame([{"year": r["month"].year, "month": r["month"], "total_oat": r["total_oat_mn"] / 1000,
                        "bb": r["header_mn"] / 1000 - r["btf_mn"] / 1000} for r in recaps])
    prog = dict(AFT_NET_PROGRAMME_MN)
    plan = HERE.parent / "sovereign_cash" / "data" / "fr" / "plan.json"
    if plan.exists():
        for y, p in json.loads(plan.read_text(encoding="utf-8"))["years"].items():
            prog.setdefault(int(y), float(p["net_mlt_issuance"]) * 1000)
    years = sorted(set(rc["year"])) if len(rc) else sorted(set(yr))
    out = []
    for y in years:
        r = rc[rc["year"] == y] if len(rc) else rc
        last = r["month"].max() if len(r) else pd.NaT
        cut = (last + pd.offsets.MonthEnd(0)) if pd.notna(last) else pd.Timestamp(y, 12, 31)
        in_y = (yr == y) & (pd.to_datetime(o["op_dt"]) <= cut)
        g = float(o.loc[in_y & (o["amount_issued"] > 0), "amount_issued"].sum()) / 1000
        b = -float(o.loc[in_y & o["op_type"].astype(str).eq("Buyback"), "amount_issued"].sum()) / 1000
        out.append({"year": y, "through": f"{cut:%Y-%m}", "recap_months": len(r),
                    "gross_tape_bn": g, "gross_aft_recap_bn": float(r["total_oat"].sum()) if len(r) else float("nan"),
                    "buybacks_tape_bn": b, "buybacks_aft_recap_bn": float(r["bb"].sum()) if len(r) else float("nan"),
                    "net_tape_bn": g - b, "programme_bn": prog.get(y, float("nan")) / 1000,
                    "gross_tape_fy_bn": float(gross.get(y, 0.0)), "buybacks_tape_fy_bn": float(bb.get(y, 0.0))})
    return pd.DataFrame(out)


def _dedup_ops(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["_k"] = (
        out["isin"].astype(str).str.upper()
        + "|"
        + pd.to_datetime(out["op_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
    )
    # prefer later source vintage (Wayback / communiqué over stale hist)
    out["_rank"] = 0
    src = out["source"].astype(str)
    out.loc[src.str.contains("2026|latest|communiqué|Latest|dernieres", case=False, na=False), "_rank"] = 1
    out = out.sort_values(["_k", "_rank"]).drop_duplicates("_k", keep="last")
    return out.drop(columns=["_k", "_rank"])


# AFT posts each next-issuance a few days before the auction: ISIN, volume range, auction day.
OUT_ANN = CACHE / "fr_aft_announcements.parquet"
RAW_ANN = RAW / "aft" / "NextIssuance"
PRESS_INDEX = urljoin(AFT, "/en/communiques-de-presse")
_ANN_VOL = re.compile(r"<\s*([\d,.\s]+)\s*--\s*([\d,.\s]+)\s*>")
_ANN_ISIN = re.compile(r"\b(FR[0-9A-Z]{10})\b")
_ANN_DATE = re.compile(r"(\d{2}/\d{2}/\d{4})")
_ANN_KIND = re.compile(r"(?:^|\n)\s*(OAT€i|OATi|OAT|BTF)\b", re.I)


def _fr_ann_date(text: str) -> pd.Timestamp:
    d, m, y = text.strip().split("/")
    return pd.Timestamp(int(y), int(m), int(d))


def _fr_ann_amount(tok: str) -> float:
    return float(re.sub(r"[^\d.]", "", tok.replace(",", "")) or 0.0)


def _fr_ann_sec_type(kind: str) -> str:
    k = (kind or "").upper().replace("€", "E")
    if k == "BTF":
        return "Bill"
    if k in {"OATI", "OATEI"}:
        return "Linker"
    return "Nominal"


def parse_aft_next_issuance_text(text: str, *, source: str = "") -> pd.DataFrame:
    """Parse an AFT next-issuance communiqué body into one row per ISIN.

    OAT/OATi communiqués publish one volume range for the whole auction; BTF pages
    publish a range per line. Shared pools are split equally across ISINs (midpoint).
    """
    text = (text or "").replace("\xa0", " ").replace("\r", "")
    isins = _ANN_ISIN.findall(text)
    if not isins:
        return pd.DataFrame()
    vols = [( _fr_ann_amount(a), _fr_ann_amount(b)) for a, b in _ANN_VOL.findall(text)]
    head, _, rest = text.partition("Auction date")
    kinds = _ANN_KIND.findall(head) or _ANN_KIND.findall(text)
    while len(kinds) < len(isins):
        kinds.append(kinds[-1] if kinds else "OAT")
    kinds = kinds[: len(isins)]
    auction_dates = _ANN_DATE.findall(rest.split("Settlement date")[0] if "Settlement date" in rest else rest[:200])
    if not auction_dates:
        auction_dates = _ANN_DATE.findall(text)
    op = _fr_ann_date(auction_dates[0]) if auction_dates else pd.NaT
    if any(_fr_ann_sec_type(k) == "Bill" for k in kinds) and "Maturity" in rest:
        mat_block = rest.split("Maturity", 1)[1].split("Volume", 1)[0]
        mats = [_fr_ann_date(d) for d in _ANN_DATE.findall(mat_block)]
    else:
        mats = [_fr_ann_date(d) for d in _ANN_DATE.findall(head)]
    while len(mats) < len(isins):
        mats.append(pd.NaT)
    if len(vols) == len(isins):
        ranges = vols
    elif len(vols) == 1:
        lo, hi = vols[0]
        n = len(isins)
        ranges = [(lo / n, hi / n)] * n
    elif vols:
        # Fall back to equal split of the sum of published ranges.
        lo = sum(v[0] for v in vols) / len(isins)
        hi = sum(v[1] for v in vols) / len(isins)
        ranges = [(lo, hi)] * len(isins)
    else:
        ranges = [(float("nan"), float("nan"))] * len(isins)
    rows = []
    for i, isin in enumerate(isins):
        lo, hi = ranges[i]
        mid = (lo + hi) / 2.0 if pd.notna(lo) and pd.notna(hi) else float("nan")
        rows.append({
            "op_dt": op,
            "isin": isin.upper(),
            "maturity_dt": mats[i] if i < len(mats) else pd.NaT,
            "sec_type": _fr_ann_sec_type(kinds[i]),
            "amount_min_mn": lo,
            "amount_max_mn": hi,
            "amount_mn": mid,
            "source": source or "AFT next-issuance",
        })
    return pd.DataFrame(rows)


def _aft_next_issuance_urls(html: str) -> list[str]:
    found = []
    for a in BeautifulSoup(html or "", "html.parser").find_all("a", href=True):
        href = a["href"]
        if "next-issuance" not in href and "prochaine-emission" not in href:
            continue
        # Prefer English bodies (stable table shape); skip result "issuance" pages.
        if "/fr/" in href and "prochaine-emission" not in href:
            continue
        found.append(urljoin(AFT, href))
    return list(dict.fromkeys(found))


def fetch_aft_announcements(*, force: bool = False, limit: int = 40) -> pd.DataFrame:
    """Download recent AFT next-issuance communiqués → ``fr_aft_announcements.parquet``."""
    RAW_ANN.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    try:
        html = get_text(PRESS_INDEX, timeout=60, impersonate=True)
        urls = _aft_next_issuance_urls(html)[:limit]
    except FetchError as exc:
        print(f"  FR AFT press index: {exc}")
    for url in urls:
        name = url.rstrip("/").split("/")[-1] + ".html"
        dest = RAW_ANN / name
        if dest.exists() and not force and dest.stat().st_size > 1000:
            continue
        try:
            get_bytes(url, dest=dest, timeout=40, impersonate=True, min_bytes=1000)
        except FetchError as exc:
            print(f"  FR AFT next-issuance skip {name}: {exc}")
    parts = []
    for path in sorted(RAW_ANN.glob("*.html")):
        try:
            html = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("article") or soup.find("main") or soup
        block = parse_aft_next_issuance_text(body.get_text("\n", strip=True), source=f"AFT {path.stem}")
        if not block.empty:
            parts.append(block)
    if not parts:
        if OUT_ANN.exists():
            return pd.read_parquet(OUT_ANN)
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["op_dt"] = pd.to_datetime(out["op_dt"])
    out["maturity_dt"] = pd.to_datetime(out["maturity_dt"])
    out = out.drop_duplicates(subset=["op_dt", "isin"], keep="last").sort_values(["op_dt", "isin"])
    out.to_parquet(OUT_ANN, index=False)
    print(f"  FR AFT announcements: {len(out)} lines through {out['op_dt'].max().date()}")
    return out


def fetch_fr(force: bool = False) -> pd.DataFrame:
    files = _download_key_figures(force=force)
    if not files:
        files = list((RAW / "aft").rglob("*.xls*"))
    wb_synd = _download_wayback_synd(force=force)
    if wb_synd is not None:
        files = list(files) + [wb_synd]
    mlt = _latest(files, "hist_mlt")
    btf = _latest(files, "hist_btf")
    synd = _latest(files, "syndic")
    parts, hist = [], []
    if mlt is not None:
        parts.append(_load_mlt(mlt))
        hist.append(parts[-1])
        print(f"  FR MLT {mlt.name}: {len(parts[-1])} rows")
    if btf is not None:
        parts.append(_load_btf(btf))
        hist.append(parts[-1])
        print(f"  FR BTF {btf.name}: {len(parts[-1])} rows")
    if synd is not None:
        parts.append(_load_synd(synd))
        print(f"  FR syndications {synd.name}: {len(parts[-1])} rows")
    latest_html = _download_latest_html(force=force)
    if force:
        live = _download_live_page()
        if live is not None and live not in latest_html:
            latest_html = list(latest_html) + [live]
    latest = _from_latest_html(latest_html)
    if not latest.empty:
        parts.append(latest)
    press_html = _download_press_html(force=force)
    press = _from_latest_html(press_html)
    if not press.empty:
        parts.append(press)
    fallback = _from_press_csv()
    if not fallback.empty:
        parts.append(fallback)
    if not parts:
        raise FetchError("AFT hist_mlt / hist_btf / syndications xlsx not available")
    ops = pd.concat(parts, ignore_index=True)
    ops = _hist_totals(_dedup_ops(ops).reset_index(drop=True), hist)
    recaps = load_recaps(_download_recaps(force=force))
    bb = _buyback_ops(recaps, ops)
    if len(bb):
        # Month-end dating must not run past the tape: lines.py takes the tape's last op_dt as asof.
        bb["op_dt"] = bb["op_dt"].clip(upper=pd.to_datetime(ops["op_dt"]).max())
        bb["issue_dt"] = bb["op_dt"]
        ops = pd.concat([ops, bb], ignore_index=True)
    ops = coerce_ops(attach_tenor(ops), "FR")
    ops = ops[ops["op_dt"].notna() & (ops["amount_issued"].fillna(0) != 0)]
    ops.to_parquet(CACHE / "ops_fr.parquet", index=False)
    try:
        fetch_aft_announcements(force=force)
    except Exception as exc:  # noqa: BLE001 — announcements feed the TBA; tape still ships
        print(f"  FR AFT announcements failed: {exc}")
    return ops
