"""JSDA 'Trends in Bond Transactions (by investor type)' — JGB outright OTC trading by investor.

    PYTHONPATH=. .venv/bin/python -m issuance.fetch_jsda            # fetch missing files, print coverage
    PYTHONPATH=. .venv/bin/python -m issuance.fetch_jsda --refresh  # also re-pull the current-FY file

Sources (https://www.jsda.or.jp/shiryoshitsu/toukei/):
  toushika/tkb/files/kokusaiichiran.xls   国債投資家別売買高, 2004-04 … 2018-04, one wide sheet pair
                                          (investor blocks side by side, months down the rows).
  tentoubaibai/koushasai<FY>[r].xlsx      公社債店頭売買高, one file per fiscal year (April–March),
                                          FY2018 onward ('r' = revised); sheet (Ｅ)/(Ｈ)/(Ｋ) =
                                          all members, outright sales / purchases / net.
  tentoubaibai/koushasai.xlsx             the current fiscal year to the latest month.
JSDA rate-limits (nginx 429), so `fetch` downloads one file at a time with ≥20 s between requests and
falls back to the Wayback Machine (raw `id_` bytes) per file when JSDA refuses.

Content: face value, ¥100mn in the files (cached tidy panel in ¥bn). Outright trades only (現先 /
repo excluded) reported by securities companies and banks' dealing desks (JSDA regular + special
members), so these are investors' trades *with dealers*: a bank or insurer that buys at the auction
itself never appears, while dealers' own auction allotments show up as *sales* by 'Others'
(file note 3/4). JGB maturity split is by issue type, not residual maturity: super-long = coupon
bonds over 10y at issue (20/30/40y, 15y floaters), long = 10y, medium = 2–5y, zero = discount bonds
over 1y and STRIPS, tbills = T-bills/FBs (国庫短期証券等; before Feb-2009 割引短期国債 + 政府短期証券).

Sign: the files publish 差引 = sales − purchases (positive = net selling). `panel()` flips it:
net_purchase_jpy_bn = (purchases − sales) / 10, so positive = the investor bought on net.

Publication: monthly, mid-way through the following month (workbook timestamps: Aug-2026 data
14 Sep 2026; March data 16 Apr 2024, 16 Apr 2025, 15 Apr 2026), so month t is public during t+1.
Rows are stamped at month t+1 (`public_month`); `evidence`'s 1-month lag on top then starts the
price window (monthly-average yields) at t+2, so nothing is used before it was published.
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE

BASE = "https://www.jsda.or.jp/shiryoshitsu/toukei/"
REFERER = BASE + "tentoubaibai/index.html"
DIR = CACHE / "jsda"
OUT = CACHE / "jsda_investor.parquet"
SOURCE = "JSDA Trends in Bond Transactions by investor type (OTC, outright, face value)"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
SLEEP = 20.0
# file name → path under BASE. Ordered by priority when months overlap: earlier wins.
CURRENT = "koushasai.xlsx"
FILES: dict[str, str] = {
    CURRENT: "tentoubaibai/koushasai.xlsx",
    "koushasai2025.xlsx": "tentoubaibai/koushasai2025.xlsx",
    "koushasai2024.xlsx": "tentoubaibai/koushasai2024.xlsx",
    "koushasai2023.xlsx": "tentoubaibai/koushasai2023.xlsx",
    "koushasai2022r.xlsx": "tentoubaibai/koushasai2022r.xlsx",
    "koushasai2021r.xlsx": "tentoubaibai/koushasai2021r.xlsx",
    "koushasair2020.xlsx": "tentoubaibai/koushasair2020.xlsx",
    "koushasai2019r.xlsx": "tentoubaibai/koushasai2019r.xlsx",
    "koushasai2018r.xlsx": "tentoubaibai/koushasai2018r.xlsx",
    "kokusaiichiran.xls": "toushika/tkb/files/kokusaiichiran.xls",
}
# Wayback captures (CDX, 2026-10-02) used when JSDA refuses. Some captures are of a sibling name
# (pre-revision file, or 'koushasai2020r' for 'koushasair2020'); the content layout is the same.
WAYBACK: dict[str, tuple[str, str]] = {
    "koushasai2025.xlsx": ("20260616021408", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2025.xlsx"),
    "koushasai2024.xlsx": ("20260427025138", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2024.xlsx"),
    "koushasai2023.xlsx": ("20250121123431", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2023.xlsx"),
    "koushasai2022r.xlsx": ("20240406000631", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2022.xlsx"),
    "koushasai2021r.xlsx": ("20220610030300", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2021.xlsx"),
    "koushasair2020.xlsx": ("20220610030252", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2020r.xlsx"),
    "koushasai2019r.xlsx": ("20220822030541", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2019r.xlsx"),
    "koushasai2018r.xlsx": ("20220822031108", "https://www.jsda.or.jp/shiryoshitsu/toukei/tentoubaibai/koushasai2018r.xlsx"),
    "kokusaiichiran.xls": ("20190828084934", "http://www.jsda.or.jp:80/shiryoshitsu/toukei/toushika/tkb/files/kokusaiichiran.xls"),
}

INVESTORS = {  # key → (Japanese label stem, English label)
    "city_banks": ("都市銀行", "City banks & long-term credit banks"),
    "regional_banks": ("地方銀行", "Regional banks"),
    "trust_banks": ("信託銀行", "Trust banks (incl. pension trust accounts)"),
    "agri_fin": ("農林系金融機関", "Agricultural & forestry financial institutions"),
    "regional2": ("第二地銀", "Second-tier regional banks"),
    "shinkin": ("信用金庫", "Shinkin banks"),
    "other_fin": ("その他金融機関", "Other financial institutions"),
    "insurers": ("生保・損保", "Life & non-life insurers"),
    "inv_trusts": ("投資信託", "Investment trusts"),
    "mutual_aid": ("官公庁共済組合", "Public officers' mutual aid associations"),
    "business_corps": ("事業法人", "Business corporations"),
    "other_corps": ("その他法人", "Other corporations / legal entities"),
    "foreigners": ("外国人", "Foreigners"),
    "individuals": ("個人", "Individuals"),
    "others": ("その他", "Others (incl. dealers' auction allotments as sales)"),
    "dealers": ("債券ディーラー", "Bond dealers"),
    "total": ("合計", "Total"),
}
MATURITIES = ("super_long", "long", "medium", "zero", "tbills")
BUCKETS = ("total_jgb", *MATURITIES)
# Approximate modified duration per JSDA bucket (years), held fixed so DV01 does not move with the
# yield level: super-long ≈ a 20–30y par bond, long = 10y, medium = 2–5y, zero = discount bonds/STRIPS
# (mostly long STRIPS, rough), bills ≈ 3–6m. ¥bn × duration × 1e-4 = ¥bn per bp.
DURATION = {"super_long": 17.0, "long": 8.5, "medium": 3.5, "zero": 8.0, "tbills": 0.4}
BANKS = ("city_banks", "regional_banks", "regional2")
REAL_MONEY = ("insurers", "trust_banks", "inv_trusts", "mutual_aid")

_ZIP = b"PK\x03\x04"
_OLE = b"\xd0\xcf\x11\xe0"


class JsdaError(RuntimeError):
    pass


# ---------------------------------------------------------------- fetch
def valid_body(name: str, body: bytes) -> bool:
    """xlsx must be a zip, xls an OLE compound file; anything else (HTML, 429 page) is junk."""
    if name.lower().endswith(".xlsx"):
        return body[:4] == _ZIP and len(body) > 5000
    if name.lower().endswith(".xls"):
        return body[:4] == _OLE and len(body) > 5000
    return False


def _get(url: str, referer: str = REFERER, timeout: int = 120):
    import requests

    return requests.get(url, headers={"User-Agent": UA, "Referer": referer, "Accept": "*/*",
                                      "Accept-Language": "ja,en;q=0.8"}, timeout=timeout)


def _download(name: str, url: str, get, referer: str = REFERER) -> bytes:
    r = get(url, referer)
    body = r.content or b""
    if r.status_code == 429:
        raise JsdaError(f"JSDA rate limit (HTTP 429) on {url}; back off ≥10 minutes")
    if r.status_code != 200 or not valid_body(name, body):
        raise JsdaError(f"{url}: HTTP {r.status_code}, {len(body)} bytes, head {body[:8]!r} — not a workbook")
    return body


def fetch(refresh: bool = False, files: list[str] | None = None, wayback: bool = True,
          sleep: float = SLEEP, get=None, jsda: bool = True) -> dict[str, Path]:
    """Download missing files into `DIR` one at a time (≥`sleep` s between requests). A valid cached
    file is never re-downloaded, except the current-FY `koushasai.xlsx` when `refresh`. A JSDA 429
    stops further JSDA requests in this call; other files then go to the Wayback capture (if any).
    `jsda=False` skips JSDA entirely (Wayback only). Invalid bodies are never written. Returns
    name → path of every valid cached file."""
    get = _get if get is None else get
    DIR.mkdir(parents=True, exist_ok=True)
    names = list(FILES) if files is None else files
    blocked, first, errors = not jsda, True, []
    for name in names:
        path = DIR / name
        if path.exists() and valid_body(name, path.read_bytes()) and not (refresh and name == CURRENT):
            continue
        body, src = None, None
        if not blocked:
            if not first:
                time.sleep(sleep)
            first = False
            try:
                body, src = _download(name, BASE + FILES[name], get), "jsda"
            except JsdaError as e:
                errors.append(str(e))
                blocked = blocked or "429" in str(e)
        if body is None and wayback and name in WAYBACK:
            ts, orig = WAYBACK[name]
            time.sleep(min(sleep, 5.0))
            try:
                body, src = _download(name, f"https://web.archive.org/web/{ts}id_/{orig}", get, referer=""), f"wayback {ts} {orig}"
            except JsdaError as e:
                errors.append(str(e))
        if body is not None:
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(body)
            tmp.replace(path)
            (DIR / f"{name}.source").write_text(f"{src} fetched {pd.Timestamp.now():%Y-%m-%d %H:%M}\n")
    have = {n: DIR / n for n in FILES if (DIR / n).exists()}
    if not have and errors:
        raise JsdaError("; ".join(errors))
    return have


# ---------------------------------------------------------------- parse
def _norm(v) -> str:
    return unicodedata.normalize("NFKC", str(v)).replace("_x000D_", "").replace("\n", " ").strip()


def _investor(label) -> str | None:
    s = _norm(label)
    if not s or s == "nan":
        return None
    for key in ("other_fin", "other_corps", "dealers", "regional2"):  # before their substrings
        if INVESTORS[key][0] in s or (key == "regional2" and "2nd Regional" in s):
            return key
    for key, (jp, _) in INVESTORS.items():
        if s.startswith(_norm(jp)) or _norm(jp) == s:
            return key
    if s.startswith("生保") or "Insurance" in s:
        return "insurers"
    return None


def _maturity(label) -> str | None:
    s = _norm(label)
    if "超長期" in s:
        return "super_long"
    if "長期" in s and "超" not in s:
        return "long"
    if "中期" in s:
        return "medium"
    if "割引短期" in s or "政府短期" in s or "国庫短期" in s:
        return "tbills"
    if s.startswith("割引"):
        return "zero"
    if "国債合計" in s or s.startswith("国債"):
        return "total_jgb"
    return None


def _month(v) -> pd.Timestamp | None:
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
        try:
            return pd.Timestamp(v).to_period("M").to_timestamp()
        except (ValueError, TypeError):
            return None
    s = _norm(v)
    m = re.fullmatch(r"(\d{4})[./年-](\d{1,2})月?", s)
    if not m:
        m = re.fullmatch(r"(\d{4})\.(\d)0?", s)  # 2004.1 read as float for October
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return None


def _num(v) -> float:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = _norm(v).replace(",", "")
    if s in ("", "nan", "-", "－", "―", "‐"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _ichiran_month(v) -> pd.Timestamp | None:
    """kokusaiichiran month cells are 'YYYY.MM' strings or floats (2004.1 = October)."""
    if isinstance(v, float) and np.isfinite(v):
        y = int(v)
        mo = int(round((v - y) * 100))
        return pd.Timestamp(y, mo, 1) if 1 <= mo <= 12 else None
    return _month(v)


def parse_ichiran(path: Path) -> pd.DataFrame:
    """kokusaiichiran.xls → long (month, investor, maturity, side, value_100mn); side ∈ sale /
    purchase / net (net = sale − purchase as published)."""
    xl = pd.ExcelFile(path)
    rows = []
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, dtype=object)
        hdr = raw.iloc[:8]
        start = next(i for i in range(len(raw)) if _ichiran_month(raw.iat[i, 0]) is not None)
        inv = side = mat = None
        cols = {}
        for j in range(1, raw.shape[1]):
            k = _investor(hdr.iat[2, j])
            if k:
                inv = k
            lab = _norm(hdr.iat[3, j])
            if lab.startswith("売"):
                side = "sale"
            elif lab.startswith("買"):
                side = "purchase"
            elif lab.startswith("差"):
                side = "net"
            m = _maturity(hdr.iat[5, j]) if _norm(hdr.iat[5, j]) != "nan" else _maturity(hdr.iat[4, j])
            if inv and side and m:
                cols[j] = (inv, side, m)
        for i in range(start, len(raw)):
            mo = _ichiran_month(raw.iat[i, 0])
            if mo is None:
                break
            for j, (k, sd, m) in cols.items():
                rows.append((mo, k, m, sd, _num(raw.iat[i, j])))
    out = pd.DataFrame(rows, columns=["month", "investor", "maturity", "side", "value_100mn"])
    # Bills are three columns before Feb-2009 (TB, FB, 国庫短期証券等): sum, keeping all-missing as NaN.
    return (out.groupby(["month", "investor", "maturity", "side"], as_index=False)["value_100mn"]
            .sum(min_count=1))


_SIDE_SHEET = {"net": "一般差引", "sale": "一般売付", "purchase": "一般買付"}


def parse_fy(path: Path) -> pd.DataFrame:
    """koushasai<FY>.xlsx → same long frame from the all-members outright sheets (Ｅ, Ｈ, Ｋ)."""
    xl = pd.ExcelFile(path)
    rows = []
    for side, key in _SIDE_SHEET.items():
        sheet = next((s for s in xl.sheet_names if key in _norm(s) and "証券" not in s), None)
        if sheet is None:
            continue
        raw = xl.parse(sheet, header=None, dtype=object)
        hdr_rows = range(min(8, len(raw)))
        jgb = next((j for i in hdr_rows for j in range(raw.shape[1]) if _norm(raw.iat[i, j]).startswith("国債")), None)
        if jgb is None:
            raise JsdaError(f"{path.name}/{sheet}: no 国債 header")
        cols = {jgb: "total_jgb"}
        for j in range(jgb + 1, min(jgb + 8, raw.shape[1])):
            m = next((_maturity(raw.iat[i, j]) for i in hdr_rows if _maturity(raw.iat[i, j]) not in (None, "total_jgb")), None)
            if m is None:
                break
            cols[j] = m
        for i in range(len(raw)):
            mo = _month(raw.iat[i, 0])
            if mo is None:
                continue
            k = _investor(raw.iat[i, 1]) or _investor(raw.iat[i, 2])
            if k is None:
                continue
            for j, m in cols.items():
                rows.append((mo, k, m, side, _num(raw.iat[i, j])))
    out = pd.DataFrame(rows, columns=["month", "investor", "maturity", "side", "value_100mn"])
    return out.groupby(["month", "investor", "maturity", "side"], as_index=False)["value_100mn"].sum(min_count=1)


def parse_file(path: Path) -> pd.DataFrame:
    p = Path(path)
    return parse_ichiran(p) if p.suffix.lower() == ".xls" else parse_fy(p)


def tidy(long: pd.DataFrame, source: str) -> pd.DataFrame:
    """Long sides → one row per (month, investor, maturity) in ¥bn, net flipped to purchases − sales.
    The published net is used where present; purchases − sales fills it otherwise."""
    w = long.pivot_table(index=["month", "investor", "maturity"], columns="side", values="value_100mn",
                         aggfunc="first").reset_index()
    for c in ("sale", "purchase", "net"):
        if c not in w:
            w[c] = np.nan
    net = (-w["net"]).where(w["net"].notna(), w["purchase"] - w["sale"])
    return pd.DataFrame({
        "month": w["month"], "investor": w["investor"], "maturity": w["maturity"],
        "net_purchase_jpy_bn": net / 10, "purchases_jpy_bn": w["purchase"] / 10, "sales_jpy_bn": w["sale"] / 10,
        "source": source,
    })


def build(paths: dict[str, Path] | None = None) -> pd.DataFrame:
    """Parse every cached file; where months overlap the higher-priority file (FILES order: newest
    vintage first) wins."""
    if paths is None:
        paths = {n: DIR / n for n in FILES if (DIR / n).exists()}
    parts = []
    for rank, name in enumerate(FILES):
        if name not in paths:
            continue
        t = tidy(parse_file(paths[name]), name)
        parts.append(t.assign(_rank=rank))
    if not parts:
        return pd.DataFrame(columns=["month", "investor", "maturity", "net_purchase_jpy_bn", "purchases_jpy_bn",
                                     "sales_jpy_bn", "source"])
    allp = pd.concat(parts, ignore_index=True)
    best = allp.groupby("month")["_rank"].transform("min")
    out = allp[allp["_rank"] == best].drop(columns="_rank")
    return out.sort_values(["investor", "maturity", "month"]).reset_index(drop=True)


_PANEL: dict[str, pd.DataFrame] = {}


def panel(refresh: bool = False) -> pd.DataFrame:
    """Tidy monthly panel: month (data month), investor, maturity (total_jgb + MATURITIES),
    net_purchase_jpy_bn (+ = net buying), purchases_jpy_bn, sales_jpy_bn, source (file)."""
    if refresh:
        fetch(refresh=True)
        _PANEL.clear()
    if "p" not in _PANEL:
        p = build()
        if not p.empty:
            p.to_parquet(OUT, index=False)
        elif OUT.exists():
            p = pd.read_parquet(OUT)
        _PANEL["p"] = p
    return _PANEL["p"].copy()


def coverage(p: pd.DataFrame | None = None) -> pd.DataFrame:
    p = panel() if p is None else p
    g = p.groupby("source")["month"].agg(first="min", last="max", n="nunique").reset_index()
    return g.sort_values("first").reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def public_month(month: pd.Timestamp) -> pd.Timestamp:
    """Data month t is published around the 20th of t+1."""
    return (pd.Timestamp(month).to_period("M") + 1).to_timestamp()


def wide(p: pd.DataFrame | None = None, col: str = "net_purchase_jpy_bn") -> pd.DataFrame:
    """month × (investor, maturity) of `col`, monthly frequency, data months."""
    p = panel() if p is None else p
    w = p.pivot_table(index="month", columns=["investor", "maturity"], values=col, aggfunc="first")
    return w.asfreq("MS").sort_index()


def dv01(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """month × investor: net purchases in ¥bn per bp, coupon + zero buckets × DURATION (bills excluded,
    they carry no duration worth counting)."""
    w = wide(p)
    invs = sorted({i for i, _ in w.columns})
    out = {}
    for inv in invs:
        s = 0.0
        for m in ("super_long", "long", "medium", "zero"):
            if (inv, m) in w:
                s = s + w[(inv, m)].fillna(0.0) * DURATION[m] * 1e-4
        out[inv] = s
    return pd.DataFrame(out, index=w.index)


def _super_long_supply(months: pd.DatetimeIndex) -> pd.Series:
    """Gross JGB coupon issuance with ≥15y to maturity at issue, ¥bn per month (repo flow_tenor)."""
    path = CACHE / "flow_tenor.parquet"
    if not path.exists():
        return pd.Series(np.nan, index=months)
    f = pd.read_parquet(path)
    f = f[(f["country"] == "JP") & (f["sec_type"] == "Nominal") &
          f["tenor_bucket"].astype(str).isin(["15-20y", "20-30y", "30y+"])]
    s = f.groupby("month")["gross_mn"].sum() / 1000
    s.index = pd.to_datetime(s.index)
    last = s.index.max()
    return s.reindex(months).fillna(0.0).where(months <= last)


METRICS: dict[str, dict] = {
    "jsda_ins_sl_3m": {"label": "Insurers' super-long net buying, 3m", "unit": "¥tn", "sign": -1, "fmt": ".2f",
                       "what": "Life + non-life insurers' net purchases (purchases − sales) of super-long JGBs "
                               "(20/30/40y issues), JSDA OTC outright, trailing 3 months."},
    "jsda_ins_sl_12m": {"label": "Insurers' super-long net buying, 12m", "unit": "¥tn", "sign": -1, "fmt": ".2f",
                        "what": "As the 3m metric, trailing 12 months. ESR-driven duration matching (2016–24) "
                                "pushed this up; the 2024–26 slowdown is the 'insurers stop buying' story."},
    "jsda_ins_sl_share": {"label": "Insurers' share of super-long issuance, 12m", "unit": "% of gross", "sign": -1,
                          "fmt": ".0f", "what": "12m insurers' super-long net buying ÷ 12m gross JGB issuance with "
                                                "≥15y to maturity (repo `flow_tenor`)."},
    "jsda_ins_dv01_12m": {"label": "Insurers' net buying DV01, 12m", "unit": "¥bn/bp", "sign": -1, "fmt": ".1f",
                          "what": "Insurers' net purchases across coupon buckets × fixed bucket durations "
                                  "(super-long 17, long 8.5, medium 3.5, zero 8), trailing 12 months."},
    "jsda_trust_dv01_12m": {"label": "Trust banks' net buying DV01, 12m", "unit": "¥bn/bp", "sign": -1, "fmt": ".1f",
                            "what": "Trust banks (pension trust accounts incl. GPIF mandates) net purchases, "
                                    "DV01-weighted as above, trailing 12 months."},
    "jsda_banks_dv01_12m": {"label": "Banks' net buying DV01, 12m", "unit": "¥bn/bp", "sign": -1, "fmt": ".1f",
                            "what": "City + regional + second-tier regional banks' net purchases, DV01-weighted, "
                                    "trailing 12 months."},
    "jsda_foreign_dv01_12m": {"label": "Foreigners' net buying DV01, 12m", "unit": "¥bn/bp", "sign": -1, "fmt": ".1f",
                              "what": "Foreign investors' net purchases of coupon JGBs, DV01-weighted, trailing "
                                      "12 months (bills excluded)."},
    "jsda_realmoney_dv01_12m": {"label": "Domestic real money net buying DV01, 12m", "unit": "¥bn/bp", "sign": -1,
                                "fmt": ".1f", "what": "Insurers + trust banks + investment trusts + public mutual "
                                                      "aid associations, DV01-weighted, trailing 12 months."},
}


def monthly_metrics(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """month (data month), metric, value."""
    p = panel() if p is None else p
    w = wide(p)
    d = dv01(p)
    sl = w[("insurers", "super_long")] if ("insurers", "super_long") in w else pd.Series(np.nan, index=w.index)
    supply = _super_long_supply(w.index)
    vals = {
        "jsda_ins_sl_3m": sl.rolling(3).sum() / 1000,
        "jsda_ins_sl_12m": sl.rolling(12).sum() / 1000,
        "jsda_ins_sl_share": 100 * sl.rolling(12).sum() / supply.rolling(12).sum().where(lambda v: v > 0),
        "jsda_ins_dv01_12m": d["insurers"].rolling(12).sum() if "insurers" in d else np.nan,
        "jsda_trust_dv01_12m": d["trust_banks"].rolling(12).sum() if "trust_banks" in d else np.nan,
        "jsda_banks_dv01_12m": d[[c for c in BANKS if c in d]].sum(axis=1, min_count=1).rolling(12).sum(),
        "jsda_foreign_dv01_12m": d["foreigners"].rolling(12).sum() if "foreigners" in d else np.nan,
        "jsda_realmoney_dv01_12m": d[[c for c in REAL_MONEY if c in d]].sum(axis=1, min_count=1).rolling(12).sum(),
    }
    rows = [pd.DataFrame({"month": w.index, "metric": k, "value": pd.Series(v, index=w.index).to_numpy()})
            for k, v in vals.items()]
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"])


def metrics(p: pd.DataFrame | None = None, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rows in `headline.panel()` shape (country='JP', month, metric, value, phase='history'), each
    data month stamped at its `public_month`, capped at the last complete month before `today`."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    end = today.to_period("M").to_timestamp() - pd.DateOffset(months=1)
    m = monthly_metrics(p)
    m["month"] = m["month"].map(public_month)
    m = m[m["month"] <= end]
    out = m.assign(country="JP", phase="history")[["country", "month", "metric", "value", "phase"]]
    return out.sort_values(["metric", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 200)
    got = fetch(refresh="--refresh" in sys.argv)
    print({k: str(v) for k, v in got.items()})
    p = panel()
    print(coverage(p).to_string(index=False))
