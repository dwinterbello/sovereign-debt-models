"""JGB futures positioning reads from Japan Exchange Group (Osaka Exchange).

Two caches:

    jpx_jgb_futures_investor.parquet   weekly "Trading by Type of Investors" for JGB futures,
                                       options on 10y JGB futures and 3M TONA futures, 2016+.
    jpx_jgb_futures_oi.parquet         daily open interest / volume / settlement per contract
                                       month. JPX only publishes the latest day, so this file
                                       accumulates one day per run; there is no back history.

Weekly sources (one CSV per week):

    sector/index.html               last ~5 weeks
    sector/00-archives-NN.html      one page per calendar year, 2016 .. last year
    web.archive.org snapshots       of sector/index.html to recover current-year weeks that
                                    rolled off the live page; the CSVs themselves are still
                                    hosted on jpx.co.jp and are fetched from there.

Two CSV layouts. Before 2026-04-13 one row per investor with volume and value side by side,
value in 10**k yen (k = column 3; 4 for JGB futures), and the aggregate codes 10/20/30/70/80
are published. From 2026-04-13 one row per (investor, volume|value), value in yen, and only
leaf codes are published (proprietary is 11, individuals 51). The aggregates are rebuilt from leaves for the
new layout and tagged ``source`` = ``...|derived``. The 2026-04-13 file in the
``tvdivq00000020o8`` folder is a JPX format sample with all-zero data and is excluded.

Product 318 was super-long 20y JGB futures before 2022-03-28 and mini-20y after; 330 was the
old mini-10y before the same date. Both keep one contract label across the switch.

Sell and buy are gross; net = buy - sell. Volume is in contracts, value in yen.
"""
from __future__ import annotations

import gzip
import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from issuance.http import FetchError, get_bytes
from issuance.paths import CACHE, RAW

HOST = "https://www.jpx.co.jp"
SECTOR = HOST + "/english/markets/statistics-derivatives/sector/"
SECTOR_JA = HOST + "/markets/statistics-derivatives/sector/"
OI_PAGE = HOST + "/english/markets/derivatives/trading-volume/index.html"
SETTLE_PAGE = HOST + "/english/markets/derivatives/settlement-price/index.html"
CDX = "https://web.archive.org/cdx/search/cdx"

RAW_JPX = RAW / "jp" / "jpx"
RAW_WEEKLY = RAW_JPX / "investor_weekly"
RAW_OI = RAW_JPX / "open_interest"
RAW_SETTLE = RAW_JPX / "settlement"
WAYBACK_SEEN = RAW_JPX / "wayback_snapshots_seen.json"

OUT_INVESTOR = CACHE / "jpx_jgb_futures_investor.parquet"
OUT_OI = CACHE / "jpx_jgb_futures_oi.parquet"

SAMPLE_DIRS = {"tvdivq00000020o8"}

PRODUCTS = {
    "317": "jgb10y",
    "330": "mini_jgb10y",
    "318": "mini_jgb20y",
    "334": "tona3m",
    "321": "jgb10y_option_put",
    "322": "jgb10y_option_call",
}

INVESTORS = {
    "10": "proprietary",
    "11": "proprietary",
    "70": "brokerage_total",
    "80": "total",
    "30": "institutions",
    "50": "individuals",
    "51": "individuals",
    "60": "foreigners",
    "41": "securities_cos",
    "31": "investment_trusts",
    "32": "business_cos",
    "33": "other_institutions",
    "20": "financial_institutions",
    "21": "life_nonlife_insurers",
    "22": "city_regional_banks",
    "23": "trust_banks",
    "24": "other_financial_institutions",
}
FIN_LEAVES = ["21", "22", "23", "24"]
INST_LEAVES = ["31", "32", "33"] + FIN_LEAVES
BROKERAGE_LEAVES = ["41", "51", "60"] + INST_LEAVES

_LINK = re.compile(
    r"/(?:english/)?markets/statistics-derivatives/sector/([a-z0-9]+)-att/"
    r"Tousi_DV_(?:pdf_)?W_([0-9_]+)\.(?:csv|pdf)"
)


def _html(url: str) -> str:
    payload = get_bytes(url, min_bytes=500)
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return payload.decode("utf-8", errors="replace")


def _links(html: str, base: str) -> dict[str, str]:
    """Week id -> CSV URL. Same week from several pages keeps the first URL seen."""
    out: dict[str, str] = {}
    for folder, week in _LINK.findall(html):
        if folder in SAMPLE_DIRS:
            continue
        out.setdefault(week, f"{base}{folder}-att/Tousi_DV_W_{week}.csv")
    return out


def _live_links() -> dict[str, str]:
    found = _links(_html(SECTOR + "index.html"), SECTOR)
    for n in range(1, 40):
        try:
            page = _html(f"{SECTOR}00-archives-{n:02d}.html")
        except FetchError:
            break
        got = _links(page, SECTOR)
        if not got:
            break
        for k, v in got.items():
            found.setdefault(k, v)
    return found


def _wayback_links(since: str | None = None) -> dict[str, str]:
    """CSV links from archived copies of the index pages. Snapshots already read are skipped."""
    since = since or str(pd.Timestamp.today().year)
    seen = set(json.loads(WAYBACK_SEEN.read_text())) if WAYBACK_SEEN.exists() else set()
    found: dict[str, str] = {}
    for base in (SECTOR, SECTOR_JA):
        target = base.replace("https://www.", "")
        try:
            listing = get_bytes(f"{CDX}?url={target}&matchType=prefix&from={since}"
                                "&filter=original:.*Tousi_DV.*&fl=original&collapse=urlkey",
                                timeout=90, min_bytes=1, retries=3).decode()
            for k, v in _links(listing, base).items():
                found.setdefault(k, v)
        except FetchError:
            pass
        for page in (target + "index.html", target):
            try:
                text = get_bytes(f"{CDX}?url={page}&from={since}&fl=timestamp,statuscode",
                                 timeout=90, min_bytes=1, retries=2).decode()
            except FetchError:
                continue
            if "<html" in text.lower():
                continue
            for line in text.split("\n"):
                parts = line.split()
                if len(parts) != 2 or parts[1] not in ("200", "-"):
                    continue
                key = f"{parts[0]}|{page}"
                if key in seen:
                    continue
                try:
                    html = _html(f"https://web.archive.org/web/{parts[0]}id_/https://www.{page}")
                except FetchError:
                    continue
                if "statistics-derivatives" not in html:
                    continue
                for k, v in _links(html, base).items():
                    found.setdefault(k, v)
                seen.add(key)
    WAYBACK_SEEN.parent.mkdir(parents=True, exist_ok=True)
    WAYBACK_SEEN.write_text(json.dumps(sorted(seen), indent=0))
    return found


def _download(links: dict[str, str]) -> list[Path]:
    RAW_WEEKLY.mkdir(parents=True, exist_ok=True)
    paths = []
    for week, url in sorted(links.items()):
        dest = RAW_WEEKLY / f"Tousi_DV_W_{week}.csv"
        if not dest.exists():
            try:
                get_bytes(url, dest=dest, min_bytes=1000, retries=2)
            except FetchError as exc:
                print(f"  jpx: skip {week}: {exc}")
                continue
        paths.append(dest)
    return paths


def _cells(payload: bytes) -> list[list[str]]:
    text = payload.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False).values.tolist()


def _leaf_rows(rows: list[dict]) -> list[dict]:
    """Rebuild the aggregate investor codes the new layout no longer publishes."""
    df = pd.DataFrame(rows)
    out = list(rows)
    keys = ["week_start", "week_end", "product_code", "unit"]
    for code, leaves in (("20", FIN_LEAVES), ("30", INST_LEAVES), ("70", BROKERAGE_LEAVES),
                         ("80", BROKERAGE_LEAVES + ["11"])):
        sub = df[df["investor_code"].isin(leaves)]
        if sub.empty:
            continue
        agg = sub.groupby(keys, as_index=False)[["sell", "buy"]].sum()
        for rec in agg.to_dict("records"):
            rec.update(investor_code=code, source=rows[0]["source"] + "|derived")
            out.append(rec)
    return out


def parse_weekly(payload: bytes, source: str) -> pd.DataFrame:
    """One weekly CSV -> long frame for the JGB-related products."""
    cells = _cells(payload)
    rows: list[dict] = []
    new_layout = bool(cells) and "Product type" in str(cells[0][0])
    for c in cells[1:] if new_layout else cells:
        product = str(c[0]).strip()
        if product not in PRODUCTS:
            continue
        if new_layout:
            if len(c) < 12 or str(c[1]).strip() != "1":
                continue
            unit = "contracts" if str(c[6]).strip() == "1" else "yen"
            rows.append({
                "week_start": c[3], "week_end": c[4], "product_code": product,
                "investor_code": str(c[5]).strip(), "unit": unit,
                "sell": float(c[7]), "buy": float(c[9]), "source": source,
            })
        else:
            if len(c) < 24 or str(c[3]).strip() != "1":
                continue
            vol_mult = 10.0 ** int(c[1])
            val_mult = 10.0 ** int(c[2])
            base = {"week_start": c[5], "week_end": c[6], "product_code": product,
                    "investor_code": str(c[7]).strip(), "source": source}
            rows.append({**base, "unit": "contracts",
                         "sell": float(c[8]) * vol_mult, "buy": float(c[14]) * vol_mult})
            rows.append({**base, "unit": "yen",
                         "sell": float(c[11]) * val_mult, "buy": float(c[17]) * val_mult})
    if not rows:
        return pd.DataFrame()
    if new_layout:
        rows = _leaf_rows(rows)
    df = pd.DataFrame(rows)
    if (df[["sell", "buy"]].abs().sum().sum()) == 0:
        return pd.DataFrame()
    df["week_start"] = pd.to_datetime(df["week_start"], format="%Y%m%d")
    df["week_end"] = pd.to_datetime(df["week_end"], format="%Y%m%d")
    df["contract"] = df["product_code"].map(PRODUCTS)
    df["investor"] = df["investor_code"].map(INVESTORS)
    df["net"] = df["buy"] - df["sell"]
    return df


def build_investor(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = parse_weekly(p.read_bytes(), p.name)
        if not df.empty:
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["investor"])
    df = (df.sort_values(["week_end", "source"])
            .drop_duplicates(["week_end", "contract", "investor", "unit"], keep="last"))
    cols = ["week_end", "week_start", "contract", "product_code", "investor", "investor_code",
            "sell", "buy", "net", "unit", "source"]
    return df[cols].sort_values(["contract", "unit", "investor", "week_end"]).reset_index(drop=True)


# ---------------------------------------------------------------- daily open interest

_OI_PRODUCTS = {
    "10-year JGB": "jgb10y",
    "mini-10-year JGB": "mini_jgb10y",
    "5-year JGB": "jgb5y",
    "mini-20-year JGB": "mini_jgb20y",
    "3-Month TONA Futures": "tona3m",
}
_SETTLE_PRODUCTS = {"FUT_JBL": "jgb10y", "FUT_JBLM": "mini_jgb10y", "FUT_JBM": "jgb5y",
                    "FUT_JBS": "mini_jgb20y", "FUT_TOA3M": "tona3m"}


def _oi_product(label) -> str | None:
    if not isinstance(label, str):
        return None
    flat = " ".join(label.split())
    for key in sorted(_OI_PRODUCTS, key=len, reverse=True):
        if flat.startswith(key):
            return _OI_PRODUCTS[key]
    return "other"


def parse_oi(path: Path) -> pd.DataFrame:
    sheet = pd.read_excel(path, header=None)
    date = pd.to_datetime(sheet.iloc[1, 0]).normalize()
    start = sheet.index[sheet[0].astype(str).str.contains("JGB Futures", na=False)][0]
    out = []
    for left in (0, 7):
        product = None
        for i in range(start + 2, len(sheet)):
            if str(sheet.iat[i, 0]).startswith("<"):
                break
            label = sheet.iat[i, left]
            if pd.notna(label):
                product = _oi_product(label)
            month = sheet.iat[i, left + 1]
            if product in (None, "other") or not isinstance(month, datetime):
                continue
            out.append({"date": date, "contract": product,
                        "contract_month": pd.Timestamp(month).strftime("%Y%m"),
                        "volume": float(sheet.iat[i, left + 2]),
                        "open_interest": float(sheet.iat[i, left + 3])})
    return pd.DataFrame(out)


def parse_settlement(path: Path) -> pd.DataFrame:
    m = re.search(r"(\d{8})", path.name)
    date = pd.Timestamp(m.group(1))
    text = path.read_bytes().decode("cp932", errors="replace")
    lines = text.splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("Issue Code"))
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])), dtype=str)
    name = df["Issue Name"].str.extract(r"^(FUT_[A-Z0-9]+)_")[0]
    df = df[name.isin(_SETTLE_PRODUCTS)].assign(contract=name.map(_SETTLE_PRODUCTS))
    return pd.DataFrame({
        "date": date, "contract": df["contract"], "contract_month": df["Contract Month"].str.strip(),
        "settlement": pd.to_numeric(df["Settlement Price"], errors="coerce"),
    })


def fetch_oi() -> pd.DataFrame:
    RAW_OI.mkdir(parents=True, exist_ok=True)
    RAW_SETTLE.mkdir(parents=True, exist_ok=True)
    for page, pat, folder in ((OI_PAGE, r'href="([^"]*open_interest_e\.xlsx)"', RAW_OI),
                              (SETTLE_PAGE, r'href="([^"]*rb_e\d{8}\.csv)"', RAW_SETTLE)):
        try:
            html = _html(page)
        except FetchError as exc:
            print(f"  jpx oi: {exc}")
            continue
        for href in sorted(set(re.findall(pat, html))):
            dest = folder / Path(href).name
            if not dest.exists():
                try:
                    get_bytes(HOST + href, dest=dest, min_bytes=1000)
                except FetchError as exc:
                    print(f"  jpx oi: {exc}")
    oi = [parse_oi(p) for p in sorted(RAW_OI.glob("*open_interest_e.xlsx"))]
    st = [parse_settlement(p) for p in sorted(RAW_SETTLE.glob("rb_e*.csv"))]
    oi_df = pd.concat(oi, ignore_index=True) if oi else pd.DataFrame(
        columns=["date", "contract", "contract_month", "volume", "open_interest"])
    st_df = pd.concat(st, ignore_index=True) if st else pd.DataFrame(
        columns=["date", "contract", "contract_month", "settlement"])
    df = oi_df.merge(st_df, on=["date", "contract", "contract_month"], how="outer")
    if OUT_OI.exists():
        old = pd.read_parquet(OUT_OI)
        df = pd.concat([old, df], ignore_index=True)
    df = (df.sort_values(["date", "contract", "contract_month"])
            .drop_duplicates(["date", "contract", "contract_month"], keep="last"))
    df = df[["date", "contract", "contract_month", "open_interest", "volume", "settlement"]]
    df.reset_index(drop=True).to_parquet(OUT_OI, index=False)
    return df


# ---------------------------------------------------------------- entry points

def foreigners_net(df: pd.DataFrame, contract: str = "jgb10y", unit: str = "contracts") -> pd.Series:
    sub = df[(df["contract"] == contract) & (df["investor"] == "foreigners") & (df["unit"] == unit)]
    return sub.set_index("week_end")["net"].sort_index()


def fetch_jpx(wayback: bool = True, with_oi: bool = True) -> pd.DataFrame:
    links = _live_links()
    if wayback:
        try:
            for k, v in _wayback_links().items():
                links.setdefault(k, v)
        except Exception as exc:  # noqa: BLE001 — archive is best-effort gap fill
            print(f"  jpx: wayback skipped: {exc}")
    paths = _download(links)
    known = {p.name for p in paths}
    paths += [p for p in RAW_WEEKLY.glob("Tousi_DV_W_*.csv") if p.name not in known]
    df = build_investor(paths)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_INVESTOR, index=False)
    if with_oi:
        try:
            fetch_oi()
        except Exception as exc:  # noqa: BLE001 — OI is a secondary read
            print(f"  jpx oi failed: {exc}")
    return df


def _summary(df: pd.DataFrame) -> None:
    weeks = df["week_end"].drop_duplicates().sort_values()
    print(f"jpx investor: {len(df):,} rows, {len(weeks)} weeks "
          f"{weeks.iloc[0].date()} .. {weeks.iloc[-1].date()}")
    gaps = weeks.diff().dt.days
    big = weeks[gaps > 10]
    if len(big):
        print("  gaps > 10d ending:", ", ".join(str(d.date()) for d in big))
    s = foreigners_net(df)
    z = (s - s.rolling(52).mean()) / s.rolling(52).std()
    print(pd.DataFrame({"foreigners_net_contracts": s, "z52": z.round(2)}).tail(8).to_string())


if __name__ == "__main__":
    _summary(fetch_jpx())
