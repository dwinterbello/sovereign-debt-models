"""Daily euro-area government bond yields beyond DE/FR/IT, % p.a.

    PYTHONPATH=. .venv/bin/python -m issuance.fetch_egb [--refresh]

Official sources, per country (series y<T> = <T>-year constant-maturity yield to maturity):
  ES  Banco de España 'Tipos de interés' publication, table TI_1_3 (ti.zip): daily average yield of
      outright spot trades in non-stripped Bonos/Obligaciones at 3y (D_G0B1F0ZN), 5y (ZO), 10y (ZP)
      and 15y (ZQ), 1987-93 onward. No 2y or 30y: BdE's 1-2y bucket is not a constant maturity.
  NL  DSTA 'Yields on DSLs': MTS Netherlands daily fixing per DSL (mid yield), 2002-2009 xlsx and
      2010+ ods (updated monthly). y<T> is linear interpolation in residual maturity between the two
      bracketing DSLs (`cm_from_bonds`), no extrapolation beyond 1.5y past the longest DSL.
  BE  NBB.Stat DF_IROLOBE2, 'yield based on a fixed residual term' (type F, interpolated on the OLO
      curve), D.<T>Y.F, 1993-12 onward (30y from 1998-02).
  PT  Banco de Portugal BPstat, yield on fixed-rate Treasury bonds by residual maturity, daily
      (LSEG data, updated weekly), series 12099454 (2y), 12099457 (5y), 12099459 (10y), 1993 onward.
  FI  Bank of Finland, yields on benchmark government bonds (primary-dealer bids at 13:00), 5y and
      10y daily from 1991; y2/y20/y30 interpolated from the per-loan benchmark yields the same report
      carries (currently listed loans only, so they start when bracketing loans exist).
  DE  Bundesbank Svensson par curve (correlations/g4 cache), cross-check only.
  FR, IT  issuance.fetch_ea_yields raw cache (BdF TEC, BdI MTS benchmark), parsed offline only.
  EA_AAA, EA_ALL  ECB YC, euro-area AAA (G_N_A) / all-issuer (G_N_C) Svensson curve: y<T> par yield
      (PY_<T>Y), z<T> spot (SR_<T>Y), daily from 2004-09.
No official daily tenor yields found for:
  AT  OeNB publishes only the outstanding-weighted average yield (UDRB, report 2.11.1) — no tenor.
  IE  Central Bank of Ireland has no bond-yield table; NTMA posts per-bond PDFs only.
  GR  Bank of Greece publishes 30 trading days plus monthly averages, behind an Akamai block (403).
These, and the tenors official sources lack (ES y2/y30, NL/PT/FI tails), come from CNBC closes
(ts-api.cnbc.com, <CC><T>Y-<CC>, about two years of daily bars): a vendor series, used only after
the last official observation (or everywhere when there is none), labelled in `source`.
Monthly: ECB IRS long-term convergence yields (IRS.M.<CC>.L.L40.CI.0000.EUR.N.Z) fill y10 for
months with no daily y10, labelled; `overlap()` scores daily monthly means against it.

Raw downloads are cached under cache/egb/ and parsed offline; a failed download parses the cache.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from issuance.paths import CACHE, REPO

RAW = CACHE / "egb"
OUT = CACHE / "egb_daily.parquet"
TENORS = (2, 5, 10, 20, 30)
COUNTRIES = ("ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR")
CROSS = ("DE", "FR", "IT")
COLS = ["country", "series", "date", "value", "source"]

BDE_TI = "https://www.bde.es/webbe/es/estadisticas/compartido/datos/zip/ti.zip"
BDE_SERIES = {"D_G0B1F0ZN": 3, "D_G0B1F0ZO": 5, "D_G0B1F0ZP": 10, "D_G0B1F0ZQ": 15}
DSTA_OLD = ("https://english.dsta.nl/site/binaries/site-content/collections/documents/2010/09/01/"
            "yields-on-dsls-from-2002-to-2009/mts-fixings-van-2002-tot-en-met-2009.xlsx")
DSTA_PAGE = "https://english.dsta.nl/latest/statistical-information/yields-on-dsls"
NBB_URL = ("https://nsidisseminate-stat.nbb.be/rest/data/BE2,DF_IROLOBE2,/D.2Y+5Y+10Y+20Y+30Y.F")
BP_URL = ("https://bpstat.bportugal.pt/data/v1/domains/26/datasets/690b7b36fd36c0dbe249c48cbbc39524/"
          "?lang=EN&series_ids=12099454,12099457,12099459&obs_since=1990-01-01")
BP_SERIES = {"12099454": 2, "12099457": 5, "12099459": 10}
BOF_URL = ("https://reports.suomenpankki.fi/WebForms/ReportViewerPage.aspx"
           "?report=/tilastot/arvopaperimarkkinat/velkapaperit/viitelainojen_korot_v2_en")
ECB_YC = ("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A+G_N_C.SV_C_YM."
          "PY_2Y+PY_5Y+PY_10Y+PY_20Y+PY_30Y+SR_2Y+SR_5Y+SR_10Y+SR_20Y+SR_30Y?format=csvdata")
ECB_IRS = ("https://data-api.ecb.europa.eu/service/data/IRS/M.{areas}.L.L40.CI.0000.EUR.N.Z"
           "?format=csvdata")
CNBC_URL = "https://ts-api.cnbc.com/harmony/app/charts/1Y.json?symbol={sym}"
BUND = REPO / "correlations" / "g4" / "cache" / "bbk" / "bbk_bund.csv"

# CNBC symbols tried per country (a symbol CNBC does not carry returns no bars and is skipped).
CNBC_TENORS = {"ES": (2, 5, 10, 30), "NL": (2, 5, 10, 30), "BE": (2, 5, 10, 30), "AT": (2, 5, 10, 30),
               "PT": (2, 5, 10, 30), "IE": (2, 5, 10, 30), "FI": (2, 5, 10, 30), "GR": (2, 5, 10, 30)}
# Interpolation limits: max distance between the bracketing bonds, and how far past the longest
# bond the long end may be held flat (years).
MAX_GAP = {2: 2.5, 5: 3.0, 10: 8.0, 20: 12.0, 30: 15.0}
MAX_EXTRAP = 1.5
# A bond quoted further than this from the median of its maturity neighbours is a bad print.
OUTLIER_PP = 0.75

SRC = {
    "ES": "BdE TI_1_3 secondary-market Bonos/Obligaciones {t}y avg yield ({code})",
    "NL": "DSTA MTS NL daily fixing, DSL mid yields interpolated to {t}y",
    "BE": "NBB DF_IROLOBE2 D.{t}Y.F (OLO fixed residual term)",
    "PT": "BdP BPstat OT fixed-rate {t}y residual maturity ({code})",
    "FI": "BoF benchmark government bond {t}y",
    "FI_i": "BoF benchmark loan yields interpolated to {t}y",
    "DE": "Bundesbank Svensson par yield {t}y (g4 cache)",
    "EA_AAA": "ECB YC AAA Svensson {m} {t}y (YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{k}_{t}Y)",
    "EA_ALL": "ECB YC all-issuer Svensson {m} {t}y (YC.B.U2.EUR.4F.G_N_C.SV_C_YM.{k}_{t}Y)",
    "CNBC": "CNBC {sym} close (vendor; days without an official {t}y)",
    "IRS": "ECB IRS long-term convergence yield, monthly (IRS.M.{c}.L.L40.CI.0000.EUR.N.Z)",
}
ES_MONTHS = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AGO": 8, "SEP": 9,
             "OCT": 10, "NOV": 11, "DIC": 12}


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _clean(s: pd.Series, today: pd.Timestamp | None = None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[(s > -5) & (s < 50)]
    return s[s.index <= (_today() if today is None else today)]


def _get(url: str, path: Path, offline: bool = False, timeout: int = 90, headers: dict | None = None) -> bytes:
    if offline:
        return path.read_bytes()
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0 (issuance research)"}, timeout=timeout)
        r.raise_for_status()
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return r.content
    except requests.RequestException:
        if not path.exists():
            raise
        return path.read_bytes()


def _long(country: str, series: str, s: pd.Series, source) -> pd.DataFrame:
    return pd.DataFrame({"country": country, "series": series, "date": pd.DatetimeIndex(s.index),
                         "value": s.to_numpy(dtype=float), "source": source})


# ------------------------------------------------------------------ parsers (offline)
def parse_bde_csv(text: str, codes: dict[str, int] = BDE_SERIES) -> pd.DataFrame:
    """BdE publication CSV (code row, metadata rows, then 'DD MMM YYYY' rows with Spanish month
    abbreviations, '_' = missing) → wide % by date, columns per `codes`."""
    rows = list(pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False).itertuples(index=False))
    head = [str(x).strip() for x in rows[0]]
    pat = re.compile(r"^(\d{2}) ([A-Z]{3}) (\d{4})$")
    dates, vals = [], []
    for r in rows[1:]:
        m = pat.match(str(r[0]).strip())
        if not m or m.group(2) not in ES_MONTHS:
            continue
        dates.append(pd.Timestamp(int(m.group(3)), ES_MONTHS[m.group(2)], int(m.group(1))))
        vals.append(list(r))
    w = pd.DataFrame(vals, columns=head, index=pd.DatetimeIndex(dates))
    out = pd.DataFrame(index=w.index)
    for code, t in codes.items():
        if code in w:
            out[f"y{t}"] = pd.to_numeric(w[code].replace("_", np.nan), errors="coerce")
    return out.dropna(how="all").sort_index()


def parse_nbb_csv(text: str) -> pd.DataFrame:
    """NBB SDMX-CSV for DF_IROLOBE2 → wide % by date, columns y<T> (type F only)."""
    d = pd.read_csv(io.StringIO(text))
    d = d[d["IROLOBE2_TYPE"] == "F"]
    d["t"] = d["IROLOBE2_MATUR"].str.extract(r"^(\d+)Y$")[0].astype(float)
    d = d.dropna(subset=["t"])
    d["date"] = pd.to_datetime(d["TIME_PERIOD"])
    w = d.pivot_table(index="date", columns="t", values="OBS_VALUE")
    w.columns = [f"y{int(c)}" for c in w.columns]
    return w.sort_index()


def parse_jsonstat(raw: str | bytes, series_dim: str, labels: dict[str, int]) -> pd.DataFrame:
    """BPstat JSON-stat dataset → wide by reference_date, column y<T> for each category of
    `series_dim` whose label starts with '<T> year'. `labels` maps series id → tenor (unused when the
    label carries the tenor)."""
    j = json.loads(raw)
    ids, size = j["id"], j["size"]
    vals = j["value"]
    if isinstance(vals, dict):
        arr = np.full(int(np.prod(size)), np.nan)
        for k, v in vals.items():
            arr[int(k)] = np.nan if v is None else v
    else:
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    arr = arr.reshape(size)
    dims = j["dimension"]
    sd, dd = ids.index(series_dim), ids.index("reference_date")
    cats = dims[series_dim]["category"]
    cidx, clab = cats["index"], cats["label"]
    cidx = cidx if isinstance(cidx, dict) else {k: i for i, k in enumerate(cidx)}
    dates = dims["reference_date"]["category"]["index"]
    dates = list(dates) if isinstance(dates, list) else sorted(dates, key=dates.get)
    out = {}
    for code, pos in cidx.items():
        m = re.match(r"(\d+) year", clab[code])
        if not m:
            continue
        sl = [0] * len(ids)
        sl[sd] = pos
        idx = tuple(slice(None) if i == dd else sl[i] for i in range(len(ids)))
        out[f"y{int(m.group(1))}"] = arr[idx]
    return pd.DataFrame(out, index=pd.to_datetime(dates)).dropna(how="all").sort_index()


def parse_bof_csv(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BoF SSRS CSV export → (benchmark wide y5/y10 by date, per-loan long date/maturity/yld)."""
    text = text.lstrip("\ufeff")
    i = text.find("txtb_row_title")
    d = pd.read_csv(io.StringIO(text[i:]), dtype=str)
    d["date"] = pd.to_datetime(d["title"], format="%d.%m.%Y", errors="coerce")
    d["v"] = pd.to_numeric(d["txtb_value"], errors="coerce")
    d = d.dropna(subset=["date", "v"])
    lab = d["txtb_currency"].astype(str)
    bm = lab.str.extract(r"Yield on gover?n?ment bonds, (\d+) year")[0]
    w = d[bm.notna()].assign(t=bm[bm.notna()].astype(int)).pivot_table(index="date", columns="t", values="v")
    w.columns = [f"y{c}" for c in w.columns]
    mat = pd.to_datetime(lab.str.extract(r"–\s*(\d{1,2} \w{3} \d{4})\s*$")[0], format="%d %b %Y", errors="coerce")
    loans = pd.DataFrame({"date": d["date"], "maturity": mat, "yld": d["v"]}).dropna()
    return w.sort_index(), loans.reset_index(drop=True)


def parse_ecb_csv(raw: str | bytes) -> pd.DataFrame:
    text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
    d = pd.read_csv(io.StringIO(text), usecols=["KEY", "TIME_PERIOD", "OBS_VALUE"], dtype={"KEY": str})
    d["date"] = pd.to_datetime(d["TIME_PERIOD"].astype(str))
    d["value"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    return d.dropna(subset=["value"])[["KEY", "date", "value"]].reset_index(drop=True)


def parse_cnbc(raw: str | bytes) -> pd.Series:
    """CNBC harmony chart JSON → daily close % by trade date (weekend bars dropped)."""
    j = json.loads(raw)
    bars = (j.get("barData") or {}).get("priceBars") or []
    if not bars:
        return pd.Series(dtype=float)
    d = pd.DataFrame(bars)
    idx = pd.to_datetime(d["tradeTime"].astype(str).str[:8], format="%Y%m%d")
    s = pd.Series(pd.to_numeric(d["close"], errors="coerce").to_numpy(), index=idx)
    return _clean(s[s.index.dayofweek < 5])


def _dsl_maturity(desc: pd.Series) -> pd.Series:
    """'NETHER 3.750 15/01/23' → 2023-01-15 (dd/mm/yy)."""
    m = desc.astype(str).str.extract(r"(\d{2})/(\d{2})/(\d{2})\s*$")
    return pd.to_datetime("20" + m[2] + "-" + m[1] + "-" + m[0], format="%Y-%m-%d", errors="coerce")


def _dsta_frame(rows: list[list], header: list[str]) -> pd.DataFrame:
    h = [str(x).strip().lower() for x in header]
    col = {k: i for i, k in enumerate(h)}
    di = col.get("date", col.get("refdate"))
    yi = col.get("mid yield", col.get("yield"))
    if di is None or yi is None:
        return pd.DataFrame(columns=["date", "maturity", "yld"])
    d = pd.DataFrame({"date": [r[di] if di < len(r) else None for r in rows],
                      "yld": [r[yi] if yi < len(r) else None for r in rows]})
    if "description" in col:
        d["maturity"] = _dsl_maturity(pd.Series([r[col["description"]] if col["description"] < len(r) else "" for r in rows]))
    else:
        d["maturity"] = pd.to_datetime(pd.Series([r[col["maturity"]] for r in rows]), errors="coerce")
    if "type" in col:
        d = d[[str(r[col["type"]]).strip().upper() == "DSL" for r in rows]]
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10], errors="coerce")
    d["yld"] = pd.to_numeric(d["yld"], errors="coerce") * 100.0
    d = d.dropna()
    return d[d["yld"] != 0.0]


def parse_dsta_xlsx(path) -> pd.DataFrame:
    """DSTA 2002-2009 workbook (one sheet per year, header row 'Date', yields as fractions)."""
    out = []
    x = pd.ExcelFile(path)
    for sh in x.sheet_names:
        d = x.parse(sh, header=None)
        hr = next((i for i in range(min(30, len(d))) if str(d.iat[i, 0]).strip().lower() in ("date", "refdate")), None)
        if hr is None:
            continue
        body = d.iloc[hr + 1:].values.tolist()
        out.append(_dsta_frame(body, list(d.iloc[hr])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "maturity", "yld"])


def parse_dsta_ods(path) -> pd.DataFrame:
    """DSTA 2010+ OpenDocument spreadsheet, streamed (content.xml is ~125 MB)."""
    from lxml import etree

    T = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    O = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
    out, header, rows = [], None, []
    with zipfile.ZipFile(path) as z:
        for ev, el in etree.iterparse(z.open("content.xml"), events=("end",), tag=(T + "table-row", T + "table")):
            if el.tag == T + "table":
                if header and rows:
                    out.append(_dsta_frame(rows, header))
                header, rows = None, []
                el.clear()
                continue
            cells = []
            for c in el.iterchildren(T + "table-cell"):
                rep = min(int(c.get(T + "number-columns-repeated", "1")), 12)
                v = c.get(O + "value") or c.get(O + "date-value") or "".join(c.itertext())
                cells.extend([v] * rep)
                if len(cells) > 12:
                    break
            el.clear()
            if header is None:
                low = [str(x).strip().lower() for x in cells]
                if ("date" in low or "refdate" in low) and any("isin" in x for x in low):
                    header = cells
                continue
            if cells and cells[0]:
                rows.append(cells)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "maturity", "yld"])


def _neighbour_median(y: np.ndarray, k: int = 2) -> np.ndarray:
    """Median of up to k bonds either side in maturity order, excluding the bond itself."""
    out = np.empty(len(y))
    for i in range(len(y)):
        nb = np.concatenate([y[max(0, i - k):i], y[i + 1:i + 1 + k]])
        out[i] = np.median(nb)
    return out


def cm_from_bonds(bonds: pd.DataFrame, tenors=TENORS, max_gap: dict | None = None,
                  max_extrap: float = MAX_EXTRAP, min_resid: float = 0.25) -> pd.DataFrame:
    """Constant-maturity yields by linear interpolation in residual maturity (years, act/365.25)
    between the bracketing bonds of each date. Bonds under `min_resid` are dropped, and so are prints
    more than OUTLIER_PP from the bond's own 5-day centred median or from the median of its
    maturity neighbours on the day (DSTA files carry mis-scaled rows); a tenor past
    the longest bond is held flat only within `max_extrap`; a bracket wider than max_gap[t] gives
    no value."""
    gap = MAX_GAP if max_gap is None else max_gap
    b = bonds.dropna(subset=["date", "maturity", "yld"]).copy()
    b["r"] = (pd.to_datetime(b["maturity"]) - pd.to_datetime(b["date"])).dt.days / 365.25
    b = b[b["r"] >= min_resid].sort_values(["maturity", "date"])
    med = b.groupby("maturity")["yld"].transform(
        lambda s: s.rolling(5, center=True, min_periods=3).median())
    b = b[~((b["yld"] - med).abs() > OUTLIER_PP)].sort_values(["date", "r"])
    res = {}
    for dt, g in b.groupby("date", sort=True):
        r, y = g["r"].to_numpy(), g["yld"].to_numpy()
        if len(y) >= 4:
            ok = np.abs(y - _neighbour_median(y)) <= OUTLIER_PP
            r, y = r[ok], y[ok]
        row = {}
        for t in tenors:
            j = np.searchsorted(r, t)
            if j < len(r) and np.isclose(r[j], t):
                row[f"y{t}"] = y[j]
            elif 0 < j < len(r):
                if r[j] - r[j - 1] <= gap.get(t, 5.0):
                    row[f"y{t}"] = y[j - 1] + (y[j] - y[j - 1]) * (t - r[j - 1]) / (r[j] - r[j - 1])
            elif j == len(r) and len(r) and t - r[-1] <= max_extrap:
                row[f"y{t}"] = y[-1]
        res[dt] = row
    return pd.DataFrame.from_dict(res, orient="index").sort_index()


# ------------------------------------------------------------------ fetchers (wide % by date)
def es_bde(offline: bool = False) -> pd.DataFrame:
    raw = _get(BDE_TI, RAW / "bde_ti.zip", offline)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower() == "ti_1_3.csv")
        return parse_bde_csv(z.read(name).decode("latin-1"))


def nl_dsta(offline: bool = False) -> pd.DataFrame:
    old = RAW / "dsta_2002_2009.xlsx"
    if not offline and not old.exists():
        try:
            _get(DSTA_OLD, old)
        except Exception as exc:  # noqa: BLE001
            print(f"  egb: DSTA 2002-09 failed: {exc}")
    new = RAW / "dsta_2010_on.ods"
    if not offline:
        try:
            page = requests.get(DSTA_PAGE, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
            doc = re.findall(r'href="(/documents/\d{4}/\d{2}/\d{2}/yields-on-dsls-from-2010-onwards)"', page)
            sub = requests.get("https://english.dsta.nl" + doc[0], headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
            url = re.findall(r'href="([^"]+\.(?:ods|xlsx))"', sub)[0]
            _get(url, new, timeout=180)
        except Exception as exc:  # noqa: BLE001
            print(f"  egb: DSTA 2010+ failed: {exc}")
    parts = []
    if old.exists():
        parts.append(parse_dsta_xlsx(old))
    if new.exists():
        parts.append(parse_dsta_ods(new))
    if not parts:
        return pd.DataFrame()
    bonds = pd.concat(parts, ignore_index=True)
    bonds.to_parquet(RAW / "dsta_bonds.parquet", index=False)
    return cm_from_bonds(bonds)


def be_nbb(offline: bool = False) -> pd.DataFrame:
    hdr = {"User-Agent": "Mozilla/5.0 (issuance research)", "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
    return parse_nbb_csv(_get(NBB_URL, RAW / "nbb_irolobe2.csv", offline, headers=hdr).decode("utf-8", "ignore"))


def pt_bpstat(offline: bool = False) -> pd.DataFrame:
    return parse_jsonstat(_get(BP_URL, RAW / "bpstat_ot.json", offline), "45", BP_SERIES)


def _bof_download() -> bytes:
    """ASP.NET report viewer: GET the form, post the earliest start date, export CSV."""
    import html as _html

    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (issuance research)"
    x = s.get(BOF_URL, timeout=120).text
    form = {k: _html.unescape(v) for k, v in
            re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', x)}
    for c in ("ctl03", "ctl05", "ctl07", "ctl09"):
        m = re.search(r'name="ReportViewer1\$ctl04\$%s\$ddValue".*?</select>' % c, x, re.S)
        opts = re.findall(r'<option[^>]*value="([^"]*)"', m.group(0))
        sel = re.findall(r'<option selected="selected" value="([^"]*)"', m.group(0))
        form[f"ReportViewer1$ctl04${c}$ddValue"] = opts[-1] if c == "ctl07" else sel[0]
    form["ReportViewer1$ctl04$ctl00"] = "View Report"
    y = s.post(BOF_URL, data=form, timeout=180).text
    eb = re.findall(r'ExportUrlBase":"([^"]+)"', y)[0].encode().decode("unicode_escape")
    r = s.get("https://reports.suomenpankki.fi" + eb + "CSV", timeout=240)
    r.raise_for_status()
    return r.content


def fi_bof(offline: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RAW / "bof_benchmark.csv"
    if not offline:
        try:
            raw = _bof_download()
            if b"txtb_row_title" in raw:
                path.write_bytes(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"  egb: BoF report failed: {exc}")
    return parse_bof_csv(path.read_bytes().decode("utf-8", "ignore"))


def ecb_yc(offline: bool = False) -> pd.DataFrame:
    return parse_ecb_csv(_get(ECB_YC, RAW / "ecb_yc.csv", offline, timeout=180))


def ecb_irs10(offline: bool = False) -> pd.DataFrame:
    """Monthly long-term convergence yields: country, month, value."""
    areas = "+".join(COUNTRIES + CROSS)
    d = parse_ecb_csv(_get(ECB_IRS.format(areas=areas), RAW / "ecb_irs_m.csv", offline))
    d["country"] = d["KEY"].str.split(".").str[2]
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    return d[["country", "month", "value"]]


def cnbc(sym: str, offline: bool = False) -> pd.Series:
    path = RAW / f"cnbc_{sym}.json"
    if offline and not path.exists():
        return pd.Series(dtype=float)
    return parse_cnbc(_get(CNBC_URL.format(sym=sym), path, offline, timeout=30))


def de_bbk() -> pd.DataFrame:
    if not BUND.exists():
        return pd.DataFrame()
    b = pd.read_csv(BUND, index_col=0, parse_dates=True)
    return pd.DataFrame({f"y{t}": b[f"bund_{t}"] for t in TENORS if f"bund_{t}" in b})


# ------------------------------------------------------------------ assemble
def splice(official: pd.Series, tail: pd.Series, off_src, tail_src: str) -> pd.DataFrame:
    """Official series, then tail values dated strictly after its last observation."""
    official = _clean(official) if len(official) else official
    last = official.index.max() if len(official) else pd.Timestamp.min
    t = tail[pd.DatetimeIndex(tail.index) > last].dropna() if len(tail) else tail
    a = pd.DataFrame({"date": official.index, "value": official.to_numpy(dtype=float), "source": off_src})
    b = pd.DataFrame({"date": t.index, "value": t.to_numpy(dtype=float), "source": tail_src})
    return pd.concat([x for x in (a, b) if len(x)], ignore_index=True) if len(a) or len(b) else a


def _wide_long(country: str, w: pd.DataFrame, src) -> list[pd.DataFrame]:
    out = []
    for c in w.columns:
        s = _clean(w[c])
        if len(s):
            out.append(_long(country, c, s, src(int(c[1:])) if callable(src) else src))
    return out


def _official(offline: bool) -> dict[str, dict[str, tuple[pd.Series, str]]]:
    """country → series → (official % series, source)."""
    off: dict[str, dict[str, tuple[pd.Series, str]]] = {c: {} for c in COUNTRIES}
    inv_es = {v: k for k, v in BDE_SERIES.items()}
    inv_bp = {v: k for k, v in BP_SERIES.items()}
    jobs = (
        ("ES", lambda: es_bde(offline), lambda t: SRC["ES"].format(t=t, code=inv_es.get(t, ""))),
        ("NL", lambda: nl_dsta(offline), lambda t: SRC["NL"].format(t=t)),
        ("BE", lambda: be_nbb(offline), lambda t: SRC["BE"].format(t=t)),
        ("PT", lambda: pt_bpstat(offline), lambda t: SRC["PT"].format(t=t, code=inv_bp.get(t, ""))),
    )
    for c, fn, src in jobs:
        try:
            w = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  egb: {c} official failed: {exc}")
            continue
        for col in w.columns:
            off[c][col] = (_clean(w[col]), src(int(col[1:])))
    try:
        bm, loans = fi_bof(offline)
        for col in bm.columns:
            off["FI"][col] = (_clean(bm[col]), SRC["FI"].format(t=int(col[1:])))
        cm = cm_from_bonds(loans, tenors=tuple(t for t in TENORS if f"y{t}" not in bm.columns))
        for col in cm.columns:
            off["FI"][col] = (_clean(cm[col]), SRC["FI_i"].format(t=int(col[1:])))
    except Exception as exc:  # noqa: BLE001
        print(f"  egb: FI official failed: {exc}")
    return off


def build(offline: bool = False) -> pd.DataFrame:
    """Long frame country, series, date, value (%), source."""
    rows: list[pd.DataFrame] = []
    off = _official(offline)
    for c in COUNTRIES:
        names = set(off[c]) | {f"y{t}" for t in CNBC_TENORS.get(c, ())}
        for name in sorted(names, key=lambda s: int(s[1:])):
            t = int(name[1:])
            s, src = off[c].get(name, (pd.Series(dtype=float), ""))
            tail = pd.Series(dtype=float)
            sym = f"{c}{t}Y-{c}"
            if t in CNBC_TENORS.get(c, ()):
                try:
                    tail = cnbc(sym, offline)
                except Exception as exc:  # noqa: BLE001
                    print(f"  egb: CNBC {sym} failed: {exc}")
            sp = splice(s, tail, src, SRC["CNBC"].format(sym=sym, t=t))
            if len(sp):
                rows.append(sp.assign(country=c, series=name))
    rows += _wide_long("DE", de_bbk(), lambda t: SRC["DE"].format(t=t))
    try:
        from issuance import fetch_ea_yields

        ea = fetch_ea_yields.daily(offline=True)
        rows.append(ea[ea["country"].isin(["FR", "IT"])])
    except Exception as exc:  # noqa: BLE001
        print(f"  egb: FR/IT cache failed: {exc}")
    try:
        yc = ecb_yc(offline)
        k = yc["KEY"].str.split(".")
        yc["cty"] = np.where(k.str[5] == "G_N_A", "EA_AAA", "EA_ALL")
        yc["kind"] = k.str[7].str.split("_").str[0]
        yc["t"] = k.str[7].str.extract(r"_(\d+)Y")[0].astype(int)
        for (cty, kind, t), g in yc.groupby(["cty", "kind", "t"]):
            s = _clean(g.set_index("date")["value"])
            m = "par" if kind == "PY" else "spot"
            rows.append(_long(cty, f"{'y' if kind == 'PY' else 'z'}{t}", s, SRC[cty].format(m=m, t=t, k=kind)))
    except Exception as exc:  # noqa: BLE001
        print(f"  egb: ECB YC failed: {exc}")
    rows = [r for r in rows if len(r)]
    if not rows:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(rows, ignore_index=True)[COLS]
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] <= _today()) & (out["date"].dt.dayofweek < 5)]
    return out.drop_duplicates(["country", "series", "date"], keep="last").sort_values(
        ["country", "series", "date"]).reset_index(drop=True)


def fetch(refresh: bool = True) -> pd.DataFrame:
    """Download (refresh) or parse the raw cache, write cache/egb_daily.parquet."""
    d = build(offline=not refresh)
    if refresh:
        try:
            ecb_irs10(offline=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  egb: ECB IRS failed: {exc}")
    if len(d):
        d.to_parquet(OUT, index=False)
    return d


def daily() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_parquet(OUT)
    return build(offline=True)


def monthly(d: pd.DataFrame | None = None, today: pd.Timestamp | None = None,
            irs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calendar-month mean of daily, complete months only (market.py convention); ECB IRS fills y10
    in months with no daily y10, labelled. Columns country, month, series, value, source, last_obs."""
    d = daily() if d is None else d
    cols = ["country", "month", "series", "value", "source", "last_obs"]
    cutoff = (_today() if today is None else pd.Timestamp(today)).to_period("M").to_timestamp()
    d = d.assign(month=pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()).sort_values("date")
    g = d.groupby(["country", "series", "month"], as_index=False).agg(
        value=("value", "mean"), source=("source", lambda s: "; ".join(dict.fromkeys(s))))
    last = d.groupby(["country", "series"])["date"].max().rename("last_obs").reset_index()
    g = g.merge(last, on=["country", "series"])
    g = g[g["month"] < cutoff]
    if irs is None:
        try:
            irs = ecb_irs10(offline=True)
        except Exception:  # noqa: BLE001
            irs = pd.DataFrame(columns=["country", "month", "value"])
    fill = []
    for c, i in irs.groupby("country"):
        have = set(g.loc[(g["country"] == c) & (g["series"] == "y10"), "month"])
        i = i[~i["month"].isin(have) & (i["month"] < cutoff)]
        if len(i):
            fill.append(pd.DataFrame({"country": c, "month": i["month"].to_numpy(), "series": "y10",
                                      "value": i["value"].to_numpy(), "source": SRC["IRS"].format(c=c),
                                      "last_obs": i["month"].max()}))
    out = pd.concat([g] + fill, ignore_index=True) if fill else g
    return out[cols].sort_values(["country", "series", "month"]).reset_index(drop=True)


def overlap(d: pd.DataFrame | None = None, irs: pd.DataFrame | None = None, series: str = "y10") -> pd.DataFrame:
    """Monthly mean of daily y10 vs ECB IRS monthly 10y, bp, split by official vs CNBC months."""
    d = daily() if d is None else d
    irs = ecb_irs10(offline=True) if irs is None else irs
    x = d[d["series"] == series].assign(month=lambda f: pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp(),
                                        kind=lambda f: np.where(f["source"].str.startswith("CNBC"), "cnbc", "official"))
    m = x.groupby(["country", "kind", "month"], as_index=False)["value"].mean()
    j = m.merge(irs.rename(columns={"value": "irs"}), on=["country", "month"])
    full = (pd.Timestamp.today().to_period("M").to_timestamp())
    j = j[j["month"] < full]
    j["diff_bp"] = 100 * (j["value"] - j["irs"])
    return j.groupby(["country", "kind"]).agg(
        n=("diff_bp", "size"), start=("month", "min"), end=("month", "max"),
        mean_bp=("diff_bp", "mean"), mean_abs_bp=("diff_bp", lambda v: v.abs().mean()),
        max_abs_bp=("diff_bp", lambda v: v.abs().max())).reset_index()


def spread_de(d: pd.DataFrame | None = None, tenor: int = 10) -> pd.DataFrame:
    """y<T> − Bund y<T> (pp) per country, daily on common dates."""
    d = daily() if d is None else d
    s = f"y{tenor}"
    x = d[d["series"] == s].pivot_table(index="date", columns="country", values="value")
    if "DE" not in x:
        return pd.DataFrame(columns=["country", "date", "value"])
    sp = x.drop(columns="DE").sub(x["DE"], axis=0)
    return sp.rename_axis(columns="country").stack().rename("value").reset_index()[["country", "date", "value"]]


def coverage(d: pd.DataFrame | None = None) -> pd.DataFrame:
    d = daily() if d is None else d
    return d.groupby(["country", "series", "source"])["date"].agg(["min", "max", "count"]).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    out = fetch(refresh=a.refresh)
    pd.set_option("display.width", 250)
    print(coverage(out).to_string())
