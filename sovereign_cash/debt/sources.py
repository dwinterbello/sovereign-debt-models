"""Official fiscal forecasts and outturns, one tidy table per source.

Every fetcher returns long rows ``country, source, vintage, basis, year, var, value, unit`` and keeps
its raw file under ``sovereign_cash/data/debt/``; a failed fetch keeps the previous cache.

  source  coverage                                    horizon   basis
  weo     IMF World Economic Outlook, all 8           2031      calendar, general government
  oecd    OECD Economic Outlook, all 8                2027      calendar, general government
  ameco   European Commission forecast (AMECO), all 8 2027      calendar, general government (EDP for EU)
  eurostat gov_10a_main S1311, DE FR IT SE NO        outturn   calendar, central government revenue, interest
  cbo     CBO ten-year baseline (US-CBO/cbo-data)     2036      US fiscal year, federal
  obr     OBR Economic and fiscal outlook, Table A.9  +5y       UK fiscal year (Apr–Mar), public sector
  cao     Cabinet Office 中長期の経済財政に関する試算   2040      JP fiscal year, national + local; general account

obr.uk sits behind a Cloudflare challenge and cbo.gov behind a bot wall; neither is worked around.
The OBR outlook is read from the copy HM Treasury lays on gov.uk, the CBO baseline from CBO's own
GitHub data repository.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.sources [--refresh] [--skip-vintages]

``--refresh`` also updates the WEO release archive (``weo_vintages.py`` → ``weo_vintages/weo_vintages.parquet``,
DBnomics 2008-04 → 2025-04 then IMF SDMX 3.0); ``weo.parquet`` remains the one live vintage the chain uses.
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "debt"
CC = ("us", "uk", "de", "fr", "it", "es", "jp", "se", "no")
ISO3 = {"us": "USA", "uk": "GBR", "de": "DEU", "fr": "FRA", "it": "ITA", "es": "ESP",
        "jp": "JPN", "se": "SWE", "no": "NOR"}
COLS = ["country", "source", "vintage", "basis", "year", "var", "value", "unit"]
UA = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}


def _get(url: str, timeout: int = 90, **kw) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers=kw.pop("headers", UA), **kw)
    r.raise_for_status()
    return r


def _cached(name: str, fn, refresh: bool) -> pd.DataFrame:
    path = RAW / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        d = fn()
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"debt.sources: {name} fetch failed ({e}); keeping cache")
            return pd.read_parquet(path)
        raise
    d = d[COLS].copy()
    d["year"] = d["year"].astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"]).reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    d.to_parquet(path, index=False)
    return d


# ---------------------------------------------------------------- IMF WEO
WEO = {"GGXWDG_NGDP": ("debt_pct", "% GDP"), "GGR_G01_GDP_PT": ("revenue_pct", "% GDP"),
       "GGXCNL_G01_GDP_PT": ("balance_pct", "% GDP"), "GGXONLB_G01_GDP_PT": ("primary_pct", "% GDP"),
       "NGDPD": ("gdp_usd_bn", "USD bn"), "LP": ("pop_mn", "mn"), "NGDP_RPCH": ("real_growth", "%"),
       "PCPIPCH": ("cpi", "%")}


def _weo() -> pd.DataFrame:
    rows = []
    vint = f"IMF WEO (datamapper, {pd.Timestamp.today():%Y-%m-%d})"
    for ind, (var, unit) in WEO.items():
        url = f"https://www.imf.org/external/datamapper/api/v1/{ind}/" + "/".join(ISO3.values())
        js = _get(url).json()
        vals = js["values"][ind]
        for cc, iso in ISO3.items():
            for yr, v in vals.get(iso, {}).items():
                rows.append((cc, "weo", vint, "cy", int(yr), var, v, unit))
    d = pd.DataFrame(rows, columns=COLS)
    # net interest = primary balance − overall balance
    w = d.pivot_table(index=["country", "year"], columns="var", values="value").reset_index()
    w["interest_pct"] = w["primary_pct"] - w["balance_pct"]
    extra = w[["country", "year", "interest_pct"]].dropna().assign(
        source="weo", vintage=vint, basis="cy", var="interest_pct", unit="% GDP").rename(columns={"interest_pct": "value"})
    return pd.concat([d, extra[COLS]], ignore_index=True)


def weo(refresh: bool = False) -> pd.DataFrame:
    return _cached("weo", _weo, refresh)


def refresh_weo_vintages(force: bool = False) -> pd.DataFrame:
    """Archive of WEO releases (DBnomics 2008-04 → 2025-04, IMF SDMX 3.0 after) → weo_vintages/weo_vintages.parquet.
    Separate from ``weo.parquet``, which stays the single live vintage the chain runs on."""
    from sovereign_cash.debt import weo_vintages as WV

    return WV.refresh_weo_vintages(force=force)


# ---------------------------------------------------------------- OECD Economic Outlook
OECD = {"GGINTP": ("interest_gross_bn", "lcu bn"), "GNINTP": ("interest_net_bn", "lcu bn"),
        "YRGT": ("revenue_bn", "lcu bn"), "NLG": ("balance_bn", "lcu bn"), "NLGX": ("primary_bn", "lcu bn"),
        "GGFL": ("debt_bn", "lcu bn"), "GDP": ("gdp_lcu_bn", "lcu bn"), "IRL": ("irl", "%"),
        "IRS": ("irs", "%"), "RFSH": ("refi_share", "%"), "GGFLQ": ("debt_oecd_pct", "% GDP")}


def _oecd() -> pd.DataFrame:
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.ECO.MAD,DSD_EO@DF_EO,/"
           f"{'+'.join(ISO3.values())}.{'+'.join(OECD)}.A?startPeriod=1995&format=csvfilewithlabels")
    d = pd.read_csv(io.StringIO(_get(url, timeout=180).text))
    vint = str(d["STRUCTURE_NAME"].iloc[0]) if "STRUCTURE_NAME" in d else "OECD EO"
    inv = {v: k for k, v in ISO3.items()}
    scale = 10.0 ** pd.to_numeric(d["UNIT_MULT"], errors="coerce").fillna(0)
    out = pd.DataFrame({
        "country": d["REF_AREA"].map(inv), "source": "oecd", "vintage": vint, "basis": "cy",
        "year": pd.to_numeric(d["TIME_PERIOD"], errors="coerce"),
        "var": d["MEASURE"].map(lambda m: OECD[m][0]), "unit": d["MEASURE"].map(lambda m: OECD[m][1]),
        "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")})
    money = out["unit"] == "lcu bn"
    out.loc[money, "value"] = out.loc[money, "value"] * scale[money] / 1e9
    return out.dropna(subset=["country", "year"])


def oecd(refresh: bool = False) -> pd.DataFrame:
    return _cached("oecd", _oecd, refresh)


# ---------------------------------------------------------------- EC AMECO
AMECO = {"1.0.319.0.UDGG": ("debt_pct", "% GDP"), "1.0.319.0.UBLGE": ("balance_pct", "% GDP"),
         "1.0.319.0.UYIGE": ("interest_gross_pct", "% GDP"), "1.0.319.0.URTG": ("revenue_pct", "% GDP"),
         "1.0.0.0.UVGD": ("gdp_lcu_bn", "lcu bn")}


def _ameco() -> pd.DataFrame:
    rows = []
    years = ",".join(str(y) for y in range(1995, 2031))
    names = {"United States": "us", "United Kingdom": "uk", "Germany": "de", "France": "fr", "Italy": "it",
             "Spain": "es", "Japan": "jp", "Sweden": "se", "Norway": "no"}
    for code, (var, unit) in AMECO.items():
        url = (f"https://ec.europa.eu/economy_finance/ameco/wq/series?fullVariable={code}"
               f"&countries={','.join(ISO3.values())}&years={years}&defaultCountries=0&Lastyear=0&Yearorder=ASC")
        t = pd.read_html(io.StringIO(_get(url).text))[0]
        for _, r in t.iterrows():
            cc = names.get(str(r["Country"]).strip())
            if not cc:
                continue
            for y in range(1995, 2031):
                if str(y) in t.columns:
                    rows.append((cc, "ameco", f"EC AMECO ({pd.Timestamp.today():%Y-%m-%d})", "cy", y, var,
                                 r[str(y)], unit))
    return pd.DataFrame(rows, columns=COLS)


def ameco(refresh: bool = False) -> pd.DataFrame:
    return _cached("ameco", _ameco, refresh)


# ---------------------------------------------------------------- Eurostat central government
def _eurostat() -> pd.DataFrame:
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/gov_10a_main?format=JSON"
           "&sector=S1311&na_item=TR&na_item=D41PAY&unit=MIO_NAC"
           "&geo=DE&geo=FR&geo=IT&geo=ES&geo=SE&geo=NO"
           "&sinceTimePeriod=1999")
    d = _get(url).json()
    dims, size = d["id"], d["size"]
    cats = {k: sorted(d["dimension"][k]["category"]["index"].items(), key=lambda kv: kv[1]) for k in dims}
    rows = []
    for k, v in d["value"].items():
        k, idx = int(k), {}
        for dim, n in zip(reversed(dims), reversed(size)):
            idx[dim] = cats[dim][k % n][0]
            k //= n
        var = {"TR": "revenue_cg_bn", "D41PAY": "interest_cg_bn"}[idx["na_item"]]
        rows.append((idx["geo"].lower(), "eurostat", "Eurostat gov_10a_main", "cy", int(idx["time"]), var,
                     v / 1e3, "lcu bn"))
    return pd.DataFrame(rows, columns=COLS)


def eurostat(refresh: bool = False) -> pd.DataFrame:
    return _cached("eurostat", _eurostat, refresh)


# ---------------------------------------------------------------- CBO
CBO_GH = "https://api.github.com/repos/US-CBO/cbo-data/contents/data/budget/{d}"
CBO_RAW = "https://raw.githubusercontent.com/US-CBO/cbo-data/main/data/budget/{d}/{f}"
CBO_VARS = {"proj_debt_held_by_public_end": ("debt_bn", "USD bn"),
            "proj_debt_avg_interest_rate": ("eff_rate", "%"),
            "proj_outlays_net_interest": ("interest_net_bn", "USD bn"),
            "proj_rev_total": ("revenue_bn", "USD bn"),
            "proj_deficit_total": ("deficit_bn", "USD bn"),
            "proj_primary_deficit": ("primary_deficit_bn", "USD bn"),
            "proj_debt_fed_reserve_holdings": ("cb_holdings_bn", "USD bn"),
            "proj_debt_held_by_public_end_gdp_share": ("debt_pct", "% GDP"),
            "proj_gdp_end_of_period": ("gdp_lcu_bn", "USD bn")}


def _cbo() -> pd.DataFrame:
    files = sorted(x["name"] for x in _get(CBO_GH.format(d="ten_year_budget")).json()
                   if x["name"].startswith("annual_fy_") and x["name"].endswith(".csv"))
    f = files[-1]
    d = pd.read_csv(io.StringIO(_get(CBO_RAW.format(d="ten_year_budget", f=f)).text))
    vint = "CBO " + f.removeprefix("annual_fy_").removesuffix(".csv")
    d = d[d["variable"].isin(CBO_VARS)]
    out = pd.DataFrame({"country": "us", "source": "cbo", "vintage": vint, "basis": "fy",
                        "year": d["date"].str.removeprefix("FY").astype(int),
                        "var": d["variable"].map(lambda v: CBO_VARS[v][0]),
                        "unit": d["variable"].map(lambda v: CBO_VARS[v][1]), "value": d["value"]})
    # CBO publishes the GDP share in per cent in some vintages and as a fraction in others
    m = out["var"].eq("debt_pct") & (out["value"] < 5)
    out.loc[m, "value"] *= 100
    return out


def cbo(refresh: bool = False) -> pd.DataFrame:
    return _cached("cbo", _cbo, refresh)


# ---------------------------------------------------------------- OBR (gov.uk copy)
OBR_ROWS = {  # Table A.9 label -> (var, unit)
    "Public sector current receipts (a)": ("revenue_pct", "% GDP"),
    "Public sector net borrowing (b-a)": ("deficit_pct", "% GDP"),
    "Primary deficit": ("primary_deficit_pct", "% GDP"),
    "Central government net cash requirement": ("cgncr_pct", "% GDP"),
    "Public sector net debt1": ("psnd_pct", "% GDP"),
    "General government gross debt": ("debt_pct", "% GDP"),
}
OBR_BN = {"Public sector net borrowing": ("deficit_bn", "GBP bn"), "Public sector net debt": ("psnd_bn", "GBP bn"),
          "Net debt interest": ("interest_net_bn", "GBP bn"), "Non-interest receipts": ("revenue_ex_int_bn", "GBP bn")}


def _obr_pdf() -> tuple[bytes, str]:
    q = _get("https://www.gov.uk/api/search.json?q=economic+and+fiscal+outlook&filter_organisations=hm-treasury"
             "&count=40&fields=title,link,public_timestamp").json()["results"]
    q += _get("https://www.gov.uk/api/search.json?q=economic+and+fiscal+outlook&count=40"
              "&fields=title,link,public_timestamp").json()["results"]
    efo = sorted({(r["public_timestamp"], r["link"], r["title"]) for r in q
                  if re.fullmatch(r"[A-Z][a-z]+ \d{4} Economic and fiscal outlook", r["title"])}, reverse=True)
    for _, link, title in efo:
        att = _get(f"https://www.gov.uk/api/content{link}").json()["details"].get("attachments", [])
        pdfs = [a["url"] for a in att if a.get("content_type") == "application/pdf"]
        if pdfs:
            return _get(pdfs[0], timeout=180).content, title.replace(" Economic and fiscal outlook", "")
    raise RuntimeError("no Economic and fiscal outlook PDF on gov.uk")


def parse_obr_a9(text: str, vint: str) -> pd.DataFrame:
    lines = text.split("\n")
    hdr = next(ln for ln in lines if re.match(r"^\d{4}-\d{2} \d{4}-\d{2}", ln.strip()))
    fys = [int(s[:4]) for s in hdr.split()]
    num = r"(-?[\d,]+\.?\d*)"
    rows, in_bn = [], False
    for ln in lines:
        s = ln.strip()
        if s == "£ billion":
            in_bn = True
        table = OBR_BN if in_bn else OBR_ROWS
        for lab, (var, unit) in table.items():
            m = re.fullmatch(re.escape(lab) + r"\s+" + r"\s+".join([num] * len(fys)), s)
            if m:
                rows += [("uk", "obr", f"OBR EFO {vint}", "fy", fy, var, float(v.replace(",", "")), unit)
                         for fy, v in zip(fys, m.groups())]
    return pd.DataFrame(rows, columns=COLS)


def _vintage_date(v: str) -> pd.Timestamp | None:
    """'October 2026' or 'OBR EFO October 2026' -> 2026-10-01; None if unreadable."""
    m = re.search(r"([A-Z][a-z]+) (\d{4})", str(v))
    try:
        return pd.Timestamp(f"{m.group(1)} 1 {m.group(2)}") if m else None
    except ValueError:
        return None


def _obr() -> pd.DataFrame:
    """Table A.9 from the newest EFO gov.uk returns. Never steps back to an older EFO than the cache holds:
    gov.uk's search does not always return the latest one (on 2 Nov 2026 it dropped October 2026 and the
    nightly refresh replaced it with March 2026), so an older find is treated as a failed fetch."""
    import pdfplumber

    pdf, vint = _obr_pdf()
    cached = RAW / "obr.parquet"
    if cached.exists():
        have = pd.read_parquet(cached)["vintage"].dropna()
        have_d = max((d for d in map(_vintage_date, have.unique()) if d is not None), default=None)
        got_d = _vintage_date(vint)
        if have_d is not None and (got_d is None or got_d < have_d):
            raise RuntimeError(f"gov.uk returned the {vint} EFO, older than the cached {have_d:%B %Y}")
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        for p in doc.pages:
            t = p.extract_text() or ""
            if "Table A.9: Fiscal aggregates" in t and "changes since" not in t.split("\n")[1]:
                d = parse_obr_a9(t, vint)
                if len(d):
                    RAW.mkdir(parents=True, exist_ok=True)
                    (RAW / "obr_efo.pdf").write_bytes(pdf)
                    return d
    raise RuntimeError("Table A.9 not found in the EFO PDF")


def obr(refresh: bool = False) -> pd.DataFrame:
    return _cached("obr", _obr, refresh)


# ---------------------------------------------------------------- Cabinet Office (JP)
CAO_URL = "https://www5.cao.go.jp/keizai2/keizai-syakai/shisan/keisuuhyo.xlsx"
CAO_CASES = {"現状投影ケース": "baseline", "成長戦略実現ケース①": "growth1", "成長戦略実現ケース②": "growth2"}
CAO_MAIN = {"名目GDP": ("gdp_lcu_bn", "JPY tn"), "名目長期金利": ("irl", "%"), "実効金利": ("eff_rate", "%"),
            "国・地方の基礎的財政収支（対名目GDP比）": ("primary_pct", "% GDP"),
            "国・地方の純利払費（対名目GDP比）": ("interest_pct", "% GDP"),
            "国・地方の財政収支（対名目GDP比）": ("balance_pct", "% GDP"),
            "国・地方の公債等残高（対名目GDP比）": ("debt_pct", "% GDP")}


CAO_GA = {"税収等": ("revenue_cg_bn", "JPY bn"), "税収": ("tax_cg_bn", "JPY bn"),
          "その他収入": ("other_rev_cg_bn", "JPY bn"), "うち利払費": ("interest_cg_bn", "JPY bn"),
          "うち債務償還費": ("redemption_cg_bn", "JPY bn")}
CAO_GA_SHEETS = {"財政の詳細計数表（現状投影・長期）": "baseline", "財政の詳細計数表（成長戦略実現①・長期）": "growth1",
                 "財政の詳細計数表（成長戦略実現②・長期）": "growth2"}


def _cao_general_account(xlsx: bytes) -> list[tuple]:
    """National general account (国の一般会計) revenue and interest, 兆円 -> JPY bn."""
    rows = []
    book = pd.read_excel(io.BytesIO(xlsx), sheet_name=None, header=None)
    for sheet, case in CAO_GA_SHEETS.items():
        d = book.get(sheet)
        if d is None:
            continue
        start = next(i for i in range(len(d)) if "国の一般会計" in str(d.iat[i, 0]))
        hdr = next(i for i in range(start, len(d)) if str(d.iat[i, 0]).replace("\u3000", "") == "年度")
        years = {j: int(float(v)) for j, v in enumerate(d.iloc[hdr]) if j >= 3 and pd.notna(v)}
        for i in range(hdr + 1, min(hdr + 14, len(d))):
            labs = [str(d.iat[i, c]).strip() for c in range(3)]
            for lab in labs:
                if lab in CAO_GA:
                    var, unit = CAO_GA[lab]
                    for j, y in years.items():
                        v = pd.to_numeric(str(d.iat[i, j]).replace("▲", "-").strip("()"), errors="coerce")
                        rows.append(("jp", f"cao_{case}", "CAO", "fy", y, var, v * 1000, unit))
                    break
    return rows


def parse_cao(xlsx: bytes) -> pd.DataFrame:
    d = pd.read_excel(io.BytesIO(xlsx), sheet_name=0, header=None)
    rows, case, years = _cao_general_account(xlsx), None, None
    for i in range(len(d)):
        c1 = d.iat[i, 1] if d.shape[1] > 1 else None
        if isinstance(c1, str) and c1.strip() in CAO_CASES:
            case = CAO_CASES[c1.strip()]
        lab = str(d.iat[i, 0]).strip()
        if lab.replace("\u3000", "") == "年度":
            years = {j: int(float(v)) for j, v in enumerate(d.iloc[i]) if j >= 3 and pd.notna(v)}
        if case and years and lab in CAO_MAIN:
            var, unit = CAO_MAIN[lab]
            for j, y in years.items():
                v = pd.to_numeric(str(d.iat[i, j]).replace("▲", "-").strip("()"), errors="coerce")
                if var == "gdp_lcu_bn":
                    v = v * 1000 if pd.notna(v) else v  # 兆円 -> JPY bn
                    unit = "JPY bn"
                rows.append(("jp", f"cao_{case}", "CAO", "fy", y, var, v, unit))
    return pd.DataFrame(rows, columns=COLS)


def _cao() -> pd.DataFrame:
    x = _get(CAO_URL).content
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "cao_keisuuhyo.xlsx").write_bytes(x)
    d = parse_cao(x)
    vint = "CAO " + pd.Timestamp.today().strftime("%Y-%m")
    page = _get("https://www5.cao.go.jp/keizai2/keizai-syakai/shisan.html").text
    m = re.search(r"shisan/(\d{2})(\d{2})hontai\.pdf", page)
    if m:
        vint = f"CAO 20{m.group(1)}-{m.group(2)}"
    d["vintage"] = vint
    return d


def cao(refresh: bool = False) -> pd.DataFrame:
    return _cached("cao", _cao, refresh)


# ---------------------------------------------------------------- all
# ---------------------------------------------------------------- outturn history (US, UK)
MTS9 = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_9"


def _us_net_interest_fy() -> pd.Series:
    """Budget function 900 'Net Interest', $bn by fiscal year (Treasury MTS Table 9, September cumulative).

    FRED FYOINT is gross interest on Treasury securities (intragovernmental trust-fund interest included):
    FY2016 $430bn / FY2024 $1,133bn against net interest of $241bn / $882bn. CBO's baseline and the IMF's
    general-government series are net, so the outturn history must be too.
    """
    r = _get(MTS9, params={
        "filter": "record_calendar_month:eq:09,classification_desc:eq:Net Interest",
        "fields": "record_fiscal_year,current_fytd_rcpt_outly_amt", "sort": "record_fiscal_year",
        "page[size]": 100})
    rows = r.json()["data"]
    s = pd.Series({int(x["record_fiscal_year"]): float(x["current_fytd_rcpt_outly_amt"]) / 1e9 for x in rows})
    if s.empty or s.index.max() < 2024:
        raise ValueError("MTS Table 9 net interest missing recent fiscal years")
    return s


def _us_hist() -> pd.DataFrame:
    """Federal receipts (FRED FYFR), net interest (Treasury MTS Table 9) and gross interest outlays (FRED FYOINT)
    by fiscal year, $bn."""
    t = pd.read_csv(io.StringIO(_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FYFR,FYOINT").text))
    t["year"] = pd.to_datetime(t.iloc[:, 0]).dt.year
    rows = []
    for col, var in (("FYFR", "revenue_bn"), ("FYOINT", "interest_gross_bn")):
        for y, v in zip(t["year"], pd.to_numeric(t[col], errors="coerce") / 1e3):
            rows.append(("us", "us_hist", "Treasury MTS via FRED", "fy", y, var, v, "USD bn"))
    # FYOINT is net interest through FY2014 (FY1995 $232bn, FY2014 $229bn = OMB function 900) and gross
    # from FY2015 (FY2015 $402bn vs net $223bn): splice net = FYOINT <= 2014, MTS Table 9 from FY2015.
    for y, v in zip(t["year"], pd.to_numeric(t["FYOINT"], errors="coerce") / 1e3):
        if y <= 2014 and pd.notna(v):
            rows.append(("us", "us_hist", "Treasury MTS via FRED (net to FY2014)", "fy", y, "interest_net_bn", v, "USD bn"))
    for y, v in _us_net_interest_fy().items():
        if y >= 2015:
            rows.append(("us", "us_hist", "Treasury MTS Table 9", "fy", y, "interest_net_bn", v, "USD bn"))
    return pd.DataFrame(rows, columns=COLS)


ONS_PSF = "https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/{s}/pusf/data"


def _uk_hist() -> pd.DataFrame:
    """Public sector current receipts (ANBT) and central-government net interest (NMFX), financial years
    from the monthly ONS public sector finances, £m; FY labelled by its start year like the OBR."""
    rows = []
    for s, var in (("anbt", "revenue_bn"), ("nmfx", "interest_net_bn")):
        m = pd.DataFrame(_get(ONS_PSF.format(s=s), headers={"User-Agent": "Mozilla/5.0"}).json()["months"])
        m["d"] = pd.to_datetime(m["date"], format="%Y %b")
        m["fy"] = m["d"].dt.year - (m["d"].dt.month < 4)
        g = m.assign(v=pd.to_numeric(m["value"], errors="coerce") / 1e3).groupby("fy")["v"].agg(["sum", "count"])
        for fy, r in g[g["count"] == 12].iterrows():
            rows.append(("uk", "uk_hist", "ONS public sector finances", "fy", fy, var, r["sum"], "GBP bn"))
    return pd.DataFrame(rows, columns=COLS)


ONS_RPI = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/{s}/mm23/data"
UK_RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "uk"
NEG_TYPES = ("Cancellation", "Reverse Auction", "Cancellation Adjustment", "Cancellation from 12-Oct-99 Switch Auction")


def _rpi() -> pd.Series:
    """RPI monthly, Jan-1987 = 100: CHAW from 1987, CDKO (Jan-1974 = 100, 394.5 in Jan-1987) before."""
    out = {}
    for s in ("cdko", "chaw"):
        m = pd.DataFrame(_get(ONS_RPI.format(s=s), headers={"User-Agent": "Mozilla/5.0"}).json()["months"])
        out[s] = pd.Series(pd.to_numeric(m["value"]).values, index=pd.to_datetime(m["date"], format="%Y %b"))
    return out["chaw"].combine_first(out["cdko"] / 3.945).sort_index()


def _ref_rpi(rpi: pd.Series, d: pd.Timestamp, lag: int) -> float:
    """Reference RPI on date d: 3-month gilts interpolate RPI(m−3)→RPI(m−2) by day of month; 8-month gilts
    take RPI(m−8)."""
    m = pd.Timestamp(d.year, d.month, 1)
    a = rpi.get(m - pd.DateOffset(months=lag), np.nan)
    if lag == 8:
        return float(a)
    b = rpi.get(m - pd.DateOffset(months=lag - 1), np.nan)
    return float(a + (d.day - 1) / d.days_in_month * (b - a))


def _gilt_xml(name: str, tag: str) -> pd.DataFrame:
    t = (UK_RAW / name).read_text(encoding="utf-8", errors="ignore")
    return pd.DataFrame([dict(re.findall(r'(\w+)="([^"]*)"', el)) for el in re.findall(rf"<{tag}[^>]*/>", t)])


def _uk_uplift() -> pd.DataFrame:
    """Inflation uplift on index-linked gilts (£bn) at each year end and today, bottom-up per gilt: nominal from
    the DMO issuance history (D2.1E, every IL gilt since 1981, cancellations / reverse auctions netted) anchored
    on D1A's amount in issue for live gilts; index ratio = reference RPI / base, base from D1A's uplift for live
    gilts and the reference RPI at first issue otherwise."""
    rpi = _rpi()
    h = _gilt_xml("D2_1E.xml", "View_Gilt_Issuance_History")
    h = h[h["INSTRUMENT_NAME"].str.contains("Index", case=False)].copy()
    h["d"] = pd.to_datetime(h["ACTUAL_DATE"])
    h["n"] = pd.to_numeric(h["NOMINAL_ISSUED"]) * np.where(h["ISSUANCE_TYPE"].isin(NEG_TYPES), -1, 1)
    live = _gilt_xml("D1A.xml", "View_GILTS_IN_ISSUE")
    live = live[live["INSTRUMENT_TYPE"].str.contains("Index")].set_index("ISIN_CODE")
    asof = pd.to_datetime(live["CLOSE_OF_BUSINESS_DATE"].iloc[0])
    years = range(2000, asof.year + 1)
    dates = [pd.Timestamp(y, 12, 31) for y in years if pd.Timestamp(y, 12, 31) < asof] + [asof]
    up = dict.fromkeys(dates, 0.0)
    nom = dict.fromkeys(dates, 0.0)
    for isin, g in h.groupby("ISIN_CODE"):
        lag = 8 if g["INDEXATION_LAG"].iloc[0].startswith("8") else 3
        first = g["d"].min()
        if isin in live.index:
            r = live.loc[isin]
            mat = pd.to_datetime(r["REDEMPTION_DATE"])
            n_now = float(r["TOTAL_AMOUNT_IN_ISSUE"])
            base = _ref_rpi(rpi, asof, lag) / (float(r["TOTAL_AMOUNT_INCLUDING_IL_UPLIFT"]) / n_now)
        else:
            # not in D1A, so redeemed by the close; mid-year redemptions are gone by that year's end
            mat = min(pd.Timestamp(int(re.search(r"(19|20)\d\d", g["INSTRUMENT_NAME"].iloc[0]).group()), 12, 31), asof)
            n_now, base = None, _ref_rpi(rpi, first, lag)
        for d in dates:
            if d < first or d >= mat:
                continue
            n = (n_now - g.loc[g["d"] > d, "n"].sum()) if n_now is not None else g.loc[g["d"] <= d, "n"].sum()
            nom[d] += n / 1e3
            up[d] += n / 1e3 * (_ref_rpi(rpi, d, lag) / base - 1)
    rows = [("uk", "uk_uplift", "DMO D1A / D2.1E + ONS RPI", "dec" if d != asof else "asof", d.year,
             "il_uplift_bn" if d != asof else "il_uplift_now_bn", up[d], "GBP bn") for d in dates]
    rows += [("uk", "uk_uplift", "DMO D1A / D2.1E + ONS RPI", "dec" if d != asof else "asof", d.year,
              "il_nominal_bn" if d != asof else "il_nominal_now_bn", nom[d], "GBP bn") for d in dates]
    yoy = rpi.iloc[-1] / rpi.iloc[-13] - 1
    rows.append(("uk", "uk_uplift", "ONS RPI", "asof", asof.year, "rpi_yoy", 100 * yoy, "%"))
    return pd.DataFrame(rows, columns=COLS)


def uk_uplift(refresh: bool = False) -> pd.DataFrame:
    return _cached("uk_uplift", _uk_uplift, refresh)


MSPD1 = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1"


def _official_stock() -> pd.DataFrame:
    """Month-end marketable stock incl. inflation uplift (local bn) and its index-linked part, from the
    issuer's own statement: US MSPD (total marketable, TIPS incl. inflation compensation), France INSEE /
    AFT (negotiable debt, indexed incl. supplement). Lines and coupon books carry linkers at par."""
    rows = []
    recs, page = [], 1
    while True:
        j = _get(MSPD1, params={"filter": "record_date:gte:2011-12-31", "page[size]": 10000, "page[number]": page,
                                "fields": "record_date,security_type_desc,security_class_desc,total_mil_amt"}).json()
        recs += j["data"]
        if page >= j["meta"]["total-pages"]:
            break
        page += 1
    m = pd.DataFrame(recs)
    m["d"] = pd.to_datetime(m["record_date"])
    m["v"] = pd.to_numeric(m["total_mil_amt"]) / 1e3
    tot = m[m["security_type_desc"] == "Total Marketable"].set_index("d")["v"]
    il = m[m["security_class_desc"] == "Treasury Inflation-Protected Securities"].set_index("d")["v"]
    for var, ser in (("marketable_bn", tot), ("linker_bn", il)):
        rows += [("us", "official_stock", "Treasury MSPD", f"{d:%Y-%m}", d.year, var, float(v), "USD bn")
                 for d, v in ser.items()]
    from sovereign_cash.fr import fetch as FR
    fr = FR.load_debt(True)
    for var, col in (("marketable_bn", "debt_total"), ("linker_bn", "indexed")):
        rows += [("fr", "official_stock", "INSEE / AFT negotiable debt", f"{d:%Y-%m}", d.year, var, float(v), "EUR bn")
                 for d, v in fr[col].dropna().items()]
    return pd.DataFrame(rows, columns=COLS)


def official_stock(refresh: bool = False) -> pd.DataFrame:
    return _cached("official_stock", _official_stock, refresh)


def us_hist(refresh: bool = False) -> pd.DataFrame:
    return _cached("us_hist", _us_hist, refresh)


def uk_hist(refresh: bool = False) -> pd.DataFrame:
    return _cached("uk_hist", _uk_hist, refresh)


NB_KF = "https://data.norges-bank.no/api/data/GOVT_KEYFIGURES/?format=csv&locale=en&startPeriod=2000-01-01"


def _no_debt() -> pd.DataFrame:
    """Norwegian government bonds and bills outside the government's own stock (nominal value less own
    holdings), Norges Bank key figures, quarter ends, NOK mn -> bn; `basis` carries the date."""
    d = pd.read_csv(io.StringIO(_get(NB_KF).text), sep=";", thousands=",")
    d["t"] = pd.to_datetime(d["TIME_PERIOD"])
    v = d[d["UNIT_MEASURE"].isin(["N_V", "V_O"])].pivot_table(index="t", columns=["INSTRUMENT_TYPE", "UNIT_MEASURE"],
                                                              values="OBS_VALUE")
    rows = []
    for ins, var in (("GBON", "bonds_mkt_bn"), ("TBIL", "bills_mkt_bn")):
        s = (v[(ins, "N_V")] - v[(ins, "V_O")].fillna(0)) / 1e3
        for t, x in s.dropna().items():
            rows.append(("no", "nb_debt", "Norges Bank key figures", f"{t:%Y-%m-%d}", t.year, var, x, "NOK bn"))
    return pd.DataFrame(rows, columns=COLS)


NB_PM = "https://data.norges-bank.no/api/data/GOVT_PRIMARY_MARKET/?format=csv&locale=en"


def _no_interest() -> pd.DataFrame:
    """Cash interest on Norwegian government debt held outside the government, by calendar year: each NGB's
    outstanding (cumulated issue / buyback volumes, own stock excluded) × coupon, plus bills × the year's
    mean auction yield. Months are month ends; the current year covers the months published so far."""
    d = pd.read_csv(io.StringIO(_get(NB_PM, timeout=180).text), sep=";", low_memory=False)
    d["t"] = pd.to_datetime(d["TIME_PERIOD"], errors="coerce")
    d["v"] = pd.to_numeric(d["OBS_VALUE"].astype(str).str.replace(",", ""), errors="coerce")
    d["mat"] = pd.to_datetime(d["MATURITY"], errors="coerce")
    cv = d[(d["UNIT_MEASURE"] == "C_V") & (d["ISSUE_TYPE"] != "OWN")]
    ends = pd.date_range("2000-01-31", pd.Timestamp.today(), freq="ME")
    out = []
    for ins in ("GBON", "TBIL"):
        x = cv[cv["INSTRUMENT_TYPE"] == ins]
        cum = x.pivot_table(index="t", columns="ISIN", values="v", aggfunc="sum").sort_index().cumsum()
        stock = cum.reindex(cum.index.union(ends)).ffill().reindex(ends).fillna(0.0)
        mat = x.groupby("ISIN")["mat"].first()
        alive = pd.DataFrame({i: ends < mat[i] for i in stock.columns}, index=ends)
        stock = stock.where(alive, 0.0)
        if ins == "GBON":
            cpn = pd.to_numeric(x.groupby("ISIN")["COUPON"].first(), errors="coerce") / 100
            out.append((stock * cpn).sum(axis=1).rename("bonds"))
        else:
            y = d[(d["INSTRUMENT_TYPE"] == "TBIL") & (d["UNIT_MEASURE"] == "Y") & (d["ISSUE_TYPE"] == "AU")]
            yy = y.groupby(y["t"].dt.year)["v"].mean() / 100
            out.append((stock.sum(axis=1) * stock.index.year.map(yy).astype(float)).rename("bills"))
        if ins == "GBON":
            last = stock.iloc[-1]
            sched = last[last > 0].groupby(mat[last[last > 0].index].dt.year).sum() / 1e3
            iss = x[(x["v"] > 0) & (x["t"] >= pd.Timestamp.today() - pd.DateOffset(years=2))]
            tenor = iss.groupby(((iss["mat"] - iss["t"]).dt.days / 365.25).round().clip(lower=1))["v"].sum() / 1e3
    m = pd.concat(out, axis=1).fillna(0.0)
    m = m[m.sum(axis=1) > 0]
    a = m.groupby(m.index.year).mean().sum(axis=1) / 1e3  # NOK mn annual rate -> bn
    rows = [("no", "nb_interest", "Norges Bank primary market", "cy", int(yr), "interest_cg_bn", float(v), "NOK bn")
            for yr, v in a.items()]
    rows += [("no", "nb_interest", "Norges Bank primary market", "maturity", int(yr), "redemption_sched_bn", float(v),
              "NOK bn") for yr, v in sched.items()]
    rows += [("no", "nb_interest", "Norges Bank primary market", "tenor", int(t), "issued_2y_by_tenor_bn", float(v),
              "NOK bn") for t, v in tenor.items()]
    return pd.DataFrame(rows, columns=COLS)


def no_debt(refresh: bool = False) -> pd.DataFrame:
    return _cached("no_debt", _no_debt, refresh)


def no_interest(refresh: bool = False) -> pd.DataFrame:
    return _cached("no_interest", _no_interest, refresh)


FETCHERS = {"weo": weo, "oecd": oecd, "ameco": ameco, "eurostat": eurostat, "cbo": cbo, "obr": obr, "cao": cao,
            "us_hist": us_hist, "uk_hist": uk_hist, "no_debt": no_debt, "no_interest": no_interest,
            "uk_uplift": uk_uplift, "official_stock": official_stock}


def load(refresh: bool = False) -> pd.DataFrame:
    out = []
    for name, fn in FETCHERS.items():
        try:
            out.append(fn(refresh))
        except Exception as e:  # noqa: BLE001
            print(f"debt.sources: {name} unavailable ({e})")
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-vintages", action="store_true", help="do not refresh the WEO vintage archive")
    a = ap.parse_args()
    d = load(a.refresh)
    print(d.groupby(["source", "vintage"]).agg(n=("value", "size"), y0=("year", "min"), y1=("year", "max"),
                                               cc=("country", lambda s: ",".join(sorted(set(s))))).to_string())
    print(np.unique(d["var"]))
    if a.refresh and not a.skip_vintages:
        try:
            v = refresh_weo_vintages(force=False)  # per-release caches; new releases are fetched, old ones kept
            print(f"weo vintages: {v['vintage_label'].nunique()} releases {v['vintage_label'].min()} → {v['vintage_label'].max()}")
        except Exception as e:  # noqa: BLE001
            print(f"debt.sources: weo vintages unavailable ({e})")
