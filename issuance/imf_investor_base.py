"""IMF Sovereign Debt Investor Base for Advanced Economies (Arslanalp & Tsuda, WP/12/284).

Quarterly estimates of who holds general-government gross debt (all instruments, consolidated),
2004Q1 onward, in six investor classes: domestic central bank (dcb), domestic banks (db), domestic
nonbanks (dnb), foreign official (fo), foreign banks (fb), foreign nonbanks (fnb). Domestic and
foreign nonbanks are residuals (domestic total − dcb − db; foreign total − fo − fb). Holdings are
at face value or valuation-adjusted to remove price effects, so `basis` is always 'nominal'; the
workbook publishes no market-value series.

Publication: the workbook says "regularly updated at end-April and end-October" (this vintage:
1 May 2026, data to 2025Q4). A quarter enters the first release at least four months after it
ends (Q4 → end-April, Q2 → end-October, Q1/Q3 wait for the next one), so every quarter is stamped
at the month after that release (May or November) and forward-filled until the next public point:
a 5–8 month lag (6.5 on average). Only the latest vintage is downloadable, so history carries
revisions the market did not see; each downloaded vintage is kept in the cache so a vintage panel
can be built going forward.

`metrics()` returns rows in `issuance.headline.panel()` shape (country, month, metric, value,
phase='history'), stamped at the publication month, so `headline.lag_for`'s default 1-month lag
on top is conservative.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE

URL = "https://www.imf.org/-/media/websites/imf/imported-datasets/external/pubs/ft/wp/2012/data/_wp12284.zip"
PAGE = ("https://www.imf.org/en/publications/wp/issues/2016/12/31/"
        "tracking-global-demand-for-advanced-economy-sovereign-debt-40135")
DIR = CACHE / "imf_investor_base"
ZIP = DIR / "_wp12284.zip"
SOURCE = "IMF Sovereign Debt Investor Base for Advanced Economies (Arslanalp & Tsuda), general government gross debt"

COUNTRIES = {"United Kingdom": "UK", "Germany": "DE", "France": "FR", "Italy": "IT", "Spain": "ES",
             "United States": "US", "Japan": "JP", "Sweden": "SE", "Norway": "NO"}
SHEETS = {
    "total": "Table 1. Total",
    "fo": "Table 2.1 ForeignOfficial",
    "fb": "Table 2.2 ForeignBank",
    "fnb": "Table 2.3 ForeignNonbank",
    "dcb": "Table 3.1 DomesticCentralBank",
    "db": "Table 3.2 DomesticBank",
    "dnb": "Table 3.3 DomesticNonbank",
}
TYPES = ("dcb", "db", "dnb", "fo", "fb", "fnb")
TYPE_LABEL = {"dcb": "Domestic central bank", "db": "Domestic banks", "dnb": "Domestic nonbanks",
              "fo": "Foreign official", "fb": "Foreign banks", "fnb": "Foreign nonbanks", "total": "Total"}
# Arslanalp & Tsuda (2012) Table 6: 0–100 risk score per investor class from the average correlation
# of log holdings changes with log 10y yield changes, 2008Q3–2011Q4 (highest positive → 0, most
# negative → 100). IRI = share-weighted mean score.
IRI_SCORE = {"dcb": 0, "db": 26, "dnb": 16, "fo": 14, "fb": 87, "fnb": 100}
# Fang, Hardy & Lewis (2025 RFS / NBER w30087): private nonbanks absorb ~70% of new debt and are the
# most price-responsive holders; for advanced economies domestic and foreign nonbanks carry the same
# marginal-absorption coefficient (0.36 each). Banks are less elastic, official holders inelastic.
ELASTIC = ("dnb", "fnb")
PRIVATE = ("db", "dnb", "fb", "fnb")
RELEASE_MONTHS = (4, 10)
RELEASE_MIN_LAG = 4

METRICS: dict[str, dict] = {
    "fnb_share": {"label": "Foreign nonbank share", "unit": "% of gov. debt", "sign": 1, "fmt": ".1f",
                  "what": "Foreign nonbank holdings ÷ general-government gross debt (IMF investor base). The most "
                          "price-elastic and flight-prone holder (Fang–Hardy–Lewis; Arslanalp–Tsuda risk score 100)."},
    "elastic_share": {"label": "Elastic-holder share", "unit": "% of gov. debt", "sign": 1, "fmt": ".1f",
                      "what": "Domestic + foreign nonbank holdings ÷ gross debt. Fang–Hardy–Lewis: private nonbanks "
                              "absorb ~70% of new debt and are the most yield-responsive holders; in advanced "
                              "economies the domestic and foreign nonbank coefficients are equal (0.36). Banks and "
                              "official holders excluded as less elastic."},
    "cb_hold_share": {"label": "Domestic central-bank share", "unit": "% of gov. debt", "sign": -1, "fmt": ".1f",
                      "what": "Domestic central-bank holdings ÷ gross debt (IMF investor base; euro area = the "
                              "national central bank's Eurosystem holdings). Inelastic holder: more of it, less "
                              "duration for private balance sheets."},
    "ibr": {"label": "Investor-base risk index", "unit": "0–100", "sign": 1, "fmt": ".0f",
            "what": "Arslanalp–Tsuda IRI: Σ share × risk score, scores fixed at their Table 6 (dcb 0, db 26, dnb 16, "
                    "foreign official 14, foreign banks 87, foreign nonbanks 100). 0 = all held by the central bank, "
                    "100 = all by foreign nonbanks. Scores were estimated on 2008Q3–2011Q4 yields."},
    "d_dcb_gdp": {"label": "Δ4q central-bank holdings / GDP", "unit": "% of GDP", "sign": -1, "fmt": ".1f",
                  "what": "Four-quarter change in domestic central-bank holdings ÷ annual nominal GDP."},
    "d_fo_gdp": {"label": "Δ4q foreign official holdings / GDP", "unit": "% of GDP", "sign": -1, "fmt": ".1f",
                 "what": "Four-quarter change in foreign official (reserve-manager) holdings ÷ GDP. Inelastic demand."},
    "d_db_gdp": {"label": "Δ4q domestic bank holdings / GDP", "unit": "% of GDP", "sign": 1, "fmt": ".1f",
                 "what": "Four-quarter change in domestic bank holdings ÷ GDP: net supply banks had to absorb."},
    "d_dnb_gdp": {"label": "Δ4q domestic nonbank holdings / GDP", "unit": "% of GDP", "sign": 1, "fmt": ".1f",
                  "what": "Four-quarter change in domestic nonbank holdings ÷ GDP: net supply placed with elastic "
                          "domestic holders."},
    "d_fb_gdp": {"label": "Δ4q foreign bank holdings / GDP", "unit": "% of GDP", "sign": 1, "fmt": ".1f",
                 "what": "Four-quarter change in foreign bank holdings ÷ GDP."},
    "d_fnb_gdp": {"label": "Δ4q foreign nonbank holdings / GDP", "unit": "% of GDP", "sign": 1, "fmt": ".1f",
                  "what": "Four-quarter change in foreign nonbank holdings ÷ GDP: net supply placed with the most "
                          "elastic holder."},
    "d_priv_gdp": {"label": "Δ4q private holdings / GDP", "unit": "% of GDP", "sign": 1, "fmt": ".1f",
                   "what": "Four-quarter change in bank + nonbank (domestic and foreign) holdings ÷ GDP: net supply "
                           "left to private balance sheets after official absorption."},
}


# ---------------------------------------------------------------- fetch
def fetch(refresh: bool = False) -> Path:
    """Path to the cached workbook; downloads the zip if absent or `refresh`. A failed download
    falls back to the cache. Each vintage is also kept as `investor_base_<YYYY-MM-DD>.xlsx`."""
    DIR.mkdir(parents=True, exist_ok=True)
    if refresh or not ZIP.exists():
        try:
            from issuance.http import _browser_get

            r = _browser_get(URL, timeout=120)
            if r.status_code != 200 or not r.content.startswith(b"PK"):
                raise RuntimeError(f"IMF investor base: HTTP {r.status_code}, {len(r.content)} bytes")
            ZIP.write_bytes(r.content)
        except Exception:
            if not ZIP.exists():
                raise
    with zipfile.ZipFile(ZIP) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        out = DIR / Path(name).name
        out.write_bytes(z.read(name))
    v = vintage(out)
    if v is not None:
        keep = DIR / f"investor_base_{v.date()}.xlsx"
        if not keep.exists():
            shutil.copyfile(out, keep)
    return out


def _workbook() -> Path:
    xs = sorted(p for p in DIR.glob("*.xlsx") if not p.name.startswith("investor_base_")) if DIR.exists() else []
    return xs[0] if xs else fetch()


def vintage(path: Path | None = None) -> pd.Timestamp | None:
    """'This Version: <date>' from the ReadMe sheet."""
    path = _workbook() if path is None else path
    r = pd.read_excel(path, "ReadMe", header=None)
    for v in r.to_numpy().ravel():
        m = re.search(r"This Version:\s*([A-Za-z]+ \d{1,2}, \d{4})", str(v))
        if m:
            return pd.Timestamp(m.group(1))
    return None


# ---------------------------------------------------------------- parse
def _quarter_end(label) -> pd.Timestamp | None:
    m = re.fullmatch(r"(\d{4})Q([1-4])", str(label).strip())
    if not m:
        return None
    return pd.Period(f"{m.group(1)}Q{m.group(2)}", "Q").to_timestamp(how="end").normalize()


def _block(raw: pd.DataFrame, header: str | None = None) -> pd.DataFrame:
    """First country block of a sheet (rows under `header`, or the top block): long
    (name, quarter_end, value). Stops at the first row that is not a country."""
    start = 0 if header is None else int(raw.index[raw[0].astype(str).str.strip() == header][0])
    quarters = {j: _quarter_end(raw.iat[start, j]) for j in range(1, raw.shape[1])}
    quarters = {j: q for j, q in quarters.items() if q is not None}
    rows = []
    for i in range(start + 1, len(raw)):
        name = raw.iat[i, 0]
        if not isinstance(name, str) or _quarter_end(name) or name.strip() in ("o/w securities",) or \
                name.lower().startswith(("percent", "billion", "total", "national")):
            break
        for j, q in quarters.items():
            rows.append((name.strip(), q, pd.to_numeric(raw.iat[i, j], errors="coerce")))
    return pd.DataFrame(rows, columns=["name", "quarter_end", "value"])


def parse(path: Path | None = None, countries: dict[str, str] | None = COUNTRIES) -> pd.DataFrame:
    """Tidy quarterly long frame: country, quarter_end, holder_type, value_local_bn, value_usd_bn,
    share_pct, gdp_local_bn, basis, source, vintage. `countries=None` keeps every IMF name."""
    path = _workbook() if path is None else path
    xl = pd.ExcelFile(path)
    parts = []
    for typ, sheet in SHEETS.items():
        b = _block(xl.parse(sheet, header=None))
        parts.append(b.assign(holder_type=typ))
    long = pd.concat(parts, ignore_index=True)
    fxraw = xl.parse("FX", header=None)
    fx = _block(fxraw).rename(columns={"value": "lc_per_usd"})
    dg = _block(fxraw, "Total Debt/GDP").rename(columns={"value": "debt_gdp_pct"})
    long = long.merge(fx, on=["name", "quarter_end"], how="left")
    tot = long[long["holder_type"] == "total"][["name", "quarter_end", "value"]].rename(columns={"value": "total"})
    long = long.merge(tot, on=["name", "quarter_end"], how="left").merge(dg, on=["name", "quarter_end"], how="left")
    if countries is not None:
        long = long[long["name"].isin(list(countries))]
        long["country"] = long["name"].map(countries)
    else:
        long["country"] = long["name"]
    long = long.dropna(subset=["value"])
    out = pd.DataFrame({
        "country": long["country"],
        "quarter_end": long["quarter_end"],
        "holder_type": long["holder_type"],
        "value_local_bn": long["value"],
        "value_usd_bn": long["value"] / long["lc_per_usd"].where(long["lc_per_usd"] > 0),
        "share_pct": 100 * long["value"] / long["total"].where(long["total"] > 0),
        "gdp_local_bn": long["total"] / (long["debt_gdp_pct"].where(long["debt_gdp_pct"] > 0) / 100),
        "basis": "nominal",
        "source": SOURCE,
        "vintage": vintage(path),
    })
    return out.sort_values(["country", "holder_type", "quarter_end"]).reset_index(drop=True)


_PANEL: dict[str, pd.DataFrame] = {}


def panel(refresh: bool = False) -> pd.DataFrame:
    """`parse()` of the cached workbook (downloaded on first use)."""
    if refresh:
        fetch(refresh=True)
        _PANEL.clear()
    if "p" not in _PANEL:
        _PANEL["p"] = parse(_workbook())
    return _PANEL["p"].copy()


# ---------------------------------------------------------------- publication stamping
def public_month(quarter_end: pd.Timestamp) -> pd.Timestamp:
    """First month the quarter is public: the month after the first end-April / end-October release
    at least RELEASE_MIN_LAG months after the quarter ends."""
    q = pd.Timestamp(quarter_end).to_period("M")
    m = q + RELEASE_MIN_LAG
    while m.month not in RELEASE_MONTHS:
        m += 1
    return (m + 1).to_timestamp()


def wide(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """country × quarter_end with one column per holder type (local bn) plus total and GDP."""
    p = panel() if p is None else p
    w = p.pivot_table(index=["country", "quarter_end"], columns="holder_type", values="value_local_bn")
    w["gdp"] = p[p["holder_type"] == "total"].set_index(["country", "quarter_end"])["gdp_local_bn"]
    return w.reset_index()


def quarterly_metrics(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """country, quarter_end, metric, value at data dates (before publication stamping)."""
    w = wide(p).sort_values(["country", "quarter_end"])
    tot = w["total"].where(w["total"] > 0)
    vals = {
        "fnb_share": 100 * w["fnb"] / tot,
        "elastic_share": 100 * w[list(ELASTIC)].sum(axis=1, min_count=len(ELASTIC)) / tot,
        "cb_hold_share": 100 * w["dcb"] / tot,
        "ibr": sum(w[t] * s for t, s in IRI_SCORE.items()) / tot,
    }
    w["priv"] = w[list(PRIVATE)].sum(axis=1, min_count=len(PRIVATE))
    for t in (*TYPES, "priv"):
        vals[f"d_{t}_gdp"] = 100 * w.groupby("country")[t].diff(4) / w["gdp"].where(w["gdp"] > 0)
    rows = [pd.DataFrame({"country": w["country"], "quarter_end": w["quarter_end"], "metric": k, "value": v})
            for k, v in vals.items() if k in METRICS]
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"])


def metrics(p: pd.DataFrame | None = None, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Monthly rows in `headline.panel()` shape: each quarter stamped at `public_month`, forward-filled
    monthly until the next public point and through the last complete month before `today`."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    end = today.to_period("M").to_timestamp() - pd.DateOffset(months=1)
    q = quarterly_metrics(p)
    q["month"] = q["quarter_end"].map(public_month)
    q = q[q["month"] <= end]
    out = []
    for (m, c), g in q.groupby(["metric", "country"]):
        # Two quarters released together: the later one is what was public that month.
        s = g.sort_values("quarter_end").groupby("month")["value"].last()
        idx = pd.date_range(s.index.min(), end, freq="MS")
        s = s.reindex(idx).ffill()
        out.append(pd.DataFrame({"country": c, "month": idx, "metric": m, "value": s.to_numpy(), "phase": "history"}))
    cols = ["country", "month", "metric", "value", "phase"]
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)[cols].sort_values(["metric", "country", "month"]).reset_index(drop=True)


def state_series(name: str = "elastic_share", p: pd.DataFrame | None = None,
                 today: pd.Timestamp | None = None) -> pd.DataFrame:
    """country, month, value of one metric as public each month (for `evidence`-style states)."""
    m = metrics(p, today)
    return m[m["metric"] == name][["country", "month", "value"]].reset_index(drop=True)


def coverage(p: pd.DataFrame | None = None) -> pd.DataFrame:
    p = panel() if p is None else p
    t = p[p["holder_type"] == "total"]
    g = t.groupby("country")["quarter_end"].agg(first="min", last="max", n="count").reset_index()
    g["first"] = g["first"].dt.to_period("Q").astype(str)
    g["last"] = g["last"].dt.to_period("Q").astype(str)
    return g


# ---------------------------------------------------------------- consistency vs holders.parquet
def compare_holders(p: pd.DataFrame | None = None, holders: pd.DataFrame | None = None) -> pd.DataFrame:
    """IMF vs `cache/holders.parquet` on matching period ends: central-bank and rest-of-world shares
    (pp of each source's own stock). UK ONS reports banks and the BoE together, so its central-bank
    line is IMF dcb + db vs `banks_and_boe`. US OFS-2 lumps the Fed with intragovernmental accounts
    (not in consolidated general-government debt), so the US central-bank line is skipped."""
    p = panel() if p is None else p
    if holders is None:
        path = CACHE / "holders.parquet"
        holders = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if holders.empty:
        return pd.DataFrame()
    w = wide(p)
    tot = w["total"]
    imf = pd.DataFrame({"country": w["country"], "date": w["quarter_end"],
                        "cb_imf": 100 * w["dcb"] / tot,
                        "cbdb_imf": 100 * (w["dcb"] + w["db"]) / tot,
                        "row_imf": 100 * (w["fo"] + w["fb"] + w["fnb"]) / tot})
    h = holders.copy()
    h["date"] = pd.to_datetime(h["asof"])
    h = h[h["stock_mn"] > 0]
    piv = h.pivot_table(index=["country", "date"], columns="holder", values="share", aggfunc="first") * 100
    piv = piv.reset_index()
    rows = []
    for c, g in piv.groupby("country"):
        j = g.merge(imf[imf["country"] == c], on=["country", "date"])
        if j.empty:
            continue
        freq = "annual" if c in ("DE", "FR") else "quarterly"
        lines = [("rest_of_world", "row_imf", "rest_of_world")]
        if c == "UK":
            lines.append(("central_bank (UK: banks+BoE)", "cbdb_imf", "banks_and_boe"))
        elif c != "US":
            lines.append(("central_bank", "cb_imf", "central_bank"))
        for label, a, b in lines:
            if b not in j:
                continue
            d = (j[a] - j[b]).dropna()
            if d.empty:
                continue
            rows.append({"country": c, "line": label, "freq": freq, "n": len(d),
                         "first": j.loc[d.index, "date"].min().date(), "last": j.loc[d.index, "date"].max().date(),
                         "imf_mean": j.loc[d.index, a].mean(), "holders_mean": j.loc[d.index, b].mean(),
                         "mean_gap_pp": d.mean(), "mean_abs_gap_pp": d.abs().mean(),
                         "corr_d4": _corr_change(j.loc[d.index], a, b)})
    return pd.DataFrame(rows)


def _corr_change(j: pd.DataFrame, a: str, b: str) -> float:
    j = j.sort_values("date")
    k = 1 if len(j) < 30 else 4
    x, y = j[a].diff(k), j[b].diff(k)
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 5 else np.nan


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    path = fetch()
    print(path, vintage(path))
    print(coverage().to_string(index=False))
    print(compare_holders().round(2).to_string(index=False))
