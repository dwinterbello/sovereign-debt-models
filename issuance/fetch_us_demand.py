"""Higher-frequency US Treasury demand: Fed custody, bank holdings, money funds and auction
allotments by investor class, as monthly metrics stamped at publication.

    PYTHONPATH=. .venv/bin/python -m issuance.fetch_us_demand [--refresh]

Sources (raw text/xls cached under cache/us_demand/; every parser reads the cache, so the module
runs offline once fetched):
  custody     FRED WMTSECL1, H.4.1 marketable Treasuries held in custody for foreign official and
              international accounts, Wednesday level, $mn (2002-12 →). Released Thursday (+1 day).
  bank_ust    FRED TNMACBW027SBOG, H.8 all commercial banks, Treasury and agency securities ex-MBS,
              SA, Wednesday, $bn (2009-07 →). H.8 has no Treasury-only weekly line; agency
              debentures are a small part of ex-MBS. Released the Friday nine days later.
  bank_tas    FRED TASACBW027SBOG, H.8 Treasury and agency securities incl. MBS, SA, $bn (1973 →).
  mmf_retail  FRED WRMFNS, H.6 retail money-market funds, week ending Monday, $bn. H.6 is monthly,
              about four weeks behind (+35 days used). ICI's weekly total is not freely fetchable
              (403) and FRED's institutional line stops in 2021.
  marketable  Fiscal Data MSPD table 1, total marketable debt held by the public, month end, $bn;
              used for the month before the stamp (the MSPD prints a few business days after).
  allotments  Treasury investor-class auction allotments (home.treasury.gov/data/investor-class-
              auction-allotments): coupons 2000 →, bills 2001 →, per issue, total issue, SOMA
              add-on and seven classes. Coupon files are reposted around the 7th and 22nd, bills
              around the 7th; the filename carries the posting date. H.8 and H.4.1 come from FRED's
              latest vintage, so H.8 benchmark revisions are not real time.

Publication stamping: `month` = the calendar month by whose end the value was public. Weekly
series use the last observation released (date + release lag) on or before the month end.
Allotments for issues in month m are stamped m+1: in every archived vintage the posting dated in
m+1 (by the 23rd) contains all of month m's issues (`vintage_lags`). Used with evidence lag 1,
so the price window starts the month after the stamp.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.http import session
from issuance.paths import CACHE

RAW = CACHE / "us_demand"
VINTAGES = RAW / "vintages"
WEEKLY_OUT = CACHE / "us_demand_weekly.parquet"
ALLOT_OUT = CACHE / "us_allotments.parquet"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
MSPD = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1")
PAGE = "https://home.treasury.gov/data/investor-class-auction-allotments"
FILES = "https://home.treasury.gov"
# FRED id → (series, multiplier to $bn, release lag in days after the observation date)
FRED_SERIES = {
    "WMTSECL1": ("custody", 1e-3, 1),
    "TNMACBW027SBOG": ("bank_ust", 1.0, 9),
    "TASACBW027SBOG": ("bank_tas", 1.0, 9),
    "WRMFNS": ("mmf_retail", 1.0, 35),
}
CLASSES = ("depository", "individuals", "dealers", "pension_ins", "inv_funds", "foreign", "other")
CLASS_LABEL = {"depository": "Depository institutions", "individuals": "Individuals", "dealers": "Dealers and brokers",
               "pension_ins": "Pension, retirement and insurance", "inv_funds": "Investment funds",
               "foreign": "Foreign and international", "other": "Other"}
ALLOT_PUBLIC_MONTHS = 1

RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- fetch
def _save(name: str, content: bytes) -> Path:
    path = RAW / name
    path.write_bytes(content)
    return path


def _fetch_fred(_s) -> None:
    # FRED stalls on the browser user agent of `issuance.http.session`; plain requests answers.
    import requests

    for sid in FRED_SERIES:
        r = requests.get(FRED.format(sid), timeout=60)
        r.raise_for_status()
        if r.text.lstrip().startswith("<") or len(r.text) < 20:
            raise ValueError(f"FRED {sid}: HTML response")
        _save(f"fred_{sid}.csv", r.content)


def _fetch_mspd(s) -> None:
    r = s.get(MSPD, params={"filter": "security_type_desc:eq:Total Marketable", "sort": "record_date",
                            "page[size]": 10000}, timeout=120)
    r.raise_for_status()
    _save("mspd_total_marketable.json", r.content)


def allotment_links(html: str) -> list[str]:
    """xls paths on the allotments page (current coupon and bill files plus the 2000–2009 history)."""
    out = re.findall(r'href="?(/system/files/276/[^">]+?\.xlsx?)"?[\s>]', html)
    return list(dict.fromkeys(p.strip() for p in out))


def _fetch_allotments(s) -> None:
    r = s.get(PAGE, timeout=60)
    r.raise_for_status()
    _save("allotments_page.html", r.content)
    for link in allotment_links(r.text):
        name = link.rsplit("/", 1)[-1].replace("%20", " ")
        f = s.get(FILES + link.replace(" ", "%20"), timeout=120)
        f.raise_for_status()
        if not f.content[:8].startswith(b"\xd0\xcf\x11\xe0") and not f.content[:2] == b"PK":
            raise ValueError(f"{name}: not an Excel file")
        _save(name, f.content)


def fetch(refresh: bool = False) -> None:
    """Download every raw source into cache/us_demand/ when missing (or always with refresh). A
    failing source keeps its cached copy."""
    s = session()
    have = lambda pat: any(RAW.glob(pat))  # noqa: E731
    jobs = [("fred", lambda: _fetch_fred(s), all((RAW / f"fred_{k}.csv").exists() for k in FRED_SERIES)),
            ("mspd", lambda: _fetch_mspd(s), (RAW / "mspd_total_marketable.json").exists()),
            ("allotments", lambda: _fetch_allotments(s), have("*IC-Coupons*.xls") and have("*IC-Bills*.xls"))]
    for name, job, cached in jobs:
        if cached and not refresh:
            continue
        try:
            job()
        except Exception as exc:  # noqa: BLE001 - a dead source keeps the cached copy
            print(f"  us_demand {name}: fetch failed ({exc}); using cache")


# ---------------------------------------------------------------- weekly
def parse_fred(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna().sort_index()


def weekly_panel() -> pd.DataFrame:
    """Long weekly rows: date (observation), released (date + release lag), series, value ($bn)."""
    parts = []
    for sid, (name, mult, lag) in FRED_SERIES.items():
        p = RAW / f"fred_{sid}.csv"
        if not p.exists():
            continue
        s = parse_fred(p.read_text()) * mult
        if name == "custody":
            s = s[s > 0]  # the first H.4.1 rows on FRED are zero placeholders
        parts.append(pd.DataFrame({"date": s.index, "released": s.index + pd.Timedelta(days=lag),
                                   "series": name, "value": s.to_numpy()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "released", "series", "value"])


def marketable() -> pd.Series:
    """Total marketable debt held by the public, $bn, by month (month-start index, month-end level)."""
    p = RAW / "mspd_total_marketable.json"
    if not p.exists():
        return pd.Series(dtype=float)
    import json
    d = pd.DataFrame(json.loads(p.read_text())["data"])
    s = pd.Series(pd.to_numeric(d["debt_held_public_mil_amt"], errors="coerce").to_numpy() / 1e3,
                  index=pd.to_datetime(d["record_date"]).dt.to_period("M").dt.to_timestamp())
    return s[~s.index.duplicated(keep="last")].sort_index()


def monthly_public(w: pd.DataFrame, series: str) -> pd.Series:
    """Per month: the latest value whose release date is on or before the month end."""
    g = w[w["series"] == series].sort_values("released")
    if g.empty:
        return pd.Series(dtype=float)
    month_end = g["released"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    s = g.groupby(month_end.dt.to_period("M").dt.to_timestamp())["value"].last()
    return s.asfreq("MS").ffill()


# ---------------------------------------------------------------- allotments
def _term_years(label: str) -> float:
    label = str(label).upper()
    m = re.search(r"(\d+)\s*-?\s*(YEAR|YR)", label)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*-?\s*(WEEK|WK)", label)
    if m:
        return float(m.group(1)) * 7 / 364
    m = re.search(r"(\d+)\s*-?\s*DAY", label)
    if m:
        return float(m.group(1)) / 364
    return np.nan


def _sec_type(label: str, bills: bool) -> str:
    u = str(label).upper()
    if "CMB" in u:
        return "CMB"
    if bills:
        return "Bill"
    if "TIPS" in u:
        return "TIPS"
    if "FRN" in u:
        return "FRN"
    if "BOND" in u:
        return "Bond"
    return "Note"


def parse_allotment_xls(path: Path | str | bytes, bills: bool | None = None) -> pd.DataFrame:
    """Wide rows per issue: issue_date, security_type, term_label, term_years, coupon_or_rate, cusip,
    maturity, total_bn, soma_bn and one $bn column per class. Handles both layouts (2000–2009 files
    in $mn with a separate term column; current files in $bn)."""
    raw = pd.read_excel(path if not isinstance(path, bytes) else io.BytesIO(path), header=None)
    text = " ".join(str(v) for v in raw.iloc[:6].to_numpy().ravel() if pd.notna(v))
    if bills is None:
        bills = "Bill Securities" in text
    scale = 1e-3 if "millions" in text.lower() else 1.0
    hdr = next(i for i in range(min(12, len(raw))) if str(raw.iat[i, 0]).strip().lower().startswith("issue"))
    head = [re.sub(r"\s+", " ", str(v)).strip().lower() for v in raw.iloc[hdr]]
    i_total = next(i for i, h in enumerate(head) if h.startswith("total"))
    body = raw.iloc[hdr + 1:].copy()
    body = body[pd.to_datetime(body[0], errors="coerce").notna()]
    if body.empty:
        return pd.DataFrame()
    two_label = i_total == 6  # old coupon file: security type + security term
    if two_label:
        label = body[1].astype(str).str.strip() + " " + body[2].astype(str).str.strip()
        rate, cusip, mat = body[3], body[4], body[5]
    else:
        label = body[1].astype(str).str.strip()
        rate, cusip, mat = body[2], body[3], body[4]
    num = body.iloc[:, i_total:i_total + 9].apply(pd.to_numeric, errors="coerce").fillna(0.0) * scale
    out = pd.DataFrame({
        "issue_date": pd.to_datetime(body[0]).dt.normalize().to_numpy(),
        "term_label": label.to_numpy(),
        "coupon_or_rate": pd.to_numeric(rate, errors="coerce").to_numpy(),
        "cusip": cusip.astype(str).str.strip().to_numpy(),
        "maturity": pd.to_datetime(mat, errors="coerce").to_numpy(),
        "total_bn": num.iloc[:, 0].to_numpy(), "soma_bn": num.iloc[:, 1].to_numpy(),
    })
    for k, c in enumerate(CLASSES):
        out[c] = num.iloc[:, 2 + k].to_numpy()
    out["security_type"] = [_sec_type(v, bills) for v in out["term_label"]]
    out["term_years"] = [_term_years(v) for v in out["term_label"]]
    life = (out["maturity"] - out["issue_date"]).dt.days / 365.25
    out["years_to_maturity"] = life
    out["term_years"] = out["term_years"].fillna(life)
    return out[out["total_bn"] > 0].reset_index(drop=True)


def _posting_date(name: str) -> pd.Timestamp | None:
    """Posting date carried by a current-file name, e.g. 'September-22_2026-IC-Coupons.xls'."""
    m = re.match(r"([A-Za-z]+)[\s\-_]*(\d{1,2})[\-_ ]+(\d{4})", name.replace("%20", " "))
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
    except ValueError:
        return None


def _bidders() -> pd.DataFrame:
    p = CACHE / "us_bidders.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _auction_dates(wide: pd.DataFrame) -> pd.Series:
    """Auction date per coupon issue from the Fiscal Data auctions cache: same CUSIP, latest auction
    on or before the issue date and at most 21 days before it (reopenings share the CUSIP)."""
    b = _bidders()
    out = pd.Series(pd.NaT, index=wide.index, dtype="datetime64[ns]")
    if b.empty:
        return out
    by = {c: g["auction_date"].sort_values().to_numpy() for c, g in b.groupby("cusip")}
    for i, c, d in zip(wide.index, wide["cusip"], wide["issue_date"]):
        a = by.get(c)
        if a is None:
            continue
        k = np.searchsorted(a, np.datetime64(d), side="right") - 1
        if k >= 0 and (np.datetime64(d) - a[k]) <= np.timedelta64(21, "D"):
            out.loc[i] = a[k]
    return out


def allotments_wide() -> pd.DataFrame:
    """All cached current and history files, one row per issue (newest file wins on overlap)."""
    files = sorted(RAW.glob("*.xls"), key=lambda p: (_posting_date(p.name) or pd.Timestamp("1990-01-01")))
    parts = []
    for p in files:
        d = parse_allotment_xls(p)
        if not d.empty:
            parts.append(d.assign(source=p.name))
    if not parts:
        return pd.DataFrame()
    w = pd.concat(parts, ignore_index=True).drop_duplicates(["issue_date", "cusip"], keep="last")
    w = w.sort_values(["issue_date", "cusip"]).reset_index(drop=True)
    w["private_bn"] = w["total_bn"] - w["soma_bn"]
    coupon = ~w["security_type"].isin(["Bill", "CMB"])
    w["auction_date"] = pd.NaT
    w.loc[coupon, "auction_date"] = _auction_dates(w[coupon])
    w["duration"] = duration(w)
    return w


def duration(w: pd.DataFrame) -> np.ndarray:
    """Approximate modified duration at issue: a par bond at the coupon over the years to maturity;
    FRNs 0.25, bills their years to maturity."""
    from issuance.market import mod_duration

    yrs = w["years_to_maturity"].clip(lower=0.01).to_numpy(float)
    cpn = w["coupon_or_rate"].fillna(2.0).clip(lower=0.125).to_numpy(float)
    d = np.array([float(mod_duration(np.array([c]), years=y)[0]) for c, y in zip(cpn, yrs)])
    t = w["security_type"].to_numpy()
    d = np.where(np.isin(t, ["Bill", "CMB"]), yrs, d)
    return np.where(t == "FRN", 0.25, d)


def allotments_panel(wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long rows: issue_date, auction_date, security_type, term_years, cusip, investor_class,
    amount_bn, share_pct (of the private allotment = total less SOMA add-on), duration."""
    w = allotments_wide() if wide is None else wide
    if w.empty:
        return pd.DataFrame()
    keep = ["issue_date", "auction_date", "security_type", "term_label", "term_years", "cusip", "total_bn",
            "soma_bn", "private_bn", "duration"]
    long = w.melt(id_vars=keep, value_vars=list(CLASSES), var_name="investor_class", value_name="amount_bn")
    long["share_pct"] = 100 * long["amount_bn"] / long["private_bn"].where(long["private_bn"] > 0)
    return long.sort_values(["issue_date", "cusip", "investor_class"]).reset_index(drop=True)


def vintage_lags() -> pd.DataFrame:
    """Per cached posting (current files and archived vintages): posting date, last issue date it
    covers, and whether it holds every issue of the previous calendar month that the final file has."""
    final = allotments_wide()
    rows = []
    for p in sorted(list(RAW.glob("*.xls")) + list(VINTAGES.glob("*.xls"))):
        post = _posting_date(p.name)
        if post is None:
            continue
        d = parse_allotment_xls(p)
        if d.empty:
            continue
        kind = "Bill" if d["security_type"].isin(["Bill", "CMB"]).all() else "Coupon"
        prev = (post.to_period("M") - 1).to_timestamp()
        ref = final[final["issue_date"].dt.to_period("M").dt.to_timestamp() == prev]
        ref = ref[ref["security_type"].isin(["Bill", "CMB"]) == (kind == "Bill")]
        have = set(zip(d["issue_date"], d["cusip"]))
        rows.append({"file": p.name, "kind": kind, "posted": post, "last_issue": d["issue_date"].max(),
                     "lag_days": (post - d["issue_date"].max()).days,
                     "prev_month_complete": bool(ref.empty or all(k in have for k in zip(ref["issue_date"], ref["cusip"])))})
    return pd.DataFrame(rows).sort_values("posted").reset_index(drop=True)


# ---------------------------------------------------------------- monthly metrics
METRICS: dict[str, dict] = {
    "custody_3m": {"label": "Fed custody for foreign official, 3m change", "unit": "USD bn", "sign": -1, "fmt": ".0f",
                   "what": "H.4.1 marketable Treasuries held for foreign official and international accounts, "
                           "change over 3 months in the month-end value public by then."},
    "custody_12m": {"label": "Fed custody for foreign official, 12m change", "unit": "USD bn", "sign": -1, "fmt": ".0f",
                    "what": "As the 3m change, over 12 months."},
    "custody_12m_pct": {"label": "Fed custody, 12m change, % of marketable", "unit": "% of marketable debt", "sign": -1,
                        "fmt": ".2f", "what": "12m custody change ÷ marketable debt held by the public (MSPD, prior month end)."},
    "bank_3m": {"label": "Bank Treasuries (H.8 ex-MBS), 3m change", "unit": "USD bn", "sign": -1, "fmt": ".0f",
                "what": "H.8 all commercial banks, Treasury and agency securities excluding MBS, SA, 3m change."},
    "bank_12m": {"label": "Bank Treasuries (H.8 ex-MBS), 12m change", "unit": "USD bn", "sign": -1, "fmt": ".0f",
                 "what": "As the 3m change, over 12 months."},
    "bank_12m_pct": {"label": "Bank Treasuries, 12m change, % of marketable", "unit": "% of marketable debt", "sign": -1,
                     "fmt": ".2f", "what": "12m H.8 ex-MBS change ÷ marketable debt held by the public."},
    "bank_tas_12m": {"label": "Bank Treasury + agency (H.8 incl. MBS), 12m change", "unit": "USD bn", "sign": -1,
                     "fmt": ".0f", "what": "H.8 Treasury and agency securities including MBS, SA, 12m change (1973 →)."},
    "mmf_12m": {"label": "Retail money funds, 12m change", "unit": "USD bn", "sign": -1, "fmt": ".0f",
                "what": "H.6 retail money-market fund assets (week ending Monday, ~4 weeks behind), 12m change."},
    "alloc_inv_funds": {"label": "Coupon allotment: investment funds, 3m", "unit": "% of private allotment", "sign": -1,
                        "fmt": ".1f", "what": "Investment funds' share of coupon allotments ex-SOMA, issues in the 3 months."},
    "alloc_pension_ins": {"label": "Coupon allotment: pension/insurance, 3m", "unit": "% of private allotment", "sign": -1,
                          "fmt": ".1f", "what": "Pension, retirement and insurance share of coupon allotments ex-SOMA, 3m."},
    "alloc_foreign": {"label": "Coupon allotment: foreign, 3m", "unit": "% of private allotment", "sign": -1, "fmt": ".1f",
                      "what": "Foreign and international share of coupon allotments ex-SOMA, 3m."},
    "alloc_dealers": {"label": "Coupon allotment: dealers, 3m", "unit": "% of private allotment", "sign": 1, "fmt": ".1f",
                      "what": "Dealers and brokers' share of coupon allotments ex-SOMA, 3m: dealers absorbing = weak end demand."},
    "alloc_foreign_dv01": {"label": "Coupon allotment: foreign, DV01-weighted, 3m", "unit": "% of private DV01", "sign": -1,
                           "fmt": ".1f", "what": "Foreign share of coupon allotments ex-SOMA weighted by approximate duration."},
    "alloc_dealers_dv01": {"label": "Coupon allotment: dealers, DV01-weighted, 3m", "unit": "% of private DV01", "sign": 1,
                           "fmt": ".1f", "what": "Dealer share of coupon allotments ex-SOMA weighted by approximate duration."},
    "alloc_inv_funds_d12": {"label": "Coupon allotment: investment funds, 12m change of 3m share", "unit": "pp",
                            "sign": -1, "fmt": ".1f",
                            "what": "Change over 12 months in the 3m investment-fund share: the level trends from "
                                    "20% (2010) to 70% (2025), so its percentile is mostly a time trend."},
    "alloc_foreign_d12": {"label": "Coupon allotment: foreign, 12m change of 3m share", "unit": "pp", "sign": -1,
                          "fmt": ".1f", "what": "Change over 12 months in the 3m foreign share."},
    "alloc_dealers_d12": {"label": "Coupon allotment: dealers, 12m change of 3m share", "unit": "pp", "sign": 1,
                          "fmt": ".1f", "what": "Change over 12 months in the 3m dealer share (level trends 52% → 15%)."},
}
# Months after the stamp before the price window starts (same convention as headline supply metrics).
LAG = {k: 1 for k in METRICS}
ALLOT_METRICS = {"alloc_inv_funds": ("inv_funds", False), "alloc_pension_ins": ("pension_ins", False),
                 "alloc_foreign": ("foreign", False), "alloc_dealers": ("dealers", False),
                 "alloc_foreign_dv01": ("foreign", True), "alloc_dealers_dv01": ("dealers", True)}


def allotment_shares(wide: pd.DataFrame, months: int = 3, dv01: bool = False) -> pd.DataFrame:
    """Per issue month: each class's share (%) of coupon allotments ex-SOMA over the `months` issue
    months ending there (amount- or duration-weighted), plus the month stamp when it is public."""
    c = wide[~wide["security_type"].isin(["Bill", "CMB"])].copy()
    c["m"] = c["issue_date"].dt.to_period("M").dt.to_timestamp()
    wt = c["duration"].to_numpy() if dv01 else np.ones(len(c))
    sums = pd.DataFrame({k: c[k].to_numpy() * wt for k in [*CLASSES, "private_bn"]}, index=c["m"]).groupby(level=0).sum()
    sums = sums.asfreq("MS", fill_value=0.0).rolling(months, min_periods=months).sum()
    out = 100 * sums[list(CLASSES)].div(sums["private_bn"].where(sums["private_bn"] > 0), axis=0)
    out.index.name = "issue_month"
    out = out.reset_index()
    out["month"] = out["issue_month"] + pd.DateOffset(months=ALLOT_PUBLIC_MONTHS)
    return out


def _chg(s: pd.Series, k: int) -> pd.Series:
    return s - s.shift(k)


def metrics(today: pd.Timestamp | None = None, weekly: pd.DataFrame | None = None,
            wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly rows in `headline.panel()` shape (country, month, metric, value, phase='history'),
    stamped at the month by whose end the value was public; complete months only."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    cur = today.to_period("M").to_timestamp()
    w = weekly_panel() if weekly is None else weekly
    w = w[w["released"] <= today]
    wide = allotments_wide() if wide is None else wide
    mk = marketable().shift(1, freq="MS")  # month m's level prints in m+1
    vals: dict[str, pd.Series] = {}
    cust = monthly_public(w, "custody")
    bank = monthly_public(w, "bank_ust")
    if not cust.empty:
        vals["custody_3m"], vals["custody_12m"] = _chg(cust, 3), _chg(cust, 12)
        vals["custody_12m_pct"] = 100 * vals["custody_12m"] / mk.reindex(cust.index)
    if not bank.empty:
        vals["bank_3m"], vals["bank_12m"] = _chg(bank, 3), _chg(bank, 12)
        vals["bank_12m_pct"] = 100 * vals["bank_12m"] / mk.reindex(bank.index)
    tas = monthly_public(w, "bank_tas")
    if not tas.empty:
        vals["bank_tas_12m"] = _chg(tas, 12)
    mmf = monthly_public(w, "mmf_retail")
    if not mmf.empty:
        vals["mmf_12m"] = _chg(mmf, 12)
    coupons = wide[~wide["security_type"].isin(["Bill", "CMB"])] if not wide.empty else wide
    if not coupons.empty:
        # the newest issue month in a posting can be partial (mid-month cut)
        last_full = coupons["issue_date"].max().to_period("M").to_timestamp() - pd.DateOffset(months=1)
        for dv in (False, True):
            sh = allotment_shares(wide, dv01=dv)
            sh = sh[sh["issue_month"] <= last_full]
            for name, (cls, d) in ALLOT_METRICS.items():
                if d == dv:
                    vals[name] = sh.set_index("month")[cls].asfreq("MS")
        for k in ("alloc_inv_funds", "alloc_foreign", "alloc_dealers"):
            vals[f"{k}_d12"] = _chg(vals[k], 12)
    parts = [pd.DataFrame({"country": "US", "month": s.index, "metric": k, "value": s.to_numpy(), "phase": "history"})
             for k, s in vals.items()]
    out = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    out = out[np.isfinite(out["value"]) & (out["month"] < cur)]
    return out.sort_values(["metric", "month"]).reset_index(drop=True)


def write(refresh: bool = False) -> pd.DataFrame:
    fetch(refresh)
    weekly_panel().to_parquet(WEEKLY_OUT, index=False)
    allotments_panel().to_parquet(ALLOT_OUT, index=False)
    return metrics()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    m = write(a.refresh)
    print(m.groupby("metric")["month"].agg(["min", "max", "count"]))
    print(vintage_lags().to_string(index=False))
