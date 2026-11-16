"""UK gilt demand by holder sector (ONS financial accounts + BoE Bankstats + BoE APF operations).

    PYTHONPATH=. .venv/bin/python -m issuance.uk_holders [--refresh]

Sources (raw files cached under `cache/uk_holders/`, parsed offline; the ONS JSON falls back to the
`uk_flows/cache/ons/` copy the uk_flows / uk_long_end stack already keeps fresh):

  ONS UKEA  quarterly financial accounts, British government securities (AF.32N11), £mn, 1987Q1→.
            Levels (market value): NIZC insurers + pension funds (S.128+S.129), NNTW MFIs (S.121 Bank
            of England incl. the APF + S.122/S.123 banks), NJSA other financial (S.12-OFI: funds, LDI
            pooled funds, hedge funds, MMFs), NLDU rest of the world, NISU households (S.14), NKKY
            private NFCs, NJFK local government, NKEQ public corporations, NYXR all UK holders.
            Transactions (revaluation-free): NBTF ICPF, NGCZ MFIs, NCMD OFI, NEXL rest of the world,
            NBMX households, NCFV NPISH, NERD private NFCs, NBZN local government, NCYT public
            corporations, NYRS all UK holders, NAVT central-government liability (total net issuance
            at transaction value). Identities NYXR = Σ UK sector levels and NAVT = NYRS + NEXL =
            Σ sector transactions hold to rounding (`sector_sums`).
  ONS PUSF  monthly: JK7N Bank of England (incl. APF) holdings of gilts at face, FZEU the Bank's own
            (non-APF) gilt holdings at face (2015→, ~£15bn).
  BoE IADB  Bankstats, monthly, £mn, NSA: RPMB3QP UK-resident MFIs ex central bank net gilt holdings
            (B1.7/B1.4, 2009-05→), RPMB4CA their net purchases (changes, 2009-06→), LPMB9P6
            non-residents' net purchases of gilts (A7.1, 1986→), LPMVRDY M4 private sector (ICPF +
            OFC + PNFC + households) net purchases of gilts (A7.1; = −LPMAVBY in A3.2).
  BoE APF   operation-level QE purchases and QT sales (`cache/apf_operations.parquet`, nominal and
            price; redemptions at par on each ISIN's maturity date) plus the 2022 financial-stability
            purchases and their sale (`uk_flows/cache/apf/fs_*.xlsx`, proceeds).

Splitting MFIs. NNTW / NGCZ mix commercial banks with the Bank of England, whose APF is classified to
S.121. Private-bank transactions = NGCZ − APF transactions (purchases and sales at their accepted
price, redemptions at par) − ΔFZEU (Bank own book, face). `split_check()` scores that residual against
the Bankstats MFI-ex-CB series summed to quarters (RPMB4CA), which is the independent measurement.

DMO GEMM turnover (all-gemms-turnover-*.xls) is gross customer vs professional turnover only; the
twelve counterparty codes are collected but not published, so it carries no net flow by investor and
is not used.

Publication (established Oct-2026). UKEA: 2026Q2 updated 30 Sep 2026, next release 22 Dec 2026 (Q3)
→ a quarter is public in the 3rd month after its quarter-end month. PUSF: Aug-2026 updated 22 Sep →
the next month. Bankstats: the 21st UK working day after month end (BoE B1.7 notes; RPMB3QP July-2026
'Published: 01 Sep 2026' = 21st working day after 31 Jul with the 31 Aug bank holiday) → month m+1 or,
when that day spills over, m+2 (`bankstats_public_month`). APF operations: results the same day.
Values are the latest vintage (ONS revises the financial accounts; not real time). Each row of
`metrics()` is stamped at the month its *last* input became public and forward-filled;
`experiments/uk_holders_test.py` adds one more month of lag on top.
"""

from __future__ import annotations

import argparse
import functools
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE, REPO

DIR = CACHE / "uk_holders"
UKF_ONS = REPO / "uk_flows" / "cache" / "ons"
UKF_APF = REPO / "uk_flows" / "cache" / "apf"
APF_OPS = CACHE / "apf_operations.parquet"
ABSORB_OUT = CACHE / "uk_holders_absorption.parquet"
ONS_URL = "https://www.ons.gov.uk/{topic}/timeseries/{cdid}/{dataset}/data"
ONS_TOPICS = (("economy/grossdomesticproductgdp", "ukea"),
              ("economy/governmentpublicsectorandtaxes/publicsectorfinance", "pusf"),
              ("economy/nationalaccounts/balanceofpayments", "ukea"))
IADB = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"

# cdid → (sector, measure). LE = level (market value), F = transactions, FACE = level at face value.
ONS_SERIES = {
    "nizc": ("icpf", "LE"), "nntw": ("mfi", "LE"), "njsa": ("ofi", "LE"), "nldu": ("overseas", "LE"),
    "nisu": ("households", "LE"), "nkky": ("nfc", "LE"), "njfk": ("local_gov", "LE"),
    "nkeq": ("public_corp", "LE"), "nyxr": ("uk_total", "LE"),
    "nbtf": ("icpf", "F"), "ngcz": ("mfi", "F"), "ncmd": ("ofi", "F"), "nexl": ("overseas", "F"),
    "nbmx": ("households", "F"), "ncfv": ("npish", "F"), "nerd": ("nfc", "F"), "nbzn": ("local_gov", "F"),
    "ncyt": ("public_corp", "F"), "nyrs": ("uk_total", "F"), "navt": ("total", "F"),
    "jk7n": ("boe", "FACE"), "fzeu": ("boe_own", "FACE"),
}
UK_LEVEL_PARTS = ("icpf", "mfi", "ofi", "households", "nfc", "local_gov", "public_corp")
UK_FLOW_PARTS = ("icpf", "mfi", "ofi", "households", "npish", "nfc", "local_gov", "public_corp")
# IADB code → (series, measure)
BOE_SERIES = {"RPMB3QP": ("banks", "LE"), "RPMB4CA": ("banks", "F"), "LPMB9P6": ("overseas", "F"),
              "LPMVRDY": ("m4ps", "F")}
FS_FILES = {"fs_long_purchases": 1, "fs_linker_purchases": 1, "fs_sales": -1}
SOURCE = {"ons": "ONS UKEA financial accounts, BGS (AF.32N11)", "pusf": "ONS PUSF BoE gilt holdings, face",
          "boe": "BoE Bankstats (IADB)", "apf": "BoE APF operations"}
SECTOR_LABEL = {
    "total": "Total net issuance (CG liabilities)", "icpf": "Insurers + pension funds",
    "ofi": "Other financial (funds, LDI, hedge funds)", "overseas": "Overseas", "mfi": "MFIs (banks + BoE)",
    "banks": "Banks (MFIs ex BoE)", "apf": "BoE APF", "boe_own": "BoE own book", "households": "Households",
    "other": "Other domestic (NFC, LG, PC, NPISH)", "m4ps": "M4 private sector (ICPF + OFC + PNFC + HH)",
}
UKEA_PUBLIC_MONTHS = 3
PUSF_PUBLIC_MONTHS = 1
BANKSTATS_WORKDAY = 21
APF_START = pd.Timestamp("2009-03-31")


# ---------------------------------------------------------------- fetch
def _ons_path(cdid: str) -> Path:
    own = DIR / f"ons_{cdid}.json"
    if own.exists():
        return own
    return UKF_ONS / f"{cdid}.json"


def _fetch_ons(cdid: str, s) -> None:
    for topic, dataset in ONS_TOPICS:
        r = s.get(ONS_URL.format(topic=topic, cdid=cdid, dataset=dataset), timeout=90)
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            (DIR / f"ons_{cdid}.json").write_bytes(r.content)
            return
    raise RuntimeError(f"ONS {cdid}: no topic path answered")


def _fetch_boe(s) -> None:
    s.cookies.set("boeconsent", "necessary", domain="www.bankofengland.co.uk")
    r = s.get(IADB, params={"csv.x": "yes", "Datefrom": "01/Jan/1986", "Dateto": "now",
                            "SeriesCodes": ",".join(BOE_SERIES), "CSVF": "TN", "UsingCodes": "Y",
                            "VPD": "Y", "VFD": "N"}, timeout=120)
    r.raise_for_status()
    if not r.text.lstrip().upper().startswith("DATE,"):
        raise RuntimeError("BoE IADB: not CSV (consent page or refusal)")
    (DIR / "boe_iadb.csv").write_text(r.text)


def _fetch_fs(s) -> None:
    for name in FS_FILES:
        meta = UKF_APF / f"{name}.meta.json"
        if (UKF_APF / f"{name}.xlsx").exists() or not meta.exists():
            continue
        r = s.get(json.loads(meta.read_text())["url"], timeout=120)
        r.raise_for_status()
        (DIR / f"{name}.xlsx").write_bytes(r.content)


def fetch(refresh: bool = False) -> None:
    """Download what is missing (or everything with `refresh`). A failed source keeps its cache."""
    from issuance.http import session

    DIR.mkdir(parents=True, exist_ok=True)
    s = session()
    for cdid in ONS_SERIES:
        if refresh or not _ons_path(cdid).exists():
            try:
                _fetch_ons(cdid, s)
            except Exception as exc:  # noqa: BLE001
                print(f"  uk_holders ONS {cdid}: fetch failed ({exc}); using cache")
    if refresh or not (DIR / "boe_iadb.csv").exists():
        try:
            _fetch_boe(s)
        except Exception as exc:  # noqa: BLE001
            print(f"  uk_holders BoE IADB: fetch failed ({exc}); using cache")
    try:
        _fetch_fs(s)
    except Exception as exc:  # noqa: BLE001
        print(f"  uk_holders APF FS files: fetch failed ({exc})")


# ---------------------------------------------------------------- parse
def parse_ons(payload: dict) -> tuple[pd.Series, pd.Series]:
    """(quarterly by quarter_end, monthly by month start), £bn."""
    q = {pd.Period(r["date"].replace(" ", ""), freq="Q").to_timestamp(how="end").normalize(): float(r["value"]) / 1e3
         for r in payload.get("quarters", []) if str(r.get("value", "")).strip() not in ("", "..")}
    m = {pd.to_datetime(r["date"], format="%Y %b"): float(r["value"]) / 1e3
         for r in payload.get("months", []) if str(r.get("value", "")).strip() not in ("", "..")}
    return pd.Series(q, dtype=float).sort_index(), pd.Series(m, dtype=float).sort_index()


def ons_panel() -> pd.DataFrame:
    """Long quarterly rows: quarter_end, cdid, sector, measure (LE / F / FACE), value_gbp_bn."""
    rows = []
    for cdid, (sector, measure) in ONS_SERIES.items():
        p = _ons_path(cdid)
        if not p.exists():
            continue
        q, _ = parse_ons(json.loads(p.read_text()))
        rows.append(pd.DataFrame({"quarter_end": q.index, "cdid": cdid.upper(), "sector": sector,
                                  "measure": measure, "value_gbp_bn": q.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["quarter_end", "cdid", "sector", "measure", "value_gbp_bn"])


def pusf_monthly(cdid: str) -> pd.Series:
    p = _ons_path(cdid)
    return parse_ons(json.loads(p.read_text()))[1] if p.exists() else pd.Series(dtype=float)


def parse_iadb(text: str) -> pd.DataFrame:
    """IADB CSV (DATE, code…) → month, series, measure, value_gbp_bn."""
    d = pd.read_csv(io.StringIO(text))
    month = pd.to_datetime(d["DATE"], format="%d %b %Y").dt.to_period("M").dt.to_timestamp()
    out = []
    for code, (series, measure) in BOE_SERIES.items():
        if code not in d:
            continue
        v = pd.to_numeric(d[code], errors="coerce") / 1e3
        out.append(pd.DataFrame({"month": month, "code": code, "series": series, "measure": measure,
                                 "value_gbp_bn": v.to_numpy()}).dropna(subset=["value_gbp_bn"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["month", "code", "series", "measure", "value_gbp_bn"])


def boe_panel(path: Path | None = None) -> pd.DataFrame:
    path = DIR / "boe_iadb.csv" if path is None else path
    if not path.exists():
        fetch()
    return parse_iadb(path.read_text())


@functools.lru_cache(maxsize=16)
def _read_fs_cached(path: str, sign: int) -> pd.DataFrame:
    """Cached: openpyxl on the APF FS workbooks is ~3s each and was re-hit every callers."""
    raw = pd.read_excel(path, header=None)
    hdr = next(i for i in range(10) if any("operation date" in str(v).lower() for v in raw.iloc[i]))
    head = [str(v).lower() for v in raw.iloc[hdr]]
    c_date = next(i for i, h in enumerate(head) if "operation date" in h)
    c_proc = next(i for i, h in enumerate(head) if "allocation (proceeds" in h)
    body = raw.iloc[hdr + 1:]
    date = pd.to_datetime(body.iloc[:, c_date], errors="coerce")
    proc = pd.to_numeric(body.iloc[:, c_proc], errors="coerce")
    d = pd.DataFrame({"date": date, "value_gbp_bn": sign * proc / 1e3}).dropna()
    return d[d["value_gbp_bn"] != 0]


def _read_fs(path: Path, sign: int) -> pd.DataFrame:
    return _read_fs_cached(str(path), sign).copy()


def maturity_from_desc(desc: pd.Series) -> pd.Series:
    """Redemption date from the APF bond code 'UKT_<coupon>_<ddmmyy>'. The joined `maturity_dt` only
    carries the year (every redemption on 1 July), which misplaces QT redemptions by up to a quarter."""
    tail = desc.astype(str).str.extract(r"_(\d{6})\s*$")[0]
    return pd.to_datetime(tail, format="%d%m%y", errors="coerce")


def apf_flows(ops: pd.DataFrame | None = None, fs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily APF transactions, £bn: kind ∈ purchase / sale (nominal × accepted price), redemption
    (−nominal held at maturity), fs (2022 financial-stability purchases and sales, proceeds)."""
    ops = pd.read_parquet(APF_OPS) if ops is None else ops
    ops = ops[ops["amount"].fillna(0) != 0].copy()
    price = ops["price"].where(ops["price"] > 0, 100.0)
    tx = pd.DataFrame({"date": pd.to_datetime(ops["settlement_dt"].fillna(ops["op_dt"])),
                       "kind": ops["direction"].to_numpy(), "isin": ops["isin"].to_numpy(),
                       "nominal_gbp_bn": ops["amount"].to_numpy() / 1e3,
                       "value_gbp_bn": ops["amount"].to_numpy() * price.to_numpy() / 1e5})
    red = []
    mats = maturity_from_desc(ops["sec_desc_long"]).fillna(pd.to_datetime(ops["maturity_dt"], errors="coerce"))
    mats = pd.Series(mats.to_numpy(), index=ops["isin"].to_numpy())
    mats = mats[~mats.index.duplicated()]
    for isin, g in tx.groupby("isin"):
        mat = mats.get(isin)
        if mat is None or pd.isna(mat):
            continue
        mat = pd.Timestamp(mat)
        held = g.loc[g["date"] <= mat, "nominal_gbp_bn"].sum()
        if held > 1e-9 and mat <= pd.Timestamp.today():
            red.append({"date": mat, "kind": "redemption", "isin": isin, "nominal_gbp_bn": -held, "value_gbp_bn": -held})
    if fs is None:
        parts = []
        for name, sign in FS_FILES.items():
            p = DIR / f"{name}.xlsx"
            p = p if p.exists() else UKF_APF / f"{name}.xlsx"
            if p.exists():
                parts.append(_read_fs(p, sign))
        fs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "value_gbp_bn"])
    fs = fs.assign(kind="fs", isin=None, nominal_gbp_bn=np.nan)
    out = pd.concat([tx, pd.DataFrame(red), fs[tx.columns]], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def apf_monthly(flows: pd.DataFrame | None = None) -> pd.DataFrame:
    """month, apf_f (transactions, £bn), apf_face (QE/QT holdings at face, month end, £bn)."""
    f = apf_flows() if flows is None else flows
    f = f.assign(month=f["date"].dt.to_period("M").dt.to_timestamp())
    idx = pd.date_range("2009-03-01", f["month"].max(), freq="MS")
    tx = f.groupby("month")["value_gbp_bn"].sum().reindex(idx, fill_value=0.0)
    face = f[f["kind"] != "fs"].groupby("month")["nominal_gbp_bn"].sum().reindex(idx, fill_value=0.0).cumsum()
    return pd.DataFrame({"month": idx, "apf_f": tx.to_numpy(), "apf_face": face.to_numpy()})


# ---------------------------------------------------------------- quarterly panel
def _q(month: pd.Series) -> pd.Series:
    return pd.PeriodIndex(month, freq="Q").to_timestamp(how="end").normalize()


def wide(p: pd.DataFrame | None = None, measure: str = "F") -> pd.DataFrame:
    p = ons_panel() if p is None else p
    w = p[p["measure"] == measure].pivot_table(index="quarter_end", columns="sector", values="value_gbp_bn")
    w.columns.name = None
    return w.sort_index()


def quarterly_panel(p: pd.DataFrame | None = None, apf: pd.DataFrame | None = None,
                    boe: pd.DataFrame | None = None, own: pd.Series | None = None) -> pd.DataFrame:
    """quarter_end × columns, £bn. Transactions: total, icpf, ofi, overseas, mfi, households, apf,
    boe_own, banks (= mfi − apf − boe_own), other (= total − named), private (= total − apf − boe_own),
    banks_bankstats (RPMB4CA summed). Levels (`*_le`): icpf, ofi, overseas, mfi, uk_total, stock
    (= uk_total + overseas), banks_le (RPMB3QP at quarter end), boe_le (= mfi − banks_le)."""
    p = ons_panel() if p is None else p
    apf = apf_monthly() if apf is None else apf
    boe = boe_panel() if boe is None else boe
    F, L = wide(p, "F"), wide(p, "LE")
    idx = F.index
    a = apf.assign(q=_q(apf["month"])).groupby("q")["apf_f"].sum().reindex(idx).fillna(0.0)
    a[idx < pd.Timestamp("2009-03-31")] = 0.0
    own = pusf_monthly("fzeu") if own is None else own
    own_q = own[own.index.month.isin([3, 6, 9, 12])]
    own_q.index = _q(pd.Series(own_q.index))
    own_f = own_q.diff().reindex(idx).fillna(0.0)
    out = pd.DataFrame(index=idx)
    for k in ("total", "icpf", "ofi", "overseas", "mfi", "households"):
        out[k] = F.get(k)
    out["apf"], out["boe_own"] = a, own_f
    out["banks"] = out["mfi"] - out["apf"] - out["boe_own"]
    named = ["icpf", "ofi", "overseas", "mfi", "households"]
    out["other"] = out["total"] - out[named].sum(axis=1, min_count=len(named))
    out["private"] = out["total"] - out["apf"] - out["boe_own"]
    bf = boe[(boe["series"] == "banks") & (boe["measure"] == "F")]
    bq = bf.assign(q=_q(bf["month"])).groupby("q")["value_gbp_bn"].agg(["sum", "count"])
    out["banks_bankstats"] = bq["sum"].where(bq["count"] == 3).reindex(idx)
    for k in ("icpf", "ofi", "overseas", "mfi", "uk_total"):
        out[f"{k}_le"] = L.get(k)
    out["stock_le"] = out["uk_total_le"] + out["overseas_le"]
    bl = boe[(boe["series"] == "banks") & (boe["measure"] == "LE")]
    bl = bl[bl["month"].dt.month.isin([3, 6, 9, 12])]
    out["banks_le"] = pd.Series(bl["value_gbp_bn"].to_numpy(), index=_q(bl["month"])).reindex(idx)
    out["boe_le"] = out["mfi_le"] - out["banks_le"]
    out.index.name = "quarter_end"
    return out


# ---------------------------------------------------------------- checks
def sector_sums(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """ONS identities per quarter (£bn gaps): NYXR vs Σ UK sector levels, NYRS vs Σ UK sector
    transactions, NAVT vs NYRS + NEXL."""
    p = ons_panel() if p is None else p
    F, L = wide(p, "F"), wide(p, "LE")
    out = pd.DataFrame({"gap_level": L["uk_total"] - L[list(UK_LEVEL_PARTS)].sum(axis=1, min_count=len(UK_LEVEL_PARTS)),
                        "gap_flow": F["uk_total"] - F[list(UK_FLOW_PARTS)].sum(axis=1, min_count=len(UK_FLOW_PARTS)),
                        "gap_total": F["total"] - F["uk_total"] - F["overseas"], "uk_total": L["uk_total"]})
    return out.dropna().reset_index()


def split_check(qp: pd.DataFrame | None = None, start: str = "2009-09-30") -> pd.DataFrame:
    """Private-bank transactions two ways: ONS NGCZ − APF − ΔFZEU vs Bankstats RPMB4CA summed to the
    quarter. Also ONS NEXL vs Bankstats LPMB9P6 summed. corr, mean gap, mean |gap|, mean |flow|."""
    qp = quarterly_panel() if qp is None else qp
    boe = boe_panel()
    ov = boe[(boe["series"] == "overseas")]
    ovq = ov.assign(q=_q(ov["month"])).groupby("q")["value_gbp_bn"].agg(["sum", "count"])
    qp = qp.assign(overseas_bankstats=ovq["sum"].where(ovq["count"] == 3).reindex(qp.index))
    rows = []
    for a, b in (("banks", "banks_bankstats"), ("overseas", "overseas_bankstats")):
        j = qp.loc[qp.index >= start, [a, b]].dropna()
        rows.append({"ons": a, "bankstats": b, "n": len(j), "corr": j[a].corr(j[b]),
                     "mean_gap_bn": (j[a] - j[b]).mean(), "mae_bn": (j[a] - j[b]).abs().mean(),
                     "mean_abs_bn": j[b].abs().mean(), "first": j.index.min(), "last": j.index.max()})
    return pd.DataFrame(rows)


def apf_face_check(apf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reconstructed APF face (QE/QT ops, redemptions at maturity) vs PUSF JK7N − FZEU, month ends."""
    apf = apf_monthly() if apf is None else apf
    j = pd.DataFrame({"apf_face": apf.set_index("month")["apf_face"],
                      "pusf": pusf_monthly("jk7n") - pusf_monthly("fzeu").reindex(pusf_monthly("jk7n").index).fillna(0.0)})
    return j.dropna().reset_index(names="month")


def coverage(p: pd.DataFrame | None = None, boe: pd.DataFrame | None = None) -> pd.DataFrame:
    p = ons_panel() if p is None else p
    boe = boe_panel() if boe is None else boe
    rows = [{"source": SOURCE["pusf" if m == "FACE" else "ons"], "code": c, "sector": s, "measure": m,
             "first": g["quarter_end"].min().date(), "last": g["quarter_end"].max().date(), "freq": "Q"}
            for (c, s, m), g in p.groupby(["cdid", "sector", "measure"])]
    rows += [{"source": SOURCE["boe"], "code": c, "sector": s, "measure": m, "first": g["month"].min().date(),
              "last": g["month"].max().date(), "freq": "M"} for (c, s, m), g in boe.groupby(["code", "series", "measure"])]
    for c in ("jk7n", "fzeu"):
        s = pusf_monthly(c)
        if len(s):
            rows.append({"source": SOURCE["pusf"], "code": c.upper(), "sector": ONS_SERIES[c][0], "measure": "FACE",
                         "first": s.index.min().date(), "last": s.index.max().date(), "freq": "M"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- publication stamping
def uk_holidays(year: int) -> set[pd.Timestamp]:
    """England & Wales bank holidays (rule-based; one-off royal holidays omitted)."""
    from dateutil.easter import easter

    def sub(d: pd.Timestamp) -> pd.Timestamp:
        return d + pd.Timedelta(days={5: 2, 6: 1}.get(d.weekday(), 0))

    def first_mon(m: int) -> pd.Timestamp:
        d = pd.Timestamp(year, m, 1)
        return d + pd.Timedelta(days=(-d.weekday()) % 7)

    def last_mon(m: int) -> pd.Timestamp:
        d = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
        return d - pd.Timedelta(days=d.weekday())

    e = pd.Timestamp(easter(year))
    xmas = pd.Timestamp(year, 12, 25)
    box = pd.Timestamp(year, 12, 26)
    if xmas.weekday() >= 5:
        xmas_obs, box_obs = sub(xmas), sub(xmas) + pd.Timedelta(days=1)
    elif box.weekday() >= 5:
        xmas_obs, box_obs = xmas, sub(box)
    else:
        xmas_obs, box_obs = xmas, box
    return {sub(pd.Timestamp(year, 1, 1)), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
            first_mon(5), last_mon(5), last_mon(8), xmas_obs, box_obs}


def bankstats_release(month) -> pd.Timestamp:
    """The 21st UK working day after the end of `month`."""
    d = pd.Timestamp(month).to_period("M").to_timestamp(how="end").normalize()
    hol = uk_holidays(d.year) | uk_holidays(d.year + 1)
    n = 0
    while n < BANKSTATS_WORKDAY:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            n += 1
    return d


def bankstats_public_month(month) -> pd.Timestamp:
    return bankstats_release(month).to_period("M").to_timestamp()


def ukea_public_month(quarter_end) -> pd.Timestamp:
    """UKEA financial accounts: public in the 3rd month after the quarter-end month (Q2 → Sep)."""
    return (pd.Timestamp(quarter_end).to_period("M") + UKEA_PUBLIC_MONTHS).to_timestamp()


def pusf_public_month(month) -> pd.Timestamp:
    return (pd.Timestamp(month).to_period("M") + PUSF_PUBLIC_MONTHS).to_timestamp()


# ---------------------------------------------------------------- metrics
METRICS: dict[str, dict] = {
    "uk_icpf_f4q": {
        "label": "UK insurer + pension 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "UKEA",
        "what": "ONS NBTF (ICPF transactions in gilts) over four quarters ÷ GDP. Liability-matching, the least "
                "price-elastic private holder (Koijen–Yogo); DB schemes' LDI hedging sits partly here and "
                "partly in OFI (pooled LDI funds)."},
    "uk_ofi_f4q": {
        "label": "UK other-financial 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "UKEA",
        "what": "ONS NCMD (OFI: investment funds, pooled LDI funds, hedge funds, MMFs) over four quarters ÷ GDP. "
                "The most elastic and most levered domestic holder; theory sign (−) is the demand reading, a "
                "carry/RV reading (they buy cheap gilts on repo) would show as 'wrong sign'."},
    "uk_overseas_f4q": {
        "label": "UK overseas 4q net purchases (ONS)", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "UKEA",
        "what": "ONS NEXL (rest-of-world transactions in gilts) over four quarters ÷ GDP."},
    "uk_banks_f4q": {
        "label": "UK private-bank 4q net purchases (ONS ex APF)", "unit": "% of GDP", "sign": -1, "fmt": ".2f",
        "lag": "UKEA",
        "what": "ONS NGCZ (MFI transactions) less APF transactions (purchases/sales at accepted price, "
                "redemptions at par) less the Bank's own book (ΔFZEU), four quarters ÷ GDP."},
    "uk_apf_f4q": {
        "label": "BoE APF 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "UKEA",
        "what": "APF transactions over the same four quarters ÷ GDP (official: QE > 0, QT sales and "
                "redemptions < 0). Stamped with the UKEA quarter for comparability; the monthly version is "
                "`uk_apf_f12m`."},
    "uk_private_f4q": {
        "label": "UK private 4q net absorption", "unit": "% of GDP", "sign": 1, "fmt": ".2f", "lag": "UKEA",
        "what": "Total net issuance at transaction value (ONS NAVT, CG liabilities) less APF and BoE-own "
                "transactions, four quarters ÷ GDP: net supply the private sector (domestic + overseas) had to "
                "absorb (+, like `absorb`)."},
    "uk_icpf_share": {
        "label": "UK insurer + pension share", "unit": "% of gilts", "sign": -1, "fmt": ".1f", "lag": "UKEA",
        "what": "ONS NIZC ÷ (NYXR + NLDU), market value."},
    "uk_icpf_d4q": {
        "label": "UK insurer + pension share, 4q change", "unit": "pp", "sign": -1, "fmt": ".1f", "lag": "UKEA",
        "what": "Four-quarter change in the ICPF share."},
    "uk_ofi_share": {
        "label": "UK other-financial share", "unit": "% of gilts", "sign": 1, "fmt": ".1f", "lag": "UKEA",
        "what": "ONS NJSA ÷ (NYXR + NLDU). Elastic / levered holders (hedge-fund basis and RV, pooled LDI): a "
                "larger share = demand that leaves first in a shock (+)."},
    "uk_overseas_share": {
        "label": "UK overseas share", "unit": "% of gilts", "sign": 1, "fmt": ".1f", "lag": "UKEA",
        "what": "ONS NLDU ÷ (NYXR + NLDU). Foreign holders are the most price-elastic (KKNY), so a high share = "
                "more future supply pressure (+), as `ecb_row_share`."},
    "uk_bank_share": {
        "label": "UK private-bank share", "unit": "% of gilts", "sign": -1, "fmt": ".1f", "lag": "Bankstats+UKEA",
        "what": "Bankstats RPMB3QP (MFIs ex central bank net gilt holdings) at quarter end ÷ (NYXR + NLDU)."},
    "uk_overseas_f3m": {
        "label": "UK overseas net purchases, 3m (BoE)", "unit": "% of GDP", "sign": -1, "fmt": ".2f",
        "lag": "Bankstats",
        "what": "BoE LPMB9P6 non-residents' net purchases of gilts over 3 months ÷ GDP."},
    "uk_overseas_f12m": {
        "label": "UK overseas net purchases, 12m (BoE)", "unit": "% of GDP", "sign": -1, "fmt": ".2f",
        "lag": "Bankstats", "what": "As the 3m measure over 12 months."},
    "uk_bank_f3m": {
        "label": "UK bank net purchases, 3m (BoE)", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "Bankstats",
        "what": "BoE RPMB4CA (UK-resident MFIs ex central bank, change in net gilt positions) over 3 months ÷ GDP."},
    "uk_bank_f12m": {
        "label": "UK bank net purchases, 12m (BoE)", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "Bankstats",
        "what": "As the 3m measure over 12 months."},
    "uk_bank_gdp": {
        "label": "UK bank gilt holdings / GDP", "unit": "% of GDP", "sign": -1, "fmt": ".1f", "lag": "Bankstats",
        "what": "BoE RPMB3QP ÷ GDP: banks' liquid-asset gilt book (grows as reserves shrink under QT)."},
    "uk_m4ps_f3m": {
        "label": "UK M4 private sector net purchases, 3m (BoE)", "unit": "% of GDP", "sign": -1, "fmt": ".2f",
        "lag": "Bankstats",
        "what": "BoE LPMVRDY (M4 private sector = ICPF + OFC + PNFC + households, net purchases of gilts) over "
                "3 months ÷ GDP: the monthly domestic non-bank bid."},
    "uk_m4ps_f12m": {
        "label": "UK M4 private sector net purchases, 12m (BoE)", "unit": "% of GDP", "sign": -1, "fmt": ".2f",
        "lag": "Bankstats", "what": "As the 3m measure over 12 months."},
    "uk_apf_f12m": {
        "label": "BoE APF net purchases, 12m", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "APF",
        "what": "APF transactions over 12 months (QE purchases, QT sales at accepted price, redemptions at par, "
                "2022 financial-stability operations) ÷ GDP; operations are public the day they happen."},
}


def _gdp_quarterly() -> pd.Series:
    from issuance import market

    g = market.load_gdp()
    return g[g["country"] == "UK"].set_index("quarter")["gdp_bn"].sort_index()


def _gdp_at(months, q: pd.Series) -> pd.Series:
    from issuance.headline import gdp_monthly

    return gdp_monthly(pd.DatetimeIndex(months), q)


def quarterly_metrics(qp: pd.DataFrame | None = None, gdp: pd.Series | None = None,
                      boe: pd.DataFrame | None = None) -> pd.DataFrame:
    """country, quarter_end, metric, value, public (month all inputs are public), before ffill."""
    qp = quarterly_panel(boe=boe) if qp is None else qp
    gdp = _gdp_quarterly() if gdp is None else gdp
    idx = pd.date_range(qp.index.min(), qp.index.max(), freq="QE")
    Q = qp.reindex(idx)
    f4 = Q[["icpf", "ofi", "overseas", "banks", "apf", "private"]].rolling(4, min_periods=4).sum()
    pub = pd.DatetimeIndex([ukea_public_month(q) for q in idx])
    gv = _gdp_at(pub, gdp).to_numpy()
    stock = Q["stock_le"].where(Q["stock_le"] > 0)
    icpf_sh = 100 * Q["icpf_le"] / stock
    bank_pub = pd.DatetimeIndex([max(p, bankstats_public_month(q.to_period("M").to_timestamp()))
                                 for p, q in zip(pub, idx)])
    vals = {
        "uk_icpf_f4q": (100 * f4["icpf"] / gv, pub), "uk_ofi_f4q": (100 * f4["ofi"] / gv, pub),
        "uk_overseas_f4q": (100 * f4["overseas"] / gv, pub), "uk_banks_f4q": (100 * f4["banks"] / gv, pub),
        "uk_apf_f4q": (100 * f4["apf"] / gv, pub), "uk_private_f4q": (100 * f4["private"] / gv, pub),
        "uk_icpf_share": (icpf_sh, pub), "uk_icpf_d4q": (icpf_sh - icpf_sh.shift(4), pub),
        "uk_ofi_share": (100 * Q["ofi_le"] / stock, pub), "uk_overseas_share": (100 * Q["overseas_le"] / stock, pub),
        "uk_bank_share": (100 * Q["banks_le"] / stock, bank_pub),
    }
    rows = [pd.DataFrame({"country": "UK", "quarter_end": idx, "metric": k, "value": v.to_numpy(), "public": p})
            for k, (v, p) in vals.items()]
    out = pd.concat(rows, ignore_index=True)
    # No APF before QE: a run of structural zeros would only add ties to the percentile history.
    out = out[(out["metric"] != "uk_apf_f4q") | (out["quarter_end"] >= APF_START)]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])


def monthly_metrics(boe: pd.DataFrame | None = None, apf: pd.DataFrame | None = None,
                    gdp: pd.Series | None = None) -> pd.DataFrame:
    """Bankstats and APF metrics: country, month (data), metric, value, public."""
    boe = boe_panel() if boe is None else boe
    apf = apf_monthly() if apf is None else apf
    gdp = _gdp_quarterly() if gdp is None else gdp
    rows = []

    def add(name, s, public):
        rows.append(pd.DataFrame({"country": "UK", "month": s.index, "metric": name, "value": s.to_numpy(),
                                  "public": public}))

    for series, stem in (("overseas", "uk_overseas"), ("banks", "uk_bank"), ("m4ps", "uk_m4ps")):
        g = boe[(boe["series"] == series) & (boe["measure"] == "F")]
        if g.empty:
            continue
        s = g.set_index("month")["value_gbp_bn"].sort_index().asfreq("MS")
        pub = pd.DatetimeIndex([bankstats_public_month(m) for m in s.index])
        gv = _gdp_at(pub, gdp).to_numpy()
        add(f"{stem}_f3m", 100 * s.rolling(3, min_periods=3).sum() / gv, pub)
        add(f"{stem}_f12m", 100 * s.rolling(12, min_periods=12).sum() / gv, pub)
    lv = boe[(boe["series"] == "banks") & (boe["measure"] == "LE")]
    if not lv.empty:
        s = lv.set_index("month")["value_gbp_bn"].sort_index().asfreq("MS")
        pub = pd.DatetimeIndex([bankstats_public_month(m) for m in s.index])
        add("uk_bank_gdp", 100 * s / _gdp_at(pub, gdp).to_numpy(), pub)
    if not apf.empty:
        s = apf.set_index("month")["apf_f"].sort_index()
        # zeros for the eleven months before the first operation, so the first 12m sum is March 2009
        first = s.index.min()
        s = pd.concat([pd.Series(0.0, index=pd.date_range(first - pd.DateOffset(months=11),
                                                         first - pd.DateOffset(months=1), freq="MS")), s])
        pub = pd.DatetimeIndex(s.index)
        add("uk_apf_f12m", 100 * s.rolling(12, min_periods=12).sum() / _gdp_at(pub, gdp).to_numpy(), pub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["country", "month", "metric", "value", "public"])
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])


def _stamp(df: pd.DataFrame, key: str, end: pd.Timestamp) -> pd.DataFrame:
    """Rows stamped at `public`, forward-filled monthly to `end`; when two data periods share a public
    month the later one is what was public."""
    out = []
    df = df[df["public"] <= end]
    for (m, c), g in df.groupby(["metric", "country"]):
        s = g.sort_values(key).groupby("public")["value"].last()
        idx = pd.date_range(s.index.min(), end, freq="MS")
        s = s.reindex(idx).ffill()
        out.append(pd.DataFrame({"country": c, "month": idx, "metric": m, "value": s.to_numpy(), "phase": "history"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def metrics(today: pd.Timestamp | None = None, qp: pd.DataFrame | None = None, boe: pd.DataFrame | None = None,
            apf: pd.DataFrame | None = None, gdp: pd.Series | None = None) -> pd.DataFrame:
    """Monthly rows in `headline.panel()` shape (country, month, metric, value, phase='history'), each
    value at the month it became public, forward-filled through the last complete month before `today`."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    end = today.to_period("M").to_timestamp() - pd.DateOffset(months=1)
    gdp = _gdp_quarterly() if gdp is None else gdp
    boe = boe_panel() if boe is None else boe
    apf = apf_monthly() if apf is None else apf
    qp = quarterly_panel(boe=boe, apf=apf) if qp is None else qp
    parts = [_stamp(quarterly_metrics(qp, gdp), "quarter_end", end),
             _stamp(monthly_metrics(boe, apf, gdp), "month", end)]
    parts = [x for x in parts if not x.empty]
    cols = ["country", "month", "metric", "value", "phase"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols].sort_values(["metric", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------- absorption
ABSORB_PARTS = {"icpf": "ICPF", "overseas": "Overseas", "banks": "Banks (ex BoE)", "ofi": "OFI (funds, LDI, HF)",
                "households": "Households", "other": "Other domestic", "apf": "BoE APF", "boe_own": "BoE own"}


def absorption(qp: pd.DataFrame | None = None, years=(2022, 2023, 2024, 2025, 2026)) -> pd.DataFrame:
    """Who absorbed UK net gilt supply, by calendar year, £bn (ONS transaction value). net_issuance =
    NAVT; the parts sum to it exactly (other = total − ICPF − OFI − overseas − MFI − households; banks =
    MFI − APF − BoE own). A partial last year is labelled with its last quarter."""
    qp = quarterly_panel() if qp is None else qp
    q = qp.dropna(subset=["total"])
    rows = []
    for y in years:
        g = q[q.index.year == y]
        if g.empty:
            continue
        label = str(y) if len(g) == 4 else f"{y} (to Q{g.index[-1].quarter})"
        row = {"period": label, "basis": "ONS quarterly", "quarters": len(g), "net_issuance": g["total"].sum()}
        for k in ABSORB_PARTS:
            row[k] = g[k].sum()
        row["private"] = g["private"].sum()
        row["parts_sum"] = sum(row[k] for k in ABSORB_PARTS)
        rows.append(row)
    return pd.DataFrame(rows)


def absorption_monthly(boe: pd.DataFrame | None = None, apf: pd.DataFrame | None = None,
                       years=(2022, 2023, 2024, 2025, 2026)) -> pd.DataFrame:
    """The monthly Bankstats split by calendar year, £bn: overseas (LPMB9P6), banks (RPMB4CA), M4 private
    sector (LPMVRDY), APF; total = their sum (excludes the public sector ex BoE and the BoE own book)."""
    boe = boe_panel() if boe is None else boe
    apf = apf_monthly() if apf is None else apf
    f = boe[boe["measure"] == "F"].pivot_table(index="month", columns="series", values="value_gbp_bn")
    f["apf"] = apf.set_index("month")["apf_f"]
    f = f.dropna(subset=["overseas", "banks", "m4ps"])
    rows = []
    for y in years:
        g = f[f.index.year == y]
        if g.empty:
            continue
        label = str(y) if len(g) == 12 else f"{y} (to {g.index[-1]:%b})"
        row = {"period": label, "basis": "BoE monthly", "months": len(g)}
        for k in ("overseas", "banks", "m4ps", "apf"):
            row[k] = g[k].sum()
        row["net_issuance"] = row["overseas"] + row["banks"] + row["m4ps"] + row["apf"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_absorption() -> pd.DataFrame:
    ab = pd.concat([absorption(), absorption_monthly()], ignore_index=True)
    if not ab.empty:
        ab.to_parquet(ABSORB_OUT, index=False)
    return ab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    fetch(a.refresh)
    print(coverage().to_string(index=False))
    print(sector_sums().drop(columns="quarter_end").abs().max().round(3).to_string())
    print(split_check().round(2).to_string(index=False))
    print(write_absorption().round(1).to_string(index=False))
