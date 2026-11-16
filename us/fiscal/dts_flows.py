"""Daily Treasury Statement Table II, every line, bucketed for flow forecasting.

Table II (Deposits and Withdrawals of Operating Cash) is the only daily,
line-item view of federal cash. The TGA closes each day at

    TGA_t = TGA_{t-1} + total deposits_t − total withdrawals_t

and the public-debt lines (Table IIIB cash issues / redemptions) are the
financing. Everything else is the operating flow this module models.

Line names drift: the Oct-2022 DTS redesign renamed most agency lines
("Social Security Benefits (EFT)" → "SSA - Benefits Payments"), and Sep-2022
dropped the hyphen in "III-B". Buckets are keyword rules over both vintages;
`bucket_coverage()` reports how much of each year lands in `*_other` so a
rename that slips through is visible rather than silent. Lines that only
exist in one vintage (Thrift Savings Plan, split out in 2022) stay in
`*_other` so each bucket is one continuous series.

  PYTHONPATH=. .venv/bin/python us/fiscal/dts_flows.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cache"
DERIVED = HERE / "derived"

URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    "/v1/accounting/dts/deposits_withdrawals_operating_cash"
)
FIELDS = "record_date,account_type,transaction_type,transaction_catg,transaction_today_amt"
START = "2012-10-01"  # FY2013: enough tax-date cycles, avoids pre-2012 TT&L format
RAW_PATH = CACHE / "dts_table2_raw.parquet"
UA = {"User-Agent": "us-fiscal/dts-flows (research; local)"}
PAGE = 10000
REFETCH_DAYS = 10  # DTS lines are occasionally restated within a few days

# Before the 2023-02-14 redesign every tax deposit arrived on one Table II line
# ("Cash FTD's Received (Table IV)"); Table IV carries the per-type split.
REDESIGN = pd.Timestamp("2023-02-14")
TABLE4_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    "/v1/accounting/dts/federal_tax_deposits"
)
FTD_LINE = re.compile(r"cash ftd", re.I)
TABLE4_MAP = {
    "Withheld Income and Employment Taxes": "dep_withheld",
    "Individual Income Taxes": "dep_nonwithheld",
    "Corporation Income Taxes": "dep_corporate",
    "Railroad Retirement Taxes": "dep_other_tax",
    "Excise Taxes": "dep_other_tax",
    "Federal Unemployment Taxes": "dep_other_tax",
    "Estate and Gift Taxes & Misc IRS Rcpts.": "dep_other_tax",
}

FINANCING_ISSUE = re.compile(r"public debt cash issues", re.I)
FINANCING_REDEEM = re.compile(r"public debt cash redemp", re.I)
# Totals and internal transfers that are not flows between the TGA and the public.
NON_FLOW = re.compile(
    r"^(null|total|sub-total|subtotal)|total (deposits|withdrawals|other)|"
    r"transfers? (to|from) (depositaries|federal reserve|tt&l|the federal reserve)|"
    r"change in balance of uncollected|from: depositaries|to: depositaries",
    re.I,
)

# (bucket, transaction_type, regex). First match wins; order matters.
BUCKET_RULES: list[tuple[str, str, str]] = [
    ("dep_withheld", "Deposits", r"withheld individual|withheld income and employment"),
    ("dep_nonwithheld", "Deposits", r"non.?withheld ind|individual income and employment taxes, not withheld"),
    ("dep_corporate", "Deposits", r"corporat\w* income tax|taxes - corporate"),
    ("dep_customs", "Deposits", r"customs"),
    ("dep_fed_earnings", "Deposits", r"federal reserve earnings"),
    ("dep_other_tax", "Deposits", r"^taxes|excise|estate and gift|unemployment (taxes|insurance deposits)|^futa"),
    ("dep_medicare_prem", "Deposits", r"medicare premium"),
    ("dep_education", "Deposits", r"education"),
    ("wd_ssa", "Withdrawals", r"social security benefits|ssa - benefits|ssa - supplemental|supple?mental security income"),
    ("wd_medicare", "Withdrawals", r"medicare|hospital insr|supple med insr"),
    ("wd_medicaid", "Withdrawals", r"medicaid"),
    ("wd_interest", "Withdrawals", r"interest on treasury securities"),
    ("wd_refunds", "Withdrawals", r"tax refunds"),
    ("wd_defense", "Withdrawals", r"defense|dod|military"),
    ("wd_va", "Withdrawals", r"veterans|^va "),
    ("wd_salaries", "Withdrawals", r"federal salaries"),
    ("wd_opm", "Withdrawals", r"civil serv retirement|opm - |federal employee insurance|federal employees insurance"),
    ("wd_education", "Withdrawals", r"education"),
    ("wd_unemployment", "Withdrawals", r"unemployment"),
    ("wd_snap", "Withdrawals", r"snap|supp nutrition|food stamps|nutrition"),
    ("wd_hhs_other", "Withdrawals", r"hhs|health|children and families|nih"),
]
_COMPILED = [(b, t, re.compile(rx, re.I)) for b, t, rx in BUCKET_RULES]

DEP_BUCKETS = [b for b, t, _ in BUCKET_RULES if t == "Deposits"] + ["dep_other"]
WD_BUCKETS = [b for b, t, _ in BUCKET_RULES if t == "Withdrawals"] + ["wd_other"]
OP_BUCKETS = DEP_BUCKETS + WD_BUCKETS


def _get(params: dict, tries: int = 4, url: str = URL) -> dict:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * 2**i)
    raise RuntimeError(f"Fiscal Data Table II: {last}")


def _download(since: str, verbose: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get(
            {
                "fields": FIELDS,
                "filter": f"record_date:gte:{since}",
                "sort": "record_date",
                "page[size]": PAGE,
                "page[number]": page,
                "format": "json",
            }
        )
        chunk = payload.get("data") or []
        rows.extend(chunk)
        total = int(payload.get("meta", {}).get("total-pages") or 1)
        if verbose and (page == 1 or page % 10 == 0 or page == total):
            print(f"   Table II since {since}: page {page}/{total}")
        if page >= total or not chunk:
            break
        page += 1
    df = pd.DataFrame(rows, columns=FIELDS.split(","))
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["amt_mn"] = pd.to_numeric(df["transaction_today_amt"].replace("null", np.nan), errors="coerce")
    return df.drop(columns=["transaction_today_amt"])


def fetch_raw(*, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Full Table II since START; incremental after the first pull."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        old = pd.read_parquet(RAW_PATH)
        fresh = True
        if fresh and not refresh:
            return old
        since = (old["record_date"].max() - pd.Timedelta(days=REFETCH_DAYS)).strftime("%Y-%m-%d")
        try:
            new = _download(since, verbose)
        except RuntimeError as exc:
            if verbose:
                print(f"   {exc}; using cached Table II")
            return old
        df = pd.concat([old[old["record_date"] < since], new], ignore_index=True)
    else:
        df = _download(START, verbose)
    df = df.sort_values(["record_date", "transaction_type"]).reset_index(drop=True)
    df.to_parquet(RAW_PATH, index=False)
    return df


def fetch_table4(*, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Table IV federal tax deposit split, START → redesign (fixed history, cached once)."""
    path = CACHE / "dts_table4_pre_redesign.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get(
            {
                "fields": "record_date,tax_deposit_type,tax_deposit_type_desc,tax_deposit_today_amt",
                "filter": f"record_date:gte:{START},record_date:lt:{REDESIGN.date()}",
                "sort": "record_date",
                "page[size]": PAGE,
                "page[number]": page,
                "format": "json",
            },
            url=TABLE4_URL,
        )
        chunk = payload.get("data") or []
        rows.extend(chunk)
        if page >= int(payload.get("meta", {}).get("total-pages") or 1) or not chunk:
            break
        page += 1
    df = pd.DataFrame(rows)
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["amt_mn"] = pd.to_numeric(df["tax_deposit_today_amt"].replace("null", np.nan), errors="coerce")
    df = df.drop(columns=["tax_deposit_today_amt"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if verbose:
        print(f"   Table IV split: {df['record_date'].nunique():,} days")
    return df


def ftd_shares(table4: pd.DataFrame) -> pd.DataFrame:
    """Daily share of tax deposits by bucket, from the Table IV tax-type lines."""
    t = table4[table4["tax_deposit_type"].isin(TABLE4_MAP)].copy()
    t["bucket"] = t["tax_deposit_type"].map(TABLE4_MAP)
    w = t.pivot_table(index="record_date", columns="bucket", values="amt_mn", aggfunc="sum").fillna(0.0).clip(lower=0)
    tot = w.sum(axis=1).replace(0.0, np.nan)
    return w.div(tot, axis=0).dropna(how="all")


def classify(catg: str, ttype: str) -> str:
    """Bucket for one Table II line: op bucket, 'fin_issue', 'fin_redeem' or 'skip'."""
    c = str(catg or "").strip()
    if FINANCING_ISSUE.search(c):
        return "fin_issue"
    if FINANCING_REDEEM.search(c):
        return "fin_redeem"
    if not c or NON_FLOW.search(c):
        return "skip"
    if ttype == "Deposits" and FTD_LINE.search(c):
        return "dep_ftd"
    for bucket, t, rx in _COMPILED:
        if t == ttype and rx.search(c):
            return bucket
    if ttype == "Deposits":
        return "dep_other"
    if ttype == "Withdrawals":
        return "wd_other"
    return "skip"


def _tga_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the account that is the TGA in each vintage (Fed account pre-2021, TGA after)."""
    acct = raw["account_type"].astype(str)
    keep = acct.str.contains("Treasury General Account|Federal Reserve Account", case=False, regex=True)
    return raw.loc[keep & raw["transaction_type"].isin(["Deposits", "Withdrawals"])]


def bucket_panel(raw: pd.DataFrame, shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Business-daily panel of bucket flows, $mn. Deposits positive, withdrawals positive.

    `shares` (from `ftd_shares`) splits the pre-redesign combined tax-deposit
    line; without it the line lands in `dep_withheld`.
    """
    df = _tga_rows(raw).copy()
    pairs = df[["transaction_catg", "transaction_type"]].drop_duplicates()
    pairs["bucket"] = [classify(c, t) for c, t in zip(pairs["transaction_catg"], pairs["transaction_type"])]
    df = df.merge(pairs, on=["transaction_catg", "transaction_type"], how="left")
    df = df[df["bucket"] != "skip"]
    wide = df.pivot_table(index="record_date", columns="bucket", values="amt_mn", aggfunc="sum").fillna(0.0)
    if "dep_ftd" in wide.columns:
        ftd = wide.pop("dep_ftd")
        if shares is None or shares.empty:
            shares = pd.DataFrame({"dep_withheld": 1.0}, index=ftd.index)
        s = shares.reindex(ftd.index)
        s = s.fillna(s.ffill()).fillna({"dep_withheld": 1.0}).fillna(0.0)
        for col in s.columns:
            wide[col] = wide.get(col, 0.0) + ftd * s[col]
    for col in OP_BUCKETS + ["fin_issue", "fin_redeem"]:
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide[OP_BUCKETS + ["fin_issue", "fin_redeem"]].sort_index()
    wide["dep_total"] = wide[DEP_BUCKETS].sum(axis=1)
    wide["wd_total"] = wide[WD_BUCKETS].sum(axis=1)
    wide["op_net"] = wide["dep_total"] - wide["wd_total"]
    wide["fin_net"] = wide["fin_issue"] - wide["fin_redeem"]
    wide["tga_change_flows"] = wide["op_net"] + wide["fin_net"]
    wide.index.name = "date"
    return wide


def identity_check(panel: pd.DataFrame, tga_mn: pd.Series) -> pd.DataFrame:
    """Flow-implied ΔTGA vs published ΔTGA close; residual in $mn."""
    tga = tga_mn.sort_index()
    d = tga.diff().reindex(panel.index)
    out = pd.DataFrame({"d_tga": d, "flows": panel["tga_change_flows"]})
    out["resid"] = out["flows"] - out["d_tga"]
    return out


def bucket_coverage(raw: pd.DataFrame, shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of gross operating flow per bucket per calendar year — watch *_other drift."""
    p = bucket_panel(raw, shares)
    gross = p[OP_BUCKETS].abs()
    yr = gross.groupby(gross.index.year).sum()
    return yr.div(yr.sum(axis=1), axis=0)


def continuity_starts(raw: pd.DataFrame, shares: pd.DataFrame | None = None, share: float = 0.04,
                      jump: tuple[float, float] = (0.85, 1.20), window: int = 250) -> dict[str, str]:
    """First date from which each bucket is one continuous series.

    Candidate breaks are dates a material line (daily mean ≥ `share` of the
    bucket's) enters or leaves. A rename shows up as a candidate too, so a
    candidate only counts if the bucket's mean over `window` business days
    after it differs from the mean before it by more than `jump` (a real
    composition change moves the level; a relabel does not).
    """
    panel = bucket_panel(raw, shares)
    df = _tga_rows(raw)
    df = df[df["record_date"] >= START]
    pairs = df[["transaction_catg", "transaction_type"]].drop_duplicates()
    pairs["bucket"] = [classify(c, t) for c, t in zip(pairs["transaction_catg"], pairs["transaction_type"])]
    df = df.merge(pairs, on=["transaction_catg", "transaction_type"])
    df = df[~df["bucket"].isin(["skip", "fin_issue", "fin_redeem", "dep_ftd"])]
    first_day, last_day = df["record_date"].min(), df["record_date"].max()
    out: dict[str, str] = {}
    for b, g in df.groupby("bucket"):
        lines = g.groupby("transaction_catg").agg(first=("record_date", "min"), last=("record_date", "max"),
                                                 tot=("amt_mn", lambda s: s.abs().sum()), n=("record_date", "nunique"))
        lines["daily"] = lines["tot"] / lines["n"].clip(lower=1)
        bucket_daily = g.groupby("record_date")["amt_mn"].sum().abs().mean()
        material = lines[lines["daily"] >= share * bucket_daily]
        starts = material.loc[material["first"] > first_day + pd.Timedelta(days=7), "first"]
        ends = material.loc[material["last"] < last_day - pd.Timedelta(days=7), "last"] + pd.Timedelta(days=1)
        real = []
        s = panel[b] if b in panel.columns else None
        for d in sorted(set(pd.concat([starts, ends]))):
            if s is None:
                break
            before = s[s.index < d].tail(window)
            after = s[s.index >= d].head(window)
            if len(before) < 60 or len(after) < 60 or before.mean() == 0:
                real.append(d)
                continue
            r = after.mean() / before.mean()
            if not (jump[0] <= r <= jump[1]):
                real.append(d)
        out[b] = str(max(real).date()) if real else START
    return out


# Reviewed against the line histories (2026-10): the detector cannot tell a
# relabel during a real level swing from a composition change.
CONTINUITY_OVERRIDE = {
    "dep_customs": START,  # relabel 2025-11-10; level jump is tariffs
    "dep_nonwithheld": START,  # continuous via the Table IV split; 2022 peak is capital gains
    "dep_education": START,  # relabel 2019-11-12; jump is the 2020 loan pause
    "wd_ssa": "2021-07-19",  # SSI moved in from "Other Withdrawals"
    "wd_defense": "2023-10-02",  # military pay / retirement itemized
    "wd_other": "2023-10-02",  # lost SSI, military pay, retirement over 2021-2023
}


def train_starts(raw: pd.DataFrame, shares: pd.DataFrame | None = None) -> dict[str, str]:
    out = continuity_starts(raw, shares)
    out.update(CONTINUITY_OVERRIDE)
    return out


def unmatched_lines(raw: pd.DataFrame, since: str = "2022-10-01", top: int = 40) -> pd.DataFrame:
    """Largest lines falling into *_other since `since` (for rule maintenance)."""
    df = _tga_rows(raw)
    df = df[df["record_date"] >= since]
    g = df.groupby(["transaction_type", "transaction_catg"])["amt_mn"].sum().reset_index()
    g["bucket"] = [classify(c, t) for c, t in zip(g["transaction_catg"], g["transaction_type"])]
    g = g[g["bucket"].isin(["dep_other", "wd_other"])]
    return g.sort_values("amt_mn", ascending=False).head(top)


# SSA bucket mixes two calendars: Wed Benefits (~$30bn) and 1st-of-month SSI (~$5bn).
# Historical label before the 2021-07 Table II redesign is the EFT line.
SSA_BENEFITS_LINES = ("SSA - Benefits Payments", "Social Security Benefits (EFT)")
SSA_SSI_LINES = ("SSA - Supplemental Security Income",)


def ssa_parts_panel(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily $mn Benefits and SSI series aligned to ``index`` (zeros where absent)."""
    df = _tga_rows(raw)

    def _sum(names: tuple[str, ...]) -> pd.Series:
        m = df["transaction_catg"].isin(names)
        if not m.any():
            return pd.Series(0.0, index=index)
        return (
            df.loc[m]
            .groupby("record_date")["amt_mn"]
            .sum()
            .reindex(index)
            .fillna(0.0)
            .astype(float)
        )

    return pd.DataFrame({"benefits": _sum(SSA_BENEFITS_LINES), "ssi": _sum(SSA_SSI_LINES)}, index=index)


def run(*, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    raw = fetch_raw(refresh=refresh, verbose=verbose)
    shares = ftd_shares(fetch_table4(verbose=verbose))
    panel = bucket_panel(raw, shares)
    DERIVED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(DERIVED / "dts_bucket_daily.parquet")
    ssa_parts_panel(raw, panel.index).to_parquet(DERIVED / "dts_ssa_parts.parquet")
    (DERIVED / "dts_train_starts.json").write_text(json.dumps(train_starts(raw, shares), indent=1))
    if verbose:
        print(f"DTS Table II buckets: {len(panel):,} days {panel.index.min().date()} → {panel.index.max().date()}")
    return panel


if __name__ == "__main__":
    run(refresh=True)
