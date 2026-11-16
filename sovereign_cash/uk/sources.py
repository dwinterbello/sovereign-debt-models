"""UK inputs for the central-government cash framework, all in £bn, monthly.

ONS PUSF series are fetched into ``sovereign_cash/data/uk/ons/`` (never into uk_flows' cache) and
parsed with ``uk_flows.ons._observations``. DMO tables are read from the uk_flows caches/outputs
that ``uk_flows/dmo.py`` maintains; nothing here writes to them.

Linker cash: a tranche raises ``nominal × clean/100 × index ratio``. At redemption the DMO's
financing arithmetic counts ``Σ tranche nominal × ratio at tranche issue`` as principal and puts
the uplift accrued after the last re-opening inside CGNCR (DMR 2026-27 Annex C note 2), so
redemption cash here is that tranche-weighted principal, not nominal × final ratio.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "sovereign_cash" / "data" / "uk"
ONS_CACHE = DATA / "ons"
UKF = REPO / "uk_flows"
UKF_OUT = UKF / "output"
DMO_CACHE = UKF / "cache" / "dmo"
SONIA_CSV = REPO / "projects" / "uk_long_end" / "workbook" / "88_mm_sonia_daily.csv"
STOCK_UK = REPO / "issuance" / "cache" / "stock_uk.parquet"
FORECAST_PRICED = REPO / "issuance" / "cache" / "forecast_priced.parquet"
APF_OPS = REPO / "issuance" / "cache" / "apf_operations.parquet"
DMO_MEDIA = "https://www.dmo.gov.uk"

ONS_URL = ("https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/publicsectorfinance/"
           "timeseries/{cdid}/pusf/data")

# cdid -> (key, label). RUUW is the headline; M98R is the remit aggregate (ex NRAM, B&B, NR).
# ANBV/MF6S/MF6T/JW2P feed the bottom-up CGNCR identity in uk/bottom_up.py.
# ANRT/ANRU/ANRV/ANRS/ANRH/MF7A map the CGNCR residual (PUSF financial transactions).
ONS_SERIES = {
    "ruuw": ("cgncr", "CG net cash requirement"),
    "m98r": ("cgncr_ex", "CG net cash requirement ex NRAM, B&B, Network Rail"),
    "acua": ("nsandi", "CG: National savings (financing flow)"),
    "anta": ("fin_bgs", "Financing of CGNCR: British government securities"),
    "bkpj": ("tbill_level", "CG: sterling Treasury bills"),
    "ansz": ("fin_dep_mfi", "Financing of CGNCR: deposits with MFIs"),
    "antb": ("fin_loans_mfi", "Financing of CGNCR: loans from MFIs"),
    "antc": ("fin_dep_cg", "Financing of CGNCR: deposits with CG by other sectors"),
    "anbv": ("cg_current_receipts", "CG total current receipts (receivable)"),
    "mf6s": ("cg_current_exp", "CG total current expenditure"),
    "mf6t": ("cg_capital_exp", "CG total capital expenditure"),
    "jw2p": ("ps_interest_paid", "PS ex banks: interest & dividends paid to private sector and RoW"),
    "mf6p": ("cg_taxes_receivable", "CG current receipts: taxes receivable"),
    "anrt": ("recv_pay", "CG: Accounts receivable/payable (F7)"),
    "anru": ("gilt_adj", "CG: Adjustment for interest on gilts (F.3)"),
    "anrv": ("other_fin", "CG: Other financial transactions"),
    "anrs": ("acq_sec", "CG: Net acquisition of company securities (F.5)"),
    "anrh": ("net_lend", "CG: Net lending to private sector and rest of world (F.4)"),
    "mf7a": ("apf_xfer", "CG: Capital transfers to Asset Purchase Facility (APF)"),
}
STALE_HOURS = 20


def _ons_obs(payload: dict, cdid: str, key: str, label: str) -> pd.DataFrame:
    if str(UKF) not in sys.path:
        sys.path.insert(0, str(UKF))
    import ons  # uk_flows/ons.py, read-only reuse of its parser

    item = ons.Item(cdid, key, label, "")
    return ons._observations(payload, item)


def fetch_ons(refresh: bool = False) -> dict[str, bool]:
    """Fetch PUSF JSON into DATA/ons. Returns cdid -> fetched-or-cached ok."""
    ONS_CACHE.mkdir(parents=True, exist_ok=True)
    ok: dict[str, bool] = {}
    s = requests.Session()
    s.headers.update({"User-Agent": "sovereign_cash/1.0 (public data)"})
    for cdid in ONS_SERIES:
        path = ONS_CACHE / f"{cdid}.json"
        fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600 < STALE_HOURS
        if path.exists() and (fresh or not refresh):
            ok[cdid] = True
            continue
        try:
            r = s.get(ONS_URL.format(cdid=cdid), timeout=90)
            r.raise_for_status()
            payload = r.json()
            if not payload.get("months"):
                raise ValueError("no monthly block")
            path.write_text(json.dumps(payload), encoding="utf-8")
            ok[cdid] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  ONS {cdid}: {exc}")
            ok[cdid] = path.exists()
    return ok


def ons_monthly() -> pd.DataFrame:
    """Wide monthly frame (£bn) of every cached ONS series, index = month start."""
    cols = {}
    meta = {}
    for cdid, (key, label) in ONS_SERIES.items():
        path = ONS_CACHE / f"{cdid}.json"
        if not path.exists() and cdid == "ruuw":
            path = UKF / "cache" / "ons" / "ruuw.json"
        if not path.exists():
            continue
        obs = _ons_obs(json.loads(path.read_text(encoding="utf-8")), cdid, key, label)
        m = obs[obs["frequency"] == "monthly"]
        if m.empty:
            continue
        cols[key] = m.set_index("date")["value"] / 1000.0
        meta[key] = {"cdid": cdid.upper(), "label": label, "first": m["date"].min(),
                     "last": m["date"].max(), "released": m["released"].max()}
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "month"
    out.attrs["meta"] = meta
    return out


# --------------------------------------------------------------------------------------------
# DMO T-bill stock (D2.2G)

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def bills_stock(path: Path = DMO_CACHE / "bills_stock.xls") -> pd.DataFrame:
    """Month-end T-bill issuance and stock (£bn). FY header rows ('2026-27') date the months."""
    raw = pd.read_excel(path, header=None)
    rows = []
    fy = None
    for r in raw.itertuples(index=False):
        a = str(r[0]).strip()
        if len(a) == 7 and a[4] == "-" and a[:4].isdigit():
            fy = int(a[:4])
            continue
        if fy is None or a[:3] not in _MONTHS:
            continue
        mo = _MONTHS[a[:3]]
        yr = fy if mo >= 4 else fy + 1
        num = pd.to_numeric(pd.Series(list(r[1:9])), errors="coerce")
        rows.append({"month": pd.Timestamp(yr, mo, 1),
                     "bill_1m": num[0], "bill_3m": num[1], "bill_6m": num[2],
                     "bill_adhoc": num[3], "bill_other": num[4],
                     "bill_gross": num[5], "bill_stock": num[6], "bill_stock_debtfin": num[7]})
    df = pd.DataFrame(rows).drop_duplicates("month").sort_values("month").set_index("month")
    return df / 1000.0


BILL_TENDERS_XML = DMO_CACHE / "bill_tenders.xml"


def bill_tenders(path: Path = BILL_TENDERS_XML) -> pd.DataFrame:
    """D2.2D weekly tender results: one row per tender (issue, maturity, size £bn, average yield %)."""
    cols = ["tender_dt", "issue_dt", "maturity_dt", "tenor", "size_bn", "yield_pct"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    import xml.etree.ElementTree as ET

    rows = []
    for di in ET.parse(path).getroot().iter("di"):
        mat, tenor = di.get("MATURITY_DATE"), di.get("TBILL_MATURITY_TYPE")
        for dd in di.iter("dd"):
            for sdd in dd.iter("sdd"):
                far = sdd.find("far")
                y = far.get("AVERAGE_YIELD_PERCENT") if far is not None else None
                rows.append({"tender_dt": pd.Timestamp(dd.get("TENDER_DATE")),
                             "issue_dt": pd.Timestamp(sdd.get("ISSUE_DATE")),
                             "maturity_dt": pd.Timestamp(mat), "tenor": tenor,
                             "size_bn": float(sdd.get("SIZE_MILLIONS")) / 1000.0,
                             "yield_pct": float(y) if y not in (None, "") else np.nan})
    return pd.DataFrame(rows, columns=cols)


def bill_tender_net(t: pd.DataFrame | None = None, cadence_days: int = 6) -> pd.Series:
    """Monthly Δ tender-bill stock (issued − maturing, £bn) for months whose weekly tender calendar is
    complete: the last issue on file falls within ``cadence_days`` of the month end. Ad-hoc and
    cash-management bills outside the weekly tender are not in D2.2D."""
    t = bill_tenders() if t is None else t
    if t.empty:
        return pd.Series(dtype=float)
    iss = t.groupby(t["issue_dt"].dt.to_period("M").dt.to_timestamp())["size_bn"].sum()
    mat = t.groupby(t["maturity_dt"].dt.to_period("M").dt.to_timestamp())["size_bn"].sum()
    net = iss.sub(mat, fill_value=0.0).sort_index()
    last = t["issue_dt"].max()
    month_end = net.index + pd.offsets.MonthEnd(0)
    complete = (net.index <= last) & (month_end - pd.Timedelta(days=cadence_days) <= last)
    return net[complete]


def bill_yield_monthly(t: pd.DataFrame | None = None, tenor: str = "3 Months") -> pd.Series:
    """Size-weighted mean tender yield (%) by issue month for one tenor."""
    t = bill_tenders() if t is None else t
    t = t[(t["tenor"] == tenor) & t["yield_pct"].notna()]
    if t.empty:
        return pd.Series(dtype=float)
    m = t["issue_dt"].dt.to_period("M").dt.to_timestamp()
    w = t["size_bn"] * t["yield_pct"]
    return w.groupby(m).sum() / t["size_bn"].groupby(m).sum()


# --------------------------------------------------------------------------------------------
# RPI / index ratios

def _rpi() -> pd.Series:
    r = pd.read_parquet(UKF_OUT / "dmo_rpi.parquet")
    return r.set_index("rpi_date")["rpi_series_jan_87"].sort_index()


def ref_rpi(dates: pd.Series, lag_months: pd.Series, rpi: pd.Series) -> pd.Series:
    """Reference RPI: 3m lag interpolated daily (post-2005 design), 8m lag monthly (old design).
    Dates past the last RPI print carry the last observed annualised monthly drift."""
    rpi = rpi.copy()
    last = rpi.index.max()
    drift = (rpi.iloc[-1] / rpi.iloc[-13]) ** (1 / 12)
    ext = pd.date_range(last + pd.offsets.MonthBegin(1), periods=48, freq="MS")
    rpi = pd.concat([rpi, pd.Series(rpi.iloc[-1] * drift ** np.arange(1, 49), index=ext)])
    out = []
    for d, lag in zip(pd.to_datetime(dates), lag_months):
        if pd.isna(d):
            out.append(np.nan)
            continue
        m0 = d.to_period("M").to_timestamp()
        if lag == 8:
            out.append(rpi.get(m0 - pd.DateOffset(months=8), np.nan))
            continue
        a = rpi.get(m0 - pd.DateOffset(months=3), np.nan)
        b = rpi.get(m0 - pd.DateOffset(months=2), np.nan)
        frac = (d.day - 1) / d.days_in_month
        out.append(a + frac * (b - a))
    return pd.Series(out, index=dates.index if hasattr(dates, "index") else None, dtype=float)


# --------------------------------------------------------------------------------------------
# Gilt operations → cash

CASH_IN = ("Outright", "Syndication")
CASH_OUT = ("Reverse Auction",)


def gilt_ops_cash() -> pd.DataFrame:
    """Per-operation cash (£bn) for DMO gilt sales and reverse auctions (D2.1E).

    Switch auctions, conversions and cancellations move no Exchequer cash and are dropped, and so
    are unpriced 'Outright' rows: those are stock creations into official holdings (e.g. the
    15-Apr-2025 and 16-Jul-2024 blocks across every line), not sales. ``date`` is already the
    settlement date (21-Apr-2026 auction → 22-Apr); monthly totals match ONS ANTA to £0.01bn in
    months without a redemption.
    """
    o = pd.read_parquet(UKF_OUT / "operations_priced.parquet")
    o = o[o["mechanism"].isin(CASH_IN + CASH_OUT)].copy()
    o = o[~(o["mechanism"].isin(CASH_IN) & o["issue_price"].isna())]
    o["linker"] = o["linker"].astype("boolean").fillna(False).astype(bool)
    o["lag"] = np.where(o["indexation_lag"].astype(str).str.startswith("8"), 8,
                        np.where(o["linker"], 3, 0))
    first = (o[o["mechanism"].isin(CASH_IN)].groupby("isin")["date"].min()
             .rename("first_issue"))
    o = o.join(first, on="isin")
    o["settle"] = o["date"]
    rpi = _rpi()
    o["ratio"] = 1.0
    lk = o["linker"]
    if lk.any():
        num = ref_rpi(o.loc[lk, "settle"], o.loc[lk, "lag"], rpi)
        den = ref_rpi(o.loc[lk, "first_issue"], o.loc[lk, "lag"], rpi)
        o.loc[lk, "ratio"] = (num / den).fillna(1.0).to_numpy()
    price = o["issue_price"].fillna(100.0)
    o["cash_bn"] = o["nominal_mn"] * price / 100.0 * o["ratio"] / 1000.0
    o.loc[o["mechanism"].isin(CASH_OUT), "cash_bn"] = -o["nominal_mn"].abs() * price / 100.0 \
        * o["ratio"] / 1000.0
    o["principal_bn"] = o["nominal_mn"] * o["ratio"] / 1000.0
    return o


def gilt_redemptions(ops: pd.DataFrame | None = None) -> pd.DataFrame:
    """Redemption cash (£bn) per gilt: nominal at redemption × tranche-weighted issue ratio."""
    red = pd.read_parquet(UKF_OUT / "dmo_redeemed.parquet")
    ops = gilt_ops_cash() if ops is None else ops
    names = pd.read_parquet(UKF_OUT / "operations_priced.parquet")[["name", "isin"]] \
        .drop_duplicates("name")
    red = red.merge(names, on="name", how="left")
    ins = ops[ops["mechanism"].isin(CASH_IN)]
    w = ins.groupby("isin").apply(
        lambda g: float((g["nominal_mn"] * g["ratio"]).sum() / g["nominal_mn"].sum()),
        include_groups=False).rename("avg_ratio")
    red = red.join(w, on="isin")
    is_link = red["name"].str.contains("Index-linked", case=False, na=False)
    red["avg_ratio"] = np.where(is_link, red["avg_ratio"].fillna(1.0), 1.0)
    red["linker"] = is_link
    red["month"] = red["redemption_date"].dt.to_period("M").dt.to_timestamp()
    held = official_holdings()
    prior = red["month"] - pd.offsets.MonthEnd(1)
    red["gov_held_mn"] = [held.get((i, p), 0.0) for i, p in zip(red["isin"], prior)]
    red["gross_cash_bn"] = red["nominal_at_redemption_mn"] * red["avg_ratio"] / 1000.0
    red["cash_bn"] = (red["nominal_at_redemption_mn"] - red["gov_held_mn"]).clip(lower=0) \
        * red["avg_ratio"] / 1000.0
    red["apf_held_mn"] = [apf_held(i, d) for i, d in zip(red["isin"], red["redemption_date"])]
    red["cb_cash_bn"] = _cb_cash(red["apf_held_mn"], red["nominal_at_redemption_mn"] - red["gov_held_mn"],
                                 red["avg_ratio"])
    return red


def _cb_cash(apf_mn: pd.Series, outside_gov_mn: pd.Series, ratio: pd.Series) -> pd.Series:
    """Redemption cash paid to the APF: its nominal (capped at what CRND/DMO do not hold) × ratio."""
    return np.minimum(apf_mn.fillna(0.0), outside_gov_mn.clip(lower=0)) * ratio / 1000.0


_APF: pd.DataFrame | None = None


def _apf_ops() -> pd.DataFrame:
    global _APF
    if _APF is None:
        if APF_OPS.exists():
            a = pd.read_parquet(APF_OPS, columns=["op_dt", "isin", "apf_stock"])
            a["op_dt"] = pd.to_datetime(a["op_dt"])
            a["apf_stock"] = pd.to_numeric(a["apf_stock"], errors="coerce")
            _APF = a.dropna(subset=["isin"]).sort_values("op_dt")
        else:
            _APF = pd.DataFrame(columns=["op_dt", "isin", "apf_stock"])
    return _APF


def apf_held(isin, asof) -> float:
    """APF nominal (£mn) in ``isin`` after its last operation before ``asof`` (BoE APF ISIN table via
    issuance/cache/apf_operations). The APF does not trade gilts inside three years of maturity, so the
    last running stock is what it holds at redemption."""
    if pd.isna(isin):
        return 0.0
    a = _apf_ops()
    g = a[(a["isin"] == isin) & (a["op_dt"] < pd.Timestamp(asof))]
    return float(g["apf_stock"].iloc[-1]) if len(g) else 0.0


def official_holdings() -> dict[tuple[str, pd.Timestamp], float]:
    """CRND + DMO holdings (D1D, nominal £mn) keyed by (isin, month_end). Redemptions of
    government-held stock move no Exchequer cash, so they are netted (DMO and ONS both do)."""
    path = UKF_OUT / "dmo_official_holdings.parquet"
    if not path.exists():
        return {}
    h = pd.read_parquet(path)
    h["month_end"] = pd.to_datetime(h["month_end"]) + pd.offsets.MonthEnd(0)
    g = h.groupby(["isin", "month_end"])["holding_nominal_mn"].sum()
    return g.to_dict()


def forward_redemptions(asof: pd.Timestamp, ops: pd.DataFrame | None = None) -> pd.DataFrame:
    """Future redemption cash by month from the DMO stock snapshot (issuance/cache/stock_uk).

    Gilts drop off the stock the day they mature, so a snapshot taken after a mid-month
    redemption leaves a hole between the cash-history cutoff (``asof``) and the stock as-of.
    Those lines are spliced back in from ``dmo_redeemed`` via ``gilt_redemptions``.
    """
    st = pd.read_parquet(STOCK_UK)
    st["maturity_dt"] = pd.to_datetime(st["maturity_dt"])
    st = st[st["maturity_dt"] > asof].copy()
    ops = gilt_ops_cash() if ops is None else ops
    ins = ops[ops["mechanism"].isin(CASH_IN)]
    w = ins.groupby("isin").apply(
        lambda g: float((g["nominal_mn"] * g["ratio"]).sum() / g["nominal_mn"].sum()),
        include_groups=False)
    st["avg_ratio"] = np.where(st["sec_type"] == "Linker",
                               st["isin"].map(w).fillna(1.0), 1.0)
    held = official_holdings()
    latest = {}
    for (isin, me), v in held.items():
        if isin not in latest or me > latest[isin][0]:
            latest[isin] = (me, v)
    st["gov_held_mn"] = st["isin"].map({k: v[1] for k, v in latest.items()}).fillna(0.0)
    st["cash_bn"] = (st["amount_nominal_mn"] - st["gov_held_mn"]).clip(lower=0) \
        * st["avg_ratio"] / 1000.0
    st["apf_held_mn"] = [apf_held(i, d) for i, d in zip(st["isin"], st["maturity_dt"])]
    st["cb_cash_bn"] = _cb_cash(st["apf_held_mn"], st["amount_nominal_mn"] - st["gov_held_mn"], st["avg_ratio"])
    st["month"] = st["maturity_dt"].dt.to_period("M").dt.to_timestamp()

    past = gilt_redemptions(ops)
    past = past[past["redemption_date"] > asof].copy()
    if len(past):
        past = past[~past["isin"].isin(st["isin"])]
        past = past.rename(columns={"redemption_date": "maturity_dt",
                                    "nominal_at_redemption_mn": "amount_nominal_mn"})
        past["sec_type"] = np.where(past["linker"], "Linker", "Conventional")
        past["as_of"] = st["as_of"].iloc[0] if len(st) else pd.NaT
        past["country"] = "UK"
        cols = [c for c in st.columns if c in past.columns]
        st = pd.concat([st, past[cols]], ignore_index=True)
    return st


def forward_gilt_plan() -> pd.DataFrame:
    """Forward UK gilt operations (nominal, £bn) from issuance/forward.py's priced plan."""
    if not FORECAST_PRICED.exists():
        return pd.DataFrame(columns=["month", "sec_type", "nominal_bn"])
    f = pd.read_parquet(FORECAST_PRICED)
    f = f[f["country"] == "UK"].copy()
    f["month"] = pd.to_datetime(f["op_dt"] + pd.offsets.BDay(1)).dt.to_period("M").dt.to_timestamp()
    f["nominal_bn"] = f["amount_mn"] / 1000.0
    return f[["month", "op_dt", "sec_type", "isin", "nominal_bn", "source"]]


def sonia_bank_monthly() -> pd.DataFrame:
    """Monthly mean SONIA − Bank Rate (bp) and month-end value from the long-end workbook."""
    if not SONIA_CSV.exists():
        return pd.DataFrame()
    s = pd.read_csv(SONIA_CSV, parse_dates=["date"]).set_index("date")
    m = s["sonia_bank_spread_bp"].resample("MS")
    return pd.DataFrame({"sonia_bank_bp": m.mean(), "sonia_bank_bp_end": m.last(),
                         "sonia_pct": s["sonia"].resample("MS").mean()})


def dmr_anchors() -> pd.DataFrame:
    path = DATA / "dmr_anchors.csv"
    return pd.read_csv(path, comment="#")


def remit_vintages() -> pd.DataFrame:
    """CGNCR(ex) vintages from DMO remit announcements (hand-entered, sovereign_cash/data/uk)."""
    v = pd.read_csv(DATA / "remit_vintages.csv", comment="#", parse_dates=["vintage_date"])
    v["fy_start"] = v["fy"].str[:4].astype(int)
    v["comparable"] = v["comparable"].astype(bool)
    return v.sort_values(["fy_start", "vintage_date"]).reset_index(drop=True)
