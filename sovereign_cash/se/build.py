"""Sweden contract outputs: ``derived/se_cash_monthly.parquet`` and ``derived/se_cash_summary.json``.

History identity (bn SEK, inflow to the buffer +):

    Δbuffer = −NBR + net_bonds + net_bills + other_fin

buffer        Riksgälden's net liquidity-management position: Δbuffer = − net borrowing in
              liquidity-management instruments (repos, deposits, on-tap bills, FX commercial paper).
              Sweden has no TGA-style cash balance; liquidity management is where the residual
              daily cash need is financed (Riksgälden, Statsupplåning 2026:1).
NBR           net borrowing requirement (= −budget balance), CH1089.
net_bonds     net borrowing at nominal amount in nominal SGBs + green + linkers + FX bonds (CH1044),
              split into net_bonds_nominal_bn (SGB + green), net_bonds_linker_bn, net_bonds_fx_bn.
net_bills     net borrowing in T-bills (CH1044).
other_fin     total net borrowing − bonds − bills − liquidity management − business-day adjustment
              (= other capital-market debt + collateral + retail + derivatives + unitemized lines).

CH1044 is Riksgälden's own decomposition of the debt change: net borrowing at *nominal* amount by
instrument, with FX revaluation, accrued inflation compensation and money-market asset changes split
out as "debt dispositions". That is stock differencing with the valuation already removed by the
source; the valuation lines are carried as information columns, not as financing. The identity
residual is NBR(CH1089, mn) − NBR(CH1044, SEK): a genuine cross-table check, not a plug.

Gross: nominal-SGB and linker auction allocations by settlement month (T+2; linkers are the
``SGB IL`` series in the same auction table) + FX-bond net borrowing when positive; bill_gross = T-bill
auction allocations. Redemptions are gross − net (they absorb switches and buybacks).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd

from sovereign_cash import contract
from sovereign_cash.contract import fmt_bn, fmt_num
from sovereign_cash.se import fetch, model

RAW = fetch.RAW
DERIVED = contract.DERIVED
OUT_MONTHLY = DERIVED / "se_cash_monthly.parquet"
OUT_SUMMARY = DERIVED / "se_cash_summary.json"
OUT_BACKTEST = DERIVED / "se_backtest.parquet"
OUT_RG = DERIVED / "se_riksgalden_forecasts.parquet"

# Riksgälden, Statsupplåning – prognos och analys 2026:1. Update each May / November.
RG_REPORT = {"label": "Statsupplåning 2026:1", "published": pd.Timestamp("2026-05-28"),
             "last_outcome": pd.Timestamp("2026-04-01")}  # last monthly outcome known when the report was made
RG_VINTAGE = f"{RG_REPORT['label']} ({RG_REPORT['published']:%Y-%m-%d})"
RG_PLAN_URL = "https://www.riksgalden.se/contentassets/27fcc0e426c54e508344a41630b84c99/statsupplaning-2026-1.pdf"
RG_ANNUAL_BALANCE = {2026: -196.0, 2027: -208.0}  # Tabell 3, budget balance = − net borrowing requirement
RG_PLAN = {  # bn SEK, Tabell 1 / 11
    "nominal_bonds": {2026: 220.0, 2027: 240.0},
    "linkers": {2026: 6.0, 2027: 6.0},
    "fx_bonds": {2026: 19.0, 2027: 19.0},
    "bill_stock_end": {2026: 238.0, 2027: 233.0},
    "liq_mgmt_stock_end": {2026: 173.0, 2027: 151.0},
}
# Outcome stocks from the same report (Tabell 1 / 11 / 20: "initial stock" = previous year end).
RG_STOCK_OUTCOME = {"bill_stock_end": {2024: 113.0, 2025: 173.0}, "liq_mgmt_stock_end": {2024: 134.0, 2025: 137.0}}
# Annual vintages for scoring the shipped forward: (label, last outcome month known, {year: balance}).
RG_ANNUAL_VINTAGES = [
    ("Statsupplåning 2025:2", pd.Timestamp("2025-10-01"), {2025: -103.0, 2026: -173.0}),
    ("Statsupplåning 2026:1", pd.Timestamp("2026-04-01"), {2026: -196.0, 2027: -208.0}),
]
NEXT_REPORT_FALLBACK = pd.Timestamp("2026-11-26")  # used only if no release announces the date
# Items Riksgälden's outcome releases call timing within its forecast year: the month they were planned in,
# the year they are now expected in, and the amount (balance terms, inflow +). While an item is outstanding
# its shortfall sits in the YTD deviation but not in Riksgälden's full year, so it is added back to the carry.
# Set received_month once an outcome release reports the payment.
TIMING_ITEMS = (
    {"item": "EU Recovery and Resilience Facility payment", "planned_month": "2026-07", "expected_year": 2026,
     "amount_bn": 15.0, "announced": "2026-08-07", "received_month": None,
     "source": "https://mfn.se/a/riksgalden/deficit-for-swedish-central-government-in-july-2026"},
)
SE_FIXED_HOLIDAYS = {(1, 1), (1, 6), (5, 1), (6, 6), (12, 24), (12, 25), (12, 26), (12, 31)}
SOURCES = [
    {"name": "Riksgälden net borrowing requirement / budget balance, monthly (QlikView Webb_Upplaning CH1089)",
     "url": "https://www.riksgalden.se/en/statistics/statistics-on-central-government-borrowing/net-borrowing-requirement/",
     "frequency": "monthly", "lag_days": 7, "verified": True},
    {"name": "Riksgälden central-government debt change by instrument (Webb_Upplaning CH1044)",
     "url": "https://www.riksgalden.se/en/statistics/statistics-regarding-swedens-central-government-debt/",
     "frequency": "monthly", "lag_days": 7, "verified": True},
    {"name": "Riksgälden auction results (webb auktionsresultat CH926)",
     "url": "https://www.riksgalden.se/en/statistics/statistics-regarding-government-securities/auction-results/",
     "frequency": "per auction", "lag_days": 0, "verified": True},
    {"name": "Riksgälden outstanding government securities (webb_upplaning_daglig CH1073)",
     "url": "https://www.riksgalden.se/en/statistics/statistics-regarding-government-securities/outstanding-government-securities/",
     "frequency": "daily snapshot", "lag_days": 1, "verified": True},
    {"name": "Riksgälden monthly outcome press releases (outcome, own monthly forecast, cumulative deviation), MFN feed",
     "url": fetch.MFN_URL, "frequency": "monthly", "lag_days": 7, "verified": True},
    {"name": f"Riksgälden {RG_VINTAGE}: annual forecast, borrowing plan and year-end stocks (hand-entered)",
     "url": RG_PLAN_URL, "frequency": "semi-annual", "lag_days": 0, "verified": True},
    {"name": "Riksgälden outcome releases: timing items within the forecast year (hand-entered, dated)",
     "url": TIMING_ITEMS[0]["source"], "frequency": "as announced", "lag_days": 0, "verified": True},
]
_MON = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
        "okt": 10, "oct": 10, "nov": 11, "dec": 12}


def _settle_month(d: pd.Series) -> pd.Series:
    return (d + BDay(2)).dt.to_period("M").dt.to_timestamp()


def _month_of(t: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(t.year, t.month, 1)


def _md(t: pd.Timestamp) -> tuple[int, int]:
    return (t.month, t.day)


def cadence_days(dates) -> int:
    """Auction spacing in days: median of the last eight gaps, rounded to whole weeks."""
    d = sorted(set(pd.to_datetime(pd.Series(dates))))
    gaps = pd.Series(d).diff().dt.days.dropna().tail(8)
    return int(max(1, round(float(gaps.median()) / 7)) * 7) if len(gaps) else 7


def season_breaks(dates, cadence: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Summer and year-end auction breaks as (last auction before, first after) month-days, exclusive,
    from the latest year in which each break was observed (longest gap > 1.5 × cadence)."""
    d = sorted(set(pd.to_datetime(pd.Series(dates))))
    pairs = [(a, b) for a, b in zip(d, d[1:]) if (b - a).days > 1.5 * cadence]
    out = []
    for months in ((6, 7, 8), (12,)):
        cand = [(a, b) for a, b in pairs if a.month in months]
        if not cand:
            continue
        yr = max(a.year for a, _ in cand)
        a, b = max((p for p in cand if p[0].year == yr), key=lambda p: (p[1] - p[0]).days)
        out.append((_md(a), _md(b)))
    return out


def _in_break(t: pd.Timestamp, brk: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    s, e = brk
    md = _md(t)
    return (s < md < e) if s < e else (md > s or md < e)


def auction_calendar(dates, until: pd.Timestamp) -> list[pd.Timestamp]:
    """Future auction dates: the observed cadence and weekday continued from the latest auction, skipping
    the summer and year-end breaks seen in the history and Swedish fixed-date public holidays."""
    d = sorted(set(pd.to_datetime(pd.Series(dates))))
    if not d:
        return []
    cad = cadence_days(d)
    brks = season_breaks(d, cad)
    out, t = [], d[-1] + pd.Timedelta(days=cad)
    while t <= until:
        if _md(t) not in SE_FIXED_HOLIDAYS and not any(_in_break(t, b) for b in brks):
            out.append(t)
        t += pd.Timedelta(days=cad)
    return out


def timing_adjustment(origin: pd.Timestamp, year: int, items=TIMING_ITEMS,
                      report_last_outcome: pd.Timestamp | None = None) -> tuple[float, list[dict]]:
    """Balance-terms sum of timing items planned after the report's last outcome and up to the origin, still
    expected inside ``year`` and not received by the origin."""
    since = RG_REPORT["last_outcome"] if report_last_outcome is None else report_last_outcome
    live = []
    for it in items:
        pm = pd.Timestamp(f"{it['planned_month']}-01")
        rec = it.get("received_month")
        if not (since < pm <= origin) or int(it["expected_year"]) != year:
            continue
        if rec is not None and pd.Timestamp(f"{rec}-01") <= origin:
            continue
        live.append(it)
    return float(sum(float(i["amount_bn"]) for i in live)), live


def _month_end(m: pd.Timestamp) -> pd.Timestamp:
    return m + MonthEnd(0)


def load() -> dict:
    nbr = pd.read_csv(RAW / "nbr_monthly.csv", parse_dates=["month"])
    debt = pd.read_csv(RAW / "debt_change_monthly.csv", parse_dates=["month"])
    auc = pd.read_csv(RAW / "auctions.csv", parse_dates=["date"])
    out = pd.read_csv(RAW / "outstanding_securities.csv")
    log = json.loads((RAW / "fetch_log.json").read_text()) if (RAW / "fetch_log.json").exists() else {}
    return {"nbr": nbr, "debt": debt, "auc": auc, "out": out, "rg": fetch.riksgalden_forecasts(), "log": log}


def auction_class(series: pd.Series) -> pd.Series:
    s = series.str.strip()
    return pd.Series(np.select([s.str.startswith("STB"), s.str.startswith("SGB IL")], ["bill", "linker"], "nominal"),
                     index=series.index)


def auction_gross(auc: pd.DataFrame) -> pd.DataFrame:
    """Allocations by settlement month (bn): columns bill / linker / nominal (outright auctions, type O)."""
    a = auc[auc["type"] == "O"].copy()
    a["month"] = _settle_month(a["date"])
    a["cls"] = auction_class(a["series"])
    return a.pivot_table(index="month", columns="cls", values="allocated_mn", aggfunc="sum").fillna(0.0) / 1000.0


def stb_maturity(series: str) -> pd.Timestamp:
    m = re.search(r"STB\s+(\d{1,2})\s+([A-Za-zåäö]{3})\w*\s+(\d{2})", series)
    if not m:
        return pd.NaT
    return pd.Timestamp(2000 + int(m.group(3)), _MON[m.group(2).lower()], int(m.group(1)))


def bill_ladder(auc: pd.DataFrame) -> pd.DataFrame:
    """One row per T-bill auction: settlement date (T+2), maturity, allocation (bn)."""
    a = auc[(auc["type"] == "O") & auc["series"].str.strip().str.startswith("STB")].copy()
    a["settle"] = a["date"] + BDay(2)
    a["maturity"] = a["series"].map(stb_maturity)
    a["amount_bn"] = a["allocated_mn"] / 1000.0
    return a[["date", "series", "settle", "maturity", "amount_bn"]].dropna().reset_index(drop=True)


def snapshot_date(d: dict) -> pd.Timestamp:
    """Date of the outstanding-securities snapshot: the fetch date, never earlier than its latest issue date."""
    o = d["out"]
    latest_issue = pd.to_datetime(o["Issue date"], errors="coerce").max()
    fetched = pd.Timestamp(d.get("log", {}).get("fetched", str(latest_issue))[:10])
    return max(fetched, latest_issue)


def bill_stock_at(t: pd.Timestamp, snap_stock: float, snap: pd.Timestamp, lad: pd.DataFrame,
                  on: str = "settle") -> float:
    """T-bill stock at the close of ``t`` rebuilt from the snapshot: remove bills settled after t and add back
    bills that matured after t (both up to the snapshot date). ``on="date"`` counts a bill from its auction
    (trade) date, the basis of Riksgälden's CH1044 net borrowing and year-end stock outcomes."""
    issued = lad.loc[(lad[on] > t) & (lad["settle"] <= snap), "amount_bn"].sum()
    matured = lad.loc[(lad["maturity"] > t) & (lad["maturity"] <= snap), "amount_bn"].sum()
    return float(snap_stock - issued + matured)


def history(d: dict) -> pd.DataFrame:
    nbr = d["nbr"].set_index("month")
    debt = d["debt"].set_index("month")
    g = auction_gross(d["auc"])
    idx = debt.index
    h = pd.DataFrame(index=idx)
    b9 = 1e9
    h["cash_requirement_bn"] = nbr["nbr"].reindex(idx) / 1000.0
    h["budget_balance_bn"] = nbr["budget_balance"].reindex(idx) / 1000.0
    h["nbr_primary_bn"] = nbr["nbr_primary"].reindex(idx) / 1000.0
    h["nbr_net_lending_bn"] = nbr["nbr_net_lending"].reindex(idx) / 1000.0
    h["nbr_interest_bn"] = nbr["nbr_interest"].reindex(idx) / 1000.0
    nb = debt.fillna(0.0) / b9
    h["net_bonds_nominal_bn"] = nb["nb_nominal_bonds"] + nb["nb_green"]
    h["net_bonds_linker_bn"] = nb["nb_linkers"]
    h["net_bonds_fx_bn"] = nb["nb_fx_bonds"]
    h["net_bonds_bn"] = h["net_bonds_nominal_bn"] + h["net_bonds_linker_bn"] + h["net_bonds_fx_bn"]
    h["net_bills_bn"] = nb["nb_bills"]
    h["buffer_change_bn"] = -nb["nb_liquidity_mgmt"]
    # From the published total, not the item sum: in 5 months the items miss unitemized lines (2002-07: 41.6bn).
    h["other_fin_bn"] = nb["net_borrowing"] - h["net_bonds_bn"] - h["net_bills_bn"] + h["buffer_change_bn"] - nb["bday_adj"]
    h["unitemized_bn"] = nb["net_borrowing"] - nb[[c for c in nb.columns if c.startswith("nb_")]].sum(axis=1)
    gc = lambda c: g[c].reindex(idx).fillna(0.0) if c in g else pd.Series(0.0, index=idx)  # noqa: E731
    h["bond_gross_nominal_bn"] = gc("nominal")
    h["bond_gross_linker_bn"] = gc("linker")
    h["bond_gross_fx_bn"] = nb["nb_fx_bonds"].clip(lower=0)
    h["bond_gross_bn"] = h["bond_gross_nominal_bn"] + h["bond_gross_linker_bn"] + h["bond_gross_fx_bn"]
    h["bond_redemption_bn"] = h["bond_gross_bn"] - h["net_bonds_bn"]
    h["bill_gross_bn"] = gc("bill")
    h["bill_redemption_bn"] = h["bill_gross_bn"] - h["net_bills_bn"]
    h["debt_bn"] = debt["debt"] / b9
    h["fx_valuation_bn"] = nb["fx_valuation"]
    h["inflation_comp_bn"] = nb["inflation_comp"]
    h["mm_assets_change_bn"] = nb["mm_assets"]
    h["identity_resid_bn"] = h["cash_requirement_bn"] - debt["nbr_debt_table"] / b9
    h["is_forecast"] = False
    h["securities_basis"] = "outcome"
    return h


def _outstanding(d: dict) -> pd.DataFrame:
    o = d["out"].copy()
    o["maturity"] = pd.to_datetime(o["Maturity date"])
    o["amount_bn"] = o["Current exchange rate incl. inflation compensation"].map(fetch.num) / 1e9
    o["nominal_bn"] = o["Nominal amount"].map(fetch.num) / 1e9
    o["mkt"] = o["Security/ Market"].str.strip()
    o["month"] = o["maturity"].dt.to_period("M").dt.to_timestamp()
    return o


def anchor_to_riksgalden(raw: pd.Series, y: pd.Series, origin: pd.Timestamp, method: str = "uniform",
                         carry: float = 0.0, annual: dict | None = None) -> pd.Series:
    """Rescale the model's months so each calendar year sums to Riksgälden's annual forecast.

    ``carry`` (current year only, balance terms, + = better) is Riksgälden's cumulative deviation from its own
    monthly path since the report: the year-to-date beat is kept, not reversed in the remaining months.
    ``method`` is the PRE-selected allocation (``model.allocate``). Years without a published total stay raw.
    """
    out = raw.copy()
    for yr, total in (annual or RG_ANNUAL_BALANCE).items():
        m = out.index.year == yr
        if not m.any():
            continue
        done = float(y[(y.index.year == yr) & (y.index <= origin)].sum())
        n_year = 12 - (origin.month if yr == origin.year else 0)
        if m.sum() < n_year:
            continue
        target = total + (carry if yr == origin.year else 0.0) - done
        out[m] = model.allocate(out[m], target, method, y, origin).to_numpy()
    return out


def tenor_profile(lad: pd.DataFrame, snap: pd.Timestamp, months_back: int = 24) -> pd.Series:
    """Share of T-bill issuance by months from settlement month to maturity month (last 24 months)."""
    r = lad[(lad["settle"] <= snap) & (lad["settle"] > snap - pd.DateOffset(months=months_back))]
    k = ((r["maturity"].dt.year - r["settle"].dt.year) * 12 + r["maturity"].dt.month - r["settle"].dt.month).clip(lower=1)
    p = r.groupby(k)["amount_bn"].sum()
    return p / p.sum()


def nominal_forward(auc: pd.DataFrame, g: pd.DataFrame, months: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Nominal SGB gross by settlement month: settled and announced auctions from the auction table, then the
    future auction calendar with an equal size per auction that lands each year exactly on Riksgälden's plan."""
    a = auc[auc["type"] == "O"]
    dates = a.loc[auction_class(a["series"]) == "nominal", "date"]
    cal = auction_calendar(dates, months[-1] + MonthEnd(0))
    n_cal = pd.Series([_month_of(t + BDay(2)) for t in cal], dtype="datetime64[ns]").value_counts()
    known = g["nominal"] if "nominal" in g else pd.Series(dtype=float)
    out = pd.Series(0.0, index=months)
    sizes = {}
    for yr in sorted(set(months.year)):
        plan = RG_PLAN["nominal_bonds"].get(yr)
        done = float(known[known.index.year == yr].sum())
        n = int(n_cal[n_cal.index.year == yr].sum())
        size = max(plan - done, 0.0) / n if (plan is not None and n) else 0.0
        sizes[yr] = {"plan_bn": plan, "settled_or_announced_bn": round(done, 2), "auctions_left": n,
                     "size_per_auction_bn": round(size, 3)}
        for m in months[months.year == yr]:
            out[m] = float(known.get(m, 0.0)) + size * int(n_cal.get(m, 0))
    meta = {"cadence_days": cadence_days(dates), "breaks": season_breaks(dates, cadence_days(dates)),
            "next_dates": [f"{t:%Y-%m-%d}" for t in cal[:6]], "by_year": sizes}
    return out, meta


def linker_forward(g: pd.DataFrame, months: pd.DatetimeIndex) -> pd.Series:
    """Linker gross: each year's plan less settled, spread over the calendar months that had linker auctions in
    the twelve months to the latest one."""
    lin = g["linker"] if "linker" in g else pd.Series(dtype=float)
    out = pd.Series(0.0, index=months)
    if lin.empty or not (lin > 0).any():
        return out
    last = lin[lin > 0].index.max()
    trail = lin[(lin.index > last - pd.DateOffset(months=12)) & (lin.index <= last)]
    active = set(trail[trail > 0].index.month)
    for yr in sorted(set(months.year)):
        fut = [m for m in months[months.year == yr] if m > last and m.month in active]
        left = max(RG_PLAN["linkers"].get(yr, 0.0) - float(lin[lin.index.year == yr].sum()), 0.0)
        for m in months[months.year == yr]:
            out[m] = float(lin.get(m, 0.0)) if m <= last else (left / len(fut) if m in fut else 0.0)
    return out


def fx_bond_month(debt: pd.DataFrame) -> int | None:
    """Calendar month of the latest FX-bond issue (> 1bn net borrowing), assumed to repeat."""
    nb = debt.set_index("month")["nb_fx_bonds"].fillna(0.0) / 1e9
    hits = nb[nb > 1.0]
    return int(hits.index.max().month) if len(hits) else None


def third_wednesday(m: pd.Timestamp) -> pd.Timestamp:
    first = pd.Timestamp(m.year, m.month, 1)
    return first + pd.Timedelta(days=(2 - first.weekday()) % 7 + 14)


def bill_forward(lad: pd.DataFrame, alive: pd.DataFrame, snap: pd.Timestamp, s0: float, last_known: pd.Timestamp,
                 months: pd.DatetimeIndex, tenor: pd.Series, plan_end: dict) -> tuple[pd.DataFrame, dict]:
    """T-bills day by day from the auction calendar and standard maturities (third Wednesday).

    Known: bills alive at the snapshot (exact maturity dates), bills settled after ``last_known`` and their
    maturities. Future auctions continue the observed cadence and weekday. Each auction rolls what matured since
    the previous roll (the last auction of a year also pre-funds maturities to 31 Dec; maturities inside an
    auction break roll at the next auction) plus an even share of the gap to
    Riksgälden's planned year-end stock over the year's remaining auctions, so the stock glides to plan and
    gross follows the maturity calendar. New issues mature k months after settlement on the tenor mix of the
    last 24 months. A year with no plan holds the stock (rollover only)."""
    lk_end = _month_end(last_known)
    end = months[-1] + MonthEnd(0)
    mats: defaultdict = defaultdict(float)
    for t, v in zip(pd.to_datetime(alive["Maturity date"]), alive["nominal_bn"]):
        mats[pd.Timestamp(t).normalize()] += float(v)
    for _, r in lad[(lad["maturity"] > lk_end) & (lad["maturity"] <= snap)].iterrows():
        mats[r["maturity"].normalize()] += r["amount_bn"]
    for _, r in lad[lad["settle"] > snap].iterrows():
        mats[r["maturity"].normalize()] += r["amount_bn"]
    known: defaultdict = defaultdict(float)
    for _, r in lad[lad["settle"] > lk_end].iterrows():
        known[r["settle"].normalize()] += r["amount_bn"]
    cal = auction_calendar(lad["date"], end)
    settles = [(t + BDay(2)).normalize() for t in cal]
    gross, red, stock, sizes = defaultdict(float), defaultdict(float), {}, defaultdict(list)
    s = s0
    covered = max(known) if known else lk_end  # maturities after this date are not yet rolled
    for day in pd.date_range(lk_end + pd.Timedelta(days=1), end, freq="D"):
        m = _month_of(day)
        if mats.get(day):
            s -= mats[day]
            red[m] += mats[day]
        if known.get(day):
            s += known[day]
            gross[m] += known[day]
        if day in settles:
            i = settles.index(day)
            left = [x for x in settles[i:] if x.year == day.year]
            upto = day if len(left) > 1 else pd.Timestamp(day.year, 12, 31)
            back = sum(v for d_, v in mats.items() if covered < d_ <= day)  # matured since the last roll
            ahead = sum(v for d_, v in mats.items() if max(covered, day) < d_ <= upto)
            covered = upto
            target = plan_end.get(day.year)
            # glide the pre-maturity stock (s + back) toward the plan
            size = max(back + ahead + ((target - s - back) / len(left) if target is not None else 0.0), 0.0)
            s += size
            gross[m] += size
            sizes[day.year].append(size)
            for kk, pk in tenor.items():
                mats[third_wednesday(m + pd.DateOffset(months=int(kk)))] += size * float(pk)
        if day == day + MonthEnd(0):
            stock[m] = s
    fut = [m for m in months if m > last_known]
    out = pd.DataFrame({"bill_gross_bn": [gross[m] for m in fut], "bill_redemption_bn": [red[m] for m in fut],
                        "bill_stock_path_bn": [stock[m] for m in fut]}, index=fut)
    out["net_bills_bn"] = out["bill_gross_bn"] - out["bill_redemption_bn"]
    meta = {"cadence_days": cadence_days(lad["date"]), "next_dates": [f"{t:%Y-%m-%d}" for t in cal[:6]],
            "mean_size_bn": {int(k): round(float(np.mean(v)), 2) for k, v in sizes.items()}}
    return out, meta


def _plan_through(d: dict[int, float], years) -> dict[int, float]:
    """Carry the last published plan year forward so post-plan months are not zero-issuance."""
    out = {int(k): float(v) for k, v in d.items()}
    if not out:
        return out
    last = max(out)
    for y in years:
        y = int(y)
        if y not in out:
            out[y] = out[last]
    return out


def forward(d: dict, y: pd.Series, chosen: str, method: str, carry: float) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.cash_season import HORIZON_END, forward_months as _fwd_m

    k, a, f = model.choose_params(chosen)
    origin = y.index[-1]
    h_live = _fwd_m(origin)
    raw = model.forecast(y, origin, k, a, f, hmax=h_live)
    if len(raw) and raw.index.max() < HORIZON_END:
        # Defensive: fill any gap to the common cutoff with the same seasonal rule.
        raw = model.forecast(y, origin, k, a, f, hmax=max(h_live, _fwd_m(origin)))
    anchored = anchor_to_riksgalden(raw, y, origin, method, carry)
    months_all = raw.index
    snap = snapshot_date(d)
    o = _outstanding(d)
    lad = bill_ladder(d["auc"])
    g = auction_gross(d["auc"])
    observed = [m for m in months_all if _month_end(m) < snap]  # auction table complete for the month
    last_known = max(observed) if observed else origin
    years = sorted(set(months_all.year))
    plan_nom = _plan_through(RG_PLAN["nominal_bonds"], years)
    plan_lin = _plan_through(RG_PLAN["linkers"], years)
    plan_fx = _plan_through(RG_PLAN["fx_bonds"], years)
    plan_bill = _plan_through(RG_PLAN["bill_stock_end"], years)

    fw = pd.DataFrame(index=months_all)
    fw["budget_balance_bn"] = anchored.loc[months_all]
    fw["budget_balance_model_bn"] = raw.loc[months_all]
    fw["cash_requirement_bn"] = -fw["budget_balance_bn"]
    fw["securities_basis"] = ["auction_table" if m in observed else "calendar" for m in months_all]

    # Temporarily swap plan maps so calendar helpers carry post-plan years.
    _saved = {k: dict(RG_PLAN[k]) for k in ("nominal_bonds", "linkers", "fx_bonds", "bill_stock_end")}
    RG_PLAN["nominal_bonds"], RG_PLAN["linkers"] = plan_nom, plan_lin
    RG_PLAN["fx_bonds"], RG_PLAN["bill_stock_end"] = plan_fx, plan_bill
    try:
        nom, nom_meta = nominal_forward(d["auc"], g, months_all)
        fxm = fx_bond_month(d["debt"])
        nb_fx = d["debt"].set_index("month")["nb_fx_bonds"].fillna(0.0).clip(lower=0) / 1e9
        gross_fx = []
        for m in months_all:
            left = max(plan_fx.get(m.year, 0.0) - float(nb_fx[nb_fx.index.year == m.year].sum()), 0.0)
            gross_fx.append(left if (fxm is not None and m.month == fxm and m > origin and left > 1.0) else 0.0)
        fw["bond_gross_nominal_bn"] = nom
        fw["bond_gross_linker_bn"] = linker_forward(g, months_all)
        fw["bond_gross_fx_bn"] = gross_fx
        by = lambda mk, col: o[o["mkt"].isin(mk)].groupby("month")[col].sum().reindex(months_all).fillna(0.0)  # noqa: E731
        red_nom = by(["SGB", "GO"], "nominal_bn")
        red_lin_cash, red_lin_nom = by(["SGB IL"], "amount_bn"), by(["SGB IL"], "nominal_bn")
        red_fx_cash, red_fx_nom = by(["PB FC"], "amount_bn"), by(["PB FC"], "nominal_bn")
        fw["bond_redemption_nominal_bn"] = red_nom
        fw["bond_redemption_linker_bn"] = red_lin_cash
        fw["bond_redemption_fx_bn"] = red_fx_cash
        fw["net_bonds_nominal_bn"] = fw["bond_gross_nominal_bn"] - red_nom
        fw["net_bonds_linker_bn"] = fw["bond_gross_linker_bn"] - red_lin_cash
        fw["net_bonds_fx_bn"] = fw["bond_gross_fx_bn"] - red_fx_cash
        fw["bond_gross_bn"] = fw[["bond_gross_nominal_bn", "bond_gross_linker_bn", "bond_gross_fx_bn"]].sum(axis=1)
        fw["bond_redemption_bn"] = red_nom + red_lin_cash + red_fx_cash
        fw["net_bonds_bn"] = fw["bond_gross_bn"] - fw["bond_redemption_bn"]
        # Linker inflation compensation and FX-bond currency differences paid at redemption are interest in the NBR,
        # so the cash redemption above the nominal amount is already inside cash_requirement: offset it here.
        fw["redemption_uplift_in_nbr_bn"] = (red_lin_cash - red_lin_nom) + (red_fx_cash - red_fx_nom)
        fw["other_fin_bn"] = fw["redemption_uplift_in_nbr_bn"]

        snap_stock = float(o.loc[o["mkt"] == "STB", "nominal_bn"].sum())
        stock = lambda t: bill_stock_at(t, snap_stock, snap, lad)  # noqa: E731
        s_origin = stock(_month_end(origin))
        gross_b, red_b, net_b, stock_end = {}, {}, {}, {}
        prev = s_origin
        for m in observed:
            s = stock(_month_end(m))
            gross_b[m] = float(lad.loc[lad["settle"].dt.to_period("M").dt.to_timestamp().eq(m), "amount_bn"].sum())
            net_b[m] = s - prev
            red_b[m] = gross_b[m] - net_b[m]
            stock_end[m], prev = s, s
        s0 = prev
        p = tenor_profile(lad, snap)
        bills, bill_meta = bill_forward(lad, o[o["mkt"] == "STB"], snap, s0, last_known, months_all, p,
                                        plan_bill)
        for col in ("bill_gross_bn", "bill_redemption_bn", "net_bills_bn", "bill_stock_path_bn"):
            obs = {"bill_gross_bn": gross_b, "bill_redemption_bn": red_b, "net_bills_bn": net_b,
                   "bill_stock_path_bn": stock_end}[col]
            fw[col] = pd.Series(obs, dtype=float).combine_first(bills[col]) if len(bills) else pd.Series(obs, dtype=float)
    finally:
        for k, v in _saved.items():
            RG_PLAN[k] = v
    # All-bills alternative: liquidity management flat, every krona of need in T-bills.
    fw["net_bills_if_liq_flat_bn"] = fw["cash_requirement_bn"] - fw["net_bonds_bn"] - fw["other_fin_bn"]
    fw["buffer_change_bn"] = -fw["cash_requirement_bn"] + fw["net_bonds_bn"] + fw["net_bills_bn"] + fw["other_fin_bn"]
    fw["is_forecast"] = True
    meta = {"snapshot_date": f"{snap:%Y-%m-%d}", "snapshot_bill_stock_bn": round(snap_stock, 1),
            "bill_stock_end_last_outcome_bn": round(s_origin, 1),
            "bill_stock_end_last_outcome_trade_bn": round(
                bill_stock_at(_month_end(origin), snap_stock, snap, lad, on="date"), 1),
            "bill_stock_end_last_auction_month_bn": round(s0, 1),
            "auction_observed_months": [f"{m:%Y-%m}" for m in observed],
            "tenor_profile_months": {int(kk): round(float(v), 3) for kk, v in p.items()},
            "bill_calendar": bill_meta, "nominal_calendar": nom_meta, "fx_bond_month": fxm,
            "horizon_end": f"{months_all.max():%Y-%m}"}
    return fw, meta


def score_shipped(y: pd.Series, chosen: str, method: str) -> list[dict]:
    """The shipped forward (Riksgälden annual × model shape) at each archived annual vintage, against the
    unanchored model and same month last year on the same months. No carry: at a report date the deviation
    from the new path is zero."""
    k, a, f = model.choose_params(chosen)
    out, pool = [], []
    for name, o, annual in RG_ANNUAL_VINTAGES:
        if o not in y.index:
            continue
        hmax = (max(annual) - o.year) * 12 + 12 - o.month
        raw = model.forecast(y, o, k, a, f, hmax=hmax)
        sh = anchor_to_riksgalden(raw, y, o, method, 0.0, annual)
        t = [m for m in raw.index if m in y.index]
        if not t:
            continue
        e = pd.DataFrame({"shipped": (sh[t] - y[t]).abs(), "model": (raw[t] - y[t]).abs(),
                          "smly": (y.shift(12).reindex(t) - y[t]).abs()})
        pool.append(e)
        out.append({"vintage": name, "origin": f"{o:%Y-%m}", "targets": f"{t[0]:%Y-%m}..{t[-1]:%Y-%m}", "n": len(t),
                    **{c: round(float(e[c].mean()), 2) for c in e}})
    if pool:
        e = pd.concat(pool)
        out.append({"vintage": "pooled", "origin": "", "targets": "", "n": len(e),
                    **{c: round(float(e[c].mean()), 2) for c in e}})
    return out


def build(write: bool = True) -> tuple[pd.DataFrame, dict]:
    d = load()
    hist = history(d)
    y = d["nbr"].set_index("month")["budget_balance"] / 1000.0
    bt = model.backtest(y)
    chosen, sel = model.select(bt)
    score = model.score_table(bt, chosen, d["rg"])
    rgm = d["rg"].set_index("month")["rg_forecast_bn"].dropna()
    abt = model.rg_anchor_backtest(y, rgm, chosen)
    method, alloc_tab = model.select_allocation(abt)
    abt_rg = abt.assign(win=model.window(abt["target"]))
    abt_rg = abt_rg[abt_rg["win"] != ""].groupby(["method", "win"])["ae_rg"].mean().unstack()
    obt = model.allocation_backtest(y, chosen)
    _, oracle_tab = model.select_allocation(obt)

    rg = d["rg"]
    last_hist = hist.index.max()
    yr_now = last_hist.year
    cur = rg[(rg["month"] == last_hist) & rg["rg_cum_dev_bn"].notna()]
    carry = float(cur["rg_cum_dev_bn"].iloc[0]) if len(cur) else 0.0
    carry_pub = f"{cur['published'].iloc[0]:%Y-%m-%d}" if len(cur) else None
    timing, timing_live = timing_adjustment(last_hist, yr_now) if len(cur) else (0.0, [])
    carry_ex = carry + timing

    fw, meta = forward(d, y, chosen, method, carry_ex)
    df = pd.concat([hist, fw]).sort_index()
    df.index.name = "month"
    df = df.reset_index()
    df["cash_requirement_fc_official_bn"] = -df["month"].map(rg.set_index("month")["rg_forecast_bn"])
    hist_mask = ~df["is_forecast"]
    nb = df["net_bills_bn"].fillna(0.0)
    # Bill stock: cumulated net bills anchored at the close of the last outcome month. CH1044 history books bills
    # on trade date and the forward path on settlement, so each side takes the anchor on its own basis.
    cum = nb.cumsum() - nb.cumsum()[df["month"] == last_hist].iloc[0]
    anchor = np.where(hist_mask, meta["bill_stock_end_last_outcome_trade_bn"], meta["bill_stock_end_last_outcome_bn"])
    df["bill_stock_implied_bn"] = anchor + cum
    df["buffer_level_implied_bn"] = df["buffer_change_bn"].fillna(0.0).cumsum()
    # Liquidity-management stock (borrowing +): Riksgälden's latest year-end outcome plus cumulated −Δbuffer.
    lm_year = max(RG_STOCK_OUTCOME["liq_mgmt_stock_end"])
    lm = (-df["buffer_change_bn"].fillna(0.0)).cumsum()
    lm_anchor_m = pd.Timestamp(lm_year, 12, 1)
    df["liq_mgmt_stock_implied_bn"] = (RG_STOCK_OUTCOME["liq_mgmt_stock_end"][lm_year]
                                       + (lm - lm[df["month"] == lm_anchor_m].iloc[0]))
    df["is_forecast"] = df["is_forecast"].astype(bool)

    from sovereign_cash.se import bottom_up as BU
    from sovereign_cash.se import bottom_up_macro as BUM
    df, bu_report = BU.run_report(d, y, chosen, method, carry_ex, df, write=write)
    df, bu_macro = BUM.run_report(d, y, chosen, method, df, write=write)

    resid = contract.validate(df[hist_mask].reset_index(drop=True))
    contract.validate(df.reset_index(drop=True))

    at = lambda col, m: float(df.loc[df["month"] == m, col].iloc[0])  # noqa: E731
    stock_check = {f"end_{yr}": {"implied_bn": round(at("bill_stock_implied_bn", pd.Timestamp(yr, 12, 1)), 1), "riksgalden_bn": v}
                   for yr, v in RG_STOCK_OUTCOME["bill_stock_end"].items()}
    lm_check = {f"end_{yr}": {"implied_bn": round(at("liq_mgmt_stock_implied_bn", pd.Timestamp(yr, 12, 1)), 1), "riksgalden_bn": v}
                for yr, v in RG_STOCK_OUTCOME["liq_mgmt_stock_end"].items()}

    fwd_all = df[df["is_forecast"]].sort_values("month")
    f12 = fwd_all.head(12)  # KPI / investor blocks stay on the next twelve months
    roy = fwd_all[fwd_all["month"].dt.year == yr_now]
    ytd = float(df.loc[hist_mask & (df["month"].dt.year == yr_now), "budget_balance_bn"].sum())
    plan_cr = -RG_ANNUAL_BALANCE[yr_now]
    rg_rest = RG_ANNUAL_BALANCE[yr_now] + carry_ex - ytd
    end_m = pd.Timestamp(yr_now, 12, 1)
    s = lambda frame, col: round(float(frame[col].sum()), 1)  # noqa: E731
    ly = df[(df["month"].dt.year == yr_now - 1) & df["month"].dt.month.isin(sorted(set(roy["month"].dt.month)))]

    def block(frame: pd.DataFrame) -> dict:
        return {"window": f"{frame['month'].min():%Y-%m}..{frame['month'].max():%Y-%m}",
                "cash_requirement_bn": s(frame, "cash_requirement_bn"),
                "budget_balance_bn": s(frame, "budget_balance_bn"),
                "budget_balance_model_unanchored_bn": s(frame, "budget_balance_model_bn"),
                "bond_gross_bn": s(frame, "bond_gross_bn"), "bond_redemption_bn": s(frame, "bond_redemption_bn"),
                "net_bonds_bn": s(frame, "net_bonds_bn"),
                "bond_gross_nominal_bn": s(frame, "bond_gross_nominal_bn"),
                "bond_redemption_nominal_bn": s(frame, "bond_redemption_nominal_bn"),
                "net_bonds_nominal_bn": s(frame, "net_bonds_nominal_bn"),
                "net_bonds_linker_bn": s(frame, "net_bonds_linker_bn"),
                "net_bonds_fx_bn": s(frame, "net_bonds_fx_bn"),
                "bill_gross_bn": s(frame, "bill_gross_bn"), "net_bills_bn": s(frame, "net_bills_bn"),
                "other_fin_bn": s(frame, "other_fin_bn"),
                "liquidity_mgmt_need_bn": round(float(-frame["buffer_change_bn"].sum()), 1)}

    oos12 = next(r for r in score if r["window"] == "OOS" and r["h"] == 12)
    ship = score_shipped(y, chosen, method)
    nd = fetch.next_dates()
    next_report = nd.get("report", {}).get("date", NEXT_REPORT_FALLBACK)
    next_outcome = nd.get("outcome")
    gdp = _gdp_bn()
    pure_model_fy = round(-ytd - float(roy["budget_balance_model_bn"].sum()), 1)
    plan_consistent_fy = round(plan_cr - carry_ex, 1)
    lm_end = at("liq_mgmt_stock_implied_bn", end_m)
    sym = "SEK"
    skill = skill_verdict(score)
    summary = {
        "country": "se", "ccy": "SEK", "asof": last_hist.strftime("%Y-%m"),
        "sources": SOURCES,
        "buffer": ("Riksgälden net liquidity-management position (repos, deposits, on-tap bills, FX commercial "
                   "paper); Δbuffer = − net borrowing in liquidity-management instruments. The level is "
                   f"Riksgälden's end-{lm_year} stock ({fmt_bn(RG_STOCK_OUTCOME['liq_mgmt_stock_end'][lm_year], sym, 0)}) "
                   "plus cumulated flows."),
        "identity_resid_bn": round(float(resid), 4),
        "score": score,
        "skill_verdict": skill,
        "score_note": (f"Walk-forward monthly origins, target months PRE {model.PRE[0]:%Y-%m}..{model.PRE[1]:%Y-%m} / "
                       f"OOS {model.OOS_START:%Y-%m}→. Model chosen on PRE: {model.describe(chosen)}. Riksgälden's "
                       "forecast is the one in its latest report before each month (1–6 months ahead), so it has no "
                       "single horizon. The model only supplies the month shape; the level is Riksgälden's."),
        "oos_h12": {"model": oos12["model"], "same_month_last_year": oos12["benchmark"],
                    "model_beats_smly": bool(oos12["model"] < oos12["benchmark"])},
        "shipped_forward_score": ship,
        "shipped_forward_score_note": ("Shipped forward = Riksgälden annual total with the model's month shape, scored "
                                       "from each archived report on every month with an outcome since. All OOS; "
                                       "n is small."),
        "allocation": {
            "chosen": method,
            "selection": ("Lowest PRE MAE vs outcomes, rest of year anchored to the sum of Riksgälden's published "
                          "monthly forecasts (origins Jun–Sep)."),
            "pre_oos_vs_outcome": {m: {k: round(float(v), 2) for k, v in r.items()} for m, r in alloc_tab.iterrows()},
            "pre_oos_vs_riksgalden_path": {m: {k: round(float(v), 2) for k, v in r.items()} for m, r in abt_rg.iterrows()},
            "pre_oos_realised_annual_anchor": {m: {k: round(float(v), 2) for k, v in r.items()} for m, r in oracle_tab.iterrows()},
        },
        "selection_table": {m: {k: round(float(v), 2) for k, v in r.items()} for m, r in sel.head(8).iterrows()},
        "selection_labels": {m: model.describe(m) for m in sel.head(8).index},
        "bottom_up": bu_report,
        "bottom_up_macro": bu_macro,
        "bill_stock_check": stock_check,
        "liq_mgmt_stock_check": lm_check,
        "forward": {
            "plan_year": yr_now,
            "plan_cash_requirement_bn": plan_cr,
            "plan_source": f"Riksgälden {RG_VINTAGE}: net borrowing requirement {fmt_bn(plan_cr, sym, 0)}",
            "plan_report_date": f"{RG_REPORT['published']:%Y-%m-%d}",
            "cash_requirement_ytd_bn": round(-ytd, 1),
            "ytd_through": last_hist.strftime("%Y-%m"),
            "riksgalden_ytd_deviation_bn": round(carry, 1),
            "riksgalden_ytd_deviation_asof": carry_pub,
            "borrowing_vs_riksgalden_path_ytd_bn": round(-carry, 1),
            "timing_items_bn": round(timing, 1),
            "timing_items": [{k: v for k, v in i.items()} for i in timing_live],
            "borrowing_vs_riksgalden_path_ex_timing_bn": round(-carry_ex, 1),
            "model_full_year_bn": plan_consistent_fy,
            "plan_consistent_full_year_bn": plan_consistent_fy,
            "plan_consistent_basis": (f"Riksgälden {RG_REPORT['label']} full year less its YTD deviation from its own "
                                      "monthly path, excluding announced timing items"),
            "pure_model_full_year_bn": pure_model_fy,
            "rest_of_year_implied_bn": round(-rg_rest, 1),
            "rest_of_year_same_months_last_year_bn": round(float(ly["cash_requirement_bn"].sum()), 1),
            "plan_gap_lands_on": (f"liquidity management (bonds and bills on plan): end-{yr_now} stock "
                                  f"{fmt_bn(lm_end, sym, 0)} vs plan "
                                  f"{fmt_bn(RG_PLAN['liq_mgmt_stock_end'][yr_now], sym, 0)}"),
            "next_report": f"{next_report:%Y-%m-%d}",
            "next_outcome": ({"month": f"{next_outcome['month']:%Y-%m}", "date": f"{next_outcome['date']:%Y-%m-%d}"}
                             if next_outcome else None),
            "forward_basis": (f"Riksgälden annual totals ({', '.join(map(str, RG_ANNUAL_BALANCE))}) with the YTD "
                              f"deviation ex timing carried in {yr_now}; month shape from the PRE-selected seasonal "
                              f"model, gap allocated '{method}' (PRE-selected). Post-plan years carry last plan "
                              f"issuance / bill-stock and the unanchored seasonal cash path through "
                              f"{fwd_all['month'].max():%Y-%m}."),
            f"rest_of_{yr_now}": block(roy),
            "next_12m": block(f12),
            "window": f"{f12['month'].min():%Y-%m}..{f12['month'].max():%Y-%m}",
            "horizon_end": f"{fwd_all['month'].max():%Y-%m}",
            "full_forward_window": f"{fwd_all['month'].min():%Y-%m}..{fwd_all['month'].max():%Y-%m}",
            "gdp_bn": gdp,
            "net_bonds_plus_bills_12m_pct_gdp": (round(100 * (s(f12, "net_bonds_bn") + s(f12, "net_bills_bn")) / gdp, 2)
                                                 if gdp else None),
            "budget_balance_12m_bn": s(f12, "budget_balance_bn"),
            "cash_requirement_12m_bn": s(f12, "cash_requirement_bn"),
            "net_bonds_12m_bn": s(f12, "net_bonds_bn"),
            "net_bonds_nominal_12m_bn": s(f12, "net_bonds_nominal_bn"),
            "net_bonds_linker_12m_bn": s(f12, "net_bonds_linker_bn"),
            "net_bonds_fx_12m_bn": s(f12, "net_bonds_fx_bn"),
            "net_bills_12m_bn": s(f12, "net_bills_bn"),
            "liquidity_mgmt_need_12m_bn": round(float(-f12["buffer_change_bn"].sum()), 1),
            "implied_net_bills_12m_bn_if_liq_mgmt_flat": s(f12, "net_bills_if_liq_flat_bn"),
            "bill_stock": {**meta, f"end_{yr_now}_path_bn": round(at("bill_stock_implied_bn", end_m), 1),
                           f"riksgalden_plan_end_{yr_now}_bn": RG_PLAN["bill_stock_end"][yr_now]},
            "liq_mgmt_stock": {f"end_{lm_year}_riksgalden_bn": RG_STOCK_OUTCOME["liq_mgmt_stock_end"][lm_year],
                               "last_outcome_month": f"{last_hist:%Y-%m}",
                               "last_outcome_implied_bn": round(at("liq_mgmt_stock_implied_bn", last_hist), 1),
                               f"end_{yr_now}_implied_bn": round(lm_end, 1),
                               f"riksgalden_plan_end_{yr_now}_bn": RG_PLAN["liq_mgmt_stock_end"][yr_now]},
            f"budget_balance_ytd_{yr_now}_bn": round(ytd, 1),
            f"unanchored_model_rest_of_{yr_now}_bn": round(float(roy["budget_balance_model_bn"].sum()), 1),
            f"riksgalden_implied_rest_of_{yr_now}_bn": round(rg_rest, 1),
            f"riksgalden_annual_{yr_now}_bn": RG_ANNUAL_BALANCE[yr_now],
            f"riksgalden_annual_{yr_now + 1}_bn": RG_ANNUAL_BALANCE.get(yr_now + 1),
            "riksgalden_vintage": RG_VINTAGE,
        },
        "notes": notes(meta, yr_now, method, stock_check, lm_check, lm_year, timing_live, carry, carry_pub, sym)
                 + [_bottom_up_note(bu_report), _bottom_up_macro_note(bu_macro)],
    }
    if write:
        DERIVED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_MONTHLY, index=False)
        bt.to_parquet(OUT_BACKTEST, index=False)
        d["rg"].to_parquet(OUT_RG, index=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return df, summary


def _gdp_bn() -> float | None:
    p = contract.ROOT / "data" / "scalers.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("gdp_bn", {}).get("se")
    return float(v) if v else None


def skill_verdict(score: list[dict]) -> str:
    """One generated sentence per window: model vs Riksgälden on the months Riksgälden published, and vs
    same month last year."""
    out = []
    for win in ("PRE", "OOS"):
        rows = [r for r in score if r["window"] == win]
        if not rows:
            continue
        rg_v = rows[0].get("riksgalden")
        own = [r["model_on_rg_months"] for r in rows if r.get("model_on_rg_months") is not None]
        parts = []
        if rg_v is not None and own:
            lose = sum(m > rg_v for m in own)
            word = ("loses to Riksgälden at every horizon" if lose == len(own) else
                    "beats Riksgälden at every horizon" if lose == 0 else f"loses to Riksgälden at {lose} of {len(own)} horizons")
            parts.append(f"{word} ({fmt_num(min(own), 1)}–{fmt_num(max(own), 1)} vs {fmt_num(rg_v, 1)})")
        beat = [r["h"] for r in rows if r["model"] < r["benchmark"]]
        miss = [r for r in rows if r["model"] >= r["benchmark"]]
        if miss:
            parts.append("does not beat same-month-last-year at " + ", ".join(
                f"h={r['h']} ({fmt_num(r['model'], 2)} vs {fmt_num(r['benchmark'], 2)})" for r in miss))
        elif beat:
            parts.append("beats same-month-last-year at every horizon")
        out.append(f"{win}: the model " + " and ".join(parts) + ".")
    return " ".join(out)


def notes(meta: dict, yr: int, method: str, stock_check: dict, lm_check: dict, lm_year: int, timing_live: list,
          carry: float, carry_pub: str | None, sym: str) -> list[str]:
    nc, bc = meta["nominal_calendar"], meta["bill_calendar"]
    sizes = ", ".join(f"{y}: {v['auctions_left']} × {fmt_bn(v['size_per_auction_bn'], sym, 2)}"
                      for y, v in nc["by_year"].items() if v["auctions_left"])
    chk = "; ".join(f"end-{k[-4:]} bills {fmt_num(stock_check[k]['implied_bn'], 1)} vs {fmt_num(stock_check[k]['riksgalden_bn'], 0)}, "
                    f"liquidity management {fmt_num(lm_check[k]['implied_bn'], 1)} vs {fmt_num(lm_check[k]['riksgalden_bn'], 0)}"
                    for k in stock_check if k in lm_check)
    brk = "; ".join(f"{pd.Timestamp(2000, *a):%d %b}–{pd.Timestamp(2000, *b):%d %b}" for a, b in nc["breaks"])
    tim = "; ".join(f"{i['item']} {fmt_bn(i['amount_bn'], sym, 0)}, planned {i['planned_month']}, now expected in "
                    f"{i['expected_year']} (release {i['announced']})" for i in timing_live)
    out = [
        "Sign convention: net borrowing requirement and all financing lines + = borrowing (deficit); "
        "the budget balance is the same number with the opposite sign.",
        "Financing is Riksgälden's net borrowing at nominal amount by instrument; FX and inflation revaluation "
        "are removed by the source. Redemptions = gross − net, so they also absorb switches and buybacks.",
        f"Nominal SGBs: settled and announced auctions, then a {nc['cadence_days']}-day auction calendar (breaks "
        f"{brk}, public holidays skipped) with one size per year that lands on Riksgälden's plan ({sizes}).",
        f"T-bills: an auction every {bc['cadence_days']} days on the observed weekday (same breaks rule). Each rolls "
        "what matured since the last one plus an even step toward Riksgälden's planned year-end stock; new issues "
        "mature on the third Wednesday with the tenor mix of the last 24 months. Liquidity management absorbs "
        "the rest.",
        "Linkers: the plan spread over the months that had linker auctions in the past year; one FX bond a year in "
        + (f"{pd.Timestamp(2000, meta['fx_bond_month'], 1):%B} (month of the latest FX issue)"
           if meta.get("fx_bond_month") else "no month (no FX issue on record)")
        + ". Redemptions from the outstanding list.",
        f"Stock checks against Riksgälden year-end outcomes: {chk}.",
    ]
    if timing_live:
        out.append(f"Timing items excluded from the YTD deviation ({fmt_bn(carry, sym, 1, plus=True)} balance terms, "
                   f"release {carry_pub}): {tim}.")
    return out


def _bottom_up_note(bu: dict) -> str:
    pre_a = bu.get("pre_mean_mae_anchored", {})
    pre_u = bu.get("pre_mean_mae_unanchored", {})
    return (f"Bottom-up CH1089 baseline: {'SHIP' if bu.get('ship') else 'NO-SHIP'} — PRE anchored mean MAE "
            f"{pre_a.get('bottom_up')} vs production {pre_a.get('production')} "
            f"({bu.get('ship_reason')}). Unanchored PRE {pre_u.get('bottom_up')} vs {pre_u.get('production')}. "
            "Alternate bottom-up cash-requirement column does not replace the live path when NO-SHIP.")


def _bottom_up_macro_note(bu: dict) -> str:
    pre_a = bu.get("pre_mean_mae_anchored", {})
    return (f"Bottom-up macro (Budgetprop/SCB×drivers) v2: {'SHIP' if bu.get('ship') else 'NO-SHIP'} — "
            f"PRE anchored mean MAE {pre_a.get('bottom_up')} vs production {pre_a.get('production')} "
            f"({bu.get('ship_reason')}).")

if __name__ == "__main__":
    df, s = build()
    print(json.dumps({k: s[k] for k in ("asof", "identity_resid_bn", "forward")}, indent=2, ensure_ascii=False))
    for r in s["score"]:
        print(r)
