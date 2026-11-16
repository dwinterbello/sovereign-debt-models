"""Norway contract outputs from Norges Bank bank-liquidity statistics.

Liquidity effects (LIQUIDITY_STATISTICS, NOK mn, accumulated from 1 January) are differenced into daily
flows. A positive liquidity effect is money paid *out of* the government's accounts at Norges Bank into
the banking system, so the government-account (buffer) inflow is the negative of the effect:

    Δbuffer        = −(gov_accounts + gov_securities + gpfg_fx + transfer_financing)
    cash_requirement = gov_accounts                       (net fiscal payments ex securities; incl. petroleum
                                                          taxes and SDFI, so negative in tax months)
    bond_gross     = −bond_issue         bond_redemption = bond_maturity
    bill_gross     = −bill_issue         bill_redemption = bill_maturity
    other_fin      = −(gpfg_fx + transfer_financing) − (gov_securities − the four lines above: repos,
                     swap-arrangement covered bonds)

``gpfg_fx`` (FX purchases for the GPFG) and ``transfer_financing`` (financing of the transfer to the state,
2018→) are the market legs of the oil-fund transfer: when NB sells fund FX for NOK the krone lands in the
government account. The transfer between the government account and the GPFG krone account is internal to
Norges Bank and is not in the bank-liquidity statistics, so the timing of other_fin is the market leg,
not the book transfer. The government-account *level* is not published here; Δbuffer is implied and the
contract identity therefore holds by construction. The real check is the cross-dataflow one:
Δ structural liquidity (LIQUIDITY_FORECAST STAT_S) = gov_accounts + gov_securities + gpfg_fx + other_fx +
notes_coins + transfer_financing (LIQUIDITY_STATISTICS).

Norges Bank's daily structural-liquidity forecast is the model. It is scored against realised structural
liquidity and against the same change a year earlier, by business-day horizon and by month.

Forward securities come from ``debt_calendar.json`` (programme, quarterly guidance, bond auction and buyback
dates, T-bill auctions per bill): NGB gross = guidance midpoint per auction date; T-bills = calendar sales and
per-bill market amounts at maturity; buybacks = median historical share bought back over the same remaining
days. Months past Norges Bank's path are flagged ``scenario`` and kept out of every headline.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from sovereign_cash import contract
from sovereign_cash.no import fetch

DERIVED = contract.DERIVED
OUT_MONTHLY = DERIVED / "no_cash_monthly.parquet"
OUT_SUMMARY = DERIVED / "no_cash_summary.json"
OUT_DAILY = DERIVED / "no_liquidity_daily.parquet"
OUT_SCORE = DERIVED / "no_nb_forecast_score.parquet"
PRE = (pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31"))
OOS_START = pd.Timestamp("2022-01-01")
HORIZONS = (1, 5, 10, 20, 60)
LINES = {
    "tab1_linje01": "gov_accounts",
    "tab1_linje02": "gov_securities",
    "tab1_linje03": "gpfg_fx",
    "tab1_linje04": "other_fx",
    "tab1_linje05": "notes_coins",
    "tab1_linje06": "nb_net_liquidity",
    "tab1_linje07": "total_liquidity_change",
    "tab1_linje14": "transfer_financing",
    "tab2_linje01": "bond_issue",
    "tab2_linje02": "bond_maturity",
    "tab2_linje07": "bill_issue",
    "tab2_linje08": "bill_maturity",
    "tab4_linje02": "customs_excise",
    "tab4_linje03": "vat",
    "tab4_linje04": "tax_net",
    "tab4_linje05": "nav",
    "tab4_linje06": "local_authorities",
    "tab4_linje07": "oil_tax",
    "tab4_linje08": "health",
    "tab4_linje09": "sdfi",
    "tab4_linje10": "other_net",
    "tab5_linje03": "cb_certificates",
}
VOLUMES = {"tab1_linje08": "sight_deposits_bn", "tab1_linje09": "reserve_deposits_bn", "tab1_linje10": "f_deposits_bn",
           "tab1_linje11": "f_loans_bn"}
TAB4 = ["customs_excise", "vat", "tax_net", "nav", "local_authorities", "oil_tax", "health", "sdfi", "other_net"]
PETROLEUM = ["oil_tax", "sdfi"]
# Norges Bank borrowing programme, quarterly guidance and auction / buyback calendars. Update each December
# (strategy and borrowing programme) and before each quarter (quarterly information).
CALENDAR = Path(__file__).resolve().parent / "debt_calendar.json"


def load_calendar(path: Path = CALENDAR) -> dict:
    return json.loads(path.read_text())


CAL = load_calendar()
NGB_PROGRAMME = {int(y): tuple(p["bonds_bn"]) for y, p in CAL["programmes"].items()}
NGB_QUARTER_GUIDE = {(g["year"], g["quarter"]): tuple(g["bonds_bn"]) for g in CAL["quarter_guidance"]}
BUYBACK_HIST_FROM = pd.Timestamp("2015-01-01")
STRUCT = ["gov_accounts", "gov_securities", "gpfg_fx", "other_fx", "notes_coins", "transfer_financing"]
SOURCES = [
    {"name": "Norges Bank LIQUIDITY_STATISTICS (government accounts, Tab4 specification, government securities, "
             "GPFG FX), business-daily, accumulated from 1 Jan",
     "url": fetch.URLS["liquidity_statistics"], "frequency": "business daily (published Mon/Thu)", "lag_days": 2,
     "verified": True},
    {"name": "Norges Bank LIQUIDITY_FORECAST (realised + forecast structural liquidity, every vintage since 2014)",
     "url": fetch.URLS["liquidity_forecast"], "frequency": "daily path, vintages twice weekly", "lag_days": 1,
     "verified": True},
    {"name": "Norges Bank NOWA", "url": fetch.URLS["nowa"], "frequency": "business daily", "lag_days": 1, "verified": True},
    {"name": "Norges Bank policy rate", "url": fetch.URLS["policy_rate"], "frequency": "business daily", "lag_days": 0,
     "verified": True},
    {"name": "Norges Bank FX transactions for the GPFG, monthly announcement", "url": fetch.URLS["fx_spu"],
     "frequency": "monthly", "lag_days": 0, "verified": True},
    {"name": "Norges Bank government securities primary market (per ISIN, outstanding, own holdings, maturity)",
     "url": fetch.URLS["govt_primary"], "frequency": "per transaction", "lag_days": 0, "verified": True},
] + [{"name": f"{s['name']} — published {s['published']}", "url": s["url"], "frequency": "hand-entered in "
      "sovereign_cash/no/debt_calendar.json", "lag_days": 0, "verified": True} for s in CAL["sources"]]


# --------------------------------------------------------------------------------------
# Daily flows


def daily_flows(stat: pd.DataFrame) -> pd.DataFrame:
    s = stat[(stat["FREQ"] == "B") & (stat["UNIT_MEASURE"] == "ALE") & stat["LIQUIDITY_DETAILS"].isin(LINES)]
    w = s.pivot_table(index="TIME_PERIOD", columns="LIQUIDITY_DETAILS", values="value", aggfunc="last")
    w.index = pd.to_datetime(w.index)
    w = w.rename(columns=LINES).sort_index()
    yr = w.index.year
    w = w.groupby(yr).ffill().fillna(0.0)
    f = w.groupby(yr).diff()
    first = ~pd.Series(yr, index=w.index).duplicated()
    f[first.to_numpy()] = w[first.to_numpy()]
    return f / 1000.0  # NOK bn


def structural_realised(fc: pd.DataFrame) -> pd.Series:
    s = fc[fc["FORECAST_TYPE"] == "STAT_S"].sort_values("FORECAST_DATE")
    s = s.drop_duplicates("TIME_PERIOD", keep="last")
    out = pd.Series(s["value"].to_numpy() / 1000.0, index=pd.to_datetime(s["TIME_PERIOD"])).sort_index()
    return out


def vintages(fc: pd.DataFrame) -> dict[pd.Timestamp, tuple[pd.Timestamp, float, pd.Series]]:
    """FORECAST_DATE → (anchor date, anchor level, forecast path) in NOK bn."""
    out = {}
    st = fc[fc["FORECAST_TYPE"] == "STAT_S"]
    fs = fc[fc["FORECAST_TYPE"] == "FCAST_S"]
    last = st.sort_values("TIME_PERIOD").groupby("FORECAST_DATE").tail(1).set_index("FORECAST_DATE")
    for v, g in fs.groupby("FORECAST_DATE"):
        if v not in last.index:
            continue
        a = last.loc[v]
        path = pd.Series(g["value"].to_numpy() / 1000.0, index=pd.to_datetime(g["TIME_PERIOD"])).sort_index()
        out[pd.Timestamp(v)] = (pd.Timestamp(a["TIME_PERIOD"]), float(a["value"]) / 1000.0, path)
    return out


def _win(t: pd.Timestamp) -> str:
    if PRE[0] <= t <= PRE[1]:
        return "PRE"
    if t >= OOS_START:
        return "OOS"
    return ""


def score_daily(vin: dict, S: pd.Series) -> pd.DataFrame:
    rows = []
    Sd = S.to_dict()
    for v, (a, A, path) in vin.items():
        pd_ = path.to_dict()
        for h in HORIZONS:
            t = a + BDay(h)
            if t not in pd_ or t not in Sd:
                continue
            ly_t, ly_a = t - pd.Timedelta(days=364), a - pd.Timedelta(days=364)
            naive = Sd[ly_t] - Sd[ly_a] if (ly_t in Sd and ly_a in Sd) else np.nan
            rows.append((v, a, t, f"{h}bd", pd_[t] - A, Sd[t] - A, naive))
    df = pd.DataFrame(rows, columns=["vintage", "anchor", "target", "h", "nb", "actual", "naive_ly"])
    return df


def score_monthly(vin: dict, S: pd.Series) -> pd.DataFrame:
    """Change in structural liquidity over calendar month m, from the last vintage before month m−k+1."""
    me = S.resample("ME").last()
    me.index = me.index.to_period("M").to_timestamp()
    vdates = sorted(vin)
    rows = []
    for m in me.index[1:]:
        prev = m - pd.DateOffset(months=1)
        end_m, end_p = m + pd.offsets.MonthEnd(0), prev + pd.offsets.MonthEnd(0)
        act = me[m] - me[prev]
        naive = (me.get(m - pd.DateOffset(months=12), np.nan) - me.get(prev - pd.DateOffset(months=12), np.nan))
        for k in (1, 2, 3):
            cut = m - pd.DateOffset(months=k - 1)
            cand = [v for v in vdates if v < cut]
            if not cand:
                continue
            v = cand[-1]
            a, A, path = vin[v]
            if end_m not in path.index:
                continue
            if a > end_p:
                continue
            start = A if a == end_p else path.get(end_p, np.nan)
            if pd.isna(start):
                continue
            rows.append((m, f"m+{k}", v, path[end_m] - start, act, naive))
    return pd.DataFrame(rows, columns=["month", "h", "vintage", "nb", "actual", "naive_ly"])


BENCHMARK_NAME = {"target": "same weekday a year earlier (change)",
                  "month": "same calendar month a year earlier (change)"}


def summarise_scores(d: pd.DataFrame, key: str) -> list[dict]:
    d = d.assign(win=d[key].map(_win))
    d = d[(d["win"] != "") & d["naive_ly"].notna()]
    out = []
    for (win, h), g in d.groupby(["win", "h"], sort=False):
        out.append({"window": win, "h": h, "model": round(float((g["nb"] - g["actual"]).abs().mean()), 2),
                    "benchmark": round(float((g["naive_ly"] - g["actual"]).abs().mean()), 2),
                    "flat": round(float(g["actual"].abs().mean()), 2), "n": int(len(g)),
                    "model_name": "Norges Bank structural-liquidity forecast",
                    "benchmark_name": BENCHMARK_NAME[key],
                    "unit": "NOK bn, MAE of change in structural liquidity from the vintage anchor"})
    order = {"PRE": 0, "OOS": 1}
    return sorted(out, key=lambda r: (order[r["window"]], int("".join(c for c in r["h"] if c.isdigit()))
                                      + (1000 if r["h"].startswith("m") else 0)))


# --------------------------------------------------------------------------------------
# Monthly contract


def history(fl: pd.DataFrame) -> pd.DataFrame:
    m = fl.resample("MS").sum()
    for c in LINES.values():
        if c not in m:
            m[c] = 0.0
    h = pd.DataFrame(index=m.index)
    h["cash_requirement_bn"] = m["gov_accounts"]
    h["cash_petroleum_bn"] = m[PETROLEUM].sum(axis=1)
    h["cash_nonpetroleum_bn"] = m["gov_accounts"] - h["cash_petroleum_bn"]
    h["bond_gross_bn"] = -m["bond_issue"]
    h["bond_redemption_bn"] = m["bond_maturity"]
    h["net_bonds_bn"] = h["bond_gross_bn"] - h["bond_redemption_bn"]
    h["bill_gross_bn"] = -m["bill_issue"]
    h["bill_redemption_bn"] = m["bill_maturity"]
    h["net_bills_bn"] = h["bill_gross_bn"] - h["bill_redemption_bn"]
    sec_other = m["gov_securities"] - (m["bond_issue"] + m["bond_maturity"] + m["bill_issue"] + m["bill_maturity"])
    h["oil_fund_legs_bn"] = -(m["gpfg_fx"] + m["transfer_financing"])
    h["securities_other_bn"] = -sec_other
    h["other_fin_bn"] = h["oil_fund_legs_bn"] + h["securities_other_bn"]
    h["buffer_change_bn"] = -(m["gov_accounts"] + m["gov_securities"] + m["gpfg_fx"] + m["transfer_financing"])
    h["gov_liquidity_effect_bn"] = -h["buffer_change_bn"]
    for c in TAB4 + ["other_fx", "notes_coins"]:
        h[f"liq_{c}_bn"] = m[c]
    h["is_forecast"] = False
    h["partial_month"] = False
    h["fc_source"] = "outcome"
    return h


def partial_month(fl: pd.DataFrame, S: pd.Series, pm: pd.Timestamp) -> pd.DataFrame:
    """Running month: observed flows to date plus the realised structural change for the days the
    liquidity statistics have not yet covered, attributed to the (non-petroleum) government accounts.
    Kept as a history row (is_forecast=False) flagged partial_month=True."""
    last_obs = fl.index.max()
    row = history(fl[fl.index >= pm]).copy()
    s_m = S[(S.index > last_obs) & (S.index <= pm + pd.offsets.MonthEnd(0))]
    extra = float(s_m.iloc[-1] - S[last_obs]) if len(s_m) and last_obs in S.index else 0.0
    row["cash_requirement_bn"] += extra
    row["cash_nonpetroleum_bn"] += extra
    row["buffer_change_bn"] -= extra
    row["gov_liquidity_effect_bn"] += extra
    row["is_forecast"] = False
    row["partial_month"] = True
    n_obs = int(((fl.index >= pm) & (fl.index <= last_obs)).sum())
    row["bdays_observed"] = n_obs
    row["bdays_in_month"] = _bdays(pm)
    tail = f"+nb_stat_s_to_{s_m.index[-1]:%Y-%m-%d}" if len(s_m) else ""
    row["fc_source"] = f"outcome_to_{last_obs:%Y-%m-%d}{tail}"
    return row


def _bdays(month: pd.Timestamp) -> int:
    return len(pd.bdate_range(month, month + pd.offsets.MonthEnd(0)))


def redemption_schedule(gpm: pd.DataFrame) -> pd.DataFrame:
    """Market amount (outstanding − government own holdings) maturing per month, NOK bn."""
    g = gpm[gpm["UNIT_MEASURE"].isin(["O_V", "V_O"])].sort_values("TIME_PERIOD", kind="stable").copy()
    names = gpm.dropna(subset=["INSTR"]).groupby("ISIN")["INSTR"].last() if "INSTR" in gpm else pd.Series(dtype=str)
    last = g.groupby(["ISIN", "UNIT_MEASURE"]).tail(1).pivot_table(index=["ISIN", "INSTRUMENT_TYPE", "MATURITY"],
                                                                    columns="UNIT_MEASURE", values="value")
    last = last.reset_index()
    own = last["V_O"].fillna(0.0) if "V_O" in last else 0.0
    last["own_bn"] = own / 1000.0
    last["market_bn"] = (last["O_V"] - own) / 1000.0
    last["maturity"] = pd.to_datetime(last["MATURITY"])
    last["month"] = last["maturity"].dt.to_period("M").dt.to_timestamp()
    last["instr"] = last["ISIN"].map(names).fillna(last["ISIN"])
    return last


def _asof(x: pd.DataFrame, t: pd.Timestamp) -> float:
    """Last value at or before t of a frame with columns t, value (0 if none)."""
    y = x[x["t"] <= t]
    return float(y["value"].iloc[-1]) if len(y) else 0.0


def buyback_history(gpm: pd.DataFrame, asof: pd.Timestamp, days: int) -> pd.DataFrame:
    """Per NGB matured between BUYBACK_HIST_FROM and asof: share of its market amount one year before maturity
    that was bought back in the last ``days`` days before maturity."""
    g = gpm[gpm["INSTRUMENT_TYPE"] == "GBON"].copy()
    g["t"] = pd.to_datetime(g["TIME_PERIOD"])
    g["mat"] = pd.to_datetime(g["MATURITY"])
    rows = []
    for isin, x in g.groupby("ISIN"):
        mat = x["mat"].iloc[0]
        if not (BUYBACK_HIST_FROM <= mat < asof):
            continue
        start = mat - pd.DateOffset(years=1)
        ov = x[x["UNIT_MEASURE"] == "O_V"].sort_values("t", kind="stable")
        vo = x[x["UNIT_MEASURE"] == "V_O"].sort_values("t", kind="stable")
        mkt = (_asof(ov, start) - _asof(vo, start)) / 1000.0
        if mkt <= 0:
            continue
        bb = x[(x["ISSUE_TYPE"] == "BB") & (x["UNIT_MEASURE"] == "V_A")]
        late = float(bb.loc[bb["t"] >= mat - pd.Timedelta(days=days), "value"].sum()) / 1000.0
        rows.append({"isin": isin, "maturity": mat, "market_start_bn": mkt, "bought_late_bn": late,
                     "share_late": late / mkt})
    return pd.DataFrame(rows, columns=["isin", "maturity", "market_start_bn", "bought_late_bn", "share_late"])


def buyback_outlook(gpm: pd.DataFrame, cal: dict, asof: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Each NGB maturing in (asof, end]: expected further buybacks = median historical share bought back over the
    same remaining days × its market amount a year before maturity; range = min..max historical share."""
    red = redemption_schedule(gpm)
    red = red[(red["INSTRUMENT_TYPE"] == "GBON") & (red["maturity"] > asof) & (red["maturity"] <= end)]
    g = gpm[gpm["INSTRUMENT_TYPE"] == "GBON"].copy()
    g["t"] = pd.to_datetime(g["TIME_PERIOD"])
    out = []
    for _, r in red.sort_values("maturity").iterrows():
        mat, now = r["maturity"], float(r["market_bn"])
        days = int((mat - asof).days)
        hist = buyback_history(gpm, asof, days)
        x = g[g["ISIN"] == r["ISIN"]]
        start = mat - pd.DateOffset(years=1)
        mkt0 = (_asof(x[x["UNIT_MEASURE"] == "O_V"].sort_values("t", kind="stable"), start)
                - _asof(x[x["UNIT_MEASURE"] == "V_O"].sort_values("t", kind="stable"), start)) / 1000.0 or now
        sh = hist["share_late"]
        med, lo, hi = (float(sh.median()), float(sh.min()), float(sh.max())) if len(sh) else (0.0, 0.0, 0.0)
        exp = min(med * mkt0, now)
        dates = sorted(pd.Timestamp(b["date"]) for b in cal.get("buybacks", [])
                       if pd.Timestamp(b["maturity"]) == mat and pd.Timestamp(b["date"]) > asof)
        out.append({"isin": r["ISIN"], "instr": r["instr"], "maturity": mat, "market_now_bn": now,
                    "market_year_before_bn": mkt0, "days_left": days, "n_hist": int(len(sh)),
                    "hist_from": int(BUYBACK_HIST_FROM.year), "share_median": med, "share_min": lo, "share_max": hi,
                    "expected_buyback_bn": exp, "buyback_dates": dates,
                    "redemption_central_bn": now - exp,
                    "redemption_low_bn": now - min(hi * mkt0, now), "redemption_high_bn": now - min(lo * mkt0, now)})
    return out


def bill_path(gpm: pd.DataFrame, cal: dict, months: pd.DatetimeIndex, asof: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Market T-bills by month from the published auction calendar: gross = calendar sales to the market,
    redemption = market amount of each bill at maturity (primary-market snapshot + calendar taps after it).
    After the calendar ends the scenario rolls maturities at constant stock."""
    tb = gpm[gpm["INSTRUMENT_TYPE"] == "TBIL"]
    snap = pd.to_datetime(tb["TIME_PERIOD"]).max()
    red = redemption_schedule(gpm)
    red = red[(red["INSTRUMENT_TYPE"] == "TBIL") & (red["maturity"] > asof)]
    snap_mkt = red.groupby("maturity")["market_bn"].sum().to_dict()
    auc = pd.DataFrame(cal.get("bill_auctions", []), columns=["date", "bill", "maturity", "market_bn", "own_bn"])
    auc["date"], auc["maturity"] = pd.to_datetime(auc["date"]), pd.to_datetime(auc["maturity"])
    auc["month"] = auc["date"].dt.to_period("M").dt.to_timestamp()
    late = auc[auc["date"] > snap]
    at_mat = defaultdict(float, snap_mkt)
    for _, a in late.iterrows():
        if a["maturity"] > asof:
            at_mat[a["maturity"]] += a["market_bn"]
    stock_now = float(sum(snap_mkt.values()) + late.loc[(late["date"] <= asof) & (late["maturity"] > asof), "market_bn"].sum())
    cal_end = auc["month"].max() if len(auc) else pd.Timestamp(asof.year, asof.month, 1)
    rows, s = [], stock_now
    for m in months:
        rd = float(sum(v for k, v in at_mat.items() if m <= k <= m + pd.offsets.MonthEnd(0)))
        if m <= cal_end:
            gr = float(auc.loc[(auc["month"] == m) & (auc["date"] > asof), "market_bn"].sum())
            basis = "calendar"
        else:
            gr, basis = rd, "rolled"
        s += gr - rd
        rows.append({"month": m, "bill_gross_bn": gr, "bill_redemption_bn": rd, "bill_stock_market_bn": s,
                     "bill_basis": basis})
    return pd.DataFrame(rows).set_index("month"), stock_now


def spu_by_month(spu: pd.DataFrame) -> dict[pd.Timestamp, float]:
    """Announced daily GPFG FX transaction per month, NOK bn/day, + = Norges Bank buys FX (pays NOK out)."""
    spu = spu.sort_values("TRANSACTION_MONTH")
    return {pd.Timestamp(r["TRANSACTION_MONTH"] + "-01"): (1.0 if r["TRANSACTION_TYPE"] == "B" else -1.0) * r["value"] / 1000.0
            for _, r in spu.iterrows()}


def ngb_month_shares(mflows: pd.DataFrame, last_hist: pd.Timestamp, years: int = 3) -> pd.Series:
    """Average share of each calendar month in annual gross NGB issuance over the last complete years."""
    g = -mflows["bond_issue"]
    yrs = [y for y in range(last_hist.year - years, last_hist.year)]
    tab = pd.DataFrame({y: g[g.index.year == y].groupby(g[g.index.year == y].index.month).sum() for y in yrs}).fillna(0.0)
    sh = (tab / tab.sum()).mean(axis=1)
    return sh.reindex(range(1, 13)).fillna(0.0)


def _quarter(m: pd.Timestamp) -> int:
    return (m.month - 1) // 3 + 1


def ngb_gross_path(months: pd.DatetimeIndex, mflows: pd.DataFrame, asof: pd.Timestamp,
                   cal: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """Forward NGB gross (bn) and a basis label per month.

    Quarter with guidance: midpoint less issued in the quarter so far, equal per remaining auction date.
    Rest of a year with a programme: midpoint less issued YTD and guided quarters, per remaining auction date.
    Year without a programme (scenario): latest programme midpoint × historical month shares."""
    cal = CAL if cal is None else cal
    prog = {int(y): tuple(p["bonds_bn"]) for y, p in cal["programmes"].items()}
    guide = {(g["year"], g["quarter"]): tuple(g["bonds_bn"]) for g in cal["quarter_guidance"]}
    dates = pd.to_datetime(pd.Series(cal.get("bond_auctions", []), dtype=str))
    fut = dates[dates > asof].dt.to_period("M").dt.to_timestamp()
    n_dates = fut.value_counts()
    sh = ngb_month_shares(mflows, asof)
    issued = -mflows["bond_issue"]
    issued = issued[issued.index <= asof]
    out, lab = {}, {}

    def spread(ms: list, total: float, label: str) -> None:
        n = np.array([float(n_dates.get(m, 0)) for m in ms])
        if n.sum() > 0:
            w = n / n.sum()
        else:
            w = sh.reindex([m.month for m in ms]).to_numpy()
            w = w / w.sum() if w.sum() > 0 else np.full(len(ms), 1.0 / len(ms))
        for m, wi, ni in zip(ms, w, n):
            out[m], lab[m] = total * wi, label + (f" · {ni:.0f} auctions" if ni else "")

    for yr in sorted({m.year for m in months}):
        ym = [m for m in months if m.year == yr]
        for (gy, q), (lo, hi) in sorted(guide.items()):
            qm = [m for m in ym if gy == yr and _quarter(m) == q]
            if not qm:
                continue
            done = float(issued[(issued.index.year == yr) & ((issued.index.month - 1) // 3 + 1 == q)].sum())
            spread(qm, max((lo + hi) / 2 - done, 0.0), f"Q{q}-{yr} guidance {lo:.0f}–{hi:.0f}bn midpoint")
        rest = [m for m in ym if m not in out]
        if not rest:
            continue
        if yr in prog:
            lo, hi = prog[yr]
            left = max((lo + hi) / 2 - float(issued[issued.index.year == yr].sum()) - sum(out.get(m, 0.0) for m in ym), 0.0)
            spread(rest, left, f"{yr} programme {lo:.0f}–{hi:.0f}bn midpoint less issued")
        else:
            ly = max(prog)
            mid = float(np.mean(prog[ly]))
            w = sh.reindex([m.month for m in rest]).to_numpy()
            for m, wi in zip(rest, w):
                out[m] = mid * wi
                lab[m] = (f"scenario: no {yr} programme yet; {ly} midpoint × {asof.year - 3}–{asof.year - 1} "
                          "month shares")
    return pd.Series(out).reindex(months).fillna(0.0), pd.Series(lab).reindex(months).fillna("")


def forward(fl: pd.DataFrame, hist: pd.DataFrame, vin: dict, S: pd.Series, spu: pd.DataFrame,
            gpm: pd.DataFrame, cal: dict | None = None) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.cash_season import HORIZON_END, forward_months as _fwd_m

    cal = CAL if cal is None else cal
    last_hist = hist.index.max()
    last_obs = fl.index.max()
    v = max(vin)
    a, A, path = vin[v]
    h_live = _fwd_m(last_hist)
    months = pd.date_range(last_hist + pd.DateOffset(months=1), periods=h_live, freq="MS")
    if months.max() < HORIZON_END:
        months = pd.date_range(last_hist + pd.DateOffset(months=1), HORIZON_END, freq="MS")
    mflows = fl.resample("MS").sum()
    for c in LINES.values():
        if c not in mflows:
            mflows[c] = 0.0
    red = redemption_schedule(gpm)
    red = red[red["maturity"] > last_obs]
    red_b = red[red["INSTRUMENT_TYPE"] == "GBON"].groupby("month")["market_bn"].sum().to_dict()
    buyb = defaultdict(float)
    bbs = buyback_outlook(gpm, cal, last_obs, months[-1] + pd.offsets.MonthEnd(0))
    for b in bbs:
        bm = [d.to_period("M").to_timestamp() for d in b["buyback_dates"]]
        bm = [m for m in bm if m in months]
        b["expected_in_path_bn"] = b["expected_buyback_bn"] if bm else 0.0
        for m in bm:
            buyb[m] += b["expected_buyback_bn"] / len(bm)
        if bm:
            red_b[b["maturity"].to_period("M").to_timestamp()] -= b["expected_buyback_bn"]
    bills, bill_now = bill_path(gpm, cal, months, last_obs)
    spu_m = spu_by_month(spu)
    ngb, ngb_lab = ngb_gross_path(months, mflows, last_obs, cal)
    tf3 = float(mflows["transfer_financing"].loc[:last_hist].tail(3).mean())

    rows = []
    end_last = last_hist + pd.offsets.MonthEnd(0)
    prev_level = A if a == end_last else float(path.get(end_last, S.loc[:end_last].iloc[-1]))
    for m in months:
        ly = m - pd.DateOffset(months=12)
        g = {k: float(mflows[k].get(ly, 0.0)) for k in mflows.columns}
        r = {"month": m}
        r["bond_gross_bn"] = float(ngb[m])
        r["ngb_gross_basis"] = ngb_lab[m]
        r["bond_buyback_bn"] = float(buyb.get(m, 0.0))
        r["bond_redemption_bn"] = float(red_b.get(m, 0.0)) + r["bond_buyback_bn"]
        for c in ("bill_gross_bn", "bill_redemption_bn", "bill_stock_market_bn", "bill_basis"):
            r[c] = bills.loc[m, c]
        if m in spu_m:
            per_day, r["gpfg_fx_basis"] = spu_m[m], "announced"
        else:
            per_day, r["gpfg_fx_basis"] = spu_m[max(spu_m)], "latest_announced_carried"
        r["gpfg_fx_nok_bn_per_day"] = per_day
        gpfg = per_day * _bdays(m)
        r["oil_fund_legs_bn"] = -(gpfg + tf3)
        r["securities_other_bn"] = 0.0
        sec = -r["bond_gross_bn"] + r["bond_redemption_bn"] - r["bill_gross_bn"] + r["bill_redemption_bn"]
        end = m + pd.offsets.MonthEnd(0)
        if end in path.index:
            dS = float(path[end]) - prev_level
            prev_level = float(path[end])
            cr = dS - (sec + gpfg + tf3 + g["other_fx"] + g["notes_coins"])
            r["fc_source"] = f"norges_bank_vintage_{v:%Y-%m-%d}"
            r["structural_change_fc_bn"] = dS
            r["scenario"] = False
        else:
            # beyond NB's path: same month last year, a scenario (it does not beat zero OOS)
            cr = g["gov_accounts"]
            r["fc_source"] = "same_month_last_year"
            r["structural_change_fc_bn"] = np.nan
            r["scenario"] = True
        r["cash_requirement_bn"] = cr
        # the petroleum / non-petroleum split of a residual or a copied month carries no information
        r["cash_petroleum_bn"] = np.nan
        r["cash_nonpetroleum_bn"] = np.nan
        # Residual: NB's structural path less this repo's own securities / GPFG / notes assumptions. Not an
        # official forecast of the government accounts; structural_change_fc_bn is NB's published series.
        r["cash_requirement_nb_implied_bn"] = cr if r["fc_source"].startswith("norges") else np.nan
        rows.append(r)
    fw = pd.DataFrame(rows).set_index("month")
    fw["net_bonds_bn"] = fw["bond_gross_bn"] - fw["bond_redemption_bn"]
    fw["net_bills_bn"] = fw["bill_gross_bn"] - fw["bill_redemption_bn"]
    fw["other_fin_bn"] = fw["oil_fund_legs_bn"] + fw["securities_other_bn"]
    fw["buffer_change_bn"] = -fw["cash_requirement_bn"] + fw["net_bonds_bn"] + fw["net_bills_bn"] + fw["other_fin_bn"]
    fw["gov_liquidity_effect_bn"] = -fw["buffer_change_bn"]
    fw["is_forecast"] = True
    fw["partial_month"] = False
    meta = {"vintage": f"{v:%Y-%m-%d}", "anchor": f"{a:%Y-%m-%d}", "anchor_level_bn": round(A, 1),
            "path_end": f"{path.index.max():%Y-%m-%d}", "path_end_level_bn": round(float(path.iloc[-1]), 1),
            "spu_announced_month": f"{max(spu_m):%Y-%m}", "spu_announced_nok_mn_per_day": round(spu_m[max(spu_m)] * 1000.0),
            "spu_previous_nok_mn_per_day": round(spu_m[sorted(spu_m)[-2]] * 1000.0),
            "bill_stock_market_now_bn": round(bill_now, 1),
            "transfer_financing_trailing3m_bn": round(tf3, 2), "buybacks": bbs}
    return fw, meta


def _latest_path(fc: pd.DataFrame, ftype: str) -> tuple[pd.Timestamp | None, pd.Series]:
    x = fc[fc["FORECAST_TYPE"] == ftype]
    if x.empty:
        return None, pd.Series(dtype=float)
    v = x["FORECAST_DATE"].max()
    x = x[x["FORECAST_DATE"] == v]
    return pd.Timestamp(v), pd.Series(x["value"].to_numpy() / 1000.0, index=pd.to_datetime(x["TIME_PERIOD"])).sort_index()


def volumes(stat: pd.DataFrame) -> pd.DataFrame:
    """Norges Bank operations and bank deposits (volume, NOK bn): sight / reserve deposits, F-deposits, F-loans."""
    s = stat[(stat["FREQ"] == "B") & (stat["UNIT_MEASURE"] == "V") & stat["LIQUIDITY_DETAILS"].isin(VOLUMES)]
    w = s.pivot_table(index="TIME_PERIOD", columns="LIQUIDITY_DETAILS", values="value", aggfunc="last")
    w.index = pd.to_datetime(w.index)
    return (w.rename(columns=VOLUMES) / 1000.0).sort_index()


def daily_frame(fl: pd.DataFrame, S: pd.Series, vin: dict, nowa: pd.DataFrame, pol: pd.DataFrame,
                fc: pd.DataFrame | None = None, stat: pd.DataFrame | None = None) -> pd.DataFrame:
    d = fl.copy()
    d["structural_liquidity_bn"] = S.reindex(d.index)
    d["gov_account_inflow_bn"] = -(fl["gov_accounts"] + fl["gov_securities"] + fl["gpfg_fx"] + fl["transfer_financing"])
    if "cb_certificates" in fl:
        d["cb_certificates_outstanding_bn"] = -fl["cb_certificates"].cumsum()
    if stat is not None:
        d = d.join(volumes(stat), how="left")
    v = max(vin)
    path = vin[v][2]
    fut = path[path.index > d.index.max()]
    fut = fut[fut.index.dayofweek < 5]
    extra = {"structural_liquidity_fc_bn": fut}
    if fc is not None:
        st = fc[fc["FORECAST_TYPE"] == "STAT_T"].sort_values("FORECAST_DATE").drop_duplicates("TIME_PERIOD", keep="last")
        T = pd.Series(st["value"].to_numpy() / 1000.0, index=pd.to_datetime(st["TIME_PERIOD"])).sort_index()
        d["total_liquidity_bn"] = T.reindex(d.index)
        for ftype, col in (("FCAST_CBC", "structural_liquidity_incl_cbc_fc_bn"), ("FCAST_T", "total_liquidity_fc_bn")):
            _, pth = _latest_path(fc, ftype)
            extra[col] = pth[pth.index.dayofweek < 5]
    fut_idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in extra.values()]) - set(d.index)))
    d = pd.concat([d, pd.DataFrame(index=fut_idx)]).sort_index()
    for col, s in extra.items():
        d[col] = s.reindex(d.index).astype(float)
    n = nowa.set_index(pd.to_datetime(nowa["TIME_PERIOD"]))["value"]
    p = pol.set_index(pd.to_datetime(pol["TIME_PERIOD"]))["value"]
    d["nowa"] = n.reindex(d.index)
    d["policy_rate"] = p.reindex(d.index).ffill()
    d["nowa_minus_policy_bp"] = (d["nowa"] - d["policy_rate"]) * 100.0
    d["is_forecast"] = d.index > fl.index.max()
    d.index.name = "date"
    return d


def ladder(gpm: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """Outstanding NGBs and T-bills by maturity: market amount and government own holdings, NOK bn."""
    r = redemption_schedule(gpm)
    r = r[r["maturity"] > asof].sort_values("maturity")
    return [{"instr": x["instr"], "isin": x["ISIN"], "type": "bond" if x["INSTRUMENT_TYPE"] == "GBON" else "bill",
             "maturity": f"{x['maturity']:%Y-%m-%d}", "market_bn": round(float(x["market_bn"]), 2),
             "own_bn": round(float(x["own_bn"]), 2)} for _, x in r.iterrows()]


def calendar_next(fw: pd.DataFrame, cal: dict, asof: pd.Timestamp, bbs: list[dict], gpm: pd.DataFrame,
                  n_months: int = 3) -> list[dict]:
    """Dated events over the next n months: NGB auctions (path size per date), buybacks (expected), T-bill
    auctions (calendar size), and bill / bond maturities (market amount at maturity)."""
    end = (asof + pd.DateOffset(months=n_months)) + pd.offsets.MonthEnd(0)
    ev = []
    dates = [pd.Timestamp(d) for d in cal.get("bond_auctions", [])]
    for d in dates:
        if asof < d <= end:
            m = d.to_period("M").to_timestamp()
            n = sum(1 for x in dates if asof < x and x.to_period("M").to_timestamp() == m)
            size = float(fw.loc[m, "bond_gross_bn"]) / n if m in fw.index and n else np.nan
            ev.append({"date": d, "event": "NGB auction", "instrument": "line announced two business days before",
                       "size_bn": size, "basis": "guidance midpoint ÷ auction dates"})
    for b in bbs:
        for d in b["buyback_dates"]:
            if asof < d <= end:
                ev.append({"date": d, "event": "NGB buyback", "instrument": b["instr"],
                           "size_bn": b["expected_buyback_bn"] / len(b["buyback_dates"]),
                           "basis": f"median historical share ({100 * b['share_median']:.0f}%)"})
        if asof < b["maturity"] <= end:
            ev.append({"date": b["maturity"], "event": "NGB maturity", "instrument": b["instr"],
                       "size_bn": b["redemption_central_bn"], "basis": "market amount less expected buybacks"})
    snap = pd.to_datetime(gpm.loc[gpm["INSTRUMENT_TYPE"] == "TBIL", "TIME_PERIOD"]).max()
    red = redemption_schedule(gpm)
    red = red[(red["INSTRUMENT_TYPE"] == "TBIL") & (red["maturity"] > asof)]
    at_mat = red.groupby("maturity")["market_bn"].sum().to_dict()
    names = red.groupby("maturity")["instr"].first().to_dict()
    for a in cal.get("bill_auctions", []):
        d, mt = pd.Timestamp(a["date"]), pd.Timestamp(a["maturity"])
        if d > snap and mt > asof:
            at_mat[mt] = at_mat.get(mt, 0.0) + a["market_bn"]
            names.setdefault(mt, a["bill"])
        if asof < d <= end:
            ev.append({"date": d, "event": "T-bill auction" + (" (new 12m)" if a.get("own_bn") else ""),
                       "instrument": a["bill"], "size_bn": float(a["market_bn"]), "basis": "published calendar"})
    for mt, v in at_mat.items():
        if asof < mt <= end:
            ev.append({"date": mt, "event": "T-bill maturity", "instrument": names.get(mt, ""), "size_bn": float(v),
                       "basis": "market amount incl. calendar taps"})
    ev.sort(key=lambda e: e["date"])
    return [{**e, "date": f"{e['date']:%Y-%m-%d}", "size_bn": round(float(e["size_bn"]), 2)} for e in ev]


def build(write: bool = True) -> tuple[pd.DataFrame, dict]:
    stat = fetch.read("liquidity_statistics")
    fc = fetch.read("liquidity_forecast")
    fl = daily_flows(stat)
    S = structural_realised(fc)
    vin = vintages(fc)
    sd = score_daily(vin, S)
    sm = score_monthly(vin, S)
    score = summarise_scores(sd, "target") + summarise_scores(sm, "month")

    hist = history(fl)
    # Complete months, plus the running month as a history row flagged partial_month=True.
    last_obs = fl.index.max()
    parts = [hist]
    if last_obs < last_obs + pd.offsets.BMonthEnd(0):
        pm = last_obs.to_period("M").to_timestamp()
        hist = hist[hist.index < pm]
        part = partial_month(fl, S, pm)
        parts = [hist, part]
        base = part
    else:
        base = hist
    fw, meta = forward(fl, base, vin, S, fetch.read("fx_spu"), fetch.read("govt_primary"))
    from sovereign_cash.no import bottom_up_macro as BU
    bu_report = BU.run_report(fl, vin, fw, meta, write=write)
    df = pd.concat(parts + [fw]).sort_index()
    df.index.name = "month"
    df = df.reset_index()
    df["is_forecast"] = df["is_forecast"].astype(bool)
    df["partial_month"] = df["partial_month"].fillna(False).astype(bool)
    df["scenario"] = df["scenario"].fillna(False).astype(bool)
    if "cash_requirement_bottomup_bn" in df.columns:
        df.loc[~df["is_forecast"], "cash_requirement_bottomup_bn"] = df.loc[
            ~df["is_forecast"], "cash_requirement_bn"]
    resid = contract.validate(df[~df["is_forecast"]].reset_index(drop=True))
    contract.validate(df.reset_index(drop=True))

    me = S.resample("ME").last()
    me.index = me.index.to_period("M").to_timestamp()
    ms = fl.resample("MS").sum()
    xchk = (me.diff() - ms[STRUCT].sum(axis=1)).dropna()
    xchk = xchk[(xchk.index >= pd.Timestamp("2014-02-01")) & (xchk.index <= hist.index.max() if len(hist) else True)]
    xbig = xchk[xchk.abs() > 0.5]
    daily = daily_frame(fl, S, vin, fetch.read("nowa"), fetch.read("policy_rate"), fc, stat)
    sp = daily.loc[(daily.index >= pd.Timestamp("2022-01-01")) & ~daily["is_forecast"], "nowa_minus_policy_bp"].dropna()

    cr = hist["cash_requirement_bn"]
    smly = pd.DataFrame({"actual": cr, "smly": cr.shift(12),
                         "mean3y": pd.concat([cr.shift(12 * j) for j in (1, 2, 3)], axis=1).mean(axis=1, skipna=False)})
    smly_score = []
    for win, lo, hi in (("PRE", PRE[0], PRE[1]), ("OOS", OOS_START, hist.index.max())):
        g = smly.loc[lo:hi].dropna()
        smly_score.append({"window": win, "n": int(len(g)),
                           "same_month_last_year": round(float((g["smly"] - g["actual"]).abs().mean()), 2),
                           "same_month_3y_mean": round(float((g["mean3y"] - g["actual"]).abs().mean()), 2),
                           "zero": round(float(g["actual"].abs().mean()), 2)})
    f12 = df[df["is_forecast"]]
    part_row = df[df["partial_month"]]
    nbm = f12[~f12["scenario"]]
    scen = f12[f12["scenario"]]
    s = lambda frame, col: round(float(frame[col].sum()), 1)  # noqa: E731
    yr_now = last_obs.year
    gq = {k: v for k, v in NGB_QUARTER_GUIDE.items() if any((m.year, _quarter(m)) == k for m in nbm["month"])}
    ytd_ngb = float(-fl.loc[fl.index.year == yr_now, "bond_issue"].sum())
    ytd_legs = float(df.loc[(~df["is_forecast"]) & (df["month"].dt.year == yr_now), "oil_fund_legs_bn"].sum())
    vol = volumes(stat)
    cbc_v, cbc = _latest_path(fc, "FCAST_CBC")
    gpm = fetch.read("govt_primary")
    lad = ladder(gpm, last_obs)
    cal_next = calendar_next(fw, CAL, last_obs, meta["buybacks"], gpm)
    cbc_out = daily["cb_certificates_outstanding_bn"].dropna()
    cbc_first = cbc_out[cbc_out.abs() > 0.01]
    recent = vol[vol.index > last_obs - pd.Timedelta(days=365)]
    f_active = bool((recent[["f_loans_bn", "f_deposits_bn"]].fillna(0.0).abs() > 0.01).any().any())
    prog = CAL["programmes"].get(str(yr_now), {})
    smly_oos = next((r for r in smly_score if r["window"] == "OOS"), {})
    dj = [m for m in xbig.index if m.month in (12, 1)]

    def block(frame: pd.DataFrame) -> dict:
        return {"window": f"{frame['month'].min():%Y-%m}..{frame['month'].max():%Y-%m}" if len(frame) else "",
                "months": int(len(frame)),
                "cash_requirement_bn": s(frame, "cash_requirement_bn"),
                "oil_fund_legs_bn": s(frame, "oil_fund_legs_bn"),
                "bond_gross_bn": s(frame, "bond_gross_bn"), "bond_redemption_bn": s(frame, "bond_redemption_bn"),
                "bond_buyback_bn": s(frame, "bond_buyback_bn"),
                "net_bonds_bn": s(frame, "net_bonds_bn"), "bill_gross_bn": s(frame, "bill_gross_bn"),
                "bill_redemption_bn": s(frame, "bill_redemption_bn"), "net_bills_bn": s(frame, "net_bills_bn"),
                "securities_other_bn": s(frame, "securities_other_bn"),
                "structural_change_fc_bn": s(frame, "structural_change_fc_bn"),
                "buffer_change_bn": s(frame, "buffer_change_bn")}

    nxt = meta["buybacks"][0] if meta["buybacks"] else None
    bill_end = nbm["bill_stock_market_bn"].iloc[-1] if len(nbm) else np.nan
    path_end = pd.Timestamp(meta["path_end"])
    scen_years = sorted({m.year for m in scen["month"]})
    scen_ngb = "; ".join(
        f"{max(NGB_PROGRAMME)} programme midpoint × {yr_now - 3}–{yr_now - 1} month shares (no {y} programme yet)"
        if y not in NGB_PROGRAMME else f"{y} programme midpoint less issued" for y in scen_years)

    summary = {
        "country": "no", "ccy": "NOK", "asof": df.loc[~df["is_forecast"], "month"].max().strftime("%Y-%m"),
        "asof_partial": bool(len(part_row)),
        "asof_last_complete_month": hist.index.max().strftime("%Y-%m"),
        "asof_daily": f"{last_obs:%Y-%m-%d}",
        "sources": SOURCES,
        "buffer": ("Consolidated NOK position of the Treasury's accounts and the GPFG krone account at Norges Bank "
                   "(transfers between them are internal to Norges Bank and invisible here). Level not published; "
                   "Δbuffer = −(government-account + government-securities + GPFG FX + transfer-financing liquidity "
                   "effects), so the history identity holds by construction."),
        "identity_resid_bn": round(float(resid), 6),
        "structural_crosscheck_resid_bn": {"max_abs": round(float(xchk.abs().max()), 2),
                                           "median_abs": round(float(xchk.abs().median()), 3),
                                           "n_months": int(len(xchk)),
                                           "months_abs_gt_0p5bn": {f"{m:%Y-%m}": round(float(v), 2) for m, v in xbig.items()}},
        "score": score,
        "score_note": (f"Norges Bank's structural-liquidity forecast, every vintage since {min(vin):%Y}. Business-day "
                       "rows: change from the vintage's last realised day; benchmark the same weekday a year earlier. "
                       "Month rows: change over month m from the last vintage before m−k+1; benchmark the same "
                       f"calendar month a year earlier. Flat = no change. PRE {PRE[0]:%Y}–{PRE[1]:%Y}, OOS "
                       f"{OOS_START:%Y}→."),
        "own_forecast_score": ("unscored: this page's Δbuffer path is Norges Bank's structural path less this page's "
                               "securities and fund-FX assumptions; only Norges Bank's structural forecast is scored"),
        "cash_requirement_naive_score": smly_score,
        "bottom_up": bu_report,
        "forward": {
            "window": f"{nbm['month'].min():%Y-%m}..{nbm['month'].max():%Y-%m}",
            "horizon_end": f"{f12['month'].max():%Y-%m}" if len(f12) else None,
            "full_forward_window": (f"{f12['month'].min():%Y-%m}..{f12['month'].max():%Y-%m}"
                                    if len(f12) else None),
            "scenario_window": (f"{scen['month'].min():%Y-%m}..{scen['month'].max():%Y-%m}" if len(scen) else None),
            "nb_horizon": block(nbm),
            "scenario": block(scen),
            "partial_month": (None if part_row.empty else {
                "month": f"{part_row['month'].iloc[0]:%Y-%m}", "source": part_row["fc_source"].iloc[0],
                "bdays_observed": int(part_row["bdays_observed"].iloc[0]),
                "bdays_in_month": int(part_row["bdays_in_month"].iloc[0]),
                "cash_requirement_bn": round(float(part_row["cash_requirement_bn"].iloc[0]), 1),
                "gov_liquidity_effect_bn": round(float(part_row["gov_liquidity_effect_bn"].iloc[0]), 1),
                "buffer_change_bn": round(float(part_row["buffer_change_bn"].iloc[0]), 1)}),
            "nb_vintage": {k: v for k, v in meta.items() if k != "buybacks"},
            "nb_covered_months": [f"{m:%Y-%m}" for m in nbm["month"]],
            "structural_liquidity_now_bn": round(float(S.iloc[-1]), 1),
            "structural_liquidity_now_date": f"{S.index[-1]:%Y-%m-%d}",
            "structural_liquidity_year_end_fc_bn": meta["path_end_level_bn"],
            "structural_liquidity_incl_cb_certificates_year_end_fc_bn": (round(float(cbc.iloc[-1]), 1) if len(cbc) else None),
            "structural_liquidity_incl_cb_certificates_vintage": (f"{cbc_v:%Y-%m-%d}" if cbc_v is not None else None),
            "total_liquidity_now_bn": round(float(daily["total_liquidity_bn"].dropna().iloc[-1]), 1),
            "cb_certificates_outstanding_bn": round(float(cbc_out.iloc[-1]), 1) if len(cbc_out) else None,
            "cb_certificates_first_date": f"{cbc_first.index[0]:%Y-%m-%d}" if len(cbc_first) else None,
            "f_loans_bn": round(float(vol["f_loans_bn"].dropna().iloc[-1]), 1),
            "f_deposits_bn": round(float(vol["f_deposits_bn"].dropna().iloc[-1]), 1),
            "f_operations_active_365d": f_active,
            "gpfg_fx": {"announced_month": meta["spu_announced_month"],
                        "announced_nok_mn_per_day": meta["spu_announced_nok_mn_per_day"],
                        "nb_buys_fx": meta["spu_announced_nok_mn_per_day"] > 0,
                        "previous_nok_mn_per_day": meta["spu_previous_nok_mn_per_day"],
                        "forward_basis": "announced month, then the latest announcement carried",
                        "ytd_year": yr_now, "oil_fund_legs_ytd_bn": round(ytd_legs, 1)},
            "ngb": {"year": yr_now, "issued_ytd_bn": round(ytd_ngb, 1),
                    "programme_bn": list(NGB_PROGRAMME.get(yr_now, ())),
                    "programme_path_bn": round(ytd_ngb + s(nbm[nbm["month"].dt.year == yr_now], "bond_gross_bn"), 1),
                    "quarter_guidance": [{"year": y, "quarter": q, "bn": list(b),
                                          "auctions": int(sum(1 for d in CAL["bond_auctions"]
                                                              if pd.Timestamp(d) > last_obs
                                                              and (pd.Timestamp(d).year, _quarter(pd.Timestamp(d))) == (y, q))),
                                          "path_bn": s(nbm[(nbm["month"].dt.year == y)
                                                           & ((nbm["month"].dt.month - 1) // 3 + 1 == q)], "bond_gross_bn")}
                                         for (y, q), b in sorted(gq.items())],
                    "next_programme_published": str(yr_now + 1) in CAL["programmes"],
                    "scenario_basis": (scen["ngb_gross_basis"].iloc[0] if len(scen) else None)},
            "next_redemption": ({"instr": nxt["instr"], "isin": nxt["isin"], "maturity": f"{nxt['maturity']:%Y-%m-%d}",
                                 "market_now_bn": round(nxt["market_now_bn"], 1),
                                 "central_bn": round(nxt["redemption_central_bn"], 1),
                                 "low_bn": round(nxt["redemption_low_bn"], 1),
                                 "high_bn": round(nxt["redemption_high_bn"], 1),
                                 "expected_buyback_bn": round(nxt["expected_buyback_bn"], 1),
                                 "expected_in_path_bn": round(nxt["expected_in_path_bn"], 1),
                                 "buyback_dates": [f"{d:%Y-%m-%d}" for d in nxt["buyback_dates"]],
                                 "share_median": round(nxt["share_median"], 3), "share_min": round(nxt["share_min"], 3),
                                 "share_max": round(nxt["share_max"], 3), "n_hist": nxt["n_hist"],
                                 "hist_from": nxt["hist_from"], "days_left": nxt["days_left"]} if nxt else None),
            "bills": {"market_stock_now_bn": meta["bill_stock_market_now_bn"],
                      "stock_horizon_end_bn": round(float(bill_end), 1),
                      "aim_end_year_bn": prog.get("bill_market_stock_end_bn"),
                      "aim_range_bn": prog.get("bill_market_stock_range_bn"),
                      "calendar_auctions_left": int(sum(1 for b in CAL["bill_auctions"] if pd.Timestamp(b["date"]) > last_obs)),
                      "net_horizon_bn": s(nbm, "net_bills_bn")},
            "calendar_next": cal_next,
            "ladder": lad,
            "ladder_asof": f"{pd.to_datetime(gpm['TIME_PERIOD']).max():%Y-%m-%d}",
        },
        "nowa_dropped": {"reason": "NOWA − policy rate is noise at this resolution; chart removed",
                         "median_bp_2022on": round(float(sp.median()), 2), "sd_bp_2022on": round(float(sp.std()), 2)},
        "notes": [
            f"History from {df['month'].min():%b-%Y}; Norges Bank forecast vintages from {min(vin):%b-%Y}."
            + (f" {part_row['month'].iloc[0]:%b-%Y} is partial: {int(part_row['bdays_observed'].iloc[0])} of "
               f"{int(part_row['bdays_in_month'].iloc[0])} business days observed; the rest is Norges Bank's realised "
               "structural liquidity, booked to fiscal flows." if len(part_row) else ""),
            "Inside Norges Bank's path, fiscal flows are a residual: Norges Bank's structural-liquidity change less this "
            "page's NGB, T-bill, fund-FX and notes-and-coins paths. Not an official forecast; shown as a total only.",
            ((f"After {path_end:%d %b %Y} the chart shows a scenario: fiscal flows are "
              + ("Tab4×macro (mainland GDP + Brent; SHIP); "
                 if bu_report.get("ship") else
                 f"same-month-last-year (OOS MAE {contract.fmt_num(smly_oos.get('same_month_last_year'))} vs "
                 f"{contract.fmt_num(smly_oos.get('zero'))} for zero); ")
              + f"NGB gross {scen_ngb}; T-bill maturities rolled at "
                "constant stock; oil-fund FX at the latest announcement.")
             if len(scen) else ""),
            (f"Bottom-up Tab4×macro (v2): {'SHIP' if bu_report['ship'] else 'NO-SHIP'} — PRE mean MAE "
             f"{bu_report['pre_mean_mae']['bottom_up']} vs production smly "
             f"{bu_report['pre_mean_mae']['production_smly']} on beyond-NB months "
             f"({bu_report['ship_reason']}). cash_requirement_bottomup_bn "
             f"{'replaces' if bu_report['ship'] else 'does not replace'} the live path beyond NB. "
             f"v1 Tab4 smly rejected."),
            (f"Buybacks: median share of the market amount bought back in the last {nxt['days_left']} days before "
             f"maturity, {nxt['n_hist']} bonds since {nxt['hist_from']}: {100 * nxt['share_median']:.0f}% "
             f"(range {100 * nxt['share_min']:.0f}–{100 * nxt['share_max']:.0f}%)." if nxt else ""),
            (f"Central-bank certificates since {cbc_first.index[0]:%b-%Y} drain reserves; total liquidity = structural "
             "liquidity less certificates and F-deposits plus F-loans." if len(cbc_first) else ""),
            (f"Norges Bank's two datasets agree month by month: median gap {contract.fmt_num(float(xchk.abs().median()), 2)}bn; "
             f"{len(xbig)} of {len(xchk)} months above 0.5bn, {len(dj)} of them December/January (31 December is "
             "not a statistics business day)."),
            (f"NOWA − policy rate since 2022: median {contract.fmt_num(float(sp.median()), 2)}bp, sd "
             f"{contract.fmt_num(float(sp.std()), 2)}bp; not charted."),
        ],
    }
    summary["notes"] = [n for n in summary["notes"] if n]
    if write:
        DERIVED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_MONTHLY, index=False)
        daily.reset_index().to_parquet(OUT_DAILY, index=False)
        pd.concat([sd.assign(kind="daily"), sm.rename(columns={"month": "target"}).assign(kind="monthly")]).to_parquet(
            OUT_SCORE, index=False)
        OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return df, summary


if __name__ == "__main__":
    df, s = build()
    print(json.dumps({k: s[k] for k in ("asof", "asof_daily", "identity_resid_bn", "structural_crosscheck_resid_bn",
                                        "forward", "nowa_dropped")}, indent=2))
    for r in s["score"]:
        print(r["window"], r["h"], r["n"], "NB", r["model"], "naive", r["benchmark"], "flat", r["flat"])
