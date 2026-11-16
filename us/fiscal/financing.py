"""DTS net financing reconciled to the auction ladder, and its forward components.

DTS "Public Debt Cash Issues − Redemptions" (Table IIIB) is, exactly,

    public auction cash                           (auction_ladder: total − SOMA)
  + SOMA add-ons at auction                       (non-cash rollovers, booked both sides)
  − every maturity at its at-issue par            (public + SOMA at issue, less buybacks)
  − buyback cash on settlement (T+1)              (par × price; the par leaves the maturity stack)
  + nonmarketable net                             (SLGS, savings bonds, other; Table IIIA)

The public-only ladder therefore misses (SOMA add-ons − SOMA at-issue maturities)
− buybacks + nonmarketables. The SOMA term equals the Fed's holdings maturing that
it did *not* buy at auction (secondary purchases: QE, reserve-management bill
buying) minus runoff: when the Fed rolls a bill it bought in the market,
Treasury pays no cash, which the public ladder counts as a redemption.

Forward, each term is known or rule-based:
  soma      SOMA holdings by CUSIP (NY Fed, weekly) maturing, less SOMA at-issue par,
            less runoff under the QT cap schedule; plus bills the Fed buys after
            the snapshot that mature inside the horizon (trailing purchase pace ×
            SOMA bill maturity profile)
  buyback   trailing 13-week buyback cash pace
  nonmkt    trailing 60-day mean

  PYTHONPATH=. .venv/bin/python us/fiscal/financing.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import auction_ladder as al  # noqa: E402

CACHE = HERE / "data" / "cache"
DERIVED = HERE / "derived"
SOMA_CACHE = CACHE / "soma"
IIIA_URL = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
            "/v1/accounting/dts/public_debt_transactions")
SOMA_API = "https://markets.newyorkfed.org/api/soma"
START = "2022-01-01"
UA = {"User-Agent": "us-fiscal/financing (research; local)"}
NONMKT_WINDOW = 60
BUYBACK_WEEKS = 13
SOMA_PURCHASE_WEEKS = 13
# Monthly cap on Treasury runoff from SOMA ($bn); maturities above the cap are rolled.
QT_CAPS = [("2000-01-01", None), ("2022-06-01", 30.0), ("2022-09-01", 60.0), ("2024-06-01", 25.0),
           ("2025-04-01", 5.0), ("2025-12-01", 0.0)]


# ── data ───────────────────────────────────────────────────────────────────

def fetch_iiia(*, refresh: bool = False) -> pd.DataFrame:
    """Table IIIA public debt transactions by security type (par, $bn), incremental."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "dts_table3a.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if len(old) and not refresh:
        return old
    since = START if old.empty else str((pd.to_datetime(old["record_date"]).max() - pd.Timedelta(days=10)).date())
    rows, page = [], 1
    while True:
        r = requests.get(IIIA_URL, params={
            "filter": f"record_date:gte:{since}", "page[size]": 10000, "page[number]": page,
            "fields": "record_date,transaction_type,security_market,security_type,security_type_desc,"
                      "transaction_today_amt"}, headers=UA, timeout=120)
        r.raise_for_status()
        j = r.json()
        rows += j["data"]
        if page >= j["meta"]["total-pages"]:
            break
        page += 1
    new = pd.DataFrame(rows)
    if len(old):
        old = old[pd.to_datetime(old["record_date"]) < pd.Timestamp(since)]
    df = pd.concat([old, new], ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def nonmarketable_daily(iiia: pd.DataFrame) -> pd.DataFrame:
    """Net cash from nonmarketables (issues − redemptions, $bn); GAS excluded (intragovernmental)."""
    d = iiia.copy()
    d["date"] = pd.to_datetime(d["record_date"])
    d["amt"] = pd.to_numeric(d["transaction_today_amt"], errors="coerce").fillna(0.0) / 1000.0
    d = d[(d["security_market"] == "Nonmarketable") & (d["security_type"] != "Government Account Series")]
    # savings: cash issue price only (interest increments are non-cash)
    d = d[~d["security_type_desc"].astype(str).str.contains("Interest Increment", case=False)]
    d["sign"] = np.where(d["transaction_type"] == "Issues", 1.0, -1.0)
    d["grp"] = np.select([d["security_type"].str.startswith("State"),
                          d["security_type"].str.contains("Savings")], ["slgs", "savings"], "other_nm")
    w = (d["amt"] * d["sign"]).groupby([d["date"], d["grp"]]).sum().unstack(fill_value=0.0)
    for c in ("slgs", "savings", "other_nm"):
        if c not in w:
            w[c] = 0.0
    w["nonmkt"] = w[["slgs", "savings", "other_nm"]].sum(axis=1)
    return w


def fetch_buyback_ops(*, refresh: bool = False) -> pd.DataFrame:
    """Buyback operations: announced maximum, offered and accepted par ($bn), by settlement date."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "buybacks_operations.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    url = IIIA_URL.replace("dts/public_debt_transactions", "od/buybacks_operations")
    rows, page = [], 1
    while True:
        r = requests.get(url, params={"page[size]": 10000, "page[number]": page}, headers=UA, timeout=120)
        r.raise_for_status()
        j = r.json()
        rows += j["data"]
        if page >= j["meta"]["total-pages"]:
            break
        page += 1
    d = pd.DataFrame(rows).replace("null", np.nan)
    out = pd.DataFrame({
        "operation_date": pd.to_datetime(d["operation_date"]),
        "settlement_date": pd.to_datetime(d["settlement_date"]),
        "operation_type": d["operation_type"], "security_type": d["security_type"],
        "maturity_bucket": d["maturity_bucket"],
        "max_bn": pd.to_numeric(d["max_par_amt_redeemed"], errors="coerce") / 1e9,
        "offered_bn": pd.to_numeric(d["total_par_amt_offered"], errors="coerce") / 1e9,
        "accepted_bn": pd.to_numeric(d["total_par_amt_accepted"], errors="coerce") / 1e9,
    })
    out.to_parquet(path, index=False)
    return out


def buyback_cash(details: pd.DataFrame) -> pd.Series:
    """Buyback cash paid on settlement (T+1 Fed business day), $bn."""
    if details is None or details.empty:
        return pd.Series(dtype=float)
    num = lambda s: pd.to_numeric(s.astype(str).replace("null", np.nan), errors="coerce")  # noqa: E731
    d = details.copy()
    d["date"] = pd.to_datetime(d["operation_date"])
    cash = num(d["par_amt_accepted"]) / 1e9 * num(d["weighted_avg_accepted_price"]).fillna(100.0) / 100.0
    settle = al.roll_forward(d["date"] + pd.Timedelta(days=1))
    return cash.groupby(settle).sum().sort_index()


def soma_auction_flows(tidy: pd.DataFrame) -> pd.DataFrame:
    """SOMA add-ons by settlement date and SOMA at-issue par by maturity pay date, $bn."""
    t = tidy[tidy["status"] == "awarded"].copy()
    t["soma_bn"] = t["soma_accepted"].fillna(0.0) / 1e9
    iss = (t["soma_bn"] * t["cash_per_par"]).groupby(t["issue_date"]).sum()
    m = t.groupby("cusip").agg(mat=("maturity_date", "max"), soma=("soma_bn", "sum"))
    m["pay"] = al.roll_forward(m["mat"])
    red = m.groupby("pay")["soma"].sum()
    out = pd.DataFrame({"iss_soma": iss, "red_soma_atissue": red}).fillna(0.0).sort_index()
    out.index.name = "date"
    return out


def soma_asof_dates() -> list[pd.Timestamp]:
    path = CACHE / "soma_asof.json"
    if path.exists() and (not __import__("os").environ.get("SOVDEBT_PULL") == "1" or time.time() - path.stat().st_mtime < 86400):
        return sorted(pd.Timestamp(d) for d in json.loads(path.read_text()))
    r = requests.get(f"{SOMA_API}/asofdates/list.json", headers=UA, timeout=60)
    r.raise_for_status()
    ds = r.json()["soma"]["asOfDates"]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ds))
    return sorted(pd.Timestamp(d) for d in ds)


def soma_snapshot(asof: pd.Timestamp) -> pd.DataFrame:
    """SOMA Treasury holdings by CUSIP as of a Wednesday: cusip, maturity, par_bn, type."""
    SOMA_CACHE.mkdir(parents=True, exist_ok=True)
    path = SOMA_CACHE / f"{asof.date()}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    r = requests.get(f"{SOMA_API}/tsy/get/asof/{asof.date()}.json", headers=UA, timeout=60)
    r.raise_for_status()
    h = pd.DataFrame(r.json()["soma"]["holdings"])
    num = lambda s: pd.to_numeric(s.replace("", np.nan), errors="coerce")  # noqa: E731
    out = pd.DataFrame({
        "cusip": h["cusip"], "maturity": pd.to_datetime(h["maturityDate"]),
        "par_bn": num(h["parValue"]) / 1e9,
        "infl_bn": num(h["inflationCompensation"]).fillna(0.0) / 1e9,
        "type": h["securityType"],
    })
    out.to_parquet(path, index=False)
    return out


def snapshot_for(origin: pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame] | None:
    ds = [d for d in soma_asof_dates() if d <= origin]
    if not ds:
        return None
    return ds[-1], soma_snapshot(ds[-1])


# ── forward components ─────────────────────────────────────────────────────

def qt_cap(month_start: pd.Timestamp) -> float | None:
    cap = None
    for d, c in QT_CAPS:
        if month_start >= pd.Timestamp(d):
            cap = c
    return cap


def soma_net_path(snapshot: pd.DataFrame, tidy: pd.DataFrame, origin: pd.Timestamp,
                  dates: pd.DatetimeIndex, asof: pd.Timestamp | None = None,
                  month_start_snapshot: pd.DataFrame | None = None) -> pd.Series:
    """Cash to Treasury from SOMA maturities after `origin`: rolled holdings not bought at auction.

    At-issue SOMA par counts only auctions settled by the snapshot: later add-ons at
    reopenings raise holdings and at-issue par equally and net to zero.
    """
    asof = origin if asof is None else asof
    s = snapshot.copy()
    s["pay"] = al.roll_forward(s["maturity"])
    s = s[s["pay"] > asof]
    # Earlier maturities in the snapshot's month have already used part of that month's cap.
    m0 = asof.replace(day=1)
    if month_start_snapshot is not None:
        e = month_start_snapshot.copy()
        e["pay"] = al.roll_forward(e["maturity"])
        s = pd.concat([e[(e["pay"] >= m0) & (e["pay"] <= asof)], s], ignore_index=True)
    s = s[(s["pay"] >= m0) & (s["pay"] <= dates[-1])]
    if s.empty:
        return pd.Series(0.0, index=dates)
    t = tidy[(tidy["status"] == "awarded") & (tidy["issue_date"] <= asof)]
    atiss = t.groupby("cusip")["soma_accepted"].sum() / 1e9
    s["soma_atissue"] = s["cusip"].map(atiss).fillna(0.0)
    s["held"] = s["par_bn"] + s["infl_bn"]
    s["runoff"] = 0.0
    for m, g in s.groupby(s["pay"].dt.to_period("M")):
        cap = qt_cap(m.start_time)
        if cap is None:
            continue
        left = cap
        order = g.assign(bill=(g["type"] == "Bills")).sort_values(["bill", "pay"])
        for i, r in order.iterrows():
            take = min(left, r["held"])
            s.at[i, "runoff"] = take
            left -= take
            if left <= 0:
                break
    s["net"] = s["held"] - s["runoff"] - s["soma_atissue"]
    s = s[s["pay"] > origin]
    out = s.groupby("pay")["net"].sum().reindex(dates).fillna(0.0)
    out.attrs["bills"] = s[s["type"] == "Bills"].groupby("pay")["net"].sum().reindex(dates).fillna(0.0)
    return out


def secondary_bill_purchases(asof: pd.Timestamp, tidy: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Fed secondary-market bill purchases over the trailing `weeks`, from week-on-week
    CUSIP holdings net of SOMA auction add-ons: (asof, cusip, bn, rem_bdays at purchase).

    Bill holdings growth alone overstates them: add-ons at auction beyond maturing
    holdings (e.g. agency MBS principal reinvested into bills) are at-issue SOMA and
    pay no cash at maturity.
    """
    ds = [d for d in soma_asof_dates() if asof - pd.Timedelta(weeks=weeks) <= d <= asof]
    t = tidy[tidy["status"] == "awarded"]
    rows = []
    for a, b in zip(ds[:-1], ds[1:]):
        A = soma_snapshot(a)
        B = soma_snapshot(b)
        A = A[A["type"] == "Bills"].set_index("cusip")["par_bn"]
        B = B[B["type"] == "Bills"].set_index("cusip")
        add = t[(t["issue_date"] > a) & (t["issue_date"] <= b)].groupby("cusip")["soma_accepted"].sum() / 1e9
        d = B["par_bn"] - A.reindex(B.index).fillna(0.0) - add.reindex(B.index).fillna(0.0)
        d = d[d > 0.05]
        rem = ((B.loc[d.index, "maturity"] - b).dt.days / 7.0 * 5.0).round().astype(int).clip(lower=1)
        rows += [(b, c, v, r) for c, v, r in zip(d.index, d.to_numpy(), rem.to_numpy())]
    return pd.DataFrame(rows, columns=["asof", "cusip", "bn", "rem_bdays"])


def soma_purchase_path(purchases: pd.DataFrame, weeks: int, dates: pd.DatetimeIndex) -> tuple[pd.Series, float]:
    """Bills the Fed buys after the snapshot that mature inside the window ($bn/day path):
    trailing secondary purchase pace × remaining-maturity-at-purchase profile."""
    pace = float(purchases["bn"].sum()) / (weeks * 5.0) if len(purchases) else 0.0
    if pace == 0.0:
        return pd.Series(0.0, index=dates), pace
    pmf = purchases.groupby("rem_bdays")["bn"].sum()
    pmf = pmf / pmf.sum()
    n = len(dates)
    k = np.zeros(n)
    for lag, w in pmf.items():
        if lag < n:
            k[lag] += w
    out = np.convolve(np.full(n, pace), k)[:n]
    return pd.Series(out, index=dates), pace


def drift_components(origin: pd.Timestamp, dates: pd.DatetimeIndex, *, tidy: pd.DataFrame | None = None,
                     nonmkt: pd.DataFrame | None = None, buybacks: pd.Series | None = None) -> pd.DataFrame:
    """Daily ($bn) forward financing terms the public ladder does not carry, as known at origin."""
    tidy = pd.read_parquet(DERIVED / "auctions_tidy.parquet") if tidy is None else tidy
    if nonmkt is None:
        nonmkt = nonmarketable_daily(fetch_iiia())
    if buybacks is None:
        buybacks = buyback_cash(al.fetch_buyback_details())
    out = pd.DataFrame(index=dates)
    snap = snapshot_for(origin)
    if snap is not None:
        asof, now = snap
        early = snapshot_for(asof.replace(day=1) - pd.Timedelta(days=1))
        sn = soma_net_path(now, tidy, max(origin, asof), dates, asof, None if early is None else early[1])
        out["soma_bn"] = sn.to_numpy()
        out["soma_bills_bn"] = sn.attrs.get("bills", pd.Series(0.0, index=dates)).to_numpy()
        bought = secondary_bill_purchases(asof, tidy, SOMA_PURCHASE_WEEKS)
        purch, pace = soma_purchase_path(bought, SOMA_PURCHASE_WEEKS, dates)
        out["soma_purchases_bn"] = purch.to_numpy()
        out.attrs["soma_purchase_pace_bn_day"] = pace
        out.attrs["soma_asof"] = str(asof.date())
    for c in ("soma_bn", "soma_bills_bn", "soma_purchases_bn"):
        if c not in out:
            out[c] = 0.0
    bb = buybacks[(buybacks.index <= origin) & (buybacks.index > origin - pd.Timedelta(weeks=BUYBACK_WEEKS))]
    out["buyback_bn"] = -float(bb.sum()) / (BUYBACK_WEEKS * 5)
    nm = nonmkt["nonmkt"][nonmkt.index <= origin].tail(NONMKT_WINDOW)
    out["nonmkt_bn"] = float(nm.mean()) if len(nm) else 0.0
    out["drift_bn"] = out[["soma_bn", "soma_purchases_bn", "buyback_bn", "nonmkt_bn"]].sum(axis=1)
    return out


# ── reconciliation ─────────────────────────────────────────────────────────

def reconcile(*, refresh: bool = False) -> pd.DataFrame:
    """Daily DTS net financing vs the full ladder identity; residual should be ~0."""
    panel = pd.read_parquet(DERIVED / "dts_bucket_daily.parquet")
    lad = pd.read_parquet(DERIVED / "auction_ladder.parquet")
    tidy = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    nm = nonmarketable_daily(fetch_iiia(refresh=refresh))
    bb = buyback_cash(al.fetch_buyback_details())
    so = soma_auction_flows(tidy)
    x = pd.DataFrame({"dts_fin_bn": panel["fin_net"] / 1000.0}).loc[START:]
    x = x.join(lad[["net_total"]].rename(columns={"net_total": "public_net_bn"}))
    x = x.join(so).join(bb.rename("buyback_bn")).join(nm[["slgs", "savings", "other_nm", "nonmkt"]]).fillna(0.0)
    x["soma_net_bn"] = x["iss_soma"] - x["red_soma_atissue"]
    x["identity_bn"] = x["public_net_bn"] + x["soma_net_bn"] - x["buyback_bn"] + x["nonmkt"]
    x["resid_bn"] = x["dts_fin_bn"] - x["identity_bn"]
    x["drift_public_bn"] = x["dts_fin_bn"] - x["public_net_bn"]
    x.index.name = "date"
    return x


def run(*, refresh: bool = False, verbose: bool = True) -> dict:
    x = reconcile(refresh=refresh)
    x.to_parquet(DERIVED / "financing_reconcile.parquet")
    q = x.resample("QE")[["drift_public_bn", "soma_net_bn", "buyback_bn", "nonmkt", "resid_bn"]].sum()
    origin = x.index.max()
    dates = pd.bdate_range(origin + pd.Timedelta(days=1), periods=70)
    comp = drift_components(origin, dates)
    summary = {
        "asof": str(origin.date()),
        "resid_daily_sd_bn": float(x["resid_bn"].loc["2024-06":].std()),
        "resid_quarter_max_abs_bn": float(q["resid_bn"].loc["2024":].abs().max()),
        "quarterly": {str(k.date()): {c: round(float(v), 1) for c, v in r.items()} for k, r in q.iterrows()},
        "forward_63bd_bn": {c: round(float(comp[c].iloc[:63].sum()), 1)
                            for c in ("soma_bn", "soma_purchases_bn", "buyback_bn", "nonmkt_bn", "drift_bn")},
        "soma_asof": comp.attrs.get("soma_asof"),
        "soma_purchase_pace_bn_day": comp.attrs.get("soma_purchase_pace_bn_day"),
    }
    (DERIVED / "financing_summary.json").write_text(json.dumps(summary, indent=1))
    if verbose:
        print(q.round(1).loc["2024":].to_string())
        print(json.dumps({k: v for k, v in summary.items() if k != "quarterly"}, indent=1))
    return summary


if __name__ == "__main__":
    run()
