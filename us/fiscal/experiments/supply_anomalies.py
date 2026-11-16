"""Out-of-sample checks of published Treasury supply / calendar anomalies.

A. Auction cycle (Lou, Yan & Zhang 2013): yields rise into a coupon auction and
   fall after. Metric per auction = (y_t - y_{t-5}) - (y_{t+5} - y_t) in bp on the
   auctioned tenor's CMT; a rate trend cancels. Clustered by auction week.
B. Month-end (Hartley & Schwarz 2019; Etula et al. 2020): coupon Treasuries rally
   on the last business days of the month.
C. Bill money premium (Greenwood, Hanson & Stein 2015): realised TGCR over a
   4-week bill's life minus its investment rate, against public bill supply.
D. Turn premia: monthly SOFR-EFFR by calendar month vs reserves, to price SR1-ZQ.

    PYTHONPATH=. .venv/bin/python us/fiscal/experiments/supply_anomalies.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
DERIVED = ROOT / "derived"
OUT = DERIVED / "supply_anomalies.json"

TENORS = {2: "dgs2", 5: "dgs5", 7: "dgs7", 10: "dgs10", 20: "dgs20", 30: "dgs30"}
PERIODS = {
    "1980-2008 (LYZ era)": ("1980-01-01", "2008-12-31"),
    "2009-2019": ("2009-01-01", "2019-12-31"),
    "2020-2026 (holdout)": ("2020-01-01", "2026-12-31"),
    "2023-2026": ("2023-01-01", "2026-12-31"),
}


def _fred(name: str) -> pd.Series:
    d = pd.read_csv(CACHE / f"fred_{name}.csv", parse_dates=["date"])
    s = pd.to_numeric(d["value"], errors="coerce")
    return pd.Series(s.values, index=d["date"]).dropna()


def _tstat_mean(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def auction_cycle() -> dict:
    a = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    a = a[(a["cls"] == "coupon") & (a["status"] != "announced")].copy()
    yrs = (pd.to_datetime(a["maturity_date"]) - pd.to_datetime(a["issue_date"])).dt.days / 365.25
    a["tenor"] = yrs.round().astype(int)
    a = a[a["tenor"].isin(TENORS)]
    curves = {t: _fred(n) * 100 for t, n in TENORS.items()}
    rows = []
    for _, r in a.iterrows():
        y = curves[r["tenor"]]
        d = pd.Timestamp(r["auction_date"])
        if d not in y.index:
            continue
        i = y.index.get_loc(d)
        if i < 5 or i + 5 >= len(y):
            continue
        pre = y.iloc[i] - y.iloc[i - 5]
        post = y.iloc[i + 5] - y.iloc[i]
        rows.append({"date": d, "tenor": r["tenor"], "pre": pre, "post": post,
                     "cycle": pre - post, "week": d.to_period("W").start_time})
    ev = pd.DataFrame(rows).drop_duplicates(["date", "tenor"])
    res = {}
    for name, (s, e) in PERIODS.items():
        sub = ev[(ev["date"] >= s) & (ev["date"] <= e)]
        wk = sub.groupby("week")["cycle"].mean()
        m, t, n = _tstat_mean(wk)
        by_t = {}
        for ten, g in sub.groupby("tenor"):
            mm, tt, nn = _tstat_mean(g["cycle"])
            pm, pt, _ = _tstat_mean(g["pre"])
            by_t[str(ten)] = {"cycle_bp": round(mm, 2), "t": round(tt, 2), "n": nn,
                              "pre_bp": round(pm, 2), "pre_t": round(pt, 2)}
        sd = wk.std(ddof=1)
        res[name] = {"pooled_week_cycle_bp": round(m, 2), "t": round(t, 2), "weeks": n,
                     "ann_sharpe_weekly_trade": round(m / sd * np.sqrt(n / max(1, (pd.Timestamp(e).year - pd.Timestamp(s).year + 1))), 2) if sd > 0 else None,
                     "by_tenor": by_t}
    ev2 = ev[ev["date"] >= "2020-01-01"].copy()
    ev2["year"] = ev2["date"].dt.year
    res["holdout_by_year"] = {str(y): round(g.groupby("week")["cycle"].mean().mean(), 2)
                              for y, g in ev2.groupby("year")}
    return res


def month_end() -> dict:
    res = {}
    for ten in (5, 10, 30):
        y = _fred(TENORS[ten]) * 100
        dy = y.diff().dropna()
        idx = dy.index
        mon = idx.to_period("M")
        last = pd.Series(idx, index=idx).groupby(mon).transform("max")
        rank = pd.Series(np.arange(len(idx)), index=idx)
        pos_last = rank.groupby(mon).transform("max")
        k = (pos_last - rank).astype(int)
        out = {}
        for name, (s, e) in {"1990-2018 (HS era)": ("1990-01-01", "2018-12-31"),
                             "2019-2026 (holdout)": ("2019-01-01", "2026-12-31"),
                             "2023-2026": ("2023-01-01", "2026-12-31")}.items():
            msk = (idx >= s) & (idx <= e)
            cell = {}
            for lab, sel in {"T": k == 0, "T-1": k == 1, "T-2": k == 2, "T-3..T-1": k.between(1, 3),
                             "T-2..T": k.between(0, 2), "other": k > 3}.items():
                x = dy[msk & sel.values]
                if lab in ("T-3..T-1", "T-2..T"):
                    x = x.groupby(x.index.to_period("M")).sum()
                m, t, n = _tstat_mean(x)
                cell[lab] = {"bp": round(m, 2), "t": round(t, 2), "n": n}
            out[name] = cell
        res[f"{ten}y"] = out
    return res


def _bill_stock() -> pd.Series:
    a = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    b = a[a["cls"].isin(["bill", "cmb"]) & a["total_accepted"].notna()].copy()
    b["iss"] = pd.to_datetime(b["issue_date"])
    b["mat"] = pd.to_datetime(b["maturity_date"])
    b["amt"] = b["total_accepted"] / 1e9
    days = pd.date_range("2017-01-01", pd.Timestamp.today().normalize(), freq="D")
    add = b.groupby("iss")["amt"].sum().reindex(days, fill_value=0)
    sub = b.groupby("mat")["amt"].sum().reindex(days, fill_value=0)
    base = b[(b["iss"] < days[0]) & (b["mat"] >= days[0])]["amt"].sum()
    return base + (add - sub).cumsum()


def bill_premium() -> dict:
    a = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    b = a[(a["cls"] == "bill") & (a["security_term"] == "4-Week")].copy()
    b["rate"] = pd.to_numeric(b["high_investment_rate"], errors="coerce")
    b = b.dropna(subset=["rate"])
    b["iss"] = pd.to_datetime(b["issue_date"])
    b["mat"] = pd.to_datetime(b["maturity_date"])
    b = b[b["iss"] >= "2018-04-05"]
    tg = pd.read_csv(CACHE / "nyfed_full_tgcr.csv", parse_dates=["date"]).set_index("date")["tgcr"]
    tg = tg.reindex(pd.date_range(tg.index.min(), tg.index.max(), freq="D")).ffill()
    stock = _bill_stock()
    rows = []
    for _, r in b.iterrows():
        w = tg.loc[r["iss"]: r["mat"] - pd.Timedelta(days=1)]
        if len(w) < 25 or r["mat"] > tg.index.max():
            continue
        mm = r["rate"] * 360 / 365
        rows.append({"iss": r["iss"], "prem_bp": (w.mean() - mm) * 100,
                     "lstock": np.log(stock.get(r["iss"], np.nan))})
    p = pd.DataFrame(rows).set_index("iss").sort_index()
    p["d4_prem"] = p["prem_bp"].shift(-4) - p["prem_bp"]
    p["d4_lstock_fwd"] = p["lstock"].shift(-4) - p["lstock"]
    p["d4_lstock_past"] = p["lstock"] - p["lstock"].shift(4)
    p["month"] = p.index.month
    debt_limit = (((p.index >= "2023-01-19") & (p.index <= "2023-06-30"))
                  | ((p.index >= "2025-01-02") & (p.index <= "2025-07-31"))
                  | ((p.index >= "2021-08-01") & (p.index <= "2021-12-31")))
    out = {"n": int(len(p)), "prem_mean_bp": round(p["prem_bp"].mean(), 1),
           "prem_last_bp": round(p["prem_bp"].iloc[-1], 1), "last_iss": str(p.index[-1].date())}
    for lab, sub in {"all": p, "ex_debt_limit": p[~debt_limit]}.items():
        cell = {}
        for x in ("d4_lstock_fwd", "d4_lstock_past"):
            d = sub[["d4_prem", x, "month"]].dropna()
            X = pd.get_dummies(d["month"], prefix="m", drop_first=True, dtype=float)
            X[x] = d[x]
            X = sm.add_constant(X)
            f = sm.OLS(d["d4_prem"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            cell[x] = {"bp_per_10pct": round(f.params[x] * 0.1, 2), "t": round(f.tvalues[x], 2),
                       "n": int(f.nobs), "r2": round(f.rsquared, 3)}
        out[lab] = cell
    p.to_csv(DERIVED / "bill_premium_4w.csv")
    return out


def turn_premia() -> dict:
    d = pd.read_parquet(DERIVED / "money_markets_daily.parquet")
    d.index = pd.to_datetime(d.index)
    m = d[["sofr_effr_bp", "sofr_iorb_bp", "reserves_bn"]].resample("ME").mean()
    m["mon"] = m.index.month
    out = {}
    for mon in (12, 1, 3, 6, 9):
        g = m[m["mon"] == mon]
        out[str(mon)] = {str(i.year): {"serff": round(r.sofr_effr_bp, 1), "sofr_iorb": round(r.sofr_iorb_bp, 1),
                                       "res": round(r.reserves_bn)} for i, r in g.iterrows()}
    m["excess"] = m["sofr_effr_bp"] - m["sofr_effr_bp"].rolling(3, center=True).median()
    out["dec_minus_neighbours"] = {str(i.year): round(v, 1) for i, v in m.loc[m["mon"] == 12, "excess"].items()}
    out["jan_minus_neighbours"] = {str(i.year): round(v, 1) for i, v in m.loc[m["mon"] == 1, "excess"].items()}
    return out


def main() -> None:
    res = {"auction_cycle": auction_cycle(), "month_end": month_end(),
           "bill_premium": bill_premium(), "turn_premia": turn_premia()}
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
