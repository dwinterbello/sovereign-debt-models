"""Refunding supply news and the ACM term premium.

Each quarterly refunding statement (Feb/May/Aug/Nov, the announcement date of the
new 10y) can change nominal coupon sizes. The revision to twelve-month coupon
supply is valued in DV01:

  rev_t = Σ_key (size after − size before) × auctions per year × DV01 per $bn

`size after` is the last auction of that key within 95 days of the statement, so a
quarter of stepped increases (Aug-2023: +$3bn a month on 2y–7y) counts at its end
level; `size before` the last auction before the statement. DV01 per $bn par =
0.1 × modified duration of a par bond at the latest coupon ($mn per bp).

Event study: ΔTP on rev, OLS with HC1 errors, over the statement window (close of
t−1 to close of t+1) and the refunding week (from the Friday close, so Monday's
financing estimates are in); placebo with every window moved 10 business days earlier.
Refundings with no size change stay in as controls. Treasury signals changes a
quarter ahead in its guidance language, so part of each revision is priced
before the statement — read the coefficient as a lower bound on the supply effect.

  PYTHONPATH=. .venv/bin/python us/fiscal/term_premium.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CACHE = HERE / "data" / "cache"
DERIVED = HERE / "derived"
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
ACM_CACHE = CACHE / "acm_daily.parquet"
START = "1998-01-01"
NOMINAL_TENORS = (2, 3, 5, 7, 10, 20, 30)
AFTER_DAYS = 95
PLACEBO_SHIFT = 10
TP_COLS = ("ACMTP02", "ACMTP05", "ACMTP10")
UA = {"User-Agent": "Mozilla/5.0 (us-fiscal research)"}


def fetch_acm(*, refresh: bool = False) -> pd.DataFrame:
    if ACM_CACHE.exists() and not refresh:
        return pd.read_parquet(ACM_CACHE)
    import io

    import requests

    r = requests.get(ACM_URL, headers=UA, timeout=60)
    r.raise_for_status()
    d = pd.read_excel(io.BytesIO(r.content), sheet_name="ACM Daily")
    d["date"] = pd.to_datetime(d["DATE"], format="%d-%b-%Y")
    keep = ["ACMY02", "ACMY10", *TP_COLS]
    d = d.set_index("date")[keep].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(ACM_CACHE)
    return d


def _mod_duration(years: float, y: float) -> float:
    """Modified duration of a semi-annual par bond."""
    if y <= 0:
        return years
    h = y / 2.0
    return (1.0 - (1.0 + h) ** (-2.0 * years)) / y


def statements(tidy: pd.DataFrame) -> pd.Series:
    """Refunding statement dates: announcement of the new 10y in Feb/May/Aug/Nov."""
    t = tidy[(tidy["cls"] == "coupon") & (tidy["status"] == "awarded")]
    yrs = ((t["maturity_date"] - t["issue_date"]).dt.days / 365.25).round()
    x = t[(yrs == 10) & (t["reopening"].astype(str).str.lower() == "no")
          & t["issue_date"].dt.month.isin([2, 5, 8, 11]) & (t["issue_date"] >= START)]
    return pd.Series(sorted(pd.to_datetime(x["announcemt_date"]).dropna().unique()), name="statement")


def revisions(tidy: pd.DataFrame | None = None) -> pd.DataFrame:
    t = pd.read_parquet(DERIVED / "auctions_tidy.parquet") if tidy is None else tidy
    c = t[(t["cls"] == "coupon") & (t["status"] == "awarded")].copy()
    c["tenor"] = ((c["maturity_date"] - c["issue_date"]).dt.days / 365.25).round().astype(int)
    c = c[c["tenor"].isin(NOMINAL_TENORS)]
    c["size_bn"] = pd.to_numeric(c["offering_amt"], errors="coerce") / 1e9
    c["y"] = pd.to_numeric(c["int_rate"], errors="coerce") / 100.0
    rows = []
    for s in statements(t):
        rec = {"statement": s, "rev_dv01_mn": 0.0, "rev_face_bn": 0.0}
        for k, g in c.groupby("tenor"):
            before = g[g["auction_date"] < s]
            after = g[(g["auction_date"] > s) & (g["auction_date"] <= s + pd.Timedelta(days=AFTER_DAYS))]
            if before.empty or after.empty:
                continue
            new = after.sort_values("auction_date").iloc[-1]
            # New issues and reopenings run at different sizes: compare like with like.
            same = before[(before["reopening"].astype(str).str.lower() == str(new["reopening"]).lower())
                          & (before["auction_date"] > s - pd.Timedelta(days=190))]
            if same.empty:
                continue
            last = same.sort_values("auction_date").iloc[-1]
            n_year = int((before["auction_date"] > s - pd.Timedelta(days=365)).sum())
            d = float(new["size_bn"] - last["size_bn"])
            if not np.isfinite(d) or n_year == 0:
                continue
            dv = 0.1 * _mod_duration(k, float(last["y"]) if pd.notna(last["y"]) else 0.03)
            rec[f"d_{k}y_bn"] = d
            rec["rev_face_bn"] += d * n_year
            rec["rev_dv01_mn"] += d * n_year * dv
        rows.append(rec)
    return pd.DataFrame(rows).fillna(0.0)


WINDOWS = {"statement": 1, "refunding_week": 3}


def event_window(acm: pd.DataFrame, dates: pd.Series, col: str, shift: int = 0, pre: int = 1) -> pd.Series:
    """Close of t−pre to close of t+1 (business days in the ACM index), optionally shifted earlier."""
    idx = acm.index
    out = {}
    s = acm[col]
    for d in dates:
        i = idx.searchsorted(d)
        i0, i1 = i - pre - shift, i + 1 - shift
        if i0 < 0 or i1 >= len(idx):
            continue
        out[d] = 100.0 * (s.iloc[i1] - s.iloc[i0])
    return pd.Series(out, name=col)


def study(ev: pd.DataFrame, acm: pd.DataFrame) -> dict:
    res = {}
    for col, (wname, pre), (label, shift) in product(TP_COLS, WINDOWS.items(),
                                                     (("event", 0), ("placebo", PLACEBO_SHIFT))):
        y = event_window(acm, ev["statement"], col, shift, pre)
        d = ev.set_index("statement").join(y, how="inner").dropna(subset=[col])
        m = sm.OLS(d[col], sm.add_constant(d["rev_dv01_mn"] / 100.0)).fit(cov_type="HC1")
        up, dn = d[d["rev_dv01_mn"] > 1e-6], d[d["rev_dv01_mn"] < -1e-6]
        res[f"{col}_{wname}_{label}"] = {
            "n": int(m.nobs), "n_revisions": int(len(up) + len(dn)),
            "bp_per_100mn_dv01": float(m.params.iloc[1]), "t": float(m.tvalues.iloc[1]), "r2": float(m.rsquared),
            "mean_bp_up": float(up[col].mean()) if len(up) else None,
            "mean_bp_down": float(dn[col].mean()) if len(dn) else None,
            "mean_bp_none": float(d.loc[d["rev_dv01_mn"].abs() <= 1e-6, col].mean()),
        }
    return res


def run(*, refresh: bool = False, verbose: bool = True) -> dict:
    acm = fetch_acm(refresh=refresh)
    ev = revisions()
    res = study(ev, acm)
    ev.to_parquet(DERIVED / "refunding_revisions.parquet", index=False)
    tp = acm.loc[acm.index >= START, list(TP_COLS)]
    out = {"acm_end": str(acm.index.max().date()), "start": START, "after_days": AFTER_DAYS,
           "placebo_shift_bdays": PLACEBO_SHIFT, "results": res,
           "tp10_now": float(tp["ACMTP10"].dropna().iloc[-1]),
           "revisions": [{"statement": str(r["statement"].date()), "rev_dv01_mn": round(float(r["rev_dv01_mn"]), 1),
                          "rev_face_bn": round(float(r["rev_face_bn"]), 0)}
                         for _, r in ev[ev["rev_dv01_mn"].abs() > 1e-6].iterrows()]}
    (DERIVED / "term_premium.json").write_text(json.dumps(out, indent=1, default=float))
    if verbose:
        print(pd.DataFrame(res).T.round(3).to_string())
        print(pd.DataFrame(out["revisions"]).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    run(refresh=ap.parse_args().refresh)
