"""Twelve-month fiscal path: operating flows, financing mix, TGA, reserves, scenarios.

Operating flows extend `flow_model` (calendar shape × trailing level per bucket) to
250 business days. At that range the trailing 250-day level is half a year stale,
so a growth variant lifts each bucket's level by its trailing year-on-year growth;
which one ships is chosen on PRE only.

Interest: the coupon schedule of securities outstanding at the origin, plus
coupons on projected issuance (year-ago settlement template, latest coupon rate
for the same class × tenor). Without the second term interest falls through the
year as maturing notes stop paying and their replacements pay nothing.

Financing: coupons repeat the year-ago template (scenario: scaled from the next
refunding), bills are steered to the refunding TGA target then follow a fitted
total-bills reaction (seasonal gap + rolling ΔTGA; tenor shares constant),
drift from `financing.drift_components`.

Backtest: month-end origins, cumulative net operating flow MAE at h = 21/63/126/250
business days. PRE = origins 2015-01→2019-02 and 2021-07→2022-12 (horizons clear
of the Covid exclusion), OOS = 2023-01→.

  PYTHONPATH=. .venv/bin/python us/fiscal/fiscal_path.py --backtest
  PYTHONPATH=. .venv/bin/python us/fiscal/fiscal_path.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import auction_ladder as al  # noqa: E402
import financing  # noqa: E402
import flow_model as fm  # noqa: E402
import tga_forecast as tf  # noqa: E402

DERIVED = HERE / "derived"
H_LONG = 500
H_YEAR = 250
# Live path runs through the debt table's last year. The scored backtest stays on H_LONG.
# The components year-overlay keeps a full FY2030 (ends 30 Sep 2030), past that date.
FORWARD_END = pd.Timestamp("2029-12-31")
CHART_END = pd.Timestamp("2030-09-30")
LONG_HORIZONS = (21, 63, 126, 250, 500)
CHOICE_HORIZONS = (21, 63, 126, 250)
GROWTH_CLIP = (-0.15, 0.25)
PRE_WINDOWS = (("2015-01-01", "2019-02-28"), ("2021-07-01", "2022-12-31"))
OOS_START = "2023-01-01"
VARIANTS = ("model", "model_g", "last_year", "flat")
COUPON_SCENARIOS = {"unchanged": 0.0, "plus5pct": 0.05, "plus10pct": 0.10}
# First refunding at which coupon sizes could change (Treasury guidance: unchanged
# "for at least the next several quarters" as of Aug-2026).
SCENARIO_START = pd.Timestamp("2027-02-01")


# ── interest on projected issuance ─────────────────────────────────────────

def latest_coupon_rates(auc: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Coupon rate (decimal) of the latest original issue per class × tenor known at origin."""
    a = auc[(auc["status"] == "awarded") & ~auc["cls"].isin(tf.BILL_CLS) & (auc["issue_date"] <= origin)].copy()
    a = a[a["reopening"].astype(str).str.lower() != "yes"]
    a["key"] = tf._coupon_key(a)
    a["rate"] = pd.to_numeric(a["int_rate"], errors="coerce") / 100.0
    return a.dropna(subset=["rate"]).sort_values("issue_date").groupby("key")["rate"].last()


def coupon_schedule_asof(auc: pd.DataFrame, origin: pd.Timestamp, end: pd.Timestamp,
                         scale: float = 0.0) -> pd.Series:
    """Coupon cash ($bn by pay date) from securities issued by `origin` plus projected issuance."""
    known = auc[auc["issue_date"] <= origin]
    base = al.coupon_schedule(known, end=end)
    proj = tf.projected_coupons(auc, origin, end)
    if proj.empty:
        return base
    rates = latest_coupon_rates(auc, origin)
    proj = proj.assign(rate=proj["key"].map(rates))
    proj = proj[proj["rate"].notna() & ~proj["key"].str.startswith("frn")]
    if scale:
        proj.loc[proj["date"] >= SCENARIO_START, "bn"] *= 1.0 + scale
    rows = []
    for _, r in proj.iterrows():
        k = 1
        while (d := r["date"] + pd.DateOffset(months=6 * k)) <= end:
            rows.append((d, r["bn"] * r["rate"] / 2.0))
            k += 1
    if not rows:
        return base
    extra = pd.DataFrame(rows, columns=["date", "bn"])
    extra["date"] = al.roll_forward(extra["date"])
    return base.add(extra.groupby("date")["bn"].sum(), fill_value=0.0).sort_index().rename("coupon_bn")


# ── long-horizon operating flows ───────────────────────────────────────────

def bucket_growth(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    hist = panel[panel.index <= origin]
    now = hist.tail(fm.LEVEL_WINDOW).mean()
    ago = hist[hist.index <= origin - pd.DateOffset(years=1)].tail(fm.LEVEL_WINDOW).mean()
    g = (now / ago.where(ago.abs() > 1e-6) - 1.0).fillna(0.0)
    return g.clip(*GROWTH_CLIP)


def long_forecast(models, panel, coupon, origin, dates, growth: bool, fy: bool = False) -> pd.DataFrame:
    fc = fm.forecast(models, panel, coupon, origin, dates)
    if fy:
        # Annual profile (COLA on the indexed lines, trailing growth on the rest),
        # then the boosted shape spreads that fiscal year. Not used in the backtest.
        from fy_overlay import apply_fy
        return apply_fy(fc, panel, origin)
    if not growth:
        return fc
    g = bucket_growth(panel, origin)
    # Trailing level is centred ~125 bdays before the origin.
    k = (np.arange(1, len(dates) + 1) + fm.LEVEL_WINDOW / 2) / fm.LEVEL_WINDOW
    for b in fc.columns:
        if b != "wd_interest":
            fc[b] = fc[b].to_numpy() * (1.0 + g.get(b, 0.0)) ** k
    return fc


def _window(o: pd.Timestamp) -> str | None:
    if any(pd.Timestamp(a) <= o <= pd.Timestamp(b) for a, b in PRE_WINDOWS):
        return "PRE"
    if o >= pd.Timestamp(OOS_START):
        return "OOS"
    return None


def long_backtest(verbose: bool = True) -> pd.DataFrame:
    panel, _ = fm.load_inputs()
    auc = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    feats = fm.calendar_features(panel.index)
    actual = fm._net(panel)
    months = pd.period_range(PRE_WINDOWS[0][0], panel.index.max(), freq="M")
    rows = []
    for p in months:
        o = panel.index[panel.index <= p.end_time]
        if len(o) == 0:
            continue
        o = o[-1]
        w = _window(o)
        fut = panel.index[panel.index > o][:H_LONG]
        if w is None or len(fut) < LONG_HORIZONS[0]:
            continue
        models = fm.fit_all(panel, feats, o)
        coupon = coupon_schedule_asof(auc, o, fut[-1] + pd.Timedelta(days=5))
        out = {"model": fm._net(long_forecast(models, panel, coupon, o, fut, growth=False)),
               "model_g": fm._net(long_forecast(models, panel, coupon, o, fut, growth=True)),
               "last_year": fm.bench_last_year(panel, o, fut),
               "flat": fm.bench_flat(panel, o, fut)}
        cum_a = actual.loc[fut].cumsum()
        lo, hi = fm.TRAIN_EXCLUDE
        for h in LONG_HORIZONS:
            if h > len(fut) or ((fut[:h] >= lo) & (fut[:h] <= hi)).any():
                continue
            rows.append({"origin": o, "window": w, "h": h, "actual": cum_a.iloc[h - 1] / 1000.0,
                         **{v: out[v].iloc[:h].sum() / 1000.0 for v in VARIANTS}})
        if verbose and p.month == 12:
            print(f"   long backtest through {o.date()}")
    bt = pd.DataFrame(rows)
    bt.to_parquet(DERIVED / "fiscal_path_backtest.parquet", index=False)
    return bt


def score_long(bt: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    s = bt.groupby(["window", "h"]).apply(lambda x: pd.Series(
        {"n": len(x), **{v: (x[v] - x["actual"]).abs().mean() for v in VARIANTS},
         **{f"bias_{v}": (x[v] - x["actual"]).mean() for v in ("model", "model_g")}}), include_groups=False)
    pre = s.loc["PRE"].loc[list(CHOICE_HORIZONS)]
    choice = "model_g" if pre["model_g"].mean() < pre["model"].mean() else "model"
    return s, choice


# ── forward ────────────────────────────────────────────────────────────────

def _totals(p: pd.DataFrame) -> dict:
    nb, nc = float(p["net_bills_total_bn"].sum()), float(p["net_coupons"].sum())
    return {"op_flow_bn": float(p["op_flow_bn"].sum()), "net_coupons_bn": nc, "net_bills_bn": nb,
            "private_net_bills_bn": float(p["private_net_bills_bn"].sum()),
            "interest_bn": float(p["interest_bn"].sum()), "bill_share": nb / (nb + nc),
            "reserves_end_bn": float(p["reserves_bn"].iloc[-1]), "reserves_min_bn": float(p["reserves_bn"].min()),
            "end": str(p.index[-1].date())}


def forward(choice: str = "model", verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    panel, _ = fm.load_inputs()
    auc = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    feats = fm.calendar_features(panel.index)
    origin = panel.index.max()
    dates = fm.bdays(origin + pd.Timedelta(days=1), FORWARD_END)
    models = fm.fit_all(panel, feats, origin)
    bb = al.fetch_buyback_details()
    comp = financing.drift_components(origin, dates, tidy=auc, buybacks=financing.buyback_cash(bb))
    from tga_deficit import fetch_tga_mn

    tga0 = float(fetch_tga_mn().loc[:origin].iloc[-1] / 1000.0)
    tgt = tf._next_target(origin)
    t_date, t_val = tgt if tgt is not None else (tf._quarter_end_after(origin), tga0)
    tga_hist = fetch_tga_mn().loc[:origin] / 1000.0
    lad = pd.read_parquet(DERIVED / "auction_ladder.parquet")
    reaction = tf.fit_bill_reaction(tga_hist, lad["net_bills"])
    shares = tf.bill_tenor_shares(auc, origin)

    mm = pd.read_parquet(DERIVED / "money_markets_daily.parquet")
    res0 = float(mm.loc[:origin, "reserves_bn"].dropna().iloc[-1])
    wk = mm["fed_assets_bn"].dropna().resample("W-WED").last().dropna()
    fed_trend = float((wk.iloc[-1] - wk.iloc[-1 - tf.FED_ASSET_WEEKS]) / (tf.FED_ASSET_WEEKS * 5))

    import reserve_demand as rd

    mmp = rd.load_panel(rd.PRODUCTION["scale"])
    curve = rd.Curve().fit(mmp)
    last = mmp.iloc[-1]

    paths, scen = {}, {}
    for name, scale in COUPON_SCENARIOS.items():
        coupon = coupon_schedule_asof(auc, origin, dates[-1] + pd.Timedelta(days=5), scale)
        fc = long_forecast(models, panel, coupon, origin, dates, growth=False, fy=True)
        if scale == 0.0:
            # Components year-overlay through FY2030. The TGA path stays on FORWARD_END.
            bucket_dates = fm.bdays(origin + pd.Timedelta(days=1), max(dates[-1], CHART_END))
            coupon_b = coupon_schedule_asof(auc, origin, bucket_dates[-1] + pd.Timedelta(days=5), 0.0)
            fc_b = long_forecast(models, panel, coupon_b, origin, bucket_dates, growth=False, fy=True)
            (fc_b / 1000.0).to_parquet(DERIVED / "fiscal_path_buckets.parquet")
        op = fm._net(fc) / 1000.0
        fin = tf.financing_path(auc, origin, dates, bb)
        if scale:
            proj = fin["coupon_iss"].where(dates.to_series(index=dates) >= SCENARIO_START, 0.0)
            fin["coupon_iss"] = fin["coupon_iss"] + scale * proj
            fin["net_coupons"] = fin["coupon_iss"] - fin["coupon_red"]
            fin["net_total"] = fin["net_bills"] + fin["net_coupons"]
        p = pd.DataFrame(index=dates)
        p["op_flow_bn"] = op.to_numpy()
        p["interest_bn"] = coupon.reindex(dates).fillna(0.0).to_numpy()
        # Cash interest inside the operating deficit: coupon schedule scaled by the
        # realised ratio, plus the off-coupon-day mean (bills). $bn.
        p["interest_cash_bn"] = fc["wd_interest"].to_numpy() / 1000.0
        for c in ("net_bills", "coupon_iss", "coupon_red", "net_coupons", "known"):
            p[c] = fin[c].to_numpy()
        for c in tf.DRIFT_COLS + ("soma_bills_bn",):
            p[c] = comp[c].to_numpy()
        p["tga_rollover_bn"] = tga0 + (p["op_flow_bn"] + p["net_bills"] + p["net_coupons"] + p["drift_bn"]).cumsum()
        p["implied_bills_bn"] = tf.steer_bills(
            p["tga_rollover_bn"], p["known"], t_date, t_val,
            tga_hist=tga_hist, reaction=reaction,
        ).to_numpy()
        p["tga_bn"] = p["tga_rollover_bn"] + p["implied_bills_bn"].cumsum()
        by_tenor = tf.split_bills_by_tenor(p["implied_bills_bn"], shares)
        for c in by_tenor.columns:
            p[c] = by_tenor[c].to_numpy()
        p["net_bills_total_bn"] = p["net_bills"] + p["implied_bills_bn"]
        pace = comp.attrs.get("soma_purchase_pace_bn_day", 0.0)
        # Fed-held bills that mature are rolled as add-ons, not repaid to the public:
        # snapshot holdings (`soma_bills_bn`) and later purchases (`soma_purchases_bn`) alike.
        p["fed_bill_purchases_bn"] = pace
        p["private_net_bills_bn"] = (p["net_bills_total_bn"] + p["soma_bills_bn"] + p["soma_purchases_bn"]
                                     - p["fed_bill_purchases_bn"])
        steps = np.arange(1, len(dates) + 1)
        p["reserves_bn"] = res0 - (p["tga_bn"] - tga0) + fed_trend * steps
        fut = rd.calendar_dummies(dates)
        fut["floor"], fut["cap"] = last["floor"], last["cap"]
        fut["coupon_settle"] = (p["net_coupons"] / 100.0).to_numpy()
        fut["x"] = 100.0 * p["reserves_bn"].to_numpy() / last["scale_bn"]
        p["sofr_iorb_bp"] = curve.predict(fut)
        p["sofr_iorb_excal_bp"] = curve.predict(fut, calendar=False)
        p.index.name = "date"
        paths[name] = p
        q = p.groupby(p.index.to_period("Q")).agg(
            op_flow_bn=("op_flow_bn", "sum"), net_coupons_bn=("net_coupons", "sum"),
            net_bills_bn=("net_bills_total_bn", "sum"), private_net_bills_bn=("private_net_bills_bn", "sum"),
            drift_bn=("drift_bn", "sum"), tga_end_bn=("tga_bn", "last"), reserves_end_bn=("reserves_bn", "last"),
            sofr_iorb_excal_avg_bp=("sofr_iorb_excal_bp", "mean"))
        scen[name] = {
            "quarters": {str(k): {c: round(float(v), 1) for c, v in r.items()}
                         for k, r in q.iterrows()},
            "total_12m": _totals(p.iloc[:H_YEAR]),
            "total_24m": _totals(p.iloc[:min(H_LONG, len(p))]),
            "total_path": _totals(p),
        }
    pd.concat(paths, names=["scenario"]).to_parquet(DERIVED / "fiscal_path.parquet")
    info = {"origin": str(origin.date()), "end": str(dates[-1].date()), "flow_variant": choice,
            "tga_origin_bn": tga0, "target_date": str(t_date.date()), "target_bn": t_val,
            "fed_assets_trend_bn_per_bday": fed_trend, "scenario_start": str(SCENARIO_START.date()),
            "bill_reaction": reaction,
            "bill_tenor_shares": {k: float(shares[k]) for k in shares.index},
            "scenarios": scen}
    (DERIVED / "fiscal_path.json").write_text(json.dumps(info, indent=1, default=float))
    if verbose:
        for name, s in scen.items():
            for span in ("total_12m", "total_24m", "total_path"):
                print(name, span, {k: v if isinstance(v, str) else round(v, 3 if k == "bill_share" else 1)
                                   for k, v in s[span].items()})
        print(pd.DataFrame(scen["unchanged"]["quarters"]).T.to_string())
    return paths["unchanged"], info


def run(backtest: bool = False, verbose: bool = True) -> dict:
    score_path = DERIVED / "fiscal_path_score.json"
    if backtest or not score_path.exists():
        bt = long_backtest(verbose)
        s, choice = score_long(bt)
        score_path.write_text(json.dumps({"choice": choice, "pre_windows": PRE_WINDOWS, "oos_start": OOS_START,
                                          "score": s.reset_index().to_dict(orient="records")}, indent=1, default=float))
        if verbose:
            print(s.round(1).to_string())
            print("PRE choice:", choice)
    choice = json.loads(score_path.read_text())["choice"]
    _, info = forward(choice, verbose)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    run(ap.parse_args().backtest)
