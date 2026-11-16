"""Forward TGA, reserves and SOFR−IORB path; implied net bill issuance to the TGA target.

    TGA_{t+h} = TGA_t + Σ (operating flow + net marketable financing + drift)

operating flow   flow_model (calendar × level, per Table II bucket)
financing        known window: every auction announced by the origin (public
                 cash, SOMA excluded) less maturities of everything outstanding
                 or announced; beyond it, coupons repeat the year-ago settlement
                 calendar (sizes have been flat since 2024) and bills either
                 roll at maturity size ('rollover') or are sized so the TGA
                 lands on Treasury's quarter-end target, then follow a fitted
                 total-bills reaction (seasonal gap + rolling ΔTGA) instead of
                 a flat hold; tenor split = trailing gross shares held constant
drift            trailing mean of DTS financing − auction ladder (Fed secondary
                 bill purchases / runoff, savings bonds, SLGS)
reserves         R_t − ΔTGA − ΔON RRP + ΔFed assets (trailing 13-week trend)
spread           reserve_demand curve + calendar + decayed residual

The implied net bill figure is (target − rollover TGA at the target date):
what bill supply must do for the TGA to land there, given the deficit path.

Backtest: weekly origins 2024+, operating flows from `flow_backtest.parquet`
and financing exactly as known at the origin (announcement dates), bills on
rollover; scored on TGA level at h = 5/10/20 against a flat TGA.

  PYTHONPATH=. .venv/bin/python us/fiscal/tga_forecast.py --backtest
  PYTHONPATH=. .venv/bin/python us/fiscal/tga_forecast.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import auction_ladder as al
import financing
import flow_model as fm

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

# Treasury's announced end-of-quarter cash balance assumptions ($bn), from the
# quarterly refunding financing estimates (Q3-2026 TBAC). Update each refunding (early Feb/May/Aug/Nov).
TGA_TARGETS = {
    "2026-09-30": 950.0,
    "2026-12-31": 850.0,
}
DRIFT_WINDOW = 60
DRIFT_COLS = ("soma_bn", "soma_purchases_bn", "buyback_bn", "nonmkt_bn", "drift_bn")
FED_ASSET_WEEKS = 13
HORIZONS = (5, 10, 20)
# Standard bill curve for the constant-share split of implied total bills.
BILL_TENORS = ("4-Week", "6-Week", "8-Week", "13-Week", "17-Week", "26-Week", "52-Week", "CMB/other")
REACTION_TRAIN = ("2018-01-01", "2025-12-31")
REACTION_ROLL = 20
REACTION_SHARE_DAYS = 90
# Clip weekly total-bill reaction to ± this many historical weekly sds.
REACTION_CLIP_SD = 3.0
# Minimum weekly gap-closure rate (half-life ≈ 8 weeks). OLS gap coef is ~0.014.
REACTION_MIN_LAMBDA = 0.08
# Do not let the reaction drive TGA below this ($bn).
REACTION_TGA_FLOOR = 200.0


# --------------------------------------------------------------------------------------
# Financing as known at an origin


def known_auctions(auc: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Auctions announced on or before the origin (announcement date, else auction − 6d)."""
    ann = auc["announcemt_date"].fillna(auc["auction_date"] - pd.Timedelta(days=6))
    return auc[ann <= origin]


BILL_CLS = ("bill", "cmb")


def _coupon_key(df: pd.DataFrame) -> pd.Series:
    """Class × whole-year original tenor (a 9y10m reopening is a 10y)."""
    yrs = ((df["maturity_date"] - df["issue_date"]).dt.days / 365.25).round().astype("Int64").astype(str)
    return df["cls"].astype(str) + "_" + yrs


def financing_path(auc: pd.DataFrame, origin: pd.Timestamp, dates: pd.DatetimeIndex,
                   buybacks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Net marketable financing ($bn) per date, as known at `origin`, bills on rollover.

    Bills: announced auctions through the last announced bill settlement;
    after it, maturing bills are rolled (net 0). Coupons: year-ago settlement
    calendar per class × tenor, replaced by the announced auction where one
    exists for that key within ±7 days.
    """
    known = known_auctions(auc, origin)
    mat = al.maturities(known, buybacks)
    fut_known = known[known["issue_date"] > origin]
    bill_end = fut_known.loc[fut_known["cls"].isin(BILL_CLS), "issue_date"].max()
    bill_end = origin if pd.isna(bill_end) else bill_end
    out = pd.DataFrame(index=dates)

    kb = fut_known[fut_known["cls"].isin(BILL_CLS)].groupby("issue_date")["public_cash_bn"].sum()
    red = mat[mat["pay_date"] > origin]
    rb = red[red["cls"].isin(BILL_CLS)].groupby("pay_date")["redeem_cash_bn"].sum()
    rc = red[~red["cls"].isin(BILL_CLS)].groupby("pay_date")["redeem_cash_bn"].sum()
    in_bill_window = dates <= bill_end
    out["known"] = in_bill_window
    out["net_bills"] = np.where(in_bill_window, kb.reindex(dates).fillna(0.0) - rb.reindex(dates).fillna(0.0), 0.0)

    proj = projected_coupons(auc, origin, dates[-1])
    kc = fut_known[~fut_known["cls"].isin(BILL_CLS)].copy()
    kc["key"] = _coupon_key(kc)
    if not proj.empty and not kc.empty:
        drop = []
        for i, r in proj.iterrows():
            m = kc[(kc["key"] == r["key"]) & ((kc["issue_date"] - r["date"]).abs() <= pd.Timedelta(days=7))]
            if not m.empty:
                drop.append(i)
        proj = proj.drop(index=drop)
    ci = pd.concat([
        proj.groupby("date")["bn"].sum() if not proj.empty else pd.Series(dtype=float),
        kc.groupby("issue_date")["public_cash_bn"].sum() if not kc.empty else pd.Series(dtype=float),
    ]).groupby(level=0).sum()
    out["coupon_iss"] = ci.reindex(dates).fillna(0.0).to_numpy()
    out["coupon_red"] = rc.reindex(dates).fillna(0.0).to_numpy()
    out["net_coupons"] = out["coupon_iss"] - out["coupon_red"]
    out["net_total"] = out["net_bills"] + out["net_coupons"]
    out.attrs["window_end"] = bill_end
    return out


def projected_coupons(auc: pd.DataFrame, origin: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Year-ago coupon/TIPS/FRN settlements re-dated one year on: (date, key, bn).

    Mid-month settlements (day 13–20) map to the 15th, month-end ones (day ≥ 26
    or the first two days of the next month) to the month end; rolled forward.
    Only settlements after the origin are returned.
    """
    hist = auc[~auc["cls"].isin(BILL_CLS) & (auc["status"] == "awarded")].copy()
    hist = hist[(hist["issue_date"] <= origin) & (hist["issue_date"] > origin - pd.DateOffset(years=2))]
    if hist.empty:
        return pd.DataFrame(columns=["date", "key", "bn"])
    hist["key"] = _coupon_key(hist)
    hist["cycle"] = hist["auction_date"].fillna(hist["issue_date"]).dt.to_period("M")
    rows = []
    g = hist.groupby(["issue_date", "key"]).agg(bn=("public_cash_bn", "sum"), cycle=("cycle", "first"))
    for (d, key), (v, cyc) in g.iterrows():
        if d.day <= 2:
            anchor = (d - pd.Timedelta(days=3)) + pd.offsets.MonthEnd(0)
        elif d.day >= 26:
            anchor = d + pd.offsets.MonthEnd(0)
        elif 13 <= d.day <= 20:
            anchor = d.replace(day=15)
        else:
            anchor = d
        eom = anchor == anchor + pd.offsets.MonthEnd(0)
        # Unchanged sizes: the last two years of auctions fill year one and two;
        # further years repeat that template (one issue per key per auction month,
        # the more recent template wins).
        n_fwd = max(2, end.year - origin.year + 1)
        for yrs in range(1, n_fwd + 1):
            nxt = anchor + pd.DateOffset(years=yrs)
            if eom:
                nxt = nxt + pd.offsets.MonthEnd(0)
            rows.append((nxt, key, v, yrs, cyc + 12 * yrs))
    p = pd.DataFrame(rows, columns=["date", "key", "bn", "lag", "cycle"])
    p["date"] = al.roll_forward(p["date"])
    p = p[(p["date"] > origin) & (p["date"] <= end)]
    # One issue per key per auction month. Prefer the one-year-ago template; two-years-ago
    # only fills cycles the last year lacks (the two can anchor to different settlement days).
    p = p.sort_values("lag").drop_duplicates(["cycle", "key"], keep="first")
    return p[["date", "key", "bn"]].reset_index(drop=True)


def financing_drift(panel_fin_net_bn: pd.Series, ladder: pd.DataFrame, origin: pd.Timestamp) -> float:
    """Trailing mean daily (DTS net financing − ladder net), $bn/day."""
    lad = ladder["net_total"].reindex(panel_fin_net_bn.index).fillna(0.0)
    r = (panel_fin_net_bn - lad)[panel_fin_net_bn.index <= origin].tail(DRIFT_WINDOW)
    return float(r.mean()) if len(r) else 0.0


# --------------------------------------------------------------------------------------
# Bill reaction (total) + tenor shares


def _bill_tenor(term: object) -> str:
    t = str(term)
    for k in BILL_TENORS[:-1]:
        if t == k:
            return k
    return "CMB/other"


def bill_tenor_shares(auc: pd.DataFrame, origin: pd.Timestamp, days: int = REACTION_SHARE_DAYS
                      ) -> pd.Series:
    """Trailing public gross bill take by tenor, normalised to 1."""
    b = auc[auc["cls"].isin(BILL_CLS)].copy()
    b["issue_date"] = pd.to_datetime(b["issue_date"])
    lo = pd.Timestamp(origin) - pd.Timedelta(days=int(days))
    b = b[(b["issue_date"] > lo) & (b["issue_date"] <= origin)]
    if b.empty:
        return pd.Series({k: 1.0 / len(BILL_TENORS) for k in BILL_TENORS})
    pub = (
        pd.to_numeric(b["total_accepted"], errors="coerce").fillna(0.0)
        - pd.to_numeric(b["soma_accepted"], errors="coerce").fillna(0.0)
    ) / 1e3
    ten = b["security_term"].map(_bill_tenor)
    sh = pub.groupby(ten).sum()
    sh = sh.reindex(BILL_TENORS).fillna(0.0)
    tot = float(sh.sum())
    if tot <= 0:
        return pd.Series({k: 1.0 / len(BILL_TENORS) for k in BILL_TENORS})
    return sh / tot


def seasonal_tga(tga: pd.Series) -> pd.Series:
    """Same-FY-day median of prior complete fiscal years ($bn)."""
    from tga_deficit import fy_day_of_year, us_fiscal_year

    s = tga.dropna().sort_index()
    by_fy: dict[int, pd.Series] = {}
    for fy, g in s.groupby(s.index.map(us_fiscal_year)):
        by_fy[int(fy)] = (
            pd.Series(g.to_numpy(), index=g.index.map(fy_day_of_year)).groupby(level=0).mean()
        )
    out = pd.Series(index=s.index, dtype=float)
    for dt in s.index:
        fy, doy = int(us_fiscal_year(dt)), int(fy_day_of_year(dt))
        pri = [by_fy[y].get(doy, np.nan) for y in by_fy if y < fy]
        pri = [float(p) for p in pri if pd.notna(p)]
        out.loc[dt] = float(np.median(pri)) if len(pri) >= 2 else np.nan
    return out


def fit_bill_reaction(
    tga: pd.Series,
    net_bills: pd.Series,
    *,
    train: tuple[str, str] = REACTION_TRAIN,
) -> dict:
    """Weekly total net bills on (intercept, TGA−seasonal, 20d ΔTGA).

    Sampled Wednesdays; y = next-7d sum of ladder net bills. Signs: above-seasonal
    TGA → fewer bills; recent TGA rise → more bills (financing into the build).
    """
    sea = seasonal_tga(tga)
    gap = (tga - sea).rename("gap")
    roll = tga.diff().rolling(REACTION_ROLL, min_periods=10).sum().rename("roll")
    nb = net_bills.sort_index()
    rows = []
    for t in tga.index[tga.index.dayofweek == 2]:
        if t < pd.Timestamp(train[0]) or t > pd.Timestamp(train[1]):
            continue
        if t not in gap.index or pd.isna(gap.loc[t]) or pd.isna(roll.loc[t]):
            continue
        fut = nb.loc[(nb.index > t) & (nb.index <= t + pd.Timedelta(days=7))]
        if len(fut) < 3:
            continue
        rows.append((float(fut.sum()), float(gap.loc[t]), float(roll.loc[t])))
    if len(rows) < 40:
        # Weak prior: mild pull toward seasonal, ignore roll.
        return {
            "intercept": 0.0, "gap": -0.05, "roll": 0.0, "y_sd": 50.0, "n": len(rows), "r2": 0.0,
            "source": "prior",
        }
    y = np.array([r[0] for r in rows])
    X = np.column_stack([np.ones(len(rows)), [r[1] for r in rows], [r[2] for r in rows]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return {
        "intercept": float(coef[0]),
        "gap": float(coef[1]),
        "roll": float(coef[2]),
        "y_sd": float(np.std(y)),
        "n": int(len(rows)),
        "r2": float(1.0 - ss_res / ss_tot),
        "source": "ols",
    }


def _seasonal_on(dates: pd.DatetimeIndex, tga_hist: pd.Series) -> pd.Series:
    """Map forward dates onto the last complete FY's same-day seasonal level."""
    from tga_deficit import fy_day_of_year, us_fiscal_year

    sea = seasonal_tga(tga_hist).dropna()
    if sea.empty:
        return pd.Series(float(tga_hist.dropna().iloc[-1]), index=dates)
    # Prefer median across the last two complete FYs of the hist seasonal itself.
    last_fy = int(us_fiscal_year(sea.index.max())) - 1
    parts = []
    for fy in (last_fy - 1, last_fy):
        sl = sea[sea.index.map(us_fiscal_year) == fy]
        if sl.empty:
            continue
        parts.append(pd.Series(sl.to_numpy(), index=sl.index.map(fy_day_of_year)).groupby(level=0).mean())
    if not parts:
        return pd.Series(float(tga_hist.dropna().iloc[-1]), index=dates)
    med = pd.concat(parts, axis=1).median(axis=1)
    return pd.Series(
        [float(med[d]) if (d := int(fy_day_of_year(t))) in med.index and pd.notna(med[d])
         else float("nan") for t in dates],
        index=dates,
    ).ffill().bfill()


def steer_bills(
    path_rollover: pd.Series,
    known: pd.Series,
    target_date: pd.Timestamp,
    target_bn: float,
    *,
    tga_hist: pd.Series | None = None,
    reaction: dict | None = None,
    hold_after: bool = False,
) -> pd.Series:
    """Total implied net bills ($bn/day).

    Through ``target_date``: close the rollover gap to ``target_bn`` on Tue/Thu
    settlements after the known window (TBAC waypoint).

    After it:
      * ``hold_after=True`` — pin TGA at ``target_bn`` (legacy backtest);
      * else weekly total-bills reaction on (TGA − seasonal, 20d ΔTGA), split
        across the week's Tue/Thu slots. Needs ``tga_hist`` + fitted ``reaction``.
    """
    idx = path_rollover.index
    t_date = idx[min(idx.searchsorted(target_date, side="right") - 1, len(idx) - 1)]
    if t_date < idx[0]:
        t_date = idx[0]
    gap0 = target_bn - float(path_rollover.loc[t_date])
    slots = idx[(~known.astype(bool).to_numpy()) & (idx <= t_date) & idx.dayofweek.isin([1, 3])]
    if len(slots) == 0:
        slots = idx[idx <= t_date][-1:]
    adj = pd.Series(0.0, index=idx)
    adj.loc[slots] = gap0 / len(slots)

    after = idx[(idx > t_date) & (~known.astype(bool).to_numpy()) & idx.dayofweek.isin([1, 3])]
    if len(after) == 0:
        return adj

    if hold_after:
        base = path_rollover + adj.cumsum()
        for s in after:
            adj.loc[s] = target_bn - float(base.loc[s])
            base = path_rollover + adj.cumsum()
        return adj

    if tga_hist is None or reaction is None:
        return adj

    sea = _seasonal_on(idx, tga_hist.dropna())
    if pd.notna(sea.loc[t_date]) and abs(float(sea.loc[t_date])) > 1e-9:
        sea = sea + (target_bn - float(sea.loc[t_date]))

    # Intercept = average net bills (structural); coupons already finance that. Flex =
    # gap pull toward seasonal (λ at least REACTION_MIN_LAMBDA) + roll only when it
    # closes the gap (same sign as the pull). Floor stops a drain spiral.
    lam = max(abs(float(reaction["gap"])), REACTION_MIN_LAMBDA)
    b_roll = float(reaction["roll"])
    clip = float(reaction.get("y_sd", 50.0)) * REACTION_CLIP_SD
    base = path_rollover + adj.cumsum()
    hist = tga_hist.dropna().sort_index()
    weeks = after.to_series().groupby(after.to_period("W-SUN")).groups
    for _, days in weeks.items():
        slots_w = pd.DatetimeIndex(days)
        t0 = slots_w[0]
        prev = base.index[base.index < t0]
        level = float(base.loc[prev[-1]]) if len(prev) else float(base.loc[t0])
        path_level = pd.concat([hist, base[base.index > hist.index.max()]])
        path_level = path_level[~path_level.index.duplicated(keep="last")].sort_index()
        roll = path_level.diff().rolling(REACTION_ROLL, min_periods=10).sum()
        roll_t = float(roll.loc[:t0].dropna().iloc[-1]) if roll.loc[:t0].dropna().size else 0.0
        sea_t = float(sea.loc[t0])
        gap = level - sea_t  # >0 above seasonal
        pull = -lam * gap  # bills that close the gap
        roll_term = b_roll * roll_t
        if pull * roll_term < 0:
            roll_term = 0.0
        week_total = float(np.clip(pull + roll_term, -clip, clip))
        end_w = slots_w[-1]
        anchor = prev[-1] if len(prev) else t0
        week_roll_delta = float(path_rollover.loc[end_w] - path_rollover.loc[anchor])
        projected = level + week_roll_delta + week_total
        if projected < REACTION_TGA_FLOOR:
            week_total += REACTION_TGA_FLOOR - projected
            week_total = float(np.clip(week_total, -clip, clip * 2))
        per = week_total / len(slots_w)
        for s in slots_w:
            adj.loc[s] = per
        base = path_rollover + adj.cumsum()
    return adj


def split_bills_by_tenor(total: pd.Series, shares: pd.Series) -> pd.DataFrame:
    """Constant-share split of a total-bills series."""
    sh = shares.reindex(BILL_TENORS).fillna(0.0)
    if float(sh.sum()) <= 0:
        sh = pd.Series({k: 1.0 / len(BILL_TENORS) for k in BILL_TENORS})
    else:
        sh = sh / sh.sum()
    return pd.DataFrame({f"bills_{k}": total.to_numpy() * float(sh[k]) for k in BILL_TENORS}, index=total.index)


def _quarter_end_after(o: pd.Timestamp, min_days: int = 14) -> pd.Timestamp:
    q = o + pd.offsets.QuarterEnd(0)
    if (q - o).days < min_days:
        q = q + pd.offsets.QuarterEnd(1)
    return q


def backtest(start: str = "2024-01-01", verbose: bool = True) -> pd.DataFrame:
    bt = pd.read_parquet(DERIVED / "flow_backtest.parquet").sort_values(["origin", "h"])
    auc = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    ladder = pd.read_parquet(DERIVED / "auction_ladder.parquet")
    panel = pd.read_parquet(DERIVED / "dts_bucket_daily.parquet")
    from tga_deficit import fetch_tga_mn

    tga = (fetch_tga_mn() / 1000.0).reindex(panel.index)
    fin_net = panel["fin_net"] / 1000.0
    bb = al.fetch_buyback_details()
    nonmkt = financing.nonmarketable_daily(financing.fetch_iiia())
    bb_cash = financing.buyback_cash(bb)
    rows = []
    for o, g in bt[bt["origin"] >= start].groupby("origin"):
        if o not in tga.index or pd.isna(tga[o]):
            continue
        dates = pd.DatetimeIndex(g["date"])
        fin = financing_path(auc, o, dates, bb)
        drift_trail = financing_drift(fin_net, ladder, o)
        drift = financing.drift_components(o, dates, tidy=auc, nonmkt=nonmkt, buybacks=bb_cash)["drift_bn"]
        op = g.set_index("date")["model"] / 1000.0
        op_act = g.set_index("date")["actual"] / 1000.0
        path = tga[o] + (op + fin["net_total"] + drift).cumsum()
        path_trail = tga[o] + (op + fin["net_total"] + drift_trail).cumsum()
        path_oracle_fin = tga[o] + (op + fin_net.reindex(dates)).cumsum()
        path_oracle_op = tga[o] + (op_act + fin["net_total"] + drift).cumsum()
        q = _quarter_end_after(o)
        q_obs = tga[:q].dropna()
        path_hold = path + steer_bills(path, fin["known"], q, tga[o], hold_after=True).cumsum()
        path_hold_trail = path_trail + steer_bills(path_trail, fin["known"], q, tga[o], hold_after=True).cumsum()
        path_tgt = (path + steer_bills(path, fin["known"], q, q_obs.iloc[-1], hold_after=True).cumsum()
                    if q_obs.index[-1] >= q - pd.Timedelta(days=4) else pd.Series(np.nan, index=dates))
        for h in HORIZONS:
            if h > len(dates):
                continue
            d = dates[h - 1]
            rows.append({
                "origin": o, "h": h, "date": d, "actual": tga.get(d), "rollover": path.iloc[h - 1],
                "steer_hold": path_hold.iloc[h - 1], "steer_target": path_tgt.iloc[h - 1],
                "rollover_trail": path_trail.iloc[h - 1], "steer_hold_trail": path_hold_trail.iloc[h - 1],
                "flat": tga[o], "oracle_financing": path_oracle_fin.iloc[h - 1],
                "oracle_flows": path_oracle_op.iloc[h - 1], "in_known_window": bool(fin["known"].iloc[h - 1]),
            })
    f = pd.DataFrame(rows).dropna(subset=["actual"])
    f.to_parquet(DERIVED / "tga_backtest.parquet", index=False)
    cols = ("rollover", "steer_hold", "steer_target", "rollover_trail", "steer_hold_trail", "flat",
            "oracle_financing", "oracle_flows")
    s = f.groupby("h").apply(lambda x: pd.Series({
        "n": len(x), **{c: (x[c] - x["actual"]).abs().mean() for c in cols},
        "share_known_window": x["in_known_window"].mean(),
    }), include_groups=False)
    (DERIVED / "tga_backtest_score.json").write_text(s.reset_index().to_json(orient="records", indent=1))
    if verbose:
        print(s.round(1).to_string())
    return s


# --------------------------------------------------------------------------------------
# Live forward path


def _next_target(origin: pd.Timestamp, min_days: int = 14) -> tuple[pd.Timestamp, float] | None:
    fut = sorted((pd.Timestamp(k), v) for k, v in TGA_TARGETS.items()
                 if (pd.Timestamp(k) - origin).days >= min_days)
    return fut[0] if fut else None


def forward(end: str | None = None, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    panel, coupon = fm.load_inputs()
    feats = fm.calendar_features(panel.index)
    origin = panel.index.max()
    tgt = _next_target(origin)
    end_ts = pd.Timestamp(end) if end else (tgt[0] if tgt else origin + pd.DateOffset(months=3))
    end_ts = max(end_ts, origin + pd.Timedelta(days=45))
    dates = fm.bdays(origin + pd.Timedelta(days=1), end_ts)
    models = fm.fit_all(panel, feats, origin)
    fc = fm.forecast(models, panel, coupon, origin, dates)
    op = fm._net(fc) / 1000.0

    auc = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    ladder = pd.read_parquet(DERIVED / "auction_ladder.parquet")
    full = pd.read_parquet(DERIVED / "dts_bucket_daily.parquet")
    bb = al.fetch_buyback_details()
    fin = financing_path(auc, origin, dates, bb)
    drift_trail = financing_drift(full["fin_net"] / 1000.0, ladder, origin)
    comp = financing.drift_components(origin, dates, tidy=auc, buybacks=financing.buyback_cash(bb))

    from tga_deficit import fetch_tga_mn

    tga0 = float(fetch_tga_mn().loc[:origin].iloc[-1] / 1000.0)
    path = pd.DataFrame(index=dates)
    path["op_flow_bn"] = op.to_numpy()
    for c in ("net_bills", "coupon_iss", "coupon_red", "net_coupons", "known"):
        path[c] = fin[c].to_numpy()
    for c in DRIFT_COLS:
        path[c] = comp[c].to_numpy()
    path["tga_rollover_bn"] = tga0 + (path["op_flow_bn"] + fin["net_total"].to_numpy() + path["drift_bn"]).cumsum()

    info: dict = {"origin": str(origin.date()), "tga_origin_bn": tga0, "known_window_end": str(fin.attrs["window_end"].date()),
                  "drift_bn_per_day": float(path["drift_bn"].mean()), "drift_trailing_bn_per_day": drift_trail,
                  "soma_asof": comp.attrs.get("soma_asof"),
                  "soma_purchase_pace_bn_day": comp.attrs.get("soma_purchase_pace_bn_day")}
    # Bills: TBAC waypoint, then total-bills reaction (seasonal gap + rolling Δ); tenor shares constant.
    t_date, t_val = tgt if tgt is not None else (_quarter_end_after(origin), tga0)
    tga_hist = fetch_tga_mn().loc[:origin] / 1000.0
    reaction = fit_bill_reaction(tga_hist, ladder["net_bills"])
    shares = bill_tenor_shares(auc, origin)
    path["implied_bills_bn"] = steer_bills(
        path["tga_rollover_bn"], path["known"], t_date, t_val,
        tga_hist=tga_hist, reaction=reaction,
    ).to_numpy()
    path["tga_target_bn"] = path["tga_rollover_bn"] + path["implied_bills_bn"].cumsum()
    by_tenor = split_bills_by_tenor(path["implied_bills_bn"], shares)
    for c in by_tenor.columns:
        path[c] = by_tenor[c].to_numpy()
    t_idx = path.index[path.index.searchsorted(t_date, side="right") - 1]
    info.update({"target_date": str(t_date.date()), "target_bn": t_val, "target_source":
                 "TGA_TARGETS (refunding)" if tgt is not None else "hold origin level (no target configured)",
                 "bill_reaction": reaction, "bill_tenor_shares": {k: float(shares[k]) for k in shares.index},
                 "tga_rollover_at_target_bn": float(path.at[t_idx, "tga_rollover_bn"]),
                 "implied_net_bills_to_target_bn": float(path.loc[:t_idx, "implied_bills_bn"].sum()),
                 "known_net_bills_bn": float(path.loc[path["known"].astype(bool), "net_bills"].sum()),
                 "net_coupons_to_target_bn": float(path.loc[:t_idx, "net_coupons"].sum()),
                 "op_flow_to_target_bn": float(path.loc[:t_idx, "op_flow_bn"].sum()),
                 **{f"{c[:-3]}_to_target_bn": float(path.loc[:t_idx, c].sum()) for c in DRIFT_COLS}})
    # Ladder net bills count Fed-held bills (bought in the market) as redeemed; private
    # supply adds those rolls back and takes out the Fed's secondary purchases.
    n_t = len(path.loc[:t_idx])
    fed_bills = comp.attrs.get("soma_purchase_pace_bn_day", 0.0) * n_t
    soma_bill_rolls = float((comp["soma_bills_bn"] + comp["soma_purchases_bn"]).iloc[:n_t].sum())
    ladder_bills = info["implied_net_bills_to_target_bn"] + float(path.loc[:t_idx, "net_bills"].sum())
    info.update({"ladder_net_bills_to_target_bn": ladder_bills,
                 "fed_bill_purchases_to_target_bn": fed_bills, "soma_bill_rolls_to_target_bn": soma_bill_rolls,
                 "private_net_bills_to_target_bn": ladder_bills + soma_bill_rolls - fed_bills})

    # Reserves and spread
    mm = pd.read_parquet(DERIVED / "money_markets_daily.parquet")
    mm_o = mm.loc[:origin].dropna(subset=["reserves_bn"]).iloc[-1]
    wal = mm["fed_assets_bn"].dropna()
    wk = wal.resample("W-WED").last().dropna()
    fed_trend = float((wk.iloc[-1] - wk.iloc[-1 - FED_ASSET_WEEKS]) / (FED_ASSET_WEEKS * 5)) if len(wk) > FED_ASSET_WEEKS else 0.0
    info["fed_assets_trend_bn_per_bday"] = fed_trend
    steps = np.arange(1, len(dates) + 1)
    for mode in ("rollover", "target"):
        path[f"reserves_{mode}_bn"] = (mm_o["reserves_bn"] - (path[f"tga_{mode}_bn"] - tga0) + fed_trend * steps)
    info["reserves_origin_bn"] = float(mm_o["reserves_bn"])

    import reserve_demand as rd

    mmp = rd.load_panel(rd.PRODUCTION["scale"])
    curve = rd.Curve().fit(mmp)
    u = mmp["y"] - curve.predict(mmp)
    rho = float(np.clip(u.tail(500).autocorr(1), 0, 0.999))
    u0 = float(u.iloc[-1])
    fut = pd.DataFrame(index=dates)
    fut = fut.join(rd.calendar_dummies(dates))
    last = mmp.iloc[-1]
    gdp_path = last["scale_bn"]
    fut["floor"], fut["cap"] = last["floor"], last["cap"]
    lad_fut = fin["net_coupons"].clip(lower=-1e9) / 100.0
    fut["coupon_settle"] = lad_fut.to_numpy()
    for mode in ("rollover", "target"):
        fut["x"] = 100.0 * path[f"reserves_{mode}_bn"].to_numpy() / gdp_path
        path[f"sofr_iorb_{mode}_bp"] = curve.predict(fut) + rho ** steps * u0
        path[f"sofr_iorb_{mode}_excal_bp"] = curve.predict(fut, calendar=False) + rho ** steps * u0
    info.update({"spread_origin_bp": float(last["y"]), "curve_resid_bp": u0, "resid_ar1": rho,
                 "curve": curve.to_dict(),
                 "reserves_at_0bp": curve.reserves_for_spread(0.0, last["scale_bn"]),
                 "reserves_at_5bp": curve.reserves_for_spread(5.0, last["scale_bn"]),
                 "slope_bp_per_100bn_now": curve.slope_bp_per_100bn(last["x"], last["scale_bn"])})

    # Error bands from the backtests (TGA level by horizon; flow model beyond 20d scales with sqrt(h)).
    tbt = DERIVED / "tga_backtest.parquet"
    if tbt.exists():
        b = pd.read_parquet(tbt)
        col = "steer_target" if tgt is not None else "steer_hold"
        q = b.assign(e=b[col] - b["actual"]).dropna(subset=["e"]).groupby("h")["e"].apply(
            lambda e: e.abs().quantile(0.8))
        h20 = float(q.get(20, q.iloc[-1]))
        band = np.array([float(np.interp(h, q.index, q.values)) if h <= 20 else h20 * np.sqrt(h / 20) for h in steps])
        path["tga_band80_bn"] = band

    path.index.name = "date"
    path.to_parquet(DERIVED / "tga_forward_path.parquet")
    (fc / 1000.0).to_parquet(DERIVED / "tga_forward_buckets.parquet")
    (DERIVED / "tga_forward_summary.json").write_text(json.dumps(info, indent=1, default=float))
    if verbose:
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in info.items() if k != "curve"}, indent=1))
    return path, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    if a.backtest:
        backtest()
    forward(a.end)


if __name__ == "__main__":
    main()
