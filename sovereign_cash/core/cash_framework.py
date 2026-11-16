"""Shared pieces of every country's cash framework: monthly deficit models, the walk-forward
scorer, the implied bill-stock path and dashboard helpers.

Formerly ``sovereign_cash.de.common`` (that name still resolves here).

Deficit models (monthly cash requirement y, deficit +, origin t, horizon 1..H):

seas_add_K    y = L/12 + s(moy); L = trailing-12m sum, s = mean month-of-year deviation from
              its calendar-year mean over the last K complete years. Additive, so a near-zero
              annual balance (DE 2014-19) does not blow it up.
seas_mult_K   y = L · p(moy); p = mean share of the month in its year's total over K years
              (the cumulative-execution profile, differenced). Undefined when a year's total is
              near zero — those origins score NaN and the model is dropped from selection.
smly          same month last year.
target        annual target with the historical additive profile: the rest of the current
              year spreads (target − YTD) over the remaining months around s(moy); later
              years repeat the target. The target is what was known at the origin.
planhc_K      'target' with the official plan scaled by an execution haircut (outturn ÷ plan
              over completed years before the origin); the haircut rides in ``targets['hc']``.
prevyear_K    'target' with last calendar year's outturn as this year's target (the benchmark
              when no official target series is available, and a naive one when it is).

Scoring: cumulative cash requirement over t+1..t+h, h = 1/3/6/12. PRE = windows that end by
Dec-2019; OOS = origins from Jan-2022. The production model is the lowest mean PRE MAE across
horizons. With fewer than ``MIN_PRE`` PRE windows at h=12 nothing is selected on data. The
common-origin sample is shared by every model with ≥ ``MIN_PRE`` origins in the window, dropping
the thinnest until the shared sample is ≥ ``MIN_PRE``; a model with no PRE origins at all
(seas_mult_K where annual totals are near zero) never sets it. Models off the common sample are
scored on their own origins, flagged ``common=False``, and cannot be selected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)
PRE_END = pd.Timestamp("2019-12-01")
OOS_START = pd.Timestamp("2022-01-01")
# Three years of monthly origins: h=12 windows overlap by 11 months, so fewer than 36 rest on
# fewer than three independent annual outcomes.
MIN_PRE = 36
KS = (3, 5)
DEFAULT_MODEL = "smly"


def monthly_from_ytd(ytd: pd.Series) -> pd.Series:
    """Year-to-date → monthly flow. A month whose predecessor (same year) is missing is NaN."""
    ytd = ytd.dropna().sort_index()
    idx = pd.date_range(ytd.index.min(), ytd.index.max(), freq="MS")
    s = ytd.reindex(idx)
    prev = s.shift(1)
    out = s - prev
    jan = s.index.month == 1
    out[jan] = s[jan]
    return out


def _complete_years(y: pd.Series, t: pd.Timestamp, k: int) -> list[int]:
    yrs = []
    for yr in range(t.year if t.month == 12 else t.year - 1, 1900, -1):
        block = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if len(block) == 12 and block.notna().all():
            yrs.append(yr)
        elif yrs or yr < y.index.min().year:
            break
        if len(yrs) == k:
            break
    return yrs


def seasonal_dev(y: pd.Series, t: pd.Timestamp, k: int) -> pd.Series | None:
    yrs = _complete_years(y, t, k)
    if len(yrs) < k:
        return None
    devs = [y.loc[f"{yr}-01-01":f"{yr}-12-01"].pipe(lambda b: b - b.mean()).to_numpy() for yr in yrs]
    return pd.Series(np.mean(devs, axis=0), index=range(1, 13))


def seasonal_share(y: pd.Series, t: pd.Timestamp, k: int, floor: float) -> pd.Series | None:
    yrs = _complete_years(y, t, k)
    if len(yrs) < k:
        return None
    shares = []
    for yr in yrs:
        b = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if abs(b.sum()) < floor:
            return None
        shares.append((b / b.sum()).to_numpy())
    return pd.Series(np.mean(shares, axis=0), index=range(1, 13))


def anchor_to_plan(fc: pd.Series, y: pd.Series, origin: pd.Timestamp,
                   plans: dict[int, float]) -> pd.Series:
    """Shift the monthly forecast so each calendar year with a plan totals to that plan.

    The model supplies the month shape; the plan supplies the annual level. Years without a
    plan, or months already past the origin inside a year, are left alone. ``y`` is history
    through the origin (same units as ``fc``).
    """
    out = fc.copy()
    for yr, total in plans.items():
        m = out.index.year == yr
        need = 12 - (origin.month if yr == origin.year else 0)
        if yr < origin.year or m.sum() < need or need == 0:
            continue
        done = float(y[(y.index.year == yr) & (y.index <= origin)].sum())
        out.loc[m] = out.loc[m] + (total - done - float(out.loc[m].sum())) / m.sum()
    return out


def forecast(y: pd.Series, t: pd.Timestamp, H: int, model: str,
             targets: dict[int, float] | None = None, mult_floor: float = 20.0) -> pd.Series | None:
    """Monthly cash-requirement forecast for t+1..t+H using data ≤ t only."""
    hist = y.loc[:t]
    months = pd.date_range(t + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    trail = hist.iloc[-12:]
    if len(trail) < 12 or trail.isna().any():
        return None
    level = float(trail.sum())
    if model == "smly":
        vals = [hist.get(m - pd.DateOffset(years=1), np.nan) if m - pd.DateOffset(years=1) <= t
                else np.nan for m in months]
        # beyond 12 months the same-month value is itself a forecast: reuse it
        for i in range(len(vals)):
            if np.isnan(vals[i]) and i >= 12:
                vals[i] = vals[i - 12]
        return pd.Series(vals, index=months)
    kind, k = model.rsplit("_", 1)[0], int(model.rsplit("_", 1)[1]) if model[-1].isdigit() else 5
    if kind == "seas_add":
        s = seasonal_dev(hist, t, k)
        return None if s is None else pd.Series([level / 12 + s[m.month] for m in months], index=months)
    if kind == "seas_mult":
        p = seasonal_share(hist, t, k, mult_floor)
        return None if p is None else pd.Series([level * p[m.month] for m in months], index=months)
    if kind == "planhc":
        if not targets:
            return None
        hc = float(targets.get("hc", 1.0))
        scaled = {yr: (v * hc if v is not None else None) for yr, v in targets.items() if isinstance(yr, int)}
        return forecast(y, t, H, f"target_{k}", targets=scaled)
    if kind.startswith("annual_"):
        # targets already hold the annual macro (or last-year) levels for open years
        if not targets:
            return None
        return forecast(y, t, H, f"target_{k}", targets={yr: v for yr, v in targets.items() if isinstance(yr, int)})
    if kind == "prevyear":
        prev = hist.loc[f"{t.year - 1}-01-01":f"{t.year - 1}-12-01"]
        if len(prev) != 12 or prev.isna().any():
            return None
        return forecast(y, t, H, f"target_{k}", targets={t.year: float(prev.sum())})
    if kind == "target":
        s = seasonal_dev(hist, t, k)
        if s is None or not targets or t.year not in targets or targets[t.year] is None:
            return None
        out = {}
        ytd = float(hist.loc[f"{t.year}-01-01":t].sum()) if t.month < 12 else None
        for yr in sorted(set(m.year for m in months)):
            in_yr = [m for m in months if m.year == yr]
            tgt = targets.get(yr) if yr == t.year else targets.get(yr, targets[t.year])
            if tgt is None:
                tgt = targets[t.year]
            if yr == t.year:
                rem_months = list(range(t.month + 1, 13))
                rem = tgt - (ytd or 0.0)
            else:
                rem_months = list(range(1, 13))
                rem = tgt
            adj = (rem - sum(s[mm] for mm in rem_months)) / len(rem_months)
            for m in in_yr:
                out[m] = s[m.month] + adj
        return pd.Series(out).reindex(months)
    raise ValueError(model)


def model_list(has_target: bool, has_haircut: bool = False) -> list[str]:
    ms = [f"seas_add_{k}" for k in KS] + [f"seas_mult_{k}" for k in KS] + ["smly"]
    ms += [f"prevyear_{k}" for k in KS]
    if has_target:
        ms += [f"target_{k}" for k in KS]
    if has_haircut:
        ms += [f"planhc_{k}" for k in KS]
    return ms


def backtest(y: pd.Series, models: list[str], targets_asof: dict[pd.Timestamp, dict[int, float]] | None = None,
             start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Walk-forward monthly origins; one row per (origin, h, model) with cumulative fc/actual."""
    y = y.sort_index()
    last = y.dropna().index.max()
    rows = []
    for t in y.index:
        if start is not None and t < start:
            continue
        if t >= last or pd.isna(y.get(t)):
            continue
        tg = targets_asof.get(t) if targets_asof else None
        for model in models:
            fc = forecast(y, t, max(HORIZONS), model, targets=tg)
            if fc is None:
                continue
            for h in HORIZONS:
                end = t + pd.DateOffset(months=h)
                if end > last:
                    continue
                act = y.loc[t + pd.offsets.MonthBegin(1):end]
                if len(act) != h or act.isna().any():
                    continue
                rows.append({"origin": t, "h": h, "model": model, "fc": float(fc.iloc[:h].sum()),
                             "actual": float(act.sum())})
    return pd.DataFrame(rows)


def window_of(bt: pd.DataFrame) -> pd.Series:
    end = pd.to_datetime([o + pd.DateOffset(months=h) for o, h in zip(bt["origin"], bt["h"])])
    w = np.where(end <= PRE_END, "PRE", np.where(bt["origin"] >= OOS_START, "OOS", "mid"))
    return pd.Series(w, index=bt.index)


def score_table(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE by window × h × model, restricted to origins where every model has a forecast."""
    if bt.empty:
        return pd.DataFrame(columns=["window", "h", "model", "mae", "bias", "n"])
    bt = bt.assign(window=window_of(bt), err=bt["fc"] - bt["actual"])
    # A model undefined in PRE (seas_mult_K when annual totals are near zero) can never be
    # selected, so it must not set the common sample of any window.
    in_pre = set(bt.loc[bt["window"] == "PRE", "model"])
    out = []
    for (w, h), g in bt.groupby(["window", "h"]):
        piv = g.pivot_table(index="origin", columns="model", values="err")
        cover = piv.notna().sum()
        live = [m for m in piv.columns if cover[m] >= MIN_PRE and (not in_pre or m in in_pre)]
        # Drop the thinnest model until the shared sample is ≥ MIN_PRE.
        while live and len(piv[live].dropna()) < MIN_PRE:
            live.remove(min(live, key=lambda m: cover[m]))
        common = piv[live].dropna()
        for m in piv.columns:
            e = common[m] if m in live else piv[m].dropna()
            if e.empty:
                continue
            out.append({"window": w, "h": int(h), "model": m, "mae": float(e.abs().mean()),
                        "bias": float(e.mean()), "n": int(len(e)), "common": m in live})
    return pd.DataFrame(out)


def choose(scores: pd.DataFrame) -> tuple[str, str]:
    """Lowest mean PRE MAE across horizons; (model, reason)."""
    pre = scores[(scores["window"] == "PRE") & scores["common"]] if not scores.empty else scores
    n12 = pre.loc[pre["h"] == 12, "n"].max() if not pre.empty else 0
    if pre.empty or not n12 or n12 < MIN_PRE:
        return DEFAULT_MODEL, (f"PRE too short to select (h=12 windows: {int(n12 or 0)} < {MIN_PRE}); "
                               f"default '{DEFAULT_MODEL}' is used, not a data choice")
    full = pre.groupby("model")["h"].nunique()
    ok = full[full == len(HORIZONS)].index
    m = pre[pre["model"].isin(ok)].groupby("model")["mae"].mean().idxmin()
    return m, f"lowest mean PRE MAE over h={list(HORIZONS)} (h=12 windows: {int(n12)})"


def score_records(scores: pd.DataFrame, model: str, bench: str) -> list[dict]:
    out = []
    for (w, h), g in scores[scores["window"].isin(["PRE", "OOS"])].groupby(["window", "h"]):
        g = g.set_index("model")
        if model not in g.index:
            continue
        rec = {"window": w, "h": int(h), "model": round(float(g.at[model, "mae"]), 2),
               "benchmark": round(float(g.at[bench, "mae"]), 2) if bench in g.index else None,
               "n": int(g.at[model, "n"]), "model_name": model, "benchmark_name": bench,
               "all_mae": {k: {"mae": round(float(r["mae"]), 2), "n": int(r["n"]), "common": bool(r["common"])}
                           for k, r in g.iterrows()}}
        out.append(rec)
    return sorted(out, key=lambda r: (r["window"] != "PRE", r["h"]))


# --------------------------------------------------------------------------------------
# Output helpers


def write_outputs(cc: str, df: pd.DataFrame, summary: dict, derived: Path) -> float:
    from sovereign_cash.contract import validate

    worst = validate(df)
    summary["identity_resid_bn"] = round(worst, 6)
    derived.mkdir(parents=True, exist_ok=True)
    df.to_parquet(derived / f"{cc}_cash_monthly.parquet", index=False)
    (derived / f"{cc}_cash_summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    return worst


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    return str(o)


def ladder_monthly(ops: pd.DataFrame, asof: pd.Timestamp, months: pd.DatetimeIndex,
                   classes: list[str]) -> pd.Series:
    """Face maturing per month from the ops tape (all amounts issued up to asof), EUR bn."""
    o = ops[ops["sec_type"].astype(str).isin(classes) & (ops["op_dt"] <= asof)]
    stock = o.groupby(o["isin"].astype(str)).agg(amt=("amount_issued", "sum"), mat=("maturity_dt", "first"))
    mm = pd.to_datetime(stock["mat"]).dt.to_period("M").dt.to_timestamp()
    return (stock.groupby(mm)["amt"].sum() / 1000.0).reindex(months).fillna(0.0)


def cb_redemptions(cc: str, months: pd.DatetimeIndex) -> pd.Series:
    """Eurosystem PSPP+PEPP holdings maturing per month (EUR bn, positive), no reinvestment.

    Read-only use of ``issuance.qt_forward._ea_redemptions`` (last ISIN-level holdings snapshot,
    each holding leaves in the month its residual maturity ends)."""
    from issuance.qt_forward import _ea_redemptions

    r = _ea_redemptions(months.max() + pd.offsets.MonthEnd(0))
    r = r[r["country"] == cc]
    s = (-r.set_index(pd.to_datetime(r["month"]))["official_face_mn"] / 1000.0).groupby(level=0).sum()
    return s.reindex(months).fillna(0.0)


def bill_tenor_shares(ops: pd.DataFrame, asof: pd.Timestamp, classes: list[str], months: int = 12) -> dict[int, float]:
    """Share of bill gross by months to maturity (1-12) over the trailing ``months`` of the tape."""
    o = ops[ops["sec_type"].astype(str).isin(classes) & (ops["op_dt"] <= asof)
            & (ops["op_dt"] > asof - pd.DateOffset(months=months)) & (ops["amount_issued"] > 0)]
    k = ((pd.to_datetime(o["maturity_dt"]) - o["op_dt"]).dt.days / 30.44).round().clip(1, 12).astype(int)
    s = o.groupby(k)["amount_issued"].sum()
    return (s / s.sum()).to_dict()


def roll_bills(net: pd.Series, ladder: pd.Series, shares: dict[int, float],
               on_tape: pd.Series | None = None, floor: pd.Series | float = 0.0,
               fixed: pd.Series | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Gross, redemptions and feasible net bills given a required net path.

    Redemptions = tape ladder plus the roll of every euro of gross that is not already on the
    tape (``on_tape``, same month), maturing in m+k with weight shares[k]. Gross is ``fixed``
    where given (an observed month or a published calendar), else net + redemptions floored at
    ``floor`` (the issuer keeps auctioning at a minimum size). When the floor binds the returned
    net exceeds the required one and the caller books the difference to the buffer.
    """
    red = ladder.astype(float).copy()
    tape = (on_tape if on_tape is not None else pd.Series(0.0, index=net.index)).reindex(net.index).fillna(0.0)
    fl = floor if isinstance(floor, pd.Series) else pd.Series(float(floor), index=net.index)
    fl = fl.reindex(net.index).fillna(0.0)
    gross = pd.Series(0.0, index=net.index)
    for i, m in enumerate(net.index):
        fx = fixed.get(m, np.nan) if fixed is not None else np.nan
        gross.iloc[i] = float(fx) if pd.notna(fx) else max(net.iloc[i] + red.iloc[i], fl.iloc[i])
        new = max(gross.iloc[i] - tape.iloc[i], 0.0)
        for k, w in shares.items():
            if i + k < len(red):
                red.iloc[i + k] += new * w
    return gross, red, gross - red


def auction_days(months: pd.DatetimeIndex, weekday: int = 0, settle_bdays: int = 2) -> pd.Series:
    """Number of weekly auctions (``weekday``, Monday = 0) settling in each month."""
    days = pd.date_range(months.min() - pd.DateOffset(days=7), months.max() + pd.offsets.MonthEnd(0), freq="D")
    days = days[days.weekday == weekday]
    settle = (days + pd.offsets.BDay(settle_bdays)).to_period("M").to_timestamp()
    return settle.value_counts().reindex(months).fillna(0).astype(int)
