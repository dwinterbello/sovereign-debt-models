"""Monthly-average SOFR − IORB and SOFR − EFFR: backtest and forward (SERFF).

SOFR (SR1) and fed funds (ZQ) futures settle on the calendar-day average of the
daily fixing over the contract month (weekends and holidays carry the prior
business day). SERFF = SR1 − ZQ, so the instrument is

    avg_m(SOFR) − avg_m(EFFR) = avg_m(SOFR − IORB) − avg_m(EFFR − IORB)

(IORB changes inside a month cancel). EFFR − IORB is sticky — it held −7bp
through 2025 and only ratcheted after months of repo pressure — so it is
held at its trailing 20-day median.

Backtest: month-end origins from 2022-01, targets m+1…m+3, expanding monthly
curve refits. Variants
    rw         last 20-business-day mean of the spread
    rw_cal     rw + (calendar block of target month − of trailing 20 days)
    curve_hold reserve curve at origin reserves + target calendar + decayed residual
    curve_res  curve_hold with the trailing 20-day mean residual held, not decayed
    curve_orc  curve at realised reserves + held residual (reserves-path oracle)
curve_orc − curve_res bounds what a perfect reserves forecast is worth.

  PYTHONPATH=. .venv/bin/python us/fiscal/serff.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import reserve_demand as rd  # noqa: E402

DERIVED = HERE / "derived"
START = "2022-01-31"
LOOK = 20
VARIANTS = ("rw", "rw_cal", "curve_hold", "curve_res", "curve_orc")
WINDOWS = {"PRE": ("2022-01-01", "2023-12-31"), "OOS": ("2024-01-01", "2099-12-31")}


def calendar_month_mean(s: pd.Series, month: pd.Period) -> float:
    """Calendar-day average, weekends/holidays carrying the prior business day."""
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    v = s[s.index <= days[-1]].reindex(s.index.union(days)).ffill().reindex(days)
    return float(v.mean()) if v.notna().all() else np.nan


def backtest(mm: pd.DataFrame) -> pd.DataFrame:
    month_ends = mm.groupby(mm.index.to_period("M")).apply(lambda g: g.index[-1])
    month_ends = month_ends[month_ends >= START]
    rows = []
    for per, o in month_ends.items():
        hist = mm[mm.index <= o]
        curve = rd.Curve().fit(hist)
        u = hist["y"] - curve.predict(hist)
        rho = float(np.clip(u.tail(500).autocorr(1), 0.0, 0.999))
        u0, u20 = float(u.iloc[-1]), float(u.tail(LOOK).mean())
        tail = hist.tail(LOOK)
        _, cal_tail = curve.parts(tail)
        y20 = float(tail["y"].mean())
        e20 = float(tail["effr_iorb_bp"].median())
        se20 = float(tail["sofr_effr_bp"].mean())
        for h in (1, 2, 3):
            tm = per + h
            d = mm[mm.index.to_period("M") == tm]
            if len(d) < 15:
                continue
            fut = d.copy()
            fut["x"] = hist["x"].iloc[-1] * hist["scale_bn"].iloc[-1] / d["scale_bn"]
            core_h, cal_h = curve.parts(fut)
            core_o, cal_o = curve.parts(d)
            steps = np.arange(1, len(d) + 1) + 21 * (h - 1)
            pred = {
                "rw": np.full(len(d), y20),
                "rw_cal": y20 + cal_h - cal_tail.mean(),
                "curve_hold": core_h + cal_h + rho ** steps * u0,
                "curve_res": core_h + cal_h + u20,
                "curve_orc": core_o + cal_o + u20,
            }
            act = calendar_month_mean(mm["y"], tm)
            act_se = calendar_month_mean(mm["sofr_effr_bp"], tm)
            row = {"origin": o, "h": h, "month": str(tm), "actual": act, "actual_serff": act_se}
            for k, v in pred.items():
                s = pd.Series(v, index=d.index)
                row[k] = calendar_month_mean(s, tm)
                row[f"{k}_serff"] = row[k] - e20 if k != "rw" else se20
            rows.append(row)
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame) -> dict:
    out = {}
    for w, (a, b) in WINDOWS.items():
        g = bt[(bt["origin"] >= a) & (bt["origin"] <= b)]
        out[w] = {}
        for h, gh in g.groupby("h"):
            out[w][f"h{h}"] = {
                "n": len(gh),
                **{v: round(float((gh[v] - gh["actual"]).abs().mean()), 2) for v in VARIANTS},
                **{f"{v}_serff": round(float((gh[f"{v}_serff"] - gh["actual_serff"]).abs().mean()), 2)
                   for v in VARIANTS},
            }
    return out


def forward(mm: pd.DataFrame, choice: str) -> dict:
    """Monthly averages on the live TGA/reserves path for every variant; `choice` is the gated one."""
    path = pd.read_parquet(DERIVED / "tga_forward_path.parquet")
    path = path[path.index > mm.index[-1]]
    e20 = float(mm["effr_iorb_bp"].tail(LOOK).median())
    curve = rd.Curve().fit(mm)
    u = mm["y"] - curve.predict(mm)
    rho = float(np.clip(u.tail(500).autocorr(1), 0.0, 0.999))
    steps = np.arange(1, len(path) + 1)
    _, cal_tail = curve.parts(mm.tail(LOOK))
    cal_fwd = path["sofr_iorb_target_bp"] - path["sofr_iorb_target_excal_bp"]
    y20 = float(mm["y"].tail(LOOK).mean())
    variants = {
        "rw": pd.Series(y20, index=path.index),
        "rw_cal": y20 + cal_fwd - cal_tail.mean(),
        "curve_hold": path["sofr_iorb_target_bp"],
        "curve_res": path["sofr_iorb_target_bp"] - rho ** steps * float(u.iloc[-1]) + float(u.tail(LOOK).mean()),
    }
    out = {"effr_iorb_hold_bp": round(e20, 2), "origin": str(mm.index[-1].date()), "choice": choice,
           "months": {}}
    months = sorted({d.to_period("M") for d in path.index})
    for per in months:
        if per.end_time.normalize() > path.index[-1] + pd.Timedelta(days=3):
            continue
        rec = {}
        for k, fwd in variants.items():
            s = pd.concat([mm["y"], fwd]).sort_index()
            v = calendar_month_mean(s, per)
            rec[k] = {"sofr_iorb_bp": round(v, 2), "serff_bp": round(v - e20, 2)}
        out["months"][str(per)] = rec
    return out


def run(verbose: bool = True) -> dict:
    mm = rd.load_panel(rd.PRODUCTION["scale"])
    bt = backtest(mm)
    bt.to_parquet(DERIVED / "serff_backtest.parquet", index=False)
    sc = score(bt)
    pre = sc["PRE"]
    curve_vars = [v for v in VARIANTS if v.startswith("curve") and v != "curve_orc"]
    pre_mae = {v: np.mean([pre[h][v] for h in pre]) for v in ("rw", "rw_cal", *curve_vars)}
    choice = min(pre_mae, key=pre_mae.get)
    fwd = forward(mm, choice)
    report = {"score": sc, "pre_mean_mae": {k: round(v, 2) for k, v in pre_mae.items()}, "choice": choice,
              "forward": fwd}
    (DERIVED / "serff.json").write_text(json.dumps(report, indent=1, default=float))
    if verbose:
        for w in sc:
            print(w)
            print(pd.DataFrame(sc[w]).T.to_string())
        print("PRE mean MAE:", report["pre_mean_mae"], "→", choice)
        print(json.dumps(fwd, indent=1))
    return report


if __name__ == "__main__":
    run()
