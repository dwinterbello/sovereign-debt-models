"""Multi-year backtest of the France/Germany/Italy component models (Nov 2026).

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.experiments.multi_year_backtest

The scenario page forecasts four calendar years, so the model has to be good over one
to three years ahead, not only the next twelve months. Each December origin from 2013
forecasts the next 36 months, conditional on the macro levels that actually happened
(the scenario page conditions on a macro path too), and the errors are scored on
calendar-year sums: the cash requirement excluding interest (interest is the debt book's
job, scored separately where the book exists) and each line.

Specifications (a line not listed holds its seasonal pattern in levels, as now):
  current   receipts / NGDP level, primary / HICP level (shipped Nov 2026)
  ngdp      receipts and primary / NGDP level: spending keeps its share of the economy
  drift     receipts / NGDP, primary / HICP with the share's own trend (real growth)
  ngdp_all  'ngdp', and every other non-interest line / NGDP level too
  naive     every line: last calendar year's share of NGDP, with its month pattern

Writes derived/multi_year_backtest.json.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sovereign_cash import monthly_components as M
from sovereign_cash.yoy_components import IDENTITY, SPEC, components

OUT = Path(__file__).resolve().parents[1] / "derived" / "multi_year_backtest.json"
SPECS = ("current", "xc", "mr", "mr_xc", "mr10_xc", "mr_xc_all", "ngdp", "drift", "naive",
         "combo", "combo_de_both", "combo_it_ngdp", "final")
# 'final' is the production model itself (monthly_components.forecast with METHOD, grants handled), on realised
# macro: what ships is what is scored. 'combo' = the debt-brake-era choice (German spending reverts).
# combos: per-country choices. Germany: spending share reverts (debt brake); Italy: capital spending and
# other revenue keep last year's share of nominal GDP; France: as shipped.
COMBO_NAIVE = {"it": ("cap_exp", "other_rev")}
CRISIS = ("2020-03-01", "2021-12-01")  # Covid months masked in '_xc' specs (the Kalman filter skips them)
H = 36
COVID = {2020, 2021}


def _share_fc(y: pd.Series, x: pd.Series, origin: pd.Timestamp, h: int, trend: bool = False,
              revert: bool = False, mask: bool = False, window: int | None = None) -> pd.Series | None:
    """Share of the driver level, forecast and multiplied back.

    Default: SARIMA(1,0,0)(0,1,1)12 on the share (a shock to the share is permanent).
    ``revert``: month means + AR(1) errors, so the share returns to its average month pattern.
    ``mask``: Covid months set missing. ``window``: years of history used.
    """
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    hist = y.loc[:origin].dropna()
    if window:
        hist = hist.loc[origin - pd.DateOffset(years=window) + pd.offsets.MonthBegin(1):]
    xa = x.reindex(hist.index.union(months))
    if xa.reindex(months).isna().any():
        return None
    z = (hist / xa.reindex(hist.index)).dropna()
    if len(z) < M.MIN_HIST:
        return None
    if mask:
        z = z.copy()
        z.loc[CRISIS[0]:CRISIS[1]] = np.nan
    z = z.asfreq("MS")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if revert:
                dum = pd.get_dummies(z.index.month).astype(float).set_axis(z.index)
                dfut = pd.get_dummies(months.month).astype(float).set_axis(months).reindex(columns=dum.columns, fill_value=0.0)
                res = SARIMAX(z, exog=dum, order=(1, 0, 0), trend="n",
                              enforce_stationarity=True).fit(disp=False, maxiter=200)
                f = np.asarray(res.forecast(h, exog=dfut))
            else:
                res = SARIMAX(z, order=M.ORDER, seasonal_order=M.SEASONAL, trend="c" if trend else "n",
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
                f = np.asarray(res.forecast(h))
    except Exception:  # noqa: BLE001
        return None
    return pd.Series(f, index=months) * xa.reindex(months)


def _level_fc(y: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    return M.fit_forecast(y.dropna(), origin, h, None, kind="level")


def _naive(y: pd.Series, x: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series:
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    last = y.loc[f"{origin.year}-01-01":f"{origin.year}-12-01"]
    sh = (last / x.reindex(last.index)).to_numpy()
    return pd.Series([sh[m.month - 1] * float(x.loc[m]) for m in months], index=months)


def forecast_lines(cc: str, spec: str, frame: pd.DataFrame, lv: dict[str, pd.Series],
                   origin: pd.Timestamp) -> dict[str, pd.Series] | None:
    ngdp = lv["ngdp_level"].shift(M.MACRO_LAG)
    hicp = lv["hicp_level"].shift(M.MACRO_LAG)
    last = min(ngdp.dropna().index.max(), hicp.dropna().index.max())
    h = min(H, (last.year - origin.year) * 12 + last.month - origin.month)
    if h < 12:
        return None
    if spec == "final":
        fc = M.forecast(cc, origin, h, use_authority=False)
        return None if fc is None else {n: fc[n] for n, _ in SPEC[cc] if n != "interest"}
    out = {}
    for name, key in SPEC[cc]:
        if name == "interest":
            continue
        y = frame[name]
        if spec.startswith("combo"):
            if name in COMBO_NAIVE.get(cc, ()):
                f = _naive(y, ngdp, origin, h)
            elif cc == "de" and (key == "cpi" or (key == "ngdp" and spec == "combo_de_both")):
                f = _share_fc(y, ngdp if key == "ngdp" else hicp, origin, h, revert=True)
            elif cc == "it" and key == "cpi" and spec == "combo_it_ngdp":
                f = _share_fc(y, ngdp, origin, h)
            elif key == "ngdp":
                f = _share_fc(y, ngdp, origin, h)
            elif key == "cpi":
                f = _share_fc(y, hicp, origin, h)
            else:
                f = _level_fc(y, origin, h)
            if f is None:
                return None
            out[name] = f
            continue
        kw = {"mask": "xc" in spec, "revert": spec.startswith("mr"), "window": 10 if spec.startswith("mr10") else None}
        if spec == "naive":
            f = _naive(y, ngdp, origin, h)
        elif key == "ngdp":
            f = _share_fc(y, ngdp, origin, h, **kw)
        elif key == "cpi":
            if spec == "ngdp":
                f = _share_fc(y, ngdp, origin, h)
            else:
                f = _share_fc(y, hicp, origin, h, trend=spec == "drift", **kw)
        elif spec == "mr_xc_all":
            f = _share_fc(y, ngdp, origin, h, **kw)
        else:
            f = _level_fc(y, origin, h)
        if f is None:
            return None
        out[name] = f
    return out


def run(countries=("fr", "de", "it")) -> dict:
    res: dict = {}
    for cc in countries:
        frame = components(cc)
        lv = M.level_drivers(cc)
        signs = {k: v for k, v in IDENTITY[cc].items() if k != "interest"}
        last_full = int(frame.dropna().index.max().year) - 1
        rows = []
        for y0 in range(2013, last_full):
            origin = pd.Timestamp(f"{y0}-12-01")
            if frame.loc[:origin].dropna().shape[0] < M.MIN_HIST:
                continue
            for spec in SPECS:
                fc = forecast_lines(cc, spec, frame, lv, origin)
                if fc is None:
                    continue
                for k in (1, 2, 3):
                    yr = y0 + k
                    if yr > last_full or not all((fc[n].index.year == yr).sum() == 12 for n in signs):
                        continue
                    act = {n: float(frame[n][frame.index.year == yr].sum()) for n in signs}
                    got = {n: float(fc[n][fc[n].index.year == yr].sum()) for n in signs}
                    a_tot = sum(signs[n] * act[n] for n in signs)
                    g_tot = sum(signs[n] * got[n] for n in signs)
                    rows.append({"origin": y0, "year": yr, "h": k, "spec": spec, "err_total": g_tot - a_tot,
                                 **{f"err_{n}": got[n] - act[n] for n in signs}})
        d = pd.DataFrame(rows)
        summ = {}
        for spec in SPECS:
            s = d[d.spec == spec]
            summ[spec] = {}
            for k in (1, 2, 3):
                a = s[s.h == k]
                b = a[~a.year.isin(COVID)]
                c = a[~a.year.isin(COVID) & ~a.origin.isin([2019, 2020, 2021])]
                summ[spec][f"h{k}"] = {
                    "n": int(len(a)), "mae_total": round(float(a.err_total.abs().mean()), 2),
                    "mae_total_ex_covid": round(float(b.err_total.abs().mean()), 2),
                    "bias_total_ex_covid": round(float(b.err_total.mean()), 2),
                    "mae_total_clean": round(float(c.err_total.abs().mean()), 2),
                    "bias_total_clean": round(float(c.err_total.mean()), 2),
                    **{f"mae_{n}_ex_covid": round(float(b[f"err_{n}"].abs().mean()), 2) for n in signs},
                    **{f"bias_{n}_ex_covid": round(float(b[f"err_{n}"].mean()), 2) for n in signs},
                }
        res[cc] = {"summary": summ, "rows": d.round(3).to_dict("records")}
    return res


if __name__ == "__main__":
    out = run()
    OUT.write_text(json.dumps(out, indent=1))
    for cc, r in out.items():
        print(f"== {cc}  (cash requirement ex interest, EUR bn; MAE all / ex covid years / bias ex covid / MAE ex covid origins)")
        for spec, s in r["summary"].items():
            print(f"  {spec:9s}", "  ".join(f"h{k}: {s[f'h{k}']['mae_total']:5.1f}/{s[f'h{k}']['mae_total_ex_covid']:5.1f}"
                                           f"/{s[f'h{k}']['bias_total_ex_covid']:+5.1f}/{s[f'h{k}']['mae_total_clean']:5.1f}" for k in (1, 2, 3)))
        lines = [k for k in r["summary"]["current"]["h2"] if k.startswith("mae_") and "total" not in k]
        # normal-times score: origins whose history ends before Covid or two years after it
        for ln in lines:
            nm = ln[4:-9]
            print(f"   {nm:12s} h2 MAE/bias ex covid:", "  ".join(
                f"{spec} {r['summary'][spec]['h2'][ln]:.1f}/{r['summary'][spec]['h2']['bias_' + nm + '_ex_covid']:+.1f}" for spec in SPECS))
