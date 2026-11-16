"""Rest-of-year cash models for Spain and Japan.

The chart reads ``derived/{cc}_cash_monthly.parquet`` through ``investor.monthly``.

Spain (calendar year, IGAE Administración Central B.9, deficit +). Live path
is ``es.bottom_up_macro`` when ``SHIP`` (IGAE tax/spend leaves × prior-year
NGDP/HICP; PRE rest-of-year |err|/|year| −11.3% vs ``seas_add_3``). Fallback
``seas_add_3`` when SHIP is off. Sample is short of the 36-origin PRE gate used
for DE/FR; the gate is the thin_cash halves (fy≤2019 / fy≥2022).

Japan (fiscal year April–March, MoF general-account cash after the tax
gross-up). MoF’s monthly file starts FY2024, so there is one complete holdout
year. ``smly`` beats a run-rate and a one-year additive seasonal on that year
(three origins). Category×macro loses on that year and is NO-SHIP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.contract import DERIVED
from sovereign_cash.core import cash_framework as C

ES_MODEL = "seas_add_3"  # scoreboard / fallback; live uses bottom_up when SHIP
JP_MODEL = "smly"
# Rest-of-year origins: month 3, 6 and 9 of the fiscal year.
ORIGIN_MONTHS = (3, 6, 9)


def load(cc: str) -> pd.Series:
    """Deficit-positive monthly cash, local bn, the same series the chart prints."""
    from sovereign_cash.debt.note.monthly_primary import es_monthly, jp_cash_tax_corrected

    if cc == "es":
        y = es_monthly()["cash_requirement_bn"]
    elif cc == "jp":
        y = jp_cash_tax_corrected()
    else:
        raise KeyError(cc)
    y = y.astype(float).dropna().sort_index()
    y = y[~y.index.duplicated(keep="last")]
    y.index = pd.to_datetime(y.index)
    return y


def fy_start_month(cc: str) -> int:
    return 4 if cc == "jp" else 1


def fy_of(ts: pd.Timestamp, start: int) -> int:
    return int(ts.year if ts.month >= start else ts.year - 1)


def fy_month(ts: pd.Timestamp, start: int) -> int:
    return ((int(ts.month) - start) % 12) + 1


def month_ts(fy: int, m: int, start: int) -> pd.Timestamp:
    """Fiscal month ``m`` (1..12) of fiscal year ``fy``."""
    cal_m = ((start - 1 + m - 1) % 12) + 1
    cal_y = fy if cal_m >= start else fy + 1
    return pd.Timestamp(cal_y, cal_m, 1)


def fy_end(cc: str, last: pd.Timestamp) -> pd.Timestamp:
    start = fy_start_month(cc)
    fy = fy_of(last, start)
    return month_ts(fy, 12, start)


def _fy_block(y: pd.Series, fy: int, start: int) -> pd.Series | None:
    idx = [month_ts(fy, m, start) for m in range(1, 13)]
    if any(t not in y.index or pd.isna(y.loc[t]) for t in idx):
        return None
    return y.loc[idx]


def fy_seas_add(y: pd.Series, origin: pd.Timestamp, start: int, k: int) -> pd.Series | None:
    """Additive seasonal on fiscal years, data through ``origin`` only.

    Calendar ``seas_add`` is wrong for an April fiscal year. Used to score
    Japan; the shipped Japan path is ``smly``, which does not need it.
    """
    fy = fy_of(origin, start)
    yrs = []
    for cand in range(fy - 1, fy - 1 - k * 3, -1):
        if _fy_block(y, cand, start) is not None:
            yrs.append(cand)
        if len(yrs) == k:
            break
    if len(yrs) < k:
        return None
    devs = []
    for yr in yrs:
        b = _fy_block(y, yr, start).to_numpy(float)
        devs.append(b - b.mean())
    s = np.mean(devs, axis=0)
    trail = y.loc[:origin].iloc[-12:]
    if len(trail) < 12 or trail.isna().any():
        return None
    level = float(trail.sum())
    out = {}
    for m in range(fy_month(origin, start) + 1, 13):
        out[month_ts(fy, m, start)] = level / 12.0 + s[m - 1]
    return pd.Series(out)


def forecast_rest(y: pd.Series, origin: pd.Timestamp, cc: str, model: str) -> pd.Series | None:
    """Monthly flows from the month after ``origin`` through the fiscal year end."""
    start = fy_start_month(cc)
    end_m = 12
    k = fy_month(origin, start)
    if k >= end_m:
        return None
    n = end_m - k
    if model == "runrate":
        block = _fy_block_partial(y, origin, start)
        if block is None or len(block) != k:
            return None
        ytd = float(block.sum())
        idx = [month_ts(fy_of(origin, start), m, start) for m in range(k + 1, 13)]
        return pd.Series(ytd / k, index=idx)
    if model == "smly" or (model.startswith("seas_add") and start == 1):
        fc = C.forecast(y, origin, n, model if model != "smly" else "smly")
        return None if fc is None or fc.isna().any() else fc
    if model.startswith("seas_add"):
        return fy_seas_add(y, origin, start, int(model.rsplit("_", 1)[1]))
    raise ValueError(model)


def _fy_block_partial(y: pd.Series, origin: pd.Timestamp, start: int) -> pd.Series | None:
    fy = fy_of(origin, start)
    k = fy_month(origin, start)
    idx = [month_ts(fy, m, start) for m in range(1, k + 1)]
    if any(t not in y.index or pd.isna(y.loc[t]) for t in idx):
        return None
    return y.loc[idx]


def rest_errors(y: pd.Series, cc: str, model: str) -> pd.DataFrame:
    """One row per complete fiscal year × origin month. ``ratio`` is |error| / |year|."""
    start = fy_start_month(cc)
    rows = []
    fys = sorted({fy_of(t, start) for t in y.index})
    for fy in fys:
        block = _fy_block(y, fy, start)
        if block is None:
            continue
        year_total = float(block.sum())
        if abs(year_total) < 1e-9:
            continue
        for k in ORIGIN_MONTHS:
            origin = month_ts(fy, k, start)
            if origin not in y.index:
                continue
            fc = forecast_rest(y.loc[:origin], origin, cc, model)
            if fc is None or fc.isna().any():
                continue
            act_idx = [month_ts(fy, m, start) for m in range(k + 1, 13)]
            act = float(y.loc[act_idx].sum())
            err = float(fc.sum()) - act
            rows.append({
                "fy": fy, "k": k, "err": err, "ratio": abs(err) / abs(year_total),
            })
    return pd.DataFrame(rows)


def mean_ratio(err: pd.DataFrame, fy_max: int | None = None, fy_min: int | None = None) -> float:
    if err is None or err.empty:
        return float("nan")
    g = err
    if fy_max is not None:
        g = g[g["fy"] <= fy_max]
    if fy_min is not None:
        g = g[g["fy"] >= fy_min]
    if g.empty:
        return float("nan")
    return float(g["ratio"].mean())


def _es_bottomup_rest(origin: pd.Timestamp, n: int) -> pd.Series | None:
    from sovereign_cash.es import bottom_up_macro as BUM

    if not BUM.SHIP:
        return None
    panel = BUM.category_panel()
    return BUM.forecast_cash(panel.loc[:origin], origin, n)


def _next_fy_block(cc: str, y_ext: pd.Series, current_end: pd.Timestamp
                   ) -> tuple[pd.Series, str] | None:
    """Full next-FY monthly path after ``current_end``, local bn, plus source label.

    Spain: category×macro through the next December when SHIP, else smly.
    Japan: same-month last year off the extended (history + rest-of-FY) series.
    """
    start = fy_start_month(cc)
    next_fy = fy_of(current_end, start) + 1
    next_end = month_ts(next_fy, 12, start)
    n = int((next_end.year - current_end.year) * 12 + (next_end.month - current_end.month))
    if n < 12:
        return None
    origin = current_end
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), next_end, freq="MS")
    if cc == "es":
        bu = _es_bottomup_rest(origin, n)
        if bu is not None and not bu.isna().any() and bu.index.max() == next_end:
            out = bu.reindex(months)
            if not out.isna().any():
                return out.astype(float), "bottom_up_macro"
        fc = C.forecast(y_ext.loc[:origin], origin, n, "smly")
        if fc is None or fc.isna().any():
            return None
        out = fc.reindex(months)
        return (None if out.isna().any() else (out.astype(float), "smly"))
    # Japan: smly month-by-month off the extended series (rest-of-FY already filled).
    vals = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        if prev not in y_ext.index or pd.isna(y_ext.loc[prev]):
            return None
        vals.append(float(y_ext.loc[prev]))
    return pd.Series(vals, index=pd.DatetimeIndex(months), dtype=float), "smly"


def build_frame(cc: str, y: pd.Series | None = None) -> pd.DataFrame | None:
    """History plus rest-of-FY and the next full FY. None when the model cannot run."""
    y = load(cc) if y is None else y.astype(float).dropna().sort_index()
    if y.empty:
        return None
    last = y.index.max()
    end = fy_end(cc, last)
    start = fy_start_month(cc)
    next_end = month_ts(fy_of(end, start) + 1, 12, start)
    model = ES_MODEL if cc == "es" else JP_MODEL
    rest = None
    nxt = None
    rest_src = model
    nxt_src = None
    bu_all = None
    # Spain BU must originate on a printed month (panel ends at ``last``). One call
    # covers rest-of-FY + next FY; splitting at ``end`` keeps source labels honest.
    if cc == "es" and last < next_end:
        n_all = int((next_end.year - last.year) * 12 + (next_end.month - last.month))
        bu_all = _es_bottomup_rest(last, n_all)
        if (bu_all is not None and not bu_all.isna().any()
                and bu_all.index.max() == next_end):
            if last < end:
                rest = bu_all.loc[:end]
                rest_src = "bottom_up_macro"
            nxt = bu_all.loc[end + pd.offsets.MonthBegin(1):next_end]
            nxt_src = "bottom_up_macro"
    if rest is None and last < end:
        rest = forecast_rest(y, last, cc, model)
        rest_src = model
        if rest is None or rest.isna().any() or rest.index.max() != end:
            return None
    y_ext = y.copy()
    if rest is not None:
        y_ext = pd.concat([y_ext, rest.astype(float)])
        y_ext = y_ext[~y_ext.index.duplicated(keep="last")].sort_index()
    if nxt is None and (rest is not None or last >= end):
        got = _next_fy_block(cc, y_ext, end)
        if got is not None:
            nxt, nxt_src = got
    if rest is None and nxt is None:
        return None
    hist = pd.DataFrame({
        "month": y.index, "cash_requirement_bn": y.to_numpy(float), "is_forecast": False,
    })
    pieces = [hist]
    if rest is not None and len(rest):
        pieces.append(pd.DataFrame({
            "month": rest.index, "cash_requirement_bn": rest.to_numpy(float), "is_forecast": True,
        }))
    if nxt is not None and len(nxt):
        pieces.append(pd.DataFrame({
            "month": nxt.index, "cash_requirement_bn": nxt.to_numpy(float), "is_forecast": True,
        }))
    hist = pd.concat(pieces, ignore_index=True)
    hist["month"] = pd.to_datetime(hist["month"])
    hist["fc_source"] = "history"
    if rest is not None and len(rest):
        hist.loc[hist["month"].isin(rest.index), "fc_source"] = rest_src
    if nxt is not None and nxt_src is not None and len(nxt):
        hist.loc[hist["month"].isin(nxt.index), "fc_source"] = nxt_src
    if cc == "es":
        hist["cash_requirement_bottomup_bn"] = np.nan
        if bu_all is not None and not bu_all.isna().any():
            hist["cash_requirement_bottomup_bn"] = hist["month"].map(
                {pd.Timestamp(k): float(v) for k, v in bu_all.items()})
    # Legislated policy on forecast months (same overlay as the scenario pack).
    from sovereign_cash.policy.monthly import add_to_monthly

    s = hist.set_index("month")["cash_requirement_bn"].astype(float)
    adj = add_to_monthly(cc, s, last)
    hist["policy_bn"] = (adj - s).reindex(hist["month"]).to_numpy()
    hist["cash_requirement_bn"] = adj.reindex(hist["month"]).to_numpy()
    # Financing identity with buffer and bills held flat: net bonds clear the cash need.
    # Debt/issuance flows overlay this on the coupon book for forecast months.
    hist["net_bills_bn"] = 0.0
    hist["other_fin_bn"] = 0.0
    hist["buffer_change_bn"] = 0.0
    hist["bond_redemption_bn"] = 0.0
    hist["net_bonds_bn"] = hist["cash_requirement_bn"].astype(float)
    hist["bond_gross_bn"] = hist["net_bonds_bn"]
    return hist.sort_values("month").reset_index(drop=True)


def publish() -> dict[str, pd.DataFrame]:
    """Write ``derived/{es,jp}_cash_monthly.parquet``. Skip a country that cannot forecast.

    Spain prefers the full Tesoro financing runner (``es.run``) when it can build;
    Japan stays on the thin cash path.
    """
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = {}
    try:
        from sovereign_cash.es import run as ER
        ER.run(dashboard=False, verbose=False)
        es_path = DERIVED / "es_cash_monthly.parquet"
        if es_path.exists():
            out["es"] = pd.read_parquet(es_path)
    except Exception:
        frame = build_frame("es")
        if frame is not None and bool(frame["is_forecast"].any()):
            frame.to_parquet(DERIVED / "es_cash_monthly.parquet", index=False)
            out["es"] = frame
    frame = build_frame("jp")
    if frame is not None and bool(frame["is_forecast"].any()):
        frame.to_parquet(DERIVED / "jp_cash_monthly.parquet", index=False)
        out["jp"] = frame
    return out
