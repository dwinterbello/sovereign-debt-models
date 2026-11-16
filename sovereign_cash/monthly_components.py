"""Monthly component forecasts: seasonal SARIMAX on each fiscal line, the cash requirement as the identity.

Driven lines are modelled as a share of their driver's LEVEL, lagged two months:
receipts / tax on nominal GDP (seasonally adjusted, interpolated from quarters),
primary spending on the HICP index. The share carries seasonal ARIMA
``(1,0,0)x(0,1,1,12)`` errors, so a line moves in proportion to its driver's level:
a one-off level shift persists and faster growth compounds. Interest is driven by
the effective rate; lines with no base repeat their seasonal pattern.

Forward drivers are authority paths (AMECO / DG Tresor RESF NGDP, WEO / RESF CPI,
Italy DFP GDP) from ``macro_authority``, not last print held flat.

The earlier specification (line level on the driver's YoY rate) is kept as
``spec="yoy"`` only so ``experiments/level_spec_backtest.py`` can be reproduced:
it ties a level to a growth rate, so a lasting growth shock gave a constant effect
and a level shift faded after a year.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sovereign_cash.core import cash_framework as C
from sovereign_cash.yoy_components import SPEC, _eurostat_json, components, deficit

ORDER = (1, 0, 0)
SEASONAL = (0, 1, 1, 12)
MIN_HIST = 48
MACRO_LAG = 2
CACHE = Path(__file__).resolve().parent / "data" / "debt"
log = logging.getLogger(__name__)

# Interest is carved and often sparse early — always level/diff, never MoM %.
PCT_MOM = {
    "fr": {"recettes", "primary"},
    "de": {"revenue", "primary"},
    "it": {"tax_rev", "other_rev", "primary", "cap_exp"},
}


@lru_cache(maxsize=8)
def monthly_drivers(cc: str) -> dict[str, pd.Series]:
    """Realised HICP YoY, NGDP YoY, and effective interest rate (pp)."""
    from sovereign_cash.interest_split import rates_monthly

    geo = cc.upper()
    path = CACHE / f"macro_m_{cc}.parquet"
    if path.exists() and (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days < 7:
        df = pd.read_parquet(path)
        out = {c: df[c].dropna() for c in df.columns}
        if "rates" not in out:
            out["rates"] = rates_monthly(cc).dropna()
        return out
    hicp = _eurostat_json("prc_hicp_midx", freq="M", unit="I15", coicop="CP00", geo=geo)
    h = (hicp.assign(month=pd.to_datetime(hicp["time"]))
         .set_index("month")["value"].sort_index().pct_change(12) * 100.0)
    gdp = _eurostat_json("namq_10_gdp", freq="Q", unit="CP_MEUR", s_adj="NSA",
                         na_item="B1GQ", geo=geo)
    gdp = gdp[gdp["time"].str.contains(r"Q[1-4]$", regex=True)].copy()
    gdp["q"] = pd.PeriodIndex(gdp["time"], freq="Q").to_timestamp(how="start")
    g = gdp.set_index("q")["value"].sort_index().pct_change(4) * 100.0
    g.index = pd.PeriodIndex(g.index, freq="Q").to_timestamp(how="start")
    months = pd.date_range(g.index.min(), h.index.max(), freq="MS")
    rates = rates_monthly(cc, through=months.max())
    out = pd.DataFrame({
        "hicp_yoy": h.reindex(months),
        "ngdp_yoy": g.reindex(months).ffill(),
        "rates": rates.reindex(months).ffill(),
    })
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return {c: out[c].dropna() for c in out.columns}


# ---------------------------------------------------------------- level specifications
# "yoy"       production: component level on the driver's YoY rate (lag MACRO_LAG)
# "loglevel"  log(component) on log(driver level): constant elasticity, estimated
# "ratio"     component / driver level, seasonal ARIMA: elasticity exactly one
SPECS = ("yoy", "loglevel", "ratio")
LEVEL_KEY = {"ngdp": "ngdp_level", "cpi": "hicp_level"}
SHOCK_KEY = {"ngdp": "ngdp", "cpi": "hicp"}  # driver -> key in the ``shock`` dict
# Levels everywhere (economics, not fit, decides). The walk-forward comparison
# (experiments/level_spec_backtest.py, Nov 2026) against the old YoY spec, PRE / OOS h12 MAE:
#   FR share of level  10.5 vs 15.4  /  14.6 vs 15.9
#   IT share of level  36.6 vs 36.6  /  35.0 vs 57.4
#   DE share of level  11.6 vs  8.9  /  79.8 vs 81.7   (23% worse on mean PRE; kept on theory)
# Free log-level elasticities of receipts: FR 0.68, DE 1.01, IT 1.11, none distinguishable from one.
LEVEL_SPEC = {"fr": "ratio", "de": "ratio", "it": "ratio"}
# How each line's share of its driver level is carried forward (ratio spec). Chosen on the multi-year
# walk-forward test (experiments/multi_year_backtest.py; December origins from 2013, one to three calendar years
# ahead, conditional on realised macro):
#   'share'     SARIMA(1,0,0)(0,1,1)12 on the share: a change in the share persists. France, and receipts.
#   'share_xc'  the same with the Covid months (Mar 2020-Dec 2021) left out of the fit, as a one-off. German
#               primary spending since the 2025 reform of the debt brake (defence above 1% of GDP exempt,
#               special funds): the 'revert' rule that scored best over 2013-24 under the brake missed 2025 by
#               -16.4bn (spending too low); 'share' +15.0bn; 'share_xc' -5.9bn.
#   'revert'    month means + AR(1): the share returns to its usual level. The debt-brake-era rule for Germany,
#               kept for the record (and the test); not used going forward.
#   'hold'      last twelve months' share of nominal GDP, month by month: Italy's capital spending and other
#               revenue, which the level SARIMA held flat in euros (bias -29 and -11bn at two years). Other
#               revenue is modelled excluding RRF grants, which are added back only as scheduled (``GRANTS``).
METHOD = {("de", "primary"): "share_xc", ("it", "cap_exp"): "hold", ("it", "other_rev"): "hold"}
CRISIS = ("2020-03-01", "2021-12-01")
GRANTS = {("it", "other_rev")}  # lines that carry EU RRF grants: stripped from history, scheduled ones added back


def line_method(cc: str, name: str) -> str:
    return METHOD.get((cc, name), "share")


def line_key(cc: str, name: str, key: str | None) -> str | None:
    """The driver a line moves with: its SPEC key, or nominal GDP for lines held as a share of it."""
    return "ngdp" if line_method(cc, name) == "hold" else key


def spec_for(cc: str, spec: str | None = None) -> str:
    return spec if spec is not None else LEVEL_SPEC.get(cc, "ratio")


@lru_cache(maxsize=8)
def level_drivers(cc: str) -> dict[str, pd.Series]:
    """Monthly nominal GDP (seasonally adjusted, log-interpolated from quarters) and the HICP index.

    Levels are indices (mean 1 over 2015) so ratios are of order one.
    """
    path = CACHE / f"macro_lvl_{cc}.parquet"
    if path.exists() and (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days < 7:
        df = pd.read_parquet(path)
        return {c: df[c].dropna() for c in df.columns}
    geo = cc.upper()
    gdp = _eurostat_json("namq_10_gdp", freq="Q", unit="CP_MEUR", s_adj="SCA", na_item="B1GQ", geo=geo)
    gdp = gdp[gdp["time"].str.contains(r"Q[1-4]$", regex=True)].copy()
    q = pd.PeriodIndex(gdp["time"], freq="Q")
    # each quarter sits on its middle month; months in between are log-linear
    mid = q.to_timestamp(how="start") + pd.offsets.MonthBegin(1)
    g = pd.Series(np.log(gdp["value"].to_numpy(float)), index=mid).sort_index()
    months = pd.date_range(g.index.min(), g.index.max(), freq="MS")
    g = np.exp(g.reindex(g.index.union(months)).interpolate(method="time").reindex(months))
    hicp = _eurostat_json("prc_hicp_midx", freq="M", unit="I15", coicop="CP00", geo=geo)
    h = hicp.assign(month=pd.to_datetime(hicp["time"])).set_index("month")["value"].sort_index()
    out = pd.DataFrame({"ngdp_level": g / g.loc["2015"].mean(), "hicp_level": h / h.loc["2015"].mean()})
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return {c: out[c].dropna() for c in out.columns}


def level_path(cc: str, key: str, origin: pd.Timestamp, h: int, *, gap: list[float] | None = None,
               annual: pd.Series | None = None) -> pd.Series:
    """Driver level through origin+h: history, then the authority's annual growth month by month.

    ``gap`` (percent, one per forecast month) multiplies the forward path by (1 + gap/100);
    history is never shocked. The result is lagged ``MACRO_LAG`` months like the YoY regressors.
    """
    from sovereign_cash.macro_authority import annual_authority

    hist = level_drivers(cc)[LEVEL_KEY[key]]
    if annual is None:
        ngdp_a, cpi_a, _ = annual_authority(cc)
        annual = ngdp_a if key == "ngdp" else cpi_a
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    full = hist.copy()
    last_m, level = hist.index.max(), float(hist.iloc[-1])
    fwd = pd.date_range(last_m + pd.offsets.MonthBegin(1), months.max(), freq="MS")
    for m in fwd:
        y = int(m.year)
        gy = float(annual.get(y, annual.loc[:y].iloc[-1] if len(annual.loc[:y]) else 0.0))
        level *= (1.0 + gy / 100.0) ** (1.0 / 12.0)
        full.loc[m] = level
    full = full.sort_index()
    if gap is not None:
        for m, gp in zip(months, gap):
            full.loc[m] = full.loc[m] * (1.0 + gp / 100.0)
    return full.shift(MACRO_LAG).rename(LEVEL_KEY[key])


def _fit_revert(z: pd.Series, origin: pd.Timestamp, h: int) -> Fit | None:
    """Share = its month mean + AR(1): the share returns to its usual level and seasonal shape."""
    hist = z.loc[:origin].dropna().asfreq("MS")
    if hist.notna().sum() < MIN_HIST:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    cols = [f"m{i:02d}" for i in range(1, 13)]
    dum = pd.DataFrame({c: (hist.index.month == i).astype(float) for i, c in enumerate(cols, start=1)}, index=hist.index)
    fut = pd.DataFrame({c: (months.month == i).astype(float) for i, c in enumerate(cols, start=1)}, index=months)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(hist, exog=dum, order=(1, 0, 0), trend="n", enforce_stationarity=True).fit(disp=False, maxiter=200)
            fc = res.forecast(h, exog=fut)
    except Exception as exc:  # noqa: BLE001
        log.warning("revert fit failed: %s", exc)
        return None
    return Fit(forecast=pd.Series(np.asarray(fc), index=months), exog=None, beta=None, beta_se=None,
               params={str(k): float(v) for k, v in res.params.items()}, nobs=int(res.nobs),
               sample_start=hist.index.min(), sample_end=hist.index.max())


def _fit_masked(z: pd.Series, origin: pd.Timestamp, h: int) -> Fit | None:
    """``_fit`` on the share with the Covid months set missing (the Kalman filter steps over them)."""
    hist = z.loc[:origin].dropna().asfreq("MS")
    hist.loc[CRISIS[0]:CRISIS[1]] = np.nan
    if hist.notna().sum() < MIN_HIST:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(hist, order=ORDER, seasonal_order=SEASONAL, enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=100)
            fc = res.forecast(h)
    except Exception as exc:  # noqa: BLE001
        log.warning("masked share fit failed: %s", exc)
        return None
    return Fit(forecast=pd.Series(np.asarray(fc), index=months), exog=None, beta=None, beta_se=None,
               params={str(k): float(v) for k, v in res.params.items()}, nobs=int(res.nobs),
               sample_start=hist.index.min(), sample_end=hist.index.max())


@lru_cache(maxsize=4)
def rrf_grants(cc: str) -> pd.Series:
    """EU RRF grant receipts by month, EUR bn: Banca d'Italia's recorded instalments (grant part estimated from the
    loan part at the plan ratio), then the instalments still scheduled (``it.events.RRF_EVENTS``). Zero elsewhere."""
    if cc != "it":
        return pd.Series(dtype=float)
    from sovereign_cash.it import events as ev
    from sovereign_cash.it import sources as S

    _, t2 = S.fetch_bdi(False)
    hist = ev.rrf_grants_history(pd.to_numeric(t2["eu_loans"], errors="coerce") / 1000.0)
    last = hist.index.max()
    fwd = pd.date_range(last + pd.offsets.MonthBegin(1), "2031-12-01", freq="MS")
    out = pd.concat([hist, ev.rrf_split(fwd)["grant_bn"]])
    return out[~out.index.duplicated()].sort_index()


def line_history(cc: str, name: str, y: pd.Series) -> pd.Series:
    """The series a line's model is fitted on: grants stripped where the line carries them."""
    if (cc, name) not in GRANTS:
        return y
    g = rrf_grants(cc).reindex(y.index).fillna(0.0)
    return y - g


def line_extra(cc: str, name: str, months: pd.DatetimeIndex) -> pd.Series:
    """What is added back to a line's forecast: the grants scheduled in its forecast months."""
    if (cc, name) not in GRANTS:
        return pd.Series(0.0, index=months)
    return rrf_grants(cc).reindex(months).fillna(0.0)


def _fit_hold(z: pd.Series, origin: pd.Timestamp, h: int) -> Fit | None:
    """Share = the same calendar month's share over the last twelve months."""
    hist = z.loc[:origin].dropna()
    last = hist.iloc[-12:]
    if len(last) < 12:
        return None
    by_month = {int(m.month): float(v) for m, v in last.items()}
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return Fit(forecast=pd.Series([by_month[int(m.month)] for m in months], index=months), exog=None, beta=None,
               beta_se=None, params={f"share_m{k:02d}": v for k, v in sorted(by_month.items())}, nobs=12,
               sample_start=last.index.min(), sample_end=last.index.max())


def _fit_level(y: pd.Series, origin: pd.Timestamp, h: int, x: pd.Series, spec: str,
               method: str = "share") -> tuple[pd.Series, Fit] | None:
    """One component on a driver level: 'loglevel' (estimated elasticity) or 'ratio' (unit elasticity).

    For 'ratio', ``method`` says how the share is carried forward (see ``METHOD``).
    """
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    hist = y.loc[:origin].dropna()
    xa = x.reindex(hist.index.union(months))
    if xa.reindex(months).isna().any():
        return None
    if spec == "loglevel":
        if (hist <= 0).any():
            return None
        f = _fit(np.log(hist), origin, h, np.log(xa).rename(x.name))
        if f is None:
            return None
        return pd.Series(np.exp(f.forecast.to_numpy()), index=months), f
    if spec == "ratio":
        z = (hist / xa.reindex(hist.index)).dropna()
        f = {"revert": _fit_revert, "hold": _fit_hold, "share_xc": _fit_masked}.get(
            method, lambda a, b, c: _fit(a, b, c, None))(z, origin, h)
        if f is None:
            return None
        return f.forecast * xa.reindex(months), f
    raise ValueError(spec)


def _exog_name(key: str) -> str:
    return {"ngdp": "ngdp_yoy", "cpi": "hicp_yoy", "rates": "rates"}.get(key, key)


def _exog_col(key: str | None, drivers: dict[str, pd.Series]) -> pd.Series | None:
    if key is None:
        return None
    name = _exog_name(key)
    if name not in drivers:
        return None
    s = drivers[name]
    # already lagged when coming from driver_path; lag here for raw monthly_drivers
    if s.attrs.get("lagged"):
        return s.rename(name)
    return s.shift(MACRO_LAG).rename(name)


def _transform(y: pd.Series, kind: str) -> pd.Series:
    if kind == "mom":
        return y.pct_change().replace([np.inf, -np.inf], np.nan)
    if kind == "diff":
        return y.diff()
    if kind == "yoy":
        return y.pct_change(12).replace([np.inf, -np.inf], np.nan)
    return y


def _stitch(fc: pd.Series, y: pd.Series, origin: pd.Timestamp, kind: str) -> pd.Series | None:
    out = {}
    level = float(y.loc[origin])
    for m in fc.index:
        v = float(fc.loc[m])
        if kind == "mom":
            if level <= 0 and v <= -1:
                return None
            level = level * (1.0 + v)
        elif kind == "diff":
            level = level + v
        elif kind == "yoy":
            base = y.get(m - pd.DateOffset(years=1), np.nan)
            if pd.isna(base):
                return None
            level = float(base) * (1.0 + v)
        else:
            level = v
        out[m] = level
    return pd.Series(out)


@dataclass(frozen=True)
class Fit:
    """One fitted component: the forecast plus what the Methodology page shows."""

    forecast: pd.Series
    exog: str | None
    beta: float | None
    beta_se: float | None
    params: dict[str, float]
    nobs: int
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp


def _fit(z: pd.Series, origin: pd.Timestamp, h: int,
         exog: pd.Series | None) -> Fit | None:
    """SARIMAX ``ORDER × SEASONAL`` on ``z`` up to ``origin``; None if it cannot fit.

    With ``exog`` this is a regression with seasonal-ARIMA errors,
    ``z_t = beta * x_t + u_t``, so the forecast moves by exactly ``beta * dx``
    when the future driver path moves by ``dx`` (the scenario engine relies on it).
    """
    hist = z.loc[:origin].dropna()
    if len(hist) < MIN_HIST:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    x_hist = x_fut = None
    if exog is not None:
        x = exog.reindex(hist.index.union(months))
        x_hist = pd.DataFrame({exog.name: x.reindex(hist.index)})
        ok = x_hist.dropna().index
        if len(ok) < MIN_HIST:
            return None
        hist = hist.loc[ok]
        x_hist = x_hist.loc[ok]
        # future: use the supplied path (authority); last-print only as last resort
        fut = x.reindex(months)
        if fut.isna().any():
            last = float(x.loc[:origin].dropna().iloc[-1])
            fut = fut.fillna(last)
        x_fut = pd.DataFrame({exog.name: fut})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                hist, exog=x_hist, order=ORDER, seasonal_order=SEASONAL,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
            fc = res.forecast(h, exog=x_fut)
    except Exception as exc:  # noqa: BLE001 - statsmodels raises many types; keep the reason
        log.warning("SARIMAX fit failed (n=%d, exog=%s): %s", len(hist),
                    None if exog is None else exog.name, exc)
        return None
    params = {str(k): float(v) for k, v in res.params.items()}
    name = None if exog is None else str(exog.name)
    beta = params.get(name) if name else None
    bse = float(res.bse[name]) if name and name in res.bse.index else None
    return Fit(
        forecast=pd.Series(np.asarray(fc), index=months),
        exog=name, beta=beta, beta_se=bse, params=params, nobs=int(res.nobs),
        sample_start=hist.index.min(), sample_end=hist.index.max(),
    )


def _sarimax(z: pd.Series, origin: pd.Timestamp, h: int,
             exog: pd.Series | None) -> pd.Series | None:
    f = _fit(z, origin, h, exog)
    return None if f is None else f.forecast


def fit_forecast(y: pd.Series, origin: pd.Timestamp, h: int,
                 exog: pd.Series | None = None, kind: str = "level") -> pd.Series | None:
    z = _transform(y, kind)
    fc = _sarimax(z, origin, h, exog)
    if fc is None:
        return None
    return _stitch(fc, y, origin, kind)


def _kind_for(cc: str, name: str, mode: str) -> str:
    if mode in ("level", "yoy"):
        return mode
    return "mom" if name in PCT_MOM.get(cc, ()) else "diff"


def _drivers_for(cc: str, origin: pd.Timestamp, h: int,
                 shock: dict[str, float] | None, use_authority: bool) -> dict[str, pd.Series]:
    if use_authority:
        from sovereign_cash.macro_authority import driver_path
        d = driver_path(cc, origin, h, shock=shock)
        # mark as already lagged
        for k in ("ngdp_yoy", "hicp_yoy"):
            if k in d:
                d[k].attrs["lagged"] = True
        return d
    return monthly_drivers(cc)


def _smly_path(y: pd.Series, origin: pd.Timestamp, h: int) -> pd.Series | None:
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    vals = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        if prev not in y.index or pd.isna(y.loc[prev]):
            return None
        vals.append(float(y.loc[prev]))
    return pd.Series(vals, index=months)


def authority_levels(cc: str, origin: pd.Timestamp, h: int,
                     shock: dict[str, float] | None = None) -> dict[str, pd.Series]:
    """Lagged driver levels on the authority path; ``shock`` as ``forecast``.

    ``{"ngdp": s}`` / ``{"hicp": s}``: s pp faster growth every year (as the YoY spec's shock);
    ``{"ngdp_level": s}`` / ``{"hicp_level": s}``: a one-off s% level shift. Both from origin+1.
    """
    from sovereign_cash.scenarios.engine import level_gap

    shock = shock or {}
    out = {}
    for key, skey in SHOCK_KEY.items():
        gap = None
        g, lv = float(shock.get(skey, 0.0)), float(shock.get(f"{skey}_level", 0.0))
        if g or lv:
            gap = [level_gap(g, "growth", i + 1) if g else 0.0 for i in range(h)]
            if lv:
                gap = [((1 + a / 100) * (1 + lv / 100) - 1) * 100 for a in gap]
        out[key] = level_path(cc, key, origin, h, gap=gap)
    return out


def forecast(cc: str, origin: pd.Timestamp, h: int = 12,
             frame: pd.DataFrame | None = None,
             drivers: dict[str, pd.Series] | None = None,
             mode: str = "level", with_exog: bool = True,
             use_authority: bool = False,
             shock: dict[str, float] | None = None,
             spec: str | None = None, levels: dict[str, pd.Series] | None = None) -> pd.DataFrame | None:
    """One origin: each SPEC line, then the deficit identity.

    Interest is allowed to fall back to same-month-last-year when the rates
    history is shorter than ``MIN_HIST`` (debt book starts mid-sample); other
    lines still fail the origin if their SARIMAX does not fit.
    """
    spec = spec_for(cc, spec)
    frame = components(cc) if frame is None else frame
    if drivers is None:
        drivers = _drivers_for(cc, origin, h, shock, use_authority)
    if spec != "yoy" and levels is None and use_authority:
        levels = authority_levels(cc, origin, h, shock)
    cols = {}
    for name, key0 in SPEC[cc]:
        y = frame[name].dropna()
        key = line_key(cc, name, key0) if spec != "yoy" else key0
        if spec != "yoy" and key in LEVEL_KEY and with_exog:
            x = (levels or {}).get(key)
            if x is None:
                x = level_drivers(cc)[LEVEL_KEY[key]].shift(MACRO_LAG)
            got = _fit_level(line_history(cc, name, y), origin, h, x, spec, line_method(cc, name))
            if got is None:
                return None
            cols[name] = got[0] + line_extra(cc, name, got[0].index)
            continue
        ex = _exog_col(key, drivers) if with_exog else None
        fc = fit_forecast(y, origin, h, ex, kind=_kind_for(cc, name, mode))
        if fc is None and name == "interest":
            fc = _smly_path(y, origin, h)
        if fc is None:
            return None
        cols[name] = fc
    out = pd.DataFrame(cols)
    out["cash_requirement"] = deficit(cc, {k: out[k] for k, _ in SPEC[cc]})
    return out


def fit_components(cc: str, origin: pd.Timestamp, h: int, *,
                   frame: pd.DataFrame | None = None,
                   drivers: dict[str, pd.Series] | None = None,
                   mode: str = "level", use_authority: bool = True,
                   spec: str | None = None) -> dict[str, Fit | None]:
    """Each SPEC line's fit exactly as ``forecast`` makes it (same data, drivers, kind)."""
    frame = components(cc) if frame is None else frame
    if drivers is None:
        drivers = _drivers_for(cc, origin, h, None, use_authority)
    out: dict[str, Fit | None] = {}
    spec = spec_for(cc, spec)
    levels = authority_levels(cc, origin, h) if spec != "yoy" else None
    for name, key0 in SPEC[cc]:
        y = frame[name].dropna()
        key = line_key(cc, name, key0) if spec != "yoy" else key0
        if spec != "yoy" and key in LEVEL_KEY:
            got = _fit_level(line_history(cc, name, y), origin, h, levels[key], spec, line_method(cc, name))
            out[name] = None if got is None else got[1]
            continue
        kind = _kind_for(cc, name, mode)
        out[name] = _fit(_transform(y, kind), origin, h, _exog_col(key, drivers))
    return out


def backtest(cc: str, step: int = 3, mode: str = "level",
             with_exog: bool = True, spec: str | None = None) -> pd.DataFrame:
    """Walk-forward cumulative errors. Exog = realised (no authority lookahead), for every spec alike."""
    frame = components(cc)
    y = deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()
    drivers = monthly_drivers(cc)
    lv = level_drivers(cc)
    levels = {k: lv[v].shift(MACRO_LAG) for k, v in LEVEL_KEY.items()}
    rows = []
    last = y.index.max()
    for i, origin in enumerate(y.index):
        if i % step:
            continue
        end = origin + pd.DateOffset(months=12)
        if end > last or len(y.loc[:origin]) < MIN_HIST:
            continue
        act = y.loc[origin + pd.offsets.MonthBegin(1):end]
        if len(act) != 12 or act.isna().any():
            continue
        fc = forecast(cc, origin, 12, frame, drivers, mode=mode, with_exog=with_exog,
                      spec=spec, levels=levels)
        if fc is None:
            continue
        smly = C.forecast(y, origin, 12, "smly")
        seas = C.forecast(y, origin, 12, "seas_mult_3")
        if seas is None:
            seas = C.forecast(y, origin, 12, "seas_add_3")
        if smly is None or seas is None:
            continue
        for h in C.HORIZONS:
            rows.append({
                "origin": origin, "h": int(h),
                "sarimax": float(fc["cash_requirement"].iloc[:h].sum()),
                "smly": float(smly.iloc[:h].sum()),
                "seas": float(seas.iloc[:h].sum()),
                "actual": float(act.iloc[:h].sum()),
            })
    return pd.DataFrame(rows)


def mae(bt: pd.DataFrame, cc: str = "fr") -> pd.DataFrame:
    if bt.empty:
        return bt
    pre_end = pd.Timestamp("2021-12-01") if cc == "it" else pd.Timestamp("2019-12-01")
    ends = [o + pd.DateOffset(months=int(h)) for o, h in zip(bt["origin"], bt["h"])]
    window = np.where(pd.to_datetime(ends) <= pre_end, "PRE",
                      np.where(bt["origin"] >= pd.Timestamp("2022-01-01"), "OOS", "mid"))
    bt = bt.assign(window=window)
    rows = []
    for (w, h), g in bt.groupby(["window", "h"]):
        rows.append({
            "window": w, "h": int(h), "n": len(g),
            "sarimax": float((g["sarimax"] - g["actual"]).abs().mean()),
            "smly": float((g["smly"] - g["actual"]).abs().mean()),
            "seas": float((g["seas"] - g["actual"]).abs().mean()),
        })
    return pd.DataFrame(rows)


def append_sx_bt(bt: pd.DataFrame, cc: str, *, mode: str = "level",
                 with_exog: bool = True, name: str = "sx_level_exog",
                 scale: float = 1.0) -> pd.DataFrame:
    """Append component SARIMAX rows into a cash backtest table.

    Origins match the ``smly`` cells already in ``bt``. ``scale`` converts the
    bn forecast into the table's units (IT uses EUR mn → ``scale=1000``).
    Extra columns on ``bt`` (e.g. ``touches_excl``) are copied from the smly row.
    """
    smly = bt[bt["model"] == "smly"]
    if smly.empty:
        return bt
    frame = components(cc)
    drivers = monthly_drivers(cc)
    cache: dict[pd.Timestamp, pd.Series | None] = {}
    rows = []
    extra = [c for c in bt.columns if c not in ("origin", "h", "model", "fc", "actual")]
    for origin, g in smly.groupby("origin"):
        if origin not in cache:
            fc = forecast(cc, origin, 12, frame, drivers, mode=mode, with_exog=with_exog)
            cache[origin] = None if fc is None else fc["cash_requirement"]
        path = cache[origin]
        for _, r in g.iterrows():
            h = int(r["h"])
            val = float(r["fc"]) if path is None else float(path.iloc[:h].sum()) * scale
            row = {"origin": origin, "h": h, "model": name, "fc": val, "actual": float(r["actual"])}
            for c in extra:
                row[c] = r[c]
            rows.append(row)
    if not rows:
        return bt
    return pd.concat([bt, pd.DataFrame(rows)[list(bt.columns)]], ignore_index=True)


def append_it_bt(bt: pd.DataFrame, x_mn: pd.Series, *, mode: str = "level",
                 with_exog: bool = True, name: str = "sx_level_exog") -> pd.DataFrame:
    """IT backtest is in EUR mn; ``x_mn`` kept for call-site compatibility."""
    del x_mn  # origins come from smly rows
    return append_sx_bt(bt, "it", mode=mode, with_exog=with_exog, name=name, scale=1000.0)


def live_path(cc: str, origin: pd.Timestamp, h: int = 12, *,
              mode: str = "level", with_exog: bool = True,
              use_authority: bool = True,
              shock: dict[str, float] | None = None, spec: str | None = None) -> pd.Series | None:
    """Monthly cash-requirement forecast in local-currency bn (authority drivers)."""
    fc = forecast(cc, origin, h, mode=mode, with_exog=with_exog,
                  use_authority=use_authority, shock=shock, spec=spec)
    return None if fc is None else fc["cash_requirement"]
