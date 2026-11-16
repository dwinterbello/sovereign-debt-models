"""France, Germany, Italy: component SARIMAX with macro regressors.

Each component is ``y_t = beta * x_t + u_t`` with seasonal-ARIMA errors, so a
change dx in the regressor path moves the forecast by exactly ``beta * dx``.

Specification note (shown on the Methodology page): x is the driver's YoY rate,
lagged two months, while y is a level. A sustained +s pp growth shock moves x
by s for good, so the component moves by a constant beta * s a month: it does
not compound. A one-off s% level shift moves x by s for twelve months only, so
the component returns to baseline after a year. Both follow from the model as
fitted; receipts proportional to the GDP level would need a log-level spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels.common import CCY, LINE_LABELS, N_YEARS, NAMES, default_onset, ts

COUNTRIES = ("fr", "de", "it")

# The baseline is our own forecast. Where the published measure is wider than the cash-account lines the component
# models cover, a perimeter line carries the rest, forecast by our own models too: Germany's special funds (the
# Germany board's run-rate model) and, for Italy, the State sector borrowing requirement's flows outside the Bank of
# Italy cash-account lines (their same-month average over the last three full years). France's SMB is the measure.
BOARD = Path(__file__).resolve().parents[2] / "derived"
MEASURE = {
    "fr": "Central government cash requirement",
    "de": "Federal cash requirement, core budget and special funds",
    "it": "State sector borrowing requirement (fabbisogno)",
}
PERIMETER = {
    "de": ("special_funds", "Special funds",
           "Sondervermögen (infrastructure and climate, Bundeswehr, others): outturn from the Bundesbank/BMF series on "
           "the Germany board, forecast by its run-rate model"),
    "it": ("perimeter", "Outside the cash-account lines",
           "MEF fabbisogno less the Bank of Italy cash-account lines: outturn measured, forecast as the same month's "
           "average over the last three full years"),
}
PERIMETER_YEARS = 3


def authority_plan(cc: str) -> dict | None:
    """The authority's own annual plan on (or nearest to) the board's measure, EUR bn, for the Methodology page."""
    import json

    data = Path(__file__).resolve().parents[2] / "data"
    try:
        if cc == "fr":
            j = json.loads((data / "fr" / "plan.json").read_text())["years"]
            return {"label": "PLF 2027 déficit à financer",
                    "years": {y: float(v["deficit_to_finance"]) for y, v in j.items() if v.get("deficit_to_finance")}}
        if cc == "de":
            j = json.loads((data / "de" / "plan.json").read_text())["years"]
            return {"label": "Planned net borrowing, core + special funds (Soll 2026, Regierungsentwurf 2027); "
                             "the board applies the usual execution shortfall to the core",
                    "years": {y: float(v["total"]) for y, v in j.items() if v.get("total")}}
        if cc == "it":
            from sovereign_cash.it import events as ev

            return {"label": "DFP 2026 settore statale (% of GDP × nominal GDP)",
                    "years": {str(y): v / 1000.0 for y, v in ev.plan_mn().items()}}
    except Exception:  # noqa: BLE001 - display only
        return None
    return None


def board_monthly(cc: str) -> pd.DataFrame | None:
    """The country cash board's monthly path (outturn and forecast), or None if it has not been built."""
    path = BOARD / f"{cc}_cash_monthly.parquet"
    if not path.exists():
        return None
    m = pd.read_parquet(path)
    m["month"] = pd.to_datetime(m["month"])
    return m.set_index("month").sort_index()


def perimeter(cc: str, model_total: pd.Series, board: pd.DataFrame, until: pd.Timestamp) -> pd.Series | None:
    """Monthly perimeter line through ``until``: outturn where measured, then our forecast.

    ``model_total`` is the cash-account identity in outturn (components), used for Italy's measured gap.
    """
    if cc not in PERIMETER:
        return None
    if cc == "de":
        sv = board["cash_requirement_sv_bn"].fillna(0.0)
        return sv.loc[:until]
    act = board.loc[~board["is_forecast"].astype(bool), "cash_requirement_bn"]
    gap = (act - model_total).dropna()
    last = gap.index.max()
    full_years = sorted(y for y in set(gap.index.year) if (gap.index.year == y).sum() == 12)[-PERIMETER_YEARS:]
    base = gap[gap.index.year.isin(full_years)]
    by_month = base.groupby(base.index.month).mean()
    fwd = pd.date_range(last + pd.offsets.MonthBegin(1), until, freq="MS")
    return pd.concat([gap, pd.Series([float(by_month[m.month]) for m in fwd], index=fwd)])


def book_interest(cc: str, book=None) -> pd.Series | None:
    """Our ISIN debt book's annual interest projection, EUR bn by year: the live book if given, else the debt
    build's saved country-year table (the same model; the nightly run writes it before the boards and the pack)."""
    if book is not None and cc in book.countries():
        return book.annual(cc, "interest_bn")
    path = BOARD / "debt_country_annual.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return d[d["cc"] == cc].set_index("year")["interest_bn"].astype(float)


def our_path(cc: str, origin: pd.Timestamp, h: int, interest: pd.Series | None,
             board: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """Our monthly forecast from ``origin``: component lines on the authority macro paths, interest from the
    debt book (``interest``, annual) in the line's usual month pattern, and the perimeter line where the
    published measure is wider. Column ``total`` is the published measure. Used by the scenario pack and by
    the Italy board past the years its short-term model covers, so both show the same numbers."""
    from sovereign_cash import monthly_components as M
    from sovereign_cash.yoy_components import IDENTITY, components, deficit

    fc = M.forecast(cc, origin, h, drivers=M._drivers_for(cc, origin, h, None, True), use_authority=True,
                    spec=M.spec_for(cc))
    if fc is None:
        raise RuntimeError(f"{cc}: component forecast failed")
    src = "component SARIMAX on the effective rate"
    comp = components(cc)
    if interest is not None:
        hist_i = comp["interest"].dropna()
        yrs = sorted(y for y in set(hist_i.index.year) if (hist_i.index.year == y).sum() == 12)[-3:]
        hi = hist_i[hist_i.index.year.isin(yrs)]
        share = hi.groupby(hi.index.month).sum() / hi.sum()
        if all(y in interest.index for y in set(fc.index.year)) and len(share) == 12:
            fc["interest"] = [float(interest[m.year]) * float(share[m.month]) for m in fc.index]
            src = "ISIN debt book projection, in the usual month pattern"
    fc["cash_requirement"] = sum(IDENTITY[cc][k] * fc[k] for k in IDENTITY[cc])
    fc["total"] = fc["cash_requirement"]
    board = board_monthly(cc) if board is None else board
    if board is not None and cc in PERIMETER:
        hist_total = deficit(cc, {k: comp[k] for k in IDENTITY[cc]}).dropna()
        per = perimeter(cc, hist_total, board, fc.index.max())
        fc[PERIMETER[cc][0]] = per.reindex(fc.index).fillna(0.0)
        fc["total"] = fc["total"] + fc[PERIMETER[cc][0]]
    return fc, src


def later_years(cc: str, months: pd.DatetimeIndex, open_year: int, column: str = "total") -> pd.Series:
    """Our path for the months after ``open_year`` (empty if none): what the cash boards use beyond the year in
    progress, so a board and the Shock scenarios page show the same forecast for those years. ``column``:
    'total' (the published measure) or 'cash_requirement' (the modelled lines only, e.g. Germany's core
    budget when the board adds the special funds itself).

    Legislated policy is added on top (same overlay as ``baseline.with_policy``): the component path is
    no-policy-change; without this the outer years on the cash board silently dropped every measure.
    """
    from sovereign_cash.policy.monthly import add_to_monthly
    from sovereign_cash.scenarios.unit_shocks import _origin

    later = months[months.year > open_year]
    if not len(later):
        return pd.Series(dtype=float)
    o = _origin(cc)
    h = (later.max().year - o.year) * 12 + later.max().month - o.month
    path, _ = our_path(cc, o, h, book_interest(cc))
    s = path[column].reindex(later)
    # Origin for netting = last month of the open year (policy starts with the first later month).
    pol_origin = pd.Timestamp(open_year, 12, 1)
    return add_to_monthly(cc, s, pol_origin)


def _driver_steps(cc: str, origin: pd.Timestamp, h: int, months: list[pd.Timestamp]) -> dict[str, list[int]]:
    """Months since a driver shock reaches each forecast month (lag included), read off the real path."""
    from sovereign_cash.macro_authority import driver_path

    out: dict[str, list[int]] = {}
    base = driver_path(cc, origin, h)
    for key, name, shock_key in (("ngdp", "ngdp_yoy", "ngdp"), ("cpi", "hicp_yoy", "hicp")):
        shocked = driver_path(cc, origin, h, shock={shock_key: 1.0})[name].reindex(months)
        diff = (shocked - base[name].reindex(months)).fillna(0.0).to_numpy()
        if not np.all(np.isclose(diff, 0.0) | np.isclose(diff, 1.0)):
            raise RuntimeError(f"{cc}/{key}: driver shock is not a 0/1 step: {diff}")
        t, steps = 0, []
        for v in diff:
            t = t + 1 if np.isclose(v, 1.0) else t
            steps.append(t)
        out[key] = steps
    return out


def aft_per_bp(years: list[int], onset: pd.Timestamp) -> list[float]:
    """AFT's +100bp triangle by calendar year from ``onset``, per bp (``interest_split`` holds the rule)."""
    from sovereign_cash import interest_split as IS

    tri = IS._aft_rate_triangle(1.0, onset)
    return [float(tri.get(y, 0.0)) for y in years]


def build(cc: str, book) -> dict:
    from sovereign_cash import interest_split as IS
    from sovereign_cash import monthly_components as M
    from sovereign_cash.macro_authority import annual_authority
    from sovereign_cash.scenarios import unit_shocks as U
    from sovereign_cash.yoy_components import IDENTITY, SPEC

    origin = U._origin(cc)
    y0 = int(origin.year)
    years = list(range(y0, y0 + N_YEARS))
    last = pd.Timestamp(f"{years[-1]}-12-01")
    h = max(12, (last.year - origin.year) * 12 + (12 - origin.month) + 1)  # as unit_shocks.run

    spec = M.spec_for(cc)
    drivers = M._drivers_for(cc, origin, h, None, True)
    # the debt build's saved projection first: the Italy board reads the same one, so the two agree exactly
    bi = book_interest(cc)
    fc, interest_src = our_path(cc, origin, h, bi if bi is not None else book_interest(cc, book))
    fits = M.fit_components(cc, origin, h, drivers=drivers, spec=spec)
    months = list(fc.index)
    month_period = [years.index(m.year) if m.year in years else -1 for m in months]

    def annual(s: pd.Series) -> list[float]:
        return [float(s[s.index.year == y].sum()) for y in years]

    signs = IDENTITY[cc]
    steps = _driver_steps(cc, origin, h, months) if spec == "yoy" else None
    channels, params = [], {}
    how = {"share": f"SARIMAX{M.ORDER}x{M.SEASONAL} on the share (a change in the share persists)",
           "share_xc": f"SARIMAX{M.ORDER}x{M.SEASONAL} on the share, Covid months left out (a change persists)",
           "revert": "month means + AR(1) on the share (the share returns to its usual level)",
           "hold": "last twelve months' share, month by month"}
    for name, key0 in SPEC[cc]:
        f = fits.get(name)
        key = M.line_key(cc, name, key0) if spec != "yoy" else key0
        method = M.line_method(cc, name) if spec != "yoy" else "yoy"
        level_exog = M.LEVEL_KEY.get(key) if spec != "yoy" else None
        params[name] = None if f is None else {
            "exog": level_exog or f.exog, "beta": f.beta, "beta_se": f.beta_se, "nobs": f.nobs,
            "sample": [ts(f.sample_start), ts(f.sample_end)], "params": f.params,
            "spec": spec if key in ("ngdp", "cpi") else "yoy", "method": method,
        }
        if key not in ("ngdp", "cpi") or f is None:
            continue
        if spec != "yoy":
            eps = 1.0 if spec == "ratio" else float(f.beta)
            channels.append({
                "id": f"{cc}_{name}_{key}", "type": "elasticity_monthly", "driver": key, "line": name,
                # grants added back to a line do not scale with its driver: the base is the modelled part only
                "elasticity": eps, "base": [float(v) for v in fc[name] - M.line_extra(cc, name, fc.index)],
                "tau": [max(0, i + 1 - M.MACRO_LAG) for i in range(len(months))],
                "method": (f"{name} as a share of the {'nominal GDP' if key == 'ngdp' else 'HICP'} level "
                           f"(lag {M.MACRO_LAG}m), {how[method]}: the line moves in proportion to the level"
                           if spec == "ratio" else
                           f"SARIMAX on log {name}, elasticity {eps:.2f} to the log level"),
                "badge": "SARIMAX, level",
                "source": "sovereign_cash.monthly_components (LEVEL_SPEC)",
            })
            continue
        if f.beta is None:
            continue
        # The regressor is a YoY rate. A sustained growth shock of s pp raises it by s for
        # good; a one-off level shift of s% raises it by s for 12 months, then the YoY is
        # back to baseline. Each month's response is beta per pp of regressor change.
        unit_level = [0.0] * len(years)
        unit_growth = [0.0] * len(years)
        for t, p in zip(steps[key], month_period):
            if p < 0 or t <= 0:
                continue
            unit_growth[p] += float(f.beta)
            if t <= 12:
                unit_level[p] += float(f.beta)
        channels.append({
            "id": f"{cc}_{name}_{key}", "type": "linear", "driver": key, "line": name,
            "per_unit": {"level": unit_level, "growth": unit_growth},
            "beta": float(f.beta), "beta_se": f.beta_se,
            "method": (f"SARIMAX{M.ORDER}x{M.SEASONAL} on {name}; regressor {f.exog} "
                       f"(YoY, lag {M.MACRO_LAG}m); Δ = β·Δx summed over the year's months"),
            "badge": "SARIMAX β",
            "source": "sovereign_cash.monthly_components",
        })

    default_methods: dict[str, str] = {}
    if book is not None and (ch := book.channel(cc, years)) is not None:
        channels.append(ch)
        channels += book.curve_channels(cc, years)
        default_methods["rates"] = "book"
    if cc == "fr" and IS.FR_AUTH.exists():
        per = aft_per_bp(years, default_onset())
        channels.append({
            "id": "fr_rates_aft", "type": "linear", "driver": "rates", "line": "interest",
            "method_group": "authority", "per_unit": {"level": per, "growth": per},
            "method": ("AFT published +100bp triangle (1y/5y/9y): year k after the shock costs the triangle at k "
                       "(straight lines between points), spread over its months from the same start date as the "
                       "book; per bp"),
            "badge": "AFT triangle",
            "source": "sovereign_cash/data/fr/authority.json rate_shock.triangle",
        })

    lines = [{"id": k, "label": LINE_LABELS.get(k, k), "sign": int(v)} for k, v in signs.items()]
    base_lines = {k: annual(fc[k]) for k in signs}
    measure, board_full = "Central government cash requirement", None
    board = board_monthly(cc)
    if board is not None:
        if cc in PERIMETER:
            pid, label, note = PERIMETER[cc]
            lines.append({"id": pid, "label": label, "sign": 1, "perimeter": True, "note": note})
            base_lines[pid] = annual(fc[pid])
        cr = board["cash_requirement_bn"]
        board_full = [float(cr[cr.index.year == y].sum()) for y in years]
        measure = MEASURE[cc]
    total = [sum(int(ln["sign"]) * base_lines[ln["id"]][i] for ln in lines) for i in range(len(years))]
    base_src = (f"our component models on the authority macro paths (receipts and spending: shares of the GDP "
                f"and price levels; interest: {interest_src})")
    # Each driven line's annual share of its driver level, outturn then forecast (the Methodology page plots it, so
    # 'share', 'share, Covid out' and 'hold' can be seen doing what they say). Grants are excluded, as modelled.
    from sovereign_cash.yoy_components import components as _components

    comp_all = _components(cc)
    lv = M.authority_levels(cc, origin, h)
    shares = {}
    for name, key0 in SPEC[cc]:
        key = M.line_key(cc, name, key0)
        if key not in ("ngdp", "cpi"):
            continue
        y_h = M.line_history(cc, name, comp_all[name].dropna()).loc[:origin]
        y_f = fc[name] - M.line_extra(cc, name, fc.index)
        y_all = pd.concat([y_h, y_f])
        x = lv[key]
        yrs = [yy for yy in range(y0 - 10, years[-1] + 1)
               if (y_all.index.year == yy).sum() == 12 and x[x.index.year == yy].notna().sum() == 12]
        sh = {str(yy): float(y_all[y_all.index.year == yy].sum() / x[x.index.year == yy].mean()) for yy in yrs}
        shares[name] = {"driver": key, "method": M.line_method(cc, name), "share": sh,
                        "first_forecast_year": str(y0), "label": LINE_LABELS.get(name, name)}
    ngdp_a, cpi_a, src = annual_authority(cc)
    entry = {
        "cc": cc, "name": NAMES[cc], "ccy": CCY[cc], "unit": f"{CCY[cc]} bn",
        "measure": measure, "period_kind": "calendar year",
        "periods": [str(y) for y in years], "origin": ts(origin),
        "months": [m.strftime("%Y-%m") for m in months], "month_period": month_period,
        "onset": ts(origin + pd.offsets.MonthBegin(1)),
        "lines": lines,
        "baseline": {"total": total, "lines": base_lines, "source": base_src,
                     "model_total": annual(fc["cash_requirement"]),
                     "authority": authority_plan(cc), "board_full_year": board_full},
        "channels": channels, "default_methods": default_methods,
        "modes": {"ngdp": ["level", "growth"], "cpi": ["level", "growth"]},
        "spec": spec,
        "drivers_baseline": {
            "ngdp": {str(y): float(ngdp_a[y]) for y in years if y in ngdp_a.index},
            "cpi": {str(y): float(cpi_a[y]) for y in years if y in cpi_a.index},
        },
        "sources": {"ngdp": src["ngdp"], "cpi": src["cpi"]},
        "model": {"family": "component SARIMAX", "spec": spec, "order": list(M.ORDER), "seasonal": list(M.SEASONAL),
                  "macro_lag_months": M.MACRO_LAG, "min_hist": M.MIN_HIST, "fits": params,
                  "identity": signs, "covid_years_excluded": False, "shares": shares},
    }
    if book is not None and (r := book.feedback_rate(cc)) is not None:
        entry["feedback"] = {"rate_pct": [r] * len(years), "source": "debt snapshot marginal new-issue rate"}
    return entry
