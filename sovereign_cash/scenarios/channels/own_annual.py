"""Our own forecast for the United Kingdom, United States, Spain, Sweden, Norway and Japan (Nov 2026).

Same economics as the France/Germany/Italy component models, at the frequency each country's long history exists:

    receipts_p = r_p * Y_p        r: share of nominal GDP (Y), carried forward by METHOD
    primary_p  = q_p * P_p        q: share of the price level (P), carried forward by METHOD
    interest_p = interest_T * B_p / B_T   our ISIN debt book's projection B, from the last outturn T
    measure_p  = primary_p + interest_p - receipts_p

on each country's own published measure and outturn history:
    United Kingdom  PSNB ex public sector banks, fiscal years (ONS public sector finances, monthly from 1997)
    United States   federal unified deficit, fiscal years (OMB/Treasury via FRED: receipts, outlays, interest)
    Spain, Sweden, Norway, Japan   general government deficit, calendar years (AMECO revenue, balance and gross
                                   interest as % of GDP, times AMECO GDP)

Y and P follow the authority macro paths (AMECO nominal GDP; IMF WEO real × house CPI through 2030, WEO CPI
beyond) the scenario page shocks. A shock to the level of Y or P moves each line by its share: unit elasticity,
exact, as for France, Germany and Italy — except UK income tax + NICs, which use a frozen-threshold fiscal-drag
elasticity to average earnings (below).

United Kingdom receipts (fiscal drag). Income tax and NICs are progressive and their main thresholds are frozen
in cash through April 2028 under prior law (Budget 2025's extension to April 2031 is costed in the policy line
as "Personal Tax" / secondary NI threshold, so the model must not assume the extension). While thresholds are
frozen, OBR EFO October 2026 para 4.14: +1pp earnings → about +£7bn of IT+NICs (~elasticity 1.3) against about
+£4.5bn if thresholds rose with CPI. The March 2025 ready reckoner's Earnings-level row is the same order
(£6–7bn). Split last-outturn receipts into an IT+NIC share and the rest; the rest stays unit-elastic to nominal
GDP; IT+NICs grow with the earnings index at ε=1.3 in freeze years and ε=1 thereafter (policy adds the
extension). Earnings: complete ONS AWE fiscal years in history / the backtest; OBR AWE determinants forward,
else NGDP growth.

How each share is carried forward ('persist': last outturn share; 'revert': back toward its sample mean at its
estimated annual persistence) is chosen per line on a walk-forward test, one to three years ahead from every
origin since 2005, conditional on realised GDP and prices (``backtest``; the Equations page shows it). For UK
receipts the fiscal-drag rule is scored alongside persist/revert and replaces persist when it wins the gate.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels.common import CCY, N_YEARS, NAMES, tau, ts

log = logging.getLogger(__name__)
SC = Path(__file__).resolve().parents[2]
DEBT = SC / "data" / "debt"
MACRO_PANEL = SC.parent / "correlations" / "derived" / "macro_panel_monthly.csv"
AUTH_UK = SC / "data" / "uk" / "authority.json"
COUNTRIES = ("uk", "us", "es", "se", "no", "jp")
COVID = {2020, 2021}
LAG = 2  # months, as the monthly component models
FIRST_ORIGIN = 2005
MEASURE = {
    "uk": "Public sector net borrowing (ex banks)",
    "us": "Federal deficit",
    "es": "General government deficit", "se": "General government deficit",
    "no": "General government deficit", "jp": "General government deficit",
}
LINES = {
    "uk": [("receipts", "Receipts", -1), ("primary", "Spending ex interest", 1), ("interest", "Debt interest", 1)],
    "us": [("receipts", "Revenues", -1), ("primary", "Outlays ex interest", 1),
           ("interest", "Interest on Treasury debt (gross)", 1)],
}
DEFAULT_LINES = [("receipts", "Revenue", -1), ("primary", "Primary spending", 1), ("interest", "Interest", 1)]
# Chosen on ``backtest`` (derived/own_annual_backtest.json). 'persist' unless 'revert' is clearly better (at least
# 10% lower mean absolute error over one to three years on that line): the default is the one the France
# component model uses, so a country only differs where its own history says so.
METHOD: dict[tuple[str, str], str] = {}
REVERT_GAIN = 0.10
BACKTEST_PATH = SC / "derived" / "own_annual_backtest.json"
# UK IT+NIC fiscal drag (OBR EFO Oct 2026 para 4.14; RR Earnings-level row consistent).
EPS_FROZEN = 1.3
EPS_INDEXED = EPS_FROZEN * 4.5 / 7.0  # CPI-indexed thresholds counterfactual in the same paragraph
FREEZE_FIRST = 2021  # FY 2021-22: personal allowance / higher-rate threshold frozen in cash
FREEZE_LAST_PRIOR = 2027  # FY 2027-28: last year under AS 2022 (freeze to April 2028); Budget 2025 extension is policy
ITNIC_SHARE_FALLBACK = 0.435  # 2025-26 IT+NIC / receipts when the authority file has no year
DRAG_SOURCE = ("OBR Economic and fiscal outlook, October 2026, paragraph 4.14 "
               "(+1pp earnings → ~£7bn IT+NICs under frozen thresholds vs ~£4.5bn if CPI-indexed); "
               "OBR ready reckoner March 2025, Earnings level → PAYE income tax and NICs")


# ---------------------------------------------------------------- periods
def fiscal(cc: str) -> str | None:
    return {"uk": "uk", "us": "us"}.get(cc)


def label(cc: str, y: int) -> str:
    f = fiscal(cc)
    return f"{y}-{str(y + 1)[2:]}" if f == "uk" else f"FY{y}" if f == "us" else str(y)


def weights(cc: str, y: int) -> dict[int, float]:
    """Calendar-year weights of period y (UK: Apr y - Mar y+1; US: Oct y-1 - Sep y)."""
    f = fiscal(cc)
    if f == "uk":
        return {y: 0.75, y + 1: 0.25}
    if f == "us":
        return {y - 1: 0.25, y: 0.75}
    return {y: 1.0}


def period_months(cc: str, y: int) -> pd.DatetimeIndex:
    f = fiscal(cc)
    start = pd.Timestamp(f"{y}-04-01") if f == "uk" else pd.Timestamp(f"{y - 1}-10-01") if f == "us" \
        else pd.Timestamp(f"{y}-01-01")
    return pd.date_range(start, periods=12, freq="MS")


def to_period(cc: str, cy: pd.Series, years) -> pd.Series:
    out = {}
    for y in years:
        w = weights(cc, y)
        if all(k in cy.index and np.isfinite(cy[k]) for k in w):
            out[y] = float(sum(cy[k] * v for k, v in w.items()))
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------- macro: nominal GDP and price level, calendar
@lru_cache(maxsize=16)
def macro_cy(cc: str) -> tuple[pd.Series, pd.Series, str]:
    """Nominal GDP (local bn) and a price index (1995 = 1), calendar years, outturn then the authority path.

    AMECO GDP levels through ``first_forecast_year() - 1``; every later year compounds
    ``annual_authority`` NGDP YoY (RESF/DFP pins, else WEO real × house CPI annuals). Price index
    compounds the same authority CPI YoY (house path through 2030, WEO fill). Numbers only — the
    inflation stack stays behind ``own_inflation`` / ``annual_authority``.
    """
    from sovereign_cash.macro_authority import annual_authority, first_forecast_year

    a = pd.read_parquet(DEBT / "ameco.parquet")
    gdp = a[(a["country"] == cc) & (a["var"] == "gdp_lcu_bn")].set_index("year")["value"].astype(float).sort_index()
    w = pd.read_parquet(DEBT / "weo.parquet")
    w = w[w["country"] == cc]
    cpi_weo = w[w["var"] == "cpi"].set_index("year")["value"].astype(float).sort_index()
    real = w[w["var"] == "real_growth"].set_index("year")["value"].astype(float).sort_index()
    ngdp_a, cpi_a, src = annual_authority(cc)
    # Drop AMECO's own forecast years so RESF/DFP/house CPI YoY actually move the share engine.
    anchor = first_forecast_year() - 1
    if anchor in gdp.index:
        gdp = gdp.loc[:anchor].copy()
    elif len(gdp):
        gdp = gdp.loc[: int(gdp.index.max())].copy()
    last = int(gdp.index.max())
    end = max(
        int(ngdp_a.index.max()) if len(ngdp_a) else last,
        int(cpi_a.index.max()) if len(cpi_a) else last,
        int(cpi_weo.index.max()) if len(cpi_weo) else last,
        last,
    )
    for y in range(last + 1, end + 1):
        g = float(ngdp_a.loc[y]) if y in ngdp_a.index and np.isfinite(ngdp_a.loc[y]) else np.nan
        if not np.isfinite(g):
            g = float(real.get(y, np.nan)) + float(cpi_weo.get(y, np.nan))
        if not np.isfinite(g):
            break
        gdp.loc[y] = float(gdp.loc[y - 1]) * (1.0 + g / 100.0)
    # Price level: authority CPI (house / RESF) with WEO fill.
    p = {1995: 1.0}
    for y in range(1996, end + 1):
        g = float(cpi_a.loc[y]) if y in cpi_a.index and np.isfinite(cpi_a.loc[y]) else np.nan
        if not np.isfinite(g):
            g = float(cpi_weo.get(y, 0.0))
        p[y] = p[y - 1] * (1.0 + g / 100.0)
    return gdp.sort_index(), pd.Series(p), f"{src['ngdp']}; {src['cpi']}"


# ---------------------------------------------------------------- history on each country's measure
def _ons_monthly(code: str) -> pd.Series:
    mon = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                       "NOV", "DEC"], start=1)}
    j = json.loads((SC / "data" / "uk" / "ons" / f"{code.lower()}.json").read_text())
    s = {}
    for m in j["months"]:
        y, mm = m["date"].split()
        s[pd.Timestamp(year=int(y), month=mon[mm], day=1)] = float(m["value"]) / 1000.0
    return pd.Series(s).sort_index()


def uk_monthly() -> pd.DataFrame:
    """PSNB ex and its lines, monthly, £bn: receipts = current budget surplus + current spending + depreciation."""
    cb, ni, cur, dep, intr = (_ons_monthly(c) for c in ("JW2T", "JW2Z", "JW2Q", "JW2S", "JW2P"))
    d = pd.DataFrame({"cb": cb, "ni": ni, "cur": cur, "dep": dep, "interest": intr}).dropna()
    d["total"] = -(d["cb"] + d["ni"])
    d["receipts"] = d["cb"] + d["cur"] + d["dep"]
    d["primary"] = d["total"] - d["interest"] + d["receipts"]
    return d[["receipts", "primary", "interest", "total"]]


@lru_cache(maxsize=16)
def history(cc: str) -> pd.DataFrame:
    """Annual outturn on the measure: receipts, primary, interest, total (local bn), by period start year."""
    if cc == "uk":
        m = uk_monthly()
        fy = m.index.year.where(m.index.month >= 4, m.index.year - 1)
        g = m.groupby(fy)
        n = g.size()
        return g.sum()[n == 12]
    if cc == "us":
        from sovereign_cash.scenarios.history import _fred

        d = _fred(["FYFR", "FYONET", "FYOINT"]) / 1000.0
        d = d.dropna()
        out = pd.DataFrame({"receipts": d["FYFR"], "interest": d["FYOINT"]})
        out["total"] = d["FYONET"] - d["FYFR"]
        out["primary"] = d["FYONET"] - d["FYOINT"]
        return out[["receipts", "primary", "interest", "total"]].loc[1995:]
    # one source for all four: AMECO revenue, balance and gross interest, % of GDP, times AMECO GDP. Gross
    # interest, because our debt book projects gross interest (the WEO's is net: negative for Norway).
    a = pd.read_parquet(DEBT / "ameco.parquet")
    a = a[a["country"] == cc].pivot_table(index="year", columns="var", values="value")
    gdp, _, _ = macro_cy(cc)
    last = int(pd.Timestamp.now().year) - 1
    a = a.loc[1995:last]
    y = gdp.reindex(a.index)
    out = pd.DataFrame({"receipts": a["revenue_pct"] * y / 100.0, "interest": a["interest_gross_pct"] * y / 100.0,
                        "total": -a["balance_pct"] * y / 100.0})
    out["primary"] = out["total"] - out["interest"] + out["receipts"]
    return out[["receipts", "primary", "interest", "total"]].dropna()


def levels(cc: str, years) -> tuple[pd.Series, pd.Series]:
    gdp, p, _ = macro_cy(cc)
    return to_period(cc, gdp, years), to_period(cc, p, years)


# ---------------------------------------------------------------- UK fiscal drag (IT + NICs × earnings)
def itnic_eps(year: int) -> float:
    """Elasticity of UK income tax + NICs to average earnings in fiscal year ``year``.

    Frozen-threshold 1.3 from FREEZE_FIRST through FREEZE_LAST_PRIOR (prior law to April 2028). From the next
    fiscal year thresholds would have risen with CPI under prior law: keep ε=1 (unit to earnings, same as the
    rest of the GDP-share model) so Budget 2025's threshold extension (policy line "Personal Tax" / NI secondary
    threshold) supplies the freeze-extension uplift and is not double-counted. The OBR's ~0.84 indexed figure is
    the marginal ready-reckoner response under indexation, not a level-path share to put under the GDP model.
    Before the freeze, ε=1 so the rule collapses toward the plain GDP share when earnings track nominal GDP.
    """
    if FREEZE_FIRST <= year <= FREEZE_LAST_PRIOR:
        return EPS_FROZEN
    return 1.0


@lru_cache(maxsize=1)
def _awe_fy() -> pd.Series:
    """ONS AWE total index, complete fiscal-year means only (April–March, 12 months)."""
    if not MACRO_PANEL.exists():
        return pd.Series(dtype=float)
    mp = pd.read_csv(MACRO_PANEL, parse_dates=["date"]).set_index("date").sort_index()
    if "awe_total_index" not in mp.columns:
        return pd.Series(dtype=float)
    s = mp["awe_total_index"].astype(float).dropna()
    fy = s.index.year.where(s.index.month >= 4, s.index.year - 1)
    g = s.groupby(fy)
    return g.mean()[g.size() >= 12].sort_index()


@lru_cache(maxsize=1)
def _obr_awe_pct() -> dict[int, float]:
    """OBR EFO AWE YoY % by fiscal-year start, from authority.json."""
    if not AUTH_UK.exists():
        return {}
    d = json.loads(AUTH_UK.read_text()).get("determinants", {}).get("awe_pct") or {}
    return {int(y): float(v) for y, v in d.items()}


@lru_cache(maxsize=1)
def _itnic_share_by_year() -> dict[int, float]:
    """IT + NICs as a share of current receipts, from OBR category levels in authority.json."""
    if not AUTH_UK.exists():
        return {}
    cat = json.loads(AUTH_UK.read_text()).get("categories_bn") or {}
    it, ni, rec = cat.get("income_tax_bn") or {}, cat.get("nics_bn") or {}, cat.get("receipts_current_receipts") or {}
    out = {}
    for y in set(it) | set(ni) | set(rec):
        r = float(rec.get(y) or rec.get(str(y)) or 0.0)
        if r <= 0:
            continue
        out[int(y)] = (float(it.get(y) or it.get(str(y)) or 0.0)
                       + float(ni.get(y) or ni.get(str(y)) or 0.0)) / r
    return out


def itnic_share(year: int) -> float:
    by = _itnic_share_by_year()
    if year in by:
        return by[year]
    if by:
        earlier = [y for y in by if y <= year]
        return by[max(earlier)] if earlier else ITNIC_SHARE_FALLBACK
    return ITNIC_SHARE_FALLBACK


def earnings_index(years) -> pd.Series:
    """Earnings index by UK fiscal-year start.

    Complete ONS AWE fiscal years for history; from the first incomplete/future year, OBR AWE determinants
    (authority.json), else nominal GDP growth. Never uses a partial in-progress AWE year as a growth rate.
    """
    years = list(years)
    awe = _awe_fy()
    obr = _obr_awe_pct()
    Y, _ = levels("uk", years)
    last_complete = int(awe.index.max()) if len(awe) else None
    out: dict[int, float] = {}
    base_y = min(years)
    if last_complete is not None and base_y <= last_complete and base_y in awe.index:
        out[base_y] = float(awe.loc[base_y])
    elif last_complete is not None and last_complete < base_y:
        lvl = float(awe.loc[last_complete])
        for y in range(last_complete + 1, base_y + 1):
            g = obr.get(y)
            if g is None and y in Y.index and y - 1 in Y.index and Y[y - 1]:
                g = 100.0 * (float(Y[y]) / float(Y[y - 1]) - 1.0)
            lvl *= 1.0 + (g or 0.0) / 100.0
        out[base_y] = lvl
    else:
        out[base_y] = float(awe.loc[base_y]) if last_complete is not None and base_y in awe.index else 1.0
    for y in years:
        if y in out:
            continue
        prev = out[y - 1]
        if last_complete is not None and y <= last_complete and y in awe.index and y - 1 in awe.index:
            out[y] = prev * float(awe.loc[y]) / float(awe.loc[y - 1])
            continue
        g = obr.get(y)
        if g is None and y in Y.index and y - 1 in Y.index and Y[y - 1]:
            g = 100.0 * (float(Y[y]) / float(Y[y - 1]) - 1.0)
        out[y] = prev * (1.0 + (g or 0.0) / 100.0)
    return pd.Series({y: out[y] for y in years}, dtype=float)


def receipts_fiscal_drag(receipts_T: float, T: int, Y: pd.Series, E: pd.Series, years) -> list[float]:
    """Receipts for each year in ``years`` from origin outturn ``T``: other unit-elastic to GDP, IT+NIC to earnings^ε.

    Through FREEZE_LAST_PRIOR, IT+NICs compound with earnings at ε=1.3. Afterwards (prior law would have
    indexed thresholds) that elevated level grows with nominal GDP, matching the rest of the share model;
    Budget 2025's extension then adds only via the policy line.
    """
    s = itnic_share(T)
    it0 = receipts_T * s
    oth0 = receipts_T * (1.0 - s)
    y0, e0 = float(Y[T]), float(E[T])
    out = []
    for y in years:
        it = it0
        e_prev, y_prev = e0, y0
        for t in range(T + 1, y + 1):
            e_t, y_t = float(E[t]), float(Y[t])
            if itnic_eps(t) == EPS_FROZEN:
                it *= (e_t / e_prev) ** EPS_FROZEN if e_prev > 0 else 1.0
            else:
                # unit to GDP: hold the IT+NIC/GDP ratio reached under the freeze
                it *= y_t / y_prev if y_prev > 0 else 1.0
            e_prev, y_prev = e_t, y_t
        out.append(oth0 * float(Y[y]) / y0 + it)
    return out


# ---------------------------------------------------------------- shares and how they are carried forward
def _rho_mean(s: pd.Series) -> tuple[float, float]:
    """Sample mean and annual persistence of a share (Covid years left out of both)."""
    x = s[~s.index.isin(COVID)]
    mu = float(x.mean())
    d = (s - mu)
    pairs = [(d[t - 1], d[t]) for t in s.index if t - 1 in s.index and t not in COVID and t - 1 not in COVID]
    if len(pairs) < 5:
        return mu, 1.0
    a, b = np.array(pairs).T
    rho = float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) > 0 else 1.0
    return mu, float(np.clip(rho, 0.0, 1.0))


def carry(s: pd.Series, origin: int, h: int, method: str) -> list[float]:
    """Share for origin+1..origin+h from outturn through ``origin``."""
    hist = s.loc[:origin].dropna()
    last = float(hist.iloc[-1])
    if method == "revert":
        mu, rho = _rho_mean(hist)
        return [mu + rho ** k * (last - mu) for k in range(1, h + 1)]
    return [last] * h


def method(cc: str, line: str) -> str:
    # UK receipts: always the OBR frozen-threshold rule, even if a stale backtest file still says persist.
    if cc == "uk" and line == "receipts":
        return "fiscal_drag"
    return METHOD.get((cc, line), "persist")


# ---------------------------------------------------------------- walk-forward test
def backtest(cc: str) -> dict:
    """One to three years ahead from each origin since 2005, given realised GDP and prices; errors in local bn."""
    h = history(cc)
    yrs = list(h.index)
    Y, P = levels(cc, yrs)
    E = earnings_index(yrs) if cc == "uk" else None
    r, q, qy = h["receipts"] / Y, h["primary"] / P, h["primary"] / Y
    rows = []
    for o in [y for y in yrs if FIRST_ORIGIN <= y < yrs[-1]]:
        for line, s, X, m0 in (("receipts", r, Y, None), ("primary", q, P, None), ("primary", qy, Y, "persist_gdp")):
            methods = ((m0,) if m0 else
                       (("persist", "revert", "fiscal_drag") if cc == "uk" and line == "receipts"
                        else ("persist", "revert")))
            for m in methods:
                if m == "fiscal_drag":
                    need = [o + k for k in (1, 2, 3) if o + k in h.index and o + k in Y.index]
                    if not need or E is None or o not in E.index or any(t not in E.index for t in need):
                        continue
                    fc_levels = receipts_fiscal_drag(float(h.at[o, "receipts"]), o, Y, E, need)
                    by_t = dict(zip(need, fc_levels))
                    for k in (1, 2, 3):
                        t = o + k
                        if t not in by_t:
                            continue
                        rows.append({"origin": o, "year": t, "h": k, "line": line, "method": m,
                                     "err": float(by_t[t] - h.at[t, line]), "gdp": float(Y[t])})
                    continue
                fc = carry(s, o, 3, "persist" if m == "persist_gdp" else m)
                for k in (1, 2, 3):
                    t = o + k
                    if t not in h.index or t not in X.index:
                        continue
                    rows.append({"origin": o, "year": t, "h": k, "line": line, "method": m,
                                 "err": float(fc[k - 1] * X[t] - h.at[t, line]), "gdp": float(Y[t])})
    d = pd.DataFrame(rows)
    out = {"lines": {}, "n_origins": int(d["origin"].nunique()) if len(d) else 0,
           "first_origin": int(d["origin"].min()) if len(d) else None}
    for line in ("receipts", "primary"):
        x = d[(d["line"] == line) & (~d["year"].isin(COVID))]
        if cc == "uk" and line == "receipts":
            cands = ("persist", "revert", "fiscal_drag")
        elif line == "primary":
            cands = ("persist", "revert", "persist_gdp")
        else:
            cands = ("persist", "revert")
        sc = {m: {f"h{k}": round(float(x[(x["method"] == m) & (x["h"] == k)]["err"].abs().mean()), 3)
                  for k in (1, 2, 3)} for m in cands if len(x[x["method"] == m])}
        for m in sc:
            sc[m]["mean"] = round(float(np.mean([sc[m][f"h{k}"] for k in (1, 2, 3)])), 3)
            sc[m]["mean_pct_gdp"] = round(float((x[x["method"] == m]["err"].abs() / x[x["method"] == m]["gdp"]).mean() * 100), 3)
            sc[m]["bias_pct_gdp"] = round(float((x[x["method"] == m]["err"] / x[x["method"] == m]["gdp"]).mean() * 100), 3)
        # the default (persist on its own driver) unless another rule is at least REVERT_GAIN better.
        best = min(sc, key=lambda m: sc[m]["mean"])
        pick = best if sc[best]["mean"] < (1 - REVERT_GAIN) * sc["persist"]["mean"] else "persist"
        out["lines"][line] = {"scores": sc, "chosen": pick}
        if line == "receipts" and "fiscal_drag" in sc:
            # freeze-era slice for the Equations page. Always ship fiscal_drag: OBR structural prior
            # (EFO Oct 2026 para 4.14). A soft freeze-era MAE gate dropped it after a Nov-2026 refresh
            # (0.2bn edge) and wiped it off the live site.
            xf = x[x["year"] >= FREEZE_FIRST]
            freeze_sc = {}
            for m in sc:
                xm = xf[xf["method"] == m]
                if len(xm):
                    freeze_sc[m] = {
                        "mean": round(float(xm["err"].abs().mean()), 3),
                        "bias_pct_gdp": round(float((xm["err"] / xm["gdp"]).mean() * 100), 3),
                        "n": int(len(xm)),
                    }
            pick = "fiscal_drag"
            out["lines"][line]["chosen"] = pick
            out["lines"][line]["fiscal_drag"] = {
                "eps_frozen": EPS_FROZEN, "eps_indexed": round(EPS_INDEXED, 4),
                "freeze_first": FREEZE_FIRST, "freeze_last_prior_law": FREEZE_LAST_PRIOR,
                "itnic_share_origin_fallback": ITNIC_SHARE_FALLBACK, "source": DRAG_SOURCE,
                "freeze_era": freeze_sc, "shipped": "structural (OBR para 4.14)",
            }
    # the measure ex interest, with each line on its chosen rule
    tot = []
    for o in sorted(d["origin"].unique()) if len(d) else []:
        for k in (1, 2, 3):
            parts = []
            for line, sign in (("receipts", -1), ("primary", 1)):
                m = out["lines"][line]["chosen"]
                e = d[(d["origin"] == o) & (d["h"] == k) & (d["line"] == line) & (d["method"] == m)]
                if len(e):
                    parts.append(sign * float(e["err"].iloc[0]))
            if len(parts) == 2:
                tot.append({"origin": int(o), "year": int(o + k), "h": k, "err": sum(parts)})
    t = pd.DataFrame(tot)
    if len(t):
        tx = t[~t["year"].isin(COVID)]
        out["total_ex_interest"] = {f"h{k}": {"mae": round(float(tx[tx["h"] == k]["err"].abs().mean()), 3),
                                              "bias": round(float(tx[tx["h"] == k]["err"].mean()), 3),
                                              "n": int((tx["h"] == k).sum())} for k in (1, 2, 3)}
        out["rows"] = [[r["origin"], r["year"], r["h"], round(r["err"], 3)] for r in tot]
    return out


def run_backtests(countries=COUNTRIES) -> dict:
    res = {}
    for cc in countries:
        try:
            res[cc] = backtest(cc)
        except Exception as exc:  # noqa: BLE001
            log.exception("own_annual backtest %s", cc)
            res[cc] = {"error": f"{type(exc).__name__}: {exc}"}
    return res


def load_methods() -> None:
    """METHOD from the stored backtest (rerun ``python -m sovereign_cash.scenarios.channels.own_annual``)."""
    if not BACKTEST_PATH.exists():
        return
    for cc, r in json.loads(BACKTEST_PATH.read_text()).items():
        for line, v in (r.get("lines") or {}).items():
            METHOD[(cc, line)] = v["chosen"]


# ---------------------------------------------------------------- the forecast
def _book_interest(cc: str, book=None) -> pd.Series | None:
    from sovereign_cash.scenarios.channels.sarimax import book_interest

    bi = book_interest(cc)
    return bi if bi is not None else book_interest(cc, book)


def forecast(cc: str, years: list[int], book=None) -> tuple[pd.DataFrame, int, dict]:
    """Annual forecast for ``years`` on the measure (full periods), from the last complete outturn period."""
    load_methods()
    h = history(cc)
    T = int(h.index.max())
    need = list(range(T + 1, max(years) + 1))
    Y, P = levels(cc, list(h.index) + need)
    r = h["receipts"] / Y.reindex(h.index)
    rm = method(cc, "receipts")
    pm = method(cc, "primary")
    Xq = Y if pm == "persist_gdp" else P
    qf = carry(h["primary"] / Xq.reindex(h.index), T, len(need), "persist" if pm == "persist_gdp" else pm)
    out = pd.DataFrame(index=need, dtype=float)
    if cc == "uk" and rm == "fiscal_drag":
        E = earnings_index(list(h.index) + need)
        out["receipts"] = receipts_fiscal_drag(float(h.at[T, "receipts"]), T, Y, E, need)
    else:
        rf = carry(r, T, len(need), rm)
        out["receipts"] = [s * Y[y] for s, y in zip(rf, need)]
    out["primary"] = [s * Xq[y] for s, y in zip(qf, need)]
    bi = _book_interest(cc, book)
    info = {"origin_period": T, "interest_src": "held at the last outturn (no debt book)"}
    if bi is not None:
        bp = to_period(cc, bi, [T] + need)
        if T in bp.index and all(y in bp.index for y in need) and bp[T] > 0:
            out["interest"] = [float(h.at[T, "interest"]) * bp[y] / bp[T] for y in need]
            info["interest_src"] = "last outturn grown with our ISIN debt book's projection"
    if "interest" not in out:
        out["interest"] = float(h.at[T, "interest"])
    out["total"] = out["primary"] + out["interest"] - out["receipts"]
    info["methods"] = {line: method(cc, line) for line in ("receipts", "primary")}
    info["shares_last"] = {"receipts_gdp_pct": round(100 * float(r.iloc[-1]), 3)}
    if cc == "uk" and rm == "fiscal_drag":
        info["fiscal_drag"] = {
            "itnic_share": round(itnic_share(T), 4), "eps_frozen": EPS_FROZEN,
            "eps_indexed": round(EPS_INDEXED, 4), "freeze_last_prior_law": FREEZE_LAST_PRIOR,
            "eps_by_year": {label("uk", y): itnic_eps(y) for y in need},
            "source": DRAG_SOURCE,
        }
        info["shares_last"]["receipts_gdp_pct"] = round(100 * float(out["receipts"].iloc[-1] / Y[out.index[-1]]), 3)
    return out.loc[years], T, info


# ---------------------------------------------------------------- pack entry
def _months(cc: str, years: list[int]) -> list[pd.Timestamp]:
    return [m for y in years for m in period_months(cc, y)]


def build(cc: str, book, onset: pd.Timestamp, published: dict | None = None,
          published_channels: bool = True) -> dict:
    """Pack entry: our baseline, unit-elasticity share channels for GDP and prices, the ISIN book for yields.

    ``published``: the entry the published rules build (OBR reckoner, CBO rules of thumb), whose channels are
    kept as the 'authority' method and whose baseline is shown beside ours for comparison.
    """
    from sovereign_cash.scenarios.channels import annual

    first = (onset.year if onset.month >= 4 else onset.year - 1) if cc == "uk" else \
        (onset.year + 1 if onset.month >= 10 else onset.year) if cc == "us" else onset.year
    years = list(range(first, first + N_YEARS))
    fc, T, info = forecast(cc, years, book)
    lines = [{"id": i, "label": lab, "sign": s} for i, lab, s in LINES.get(cc, DEFAULT_LINES)]
    months = _months(cc, years)
    month_period = [years.index(int(y)) for y in years for _ in range(12)]
    # the part of the first period already out-turned (UK months of the fiscal year in progress)
    stub = None
    base_lines = {ln["id"]: [float(fc.at[y, ln["id"]]) for y in years] for ln in lines}
    if cc == "uk":
        m = uk_monthly()
        done = m.loc[(m.index >= months[0]) & (m.index <= months[11])]
        if len(done):
            # remaining months of the year in progress = forecast year - outturn so far, scaled by the usual share
            # of the year those months carry (three previous fiscal years)
            prev = [y for y in sorted(set(int(i) for i in history(cc).index))][-3:]
            share = {}
            for k in ("receipts", "primary", "interest"):
                tot, part = 0.0, 0.0
                for y in prev:
                    pm = period_months(cc, y)
                    tot += float(m.loc[pm, k].sum())
                    part += float(m.loc[pm[:len(done)], k].sum())
                share[k] = part / tot if tot else len(done) / 12
            for k in ("receipts", "primary", "interest"):
                base_lines[k][0] = float(fc.at[years[0], k]) * (1 - share[k])
            stub = {"months": [done.index.min().strftime("%Y-%m"), done.index.max().strftime("%Y-%m")],
                    "lines": {k: float(done[k].sum()) for k in ("receipts", "primary", "interest")}}
            stub["total"] = stub["lines"]["primary"] + stub["lines"]["interest"] - stub["lines"]["receipts"]
            months = [mm for mm in months if mm > done.index.max()]
            month_period = [years.index(int(y)) for y in years for _ in range(12)][len(done):]
    total = [sum(ln["sign"] * base_lines[ln["id"]][i] for ln in lines) for i in range(len(years))]
    # monthly bases for the exact share channels: each period's line spread evenly over its remaining months
    per_n = [month_period.count(i) for i in range(len(years))]
    tau_m = [max(0, tau(mm, onset) - LAG) for mm in months]
    chans = []
    p_on_gdp = method(cc, "primary") == "persist_gdp"
    drag = cc == "uk" and method(cc, "receipts") == "fiscal_drag"
    if drag:
        # IT+NIC at the frozen-threshold earnings elasticity (NGDP proxies earnings under a GDP shock); other
        # receipts unit-elastic to NGDP. Bases sum to the receipts baseline.
        s = float((info.get("fiscal_drag") or {}).get("itnic_share") or itnic_share(T))
        for part, share, eps, note in (
                ("other", 1.0 - s, 1.0, "non-IT/NIC receipts as a share of nominal GDP"),
                ("itnic", s, EPS_FROZEN,
                 f"income tax + NICs at frozen-threshold elasticity {EPS_FROZEN} to earnings "
                 f"(unit from FY {FREEZE_LAST_PRIOR + 1}; Budget 2025 extension is policy)")):
            chans.append({"id": f"{cc}_ngdp_receipts_{part}_model", "type": "elasticity_monthly",
                          "driver": "ngdp", "line": "receipts", "elasticity": float(eps),
                          "base": [base_lines["receipts"][p] * share / per_n[p] for p in month_period],
                          "tau": tau_m, "method_group": "model",
                          "method": f"{note} (lag {LAG}m); {DRAG_SOURCE}",
                          "badge": "Our model", "source": "sovereign_cash.scenarios.channels.own_annual"})
    line_specs = [] if drag else [("receipts", "ngdp", "nominal GDP")]
    line_specs.append(("primary", "ngdp" if p_on_gdp else "cpi",
                       "nominal GDP" if p_on_gdp else "the price level"))
    for line, driver, label_ in line_specs:
        chans.append({"id": f"{cc}_{driver}_{line}_model", "type": "elasticity_monthly", "driver": driver,
                      "line": line, "elasticity": 1.0,
                      "base": [base_lines[line][p] / per_n[p] for p in month_period],
                      "tau": tau_m, "method_group": "model",
                      "method": f"{line} as a share of {label_} ({method(cc, line)}; lag {LAG}m): the line moves "
                                f"in proportion to the level, as the France/Germany/Italy models",
                      "badge": "Our model", "source": "sovereign_cash.scenarios.channels.own_annual"})
    if cc == "no":
        pet, pet_src = annual._no_petroleum_base(years)
        chans.append(annual._elastic(cc, "brent", "receipts", -1.0, pet,
                                     [[tau(mm, onset) for mm in period_months(cc, y)] for y in years],
                                     badge="Model elasticity",
                                     method="Tab4 bottom-up: oil tax and SDFI x (1 + Brent YoY), unit pass-through",
                                     source=f"sovereign_cash.no.bottom_up_macro; petroleum base {pet_src}"))
        nonpet = [base_lines["receipts"][p] + pet[p] for p in range(len(years))]
        for ch in chans:
            if ch.get("line") == "receipts" and ch.get("driver") == "ngdp":
                ch["base"] = [nonpet[p] / per_n[p] for p in month_period]
                break
    default = {"ngdp": "model", "cpi": "model"}
    if published and published_channels:
        for ch in published.get("channels", []):
            if ch["driver"] in ("ngdp", "cpi"):
                chans.append({**ch, "method_group": "authority"})
            elif ch["driver"] not in ("ngdp", "cpi", "brent"):
                chans.append(ch)
        default.update({k: v for k, v in (published.get("default_methods") or {}).items() if k not in default})
    elif book is not None:
        ch = book.channel(cc, years)
        if ch is not None:
            chans.append(ch)
            chans += book.curve_channels(cc, years)
            default["rates"] = "book"
    gdp_p, _ = levels(cc, years)
    rec_note = (f"receipts with IT+NIC fiscal-drag elasticity {EPS_FROZEN} to earnings under frozen thresholds "
                f"(unit from FY {FREEZE_LAST_PRIOR + 1}; Budget 2025 extension in the policy line)"
                if drag else f"receipts as a share of nominal GDP ({info['methods']['receipts']})")
    entry = {
        "cc": cc, "name": NAMES[cc], "ccy": CCY[cc], "unit": f"{CCY[cc]} bn",
        "measure": MEASURE[cc],
        "period_kind": {"uk": "fiscal year (April-March)", "us": "fiscal year (October-September)"}.get(
            cc, "calendar year"),
        "periods": [label(cc, y) for y in years], "origin": ts(onset - pd.offsets.MonthBegin(1)), "onset": ts(onset),
        "months": [mm.strftime("%Y-%m") for mm in months], "month_period": month_period,
        "lines": lines,
        "baseline": {"total": total, "lines": base_lines, "model_total": total,
                     "source": (f"our annual component model: {rec_note} and spending as a "
                                f"share of the price level ({info['methods']['primary']}), interest: "
                                f"{info['interest_src']}; macro: {macro_cy(cc)[2]}")},
        "channels": chans, "default_methods": default,
        "modes": {"ngdp": ["level", "growth"], "cpi": ["level", "growth"], "brent": ["level"]},
        "model": {"family": "annual component model", "origin_period": label(cc, T), **info},
        "gdp": [float(gdp_p[y]) for y in years],
        "own_stub": stub,
    }
    if published:
        entry["baseline"]["authority"] = {"label": published["baseline"]["source"],
                                          "years": dict(zip(published["periods"], published["baseline"]["total"]))}
        for k in ("feedback",):
            if k in published:
                entry[k] = published[k]
    if "feedback" not in entry and book is not None and (rr := book.feedback_rate(cc)) is not None and np.isfinite(rr):
        entry["feedback"] = {"rate_pct": [rr] * len(years), "source": "debt snapshot marginal new-issue rate"}
    return entry


if __name__ == "__main__":
    res = run_backtests()
    BACKTEST_PATH.write_text(json.dumps(res, indent=1))
    for cc, r in res.items():
        if "error" in r:
            print(cc, r["error"])
            continue
        ls = r["lines"]
        bits = []
        for k, v in ls.items():
            sc = v["scores"]
            bit = (f"{k} persist {sc['persist']['mean']:.1f} ({sc['persist']['mean_pct_gdp']:.2f}% GDP)"
                   f" revert {sc['revert']['mean']:.1f} ({sc['revert']['mean_pct_gdp']:.2f}%)")
            if "fiscal_drag" in sc:
                bit += (f" fiscal_drag {sc['fiscal_drag']['mean']:.1f} "
                        f"({sc['fiscal_drag']['mean_pct_gdp']:.2f}%)")
            bit += f" -> {v['chosen']}"
            bits.append(bit)
        print(f"{cc}: origins {r['n_origins']} from {r['first_origin']}; " + "; ".join(bits)
              + "; total ex int MAE h1/h2/h3 " + "/".join(f"{r['total_ex_interest'][f'h{k}']['mae']:.1f}" for k in (1, 2, 3)))
