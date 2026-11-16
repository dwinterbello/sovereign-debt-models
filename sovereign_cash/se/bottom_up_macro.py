"""Sweden v2 bottom-up NBR: Budgetprop / KI / SCB categories × macro drivers.

Identity (NBR sign, + = borrowing, bn SEK):

    nbr = −tax_rev + primary_spend + net_lending + interest
    budget_balance = −nbr
    cash_requirement = nbr

``tax_rev`` is SCB CG budget revenue; ``primary_spend = SCB exp − interest −
net_lending`` so the four lines reconstitute ``exp − rev = −budget_balance``.
Before the SCB sample (2014) fall back to CH1089 component seasonals (v1 params).

Month shape = PRE-locked seasonal per category (not a headline seasonal). Annual
levels = lag-1 macro growth (NGDP/wages/CPI) and Budgetprop / KI when the
vintage is known at the origin. Interest tried as debt stock × effective rate;
that path loses category and anchored headline PRE to a PRE-selected interest
seasonal, so the alternate column keeps the seasonal (documented reject).

RG annual / rest-of-year pin is applied **after** the category sum (same helper
as production). Ship only if RG-anchored PRE mean MAE beats production by ≥5%
and is worse at ≤1 scored horizon.

v1 ``bottom_up.py`` (CH1089 component seasonals) stays in place with SHIP=False.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash import contract
from sovereign_cash.se import build as B
from sovereign_cash.se import bottom_up as V1
from sovereign_cash.se import model as M

OUT_SCORE = contract.DERIVED / "se_bottomup_macro_score.json"
AUTH = Path(__file__).resolve().parents[1] / "data" / "se" / "authority.json"
SCB_CACHE = Path(__file__).resolve().parents[1] / "data" / "se" / "scb_budget_monthly.parquet"
SCB_URL = ("https://api.scb.se/OV0104/v1/doris/en/ssd/START/NR/NR0111/NR0111A/"
           "BudgetStatForvM")

EDGE = 0.05
SCORE_H = M.SCORE_H
ORIGIN_START = pd.Timestamp("2010-12-01")
ANCHOR_ORIGIN_MONTHS = (6, 7, 8, 9)
SKIP_YEARS = {2020, 2021}
MIN_FIT = 5

# Category → default macro driver key (annual % from yoy_components.macro / wages).
DRIVERS = {
    "tax_rev": ("ngdp", "wages"),
    "primary_spend": ("cpi",),
    "net_lending": None,  # discrete; Budgetprop / KI annual or seasonal
    "interest": ("rates",),  # stock × effective rate tried; seasonal kept (see rejects)
}

# PRE-locked month-shape params (selected on category PRE MAE over SCORE_H; not OOS).
# Small grid: smly + seas k∈{3,5} × a∈{0,0.5} × {mean,median}.
CAT_SEAS: dict[str, tuple[int, float, str]] = {
    "tax_rev": (1, 0.0, "mean"),           # smly
    "primary_spend": (3, 0.5, "mean"),     # seas_k3_a0.5_mean
    "net_lending": (5, 0.0, "median"),     # seas_k5_a0_median
    "interest": (3, 0.5, "mean"),          # seas_k3_a0.5_mean
}
# Pre-SCB / thin history: frozen v1 CH1089 winners (same PRE gate, no re-search).
V1_FALLBACK: dict[str, tuple[int, float, str]] = {
    "nbr_primary": (1, 0.0, "mean"),
    "nbr_net_lending": (5, 0.0, "median"),
    "nbr_interest": (3, 0.5, "mean"),
}


def load_authority() -> dict:
    return json.loads(AUTH.read_text()) if AUTH.exists() else {}


def fetch_scb_budget(refresh: bool = False) -> pd.DataFrame:
    """Monthly CG budget revenue / expenditure (bn SEK). Cached under data/se/."""
    if SCB_CACHE.exists() and not refresh:
        df = pd.read_parquet(SCB_CACHE)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    query = {
        "query": [
            {"code": "StatligForvaltning",
             "selection": {"filter": "item", "values": ["100"]}},
            {"code": "ContentsCode",
             "selection": {"filter": "item", "values": ["000002Q2", "000002Q4", "000002Q3"]}},
        ],
        "response": {"format": "json"},
    }
    req = urllib.request.Request(
        SCB_URL, data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.loads(r.read())
    rows = []
    for d in raw["data"]:
        _sector, month = d["key"]
        rev, exp, bal = (float(x) for x in d["values"])
        rows.append({
            "month": pd.Timestamp(month.replace("M", "-") + "-01"),
            "tax_rev": rev / 1000.0,
            "exp": exp / 1000.0,
            "bal": bal / 1000.0,
        })
    df = pd.DataFrame(rows).set_index("month").sort_index()
    SCB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCB_CACHE)
    return df


def _wages_yoy() -> pd.Series:
    """Annual compensation-of-employees YoY (Eurostat namq D1), calendar-year mean."""
    from sovereign_cash.yoy_components import _eurostat_json
    cache = contract.DERIVED.parent / "data" / "debt" / "macro_wages_se.parquet"
    # store next to other macro caches
    cache = Path(__file__).resolve().parents[1] / "data" / "debt" / "macro_wages_se.parquet"
    if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).days < 30:
        s = pd.read_parquet(cache)["wages"]
        return s.dropna()
    try:
        w = _eurostat_json("namq_10_gdp", freq="Q", unit="CP_MEUR", s_adj="NSA",
                           na_item="D1", geo="SE")
        w = w[w["time"].str.contains(r"Q[1-4]$", regex=True)].copy()
        w["q"] = pd.PeriodIndex(w["time"], freq="Q").to_timestamp(how="start")
        g = w.set_index("q")["value"].sort_index().pct_change(4) * 100.0
        ann = g.groupby(g.index.year).mean()
        ann = ann[g.groupby(g.index.year).count() >= 4]
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"wages": ann}).to_parquet(cache)
        return ann.dropna()
    except Exception:
        return pd.Series(dtype=float)


def macro_drivers() -> dict[str, pd.Series]:
    from sovereign_cash.yoy_components import macro
    d = dict(macro("se"))
    w = _wages_yoy()
    if len(w):
        d["wages"] = w
    return d


def debt_stock_ye(d: dict) -> pd.Series:
    debt = d["debt"].set_index("month")["debt"] / 1e9
    return debt.groupby(debt.index.year).last()


def categories(d: dict, scb: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """History category frame in bn SEK (NBR-sign for lending/interest; tax/spend positive levels)."""
    comps = V1.components_bn(d["nbr"])
    scb = fetch_scb_budget() if scb is None else scb
    inte = comps["nbr_interest"]
    lend = comps["nbr_net_lending"]
    # align SCB to CH1089 months; strip interest and lending out of SCB exp
    tax = scb["tax_rev"].reindex(inte.index)
    spend = (scb["exp"] - inte - lend).reindex(inte.index)
    # pre-SCB: leave tax/spend NaN; callers fall back to CH1089 primary
    return {
        "tax_rev": tax,
        "primary_spend": spend,
        "net_lending": lend,
        "interest": inte,
        "nbr_primary": comps["nbr_primary"],
        "nbr": comps["nbr"],
        "budget_balance": comps["budget_balance"],
    }


def identity_resid_bn(cats: dict[str, pd.Series]) -> float:
    """Max |nbr − (−tax + spend + lend + interest)| on months where SCB exists."""
    mask = cats["tax_rev"].notna() & cats["primary_spend"].notna()
    if not mask.any():
        return float("nan")
    nbr_hat = (-cats["tax_rev"] + cats["primary_spend"]
               + cats["net_lending"] + cats["interest"])
    return float((cats["nbr"].loc[mask] - nbr_hat.loc[mask]).abs().max())


def _annual(series: pd.Series) -> pd.Series:
    g = series.groupby(series.index.year)
    tot = g.sum()
    return tot[g.count() == 12]


def _fit_yoy(levels: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
             keys: tuple[str, ...], floor: bool = True) -> tuple[float, np.ndarray] | None:
    last = origin.year - 1
    ys, xs = [], []
    for yr in levels.index:
        if yr > last or yr in SKIP_YEARS or (yr - 1) not in levels.index:
            continue
        row = []
        ok = True
        for k in keys:
            if k not in drivers or (yr - 1) not in drivers[k].index:
                ok = False
                break
            row.append(float(drivers[k].loc[yr - 1]) / 100.0)
        if not ok:
            continue
        base = float(levels.loc[yr - 1])
        if abs(base) < 1.0:
            continue
        ys.append(float(levels.loc[yr]) / base - 1.0)
        xs.append(row)
    if len(ys) < MIN_FIT:
        return None
    X = np.column_stack([np.ones(len(xs)), np.asarray(xs)])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    slopes = coef[1:].astype(float)
    if floor:
        slopes = np.maximum(slopes, 0.0)
        intercept = float(np.mean(ys) - slopes @ np.mean(xs, axis=0))
    else:
        intercept = float(coef[0])
    return intercept, slopes


def _growth(levels: pd.Series, drivers: dict[str, pd.Series], origin: pd.Timestamp,
            keys: tuple[str, ...]) -> float:
    fit = _fit_yoy(levels, drivers, origin, keys)
    yr = origin.year - 1
    if fit is None or any(k not in drivers or yr not in drivers[k].index for k in keys):
        return 0.0
    a, b = fit
    x = np.array([float(drivers[k].loc[yr]) / 100.0 for k in keys])
    return float(np.clip(a + b @ x, -0.30, 0.40))


def _smly_grown(series: pd.Series, origin: pd.Timestamp, hmax: int, g: float) -> pd.Series:
    out = {}
    for h in range(1, hmax + 1):
        t = origin + pd.DateOffset(months=h)
        prev = t - pd.DateOffset(years=1)
        if prev not in series.index or pd.isna(series[prev]):
            continue
        out[t] = float(series[prev]) * (1.0 + g)
    return pd.Series(out, dtype=float)


def _eff_rate(interest_ann: pd.Series, debt_ye: pd.Series) -> pd.Series:
    out = {}
    for yr in interest_ann.index:
        if yr not in debt_ye.index or (yr - 1) not in debt_ye.index:
            continue
        stock = 0.5 * (float(debt_ye[yr]) + float(debt_ye[yr - 1]))
        if stock <= 0:
            continue
        out[yr] = 100.0 * float(interest_ann[yr]) / stock
    return pd.Series(out, dtype=float)


def _interest_annual(origin: pd.Timestamp, year: int, eff: pd.Series,
                     debt_ye: pd.Series, drivers: dict[str, pd.Series],
                     auth: dict) -> float:
    """Stock_{y−1} × forecast effective rate / 100. Authority UO26 overlay when known."""
    bp = auth.get("budgetprop", {}).get("years", {})
    row = bp.get(str(year))
    if row and _auth_known(auth, origin, year) and "interest_bn" in row:
        return float(row["interest_bn"])
    last = origin.year - 1
    ys, xs = [], []
    for yr in eff.index:
        if yr > last or yr in SKIP_YEARS or (yr - 1) not in eff.index:
            continue
        ys.append(float(eff[yr]))
        # prefer debt-book / macro rates lag; else AR(1) on own eff
        if "rates" in drivers and (yr - 1) in drivers["rates"].index:
            xs.append(float(drivers["rates"].loc[yr - 1]))
        else:
            xs.append(float(eff[yr - 1]))
    if len(ys) >= MIN_FIT:
        X = np.column_stack([np.ones(len(xs)), xs])
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        x = float(drivers["rates"].loc[year - 1]) if "rates" in drivers and (year - 1) in drivers["rates"].index \
            else float(eff.get(min(year - 1, last), eff.iloc[-1] if len(eff) else 2.0))
        r = float(coef[0] + coef[1] * x)
    else:
        r = float(eff.get(last, 2.0))
    stock = float(debt_ye.get(year - 1, debt_ye.iloc[-1] if len(debt_ye) else 1000.0))
    return stock * max(r, 0.0) / 100.0


def _interest_monthly(origin: pd.Timestamp, hmax: int, interest: pd.Series,
                      eff: pd.Series, debt_ye: pd.Series,
                      drivers: dict[str, pd.Series], auth: dict) -> pd.Series:
    years = {(origin + pd.DateOffset(months=h)).year for h in range(1, hmax + 1)}
    ann = {yr: _interest_annual(origin, yr, eff, debt_ye, drivers, auth) for yr in years}
    out = {}
    for h in range(1, hmax + 1):
        t = origin + pd.DateOffset(months=h)
        shares = []
        for j in range(1, 6):
            p = pd.Timestamp(t.year - j, t.month, 1)
            if p not in interest.index:
                continue
            mons = [pd.Timestamp(t.year - j, mo, 1) for mo in range(1, 13)]
            if not all(m in interest.index for m in mons):
                continue
            tot = float(interest.reindex(mons).sum())
            if abs(tot) < 1e-6:
                continue
            shares.append(float(interest[p]) / tot)
        sh = float(np.mean(shares)) if shares else 1.0 / 12.0
        out[t] = ann[t.year] * sh
    return pd.Series(out, dtype=float)


def _auth_known(auth: dict, origin: pd.Timestamp, year: int) -> bool:
    """Authority vintage date must be ≤ origin (no lookahead)."""
    bp = auth.get("budgetprop", {})
    vint = bp.get("vintage_date")
    years = bp.get("years", {})
    if str(year) not in years or not vint:
        return False
    return pd.Timestamp(vint) <= origin


def _lending_fc(series: pd.Series, origin: pd.Timestamp, hmax: int, auth: dict) -> pd.Series:
    years = {(origin + pd.DateOffset(months=h)).year for h in range(1, hmax + 1)}
    base = M.forecast(series, origin, *CAT_SEAS["net_lending"], hmax=hmax)
    bp = auth.get("budgetprop", {}).get("years", {})
    for yr in years:
        row = bp.get(str(yr))
        if not row or not _auth_known(auth, origin, yr) or "net_lending_bn" not in row:
            continue
        target = float(row["net_lending_bn"])
        ms = [origin + pd.DateOffset(months=h) for h in range(1, hmax + 1)
              if (origin + pd.DateOffset(months=h)).year == yr]
        if not ms:
            continue
        # open year: remaining = annual − known
        known = series[(series.index.year == yr) & (series.index <= origin)]
        rem = target - float(known.sum()) if len(known) else target
        shares = []
        idx = pd.DatetimeIndex(ms)
        for j in range(1, 6):
            past = [pd.Timestamp(t.year - j, t.month, 1) for t in idx]
            if not all(p in series.index for p in past):
                continue
            v = series.reindex(past).to_numpy()
            tot = v.sum()
            if abs(tot) < 1e-9:
                continue
            shares.append(v / tot)
        if shares:
            w = np.mean(shares, axis=0)
            w = w / w.sum() if abs(w.sum()) > 1e-12 else np.ones(len(idx)) / len(idx)
        else:
            w = np.ones(len(idx)) / len(idx)
        for t, wi in zip(idx, w):
            base.loc[t] = rem * float(wi)
    return base


def _authority_scale(series: pd.Series, origin: pd.Timestamp, hmax: int,
                     annual_key: str, auth: dict, sign: float = 1.0) -> pd.Series | None:
    """If Budgetprop annual exists for a forecast year, rescale smly path to that annual."""
    if origin not in series.index:
        return None
    bp = auth.get("budgetprop", {}).get("years", {})
    applicable = False
    for h in range(1, hmax + 1):
        t = origin + pd.DateOffset(months=h)
        row = bp.get(str(t.year))
        if row and annual_key in row and _auth_known(auth, origin, t.year):
            applicable = True
            break
    if not applicable:
        return None
    out = M.forecast(series, origin, 1, 0.0, "mean", hmax=hmax)
    for yr in out.index.year.unique():
        row = bp.get(str(int(yr)))
        if not row or annual_key not in row or not _auth_known(auth, origin, int(yr)):
            continue
        target = sign * float(row[annual_key])
        known = series[(series.index.year == yr) & (series.index <= origin)]
        rem = target - float(known.sum()) if len(known) else target
        chunk = out.loc[out.index.year == yr]
        s = float(chunk.sum())
        if abs(s) < 1e-9:
            out.loc[chunk.index] = rem / max(len(chunk), 1)
        else:
            out.loc[chunk.index] = chunk * (rem / s)
    return out


def _rescale_year_chunks(fc: pd.Series, hist: pd.Series, origin: pd.Timestamp,
                         annual_targets: dict[int, float]) -> pd.Series:
    """Rescale each forecast-year chunk so known YTD + forecast = annual target."""
    out = fc.copy()
    for yr, target in annual_targets.items():
        chunk = out.loc[out.index.year == yr]
        if chunk.empty:
            continue
        known = hist[(hist.index.year == yr) & (hist.index <= origin)]
        rem = float(target) - float(known.sum()) if len(known) else float(target)
        s = float(chunk.sum())
        if abs(s) < 1e-9:
            out.loc[chunk.index] = rem / max(len(chunk), 1)
        else:
            out.loc[chunk.index] = chunk * (rem / s)
    return out


def _macro_annual_targets(hist: pd.Series, drivers: dict[str, pd.Series],
                          origin: pd.Timestamp, keys: tuple[str, ...],
                          years: set[int]) -> dict[int, float]:
    g = _growth(_annual(hist), drivers, origin, keys)
    ann = _annual(hist)
    out = {}
    for yr in years:
        base = float(ann.get(yr - 1, np.nan))
        if not np.isfinite(base):
            continue
        out[yr] = base * (1.0 + g)
    return out


def _auth_tax_targets(auth: dict, origin: pd.Timestamp, years: set[int]) -> dict[int, float]:
    """Budgetprop tax_rev_bn when vintage ≤ origin; else KI F3304 taxes (NR) as soft fill."""
    out: dict[int, float] = {}
    bp = auth.get("budgetprop", {}).get("years", {})
    ki = auth.get("ki", {}).get("years", {})
    ki_vint = auth.get("ki", {}).get("vintage_date")
    for yr in years:
        row = bp.get(str(yr))
        if row and "tax_rev_bn" in row and _auth_known(auth, origin, yr):
            out[yr] = float(row["tax_rev_bn"])
            continue
        krow = ki.get(str(yr))
        if krow and "tax_rev_bn" in krow and ki_vint and pd.Timestamp(ki_vint) <= origin:
            out[yr] = float(krow["tax_rev_bn"])
    return out


def forecast_nbr(cats: dict[str, pd.Series], drivers: dict[str, pd.Series],
                 debt_ye: pd.Series, origin: pd.Timestamp, hmax: int = 12,
                 auth: dict | None = None) -> pd.Series:
    """Category×macro NBR path through ``origin`` only."""
    auth = auth if auth is not None else load_authority()
    tax = cats["tax_rev"].dropna()
    spend = cats["primary_spend"].dropna()
    lend = cats["net_lending"]
    inte = cats["interest"]

    tax_h = tax.loc[:origin].dropna()
    spend_h = spend.loc[:origin].dropna()
    use_scb = (origin in tax_h.index and origin in spend_h.index
               and len(tax_h) >= 24 and len(spend_h) >= 24)

    if use_scb:
        years = {(origin + pd.DateOffset(months=h)).year for h in range(1, hmax + 1)}
        tax_fc = M.forecast(tax_h, origin, *CAT_SEAS["tax_rev"], hmax=hmax)
        sp_fc = M.forecast(spend_h, origin, *CAT_SEAS["primary_spend"], hmax=hmax)
        lend_fc = M.forecast(lend.loc[:origin], origin, *CAT_SEAS["net_lending"], hmax=hmax)
        inte_fc = M.forecast(inte.loc[:origin], origin, *CAT_SEAS["interest"], hmax=hmax)

        # Annual levels: Budgetprop/KI tax when known; else lag-1 NGDP (and wages blend).
        auth_tax = _auth_tax_targets(auth, origin, years)
        if auth_tax:
            tax_fc = _rescale_year_chunks(tax_fc, tax_h, origin, auth_tax)
        else:
            g_tax = _growth(_annual(tax_h), drivers, origin, ("ngdp",))
            if "wages" in drivers:
                g_w = _growth(_annual(tax_h), drivers, origin, ("wages",))
                g_tax = 0.5 * g_tax + 0.5 * g_w
            tax_fc = _rescale_year_chunks(
                tax_fc, tax_h, origin,
                {yr: float(_annual(tax_h).get(yr - 1, np.nan)) * (1.0 + g_tax)
                 for yr in years if (yr - 1) in _annual(tax_h).index},
            )
        sp_tgt = _macro_annual_targets(spend_h, drivers, origin, ("cpi",), years)
        if sp_tgt:
            sp_fc = _rescale_year_chunks(sp_fc, spend_h, origin, sp_tgt)
        # Net lending: Budgetprop annual overwrites seasonal when vintage known.
        lend_fc = _lending_fc(lend.loc[:origin], origin, hmax, auth)
        return ((-tax_fc).add(sp_fc, fill_value=0.0)
                .add(lend_fc, fill_value=0.0).add(inte_fc, fill_value=0.0))

    # Pre-SCB: frozen CH1089 component seasonals (v1 winners).
    pri = cats["nbr_primary"].loc[:origin]
    pri_fc = M.forecast(pri, origin, *V1_FALLBACK["nbr_primary"], hmax=hmax)
    lend_fc = M.forecast(lend.loc[:origin], origin, *V1_FALLBACK["nbr_net_lending"], hmax=hmax)
    inte_fc = M.forecast(inte.loc[:origin], origin, *V1_FALLBACK["nbr_interest"], hmax=hmax)
    return pri_fc.add(lend_fc, fill_value=0.0).add(inte_fc, fill_value=0.0)


def forecast_balance(cats: dict[str, pd.Series], drivers: dict[str, pd.Series],
                     debt_ye: pd.Series, origin: pd.Timestamp, hmax: int = 12,
                     auth: dict | None = None) -> pd.Series:
    return -forecast_nbr(cats, drivers, debt_ye, origin, hmax=hmax, auth=auth)


def _headline_bt(y: pd.Series, fc_fn) -> pd.DataFrame:
    rows = []
    for o in [o for o in y.index if o >= ORIGIN_START and o < y.index[-1]]:
        fc = fc_fn(o, 12)
        for t, v in fc.items():
            if t in y.index and np.isfinite(v):
                h = (t.year - o.year) * 12 + t.month - o.month
                rows.append((o, t, h, float(v), float(y[t])))
    bt = pd.DataFrame(rows, columns=["origin", "target", "h", "fc", "actual"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    bt["win"] = M.window(bt["target"])
    return bt


def _rg_anchor_bt(y: pd.Series, rgm: pd.Series, raw_fn, method: str) -> pd.DataFrame:
    rows = []
    for o in y.index:
        if o.month not in ANCHOR_ORIGIN_MONTHS:
            continue
        ms = pd.date_range(o + pd.DateOffset(months=1), pd.Timestamp(o.year, 12, 1), freq="MS")
        if not all(m in rgm.index and m in y.index for m in ms):
            continue
        raw = raw_fn(o, len(ms)).reindex(ms)
        if raw.isna().any():
            continue
        tot = float(rgm.reindex(ms).sum())
        al = M.allocate(raw, tot, method, y, o)
        for t in ms:
            h = (t.year - o.year) * 12 + t.month - o.month
            rows.append((o, t, h, float(al[t]), float(y[t])))
    bt = pd.DataFrame(rows, columns=["origin", "target", "h", "fc", "actual"])
    bt["ae"] = (bt["fc"] - bt["actual"]).abs()
    bt["win"] = M.window(bt["target"])
    return bt


def score_pair(y: pd.Series, bu_fn, prod_name: str, method: str, rgm: pd.Series) -> dict:
    k, a, f = M.choose_params(prod_name)
    bt_bu_u = _headline_bt(y, bu_fn)
    bt_pr_u = _headline_bt(y, lambda o, h: M.forecast(y, o, k, a, f, hmax=h))
    bt_bu_a = _rg_anchor_bt(y, rgm, bu_fn, method)
    bt_pr_a = _rg_anchor_bt(y, rgm, lambda o, h: M.forecast(y, o, k, a, f, hmax=h), method)

    def pack(bt_bu, bt_pr):
        out = {}
        for win in ("PRE", "OOS"):
            rows = {}
            for h in SCORE_H:
                bu = bt_bu[(bt_bu["win"] == win) & (bt_bu["h"] == h)]["ae"]
                pr = bt_pr[(bt_pr["win"] == win) & (bt_pr["h"] == h)]["ae"]
                if bu.empty and pr.empty:
                    continue
                rows[h] = {
                    "bottom_up": round(float(bu.mean()), 3) if len(bu) else None,
                    "production": round(float(pr.mean()), 3) if len(pr) else None,
                    "n": int(max(len(bu), len(pr))),
                }
            out[win] = rows
        return out

    return {"unanchored": pack(bt_bu_u, bt_pr_u), "anchored": pack(bt_bu_a, bt_pr_a)}


def mean_mae(score: dict[int, dict], key: str) -> float:
    vals = [score[h][key] for h in SCORE_H if h in score and score[h].get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def ship_decision(pre_anchored: dict[int, dict]) -> tuple[bool, str]:
    if not pre_anchored or len(pre_anchored) < 2:
        return False, "incomplete PRE anchored score table"
    bu = mean_mae(pre_anchored, "bottom_up")
    prod = mean_mae(pre_anchored, "production")
    if not np.isfinite(bu) or not np.isfinite(prod):
        return False, "non-finite PRE anchored MAE"
    if not (bu < (1.0 - EDGE) * prod):
        return False, (f"PRE anchored mean MAE macro {bu:.3f} does not beat production "
                       f"{prod:.3f} by ≥{EDGE:.0%}")
    worse = sum(1 for h in SCORE_H if h in pre_anchored
                and pre_anchored[h].get("bottom_up") is not None
                and pre_anchored[h].get("production") is not None
                and pre_anchored[h]["bottom_up"] > pre_anchored[h]["production"])
    if worse > 1:
        return False, f"worse than production at {worse} anchored horizons (max 1 allowed)"
    return True, "PRE anchored gate passed"


def category_mae_pre(cats: dict[str, pd.Series], drivers: dict[str, pd.Series],
                     debt_ye: pd.Series, auth: dict) -> dict:
    """Per-category PRE MAE (bn) of the macro rule vs smly on the category itself."""
    tab = {}
    for name, keys in DRIVERS.items():
        if name == "interest":
            # score annual interest vs smly annual on PRE years
            inte = cats["interest"]
            eff = _eff_rate(_annual(inte), debt_ye)
            rows_m, rows_s = [], []
            for o in [o for o in inte.index if o >= ORIGIN_START and o.month == 12
                      and o.year + 1 <= inte.index.max().year]:
                yr = o.year + 1
                if M.window(pd.Series([pd.Timestamp(yr, 6, 1)])).iloc[0] != "PRE":
                    continue
                fc = _interest_annual(o, yr, eff, debt_ye, drivers, auth)
                act = float(_annual(inte).get(yr, np.nan))
                if not np.isfinite(act):
                    continue
                sm = float(_annual(inte).get(yr - 1, np.nan))
                rows_m.append(abs(fc - act))
                rows_s.append(abs(sm - act) if np.isfinite(sm) else np.nan)
            seas_mae = V1._comp_mae(inte, *CAT_SEAS["interest"]) if len(inte) else np.inf
            tab[name] = {
                "driver": "debt_stock × effective_rate (tried); shipped = PRE seas_k3_a0.5_mean",
                "rule": "CAT_SEAS interest; stock×rate annual → shares rejected on PRE",
                "pre_mae_macro": round(float(np.nanmean(rows_m)), 3) if rows_m else None,
                "pre_mae_smly": round(float(np.nanmean(rows_s)), 3) if rows_s else None,
                "pre_mae_seas": round(seas_mae, 3) if np.isfinite(seas_mae) else None,
                "series": "CH1089 Räntor på statsskulden",
            }
            continue
        if keys is None:
            s = cats[name].dropna()
            mae_smly = V1._comp_mae(s, 1, 0.0, "mean") if len(s) else np.inf
            mae_seas = V1._comp_mae(s, *CAT_SEAS[name]) if len(s) else np.inf
            tab[name] = {
                "driver": "None (Budgetprop / KI annual when vintage ≤ origin; else CAT_SEAS)",
                "rule": "seas_k5_a0_median / authority annual × month shares",
                "pre_mae_macro": round(mae_seas, 3) if np.isfinite(mae_seas) else None,
                "pre_mae_smly": round(mae_smly, 3) if np.isfinite(mae_smly) else None,
                "series": "CH1089 Nettoutlåning / Budgetprop nettoutlåning",
                "justification": "agency deposits/loans are discrete one-offs; no continuous determinant",
            }
            continue
        s = cats[name].dropna()
        if len(s) < 36:
            tab[name] = {
                "driver": "+".join(keys),
                "rule": "CAT_SEAS month shape × lag-1 macro annual; Budgetprop/KI tax when known",
                "pre_mae_macro": None,
                "pre_mae_smly": None,
                "series": "SCB BudgetStatForvM" if name in ("tax_rev", "primary_spend") else name,
                "note": "series too short for category MAE",
            }
            continue
        rows_m, rows_s = [], []
        kk, aa, ff = CAT_SEAS[name]
        for o in [o for o in s.index if o >= max(ORIGIN_START, s.index.min() + pd.DateOffset(years=3))
                  and o < s.index[-1]]:
            g = _growth(_annual(s.loc[:o]), drivers, o, keys)
            fc = M.forecast(s.loc[:o], o, kk, aa, ff, hmax=12)
            years = {t.year for t in fc.index}
            tgt = {yr: float(_annual(s.loc[:o]).get(yr - 1, np.nan)) * (1.0 + g)
                   for yr in years if (yr - 1) in _annual(s.loc[:o]).index}
            if tgt:
                fc = _rescale_year_chunks(fc, s.loc[:o], o, tgt)
            sm = M.forecast(s.loc[:o], o, 1, 0.0, "mean", hmax=12)
            for t, v in fc.items():
                if t not in s.index:
                    continue
                if M.window(pd.Series([t])).iloc[0] != "PRE":
                    continue
                rows_m.append(abs(v - float(s[t])))
                if t in sm.index:
                    rows_s.append(abs(float(sm[t]) - float(s[t])))
        tab[name] = {
            "driver": "+".join(keys),
            "rule": (f"CAT_SEAS{CAT_SEAS[name]} month shape × lag-1 macro annual; "
                     "Budgetprop/KI tax rescale when vintage ≤ origin"),
            "pre_mae_macro": round(float(np.mean(rows_m)), 3) if rows_m else None,
            "pre_mae_smly": round(float(np.mean(rows_s)), 3) if rows_s else None,
            "series": ("SCB BudgetStatForvM total revenue" if name == "tax_rev"
                       else "SCB exp − CH1089 interest − net lending"),
        }
    return tab


def build_report(d: dict, y: pd.Series, chosen: str, method: str,
                 write: bool = True) -> dict:
    auth = load_authority()
    scb = fetch_scb_budget()
    cats = categories(d, scb)
    drivers = macro_drivers()
    debt_ye = debt_stock_ye(d)
    rgm = d["rg"].set_index("month")["rg_forecast_bn"].dropna()

    def bu_fn(o, h):
        return forecast_balance(cats, drivers, debt_ye, o, hmax=h, auth=auth)

    scores = score_pair(y, bu_fn, chosen, method, rgm)
    un, an = scores["unanchored"], scores["anchored"]
    ship, reason = ship_decision(an.get("PRE", {}))
    cat_tab = category_mae_pre(cats, drivers, debt_ye, auth)

    report = {
        "ship": ship,
        "ship_reason": reason,
        "version": "v2_macro",
        "edge": EDGE,
        "horizons": list(SCORE_H),
        "production_model": chosen,
        "allocation_method": method,
        "identity_resid_bn": round(identity_resid_bn(cats), 6),
        "category_drivers": cat_tab,
        "pre_unanchored": {str(h): un["PRE"][h] for h in sorted(un.get("PRE", {}))},
        "oos_unanchored": {str(h): un["OOS"][h] for h in sorted(un.get("OOS", {}))},
        "pre_anchored": {str(h): an["PRE"][h] for h in sorted(an.get("PRE", {}))},
        "oos_anchored": {str(h): an["OOS"][h] for h in sorted(an.get("OOS", {}))},
        "pre_mean_mae_unanchored": {
            "bottom_up": round(mean_mae(un.get("PRE", {}), "bottom_up"), 3),
            "production": round(mean_mae(un.get("PRE", {}), "production"), 3),
        },
        "oos_mean_mae_unanchored": {
            "bottom_up": round(mean_mae(un.get("OOS", {}), "bottom_up"), 3),
            "production": round(mean_mae(un.get("OOS", {}), "production"), 3),
        },
        "pre_mean_mae_anchored": {
            "bottom_up": round(mean_mae(an.get("PRE", {}), "bottom_up"), 3),
            "production": round(mean_mae(an.get("PRE", {}), "production"), 3),
        },
        "oos_mean_mae_anchored": {
            "bottom_up": round(mean_mae(an.get("OOS", {}), "bottom_up"), 3),
            "production": round(mean_mae(an.get("OOS", {}), "production"), 3),
        },
        "rejects": [
            "v1 CH1089 component seasonals under RG pin: anchored PRE 10.567 vs production 10.045",
            "SCB smly×NGDP/CPI (prior v2): anchored PRE ~16.8; lag-1 NGDP/tax YoY corr ~0.18",
            "interest stock×rate annual (shares or seasonal renormalised): anchored PRE 11.1–13.2",
            "category PRE seasonals + macro annual: anchored PRE ~10.35 (best honest; still <5%)",
            "prod↔v1 shape blend α=0.20: anchored PRE 9.882 — beats prod by ~1.6% only; forbidden as seasonal rebrand",
            "tax/spend calendar shares × last annual: anchored PRE 10.076",
            "Statskontoret / KI annuals change open-year levels only; RG-anchored gate is shape-only",
        ],
        "cat_seas": {k: {"k": v[0], "a": v[1], "f": v[2]} for k, v in CAT_SEAS.items()},
        "authority_tables": [
            {"name": "Budgetpropositionen / Ekonomiska vårpropositionen — inkomster, utgifter, UO26, nettoutlåning",
             "vintage": auth.get("budgetprop", {}).get("vintage"),
             "path": "data/se/authority.json#budgetprop"},
            {"name": "Konjunkturinstitutet F3328 budgetsaldo / F3304 skatter & ränteutgifter",
             "vintage": auth.get("ki", {}).get("vintage"),
             "path": "data/se/authority.json#ki"},
            {"name": "Statskontoret prognos statens budget (juni 2026)",
             "vintage": auth.get("statskontoret", {}).get("vintage"),
             "path": "data/se/authority.json#statskontoret"},
            {"name": "SCB BudgetStatForvM central government budget monthly revenue/expenditure",
             "path": str(SCB_CACHE.relative_to(Path(__file__).resolve().parents[1]))
             if SCB_CACHE.exists() else "data/se/scb_budget_monthly.parquet"},
            {"name": "Riksgälden CH1089 Primärt saldo / Nettoutlåning / Räntor",
             "path": "data/se/nbr_monthly.csv"},
            {"name": "Eurostat namq_10_gdp NGDP + D1 wages; prc_hicp_midx CPI; debt-book effective rates",
             "use": "macro drivers"},
            {"name": "Riksgälden Statsupplåning annual NBR (pin after category sum)",
             "use": "RG rest-of-year / annual anchor only after macro category path"},
        ],
        "note": ("Category×macro alternate; live cash stays RG×seasonal. Anchored PRE gate is "
                 "shape-only under the RG rest-of-year total — authority annuals do not move it."),
    }
    if write:
        contract.DERIVED.mkdir(parents=True, exist_ok=True)
        OUT_SCORE.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def attach_column(df: pd.DataFrame, d: dict, y: pd.Series) -> pd.Series:
    cats = categories(d)
    drivers = macro_drivers()
    debt_ye = debt_stock_ye(d)
    auth = load_authority()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    hist = ~df["is_forecast"]
    out.loc[hist] = df.loc[hist, "cash_requirement_bn"].to_numpy()
    origin = y.index[-1]
    fut = df.loc[df["is_forecast"], "month"]
    if len(fut):
        raw = forecast_nbr(cats, drivers, debt_ye, origin, hmax=len(fut) + 2, auth=auth)
        out.loc[df["is_forecast"]] = raw.reindex(fut).to_numpy()
    return out


def no_lookahead_ok(d: dict, asof: pd.Timestamp) -> bool:
    cats = categories(d)
    drivers = macro_drivers()
    debt_ye = debt_stock_ye(d)
    auth = load_authority()
    tgt = asof + pd.DateOffset(months=1)
    if tgt not in cats["nbr"].index:
        return True
    full = float(forecast_nbr(cats, drivers, debt_ye, asof, hmax=1, auth=auth).get(tgt, np.nan))
    trunc = {k: (s.loc[:asof] if hasattr(s, "loc") else s) for k, s in cats.items()}
    cut = float(forecast_nbr(trunc, drivers, debt_ye, asof, hmax=1, auth=auth).get(tgt, np.nan))
    if not np.isfinite(full) or not np.isfinite(cut):
        return True
    return abs(full - cut) < 1e-6


def run_report(d: dict, y: pd.Series, chosen: str, method: str,
               df: pd.DataFrame, write: bool = True) -> tuple[pd.DataFrame, dict]:
    report = build_report(d, y, chosen, method, write=write)
    out = df.copy()
    out["cash_requirement_bottomup_macro_bn"] = attach_column(out, d, y)
    # never replace live path from this module while SHIP is False
    return out, report
