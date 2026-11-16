"""Bottom-up cash requirement for France, Germany and Italy.

Each country is the published monthly split, not one seasonal series. A component that
has a base grows with the macro series that base is: receipts and tax with nominal GDP,
current spending with CPI. Both drivers are the previous complete calendar year, so a
forecast made in year Y uses Y−1. Components that change sign or have no base (special
accounts, other revenue, capital spending, the Italian treasury balance) repeat last year.

The cash requirement is the identity of those components. Covid years 2020 and 2021 are
left out of the fit. Fewer than six clean years and the component repeats last year.
A negative slope is set to zero and the intercept refit: the base is not allowed to
shrink when its macro series rises.

Scored on the 12-month sum against repeating last year (local-currency bn). It does
not replace the cash path. France PRE 17.2 against 13.3 (the slope floored at zero,
so this is a drift). A fixed pass-through of one half of nominal GDP into receipts
beats last year by 0.1 on the French PRE mean and loses to the seasonal model
(7.1 against 9.6). Year-to-date pace is worse in both France and Italy, including
when Italy scales only tax and current spending. Germany PRE 11.7 against 7.4 on
12 origins. The German path uses the spending plan and the revenue plan for the
open year (see ``de.model.open_year_core``); that split beats the single balance
haircut at every horizon, and last year still wins the pre-2020 window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C

MIN_YEARS = 6
SKIP_YEARS = {2020, 2021}
HORIZON = 12

# driver None repeats the same month last year.
# Interest is carved from published spending (see interest_split) and driven by
# the effective rate; primary spending keeps CPI; receipts keep NGDP.
SPEC = {
    "fr": [("recettes", "ngdp"), ("primary", "cpi"), ("interest", "rates"), ("speciaux", None)],
    "de": [("revenue", "ngdp"), ("primary", "cpi"), ("interest", "rates")],
    "it": [("tax_rev", "ngdp"), ("other_rev", None), ("primary", "cpi"), ("interest", "rates"),
           ("cap_exp", None), ("treasury_bal", None)],
}


# Cash requirement = sum of sign × line. + adds to borrowing (spending), − reduces it (receipts).
IDENTITY = {
    "fr": {"primary": 1, "interest": 1, "recettes": -1, "speciaux": -1},
    "de": {"primary": 1, "interest": 1, "revenue": -1},
    "it": {"primary": 1, "interest": 1, "cap_exp": 1, "tax_rev": -1, "other_rev": -1, "treasury_bal": -1},
}


def deficit(cc: str, parts: dict[str, pd.Series]) -> pd.Series:
    """The cash-requirement identity for ``cc`` (see ``IDENTITY``)."""
    terms = IDENTITY[cc]
    out = None
    for name, sign in terms.items():
        part = parts[name] if sign > 0 else -parts[name]
        out = part if out is None else out + part
    return out


def _eurostat_json(dataset: str, **filters: str) -> pd.DataFrame:
    import json
    import urllib.request
    q = "&".join(f"{k}={v}" for k, v in filters.items())
    url = (f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
           f"{dataset}?format=JSON&{q}")
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = json.loads(r.read())
    dims, sizes, vals = raw["id"], raw["size"], raw["value"]
    idx = {d: list(raw["dimension"][d]["category"]["index"].keys()) for d in dims}
    rows = []
    for k, v in vals.items():
        pos = int(k)
        coords = []
        for s in reversed(sizes):
            coords.append(pos % s)
            pos //= s
        lab = {d: idx[d][c] for d, c in zip(dims, reversed(coords))}
        rows.append({**lab, "value": float(v)})
    return pd.DataFrame(rows)


def _macro_eurostat(cc: str) -> dict[str, pd.Series] | None:
    """Calendar-year means of quarterly nominal-GDP YoY and monthly HICP YoY.

    The fiscal components are monthly; AMECO/WEO are annual only. Quarterly GDP
    (namq_10_gdp) and monthly HICP (prc_hicp_midx) are the matching frequencies.
    Failed fetch returns None so the caller can fall back.
    """
    from pathlib import Path
    geo = cc.upper()
    cache = Path(__file__).resolve().parent / "data" / "debt" / f"macro_hq_{cc}.parquet"
    try:
        if cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).days < 7:
            df = pd.read_parquet(cache)
            return {c: df[c].dropna() for c in ("ngdp", "cpi") if c in df}
        gdp = _eurostat_json("namq_10_gdp", freq="Q", unit="CP_MEUR", s_adj="NSA",
                             na_item="B1GQ", geo=geo)
        gdp = gdp[gdp["time"].str.contains(r"Q[1-4]$", regex=True)].copy()
        gdp["q"] = pd.PeriodIndex(gdp["time"], freq="Q").to_timestamp(how="start")
        g = gdp.set_index("q")["value"].sort_index().pct_change(4) * 100.0
        hicp = _eurostat_json("prc_hicp_midx", freq="M", unit="I15", coicop="CP00", geo=geo)
        h = (hicp.assign(month=pd.to_datetime(hicp["time"]))
             .set_index("month")["value"].sort_index().pct_change(12) * 100.0)
        ngdp = g.groupby(g.index.year).mean()
        cpi = h.groupby(h.index.year).mean()
        # only complete years (4 quarters / 12 months of YoY)
        ngdp = ngdp[g.groupby(g.index.year).count() >= 4]
        cpi = cpi[h.groupby(h.index.year).count() >= 12]
        out = pd.DataFrame({"ngdp": ngdp, "cpi": cpi})
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
        return {"ngdp": ngdp.dropna(), "cpi": cpi.dropna()}
    except Exception:
        return None


def macro(cc: str) -> dict[str, pd.Series]:
    """Annual percent drivers for component regressions.

    Prefer Eurostat quarterly nominal GDP YoY and monthly HICP YoY (calendar-year
    means). Fall back to AMECO levels and WEO CPI when the fetch is cold.
    ``rates`` is the effective interest rate (interest/debt × 100) from the debt book.
    """
    from sovereign_cash.interest_split import annual_effective_rate
    rates = annual_effective_rate(cc)
    hq = _macro_eurostat(cc)
    if hq is not None and len(hq["ngdp"]) >= 8 and len(hq["cpi"]) >= 8:
        return {**hq, "rates": rates}
    from pathlib import Path
    base = Path(__file__).resolve().parent / "data" / "debt"
    ameco = pd.read_parquet(base / "ameco.parquet")
    weo = pd.read_parquet(base / "weo.parquet")
    g = (ameco[(ameco["country"] == cc) & (ameco["var"] == "gdp_lcu_bn")]
         .sort_values("year").set_index("year")["value"].astype(float))
    ngdp = (g / g.shift(1) - 1.0) * 100.0
    cpi = (weo[(weo["country"] == cc) & (weo["var"] == "cpi")]
           .set_index("year")["value"].astype(float).sort_index())
    return {"ngdp": ngdp.dropna(), "cpi": cpi.dropna(), "rates": rates}


def _annual_pairs(y: pd.Series, driver: pd.Series, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """YoY of a complete calendar year on the previous year's macro. Both in decimal."""
    last = origin.year - 1
    ys, xs = [], []
    for year in range(int(y.index.min().year) + 1, last + 1):
        if year in SKIP_YEARS or (year - 1) not in driver.index:
            continue
        a = y.loc[f"{year}-01-01":f"{year}-12-01"]
        b = y.loc[f"{year - 1}-01-01":f"{year - 1}-12-01"]
        if len(a) != 12 or len(b) != 12 or a.isna().any() or b.isna().any():
            continue
        base = float(b.sum())
        if base <= 0:
            continue
        ys.append(float(a.sum()) / base - 1.0)
        xs.append(float(driver.loc[year - 1]) / 100.0)
    return np.asarray(ys), np.asarray(xs)


def fit(y: pd.Series, driver: pd.Series, origin: pd.Timestamp) -> tuple[float, float] | None:
    """YoY = a + b × macro. None when the sample is short."""
    ys, xs = _annual_pairs(y, driver, origin)
    if len(ys) < MIN_YEARS:
        return None
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(xs)), xs]), ys, rcond=None)
    # a negative loading fights the base: receipts do not shrink when nominal GDP rises
    slope = max(float(coef[1]), 0.0)
    intercept = float(np.mean(ys) - slope * np.mean(xs))
    return intercept, slope


def forecast_component(y: pd.Series, driver: pd.Series | None, origin: pd.Timestamp,
                       h: int = HORIZON) -> pd.Series:
    """Same month last year, scaled by the fitted YoY when the component has a driver."""
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    base = []
    for m in months:
        prev = m - pd.DateOffset(years=1)
        base.append(float(y[prev]) if prev in y.index and pd.notna(y[prev]) else np.nan)
    out = pd.Series(base, index=months)
    if driver is None:
        return out
    coef = fit(y.loc[:origin], driver, origin)
    if coef is None or (origin.year - 1) not in driver.index:
        return out
    a, b = coef
    yoy = a + b * float(driver.loc[origin.year - 1]) / 100.0
    return out.where(out <= 0, out * (1.0 + yoy))


def components(cc: str) -> pd.DataFrame:
    """Monthly flows, local-currency bn. Italy's source is EUR mn.

    Spending is split into ``primary`` + ``interest`` via ``interest_split`` so
    rates can drive the interest line separately from CPI on the rest.
    """
    from sovereign_cash.interest_split import SPEND_COL, split as split_interest

    if cc == "fr":
        from sovereign_cash.fr import fetch as F
        smb = F.load_smb()
        m = {c: C.monthly_from_ytd(smb[c]) for c in
             ("recettes_ytd", "depenses_ytd", "comptes_speciaux_ytd")}
        df = pd.DataFrame(m).rename(columns={"recettes_ytd": "recettes", "depenses_ytd": "depenses",
                                             "comptes_speciaux_ytd": "speciaux"})
    elif cc == "de":
        from sovereign_cash.de.model import core_budget
        core = core_budget()
        df = pd.DataFrame({c: C.monthly_from_ytd(core[c].dropna())
                           for c in ("ytd_revenue", "ytd_expenditure")})
        df = df.rename(columns={"ytd_revenue": "revenue", "ytd_expenditure": "expenditure"})
    elif cc == "it":
        df = _italy() / 1000.0
    else:
        raise KeyError(cc)
    df = df.sort_index()
    spend_name = SPEND_COL[cc]
    carved = split_interest(cc, df[spend_name])
    df["interest"] = carved["interest"]
    df["primary"] = carved["primary"]
    return df


def _italy() -> pd.DataFrame:
    from pathlib import Path
    from sovereign_cash.it import sources as S
    cache = Path(S.RAW) / "bdi_t1.parquet"
    pdfs = list((Path(S.RAW) / "bdi_fpi").glob("*.pdf"))
    newest = max((p.stat().st_mtime for p in pdfs), default=0)
    if cache.exists() and cache.stat().st_mtime >= newest:
        return pd.read_parquet(cache)
    t1, _ = S.fetch_bdi(refresh=False)
    keep = ["tax_rev", "other_rev", "cur_exp", "cap_exp", "treasury_bal"]
    out = t1[keep].astype(float)
    out.to_parquet(cache)
    return out


def forecast(cc: str, origin: pd.Timestamp, h: int = HORIZON,
             frame: pd.DataFrame | None = None, drivers: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    frame = components(cc) if frame is None else frame
    drivers = macro(cc) if drivers is None else drivers
    cols = {}
    for name, key in SPEC[cc]:
        drv = None if key is None else drivers[key]
        cols[name] = forecast_component(frame[name].dropna(), drv, origin, h)
    out = pd.DataFrame(cols)
    out["cash_requirement"] = deficit(cc, {k: out[k] for k, _ in SPEC[cc]})
    return out


def score(cc: str, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cumulative 12-month error of the identity against repeating the aggregate."""
    frame = components(cc) if frame is None else frame
    drivers = macro(cc)
    y = deficit(cc, {name: frame[name] for name, _ in SPEC[cc]}).dropna()
    rows = []
    for origin in y.index:
        end = origin + pd.DateOffset(months=HORIZON)
        act = y.loc[origin + pd.offsets.MonthBegin(1):end]
        if len(act) != HORIZON or act.isna().any():
            continue
        fitted = True
        for name, key in SPEC[cc]:
            # rates history is short (debt book); interest falls back to smly when unfit
            if key and key != "rates" and fit(frame[name].dropna().loc[:origin], drivers[key], origin) is None:
                fitted = False
        if not fitted:
            continue
        fc = forecast(cc, origin, HORIZON, frame, drivers)["cash_requirement"]
        if fc.isna().any():
            continue
        smly = pd.Series([float(y.get(m - pd.DateOffset(years=1), np.nan)) for m in fc.index], index=fc.index)
        if smly.isna().any():
            continue
        seas = C.forecast(y, origin, HORIZON, "seas_mult_3")
        rows.append({"origin": origin, "end": end, "yoy": float(fc.sum()), "smly": float(smly.sum()),
                     "seas": float(seas.sum()) if seas is not None else np.nan, "actual": float(act.sum())})
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    pre_end = pd.Timestamp("2021-12-01") if cc == "it" else pd.Timestamp("2019-12-01")
    oos = pd.Timestamp("2022-01-01")
    bt["window"] = np.where(bt["end"] <= pre_end, "PRE", np.where(bt["origin"] >= oos, "OOS", "mid"))
    return bt


def mae(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt
    rows = []
    for window, g in bt.groupby("window"):
        rows.append({"window": window, "n": len(g),
                     "yoy": float((g["yoy"] - g["actual"]).abs().mean()),
                     "smly": float((g["smly"] - g["actual"]).abs().mean()),
                     "seas": float((g["seas"] - g["actual"]).abs().mean())})
    return pd.DataFrame(rows)


def _month_share(y: pd.Series, origin: pd.Timestamp, k: int = 3) -> pd.Series | None:
    """Mean share of each calendar month in a complete year, from years fully known."""
    shares = []
    for yr in range(int(origin.year) - 1, int(origin.year) - 12, -1):
        if yr in SKIP_YEARS:
            continue
        b = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if len(b) != 12 or b.isna().any() or float(b.sum()) <= 0:
            continue
        shares.append(pd.Series(b.to_numpy() / float(b.sum()), index=range(1, 13)))
        if len(shares) == k:
            break
    if not shares:
        return None
    return pd.concat(shares, axis=1).mean(axis=1)


def _execution(y: pd.Series, plan_by_year: dict[int, float], origin: pd.Timestamp,
               k: int = 5) -> float:
    """Outturn / plan on complete years before the origin. 1 when there is no history."""
    ratios = []
    for yr in range(int(origin.year) - 1, int(origin.year) - 12, -1):
        if yr in SKIP_YEARS or yr not in plan_by_year or plan_by_year[yr] <= 0:
            continue
        b = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if len(b) != 12 or b.isna().any():
            continue
        ratios.append(float(b.sum()) / plan_by_year[yr])
        if len(ratios) == k:
            break
    return float(np.mean(ratios)) if ratios else 1.0


def close_to_plan(y: pd.Series, plan: float, ratio: float, origin: pd.Timestamp,
                  h: int, share: pd.Series) -> pd.Series:
    """Rest of this year closes to plan × execution. A later year repeats that plan."""
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    ytd = float(y.loc[f"{origin.year}-01-01":origin].sum())
    target = plan * ratio
    rest = [m for m in range(int(origin.month) + 1, 13)]
    if rest:
        w = share.reindex(rest)
        w = w / float(w.sum())
        rem = target - ytd
    else:
        w, rem = None, None
    out = {}
    for m in months:
        if m.year == origin.year:
            out[m] = rem * float(w.loc[m.month])
        else:
            out[m] = target * float(share.loc[m.month])
    return pd.Series(out)


def pace(y: pd.Series, origin: pd.Timestamp, h: int, carry: bool) -> pd.Series | None:
    """Same-month last year, scaled by this year's year-to-date against last year's.

    ``carry`` keeps the scale into the next calendar year. Otherwise those months
    repeat last year unscaled.
    """
    start = pd.Timestamp(year=int(origin.year), month=1, day=1)
    now = y.loc[start:origin]
    then = y.loc[start - pd.DateOffset(years=1):origin - pd.DateOffset(years=1)]
    if now.empty or len(then) != len(now) or then.isna().any() or now.isna().any():
        return None
    base = float(then.sum())
    if base == 0:
        return None
    scale = float(now.sum()) / base
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = {}
    for m in months:
        prev = y.get(m - pd.DateOffset(years=1), np.nan)
        if pd.isna(prev):
            return None
        out[m] = float(prev) * (scale if carry or m.year == origin.year else 1.0)
    return pd.Series(out)


def passthrough(y: pd.Series, driver: pd.Series | None, beta: float,
                origin: pd.Timestamp, h: int) -> pd.Series | None:
    """Same-month last year times one plus beta times last complete year's macro."""
    growth = 0.0
    if driver is not None and beta:
        yr = int(origin.year) - 1
        if yr not in driver.index:
            return None
        growth = beta * float(driver.loc[yr]) / 100.0
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = {}
    for m in months:
        prev = y.get(m - pd.DateOffset(years=1), np.nan)
        if pd.isna(prev):
            return None
        out[m] = float(prev) * (1.0 + growth) if prev > 0 else float(prev)
    return pd.Series(out)


def _window(origin: pd.Timestamp, end: pd.Timestamp, cc: str) -> str:
    pre_end = pd.Timestamp("2021-12-01") if cc == "it" else pd.Timestamp("2019-12-01")
    if end <= pre_end:
        return "PRE"
    if origin >= pd.Timestamp("2022-01-01"):
        return "OOS"
    return "mid"


def _mae_table(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    out = []
    for window, g in bt.groupby("window"):
        rec = {"window": window, "n": len(g)}
        for c in cols:
            rec[c] = float((g[c] - g["actual"]).abs().mean())
        out.append(rec)
    return pd.DataFrame(out)


def search_macro(cc: str, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fixed pass-through of last year's nominal GDP and CPI, plus year-to-date pace.

    Receipts and tax take the GDP factor. Spending takes the CPI factor. Everything
    else repeats last year. Pace scales each line by its own year-to-date ratio.
    """
    frame = components(cc) if frame is None else frame
    drivers = macro(cc)
    names = [n for n, _ in SPEC[cc]]
    y = deficit(cc, {n: frame[n] for n in names}).dropna()
    betas = (0.0, 0.5, 1.0)
    rows = []
    for origin in y.index:
        end = origin + pd.DateOffset(months=HORIZON)
        act = y.loc[origin + pd.offsets.MonthBegin(1):end]
        if len(act) != HORIZON or act.isna().any():
            continue
        parts = {n: frame[n].dropna() for n in names}
        smly = {n: passthrough(parts[n], None, 0.0, origin, HORIZON) for n in names}
        if any(v is None for v in smly.values()):
            continue
        rec = {"origin": origin, "end": end, "window": _window(origin, end, cc),
               "actual": float(act.sum()),
               "smly": float(deficit(cc, smly).sum())}
        for br in betas:
            for bs in betas:
                grown = {}
                ok = True
                for n, key in SPEC[cc]:
                    beta = br if key == "ngdp" else bs if key == "cpi" else 0.0
                    grown[n] = passthrough(parts[n], drivers.get(key) if key else None, beta, origin, HORIZON)
                    if grown[n] is None:
                        ok = False
                if ok:
                    rec[f"b{br:g}_{bs:g}"] = float(deficit(cc, grown).sum())
        for carry, label in ((False, "pace"), (True, "pace_carry")):
            scaled = {}
            ok = True
            for n in names:
                scaled[n] = pace(parts[n], origin, HORIZON, carry)
                if scaled[n] is None:
                    ok = False
            if ok:
                rec[label] = float(deficit(cc, scaled).sum())
        agg = pace(y, origin, HORIZON, False)
        if agg is not None:
            rec["pace_agg"] = float(agg.sum())
        rows.append(rec)
    cols = ["smly", "pace", "pace_carry", "pace_agg"] + [f"b{br:g}_{bs:g}" for br in betas for bs in betas]
    present = [c for c in cols if any(c in r for r in rows)]
    return _mae_table(rows, present)


def search_de_plan() -> pd.DataFrame:
    """German core: close spending and revenue to their own Soll, against the balance haircut.

    The Soll used at an origin is the one printed by that month. The execution ratio is the
    mean of complete earlier years, Covid excluded. Next year's months inside the window
    repeat this year's plan (``plan``) or last year (``split``).
    """
    from sovereign_cash.de.fetch import load_bmf_mb
    from sovereign_cash.de.model import core_budget, haircut_at, haircuts

    core = core_budget()
    mb = load_bmf_mb()
    exp = C.monthly_from_ytd(core["ytd_expenditure"].dropna())
    rev = C.monthly_from_ytd(core["ytd_revenue"].dropna())
    cash = (exp - rev).dropna()
    final = {}
    for col in ("soll_expenditure", "soll_revenue", "soll_fin_balance"):
        out = {}
        for yr, g in mb.groupby(mb.index.year):
            g = g.dropna(subset=[col])
            g = g[g["soll_year"] == yr]
            if len(g):
                out[int(yr)] = float(g.iloc[-1][col])
        final[col] = out
    hc = haircuts(cash, core)
    rows = []
    for origin in cash.index:
        end = origin + pd.DateOffset(months=HORIZON)
        act = cash.loc[origin + pd.offsets.MonthBegin(1):end]
        if len(act) != HORIZON or act.isna().any():
            continue
        known = mb.loc[:origin].dropna(subset=["soll_expenditure", "soll_revenue", "soll_fin_balance"])
        known = known[known["soll_year"] == origin.year]
        if known.empty:
            continue
        plan_e, plan_r = float(known.iloc[-1]["soll_expenditure"]), float(known.iloc[-1]["soll_revenue"])
        plan_d = -float(known.iloc[-1]["soll_fin_balance"])
        sh_e, sh_r = _month_share(exp, origin), _month_share(rev, origin)
        if sh_e is None or sh_r is None:
            continue
        re, rr = _execution(exp, final["soll_expenditure"], origin), _execution(rev, final["soll_revenue"], origin)
        fe, fr = close_to_plan(exp, plan_e, re, origin, HORIZON, sh_e), close_to_plan(rev, plan_r, rr, origin, HORIZON, sh_r)
        raw_e, raw_r = close_to_plan(exp, plan_e, 1.0, origin, HORIZON, sh_e), close_to_plan(rev, plan_r, 1.0, origin, HORIZON, sh_r)
        # next calendar year repeats last year; this year still closes to the plan
        fe_s, fr_s = fe.copy(), fr.copy()
        for m in fe.index:
            if m.year != origin.year:
                pe, pr = exp.get(m - pd.DateOffset(years=1), np.nan), rev.get(m - pd.DateOffset(years=1), np.nan)
                if pd.isna(pe) or pd.isna(pr):
                    fe_s = None
                    break
                fe_s.loc[m], fr_s.loc[m] = float(pe), float(pr)
        smly = pd.Series([float(cash.get(m - pd.DateOffset(years=1), np.nan)) for m in fe.index], index=fe.index)
        ratio, _ = haircut_at(hc, int(origin.year))
        bal = C.forecast(cash, origin, HORIZON, "planhc_3",
                         targets={int(origin.year): plan_d, "hc": ratio})
        if smly.isna().any() or bal is None or fe_s is None:
            continue
        rows.append({"origin": origin, "end": end, "window": _window(origin, end, "de"),
                     "actual": float(act.sum()), "plan": float((fe - fr).sum()),
                     "raw": float((raw_e - raw_r).sum()), "split": float((fe_s - fr_s).sum()),
                     "planhc": float(bal.sum()), "smly": float(smly.sum())})
    return _mae_table(rows, ["plan", "raw", "split", "planhc", "smly"])
