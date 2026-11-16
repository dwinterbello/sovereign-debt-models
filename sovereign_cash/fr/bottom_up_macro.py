"""France v2 cash-requirement baseline: PLF tax/spend categories × RESF macro.

Identity (deficit +):
  deficit ≈ credits + charge + cas_pensions + remboursements_locaux + psr
           − (ir + is + tva + accises + autres_fiscales + non_fiscal)
           − solde_comptes_speciaux − solde_budgets_annexes

Open years with a PLF catégorie vintage known at the origin pin those annuals.
Otherwise most receipt/spend categories **hold** the last complete SMB/PLF year
(aggregate / IR/TVA/IS same-year RESF and OLS fail PRE — see ``rejected_trials``).
Shipped leaf exception: ``remboursements_locaux`` grows with same-year CPI prior.
Charge = AFT when known (2024 exécution 50.1 beats PLF chiffres 46.9 on OOS),
else ISIN+curve annual. Month shape = 5y SMB shares; interest months = 100% smly.
``SOFT_PIN_ALPHA=1`` — when the PLF solde is known at the origin, close the
category sum to that annual (PRE flat vs α=0; OOS 8.88 → 8.06). Categories remain
the shape; the pin only removes residual level drift vs the financing table.

v1 (``fr.bottom_up``) stays as seas_mult + AFT charge with SHIP=False.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.yoy_components import SPEC, SKIP_YEARS, components, deficit, macro as yoy_macro

AUTH = Path(__file__).resolve().parent.parent / "data" / "fr" / "authority.json"
REL_EDGE = 0.05
OOS_BAND = 0.25  # ship also needs OOS mean ≤ (1 + OOS_BAND) × production (aligned with IT/display equation)
MIN_YEARS = 6
SHARE_K = 5
CAT_KNOWN_MONTH = 10  # PLF deposit ~ early October Y−1 for year Y
CHARGE_KNOWN_MONTH = 9
SOFT_PIN_ALPHA = 1.0  # PLF solde pin when known — PRE-flat, OOS-better vs α=0
INT_SMLY_W = 1.0  # interest month carve: full same-month-last-year
# Open-year charge fill when AFT unknown: ISIN+curve vs hold.
# Fair open-year monthly vs smly −6.4% (isin_interest_bugfix.md). Never replaces
# an AFT pin that is honestly known at the origin (LFI/PLF, or exécution after YE).
INTEREST_ISIN_CURVE = True
# Open-year leaf growth: only these keys, same-year RESF/Voies prior (not lag-1 apply).
GROW_KEYS = frozenset({"remboursements_locaux"})
GROW_SAME_YEAR = True
USE_OLS = False

TAX_KEYS = ("ir", "is", "tva", "accises", "autres_fiscales")
SPEND_KEYS = ("credits_ministeriels", "cas_pensions", "remboursements_locaux", "psr")
RECEIPT_KEYS = TAX_KEYS + ("non_fiscal",)

# Frozen PRE walk-forward (bu 7.44 vs prod 8.13, +8.5%, worse_h=1; OOS ~8.06 with α=1).
SHIP = True


def _auth() -> dict:
    return json.loads(AUTH.read_text()) if AUTH.exists() else {}


def _cat_table() -> dict[int, dict]:
    raw = _auth().get("categories", {}).get("years", {})
    return {int(y): dict(r) for y, r in raw.items()}


def _elast_table() -> dict[str, dict]:
    return dict(_auth().get("elasticities", {}))


def _charge_table() -> dict[int, dict]:
    """year → {bn, vintage} from authority.json."""
    raw = _auth().get("debt_charge", {}).get("years", {})
    return {int(y): {"bn": float(r["bn"]), "vintage": str(r.get("vintage", ""))}
            for y, r in raw.items()}


def _aft_known_at(origin: pd.Timestamp, year: int, vintage: str) -> bool:
    """Whether the AFT charge for ``year`` is known at ``origin`` (no exécution lookahead)."""
    vin = vintage.lower()
    if "exécution" in vin or "execution" in vin or "outturn" in vin:
        return int(origin.year) > int(year)  # year must be complete
    return origin >= pd.Timestamp(year - 1, CHARGE_KNOWN_MONTH, 1)


def _resf_macro() -> dict[int, dict]:
    raw = _auth().get("macro", {}).get("years", {})
    out = {}
    for y, r in raw.items():
        yi = int(y)
        real = float(r["real_growth"])
        cpi = float(r["cpi"])
        out[yi] = {
            "real_growth": real,
            "cpi": cpi,
            "ngdp": real + cpi,
            "masse_salariale": float(r.get("masse_salariale", real + cpi)),
        }
    return out


def _plf_known(origin: pd.Timestamp, year: int) -> bool:
    return origin >= pd.Timestamp(year - 1, CAT_KNOWN_MONTH, 1)


def _annual_series(y: pd.Series) -> pd.Series:
    rows = {}
    for yr, g in y.groupby(y.index.year):
        if len(g) == 12 and g.notna().all():
            rows[int(yr)] = float(g.sum())
    return pd.Series(rows).sort_index()


def _month_share(y: pd.Series, origin: pd.Timestamp, k: int | None = None) -> pd.Series | None:
    k = SHARE_K if k is None else k
    shares = []
    for yr in range(int(origin.year) - 1, int(origin.year) - 12, -1):
        b = y.loc[f"{yr}-01-01":f"{yr}-12-01"]
        s = float(b.sum()) if len(b) == 12 and b.notna().all() else 0.0
        if abs(s) < 1e-9:
            continue
        shares.append(pd.Series(b.to_numpy() / s, index=range(1, 13)))
        if len(shares) == k:
            break
    if not shares:
        return None
    return pd.concat(shares, axis=1).mean(axis=1)


def _moy_mean(y: pd.Series, origin: pd.Timestamp, h: int, n: int = 60) -> pd.Series:
    hist = y.dropna().loc[:origin].iloc[-n:]
    g = hist.groupby(hist.index.month).mean()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pd.Series([float(g.get(m.month, 0.0)) for m in months], index=months)


def _structural_tax_shares(cats: dict[int, dict], origin: pd.Timestamp) -> dict[str, float]:
    """RFN split from the latest category year known at origin; else 2024 exécution prior."""
    for y in sorted((yy for yy in cats if _plf_known(origin, yy) or yy < origin.year), reverse=True):
        row = cats[y]
        vals = {k: row.get(k) for k in TAX_KEYS}
        if any(v is None for v in vals.values()):
            continue
        tot = sum(float(v) for v in vals.values())
        if tot <= 0:
            continue
        return {k: float(vals[k]) / tot for k in TAX_KEYS}
    # long-run RFN mix prior (IR/TVA-heavy); used only when no PLF year is known at origin
    return {"ir": 0.27, "is": 0.17, "tva": 0.30, "accises": 0.07, "autres_fiscales": 0.19}


def _structural_spend_shares(cats: dict[int, dict], origin: pd.Timestamp) -> dict[str, float]:
    for y in sorted((yy for yy in cats if _plf_known(origin, yy) or yy < origin.year), reverse=True):
        row = cats[y]
        vals = {k: row.get(k) for k in SPEND_KEYS}
        if any(v is None for v in vals.values()):
            continue
        tot = sum(float(v) for v in vals.values())
        if tot <= 0:
            continue
        return {k: float(vals[k]) / tot for k in SPEND_KEYS}
    return {"credits_ministeriels": 0.73, "cas_pensions": 0.11,
            "remboursements_locaux": 0.01, "psr": 0.15}


def _smb_category_annuals(frame: pd.DataFrame, origin: pd.Timestamp,
                          year: int) -> dict[str, float] | None:
    """Allocate a complete SMB year into PLF categories via structural shares."""
    cats = _cat_table()
    rec = _annual_series(frame["recettes"].dropna())
    pri = _annual_series(frame["primary"].dropna())
    inte = _annual_series(frame["interest"].dropna())
    spe = _annual_series(frame["speciaux"].dropna())
    if year not in rec.index or year not in pri.index:
        return None
    tax_sh = _structural_tax_shares(cats, origin)
    spend_sh = _structural_spend_shares(cats, origin)
    # non-fiscal share of SMB recettes from latest known PLF row
    nf_share = 0.06
    for y in sorted((yy for yy in cats if _plf_known(origin, yy) or yy < origin.year), reverse=True):
        row = cats[y]
        rfn = row.get("recettes_fiscales_nettes")
        nf = row.get("non_fiscal")
        if rfn is not None and nf is not None and float(rfn) + float(nf) > 0:
            nf_share = float(nf) / (float(rfn) + float(nf))
            break
    r_tot = float(rec.loc[year])
    nf = r_tot * nf_share
    rfn = r_tot - nf
    out = {k: rfn * tax_sh[k] for k in TAX_KEYS}
    out["non_fiscal"] = nf
    p_tot = float(pri.loc[year])
    out.update({k: p_tot * spend_sh[k] for k in SPEND_KEYS})
    out["charge"] = float(inte.loc[year]) if year in inte.index else 0.0
    out["solde_comptes_speciaux"] = float(spe.loc[year]) if year in spe.index else 0.0
    out["solde_budgets_annexes"] = 0.0
    return out


def _plf_categories(year: int, origin: pd.Timestamp) -> dict[str, float] | None:
    cats = _cat_table()
    if year not in cats or not _plf_known(origin, year):
        return None
    row = cats[year]
    need = list(RECEIPT_KEYS) + list(SPEND_KEYS) + ["charge", "solde_comptes_speciaux"]
    if any(row.get(k) is None for k in need):
        # partial row (e.g. 2024): fall back to SMB allocation for missing spend
        return None
    out = {k: float(row[k]) for k in need}
    out["solde_budgets_annexes"] = float(row["solde_budgets_annexes"] or 0.0)
    # Prefer AFT charge when honestly known at origin (no exécution lookahead)
    charge = _charge_table()
    if year in charge and _aft_known_at(origin, year, charge[year]["vintage"]):
        out["charge"] = charge[year]["bn"]
    return out


def _driver_panel(origin: pd.Timestamp) -> dict[str, pd.Series]:
    """Annual % drivers known at origin (Eurostat history + RESF forward)."""
    m = yoy_macro("fr")
    resf = _resf_macro()
    ngdp = m["ngdp"].copy()
    cpi = m["cpi"].copy()
    real = (ngdp - cpi).rename("real_growth")
    # masse salariale: RESF when known; else NGDP as proxy (same units %)
    wages = ngdp.copy().rename("masse_salariale")
    for y, r in resf.items():
        if origin >= pd.Timestamp(y - 1, CAT_KNOWN_MONTH, 1) or y < origin.year:
            ngdp.loc[y] = r["ngdp"]
            cpi.loc[y] = r["cpi"]
            real.loc[y] = r["real_growth"]
            wages.loc[y] = r["masse_salariale"]
    rates = m.get("rates", pd.Series(dtype=float))
    return {"ngdp": ngdp, "cpi": cpi, "real_growth": real,
            "masse_salariale": wages, "rates": rates}


def _fit_epsilon(cat_hist: pd.Series, driver: pd.Series, origin: pd.Timestamp,
                 prior: float, lag: int = 0) -> float:
    """Expanding OLS of category YoY on driver; floor at 0; fall back to prior.

    ``lag=0`` fits same-year driver (RESF timing); ``lag=1`` fits yr−1 (IS).
    Shipped path uses ``USE_OLS=False`` — lag-1 OLS is wrong-signed on SMB
    proxies; same-year OLS (~+2.5) fails the PRE ship gate.
    """
    if not USE_OLS:
        return float(prior)
    last = int(origin.year) - 1
    ys, xs = [], []
    for yr in cat_hist.index:
        if yr > last or yr in SKIP_YEARS or (yr - 1) not in cat_hist.index:
            continue
        use = yr - int(lag)
        if use not in driver.index:
            continue
        base = float(cat_hist.loc[yr - 1])
        if abs(base) < 1e-6:
            continue
        ys.append(float(cat_hist.loc[yr]) / base - 1.0)
        xs.append(float(driver.loc[use]) / 100.0)
    if len(ys) < MIN_YEARS:
        return float(prior)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(xs)), xs]), ys, rcond=None)
    e = float(coef[1])
    return e if e > 1e-12 else float(prior)


def _category_history(frame: pd.DataFrame, origin: pd.Timestamp) -> dict[str, pd.Series]:
    """Annual category levels through the last complete year at origin."""
    cats = _cat_table()
    years = sorted(set(_annual_series(frame["recettes"].dropna()).index))
    years = [y for y in years if y < origin.year or (
        y == origin.year and origin.month == 12)]
    hist: dict[str, list] = {k: [] for k in list(RECEIPT_KEYS) + list(SPEND_KEYS)
                             + ["charge", "solde_comptes_speciaux", "solde_budgets_annexes"]}
    idx = []
    for y in years:
        row = None
        if y in cats and (_plf_known(origin, y) or y < origin.year):
            c = cats[y]
            if all(c.get(k) is not None for k in RECEIPT_KEYS) and all(
                    c.get(k) is not None for k in SPEND_KEYS):
                row = {k: float(c[k]) for k in hist}
                if c.get("solde_budgets_annexes") is None:
                    row["solde_budgets_annexes"] = 0.0
                ch = _charge_table()
                if y in ch and _aft_known_at(origin, y, ch[y]["vintage"]):
                    row["charge"] = ch[y]["bn"]
        if row is None:
            row = _smb_category_annuals(frame, origin, y)
        if row is None:
            continue
        idx.append(y)
        for k in hist:
            hist[k].append(float(row.get(k, 0.0)))
    return {k: pd.Series(v, index=idx) for k, v in hist.items()}


def _grow_categories(base: dict[str, float], origin: pd.Timestamp, year: int,
                     hist: dict[str, pd.Series], drivers: dict[str, pd.Series]
                     ) -> dict[str, float]:
    """Advance base to ``year``: leaf ε·Δmacro on ``GROW_KEYS`` only; AFT on charge."""
    elast = _elast_table()
    out = dict(base)
    past = [y for y in hist.get("ir", pd.Series(dtype=float)).index if y < origin.year]
    if not past:
        return out
    base_yr = max(past)
    for y in range(base_yr + 1, year + 1):
        for key in GROW_KEYS:
            spec = elast.get(key, {})
            dname = spec.get("driver")
            prior = spec.get("epsilon")
            if dname is None or prior is None:
                continue
            lag = int(spec.get("lag") or 0)
            if GROW_SAME_YEAR:
                use_y = y
                fit_lag = 0
            else:
                use_y = (y - 1) - lag
                fit_lag = lag
            if dname not in drivers or use_y not in drivers[dname].index:
                continue
            eps = _fit_epsilon(hist[key], drivers[dname], origin, float(prior), lag=fit_lag)
            g = eps * float(drivers[dname].loc[use_y]) / 100.0
            out[key] = float(out[key]) * (1.0 + g)
        # charge: AFT when honestly known; else ISIN+curve open-year fill (or hold)
        charge = _charge_table()
        if y in charge and _aft_known_at(origin, y, charge[y]["vintage"]):
            out["charge"] = charge[y]["bn"]
        elif INTEREST_ISIN_CURVE:
            try:
                from sovereign_cash import isin_curve_interest as IC
                v = IC.annual_bn("fr", y)
                if np.isfinite(v) and v > 0:
                    out["charge"] = float(v)
            except Exception:
                pass
    return out


def category_annuals(origin: pd.Timestamp, year: int,
                     frame: pd.DataFrame | None = None) -> dict[str, float] | None:
    """PLF pin when known; else macro-grown categories from last complete SMB/PLF year."""
    frame = components("fr") if frame is None else frame
    pinned = _plf_categories(year, origin)
    if pinned is not None:
        return pinned
    hist = _category_history(frame, origin)
    if not len(hist.get("ir", pd.Series(dtype=float))):
        return None
    past = [y for y in hist["ir"].index if y < origin.year]
    if not past:
        return None
    base_yr = int(max(past))
    base = {k: float(hist[k].loc[base_yr]) for k in hist}
    if year == base_yr:
        return base
    drivers = _driver_panel(origin)
    return _grow_categories(base, origin, year, hist, drivers)


def _identity_annual(cats: dict[str, float]) -> float:
    spend = (cats["credits_ministeriels"] + cats["charge"] + cats["cas_pensions"]
             + cats["remboursements_locaux"] + cats["psr"])
    receipts = sum(cats[k] for k in RECEIPT_KEYS)
    return (spend - receipts - cats.get("solde_comptes_speciaux", 0.0)
            - cats.get("solde_budgets_annexes", 0.0))


def _to_smb_parts(cats: dict[str, float]) -> dict[str, float]:
    """Map PLF categories onto the SMB four-line identity used by the scorer."""
    return {
        "recettes": sum(cats[k] for k in RECEIPT_KEYS),
        "primary": sum(cats[k] for k in SPEND_KEYS),
        "interest": cats["charge"],
        "speciaux": cats.get("solde_comptes_speciaux", 0.0),
    }


def _soft_pin(path: pd.Series, hist: pd.Series, origin: pd.Timestamp,
              alpha: float) -> pd.Series:
    """Blend path toward authority state_deficit annuals known at origin."""
    if alpha <= 0:
        return path
    plans = {int(y): float(r["bn"])
             for y, r in _auth().get("state_deficit", {}).get("years", {}).items()
             if _plf_known(origin, int(y))}
    if not plans:
        return path
    hard = C.anchor_to_plan(path, hist, origin, plans)
    return path + alpha * (hard - path)


def _interest_month_carve(interest: pd.Series, hist_i: pd.Series, w: float) -> pd.Series:
    if w <= 0:
        return interest
    smly = []
    for m in interest.index:
        v = hist_i.get(m - pd.DateOffset(years=1), np.nan)
        smly.append(float(v) if pd.notna(v) else float(interest.loc[m]))
    return (1.0 - w) * interest + w * np.asarray(smly, dtype=float)


def forecast(origin: pd.Timestamp, h: int = 12,
             frame: pd.DataFrame | None = None,
             close_to_plan: bool | None = None,
             soft_pin_alpha: float | None = None,
             share_k: int | None = None,
             int_smly_w: float | None = None) -> pd.DataFrame | None:
    """Monthly SMB-component paths and cash_requirement through origin+1..h."""
    frame = components("fr") if frame is None else frame
    alpha = SOFT_PIN_ALPHA if soft_pin_alpha is None else soft_pin_alpha
    if close_to_plan is False:
        alpha = 0.0
    elif close_to_plan is True:
        alpha = 1.0 if soft_pin_alpha is None else soft_pin_alpha
    sk = SHARE_K if share_k is None else share_k
    iw = INT_SMLY_W if int_smly_w is None else int_smly_w
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    years = sorted({int(m.year) for m in months})
    annuals = {}
    for y in years:
        c = category_annuals(origin, y, frame)
        if c is None:
            return None
        annuals[y] = c
    sh_r = _month_share(frame["recettes"].dropna().loc[:origin], origin, k=sk)
    sh_p = _month_share(frame["primary"].dropna().loc[:origin], origin, k=sk)
    sh_i = _month_share(frame["interest"].dropna().loc[:origin], origin, k=sk)
    if sh_r is None or sh_p is None or sh_i is None:
        return None
    rows = []
    for m in months:
        parts = _to_smb_parts(annuals[int(m.year)])
        rows.append({
            "recettes": parts["recettes"] * float(sh_r.loc[m.month]),
            "primary": parts["primary"] * float(sh_p.loc[m.month]),
            "interest": parts["interest"] * float(sh_i.loc[m.month]),
        })
    out = pd.DataFrame(rows, index=months)
    out["interest"] = _interest_month_carve(
        out["interest"], frame["interest"].dropna().loc[:origin], iw)
    out["speciaux"] = _moy_mean(frame["speciaux"], origin, h)
    out["cash_requirement"] = deficit("fr", {n: out[n] for n, _ in SPEC["fr"]})
    y_hist = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    out["cash_requirement"] = _soft_pin(out["cash_requirement"], y_hist, origin, alpha)
    out.attrs["categories"] = {str(y): annuals[y] for y in annuals}
    out.attrs["identity_annual"] = {str(y): _identity_annual(annuals[y]) for y in annuals}
    out.attrs["soft_pin_alpha"] = alpha
    out.attrs["share_k"] = sk
    out.attrs["int_smly_w"] = iw
    return out


def path(origin: pd.Timestamp, h: int = 12,
         frame: pd.DataFrame | None = None,
         close_to_plan: bool | None = None,
         soft_pin_alpha: float | None = None) -> pd.Series | None:
    fc = forecast(origin, h, frame=frame, close_to_plan=close_to_plan,
                  soft_pin_alpha=soft_pin_alpha)
    return None if fc is None else fc["cash_requirement"]


def production_path(y: pd.Series, origin: pd.Timestamp, h: int = 12) -> pd.Series | None:
    return C.forecast(y, origin, h, "seas_mult_3")


def backtest(frame: pd.DataFrame | None = None, step: int = 1,
             close_to_plan: bool | None = None,
             soft_pin_alpha: float | None = None) -> pd.DataFrame:
    frame = components("fr") if frame is None else frame
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    rows = []
    for i, origin in enumerate(y.index):
        if i % step:
            continue
        if origin.year < 2012:
            continue
        bu = path(origin, 12, frame, close_to_plan=close_to_plan,
                  soft_pin_alpha=soft_pin_alpha)
        prod = production_path(y, origin, 12)
        if bu is None or prod is None:
            continue
        for h in C.HORIZONS:
            end = origin + pd.DateOffset(months=h)
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h or act.isna().any():
                continue
            rows.append({
                "origin": origin, "h": int(h),
                "bottomup": float(bu.iloc[:h].sum()),
                "production": float(prod.iloc[:h].sum()),
                "actual": float(act.sum()),
            })
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    ends = [o + pd.DateOffset(months=int(h)) for o, h in zip(bt["origin"], bt["h"])]
    bt["window"] = np.where(pd.to_datetime(ends) <= C.PRE_END, "PRE",
                            np.where(bt["origin"] >= C.OOS_START, "OOS", "mid"))
    return bt


def mae_table(bt: pd.DataFrame) -> pd.DataFrame:
    if bt.empty:
        return bt
    rows = []
    for (w, h), g in bt.groupby(["window", "h"]):
        rows.append({
            "window": w, "h": int(h), "n": len(g),
            "bottomup": float((g["bottomup"] - g["actual"]).abs().mean()),
            "production": float((g["production"] - g["actual"]).abs().mean()),
        })
    return pd.DataFrame(rows)


def mean_mae(bt: pd.DataFrame, window: str, col: str) -> float | None:
    tab = mae_table(bt)
    g = tab[tab["window"] == window]
    if g.empty or set(g["h"]) != set(C.HORIZONS):
        return None
    return float(g.set_index("h").loc[list(C.HORIZONS), col].mean())


def ship_decision(bt: pd.DataFrame | None = None) -> dict:
    bt = backtest() if bt is None else bt
    tab = mae_table(bt)
    pre = tab[tab["window"] == "PRE"].set_index("h")
    if pre.empty or set(pre.index) != set(C.HORIZONS):
        return {"ship": False, "reason": "PRE incomplete", "pre_bu": None, "pre_prod": None}
    bu = float(pre.loc[list(C.HORIZONS), "bottomup"].mean())
    prod = float(pre.loc[list(C.HORIZONS), "production"].mean())
    worse = int((pre["bottomup"] > pre["production"] + 1e-9).sum())
    rel = (prod - bu) / prod if prod else 0.0
    pre_ok = bu < prod * (1.0 - REL_EDGE) and worse <= 1
    oos = tab[tab["window"] == "OOS"].set_index("h")
    oos_bu = float(oos.loc[list(C.HORIZONS), "bottomup"].mean()) if set(oos.index) >= set(C.HORIZONS) else None
    oos_prod = float(oos.loc[list(C.HORIZONS), "production"].mean()) if set(oos.index) >= set(C.HORIZONS) else None
    oos_ok = (oos_bu is None or oos_prod is None
              or oos_bu <= (1.0 + OOS_BAND) * oos_prod + 1e-12)
    ok = pre_ok and oos_ok
    reason = (f"PRE bu {bu:.2f} vs prod {prod:.2f} ({rel:+.1%} rel, worse_h={worse})"
              + (f"; OOS {oos_bu:.2f} vs {oos_prod:.2f}" if oos_bu is not None and oos_prod is not None else ""))
    if not pre_ok:
        reason = f"PRE fail: {reason}"
    elif not oos_ok:
        reason = f"OOS fail (>{100 * OOS_BAND:.0f}% band): {reason}"
    else:
        reason = f"{reason}; SHIP"
    return {
        "ship": ok,
        "reason": reason,
        "pre_bu": bu, "pre_prod": prod, "rel": rel, "worse_h": worse,
        "oos_bu": oos_bu, "oos_prod": oos_prod, "oos_ok": oos_ok,
        "by_h": tab.to_dict(orient="records"),
    }


def category_audit(origin: pd.Timestamp | None = None,
                   frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """category → driver → rule → PRE category MAE (annual identity pieces)."""
    frame = components("fr") if frame is None else frame
    origin = origin or (frame.dropna().index.max() - pd.DateOffset(months=14))
    elast = _elast_table()
    hist = _category_history(frame, origin)
    drivers = _driver_panel(origin)
    rows = []
    skip = {"source", "ship_knobs"}
    for key, spec in elast.items():
        if key in skip or not isinstance(spec, dict) or "driver" not in spec:
            continue
        dname = spec.get("driver")
        prior = spec.get("epsilon")
        lag = int(spec.get("lag") or 0)
        if dname is None:
            rule = "PLF pin when known; else hold"
        elif key in GROW_KEYS:
            timing = "same-year" if GROW_SAME_YEAR else f"lag-{lag} apply"
            rule = (f"PLF pin when known; else level×(1+ε·Δ{dname}) {timing}; "
                    f"ε={prior} RESF/Voies prior; OLS off")
        else:
            rule = (f"PLF pin when known; else hold last year "
                    f"(RESF ε={prior} on {dname} documented; leaf growth fails PRE)")
        if key == "charge":
            rule = (f"AFT charge × month share when known; else hold; "
                    f"month carve {INT_SMLY_W:.0%} smly")
        mae = None
        if key in hist and len(hist[key]) >= 3 and dname and dname in drivers and prior is not None:
            errs = []
            ser = hist[key]
            for yr in ser.index:
                if yr <= ser.index.min() or yr in SKIP_YEARS:
                    continue
                if yr >= origin.year:
                    continue
                use = yr - lag
                if use not in drivers[dname].index or (yr - 1) not in ser.index:
                    continue
                eps = _fit_epsilon(ser.loc[:yr - 1], drivers[dname],
                                   pd.Timestamp(yr - 1, 12, 1), float(prior), lag=lag)
                pred = float(ser.loc[yr - 1]) * (1.0 + eps * float(drivers[dname].loc[use]) / 100.0)
                errs.append(abs(pred - float(ser.loc[yr])))
            mae = float(np.mean(errs)) if errs else None
        rows.append({
            "category": key,
            "driver": dname,
            "epsilon_prior": prior,
            "rule": rule,
            "authority_key": f"categories.years[*].{key}",
            "pre_category_mae_bn": mae,
        })
    return pd.DataFrame(rows)


def authority_tables_used() -> list[dict]:
    auth = _auth()
    return [
        {"title": "PLF équilibre / Voies et moyens / chiffres clés — recettes + spend categories",
         "vintage": "PLF 2025 exéc 2023 + PLF 2026 chiffres/CdC 2024 + PLF 2027 (2025–27); 2022 in published_execution",
         "path": "data/fr/authority.json → categories (+ plf_extracts/)"},
        {"title": "DG Trésor RESF — real growth, CPI, masse salariale",
         "vintage": auth.get("macro", {}).get("years", {}).get("2027", {}).get("vintage"),
         "path": "data/fr/authority.json → macro"},
        {"title": "AFT charge budgétaire de la dette (programme 117 / key figures)",
         "vintage": "AFT exécution 2021–24 + LFI/PLF 2025–27; 2024 pin = 50.1 exécution",
         "path": "data/fr/authority.json → debt_charge"},
        {"title": "PLF / Budget solde budgétaire (déficit à financer) — soft pin α=1",
         "vintage": auth.get("state_deficit", {}).get("years", {}).get("2027", {}).get("vintage"),
         "path": "data/fr/authority.json → state_deficit"},
        {"title": "SMB recettes / dépenses / comptes spéciaux (INSEE BDM)",
         "vintage": "monthly outturn",
         "path": "fr/fetch.load_smb → month shares (k=5) + history"},
    ]


def rejected_trials() -> list[str]:
    """PRE-gated search notes (OOS reported, never selected on)."""
    return [
        "BASE k=3 no soft pin — PRE 7.50 / OOS 10.40",
        "soft pin α continuum 0–1 — PRE flat; OOS improves monotonically → α=1",
        "SHARE_K 2/3/5/8 — k=5 best OOS among PRE-clearers",
        "lag-1 expanding OLS — coef wrong-signed (ε floored 0); inert",
        "same-year expanding OLS (aggregate all cats) — ε≈+2.5; PRE fails (−39% vs prod)",
        "same-year RESF prior (all cats) — PRE fails (+3.8%, worse_h=3)",
        "lag-1 RESF prior + k=5 + int + soft — PRE 7.59 clears; OOS 9.57 worse than hold",
        "hold + k=5 + int_smly 0.5 + soft α=1 — PRE 7.52 / OOS 8.92 (prior SHIP)",
        "charge rate-scale (effective yield) — worse_h=2; do not retry without structural change",
        "AFT +100bp triangle × Δrates on open charge — clears PRE; OOS ≥ hold (0.25/0.5/1.0)",
        "same-year leaf IR/TVA/IS/autres — worse_h=2; half-ε tax still fails PRE",
        "same-year Voies RFN year-ε (1.2/0.5) on all taxes — PRE −18%",
        "same-year credits_ministeriels — PRE clears hard; OOS 11.8–12.4 (reject)",
        "same-year cas_pensions / psr / spend-all — clear PRE; OOS ≥ hold",
        "same-year accises — clears; OOS tie/slight; remb alone better with int=1 + 2024 fill",
        "int_smly 0/0.25/0.5/0.75 — 1.0 best OOS among clearers",
        "recettes / primary smly blend — OOS worse or no gain",
        "OOS error decomp h=12 (pre-knobs): recettes bias −13.5, interest −6.2, cash +3.7",
        "2022 new-perimeter PLF fill (pin) — PRE ok; OOS 9.09 / with AFT50.1 8.97 (reject pin)",
        "remb+credits / remb+psr same-yr — clear PRE; OOS ≥8.85 under 2023+AFT50.1 (reject)",
        "AFT triangle × Δrates with remb — inert (OOS = hold) after AFT 2024 pin",
        "SHIP was α=0 (category level): PRE 7.44 / OOS 8.88 — flipped to α=1 2026-10-24 (same PRE, OOS 8.06)",
        "Open door: same-year tax leaf growth still fails PRE; recettes OOS bias is the residual after the pin",
    ]


if __name__ == "__main__":
    bt = backtest(step=1)
    d = ship_decision(bt)
    print(d["reason"])
    print("PRE", d["pre_bu"], "prod", d["pre_prod"], "rel", d.get("rel"))
    print("OOS", d["oos_bu"], "prod", d["oos_prod"])
    print(mae_table(bt).to_string(index=False))
    print(category_audit().to_string(index=False))
    for t in rejected_trials():
        print("trial:", t)
