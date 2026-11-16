"""Cash bottom-up category forecasts → annual fiscal bridge for the debt book.

SHIP countries (UK / DE / FR / IT) already drive monthly ``cash_requirement_bn`` when
wired. The debt board still grew central revenue with WEO GG revenue shares and kept
interest on the ISIN / curve book. This module annualises the *receipt* (and cash
interest) categories from each country's ``bottom_up_macro`` so ``debt.build.project``
can grow ``revenue_bn`` on that path.

Interest / GDP on the board stays the ISIN book (marketable coupons + bills). Cash DI /
AFT charge / BMF carve are written alongside as ``interest_cash_bu_bn`` — different
perimeter, not a substitute.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.bu_fiscal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DERIVED = HERE.parent / "derived"

# Receipt keys that map to central-government revenue growth (not spending / residual).
UK_RECEIPTS = ("it", "nics", "vat", "ct", "other_tax", "nontax")
FR_RECEIPTS = ("ir", "is", "tva", "accises", "autres_fiscales", "non_fiscal")
SHIP_CC = ("uk", "de", "fr", "it")


def _uk_later_betas() -> dict[str, tuple[str, float]]:
    """Leaf → (driver, β) as the UK cash bridge applies past the open year.

    Consumed by ``bu_inflation_sens.shipped_map`` (kind is attached there from
    ``UK_RECEIPTS``). Built from the same elas / SPEC the BU module ships.
    """
    from sovereign_cash.uk import bottom_up_macro as bu

    e = bu._rr_elas()
    s = bu.SPEC
    return {
        "it": ("wages", float(e["it_awe"])),
        "nics": ("wages", float(e["nics_awe"])),
        "vat": ("ngdp", float(e["vat_cons"])),
        "ct": ("ngdp", float(e["ct_profits"])),
        "other_tax": ("ngdp", float(s["other_tax_retail"])),
        "nontax": ("ngdp", float(s["nontax_gdp"])),
        "welfare": ("cpi", float(s["welfare_cpi"])),
        "rdel_proxy": ("ngdp", float(s.get("rdel_gdp", 0.0) or 0.0)),
        "cap": ("ngdp", float(s.get("cap_gdp", 0.0) or 0.0)),
    }


def _year_sums(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby(s.index.year).sum()


def _complete_open_year(hist: pd.Series, fc: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """Calendar-year sums: history through ``origin``, forecast after, no double count."""
    h = hist.dropna()
    f = fc.dropna()
    open_y = int(origin.year)
    parts = []
    # complete years before open year from history only
    if len(h):
        parts.append(h[h.index.year < open_y])
    # open year = hist ≤ origin + fc in open year
    h_open = h[(h.index.year == open_y) & (h.index <= origin)] if len(h) else h
    f_open = f[f.index.year == open_y] if len(f) else f
    if len(h_open) or len(f_open):
        open_sum = float(h_open.sum()) + float(f_open.sum())
        parts.append(pd.Series([open_sum], index=pd.DatetimeIndex([pd.Timestamp(open_y, 1, 1)])))
    # later years from forecast only (require full 12 months)
    if len(f):
        later = f[f.index.year > open_y]
        for y, g in later.groupby(later.index.year):
            if len(g) < 12:
                continue
            parts.append(pd.Series([float(g.sum())],
                                   index=pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1)])))
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    out.index = [int(t.year) for t in out.index]
    return out.groupby(level=0).sum()


def _uk(y0: int, y_end: int) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.uk import bottom_up_macro as bu

    auth = bu.authority()
    ship = bool(bu.SHIP and auth.get("ship", True))
    if not ship:
        return pd.DataFrame(), {"ship": False, "reason": "UK BU not shipped"}
    comp = bu.components()
    origin = pd.Timestamp(comp.index.max())
    # enough months for full calendar years through y_end
    h = max((y_end - origin.year + 1) * 12, 12)
    paths = bu.category_paths(comp, origin, h=h)
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    rows = []
    for k, arr in paths.items():
        fc = pd.Series(np.asarray(arr, dtype=float), index=months)
        hist = comp[k] if k in comp.columns else pd.Series(dtype=float)
        ann = _complete_open_year(hist, fc, origin)
        # extend any missing years past the last full fc year flat (drivers already in path)
        if len(ann):
            last_y, last_v = int(ann.index.max()), float(ann.iloc[-1])
            for y in range(last_y + 1, y_end + 1):
                ann.loc[y] = last_v
        kind = ("receipt" if k in UK_RECEIPTS
                else ("interest_cash" if k == "interest" else "other"))
        for y, v in ann.items():
            rows.append({"cc": "uk", "year": int(y), "category": k,
                         "value_bn": float(v), "kind": kind})
    # receipts_total
    rec_hist = comp[list(UK_RECEIPTS)].sum(axis=1)
    rec_fc = sum((pd.Series(np.asarray(paths[k], dtype=float), index=months)
                  for k in UK_RECEIPTS), start=pd.Series(0.0, index=months))
    for y, v in _complete_open_year(rec_hist, rec_fc, origin).items():
        rows.append({"cc": "uk", "year": int(y), "category": "receipts_total",
                     "value_bn": float(v), "kind": "receipt"})
    # fill forward receipts_total
    rec_ann = _complete_open_year(rec_hist, rec_fc, origin)
    if len(rec_ann):
        last_y, last_v = int(rec_ann.index.max()), float(rec_ann.iloc[-1])
        have = {int(r["year"]) for r in rows if r["category"] == "receipts_total"}
        for y in range(last_y + 1, y_end + 1):
            if y not in have:
                rows.append({"cc": "uk", "year": int(y), "category": "receipts_total",
                             "value_bn": last_v, "kind": "receipt"})
    meta = {"ship": True, "origin": str(origin.date()), "source": "uk.bottom_up_macro"}
    return pd.DataFrame(rows), meta


def _fr(y0: int, y_end: int) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.fr import bottom_up_macro as bu

    if not bu.SHIP:
        return pd.DataFrame(), {"ship": False, "reason": "FR BU SHIP=False"}
    # origin = last month with SMB history
    from sovereign_cash.fr import fetch as F
    smb = F.load_smb()
    origin = pd.Timestamp(smb.index.max()) if smb is not None and len(smb) else pd.Timestamp.today().normalize()
    rows = []
    for y in range(y0, y_end + 1):
        cats = bu.category_annuals(origin, y)
        if not cats:
            continue
        for k, v in cats.items():
            if v is None or not np.isfinite(float(v)):
                continue
            kind = ("receipt" if k in FR_RECEIPTS
                    else ("interest_cash" if k == "charge" else "other"))
            rows.append({"cc": "fr", "year": int(y), "category": k,
                         "value_bn": float(v), "kind": kind})
        rec = sum(float(cats[k]) for k in FR_RECEIPTS if cats.get(k) is not None
                  and np.isfinite(float(cats[k])))
        rows.append({"cc": "fr", "year": int(y), "category": "receipts_total",
                     "value_bn": rec, "kind": "receipt"})
    return pd.DataFrame(rows), {"ship": True, "origin": str(origin.date()),
                               "source": "fr.bottom_up_macro.category_annuals"}


def _extend_annual(base_year: int, base_val: float, y_end: int,
                   yoy: pd.Series | None, beta: float = 1.0) -> dict[int, float]:
    """Grow a calendar-year total past the smly horizon with lag-1 YoY × β."""
    out = {int(base_year): float(base_val)}
    for y in range(int(base_year) + 1, int(y_end) + 1):
        g = 0.0
        if yoy is not None and (y - 1) in yoy.index and np.isfinite(yoy.loc[y - 1]):
            g = beta * float(yoy.loc[y - 1]) / 100.0
        out[y] = out[y - 1] * (1.0 + g)
    return out


def _it(y0: int, y_end: int) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.it import bottom_up_macro as bu
    from sovereign_cash.yoy_components import macro

    v = bu.ship_rule()
    if not (bu.SHIP and v.get("ship")):
        return pd.DataFrame(), {"ship": False, "reason": v.get("reason", "IT not shipped")}
    frame = bu._frame_bn()
    origin = pd.Timestamp(frame.dropna(how="all").index.max())
    # smly only covers 12 months past origin; later years grow on NGDP.
    comps = bu.forecast_components(origin, 12)
    if comps is None:
        return pd.DataFrame(), {"ship": False, "reason": "IT forecast_components None"}
    drivers = macro("it")
    rows = []
    for col in comps.columns:
        if col == "cash_identity":
            continue
        kind = ("receipt" if col == "tax_rev"
                else ("interest_cash" if col == "interest" else "other"))
        ann = _complete_open_year(frame[col], comps[col], origin)
        if col == "tax_rev":
            beta, yoy = float(bu.BETA_TAX), drivers.get("ngdp")
        elif col == "primary":
            beta, yoy = float(bu.BETA_PRIMARY), drivers.get("cpi")
        else:
            beta, yoy = 0.0, None
        # grow from last complete year (open year is complete after splice)
        if len(ann):
            last_y, last_v = int(ann.index.max()), float(ann.iloc[-1])
            for y, val in _extend_annual(last_y, last_v, y_end, yoy, beta).items():
                ann.loc[y] = val
        for y, val in ann.items():
            rows.append({"cc": "it", "year": int(y), "category": col,
                         "value_bn": float(val), "kind": kind})
    tax_rows = [r for r in rows if r["category"] == "tax_rev"]
    for r in tax_rows:
        rows.append({**r, "category": "receipts_total", "kind": "receipt"})
    return pd.DataFrame(rows), {"ship": True, "origin": str(origin.date()),
                               "source": "it.bottom_up_macro"}


def _de(y0: int, y_end: int) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.de import bottom_up_macro as bu

    v = bu.ship_rule()
    if not v.get("ship"):
        return pd.DataFrame(), {"ship": False, "reason": v.get("reason", "DE not shipped")}
    from sovereign_cash.de import bottom_up as v1
    exp, rev, _, mb, maps = v1._flows()
    origin = pd.Timestamp(rev.dropna().index.max())
    paths = bu.revenue_interest_paths(origin, h=12)
    if paths is None:
        return pd.DataFrame(), {"ship": False, "reason": "DE revenue_interest_paths None"}
    drivers = bu.macro_drivers()
    rows = []
    hist_map = {"revenue": rev, "interest": None, "primary": None}
    for col, kind, beta_key, drv_key in (
            ("revenue", "receipt", "LATER_REV_BETA", "ngdp"),
            ("interest", "interest_cash", "LATER_INTEREST_BETA", "rates"),
            ("primary", "other", "LATER_PRIMARY_BETA", "cpi")):
        hist = hist_map[col] if hist_map[col] is not None else pd.Series(dtype=float)
        ann = _complete_open_year(hist, paths[col], origin)
        beta = float(getattr(bu, beta_key, 0.0))
        if len(ann):
            last_y, last_v = int(ann.index.max()), float(ann.iloc[-1])
            for y, val in _extend_annual(last_y, last_v, y_end, drivers.get(drv_key), beta).items():
                ann.loc[y] = val
        cat = "receipts_total" if col == "revenue" else col
        for y, val in ann.items():
            rows.append({"cc": "de", "year": int(y), "category": cat,
                         "value_bn": float(val), "kind": kind})
    return pd.DataFrame(rows), {"ship": True, "origin": str(origin.date()),
                               "source": "de.bottom_up_macro.revenue_interest_paths"}


_BUILDERS = {"uk": _uk, "fr": _fr, "it": _it, "de": _de}


def build(y0: int | None = None, y_end: int = 2031) -> tuple[pd.DataFrame, dict]:
    """Category-level annual table + meta for every SHIP cash-BU country."""
    y0 = int(pd.Timestamp.today().year - 1) if y0 is None else int(y0)
    parts, meta = [], {}
    for cc, fn in _BUILDERS.items():
        try:
            df, m = fn(y0, y_end)
        except Exception as e:  # noqa: BLE001 — per-country failure must not kill the debt build
            meta[cc] = {"ship": False, "reason": f"{type(e).__name__}: {e}"}
            continue
        meta[cc] = m
        if len(df):
            parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["cc", "year", "category", "value_bn", "kind"])
    return out, meta


def receipts_index(table: pd.DataFrame, cc: str, y0: int,
                   years: range | list[int]) -> dict[int, float] | None:
    """``receipts[y] / receipts[y0]`` from ``receipts_total``; None if y0 missing / non-positive."""
    t = table[(table["cc"] == cc) & (table["category"] == "receipts_total")]
    if t.empty:
        return None
    s = t.groupby("year")["value_bn"].sum()
    base = float(s.get(y0, np.nan))
    if not np.isfinite(base) or abs(base) < 1e-9:
        # fall back to last complete year ≤ y0
        prior = s[s.index <= y0]
        if prior.empty:
            return None
        y0 = int(prior.index.max())
        base = float(prior.iloc[-1])
        if abs(base) < 1e-9:
            return None
    return {int(y): float(s[y] / base) if y in s.index and np.isfinite(s[y]) else np.nan
            for y in years}


def interest_cash_path(table: pd.DataFrame, cc: str,
                       years: range | list[int]) -> dict[int, float]:
    t = table[(table["cc"] == cc) & (table["kind"] == "interest_cash")]
    if t.empty:
        return {}
    # prefer named charge / interest over duplicates
    s = t.groupby("year")["value_bn"].sum()
    return {int(y): float(s[y]) for y in years if y in s.index}


def write(table: pd.DataFrame | None = None, meta: dict | None = None) -> Path:
    table, meta = (table, meta) if table is not None else build()
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / "debt_bu_fiscal.parquet"
    table.to_parquet(path, index=False)
    (DERIVED / "debt_bu_fiscal.json").write_text(json.dumps(meta, indent=1, default=str))
    return path


def load() -> tuple[pd.DataFrame, dict]:
    p, j = DERIVED / "debt_bu_fiscal.parquet", DERIVED / "debt_bu_fiscal.json"
    if not p.exists():
        return build()
    meta = json.loads(j.read_text()) if j.exists() else {}
    return pd.read_parquet(p), meta


if __name__ == "__main__":
    t, m = build()
    write(t, m)
    print(json.dumps(m, indent=1, default=str))
    if len(t):
        print(t.groupby(["cc", "kind"])["value_bn"].count().to_string())
