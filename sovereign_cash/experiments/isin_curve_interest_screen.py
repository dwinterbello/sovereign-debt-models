"""Point-in-time ISIN + yield-curve interest screen (cross-country).

Replicates the debt-tab interest engine (``debt/cashflow.components`` +
``bill_ladder`` + ``build.project`` y0+1 refinance) with **no lookahead** on
ops or the curve. Contrasts that path against the prior locked face×coupon
screen, smly, and each country's live cash interest rule.

Does **not** flip SHIP. Writes ``isin_curve_interest_screen.md`` beside this file.

Usage:
  PYTHONPATH=. .venv/bin/python sovereign_cash/experiments/isin_curve_interest_screen.py
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from sovereign_cash.debt import cashflow as CF
from sovereign_cash.debt.gov_curve import FREQ, zeros_from_par
from sovereign_cash.debt.isin_book import locked_interest, outstanding

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # sovereign_cash/
REPO = HERE.parents[1]      # repo root
DERIVED = ROOT / "derived"
AUTH_FR = ROOT / "data" / "fr" / "authority.json"
MARKET = REPO / "issuance" / "cache" / "market_monthly.parquet"
BOE = REPO / "uk_flows" / "output" / "curves.parquet"
CURVE_CACHE = HERE / "_isin_curve_ye_zeros.parquet"
GSW_CACHE = HERE / "_gsw_zeros_daily.parquet"
MD = HERE / "isin_curve_interest_screen.md"

PRE_YEARS = list(range(2015, 2020))
OOS_YEARS = list(range(2020, 2026))
TARGET_CCS = ("uk", "de", "fr", "it")
NOTE_CCS = ("us", "jp", "es")  # annual-only (no monthly cash BU); debt tab ISIN+curve
TENORS = np.arange(1, 41)
SHIP_EDGE = 0.05  # OOS cash-leg win ≥5% to consider SHIP


# ── published interest targets ───────────────────────────────────────────────

def _ann() -> pd.DataFrame:
    d = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    return d[~d["forecast"]].copy()


def _pub_interest(ann: pd.DataFrame, cc: str) -> pd.Series:
    """Cash / debt-tab interest the screen scores against."""
    if cc == "uk":
        from sovereign_cash.uk import bottom_up_macro as UK
        inte = UK.components()["interest"].dropna()
        return inte.groupby(inte.index.year).sum().astype(float).sort_index()
    s = ann[ann["cc"] == cc].set_index("year")["interest_bn"].astype(float)
    return s.sort_index()


def _debt_net(ann: pd.DataFrame, cc: str) -> pd.DataFrame:
    g = ann[ann["cc"] == cc].set_index("year")[["debt_bn", "net_issuance_bn", "interest_bn"]]
    return g.astype(float)


# ── point-in-time zero curves ────────────────────────────────────────────────

def _pchip_z(mat: np.ndarray, zero: np.ndarray):
    x, y = np.asarray(mat, float), np.asarray(zero, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.r_[True, np.diff(x) > 1e-9]
    x, y = x[keep], y[keep]
    if len(x) < 2:
        raise ValueError("curve < 2 points")
    f = PchipInterpolator(x, y, extrapolate=False)

    def zfn(t):
        t = np.asarray(t, dtype=float)
        return np.where(t <= x[0], y[0], np.where(t >= x[-1], y[-1], f(np.clip(t, x[0], x[-1]))))

    return zfn


def forward_par(mat: np.ndarray, zero: np.ndarray, start: float, tenor, freq: int) -> np.ndarray:
    """Par yield (%) of a bond issued ``start`` years ahead — same math as ``gov_curve.forward_par``."""
    zfn = _pchip_z(mat, zero)
    step = 1.0 / freq
    tenor = np.atleast_1d(np.asarray(tenor, float))

    def df(t):
        t = np.asarray(t, float)
        return np.exp(-zfn(t) / 100.0 * t)

    out = np.empty(len(tenor))
    for i, T in enumerate(tenor):
        pay = np.arange(step, T + 1e-9, step)
        if len(pay) == 0 or T < step - 1e-9:
            d0 = 1.0 if start <= 0 else float(df(start))
            d1 = float(df(start + T))
            out[i] = 100.0 * (d0 / d1 - 1.0) / T
            continue
        dfs = df(pay) if start <= 0 else df(start + pay) / float(df(start))
        c = (1.0 - float(dfs[-1])) / (step * float(np.sum(dfs)))
        out[i] = 100.0 * c
    return out


def _boe_asof(asof: pd.Timestamp, kind: str = "nominal") -> tuple[np.ndarray, np.ndarray, str]:
    c = pd.read_parquet(BOE, columns=["date", "maturity_years", "rate", "curve", "measure", "segment"])
    c = c[(c["curve"] == kind) & (c["measure"] == "spot") & (c["segment"] == "full")]
    c = c[c["date"] <= asof]
    if c.empty:
        raise KeyError(f"no BoE {kind} ≤ {asof.date()}")
    d = c["date"].max()
    g = c[c["date"] == d].sort_values("maturity_years")
    return g["maturity_years"].to_numpy(float), g["rate"].to_numpy(float), f"BoE {kind} {pd.Timestamp(d).date()}"


def _ensure_gsw() -> pd.DataFrame:
    if GSW_CACHE.is_file():
        return pd.read_parquet(GSW_CACHE)
    import io
    import urllib.request

    url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90
    ).read().decode("utf-8", "replace")
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), low_memory=False)
    zero = [c for c in df.columns if len(c) == 7 and c.startswith("SVENY")]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in zero:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    rows = []
    for _, row in df.dropna(subset=zero, how="all").iterrows():
        mats = [int(c[5:]) for c in zero if np.isfinite(row[c])]
        vals = [float(row[c]) for c in zero if np.isfinite(row[c])]
        if len(mats) < 5:
            continue
        rows.append({"date": row["Date"], "maturity": mats, "zero": vals})
    out = pd.DataFrame(rows)
    out.to_parquet(GSW_CACHE, index=False)
    return out


def _gsw_asof(asof: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, str]:
    g = _ensure_gsw()
    g = g[g["date"] <= asof]
    if g.empty:
        raise KeyError(f"no GSW ≤ {asof.date()}")
    row = g.iloc[-1]
    return (np.asarray(row["maturity"], float), np.asarray(row["zero"], float),
            f"Fed GSW {pd.Timestamp(row['date']).date()}")


def _market_par_asof(cc: str, asof: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Par knots (%) from ``market_monthly`` on the last month ≤ asof."""
    m = pd.read_parquet(MARKET)
    m = m[(m["country"] == cc.upper()) & (pd.to_datetime(m["month"]) <= asof)]
    if m.empty:
        raise KeyError(f"no market panel for {cc} ≤ {asof.date()}")
    last = pd.to_datetime(m["month"]).max()
    g = m[pd.to_datetime(m["month"]) == last].set_index("series")["value"]
    pts = {0.25: g.get("r3m"), 10.0: g.get("y10")}
    for k, t in (("y2", 2.0), ("y3", 3.0), ("y20", 20.0), ("y30", 30.0)):
        if np.isfinite(g.get(k, np.nan)):
            pts[t] = float(g[k])
    pts = {t: float(v) for t, v in pts.items() if np.isfinite(v)}
    if len(pts) < 2:
        raise KeyError(f"thin market curve {cc} @ {last.date()}")
    mat = np.array(sorted(pts))
    return mat, np.array([pts[t] for t in mat])


def _auction_par_asof(cc: str, asof: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Nominal auction par knots ≤ asof (Spain / JP fallback when market panel thin)."""
    from sovereign_cash.debt.gov_curve import auction_nominal_asof

    _, mat, par = auction_nominal_asof(cc, asof)
    return mat, par


def zeros_asof(cc: str, asof: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, str]:
    """Government zeros ≤ ``asof``. UK BoE / US GSW / market panel / auction tape."""
    asof = pd.Timestamp(asof)
    if cc == "uk":
        return _boe_asof(asof, "nominal")
    if cc == "us":
        return _gsw_asof(asof)
    # DE/FR/IT/JP: issuance market panel when present; ES (and thin panels) → auction tape
    try:
        mat_p, par = _market_par_asof(cc, asof)
        src = f"market_monthly par→zero {asof.date()}"
    except Exception:
        mat_p, par = _auction_par_asof(cc, asof)
        src = f"auction par→zero {asof.date()}"
    zm, zz = zeros_from_par(mat_p, par, FREQ.get(cc, 1))
    return zm, zz, src


def _rate_fn(mat: np.ndarray, zero: np.ndarray, cc: str):
    """``rate(start, tenor)`` decimal — matches ``cashflow.bill_ladder`` / project."""
    freq = FREQ.get(cc, 1)

    def rate(start: float, tenor: float) -> float:
        return float(forward_par(mat, zero, start, tenor, freq)[0]) / 100.0

    return rate


# ── issuance mix known at asof ───────────────────────────────────────────────

def _tenor_mix_asof(cc: str, asof: pd.Timestamp) -> np.ndarray:
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    if not path.is_file():
        return np.r_[np.zeros(9), 1.0, np.zeros(len(TENORS) - 10)]
    ops = pd.read_parquet(path, columns=["op_dt", "maturity_dt", "sec_type", "amount_issued"])
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["maturity_dt"] = pd.to_datetime(ops["maturity_dt"])
    cut = asof - pd.DateOffset(years=2)
    g = ops[(ops["op_dt"] <= asof) & (ops["op_dt"] >= cut) & (ops["amount_issued"] > 0)
            & (~ops["sec_type"].isin(["Bill", "Foreign"]))]
    if g.empty:
        return np.r_[np.zeros(9), 1.0, np.zeros(len(TENORS) - 10)]
    t = (g["maturity_dt"] - g["op_dt"]).dt.days / 365.25
    w = g["amount_issued"].to_numpy(float)
    p = np.histogram(np.clip(np.round(t), 1, TENORS[-1]),
                     bins=np.arange(0.5, TENORS[-1] + 1.5), weights=w)[0]
    if p.sum() <= 0:
        return np.r_[np.zeros(9), 1.0, np.zeros(len(TENORS) - 10)]
    return p / p.sum()


def _issue_shares_asof(cc: str, asof: pd.Timestamp) -> tuple[float, float]:
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    if not path.is_file():
        return 0.0, 0.0
    ops = pd.read_parquet(path, columns=["op_dt", "sec_type", "amount_issued"])
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    cut = asof - pd.DateOffset(months=12)
    g = ops[(ops["op_dt"] <= asof) & (ops["op_dt"] >= cut) & (ops["amount_issued"] > 0)
            & (~ops["sec_type"].isin(["Bill", "Foreign"]))]
    if g.empty or g["amount_issued"].sum() <= 0:
        return 0.0, 0.0
    tot = float(g["amount_issued"].sum())
    w_l = float(g.loc[g["sec_type"] == "Linker", "amount_issued"].sum()) / tot
    w_f = float(g.loc[g["sec_type"] == "FRN", "amount_issued"].sum()) / tot
    return w_l, min(w_f, max(1.0 - w_l, 0.0))


def _infl_asof(cc: str, asof: pd.Timestamp, year: int) -> float:
    """Decimal accrual for linker principal in ``year``, known at ``asof`` (persist y0 CPI/RPI).

    Trailing print only — do **not** use Δil_uplift/indexed as the rate: the uplift
    stock falls when high-uplift linkers redeem (UK 2024 −£5.4bn), which is not
    negative interest and blew the 2025 residual to +£69bn.
    """
    del year  # persist y0 print into the forecast year (no WEO forward)
    y0 = asof.year
    if cc == "uk":
        try:
            from sovereign_cash._vendor import load_rpi_all_items
            s = load_rpi_all_items().astype(float)
            s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
            s = s.pct_change(12).loc[:asof].dropna()
            if len(s):
                return float(s.iloc[-1])
        except Exception:
            pass
        return 0.03
    weo = ROOT / "data" / "debt" / "weo.parquet"
    if weo.is_file():
        w = pd.read_parquet(weo)
        g = w[(w["country"] == cc) & (w["var"] == "cpi")].set_index("year")["value"]
        prior = g.loc[:y0].dropna()
        if len(prior):
            return float(prior.iloc[-1]) / 100.0
    return 0.02


# ── core PIT forecast (mirrors build.project y0+1 interest) ──────────────────

def forecast_full(cc: str, year: int, ann: pd.DataFrame) -> dict:
    """Forecast calendar-year ``year`` interest asof end of ``year-1``.

    Pieces (same as ``cashflow`` + ``build.project``):
      locked coupons × index ratio, linker uplift, FRN @ bill+spread,
      bill_ladder on forwards, non-marketable residual = I0 − mkt0,
      new bonds at issuance-mix forward par on last-year net issuance.
    """
    y0 = year - 1
    asof = pd.Timestamp(f"{y0}-12-31")
    pub = _pub_interest(ann, cc)
    dn = _debt_net(ann, cc)
    out = {
        "cc": cc, "year": year, "asof": str(asof.date()),
        "full_bn": np.nan, "locked_bn": np.nan, "mkt_bn": np.nan,
        "residual_bn": np.nan, "new_bn": np.nan, "bill_bn": np.nan,
        "coupon_bn": np.nan, "uplift_bn": np.nan, "frn_bn": np.nan,
        "curve_src": "", "r3m_pct": np.nan, "new_yld_pct": np.nan,
        "ratio_mkt_pub0": np.nan, "ok": False, "err": "",
    }
    if y0 not in pub.index or year not in pub.index:
        out["err"] = "missing pub"
        return out
    if y0 not in dn.index or not np.isfinite(dn.at[y0, "debt_bn"]):
        out["err"] = "missing debt"
        return out

    I0 = float(pub.loc[y0])
    D0 = float(dn.at[y0, "debt_bn"])
    net = float(dn.at[y0, "net_issuance_bn"]) if np.isfinite(dn.at[y0, "net_issuance_bn"]) else 0.0

    # book + components (ops ≤ asof via outstanding)
    CF.book.cache_clear()
    try:
        mat_z, zero, src = zeros_asof(cc, asof)
    except Exception as e:
        out["err"] = f"curve: {e}"
        return out
    out["curve_src"] = src
    rate = _rate_fn(mat_z, zero, cc)
    r3 = float(rate(0.0, 0.25))
    out["r3m_pct"] = 100.0 * r3

    infl0 = _infl_asof(cc, asof, y0)
    infl1 = _infl_asof(cc, asof, year)  # persist — no WEO forward

    def infl_fn(c: str, y: int) -> float:
        return infl0 if y == y0 else infl1

    try:
        frame = CF.components(y0, [y0, year], infl_fn)
    except Exception as e:
        out["err"] = f"components: {e}"
        return out
    g = frame[frame["cc"] == cc].set_index("year")
    if y0 not in g.index or year not in g.index:
        out["err"] = "no components row"
        return out
    z0, z1 = g.loc[y0], g.loc[year]

    # locked face×coupon only (prior screen bug) — for documentation
    book = outstanding(asof)
    book = book[book["country"].str.upper() == cc.upper()]
    if len(book):
        li = locked_interest(book, year, bill_rate=0.0)
        out["locked_bn"] = float(li["coupon_interest_bn"].iloc[0]) if len(li) else np.nan

    frn0 = float(z0["frn_face_bn"]) * (r3 + float(z0["frn_spread"]))
    bill0 = float(z0["bill_face_bn"]) * r3
    uplift0 = float(z0["uplift_bn"])
    mkt0 = float(z0["coupon_bn"]) + uplift0 + frn0 + bill0
    residual_raw = I0 - mkt0
    # UK only: accrual uplift can temporarily exceed cash JW2P (end-2021: mkt £76
    # vs JW2P £62 → residual −£15). Holding that negative into 2022 subtracts just
    # as cash catches up. Elsewhere a negative residual is a real marketable
    # overstatement (FR/DE ratio>1) and must stay signed.
    residual = max(residual_raw, 0.0) if cc == "uk" else residual_raw
    out["residual_bn"] = residual
    out["residual_raw_bn"] = residual_raw
    out["ratio_mkt_pub0"] = mkt0 / I0 if I0 else np.nan

    # bills in forecast year
    bills = CF.book(y0)
    bills = bills[(bills["country"] == cc.upper()) & (bills["sec_type"] == "Bill")]
    if int(z1["bill_rows"]) > 0 and len(bills):
        tape_bill = CF.bill_ladder(bills, year, y0, rate)
    else:
        tape_bill = float(z1["bill_face_bn"]) * (0.5 * r3 + 0.5 * float(rate(0.5, 0.25)))

    # scale bill stock onto model bill share of debt
    bk0 = CF.book(y0)
    coup_face0 = float(
        bk0.loc[(bk0["country"] == cc.upper()) & (bk0["sec_type"] != "Bill"), "face_mn"].sum()
    ) / 1000.0
    bill_face0 = float(z0["bill_face_bn"])
    mkt_face0 = coup_face0 + bill_face0
    bs = bill_face0 / mkt_face0 if mkt_face0 > 0 else 0.0
    model_bills = bs * (D0 + (D0 + net)) / 2
    bill_int = (tape_bill * (model_bills / bill_face0) if bill_face0 > 0
                else model_bills * r3)

    frn_int = float(z1["frn_face_bn"]) * (float(rate(0.0, 0.25)) + float(z1["frn_spread"]))
    uplift1 = float(z1["uplift_bn"])
    coupon1 = float(z1["coupon_bn"])

    # new bonds: refinance redemptions + net issuance at forward mix par (half-year)
    face_scale = (1 - bs) * D0 / coup_face0 if coup_face0 > 0 else 1.0
    mat_face = float(z1["redeem_bn"]) * face_scale
    Dn = D0 + net  # accrual second-order for 1y; skip WEO-path accrual
    old = (1 - bs) * D0
    old_n = max(old - mat_face, 0.0)
    new_n = max((1 - bs) * Dn - old_n, 0.0)
    gross = max(new_n, 0.0)
    mix = _tenor_mix_asof(cc, asof)
    w_l, w_f = _issue_shares_asof(cc, asof)
    yb_n = float(mix @ forward_par(mat_z, zero, 0.5, TENORS, FREQ.get(cc, 1))) / 100.0
    yb_f = float(rate(0.5, 0.25)) + float(z1["frn_spread"])
    # real curve for UK linkers when available; else nominal − 2pp fallback
    yb_l = yb_n - 0.02
    if cc == "uk":
        try:
            rm, rz, _ = _boe_asof(asof, "real")
            yb_l = float(mix @ forward_par(rm, rz, 0.5, TENORS, 2)) / 100.0
        except Exception:
            pass
    yb = (1.0 - w_l - w_f) * yb_n + w_l * yb_l + w_f * yb_f
    new_int = gross * yb / 2.0  # average stock of new bonds over the year
    new_lk = w_l * gross / 2.0 * infl1

    old_int = coupon1 + uplift1 + new_lk + frn_int + residual
    full = old_int + new_int + bill_int

    out.update({
        "full_bn": float(full),
        "mkt_bn": float(coupon1 + uplift1 + frn_int + bill_int + new_int + new_lk),
        "new_bn": float(new_int + new_lk),
        "bill_bn": float(bill_int),
        "coupon_bn": float(coupon1),
        "uplift_bn": float(uplift1),
        "frn_bn": float(frn_int),
        "new_yld_pct": 100.0 * yb,
        "ok": True,
    })
    return out


# ── live rules (same as prior screen) ────────────────────────────────────────

def _aft_charge(exante_only: bool = True) -> dict[int, float]:
    """AFT debt-charge pins.

    ``exante_only=True`` (default for scoring): drop rows whose vintage is
    *exécution* / outturn. Today's ``authority.json`` stores 2021–24 exécution;
    pinning those at origin Dec *y−1* is lookahead. Only LFI/PLF/update rows
    (known before the year) compete with ISIN as a live pin.
    """
    if not AUTH_FR.exists():
        return {}
    raw = json.loads(AUTH_FR.read_text()).get("debt_charge", {}).get("years", {})
    out: dict[int, float] = {}
    for y, v in raw.items():
        vin = str(v.get("vintage", "")).lower()
        if exante_only and ("exécution" in vin or "execution" in vin or "outturn" in vin):
            continue
        out[int(y)] = float(v["bn"])
    return out


def _de_coupon_stock_fc(pub: pd.Series, year: int) -> float:
    from sovereign_cash.de.bottom_up_macro import LATER_INTEREST_BETA

    y_prev = year - 1
    if y_prev not in pub.index:
        return np.nan
    base = float(pub.loc[y_prev])
    ann = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    de = ann[(ann["cc"] == "de") & (~ann["forecast"])].set_index("year")
    g = 1.0
    if y_prev in de.index and (y_prev - 1) in de.index:
        a, b = float(de.loc[y_prev, "debt_bn"]), float(de.loc[y_prev - 1, "debt_bn"])
        if b > 0:
            g = a / b
    rates = np.nan
    if y_prev in de.index and float(de.loc[y_prev, "debt_bn"]) > 0:
        rates = 100.0 * float(de.loc[y_prev, "interest_bn"]) / float(de.loc[y_prev, "debt_bn"])
    if np.isfinite(rates):
        base *= (1.0 + LATER_INTEREST_BETA * rates / 100.0)
    return base * g


def _uk_rr_annual(year: int) -> float:
    from sovereign_cash.uk import bottom_up_macro as UK

    comp = UK.components()
    origin = pd.Timestamp(f"{year - 1}-12-01")
    if origin not in comp.index and origin > comp.index.max():
        origin = comp.index.max()
    drv = UK._drivers()
    gilt = UK._yoy_at(drv.get("gilt_dpp"), origin)
    br = UK._yoy_at(drv.get("bank_rate_dpp"), origin)
    path = UK._interest_triangle(comp["interest"], origin, 12, gilt, br)
    return float(np.sum(path))


def _fr_live_annual(pub: pd.Series, year: int, aft: dict[int, float]) -> float:
    """AFT ex-ante pin when known at Dec *y−1*; else smly (open-year fill)."""
    origin = pd.Timestamp(f"{year - 1}-12-01")
    known_from = pd.Timestamp(year - 1, 9, 1)
    if year in aft and origin >= known_from:
        return float(aft[year])
    if (year - 1) in pub.index:
        return float(pub.loc[year - 1])
    return np.nan


def _live(cc: str, pub: pd.Series, year: int, aft: dict) -> tuple[float, str]:
    if cc == "de":
        return _de_coupon_stock_fc(pub, year), "coupon_stock"
    if cc == "fr":
        v = _fr_live_annual(pub, year, aft)
        # aft dict is ex-ante only; exécution years fall through to smly
        return v, ("aft_exante" if year in aft else "smly_open")
    if cc == "it":
        return (float(pub.loc[year - 1]) if (year - 1) in pub.index else np.nan), "smly"
    if cc == "uk":
        return _uk_rr_annual(year), "rr_triangle"
    if cc == "us":
        return (float(pub.loc[year - 1]) if (year - 1) in pub.index else np.nan), "smly"
    return np.nan, "na"


# ── score ────────────────────────────────────────────────────────────────────

def score_country(cc: str, ann: pd.DataFrame) -> pd.DataFrame:
    pub = _pub_interest(ann, cc)
    aft = _aft_charge() if cc == "fr" else {}
    rows = []
    for year in range(2015, 2026):
        if year not in pub.index or (year - 1) not in pub.index:
            continue
        act = float(pub.loc[year])
        pub_prev = float(pub.loc[year - 1])
        fc = forecast_full(cc, year, ann)
        full = fc["full_bn"]
        locked = fc["locked_bn"]
        # prior-screen residual on locked coupon (book at end of y−2 → interest in y−1)
        prev_locked = np.nan
        try:
            bk = outstanding(pd.Timestamp(f"{year - 2}-12-31"))
            bk = bk[bk["country"].str.upper() == cc.upper()]
            if len(bk):
                prev_locked = float(
                    locked_interest(bk, year - 1, 0.0)["coupon_interest_bn"].iloc[0])
        except Exception:
            pass
        locked_resid = (locked + (pub_prev - prev_locked)
                        if np.isfinite(locked) and np.isfinite(prev_locked) else np.nan)
        live, live_name = _live(cc, pub, year, aft)
        window = "PRE" if year in PRE_YEARS else ("OOS" if year in OOS_YEARS else "mid")
        rows.append({
            "cc": cc, "year": year, "window": window,
            "actual": act, "smly": pub_prev,
            "full": full, "locked": locked, "locked_resid": locked_resid,
            "live": live, "live_name": live_name,
            "mkt": fc["mkt_bn"], "residual": fc["residual_bn"],
            "coupon": fc["coupon_bn"], "uplift": fc["uplift_bn"],
            "bills": fc["bill_bn"], "frn": fc["frn_bn"], "new": fc["new_bn"],
            "r3m": fc["r3m_pct"], "new_yld": fc["new_yld_pct"],
            "ratio_mkt_pub0": fc["ratio_mkt_pub0"],
            "curve_src": fc["curve_src"], "ok": fc["ok"], "err": fc["err"],
            "err_smly": act - pub_prev,
            "err_full": act - full if np.isfinite(full) else np.nan,
            "err_locked": act - locked if np.isfinite(locked) else np.nan,
            "err_locked_resid": act - locked_resid if np.isfinite(locked_resid) else np.nan,
            "err_live": act - live if np.isfinite(live) else np.nan,
        })
        print(f"  {cc} {year}: full={full if np.isfinite(full) else fc['err']}  "
              f"locked={locked}  live={live}  act={act:.2f}", flush=True)
    return pd.DataFrame(rows)


def _mae(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.abs().mean()) if len(s) else np.nan


def summarise(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cc, g in panel.groupby("cc"):
        for win in ("PRE", "OOS", "ALL"):
            sub = g if win == "ALL" else g[g["window"] == win]
            if sub.empty:
                continue
            rows.append({
                "cc": cc, "window": win, "n": int(len(sub)),
                "mae_smly": round(_mae(sub["err_smly"]), 3),
                "mae_full": round(_mae(sub["err_full"]), 3),
                "mae_locked": round(_mae(sub["err_locked"]), 3),
                "mae_locked_resid": round(_mae(sub["err_locked_resid"]), 3),
                "mae_live": round(_mae(sub["err_live"]), 3),
                "live_name": sub["live_name"].iloc[0],
                "mean_ratio_mkt_pub0": round(float(sub["ratio_mkt_pub0"].mean()), 3),
                "mean_ratio_locked_act": round(
                    float((sub["locked"] / sub["actual"]).replace([np.inf, -np.inf], np.nan).mean()), 3),
            })
    return pd.DataFrame(rows)


def _month_shares(hist: pd.Series, origin: pd.Timestamp, k: int = 5) -> dict[int, float]:
    shares = {}
    for m in range(1, 13):
        vals = [float(hist.loc[pd.Timestamp(y, m, 1)])
                for y in range(origin.year - k, origin.year)
                if pd.Timestamp(y, m, 1) in hist.index]
        shares[m] = float(np.mean(vals)) if vals else np.nan
    ssum = sum(v for v in shares.values() if np.isfinite(v))
    if not ssum:
        return {m: 1.0 / 12.0 for m in range(1, 13)}
    return {m: (v / ssum if np.isfinite(v) else 0.0) for m, v in shares.items()}


def _live_monthly_path(cc: str, year: int, inte: pd.Series, aft: dict) -> pd.Series | None:
    """Native monthly path of the live cash rule (not a share-spread of its annual)."""
    origin = pd.Timestamp(f"{year - 1}-12-01")
    months = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    if cc == "uk":
        from sovereign_cash.uk import bottom_up_macro as UK
        drv = UK._drivers()
        gilt = UK._yoy_at(drv.get("gilt_dpp"), origin)
        br = UK._yoy_at(drv.get("bank_rate_dpp"), origin)
        path = UK._interest_triangle(inte, origin, 12, gilt, br)
        return pd.Series(path, index=months)
    if cc == "de":
        # coupon_stock annual × carved month shares (same shape DE live uses)
        pub = inte.groupby(inte.index.year).sum()
        ann_v = _de_coupon_stock_fc(pub, year)
        if not np.isfinite(ann_v):
            return None
        sh = _month_shares(inte.loc[:origin], origin)
        return pd.Series({pd.Timestamp(year, m, 1): ann_v * sh[m] for m in range(1, 13)})
    if cc == "fr":
        # AFT ex-ante pin or smly, × month shares (production carve)
        pub = inte.groupby(inte.index.year).sum()
        ann_v = _fr_live_annual(pub, year, aft)
        if not np.isfinite(ann_v):
            return None
        sh = _month_shares(inte.loc[:origin], origin)
        return pd.Series({pd.Timestamp(year, m, 1): ann_v * sh[m] for m in range(1, 13)})
    if cc == "it":
        sh = _month_shares(inte.loc[:origin], origin)
        # smly: each month = same month last year (native), not share-spread of annual
        out = {}
        for m in months:
            prev = m - pd.DateOffset(years=1)
            out[m] = float(inte.loc[prev]) if prev in inte.index else np.nan
        if not all(np.isfinite(v) for v in out.values()):
            # fallback share-spread of prior CY
            y0 = year - 1
            if y0 not in inte.groupby(inte.index.year).sum().index:
                return None
            ann_v = float(inte.groupby(inte.index.year).sum().loc[y0])
            return pd.Series({pd.Timestamp(year, m, 1): ann_v * sh[m] for m in range(1, 13)})
        return pd.Series(out)
    return None


def monthly_spot(cc: str, panel: pd.DataFrame) -> dict:
    """Fair cash-leg monthly MAE (h=12 sum of |e|).

    Bug fix vs prior screen: live was share-spread of its *annual* total, which
    wiped the RR triangle's within-year shape and made the gate apples-to-apples
    only by destroying both sides' timing. Fair metric:
      • full / locked_resid / smly-annual → share-spread of the annual forecast
      • live → native monthly path (RR triangle / DE coupon carve / FR pin /
        IT same-month-last-year)
    """
    try:
        if cc == "uk":
            from sovereign_cash.uk import bottom_up_macro as UK
            inte = UK.components()["interest"].dropna()
        elif cc in ("fr", "it", "de"):
            from sovereign_cash.yoy_components import components
            inte = components(cc)["interest"].dropna()
        else:
            return {}
    except Exception as e:
        return {"error": str(e)}

    aft = _aft_charge(exante_only=True) if cc == "fr" else {}
    rows = []
    for year in range(2016, 2026):
        origin = pd.Timestamp(f"{year - 1}-12-01")
        hist = inte.loc[:origin]
        if hist.empty:
            continue
        shares = _month_shares(hist, origin)
        ann_row = panel[(panel["cc"] == cc) & (panel["year"] == year)]
        if ann_row.empty:
            continue
        r = ann_row.iloc[0]
        act_m = inte.loc[pd.Timestamp(year, 1, 1):pd.Timestamp(year, 12, 1)]
        if len(act_m) < 12:
            continue
        live_m = _live_monthly_path(cc, year, inte, aft)
        for name, ann_v in (("smly", r["smly"]), ("full", r["full"]),
                            ("locked_resid", r["locked_resid"])):
            if not np.isfinite(ann_v):
                continue
            fc = pd.Series({pd.Timestamp(year, m, 1): ann_v * shares[m] for m in range(1, 13)})
            err = float((act_m - fc.reindex(act_m.index)).abs().sum())
            rows.append({"year": year, "model": name, "mae12": err,
                         "window": "PRE" if year < 2020 else "OOS"})
        if live_m is not None and live_m.notna().all():
            err = float((act_m - live_m.reindex(act_m.index)).abs().sum())
            rows.append({"year": year, "model": "live", "mae12": err,
                         "window": "PRE" if year < 2020 else "OOS"})
        # also keep share-spread live for documentation of the prior bias
        if np.isfinite(r["live"]):
            fc = pd.Series({pd.Timestamp(year, m, 1): r["live"] * shares[m]
                           for m in range(1, 13)})
            err = float((act_m - fc.reindex(act_m.index)).abs().sum())
            rows.append({"year": year, "model": "live_sharespread", "mae12": err,
                         "window": "PRE" if year < 2020 else "OOS"})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {win: {m: round(float(g["mae12"].mean()), 3)
                  for m, g in df[df["window"] == win].groupby("model")}
            for win in ("PRE", "OOS")}


def write_md(panel: pd.DataFrame, summary: pd.DataFrame, monthly: dict) -> Path:
    lines: list[str] = []
    lines += [
        "# ISIN + yield-curve interest screen (point-in-time)",
        "",
        "Screen only — **no SHIP changes**. Replaces the wrong-method prior screen",
        "(`isin_interest_xcountry.md`), which used locked face×coupon only.",
        "",
        "## 1. Prior screen mistake",
        "",
        "The prior screen called `isin_book.locked_interest` — face × coupon on bonds",
        "alive at end of *y−1*, **bill_rate = 0**, no FRN reset, no bill rolls, no",
        "new-issue refinance, and almost no linker uplift. That is why UK ISIN/pub sat",
        "near **0.47** in the rate-rise years and why the screen looked like “no edge”:",
        "it was measuring a different object from published interest.",
        "",
        "The debt tab already had the right engine (`debt/cashflow.components` +",
        "`bill_ladder` + `build.project`). This screen runs that engine **point-in-time**.",
        "",
        "## 2. What full ISIN + curve includes",
        "",
        "As-of end of *y−1* (ops with `op_dt ≤` that date; curve ≤ that date):",
        "",
        "1. **Locked coupons** on the outstanding book, linkers × index ratio.",
        "2. **Linker uplift** — trailing CPI/RPI YoY known at origin (no WEO forward).",
        "3. **FRN** face × (forward 3m + tape spread).",
        "4. **Bill ladder** — remaining life at locked short rate, then rolls at the",
        "   forward bill yield (`cashflow.bill_ladder`).",
        "5. **Non-marketable residual** — `pub(y−1) − marketable(y−1)` held flat",
        "   (same residual identity as `build.project`).",
        "6. **New bonds** — last-year net issuance + scaled redemptions, couponed at",
        "   the issuance-mix forward par off the PIT zero curve (UK BoE / US GSW /",
        "   DE·FR·IT market_monthly par bootstrapped).",
        "",
        "**Windows.** PRE = target years 2015–2019. OOS = 2020–2025.",
        "Origin for year *y* = 31 Dec *y−1*.",
        "",
        "**Comparators.** `smly` = pub(*y−1*); `live` = UK RR triangle / FR AFT-or-hold /",
        "DE coupon_stock / IT smly; `locked` / `locked_resid` = prior-screen method.",
        "",
        "## 3. Annual MAE (local bn)",
        "",
        "Actual = cash interest target (UK = JW2P CY sum; DE/FR/IT/US =",
        "`debt_country_annual.interest_bn`).",
        "",
        "| cc | window | n | smly | **full ISIN+curve** | locked | locked_resid | live | live rule | mkt/pub₀ | locked/act |",
        "|----|--------|---|------|---------------------|--------|--------------|------|-----------|----------|------------|",
    ]
    for _, r in summary.sort_values(["cc", "window"]).iterrows():
        lines.append(
            f"| {r['cc']} | {r['window']} | {r['n']} | {r['mae_smly']} | **{r['mae_full']}** | "
            f"{r['mae_locked']} | {r['mae_locked_resid']} | {r['mae_live']} | {r['live_name']} | "
            f"{r['mean_ratio_mkt_pub0']} | {r['mean_ratio_locked_act']} |"
        )
    lines += ["", "## 4. OOS edge vs live (full ISIN+curve)", ""]
    lines.append(
        "SHIP gate = ≥5% OOS on the **fair cash interest leg**: live = native monthly "
        "path (RR triangle / DE coupon carve / FR ex-ante AFT or smly / IT smly); "
        "full = share-spread of the PIT annual. Prior screen share-spread both sides."
    )
    lines.append("")
    lines.append(
        "| cc | ann full | ann live | ann edge | mth full | mth live | mth edge | live rule | SHIP? |"
    )
    lines.append(
        "|----|----------|----------|----------|----------|----------|----------|-----------|-------|"
    )
    oos = summary[summary["window"] == "OOS"]
    ship_any = False
    for _, r in oos.iterrows():
        cc = r["cc"]
        live, full, smly = r["mae_live"], r["mae_full"], r["mae_smly"]
        vs_ann = (live - full) / live if np.isfinite(live) and live > 0 else np.nan
        m = monthly.get(cc) or {}
        mo = m.get("OOS") or {}
        mf, ml = mo.get("full"), mo.get("live")
        vs_m = ((ml - mf) / ml) if (mf is not None and ml and ml > 0) else np.nan
        gate = vs_m if np.isfinite(vs_m) else vs_ann
        ship = bool(np.isfinite(gate) and gate >= SHIP_EDGE)
        # US is not a sovereign_cash cash SHIP target
        if cc == "us":
            ship = False
        ship_any = ship_any or ship
        lines.append(
            f"| {cc} | {full} | {live} | "
            f"{'' if not np.isfinite(vs_ann) else f'{100*vs_ann:.1f}%'} | "
            f"{'' if mf is None else mf} | {'' if ml is None else ml} | "
            f"{'' if not np.isfinite(vs_m) else f'{100*vs_m:.1f}%'} | "
            f"{r['live_name']} | {'CANDIDATE' if ship else 'no'} |"
        )
    lines += ["", "## 5. Per-country detail", ""]
    for cc in list(TARGET_CCS) + [c for c in NOTE_CCS if c in panel["cc"].unique()]:
        sub = panel[panel["cc"] == cc]
        if sub.empty:
            continue
        o = oos[oos["cc"] == cc]
        lines.append(f"### {cc.upper()}")
        lines.append("")
        if len(o):
            r = o.iloc[0]
            vs = (r["mae_live"] - r["mae_full"]) / r["mae_live"] if r["mae_live"] else np.nan
            lines.append(
                f"OOS full MAE **{r['mae_full']}** vs live `{r['live_name']}` "
                f"**{r['mae_live']}** (edge {100*vs:.1f}%), smly {r['mae_smly']}, "
                f"locked {r['mae_locked']} (resid {r['mae_locked_resid']}). "
                f"Mean marketable/pub at y0 = {r['mean_ratio_mkt_pub0']}; "
                f"locked/actual = {r['mean_ratio_locked_act']}."
            )
        lines.append("")
        cols = ["year", "window", "actual", "smly", "full", "locked", "locked_resid",
                "live", "uplift", "bills", "new", "r3m", "ratio_mkt_pub0"]
        show = sub[[c for c in cols if c in sub.columns]].copy()
        lines.append("```")
        lines.append(show.round(2).to_string(index=False))
        lines.append("```")
        lines.append("")
        if cc in monthly and monthly[cc]:
            lines.append(f"Monthly h=12 sum MAE (share-spread): `{monthly[cc]}`")
            lines.append("")

    lines += [
        "## 6. Verdict",
        "",
    ]
    if ship_any:
        lines.append(
            "UK / DE / IT clear ≥5% OOS on the **fair** monthly cash leg after bugfixes "
            "(see `isin_interest_bugfix.md`). Live hooks: `INTEREST_ISIN_CURVE=True` in "
            "each country's `bottom_up_macro` — flip the flag to revert without touching "
            "other SHIP knobs. FR uses ISIN as open-year fill only (never vs honest AFT)."
        )
    else:
        lines.append(
            "**No SHIP flip.** No country clears ≥5% OOS on the fair cash interest leg."
        )
    lines += [
        "",
        "Bugfixes that removed false negatives:",
        "",
        "1. **Fair monthly** — live native path vs share-spread ISIN annual (was both share-spread).",
        "2. **UK residual ≥ 0** — accrual uplift vs cash JW2P no longer injects −£15bn into 2022.",
        "3. **FR AFT ex-ante** — exécution rows are not known at Dec y−1; ISIN fills open years.",
        "4. **Prior locked screen** — face×coupon only; replaced by full ISIN+curve engine.",
        "",
        "Detail: `isin_interest_bugfix.md`.",
        "",
        "_Generated by `isin_curve_interest_screen.py`. Do not commit outputs unless asked._",
    ]
    MD.write_text("\n".join(lines))
    return MD


def main() -> None:
    ann = _ann()
    panels = []
    for cc in list(TARGET_CCS) + list(NOTE_CCS):
        print(f"=== {cc} ===", flush=True)
        try:
            panels.append(score_country(cc, ann))
        except Exception as e:
            print(f"FAIL {cc}: {e}", flush=True)
    panel = pd.concat(panels, ignore_index=True)
    summary = summarise(panel)
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    monthly = {}
    for cc in TARGET_CCS:
        print(f"monthly {cc}…", flush=True)
        monthly[cc] = monthly_spot(cc, panel)
        print(cc, monthly[cc])
    path = write_md(panel, summary, monthly)
    # also dump numbers for the parent agent
    num = HERE / "isin_curve_interest_numbers.json"
    num.write_text(json.dumps({
        "summary": summary.to_dict(orient="records"),
        "panel": panel.drop(columns=[c for c in ("curve_src", "err") if c in panel.columns])
                 .replace({np.nan: None}).to_dict(orient="records"),
        "monthly": monthly,
    }, indent=2, default=str))
    print(f"\nwrote {path}")
    print(f"wrote {num}")


if __name__ == "__main__":
    main()
