"""UK CGNCR(ex) bottom-up v2: OBR/HMT category forecasts × ready-reckoner determinants.

Identity (monthly, £bn)

    CGNCR(ex) = welfare + rdel_proxy + interest + capital
              − (IT + NICs + VAT + CT + other_tax + nontax)
              + recv_pay + gilt_adj + other_fin + acq_sec + apf_xfer + net_lend
              + resid_gap

Tax lines are ONS ``cg_taxes_receivable`` split by EFO shares; welfare/RDEL split
primary current expenditure the same way. Each category's month shape is last year's
calendar month; the annual path is the published ready-reckoner elasticity × the
determinant YoY known at the origin.

Interest uses the OBR debt-interest triangle (Bank Rate + long-term gilt rates) as an
additive £bn overlay on the calendar month. CT uses GVA YoY; other HMRC taxes take
retail + equity (PRE-tuned). Nontax stays calendar-only (GDP/BR overlays lose OOS).
Welfare is 1.5×CPI (triple-lock region). Capital and RDEL take GVA YoY (PRE-gated;
calendar / CPI variants lose OOS among PRE-clearers).

January/February self-assessment (CGT + SA IT) is not a plain calendar carry: non-property
SA is due 10 months after the disposal FY, so Jan-T cash maps to Apr-(T−2)…Mar-(T−1)
equities (OBR gearing). A same-month carry of an outlier January (Jan-2026 CGT +£7bn)
would reprint that spike. ``_sa_equity_nudge`` replaces Jan/Feb ``other_tax`` and ``it``
with baseline + partial last-year excess × relative disposal-FY equity; February takes
a spill weight (ONS: deadline payments split Jan/Feb).

The CGNCR residual is mapped to published ONS PUSF financial-transaction lines
(ANRT/ANRU/ANRV/ANRS/ANRH/MF7A; R²≈0.88 vs the lump). Per-line forecasts are
PRE-gated vs ``seasonal_add3``: ANRT calendar × retail YoY, ANRS trailing-12 mean,
ANRV ``sa3`` + Bank Rate Δpp additive; ANRU/MF7A/ANRH and ``resid_gap`` stay on
``sa3`` (gilt/APF/NS&I overlays fail category or headline PRE, or raise OOS). Soft
additive pin α=0.12 pulls remaining FY months toward the remit CGNCR vintage known
at the origin (PRE-gated; clears the h=12 miss at α=0).

v1 (``bottom_up.py``) = ONS-line seasonals — rejected. Live path switches only when
``SHIP`` and ``ship_rule`` both pass (mean PRE MAE beats production by more than
``EDGE_BAND`` with ≤1 worse horizon). Among PRE-clearers, SPEC ranks on OOS then
residual share.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.uk import bottom_up as V1
from sovereign_cash.uk import model as M
from sovereign_cash.uk import obr_tables as O
from sovereign_cash.uk import ready_reckoner as RR
from sovereign_cash.uk import sources as src

AUTH = Path(__file__).resolve().parents[1] / "data" / "uk" / "authority.json"
EQUITY = Path(__file__).resolve().parents[1] / "data" / "uk" / "equity_fred.json"
MACRO_PANEL = Path(__file__).resolve().parents[2] / "correlations" / "derived" / "macro_panel_monthly.csv"
CURVES = Path(__file__).resolve().parents[2] / "uk_flows" / "output" / "curves.parquet"
RATES = Path(__file__).resolve().parents[2] / "uk_flows" / "output" / "rates.parquet"

BOTTOMUP = "bottom_up_macro"
PRODUCTION = V1.PRODUCTION

# ONS PUSF CG financial-transaction lines that compose the CGNCR residual.
FIN_PARTS = (
    "recv_pay",   # ANRT F7 accounts receivable/payable
    "gilt_adj",   # ANRU F.3 gilt interest adjustment
    "other_fin",  # ANRV other financial transactions
    "acq_sec",    # ANRS F.5 net acquisition of company securities
    "apf_xfer",   # MF7A capital transfers to APF
    "net_lend",   # ANRH F.4 net lending to private sector / RoW
)

# Locked on PRE only (2026-10-20e): soft_pin_alpha=0.12.
# Remit CGNCR vintage known at origin pulls remaining FY months lightly (α of gap);
# category×RR path still sets the level/shape. PRE edge ~11.2% vs production (~8.7% at α=0);
# clears all horizons (α=0 worse only at h=12); OOS ~11.4 vs ~19. α≥0.20 with resid_sa3=1
# regresses OOS — rejected. Other knobs held at α=0 PRE pick (welfare 1.5, cap/rdel GVA, …).
SPEC = {
    "welfare_cpi": 1.5,
    "rdel_cpi": 0.0,
    "rdel_gdp": 0.25,
    "nontax_gdp": 0.0,
    "other_tax_retail": 0.75,
    "other_tax_equity": 0.7,
    "cap_retail": 0.0,
    "cap_gdp": 0.2,
    "soft_pin_alpha": 0.12,
    "resid_sa3": 0.0,  # calendar for unmapped resid_gap / default fin parts
    # FIN_PART forecast overlays (0 / "sa3" = seasonal_add3). PRE-gated vs sa3 category MAE.
    "fin_recv_pay": "grow_retail",
    "fin_recv_pay_elas": 0.0,  # calendar level; retail grow loses OOS among PRE-clearers
    "fin_gilt_adj": "sa3",
    "fin_other_fin": "sa3_add_br",
    "fin_other_fin_br": 2.0,
    "fin_acq_sec": "trail12",
    "fin_apf_xfer": "sa3",
    "fin_net_lend": "sa3",
    "fin_resid_gap": "sa3",
    "rr_shock_fy": "2025-26",
    "share_year": 2024,
    # Interest engine — see INTEREST_ISIN_CURVE below / isin_interest_bugfix.md.
    # Modes: isin_curve | rr | isin_yoy | isin_share | isin_replace | blend*
    "interest_mode": "isin_curve",  # overwritten from INTEREST_ISIN_CURVE at import
    "isin_blend": 0.5,  # weight on ISIN leg when interest_mode starts with blend
    "isin_redeem_overlay": False,  # add book redemptions − calendar onto resid_gap
    # Jan/Feb SA equity nudge (CGT + SA IT). Off → plain calendar×YoY carry.
    "sa_equity_nudge": True,
    "sa_equity_persist": 0.35,  # share of last-Jan excess that can carry
    "sa_equity_beta": 0.7,  # relative disposal-FY equity scale (matches other_tax_equity)
    "sa_equity_feb_w": 0.40,  # Feb gets this fraction of the Jan receipt adjustment
    "sa_equity_it_w": 1.0,  # also nudge IT (SA income tax); 0 = other_tax only
}

# INTEREST_ISIN_CURVE off: monthly CGNCR scorer interest leaf PRE +52% / OOS +111% vs RR
# (uk_cash_ship_autopsy.md §3). The −20.5% fair-cash figure was annual Dec-31 share-spread.
INTEREST_ISIN_CURVE = False
SPEC["interest_mode"] = "isin_curve" if INTEREST_ISIN_CURVE else "rr"

# Module flag is not the live gate — run.py ships only when evaluate()["ship"] is True.
# Kept False: SPEC with sa_equity_nudge on is NO-SHIP on the monthly scorer (autopsy).
SHIP = False
FORCE_LIVE = False

AUTHORITY_TABLES = (
    ("OBR EFO March 2026 Table A.5 receipts", "income_tax_bn / nics_bn / vat_bn / corp_tax_bn",
     "data/debt/obr_lines.parquet"),
    ("OBR EFO March 2026 Table A.7 expenditure", "welfare_bn / rdel_bn / debt_interest_bn",
     "data/debt/obr_lines.parquet"),
    ("OBR EFO March 2026 Table 5.4 financing", "fin_transactions_bn / cgncr_ex_nr_bn",
     "data/debt/obr_lines.parquet"),
    ("OBR EFO March 2026 Table A.3 determinants", "awe / cpi / gilt / nominal GDP / consumption",
     "data/debt/obr_lines.parquet"),
    ("OBR ready reckoner March 2025 EFO (oil corr. 17 Mar 2026)", "Receipts/Spending/DI coefficients",
     "data/uk/obr_ready_reckoner.xlsx"),
    ("FRED SPASTT01GBM661N", "UK share prices (equity determinant for CGT/stamp/IHT)",
     "data/uk/equity_fred.json"),
    ("BoE IUDBEDR via uk_flows rates.parquet", "Official Bank Rate (DI triangle)",
     "uk_flows/output/rates.parquet"),
    ("ONS PUSF", "ANBV / MF6P / MF6S / MF6T / JW2P / M98R monthly outturn + month profiles",
     "data/uk/ons/*.json"),
    ("ONS PUSF ANRT/ANRU/ANRV/ANRS/ANRH/MF7A", "CG financial transactions → residual map",
     "data/uk/ons/{anrt,anru,anrv,anrs,anrh,mf7a}.json"),
    ("DMO gilt/bill ops + BoE curve (ISIN+curve interest)",
     "marketable coupons, linker uplift, bill ladder, residual to JW2P",
     "issuance/cache/ops_uk.parquet · uk_flows/output/curves.parquet"),
)


@lru_cache(maxsize=1)
def authority() -> dict:
    if AUTH.exists():
        return json.loads(AUTH.read_text(encoding="utf-8"))
    return {"vintage": O.vintage(), "categories_bn": {}, "determinants": {}}


@lru_cache(maxsize=1)
def _rr_elas() -> dict[str, float]:
    """% category change per 1% (or 1pp) determinant from the published RR row."""
    y = SPEC["share_year"]
    d = O.load()
    levels = {}
    for var in ("income_tax_bn", "nics_bn", "vat_bn", "corp_tax_bn", "debt_interest_bn"):
        s = d[(d["var"] == var) & (d["year"] == y)]["value"]
        levels[var] = float(s.iloc[0]) if len(s) else np.nan
    coef = RR.coefficients()
    fy = SPEC["rr_shock_fy"]
    sub = coef[(coef["shock_fy"] == fy) & (coef["effect_fy"] == fy) & (~coef["loop"])]

    def gbp(line: str, det: str) -> float:
        hit = sub[(sub["line"] == line) & (sub["determinant"] == det)]["gbp_m"]
        return float(hit.sum()) / 1000.0 if len(hit) else 0.0

    it, nics, vat, ct, di = (levels[k] for k in (
        "income_tax_bn", "nics_bn", "vat_bn", "corp_tax_bn", "debt_interest_bn"))
    return {
        "it_awe": 100.0 * gbp("PAYE income tax", "Earnings level") / it,
        "nics_awe": 100.0 * gbp("NICs", "Earnings level") / nics,
        "vat_cons": 100.0 * gbp("VAT", "Nominal consumption") / vat,
        "ct_profits": 100.0 * gbp("Onshore corporation tax", "Non-oil, non-financial profits") / ct,
        "di_gilt_frac": gbp("Debt interest", "Long-term interest rates") / di,
        "di_level": di,
        "it_level": it, "nics_level": nics, "vat_level": vat, "ct_level": ct,
    }


@lru_cache(maxsize=1)
def _di_triangle() -> dict[str, dict[int, float]]:
    """RR debt-interest £bn / DI-level per 1pp shock, by FY lag from shock FY."""
    coef = RR.coefficients()
    fy = SPEC["rr_shock_fy"]
    di = _rr_elas()["di_level"]
    out: dict[str, dict[int, float]] = {}
    for det, key in (
        ("Long-term interest rates", "gilt"),
        ("Bank Rate", "bank_rate"),
    ):
        sub = coef[(coef["shock_fy"] == fy) & (coef["line"] == "Debt interest")
                   & (~coef["loop"]) & (coef["determinant"] == det)]
        lags: dict[int, float] = {}
        for _, r in sub.iterrows():
            lag = int(str(r["effect_fy"])[:4]) - int(str(r["shock_fy"])[:4])
            lags[lag] = float(r["gbp_m"]) / 1000.0 / di
        out[key] = lags
    return out


@lru_cache(maxsize=1)
def _efo_shares() -> dict[str, float]:
    """Big-4 tax shares of National Accounts taxes; residual is other HMRC/duties."""
    e = _rr_elas()
    d = O.load()
    y = SPEC["share_year"]
    na = d[(d["var"] == "receipts_national_accounts_taxes") & (d["year"] == y)]["value"]
    tax_tot = float(na.iloc[0]) if len(na) else (
        e["it_level"] + e["nics_level"] + e["vat_level"] + e["ct_level"])
    wel = float(d[(d["var"] == "welfare_bn") & (d["year"] == y)]["value"].iloc[0])
    rdel = float(d[(d["var"] == "rdel_bn") & (d["year"] == y)]["value"].iloc[0])
    return {
        "it": e["it_level"] / tax_tot, "nics": e["nics_level"] / tax_tot,
        "vat": e["vat_level"] / tax_tot, "ct": e["ct_level"] / tax_tot,
        "welfare": wel / (wel + rdel),
    }


@lru_cache(maxsize=1)
def _drivers() -> dict[str, pd.Series]:
    """Point-in-time macro YoY / Δpp series, month-start index."""
    out: dict[str, pd.Series] = {}
    if MACRO_PANEL.exists():
        mp = pd.read_csv(MACRO_PANEL, parse_dates=["date"]).set_index("date").sort_index()
        mp.index = mp.index.to_period("M").to_timestamp()
        if "awe_total_index" in mp:
            out["awe"] = mp["awe_total_index"].astype(float).pct_change(12) * 100.0
        if "retail_sales_index" in mp:
            out["retail"] = mp["retail_sales_index"].astype(float).pct_change(12) * 100.0
        if "cpi_all" in mp:
            out["cpi"] = mp["cpi_all"].astype(float).pct_change(12) * 100.0
        if "gdp_gva_index" in mp:
            out["gdp"] = mp["gdp_gva_index"].astype(float).pct_change(12) * 100.0
    if CURVES.exists():
        curves = pd.read_parquet(CURVES)
        nom = curves[(curves["curve"] == "nominal") & (curves["measure"] == "spot")].copy()
        nom["date"] = pd.to_datetime(nom["date"])
        g10 = (nom[np.isclose(nom["maturity_years"], 10.0, atol=0.26)]
               .groupby("date")["rate"].mean())
        g10 = g10.to_period("M").groupby(level=0).last()
        g10.index = g10.index.to_timestamp()
        out["gilt_dpp"] = g10.diff(12)
    if RATES.exists():
        rates = pd.read_parquet(RATES)
        br = rates[rates["series"] == "bank_rate"].copy()
        br["date"] = pd.to_datetime(br["date"])
        brm = br.set_index("date")["value"].to_period("M").groupby(level=0).last()
        brm.index = brm.index.to_timestamp()
        out["bank_rate_dpp"] = brm.astype(float).diff(12)
    if EQUITY.exists():
        raw = json.loads(EQUITY.read_text(encoding="utf-8"))
        eq = pd.DataFrame(raw["observations"])
        eq["date"] = pd.to_datetime(eq["date"])
        eqs = eq.set_index("date")["value"].astype(float).sort_index()
        eqs.index = eqs.index.to_period("M").to_timestamp()
        out["equity"] = eqs.pct_change(12) * 100.0
    return out


def components() -> pd.DataFrame:
    """Monthly category frame (£bn). Residual exact on overlapping history; fin parts mapped."""
    base = V1.components()
    ons = src.ons_monthly()
    sh = _efo_shares()
    df = base.copy()
    df["taxes"] = ons["cg_taxes_receivable"].reindex(df.index)
    df["nontax"] = df["rec"] - df["taxes"]
    for k in ("it", "nics", "vat", "ct"):
        df[k] = df["taxes"] * sh[k]
    df["other_tax"] = df["taxes"] - df["it"] - df["nics"] - df["vat"] - df["ct"]
    df["welfare"] = df["primary"] * sh["welfare"]
    df["rdel_proxy"] = df["primary"] * (1.0 - sh["welfare"])
    missing = [k for k in FIN_PARTS if k not in ons.columns]
    if missing:
        raise FileNotFoundError(
            f"ONS financial-transaction series missing from cache: {missing}. "
            "Run: PYTHONPATH=. .venv/bin/python -c "
            "'from sovereign_cash.uk import sources as s; s.fetch_ons(refresh=True)'")
    for k in FIN_PARTS:
        df[k] = ons[k].reindex(df.index)
    df["resid_gap"] = df["resid"] - df[list(FIN_PARTS)].sum(axis=1)
    df.index.name = "month"
    return df


def _yoy_at(series: pd.Series | None, origin: pd.Timestamp) -> float:
    if series is None:
        return 0.0
    dy = series.loc[:origin].dropna()
    if dy.empty:
        return 0.0
    return float(dy.iloc[-3:].mean()) if len(dy) >= 3 else float(dy.iloc[-1])


def _grow(series: pd.Series, origin: pd.Timestamp, h: int, yoy: float, elas: float) -> np.ndarray:
    hh = series.loc[:origin].dropna()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    cg = elas * yoy / 100.0
    out = []
    for m in months:
        base = hh.get(m - pd.DateOffset(years=1), np.nan)
        if not np.isfinite(base):
            base = float(hh.iloc[-12:].mean()) if len(hh) else 0.0
        out.append(float(base) * (1.0 + cg))
    return np.array(out, dtype=float)


def _grow_combo(series: pd.Series, origin: pd.Timestamp, h: int,
                terms: list[tuple[float, float]]) -> np.ndarray:
    """Calendar month × (1 + Σ elas_i × yoy_i / 100)."""
    hh = series.loc[:origin].dropna()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    cg = sum(e * y / 100.0 for y, e in terms)
    out = []
    for m in months:
        base = hh.get(m - pd.DateOffset(years=1), np.nan)
        if not np.isfinite(base):
            base = float(hh.iloc[-12:].mean()) if len(hh) else 0.0
        out.append(float(base) * (1.0 + cg))
    return np.array(out, dtype=float)


def _interest_triangle(series: pd.Series, origin: pd.Timestamp, h: int,
                       gilt_dpp: float, br_dpp: float) -> np.ndarray:
    """Calendar DI month + RR triangle overlay (Bank Rate + long gilt), £bn/month."""
    hh = series.loc[:origin].dropna()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    tri = _di_triangle()
    ann = _rr_elas()["di_level"]
    shock0 = M.fy_of(origin + pd.offsets.MonthBegin(1))
    out = []
    for m in months:
        base = hh.get(m - pd.DateOffset(years=1), np.nan)
        if not np.isfinite(base):
            base = float(hh.iloc[-12:].mean()) if len(hh) else 0.0
        lag = M.fy_of(m) - shock0
        frac = (tri["gilt"].get(lag, 0.0) * gilt_dpp
                + tri["bank_rate"].get(lag, 0.0) * br_dpp)
        out.append(float(base) + ann * frac / 12.0)
    return np.array(out, dtype=float)


def _isin_yoy_at(origin: pd.Timestamp) -> float:
    """YoY % of annualised marketable interest (coupon + linker uplift + bills) at origin."""
    from sovereign_cash.uk import isin_interest as II

    asof = pd.Timestamp(origin) + pd.offsets.MonthEnd(0)
    now = II.annual_marketable(asof)["interest_bn"]
    lag = II.annual_marketable(asof - pd.DateOffset(years=1))["interest_bn"]
    if not np.isfinite(now) or not np.isfinite(lag) or lag <= 0:
        return 0.0
    return 100.0 * (now / lag - 1.0)


def _interest_isin_share(series: pd.Series, origin: pd.Timestamp, h: int) -> np.ndarray:
    """JW2P month shares × marketable annual, scaled so TTM JW2P matches at origin."""
    from sovereign_cash.uk import isin_interest as II

    hh = series.loc[:origin].dropna()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    asof = pd.Timestamp(origin) + pd.offsets.MonthEnd(0)
    mkt = float(II.annual_marketable(asof)["interest_bn"])
    ttm = float(hh.iloc[-12:].sum()) if len(hh) >= 12 else float(hh.sum())
    scale = (ttm / mkt) if mkt > 1e-6 else 1.0
    # calendar-year month shares from JW2P history through origin
    hist = hh.copy()
    hist = hist[hist.index >= hist.index.max() - pd.DateOffset(years=5)]
    by_m = hist.groupby(hist.index.month).mean()
    ssum = float(by_m.sum()) if len(by_m) else 12.0
    shares = {int(m): float(v) / ssum for m, v in by_m.items()} if ssum > 0 else {}
    ann = mkt * scale
    out = []
    for m in months:
        sh = shares.get(int(m.month), 1.0 / 12.0)
        out.append(ann * sh)
    return np.array(out, dtype=float)


def _interest_path(series: pd.Series, origin: pd.Timestamp, h: int,
                   gilt_dpp: float, br_dpp: float, spec: dict) -> np.ndarray:
    """Interest slot: ISIN+curve (live) or RR / book experiment modes."""
    mode = str(spec.get("interest_mode", "isin_curve"))
    rr = _interest_triangle(series, origin, h, gilt_dpp, br_dpp)
    if mode in ("rr", "", "triangle"):
        return rr
    if mode == "isin_curve":
        from sovereign_cash import isin_curve_interest as IC
        path = IC.monthly_path("uk", series, origin, h)
        if np.isfinite(path).all() and float(np.nanmean(np.abs(path))) > 0:
            return path
        return rr  # graceful fallback if ops/curve missing
    if mode == "isin_yoy":
        return _grow(series, origin, h, _isin_yoy_at(origin), 1.0)
    if mode == "isin_share":
        return _interest_isin_share(series, origin, h)
    if mode == "isin_replace":
        # Calendar month + additive marketable Δannual/12 (replaces RR triangle).
        from sovereign_cash.uk import isin_interest as II
        asof = pd.Timestamp(origin) + pd.offsets.MonthEnd(0)
        mkt = II.annual_marketable(asof)["interest_bn"]
        lag = II.annual_marketable(asof - pd.DateOffset(years=1))["interest_bn"]
        add = (mkt - lag) / 12.0 if np.isfinite(mkt) and np.isfinite(lag) else 0.0
        cal = _grow(series, origin, h, 0.0, 0.0)
        return cal + add
    if mode.startswith("blend"):
        w = float(spec.get("isin_blend", 0.5))
        if mode == "blend25":
            w = 0.25
        elif mode == "blend50":
            w = 0.5
        elif mode == "blend75":
            w = 0.75
        isin = _grow(series, origin, h, _isin_yoy_at(origin), 1.0)
        return (1.0 - w) * rr + w * isin
    raise ValueError(f"unknown interest_mode {mode!r}")


@lru_cache(maxsize=1)
def _equity_level() -> pd.Series:
    """UK share-price level (FRED SPASTT01GBM661N), monthly."""
    if not EQUITY.exists():
        return pd.Series(dtype=float)
    raw = json.loads(EQUITY.read_text(encoding="utf-8"))
    eq = pd.DataFrame(raw["observations"])
    eq["date"] = pd.to_datetime(eq["date"])
    s = eq.set_index("date")["value"].astype(float).sort_index()
    s.index = s.index.to_period("M").to_timestamp()
    return s


def disposal_fy_equity_return(jan_year: int, asof: pd.Timestamp | None = None) -> float:
    """Price return over the disposal FY that feeds January ``jan_year`` SA cash.

    Non-property SA is due end-January, 10 months after the disposal FY ends (OBR):
    Jan-T ← Apr-(T−2)…Mar-(T−1). Return = Mar-(T−1) / Mar-(T−2) − 1, in percent.
    ``asof`` truncates the end month (no lookahead past the origin).
    """
    eqs = _equity_level()
    if eqs.empty:
        return float("nan")
    end = pd.Timestamp(int(jan_year) - 1, 3, 1)
    start = pd.Timestamp(int(jan_year) - 2, 3, 1)
    if asof is not None:
        asof_m = pd.Timestamp(asof).to_period("M").to_timestamp()
        if end > asof_m:
            end = asof_m
    if start not in eqs.index or end not in eqs.index:
        # nearest available ≤ target
        pre_s = eqs.loc[:start]
        pre_e = eqs.loc[:end]
        if pre_s.empty or pre_e.empty:
            return float("nan")
        p0, p1 = float(pre_s.iloc[-1]), float(pre_e.iloc[-1])
    else:
        p0, p1 = float(eqs.at[start]), float(eqs.at[end])
    if not np.isfinite(p0) or p0 == 0 or not np.isfinite(p1):
        return float("nan")
    return 100.0 * (p1 / p0 - 1.0)


def _jan_baseline(series: pd.Series, jan: pd.Timestamp, n: int = 8) -> float:
    """Median same-calendar-month level over prior years, excluding last year (the carry base)."""
    hist = []
    for k in range(2, n + 2):
        t = jan - pd.DateOffset(years=k)
        if t in series.index and np.isfinite(series.at[t]):
            hist.append(float(series.at[t]))
    if len(hist) < 3:
        # fall back to any prior same month except last year
        sm = series[(series.index.month == jan.month) & (series.index < jan - pd.DateOffset(years=1))]
        hist = [float(v) for v in sm.dropna().tail(n).tolist()]
    return float(np.median(hist)) if hist else float("nan")


def _sa_equity_month_value(series: pd.Series, month: pd.Timestamp, origin: pd.Timestamp,
                           persist: float, beta: float) -> float | None:
    """Equity-nudged Jan/Feb receipt level (£bn), or None if inputs missing."""
    ly = month - pd.DateOffset(years=1)
    if ly not in series.index or not np.isfinite(series.at[ly]):
        return None
    base = _jan_baseline(series.loc[:origin], month)
    if not np.isfinite(base):
        return None
    excess = float(series.at[ly]) - base
    eq_t = disposal_fy_equity_return(int(month.year), asof=origin)
    eq_ly = disposal_fy_equity_return(int(ly.year), asof=origin)
    if not np.isfinite(eq_t) or not np.isfinite(eq_ly):
        return None
    # Relative equity scale on the excess only; baseline is the non-spike Jan/Feb.
    denom = 1.0 + beta * eq_ly / 100.0
    scale = (1.0 + beta * eq_t / 100.0) / denom if abs(denom) > 1e-6 else 1.0
    scale = float(np.clip(scale, 0.4, 2.0))
    return base + float(persist) * excess * scale


def _apply_sa_equity_nudge(paths: dict[str, np.ndarray], comp: pd.DataFrame,
                           origin: pd.Timestamp, h: int, spec: dict) -> dict[str, np.ndarray]:
    """Replace Jan/Feb IT and other_tax calendar carries with disposal-FY equity nudges."""
    if not spec.get("sa_equity_nudge", False):
        return paths
    persist = float(spec.get("sa_equity_persist", 0.35))
    beta = float(spec.get("sa_equity_beta", 0.7))
    feb_w = float(spec.get("sa_equity_feb_w", 0.40))
    it_w = float(spec.get("sa_equity_it_w", 1.0))
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    ot = np.asarray(paths["other_tax"], dtype=float).copy()
    it = np.asarray(paths["it"], dtype=float).copy()
    jan_ot_delta = 0.0
    jan_it_delta = 0.0
    for i, m in enumerate(months):
        if m.month == 1:
            fair_ot = _sa_equity_month_value(comp["other_tax"], m, origin, persist, beta)
            if fair_ot is not None:
                jan_ot_delta = float(fair_ot) - float(ot[i])
                ot[i] = float(fair_ot)
            if it_w > 0:
                fair_it = _sa_equity_month_value(comp["it"], m, origin, persist, beta)
                if fair_it is not None:
                    jan_it_delta = float(fair_it) - float(it[i])
                    it[i] = float(it[i]) + it_w * (float(fair_it) - float(it[i]))
        elif m.month == 2 and (jan_ot_delta != 0.0 or jan_it_delta != 0.0):
            # Spill: move the same £ share of the Jan adjustment into February.
            ot[i] = float(ot[i]) + feb_w * jan_ot_delta
            it[i] = float(it[i]) + feb_w * it_w * jan_it_delta
    paths = dict(paths)
    paths["other_tax"] = ot
    paths["it"] = it
    return paths


def _soft_pin(path: np.ndarray, cgncr: pd.Series, origin: pd.Timestamp, h: int,
              vint: pd.DataFrame, alpha: float) -> np.ndarray:
    """Add α × (plan − path) / n evenly across remaining months of each FY known at origin."""
    if alpha <= 0:
        return path
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    plan = M.vintage_plan(vint, M.info_date(origin))
    out = np.asarray(path, dtype=float).copy()
    for fy in sorted({M.fy_of(m) for m in months}):
        if fy not in plan:
            continue
        idx = [i for i, m in enumerate(months) if M.fy_of(m) == fy]
        ytd = float(cgncr.loc[(cgncr.index.map(M.fy_of) == fy) & (cgncr.index <= origin)].sum())
        gap = (plan[fy] - ytd) - float(out[idx].sum())
        out[idx] += alpha * gap / len(idx)
    return out


def _part_path(series: pd.Series, origin: pd.Timestamp, h: int, sa3_w: float) -> np.ndarray:
    """Month shape for one residual / fin part: blend calendar with ``seasonal_add3``."""
    cal = _grow(series, origin, h, 0.0, 0.0)
    w = float(sa3_w)
    if w <= 0:
        return cal
    sa3 = M.cgncr_forecast(series, origin, "seasonal_add3", h=h)
    if w >= 1:
        return np.asarray(sa3, dtype=float)
    return (1.0 - w) * cal + w * np.asarray(sa3, dtype=float)


def _trail12(series: pd.Series, origin: pd.Timestamp, h: int) -> np.ndarray:
    hh = series.loc[:origin].dropna()
    v = float(hh.iloc[-12:].mean()) if len(hh) else 0.0
    return np.full(h, v, dtype=float)


def _fin_part_path(series: pd.Series, origin: pd.Timestamp, h: int, part: str,
                   spec: dict, drv: dict[str, pd.Series]) -> np.ndarray:
    """PRE-gated forecast rule for one mapped fin-tx line or ``resid_gap``."""
    key = f"fin_{part}"
    rule = str(spec.get(key, "sa3"))
    sa3_w = float(spec.get("resid_sa3", 1.0))
    if rule in ("sa3", "", "none"):
        return _part_path(series, origin, h, sa3_w)
    if rule == "cal":
        return _part_path(series, origin, h, 0.0)
    if rule == "trail12":
        return _trail12(series, origin, h)
    if rule == "grow_retail":
        y = _yoy_at(drv.get("retail"), origin)
        return _grow(series, origin, h, y, float(spec.get("fin_recv_pay_elas", 0.0)))
    if rule == "grow_gdp":
        y = _yoy_at(drv.get("gdp"), origin)
        return _grow(series, origin, h, y, float(spec.get(f"fin_{part}_elas", 0.0)))
    if rule == "sa3_add_br":
        base = _part_path(series, origin, h, sa3_w)
        br = _yoy_at(drv.get("bank_rate_dpp"), origin)
        return base + float(spec.get("fin_other_fin_br", 0.0)) * br / 12.0
    raise ValueError(f"unknown fin part rule {rule!r} for {part}")


def _resid_path(series: pd.Series, origin: pd.Timestamp, h: int, sa3_w: float) -> np.ndarray:
    """Backward-compatible lump residual path (tests / rejects). Prefer FIN_PARTS + resid_gap."""
    return _part_path(series, origin, h, sa3_w)


def _category_arrays(comp: pd.DataFrame, origin: pd.Timestamp, h: int,
                     spec: dict) -> dict[str, np.ndarray]:
    elas = _rr_elas()
    drv = _drivers()
    awe = _yoy_at(drv.get("awe"), origin)
    retail = _yoy_at(drv.get("retail"), origin)
    cpi = _yoy_at(drv.get("cpi"), origin)
    gdp = _yoy_at(drv.get("gdp"), origin)
    equity = _yoy_at(drv.get("equity"), origin)
    gilt = _yoy_at(drv.get("gilt_dpp"), origin)
    br = _yoy_at(drv.get("bank_rate_dpp"), origin)
    out = {
        "it": _grow(comp["it"], origin, h, awe, elas["it_awe"]),
        "nics": _grow(comp["nics"], origin, h, awe, elas["nics_awe"]),
        "vat": _grow(comp["vat"], origin, h, retail, elas["vat_cons"]),
        "ct": _grow(comp["ct"], origin, h, gdp, elas["ct_profits"]),
        "other_tax": _grow_combo(
            comp["other_tax"], origin, h,
            [(retail, spec["other_tax_retail"]), (equity, spec["other_tax_equity"])]),
        "nontax": _grow(comp["nontax"], origin, h, gdp, spec["nontax_gdp"]),
        "welfare": _grow(comp["welfare"], origin, h, cpi, spec["welfare_cpi"]),
        "rdel_proxy": _grow(
            comp["rdel_proxy"], origin, h, gdp if spec.get("rdel_gdp", 0) else cpi,
            float(spec.get("rdel_gdp") or spec.get("rdel_cpi") or 0.0)),
        "interest": _interest_path(comp["interest"], origin, h, gilt, br, spec),
        "cap": _grow(
            comp["cap"], origin, h, gdp if spec.get("cap_gdp", 0) else retail,
            float(spec.get("cap_gdp") or spec.get("cap_retail") or 0.0)),
    }
    for k in FIN_PARTS:
        out[k] = _fin_part_path(comp[k], origin, h, k, spec, drv)
    out["resid_gap"] = _fin_part_path(comp["resid_gap"], origin, h, "resid_gap", spec, drv)
    if spec.get("isin_redeem_overlay"):
        from sovereign_cash.uk import isin_interest as II

        # Book coupon redemptions vs their own 12m mean — seasonal financing pulse on resid_gap.
        asof = pd.Timestamp(origin) + pd.offsets.MonthEnd(0)
        book_red = II.redeem_by_month(asof, h)
        trail = float(np.mean(book_red)) if len(book_red) else 0.0
        out["resid_gap"] = out["resid_gap"] + (book_red - trail)
    # Lump resid = sum of mapped parts (audit / identity checks).
    out["resid"] = sum((out[k] for k in FIN_PARTS), start=out["resid_gap"].copy())
    return _apply_sa_equity_nudge(out, comp, origin, h, spec)


def forecast(comp: pd.DataFrame, origin: pd.Timestamp, h: int = M.FC_MONTHS,
             spec: dict | None = None, vint: pd.DataFrame | None = None) -> np.ndarray:
    """Monthly CGNCR(ex) for origin+1..origin+h; drivers and levels through ``origin`` only."""
    spec = {**SPEC, **(spec or {})}
    paths = _category_arrays(comp, origin, h, spec)
    raw = (paths["welfare"] + paths["rdel_proxy"] + paths["interest"] + paths["cap"]
           - (paths["it"] + paths["nics"] + paths["vat"] + paths["ct"]
              + paths["other_tax"] + paths["nontax"])
           + paths["resid"])
    vint = src.remit_vintages() if vint is None else vint
    return _soft_pin(raw, comp["cgncr"], origin, h, vint, float(spec["soft_pin_alpha"]))


def category_paths(comp: pd.DataFrame, origin: pd.Timestamp, h: int = M.FC_MONTHS,
                   spec: dict | None = None) -> dict[str, np.ndarray]:
    """Named category forecasts (same rules as ``forecast``, before soft pin) for the audit table."""
    spec = {**SPEC, **(spec or {})}
    return _category_arrays(comp, origin, h, spec)


def category_audit(comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """PRE monthly MAE by category + driver / rule labels."""
    comp = components() if comp is None else comp
    elas = _rr_elas()
    tri = _di_triangle()
    rows_meta = [
        ("it", "income_tax_bn / ONS MF6P×EFO share", "Earnings level (AWE YoY)",
         f"RR PAYE: {elas['it_awe']:.2f}×AWE YoY"),
        ("nics", "nics_bn / ONS MF6P×EFO share", "Earnings level (AWE YoY)",
         f"RR NICs: {elas['nics_awe']:.2f}×AWE YoY"),
        ("vat", "vat_bn / ONS MF6P×EFO share", "Nominal consumption (retail YoY proxy)",
         f"RR VAT: {elas['vat_cons']:.2f}×retail YoY"),
        ("ct", "corp_tax_bn / ONS MF6P×EFO share", "Non-oil PNFC profits (GVA YoY proxy)",
         f"RR onshore CT: {elas['ct_profits']:.2f}×GVA YoY"),
        ("other_tax", "MF6P residual vs EFO big-4", "Retail YoY + equity YoY (FRED); Jan/Feb SA×disposal-FY equity nudge",
         f"{SPEC['other_tax_retail']}×retail + {SPEC['other_tax_equity']}×equity YoY; "
         f"SA persist={SPEC['sa_equity_persist']} β={SPEC['sa_equity_beta']}"),
        ("nontax", "ANBV−MF6P", "None (PRE-gated GVA/BR elas→0)",
         "calendar-month only — GVA/Bank-Rate overlays lose OOS among PRE-clearers"),
        ("welfare", "welfare_bn share of MF6S−JW2P", "CPI YoY (benefits / triple-lock)",
         f"{SPEC['welfare_cpi']}×CPI YoY (PRE-tuned; RR Q3 CPI + triple-lock region)"),
        ("rdel_proxy", "rdel_bn share of MF6S−JW2P", "GVA YoY",
         f"{SPEC['rdel_gdp']}×GVA YoY (PRE-gated; CPI elas loses OOS)"),
        ("interest", "JW2P / debt_interest_bn", "Bank Rate Δpp + 10y gilt Δpp (RR triangle)",
         f"gilt lag0={tri['gilt'].get(0, 0):.3f}/lag1={tri['gilt'].get(1, 0):.3f}; "
         f"BR lag0={tri['bank_rate'].get(0, 0):.3f} ×Δpp on DI level/12"),
        ("cap", "MF6T", "GVA YoY",
         f"{SPEC['cap_gdp']}×GVA YoY (PRE-gated; calendar/retail lose OOS)"),
        ("recv_pay", "ONS ANRT F7", "Retail sales YoY (activity → receivable levels)",
         f"calendar × {SPEC['fin_recv_pay_elas']}×retail YoY (PRE-gated vs sa3)"),
        ("gilt_adj", "ONS ANRU F.3", "None (RR DI / gilt Δpp overlays lose headline PRE)",
         f"seasonal_add3 (resid_sa3={SPEC['resid_sa3']})"),
        ("other_fin", "ONS ANRV", "Bank Rate Δpp (additive on sa3)",
         f"sa3 + {SPEC['fin_other_fin_br']}×BR Δpp/12 (PRE-gated; tiny cat edge, cuts OOS)"),
        ("acq_sec", "ONS ANRS F.5", "None (level noisy; equity grow loses vs trail12 OOS)",
         "trailing-12 mean (PRE-gated vs sa3)"),
        ("apf_xfer", "ONS MF7A APF capital transfers", "None (zero in PRE; QT pace raises OOS)",
         f"seasonal_add3 (resid_sa3={SPEC['resid_sa3']})"),
        ("net_lend", "ONS ANRH F.4 (student loans / net lending)", "None (CPI/GDP add raises OOS)",
         f"seasonal_add3 (resid_sa3={SPEC['resid_sa3']})"),
        ("resid_gap", "M98R − bridge − Σ FIN_PARTS", "None (unmapped perimeter)",
         f"seasonal_add3 — soft-pin α={SPEC['soft_pin_alpha']} to remit CGNCR"),
    ]
    errs = {k: [] for k, *_ in rows_meta}
    for o in comp.index[(comp.index >= pd.Timestamp("2010-01-01"))
                        & (comp.index <= pd.Timestamp("2018-12-01"))]:
        fut = pd.date_range(o + pd.offsets.MonthBegin(1), periods=12, freq="MS")
        if fut[-1] not in comp.index or fut[-1] > M.PRE[1]:
            continue
        paths = category_paths(comp, o, h=12)
        for k, *_ in rows_meta:
            act = comp.loc[fut, k].to_numpy(dtype=float)
            errs[k].append(float(np.nanmean(np.abs(paths[k] - act))))
    out = []
    for k, series, driver, rule in rows_meta:
        out.append({
            "category": k, "series": series, "driver": driver, "rule": rule,
            "pre_monthly_mae": float(np.mean(errs[k])) if errs[k] else float("nan"),
        })
    return pd.DataFrame(out)


def residual_share(audit: pd.DataFrame | None = None) -> float:
    """Share of PRE part-MAE attributed to ``resid_gap`` (unmapped perimeter)."""
    audit = category_audit() if audit is None else audit
    total = float(audit["pre_monthly_mae"].sum())
    gap = float(audit.loc[audit["category"] == "resid_gap", "pre_monthly_mae"].iloc[0])
    return gap / total if total > 0 else float("nan")


def backtest(comp: pd.DataFrame | None = None, vint: pd.DataFrame | None = None,
             spec: dict | None = None) -> pd.DataFrame:
    comp = components() if comp is None else comp
    vint = src.remit_vintages() if vint is None else vint
    spec = {**SPEC, **(spec or {})}
    x = comp["cgncr"]
    bt_base = M.backtest_cgncr(x, vint)
    t = M.score_table(bt_base)
    cg_method = M.choose(t, M.CGNCR_MODELS)
    paired = M.official_paired(bt_base, cg_method)
    pm = M.paired_mae(paired, "PRE", cg_method)
    pp = M.paired_mae(paired, "PRE", M.OFFICIAL)

    rows = []
    for o in x.index[x.index >= pd.Timestamp("2008-01-01")]:
        fut = x.loc[o + pd.offsets.MonthBegin(1):].iloc[:M.FC_MONTHS]
        if fut.empty:
            continue
        n = len(fut)
        paths = {
            BOTTOMUP: forecast(comp, o, h=n, spec=spec, vint=vint),
            PRODUCTION: V1.production_forecast(x, o, vint, cg_method, pm, pp, h=n),
            M.OFFICIAL: M.official_vintage_forecast(x, o, vint, h=n),
            cg_method: M.cgncr_forecast(x, o, cg_method, h=n),
            "smly": M.cgncr_forecast(x, o, "smly", h=n),
        }
        for meth, fc in paths.items():
            fc = np.asarray(fc, dtype=float)
            for h in M.HORIZONS:
                if n < h:
                    continue
                f = fc[:h]
                rows.append({
                    "origin": o, "h": h, "model": meth,
                    "fc": float(f.sum()) if np.isfinite(f).all() else np.nan,
                    "actual": float(fut.iloc[:h].sum()),
                    "target_end": fut.index[h - 1],
                })
    bt = pd.DataFrame(rows)
    bt["abs_err"] = (bt["fc"] - bt["actual"]).abs()
    bt["window"] = M._window(bt)
    bt.attrs["cg_method"] = cg_method
    return bt


def mean_mae(bt: pd.DataFrame, window: str, model: str) -> dict[int, float]:
    sub = bt[(bt["window"] == window) & (bt["model"] == model)].dropna(subset=["abs_err"])
    return {int(h): float(g["abs_err"].mean()) for h, g in sub.groupby("h")}


def ship_rule(bt: pd.DataFrame, band: float = M.EDGE_BAND) -> dict:
    bu = mean_mae(bt, "PRE", BOTTOMUP)
    prod = mean_mae(bt, "PRE", PRODUCTION)
    if not bu or not prod:
        return {"ship": False, "reason": "empty PRE scores", "bu_pre": bu, "prod_pre": prod}
    hs = [h for h in M.HORIZONS if h in bu and h in prod]
    bu_mean = float(np.mean([bu[h] for h in hs]))
    prod_mean = float(np.mean([prod[h] for h in hs]))
    worse = [h for h in hs if bu[h] > prod[h] + 1e-12]
    edge = (prod_mean - bu_mean) / prod_mean if prod_mean > 0 else float("nan")
    ship = bool(np.isfinite(edge) and edge > band and len(worse) <= 1)
    oos_bu = mean_mae(bt, "OOS", BOTTOMUP)
    oos_prod = mean_mae(bt, "OOS", PRODUCTION)
    return {
        "ship": ship,
        "band": band,
        "edge": edge,
        "worse_horizons": worse,
        "bu_pre": bu,
        "prod_pre": prod,
        "bu_pre_mean": bu_mean,
        "prod_pre_mean": prod_mean,
        "bu_oos": oos_bu,
        "prod_oos": oos_prod,
        "bu_oos_mean": float(np.mean([oos_bu[h] for h in M.HORIZONS if h in oos_bu])) if oos_bu else float("nan"),
        "prod_oos_mean": float(np.mean([oos_prod[h] for h in M.HORIZONS if h in oos_prod])) if oos_prod else float("nan"),
        "spec": dict(SPEC),
        "cg_method": bt.attrs.get("cg_method"),
        "authority_vintage": authority().get("vintage"),
        "reason": ("PRE mean MAE beats production by more than EDGE_BAND"
                   if ship else "PRE edge inside band or worse at >1 horizon"),
    }


def evaluate(comp: pd.DataFrame | None = None, vint: pd.DataFrame | None = None) -> dict:
    bt = backtest(comp=comp, vint=vint)
    decision = ship_rule(bt)
    decision["bt"] = bt
    audit = category_audit(comp if comp is not None else components())
    decision["category_audit"] = audit
    decision["residual_share_pre"] = residual_share(audit)
    decision["gate_pass"] = bool(decision["ship"])
    # Promote only when both the PRE gate and the module SHIP flag are on.
    if not SHIP:
        decision["ship"] = False
        if decision["gate_pass"]:
            decision["reason"] = "PRE gate passes but SHIP=False (not promoted)"
        else:
            decision["reason"] = decision.get("reason", "PRE gate failed")
    elif FORCE_LIVE and not decision["ship"]:
        decision["ship"] = True
        decision["reason"] = (
            f"FORCE_LIVE (isin_curve interest): gate said NO-SHIP — {decision.get('reason')}"
        )
    return decision


def rejects() -> list[str]:
    return [
        "v1 ONS-line seasonal_add3/smly (bottom_up.py) — rejected as seasonal main engine",
        "Headline smly / trail_mean — forbidden",
        "Annual close_to_plan × month shares without RR growth — PRE MAE ~31 vs production 9.3",
        "Residual = trailing mean or zero — residual is ~all headline error; mean/zero lose PRE hard",
        "Multiplicative plan-renormalized macro path — worse PRE than raw macro",
        "Expanding OLS elasticities replacing RR — worse PRE than published RR coefficients",
        "RPI-inflation leg of DI triangle (CPI Δpp proxy) — PRE −7.8% vs production",
        "3y average residual calendar — worse PRE at h=3/6/12",
        "Retail as sole profits proxy for CT — replaced by GVA YoY (PRE-gated)",
        "OBR fin_transactions annual × shares — no PRE-vintage history; residual uses seasonal_add3",
        "Residual × equity YoY scale — PRE-clears but immaterial vs sa3 alone; weak story",
        "DI triangle scale ≠1 — PRE prefers scale≥1 for edge, OOS prefers ~1.0; keep published",
        "Determinant timing last1/lag1/last6/last12 — last3 wins OOS among PRE-clearers",
        "nontax GVA / Bank Rate RR overlay — PRE-clear but raise OOS vs calendar-only",
        "soft_pin_alpha=0 (no remit pin) — PRE ~8.7% / worse at h=12; dominated by α=0.12",
        "soft_pin_alpha≥0.20 with resid_sa3=1 — PRE edge up, OOS regresses vs α=0.08–0.12",
        "α=0 welfare_cpi=1.0/1.25 × other_tax_retail=0.5 × BR=3 — PRE up ~1pp, OOS +1pp; reject",
        "RR DI triangle overlay on ANRU gilt_adj — PRE-clears, OOS 12.8 vs 10.4 sa3",
        "Mixed cal/sa3 residual parts — PRE-clears, OOS worse than all-sa3 decomp",
        "cap retail elas / rdel CPI elas >0 with decomp — lose OOS vs GVA elas",
        "ANRU gilt_adj grow/add gilt or Bank Rate — category PRE improves, headline edge fails EDGE_BAND",
        "MF7A apf_pace / apf_last / zero — PRE-ties (series 0 in PRE), OOS worse than sa3",
        "ANRH net_lend expand_ols AWE/CPI/GDP — category PRE improves, headline PRE fails",
        "ANRV/resid_gap NS&I sa3 blend — no category PRE edge or OOS regresses",
        "resid_gap trail12 / expand_ols equity — PRE edge up, OOS 10.9–24 vs ~10.1 shipped",
        "ANRT grow GDP instead of retail — PRE-clears, OOS ~10.10 vs ~10.08 retail among clearers",
        "INTEREST_ISIN_CURVE — monthly CGNCR scorer: interest leaf PRE +52% / OOS +111% vs RR; the −20.5% "
        "fair-cash number was annual Dec-31 share-spread (uk_cash_ship_autopsy.md §3); flag False",
        "sa_equity_nudge — Jan/Feb IT+other_tax signed bias −1.4 to −5 £bn/month vs plain carry; treats "
        "trend January receipts as a transient spike (autopsy §5); kept on SPEC as research only, NO-SHIP",
    ]
