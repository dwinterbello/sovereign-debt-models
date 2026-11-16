"""One-offs and run-rates on the US DTS forecast fiscal years, before ``us_dts.rebased`` uses their growth.

The DTS bucket path (``us/fiscal``) steps every line by its trailing 250-day level and growth. That carries FY2026's
one-offs forward as if they recur, and compounds trailing growth (defence +15%, Medicare +13% in FY2027). Every line
is put on a driver here. Each assumption is a named constant with its source. ``log()`` returns what changed.

  wd_other         FY26 holds ~$147bn of IEEPA tariff refunds (CBP withdrawals; Supreme Court, 20 Feb 2026). Out of the
                   base; the underlying line grows with nominal GDP; a refund tail is left in the first forecast year.
  wd_refunds       FY26 surge from OBBBA deductions backdated to TY2025 while withholding was unchanged; withholding
                   tables changed in January 2026, so only REFUND_PERSIST of it recurs.
  dep_corporate    on the post-reconciliation run-rate: FY+1 = FY26 x (1 + CORP_FIRST_GROWTH), then nominal GDP.
  dep_customs      on the post-ruling run-rate (last three full months x 12), then imports.
  dep_withheld     wage bill (payrolls x average hourly earnings, last 12 months) x WAGE_ELAST.
  dep_nonwithheld  nominal GDP plus EQ_ELAST x (FY-average S&P 500 growth - SP_NEUTRAL); S&P flat at the latest
                   month's average for the first forecast year, then SP_NEUTRAL a year.
  spending         Social Security on CBO's February 2026 growth, Medicare at CBO's 2026-30 average (timing shifts
                   smoothed); Medicaid at nominal GDP + 1.5pp and SNAP at nominal GDP (the 2025 act's cuts come through
                   the policy overlay, so not twice); veterans at nominal GDP + 2pp; defence and every other line at
                   nominal GDP.
  other receipts   other deposits, other taxes and education receipts at nominal GDP; Medicare premiums with Medicare;
                   Fed earnings held at the FY26 level (no remittance growth while the Fed carries a deferred asset).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FCACHE = REPO / "us" / "fiscal" / "data" / "cache"
INFL = REPO / "us" / "inflation" / "results" / "cache"
ENABLED = True

CBP_TAIL_BN = 35.0          # IEEPA refunds still due in FY27: ~$166-175bn collected (BDO; Wharton) less ~$130-147bn paid
                            # in FY26 (AAF; DTS); CBP withdrawals $8.8bn in Oct and $3.3bn to 10 Nov vs ~$1.5bn a month
REFUND_GROWTH = 0.05        # underlying refund growth (nominal incomes)
REFUND_PERSIST = 0.6        # share of the FY26 refund surge that recurs: the 2026 W-4 does not apply the tips, overtime and
                            # senior deductions by default (Step 4(b) opt-in), so most of the over-withholding repeats
REFUND_LAST_FY = 2029       # those deductions expire after tax year 2028: the last refunds they create are in FY2029
CUSTOMS_SCALE = 1.0         # multiplier on the customs run-rate (range: Section 301 rates and legal challenges)
SS_GROWTH = {0: 0.062, 1: 0.050}     # CBO Feb 2026: Social Security 1,666 -> 1,769 (FY27), then ~5.0% a year to 2036
MEDICARE_CBO = {2026: 1061.0, 2027: 1137.0, 2028: 1200.0, 2029: 1283.0, 2030: 1353.0}  # CBO Feb 2026, net mandatory
MEDICARE_GROWTH = (MEDICARE_CBO[2030] / MEDICARE_CBO[2026]) ** 0.25 - 1  # ~6.3% a year; smooths payment-timing shifts
MEDICAID_OVER_NGDP = 0.015  # pre-act trend over nominal GDP; the act's cuts are in the policy overlay
VA_OVER_NGDP = 0.02         # disability compensation caseload (PACT Act); assumption
SPEND_ON_DRIVER = ("wd_ssa", "wd_medicare", "wd_medicaid", "wd_va", "wd_defense", "wd_salaries", "wd_opm",
                   "wd_education", "wd_unemployment", "wd_snap", "wd_hhs_other")
RECEIPT_ON_DRIVER = ("dep_other", "dep_other_tax", "dep_education", "dep_medicare_prem", "dep_fed_earnings")
OTHER_RECEIPTS_OVER_NGDP = 0.0  # range: other deposits, other taxes and education receipts vs nominal GDP
CORP_FIRST_GROWTH = 0.10    # Jun-Oct 2026 +26% y/y on the Sep-2025 low; R&D catch-up deductions fade (Tax Foundation)
IMPORT_GROWTH = 0.04        # nominal goods imports
WAGE_ELAST = 1.0            # DTS 2014-26 ex Covid 0.73 (se 0.14, TCJA in sample); statutory payroll 1.0 + PIT ~1.3
EQ_ELAST = 0.7              # DTS FY2018-26 ex 2020-23: 0.59; FY2009 anchor ~0.85 (-36% April payments, S&P -43% log)
SP_NEUTRAL = 0.07           # long-run nominal equity price growth

_LOG: list[dict] = []
_LOGS: dict[tuple, list[dict]] = {}  # by params(), so log() always describes the table the cache returns


def _fred(path: Path) -> pd.Series:
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"]
    return pd.to_numeric(s, errors="coerce").dropna()


def _fy(idx) -> np.ndarray:
    t = pd.DatetimeIndex(idx)
    return t.year + (t.month >= 10).astype(int)


def _ngdp_growth(years: list[int]) -> dict[int, float]:
    from sovereign_cash.scenarios.channels import own_annual as OA

    span = list(range(min(years) - 1, max(years) + 1))
    g, _ = OA.levels("us", span)
    return {y: float(g[y] / g[y - 1] - 1.0) for y in years}


def log() -> list[dict]:
    return list(_LOGS.get(params(), _LOG))


def params() -> tuple:
    """Every assumption, so caches downstream are keyed on them."""
    return (ENABLED, CBP_TAIL_BN, REFUND_GROWTH, REFUND_PERSIST, REFUND_LAST_FY, CUSTOMS_SCALE, CORP_FIRST_GROWTH,
            IMPORT_GROWTH, WAGE_ELAST, EQ_ELAST, SP_NEUTRAL, MEDICAID_OVER_NGDP, VA_OVER_NGDP,
            tuple(sorted(SS_GROWTH.items())), MEDICARE_GROWTH, SPEND_ON_DRIVER, RECEIPT_ON_DRIVER,
            OTHER_RECEIPTS_OVER_NGDP)


def adjust(fy: pd.DataFrame, hist: pd.DataFrame, first_fc: int) -> pd.DataFrame:
    """``fy``: us_dts FY table ($bn, bucket columns); ``hist``: daily history ($bn); ``first_fc``: first forecast FY."""
    _LOG.clear()
    if not ENABLED:
        return fy
    base = first_fc - 1
    years = [int(y) for y in fy.index if int(y) >= first_fc]
    if not years or base not in fy.index or (base - 1) not in fy.index:
        return fy
    out = fy.copy()
    ng = _ngdp_growth(years)

    def put(col, vals, why):
        old = [float(out.at[y, col]) for y in years]
        for y, v in zip(years, vals):
            out.at[y, col] = float(v)
        _LOG.append({"line": col, "why": why, "years": years, "model": [round(v, 1) for v in old],
                     "adjusted": [round(float(v), 1) for v in vals]})

    # tariff refunds
    t2 = pd.read_parquet(FCACHE / "dts_table2_raw.parquet")
    t2["record_date"] = pd.to_datetime(t2["record_date"])
    cbp = t2[(t2.transaction_catg == "DHS - Customs & Border Protection (CBP)") & (t2.transaction_type == "Withdrawals")]
    cbp_fy = cbp.groupby(_fy(cbp.record_date)).amt_mn.sum() / 1000.0
    normal = float(cbp_fy.reindex([base - 2, base - 1]).mean())
    excess = float(cbp_fy.get(base, normal) - normal)
    x, v = float(fy.at[base, "wd_other"]) - excess, []
    for y in years:
        x *= 1 + ng[y]
        v.append(x + (CBP_TAIL_BN if y == first_fc else 0.0))
    put("wd_other", v, f"FY{base} carries ${excess:.0f}bn of IEEPA tariff refunds (CBP ${cbp_fy.get(base, 0):.0f}bn vs "
        f"${normal:.0f}bn normal): out of the base, line grown with nominal GDP, ${CBP_TAIL_BN:.0f}bn tail in FY{first_fc}")
    # refunds
    r0, r1 = float(fy.at[base - 1, "wd_refunds"]), float(fy.at[base, "wd_refunds"])
    trend = r0 * (1 + REFUND_GROWTH)
    surge = r1 - trend
    put("wd_refunds", [(trend + (REFUND_PERSIST * surge if y <= REFUND_LAST_FY else 0.0)) * (1 + REFUND_GROWTH) ** (y - base)
                       for y in years],
        f"FY{base} surge ${surge:.0f}bn over trend (deductions backdated to TY2025); {REFUND_PERSIST:.0%} recurs "
        f"to FY{REFUND_LAST_FY} (the deductions expire after TY2028)")
    # corporate
    v = [float(fy.at[base, "dep_corporate"]) * (1 + CORP_FIRST_GROWTH)]
    for y in years[1:]:
        v.append(v[-1] * (1 + ng[y]))
    put("dep_corporate", v, f"FY{first_fc} = FY{base} +{CORP_FIRST_GROWTH:.0%} (post-reconciliation run-rate), then NGDP")
    # customs
    m = hist.resample("ME").sum()
    m = m[m.index < hist.index.max().to_period("M").to_timestamp()]  # complete months only
    rr = float(m["dep_customs"].iloc[-3:].mean() * 12.0) * CUSTOMS_SCALE
    put("dep_customs", [rr * (1 + IMPORT_GROWTH) ** (i + 1) for i in range(len(years))],
        f"run-rate ${rr:.0f}bn/yr (last 3 full months; IEEPA collections stopped Feb 2026), imports +{IMPORT_GROWTH:.0%}")
    # withheld
    wb = (_fred(INFL / "fred_PAYEMS.csv") * _fred(INFL / "fred_CES0500000003.csv")).dropna()
    wbg = float(wb.iloc[-12:].mean() / wb.iloc[-24:-12].mean() - 1.0)
    w = float(fy.at[base, "dep_withheld"])
    put("dep_withheld", [w * (1 + WAGE_ELAST * wbg) ** (i + 1) for i in range(len(years))],
        f"wage bill +{wbg:.1%} a year (payrolls x hourly earnings, last 12m), elasticity {WAGE_ELAST}")
    # non-withheld
    sp = _fred(FCACHE / "fred_sp500.csv")
    sp_fy = sp.groupby(_fy(sp.index)).mean()
    last_m = float(sp[sp.index >= sp.index.max().to_period("M").to_timestamp()].mean())
    g_sp = {y: (last_m / float(sp_fy[base]) - 1.0) if y == first_fc else SP_NEUTRAL for y in years}
    x, v = float(fy.at[base, "dep_nonwithheld"]), []
    for y in years:
        x *= float(np.exp(np.log1p(ng[y]) + EQ_ELAST * (np.log1p(g_sp[y]) - np.log1p(SP_NEUTRAL))))
        v.append(x)
    put("dep_nonwithheld", v, f"NGDP + {EQ_ELAST} x (S&P FY-average growth - {SP_NEUTRAL:.0%}); S&P flat at the latest "
        f"month (FY{first_fc} +{g_sp[first_fc]:.1%}), then +{SP_NEUTRAL:.0%}")
    # spending lines on drivers
    for col in SPEND_ON_DRIVER:
        if col not in out.columns:
            continue
        x, v = float(fy.at[base, col]), []
        for i, y in enumerate(years):
            if col == "wd_ssa":
                g = SS_GROWTH[min(i, 1)]
            elif col == "wd_medicare":
                g = MEDICARE_GROWTH
            elif col == "wd_medicaid":
                g = ng[y] + MEDICAID_OVER_NGDP
            elif col == "wd_va":
                g = ng[y] + VA_OVER_NGDP
            else:
                g = ng[y]
            x *= 1 + g
            v.append(x)
        put(col, v, {"wd_ssa": "CBO Feb 2026 Social Security growth", "wd_medicare": "CBO Feb 2026 Medicare growth",
                     "wd_medicaid": f"nominal GDP + {MEDICAID_OVER_NGDP:.1%} (the act's cuts in the overlay)",
                     "wd_va": f"nominal GDP + {VA_OVER_NGDP:.0%}"}.get(col, "nominal GDP (was trailing growth)"))
    # remaining receipt lines on drivers
    for col in RECEIPT_ON_DRIVER:
        if col not in out.columns:
            continue
        x, v = float(fy.at[base, col]), []
        for y in years:
            if col == "dep_fed_earnings":
                g = 0.0
            elif col == "dep_medicare_prem":
                g = MEDICARE_GROWTH
            else:
                g = ng[y] + OTHER_RECEIPTS_OVER_NGDP
            x *= 1 + g
            v.append(x)
        put(col, v, {"dep_fed_earnings": "held at FY26 (Fed deferred asset)",
                     "dep_medicare_prem": "with Medicare (CBO average growth)"}.get(col, "nominal GDP (was trailing growth)"))
    _LOGS[params()] = list(_LOG)
    # totals
    deps = [c for c in out.columns if c.startswith("dep_") and c != "dep_total"]
    wdp = [c for c in out.columns if c.startswith("wd_") and c not in ("wd_total", "wd_interest")]
    out.loc[years, "receipts"] = out.loc[years, deps].sum(axis=1)
    out.loc[years, "primary_spend"] = out.loc[years, wdp].sum(axis=1)
    return out
