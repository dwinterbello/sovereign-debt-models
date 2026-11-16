"""Line-by-line cyclical response of the budget: the downturn (and upturn) behind the note's growth scenarios.

The scenario engine moves total receipts with nominal GDP and holds primary spending in cash. That misses where a
downturn actually lands: corporate tax and capital-gains-linked receipts fall much faster than GDP, unemployment
spending rises, and the US federal budget is unusually exposed to equities. This module puts each line on its driver.

One model for every scenario leg, on the same lines:
  growth     nominal GDP ``size_pp`` a year below / above baseline for two years from ``y0 + 1``, no catch-up (level
             gap ``size`` in year 1, ``2 x size`` after), as ``debt.note.scenario_grid``. REAL_SHARE of the gap is
             real output (the cyclical channels below), the rest the price level (the price leg).
  inflation  ``price_lines``: the price level ``infl_pp`` a year above baseline for two years, real output unchanged.
  rates      ``rates_lines``: the non-interest receipts that rates move. Interest itself is the ISIN book.

Price leg, every country: receipts follow their nominal bases in the same year (unit-elastic; the UK on the OBR
reckoner, which carries fiscal drag from frozen thresholds); spending follows prices a year later (PRICE_LAG_YEARS):
benefits and pensions are uprated each year on the previous year's inflation (US COLA and Thrifty Food Plan, UK
September CPI and the triple lock, euro-area January / July pension indexation), pay settlements and departmental
budgets are set a year ahead, and PRICE_PASS_THROUGH of spending follows (discretionary purchases are budgeted in
cash). The UK keeps the OBR reckoner and its convention (departmental spending fixed in cash within the Spending
Review). Checked against CBO's rule of thumb for inflation in the tests.

US (federal, DTS lines; ``us_dts.fy_table`` with ``us_dts_adjust`` one-offs out). Drivers follow from the gap:
  unemployment  OKUN_US x real gap (2001: +2pp on a ~3% gap)
  wage bill     WAGE_TO_NGDP x nominal gap
  equities      EQUITY path (FY-average S&P 500, -30% peak to trough), scaled with the shock
  profits       PROFIT_TO_NGDP x nominal gap, capped at the PROFIT path (2001 ~-20%; 2008-09 ~-35%)
  AI capex      AI_CAPEX path; a smaller capex cut means fewer expensing deductions (a cushion to corporate tax)
  imports       IMPORT_TO_NGDP x nominal gap
  line elasticities: us_dts_adjust (withheld, non-withheld) and the constants below (corporate, UI, SNAP, Medicaid,
  refunds, COLA). State and local budgets are assumed to balance (balanced-budget rules), so general government
  moves with the federal lines.

UK (public sector, PSNB): the OBR ready reckoner (``uk.ready_reckoner``, March 2025 EFO), line by line, with the
determinants moved together: earnings and employment (wage bill = WAGE_TO_NGDP x gap), unemployment (OKUN_UK),
nominal and real consumption, profits and equity prices (the US paths), imports, house prices and transactions, the
price level and benefit uprating. Debt interest and the financing loop are left to the ISIN book.

France has no published line-level reckoner. The PLF's Voies et moyens (tome 1, in data/fr/plf_extracts) sets out each
tax's base and how it is collected, and the corporate tax timing matters in a downturn: four acomptes on the last
closed year's profit, autolimitation and the large-firm fifth acompte on the current year, the solde in May of the
next year. So French corporate tax responds FR_IS_SAME_YEAR in the shock year and the rest a year later (2008-09:
the collapse in net IS came in 2009, after 2008 profits fell). The PLF's own planning priors (data/fr/authority.json:
IR, TVA and IS unit-elastic to their bases) are medium-term and are not used for the cycle.

DE, FR, IT, ES, JP (general government; no line-level national reckoner in the framework): OECD revenue elasticities to the output gap by tax (PIT, social
contributions, corporate, indirect; non-tax revenue zero) and the current-primary-spending elasticity including
unemployment and earnings-related benefits (Price, Dang and Botev 2015, OECD ECO/WKP(2015)93, Tables 8-10). The price
level is the common price leg above (fiscal drag from unindexed thresholds in IT, ES, JP is not modelled). Corporate tax is
on the OECD corporate elasticity too, raised in a downturn by CIT_STRESS (OECD elasticities average the cycle and
understate sharp recessions), fading back to the OECD value by the fourth year, and with each country's advance-payment
timing (CIT_SAME_YEAR). No stress in an upturn: losses carried forward absorb the recovery in profits.

Stimulus (optional): STIMULUS_PCT_GDP of shocked GDP, all countries, downturns only.
Interest is not here: the ISIN-book channels in the scenario pack move it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SC = Path(__file__).resolve().parents[1]
DERIVED = SC / "derived"
CANON_PP = 2.0          # the recession the paths below describe: -2pp nominal growth for two years
REAL_SHARE = 0.75       # share of the nominal gap that is real output
FR_IS_SAME_YEAR = 0.35  # share of the French corporate-tax response in the shock year (Voies et moyens acompte/solde)
UK_SOURCE = "obr"       # "obr": OBR ready reckoner; "oecd": the OECD elasticities (kept for comparison)
PRICE_LAG_YEARS = 1     # spending follows the price level a year later (uprating on last year's inflation, budgets set ahead)
PRICE_PASS_THROUGH = 0.8  # share of primary spending that follows prices within the horizon (benefits, pensions, pay,
                          # health); the rest (discretionary purchases, investment) is budgeted in cash. Calibrated to
                          # CBO's inflation rule of thumb (outlays 44 / 95 / 156bn per pp in years 2-4 against ~62bn a
                          # year for full one-year-lagged indexation: 0.7-0.84) and consistent with the EU spending mix
                          # (social protection ~39% of spending, Eurostat, plus pay ~20% and health).
# Corporate tax. US and UK: on the common profit path (profit_path) at CORP_PROFIT_ELAST (US federal history; the OBR
# reckoner's profits determinant). DE, FR, IT, ES, JP: the OECD output-gap elasticity times CIT_STRESS in a downturn.
# Calibration: Italy, the one big-four country whose 2007-09 receipts fell less than half, central-government tax on
# profits went from 2.98% to 2.26% of GDP (OECD Revenue Statistics 1210), ~-26% in euros over a ~5pp output-gap swing and
# despite the 2008 IRES cut (33% -> 27.5%): an elasticity of ~4.5 at the trough against the OECD's 2.61 (x1.7).
# Applied to all five, so the cross-country comparison rests on the OECD elasticities, not on a common US profit path.
CIT_STRESS = {1: 1.7, 2: 1.7, 3: 1.35}  # by year after onset; 1.0 from year 4 (back to the OECD average-cycle value)
JP_CIT_SAME_YEAR = 0.5    # Japan: interim payment on half the prior year's tax, final two months after the year end
CIT_SAME_YEAR = {"fr": FR_IS_SAME_YEAR, "jp": JP_CIT_SAME_YEAR,
                 "it": 0.5,   # Italy: IRES acconti on the prior year's tax (40% June, 60% November; a forecast-based
                              # reduction is allowed at the firm's risk), saldo the next June
                 "de": 0.5}   # Germany: Vorauszahlungen on the last assessment, cut on application in a downturn
                              # Spain (pagos fraccionados on current-year profit for large firms): same year
UB_SHARE_NOW = {"es": 3.75, "fr": 3.4}  # unemployment spending, % of current primary spending, today: ESSPROS 2024 early
                          # estimates (Spain 1.5% of GDP, France 1.75%) on current primary spending of ~40% / ~51% of GDP.
                          # The OECD weights are 2002-11 averages (Spain 6.26%, crisis unemployment). Others: OECD weights.
# Corporate tax, % of total general-government revenue, today. The OECD weights (Table 9) are 2002-11 averages: boom-era
# Spain (8.31%), pre-reform Germany (6.11%) and a low Japan (9.61%). Rebased: OECD Revenue Statistics 2025, Table 1.1,
# 2023 CIT share of total tax x tax / GDP (FR 5.4 x 43.9, DE 6.0 x 37.3, IT 6.6 x 41.5, JP 13.9 x 33.7, ES 8.0 x 36.4),
# over general-government revenue / GDP in the engine (2025: FR 52.1, DE 47.9, IT 48.1, JP 37.1, ES 42.9).
CIT_SHARE_NOW = {"fr": 4.55, "de": 4.68, "it": 5.70, "jp": 12.63, "es": 6.78}
UK_DEL_SHARE = 0.5        # departmental spending (DEL) ~half of UK primary spending (OBR EFO). The reckoner holds DEL in
                          # cash; it stays in cash through the 2025 Spending Review's plans (to 2028-29), then the next
                          # review resets it to prices like every other country's spending (common price leg).
UK_DEL_CASH_UNTIL = 2028  # last fiscal year (starting April) of the Spending Review's cash plans
STIMULUS_PCT_GDP = {1: 0.0, 2: 1.0, 3: 0.3}   # by year after onset; 2001 ~0.4% (rebates), 2008 ~1.0% (ESA), 2009 ARRA ~2%/yr

# ------------------------------------------------------------------ OECD elasticities (Price, Dang & Botev 2015)
OECD_SOURCE = ("Price, Dang and Botev (2015), Adjusting fiscal balances for the business cycle, OECD Economics "
               "Department WP 1275, Tables 8-10")
# tax: (elasticity to the output gap, share of total GG revenue %); cp: current primary spending elasticity to the gap
# (incl. unemployment and earnings-related benefits) and current primary share of total spending (%); ub: unemployment
# spending elasticity and its share of current primary spending (%); semi: OECD budget-balance semi-elasticity.
OECD = {
    "fr": {"pit": (1.85, 17.47), "ssc": (0.70, 35.80), "cit": (3.09, 5.07), "ind": (0.91, 30.10),
           "cp": (-0.14, 91.68), "ub": (-2.33, 3.14), "semi": 0.61},
    "de": {"pit": (1.86, 19.20), "ssc": (0.52, 39.24), "cit": (1.97, 6.11), "ind": (0.95, 24.57),
           "cp": (-0.15, 91.07), "ub": (-3.07, 3.58), "semi": 0.48},
    "it": {"pit": (1.43, 24.75), "ssc": (0.55, 28.30), "cit": (2.61, 5.97), "ind": (1.10, 31.34),
           "cp": (-0.04, 85.83), "ub": (-2.91, 1.28), "semi": 0.50},
    "jp": {"pit": (1.76, 15.88), "ssc": (0.72, 34.37), "cit": (2.17, 9.61), "ind": (1.04, 25.82),
           "cp": (-0.07, 89.12), "ub": (-3.14, 1.16), "semi": 0.41},
    "es": {"pit": (1.76, 19.09), "ssc": (0.71, 33.63), "cit": (2.11, 8.31), "ind": (1.41, 28.56),
           "cp": (-0.44, 86.70), "ub": (-6.77, 6.26), "semi": 0.63},
    "uk": {"pit": (1.71, 31.15), "ssc": (0.73, 20.31), "cit": (2.71, 8.07), "ind": (1.29, 31.26),
           "cp": (-0.10, 89.31), "ub": (-4.28, 0.78), "semi": 0.59},
    "us": {"pit": (2.08, 28.66), "ssc": (1.00, 21.07), "cit": (1.95, 7.69), "ind": (1.00, 22.42),
           "cp": (-0.18, 86.22), "ub": (-7.65, 1.46), "semi": 0.50},
}
TAXES = ("pit", "ssc", "cit", "ind")
# UK: OBR ready reckoner
OKUN_UK = 0.4             # unemployment pp per 1% real output gap (UK labour hoarding; 2008-09 ~+3pp on ~-6% GDP)
EMP_TO_REAL = 0.4         # employment level % per 1% real gap
LF_UK_M = 34.0            # labour force, millions
HOUSE_TO_NGDP = 1.5       # house prices (2008-09: ~-15%)
TRANS_TO_NGDP = 4.0       # property transactions (2008-09: ~-40%)
UK_GROUP = {  # OBR line -> chart line
    "PAYE income tax": "uk_inc", "SA income tax": "uk_inc", "NICs": "uk_inc",
    "Capital gains tax": "uk_asset", "Stamp taxes": "uk_asset", "Inheritance tax": "uk_asset",
    "Onshore corporation tax": "uk_cit", "Offshore corporation tax": "uk_cit", "Energy profits levy": "uk_cit",
    "Company and other tax credits": "uk_cit",
    "Universal credit": "uk_ben", "Working age benefits": "uk_ben", "Employment and support allowances": "uk_ben",
    "Other benefits": "uk_ben", "Disability benefits": "uk_ben", "State pensions": "uk_pen",
    "Additional pensions": "uk_pen", "Unfunded public sector pensions": "uk_pen"}
UK_SPEND_BLOCK = "spending"
UK_LABEL = {"uk_inc": "Income tax & NICs", "uk_asset": "CGT, stamp duty, IHT (assets)", "uk_cit": "Corporation tax",
            "uk_ind": "VAT, duties, other receipts", "uk_ben": "Benefits", "uk_pen": "Pensions (uprating)",
            "uk_osp": "Other spending"}
TAX_LABEL = {"pit": "Personal income tax", "ssc": "Social contributions", "cit": "Corporate tax",
             "ind": "Indirect taxes", "cp": "Benefits (unemployment, earnings-related)", "stimulus": "Stimulus"}
PRICE_LABEL = "Price level net of indexation"

# ------------------------------------------------------------------ US federal lines
OKUN_US = 0.65            # unemployment pp per 1% real output gap (2001: +2.0pp)
UR_BASE = 4.3
WAGE_TO_NGDP = 0.9        # wage bill vs nominal GDP gap (labour share rises in downturns)
IMPORT_TO_NGDP = 2.5      # nominal goods imports (2001: ~-6%; 2009: ~-26% on ~-3% NGDP)
PROFIT_TO_NGDP = 5.0      # pre-tax profits vs nominal GDP gap, capped by the PROFIT path below
EQUITY = {1: -0.12, 2: -0.25, 3: -0.15, 4: -0.10}     # FY-average S&P 500 vs baseline (-30% peak to trough)
PROFIT = {1: -0.10, 2: -0.20, 3: -0.15}               # caps; from year 4 profits are back in line with nominal GDP
AI_CAPEX = {1: -0.15, 2: -0.30, 3: -0.25, 4: -0.20}
CORP_PROFIT_ELAST = 1.5   # FY2000-03 -36% and FY2007-09 -63% on profit falls of ~20-25% and ~35-40%; no NOL carryback
AI_CAPEX_BASE_BN = 550.0  # assumption: AI-related US capex run-rate
AI_EXPENSED_SHARE = 0.6   # equipment and software expensed (not structures)
CORP_RATE = 0.21
UI_UR_ELAST = 1.54        # log(UI / wage bill) on log(unemployment rate), DTS 2014-26 ex Covid (se 0.10, R2 0.68)
SNAP_PER_PP = 0.05        # +5% enrolment per 1pp (literature 2.3-15%: Ziliak; Bitler-Hoynes 3.4; Ganong-Liebman 15)
MEDICAID_PER_PP = 0.01    # +1m enrolees per 1pp (Dorn et al.) on ~80m, new enrolees ~2/3 of average cost
REFUNDS = {1: 0.03, 2: 0.08, 3: 0.05, 4: 0.03}        # refunds rise (over-withholding, EITC); FY2009 +24% incl. MWPC
SSA_EARLY = {1: 0.002, 2: 0.005, 3: 0.005, 4: 0.004}  # early retirement claims
OTHER_TAX_ELAST = 0.5
US_LINE_LABEL = {"dep_withheld": "Withheld income & payroll tax", "dep_nonwithheld": "Non-withheld tax (equities)",
                 "dep_corporate": "Corporate tax", "dep_customs": "Customs", "dep_other_tax": "Other receipts",
                 "dep_other": "Other receipts", "wd_refunds": "Refunds", "wd_unemployment": "Unemployment insurance",
                 "wd_snap": "SNAP", "wd_medicaid": "Medicaid", "wd_ssa": "Social Security (early claims)",
                 "stimulus": "Stimulus"}


def gap(k: int, size_pp: float) -> float:
    """Nominal GDP level gap (fraction) k years after onset (k = 1, 2, ...)."""
    return 0.0 if k <= 0 else (size_pp if k == 1 else 2 * size_pp) / 100.0


def _k(y: int, y0: int) -> int:
    return y - y0


@lru_cache(maxsize=1)
def _annual() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "debt_country_annual.parquet").set_index(["cc", "year"])


def _gg(cc: str, y: int) -> dict:
    a = _annual().loc[(cc, y)]
    dpct = a["deficit_authority_pct_gdp"] if pd.notna(a["deficit_authority_pct_gdp"]) else a["deficit_weo_pct_gdp"]
    rev, gdp = float(a["gg_revenue_bn"]), float(a["gdp_bn"])
    return {"gdp": gdp, "rev": rev, "spend": rev + float(dpct) / 100 * gdp}


def profit_path(k: int, size_pp: float) -> float:
    """Nominal pre-tax profits vs baseline (fraction) k years after onset: the canonical path, capped by the gap, and
    back in line with nominal GDP from year 4 (profits recover before output). Used for the US and the UK."""
    if k <= 0 or size_pp == 0.0:
        return 0.0
    s = -size_pp / CANON_PP
    ng = gap(k, size_pp)
    if k not in PROFIT:
        return ng
    pr = PROFIT[k] * s
    return max(pr, PROFIT_TO_NGDP * ng) if s > 0 else min(pr, PROFIT_TO_NGDP * ng)


def cp_elasticity(cc: str) -> float:
    """Current primary spending elasticity, unemployment part rebased to today's spending share where known."""
    o = OECD[cc]
    eps_cp = o["cp"][0]
    if cc in UB_SHARE_NOW:
        eps_ub, sh_ub = o["ub"]
        eps_cp += eps_ub * (UB_SHARE_NOW[cc] - sh_ub) / 100
    return eps_cp


def cit_share(cc: str) -> float:
    """Corporate tax, % of total GG revenue: today's share where rebased (CIT_SHARE_NOW), else the OECD weight."""
    return CIT_SHARE_NOW.get(cc, OECD[cc]["cit"][1])


def _oecd(cc: str, y: int, k: int, size_pp: float, cit_stress: float) -> dict[str, float]:
    """GG line responses, local bn; receipts + = more receipts, spending + = more spending.

    Corporate tax: the OECD output-gap elasticity on the real gap, times CIT_STRESS in a downturn, split between this
    year's and last year's gap by the advance-payment timing (CIT_SAME_YEAR). Its price part is in the price leg."""
    o, g = OECD[cc], _gg(cc, y)
    ng = gap(k, size_pp)
    real = REAL_SHARE * ng
    out = {}
    for t in ("pit", "ssc", "ind"):
        eps, sh = o[t]
        out[t] = g["rev"] * sh / 100 * eps * real
    same = CIT_SAME_YEAR.get(cc, 1.0)
    real_cit = same * real + (1 - same) * REAL_SHARE * gap(k - 1, size_pp)
    stress = CIT_STRESS.get(k, 1.0) if size_pp < 0 else 1.0
    out["cit"] = g["rev"] * cit_share(cc) / 100 * o["cit"][0] * stress * cit_stress * real_cit
    out["cp"] = g["spend"] * o["cp"][1] / 100 * cp_elasticity(cc) * real
    return out


def _us_base() -> pd.DataFrame:
    from sovereign_cash.scenarios.channels import us_dts as U

    return U.fy_table()  # cached there, keyed on the adjustment switch and its parameters


def _us(y: int, k: int, size_pp: float) -> dict[str, float]:
    """US federal line responses, $bn (FY ~ calendar year)."""
    from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ

    B = _us_base()
    if y not in B.index:
        y = int(B.index.max())
    s = -size_pp / CANON_PP                      # 1.0 = the canonical recession; -1.0 its upturn mirror
    ng = gap(k, size_pp)
    kk = min(k, 4)
    rg = REAL_SHARE * ng                         # real part; the price part is the price leg
    ur = -OKUN_US * rg * 100                     # pp
    wb = WAGE_TO_NGDP * rg
    eq = EQUITY.get(kk, 0.0) * s
    pr = profit_path(k, size_pp)
    cap = AI_CAPEX.get(kk, 0.0) * s
    imp = IMPORT_TO_NGDP * rg
    b = lambda c: float(B.at[y, c])  # noqa: E731
    out = {
        "dep_withheld": b("dep_withheld") * ADJ.WAGE_ELAST * wb,
        "dep_nonwithheld": b("dep_nonwithheld") * (np.exp(np.log1p(rg) + ADJ.EQ_ELAST * np.log1p(eq)) - 1.0),
        "dep_corporate": b("dep_corporate") * CORP_PROFIT_ELAST * pr
                         - cap * AI_CAPEX_BASE_BN * AI_EXPENSED_SHARE * CORP_RATE,
        "dep_customs": b("dep_customs") * imp,
        "dep_other_tax": b("dep_other_tax") * OTHER_TAX_ELAST * rg,
        "dep_other": b("dep_other") * OTHER_TAX_ELAST * rg,
        "wd_refunds": b("wd_refunds") * REFUNDS.get(kk, 0.0) * s,
        "wd_unemployment": b("wd_unemployment") * (((UR_BASE + ur) / UR_BASE) ** UI_UR_ELAST * (1 + wb) - 1.0),
        "wd_snap": b("wd_snap") * SNAP_PER_PP * ur,
        "wd_medicaid": b("wd_medicaid") * MEDICAID_PER_PP * ur,
        "wd_ssa": b("wd_ssa") * SSA_EARLY.get(kk, 0.0) * s,
    }
    return out


def _uk_fy(y: int) -> str:
    return f"{y}-{str(y + 1)[2:]}"


@lru_cache(maxsize=16)
def _uk_all(size_pp: float, y0: int, last: int) -> dict[int, dict[str, float]]:
    """UK line responses by calendar year (= fiscal year starting April), £bn, from the OBR ready reckoner."""
    from sovereign_cash.uk import ready_reckoner as RR

    s = -size_pp / CANON_PP
    fys = [y for y in range(y0 + 1, last + 1) if _uk_fy(y) in RR.YEARS]
    shocks: dict[str, dict[str, float]] = {}

    def put(det, y, v):
        shocks.setdefault(det, {})[_uk_fy(y)] = float(v)

    for y in fys:
        k = y - y0
        g = 100 * gap(k, size_pp)
        real = REAL_SHARE * g
        emp = EMP_TO_REAL * real
        kk = min(k, 4)
        pr = 100 * profit_path(k, size_pp)
        put("Earnings level", y, WAGE_TO_NGDP * real - emp)
        put("Employment level", y, emp)
        put("Unemployment (100,000 change)", y, -OKUN_UK * real / 100 * LF_UK_M)  # scenario input is in millions
        put("Nominal consumption", y, real)
        put("Real consumption", y, real)
        put("Real GDP", y, real)
        put("Calendar year nominal GDP", y, real)
        put("Non-seasonally adjusted nominal GDP", y, real)
        put("Non-oil, non-financial profits", y, pr)
        put("Financial profits", y, pr)
        put("Equity prices", y, 100 * EQUITY.get(kk, 0.0) * s)
        put("Imports", y, IMPORT_TO_NGDP * real)
        put("House prices", y, HOUSE_TO_NGDP * real)
        put("Property transactions", y, TRANS_TO_NGDP * real)
    eff = RR.run(shocks)
    eff = eff[eff["block"].isin(["receipts", UK_SPEND_BLOCK])]
    out: dict[int, dict[str, float]] = {}
    for _, r in eff.iterrows():
        y = int(str(r["effect_fy"])[:4])
        grp = UK_GROUP.get(r["line"], "uk_ind" if r["block"] == "receipts" else "uk_osp")
        key = ("rec:" if r["block"] == "receipts" else "spd:") + grp
        out.setdefault(y, {})
        out[y][key] = out[y].get(key, 0.0) + float(r["gbp_bn"])
    return out


def _uk(y: int, k: int, size_pp: float, y0: int) -> dict[str, float]:
    from sovereign_cash.uk import ready_reckoner as RR

    last = max(int(f[:4]) for f in RR.YEARS)
    A = _uk_all(float(size_pp), y0, last)
    return dict(A.get(min(y, last), {}))


# ------------------------------------------------------------------ price leg (common to every scenario)
UK_PRICE_LEVEL = ("CPI index", "Average RPI", "Earnings level", "Nominal consumption", "Non-oil, non-financial profits",
                  "Financial profits", "Calendar year nominal GDP", "Non-seasonally adjusted nominal GDP", "Imports",
                  "House prices")
UK_PRICE_RATE = ("Q3 CPI", "Q3 CPI for benefits (zero floor)", "Triple lock", "Q3 average earnings growth",
                 "FY average earnings growth")


def _price_gap(k: int, infl_pp: float) -> float:
    return 0.0 if k <= 0 else (infl_pp if k == 1 else 2 * infl_pp) / 100.0


@lru_cache(maxsize=32)
def _uk_price(path: tuple, nominal_paths: bool = False) -> dict[int, dict[str, float]]:
    """OBR reckoner, price determinants; ``path`` = ((fy, level %, rate pp), ...)."""
    from sovereign_cash.uk import ready_reckoner as RR

    shocks: dict[str, dict[str, float]] = {}
    for fy, lvl, rate in path:
        for d in UK_PRICE_LEVEL:
            if nominal_paths and d in UK_PROFIT_DETS:
                continue
            shocks.setdefault(d, {})[fy] = lvl
        for d in UK_PRICE_RATE:
            shocks.setdefault(d, {})[fy] = rate
    eff = RR.run(shocks)
    eff = eff[eff["block"].isin(["receipts", UK_SPEND_BLOCK])]
    out: dict[int, dict[str, float]] = {}
    for _, r in eff.iterrows():
        y = int(str(r["effect_fy"])[:4])
        key = "px:" + ("rec:" if r["block"] == "receipts" else "spd:") + UK_GROUP.get(
            r["line"], "uk_ind" if r["block"] == "receipts" else "uk_osp")
        out.setdefault(y, {})
        out[y][key] = out[y].get(key, 0.0) + float(r["gbp_bn"])
    return out


def _gg_interest(cc: str, y: int) -> float:
    a = _annual().loc[(cc, y)]
    v = a["gg_interest_bn"] if pd.notna(a["gg_interest_bn"]) else a["interest_bn"]
    return float(v)


UK_PROFIT_DETS = ("Non-oil, non-financial profits", "Financial profits")
US_NOMINAL_PATH = ("dep_corporate", "dep_nonwithheld")  # profits and equities: nominal paths in the growth leg


def _price(cc: str, years: list[int], y0: int, pgap, nominal_paths: bool = False) -> dict[int, dict[str, float]]:
    """Price leg for a price-level gap path ``pgap(k)`` (fraction), keys prefixed ``px:``.

    ``nominal_paths``: the growth leg already moves profits and equities on nominal paths, so the lines they drive
    (US corporate and non-withheld tax, UK profits, corporate tax elsewhere) are left out here."""
    out = {y: {} for y in years}
    if cc == "uk" and UK_SOURCE == "obr":
        from sovereign_cash.uk import ready_reckoner as RR
        last = max(int(f[:4]) for f in RR.YEARS)
        path = tuple((_uk_fy(y), round(100 * pgap(y - y0), 10), round(100 * (pgap(y - y0) - pgap(y - y0 - 1)), 10))
                     for y in range(y0 + 1, last + 1) if _uk_fy(y) in RR.YEARS)
        if any(abs(l) > 0 for _, l, _ in path):
            U = _uk_price(path, nominal_paths)
            for y in years:
                if y > y0:
                    out[y] = dict(U.get(min(y, last), {}))
                    p_lag = pgap(y - y0 - PRICE_LAG_YEARS) if y > UK_DEL_CASH_UNTIL else 0.0  # reset at the next review
                    if p_lag:
                        out[y]["px:spd:uk_del"] = ((_gg(cc, y)["spend"] - _gg_interest(cc, y)) * UK_DEL_SHARE
                                                   * PRICE_PASS_THROUGH * p_lag)
        return out
    for y in years:
        k = y - y0
        p, p_lag = pgap(k), pgap(k - PRICE_LAG_YEARS)
        if k <= 0 or (p == 0.0 and p_lag == 0.0):
            continue
        if cc == "us":
            B = _us_base()
            yy = y if y in B.index else int(B.index.max())
            d = {f"px:{c}": float(B.at[yy, c]) * p for c in B.columns
                 if c.startswith("dep_") and c not in ("dep_total", "dep_fed_earnings")
                 and not (nominal_paths and c in US_NOMINAL_PATH)}
            d.update({f"px:{c}": float(B.at[yy, c]) * PRICE_PASS_THROUGH * p_lag for c in B.columns
                      if c.startswith("wd_") and c not in ("wd_total", "wd_interest")})
        else:
            g = _gg(cc, y)
            rec = g["rev"]  # corporate tax is on the real gap here, so its price part stays in the price leg
            d = {"px:receipts": rec * p,
                 "px:spending": (g["spend"] - _gg_interest(cc, y)) * PRICE_PASS_THROUGH * p_lag}
        out[y] = d
    return out


def price_lines(cc: str, infl_pp: float, years: list[int], y0: int) -> dict[int, dict[str, float]]:
    """Inflation ``infl_pp`` a year above baseline for two years, real output unchanged: the common price leg."""
    if infl_pp == 0.0:
        return {y: {} for y in years}
    return _price(cc, years, y0, lambda k: _price_gap(k, infl_pp))


def lines(cc: str, size_pp: float, years: list[int], y0: int, *, stimulus: bool = False,
          cit_stress: float = 1.0) -> dict[int, dict[str, float]]:
    """Per year, each line's response (local bn). Receipt lines + = more receipts; spending lines + = more spending."""
    out = {}
    for y in years:
        k = _k(y, y0)
        if k <= 0 or size_pp == 0.0:
            out[y] = {}
            continue
        if cc == "us":
            d = _us(y, k, size_pp)
        elif cc == "uk" and UK_SOURCE == "obr":
            d = _uk(y, k, size_pp, y0)
        else:
            d = _oecd(cc, y, k, size_pp, cit_stress)
        if stimulus and size_pp < 0:
            g = _gg(cc, y)["gdp"] * (1 + gap(k, size_pp))
            d["stimulus"] = STIMULUS_PCT_GDP.get(k, 0.0) * abs(size_pp) / CANON_PP / 100 * g
        out[y] = d
    if size_pp:  # the price part of the nominal gap, on the common price leg
        P = _price(cc, years, y0, lambda k: (1 - REAL_SHARE) * gap(k, size_pp), nominal_paths=True)
        for y in years:
            for kk, v in P.get(y, {}).items():
                out[y][kk] = out[y].get(kk, 0.0) + v
    return out


def is_receipt(line: str) -> bool:
    ln = line[3:] if line.startswith("px:") else line
    return ln.startswith(("dep_", "rec:")) or ln in TAXES or ln == "receipts"


def response(cc: str, size_pp: float, years: list[int], y0: int, **kw) -> dict[int, dict[str, float]]:
    """Per year: d_rev (+ = more receipts) and d_spend (+ = more spending), local bn, general government."""
    L = lines(cc, size_pp, years, y0, **kw)
    return {y: {"d_rev": sum(v for k, v in d.items() if is_receipt(k)),
                "d_spend": sum(v for k, v in d.items() if not is_receipt(k))} for y, d in L.items()}


def balance_semi_elasticity(cc: str, y: int) -> float:
    """Model change in the GG balance ratio (pp) for a 1% real output gap, price level unchanged.

    The OECD comparator: Table 10 ``semi``. Includes the denominator (GDP) effect, as the OECD semi-elasticity does.
    """
    o, g = OECD[cc], _gg(cc, y)
    real = 0.01
    d_rev = sum(g["rev"] * (cit_share(cc) if t == "cit" else o[t][1]) / 100 * o[t][0] * real for t in TAXES)
    d_spend = g["spend"] * o["cp"][1] / 100 * cp_elasticity(cc) * real
    b0 = (g["rev"] - g["spend"]) / g["gdp"]
    b1 = (g["rev"] + d_rev - g["spend"] - d_spend) / (g["gdp"] * (1 + real))
    return 100 * (b1 - b0)


def table(size_pp: float = -CANON_PP, y0: int = 2026, years=(2027, 2028, 2029, 2030), **kw) -> pd.DataFrame:
    """Long table: cc, year, line, label, bn, pct_gdp (+ = adds to the deficit)."""
    rows = []
    for cc in ("us", "uk", "de", "fr", "it", "es", "jp"):
        L = lines(cc, size_pp, list(years), y0, **kw)
        for y, d in L.items():
            g = _gg(cc, y)["gdp"] * (1 + gap(_k(y, y0), size_pp))
            for ln, v in d.items():
                add = -v if is_receipt(ln) else v
                lab = (PRICE_LABEL if ln.startswith("px:") else US_LINE_LABEL.get(ln, ln) if cc == "us" else
                       UK_LABEL.get(ln.split(":")[-1], ln) if ln.startswith(("rec:", "spd:")) else TAX_LABEL.get(ln, ln))
                rows.append({"cc": cc, "year": y, "line": ln, "label": lab, "bn": add, "pct_gdp": 100 * add / g})
    return pd.DataFrame(rows)


# ================================================================== rates leg, on the same lines
# Every scenario leg moves the primary on the same lines: growth (cyclical + price leg), inflation (price leg) and
# rates (the non-interest receipts below). Interest itself stays on the ISIN book.
UK_RATE_DETS = ("Bank Rate", "Deposit rate")  # interest & dividend receipts, SA tax on savings income
BOJ_EXCESS_RESERVES_BN = 430_000.0  # yen bn. FY2025: ¥2.71tn interest on excess reserves at ~0.6% average (BoJ accounts,
                                    # 27 May 2026); the balance sheet is shrinking, so ~¥430tn from 2027
BOJ_REMIT_BASE_BN = 1_000.0         # yen bn a year: FY2025 paid ¥1.83tn; the June 2026 hike to 1.0% costs ~¥1tn a year
                                    # more on reserves. The payment cannot go below zero (losses stay at the BoJ)
BOJ_REMIT_UP_CAP_BN = 1_150.0       # a cut raises the payment at most back to its FY2024 level (net income ¥2.26tn): with
                                    # ~¥45tn of unrealised JGB losses the BoJ would rebuild provisions before paying more
UK_BILLS_BN = 100.0                 # Treasury bills: in the ISIN book, so netted from the reckoner's Bank Rate interest


@lru_cache(maxsize=16)
def _uk_rates(size_bp: float, y0: int, last: int) -> dict[int, dict[str, float]]:
    from sovereign_cash.uk import ready_reckoner as RR

    shocks = {d: {_uk_fy(y): size_bp / 100.0 for y in range(y0 + 1, last + 1) if _uk_fy(y) in RR.YEARS}
              for d in UK_RATE_DETS}
    eff = RR.run(shocks)
    out: dict[int, dict[str, float]] = {}
    for _, r in eff.iterrows():
        y = int(str(r["effect_fy"])[:4])
        if r["block"] == "debt_interest":  # Bank Rate on reserves (APF), NS&I and bills: interest, kept apart
            key = "int:uk_bank_rate"
        elif r["block"] in ("receipts", UK_SPEND_BLOCK):
            key = ("rec:" if r["block"] == "receipts" else "spd:") + UK_GROUP.get(
                r["line"], "uk_ind" if r["block"] == "receipts" else "uk_osp")
        else:
            continue
        out.setdefault(y, {})
        out[y][key] = out[y].get(key, 0.0) + float(r["gbp_bn"])
    return out


def rates_interest(cc: str, bp: float, years: list[int], y0: int) -> dict[int, float]:
    """Interest a rates shift moves outside the ISIN book (local bn). UK: the reckoner's Bank Rate debt-interest block
    (reserves remunerated at Bank Rate through the APF, NS&I) less the bills, which the book already reprices."""
    out = {y: 0.0 for y in years}
    if bp == 0.0 or cc != "uk" or UK_SOURCE != "obr":
        return out
    from sovereign_cash.uk import ready_reckoner as RR
    last = max(int(f[:4]) for f in RR.YEARS)
    R = _uk_rates(float(bp), y0, last)
    for y in years:
        if y > y0:
            out[y] = R.get(min(y, last), {}).get("int:uk_bank_rate", 0.0) - UK_BILLS_BN * bp / 1e4
    return out


def rates_lines(cc: str, bp: float, years: list[int], y0: int) -> dict[int, dict[str, float]]:
    """Non-interest receipts that a persistent parallel move in rates shifts (interest itself is the ISIN book).

    UK: OBR reckoner Bank Rate and deposit-rate lines (interest and dividend receipts, self-assessment tax on savings
    income, the Bank's reserves). Japan: the BoJ's payment to the Treasury, which still runs (¥1.83tn for FY2025):
    interest on excess reserves moves with the policy rate at once, the payment falls by that much, floored at zero,
    and is received the May after the fiscal year. US: none within the horizon: the Fed remits nothing while it carries
    a deferred asset (~$244bn, Q1 2026), so neither a rise nor a 75bp cut reaches the Treasury before 2030. Euro area:
    none (national central banks carry losses forward; no distributions within the horizon).
    """
    out = {y: {} for y in years}
    if bp == 0.0:
        return out
    if cc == "jp":
        d = min(BOJ_REMIT_UP_CAP_BN, max(-BOJ_REMIT_BASE_BN, -BOJ_EXCESS_RESERVES_BN * bp / 1e4))
        for y in years:
            if y >= y0 + 2:  # FY y0+1 (from April) paid in May y0+2
                out[y] = {"rec:boj": d}
    elif cc == "uk" and UK_SOURCE == "obr":
        from sovereign_cash.uk import ready_reckoner as RR
        last = max(int(f[:4]) for f in RR.YEARS)
        R = _uk_rates(float(bp), y0, last)
        for y in years:
            if y > y0:
                out[y] = {k: v for k, v in R.get(min(y, last), {}).items() if not k.startswith("int:")}
    return out


def clear_caches() -> None:
    """Clear every cache that holds a parameter-dependent result (call after changing module constants)."""
    from sovereign_cash.scenarios.channels import us_dts as U

    for f in (_annual, _uk_all, _uk_price, _uk_rates, U.fy_table):
        f.cache_clear()


def response_all(cc: str, years: list[int], y0: int, *, g_pp: float = 0.0, infl_pp: float = 0.0, r_bp: float = 0.0,
                 stimulus: bool = False) -> dict[int, dict[str, float]]:
    """d_rev / d_spend (local bn) from the growth, inflation and rates legs together (cross effects ignored)."""
    parts = [lines(cc, g_pp, years, y0, stimulus=stimulus), price_lines(cc, infl_pp, years, y0),
             rates_lines(cc, r_bp, years, y0)]
    out = {}
    for y in years:
        rec = sum(v for P in parts for k, v in P.get(y, {}).items() if is_receipt(k))
        spd = sum(v for P in parts for k, v in P.get(y, {}).items() if not is_receipt(k))
        out[y] = {"d_rev": rec, "d_spend": spd}
    return out
