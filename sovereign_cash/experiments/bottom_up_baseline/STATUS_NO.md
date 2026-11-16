# STATUS_NO — bottom-up baseline v2

**Verdict: SHIP**

Commit base: `0b17ae37`. Gate: beyond-NB PRE mean MAE ≥5% better than production smly, worse at ≤1 horizon. Live beyond-NB cash_requirement = Tab4×macro; inside NB path = official liquidity residual (unchanged).

## Headline (beyond-NB)

| Path | PRE mean | OOS mean | PRE worse horizons |
|---|---|---|---|
| Production smly | 6.73 | 14.97 | — |
| v1 Tab4 rules | 6.67 | 10.40 | 2 (h3/h6) → **NO-SHIP** |
| v2 macro w=1 mainland resid | 6.35 | 14.58 | 1 (h1) → SHIP ~5.7% |
| **v2 macro w=0.5 (shipped)** | **5.95** | **14.55** | **0** → **SHIP 11.6%** |

Ship threshold ≈ **6.39**. Beyond-NB PRE targets = six Januarys 2014–19 (NB path ends 31 Dec). n=6 per horizon.

## Category audit (shipped)

| Category | Driver | Rule | PRE MAE | PRE smly |
|---|---|---|---|---|
| oil_tax | brent_yoy | unit_mult | 2.74 | 3.92 |
| sdfi | brent_yoy | unit_mult (a priori) | 1.58 | 0.74 |
| vat | gdp_yoy | add_expanding ×0.5 | 4.60 | 4.78 |
| tax_net | gdp_yoy | add_expanding ×0.5 | 2.58 | 2.93 |
| nav | gdp_yoy | add_expanding ×0.5 | 1.41 | 1.88 |
| health | gdp_yoy | add_expanding ×0.5 | 6.87 | 6.94 |
| local_authorities | gdp_yoy | add_expanding ×0.5 | 0.39 | 0.53 |
| customs_excise | gdp_yoy | smly | 2.59 | 2.59 |
| other_net | pol_d12 | smly | 5.27 | 5.27 |

Identity max |Σ Tab4 − gov_accounts| < 0.01 bn. Mainland lines PRE-gated on category MAE; petroleum unit_mult a priori (gating sdfi→smly kills headline PRE −6.7%). `MAINLAND_RESID_WEIGHT=0.5` a priori half-pass-through (not PRE-max; w=0.25 better on thin January set).

## Authority / liquidity anchor

- NB LIQUIDITY_STATISTICS Tab4 → Tab1 gov_accounts (mainland + petroleum)
- NB LIQUIDITY_FORECAST FCAST_S — pin inside published horizon
- SSB 09190 mainland GDP (`data/no/mainland_gdp.parquet`)
- Brent USD (FRED / correlations panel) on oil_tax + sdfi
- NB policy rate screened for other_net — not selected
- MoF non-oil deficit annuals: empty placeholder in `data/no/authority.json`
- NGB / T-bill / GPFG FX financing unchanged

## Rejects

- v1 Tab4 smly/cal5/pace_dev/petro_yoy (edge <5%)
- Uniform mainland unit/half on all tax lines; annual GDP × month shares
- Brent lag-6 OLS oil-only; fitted Brent β via pct_change
- Category-gate sdfi to smly; oil=smly+sdfi unit; petro-only dropping mainland×macro
- Beating NB inside its vintage window

## Paths

- `sovereign_cash/no/bottom_up_macro.py`
- `sovereign_cash/no/bottom_up.py` (v1, NO-SHIP)
- `sovereign_cash/no/build.py` / `dashboard.py`
- `sovereign_cash/data/no/authority.json`
- `sovereign_cash/data/no/mainland_gdp.parquet`
- `sovereign_cash/derived/no_bottomup_macro_score.json`
- `sovereign_cash/derived/no_cash_monthly.parquet` / `no_cash_summary.json`
- `sovereign_cash/tests/test_no_bottom_up_macro.py`
- `sovereign_cash/tests/test_no_bottom_up.py`

No commit. No publish.
