# STATUS_DE — bottom-up baseline v2

**Verdict: SHIP**

Commit base: `0b17ae37`. Gate: PRE mean MAE ≥5% better than live/v1 (`bottom_up` Soll×hist then smly), worse at ≤1 horizon.

`planhc_3` remains the residual score-table pin — **not PRE-gated** (17 h=12 PRE windows < `MIN_PRE`=36; loses to `prevyear_3` on PRE; shipped a priori for debt-brake / Sondervermögen ramp). Live core cash is bottom-up macro when this SHIP clears.

## Headline

| Path | PRE mean | OOS mean | PRE edge vs v1 |
|---|---|---|---|
| Live/v1 (production gate) | 6.61 | 10.01 | — |
| v2 macro (shipped) | **6.20** | **7.37** | **+6.2%** |
| planhc_3 (not PRE-gated) | ~12.8 | ~20.0 | loses PRE |

Ship threshold ≈ **6.28**. Clears; worse at 0/4 PRE horizons.

### PRE / OOS by h (cumulative MAE, EUR bn)

| h | n PRE | macro | v1 | n OOS | macro | v1 |
|---|---|---|---|---|---|---|
| 1 | 30 | 2.75 | 2.90 | 42 | 4.16 | 4.48 |
| 3 | 28 | 3.99 | 4.22 | 40 | 6.50 | 6.99 |
| 6 | 25 | 6.00 | 6.37 | 37 | 7.00 | 8.62 |
| 12 | 19 | 12.04 | 12.93 | 31 | 11.80 | 19.92 |

## Category audit (production path)

| Category | Series | Driver / rule | Role |
|---|---|---|---|
| Einnahmen (agg) | BMF MB Soll/Ist | open: Soll × (95% hist + 5% YTD); later smly × (1+1.0·NGDP) | shipped |
| Ausgaben primary | BMF MB Soll/Ist + T1 | open: Soll × (80% hist + 20% YTD); later smly × (1+w_sozial·0.35·wages) | shipped |
| Zins | debt-book + FA stock | annual[y−1] × month-share × FA stock YoY^Δy × (1+2.0·rates) | shipped |
| t1_sozial | BMF MB T1 Aufgabenbereiche | wages β=0.35 on later primary share (~0.54) | shipped overlay |
| Lohnsteuer / USt / KSt / … | BMF MB T3 Steuerarten | a priori NGDP β 1.0 / 1.0 / 1.2 | audit only (leaf pin = annual MAE no-op) |
| Sondervermögen | `plan.json` | annual plan pace (SVIK/SVBw); no PRE-clearing monthly driver | funds path unchanged |

Identity: `core ≈ primary + interest − revenue`. Leaf Steuerarten unscaled / renormalized paths fail PRE or are cumulative no-ops.

## Authority tables

- `data/de/bmf_mb/*.html` — BMF Monatsbericht T0 totals, T1 Aufgabenbereiche, T2 ökonomische Arten, T3 Steuerarten
- `data/de/plan.json` — Bundeshaushalt Soll core NKA + SVIK + SVBw
- `data/de/authority.json` — elasticities, selection audit, production rules
- `data/de/schuldenbericht_dt.xlsx` — Finanzagentur debt stock (interest path)
- AMECO/Eurostat NGDP, HICP; Eurostat namq D1 wages; debt-book effective rates

## Rejects (recorded)

- Leaf Soll open-year (unscaled) — PRE ~4× worse
- Leaf month-shares on aggregate Soll — PRE edge 4.2% (<5%)
- Scaled leaf Steuerarten renormalized to Soll — cumulative MAE no-op
- Equal open YTD weights; primary later CPI/NGDP β>0 — OOS worse
- Interest smly×rates; BMF Zins leaf — dominated by coupon×FA-stock or PRE-fail
- Special-fund monthly drivers — SVIK/SVBw absent in PRE
- T1 free-level Σ / employment overlay / wages β=1 — PRE-ok subset; OOS worse than β=0.35

## planhc honesty

- Score / overview: `planhc_3` labelled **not PRE-gated**
- `model_reason` states thin PRE (17 < 36), loses every PRE horizon to `prevyear_3`, a priori debt-brake / special-fund pin
- Live core ≠ planhc when macro SHIP is True

## Paths

- `sovereign_cash/de/bottom_up_macro.py`
- `sovereign_cash/de/bottom_up.py` (v1 fallback)
- `sovereign_cash/de/model.py` / `run.py` (planhc not-PRE-gated labels)
- `sovereign_cash/data/de/authority.json`
- `sovereign_cash/data/de/plan.json`
- `sovereign_cash/data/de/de_bottomup_macro_score.json`
- `sovereign_cash/tests/test_de_bottom_up_macro.py`
- `sovereign_cash/experiments/bottom_up_baseline/STATUS_DE.md`

No commit. No publish. Other countries untouched.
