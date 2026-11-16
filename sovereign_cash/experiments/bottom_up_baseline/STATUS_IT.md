# STATUS_IT — bottom-up baseline v2

**Verdict: SHIP**

Commit base: `0b17ae37`. Gate: PRE mean MAE ≥5% better than plan×smly production, ≤1 worse horizon; OOS ≤+25% vs production (PROMPT_IT). Live cash = macro_v2 categories + residual smly; BOT stock held → liquidity band (`model.hold_bills`).

## Headline

| Path | PRE mean | OOS mean | PRE edge vs prod | OOS vs prod |
|---|---|---|---|---|
| Production (plan×smly) | 5.518 | 11.270 | — | — |
| v1 BdI-line smly + residual | ~tie | fail | ~0.6% | rejected |
| **v2 macro (SHIP)** | **5.166** | **10.298** | **+6.4%** | **−8.6%** |

Ship threshold ≈ **5.242**. Worse horizons: 1/4 (h=1 only). Spec locked 2026-10-20e: β_tax=0.55, β_pri=0.35, treas×0.94, cap×NGDP 0.2, interest=smly, NGDP clip ±2.49pp PRE-support, `DFP_SHARE=False`.

### PRE / OOS by horizon (EUR bn MAE)

| h | PRE bu | PRE prod | OOS bu | OOS prod |
|---|---|---|---|---|
| 1 | 3.115 | 3.111 | 5.231 | 5.260 |
| 3 | 4.981 | 5.200 | 9.047 | 9.191 |
| 6 | 6.265 | 6.984 | 11.927 | 12.358 |
| 12 | 6.301 | 6.776 | 14.987 | 18.272 |

## Category audit

| Category | Series | Driver / rule | PRE MAE (12m sum, bn) |
|---|---|---|---|
| tax_rev | BdI T1 | smly × (1 + 0.55 × clip(NGDP YoY_{t−1})) | 11.66 |
| primary | BdI T1 | smly × (1 + 0.35 × HICP YoY_{t−1}) | 23.78 |
| interest | BdI carve | smly (sx / stock×rate rejected) | 9.08 |
| other_rev | BdI T1 | smly — no single DFP determinant | 2.86 |
| cap_exp | BdI T1 | smly × (1 + 0.2 × clip(NGDP YoY)) | 9.62 |
| treasury_bal | BdI T1 | smly × 0.94 | 32.15 |
| residual | MEF − BdI identity | smly × 1.0 | ~3.7 OOS 12m |

Identity resid share |r|/|cash| ~0.12 (2015–19). Residual not the binding OOS gap (treasury / cap / tax dominate).

## Residual screens (overnight finish; PRE still clear)

| Spec | PRE mean | edge | worse | OOS mean | Gate |
|---|---|---|---|---|---|
| **smly × 1.0 (locked)** | **5.166** | **+6.4%** | **1** | **10.298** | **CLEAR** |
| smly × 0.9 | 5.245 | +4.9% | 0 | 10.402 | fail PRE |
| smly × 0.8 / 0.75 / 0.5 | worse | <5% | — | — | fail |
| residual = 0 | 6.221 | −12.7% | 4 | 12.661 | fail |
| trail-12 mean | 5.300 | +3.9% | 2 | 10.024 | fail PRE |
| smly − trail mean | 6.122 | −10.9% | 4 | 12.939 | fail |

No residual change. Among clearers locked SHIP wins.

## Authority tables

- `data/it/authority.json` — vintages DEF_2021 / DEF_2024 / DFP_2025 / DFP_2026 (point-in-time)
- `data/it/plan/Tavole-DFP-2026-Sezione-II.xlsx`, `Tavole_DFP-2025.xlsx`
- `data/it/plan/DEF21-Sez-II.pdf`, `DEF24-Sez-II.pdf` (IV.2-2 hand-entered)
- `data/it/bdi_t1.parquet` — BdI Tavola 1 month shares / levels
- Macro: Eurostat NGDP/HICP via `yoy_components.macro`; DFP NGDP in authority
- MEF provisional fabbisogno — score target + residual
- `data/it/tax/Base_dati_entrate_cassa_agosto2026.xlsx` — IRPEF/IRES/IVA erarial (≠ BdI; not wired)
- RRF grant share — forward subtract in `run.forward`
- BOT → liquidity: `model.hold_bills` / `bot_seasonal_net` (band P10–P90)

## Rejects (selected)

- v1 component smly + residual — PRE ~0% edge
- DFP all-years share allocation — OOS 50.8; PRE fire 0/50 (COVID excl)
- tax split dirette×NGDP + indirette×CPI — PRE &lt;5%
- interest stock×rate OLS — PRE −11%; AR clears but OOS worse than SHIP
- Finanze erarial as BdI tax levels — perimeter +4%
- residual scale / zero / trail / debias — fail PRE or dominated (this overnight)

## Paths

- `sovereign_cash/it/bottom_up_macro.py` (`SHIP=True`)
- `sovereign_cash/it/bottom_up.py` (v1 comparison)
- `sovereign_cash/it/model.py` / `run.py` — BOT residual to liquidity band
- `sovereign_cash/data/it/authority.json`
- `sovereign_cash/tests/test_it_bottom_up_macro.py`
- `sovereign_cash/experiments/bottom_up_baseline/STATUS_IT.md`

No commit. No publish. Other countries untouched.
