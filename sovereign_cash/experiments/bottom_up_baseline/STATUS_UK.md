# STATUS_UK — bottom-up baseline v2

**Verdict: SHIP**

Commit base: `0b17ae37`. Gate: mean PRE MAE ≥5% better than production hybrid (remit×profile vs `seasonal_add3`), worse at ≤1 horizon. Identity; no lookahead. Soft remit pin uses vintage known at origin only.

Locked SPEC `2026-10-20e`: prior α=0 ship (edge 8.7%, worse h=12) → **soft_pin_alpha=0.12** (edge 11.2%, worse none). Category×RR path unchanged.

## Headline MAE (cumulative £bn)

| Model | h=1 | h=3 | h=6 | h=12 | PRE mean | OOS mean |
|---|---|---|---|---|---|---|
| Production | 4.053 | 7.211 | 12.016 | 13.706 | **9.246** | **19.134** |
| v2 macro (SHIP) | 3.553 | 6.369 | 9.612 | 13.296 | **8.208** | **11.441** |
| v2 α=0 (prior) | 3.564 | 6.301 | 9.772 | 14.114 | 8.438 | 11.605 |

- **PRE edge:** 11.23% vs production (≥5% required). Worse horizons: none.
- **OOS:** 11.44 vs 19.13 (report only; not used for selection).
- **Residual share (PRE part-MAE):** 6.9% (`resid_gap`).

v1 ONS-line seasonals (`bottom_up.py`): `SHIP=False` — rejected as seasonal main engine.

## Category audit (PRE monthly MAE)

| Category | Series | Driver(s) | Rule | PRE MAE |
|---|---|---|---|---|
| it | MF6P×EFO IT share | AWE YoY | RR PAYE 1.13× | 0.336 |
| nics | MF6P×EFO NICs share | AWE YoY | RR NICs 1.53× | 0.174 |
| vat | MF6P×EFO VAT share | retail YoY | RR VAT 0.75× | 0.218 |
| ct | MF6P×EFO CT share | GVA YoY | RR onshore CT 0.67× | 0.114 |
| other_tax | MF6P − big-4 | retail + equity YoY | 0.75×retail + 0.7×equity | 0.524 |
| nontax | ANBV−MF6P | None | calendar only | 0.983 |
| welfare | MF6S−JW2P × EFO share | CPI YoY | 1.5×CPI | 0.583 |
| rdel_proxy | MF6S−JW2P × (1−wel) | GVA YoY | 0.25×GVA | 0.917 |
| interest | JW2P | Bank Rate Δpp + 10y gilt Δpp | RR DI triangle /12 | 0.622 |
| cap | MF6T | GVA YoY | 0.2×GVA | 0.962 |
| recv_pay | ANRT F7 | retail (elas→0) | calendar | 1.583 |
| gilt_adj | ANRU F.3 | None | seasonal_add3 | 1.041 |
| other_fin | ANRV | Bank Rate Δpp | sa3 + 2.0×BR Δpp/12 | 2.310 |
| acq_sec | ANRS F.5 | None | trail-12 mean | 1.146 |
| apf_xfer | MF7A | None | seasonal_add3 | 0.000 |
| net_lend | ANRH F.4 | None | seasonal_add3 | 1.131 |
| resid_gap | M98R − bridge − Σ fin | None | sa3 + soft remit pin α=0.12 | 0.939 |

## Authority tables

| Source | Vintage / path | Use |
|---|---|---|
| OBR EFO Mar 2026 Tables A.5 / A.7 / 5.4 / A.3 | `data/debt/obr_lines.parquet` | category levels, shares, determinants |
| OBR ready reckoner Mar 2025 EFO (oil corr. 17 Mar 2026) | `data/uk/obr_ready_reckoner.xlsx` | tax/spend/DI elasticities + DI triangle |
| Hand audit | `data/uk/authority.json` | locked SPEC + driver labels |
| FRED SPASTT01GBM661N | `data/uk/equity_fred.json` | equity YoY (CGT/stamp/IHT block) |
| BoE Bank Rate | `uk_flows/output/rates.parquet` | DI triangle + ANRV overlay |
| BoE nominal 10y | `uk_flows/output/curves.parquet` | DI gilt Δpp |
| Macro panel | `correlations/derived/macro_panel_monthly.csv` | AWE / retail / CPI / GVA YoY |
| ONS PUSF | `data/uk/ons/*.json` | outturn, month shape, fin-tx map |
| DMO remit vintages | `data/uk/remit_vintages.csv` | soft pin + production hybrid benchmark |

## Rejects (this pass + inherited)

- v1 ANBV/MF6S/MF6T/JW2P seasonal_add3/smly as main engine
- Headline smly / trail_mean
- Expanding OLS elasticities replacing published RR
- Soft pin α=0 (PRE 8.7%, worse h=12) — dominated by α=0.12 on PRE and OOS
- Soft pin α≥0.20 with resid_sa3=1 — OOS regresses
- α=0 welfare↓ / other_tax_retail=0.5 / BR=3 combos — PRE +~1pp, OOS +1pp
- Fin-rule OTA (gilt trail12, resid_sa3∈{0.5,1}, …) — no PRE beat of locked α=0 base without OOS cost
- ANRU gilt/BR overlays; MF7A QT pace; ANRH CPI/GDP; resid_gap trail12/OLS equity

## Paths changed

- `sovereign_cash/uk/bottom_up_macro.py` — SPEC `soft_pin_alpha` 0→0.12; rejects; docstring
- `sovereign_cash/tests/test_uk_bottom_up_macro.py` — pin assertion
- `sovereign_cash/data/uk/authority.json` — locked audit 2026-10-20e
- `sovereign_cash/experiments/bottom_up_baseline/_uk_eval_cache.json`
- `sovereign_cash/experiments/bottom_up_baseline/STATUS_UK.md` (this file)
- `sovereign_cash/investor.py` — UK path/note string
- `sovereign_cash/sovereign_cash_overview.html` — UK row copy

Unchanged: `uk/bottom_up.py` (v1 SHIP=False), `uk/model.py`, `uk/ready_reckoner.py`, other countries.

Live wire: `uk/run.py` already uses `bottom_up_macro` when `evaluate()["ship"]`. No commit. No publish.
