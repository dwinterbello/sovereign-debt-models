# STATUS_ES — IGAE category × macro

**Verdict: SHIP**

Gate: rest-of-year |err|/|year| at origins k=3/6/9 vs thin_cash `seas_add_3`.
PRE = fy ≤ 2019; OOS = fy ≥ 2022. Need PRE ≥5% better, OOS ≤+25%. Sample is
below DE/FR `MIN_PRE`=36 origin windows — this is the thin_cash halves gate.

| model | PRE | OOS | PRE edge |
|---|---:|---:|---:|
| seas_add_3 (fallback) | 0.159 | 0.320 | — |
| smly leaves (β=0) | 0.149 | 0.384 | +6.1% |
| **tax β=0.5 NGDP / spend β=0.35 HICP (SHIP)** | **0.141** | **0.361** | **+11.3%** |
| full pass-through β=1 | 0.151 | 0.364 | +4.6% |
| totals only NGDP/CPI | 0.147 | 0.335 | +7.3% |

Live: `thin_cash.build_frame("es")` → `es.bottom_up_macro` when `SHIP`.
Interest on Debt & supply: ISIN+curve (`GOV_CURVE`).

## Japan (same screen)

**Verdict: NO-SHIP.** MoF GA monthly from FY2024 only; category×NGDP mean ratio
~0.79 vs smly ~0.09 on available origins. Live stays `smly`.

Artifacts: `experiments/es_jp_bottom_up_screen.py`, `es_jp_bottom_up_STATUS.md`,
`es/bottom_up_macro.py`.
