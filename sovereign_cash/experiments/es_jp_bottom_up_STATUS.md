# STATUS_ES / STATUS_JP — category×macro screen

## Both years (shipped cash countries)

UK / FR / DE / IT / NO (beyond NB): live bottom-up covers the **open budget year and later years** on one path (rules differ by year — e.g. DE Soll in the open year, smly×macro later; FR PLF pin when known). Interest ISIN+curve is annual×month shares on those same forward months.

## Spain

**Verdict: SHIP (live)** — `tax_ngdp_0.5_spend_cpi_0.35` PRE edge 11.3%; wired in `thin_cash`

Panel months: 139. Identity MAE vs published B.9 cash: 0.0000 bn.

| model | PRE | OOS | PRE edge |
|---|---:|---:|---:|
| baseline_seas_add_3 | 0.159 | 0.320 |  |
| smly_leaves | 0.149 | 0.384 | 6.1% |
| tax_ngdp_0.5_spend_cpi_0.35 | 0.141 | 0.361 | 11.3% |
| tax_ngdp_1_spend_cpi_1 | 0.151 | 0.364 | 4.6% |
| totals_ngdp_cpi | 0.147 | 0.335 | 7.3% |

Baseline = thin_cash `seas_add_3`. Gate = PRE ≥5% better, OOS ≤+25%.

## Japan

**Verdict: NO-SHIP** — NO-SHIP: MoF monthly starts FY2024 (n origins too thin for PRE gate); BU edge vs smly = -8.139380463105104 on available FY

Months on disk: 29. smly mean ratio=0.08633708596192198; BU=0.7890674766818156.

Live: ES wired to `es.bottom_up_macro` (`SHIP`) via `thin_cash.publish`; JP stays
`smly`. Debt interest remains ISIN+curve (`GOV_CURVE`).
