# STATUS_SE — bottom-up baseline v2

**Verdict: NO-SHIP**

Commit base: `0b17ae37`. Gate: RG-anchored PRE mean MAE ≥5% better than production (`seas_k5_a0.5_mean` + uniform), worse at ≤1 horizon. Threshold ≈ **9.54**.

Live cash unchanged: Riksgälden annual / rest-of-year × production seasonal.

## Headline

| Path | PRE anchored mean | OOS anchored mean | PRE unanchored | OOS unanchored |
|---|---|---|---|---|
| Production | 10.045 | 12.930 | 11.277 | 14.572 |
| v1 CH1089 seasonals | 10.567 | 15.635 | 10.104 | 14.783 |
| v2 prior (SCB smly×macro + stock×rate int) | 16.794 | 13.088 | 19.159 | 18.121 |
| **v2 now (CAT_SEAS × macro annual)** | **10.349** | **14.658** | **11.327** | — |

Ship gap: 10.349 / 10.045 = −3.0% relative (need −5%). Best forbidden blend (prod + 0.2·(v1−prod)): **9.882** (−1.6%) — still fails gate and is a seasonal rebrand.

## Category audit (v2 now)

| Category | Series | Driver / rule | PRE MAE path | PRE MAE smly |
|---|---|---|---|---|
| tax_rev | SCB BudgetStatForvM | ngdp+wages; CAT_SEAS smly × annual rescale; Budgetprop/KI when known | 6.503 | 5.144 |
| primary_spend | SCB exp − int − lend | cpi; CAT_SEAS seas_k3_a0.5_mean × annual | 4.534 | 3.948 |
| net_lending | CH1089 / Budgetprop | None (discrete); seas_k5_a0_median / auth annual | 6.933 | 8.829 |
| interest | CH1089 Räntor | stock×rate tried (PRE ann MAE 14.1); **kept** seas_k3_a0.5_mean (1.645) | 14.059 (stock×rate) | 11.319 |

Identity resid (SCB months): 0.076 bn.

CAT_SEAS locked on category PRE only (smly / k∈{3,5} × a∈{0,0.5} × mean|median).

## Authority tables

| Source | Vintage | Path |
|---|---|---|
| Budgetpropositionen / Vårpropositionen 2026 | 2026-04-15 | `data/se/authority.json#budgetprop` |
| Konjunkturinstitutet F3328 / F3304 | Sept 2026 (`2026-09-16`) | `data/se/authority.json#ki` |
| Statskontoret budgetprognos | Juni 2026 | `data/se/authority.json#statskontoret` |
| SCB BudgetStatForvM | cached parquet | `data/se/scb_budget_monthly.parquet` |
| Riksgälden CH1089 | monthly | `data/se/nbr_monthly.csv` |
| Riksgälden Statsupplåning | 2026:1 | RG pin after category sum only |
| Eurostat NGDP / HICP / D1 wages | — | macro drivers |

KI 2026 budgetsaldo −140 bn vs Budgetprop −184 vs Riksgälden −196 vs Statskontoret −209. Authority annuals do **not** move the RG-anchored gate (shape-only under a common rest-of-year total).

## Rejects (overnight)

- v1 per-component seasonals under RG pin (anchored PRE 10.567)
- SCB smly×macro + interest stock×rate (anchored PRE ~16.8)
- Interest stock×rate annual → shares or seasonal renormalised (11.1–13.2)
- Tax/spend calendar shares × last annual (10.076)
- prod↔v1 blend α∈[0.05,0.50]: best 9.882 — not ≥5%; not category×macro
- Lag-1 NGDP vs SCB revenue YoY corr ~0.18 (contemporaneous ~0.62 = lookahead)

## Gaps (why NO-SHIP)

1. **Gate is shape under RG pin.** Category macro annuals and KI/Budgetprop levels are washed out; only within-year allocation vs `seas_k5_a0.5_mean` scores.
2. **No published ready-reckoner.** Bill / KI give annuals, not tax-by-tax elasticities or month profiles conditional on macro.
3. **Weak lag-1 pass-through.** Honest no-lookahead growth is near zero on PRE; forcing it raises category MAE.
4. **Interest stock×rate loses.** Coupon calendar is better captured by a short seasonal than by debt×eff rate at monthly resolution.
5. **Residual / perimeter.** SCB vs CH1089 identity 76 mn max; net lending remains discrete.

## Paths

- `sovereign_cash/se/bottom_up_macro.py`
- `sovereign_cash/se/bottom_up.py` (v1, SHIP=False)
- `sovereign_cash/data/se/authority.json` (budgetprop + ki + statskontoret)
- `sovereign_cash/derived/se_bottomup_macro_score.json`
- `sovereign_cash/derived/se_bottomup_score.json`
- `sovereign_cash/tests/test_se_bottom_up_macro.py`
- `sovereign_cash/experiments/bottom_up_baseline/STATUS_SE.md`

No commit. No publish. Other countries untouched.
