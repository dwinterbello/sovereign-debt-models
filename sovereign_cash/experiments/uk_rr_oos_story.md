# UK ready-reckoner × drivers — does it win out of sample?

**Verdict: yes.** Applying published OBR ready-reckoner elasticities to live AWE, retail, CPI, GVA, equity, Bank Rate and gilt moves beats the production remit×profile hybrid on every OOS horizon. Soft pin α=0.12 (SHIP) helps a little versus α=0; it does not create the win.

Scored 2026-10-27 off `uk/bottom_up_macro.py` (`SHIP=True`, `soft_pin_alpha=0.12`). PRE = origins 2012–2019 (Covid-free); OOS = origins from 2022-01. Numbers companion: `uk_rr_oos_numbers.json`. Gate remains PRE-only; OOS is report-only.

## Headline cumulative CGNCR MAE (£bn)

| Path | h=1 | h=3 | h=6 | h=12 | mean |
|---|---|---|---|---|---|
| Production hybrid (PRE) | 4.05 | 7.21 | 12.02 | 13.71 | **9.25** |
| RR×drivers α=0.12 (PRE) | 3.55 | 6.37 | 9.61 | 13.29 | **8.21** (−11.2%) |
| RR×drivers α=0 (PRE) | 3.56 | 6.30 | 9.77 | 14.12 | 8.44 (−8.7%; worse at h=12) |
| Production hybrid (OOS) | 7.24 | 14.24 | 21.98 | 32.10 | **18.89** |
| RR×drivers α=0.12 (OOS) | 5.37 | 10.70 | 13.76 | 15.53 | **11.34** (−40.0%) |
| RR×drivers α=0 (OOS) | 5.54 | 11.32 | 14.62 | 14.75 | 11.56 (−38.8%) |

OOS edge is larger than PRE edge. The hybrid’s 12-month OOS MAE (~£32bn) is more than double the RR path (~£15–16bn). Mean signed OOS bias on the SHIP path is small (−£1.6bn at h=12); the error is not a one-way level miss.

## Soft pin α=0.12 vs α=0

α=0.12 wins PRE (8.21 vs 8.44; no worse horizon vs worse at h=12) and wins OOS mean (11.34 vs 11.56). The pin helps h=1/3/6 OOS and costs a little at h=12 (15.53 vs 14.75). Story: a light pull toward the remit vintage known at origin cleans short-horizon residual noise without rewriting the RR tax/spend engine. α≥0.20 was previously rejected because OOS regresses — the remit is useful as a soft anchor, not as the level.

## What the category errors say

OOS monthly MAE (category level) vs PRE:

| Block | Lines that stay relatively tight OOS | Lines that blow up OOS |
|---|---|---|
| Tax RR | IT, NICs, CT (and VAT worse but still secondary on signed CGNCR) | — |
| Spend / DI | — | welfare, rdel proxy, interest, capital |
| Financing / perimeter | net lending, acq. of securities | gilt_adj, recv/payables, other_fin, APF transfers, resid_gap |

Ranked by mean absolute contribution to the 12-month CGNCR identity error on OOS, the worst lines are **rdel_proxy**, **gilt_adj**, **interest**, **welfare**, **cap** — not PAYE/NICs/VAT/CT. The published RR tax coefficients on AWE/retail/GVA are doing real work; the OOS failure surface is expenditure proxies, debt interest, and the financing residual the ONS bridge does not pin to a determinant.

Signed OOS contributions (12m, mean): welfare and nontax tend to push CGNCR **up** in the forecast (over-forecast spend / under-forecast nontax receipts); interest, capital and rdel tend the other way. They partially cancel in the headline, which is why leaf MAE looks worse than the cumulative CGNCR MAE.

Year cut (h=12 OOS): production dies in 2023 (MAE ~£55bn, bias ~−£55bn — remit×seasonal massively under-calls the cash need). RR×drivers is wrong the same way but at ~£19bn MAE / −£15bn bias. 2024–25 the gap narrows; 2026 both paths are high and same-signed. The OOS win is mostly “do not trust the remit shape when the fiscal regime has moved,” not “every category is solved.”

## Plain-language story

UK monthly cash need is not a pure seasonal object. When you take OBR’s own published elasticities and feed them the determinants the desk already watches — wages, retail, prices, activity, equities, Bank Rate, gilts — the tax engine tracks outturn well enough that the bottom-up path beats a remit-pinned seasonal hybrid by about **40%** on post-2021 origins, after already clearing the pre-2020 PRE gate by ~11%.

What still fails is the rest of the state: departmental spend proxies, debt interest under rate shocks, gilt and APF financing lines, and an unmapped residual. Those lines dominate OOS category error. Soft-pinning a tenth of the residual to the latest remit vintage helps a little and is the locked SHIP setting; turning the pin off barely moves OOS and re-opens a PRE miss at 12 months.

So the honest reading for UK cash forecasting: **determinants + RR are the right core for receipts; expenditure and financing remain a residual problem; the remit is a soft prior, not the model.** Absolute 12-month OOS MAE is still mid-teens £bn — useful relative to the hybrid, not a precision cash desk.

## What this note does not change

- `SHIP` stays True; `soft_pin_alpha` stays 0.12.
- No commit. Supporting re-score: `uk_rr_oos_numbers.json`.
- Cross-check: `experiments/bottom_up_baseline/STATUS_UK.md` (locked 2026-10-20e; same qualitative verdict; means here refresh with current ONS/remit cache).
