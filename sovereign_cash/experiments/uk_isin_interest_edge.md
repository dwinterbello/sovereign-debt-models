# UK ISIN-book interest vs OBR RR triangle

**Verdict: no — ISIN interest does not help OOS headline.** Every replace/blend loses to SHIP on mean cumulative CGNCR MAE. Least-bad PRE-safe ISIN spec is `blend25` (OOS 12.37 vs SHIP 11.34, **+9.1%** worse; PRE 8.40 vs 8.21). Ship gate (≥5% OOS improvement, PRE not >5% worse) fails. Do not flip SHIP.

Scored walk-forward off `uk/bottom_up_macro.py` with `interest_mode` overlays (default SHIP remains `rr`). PRE = origins 2012–2019 (Covid-free targets); OOS = origins from 2022-01. Headline = cumulative CGNCR(ex) MAE £bn. Companion: `uk_isin_interest_numbers.json`.

## Why the ISIN book is not JW2P

Point-in-time marketable interest = face×coupon (linkers × index ratio) + linker stock×RPI YoY uplift + bill stock×Bank Rate. At 2026-09-30 that is coupon £72.8bn + uplift £27.5bn + bills £3.5bn = **£103.8bn**, against JW2P TTM **£128.4bn** (gap £24.6bn — NS&I / other PS interest / perimeter). Book marketable YoY correlates with JW2P YoY at **0.77**; gilt Δpp / Bank Rate Δpp (the RR drivers) at **0.75** / **0.64**. The book sees locked coupons and RPI accrual; the RR triangle still carries the rate-repricing residual JW2P pays that the locked stock does not.

## Headline cumulative CGNCR MAE (£bn)

| Spec | PRE h1 | h3 | h6 | h12 | PRE mean (Δ vs SHIP) | OOS h1 | h3 | h6 | h12 | OOS mean (Δ vs SHIP) |
|---|---|---|---|---|---|---|---|---|---|---|
| SHIP rr | 3.55 | 6.37 | 9.61 | 13.29 | **8.21** (+0.0%) | 5.37 | 10.70 | 13.76 | 15.53 | **11.34** (+0.0%) |
| isin_yoy | 3.62 | 6.75 | 10.39 | 15.83 | **9.15** (+11.4%) | 5.93 | 13.29 | 20.06 | 36.15 | **18.86** (+66.3%) |
| isin_share | 3.57 | 6.58 | 10.07 | 14.00 | **8.55** (+4.2%) | 5.71 | 11.82 | 15.63 | 17.62 | **12.69** (+11.9%) |
| isin_replace | 3.62 | 6.71 | 10.33 | 15.63 | **9.07** (+10.5%) | 5.82 | 12.79 | 18.16 | 27.50 | **16.07** (+41.7%) |
| blend25 | 3.57 | 6.45 | 9.78 | 13.82 | **8.40** (+2.4%) | 5.43 | 11.17 | 14.83 | 18.06 | **12.37** (+9.1%) |
| blend50 | 3.58 | 6.55 | 9.97 | 14.46 | **8.64** (+5.3%) | 5.53 | 11.84 | 16.08 | 22.15 | **13.90** (+22.6%) |
| blend75 | 3.60 | 6.64 | 10.18 | 15.11 | **8.88** (+8.2%) | 5.68 | 12.57 | 17.73 | 28.84 | **16.21** (+42.9%) |
| SHIP + redeem | 11.66 | 26.91 | 37.32 | 13.45 | **22.33** (+172%) | 20.98 | 48.20 | 60.64 | 16.65 | **36.62** (+223%) |
| blend50 + redeem | 11.68 | 26.93 | 37.37 | 14.43 | **22.60** (+175%) | 21.18 | 48.42 | 60.57 | 21.85 | **38.01** (+235%) |

Production hybrid (same run): PRE mean ~9.25, OOS mean ~18.9 — SHIP rr still the OOS winner among interest engines tested.

## Interest-category monthly MAE (£bn)

| Spec | PRE interest MAE | OOS interest MAE |
|---|---|---|
| SHIP rr | 0.578 | 3.011 |
| isin_yoy | 0.590 | 3.388 |
| isin_share | 0.588 | **2.521** |
| isin_replace | 0.586 | 2.966 |
| blend25 | 0.570 | 2.891 |
| blend50 | 0.569 | 2.923 |
| blend75 | 0.574 | 3.112 |

`isin_share` is the only leaf win: OOS interest MAE 2.52 vs SHIP 3.01 (−16%). Headline still +12% worse — same accurate-leaf / cancellation pattern as other UK cash leaves. Blends improve the leaf a little and still lose the headline as ISIN weight rises.

## Spec notes

- `rr` (SHIP): JW2P calendar month + OBR DI triangle (Bank Rate + gilt Δpp).
- `isin_yoy`: calendar × (1 + marketable interest YoY from the PIT book).
- `isin_share`: PIT marketable annual × JW2P month shares, scaled to TTM JW2P at origin.
- `isin_replace`: calendar + Δ marketable annual / 12 (no RR).
- `blend25/50/75`: weight on `isin_yoy` vs `rr`.
- `+ redeem`: resid_gap += (book coupon redemptions − their path mean). Financing pulse is tens of £bn in redemption months — destroys headline; not a cheap overlay.

## Ship gate

- Best OOS among PRE-safe (PRE ≤ SHIP×1.05): **blend25** — still **worse** OOS than SHIP.
- Need OOS mean ≤ SHIP×0.95 with PRE ≤ SHIP×1.05: **fails**.
- Flip SHIP? **no**.

## Plain reading

The RR×drivers OOS story said debt interest blows up as a *category*. Replacing the RR triangle with the live ISIN book does not fix that on the CGNCR headline: pure ISIN YoY is the worst OOS (+66%), and every blend that dilutes RR loses. The book *with* linker uplift and bills is a real marketable interest object (corr 0.77 with JW2P YoY) and can tighten the interest leaf under a share-scaled month path — but that leaf gain does not pass through to CGNCR. Redemption timing from the book is too large and wrongly signed as a residual add-on; financing needs a dedicated gilt_adj / sales identity, not a demeaned redeem dump.

So the missing UK RR OOS edge is **not** “use the ISIN coupon book for interest.” Interest stays on the RR triangle; residual financing / rdel / welfare remain the open surface.

## What this note does not change

- `SHIP` stays True; `interest_mode` stays `"rr"`; `isin_redeem_overlay` stays False.
- No commit. Re-score: `PYTHONPATH=. .venv/bin/python sovereign_cash/experiments/uk_isin_interest_edge.py`.
- Hook left in place (default off): `SPEC['interest_mode']` / `isin_redeem_overlay` in `uk/bottom_up_macro.py`; book math in `uk/isin_interest.py`.
