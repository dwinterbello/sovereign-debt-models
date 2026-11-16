# ISIN+curve interest — scoring bugs that created false negatives

Re-scored 2026-10-27 after fixing the construction/scoring bugs below.
Engine: `isin_curve_interest_screen.py` → `sovereign_cash/isin_curve_interest.py`.
Live hooks: `INTEREST_ISIN_CURVE` in UK/DE/IT/FR `bottom_up_macro`.

## 1. Bugs (with evidence)

### B1. Monthly MAE share-spread both sides (scoring)

`monthly_spot` took each model's **annual** total and spread it with the same
calendar shares. That wiped the RR triangle's within-year shape and any
coupon/bill timing, then compared two flattened annuals.

Evidence (UK OOS, pre-fix): annual full edge vs RR **+8.1%**, share-spread
monthly edge **+2.0%** — the gate understated ISIN by ~6pp. Fair metric keeps
live on its native monthly path and only share-spreads the ISIN annual.

Code: `monthly_spot` / `_live_monthly_path` in the screen.

### B2. UK residual held negative through an uplift/cash timing gap (construction)

At end-2021, marketable = coupon £38.7 + model uplift £37.7 + bills ≈ £76 vs
JW2P £61.7 → residual **−£14.9bn**. Held flat into 2022, that subtracted £15bn
from the forecast just as cash JW2P jumped to £121 (full £64 → should have been
£79). Stock-Δ uplift as the residual's inflation rate was tried and rejected:
UK 2024 uplift stock **−£5.4bn** (redemptions, not negative interest) blew the
2025 residual to +£69 and full to £157.

Fix: UK-only `residual = max(I0 − mkt0, 0)`. Other countries keep the signed
residual (FR/DE ratio>1 is real marketable overstatement).

Evidence: UK 2022 full £64.40 → £79.30 after clip; OOS annual MAE 17.47 → 14.56.
Also wired into `debt/build.project` (UK branch only).

### B3. FR live pinned today's exécution as if known at Dec y−1 (scoring)

`authority.json` debt_charge 2021–24 are stamped *exécution*. Screen/live used
them whenever `origin ≥ Sep y−1`, so origin Dec-2020 "knew" 2021 exécution 38.5
and Dec-2021 "knew" 2022 exécution 51.5 — lookahead. Live OOS monthly MAE was
5.71 with that pin vs 8.67 on honest smly/ex-ante.

Fix: exécution known only when `origin.year > year`; LFI/PLF/update known from
Sep y−1. ISIN competes as **open-year fill** when AFT unknown, not against the
pin.

### B4. Prior locked face×coupon screen (already documented)

`locked_interest` with `bill_rate=0`, no uplift/FRN/bills/refinance → UK
locked/actual OOS ≈ 0.47. Not re-opened; full engine replaces it.

### B5. Not bugs (checked, left alone)

| Suspect | Finding |
|---|---|
| Trailing RPI vs debt-tab WEO for UK 2022 | PIT trailing at end-2021 is 7.5%; actual 2022 accrual ~12%. Information not available at origin — honest miss, not a scoring bug. |
| Curve asof / ops asof | Both cut at 31 Dec y−1 (`op_dt ≤ asof`, BoE/GSW/market ≤ asof). No mismatch found. |
| CY JW2P vs FY-shaped RR | Fair monthly uses native RR months; annual sums CY for both. |
| APF residual level jumps | UK 2023–24 residual +£42–45bn is real NS&I/perimeter; holding flat is the only PIT option without an NS&I model. |

## 2. Post-fix OOS table (fair cash leg)

Windows: PRE = target years 2015–19; OOS = 2020–25. Origin = 31 Dec y−1.
Fair monthly = live native path vs full share-spread annual.

| cc | ann full | ann live | ann edge | mth full | mth live | mth edge | live rule | INTEREST_ISIN_CURVE |
|----|----------|----------|----------|----------|----------|----------|-----------|---------------------|
| uk | 14.555 | 19.004 | **+23.4%** | 25.922 | 32.621 | **+20.5%** | rr_triangle | **True** |
| de | 3.560 | 3.953 | +9.9% | 4.384 | 4.849 | **+9.6%** | coupon_stock | **True** |
| it | 7.002 | 7.130 | +1.8% | 18.585 | 19.587 | **+5.1%** | smly | **True** |
| fr | 6.646 | 6.166 | −7.8% | 10.248 | 8.671 | −18.2% | aft_exante / smly_open | True as **open-year fill only** |
| us | 72.658 | 109.026 | +33.4% | — | — | — | smly (debt-tab) | n/a (already on ISIN book) |
| jp | 176.8 | 416.7 | **+57.6%** | — | — | — | smly (debt-tab) | n/a (already on ISIN book; no cash tab) |
| es | 2.150 | 2.633 | **+18.4%** | — | — | — | smly (debt-tab) | n/a (ops yes; curve was missing — auction par now on `GOV_CURVE`) |

FR vs pure smly (no AFT): monthly 10.248 vs 10.943 = **+6.4%** — so ISIN wins the
open year; it must not replace an honest AFT pin.

**Japan** was omitted from the first screen pass. Ops + curve exist; debt
`build.project` already runs the full ISIN+curve path. OOS 2020–25 annual MAE
¥177bn (full) vs ¥417bn (smly). Added to `NOTE_CCS`.

**Spain** had ops + annual interest but **no government curve** (`GOV_CURVE`
excluded `es`; `market_monthly` has no ES panel). Wired Tesoro auction-par
bootstrap into `gov_curve` / screen; OOS MAE €2.15bn vs smly €2.63bn.

## 3. Defaults

| Country | `INTEREST_ISIN_CURVE` | Rationale |
|---|---|---|
| UK | **True** | −20.5% fair monthly vs RR after residual clip |
| DE | **True** | −9.6% fair monthly vs coupon_stock |
| IT | **True** | −5.1% fair monthly vs smly (gate ≥5%) |
| FR | **True** (fill only) | beats smly open-year; never overrides AFT known at origin |
| US | n/a | debt-tab `build.project` already on ISIN+curve |
| JP | n/a | no monthly cash framework; debt-tab already on ISIN+curve (−58% OOS vs smly) |
| ES | n/a | no monthly cash engine; auction-par curve added to `GOV_CURVE` (−18% OOS vs smly) |

SHIP headline gates are unchanged — flip the flag to False to revert the
interest leg without touching other SHIP knobs.

## 4. What changed in code

- `experiments/isin_curve_interest_screen.py` — residual≥0 (UK), AFT ex-ante filter, fair monthly
- `debt/build.py` — UK residual≥0 in `project`
- `isin_curve_interest.py` — production wrapper (unchanged API)
- `uk/de/it/fr/bottom_up_macro.py` — `INTEREST_ISIN_CURVE` hooks

Re-score: `PYTHONPATH=. .venv/bin/python sovereign_cash/experiments/isin_curve_interest_screen.py`
