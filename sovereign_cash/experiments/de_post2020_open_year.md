# Germany open-budget-year rule — post-2020 re-evaluation

**Verdict: keep `OPEN_MODE='soll'`. Do not switch the live cash engine to smly×macro for the open year.**

Question asked: after the debt-brake / Sondervermögen regime, is pinning the open budget year to ministry Soll still right, or should the open year use the same later-year smly×macro rule (revenue × NGDP, primary × sozial-wage, interest coupon×stock), with Soll kept only as a plan comparison line?

Answer from walk-forward MAE: Soll is *more* valuable post-2020, not less. Dropping it roughly doubles OOS error. PRE failure of `none` is real but not the reason to keep Soll — post-2020 OOS alone rejects the switch.

## Specs scored

All share the same later-year framework. Only the open calendar year (months with `m.year == origin.year`) differs:

| `open_mode` | Open-year level |
|---|---|
| **`soll`** (shipped) | Bundeshaushalt Soll × asymmetric hist/YTD execution (spend 80/20, revenue 95/5) |
| **`none`** | Same smly×macro path as later years (Soll unused by the cash engine) |
| **`hist_year`** | Last complete calendar Ist × hist/YTD blend (no ministry Soll) |

Code: `sovereign_cash/de/bottom_up_macro.py` (`forecast_core(..., open_mode=)`). Windows match production: PRE = forecast end ≤ Dec-2019; OOS = origin ≥ Jan-2022 (2020–21 mid excluded). Horizons 1/3/6/12; mean = average of the four horizon MAEs (EUR bn cumulative core cash requirement).

## Headline

| Mode | PRE mean | OOS mean | OOS vs soll |
|---|---|---|---|
| **soll** | **6.20** | **7.37** | — |
| none | 8.55 (+38%) | 16.92 | **−130%** (worse) |
| hist_year | 5.83 (−6%) | 24.19 | **−228%** (worse) |
| v1 bottom_up | 6.61 | 10.01 | — |

OOS by horizon (soll / none / hist_year):

| h | n | soll | none | hist_year |
|---|---|---|---|---|
| 1 | 42 | 4.16 | 6.14 | 8.93 |
| 3 | 40 | 6.50 | 12.22 | 19.29 |
| 6 | 37 | 7.00 | 19.69 | 29.43 |
| 12 | 31 | 11.80 | 29.63 | 39.10 |

PRE by horizon:

| h | n | soll | none | hist_year |
|---|---|---|---|---|
| 1 | 30 | 2.75 | 2.83 | 2.25 |
| 3 | 28 | 3.99 | 5.82 | 3.60 |
| 6 | 25 | 6.00 | 10.09 | 6.14 |
| 12 | 19 | 12.04 | 15.48 | 11.36 |

Same ranking if the OOS cut is “target month end ≥ 2022” instead of origin ≥ 2022 (soll 11.0 / none 16.8 / hist_year 23.1).

## Why Soll wins after 2020

Later-year paths are identical under `soll` and `none` (by construction). The entire OOS gap is open-year months.

At h=12 OOS (n=44 origins with a full year of outcomes):

| | soll | none |
|---|---|---|
| Open-year MAE | 19.1 | 32.4 |
| Later-year MAE | 13.3 | 13.3 |
| Open-year signed bias (fc − act) | **+0.8** | **+14.7** |

`none` systematically over-forecasts the cash requirement in the remaining open year (~€15bn mean over-forecast). Soll’s execution haircut on the printed Haushalt removes almost all of that bias. Example origin 2023-06 (Jul–Dec open): actual sum 17.1, Soll path 21.8, smly×macro 51.5.

So the ministry plan is not a pre-2020 artefact that the debt-brake made obsolete for *core* cash. Core Soll still anchors the open year; special funds stay on `plan.json` separately. The score-table pin `planhc_3` remains a non-PRE-gated residual — not the live core.

`hist_year` (own last Ist, no Soll) clears PRE slightly better than Soll but collapses OOS — last year’s outturn is a worse open-year level than the printed Soll once deficits and execution regimes jump.

## Switch gate (from the task)

Change the live default only if post-2020 OOS mean MAE improves ≥5% vs `soll` and PRE is not catastrophic.

| Candidate | OOS win vs soll | PRE cost | Switch? |
|---|---|---|---|
| `none` | −130% | +38% | **No** |
| `hist_year` | −228% | −6% (better PRE) | **No** |

Recommended live default: **`OPEN_MODE='soll'` unchanged**. Soll stays the cash-engine open-year pin; keep using ministry Soll as the plan comparison line as well (same source). No `POST2020_OPEN_MODE` override needed — post-2020 prefers the same rule as PRE.

## Role of Soll going forward

- **Cash engine (open year):** Soll × hist/YTD blend — shipped.
- **Later years:** smly × macro (unchanged).
- **Plan comparison / board:** Soll annuals in BMF MB / `plan.json` core NKA — display and residual `planhc_3` only; do not replace the open-year engine with bare smly×macro.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from sovereign_cash.de import bottom_up as V1, bottom_up_macro as BU, common as C
import numpy as np, pandas as pd
exp, rev, cash, mb, maps = V1._flows()
drivers, raw = BU.macro_drivers(), BU.load_categories()
# score forecast_core(..., open_mode=m) for m in soll|none|hist_year
# window: PRE if end<=C.PRE_END else OOS if origin>=C.OOS_START
PY
```

## Paths

- `sovereign_cash/de/bottom_up_macro.py` — `OPEN_MODE`, `forecast_core`
- `sovereign_cash/experiments/bottom_up_baseline/STATUS_DE.md` — SHIP baseline (Soll open)
- `sovereign_cash/experiments/de_post2020_open_year.md` — this note

No commit. Live default unchanged.
