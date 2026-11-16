# UK monthly cash bottom-up — why it is NO-SHIP (autopsy)

Scored 2026-11-08 by `uk_cash_ship_autopsy.py` (companion `uk_cash_ship_autopsy.json`).
Engine `uk/bottom_up_macro.py` at working tree (`SHIP=True`, `FORCE_LIVE=False`,
`INTEREST_ISIN_CURVE=True`, α=0.12); production = `seasonal_add3` / remit hybrid
(`uk/bottom_up.production_forecast`). Windows are the module's own: PRE origins
2012-01..2019-12, OOS origins 2022-01→, Covid-touching rows dropped; score = cumulative
CGNCR(ex) MAE £bn at h = 1/3/6/12 and the four-horizon mean. Gate = `M.EDGE_BAND` = **0.03**
(the comments in `bottom_up_macro.py` saying "5%" are wrong) and ≤1 worse horizon.
No production code was changed.

## Verdict (10 lines)

1. The live summary is **not stale**: fresh `evaluate()` = 9.825 vs 8.993, edge −9.3%, worse at h=3 and h=12 — identical to `uk_cash_summary.json` (built 2026-11-08 01:25).
2. The Oct-27 "SHIP, PRE −11.2% / OOS −40%" spec (RR interest, no SA nudge) **reproduces exactly on today's data**: BU PRE 8.206, OOS 11.341.
3. Production got better, not worse: the Oct-29 `PLAN_SCALE_CHANNEL` change in `model.plan_profile` took its PRE mean 9.246 → 8.993. That alone cut the old spec's edge 11.2% → 8.7% (still a pass).
4. The SA equity nudge (`sa_equity_nudge=True`, commit 80f0b77c) costs BU PRE 8.206 → 8.899 and OOS 11.34 → 13.64; edge falls to **+1.0%** with h=12 worse → gate fails **before** any interest change.
5. The nudge is a biased leaf, not noise: Jan/Feb IT and other-tax errors go from −0.3/0.0 to −1.65/−1.39 £bn per month in PRE and −5.0/−4.5 in OOS (Jan-2026 other_tax −8.9 vs −1.8 on the plain carry). It treats trend growth in January receipts as a transient spike and shrinks it by 65%.
6. `isin_curve` interest adds a second hit: BU 8.899 → 9.825 PRE, 13.64 → 20.97 OOS. The interest leaf alone is worse than the RR triangle on PRE (12m cum MAE 4.98 vs 3.27) and OOS (28.8 vs 18.8), with OOS 12m bias +24.6 £bn (2023 origins +74).
7. The isin wrapper `isin_curve_interest.monthly_path` was never scored on monthly cash origins: `annual_bn(uk, Y)` is NaN before 2016 (degenerate fallback), is a 31-Dec-(Y−1) forecast (lookahead for Jan–Nov origins, 11-month stale for the open year), and its `last_cy/fc_cy` scale carries last year's forecast error multiplicatively. The "−20.5% fair cash leg" in `isin_interest_bugfix.md` was annual Dec-31 origins share-spread — a different object.
8. Soft pin α=0.12 is small, real, and honest: +2.5pp PRE edge on the Oct-27 spec (8.438 → 8.206, clears h=12), +0.3pp with the nudge on, OOS better at h=1–6 under every engine. It uses the same remit vintages production already uses at long horizons. No legislated-step overlay exists in the UK BU backtest.
9. Answer: **(D)** — two post-gate code changes (SA nudge, isin monthly wrapper) degraded the RR path, and a production fix ate a quarter of the old edge. Not (A): production improved. Not (B): RR×drivers still beats production by 8.7% PRE / 40% OOS as scored Oct-27. (C) is half-true: isin_curve broke it further, but the gate was already failed by the nudge.
10. Restoring a passing gate requires reverting both the nudge and `INTEREST_ISIN_CURVE` (or scoring a replacement for each on PRE); flipping only `INTEREST_ISIN_CURVE=False` gives +1.0% edge and still fails.

## 1. Gate table (fresh re-score, same origins for every row)

PRE n = 96/95/92/86 origins at h = 1/3/6/12; OOS n = 54/52/49/43.

| Path | PRE h1 | h3 | h6 | h12 | **mean** | edge | worse | gate | OOS h1 | h3 | h6 | h12 | **mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Production hybrid (`seasonal_add3` / remit) | 4.051 | 6.853 | 11.679 | 13.389 | **8.993** | — | — | — | 5.736 | 12.812 | 20.765 | 32.917 | **18.057** |
| (a) SPEC: isin_curve, α=0.12, SA nudge | 3.685 | 6.999 | 11.162 | 17.453 | **9.825** | **−9.3%** | 3, 12 | FAIL | 6.312 | 13.856 | 21.637 | 42.073 | 20.969 |
| (b) rr, α=0.12, SA nudge | 3.632 | 6.616 | 10.218 | 15.129 | 8.899 | +1.0% | 12 | FAIL (<3%) | 5.872 | 12.450 | 15.874 | 20.347 | 13.636 |
| isin_curve, α=0 | 3.696 | 7.138 | 11.411 | 17.699 | 9.986 | −11.0% | 3, 12 | FAIL | 6.715 | 14.911 | 24.178 | 44.408 | 22.553 |
| rr, α=0 | 3.646 | 6.641 | 10.312 | 15.124 | 8.931 | +0.7% | 12 | FAIL | 6.102 | 13.262 | 17.572 | 20.186 | 14.281 |
| (e) Oct-27 spec: rr, α=0.12, no SA nudge | 3.553 | 6.368 | 9.612 | 13.292 | **8.206** | **+8.7%** | — | PASS | 5.371 | 10.704 | 13.762 | 15.526 | **11.341** |
| rr, α=0, no SA nudge | 3.564 | 6.300 | 9.771 | 14.117 | 8.438 | +6.2% | 12 | PASS | 5.542 | 11.322 | 14.618 | 14.748 | 11.557 |
| Production with pre-Oct-29 `plan_profile` (control) | 4.053 | 7.211 | 12.016 | 13.706 | 9.246 | — | — | — | — | — | — | — | — |

Row (e) matches `uk_rr_oos_numbers.json` to the third decimal (8.206 / 11.341). The production
control row matches that file's 9.246. The whole move from "SHIP −11.2%" to "NO-SHIP −9.3%" is
therefore three identifiable changes, not data drift:

| Step | BU PRE mean | prod PRE mean | edge | worse |
|---|---|---|---|---|
| Oct-27 story (rr, no nudge, old plan_profile) | 8.206 | 9.246 | +11.2% | — |
| + `PLAN_SCALE_CHANNEL` in production (Oct-29) | 8.206 | 8.993 | +8.7% | — |
| + SA equity nudge (Oct-29) | 8.899 | 8.993 | +1.0% | 12 |
| + `INTEREST_ISIN_CURVE=True` (Oct-29) = live SPEC | 9.825 | 8.993 | −9.3% | 3, 12 |

Signed 12m bias (BU − actual, £bn): Oct-27 spec PRE +6.1 / OOS −1.6; with nudge PRE +11.0 /
OOS +12.8; with nudge + isin PRE +11.9 / OOS +36.9. Each change adds a one-way error.

Paired per-origin, isin minus rr |err| on PRE: h1 +0.05 (isin worse on 55% of origins),
h3 +0.38 (65%), h6 +0.95 (69%), h12 +2.32 (77%).

## 2. Category audit

Monthly MAE pooled over h=1..12 and mean |signed 12m contribution| to the CGNCR identity
error (sign: spend/fin +, receipts −). Same 86 PRE / 43 OOS origins.

### 2a. PRE (SA nudge on; only `interest` differs between engines)

| Category | monthly MAE isin / rr | 12m abs contrib isin / rr | 12m signed bias (rr) |
|---|---|---|---|
| other_fin (ANRV) | 2.20 / 2.20 | 9.66 / 9.66 | −3.19 |
| nontax | 1.10 / 1.10 | 8.77 / 8.77 | +7.51 |
| acq_sec (ANRS) | 1.07 / 1.07 | 8.09 / 8.09 | −0.23 |
| cap | 0.89 / 0.89 | 6.99 / 6.99 | −1.41 |
| rdel_proxy | 0.91 / 0.91 | 5.90 / 5.90 | −5.87 |
| resid_gap | 0.96 / 0.96 | 5.68 / 5.68 | +1.03 |
| recv_pay (ANRT) | 1.38 / 1.38 | 5.43 / 5.43 | +0.87 |
| gilt_adj (ANRU) | 1.05 / 1.05 | 5.28 / 5.28 | +0.22 |
| other_tax | 0.63 / 0.63 | 5.09 / 5.09 | +1.88 |
| net_lend (ANRH) | 1.26 / 1.26 | 5.06 / 5.06 | −0.52 |
| it | 0.50 / 0.50 | 4.99 / 4.99 | +4.94 |
| **interest** | **0.77 / 0.58** | **4.98 / 3.27** | +0.90 (isin) / −0.25 (rr) |
| welfare | 0.52 / 0.52 | 2.97 / 2.97 | +1.61 |
| vat / nics / ct | 0.18 / 0.16 / 0.10 | 1.70 / 1.20 / 1.03 | +1.6 / +0.6 / +1.0 |

Without the SA nudge (Oct-27 spec) the two receipts leaves it touches are: `it` monthly MAE
0.31 (vs 0.50), 12m signed bias +2.27 (vs +4.94); `other_tax` 0.52 (vs 0.63), bias −0.91 (vs
+1.88). The nudge adds ~+3.5 £bn of one-way 12m bias on PRE through those two leaves.

### 2b. OOS

| Category | monthly MAE isin / rr / rr-no-nudge | 12m abs contrib isin / rr | 12m signed bias (rr) |
|---|---|---|---|
| **interest** | **3.90 / 2.94 / 2.94** | **28.82 / 18.82** | +24.6 (isin) / −0.95 (rr) |
| rdel_proxy | 2.62 | 27.36 | −27.3 |
| gilt_adj | 4.00 | 23.89 | −6.7 |
| cap | 2.54 | 22.01 | −15.1 |
| welfare | 1.78 | 19.74 | +10.8 |
| apf_xfer | 1.73 | 19.40 | −4.7 |
| resid_gap | 2.78 | 16.13 | +5.7 |
| other_tax | 1.44 / 1.44 / **0.87** | 15.21 / 15.21 / **9.54** | +15.2 (nudge) / +8.3 (no nudge) |
| nontax | 1.44 | 13.21 | +12.8 |
| other_fin | 2.53 | 11.72 | +8.5 |
| vat | 0.86 | 10.27 | +10.3 |
| it | 1.17 / 1.17 / **0.50** | 9.58 / 9.58 / **4.02** | +9.6 (nudge) / +0.3 (no nudge) |
| recv_pay | 2.61 | 5.09 | −3.3 |
| ct / nics / acq_sec / net_lend | ≤0.9 | ≤3.9 | small |

Reading: on OOS the isin interest leaf is the single largest line (28.8 of 12m contribution,
bias +24.6). Under rr the worst lines are the expenditure proxies (rdel, cap), gilt_adj and
welfare — the Oct-27 story's conclusion, unchanged. The SA nudge roughly doubles the `it` and
`other_tax` contributions and turns them one-way.

## 3. Interest leaf alone (same origins; 12m cumulative MAE / bias £bn)

| Engine | PRE monthly MAE | PRE cum MAE h1/h3/h6/h12 | PRE 12m bias | OOS monthly MAE | OOS cum MAE h1/h3/h6/h12 | OOS 12m bias |
|---|---|---|---|---|---|---|
| isin_curve (`monthly_path`) | 0.774 | 0.71 / 1.78 / 2.94 / **4.98** | +0.90 | 3.897 | 3.33 / 7.68 / 13.40 / **28.82** | **+24.6** |
| rr triangle (shipped before) | 0.582 | 0.56 / 1.16 / 1.91 / 3.27 | −0.25 | 2.938 | 3.03 / 7.18 / 12.60 / 18.82 | −0.95 |
| plain calendar carry | 0.557 | 0.56 / 1.06 / 1.66 / **2.43** | +0.23 | 2.939 | 3.27 / 6.61 / 11.17 / **15.03** | −8.2 |

By origin year, isin 12m error: PRE 2012 +4.0, 2013 −7.0, 2017 +3.3, 2018 +8.0; OOS 2022
+24.9, **2023 +74.5**, 2024 −4.7, 2025 −3.1. RR: 2022 −31.5, 2023 +27.6, 2024 −2.3.

Mechanics of the isin failure in `sovereign_cash/isin_curve_interest.py::monthly_path`:
- `annual_bn("uk", Y)` is NaN for Y ≤ 2015 (`forecast_full` needs the book/curve/published
  I0 for Y−1). Fallback = same-month-last-year × 12 × month share, i.e. a calendar carry
  distorted by the JW2P seasonal profile. Half of PRE targets run on this.
- For Y ≥ 2016 the annual is the 31-Dec-(Y−1) forecast. For an origin in Jan–Nov it is
  lookahead on next-year months (book, curve and published I0 for Y−1 are not yet known) and
  11-months stale on the open year (no in-year JW2P print enters).
- `scale = last_cy_actual / annual_bn(last_cy)`: last year's forecast error becomes a
  multiplicative correction on this year's level. 2022 JW2P actual ≫ the 79.3 forecast, so
  2023 origins carry the 2023/24 annuals (135/133) up by ~1.4–1.5× → +74 £bn.
- The "fair cash leg −20.5%" in `isin_interest_bugfix.md` is annual Dec-31-origin forecasts
  share-spread vs the RR live rule. The monthly cash engine never saw that number; the
  wrapper's own monthly backtest is this table.

Note for later (not promoted here): the plain calendar carry beats the RR triangle on the
interest leaf on both windows. The RR triangle's headline edge over calendar was never
re-checked after the fin-part decomposition; it is a candidate for a PRE-gated re-score.

## 4. Policy overlays

- **Soft pin α** (remit CGNCR vintage known at `info_date(origin)`, α × gap spread over the
  FY's remaining months): PRE mean 8.438 → 8.206 (+2.5pp edge, clears h=12) on the Oct-27 spec;
  8.931 → 8.899 with the nudge on; 9.986 → 9.825 under isin. OOS better at h=1/3/6 in every
  pair, slightly worse at h=12 on the no-nudge spec (14.75 → 15.53). Honest by construction
  (vintage_date filter; production uses the same `vintage_plan` at long horizons —
  `model_beats` is False at h=12, so production already takes the plan past the FY). Real edge,
  small, does not create or rescue the ship decision.
- **Remit pin / live hard close** (`run.py` `live_pin`): live path only, not in the scorer.
- **Legislated steps**: no policy-step overlay enters the UK BU backtest (no `policy/` import in
  `bottom_up_macro.py`). Not a factor.
- **`PLAN_SCALE_CHANNEL`** (Oct-29, `model.plan_profile`): a production-side overlay that
  spreads the plan residual uniformly when plan ≥ 1.2 × prior FY. It improved production PRE
  9.246 → 8.993 and is why the control bar moved. Its PRE edge is honest (uses only the prior
  complete FY and the vintage at origin).

## 5. Live summary vs fresh evaluate()

| Field | live `uk_cash_summary.json` | fresh |
|---|---|---|
| bu_pre h1/3/6/12 | 3.685 / 6.999 / 11.162 / 17.453 | 3.685 / 6.999 / 11.162 / 17.453 |
| prod_pre | 4.051 / 6.853 / 11.679 / 13.389 | identical |
| edge / ship / worse | −0.093 / False / (3, 12) | −0.093 / False / [3, 12] |
| bu_oos | 6.312 / 13.856 / 21.637 / 42.073 | identical |

Not stale. The skill note "BU 9% worse; NO-SHIP (research only)" is correct for the SPEC as
wired.

## 6. Files

- `sovereign_cash/experiments/uk_cash_ship_autopsy.py` — re-score (≈75 s).
- `sovereign_cash/experiments/uk_cash_ship_autopsy.json` — `gates.*` (per variant: PRE/OOS by
  horizon, bias, per-origin-year h=12), `category.*`, `interest_leaf.*`, `isin_engine`,
  `production`, `live_summary`, `addendum` (scale-channel control, no-nudge category audit,
  Jan/Feb leaf errors by target month).
