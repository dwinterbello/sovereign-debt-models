# ISIN-book interest as a cash-forecast edge (cross-country screen)

Screen only — no SHIP changes. Annual asof-honest forecasts of central-government
interest from the issuance ops ISIN book (`sovereign_cash/debt/isin_book.py`),
scored against each country's live interest rule and against same-year-last-year (smly).

**Windows.** PRE = target years 2015–2019. OOS = 2020–2025 (rate-rise regime).
Origin for year *y* = end of *y−1* (book and published interest known through Dec *y−1*).

**ISIN variants.**
- `isin_raw` — locked coupon cash from bonds outstanding at end-*y−1* (no bills, no new issuance).
- `isin_resid` — `isin_raw(y) + (pub(y−1) − isin_raw(y−1))` (carry non-marketable residual).
- `isin_grow` — `pub(y−1) × isin_raw(y)/isin_raw(y−1)` (shape from the book, level from published).

## 1. Coverage inventory

| cc | ops? | rows | op_dt range | unique ISIN | 2024 book n / face bn | note |
|----|------|------|-------------|-------------|------------------------|------|
| uk | True | 6470 | 1981-03-27→2026-10-23 | 201 | 138.0 / 2483.4 | coupon empty on tape; parsed from sec_desc_long; bills often lack ISIN |
| de | True | 2344 | 1999-01-06→2026-10-20 | 683 | 86.0 / 1851.3 |  |
| fr | True | 6599 | 1999-01-04→2026-10-26 | 1607 | 100.0 / 2538.6 |  |
| it | True | 5202 | 2000-01-03→2026-10-13 | 1167 | 162.0 / 2301.3 |  |
| us | True | 13163 | 1979-10-31→2026-10-21 | 5097 | 457.0 / 28132.4 |  |
| jp | True | 5324 | 1979-08-27→2026-10-23 | 2840 | 358.0 / 1133722.0 | bills special-cased in cashflow.attach_japan_bills (MoF stock) |
| se | True | 2935 | 1990-01-01→2026-10-21 | 423 | 35.0 / 8332.4 | BROKEN: outstanding snapshots + auction taps double-count face |
| no | False | 0 | nan→nan | 0 | nan / nan | no ops parquet |

NO has no `ops_no.parquet`. Its published interest path is already an ISIN bottom-up
from Norges Bank primary-market volumes (`debt/sources._no_interest`). SE's ops file
double-counts CH1073 outstanding snapshots with CH926 auction taps — face ~8× debt;
do not use it for interest until filtered.

## 2. Where ISIN is already used

- **Debt tab / forward interest** (`debt/build.project`, `debt/cashflow.py`): every country
  with an ops book uses the end-*y0* ISIN coupon stock (plus bill ladder / JP MoF bills),
  scaled so *y0* matches the published interest outturn; the residual is held as
  non-marketable. That is the Dalio interest path, not the monthly cash nowcast.
- **DE cash later years** (`de/bottom_up_macro.LATER_INTEREST_MODE=coupon_stock`): uses
  *published* annual interest × FA stock growth × rates prior — **not** the raw ISIN
  coupon sum. Name is coupon×stock; source is the debt-book outturn table.
- **NO debt interest outturn**: ISIN bottom-up already (see above).
- **FR / IT / UK / SE cash**: live interest is AFT pin+smly, smly, OBR RR triangle,
  and seasonal — none read the ops tape for the monthly cash leg.

## 3. Annual MAE (local bn)

Actual = cash interest target (UK = JW2P CY sum; DE/FR/IT = `debt_country_annual.interest_bn`).

| cc | window | n | smly | isin_raw | isin_resid | isin_grow | live | live rule | mean pub−ISIN gap | ISIN/pub |
|----|--------|---|------|----------|------------|-----------|------|-----------|-------------------|----------|
| de | ALL | 11 | 3.569 | 4.089 | 2.514 | 2.768 | 3.46 | coupon_stock | 2.87 | 0.928 |
| de | OOS | 6 | 4.502 | 7.206 | 3.856 | 4.307 | 3.953 | coupon_stock | 5.03 | 0.878 |
| de | PRE | 5 | 2.169 | 0.349 | 0.502 | 0.461 | 2.722 | coupon_stock | 0.27 | 0.988 |
| fr | ALL | 11 | 5.159 | 6.377 | 4.191 | 3.998 | 4.391 | aft_or_hold | 5.02 | 0.902 |
| fr | OOS | 6 | 7.26 | 10.724 | 5.721 | 5.411 | 5.98 | aft_or_hold | 8.56 | 0.837 |
| fr | PRE | 5 | 2.009 | 1.16 | 1.896 | 1.88 | 2.009 | aft_or_hold | 0.77 | 0.98 |
| it | ALL | 11 | 4.994 | 25.508 | 4.904 | 6.671 | 4.994 | smly | 25.51 | 0.643 |
| it | OOS | 6 | 7.13 | 30.578 | 7.406 | 10.157 | 7.13 | smly | 30.58 | 0.6 |
| it | PRE | 5 | 1.792 | 19.424 | 1.15 | 1.443 | 1.792 | smly | 19.42 | 0.695 |
| uk | ALL | 11 | 11.435 | 39.867 | 10.931 | 13.14 | 12.253 | rr_triangle | 39.87 | 0.59 |
| uk | OOS | 6 | 17.724 | 60.147 | 17.035 | 21.209 | 19.004 | rr_triangle | 60.15 | 0.474 |
| uk | PRE | 5 | 3.888 | 15.531 | 3.607 | 3.458 | 4.153 | rr_triangle | 15.53 | 0.729 |

## 4. OOS rank (best ISIN variant vs live)

| rank | cc | best ISIN | MAE | live MAE | edge vs live | edge vs smly | structural gap |
|------|----|-----------|-----|----------|--------------|--------------|----------------|
| 1 | uk | isin_resid | 17.035 | 19.004 | 10.4% | 3.9% | gap 60.15 (ratio 0.474) |
| 2 | fr | isin_grow | 5.411 | 5.98 | 9.5% | 25.5% | gap 8.56 (ratio 0.837) |
| 3 | de | isin_resid | 3.856 | 3.953 | 2.5% | 14.3% | gap 5.03 (ratio 0.878) |
| 4 | it | isin_resid | 7.406 | 7.13 | -3.9% | -3.9% | gap 30.58 (ratio 0.6) |

## 5. Per-country verdict

### UK

Target = JW2P calendar-year sum (cash), not NMFX. OOS best ISIN `isin_resid` 17.035 vs RR triangle 19.004 (edge 10.4%) and smly 17.724. ISIN ≈ smly annually; RR can look worse on a CY sum because the triangle is FY-shaped. Monthly h=12: live RR still beats ISIN-grow. Marketable coupons miss linker uplift / APF — ratio ~0.3–0.6 in 2021–25. Not a replacement for the RR leg.

```
 year window  actual   smly  isin_raw  isin_resid  isin_grow   live  gap_pub_isin
 2015    PRE   52.57  56.46     42.38       56.45      56.45  54.36         10.19
 2016    PRE   56.18  52.57     41.81       52.00      51.86  52.34         14.38
 2017    PRE   61.70  56.18     41.81       56.19      56.19  54.19         19.89
 2018    PRE   58.86  61.70     41.10       60.99      60.65  62.07         17.76
 2019    PRE   55.29  58.86     39.85       57.61      57.07  59.70         15.44
 2020    OOS   43.51  55.29     37.66       53.10      52.25  53.69          5.84
 2021    OOS   61.67  43.51     38.24       44.08      44.17  41.01         23.43
 2022    OOS  121.22  61.67     36.65       60.08      59.11  63.35         84.56
 2023    OOS  126.81 121.22     38.03      122.60     125.77 133.88         88.78
 2024    OOS  123.28 126.81     44.10      132.88     147.05 133.59         79.17
 2025    OOS  131.00 123.28     51.91      131.08     145.10 123.06         79.09
```

Monthly h=12 sum MAE (share-spread annuals): `{'PRE': {'isin_grow': 10.151, 'live': 10.481, 'smly': 10.071}, 'OOS': {'isin_grow': 30.254, 'live': 28.458, 'smly': 27.976}}`

### DE

OOS best ISIN `isin_resid` MAE 3.856 vs live `coupon_stock` 3.953 (edge 2.5%) and smly 4.502. Pre-2020 raw ISIN ≈ published (ratio ~0.99); post-2022 gap opens (new coupons + bills). Live already does published×FA-stock×rates — ISIN resid is a statistical tie, not free edge. Keep live.

```
 year window  actual  smly  isin_raw  isin_resid  isin_grow  live  gap_pub_isin
 2015    PRE   23.75   NaN     22.75         NaN        NaN   NaN          1.01
 2016    PRE   20.61 23.75     20.42       21.43      21.33 24.63          0.18
 2017    PRE   18.25 20.61     18.34       18.52      18.51 21.19         -0.09
 2018    PRE   17.15 18.25     16.79       16.71      16.71 18.74          0.36
 2019    PRE   15.07 17.15     15.18       15.54      15.51 17.42         -0.11
 2020    OOS   10.72 15.07     13.37       13.26      13.27 15.54         -2.65
 2021    OOS   10.56 10.72     14.43       11.79      11.57 12.54         -3.88
 2022    OOS   16.84 10.56     13.17        9.29       9.63 11.91          3.67
 2023    OOS   24.65 16.84     13.83       17.50      17.69 18.35         10.81
 2024    OOS   31.68 24.65     19.10       29.92      34.04 26.60         12.57
 2025    OOS   33.05 31.68     23.39       35.96      38.79 33.66          9.66
```

Monthly h=12 sum MAE (share-spread annuals): `{'PRE': {'isin_grow': 1.299, 'live': 2.459, 'smly': 2.126}, 'OOS': {'isin_grow': 4.834, 'live': 4.849, 'smly': 5.392}}`

### FR

OOS annual: `isin_grow` 5.411 vs AFT/hold 5.98 (edge 9.5%) and smly 7.26. Monthly share-spread flips it: live AFT wins OOS (see monthly block) because the charge pin is the right level when known. ISIN/pub ~0.84 OOS — BTF discount and non-OAT items sit outside the coupon book. Free edge only for pre-pin months or marketable-only stress; do not replace AFT.

```
 year window  actual  smly  isin_raw  isin_resid  isin_grow  live  gap_pub_isin
 2015    PRE   38.49   NaN     36.57         NaN        NaN   NaN          1.92
 2016    PRE   36.43 38.49     36.43       38.36      38.35 38.49          0.00
 2017    PRE   35.46 36.43     34.90       34.91      34.91 36.43          0.55
 2018    PRE   36.04 35.46     33.70       34.26      34.24 35.46          2.34
 2019    PRE   31.61 36.04     32.60       34.94      34.86 36.04         -0.99
 2020    OOS   25.39 31.61     31.71       30.72      30.75 31.61         -6.32
 2021    OOS   30.97 25.39     31.14       24.83      24.94 38.50         -0.18
 2022    OOS   46.05 30.97     29.91       29.73      29.73 51.50         16.15
 2023    OOS   41.42 46.05     29.02       45.17      44.69 54.80         12.40
 2024    OOS   46.85 41.42     32.73       45.12      46.70 50.10         14.12
 2025    OOS   53.46 46.85     38.28       52.40      54.80 53.50         15.18
```

Monthly h=12 sum MAE (share-spread annuals): `{'PRE': {'isin_grow': 2.844, 'live': 3.179, 'smly': 3.179}, 'OOS': {'isin_grow': 10.719, 'live': 5.71, 'smly': 10.943}}`

### IT

OOS: best ISIN `isin_resid` 7.406 **loses** to smly 7.13 (edge -3.9%). Structural gap ~€25–30bn (ratio ~0.60) — BOT discount, CCT floaters, non-marketable. Stock×rate and SARIMAX+rates already failed IT's PRE/OOS gate; ISIN scaled variants do not reopen it. Do not wire.

```
 year window  actual  smly  isin_raw  isin_resid  isin_grow  live  gap_pub_isin
 2015    PRE   66.68   NaN     47.01         NaN        NaN   NaN         19.67
 2016    PRE   64.98 66.68     45.45       65.12      64.47 66.68         19.53
 2017    PRE   64.06 64.98     43.89       63.42      62.75 64.98         20.17
 2018    PRE   63.55 64.06     42.97       63.15      62.72 64.06         20.58
 2019    PRE   59.51 63.55     42.35       62.93      62.63 63.55         17.16
 2020    OOS   56.50 59.51     42.99       60.15      60.40 59.51         13.51
 2021    OOS   62.82 56.50     41.92       55.42      55.09 56.50         20.91
 2022    OOS   81.41 62.82     38.48       59.39      57.68 62.82         42.93
 2023    OOS   76.30 81.41     38.56       81.48      81.57 81.41         37.74
 2024    OOS   83.80 76.30     46.97       84.71      92.95 76.30         36.83
 2025    OOS   86.03 83.80     54.47       91.30      97.19 83.80         31.56
```

Monthly h=12 sum MAE (share-spread annuals): `{'PRE': {'isin_grow': 20.395, 'live': 20.807, 'smly': 20.807}, 'OOS': {'isin_grow': 20.536, 'live': 17.874, 'smly': 17.874}}`

## 6. US / JP / SE / NO (notes only)

- **US**: ops book complete; fiscal TGA path already has its own interest from the
  coupon ladder / CBO. ISIN raw / published gap widens after 2018 as FRNs and bill
  costs dominate — same residual problem as UK. Not scored for cash here (US cash
  is `us/fiscal`, not `sovereign_cash`).
- **JP**: ops coupon book usable; bills absent from tape (MoF stock bolted on in
  `cashflow.attach_japan_bills`). Interest outturn is budget-basis; ISIN is a fair
  marketable prior for the debt tab (already wired). No JP monthly cash bottom-up
  interest leg in `sovereign_cash`.
- **SE**: ops unusable until outstanding snapshots are dropped or auctions-only
  netted. Stock×rate already tried in `se/bottom_up_macro` and lost PRE to seasonal;
  fix the book before re-testing.
- **NO**: already ISIN-bottom-up for the debt interest outturn; no separate ops file.
  Cash NBR interest stays seasonal / guidance-anchored.

## 7. Recommendation

**Largest OOS annual edge vs live:** `uk` via `isin_resid` (10.4% vs rr_triangle) — read with the monthly caveat in §5.

1. **Do not wire any country this pass.** No OOS win that survives both the annual
   table and the monthly share-spread check while clearing a honest PRE gate.
2. **IT** — hypothesized free edge (live=smly, book intact) **fails OOS**; residual
   ~€25–30bn dominates. Leave smly.
3. **FR** — annual `isin_grow` slightly beats AFT/hold; monthly AFT wins. Keep AFT.
4. **DE** — ISIN resid ties live `coupon_stock`. Keep live (already stock-scaled).
5. **UK** — ISIN ≈ smly on JW2P; misses linker uplift. Keep RR triangle for cash.
6. Fix **SE** ops (drop CH1073 outstanding rows or stop summing them with auctions)
   before any SE ISIN claim. **NO** is already ISIN-bottom-up on the debt outturn.

Bottom-up ISIN data is **not** an automatic cash-forecast edge for all countries.
It is already the debt-tab interest engine. On monthly cash it needs (a) a clean
tape, (b) marketable coupons ≈ published interest, (c) a live rule weaker than the
book. That intersection is empty among FR/DE/IT/UK on this screen.

_Generated by `isin_interest_xcountry.py`._