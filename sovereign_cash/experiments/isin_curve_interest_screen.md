# ISIN + yield-curve interest screen (point-in-time)

Screen only — **no SHIP changes**. Replaces the wrong-method prior screen
(`isin_interest_xcountry.md`), which used locked face×coupon only.

## 1. Prior screen mistake

The prior screen called `isin_book.locked_interest` — face × coupon on bonds
alive at end of *y−1*, **bill_rate = 0**, no FRN reset, no bill rolls, no
new-issue refinance, and almost no linker uplift. That is why UK ISIN/pub sat
near **0.47** in the rate-rise years and why the screen looked like “no edge”:
it was measuring a different object from published interest.

The debt tab already had the right engine (`debt/cashflow.components` +
`bill_ladder` + `build.project`). This screen runs that engine **point-in-time**.

## 2. What full ISIN + curve includes

As-of end of *y−1* (ops with `op_dt ≤` that date; curve ≤ that date):

1. **Locked coupons** on the outstanding book, linkers × index ratio.
2. **Linker uplift** — trailing CPI/RPI YoY known at origin (no WEO forward).
3. **FRN** face × (forward 3m + tape spread).
4. **Bill ladder** — remaining life at locked short rate, then rolls at the
   forward bill yield (`cashflow.bill_ladder`).
5. **Non-marketable residual** — `pub(y−1) − marketable(y−1)` held flat
   (same residual identity as `build.project`).
6. **New bonds** — last-year net issuance + scaled redemptions, couponed at
   the issuance-mix forward par off the PIT zero curve (UK BoE / US GSW /
   DE·FR·IT market_monthly par bootstrapped).

**Windows.** PRE = target years 2015–2019. OOS = 2020–2025.
Origin for year *y* = 31 Dec *y−1*.

**Comparators.** `smly` = pub(*y−1*); `live` = UK RR triangle / FR AFT-or-hold /
DE coupon_stock / IT smly; `locked` / `locked_resid` = prior-screen method.

## 3. Annual MAE (local bn)

Actual = cash interest target (UK = JW2P CY sum; DE/FR/IT/US =
`debt_country_annual.interest_bn`).

| cc | window | n | smly | **full ISIN+curve** | locked | locked_resid | live | live rule | mkt/pub₀ | locked/act |
|----|--------|---|------|---------------------|--------|--------------|------|-----------|----------|------------|
| de | ALL | 10 | 3.569 | **2.542** | 4.397 | 2.514 | 3.46 | coupon_stock | 1.151 | 0.925 |
| de | OOS | 6 | 4.502 | **3.56** | 7.206 | 3.856 | 3.953 | coupon_stock | 1.255 | 0.878 |
| de | PRE | 4 | 2.169 | **1.015** | 0.184 | 0.502 | 2.722 | coupon_stock | 0.994 | 0.995 |
| fr | ALL | 10 | 5.159 | **4.619** | 6.822 | 4.191 | 4.503 | smly_open | 1.184 | 0.897 |
| fr | OOS | 6 | 7.26 | **6.646** | 10.724 | 5.721 | 6.166 | smly_open | 1.238 | 0.837 |
| fr | PRE | 4 | 2.009 | **1.578** | 0.97 | 1.896 | 2.009 | smly_open | 1.102 | 0.988 |
| it | ALL | 10 | 4.994 | **4.699** | 26.092 | 4.904 | 4.994 | smly | 0.767 | 0.637 |
| it | OOS | 6 | 7.13 | **7.002** | 30.578 | 7.406 | 7.13 | smly | 0.794 | 0.6 |
| it | PRE | 4 | 1.792 | **1.244** | 19.361 | 1.15 | 1.792 | smly | 0.726 | 0.693 |
| uk | ALL | 11 | 11.435 | **10.266** | 42.835 | 11.636 | 12.253 | rr_triangle | 0.925 | 0.568 |
| uk | OOS | 6 | 17.724 | **14.555** | 60.147 | 17.035 | 19.004 | rr_triangle | 0.914 | 0.474 |
| uk | PRE | 5 | 3.888 | **3.832** | 16.866 | 3.538 | 4.153 | rr_triangle | 0.942 | 0.71 |
| us | ALL | 10 | 80.613 | **54.154** | 180.144 | 56.783 | 80.613 | smly | 1.212 | 0.695 |
| us | OOS | 6 | 109.026 | **72.658** | 259.427 | 77.31 | 109.026 | smly | 1.183 | 0.617 |
| us | PRE | 4 | 37.994 | **26.4** | 61.219 | 25.993 | 37.994 | smly | 1.255 | 0.812 |

## 4. OOS edge vs live (full ISIN+curve)

SHIP gate = ≥5% OOS on the **fair cash interest leg**: live = native monthly path (RR triangle / DE coupon carve / FR ex-ante AFT or smly / IT smly); full = share-spread of the PIT annual. Prior screen share-spread both sides.

| cc | ann full | ann live | ann edge | mth full | mth live | mth edge | live rule | SHIP? |
|----|----------|----------|----------|----------|----------|----------|-----------|-------|
| de | 3.56 | 3.953 | 9.9% | 4.384 | 4.849 | 9.6% | coupon_stock | CANDIDATE |
| fr | 6.646 | 6.166 | -7.8% | 10.248 | 8.671 | -18.2% | smly_open | no |
| it | 7.002 | 7.13 | 1.8% | 18.585 | 19.587 | 5.1% | smly | CANDIDATE |
| uk | 14.555 | 19.004 | 23.4% | 25.922 | 32.621 | 20.5% | rr_triangle | CANDIDATE |
| us | 72.658 | 109.026 | 33.4% |  |  |  | smly | no |

## 5. Per-country detail

### UK

OOS full MAE **14.555** vs live `rr_triangle` **19.004** (edge 23.4%), smly 17.724, locked 60.147 (resid 17.035). Mean marketable/pub at y0 = 0.914; locked/actual = 0.474.

```
 year window  actual   smly   full  locked  locked_resid   live  uplift  bills  new   r3m  ratio_mkt_pub0
 2015    PRE   52.57  56.46    NaN     NaN           NaN  54.36     NaN    NaN  NaN   NaN             NaN
 2016    PRE   56.18  52.57  52.93   41.81         52.00  52.34    4.61   0.34 1.46  0.37            0.93
 2017    PRE   61.70  56.18  56.32   41.81         56.19  54.19    9.70  -0.03 1.01 -0.03            0.94
 2018    PRE   58.86  61.70  62.32   41.10         60.99  62.07   17.35   0.29 0.88  0.37            0.97
 2019    PRE   55.29  58.86  58.51   39.85         57.61  59.70   11.89   0.55 1.03  0.74            0.93
 2020    OOS   43.51  55.29  54.55   37.66         53.10  53.69   10.05   0.66 0.69  0.57            0.92
 2021    OOS   61.67  43.51  44.86   38.24         44.08  41.01    5.55  -0.09 0.43 -0.16            1.06
 2022    OOS  121.22  61.67  79.30   36.65         60.08  63.35   39.62   0.20 1.84  0.41            1.24
 2023    OOS  126.81 121.22 134.88   38.03        122.60 133.88   84.78   2.33 3.50  3.47            0.96
 2024    OOS  123.28 126.81 132.69   44.10        132.88 133.59   32.57   4.56 5.33  4.80            0.65
 2025    OOS  131.00 123.28 131.09   51.91        131.08 123.06   22.28   4.71 8.56  4.48            0.66
```

Monthly h=12 sum MAE (share-spread): `{'PRE': {'full': 9.99, 'live': 8.158, 'live_sharespread': 10.481, 'locked_resid': 10.115, 'smly': 10.071}, 'OOS': {'full': 25.922, 'live': 32.621, 'live_sharespread': 28.458, 'locked_resid': 27.907, 'smly': 27.976}}`

### DE

OOS full MAE **3.56** vs live `coupon_stock` **3.953** (edge 9.9%), smly 4.502, locked 7.206 (resid 3.856). Mean marketable/pub at y0 = 1.255; locked/actual = 0.878.

```
 year window  actual  smly  full  locked  locked_resid  live  uplift  bills   new   r3m  ratio_mkt_pub0
 2016    PRE   20.61 23.75 22.25   20.42         21.43 24.63    0.52  -0.06  0.09 -0.26            0.95
 2017    PRE   18.25 20.61 19.49   18.34         18.52 21.19    0.30  -0.16 -0.17 -0.58            0.94
 2018    PRE   17.15 18.25 17.01   16.79         16.71 18.74    1.30  -0.07 -0.07 -0.56            1.06
 2019    PRE   15.07 17.15 16.11   15.18         15.54 17.42    1.49  -0.08 -0.06 -0.50            1.03
 2020    OOS   10.72 15.07 13.46   13.37         13.26 15.54    1.07  -0.10 -0.43 -0.54            1.03
 2021    OOS   10.56 10.72  8.80   14.43         11.79 12.54    0.30  -0.85 -1.10 -0.67            1.37
 2022    OOS   16.84 10.56  9.22   13.17          9.29 11.91    2.83  -1.15 -0.70 -0.67            1.47
 2023    OOS   24.65 16.84 19.71   13.83         17.50 18.35    8.01   3.20  3.40  2.13            1.52
 2024    OOS   31.68 24.65 27.45   19.10         29.92 26.60    5.43   4.88  3.06  3.13            1.21
 2025    OOS   33.05 31.68 33.14   23.39         35.96 33.66    2.19   2.60  2.57  2.31            0.93
```

Monthly h=12 sum MAE (share-spread): `{'PRE': {'full': 2.104, 'live': 3.19, 'live_sharespread': 3.19, 'locked_resid': 1.847, 'smly': 2.794}, 'OOS': {'full': 4.384, 'live': 4.849, 'live_sharespread': 4.849, 'locked_resid': 4.55, 'smly': 5.392}}`

### FR

OOS full MAE **6.646** vs live `smly_open` **6.166** (edge -7.8%), smly 7.26, locked 10.724 (resid 5.721). Mean marketable/pub at y0 = 1.238; locked/actual = 0.837.

```
 year window  actual  smly  full  locked  locked_resid  live  uplift  bills   new   r3m  ratio_mkt_pub0
 2016    PRE   36.43 38.49 36.90   36.43         38.36 38.49    0.30  -0.35  0.51 -0.22            1.05
 2017    PRE   35.46 36.43 34.77   34.90         34.91 36.43    0.89  -0.71  0.35 -0.51            1.07
 2018    PRE   36.04 35.46 34.28   33.70         34.26 35.46    3.69  -0.65  0.44 -0.50            1.13
 2019    PRE   31.61 36.04 35.00   32.60         34.94 36.04    6.53  -0.51  0.61 -0.43            1.16
 2020    OOS   25.39 31.61 29.86   31.71         30.72 31.61    3.92  -0.58 -0.09 -0.54            1.20
 2021    OOS   30.97 25.39 23.23   31.14         24.83 25.39    1.52  -1.09 -0.68 -0.64            1.35
 2022    OOS   46.05 30.97 29.50   29.91         29.73 30.97    6.40  -1.05  0.01 -0.65            1.22
 2023    OOS   41.42 46.05 49.34   29.02         45.17 46.05   18.78   3.42  4.37  2.23            1.16
 2024    OOS   46.85 41.42 45.92   32.73         45.12 41.42   18.14   5.62  4.68  3.16            1.39
 2025    OOS   53.46 46.85 51.20   38.28         52.40 53.50    7.23   5.23  4.87  2.45            1.11
```

Monthly h=12 sum MAE (share-spread): `{'PRE': {'full': 3.651, 'live': 3.944, 'live_sharespread': 3.944, 'locked_resid': 3.703, 'smly': 3.944}, 'OOS': {'full': 10.248, 'live': 8.671, 'live_sharespread': 10.722, 'locked_resid': 10.738, 'smly': 10.943}}`

### IT

OOS full MAE **7.002** vs live `smly` **7.13** (edge 1.8%), smly 7.13, locked 30.578 (resid 7.406). Mean marketable/pub at y0 = 0.794; locked/actual = 0.6.

```
 year window  actual  smly  full  locked  locked_resid  live  uplift  bills  new   r3m  ratio_mkt_pub0
 2016    PRE   64.98 66.68 65.53   45.45         65.12 66.68    0.27  -0.08 0.74 -0.07            0.72
 2017    PRE   64.06 64.98 64.04   43.89         63.42 64.98   -0.14  -0.22 1.38 -0.20            0.71
 2018    PRE   63.55 64.06 63.45   42.97         63.15 64.06    1.87  -0.29 1.34 -0.27            0.73
 2019    PRE   59.51 63.55 63.81   42.35         62.93 63.55    1.78   0.21 2.52  0.19            0.75
 2020    OOS   56.50 59.51 58.74   42.99         60.15 59.51    1.11  -0.23 0.91 -0.21            0.77
 2021    OOS   62.82 56.50 54.45   41.92         55.42 56.50   -0.32  -0.59 0.49 -0.47            0.77
 2022    OOS   81.41 62.82 61.69   38.48         59.39 62.82    3.41  -0.47 1.04 -0.44            0.70
 2023    OOS   76.30 81.41 84.46   38.56         81.48 81.41   17.48   2.56 5.65  2.44            0.80
 2024    OOS   83.80 76.30 80.82   46.97         84.71 76.30   12.34   4.11 6.98  3.53            0.93
 2025    OOS   86.03 83.80 85.50   54.47         91.30 83.80    2.46   3.31 4.54  2.64            0.79
```

Monthly h=12 sum MAE (share-spread): `{'PRE': {'full': 20.923, 'live': 22.679, 'live_sharespread': 21.077, 'locked_resid': 20.798, 'smly': 21.077}, 'OOS': {'full': 18.585, 'live': 19.587, 'live_sharespread': 17.874, 'locked_resid': 19.257, 'smly': 17.874}}`

### US

OOS full MAE **72.658** vs live `smly` **109.026** (edge 33.4%), smly 109.026, locked 259.427 (resid 77.31). Mean marketable/pub at y0 = 1.183; locked/actual = 0.617.

```
 year window  actual   smly   full  locked  locked_resid   live  uplift  bills   new  r3m  ratio_mkt_pub0
 2016    PRE  240.03 223.18 230.32  225.66        227.28 223.18    2.10  12.38 21.79 0.79            1.20
 2017    PRE  262.55 240.03 249.00  225.75        240.12 240.03   27.67  16.75 22.62 0.90            1.23
 2018    PRE  324.98 262.55 272.99  236.86        273.66 262.55   45.02  35.12 24.21 1.76            1.31
 2019    PRE  375.16 324.98 344.81  269.57        357.68 324.98   51.72  66.64 35.51 2.58            1.29
 2020    OOS  345.47 375.16 379.77  285.81        391.40 375.16   38.32  42.11 25.04 1.63            1.06
 2021    OOS  352.34 345.47 335.43  265.79        325.45 345.47   27.43   6.49 13.69 0.12            0.95
 2022    OOS  475.89 352.34 362.06  269.91        356.45 352.34  103.85  16.93 23.99 0.42            1.17
 2023    OOS  658.27 475.89 547.03  337.13        543.10 475.89  179.96 185.64 82.97 4.73            1.56
 2024    OOS  879.88 658.27 741.32  426.27        747.41 658.27   88.48 292.72 99.00 4.74            1.29
 2025    OOS  969.94 879.88 948.83  540.31        993.92 879.88   63.57 281.31 99.05 4.22            1.07
```

## 6. Verdict

UK / DE / IT clear ≥5% OOS on the **fair** monthly cash leg after bugfixes
(see `isin_interest_bugfix.md`). Live hooks: `INTEREST_ISIN_CURVE=True` in
each country's `bottom_up_macro` — flip the flag to revert without touching
other SHIP knobs. FR uses ISIN as open-year fill only (never vs honest AFT).

Bugfixes that removed false negatives:

1. **Fair monthly** — live native path vs share-spread ISIN annual (was both share-spread).
2. **UK residual ≥ 0** — accrual uplift vs cash JW2P no longer injects −£15bn into 2022.
3. **FR AFT ex-ante** — exécution rows are not known at Dec y−1; ISIN fills open years.
4. **Prior locked screen** — face×coupon only; replaced by full ISIN+curve engine.

Detail: `isin_interest_bugfix.md`.

_Generated by `isin_curve_interest_screen.py`. Do not commit outputs unless asked._