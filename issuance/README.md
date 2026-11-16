# Sovereign issuance → USD DV01

ISIN-level government-bond issuance for UK, DE, FR, IT, US. Official demand
(APF, PSPP, PEPP, SOMA) is netted where the public data supports it. Everything is
converted to dollar DV01 and drawn on `issuance_dv01_dashboard.html`.

```bash
PYTHONPATH=. .venv/bin/python -m issuance.run_refresh
PYTHONPATH=. .venv/bin/python -m issuance.run_refresh --only uk de us apf ecb fx flows reconcile official dv01 rv dashboard
PYTHONPATH=. .venv/bin/pytest issuance/tests -q
```

## Sources

| Country | Gross issuance | Official demand | Aggregate check |
|---|---|---|---|
| UK | DMO XML D2.1E + D2.2D | BoE APF operations (ISIN) | D1A gilts in issue vs cumulated ops |
| DE | Finanzagentur `emissionshistorie_en.xlsx` | ECB PSPP+PEPP + WAM | annual history vs ops |
| FR | AFT `hist_mlt` / `hist_btf` / syndications + Wayback latest-auctions / 2026 communiqués | ECB PSPP+PEPP + WAM | AFT annual vs ops |
| IT | MEF `parsed_data/*_combined.xlsx` + Banca d'Italia `Aste_anno_corrente.zip` / `storico_aste.zip` | ECB PSPP+PEPP + WAM | MEF stats vs ops |
| US | Fiscal Data `auctions_query` (CUSIP→ISIN) | NY Fed SOMA monthly CUSIP (notes/bonds/TIPS/FRN; bills stored, not in coupon DV01) | annual accepted vs ops |

Cloud notes: DMO XML is often captcha-blocked. `fetch_uk` still requests
D2.1E gilts and D2.2D T-bills; on a block, gilts fall back to the tracked
long-end workbook (`42_` / `35_`) and bills are omitted. Live AFT is
Cloudflare-blocked on cloud, so FR 2026 coupon auctions come from Wayback
`dernieres-adjudications` snapshots, the 1999–2026 syndication xlsx, and
tracked AFT communiqué rows (`issuance/data/fr_2026_press_auctions.json`).
IT 2026 auctions come from BdI `Aste_anno_corrente.zip` (live). US Fiscal
Data already runs through the latest auction. DE same.

EA official demand is **not** ISIN-published. `qe_alloc.allocate_isin` ages
last month’s face and allocates only the flow:

```
min ||h − h_neutral||²
s.t. Σ h = S*,   Σ h y = S* · W_pub
     aged ≤ h ≤ cap·o   (net buy; purchase window only)
     0 ≤ h ≤ aged       (net sale)
```

`h_neutral` is last month’s surviving face plus pro-rata flow
(purchases ∝ eligible outstanding, sales ∝ holdings). That is the
market-neutral CB rule — a 10bn sale does not land on one ISIN.
Published WAM is the target. Two-way (`_switch`) then the cap box
(`_reshuffle`) open only if the one-sided aging box cannot reach
W_pub. S* is the largest face ≤ min(published stock, Σ cap·o) whose
cap box still contains W_pub. Cap-fill pins WAM at the universe
average — that is why DE PSPP 2017–24 sat at ~8y against a 6.6y print.
Leftover is `non_central` slack and is **not** subtracted from coupon DV01.
Combined PSPP+PEPP can exceed 33% of an issue. US SOMA is
CUSIP-exact (NY Fed monthly); first sample month is stock only, not a
flow dump.

**DFA Own Holdings** stay on the ops file (they are outstanding) but are
dropped from market flows / DV01. Market Adjustment taps stay in — those are
bilateral placements into the market.

## RV metrics (trading)

All in `rv_wam.parquet` / `rv_supply.parquet` and on the dashboard.

| Metric | Definition |
|---|---|
| Coupon USD DV01 | residual modified duration × dirty price × spot USD, ex-bills |
| Private DV01 | net coupon DV01 − official DV01. QT < 0 official → private supply rises |
| 12m / 3m private | rolling sum, per-country own vintage |
| priv vs DE | private DV01 − Germany (Bund spread supply) |
| 2s10s | belly(7–15y) − front(1–7y). + = 10s-heavy |
| 10s30s | long(15y+) − belly. 15–20y is with the long end |
| 5s30s fly | front + long − 2·belly |
| Ultra / 15–20y | bucket private DV01 |
| Absorb % | official face / coupon face. QT negative |
| WAM stock | residual WAM of live coupon notional (bills out) |
| WAM public | APF ISIN residual / ECB published PSPP+PEPP blend / SOMA CUSIP residual (ex-bills) |
| WAM private | `(S·Ws − H·Wh) / (S − H)`. Identity err ~1e-9. US uses SOMA residual WAM |
| WAM issuance | residual WAM of that month's coupon issues (the **flow**) |
| Issuance WAM 3m/12m | amount-weighted rolling WAM of the flow |
| Duration issued | face × residual years (size of the duration flow) |
| Issuance gap | WAM_issuance − WAM_stock. + = new supply lengthens |
| Official share | H / S of coupon face |
| Official WAM gap | WAM_public − WAM_private (duration extraction) |
| Supply 3m / 12m | private coupon USD DV01 on multi-country plots (not z) |
| Cover / BTC | amount-weighted auction cover. IT stored ×100, divided back |
| UK tail z | D2.1A tail bp, rolling z, UK-only panel. + = fat stop = weak bid |

## Coverage (triple-check)

- UK D1A: 103/104 ISINs within £50mn. Residual gap £631mn on `GB00BL6C7720` 4⅛% 2027.
- APF live unmatured ~£406bn (gross ops 851bn includes matured). Tenor join from gilt ops.
- ECB latest: PSPP published cumulative is the stock check. PSPP cells use thousands commas
  (`-9,418`); that used to drop August and leave DE at July PSPP 445bn instead of 435bn.
  After the parse: DE PSPP 435 + PEPP 300 = 735bn; FR 360+240; IT 283+221.
  PEPP cells are unquoted integers; both programs merge WAM on calendar month
  (PSPP month-end dates vs PEPP `%b-%y`). Combined public WAM is face-weighted.
  ECB rows are **issuer jurisdiction**. The DE universe is Finanzagentur
  (Schatz/Bobl/Bund/ILB) only. S* is the largest Bund face whose 33% box
  still contains the published WAM — the Bund book itself, not a
  universe-weighted cap-fill. Face that cannot sit on that book is slack.
- FR syndications do not overlap `hist_mlt` (0 ISIN+date collisions).
- IT unknown tenor is −€16bn of old buybacks / missing maturity — not in DV01.
- US SOMA CUSIP stock is netted from coupon DV01. WAM_private uses SOMA residual WAM.
- PSPP+PEPP month 1 flow equals that month's opening net (cumsum). Later months are stock diffs.
- DE USD-Bonds are `sec_type=Foreign` and stay out of the euro coupon book.
- Rows with no maturity stay on the ops file and out of flows/DV01 (`reconcile.json` `unallocated_maturity`).
- FR syndication `op_dt` is the settlement date in the AFT file. There is no auction date column.

## Outputs

All rebuildable, under `issuance/cache/` (parquet) and `issuance/derived/`.

- `ops_{uk,de,fr,it,us}.parquet` — canonical operations
- `flow_isin.parquet` / `flow_tenor.parquet`
- `official_flow.parquet` / `official_isin_flow.parquet` / `official_qe_fit.parquet`
- `flow_tenor_private.parquet`
- `dv01_isin.parquet` / `dv01_tenor.parquet`
- `rv_wam.parquet` / `rv_supply.parquet` / `rv_demand.parquet` / `rv_wam_tenor.parquet`
- `derived/reconcile.json`
- `market_monthly.parquet` — monthly-average 2y/10y/30y par yields per country (complete months
  only) plus 10s30s, 10y vs Bund, 10y vs the UK/DE/US/JP mean and UK 20y gilt − SONIA OIS
  zero-coupon spread (`market.py`). UK par yields are computed from the BoE spot curve
  (continuously compounded; 1.5bp / 3.5bp mean absolute gap to the BoE's published 10y / 20y
  par). FR and IT have only the monthly OECD 10y (no 30y), so the peer mean excludes them.
- `evidence.parquet` — the living metric-vs-price test table (`evidence.py`)
- `issuance_dv01_dashboard.html` — interactive (country / from-year / z-window, MATLAB theme)

## Dashboard layout

The board is an instrument, not commentary. No tab carries a written conclusion; notes say how
to read a chart, and every number in a note is generated at build time.

- **Overview** — one row per country: price (10y, relative measure, 3m change) and the headline
  metrics (`headline.HEADLINE`), each with its percentile in the country's own history since the
  sidebar start year, coloured by *pressure-oriented* percentile only. The row under each
  heading is that metric's evidence verdict. Small multiples: any headline metric vs the 10y.
- **Country** — five fixed questions, same order everywhere: private supply, absorption, demand
  (score/pillars, central-bank share, cover), price, what is coming (next 12m gross coupon DV01 by
  conviction vs the same months last year; next eight weeks of auctions). Level / percentile /
  12m-change switch, 10th–90th percentile band, policy events from `data/events.json`.
- **Demand**, **Calendar & lines**, **Detail ▾** (the older supporting tabs) and **Reference**
  (evidence table, glossary, method). Detail cards whose metric tests as *no link* or *wrong
  sign* in all three windows carry a "supporting" tag set from the evidence table, not by hand.

One metric definition (`headline.py`) feeds both the screens and the tests. `evidence.py` z-scores
each metric within country over 60 months (published z-scores used as is), signs it so + is
pressure, and regresses the price change over the next 3 months after the publication lag
(`headline.lag_for`: 1m supply/auction, 2m score, 2m for DE/FR/IT measures built on the
bimonthly ECB purchase split) on it, controlling for the price change over the 12 months to the
reading (DV01 rises when yields fall, so supply in DV01 partly encodes the price path): pooled,
country-demeaned, every month, Driscoll–Kraay t (overlap and common shocks), the smallest t with
any one country left out, and a chance p — the share of circular rotations of the metric against
price (every shift ≥ 12 months, same shift for every country) with a t at least as large. A
forward link needs β > 0, t ≥ 2, leave-one-out t ≥ 1 and p ≤ 0.10; wrong sign mirrors it. The
verdict tag reads only targets pooling 3+ countries; 10y vs Bund (FR, IT) and UK 20y ASW are
shown, never decisive. Verdicts: *leads price*, *moves with price* (no forward link, same-window
12m correlation ≥ 0.3), *wrong sign*, *no link*. The verdict window is 2010+; 2010–18 and 2019+
rerun the rule on each half and get their own tag where they differ, so a metric that holds in
both halves has passed an out-of-sample check. Re-read the Reference tab after a refresh rather
than quoting results from here. The evidence stage takes ~45s (the rotations).

Three ways the earlier version over-claimed, now closed: quarterly sampling made results depend
on which month started the grid (UK tail t 3.0 / 0.3 / −0.4 across the three phases); a verdict
could rest on a one- or two-country target (cover's "lead" was FR/IT spread only, and Italy
alone — without IT the 10y t is 0.2); and "moves with price" for every quantity metric was the
UK 20y ASW alone.

`priv12` / `gross12` re-express each month's USD DV01 at the latest month's FX before the 12m
sum. DV01 is converted at each auction's spot rate, so at spot FX the yen's fall reads as a cut
in JP supply (mid-2021 private DV01: 780 at spot, 531 at today's rate). The ratio metrics are
FX-neutral already. Detail tabs stay at each operation's spot rate and say so in their notes.

The current calendar month, and any month whose auctions the tape has not caught up with, is
forecast, never history: `forward._fold_remainder` adds the forecast operations still to come in
the last tape month (`fc_remainder_dv01`), which earlier were dropped, so a country with one
auction on the tape in the current month lost the rest of that month from every forward 12m
value. Country-tab forward bars start at the current month: realised so far + still to come,
against the same calendar months a year earlier.

`events.json` is append-only and holds regime changes only (QE/QT starts and ends, YCC, the
mini-budget); do not add a date that is not settled.
