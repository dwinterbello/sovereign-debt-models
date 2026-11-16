# Sovereign cash and DM debt models

The sovereign-cash framework with its data: country cash-requirement models, the debt board, scenarios, debt paths,
issuance and absorption, US fiscal, and a spread fair-value model, plus the Sovereign financing dashboard pages. No
credentials are needed.

## Set up

    python -m venv .venv
    .venv\Scripts\activate            # Windows   (source .venv/bin/activate on Mac / Linux)
    pip install -r requirements.txt
    set PYTHONPATH=.                  # Windows   (export PYTHONPATH=. on Mac / Linux)

Python 3.12. Run every command from this folder.

## Data: one pull, one zip

When the machine running the models has no internet, data is pulled on another machine with this same folder:

    python pull_data.py               # every source -> data_pull_<yyyymmdd_hhmm>.zip (data files only, no code)
    python pull_data.py --list        # the steps; --only <step ...> runs a subset; --full zips all data files

Carry the zip across, then:

    python pull_data.py --unpack data_pull_<yyyymmdd_hhmm>.zip

A source that fails keeps its previous file; the end-of-pull summary (ok / partial / FAILED per step) and the zip's
`pull_manifest.json` say which. The folder already holds data as of 15 November 2026, so the models run before any pull.

## Run the models (offline)

    python -m sovereign_cash.run_all                  # country cash models, debt board, scenarios, dashboards
    python us/fiscal/run_refresh.py                   # US daily cash, TGA, supply bridge, debt limit, US pages
    python sovereign_cash/debt/note/spread_fv_audit/spread_fv.py   # spread fair value (own README)

Single pieces: `python -m sovereign_cash.run_all --only uk fr`, or a country's `sovereign_cash.<cc>.run`. Outputs go
to `sovereign_cash/derived/`, `us/fiscal/derived/`, `issuance/derived/` and HTML pages in `sovereign_cash/`.

The issuance downloads run in the pull. To rerun only the issuance models offline (ISIN flows, reconciliation,
official demand, DV01, RV, open lines and the forward book, float, imbalance, scenarios):

    python -m issuance.run_refresh --only flows reconcile official dv01 rv lines float imbalance

## Dashboard pages

    python serve_site.py              # http://127.0.0.1:8765/ in the browser
    python serve_site.py --build      # rebuild every page from the current data first (offline), then serve

The Sovereign financing pages in one tabbed hub: overview, shock scenarios, methodology, US cash / flows / debt ceiling,
cash by country (UK, DE, FR, IT, ES, SE, NO, FYTD) and debt stock. The server works without internet: it serves the
Plotly charting library from `site_assets/` instead of the CDN the pages name.

## What is where

| Folder | What it holds |
|---|---|
| `sovereign_cash/<cc>/` | Country cash-requirement models (uk, de, fr, it, es, se, no; US in `us/fiscal`) |
| `sovereign_cash/scenarios/` | Primary-balance engine, the cyclical line model, rates / growth / inflation channels |
| `sovereign_cash/debt/` | Debt board: ISIN-level interest book, debt paths, absorption, signal, sources |
| `sovereign_cash/debt/note/` | Debt paths, sustain engine, scenario grid, rate and repression shocks, monthly primary (ES, JP), FYTD tracking, peer basis, US lines, absorption stack, spread fair value |
| `sovereign_cash/methodology/` | Every formula with its code path |
| `issuance/` | Bond-level issuance pipeline: fetchers, ISIN flows, DV01, holders, forward book, float, market yields |
| `us/fiscal/` | US daily cash flows, TGA, supply bridge, debt limit, money markets |
| `correlations/g4/sources.py` | Public curve feeds (Bundesbank, MoF, ECB, BoJ, FRED, Fed GSW) |
| `uk_flows/` | DMO and ONS downloaders (`dmo.py`, `ons.py`) and their caches; UK gilt data files |
| `serve_site.py`, `site_assets/` | Local server for the dashboard pages; the Plotly bundles they use |
| `projects/uk_long_end/workbook/`, `flow/output/` | Data files |

## Fixed inputs (not refreshed by the pull)

| Input | File |
|---|---|
| House CPI paths | `sovereign_cash/derived/own_inflation_paths.json`, `own_inflation_history_frozen.json` |
| UK RPI all items | `sovereign_cash/_vendor_data/rpi_all_items.csv` |
| BoE gilt curves, priced DMO operations, UK rates | `uk_flows/output/*.parquet` |
| UK long-end tables, NY Fed dealer positions | `projects/uk_long_end/workbook/`, `flow/output/` |

Small shared helpers live in `sovereign_cash/_vendor.py` (`dk_ols`, `parse_coupon`, the RPI reader) and
`issuance/_td_events.py` (TreasuryDirect auction closes).

## Checks

Run offline in a fresh environment holding only `requirements.txt`:
- all 274 modules import;
- every stage of `sovereign_cash.run_all` runs: the UK, DE, FR, IT, SE and NO cash models, the debt board, debt
  page, signal, FYTD, methods, the scenario pack for all nine countries (with its checks), the scenario pages, the
  hub and financing dashboards;
- `us/fiscal/run_refresh.py` completes; the spread fair-value model reproduces its published numbers;
- the issuance model stages (ISIN flows to the forward book, float and imbalance) run with every input present;
- `pull_data.py`: every step launches, a blocked source is reported and keeps its file, the zip builds and unpacks.
  The ECB 10y yields, Eurostat and the ONS series of the macro panel were checked against the live sites.

On a first pull, read the summary: the `curves` step (FRED, ECB, MoF, BoJ, Fed GSW feeds behind `issuance/market.py`)
is the one least exercised.
