# Scenario layer

Shock any assumption (nominal GDP, inflation, bond yields, Brent) for nine sovereigns
and see every number it moves, with the formulas on a page next to it.

```
models + authority tables ──► scenarios/pack.py ──► derived/scenario_pack.json
                                                        │
                    scenarios/engine.py  ◄── same arithmetic ──►  web/engine.js
                          │                                          │
                 tests (parity, checks)                  shock_dashboard.html, methodology.html
```

| Piece | What it does |
|---|---|
| `channels/sarimax.py` | FR, IT: components as a share of the GDP / price level; DE: on YoY rates (`monthly_components.LEVEL_SPEC`, chosen by `experiments/level_spec_backtest.py`) |
| `channels/obr.py` | UK: OBR ready reckoner, shock-year × effect-year coefficients |
| `channels/cbo.py` | US: CBO rules of thumb (sustained shocks only) |
| `channels/annual.py` | ES, SE, NO, JP: elasticities on general-government lines |
| `cyclical.py` | the note's growth scenarios line by line: US federal DTS lines on their drivers (wages, equities, profits net of the AI-capex expensing cushion, imports, unemployment, COLA); UK on the OBR ready reckoner line by line; DE, FR, IT, ES, JP on OECD output-gap elasticities by tax and for benefits (Price, Dang & Botev 2015). Used by `debt.note.scenario_grid` and the ex6 downturn line |
| `channels/us_dts_adjust.py` | FY2026 one-offs out of the US DTS forecast years (IEEPA tariff refunds, refund catch-up) and corporate, customs, withheld, non-withheld on drivers; applied in `us_dts.fy_table()` |
| `channels/debt_book.py` | all nine: ISIN-book interest at −200…+300bp, parallel or anchored at 3m and 10y |
| `curve_pca.py` | curve shapes: every maturity's expected move given the 3m and 10y points (covariance of monthly changes since 2000) |
| `engine.py` / `web/engine.js` | `evaluate(pack, cc, shocks)`, pure arithmetic, line-for-line twins |
| `pack.py` | builds the pack, then `check()` re-runs the real models against the engine |
| `unit_shocks.py` | the published one-at-a-time tables on the country boards |
| `../equations/` | the shock formulas (registry), the page helpers and `shock_sections` of the Methodology page (`../methodology/`) |
| `../web/pages.py` | writes the two pages, pack and engine inlined |

## Run

```bash
PYTHONPATH=. .venv/bin/python -m sovereign_cash.scenarios.curve_pca   # curve factors (weekly is plenty)
PYTHONPATH=. .venv/bin/python -m sovereign_cash.scenarios.pack   # pack + checks (~8 min)
PYTHONPATH=. .venv/bin/python -m sovereign_cash.web.pages        # the two pages
PYTHONPATH=. .venv/bin/python -m pytest sovereign_cash/tests/test_engine_parity.py sovereign_cash/tests/test_scenario_pack.py
```

`sovereign_cash.run_all` does all of it nightly. The pack refuses to hide failures:
a country that fails is listed in `errors` and shown in red on the page.

## Rules

- Change `engine.py` and `web/engine.js` together. `test_engine_parity.py` holds them to 1e-9.
- A new channel needs `method`, `source` and `badge`; the Methodology page prints them.
- Never shock history. Shocks start the month after the build (`channels/common.default_onset`);
  published rules (AFT, CBO, OBR) are moved to the same start so they compare like for like.
- Interest on extra borrowing is one switch (`feedback`) for every country: OBR's CGNCR loop and
  CBO's debt-service lines are left out of the published channels so it is never counted twice.
- An elasticity with no fitted model behind it is badged `Assumed elasticity`.
- Refactors are checked with `tests/golden/compare.py` (0 differences above 1e-9).

## Adding a country

Write `build(book, onset) -> dict` returning the entry shape the other builders return
(`periods`, `lines` with signs, `baseline`, `channels`, `default_methods`, `modes`), register
it in `pack.builders`, and add it to `pack.ORDER`. The pack tests check the shape.
