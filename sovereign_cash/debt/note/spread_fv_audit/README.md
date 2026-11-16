# Euro-area spread fair value: audit pack

Self-contained: no repository, credentials or network needed.

    python -m venv .venv && .venv/Scripts/activate      # Windows  (source .venv/bin/activate on Mac/Linux)
    pip install -r requirements.txt
    python spread_fv.py

Writes `outputs/results.json`, `outputs/regression_tables.txt` (HAC standard errors for every specification) and
`outputs/fair_value.png`. Runs in a few seconds.

## Inputs (all public, snapshot 15 Nov 2026)
| File | Source |
|---|---|
| `inputs/ecb_irs_m.csv` | ECB Data Portal, series `IRS.M.<CC>.L.L40.CI.0000.EUR.N.Z` (10y convergence yields, monthly), raw download |
| `inputs/eurostat_gg.csv` | Eurostat (21 Oct 2026): general-government debt, net lending, interest payable, revenue (% GDP), nominal GDP; AT BE DE ES FI FR IE IT NL PT, 1995-2025 |
| `inputs/ameco_gg.csv` | European Commission AMECO, autumn 2026: the same for DE FR IT ES, used for 2026 (observation) and 2026-27 (fair value) |
| `inputs/note_scenarios.json` | The note's scenario grid: 2030 change in interest / revenue and primary deficit |
| `inputs/note_paths.json` | From the note's model: general-government interest / revenue 2026 and 2030, today's primary balance (FR, IT, DE); latest OAT-Bund |

To refresh, re-download the ECB series, the four Eurostat series (`gov_10dd_edpt1` GD/B9/D41PAY, `gov_10a_main` TR, `nama_10_gdp` B1GQ; % of GDP / EUR mn) and the AMECO variables, and overwrite the files; the script does not change.

## Method
Pooled OLS, annual 1999-2026, nine countries against Germany (AT BE ES FI FR IE IT NL PT; n = 246), no country effects. Greece is excluded (2012 restructuring: default, not fiscal, pricing). Each regime
(1999-2007, 2008-09, 2010-12, 2013-21, 2022-26) has its own intercept and slope on the fiscal variable.
Preferred: spread on the gap to Germany in general-government interest / revenue, plus the primary-balance gap.
Fair value = fitted spread under 2022-26 pricing; band = 80% prediction interval.

## Robustness reported
Debt / GDP instead of interest / revenue; own primary balance; debt-stabilising fiscal gap; debt spline above 90% of
GDP; lagged fundamentals (reverse causality); post-2013 sample with year effects; leave-one-country-out;
country-clustered standard errors; residual autocorrelation; the 2025 cross-section of actual against fair value.
Also: the note's scenarios run through the model (change in equilibrium, bp) and a market-implied probability of
crisis-regime pricing (an illustrative two-state mixture).

## Known limitations
- Nine countries, but the cross-section is still modest and the 80% bands are wide (France 2026: roughly 0-155bp).
- Ireland's GDP is distorted by multinationals since 2015; interest / revenue (the preferred variable) is unaffected.
- The earlier three-country version (France, Italy, Spain only) gives a lower equilibrium (~60-80bp) and milder
  crisis pricing (~210bp); the market-implied crisis probability is sensitive to that choice (~15-25% here, ~30-45%
  there).
- Regime dates are set ex post; the 2022-26 regime has 15 observations.
- 2026-27 fiscal values are Commission forecasts; 2026 spreads are a January-October average.
- Residual autocorrelation (0.2-0.6) means the plain-OLS prediction bands are somewhat too narrow.
- The split of today's OAT-Bund into equilibrium, euro-area common move and France-specific premium is a
  back-of-envelope calculation (labelled in results.json), not a regression output.
