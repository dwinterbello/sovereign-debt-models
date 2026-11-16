# Equity × corporate income tax (DTS `dep_corporate`)

## Verdict: **no** — do not ship

UK-style positive equity → corporate does not clear a material OOS bar. Production
stays trailing 250d mean × HGB calendar (`choice=model`). No change to
`fy_overlay` / `fiscal_path`.

Corporate daily MAE ($bn), monthly origins, windows = `flow_model.WINDOWS`:

| | PRE (2022–23) | OOS (2024+) | ΔPRE | ΔOOS |
|---|---:|---:|---:|---:|
| baseline | 0.7375 | 0.7338 | — | — |
| best UK-style: surprise ε=**0.25**, lag **12** | 0.7342 | 0.7312 | −0.0033 | −0.0026 (−0.35%) |
| next UK-style: surprise ε=1.0, lag 3 | 0.7673 | 0.7218 | +0.0298 | −0.0120 (−1.6%) |
| best fade (wrong sign): scale ε=**−0.5**, lag **0** | 0.7270 | 0.7037 | −0.0105 | −0.0301 (−4.1%) |
| resid OLS on equity YoY L0 | 0.7635 | 0.7792 | +0.026 | +0.045 |

Ship rule: positive ε, OOS better, PRE ≤ baseline×1.05, OOS cut ≥2% relative.
`clear_positive_count=0`. Best ε/lag for the *thesis* would be surprise **0.25 / 12**
(immaterial) or **1.0 / 3** (PRE +4%, OOS −1.6% — under the bar).

## How corporate is forecast today

- **Scored / live daily path** (`flow_model.py`): `dep_corporate` = trailing
  250-business-day mean × HGB calendar shape. PRE choice = `model` (beats flat;
  not a spike candidate). Calendar includes `dist_corp` to Apr/Jun/Sep/Dec
  estimated-payment deadlines.
- **Multi-year path** (`fy_overlay.apply_fy` via `fiscal_path.py`): same shape;
  level stepped by trailing 250d growth. COLA only on SSA/VA/military/OPM —
  **corporate has no equity driver**.
- UK analogue (`sovereign_cash/uk/bottom_up_macro.py`): `other_tax` × retail YoY +
  equity YoY (FRED `SPASTT01GBM661N`, ε=0.7 on equity).

## Experiment

- Equity: FRED `SP500` daily → monthly mean → YoY % (cached
  `us/fiscal/data/cache/fred_sp500.csv`; through 2026-10, last ~14.9%).
  `WILL5000PR` 404 on FRED CSV; `SPASTT01USM661N` available via `--code`.
- At origin *o* only month-ends ≤ *o* (minus lag) — no lookahead.
- Overlays (not production): **scale** `ŷ×(1+ε·yoy/100)`; **surprise** with
  `(yoy − prior-12m mean YoY)`; **resid_ols_L0** expanding OLS of monthly miss
  on equity YoY (warmed from 2018). ε ∈ {−1…+2}, lags 0/1/3/6/12.

## Read of the signs

- Raw **ε>0**: every cell loses OOS — trailing mean already embeds recent equity.
- Raw **ε<0**: OOS improves (ε=−0.5 lag 0 also helps PRE). That is a fade of
  equity into the mean, not an equity→tax link; do not promote under this thesis.
- **Surprise ε>0**: small OOS cuts only; none ≥2% relative.
- Residual OLS: worse on both windows.
- Net op-flow / TGA not re-scored: corporate MAE moves of ≤4% are below headline
  op-flow noise (`flow_model_score.json` h=5 OOS ~18 vs flat ~65).

## Reproduce

```bash
cd us/fiscal && ../../.venv/bin/python experiments/equity_corporate_ab.py --write-md
```

- Script: `us/fiscal/experiments/equity_corporate_ab.py`
- Note: `us/fiscal/experiments/equity_corporate_tax.md`
- Artifacts: `us/fiscal/derived/equity_corporate_ab.json`,
  `equity_corporate_ab_detail.parquet`
