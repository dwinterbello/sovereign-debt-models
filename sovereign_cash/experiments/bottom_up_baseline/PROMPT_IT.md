# Agent brief — ITALY v2 (DFP/BdI categories × macro)

`_SHARED.md` + this file.

## Target

Fabbisogno from **DFP/MEF revenue and expenditure categories × macro** — not BdI-line smly + residual.

v1 almost tied smly (0.6% PRE) and failed OOS. **Rejected.** Build `it/bottom_up_macro.py`.

## Authority category sources

| Source | Use |
|---|---|
| **DFP** Tavole | fabbisogno; GDP (II.1.2.2); tax revenue paths; interest; if PDF tables list IRPEF/IRES/IVA — hand-enter |
| **BdI Tavola 1** | monthly tax / other / current / capital / treasury — calibrate profiles |
| **MEF** monthly fabbisogno | target print + residual only after categories exhaust |
| Macro | DFP NGDP, WEO/HICP, effective rates (`interest_split`) |
| RRF | keep grant adjustment |

Drop OOS+10% hard fail if PRE clears 5% **and** OOS is not >+25% (align with FR/DE/UK practice); still report OOS. Prefer specs that help both.

## Identity

Map DFP annual categories → monthly via BdI shares; grow with DFP determinants / expanding β on NGDP/CPI/rates per category.

## Gates

PRE ≥5% vs plan×smly production; ≤1 worse horizon.

## Entry

`it/events.py`, `it/bottom_up.py`, BdI T1 parquet, `monthly_components.py`.
