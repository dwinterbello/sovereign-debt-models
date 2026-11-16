# FR seasonal × macro augment (2026-10-27)

**Verdict: CPI-on-primary clears PRE and improves OOS; equity only as IS-share scale; do not replace seasonals with ARIMAX.**

Scripts: `fr_seas_macro_augment.py` (+ inline IS-share / joint grids).  
Equity: FRED `SPASTT01FRM661N` / `…EZ…` / `…DEM…` in `data/fr/equity_{fred,ez_fred,de_fred}.json`.  
CPI: Eurostat FR HICP YoY. Gate vs `seas_mult_3` (≥5% PRE, worse_h≤1).

## What works (UK-style: calendar × (1+ε·driver))

| Spec | PRE vs seas | OOS (seas≈13.8) |
|---|---|---|
| primary × (1+0.5·HICP) lag0/1 | +7.9% / +7.3% | **12.2** |
| + recettes × (1+0.17·0.75·eq) lag1 | | |
| … eq_de | +8.9% | **9.9** |
| … eq_ez | +7.2% | **10.2** |
| … eq_eu | +7.6% | **10.3** |
| … eq_fr | +6.6% | 10.9 |

CPI on **primary** (spend) is the direct indexed channel. Mild equity on the **IS weight (~17%) of recettes** helps OOS; DE/EZ beat FR-only.

## What fails

- Scaling **all** recettes by equity (any geography, ε≥0.25) — PRE −15% to −200%.
- IS-share equity **alone** (no CPI) — fails PRE.
- Residual `seas + β·driver` overlays — <5% edge.
- Replacing seasonals with leaf ARIMAX / MoM / QoQ (prior experiment).

## Why prior “macro” looked dead

Annual PLF-leaf growers and full-basket SARIMAX **replaced** the seasonal. The link is an **overlay on the seasonal**, not a substitute. CPI→tax was the wrong leaf; CPI→primary spend is the right one.

## Receipts × CPI (2026-10-27 follow-up)

Hypothesis: nominal recettes should track CPI. Tested as seasonal × (1+ε·HICP).

| Spec | PRE vs seas | Gate |
|---|---|---|
| recettes × CPI alone (ε 0.25–1.5, lags 0–12) | best ~+4.7% (ε=0.25 lag12) | **fail** (need ≥5%, worse_h≤1) |
| TVA-share only × CPI (ε 0.5–2) | best ~+4.4% | **fail** |
| recettes × NGDP alone | ≤+0.3% | **fail** |
| recettes×CPI ε=0.25–0.5 **+** primary×CPI ε=0.5 | +5–7% | **clear** |
| above + IS-share × eq_de ε=0.75 lag1 | +6.5%, OOS **9.96** | **clear** |

Receipts take a **smaller** CPI loading than primary (≈0.25 vs 0.5). Full 1:1 CPI on recettes alone does not clear. Deficit identity: scaling both sides by the same ε partly offsets; asymmetric ε is what survives.

## Candidate (not wired)

`seas_mult_3` components:
- `primary *= (1 + 0.5 · HICP_lag1 / 100)`
- `recettes *= (1 + 0.25 · HICP_lag1 / 100 + 0.17 · 0.75 · eq_DE_lag1 / 100)`
interest/speciaux unchanged. Soft-pin to PLF still on top in live BUM — pin mutes annual level; overlay keeps month shape / open years. Wire only after a BUM-path score (not bare seas) passes the same gate.
