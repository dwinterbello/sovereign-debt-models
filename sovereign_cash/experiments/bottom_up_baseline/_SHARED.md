# Bottom-up baseline — shared contract (v2: macro + published categories)

Paste this **above** the country brief.

---

## Mission

Build a **macro-driven, category-level BASELINE** for monthly cash requirement that can replace today's live path.

**Forbidden as the main engine:** same-month-last-year on the headline, trailing seasonals on the headline, or “seasonal on each line + residual smly” dressed up as bottom-up. That was v1 and it is rejected.

**Required:**

1. **Published category forecasts** from the treasury / budget office / OBR / fiscal council wherever they exist (tax by tax, spending by programme / economic category). Hand-enter annuals into `data/<cc>/authority.json` (or country equivalent) with vintage dates. Do not invent categories the agency does not publish.
2. **Macro determinants** that the agency itself uses (or that score on PRE): NGDP / wages / employment / CPI / consumption / corporate profits / oil / policy rate / yield — as published in their ready reckoner, RESF, EFO determinants, DFP, etc.
3. **Identity:** categories → cash requirement via the official accounting bridge. Residual only for the part of the perimeter you cannot map; residual must not be >~30% of MAE on history or you have not gone deep enough.
4. Month shape: calendar / seasonal on **each category** after the annual (or quarterly) macro path is set — not a single seasonal on the deficit.

## Hard gates

1. **PRE selection only.** Report OOS; never pick on OOS. Country PRE/OOS windows as in existing model modules.
2. **Identity** on history and forecast months (existing residual tolerance).
3. **No lookahead.** Authority vintages and macro prints known at origin *t* only.
4. **Ship rule:** mean PRE MAE across horizons beats **current production** by ≥5% relative and worse at ≤1 horizon. If you cannot beat production with honest macro categories, return **NO-SHIP** with the table — do **not** ship a seasonal rebrand.
5. **Macro audit:** every shipped category forecast must name (a) the official series or hand JSON key, (b) the macro driver(s), (c) the elasticity / rule. A category with driver `None` / smly-only needs a one-line justification (e.g. discrete one-off, no determinant published).
6. No commit / email / full deploy / port 8770. No demand/habitat / Curve RV / inflation MAE digressions.

## What v1 did wrong (do not repeat)

- FR/DE/UK/IT “bottom_up” that is mostly `seas_*` / `smly` per line.
- Calling AFT charge or Soll a “model” without tax/spend category × macro paths.
- Residual smly carrying the MEF/ONS perimeter instead of mapping more categories.
- Rejecting GDP/CPI pass-through once at the **aggregate** and never retrying at **tax-by-tax / spending-by-category** with the agency’s own elasticities.

## Method

1. Inventory every **category forecast** the authority publishes (PDFs/XLSX/APIs already in repo or fetchable). Examples: OBR EFO receipts/expenditure tables + determinants; HMT PSF; PLF recettes nettes by impôt; BMF Steuerarten; DFP Tavole; Riksgälden/Budgetpropositionen lines; NB/MoF non-oil.
2. Map each category → monthly cash using published month profiles or estimated shares **conditional on the annual macro path**.
3. Fit or apply elasticities: prefer **published** ready-reckoner / RESF / OBR coefficients; else expanding OLS of category YoY on the determinant(s) the authority cites, PRE-gated.
4. Interest: stock × effective rate or agency charge path driven by **yield / policy path**, not smly alone.
5. Score category MAE and headline cash MAE on PRE; ship only if headline clears the gate.
6. Implement as `cash_requirement_bottomup_bn` first; replace live only on SHIP. Keep v1 modules as `*_v1` or behind `SHIP=False` — do not delete scores.

## Deliverables

- Code + tests.
- Table: category → driver(s) → rule → PRE category MAE.
- Headline PRE/OOS vs production.
- SHIP / NO-SHIP.
- List of authority tables used (title, vintage, file path).

## Repo

`PYTHONPATH=. .venv/bin/python`, `OMP_NUM_THREADS=1`. Read country `model.py` and existing `data/<cc>/`. Absolute Mode.
