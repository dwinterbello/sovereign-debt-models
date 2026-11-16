# Agent brief — GERMANY v2 (BMF Steuerarten × macro + Soll categories)

`_SHARED.md` + this file.

## Target

Core cash need from **tax-by-tax and spending categories** with macro drivers — not two-line Soll×haircut.

v1 (`de/bottom_up.py`) = Ausgaben/Einnahmen Soll × hist ratio. Better than smly but **not category×macro**. Build `de/bottom_up_macro.py`; leave v1 as fallback.

## Authority category sources

| Source | Use |
|---|---|
| **BMF Monatsbericht** | Ist/Soll by Steuerart where printed (Lohnsteuer, USt, KSt, … from 2017+); Ausgaben economic splits if present |
| **Bundeshaushalt Soll** | `data/de/plan.json` + BMF — core vs special funds; push funds separately with their own drivers |
| **Gemeinschaftssteuern / Destatis** | if in `data/de/` or fetchable — bases |
| AMECO/WEO or Destatis | NGDP, wages, CPI as determinants (point-in-time) |
| Debt interest | Finanzagentur / debt book × Bund curve — not buried in Ausgaben smly |

## Identity

```
core deficit ≈ Σ spend categories + interest − Σ tax categories − other revenue
```

Open-year: category Soll (or last Ist grown by macro) × month share. Elasticities: Lohnsteuer–wages, USt–consumption/NGDP, etc., PRE-gated.

## Gates

PRE ≥5% vs current live (v1 or planhc overlay — beat whatever ships today).

## Entry

`de/fetch.py`, `data/de/bmf_mb/`, `de/bottom_up.py`, `yoy_components.py`.
