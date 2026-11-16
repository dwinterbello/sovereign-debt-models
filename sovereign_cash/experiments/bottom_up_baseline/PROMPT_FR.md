# Agent brief — FRANCE v2 (PLF/RESF tax & spend × macro)

`_SHARED.md` + this file.

## Target

State deficit from **PLF recettes by tax + dépenses by category**, grown with **RESF macro**, not SMB-line seasonals + AFT charge only.

v1 (`fr/bottom_up.py`) = seas_mult on recettes/primary + AFT charge pin. **Rejected as lazy.** Keep `SHIP=False` or rename; build `fr/bottom_up_macro.py`.

## Authority category sources

| Source | Path / note | Use |
|---|---|---|
| **RESF** | `data/fr/authority.json` macro; extend with RESF receipt/spend growth notes | real, CPI, activity → tax bases |
| **PLF / LFI** équilibre | PLF PDF / budget.gouv tables (hand JSON) | IR, IS, TVA, autres fiscales, non-fiscal; charge dette; crédits ministériels; PSR; comptes spéciaux |
| **AFT** charge | `authority.json` debt_charge | interest annual — driven also by rate assumptions in PLF (`rate_assumptions_plf2026`) |
| SMB monthly | `fr/fetch.load_smb` | month profile / outturn; not the annual engine |
| plan.json | déficit à financer | optional hard close **after** category sum — compete with/without on PRE |

Extend `authority.json` with a `categories` block: year → {ir, is, tva, ...} bn from PLF 2026/2027 tables (numbers already partially in PLF text from prior work: e.g. recettes fiscales nettes, charge 62.6/72.9).

## Identity

```
deficit ≈ Σ dépenses catégories + charge − Σ recettes fiscales − non-fiscal − solde comptes spéciaux
```

Each tax: `level_y = level_{y-1} × (1 + ε · Δmacro)` with ε from RESF elasticity language or expanding OLS of SMB-implied tax proxies on NGDP/wages/CPI where split exists.

## Gates

PRE ≥5% vs production (PLF×seasonal or current live). Macro audit table required.

## Entry

`data/fr/authority.json`, `plan.json`, `fr/bottom_up.py`, `macro_authority.py`, PLF excerpts under agent-tools / aft HTML.
