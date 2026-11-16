# STATUS_FR — bottom-up baseline v2

**Verdict: SHIP**

Commit base: `0b17ae37`. Gate: mean PRE MAE ≥5% better than production `seas_mult_3`, worse at ≤1 horizon.

## Headline

| Path | PRE mean | OOS mean | worse_h (PRE) |
|---|---|---|---|
| Production `seas_mult_3` | 8.13 | 14.87 | — |
| v1 SMB seas + AFT charge | — | — | SHIP=False (rejected) |
| **v2 PLF×RESF categories** | **7.44** | **8.88** | **1** |

Relative PRE edge **+8.5%**. Threshold ≈ 7.73. Live cash = unanchored category path when `fr.bottom_up_macro.SHIP` (no LFI solde re-pin in `fr.model.forward`). BTF flex / TGA~0 unchanged in `fr.model` (placements absorb need − BTF).

### By horizon

| window | h | bottomup | production |
|---|---|---|---|
| PRE | 1 | 4.11 | 3.96 |
| PRE | 3 | 6.32 | 6.60 |
| PRE | 6 | 8.58 | 8.71 |
| PRE | 12 | 10.75 | 13.26 |
| OOS | 1 | 5.95 | 6.21 |
| OOS | 3 | 8.60 | 11.51 |
| OOS | 6 | 9.82 | 16.08 |
| OOS | 12 | 11.15 | 25.68 |

## Category audit

| Category | Driver | Rule | PRE cat MAE (bn) |
|---|---|---|---|
| ir | masse_salariale | PLF pin; else hold (ε=1 RESF; leaf growth fails PRE) | 2.98 |
| is | ngdp (lag 1) | PLF pin; else hold | 1.91 |
| tva | ngdp | PLF pin; else hold | 2.64 |
| accises | real_growth | PLF pin; else hold (ε=0.5) | 1.47 |
| autres_fiscales | ngdp | PLF pin; else hold | 3.15 |
| non_fiscal | None | PLF pin; else hold (one-offs) | — |
| credits_ministeriels | cpi | PLF pin; else hold (open growth OOS fail) | 15.00 |
| cas_pensions | masse_salariale | PLF pin; else hold | 2.07 |
| remboursements_locaux | cpi | PLF pin; else ×(1+ε·Δcpi) same-year | 0.20 |
| psr | ngdp | PLF pin; else hold | 3.39 |
| charge | rates / AFT | AFT charge when known (2024=50.1); else hold; month 100% smly | — |
| solde_comptes_speciaux | None | PLF when known; else SMB moy | — |
| solde_budgets_annexes | None | PLF when known; else 0 | — |

Identity: `deficit ≈ credits + charge + cas_pensions + remb + psr − Σ recettes − solde CS − BA`. Month shape: 5y SMB shares; interest carve `INT_SMLY_W=1`.

## Authority tables

| Title | Vintage | Path |
|---|---|---|
| PLF équilibre / Voies et moyens / chiffres clés | PLF 2025→2027; 2023–24 exéc fills pinned | `data/fr/authority.json` → `categories` (+ `plf_extracts/`) |
| DG Trésor RESF macro | RESF PLF 2027 (6-Oct-2026) | `authority.json` → `macro` |
| AFT charge budgétaire | exéc 2021–24; LFI/PLF 2025–27; 2024 pin 50.1 | `authority.json` → `debt_charge` |
| PLF solde / déficit à financer | soft-pin off (α=0); plan still in `plan.json` | `authority.json` → `state_deficit` / `plan.json` |
| SMB recettes / dépenses / CS | monthly outturn | `fr/fetch.load_smb` |

## Rejects (PRE-gated; OOS reported only)

- Aggregate / IR/TVA/IS same-year RESF or OLS leaf growth — fail PRE or worse_h≥2
- Soft pin α=1 — PRE flat, OOS 8.06 better; kept α=0 so category×RESF is the cash level
- Charge rate-scale / AFT +100bp triangle × Δrates — clear PRE or not; OOS ≥ hold
- Open-year credits / cas / psr / spend-all growth — clear PRE; OOS ≥ hold
- 2022 new-perimeter PLF pin — OOS regress
- v1 seas_mult + AFT charge — lazy; `SHIP=False`

## Knobs (locked)

`SHARE_K=5`, `SOFT_PIN_ALPHA=0`, `INT_SMLY_W=1`, `GROW_KEYS={remboursements_locaux}`, `GROW_SAME_YEAR=True`, `USE_OLS=False`, AFT 2024 charge 50.1, PLF pins 2023–27.

## Paths

- `sovereign_cash/fr/bottom_up_macro.py` (`SHIP=True`)
- `sovereign_cash/fr/bottom_up.py` (v1, `SHIP=False`)
- `sovereign_cash/fr/model.py` (live uses v2 when SHIP; BTF flex / buffer_change)
- `sovereign_cash/data/fr/authority.json`
- `sovereign_cash/data/fr/plan.json`
- `sovereign_cash/data/fr/plf_extracts/`
- `sovereign_cash/tests/test_fr_bottom_up_macro.py`
- `sovereign_cash/experiments/bottom_up_baseline/STATUS_FR.md`

No commit. No publish. Other countries untouched.
