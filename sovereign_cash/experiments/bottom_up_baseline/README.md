# Bottom-up baseline v2 — macro + published categories

v1 (seasonal/smly per line) is **rejected**. Agents use `_SHARED.md` (v2) + `PROMPT_<CC>.md`.

| Agent | Must use |
|---|---|
| UK | OBR EFO / ready reckoner categories × determinants |
| FR | PLF tax/spend categories × RESF macro |
| DE | BMF Steuerarten / Soll categories × macro |
| IT | DFP/BdI categories × macro |
| SE | Budgetprop / CH1089 × macro (RG pin OK after) |
| NO | MoF/Tab4 × oil & mainland GDP |

Implement `*_bottom_up_macro.py`; leave v1 in place with SHIP=False.
