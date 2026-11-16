# Agent brief — NORWAY v2 (MoF non-oil / Tab4 × macro)

`_SHARED.md` + this file.

## Target

Beyond NB window: **non-petroleum fiscal categories × macro**; inside NB window: stick to NB or explain residual with MoF categories — not Tab4 smly.

v1 NO-SHIP — fine; build `no/bottom_up_macro.py` with real drivers (oil price, mainland GDP, interest).

## Authority category sources

| Source | Use |
|---|---|
| NB Tab4 / Tab1 | component history |
| NB liquidity forecast | pin inside published horizon |
| MoF National Budget / petroleum fund | non-oil deficit, oil cash, if hand-enterable |
| Macro | mainland GDP, oil price (Brent), policy rate / NOWA |

## Gates

Beyond-NB PRE ≥5% vs smly; do not fake wins inside NB vintage window.

## Entry

`no/bottom_up.py`, `no/build.py`, `data/no/`.
