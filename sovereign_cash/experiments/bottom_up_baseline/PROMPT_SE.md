# Agent brief — SWEDEN v2 (Budgetpropositionen / RG categories × macro)

`_SHARED.md` + this file.

## Target

NBR from **primary balance + net lending + interest** with **macro drivers**, not three independent seasonals under an RG pin.

v1 NO-SHIP on anchored PRE — OK to keep RG annual pin, but **shape and open-year levels** must come from category×macro. Build `se/bottom_up_macro.py`.

## Authority category sources

| Source | Use |
|---|---|
| Riksgälden CH1089 | primary / net lending / interest history |
| **Budgetpropositionen / Spring Fiscal Policy Bill** | tax and expenditure forecasts; hand JSON into `data/se/authority.json` |
| Statsupplåning | NBR annual pin after category sum |
| Macro | SCB NGDP, CPI, wages; policy rate for interest |

Interest: stock × rate path, not seasonal twin of primary.

## Gates

PRE anchored MAE ≥5% better than current RG×seasonal production, or NO-SHIP with category table.

## Entry

`se/bottom_up.py`, `se/build.py`, `data/se/authority.json`.
