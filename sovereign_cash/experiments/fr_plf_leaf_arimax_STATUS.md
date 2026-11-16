# FR PLF-leaf / transform ARIMAX screen (2026-10-27)

**Verdict: none ship.** Gate = mean PRE MAE ≥5% better than `seas_mult_3`, `worse_h ≤ 1`.

Script: `sovereign_cash/experiments/fr_plf_leaf_arimax.py`  
Equity: FRED `SPASTT01FRM661N` → `data/fr/equity_fred.json`  
CPI: Eurostat FR HICP YoY (desk inflation path = realised HICP in PRE; no EA nowcast in PRE window).  
Leaves: synthetic monthly = SMB recettes/primary × PLF structural shares (INSEE has no IR/IS/TVA monthly).

## Aggregate SMB SARIMAX

| Spec | PRE | vs seas | worse_h | OOS |
|---|---|---|---|---|
| level \| noexog | **6.96** | −2.7% | 2 | 12.2 |
| level \| equity+hicp lag6 | 7.18 | −5.9% | 1 | 10.5 |
| level \| ngdp+hicp lag2 | 8.62 | −27% | 3 | 11.1 |
| level \| equity+hicp lag2 | 9.51 | −40% | 3 | 10.7 |
| mom / diff / qoq (any exog) | 18–40 | −170…−500% | 4 | blow-up |
| yoy (any) | ~10 | −50% | 4 | blow-up |

seas PRE ≈ 6.78. Closest macro-linked is equity lag-6; still worse than seasonal and fails the 5% edge.

## PLF-leaf monthly SARIMAX (eq→IS, CPI on spend)

| Spec | PRE | vs seas | worse_h | OOS |
|---|---|---|---|---|
| level eq→IS lag6 | 7.34 | −8.2% | 1 | 13.1 |
| level all_cpi lag2 | 7.56 | −11.5% | 1 | 14.0 |
| level eq→IS + cpi rest lag2 | 7.77 | −14.6% | 1 | 12.0 |
| level eq→IS+cpi lag2 | 8.49 | −25% | 3 | 10.7 |
| mom / qoq leaves | 28–38 | −300…−460% | 4 | blow-up |
| yoy leaves | ~9–10 | −40…−50% | 4 | 15–24 |

## Takeaway

Transforms are not the missing fix: MoM / QoQ / YoY / diff make FR cash much worse. Level SARIMAX without exog almost ties seasonals (−2.7%) and still fails the gate. Desk equity → IS and desk CPI on leaves do not clear PRE; best equity lag (6m) is still worse than `seas_mult_3`. Macro-everywhere on current year is not supported by this screen — keep PLF pin + seasonal months; leave leaf ARIMAX off.
