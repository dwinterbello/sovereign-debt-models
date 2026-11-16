# Agent brief — UK v2 (OBR/HMT category × macro)

`_SHARED.md` + this file.

## Target

Monthly **CGNCR(ex)** from **OBR/HMT category forecasts × determinants**, not ONS-line seasonals.

v1 (`uk/bottom_up.py`) = ANBV/MF6S/MF6T/JW2P with seasonal_add3/smly. **Rejected.** Keep it as reference / `SHIP=False`; build v2 alongside (e.g. `uk/bottom_up_macro.py`).

## Authority category sources (use these)

| Source | Where / how | Categories |
|---|---|---|
| **OBR EFO** receipts | `uk/ready_reckoner.py`, `data/uk/obr_ready_reckoner.xlsx`, EFO tables already scraped into OBR paths under sovereign_cash / projects | Income tax, NICs, VAT, corp tax, other taxes, non-tax |
| **OBR EFO** expenditure | same | debt interest, welfare, DEL/AME splits as published |
| **OBR determinants** | EFO determinants table / RR | wages/employment, consumer spending, equity prices, interest rates, inflation, GDP |
| **DMO/OBR CGNCR** | remit vintages | annual pin / check only — not the shape engine |
| ONS PUSF | monthly outturn | calibrate month profiles and residual; not the forecast rule |

HMT / OBR publish the mapping: Δdeterminant → Δreceipt category (ready reckoner). **Apply that.** Baseline path = last EFO category levels grown with realised + forecast determinants known at origin (BoE/OBR paths already in repo where possible).

## Identity (illustrative — match OBR bridge exactly)

```
CGNCR ≈ −Δ(current receipts) + Δ(current exp incl. debt interest) + Δ(capital) + financial transactions
```

Debt interest: OBR debt-interest forecast / triangle on gilt rate path — **not** JW2P smly.

## Gates

Beat production CGNCR path on PRE by EDGE_BAND (`uk/model.py`). Bills still solve to remit end-March. Do not redesign the ready reckoner file format; **consume** its coefficients for the baseline, not only scenarios.

## Entry

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python -m pytest sovereign_cash/tests/test_ready_reckoner.py -q
# read uk/ready_reckoner.py, data/uk/obr_ready_reckoner.xlsx, uk/model.py, existing bottom_up.py (v1)
```
