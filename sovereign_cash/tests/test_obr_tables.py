"""OBR EFO line parser. Fixtures are the wrapped rows from the March 2026 PDF."""

import pandas as pd

from sovereign_cash.uk import model as M
from sovereign_cash.uk.obr_tables import cgncr_ex_nr, parse_pages, stitch

A5 = """
Table A.5: Current receipts
2024-25 2025-26 2026-27 2027-28 2028-29 2029-30 2030-31
Income tax1 305.9 331.4 359.6 384.6 395.6 413.8 430.8
of which: Pay as you earn 262.1 280.4 296.3 310.2 324.3 343.5 360.1
Self assessment 48.2 55.9 64.4 71.1 74.7 76.3 77.0
Other income tax -4.4 -4.8 -1.0 3.4 -3.4 -6.0 -6.3
National Insurance contributions 171.4 204.0 211.6 218.9 226.5 237.6 245.8
Value added tax 173.3 180.9 188.7 197.8 205.8 214.7 223.8
Corporation tax2 93.2 98.8 102.5 109.1 113.4 117.8 123.2
"""

A7 = """
Table A.7: Total managed expenditure
2024-25 2025-26 2026-27 2027-28 2028-29 2029-30 2030-31
PSCE in RDEL 493.8 522.9 548.2 563.7 581.8 594.2 614.2
of which:
Welfare spending 314.8 332.9 352.8 363.1 373.8 389.9 406.9
Central government debt interest, net of
105.7 109.7 109.4 116.7 125.9 134.7 137.1
APF1
"""

T54 = """
Table 5.4: Reconciliation of PSNB and CGNCR
2024-25 2025-26 2026-27 2027-28 2028-29 2029-30 2030-31
Public sector borrowing (a) 152.7 132.7 115.5 96.5 86.0 63.4 59.0
Financial transactions (b) -79.9 -24.7 -0.7 45.2 54.0 30.6 30.7
Public sector net cash
72.9 108.0 114.8 141.7 140.0 94.0 89.7
requirement (PSNCR) (a+b)
Less local authorities and public
-107.3 -37.2 -23.4 2.6 0.3 -0.6 -16.7
corporations NCR (c)
CG net cash requirement
180.2 145.2 138.2 139.1 139.7 94.5 106.4
(CGNCR) (a+b-c)
Less Network Rail NCR (d) -0.3 0.6 1.0 -6.0 1.3 1.3 0.5
CGNCR ex Network Rail
180.5 144.7 137.2 145.1 138.4 93.2 105.9
(a+b-c-d)
"""

A3 = """
Table A.3: Determinants of the fiscal forecast
2024-25 2025-26 2026-27 2027-28 2028-29 2029-30 2030-31 forecast
Wages and salaries 7.1 6.1 3.2 2.9 2.8 2.8 2.9 22.5
RPI 3.3 4.2 2.9 2.9 2.8 2.9 2.1 19.2
CPI 2.4 3.4 2.0 1.9 2.0 2.0 2.0 14.2
Average weekly earnings5 5.6 4.6 3.2 2.2 2.1 2.3 2.4 18.0
Bank Rate (per cent) 4.9 4.0 3.4 3.6 3.7 3.9 4.0 -0.9
Market gilt rates (per cent)9 4.3 4.4 4.5 4.8 5.0 5.2 5.4 1.1
"""


def test_wrapped_bridge_rows_stay_on_their_labels():
    labels = [lab for lab, _ in stitch(T54)]
    assert "Public sector net cash" in labels
    assert "Less local authorities and public" in labels
    assert not any("PSNCR" in lab and "local" in lab.lower() for lab in labels)


def test_parse_matches_the_remit_and_drops_the_growth_column():
    d = parse_pages([A5, A7, T54, A3], "OBR EFO March 2026")
    ex = d[(d["var"] == "cgncr_ex_nr_bn") & (d["year"] == 2026)]["value"].iloc[0]
    assert ex == 137.2
    assert d[(d["var"] == "cgncr_ex_nr_bn") & (d["year"] == 2029)]["value"].iloc[0] == 93.2
    assert d[(d["var"] == "debt_interest_bn") & (d["year"] == 2026)]["value"].iloc[0] == 109.4
    assert d[(d["var"] == "income_tax_bn") & (d["year"] == 2030)]["value"].iloc[0] == 430.8
    paye = d[(d["label"] == "Pay as you earn") & (d["year"] == 2024)]["value"].iloc[0]
    assert paye == 262.1
    assert d[(d["label"] == "Self assessment") & (d["year"] == 2024)]["value"].iloc[0] == 48.2
    bank = d[(d["var"] == "bank_rate_pct") & (d["year"] == 2030)]["value"].iloc[0]
    assert bank == 4.0  # the 8th column is the growth total, not a year
    assert set(d["var"]) >= {"wages_pct", "cpi_pct", "rpi_pct", "gilt_rate_pct", "welfare_bn", "rdel_bn"}


def test_remit_year_wins_and_an_unpublished_year_takes_obr():
    anchors = pd.DataFrame({
        "fy": ["2026-27", "2027-28"],
        "kind": ["plan", "illustrative"],
        "cgncr_ex": [137.2, 145.1],
    })
    plan = M.cgncr_annual_plan(anchors, extra={2026: 999.0, 2029: 93.2, 2030: 105.9})
    assert plan[2026] == 137.2
    assert plan[2027] == 145.1
    assert plan[2029] == 93.2
    assert plan[2030] == 105.9


def test_cached_table_has_the_remit_year():
    got = cgncr_ex_nr()
    if not got:
        return
    assert got[2026] == 137.2
