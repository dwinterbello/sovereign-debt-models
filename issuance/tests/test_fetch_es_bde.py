"""BdE Bonos/Obligaciones announcement PDF → ISIN rows + historico/CNMV helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from issuance.fetch_es import (
    _parse_label,
    _sheet_op_date,
    _sheet_year_ok,
    attach_isins,
    parse_bde_announcement_text,
    parse_bde_carv_text,
    parse_cnmv_isin_html,
)

SAMPLE = """
BONOS Y OBLIGACIONES DEL ESTADO
Emisiones para el  10 de marzo de 2026
Oblig. €i a 10 años
 al 1,15%
ISIN: ES0000012O18
TRAMO 08
Bono a 3 años
 al 2,35%
ISIN: ES0000012P90
TRAMO 03
Obligación a 7 años
 al 3,00%
ISIN: ES0000012P74
TRAMO 06
Obligación a 15 años
 al 3,50%
ISIN: ES0000012O75
TRAMO 05
Emisión 01.10.2024 20.01.2026 07.10.2025 04.03.2025
Anotación del
tramo 10.03.2026 10.03.2026 10.03.2026 10.03.2026
Amortización  30.11.2036 31.03.2029 31.01.2033 31.01.2041
"""


def test_parse_bde_announcement_text_isins_and_settle():
    df = parse_bde_announcement_text(SAMPLE, source="fixture")
    assert len(df) == 4
    assert df["settle_dt"].iloc[0] == pd.Timestamp("2026-03-10")
    assert list(df["isin"]) == [
        "ES0000012O18", "ES0000012P90", "ES0000012P74", "ES0000012O75"]
    assert list(df["coupon"]) == [1.15, 2.35, 3.00, 3.50]
    assert list(df["maturity_dt"]) == [
        pd.Timestamp("2036-11-30"),
        pd.Timestamp("2029-03-31"),
        pd.Timestamp("2033-01-31"),
        pd.Timestamp("2041-01-31"),
    ]


def test_attach_isins_bono_prefers_b_over_nearby_obligacion():
    """Historico tipo 2.80 ≠ true 2.50; tenor alone would pick a nearby O line."""
    ops = pd.DataFrame([{
        "country": "ES", "ccy": "EUR",
        "op_dt": pd.Timestamp("2024-11-16"),
        "issue_dt": pd.Timestamp("2024-11-19"),
        "sec_type": "Nominal", "sec_desc": "B-3a",
        "amount_issued": 2000.0, "coupon": 2.80,
        "resid_mat_yrs": 3.0, "isin": pd.NA,
        "maturity_dt": pd.NaT,
    }])
    stock = pd.DataFrame([
        {"isin": "ES0000012A89", "coupon": 1.45, "maturity_dt": pd.Timestamp("2027-10-31"),
         "sec_type": "Nominal", "sec_desc_long": "ES0000012A89 O 1,45**", "issue_dt": pd.NaT},
        {"isin": "ES0000012M77", "coupon": 2.50, "maturity_dt": pd.Timestamp("2027-05-31"),
         "sec_type": "Nominal", "sec_desc_long": "ES0000012M77 B 2,50**", "issue_dt": pd.Timestamp("2024-01-09")},
        {"isin": "ES0000012L29", "coupon": 2.80, "maturity_dt": pd.Timestamp("2026-05-31"),
         "sec_type": "Nominal", "sec_desc_long": "ES0000012L29 B 2,80**", "issue_dt": pd.Timestamp("2023-01-01")},
    ])
    out = attach_isins(ops, stock=stock, results=pd.DataFrame(), bde=pd.DataFrame())
    assert out.loc[0, "isin"] == "ES0000012M77"


def test_attach_isins_prefers_bde_over_blank():
    ops = pd.DataFrame([{
        "country": "ES", "ccy": "EUR",
        "op_dt": pd.Timestamp("2026-03-05"),
        "issue_dt": pd.Timestamp("2026-03-10"),
        "sec_type": "Nominal", "sec_desc": "B-3a",
        "amount_issued": 2000.0, "coupon": 2.35,
        "resid_mat_yrs": 3.0, "isin": pd.NA,
        "maturity_dt": pd.NaT,
    }])
    bde = parse_bde_announcement_text(SAMPLE)
    out = attach_isins(ops, stock=pd.DataFrame(), results=pd.DataFrame(), bde=bde)
    assert out.loc[0, "isin"] == "ES0000012P90"
    assert out.loc[0, "maturity_dt"] == pd.Timestamp("2029-03-31")


def test_sheet_year_ok_drops_corrupt_2022_on_old_tabs():
    assert _sheet_year_ok(pd.Timestamp("2015-06-01"), 2015)
    assert not _sheet_year_ok(pd.Timestamp("2022-06-01"), 2001)
    assert not _sheet_year_ok(pd.Timestamp("2025-01-01"), 2022)
    # Only 2022–24 carry Y+1 stamps; the 2025 tab is dated correctly.
    assert not _sheet_year_ok(pd.Timestamp("2026-01-08"), 2025)


def test_sheet_op_date_shifts_plus_one_tabs_back_a_year():
    assert _sheet_op_date(pd.Timestamp("2015-06-01"), 2015) == pd.Timestamp("2015-06-01")
    assert _sheet_op_date(pd.Timestamp("2023-01-05"), 2022) == pd.Timestamp("2022-01-05")
    assert _sheet_op_date(pd.Timestamp("2024-01-10"), 2023) == pd.Timestamp("2023-01-10")
    assert _sheet_op_date(pd.Timestamp("2025-12-10"), 2024) == pd.Timestamp("2024-12-10")
    assert _sheet_op_date(pd.Timestamp("2022-06-01"), 2001) is None


def test_historico_plus_one_tabs_land_on_auction_days():
    """Stamped Wed/Fri/Sat; shifted back a year Tesoro's Tue/Thu (a few Wed)."""
    from issuance.fetch_es import RAW_ES, _auction_sheet

    path = RAW_ES / "historico_subastas.xlsx"
    if not path.exists():
        pytest.skip("historico_subastas.xlsx not cached")
    for year in (2022, 2023, 2024):
        days = pd.to_datetime(_auction_sheet(path, str(year))["op_dt"]).drop_duplicates()
        assert len(days) > 30
        assert (days.dt.year == year).all()
        assert days.dt.dayofweek.isin([1, 3]).mean() > 0.9
        assert days.dt.dayofweek.isin([1, 2, 3]).all()


def test_parse_label_bei_is_linker_with_tenor():
    assert _parse_label("B€i-5a") == ("Linker", 5.0)
    assert _parse_label("Bei-3a") == ("Linker", 3.0)
    assert _parse_label("O€i-10a") == ("Linker", 10.0)
    assert _parse_label("B-3a") == ("Nominal", 3.0)
    assert _parse_label("O->30a") == ("Nominal", 30.0)
    assert _parse_label("O->30a (s)") == ("Nominal", 30.0)


def test_parse_bde_carv_keeps_estado_bonos():
    text = (
        "ES00000120H2;BON;BONOS ESTADO                                 ;"
        "17/01/2006;  2.900;        1000.00; 1;T;31/10/2008;100.000;N;"
        ";;;;;;;;;; 3;F; 1;\n"
        "ES00000126W8;BON;OBLIGACIONES ESTADO                         ;"
        "14/10/2014;  0.551;        1000.00; 1;T;30/11/2019;100.000;N;"
        ";;;;;;;;;; 6;I; 1;\n"
        "ES0000090516;AUT;BONOS JUNTA ANDALUCIA                      ;"
        "07/11/2005;  2.875;        1000.00; 1;T;07/11/2008;100.000;N;\n"
        "ES0000011694;NDP;PRL. STRIP ESTADO                          ;"
        "15/07/1997;  0.000;        1000.00; 1;T;31/01/2008;100.000;N;\n"
    )
    df = parse_bde_carv_text(text)
    assert set(df["isin"]) == {"ES00000120H2", "ES00000126W8"}
    link = df[df["isin"] == "ES00000126W8"].iloc[0]
    assert link["sec_type"] == "Linker"
    assert float(df[df["isin"] == "ES00000120H2"].iloc[0]["coupon"]) == 2.9


def test_parse_outstanding_html_european_amounts():
    from issuance.fetch_es import parse_outstanding_html

    html = """
    <table>
      <tr><th>Código ISIN</th><th>Fecha de vencimiento</th>
          <th>Saldo en circulación (*)</th></tr>
      <tr><td>ES0L01908166</td><td>16/08/2019</td><td>8.416,4</td></tr>
    </table>
    <table>
      <tr><th>Código ISIN</th><th>Fecha de vencimiento</th>
          <th>Saldo en circulación (*)</th>
          <th>Coeficiente indexación</th><th>Valor facial</th></tr>
      <tr><td>ES00000121L2 O 4,60</td><td>30/07/2019</td><td>20.758,4</td>
          <td></td><td></td></tr>
      <tr><td>ES00000126W8 B €i 0,55**</td><td>30/11/2019</td><td>10.073,7</td>
          <td>1,04764</td><td>9.615,7</td></tr>
    </table>
    """
    df = parse_outstanding_html(html)
    assert len(df) == 3
    bill = df[df["isin"] == "ES0L01908166"].iloc[0]
    assert abs(bill["stock_mn"] - 8416.4) < 0.1
    assert bill["maturity_dt"] == pd.Timestamp("2019-08-16")
    nom = df[df["isin"] == "ES00000121L2"].iloc[0]
    assert abs(nom["coupon"] - 4.6) < 0.01
    assert nom["sec_type"] == "Nominal"
    link = df[df["isin"] == "ES00000126W8"].iloc[0]
    assert link["sec_type"] == "Linker"
    assert abs(link["index_ratio"] - 1.04764) < 1e-4


def test_parse_cnmv_isin_html_skips_strips():
    html = """
    <table>
    <tr><td>ES00000123C7</td><td>ESTADO/5.90 OBESEGR 20260730</td>
        <td>23.870.214.000,00</td><td>15/03/2011</td><td>30/07/2026</td>
        <td>1.000,00</td><td>EUR</td><td>5,9000</td><td>DBFTFB</td></tr>
    <tr><td>ES00000123I4</td><td>ESTADO/CUPCER PPALSEGR 20260730</td>
        <td>787.700.000,00</td><td>02/08/2011</td><td>30/07/2026</td>
        <td>100.000,00</td><td>EUR</td><td></td><td>DBZTFB</td></tr>
    </table>
    """
    df = parse_cnmv_isin_html(html)
    assert list(df["isin"]) == ["ES00000123C7"]
    assert float(df.loc[0, "coupon"]) == 5.9
    assert df.loc[0, "maturity_dt"] == pd.Timestamp("2026-07-30")
