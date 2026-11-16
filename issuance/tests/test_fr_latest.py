from issuance.fr_latest import parse_aft_tables
from issuance.fr_text import parse_french_maturity
from issuance.fetch_fr import _from_press_csv
import pandas as pd


HTML = """
<table>
<tr><th></th><th>OAT 25/05/2036 1.25%</th><th>OAT 25/11/2036 3.70%</th></tr>
<tr><td>Auction date</td><td>03/09/2026</td><td>03/09/2026</td></tr>
<tr><td>Settlement date</td><td>07/09/2026</td><td>07/09/2026</td></tr>
<tr><td>Amount bid*</td><td>10,220</td><td>15,666</td></tr>
<tr><td>Amount served*</td><td>3,300</td><td>6,862</td></tr>
<tr><td>Total amount issued*</td><td>4,193</td><td>9,268</td></tr>
<tr><td>Bid to cover ratio**</td><td>3.10</td><td>2.28</td></tr>
<tr><td>Weighted average price</td><td>76.96%</td><td>95.66%</td></tr>
<tr><td>Weighted average rate</td><td>4.19%</td><td>4.23%</td></tr>
<tr><td>ISIN Code</td><td>FR0013154044</td><td>FR0014018YR0</td></tr>
</table>
"""


def test_slash_maturity_from_oat_header():
    assert parse_french_maturity("OAT 25/05/2036 1.25%") == pd.Timestamp(2036, 5, 25)
    assert parse_french_maturity("03/09/2026") == pd.Timestamp(2026, 9, 3)


def test_parse_aft_latest_table():
    out = parse_aft_tables(HTML, source="test")
    assert len(out) == 2
    assert out.iloc[0]["isin"] == "FR0013154044"
    assert abs(out.iloc[0]["amount_issued"] - 4193) < 1e-6
    assert abs(out.iloc[1]["amount_issued"] - 9268) < 1e-6
    assert out.iloc[0]["op_dt"] == pd.Timestamp(2026, 9, 3)
    assert out.iloc[0]["sec_type"] == "Nominal"
    assert abs(out.iloc[0]["btc"] - 3.10) < 1e-9


def test_press_csv_has_2026():
    out = _from_press_csv()
    assert not out.empty
    assert out["op_dt"].min() >= pd.Timestamp(2026, 1, 1)
    assert (out["isin"].astype(str).str.startswith("FR")).all()
    assert out["amount_issued"].min() > 0
