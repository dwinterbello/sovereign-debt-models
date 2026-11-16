import pandas as pd
import pytest

from issuance.fetch_fr import _hist_totals, annual_vs_aft, load_recaps, recap_from_text
from issuance.paths import CACHE, RAW

RECAPS = sorted((RAW / "aft" / "MonthlyRecap").glob("*_operations_mensuelles.pdf"))
OPS = CACHE / "ops_fr.parquet"
HIST_MLT = sorted((RAW / "aft" / "KeyFigures" / "OAT").glob("*_hist_mlt.xlsx"))

EURO_HEADER = """Adjudication OAT 3,00% 25/05/2033 11/01/2024 4 000 15/01/2024 25/05/2033 98,00%
TOTAL OAT 33 972
Rachats de gré à gré :
Montant nominal des rachats de gré à gré pour le compte de l'État : 480 000 000 €
Libellé du contrat
OAT 0,00% 25/02/2025 480 000 000
Pensions livrées de l'AFT
"""

MIXED = """TOTAL OAT 41,420
Rachats de gré à gré
Montant nominal des rachats de gré à gré pour le compte de l'État : 19655,50 M€
LIbellé du contrat Nominal de l'opération (en M d'euros)
OAT 0,00% 25/02/2026 3150,50
OAT 3,50 % 25/04/2026 1 330,00
OATi 0.10% 01/03/2025 950,00
OAT€i 1,85% 25/07/2027 14 224,00
BTF 18 MARS 2026 (08/12/2025) 1
Pensions livrées de l'AFT
"""


def test_recap_parses_euro_header_and_units():
    r = recap_from_text(EURO_HEADER, pd.Timestamp(2024, 1, 1))
    assert r["ok"]
    assert r["header_mn"] == pytest.approx(480.0)
    assert [x["nominal_mn"] for x in r["lines"]] == [pytest.approx(480.0)]
    assert r["total_oat_mn"] == pytest.approx(33972.0)


def test_recap_parses_decimal_and_thousands_commas():
    r = recap_from_text(MIXED, pd.Timestamp(2024, 10, 1))
    assert r["ok"], r
    assert r["total_oat_mn"] == pytest.approx(41420.0)
    assert r["btf_mn"] == pytest.approx(1.0)
    got = {(x["kind"], x["coupon"]): x["nominal_mn"] for x in r["lines"]}
    assert got == {("OAT", 0.0): 3150.5, ("OAT", 3.5): 1330.0, ("OATi", 0.1): 950.0, ("OAT€i", 1.85): 14224.0}


def test_recap_flags_header_mismatch():
    r = recap_from_text(MIXED.replace("19655,50", "19000,00"), pd.Timestamp(2024, 10, 1))
    assert not r["ok"]


def test_hist_totals_override_communique_amounts():
    ops = pd.DataFrame({"isin": ["FR1", "FR2", "FR3"], "op_dt": pd.to_datetime(["2026-04-16"] * 3),
                        "op_type": ["Auction", "Auction", "Syndication"], "amount_issued": [4355.0, 100.0, 8000.0]})
    hist = pd.DataFrame({"isin": ["FR1", "FR3"], "op_dt": pd.to_datetime(["2026-04-16"] * 2),
                         "amount_issued": [5216.0, 9000.0]})
    out = _hist_totals(ops.copy(), [hist])
    assert out["amount_issued"].tolist() == [5216.0, 100.0, 8000.0]
    assert out.attrs["onc_added_mn"] == {2026: pytest.approx(861.0)}


needs_data = pytest.mark.skipif(not (RECAPS and OPS.exists() and HIST_MLT),
                                reason="AFT recaps / hist_mlt / ops_fr cache not on disk")


@pytest.fixture(scope="module")
def table():
    ops = pd.read_parquet(OPS)
    return ops, load_recaps(RECAPS), annual_vs_aft(ops, load_recaps(RECAPS)).set_index("year")


@needs_data
def test_every_recap_reconciles_to_its_header(table):
    _, recaps, _ = table
    assert all(r["ok"] for r in recaps), [r["file"] for r in recaps if not r["ok"]]


@needs_data
@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_tape_gross_matches_aft_recap_incl_noncompetitive(table, year):
    t = table[2]
    if year not in t.index:
        pytest.skip(f"no {year} recaps")
    assert t.at[year, "gross_tape_bn"] == pytest.approx(t.at[year, "gross_aft_recap_bn"], abs=0.1)


@needs_data
@pytest.mark.parametrize("year", [2024, 2025])
def test_full_year_net_of_buybacks_reconciles_to_programme(table, year):
    t = table[2]
    assert t.at[year, "recap_months"] == 12
    assert t.at[year, "gross_tape_bn"] == pytest.approx({2024: 339.8, 2025: 347.67}[year], abs=0.1)
    assert t.at[year, "buybacks_tape_bn"] == pytest.approx(t.at[year, "buybacks_aft_recap_bn"], abs=0.05)
    assert t.at[year, "buybacks_tape_bn"] > 40
    assert abs(t.at[year, "net_tape_bn"] / t.at[year, "programme_bn"] - 1) < 0.015


@needs_data
def test_buybacks_carry_isin_and_maturity(table):
    ops = table[0]
    bb = ops[ops["op_type"].astype(str).eq("Buyback")]
    assert len(bb) and (bb["amount_issued"] < 0).all()
    assert bb["isin"].notna().all() and bb["maturity_dt"].notna().all()
    assert bb["op_dt"].max() <= ops.loc[ops["op_type"].astype(str).ne("Buyback"), "op_dt"].max()


@needs_data
def test_oat_auctions_equal_hist_total(table):
    ops = table[0]
    h = pd.read_excel(HIST_MLT[-1], header=0).iloc[:, [1, 5, 11]]
    h.columns = ["op_dt", "isin", "total"]
    h["op_dt"] = pd.to_datetime(h["op_dt"], errors="coerce")
    tot = h.dropna().groupby(["isin", h["op_dt"].dt.normalize()])["total"].sum()
    a = ops[ops["op_type"].astype(str).eq("Auction") & ops["sec_type"].astype(str).ne("Bill")
            & (ops["op_dt"] >= "2024-01-01")]
    keys = pd.MultiIndex.from_arrays([a["isin"].astype(str), a["op_dt"].dt.normalize()])
    ref = pd.Series(tot.reindex(keys).to_numpy(), index=a.index).dropna()
    assert len(ref) > 300
    assert (a.loc[ref.index, "amount_issued"] - ref).abs().max() < 0.5
