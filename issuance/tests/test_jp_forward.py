import pandas as pd
import pytest

from issuance import jp_forward as jf
from issuance.paths import CACHE

OPS = CACHE / "ops_jp.parquet"
ASOF = pd.Timestamp("2026-09-30")
HORIZON = pd.Timestamp("2027-09-30")


@pytest.mark.parametrize("product,day,mat", [
    ("10y", "2026-07-02", "2036-06-20"),
    ("10y", "2026-10-06", "2036-09-20"),
    ("20y", "2026-07-14", "2046-06-20"),
    ("30y", "2026-07-07", "2056-06-20"),
    ("5y", "2026-07-09", "2031-06-20"),
    ("2y", "2026-09-30", "2028-09-01"),
    ("40y", "2026-05-27", "2066-03-20"),
])
def test_new_issue_maturity_matches_tape_convention(product, day, mat):
    assert jf.new_maturity(product, pd.Timestamp(day)) == pd.Timestamp(mat)


@pytest.fixture(scope="module")
def rows():
    if not OPS.exists():
        pytest.skip("ops_jp cache absent")
    return pd.DataFrame(jf.jp_rows(pd.read_parquet(OPS), ASOF, HORIZON))


def test_second_half_fy2026_matches_plan(rows):
    a = rows[(rows["op_type"] == "Auction") & (rows["op_dt"] <= "2027-03-31")]
    plan_half = sum(jf.PLAN_MN[p] * n for p, n in {
        "2y": 6, "5y": 6, "10y": 6, "20y": 6, "30y": 6, "40y": 3, "10y IL": 2, "GX10y": 1, "GX5y": 1,
    }.items())
    assert a["amount_mn"].sum() == pytest.approx(plan_half, rel=0.06)


def test_one_regular_auction_per_product_per_month(rows):
    a = rows[rows["op_type"] == "Auction"].copy()
    a["product"] = a["isin"].str.extract(r"(JP(?:IL|GX)?\d+y)")[0]
    n = a.groupby([a["op_dt"].dt.to_period("M"), "product"]).size()
    assert n.max() == 1


def test_liquidity_auctions_reopen_live_issues(rows):
    lea = rows[rows["op_type"] == "Liquidity Enhancement"]
    assert len(lea) > 0
    assert lea["isin"].str.fullmatch(r"JP[A-Z]*\d+y#\d+").all()
    assert (lea["maturity_dt"] > lea["op_dt"] + pd.DateOffset(years=1)).all()
    fy_half = lea[lea["op_dt"] <= "2027-03-31"]["amount_mn"].sum()
    assert 0.6 * jf.PLAN_LEA_FY_MN / 2 < fy_half < 1.2 * jf.PLAN_LEA_FY_MN / 2


def test_liquidity_amounts_follow_mof_release(rows):
    from issuance.jp_calendar import load_lea

    ann = load_lea()
    if "2026-10" not in ann:
        pytest.skip("LEA release absent")
    lea = rows[rows["op_type"] == "Liquidity Enhancement"]
    by_m = lea.groupby(lea["op_dt"].dt.strftime("%Y-%m"))["amount_mn"].sum()
    for m in ("2026-10", "2026-11", "2026-12"):
        assert by_m[m] == pytest.approx(sum(ann[m].values()))
    assert by_m["2027-01"] == pytest.approx(by_m["2026-10"])


def test_parse_lea_release():
    from issuance.jp_calendar import parse_lea

    page = """<p>October&nbsp; 2026</p><p>November 2026</p><p>December 2026</p>
    <p>(1) Issues with remaining maturities of 1-5 years</p><p>2-year (issue numbers 478 to 490)</p>
    <p>Not be implemented</p><p>Approximately 700 billion yen</p><p>Not be implemented</p>
    <p>(2) Issues with remaining maturities of 5-11 years</p>
    <p>Approximately 500 billion yen</p><p>Approximately 500 billion yen</p><p>Approximately 1 trillion yen</p>"""
    got = parse_lea(page)
    assert got["2026-11"] == {"1-5": 700_000.0, "5-11": 500_000.0}
    assert got["2026-10"]["1-5"] == 0.0
    assert got["2026-12"]["5-11"] == 1_000_000.0


def _product(isin: str) -> str:
    m = pd.Series([isin]).str.extract(r"(IL|GX)?(\d+y)").iloc[0]
    return {"IL": "10y IL", "GX": "GX" + str(m[1])}.get(m[0], m[1]) if isinstance(m[0], str) else m[1]


def test_irregular_products_sit_in_plan_months(rows):
    a = rows[rows["op_type"] == "Auction"].copy()
    a["product"] = a["isin"].map(_product)
    for p, months in jf.PLAN_MONTHS.items():
        got = set(a.loc[a["product"] == p, "op_dt"].dt.month)
        assert got <= months, (p, got)


def test_fy2026_tape_plus_forward_matches_mof_plan(rows):
    """FY2026 = tape Apr–Sep + forward Oct–Mar, each product against its plan count × size."""
    ops = pd.read_parquet(OPS)
    fy = ops[(ops["op_dt"] >= "2026-04-01") & (ops["op_dt"] <= ASOF) & (ops["amount_issued"] > 0)]
    fwd = rows[rows["op_dt"] <= "2027-03-31"]
    lea = fy.loc[fy["op_type"] == "Liquidity Enhancement", "amount_issued"].sum() \
        + fwd.loc[fwd["op_type"] == "Liquidity Enhancement", "amount_mn"].sum()
    assert 0.85 * jf.PLAN_LEA_FY_MN < lea <= jf.PLAN_LEA_FY_MN
    a = fwd[fwd["op_type"] == "Auction"].assign(product=lambda d: d["isin"].map(_product))
    tape = fy[fy["op_type"] == "Auction"]
    base = tape["amount_issued"] - pd.to_numeric(tape["npc2_mn"], errors="coerce").fillna(0.0)
    for p, size in jf.PLAN_MN.items():
        n = {"40y": 6, "10y IL": 4, "GX10y": 2, "GX5y": 2}.get(p, 12)
        share = jf.npc2_share(jf._tape(ops), ASOF).get(p, 0.0)
        total = base[tape["product"] == p].sum() + a.loc[a["product"] == p, "amount_mn"].sum() / (1 + share)
        assert total == pytest.approx(size * n, rel=0.03), p


def test_bill_forward_runs_at_tape_pace():
    """Next six months of bills = last year's auction days at trailing sizes: each tenor's count
    matches the same window a year earlier, and total face is within 35% of it."""
    if not OPS.exists():
        pytest.skip("ops_jp cache absent")
    ops = pd.read_parquet(OPS)
    bills = ops[ops["sec_type"] == "Bill"]
    if bills.empty:
        pytest.skip("no JP bills on the tape")
    end = ASOF + pd.DateOffset(months=6)
    fwd = jf.bill_rows(ops, ASOF, end)
    assert set(fwd["product"]) == {"TDB 3m", "TDB 6m", "TDB 1y"}
    assert (fwd["op_dt"] > ASOF).all() and (fwd["op_dt"] <= end).all()
    prior = bills[(bills["op_dt"] > ASOF - pd.Timedelta(weeks=52)) & (bills["op_dt"] <= end - pd.Timedelta(weeks=52))]
    assert fwd.groupby("product").size().to_dict() == prior.groupby("product").size().to_dict()
    assert fwd["amount_mn"].sum() == pytest.approx(prior["amount_issued"].sum(), rel=0.35)
    last = bills[bills["op_dt"] <= ASOF].sort_values("op_dt").groupby("product").tail(jf.BILL_SIZE_N)
    size = fwd.groupby("product")["amount_mn"].first()
    want = last.groupby("product")["amount_issued"].mean().reindex(size.index)
    assert size.to_dict() == pytest.approx(want.to_dict())


def test_by_isin_keeps_jgb_issue_numbers():
    if not OPS.exists():
        pytest.skip("ops_jp cache absent")
    from issuance.isin_flow import by_isin

    ops = pd.read_parquet(OPS)
    out = by_isin(ops)
    assert out["isin"].str.startswith("JP10y#").any()
