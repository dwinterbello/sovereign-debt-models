from __future__ import annotations

import pandas as pd
import pytest

from issuance import fetch_jp as jp
from issuance.paths import CACHE
from issuance.schema import OPS_COLUMNS

RESULT_0903 = jp.PAGES / "eresul20260903a.htm"


@pytest.fixture(scope="module")
def ops() -> pd.DataFrame:
    if not jp.WORKBOOK.exists():
        pytest.skip("MoF workbook not cached")
    book = jp._finish(jp._workbook(force=False, refresh=False))
    parts = [book]
    if jp.TBILL_WORKBOOK.exists():
        parts.append(jp._finish(jp._tbill_workbook(refresh=False)))
    pages = sorted(jp.PAGES.glob("eresul*.htm")) if jp.PAGES.exists() else []
    recs = []
    for page in pages:
        if page.name.endswith("a.htm") or not (jp.PAGES / page.name.replace(".htm", "a.htm")).exists():
            rec = jp.parse_result(page.read_bytes())
            if rec is not None:
                rec["op_type"], rec["source"] = "Auction", "MoF auction result page"
                recs.append(rec)
    if recs:
        parts.append(jp._finish(pd.DataFrame(recs)))
    out = pd.concat(parts, ignore_index=True)
    out = out[out["op_dt"].notna() & out["isin"].notna() & (out["amount_issued"].fillna(0) > 0)]
    out = out.drop_duplicates(["isin", "op_dt", "op_type"], keep="first")
    return jp._coerce(jp._fill_static(out))


def test_schema_matches_uk(ops):
    uk = CACHE / "ops_uk.parquet"
    expected = list(OPS_COLUMNS) + list(jp.EXTRA_COLUMNS)
    assert list(ops.columns) == expected
    if uk.exists():
        uk_cols = list(pd.read_parquet(uk).columns)
        assert set(uk_cols) <= set(ops.columns)
        uk_frame = pd.read_parquet(uk)
        for col in uk_cols:
            if col in ("sec_desc", "sec_desc_dyn"):
                continue
            assert ops[col].dtype.kind == uk_frame[col].dtype.kind, col


def test_amounts_and_tails(ops):
    assert (ops["amount_issued"] > 0).all()
    assert (ops["country"] == "JP").all() and (ops["ccy"] == "JPY").all()
    assert set(ops["sec_type"].dropna()) <= {"Nominal", "Linker", "FRN", "Bill"}
    tail = ops["tail_bp"].dropna()
    assert len(tail) > 1000
    assert (tail >= 0).mean() > 0.98
    assert not ops.duplicated(["isin", "op_dt"]).any()
    coupon = ops["sec_type"] != "Bill"
    assert ops.loc[coupon, "isin"].str.fullmatch(r"JP(IL|GX|FRN|D)?\d+y#\d+").all()
    assert ops.loc[~coupon, "isin"].str.fullmatch(r"JPTDB#\d+").all()


def test_identifiers(ops):
    ids = set(ops["isin"])
    for isin in ("JP10y#383", "JP30y#91", "JP40y#19", "JP2y#488", "JPIL10y#29", "JPGX10y#3", "JPGX5y#4"):
        assert isin in ids, isin


def test_workbook_units(ops):
    row = ops[(ops["isin"] == "JP30y#91") & (ops["op_dt"] == "2026-08-06")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["amount_issued"] == pytest.approx(455_800 + 143_400 + 52_000)
    assert r["tail_bp"] == pytest.approx(1.5)
    assert r["tail_yen"] == pytest.approx(0.21)


def test_sep_2026_30y_result_page():
    if not RESULT_0903.exists():
        pytest.skip("result page not cached")
    rec = jp.parse_result(RESULT_0903.read_bytes())
    assert rec["product"] == "30y" and rec["issue"] == 91
    assert rec["op_dt"] == pd.Timestamp("2026-09-03")
    assert rec["accepted_mn"] == pytest.approx(456_200)
    assert rec["low_px"] == pytest.approx(98.65)
    assert rec["avg_y"] == pytest.approx(4.079)
    assert rec["npc2_mn"] == pytest.approx(54_500)


def test_sep_2026_30y_in_cache():
    out = CACHE / "ops_jp.parquet"
    if not out.exists():
        pytest.skip("ops_jp not built")
    ops = pd.read_parquet(out)
    row = ops[(ops["isin"] == "JP30y#91") & (ops["op_dt"] == "2026-09-03")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["amount_issued"] - r["npc1_mn"] - r["npc2_mn"] == pytest.approx(456_200)
    assert r["yield"] == pytest.approx(4.079)
    assert r["tail_yen"] == pytest.approx(98.93 - 98.65)


# FY2026 issuance plan (26 Dec 2025), Market Issuance Plan by Issue: TBs (1y + 6m), trillion yen.
PLAN_TB_FY2026_MN = 40_800_000.0


def _fy(ts: pd.Series) -> pd.Series:
    return ts.dt.year.where(ts.dt.month >= 4, ts.dt.year - 1)


def test_tbill_workbook_units(ops):
    if not jp.TBILL_WORKBOOK.exists():
        pytest.skip("T-bill workbook not cached")
    row = ops[ops["isin"] == "JPTDB#1375"]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["product"] == "TDB 1y" and r["sec_type"] == "Bill"
    assert r["op_dt"] == pd.Timestamp("2026-04-16")
    assert r["amount_issued"] == pytest.approx(2_999_910, abs=1)
    assert r["yield"] == pytest.approx(1.1071)
    assert r["tenor_bucket"] == "0-1y"


def test_bill_face_is_plan_scale(ops):
    """Every complete FY since unification has bills; the 1y line is plan-sized and 1y + 6m exceeds
    the plan's TB line (unified 1y bills and 6m bills carry FB on top of TB)."""
    if not jp.TBILL_WORKBOOK.exists():
        pytest.skip("T-bill workbook not cached")
    b = ops[ops["sec_type"] == "Bill"]
    fy = b.groupby([_fy(b["op_dt"]), "product"])["amount_issued"].sum().unstack(fill_value=0.0)
    assert (fy.loc[2009:2025].sum(axis=1) > 100_000_000).all()
    for year in (2024, 2025):
        assert 0.5 * PLAN_TB_FY2026_MN < fy.at[year, "TDB 1y"] < 2.0 * PLAN_TB_FY2026_MN
        assert fy.at[year, "TDB 1y"] + fy.at[year, "TDB 6m"] > PLAN_TB_FY2026_MN
    cy = b.groupby(b["op_dt"].dt.year)["amount_issued"].sum()
    assert cy.loc[2025] > PLAN_TB_FY2026_MN


def test_tbill_result_page():
    pages = sorted(jp.PAGES.glob("tb_eresul*.htm")) if jp.PAGES.exists() else []
    if not pages:
        pytest.skip("T-bill result pages not cached")
    rec = jp.parse_tbill_result(pages[-1].read_bytes(), "TDB 3m")
    assert rec is not None and rec["accepted_mn"] > 0 and rec["npc1_mn"] > 0
    assert rec["maturity_dt"] > rec["op_dt"]
    assert 0 < rec["low_px"] <= rec["avg_px"] < 100


def test_tbill_calendar_links():
    page = jp.PAGES / "cal_2610e.htm"
    if not page.exists():
        pytest.skip("Oct-2026 calendar not cached")
    links = jp.tbill_links(page.read_bytes())
    assert {"TDB 3m", "TDB 6m", "TDB 1y"} <= {link["product"] for link in links}
    assert all("etbillresul" in link["result"] for link in links)
    assert not any("etbill" in (link["main"] or "") for link in jp.calendar_links(page.read_bytes()))


def test_bill_isins_reach_flow_isin(ops):
    from issuance.isin_flow import by_isin

    if not (ops["sec_type"] == "Bill").any():
        pytest.skip("no bills")
    out = by_isin(ops[ops["op_dt"] >= "2026-01-01"])
    assert out.loc[out["sec_type"] == "Bill", "isin"].str.startswith("JPTDB#").any()


def test_security_labels():
    assert jp.bill_product("Treasury Discount Bills (6-month) (1411)") == "TDB 6m"
    assert jp.bill_product("1-year") == "TDB 1y"
    assert jp.isin_for("TDB 3m", 1413) == "JPTDB#1413"
    assert jp.product_from_security("10-year Inflation-Indexed Bonds") == "10y IL"
    assert jp.product_from_security("10-Year Climate Transition") == "GX10y"
    assert jp.product_from_security("40-Year") == "40y"
    assert jp.isin_for("GX5y", "4") == "JPGX5y#4"
