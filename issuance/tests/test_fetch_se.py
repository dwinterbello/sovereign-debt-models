"""SE outstanding CSV + auctions → ops_se."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from issuance import fetch_se as F

CSV = Path(__file__).resolve().parents[2] / "sovereign_cash" / "data" / "se" / "outstanding_securities.csv"
AUC = Path(__file__).resolve().parents[2] / "sovereign_cash" / "data" / "se" / "auctions.csv"


@pytest.mark.skipif(not CSV.is_file(), reason="SE outstanding CSV missing")
def test_outstanding_to_ops_has_sgb_and_bills():
    ops = F.outstanding_to_ops(CSV)
    assert set(ops["sec_type"]) >= {"Nominal", "Linker", "Bill"}
    assert ops["isin"].str.match(r"^(SE|XS)").all()
    assert (ops["amount_issued"] > 0).all()
    face_bn = ops["amount_issued"].sum() / 1e3
    assert 900 < face_bn < 1600
    il = ops[ops["sec_type"] == "Linker"]
    assert len(il) >= 5
    assert (pd.to_datetime(ops["op_dt"]) <= pd.Timestamp("1990-01-01")).all()
    assert ops["sec_type"].eq("Bill").sum() >= 3
    assert (ops["source"] == F.STOCK_SOURCE).all()


@pytest.mark.skipif(not (CSV.is_file() and AUC.is_file()), reason="SE auction/outstanding CSV missing")
def test_auctions_to_ops_join_live_isins_with_yield():
    taps = F.auctions_to_ops(AUC, CSV)
    assert len(taps) >= 100
    assert taps["isin"].str.match(r"^(SE|XS)").all()
    assert taps["yield"].notna().mean() > 0.9
    assert set(taps["op_type"]) <= {"Auction", "Syndication", "Buyback", "Exchange"}
    # YTD window the debt scorer reads must have nominal taps with yields
    ytd = taps[(pd.to_datetime(taps["op_dt"]).dt.year == pd.Timestamp.today().year)
               & taps["sec_type"].isin(["Nominal", "FRN"])
               & taps["yield"].notna()
               & (taps["amount_issued"] > 0)]
    assert len(ytd) >= 5
    assert (taps["source"] == F.AUCTION_SOURCE).all()


@pytest.mark.skipif(not (CSV.is_file() and AUC.is_file()), reason="SE auction/outstanding CSV missing")
def test_matured_series_kept_and_annual_gross_reconciles_to_auctions():
    taps = F.auctions_to_ops(AUC, CSV)
    taps["y"] = pd.to_datetime(taps["op_dt"]).dt.year
    bill = taps["sec_type"].eq("Bill")
    bills_24 = taps.loc[bill & taps["y"].eq(2024), "amount_issued"].sum() / 1e3
    assert 200 < bills_24 < 400
    synth = taps[taps["isin"].str.startswith("SE-")]
    assert synth["isin"].str.len().ge(10).all() and synth["maturity_dt"].notna().all()

    a = pd.read_csv(AUC)
    a["y"] = pd.to_datetime(a["date"]).dt.year
    a = a[a["type"].isin(["O", "B", "S"]) & a["y"].between(2020, 2026)]
    a_bill = a["series"].str.upper().str.startswith("STB")
    t = taps[taps["y"].between(2020, 2026)]
    for y in range(2020, 2027):
        for is_bill in (True, False):
            want = a.loc[a["y"].eq(y) & a_bill.eq(is_bill), "allocated_mn"].sum()
            got = t.loc[t["y"].eq(y) & t["sec_type"].eq("Bill").eq(is_bill), "amount_issued"].sum()
            assert abs(got - want) < 1.0, (y, is_bill, got, want)


def test_series_name_parsing():
    assert F._series_maturity("STB 18 Sep 96") == pd.Timestamp("1996-09-18")
    assert F._series_maturity("SGB 1047 5% 1 Dec 20") == pd.Timestamp("2020-12-01")
    assert F._series_coupon("SGB IL 3109 1.0% 1 Jun 25") == 1.0
    assert F._series_coupon("STB 20 Mar 24") == 0.0
    assert F._synthetic_key("SGB IL 3105 3.5% 1 Dec 15", pd.Timestamp("2015-12-01")) == "SE-SGBIL3105-20151201"
    assert F._synthetic_key("STB 20 Mar 24", pd.Timestamp("2024-03-20")) == "SE-STB-20240320"


@pytest.mark.skipif(not (CSV.is_file() and AUC.is_file()), reason="SE auction/outstanding CSV missing")
def test_stock_and_taps_do_not_double_count_face():
    from sovereign_cash.debt.isin_book import outstanding

    ops = F.fetch_se(force=False)
    assert F.STOCK_SOURCE in set(ops["source"]) and F.AUCTION_SOURCE in set(ops["source"])
    book = outstanding()
    se = book[book["country"] == "SE"]
    stock_face = ops.loc[ops["source"].eq(F.STOCK_SOURCE), "amount_issued"].sum()
    assert abs(se["face_mn"].sum() - stock_face) < 1.0
