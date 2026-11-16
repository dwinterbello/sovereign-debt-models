"""BoJ JGB holdings by issue, purchase operations, monthly flow. Runs on cached raw files."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from issuance import fetch_boj as boj

ID = re.compile(r"^JP(IL|GX)?\d+y#\d+$|^JPTB#\d+$")
FRN = re.compile(r"^JPFRN15y#\d+$")


def _raw(directory, pattern):
    files = sorted(p for p in directory.glob(pattern) if p.suffix in {".xlsx", ".xls", ".zip"})
    if not files:
        pytest.skip(f"no cached BoJ raw files in {directory}")
    return files


@pytest.fixture(scope="module")
def holdings():
    return boj.build_holdings(_raw(boj.MEI_DIR, "mei*"))


@pytest.fixture(scope="module")
def operations():
    return boj.build_operations(_raw(boj.OPE_DIR, "ope*"))


def test_ids_follow_convention(holdings):
    ids = holdings["isin"].unique()
    bad = [i for i in ids if not ID.match(i) and not FRN.match(i)]
    assert not bad, bad[:10]
    # 15y floaters (last issued 2008) are the only out-of-convention ids, all redeemed.
    latest = holdings[holdings["date"] == holdings["date"].max()]
    assert latest["isin"].map(lambda i: bool(ID.match(i))).all()


def test_known_layouts_parse():
    one = boj.MEI_DIR / "mei260918.xlsx"
    if not one.exists():
        pytest.skip("mei260918.xlsx not cached")
    h = boj.parse_holdings_file(one)
    assert h["date"].iloc[0] == pd.Timestamp("2026-09-18")
    row = h.set_index("isin").loc["JP10y#344"]
    assert row["face_mn"] == 80279 * 100
    assert {"GX5y", "GX10y", "IL10y", "40y"} <= set(h["product"])


def test_latest_total_plausible(holdings):
    latest = holdings[holdings["date"] == holdings["date"].max()]
    total = latest["face_mn"].sum()
    print(f"BoJ JGB holdings {latest['date'].iloc[0]:%Y-%m-%d}: {total / 1e6:.2f}tn yen face")
    assert 4.5e8 <= total <= 6.0e8


def test_book_value_cross_check(holdings):
    # BoJ Accounts 31-Aug-2026: JGBs 519,927,691,026 thousand yen (book value, amortised cost).
    snap = holdings[holdings["date"] == pd.Timestamp("2026-08-31")]
    if snap.empty:
        pytest.skip("2026-08-31 snapshot not cached")
    face = snap["face_mn"].sum()
    book = 519_927_691_026 / 1e3
    assert abs(face / book - 1) < 0.01


def test_no_negative_holdings(holdings):
    assert (holdings["face_mn"] >= 0).all()
    assert not holdings.duplicated(["date", "isin"]).any()


def test_history_is_continuous(holdings):
    totals = holdings.groupby("date")["face_mn"].sum()
    assert totals.index.min() <= pd.Timestamp("2013-04-30")
    gaps = totals.index.to_series().diff().dt.days.dropna()
    assert gaps.max() <= 35
    # Between snapshots the stock moves by purchases less redemptions: never > 25tn.
    assert totals.diff().abs().max() < 2.5e7


def test_flow_identity_and_maturity(holdings):
    flow = boj.build_flow(holdings)
    assert ((flow["holdings_mn"] == 0) | ~flow["matured"]).all()
    matured = flow[flow["matured"]]
    assert (matured["flow_mn"] < 0).all()
    # Stock change = sum of flows, month by month.
    stock = flow.groupby("month")["holdings_mn"].sum()
    chg = flow.groupby("month")["flow_mn"].sum(min_count=1)
    assert (stock.diff().iloc[1:] - chg.iloc[1:]).abs().max() < 1.0


def test_operations(operations):
    assert not operations.empty
    assert (operations["accepted_mn"] >= 0).all()
    assert (operations["date"] >= operations["offer_date"]).all()
    spec = operations.dropna(subset=["isin"])
    assert spec["isin"].map(lambda i: bool(ID.match(i))).all()
    plan = boj.build_plan(operations).dropna(subset=["realised_mn"])
    # Taper schedule: realised within 2% of the announced monthly amount.
    assert ((plan["realised_mn"] / plan["planned_mn"] - 1).abs() < 0.02).all()
