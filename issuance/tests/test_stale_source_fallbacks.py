"""Fallbacks for sources that stall: US Fiscal Data, the Tesoro monthly bulletin, MEF buybacks."""
from __future__ import annotations

import pandas as pd

from issuance import fetch_es, fetch_it, fetch_us


def test_us_top_up_fills_auctions_fiscal_data_lacks(monkeypatch):
    raw = pd.DataFrame({
        "cusip": ["A", "B"], "auction_date": ["2026-10-06", "2026-10-21"],
        "total_accepted": ["59377561000", "null"], "security_type": ["Note", "Bond"],
    })
    td = pd.DataFrame({
        "cusip": ["B", "C"], "auction_date": ["2026-10-21", "2026-10-26"],
        "total_accepted": ["13000000900", "76667275400"], "security_type": ["Bond", "Note"],
    })
    monkeypatch.setattr(fetch_us, "_td_recent", lambda days=120: td)
    out = fetch_us._top_up(raw)
    assert list(out["cusip"]) == ["A", "B", "C"]
    assert out.loc[out["cusip"].eq("B"), "total_accepted"].item() == "13000000900"
    assert len(fetch_us._top_up(out)) == 3


def test_es_results_after_bulletin_only_adds_later_days():
    ops0 = fetch_es.coerce_ops(pd.DataFrame({
        "op_dt": [pd.Timestamp("2026-10-15")], "amount_issued": [2400.0], "sec_type": ["Nominal"]}), "ES")
    res = pd.DataFrame({
        "op_dt": pd.to_datetime(["2026-10-15", "2026-11-05"]),
        "issue_dt": pd.to_datetime(["2026-10-20", "2026-11-10"]),
        "maturity_dt": pd.to_datetime(["2029-03-31", "2036-10-31"]),
        "amount_issued": [2400.0, 2552.46], "coupon": [2.35, 3.40], "sec_type": ["Nominal", "Nominal"],
    })
    out = fetch_es.results_after_bulletin(ops0, res)
    assert len(out) == 2
    assert out["op_dt"].max() == pd.Timestamp("2026-11-05")


def test_it_buyback_announcement_and_results_parse():
    ann = (
        "BTP 2,35% \nIT0005660052 / 1 \n15/07/2025 - 15/01/2029\n"
        "BTP 0,45% \nIT0005467482 / 2 \n15/11/2021 - 15/02/2029\n"
        "Importo offerto (mln. €) 0/5.000 0/5.000\n"
        "Data asta 05/10/2026 05/10/2026\nData regolamento 07/10/2026 07/10/2026\n"
    )
    a = fetch_it.parse_bdi_buyback_announcement(ann)
    assert list(a["isin"]) == ["IT0005660052", "IT0005467482"]
    assert a["op_dt"].eq(pd.Timestamp("2026-10-05")).all()
    assert a["coupon"].tolist() == [2.35, 0.45]
    res = ("ISIN IT0005660052\nImporto acquistato 1.519.000.000\nPrezzo medio ponderato 97,246\n"
           "ISIN IT0005467482\nImporto acquistato 1.727.000.000\nPrezzo medio ponderato 92,978\n")
    got = fetch_it.parse_bdi_buyback_results(res, list(a["isin"]))
    assert got == {"IT0005660052": (1519.0, 97.246), "IT0005467482": (1727.0, 92.978)}
    # An issuance release (Importo assegnato) is not a buyback.
    assert fetch_it.parse_bdi_buyback_results("ISIN IT0005729931\nImporto assegnato 3.500.000.000",
                                              ["IT0005729931"]) == {}
