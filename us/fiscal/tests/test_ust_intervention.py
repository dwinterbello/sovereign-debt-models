"""Unit tests for liquidity vs fiscal gauges (no network)."""

from __future__ import annotations

import pandas as pd

from ust_intervention import (
    build_diagnostic,
    monthly_buybacks,
    parse_buybacks,
    parse_interest_public_bn,
)


def test_otr_richness_sign():
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    gsw = pd.Series([5.30] * 5, index=idx)
    cmt = pd.Series([5.20] * 5, index=idx)
    dummy = pd.Series(dtype=float)
    auctions = pd.DataFrame(columns=["auction_date", "high_yield"])
    panel = build_diagnostic(
        gdp_q_bn=pd.Series([20000.0], index=pd.to_datetime(["2024-01-01"])),
        gsw_30=gsw,
        dgs30=cmt,
        sofr=pd.Series([5.33] * 5, index=idx),
        tgcr=pd.Series([5.30] * 5, index=idx),
        floor=pd.Series([5.40] * 5, index=idx),
        rrp=pd.Series([100.0] * 5, index=idx),
        interest_m_bn=pd.Series(dtype=float),
        nipa_interest_q_bn=pd.Series([400.0], index=pd.to_datetime(["2024-01-01"])),
        avg_rate=dummy,
        auctions_30=auctions,
        buybacks_m=pd.DataFrame(),
    )
    assert abs(panel["otr_richness_30_bp"].iloc[-1] - 10.0) < 1e-9
    assert abs(panel["sofr_iorb_bp"].iloc[-1] - (-7.0)) < 1e-9
    assert abs(panel["sofr_tgcr_bp"].iloc[-1] - 3.0) < 1e-9
    assert abs(panel["nipa_interest_pct_gdp"].iloc[-1] - 2.0) < 1e-9


def test_interest_public_sums_month():
    raw = pd.DataFrame(
        {
            "record_date": ["2026-07-31", "2026-07-31", "2026-07-31"],
            "expense_catg_desc": [
                "INTEREST EXPENSE ON PUBLIC ISSUES",
                "INTEREST EXPENSE ON PUBLIC ISSUES",
                "INTEREST EXPENSE ON GOVT ACCOUNT SERIES",
            ],
            "month_expense_amt": ["1e9", "2e9", "9e9"],
        }
    )
    s = parse_interest_public_bn(raw)
    assert len(s) == 1
    assert abs(s.iloc[0] - 3.0) < 1e-9


def test_buybacks_split_cash_vs_liq():
    raw = pd.DataFrame(
        {
            "operation_date": ["2025-06-02", "2025-06-04", "2025-06-10"],
            "operation_type": ["Cash Management", "Liquidity Support", "Liquidity Support"],
            "security_type": ["Nominal Coupons"] * 3,
            "maturity_bucket": ["1Mo to 2Y", "20Y to 30Y", "10Y to 20Y"],
            "total_par_amt_accepted": ["10e9", "2e9", "3e9"],
            "total_par_amt_offered": ["20e9", "4e9", "6e9"],
        }
    )
    parsed = parse_buybacks(raw)
    m = monthly_buybacks(parsed)
    row = m.iloc[0]
    assert abs(row["cm_front_bn"] - 10.0) < 1e-9
    assert abs(row["liq_20_30_bn"] - 2.0) < 1e-9
    assert abs(row["liq_10_20_bn"] - 3.0) < 1e-9
    assert row["liq_other_bn"] == 0.0
