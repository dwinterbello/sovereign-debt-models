import pandas as pd
import pytest

from issuance.fiscal_bridge import official_bill_forward, scaled_last_year


def _hist():
    months = pd.date_range("2025-10-01", "2026-09-01", freq="MS")
    return pd.DataFrame({"country": "US", "month": months, "bill_dv01_usd": range(50, 62),
                         "official_bill_dv01_usd": [1.0] * 9 + [4.0, 5.0, 6.0]})


def test_bill_dv01_scales_year_ago_by_stock_ratio_and_chains():
    hist = _hist()
    extra = pd.DataFrame({"country": "US", "month": pd.date_range("2026-10-01", "2027-10-01", freq="MS")})
    ratio = {m: 1.1 for m in extra["month"]}
    s = scaled_last_year(hist, extra, "bill_dv01_usd", ratio)
    assert s.iloc[0] == pytest.approx(50 * 1.1)
    # Oct-2027 chains on the Oct-2026 forecast, so both ratios compound.
    assert s.iloc[-1] == pytest.approx(50 * 1.1 * 1.1)


def test_non_us_rows_are_not_scaled():
    hist = _hist().assign(country="DE")
    extra = pd.DataFrame({"country": "DE", "month": [pd.Timestamp("2026-10-01")]})
    s = scaled_last_year(hist, extra, "bill_dv01_usd", {pd.Timestamp("2026-10-01"): 2.0})
    assert s.iloc[0] == pytest.approx(50.0)


def test_official_bills_hold_the_trailing_quarter():
    hist = _hist()
    extra = pd.DataFrame({"country": ["US", "DE"], "month": [pd.Timestamp("2026-10-01")] * 2})
    o = official_bill_forward(hist, extra)
    assert list(o.index) == [0]
    assert o.iloc[0] == pytest.approx(5.0)
