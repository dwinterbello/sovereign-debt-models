"""The naive tail of a cash framework carries redemptions with no gross issuance past the agency plan.

Until 2026-10-25 `history()` rolled those months into the 12m private-supply series, so the France
all-tenor forward read −9% of GDP a year by 2029 (Germany −4%) on the curve exhibit. The series must
stop at the last forecast month that still issues bonds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.debt.build import history, plan_horizon


def _flows() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", "2029-12-01", freq="MS")
    fc = months >= "2026-10-01"
    gross = np.where(months <= "2027-12-01", 30.0, 0.0)
    red = np.where(months.month.isin([2, 5, 9, 11]), 40.0, 0.0)
    return pd.DataFrame({"cc": "fr", "month": months, "is_forecast": fc, "bond_gross_bn": gross,
                         "bond_redemption_bn": red, "net_bonds_bn": gross - red, "net_bills_bn": 0.2,
                         "cb_net_bn": 0.0})


def test_plan_horizon_is_last_forecast_month_with_gross_issuance():
    fl = _flows()
    assert plan_horizon(fl) == pd.Timestamp("2027-12-01")
    # no gross column (US bridge / JP book) -> no truncation
    assert plan_horizon(fl.drop(columns=["bond_gross_bn"])) is None


def test_history_stops_at_the_plan_horizon_instead_of_going_negative():
    fl = _flows()
    ann = pd.DataFrame({"country": "fr", "year": range(2024, 2030), "gdp_lcu_bn": 3000.0})
    book = pd.DataFrame(columns=["country", "month", "coupon_bn", "bill_stock_bn", "cb_bn", "coupon_01_bn"])
    h = history(book, fl, ann)
    assert h["month"].max() == pd.Timestamp("2027-12-01")
    fwd = h.loc[h["is_forecast"].astype(bool), "private12_pct_gdp"].dropna()
    assert len(fwd) and (fwd > 0).all()
