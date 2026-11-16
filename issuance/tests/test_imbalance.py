import pandas as pd
import pytest

from issuance.float_rv import _onto_stock
from issuance.imbalance import SLEEVE_OF, _cover_forward


def test_cover_forward_is_recursive_three_month_mean():
    idx = pd.date_range("2026-01-01", periods=4, freq="MS")
    s = pd.Series([2.0, 3.0, 4.0, 5.0], index=idx)
    out = _cover_forward(s, pd.Timestamp("2026-06-01"))
    assert out.loc["2026-04-01"] == pytest.approx(4.0)
    assert out.loc["2026-05-01"] == pytest.approx(4.0)
    assert out.loc["2026-06-01"] == pytest.approx((4.0 + 5.0 + 4.0) / 3)


def test_sleeves_follow_dmo_cut():
    assert SLEEVE_OF["3-7y"] == "Short"
    assert SLEEVE_OF["7-10y"] == "Medium"
    assert SLEEVE_OF["10-15y"] == "Medium"
    assert SLEEVE_OF["15-20y"] == "Long"
    assert SLEEVE_OF["30y+"] == "Long"


def test_new_30y_flow_lands_in_held_bucket():
    m = pd.Timestamp("2026-11-01")
    stock = pd.DataFrame({"country": ["US", "US"], "tenor_bucket": ["15-20y", "20-30y"], "month": [m, m]})
    flow = pd.DataFrame({"country": ["US", "US"], "month": [m, m], "tenor_bucket": ["30y+", "20-30y"], "flow_dv01": [36.6, 5.0]})
    out = _onto_stock(flow, stock, ["flow_dv01"])
    assert out["tenor_bucket"].tolist() == ["20-30y"]
    assert out["flow_dv01"].sum() == pytest.approx(41.6)
