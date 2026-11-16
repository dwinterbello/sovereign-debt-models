from issuance.fr_text import parse_coupon, parse_french_maturity, scale_price, scale_yield
import pandas as pd


def test_french_maturity_oat_line():
    dt = parse_french_maturity("OAT 4% 25 avril 2009")
    assert dt == pd.Timestamp(2009, 4, 25)


def test_french_maturity_linker():
    dt = parse_french_maturity("OATi 3,4% 25 juillet 2029")
    assert dt == pd.Timestamp(2029, 7, 25)


def test_coupon_french_decimal():
    assert parse_coupon("OATi 3,4% 25 juillet 2029") == 3.4
    assert parse_coupon("OAT 4% 25 avril 2009") == 4.0


def test_aft_yield_and_price_scale():
    y = scale_yield(pd.Series([0.0381, 0.0463, 0.0329]))
    assert abs(y.iloc[0] - 3.81) < 1e-9
    px = scale_price(pd.Series([1.0155, 1.1398, 0.9991]))
    assert abs(px.iloc[0] - 101.55) < 1e-9
    # already-canonical units stay put
    assert abs(scale_yield(pd.Series([3.81, 4.63])).iloc[0] - 3.81) < 1e-9
    assert abs(scale_price(pd.Series([101.55, 99.91])).iloc[0] - 101.55) < 1e-9
