"""float_share_fc must cover every ops country that schema knows a currency for."""

from issuance.float_rv import CCY
from issuance.schema import COUNTRY_CCY


def test_float_ccy_tracks_schema_including_se():
    assert CCY == COUNTRY_CCY
    assert CCY["SE"] == "SEK"
    assert "SEK" not in CCY  # keys are country codes


def test_stock_dv01_skips_unknown_country_without_raising():
    import pandas as pd

    from issuance.float_rv import _stock_dv01

    ops = pd.DataFrame({
        "country": ["XX", "XX"],
        "isin": ["XX0001", "XX0001"],
        "op_dt": pd.to_datetime(["2024-01-15", "2024-06-15"]),
        "maturity_dt": pd.to_datetime(["2034-01-15", "2034-01-15"]),
        "amount_issued": [1000.0, 500.0],
        "coupon": [3.0, 3.0],
        "sec_type": ["Nominal", "Nominal"],
        "yield": [3.0, 3.1],
    })
    out = _stock_dv01(ops)
    assert out.empty
