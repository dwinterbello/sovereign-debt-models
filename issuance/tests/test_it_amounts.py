import pandas as pd

from issuance.fetch_it import _scale_amount


def test_scale_euros_and_thousands():
    s = pd.Series([2500.0, 1_420_240.0, 8_000_000_000.0, -1_400_000_000.0])
    out = _scale_amount(s)
    assert abs(out.iloc[0] - 2500.0) < 1e-9
    assert abs(out.iloc[1] - 1420.24) < 1e-6
    assert abs(out.iloc[2] - 8000.0) < 1e-9
    assert abs(out.iloc[3] + 1400.0) < 1e-9
