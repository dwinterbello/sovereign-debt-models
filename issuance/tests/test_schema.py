import pandas as pd

from issuance.schema import OPS_COLUMNS, coerce_ops


def test_coerce_fills_country_and_ccy():
    raw = pd.DataFrame({
        "op_dt": ["2024-01-15"],
        "maturity_dt": ["2034-01-15"],
        "isin": ["gb00b1234567"],
        "amount_issued": ["2500"],
    })
    out = coerce_ops(raw, "UK")
    assert list(out.columns) == list(OPS_COLUMNS)
    assert out.loc[0, "country"] == "UK"
    assert out.loc[0, "ccy"] == "GBP"
    assert out.loc[0, "isin"] == "GB00B1234567"
    assert abs(out.loc[0, "resid_mat_yrs"] - 10) < 0.05
