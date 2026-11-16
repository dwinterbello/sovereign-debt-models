from issuance.tenor import attach_tenor, bucket_from_years, years_from_label
import pandas as pd


def test_bucket_edges():
    assert bucket_from_years(0.5) == "0-1y"
    assert bucket_from_years(2.0) == "1-3y"
    assert bucket_from_years(7.0) == "7-10y"
    assert bucket_from_years(30.0) == "30y+"
    assert bucket_from_years(None) == "unknown"


def test_static_labels():
    assert years_from_label("10y") == 10
    assert years_from_label("30yL") == 30
    assert years_from_label("3m") == 0.25


def test_attach_ignores_unmapped_labels():
    df = pd.DataFrame({
        "op_dt": ["2020-01-15"],
        "maturity_dt": [pd.NaT],
        "sec_desc": ["not-a-tenor"],
        "resid_mat_yrs": [pd.NA],
    })
    out = attach_tenor(df)
    assert pd.isna(out.loc[0, "resid_mat_yrs"])
    assert out.loc[0, "tenor_bucket"] == "unknown"


def test_attach_from_dates():
    df = pd.DataFrame({
        "op_dt": ["2020-01-15"],
        "maturity_dt": ["2030-01-15"],
    })
    out = attach_tenor(df)
    assert abs(out.loc[0, "resid_mat_yrs"] - 10) < 0.05
    assert out.loc[0, "tenor_bucket"] == "10-15y"
