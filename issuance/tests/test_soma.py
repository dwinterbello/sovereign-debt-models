import pandas as pd

from issuance.fetch_soma import _isin, soma_flow


def test_cusip_to_isin():
    assert _isin("912810US5") == "US912810US5"
    assert _isin("912795NF4") == "US912795NF4"
    assert pd.isna(_isin(""))


def test_soma_flow_first_month_zero_and_dropout():
    h = pd.DataFrame({
        "month": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
        "isin": ["USAAA", "USBBB", "USAAA"],
        "official_stock_mn": [100.0, 40.0, 80.0],
        "tenor_bucket": ["7-10y", "3-7y", "7-10y"],
        "sec_type": ["Nominal", "Nominal", "Nominal"],
        "resid_mat_yrs": [8.0, 5.0, 7.9],
        "maturity_dt": pd.to_datetime(["2032-01-01", "2029-01-01", "2032-01-01"]),
        "coupon": [3.0, 2.0, 3.0],
    })
    flow = soma_flow(h)
    jan = flow[flow["month"] == pd.Timestamp("2024-01-01")]
    assert jan.empty or (jan["official_mn"].abs() < 1e-9).all()
    feb = flow[flow["month"] == pd.Timestamp("2024-02-01")].set_index("isin")["official_mn"]
    assert abs(feb["USAAA"] + 20.0) < 1e-9
    assert abs(feb["USBBB"] + 40.0) < 1e-9


def test_soma_flow_drops_bills():
    h = pd.DataFrame({
        "month": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
        "isin": ["USAAA", "USBILL", "USAAA", "USBILL"],
        "official_stock_mn": [100.0, 50.0, 80.0, 80.0],
        "tenor_bucket": ["7-10y", "0-1y", "7-10y", "0-1y"],
        "sec_type": ["Nominal", "Bill", "Nominal", "Bill"],
        "resid_mat_yrs": [8.0, 0.2, 7.9, 0.15],
        "maturity_dt": pd.to_datetime(["2032-01-01", "2024-04-01", "2032-01-01", "2024-04-01"]),
        "coupon": [3.0, 0.0, 3.0, 0.0],
    })
    flow = soma_flow(h)
    assert "USBILL" not in set(flow["isin"])
    assert set(flow["isin"]) <= {"USAAA"}
