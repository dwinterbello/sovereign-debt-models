import pandas as pd

from issuance.dv01 import attach_dv01, dirty_price, dv01_local_mn, macaulay_duration, modified_duration


def test_par_bond_duration():
    # 10y 3% annual-approx, semi-annual par-ish: duration near 8.5-9y
    mac = macaulay_duration(10.0, 3.0, 3.0)
    assert 7.5 < mac < 10.0


def test_bill_duration_is_residual():
    assert abs(macaulay_duration(0.5, 0.0, 4.0) - 0.5) < 1e-9


def test_dv01_scale():
    # £1bn face, price 100, 8y modified duration → £0.8mn / bp
    dmod = modified_duration(8.5, 4.0, 4.0)
    dv01 = dv01_local_mn(1000.0, 8.5, 4.0, 4.0, 100.0)
    assert 0.5 < dv01 < 1.2
    assert abs(dv01 - 1000.0 * dmod / 10000.0) < 1e-9


def test_dirty_price_par():
    px = dirty_price(10.0, 4.0, 4.0)
    assert 99.0 < px < 101.0


def test_linker_missing_yield_uses_real_coupon():
    flow = pd.DataFrame({
        "month": [pd.Timestamp("2024-01-01")],
        "ccy": ["USD"],
        "gross_mn": [1000.0],
        "tenor_bucket": ["7-10y"],
        "resid_mat_yrs": [8.0],
        "coupon": [1.0],
        "sec_type": ["Linker"],
    })
    fx = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")],
        "USDUSD": [1.0], "EURUSD": [1.1], "GBPUSD": [1.25],
    })
    out = attach_dv01(flow, fx)
    at_nominal_3 = dv01_local_mn(1000.0, 8.0, 1.0, 3.0, None)
    assert abs(float(out.iloc[0]["dv01_usd_mn"]) - at_nominal_3) > 0.01


def test_us_forward_uses_tape_residual_and_yield(monkeypatch):
    import issuance.dv01 as dv
    from issuance.dv01 import dv01_local_mn
    from issuance.forward import _row, price_paint

    monkeypatch.setattr(dv, "attach_dv01", lambda df, fx=None: df)
    ops = pd.DataFrame({
        "country": ["US", "US", "US"],
        "op_dt": [pd.Timestamp("2026-09-09"), pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-17")],
        "isin": ["US91282CRF0", "US912810UW6", "US91282CRE3"],
        "sec_type": ["Nominal", "Nominal", "Linker"],
        "resid_mat_yrs": [9.93, 29.93, 9.83],
        "coupon": [4.625, 5.125, 2.375],
        "yield": [4.834, 5.308, 2.653],
    })
    rows = [
        _row("US 10y", "10-15y", "2026-10-09", 39_000.0, "refunding", "Nominal"),
        _row("US 30y", "30y+", "2026-10-10", 22_000.0, "refunding", "Nominal"),
        _row("US 30y", "30y+", "2026-11-10", 25_000.0, "tbac", "Nominal"),
        _row("US 20y", "20-30y", "2026-10-15", 13_000.0, "refunding", "Nominal"),
        _row("US TIPS 10y", "7-10y", "2026-11-17", 19_000.0, "tbac", "Linker"),
        _row("US 3y", "3-7y", "2026-10-08", 58_000.0, "refunding", "Nominal"),
        _row("US frn", "1-3y", "2026-10-23", 30_000.0, "refunding", "FRN"),
    ]
    paint = pd.DataFrame(rows)
    paint["country"] = "US"
    out = price_paint(paint, ops, pd.DataFrame())
    oct30 = out[(out["isin"].eq("US 30y")) & (out["op_dt"].eq(pd.Timestamp("2026-10-10")))].iloc[0]
    nov30 = out[(out["isin"].eq("US 30y")) & (out["op_dt"].eq(pd.Timestamp("2026-11-10")))].iloc[0]
    oct10 = out[out["isin"].eq("US 10y")].iloc[0]
    tips = out[out["isin"].eq("US TIPS 10y")].iloc[0]
    frn = out[out["isin"].eq("US frn")].iloc[0]
    # October is a reopening: 30y stays under 30, 10y stays under 10.
    assert abs(float(oct30["resid_mat_yrs"]) - 29.93) < 1e-9
    assert oct30["tenor_bucket"] == "20-30y"
    assert abs(float(oct30["coupon"]) - 5.125) < 1e-9
    assert abs(float(oct30["yield"]) - 5.308) < 1e-9
    assert nov30["tenor_bucket"] == "30y+"
    assert abs(float(nov30["resid_mat_yrs"]) - 30.01) < 1e-9
    assert oct10["tenor_bucket"] == "7-10y"
    assert abs(float(out[out["isin"].eq("US 20y")].iloc[0]["resid_mat_yrs"]) - 19.92) < 1e-9
    assert out[out["isin"].eq("US 20y")].iloc[0]["tenor_bucket"] == "15-20y"
    assert abs(float(out[out["isin"].eq("US 3y")].iloc[0]["resid_mat_yrs"]) - 3.0) < 1e-9
    assert tips["sec_type"] == "Linker"
    assert abs(float(tips["yield"]) - 2.653) < 1e-9
    assert tips["tenor_bucket"] == "7-10y"
    assert frn["sec_type"] == "FRN"
    at_tape = dv01_local_mn(22_000.0, 29.93, 5.125, 5.308, None)
    at_par3 = dv01_local_mn(22_000.0, 30.0, 3.0, 3.0, None)
    assert at_tape < at_par3 * 0.85
