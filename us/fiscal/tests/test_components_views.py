"""Components year-overlay: the within-year cumulative is the stock; the print is the flow."""

import pandas as pd

from build_components_dashboard import _year_views, bucket_forecast


def test_stock_is_the_within_year_cumsum_and_flow_is_the_print():
    idx = pd.to_datetime(["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06", "2025-10-07"])
    s = pd.Series(1.0, index=idx)
    v = _year_views(s)
    flow, stock = v["fy"]["flow"]["2026"], v["fy"]["stock"]["2026"]
    assert flow[0] == [1, 2, 3, 6, 7]
    assert flow[1] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert stock[0] == flow[0]
    assert stock[1] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_default_window_is_the_matlab_order():
    from build_dashboard import MATLAB, year_color

    shown = [year_color(y) for y in range(2025, 2031)]
    assert shown == list(reversed(MATLAB[:6]))
    assert year_color(2030) == MATLAB[0]


def test_bucket_forecast_keeps_the_shorter_steer_file_when_nothing_is_longer(tmp_path, monkeypatch):
    import build_components_dashboard as mod

    monkeypatch.setattr(mod, "DERIVED", tmp_path)
    short = pd.DataFrame({"dep_withheld": [1.0, 2.0]}, index=pd.to_datetime(["2026-10-16", "2026-12-31"]))
    assert bucket_forecast(short).index.max() == pd.Timestamp("2026-12-31")

    long = pd.DataFrame({"dep_withheld": [1.0, 3.0]}, index=pd.to_datetime(["2026-10-16", "2029-12-31"]))
    long.to_parquet(tmp_path / "fiscal_path_buckets.parquet")
    out = bucket_forecast(short)
    assert out.index.max() == pd.Timestamp("2029-12-31")
    assert out["dep_withheld"].iloc[-1] == 3.0
