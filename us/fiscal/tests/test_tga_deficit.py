"""Unit tests for TGA-implied deficit construction (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tga_deficit import (
    FRED_NOMINAL_GDP,
    LOOKBACK_12M,
    build_panel,
    interpolate_to_daily,
    parse_debt_public_mn,
    parse_tga_close_mn,
)


def test_parse_tga_prefers_closing_balance():
    dts = pd.DataFrame(
        {
            "record_date": ["2022-04-18", "2022-04-18", "2022-04-18", "2022-04-18"],
            "account_type": [
                "Treasury General Account (TGA) Opening Balance",
                "Total TGA Deposits (Table II)",
                "Total TGA Withdrawals (Table II) (-)",
                "Treasury General Account (TGA) Closing Balance",
            ],
            "close_today_bal": ["null", "null", "null", "null"],
            "open_today_bal": ["100", "20", "15", "841253"],
        }
    )
    s = parse_tga_close_mn(dts)
    assert len(s) == 1
    assert s.iloc[0] == 841253


def test_parse_tga_federal_reserve_account_and_legacy_tga():
    dts = pd.DataFrame(
        {
            "record_date": ["2021-09-30", "2021-10-01"],
            "account_type": ["Federal Reserve Account", "Treasury General Account (TGA)"],
            "close_today_bal": ["215160", "132452"],
            "open_today_bal": ["173745", "215160"],
        }
    )
    s = parse_tga_close_mn(dts)
    assert list(s.values) == [215160.0, 132452.0]


def test_parse_debt_public():
    dts = pd.DataFrame(
        {
            "record_date": ["2026-08-20", "2026-08-20"],
            "debt_catg": ["Debt Held by the Public", "Intragovernmental Holdings"],
            "close_today_bal": ["32278965", "7754292"],
        }
    )
    s = parse_debt_public_mn(dts)
    assert len(s) == 1
    assert s.iloc[0] == 32278965


def test_gdp_interpolation_midpoint():
    q = pd.Series(
        [100.0, 200.0],
        index=pd.to_datetime(["2020-01-01", "2020-04-01"]),
        name="GDP",
    )
    q = pd.Series(
        [100.0, 200.0],
        index=pd.to_datetime(["2020-01-01", "2020-04-01"]),
        name="GDP",
    )
    idx = pd.date_range("2020-01-01", "2020-04-01", freq="D")
    d = interpolate_to_daily(q, idx)
    mid = pd.Timestamp("2020-02-16")  # ~halfway Jan 1 → Apr 1
    assert abs(d.loc[mid] - 150.0) < 2.0
    after = pd.date_range("2020-04-02", "2020-04-10", freq="D")
    held = interpolate_to_daily(q, after)
    assert (held == 200.0).all()


def test_denominator_is_quarterly_nominal_gdp():
    # FRED series GDP is BEA NIPA 1.1.5, quarterly nominal SAAR, $bn.
    # GDPC1 is real. Production must not switch to real GDP.
    assert FRED_NOMINAL_GDP == "GDP"


def test_identity_constant_tga():
    idx = pd.date_range("2018-01-01", periods=400, freq="D")
    tga = pd.Series(800_000.0, index=idx)  # $mn
    debt = pd.Series(20_000_000.0 + np.arange(len(idx)) * 2.0, index=idx)  # +$2mn/day
    gdp = pd.Series([20000.0], index=pd.to_datetime(["2018-01-01"]))
    panel = build_panel(tga, debt, gdp)
    last = panel.dropna(subset=["implied_deficit_12m_bn"]).iloc[-1]
    expected_bn = (2.0 * LOOKBACK_12M) / 1000.0  # $mn → $bn
    assert abs(last["implied_deficit_12m_bn"] - expected_bn) < 1e-9
    assert abs(last["implied_deficit_12m_pct_gdp"] - 100.0 * expected_bn / 20000.0) < 1e-9
    assert last["d_tga_12m_bn"] == 0.0


def test_identity_tga_absorbs_issuance():
    idx = pd.date_range("2018-01-01", periods=400, freq="D")
    # Issuance parked in TGA: Δdebt = ΔTGA → implied deficit 0
    tga = pd.Series(100_000.0 + np.arange(len(idx)) * 5.0, index=idx)
    debt = pd.Series(10_000_000.0 + np.arange(len(idx)) * 5.0, index=idx)
    gdp = pd.Series([25000.0], index=pd.to_datetime(["2018-01-01"]))
    panel = build_panel(tga, debt, gdp)
    last = panel.dropna(subset=["implied_deficit_12m_bn"]).iloc[-1]
    assert abs(last["implied_deficit_12m_bn"]) < 1e-9
    assert abs(last["implied_deficit_12m_pct_gdp"]) < 1e-9


def test_daily_flow_sums_to_12m():
    idx = pd.date_range("2018-01-01", periods=400, freq="D")
    tga = pd.Series(800_000.0, index=idx)
    debt = pd.Series(20_000_000.0 + np.arange(len(idx)) * 2.0, index=idx)
    gdp = pd.Series([20000.0], index=pd.to_datetime(["2018-01-01"]))
    panel = build_panel(tga, debt, gdp)
    last = panel.dropna(subset=["implied_deficit_12m_bn"]).iloc[-1]
    flow_12m = panel["implied_deficit_1d_bn"].iloc[-LOOKBACK_12M:].sum()
    assert abs(flow_12m - last["implied_deficit_12m_bn"]) < 1e-9


def test_fy_calendar_and_fytd_resets_oct1():
    from tga_deficit import fy_day_of_year, fytd_cumsum, fytd_through_md, us_fiscal_year

    assert us_fiscal_year("2025-10-01") == 2026
    assert us_fiscal_year("2026-09-30") == 2026
    assert us_fiscal_year("2026-08-20") == 2026
    assert fy_day_of_year("2025-10-01") == 1
    assert fy_day_of_year("2024-09-30") == 366  # FY2024 contains 29 Feb 2024
    assert fy_day_of_year("2025-09-30") == 365

    idx = pd.to_datetime(["2024-09-30", "2024-10-01", "2024-10-02"])
    s = pd.Series([1.0, 5.0, 3.0], index=idx)
    ytd = fytd_cumsum(s)
    assert ytd.loc[fy_day_of_year("2024-09-30"), 2024] == 1.0
    assert ytd.loc[1, 2025] == 5.0
    assert ytd.loc[2, 2025] == 8.0

    through = fytd_through_md(s, 10, 1)
    assert through.loc[2024] == 0.0
    assert through.loc[2025] == 5.0


def test_fy_pivot_preserves_level_not_cumsum():
    from tga_deficit import fy_day_of_year, fy_pivot

    idx = pd.to_datetime(["2024-10-01", "2024-10-02", "2025-10-01", "2026-08-20"])
    s = pd.Series([800.0, 810.0, 900.0, 935.0], index=idx)
    wide = fy_pivot(s)
    assert wide.loc[1, 2025] == 800.0
    assert wide.loc[2, 2025] == 810.0
    assert wide.loc[1, 2026] == 900.0
    assert wide.loc[fy_day_of_year("2026-08-20"), 2026] == 935.0
    assert wide.loc[2, 2025] != 1610.0


def test_tga_seasonal_figure_is_level_overlay():
    from build_dashboard import _fig_tga_seasonal, _value_at_doy
    from tga_deficit import fy_day_of_year, fy_pivot

    dates = pd.date_range("2023-10-01", "2026-08-20", freq="B")
    panel = pd.DataFrame({"date": dates, "tga_bn": 500.0 + (dates.dayofyear % 40) * 10.0})
    fig = _fig_tga_seasonal(panel)
    names = [t.name for t in fig.data]
    assert "FY2026" in names
    assert "FY2025" in names
    fy2026 = next(t for t in fig.data if t.name == "FY2026")
    assert abs(float(fy2026.y[-1]) - float(panel["tga_bn"].iloc[-1])) < 1e-9
    assert "TGA balance by fiscal-year day" in (fig.layout.title.text or "")
    shapes = fig.layout.shapes or []
    assert any(getattr(s, "type", None) == "rect" for s in shapes)

    wide = fy_pivot(panel.set_index("date")["tga_bn"])
    asof = panel["date"].iloc[-1]
    prev = _value_at_doy(wide, 2025, fy_day_of_year(asof))
    assert pd.notna(prev)


def test_flow_seasonal_is_fytd_overlay():
    from build_dashboard import _fig_flow_seasonal

    idx = pd.bdate_range("2024-10-01", "2026-10-15")
    hist = pd.Series(1.0, index=idx)
    fut = pd.Series(1.0, index=pd.bdate_range("2026-10-16", "2027-03-01"))
    fig = _fig_flow_seasonal(hist, fut, title="Deposits aggregate by fiscal-year day ($bn FYTD)")
    names = [t.name for t in fig.data]
    assert "FY2026" in names and "FY2025" in names
    assert "Deposits aggregate" in (fig.layout.title.text or "")
    # FYTD of $1/day over ~250 business days in FY2026 to origin is far above 1.
    fy2026 = next(t for t in fig.data if t.name == "FY2026" and (not t.line.dash or t.line.dash == "solid"))
    assert float(fy2026.y[-1]) > 100.0


def test_fy_overlay_marks_the_forecast_and_keeps_three_history_years():
    from build_dashboard import FORECAST_DASH, MATLAB, _fig_fy_overlay

    idx = pd.bdate_range("2024-10-01", "2026-10-15")
    fut = pd.bdate_range("2026-10-16", "2027-11-02")
    s = pd.Series(1.0, index=idx.append(fut))
    fig = _fig_fy_overlay(s, title="t", ylab="$bn", origin=pd.Timestamp("2026-10-15"))
    names = [t.name for t in fig.data]
    assert "FY2025" in names and "FY2026" in names and "FY2027" in names
    assert "FY2024" not in names and "FY2028" in names
    solid = {t.name: t.line.color for t in fig.data if t.showlegend and (not t.line.dash or t.line.dash == "solid")}
    assert solid["FY2026"] == MATLAB[4] and solid["FY2025"] == MATLAB[5]
    assert any(t.name == "FY2028" and t.line.dash == FORECAST_DASH and t.line.color == MATLAB[2] for t in fig.data)
    assert any(t.name == "FY2027" and str(t.line.color).startswith("#") for t in fig.data)
    assert any(getattr(sh, "x0", None) is not None for sh in (fig.layout.shapes or []))


def test_top_dts_charts_extend_with_forecast_past_last_print():
    """GDP held flat past the panel — a bare reindex left the %/GDP forecasts empty."""
    from build_dashboard import _fig_12m, _fig_components, _fig_dts_split, _fig_dts_split_bn, _gdp_held

    idx = pd.bdate_range("2024-01-02", "2026-10-15")
    tga = 800.0 + np.arange(len(idx)) * 0.1
    debt = 20_000.0 + np.arange(len(idx)) * 5.0
    panel = pd.DataFrame({
        "date": idx,
        "gdp_saar_bn": 30_000.0,
        "tga_bn": tga,
        "debt_public_bn": debt,
        "d_tga_12m_bn": 40.0,
        "d_debt_12m_bn": 1_800.0,
        "implied_deficit_12m_pct_gdp": 6.0,
        "implied_deficit_3m_ann_pct_gdp": 6.5,
        "implied_deficit_3m_ann_pct_gdp_ma20": 6.4,
        "implied_deficit_1d_bn": 5.0,
        "dts_fiscal_12m_pct_gdp": 5.0,
        "dts_primary_12m_pct_gdp": 3.3,
        "dts_interest_12m_pct_gdp": 1.7,
        "dts_fiscal_12m_bn": 1_500.0,
        "dts_primary_12m_bn": 1_000.0,
        "dts_interest_12m_bn": 500.0,
        "dts_fiscal_1d_bn": 4.0,
        "dts_primary_1d_bn": 2.5,
        "dts_interest_1d_bn": 1.5,
        "mts_deficit_12m_pct_gdp": 5.5,
    })
    fut_idx = pd.bdate_range("2026-10-16", "2027-03-31")
    fut = pd.Series(4.0, index=fut_idx)
    d_tga = pd.Series(0.2, index=fut_idx)
    forecast = {
        "dts_fiscal_1d_bn": fut,
        "dts_primary_1d_bn": fut * 0.6,
        "dts_interest_1d_bn": fut * 0.4,
        "implied_deficit_1d_bn": fut,
        "d_tga_1d_bn": d_tga,
        "tga_bn": float(tga[-1]) + d_tga.cumsum(),
        "debt_public_bn": float(debt[-1]) + (fut + d_tga).cumsum(),
    }
    held = _gdp_held(panel, fut.index)
    assert held.notna().all() and float(held.iloc[0]) == 30_000.0
    names_12 = [t.name for t in _fig_12m(panel, forecast).data]
    assert "DTS cash 12m forecast" in names_12
    assert "TGA-implied 12m forecast" in names_12
    names_pct = [t.name for t in _fig_dts_split(panel, forecast).data]
    assert "DTS cash deficit forecast" in names_pct
    names_bn = [t.name for t in _fig_dts_split_bn(panel, forecast).data]
    assert "DTS cash deficit forecast" in names_bn
    names_id = [t.name for t in _fig_components(panel, forecast).data]
    assert "12m Δ TGA forecast" in names_id and "12m Δ debt forecast" in names_id
    from build_dashboard import _fig_3m

    names_3m = [t.name for t in _fig_3m(panel, forecast).data]
    assert "3m ann. forecast" in names_3m and "3m ann. 20d MA forecast" in names_3m


def test_dts_primary_is_fiscal_minus_interest():
    idx = pd.date_range("2018-01-01", periods=400, freq="D")
    tga = pd.Series(100_000.0 + np.arange(len(idx)) * 5.0, index=idx)  # +$5mn/day
    debt = pd.Series(10_000_000.0, index=idx)
    gdp = pd.Series([25000.0], index=pd.to_datetime(["2018-01-01"]))
    issues = pd.Series(12.0, index=idx)  # $mn
    redeem = pd.Series(2.0, index=idx)
    interest = pd.Series(3.0, index=idx)
    panel = build_panel(
        tga,
        debt,
        gdp,
        dts_issues_mn=issues,
        dts_redeem_mn=redeem,
        dts_interest_mn=interest,
    )
    row = panel.dropna(subset=["dts_fiscal_1d_bn"]).iloc[-1]
    # fiscal = (12-2)/1000 − 5/1000 = 0.005; primary = 0.005 − 0.003 = 0.002
    assert abs(row["dts_fiscal_1d_bn"] - 0.005) < 1e-9
    assert abs(row["dts_primary_1d_bn"] - 0.002) < 1e-9
    assert abs(row["dts_interest_1d_bn"] - 0.003) < 1e-9
    last = panel.dropna(subset=["dts_primary_12m_bn"]).iloc[-1]
    assert abs(last["dts_primary_12m_bn"] - 0.002 * LOOKBACK_12M) < 1e-9
    assert abs(last["dts_fiscal_12m_bn"] - last["dts_primary_12m_bn"] - last["dts_interest_12m_bn"]) < 1e-9
    fytd = panel.dropna(subset=["dts_fiscal_fytd_bn"]).iloc[-1]
    assert abs(fytd["dts_fiscal_fytd_bn"] - fytd["dts_primary_fytd_bn"] - fytd["dts_interest_fytd_bn"]) < 1e-9

