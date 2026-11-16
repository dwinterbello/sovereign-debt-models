"""Debt subject to limit, EM capacity, seasonal TGA, contact / X-date."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from debt_ceiling import (
    EM_FLOOR_BN,
    em_capacity_from_history,
    parse_debt_limit_daily,
    project,
    seasonal_tga_path,
)


def _raw_limit_rows(dates, *, pub, ig, limit, notsubj=0.0):
    rows = []
    for d, p, i, lim in zip(dates, pub, ig, limit):
        rows += [
            {"record_date": d, "debt_catg": "Debt Held by the Public", "close_today_bal": p * 1000},
            {"record_date": d, "debt_catg": "Intragovernmental Holdings", "close_today_bal": i * 1000},
            {"record_date": d, "debt_catg": "Debt Not Subject to Limit", "close_today_bal": notsubj * 1000},
            {"record_date": d, "debt_catg": "Statutory Debt Limit", "close_today_bal": lim * 1000},
            {"record_date": d, "debt_catg": "Other Debt Subject to Limit", "close_today_bal": 0},
        ]
    return pd.DataFrame(rows)


def test_parse_stl_headroom():
    raw = _raw_limit_rows(
        ["2026-01-02", "2026-01-03"],
        pub=[30_000.0, 30_100.0],
        ig=[7_000.0, 7_010.0],
        limit=[41_000.0, 41_000.0],
        notsubj=100.0,
    )
    d = parse_debt_limit_daily(raw)
    assert abs(d.loc[pd.Timestamp("2026-01-02"), "stl_bn"] - (30_000 + 7_000 - 100)) < 1e-6
    assert abs(d.loc[pd.Timestamp("2026-01-03"), "headroom_bn"] - (41_000 - 37_010)) < 1e-6


def test_em_capacity_uses_at_limit_intragov_drop():
    idx = pd.bdate_range("2025-01-02", periods=80)
    # 40 days at the limit with a $300bn intragov drop, then relief.
    ig = np.concatenate([np.linspace(7500, 7200, 40), np.linspace(7200, 7400, 40)])
    head = np.concatenate([np.full(40, 0.02), np.full(40, 500.0)])
    hist = pd.DataFrame({"headroom_bn": head, "intragov_bn": ig, "stl_bn": 36_000.0, "limit_bn": 36_000.0}, index=idx)
    info = em_capacity_from_history(hist)
    assert info["em_from_intragov_drop_bn"] == pytest.approx(300.0, abs=1.0)
    assert info["em_capacity_bn"] >= EM_FLOOR_BN


def test_seasonal_tga_follows_prior_fy_shape_after_target():
    # Two prior FYs with a +100bn hump 60 days after a Dec-31-like anchor doy.
    from tga_deficit import fy_day_of_year, fy_start

    rows = []
    for fy in (2024, 2025):
        start = fy_start(pd.Timestamp(fy, 6, 1))
        for k in range(1, 200):
            dt = start + pd.Timedelta(days=k - 1)
            # Level = 800 + (doy - 92) for doy around New Year; simple ramp.
            doy = fy_day_of_year(dt)
            rows.append((dt, 800.0 + 0.5 * (doy - 92)))
    hist = pd.Series({d: v for d, v in rows}).sort_index()
    target = pd.Timestamp("2025-12-31")
    steered = pd.Series(850.0, index=pd.bdate_range("2025-12-01", "2026-06-30"))
    # Force steered flat at 850 through and after target.
    out = seasonal_tga_path(hist, steered, target, 850.0)
    after = out[out.index > target]
    assert len(after) > 20
    # Not a flat hold — seasonal shape moves the level.
    assert after.std() > 1.0


def test_project_marks_contact_when_issuance_eats_headroom():
    idx_h = pd.bdate_range("2026-01-02", periods=40)
    hist = pd.DataFrame({
        "stl_bn": 40_000.0 + np.arange(len(idx_h)),
        "limit_bn": 40_200.0,
        "headroom_bn": 200.0 - np.arange(len(idx_h)),
        "intragov_bn": 7_500.0 + 0.1 * np.arange(len(idx_h)),
        "debt_public_stl_bn": 32_500.0,
        "debt_not_subject_bn": 0.0,
        "other_subject_bn": 0.0,
    }, index=idx_h)
    origin = idx_h[-1]
    fut = pd.bdate_range(origin + pd.Timedelta(days=1), periods=200)
    path = pd.DataFrame({
        "net_bills_total_bn": 5.0,
        "net_coupons": 0.0,
        "tga_bn": 400.0,
        "tga_rollover_bn": np.linspace(400, 100, len(fut)),
        "op_flow_bn": -10.0,
        "known": False,
    }, index=fut)
    tga_hist = pd.Series(400.0, index=idx_h)
    em = {"em_capacity_bn": 250.0, "em_floor_bn": 250.0, "em_from_intragov_drop_bn": 250.0,
          "em_calibration_runs": [], "em_source": "test"}
    forward, summary = project(hist, path, tga_hist, em_info=em)
    assert summary["contact_date"] is not None
    assert pd.Timestamp(summary["contact_date"]) > origin
    assert "tga_seasonal_bn" in forward.columns and "tga_ceiling_bn" in forward.columns
    # TGA@contact (~400) + EM 250 = 650; $10bn/day → X-date ~60d after contact
    assert summary["x_date"] is not None
    assert pd.Timestamp(summary["x_date"]) > pd.Timestamp(summary["contact_date"])


def test_ceiling_tga_continuous_at_contact():
    """Post-contact TGA drains on op_flow (+ drift), ignoring path coupon/bill cash."""
    idx_h = pd.bdate_range("2026-01-02", periods=40)
    hist = pd.DataFrame({
        "stl_bn": 40_000.0 + np.arange(len(idx_h)),
        "limit_bn": 40_200.0,
        "headroom_bn": 200.0 - np.arange(len(idx_h)),
        "intragov_bn": 7_500.0,
        "debt_public_stl_bn": 32_500.0,
        "debt_not_subject_bn": 0.0,
        "other_subject_bn": 0.0,
    }, index=idx_h)
    fut = pd.bdate_range(idx_h[-1] + pd.Timedelta(days=1), periods=200)
    steered = np.full(len(fut), 800.0)
    # Rollover (with coupons) would stay elevated; ceiling path must ignore it.
    rollover = 650.0 + np.arange(len(fut), dtype=float)  # rising — must not be used
    path = pd.DataFrame({
        "net_bills_total_bn": 5.0,
        "net_coupons": 2.0,
        "tga_bn": steered,
        "tga_rollover_bn": rollover,
        "op_flow_bn": -10.0,
        "drift_bn": 0.0,
    }, index=fut)
    em = {"em_capacity_bn": 250.0, "em_floor_bn": 250.0, "em_from_intragov_drop_bn": 250.0,
          "em_calibration_runs": [], "em_source": "test"}
    forward, summary = project(hist, path, pd.Series(800.0, index=idx_h), em_info=em)
    contact = pd.Timestamp(summary["contact_date"])
    tga = forward["tga_ceiling_bn"]
    prev = tga.loc[tga.index < contact].iloc[-1]
    assert summary["tga_at_contact_bn"] == pytest.approx(800.0)
    # Contact day applies that day's op_flow: 800 → 790.
    assert tga.loc[contact] == pytest.approx(prev - 10.0)
    after = forward.index[forward.index >= contact]
    # Day 50 post-contact: 800 − 10×50 = 300 (not the rising rollover).
    assert tga.loc[after[49]] == pytest.approx(300.0)
    # Runway = 800 + 250 − 50 = 1000 at $10bn/day → 100th business day from contact.
    assert pd.Timestamp(summary["x_date"]) == after[99]
