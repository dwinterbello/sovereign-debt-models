"""FY overlay: COLA on the indexed lines, boosted shape spreads the year."""

import pandas as pd
import pytest

from fy_overlay import apply_fy, cola_pct, cola_ratio, fers_diet, fy_of


def test_fers_diet_and_announced_2027_cola():
    assert cola_pct(2027) == 3.6
    assert fers_diet(3.6) == 2.6
    assert fers_diet(2.5) == 2.0
    assert fers_diet(1.3) == 1.3


def test_fy2027_ssa_step_is_the_cola_against_a_base_that_still_holds_a_quarter_of_the_old_rate():
    # FY2026 is three months on the Jan-2025 COLA and nine on Jan-2026. FY2027 is three months
    # on Jan-2026 and nine on the announced 3.6%. The ratio is not 0.75*3.6 because the base
    # year was not fully on the Jan-2026 rate either.
    old = 1 / 1.028
    base = (3 * old + 9 * 1.0) / 12
    nxt = (3 * 1.0 + 9 * 1.036) / 12
    assert cola_ratio(2027, 2026, "full") == pytest.approx(nxt / base)
    assert cola_ratio(2027, 2026, "blend") < cola_ratio(2027, 2026, "full")


def test_shape_is_kept_and_the_year_mean_is_the_overlay():
    idx = pd.bdate_range("2026-10-16", "2027-09-30")
    # Rising through the year so the shape is not flat.
    shape = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    hist_idx = pd.bdate_range("2025-10-01", periods=260)
    panel = pd.DataFrame({"wd_ssa": 100.0, "wd_defense": 50.0}, index=hist_idx)
    fc = pd.DataFrame({"wd_ssa": shape * 2.0, "wd_defense": shape}, index=idx)
    origin = pd.Timestamp("2026-10-15")
    out = apply_fy(fc, panel, origin, shares={"wd_ssa": {"full": 1.0, "blend": 0.0}},
                   growth=pd.Series({"wd_ssa": 0.05, "wd_defense": 0.05}))
    assert fy_of(origin) == 2027
    # Pure SSA ignores the 5% nominal drift and takes the COLA ratio (volume from a flat history is 0).
    assert out["wd_ssa"].mean() == pytest.approx(100.0 * cola_ratio(2027, 2026, "full"))
    # Shape order preserved.
    assert out["wd_ssa"].is_monotonic_increasing
    # Defense has no indexed share, so it takes the nominal step. base FY is ~FY2026, forecast is FY2027.
    assert out["wd_defense"].mean() == pytest.approx(50.0 * 1.05)
