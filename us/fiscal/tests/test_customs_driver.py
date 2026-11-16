"""Customs duties = effective rate × imports (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from customs_driver import (
    customs_forecast,
    effective_rate,
    import_gdp_share,
    imports_daily_mn,
    last_complete_import_month,
)


def _series() -> tuple[pd.Series, pd.Series, pd.Series]:
    # 18 months of imports at $100bn/month; customs at 5% then 10%.
    months = pd.date_range("2023-01-31", periods=18, freq="ME")
    imports = pd.Series(100_000.0, index=months)  # $mn
    customs_m = pd.Series([5_000.0] * 12 + [10_000.0] * 6, index=months)
    # Spray customs across business days so monthly sums match.
    days = pd.bdate_range("2023-01-01", "2024-06-30")
    customs = pd.Series(0.0, index=days)
    for m, v in customs_m.items():
        mask = (customs.index.to_period("M") == m.to_period("M"))
        n = int(mask.sum())
        customs.loc[mask] = v / n
    gdp = pd.Series([20_000.0, 20_400.0, 20_800.0, 21_200.0],
                    index=pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]))
    return customs, imports, gdp


def test_effective_rate_uses_only_months_printed_by_origin():
    customs, imports, _gdp = _series()
    # Origin in May 2024: last complete import month is April (index Apr-30).
    # Trailing 12m ending Apr-24 = May-23..Apr-24: 8 months @5% + 4 @10% → 6.667%.
    origin = pd.Timestamp("2024-05-10")
    assert last_complete_import_month(imports, origin) == pd.Timestamp("2024-04-30")
    rate = effective_rate(customs, imports, origin, months=12)
    assert abs(rate - (8 * 5_000 + 4 * 10_000) / (12 * 100_000)) < 1e-9


def test_import_share_and_daily_pace():
    _c, imports, gdp = _series()
    origin = pd.Timestamp("2023-12-15")
    share = import_gdp_share(imports, gdp, origin, months=12)
    # import SAAR = 100e3*12/1000 = 1200; GDP ~20k → share ~0.06
    assert 0.05 < share < 0.07
    from customs_driver import BUSINESS_DAYS_PER_YEAR

    daily = imports_daily_mn(share, gdp, pd.DatetimeIndex(["2024-01-15"]))
    assert daily.iloc[0] == share * float(gdp.iloc[-1]) / BUSINESS_DAYS_PER_YEAR * 1000.0


def test_customs_forecast_scales_with_rate_and_shape():
    customs, imports, gdp = _series()
    origin = pd.Timestamp("2024-05-10")
    dates = pd.bdate_range("2024-05-11", periods=5)
    flat = customs_forecast(customs, origin, dates, shape=None, imports_mn=imports, gdp_saar_bn=gdp)
    shaped = customs_forecast(customs, origin, dates, shape=np.full(5, 2.0),
                              imports_mn=imports, gdp_saar_bn=gdp)
    assert np.allclose(shaped, flat * 2.0)
    rate = effective_rate(customs, imports, origin)
    share = import_gdp_share(imports, gdp, origin)
    from customs_driver import BUSINESS_DAYS_PER_YEAR

    expected = rate * share * float(gdp.dropna().iloc[-1]) / BUSINESS_DAYS_PER_YEAR * 1000.0
    assert abs(flat[0] - expected) / expected < 0.02


def test_forecast_choice_customs_wires_through(monkeypatch):
    import flow_model as fm

    monkeypatch.setattr(fm, "load_choice", lambda: {"dep_customs": "customs", "wd_interest": "model"})
    days = fm.bdays("2023-01-03", "2024-03-01")
    n = len(days)
    panel = pd.DataFrame({
        "dep_customs": np.full(n, 100.0),
        "dep_other": np.full(n, 50.0),
        "wd_other": np.full(n, 40.0),
        "wd_interest": np.full(n, 10.0),
    }, index=days)
    coupon = pd.Series(0.01, index=days)
    origin = pd.Timestamp("2024-01-31")
    assert origin in days
    feats = fm.calendar_features(days)
    models = fm.fit_all(panel, feats, origin)
    imports = pd.Series(200_000.0, index=pd.date_range("2023-01-31", periods=18, freq="ME"))
    gdp = pd.Series([25_000.0], index=pd.to_datetime(["2023-01-01"]))
    fut = days[days > origin][:10]
    out = fm.forecast(models, panel, coupon, origin, fut, choice={"dep_customs": "customs"},
                      customs_imports=imports, customs_gdp=gdp)
    flat = fm.forecast(models, panel, coupon, origin, fut, choice={"dep_customs": "flat"},
                       customs_imports=imports, customs_gdp=gdp)
    assert out["dep_customs"].notna().all()
    assert not np.allclose(out["dep_customs"].to_numpy(), flat["dep_customs"].to_numpy())
    # Rate ≈ (100 × bdays/month) / 200_000 ≈ 1%; share ≈ 0.096 → daily below the $100mn trailing mean.
    assert float(out["dep_customs"].mean()) < 100.0
