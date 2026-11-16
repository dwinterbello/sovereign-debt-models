"""The component model keeps the sign of the base, and France does not beat last year."""
from __future__ import annotations

import pandas as pd

from sovereign_cash.yoy_components import components, deficit, fit, forecast, mae, score


def test_slope_cannot_fight_the_base():
    # grows while the macro series is low, shrinks once it is high
    rows, level = [], 100.0
    for year in range(2000, 2021):
        level *= 1.10 if year < 2012 else 0.90
        for month in range(1, 13):
            rows.append((pd.Timestamp(year, month, 1), level / 12))
    y = pd.Series(dict(rows))
    driver = pd.Series({year: float(year - 2000) for year in range(2000, 2021)})
    _, slope = fit(y, driver, pd.Timestamp("2020-12-01"))
    assert slope == 0.0


def test_france_forecast_is_the_identity_and_loses_pre():
    frame = components("fr").dropna()
    gap = (deficit("fr", frame)
           - (frame["primary"] + frame["interest"] - frame["recettes"] - frame["speciaux"])).abs()
    assert float(gap.max()) < 1e-9
    fc = forecast("fr", pd.Timestamp("2024-12-01"), frame=frame)
    assert len(fc) == 12
    rebuilt = fc["primary"] + fc["interest"] - fc["recettes"] - fc["speciaux"]
    assert (fc["cash_requirement"] - rebuilt).abs().max() < 1e-9
    table = mae(score("fr", frame)).set_index("window")
    assert table.loc["PRE", "yoy"] > table.loc["PRE", "smly"]
