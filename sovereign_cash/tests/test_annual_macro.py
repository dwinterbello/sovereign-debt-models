"""Component-annual level × seasonal shape."""
from __future__ import annotations

import pandas as pd
import pytest

from sovereign_cash.annual_macro import (
    component_book, component_coefs, component_year, forecast_monthly, macro, targets_asof,
)
from sovereign_cash.core import cash_framework as C
from sovereign_cash.yoy_components import SPEC, components, deficit


def test_component_year_is_the_identity():
    frame = components("fr")
    drivers = macro("fr")
    origin = pd.Timestamp("2019-06-01")
    parts = {}
    for name, key in SPEC["fr"]:
        from sovereign_cash.annual_macro import annual, predict_year
        ann = annual(frame[name].dropna())
        if key is None:
            parts[name] = float(ann.loc[max(y for y in ann.index if y < origin.year)])
        else:
            parts[name] = predict_year(ann, drivers, origin, 2019, (key,), rule="unit")
    got = component_year("fr", frame, drivers, origin, 2019, rule="unit")
    assert got == pytest.approx(
        parts["primary"] + parts["interest"] - parts["recettes"] - parts["speciaux"]
    )


def test_france_receipts_gdp_slope_is_negative_before_floor():
    """SMB recettes annual YoY does not track NGDP in-sample; floor collapses to a drift.

    Italy / Germany tax–revenue loadings are positive on the same driver. Not a sign flip
    in the identity: cash requirement = spending − receipts.
    """
    free = component_coefs("fr", pd.Timestamp("2019-06-01"), rule="free")
    assert free["recettes"] is not None and free["recettes"][1] < 0
    floored = component_coefs("fr", pd.Timestamp("2019-06-01"), rule="floor")
    assert floored["recettes"][1] == 0.0
    it = component_coefs("it", pd.Timestamp("2024-06-01"), rule="free")
    assert it["tax_rev"] is not None and it["tax_rev"][1] > 0


def test_macro_prefers_eurostat_quarterly_gdp_and_monthly_hicp():
    from pathlib import Path
    from sovereign_cash.yoy_components import macro
    d = macro("fr")
    cache = Path("sovereign_cash/data/debt/macro_hq_fr.parquet")
    assert cache.exists()
    assert d["ngdp"].loc[2022] == pytest.approx(pd.read_parquet(cache)["ngdp"].loc[2022])
    # quarterly YoY mean is in the same ballpark as the WEO annual print, not a level
    assert 3 < d["ngdp"].loc[2022] < 10


def test_component_targets_plus_shape_hit_the_year_total():
    frame = components("fr")
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    drivers = macro("fr")
    origin = pd.Timestamp("2018-06-01")
    tg = targets_asof(y, drivers, origin, None, cc="fr", frame=frame, rule="unit")
    assert tg is not None
    fc = forecast_monthly(y, origin, 12, tg)
    ytd = float(y.loc["2018-01-01":origin].sum())
    assert ytd + float(fc.loc[fc.index.year == 2018].sum()) == pytest.approx(tg[2018])


def test_france_component_unit_loses_to_seas_on_pre():
    y = deficit("fr", {n: components("fr")[n] for n, _ in SPEC["fr"]}).dropna()
    book = component_book("fr", rule="unit")
    ann = C.backtest(y, ["annual_comp_unit_3"], targets_asof=book)
    seas = C.backtest(y, ["seas_mult_3"])
    means = C.score_table(pd.concat([ann, seas], ignore_index=True))
    means = means[(means["window"] == "PRE") & means["common"]].groupby("model")["mae"].mean()
    assert means["annual_comp_unit_3"] > means["seas_mult_3"]
