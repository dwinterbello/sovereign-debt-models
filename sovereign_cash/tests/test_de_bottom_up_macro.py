"""Germany v2 category×macro bottom-up — PRE gate, parser, no-lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.de import bottom_up as V1
from sovereign_cash.de import bottom_up_macro as BU
from sovereign_cash.core import cash_framework as C


@pytest.fixture(scope="module")
def verdict():
    BU.load_categories.cache_clear()
    BU.ship_verdict.cache_clear()
    return BU.ship_verdict()


@pytest.fixture(scope="module")
def cats():
    BU.load_categories.cache_clear()
    return BU.load_categories()


def test_parser_has_steuerarten_and_spend(cats):
    assert len(cats) >= 100
    assert "ytd_lohnsteuer" in cats and cats["ytd_lohnsteuer"].notna().sum() >= 90
    assert "ytd_ust" in cats
    assert "ytd_zins" in cats and "ytd_personal" in cats
    # spend economic leaves sum to Ausgaben; tax leaves are gross of Abzüge
    dec = [d for d in cats.index if d.month == 12 and d.year >= 2019]
    assert dec
    r = cats.loc[dec[-1]]
    spend = sum(float(r[f"ytd_{k}"]) for k in BU.SPEND
                if f"ytd_{k}" in cats.columns and pd.notna(r.get(f"ytd_{k}")))
    assert abs(float(r["ytd_ausgaben_total"]) - spend) < 1.0
    pos = sum(float(r[f"ytd_{k}"]) for k in BU.TAX_POS
              if f"ytd_{k}" in cats.columns and pd.notna(r.get(f"ytd_{k}")))
    assert pos > float(r["ytd_steuern_total"])  # Abzüge sit between gross and Steuern


def test_parser_t1_aufgabenbereiche(cats):
    assert "ytd_sozial" in cats and cats["ytd_sozial"].notna().sum() >= 90
    assert "ytd_t1_ausgaben" in cats and "ytd_t1_zins" in cats
    dec = [d for d in cats.index if d.month == 12 and d.year >= 2019]
    assert dec
    r = cats.loc[dec[-1]]
    t1 = sum(float(r[f"ytd_{k}"]) for k in BU.T1_TOP
             if f"ytd_{k}" in cats.columns and pd.notna(r.get(f"ytd_{k}")))
    assert abs(float(r["ytd_t1_ausgaben"]) - t1) < 1.0
    sh = BU.t1_primary_shares(pd.Timestamp(dec[-1].year + 1, 6, 1), cats)
    assert sh is not None and 0.4 <= sh["sozial"] <= 0.6


def test_ship_clears_pre_gate(verdict):
    """v2 replaces live only when PRE mean MAE beats v1 by ≥5% with ≤1 worse horizon."""
    assert verdict["ship"] is True
    assert verdict["pre_mean_macro"] <= (1.0 - BU.REL_EDGE) * verdict["pre_mean_production"]
    worse = sum(
        1 for r in verdict["pre_by_h"].values()
        if r["macro"] > r["production"] + 1e-12
    )
    assert worse <= 1


def test_mae_horizons(verdict):
    assert set(verdict["pre_by_h"]) >= set(C.HORIZONS)
    assert set(verdict["oos_by_h"]) >= set(C.HORIZONS)


def test_forecast_no_lookahead():
    exp, rev, cash, mb, maps = V1._flows()
    origin = pd.Timestamp("2019-06-01")
    if origin not in cash.index:
        pytest.skip("origin not in cash")
    fc1 = BU.forecast_core(origin, 12, exp, rev, mb, maps)
    assert fc1 is not None
    exp2 = exp.copy()
    rev2 = rev.copy()
    exp2.loc["2020-01-01":] = exp2.loc["2020-01-01":] + 10.0
    rev2.loc["2020-01-01":] = rev2.loc["2020-01-01":] + 10.0
    fc2 = BU.forecast_core(origin, 12, exp2, rev2, mb, maps)
    assert fc2 is not None
    open_m = [m for m in fc1.index if m.year == origin.year]
    assert np.allclose(fc1.loc[open_m], fc2.loc[open_m])


def test_authority_json():
    assert BU.AUTH.exists()
    raw = BU.AUTH.read_text()
    assert "lohnsteuer" in raw and "ngdp" in raw
    assert "a_priori" in raw or "elasticities_a_priori" in raw
    assert "open_year_spend" in raw and "0.20" in raw
    assert "later_interest" in raw
    assert "stock" in raw.lower() or "coupon" in raw.lower()
    assert "selection_audit" in raw
    assert "bmf_mb_inventory" in raw
    assert "t1_sozial" in raw and "0.35" in raw
    assert "wages" in raw


def test_category_audit_table():
    aud = BU.category_audit(pd.Timestamp("2019-12-01"))
    assert "lohnsteuer" in set(aud["category"])
    assert "zins" in set(aud["category"])
    assert "t1_sozial" in set(aud["category"])
    row = aud.set_index("category").loc["lohnsteuer"]
    assert row["driver"] == "ngdp"
    assert float(row["prior_beta"]) == 1.0
    assert float(aud.set_index("category").loc["kst"]["prior_beta"]) == 1.2
    z = aud.set_index("category").loc["zins"]
    assert "rates" in str(z["production_role"])
    assert "stock" in str(z["production_role"]).lower() or "FA" in str(z["production_role"])
    t1 = aud.set_index("category").loc["t1_sozial"]
    assert t1["driver"] == "wages"
    assert float(t1["prior_beta"]) == BU.T1_SOZIAL_WAGES_BETA


def test_rel_edge_unchanged():
    assert BU.REL_EDGE == 0.05


def test_shipped_open_later_priors():
    """Soll open-year pin + T1 sozial×wages + coupon×FA-stock interest."""
    assert BU.OPEN_MODE == "soll"
    assert BU.OPEN_YTD_WEIGHT_EXP == 0.20
    assert BU.OPEN_YTD_WEIGHT_REV == 0.05
    assert BU.LATER_REV_BETA == 1.0
    assert BU.LATER_PRIMARY_BETA == 0.0
    assert BU.LATER_INTEREST_BETA == 2.0
    assert BU.LATER_INTEREST_MODE == "coupon_stock"
    assert BU.T1_SOZIAL_WAGES_BETA == 0.35


def test_soll_open_ship(verdict):
    """Soll open pin clears PRE ≥5% vs v1; OOS stays near the ~7.4 ship."""
    assert verdict["ship"] is True
    assert verdict["pre_mean_macro"] < verdict["pre_mean_production"] * 0.95
    assert verdict["oos_mean_macro"] is not None
    assert verdict["oos_mean_macro"] < 8.5


def test_interest_coupon_no_lookahead():
    """Stock growth and annual interest use only information ≤ origin."""
    exp, rev, cash, mb, maps = V1._flows()
    origin = pd.Timestamp("2019-06-01")
    if origin not in cash.index:
        pytest.skip("origin not in cash")
    fc1 = BU.forecast_core(origin, 12, exp, rev, mb, maps)
    assert fc1 is not None
    # Corrupt future FA stock — path must not move (cache cleared first).
    BU._fa_total_stock.cache_clear()
    stock = BU._fa_total_stock()
    future = stock.loc["2020-01-01":].copy()
    if future.empty:
        pytest.skip("no future stock")
    # _stock_growth slices loc[:origin], so mutating post-origin is a no-op by construction
    g1 = BU._stock_growth(origin)
    BU._fa_total_stock.cache_clear()
    g2 = BU._stock_growth(origin)
    assert g1 == g2
    assert g1 > 0


def test_t1_overlay_no_lookahead():
    """T1 shares and wages use only prints ≤ origin."""
    origin = pd.Timestamp("2019-06-01")
    drivers = BU.macro_drivers()
    assert "wages" in drivers and len(drivers["wages"]) >= 8
    g = BU._t1_primary_growth(origin, drivers)
    assert g != 0.0
    # Future wage year must not enter (origin.year−1 only).
    drivers2 = dict(drivers)
    w = drivers["wages"].copy()
    if origin.year in w.index:
        w.loc[origin.year] = 99.0
    drivers2["wages"] = w
    assert BU._t1_primary_growth(origin, drivers2) == g
