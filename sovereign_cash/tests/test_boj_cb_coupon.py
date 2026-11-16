"""BoJ ISIN holdings → coupon remittance netted from Japan GG interest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt.cashflow import CB_NET_CC, OFFICIAL_ISIN, cb_coupon_path, components

DERIVED = Path(__file__).resolve().parents[1] / "derived"


@pytest.mark.skipif(not OFFICIAL_ISIN.exists(), reason="issuance official_isin_flow cache missing")
def test_boj_cb_coupon_path_is_material_and_ages():
    assert "jp" in CB_NET_CC
    s = cb_coupon_path("jp", 2025, list(range(2025, 2031)))
    assert len(s) == 6
    # ~¥3.5tn on ~¥540tn BoJ coupon face at ~0.65% average coupon
    assert 2_000 < float(s[2025]) < 6_000
    # QT runoff: locked BoJ coupons decline as the book matures
    assert float(s[2030]) < float(s[2025])
    assert cb_coupon_path("de", 2025, [2025, 2026]).empty


@pytest.mark.skipif(not OFFICIAL_ISIN.exists(), reason="issuance official_isin_flow cache missing")
def test_components_carries_cb_coupon_for_japan():
    frame = components(2025, [2025, 2026], lambda cc, y: 0.0)
    jp = frame[frame["cc"] == "jp"].set_index("year")
    assert jp.at[2025, "cb_coupon_bn"] > 2_000
    assert jp.at[2025, "cb_coupon_bn"] < jp.at[2025, "coupon_bn"]
    us = frame[frame["cc"] == "us"]
    if len(us):
        assert (us["cb_coupon_bn"] == 0.0).all()


@pytest.mark.skipif(
    not (DERIVED / "debt_country_annual.parquet").exists(),
    reason="run sovereign_cash.debt.build first",
)
def test_japan_gg_keeps_boj_coupons_as_memo_only():
    """BoJ ISIN coupons are stored but not subtracted from GG (GFSM: remittance ≠ negative interest)."""
    ca = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    jp = ca[ca["cc"] == "jp"].set_index("year")
    if "cb_coupon_bn" not in jp.columns or not np.isfinite(jp["cb_coupon_bn"].get(2026, np.nan)):
        pytest.skip("rebuild debt_country_annual after BoJ memo column")
    hist = jp.loc[~jp["forecast"].astype(bool)]
    fc = jp.loc[jp["forecast"].astype(bool)]
    assert len(fc) and np.isfinite(fc["cb_coupon_bn"]).any()
    y0 = int(hist.index.max())
    gap0 = float(jp.at[y0, "gg_interest_bn"] - jp.at[y0, "interest_bn"])
    for y in fc.index:
        cb = float(fc.at[y, "cb_coupon_bn"])
        if not np.isfinite(cb):
            continue
        expect = float(fc.at[y, "interest_bn"]) + gap0  # no − cb
        assert abs(expect - float(fc.at[y, "gg_interest_bn"])) < 1e-6
        assert cb > 0
    assert float(fc["cb_coupon_bn"].iloc[-1]) < float(fc["cb_coupon_bn"].iloc[0])  # ages down
    assert float(hist["gg_interest_pct_gdp"].iloc[-1]) > 0.5


@pytest.mark.skipif(
    not (DERIVED / "debt_country_annual.parquet").exists(),
    reason="run sovereign_cash.debt.build first",
)
def test_japan_gg_identity_boj_memo_not_netted():
    """gg_interest = central + y0 residual; cb_coupon_bn is memo; history has no cb; only JP has cb."""
    ca = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    if "cb_coupon_bn" not in ca.columns or not np.isfinite(ca.loc[ca["cc"] == "jp", "cb_coupon_bn"]).any():
        pytest.skip("rebuild debt_country_annual after BoJ memo column")
    jp = ca[ca["cc"] == "jp"].set_index("year")
    y0 = int(jp.index[~jp["forecast"].astype(bool)].max())
    resid = float(jp.at[y0, "gg_interest_bn"] - jp.at[y0, "interest_bn"])
    fc = jp[jp.index > y0]
    cb = fc["cb_coupon_bn"].astype(float)
    assert cb.notna().all() and (cb > 0).all()
    ident = fc["interest_bn"] + resid  # memo only — do not subtract cb
    assert (fc["gg_interest_bn"] - ident).abs().max() < 1e-6
    assert (cb.diff().dropna() < 0).all()
    assert cb.iloc[0] < 0.40 * fc["interest_bn"].iloc[0]
    assert jp.loc[jp.index <= y0, "cb_coupon_bn"].isna().all()
    others = ca[(ca["cc"] != "jp") & ca["cb_coupon_bn"].notna()]
    assert others.empty or (others["cb_coupon_bn"] == 0).all()
    assert (100 * fc["gg_interest_bn"] / fc["gdp_bn"] - fc["gg_interest_pct_gdp"]).abs().max() < 1e-6
