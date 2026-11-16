"""GG forward interest = central bill + y0 sub-central gap (− BoJ coupons for Japan)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DERIVED = Path(__file__).resolve().parents[1] / "derived"
CA = DERIVED / "debt_country_annual.parquet"
PANEL = ("us", "uk", "de", "fr", "it", "es", "jp")


@pytest.mark.skipif(not CA.exists(), reason="run sovereign_cash.debt.build first")
def test_gg_interest_forward_identity():
    ca = pd.read_parquet(CA)
    assert "forecast" in ca.columns
    for cc in PANEL:
        g = ca[ca["cc"] == cc].set_index("year").sort_index()
        hist = g.loc[~g["forecast"].astype(bool)]
        fc = g.loc[g["forecast"].astype(bool)]
        if hist.empty or fc.empty:
            continue
        y0 = int(hist.index.max())
        if y0 not in g.index or not np.isfinite(g.at[y0, "gg_interest_bn"]):
            continue
        if not np.isfinite(g.at[y0, "interest_bn"]):
            continue
        gap0 = float(g.at[y0, "gg_interest_bn"] - g.at[y0, "interest_bn"])
        for y in fc.index:
            if not np.isfinite(g.at[y, "gg_interest_bn"]) or not np.isfinite(g.at[y, "interest_bn"]):
                continue
            expect = float(g.at[y, "interest_bn"]) + gap0
            assert abs(expect - float(g.at[y, "gg_interest_bn"])) < 1e-6, (cc, y, expect, g.at[y, "gg_interest_bn"])


@pytest.mark.skipif(not CA.exists(), reason="run sovereign_cash.debt.build first")
def test_us_a1_series_is_gg_not_federal():
    """Exhibit A1 plots US GG interest / GDP (like peers); federal stays on Exhibit 2."""
    from sovereign_cash.debt.note.service_rev import bench_frame

    ca = pd.read_parquet(CA)
    us = ca[(ca["cc"] == "us") & (ca["year"] == 2030)].iloc[0]
    b = bench_frame()
    assert abs(float(b.at["us", "o30"]) - float(us["gg_interest_pct_gdp"])) < 1e-9
    assert abs(float(b.at["us", "o30"]) - float(us["interest_pct_gdp"])) > 0.5
