"""How each line's share is carried forward, and the RRF grants (Nov 2026)."""
from __future__ import annotations

import pandas as pd

from sovereign_cash import monthly_components as M


def test_german_spending_no_longer_reverts():
    assert M.line_method("de", "primary") == "share_xc"
    assert "revert" not in M.METHOD.values()


def test_italy_lines_held_on_gdp():
    assert M.line_key("it", "cap_exp", None) == "ngdp"
    assert M.line_key("it", "other_rev", None) == "ngdp"
    assert M.line_key("fr", "primary", "cpi") == "cpi"


def test_rrf_grants_end_after_the_final_instalment():
    g = M.rrf_grants("it")
    assert g.loc["2021":"2025"].sum() > 40
    assert g.loc["2027":].sum() == 0
    months = pd.date_range("2026-09-01", "2027-12-01", freq="MS")
    extra = M.line_extra("it", "other_rev", months)
    assert extra.loc["2027":].sum() == 0 and extra.sum() >= 0
    assert M.line_extra("fr", "recettes", months).sum() == 0
