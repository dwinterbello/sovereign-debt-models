"""QA authority bench: history consistency, 2030 interest vs A1, flag taxonomy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt.note import qa_authority_bench as QA
from sovereign_cash.debt.note import service_rev as SR

SC = Path(__file__).resolve().parents[1]
HAS = (SC / "derived" / "debt_country_annual.parquet").exists()
pytestmark = pytest.mark.skipif(not HAS, reason="run sovereign_cash.debt.build first")


def test_frame_has_history_years_and_matches_a1_forecast():
    b = QA.frame()
    assert list(b.index) == list(QA.PANEL)
    for y in QA.Y_HIST:
        assert f"ig_ours_{y}" in b.columns and f"ig_imf_{y}" in b.columns
        assert f"ig_imf_x_{y}" in b.columns  # like-for-like gross-eq
        assert f"pri_ours_{y}" in b.columns and f"pri_imf_{y}" in b.columns
    a1 = SR.bench_frame()
    for cc in QA.PANEL:
        assert abs(b.at[cc, "ig_ours_2030"] - a1.at[cc, "o30"]) < 1e-9
        assert abs(b.at[cc, "ig_imf_2030"] - a1.at[cc, "imf30"]) < 1e-9
        assert abs(b.at[cc, "ig_ec_2027"] - a1.at[cc, "ec27"]) < 1e-9
        # gross-eq = raw WEO net + OECD netting when both exist
        raw, gx = b.at[cc, "ig_imf_2030"], b.at[cc, "ig_imf_x_2030"]
        net = SR._netting_pp(cc, 2027)
        if np.isfinite(raw) and np.isfinite(net):
            assert abs(gx - (raw + net)) < 1e-9


def test_gaps_flag_history_and_forecast():
    g = QA.gaps()
    assert {"history", "forecast"} <= set(g["family"])
    assert set(g[g.family == "history"].year.unique()) == set(QA.Y_HIST)
    for y in QA.Y_HIST:
        hist = g[(g.family == "history") & (g.year == y) & (g.metric == "interest_gdp")
                 & (g.source == "Commission")]
        # GG interest vs AMECO gross should be tight when the Commission cell exists
        euro = hist[hist.cc.isin(("de", "fr", "it", "es"))]
        assert (euro.gap_pp.abs() <= 0.15).all(), euro[["cc", "year", "gap_pp"]].to_string()
    # IMF interest rows compare to gross-eq authority (gap_ex_netting == gap)
    imf_ig = g[(g.metric == "interest_gdp") & (g.source == "IMF")]
    assert not imf_ig.empty
    assert (imf_ig.gap_pp - imf_ig.gap_ex_netting_pp).abs().max() < 1e-9
    assert set(g.flag) <= {"ok", "basis_fail", "needs_conviction", "conviction",
                           "basis_contaminated", "rates_vintage"}
    assert {"ok", "needs_conviction"} & set(g.flag)
    fc = g[g.family == "forecast"]
    assert "hist_gap_2019" in fc.columns


def test_build_writes_artifacts(tmp_path):
    paths = QA.build(tmp_path)
    assert paths["bench_hist"].stat().st_size > 40_000
    assert paths["bench_forecast"].stat().st_size > 50_000
    assert paths["gaps_hist"].stat().st_size > 15_000
    assert paths["flags"].exists()
    g = pd.read_csv(paths["csv"])
    assert {"flag", "hist_gap_2019", "family"} <= set(g.columns)
    assert set(g[g.family == "history"].year.unique()) == set(QA.Y_HIST)
    assert np.isfinite(g["gap_pp"]).all()


def test_us_interest_rev_matches_ameco_not_weo_revenue():
    """GG interest is AMECO/OECD gross; revenue must be AMECO URTG or IR invents a US gap."""
    g = QA.gaps()
    us = g[(g.family == "history") & (g.cc == "us") & (g.metric == "interest_rev")
           & (g.source == "Commission")]
    assert not us.empty
    assert (us.gap_pp.abs() <= 0.35).all(), us[["year", "ours", "authority", "gap_pp"]].to_string()


def test_history_debt_ours_is_annual_year_not_calendar_snap():
    """Snap stamps calendar-year WEO; history cells must use that year's annual stock."""
    b = QA.frame()
    ann = pd.read_parquet(SC / "derived" / "debt_annual.parquet")
    for y in QA.Y_HIST:
        for cc in QA.PANEL:
            ours = b.at[cc, f"debt_ours_{y}"]
            if not np.isfinite(ours):
                continue
            row = ann[(ann.country == cc) & (ann.year == y)]
            assert len(row), f"missing annual debt {cc} {y}"
            assert abs(ours - float(row.gg_debt_pct.iloc[0])) < 1e-9
