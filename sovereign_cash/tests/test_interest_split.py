"""Interest carved from spending; rates shock from the ISIN book."""
from __future__ import annotations

import pytest

from sovereign_cash.interest_split import annual_interest, rates_shock_delta
from sovereign_cash.yoy_components import SPEC, components, deficit


@pytest.mark.parametrize("cc", ("fr", "it"))
def test_split_sums_to_annual(cc):
    frame = components(cc)
    ann = annual_interest(cc)
    y = int(frame.index.max().year) - 1
    if y not in ann.index:
        pytest.skip("no interest year")
    got = float(frame.loc[f"{y}-01-01":f"{y}-12-01", "interest"].sum())
    assert got == pytest.approx(float(ann.loc[y]), rel=0.02)


@pytest.mark.parametrize("cc", ("fr", "it"))
def test_primary_plus_interest_matches_spend(cc):
    from sovereign_cash.interest_split import SPEND_COL
    frame = components(cc)
    spend = frame[SPEND_COL[cc]]
    # after carve, primary + interest ≈ spend (clip can drop negative months)
    recon = frame["primary"] + frame["interest"]
    mask = spend > 0
    assert float((recon.loc[mask] - spend.loc[mask]).abs().median()) < 0.05


def test_rates_shock_positive_for_it():
    d = rates_shock_delta("it", 100.0)
    assert d
    # +100bp raises interest in every forward year
    assert all(v >= 0 for v in d.values())
    assert max(d.values()) > 0.5


def test_identity_uses_primary_and_interest():
    frame = components("it")
    y = deficit("it", {n: frame[n] for n, _ in SPEC["it"]})
    assert y.notna().any()
    assert "interest" in {n for n, _ in SPEC["it"]}
    assert "primary" in {n for n, _ in SPEC["it"]}


@pytest.mark.parametrize("cc", ("fr", "de", "it"))
def test_part_year_gets_its_share_not_the_whole_year(cc):
    """The year in progress carries only its months' share of the annual interest."""
    frame = components(cc)
    last = frame["interest"].dropna().index.max()
    if last.month == 12:
        pytest.skip("no part-year at the end of the data")
    y = last.year
    ann = annual_interest(cc)
    if y not in ann.index:
        pytest.skip("no interest total for the year in progress")
    got = float(frame.loc[f"{y}-01-01":f"{y}-12-01", "interest"].sum())
    assert got < float(ann.loc[y]) * (last.month / 12 + 0.15)
    prev = float(frame.loc[f"{y - 1}-01-01":f"{y - 1}-{last.month:02d}-01", "interest"].sum())
    assert got == pytest.approx(prev * float(ann.loc[y]) / float(ann.loc[y - 1]), rel=0.25)
