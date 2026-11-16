import pandas as pd
import pytest

from issuance.dv01 import macaulay_duration
from issuance.forward import business_day
from issuance.index_ratio import RPI_XML, index_ratio
from issuance.uk_remit import index_ratios


def test_duration_follows_residual_not_half_year_grid():
    # A par 2.35y bond's Macaulay duration is under its maturity; the half-year grid made it 2.5y.
    assert macaulay_duration(2.35, 2.34, 2.34) < 2.35
    assert macaulay_duration(2.35, 2.34, 2.34) < macaulay_duration(2.45, 2.34, 2.34)
    assert macaulay_duration(0.25, 4.0, 4.0) == pytest.approx(0.25)


@pytest.mark.skipif(not RPI_XML.exists(), reason="DMO RPI cache")
def test_uk_index_ratio_matches_dmo():
    published = index_ratios()
    for isin, ratio in published.items():
        assert index_ratio("UK", isin, pd.Timestamp("2026-10-01")) == pytest.approx(ratio, rel=0.012)


def test_new_line_has_unit_ratio():
    assert index_ratio("UK", "NEW IL gilt", pd.Timestamp("2027-01-01")) == 1.0


def test_business_day_stays_in_month():
    assert business_day(pd.Timestamp("2026-11-28"), "IT") == pd.Timestamp("2026-11-27")
    assert business_day(pd.Timestamp("2026-11-01"), "IT") == pd.Timestamp("2026-11-02")
    assert business_day(pd.Timestamp("2026-12-25"), "IT") == pd.Timestamp("2026-12-23")
    assert business_day(pd.Timestamp("2026-11-26"), "US") == pd.Timestamp("2026-11-25")
    assert business_day(pd.Timestamp("2027-03-26"), "FR") == pd.Timestamp("2027-03-25")
