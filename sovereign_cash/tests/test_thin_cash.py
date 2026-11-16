"""Spain and Japan rest-of-year cash models. Offline on the local caches."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import thin_cash as T


def _es():
    y = T.load("es")
    if y.index.min() > pd.Timestamp("2016-01-01") or y.index.max() < pd.Timestamp("2025-12-01"):
        pytest.skip("Spain IGAE cache does not cover 2015–2025")
    return y


def _jp():
    y = T.load("jp")
    if T._fy_block(y, 2024, 4) is None or T._fy_block(y, 2025, 4) is None:
        pytest.skip("Japan tax-corrected cash does not cover FY2024 and FY2025")
    return y


def test_spain_seas_add_3_beats_runrate_on_both_halves():
    y = _es()
    shipped = T.rest_errors(y, "es", T.ES_MODEL)
    naive = T.rest_errors(y, "es", "runrate")
    smly = T.rest_errors(y, "es", "smly")
    common = shipped.merge(naive, on=["fy", "k"], suffixes=("_s", "_r")).merge(smly, on=["fy", "k"])
    assert len(common) >= 20
    for lo, hi in ((None, 2022), (2023, None)):
        g = common
        if hi is not None:
            g = g[g["fy"] <= hi]
        if lo is not None:
            g = g[g["fy"] >= lo]
        assert len(g) >= 9
        assert g["ratio_s"].mean() < g["ratio_r"].mean()
        assert g["ratio_s"].mean() < g["ratio"].mean()


def test_japan_smly_beats_runrate_on_the_only_holdout_year():
    y = _jp()
    smly = T.rest_errors(y, "jp", "smly")
    naive = T.rest_errors(y, "jp", "runrate")
    seas = T.rest_errors(y, "jp", "seas_add_1")
    hold = smly[smly["fy"] == 2025]
    assert len(hold) == 3
    assert T.mean_ratio(hold) < T.mean_ratio(naive[naive["fy"] == 2025])
    assert T.mean_ratio(hold) < T.mean_ratio(seas[seas["fy"] == 2025])
    # One fiscal year. Not a PRE gate.
    assert hold["fy"].nunique() == 1


def test_publish_covers_the_open_fiscal_year():
    written = T.publish()
    assert set(written) == {"es", "jp"}
    for cc, frame in written.items():
        y = T.load(cc)
        last = y.index.max()
        end = T.fy_end(cc, last)
        start = T.fy_start_month(cc)
        next_end = T.month_ts(T.fy_of(end, start) + 1, 12, start)
        fc = frame.loc[frame["is_forecast"]].set_index("month")["cash_requirement_bn"]
        # Rest of the open FY plus the next full FY (seasonal / bottom-up).
        expect = pd.date_range(last + pd.offsets.MonthBegin(1), next_end, freq="MS")
        assert list(fc.index) == list(expect)
        assert np.isfinite(fc.to_numpy(float)).all()
        hist = frame.loc[~frame["is_forecast"], "month"]
        assert hist.max() == last
        assert not frame["month"].duplicated().any()


def test_es_run_module_writes_parquet():
    from sovereign_cash.es import run as R

    s = R.run(dashboard=False, verbose=False)
    assert s["cc"] == "es"
    assert s["n_forecast"] >= 1


def test_spain_live_uses_bottom_up_macro():
    from sovereign_cash.es import bottom_up_macro as BUM

    assert BUM.SHIP is True
    frame = T.build_frame("es")
    assert frame is not None
    fc = frame.loc[frame["is_forecast"]]
    assert len(fc) >= 1
    # Rest-of-FY and next FY both from one printed-origin BU call.
    assert (fc["fc_source"] == "bottom_up_macro").all()
    # live = bottom-up + legislated policy (policy.monthly, the scenario pack's overlay)
    assert np.allclose(
        fc["cash_requirement_bn"].to_numpy(float),
        (fc["cash_requirement_bottomup_bn"] + fc["policy_bn"].fillna(0.0)).to_numpy(float),
        equal_nan=False,
    )
