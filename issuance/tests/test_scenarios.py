import numpy as np
import pandas as pd
import pytest

from issuance.scenarios import FRONT, GAP_BN, GAP_END, extra_flow, full_increment_bn, phase_weight


def test_phase_weight_steps_per_refunding_quarter():
    months = pd.date_range("2027-01-01", "2027-12-01", freq="MS")
    w = phase_weight(months, pd.Timestamp("2027-02-01"))
    assert list(np.round(w * 3).astype(int)) == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]


@pytest.mark.parametrize("start,share", [("2027-02-01", 1.0), ("2027-05-01", 0.778)])
def test_full_increment_funds_the_gap_share(start, share):
    x = full_increment_bn(start, share)
    months = pd.date_range(start, GAP_END, freq="MS")
    assert x * phase_weight(months, pd.Timestamp(start)).sum() == pytest.approx(share * GAP_BN)
    assert full_increment_bn(None, 1.0) == 0.0


def _fwd():
    rows = []
    for m in pd.date_range("2027-04-01", "2027-07-01", freq="MS"):
        for tok, size, b in [("2y", 69000, "1-3y"), ("5y", 70000, "3-7y"), ("30y", 22000, "20-30y"), ("frn", 28000, "1-3y")]:
            rows.append({"month": m, "tok": tok, "gross_mn": size, "tenor_bucket": b, "dv01_per_mn": 0.001})
    return pd.DataFrame(rows)


def test_mix_puts_nothing_in_terms_outside_it():
    sc = {"start": "2027-05-01", "gap_share": 0.778, "mix": FRONT}
    out = extra_flow(sc, _fwd())
    assert out.loc[out["month"] == "2027-04-01", "d_face_mn"].sum() == 0
    assert out.loc[out["tenor_bucket"] == "20-30y", "d_face_mn"].sum() == 0
    x = full_increment_bn(sc["start"], sc["gap_share"]) * 1000
    may = out.loc[out["month"] == "2027-05-01", "d_face_mn"].sum()
    # 3y and 7y are in the mix but not in this toy forward, so only their share is missing.
    present = (FRONT["2y"] + FRONT["5y"] + FRONT["frn"]) / sum(FRONT.values())
    assert may == pytest.approx(x / 3 * present)


def test_prorata_spreads_by_size_and_sums_to_increment():
    sc = {"start": "2027-05-01", "gap_share": 1.0, "mix": "prorata"}
    out = extra_flow(sc, _fwd())
    may = out[out["month"] == "2027-05-01"]
    x = full_increment_bn(sc["start"], 1.0) * 1000
    assert may["d_face_mn"].sum() == pytest.approx(x / 3)
    assert may.loc[may["tenor_bucket"] == "20-30y", "d_face_mn"].sum() == pytest.approx(x / 3 * 22000 / 189000)
