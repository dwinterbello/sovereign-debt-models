import pandas as pd

from issuance.demand import normalize_btc


def test_normalize_btc_rescales_it():
    s = pd.Series([1.8, 180.0, 2.4, 156.0])
    out = normalize_btc(s)
    assert abs(out.iloc[0] - 1.8) < 1e-9
    assert abs(out.iloc[1] - 1.8) < 1e-9
    assert abs(out.iloc[2] - 2.4) < 1e-9
    assert abs(out.iloc[3] - 1.56) < 1e-9


def test_d21a_sep_2026_row():
    from issuance.demand import UK_D21A, _parse_d21a
    if not UK_D21A.exists():
        return
    out = _parse_d21a(UK_D21A)
    row = out[out["date"] == "2026-09-22"].iloc[0]
    assert abs(row["total_nominal_mn"] - 5527.75) < 1e-6
    assert abs(row["nominal_auction_mn"] + row["nominal_paof_mn"] - row["total_nominal_mn"]) < 0.05
    assert abs(row["cover"] - 3.07) < 1e-9
    assert abs(row["tail_bp"] - 0.4) < 1e-9


def test_normalize_btc_leaves_de():
    s = pd.Series([1.2, 2.4, 3.2])
    out = normalize_btc(s)
    assert list(out.round(2)) == [1.2, 2.4, 3.2]
