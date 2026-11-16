from issuance.fetch_fx import _series_from_csv


def test_ecb_sdw_columns():
    csv = (
        b"KEY,FREQ,CURRENCY,TIME_PERIOD,OBS_VALUE\n"
        b"EXR,D,USD,2024-01-02,1.10\n"
        b"EXR,D,USD,2024-01-03,1.11\n"
    )
    s = _series_from_csv(csv)
    assert len(s) == 2
    assert abs(float(s.iloc[0]) - 1.10) < 1e-9


def test_fred_columns():
    csv = b"observation_date,DEXUSEU\n2024-01-02,1.10\n2024-01-03,1.11\n"
    s = _series_from_csv(csv)
    assert len(s) == 2
    assert abs(float(s.iloc[-1]) - 1.11) < 1e-9
