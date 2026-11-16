import json

import numpy as np
import pandas as pd
import pytest

from issuance import fetch_egb as E

BDE_CSV = (
    '"CÓDIGO DE LA SERIE",D_DTES00B7,D_G0B1F0ZN,D_G0B1F0ZO,D_G0B1F0ZP,D_G0B1F0ZQ\n'
    '"FRECUENCIA","DIARIA","DIARIA","DIARIA","DIARIA","DIARIA"\n'
    '"02 ENE 2024",3.8,2.95,2.90,"_",3.40\n'
    '"03 ENE 2024",3.8,2.97,2.92,3.05,3.42\n'
    '"30 DIC 2024","_",2.30,2.50,3.00,3.20\n'
    '"NOTAS","","","","",""\n'
)

NBB_CSV = (
    "DATAFLOW,FREQ,IROLOBE2_MATUR,IROLOBE2_TYPE,TIME_PERIOD,OBS_VALUE,OBS_STATUS,DECIMALS\n"
    "BE2:DF_IROLOBE2(1.0),D,10Y,F,2026-09-01,3.92,A,2\n"
    "BE2:DF_IROLOBE2(1.0),D,10Y,A,2026-09-01,3.76,A,2\n"
    "BE2:DF_IROLOBE2(1.0),D,2Y,F,2026-09-01,2.31,A,2\n"
    "BE2:DF_IROLOBE2(1.0),D,10Y,F,2026-09-02,3.94,A,2\n"
    "BE2:DF_IROLOBE2(1.0),D,12YM,A,2026-09-02,4.10,A,2\n"
)

BOF_CSV = (
    "\ufefftitle,description\nYields,\"blah\"\n\n"
    "txtb_row_title,title,txtb_currency,txtb_value\n"
    "Period,1.10.2026,\"Yield on goverment bonds, 5 year\",3.52\n"
    "Period,1.10.2026,\"Yield on goverment bonds, 10 year\",3.92\n"
    "Period,1.10.2026,Loan period 30 Aug 2022 – 15 Apr 2027,\n"
    "Period,1.10.2026,Loan period 7 Feb 2012 – 4 Jul 2028,3.25\n"
    "Period,30.9.2026,\"Yield on goverment bonds, 5 year\",3.54\n"
)


def _jsonstat():
    return json.dumps({
        "id": ["45", "40", "reference_date"], "size": [2, 1, 3],
        "dimension": {
            "45": {"category": {"index": {"2743": 0, "2740": 1}, "label": {"2743": "2 years", "2740": "10 years"}}},
            "40": {"category": {"index": {"4263": 0}, "label": {"4263": "Daily"}}},
            "reference_date": {"category": {"index": ["2026-09-23", "2026-09-24", "2026-09-25"]}},
        },
        "value": [2.1, 2.2, None, 3.1, 3.2, 3.3],
    })


def test_parse_bde_csv_spanish_dates_and_missing():
    w = E.parse_bde_csv(BDE_CSV)
    assert list(w.columns) == ["y3", "y5", "y10", "y15"]
    assert w.index[0] == pd.Timestamp("2024-01-02") and w.index[-1] == pd.Timestamp("2024-12-30")
    assert np.isnan(w.loc["2024-01-02", "y10"]) and w.loc["2024-01-03", "y10"] == pytest.approx(3.05)


def test_parse_nbb_keeps_fixed_term_only():
    w = E.parse_nbb_csv(NBB_CSV)
    assert set(w.columns) == {"y2", "y10"}
    assert w.loc["2026-09-01", "y10"] == pytest.approx(3.92)


def test_parse_jsonstat_maps_tenors():
    w = E.parse_jsonstat(_jsonstat(), "45", E.BP_SERIES)
    assert w.loc["2026-09-25", "y10"] == pytest.approx(3.3)
    assert np.isnan(w.loc["2026-09-25", "y2"]) and w.loc["2026-09-23", "y2"] == pytest.approx(2.1)


def test_parse_bof_benchmarks_and_loans():
    bm, loans = E.parse_bof_csv(BOF_CSV)
    assert bm.loc["2026-10-01", "y10"] == pytest.approx(3.92) and bm.loc["2026-09-30", "y5"] == pytest.approx(3.54)
    assert len(loans) == 1 and loans["maturity"].iloc[0] == pd.Timestamp("2028-07-04")


def test_dsta_frame_fraction_to_percent_and_drops_matured():
    hdr = ["RefDate", "Type", "ISIN Code", "Description", "Maturity", "Mid Price", "Mid Yield"]
    rows = [["2026-01-02T00:00:00", "DSL", "NL1", "NETHER 0.750 15/07/27", "07/15/2027", "98", "2.0645E-2"],
            ["2026-01-02T00:00:00", "DSL", "NL2", "NETHER 7.500 15/01/23", "01/15/2023", "0", "0"],
            ["2026-01-02T00:00:00", "DTC", "NL3", "DTC 0 15/03/26", "03/15/2026", "99", "1.9E-2"]]
    d = E._dsta_frame(rows, hdr)
    assert len(d) == 1
    assert d["yld"].iloc[0] == pytest.approx(2.0645) and d["maturity"].iloc[0] == pd.Timestamp("2027-07-15")


def test_cm_from_bonds_interpolates_and_limits():
    dt = pd.Timestamp("2026-01-01")
    b = pd.DataFrame({"date": dt, "maturity": [dt + pd.Timedelta(days=round(365.25 * r)) for r in (1.0, 3.0, 9.0, 11.0, 18.0, 29.0)],
                      "yld": [2.0, 2.4, 3.0, 3.2, 3.6, 4.0]})
    w = E.cm_from_bonds(b)
    assert w.loc[dt, "y2"] == pytest.approx(2.2, abs=1e-3)
    assert w.loc[dt, "y10"] == pytest.approx(3.1, abs=1e-3)
    assert w.loc[dt, "y30"] == pytest.approx(4.0)          # 1y past the longest bond: held flat
    assert "y5" not in w or np.isnan(w.loc[dt, "y5"])        # 3y–9y bracket wider than MAX_GAP[5]
    assert w.loc[dt, "y20"] == pytest.approx(3.6 + 0.4 * 2 / 11, abs=1e-3)


def test_splice_tail_only_after_official_and_labelled():
    off = pd.Series([3.0, 3.1], index=pd.to_datetime(["2026-09-01", "2026-09-02"]))
    tail = pd.Series([9.0, 3.2, 3.3], index=pd.to_datetime(["2026-09-01", "2026-09-03", "2026-09-04"]))
    sp = E.splice(off, tail, "OFF", "CNBC x")
    assert list(sp["value"]) == [3.0, 3.1, 3.2, 3.3]
    assert list(sp["source"]) == ["OFF", "OFF", "CNBC x", "CNBC x"]


def test_clean_drops_future_and_absurd():
    s = pd.Series([3.0, 99.0, 3.1], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2099-01-01"]))
    out = E._clean(s, today=pd.Timestamp("2026-06-01"))
    assert list(out) == [3.0]


def test_monthly_complete_months_mean_and_irs_fill():
    dates = pd.bdate_range("2026-07-01", "2026-09-15")
    d = pd.DataFrame({"country": "ES", "series": "y10", "date": dates, "value": 3.0, "source": "BdE"})
    d.loc[d["date"].dt.month == 8, "value"] = 3.5
    irs = pd.DataFrame({"country": "ES", "month": pd.to_datetime(["2026-06-01", "2026-07-01"]), "value": [2.9, 9.9]})
    m = E.monthly(d, today=pd.Timestamp("2026-09-20"), irs=irs)
    y = m.set_index("month")
    assert list(y.index) == list(pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"]))  # Sep incomplete
    assert y.loc["2026-08-01", "value"] == pytest.approx(3.5)
    assert y.loc["2026-07-01", "value"] == pytest.approx(3.0)       # daily wins over IRS
    assert y.loc["2026-06-01", "source"].startswith("ECB IRS") and y.loc["2026-06-01", "value"] == pytest.approx(2.9)


def test_spread_de_sign():
    dt = pd.to_datetime(["2026-09-01"])
    d = pd.DataFrame({"country": ["DE", "ES"], "series": "y10", "date": [dt[0], dt[0]], "value": [2.5, 3.2], "source": "x"})
    sp = E.spread_de(d)
    assert sp.loc[sp["country"] == "ES", "value"].iloc[0] == pytest.approx(0.7)


# ------------------------------------------------------------------ built cache
cache = pytest.mark.skipif(not E.OUT.exists(), reason="cache/egb_daily.parquet not built")


@cache
def test_cache_units_dates_and_coverage():
    d = pd.read_parquet(E.OUT)
    assert set(d.columns) == set(E.COLS)
    assert d["value"].between(-2.0, 30.0).all()  # PT 2y/5y printed ~22% in Jan-2012
    assert d["date"].max() <= pd.Timestamp.today().normalize()
    assert not d.duplicated(["country", "series", "date"]).any()
    assert d["source"].notna().all() and (d["source"] != "").all()
    have = set(d.loc[d["series"] == "y10", "country"])
    assert {"ES", "NL", "BE", "PT", "FI", "DE"} <= have
    first = d[(d["series"] == "y10") & ~d["source"].str.startswith("CNBC")].groupby("country")["date"].min()
    for c in ("ES", "BE", "PT", "FI"):
        assert first[c] <= pd.Timestamp("2000-01-03")
    assert first["NL"] <= pd.Timestamp("2002-03-01")  # DSTA 10y bracket is sparse 2002-05


@cache
@pytest.mark.skipif(not (E.RAW / "ecb_irs_m.csv").exists(), reason="ECB IRS raw not cached")
def test_overlap_vs_ecb_irs_monthly_10y():
    ov = E.overlap().set_index(["country", "kind"])
    for c in ("ES", "NL", "BE", "PT", "FI", "DE"):
        r = ov.loc[(c, "official")]
        assert r["n"] >= 120
        assert r["mean_abs_bp"] <= 15.0, (c, r.to_dict())
