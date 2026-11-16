"""French and Italian daily yields: parsers, splice, and how they enter issuance.market."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from issuance import fetch_ea_yields as ea
from issuance import market

K10 = "FM.D.FR.EUR.FR2.BB.FRMOYTEC10.HSTA"
BDF = json.dumps([
    {"series_key": K10, "time_period_start": "2024-01-02", "time_period_end": "2024-01-02", "obs_value": 2.61},
    {"series_key": K10, "time_period_start": "2024-01-03T00:00:00+00:00", "obs_value": "2.65"},
    {"series_key": K10, "time_period_start": "2024-01-04", "obs_value": None},
    {"series_key": "FM.D.FR.EUR.FR2.BB.FRMOYTEC2.HSTA", "time_period_start": "2024-01-03", "obs_value": 2.4},
    {"series_key": K10, "time_period_start": "2099-01-02", "obs_value": 9.99},
])

BDI = ('"DATA_OSS";"MFN_BMK.D.001A.922.0.EUR.9";"MFN_BMK.D.020.922.0.EUR.203";"MFN_BMK.D.020.922.0.EUR.205";'
       '"MFN_BMK.D.020.922.0.EUR.210";"MFN_BMK.D.020.922.0.EUR.230";"MFN_BMK.D.035A.922.0.EUR.9"\n'
       '"2024/01/08";"";"3.1";"3.3";"3.75";"4.4";"4.0"\n'
       '"2024/01/07";"";"";"";"";"";""\n'
       '"2024/01/05";"";"3,05";"3,25";"3,70";"4,35";"3,9"\n')


def _bars(rows):
    return json.dumps({"barData": {"priceBars": [
        {"open": str(v), "high": str(v), "low": str(v), "close": str(v), "volume": 0, "tradeTime": f"{d}000000"}
        for d, v in rows]}})


def test_parse_bdf_percent_and_no_future_dates():
    s = ea.parse_bdf_json(BDF, K10)
    assert list(s.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert s.iloc[-1] == pytest.approx(2.65)  # percent, not decimal or bp
    r = ea.parse_bdf_json(json.dumps({"total_count": 1, "results": json.loads(BDF)[:1]}), K10)
    assert r.tolist() == [pytest.approx(2.61)]  # /records shape


def test_bdf_without_key_skips_and_france_falls_back_to_cnbc(monkeypatch, tmp_path):
    monkeypatch.delenv(ea.BDF_KEY_ENV, raising=False)
    monkeypatch.setattr(ea, "BDF_KEY_FILE", tmp_path / "webstat.env")
    monkeypatch.setattr(ea, "RAW", tmp_path)
    monkeypatch.setattr(ea.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    assert ea.api_key() is None
    with pytest.raises(ea.NoApiKey):
        ea.bdf_tec(10)
    (tmp_path / "cnbc_FR10Y-FR.json").write_text(_bars([("20240105", 3.0), ("20240108", 3.1)]))
    d = ea.daily(offline=True)
    fr = d[d["country"] == "FR"]
    assert set(fr["series"]) == {"y10"} and fr["source"].str.startswith("CNBC FR10Y-FR").all()
    # a key from the gitignored file is read; a cached keyed download is parsed without a key
    (tmp_path / "webstat.env").write_text(f"# comment\n{ea.BDF_KEY_ENV}='abc'\n")
    assert ea.api_key() == "abc"
    (tmp_path / "webstat.env").unlink()
    (tmp_path / "bdf_tec10.json").write_text(BDF)
    s = ea.bdf_tec(10)
    assert s.index.max() == pd.Timestamp("2024-01-03")
    fr = ea.daily(offline=True).query("country == 'FR'")
    assert fr["source"].tolist() == [ea.SRC_BDF.format(t=10)] * 2 + [ea.SRC_CNBC_FR.format(t=10)] * 2


def test_parse_bdi_maps_tenors_and_both_decimal_marks():
    d = ea.parse_bdi_csv(BDI)
    assert list(d.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")]  # sorted, blank day dropped
    assert d.loc["2024-01-05", "y10"] == pytest.approx(3.70)
    assert d.loc["2024-01-08", "y3"] == pytest.approx(3.1)
    assert d["ctz2"].isna().all()
    assert {"y3", "y5", "y10", "y30"} <= set(d.columns)


def test_parse_cnbc_drops_weekend_bars_and_future():
    raw = _bars([("20240105", 3.7), ("20240106", 3.71), ("20240108", 3.8), ("20990101", 9.0)])
    s = ea.parse_cnbc(raw)
    assert list(s.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")]
    assert s.iloc[0] == pytest.approx(3.7)


def test_splice_uses_tail_only_after_official():
    off = pd.Series([1.0, 1.1], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    tail = pd.Series([9.0, 9.1, 1.2], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    sp = ea.splice_tail(off, tail)
    assert sp["origin"].tolist() == ["official", "official", "tail"]
    assert sp["value"].tolist() == [1.0, 1.1, 1.2]


def test_ea_series_records_tail_and_official_source():
    f = pd.DataFrame({"country": "IT", "series": "y10", "date": pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"]),
                      "value": [3.0, 3.1, 3.2], "source": ["BdI", "BdI", "CNBC"]})
    (c, name, src, s), = market.ea_series(f)
    assert (c, name, src) == ("IT", "y10", "BdI") and len(s) == 3
    assert s.attrs["tail_from"] == pd.Timestamp("2024-02-01") and s.attrs["tail_source"] == "CNBC"


def _fake_sets(today):
    """FR/IT/DE daily series on business days through `today`; IT 10y has a vendor tail in the last month."""
    idx = pd.bdate_range("2019-01-01", today)
    rng = np.random.default_rng(0)
    walk = lambda lvl: pd.Series(lvl + np.cumsum(rng.normal(0, 0.03, len(idx))), index=idx)
    it10 = walk(2.5)
    it10.attrs = {"tail_from": pd.Timestamp("2024-04-15"), "tail_source": "CNBC IT10Y-IT"}
    sets = [("DE", "y2", "Bundesbank", walk(0.5)), ("DE", "y10", "Bundesbank", walk(1.0)),
            ("FR", "y2", "BdF TEC2", walk(0.8)), ("FR", "y10", "BdF TEC10", walk(1.5)), ("FR", "y30", "BdF TEC30", walk(2.0)),
            ("IT", "y3", "BdI 3y", walk(1.5)), ("IT", "y10", "BdI 10y", it10), ("IT", "y30", "BdI 30y", walk(3.0))]
    for c in ("DE", "FR", "IT"):
        sets.append((c, "r3m", "euro 3m", pd.Series(0.5, index=pd.date_range("2019-01-01", today, freq="MS"))))
    return sets


def test_build_monthly_convention_and_fr_it_derived(monkeypatch):
    today = pd.Timestamp("2024-05-20")
    sets = _fake_sets(today)
    monkeypatch.setattr(market, "_daily_sets", lambda: sets)
    out = market.build(today=today)
    get = lambda c, s: out[(out["country"] == c) & (out["series"] == s)].set_index("month")["value"]
    fr10 = dict((c, s) for c, n, _, s in sets if n == "y10")["FR"]
    # complete calendar months only, simple mean of the daily values
    assert get("FR", "y10").index.max() == pd.Timestamp("2024-04-01")
    assert get("FR", "y10").loc["2024-03-01"] == pytest.approx(fr10.loc["2024-03"].mean())
    # slope proxies on each country's own short leg; s1030 now exists
    assert np.allclose(get("FR", "tp10"), (get("FR", "y10") - get("FR", "y2")).dropna())
    assert np.allclose(get("IT", "tp10"), (get("IT", "y10") - get("IT", "y3")).dropna())
    assert np.allclose(get("IT", "s1030"), (get("IT", "y30") - get("IT", "y10")).dropna())
    # own realised vol, not the DE proxy
    src = out[(out["country"] == "FR") & (out["series"] == "rvol10")]["source"]
    assert src.str.contains("BdF TEC10").all()
    # IT roll-down uses the 3y
    rx = market.excess_return(get("IT", "y10"), get("IT", "r3m"), 3, get("IT", "y3"), short_years=3.0)
    assert np.allclose(get("IT", "rx10_3").sort_index(), rx[rx.index >= get("IT", "y3").index.min()].sort_index())
    # a month that uses the vendor tail says so
    it = out[(out["country"] == "IT") & (out["series"] == "y10")].set_index("month")["source"]
    assert "CNBC" not in it.loc["2024-03-01"] and "CNBC IT10Y-IT from 2024-04-15" in it.loc["2024-04-01"]
    # rel10 peer mean is still the four daily markets only
    assert market.PEERS == ("UK", "DE", "US", "JP")


def test_short_leg_starting_late_keeps_rx_and_tp10_history(monkeypatch):
    today = pd.Timestamp("2024-05-20")
    sets = [(c, n, src, s[s.index >= "2022-03-01"] if (c, n) == ("FR", "y2") else s)
            for c, n, src, s in _fake_sets(today)]
    monkeypatch.setattr(market, "_daily_sets", lambda: sets)
    out = market.build(today=today)
    get = lambda c, s: out[(out["country"] == c) & (out["series"] == s)].set_index("month")
    tp, rx = get("FR", "tp10"), get("FR", "rx10_3")
    assert tp.index.min() == pd.Timestamp("2019-01-01") and rx.index.min() == pd.Timestamp("2019-01-01")
    early, late = pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-01")
    assert tp.loc[early, "value"] == pytest.approx(get("FR", "y10").loc[early, "value"] - get("DE", "y2").loc[early, "value"])
    assert tp.loc[late, "value"] == pytest.approx(get("FR", "y10").loc[late, "value"] - get("FR", "y2").loc[late, "value"])
    assert "Bund 2y" in tp.loc[early, "source"] and "Bund" not in tp.loc[late, "source"]
    assert "no roll-down" in rx.loc[early, "source"] and "2y roll-down" in rx.loc[late, "source"]
    assert not rx.index.duplicated().any() and not tp.index.duplicated().any()


def test_excess_return_default_short_leg_is_the_2y():
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    y10 = pd.Series(np.linspace(1, 3, 30), index=idx)
    y2 = y10 - 1
    r = pd.Series(0.5, index=idx)
    a = market.excess_return(y10, r, 12, y2)
    b = market.excess_return(y10, r, 12, y2, short_years=2.0)
    pd.testing.assert_series_equal(a, b)


def test_merge_daily_replaces_only_fr_it_rows():
    old = pd.DataFrame({"country": ["US", "FR"], "series": ["y10", "y10"],
                        "date": pd.to_datetime(["2024-01-02", "2023-01-02"]), "value": [4.0, 9.9]})
    new = pd.DataFrame({"country": ["FR", "FR", "IT"], "series": ["y10", "y5", "y3"],
                        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
                        "value": [2.6, 2.4, 3.1], "source": ["BdF", "BdF", "BdI"]})
    m = market.merge_daily(old, new, "2000-01-01")
    assert set(zip(m["country"], m["series"])) == {("US", "y10"), ("FR", "y10"), ("IT", "y3")}  # y5 not written
    assert m.loc[m["country"] == "FR", "value"].tolist() == [2.6]
    assert m.loc[m["country"] == "US", "value"].tolist() == [4.0]


def _cached_oecd(country):
    p = market.FRED_CACHE / f"{market.OECD[country]}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0])).dropna()


@pytest.mark.parametrize("country,tol_mean,tol_max", [("FR", 0.04, 0.15), ("IT", 0.005, 0.02)])
def test_daily_ten_year_monthly_mean_matches_oecd(country, tol_mean, tol_max):
    """FR TEC10 vs the OECD FR long rate (its own benchmark basis); IT's OECD rate is the BdI benchmark."""
    try:
        d = ea.daily(offline=True)
    except Exception:  # noqa: BLE001
        pytest.skip("raw cache absent")
    o = _cached_oecd(country)
    s = d[(d["country"] == country) & (d["series"] == "y10") & ~d["source"].str.startswith("CNBC")]
    if o is None or s.empty:
        pytest.skip("caches absent")
    s = s.set_index("date")["value"]
    m = s.groupby(s.index.to_period("M").to_timestamp()).mean()
    m = m[m.index < s.index.max().to_period("M").to_timestamp()]
    j = pd.concat([m.rename("d"), o.rename("o")], axis=1, join="inner").dropna()
    assert len(j) > 200
    diff = (j["d"] - j["o"]).abs()
    assert diff.mean() < tol_mean and diff.max() < tol_max
    assert s.index.max() <= pd.Timestamp.today()
    assert s.between(-2, 20).all()  # percent


def test_market_cache_fr_it_daily_series():
    mk = market.load()
    fr = mk[mk["country"] == "FR"]
    if "y2" not in set(fr["series"]):
        pytest.skip("market cache without FR daily")
    for c, need in (("FR", {"y2", "y10", "y30", "s1030", "tp10", "rvol10", "rx10_3"}),
                    ("IT", {"y3", "y10", "y30", "s1030", "tp10", "rvol10", "rx10_3"})):
        assert need <= set(mk.loc[mk["country"] == c, "series"])
    assert not (mk["month"] >= pd.Timestamp.today().to_period("M").to_timestamp()).any()
