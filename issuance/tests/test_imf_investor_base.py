import numpy as np
import pandas as pd
import pytest

from issuance import imf_investor_base as ib

QUARTERS = ["2004Q1", "2004Q2", "2004Q3", "2004Q4", "2005Q1", "2005Q2"]
NAMES = ["Australia", "Germany", "Japan", "United Kingdom", "United States"]
# per-type holdings (bn LC), constant across quarters except a ramp on fnb
BASE = {"fo": 10.0, "fb": 5.0, "fnb": 20.0, "dcb": 15.0, "db": 30.0, "dnb": 20.0}


def _sheet(header: str, values: dict[str, list[float]], tail: bool = True) -> pd.DataFrame:
    rows = [[header, *QUARTERS]] + [[n, *v] for n, v in values.items()]
    if tail:
        rows += [[None] * (len(QUARTERS) + 1), ["Percent of Total Debt", *QUARTERS], ["Germany", *[99.0] * len(QUARTERS)]]
    return pd.DataFrame(rows)


@pytest.fixture
def workbook(tmp_path):
    ramp = np.arange(len(QUARTERS), dtype=float)
    types = {t: {n: list(v + (ramp if t == "fnb" else 0.0 * ramp)) for n in NAMES} for t, v in BASE.items()}
    total = {n: [sum(types[t][n][i] for t in BASE) for i in range(len(QUARTERS))] for n in NAMES}
    types["fnb"]["Japan"][2] = "…"
    path = tmp_path / "Database_For_Posting_AM_Frozen.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame([[None, "Sovereign Debt Investor Base for Advanced Economies"],
                      [None, "(This Version: May 1, 2026)"]]).to_excel(xw, sheet_name="ReadMe", header=False, index=False)
        _sheet("Billion LC", total).to_excel(xw, sheet_name=ib.SHEETS["total"], header=False, index=False)
        for t, sheet in ib.SHEETS.items():
            if t != "total":
                _sheet("Billion LC", types[t]).to_excel(xw, sheet_name=sheet, header=False, index=False)
        fx = _sheet("National currency/USD", {n: [2.0] * len(QUARTERS) for n in NAMES}, tail=False)
        dg = _sheet("Total Debt/GDP", {n: [50.0] * len(QUARTERS) for n in NAMES}, tail=False)
        pd.concat([fx, pd.DataFrame([[None] * fx.shape[1]]), dg], ignore_index=True).to_excel(
            xw, sheet_name="FX", header=False, index=False)
    return path


def test_parse_fixture(workbook):
    p = ib.parse(workbook)
    assert set(p["country"]) == {"DE", "JP", "UK", "US"}
    assert set(p["holder_type"]) == set(ib.TYPES) | {"total"}
    assert p["vintage"].iloc[0] == pd.Timestamp("2026-05-01")
    assert (p["basis"] == "nominal").all()
    assert p["quarter_end"].min() == pd.Timestamp("2004-03-31")
    # the 'Percent of Total Debt' block below the first block is not read
    assert (p["value_local_bn"] != 99.0).all()
    de = p[(p["country"] == "DE") & (p["quarter_end"] == pd.Timestamp("2004-03-31"))].set_index("holder_type")
    assert de.loc["total", "value_local_bn"] == pytest.approx(100.0)
    assert de.loc["fnb", "value_usd_bn"] == pytest.approx(10.0)
    assert de.loc["total", "gdp_local_bn"] == pytest.approx(200.0)
    # '…' is missing, not zero
    jp = p[(p["country"] == "JP") & (p["holder_type"] == "fnb")]
    assert len(jp) == len(QUARTERS) - 1


def test_shares_sum_fixture(workbook):
    p = ib.parse(workbook)
    six = p[p["holder_type"].isin(ib.TYPES)].groupby(["country", "quarter_end"])["share_pct"].sum()
    six = six.drop(index=("JP", pd.Timestamp("2004-09-30")))
    assert np.allclose(six, 100.0)


def test_country_map():
    assert ib.COUNTRIES["United Kingdom"] == "UK" and ib.COUNTRIES["United States"] == "US"
    assert set(ib.COUNTRIES.values()) == {"UK", "DE", "FR", "IT", "ES", "US", "JP", "SE", "NO"}


@pytest.mark.parametrize("qe,pub", [("2025-12-31", "2026-05-01"), ("2025-09-30", "2026-05-01"),
                                    ("2025-06-30", "2025-11-01"), ("2025-03-31", "2025-11-01"),
                                    ("2024-12-31", "2025-05-01")])
def test_public_month(qe, pub):
    assert ib.public_month(pd.Timestamp(qe)) == pd.Timestamp(pub)


def test_public_month_lag_bounds():
    for q in pd.date_range("2004-03-31", "2025-12-31", freq="QE"):
        lag = (ib.public_month(q).to_period("M") - q.to_period("M")).n
        assert 5 <= lag <= 8


def test_metrics_no_lookahead(workbook):
    p = ib.parse(workbook)
    m = ib.metrics(p, today=pd.Timestamp("2006-03-15"))
    us = m[(m["country"] == "US") & (m["metric"] == "fnb_share")].set_index("month")["value"]
    # 2004Q1/Q2 first public Nov-2004 (Q2 wins), Q3/Q4 May-2005, Q1/Q2 2005 Nov-2005
    assert us.index.min() == pd.Timestamp("2004-11-01")
    assert us.index.max() == pd.Timestamp("2006-02-01")
    q = ib.quarterly_metrics(p)
    qv = q[(q["country"] == "US") & (q["metric"] == "fnb_share")].set_index("quarter_end")["value"]
    assert us[pd.Timestamp("2004-11-01")] == pytest.approx(qv[pd.Timestamp("2004-06-30")])
    assert us[pd.Timestamp("2005-04-01")] == pytest.approx(qv[pd.Timestamp("2004-06-30")])
    assert us[pd.Timestamp("2005-05-01")] == pytest.approx(qv[pd.Timestamp("2004-12-31")])
    assert us[pd.Timestamp("2006-02-01")] == pytest.approx(qv[pd.Timestamp("2005-06-30")])
    # every stamped value was public at its month
    for month, v in us.items():
        public = [qe for qe in qv.index if ib.public_month(qe) <= month]
        assert v == pytest.approx(qv[max(public)])
    assert set(m.columns) == {"country", "month", "metric", "value", "phase"}
    assert (m["phase"] == "history").all()


def test_metric_definitions(workbook):
    p = ib.parse(workbook)
    q = ib.quarterly_metrics(p)
    de = q[(q["country"] == "DE") & (q["quarter_end"] == pd.Timestamp("2004-03-31"))].set_index("metric")["value"]
    assert de["fnb_share"] == pytest.approx(20.0)
    assert de["elastic_share"] == pytest.approx(40.0)
    assert de["cb_hold_share"] == pytest.approx(15.0)
    assert de["ibr"] == pytest.approx((30 * 26 + 20 * 16 + 10 * 14 + 5 * 87 + 20 * 100) / 100)
    d = q[(q["country"] == "DE") & (q["metric"] == "d_fnb_gdp")].set_index("quarter_end")["value"]
    # fnb +4 over four quarters on GDP of total / 0.5
    assert d[pd.Timestamp("2005-03-31")] == pytest.approx(100 * 4 / (104 / 0.5))


def test_metric_signs():
    s = {k: v["sign"] for k, v in ib.METRICS.items()}
    assert s["fnb_share"] == 1 and s["elastic_share"] == 1 and s["ibr"] == 1
    assert s["cb_hold_share"] == -1 and s["d_dcb_gdp"] == -1 and s["d_fo_gdp"] == -1
    assert all(s[f"d_{t}_gdp"] == 1 for t in ("db", "dnb", "fb", "fnb", "priv"))
    assert set(ib.ELASTIC) == {"dnb", "fnb"}
    assert ib.IRI_SCORE["dcb"] == 0 and ib.IRI_SCORE["fnb"] == 100
    for meta in ib.METRICS.values():
        assert {"label", "unit", "sign", "fmt", "what"} <= set(meta)


cached = pytest.mark.skipif(not any(ib.DIR.glob("Database*.xlsx")) if ib.DIR.exists() else True,
                            reason="IMF workbook not cached")


@cached
def test_cached_workbook():
    p = ib.parse()
    assert set(p["country"]) == set(ib.COUNTRIES.values())
    assert p["quarter_end"].min() == pd.Timestamp("2004-03-31")
    six = p[p["holder_type"].isin(ib.TYPES)].groupby(["country", "quarter_end"])["share_pct"].sum()
    assert (six - 100).abs().max() < 0.5
    w = ib.wide(p)
    assert (w["gdp"] > 0).all()
    us = w[(w["country"] == "US") & (w["quarter_end"] == pd.Timestamp("2020-12-31"))]
    assert 15000 < float(us["total"].iloc[0]) < 40000
