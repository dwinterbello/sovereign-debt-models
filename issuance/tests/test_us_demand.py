import numpy as np
import pandas as pd
import pytest

from issuance import fetch_us_demand as ud

RAW = ud.RAW
COUPONS = sorted(RAW.glob("*IC-Coupons*.xls"))
BILLS = sorted(RAW.glob("*IC-Bills*.xls"))
OLD_COUPONS = RAW / "Website-PDO-4-A-Coupons-Jan-2000-Sep 2009.xls"
needs_cache = pytest.mark.skipif(not COUPONS or not BILLS, reason="allotment files not cached")


def test_allotment_links_quoted_and_unquoted():
    html = ('<a href=/system/files/276/September-22_2026-IC-Coupons.xls>x</a>'
            '<a href="/system/files/276/Website-PDO-4-A-Coupons-Jan-2000-Sep 2009.xls" rel=x>y</a>')
    assert ud.allotment_links(html) == ["/system/files/276/September-22_2026-IC-Coupons.xls",
                                        "/system/files/276/Website-PDO-4-A-Coupons-Jan-2000-Sep 2009.xls"]


@pytest.mark.parametrize("name,date", [("September-22_2026-IC-Coupons.xls", "2026-09-22"),
                                       ("December%207_2021-IC-Bills.xls", "2021-12-07"),
                                       ("March%2024-2025-IC-Coupons.xls", "2025-03-24"),
                                       ("August-8-2022-IC-Coupons_0.xls", "2022-08-08")])
def test_posting_date(name, date):
    assert ud._posting_date(name) == pd.Timestamp(date)


def test_posting_date_history_files_have_none():
    assert ud._posting_date("Website-PDO-4-A-Coupons-Jan-2000-Sep 2009.xls") is None


@needs_cache
def test_current_coupon_parse_and_shares():
    d = ud.parse_allotment_xls(COUPONS[-1])
    assert set(d["security_type"]) <= {"Note", "Bond", "TIPS", "FRN"}
    priv = d["total_bn"] - d["soma_bn"]
    share = d[list(ud.CLASSES)].sum(axis=1) / priv
    assert (share.sub(1).abs() < 0.002).all()
    ten = d[d["term_label"].str.contains("10-Year Note")]
    assert 15 < ten["total_bn"].median() < 80  # $bn, not $mn
    assert (ten["term_years"] == 10).all()


@needs_cache
def test_old_coupon_file_scaled_to_bn():
    if not OLD_COUPONS.exists():
        pytest.skip("history file not cached")
    d = ud.parse_allotment_xls(OLD_COUPONS)
    first = d.sort_values("issue_date").iloc[0]
    assert first["issue_date"] == pd.Timestamp("2000-01-18") and first["security_type"] == "TIPS"
    assert first["total_bn"] == pytest.approx(6.318185, rel=1e-6)
    assert d["term_years"].notna().all()


@needs_cache
def test_bills_parse():
    d = ud.parse_allotment_xls(BILLS[-1])
    assert set(d["security_type"]) <= {"Bill", "CMB"}
    b13 = d[d["term_label"].str.contains("13-Week")]
    assert b13["term_years"].iloc[0] == pytest.approx(0.25, abs=1e-6)


@needs_cache
def test_panel_shares_sum_to_100():
    p = ud.allotments_panel()
    tot = p.groupby(["issue_date", "cusip"])["share_pct"].sum()
    assert tot.between(99.9, 100.1).all()
    assert set(p["investor_class"]) == set(ud.CLASSES)


@needs_cache
def test_vintages_hold_previous_month():
    v = ud.vintage_lags()
    assert len(v) >= 3
    assert v["prev_month_complete"].all()
    assert (v["lag_days"] <= 14).all()


def test_duration_types():
    w = pd.DataFrame({"security_type": ["FRN", "Note", "Bill"], "years_to_maturity": [2.0, 10.0, 0.25],
                      "coupon_or_rate": [0.1, 4.0, 4.0]})
    d = ud.duration(w)
    assert d[0] == 0.25 and 7.5 < d[1] < 8.5 and d[2] == pytest.approx(0.25)


def _weekly(dates, vals, lag, series="custody"):
    dates = pd.to_datetime(dates)
    return pd.DataFrame({"date": dates, "released": dates + pd.Timedelta(days=lag), "series": series, "value": vals})


def test_monthly_public_respects_release():
    # Wednesday 2026-09-30 is released 2026-10-01: September's public value is the 09-23 level.
    w = _weekly(["2026-09-16", "2026-09-23", "2026-09-30"], [1.0, 2.0, 3.0], 1)
    s = ud.monthly_public(w, "custody")
    assert s.loc["2026-09-01"] == 2.0 and s.loc["2026-10-01"] == 3.0
    # H.8: Wednesday + 9 days
    w = _weekly(["2026-09-16", "2026-09-23"], [1.0, 2.0], 9, "bank_ust")
    s = ud.monthly_public(w, "bank_ust")
    assert s.loc["2026-09-01"] == 1.0 and s.loc["2026-10-01"] == 2.0


def _wide(months, foreign):
    rows = []
    for m, f in zip(months, foreign):
        r = {"issue_date": pd.Timestamp(m) + pd.Timedelta(days=14), "security_type": "Note", "term_label": "10-Year Note",
             "term_years": 10.0, "years_to_maturity": 10.0, "coupon_or_rate": 4.0, "cusip": f"X{m}",
             "maturity": pd.Timestamp(m) + pd.DateOffset(years=10), "total_bn": 10.0, "soma_bn": 0.0,
             "private_bn": 10.0, "duration": 8.0, "auction_date": pd.NaT}
        r.update({c: 0.0 for c in ud.CLASSES})
        r["foreign"], r["inv_funds"] = f, 10.0 - f
        rows.append(r)
    return pd.DataFrame(rows)


def test_allotments_stamped_month_after_issue():
    months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    foreign = [1.0] * 12
    foreign[5] = 9.0  # June issues
    wide = _wide(months, foreign)
    empty = pd.DataFrame(columns=["date", "released", "series", "value"])
    m = ud.metrics(today=pd.Timestamp("2025-03-01"), weekly=empty, wide=wide)
    s = m[m["metric"] == "alloc_foreign"].set_index("month")["value"]
    # the June spike enters the 3m share first in the row stamped July
    assert s.loc["2024-06-01"] == pytest.approx(10.0)
    assert s.loc["2024-07-01"] > 30
    # the last issue month (Dec) is treated as possibly partial; stamps stop at Dec (= Nov issues)
    assert s.index.max() == pd.Timestamp("2024-12-01")


def test_metrics_no_lookahead_on_weekly():
    dates = pd.date_range("2020-01-01", "2026-09-30", freq="W-WED")
    w = pd.concat([_weekly(dates, np.arange(len(dates), dtype=float), 1),
                   _weekly(dates, np.arange(len(dates), dtype=float) * 2, 9, "bank_ust")], ignore_index=True)
    wide = pd.DataFrame()
    early = ud.metrics(today=pd.Timestamp("2025-06-15"), weekly=w, wide=wide)
    full = ud.metrics(today=pd.Timestamp("2026-10-02"), weekly=w, wide=wide)
    assert early["month"].max() < pd.Timestamp("2025-06-01")
    j = early.merge(full, on=["metric", "month"], suffixes=("_e", "_f"))
    assert len(j) == len(early)
    assert np.allclose(j["value_e"], j["value_f"])
    assert set(early["phase"]) == {"history"} and set(early["country"]) == {"US"}


def test_metric_signs():
    assert ud.METRICS["alloc_dealers"]["sign"] == 1 and ud.METRICS["alloc_dealers_d12"]["sign"] == 1
    assert all(ud.METRICS[k]["sign"] == -1 for k in ud.METRICS if not k.startswith("alloc_dealers"))
