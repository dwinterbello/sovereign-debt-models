import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance import evidence, headline, market


def test_excess_return_matches_the_closed_form():
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    flat = pd.Series(4.0, index=idx)
    r = pd.Series(1.0, index=idx)
    for h in (3, 12):
        rx = market.excess_return(flat, r, h, flat)
        assert np.allclose(rx, (4.0 - 1.0) * h / 12)
    # 10y jumps 4% -> 5% at t+h with the 2y at 4%: roll-down from the 2y-10y line, duration at 4%
    y10 = pd.Series(4.0, index=idx)
    y10.iloc[15:] = 5.0
    y2 = pd.Series(4.0, index=idx)
    h, t = 3, idx[12]
    rx = market.excess_return(y10, r, h, y2)
    d = (1 - 1.02 ** -20) / 0.04
    y_end = 5.0 - (h / 12) / 8 * (5.0 - 4.0)
    assert abs(rx[t] - (4.0 * h / 12 - d * (y_end - 4.0) - 1.0 * h / 12)) < 1e-12
    assert rx[t] < 0  # yields rose: the bond lost to cash
    # no 2y: no roll-down
    rx0 = market.excess_return(y10, r, h)
    assert abs(rx0[t] - (4.0 * h / 12 - d * 1.0 - h / 12)) < 1e-12
    # the last h months have no realised return yet
    assert rx.index.max() == idx[-1 - h]


def test_mod_duration_limits():
    assert market.mod_duration(0.0) == 10.0
    assert abs(market.mod_duration(4.0) - (1 - 1.02 ** -20) / 0.04) < 1e-12
    neg = market.mod_duration(-0.3)
    assert np.isfinite(neg) and neg > 10


def test_realised_vol_and_state_percentiles_never_look_ahead():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2010-01-01", "2016-12-31")
    s = pd.Series(np.cumsum(rng.normal(scale=0.05, size=len(days))), index=days)
    v = market.realised_vol(s)
    cut = pd.Timestamp("2014-06-01")
    s2 = s.copy()
    s2[s2.index >= cut + pd.offsets.MonthBegin(1)] += rng.normal(scale=1.0, size=(s2.index >= cut + pd.offsets.MonthBegin(1)).sum())
    v2 = market.realised_vol(s2)
    pd.testing.assert_series_equal(v[v.index <= cut], v2[v2.index <= cut])
    assert not np.allclose(v[v.index > cut].to_numpy(), v2[v2.index > cut].reindex(v[v.index > cut].index).to_numpy())
    # a flat 2bp/day random walk annualises to about 2·√252 bp
    w = pd.Series(np.cumsum(rng.choice([-0.02, 0.02], size=len(days))), index=days)
    assert abs(market.realised_vol(w).median() - 2 * np.sqrt(252)) < 1.0

    def mk(vals):
        months = pd.date_range("2010-01-01", periods=len(vals), freq="MS")
        return pd.DataFrame({"country": "AA", "month": months, "series": "rvol10", "value": vals})

    a = rng.normal(size=100)
    b = a.copy()
    b[70:] = 100.0
    empty = pd.DataFrame(columns=["country", "month", "cb_hold_share", "phase"])
    sa = evidence.states(mk(a), empty).set_index("month")["s"]
    sb = evidence.states(mk(b), empty).set_index("month")["s"]
    pd.testing.assert_series_equal(sa.iloc[:70 - 24], sb.iloc[:70 - 24])
    assert sa.between(-0.5, 0.5).all()


def test_dk_ols_matches_statsmodels_hac_groupsum():
    rng = np.random.default_rng(3)
    n_t, n_c = 80, 4
    month = np.repeat(np.arange(n_t), n_c)
    X = np.c_[np.ones(n_t * n_c), rng.normal(size=(n_t * n_c, 2))]
    y = X @ [0.1, 0.5, -0.2] + rng.normal(size=n_t * n_c)
    b, t = evidence.dk_ols(y, X, month, 4)
    ref = sm.OLS(y, X).fit(cov_type="hac-groupsum", cov_kwds={"time": month, "maxlags": 4})
    assert np.allclose(b, ref.params) and np.allclose(t, ref.tvalues)


def _panel(n=180, seed=0, beta_x=0.0, beta_xs=0.0):
    """Two countries with a metric, a vol state and a 10y whose next-3m change loads on x and x·s."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2008-01-01", periods=n, freq="MS")
    p, m = [], []
    for c in ("AA", "BB"):
        x = np.cumsum(rng.normal(size=n))
        lv = np.zeros(n)
        for t in range(1, n):
            lv[t] = 0.85 * lv[t - 1] + rng.normal(scale=0.2)
        rv = np.exp(lv)
        z = evidence.pressure_pct(pd.Series(x)).to_numpy()
        s = (evidence.pressure_pct(pd.Series(rv)) / 10).to_numpy()
        y = np.zeros(n)
        for t in range(n):
            y[t] = (y[t - 1] if t else 3.0) + rng.normal(scale=0.05)
            src = t - 4
            if src >= 0 and np.isfinite(z[src]) and np.isfinite(s[src]):
                y[t] += (beta_x * z[src] + beta_xs * z[src] * s[src]) / 3
        p.append(pd.DataFrame({"country": c, "month": months, "metric": "priv12", "value": x, "phase": "history"}))
        m.append(pd.DataFrame({"country": c, "month": months, "series": "y10", "value": y, "source": "t", "last_obs": months[-1]}))
        m.append(pd.DataFrame({"country": c, "month": months, "series": "rvol10", "value": rv, "source": "t", "last_obs": months[-1]}))
    return pd.concat(p, ignore_index=True), pd.concat(m, ignore_index=True)


def _inter(p, m):
    empty = pd.DataFrame(columns=["country", "month", "cb_hold_share", "phase"])
    st = evidence.states(m, empty)
    out = evidence.interactions(p, m, st, metrics=["priv12"], targets=["y10"], state_names=["rvol"])
    return out[out["horizon"] == 3].iloc[0]


def test_interaction_is_null_on_random_data_and_recovers_a_planted_one():
    for seed in (1, 2):
        row = _inter(*_panel(seed=seed))
        assert abs(row["t2"]) < 2.5 and row["verdict"] != "forward link"
    row = _inter(*_panel(seed=0, beta_xs=0.6))
    assert row["beta2_bp"] > 0 and row["t2"] > 3
    assert row["beta_hi_bp"] > row["beta_lo_bp"]


def test_return_targets_use_the_start_month_value_and_the_realised_12m_control():
    months = pd.date_range("2010-01-01", periods=60, freq="MS")
    rx3 = pd.Series(np.arange(60, dtype=float), index=months)
    rx12 = pd.Series(np.arange(60, dtype=float) * 10, index=months)
    m = pd.concat([pd.DataFrame({"country": "AA", "month": months, "series": s, "value": v.to_numpy()})
                   for s, v in (("rx10_3", rx3), ("rx10_12", rx12))])
    fw = evidence.forward_changes(m, lags=(1,))
    a = fw[(fw["target"] == "rx10_3")].set_index("month")
    assert set(a["h"]) == {3}
    t = months[20]
    assert a.loc[t, "y"] == rx3[months[21]] and a.loc[t, "y12"] == rx12[months[8]]
    assert set(fw.loc[fw["target"] == "rx10_12", "h"]) == {12}


def test_gdp_is_used_only_once_published_and_grows_past_the_last_quarter():
    q = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=pd.date_range("2024-01-01", periods=5, freq="QS"))
    months = pd.date_range("2024-04-01", "2025-12-01", freq="MS")
    g = headline.gdp_monthly(months, q)
    assert np.isnan(g[pd.Timestamp("2024-04-01")])  # Q1 not out until May
    assert g[pd.Timestamp("2024-05-01")] == 100.0
    assert g[pd.Timestamp("2025-04-01")] == 103.0  # Q4-2024 from April; Q1-2025 (104) only from May
    assert g[pd.Timestamp("2025-05-01")] == 104.0
    assert abs(g[pd.Timestamp("2025-08-01")] - 104.0 * 1.04 ** 0.25) < 1e-9  # Q2-2025 due, not published


def test_new_metric_and_targets_are_registered():
    assert headline.METRICS["dur_stock_gdp"]["sign"] == 1
    assert "dur_stock_gdp" not in headline.HEADLINE
    assert {"rx10_3", "rx10_12", "tp10"} <= set(evidence.TARGETS) <= set(evidence.TARGET_SIGN)
    assert {"rx10_3", "rx10_12", "tp10", "rvol10"} <= set(market.MARKET_META)
    c = evidence.chance(pd.DataFrame({"verdict": ["no link"] * 100, "window": "2010+"}))
    assert c["n_tests"] == 100 and 2 < c["expected_by_chance"] < 2.5
