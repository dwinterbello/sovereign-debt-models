"""The demand tape keeps country-specific sectors. UK ICPF and OFI are not one IMF residual."""
import numpy as np
import pandas as pd

from sovereign_cash.debt.buyers import (
    _beta, _corr, _fwl, _lead, _same_sign, absorption, flows, granular, impact, marginal, response, tracker,
)


def test_corr_recovers_a_buyer_of_the_selloff():
    dy = np.linspace(-20, 40, 16)
    flow = 0.02 * dy
    corr, slope, n = _corr(flow, dy)
    assert n == 16
    assert corr > 0.99
    assert abs(slope - 2.0) < 1e-6


def test_uk_tape_splits_icpf_from_ofi():
    fl = flows()
    uk = set(fl.loc[fl["cc"] == "uk", "sector"])
    assert {"icpf", "ofi", "overseas"} <= uk
    assert set(fl.loc[fl["cc"] == "uk", "tape"]) == {"ONS transactions"}
    jp = set(fl.loc[fl["cc"] == "jp", "tape"])
    assert {"JSDA transactions", "IMF nominal change"} <= jp
    js = fl[(fl["cc"] == "jp") & (fl["tape"] == "JSDA transactions")]
    assert {"dy2", "dy10", "dy30"} <= set(js["match_tenor"])
    assert js.groupby(["quarter_end", "sector"]).size().max() >= 3
    assert "OFS-2 par change" in set(fl.loc[fl["cc"] == "us", "tape"])
    us = fl[(fl["cc"] == "us") & (fl["tape"] == "Treasury auction allotments")]
    assert {"dy2", "dy10", "dy30"} <= set(us["match_tenor"])
    assert "JPX JGB futures" in set(fl.loc[fl["cc"] == "jp", "tape"])
    assert (js["sector"] == "boj").any()
    tic = fl[(fl["cc"] == "us") & (fl["tape"] == "TIC long-term")]
    assert set(tic["sector"]) == {"official", "private"}
    assert tic.groupby("quarter_end")["sector"].nunique().min() == 2


def test_granular_shock_tracks_through_the_latest_quarter():
    g = granular()
    uk = g[(g["cc"] == "uk") & (g["tape"] == "ONS transactions")]
    assert uk["quarter_end"].max() >= pd.Timestamp("2025-12-31")
    assert set(uk["tenor"]) >= {"dy2", "dy10", "dy30", "ds1030"}
    one = uk[uk["tenor"] == "dy10"]
    assert one["z_pct"].notna().sum() > 8
    assert one["supply_pct"].notna().sum() > 8
    us = g[g["tape"].eq("Treasury auction allotments")]
    assert {"dy2", "dy10", "dy30", "ds1030"} <= set(us["tenor"])


def test_partial_slope_holds_the_rest_of_the_tape_fixed():
    rng = np.random.default_rng(0)
    z = rng.normal(size=40)
    x = z + rng.normal(size=40)
    y = 2 * z + rng.normal(scale=0.05, size=40)
    beta, t, _p, df, _rx, _ry = _fwl(y, x, z)
    assert abs(beta) < 0.5 and abs(t) < 2
    y2 = 1.5 * x + rng.normal(scale=0.05, size=40)
    beta2, t2, p2, _df2, _a, _b = _fwl(y2, x, z)
    assert beta2 > 1 and t2 > 2 and p2 > 0.8 and df > 10


def test_half_window_must_share_the_sign():
    assert _same_sign(0.4, 0.2, 0.5)
    assert not _same_sign(0.4, -0.2, 0.5)
    assert not _same_sign(0.4, 0.2, np.nan)


def test_response_has_both_eras_and_a_marginal_buyer():
    fl = flows()
    r = response(fl)
    uk = r[(r["cc"] == "uk") & (r["sector"] == "icpf")]
    assert set(uk["era"]) == {"2016–2021", "2022–"}
    assert uk["n"].min() >= 12
    m = marginal(fl)
    assert set(m["cc"]) >= {"uk", "us", "de", "jp"}
    assert m["share_of_gross"].between(0, 1).all()


def test_absorption_of_a_partition_sums_to_one():
    rng = np.random.default_rng(0)
    supply = rng.normal(10, 3, 20)
    shares = np.array([0.2, 0.5, 0.3])
    slopes = [_beta(share * supply, supply)[0] for share in shares]
    assert abs(sum(slopes) - 1) < 1e-9


def test_uk_absorption_and_tenor_impact_are_estimated():
    fl = flows()
    a = absorption(fl)
    uk = a[(a["cc"] == "uk") & (a["sector"] == "icpf") & (a["era"] == "2022–")]
    assert len(uk) == 1 and uk["n"].iloc[0] >= 12 and np.isfinite(uk["marginal"].iloc[0])
    moved = impact(fl)
    tenors = set(moved.loc[(moved["cc"] == "uk") & (moved["sector"] == "icpf"), "tenor"])
    assert {"dy2", "dy10", "ds1030"} <= tenors
    assert "Treasury auction allotments" not in set(moved["tape"])
    y10 = moved[(moved["cc"] == "uk") & (moved["sector"] == "icpf")
                & (moved["era"] == "2022–") & (moved["tenor"] == "dy10")]
    assert abs(float(y10["per_100bp"].iloc[0])) < 5
    assert set(moved.loc[(moved["cc"] == "uk") & (moved["sector"] == "icpf") & (moved["tenor"] == "dy10"), "era"]) == {
        "2016–2021", "2022–"}


def test_lead_ignores_a_pack_inside_the_noise():
    assert _lead(np.array([0.40, 0.30, 0.20]), 12)[3] is False
    i, r, _, identified = _lead(np.array([0.95, 0.05, 0.00, -0.10]), 40)
    assert i == 0 and r > 0.9 and identified is True


def test_tracker_reaches_the_latest_quarter():
    tr = tracker(flows())
    uk = tr[(tr["cc"] == "uk") & (tr["sector"] == "icpf")].dropna(subset=["level"])
    assert uk["quarter_end"].max() >= pd.Timestamp("2025-01-01")
    assert uk["unit"].iloc[0] == "% of GDP"
    last = uk["quarter_end"].max()
    row = uk[uk["quarter_end"] == last].iloc[0]
    assert np.isfinite(row["level"]) and np.isfinite(row["dy10_4q"])
    us = tr[tr["tape"].eq("Treasury auction allotments")]
    assert set(us["unit"]) == {"% of coupon auctions"}
    live = tr[tr["quarter_end"] == tr.groupby(["cc", "tape"])["quarter_end"].transform("max")]
    bought = live[live["flow_4q"] > 0]
    assert bought["share_of_gross"].between(0, 1).all()
