"""UK cash framework tests. No network: synthetic frames plus the cached derived outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

MONTHLY = contract.DERIVED / "uk_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "uk_cash_summary.json"


def _season(n_years: int = 12, trend: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-04-01", periods=12 * n_years, freq="MS")
    prof = np.array([-30, 8, 20, 15, 22, 14, 6, 10, 14, 18, 12, 14], dtype=float)  # Jan..Dec
    vals = prof[idx.month - 1] + trend * np.arange(len(idx)) + rng.normal(0, 0.5, len(idx))
    return pd.Series(vals, index=idx)


def test_seasonal_add_recovers_profile():
    x = _season()
    origin = x.index[-13]
    fc = M.cgncr_forecast(x, origin, "seasonal_add3")
    actual = x.iloc[-12:].to_numpy()
    assert np.abs(fc - actual).mean() < 1.5
    assert np.abs(M.cgncr_forecast(x, origin, "trail_mean") - actual).mean() > 5


def test_smly_is_same_month_last_year():
    x = _season()
    origin = x.index[-13]
    fc = M.cgncr_forecast(x, origin, "smly")
    np.testing.assert_allclose(fc, x.iloc[-24:-12].to_numpy())


def test_forecast_uses_no_future_data():
    x = _season()
    origin = x.index[-13]
    poisoned = x.copy()
    poisoned.iloc[-12:] = 1e6
    for m in M.CGNCR_MODELS + M.CGNCR_BENCH:
        np.testing.assert_allclose(M.cgncr_forecast(x, origin, m), M.cgncr_forecast(poisoned, origin, m))


def test_covid_months_excluded_from_seasonal_estimation():
    idx = pd.date_range("2014-04-01", "2023-03-01", freq="MS")
    x = pd.Series(10.0, index=idx)
    x[(idx >= M.COVID_EST[0]) & (idx <= M.COVID_EST[1])] = 200.0
    fc = M.cgncr_forecast(x, pd.Timestamp("2023-03-01"), "seasonal_add5")
    assert np.allclose(fc, 10.0, atol=1e-9)


def test_choose_uses_pre_only():
    t = pd.DataFrame({
        "window": ["PRE", "PRE", "OOS", "OOS"],
        "h": [1, 1, 1, 1],
        "model": ["a", "b", "a", "b"],
        "mae": [1.0, 2.0, 9.0, 0.1],
        "n": [10] * 4,
    })
    assert M.choose(t, ("a", "b")) == "a"


def test_window_flags_covid_reach():
    bt = pd.DataFrame({
        "origin": pd.to_datetime(["2015-01-01", "2019-10-01", "2020-06-01", "2023-01-01"]),
        "target_end": pd.to_datetime(["2015-04-01", "2020-04-01", "2020-09-01", "2023-04-01"]),
    })
    assert list(M._window(bt)) == ["PRE", "COVID", "COVID", "OOS"]


def test_bill_beta_fit_recovers_beta():
    idx = pd.date_range("2010-01-01", periods=96, freq="MS")
    rng = np.random.default_rng(1)
    need = rng.normal(5, 10, len(idx))
    hh = pd.DataFrame({"cash_requirement_bn": need, "net_bonds_bn": 0.0, "of_nsandi_bn": 0.0}, index=idx)
    hh["net_bills_bn"] = 0.3 * need + np.where(idx.month == 3, 4.0, 0.0) + rng.normal(0, 0.1, len(idx))
    alpha, beta = M.bill_beta_fit(hh)
    assert abs(beta - 0.3) < 0.02
    assert abs(alpha[2] - 4.0) < 0.2


def test_bills_stock_parser_dates_fy_rows(tmp_path):
    rows = [["Data Date: x", None, None, None, None, None, None, None, None],
            ["Month End", "1m", "3m", "6m", "adhoc", "other", "total", "stock", "debtfin"],
            ["2025-26", None, None, None, None, None, None, None, None],
            ["Mar", 1000, 2000, 3000, 0, 0, 6000, 96000, 89500],
            ["Jan", 1000, 2000, 3000, 0, 0, 6000, 90000, None],
            ["Apr", 1000, 2000, 3000, 0, 0, 6000, 95000, None]]
    p = tmp_path / "b.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    b = src.bills_stock(p)
    assert list(b.index) == [pd.Timestamp("2025-04-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-01")]
    assert b.loc["2026-03-01", "bill_stock"] == pytest.approx(96.0)
    assert b.loc["2026-03-01", "bill_stock_debtfin"] == pytest.approx(89.5)


def test_ref_rpi_three_month_lag_interpolates():
    rpi = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
                     210.0, 220.0], index=pd.date_range("2020-01-01", periods=13, freq="MS"))
    d = pd.Series(pd.to_datetime(["2020-05-01", "2020-05-16"]))
    r = src.ref_rpi(d, pd.Series([3, 3]), rpi)
    assert r.iloc[0] == pytest.approx(110.0)          # May → Feb
    assert r.iloc[1] == pytest.approx(110.0 + 15 / 31 * 10.0)


def test_window_screens_covid_inputs_too():
    bt = pd.DataFrame({
        "origin": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01", "2021-06-01"]),
        "target_end": pd.to_datetime(["2022-04-01", "2022-03-01", "2022-04-01", "2021-07-01"]),
    })
    # trailing-12m inputs at a Jan/Feb-2022 origin still contain Feb/Mar-2021
    assert list(M._window(bt)) == ["COVID", "COVID", "OOS", "COVID"]


def test_vintage_plan_uses_only_published_comparable():
    v = pd.DataFrame({"fy_start": [2018, 2018, 2018, 2012], "vintage_date": pd.to_datetime(
        ["2018-03-13", "2018-10-29", "2019-04-24", "2012-03-21"]), "kind": ["plan", "revision", "outturn", "plan"],
        "cgncr_ex": [40.6, 31.2, 37.3, 121.0], "comparable": [True, True, True, False]})
    assert M.vintage_plan(v, pd.Timestamp("2018-09-30")) == {2018: 40.6}
    assert M.vintage_plan(v, pd.Timestamp("2019-12-31")) == {2018: 31.2}   # outturn and 2012 excluded
    assert M.info_date(pd.Timestamp("2018-09-01")) == pd.Timestamp("2018-10-21")


def test_end_march_closure_parts_sum_to_gap():
    plan = {"cgncr": 137.2, "gilt_sales": 246.2, "redemptions": 140.2, "nsandi": 15.2, "other": 0.3,
            "tbill_debtfin": 5.0, "net_cash_end": 2.3}
    same = M.end_march_closure(12.9, plan, {k: plan[k] for k in ("cgncr", "gilt_sales", "redemptions", "nsandi", "other")})
    assert same["gap"] == pytest.approx(same["parts"]["published_rounding"])          # 2.3 vs 2.2 arithmetic
    fy = {"cgncr": 135.7, "gilt_sales": 246.2, "redemptions": 140.2, "nsandi": 8.3, "other": 0.3}
    c = M.end_march_closure(12.9, plan, fy)
    assert c["gap"] == pytest.approx(sum(c["parts"].values()))
    assert c["parts"]["nsandi"] == pytest.approx(6.9) and c["parts"]["cgncr"] == pytest.approx(-1.5)
    assert c["tbill_debtfin_required"] == pytest.approx(5.0 + c["gap"])
    end = 12.9 + fy["gilt_sales"] + c["tbill_debtfin_required"] + fy["nsandi"] + fy["other"] - fy["cgncr"] - fy["redemptions"]
    assert end == pytest.approx(plan["net_cash_end"])


def test_solve_bills_hits_target_and_leaves_later_months():
    d = np.array([1.0, -2.0, 3.0, 4.0, 5.0])
    in_fy = np.array([True, True, True, False, False])
    out = M.solve_bills(d, in_fy, s0=90.0, target_stock=100.0)
    assert 90.0 + out[:3].sum() == pytest.approx(100.0)
    np.testing.assert_allclose(out[3:], d[3:])


def test_solve_bills_keeps_pinned_months():
    d = np.array([-2.0, 1.0, 3.0, 4.0])
    in_fy = np.array([True, True, True, False])
    fixed = np.array([True, False, False, False])
    out = M.solve_bills(d, in_fy, s0=90.0, target_stock=100.0, fixed=fixed)
    assert out[0] == -2.0 and out[3] == 4.0
    assert 90.0 + out[:3].sum() == pytest.approx(100.0)
    assert out[1] - d[1] == pytest.approx(out[2] - d[2])                        # even over free months


def test_bill_tender_net_only_complete_months():
    t = pd.DataFrame({"issue_dt": pd.to_datetime(["2026-08-03", "2026-08-31", "2026-09-07", "2026-09-28",
                                                  "2026-10-05"]),
                      "maturity_dt": pd.to_datetime(["2026-09-07", "2026-11-30", "2026-12-07", "2026-10-26",
                                                     "2027-01-04"]),
                      "size_bn": [1.0, 2.0, 3.0, 4.0, 5.0], "tenor": "3 Months", "yield_pct": 4.0})
    net = src.bill_tender_net(t.iloc[:4])                                       # last issue 28-Sep
    assert list(net.index) == [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")]
    assert net["2026-09-01"] == pytest.approx(3.0 + 4.0 - 1.0)
    assert pd.Timestamp("2026-10-01") not in src.bill_tender_net(t).index        # one October issue only


def test_model_beats_needs_edge_band():
    model = {6: 11.22, 12: 17.98}
    plan = {6: 12.55, 12: 13.03}
    assert M.model_beats(model, plan, 6)                                         # 10.6% edge
    assert not M.model_beats(model, plan, 7)                                     # 12.35 vs 12.63: inside 3%
    assert not M.model_beats(model, plan, 12)
    assert M.mae_at({1: 1.0, 3: 3.0}, 2) == pytest.approx(2.0)


def test_remit_other_items_sum_to_plan():
    row = pd.Series({"other_financing": 1.5, "sukuk_redemption": 0.5, "secondary_purchase_fin": 0.7})
    months = pd.date_range("2026-04-01", periods=12, freq="MS")
    m = M.remit_other_monthly(row, months, 2026)
    assert m.sum() == pytest.approx(0.3)
    assert m["2026-07-01"] == pytest.approx(0.8 / 12 - 0.5)                      # Sukuk matured 22-Jul-2026
    assert M.remit_other_monthly(row, pd.date_range("2027-04-01", periods=3, freq="MS"), 2026).sum() == 0.0


def test_apf_held_is_last_stock_before_date(monkeypatch):
    ops = pd.DataFrame({"op_dt": pd.to_datetime(["2020-01-10", "2023-10-19", "2024-07-15"]),
                        "isin": ["X", "X", "Y"], "apf_stock": [7000.0, 5742.8, 23428.0]})
    monkeypatch.setattr(src, "_APF", ops)
    assert src.apf_held("X", "2026-10-22") == pytest.approx(5742.8)
    assert src.apf_held("X", "2021-01-01") == pytest.approx(7000.0)
    assert src.apf_held("Z", "2026-10-22") == 0.0
    cb = src._cb_cash(pd.Series([5742.8, 9000.0]), pd.Series([31567.6, 8000.0]), pd.Series([1.0, 1.0]))
    np.testing.assert_allclose(cb, [5.7428, 8.0])                                  # capped at non-CRND stock


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.uk.run first")
def test_forward_closes_end_march_inside_identity():
    df = pd.read_parquet(MONTHLY).set_index("month")
    s = json.loads(SUMMARY.read_text())
    e = s["forward"]["end_march"]
    fy = int(e["fy"][:4])
    mar, mar_prev = pd.Timestamp(fy + 1, 3, 1), pd.Timestamp(fy, 3, 1)
    fc = df[df["is_forecast"]]
    assert fc.loc[mar, "dmo_net_cash_bn"] == pytest.approx(e["net_cash_plan_end_march_bn"], abs=2e-3)
    assert e["net_cash_model_end_march_bn"] == pytest.approx(e["net_cash_plan_end_march_bn"], abs=1e-3)
    cm = e["cash_mgmt_bills_end_march_assumed_bn"]
    assert fc.loc[mar, "buffer_level_implied_bn"] == pytest.approx(e["net_cash_plan_end_march_bn"] + cm, abs=2e-3)
    # the forward buffer rolls from the FY-to-date DMO-basis level with the identity's Δbuffer
    b0 = df.loc[df.index < fc.index[0], "buffer_level_dmo_basis_bn"].dropna().iloc[-1]
    np.testing.assert_allclose(fc["buffer_level_implied_bn"], b0 + fc["buffer_change_bn"].cumsum(), atol=1e-9)
    # FY bills: debt-financing need beyond plan is exactly the revision gap
    dbills = df.loc[mar, "bill_stock_implied_bn"] - df.loc[mar_prev, "bill_stock_bn"]
    debtfin_prev = df.loc[mar_prev, "bill_stock_debtfin_bn"]
    cm_prev = df.loc[mar_prev, "bill_stock_bn"] - debtfin_prev
    assert dbills - (cm - cm_prev) == pytest.approx(e["tbill_debtfin_required_bn"], abs=2e-3)
    assert e["tbill_debtfin_required_bn"] - e["tbill_debtfin_plan_bn"] == pytest.approx(e["next_revision_gap_bn"], abs=2e-3)
    assert sum(e["next_revision_gap_parts_bn"].values()) == pytest.approx(e["next_revision_gap_bn"], abs=5e-3)
    # forward other financing is NS&I + remit items only
    np.testing.assert_allclose(fc["other_fin_bn"], fc["of_nsandi_bn"] + fc["remit_other_fin_bn"], atol=1e-12)
    assert e["gilt_sales_fy_bn"] == pytest.approx(e["gilt_sales_plan_bn"], abs=1.0)
    # Redemptions follow the DMO ladder, not the remit plan row; the difference is the gap part.
    assert e["redemptions_fy_bn"] - e["redemptions_plan_bn"] == pytest.approx(
        e["next_revision_gap_parts_bn"]["redemptions"], abs=5e-3)


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.uk.run first")
def test_forward_redemptions_splices_already_matured_lines():
    """Cash history ends Sep; stock as-of is after 22 Oct — the 0⅜% 2026 must still appear."""
    if not (src.STOCK_UK.exists() and (src.UKF_OUT / "dmo_redeemed.parquet").exists()):
        pytest.skip("DMO stock / redeemed caches missing")
    red = src.forward_redemptions(pd.Timestamp("2026-09-30"))
    oct = red[red["maturity_dt"] == pd.Timestamp("2026-10-22")]
    assert len(oct) == 1
    assert oct.iloc[0]["cash_bn"] == pytest.approx(31.57, abs=0.05)
    assert oct.iloc[0]["cb_cash_bn"] == pytest.approx(5.74, abs=0.05)
    # stock still supplies the later FY maturities
    assert (red["maturity_dt"] >= pd.Timestamp("2027-01-01")).any()


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.uk.run first")
def test_apf_split_of_redemptions():
    df = pd.read_parquet(MONTHLY).set_index("month")
    s = json.loads(SUMMARY.read_text())["forward"]
    assert (df["bond_redemption_cb_bn"] <= df["bond_redemption_bn"] + 1e-9).all()
    np.testing.assert_allclose(df["net_bonds_private_bn"], df["net_bonds_bn"] + df["bond_redemption_cb_bn"], atol=1e-12)
    cal = {r["date"]: r for r in s["redemption_calendar"]}
    # Oct-26 matured before the stock as-of; must still appear via the redeemed splice.
    assert "2026-10-22" in cal
    assert cal["2026-10-22"]["apf_held_bn"] == pytest.approx(5.7, abs=0.05)
    assert cal["2026-10-22"]["cash_bn"] == pytest.approx(31.6, abs=0.1)
    if "2027-07-22" in cal:
        assert cal["2027-07-22"]["apf_held_bn"] == pytest.approx(23.4, abs=0.05)
        assert cal["2027-07-22"]["nominal_bn"] == pytest.approx(41.9, abs=0.05)
    fc = df[df["is_forecast"]]
    assert s["cb_held_redemptions_12m_bn"] == pytest.approx(fc["bond_redemption_cb_bn"].sum(), abs=1e-3)
    assert s["private_net_bonds_12m_bn"] == pytest.approx(
        s["gilt_gross_12m_bn"] - (s["gilt_redemptions_12m_bn"] - s["cb_held_redemptions_12m_bn"]), abs=1e-2)


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.uk.run first")
def test_outputs_satisfy_contract():
    from sovereign_cash.cash_season import HORIZON_END, forward_months

    df = pd.read_parquet(MONTHLY)
    resid = contract.validate(df)
    assert resid < 1e-6
    hist = df[~df["is_forecast"]]
    fc = df[df["is_forecast"]]
    origin = pd.Timestamp(hist["month"].max())
    assert len(fc) == forward_months(origin)
    assert len(fc) >= M.FC_MONTHS
    assert fc["month"].min() > hist["month"].max()
    assert pd.Timestamp(fc["month"].max()) == HORIZON_END
    assert fc["bill_stock_implied_bn"].notna().all()
    named = [c for c in df.columns if c.startswith("of_") and c != "of_unexplained_bn"]
    h = hist.dropna(subset=named)
    np.testing.assert_allclose(h[named + ["of_unexplained_bn"]].sum(axis=1), h["other_fin_bn"], atol=1e-9)


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.uk.run first")
def test_history_matches_dmo_financing_arithmetic():
    df = pd.read_parquet(MONTHLY).set_index("month")
    fy25 = df.loc["2025-04-01":"2026-03-01"]
    assert fy25["cash_requirement_bn"].sum() == pytest.approx(135.9, abs=0.15)   # DMO outturn
    assert fy25["bond_gross_bn"].sum() == pytest.approx(303.9, abs=1.0)          # gilt sales
    assert fy25["bond_redemption_bn"].sum() == pytest.approx(168.2, abs=1.0)
    # FY26-27 redemptions = DMO maturity ladder (cash), incl. mid-month lines that have already
    # dropped off the stock snapshot (Oct-26 0⅜% 2026) — matches the remit plan row ≈ 140.2.
    fy26 = df.loc["2026-04-01":"2027-03-01", "bond_redemption_bn"].sum()
    e = json.loads(SUMMARY.read_text())["forward"]["end_march"]
    assert fy26 == pytest.approx(e["redemptions_fy_bn"], abs=0.05)
    assert fy26 == pytest.approx(140.2, abs=1.0)


PAGE = contract.ROOT / "uk_cash_dashboard.html"


@pytest.mark.skipif(not PAGE.exists(), reason="run sovereign_cash.uk.run first")
def test_page_loads_plotly_first_and_has_no_raw_ids():
    import html as _html
    import re

    s = PAGE.read_text(encoding="utf-8")
    assert s.find("cdn.plot.ly") < s.find("Plotly.newPlot")
    assert s.count("cdn.plot.ly") == 1
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", text))
    assert not re.findall(r"\b\w+_bn\b|seasonal_add\d|bill_beta|profile_mult\d|trail_mean|\bsmly\b", text)
    assert "£-" not in text and "£+" not in text
    assert not re.search(r"h=\[|\['", text)


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.uk.run first")
def test_forward_flags_and_pins():
    df = pd.read_parquet(MONTHLY).set_index("month")
    fc = df[df["is_forecast"]]
    e = json.loads(SUMMARY.read_text())["forward"]["end_march"]
    fy_end = pd.Timestamp(int(e["fy"][:4]) + 1, 3, 1)
    assert (fc["gilts_assumed"] == (fc.index > fy_end)).all()
    assert (fc["bills_assumed"] == (fc.index > fy_end)).all()
    if e["bill_after_fy"] == "flat":
        assert (fc.loc[fc.index > fy_end, "net_bills_bn"] == 0).all()
    tn = src.bills_stock()["bill_stock"].diff().dropna().combine_first(src.bill_tender_net())
    for m in e["bill_pinned_months"]:
        m = pd.Timestamp(m)
        if m in tn.index:
            assert fc.at[m, "net_bills_bn"] == pytest.approx(tn[m])
    if e["cgncr_source_in_fy"] == "plan":
        assert e["cgncr_ex_fc_fy_bn"] == pytest.approx(e["cgncr_ex_plan_bn"], abs=1e-3)


@pytest.mark.skipif(not SUMMARY.exists(), reason="run sovereign_cash.uk.run first")
def test_summary_fields():
    s = json.loads(SUMMARY.read_text())
    for k in ("country", "ccy", "asof", "sources", "buffer", "identity_resid_bn", "score", "forward", "notes"):
        assert k in s
    assert s["country"] == "UK" and s["ccy"] == "GBP"
    assert all({"name", "url", "frequency", "lag_days", "verified"} <= set(x) for x in s["sources"])
    assert s["score"], s.get("score_note")
    pre = {r["model_name"] for r in s["score"] if r["window"] == "PRE"}
    assert pre and pre <= set(M.CGNCR_MODELS + M.BILL_MODELS + M.BILL_BENCH)
    assert s["forward"]["end_march"]["net_cash_plan_end_march_bn"] == pytest.approx(2.3)


def test_plan_profile_scale_channel_flattens_residual():
    """Remit ≫ prior FY: residual uniform — do not freeze hist month shares (DE-class)."""
    # Two complete FYs with a strong Jan spike; origin Sep of next FY.
    idx = pd.date_range("2023-04-01", "2026-09-01", freq="MS")
    prof = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 40], dtype=float)  # Apr..Mar; Mar heavy
    vals = []
    for m in idx:
        fy = M.fy_of(m)
        # map calendar month to FY position
        pos = (m.month - 4) % 12
        vals.append(float(prof[pos]))
    x = pd.Series(vals, index=idx)
    origin = pd.Timestamp("2026-09-01")
    prior = float(x.loc["2025-04-01":"2026-03-01"].sum())
    # Stable remit: hist shape preserved (Mar-heavy rest after Sep → Mar dominates).
    plan_stable = {2026: prior * 1.05}
    fc_s = M.plan_profile(x, origin, plan_stable, h=6)
    months = pd.date_range("2026-10-01", periods=6, freq="MS")
    assert float(fc_s[months.get_loc(pd.Timestamp("2027-03-01"))]) > float(
        fc_s[months.get_loc(pd.Timestamp("2026-10-01"))])
    # Scale-channel remit: flat residual across remaining months.
    plan_ramp = {2026: prior * 1.5}
    fc_r = M.plan_profile(x, origin, plan_ramp, h=6)
    assert abs(float(fc_r[0]) - float(fc_r[-1])) < 1e-9
