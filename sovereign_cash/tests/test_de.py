"""Germany cash framework — offline tests on cached inputs and written outputs (no network)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.contract import DERIVED, REQUIRED, validate
from sovereign_cash.core import cash_framework as C
from sovereign_cash.de import fetch as F

MB = sorted(F.MB_DIR.glob("20*.html"))


# ---------------------------------------------------------------- shared model pieces

def _toy(years: int = 8, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2010-01-01", periods=12 * years, freq="MS")
    prof = np.array([5, -3, 2, 8, -6, 1, 4, -2, -7, 3, 6, 10], float)
    rng = np.random.default_rng(seed)
    return pd.Series(np.tile(prof, years) + rng.normal(0, 0.5, len(idx)) + 1.0, index=idx)


def test_monthly_from_ytd_roundtrip():
    y = _toy()
    ytd = y.groupby(y.index.year).cumsum()
    assert np.allclose(C.monthly_from_ytd(ytd), y)


def test_monthly_from_ytd_gap_is_nan():
    y = _toy()
    ytd = y.groupby(y.index.year).cumsum().drop(pd.Timestamp("2012-05-01"))
    m = C.monthly_from_ytd(ytd)
    assert np.isnan(m[pd.Timestamp("2012-05-01")]) and np.isnan(m[pd.Timestamp("2012-06-01")])


def test_smly_and_seasonal_models():
    y = _toy()
    t = pd.Timestamp("2015-06-01")
    s = C.forecast(y, t, 12, "smly")
    assert np.allclose(s.values, y.loc["2014-07-01":"2015-06-01"].values)
    a = C.forecast(y, t, 12, "seas_add_3")
    assert a.sum() == pytest.approx(y.loc[:t].iloc[-12:].sum())  # deviations sum to zero
    m = C.forecast(y, t, 12, "seas_mult_3")
    assert m.sum() == pytest.approx(y.loc[:t].iloc[-12:].sum())


def test_target_model_hits_target():
    y = _toy()
    t = pd.Timestamp("2015-04-01")
    fc = C.forecast(y, t, 8, "target_3", targets={2015: 100.0})
    ytd = y.loc["2015-01-01":t].sum()
    assert ytd + fc.sum() == pytest.approx(100.0)
    pv = C.forecast(y, t, 8, "prevyear_3")
    assert ytd + pv.sum() == pytest.approx(y.loc["2014"].sum())


def test_backtest_uses_no_future_data():
    y = _toy()
    bt = C.backtest(y, ["seas_add_3"])
    y2 = y.copy()
    y2.loc["2016-01-01":] += 50.0
    bt2 = C.backtest(y2, ["seas_add_3"])
    early = bt[bt["origin"] < pd.Timestamp("2015-12-01")].reset_index(drop=True)
    early2 = bt2[bt2["origin"] < pd.Timestamp("2015-12-01")].reset_index(drop=True)
    assert np.allclose(early["fc"], early2["fc"])


def test_choose_refuses_short_pre():
    sc = pd.DataFrame([{"window": "PRE", "h": 12, "model": "seas_add_3", "mae": 1.0, "bias": 0, "n": 3,
                        "common": True}])
    model, why = C.choose(sc)
    assert model == C.DEFAULT_MODEL and "too short" in why


def test_roll_bills_conserves_net():
    idx = pd.date_range("2027-01-01", periods=12, freq="MS")
    net = pd.Series(np.linspace(-5, 5, 12), index=idx)
    ladder = pd.Series(10.0, index=idx)
    gross, red, feas = C.roll_bills(net, ladder, {3: 0.5, 6: 0.5})
    assert np.allclose(gross - red, net) and np.allclose(feas, net)
    assert red.iloc[3] > ladder.iloc[3]  # January's new bills mature in April
    g2, r2, f2 = C.roll_bills(net - 30.0, ladder, {3: 1.0}, floor=4.0)
    assert (g2 >= 4.0 - 1e-12).all() and np.allclose(f2, g2 - r2) and (f2 >= net - 30.0 - 1e-12).all()


def test_roll_bills_fixed_and_on_tape():
    idx = pd.date_range("2026-08-01", periods=8, freq="MS")
    ladder = pd.Series(5.0, index=idx)
    fixed = pd.Series([20.0, 15.0] + [np.nan] * 6, index=idx)
    on_tape = pd.Series([20.0, 0.0] + [0.0] * 6, index=idx)
    g, r, n = C.roll_bills(pd.Series(0.0, index=idx), ladder, {1: 1.0}, on_tape=on_tape, fixed=fixed)
    assert g.iloc[0] == 20.0 and g.iloc[1] == 15.0
    assert r.iloc[1] == 5.0           # Aug gross is already on the tape ladder: no double count
    assert r.iloc[2] == 20.0          # Sep gross (not on tape) rolls into Oct
    assert np.allclose(n.iloc[2:], 0.0)  # flat stock after the fixed months


# ---------------------------------------------------------------- parsers on cached files

@pytest.mark.skipif(not MB, reason="no cached Monatsbericht pages")
def test_parse_mb_cached_pages():
    bad = []
    for p in MB:
        r = F.parse_mb(p.read_text(encoding="utf-8"))
        if r is None or r.get("ytd_fin_balance") is None:
            bad.append(p.name)
    assert not bad, bad


@pytest.mark.skipif(not MB, reason="no cached Monatsbericht pages")
def test_mb_ytd_identity():
    mb = F.load_bmf_mb()
    d = (mb["ytd_revenue"] - mb["ytd_expenditure"] - mb["ytd_fin_balance"]).abs()
    # Finanzierungssaldo = Einnahmen − Ausgaben (rounding to 0.1bn, plus coin/Rücklagen lines in some years)
    assert (d < 1.0).mean() > 0.9


@pytest.mark.skipif(not (F.DATA / "bmf_mb_2016-12.pdf").exists(), reason="archive pdf not cached")
def test_sdds_pdf():
    df = F.parse_sdds_pdf(F.DATA / "bmf_mb_2016-12.pdf")
    assert df.index.min() == pd.Timestamp("2011-01-01")
    assert df.index.max() == pd.Timestamp("2016-11-01")
    assert df.at[pd.Timestamp("2013-12-01"), "ytd_fin_balance"] == pytest.approx(-22.348)


@pytest.mark.skipif(not (F.DATA / "schuldenbericht_dt.xlsx").exists(), reason="Finanzagentur file not cached")
def test_finanzagentur_debt_identity():
    from sovereign_cash.de import model as M

    h, _ = M.build_history()
    assert h["fa_debt_identity_bn"].abs().max() < 0.05
    step = h.index.to_series().dt.to_period("M").diff().apply(lambda x: getattr(x, "n", np.nan)) == 1
    d = (h["buffer_change_bn"] - h["buffer_level_bn"].diff())[step]
    assert d.abs().max() < 1e-9


# ---------------------------------------------------------------- written outputs

OUT = DERIVED / "de_cash_monthly.parquet"
SUM = DERIVED / "de_cash_summary.json"


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/de/run.py first")
def test_outputs_contract():
    df = pd.read_parquet(OUT)
    assert set(REQUIRED) <= set(df.columns)
    assert validate(df) < 1e-6
    fc = df[df["is_forecast"]]
    assert len(fc) >= 12
    assert fc["bill_stock_implied_bn"].notna().all()
    hist = df[~df["is_forecast"]]
    assert hist["other_fin_gap_bn"].notna().all()
    s = json.loads(SUM.read_text())
    for k in ("country", "ccy", "asof", "sources", "buffer", "identity_resid_bn", "score", "forward", "notes"):
        assert k in s
    assert s["country"] == "DE" and s["ccy"] == "EUR"
    assert all({"name", "url", "frequency", "lag_days", "verified"} <= set(x) for x in s["sources"])
    assert s["score"] or s.get("score_note")
    for r in s["score"]:
        assert {"window", "h", "model", "benchmark", "n"} <= set(r)
    assert (DERIVED.parent / "de_cash_dashboard.html").exists()
    f = s["forward"]
    assert "soll_cash_requirement_year_bn" not in f  # investor board must read the plan keys (core + funds)
    for k in ("plan_year", "plan_cash_requirement_bn", "plan_source", "cash_requirement_ytd_bn", "ytd_through",
              "model_full_year_bn", "plan_gap_lands_on", "cb_held_redemptions_12m_bn", "private_net_bonds_12m_bn"):
        assert k in f, k
    assert f["plan_cash_requirement_bn"] == pytest.approx(181.6)
    tot = df["cash_requirement_core_bn"] + df["cash_requirement_sv_bn"]
    assert np.allclose(tot, df["cash_requirement_bn"])
    assert np.allclose(hist["net_bills_bn"], hist["net_bubills_bn"] + hist["net_mm_budget_bn"])


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/de/run.py first")
def test_observed_forward_months_use_actuals():
    from sovereign_cash.de import model as M

    df = pd.read_parquet(OUT).set_index("month")
    fc = df[df["is_forecast"]]
    hist = df[~df["is_forecast"]]
    core = M.core_budget()
    cash = -C.monthly_from_ytd(core["ytd_fin_balance"])
    obs = fc[fc["cash_observed"]]
    if obs.empty:
        pytest.skip("BMF not ahead of Finanzagentur this vintage")
    for m in obs.index:
        assert obs.at[m, "cash_requirement_core_bn"] == pytest.approx(cash[m])
        assert "core observed" in obs.at[m, "fc_source"]
    tape = fc[fc["fc_source"].str.contains("Bubills tape")]
    assert len(tape) >= 1
    # implied stocks start from the last actual
    first = fc.index[0]
    assert fc.at[first, "bill_stock_implied_bn"] == pytest.approx(hist["bill_stock_bn"].iloc[-1] + fc.at[first, "net_bubills_bn"])
    assert fc.at[first, "buffer_level_implied_bn"] == pytest.approx(hist["buffer_level_bn"].iloc[-1] + fc.at[first, "buffer_change_bn"])
    # After the published calendar: hist need path, demeaned so end stock ≈ calendar-end stock
    cal_end = pd.Timestamp(json.loads((M.BUBILL_CAL).read_text())["quarter_end"]).to_period("M").to_timestamp()
    after = fc[fc.index > cal_end]
    assert len(after) >= 1
    assert after["net_bubills_bn"].abs().sum() > 0.0  # not a flat zero path
    assert float(after["net_bubills_bn"].sum()) == pytest.approx(0.0, abs=0.05)
    assert after["fc_source"].str.contains("hist need").all()
    assert (fc["bond_redemption_cb_bn"] >= 0).all() and fc["bond_redemption_cb_bn"].sum() > 0


def test_haircut_band_and_gap_exclusion():
    from sovereign_cash.de import model as M

    idx = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    cash = pd.Series(5.0, index=idx)                      # 60bn a year
    core = pd.DataFrame({"soll_fin_balance": -100.0, "soll_year": idx.year}, index=idx)
    gap = pd.Series(0.0, index=idx)
    gap.loc["2024"] = -3.5                                # −42bn in 2024
    hc = M.haircuts(cash, core, gap)
    assert not hc.at[2024, "eligible"] and hc.at[2024, "in_band"]
    assert not hc.at[2021, "in_band"]                     # Covid
    c, yrs = M.haircut_at(hc, 2026)
    assert 2024 not in yrs and c == pytest.approx(0.6)
    lo, hi = M.haircut_band(hc, 2026)
    assert lo <= c <= hi


def test_open_year_closes_spending_and_revenue_separately():
    from sovereign_cash.de import model as M

    idx = pd.date_range("2018-01-01", "2024-06-01", freq="MS")
    exp = pd.Series(10.0, index=idx)
    rev = pd.Series(8.0, index=idx)
    origin = pd.Timestamp("2024-06-01")
    paths = M.open_year_core(exp, rev, origin, 12, 100.0, 80.0, (1.2, 1.0, 1.4), (1.2, 1.0, 1.4))
    rest = paths["central"].loc[:pd.Timestamp("2024-12-01")]
    assert rest.sum() == pytest.approx((100.0 * 1.2 - 60.0) - (80.0 * 1.2 - 48.0))
    assert (paths["lo"].loc[:pd.Timestamp("2024-12-01")] <= rest + 1e-9).all()
    assert (paths["hi"].loc[:pd.Timestamp("2024-12-01")] >= rest - 1e-9).all()
    monthly = pd.Series(10.0, index=pd.date_range("2016-01-01", "2025-12-01", freq="MS"))
    plans = {y: 100.0 for y in range(2016, 2026)}
    plans[2024] = 200.0
    mean, lo_r, hi_r, years = M.line_ratio(monthly, plans, 2026)
    assert 2020 not in years and 2021 not in years and years[0] == 2025
    assert mean == pytest.approx((1.2 * 4 + 0.6) / 5)
    assert lo_r == pytest.approx(0.6) and hi_r == pytest.approx(1.2)


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/de/run.py first")
def test_forward_scenarios_and_other_funds():
    df = pd.read_parquet(OUT).set_index("month")
    fc = df[df["is_forecast"] & ~df["cash_observed"]]
    assert np.allclose(fc["cash_requirement_sv_other_bn"], 0.0)  # no extrapolated one-off repayments
    parts = fc["cash_requirement_sv_infrastruktur_bn"] + fc["cash_requirement_sv_bundeswehr_bn"] + fc["cash_requirement_sv_other_bn"]
    assert np.allclose(parts, fc["cash_requirement_sv_bn"])
    assert (fc["cash_requirement_core_lo_bn"] <= fc["cash_requirement_core_hi_bn"] + 1e-9).all()
    assert df.loc[df["is_forecast"], "buffer_level_plan_bn"].notna().all()
    s = json.loads(SUM.read_text())["forward"]
    assert s["model_full_year_lo_bn"] <= s["model_full_year_bn"] <= s["model_full_year_hi_bn"]
    assert sum(s["model_minus_plan_split_bn"].values()) == pytest.approx(s["model_minus_plan_bn"], abs=0.15)


@pytest.mark.skipif(not (DERIVED.parent / "de_cash_dashboard.html").exists(), reason="dashboard not built")
def test_page_text_is_clean():
    import re

    page = (DERIVED.parent / "de_cash_dashboard.html").read_text()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    assert "{'" not in text and "€-" not in text            # no raw dicts, true minus signs
    assert not re.search(r"\b[a-z]+(?:_[a-z0-9]+)+_bn\b", text)  # no raw column names
    assert not re.search(r"\bDE000[0-9A-Z]{7}\b", text)       # ISINs only in hover
    assert "modelled" in text


def test_stock_chart_starts_at_last_actual():
    from sovereign_cash.de import dashboard as D

    idx = pd.date_range("2026-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"month": idx, "is_forecast": [False] * 3 + [True] * 3,
                       "bill_stock_bn": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan],
                       "bill_stock_implied_bn": [np.nan] * 3 + [13.0, 14.0, 15.0]})
    fig = D._fig_stock(df, "Bubill")
    imp = [t for t in fig.data if "forward path" in (t.name or "")][0]
    assert pd.Timestamp(imp.x[0]) == idx[2] and imp.y[0] == 12.0


def test_sv_forward_bundeswehr_hist_backload_when_scale_stable():
    """Stable-scale years: FA hist is H2-heavy — rest-of-year must not be flat YTD."""
    from sovereign_cash.de import model as M

    # Three complete BW years sum 21; plan 22 stays inside SV_SCALE_CHANNEL of peak.
    bw_prof = np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0, 5.0])  # sum 21
    rows = []
    for y in (2023, 2024, 2025):
        for i, m in enumerate(pd.date_range(f"{y}-01-01", periods=12, freq="MS")):
            rows.append({"month": m, "bw": float(bw_prof[i]), "inf": 2.0})
    for i, m in enumerate(pd.date_range("2026-01-01", periods=8, freq="MS")):
        rows.append({"month": m, "bw": float(bw_prof[i]) * 1.1, "inf": 3.5})
    h = pd.DataFrame(rows).set_index("month")
    h["cash_requirement_sv_bundeswehr_bn"] = h["bw"]
    h["cash_requirement_sv_infrastruktur_bn"] = h["inf"]
    plan = {2026: {"bundeswehr": 22.0, "infrastruktur": 50.0},
            2027: {"bundeswehr": 22.0, "infrastruktur": 50.0}}
    months = pd.date_range("2026-09-01", periods=16, freq="MS")
    fc, info = M.sv_forward(h, months, plan)
    assert info["sv_profile"]["bundeswehr"] == "hist_month_share"
    assert info["sv_scale_channel"]["bundeswehr"] is False
    rest = fc.loc["2026-09":"2026-12", "cash_requirement_sv_bundeswehr_bn"]
    assert float(rest.loc["2026-12-01"]) > float(rest.loc["2026-09-01"])
    ytd = float(h.loc["2026-01":"2026-08", "bw"].sum())
    flat_ye = ytd * 12 / 8
    assert float(info["sv_implied_ye_bn"]["bundeswehr"]) > flat_ye + 1.0


def test_sv_forward_scale_channel_plan_residual_not_hist_shape():
    """Defence / fund ramp: Wirtschaftsplan ≫ hist years → YE=plan, residual flat."""
    from sovereign_cash.de import model as M

    bw_prof = np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.5, 3.0, 5.0])  # sum 21
    rows = []
    for y in (2023, 2024, 2025):
        for i, m in enumerate(pd.date_range(f"{y}-01-01", periods=12, freq="MS")):
            rows.append({"month": m, "bw": float(bw_prof[i]), "inf": 2.0})
    for i, m in enumerate(pd.date_range("2026-01-01", periods=8, freq="MS")):
        rows.append({"month": m, "bw": float(bw_prof[i]) * 1.1, "inf": 3.5})
    h = pd.DataFrame(rows).set_index("month")
    h["cash_requirement_sv_bundeswehr_bn"] = h["bw"]
    h["cash_requirement_sv_infrastruktur_bn"] = h["inf"]
    # 40 / 21 ≥ 1.20 → scale channel
    plan = {2026: {"bundeswehr": 40.0, "infrastruktur": 50.0},
            2027: {"bundeswehr": 40.0, "infrastruktur": 50.0}}
    months = pd.date_range("2026-09-01", periods=16, freq="MS")
    fc, info = M.sv_forward(h, months, plan)
    assert info["sv_scale_channel"]["bundeswehr"] is True
    assert info["sv_profile"]["bundeswehr"] == "plan_residual_flat"
    assert abs(float(info["sv_implied_ye_bn"]["bundeswehr"]) - 40.0) < 1e-9
    rest = fc.loc["2026-09":"2026-12", "cash_requirement_sv_bundeswehr_bn"]
    # Flat residual — Dec must not inherit the hist ~2.5× Sep spike.
    assert abs(float(rest.loc["2026-12-01"]) - float(rest.loc["2026-09-01"])) < 0.05
    ytd = float(h.loc["2026-01":"2026-08", "bw"].sum())
    assert abs(float(rest.sum()) - (40.0 - ytd)) < 0.05
