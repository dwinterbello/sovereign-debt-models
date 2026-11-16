"""Sweden cash framework tests. No network: synthetic frames, fixed release text and cached derived outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.se import build as B
from sovereign_cash.se import fetch as F
from sovereign_cash.se import model as M

MONTHLY = contract.DERIVED / "se_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "se_cash_summary.json"


def _season(n_years: int = 12, trend: float = 0.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-01", periods=12 * n_years, freq="MS")
    prof = np.array([30, -20, -40, 10, 25, -60, 15, -5, -10, 20, 5, -70], dtype=float)
    return pd.Series(prof[idx.month - 1] + trend * np.arange(len(idx)) + rng.normal(0, 0.5, len(idx)), index=idx)


def test_parse_release_level():
    t = ("Statens budget visade ett underskott på 12,3 miljarder kronor i augusti. "
         "Riksgäldens prognos var ett underskott på 10,0 miljarder kronor.")
    p = F.parse_release("2026-09-07", t)
    assert p["month"] == pd.Timestamp("2026-08-01")
    assert p["outcome_bn"] == pytest.approx(-12.3)
    assert p["rg_forecast_bn"] == pytest.approx(-10.0)


def test_parse_release_deviation_and_year_roll():
    t = "Statens budget visade ett överskott på 5 miljarder kronor i december. Det är 2 miljarder kronor högre än prognos."
    p = F.parse_release("2027-01-11", t)
    assert p["month"] == pd.Timestamp("2026-12-01")
    assert p["outcome_bn"] == pytest.approx(5.0)
    assert p["rg_forecast_bn"] == pytest.approx(3.0)
    t2 = "Statens budget visade ett underskott på 8 miljarder kronor i mars. Utfallet var 1 miljard kronor sämre än prognos."
    assert F.parse_release("2026-04-08", t2)["rg_forecast_bn"] == pytest.approx(-7.0)


def test_parse_release_in_line_and_unparseable():
    t = "Statens budget visade ett överskott på 4 miljarder kronor i maj, i linje med Riksgäldens prognos."
    assert F.parse_release("2026-06-05", t)["rg_forecast_bn"] == pytest.approx(4.0)
    assert F.parse_release("2026-06-05", "Riksgälden emitterar en ny obligation.") is None


def test_k1_no_level_is_same_month_last_year():
    y = _season()
    origin = y.index[-13]
    fc = M.forecast(y, origin, k=1, a=0.0, f="mean", hmax=12)
    np.testing.assert_allclose(fc.to_numpy(), y.iloc[-24:-12].to_numpy())


def test_level_adjustment_tracks_trend():
    y = _season(trend=0.5)
    origin = y.index[-13]
    act = y.iloc[-12:].to_numpy()
    flat = M.forecast(y, origin, k=3, a=0.0, f="mean").to_numpy()
    adj = M.forecast(y, origin, k=3, a=1.0, f="mean").to_numpy()
    assert np.abs(adj - act).mean() < np.abs(flat - act).mean()


def test_anchor_hits_annual_total():
    idx = pd.date_range("2026-01-01", "2026-08-01", freq="MS")
    y = pd.Series(5.0, index=idx)
    raw = pd.Series(-10.0, index=pd.date_range("2026-09-01", "2027-12-01", freq="MS"))
    out = B.anchor_to_riksgalden(raw, y, idx[-1])
    assert out[out.index.year == 2026].sum() + y.sum() == pytest.approx(B.RG_ANNUAL_BALANCE[2026])
    assert out[out.index.year == 2027].sum() == pytest.approx(B.RG_ANNUAL_BALANCE[2027])
    d26 = out[out.index.year == 2026] - raw[raw.index.year == 2026]
    assert np.allclose(d26, d26.iloc[0])  # level shift only, shape preserved


GRID = """Statens betalningar resulterade i ett överskott på 61,9 miljarder kronor i augusti.
Riksgäldens prognos var ett överskott på 26,6 miljarder kronor.

+----------------------+---------+---------+-----------+-------+----------+--+
| Statens budgetsaldo  |         |         |           |       |          |  |
+----------------------+---------+---------+-----------+-------+----------+--+
|                      |  Utfall | Prognos | Differens |  Ack. |   Utfall |  |
|                      |         |         |           | diff. |   12-mån |  |
+----------------------+---------+---------+-----------+-------+----------+--+
| Budgetsaldo          |  61 935 |  26 619 |    35 316 |    13 | -109 219 |  |
|                      |         |         |           |   784 |          |  |
+----------------------+---------+---------+-----------+-------+----------+--+
"""


def test_parse_release_table_wins_and_wrapped_cells():
    p = F.parse_release("2026-09-07", GRID)
    assert p["parsed_from"] == "table"
    assert p["outcome_bn"] == pytest.approx(61.935)
    assert p["rg_forecast_bn"] == pytest.approx(26.619)
    assert p["rg_cum_dev_bn"] == pytest.approx(13.784)  # "13" / "784" wrapped over two lines


def test_parse_release_text_fallback_som_var():
    t = "Statens betalningar resulterade i ett underskott på 40,9 miljarder kronor i juni. " \
        "Det var i linje med Riksgäldens prognos, som var ett underskott på 39,8 miljarder."
    assert F.parse_release("2024-07-05", t)["rg_forecast_bn"] == pytest.approx(-39.8)


def test_auction_class_linkers_separate():
    s = pd.Series(["SGB IL 3116 0.75% 1 Jun 36", "SGB 1067 2.5% 15 Oct 36", "STB 16 Dec 26"])
    assert list(B.auction_class(s)) == ["linker", "nominal", "bill"]
    assert B.stb_maturity("STB 21 Okt 26") == pd.Timestamp("2026-10-21")


def test_bill_stock_anchor_date():
    """Snapshot dated after the last outcome month: remove bills settled since, add back bills matured since."""
    lad = pd.DataFrame({
        "settle": pd.to_datetime(["2026-06-11", "2026-09-03", "2026-09-17", "2026-10-01"]),
        "maturity": pd.to_datetime(["2026-09-16", "2027-09-15", "2027-03-17", "2027-01-20"]),
        "amount_bn": [42.5, 12.5, 20.0, 12.5]})
    snap = pd.Timestamp("2026-10-01")
    snap_stock = 12.5 + 20.0 + 12.5 + 100.0  # alive at snapshot (100 = older bills)
    end_aug = B.bill_stock_at(pd.Timestamp("2026-08-31"), snap_stock, snap, lad)
    end_sep = B.bill_stock_at(pd.Timestamp("2026-09-30"), snap_stock, snap, lad)
    assert end_aug == pytest.approx(100.0 + 42.5)
    assert end_sep == pytest.approx(100.0 + 12.5 + 20.0)
    assert end_sep - end_aug == pytest.approx(32.5 - 42.5)


def test_seasonal_share_anchor_and_carry():
    y = _season()
    origin = y.index[-5]  # Aug
    raw = M.forecast(y, origin, 5, 0.5, "mean", hmax=4)
    total = float(raw.sum()) - 100.0
    for meth in M.ALLOC_METHODS:
        al = M.allocate(raw, total, meth, y, origin)
        assert al.sum() == pytest.approx(total)
    disp = M.month_dispersion(y, origin)
    al = M.allocate(raw, total, "err_share", y, origin)
    gap = al - raw
    assert np.argmin(gap.to_numpy()) == np.argmax(disp.reindex(raw.index.month).to_numpy())
    idx = pd.date_range("2026-01-01", "2026-08-01", freq="MS")
    yy = pd.Series(5.0, index=idx)
    r = pd.Series(-10.0, index=pd.date_range("2026-09-01", "2027-12-01", freq="MS"))
    out = B.anchor_to_riksgalden(r, yy, idx[-1], "abs_share", carry=13.8)
    assert out[out.index.year == 2026].sum() + yy.sum() == pytest.approx(B.RG_ANNUAL_BALANCE[2026] + 13.8)
    assert out[out.index.year == 2027].sum() == pytest.approx(B.RG_ANNUAL_BALANCE[2027])


def test_auction_calendar_breaks_and_holidays():
    # weekly Wednesdays with a summer break (8 Jul → 5 Aug) and a year-end break (17 Dec → 14 Jan) in the history
    hist = list(pd.date_range("2025-11-05", "2025-12-17", freq="7D")) + list(pd.date_range("2026-01-14", "2026-07-08", freq="7D")) \
        + list(pd.date_range("2026-08-05", "2026-09-30", freq="7D"))
    assert B.cadence_days(hist) == 7
    brk = B.season_breaks(hist, 7)
    assert ((7, 8), (8, 5)) in brk and ((12, 17), (1, 14)) in brk
    cal = B.auction_calendar(hist, pd.Timestamp("2027-08-31"))
    assert cal[0] == pd.Timestamp("2026-10-07")
    assert all(t.weekday() == 2 for t in cal)
    assert pd.Timestamp("2026-12-16") in cal and pd.Timestamp("2026-12-23") not in cal
    assert pd.Timestamp("2027-01-13") not in cal and pd.Timestamp("2027-01-20") in cal
    assert pd.Timestamp("2027-07-07") in cal and pd.Timestamp("2027-07-14") not in cal
    assert pd.Timestamp("2027-08-04") not in cal and pd.Timestamp("2027-08-11") in cal


def test_timing_adjustment_window_and_receipt():
    items = ({"item": "x", "planned_month": "2026-07", "expected_year": 2026, "amount_bn": 15.0, "received_month": None},)
    since = pd.Timestamp("2026-04-01")
    assert B.timing_adjustment(pd.Timestamp("2026-08-01"), 2026, items, since)[0] == 15.0
    assert B.timing_adjustment(pd.Timestamp("2026-06-01"), 2026, items, since)[0] == 0.0  # not yet in the YTD
    assert B.timing_adjustment(pd.Timestamp("2026-08-01"), 2026, items, pd.Timestamp("2026-08-01"))[0] == 0.0  # new report
    got = ({**items[0], "received_month": "2026-10"},)
    assert B.timing_adjustment(pd.Timestamp("2026-09-01"), 2026, got, since)[0] == 15.0
    assert B.timing_adjustment(pd.Timestamp("2026-10-01"), 2026, got, since)[0] == 0.0  # arrival is in the outcomes
    assert B.timing_adjustment(pd.Timestamp("2026-08-01"), 2026, ({**items[0], "expected_year": 2027},), since)[0] == 0.0


def test_next_dates_from_release():
    rel = [{"date": "2026-09-07T06:00:00Z", "lang": "sv", "text":
            "Utfallet för september 2026 publiceras den 7 oktober 2026 klockan 08.00.\n\nDatum för publicering av ny "
            "prognos över Sveriges ekonomi och statens\nupplåning är den 26 november 2026."}]
    nd = F.next_dates(rel)
    assert nd["outcome"]["month"] == pd.Timestamp("2026-09-01") and nd["outcome"]["date"] == pd.Timestamp("2026-10-07")
    assert nd["report"]["date"] == pd.Timestamp("2026-11-26")


def test_bill_forward_rolls_and_hits_plan():
    lad = pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-09-15", "2026-09-29"]),
                        "series": ["a", "b", "c"],
                        "settle": pd.to_datetime(["2026-09-03", "2026-09-17", "2026-10-01"]),
                        "maturity": pd.to_datetime(["2026-12-16", "2027-03-17", "2027-01-20"]),
                        "amount_bn": [10.0, 10.0, 10.0]})
    alive = pd.DataFrame({"Maturity date": ["2026-12-16", "2027-03-17", "2027-01-20", "2026-10-21"],
                          "nominal_bn": [10.0, 10.0, 10.0, 50.0]})
    months = pd.date_range("2026-10-01", "2027-12-01", freq="MS")
    tenor = pd.Series({3: 0.5, 6: 0.5})
    s0 = 70.0  # end-September: everything alive except the bill settled on 1 October
    out, meta = B.bill_forward(lad, alive, pd.Timestamp("2026-10-01"), s0, pd.Timestamp("2026-09-01"), months, tenor,
                               {2026: 100.0, 2027: 90.0})
    assert out.loc["2026-12-01", "bill_stock_path_bn"] == pytest.approx(100.0)
    assert out.loc["2027-12-01", "bill_stock_path_bn"] == pytest.approx(90.0)
    np.testing.assert_allclose(s0 + out["net_bills_bn"].cumsum(), out["bill_stock_path_bn"])
    assert out.loc["2026-10-01", "bill_redemption_bn"] == pytest.approx(50.0)
    assert out["bill_gross_bn"].min() >= 0.0


def test_allocation_picks_lowest_pre():
    t = pd.to_datetime(["2015-01-01"] * 2 + ["2023-01-01"] * 2)
    abt = pd.DataFrame({"method": ["a", "b", "a", "b"], "target": t, "ae": [1.00, 1.004, 5.0, 1.0], "ae_rg": [9, 0, 9, 0]})
    assert M.select_allocation(abt)[0] == "a"


def test_skill_verdict_states_losses():
    score = [{"window": "OOS", "h": h, "model": m, "benchmark": 15.1, "riksgalden": 11.5, "model_on_rg_months": m}
             for h, m in ((1, 14.0), (12, 15.7))]
    v = B.skill_verdict(score)
    assert "loses to Riksgälden at every horizon" in v and "h=12 (15.70 vs 15.10)" in v


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.se.run first")
def test_derived_forward_plan_and_stocks():
    df = pd.read_parquet(MONTHLY)
    s = json.loads(SUMMARY.read_text())
    fw = s["forward"]
    for k, v in s["bill_stock_check"].items():
        assert abs(v["implied_bn"] - v["riksgalden_bn"]) < 1.5, k
    for k, v in s["liq_mgmt_stock_check"].items():
        assert abs(v["implied_bn"] - v["riksgalden_bn"]) < 1.5, k
    assert fw["plan_cash_requirement_bn"] == -B.RG_ANNUAL_BALANCE[fw["plan_year"]]
    # Riksgälden-consistent year: plan less the YTD deviation with announced timing items added back.
    expect = fw["plan_cash_requirement_bn"] - fw["riksgalden_ytd_deviation_bn"] - fw["timing_items_bn"]
    assert fw["plan_consistent_full_year_bn"] == pytest.approx(expect, abs=0.11)
    assert fw["model_full_year_bn"] == fw["plan_consistent_full_year_bn"]
    yr = df["month"].dt.year == fw["plan_year"]
    assert df.loc[yr, "cash_requirement_bn"].sum() == pytest.approx(fw["plan_consistent_full_year_bn"], abs=0.11)
    assert fw["pure_model_full_year_bn"] == pytest.approx(
        fw["cash_requirement_ytd_bn"] - fw[f"unanchored_model_rest_of_{fw['plan_year']}_bn"], abs=0.11)
    f = df[df["is_forecast"]]
    parts = f[["net_bonds_nominal_bn", "net_bonds_linker_bn", "net_bonds_fx_bn"]].sum(axis=1)
    np.testing.assert_allclose(parts, f["net_bonds_bn"], atol=1e-9)
    end = df[df["month"] == pd.Timestamp(fw["plan_year"], 12, 1)].iloc[0]
    assert end["bill_stock_implied_bn"] == pytest.approx(B.RG_PLAN["bill_stock_end"][fw["plan_year"]], abs=0.05)
    # Nominal SGB: settled + announced + calendar auctions land exactly on the year's plan.
    assert df.loc[yr, "bond_gross_nominal_bn"].sum() == pytest.approx(B.RG_PLAN["nominal_bonds"][fw["plan_year"]], abs=0.05)
    # The calendar is not flat: auction breaks show up as smaller months.
    assert f["bond_gross_nominal_bn"].nunique() > 2
    # The bill path is not a straight line: gross follows the maturity calendar.
    assert f.loc[f["securities_basis"] == "calendar", "bill_gross_bn"].std() > 5.0
    assert fw["bill_stock"]["snapshot_bill_stock_bn"] > 0 and fw["bill_stock"]["snapshot_date"]


@pytest.mark.skipif(not (contract.ROOT / "se_cash_dashboard.html").exists(), reason="run sovereign_cash.se.run first")
def test_page_text_hygiene():
    import re
    t = (contract.ROOT / "se_cash_dashboard.html").read_text()
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    assert not re.findall(r"(?<![\w\-:/.])-\d", text), "hyphen-minus in page numbers"
    for raw in ("_bn", "seas_k", "err_share", "securities_basis", "{'", "May plan", "assumed for July"):
        assert raw not in text, raw


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.se.run first")
def test_derived_contract():
    df = pd.read_parquet(MONTHLY)
    assert contract.validate(df) < 1.0
    h = df[~df["is_forecast"]]
    assert h["month"].min() <= pd.Timestamp("2002-01-01")
    assert (df["is_forecast"].astype(int).diff().fillna(0) >= 0).all()
    s = json.loads(SUMMARY.read_text())
    for k in ("country", "ccy", "asof", "sources", "buffer", "identity_resid_bn", "score", "forward", "notes"):
        assert k in s
    assert s["country"] == "se" and s["identity_resid_bn"] < 0.01
    pre = [r for r in s["score"] if r["window"] == "PRE" and r["h"] == 1][0]
    assert pre["model"] < pre["benchmark"]
    # OOS h=12 is reported as it is: the published verdict must match the score row, whichever way it falls.
    oos = [r for r in s["score"] if r["window"] == "OOS" and r["h"] == 12][0]
    assert s["oos_h12"]["model"] == oos["model"] and s["oos_h12"]["same_month_last_year"] == oos["benchmark"]
    assert s["oos_h12"]["model_beats_smly"] == (oos["model"] < oos["benchmark"])
    if not s["oos_h12"]["model_beats_smly"]:
        assert "does not beat same-month-last-year at h=12" in s["skill_verdict"]
    oos1 = [r for r in s["score"] if r["window"] == "OOS" and r["h"] == 1][0]
    if oos1["model_on_rg_months"] > oos1["riksgalden"]:
        assert "OOS: the model loses to Riksgälden" in s["skill_verdict"]
    assert s["shipped_forward_score"] and s["shipped_forward_score"][-1]["vintage"] == "pooled"
