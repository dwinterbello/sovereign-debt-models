"""Norway cash framework tests. No network: synthetic SDMX-shaped frames and cached derived outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.no import build as B

MONTHLY = contract.DERIVED / "no_cash_monthly.parquet"
SUMMARY = contract.DERIVED / "no_cash_summary.json"
DAILY = contract.DERIVED / "no_liquidity_daily.parquet"
PAGE = contract.ROOT / "no_cash_dashboard.html"


def _ale(flows: pd.DataFrame) -> pd.DataFrame:
    """Daily flows (NOK mn, columns = SDMX detail codes) → long ALE frame accumulated from 1 January."""
    acc = flows.groupby(flows.index.year).cumsum()
    long = acc.stack().rename("value").reset_index()
    long.columns = ["TIME_PERIOD", "LIQUIDITY_DETAILS", "value"]
    long["FREQ"], long["UNIT_MEASURE"] = "B", "ALE"
    return long


def _flows(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-11-03", "2026-02-27")
    f = pd.DataFrame(0.0, index=idx, columns=list(B.LINES))
    f["tab1_linje01"] = rng.normal(0, 3000, len(idx))
    f["tab1_linje03"] = 300.0
    f["tab1_linje14"] = -124.0
    f["tab2_linje01"] = np.where(idx.day == 15, -5000.0, 0.0)
    f["tab2_linje02"] = np.where(idx.day == 17, 2000.0, 0.0)
    f["tab2_linje07"] = np.where(idx.day == 10, -4000.0, 0.0)
    f["tab2_linje08"] = np.where(idx.day == 20, 3000.0, 0.0)
    f["tab2_linje03"] = 0.0
    f["tab1_linje02"] = f[["tab2_linje01", "tab2_linje02", "tab2_linje07", "tab2_linje08"]].sum(axis=1) + 50.0
    return f


def test_ale_differencing_across_year_boundary():
    f = _flows()
    got = B.daily_flows(_ale(f))
    exp = f.rename(columns=B.LINES)[got.columns] / 1000.0
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy(), atol=1e-9)
    assert got.loc["2026-01-01", "gpfg_fx"] == pytest.approx(0.3)


def test_history_identity_and_lines():
    fl = B.daily_flows(_ale(_flows()))
    h = B.history(fl).reset_index().rename(columns={"index": "month", "TIME_PERIOD": "month"})
    assert contract.validate(h) < 1e-9
    jan = h[h["month"] == pd.Timestamp("2026-01-01")].iloc[0]
    assert jan["bond_gross_bn"] == pytest.approx(5.0)
    assert jan["bill_redemption_bn"] == pytest.approx(3.0)
    assert jan["securities_other_bn"] == pytest.approx(-0.05 * len(pd.bdate_range("2026-01-01", "2026-01-31")))
    assert jan["oil_fund_legs_bn"] == pytest.approx(-(0.3 - 0.124) * len(pd.bdate_range("2026-01-01", "2026-01-31")))


def test_score_daily_perfect_forecast():
    idx = pd.bdate_range("2025-01-01", "2026-06-30")
    S = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 5, len(idx))), index=idx)
    a = pd.Timestamp("2026-03-02")
    vin = {a + pd.Timedelta(days=1): (a, float(S[a]), S[S.index > a])}
    sc = B.score_daily(vin, S)
    assert set(sc["h"]) == {f"{h}bd" for h in B.HORIZONS}
    np.testing.assert_allclose(sc["nb"], sc["actual"])
    assert sc["naive_ly"].notna().all()


def test_score_monthly_starts_from_forecast_month_end():
    idx = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    S = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    a = pd.Timestamp("2026-02-26")  # last realised day before month end
    path = S[S.index > a] + 100.0  # level bias of 100 that a correct change score must ignore
    vin = {pd.Timestamp("2026-02-27"): (a, float(S[a]), path)}
    sc = B.score_monthly(vin, S)
    mar = sc[(sc["month"] == pd.Timestamp("2026-03-01")) & (sc["h"] == "m+1")].iloc[0]
    assert mar["nb"] == pytest.approx(mar["actual"])


def test_benchmark_labels_per_kind():
    d = pd.DataFrame({"target": [pd.Timestamp("2023-03-01")], "h": ["5bd"], "nb": [1.0], "actual": [0.0],
                      "naive_ly": [2.0]})
    assert "weekday" in B.summarise_scores(d, "target")[0]["benchmark_name"]
    m = d.rename(columns={"target": "month"}).assign(h="m+1")
    assert "calendar month" in B.summarise_scores(m, "month")[0]["benchmark_name"]


def _mflows_with_history() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    shape = {1: 5, 2: 25, 3: 7, 4: 3, 5: 20, 6: 8, 7: 2, 8: 5, 9: 8, 10: 9, 11: 6, 12: 2}
    return pd.DataFrame({"bond_issue": [-float(shape[m.month]) for m in idx]}, index=idx)


CAL = {"programmes": {"2026": {"bonds_bn": [100.0, 110.0], "bill_market_stock_end_bn": 34.0}},
       "quarter_guidance": [{"year": 2026, "quarter": 4, "bonds_bn": [16.0, 19.0]}],
       "bond_auctions": ["2026-09-16", "2026-10-07", "2026-10-14", "2026-10-21", "2026-10-28", "2026-11-11",
                         "2026-11-25", "2026-12-02", "2026-12-09"],
       "buybacks": [{"date": "2026-11-25", "bond": "NGB 02/2027", "maturity": "2027-02-17"}],
       "bill_auctions": [
           {"date": "2026-09-14", "bill": "NTB 09/2027", "maturity": "2027-09-15", "market_bn": 6.0, "own_bn": 4.0},
           {"date": "2026-10-05", "bill": "NTB 12/2026", "maturity": "2026-12-16", "market_bn": 2.0, "own_bn": 0.0},
           {"date": "2026-10-19", "bill": "NTB 03/2027", "maturity": "2027-03-17", "market_bn": 2.0, "own_bn": 0.0},
           {"date": "2026-11-02", "bill": "NTB 06/2027", "maturity": "2027-06-16", "market_bn": 2.0, "own_bn": 0.0},
           {"date": "2026-11-16", "bill": "NTB 03/2027", "maturity": "2027-03-17", "market_bn": 2.0, "own_bn": 0.0},
           {"date": "2026-12-14", "bill": "NTB 12/2027", "maturity": "2027-12-15", "market_bn": 6.0, "own_bn": 4.0}]}


def test_ngb_path_splits_guidance_by_auction_dates_and_scenario_after():
    mf = _mflows_with_history()
    months = pd.date_range("2026-10-01", periods=12, freq="MS")
    g, lab = B.ngb_gross_path(months, mf, pd.Timestamp("2026-09-29"), CAL)
    q4 = g[g.index.year == 2026]
    assert q4.sum() == pytest.approx(17.5)
    np.testing.assert_allclose(q4.to_numpy(), 17.5 / 8 * np.array([4, 2, 2]))
    assert all("guidance" in x for x in lab[lab.index.year == 2026])
    y27 = g[g.index.year == 2027]
    sh = B.ngb_month_shares(mf, pd.Timestamp("2026-09-29"))
    assert y27.sum() == pytest.approx(105.0 * sh.loc[1:9].sum())
    assert all(x.startswith("scenario") for x in lab[lab.index.year == 2027])


def _gpm_rows(isin, typ, mat, events, instr):
    out = []
    for t, unit, v, it in events:
        out.append({"INSTRUMENT_TYPE": typ, "UNIT_MEASURE": unit, "ISIN": isin, "MATURITY": mat, "ISSUE_TYPE": it,
                    "TIME_PERIOD": t, "value": float(v), "INSTR": instr})
    return out


def test_bill_path_from_calendar_reaches_published_aim():
    rows = (_gpm_rows("D", "TBIL", "2026-12-16", [("2026-06-01", "O_V", 16000, "AU"), ("2026-06-01", "V_O", 6000, "AU")], "NTB 12/2026")
            + _gpm_rows("M", "TBIL", "2027-03-17", [("2026-08-31", "O_V", 14000, "AU"), ("2026-08-31", "V_O", 4000, "AU")], "NTB 03/2027")
            + _gpm_rows("J", "TBIL", "2027-06-16", [("2026-06-15", "O_V", 10000, "AU"), ("2026-06-15", "V_O", 4000, "AU")], "NTB 06/2027")
            + _gpm_rows("S", "TBIL", "2027-09-15", [("2026-09-14", "O_V", 10000, "AU"), ("2026-09-14", "V_O", 4000, "AU")], "NTB 09/2027"))
    months = pd.date_range("2026-10-01", periods=12, freq="MS")
    p, now = B.bill_path(pd.DataFrame(rows), CAL, months, pd.Timestamp("2026-09-29"))
    assert now == pytest.approx(32.0)
    np.testing.assert_allclose(p["bill_gross_bn"].iloc[:3], [4.0, 4.0, 6.0])
    assert p.loc["2026-12-01", "bill_redemption_bn"] == pytest.approx(12.0)  # 10 market + the 5-Oct tap
    assert p.loc["2026-12-01", "bill_stock_market_bn"] == pytest.approx(34.0)
    assert p.loc["2027-03-01", "bill_redemption_bn"] == pytest.approx(14.0)
    after = p[p.index > pd.Timestamp("2026-12-01")]
    assert (after["bill_basis"] == "rolled").all()
    np.testing.assert_allclose(after["bill_gross_bn"], after["bill_redemption_bn"])


def test_buyback_outlook_median_share_and_range():
    rows = []
    for k, (mat, bought) in enumerate((("2023-05-24", 10000), ("2024-03-14", 0), ("2025-03-13", 20000))):
        start = pd.Timestamp(mat) - pd.DateOffset(years=1, days=10)
        ev = [(f"{start:%Y-%m-%d}", "O_V", 104000, "AU"), (f"{start:%Y-%m-%d}", "V_O", 4000, "OWN")]
        if bought:
            ev.append((f"{pd.Timestamp(mat) - pd.Timedelta(days=60):%Y-%m-%d}", "V_A", bought, "BB"))
        rows += _gpm_rows(f"H{k}", "GBON", mat, ev, f"NGB {k}")
    rows += _gpm_rows("NOW", "GBON", "2027-02-17", [("2025-01-08", "O_V", 54000, "AU"), ("2025-01-08", "V_O", 4000, "OWN")],
                      "NGB 02/2027")
    out = B.buyback_outlook(pd.DataFrame(rows), CAL, pd.Timestamp("2026-09-29"), pd.Timestamp("2027-09-30"))
    assert len(out) == 1
    b = out[0]
    assert b["share_median"] == pytest.approx(0.10) and b["share_max"] == pytest.approx(0.20)
    assert b["expected_buyback_bn"] == pytest.approx(5.0)
    assert b["redemption_central_bn"] == pytest.approx(45.0)
    assert b["redemption_low_bn"] == pytest.approx(40.0) and b["redemption_high_bn"] == pytest.approx(50.0)
    assert [f"{d:%Y-%m-%d}" for d in b["buyback_dates"]] == ["2026-11-25"]


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.no.run first")
def test_derived_forward_horizon_bills_buybacks_gpfg():
    df = pd.read_parquet(MONTHLY)
    s = json.loads(SUMMARY.read_text())
    fw = s["forward"]
    f = df[df["is_forecast"]]
    assert "cash_requirement_fc_official_bn" not in df.columns
    nb = f[~f["scenario"]]
    sc = f[f["scenario"]]
    assert nb["fc_source"].str.startswith("norges_bank").all()
    if s["bottom_up"].get("ship"):  # Tab4 x macro shipped by its PRE gate (0b17ae37b): it replaces the smly tail
        assert (sc["fc_source"] == "bottom_up_macro").all()
    else:
        assert sc["fc_source"].str.startswith("same_month").all()  # the overview drops these by this prefix
    np.testing.assert_allclose(nb["cash_requirement_nb_implied_bn"], nb["cash_requirement_bn"])
    # headline window = Norges Bank's path only
    assert fw["window"] == f"{nb['month'].min():%Y-%m}..{nb['month'].max():%Y-%m}"
    assert fw["nb_horizon"]["buffer_change_bn"] == pytest.approx(round(nb["buffer_change_bn"].sum(), 1))
    # forward split of a residual is not published
    assert f["cash_petroleum_bn"].isna().all() and f["cash_nonpetroleum_bn"].isna().all()
    # bills: stock path consistent with gross − redemption
    stock = np.concatenate([[fw["bills"]["market_stock_now_bn"]], f["bill_stock_market_bn"].to_numpy()])
    np.testing.assert_allclose(f["net_bills_bn"], np.diff(stock), atol=0.06)
    # NGB inside guidance and programme
    for g in fw["ngb"]["quarter_guidance"]:  # path_bn is the remaining auctions; once the quarter has started
        assert g["bn"][0] <= g["issued_bn"] + g["path_bn"] <= g["bn"][1]  # the guidance covers issued + path
    lo, hi = fw["ngb"]["programme_bn"]
    assert lo <= fw["ngb"]["programme_path_bn"] <= hi
    # buybacks move from the maturity month to the buyback month
    nr = fw["next_redemption"]
    if nr and nr["expected_in_path_bn"] > 0:
        mm = pd.Timestamp(nr["maturity"]).to_period("M").to_timestamp()
        row = df[df["month"] == mm]
        if len(row):
            assert row["bond_redemption_bn"].iloc[0] == pytest.approx(nr["central_bn"], abs=0.06)
        assert df["bond_buyback_bn"].sum() == pytest.approx(nr["expected_in_path_bn"], abs=0.06)
        assert nr["low_bn"] <= nr["central_bn"] <= nr["high_bn"] <= nr["market_now_bn"] + 1e-9
    # GPFG: at most the first month announced (published at the end of the prior month), then carried
    assert f["gpfg_fx_basis"].iloc[0] in ("announced", "latest_announced_carried")
    assert (f["gpfg_fx_basis"].iloc[1:] == "latest_announced_carried").all()
    assert f["gpfg_fx_nok_bn_per_day"].nunique() == 1
    np.testing.assert_allclose(df["gov_liquidity_effect_bn"], -df["buffer_change_bn"], atol=1e-9)
    h = df[~df["is_forecast"]]
    np.testing.assert_allclose(h["cash_nonpetroleum_bn"] + h["cash_petroleum_bn"], h["cash_requirement_bn"], atol=1e-9)
    part = df[df["partial_month"]]
    assert len(part) <= 1 and (~part["is_forecast"]).all()


@pytest.mark.skipif(not MONTHLY.exists(), reason="run sovereign_cash.no.run first")
def test_derived_contract():
    df = pd.read_parquet(MONTHLY)
    assert contract.validate(df) < 1e-6
    assert df.loc[~df["is_forecast"], "month"].min() == pd.Timestamp("2008-01-01")
    assert (df["is_forecast"].astype(int).diff().fillna(0) >= 0).all()
    s = json.loads(SUMMARY.read_text())
    assert s["country"] == "no" and s["identity_resid_bn"] == 0.0
    for r in s["score"]:
        assert r["model"] < r["benchmark"], r
        assert ("weekday" in r["benchmark_name"]) == r["h"].endswith("bd")
    assert s["own_forecast_score"].startswith("unscored")
    d = pd.read_parquet(DAILY)
    assert {"structural_liquidity_bn", "structural_liquidity_fc_bn", "nowa", "gov_accounts"} <= set(d.columns)
    xc = s["structural_crosscheck_resid_bn"]["months_abs_gt_0p5bn"]
    assert all(v != 0 for v in xc.values())


@pytest.mark.skipif(not PAGE.exists(), reason="run sovereign_cash.no.run first")
def test_page_has_no_raw_ids_or_ascii_negatives():
    import re

    page = PAGE.read_text()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    for raw in ("_bn", "fc_source", "partial_month", "FCAST_", "same_month_last_year", "norges_bank_vintage"):
        assert raw not in text, raw
    assert not re.search(r"(?<![\w/–-])-\d", text)  # negatives use a true minus
    # the seasonal views (0b17ae37b) embed ~580KB of per-year chart data (NOS, NOCATS); the budget still catches
    # an accidental frame dump
    assert len(page) < 1_000_000
