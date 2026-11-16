import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt import build as B
from sovereign_cash.debt import sources as S

DERIVED = Path(__file__).resolve().parents[1] / "derived"
SNAP = DERIVED / "debt_snapshot.parquet"
RAW = Path(__file__).resolve().parents[1] / "data" / "debt"

A9 = """Table A.9: Fiscal aggregates
Outturn Forecast
2024-25 2025-26 2026-27 2027-28
Per cent of GDP
Public sector net borrowing (b-a) 5.1 4.5 3.6 2.9
General government gross debt 101.0 102.1 102.0 102.5
£ billion
Public sector net borrowing 152.7 132.7 115.5 96.5
Net debt interest 82.2 90.0 89.0 94.9
Non-interest receipts 1,095.0 1,194.0 1,261.0 1,330.0
"""


def test_parse_obr_a9_reads_both_blocks():
    d = S.parse_obr_a9(A9, "March 2026")
    v = d.set_index(["var", "year"])["value"]
    assert v[("deficit_bn", 2026)] == 115.5
    assert v[("revenue_ex_int_bn", 2024)] == 1095.0
    assert v[("interest_net_bn", 2027)] == 94.9
    assert set(d["year"]) == {2024, 2025, 2026, 2027}
    pct = d[d["unit"] == "% GDP"]
    assert len(pct) and (pct["value"] < 200).all()


@pytest.mark.skipif(not (RAW / "cao_keisuuhyo.xlsx").exists(), reason="CAO workbook not cached")
def test_parse_cao_cases_and_general_account():
    d = S.parse_cao((RAW / "cao_keisuuhyo.xlsx").read_bytes())
    base = d[d["source"] == "cao_baseline"]
    assert {"gdp_lcu_bn", "eff_rate", "balance_pct", "revenue_cg_bn", "interest_cg_bn"} <= set(base["var"])
    rev = base[base["var"] == "revenue_cg_bn"].set_index("year")["value"]
    assert 50_000 < rev.loc[2026] < 200_000  # JPY bn: ~¥90tn of tax revenue


def test_reinvestment_identity():
    snap = pd.DataFrame([{"cc": "de", "rollovers_bn": 300.0, "cb_01_bn": 70.0, "net_issuance_fwd": 180.0,
                          "fw_cb_net": -60.0, "fx": 1.1, "gdp_bn": 4600.0, "hn_private": 220.0},
                         {"cc": "us", "rollovers_bn": 10_000.0, "cb_01_bn": 450.0, "net_issuance_fwd": 1500.0,
                          "fw_cb_net": 0.0, "fx": 1.0, "gdp_bn": 32_500.0, "hn_private": 2900.0}])
    r = B.reinvestment(snap).set_index(["cc", "reinvest"])
    # central-bank maturities reach the market once (through fw_cb_net), never as a private non-roll
    assert r.at[("de", 1.0), "new_money_bn"] == pytest.approx(240.0)
    assert r.at[("de", 0.9), "new_money_bn"] == pytest.approx(240.0 + 0.1 * 230.0)
    assert r.at[("us", 0.9), "new_money_bn"] == pytest.approx(1500.0 + 0.1 * 9550.0)


@pytest.mark.skipif(not SNAP.exists(), reason="run sovereign_cash.debt.build first")
def test_snapshot_identities():
    s = pd.read_parquet(SNAP).set_index("cc")
    lad = s.loc[list(B.LADDER)]
    ok = lad["gross_issuance_fwd"].notna()
    assert np.allclose(lad.loc[ok, "gross_issuance_fwd"], lad.loc[ok, "rollovers_bn"] + lad.loc[ok, "net_issuance_fwd"])
    assert np.allclose(s["net_issuance_fwd"], s["fw_net_bonds"] + s["fw_net_bills"])
    ds = s["debt_service_pct_rev"].dropna()
    assert np.allclose(ds, (s["interest_pct_rev"] + s["rollovers_pct_rev"]).loc[ds.index])
    assert (s.loc[list(B.LADDER), "fw_months"] == 12).all()
    assert s.at["us", "fw_cb_net"] == 0.0  # the Fed rolls at auction


@pytest.mark.skipif(not (DERIVED / "debt_flows_monthly.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_cb_purchases_have_no_series_start_jump():
    f = pd.read_parquet(DERIVED / "debt_flows_monthly.parquet")
    jp = f[f["cc"] == "jp"].set_index("month")["cb_net_bn"].dropna()
    assert jp.abs().max() < 30_000  # BoJ holdings open at ~¥90tn in Mar-2013; that level is not a purchase


@pytest.mark.skipif(not (DERIVED / "debt_flows_monthly.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_jp_forward_net_is_coupon_book_not_cash_smly():
    """JP forecast net bonds = MoF-plan coupon stock Δ, not GA cash_requirement smly."""
    from sovereign_cash import investor as iv

    f = pd.read_parquet(DERIVED / "debt_flows_monthly.parquet")
    jp = f[f["cc"] == "jp"].sort_values("month")
    fw = jp[jp["is_forecast"]].head(12)
    assert len(fw) == 12
    cash = iv.monthly("jp")
    assert cash is not None
    c = cash.set_index(pd.to_datetime(cash["month"]))["cash_requirement_bn"]
    # Must not be the cash-requirement splice (that summed ~¥34tn vs book ~¥12–18tn).
    assert float(fw["net_bonds_bn"].sum()) != pytest.approx(float(c.reindex(fw["month"]).sum()), abs=1.0)
    # Forward 12m net stays in the same neighbourhood as the trailing coupon-book net.
    tr = jp[~jp["is_forecast"]].tail(12)
    gdp = float(pd.read_parquet(SNAP).set_index("cc").at["jp", "gdp_bn"])
    step_pp = 100.0 * (float(fw["net_bonds_bn"].sum()) - float(tr["net_bonds_bn"].sum())) / gdp
    assert abs(step_pp) < 3.0, f"JP fw−tr net step {step_pp:.2f}pp of GDP (cash smly was ~+4pp)"


@pytest.mark.skipif(not (DERIVED / "debt_flows_monthly.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_no_country_splices_forecast_net_to_cash_requirement():
    """Forecast net_bonds must not be the cash-deficit series when a real net_bonds column exists."""
    from sovereign_cash import investor as iv

    f = pd.read_parquet(DERIVED / "debt_flows_monthly.parquet")
    bad = []
    for cc in ("uk", "de", "fr", "it", "es", "jp", "se", "no"):
        cash = iv.monthly(cc, naive_tail=True)
        if cash is None or "cash_requirement_bn" not in cash.columns:
            continue
        if "net_bonds_bn" not in cash.columns:
            continue
        fw = f[(f["cc"] == cc) & f["is_forecast"]].sort_values("month").head(12)
        if fw.empty:
            continue
        c = cash.set_index(pd.to_datetime(cash["month"]))
        need = c.reindex(fw["month"])["cash_requirement_bn"]
        nb = c.reindex(fw["month"])["net_bonds_bn"]
        # Only flag when cash need ≠ cash net (otherwise the framework is bond-cleared).
        if need.notna().sum() < 10 or nb.notna().sum() < 10:
            continue
        if np.allclose(need.to_numpy(float), nb.to_numpy(float), atol=0.05, equal_nan=False):
            continue
        fl = fw.set_index("month")["net_bonds_bn"]
        if np.allclose(fl.to_numpy(float), need.reindex(fl.index).to_numpy(float), atol=0.05, equal_nan=False):
            bad.append(cc)
    assert not bad, f"forecast net_bonds == cash_requirement (wrong splice): {bad}"


@pytest.mark.skipif(not (DERIVED / "debt_signal.json").exists(), reason="run sovereign_cash.debt.signal first")
def test_signal_flags_consistent():
    t = pd.DataFrame(json.loads((DERIVED / "debt_signal.json").read_text())["tests"])
    assert not (t["supports_thesis"] & ~t["passes"]).any()
    assert ((t["supports_thesis"] | t["against_thesis"]) == t["passes"]).all()
    assert t["oos_r2"].notna().all()


@pytest.mark.skipif(not (DERIVED / "debt_country_annual.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_dts_year_is_used_only_when_the_calendar_year_is_complete():
    idx = pd.date_range("2027-01-01", "2028-06-01", freq="MS")
    b = pd.DataFrame({"net_coupons_bn": 100.0, "private_net_bills_bn": 40.0, "interest_cash_bn": 90.0}, index=idx)
    y = B.us_dts_years(b)
    assert list(y.index) == [2027]
    assert y.at[2027, "net_issuance_bn"] == pytest.approx(140.0 * 12)
    assert y.at[2027, "interest_cash_bn"] == pytest.approx(90.0 * 12)
    assert not np.isfinite(y.at[2027, "bills_bn"])


def test_country_annual_debt_moves_with_net_issuance():
    c = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    for cc, d in c.groupby("cc"):
        d = d.set_index("year").sort_index().dropna(subset=["debt_bn", "net_issuance_bn"])
        ni = d["net_issuance_bn"] + d.get("il_accrual_bn", pd.Series(0.0, index=d.index)).fillna(0.0)
        dd = d["debt_bn"].diff().dropna()
        ni = ni.reindex(dd.index)
        if cc in B.OFFICIAL or cc == "uk":  # history is the stock incl. uplift; forward adds the accrual
            dd, ni = dd[dd.index >= B.Y0 + 2], ni[ni.index >= B.Y0 + 2]
        assert np.allclose(dd, ni, rtol=1e-6, atol=1e-6), cc
    us = c[(c["cc"] == "us") & (c["year"] == 2025)].iloc[0]
    assert 180 < us["debt_service_pct_rev"] < 230  # Dalio: debt service ~2x revenue
    us15 = c[(c["cc"] == "us") & (c["year"] == 2015)].iloc[0]
    assert 13_000 < us15["debt_bn"] < 14_000  # MSPD total marketable Dec-2015 incl. TIPS compensation
    assert (c["debt_service_ex_bills_pct_rev"] <= c["debt_service_pct_rev"] + 1e-9).all()


def test_uk_uplift_matches_dmo():
    # nominal books miss the index-linked uplift (£275bn, ~9pp GDP); the bottom-up build must equal D1A
    u = S.uk_uplift(False)
    now = u.set_index("var")["value"]
    assert 200 < now["il_uplift_now_bn"] < 400 and 350 < now["il_nominal_now_bn"] < 550
    s = pd.read_parquet(SNAP).set_index("cc")
    assert s.at["uk", "marketable_bn"] > s.at["uk", "coupon_bn"] + s.at["uk", "bills_bn"] + 200


@pytest.mark.skipif(not (DERIVED / "debt_snapshot.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_effective_rates_are_on_the_debt_basis():
    # interest and debt on different bases show up as an effective rate no bond book could pay
    # (NO S1311 interest over marketable debt read 7.5%; JP budget-year interest 1.1% vs ~0.8% JGB coupon)
    s = pd.read_parquet(DERIVED / "debt_snapshot.parquet").set_index("cc")
    assert s["eff_rate"].between(0.3, 5.0).all(), s["eff_rate"].to_dict()
    assert s.loc["jp", "eff_rate"] < 1.0


@pytest.mark.skipif(not (DERIVED / "debt_country_annual.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_country_annual_covers_every_country():
    c = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    for cc, d in c.groupby("cc"):
        d = d.set_index("year")
        assert d.loc[2016:2030, ["debt_bn", "rollovers_bn", "interest_bn", "revenue_bn"]].notna().all().all(), cc
    jp = c[c["cc"] == "jp"].set_index("year")["revenue_bn"]
    assert jp.loc[2020:2021].diff().iloc[-1] < 10_000  # no Covid carry-over swing in revenue


@pytest.mark.skipif(not (DERIVED / "debt_country_annual.parquet").exists(), reason="run sovereign_cash.debt.build first")
def test_unified_forecast_stays_near_the_authorities():
    # one rule set for every country; a forecast far from every authority means a broken input, not a view
    c = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    last = c[c["year"] == c["year"].max()].set_index("cc")
    ratio = (last["interest_bn"] / last["interest_authority_bn"]).dropna()
    assert ratio.between(0.7, 1.4).all(), ratio.to_dict()
    s = pd.read_parquet(DERIVED / "debt_snapshot.parquet")
    assert 0.0 <= s["gap_persistence"].iloc[0] < 0.8
    assert s["new_bond_yield"].notna().all()


def test_redemption_schedule_matches_the_bond_stock():
    # an unparsed securities list returns zeros without raising (SE until 2026-10-14)
    src = S.load(False)
    sched = B.redemption_schedule(src)
    s = pd.read_parquet(SNAP).set_index("cc")
    for cc in B.CC:
        bonds = s.at[cc, "marketable_bn"] - (s.at[cc, "bills_bn"] if np.isfinite(s.at[cc, "bills_bn"]) else 0.0)
        r = sched[cc]
        assert 0.8 < r[r.index >= pd.Timestamp.today().year].sum() / bonds < 1.25, cc
        assert abs(B.tenor_mix(cc, sched).sum() - 1) < 1e-9


def test_gg_forward_is_central_plus_frozen_residual():
    """GG interest = central bill + the last-outturn gap (BoJ coupons are memo-only, not subtracted)."""
    c = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    for cc, d in c.groupby("cc"):
        d = d.set_index("year")
        resid = d.at[2025, "gg_interest_bn"] - d.at[2025, "interest_bn"]
        for y in range(2026, 2030):
            assert d.at[y, "gg_interest_bn"] == pytest.approx(d.at[y, "interest_bn"] + resid, rel=0, abs=1e-6), (cc, y)


def test_failed_curve_fetch_keeps_the_previous_row(monkeypatch, tmp_path):
    import sovereign_cash.debt.gov_curve as g
    assert g.CACHE.is_file()
    cache = tmp_path / "gov_zeros.parquet"
    cache.write_bytes(g.CACHE.read_bytes())
    monkeypatch.setattr(g, "CACHE", cache)
    monkeypatch.setattr(g, "LOG", tmp_path / "log.json")
    monkeypatch.setattr(g, "TV_PAR", tmp_path / "tv.parquet")
    for name in ("_boe", "_gsw", "_bbk", "_mof", "fetch_tradingview", "_auction_real"):
        monkeypatch.setattr(g, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = g.build()
    assert set(out["cc"]) >= {"uk", "us", "de", "jp", "fr", "it"}
    log = json.loads((tmp_path / "log.json").read_text())
    assert log["kept"] and set(log["ok"]) <= {"breakeven_real"}


def test_gov_par_shock_is_one_point():
    # +100bp on the zero curve must land as about +100bp on the par yield. A bootstrap that
    # returns the zero, or a shock applied twice, fails this.
    from sovereign_cash.debt.gov_curve import CACHE, forward_par
    if not CACHE.is_file():
        pytest.skip("gov curve cache not built")
    for cc in ("uk", "us", "de", "jp", "fr", "it"):
        spot = float(forward_par(cc, 0.0, 10)[0])
        shocked = float(forward_par(cc, 0.0, 10, 100)[0])
        assert 0.0 < spot < 12.0, (cc, spot)
        assert 0.90 < shocked - spot < 1.15, (cc, spot, shocked)


def test_linker_coefficients_and_boj_tbills():
    from sovereign_cash.debt.index_ratio import boj_tbill_bn, ratios
    r = ratios(pd.Timestamp("2026-10-17"))
    assert r["DE0001030583"] == pytest.approx(1.26842, abs=1e-4)
    assert r["JPIL10y#22"] == pytest.approx(1.16262, abs=1e-4)
    assert r["IT0005482994"] == pytest.approx(1.22950, abs=1e-4)
    assert r["IT0005532723"] == pytest.approx(1.00466, abs=1e-4)  # BTP Italia ex line, same maturity
    assert boj_tbill_bn() == 0.0


def test_forwards_are_exact_at_the_published_points():
    x, z = np.array([0.25, 2.0, 10.0, 30.0]), np.array([3.9, 4.6, 5.0, 5.4])
    assert np.allclose(B.forward((x, z), 0.0, x), z)
    f = [B.forward((x, z), k, 0.25) for k in np.arange(0.5, 5, 0.25)]
    assert np.max(np.abs(np.diff(f))) < 0.25
