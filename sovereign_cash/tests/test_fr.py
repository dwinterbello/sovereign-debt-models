"""France cash framework — offline tests on cached inputs and written outputs (no network)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.contract import DERIVED, REQUIRED, validate
from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import fetch as F

CACHED = (F.DATA / "insee_smb.xml").exists() and (F.DATA / "insee_debt.xml").exists()


@pytest.mark.skipif(not CACHED, reason="INSEE cache missing")
def test_insee_smb_parse():
    smb = F.load_smb()
    assert smb.index.min() <= pd.Timestamp("2009-01-01")
    assert {"solde_ytd", "depenses_ytd", "recettes_ytd"} <= set(smb.columns)
    # solde = recettes − dépenses + comptes spéciaux (rounding)
    d = (smb["recettes_ytd"] - smb["depenses_ytd"] + smb["comptes_speciaux_ytd"] - smb["solde_ytd"]).dropna().abs()
    assert (d < 0.5).mean() > 0.95
    # 2025 State deficit ~EUR 125bn
    assert -140 < smb.at[pd.Timestamp("2025-12-01"), "solde_ytd"] < -110


@pytest.mark.skipif(not CACHED, reason="INSEE cache missing")
def test_insee_debt_parse():
    d = F.load_debt()
    assert (d["btf"] + d["mlt"] - d["debt_total"]).dropna().abs().max() < 0.5
    assert 100 < d["btf"].dropna().iloc[-1] < 400


@pytest.mark.skipif(not CACHED, reason="INSEE cache missing")
def test_history_identity_and_ladder():
    from sovereign_cash.fr import model as M

    h, meta = M.build_history()
    resid = h["buffer_change_bn"] - (-h["cash_requirement_bn"] + h["net_bonds_bn"] + h["net_bills_bn"]
                                     + h["other_fin_bn"])
    assert resid.abs().max() < 1e-9
    assert (h["buffer_change_bn"] == 0).all()
    # BTF tape ladder = implied BTF redemptions (gross − Δ stock) in recent months
    ops = meta["ops"]
    for t in pd.date_range("2025-03-01", "2026-03-01", freq="3MS"):
        ms = pd.date_range(t, periods=1, freq="MS")
        lad = C.ladder_monthly(ops, t + pd.offsets.MonthEnd(0), ms, ["Bill"]).iloc[0]
        assert lad == pytest.approx(h.at[t, "bill_redemption_bn"], abs=0.5)


OUT = DERIVED / "fr_cash_monthly.parquet"
SUM = DERIVED / "fr_cash_summary.json"


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/fr/run.py first")
def test_outputs_contract():
    df = pd.read_parquet(OUT)
    assert set(REQUIRED) <= set(df.columns)
    assert validate(df) < 1e-6
    fc = df[df["is_forecast"]]
    assert len(fc) >= 12
    assert np.allclose(fc["bill_gross_bn"] - fc["bill_redemption_bn"], fc["net_bills_bn"])
    s = json.loads(SUM.read_text())
    for k in ("country", "ccy", "asof", "sources", "buffer", "identity_resid_bn", "score", "forward", "notes"):
        assert k in s
    assert s["country"] == "FR"
    assert any(not x["verified"] for x in s["sources"])  # budget.gouv.fr PDF is reported unverified
    for r in s["score"]:
        assert {"window", "h", "model", "benchmark", "n"} <= set(r)
    pre = [r for r in s["score"] if r["window"] == "PRE"]
    chosen = s["forward"]["model"]
    # selection is on PRE: the chosen model has the lowest mean PRE MAE among full-horizon common models
    means = {}
    for r in pre:
        for m, v in r["all_mae"].items():
            if v["common"]:
                means.setdefault(m, []).append(v["mae"])
    means = {m: np.mean(v) for m, v in means.items() if len(v) == len(C.HORIZONS)}
    assert min(means, key=means.get) == chosen or np.isclose(means[chosen], min(means.values()))
    assert (DERIVED.parent / "fr_cash_dashboard.html").exists()
    f = s["forward"]
    for k in ("plan_year", "plan_cash_requirement_bn", "plan_source", "cash_requirement_ytd_bn", "ytd_through",
              "model_full_year_bn", "plan_gap_lands_on", "plan_btf_stock_end_year_bn",
              "cb_held_redemptions_12m_bn", "private_net_bonds_12m_bn"):
        assert k in f, k
    assert f["private_net_bonds_12m_bn"] == pytest.approx(f["net_bonds_12m_bn"] + f["cb_held_redemptions_12m_bn"], abs=0.15)
    assert "estimated" in f["cb_held_basis"]


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/fr/run.py first")
def test_oat_reconciliation_reported():
    """Calendar-year net OAT vs programme − amortisation ± indexation: reported, and flagged on the page if off."""
    s = json.loads(SUM.read_text())
    r = s["forward"]["oat_reconciliation"]
    assert r is not None
    assert r["expected_bn"] == pytest.approx(r["programme_bn"] - r["amortisation_bn"] + r["indexation_bn"], abs=0.02)
    assert r["gap_bn"] == pytest.approx(r["net_oat_bn"] - r["expected_bn"], abs=0.02)
    page = (DERIVED.parent / "fr_cash_dashboard.html").read_text(encoding="utf-8")
    if not r["within_tol"]:
        assert any("beyond the" in n for n in s["notes"]) and "beyond the" in page


@pytest.mark.xfail(reason="2026 Δ OAT stock exceeds programme − LFI amortisation ± indexation by ~40bn "
                          "(buybacks of 2026 lines after the LFI vintage are not observable here)", strict=False)
@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/fr/run.py first")
def test_oat_reconciliation_within_tolerance():
    r = json.loads(SUM.read_text())["forward"]["oat_reconciliation"]
    assert abs(r["gap_bn"]) <= r["tol_bn"]


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/fr/run.py first")
def test_forward_other_fin_meets_plan_year():
    from sovereign_cash.fr import model as M

    s = json.loads(SUM.read_text())["forward"]
    df = pd.read_parquet(OUT).set_index("month")
    yr = s["plan_year"]
    p_of = M.plan_other_fin(F.load_json("plan.json")["years"][str(yr)])
    assert p_of == pytest.approx(s["plan_other_fin_implied_bn"], abs=0.05)
    assert df.loc[df.index.year == yr, "other_fin_bn"].sum() == pytest.approx(p_of, abs=0.05)


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/fr/run.py first")
def test_page_text_is_clean():
    import re

    page = (DERIVED.parent / "fr_cash_dashboard.html").read_text(encoding="utf-8")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    visible = re.sub(r"<[^>]+>", " ", text)
    for bad in ("seas_mult", "seas_add", "gapseas", "gap0", "_bn", "{'", "€-", "FR00", "fr_month_targets"):
        assert bad not in visible, bad
    assert "title='FR00" in text  # ISINs live in hover titles only


ENCOURS = all((F.DATA / f"aft_encours_{k}.html").exists() for k in F.ENCOURS_KINDS)


@pytest.mark.skipif(not ENCOURS, reason="AFT outstanding pages not cached")
def test_ladder_nets_buybacks():
    from sovereign_cash.fr import model as M

    months = pd.date_range("2026-08-01", "2027-07-01", freq="MS")
    enc, _ = M.oat_ladder(months)
    ops = pd.read_parquet(M.OPS)
    nov = enc[enc["isin"] == "FR0013200813"]["nominal_bn"].sum()
    issued = ops[ops["isin"].astype(str) == "FR0013200813"]["amount_issued"].sum() / 1000
    assert 0 < nov < issued - 1  # buybacks done are netted
    ei = enc[enc["isin"] == "FR0011008705"].iloc[0]  # OAT€i Jul-2027 at indexed value
    assert 1.3 < ei["coef"] < 1.6 and ei["redemption_bn"] == pytest.approx(ei["nominal_bn"] * ei["coef"])
    # OATi lines get their own (French CPI) index ratio, not face
    oati = M.oat_ladder(pd.date_range("2026-10-01", "2032-12-01", freq="MS"))[0]
    oati = oati[oati["kind"] == "oati"]
    assert len(oati) and (oati["coef"] > 1.0).all()


@pytest.mark.skipif(not ENCOURS, reason="AFT outstanding pages not cached")
def test_oat_ladder_splices_already_matured_lines():
    """Cash window covers Sep; encours as-of is after 24 Sep — the 2.5% Sep-26 OAT must still appear."""
    from pathlib import Path

    from sovereign_cash.fr import model as M

    if not Path(M.OPS).exists():
        pytest.skip("ops_fr missing")
    # Encours today starts at Nov-26; Sep-24 matured off the page.
    months = pd.date_range("2026-09-01", "2027-07-01", freq="MS")
    enc, _ = M.oat_ladder(months)
    sep = enc[enc["maturity"] == pd.Timestamp("2026-09-24")]
    assert len(sep) == 1
    assert sep.iloc[0]["isin"] == "FR001400FYQ4"
    assert sep.iloc[0]["nominal_bn"] == pytest.approx(22.965, abs=0.05)
    # Live encours line still present once, not doubled by the ops splice.
    assert len(enc[enc["isin"] == "FR0013200813"]) == 1


@pytest.mark.skipif(not ENCOURS, reason="AFT outstanding pages not cached")
def test_eurosystem_holdings_capped_and_indexed():
    from sovereign_cash.fr import model as M

    enc, _ = M.oat_ladder(pd.date_range("2026-10-01", "2027-07-01", freq="MS"))
    cb = M.cb_holdings(enc)
    assert (cb["cb_face_bn"] <= cb["nominal_bn"] + 1e-9).all()
    assert cb["cb_cash_bn"].to_numpy() == pytest.approx((cb["cb_face_bn"] * cb["coef"]).to_numpy())


def test_btf_supplement_drops_auctions_on_tape(monkeypatch):
    from sovereign_cash.fr import model as M

    months = pd.date_range("2026-09-01", "2026-10-01", freq="MS")
    ops = pd.DataFrame({"sec_type": ["Bill"], "op_dt": [pd.Timestamp("2026-09-28")]})
    js = {"auctions": [{"op_dt": "2026-09-28", "settle_dt": "2026-09-30", "total_mn": 7000.0},
                       {"op_dt": "2026-10-05", "settle_dt": "2026-10-07", "total_mn": 6000.0}]}
    monkeypatch.setattr(F, "load_json", lambda name: js)
    s, used = M.btf_supplement(months, ops)
    assert s[pd.Timestamp("2026-09-01")] == 0.0 and s[pd.Timestamp("2026-10-01")] == pytest.approx(6.0)
    assert [a["op_dt"] for a in used] == ["2026-10-05"]


def test_btf_path_absorbs_need_then_hits_anchor():
    from sovereign_cash.fr import model as M

    months = pd.date_range("2026-10-01", "2026-12-01", freq="MS")
    z = pd.Series(0.0, index=months)
    need = pd.Series([30.0, -10.0, 5.0], index=months)
    # provisional sum 25; YE target net = 191-200 = −9 → uniform −11.333; red large enough not to floor
    gross, red, net = M.btf_path(need, pd.Series(25.0, index=months), {}, z, pd.Series(float("nan"), index=months),
                                 {}, 200.0, {pd.Timestamp("2026-12-01"): 191.0})
    assert net.to_numpy() == pytest.approx([18.666667, -21.333333, -6.333333], abs=1e-5)
    assert 200.0 + float(net.sum()) == pytest.approx(191.0)
    assert (gross >= 0).all()
    assert net.idxmin() == pd.Timestamp("2026-11-01")
    assert net.idxmax() == pd.Timestamp("2026-10-01")


def test_buyback_forward_cuts_next_year():
    from sovereign_cash.fr import model as M

    months = pd.date_range("2026-10-01", "2027-03-01", freq="MS")
    enc = pd.DataFrame({"maturity": pd.to_datetime(["2027-02-25", "2027-05-25"]), "redemption_bn": [30.0, 10.0]})
    enc["month"] = enc["maturity"].dt.to_period("M").dt.to_timestamp()
    bb, cut = M.buyback_forward(months, {m: 22.0 for m in months[:3]}, 1.1, enc)
    assert bb.sum() == pytest.approx(3 * 22.0 * (1 - 1 / 1.1))
    assert cut[pd.Timestamp("2027-02-01")] == pytest.approx(bb[bb.index.year == 2026].sum() * 0.75)


@pytest.mark.skipif(not OUT.exists(), reason="run sovereign_cash/fr/run.py first")
def test_observed_forward_months_use_actuals():
    df = pd.read_parquet(OUT).set_index("month")
    debt = F.load_debt()
    hist = df[~df["is_forecast"]]
    fc = df[df["is_forecast"]]
    obs = fc[fc["financing_observed"]]
    if obs.empty:
        pytest.skip("no forward month has a published BTF stock yet")
    for m in obs.index:
        p = m - pd.offsets.MonthBegin(1)
        assert obs.at[m, "net_bills_bn"] == pytest.approx(debt.at[m, "btf"] - debt.at[p, "btf"], abs=1e-9)
        assert obs.at[m, "bill_stock_implied_bn"] == pytest.approx(debt.at[m, "btf"], abs=1e-9)
    # the implied path starts from the last actual stock
    first = fc.index[0]
    assert fc.at[first, "bill_stock_implied_bn"] == pytest.approx(hist["bill_stock_bn"].iloc[-1] + fc.at[first, "net_bills_bn"])
    assert (fc["bond_redemption_cb_bn"] >= 0).all() and fc["bond_redemption_cb_bn"].sum() > 0
    assert (fc["bill_gross_bn"] >= 0).all()
    # the modelled BTF path lands on the AFT year-end level
    s = json.loads(SUM.read_text())["forward"]
    dec = pd.Timestamp(s["plan_year"], 12, 1)
    if dec in fc.index and not fc.at[dec, "financing_observed"]:
        assert fc.at[dec, "bill_stock_implied_bn"] == pytest.approx(s["plan_btf_stock_end_year_bn"], abs=0.05)
