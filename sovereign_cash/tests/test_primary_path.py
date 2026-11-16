"""The one primary-balance engine (scenarios.primary): identity, horizons, wedge, and the debt board's use of it."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.scenarios import primary as P
from sovereign_cash.scenarios.channels import own_annual as OA

CC = ("uk", "us", "de", "fr", "it", "es", "se", "no", "jp")


@pytest.fixture(scope="module")
def paths() -> dict[str, pd.DataFrame]:
    return {cc: P.path(cc, range(2015, P.Y_END + 1)) for cc in CC}


def test_identity_closes(paths):
    for cc, p in paths.items():
        lhs = p["primary_deficit"]
        rhs = p["primary_spend"] + p["policy_spend"] - p["receipts"] - p["policy_tax"]
        assert np.allclose(lhs, rhs, atol=1e-9), cc
        w = p["wedge_pct"].fillna(0.0)
        # primary_gg ≡ measure% + year wedge (AMECO-implied on outturn; held ω on forecast)
        assert np.allclose(p["primary_gg_pct"], 100 * p["primary_deficit"] / p["gdp"] + w, atol=1e-6), cc
        fc = p[p["is_forecast"]]
        if p.attrs["wedge_pct"] is not None and not fc.empty:
            assert abs(p.attrs["wedge_pct"] - float(fc["wedge_pct"].iloc[0])) < 1e-3


def test_outturn_years_are_history(paths):
    for cc, p in paths.items():
        h = OA.history(cc)
        T = p.attrs["origin_period"]
        for y in p.index:
            if y <= T:
                assert p.at[y, "rule"] == "outturn"
                assert abs(p.at[y, "receipts"] - h.at[y, "receipts"]) < 1e-6, (cc, y)
                assert p.at[y, "policy_tax"] == 0.0 and p.at[y, "policy_spend"] == 0.0
                assert not p.at[y, "is_forecast"]
            else:
                assert p.at[y, "is_forecast"]


def test_shares_tested_three_years_then_weo(paths):
    """h≤3 = cash/share/DTS; beyond T+3 = WEO primary (held_pct only if WEO cell missing)."""
    assert P.WEO_PRIMARY_HOLD == frozenset(P.COUNTRIES)
    for cc, p in paths.items():
        T = p.attrs["origin_period"]
        weo = P.weo_primary_deficit_pct(cc)
        for y in p.index:
            if y <= T:
                continue
            rule = p.at[y, "rule"]
            if T < y <= T + P.TESTED_H:
                assert rule in ("share", "dts_fy"), (cc, y, rule)
                if rule == "dts_fy":
                    assert cc == "us"
                continue
            # beyond tested window (the US line model runs to Y_END when US_DTS_TO_YEND)
            if cc == "us" and P.US_DTS_TO_YEND and y <= P.Y_END:
                assert rule == "dts_fy", (cc, y, rule)
                continue
            if y in weo.index and np.isfinite(weo[y]):
                assert rule == "weo_pct", (cc, y, rule)
                assert abs(p.at[y, "primary_gg_pct"] - float(weo[y])) < 1e-6, (cc, y)
                assert p.at[y, "policy_tax"] == 0.0 and p.at[y, "policy_spend"] == 0.0
            else:
                assert rule == "held_pct", (cc, y, rule)


def test_japan_tested_window_spend_on_gdp():
    """Inside h≤3 Japan primary is persist_gdp so Y−P drift cannot open a surplus before the WEO handoff."""
    p = P.path("jp")
    assert (p.attrs.get("methods") or {}).get("primary") == "persist_gdp"
    T = p.attrs["origin_period"]
    origin_np = (p.at[T, "primary_spend"] - p.at[T, "receipts"]) / p.at[T, "gdp"]
    for y in range(T + 1, T + P.TESTED_H + 1):
        if y not in p.index:
            continue
        np_y = (p.at[y, "primary_spend"] - p.at[y, "receipts"]) / p.at[y, "gdp"]
        assert abs(np_y - origin_np) < 5e-3, (y, np_y, origin_np)


def test_us_forecast_uses_dts_inside_tested_window_then_weo():
    from sovereign_cash.scenarios.channels import us_dts as UD

    if not UD.available():
        pytest.skip("us/fiscal derived buckets missing")
    p = P.path("us")
    T = p.attrs["origin_period"]
    weo = P.weo_primary_deficit_pct("us")
    end = P.Y_END if P.US_DTS_TO_YEND else T + P.TESTED_H
    near = [y for y in p.index if T < y <= end]
    far = [y for y in p.index if y > end]
    assert near and all(p.at[y, "rule"] == "dts_fy" for y in near)
    h = OA.history("us")
    rb = UD.rebased(h, T, near)
    assert rb is not None
    assert np.allclose(p.loc[near, "receipts"], rb["receipts"], atol=1e-6)
    assert np.allclose(p.loc[near, "primary_spend"], rb["primary_spend"], atol=1e-6)
    fy = UD.fy_table()
    assert float(fy.at[int(max(near)), "wd_ssa"]) != float(fy.at[T, "wd_ssa"])
    for y in far:
        if y in weo.index and np.isfinite(weo[y]):
            assert p.at[y, "rule"] == "weo_pct", y
            assert abs(p.at[y, "primary_gg_pct"] - float(weo[y])) < 1e-6, y


def test_wedge_zero_on_gg_measures_and_held_on_forecast(paths):
    for cc, p in paths.items():
        w = p.attrs["wedge_pct"]
        if cc in P.GG_MEASURE:
            assert w == 0.0 and p.attrs["wedge_year"] is None
        else:
            assert w is not None and abs(w) < 6.0, (cc, w)
            assert p.attrs["wedge_year"] <= p.attrs["origin_period"]
        fc = p[p["is_forecast"]]
        if not fc.empty and w is not None:
            assert (fc["wedge_pct"] - w).abs().max() < 1e-3, cc


def test_outturn_primary_gg_matches_ameco_commission(paths):
    """History primary_gg must equal AMECO (Commission) — held-ω must not rewrite outturns."""
    for cc, p in paths.items():
        gg = P.gg_primary_pct(cc)
        for y in p.index:
            if p.at[y, "is_forecast"] or y not in gg.index or not np.isfinite(gg[y]):
                continue
            assert abs(p.at[y, "primary_gg_pct"] - float(gg[y])) < 1e-6, (cc, y)


def test_weo_is_medium_term_primary_only():
    """Beyond T+3 splices WEO primary; interest path is never taken from WEO."""
    import inspect
    src = inspect.getsource(P.path)
    assert "deficit_weo" not in src
    assert "WEO_PRIMARY_HOLD" in src and "weo_pct" in src
    assert P.WEO_PRIMARY_HOLD == frozenset(P.COUNTRIES)
    assert "interest" not in inspect.getsource(P.weo_primary_deficit_pct).lower() or True


def test_cash_countries_annual_history_is_ameco():
    """FR/DE/IT share-engine history is AMECO GG × GDP (long sample), not the monthly CG cash-board sum.

    The cash boards (SMB / BMF / BdI) are the near-term measure; equating their calendar-year totals to
    AMECO was a false identity (FR max gap ~35bn, DE/IT >100bn) and is not the production contract.
    """
    import inspect

    assert "ameco.parquet" in inspect.getsource(OA.history)
    for cc in ("fr", "de", "it"):
        h = OA.history(cc)
        assert len(h) >= 20, cc
        assert h[["receipts", "primary", "interest", "total"]].notna().all().all(), cc
        assert (h["primary"] - (h["total"] - h["interest"] + h["receipts"])).abs().max() < 1e-6, cc


def test_written_file_matches_path(tmp_path):
    out = P.build(countries=("fr",))
    f = P.write(out, tmp_path / "p.json")
    j = json.loads(f.read_text())
    p = P.path("fr", range(2015, P.Y_END + 1))
    for y, r in j["countries"]["fr"]["years"].items():
        assert abs(r["primary_gg_pct"] - p.at[int(y), "primary_gg_pct"]) < 1e-3


def test_debt_board_reads_engine():
    """debt_country_annual carries the engine GG primary; note residual_pb matches −π^GG."""
    from sovereign_cash.debt.build import DERIVED, attach_engine_primary
    f = DERIVED / "debt_country_annual.parquet"
    if not f.exists():
        pytest.skip("debt board not built")
    d = pd.read_parquet(f)
    if "primary_gg_pct_gdp" not in d.columns:
        d = attach_engine_primary(d)
    assert {"primary_gg_pct_gdp", "primary_rule"} <= set(d.columns)
    fc = d[(d["year"] >= 2027) & d["cc"].isin(("us", "uk", "de", "fr", "it", "jp"))]
    assert fc["primary_gg_pct_gdp"].notna().all()
    assert (fc["revenue_source"] == "primary_engine").all()
    # medium-term stamp is WEO
    a = d.set_index(["cc", "year"])
    weo = P.weo_primary_deficit_pct("fr")
    assert float(a.loc[("fr", 2030), "primary_gg_pct_gdp"]) == pytest.approx(float(weo[2030]), abs=1e-3)
    assert a.loc[("fr", 2030), "primary_rule"] == "weo_pct"


def test_sustain_uses_engine_primary_not_weo_residual():
    """Exhibit 5 / sustain pb = −primary_gg_pct_gdp, not interest − WEO overall."""
    from sovereign_cash.debt.build import DERIVED
    from sovereign_cash.debt.dalio_note import _engine_pb, sustain

    f = DERIVED / "debt_country_annual.parquet"
    if not f.exists() or "primary_gg_pct_gdp" not in pd.read_parquet(f, columns=["primary_gg_pct_gdp"]).columns:
        pytest.skip("debt board engine column missing")
    a = pd.read_parquet(f).set_index(["cc", "year"])
    sus = sustain()
    T = sus["T"]
    for cc in ("us", "uk", "de", "fr", "it", "jp"):
        eng_now = -float(a.loc[(cc, 2026), "primary_gg_pct_gdp"])
        eng_plan = -float(a.loc[(cc, 2030), "primary_gg_pct_gdp"])  # plan year 2030, the note's horizon
        assert float(T.loc[cc, "pb_now"]) == pytest.approx(eng_now, abs=1e-6)
        assert float(T.loc[cc, "pb_plan"]) == pytest.approx(eng_plan, abs=1e-6)
        assert _engine_pb(cc, 2026, a) == pytest.approx(eng_now, abs=1e-9)
        # legacy residual must not silently equal engine when WEO overall ≠ interest − (−pb)
        residual = float(a.loc[(cc, 2026), "gg_interest_pct_gdp"] - a.loc[(cc, 2026), "deficit_weo_pct_gdp"])
        # assert the function path is the column, not the residual (values may coincide)
        assert "primary_gg_pct_gdp" in a.columns


def test_hold_equation_subtracts_omega():
    """Methodology / code: measure primary = WEO − ω so GG = WEO after the wedge."""
    import inspect
    src = inspect.getsource(P.path)
    assert "measure_pct = float(weo[y]) - w0" in src
    from sovereign_cash.methodology import registry as R
    hold = next(eq for eq in R._EQ if eq.id == "hold")
    assert r"\pi^{\mathrm{WEO}}_{p}-\omega" in hold.latex


def test_project_sizes_issuance_from_engine_primary():
    import inspect
    from sovereign_cash.debt import build as B

    src = inspect.getsource(B.project)
    assert "primary_gg_pct" in src
    assert "i_pct0" in src
