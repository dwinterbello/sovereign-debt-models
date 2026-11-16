"""Cyclical line model (scenarios.cyclical) and the US DTS one-off adjustments (channels.us_dts_adjust)."""
from __future__ import annotations

import pytest

from sovereign_cash.scenarios import cyclical as C

NON_US = ("uk", "de", "fr", "it", "es", "jp")


@pytest.mark.parametrize("cc", NON_US)
def test_semi_elasticity_matches_oecd(cc):
    # the model's balance response to a 1% real gap, with today's revenue and spending ratios, against OECD Table 10
    assert abs(C.balance_semi_elasticity(cc, 2028) - C.OECD[cc]["semi"]) < 0.1


def test_downturn_raises_every_deficit_and_upturn_lowers_it():
    dn = C.table(size_pp=-2.0).groupby(["cc", "year"]).pct_gdp.sum()
    up = C.table(size_pp=2.0).groupby(["cc", "year"]).pct_gdp.sum()
    assert (dn > 0).all() and (up < 0).all()


def test_us_lines_in_range_of_oecd_and_history():
    # canonical recession, FY2028: real gap ~3%. OECD US semi-elasticity 0.5 -> ~1.5pp; 2001-02 receipts fell ~2.4pp
    t = C.table(size_pp=-2.0)
    us = t[(t.cc == "us") & (t.year == 2028)].pct_gdp.sum()
    assert 1.0 < us < 3.0
    eq = t[(t.cc == "us") & (t.year == 2028) & (t.line == "dep_nonwithheld")].pct_gdp.sum()
    assert eq > 0.3  # equities are the largest US swing line


def test_stimulus_only_in_downturns():
    up = C.lines("fr", 2.0, [2027, 2028], 2026, stimulus=True)
    assert all("stimulus" not in d for d in up.values())
    dn = C.lines("fr", -2.0, [2027, 2028], 2026, stimulus=True)
    assert dn[2028]["stimulus"] > 0


def test_us_one_offs_out_of_the_base():
    from sovereign_cash.scenarios.channels import us_dts as U

    raw, adj = U.fy_table(adjusted=False), U.fy_table()
    fc = [y for y in adj.index if y >= 2027]
    if not fc:
        pytest.skip("no forecast fiscal year in the DTS path")
    y = fc[0]
    assert raw.at[y, "wd_other"] - adj.at[y, "wd_other"] > 100.0     # tariff refunds do not recur
    assert adj.at[y, "wd_refunds"] < raw.at[y, "wd_refunds"]         # refund catch-up does not recur
    assert adj.at[y, "receipts"] == pytest.approx(
        adj.loc[y, [c for c in adj.columns if c.startswith("dep_") and c != "dep_total"]].sum())


def test_us_tax_overlay_not_double_counted():
    from sovereign_cash.scenarios import primary as P

    t = P.path("us")
    dts = t[t["rule"] == "dts_fy"]
    assert (dts["policy_tax"] == 0.0).all()


def test_uk_obr_reckoner_close_to_oecd_total():
    # line-level OBR ready reckoner against the OECD aggregate for the same downturn, 2028
    t = C.table(size_pp=-2.0)
    obr = t[(t.cc == "uk") & (t.year == 2028)].pct_gdp.sum()
    real = C.REAL_SHARE * 0.04
    oecd = 100 * real * C.OECD["uk"]["semi"]  # balance-ratio semi-elasticity x real gap, before the price effect
    assert abs(obr - oecd) < 1.0
    assert t[(t.cc == "uk") & (t.year == 2029) & (t.line == "rec:uk_asset")].pct_gdp.sum() > 0.2  # CGT lags equities


def test_fr_corporate_tax_lags_a_year():
    L = C.lines("fr", -2.0, [2027, 2028], 2026)
    oe = C.OECD["fr"]["cit"]
    full_2027 = C._gg("fr", 2027)["rev"] * oe[1] / 100 * C.CORP_PROFIT_ELAST * C.profit_path(1, -2.0)
    assert abs(L[2027]["cit"]) < abs(full_2027)          # only part lands in the shock year
    assert abs(L[2028]["cit"]) > abs(L[2027]["cit"])      # the solde and the new acomptes land next year


def test_rates_move_rate_linked_receipts_only_where_modelled():
    up = C.response_all("uk", [2027, 2028], 2026, r_bp=75.0)
    assert up[2028]["d_rev"] > 0                       # interest receipts and tax on savings income rise
    us = C.response_all("us", [2027, 2028], 2026, r_bp=75.0)
    assert us[2028]["d_rev"] == 0.0                    # the Fed remits nothing while it carries a deferred asset
    fr = C.response_all("fr", [2027, 2028], 2026, r_bp=75.0)
    assert fr[2028] == {"d_rev": 0.0, "d_spend": 0.0}
    jp = C.response_all("jp", [2027, 2028], 2026, r_bp=75.0)
    assert jp[2027]["d_rev"] == 0.0                    # FY2027 profits are paid in May 2028
    assert -C.BOJ_REMIT_BASE_BN <= jp[2028]["d_rev"] < 0   # the BoJ's payment falls, floored at zero
    jp_dn = C.response_all("jp", [2028], 2026, r_bp=-75.0)
    assert jp_dn[2028]["d_rev"] > -jp[2028]["d_rev"]  # a cut is not floored


def test_us_price_leg_matches_cbo_rule_of_thumb():
    # CBO: inflation 1pp a year higher, sustained. Our price leg on the same shock: receipts in the year, spending a year
    # later. Compare the primary improvement in years 2-4 with CBO's receipts-minus-outlays (scenario pack, per pp).
    import json
    pack = json.loads((C.DERIVED / "scenario_pack.json").read_text())
    ch = {c["id"]: c for c in pack["countries"]["us"]["channels"]}
    cbo = [r - o for r, o in zip(ch["us_cpi_receipts"]["per_unit"]["growth"], ch["us_cpi_primary"]["per_unit"]["growth"])]
    years = [2027, 2028, 2029, 2030]
    P = C._price("us", years, 2026, lambda k: max(k, 0) / 100.0)
    ours = [sum(v if C.is_receipt(k) else -v for k, v in P[y].items()) for y in years]
    for a, b in zip(ours[1:], cbo[1:]):
        assert 0.5 < a / b < 2.0


def test_reflation_moves_the_primary_on_the_same_lines():
    r = C.response_all("it", [2027, 2028, 2029, 2030], 2026, infl_pp=2.0)
    assert r[2027]["d_rev"] > 0 and r[2027]["d_spend"] == 0.0     # receipts in the year, spending a year later
    assert r[2028]["d_spend"] > 0


# ---------------------------------------------------------------- properties of the one model (stress test, 13 Nov)
YRS = [2027, 2028, 2029, 2030]
CCS = ("us", "uk", "de", "fr", "it", "es", "jp")


def test_zero_shock_is_exactly_zero():
    for cc in CCS:
        r = C.response_all(cc, YRS, 2026)
        assert all(v == {"d_rev": 0.0, "d_spend": 0.0} for v in r.values())


def test_growth_leg_is_linear_and_antisymmetric_outside_the_us():
    # Since 0339d5bc5 corporate tax is stressed in a downturn only (CIT_STRESS; losses carried forward absorb an
    # upturn), so it is antisymmetric up to that factor; every other line stays exactly antisymmetric.
    for cc in ("de", "fr", "it", "es", "jp"):
        dn, up = C.lines(cc, -2.0, YRS, 2026), C.lines(cc, 2.0, YRS, 2026)
        for y in YRS:
            assert set(dn[y]) == set(up[y])
            for k in dn[y]:
                a, b = dn[y][k], up[y][k]
                f = C.CIT_STRESS.get(y - 2026, 1.0) if k == "cit" else 1.0
                assert abs(a + f * b) <= 1e-9 * max(1.0, abs(a)), (cc, y, k)


def test_spain_unemployment_weight_is_rebased_to_today():
    assert C.cp_elasticity("es") > C.OECD["es"]["cp"][0]          # today's smaller unemployment share
    assert abs(C.balance_semi_elasticity("es", 2027) - 0.63) < 0.06  # back to OECD / EC semi-elasticity


def test_corporate_tax_moves_on_the_oecd_gap_elasticity_with_downturn_stress():
    # 0339d5bc5 moved DE/FR/IT/ES/JP corporate tax off the common profit path (now US and UK only) onto the OECD
    # output-gap elasticity, today's CIT revenue share, CIT_STRESS in a downturn and the advance-payment timing.
    L = C.lines("de", -2.0, YRS, 2026)
    g = C._gg("de", 2028)
    same = C.CIT_SAME_YEAR["de"]
    real_cit = C.REAL_SHARE * (same * C.gap(2, -2.0) + (1 - same) * C.gap(1, -2.0))
    want = g["rev"] * C.cit_share("de") / 100 * C.OECD["de"]["cit"][0] * C.CIT_STRESS[2] * real_cit
    assert abs(L[2028]["cit"] - want) < 1e-9
    assert "px:receipts" in L[2028]                                  # price part on the other receipts only


def test_uk_price_leg_uprates_departmental_spending():
    P = C.price_lines("uk", 2.0, YRS, 2026)
    assert P[2028].get("px:spd:uk_del", 0.0) == 0.0 and P[2029]["px:spd:uk_del"] > 0  # cash plans to 2028-29


def test_us_cache_follows_the_adjustment_switch():
    from sovereign_cash.scenarios.channels import us_dts as U
    from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ
    a = float(U.fy_table().loc[2027, "wd_other"])
    try:
        ADJ.ENABLED = False
        b = float(U.fy_table().loc[2027, "wd_other"])
    finally:
        ADJ.ENABLED = True
    assert b > a + 100                                  # the tariff refunds come back without the adjustment
    assert float(U.fy_table().loc[2027, "wd_other"]) == a


def test_profits_rejoin_nominal_gdp_from_year_four():
    assert C.profit_path(4, -2.0) == C.gap(4, -2.0) and C.profit_path(3, -2.0) < C.gap(3, -2.0)


def test_uk_bank_rate_interest_is_kept_once():
    r = C.rates_interest("uk", 75.0, [2027, 2028], 2026)
    assert r[2027] > 3.0                                                     # reserves at Bank Rate: ~GBP5bn less bills
    assert all(not k.startswith("int:") for d in C.rates_lines("uk", 75.0, [2027], 2026).values() for k in d)
    assert C.rates_interest("fr", 75.0, [2027], 2026)[2027] == 0.0


def test_boj_payment_is_bounded_both_ways():
    up = C.response_all("jp", [2028], 2026, r_bp=200.0)[2028]["d_rev"]
    dn = C.response_all("jp", [2028], 2026, r_bp=-200.0)[2028]["d_rev"]
    assert up == -C.BOJ_REMIT_BASE_BN and dn == C.BOJ_REMIT_UP_CAP_BN
