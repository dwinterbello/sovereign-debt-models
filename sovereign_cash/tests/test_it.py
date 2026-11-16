"""Pure-function tests for the Italian cash framework (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash import contract
from sovereign_cash.it import ladder as lad
from sovereign_cash.it import model as M
from sovereign_cash.it import sources as src


def test_it_number():
    assert src.it_number("(17.067)") == 17067.0
    assert src.it_number("-2.391") == -2391.0
    assert src.it_number("..") == 0.0
    assert src.it_number("(..)") == 0.0
    assert src.it_number("7,9") == pytest.approx(7.9)
    assert np.isnan(src.it_number("...."))


def test_parse_liquidity_csv_and_three_column_pdf():
    csv = ("Mese di riferimento;Saldo dei depositi governativi 1;Operazioni di impiego della liquidità 2;Totale\n"
           "ago-26;4.278;53.056;57.334;;;;\nset-26;4.457;34.756;39.213;;;;\nott-26;;;;;;;\n")
    p = src.parse_liquidity_text(csv)
    assert list(p["month"]) == [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")]
    assert p["total_mn"].tolist() == [57334.0, 39213.0]
    assert p["mm_mn"].iloc[0] == pytest.approx(53056.0)
    pdf = "nov-11 789 16.000 6.417 23.206\ndic-11 6.292 17.000 0 23.292"
    q = src.parse_liquidity_text(pdf)
    assert q["account_mn"].tolist() == [789.0, 6292.0]
    assert q["total_mn"].tolist() == [23206.0, 23292.0]


def test_parse_bdi_table2_rows_and_identity():
    text = "\n".join([
        "2025 .................... -7.607 -9.154 330 113.925 -1 20.304 -2.692 124.259 -14.741 -5.981 109.518",
        "2025 - nov. .......... -3.620 -570 -830 -3.545 900 -3.100 3.636 -6.558 12.791 -1.839 6.233",
        "dic. ........... -20.233 -1.296 -2.708 -10.618 -968 9.731 -4.060 -28.856 12.044 20.520 -16.812",
        "2026 - gen........... (17.067) (-2.391) (-1.035) (3.777) (-481) (..) (-2.465) (16.864) (-9.520) (-17.672) (7.345)",
    ])
    rows = src.parse_bdi_table(text, (10, 11))
    assert [m for m, _ in rows] == [pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")]
    v = dict(zip(src.T2_COLS_11, rows[-1][1]))
    assert v["eu_loans"] == 0.0 and v["fabb_ap"] == 7345.0
    parts = v["deposits"] + v["st_securities"] + v["mlt_securities"] + v["mfi_loans"] + v["eu_loans"] + v["other_liab"]
    assert abs(parts - v["debt_transactions"]) <= 3
    assert v["debt_transactions"] + v["liq_change_neg"] == pytest.approx(v["fabb_ap"], abs=3)


REL_2026 = ("Comunicato Stampa N° 93 del 01/09/2026 Nel mese di agosto 2026 il saldo del settore statale si è chiuso, "
            "in via provvisoria, con un fabbisogno di 12.300 milioni, a fronte di un agosto 2025 che si era chiuso con "
            "un fabbisogno di 204 milioni. Sul sito del Dipartimento.")
REL_2016 = ("Comunicato Stampa N° 141 del 01/08/2016 Nel mese di luglio si è realizzato un avanzo del settore statale "
            "pari, in via provvisoria, a circa 4.400 milioni, che si confronta con un fabbisogno di 2.412 milioni nel "
            "corrispondente mese del 2015. Il fabbisogno dei primi sette mesi dell’anno si attesta sui 22.900 milioni.")
REL_2011 = ("Comunicato Stampa N° 194 del 01/12/2011 A tutto novembre del 2011 si è registrato complessivamente un "
            "fabbisogno del settore statale di circa 69.300 milioni, inferiore di circa 8.600 milioni. Il fabbisogno "
            "mensile di novembre 2011 si è attestato, in via provvisoria, a circa 8.500 milioni.")
REL_2014 = ("Comunicato Stampa N° 57 del 03/03/2014 Rimandato l’incasso premi INAIL a maggio +8% il gettito da IVA. "
            "Nel mese di febbraio 2014 si è realizzato un fabbisogno del settore statale pari, in via provvisoria, a "
            "circa 12.800 milioni, che si confronta con gli 11.845 milioni del mese di febbraio 2013.")
REL_2017 = ("Comunicato Stampa N° 138 del 01/08/2017 Nel mese di luglio 2017 il settore statale ha registrato un avanzo "
            "di 11.200 milioni, con un miglioramento di circa 7.000 milioni.")
REL_TABLE = ("Comunicato Stampa N° 124 del 14/09/2012 Il Ministero comunica i dati di sintesi del conto del settore "
             "statale del mese di luglio 2012. Milioni di euro Formazione del fabbisogno Entrate 47.096 Spese 45.293 "
             "Fabbisogno (-) / Disponibilità (+) 1.803 Copertura Totale -1.803")


def _one(text: str, kind: str = "provisional") -> dict:
    r = [x for x in src.parse_mef_release(text) if x["kind"] == kind]
    assert len(r) == 1, src.parse_mef_release(text)
    return r[0]


def test_parse_mef_release_variants():
    r = _one(REL_2026)
    assert (r["month"], r["fabb_mn"]) == (pd.Timestamp("2026-08-01"), 12300.0)
    p = _one(REL_2026, "prior_year")
    assert (p["month"], p["fabb_mn"]) == (pd.Timestamp("2025-08-01"), 204.0)
    r = _one(REL_2016)
    assert (r["month"], r["fabb_mn"]) == (pd.Timestamp("2016-07-01"), -4400.0)
    assert _one(REL_2016, "prior_year")["fabb_mn"] == 2412.0
    # Year-to-date figure first in the text: the flagged monthly number wins.
    r = _one(REL_2011)
    assert (r["month"], r["fabb_mn"]) == (pd.Timestamp("2011-11-01"), 8500.0)
    # Headline names another month ('a maggio'): the month is February.
    r = _one(REL_2014)
    assert (r["month"], r["fabb_mn"]) == (pd.Timestamp("2014-02-01"), 12800.0)
    assert not [x for x in src.parse_mef_release(REL_2014) if x["kind"] == "prior_year"]
    r = _one(REL_2017)
    assert (r["month"], r["fabb_mn"]) == (pd.Timestamp("2017-07-01"), -11200.0)
    r = _one(REL_TABLE, "final")
    assert (r["month"], r["fabb_mn"]) == (pd.Timestamp("2012-07-01"), -1803.0)


def test_mef_monthly_prefers_final_and_guards_ytd():
    rel = pd.DataFrame([
        {"pub_date": pd.Timestamp("2025-09-01"), "month": pd.Timestamp("2025-08-01"), "fabb_mn": 100.0, "kind": "provisional"},
        {"pub_date": pd.Timestamp("2026-09-01"), "month": pd.Timestamp("2025-08-01"), "fabb_mn": 204.0, "kind": "prior_year"},
        {"pub_date": pd.Timestamp("2025-10-01"), "month": pd.Timestamp("2025-09-01"), "fabb_mn": 25300.0, "kind": "provisional"},
        {"pub_date": pd.Timestamp("2026-10-01"), "month": pd.Timestamp("2025-09-01"), "fabb_mn": 75000.0, "kind": "prior_year"},
    ])
    m = src.mef_monthly(rel)
    assert m.loc["2025-08-01", "fabb_mn"] == 204.0
    assert m.loc["2025-09-01", "fabb_mn"] == 25300.0
    assert m.attrs["dropped_finals"] == 1


def test_parse_maturities_and_composition():
    scad = ("TITOLI IN SCADENZA NEI PROSSIMI 12 MESI (mln. Euro);;;;;;;\nAggiornamento al 31 agosto 2026 *;;;;;;;\n"
            "MESI;BOT;BTP (1);BTPi;BTP Italia;CCTeu;Altre emissioni (2);TOTALE\n"
            "set-26;19.637;15.509;17.553;0;0;0;52.699\nott-26;9.900;0;0;0;0;0;9.900\nTOTALE;1;2;3;4;5;6;7\n")
    s = src.parse_maturities_text(scad)
    assert s.attrs["asof"] == "2026-08-31"
    assert s["total_mn"].tolist() == [52699.0, 9900.0]
    comp = ("COMPOSIZIONE DEI TITOLI DI STATO;;\nin circolazione al  ;28-feb-21;\nTipologia titolo;mln. Euro;%\n"
            "BOT;123.622,52;5,60%\nBOT Flessibili;0;0,00%\nTotale;2.214.327,94;100,00%\n")
    c = src.parse_composition_text(comp)
    assert c["month"] == pd.Timestamp("2021-02-01")
    assert c["bot_stock_mn"] == pytest.approx(123622.52)
    assert c["total_stock_mn"] == pytest.approx(2214327.94)


def _ops(rows):
    cols = ["op_dt", "issue_dt", "maturity_dt", "isin", "sec_type", "op_type", "amount_issued", "price"]
    return pd.DataFrame(rows, columns=cols).assign(
        **{c: lambda d, c=c: pd.to_datetime(d[c]) for c in ("op_dt", "issue_dt", "maturity_dt")})


def test_monthly_ladder_cash_par_buybacks_and_ctz():
    ops = _ops([
        ("2025-01-10", "2025-01-15", "2025-07-15", "IT0000000001", "Bill", "Auction", 1000.0, 98.0),
        ("2025-01-28", "2025-02-01", "2027-01-30", "IT0000000002", "Bill", "Auction", 2000.0, 95.0),  # CTZ
        ("2025-01-12", "2024-09-01", "2030-09-01", "IT0000000003", "Nominal", "Auction", 3000.0, 101.0),  # reopening
        ("2025-03-05", "2025-03-10", "2030-09-01", "IT0000000003", "Nominal", "Buyback", -500.0, 102.0),
        ("2025-02-01", "2025-02-05", "2030-02-05", "US0000000004", "Nominal", "Syndication", 4000.0, 100.0),
    ])
    lm = lad.monthly_ladder(ops)
    assert lm.loc["2025-01-01", "bill_gross_mn"] == pytest.approx(980.0)
    assert lm.loc["2025-07-01", "bill_redemption_mn"] == pytest.approx(1000.0)
    assert lm.loc["2025-02-01", "bond_gross_mn"] == pytest.approx(1900.0)  # CTZ is a bond; USD global dropped
    assert lm.loc["2025-01-01", "bond_gross_mn"] == pytest.approx(3030.0)  # reopening settles after the auction
    assert lm.loc["2025-03-01", "bond_redemption_mn"] == pytest.approx(510.0)  # buyback cash
    assert lm.loc["2030-09-01", "bond_maturing_mn"] == pytest.approx(2500.0)  # net of buyback, at par


def _seasonal(years: int = 12, level: float = 5000.0, trend: float = 0.0) -> pd.Series:
    idx = pd.date_range("2010-01-01", periods=12 * years, freq="MS")
    prof = np.array([2, 10, 20, 8, 9, 6, -12, 3, 18, 10, 5, -14], dtype=float) * 1000.0
    prof -= prof.mean()
    t = np.arange(len(idx))
    return pd.Series(level + trend * t + np.tile(prof, years), index=idx)


def test_forecasts_on_exact_seasonal_series():
    x = _seasonal()
    o = len(x) - 13
    truth = x.iloc[o + 1:o + 13].to_numpy()
    for m in ("smly", "add_1", "add_3", "add_5", "mul_3", "mul_5"):
        assert np.allclose(M.forecast(x, o, m), truth), m
    assert np.allclose(M.forecast(x, o, "flat"), x.iloc[o - 11:o + 1].mean())
    # A linear trend: the level-adjusted profile is exact for h=12, same-month-last-year lags it.
    xt = _seasonal(trend=50.0)
    f = M.forecast(xt, o, "add_3")
    assert abs(f[-1] - xt.iloc[o + 12]) < abs(M.forecast(xt, o, "smly")[-1] - xt.iloc[o + 12])


def test_covid_screen_drops_excluded_inputs_for_every_model():
    clean = _seasonal(16)
    o = clean.index.get_loc(pd.Timestamp("2022-12-01"))  # origin level x[o−11..o] clear of EXCL
    truth = clean.iloc[o + 1:o + 13].to_numpy()
    x = clean.copy()
    x.loc["2020-04-01"] += 50_000.0  # inside EXCL: a k=3 same-month input for h=4
    for m in ("add_3", "add_5", "mul_3", "mul_5"):
        assert np.allclose(M.forecast(x, o, m), truth), m
    # The same shock outside the window does move the forecast: the screen, not the data, removed it.
    y = clean.copy()
    y.loc["2019-04-01"] += 50_000.0
    assert not np.allclose(M.forecast(y, o, "add_5"), truth)
    assert M.in_excl(pd.Timestamp("2020-03-01")) and M.in_excl(pd.Timestamp("2021-02-01"))
    assert not M.in_excl(pd.Timestamp("2020-02-01")) and not M.in_excl(pd.Timestamp("2021-03-01"))


def test_choose_uses_pre_only():
    sc = pd.DataFrame([
        {"window": "PRE", "h": 1, "model": "a", "mae_bn": 1.0, "n": 10},
        {"window": "PRE", "h": 1, "model": "b", "mae_bn": 2.0, "n": 10},
        {"window": "OOS", "h": 1, "model": "a", "mae_bn": 9.0, "n": 10},
        {"window": "OOS", "h": 1, "model": "b", "mae_bn": 0.1, "n": 10},
        {"window": "PRE", "h": 1, "model": "c", "mae_bn": 0.5, "n": 3},  # does not cover every origin
    ])
    assert M.choose(sc) == "a"


def test_backtest_windows_screen_inputs_and_targets():
    x = _seasonal(14)
    bt = M.backtest(x, models=("smly",))
    w = M.window_of(bt)
    scored = bt[w != ""]
    for r in scored.itertuples(index=False):
        lo, hi = r.origin - pd.DateOffset(months=11), r.origin + pd.DateOffset(months=r.h)
        assert not (lo <= M.EXCL[1] and hi >= M.EXCL[0]), (r.origin, r.h)  # inputs and targets clear
    pre = bt[w == "PRE"]
    assert pre["origin"].min() >= pd.Timestamp("2015-01-01") and pre["origin"].max() <= pd.Timestamp("2021-12-01")
    assert (bt.loc[w == "OOS", "origin"] >= pd.Timestamp("2022-01-01")).all()
    # Targets clear, inputs not (smly reads Jul-2020 for an Jun-2021 origin): excluded, not scored.
    cell = bt[(bt["origin"] == pd.Timestamp("2021-06-01")) & (bt["h"] == 1)]
    assert cell["touches_excl"].all() and (w[cell.index] == "").all()
    assert (bt.loc[w == "PRE", "origin"] < M.EXCL[0]).all()
    assert np.allclose(bt["fc"], bt["actual"])  # exact seasonal: smly is perfect


def test_settlement_guard_rejects_far_issue_dates_and_drops_nat_maturity():
    ops = _ops([
        ("2009-09-09", "2025-09-01", "2040-09-01", None, "Nominal", "Syndicate", 6000.0, 98.186),  # bad issue_dt
        ("2025-01-12", "2024-09-01", "2030-09-01", "IT0000000003", "Nominal", "Auction", 3000.0, 101.0),  # reopening
        ("2025-03-05", "2025-03-10", "2032-03-10", "IT0000000005", "Nominal", "Auction", 1000.0, 100.0),
        ("2010-10-20", "2025-10-15", None, None, "FRN", "Auction", 5100.0, 99.789),  # no maturity
    ]).assign(source="tape")
    c = lad.classify(ops)
    assert len(c) == 3
    s = c.set_index(c["op_dt"].dt.strftime("%Y-%m-%d"))["settle"]
    assert s["2009-09-09"] == pd.Timestamp("2009-09-11")  # op + 2 bdays, not 2025
    assert s["2025-01-12"] == pd.Timestamp("2025-01-14")
    assert s["2025-03-05"] == pd.Timestamp("2025-03-10")  # a normal T+3 issue_dt is kept
    lm = lad.monthly_ladder(ops)
    assert lm.loc["2025-01-01":"2025-12-01", "bond_gross_mn"].sum() == pytest.approx(3030.0 + 1000.0)


def test_exchange_legs_cash_neutral_par_at_maturity():
    ops = _ops([
        ("2024-01-17", "2024-01-19", "2034-03-01", "IT0000000007", "Nominal", "Exchange", 4000.0, 102.9),
        ("2024-01-17", "2024-01-19", "2026-06-01", "IT0000000008", "Nominal", "Buyback", -3900.0, 105.0),
        ("2023-11-22", "2023-11-24", "2024-05-30", "IT0000000009", "Nominal", "Exchange", 14093.0, 98.9),
    ]).assign(source="tape")
    lm = lad.monthly_ladder(ops)
    assert lm["bond_gross_mn"].sum() == 0.0 and lm["buyback_mn"].sum() == 0.0  # no cash at settlement
    assert lm.loc["2034-03-01", "bond_maturing_mn"] == pytest.approx(4000.0)
    assert lm.loc["2026-06-01", "bond_maturing_mn"] == pytest.approx(-3900.0)
    assert "2024-05-01" not in lm.index or lm.loc["2024-05-01", "bond_maturing_mn"] == 0.0  # > cap: tape error


def test_known_events_overlay_adds_once_and_feeds_forward():
    from sovereign_cash.it import events as ev

    base = _ops([
        ("2026-09-10", "2026-09-15", "2033-09-15", "IT0005722845", "Nominal", "Auction", 3500.0, 96.41),
    ]).assign(source="MEF auctions_combined.xlsx", country="IT")
    sup = ev.supplement(base)
    added = sup[sup["source"] == ev.KNOWN_EVENT]
    assert len(added) == len(ev.BOND_EVENTS)
    green = added[added["maturity_dt"] == pd.Timestamp("2038-10-30")].iloc[0]
    assert green["amount_issued"] == 8000.0 and green["price"] == 99.591
    assert green["issue_dt"] == pd.Timestamp("2026-09-29")
    assert len(ev.supplement(sup)) == len(sup)  # idempotent
    # Once the tape carries the Green line, the overlay steps aside.
    tape_green = _ops([("2026-09-22", "2026-09-29", "2038-10-30", "IT0005999999", "Nominal", "Syndication",
                        8000.0, 99.591)]).assign(source="MEF comunicato", country="IT")
    sup2 = ev.supplement(pd.concat([base, tape_green], ignore_index=True))
    assert (sup2["maturity_dt"] == pd.Timestamp("2038-10-30")).sum() == 1
    valore = added[added["maturity_dt"] == pd.Timestamp("2031-10-27")].iloc[0]
    assert valore["amount_issued"] == pytest.approx(ev.VALORE_FALLBACK_MN)  # assumption when tape lacks them
    months = pd.date_range("2026-09-01", periods=12, freq="MS")
    fb = lad.forward_bonds(pd.Timestamp("2026-08-01"), months, sup, fwd_path=lad.REPO / "no_such.parquet")
    assert fb["2026-09-01"] == pytest.approx(3500 * 0.9641 + 8000 * 0.99591)
    assert fb.attrs["tape_last_op"] == "2026-09-10"  # events never move the forward.py cut-off
    of = ev.other_fin_bn(months)
    assert of["2026-12-01"] == pytest.approx(28.4 * (1 - ev.rrf_grant_share()))  # loan part only
    assert of.drop(pd.Timestamp("2026-12-01")).eq(0).all()


def test_known_event_already_in_the_forward_counts_once(tmp_path):
    from sovereign_cash.it import events as ev

    base = _ops([
        ("2026-09-10", "2026-09-15", "2033-09-15", "IT0005722845", "Nominal", "Auction", 3500.0, 96.41),
    ]).assign(source="MEF auctions_combined.xlsx", country="IT")
    sup = ev.supplement(base)
    valore = sup[sup["maturity_dt"] == pd.Timestamp("2031-10-27")].iloc[0]
    fwd = pd.DataFrame({"country": ["IT", "IT"], "op_dt": pd.to_datetime(["2026-10-23", "2026-10-13"]),
                        "amount_mn": [16_000.0, 4_000.0], "maturity_dt": pd.to_datetime(["2031-10-27", "2029-10-01"]),
                        "isin": ["IT BTP Valore 2031", "IT0005000001"]})
    path = tmp_path / "fwd.parquet"
    fwd.to_parquet(path)
    months = pd.date_range("2026-09-01", periods=12, freq="MS")
    fb = lad.forward_bonds(pd.Timestamp("2026-08-01"), months, sup, fwd_path=path)
    # October = the forward's Valore and auction, not the known-event Valore on top
    assert fb["2026-10-01"] == pytest.approx(16_000.0 + 4_000.0)
    assert valore["amount_issued"] > 0
    assert fb.attrs["known_events_mn"]["2026-10-01"] == 0.0


def test_mef_history_uses_previous_month_vintage_ex_altre():
    v = pd.DataFrame([
        {"month": pd.Timestamp("2026-07-01"), "vintage": pd.Timestamp("2026-06-30"), "bot_mn": 25000.0,
         "total_mn": 41225.0, "other_mn": 0.0},
        {"month": pd.Timestamp("2026-08-01"), "vintage": pd.Timestamp("2026-06-30"), "bot_mn": 1.0,
         "total_mn": 2.0, "other_mn": 0.0},  # not the first month of its vintage
        {"month": pd.Timestamp("2026-08-01"), "vintage": pd.Timestamp("2026-07-31"), "bot_mn": 8800.0,
         "total_mn": 39632.0, "other_mn": 2000.0},
    ])
    h = lad.mef_history(v)
    assert h.loc["2026-07-01", "bond_mef_mn"] == pytest.approx(16225.0)
    assert h.loc["2026-08-01", "bond_mef_mn"] == pytest.approx(28832.0)
    assert h.loc["2026-08-01", "other_mef_mn"] == pytest.approx(2000.0)


def test_hold_bills_keeps_stock_on_target_and_lets_liquidity_absorb():
    flows = np.array([0.0, 0.0, 40.0, 0.0])
    red0 = np.array([10.0, 10.0, 5.0, 10.0])
    target = np.array([1.0, -1.0, 0.0, 0.0])
    net, gross, red = M.hold_bills(50.0, flows, 30.0, red0, {1: 1.0}, target)
    assert np.allclose(net, target)  # no steering: the surplus stays in liquidity
    assert (gross >= 0).all() and np.allclose(gross - red, net)
    assert np.allclose(red[1:], red0[1:] + gross[:-1])  # 1-month bills roll into next month
    lvl = 50.0 + np.cumsum(flows + net)
    assert lvl.max() > 70.0
    # A gap below the floor is filled with BOT; pinned months keep their settled gross.
    net2, gross2, _ = M.hold_bills(35.0, np.array([-20.0, 0.0]), 30.0, np.array([10.0, 10.0]), {1: 1.0},
                                   np.zeros(2), np.array([np.nan, np.nan]))
    assert 35.0 - 20.0 + net2[0] == pytest.approx(30.0)
    net3, gross3, _ = M.hold_bills(35.0, np.array([-20.0, 0.0]), 30.0, np.array([10.0, 10.0]), {1: 1.0},
                                   np.zeros(2), np.array([4.0, np.nan]))
    assert gross3[0] == 4.0 and net3[0] == -6.0


def test_bot_seasonal_net_is_demeaned_calendar_pattern():
    idx = pd.date_range("2019-01-01", "2026-08-01", freq="MS")
    step = np.where(idx.month == 4, 3.0, 0.0) + 0.5  # +3 every April on top of a drift
    stock = pd.Series(100.0 + np.cumsum(step), index=idx)
    seas = M.bot_seasonal_net(stock, pd.Timestamp("2026-08-01"))
    assert seas.sum() == pytest.approx(0.0, abs=1e-9)
    assert seas[4] == pytest.approx(3.0 - 3.0 / 12) and seas[5] == pytest.approx(-3.0 / 12)


def test_rrf_grants_stripped_from_base_and_added_back():
    from sovereign_cash.it import events as ev

    idx = pd.date_range("2025-01-01", "2026-08-01", freq="MS")
    eu = pd.Series(0.0, index=idx)
    eu["2025-12-01"], eu["2025-11-01"] = 9.0, -3.0  # an instalment's loan part; a repayment is not an instalment
    g = ev.rrf_grants_history(eu)
    assert g["2025-12-01"] == pytest.approx(9.0 * ev.RRF_GRANTS_BN / ev.RRF_LOANS_BN)
    assert g.drop(pd.Timestamp("2025-12-01")).eq(0).all()
    months = pd.date_range("2026-09-01", periods=12, freq="MS")
    split = ev.rrf_split(months)
    tot = sum(e["amount_bn"] for e in ev.RRF_EVENTS if pd.Timestamp(e["month"]) in months)
    assert (split["grant_bn"] + split["loan_bn"]).sum() == pytest.approx(tot)
    assert split["grant_bn"].sum() == pytest.approx(tot * ev.rrf_grant_share())
    assert ev.other_fin_bn(months).sum() == pytest.approx(split["loan_bn"].sum())


def test_zero_model_and_drift_choice():
    idx = pd.date_range("2012-01-01", "2026-08-01", freq="MS")
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0.0, 5000.0, len(idx)), index=idx)  # pure noise: zero should win
    assert np.allclose(M.forecast(y, len(y) - 1, "zero"), 0.0)
    sc = M.score(M.backtest(y, models=("zero", "flat", "smly")))
    assert M.choose(sc) == "zero"


def test_forward_issues_tagged_announced_vs_assumed(tmp_path):
    from sovereign_cash.it import events as ev

    base = _ops([
        ("2026-03-06", "2026-03-10", "2032-03-10", "IT0005696338", "Nominal", "Syndication", 16222.584, 100.0),
        ("2026-09-22", "2026-09-29", "2038-10-30", "IT0005733099", "Nominal", "Syndication", 8000.0, 99.591),
    ]).assign(source=["MEF BTP Valore Mar-2026", "MEF comunicato BTP Green syndication"], country="IT")
    sup = ev.supplement(base)
    fwd = pd.DataFrame({"country": ["IT"] * 4,
                        "op_dt": pd.to_datetime(["2026-10-23", "2026-10-13", "2027-03-05", "2027-01-07"]),
                        "amount_mn": [16_000.0, 4_000.0, 16222.584, 15_000.0],
                        "maturity_dt": pd.to_datetime(["2031-10-27", "2029-10-01", "2033-03-09", "2034-03-14"]),
                        "isin": ["IT BTP Valore 2031", "IT0005000001", "next synd 3-7y", "next synd 7-10y"],
                        "source": ["announced retail", "pattern", "syndication pattern", "syndication pattern"],
                        "sec_type": ["Nominal"] * 4})
    path = tmp_path / "fwd.parquet"
    fwd.to_parquet(path)
    months = pd.date_range("2026-09-01", periods=12, freq="MS")
    fb = lad.forward_bonds(pd.Timestamp("2026-08-01"), months, sup, fwd_path=path)
    it = fb.attrs["items"].set_index("id")
    assert it.loc["IT0005733099", "basis"] == "announced" and "BTP Green" in it.loc["IT0005733099", "label"]
    assert it.loc["IT BTP Valore 2031", "basis"] == "announced"
    assert it.loc["next synd 3-7y", "basis"] == "assumed" and it.loc["next synd 3-7y", "kind"] == "retail"
    assert it.loc["next synd 7-10y", "kind"] == "syndication"
    assert fb.attrs["announced_mn"]["2026-10-01"] == pytest.approx(16_000.0)
    assert fb.attrs["assumed_mn"].sum() == pytest.approx(16222.584 + 15_000.0)
    assert "IT0005000001" not in it.index  # regular auctions are not lumps


def test_anchor_to_plan_hits_annual_totals():
    idx = pd.date_range("2025-01-01", "2026-08-01", freq="MS")
    x = pd.Series(10.0, index=idx)
    fc = pd.Series(np.arange(16, dtype=float), index=pd.date_range("2026-09-01", periods=16, freq="MS"))
    out = M.anchor_to_plan(fc, x, pd.Timestamp("2026-08-01"), {2026: 120.0, 2027: 100.0})
    assert out["2026"].sum() + 80.0 == pytest.approx(120.0)
    assert out["2027"].sum() == pytest.approx(100.0)
    assert np.allclose(np.diff(out["2027"].to_numpy()), 1.0)  # shape kept


def test_contract_identity_on_toy_frame():
    df = pd.DataFrame({
        "month": pd.date_range("2025-01-01", periods=3, freq="MS"), "is_forecast": [False, False, True],
        "cash_requirement_bn": [10.0, -5.0, 3.0], "bond_gross_bn": [20.0, 10.0, 5.0],
        "bond_redemption_bn": [5.0, 15.0, 0.0], "net_bonds_bn": [15.0, -5.0, 5.0],
        "bill_gross_bn": [8.0, 8.0, 8.0], "bill_redemption_bn": [7.0, 9.0, 8.0], "net_bills_bn": [1.0, -1.0, 0.0],
        "other_fin_bn": [0.5, 0.0, 0.0],
    })
    df["buffer_change_bn"] = -df["cash_requirement_bn"] + df["net_bonds_bn"] + df["net_bills_bn"] + df["other_fin_bn"]
    assert contract.validate(df) == pytest.approx(0.0)
