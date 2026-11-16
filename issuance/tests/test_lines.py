import pandas as pd

from issuance.forward import project
from issuance.lines import _paint_bucket


def _pin_sep_remit(monkeypatch):
    """These cases are the 30 Sep origin; pin the remit report of that date (29 Sep), not the live one."""
    from issuance import uk_remit
    monkeypatch.setattr(uk_remit, "remit_progress", lambda *a, **k: dict(uk_remit.FALLBACK))


def test_uk_forward_is_the_d5d_calendar(monkeypatch):
    _pin_sep_remit(monkeypatch)
    from issuance.forward import UK_CALENDAR
    from issuance.uk_remit import remit_progress

    ops = pd.DataFrame({
        "op_dt": [pd.Timestamp("2020-01-01")],
        "amount_issued": [1.0],
        "resid_mat_yrs": [5.0],
        "sec_type": ["Nominal"],
        "isin": ["X"],
    })
    asof = pd.Timestamp("2026-09-30")
    paint = project("UK", ops, pd.DataFrame(), asof)
    # Auctions on or before the remit report date are already in its sales to date.
    cut = max(asof, remit_progress()["asof"])
    named = [row for row in UK_CALENDAR if pd.Timestamp(row[0]) > cut and row[1] and row[3] is not None]
    cal = paint[paint["source"].eq("calendar")]
    assert len(cal) == len(named)
    assert set(zip(pd.to_datetime(cal["op_dt"]).dt.strftime("%Y-%m-%d"), cal["isin"])) == {(row[0], row[1]) for row in named}
    nov = paint[paint["op_dt"].eq("2026-11-10")]
    assert list(nov["isin"]) == ["GB00BT7J0241"]
    assert list(nov["tenor_bucket"]) == ["20-30y"]
    nov11 = paint[paint["op_dt"].eq("2026-11-11")]
    assert list(nov11["isin"]) == ["GB00BMBL1G81"]
    assert list(nov11["tenor_bucket"]) == ["1-3y"]
    # Auction morning: the offer is still paint until D2.1E has the result.
    # 5⅜% 2056 settled the same calendar day must not hide 0⅛% 2028.
    morning = pd.DataFrame({
        "op_dt": [pd.Timestamp("2026-11-11")],
        "amount_issued": [1_000.0],
        "sec_type": ["Nominal"],
        "isin": ["GB00BT7J0241"],
        "maturity_dt": [pd.Timestamp("2056-01-31")],
        "price": [98.0],
        "op_type": ["Auction"],
        "country": ["UK"],
    })
    live = project("UK", morning, pd.DataFrame(), pd.Timestamp("2026-11-11"))
    still = live[live["op_dt"].eq("2026-11-11")]
    assert list(still["isin"]) == ["GB00BMBL1G81"]
    assert float(still["amount_mn"].iloc[0]) == 1000.0
    assert "GB00BT7J0241" not in set(still["isin"])
    assert "GB00BT7J0241" not in set(live.loc[live["op_dt"].eq("2026-11-10"), "isin"])
    # Once D2.1E has this ISIN on the auction day, do not paint it again.
    settled = pd.concat([
        morning,
        pd.DataFrame({
            "op_dt": [pd.Timestamp("2026-11-11")],
            "amount_issued": [1_000.0],
            "sec_type": ["Nominal"],
            "isin": ["GB00BMBL1G81"],
            "maturity_dt": [pd.Timestamp("2028-01-31")],
            "price": [99.0],
            "op_type": ["Auction"],
            "country": ["UK"],
        }),
    ], ignore_index=True)
    done = project("UK", settled, pd.DataFrame(), pd.Timestamp("2026-11-11"))
    assert "GB00BMBL1G81" not in set(done.loc[done["op_dt"].eq("2026-11-11"), "isin"])
    # The next origin drops the auction date. Settlement on T+1 does not double-count.
    nxt = project("UK", morning, pd.DataFrame(), pd.Timestamp("2026-11-12"))
    assert "GB00BMBL1G81" not in set(nxt.loc[nxt["op_dt"].eq("2026-11-11"), "isin"])
    # 7 Jan is on D5D with no gilt. It takes the first October conventional, not a rump.
    jan = paint[paint["op_dt"].eq("2027-01-07")]
    assert list(jan["isin"]) == ["GB00BWBR1P52"]
    assert list(jan["source"]) == ["pattern"]
    assert "30y+" not in set(paint["tenor_bucket"])
    # 5⅜% 2056 is the 10 Nov auction only. It is not repeated into 2027 auctions.
    auctions = paint[~paint["source"].str.startswith("syndication")]
    assert int((auctions["isin"] == "GB00BT7J0241").sum()) == 1
    assert paint["op_dt"].max() >= pd.Timestamp("2027-09-01")


def test_pick_line_keeps_tenor_and_skips_rumps():
    from issuance.forward import _pick_line

    asof = pd.Timestamp("2026-09-01")
    opened = pd.DataFrame([
        {"isin": "RUMP", "tenor_bucket": "10-15y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=400), "median_size_mn": 400.0},
        {"isin": "AUCT", "tenor_bucket": "10-15y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=20), "median_size_mn": 3500.0},
        {"isin": "SHORT", "tenor_bucket": "3-7y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=500), "median_size_mn": 4000.0},
    ])
    assert _pick_line(opened, "10-15y", False, 6000.0) == "AUCT"

    longs = pd.DataFrame([
        {"isin": "TEN", "tenor_bucket": "7-10y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=400), "median_size_mn": 5000.0},
        {"isin": "LONG", "tenor_bucket": "20-30y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=30), "median_size_mn": 1500.0},
    ])
    assert _pick_line(longs, "30y+", False, 1500.0) == "LONG"

    pair = pd.DataFrame([
        {"isin": "A", "tenor_bucket": "3-7y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=40), "median_size_mn": 4000.0},
        {"isin": "B", "tenor_bucket": "3-7y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=10), "median_size_mn": 4000.0},
    ])
    assert _pick_line(pair, "3-7y", False, 4000.0) == "A"
    assert _pick_line(pair, "3-7y", False, 4000.0) == "B"


def test_bucket_month_is_split_at_the_lines_own_print():
    from issuance.forward import _spread

    asof = pd.Timestamp("2026-09-01")
    opened = pd.DataFrame([
        {"isin": "A", "tenor_bucket": "3-7y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=40), "median_size_mn": 3000.0, "p90_size_mn": 4000.0},
        {"isin": "B", "tenor_bucket": "3-7y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=10), "median_size_mn": 3000.0, "p90_size_mn": 4000.0},
    ])
    rows = _spread(opened, "3-7y", False, 12_000.0, asof, "pattern")
    by = {}
    for row in rows:
        by[row["isin"]] = by.get(row["isin"], 0.0) + row["amount_mn"]
    assert abs(sum(by.values()) - 12_000.0) < 1.0
    assert by.get("A", 0) <= 4000.0 + 1e-6
    assert by.get("B", 0) <= 4000.0 + 1e-6


def test_a_mini_line_does_not_absorb_the_month():
    from issuance.forward import _spread

    asof = pd.Timestamp("2026-09-01")
    opened = pd.DataFrame([
        {"isin": "MINI", "tenor_bucket": "3-7y", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=30), "median_size_mn": 0.4, "p90_size_mn": 0.8},
    ])
    rows = _spread(opened, "3-7y", False, 8_000.0, asof, "pattern")
    mini = sum(r["amount_mn"] for r in rows if r["isin"] == "MINI")
    assert mini < 50
    assert abs(sum(r["amount_mn"] for r in rows) - 8_000.0) < 1.0


def test_uk_syndications_close_the_remit_programmes():
    from issuance.forward import UK_SYNDICATIONS

    fy = [r for r in UK_SYNDICATIONS if r[0] <= "2027-03-31"]
    assert [r[1] for r in fy if r[4] == "Linker"] == ["NEW IL 2043"]
    assert fy[0][0].startswith("2026-11")
    assert len([r for r in fy if r[4] == "Nominal"]) == 2
    paint = project("UK", pd.DataFrame(columns=["op_dt", "isin", "amount_issued", "sec_type", "resid_mat_yrs"]),
                    pd.DataFrame(), pd.Timestamp("2026-09-30"))
    synd = paint[paint["source"].str.startswith("syndication")]
    assert len(synd) == len(UK_SYNDICATIONS)
    assert synd["maturity_dt"].notna().all()


def test_uk_calendar_does_not_invent_a_gilt(monkeypatch):
    _pin_sep_remit(monkeypatch)
    asof = pd.Timestamp("2026-09-30")
    ops = pd.DataFrame({
        "op_dt": [pd.Timestamp("2026-01-15")],
        "amount_issued": [1.0],
        "resid_mat_yrs": [8.0],
        "sec_type": ["Nominal"],
        "isin": ["GB00OLD"],
    })
    opened = pd.DataFrame([
        {"isin": "GB00LONG", "tenor_bucket": "30y+", "sec_type": "Nominal",
         "last_tap": asof - pd.Timedelta(days=30), "median_size_mn": 2000.0},
    ])
    paint = project("UK", ops, opened, asof)
    assert paint[paint["isin"].eq("GB00LONG")].empty
    auctions = paint[~paint["source"].str.startswith("syndication")]
    assert auctions[auctions["isin"].astype(str).str.startswith("NEW ")].empty
    # 13 Oct is the minutes' long tender, not an auction.
    oct13 = paint[pd.to_datetime(paint["op_dt"]).dt.strftime("%Y-%m-%d").eq("2026-10-13")]
    assert list(oct13["source"]) == ["tender"]


def test_a_full_gilt_stops_and_the_next_line_starts():
    from issuance.forward import _fill_to_amount

    asof = pd.Timestamp("2026-10-01")
    opened = pd.DataFrame([
        {"isin": "FULL", "tenor_bucket": "7-10y", "sec_type": "Nominal",
         "last_tap": asof, "median_size_mn": 4000.0, "p90_size_mn": 5000.0,
         "stock_mn": 40000.0, "room_mn": 0.0},
        {"isin": "BUILD", "tenor_bucket": "7-10y", "sec_type": "Nominal",
         "last_tap": asof, "median_size_mn": 4000.0, "p90_size_mn": 5000.0,
         "stock_mn": 36000.0, "room_mn": 4000.0},
    ])
    opened.attrs["line_cap"] = {("Nominal", "7-10y"): 40000.0}
    opened.attrs["line_tap"] = {("Nominal", "7-10y"): 4000.0}
    rows = _fill_to_amount(opened, "7-10y", False, 20_000.0, asof, asof + pd.Timedelta(days=80), "remit")
    full = sum(r["amount_mn"] for r in rows if r["isin"] == "FULL")
    build = sum(r["amount_mn"] for r in rows if r["isin"] == "BUILD")
    nxt = [r for r in rows if str(r["isin"]).startswith("next ")]
    assert full == 0
    assert abs(build - 4000.0) < 1.0
    assert nxt and sum(r["amount_mn"] for r in nxt) > 0
    assert abs(sum(r["amount_mn"] for r in rows) - 20_000.0) < 1.0
    by = {}
    for row in nxt:
        by[row["isin"]] = by.get(row["isin"], 0.0) + row["amount_mn"]
    assert max(by.values()) <= 40_000.0 + 1e-6


def test_paint_respects_envelope_and_cap():
    asof = pd.Timestamp("2026-01-01")
    lines = pd.DataFrame([
        {"isin": "A", "tenor_bucket": "7-10y", "stock_mn": 1000.0, "median_gap_days": 28.0,
         "median_size_mn": 100.0, "last_tap": asof - pd.Timedelta(days=40), "peak_mn": 1400.0},
        {"isin": "B", "tenor_bucket": "7-10y", "stock_mn": 1300.0, "median_gap_days": 28.0,
         "median_size_mn": 100.0, "last_tap": asof - pd.Timedelta(days=10), "peak_mn": 1400.0},
    ])
    rows = _paint_bucket(lines, envelope=500.0, asof=asof)
    assert rows
    assert sum(r["amount_mn"] for r in rows) <= 500.0 + 1e-6
    by_isin = {}
    for row in rows:
        by_isin[row["isin"]] = by_isin.get(row["isin"], 0.0) + row["amount_mn"]
    assert by_isin.get("A", 0) <= 400.0 + 1e-6
    assert by_isin.get("B", 0) <= 100.0 + 1e-6
