import pandas as pd
import pytest

from issuance import forward
from issuance.uk_remit import FALLBACK, REMIT_PDF, SECTORS, bond_price, dmo_sector, remit_progress

MATURITY = {
    "GB00BWBR1N39": ("2036-03-07", "Nominal"), "GB00BT7HZZ68": ("2035-09-22", "Linker"),
    "GB00BWBR1P52": ("2031-10-22", "Nominal"), "GB00BNNGP551": ("2031-08-10", "Linker"),
    "GB00BWGDRG14": ("2032-03-07", "Nominal"), "GB00BVP99905": ("2041-01-31", "Nominal"),
    "GB00BT7J0241": ("2056-01-31", "Nominal"), "GB00BVP99780": ("2030-06-07", "Nominal"),
    "GB00BVP99897": ("2039-07-31", "Nominal"), "GB00B39R3707": ("2049-12-07", "Nominal"),
    "GB00B128DP45": ("2046-12-07", "Nominal"), "GB00BPJJKP77": ("2043-10-22", "Nominal"),
    "GB00BMY62Z61": ("2038-09-22", "Linker"), "GB00BT7J0134": ("2049-09-22", "Linker"),
}


def _tape() -> pd.DataFrame:
    return pd.DataFrame([
        {"country": "UK", "op_dt": pd.Timestamp("2025-06-02"), "isin": i, "maturity_dt": pd.Timestamp(m),
         "price": 95.0, "amount_issued": 1.0, "op_type": "Auction", "sec_type": s}
        for i, (m, s) in MATURITY.items()
    ])


def test_remit_report_parses_to_the_29_sep_numbers():
    if not REMIT_PDF.exists():
        pytest.skip("no remit PDF")
    rp = remit_progress()
    if rp["asof"] != FALLBACK["asof"]:
        pytest.skip("newer report")
    for key in ("auction_required", "auctions_remaining", "synd_planned", "synd_done", "synd_scheduled_n"):
        assert rp[key] == FALLBACK[key]
    assert rp["unallocated"] == FALLBACK["unallocated"]
    assert rp["total_planned"] - rp["sales_to_date"] == pytest.approx(104_086.0)


def test_dmo_sectors_and_price():
    assert dmo_sector(6.9, False) == "short"
    assert dmo_sector(14.6, False) == "medium"
    assert dmo_sector(15.2, False) == "long"
    assert dmo_sector(30.0, True) == "IL"
    assert bond_price(4.0, 4.0, 10.0) == pytest.approx(100.0)


def test_uk_forward_meets_the_remit_in_cash_by_sector():
    rp = remit_progress()
    rows, targets = forward.uk_plan(_tape(), rp["asof"])
    fit = forward._fit_cash([dict(r) for r in rows], targets)
    got: dict = {}
    for r in fit:
        got[r["_key"]] = got.get(r["_key"], 0.0) + r["amount_mn"] * r["_cash"]
    for s in SECTORS:
        assert got.get(("fy", s, "auction"), 0.0) == pytest.approx(rp["auction_required"][s], abs=1.0)
        want = max(rp["synd_planned"][s] - rp["synd_done"][s], 0.0)
        assert got.get(("fy", s, "syndication"), 0.0) == pytest.approx(want, abs=1.0)
    assert got[("fy", "all", "tender")] == pytest.approx(rp["unallocated"], abs=1.0)
    total = sum(v for k, v in got.items() if k[0] == "fy")
    # Medium syndications overran their plan by £29mn; that is not clawed back from auctions.
    assert total == pytest.approx(rp["total_planned"] - rp["sales_to_date"], abs=50.0)


def test_q4_auction_dates_go_to_the_sectors_with_auctions_left():
    rp = remit_progress()
    rows, _ = forward.uk_plan(_tape(), rp["asof"])
    n = {s: 0 for s in SECTORS}
    for r in rows:
        period, s, method = r["_key"]
        if period == "fy" and method == "auction":
            n[s] += 1
    assert n["long"] == rp["auctions_remaining"]["long"]
    assert n["IL"] == rp["auctions_remaining"]["IL"]
    q4_long = [r for r in rows if r["_key"] == ("fy", "long", "auction") and r["op_dt"] >= pd.Timestamp("2027-01-01")]
    assert {r["isin"] for r in q4_long} <= set(forward.UK_Q4_MENU["long"])
    before = [r for r in rows if r["_key"] == ("fy", "long", "auction") and r["op_dt"] < pd.Timestamp("2027-01-01")]
    assert len(q4_long) == rp["auctions_remaining"]["long"] - len(before)


def test_fr_months_meet_the_net_programmes():
    ops = pd.DataFrame({
        "country": "FR", "sec_type": "Nominal",
        "op_dt": list(pd.date_range("2025-01-01", "2025-12-01", freq="MS") + pd.Timedelta(days=14))
        + list(pd.date_range("2026-01-01", "2026-09-01", freq="MS") + pd.Timedelta(days=14)),
        "amount_issued": [30_000.0] * 11 + [5_000.0] + [33_000.0] * 9,
    })
    t = forward.fr_month_targets(ops)
    gross = 9 * 33_000.0 / (forward.FR_PROGRAMME_MN * forward.FR_DONE_SHARE)
    q4 = sum(v for k, v in t.items() if k.year == 2026)
    y27 = sum(v for k, v in t.items() if k.year == 2027)
    assert q4 / gross == pytest.approx(forward.FR_PROGRAMME_MN * (1 - forward.FR_DONE_SHARE))
    assert y27 / gross == pytest.approx(forward.FR_PROGRAMME_NEXT_MN)
    assert t[pd.Period("2026-12", "M")] < t[pd.Period("2026-11", "M")] / 4
