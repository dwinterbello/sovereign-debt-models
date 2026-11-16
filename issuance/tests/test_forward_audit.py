import pandas as pd

from issuance.fetch_uk import tag_creations
from issuance.forward_audit import GRADE, _basis, auction_sizes, grade_mix, line_audit


def _op(country, isin, day, amt, sec="Nominal", op="Auction", bucket="7-10y", mat="2035-01-01"):
    return {"country": country, "isin": isin, "op_dt": pd.Timestamp(day), "amount_issued": amt,
            "sec_type": sec, "op_type": op, "tenor_bucket": bucket, "maturity_dt": pd.Timestamp(mat),
            "sec_desc_long": isin, "official_take_mn": None}


def test_whole_list_creation_is_own_holdings():
    rows = [_op("UK", f"GB{i:010d}", "2025-04-15", 300.0) for i in range(25)]
    rows.append(_op("UK", "GB9999999999", "2025-05-01", 4000.0))
    out = tag_creations(pd.DataFrame(rows))
    assert (out.loc[:24, "op_type"] == "Own Holdings").all()
    assert out.loc[25, "op_type"] == "Auction"


def test_priced_auction_on_creation_day_stays_auction():
    rows = [_op("UK", f"GB{i:010d}", "2008-10-22", 1800.0) for i in range(29)]
    rows.append(dict(_op("UK", "GB00B0LNX641", "2008-10-22", 4750.0), price=100.9, **{"yield": 3.848}))
    out = tag_creations(pd.DataFrame(rows))
    assert (out.loc[:28, "op_type"] == "Own Holdings").all()
    assert out.loc[29, "op_type"] == "Auction"


def test_basis_grades():
    assert _basis("UK", "calendar", "GB00BWBR1N39", pd.Timestamp("2026-11-01")) == "announced"
    assert _basis("UK", "calendar", "NEW long gilt", pd.Timestamp("2026-11-01")) == "programme"
    assert _basis("US", "refunding", "US 10y", pd.Timestamp("2026-11-01")) == "announced"
    assert _basis("US", "treasury", "US 30y", pd.Timestamp("2026-11-12")) == "announced"
    assert _basis("FR", "pattern", "FR0000000001", pd.Timestamp("2027-06-01")) == "programme"
    assert _basis("DE", "pattern", "DE0000000001", pd.Timestamp("2027-03-01")) == "projection"
    assert _basis("IT", "pattern", "IT0000000001", pd.Timestamp("2027-03-01")) == "extrapolated"
    assert GRADE["projection"] == "Low"


def _fwd(rows):
    f = pd.DataFrame(rows, columns=["country", "isin", "op_dt", "gross_mn", "source", "dv01_usd_mn", "tenor_bucket", "sec_type"])
    f["op_dt"] = pd.to_datetime(f["op_dt"])
    f["sleeve"] = "Medium"
    return f


def test_oversized_tap_is_flagged_and_downgraded():
    ops = pd.DataFrame([_op("FR", "FR1", f"2026-0{m}-05", 3000.0) for m in range(1, 9)])
    fwd = _fwd([["FR", "FR1", "2026-11-05", 9000.0, "pattern", 5.0, "7-10y", "Nominal"],
                ["FR", "FR1", "2026-12-03", 9000.0, "pattern", 5.0, "7-10y", "Nominal"]])
    ln = line_audit(ops, fwd)
    assert "tap 3.0x own median" in ln.iloc[0]["flags"]
    assert ln.iloc[0]["grade"] == "Low"
    assert grade_mix(ln).loc["FR", "Low"] == 1.0


def test_bills_forward_repeats_last_year():
    ops = pd.DataFrame([_op("UK", "GBB1", "2026-03-02", 2000.0, sec="Bill", bucket="0-1y", mat="2026-06-01"),
                        _op("UK", "GBB2", "2026-09-07", 2500.0, sec="Bill", bucket="0-1y", mat="2026-12-07"),
                        _op("UK", "GB1", "2026-09-01", 3000.0)])
    rows = pd.DataFrame({"country": ["UK"], "isin": ["GB1"], "op_dt": [pd.Timestamp("2027-06-01")],
                         "sleeve": ["Medium"], "amount_mn": [3000.0], "source": ["pattern"],
                         "basis": ["programme"], "grade": ["Medium"]})
    out = auction_sizes(ops, rows)
    rep = out[(out["sleeve"] == "Bills") & (out["phase"] == "forward")]
    assert list(rep["op_dt"]) == [pd.Timestamp("2027-03-01")]
    assert rep["basis"].iloc[0] == "repeat last year"
    assert rep["label"].iloc[0] == "Bill 13w"


def test_de_2027_scale_adds_net_borrowing_and_redemption_change():
    from issuance.forward import DE_NET_BORROWING, de_2027_scale

    ops = pd.DataFrame({
        "country": "DE", "isin": ["A", "B", "C"], "sec_type": "Nominal", "op_type": "Auction",
        "op_dt": pd.to_datetime(["2026-03-01", "2020-01-01", "2020-01-01"]),
        "maturity_dt": pd.to_datetime(["2036-03-01", "2026-06-01", "2027-06-01"]),
        "amount_issued": [300_000.0, 200_000.0, 210_000.0],
        "tenor_bucket": ["7-10y", "1-3y", "1-3y"],
    })
    scale = de_2027_scale(ops)
    from issuance.forward import DE_CALENDAR
    g26 = 300_000.0 + sum(a for d, _, _, a in DE_CALENDAR if a and pd.Timestamp(d).year == 2026 and pd.Timestamp(d) > pd.Timestamp("2026-03-01"))
    want = (g26 + DE_NET_BORROWING[2027] - DE_NET_BORROWING[2026] + 210_000.0 - 200_000.0) / g26
    assert abs(scale - want) < 1e-9
