import pandas as pd

from issuance.fetch_uk import D21A_XLS, RAW, _d21a_results, _gilt_frame, _gilt_key, paof_at_takeup

_SAMPLE = b"""<Data>
  <View_Gilt_Issuance_History INSTRUMENT_NAME="4 1/8% Treasury Gilt 2027" ISIN_CODE="GB00BL6C7720" ACTUAL_DATE="2022-10-13T00:00:00" ISSUANCE_TYPE="Outright" NOMINAL_ISSUED="4000" ISSUE_CLEAN_PRICE="99.5" ISSUE_YIELD="4.1" />
</Data>"""


def test_d21e_attributes_are_rows():
    out = _gilt_frame(_SAMPLE)
    assert len(out) == 1
    assert out.loc[0, "isin"] == "GB00BL6C7720"
    assert float(out.loc[0, "amount_issued"]) == 4000
    assert out.loc[0, "op_type"] == "Auction"


def test_gilt_key_reads_both_coupon_spellings():
    assert _gilt_key("4⅝% Treasury Gilt 2032") == _gilt_key("4 5/8% Treasury Gilt 2032") == (4.625, 2032, False)
    assert _gilt_key("0¾% Index-linked Treasury Gilt 2034") == (0.75, 2034, True)
    assert _gilt_key("0 1/8% Index-linked Treasury Gilt 2031") == (0.125, 2031, True)


def _row(name, isin, dt, amt, op="Auction"):
    return {"sec_desc_long": name, "isin": isin, "op_dt": dt, "amount_issued": amt,
            "op_type": op, "sec_type": "Linker" if "Index" in name else "Nominal", "source": "DMO D2.1E"}


def _d21a(rows):
    out = pd.DataFrame(rows, columns=["date", "name", "auction_mn", "paof_mn"])
    out["date"] = pd.to_datetime(out["date"])
    out["key"] = out["name"].map(_gilt_key)
    return out


def test_paof_entitlement_is_cut_to_takeup():
    g32, g36, g40, g11, g49 = ("4 5/8% Treasury Gilt 2032", "4 7/8% Treasury Gilt 2036",
                               "4¼% Treasury Gilt 2040", "4¼% Treasury Gilt 2011", "4¼% Treasury Gilt 2049")
    ops = pd.DataFrame([
        _row(g32, "A", "2026-09-23", 4749.999), _row(g32, "A", "2026-09-23", 1187.5, "Tender"),  # split, 25% entitlement
        _row(g36, "B", "2026-08-19", 5000.0),                                                    # combined, entitlement
        _row(g40, "C", "2026-09-17", 750.0), _row(g40, "C", "2026-09-17", 75.0, "Tender"),       # zero take-up
        _row(g11, "D", "2008-10-22", 4760.0),                                                    # before the facility
        _row(g49, "E", "2026-03-26", 1562.0),                                                    # D2.1E below D2.1A
    ])
    d21a = _d21a([
        ("2026-09-22", "4⅝% Treasury Gilt 2032", 4750.0, 777.75),
        ("2026-08-18", "4⅞% Treasury Gilt 2036", 4000.0, 73.78),
        ("2026-09-16", "4¼% Treasury Gilt 2040", 750.0, 0.0),
        ("2008-10-21", "4¼% Treasury Gilt 2011", 4750.0, 0.0),
        ("2026-03-25", "4¼% Treasury Gilt 2049", 1250.0, 500.0),
        ("2009-06-02", "4½% Treasury Gilt 2019", 3000.0, 100.0),
    ])
    out = paof_at_takeup(ops, d21a)
    by = out.groupby("isin")["amount_issued"].sum()
    assert abs(by["A"] - 5527.75) < 1e-6
    assert (out["isin"] == "A").sum() == 2
    assert abs(by["B"] - 4073.78) < 1e-6
    assert abs(by["C"] - 750.0) < 1e-6 and (out["isin"] == "C").sum() == 1
    assert by["D"] == 4760.0
    assert by["E"] == 1562.0


def test_sep_2026_auction_at_d21a_takeup():
    """22-Sep-26 4⅝% 2032: D2.1A 4,750 + PAOF 777.75, not the £1,187.5mn entitlement."""
    xml = RAW / "uk" / "D2_1E.xml"
    if not xml.exists() or not D21A_XLS.exists():
        return
    ops = paof_at_takeup(_gilt_frame(xml.read_bytes()), _d21a_results())
    hit = ops[(ops["isin"] == "GB00BWGDRG14") & (pd.to_datetime(ops["op_dt"]).dt.normalize() == "2026-09-23")]
    assert abs(pd.to_numeric(hit["amount_issued"]).sum() - 5527.75) < 0.01
