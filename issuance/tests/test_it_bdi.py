from pathlib import Path

import pandas as pd
import pytest

from issuance.fetch_it import _it_coupon, _merge_it, _sec_type, parse_bdi_workbook


def test_it_coupon_from_btp_line():
    assert abs(_it_coupon("BTP 3,35% 15.7.2026 - 15.9.2033") - 3.35) < 1e-9
    assert abs(_it_coupon("BTP 2,4% 15.10.2030") - 2.4) < 1e-9


def test_btpi_is_linker():
    assert _sec_type("BTPI", "BTP€i 0.10%") == "Linker"
    assert _sec_type("BTP", "BTP 3%") == "Nominal"


def test_ctz_is_coupon_family_not_bill():
    assert _sec_type("CTZ", "CTZ 30.5.2022 - 30.5.2024") == "Nominal"
    assert _sec_type(None, "CTZ 30.5.2022 - 30.5.2024") == "Nominal"
    assert _sec_type("BOT", "BOT 182 gg") == "Bill"


def test_merge_adds_bdi_supplementary_inside_mef_window():
    mef = pd.DataFrame({
        "op_dt": pd.to_datetime(["2024-01-11", "2025-12-11"]),
        "op_type": ["Auction", "Auction"],
        "isin": ["IT0001", "IT0002"],
        "amount_issued": [4000.0, 1000.0],
    })
    bdi = pd.DataFrame({
        "op_dt": pd.to_datetime(["2024-01-11", "2024-01-12", "2025-12-11", "2025-12-12"]),
        "op_type": ["Auction"] * 4,
        "isin": ["IT0001", "IT0001", "IT0002", "IT0002"],
        "amount_issued": [4000.0, 1200.0, 1000.0, 200.0],
        "tranche_kind": ["O", "S", "O", "S"],
    })
    out = _merge_it(mef, bdi)
    assert len(out) == 4
    assert out.loc[out["isin"].eq("IT0001"), "amount_issued"].sum() == 5200.0
    assert out.loc[out["isin"].eq("IT0002"), "amount_issued"].sum() == 1200.0


STORICO = Path(__file__).resolve().parents[2] / "data" / "raw" / "mef" / "bdi" / "storico_aste" / "Aste dal 2002 al 2025.xlsx"
PARSED = Path(__file__).resolve().parents[2] / "data" / "raw" / "mef" / "parsed_data" / "auctions_combined.xlsx"


def test_2024_gross_includes_bdi_supplementary_face():
    if not (STORICO.exists() and PARSED.exists()):
        pytest.skip("BdI storico / MEF parsed workbooks not cached")
    from issuance.fetch_it import _from_auctions

    bdi = parse_bdi_workbook(STORICO)
    s24 = bdi[bdi["tranche_kind"].eq("S") & bdi["op_dt"].dt.year.eq(2024)]["amount_issued"].sum()
    assert 30_000 < s24 < 36_000  # BdI register: EUR 33.0bn of 2024 supplementary tranches
    ctz = bdi[bdi["sec_desc_long"].astype(str).str.startswith("CTZ")]
    assert not ctz.empty and ctz["sec_type"].eq("Nominal").all()
    mef = _from_auctions(PARSED)
    merged = _merge_it(mef, bdi)
    year = merged["op_dt"].dt.year.eq(2024) & (merged["amount_issued"] > 0)
    gain = merged.loc[year, "amount_issued"].sum() - mef.loc[mef["op_dt"].dt.year.eq(2024) & (mef["amount_issued"] > 0), "amount_issued"].sum()
    assert abs(gain - s24) < 1.0


def test_mef_syndications_are_not_auctions():
    from issuance.fetch_it import _mef_op_type

    files = pd.Series([
        "BTP_10_Years_Syndicate_placement_08_01_2025___PDF__42_Kb__.pdf",
        "15_Year_BTP_Syndacate_placement_results__15.01.2013.pdf",
        "Placement_result_of_the_new_30_year_BTP_16_10_2007___PDF__25_Kb__.pdf",
        "BTP_Valore_28_10_2032__6th__-_Results_of_the_placement___PDF__204_Kb__.pdf",
        "BTP_Italia_04_06_2032__20th__-_Results_of_the_placement___PDF__166_Kb__.pdf",
        "15_Year_BTP_Re-Opening_Results_21_03_2002___PDF__88_Kb__.pdf",
        "5-Years-BTP-Exchange-Offer-Results-22.03.2002.pdf",
        "10_Year_BTP_Auction_Results_13_05_2025___PDF__40_Kb__.pdf",
        "30-Year-BTP--Latest-Results-14.01.2000.pdf",
    ])
    raw = pd.Series(["Auction", "Auction", "Syndicate", "Auction", "Auction", "Syndicate",
                     "Auction", "Auction", "Auction"])
    assert _mef_op_type(raw, files).tolist() == ["Syndication"] * 6 + ["Exchange", "Auction", "Auction"]


def test_nominal_auctions_match_bdi_register():
    """MEF syndications booked as auctions put Nominal auctions EUR 11-74bn/yr over BdI (2020-25)."""
    if not (STORICO.exists() and PARSED.exists()):
        pytest.skip("BdI storico / MEF parsed workbooks not cached")
    from issuance.fetch_it import _from_auctions

    bdi = parse_bdi_workbook(STORICO)
    mef = _from_auctions(PARSED)
    assert not mef["op_type"].eq("Syndicate").any()
    merged = _merge_it(mef, bdi)
    nom = merged[merged["sec_type"].eq("Nominal") & merged["op_type"].eq("Auction")]
    ops = nom.groupby(nom["op_dt"].dt.year)["amount_issued"].sum()
    reg = bdi[bdi["sec_type"].eq("Nominal")]
    reg = reg.groupby(reg["op_dt"].dt.year)["amount_issued"].sum()
    gap = (ops - reg).loc[2010:2025] / 1e3
    # Never above the register; below it only by off-the-run taps MEF did not parse (<= EUR 4.3bn/yr).
    assert gap.max() < 0.1, gap[gap >= 0.1].to_dict()
    assert gap.min() > -5.0, gap[gap <= -5.0].to_dict()


def test_dropped_lines_restored_only_when_distinct(tmp_path):
    from issuance.fetch_it import _dropped_auction_lines

    log = pd.DataFrame({
        "operation_id": ["IT0005410920_2021-05-14_2020-05-12", "IT0005387078_2020-10-14_2020-05-12",
                         "IT0005387078_2020-10-14_2020-05-12", "IT0003271019_2007-06-01_2002-11-19", None],
        "pdf_path": [r"C:\x\12_Month_BOTs_Auction_Results_12-13.05.2020.pdf",
                     r"C:\x\Flexible_BOTs_Auction_Results_12_05_2020.pdf",
                     r"C:\x\Flexible_BOTs_Auction_Results_12_05_2020_it.pdf",
                     r"C:\x\5_Year_BTP_Exchange_Offer_Results_19_11_2002.pdf",
                     r"C:\x\no_isin.pdf"],
        "duplicate_status": ["DROPPED (fuzzy+isin)"] * 4 + ["DROPPED (fuzzy)"],
        "op_dt": pd.to_datetime(["2020-05-12"] * 3 + ["2002-11-19", "2020-05-12"]),
        "sec_cd": ["BOT", "BOT", "BOT", "BTP", "BOT"],
        "amount_allotted": [7000.0, 3500.0, 3500.0, 900.0, 3500.0],
    })
    with pd.ExcelWriter(tmp_path / "mef_parsing_log.xlsx") as xw:
        log.to_excel(xw, sheet_name="auctions", index=False)
    kept = pd.DataFrame({"isin": ["IT0005410920"], "op_dt": pd.to_datetime(["2020-05-12"])})
    out = _dropped_auction_lines(tmp_path, kept)
    assert out["isin"].tolist() == ["IT0005387078"]
    assert out["amount_allotted"].tolist() == [3500.0]


def test_bill_and_linker_auctions_match_bdi_register():
    """Second BOT / BTP€i lines on one day were fuzzy-dropped by the MEF combine (2015-25)."""
    if not (STORICO.exists() and PARSED.exists()):
        pytest.skip("BdI storico / MEF parsed workbooks not cached")
    from issuance.fetch_it import _from_auctions

    bdi = parse_bdi_workbook(STORICO)
    merged = _merge_it(_from_auctions(PARSED), bdi)
    for kind in ("Bill", "Linker"):
        a = merged[merged["sec_type"].eq(kind) & merged["op_type"].eq("Auction")]
        r = bdi[bdi["sec_type"].eq(kind)]
        gap = (a.groupby(a["op_dt"].dt.year)["amount_issued"].sum()
               - r.groupby(r["op_dt"].dt.year)["amount_issued"].sum()).loc[2015:2025]
        assert gap.abs().max() < 5.0, (kind, gap[gap.abs() >= 5.0].to_dict())


def test_merge_appends_bdi_after_mef_vintage():
    mef = pd.DataFrame({
        "op_dt": pd.to_datetime(["2025-12-11", "2025-11-01"]),
        "op_type": ["Auction", "TAP"],
        "isin": ["IT0001", "IT0002"],
        "amount_issued": [1000.0, 500.0],
    })
    bdi = pd.DataFrame({
        "op_dt": pd.to_datetime(["2025-12-11", "2026-01-09", "2026-09-10"]),
        "op_type": ["Auction", "Auction", "Auction"],
        "isin": ["IT0001", "IT0003", "IT0004"],
        "amount_issued": [1000.0, 8000.0, 3500.0],
    })
    out = _merge_it(mef, bdi)
    assert len(out) == 4
    assert pd.Timestamp("2026-09-10") in set(out["op_dt"])
    assert (out["op_dt"] == pd.Timestamp("2025-12-11")).sum() == 1


def test_parse_bdi_current_if_present():
    path = Path("/tmp/bdi_aste/cur")
    files = list(path.glob("*.xls")) if path.exists() else []
    if not files:
        return
    out = parse_bdi_workbook(files[0])
    assert not out.empty
    assert out["op_dt"].min() >= pd.Timestamp(2026, 1, 1)
    assert out["isin"].astype(str).str.startswith("IT").all()
    assert out.loc[out["amount_issued"] > 0, "amount_issued"].min() > 0
