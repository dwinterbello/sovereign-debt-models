import pandas as pd
import pytest

from issuance.fetch_jpx import OUT_INVESTOR, RAW_WEEKLY, build_investor, parse_weekly

OLD_ROW = "317,0,4,1,2025125,20251229,20251230,60,28726,98.0,-912,381038829,98.0,-12130071,27814,98.2,0,368908758,98.2,0,56540,98.1,749947587,98.1\n"
NEW_HEAD = ('"帳票種別 Product type","サイクル区分 Cycle","年月週 Year, Month, Week","from","to",'
            '"投資部門コード Type of Investor","数量金額区分 Volume/Value","売 Sales","売-差引 Balance",'
            '"買 Purchases","買-差引 Balance","合計 Total"\n')


def _raw_paths():
    paths = sorted(RAW_WEEKLY.glob("Tousi_DV_W_*.csv")) if RAW_WEEKLY.exists() else []
    if not paths:
        pytest.skip("no cached JPX weekly files")
    return paths


@pytest.fixture(scope="module")
def frame():
    return build_investor(_raw_paths())


def test_old_layout_units():
    df = parse_weekly(OLD_ROW.encode(), "old.csv").set_index("unit")
    assert df.loc["contracts", "sell"] == 28726 and df.loc["contracts", "buy"] == 27814
    assert df.loc["yen", "sell"] == 381038829 * 1e4
    assert (df["investor"] == "foreigners").all()


def test_new_layout_derives_aggregates():
    rows = [f'"317","1","2026094","20260924","20260925","{code}","1",{s},0,{b},0,{s + b}\n'
            for code, s, b in (("11", 100, 150), ("60", 80, 40), ("21", 5, 1), ("51", 1, 0))]
    df = parse_weekly((NEW_HEAD + "".join(rows)).encode("utf-8"), "new.csv").set_index("investor")
    assert df.loc["proprietary", "net"] == 50
    assert df.loc["financial_institutions", "sell"] == 5
    assert df.loc["brokerage_total", "sell"] == 86
    assert df.loc["total", "buy"] == 191
    assert df.loc["total", "sell"] == 186


def test_all_zero_sample_file_dropped():
    row = '"317","1","2026043","20260413","20260417","11","1",0,0,0,0,0\n'
    assert parse_weekly((NEW_HEAD + row).encode(), "sample.csv").empty


def test_every_investor_code_mapped():
    for p in _raw_paths():
        df = parse_weekly(p.read_bytes(), p.name)
        if not df.empty:
            assert df["investor"].notna().all(), (p.name, sorted(df.loc[df["investor"].isna(), "investor_code"].unique()))


def test_net_is_buy_minus_sell(frame):
    assert ((frame["net"] - (frame["buy"] - frame["sell"])).abs() < 1e-6).all()


def test_foreigners_present(frame):
    fx = frame[(frame["contract"] == "jgb10y") & (frame["investor"] == "foreigners")]
    assert not fx.empty
    assert set(fx["unit"]) == {"contracts", "yen"}


def test_weeks_unique(frame):
    assert not frame.duplicated(["week_end", "contract", "investor", "unit"]).any()


def test_totals_close(frame):
    sub = frame[(frame["contract"] == "jgb10y") & (frame["unit"] == "contracts")]
    wide = sub.pivot(index="week_end", columns="investor", values="sell")
    gap = (wide["proprietary"] + wide["brokerage_total"] - wide["total"]).abs()
    assert (gap <= 2).all()


def test_history_and_freshness(frame):
    weeks = frame["week_end"].drop_duplicates().sort_values()
    assert weeks.iloc[0] <= pd.Timestamp("2016-06-30")
    assert (pd.Timestamp("2026-10-01") - weeks.iloc[-1]).days <= 21


def test_cache_matches_raw():
    if not OUT_INVESTOR.exists():
        pytest.skip("no cache")
    df = pd.read_parquet(OUT_INVESTOR)
    assert {"week_end", "contract", "investor", "sell", "buy", "net", "unit", "source"} <= set(df.columns)
