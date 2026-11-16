from __future__ import annotations

import pandas as pd
import pytest

from issuance.fetch_xborder import OUT, parse_mfh_history, parse_slt_table3

COUNTRIES = {"EA", "DE", "FR", "IT", "US", "UK"}
UNITS = {"EA": "bn_eur", "DE": "bn_eur", "FR": "bn_eur", "IT": "bn_eur",
         "US": "bn_usd", "UK": "bn_gbp"}
REQUIRED = {
    ("EA", "ecb_bop_liab_govt_lt_debt"), ("EA", "ecb_bop_liab_govt_st_debt"),
    ("EA", "ecb_bop_liab_all_lt_debt"), ("EA", "ecb_bop_assets_all_lt_debt"),
    ("EA", "ecb_bop_assets_foreign_govt"),
    ("DE", "ecb_bop_liab_govt_lt_debt"), ("FR", "ecb_bop_liab_govt_lt_debt"),
    ("IT", "ecb_bop_liab_govt_lt_debt"), ("IT", "bdi_row_holdings"), ("IT", "bdi_row_net_change"),
    ("US", "tic_net_lt_treasuries"), ("US", "tic_net_lt_treasuries_official"),
    ("US", "tic_net_lt_treasuries_private"), ("US", "tic_holdings_total"),
    ("US", "tic_holdings_official"), ("US", "tic_holdings_japan"), ("US", "tic_holdings_china"),
    ("US", "tic_holdings_uk"),
    ("UK", "boe_overseas_gilt_net_purchases"),
}
QUARTERLY_REQUIRED = {("UK", "ons_overseas_gilt_tx_q")}


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    if not OUT.exists():
        pytest.skip("xborder_flows.parquet not built")
    return pd.read_parquet(OUT)


def test_schema(flows):
    assert list(flows.columns) == ["country", "period", "series", "value", "unit", "freq", "source"]
    assert set(flows["country"]) <= COUNTRIES
    assert pd.api.types.is_datetime64_any_dtype(flows["period"])
    assert (flows["period"].dt.day == 1).all()
    assert pd.api.types.is_float_dtype(flows["value"])
    assert flows["value"].notna().all()
    assert set(flows["freq"]) <= {"M", "Q"}
    assert (flows["unit"] == flows["country"].map(UNITS)).all()
    assert flows["series"].str.fullmatch(r"[a-z0-9_]+").all()
    assert flows["source"].str.len().gt(0).all()
    q = flows[flows["freq"] == "Q"]
    assert q["period"].dt.month.isin([1, 4, 7, 10]).all()


def test_no_duplicates(flows):
    assert not flows.duplicated(["country", "period", "series"]).any()


def test_series_recent(flows):
    present = set(zip(flows["country"], flows["series"]))
    missing = (REQUIRED | QUARTERLY_REQUIRED) - present
    assert not missing, missing
    latest = flows.groupby(["country", "series"])["period"].max()
    today = pd.Timestamp.today().to_period("M").to_timestamp()
    stale = {k: v.date() for k, v in latest.items()
             if k in REQUIRED and v < today - pd.DateOffset(months=6)}
    stale |= {k: v.date() for k, v in latest.items()
              if k in QUARTERLY_REQUIRED and v < today - pd.DateOffset(months=9)}
    assert not stale, stale


def test_tic_holdings_sane(flows):
    tot = flows[flows["series"] == "tic_holdings_total"].set_index("period")["value"]
    jp = flows[flows["series"] == "tic_holdings_japan"].set_index("period")["value"]
    assert (jp.reindex(tot.index).dropna() < tot.reindex(jp.index).dropna()).all()
    assert tot.iloc[-1] > 5000


def test_parsers_minimal():
    t3 = ("Japan\t42609\t2026-07\t1103943\t891\t1023754\t-8846\t-12120\t80189\t9737\n"
          "Grand Total\t99996\t2020-01\t7027316\tn.a.\t6321567\tn.a.\tn.a.\t705749\t-10308\n")
    df = parse_slt_table3(t3)
    assert df.loc[0, "for_lt_treas_net"] == -8846
    assert pd.isna(df.loc[1, "for_lt_treas_net"])
    mfh = ("\tDec\tJun\tJun\n"
           "Country\t2002\t2002\t2002\n"
           "Japan\t378.1\t349.2\t335.8\n"
           "Grand Total\t1.0\t2.0\t3.0\n")
    m = parse_mfh_history(mfh)
    jun = m[(m["name"] == "Japan") & (m["period"] == "2002-06-01")]
    assert jun["bn"].iloc[0] == 349.2
