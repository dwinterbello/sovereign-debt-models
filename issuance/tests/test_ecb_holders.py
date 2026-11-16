"""ECB holder-sector panel: parsing, mapping, publication stamping, sector identities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from issuance import ecb_holders as eh

HDR_NA = ("KEY,FREQ,ADJUSTMENT,REF_AREA,COUNTERPART_AREA,REF_SECTOR,COUNTERPART_SECTOR,CONSOLIDATION,"
          "ACCOUNTING_ENTRY,STO,INSTR_ASSET,MATURITY,EXPENDITURE,UNIT_MEASURE,CURRENCY_DENOM,VALUATION,PRICES,"
          "TRANSFORMATION,CUST_BREAKDOWN,TIME_PERIOD,OBS_VALUE,UNIT_MULT\n")


def _na_line(flow, ref_area, cpa, sector, sto, val, period, value, cps="S13", entry="A"):
    key = f"{flow}.Q.N.{ref_area}.{cpa}.{sector}.{cps}.N.{entry}.{sto}.F3.T._Z.XDC._T.{val}.V.N._T"
    return (f"{key},Q,N,{ref_area},{cpa},{sector},{cps},N,{entry},{sto},F3,T,_Z,XDC,_T,{val},V,N,_T,"
            f"{period},{value},6\n")


SECTORS = {"S11": 10.0, "S121": 500.0, "S122": 150.0, "S123": 5.0, "S124": 200.0, "S125A": 2.0, "S128": 70.0,
           "S129": 110.0, "S13": 200.0, "S1311": 190.0, "S1M": 20.0}


def _shss_csv(tmp_path):
    s = dict(SECTORS)
    s["S12P"] = s["S124"] + s["S125A"] + 5.0
    s["S12Q"] = s["S128"] + s["S129"]
    s["S12"] = s["S121"] + s["S122"] + s["S123"] + s["S12P"] + s["S12Q"]
    s["S1"] = s["S11"] + s["S12"] + s["S13"] + s["S1M"]
    lines = [HDR_NA]
    for c, k in (("DE", 1.0), ("IT", 2.0)):
        for code, v in s.items():
            for period, g in (("2021-Q1", 1.0), ("2021-Q2", 1.1)):
                lines.append(_na_line("SHSS", "U2", c, code, "LE", "M", period, v * k * g * 1e3))
                lines.append(_na_line("SHSS", "U2", c, code, "LE", "F", period, v * k * g * 0.9e3))
                lines.append(_na_line("SHSS", "U2", c, code, "F", "M", period, v * k * 0.05e3))
    # a non-euro-area holder row and a non-S13 issuer row must be ignored
    lines.append(_na_line("SHSS", "DE", "W2", "S122", "LE", "M", "2021-Q1", 9e9))
    lines.append(_na_line("SHSS", "U2", "DE", "S122", "LE", "M", "2021-Q1", 9e9, cps="S11"))
    p = tmp_path / "shss.csv"
    p.write_text("".join(lines))
    return p, s


def test_shss_parse_units_and_mapping(tmp_path):
    p, s = _shss_csv(tmp_path)
    r = eh.shss_raw(p)
    assert set(r["country"]) == {"DE", "IT"}
    v = r[(r["country"] == "DE") & (r["code"] == "S122") & (r["measure"] == "LE") & (r["valuation"] == "M")
          & (r["quarter_end"] == pd.Timestamp("2021-03-31"))]["value_eur_bn"]
    assert v.tolist() == [pytest.approx(150.0)]  # UNIT_MULT 6 → EUR mn → bn
    m = r.drop_duplicates("code").set_index("code")["sector"]
    assert m["S121"] == "eurosystem" and m["S122"] == "banks" and m["S12Q"] == "icpf" and m["S124"] == "funds"
    assert m["S1M"] == "households" and m["S1"] == "ea" and pd.isna(m["S1311"])
    assert r["quarter_end"].max() == pd.Timestamp("2021-06-30")


def test_sector_sums_close(tmp_path):
    p, _ = _shss_csv(tmp_path)
    g = eh.sector_sums(p)
    assert len(g) and g[["gap_s1", "gap_s12", "gap_icpf"]].abs().max().max() < 1e-6


def test_shs_needs_both_maturities(tmp_path):
    hdr = ("KEY,FREQ,HOLDER_AREA,ISSUER_AREA,HOLDER_SECTOR,ISSUER_SECTOR,FUNCTIONAL_CAT,STO,INSTR_ASSET,MATURITY,"
           "UNIT_MEASURE,VALUATION,SECURITY_TYPE,MA_FLAG,IS_IN_EADB,SECURITISATION_TYP,DEBT_TYPE,TPH,NOM_CURR,"
           "COMPILING_ORG,TIME_PERIOD,OBS_VALUE,UNIT_MULT\n")
    rows = [hdr]
    for code, mats in (("S12T", "SL"), ("S12Q", "L")):
        for mat in mats:
            rows.append(f"k,Q,I8,FR,{code},S13,_X,LE,F3,{mat},EUR,M,A,A,A,_X,_X,V,_T,ZZZ,2020-Q4,1000,6\n")
    p = tmp_path / "shs.csv"
    p.write_text("".join(rows))
    r = eh.shs_raw(p)
    assert r["code"].tolist() == ["S12T"]
    assert r["sector"].tolist() == ["mfi_ex_cb"] and r["value_eur_bn"].tolist() == [pytest.approx(2.0)]


def test_public_months():
    assert eh.shss_public_month(pd.Timestamp("2026-06-30")) == pd.Timestamp("2026-09-01")
    assert eh.qsa_public_month(pd.Timestamp("2026-03-31")) == pd.Timestamp("2026-07-01")
    assert eh.bsi_public_month(pd.Timestamp("2026-08-01")) == pd.Timestamp("2026-09-01")


def test_stamp_no_lookahead():
    q = pd.DataFrame({"country": "DE", "metric": "m", "quarter_end": pd.to_datetime(["2025-12-31", "2026-03-31"]),
                      "value": [1.0, 2.0]})
    q["public"] = [max(eh.shss_public_month(x), eh.qsa_public_month(x)) for x in q["quarter_end"]]
    out = eh._stamp(q, "quarter_end", pd.Timestamp("2026-09-01")).set_index("month")["value"]
    assert out.index.min() == pd.Timestamp("2026-04-01")  # Q4 public in April, not January
    assert out.loc["2026-06-01"] == 1.0 and out.loc["2026-07-01"] == 2.0
    # a period not yet public by `end` is dropped
    out2 = eh._stamp(q, "quarter_end", pd.Timestamp("2026-06-01"))
    assert (out2["value"] == 1.0).all()


def test_bsi_metrics_stamped_month_after(tmp_path):
    b = pd.DataFrame({"country": "IT", "month": pd.date_range("2024-01-01", periods=24, freq="MS"),
                      "value_eur_bn": 400.0, "flow_eur_bn": 1.0, "source": "x"})
    gdp = {"IT": pd.Series(2000.0, index=pd.date_range("2020-01-01", "2026-04-01", freq="QS"))}
    m = eh.metrics(today=pd.Timestamp("2026-02-15"), p=_tiny_panel(), b=b, gdp=gdp)
    f12 = m[m["metric"] == "bsi_bank_f12m"].set_index("month")["value"]
    # first 12m sum ends Dec-2024 (data) → public Jan-2025
    assert f12.index.min() == pd.Timestamp("2025-01-01")
    assert f12.iloc[0] == pytest.approx(100 * 12 / 2000.0)
    assert m["month"].max() == pd.Timestamp("2026-01-01")
    assert (m["phase"] == "history").all()


def _tiny_panel():
    rows = []
    for q in pd.date_range("2021-03-31", periods=8, freq="QE"):
        for s, v in (("total", 1000.0), ("ea", 800.0), ("row", 200.0), ("mfi_ex_cb", 150.0), ("icpf", 100.0),
                     ("ofi", 80.0), ("households", 10.0), ("ea_ex_cb", 300.0), ("eurosystem", 500.0)):
            rows.append({"country": "IT", "quarter_end": q, "sector": s, "measure": "LE", "value_eur_bn": v,
                         "source": "t"})
            rows.append({"country": "IT", "quarter_end": q, "sector": s, "measure": "F", "value_eur_bn": v / 100,
                         "source": "t"})
    return pd.DataFrame(rows)


def test_quarterly_metrics_shares():
    gdp = {"IT": pd.Series(2000.0, index=pd.date_range("2020-01-01", "2026-04-01", freq="QS"))}
    q = eh.quarterly_metrics(_tiny_panel(), gdp)
    last = q[q["quarter_end"] == q["quarter_end"].max()].set_index("metric")["value"]
    assert last["ecb_bank_share"] == pytest.approx(15.0)
    assert last["ecb_row_share"] == pytest.approx(20.0)
    assert last["ecb_icpf_f4q"] == pytest.approx(100 * 4 * 1.0 / 2000.0)
    assert set(q["metric"]) <= set(eh.METRICS)
    assert all(eh.METRICS[k]["sign"] in (1, -1) for k in eh.METRICS)


def test_implied_flows_exact_under_common_revaluation():
    qe = pd.date_range("2020-03-31", periods=3, freq="QE")
    qsa = pd.DataFrame({"country": "FR", "quarter_end": list(qe) * 2, "sector": "total",
                        "measure": ["LE"] * 3 + ["F"] * 3, "value_eur_bn": [100.0, 115.0, 110.0, 0.0, 10.0, 5.0]})
    # revaluation: q2 (115 − 100 − 10)/100 = 5%, q3 (110 − 115 − 5)/115
    st = pd.DataFrame({"country": "FR", "quarter_end": qe, "sector": "icpf", "measure": "LE",
                       "value_eur_bn": [50.0, 50.0 * 1.05 + 3.0, np.nan], "source": "t"})
    f = eh._implied_flows(st.dropna(), qsa)
    assert f["value_eur_bn"].tolist() == [pytest.approx(3.0)]


@pytest.mark.skipif(not (eh.DIR / "shss.csv").exists(), reason="no cached SHSS")
def test_cached_identities_and_splice():
    g = eh.sector_sums()
    assert g[["gap_s1", "gap_s12", "gap_icpf"]].abs().max().max() < 0.01
    sc = eh.splice_check()
    core = sc[sc["code"].isin(["S12T", "S12Q", "S1ZK"])]
    assert (core["gap_pct"].abs() < 2.0).all()
    p = eh.shss_panel()
    w = eh.wide(p)
    w = w[w["quarter_end"] >= eh.SHSS_START].dropna(subset=["total", "ea"])
    assert ((w["ea"] + w["row"] - w["total"]).abs() < 1e-6).all()
    parts = w[["eurosystem", "mfi_ex_cb", "ofi", "icpf", "households", "nfc", "government"]].sum(axis=1)
    assert ((parts - w["ea"]).abs() / w["ea"] < 1e-6).all()
