"""UK holder-sector panel: parsing, units, APF split, publication stamping, identities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from issuance import uk_holders as uh


def _ons_payload(quarters=None, months=None):
    return {"description": {"title": "t"},
            "quarters": [{"date": d, "value": v} for d, v in (quarters or [])],
            "months": [{"date": d, "value": v} for d, v in (months or [])]}


def test_parse_ons_units_and_dates():
    q, m = uh.parse_ons(_ons_payload([("2026 Q1", "8722"), ("2026 Q2", "11208"), ("2026 Q3", "")],
                                     [("2026 JUL", "15329"), ("2026 AUG", "..")]))
    assert q.index.tolist() == [pd.Timestamp("2026-03-31"), pd.Timestamp("2026-06-30")]
    assert q.iloc[-1] == pytest.approx(11.208)  # £mn → £bn
    assert m.index.tolist() == [pd.Timestamp("2026-07-01")] and m.iloc[0] == pytest.approx(15.329)


def test_parse_iadb_units_and_mapping():
    text = ("DATE,RPMB3QP,RPMB4CA,LPMB9P6,LPMVRDY\n"
            "31 Jul 2026,243189,9920,10052,-3881\n31 Aug 2026,248943,6296,2273,8589\n")
    d = uh.parse_iadb(text)
    got = d.set_index(["month", "code"])["value_gbp_bn"]
    assert got[(pd.Timestamp("2026-08-01"), "RPMB3QP")] == pytest.approx(248.943)
    assert got[(pd.Timestamp("2026-07-01"), "LPMVRDY")] == pytest.approx(-3.881)
    m = d.drop_duplicates("code").set_index("code")
    assert m.loc["RPMB4CA", "series"] == "banks" and m.loc["RPMB4CA", "measure"] == "F"
    assert m.loc["RPMB3QP", "measure"] == "LE" and m.loc["LPMB9P6", "series"] == "overseas"


def test_maturity_from_desc():
    s = uh.maturity_from_desc(pd.Series(["UKT_3.75_070921", "UKT_0.125_300126", "junk"]))
    assert s.iloc[0] == pd.Timestamp("2021-09-07") and s.iloc[1] == pd.Timestamp("2026-01-30")
    assert pd.isna(s.iloc[2])


def _ops():
    return pd.DataFrame({
        "op_dt": pd.to_datetime(["2020-01-10", "2020-02-10", "2023-03-01"]),
        "settlement_dt": pd.to_datetime(["2020-01-11", "2020-02-11", "2023-03-02"]),
        "isin": ["A", "A", "A"], "sec_desc_long": ["UKT_1_070624"] * 3,
        "amount": [100.0, 50.0, -30.0], "price": [110.0, 100.0, 90.0],
        "direction": ["purchase", "purchase", "sale"],
        "maturity_dt": pd.to_datetime(["2024-07-01"] * 3)})  # year-only join, deliberately wrong day


def test_apf_flows_signs_and_redemption():
    fs = pd.DataFrame({"date": pd.to_datetime(["2022-09-29", "2022-12-01"]), "value_gbp_bn": [5.0, -5.2]})
    f = uh.apf_flows(_ops(), fs)
    by = f.groupby("kind")["value_gbp_bn"].sum()
    assert by["purchase"] == pytest.approx(0.11 + 0.05)  # nominal × price / 100, £bn
    assert by["sale"] == pytest.approx(-0.027)
    red = f[f["kind"] == "redemption"]
    # redemption at par on the date in the bond code (7 Jun 2024), not the year-only join
    assert red["date"].tolist() == [pd.Timestamp("2024-06-07")]
    assert red["value_gbp_bn"].iloc[0] == pytest.approx(-0.12)
    assert by["fs"] == pytest.approx(-0.2)
    m = uh.apf_monthly(f).set_index("month")
    assert m.loc["2024-05-01", "apf_face"] == pytest.approx(0.12)
    assert m.loc["2024-06-01", "apf_face"] == pytest.approx(0.0)
    assert m["apf_f"].sum() == pytest.approx(f["value_gbp_bn"].sum())


def test_public_months():
    assert uh.ukea_public_month(pd.Timestamp("2026-06-30")) == pd.Timestamp("2026-09-01")
    assert uh.ukea_public_month(pd.Timestamp("2026-09-30")) == pd.Timestamp("2026-12-01")
    assert uh.pusf_public_month(pd.Timestamp("2026-08-01")) == pd.Timestamp("2026-09-01")
    # RPMB3QP July-2026 'Published: 01 Sep 2026' (31 Aug bank holiday pushes the 21st working day)
    assert uh.bankstats_release(pd.Timestamp("2026-07-01")) == pd.Timestamp("2026-09-01")
    assert uh.bankstats_public_month(pd.Timestamp("2026-07-01")) == pd.Timestamp("2026-09-01")
    assert uh.bankstats_public_month(pd.Timestamp("2026-08-01")) == pd.Timestamp("2026-09-01")
    # Easter (3 and 6 Apr 2026) pushes March into May
    assert uh.bankstats_public_month(pd.Timestamp("2026-03-01")) == pd.Timestamp("2026-05-01")
    hol = uh.uk_holidays(2026)
    assert pd.Timestamp("2026-08-31") in hol and pd.Timestamp("2026-12-28") in hol and pd.Timestamp("2026-05-04") in hol


def test_stamp_no_lookahead():
    q = pd.DataFrame({"country": "UK", "metric": "m", "quarter_end": pd.to_datetime(["2025-12-31", "2026-03-31"]),
                      "value": [1.0, 2.0]})
    q["public"] = [uh.ukea_public_month(x) for x in q["quarter_end"]]
    out = uh._stamp(q, "quarter_end", pd.Timestamp("2026-09-01")).set_index("month")["value"]
    assert out.index.min() == pd.Timestamp("2026-03-01")  # Q4 public in March, not January
    assert out.loc["2026-05-01"] == 1.0 and out.loc["2026-06-01"] == 2.0
    out2 = uh._stamp(q, "quarter_end", pd.Timestamp("2026-05-01"))
    assert (out2["value"] == 1.0).all()


def _boe_fixture(months):
    rows = []
    for m in months:
        for code, (s, meas), v in (("RPMB3QP", uh.BOE_SERIES["RPMB3QP"], 200.0), ("RPMB4CA", uh.BOE_SERIES["RPMB4CA"], 1.0),
                                   ("LPMB9P6", uh.BOE_SERIES["LPMB9P6"], 2.0), ("LPMVRDY", uh.BOE_SERIES["LPMVRDY"], 3.0)):
            rows.append({"month": m, "code": code, "series": s, "measure": meas, "value_gbp_bn": v})
    return pd.DataFrame(rows)


GDP = pd.Series(3000.0, index=pd.date_range("2018-01-01", "2026-04-01", freq="QS"))


def test_monthly_metrics_stamped_and_scaled():
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    apf = pd.DataFrame({"month": months, "apf_f": -1.0, "apf_face": 0.0})
    m = uh.metrics(today=pd.Timestamp("2026-02-15"), qp=_tiny_qp(), boe=_boe_fixture(months), apf=apf, gdp=GDP)
    f12 = m[m["metric"] == "uk_bank_f12m"].set_index("month")["value"]
    # first 12m sum ends Dec-2024 → 21st working day after 31 Dec 2024 is in Jan-2025
    assert f12.index.min() == pd.Timestamp("2025-01-01")
    assert f12.iloc[0] == pytest.approx(100 * 12 * 1.0 / 3000.0)
    ov = m[m["metric"] == "uk_overseas_f3m"].set_index("month")["value"]
    assert ov.iloc[0] == pytest.approx(100 * 3 * 2.0 / 3000.0)
    # APF operations are public the month they happen; padding starts the 12m sum at the first op
    a = m[m["metric"] == "uk_apf_f12m"].set_index("month")["value"]
    assert a.index.min() == pd.Timestamp("2024-01-01") and a.loc["2024-01-01"] == pytest.approx(100 * -1 / 3000.0)
    assert a.loc["2024-12-01"] == pytest.approx(100 * -12 / 3000.0)
    assert m["month"].max() == pd.Timestamp("2026-01-01")
    assert (m["phase"] == "history").all() and set(m["metric"]) <= set(uh.METRICS)


def test_metric_signs():
    assert all(v["sign"] in (1, -1) for v in uh.METRICS.values())
    for k, v in uh.METRICS.items():
        if k.endswith(("_f3m", "_f12m", "_f4q")) and k != "uk_private_f4q":
            assert v["sign"] == -1, k  # net buying = demand
    assert uh.METRICS["uk_private_f4q"]["sign"] == 1
    assert uh.METRICS["uk_overseas_share"]["sign"] == 1 and uh.METRICS["uk_icpf_share"]["sign"] == -1
    assert all({"label", "unit", "sign", "fmt", "what"} <= set(v) for v in uh.METRICS.values())


def _ons_fixture():
    rows = []
    qs = pd.date_range("2023-03-31", periods=8, freq="QE")
    lev = {"icpf": 500.0, "mfi": 600.0, "ofi": 300.0, "overseas": 700.0, "households": 3.0, "nfc": 1.0,
           "local_gov": 0.5, "public_corp": 0.5}
    flo = {"icpf": 10.0, "mfi": -20.0, "ofi": 15.0, "overseas": 8.0, "households": 0.0, "npish": 0.0, "nfc": 0.1,
           "local_gov": 0.0, "public_corp": 0.1}
    for q in qs:
        for s, v in lev.items():
            rows.append({"quarter_end": q, "cdid": s, "sector": s, "measure": "LE", "value_gbp_bn": v})
        rows.append({"quarter_end": q, "cdid": "nyxr", "sector": "uk_total", "measure": "LE",
                     "value_gbp_bn": sum(v for s, v in lev.items() if s != "overseas")})
        for s, v in flo.items():
            rows.append({"quarter_end": q, "cdid": s, "sector": s, "measure": "F", "value_gbp_bn": v})
        uk = sum(v for s, v in flo.items() if s != "overseas")
        rows.append({"quarter_end": q, "cdid": "nyrs", "sector": "uk_total", "measure": "F", "value_gbp_bn": uk})
        rows.append({"quarter_end": q, "cdid": "navt", "sector": "total", "measure": "F", "value_gbp_bn": uk + 8.0})
    return pd.DataFrame(rows)


def _tiny_qp():
    months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    apf = pd.DataFrame({"month": months, "apf_f": -8.0, "apf_face": 0.0})
    own = pd.Series(10.0, index=months)
    return uh.quarterly_panel(_ons_fixture(), apf, _boe_fixture(months), own)


def test_identities_and_split():
    p = _ons_fixture()
    g = uh.sector_sums(p)
    assert len(g) == 8 and g[["gap_level", "gap_flow", "gap_total"]].abs().max().max() < 1e-9
    qp = _tiny_qp()
    assert (qp["apf"] == -24.0).all() and (qp["boe_own"].iloc[1:] == 0.0).all()
    assert np.allclose(qp["banks"], qp["mfi"] - qp["apf"] - qp["boe_own"])
    assert np.allclose(qp["banks"].iloc[1:], 4.0)  # −20 − (−24) − 0
    assert (qp["banks_bankstats"] == 3.0).all()  # three months of RPMB4CA
    assert (qp["banks_le"] == 200.0).all() and (qp["boe_le"] == 400.0).all()
    assert (qp["stock_le"] == 2105.0).all()
    ab = uh.absorption(qp, years=(2023, 2024))
    assert (ab["parts_sum"] - ab["net_issuance"]).abs().max() < 1e-9
    assert ab["overseas"].tolist() == [32.0, 32.0] and ab["apf"].tolist() == [-96.0, -96.0]


def test_quarterly_metrics_values():
    qm = uh.quarterly_metrics(_tiny_qp(), GDP)
    last = qm[qm["quarter_end"] == qm["quarter_end"].max()].set_index("metric")["value"]
    assert last["uk_icpf_share"] == pytest.approx(100 * 500 / 2105)
    assert last["uk_overseas_share"] == pytest.approx(100 * 700 / 2105)
    assert last["uk_icpf_f4q"] == pytest.approx(100 * 40 / 3000)
    assert last["uk_apf_f4q"] == pytest.approx(100 * -96 / 3000)
    assert last["uk_private_f4q"] == pytest.approx(100 * 4 * (13.2 + 24.0) / 3000)  # NAVT − APF − BoE own
    assert set(qm["metric"]) <= set(uh.METRICS)
    pub = qm[qm["metric"] == "uk_icpf_f4q"].set_index("quarter_end")["public"]
    assert pub[pd.Timestamp("2024-06-30")] == pd.Timestamp("2024-09-01")


@pytest.mark.skipif(not uh._ons_path("navt").exists() or not (uh.DIR / "boe_iadb.csv").exists(),
                    reason="no cached ONS / IADB")
def test_cached_identities_and_split():
    g = uh.sector_sums()
    g = g[g["quarter_end"] >= "2000-01-01"]
    assert g["gap_flow"].abs().max() < 0.01 and g["gap_total"].abs().max() < 0.01
    assert g["gap_level"].abs().max() < 3.0
    sc = uh.split_check().set_index("ons")
    assert sc.loc["overseas", "corr"] > 0.99 and sc.loc["banks", "corr"] > 0.9
    assert sc.loc["banks", "mae_bn"] < 3.0
    fc = uh.apf_face_check()
    assert (fc["apf_face"] - fc["pusf"]).abs().median() < 0.5
    m = uh.metrics(today=pd.Timestamp("2026-10-02"))
    assert m["month"].max() == pd.Timestamp("2026-09-01")
    assert set(m["metric"]) == set(uh.METRICS)


def test_cached_ons_json_readable():
    p = uh._ons_path("nbtf")
    if not p.exists():
        pytest.skip("no cached NBTF")
    q, _ = uh.parse_ons(json.loads(p.read_text()))
    assert q.index.min() <= pd.Timestamp("1990-03-31") and np.isfinite(q).all()
