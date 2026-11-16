from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from issuance import fetch_jsda as js

HDR = ["国債\nGovernment Bonds", None, None, None, None, None]
SUB = [None, "超長期\nInterest-bearing Long-term (over 10-year)", "利付長期\nInterest-bearing Long-term",
       "利付中期\nInterest-bearing Medium-term", "割引\nZero-Coupon", "国庫短期証券等\nTreasury Discount Bills"]


def _sheet(ws, rows):
    ws.append(["（単位：億円)\n(100 million yen)", "売買の別", None, "差引"])
    ws.append([None, "売買の種類", None, "一般売買"])
    ws.append(["年/月\nYear/Month", "債券の種類", None, *HDR])
    ws.append([None, None, None, *SUB])
    for r in rows:
        ws.append(r)


def _fixture(path):
    """Two months × three investors in the FY workbook layout (¥100mn, net = sale − purchase)."""
    from openpyxl import Workbook

    sale = {("2025/04", "生保・損保"): [1, 2, 3, 0, 4], ("2025/04", "外国人"): [10, 20, 30, 0, 400],
            ("2025/05", "生保・損保"): [5, 0, 0, 0, 0], ("2025/05", "外国人"): [0, 0, 0, 0, 0]}
    buy = {("2025/04", "生保・損保"): [101, 12, 3, 0, 4], ("2025/04", "外国人"): [10, 40, 30, 0, 900],
           ("2025/05", "生保・損保"): [55, 0, 0, 0, 0], ("2025/05", "外国人"): [0, 0, 7, 0, 0]}
    en = {"生保・損保": "Life & Non-Life Insurance Companies", "外国人": "Foreigners"}
    wb = Workbook()
    wb.active.title = "本表について"
    for name, vals in (("(Ｅ)一般売付額", sale), ("(Ｈ)一般買付額", buy), ("(Ｋ)一般差引", None),
                       ("(ｋ)一般差引【証券】", None)):
        ws = wb.create_sheet(name)
        rows = []
        for (mo, inv) in sale:
            if vals is None:
                v = [s - b for s, b in zip(sale[(mo, inv)], buy[(mo, inv)])]
                if "証券" in name:
                    v = [999] * 5  # securities-companies-only decoy must be ignored
            else:
                v = vals[(mo, inv)]
            rows.append([mo, inv, en[inv], sum(v), *v])
        tot = {}
        for (mo, inv), _ in sale.items():
            tot.setdefault(mo, np.zeros(5))
        for (mo, inv) in sale:
            v = (np.array(sale[(mo, inv)]) - np.array(buy[(mo, inv)])) if vals is None else np.array(vals[(mo, inv)])
            tot[mo] = tot[mo] + (999 if (vals is None and "証券" in name) else v)
        for mo, v in tot.items():
            rows.append([mo, "合計", "Total", float(v.sum()), *v.tolist()])
        _sheet(ws, rows)
    wb.save(path)


@pytest.fixture()
def fy(tmp_path):
    p = tmp_path / "koushasai2025.xlsx"
    _fixture(p)
    return p


def test_fy_parse_units_sign_and_buckets(fy):
    t = js.tidy(js.parse_fy(fy), "fixture")
    one = t.set_index(["month", "investor", "maturity"])
    # insurers 2025-04 super-long: sale 1, purchase 101 (¥100mn) → +10.0 ¥bn net buying
    r = one.loc[(pd.Timestamp("2025-04-01"), "insurers", "super_long")]
    assert r["net_purchase_jpy_bn"] == pytest.approx(10.0)
    assert r["purchases_jpy_bn"] == pytest.approx(10.1)
    assert r["sales_jpy_bn"] == pytest.approx(0.1)
    # foreigners bills: sale 400, purchase 900 → +50 ¥bn; decoy 【証券】 sheet ignored
    assert one.loc[(pd.Timestamp("2025-04-01"), "foreigners", "tbills"), "net_purchase_jpy_bn"] == pytest.approx(50.0)
    # maturity buckets sum to the JGB total
    w = t.pivot_table(index=["month", "investor"], columns="maturity", values="net_purchase_jpy_bn")
    assert np.allclose(w["total_jgb"], w[list(js.MATURITIES)].sum(axis=1))
    assert set(t["investor"]) == {"insurers", "foreigners", "total"}


def test_net_is_published_sale_minus_purchase(fy):
    long = js.parse_fy(fy)
    w = long.pivot_table(index=["month", "investor", "maturity"], columns="side", values="value_100mn")
    assert np.allclose(w["net"], w["sale"] - w["purchase"])


def test_month_and_investor_labels():
    assert js._month("2025/04") == pd.Timestamp("2025-04-01")
    assert js._month(pd.Timestamp("2019-07-31")) == pd.Timestamp("2019-07-01")
    assert js._ichiran_month(2004.1) == pd.Timestamp("2004-10-01")
    assert js._ichiran_month("2004.04") == pd.Timestamp("2004-04-01")
    assert js._investor("都市銀行_x000D_\n(長信銀等を含む。)") == "city_banks"
    assert js._investor("第二地銀協加盟行") == "regional2"
    assert js._investor("その他金融機関") == "other_fin"
    assert js._investor("その他") == "others"
    assert js._investor("生保・損保\nInsurance Companies") == "insurers"
    assert js._maturity("割引短期国債\nTreasury Bills") == "tbills"
    assert js._maturity("割引_x000D_\nZero-Coupon") == "zero"


def _synthetic(n=30, start="2023-01-01"):
    rng = np.random.default_rng(0)
    months = pd.date_range(start, periods=n, freq="MS")
    rows = [{"month": mo, "investor": inv, "maturity": m, "net_purchase_jpy_bn": rng.normal(),
             "purchases_jpy_bn": np.nan, "sales_jpy_bn": np.nan, "source": "synthetic"}
            for mo in months for inv in ("insurers", "trust_banks", "city_banks", "foreigners")
            for m in js.MATURITIES]
    return pd.DataFrame(rows)


def test_publication_stamp_no_lookahead():
    t = _synthetic()
    m = js.metrics(t, today=pd.Timestamp("2026-10-02"))
    data = js.monthly_metrics(t)
    assert (m["month"].min() > data["month"].min())
    assert js.public_month(pd.Timestamp("2025-05-01")) == pd.Timestamp("2025-06-01")
    # a value stamped at month s uses data through s − 1 only
    m3 = m[m["metric"] == "jsda_ins_sl_3m"].set_index("month")["value"]
    d3 = data[data["metric"] == "jsda_ins_sl_3m"].set_index("month")["value"]
    for s, v in m3.items():
        assert v == d3.loc[s - pd.DateOffset(months=1)]
    # nothing is stamped at or after the current month
    assert js.metrics(t, today=pd.Timestamp("2024-06-15"))["month"].max() <= pd.Timestamp("2024-05-01")


def test_metrics_are_backward_looking_only():
    t = _synthetic()
    full = js.monthly_metrics(t)
    cut = js.monthly_metrics(t[t["month"] <= "2024-04-01"])
    assert len(cut)
    a = full[full["month"] <= "2024-04-01"].set_index(["metric", "month"])["value"]
    b = cut.set_index(["metric", "month"])["value"]
    common = a.index.intersection(b.index)
    assert np.allclose(a.loc[common], b.loc[common])


class _Resp:
    def __init__(self, code, body):
        self.status_code, self.content = code, body


def test_fetch_rejects_429_and_html(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "DIR", tmp_path)
    calls = []

    def get429(url, referer):
        calls.append(url)
        return _Resp(429, b"<html><head><title>429 Too Many Requests</title></head></html>")

    with pytest.raises(js.JsdaError, match="429"):
        js.fetch(files=["koushasai2025.xlsx", "koushasai2024.xlsx"], wayback=False, sleep=0, get=get429)
    assert len(calls) == 1  # a 429 stops further JSDA requests
    assert not list(tmp_path.glob("*.xlsx"))

    def get_html(url, referer):
        return _Resp(200, b"<!DOCTYPE html><html>" + b" " * 6000)

    with pytest.raises(js.JsdaError, match="not a workbook"):
        js.fetch(files=["kokusaiichiran.xls"], wayback=False, sleep=0, get=get_html)
    assert not list(tmp_path.glob("*.xls*"))


def test_fetch_writes_valid_and_skips_cached(tmp_path, monkeypatch, fy):
    monkeypatch.setattr(js, "DIR", tmp_path / "c")
    body = fy.read_bytes()
    calls = []

    def ok(url, referer):
        calls.append(url)
        return _Resp(200, body)

    got = js.fetch(files=[js.CURRENT, "koushasai2025.xlsx"], sleep=0, get=ok)
    assert set(got) == {js.CURRENT, "koushasai2025.xlsx"} and len(calls) == 2
    js.fetch(files=[js.CURRENT, "koushasai2025.xlsx"], sleep=0, get=ok)
    assert len(calls) == 2  # valid cache, no request
    js.fetch(refresh=True, files=[js.CURRENT, "koushasai2025.xlsx"], sleep=0, get=ok)
    assert calls[-1].endswith("tentoubaibai/koushasai.xlsx") and len(calls) == 3  # only the current FY


def test_fetch_falls_back_to_wayback(tmp_path, monkeypatch, fy):
    monkeypatch.setattr(js, "DIR", tmp_path / "c")
    body = fy.read_bytes()

    def get(url, referer):
        if "web.archive.org" in url:
            assert "id_/" in url
            return _Resp(200, body)
        return _Resp(429, b"<html>429</html>")

    got = js.fetch(files=["koushasai2025.xlsx"], sleep=0, get=get)
    assert "koushasai2025.xlsx" in got
    assert "wayback" in (tmp_path / "c" / "koushasai2025.xlsx.source").read_text()


CACHED = js.DIR / "kokusaiichiran.xls"


@pytest.mark.skipif(not CACHED.exists() or not (js.DIR / "koushasai2025.xlsx").exists(), reason="JSDA cache absent")
def test_cached_files_known_cells_and_identities():
    p = js.build()
    one = p.set_index(["month", "investor", "maturity"])["net_purchase_jpy_bn"]
    # kokusaiichiran 2004-04 city banks super-long: published net (sale − purchase) = 3361 ¥100mn
    assert one.loc[(pd.Timestamp("2004-04-01"), "city_banks", "super_long")] == pytest.approx(-336.1)
    # koushasai2025 2025-04 city banks super-long net 5179; foreigners JGB total −255337
    assert one.loc[(pd.Timestamp("2025-04-01"), "city_banks", "super_long")] == pytest.approx(-517.9)
    assert one.loc[(pd.Timestamp("2025-04-01"), "foreigners", "total_jgb")] == pytest.approx(25533.7)
    w = js.wide(p)
    tot = w.xs("total_jgb", axis=1, level=1)
    parts = sum(w.xs(m, axis=1, level=1).fillna(0) for m in js.MATURITIES)
    assert float((tot - parts).abs().max().max()) < 1.0
    months = pd.date_range("2004-04-01", p["month"].max(), freq="MS")
    assert set(months) <= set(p["month"])
    # each month comes from exactly one file
    assert p.groupby("month")["source"].nunique().max() == 1
