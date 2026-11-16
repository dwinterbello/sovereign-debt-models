"""AFT next-issuance communiqués → upcoming auction rows for the FR forward / Today strip."""

from __future__ import annotations

import pandas as pd
import pytest

from issuance.fetch_fr import (
    _aft_next_issuance_urls,
    _fr_ann_amount,
    parse_aft_next_issuance_text,
)
from issuance.forward import _fr_announced, _fr_drop_pattern_covered, _row

FIXTURE_OAT = """
On Thursday 15 October 2026, at 10.50 a.m., Paris time, Agence France Trésor will auction the following securities:
OAT
25/02/2031
2.70%
OAT
25/11/2031
0.00%
OAT
25/02/2032
3.25%
OAT
25/11/2033
3.50%
Auction date
15/10/2026
15/10/2026
15/10/2026
15/10/2026
Settlement date
19/10/2026
19/10/2026
19/10/2026
19/10/2026
Volume announced*
<
10,000 -- 12,000
>
ISIN Code
FR001400Z2L7
FR0014002WK3
FR0014018OI0
FR001400L834
* In millions of euros
"""

FIXTURE_INDEXED = """
On Thursday 15 October 2026, at 11.50 a.m., Paris time, Agence France Trésor will auction the following securities:
OAT€i
25/07/2032
3.15%
OATi
01/03/2039
0.55%
OAT€i
25/07/2040
1.80%
Auction date
15/10/2026
15/10/2026
15/10/2026
Settlement date
19/10/2026
19/10/2026
19/10/2026
Volume announced*
<
750 -- 1,250
>
ISIN Code
FR0000188799
FR001400IKW5
FR0010447367
"""

FIXTURE_BTF = """
On Monday 26 October 2026, at 2.50 p.m., Paris time, Agence France Trésor will auction the following securities:
BTF
3 month
BTF
6 month
BTF
6 month
BTF
12 month
Auction date
26/10/2026
26/10/2026
26/10/2026
26/10/2026
Issue
14 weeks
25 weeks
27 weeks
49 weeks
Settlement date
28/10/2026
28/10/2026
28/10/2026
28/10/2026
Maturity
03/02/2027
21/04/2027
05/05/2027
06/10/2027
Volume announced*
< 2,900 -- 3,300 >
< 200 -- 600 >
< 1,100 -- 1,500 >
< 800 -- 1,200 >
ISIN Code
FR0129886125
FR0129570620
FR0129886174
FR0129886208
"""


def test_fr_ann_amount_strips_commas():
    assert _fr_ann_amount("10,000") == 10000.0
    assert _fr_ann_amount("1,250") == 1250.0


def test_parse_oat_pool_splits_equally():
    df = parse_aft_next_issuance_text(FIXTURE_OAT, source="fixture")
    assert list(df["isin"]) == [
        "FR001400Z2L7", "FR0014002WK3", "FR0014018OI0", "FR001400L834"]
    assert (df["sec_type"] == "Nominal").all()
    assert (df["op_dt"] == pd.Timestamp("2026-10-15")).all()
    assert float(df["amount_mn"].iloc[0]) == 2750.0  # mid 11_000 / 4
    assert float(df["amount_min_mn"].iloc[0]) == 2500.0


def test_parse_indexed_linkers():
    df = parse_aft_next_issuance_text(FIXTURE_INDEXED, source="fixture")
    assert list(df["sec_type"]) == ["Linker", "Linker", "Linker"]
    assert float(df["amount_mn"].sum()) == pytest.approx(1000.0)  # mid 1000 pool


def test_parse_btf_per_line_ranges():
    df = parse_aft_next_issuance_text(FIXTURE_BTF, source="fixture")
    assert list(df["sec_type"]) == ["Bill"] * 4
    assert float(df.loc[df["isin"].eq("FR0129886125"), "amount_mn"].iloc[0]) == 3100.0
    assert (df["maturity_dt"] == pd.Timestamp("2027-02-03")).any()


def test_aft_next_issuance_urls_from_press_index():
    html = """
    <a href="/en/publications/communiques-presse/09-october-2026-next-issuance-oats">oat</a>
    <a href="/en/publications/communiques-presse/15-october-2026-issuance-oats">result</a>
    <a href="/fr/publications/communiques-presse/09-octobre-2026-prochaine-emission-oat">fr</a>
    """
    urls = _aft_next_issuance_urls(html)
    assert any(u.endswith("09-october-2026-next-issuance-oats") for u in urls)
    assert not any("issuance-oats" in u and "next-issuance" not in u for u in urls)
    assert any("prochaine-emission" in u for u in urls)


def test_drop_pattern_on_announced_auction_day():
    ann = [_row("FR001400Z2L7", "3-7y", "2026-10-15", 2750.0, "calendar", "Nominal")]
    pattern = [
        _row("FR001400Z2L7", "3-7y", "2026-10-15", 2000.0, "pattern", "Nominal"),
        _row("FR001400AAAA", "10-15y", "2026-10-15", 3000.0, "pattern", "Nominal"),
        _row("FR001400BBBB", "10-15y", "2026-11-05", 3000.0, "pattern", "Nominal"),
    ]
    kept = _fr_drop_pattern_covered(pattern, ann)
    assert {r["isin"] for r in kept} == {"FR001400BBBB"}


def test_fr_announced_skips_bills_and_past(tmp_path, monkeypatch):
    import issuance.paths as paths

    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({
        "op_dt": pd.to_datetime(["2026-10-15", "2026-10-26", "2026-10-10"]),
        "isin": ["FR001400Z2L7", "FR0129886125", "FR001400IKW5"],
        "maturity_dt": pd.to_datetime(["2031-02-25", "2027-02-03", "2039-03-01"]),
        "sec_type": ["Nominal", "Bill", "Linker"],
        "amount_mn": [2750.0, 3100.0, 333.0],
        "amount_min_mn": [2500.0, 2900.0, 250.0],
        "amount_max_mn": [3000.0, 3300.0, 416.0],
        "source": ["t"] * 3,
    }).to_parquet(cache / "fr_aft_announcements.parquet", index=False)
    monkeypatch.setattr(paths, "CACHE", cache)
    rows = _fr_announced(pd.Timestamp("2026-10-14"))
    assert {r["isin"] for r in rows} == {"FR001400Z2L7"}
    assert rows[0]["source"] == "calendar"
    assert rows[0]["amount_mn"] == 2750.0
