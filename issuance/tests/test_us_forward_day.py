"""US coupon dates: Treasury's close wins over the TBAC day-of-month."""

import pandas as pd

from issuance.forward import _us
from issuance.forward_audit import GRADE, _basis


def _by(rows, isin, month="2026-11"):
    return [r for r in rows if r["isin"] == isin and str(r["op_dt"])[:7] == month]


def test_announced_30y_stays_on_the_close_after_the_model_day():
    # Model day is the 10th. Treasury auctioned the new 30y on the 12th.
    # asof on the morning of the close must still emit that row.
    announced = {("2026-11", ("Nominal", 30)): pd.Timestamp("2026-11-12")}
    rows = _us(pd.Timestamp("2026-11-12"), announced=announced)
    hit = _by(rows, "US 30y")
    assert [r["op_dt"] for r in hit] == [pd.Timestamp("2026-11-12")]
    assert hit[0]["amount_mn"] == 25_000.0
    assert hit[0]["source"] == "treasury"
    assert _basis("US", hit[0]["source"], hit[0]["isin"], hit[0]["op_dt"]) == "announced"
    assert GRADE["announced"] == "High"
    # The 3y (model day the 8th) and 10y (the 9th) already closed. They are not
    # in this map, so they stay on the model day inside the 21-day window.
    assert any(r["isin"] == "US 3y" and r["op_dt"] == pd.Timestamp("2026-11-08") for r in rows)
    assert any(r["isin"] == "US 10y" and r["op_dt"] == pd.Timestamp("2026-11-09") for r in rows)


def test_held_close_is_dropped():
    announced = {
        ("2026-11", ("Nominal", 3)): pd.Timestamp("2026-11-09"),
        ("2026-11", ("Nominal", 10)): pd.Timestamp("2026-11-10"),
        ("2026-11", ("Nominal", 30)): pd.Timestamp("2026-11-12"),
    }
    rows = _us(pd.Timestamp("2026-11-12"), announced=announced)
    assert _by(rows, "US 3y") == []
    assert _by(rows, "US 10y") == []
    assert [r["op_dt"] for r in _by(rows, "US 30y")] == [pd.Timestamp("2026-11-12")]


def test_project_keeps_the_us_close_on_the_origin_and_drops_the_model_day(monkeypatch):
    """`_us` emits the close on asof. `project` must not delete that row."""
    from issuance import forward

    close = forward._row("US 30y", "30y+", "2026-11-12", 25_000.0, "treasury", "Nominal")
    monkeypatch.setattr(forward, "_us", lambda asof, ops=None, announced=None: [close])
    kept = forward.project("US", pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2026-11-12"))
    assert list(kept["op_dt"]) == [pd.Timestamp("2026-11-12")]
    assert list(kept["amount_mn"]) == [25_000.0]
    assert list(kept["source"]) == ["treasury"]

    stale = forward._row("US 30y", "30y+", "2026-11-10", 25_000.0, "tbac", "Nominal")
    monkeypatch.setattr(forward, "_us", lambda asof, ops=None, announced=None: [stale])
    dropped = forward.project("US", pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2026-11-12"))
    assert dropped.empty


def test_feed_down_keeps_the_model_day_inside_21_days():
    rows = _us(pd.Timestamp("2026-11-12"), announced={})
    hit = _by(rows, "US 30y")
    assert [r["op_dt"] for r in hit] == [pd.Timestamp("2026-11-10")]
    assert hit[0]["source"] == "tbac"
    # October's 30y (the 10th) is outside the 21-day window.
    assert _by(rows, "US 30y", "2026-10") == []
