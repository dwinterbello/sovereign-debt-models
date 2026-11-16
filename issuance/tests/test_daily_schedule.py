import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("issuance_daily", ROOT / "scripts" / "issuance_daily.py")
daily = importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)
L = daily.LONDON


def at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=L)


def test_slots_are_weekday_london():
    assert daily.last_slot(at(2026, 10, 1, 12, 5)) == at(2026, 10, 1, 12, 0)
    assert daily.last_slot(at(2026, 10, 1, 11, 59)) == at(2026, 9, 30, 23, 0)
    # Saturday and Sunday fall back to Friday's last slot; Monday morning too.
    assert daily.last_slot(at(2026, 10, 4, 15)) == at(2026, 10, 2, 23, 0)
    assert daily.last_slot(at(2026, 10, 5, 9)) == at(2026, 10, 2, 23, 0)
    assert daily.next_slot(at(2026, 10, 2, 23, 30)) == at(2026, 10, 5, 12, 0)


def test_due_catches_up_and_retries_hourly():
    now = at(2026, 10, 1, 19, 10)
    assert daily.due({}, now)
    assert not daily.due({"last_ok": at(2026, 10, 1, 18, 40).isoformat()}, now)
    # Slept through 18:30: the earlier 12:00 success does not cover it.
    assert daily.due({"last_ok": at(2026, 10, 1, 12, 10).isoformat()}, now)
    failed = {"last_ok": at(2026, 10, 1, 12, 10).isoformat(), "last_attempt": (now - timedelta(minutes=20)).isoformat()}
    assert not daily.due(failed, now)
    failed["last_attempt"] = (now - timedelta(minutes=61)).isoformat()
    assert daily.due(failed, now)
