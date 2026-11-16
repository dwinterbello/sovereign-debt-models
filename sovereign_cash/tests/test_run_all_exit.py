"""Exit codes when optional dashboard stages miss warm derived caches."""
from __future__ import annotations

from sovereign_cash import run_all


def test_optional_stages_frozenset():
    assert run_all._OPTIONAL_DASHBOARD_STAGES == frozenset({"fytd", "scenarios"})


def test_dashboard_only_degraded_when_only_optional_fail(monkeypatch):
    """Warm derived + missing float_share_fc → hubs build, scenarios skip → exit 2."""
    calls: list[str] = []

    import sovereign_cash.fytd_page as fytd_page
    import sovereign_cash.methods_page as mp

    monkeypatch.setattr(fytd_page, "build", lambda: (_ for _ in ()).throw(FileNotFoundError("tga")))
    monkeypatch.setattr(run_all, "scenarios", lambda: (_ for _ in ()).throw(FileNotFoundError("fc")))
    monkeypatch.setattr(mp, "build", lambda: calls.append("methods"))
    monkeypatch.setattr(run_all, "build", lambda refresh=False: calls.append("hub"))
    monkeypatch.setattr(run_all, "build_financing", lambda: calls.append("financing"))
    monkeypatch.setattr(run_all, "_assert_methodology_hub", lambda: None)

    rc = run_all._dashboard_only()
    assert rc == 2
    assert "hub" in calls and "financing" in calls