"""The note must read the repricing file debt.build writes.

``sovereign_cash/debt/derived/debt_repricing.parquet`` is a stale copy. It has
no Spain row. Looking Spain up there raised KeyError and failed the
sovereign_cash job after ES joined the annual table.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_stubs() -> None:
    if "sovereign_cash.debt.note.shock_gg" in sys.modules:
        return

    def mod(name: str, path: Path | None = None) -> types.ModuleType:
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        if path is not None:
            m.__path__ = [str(path)]
            m.__package__ = name
        return m

    root = Path(__file__).resolve().parents[1]
    mpl = mod("matplotlib")
    mpl.use = lambda *a, **k: None
    mod("matplotlib.pyplot")
    mod("numpy")
    mod("pandas")
    mod("sovereign_cash", root)
    mod("sovereign_cash.debt", root / "debt")
    mod("sovereign_cash.debt.note", root / "debt" / "note")
    mod("sovereign_cash.debt.note.rebuild_ex7_repression")


def test_repricing_path_prefers_the_file_debt_build_writes(tmp_path, monkeypatch):
    _install_stubs()
    from sovereign_cash.debt.note import shock_gg as sg

    derived = tmp_path / "derived"
    legacy = tmp_path / "debt" / "derived"
    derived.mkdir()
    legacy.mkdir(parents=True)
    fresh = derived / "debt_repricing.parquet"
    stale = legacy / "debt_repricing.parquet"
    fresh.write_bytes(b"fresh")
    stale.write_bytes(b"stale")
    monkeypatch.setattr(sg, "SC", tmp_path)
    assert sg.repricing_path() == fresh


def test_repricing_path_falls_back_when_canonical_is_absent(tmp_path, monkeypatch):
    _install_stubs()
    from sovereign_cash.debt.note import shock_gg as sg

    legacy = tmp_path / "debt" / "derived"
    legacy.mkdir(parents=True)
    stale = legacy / "debt_repricing.parquet"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(sg, "SC", tmp_path)
    assert sg.repricing_path() == stale


def test_rate_table_source_skips_a_country_missing_from_one_frame():
    """The skip is the line that used to be an unguarded .at[cc]."""
    src = Path(__file__).resolve().parents[1].joinpath("debt", "note", "shock_gg.py").read_text()
    assert "repricing_path()" in src
    assert 'SC / "debt" / "derived" / "debt_repricing.parquet"' not in src.split("def repricing_path", 1)[0]
    assert "if cc not in b.index or cc not in cur.index or cc not in sh.index" in src
