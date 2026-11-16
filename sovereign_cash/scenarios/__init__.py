"""Scenario layer: shock definitions, the evaluation engine and the scenario pack.

``unit_shocks``  the published one-at-a-time tables (GDP +1pp, CPI +1pp, +100bp)
``engine``       evaluate(pack, cc, shocks): pure-stdlib arithmetic, mirrored in JS
``pack``         builds the versioned scenario pack from the fitted models

The legacy names (``run``, ``table_html``, ...) resolve lazily from
``unit_shocks`` so ``from sovereign_cash import scenarios as SX`` keeps working
without importing statsmodels for code that only needs the engine.
"""
from __future__ import annotations

import importlib

_LEGACY = {"MACRO_SHOCKS", "ROOT", "YEARS_AHEAD", "_origin", "load", "run", "save", "table_html"}


def __getattr__(name: str):
    if name in _LEGACY:
        return getattr(importlib.import_module("sovereign_cash.scenarios.unit_shocks"), name)
    raise AttributeError(f"module 'sovereign_cash.scenarios' has no attribute {name!r}")
