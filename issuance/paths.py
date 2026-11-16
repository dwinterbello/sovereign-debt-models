"""Portable paths for the issuance stack. Raw caches stay local; derived is rebuildable."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data"
RAW = DATA / "raw"
CACHE = HERE / "cache"
DERIVED = HERE / "derived"
DASHBOARD = HERE / "issuance_dv01_dashboard.html"

for _path in (RAW, CACHE, DERIVED, RAW / "aft", RAW / "mef", RAW / "de", RAW / "uk", RAW / "us", RAW / "ecb"):
    _path.mkdir(parents=True, exist_ok=True)
