"""The note's generated prose carries no hard-typed quantities; every placeholder resolves."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTE = Path(__file__).resolve().parents[1] / "debt" / "note"
SECTIONS = sorted((NOTE / "sections").glob("*.html"))
NUMBERS = NOTE / "note_numbers.json"
MANIFEST = NOTE / "figures" / "manifest.json"

# Scenario definitions and fixed conventions that may appear as literals outside hand blocks.
ALLOWED = {"+100bp", "−100bp", "-100bp", "+150bp", "−50bp", "-50bp", "+2pp", "+0.5pp", "−0.5pp", "-0.5pp", "−1pp", "-1pp",
           "20%", "80%", "0.2pp", "12m", "12-month", "0–1y", "1y", "1–7y", "7–15y", "15y+", "10y", "2y", "5y", "30y", "40y",
           "20y", "3y", "1–3y", "3–7y", "7–10y", "10–20y", "20y+"}
QUANTITY = re.compile(r"[+−-]?\d+(?:[.,]\d+)?\s?(?:pp|%|bn|bp|×|y\b|tn)")
HAND = re.compile(r"<!-- hand \d{4}-\d{2}-\d{2} -->.*?<!-- /hand -->", re.S)
PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_:]+)(?:\|[^}]+)?\}\}")


def _generated(text: str) -> str:
    text = HAND.sub("", text)
    text = PLACEHOLDER.sub("", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return text


@pytest.mark.parametrize("path", SECTIONS, ids=lambda p: p.name)
def test_no_bare_quantities_outside_hand_blocks(path: Path):
    body = _generated(path.read_text(encoding="utf-8"))
    hits = [m.group(0) for m in QUANTITY.finditer(body) if m.group(0).replace(" ", "") not in ALLOWED]
    assert not hits, f"{path.name}: hard-typed quantities {hits}"


def test_hand_blocks_are_dated_and_closed():
    for p in SECTIONS:
        s = p.read_text(encoding="utf-8")
        assert s.count("<!-- hand ") == s.count("<!-- /hand -->"), p.name
        assert "<!-- hand -->" not in s, f"{p.name}: undated hand block"


@pytest.mark.skipif(not NUMBERS.exists(), reason="note not built")
def test_every_placeholder_resolves():
    N = json.load(open(NUMBERS))
    figs = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
    missing = []
    for p in SECTIONS:
        for m in PLACEHOLDER.finditer(p.read_text(encoding="utf-8")):
            key = m.group(1)
            if key.startswith("fig:"):
                if key[4:] not in figs:
                    missing.append((p.name, key))
            elif key.startswith(("table:", "block:")):
                continue
            elif key not in N:
                missing.append((p.name, key))
    assert not missing, missing


@pytest.mark.skipif(not NUMBERS.exists(), reason="note not built")
def test_numbers_come_from_the_engine_not_the_imf():
    """Ex1/Ex5 primaries are the engine's general-government primary, so pb_now equals −π^GG at the forecast year."""
    N = json.load(open(NUMBERS))
    y = N["y_fc"]
    for cc in ("us", "uk", "de", "fr", "it", "jp"):
        assert abs(N[f"pb_now_{cc}"] + N[f"pb_gg_{cc}_{y}"]) < 1e-3, cc  # json rounds to 4dp


@pytest.mark.skipif(not MANIFEST.exists(), reason="note not built")
def test_static_figures_are_the_only_hand_built_ones():
    figs = json.load(open(MANIFEST))
    static = {k for k, v in figs.items() if v["source"].startswith("static")}
    assert static <= {"ex1d", "ex10"}, static
    for k, v in figs.items():
        assert (NOTE / v["file"]).exists(), k
