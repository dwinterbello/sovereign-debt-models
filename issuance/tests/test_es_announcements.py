"""Tesoro calendar + BdE BON PDFs → upcoming auction rows for the ES forward / Today strip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from issuance.fetch_es import parse_bde_announcement_text, parse_tesoro_calendar_text
from issuance.forward import _es_announced, _es_drop_pattern_covered, _row

FIXTURE_BON = """
BONOS Y OBLIGACIONES DEL ESTADO
Emisiones para el  20 de octubre de 2026
DEUDA DEL ESTADO
SEGREGABLE
Bono a 3 años
al 2,35%
ISIN: ES0000012P90
TRAMO 09
Bono a 5 años
al 2,60%
ISIN: ES0000012Q16
TRAMO 08
Obligación a 10 años
(VR 7) al 3,55%
ISIN: ES0000012L78
TRAMO 09
Emisión 20.01.2026 24.02.2026 14.06.2023
Anotación del tramo 20.10.2026 20.10.2026 20.10.2026
Amortización  31.03.2029 31.05.2031 31.10.2033
Resolución subasta 15.10.2026 15.10.2026 15.10.2026
"""

FIXTURE_CAL = """
Auctions 13/01/2026 20/01/2026 08/01/2026* 15/01/2026
Auctions 06/10/2026 13/10/2026 01/10/2026* 15/10/2026
Auctions 03/11/2026 10/11/2026 05/11/2026* 19/11/2026
"""

RAW_BON = Path(__file__).resolve().parents[2] / "data" / "raw" / "es" / "bde_subastas"


def test_parse_bde_resolucion_is_auction_day():
    df = parse_bde_announcement_text(FIXTURE_BON, source="fixture")
    assert list(df["isin"]) == ["ES0000012P90", "ES0000012Q16", "ES0000012L78"]
    assert (df["op_dt"] == pd.Timestamp("2026-10-15")).all()
    assert (df["settle_dt"] == pd.Timestamp("2026-10-20")).all()
    assert (df["maturity_dt"] == pd.Timestamp("2029-03-31")).any()


def test_parse_tesoro_calendar_takes_bonos_columns():
    df = parse_tesoro_calendar_text(FIXTURE_CAL)
    assert set(df["op_dt"]) == {
        pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-15"),
        pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-15"),
        pd.Timestamp("2026-11-05"), pd.Timestamp("2026-11-19"),
    }


def test_drop_pattern_covered_by_bde_lines():
    ann = [_row("ES0000012P90", "1-3y", "2026-10-15", 2000.0, "calendar", "Nominal")]
    pattern = [
        _row("ES0000012P90", "1-3y", "2026-10-15", 1800.0, "pattern", "Nominal"),
        _row("ES0000012Q16", "3-7y", "2026-10-15", 2200.0, "pattern", "Nominal"),
        _row("ES0000012AAAA", "10-15y", "2026-11-05", 2500.0, "pattern", "Nominal"),
    ]
    kept = _es_drop_pattern_covered(pattern, ann)
    assert {r["isin"] for r in kept} == {"ES0000012AAAA"}


def test_cached_bon_pdf_parses_when_present():
    path = RAW_BON / "BON_OBL_15102026.pdf"
    if not path.exists():
        return
    from pypdf import PdfReader

    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    df = parse_bde_announcement_text(text, source=path.name)
    assert set(df["isin"]) >= {"ES0000012P90", "ES0000012Q16", "ES0000012L78"}
    assert (df["op_dt"] == pd.Timestamp("2026-10-15")).all()


def test_es_announced_uses_last_tap(tmp_path, monkeypatch):
    import issuance.paths as paths

    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({
        "op_dt": pd.to_datetime(["2026-11-05", "2026-10-15"]),
        "settle_dt": pd.to_datetime(["2026-11-10", "2026-10-20"]),
        "isin": ["ES0000012P90", "ES0000012Q16"],
        "maturity_dt": pd.to_datetime(["2029-03-31", "2031-05-31"]),
        "sec_type": ["Nominal", "Nominal"],
        "coupon": [2.35, 2.60],
        "source": ["t"] * 2,
    }).to_parquet(cache / "es_bde_announcements.parquet", index=False)
    monkeypatch.setattr(paths, "CACHE", cache)
    ops = pd.DataFrame({
        "isin": ["ES0000012P90", "ES0000012P90"],
        "op_dt": pd.to_datetime(["2026-09-03", "2026-10-01"]),
        "amount_issued": [1800.0, 2100.0],
        "op_type": ["Auction", "Auction"],
    })
    rows = _es_announced(pd.Timestamp("2026-10-26"), ops)
    assert {r["isin"] for r in rows} == {"ES0000012P90"}
    assert rows[0]["amount_mn"] == 2100.0
    assert rows[0]["source"] == "calendar"
