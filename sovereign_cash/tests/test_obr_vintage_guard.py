"""The OBR refresh never steps back to an older EFO than it already holds (2 Nov 2026 regression)."""
from __future__ import annotations

import pandas as pd

from sovereign_cash.debt import sources as S


def _cache(tmp_path, vintage: str) -> None:
    pd.DataFrame([{"country": "uk", "source": "obr", "vintage": vintage, "basis": "fy", "year": 2026,
                   "var": "deficit_bn", "value": 133.2, "unit": "GBP bn"}]).to_parquet(tmp_path / "obr.parquet")


def test_older_efo_keeps_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RAW", tmp_path)
    _cache(tmp_path, "OBR EFO October 2026")
    monkeypatch.setattr(S, "_obr_pdf", lambda: (b"%PDF-older", "March 2026"))
    got = S.obr(refresh=True)
    assert set(got["vintage"]) == {"OBR EFO October 2026"}
    assert not (tmp_path / "obr_efo.pdf").exists()


def test_vintage_dates_order():
    assert S._vintage_date("OBR EFO October 2026") > S._vintage_date("March 2026")
    assert S._vintage_date("nonsense") is None
