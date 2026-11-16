"""BdI ANNUNCIO-EMISSIONE PDFs → upcoming auction rows for the IT forward / Today strip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from issuance.fetch_it import (
    _bdi_annuncio_urls,
    _it_ann_amount,
    parse_bdi_announcement_text,
)
from issuance.forward import _it_announced, _it_drop_pattern_covered, _row

FIXTURE_22 = """
ANNUNCIO OPERAZIONI DI EMISSIONE TITOLI DI STATO DEL GIORNO 22/10/2026

BTP 3%
IT0005729733 / 5
28/08/2026 - 30/10/2028
BTPI 1,25%
IT0005138828 / 32
15/09/2015 - 15/09/2032
BTPI 2,35%
IT0003745541 / 27
15/09/2004 - 15/09/2035
Importo offerto (mln. €) 1.500/2.000 500/750 500/750
Cedole int. lordi sem. 0,516393% (SHORT) 0,625% 1,175%
Calendario Operativo
Termine prenot.
Pubblico 26/10/2026 26/10/2026 26/10/2026
Data asta 27/10/2026 27/10/2026 27/10/2026
Ora cut-off asta 11:00 11:00 11:00
Data regolamento 29/10/2026 29/10/2026 29/10/2026
"""

FIXTURE_26 = """
ANNUNCIO OPERAZIONI DI EMISSIONE TITOLI DI STATO DEL GIORNO 26/10/2026

BTP 3,5%
IT0005580094 / 8
16/01/2024 - 15/02/2031
BTP 3,95%
IT0005733636 / 3
01/10/2026 - 01/02/2032
BTP 4%
IT0005729931 / 5
01/09/2026 - 01/10/2036
Importo offerto (mln. €) 750/1.250 1.250/1.750 3.000/3.500
Data asta 29/10/2026 29/10/2026 29/10/2026

CCTEU
IT0005707689 / 11
15/04/2026 - 15/04/2036
Importo offerto (mln. €) 1.000/1.250
Data asta 29/10/2026
"""

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "mef" / "bdi" / "annunci"


def test_it_ann_amount_italian_thousands_and_range():
    assert _it_ann_amount("1.500/2.000") == (1500.0, 2000.0, 1750.0)
    assert _it_ann_amount("500/750") == (500.0, 750.0, 625.0)
    assert _it_ann_amount("2.500") == (2500.0, 2500.0, 2500.0)


def test_parse_btpei_offering_22_oct():
    df = parse_bdi_announcement_text(FIXTURE_22, source="fixture")
    assert list(df["isin"]) == ["IT0005729733", "IT0005138828", "IT0003745541"]
    assert list(df["sec_type"]) == ["Nominal", "Linker", "Linker"]
    assert (df["op_dt"] == pd.Timestamp("2026-10-27")).all()
    assert df.loc[df["isin"].eq("IT0005138828"), "amount_mn"].iloc[0] == 625.0
    assert df.loc[df["isin"].eq("IT0005729733"), "amount_mn"].iloc[0] == 1750.0


def test_parse_multi_block_with_ccteu():
    df = parse_bdi_announcement_text(FIXTURE_26, source="fixture")
    assert len(df) == 4
    assert df["isin"].tolist()[-1] == "IT0005707689"
    assert df["sec_type"].tolist()[-1] == "FRN"
    assert float(df.loc[df["isin"].eq("IT0005729931"), "amount_mn"].iloc[0]) == 3250.0
    assert (df["op_dt"] == pd.Timestamp("2026-10-29")).all()


def test_bdi_annuncio_urls_from_index_html():
    html = '''
    <a href="/compiti/operazioni-mef/annunci-operazioni/documenti/2026/ANNUNCIO-EMISSIONE-DEL-22.10.2026.pdf
">x</a>
    <a href="/compiti/.../ANNUNCIO-RIACQUISTO-DEL-02.10.2026.pdf">buyback</a>
    <a href="/compiti/operazioni-mef/annunci-operazioni/documenti/2026/Annuncio-emissioni-del-07.09.2026.pdf.pdf">dup ext</a>
    '''
    urls = _bdi_annuncio_urls(html)
    assert any(u.rstrip("/").endswith("ANNUNCIO-EMISSIONE-DEL-22.10.2026.pdf") for u in urls)
    assert not any("RIACQUISTO" in u for u in urls)
    assert any(u.endswith("Annuncio-emissioni-del-07.09.2026.pdf.pdf") for u in urls)


def test_drop_pattern_covered_by_announced_linker():
    ann = [_row("IT0005138828", "3-7y", "2026-10-27", 625.0, "calendar", "Linker")]
    ann[0]["maturity_dt"] = pd.Timestamp("2032-09-15")
    pattern = [
        _row("IT BTP€i 10-15y", "10-15y", "2026-10-25", 1000.0, "pattern", "Linker"),
        _row("IT0005722845", "3-7y", "2026-10-28", 1100.0, "pattern", ""),
        _row("IT0005138828", "3-7y", "2026-11-12", 600.0, "pattern", "Linker"),
    ]
    kept = _it_drop_pattern_covered(pattern, ann)
    isins = [r["isin"] for r in kept]
    assert "IT BTP€i 10-15y" not in isins
    assert "IT0005138828" not in isins
    assert "IT0005722845" in isins


def test_cached_pdf_parses_when_present():
    path = RAW / "ANNUNCIO-EMISSIONE-DEL-22.10.2026.pdf"
    if not path.exists():
        return
    from pypdf import PdfReader

    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    df = parse_bdi_announcement_text(text, source=path.name)
    assert set(df["isin"]) >= {"IT0005138828", "IT0003745541", "IT0005729733"}
    assert (df["op_dt"] == pd.Timestamp("2026-10-27")).all()


def test_it_announced_skips_bills_and_past(tmp_path, monkeypatch):
    import issuance.paths as paths

    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({
        "op_dt": pd.to_datetime(["2026-10-27", "2026-10-28", "2026-10-20"]),
        "isin": ["IT0005138828", "IT0005684888", "IT0003745541"],
        "maturity_dt": pd.to_datetime(["2032-09-15", "2026-12-14", "2035-09-15"]),
        "sec_type": ["Linker", "Bill", "Linker"],
        "amount_mn": [625.0, 2500.0, 625.0],
        "amount_min_mn": [500.0, 2500.0, 500.0],
        "amount_max_mn": [750.0, 2500.0, 750.0],
        "source": ["t"] * 3,
    }).to_parquet(cache / "it_bdi_announcements.parquet", index=False)
    monkeypatch.setattr(paths, "CACHE", cache)
    rows = _it_announced(pd.Timestamp("2026-10-26"))
    assert {r["isin"] for r in rows} == {"IT0005138828"}
    assert rows[0]["source"] == "calendar"
    assert rows[0]["amount_mn"] == 625.0
