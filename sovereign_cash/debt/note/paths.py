"""Code lives under ``sovereign_cash/debt/note/``; one-off report *deliverables* under ``reports/``.

The Word/PDF/HTML outlook is a desk report, not a production board. Builders write there;
``portal.catalog`` may still publish the HTML. Docx is local/Downloads only — gitignored.
"""
from __future__ import annotations

from pathlib import Path

NOTE = Path(__file__).resolve().parent
REPO = NOTE.parents[2]
REPORT = REPO / "reports" / "dm_sovereign_debt_outlook"
ARCHIVE = REPORT / "archive"

SECTIONS = NOTE / "sections"
FIG = NOTE / "figures"
STATIC = FIG / "static"
NUMBERS = NOTE / "note_numbers.json"
MANIFEST = FIG / "manifest.json"
TEMPLATE = NOTE / "docx_template.docx"

OUT_HTML = REPORT / "Sovereign_Debt_Outlook.html"
OUT_PDF = REPORT / "Sovereign_Debt_Outlook.pdf"
OUT_DOCX = REPORT / "DM_Sovereign_Debt_Outlook.docx"


def ensure_report_dirs() -> Path:
    REPORT.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    return REPORT
