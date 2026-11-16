"""OBR EFO line items: the receipt, spending and PSNB-to-CGNCR build, plus the determinants they cite.

One vintage, parsed from the gov.uk EFO PDF (obr.uk is bot-walled). The cash path uses
``cgncr_ex_nr`` — CGNCR excluding Network Rail — which is the figure the DMO remit copies.
Years the remit has not published fall back to this table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "sovereign_cash" / "data" / "debt" / "obr_lines.parquet"

# Normalised row label -> (var, unit). Footnote digits are stripped before matching.
ROWS = {
    "income tax": ("income_tax_bn", "GBP bn"),
    "national insurance contributions": ("nics_bn", "GBP bn"),
    "value added tax": ("vat_bn", "GBP bn"),
    "corporation tax": ("corp_tax_bn", "GBP bn"),
    "psce in rdel": ("rdel_bn", "GBP bn"),
    "welfare spending": ("welfare_bn", "GBP bn"),
    "central government debt interest, net of": ("debt_interest_bn", "GBP bn"),
    "public sector borrowing (a)": ("psnb_bn", "GBP bn"),
    "financial transactions (b)": ("fin_transactions_bn", "GBP bn"),
    "public sector net cash": ("psncr_bn", "GBP bn"),
    "less local authorities and public": ("la_pc_ncr_bn", "GBP bn"),
    "cg net cash requirement": ("cgncr_bn", "GBP bn"),
    "less network rail ncr (d)": ("network_rail_ncr_bn", "GBP bn"),
    "cgncr ex network rail": ("cgncr_ex_nr_bn", "GBP bn"),
    "wages and salaries": ("wages_pct", "%"),
    "cpi": ("cpi_pct", "%"),
    "rpi": ("rpi_pct", "%"),
    "average weekly earnings": ("awe_pct", "%"),
    "bank rate (per cent)": ("bank_rate_pct", "%"),
    "market gilt rates (per cent)": ("gilt_rate_pct", "%"),
}

# Order on the page. Group is the section header.
PAGE = (
    ("psnb_bn", "Public sector net borrowing", "Bridge"),
    ("fin_transactions_bn", "Financial transactions", "Bridge"),
    ("la_pc_ncr_bn", "Local authorities and public corporations", "Bridge"),
    ("cgncr_bn", "CGNCR", "Bridge"),
    ("network_rail_ncr_bn", "Network Rail", "Bridge"),
    ("cgncr_ex_nr_bn", "CGNCR ex Network Rail", "Bridge"),
    ("income_tax_bn", "Income tax", "Receipts and spending"),
    ("nics_bn", "National Insurance", "Receipts and spending"),
    ("vat_bn", "VAT", "Receipts and spending"),
    ("corp_tax_bn", "Corporation tax", "Receipts and spending"),
    ("welfare_bn", "Welfare", "Receipts and spending"),
    ("debt_interest_bn", "Debt interest, net of APF", "Receipts and spending"),
    ("rdel_bn", "Resource DEL", "Receipts and spending"),
    ("wages_pct", "Wages and salaries", "Determinants"),
    ("cpi_pct", "CPI", "Determinants"),
    ("rpi_pct", "RPI", "Determinants"),
    ("awe_pct", "Average earnings", "Determinants"),
    ("bank_rate_pct", "Bank Rate", "Determinants"),
    ("gilt_rate_pct", "Gilt rates", "Determinants"),
)

_FOOT = re.compile(r"(?<=[A-Za-z\)])\d+(?:,\d+)*")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_NEWROW = re.compile(r"^(?:[A-Z]|Less |CG)")


def _norm(label: str) -> str:
    s = _FOOT.sub("", label)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _numbers(line: str) -> list[float]:
    return [float(x.replace(",", "")) for x in _NUM.findall(line)]


def _words(line: str) -> str:
    return _NUM.sub(" ", line).strip()


def stitch(text: str) -> list[tuple[str, list[float]]]:
    """Join wrapped OBR table rows. A numbers-only line closes the label above it.
    A new capitalised row discards a leftover suffix ('requirement (PSNCR)')."""
    out: list[tuple[str, list[float]]] = []
    label: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("Source:"):
            continue
        if ln.lower().startswith("of which") and not _numbers(_FOOT.sub("", ln)):
            continue
        bare = _FOOT.sub("", ln)
        nums, words = _numbers(bare), _words(bare)
        if nums and not words:
            if label:
                out.append((" ".join(label), nums))
                label = []
            continue
        if nums and words:
            if label:
                out.append((" ".join(label), []))
                label = []
            out.append((words, nums))
            continue
        if label and _NEWROW.match(ln):
            label = []
        if ln[:1].isalpha() or ln.startswith("("):
            label.append(ln)
    return out


def _header(text: str) -> list[int]:
    fys = [int(y) for y in re.findall(r"(20\d{2})-\d{2}", text)]
    header: list[int] = []
    for y in fys:
        if header and y != header[-1] + 1:
            if len(header) >= 5:
                break
            header = []
        header.append(y)
    return header


def _block(text: str) -> tuple[str, str]:
    m = re.search(r"Table ([A-Z]\.\d+|\d+\.\d+)", text)
    tid = m.group(1) if m else ""
    unit = "%" if tid == "A.3" else "GBP bn"
    name = {"A.5": "receipts", "A.7": "expenditure", "4.6": "welfare", "5.4": "financing", "A.3": "determinants"}.get(tid, tid or "other")
    return name, unit


def _slug(key: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return s[:48] or "row"


def parse_pages(pages: list[str], vint: str) -> pd.DataFrame:
    """Every numbered row in the level tables. A named row keeps its stable ``var``; the rest are kept
    under a slug. The extra growth column on the determinant table is dropped. 'Changes since' pages
    are not read."""
    rows = []
    seen: set[str] = set()
    for text in pages:
        if re.search(r"changes since", text, re.I):
            continue
        header = _header(text)
        if len(header) < 5:
            continue
        block, unit = _block(text)
        n = 0
        for label, nums in stitch(text):
            if len(nums) < len(header):
                continue
            key = _norm(label)
            if not key or key in {"outturn", "forecast", "outturn forecast"}:
                continue
            show = re.sub(r"^of which:\s*", "", label, flags=re.I).strip()
            if key in ROWS and not key.startswith("of which"):
                var, unit_row = ROWS[key]
            else:
                var = f"{block}_{_slug(key)}"
                unit_row = "%" if "per cent" in key else unit
            if var in seen:
                continue
            seen.add(var)
            for fy, v in zip(header, nums):
                rows.append(("uk", "obr", vint, "fy", fy, var, v, unit_row, block, show, n))
            n += 1
    return pd.DataFrame(rows, columns=["country", "source", "vintage", "basis", "year", "var", "value", "unit",
                                       "block", "label", "ord"])


def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=["country", "source", "vintage", "basis", "year", "var", "value", "unit"])
    return pd.read_parquet(PATH)


def cgncr_ex_nr() -> dict[int, float]:
    """OBR CGNCR excluding Network Rail, £bn, by fiscal-year start. Empty if the table is not cached."""
    d = load()
    if d.empty:
        return {}
    s = d[d["var"] == "cgncr_ex_nr_bn"]
    return {int(y): float(v) for y, v in zip(s["year"], s["value"])}


def vintage() -> str:
    d = load()
    if d.empty:
        return ""
    return str(d["vintage"].iloc[0])
