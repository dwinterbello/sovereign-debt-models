"""DMO remit progress (currentremit.pdf) and cash pricing of gilt rows.

The remit is in cash. The tape and the forward are in nominal. Cash per £100 nominal is the clean
price, times the index ratio for a linker. The report gives, per DMO sector, what auctions and
syndications still have to raise and what is left of the unallocated portion; the forward is
scaled so the rest of the financial year raises exactly that.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pandas as pd

from issuance.paths import RAW, REPO

REMIT_PDF = RAW / "uk" / "remit" / "currentremit.pdf"
REMIT_URL = "https://www.dmo.gov.uk/dmo_static_reports/currentremit.pdf"
INDEX_RATIOS = REPO / "uk_flows" / "cache" / "dmo" / "index_ratios.xml"
SECTORS = ("short", "medium", "long", "IL")

# 29 Sep 2026 report, used when the PDF is missing or unreadable.
FALLBACK = {
    "asof": pd.Timestamp("2026-09-29"),
    "fy_end": pd.Timestamp("2027-03-31"),
    "auction_required": {"short": 35_334.0, "medium": 26_849.0, "long": 7_400.0, "IL": 5_285.0},
    "auctions_remaining": {"short": 8, "medium": 7, "long": 4, "IL": 6},
    "synd_planned": {"short": 0.0, "medium": 23_800.0, "long": 15_000.0, "IL": 8_250.0},
    "synd_done": {"short": 0.0, "medium": 23_829.0, "long": 3_979.0, "IL": 4_774.0},
    "synd_scheduled_n": {"short": 0, "medium": 2, "long": 3, "IL": 2},
    "unallocated": 14_751.0,
    "total_planned": 246_200.0,
    "sales_to_date": 142_114.0,
}


def fetch_remit_pdf() -> bool:
    from issuance.http import get_bytes

    try:
        payload = get_bytes(REMIT_URL, impersonate=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  DMO remit progress: {exc}")
        return False
    if not payload.startswith(b"%PDF"):
        return False
    REMIT_PDF.parent.mkdir(parents=True, exist_ok=True)
    REMIT_PDF.write_bytes(payload)
    return True


def _nums(line: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*", line)]


def remit_progress(path=REMIT_PDF) -> dict:
    if not path.exists():
        return dict(FALLBACK)
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = pdf.pages[0].extract_text() or ""
    except Exception:  # noqa: BLE001
        return dict(FALLBACK)
    rows = {}
    for line in text.splitlines():
        for key, label in (
            ("auction_required", "Auction sales required to meet plans"),
            ("auctions_remaining", "Number of auctions remaining"),
            ("synd_planned", "Planned syndication sales"),
            ("synd_done", "Syndication sales to date"),
            ("synd_scheduled_n", "Number of syndications scheduled"),
            ("total_planned", "Total planned gilt sales"),
            ("sales_to_date", "Total gilt sales to date"),
        ):
            if line.strip().startswith(label) and key not in rows:
                rows[key] = _nums(line[len(label) + line.find(label):])
        if line.strip().startswith("Unallocated Portion") and "unallocated" not in rows:
            rows["unallocated"] = _nums(line)
    date = re.search(r"to (\d{1,2} \w+ \d{4})", text)
    try:
        out = {"asof": pd.Timestamp(date.group(1)) if date else FALLBACK["asof"]}
        for key in ("auction_required", "synd_planned", "synd_done"):
            out[key] = dict(zip(SECTORS, rows[key][:4]))
        for key in ("auctions_remaining", "synd_scheduled_n"):
            out[key] = dict(zip(SECTORS, [int(v) for v in rows[key][:4]]))
        out["unallocated"] = rows["unallocated"][0]
        out["total_planned"] = rows["total_planned"][4]
        out["sales_to_date"] = rows["sales_to_date"][4]
    except (KeyError, IndexError, AttributeError):
        return dict(FALLBACK)
    fy = out["asof"].year if out["asof"].month >= 4 else out["asof"].year - 1
    out["fy_end"] = pd.Timestamp(f"{fy + 1}-03-31")
    return out


def index_ratios(path=INDEX_RATIOS) -> dict[str, float]:
    """Latest published index ratio per 3-month-lag linker ISIN."""
    if not path.exists():
        return {}
    out: dict[str, tuple[str, float]] = {}
    for el in ET.parse(path).getroot().iter("GILTS"):
        isin, day, ratio = el.get("ISIN_CODE"), el.get("SETTLEMENT_DATE", ""), el.get("INDEX_RATIO_OR_RPI")
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            continue
        if isin and (isin not in out or day > out[isin][0]):
            out[isin] = (day, r)
    return {k: v[1] for k, v in out.items()}


def dmo_sector(years: float, linker: bool) -> str:
    if linker:
        return "IL"
    if years < 7:
        return "short"
    if years <= 15:
        return "medium"
    return "long"


def bond_price(coupon: float, yld: float, years: float) -> float:
    """Clean price per 100, semi-annual, settlement on a coupon date."""
    n = max(int(round(years * 2)), 1)
    c, y = coupon / 2.0, yld / 200.0
    if abs(y) < 1e-12:
        return c * n + 100.0
    disc = (1 + y) ** -n
    return c * (1 - disc) / y + 100.0 * disc
