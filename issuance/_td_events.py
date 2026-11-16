"""TreasuryDirect announced auctions: _td_announced, _td_security_type, _us_product (auction closes
used for the US forward calendar)."""
from __future__ import annotations

import json
import urllib.request
from datetime import date


def _td_security_type(row: dict) -> str:
    """TreasuryDirect `securityType` is Note for a TIPS. The sleeve is `type`."""
    if str(row.get("type") or "").upper() == "TIPS" or str(row.get("inflationProtected") or "").lower() == "yes":
        return "TIPS"
    return str(row.get("securityType") or "")


def _td_announced(start: date) -> list[dict]:
    """TreasuryDirect announced list, normalised to the Fiscal Data row shape.

    Fiscal Data `upcoming_auctions` lags the refunding: on 2026-11-09 its latest
    auction_date was 2026-10-29, so the desk substituted the issuance model's
    day-of-month (10y on 9 Nov) for the announced 3y that day and 10y on 10 Nov.
    """
    url = "https://www.treasurydirect.gov/TA_WS/securities/announced?format=json"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read())
    rows = []
    for r in raw if isinstance(raw, list) else []:
        d = (r.get("competitiveAuctionCloseDate") or "")[:10]
        if not d or d < start.isoformat():
            continue
        kind = _td_security_type(r)
        cmb = r.get("cashManagementBill")
        if cmb in (True, "Yes", "true", "True"):
            kind = "CMB"
        amt = r.get("offeringAmount")
        if amt in ("", None):
            amt = None
        rows.append({
            "auction_date": d,
            "security_type": kind,
            "security_term": r.get("securityTerm") or "",
            "offering_amt": amt,
            "reopening": r.get("reopening") or "",
            "announcemt_date": (r.get("announcementDate") or "")[:10],
            "cusip": r.get("cusip") or "",
        })
    rows.sort(key=lambda r: (r["auction_date"], r["security_term"]))
    return rows


def _us_product(kind: str, term: str) -> tuple[str, int]:
    """(TIPS | FRN | Nominal, whole years rounded up): '19-Year 10-Month' Bond -> ('Nominal', 20)."""
    k = kind.upper()
    sleeve = "TIPS" if "TIPS" in k else ("FRN" if "FRN" in k else "Nominal")
    words = term.replace("(", " ").replace(")", " ").replace("-", " ").split()
    years = 0
    for i, w in enumerate(words):
        if w.isdigit() and i + 1 < len(words):
            unit = words[i + 1].lower()
            if unit.startswith("year"):
                years += int(w)
            elif unit.startswith("month") and int(w):
                years += 1
        elif w.lower().endswith("y") and w[:-1].isdigit():
            years = int(w[:-1])
    if sleeve == "FRN":
        years = 2
    return sleeve, years
