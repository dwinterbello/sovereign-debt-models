"""MoF JGB auction calendar (dates only) for the forward.

    PYTHONPATH=. .venv/bin/python -m issuance.jp_calendar

MoF posts each month's calendar about three months ahead; amounts are announced a week before
each auction, so sizes come from the issuance plan. Writes ``data/jp_calendar.json``:
one row per coupon auction with date and product.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from issuance.http import session

URL = "https://www.mof.go.jp/english/policy/jgbs/auction/calendar/{yymm}e.htm"
OUT = Path(__file__).resolve().parent / "data" / "jp_calendar.json"
_DATE = re.compile(r"^([A-Z][a-z]{2})\.? (\d{1,2}), (\d{4})$")
_PRODUCT = [
    (re.compile(r"^5-year Japan Climate Transition"), "GX5y"),
    (re.compile(r"^10-year Japan Climate Transition"), "GX10y"),
    (re.compile(r"^10-year Inflation-Indexed"), "10y IL"),
    (re.compile(r"^(2|5|10|20|30|40)-year$"), None),
    (re.compile(r"^Liquidity Enhancement Auction \(remaining maturities of ([\d.]+)-([\d.]+) years\)"), "LEA"),
]


def _lines(text: str) -> list[str]:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", text, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", text))
    return [s.strip() for s in text.splitlines() if s.strip()]


def parse(text: str) -> list[dict]:
    lines = _lines(text)
    head = "(1) Government Bonds and Treasury Discount Bills"
    if head not in lines:
        return []
    lines = lines[lines.index(head):]
    for end in ("(2) Buy-back", "(3) Borrowing", "[Notes]"):
        if end in lines:
            lines = lines[:lines.index(end)]
            break
    lines = [f"{a} {b}" if a == "Liquidity Enhancement Auction" else a for a, b in zip(lines, lines[1:] + [""])]
    rows, day = [], None
    for s in lines:
        m = _DATE.match(s)
        if m:
            day = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
            continue
        if day is None:
            continue
        for pat, name in _PRODUCT:
            hit = pat.match(s)
            if not hit:
                continue
            if name is None:
                product = f"{hit.group(1)}y"
            elif name == "LEA":
                product = f"LEA {hit.group(1)}-{hit.group(2)}y"
            else:
                product = name
            rows.append({"date": day.strftime("%Y-%m-%d"), "product": product})
            break
    return rows


def fetch_calendar(start: str | None = None, months: int = 6) -> list[dict]:
    first = pd.Timestamp(start) if start else pd.Timestamp.today().normalize().replace(day=1)
    s = session()
    rows = []
    for k in range(months):
        m = first + pd.DateOffset(months=k)
        r = s.get(URL.format(yymm=m.strftime("%y%m")), timeout=60)
        if r.status_code != 200:
            break
        got = parse(r.text)
        if not got:
            break
        rows += got
    if rows:
        OUT.write_text(json.dumps({"fetched": pd.Timestamp.today().strftime("%Y-%m-%d"), "auctions": rows}, indent=1))
    return rows


LEA_INDEX = "https://www.mof.go.jp/english/policy/jgbs/topics/press_release/index.htm"
LEA_OUT = Path(__file__).resolve().parent / "data" / "jp_lea_amounts.json"
_LEA_LINK = re.compile(r'href="([^"]+)"[^>]*>\s*Issuance Amounts for Liquidity Enhancement Auctions in ([^<]+)')
_MONTH = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})$")
_ZONE = re.compile(r"^\(\d\) Issues with remaining maturities of ([\d.]+)-([\d.]+) years")
_AMOUNT = re.compile(r"^Approximately ([\d,.]+) (billion|trillion) yen|^Not be implemented")


def parse_lea(text: str) -> dict[str, dict[str, float]]:
    """Quarterly LEA release -> {"YYYY-MM": {"lo-hi": ¥mn}}; "Not be implemented" is 0."""
    months, out, zone, k = [], {}, None, 0
    for s in _lines(text):
        s = s.replace("\xa0", " ")
        if m := _MONTH.match(re.sub(r"\s+", " ", s)):
            months.append(pd.Timestamp(f"{m.group(1)} 1 {m.group(2)}").strftime("%Y-%m"))
        elif m := _ZONE.match(s):
            zone, k = f"{m.group(1)}-{m.group(2)}", 0
        elif zone and (m := _AMOUNT.match(s)) and k < len(months):
            amt = 0.0 if m.group(1) is None else float(m.group(1).replace(",", "")) * (1e6 if m.group(2) == "trillion" else 1e3)
            out.setdefault(months[k], {})[zone] = amt
            k += 1
    return out


def fetch_lea_amounts(releases: int = 4) -> dict[str, dict[str, float]]:
    s = session()
    r = s.get(LEA_INDEX, timeout=60)
    r.raise_for_status()
    got: dict[str, dict[str, float]] = {}
    for href, _ in _LEA_LINK.findall(r.text)[:releases]:
        page = s.get(href if href.startswith("http") else f"https://www.mof.go.jp{href}", timeout=60)
        if page.status_code == 200:
            for m, z in parse_lea(page.text).items():
                got.setdefault(m, z)
    if got:
        LEA_OUT.write_text(json.dumps({"fetched": pd.Timestamp.today().strftime("%Y-%m-%d"),
                                       "months": dict(sorted(got.items()))}, indent=1))
    return got


def load_lea() -> dict[str, dict[str, float]]:
    return json.loads(LEA_OUT.read_text())["months"] if LEA_OUT.exists() else {}


def load() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=["date", "product"])
    d = pd.DataFrame(json.loads(OUT.read_text())["auctions"])
    d["date"] = pd.to_datetime(d["date"])
    return d


if __name__ == "__main__":
    got = fetch_calendar()
    print(f"{len(got)} auctions", got[:3], got[-3:])
    lea = fetch_lea_amounts()
    print(f"LEA amounts for {len(lea)} months", list(lea)[-3:])
