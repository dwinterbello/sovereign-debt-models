"""Parse HM Treasury's Budget policy-decisions table (every measure, £m by fiscal year) from the Budget PDF.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.policy.uk_tables

Sign convention of the table: + is an Exchequer gain (lower borrowing), − a cost. Each measure keeps its
number, short name, head (Tax/Spend) and values; the parse is accepted only if the measures add up to the
table's own 'Total policy decisions' row in every year.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parents[1] / "data" / "policy" / "uk"
VAL = r"[+\-−–]?[\d,]+m"
ROW = re.compile(rf"^(.*?)\b(Tax|Spend|Tax/Spend|Spend/Tax)\s+((?:{VAL}\s*){{6}})(.*)$")
NUM = re.compile(r"^(\d{1,3})\s+(.*)$")
TOL_M = 50  # each row is rounded to the nearest 5m: ~90 rows can drift tens of millions from the published total


def _v(tok: str) -> float:
    t = tok.replace("−", "-").replace("–", "-").replace(",", "").rstrip("m")
    return float(t)


def parse(pdf: Path, table: str, first_fy: int) -> dict:
    with pdfplumber.open(pdf) as d:
        text = "\n".join((p.extract_text() or "") for p in d.pages)
    start = text.index(f"Table {table}")
    end = text.index("Total policy decisions", start)
    tail = text[end:end + 400].split("\n")[0]
    body = text[start:end].split("\n")
    measures, cur = [], None
    for ln in body:
        s = ln.strip()
        if not s or s.startswith("Head") or s.startswith("Budget ") or s.startswith("Table "):
            continue
        m = NUM.match(s)
        r = ROW.match(s)
        if m and not r:
            cur = {"n": int(m.group(1)), "name": [m.group(2)], "head": None, "values": None}
            measures.append(cur)
            continue
        if r:
            if m and (cur is None or cur.get("values") is not None):
                cur = {"n": int(m.group(1)), "name": [m.group(2).split(r.group(2))[0].strip()], "head": None,
                       "values": None}
                measures.append(cur)
            if cur is None:
                continue
            pre = r.group(1).strip()
            if pre and not m:
                cur["name"].append(pre)
            cur["head"] = r.group(2)
            cur["values"] = [_v(t) for t in re.findall(VAL, r.group(3))]
            post = r.group(4).strip()
            if post:
                cur["name"].append(post)
            continue
        if cur is not None and cur.get("values") is None:
            cur["name"].append(s)
        elif cur is not None:
            cur.setdefault("after", []).append(s)
    total = [_v(t) for t in re.findall(VAL, tail)][:6]
    out = []
    for i, m in enumerate(x for x in measures if x["values"] is not None):
        m["n"] = i + 1  # the table numbers rows 1..N; footnote digits can glue onto a number in the PDF text
    for m in measures:
        if m["values"] is None:
            continue
        full = " ".join(m["name"] + m.get("after", [])[:2]).strip()
        short = full.split(":")[0].strip() if ":" in full else full[:60]
        out.append({"n": m["n"], "short": short, "name": full, "head": m["head"], "values_m": m["values"]})
    years = [f"{y}-{str(y + 1)[2:]}" for y in range(first_fy, first_fy + 6)]
    s = [round(sum(x["values_m"][i] for x in out), 1) for i in range(6)]
    return {"source": pdf.name, "table": table, "years": years, "measures": out, "total_m": total,
            "sum_of_measures_m": s, "reconciles": all(abs(a - b) <= TOL_M for a, b in zip(s, total))}


EVENTS = {"budget_2025": ("4.1", 2025, "Budget 2025 (26 November 2025)"),
          "budget_2026": ("6.1", 2026, "Budget 2026 (October 2026)")}
URLS = {"spring_statement_2025": "https://assets.publishing.service.gov.uk/media/67e3ec2df356a2dc0e39b488/"
                                 "E03274109_HMT_Spring_Statement_Mar_25_Web_Accessible_.pdf",
        "budget_2025": "https://assets.publishing.service.gov.uk/media/69286818a245b0985f0341f3/"
                       "E03444720_Budget_2025_Web_Accessible.pdf",
        "budget_2026": "https://assets.publishing.service.gov.uk/media/6ae20b85460c1346f7be784e/"
                       "E03668986_HMT_Budget_2026_ELAY__2_.pdf"}


def _pdf(key: str) -> Path:
    """The Budget PDF (kept out of git; fetched from gov.uk when missing)."""
    import urllib.request

    path = HERE / f"{key}.pdf"
    if not path.exists():
        req = urllib.request.Request(URLS[key], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed gov.uk URL
            path.write_bytes(r.read())
    return path


if __name__ == "__main__":
    for key, (table, fy, title) in EVENTS.items():
        res = parse(_pdf(key), table, fy)
        res["event"] = title
        (HERE / f"{key}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(key, len(res["measures"]), "measures; reconciles:", res["reconciles"])
        print("table total ", res["total_m"])
        print("sum measures", res["sum_of_measures_m"])
        for x in sorted(res["measures"], key=lambda x: -abs(x["values_m"][4]))[:12]:
            print(f"{x['n']:3d} {x['head']:5s} {x['values_m']}  {x['short'][:60]}")
