"""Table helpers (TABLE_CSS, _table, _read) shared by the US and country dashboards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = Path(__file__).resolve().parent / "derived"

TABLE_CSS = """
table.sc { border-collapse: collapse; font: 11px/1.4 var(--mono); margin: 6px 0 10px; }
table.sc th, table.sc td { border: 1px solid var(--grid); padding: 3px 8px; text-align: right; }
table.sc th { color: var(--muted); font-weight: 600; }
table.sc td.l, table.sc th.l { text-align: left; }
table.sc td.best { font-weight: 700; color: var(--accent); }
"""


def _read(name: str):
    p = DERIVED / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_parquet(p)


def _table(rows: list[dict], cols: list[str], *, best_low: list[str] | None = None, fmt: str = "{:.1f}") -> str:
    head = "".join(f"<th class='{'l' if i == 0 else ''}'>{c}</th>" for i, c in enumerate(cols))
    body = []
    for r in rows:
        vals = [r.get(c) for c in cols]
        best = None
        if best_low:
            cand = {c: r[c] for c in best_low if isinstance(r.get(c), (int, float)) and np.isfinite(r[c])}
            best = min(cand, key=cand.get) if cand else None
        cells = []
        for i, (c, v) in enumerate(zip(cols, vals)):
            txt = fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            cls = "l" if i == 0 else ("best" if c == best else "")
            cells.append(f"<td class='{cls}'>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='sc'><tr>{head}</tr>{''.join(body)}</table>"
