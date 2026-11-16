"""OBR ready reckoner, March 2025 EFO, oil row corrected 17 March 2026.

The spreadsheet is linear. A scenario input (per cent, or the unit named on
the determinant) is scaled the way the determinant sheet scales it, then
multiplied by the £ million coefficient for that line and year. Receipts that
respond with a lag already have the lag in the effect formula. Debt interest
is a triangle: a one-year shock lands in that year and the years after. The
financing loop adds, once, the debt-interest total in £bn times the CGNCR
coefficient. It does not iterate, and it does not feed the primary deficit
back in.

Source workbook: sovereign_cash/data/uk/obr_ready_reckoner.xlsx
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import pandas as pd

XLSX = Path(__file__).resolve().parents[1] / "data" / "uk" / "obr_ready_reckoner.xlsx"
VINTAGE = "OBR ready reckoner, March 2025 EFO (oil corrected 17 March 2026)"
YEARS = ("2023-24", "2024-25", "2025-26", "2026-27", "2027-28", "2028-29", "2029-30")

_EFFECTS = ("Receipts_effects", "Spending_effects", "Debt_interest_effects")
_PROD = re.compile(
    r"^=(?:'([^']+)'|([A-Za-z0-9_]+))!\$?([A-Z]+)\$?(\d+)"
    r"\*(?:'([^']+)'|([A-Za-z0-9_]+))!\$?([A-Z]+)\$?(\d+)\s*$"
)
_LOOP = re.compile(
    r"^=SUM\(\$([A-Z])\$6:\$[A-Z]\$33\)/1000"
    r"\*(?:'([^']+)'|([A-Za-z0-9_]+))!\$?([A-Z]+)\$?(\d+)\s*$"
)


def _num(v) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _sheet_maps(wb) -> tuple[dict, dict]:
    """(sheet, col, row) -> (determinant, shock year, scale). RR cell -> £m."""
    det: dict[tuple[str, str, int], tuple[str, str, float]] = {}
    for name in ("Receipts_determinants", "Spending_determinants", "Debt_interest_determinants"):
        ws = wb[name]
        years = {}
        for c in range(2, ws.max_column + 1):
            v = ws.cell(2, c).value
            if isinstance(v, str) and v[:4].isdigit():
                years[c] = v
        for r in range(3, ws.max_row + 1):
            label = ws.cell(r, 1).value
            if not isinstance(label, str):
                continue
            cell = ws.cell(r, 2).value or ws.cell(r, 3).value
            text = getattr(cell, "text", "") or ""
            scale = 0.1 if "/10" in text else 10.0 if "*10" in text else 1.0
            for c, year in years.items():
                det[(name, _col(c), r)] = (label, year, scale)
    coef: dict[tuple[str, str, int], float] = {}
    for name in ("Receipts_RR£", "Spending_RR£", "Debt_interest_RR"):
        ws = wb[name]
        for r in range(1, ws.max_row + 1):
            for c in range(1, ws.max_column + 1):
                n = _num(ws.cell(r, c).value)
                if n is not None:
                    coef[(name, _col(c), r)] = n
    return det, coef


def _col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _col_n(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def _labels(ws) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    det_l, line_l, shock_l = {}, {}, {}
    det = line = ""
    for r in range(1, ws.max_row + 1):
        a, b = ws.cell(r, 1).value, ws.cell(r, 2).value
        if isinstance(a, str) and not a.startswith("Effect") and a != "Back to notes":
            det = a
        if isinstance(b, str) and b not in YEARS and not b.startswith("Total"):
            line = b
        elif isinstance(b, str) and b in YEARS:
            shock_l[r] = b
        if det:
            det_l[r] = det
        if line:
            line_l[r] = line
    return det_l, line_l, shock_l


def _effect_years(ws) -> dict[int, str]:
    out = {}
    for r in range(1, 8):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and v in YEARS:
                out[c] = v
        if out:
            return out
    return out


def _terms(wb) -> pd.DataFrame:
    det, coef = _sheet_maps(wb)
    rows = []
    for sheet in _EFFECTS:
        ws = wb[sheet]
        det_l, line_l, shock_l = _labels(ws)
        years = _effect_years(ws)
        block = "receipts" if sheet.startswith("Receipts") else (
            "debt_interest" if sheet.startswith("Debt") else "spending")
        for r in range(1, ws.max_row + 1):
            for c, effect_fy in years.items():
                raw = ws.cell(r, c).value
                if not isinstance(raw, str) or not raw.startswith("="):
                    continue
                m = _PROD.match(raw)
                if m:
                    dsheet, dcol, drow = m.group(1) or m.group(2), m.group(3), int(m.group(4))
                    rsheet, rcol, rrow = m.group(5) or m.group(6), m.group(7), int(m.group(8))
                    key = (dsheet, dcol, drow)
                    if key not in det or (rsheet, rcol, rrow) not in coef:
                        continue
                    name, shock_fy, scale = det[key]
                    rows.append({
                        "block": block, "determinant": name,
                        "line": "Debt interest" if block == "debt_interest" else line_l.get(r, ""),
                        "shock_fy": shock_fy, "effect_fy": effect_fy,
                        "scale": scale, "gbp_m": coef[(rsheet, rcol, rrow)], "loop": False,
                    })
                    continue
                loop = _LOOP.match(raw)
                if loop:
                    rsheet = loop.group(2) or loop.group(3)
                    rcol, rrow = loop.group(4), int(loop.group(5))
                    if (rsheet, rcol, rrow) not in coef:
                        continue
                    rows.append({
                        "block": "debt_interest", "determinant": "CGNCR financing loop",
                        "line": "Debt interest",
                        "shock_fy": shock_l.get(r, ""), "effect_fy": effect_fy,
                        "scale": 1.0, "gbp_m": coef[(rsheet, rcol, rrow)], "loop": True,
                    })
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def coefficients() -> pd.DataFrame:
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, data_only=False)
    try:
        return _terms(wb)
    finally:
        wb.close()


def determinants() -> list[str]:
    d = coefficients()
    names = d.loc[~d["loop"], "determinant"].drop_duplicates()
    return list(names)


def run(shocks: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Shocks are the scenario-input units, by fiscal year '2026-27'.

    A missing year is zero. Returns one row per block, determinant and line
    that moves, in £ million and £ billion, by effect year.
    """
    d = coefficients()
    direct = d.loc[~d["loop"]].copy()
    hit = []
    for name, by_year in shocks.items():
        sub = direct[direct["determinant"] == name]
        if sub.empty:
            known = ", ".join(determinants()[:8])
            raise KeyError(f"unknown determinant {name!r}. Known names start: {known}")
        for fy, value in by_year.items():
            if value is None or value == 0:
                continue
            part = sub[sub["shock_fy"] == fy]
            if part.empty:
                continue
            part = part.copy()
            part["gbp_m"] = part["gbp_m"] * part["scale"] * float(value)
            hit.append(part)
    if not hit:
        out = direct.iloc[0:0].copy()
    else:
        out = pd.concat(hit, ignore_index=True)
        out = (out.groupby(["block", "determinant", "line", "effect_fy"], as_index=False)["gbp_m"].sum())
    loop = d.loc[d["loop"]]
    if not out.empty and not loop.empty:
        interest = (out[out["block"] == "debt_interest"]
                    .groupby("effect_fy")["gbp_m"].sum())
        extra = []
        for _, row in loop.iterrows():
            base = float(interest.get(row["effect_fy"], 0.0))
            if base == 0 or row["gbp_m"] == 0:
                continue
            extra.append({
                "block": "debt_interest", "determinant": "CGNCR financing loop",
                "line": "Debt interest", "effect_fy": row["effect_fy"],
                "gbp_m": base / 1000.0 * float(row["gbp_m"]),
            })
        if extra:
            out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    if out.empty:
        return out
    out["gbp_bn"] = out["gbp_m"] / 1000.0
    return out.sort_values(["block", "line", "determinant", "effect_fy"]).reset_index(drop=True)


def psnb(effects: pd.DataFrame) -> pd.Series:
    """Change in public sector net borrowing, £bn. Spending and debt interest add; receipts subtract."""
    if effects.empty:
        return pd.Series(dtype=float)
    sign = effects["block"].map({"receipts": -1.0, "spending": 1.0, "debt_interest": 1.0})
    return (effects["gbp_bn"] * sign).groupby(effects["effect_fy"]).sum().reindex(list(YEARS))


def unit(name: str, years: tuple[str, ...] = YEARS[2:]) -> dict[str, dict[str, float]]:
    """+1 on `name` in each year from 2025-26. The unit is the determinant's own."""
    return {name: {fy: 1.0 for fy in years}}
