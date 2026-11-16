"""Our forecast against each budgetary authority's, line by line: where every bn of difference comes from.

For each country and forecast period the authority's forecast is split into the same three lines as ours (receipts,
spending excluding interest, interest), on the authority's own published numbers:

    United Kingdom   OBR Economic and fiscal outlook (current receipts; TME less central government debt interest
                     net of the APF; that interest), the budgetary authority's forecast
    United States    CBO baseline (revenues; outlays less net interest; net interest)
    Spain, Sweden, Norway, Japan   European Commission (AMECO): revenue, gross interest, balance
    France, Germany, Italy         the government's own plan on the cash-board measure (total only)

Lines are not always defined identically (our UK and US interest is gross, the authorities' net), so the difference
in each line is split into the definitional difference already present in the last outturn year, grown with GDP,
and the forecast difference after it. The rows add up exactly to our forecast minus theirs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SC = Path(__file__).resolve().parents[1]
DEBT = SC / "data" / "debt"
POLICY = SC / "data" / "policy"
LINES = ("receipts", "primary", "interest")
REASONS = {
    "uk": {"receipts": "The OBR has receipts rising from 40.1% of GDP in 2025-26 to 42.7% by 2029-30. Our model gives "
                       "income tax + NICs a frozen-threshold earnings elasticity of 1.3 through FY 2027-28 (OBR EFO "
                       "Oct 2026 para 4.14); from FY 2028-29 thresholds would have indexed under prior law so that "
                       "component returns to unit elasticity and Budget 2025's extension to April 2031 sits only in "
                       "the policy line. Remaining gap: capital taxes, corporation tax and non-dom one-offs.",
           "primary": "The OBR's welfare (triple lock, health and disability caseloads) and other AME grow faster than "
                      "prices; our model grows spending with prices (departmental plans are in our policy line).",
           "interest": "Ours is gross central government interest from our bond-by-bond book at today's curve; the "
                       "OBR's is net of the Bank of England's APF."},
    "us": {"receipts": "CBO's revenues include tariff receipts at the rates in force when it forecast and its own "
                       "income and payroll tax projections; ours hold revenues' share of GDP moving back toward its "
                       "mean.",
           "primary": "Definitions: ours exclude intragovernmental interest offsets that CBO nets in outlays; the "
                      "forecast part is CBO's mandatory spending growth (Social Security, Medicare) above prices.",
           "interest": "Ours is gross interest on Treasury debt from our book; CBO's is net interest."},
    "default": {"receipts": "Revenue's share of GDP: ours carried from the last outturn; the Commission's forecast.",
                "primary": "Primary spending: ours grows with prices (Sweden: with GDP); the Commission's forecast.",
                "interest": "Interest: ours from our bond-by-bond book at today's curve; the Commission's estimate."},
}


def _ours(c: dict) -> dict[str, list[float]]:
    """Our lines per period, full periods (the UK's first year includes its months of outturn), with policy."""
    st = (c.get("history") or {}).get("stub") or {}
    out = {}
    lines = {ln["id"]: ln for ln in c["lines"]}
    pol = (c.get("policy") or {}).get("by_line") or {}
    for k in LINES:
        if k not in lines:
            return {}
        v = [float(x) for x in c["baseline"]["lines"][k]]
        if st:
            v[0] += float(st["lines"].get(k, 0.0))
        if k in pol:
            v = [a + b for a, b in zip(v, pol[k])]
        out[k] = v
    out["total"] = [out["primary"][i] + out["interest"][i] - out["receipts"][i] for i in range(len(c["periods"]))]
    return out


def _year(label: str) -> int:
    return int(label[2:]) if label.startswith("FY") else int(label[:4])


def authority(cc: str) -> dict | None:
    """{year: {receipts, primary, interest, total}} on the authority's published numbers, and its label."""
    if cc == "uk":
        f = json.loads((POLICY / "uk" / "obr_decisions.json").read_text())["obr_forecast"]
        ys = range(f["first_fy"], f["first_fy"] + len(f["receipts"]))
        rows = {y: {"receipts": r, "primary": t - i, "interest": i, "total": t - r}
                for y, r, t, i in zip(ys, f["receipts"], f["tme"], f["interest"])}
        return {"label": f["vintage"], "source": f["source"], "url": f["url"], "rows": rows}
    if cc == "us":
        d = pd.read_parquet(DEBT / "cbo.parquet")
        p = d.pivot_table(index="year", columns="var", values="value")
        rows = {}
        for y, r in p.iterrows():
            if not all(np.isfinite(r.get(k, np.nan)) for k in ("revenue_bn", "deficit_bn", "interest_net_bn")):
                continue
            tot = -float(r["deficit_bn"])
            rows[int(y)] = {"receipts": float(r["revenue_bn"]), "interest": float(r["interest_net_bn"]),
                            "primary": float(r["revenue_bn"]) + tot - float(r["interest_net_bn"]), "total": tot}
        return {"label": f"CBO {d['vintage'].iloc[0].split()[-1]} baseline", "source": "CBO budget projections",
                "url": "https://github.com/US-CBO/cbo-data", "rows": rows}
    if cc in ("es", "se", "no", "jp"):
        a = pd.read_parquet(DEBT / "ameco.parquet")
        p = a[a["country"] == cc].pivot_table(index="year", columns="var", values="value")
        rows = {}
        for y, r in p.iterrows():
            vals = [r.get(k, np.nan) for k in ("revenue_pct", "balance_pct", "interest_gross_pct", "gdp_lcu_bn")]
            if not all(np.isfinite(v) for v in vals):
                continue
            rev, bal, it, g = (float(v) for v in vals)
            rows[int(y)] = {"receipts": rev * g / 100, "interest": it * g / 100, "total": -bal * g / 100,
                            "primary": (rev - bal - it) * g / 100}
        return {"label": "European Commission (AMECO)", "source": "AMECO, general government",
                "url": "https://economy-finance.ec.europa.eu/economic-research-and-databases/economic-databases/ameco-database_en",
                "rows": rows}
    return None


def build(pack: dict) -> dict:
    out = {}
    for cc, c in pack["countries"].items():
        ours = _ours(c)
        auth = authority(cc)
        years = [_year(p) for p in c["periods"]]
        gdp = c.get("gdp")
        if not ours or not auth or not gdp:
            plan = (c["baseline"].get("authority") or {}).get("years") or {}
            if plan:
                tot = [float(c["baseline"].get("with_policy", c["baseline"]["total"])[i])
                       + (float(((c.get("history") or {}).get("stub") or {}).get("total", 0.0)) if i == 0 else 0.0)
                       for i in range(len(years))]
                out[cc] = {"label": (c["baseline"].get("authority") or {}).get("label", "Authority plan"),
                           "periods": c["periods"], "total_only": True,
                           "authority": [plan.get(p) for p in c["periods"]], "ours": tot}
            continue
        hist = c.get("history") or {}
        hyears = [_year(p) for p in hist.get("periods", [])]
        # base year: the last year with our outturn and the authority's number
        common = [y for y in hyears if y in auth["rows"] and all(
            hist["lines"].get(k) and hist["lines"][k][hyears.index(y)] is not None for k in LINES)]
        base = max(common) if common else None
        hgdp = hist.get("gdp") or []
        rows = []
        for i, y in enumerate(years):
            a = auth["rows"].get(y)
            if a is None:
                continue
            row = {"period": c["periods"][i], "authority": a["total"], "ours": ours["total"][i], "gdp": gdp[i],
                   "lines": {}}
            for k in LINES:
                sign = -1.0 if k == "receipts" else 1.0
                gap = sign * (ours[k][i] - a[k])  # effect on our deficit relative to theirs
                defn = 0.0
                if base is not None and hgdp:
                    j = hyears.index(base)
                    g0 = hgdp[j]
                    defn = sign * (float(hist["lines"][k][j]) - auth["rows"][base][k]) * (gdp[i] / g0 if g0 else 1.0)
                row["lines"][k] = {"ours": ours[k][i], "authority": a[k], "effect": gap,
                                   "definitions": defn, "forecast": gap - defn,
                                   "effect_pct_gdp": 100 * gap / gdp[i],
                                   "forecast_pct_gdp": 100 * (gap - defn) / gdp[i]}
            row["gap"] = ours["total"][i] - a["total"]
            rows.append(row)
        if rows:
            rs = REASONS.get(cc, REASONS["default"])
            out[cc] = {"label": auth["label"], "source": auth["source"], "url": auth["url"], "base_year": base,
                       "rows": rows, "reasons": rs, "unit": c["unit"]}
    return out
