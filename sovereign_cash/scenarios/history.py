"""Outturn history for the scenario page: the baseline's measure, before the baseline starts.

Display only. The engine never reads it and history is never shocked; it is there so a
reviewer sees the forecast against what actually happened.

Each country's history is on the same definition as its baseline, line by line:

* France, Germany, Italy: the monthly cash components the SARIMAX is fitted on, summed by
  calendar year (the same series and identity as the baseline). The months of the first
  forecast year already out-turned are kept as ``stub``, so the page can show that year whole.
* Spain, Sweden, Norway, Japan: the debt book's country-year table, built exactly as the
  baseline is (deficit % of GDP x GDP; GG revenue and interest).
* United Kingdom: PSNB ex public sector banks from ONS (-(JW2T + JW2Z)), by fiscal year.
  ONS's interest and receipts lines are not on the OBR's net-interest basis, so only the
  total is shown.
* United States: the unified deficit and revenues by fiscal year (OMB/Treasury, via FRED).
  FRED's interest series is gross, CBO's is net, so interest and outlays are not shown.

A line missing from ``lines`` has no history on the baseline's definition; the page says so.
"""
from __future__ import annotations

import io
import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels.common import period_weights

log = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parents[1] / "data"

N_HIST = 10  # years of outturn before the first forecast period
_MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _r(xs) -> list[float | None]:
    return [None if x is None or not np.isfinite(x) else round(float(x), 6) for x in xs]


def _gdp(table: pd.DataFrame, periods: list[str]) -> list[float | None]:
    g = table["gdp_bn"]
    out = []
    for p in periods:
        w = period_weights(p)
        out.append(float(sum(g[y] * k for y, k in w.items())) if all(y in g.index and np.isfinite(g[y]) for y in w)
                   else None)
    return out


def _sarimax(cc: str, c: dict) -> dict:
    from sovereign_cash.scenarios.channels.sarimax import PERIMETER, board_monthly, perimeter
    from sovereign_cash.yoy_components import IDENTITY, components, deficit

    frame = components(cc)
    model_total = deficit(cc, {k: frame[k] for k in IDENTITY[cc]}).dropna()
    frame = frame.reindex(model_total.index)
    board = board_monthly(cc)
    first = pd.Timestamp(c["months"][0] + "-01")
    if board is not None and any(ln.get("perimeter") for ln in c["lines"]):
        per = perimeter(cc, model_total, board, model_total.index.max())
        frame[PERIMETER[cc][0]] = per.reindex(frame.index)
    ids = [ln["id"] for ln in c["lines"]]
    signs = {ln["id"]: ln["sign"] for ln in c["lines"]}
    frame = frame.loc[frame.index < first, ids]
    y0 = int(c["periods"][0])
    years = list(range(y0 - N_HIST, y0))
    lines: dict[str, list] = {}
    for i in ids:
        s = frame[i]
        lines[i] = [float(s[s.index.year == y].sum()) if s[s.index.year == y].notna().sum() == 12 else None
                    for y in years]
    total = [None if any(lines[i][k] is None for i in ids) else sum(signs[i] * lines[i][k] for i in ids)
             for k in range(len(years))]
    part = frame.loc[frame.index.year == y0].dropna()
    stub = None
    if len(part):
        st = {i: float(part[i].sum()) for i in ids}
        stub = {"months": [part.index.min().strftime("%Y-%m"), part.index.max().strftime("%Y-%m")],
                "total": round(sum(signs[i] * st[i] for i in ids), 6), "lines": {i: round(v, 6) for i, v in st.items()}}
    src = {"fr": "AFT/DGFiP monthly State budget outturn (SMB)",
           "de": "BMF monthly federal budget outturn and special funds",
           "it": "MEF monthly State sector borrowing requirement; Bank of Italy cash-account lines"}[cc]
    return {"periods": [str(y) for y in years], "total": total, "lines": lines, "stub": stub,
            "source": f"{src}, as on the cash board, summed by calendar year"}


def _annual(cc: str, c: dict, book) -> dict:
    y0 = int(c["periods"][0])
    years = list(range(y0 - N_HIST, y0))
    a = book.annual_table(cc).reindex(years)
    pct = a["deficit_authority_pct_gdp"].where(a["deficit_authority_pct_gdp"].notna(), a["deficit_weo_pct_gdp"])
    total = pct * a["gdp_bn"] / 100.0
    revenue = a["gg_revenue_bn"]
    interest = a["gg_interest_bn"].where(a["gg_interest_bn"].notna(), a["interest_bn"])
    primary = total - interest + revenue
    return {"periods": [str(y) for y in years], "total": total.tolist(),
            "lines": {"receipts": revenue.tolist(), "primary": primary.tolist(), "interest": interest.tolist()},
            "source": "Deficit % of GDP x nominal GDP (authority outturn, else IMF WEO); GG revenue and interest; "
                      "built as the baseline is"}


def _ons_fy(code: str) -> dict[int, float]:
    j = json.loads((DATA / "uk" / "ons" / f"{code.lower()}.json").read_text())
    out: dict[int, list[float]] = {}
    for m in j["months"]:
        y, mm = m["date"].split()
        y, mm = int(y), _MON[mm]
        out.setdefault(y if mm >= 4 else y - 1, []).append(float(m["value"]))
    return {fy: sum(v) / 1000.0 for fy, v in out.items() if len(v) == 12}


def _uk(c: dict) -> dict:
    fy0 = int(c["periods"][0][:4])
    fys = list(range(fy0 - N_HIST, fy0))
    cb, ni = _ons_fy("JW2T"), _ons_fy("JW2Z")  # current budget and net investment, surplus-signed
    total = [-(cb[f] + ni[f]) if f in cb and f in ni else None for f in fys]
    return {"periods": [f"{f}-{str(f + 1)[2:]}" for f in fys], "total": total, "lines": {},
            "source": "ONS public sector finances, PSNB ex public sector banks (-(JW2T + JW2Z)), by fiscal year. "
                      "ONS interest and receipts are not on OBR's net-interest basis, so line history is not shown"}


def _fred(codes: list[str], max_age_days: float = 7.0) -> pd.DataFrame:
    path = DATA / "debt" / ("fred_" + "_".join(c.lower() for c in codes) + ".csv")
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400:
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(codes)
            text = urllib.request.urlopen(url, timeout=30).read().decode()
            pd.read_csv(io.StringIO(text))  # parse before replacing the cache
            path.write_text(text)
        except Exception:  # noqa: BLE001 - offline: use the cache if there is one
            log.warning("history: FRED fetch failed; using cache", exc_info=True)
            if not path.exists():
                raise
    d = pd.read_csv(path)
    d.index = pd.to_datetime(d.iloc[:, 0]).dt.year  # FRED dates fiscal years by their end (30 Sep)
    return d.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")


def _us(c: dict) -> dict:
    fy0 = int(c["periods"][0][2:])
    fys = list(range(fy0 - N_HIST, fy0))
    d = _fred(["FYFSD", "FYFR"]).reindex(fys)
    return {"periods": [f"FY{f}" for f in fys], "total": (-d["FYFSD"] / 1000.0).tolist(),
            "lines": {"receipts": (d["FYFR"] / 1000.0).tolist()},
            "source": "OMB/Treasury unified deficit (FYFSD) and receipts (FYFR) via FRED. FRED's interest is gross "
                      "and CBO's is net, so interest and outlays history is not shown"}


def _own(cc: str, c: dict) -> dict:
    """The annual component model's own outturn: every line on the measure's definition, so history and forecast
    join (UK PSNB from ONS by fiscal year; US unified deficit; the others AMECO general government)."""
    from sovereign_cash.scenarios.channels import own_annual as O

    h = O.history(cc)
    first = c["periods"][0]
    start = next(y for y in range(1990, 2100) if O.label(cc, y) == first)
    years = list(range(start - N_HIST, start))
    hh = h.reindex(years)
    src = {"uk": "ONS public sector finances (PSNB ex: -(JW2T + JW2Z); receipts JW2T + JW2Q + JW2S; interest JW2P)",
           "us": "OMB/Treasury via FRED: receipts FYFR, outlays FYONET, gross interest FYOINT",
           }.get(cc, "AMECO general government revenue, balance and gross interest, % of GDP x AMECO GDP")
    return {"periods": [O.label(cc, y) for y in years], "total": hh["total"].tolist(),
            "lines": {k: hh[k].tolist() for k in ("receipts", "primary", "interest")},
            "stub": c.get("own_stub"), "source": src}


def build(cc: str, c: dict, book) -> dict | None:
    """History for one country of the pack, or None (logged) if its sources are not to hand."""
    from sovereign_cash.scenarios.channels import own_annual, sarimax

    try:
        if cc in sarimax.COUNTRIES:
            h = _sarimax(cc, c)
        elif cc in own_annual.COUNTRIES:
            h = _own(cc, c)
        else:
            return None
    except Exception:  # noqa: BLE001 - history is display only; the pack stands without it
        log.exception("history: %s failed", cc)
        return None
    h["total"] = _r(h["total"])
    h["lines"] = {k: _r(v) for k, v in h["lines"].items()}
    h["gdp"] = _r(_gdp(book.annual_table(cc), h["periods"])) if book is not None else None
    return h
