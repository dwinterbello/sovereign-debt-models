"""Unit-shock scenario table for FR / IT / DE cash — UK ready-reckoner style.

The live cash path is unchanged (seasonals / plan × shape). This table is the
model-implied change in the cash requirement from a unit shock:

  GDP +1pp     → NGDP YoY on receipt / tax lines (SARIMAX)
  CPI +1pp     → HICP YoY on primary spending (SARIMAX; interest carved out)
  Rates +100bp → ISIN-book interest triangle (``debt_repricing`` baseline/shock)

Rates are not a SARIMAX slope: monthly books do not publish interest, so the
debt module's locked-coupon + new-issuance path is the bottom-up rates channel
(same role as the UK OBR debt-interest ready reckoner).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sovereign_cash.interest_split import rates_shock_delta
from sovereign_cash.macro_authority import annual_authority
from sovereign_cash.monthly_components import LEVEL_SPEC, components, deficit, live_path
from sovereign_cash.yoy_components import SPEC

ROOT = Path(__file__).resolve().parents[1]
YEARS_AHEAD = 3  # calendar years from the origin year
MACRO_SHOCKS = (
    ("gdp", "Nominal GDP growth +1pp a year", {"ngdp": 1.0}),
    ("cpi", "CPI inflation +1pp a year", {"hicp": 1.0}),
)


def _origin(cc: str) -> pd.Timestamp:
    frame = components(cc)
    y = deficit(cc, {n: frame[n] for n, _ in SPEC[cc]}).dropna()
    return y.index.max()


def _annual_sums(s: pd.Series, years: list[int]) -> dict[int, float]:
    out = {}
    for y in years:
        chunk = s[s.index.year == y]
        out[y] = float(chunk.sum()) if len(chunk) else float("nan")
    return out


def run(cc: str, *, years: list[int] | None = None) -> dict:
    """Baseline and shocked annual cash-requirement sums (local-currency bn)."""
    origin = _origin(cc)
    if years is None:
        y0 = int(origin.year)
        years = list(range(y0, y0 + YEARS_AHEAD + 1))
    # enough months to cover the last scenario year
    last = pd.Timestamp(f"{years[-1]}-12-01")
    h = max(12, (last.year - origin.year) * 12 + (12 - origin.month) + 1)
    spec = LEVEL_SPEC.get(cc, "ratio")
    base = live_path(cc, origin, h, mode="level", with_exog=True, use_authority=True, spec=spec)
    if base is None:
        return {"cc": cc, "origin": str(origin.date()), "years": years, "ok": False,
                "error": "SARIMAX fit failed"}
    base_y = _annual_sums(base, years)
    ngdp_a, cpi_a, src = annual_authority(cc)
    rows = []
    for key, title, shock in MACRO_SHOCKS:
        path = live_path(cc, origin, h, mode="level", with_exog=True,
                         use_authority=True, shock=shock, spec=spec)
        if path is None:
            continue
        sh_y = _annual_sums(path, years)
        rows.append({
            "key": key, "title": title, "shock": shock,
            "delta": {str(y): sh_y[y] - base_y[y] for y in years},
            "level": {str(y): sh_y[y] for y in years},
        })
    rate_d = rates_shock_delta(cc, 100.0)
    if rate_d:
        rows.append({
            "key": "rates", "title": "Bond rates +100bp",
            "shock": {"rates_bp": 100.0},
            "delta": {str(y): float(rate_d.get(y, 0.0)) for y in years},
            "level": {str(y): base_y[y] + float(rate_d.get(y, 0.0)) for y in years},
            "channel": "debt_repricing ISIN book (locked coupons + new issuance)",
        })
    return {
        "cc": cc, "origin": str(origin.date()), "years": years, "ok": True,
        "base": {str(y): base_y[y] for y in years},
        "drivers": {
            "ngdp_source": src["ngdp"],
            "cpi_source": src["cpi"],
            "rates_source": (
                "AFT +100bp triangle (1y/5y/9y)" if cc == "fr"
                else "debt interest/debt; +100bp from debt_repricing shock case"
            ),
            "ngdp": {str(int(y)): float(v) for y, v in ngdp_a.loc[years[0]:years[-1]].items()},
            "cpi": {str(int(y)): float(v) for y, v in cpi_a.loc[years[0]:years[-1]].items()},
        },
        "shocks": rows,
        "spec": spec,
        "note": ("Δ cash bn. GDP/CPI via component SARIMAX (" + {"ratio": "receipts and spending in "
                 "proportion to the GDP and price levels", "loglevel": "log levels", "yoy": "on YoY growth rates"}[spec]
                 + "); rates via the ISIN book or the AFT triangle (+100bp). Live cash path unchanged."),
    }


def table_html(cc: str, payload: dict | None = None, *, ccy: str = "€") -> str:
    """Kept for callers; the HTML lives in ``sovereign_cash.web.tables``."""
    from sovereign_cash.web.tables import unit_shock_table

    return unit_shock_table(cc, payload or run(cc), ccy=ccy)


def save(cc: str, payload: dict | None = None) -> Path:
    path = ROOT / "derived" / f"{cc}_cash_scenarios.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = payload or run(cc)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load(cc: str) -> dict | None:
    path = ROOT / "derived" / f"{cc}_cash_scenarios.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
