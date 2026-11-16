"""Spain, Sweden, Norway, Japan: annual central-government lines x elasticities.

Baseline: the general-government deficit, authority path where published, else
IMF WEO, times nominal GDP; revenue and interest on the same general-government
basis from the debt book's annual table; primary spending = deficit + revenue -
interest. (The book's net issuance would mix in cash-buffer moves, e.g. Spain's
2026 pre-funding.) Rates move central-government interest from the ISIN book.
Macro responses are elasticities on those lines, from the country's own model
where it has one:

  ES  the shipped IGAE bottom-up: category betas (tax 0.5 on NGDP, spending 0.35
      on CPI, ...) weighted by category shares, x the 0.7 leaf blend, prior-year
      drivers (12-month lag)
  NO  the shipped Tab4 bottom-up: petroleum lines move one-for-one with Brent
  SE, JP, and NO's NGDP/CPI: no shipped macro model -> stated assumptions,
      unit-elastic receipts and half of primary spending indexed to CPI with a
      one-year lag. Badged "assumed"; the dashboard can scale each one 0-2x.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels.common import CCY, N_YEARS, NAMES, calendar_periods, tau_months, ts

log = logging.getLogger(__name__)
COUNTRIES = ("es", "se", "no", "jp")
LINES = [{"id": "receipts", "label": "Revenue", "sign": -1},
         {"id": "primary", "label": "Primary spending", "sign": 1},
         {"id": "interest", "label": "Interest", "sign": 1}]
ASSUMED = {"receipts_ngdp": 1.0, "primary_cpi": 0.5, "primary_cpi_lag": 12}
ASSUMED_NOTE = ("Assumption, not fitted: receipts unit-elastic to nominal GDP; half of primary spending "
                "indexed to CPI with a one-year lag")


def _es_elasticities() -> dict[str, float]:
    from sovereign_cash.es import bottom_up_macro as E

    panel = pd.read_parquet(E.CACHE)
    last = panel.dropna(how="all").iloc[-12:]
    tin = float(last[[k for k in E.IN_KEYS if k in last]].sum().sum())
    tout = float(last[[k for k in E.OUT_KEYS if k in last]].sum().sum())

    def eps(keys, driver, total):
        num = sum(E.BETAS.get(k, 0.0) * float(last[k].sum()) for k in keys
                  if k in last and E.DRIVERS.get(k) == driver)
        return E.LEAF_BLEND * num / total if total else 0.0

    return {"receipts_ngdp": eps(E.IN_KEYS, "ngdp", tin), "primary_cpi": eps(E.OUT_KEYS, "cpi", tout),
            "primary_ngdp": eps(E.OUT_KEYS, "ngdp", tout), "lag": 12}


def _no_petroleum_base(years: list[int]) -> tuple[list[float], str]:
    """Petroleum cash flows in cash-requirement sign, last 12 outturn months held flat."""
    from sovereign_cash.contract import DERIVED

    m = pd.read_parquet(DERIVED / "no_cash_monthly.parquet")
    h = m[~m["is_forecast"].astype(bool)].dropna(subset=["cash_petroleum_bn"]).tail(12)
    level = float(h["cash_petroleum_bn"].sum())
    return [level] * len(years), f"{h['month'].min():%Y-%m}..{h['month'].max():%Y-%m} outturn, held flat"


def _elastic(cc, driver, line, eps, base, taus, *, badge, method, source):
    return {"id": f"{cc}_{driver}_{line}", "type": "elasticity", "driver": driver, "line": line,
            "elasticity": float(eps), "base": [float(b) for b in base], "tau_months": taus,
            "badge": badge, "method": method, "source": source}


def build(cc: str, book, onset: pd.Timestamp) -> dict:
    if book is None:
        raise RuntimeError(f"{cc}: needs the debt book for its baseline")
    years = [y for y in range(book.y0 + 1, book.y0 + 1 + N_YEARS)]
    a = book.annual_table(cc).reindex(years)
    pct = a["deficit_authority_pct_gdp"].where(a["deficit_authority_pct_gdp"].notna(), a["deficit_weo_pct_gdp"])
    pct_src = ["authority" if pd.notna(v) else "IMF WEO" for v in a["deficit_authority_pct_gdp"]]
    total = pct * a["gdp_bn"] / 100.0
    revenue = a["gg_revenue_bn"]
    interest = a["gg_interest_bn"].where(a["gg_interest_bn"].notna(), a["interest_bn"])
    if total.isna().any() or interest.isna().any() or revenue.isna().any():
        raise RuntimeError(f"{cc}: annual table has gaps in {years}")
    primary = total - interest + revenue
    periods = calendar_periods(years)
    t0 = tau_months(periods, onset)
    channels, notes = [], {}

    if cc == "es":
        e = _es_elasticities()
        tl = tau_months(periods, onset, lag=e["lag"])
        src = "sovereign_cash.es.bottom_up_macro (BETAS x IGAE category shares x LEAF_BLEND)"
        channels += [
            _elastic(cc, "ngdp", "receipts", e["receipts_ngdp"], revenue, tl, badge="Model elasticity",
                     method="IGAE bottom-up pass-through on prior-year NGDP growth", source=src),
            _elastic(cc, "cpi", "primary", e["primary_cpi"], primary, tl, badge="Model elasticity",
                     method="IGAE bottom-up pass-through on prior-year HICP", source=src),
        ]
        if e["primary_ngdp"]:
            channels.append(_elastic(cc, "ngdp", "primary", e["primary_ngdp"], primary, tl,
                                     badge="Model elasticity", method="IGAE capital spending on NGDP",
                                     source=src))
        notes["elasticities"] = e
    else:
        rec_base = revenue
        if cc == "no":
            pet, pet_src = _no_petroleum_base(years)
            channels.append(_elastic(cc, "brent", "receipts", -1.0, pet, t0, badge="Model elasticity",
                                     method="Tab4 bottom-up: oil tax and SDFI x (1 + Brent YoY), unit pass-through",
                                     source=f"sovereign_cash.no.bottom_up_macro; petroleum base {pet_src}"))
            # the elasticity carries the sign: petroleum flows are negative cash requirement
            rec_base = revenue + pd.Series(pet, index=years)  # non-petroleum part (pet < 0)
            notes["petroleum_base"] = pet_src
        tl = tau_months(periods, onset, lag=ASSUMED["primary_cpi_lag"])
        channels += [
            _elastic(cc, "ngdp", "receipts", ASSUMED["receipts_ngdp"], rec_base, t0, badge="Assumed elasticity",
                     method=ASSUMED_NOTE, source="assumption"),
            _elastic(cc, "cpi", "primary", ASSUMED["primary_cpi"], primary, tl, badge="Assumed elasticity",
                     method=ASSUMED_NOTE, source="assumption"),
        ]
    ch = book.channel(cc, years)
    if ch is not None:
        channels.append(ch)
        channels += book.curve_channels(cc, years)
    entry = {
        "cc": cc, "name": NAMES[cc], "ccy": CCY[cc], "unit": f"{CCY[cc]} bn",
        "measure": "General government deficit", "period_kind": "calendar year",
        "periods": [str(y) for y in years], "origin": ts(onset - pd.offsets.MonthBegin(1)), "onset": ts(onset),
        "lines": LINES,
        "baseline": {"total": total.tolist(), "lines": {"receipts": revenue.tolist(), "primary": primary.tolist(),
                                                        "interest": interest.tolist()},
                     "source": "deficit % GDP (" + ", ".join(f"{y} {s}" for y, s in zip(years, pct_src))
                               + ") x nominal GDP; GG revenue and interest, sovereign_cash.debt"},
        "channels": channels, "default_methods": {"rates": "book"} if ch is not None else {},
        "modes": {"ngdp": ["level", "growth"], "cpi": ["level", "growth"], "brent": ["level"]},
        "model": {"family": "annual elasticities on the debt-book projection", "notes": notes},
    }
    if (r := book.feedback_rate(cc)) is not None and np.isfinite(r):
        entry["feedback"] = {"rate_pct": [r] * len(years), "source": "debt snapshot marginal new-issue rate"}
    return entry
