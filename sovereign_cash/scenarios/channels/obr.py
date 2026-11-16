"""United Kingdom: OBR ready reckoner, used the way OBR documents it.

The reckoner gives, for each determinant, the effect on each fiscal year of a
one-unit change in a given year. A dashboard driver is a bundle of determinants
moved together (a 1% nominal GDP level shift moves earnings, consumption and
profits by 1%). Level determinants (% levels) take the level gap; rate
determinants (pp: uprating inflation) take the growth in that year; interest
rates are persistent (shifted in every year whatever the mode).

OBR's one-pass CGNCR financing loop (interest on the extra borrowing) is left
out of every channel: interest on extra borrowing is the dashboard's own switch
for all countries. Bank Rate is OBR's channel through interest on central-bank
reserves created by QE; the ISIN book does not hold reserves, so "our model"
for the UK is the book plus this channel.

Measure: public sector net borrowing (PSNB), OBR's fiscal aggregate, not the
CGNCR the UK cash board forecasts. The baseline is OBR's latest EFO PSNB.
"""
from __future__ import annotations

import pandas as pd

from sovereign_cash.scenarios.channels.common import CCY, N_YEARS, NAMES, exposure, period_end_tau, ts, uk_fy_periods

# driver -> (determinant, kind, units per one unit of the driver)
BUNDLES: dict[str, list[tuple[str, str, float]]] = {
    "ngdp": [
        ("Earnings level", "level", 1.0),
        ("Nominal consumption", "level", 1.0),
        ("Non-oil, non-financial profits", "level", 1.0),
        ("Financial profits", "level", 1.0),
        ("Calendar year nominal GDP", "level", 1.0),
        ("Non-seasonally adjusted nominal GDP", "level", 1.0),
    ],
    "cpi": [
        ("CPI index", "level", 1.0),
        ("Average RPI", "level", 1.0),
        ("Q3 CPI", "rate", 1.0),
        ("Q2 RPI", "rate", 1.0),
        ("Q3 CPI for benefits (zero floor)", "rate", 1.0),
        ("RPI inflation", "rate", 1.0),
    ],
    "rates": [  # determinants in pp; the dashboard's rates driver is in bp
        ("Long-term interest rates", "persistent", 0.01),
        ("Bank Rate", "persistent", 0.01),
    ],
}
LOOP = "CGNCR financing loop"
RESERVES = "Bank Rate"
BLOCK_LINE = {"receipts": "receipts", "spending": "primary", "debt_interest": "interest"}
LINES = [{"id": "receipts", "label": "Receipts", "sign": -1},
         {"id": "primary", "label": "Spending ex interest", "sign": 1},
         {"id": "interest", "label": "Debt interest", "sign": 1}]


def _coef(R, det: str, fys: list[str]) -> dict[str, list[list[float]]]:
    """coef[block][j][p], £bn per unit of ``det`` in shock year j, effect year p."""
    out = {b: [[0.0] * len(fys) for _ in fys] for b in BLOCK_LINE}
    for j, fy in enumerate(fys):
        eff = R.run({det: {fy: 1.0}})
        if eff.empty:
            continue
        eff = eff[eff["determinant"] != LOOP]  # the financing loop is the dashboard's own switch
        g = eff.groupby(["block", "effect_fy"])["gbp_bn"].sum()
        for (block, efy), v in g.items():
            if efy in fys and block in out:
                out[block][j][fys.index(efy)] += float(v)
    return out


def build(book, onset: pd.Timestamp) -> dict:
    from sovereign_cash.debt import sources as S
    from sovereign_cash.uk import ready_reckoner as R

    first_fy = onset.year if onset.month >= 4 else onset.year - 1
    fy_labels = [f"{y}-{str(y + 1)[2:]}" for y in range(first_fy, first_fy + N_YEARS)]
    missing = [fy for fy in fy_labels if fy not in R.YEARS]
    fys = [fy for fy in fy_labels if fy in R.YEARS]
    periods = uk_fy_periods(first_fy, len(fys))
    taus = period_end_tau(periods, onset)
    expo = exposure(periods, onset)
    known = set(R.determinants())

    channels, used = [], {}
    for driver, bundle in BUNDLES.items():
        for kind in ("level", "rate", "persistent"):
            dets = [(d, u) for d, k, u in bundle if k == kind and d in known]
            if not dets:
                continue
            tot = {b: [[0.0] * len(fys) for _ in fys] for b in BLOCK_LINE}
            for det, units in dets:
                c = _coef(R, det, fys)
                for b in BLOCK_LINE:
                    for j in range(len(fys)):
                        for p in range(len(fys)):
                            tot[b][j][p] += units * c[b][j][p]
            used[f"{driver}/{kind}"] = [d for d, _ in dets]
            for block, line in BLOCK_LINE.items():
                if not any(any(abs(v) > 0 for v in row) for row in tot[block]):
                    continue
                channels.append({
                    "id": f"uk_{driver}_{kind}_{line}", "type": "matrix", "driver": driver, "line": line,
                    "kind": kind, "tau": taus, "exposure": expo, "coef": tot[block],
                    "determinants": [d for d, _ in dets],
                    "method": ("OBR ready reckoner: shock-year x effect-year coefficients, "
                               f"{kind} determinants moved together; first year pro-rated by exposure"),
                    "badge": "OBR reckoner",
                    "source": R.VINTAGE,
                })

    # our model: QE reserves remunerated at Bank Rate (OBR's coefficient), with the parallel
    # shift ("rates") or the 3-month anchor of a curve shift ("r3m")
    if RESERVES in known:
        c = _coef(R, RESERVES, fys)
        for block, line in BLOCK_LINE.items():
            coef = [[0.01 * v for v in row] for row in c[block]]
            if not any(any(abs(v) > 0 for v in row) for row in coef):
                continue
            for driver, group in (("rates", "book"), ("r3m", None)):
                ch = {
                    "id": f"uk_{driver}_reserves_{line}", "type": "matrix", "driver": driver, "line": line,
                    "kind": "persistent", "tau": taus, "exposure": expo, "coef": coef,
                    "determinants": [RESERVES],
                    "method": ("Interest on QE reserves (and other Bank Rate-linked flows) at OBR's ready-reckoner "
                               "coefficient for Bank Rate; first year pro-rated"),
                    "badge": "QE reserves",
                    "source": R.VINTAGE,
                }
                if group:
                    ch["method_group"] = group
                channels.append(ch)

    obr = S.obr()
    piv = obr.pivot_table(index="year", columns="var", values="value")
    fy_years = [int(fy[:4]) for fy in fys]
    total = [float(piv.at[y, "deficit_bn"]) for y in fy_years]
    interest = [float(piv.at[y, "interest_net_bn"]) for y in fy_years]
    receipts = [float(piv.at[y, "revenue_ex_int_bn"]) for y in fy_years]
    primary = [t - i + r for t, i, r in zip(total, interest, receipts)]

    default_methods: dict[str, str] = {}
    if book is not None:
        # calendar -> April-March fiscal year: 9/12 of y, 3/12 of y+1
        weights = [{y: 0.75, y + 1: 0.25} for y in fy_years]
        ch = book.channel("uk", [], weights=weights, note="; calendar years weighted onto April-March")
        if ch is not None:
            for c in channels:
                if c["driver"] == "rates" and not c.get("method_group"):
                    c["method_group"] = "authority"
            channels.append(ch)
            channels += book.curve_channels("uk", [], weights=weights)
            default_methods["rates"] = "book"
    entry = {
        "cc": "uk", "name": NAMES["uk"], "ccy": CCY["uk"], "unit": "GBP bn",
        "measure": "Public sector net borrowing (OBR)", "period_kind": "fiscal year (April-March)",
        "periods": fys, "origin": ts(onset - pd.offsets.MonthBegin(1)), "onset": ts(onset),
        "lines": LINES,
        "baseline": {"total": total, "lines": {"receipts": receipts, "primary": primary, "interest": interest},
                     "source": str(obr["vintage"].iloc[-1]) + " PSNB"},
        "channels": channels, "default_methods": default_methods,
        "modes": {"ngdp": ["level", "growth"], "cpi": ["level", "growth"]},
        "model": {"family": "OBR ready reckoner", "vintage": R.VINTAGE, "bundles": used,
                  "years_outside_reckoner": missing,
                  "loop": "OBR's CGNCR financing loop excluded; the interest-on-extra-borrowing switch replaces it"},
    }
    if book is not None and (r := book.feedback_rate("uk")) is not None:
        entry["feedback"] = {"rate_pct": [r] * len(fys), "source": "debt snapshot marginal new-issue rate"}
    return entry
