"""The note's debt arithmetic on its own primary engine, not on the WEO deficit.

``debt.dalio_note.sustain`` takes today's primary as GG interest less the WEO GG deficit. The note's gap, debt paths and
delivery heatmap are on the engine primary (``scenarios.primary``: cash / share / DTS through T+3, legislated policy,
general-government basis; the US line model to FY2030) and on the IMF WEO primary for the plan, as the note's text and
Exhibit 1 caption say. Everything else (d0, nominal growth, marginal rate, repricing speed, revenue) is dalio_note's.

    from sovereign_cash.debt.note.sustain_engine import sustain
"""
from __future__ import annotations

import pandas as pd

PLAN_YEAR = 2030
NOW_YEAR = 2026


def engine_pb(cc: str) -> dict[str, float]:
    """Primary balance, % GDP (+ = surplus): today from the engine, plan from the IMF WEO."""
    from sovereign_cash.scenarios import primary as P

    p = P.path(cc)
    now = -float(p.at[NOW_YEAR, "primary_gg_pct"])
    w = P.weo_primary_deficit_pct(cc)
    plan = -float(w[PLAN_YEAR]) if PLAN_YEAR in w.index else -float(p.at[PLAN_YEAR, "primary_gg_pct"])
    return {"now": now, "plan": plan}


def sustain(horizon: int = 10) -> dict:
    from sovereign_cash.debt import dalio_note as DN
    from sovereign_cash.debt import signal as S

    base = DN.sustain(horizon)
    a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet").set_index(["cc", "year"])
    rows, curves, paths = {}, {}, {}
    for cc, r in base["T"].iterrows():
        pb = engine_pb(cc)
        d0, g, rm, rho = float(r["d0"]), float(r["g"]) / 100, float(r["rm"]) / 100, float(r["rho"])
        r0 = float(a.loc[(cc, NOW_YEAR), "gg_interest_pct_gdp"]) / d0
        rev = float(a.loc[(cc, NOW_YEAR), "gg_revenue_bn"] / a.loc[(cc, NOW_YEAR), "gdp_bn"])

        def sim(shock_bp: float, p: float) -> tuple[list[float], float]:
            d, rr, ds = d0, r0, [d0]
            for _ in range(horizon):
                rr += rho * (rm + shock_bp / 1e4 - rr)
                it = rr * d
                d = d * (1 + rr) / (1 + g) - p
                ds.append(d)
            return ds, it / rev

        pbs = (rm - g) / (1 + g) * d0
        rs = {k: g + v / 100 * (1 + g) / (d0 / 100) for k, v in pb.items()}
        rows[cc] = {**r.to_dict(), "pb_now": pb["now"], "pb_plan": pb["plan"], "pbs": pbs,
                    "adj_now": pbs - pb["now"], "adj_plan": pbs - pb["plan"], "rs_now": 100 * rs["now"],
                    "rs_plan": 100 * rs["plan"], "gap_now": 1e4 * (rs["now"] - rm), "gap_plan": 1e4 * (rs["plan"] - rm),
                    "ir10": sim(0, pb["now"])[1], "ir10_100": sim(100, pb["now"])[1], "ir10_300": sim(300, pb["now"])[1],
                    "d10": sim(0, pb["now"])[0][-1], "d10_plan": sim(0, pb["plan"])[0][-1],
                    "adj_eff": (r0 - g) / (1 + g) * d0 - pb["now"], "dstar_now": pb["now"] * (1 + g) / (rm - g),
                    "step": [sim(x, pb["now"])[1] - sim(x - 100, pb["now"])[1] for x in (100, 200, 300)]}
        curves[cc] = [sim(x, pb["now"])[1] for x in DN.SHOCKS]
        paths[cc] = {"now": sim(0, pb["now"])[0], "plan": sim(0, pb["plan"])[0]}
    return {"T": pd.DataFrame(rows).T, "curves": curves, "paths": paths, "horizon": horizon}
