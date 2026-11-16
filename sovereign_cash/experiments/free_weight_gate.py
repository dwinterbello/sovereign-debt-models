"""Gate free blend/shrink weights.

ES ``LEAF_BLEND`` and NO ``MAINLAND_RESID_WEIGHT`` are re-scored on the same
meters the country ship rules use; DE open-year Soll blends stay on the scored
experiment pointer.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.experiments.free_weight_gate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "free_weight_gate.json"


def _es() -> dict:
    """Rest-of-year |err|/|year| vs seas_add_3, same halves as thin_cash / STATUS."""
    from sovereign_cash import thin_cash as TC
    from sovereign_cash.es import bottom_up_macro as B
    from sovereign_cash.monthly_components import monthly_drivers

    locked = float(B.LEAF_BLEND)
    y = TC.load("es")
    panel = B.category_panel()
    drivers = monthly_drivers("es")
    start = TC.fy_start_month("es")
    prod = TC.rest_errors(y, "es", "seas_add_3")
    pre_prod = TC.mean_ratio(prod, fy_max=2019)
    oos_prod = TC.mean_ratio(prod, fy_min=2022)

    def score_blend(blend: float) -> dict:
        B.LEAF_BLEND = blend
        rows = []
        fys = sorted({TC.fy_of(t, start) for t in y.index})
        for fy in fys:
            block = TC._fy_block(y, fy, start)
            if block is None:
                continue
            year_total = float(block.sum())
            if abs(year_total) < 1e-9:
                continue
            for k in TC.ORIGIN_MONTHS:
                origin = TC.month_ts(fy, k, start)
                if origin not in y.index:
                    continue
                n = 12 - k
                fc = B.forecast_cash(panel.loc[:origin], origin, n, drivers=drivers)
                if fc is None or fc.isna().any():
                    continue
                act_idx = [TC.month_ts(fy, m, start) for m in range(k + 1, 13)]
                act = float(y.loc[act_idx].sum())
                err = float(fc.sum()) - act
                rows.append({"fy": fy, "k": k, "ratio": abs(err) / abs(year_total)})
        err = pd.DataFrame(rows)
        pre = TC.mean_ratio(err, fy_max=2019)
        oos = TC.mean_ratio(err, fy_min=2022)
        ship = (
            np.isfinite(pre) and np.isfinite(oos)
            and pre <= (1.0 - 0.05) * pre_prod + 1e-12
            and oos <= (1.0 + 0.25) * oos_prod + 1e-12
        )
        reason = (
            f"PRE {pre:.3f} vs seas_add_3 {pre_prod:.3f}; "
            f"OOS {oos:.3f} vs {oos_prod:.3f}"
            + ("" if ship else "; FAIL gate")
        )
        return {"blend": blend, "ship": bool(ship), "reason": reason,
                "pre": None if not np.isfinite(pre) else round(float(pre), 4),
                "oos": None if not np.isfinite(oos) else round(float(oos), 4)}

    cands = [0.5, 0.6, 0.7, 0.8, 1.0]
    rows = []
    base = locked
    for b in cands:
        try:
            rows.append(score_blend(b))
        except Exception as exc:  # noqa: BLE001
            rows.append({"blend": b, "error": f"{type(exc).__name__}: {exc}"})
    B.LEAF_BLEND = base

    ships = [r for r in rows if r.get("ship")]
    keep, reason = locked, f"incumbent LEAF_BLEND={locked} retained"
    if ships:
        # Prefer lowest PRE among shipping; break ties toward incumbent.
        ships_sorted = sorted(
            ships,
            key=lambda r: (r["pre"] if r.get("pre") is not None else 9e9,
                           abs(r["blend"] - locked)),
        )
        best = ships_sorted[0]
        if any(abs(r["blend"] - locked) < 1e-9 for r in ships):
            keep = locked
            reason = (
                f"LEAF_BLEND={locked} ships on rest-of-year gate; "
                f"shipping set {[r['blend'] for r in ships]}; "
                f"best PRE among ships = {best['blend']} @ {best['pre']}"
            )
        else:
            keep = best["blend"]
            reason = f"incumbent failed; promoting {keep} (PRE {best['pre']})"
    elif not any(r.get("error") for r in rows):
        reason = (
            f"no blend cleared PRE≥5% vs seas_add_3 and OOS≤+25%; "
            f"incumbent {locked} retained as research default"
        )

    return {"cc": "es", "param": "LEAF_BLEND", "locked": keep, "incumbent": locked,
            "reason": reason, "prod_pre": round(float(pre_prod), 4),
            "prod_oos": round(float(oos_prod), 4), "grid": rows}


def _no() -> dict:
    """Beyond-NB PRE/OOS mean MAE vs smly — same ship_decision meter as production."""
    from sovereign_cash.no import bottom_up_macro as B
    from sovereign_cash.no import build as NB
    from sovereign_cash.no import fetch

    locked = float(B.MAINLAND_RESID_WEIGHT)
    status_fallback = {
        0.5: {"pre_bu": 5.95, "pre_smly": 6.73, "oos_bu": 14.55, "oos_smly": 14.97,
              "note": "STATUS_NO shipped row"},
        1.0: {"pre_bu": 6.35, "pre_smly": 6.73, "oos_bu": 14.58, "oos_smly": 14.97,
              "note": "STATUS_NO w=1 mainland resid"},
    }

    def live_grid() -> list[dict] | None:
        try:
            fl = NB.daily_flows(fetch.read("liquidity_statistics"))
            vin = NB.vintages(fetch.read("liquidity_forecast"))
            m = B.monthly_tab4(fl)
            drivers = B.macro_drivers()
            rules = B.select_use_macro(m, drivers)
        except Exception:
            return None
        rows = []
        base = B.MAINLAND_RESID_WEIGHT
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            B.MAINLAND_RESID_WEIGHT = w
            try:
                pre = B.score_beyond_nb(m, vin, drivers, rules, "PRE")
                oos = B.score_beyond_nb(m, vin, drivers, rules, "OOS")
                ship, reason = B.ship_decision(pre)
                rows.append({
                    "weight": w, "ship": bool(ship), "reason": reason,
                    "pre": round(B.mean_mae(pre, "bottom_up"), 3),
                    "pre_smly": round(B.mean_mae(pre, "production_smly"), 3),
                    "oos": round(B.mean_mae(oos, "bottom_up"), 3),
                    "oos_smly": round(B.mean_mae(oos, "production_smly"), 3),
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({"weight": w, "error": f"{type(exc).__name__}: {exc}"})
        B.MAINLAND_RESID_WEIGHT = base
        return rows

    rows = live_grid()
    source = "live score_beyond_nb"
    if rows is None or all(r.get("error") for r in rows):
        source = "STATUS_NO.md scored table (live reload failed)"
        rows = []
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s = status_fallback.get(w)
            if s is None:
                rows.append({"weight": w, "error": "not scored in STATUS_NO table"})
                continue
            pre, prod = s["pre_bu"], s["pre_smly"]
            ship = pre < (1.0 - B.EDGE) * prod
            rows.append({
                "weight": w, "ship": bool(ship), "reason": s.get("note", ""),
                "pre": s["pre_bu"], "pre_smly": s["pre_smly"],
                "oos": s["oos_bu"], "oos_smly": s["oos_smly"],
            })

    ships = [r for r in rows if r.get("ship")]
    keep, reason = locked, f"incumbent MAINLAND_RESID_WEIGHT={locked} retained"
    if ships:
        ships_sorted = sorted(
            ships,
            key=lambda r: (r["pre"] if r.get("pre") is not None else 9e9,
                           abs(r["weight"] - locked)),
        )
        best = ships_sorted[0]
        if any(abs(r["weight"] - locked) < 1e-9 for r in ships):
            keep = locked
            reason = (
                f"MAINLAND_RESID_WEIGHT={locked} ships; "
                f"shipping set {[r['weight'] for r in ships]}; "
                f"best PRE among ships = {best['weight']} @ {best['pre']} ({source})"
            )
        else:
            keep = best["weight"]
            reason = f"incumbent failed; promoting {keep} ({source})"
    return {"cc": "no", "param": "MAINLAND_RESID_WEIGHT", "locked": keep,
            "incumbent": locked, "reason": reason, "source": source, "grid": rows}


def _de_open() -> dict:
    return {
        "cc": "de", "param": "OPEN_MODE / OPEN_YTD_WEIGHT_*",
        "locked": "soll + hist/YTD 80/20 spend, 95/5 revenue",
        "reason": "Scored in experiments/de_post2020_open_year.md: dropping Soll doubles OOS error; "
                  "hist_year clears PRE slightly then collapses OOS. Weights are part of the shipped Soll path, "
                  "not free parameters without a test.",
        "source": "sovereign_cash/experiments/de_post2020_open_year.md",
    }


def main() -> dict:
    out = {"es": _es(), "no": _no(), "de": _de_open()}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in out.items()},
                     indent=2))
    return out


if __name__ == "__main__":
    main()
