"""Walk-forward score of the debt-ratio recursion (Methodology ``debtpath``).

At each complete outturn origin ``o``, project debt/GDP h = 1…H years with:

  d_{t+1} = d_t (1+r_t)/(1+g_t) − pb_t

using the origin's effective rate r_0 (interest/debt), held primary surplus pb_o, and
**realised** year-by-year nominal GDP growth (same information set as the share-engine
test: macro known; the recursion is what is scored). r_t reprices toward the trailing
3y mean new effective rate at ρ = 1 − (1 − s_3y)^{1/3} with s_3y = 0.35 default when
the ISIN book is unavailable for that origin.

Benchmarks: flat debt ratio; last-Δ persistence.

PRE = origins ≤ 2019; OOS = origins ≥ 2022; Covid origin years 2020–21 dropped.
Writes ``derived/debt_path_score.json``.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.debt_path_score
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SC = Path(__file__).resolve().parents[1]
DEBT = SC / "data" / "debt"
DERIVED = SC / "derived"
OUT = DERIVED / "debt_path_score.json"
CCS = ("us", "uk", "de", "fr", "it", "jp", "es", "se")
H_MAX = 5
S3Y_DEFAULT = 0.35
COVID = {2020, 2021}
LAST_OUTTURN = int(pd.Timestamp.now().year) - 1


def _panel(cc: str) -> pd.DataFrame:
    a = pd.read_parquet(DEBT / "ameco.parquet")
    p = a[a["country"] == cc].pivot_table(index="year", columns="var", values="value").astype(float)
    p = p.loc[:LAST_OUTTURN].dropna(subset=["debt_pct", "balance_pct", "interest_gross_pct", "gdp_lcu_bn"])
    p = p.copy()
    p["g"] = p["gdp_lcu_bn"].pct_change()
    p["r"] = p["interest_gross_pct"] / p["debt_pct"].replace(0, np.nan)
    p["pb"] = p["balance_pct"] + p["interest_gross_pct"]  # primary surplus % GDP
    return p.dropna(subset=["g", "r", "pb", "debt_pct"])


def _rho(s3y: float = S3Y_DEFAULT) -> float:
    return 1.0 - (1.0 - float(s3y)) ** (1.0 / 3.0)


def project(d0: float, r0: float, pb: float, g_path: list[float], r_bar: float,
            rho: float) -> list[float]:
    """Return [d0, d1, … dH] along g_path (len H)."""
    d, r, out = float(d0), float(r0), [float(d0)]
    for g in g_path:
        r = r + rho * (r_bar - r)
        d = d * (1.0 + r) / (1.0 + g) - pb
        out.append(float(d))
    return out


def score_cc(cc: str) -> dict:
    p = _panel(cc)
    years = [int(y) for y in p.index]
    rho = _rho()
    rows = []
    for o in years:
        if o in COVID or o + 1 not in p.index:
            continue
        # need at least h=1 realised
        for h in range(1, H_MAX + 1):
            tgt = o + h
            if tgt not in p.index or tgt in COVID:
                continue
            g_path = [float(p.at[o + k, "g"]) for k in range(1, h + 1)]
            if any(not np.isfinite(g) for g in g_path):
                continue
            d0 = float(p.at[o, "debt_pct"])
            r0 = float(p.at[o, "r"])
            pb = float(p.at[o, "pb"])
            # trailing mean effective rate as "marginal" proxy at origin
            trail = p.loc[:o, "r"].tail(3)
            r_bar = float(trail.mean()) if len(trail) else r0
            path = project(d0, r0, pb, g_path, r_bar, rho)
            d_hat = path[-1]
            d_act = float(p.at[tgt, "debt_pct"])
            d_flat = d0
            d_delta = d0 + h * float(p.at[o, "debt_pct"] - p.at[o - 1, "debt_pct"]) if o - 1 in p.index else d0
            rows.append({
                "origin": o, "h": h, "year": tgt,
                "err": d_hat - d_act,
                "err_flat": d_flat - d_act,
                "err_delta": d_delta - d_act,
                "d_hat": d_hat, "d_act": d_act,
            })
    if not rows:
        return {"cc": cc, "error": "no rows"}
    df = pd.DataFrame(rows)
    df["window"] = np.where(df["origin"] <= 2019, "PRE",
                            np.where(df["origin"] >= 2022, "OOS", "COVID"))
    out: dict = {"cc": cc, "rho": round(rho, 4), "s3y": S3Y_DEFAULT, "n": int(len(df))}
    for w in ("PRE", "OOS"):
        sub = df[df["window"] == w]
        if sub.empty:
            out[w] = None
            continue
        by_h = {}
        for h, g in sub.groupby("h"):
            by_h[int(h)] = {
                "n": int(len(g)),
                "mae": round(float(g["err"].abs().mean()), 3),
                "mae_flat": round(float(g["err_flat"].abs().mean()), 3),
                "mae_delta": round(float(g["err_delta"].abs().mean()), 3),
                "bias": round(float(g["err"].mean()), 3),
            }
        mean_mae = float(np.mean([v["mae"] for v in by_h.values()]))
        mean_flat = float(np.mean([v["mae_flat"] for v in by_h.values()]))
        mean_delta = float(np.mean([v["mae_delta"] for v in by_h.values()]))
        out[w] = {
            "by_h": by_h,
            "mae": round(mean_mae, 3),
            "mae_flat": round(mean_flat, 3),
            "mae_delta": round(mean_delta, 3),
            "edge_vs_flat": round((mean_flat - mean_mae) / mean_flat, 4) if mean_flat else None,
            "edge_vs_delta": round((mean_delta - mean_mae) / mean_delta, 4) if mean_delta else None,
            "beats_flat": mean_mae < mean_flat - 1e-9,
            "beats_delta": mean_mae < mean_delta - 1e-9,
        }
    # ship arithmetic as the board path if it beats both benchmarks on PRE and does not lose OOS by >25%
    pre, oos = out.get("PRE"), out.get("OOS")
    ship = False
    reason = "no PRE window"
    if pre and oos and pre["beats_flat"] and pre["beats_delta"]:
        if oos["mae"] <= 1.25 * min(oos["mae_flat"], oos["mae_delta"]) + 1e-9:
            ship = True
            reason = (f"PRE mae {pre['mae']:.2f} vs flat {pre['mae_flat']:.2f} / Δ {pre['mae_delta']:.2f}; "
                      f"OOS {oos['mae']:.2f} vs flat {oos['mae_flat']:.2f} / Δ {oos['mae_delta']:.2f}")
        else:
            reason = f"PRE clear; OOS lose >25% vs benchmarks ({oos['mae']:.2f})"
    elif pre and pre["beats_flat"] and pre["beats_delta"]:
        reason = f"PRE clear; OOS missing ({pre['mae']:.2f} vs flat {pre['mae_flat']:.2f})"
    elif pre:
        reason = (f"PRE miss vs flat/Δ ({pre['mae']:.2f} vs {pre['mae_flat']:.2f}/{pre['mae_delta']:.2f})"
                  + (f"; OOS {oos['mae']:.2f}" if oos else ""))
    out["ship_arithmetic"] = ship
    out["reason"] = reason
    return out


def build(countries=CCS) -> dict:
    res = {"built_at": pd.Timestamp.utcnow().isoformat() + "Z", "countries": {}, "pool": {}}
    for cc in countries:
        try:
            res["countries"][cc] = score_cc(cc)
        except Exception as exc:  # noqa: BLE001
            res["countries"][cc] = {"cc": cc, "error": f"{type(exc).__name__}: {exc}"}
    # pool PRE/OOS
    for w in ("PRE", "OOS"):
        maes, flats, deltas = [], [], []
        for c in res["countries"].values():
            block = c.get(w)
            if not block:
                continue
            maes.append(block["mae"])
            flats.append(block["mae_flat"])
            deltas.append(block["mae_delta"])
        if maes:
            res["pool"][w] = {
                "mae": round(float(np.mean(maes)), 3),
                "mae_flat": round(float(np.mean(flats)), 3),
                "mae_delta": round(float(np.mean(deltas)), 3),
                "n_cc": len(maes),
                "beats_flat": float(np.mean(maes)) < float(np.mean(flats)),
                "beats_delta": float(np.mean(maes)) < float(np.mean(deltas)),
            }
    ships = [c["cc"] for c in res["countries"].values() if c.get("ship_arithmetic")]
    res["ship_countries"] = ships
    res["verdict"] = (
        f"Debt recursion beats flat and last-Δ on PRE and clears OOS band in {len(ships)}/{len(countries)} "
        f"countries ({', '.join(ships) or 'none'}). Pool PRE mae {res['pool'].get('PRE', {}).get('mae')} vs "
        f"flat {res['pool'].get('PRE', {}).get('mae_flat')}; OOS {res['pool'].get('OOS', {}).get('mae')} vs "
        f"flat {res['pool'].get('OOS', {}).get('mae_flat')}. Macro growth is realised (same information set as "
        f"the share engine); the scored object is the recursion, not a joint macro+debt forecast."
    )
    DERIVED.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = build()
    print(r["verdict"])
    for cc, c in r["countries"].items():
        if c.get("error"):
            print(cc, "ERR", c["error"])
        else:
            print(cc, "SHIP" if c.get("ship_arithmetic") else "NO", c.get("reason"),
                  "PRE", (c.get("PRE") or {}).get("mae"), "OOS", (c.get("OOS") or {}).get("mae"))
