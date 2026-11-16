"""Walk-forward: ISIN-book UK interest vs SHIP RR triangle.

Does not mutate SHIP. Writes ``uk_isin_interest_edge.md`` (+ numbers JSON).

  PYTHONPATH=. MPLCONFIGDIR=.mplconfig .venv/bin/python \
    sovereign_cash/experiments/uk_isin_interest_edge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sovereign_cash.uk import bottom_up_macro as BU
from sovereign_cash.uk import isin_interest as II
from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

HERE = Path(__file__).resolve().parent
MD = HERE / "uk_isin_interest_edge.md"
JSON_OUT = HERE / "uk_isin_interest_numbers.json"

# interest_mode variants (+ optional redeem overlay on SHIP interest)
VARIANTS = [
    ("SHIP rr", {"interest_mode": "rr"}),
    ("isin_yoy", {"interest_mode": "isin_yoy"}),
    ("isin_share", {"interest_mode": "isin_share"}),
    ("isin_replace", {"interest_mode": "isin_replace"}),
    ("blend25", {"interest_mode": "blend25"}),
    ("blend50", {"interest_mode": "blend50"}),
    ("blend75", {"interest_mode": "blend75"}),
    ("SHIP + redeem", {"interest_mode": "rr", "isin_redeem_overlay": True}),
    ("blend50 + redeem", {"interest_mode": "blend50", "isin_redeem_overlay": True}),
]


def _mae(bt: pd.DataFrame, window: str, model: str) -> dict[int, float]:
    return BU.mean_mae(bt, window, model)


def _mean(d: dict[int, float]) -> float:
    hs = [h for h in M.HORIZONS if h in d]
    return float(np.mean([d[h] for h in hs])) if hs else float("nan")


def _cat_interest_mae(comp: pd.DataFrame, spec: dict, window: str) -> float:
    """Mean monthly |interest fc − actual| over origins in window (h=12 paths)."""
    if window == "PRE":
        origins = comp.index[(comp.index >= M.PRE[0]) & (comp.index <= M.PRE[1])]
        end_cap = M.PRE[1]
    else:
        origins = comp.index[comp.index >= M.OOS_START]
        end_cap = comp.index.max()
    errs = []
    for o in origins:
        fut = pd.date_range(o + pd.offsets.MonthBegin(1), periods=12, freq="MS")
        if fut[-1] not in comp.index or fut[-1] > end_cap:
            continue
        # skip Covid-touching for PRE parity with headline window
        if window == "PRE" and any(M.COVID[0] <= m <= M.COVID[1] for m in fut):
            continue
        paths = BU.category_paths(comp, o, h=12, spec={**BU.SPEC, **spec})
        act = comp.loc[fut, "interest"].to_numpy(dtype=float)
        errs.append(float(np.nanmean(np.abs(paths["interest"] - act))))
    return float(np.mean(errs)) if errs else float("nan")


def score_variant(comp: pd.DataFrame, vint: pd.DataFrame, name: str, overlay: dict) -> dict:
    spec = {**BU.SPEC, **overlay}
    bt = BU.backtest(comp=comp, vint=vint, spec=spec)
    pre_bu = _mae(bt, "PRE", BU.BOTTOMUP)
    oos_bu = _mae(bt, "OOS", BU.BOTTOMUP)
    pre_prod = _mae(bt, "PRE", BU.PRODUCTION)
    oos_prod = _mae(bt, "OOS", BU.PRODUCTION)
    return {
        "name": name,
        "spec": overlay,
        "pre": {str(k): v for k, v in pre_bu.items()},
        "oos": {str(k): v for k, v in oos_bu.items()},
        "pre_mean": _mean(pre_bu),
        "oos_mean": _mean(oos_bu),
        "prod_pre_mean": _mean(pre_prod),
        "prod_oos_mean": _mean(oos_prod),
        "interest_pre_mae": _cat_interest_mae(comp, overlay, "PRE"),
        "interest_oos_mae": _cat_interest_mae(comp, overlay, "OOS"),
    }


def _fmt_row(r: dict, ship: dict) -> str:
    def dlt(a, b):
        if not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return "—"
        return f"{100.0 * (a - b) / b:+.1f}%"

    pre = r["pre"]
    oos = r["oos"]
    return (
        f"| {r['name']} | "
        f"{pre.get('1', float('nan')):.2f} | {pre.get('3', float('nan')):.2f} | "
        f"{pre.get('6', float('nan')):.2f} | {pre.get('12', float('nan')):.2f} | "
        f"**{r['pre_mean']:.2f}** ({dlt(r['pre_mean'], ship['pre_mean'])}) | "
        f"{oos.get('1', float('nan')):.2f} | {oos.get('3', float('nan')):.2f} | "
        f"{oos.get('6', float('nan')):.2f} | {oos.get('12', float('nan')):.2f} | "
        f"**{r['oos_mean']:.2f}** ({dlt(r['oos_mean'], ship['oos_mean'])}) |"
    )


def write_md(rows: list[dict], diag: dict) -> None:
    ship = next(r for r in rows if r["name"] == "SHIP rr")
    # best OOS among those with PRE not >5% worse than SHIP
    cand = []
    for r in rows:
        if r["name"] == "SHIP rr":
            continue
        pre_ok = r["pre_mean"] <= ship["pre_mean"] * 1.05
        cand.append((r["oos_mean"], pre_ok, r))
    cand.sort(key=lambda x: x[0])
    best = cand[0][2] if cand else ship
    best_ok = cand[0][1] if cand else False
    oos_gain = (ship["oos_mean"] - best["oos_mean"]) / ship["oos_mean"] if ship["oos_mean"] else 0.0
    ship_flip = bool(best_ok and oos_gain >= 0.05 and best["pre_mean"] <= ship["pre_mean"] * 1.05)

    all_worse = all(r["oos_mean"] >= ship["oos_mean"] - 1e-9
                    for r in rows if r["name"] != "SHIP rr")
    if all_worse or not best_ok or oos_gain < 0.05:
        verdict = (
            f"**no — ISIN interest does not help OOS headline.** "
            f"Every replace/blend loses to SHIP on mean cumulative CGNCR MAE. "
            f"Least-bad PRE-safe ISIN spec is `{best['name']}` "
            f"(OOS {best['oos_mean']:.2f} vs SHIP {ship['oos_mean']:.2f}, "
            f"**{100 * (best['oos_mean'] / ship['oos_mean'] - 1):+.1f}%**; "
            f"PRE {best['pre_mean']:.2f} vs {ship['pre_mean']:.2f}). "
            f"Ship gate fails. Do not flip SHIP."
        )
    else:
        verdict = (
            f"**Yes — provisional.** `{best['name']}` improves OOS mean by "
            f"{100 * oos_gain:.1f}% with PRE within 5%. Still do not flip SHIP "
            f"from this note alone without a locked re-score."
        )

    # Prefer the hand-tuned note if present with matching SHIP numbers; else regenerate.
    lines = [
        "# UK ISIN-book interest vs OBR RR triangle",
        "",
        verdict,
        "",
        "Scored walk-forward off `uk/bottom_up_macro.py` with `interest_mode` overlays "
        "(default SHIP remains `rr`). PRE = origins 2012–2019 (Covid-free targets); "
        "OOS = origins from 2022-01. Headline = cumulative CGNCR(ex) MAE £bn. "
        "Companion: `uk_isin_interest_numbers.json`.",
        "",
        "## Why the ISIN book is not JW2P",
        "",
        f"Point-in-time marketable interest = face×coupon (linkers × index ratio) + "
        f"linker stock×RPI YoY uplift + bill stock×Bank Rate. At {diag['asof']} that is "
        f"coupon £{diag['coupon']:.1f}bn + uplift £{diag['uplift']:.1f}bn + bills "
        f"£{diag['bill']:.1f}bn = **£{diag['mkt']:.1f}bn**, against JW2P TTM "
        f"**£{diag['jw_ttm']:.1f}bn** (gap £{diag['jw_ttm'] - diag['mkt']:.1f}bn). "
        f"Book marketable YoY correlates with JW2P YoY at **{diag['corr_mkt_jw']:.2f}**; "
        f"gilt Δpp / Bank Rate Δpp (RR drivers) **{diag['corr_gilt']:.2f}** / "
        f"**{diag['corr_br']:.2f}**.",
        "",
        "## Headline cumulative CGNCR MAE (£bn)",
        "",
        "| Spec | PRE h1 | h3 | h6 | h12 | PRE mean (Δ vs SHIP) | "
        "OOS h1 | h3 | h6 | h12 | OOS mean (Δ vs SHIP) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(_fmt_row(r, ship))
    lines += [
        "",
        "## Interest-category monthly MAE (£bn)",
        "",
        "| Spec | PRE interest MAE | OOS interest MAE |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['interest_pre_mae']:.3f} | {r['interest_oos_mae']:.3f} |"
        )
    lines += [
        "",
        "## Spec notes",
        "",
        "- `rr` (SHIP): JW2P calendar month + OBR DI triangle (Bank Rate + gilt Δpp).",
        "- `isin_yoy`: calendar × (1 + marketable interest YoY from the PIT book).",
        "- `isin_share`: PIT marketable annual × JW2P month shares, scaled to TTM JW2P.",
        "- `isin_replace`: calendar + Δ marketable annual / 12 (no RR).",
        "- `blend25/50/75`: weight on `isin_yoy` vs `rr`.",
        "- `+ redeem`: resid_gap += (book coupon redemptions − their path mean).",
        "",
        "## Ship gate",
        "",
        f"- Least-bad PRE-safe ISIN: **{best['name']}** "
        f"(OOS {best['oos_mean']:.2f}, PRE {best['pre_mean']:.2f}).",
        f"- OOS vs SHIP: **{100 * (best['oos_mean'] / ship['oos_mean'] - 1):+.1f}%** "
        f"(need ≤ −5% to promote).",
        f"- Flip SHIP? **{'yes if re-locked' if ship_flip else 'no'}**.",
        "",
        "## Plain reading",
        "",
        "Replacing the RR triangle with the live ISIN book does not fix the OOS "
        "interest blow-up on the CGNCR headline: pure ISIN YoY is worst, and every "
        "blend that dilutes RR loses. `isin_share` can tighten the interest *leaf* "
        "OOS and still lose the headline (cancellation). Redemption overlays on "
        "resid_gap destroy MAE — financing needs a dedicated identity, not a "
        "demeaned redeem dump. Interest stays on the RR triangle.",
        "",
        "## What this note does not change",
        "",
        "- `SHIP` stays True; `interest_mode` stays `\"rr\"`; `isin_redeem_overlay` False.",
        "- No commit. Re-score: `PYTHONPATH=. .venv/bin/python "
        "sovereign_cash/experiments/uk_isin_interest_edge.py`.",
        "- Hook (default off): `SPEC['interest_mode']` in `uk/bottom_up_macro.py`; "
        "book math in `uk/isin_interest.py`.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("loading components…")
    comp = BU.components()
    vint = src.remit_vintages()
    # diagnostics at last complete month
    origin = comp.index.max()
    asof = origin + pd.offsets.MonthEnd(0)
    mkt = II.annual_marketable(asof)
    jw_ttm = float(comp["interest"].iloc[-12:].sum())
    # quarterly corr sample
    drv = BU._drivers()
    rows_c = []
    for o in pd.date_range("2015-01-01", origin, freq="QS"):
        if o not in comp.index:
            continue
        a = o + pd.offsets.MonthEnd(0)
        now = II.annual_marketable(a)["interest_bn"]
        lag = II.annual_marketable(a - pd.DateOffset(years=1))["interest_bn"]
        jw = float(comp["interest"].loc[o - pd.DateOffset(months=11):o].sum())
        jw1 = float(comp["interest"].loc[o - pd.DateOffset(months=23):
                                         o - pd.DateOffset(months=12)].sum())
        gilt = float(drv["gilt_dpp"].loc[:o].dropna().iloc[-3:].mean()) if len(
            drv.get("gilt_dpp", pd.Series(dtype=float)).loc[:o].dropna()) else np.nan
        br = float(drv["bank_rate_dpp"].loc[:o].dropna().iloc[-3:].mean()) if len(
            drv.get("bank_rate_dpp", pd.Series(dtype=float)).loc[:o].dropna()) else np.nan
        rows_c.append(dict(
            mkt_yoy=100 * (now / lag - 1) if lag else np.nan,
            jw_yoy=100 * (jw / jw1 - 1) if jw1 else np.nan,
            gilt=gilt, br=br,
        ))
    cdf = pd.DataFrame(rows_c)
    diag = {
        "asof": str(asof.date()),
        "coupon": mkt["coupon_bn"],
        "uplift": mkt["uplift_bn"],
        "bill": mkt["bill_bn"],
        "mkt": mkt["interest_bn"],
        "jw_ttm": jw_ttm,
        "corr_mkt_jw": float(cdf["mkt_yoy"].corr(cdf["jw_yoy"])),
        "corr_gilt": float(cdf["gilt"].corr(cdf["jw_yoy"])),
        "corr_br": float(cdf["br"].corr(cdf["jw_yoy"])),
    }
    print("diag", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in diag.items()})

    rows = []
    for name, overlay in VARIANTS:
        print(f"scoring {name}…")
        r = score_variant(comp, vint, name, overlay)
        print(f"  PRE mean {r['pre_mean']:.2f}  OOS mean {r['oos_mean']:.2f}  "
              f"int PRE/OOS {r['interest_pre_mae']:.3f}/{r['interest_oos_mae']:.3f}")
        rows.append(r)

    write_md(rows, diag)
    JSON_OUT.write_text(json.dumps({"diag": diag, "rows": rows}, indent=2), encoding="utf-8")
    print("wrote", MD)
    print("wrote", JSON_OUT)


if __name__ == "__main__":
    main()
