"""Fair audit: FR seas×CPI/equity overlay vs live PLF bottom-up.

The research overlay beat bare seas_mult_3 on PRE but lost to live BUM
(~8.3 vs ~7.4). That comparison mixes information sets: BUM pins PLF
category annuals when known and soft-pins the path to the PLF solde
(α=1); the overlay was scored on open seasonals with neither.

This script re-scores apples-to-apples hybrids on PRE / mid / OOS and
writes ``fr_overlay_vs_plf_audit.md``. Does not mutate SHIP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import bottom_up_macro as BUM
from sovereign_cash.yoy_components import SPEC, components, deficit
from sovereign_cash.experiments import fr_seas_macro_augment as AUG

OUT = Path(__file__).resolve().parent / "fr_overlay_vs_plf_audit.md"
STEP = 1  # dense walk so mid/OOS counts match BUM ship_decision
HORIZONS = C.HORIZONS
PRE_END = C.PRE_END
OOS_START = C.OOS_START

# Candidate from fr_seas_macro_augment_STATUS.md (not shipped).
EPS_PRI_HICP = 0.5
EPS_REC_HICP = 0.25
EPS_REC_EQ = 0.17 * 0.75  # IS-share × ε
EQ_KEY = "eq_de"
LAG = 1
BLEND_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def _yoy_at(s: pd.Series, origin: pd.Timestamp, lag: int) -> float:
    return AUG._yoy_at(s, origin, lag)


def overlay_path(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                 drv: dict[str, pd.Series]) -> pd.Series | None:
    """seas_mult_3 components × candidate CPI/equity scales; deficit identity."""
    hicp = _yoy_at(drv["hicp"], origin, LAG) / 100.0
    eq = _yoy_at(drv[EQ_KEY], origin, LAG) / 100.0
    g_pri = EPS_PRI_HICP * hicp
    g_rec = EPS_REC_HICP * hicp + EPS_REC_EQ * eq
    cols = {}
    for name, _ in SPEC["fr"]:
        y = frame[name].dropna()
        fc = C.forecast(y, origin, h, "seas_mult_3")
        if fc is None:
            fc = C.forecast(y, origin, h, "smly")
        if fc is None:
            return None
        if name == "primary":
            fc = fc * (1.0 + g_pri)
        elif name == "recettes":
            fc = fc * (1.0 + g_rec)
        cols[name] = fc
    return deficit("fr", cols)


def soft_pin_path(path: pd.Series, hist: pd.Series, origin: pd.Timestamp,
                  alpha: float) -> pd.Series:
    return BUM._soft_pin(path, hist, origin, alpha)


def year_has_plf_pin(origin: pd.Timestamp, year: int) -> bool:
    """True when BUM would take PLF category annuals for that year."""
    return BUM._plf_categories(year, origin) is not None


def hybrid_a_month_shape(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                         drv: dict[str, pd.Series],
                         y_hist: pd.Series) -> pd.Series | None:
    """BUM annuals (incl. PLF pin / hold) + overlay within-year month shares.

    For each calendar year in the window: take BUM's year total, redistribute
    across months with the overlay path's within-year shares. Soft-pin α matches
    live BUM (1.0) so the annual level stays on the same footing.
    """
    bum = BUM.path(origin, h, frame, soft_pin_alpha=0.0)
    ov = overlay_path(origin, h, frame, drv)
    if bum is None or ov is None:
        return None
    out = bum.copy()
    for yr in sorted(set(int(m.year) for m in out.index)):
        m = out.index.year == yr
        if not m.any():
            continue
        ov_y = ov.loc[m]
        s = float(ov_y.sum())
        if abs(s) < 1e-9:
            continue
        # Keep BUM year total; replace month shape with overlay shares.
        tot = float(bum.loc[m].sum())
        out.loc[m] = tot * (ov_y / s).to_numpy()
    return soft_pin_path(out, y_hist, origin, BUM.SOFT_PIN_ALPHA)


def hybrid_b_residual(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                      drv: dict[str, pd.Series], y_hist: pd.Series,
                      alpha: float) -> pd.Series | None:
    """path = BUM_α1 + α · (overlay_pinned − BUM_α1)."""
    bum = BUM.path(origin, h, frame, soft_pin_alpha=1.0)
    ov = overlay_path(origin, h, frame, drv)
    if bum is None or ov is None:
        return None
    ov_p = soft_pin_path(ov, y_hist, origin, 1.0)
    return bum + alpha * (ov_p - bum)


def hybrid_c_overlay_pinned(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                            drv: dict[str, pd.Series],
                            y_hist: pd.Series) -> pd.Series | None:
    """Critical fairness: overlay month path + same soft-pin α=1 to PLF solde."""
    ov = overlay_path(origin, h, frame, drv)
    if ov is None:
        return None
    return soft_pin_path(ov, y_hist, origin, 1.0)


def hybrid_d_switch(origin: pd.Timestamp, h: int, frame: pd.DataFrame,
                    drv: dict[str, pd.Series],
                    y_hist: pd.Series) -> pd.Series | None:
    """Overlay months in years without PLF category pin; BUM when PLF known."""
    bum = BUM.path(origin, h, frame, soft_pin_alpha=0.0)
    ov = overlay_path(origin, h, frame, drv)
    if bum is None or ov is None:
        return None
    out = ov.copy()
    for yr in sorted(set(int(m.year) for m in out.index)):
        m = out.index.year == yr
        if year_has_plf_pin(origin, yr):
            out.loc[m] = bum.loc[m].to_numpy()
    return soft_pin_path(out, y_hist, origin, 1.0)


def walk() -> pd.DataFrame:
    frame = components("fr")
    y = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
    drv = AUG.drivers()
    last = y.index.max()
    rows = []
    for i, origin in enumerate(y.index):
        if i % STEP:
            continue
        if origin.year < 2012:
            continue
        end12 = origin + pd.DateOffset(months=12)
        if end12 > last or len(y.loc[:origin]) < 48:
            continue
        act12 = y.loc[origin + pd.offsets.MonthBegin(1):end12]
        if len(act12) != 12 or act12.isna().any():
            continue

        seas = C.forecast(y, origin, 12, "seas_mult_3")
        bum1 = BUM.path(origin, 12, frame, soft_pin_alpha=1.0)
        bum0 = BUM.path(origin, 12, frame, soft_pin_alpha=0.0)
        ov = overlay_path(origin, 12, frame, drv)
        if seas is None or bum1 is None or bum0 is None or ov is None:
            continue
        ha = hybrid_a_month_shape(origin, 12, frame, drv, y)
        hc = hybrid_c_overlay_pinned(origin, 12, frame, drv, y)
        hd = hybrid_d_switch(origin, 12, frame, drv, y)
        hbs = {a: hybrid_b_residual(origin, 12, frame, drv, y, a)
               for a in BLEND_ALPHAS}
        if ha is None or hc is None or hd is None or any(v is None for v in hbs.values()):
            continue

        # How much of the 12m window sits under a PLF category pin / soft-pin year?
        months = seas.index
        n_plf = sum(1 for m in months if year_has_plf_pin(origin, int(m.year)))
        plans = {int(yy) for yy, r in BUM._auth().get("state_deficit", {}).get("years", {}).items()
                 if BUM._plf_known(origin, int(yy))}
        n_soft = sum(1 for m in months if int(m.year) in plans)

        paths = {
            "seas": seas,
            "bum_a1": bum1,
            "bum_a0": bum0,
            "overlay": ov,
            "hy_a_shape": ha,
            "hy_c_ov_pin": hc,
            "hy_d_switch": hd,
            **{f"hy_b_a{a}": hbs[a] for a in BLEND_ALPHAS},
        }
        for h in HORIZONS:
            end = origin + pd.DateOffset(months=h)
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h:
                continue
            a = float(act.sum())
            row = {
                "origin": origin, "h": int(h), "actual": a,
                "n_plf_months": n_plf, "n_soft_months": n_soft,
            }
            for name, p in paths.items():
                row[name] = float(p.iloc[:h].sum())
            rows.append(row)
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    ends = [o + pd.DateOffset(months=int(h)) for o, h in zip(bt["origin"], bt["h"])]
    bt["window"] = np.where(pd.to_datetime(ends) <= PRE_END, "PRE",
                            np.where(bt["origin"] >= OOS_START, "OOS", "mid"))
    return bt


def mean_mae(bt: pd.DataFrame, window: str, col: str) -> float | None:
    g = bt[bt["window"] == window]
    if g.empty:
        return None
    by_h = g.groupby("h").apply(
        lambda x: (x[col] - x["actual"]).abs().mean(), include_groups=False)
    if set(by_h.index) != set(HORIZONS):
        return None
    return float(by_h.loc[list(HORIZONS)].mean())


def mae_by_h(bt: pd.DataFrame, window: str, col: str) -> dict[int, float]:
    g = bt[bt["window"] == window]
    out = {}
    for h, x in g.groupby("h"):
        out[int(h)] = float((x[col] - x["actual"]).abs().mean())
    return out


def pin_coverage(bt: pd.DataFrame) -> dict:
    """Share of h=12 origins with any PLF / soft-pin months, by window."""
    h12 = bt[bt["h"] == 12]
    out = {}
    for w, g in h12.groupby("window"):
        out[w] = {
            "n": int(len(g)),
            "frac_any_plf": float((g["n_plf_months"] > 0).mean()),
            "frac_any_soft": float((g["n_soft_months"] > 0).mean()),
            "mean_plf_months": float(g["n_plf_months"].mean()),
            "mean_soft_months": float(g["n_soft_months"].mean()),
        }
    return out


COLS = [
    ("seas", "seas_mult_3"),
    ("bum_a1", "BUM α=1 (live)"),
    ("bum_a0", "BUM α=0"),
    ("overlay", "overlay alone"),
    ("hy_a_shape", "a) BUM annual + overlay shape"),
    ("hy_b_a0.25", "b) BUM+0.25·(ov−BUM)"),
    ("hy_b_a0.5", "b) BUM+0.5·(ov−BUM)"),
    ("hy_b_a0.75", "b) BUM+0.75·(ov−BUM)"),
    ("hy_b_a1.0", "b) BUM+1.0·(ov−BUM) [=ov pinned]"),
    ("hy_c_ov_pin", "c) overlay + soft-pin α=1"),
    ("hy_d_switch", "d) overlay iff no PLF; else BUM"),
]


def render_md(bt: pd.DataFrame) -> str:
    """Rebuild the audit note from scored numbers (kept in sync with the hand note)."""
    cov = pin_coverage(bt)
    windows = ("PRE", "mid", "OOS")
    table = {lab: {w: mean_mae(bt, w, col) for w in windows} for col, lab in COLS}

    def fmt(x):
        return "—" if x is None else f"{x:.2f}"

    seas_pre, ov_pre = table["seas_mult_3"]["PRE"], table["overlay alone"]["PRE"]
    bum1_pre = table["BUM α=1 (live)"]["PRE"]
    seas_oos, ov_oos = table["seas_mult_3"]["OOS"], table["overlay alone"]["OOS"]
    bum1_oos, bum0_oos = table["BUM α=1 (live)"]["OOS"], table["BUM α=0"]["OOS"]
    ovpin_oos = table["c) overlay + soft-pin α=1"]["OOS"]
    fair_keys = [
        "BUM α=1 (live)", "BUM α=0", "c) overlay + soft-pin α=1",
        "a) BUM annual + overlay shape", "d) overlay iff no PLF; else BUM",
        "b) BUM+0.25·(ov−BUM)", "b) BUM+0.5·(ov−BUM)", "b) BUM+0.75·(ov−BUM)",
    ]
    fair_oos = {k: table[k]["OOS"] for k in fair_keys if table[k]["OOS"] is not None}
    winner = min(fair_oos, key=fair_oos.get)
    winner_mae = fair_oos[winner]
    pre_gap = (ov_pre - bum1_pre) if ov_pre and bum1_pre else None
    oos_gap = (ov_oos - bum1_oos) if ov_oos and bum1_oos else None
    oos_gap_pin = (ovpin_oos - bum1_oos) if ovpin_oos and bum1_oos else None
    c_oos = cov.get("OOS", {})

    lines = [
        "# FR overlay vs PLF bottom-up — fairness audit",
        "",
        f"Generated by `fr_overlay_vs_plf_audit.py`. Walk step={STEP}, origins from 2012 "
        f"(same frame as live BUM `backtest`), horizons {list(HORIZONS)}. "
        f"PRE = window end ≤ {PRE_END:%Y-%m}; OOS = origins ≥ {OOS_START:%Y-%m}; mid = the rest. "
        f"Overlay candidate = seas_mult_3 with primary×(1+{EPS_PRI_HICP}·HICP lag{LAG}) and "
        f"recettes×(1+{EPS_REC_HICP}·HICP lag{LAG} + {EPS_REC_EQ:.3f}·eq_DE lag{LAG}). "
        "Does not change SHIP.",
        "",
        "## Verdict",
        "",
        "The bare-overlay vs live-BUM comparison is **unfair on OOS, fair on PRE**. "
        "Soft-pin α=1 is not the stack — the PLF **category** annual pin is. Even after "
        "equalising soft-pin and after switching to BUM whenever PLF categories are known, "
        f"live BUM α=1 still wins post-2020 OOS ({fmt(bum1_oos)}). Nothing ships.",
        "",
        "## Is “overlay loses to PLF” apples-to-apples?",
        "",
        f"**PRE — yes, fair on pins.** Soft-pin α=1 is inert (BUM α=0 = α=1 = {fmt(bum1_pre)}). "
        f"The PRE gap (overlay {fmt(ov_pre)} vs BUM {fmt(bum1_pre)}, Δ≈{fmt(pre_gap)}) is a "
        "different annual constructor (hold+shares vs scaled seasonals), not a stacked pin.",
        "",
        f"**OOS — no, stacked.** Category PLF pin hits "
        f"{c_oos.get('frac_any_plf', 0):.0%} of OOS h=12 origins "
        f"(mean {c_oos.get('mean_plf_months', 0):.1f}/12 months); soft pin only "
        f"{c_oos.get('frac_any_soft', 0):.0%} (mean {c_oos.get('mean_soft_months', 0):.1f}). "
        f"BUM α=0 already {fmt(bum0_oos)} vs seas {fmt(seas_oos)}; α=1 only to {fmt(bum1_oos)}. "
        f"Overlay soft-pinned: {fmt(ov_oos)} → {fmt(ovpin_oos)} "
        f"(bare−BUM ≈{fmt(oos_gap)}; pinned−BUM ≈{fmt(oos_gap_pin)}). "
        "The edge is the category table, not the residual solde soft pin.",
        "",
        "**STATUS PRE win vs seas was sample-sensitive.** step=3 can clear seas; this dense "
        f"BUM-matched walk has overlay PRE {fmt(ov_pre)} vs seas {fmt(seas_pre)}.",
        "",
        "## Mean MAE (h=1/3/6/12 average)",
        "",
        "| Spec | PRE | mid | OOS |",
        "|---|---:|---:|---:|",
    ]
    for _, lab in COLS:
        r = table[lab]
        oos_s = f"**{fmt(r['OOS'])}**" if lab == winner else fmt(r["OOS"])
        lines.append(f"| {lab} | {fmt(r['PRE'])} | {fmt(r['mid'])} | {oos_s} |")
    lines += [
        "",
        "Mid (mostly 2020–21): overlay/seas beat BUM — hold annuals miss the Covid level break.",
        "",
        "## OOS by horizon",
        "",
        "| Spec | h1 | h3 | h6 | h12 |",
        "|---|---:|---:|---:|---:|",
    ]
    sel = ["seas", "bum_a1", "bum_a0", "overlay", "hy_c_ov_pin", "hy_a_shape", "hy_d_switch"]
    labs = dict(COLS)
    for col in sel:
        byh = mae_by_h(bt, "OOS", col)
        lines.append(
            f"| {labs[col]} | {fmt(byh.get(1))} | {fmt(byh.get(3))} | "
            f"{fmt(byh.get(6))} | {fmt(byh.get(12))} |"
        )
    lines += [
        "",
        "## Pin coverage (h=12 origins)",
        "",
        "| Window | n | frac any PLF cat | mean PLF months | frac any soft | mean soft months |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for w in windows:
        c = cov.get(w, {})
        if not c:
            continue
        lines.append(
            f"| {w} | {c['n']} | {c['frac_any_plf']:.0%} | {c['mean_plf_months']:.1f} | "
            f"{c['frac_any_soft']:.0%} | {c['mean_soft_months']:.1f} |"
        )
    lines += [
        "",
        "## Spec definitions",
        "",
        "- **seas_mult_3** — production seasonal on the SMB deficit.",
        "- **BUM α=1 / α=0** — live `bottom_up_macro` (PLF categories when known, else hold + "
        "remboursements_locaux grow; 5y shares; interest smly). α = soft pin to `state_deficit`.",
        f"- **overlay alone** — candidate seasonal × macro scales; no PLF.",
        "- **(a)** BUM year totals, months reshaped to overlay within-year shares, then soft-pin α=1.",
        "- **(b)** BUM α=1 + α·(overlay_softpinned − BUM). α=1 = (c).",
        "- **(c)** overlay + soft-pin α=1 to the same PLF solde — fairness check on the residual pin.",
        "- **(d)** per calendar year: BUM months if PLF categories known at origin, else overlay; "
        "then soft-pin α=1.",
        "",
        "## Post-2020 winner and ship",
        "",
        f"Fair OOS winner: **{winner}** at {fmt(winner_mae)}. "
        + (
            "Keep `SHIP=True` on live BUM. Overlay stays research: it can beat bare seas on "
            "mid/OOS but loses the fair PLF-aware contest. Do not flip SHIP."
            if winner.startswith("BUM α=1")
            else "Do not flip SHIP without a separate PRE ≥5% gate vs live BUM and a holdout check."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("walking…")
    bt = walk()
    if bt.empty:
        raise SystemExit("empty backtest")
    print("origins h12:", bt[bt["h"] == 12].groupby("window").size().to_dict())
    for _, lab in COLS:
        pass
    for col, lab in COLS:
        print(f"{lab:40s}  PRE={mean_mae(bt, 'PRE', col)}  "
              f"mid={mean_mae(bt, 'mid', col)}  OOS={mean_mae(bt, 'OOS', col)}")
    md = render_md(bt)
    OUT.write_text(md)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
