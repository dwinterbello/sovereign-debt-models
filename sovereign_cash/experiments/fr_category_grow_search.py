"""PRE-gated search: per-impôt RESF growth + AFT charge triangle refinements.

Among PRE clearers (≥5% edge, ≤1 worse h) pick min OOS. Does not mutate SHIP.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import bottom_up_macro as BUM
from sovereign_cash.yoy_components import components, deficit, SPEC

RECEIPT = list(BUM.RECEIPT_KEYS)
SPEND = list(BUM.SPEND_KEYS)
ALL_GROW = RECEIPT + SPEND


def _score(bt: pd.DataFrame) -> dict:
    d = BUM.ship_decision(bt)
    return {
        "ship": d["ship"],
        "pre": d["pre_bu"],
        "prod_pre": d["pre_prod"],
        "rel": d.get("rel"),
        "worse_h": d.get("worse_h"),
        "oos": d["oos_bu"],
        "prod_oos": d["oos_prod"],
        "reason": d["reason"],
    }


def _patch_grow(keys: frozenset[str], lag_mode: str = "doc"):
    """Grow only ``keys`` with RESF prior; lag_mode=doc|same|lag1."""
    elast = BUM._elast_table()

    def grow(base, origin, year, hist, drivers):
        out = dict(base)
        past = [y for y in hist.get("ir", pd.Series(dtype=float)).index if y < origin.year]
        if not past:
            return out
        base_yr = max(past)
        for y in range(base_yr + 1, year + 1):
            for key in keys:
                spec = elast.get(key, {})
                dname = spec.get("driver")
                prior = spec.get("epsilon")
                if dname is None or prior is None:
                    continue
                lag_doc = int(spec.get("lag") or 0)
                if lag_mode == "same":
                    lag = 0
                    use_y = y  # same-year RESF
                elif lag_mode == "lag1":
                    lag = 1
                    use_y = y - 1
                else:  # doc: apply prior with documented lag on yr-1 driver for lag=0 (shipped lag-1 apply)
                    # Match current module when GROW_WITH_PRIOR: use_y = (y-1) - lag
                    lag = lag_doc
                    use_y = (y - 1) - lag
                if dname not in drivers or use_y not in drivers[dname].index:
                    continue
                eps = float(prior)
                g = eps * float(drivers[dname].loc[use_y]) / 100.0
                out[key] = float(out[key]) * (1.0 + g)
            charge = BUM._charge_table()
            if y in charge and origin >= pd.Timestamp(y - 1, BUM.CHARGE_KNOWN_MONTH, 1):
                out["charge"] = charge[y]
        return out

    return grow


def _patch_grow_same_year(keys: frozenset[str]):
    """Same-year RESF prior on selected keys only (structural vs aggregate)."""
    return _patch_grow(keys, lag_mode="same")


def _triangle_charge(base_charge: float, origin: pd.Timestamp, year: int,
                     drivers: dict[str, pd.Series], scale: float) -> float:
    """AFT +100bp triangle scaled by Δrates (pp) from last complete year."""
    rates = drivers.get("rates")
    if rates is None or rates.empty:
        return base_charge
    y0 = int(origin.year) - 1
    if y0 not in rates.index or year not in rates.index:
        # fall back: use last known vs year-1
        known = [y for y in rates.index if y <= year]
        if len(known) < 2 or y0 not in rates.index:
            return base_charge
    r0 = float(rates.loc[y0])
    ry = float(rates.loc[year]) if year in rates.index else float(rates.dropna().iloc[-1])
    d_pp = ry - r0  # percentage points
    # triangle year-h from open year: h = year - (origin.year) approx; use year - y0
    h = max(1, year - y0)
    tri = BUM._auth().get("rate_shock", {}).get("triangle", {})
    pts = sorted((int(k), float(v)) for k, v in tri.items())
    if not pts:
        return base_charge

    def at_h(hh: float) -> float:
        if hh <= pts[0][0]:
            return pts[0][1] * (hh / pts[0][0]) if pts[0][0] else 0.0
        if hh >= pts[-1][0]:
            return pts[-1][1]
        for (h0, v0), (h1, v1) in zip(pts, pts[1:]):
            if h0 <= hh <= h1:
                w = (hh - h0) / (h1 - h0)
                return v0 + w * (v1 - v0)
        return pts[-1][1]

    # +100bp → triangle; scale by d_pp / 1.0
    delta = at_h(h) * (d_pp / 1.0) * scale
    return base_charge + delta


def run_baseline(frame) -> dict:
    bt = BUM.backtest(frame, step=1)
    return {"name": "SHIP hold", **_score(bt)}


def run_variant(frame, name: str, grow_fn=None, charge_mode: str = "hold",
                charge_scale: float = 1.0, int_w: float | None = None) -> dict:
    orig_grow = BUM._grow_categories
    orig_forecast = BUM.forecast

    if grow_fn is not None:
        BUM._grow_categories = grow_fn
        prev_keys = BUM.GROW_KEYS
        BUM.GROW_KEYS = frozenset(ALL_GROW)  # type: ignore — grow_fn itself filters

    def forecast_wrap(origin, h=12, frame=None, close_to_plan=None,
                      soft_pin_alpha=None, share_k=None, int_smly_w=None):
        # replicate forecast with optional charge triangle on annuals
        iw = BUM.INT_SMLY_W if int_w is None else int_w
        if int_smly_w is not None:
            iw = int_smly_w
        fc = orig_forecast(origin, h, frame=frame, close_to_plan=close_to_plan,
                           soft_pin_alpha=soft_pin_alpha, share_k=share_k,
                           int_smly_w=iw)
        if fc is None or charge_mode == "hold":
            return fc
        # rebuild interest annual from triangle if charge not AFT-pinned
        cats = dict(fc.attrs.get("categories", {}))
        drivers = BUM._driver_panel(origin)
        charge_tbl = BUM._charge_table()
        changed = False
        for ys, row in list(cats.items()):
            y = int(ys)
            aft = (y in charge_tbl and
                   origin >= pd.Timestamp(y - 1, BUM.CHARGE_KNOWN_MONTH, 1))
            if aft:
                continue
            if charge_mode == "triangle":
                row = dict(row)
                row["charge"] = _triangle_charge(
                    float(row["charge"]), origin, y, drivers, charge_scale)
                cats[ys] = row
                changed = True
            elif charge_mode == "triangle_partial":
                # only apply a fraction of triangle overlay
                row = dict(row)
                base = float(row["charge"])
                tri = _triangle_charge(base, origin, y, drivers, charge_scale)
                row["charge"] = base + charge_scale * (tri - base)
                cats[ys] = row
                changed = True
        if not changed:
            return fc
        # remap months
        months = fc.index
        sh_i = BUM._month_share(frame["interest"].dropna().loc[:origin], origin)
        if sh_i is None:
            return fc
        new_i = []
        for m in months:
            ann = cats[str(int(m.year))]["charge"]
            new_i.append(ann * float(sh_i.loc[m.month]))
        fc = fc.copy()
        fc["interest"] = BUM._interest_month_carve(
            pd.Series(new_i, index=months),
            frame["interest"].dropna().loc[:origin],
            iw)
        fc["cash_requirement"] = deficit("fr", {n: fc[n] for n, _ in SPEC["fr"]})
        y_hist = deficit("fr", {n: frame[n] for n, _ in SPEC["fr"]}).dropna()
        alpha = BUM.SOFT_PIN_ALPHA if soft_pin_alpha is None else soft_pin_alpha
        if close_to_plan is False:
            alpha = 0.0
        fc["cash_requirement"] = BUM._soft_pin(fc["cash_requirement"], y_hist, origin, alpha)
        fc.attrs["categories"] = cats
        return fc

    BUM.forecast = forecast_wrap
    try:
        bt = BUM.backtest(frame, step=1)
        return {"name": name, **_score(bt)}
    finally:
        BUM._grow_categories = orig_grow
        BUM.forecast = orig_forecast
        if grow_fn is not None:
            BUM.GROW_KEYS = prev_keys  # type: ignore


def main():
    frame = components("fr")
    rows = [run_baseline(frame)]
    print("baseline", rows[0], flush=True)

    # 1) single-category same-year RESF prior
    for key in ALL_GROW:
        r = run_variant(frame, f"same_yr:{key}",
                        grow_fn=_patch_grow_same_year(frozenset([key])))
        rows.append(r)
        print(r["name"], f"PRE={r['pre']:.3f} worse={r['worse_h']} ship={r['ship']} OOS={r['oos']}", flush=True)

    # 2) single-category lag-1 / doc apply
    for key in ("ir", "tva", "is", "credits_ministeriels", "cas_pensions"):
        for mode in ("doc", "lag1"):
            r = run_variant(frame, f"{mode}:{key}",
                            grow_fn=_patch_grow(frozenset([key]), lag_mode=mode))
            rows.append(r)
            print(r["name"], f"PRE={r['pre']:.3f} worse={r['worse_h']} ship={r['ship']} OOS={r['oos']}", flush=True)

    # 3) pairs of best singles that cleared or nearly cleared
    tax_pairs = list(itertools.combinations(["ir", "tva", "is", "accises", "autres_fiscales"], 2))
    for a, b in tax_pairs:
        r = run_variant(frame, f"same_yr:{a}+{b}",
                        grow_fn=_patch_grow_same_year(frozenset([a, b])))
        rows.append(r)
        print(r["name"], f"PRE={r['pre']:.3f} worse={r['worse_h']} ship={r['ship']} OOS={r['oos']}", flush=True)

    # 4) charge triangle on hold growth
    for scale in (0.25, 0.5, 1.0):
        for mode in ("triangle",):
            r = run_variant(frame, f"charge_{mode}_s{scale}",
                            grow_fn=None, charge_mode=mode, charge_scale=scale)
            rows.append(r)
            print(r["name"], f"PRE={r['pre']:.3f} worse={r['worse_h']} ship={r['ship']} OOS={r['oos']}", flush=True)

    # 5) int_smly weights with hold
    for iw in (0.0, 0.25, 0.75, 1.0):
        r = run_variant(frame, f"int_w={iw}", grow_fn=None, charge_mode="hold", int_w=iw)
        rows.append(r)
        print(r["name"], f"PRE={r['pre']:.3f} worse={r['worse_h']} ship={r['ship']} OOS={r['oos']}", flush=True)

    df = pd.DataFrame(rows)
    clear = df[df["ship"] == True].copy()  # noqa: E712
    print("\n=== PRE clearers ===")
    if clear.empty:
        print("NONE")
    else:
        clear = clear.sort_values("oos")
        print(clear[["name", "pre", "rel", "worse_h", "oos"]].to_string(index=False))
        print("best OOS:", clear.iloc[0].to_dict())
    print("\n=== near misses (rel>0, worse<=2) ===")
    near = df[(df["rel"].fillna(-1) > 0) & (df["worse_h"].fillna(9) <= 2)].sort_values("oos")
    print(near[["name", "pre", "rel", "worse_h", "oos", "ship"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
