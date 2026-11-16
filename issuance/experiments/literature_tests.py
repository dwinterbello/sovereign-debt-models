"""The three literature corrections to the evidence table, printed compactly.

  1. stock, not flow   dur_stock_gdp (Greenwood & Vayanos 2014; D'Amico & King 2013)
  2. excess returns    rx10_3 / rx10_12 / tp10 as targets
  3. state dependence  supply × realised vol (Vayanos–Vila 2021; GHSS 2023) and supply × private
                       share of the coupon stock (Eren, Schrimpf & Xia, BIS WP 1105)

Reads issuance/cache/evidence*.parquet (written by `evidence.write()`); `--recompute` rebuilds both.

    PYTHONPATH=. .venv/bin/python -m issuance.experiments.literature_tests [--recompute]
"""

from __future__ import annotations

import sys

import pandas as pd

from issuance import evidence as E

FLOW = ("priv12", "gross12", "absorb", "absorb_long", "cb_share")
STOCK = E.NEW_METRICS
PRICE_COLS = ["metric", "target", "horizon", "beta_bp", "t", "loo_min_t", "p_chance", "oos_r2", "n_countries", "verdict"]
INTER_COLS = ["metric", "target", "horizon", "beta1_bp", "beta2_bp", "t2", "loo_min_t", "p_chance", "oos_r2",
              "beta_hi_bp", "t_hi", "beta_lo_bp", "t_lo", "n_countries", "verdict"]


def _show(title: str, d: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n== {title} ({len(d)} rows)")
    if d.empty:
        print("  none")
        return
    print(d[cols].round({c: 2 for c in cols if c not in ("oos_r2",)} | {"oos_r2": 3}).to_string(index=False))


def main(recompute: bool = False) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    if recompute:
        E.write()
    ev, inter = E.load(), E.load_interactions()
    if ev.empty:
        print("no evidence table; run with --recompute")
        return
    w = ev[ev["window"] == "2010+"]
    order = {t: i for i, t in enumerate(E.TARGETS)}
    srt = lambda d: d.assign(_o=d["target"].map(order)).sort_values(["metric", "_o", "horizon"]).drop(columns="_o")  # noqa: E731
    _show("stock × excess return / term premium, 2010+", srt(w[w["metric"].isin(STOCK) & w["target"].isin(E.NEW_TARGETS)]), PRICE_COLS)
    _show("stock × yield targets, 2010+", srt(w[w["metric"].isin(STOCK) & ~w["target"].isin(E.NEW_TARGETS)]), PRICE_COLS)
    _show("flow × excess return / term premium, 2010+", srt(w[w["metric"].isin(FLOW) & w["target"].isin(E.NEW_TARGETS)]), PRICE_COLS)
    _show("other metrics × excess return / term premium, 2010+",
          srt(w[~w["metric"].isin(FLOW + STOCK) & w["target"].isin(E.NEW_TARGETS)]), PRICE_COLS)
    for half in ("2010–18", "2019+"):
        h = ev[(ev["window"] == half) & (ev["metric"].isin(STOCK) | ev["target"].isin(E.NEW_TARGETS))]
        h = h[h["verdict"].isin(["forward link", "wrong sign"])]
        _show(f"new-test forward links / wrong signs, {half}", srt(h), PRICE_COLS)
    if not inter.empty:
        for state in E.STATES:
            d = srt(inter[inter["state"] == state])
            _show(f"supply × {state} ({E.STATES[state]}): passing or wrong-signed β2",
                  d[d["verdict"].isin(["forward link", "wrong sign"])], INTER_COLS)
            _show(f"supply × {state}: stock and flow metrics, every target", d[d["metric"].isin(FLOW + STOCK)], INTER_COLS)
    print("\n== multiple testing (2010+ window; expected = n × P(t ≥ T_BAR) under the null, an upper bound)")
    budget = {"all evidence rows": E.chance(ev), "new evidence rows": E.chance(E.new_rows(ev))}
    if not inter.empty:
        budget |= {f"interactions: {s}": E.chance(inter[inter["state"] == s]) for s in E.STATES}
        budget["interactions: all"] = E.chance(inter)
    print(pd.DataFrame(budget).T.round(2).to_string())


if __name__ == "__main__":
    main("--recompute" in sys.argv)
