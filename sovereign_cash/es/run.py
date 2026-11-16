"""Spain cash: IGAE bottom-up rest-of-year → derived parquet + dashboard.

Thin-board country (no full Tesoro financing identity). Cash need is IGAE B.9
via ``es.bottom_up_macro`` when ``SHIP``; ``thin_cash.build_frame`` adds policy
overlays and the financing stub columns the investor view expects.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.es.run
  PYTHONPATH=. .venv/bin/python -m sovereign_cash.es.run --refresh  # re-parse IGAE xlsx
"""

from __future__ import annotations

import argparse
import json

from sovereign_cash import thin_cash as T
from sovereign_cash.contract import DERIVED
from sovereign_cash.es import bottom_up_macro as BUM
from sovereign_cash.es import dashboard as D

OUT = DERIVED / "es_cash_monthly.parquet"


def run(*, refresh: bool = False, dashboard: bool = True, verbose: bool = True) -> dict:
    if refresh:
        BUM.category_panel(refresh=True)
    frame = T.build_frame("es")
    if frame is None or frame.empty:
        raise RuntimeError("Spain: could not build es_cash_monthly (IGAE / macro cache)")
    DERIVED.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    ev = BUM.evaluate()
    summary = {
        "cc": "es",
        "model": "bottom_up_macro",
        "ship": BUM.SHIP,
        "pre_mean_ratio": ev.get("pre_mean_ratio"),
        "prod_pre_mean_ratio": ev.get("prod_pre_mean_ratio"),
        "n_forecast": int(frame["is_forecast"].sum()),
        "through": str(frame.loc[~frame["is_forecast"].astype(bool), "month"].max().date()),
        "path": str(OUT),
    }
    html = D.build() if dashboard else None
    if verbose:
        msg = f"Wrote {OUT}"
        if html is not None:
            msg += f", {html}"
        print(msg)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-parse IGAE Administración Central xlsx")
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args()
    s = run(refresh=args.refresh, dashboard=not args.no_dashboard)
    print(json.dumps(s, indent=2))