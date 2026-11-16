#!/usr/bin/env python3
"""Scrape DTS TGA + debt from Fiscal Data, build daily implied-deficit / GDP plot,
then the funding-liquidity stack (DTS flow model → TGA / reserves path → reserve
curve → SERFF → repo structure → liquidity_dashboard.html).

  python us/fiscal/run_refresh.py
  python us/fiscal/run_refresh.py --force
  python us/fiscal/run_refresh.py --backtests      # also re-score flow model / TGA path (slow)
  python us/fiscal/run_refresh.py --skip-liquidity # deficit dashboard only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_liquidity(*, refresh: bool, backtests: bool) -> None:
    import auction_ladder
    import dark_spot
    import dts_flows
    import financing
    import fiscal_path
    import flow_model
    import market_implied
    import money_markets
    import reserve_demand
    import serff
    import supply_bridge
    import term_premium
    import tga_forecast

    dts_flows.run(refresh=refresh)
    auction_ladder.run(refresh=refresh)
    financing.run(refresh=refresh)
    money_markets.run(refresh=refresh)
    if backtests:
        bt = flow_model.backtest()
        s = flow_model.score(bt)
        (flow_model.DERIVED / "flow_model_score.json").write_text(json.dumps(s.to_dict(orient="records"), indent=1))
        flow_model.daily_bucket_score()
    reserve_demand.run()
    if backtests:
        tga_forecast.backtest()
    tga_forecast.forward()
    fiscal_path.run(backtest=backtests)
    import debt_ceiling

    debt_ceiling.run(refresh=refresh)
    supply_bridge.run(verbose=False)
    serff.run()
    try:
        market_implied.run()
    except Exception as exc:  # noqa: BLE001  (Yahoo delayed quotes; keep the prior snapshot)
        print(f"   market-implied SERFF skipped: {exc}")
    try:
        term_premium.run(refresh=refresh, verbose=False)
    except Exception as exc:  # noqa: BLE001  (NY Fed ACM file; keep the prior result)
        print(f"   ACM term premium skipped: {exc}")
    dark_spot.run()
    from build_components_dashboard import build as build_components
    from build_ceiling_dashboard import build as build_ceiling

    build_components()
    build_ceiling()


def _dashboard_only() -> None:
    """Rebuild deficit / liquidity / components / ceiling / hub HTML from warm derived."""
    from build_ceiling_dashboard import build as build_ceiling
    from build_components_dashboard import build as build_components
    from build_dashboard import build
    build_liquidity = None
    from build_us_hub import build as build_hub

    build()
    try:
        build_liquidity()
    except Exception as exc:  # noqa: BLE001
        print(f"   liquidity dashboard skipped: {exc}")
    build_components()
    build_ceiling()
    build_hub()


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh US TGA-implied deficit and funding-liquidity dashboards")
    ap.add_argument("--force", action="store_true", help="Re-download DTS / FRED")
    ap.add_argument("--skip-dashboard", action="store_true")
    ap.add_argument("--skip-liquidity", action="store_true")
    ap.add_argument("--backtests", action="store_true", help="Re-run flow-model and TGA-path backtests")
    ap.add_argument(
        "--dashboard-only",
        action="store_true",
        help="site Refresh: rebuild HTML from existing parquet/CSV; no DTS/FRED",
    )
    args = ap.parse_args()

    if args.dashboard_only:
        _dashboard_only()
        return

    from tga_deficit import run
    from ust_intervention import run as run_diagnostic
    from build_dashboard import build
    from build_components_dashboard import build as build_components
    from build_ceiling_dashboard import build as build_ceiling
    from build_us_hub import build as build_hub

    run(refresh=args.force)
    run_diagnostic(refresh=args.force)
    if not args.skip_liquidity:
        run_liquidity(refresh=args.force, backtests=args.backtests)
    if not args.skip_dashboard:
        build()
        if args.skip_liquidity:
            build_components()
            build_ceiling()
    build_hub()


if __name__ == "__main__":
    main()
