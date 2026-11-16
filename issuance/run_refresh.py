#!/usr/bin/env python3
"""Rebuild sovereign issuance, official demand, USD DV01, and the dashboard.

    PYTHONPATH=. .venv/bin/python -m issuance.run_refresh
    PYTHONPATH=. .venv/bin/python -m issuance.run_refresh --only uk de us ecb
    PYTHONPATH=. .venv/bin/python -m issuance.run_refresh --force --skip-dashboard
"""

from __future__ import annotations

import argparse
import json
import traceback

from issuance.paths import CACHE, DERIVED


STAGES = (
    "uk", "de", "fr", "it", "es", "us", "jp", "se",
    "boj", "apf", "soma", "ecb", "fx",
    "flows", "reconcile", "official", "dv01", "rv", "holders", "lines", "float", "imbalance", "positioning",
    "market", "literature", "evidence",
)


def _run(name: str, fn, *args, **kwargs):
    print(f"\n=== {name} ===")
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        print(f"FAILED {name}: {exc}")
        traceback.print_exc()
        return None
    if hasattr(result, "shape"):
        print(f"  {name}: {result.shape}")
    elif isinstance(result, dict):
        print(f"  {name}: {list(result)[:8]}")
    else:
        print(f"  {name}: done")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of stages")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-dashboard", action="store_true")
    args = ap.parse_args()
    wanted = set(args.only) if args.only else set(STAGES)
    if args.skip_dashboard:
        wanted.discard("dashboard")

    ops_parts = []
    if "uk" in wanted:
        from issuance.fetch_uk import fetch_uk
        ops_parts.append(_run("uk", fetch_uk, force=args.force))
    if "de" in wanted:
        from issuance.fetch_de import fetch_de
        ops_parts.append(_run("de", fetch_de, force=args.force))
    if "fr" in wanted:
        from issuance.fetch_fr import fetch_fr
        ops_parts.append(_run("fr", fetch_fr, force=args.force))
    if "it" in wanted:
        from issuance.fetch_it import fetch_it
        ops_parts.append(_run("it", fetch_it, force=args.force))
    if "es" in wanted:
        from issuance.fetch_es import fetch_es
        ops_parts.append(_run("es", fetch_es, force=args.force))
    if "us" in wanted:
        from issuance.fetch_us import fetch_us
        ops_parts.append(_run("us", fetch_us, force=args.force))
    if "jp" in wanted:
        from issuance.fetch_jp import fetch_jp
        from issuance.jp_calendar import fetch_calendar, fetch_lea_amounts
        ops_parts.append(_run("jp", fetch_jp, force=args.force))
        _run("jp_calendar", fetch_calendar)
        _run("jp_lea_amounts", fetch_lea_amounts)
    if "se" in wanted:
        from issuance.fetch_se import fetch_se
        ops_parts.append(_run("se", fetch_se, force=args.force))
    if "boj" in wanted:
        from issuance.fetch_boj import fetch_boj
        _run("boj", fetch_boj, force=args.force)
    if "apf" in wanted:
        from issuance.fetch_apf import fetch_apf
        _run("apf", fetch_apf, force=args.force)
    if "soma" in wanted:
        from issuance.fetch_soma import fetch_soma
        _run("soma", fetch_soma, force=args.force)
    if "ecb" in wanted:
        from issuance.fetch_ecb_qe import fetch_ecb_qe
        _run("ecb", fetch_ecb_qe, force=args.force)
    if "fx" in wanted:
        from issuance.fetch_fx import fetch_fx
        _run("fx", fetch_fx, force=args.force)

    from issuance.isin_flow import load_ops, write_flows
    ops = load_ops()
    print(f"\nops loaded: {len(ops):,} rows, countries={sorted(ops['country'].dropna().unique())}")

    if "flows" in wanted:
        from issuance.isin_flow import write_flows as _wf
        isin, tenor = _run("flows", _wf, ops) or (None, None)
    else:
        from issuance.isin_flow import by_isin, by_tenor
        isin, tenor = by_isin(ops), by_tenor(ops)

    if "reconcile" in wanted:
        from issuance.reconcile import write_reconcile
        report = _run("reconcile", write_reconcile, ops)
        if report:
            print(json.dumps(report.get("uk_stock", {}), indent=2, default=str)[:1200])

    if "official" in wanted:
        from issuance.public_demand import official_flow, net_private
        official = _run("official", official_flow, ops)
        if tenor is not None:
            _run("net_private", net_private, tenor, official)

    if "dv01" in wanted and isin is not None:
        from issuance.dv01 import write_dv01
        _run("dv01", write_dv01, isin, tenor)

    if "rv" in wanted:
        import pandas as pd
        from issuance.rv_metrics import write_rv
        tenor_dv = pd.read_parquet(CACHE / "dv01_tenor.parquet") if (CACHE / "dv01_tenor.parquet").exists() else None
        off_rv = pd.read_parquet(CACHE / "official_flow.parquet") if (CACHE / "official_flow.parquet").exists() else None
        if tenor_dv is not None:
            _run("rv", write_rv, ops, tenor_dv, off_rv)

    if "lines" in wanted:
        from issuance.lines import write_lines
        _run("lines", write_lines, ops)
        from issuance.forward_audit import write_audit
        _run("audit", write_audit, ops)
    if "float" in wanted:
        from issuance.float_rv import write_float
        _run("float", write_float, ops)
    if "imbalance" in wanted:
        from issuance.imbalance import write_imbalance
        _run("imbalance", write_imbalance, ops)
        from issuance.scenarios import write_scenarios
        _run("scenarios", write_scenarios)
    if "holders" in wanted:
        from issuance.fetch_holders import fetch_holders
        _run("holders", fetch_holders)
    if "positioning" in wanted:
        from issuance.fetch_jp_flows import fetch_jp_flows
        from issuance.fetch_us_bidders import fetch_us_bidders
        _run("jp_flows", fetch_jp_flows)
        _run("us_bidders", fetch_us_bidders)
        from issuance.fetch_jpx import fetch_jpx
        from issuance.fetch_xborder import fetch_xborder
        _run("jpx", fetch_jpx)
        _run("xborder", fetch_xborder, force=args.force)
        from issuance.positioning import write_positioning
        _run("positioning", write_positioning)
        from issuance.flow_score import write as write_flow_score
        from issuance.habitat import write as write_habitat
        _run("flow_score", write_flow_score)
        _run("habitat", write_habitat)
        from issuance.experiments.buyer_tape import write as write_buyer_tape
        _run("buyer_tape", write_buyer_tape)
    if "market" in wanted:
        from issuance.market import write as write_market
        _run("market", write_market)
    if "literature" in wanted:
        from issuance import ecb_holders, fetch_egb, fetch_jsda, fetch_us_demand, imf_investor_base, uk_holders
        _run("imf_investor_base", imf_investor_base.fetch, refresh=True)
        _run("ecb_holders", ecb_holders.fetch, refresh=True)
        _run("us_demand", fetch_us_demand.fetch, refresh=True)
        _run("us_demand_cache", fetch_us_demand.write)
        _run("jsda", fetch_jsda.fetch, refresh=True)
        _run("uk_holders", uk_holders.fetch, refresh=True)
        _run("egb", fetch_egb.fetch, refresh=True)
    if "evidence" in wanted:
        from issuance.evidence import write as write_evidence
        _run("evidence", write_evidence)

    print(f"\ncache: {CACHE}")
    print(f"derived: {DERIVED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
