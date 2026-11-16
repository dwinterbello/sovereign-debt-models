"""Norway cash framework: fetch (optional) → contract outputs → dashboard.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.no.run            # rebuild from data/no/
  PYTHONPATH=. .venv/bin/python -m sovereign_cash.no.run --refresh  # re-fetch Norges Bank SDMX first
"""

from __future__ import annotations

import argparse
import json

from sovereign_cash.no import build, dashboard, fetch


def run(refresh: bool = False) -> dict:
    if refresh or not (fetch.RAW / "liquidity_statistics.parquet").exists():
        print("fetch:", fetch.fetch_all())
    df, summary = build.build()
    out = dashboard.build()
    print(f"Wrote {build.OUT_MONTHLY}, {build.OUT_SUMMARY}, {build.OUT_DAILY}, {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    s = run(refresh=ap.parse_args().refresh)
    print(json.dumps(s["forward"], indent=2, ensure_ascii=False))
