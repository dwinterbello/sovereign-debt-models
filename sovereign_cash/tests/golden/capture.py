"""Freeze the scenario-layer outputs so a refactor can be checked against them.

    PYTHONPATH=. .venv/bin/python sovereign_cash/tests/golden/capture.py [--out DIR]

Writes one JSON per block plus ``inputs.json`` (sha256 of every input file the
blocks read). Compare a later run with ``compare.py``: a mismatch with matching
input hashes is a code change; a mismatch with different hashes is new data.
Each block is isolated, so one failure is recorded and the rest still run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "sovereign_cash"
DEFAULT_OUT = Path(__file__).resolve().parent / "data"
MACRO_CC = ("fr", "de", "it")


def _series(s: pd.Series | None) -> dict | None:
    if s is None:
        return None
    s = s.dropna()
    return {str(pd.Timestamp(k).date()) if not isinstance(k, (int, np.integer)) else str(int(k)): float(v)
            for k, v in s.items()}


def _frame(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    return {c: _series(df[c]) for c in df.columns}


def block_scenarios() -> dict:
    from sovereign_cash import scenarios
    return {cc: scenarios.run(cc) for cc in MACRO_CC}


def block_components() -> dict:
    from sovereign_cash import monthly_components as M
    from sovereign_cash import scenarios
    out = {}
    for cc in MACRO_CC:
        origin = scenarios._origin(cc)
        h = 12 * (scenarios.YEARS_AHEAD + 1) + (12 - origin.month) + 1
        fc = M.forecast(cc, origin, h, use_authority=True)
        out[cc] = {"origin": str(origin.date()), "h": h, "paths": _frame(fc)}
    return out


def block_drivers() -> dict:
    from sovereign_cash import scenarios
    from sovereign_cash.macro_authority import driver_path
    out = {}
    for cc in MACRO_CC:
        origin = scenarios._origin(cc)
        cut = origin - pd.DateOffset(months=24)
        cc_out = {"origin": str(origin.date())}
        for tag, shock in (("base", None), ("ngdp", {"ngdp": 1.0}), ("hicp", {"hicp": 1.0}),
                           ("rates", {"rates": 1.0})):
            d = driver_path(cc, origin, 60, shock=shock)
            cc_out[tag] = {k: _series(v.loc[cut:]) for k, v in d.items() if isinstance(v, pd.Series)}
        out[cc] = cc_out
    return out


def block_authority() -> dict:
    from sovereign_cash.macro_authority import annual_authority
    out = {}
    for cc in MACRO_CC:
        ngdp, cpi, src = annual_authority(cc)
        out[cc] = {"ngdp": _series(ngdp), "cpi": _series(cpi), "src": src}
    return out


def block_rates() -> dict:
    from sovereign_cash.interest_split import annual_effective_rate, annual_interest, rates_shock_delta
    d = pd.read_parquet(SC / "derived" / "debt_repricing.parquet")
    ccs = sorted(set(d["cc"].astype(str)) | set(MACRO_CC))
    out = {}
    for cc in ccs:
        out[cc] = {
            "p100": {str(k): v for k, v in rates_shock_delta(cc, 100.0).items()},
            "p50": {str(k): v for k, v in rates_shock_delta(cc, 50.0).items()},
            "m100": {str(k): v for k, v in rates_shock_delta(cc, -100.0).items()},
        }
        if cc in MACRO_CC:
            out[cc]["annual_interest"] = _series(annual_interest(cc))
            out[cc]["effective_rate"] = _series(annual_effective_rate(cc))
    cases = d.groupby(["cc", "scenario", "case"]).size().reset_index(name="n")
    out["_repricing_cases"] = cases.astype({"n": int}).to_dict("records")
    return out


def block_uk_reckoner() -> dict:
    from sovereign_cash.uk import ready_reckoner as R
    out = {"vintage": R.VINTAGE, "units": {}}
    for det in R.determinants():
        try:
            out["units"][det] = _series(R.psnb(R.run(R.unit(det))))
        except Exception as e:  # keep going; record which determinant failed
            out["units"][det] = {"_error": repr(e)}
    coef = R.coefficients()
    out["coefficients_shape"] = list(coef.shape)
    out["coefficients_sum"] = float(pd.to_numeric(coef.select_dtypes("number").stack(), errors="coerce").sum())
    return out


def block_debt_sensitivities() -> dict:
    from sovereign_cash.debt import sensitivities as S
    return json.loads(json.dumps(S.tables(), default=str))


BLOCKS = {
    "scenarios": block_scenarios,
    "components": block_components,
    "drivers": block_drivers,
    "authority": block_authority,
    "rates": block_rates,
    "uk_reckoner": block_uk_reckoner,
    "debt_sensitivities": block_debt_sensitivities,
}

INPUT_GLOBS = (
    "sovereign_cash/derived/*.parquet",
    "sovereign_cash/derived/*.json",
    "sovereign_cash/data/debt/*.parquet",
    "sovereign_cash/data/*/authority.json",
    "sovereign_cash/data/uk/obr_ready_reckoner.xlsx",
)


def input_hashes() -> dict:
    out = {}
    for pat in INPUT_GLOBS:
        for p in sorted(REPO.glob(pat)):
            out[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "inputs.json").write_text(json.dumps(input_hashes(), indent=1, sort_keys=True))
    status = {}
    for name, fn in BLOCKS.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        try:
            payload = fn()
            (a.out / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
            status[name] = {"ok": True, "secs": round(time.time() - t0, 1)}
        except Exception:
            status[name] = {"ok": False, "secs": round(time.time() - t0, 1), "error": traceback.format_exc()}
        print(name, status[name]["ok"], status[name]["secs"], flush=True)
    (a.out / "status.json").write_text(json.dumps(status, indent=1))
    return 0 if all(s["ok"] for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
