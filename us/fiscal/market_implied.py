"""Futures-implied SERFF and SOFR − IORB against the model's monthly forward.

SR1 (1m SOFR, CME) and ZQ (30d fed funds, CBOT) both settle on the calendar-day
average of their rate over the contract month, so for the same month

    SERFF_mkt   = (100 − SR1) − (100 − ZQ)               policy path cancels
    SOFR−IORB   = SERFF_mkt + (EFFR − IORB)              EFFR − IORB held as in serff.py

Quotes are Yahoo delayed settles; SR1 back months trade thinly, so a stale or
missing leg drops that month. Each run appends a snapshot to
`market_serff_history.parquet` so the market can be scored against realised
SERFF once months settle.

  PYTHONPATH=. .venv/bin/python us/fiscal/market_implied.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DERIVED = HERE / "derived"
MONTH_CODES = "FGHJKMNQUVXZ"
N_MONTHS = 6


def contract(root: str, period: pd.Period) -> str:
    exch = "CBT" if root == "ZQ" else "CME"
    return f"{root}{MONTH_CODES[period.month - 1]}{period.year % 100:02d}.{exch}"


def fetch_quotes(months: list[pd.Period]) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for p in months:
        row = {"month": str(p)}
        for root in ("SR1", "ZQ"):
            t = contract(root, p)
            try:
                h = yf.Ticker(t).history(period="10d")
            except Exception:  # noqa: BLE001
                h = pd.DataFrame()
            if len(h):
                row[f"{root.lower()}_px"] = float(h["Close"].iloc[-1])
                row[f"{root.lower()}_date"] = str(h.index[-1].date())
        rows.append(row)
    return pd.DataFrame(rows)


def implied(q: pd.DataFrame, effr_iorb_bp: float) -> pd.DataFrame:
    out = q.copy()
    for c in ("sr1_px", "zq_px", "sr1_date", "zq_date"):
        if c not in out:
            out[c] = pd.NA
    out["sofr_mkt"] = 100.0 - out["sr1_px"].astype(float)
    out["effr_mkt"] = 100.0 - out["zq_px"].astype(float)
    out["serff_mkt_bp"] = 100.0 * (out["sofr_mkt"] - out["effr_mkt"])
    out["sofr_iorb_mkt_bp"] = out["serff_mkt_bp"] + effr_iorb_bp
    return out


def run(verbose: bool = True) -> dict:
    sj = json.loads((DERIVED / "serff.json").read_text())
    fwd = sj["forward"]
    origin = pd.Timestamp(fwd["origin"])
    months = [origin.to_period("M") + k for k in range(1, N_MONTHS + 1)]
    q = implied(fetch_quotes(months), fwd["effr_iorb_hold_bp"])
    model = fwd["months"]
    for v in ("rw", "rw_cal", "curve_hold"):
        q[f"serff_{v}_bp"] = q["month"].map(lambda m, v=v: model.get(m, {}).get(v, {}).get("serff_bp"))
    q["mkt_minus_choice_bp"] = q["serff_mkt_bp"] - q[f"serff_{sj['choice']}_bp"].astype(float)
    q["snapshot"] = str(pd.Timestamp.today().normalize().date())

    hist_path = DERIVED / "market_serff_history.parquet"
    hist = pd.read_parquet(hist_path) if hist_path.exists() else pd.DataFrame()
    hist = pd.concat([hist[hist.get("snapshot", pd.Series(dtype=str)) != q["snapshot"].iloc[0]] if len(hist) else hist,
                      q], ignore_index=True)
    hist.to_parquet(hist_path, index=False)

    out = {"snapshot": q["snapshot"].iloc[0], "model_origin": fwd["origin"], "model_choice": sj["choice"],
           "effr_iorb_hold_bp": fwd["effr_iorb_hold_bp"],
           "months": q.drop(columns=["snapshot"]).round(3).astype(object).where(q.notna(), None)
           .to_dict(orient="records")}
    (DERIVED / "market_serff.json").write_text(json.dumps(out, indent=1, default=str))
    if verbose:
        cols = ["month", "sofr_mkt", "effr_mkt", "serff_mkt_bp", "sofr_iorb_mkt_bp", "serff_rw_bp",
                "serff_curve_hold_bp", "mkt_minus_choice_bp"]
        print(q[cols].round(2).to_string(index=False))
    return out


if __name__ == "__main__":
    run()
