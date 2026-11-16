"""Shared helpers.

dk_ols:             Driscoll-Kraay pooled OLS
parse_coupon:       coupon from a gilt name
load_rpi_all_items: ONS RPI all items (CDKO), read from _vendor_data/rpi_all_items.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

H = 12
VULGAR = {"¼": 0.25, "½": 0.5, "¾": 0.75, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
          "⅓": 1 / 3, "⅔": 2 / 3, "⅙": 1 / 6, "⅚": 5 / 6}




def dk_ols(z: pd.DataFrame, y: str, X: list[str], lags: int = H) -> dict:
    """Pooled OLS (no constant: variables are demeaned by month) with Driscoll-Kraay errors: moment
    contributions summed within each month, then Newey-West across months (robust to cross-country
    correlation and to the overlap of 12-month changes)."""
    z = z.dropna(subset=[y, *X]).sort_values("month")
    XX, yy = z[X].to_numpy(), z[y].to_numpy()
    b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    e = yy - XX @ b
    h = pd.DataFrame(XX * e[:, None]).groupby(z["month"].to_numpy()).sum().to_numpy()
    S = h.T @ h
    for L in range(1, lags + 1):
        G = h[L:].T @ h[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(XX.T @ XX)
    se = np.sqrt(np.diag(Q @ S @ Q))
    return {"beta": dict(zip(X, b)), "t": dict(zip(X, b / se)), "n": int(len(z)),
            "r2": float(1 - e.var() / yy.var())}


def parse_coupon(name: str) -> float | None:
    """Pull the coupon out of a gilt name: `0 3/8% Treasury Gilt 2026` is 0.375, `3¾%` is 3.75."""
    if not isinstance(name, str):
        return None
    head = name.split("%")[0].strip()
    if not head:
        return None
    total = 0.0
    seen = False
    for symbol, value in VULGAR.items():
        if symbol in head:
            total += value
            head = head.replace(symbol, " ")
            seen = True
    fraction = re.search(r"(\d+)\s*/\s*(\d+)", head)
    if fraction:
        total += int(fraction.group(1)) / int(fraction.group(2))
        head = head[:fraction.start()] + " " + head[fraction.end():]
        seen = True
    whole = re.search(r"(\d+(?:\.\d+)?)", head)
    if whole:
        total += float(whole.group(1))
        seen = True
    return round(total, 6) if seen else None


def load_rpi_all_items() -> pd.Series:
    s = pd.read_csv(Path(__file__).resolve().parent / "_vendor_data" / "rpi_all_items.csv", index_col=0,
                    parse_dates=True)["rpi"]
    s.index.name = None
    return s.astype(float)
