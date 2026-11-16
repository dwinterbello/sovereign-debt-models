"""Output contract shared by every country cash framework under ``sovereign_cash/<cc>/``.

Each country module writes, under ``sovereign_cash/derived/``:

``<cc>_cash_monthly.parquet``
    One row per calendar month (``month`` = month-start Timestamp), history and
    forecast in the same frame, local currency **bn**. Sign convention: inflows to the
    government cash buffer are positive.

``<cc>_cash_summary.json``
    ``country``, ``ccy``, ``asof`` (last observed month), ``sources`` (list of
    {name, url, frequency, lag_days, verified: bool}), ``buffer`` (what the cash
    buffer is: central-bank account, T-bill stock, ...), ``identity_resid_bn`` (max |monthly
    residual| of the history identity), ``score`` (list of {window, h, model, benchmark,
    n} or [] with a reason in ``score_note``), ``forward`` (headline numbers for the next
    12 months), ``notes`` (list of str).

and a standalone page ``sovereign_cash/<cc>_cash_dashboard.html``.

History identity (where the data allow it):

    Δbuffer = −cash_requirement + net_bonds + net_bills + other_fin

``other_fin`` absorbs what the sources do not split (privatisations, on-lending,
EU loans); it must be reported, never silently dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "derived"

REQUIRED = (
    "month",
    "is_forecast",
    "cash_requirement_bn",  # borrowing need from fiscal flows (deficit +), excl. debt redemption
    "bond_gross_bn",
    "bond_redemption_bn",
    "net_bonds_bn",
    "bill_gross_bn",
    "bill_redemption_bn",
    "net_bills_bn",
    "other_fin_bn",
    "buffer_change_bn",  # implied by the identity on forecast rows; observed where published
)
OPTIONAL = (
    "buffer_level_bn",  # published cash balance / T-bill stock / account, if any
    "buffer_level_implied_bn",
    "bill_stock_bn",
    "bill_stock_implied_bn",
    "cash_requirement_fc_official_bn",  # agency's own forecast, if published
    "cash_requirement_bottomup_bn",  # component-rebuilt cash path (baseline candidate)
)


MINUS = "\u2212"


def fmt_bn(v, sym: str = "", dp: int = 1, *, plus: bool = False, unit: str = "bn") -> str:
    """Money for page text: −£1.5bn, +€2.0bn, SEK −3.1bn (symbol after the sign; 3-letter codes keep a space)."""
    if v is None or pd.isna(v):
        return "—"
    s = MINUS if v < 0 and round(abs(v), dp) > 0 else ("+" if plus and v > 0 else "")
    num = f"{abs(v):,.{dp}f}{unit}"
    if len(sym) >= 3:
        return f"{sym} {s}{num}"
    return f"{s}{sym}{num}"


def fmt_num(v, dp: int = 1, *, plus: bool = False) -> str:
    """Signed number with a true minus sign."""
    return fmt_bn(v, "", dp, plus=plus, unit="")


def validate(df: pd.DataFrame, *, tol_bn: float = 1.0) -> float:
    """Check columns and the identity on history rows; return max |residual| (bn)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not df["month"].is_monotonic_increasing or df["month"].duplicated().any():
        raise ValueError("month must be unique and increasing")
    resid = (df["buffer_change_bn"] - (-df["cash_requirement_bn"] + df["net_bonds_bn"]
                                       + df["net_bills_bn"] + df["other_fin_bn"]))
    worst = float(resid.abs().max()) if len(resid) else 0.0
    if worst > tol_bn:
        raise ValueError(f"identity residual {worst:.2f}bn > {tol_bn}bn; put the gap in other_fin_bn")
    return worst
