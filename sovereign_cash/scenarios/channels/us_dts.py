"""US federal primary from the DTS bucket path (``us/fiscal``), rebased to FRED FY levels.

``fiscal_path_buckets`` (live, through FY2030) carries COLA on SSA / SSI / VA / military
retirement, a FERS-diet blend on OPM, beneficiary *volume drift* capped at 3% on those
lines, and trailing 250-day growth on the rest — not Census population or CBO long-term
ageing. History is ``dts_bucket_daily`` ($mn); the parquet forecast is already $bn.

Raw DTS cash FY totals do not match FRED ``FYFR`` / ``FYONET`` / ``FYOINT`` (different
perimeter and accounting). The AMECO GG wedge is calibrated on FRED, so forecast years
are FRED origin levels × DTS component growth from that origin FY — individual ``dep_*`` /
``wd_*`` lines, then resummed.

    PYTHONPATH=. .venv/bin/python -c "from sovereign_cash.scenarios.channels import us_dts as U; print(U.fy_table().tail())"
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
US_DERIVED = REPO / "us" / "fiscal" / "derived"
HIST = US_DERIVED / "dts_bucket_daily.parquet"
FC = US_DERIVED / "fiscal_path_buckets.parquet"
MIN_FY_DAYS = 200  # ~ business days in a complete FY


def _fy_index(idx: pd.DatetimeIndex) -> pd.Series:
    t = pd.DatetimeIndex(idx)
    return pd.Series(t.year + (t.month >= 10).astype(int), index=t)


def _deps(cols) -> list[str]:
    return [c for c in cols if c.startswith("dep_") and c != "dep_total"]


def _wd_primary(cols) -> list[str]:
    return [c for c in cols if c.startswith("wd_") and c not in ("wd_total", "wd_interest")]


@lru_cache(maxsize=1)
def daily_panel() -> pd.DataFrame:
    """Combined daily bucket panel in $bn (history $mn → /1000, forecast already $bn)."""
    if not HIST.exists() or not FC.exists():
        return pd.DataFrame()
    h = pd.read_parquet(HIST)
    f = pd.read_parquet(FC)
    cols = [c for c in f.columns if c.startswith(("dep_", "wd_"))]
    h = h.reindex(columns=cols).astype(float) / 1000.0
    f = f.reindex(columns=cols).astype(float)
    # forecast starts the day after history; drop any overlap on the hist side
    if len(f) and len(h):
        h = h[h.index < f.index.min()]
    out = pd.concat([h, f]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def fy_table(adjusted: bool = True) -> pd.DataFrame:
    """Complete fiscal years (Oct–Sep): receipts, primary_spend, interest, plus bucket columns ($bn).

    ``adjusted``: forecast years go through ``us_dts_adjust`` (FY2026 one-offs out, lines on their drivers). The cache
    is keyed on ``us_dts_adjust.ENABLED`` too, so flipping that flag can never return a stale table.
    """
    from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ

    return _fy_table(bool(adjusted and ADJ.ENABLED), ADJ.params())


@lru_cache(maxsize=8)
def _fy_table(adjusted: bool, _params: tuple) -> pd.DataFrame:  # noqa: C901
    d = daily_panel()
    if d.empty:
        return pd.DataFrame()
    fy = _fy_index(d.index)
    n = fy.groupby(fy).size()
    complete = sorted(int(y) for y, k in n.items() if int(k) >= MIN_FY_DAYS)
    if not complete:
        return pd.DataFrame()
    sums = d.groupby(fy).sum()
    sums = sums.loc[[y for y in complete if y in sums.index]]
    deps, wdp = _deps(sums.columns), _wd_primary(sums.columns)
    out = pd.DataFrame({
        "receipts": sums[deps].sum(axis=1),
        "primary_spend": sums[wdp].sum(axis=1),
        "interest": sums["wd_interest"] if "wd_interest" in sums.columns else np.nan,
    }, index=sums.index.astype(int))
    out.index.name = "year"
    for c in deps + wdp + (["wd_interest"] if "wd_interest" in sums.columns else []):
        out[c] = sums[c]
    if adjusted:
        from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ

        hist_end = pd.read_parquet(HIST).index.max()
        first_fc = int(_fy_index(pd.DatetimeIndex([pd.Timestamp(hist_end)])).iloc[0])
        hist = d[d.index <= pd.Timestamp(hist_end)]
        out = ADJ.adjust(out, hist, first_fc)
    return out


def rebased(fred: pd.DataFrame, origin: int, years: list[int]) -> pd.DataFrame | None:
    """FRED levels at ``origin`` × DTS component growth from that FY → ``years`` (forecast FYs).

    ``fred`` columns: ``receipts``, ``primary``, ``interest`` (FRED $bn). Returns None if the
    DTS origin FY or a requested year is missing.
    """
    fy = fy_table()
    if fy.empty or origin not in fy.index or origin not in fred.index:
        return None
    years = [int(y) for y in years if int(y) != origin]
    if not years or any(y not in fy.index for y in years):
        return None
    deps, wdp = _deps(fy.columns), _wd_primary(fy.columns)
    r0 = float(fred.at[origin, "receipts"])
    p0 = float(fred.at[origin, "primary"])
    i0 = float(fred.at[origin, "interest"])
    dts_r0 = float(fy.at[origin, "receipts"])
    dts_p0 = float(fy.at[origin, "primary_spend"])
    dts_i0 = float(fy.at[origin, "interest"])
    if min(abs(dts_r0), abs(dts_p0), abs(r0), abs(p0)) < 1e-6:
        return None

    # component shares of FRED at origin from DTS mix
    rec_share = {c: float(fy.at[origin, c]) / dts_r0 for c in deps}
    pri_share = {c: float(fy.at[origin, c]) / dts_p0 for c in wdp}

    rows = []
    for y in years:
        rec = 0.0
        for c, sh in rec_share.items():
            base = r0 * sh
            g0 = float(fy.at[origin, c])
            rec += base * (float(fy.at[y, c]) / g0) if abs(g0) > 1e-9 else base * (float(fy.at[y, "receipts"]) / dts_r0)
        pri = 0.0
        for c, sh in pri_share.items():
            base = p0 * sh
            g0 = float(fy.at[origin, c])
            pri += base * (float(fy.at[y, c]) / g0) if abs(g0) > 1e-9 else base * (float(fy.at[y, "primary_spend"]) / dts_p0)
        if abs(dts_i0) > 1e-9 and np.isfinite(dts_i0):
            interest = i0 * float(fy.at[y, "interest"]) / dts_i0
        else:
            interest = i0
        rows.append({"year": y, "receipts": rec, "primary_spend": pri, "interest": interest,
                     "rule": "dts_fy"})
    return pd.DataFrame(rows).set_index("year")


def available() -> bool:
    return HIST.exists() and FC.exists() and not fy_table().empty


fy_table.cache_clear = _fy_table.cache_clear  # type: ignore[attr-defined]
