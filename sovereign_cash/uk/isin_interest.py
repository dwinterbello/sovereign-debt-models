"""Point-in-time UK marketable interest from the live ISIN / ops book.

Used by ``bottom_up_macro`` experiment modes (``interest_mode``) and
``experiments/uk_isin_interest_edge``. No lookahead: only ops with
``op_dt ≤ asof`` and ``maturity_dt > asof``.

Perimeter is marketable gilts + bills (face × coupon, linker RPI accrual,
bill stock × Bank Rate). JW2P is a wider PS interest paid series; callers
scale or blend — this module does not invent the non-marketable gap.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from sovereign_cash.debt.isin_book import _coupon, _ops

UPLIFT = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "debt" / "uk_uplift.parquet"
RATES = __import__("pathlib").Path(__file__).resolve().parents[2] / "uk_flows" / "output" / "rates.parquet"


@lru_cache(maxsize=1)
def _uk_ops() -> pd.DataFrame:
    ops = _ops()
    u = ops[ops["country"] == "UK"].copy()
    u["coupon_pct"] = _coupon(u)
    return u


@lru_cache(maxsize=1)
def _index_ratio_by_year() -> pd.Series:
    if not UPLIFT.exists():
        return pd.Series(dtype=float)
    up = pd.read_parquet(UPLIFT)
    nom = up.loc[up["var"] == "il_nominal_bn"].set_index("year")["value"]
    upl = up.loc[up["var"] == "il_uplift_bn"].set_index("year")["value"]
    return ((nom + upl) / nom).dropna().astype(float)


@lru_cache(maxsize=1)
def _rpi_yoy() -> pd.Series:
    from sovereign_cash._vendor import load_rpi_all_items

    s = load_rpi_all_items().astype(float)
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s.pct_change(12)


@lru_cache(maxsize=1)
def _bank_rate() -> pd.Series:
    if not RATES.exists():
        return pd.Series(dtype=float)
    r = pd.read_parquet(RATES)
    br = r[r["series"] == "bank_rate"].copy()
    br["date"] = pd.to_datetime(br["date"])
    m = br.set_index("date")["value"].astype(float).to_period("M").groupby(level=0).last()
    m.index = m.index.to_timestamp()
    return m


def _ratio_at(asof: pd.Timestamp) -> float:
    """Last complete calendar-year index ratio known at ``asof`` (no same-year lookahead)."""
    ir = _index_ratio_by_year()
    y = int(asof.year) - 1
    prior = ir.loc[:y] if len(ir) else ir
    return float(prior.iloc[-1]) if len(prior) else 1.0


def _rpi_at(asof: pd.Timestamp) -> float:
    """Trailing RPI YoY (decimal) known through ``asof`` month."""
    s = _rpi_yoy().loc[: pd.Timestamp(asof).replace(day=1)].dropna()
    return float(s.iloc[-1]) if len(s) else 0.03


def _br_at(asof: pd.Timestamp) -> float:
    s = _bank_rate().loc[: pd.Timestamp(asof).replace(day=1)].dropna()
    return float(s.iloc[-1]) / 100.0 if len(s) else 0.04


def book_at(asof: pd.Timestamp) -> pd.DataFrame:
    """One row per live UK instrument at ``asof`` (face £mn)."""
    asof = pd.Timestamp(asof)
    u = _uk_ops()
    o = u[(u["op_dt"] <= asof) & (u["maturity_dt"] > asof) & u["amount_issued"].notna()].copy()
    key = o["isin"].astype("string")
    miss = key.isna() | key.isin(["<NA>", "None", ""])
    o = o.assign(key=key)
    o.loc[miss, "key"] = (
        "bill|" + o.loc[miss, "maturity_dt"].dt.strftime("%Y-%m-%d")
        + "|" + o.loc[miss].groupby("maturity_dt").cumcount().astype(str)
    )
    issued = o[o["amount_issued"] > 0]
    g = o.groupby("key", as_index=False).agg(
        face_mn=("amount_issued", "sum"),
        maturity_dt=("maturity_dt", "max"),
    )
    typ = issued.groupby("key")["sec_type"].agg(
        lambda s: s.mode().iloc[0] if len(s) else "Nominal")
    cpn = issued.groupby("key")["coupon_pct"].median()
    g = g.merge(typ.rename("sec_type"), on="key", how="left")
    g = g.merge(cpn.rename("coupon_pct"), on="key", how="left")
    return g[g["face_mn"] > 1.0].copy()


def annual_marketable(asof: pd.Timestamp) -> dict[str, float]:
    """Annualised marketable interest £bn at ``asof`` (coupon + linker uplift + bills)."""
    asof = pd.Timestamp(asof)
    b = book_at(asof)
    ratio = _ratio_at(asof)
    rpi = _rpi_at(asof)
    br = _br_at(asof)
    coup = b[b["sec_type"] != "Bill"]
    link = b[b["sec_type"] == "Linker"]
    bills = b[b["sec_type"] == "Bill"]
    scale = np.where(coup["sec_type"].eq("Linker"), ratio, 1.0)
    cpn = coup["coupon_pct"].fillna(0.0) / 100.0
    coupon = float((cpn * coup["face_mn"] / 1000.0 * scale).sum())
    lk = float(link["face_mn"].sum()) / 1000.0 * ratio
    uplift = lk * rpi
    bill_face = float(bills["face_mn"].sum()) / 1000.0 if len(bills) else 0.0
    bill_int = bill_face * br
    return {
        "coupon_bn": coupon,
        "uplift_bn": uplift,
        "bill_bn": bill_int,
        "interest_bn": coupon + uplift + bill_int,
        "linker_indexed_bn": lk,
        "bill_face_bn": bill_face,
        "index_ratio": ratio,
        "rpi_yoy": rpi,
        "bill_rate": br,
    }


def redeem_next_bn(asof: pd.Timestamp, months: int = 12) -> float:
    """Coupon-bond face maturing in the next ``months`` (£bn). Bills excluded."""
    asof = pd.Timestamp(asof)
    b = book_at(asof)
    coup = b[b["sec_type"] != "Bill"]
    end = asof + pd.DateOffset(months=months)
    return float(coup.loc[(coup["maturity_dt"] > asof) & (coup["maturity_dt"] <= end),
                          "face_mn"].sum()) / 1000.0


def redeem_by_month(asof: pd.Timestamp, h: int) -> np.ndarray:
    """Monthly coupon-bond redemptions (£bn) for origin+1 .. origin+h from the locked book."""
    asof = pd.Timestamp(asof)
    b = book_at(asof)
    coup = b[b["sec_type"] != "Bill"]
    months = pd.date_range(asof.replace(day=1) + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = np.zeros(h, dtype=float)
    if coup.empty:
        return out
    mat_m = coup["maturity_dt"].dt.to_period("M").dt.to_timestamp()
    face = coup["face_mn"].to_numpy(float) / 1000.0
    for i, m in enumerate(months):
        out[i] = float(face[mat_m == m].sum())
    return out


@lru_cache(maxsize=1)
def coupon_run_monthly() -> pd.Series:
    """Month-start index → annualised coupon run £bn (par face, no uplift) for diagnostics."""
    idx = pd.date_range("2008-01-01", pd.Timestamp.today().normalize(), freq="MS")
    vals = []
    for m in idx:
        asof = m + pd.offsets.MonthEnd(0)
        b = book_at(asof)
        coup = b[b["sec_type"] != "Bill"]
        cpn = coup["coupon_pct"].fillna(0.0) / 100.0
        vals.append(float((cpn * coup["face_mn"] / 1000.0).sum()))
    return pd.Series(vals, index=idx, name="coupon_run_bn")


@lru_cache(maxsize=1)
def marketable_annual_monthly() -> pd.Series:
    """Month-start → annualised marketable interest £bn (coupon + uplift + bills)."""
    idx = pd.date_range("2008-01-01", pd.Timestamp.today().normalize(), freq="MS")
    vals = [annual_marketable(m + pd.offsets.MonthEnd(0))["interest_bn"] for m in idx]
    return pd.Series(vals, index=idx, name="mkt_interest_bn")
