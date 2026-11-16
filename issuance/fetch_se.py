"""Swedish government securities ISIN book + auction taps from Riksgälden.

    PYTHONPATH=. .venv/bin/python -m issuance.fetch_se [--force]

Sources:
  * ``outstanding_securities.csv`` (Qlik CH1073) — live face per ISIN (linkers × index
    factor). ``op_type=Other``, floored ``op_dt``. Drives stock / MTM / WAM /
    ``cashflow.components``.
  * ``auctions.csv`` (Qlik CH926) — outright / syndication / buyback allocations with
    ``avg_yield``. Joined to ISINs via series name / SGB loan number / STB maturity.
    Drives ``isin_flow`` / ``dv01_isin`` / ``realised_ytd`` / linker issue share.

Matured series are not on today's outstanding list and have no ISIN there. They keep a
synthetic key (``SE-STB-yyyymmdd``, ``SE-SGB1047-yyyymmdd``) with maturity / coupon parsed
from the series name, so annual gross bills / bonds reconcile to the auction CSV. Stock
face still comes only from the outstanding rows (``isin_book.outstanding`` filters on source).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE
from issuance.schema import coerce_ops, concat_ops
from issuance.tenor import attach_tenor

SE_DIR = Path(__file__).resolve().parents[1] / "sovereign_cash" / "data" / "se"
SE_CSV = SE_DIR / "outstanding_securities.csv"
AUCTIONS_CSV = SE_DIR / "auctions.csv"
OUT_OPS = CACHE / "ops_se.parquet"

KEEP = ("SGB", "SGB IL", "STB", "GO")
TYPE_MAP = {"SGB": "Nominal", "SGB IL": "Linker", "STB": "Bill", "GO": "Nominal"}
STOCK_SOURCE = "Riksgälden outstanding CH1073"
AUCTION_SOURCE = "Riksgälden auctions CH926"

_MON = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "oct": 10, "nov": 11, "dec": 12,
}
_LOAN = re.compile(r"(SGB(?:\s+IL)?)\s+(\d+)", re.I)
_STB = re.compile(r"STB\s+(\d{1,2})\s+([A-Za-zåäö]{3})\w*\s+(\d{2})", re.I)
_TAIL_DATE = re.compile(r"(\d{1,2})\s+([A-Za-zåäö]{3})\w*\s+(\d{2})\s*$", re.I)
_CPN = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _norm_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip()
        .str.replace(",", ".", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def _loan_key(s) -> str | None:
    m = _LOAN.search(str(s))
    if not m:
        return None
    return f"{m.group(1).upper().replace('  ', ' ')} {m.group(2)}"


def _dmy(m: re.Match | None) -> pd.Timestamp:
    if not m:
        return pd.NaT
    mon = _MON.get(m.group(2).lower()[:3])
    if mon is None:
        return pd.NaT
    yy = int(m.group(3))
    # Auction tape starts 1996: two-digit years ≥ 90 are 19xx.
    return pd.Timestamp((1900 if yy >= 90 else 2000) + yy, mon, int(m.group(1)))


def _stb_maturity(series: str) -> pd.Timestamp:
    return _dmy(_STB.search(str(series)))


def _series_maturity(series: str) -> pd.Timestamp:
    """Maturity from the trailing ``d Mon yy`` of an auction series name (SGB and STB)."""
    return _dmy(_TAIL_DATE.search(str(series)))


def _series_coupon(series: str) -> float:
    if str(series).upper().startswith("STB"):
        return 0.0
    m = _CPN.search(str(series))
    return float(m.group(1).replace(",", ".")) if m else np.nan


def _synthetic_key(series: str, maturity: pd.Timestamp) -> str | None:
    """Stable id for a series with no live ISIN: ``SE-STB-yyyymmdd`` / ``SE-SGB1047-yyyymmdd``."""
    if pd.isna(maturity):
        return None
    d = pd.Timestamp(maturity).strftime("%Y%m%d")
    if str(series).upper().startswith("STB"):
        return f"SE-STB-{d}"
    loan = _loan_key(series)
    return f"SE-{loan.replace(' ', '')}-{d}" if loan else None


def outstanding_to_ops(path: Path | None = None) -> pd.DataFrame:
    """Riksgälden outstanding CSV → stock rows (one per live ISIN)."""
    path = path or SE_CSV
    if not path.is_file():
        raise FileNotFoundError(f"SE outstanding list missing: {path}")
    raw = pd.read_csv(path)
    se = raw[raw["Security/ Market"].isin(KEEP)].copy()
    if se.empty:
        raise RuntimeError("SE outstanding list has no SGB/SGB IL/STB/GO rows")
    nom = _num(se["Nominal amount"])
    idx = _num(se["Index factor*"]).fillna(1.0)
    face_mn = nom / 1e6 * np.where(se["Security/ Market"].eq("SGB IL"), idx, 1.0)
    cpn = _num(se["Coupon"])
    cpn = cpn.where(se["Coupon type"].astype(str).str.lower().ne("zero"), 0.0)
    issue = pd.to_datetime(se["Issue date"], errors="coerce")
    mat = pd.to_datetime(se["Maturity date"], errors="coerce")
    asof = pd.Timestamp.today().normalize()
    # Stock snapshot: floored op_dt so book(end-y0) keeps the live bill ladder.
    op_floor = pd.Timestamp("1990-01-01")
    rows = pd.DataFrame({
        "op_dt": op_floor,
        "issue_dt": issue,
        "maturity_dt": mat,
        "isin": se["ISIN"].astype(str).str.strip().str.upper(),
        "sec_desc_long": se["Securities"].astype(str),
        "amount_issued": face_mn,
        "coupon": cpn,
        "yield": np.nan,
        "sec_type": se["Security/ Market"].map(TYPE_MAP),
        "op_type": "Other",
        "sec_desc": se["Security/ Market"].astype(str),
        "is_forecast": False,
        "source": STOCK_SOURCE,
    })
    rows = rows[rows["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9,12}$", na=False)]
    rows = rows[rows["amount_issued"].notna() & (rows["amount_issued"] > 0)]
    rows = rows[rows["maturity_dt"].notna() & (rows["maturity_dt"] > asof)]
    return attach_tenor(coerce_ops(rows, "SE"))


def _isin_map(path: Path | None = None) -> pd.DataFrame:
    """Live outstanding → join keys (exact name, loan number, STB maturity)."""
    path = path or SE_CSV
    raw = pd.read_csv(path)
    se = raw[raw["Security/ Market"].isin(KEEP)].copy()
    se["isin"] = se["ISIN"].astype(str).str.strip().str.upper()
    se = se[se["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9,12}$", na=False)]
    se["name"] = _norm_name(se["Securities"])
    se["loan"] = se["Securities"].map(_loan_key)
    se["maturity_dt"] = pd.to_datetime(se["Maturity date"], errors="coerce")
    se["coupon"] = _num(se["Coupon"])
    se["sec_type"] = se["Security/ Market"].map(TYPE_MAP)
    se["sec_desc_long"] = se["Securities"].astype(str)
    return se[["isin", "name", "loan", "maturity_dt", "coupon", "sec_type", "sec_desc_long", "Security/ Market"]]


def auctions_to_ops(path: Path | None = None, outstanding: Path | None = None) -> pd.DataFrame:
    """Auction results → ops taps: live ISIN where the join hits, else a synthetic series key."""
    path = path or AUCTIONS_CSV
    if not path.is_file():
        return coerce_ops(pd.DataFrame(), "SE")
    a = pd.read_csv(path)
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a[a["date"].notna() & a["type"].isin(["O", "B", "S"])].copy()
    a["allocated_mn"] = pd.to_numeric(a["allocated_mn"], errors="coerce")
    a["avg_yield"] = pd.to_numeric(a["avg_yield"], errors="coerce")
    # Buybacks are reported with negative amounts; the sign is reapplied from ``type`` below.
    a["allocated_mn"] = a["allocated_mn"].abs()
    a["tendered_mn"] = pd.to_numeric(a["tendered_mn"], errors="coerce").abs()
    a = a[a["allocated_mn"].notna() & (a["allocated_mn"] > 0)]
    a["name"] = _norm_name(a["series"])
    a["loan"] = a["series"].map(_loan_key)
    a["stb_mat"] = a["series"].map(_stb_maturity)

    ref = _isin_map(outstanding)
    by_name = ref.drop_duplicates("name").set_index("name")
    by_loan = ref[ref["loan"].notna()].drop_duplicates("loan").set_index("loan")
    by_mat = (
        ref[ref["Security/ Market"].eq("STB")]
        .dropna(subset=["maturity_dt"])
        .drop_duplicates("maturity_dt")
        .set_index("maturity_dt", drop=False)
    )

    hit = by_name.reindex(a["name"])
    isin = hit["isin"].to_numpy(copy=True, dtype=object)
    mat = hit["maturity_dt"].to_numpy(copy=True, dtype=object)
    cpn = hit["coupon"].to_numpy(copy=True, dtype=float)
    stype = hit["sec_type"].to_numpy(copy=True, dtype=object)
    desc = hit["sec_desc_long"].to_numpy(copy=True, dtype=object)

    loan_hit = a["loan"].notna() & pd.isna(isin)
    if loan_hit.any():
        L = by_loan.reindex(a.loc[loan_hit, "loan"])
        idx = np.flatnonzero(loan_hit.to_numpy())
        isin[idx] = L["isin"].to_numpy()
        mat[idx] = L["maturity_dt"].to_numpy()
        cpn[idx] = L["coupon"].to_numpy()
        stype[idx] = L["sec_type"].to_numpy()
        desc[idx] = L["sec_desc_long"].to_numpy()

    bill = a["series"].astype(str).str.upper().str.startswith("STB") & pd.isna(isin)
    if bill.any():
        B = by_mat.reindex(pd.to_datetime(a.loc[bill, "stb_mat"]))
        idx = np.flatnonzero(bill.to_numpy())
        isin[idx] = B["isin"].to_numpy()
        mat[idx] = B["maturity_dt"].to_numpy()
        cpn[idx] = B["coupon"].to_numpy()
        stype[idx] = "Bill"
        desc[idx] = B["sec_desc_long"].to_numpy()

    # Matured series are not on today's outstanding list: keep them on a synthetic
    # series+maturity key with maturity / coupon / type parsed from the series name.
    miss = pd.isna(isin)
    if miss.any():
        idx = np.flatnonzero(miss)
        ser = a["series"].astype(str).to_numpy()[idx]
        smat = [_series_maturity(s) for s in ser]
        isin[idx] = [_synthetic_key(s, m) for s, m in zip(ser, smat)]
        mat[idx] = smat
        cpn[idx] = [_series_coupon(s) for s in ser]
        up = np.char.upper(ser.astype(str))
        stype[idx] = np.where(np.char.startswith(up, "STB"), "Bill",
                              np.where(np.char.startswith(up, "SGB IL"), "Linker", "Nominal"))
        desc[idx] = ser

    a = a.assign(isin=isin, maturity_dt=mat, coupon=cpn, sec_type=stype, sec_desc_long=desc)
    a = a[a["isin"].notna() & a["maturity_dt"].notna()].copy()

    op_type = np.select(
        [
            a["auction_type"].astype(str).str.contains("Syndikering", case=False, na=False),
            a["type"].eq("B"),
            a["type"].eq("S"),
        ],
        ["Syndication", "Buyback", "Exchange"],
        default="Auction",
    )
    sign = np.where(a["type"].eq("B"), -1.0, 1.0)
    rows = pd.DataFrame({
        "op_dt": a["date"],
        "issue_dt": a["date"],
        "maturity_dt": pd.to_datetime(a["maturity_dt"]),
        "isin": a["isin"].astype(str),
        "sec_desc_long": a["sec_desc_long"].astype(str),
        "amount_issued": sign * a["allocated_mn"].astype(float),
        "amount_bid": pd.to_numeric(a["tendered_mn"], errors="coerce"),
        "coupon": pd.to_numeric(a["coupon"], errors="coerce"),
        "yield": a["avg_yield"],
        "sec_type": a["sec_type"].astype(str),
        "op_type": op_type,
        "sec_desc": a["series"].astype(str),
        "is_forecast": False,
        "source": AUCTION_SOURCE,
    })
    return attach_tenor(coerce_ops(rows, "SE"))


def fetch_se(force: bool = False) -> pd.DataFrame:
    """Write ``ops_se.parquet`` = stock snapshot + auction taps joined to live ISINs."""
    if force:
        try:
            from sovereign_cash.se import fetch as SF
            SF.fetch_outstanding()
            SF.fetch_auctions()
        except Exception as e:  # noqa: BLE001 — keep cached CSVs if Qlik is down
            print(f"fetch_se: Qlik refresh failed ({type(e).__name__}: {e}); using cached CSV")
    stock = outstanding_to_ops()
    taps = auctions_to_ops()
    ops = concat_ops([stock, taps])
    CACHE.mkdir(parents=True, exist_ok=True)
    ops.to_parquet(OUT_OPS, index=False)
    return ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ops = fetch_se(force=args.force)
    stock = ops[ops["source"].eq(STOCK_SOURCE)]
    taps = ops[ops["source"].eq(AUCTION_SOURCE)]
    face = stock.groupby("sec_type")["amount_issued"].sum() / 1e3
    print(f"ops_se stock: {len(stock)} lines, face SEK bn:\n{face.round(1).to_string()}")
    synth = taps["isin"].astype(str).str.startswith("SE-")
    print(f"ops_se taps: {len(taps)} rows, {int(synth.sum())} on synthetic keys "
          f"({taps['op_dt'].min().date()} → {taps['op_dt'].max().date()})")
    print(f"wrote {OUT_OPS}")


if __name__ == "__main__":
    main()
