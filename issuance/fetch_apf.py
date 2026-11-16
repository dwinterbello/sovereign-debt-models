"""BoE APF gilt operations from the published purchase/sale workbooks."""

from __future__ import annotations

import pandas as pd

from issuance.http import get_bytes
from issuance.paths import CACHE, RAW, REPO
from issuance.tenor import attach_tenor

WORKBOOK_OPS = REPO / "projects" / "uk_long_end" / "workbook" / "42_issuance_concession_ops.csv"

BOE = {
    "purchases": (
        "https://www.bankofengland.co.uk/-/media/boe/files/markets/"
        "asset-purchase-facility/gilt-purchase-operational-results.xlsx",
        "APF Gilts",
    ),
    "sales": (
        "https://www.bankofengland.co.uk/-/media/boe/files/markets/"
        "asset-purchase-facility/gilt-sales-time-series.xlsx",
        "APF gilt sales",
    ),
}


def _read_ops(path, direction: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header = None
    for i in range(min(8, len(raw))):
        vals = [str(v).lower() for v in raw.iloc[i] if pd.notna(v)]
        joined = " ".join(vals)
        if "operation date" in joined and "isin" in joined:
            header = i
            break
    if header is None:
        raise RuntimeError(f"APF header not found in {path}")
    cols = [str(c).strip() for c in raw.iloc[header].tolist()]
    body = raw.iloc[header + 1:].copy()
    body.columns = cols
    # First column is sometimes an empty index on the sales sheet.
    if body.columns[0] in {"nan", "None", ""} or str(body.columns[0]).lower() == "nan":
        body = body.iloc[:, 1:]
    rename = {}
    for col in body.columns:
        cl = str(col).lower().replace("\n", " ")
        if cl == "operation date" or cl.startswith("operation date"):
            rename[col] = "op_dt"
        elif "settlement" in cl:
            rename[col] = "settlement_dt"
        elif cl == "isin":
            rename[col] = "isin"
        elif cl.startswith("bond"):
            rename[col] = "sec_desc_long"
        elif "total allocation (nominal" in cl:
            rename[col] = "amount"
        elif "weighted average accepted yield" in cl:
            rename[col] = "yield"
        elif "weighted average accepted price" in cl:
            rename[col] = "price"
    df = body.rename(columns=rename)
    keep = [c for c in ("op_dt", "settlement_dt", "isin", "sec_desc_long", "amount", "yield", "price") if c in df.columns]
    df = df[keep].copy()
    df["op_dt"] = pd.to_datetime(df["op_dt"], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    df["yield"] = pd.to_numeric(df.get("yield"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["direction"] = direction
    if direction == "sale":
        df["amount"] = -df["amount"].abs()
    df["program"] = "APF"
    return df.dropna(subset=["op_dt", "amount"])


def fetch_apf(force: bool = False) -> pd.DataFrame:
    dest_dir = RAW / "uk"
    frames = []
    for name, (url, _sheet) in BOE.items():
        dest = dest_dir / f"boe_apf_{name}.xlsx"
        if force or not dest.exists():
            get_bytes(url, dest=dest, timeout=180)
        frames.append(_read_ops(dest, "purchase" if name == "purchases" else "sale"))
    ops = pd.concat(frames, ignore_index=True)
    ops = _join_gilt_maturity(ops)
    ops = attach_tenor(ops)
    ops = ops.sort_values("op_dt").reset_index(drop=True)
    ops["apf_stock"] = ops.groupby("isin")["amount"].cumsum()
    ops["apf_stock_total"] = ops["amount"].cumsum()
    ops.to_parquet(CACHE / "apf_operations.parquet", index=False)
    latest = ops.sort_values("op_dt").groupby("isin").tail(1)
    latest.to_parquet(CACHE / "apf_holdings.parquet", index=False)
    return ops


def _join_gilt_maturity(ops: pd.DataFrame) -> pd.DataFrame:
    """APF workbooks have no redemption date. Residual tenor is required to net official demand."""
    parts = []
    cached = CACHE / "ops_uk.parquet"
    if cached.exists():
        uk = pd.read_parquet(cached)
        parts.append(uk[["isin", "maturity_dt"]])
    if WORKBOOK_OPS.exists():
        raw = pd.read_csv(WORKBOOK_OPS, usecols=lambda c: c in {"isin", "redemption_date"})
        parts.append(pd.DataFrame({
            "isin": raw["isin"].astype("string").str.upper(),
            "maturity_dt": pd.to_datetime(raw.get("redemption_date"), errors="coerce"),
        }))
    if not parts:
        return ops
    mat = pd.concat(parts, ignore_index=True)
    mat["isin"] = mat["isin"].astype("string").str.upper()
    mat = mat.dropna(subset=["isin", "maturity_dt"]).drop_duplicates("isin")
    out = ops.copy()
    out["isin"] = out["isin"].astype("string").str.upper()
    out = out.merge(mat, on="isin", how="left")
    return out
