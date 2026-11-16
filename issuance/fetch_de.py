"""German Finanzagentur issuance history (official xlsx)."""

from __future__ import annotations

import pandas as pd

from issuance.http import FetchError, get_bytes
from issuance.paths import CACHE, RAW
from issuance.schema import coerce_ops
from issuance.tenor import attach_tenor

DE_URL = (
    "https://www.deutsche-finanzagentur.de/fileadmin/user_upload/"
    "Institutionelle-investoren/auktionen/emissionshistorie_en.xlsx"
)

OP_MAP = {
    "Auc": "Auction",
    "Syn": "Syndication",
    "OH": "Own Holdings",
    "M-A": "Market Adjustment",
}


def _sec_type(name: object) -> str:
    text = str(name).lower()
    if "usd" in text or "dollar" in text:
        return "Foreign"
    if "ilb" in text:
        return "Linker"
    if "bubill" in text:
        return "Bill"
    return "Nominal"


def fetch_de(force: bool = False) -> pd.DataFrame:
    xlsx = RAW / "de" / "de_issuance_history.xlsx"
    if force or not xlsx.exists():
        try:
            get_bytes(DE_URL, dest=xlsx, timeout=180)
        except FetchError as exc:
            if not xlsx.exists():
                raise
            print(f"  DE refetch failed ({exc}); using last xlsx")
    raw = pd.read_excel(xlsx, header=None)
    # First data row: numeric issue number in col 0 or an ISIN in col 2.
    start = None
    for i in range(min(40, len(raw))):
        isin = str(raw.iloc[i, 2]) if raw.shape[1] > 2 else ""
        if isin.startswith("DE") and len(isin) >= 12:
            start = i
            break
    if start is None:
        raise RuntimeError("Finanzagentur xlsx: no ISIN data row found")
    body = raw.iloc[start:].copy()
    body = body[body.iloc[:, 2].astype(str).str.startswith("DE")]
    out = pd.DataFrame({
        "op_dt": pd.to_datetime(body.iloc[:, 1], errors="coerce"),
        "isin": body.iloc[:, 2].astype(str).str.strip(),
        "sec_desc_long": body.iloc[:, 3],
        "coupon": pd.to_numeric(body.iloc[:, 4], errors="coerce") * 100.0,  # 0.0375 → 3.75
        "maturity_dt": pd.to_datetime(body.iloc[:, 5], errors="coerce"),
        "sec_desc": body.iloc[:, 6].astype(str).str.replace(" ", "", regex=False).str.lower(),
        "amount_issued": pd.to_numeric(body.iloc[:, 7], errors="coerce"),
        "op_type": body.iloc[:, 9].map(lambda x: OP_MAP.get(str(x).strip(), "Other")),
        "amount_bid": pd.to_numeric(body.iloc[:, 10], errors="coerce") if body.shape[1] > 10 else pd.NA,
        "price": pd.to_numeric(body.iloc[:, 15], errors="coerce") if body.shape[1] > 15 else pd.NA,
        "yield": pd.to_numeric(body.iloc[:, 16], errors="coerce") if body.shape[1] > 16 else pd.NA,
        "btc": pd.to_numeric(body.iloc[:, 18], errors="coerce") if body.shape[1] > 18 else pd.NA,
    })
    out["sec_type"] = out["sec_desc_long"].map(_sec_type)
    out["source"] = "DFA emissionshistorie"
    out["is_forecast"] = False
    ops = coerce_ops(attach_tenor(out), "DE")
    ops = ops[ops["op_dt"].notna() & (ops["amount_issued"].fillna(0) != 0)]
    ops.to_parquet(CACHE / "ops_de.parquet", index=False)
    return ops
