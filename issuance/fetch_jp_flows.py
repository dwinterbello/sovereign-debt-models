"""Japan cross-border bond flows, Ministry of Finance.

Two files from "International Transactions in Securities" (designated major investors):

    week.csv     weekly, 2005+. Residents' net purchases of foreign securities (assets) and
                 non-residents' net purchases of Japanese securities (liabilities), split into
                 equity, long-term debt and short-term debt.
    monthb3.csv  monthly, 2005+. Residents' net purchases of foreign long-term debt by investor
                 type (banks, trust accounts, life insurers, investment trusts, ...).

Units in the files are 100 million yen; the cache is in billion yen. Positive = net purchase.
The weekly series is the fastest read of repatriation (Japanese selling foreign bonds) and of
foreign demand for JGBs; the monthly split says who is doing it.
"""
from __future__ import annotations

import io
import re
import unicodedata

import pandas as pd

from issuance.http import get_bytes
from issuance.paths import CACHE, RAW

BASE = "https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/"
WEEK_COLS = [
    "a_eq_acq", "a_eq_disp", "a_eq_net", "a_lt_acq", "a_lt_disp", "a_lt_net", "a_sub_net",
    "a_st_acq", "a_st_disp", "a_st_net", "a_tot_net",
    "l_eq_acq", "l_eq_disp", "l_eq_net", "l_lt_acq", "l_lt_disp", "l_lt_net", "l_sub_net",
    "l_st_acq", "l_st_disp", "l_st_net", "l_tot_net",
]
INVESTORS = [
    "total", "public_sector", "central_bank", "general_government", "deposit_taking", "banks",
    "trust_banks", "other_sectors", "other_financial", "trust_accounts", "banks_trust",
    "trust_banks_trust", "securities_firms", "life_insurers", "nonlife_insurers",
    "investment_trusts", "others",
]


def _rows(payload: bytes) -> list[list[str]]:
    text = unicodedata.normalize("NFKC", payload.decode("shift_jis", errors="replace"))
    return pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False).values.tolist()


def _num(cell: str) -> float:
    cell = str(cell).replace(",", "").strip()
    return float(cell) if re.fullmatch(r"-?\d+(\.\d+)?", cell) else float("nan")


def _week_end(label: str) -> pd.Timestamp | None:
    m = re.match(r"\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\s*[〜~-]\s*(\d{1,2})\.\s*(\d{1,2})", label)
    if not m:
        return None
    y, m1, _, m2, d2 = map(int, m.groups())
    return pd.Timestamp(year=y + (1 if m2 < m1 else 0), month=m2, day=d2)


def weekly(payload: bytes) -> pd.DataFrame:
    out = []
    for row in _rows(payload):
        end = _week_end(str(row[0]))
        if end is None:
            continue
        vals = [_num(c) for c in row[1:1 + len(WEEK_COLS)]]
        out.append({"week_end": end, **dict(zip(WEEK_COLS, vals))})
    df = pd.DataFrame(out).drop_duplicates("week_end").sort_values("week_end")
    df[WEEK_COLS] = df[WEEK_COLS] / 10.0
    return df.reset_index(drop=True)


def by_investor(payload: bytes) -> pd.DataFrame:
    out, year = [], None
    for row in _rows(payload):
        y = re.match(r"\s*(\d{4})", str(row[0]))
        if y:
            year = int(y.group(1))
        mon = str(row[2]).strip()[:3]
        if year is None or mon not in {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                       "Oct", "Nov", "Dec"}:
            continue
        month = pd.Timestamp(f"{year}-{mon}-01")
        for k, name in enumerate(INVESTORS):
            net = _num(row[3 + 3 * k + 2])
            out.append({"month": month, "investor": name, "net_bn_jpy": net / 10.0})
    df = pd.DataFrame(out)
    return df.dropna(subset=["net_bn_jpy"]).reset_index(drop=True)


def fetch_jp_flows() -> dict[str, pd.DataFrame]:
    raw = RAW / "jp"
    wk = weekly(get_bytes(BASE + "week.csv", raw / "mof_week.csv", min_bytes=10_000))
    inv = by_investor(get_bytes(BASE + "monthb3.csv", raw / "mof_monthb3.csv", min_bytes=10_000))
    # Identity: the investor split sums to the total row (rounding in the source is ±1 unit).
    piv = inv.pivot_table(index="month", columns="investor", values="net_bn_jpy")
    parts = piv[["public_sector", "deposit_taking", "other_sectors"]].sum(axis=1)
    gap = (piv["total"] - parts).abs().max()
    if gap > 1.0:
        raise ValueError(f"MoF investor split does not add to total (max gap {gap:.1f}bn)")
    wk.to_parquet(CACHE / "jp_flows_weekly.parquet", index=False)
    inv.to_parquet(CACHE / "jp_flows_investor.parquet", index=False)
    print(f"  jp flows: weekly {wk.week_end.min().date()}..{wk.week_end.max().date()} ({len(wk)}), "
          f"investor months {inv.month.nunique()}")
    return {"weekly": wk, "investor": inv}


if __name__ == "__main__":
    fetch_jp_flows()
