"""Parse AFT latest-auctions / communiqué HTML (transposed tables)."""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from issuance.fr_text import parse_coupon, parse_french_maturity, scale_price, scale_yield

_LABEL = {
    "auction date": "op_dt",
    "date d'adjudication": "op_dt",
    "date dadjudication": "op_dt",
    "settlement date": "issue_dt",
    "date de règlement": "issue_dt",
    "date de reglement": "issue_dt",
    "maturity": "maturity_dt",
    "échéance": "maturity_dt",
    "echeance": "maturity_dt",
    "amount bid": "amount_bid",
    "volume demandé": "amount_bid",
    "volume demande": "amount_bid",
    "total amount issued": "amount_issued",
    "volume total émis": "amount_issued",
    "volume total emis": "amount_issued",
    "amount served": "amount_served",
    "volume adjugé": "amount_served",
    "volume adjuge": "amount_served",
    "bid to cover ratio": "btc",
    "taux de couverture": "btc",
    "weighted average price": "price",
    "prix moyen pondéré": "price",
    "prix moyen pondere": "price",
    "weighted average rate": "yield",
    "taux moyen pondéré": "yield",
    "taux moyen pondere": "yield",
    "isin code": "isin",
    "code isin": "isin",
}


def _norm_label(text: str) -> str:
    t = str(text).lower()
    t = t.replace("’", "'").replace("`", "'")
    t = re.sub(r"[*]+", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("é", "e").replace("è", "e")
    return t


def _aft_num(text: object) -> float:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return float("nan")
    t = str(text).strip().replace("\xa0", " ").replace("%", "")
    t = t.replace(" ", "")
    if t in {"", "-", "—", "n/a", "na"}:
        return float("nan")
    if "--" in t or "–" in t:
        return float("nan")
    if "," in t and "." in t:
        t = t.replace(",", "")
    elif "," in t:
        left, right = t.split(",", 1)
        if right.isdigit() and len(right) == 3 and left.replace("-", "").isdigit():
            t = left + right
        else:
            t = left + "." + right
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _sec_type(name: object) -> str:
    text = str(name or "").lower()
    if "oati" in text or "oat€" in text or "oatei" in text:
        return "Linker"
    if "btf" in text:
        return "Bill"
    return "Nominal"


def parse_aft_tables(html: str, source: str = "AFT latest") -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for table in soup.find_all("table"):
        grid = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                grid.append(cells)
        if len(grid) < 3:
            continue
        headers = grid[0]
        n = max(len(r) for r in grid)
        mapped: dict[str, list] = {}
        names = [""] * n
        for j, h in enumerate(headers):
            names[j] = h
        for raw in grid[1:]:
            label = _norm_label(raw[0] if raw else "")
            key = _LABEL.get(label)
            if key is None:
                for needle, dest in _LABEL.items():
                    if label.startswith(needle):
                        key = dest
                        break
            if key is None:
                continue
            vals = list(raw[1:]) + [""] * (n - len(raw))
            mapped[key] = vals
        if "op_dt" not in mapped:
            continue
        width = len(mapped["op_dt"])
        for i in range(width):
            desc = names[i + 1] if i + 1 < len(names) else ""
            issued = _aft_num((mapped.get("amount_issued") or [""] * width)[i] if i < len(mapped.get("amount_issued") or []) else "")
            served = _aft_num((mapped.get("amount_served") or [""] * width)[i] if i < len(mapped.get("amount_served") or []) else "")
            amount = issued if pd.notna(issued) else served
            isin = ""
            if "isin" in mapped and i < len(mapped["isin"]):
                isin = str(mapped["isin"][i]).strip().upper()
            mat = pd.NaT
            if "maturity_dt" in mapped and i < len(mapped["maturity_dt"]):
                mat = parse_french_maturity(mapped["maturity_dt"][i])
            if pd.isna(mat):
                mat = parse_french_maturity(desc)
            rows.append({
                "op_dt": parse_french_maturity(mapped["op_dt"][i] if i < len(mapped["op_dt"]) else ""),
                "issue_dt": parse_french_maturity((mapped.get("issue_dt") or [""])[i] if i < len(mapped.get("issue_dt") or []) else ""),
                "maturity_dt": mat,
                "isin": isin or pd.NA,
                "sec_desc_long": desc,
                "amount_bid": _aft_num((mapped.get("amount_bid") or [""])[i] if i < len(mapped.get("amount_bid") or []) else ""),
                "amount_issued": amount,
                "btc": _aft_num((mapped.get("btc") or [""])[i] if i < len(mapped.get("btc") or []) else ""),
                "price": scale_price(pd.Series([_aft_num((mapped.get("price") or [""])[i] if i < len(mapped.get("price") or []) else "")])).iloc[0],
                "yield": scale_yield(pd.Series([_aft_num((mapped.get("yield") or [""])[i] if i < len(mapped.get("yield") or []) else "")])).iloc[0],
                "coupon": parse_coupon(desc),
                "op_type": "Auction",
                "sec_type": _sec_type(desc),
                "source": source,
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out[out["op_dt"].notna() & (pd.to_numeric(out["amount_issued"], errors="coerce").fillna(0) != 0)]
    return out.reset_index(drop=True)
