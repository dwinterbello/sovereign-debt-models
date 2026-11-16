"""Auction-micro demand to sit next to supply DV01 z-scores.

UK cover / tail / PAOF come from the tracked long-end D2.1A extract
(`40_auction_ops.csv`). DE / FR / IT / US use bid-to-cover on the ops file.
IT stores cover × 100; anything above 20 is divided by 100.

Sleeve BTC (all / int / long) is residual maturity: int is (2y, 10y], long is >10y.
UK coupons use DMO `years_to_maturity` because ops BTC is bills-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.isin_flow import market_ops
from issuance.paths import CACHE, REPO
from issuance.rv_metrics import _coupon_ops, rolling_z

UK_AUCTIONS = REPO / "projects" / "uk_long_end" / "workbook" / "40_auction_ops.csv"
UK_D21A = REPO / "data" / "raw" / "uk" / "D2_1A.pdf"
Z_WINDOW = 36
Z_MINP = 18
SLEEVES = ("all", "int", "long")


def normalize_btc(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s <= 20.0, s / 100.0)


def _weighted(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def sleeve_of_years(yrs: float | pd.Series) -> pd.Series | str | float:
    """Map residual maturity years to int / long; ≤2y is neither (bills / short coupons)."""
    if isinstance(yrs, pd.Series):
        y = pd.to_numeric(yrs, errors="coerce")
        out = pd.Series(np.where(y > 10.0, "long", np.where(y > 2.0, "int", "short")), index=yrs.index)
        return out.where(y.notna())
    y = float(yrs) if pd.notna(yrs) else float("nan")
    if not np.isfinite(y):
        return float("nan")
    if y > 10.0:
        return "long"
    if y > 2.0:
        return "int"
    return "short"


def _agg_btc(sub: pd.DataFrame, cover_col: str, w_col: str) -> dict:
    return {
        "btc": _weighted(sub[cover_col], sub[w_col]),
        "n_auctions": int(pd.to_numeric(sub[cover_col], errors="coerce").notna().sum()),
        "coupon_issued_mn": float(pd.to_numeric(sub[w_col], errors="coerce").fillna(0).sum()),
    }


def _ops_btc_monthly(ops: pd.DataFrame) -> pd.DataFrame:
    """Coupon auction BTC by country × month × sleeve (all / int / long). UK skipped (DMO)."""
    g = market_ops(_coupon_ops(ops))
    if g.empty:
        return pd.DataFrame()
    g = g[g["amount_issued"] > 0].copy()
    g["month"] = pd.to_datetime(g["op_dt"]).dt.to_period("M").dt.to_timestamp()
    g["btc_n"] = normalize_btc(g["btc"]) if "btc" in g.columns else np.nan
    if "amount_bid" in g.columns:
        cover = pd.to_numeric(g["amount_bid"], errors="coerce") / pd.to_numeric(g["amount_issued"], errors="coerce")
        g["btc_n"] = g["btc_n"].fillna(cover.where((cover > 0.4) & (cover < 20)))
    g["sleeve"] = sleeve_of_years(g["resid_mat_yrs"]) if "resid_mat_yrs" in g.columns else "short"
    rows = []
    for (c, m), sub in g.groupby(["country", "month"]):
        if c == "UK":
            continue
        base = {"country": c, "month": m}
        rows.append({**base, "sleeve": "all", **_agg_btc(sub, "btc_n", "amount_issued")})
        for sl in ("int", "long"):
            part = sub[sub["sleeve"] == sl]
            if part.empty:
                continue
            rows.append({**base, "sleeve": sl, **_agg_btc(part, "btc_n", "amount_issued")})
    return pd.DataFrame(rows)


def _parse_d21a(path) -> pd.DataFrame:
    """DMO D2.1A PDF. Text order is total, cover, price, yield, tail, [paof], auction, cash."""
    import re
    from pypdf import PdfReader

    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    num = r"(?:N/A|\d{1,3}(?:,\d{3})*\.\d+)"
    rows = []
    for part in re.split(r"(?=\d{2}-[A-Za-z]{3}-\d{4})", text):
        m = re.match(r"(\d{2}-[A-Za-z]{3}-\d{4})\s+(.+)", part.strip(), re.S)
        if not m:
            continue
        date, rest = m.group(1), m.group(2)
        nums = re.findall(num, rest)
        name = re.sub(r"\s+", " ", re.split(num, rest)[0]).strip()
        if len(nums) < 7 or not any(w in name for w in ("Gilt", "Stock", "gilt")):
            continue

        def f(x: str) -> float:
            return float("nan") if x == "N/A" else float(x.replace(",", ""))

        vals = [f(x) for x in nums[:8]]
        if len(vals) >= 8:
            total, cover, price, yld, tail, paof, auction, cash = vals[:8]
        else:
            total, cover, price, yld, tail, auction, cash = vals[:7]
            paof = 0.0
        rows.append({
            "date": date, "name": name, "total_nominal_mn": total, "cover": cover,
            "average_accepted_price": price, "yield_at_aap": yld, "tail_bp": tail,
            "nominal_paof_mn": paof, "nominal_auction_mn": auction, "cash_raised_mn": cash,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], format="%d-%b-%Y")
    out["paof_takeup"] = out["nominal_paof_mn"] / out["nominal_auction_mn"].replace(0, pd.NA)
    out["linker"] = out["name"].str.contains("ndex", case=False)
    return out


def _attach_uk_maturity(parsed: pd.DataFrame) -> pd.DataFrame:
    """D2.1A PDF has no residual maturity; join years from the tracked DMO extract."""
    if parsed.empty or not UK_AUCTIONS.exists():
        return parsed
    csv = pd.read_csv(UK_AUCTIONS)
    if "years_to_maturity" not in csv.columns:
        return parsed
    left = parsed.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    csv = csv.copy()
    csv["date"] = pd.to_datetime(csv["date"], errors="coerce")
    keep = [c for c in ("date", "name", "years_to_maturity", "isin", "bucket", "redemption_date")
            if c in csv.columns]
    merged = left.merge(csv[keep].drop_duplicates(["date", "name"]), on=["date", "name"], how="left")
    return merged


def _uk_auction_frame() -> pd.DataFrame:
    if UK_D21A.exists():
        try:
            parsed = _parse_d21a(UK_D21A)
        except Exception as exc:
            print(f"  D2.1A PDF unavailable ({exc})")
            parsed = pd.DataFrame()
        if not parsed.empty:
            parsed = _attach_uk_maturity(parsed)
            n = int(pd.to_numeric(parsed.get("years_to_maturity"), errors="coerce").notna().sum())
            print(f"  UK auctions from D2.1A PDF: {len(parsed)} rows through {parsed['date'].max().date()}"
                  f" ({n} with maturity)")
            return parsed
    if not UK_AUCTIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(UK_AUCTIONS)


def _uk_years(raw: pd.DataFrame) -> pd.Series:
    if "years_to_maturity" in raw.columns:
        return pd.to_numeric(raw["years_to_maturity"], errors="coerce")
    if "redemption_date" in raw.columns and "date" in raw.columns:
        return (pd.to_datetime(raw["redemption_date"], errors="coerce")
                - pd.to_datetime(raw["date"], errors="coerce")).dt.days / 365.25
    return pd.Series(np.nan, index=raw.index)


def _uk_auction_monthly() -> pd.DataFrame:
    raw = _uk_auction_frame()
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw["month"] = pd.to_datetime(raw["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    raw = raw.dropna(subset=["month"])
    w = pd.to_numeric(raw.get("total_nominal_mn", raw.get("nominal_mn")), errors="coerce")
    raw["_w"] = w
    raw["sleeve"] = sleeve_of_years(_uk_years(raw))
    rows = []
    for m, sub in raw.groupby("month"):
        base = {"country": "UK", "month": m}
        rows.append({
            **base, "sleeve": "all",
            "btc": _weighted(sub["cover"], sub["_w"]),
            "tail_bp": _weighted(sub["tail_bp"], sub["_w"]),
            "paof_takeup": _weighted(sub.get("paof_takeup"), sub["_w"]) if "paof_takeup" in sub.columns else np.nan,
            "cover_z_dmo": _weighted(sub.get("cover_z"), sub["_w"]) if "cover_z" in sub.columns else np.nan,
            "tail_z_dmo": _weighted(sub.get("tail_z"), sub["_w"]) if "tail_z" in sub.columns else np.nan,
            "n_auctions": int(len(sub)),
            "coupon_issued_mn": float(sub["_w"].fillna(0).sum()),
        })
        for sl in ("int", "long"):
            part = sub[sub["sleeve"] == sl]
            if part.empty:
                continue
            rows.append({
                **base, "sleeve": sl,
                "btc": _weighted(part["cover"], part["_w"]),
                "tail_bp": _weighted(part["tail_bp"], part["_w"]),
                "paof_takeup": _weighted(part.get("paof_takeup"), part["_w"]) if "paof_takeup" in part.columns else np.nan,
                "n_auctions": int(len(part)),
                "coupon_issued_mn": float(part["_w"].fillna(0).sum()),
            })
    return pd.DataFrame(rows)


def _wide_from_long(long: pd.DataFrame) -> pd.DataFrame:
    """One row per country×month: btc / btc_int / btc_long (+ z and UK extras on all)."""
    if long.empty:
        return long
    all_ = long[long["sleeve"] == "all"].drop(columns=["sleeve"], errors="ignore")
    out = all_.copy()
    for sl in ("int", "long"):
        sub = long[long["sleeve"] == sl][["country", "month", "btc", "n_auctions", "coupon_issued_mn"]].rename(
            columns={"btc": f"btc_{sl}", "n_auctions": f"n_auctions_{sl}",
                     "coupon_issued_mn": f"coupon_issued_mn_{sl}"}
        )
        out = out.merge(sub, on=["country", "month"], how="left")
    out = out.sort_values(["country", "month"])
    for col, zcol in (("btc", "btc_z"), ("btc_int", "btc_z_int"), ("btc_long", "btc_z_long")):
        if col not in out.columns:
            out[col] = np.nan
        out[zcol] = out.groupby("country", group_keys=False)[col].apply(
            lambda s: rolling_z(s, Z_WINDOW, Z_MINP)
        )
    if "tail_bp" in out.columns:
        out["tail_z"] = out.groupby("country", group_keys=False)["tail_bp"].apply(
            lambda s: rolling_z(s, Z_WINDOW, Z_MINP)
        )
    else:
        out["tail_z"] = np.nan
    return out


def demand_panel(ops: pd.DataFrame) -> pd.DataFrame:
    parts = [_ops_btc_monthly(ops), _uk_auction_monthly()]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True)
    return _wide_from_long(long)


def write_demand(ops: pd.DataFrame) -> pd.DataFrame:
    out = demand_panel(ops)
    if not out.empty:
        out.to_parquet(CACHE / "rv_demand.parquet", index=False)
    return out
