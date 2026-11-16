"""Flow demand score: composite per country, with tenor sleeves where data exist.

Pillars, each a signed z-score where positive means more demand for the country's bonds:

- auction: bid-to-cover (all six), plus int/long sleeve BTC where residual maturity is
  available; tail (UK, JP, inverted); PAOF take-up (UK); primary-dealer share inverted and
  indirect share (US), split by auction term when known; US end-user allotment share
  (investment funds + foreign + pension) by term sleeve.
- foreign: non-resident net purchases of government bonds. ECB BoP for DE/FR/IT, BoE for
  gilts, TIC long-term Treasuries, MoF weekly for Japanese long-term debt. JP also gets
  JSDA foreigners by maturity: medium → int, long+super_long → long.
- official: central-bank and official purchases in DV01 (negative under QT). Same series
  on every sleeve (IMF stock is not tenor-split).
- positioning: real-money futures flow. Shown, not scored.

Sleeve scores (`score`, `score_int`, `score_long`): equal-weight mean of auction / foreign /
official for that sleeve. Missing sleeve foreign falls back to country-level foreign.
Missing sleeve auction drops that pillar for the sleeve. Positioning stays out of the score
(see issuance/experiments/flow_score_test.py).

Every component is a three-month flow, z-scored against its own trailing 36 months (24 at
least) with no look-ahead. Late publishers carry forward up to three months; ``asof``
records the month each value comes from.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE

ROOT = Path(__file__).resolve().parents[1]
CFTC = ROOT / "us" / "fiscal" / "derived" / "cftc_ust_positioning.parquet"
COUNTRIES = ("UK", "DE", "FR", "IT", "ES", "US", "JP")
PILLARS = ("auction", "foreign", "official", "positioning")
SCORED = ("auction", "foreign", "official")
SLEEVES = ("all", "int", "long")
SCORE_PILLARS = ("score", "score_int", "score_long")
Z_WINDOW, Z_MIN, CARRY = 36, 24, 3
START = pd.Timestamp("2005-01-01")
COLS = ["country", "month", "pillar", "component", "label", "value", "sign", "sleeve"]

# JSDA maturity buckets → score sleeve.
JSDA_SLEEVE = {"medium": "int", "long": "long", "super_long": "long"}
# US auction term label → residual years (bills / FRN excluded).
US_TERM_Y = {
    "2y": 2.0, "3y": 3.0, "5y": 5.0, "7y": 7.0, "10y": 10.0, "20y": 20.0, "30y": 30.0,
    "TIPS 5y": 5.0, "TIPS 10y": 10.0, "TIPS 20y": 20.0, "TIPS 30y": 30.0,
}
US_END_USER = ("inv_funds", "foreign", "pension_ins")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    s = s.dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return getattr(s.groupby(level=0), how)()


def _comp(country, pillar, component, label, s: pd.Series, sign: int = 1,
          sleeve: str = "all") -> pd.DataFrame:
    s = s.dropna()
    if s.empty:
        return pd.DataFrame(columns=COLS)
    return pd.DataFrame({
        "country": country, "month": s.index, "pillar": pillar, "component": component,
        "label": label, "value": s.values.astype(float), "sign": sign, "sleeve": sleeve,
    })


def _sleeve_of_years(y: float) -> str | None:
    if not np.isfinite(y):
        return None
    if y > 10.0:
        return "long"
    if y > 2.0:
        return "int"
    return None


def _demand() -> pd.DataFrame:
    d = _read(CACHE / "rv_demand.parquet")
    if d.empty:
        d = _read(CACHE / "rv_supply.parquet")
    return d


def _auction() -> list[pd.DataFrame]:
    dem = _demand()
    out: list[pd.DataFrame] = []
    cover_cols = (("all", "btc", "Bid-to-cover"),
                  ("int", "btc_int", "Bid-to-cover (2–10y)"),
                  ("long", "btc_long", "Bid-to-cover (>10y)"))
    for c in COUNTRIES:
        g = dem[dem["country"] == c].set_index("month").sort_index() if not dem.empty else pd.DataFrame()
        if g.empty:
            continue
        for sleeve, col, label in cover_cols:
            if col not in g.columns or g[col].notna().sum() < Z_MIN:
                continue
            out.append(_comp(c, "auction", f"cover_{sleeve}" if sleeve != "all" else "cover",
                             label, _monthly(g[col]).rolling(3, min_periods=1).mean(), sleeve=sleeve))
        if c == "UK" and "tail_bp" in g.columns:
            out.append(_comp(c, "auction", "tail", "Tail (bp)",
                             _monthly(g["tail_bp"]).rolling(3, min_periods=1).mean(), -1))
            if "paof_takeup" in g.columns:
                out.append(_comp(c, "auction", "paof", "PAOF take-up",
                                 _monthly(g["paof_takeup"]).rolling(3, min_periods=1).mean()))
    jp = _read(CACHE / "ops_jp.parquet")
    if not jp.empty:
        a = jp[jp["op_type"].astype(str).eq("Auction") & jp["sec_type"].astype(str).ne("Bill")].set_index("op_dt")["tail_bp"]
        out.append(_comp("JP", "auction", "tail", "Tail (bp)",
                         _monthly(a).rolling(3, min_periods=1).mean(), -1))
    out.extend(_us_bidder_auction())
    out.extend(_us_allot_auction())
    return out


def _us_term_sleeve(term: str) -> str | None:
    t = str(term).strip()
    if t in US_TERM_Y:
        return _sleeve_of_years(US_TERM_Y[t])
    m = re.match(r"(?:TIPS\s+)?(\d+)y", t, re.I)
    if m:
        return _sleeve_of_years(float(m.group(1)))
    return None


def _us_bidder_auction() -> list[pd.DataFrame]:
    bid = _read(CACHE / "us_bidders.parquet")
    if bid.empty:
        return []
    b = bid.copy()
    b["sleeve"] = b["term"].map(_us_term_sleeve)
    b = b.dropna(subset=["sleeve"])
    b["auction_date"] = pd.to_datetime(b["auction_date"])
    out = []
    for sleeve, sub in b.groupby("sleeve"):
        idx = sub.set_index("auction_date")
        out.append(_comp("US", "auction", f"dealer_{sleeve}", f"Primary-dealer share ({sleeve})",
                         _monthly(idx["dealer_share"]).rolling(3, min_periods=1).mean(), -1,
                         sleeve=sleeve))
        out.append(_comp("US", "auction", f"indirect_{sleeve}", f"Indirect share ({sleeve})",
                         _monthly(idx["indirect_share"]).rolling(3, min_periods=1).mean(),
                         sleeve=sleeve))
    # Country-level (all coupon terms) for the all-sleeve score — same as before.
    all_c = b.set_index("auction_date")
    out.append(_comp("US", "auction", "dealer", "Primary-dealer share",
                     _monthly(all_c["dealer_share"]).rolling(3, min_periods=1).mean(), -1))
    out.append(_comp("US", "auction", "indirect", "Indirect share",
                     _monthly(all_c["indirect_share"]).rolling(3, min_periods=1).mean()))
    return out


def _us_allot_auction() -> list[pd.DataFrame]:
    """End-user allotment share (funds + foreign + pension) of coupon auctions by sleeve."""
    al = _read(CACHE / "us_allotments.parquet")
    if al.empty:
        return []
    a = al.copy()
    a["term_years"] = pd.to_numeric(a["term_years"], errors="coerce")
    a = a[a["term_years"] > 2.0].copy()
    a["sleeve"] = a["term_years"].map(_sleeve_of_years)
    a = a.dropna(subset=["sleeve"])
    a["month"] = pd.to_datetime(a["issue_date"]).dt.to_period("M").dt.to_timestamp()
    a["amount_bn"] = pd.to_numeric(a["amount_bn"], errors="coerce")
    out = []
    for sleeve, sub in a.groupby("sleeve"):
        tot = sub.groupby("month")["amount_bn"].sum()
        end = sub[sub["investor_class"].isin(US_END_USER)].groupby("month")["amount_bn"].sum()
        share = (end / tot.replace(0, np.nan)).dropna()
        if share.notna().sum() < Z_MIN:
            continue
        out.append(_comp("US", "auction", f"allot_end_{sleeve}",
                         f"End-user allotment share ({sleeve})",
                         share.rolling(3, min_periods=1).mean(), sleeve=sleeve))
    return out


XB = {
    "DE": ("ecb_bop_liab_govt_lt_debt", "Non-residents, govt LT debt (ECB BoP)"),
    "FR": ("ecb_bop_liab_govt_lt_debt", "Non-residents, govt LT debt (ECB BoP)"),
    "IT": ("ecb_bop_liab_govt_lt_debt", "Non-residents, govt LT debt (ECB BoP)"),
    "UK": ("boe_overseas_gilt_net_purchases", "Overseas gilt purchases (BoE)"),
    "US": ("tic_net_lt_treasuries", "Foreign LT Treasury purchases (TIC)"),
}


def _foreign() -> list[pd.DataFrame]:
    xb = _read(CACHE / "xborder_flows.parquet")
    out = []
    for c, (series, label) in XB.items():
        g = (xb[(xb["country"] == c) & (xb["series"] == series)].set_index("period")["value"]
             if not xb.empty else pd.Series(dtype=float))
        if len(g):
            out.append(_comp(c, "foreign", "nonres", label,
                             _monthly(g, "sum").rolling(3, min_periods=3).sum()))
    wk = _read(CACHE / "jp_flows_weekly.parquet")
    if not wk.empty:
        s = _monthly(wk.set_index("week_end")["l_lt_net"], "sum")
        out.append(_comp("JP", "foreign", "nonres", "Foreigners, JP LT debt (MoF weekly)",
                         s.rolling(3, min_periods=3).sum()))
    out.extend(_jp_jsda_foreign())
    return out


def _jp_jsda_foreign() -> list[pd.DataFrame]:
    js = _read(CACHE / "jsda_investor.parquet")
    if js.empty:
        return []
    j = js[js["investor"].astype(str).eq("foreigners")].copy()
    j["sleeve"] = j["maturity"].map(JSDA_SLEEVE)
    j = j.dropna(subset=["sleeve"])
    j["month"] = pd.to_datetime(j["month"]).dt.to_period("M").dt.to_timestamp()
    out = []
    for sleeve, sub in j.groupby("sleeve"):
        s = sub.groupby("month")["net_purchase_jpy_bn"].sum()
        out.append(_comp("JP", "foreign", f"jsda_{sleeve}",
                         f"JSDA foreigners net ({sleeve})",
                         s.rolling(3, min_periods=3).sum(), sleeve=sleeve))
    return out


def _official() -> list[pd.DataFrame]:
    sup = _read(CACHE / "rv_supply.parquet")
    out = []
    for c in COUNTRIES:
        g = sup[sup["country"] == c].set_index("month").sort_index()
        if len(g) and "official_dv01_usd" in g.columns:
            s = _monthly(g["official_dv01_usd"], "sum").rolling(3, min_periods=3).sum()
            out.append(_comp(c, "official", "official", "Official purchases (DV01)", s))
    return out


def _positioning() -> list[pd.DataFrame]:
    out = []
    cftc = _read(CFTC)
    if not cftc.empty:
        cftc = cftc.reset_index() if "date" not in cftc.columns else cftc
        lvl = _monthly(cftc.set_index("date")["asset_dv01_mn"], "last")
        out.append(_comp("US", "positioning", "asset_mgr",
                         "Asset managers, 3m DV01 change (CFTC)", lvl.diff(3)))
    jpx = _read(CACHE / "jpx_jgb_futures_investor.parquet")
    if not jpx.empty:
        f = jpx[jpx["contract"].astype(str).eq("jgb10y") & jpx["unit"].astype(str).eq("yen")
                & jpx["investor"].astype(str).eq("foreigners")]
        s = _monthly(f.set_index("week_end")["net"], "sum").rolling(3, min_periods=3).sum() / 1e12
        out.append(_comp("JP", "positioning", "foreign_fut",
                         "Foreigners, 10y JGB futures 3m net (JPX)", s))
    return out


def components() -> pd.DataFrame:
    parts = [p for fn in (_auction, _foreign, _official, _positioning) for p in fn() if len(p)]
    if not parts:
        return pd.DataFrame(columns=COLS)
    return pd.concat(parts, ignore_index=True)


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(Z_WINDOW, min_periods=Z_MIN).mean()
    sd = s.rolling(Z_WINDOW, min_periods=Z_MIN).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-4, 4)


def _pillar_name(pillar: str, sleeve: str) -> str:
    if sleeve == "all" or pillar == "positioning":
        return pillar
    return f"{pillar}_{sleeve}"


def _score_name(sleeve: str) -> str:
    return "score" if sleeve == "all" else f"score_{sleeve}"


def build() -> pd.DataFrame:
    """Long table: component rows, pillar rows (`_pillar`), score rows (`_score`).

    Sleeve-specific pillars are named `auction_int`, `foreign_long`, …; all-sleeve keeps
    the original names. Scores are `score`, `score_int`, `score_long`.
    """
    comp = components()
    if comp.empty:
        return comp
    if "sleeve" not in comp.columns:
        comp["sleeve"] = "all"
    end = comp["month"].max()
    grid = pd.date_range(START, end, freq="MS")
    rows = []
    for (c, p, k, sl), g in comp.groupby(["country", "pillar", "component", "sleeve"], sort=False):
        g = g.drop_duplicates("month").set_index("month").sort_index()
        z = _zscore(g["value"]) * int(g["sign"].iloc[0])
        frame = pd.DataFrame({"value": g["value"], "z": z})
        frame["asof"] = frame.index
        frame = frame.reindex(frame.index.union(grid)).ffill(limit=CARRY).reindex(grid)
        frame = frame.dropna(subset=["z"])
        rows.append(frame.assign(
            country=c, pillar=p, component=k, label=g["label"].iloc[0], sleeve=sl,
        ).rename_axis("month").reset_index())
    long = pd.concat(rows, ignore_index=True)

    # Pillar means within each sleeve (positioning stays all-only).
    pil_parts = []
    for (c, p, sl), g in long.groupby(["country", "pillar", "sleeve"], sort=False):
        name = _pillar_name(p, sl)
        agg = g.groupby("month", as_index=False).agg(z=("z", "mean"), asof=("asof", "min"))
        pil_parts.append(agg.assign(
            country=c, pillar=name, component="_pillar", label=name, value=np.nan, sleeve=sl,
        ))
    pil = pd.concat(pil_parts, ignore_index=True) if pil_parts else pd.DataFrame()

    # Wide lookup for score construction.
    score_rows = []
    if not pil.empty:
        wide = pil.pivot_table(index=["country", "month"], columns="pillar", values="z", aggfunc="mean")
        asof = pil.groupby(["country", "month"])["asof"].min()
        for c, m in wide.index:
            row = wide.loc[(c, m)]
            a = asof.loc[(c, m)] if (c, m) in asof.index else m
            for sleeve in SLEEVES:
                parts = []
                auc = _pillar_name("auction", sleeve)
                if auc in row.index and pd.notna(row[auc]):
                    parts.append(float(row[auc]))
                frn = _pillar_name("foreign", sleeve)
                if frn in row.index and pd.notna(row[frn]):
                    parts.append(float(row[frn]))
                elif sleeve != "all" and "foreign" in row.index and pd.notna(row["foreign"]):
                    parts.append(float(row["foreign"]))
                if "official" in row.index and pd.notna(row["official"]):
                    parts.append(float(row["official"]))
                if not parts:
                    continue
                score_rows.append({
                    "country": c, "month": m, "pillar": _score_name(sleeve),
                    "component": "_score", "label": f"Flow demand score ({sleeve})",
                    "value": float(len(parts)), "z": float(np.mean(parts)), "asof": a,
                    "sleeve": sleeve,
                })
    sc = pd.DataFrame(score_rows)

    # Attach sleeve on component rows' display pillar (keep base pillar name on comps).
    long = long.copy()
    out = pd.concat([long, pil, sc], ignore_index=True)
    if "sleeve" not in out.columns:
        out["sleeve"] = "all"
    return out[["country", "month", "pillar", "component", "label", "value", "z", "asof", "sleeve"]]


def write_gate(out: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matched-price f3l2 ship flags → cache/flow_score_gate.json (score_long usually fails)."""
    from issuance.experiments.flow_score_test import gate
    g = gate()
    path = CACHE / "flow_score_gate.json"
    g.to_json(path, orient="records", date_format="iso")
    return g


def write() -> pd.DataFrame:
    out = build()
    out.to_parquet(CACHE / "flow_score.parquet", index=False)
    try:
        g = write_gate(out)
        print("flow_score gate:", g.set_index("signal")["ship"].to_dict())
    except Exception as exc:
        print(f"  flow_score gate skipped ({exc})")
    return out


if __name__ == "__main__":
    o = write()
    last = o[o["component"].isin(["_pillar", "_score"])]
    last = last[last["month"] == last["month"].max()]
    print(last.pivot_table(index="country", columns="pillar", values="z").round(2))
