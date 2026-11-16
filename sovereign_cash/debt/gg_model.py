"""General-government deficit path for the main DM countries.

Object: GG balance % GDP (negative = deficit). Not state/central cash.

Identity used every year:
  primary = balance + interest          # both % GDP
  cycle   = ε × (real growth − g*)      # level contribution to the primary
  structural_primary = primary − cycle
  balance = structural_primary + cycle − interest

History: AMECO balance / interest_gross where present, else WEO.
Growth conditioner: WEO real_growth (RESF-style external, not fitted).
Interest forward (live): WEO interest_pct path (same conditioner family).
Structural primary forward: hold last complete outturn (no free consolidation) —
  that is the "no expenditure effort" case that sits next to Gov 5.0% / OE 5.7%.

PRE gate: origins 2012..(last_outturn−3), horizons 1..3, MAE on balance vs realised
AMECO/WEO outturns. Peer = hold last outturn balance (no WEO look-ahead). Ships if
mean PRE MAE ≤ hold MAE.

Live growth/interest conditioners use today's WEO — same information set as the debt
board's WEO columns; AMECO remains authority until this module is wired.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.gg_model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "debt"
DERIVED = HERE.parent / "derived"

# Main board countries. JP/SE/NO can join once PRE clears.
COUNTRIES = ("us", "uk", "de", "fr", "it")

# Budget semi-elasticity to the output gap (OECD/EC ballpark).
EPS = {
    "us": 0.50,
    "uk": 0.50,
    "de": 0.55,
    "fr": 0.55,
    "it": 0.50,
}
# Potential real growth % (medium-term). Used only for the cycle split.
GSTAR = {
    "us": 1.8,
    "uk": 1.5,
    "de": 1.1,
    "fr": 1.2,
    "it": 0.8,
}

PRE_START = 2012
H_MAX = 3
REL_EDGE = 0.0  # must not lose to hold-last on PRE mean MAE


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ameco = pd.read_parquet(DATA / "ameco.parquet")
    weo = pd.read_parquet(DATA / "weo.parquet")
    return ameco, weo


def _pivot(df: pd.DataFrame, cc: str) -> pd.DataFrame:
    d = df[df["country"] == cc]
    return d.pivot_table(index="year", columns="var", values="value").sort_index()


def history(cc: str, ameco: pd.DataFrame, weo: pd.DataFrame) -> pd.DataFrame:
    """Annual GG panel through the last overlapping outturn year."""
    A, W = _pivot(ameco, cc), _pivot(weo, cc)
    # Prefer AMECO balance/interest for EU+UK when present; US is WEO/CBO-adjacent via WEO.
    bal = A["balance_pct"] if "balance_pct" in A.columns and A["balance_pct"].notna().sum() > 10 else W["balance_pct"]
    if "interest_gross_pct" in A.columns and A["interest_gross_pct"].notna().sum() > 10:
        inter = A["interest_gross_pct"]
    else:
        inter = W["interest_pct"]
    g = W["real_growth"]
    rev = A["revenue_pct"] if "revenue_pct" in A.columns else W["revenue_pct"]
    out = pd.DataFrame({
        "balance": bal,
        "interest": inter,
        "growth": g,
        "revenue": rev,
    }).dropna(subset=["balance", "interest", "growth"])
    out["primary"] = out["balance"] + out["interest"]
    eps, gstar = EPS[cc], GSTAR[cc]
    out["cycle"] = eps * (out["growth"] - gstar)
    out["structural_primary"] = out["primary"] - out["cycle"]
    out["deficit"] = -out["balance"]
    use_ameco_bal = "balance_pct" in A.columns and A["balance_pct"].notna().sum() > 10
    use_ameco_i = "interest_gross_pct" in A.columns and A["interest_gross_pct"].notna().sum() > 10
    out.attrs["balance_source"] = "ameco" if use_ameco_bal else "weo"
    out.attrs["interest_source"] = "ameco" if use_ameco_i else "weo"
    return out


def last_outturn_year(hist: pd.DataFrame, weo: pd.DataFrame, cc: str) -> int:
    """Last year treated as outturn: min(hist end, WEO year before the forecast block).

    WEO marks nothing; AMECO spring vintage usually has Y-1 as near-final. Use hist max
    where balance is present, capped at calendar year - 1.
    """
    y = int(hist.dropna(subset=["balance"]).index.max())
    return min(y, pd.Timestamp.today().year - 1)


def forecast(cc: str, hist: pd.DataFrame, weo_cc: pd.DataFrame,
             origin_year: int, h: int = H_MAX) -> pd.DataFrame:
    """h-year GG path from origin_year (last known outturn)."""
    if origin_year not in hist.index:
        return pd.DataFrame()
    row0 = hist.loc[origin_year]
    eps, gstar = EPS[cc], GSTAR[cc]
    sp = float(row0["structural_primary"])
    rows = []
    prev_i = float(row0["interest"])
    for k in range(1, h + 1):
        y = origin_year + k
        # Growth conditioner: WEO if present, else g*
        if y in weo_cc.index and pd.notna(weo_cc.loc[y].get("real_growth", np.nan)):
            g = float(weo_cc.loc[y, "real_growth"])
            g_src = "weo"
        else:
            g = gstar
            g_src = "gstar"
        # Interest conditioner: WEO path when present, else hold last
        if y in weo_cc.index and pd.notna(weo_cc.loc[y].get("interest_pct", np.nan)):
            inter = float(weo_cc.loc[y, "interest_pct"])
            i_src = "weo"
        else:
            inter = prev_i
            i_src = "hold"
        cycle = eps * (g - gstar)
        primary = sp + cycle  # structural held
        bal = primary - inter
        rows.append({
            "year": y,
            "h": k,
            "balance": bal,
            "deficit": -bal,
            "interest": inter,
            "growth": g,
            "cycle": cycle,
            "structural_primary": sp,
            "primary": primary,
            "growth_source": g_src,
            "interest_source": i_src,
        })
        prev_i = inter
    return pd.DataFrame(rows)


def backtest(cc: str, ameco: pd.DataFrame, weo: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward on realised history only.

    Growth/interest conditioners in the backtest use *realised* WEO/AMECO values at
    target year (upper bound on skill). Peer = hold origin balance.
    """
    hist = history(cc, ameco, weo)
    W = _pivot(weo, cc)
    last = last_outturn_year(hist, weo, cc)
    # Restrict hist to outturn years only (drop AMECO forward rows past last)
    hist = hist.loc[hist.index <= last]
    rows = []
    for origin in range(PRE_START, last - H_MAX + 1):
        if origin not in hist.index:
            continue
        hold = float(hist.loc[origin, "balance"])
        # Realised growth/interest at target years (oracle conditioner — documented).
        fc = forecast(cc, hist.loc[:origin], W, origin, H_MAX)
        for _, r in fc.iterrows():
            y = int(r["year"])
            if y not in hist.index or y > last:
                continue
            actual = float(hist.loc[y, "balance"])
            rows.append({
                "country": cc,
                "origin": origin,
                "year": y,
                "h": int(r["h"]),
                "forecast": float(r["balance"]),
                "actual": actual,
                "err": float(r["balance"]) - actual,
                "hold_fc": hold,
                "hold_err": hold - actual,
            })
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"mae": None, "hold_mae": None, "n": 0, "pass": False}
    mae = float(bt["err"].abs().mean())
    hold_mae = float(bt["hold_err"].abs().mean())
    ok = mae <= hold_mae * (1.0 + REL_EDGE)
    by_h = {
        int(h): {
            "mae": float(g["err"].abs().mean()),
            "hold_mae": float(g["hold_err"].abs().mean()),
            "n": int(len(g)),
        }
        for h, g in bt.groupby("h")
    }
    return {"mae": mae, "hold_mae": hold_mae, "n": int(len(bt)), "pass": ok, "by_h": by_h}


def live_path(cc: str, ameco: pd.DataFrame, weo: pd.DataFrame,
              h: int = 5) -> tuple[pd.DataFrame, dict]:
    hist = history(cc, ameco, weo)
    W = _pivot(weo, cc)
    origin = last_outturn_year(hist, weo, cc)
    hist = hist.loc[hist.index <= origin]
    fc = forecast(cc, hist, W, origin, h)
    meta = {
        "country": cc,
        "origin_year": origin,
        "eps": EPS[cc],
        "gstar": GSTAR[cc],
        "balance_source": hist.attrs.get("balance_source"),
        "interest_source_hist": hist.attrs.get("interest_source"),
        "rule": "structural_primary held; cycle=ε(g−g*); interest/growth from WEO when present",
    }
    # attach last outturn deficit for the note
    meta["outturn_deficit"] = float(hist.loc[origin, "deficit"])
    return fc, meta


def run(write: bool = True) -> dict:
    ameco, weo = _load()
    DERIVED.mkdir(parents=True, exist_ok=True)
    live_rows, bt_rows, scores = [], [], {}
    for cc in COUNTRIES:
        bt = backtest(cc, ameco, weo)
        sc = score(bt)
        scores[cc] = sc
        bt = bt.assign(country=cc)
        bt_rows.append(bt)
        fc, meta = live_path(cc, ameco, weo, h=5)
        fc = fc.assign(country=cc, **{k: meta[k] for k in ("origin_year", "eps", "gstar")})
        live_rows.append(fc)
        scores[cc]["origin_year"] = meta["origin_year"]
        scores[cc]["outturn_deficit"] = meta["outturn_deficit"]
        scores[cc]["live"] = {
            int(r.year): round(float(r.deficit), 2) for _, r in fc.iterrows()
        }
        # AMECO/WEO peer levels for the same years
        hist = history(cc, ameco, weo)
        W = _pivot(weo, cc)
        A = _pivot(ameco, cc)
        peer = {}
        for y in fc["year"]:
            y = int(y)
            peer[y] = {
                "weo": round(float(-W.loc[y, "balance_pct"]), 2)
                if y in W.index and pd.notna(W.loc[y].get("balance_pct", np.nan)) else None,
                "ameco": round(float(-A.loc[y, "balance_pct"]), 2)
                if "balance_pct" in A.columns and y in A.index
                and pd.notna(A.loc[y].get("balance_pct", np.nan)) else None,
            }
        scores[cc]["peer_deficit"] = peer

    live = pd.concat(live_rows, ignore_index=True) if live_rows else pd.DataFrame()
    bt_all = pd.concat(bt_rows, ignore_index=True) if bt_rows else pd.DataFrame()
    summary = {
        "asof": str(pd.Timestamp.today().date()),
        "rule": "GG balance = held structural primary + ε(g−g*) − interest; g & interest from WEO",
        "countries": scores,
        "ship": {cc: bool(scores[cc]["pass"]) for cc in COUNTRIES},
        "note": (
            "PRE pass = mean |err| ≤ hold-last balance. Backtest growth/interest use "
            "realised conditioners (upper bound). Live path is our GG view; AMECO/CBO/OBR "
            "remain debt-board authority until ship flags are wired in build.project."
        ),
    }
    if write:
        live.to_parquet(DERIVED / "gg_forecast.parquet", index=False)
        bt_all.to_parquet(DERIVED / "gg_backtest.parquet", index=False)
        (DERIVED / "gg_model_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    s = run(write=True)
    print(json.dumps(s, indent=2))
    print("wrote", DERIVED / "gg_forecast.parquet")


if __name__ == "__main__":
    main()
