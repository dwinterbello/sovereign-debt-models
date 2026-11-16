"""Overnight money-market panel and a daily reserves estimate.

Rates and volumes (NY Fed reference rates API):
    SOFR, TGCR, BGCR (secured), EFFR, OBFR (unsecured) — rate, 1st/99th
    percentile, volume ($bn).
Balance sheet (FRED, H.4.1, $mn):
    WRBWFRBL reserves (Wednesday level), WALCL total assets, WTREGEN TGA,
    WLRRAL RRP (all Wednesday levels).
Facilities: RRPONTSYD (ON RRP take-up, $bn daily), RPONTSYD (repo ops incl.
    SRF, $bn daily), IORB, ON RRP award rate, target range.

Daily reserves are not published. Between Wednesdays

    R_t ≈ R_wed + ΔRepo ops − ΔTGA − ΔON RRP            (other assets/liabs held flat)

with ΔTGA from the DTS close. `reserves_daily` returns the estimate and the
check against the next Wednesday's published level.

  PYTHONPATH=. .venv/bin/python us/fiscal/money_markets.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tga_deficit import fetch_fred, fetch_tga_mn

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cache"
DERIVED = HERE / "derived"

NYFED = "https://markets.newyorkfed.org/api/rates/{kind}/{product}/search.json"
PRODUCTS = {"sofr": "secured", "tgcr": "secured", "bgcr": "secured", "effr": "unsecured", "obfr": "unsecured"}
UA = {"User-Agent": "us-fiscal/money-markets (research; local)"}
START = "2018-04-02"
FRED_WEEKLY = {"WRBWFRBL": "reserves_wed_mn", "WALCL": "fed_assets_wed_mn", "WTREGEN": "tga_wed_mn",
               "WLRRAL": "rrp_wed_mn"}
FRED_DAILY = {"RRPONTSYD": "rrp_bn", "RPONTSYD": "repo_ops_bn", "IORB": "iorb", "RRPONTSYAWARD": "rrp_rate",
              "DFEDTARU": "target_hi", "DFEDTARL": "target_lo"}


def fetch_nyfed(product: str, *, refresh: bool = False) -> pd.DataFrame:
    """Rate, percentiles and volume for one NY Fed reference rate."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"nyfed_full_{product}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")
    try:
        r = requests.get(
            NYFED.format(kind=PRODUCTS[product], product=product),
            params={"startDate": START, "endDate": pd.Timestamp.today().strftime("%Y-%m-%d")},
            headers=UA, timeout=90,
        )
        r.raise_for_status()
        rows = r.json().get("refRates") or []
    except Exception as exc:  # noqa: BLE001
        if path.exists():
            print(f"   NY Fed {product}: {exc}; using cache")
            return pd.read_csv(path, parse_dates=["date"]).set_index("date")
        raise
    def num(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    rec = []
    for x in rows:
        if np.isnan(num(x.get("percentRate"))):
            continue
        rec.append({
            "date": pd.Timestamp(x["effectiveDate"]),
            f"{product}": num(x["percentRate"]),
            f"{product}_p1": num(x.get("percentPercentile1")),
            f"{product}_p99": num(x.get("percentPercentile99")),
            f"{product}_vol_bn": num(x.get("volumeInBillions")),
        })
    df = pd.DataFrame(rec).drop_duplicates("date").set_index("date").sort_index()
    df.reset_index().to_csv(path, index=False)
    return df


def reserves_daily(weekly: pd.DataFrame, tga_bn: pd.Series, rrp_bn: pd.Series, repo_bn: pd.Series,
                   index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily reserves ($bn) rolled from the last published Wednesday; plus next-Wednesday check."""
    wed = (weekly["reserves_wed_mn"] / 1000.0).dropna()
    tga = tga_bn.reindex(index).ffill()
    rrp = rrp_bn.reindex(index).ffill().fillna(0.0)
    repo = repo_bn.reindex(index).fillna(0.0)
    anchor_date = pd.Series(wed.index, index=wed.index).reindex(index, method="ffill")
    # A holiday Wednesday's H.4.1 level is the prior business day's close.
    pos = index.searchsorted(pd.DatetimeIndex(anchor_date.fillna(index[0])), side="right") - 1
    anchor_bd = pd.DatetimeIndex(np.where(anchor_date.notna(), index[np.clip(pos, 0, None)], pd.NaT))
    out = pd.DataFrame(index=index)
    out["anchor"] = anchor_date
    out["reserves_anchor_bn"] = wed.reindex(index, method="ffill")
    t0 = tga.reindex(anchor_bd).to_numpy()
    r0 = rrp.reindex(anchor_bd).to_numpy()
    p0 = repo.reindex(anchor_bd).to_numpy()
    out["reserves_est_bn"] = (out["reserves_anchor_bn"] - (tga.to_numpy() - t0) - (rrp.to_numpy() - r0)
                              + (repo.to_numpy() - p0))
    out.loc[out.index.isin(wed.index), "reserves_est_bn"] = wed.reindex(out.index[out.index.isin(wed.index)]).to_numpy()
    # Out-of-sample check: estimate made from the previous Wednesday vs the published one.
    chk = []
    wd = list(wed.index)
    def _bd(d: pd.Timestamp) -> pd.Timestamp | None:
        i = index.searchsorted(d, side="right") - 1
        return index[i] if i >= 0 else None

    for a, b in zip(wd[:-1], wd[1:]):
        ab, bb = _bd(a), _bd(b)
        if ab is None or bb is None or ab == bb:
            continue
        est = wed[a] - (tga[bb] - tga[ab]) - (rrp[bb] - rrp[ab]) + (repo[bb] - repo[ab])
        chk.append((b, est, wed[b]))
    out.attrs["wed_check"] = pd.DataFrame(chk, columns=["date", "est", "actual"]).set_index("date")
    return out


def build_panel(*, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates = pd.concat([fetch_nyfed(p, refresh=refresh) for p in PRODUCTS], axis=1).sort_index()
    idx = rates.index
    df = rates.copy()
    for code, name in FRED_DAILY.items():
        try:
            df[name] = fetch_fred(code, refresh=refresh).reindex(idx).ffill()
        except Exception as exc:  # noqa: BLE001
            print(f"   FRED {code}: {exc}")
            df[name] = np.nan
    try:  # IORB starts 2021-07-29; IOER before
        df["iorb"] = df["iorb"].combine_first(fetch_fred("IOER", refresh=refresh).reindex(idx).ffill())
    except Exception as exc:  # noqa: BLE001
        print(f"   FRED IOER: {exc}")
    weekly = pd.DataFrame({name: fetch_fred(code, refresh=refresh) for code, name in FRED_WEEKLY.items()})
    tga_bn = fetch_tga_mn(refresh=refresh) / 1000.0
    df["tga_bn"] = tga_bn.reindex(idx).ffill()
    res = reserves_daily(weekly, tga_bn, df["rrp_bn"], df["repo_ops_bn"], idx)
    df["reserves_bn"] = res["reserves_est_bn"]
    df["fed_assets_bn"] = (weekly["fed_assets_wed_mn"] / 1000.0).reindex(idx, method="ffill")
    for a, b in (("sofr", "iorb"), ("tgcr", "iorb"), ("effr", "iorb"), ("sofr", "effr"), ("sofr", "tgcr"),
                 ("bgcr", "tgcr"), ("sofr", "rrp_rate")):
        df[f"{a}_{b}_bp"] = 100.0 * (df[a] - df[b])
    df["sofr_p99_iorb_bp"] = 100.0 * (df["sofr_p99"] - df["iorb"])
    return df, res.attrs["wed_check"]


def run(*, refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    df, chk = build_panel(refresh=refresh)
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.rename_axis("date").to_parquet(DERIVED / "money_markets_daily.parquet")
    chk.to_csv(DERIVED / "reserves_wed_check.csv")
    if verbose:
        last = df.dropna(subset=["sofr"]).iloc[-1]
        err = (chk["est"] - chk["actual"]).loc["2023":]
        print(f"Money markets {last.name.date()}: SOFR−IORB {last['sofr_iorb_bp']:.1f}bp  "
              f"SOFR vol ${last['sofr_vol_bn']:.0f}bn  reserves≈${last['reserves_bn']:.0f}bn  "
              f"RRP ${last['rrp_bn']:.0f}bn  | Wed roll-forward MAE ${err.abs().mean():.0f}bn")
    return df


if __name__ == "__main__":
    run(refresh=True)
