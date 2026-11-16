"""Fiscal path → supply: monthly financing by settlement month, the handoff to `issuance/`.

Direction of the link: coupons flow issuance → fiscal (the year-ago template and the
announced refunding sizes are inputs to `fiscal_path`); bills, Fed purchases and
buybacks flow fiscal → issuance (`issuance/fiscal_bridge.py` reads this file).

All flows are cash, $bn, summed by settlement month from `fiscal_path.parquet`:

  ΔTGA = −deficit + net coupons + net bills (announced + implied) + drift
  drift = SOMA rolls (cash) + Fed-bought bills maturing + buybacks + nonmarketables

Private net bills (supply the public must absorb) = ladder net bills + Fed-held bills
that mature and are rolled as add-ons (snapshot holdings and later purchases) − Fed
secondary bill purchases. Stocks are par: bills accrete their discount, so the par stock
grows faster than net cash by ≈ stock × bill yield / 252 per business day.

Stocks at the origin come from `auctions_tidy` (outstanding total accepted, buybacks
netted by CUSIP) less the SOMA snapshot. TIPS are carried at par (no accretion).
TGA_TARGETS end at the last configured quarter; after it the path holds that level.

  PYTHONPATH=. .venv/bin/python us/fiscal/supply_bridge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import auction_ladder as al  # noqa: E402
import financing  # noqa: E402
import tga_forecast as tf  # noqa: E402

DERIVED = HERE / "derived"
OUT = DERIVED / "supply_bridge.parquet"
META = DERIVED / "supply_bridge.json"
IDENTITY_TOL_BN = 1.0
BILL_DAYS_YEAR = 252.0


def origin_stocks(origin: pd.Timestamp, tidy: pd.DataFrame | None = None) -> dict:
    """Marketable par outstanding at `origin` ($bn), total and private (ex SOMA)."""
    t = pd.read_parquet(DERIVED / "auctions_tidy.parquet") if tidy is None else tidy
    a = t[(t["status"] == "awarded") & (t["issue_date"] <= origin) & (t["maturity_date"] > origin)].copy()
    a["par_bn"] = pd.to_numeric(a["total_accepted"], errors="coerce").fillna(0.0) / 1e9
    by = a.groupby("cusip").agg(cls=("cls", "first"), par_bn=("par_bn", "sum"))
    try:
        bb = al.fetch_buyback_details()
        bb = bb[pd.to_datetime(bb["operation_date"]) <= origin]
        bbp = al._num(bb["par_amt_accepted"].astype(str)).groupby(bb["cusip_nbr"]).sum() / 1e9
        by["par_bn"] = by["par_bn"] - bbp.reindex(by.index).fillna(0.0)
    except Exception:  # noqa: BLE001  (buyback history unavailable: stock overstated by ≤ cumulative buybacks)
        pass
    bills = by["cls"].isin(tf.BILL_CLS)
    pub = a[a["cls"].isin(tf.BILL_CLS)]["public_par_bn"]
    out = {"bills_total_bn": float(by.loc[bills, "par_bn"].sum()),
           "bills_public_issued_bn": float(pd.to_numeric(pub, errors="coerce").sum()),
           "coupons_total_bn": float(by.loc[~bills, "par_bn"].sum())}
    snap = financing.snapshot_for(origin)
    soma_b = soma_c = 0.0
    if snap is not None:
        s = snap[1]
        soma_b = float(s.loc[s["type"] == "Bills", "par_bn"].sum())
        soma_c = float(s.loc[s["type"] != "Bills", "par_bn"].sum())
        out["soma_asof"] = str(snap[0].date())
    out.update({"soma_bills_bn": soma_b, "soma_coupons_bn": soma_c,
                "bills_private_bn": out["bills_total_bn"] - soma_b,
                "coupons_private_bn": out["coupons_total_bn"] - soma_c})
    return out


def bill_yield(origin: pd.Timestamp, tidy: pd.DataFrame | None = None) -> float:
    """Latest 13-week bill investment rate (decimal) awarded by `origin`."""
    t = pd.read_parquet(DERIVED / "auctions_tidy.parquet") if tidy is None else tidy
    b = t[(t["cls"] == "bill") & (t["status"] == "awarded") & (t["issue_date"] <= origin)
          & (t["security_term"].astype(str) == "13-Week")]
    r = pd.to_numeric(b.sort_values("issue_date")["high_investment_rate"], errors="coerce").dropna()
    return float(r.iloc[-1]) / 100.0 if len(r) else 0.04


def monthly(p: pd.DataFrame, tga0: float, stocks: dict, y: float) -> pd.DataFrame:
    """One scenario's daily path → settlement-month bridge."""
    d = p.copy()
    d["soma_bill_rolls_bn"] = d["soma_bills_bn"] + d["soma_purchases_bn"]
    d["soma_coupon_rolls_bn"] = d["soma_bn"] - d["soma_bills_bn"]
    if "fed_bill_purchases_bn" not in d:
        d["fed_bill_purchases_bn"] = 0.0
    # Daily par stocks: bills accrete their discount; coupon stock moves with net coupons,
    # Fed-held coupon maturities rolled as add-ons, and buybacks (cash ≈ par for coupons).
    priv_b = np.empty(len(d))
    disc = np.empty(len(d))
    s = stocks["bills_private_bn"]
    for i, f in enumerate(d["private_net_bills_bn"].to_numpy()):
        disc[i] = s * y / BILL_DAYS_YEAR
        s += f + disc[i]
        priv_b[i] = s
    d["bill_discount_bn"] = disc
    d["private_bill_stock_bn"] = priv_b
    # Public-issued stock (what gross public bill auctions roll): ladder net bills + discount.
    pub_b = np.empty(len(d))
    s = stocks["bills_public_issued_bn"]
    for i, f in enumerate(d["net_bills_total_bn"].to_numpy()):
        s += f + s * y / BILL_DAYS_YEAR
        pub_b[i] = s
    d["public_bill_stock_bn"] = pub_b
    d["private_coupon_stock_bn"] = stocks["coupons_private_bn"] + (
        d["net_coupons"] + d["soma_coupon_rolls_bn"] + d["buyback_bn"]).cumsum()
    m = d.groupby(d.index.to_period("M"))
    out = pd.DataFrame({
        "n_bdays": m.size(),
        "deficit_bn": -m["op_flow_bn"].sum(),
        "interest_bn": m["interest_bn"].sum(),
        "interest_cash_bn": m["interest_cash_bn"].sum() if "interest_cash_bn" in d.columns else m["interest_bn"].sum(),
        "coupon_iss_cash_bn": m["coupon_iss"].sum(),
        "coupon_red_bn": m["coupon_red"].sum(),
        "net_coupons_bn": m["net_coupons"].sum(),
        "announced_net_bills_bn": m["net_bills"].sum(),
        "implied_bills_bn": m["implied_bills_bn"].sum(),
        "net_bills_total_bn": m["net_bills_total_bn"].sum(),
        "soma_rollover_bn": m["soma_bn"].sum(),
        "soma_bill_rolls_bn": m["soma_bill_rolls_bn"].sum(),
        "soma_purchases_maturing_bn": m["soma_purchases_bn"].sum(),
        "fed_bill_purchases_bn": m["fed_bill_purchases_bn"].sum(),
        "private_net_bills_bn": m["private_net_bills_bn"].sum(),
        "buyback_cash_bn": -m["buyback_bn"].sum(),
        "nonmkt_bn": m["nonmkt_bn"].sum(),
        "drift_bn": m["drift_bn"].sum(),
        "tga_end_bn": m["tga_bn"].last(),
        "bill_discount_bn": m["bill_discount_bn"].sum(),
        "private_bill_stock_bn": m["private_bill_stock_bn"].last(),
        "public_bill_stock_bn": m["public_bill_stock_bn"].last(),
        "private_coupon_stock_bn": m["private_coupon_stock_bn"].last(),
        "known_share": m["known"].mean(),
    })
    out["d_tga_bn"] = out["tga_end_bn"].diff()
    out.iloc[0, out.columns.get_loc("d_tga_bn")] = out["tga_end_bn"].iloc[0] - tga0
    out["identity_resid_bn"] = out["d_tga_bn"] - (-out["deficit_bn"] + out["net_coupons_bn"]
                                                  + out["net_bills_total_bn"] + out["drift_bn"])
    out["bill_share_stock"] = out["private_bill_stock_bn"] / (out["private_bill_stock_bn"]
                                                              + out["private_coupon_stock_bn"])
    tot = out["net_bills_total_bn"] + out["net_coupons_bn"]
    out["bill_share_flow"] = out["net_bills_total_bn"] / tot.where(tot.abs() > 1.0)
    out.index = out.index.to_timestamp()
    out.index.name = "month"
    return out


def gap_fy(bridge: pd.DataFrame, start: str, end: str) -> dict:
    """Net bills that fund a window with coupon sizes held (Coupons Constant)."""
    b = bridge.loc["unchanged"]
    w = b[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
    return {"start": start, "end": end, "months": int(len(w)), "complete": bool(len(w) == 24),
            "net_bills_bn": float(w["net_bills_total_bn"].sum()),
            "private_net_bills_bn": float(w["private_net_bills_bn"].sum()),
            "deficit_bn": float(w["deficit_bn"].sum()), "net_coupons_bn": float(w["net_coupons_bn"].sum())}


def run(verbose: bool = True) -> pd.DataFrame:
    fp = pd.read_parquet(DERIVED / "fiscal_path.parquet")
    info = json.loads((DERIVED / "fiscal_path.json").read_text())
    origin = pd.Timestamp(info["origin"])
    tga0 = float(info["tga_origin_bn"])
    tidy = pd.read_parquet(DERIVED / "auctions_tidy.parquet")
    stocks = origin_stocks(origin, tidy)
    y = bill_yield(origin, tidy)
    parts = {s: monthly(fp.loc[s], tga0, stocks, y) for s in fp.index.get_level_values(0).unique()}
    bridge = pd.concat(parts, names=["scenario"])
    worst = float(bridge["identity_resid_bn"].abs().max())
    if worst > IDENTITY_TOL_BN:
        raise ValueError(f"supply bridge identity off by {worst:.2f}bn (tolerance {IDENTITY_TOL_BN})")
    bridge.to_parquet(OUT)
    targets = sorted(tf.TGA_TARGETS)
    meta = {"origin": str(origin.date()), "end": info["end"], "scenario_start": info["scenario_start"],
            "tga_targets": tf.TGA_TARGETS, "tga_targets_end": targets[-1] if targets else None,
            "after_targets": "TGA held at the last configured target",
            "bill_yield": y, "origin_stocks_bn": stocks, "identity_max_resid_bn": worst,
            "gap_fy27_28": gap_fy(bridge, "2026-10-01", "2028-09-01"),
            "private_bill_share_stock_origin": stocks["bills_private_bn"]
            / (stocks["bills_private_bn"] + stocks["coupons_private_bn"])}
    META.write_text(json.dumps(meta, indent=1, default=float))
    if verbose:
        u = bridge.loc["unchanged"]
        cols = ["deficit_bn", "net_coupons_bn", "net_bills_total_bn", "fed_bill_purchases_bn",
                "soma_bill_rolls_bn", "private_net_bills_bn", "buyback_cash_bn", "tga_end_bn",
                "private_bill_stock_bn", "bill_share_stock"]
        print(u[cols].round(1).assign(bill_share_stock=u["bill_share_stock"].round(4)).to_string())
        print("FY27-28 gap (coupons constant):", {k: round(v, 1) if isinstance(v, float) else v
                                                  for k, v in meta["gap_fy27_28"].items()})
    return bridge


if __name__ == "__main__":
    run()
