"""Supply-demand scenario tables for the debt & supply tab.

Assembles next-12m and path shocks from existing derived outputs
(``debt_snapshot``, ``debt_reinvestment``, ``debt_repricing``),
``investor.tracking``, and ``long_end.drifts``. Does not re-run country
cash engines or touch ``uk/de/fr/.../model.py``.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.sensitivities
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DERIVED = HERE.parent / "derived"
PEERS = ("us", "uk", "de", "fr", "it", "se", "no")
NAME = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy",
        "se": "Sweden", "no": "Norway"}


def _snap() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "debt_snapshot.parquet").set_index("cc")


def _rv() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "debt_reinvestment.parquet")


def _rp() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "debt_repricing.parquet")


def _private_dv01_12m() -> dict[str, float]:
    """USD mn/bp of private flow over the next 12 forecast months (issuance float forward)."""
    from sovereign_cash.investor import private_dv01

    dv = private_dv01()
    if dv.empty:
        return {}
    out = {}
    for cc, g in dv.groupby("cc"):
        g = g.sort_values("month").head(12)
        out[str(cc)] = float(g["flow_dv01"].sum())
    return out


def next12(snap: pd.DataFrame | None = None, rv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Next-12m private supply, DV01, rollovers, bill share, CB flow, absorption haircuts."""
    snap = _snap() if snap is None else snap
    rv = _rv() if rv is None else rv
    dv01 = _private_dv01_12m()
    rows = []
    for cc in PEERS:
        if cc not in snap.index:
            continue
        r = snap.loc[cc]
        net = float(r["net_issuance_fwd"])
        cb = float(r["fw_cb_net"]) if np.isfinite(r["fw_cb_net"]) else 0.0
        # US: SOMA rolls at auction → private net ≈ gross net. Else CB runoff adds to private.
        priv = net if cc == "us" else net - cb
        gdp = float(r["gdp_bn"])
        fx = float(r["fx"])
        rein = rv[rv["cc"] == cc].set_index("reinvest")
        cb01 = float(r["cb_01_bn"]) if np.isfinite(r["cb_01_bn"]) else 0.0
        # US SOMA-stop: coupon holdings maturing in 12m hit the private market.
        soma_stop = priv + cb01 if cc == "us" else priv
        rows.append({
            "cc": cc,
            "window": r["fw_window"],
            "private_net_bn": priv,
            "private_net_pct_gdp": 100.0 * priv / gdp,
            "private_net_usd_bn": priv * fx,
            "private_dv01_usd_mn": dv01.get(cc, np.nan),
            "rollovers_pct_gdp": 100.0 * float(r["rollovers_bn"]) / gdp,
            "bill_share_fwd": float(r["bill_share_fwd"]) if np.isfinite(r["bill_share_fwd"]) else np.nan,
            "bill_share_norm": float(r["bill_share_norm"]) if np.isfinite(r["bill_share_norm"]) else np.nan,
            "cb_net_pct_gdp": 100.0 * cb / gdp,
            "cb01_pct_gdp": 100.0 * cb01 / gdp,
            "soma_stop_priv_pct_gdp": 100.0 * soma_stop / gdp,
            "reinvest_100_pct_gdp": float(rein.at[1.0, "new_money_pct_gdp"]) if 1.0 in rein.index else np.nan,
            "reinvest_95_pct_gdp": float(rein.at[0.95, "new_money_pct_gdp"]) if 0.95 in rein.index else np.nan,
            "reinvest_90_pct_gdp": float(rein.at[0.9, "new_money_pct_gdp"]) if 0.9 in rein.index else np.nan,
            "reinvest_90_vs_buy": float(rein.at[0.9, "vs_private_buying_last_year"]) if 0.9 in rein.index else np.nan,
        })
    return pd.DataFrame(rows).set_index("cc")


def path2029(rp: pd.DataFrame | None = None) -> pd.DataFrame:
    """2029 path: baseline vs borrowed +100bp (extra net) vs recession (debt / rollovers)."""
    rp = _rp() if rp is None else rp
    yl = int(rp["year"].max())
    rows = []
    for cc in PEERS:
        d = rp[(rp["cc"] == cc) & (rp["year"] == yl)]
        if d.empty:
            continue
        g = d.set_index(["scenario", "case"])

        def row(sc, case):
            r = g.loc[(sc, case)]
            return {k: float(r[k]) for k in (
                "debt_pct_gdp", "net_issuance_pct_gdp", "rollovers_pct_gdp",
                "net_issuance_bn", "rollovers_bn", "gdp_bn")}

        base = row("baseline", "current")
        bor = row("baseline", "shock_borrowed")
        rec = row("recession", "current")
        rows.append({
            "cc": cc, "year": yl,
            "debt_pct_gdp": base["debt_pct_gdp"],
            "net_pct_gdp": base["net_issuance_pct_gdp"],
            "roll_pct_gdp": base["rollovers_pct_gdp"],
            "d_net_borrow100_pct_gdp": bor["net_issuance_pct_gdp"] - base["net_issuance_pct_gdp"],
            "d_debt_recession_pp": rec["debt_pct_gdp"] - base["debt_pct_gdp"],
            "d_roll_recession_pp": rec["rollovers_pct_gdp"] - base["rollovers_pct_gdp"],
            "d_net_recession_pct_gdp": rec["net_issuance_pct_gdp"] - base["net_issuance_pct_gdp"],
        })
    return pd.DataFrame(rows).set_index("cc")


def curve_drift() -> pd.DataFrame:
    """15y+ private face vs mean → implied next-12m Δ10s30s (pooled long-end slope)."""
    from sovereign_cash.debt import long_end as LE

    d = LE.drifts()
    T = d["T"]
    slope, tstat = d["R"]["long"]["pooled"][0], d["R"]["long"]["pooled"][1]
    # long_end covers G6 (+ eur aggregate); SE/NO have no 15y+ face tape here
    order = ("us", "uk", "de", "fr", "it", "jp", "se", "no")
    rows = []
    for cc in order:
        if cc not in T.index:
            rows.append({"cc": cc, "long_fwd": np.nan, "long_mean": np.nan, "long_dev": np.nan,
                         "long_bp": np.nan, "all_fwd": np.nan, "all_bp": np.nan, "joint_bp": np.nan})
            continue
        r = T.loc[cc]
        all_bp = float(r["all_bp"]) if "all_bp" in r.index and np.isfinite(r["all_bp"]) else np.nan
        joint = float(r["joint_bp"]) if "joint_bp" in r.index and np.isfinite(r["joint_bp"]) else np.nan
        rows.append({
            "cc": cc,
            "long_fwd": float(r["long_fwd"]),
            "long_mean": float(r["long_mean"]),
            "long_dev": float(r["long_dev"]),
            "long_bp": float(r["long_bp"]),
            "all_fwd": float(r["all_fwd"]),
            "all_bp": all_bp,
            "joint_bp": joint,
        })
    out = pd.DataFrame(rows).set_index("cc")
    out.attrs["slope_bp_per_pp"] = float(slope)
    out.attrs["slope_t"] = float(tstat)
    return out


def plan_gap() -> pd.DataFrame:
    """Model−plan cash gap + ``investor.CHANNEL`` (reserves path, shock absorber).

    ``lands_on`` is the live tracking note when present; else the CHANNEL absorber.
    ``reserves`` / ``absorbs`` are institutional facts from CHANNEL for every peer.
    """
    from sovereign_cash.investor import CHANNEL, tracking

    track = {r["cc"]: r for r in tracking()}
    rows = []
    for cc in PEERS:
        t = track.get(cc, {})
        ch = CHANNEL.get(cc, ("", ""))
        reserves, absorbs = ch[0], ch[1]
        p = DERIVED / f"{cc}_cash_summary.json"
        fw = json.loads(p.read_text()).get("forward", {}) if p.exists() else {}
        note = ""
        if cc == "uk" and t.get("lands_on"):
            note = t["lands_on"]
        elif cc == "de":
            sv = fw.get("cash_requirement_sv_12m_bn")
            base = fw.get("plan_gap_lands_on") or t.get("lands_on") or absorbs
            note = (f"SV 12m €{sv:.0f}bn; " if isinstance(sv, (int, float)) else "") + str(base)
        elif t.get("lands_on") and t["lands_on"] not in ("bills / cash buffer",):
            note = t["lands_on"]
        elif fw.get("plan_gap_lands_on"):
            note = fw["plan_gap_lands_on"]
        else:
            note = absorbs
        gap = t.get("gap", fw.get("model_minus_plan_bn", np.nan))
        if gap is None:
            gap = np.nan
        # US has no country cash plan card; TBAC bill-share guidance is the CHANNEL absorber.
        plan_bn = (float(t["plan"]) if t.get("plan") is not None else
                   float(fw["plan_cash_requirement_bn"]) if fw.get("plan_cash_requirement_bn") is not None else np.nan)
        model_bn = float(t["model"]) if t.get("model") is not None else np.nan
        rows.append({
            "cc": cc,
            "gap_bn": float(gap) if np.isfinite(gap) else np.nan,
            "plan_bn": plan_bn,
            "model_bn": model_bn,
            "lands_on": note or absorbs or "—",
            "reserves": reserves or "—",
            "absorbs": absorbs or "—",
            "plan_src": t.get("plan_src") or fw.get("plan_source") or (
                "TBAC refunding / TGA target" if cc == "us" else ""),
        })
    return pd.DataFrame(rows).set_index("cc")


def tables() -> dict:
    """All panels for the page (and a JSON dump for the design stub)."""
    n12 = next12()
    p29 = path2029()
    drift = curve_drift()
    plan = plan_gap()
    return {
        "peers": list(PEERS),
        "next12": n12,
        "path2029": p29,
        "curve": drift,
        "plan": plan,
        "curve_slope_bp_per_pp": drift.attrs.get("slope_bp_per_pp"),
        "curve_slope_t": drift.attrs.get("slope_t"),
    }


def write_stub() -> Path:
    """Thin JSON beside the HTML so a rebuild without long_end still has last numbers."""
    t = tables()
    out = {
        "peers": t["peers"],
        "curve_slope_bp_per_pp": t["curve_slope_bp_per_pp"],
        "curve_slope_t": t["curve_slope_t"],
        "next12": t["next12"].reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "path2029": t["path2029"].reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "curve": t["curve"].reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "plan": t["plan"].reset_index().replace({np.nan: None}).to_dict(orient="records"),
    }
    path = DERIVED / "debt_supply_sensitivities.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    path = write_stub()
    t = tables()
    print(t["next12"].round(2).to_string())
    print(t["path2029"].round(2).to_string())
    print(t["curve"].round(2).to_string())
    print(t["plan"][["gap_bn", "absorbs", "lands_on"]].to_string())
    print(f"Wrote {path}")
