"""Budget financing behind each country's coupon supply, for the Country tab.

Read-only view of the cash frameworks (`sovereign_cash/derived/<cc>_cash_*`) and the US supply
bridge (`us/fiscal/derived/supply_bridge.parquet`). Each is a monthly cash identity in local
currency bn:

    change in cash buffer = −cash requirement + net bonds + net bills + other financing

Nothing here re-models anything; it lines the frameworks up with this board's coupon calendar.
The forward bond line of the UK, DE, FR and IT frameworks is this board's calendar after their own
tape, so the comparison is a consistency check (timing, cash vs face, DE market placement), not
an independent forecast. Only the US coupon path is built separately (fiscal coupon template).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.paths import CACHE, REPO

FRAMEWORK = {"UK": "uk", "DE": "de", "FR": "fr", "IT": "it", "ES": "es", "JP": "jp", "US": "us"}
CCY = {"UK": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR", "JP": "JPY", "US": "USD"}
LINK = {"US": "../us/fiscal/us_fiscal_dashboard.html"}
# (column, label, how the row is read); the order is the identity's.
ROWS = (
    ("cash_requirement_bn", "Cash requirement", "Budget deficit to finance (+ = borrowing need)."),
    ("bond_gross_bn", "Gross bonds", "Coupon bonds sold, cash."),
    ("bond_redemption_bn", "Bond redemptions", "Coupon bonds maturing (net of buybacks where the issuer runs them)."),
    ("net_bonds_bn", "Net bonds", "Gross bonds − redemptions."),
    ("bond_redemption_cb_bn", "Redemptions held by the central bank",
     "Maturing bonds the central bank holds and does not reinvest: cash leaves the Treasury but not the private market."),
    ("private_net_bonds_bn", "Net bonds to the private market", "Net bonds + central-bank-held redemptions."),
    ("net_bills_bn", "Net bills", "Change in the bill stock."),
    ("other_fin_bn", "Other financing", "Everything else that funds the Treasury (retail, EU loans, deposits); closes the identity."),
    ("buffer_change_bn", "Change in cash buffer", "Treasury cash: + = builds."),
)
NO_FRAMEWORK: dict[str, str] = {}


def _summary(cc: str) -> dict:
    from sovereign_cash import investor
    return investor.summary(cc)


def _frame(c: str) -> pd.DataFrame | None:
    """Monthly identity with every ROWS column; months not yet published flagged `is_forecast`."""
    from sovereign_cash import investor
    cc = FRAMEWORK[c]
    if cc == "us":
        p = REPO / "us" / "fiscal" / "derived" / "supply_bridge.parquet"
        if not p.exists():
            return None
        b = pd.read_parquet(p).loc["unchanged"].copy()
        b.index = pd.to_datetime(b.index)
        b = b.iloc[1:]  # origin month is part history
        d = pd.DataFrame({"month": b.index, "is_forecast": True,
                          "cash_requirement_bn": b["deficit_bn"].to_numpy(),
                          "bond_gross_bn": b["coupon_iss_cash_bn"].to_numpy(),
                          "bond_redemption_bn": b["coupon_red_bn"].to_numpy(),
                          "net_bonds_bn": b["net_coupons_bn"].to_numpy(),
                          "bond_redemption_cb_bn": 0.0,
                          "net_bills_bn": b["private_net_bills_bn"].to_numpy(),
                          "buffer_change_bn": b["d_tga_bn"].to_numpy()})
        d["other_fin_bn"] = d["buffer_change_bn"] + d["cash_requirement_bn"] - d["net_bonds_bn"] - d["net_bills_bn"]
    else:
        d = investor.monthly(cc)
        if d is None:
            return None
        d = d.copy()
        if "bond_gross_bn" not in d and {"net_bonds_bn", "bond_redemption_bn"} <= set(d):
            d["bond_gross_bn"] = d["net_bonds_bn"] + d["bond_redemption_bn"]
    if "bond_redemption_cb_bn" not in d:
        d["bond_redemption_cb_bn"] = np.nan
    d["private_net_bonds_bn"] = d["net_bonds_bn"] + d["bond_redemption_cb_bn"]
    return d.sort_values("month").reset_index(drop=True)


def _board_gross(c: str) -> pd.Series:
    """This board's gross coupon issuance per auction month, local bn: tape, plus forecast operations."""
    out = pd.Series(dtype=float)
    sup = CACHE / "rv_supply.parquet"
    if sup.exists():
        r = pd.read_parquet(sup, columns=["country", "month", "coupon_gross_mn"])
        r = r[r["country"] == c]
        out = pd.Series(r["coupon_gross_mn"].to_numpy() / 1e3, index=pd.to_datetime(r["month"])).dropna()
    fp = CACHE / "forecast_priced.parquet"
    if fp.exists():
        from issuance.forward_audit import NON_COUPON
        f = pd.read_parquet(fp, columns=["country", "op_dt", "amount_mn", "sec_type"])
        f = f[(f["country"] == c) & ~f["sec_type"].astype(str).isin(NON_COUPON)]
        m = pd.to_datetime(f["op_dt"]).dt.to_period("M").dt.to_timestamp()
        fc = f.groupby(m)["amount_mn"].sum() / 1e3
        out = out.add(fc, fill_value=0.0)
    return out.sort_index()


def _basis(c: str, fw: dict) -> str:
    """How the framework's gross-bond line relates to this board, from the framework's own summary."""
    if c == "US":
        return ("Coupons: the fiscal path's template (year-ago and two-years-ago settlements at today's auction sizes), "
                "built separately from this board's US calendar. Net bills are private: Fed bill purchases and rolls "
                "sit in other financing.")
    if c == "UK":
        um = fw.get("end_march", {})
        return (f"Gilts: tape to {um.get('gilt_tape_end', '—')}, then the remit remaining "
                f"(£{um.get('gilt_remit_remaining_bn', float('nan')):.1f}bn, {um.get('gilt_remit_window', '—')}) on this "
                f"board's calendar, then this board's forward × cash per nominal {um.get('gilt_cash_per_nominal', float('nan')):.3f}. "
                "Same calendar as this board: differences are timing and cash vs face only.")
    if c == "DE":
        return (f"Coupons: Finanzagentur tape to {fw.get('tape_last_op', '—')}, then this board's calendar × "
                f"{fw.get('gross_market_ratio', float('nan')):.3f} (market placement ÷ auctioned volume over the last 12 "
                "months: the Finanzagentur retains part of each auction and sells it later). This board counts auctioned "
                "volume, so the framework sits below it by that ratio.")
    if c == "FR":
        return (f"OAT gross: tape to {fw.get('tape_last_op', '—')}, then this board's programme targets. Same source "
                "as this board: not an independent check. Redemptions net of buybacks.")
    if c == "IT":
        return (f"BTP gross: tape to {fw.get('bond_gross_tape_last_op', '—')}, then this board's calendar; announced "
                "MEF issues already on that calendar are counted once. Same calendar as this board: differences are "
                "timing and cash vs face only.")
    return ""


_MODEL_WORDS = (
    (r"\bseasonal_add(\d+) seasonal model\b", r"seasonal model (trailing 12-month level plus the average \1-year monthly pattern)"),
    (r"\b(?:seasonal_add|seas_add_)(\d+)\b", r"seasonal model (trailing 12-month level plus the average \1-year monthly pattern)"),
    (r"\b(?:profile_mult|seas_mult_)(\d+)\b", r"seasonal model (trailing 12-month level times the average \1-year monthly share)"),
    (r"\bplanhc_(\d+)\b", r"budget plan times the execution shortfall of the last \1 completed years"),
    (r"\bh=12 MAE ([\d.]+) vs smly ([\d.]+)", r"12 months ahead, mean absolute error \1bn against \2bn for same-month-last-year"),
    (r"\bsmly\b", "same month last year"),
    (r"\bchosen on PRE\b", "chosen on the earlier test window, scored on the later one"),
    (r"\btoo few PRE windows to test\b", "too few early-window forecasts to test"),
    (r"\(not PRE-gated\)", "(not tested on the earlier window)"),
    (r"\b(ties|beats|loses to) naive\b", r"\1 the naive forecast"),
    (r"^DTS flow model \(HGB calendar × trailing level\), TGA steered to refunding target",
     "Daily Treasury Statement flow model (calendar pattern × trailing level); Treasury cash balance steered to the refunding target"),
    (r";\s*sovereign_cash/\S+", ""),
    (r"\bremit CGNCR \(ex NRAM etc\.\)", "remit: central government net cash requirement excluding NRAM and similar"),
    (r"\b2026 Soll\b", "2026 Soll (budgeted amount)"),
    (r"\b250-bday cum\. op-flow MAE ([\d.]+) vs ([\d.]+) last-year",
     r"250 business days ahead, cumulative operating-flow mean absolute error $\1bn against $\2bn for last year's flows"),
)


def _plain(text: str | None) -> str | None:
    """Model ids and abbreviations from the framework summaries, in words."""
    import re

    if not text:
        return text
    for pat, rep in _MODEL_WORDS:
        text = re.sub(pat, rep, text)
    return text


def payload(today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp.today().normalize() if today is None else today
    try:
        from sovereign_cash import investor
        sc = investor.scalers()
        tracking = {r["cc"]: r for r in investor.tracking()}
        skill = {r["cc"]: r for r in investor.skill()}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"cash frameworks unavailable ({exc})", "none": NO_FRAMEWORK}
    fx = sc.get("fx_usd_per_unit", {})
    gdp = sc.get("gdp_bn", {})
    out: dict = {"none": NO_FRAMEWORK, "fxDate": fx.get("date"), "byCountry": {}}
    cur = today.to_period("M").to_timestamp()
    for c, cc in FRAMEWORK.items():
        d = _frame(c)
        if d is None:
            continue
        f = d[d["is_forecast"]].head(12)
        if f.empty:
            continue
        fw = _summary(cc).get("forward", {}) if cc != "us" else {}
        g = gdp.get(cc, np.nan)
        rate = fx.get(CCY[c], np.nan)
        rows = []
        for col, label, how in ROWS:
            v = f[col].sum(min_count=1) if col in f else np.nan
            if not np.isfinite(v):
                continue
            rows.append({"k": col, "label": label, "how": how, "bn": round(float(v), 1),
                         "pct_gdp": round(float(100 * v / g), 2) if np.isfinite(g) else None,
                         "usd": round(float(v * rate), 1) if np.isfinite(rate) else None})
        resid = float((f["buffer_change_bn"] + f["cash_requirement_bn"] - f["net_bonds_bn"]
                       - f["net_bills_bn"] - f["other_fin_bn"]).abs().max())
        # Monthly gross bonds: framework (settlement month, cash) vs this board (auction month, face), from
        # twelve months before the window to its end.
        x = pd.date_range(f["month"].iloc[0] - pd.DateOffset(months=12), f["month"].iloc[-1], freq="MS")
        fr = d.set_index("month")["bond_gross_bn"].reindex(x)
        bd = _board_gross(c).reindex(x)
        tr = tracking.get(cc)
        sk = skill.get(cc)
        out["byCountry"][c] = {
            "ccy": CCY[c], "window": f"{f['month'].iloc[0]:%b-%y}–{f['month'].iloc[-1]:%b-%y}", "n": int(len(f)),
            "fcStart": f["month"].iloc[0].strftime("%Y-%m-%d"),
            "startsBeforeBoard": bool(f["month"].iloc[0] < cur),
            "rows": rows, "identityMaxResid": round(resid, 3),
            "gdp": round(float(g), 1) if np.isfinite(g) else None,
            "x": [m.strftime("%Y-%m-%d") for m in x],
            "frameworkGross": [None if not np.isfinite(v) else round(float(v), 2) for v in fr],
            "boardGross": [None if not np.isfinite(v) else round(float(v), 2) for v in bd],
            "basis": _basis(c, fw),
            "plan": None if not tr else {k: (_plain(tr.get(k)) if k == "plan_src" else tr.get(k))
                                         for k in ("year", "ytd", "ytd_through", "plan", "plan_src", "model", "gap", "lands_on")},
            "skill": None if not sk else {
                "basis": _plain(sk.get("basis")), "oos": _plain(sk.get("oos")),
                "verdict": "this is the naive forecast itself" if str(sk.get("basis", "")).startswith("smly")
                else _plain(sk.get("verdict"))},
            "absorbs": investor.CHANNEL.get(cc, ("", ""))[1],
            "link": LINK.get(c, f"../sovereign_cash/{cc}_cash_dashboard.html"),
        }
    return out
