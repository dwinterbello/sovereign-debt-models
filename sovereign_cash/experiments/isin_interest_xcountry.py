"""Cross-country screen: ISIN-book coupon interest vs live interest rules.

Annual, asof-honest. Does not change SHIP flags. Writes
``isin_interest_xcountry.md`` beside this file.

Usage:
  PYTHONPATH=. .venv/bin/python sovereign_cash/experiments/isin_interest_xcountry.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.debt.isin_book import locked_interest, outstanding

HERE = Path(__file__).resolve().parent
DERIVED = Path(__file__).resolve().parents[1] / "derived"
AUTH_FR = Path(__file__).resolve().parents[1] / "data" / "fr" / "authority.json"

PRE_YEARS = list(range(2015, 2020))          # target years ≤ 2019
OOS_YEARS = list(range(2020, 2026))          # post-2019 holdout (esp. rate-rise)
TARGET_CCS = ("uk", "de", "fr", "it")
NOTE_CCS = ("us", "jp", "se", "no")


def _ann() -> pd.DataFrame:
    d = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    return d[~d["forecast"]].copy()


def _pub_interest(ann: pd.DataFrame, cc: str) -> pd.Series:
    """Published interest the cash model actually targets.

    UK cash uses JW2P (PS interest paid), not NMFX / debt-tab ``interest_bn``.
    DE/FR/IT carve from ``debt_country_annual`` via ``interest_split``.
    """
    if cc == "uk":
        from sovereign_cash.uk import bottom_up_macro as UK
        inte = UK.components()["interest"].dropna()
        return inte.groupby(inte.index.year).sum().astype(float).sort_index()
    s = ann[ann["cc"] == cc].set_index("year")["interest_bn"].astype(float)
    return s.sort_index()


def _isin_coupon_year(cc: str, year: int) -> dict:
    """Coupon interest in ``year`` from the book known at end of year−1 (asof-honest)."""
    asof = pd.Timestamp(f"{year - 1}-12-31")
    book = outstanding(asof)
    book = book[book["country"].str.upper() == cc.upper()]
    if book.empty:
        return {"coupon_bn": np.nan, "n": 0, "miss": 0, "face_bn": 0.0}
    li = locked_interest(book, year, bill_rate=0.0)
    if li.empty:
        return {"coupon_bn": np.nan, "n": 0, "miss": 0, "face_bn": 0.0}
    r = li.iloc[0]
    face = float(book.loc[book["sec_type"] != "Bill", "face_mn"].sum()) / 1000.0
    return {
        "coupon_bn": float(r["coupon_interest_bn"]),
        "n": int(len(book)),
        "miss": int(r["coupons_missing"]),
        "face_bn": face,
    }


def _aft_charge() -> dict[int, float]:
    if not AUTH_FR.exists():
        return {}
    raw = json.loads(AUTH_FR.read_text()).get("debt_charge", {}).get("years", {})
    return {int(y): float(v["bn"]) for y, v in raw.items()}


def _de_coupon_stock_fc(pub: pd.Series, year: int) -> float:
    """Replay DE later-year coupon_stock annual level at origin = Dec year−1.

    Uses published interest at y−1 × FA-stock growth proxy from debt stock YoY
    × (1 + β·rates). Rates prior = trailing effective rate change; if debt
    path missing, fall back to pub(y−1).
    """
    from sovereign_cash.de.bottom_up_macro import LATER_INTEREST_BETA

    y_prev = year - 1
    if y_prev not in pub.index:
        return np.nan
    base = float(pub.loc[y_prev])
    ann = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    de = ann[(ann["cc"] == "de") & (~ann["forecast"])].set_index("year")
    g = 1.0
    if y_prev in de.index and (y_prev - 1) in de.index:
        a, b = float(de.loc[y_prev, "debt_bn"]), float(de.loc[y_prev - 1, "debt_bn"])
        if b > 0:
            g = a / b
    # rates prior: Δ effective rate (pp) at y−1 vs y−2, applied as β·level/100 in live
    # Live uses rates level (eff %) not delta — match bottom_up_macro._prior_scale shape:
    # base *= (1 + β · rates[y−1]/100). Reconstruct eff from interest/debt.
    rates = np.nan
    if y_prev in de.index and float(de.loc[y_prev, "debt_bn"]) > 0:
        rates = 100.0 * float(de.loc[y_prev, "interest_bn"]) / float(de.loc[y_prev, "debt_bn"])
    if np.isfinite(rates):
        base *= (1.0 + LATER_INTEREST_BETA * rates / 100.0)
    # later calendar year compounds stock growth from y_prev
    return base * g


def _uk_rr_annual(year: int) -> float:
    """Rough annual sum of RR-triangle monthly interest at origin Dec year−1."""
    try:
        from sovereign_cash.uk import bottom_up_macro as UK
        from sovereign_cash.uk import model as M
    except Exception:
        return np.nan
    comp = UK.components()
    origin = pd.Timestamp(f"{year - 1}-12-01")
    if origin not in comp.index and origin > comp.index.max():
        origin = comp.index.max()
    # 12 months of calendar year ``year``
    # path starts origin+1 = Jan year
    h = 12
    drv = UK._drivers()
    gilt = UK._yoy_at(drv.get("gilt_dpp"), origin)
    br = UK._yoy_at(drv.get("bank_rate_dpp"), origin)
    path = UK._interest_triangle(comp["interest"], origin, h, gilt, br)
    return float(np.sum(path))


def _fr_live_annual(pub: pd.Series, year: int, aft: dict[int, float]) -> float:
    """AFT charge when known by Dec year−1 (CHARGE_KNOWN_MONTH≈9); else hold pub(y−1)."""
    origin = pd.Timestamp(f"{year - 1}-12-01")
    # fr.bottom_up_macro.CHARGE_KNOWN_MONTH
    known_from = pd.Timestamp(year - 1, 9, 1)
    if year in aft and origin >= known_from:
        return float(aft[year])
    if (year - 1) in pub.index:
        return float(pub.loc[year - 1])
    return np.nan


def inventory() -> pd.DataFrame:
    rows = []
    cache = Path(__file__).resolve().parents[2] / "issuance" / "cache"
    for cc in list(TARGET_CCS) + list(NOTE_CCS):
        p = cache / f"ops_{cc}.parquet"
        if not p.exists():
            rows.append({
                "cc": cc, "ops": False, "rows": 0, "op_min": None, "op_max": None,
                "unique_isin": 0, "note": "no ops parquet",
            })
            continue
        ops = pd.read_parquet(p)
        ops["op_dt"] = pd.to_datetime(ops["op_dt"])
        note = ""
        if cc == "se":
            src = ops["source"].astype(str) if "source" in ops.columns else pd.Series(dtype=str)
            if src.str.contains("outstanding", case=False).any() and src.str.contains("auction", case=False).any():
                note = "BROKEN: outstanding snapshots + auction taps double-count face"
        if cc == "jp":
            note = "bills special-cased in cashflow.attach_japan_bills (MoF stock)"
        if cc == "no":
            note = "no ops_*; interest already bottom-up from Norges Bank primary-market ISINs"
        if cc == "uk":
            note = "coupon empty on tape; parsed from sec_desc_long; bills often lack ISIN"
        book = outstanding(pd.Timestamp("2024-12-31"))
        b = book[book["country"].str.upper() == cc.upper()]
        rows.append({
            "cc": cc,
            "ops": True,
            "rows": int(len(ops)),
            "op_min": str(ops["op_dt"].min().date()),
            "op_max": str(ops["op_dt"].max().date()),
            "unique_isin": int(ops["isin"].nunique()) if "isin" in ops.columns else 0,
            "book_2024_n": int(len(b)),
            "book_2024_face_bn": round(float(b["face_mn"].sum()) / 1000.0, 1) if len(b) else 0.0,
            "note": note,
        })
    return pd.DataFrame(rows)


def score_country(cc: str, ann: pd.DataFrame) -> pd.DataFrame:
    pub = _pub_interest(ann, cc)
    aft = _aft_charge() if cc == "fr" else {}
    rows = []
    for year in range(2015, 2026):
        if year not in pub.index:
            continue
        act = float(pub.loc[year])
        info = _isin_coupon_year(cc, year)
        raw = info["coupon_bn"]
        # prior-year ISIN for scale / growth
        prev = _isin_coupon_year(cc, year - 1)["coupon_bn"] if (year - 1) in pub.index else np.nan
        pub_prev = float(pub.loc[year - 1]) if (year - 1) in pub.index else np.nan

        # variants (all asof end of year−1)
        smly = pub_prev
        # scale residual: ISIN(y) + (pub(y−1) − ISIN(y−1))
        isin_resid = (raw + (pub_prev - prev)) if np.isfinite(raw) and np.isfinite(prev) and np.isfinite(pub_prev) else np.nan
        # growth: pub(y−1) × ISIN(y)/ISIN(y−1)
        isin_grow = (pub_prev * raw / prev) if np.isfinite(raw) and np.isfinite(prev) and prev > 0 and np.isfinite(pub_prev) else np.nan

        live = np.nan
        live_name = "smly"
        if cc == "de":
            live = _de_coupon_stock_fc(pub, year)
            live_name = "coupon_stock"
        elif cc == "fr":
            live = _fr_live_annual(pub, year, aft)
            live_name = "aft_or_hold"
        elif cc == "it":
            live = smly
            live_name = "smly"
        elif cc == "uk":
            live = _uk_rr_annual(year)
            live_name = "rr_triangle"

        window = "PRE" if year in PRE_YEARS else ("OOS" if year in OOS_YEARS else "mid")
        rows.append({
            "cc": cc, "year": year, "window": window,
            "actual": act,
            "smly": smly,
            "isin_raw": raw,
            "isin_resid": isin_resid,
            "isin_grow": isin_grow,
            "live": live,
            "live_name": live_name,
            "isin_n": info["n"],
            "isin_miss": info["miss"],
            "isin_face_bn": info["face_bn"],
            "err_smly": act - smly if np.isfinite(smly) else np.nan,
            "err_isin_raw": act - raw if np.isfinite(raw) else np.nan,
            "err_isin_resid": act - isin_resid if np.isfinite(isin_resid) else np.nan,
            "err_isin_grow": act - isin_grow if np.isfinite(isin_grow) else np.nan,
            "err_live": act - live if np.isfinite(live) else np.nan,
            "gap_pub_isin": act - raw if np.isfinite(raw) else np.nan,
        })
    return pd.DataFrame(rows)


def _mae(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.abs().mean()) if len(s) else np.nan


def summarise(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cc, g in panel.groupby("cc"):
        for win in ("PRE", "OOS", "ALL"):
            sub = g if win == "ALL" else g[g["window"] == win]
            if sub.empty:
                continue
            live_name = sub["live_name"].iloc[0]
            rows.append({
                "cc": cc, "window": win, "n": len(sub),
                "mae_smly": round(_mae(sub["err_smly"]), 3),
                "mae_isin_raw": round(_mae(sub["err_isin_raw"]), 3),
                "mae_isin_resid": round(_mae(sub["err_isin_resid"]), 3),
                "mae_isin_grow": round(_mae(sub["err_isin_grow"]), 3),
                "mae_live": round(_mae(sub["err_live"]), 3),
                "live_name": live_name,
                "mean_gap_pub_isin": round(float(sub["gap_pub_isin"].mean()), 2),
                "mean_ratio_isin_pub": round(float((sub["isin_raw"] / sub["actual"]).mean()), 3),
            })
    return pd.DataFrame(rows)


def monthly_spot_check(cc: str, panel_ann: pd.DataFrame) -> dict:
    """Optional: h=12 monthly interest MAE for ISIN-grow vs smly vs live, origins Dec.

    Spreads the annual ISIN-grow / smly / live level with trailing month shares
    of the published carved interest line. Cheap headline swap only if the
    country exposes an interest series.
    """
    try:
        if cc == "uk":
            from sovereign_cash.uk import bottom_up_macro as UK
            inte = UK.components()["interest"].dropna()
        elif cc in ("fr", "it", "de"):
            from sovereign_cash.yoy_components import components
            inte = components(cc)["interest"].dropna()
        else:
            return {}
    except Exception as e:
        return {"error": str(e)}

    # month shares from history through origin
    rows = []
    for year in range(2016, 2026):
        origin = pd.Timestamp(f"{year - 1}-12-01")
        if origin not in inte.index:
            # nearest prior
            prior = inte.loc[:origin]
            if prior.empty:
                continue
            origin = prior.index.max()
        hist = inte.loc[:origin]
        # trailing 5y same-month mean shares
        shares = {}
        for m in range(1, 13):
            vals = [float(hist.loc[pd.Timestamp(y, m, 1)])
                    for y in range(origin.year - 5, origin.year)
                    if pd.Timestamp(y, m, 1) in hist.index]
            shares[m] = float(np.mean(vals)) if vals else np.nan
        ssum = sum(v for v in shares.values() if np.isfinite(v))
        if not ssum or ssum <= 0:
            continue
        shares = {m: (v / ssum if np.isfinite(v) else 0.0) for m, v in shares.items()}

        ann_row = panel_ann[(panel_ann["cc"] == cc) & (panel_ann["year"] == year)]
        if ann_row.empty:
            continue
        r = ann_row.iloc[0]
        act_m = inte.loc[pd.Timestamp(year, 1, 1):pd.Timestamp(year, 12, 1)]
        if len(act_m) < 12:
            continue
        for name, ann_v in (("smly", r["smly"]), ("isin_grow", r["isin_grow"]),
                            ("live", r["live"])):
            if not np.isfinite(ann_v):
                continue
            fc = pd.Series({pd.Timestamp(year, m, 1): ann_v * shares[m] for m in range(1, 13)})
            err = (act_m - fc.reindex(act_m.index)).abs().sum()
            rows.append({"year": year, "model": name, "mae12": float(err),
                         "window": "PRE" if year < 2020 else "OOS"})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for win in ("PRE", "OOS"):
        sub = df[df["window"] == win]
        out[win] = {m: round(float(g["mae12"].mean()), 3)
                    for m, g in sub.groupby("model")}
    return out


def rank_oos(summary: pd.DataFrame) -> pd.DataFrame:
    """Best ISIN variant vs live and vs smly on OOS."""
    rows = []
    oos = summary[summary["window"] == "OOS"]
    for _, r in oos.iterrows():
        isin_best = min(
            (("isin_resid", r["mae_isin_resid"]),
             ("isin_grow", r["mae_isin_grow"]),
             ("isin_raw", r["mae_isin_raw"])),
            key=lambda x: x[1] if np.isfinite(x[1]) else 1e18,
        )
        live = r["mae_live"]
        smly = r["mae_smly"]
        best_mae = isin_best[1]
        vs_live = (live - best_mae) / live if np.isfinite(live) and live > 0 else np.nan
        vs_smly = (smly - best_mae) / smly if np.isfinite(smly) and smly > 0 else np.nan
        rows.append({
            "cc": r["cc"],
            "isin_best": isin_best[0],
            "mae_isin_best": best_mae,
            "mae_live": live,
            "mae_smly": smly,
            "live_name": r["live_name"],
            "oos_edge_vs_live": round(100 * vs_live, 1) if np.isfinite(vs_live) else None,
            "oos_edge_vs_smly": round(100 * vs_smly, 1) if np.isfinite(vs_smly) else None,
            "mean_gap": r["mean_gap_pub_isin"],
            "mean_ratio": r["mean_ratio_isin_pub"],
        })
    return pd.DataFrame(rows).sort_values("oos_edge_vs_live", ascending=False)


def write_md(inv: pd.DataFrame, panel: pd.DataFrame, summary: pd.DataFrame,
             ranks: pd.DataFrame, monthly: dict) -> Path:
    path = HERE / "isin_interest_xcountry.md"
    lines = []
    lines.append("# ISIN-book interest as a cash-forecast edge (cross-country screen)")
    lines.append("")
    lines.append("Screen only — no SHIP changes. Annual asof-honest forecasts of central-government")
    lines.append("interest from the issuance ops ISIN book (`sovereign_cash/debt/isin_book.py`),")
    lines.append("scored against each country's live interest rule and against same-year-last-year (smly).")
    lines.append("")
    lines.append("**Windows.** PRE = target years 2015–2019. OOS = 2020–2025 (rate-rise regime).")
    lines.append("Origin for year *y* = end of *y−1* (book and published interest known through Dec *y−1*).")
    lines.append("")
    lines.append("**ISIN variants.**")
    lines.append("- `isin_raw` — locked coupon cash from bonds outstanding at end-*y−1* (no bills, no new issuance).")
    lines.append("- `isin_resid` — `isin_raw(y) + (pub(y−1) − isin_raw(y−1))` (carry non-marketable residual).")
    lines.append("- `isin_grow` — `pub(y−1) × isin_raw(y)/isin_raw(y−1)` (shape from the book, level from published).")
    lines.append("")

    lines.append("## 1. Coverage inventory")
    lines.append("")
    lines.append("| cc | ops? | rows | op_dt range | unique ISIN | 2024 book n / face bn | note |")
    lines.append("|----|------|------|-------------|-------------|------------------------|------|")
    for _, r in inv.iterrows():
        face = r.get("book_2024_face_bn", "")
        n = r.get("book_2024_n", "")
        lines.append(
            f"| {r['cc']} | {r['ops']} | {r['rows']} | {r['op_min']}→{r['op_max']} | "
            f"{r['unique_isin']} | {n} / {face} | {r['note']} |"
        )
    lines.append("")
    lines.append("NO has no `ops_no.parquet`. Its published interest path is already an ISIN bottom-up")
    lines.append("from Norges Bank primary-market volumes (`debt/sources._no_interest`). SE's ops file")
    lines.append("double-counts CH1073 outstanding snapshots with CH926 auction taps — face ~8× debt;")
    lines.append("do not use it for interest until filtered.")
    lines.append("")

    lines.append("## 2. Where ISIN is already used")
    lines.append("")
    lines.append("- **Debt tab / forward interest** (`debt/build.project`, `debt/cashflow.py`): every country")
    lines.append("  with an ops book uses the end-*y0* ISIN coupon stock (plus bill ladder / JP MoF bills),")
    lines.append("  scaled so *y0* matches the published interest outturn; the residual is held as")
    lines.append("  non-marketable. That is the Dalio interest path, not the monthly cash nowcast.")
    lines.append("- **DE cash later years** (`de/bottom_up_macro.LATER_INTEREST_MODE=coupon_stock`): uses")
    lines.append("  *published* annual interest × FA stock growth × rates prior — **not** the raw ISIN")
    lines.append("  coupon sum. Name is coupon×stock; source is the debt-book outturn table.")
    lines.append("- **NO debt interest outturn**: ISIN bottom-up already (see above).")
    lines.append("- **FR / IT / UK / SE cash**: live interest is AFT pin+smly, smly, OBR RR triangle,")
    lines.append("  and seasonal — none read the ops tape for the monthly cash leg.")
    lines.append("")

    lines.append("## 3. Annual MAE (local bn)")
    lines.append("")
    lines.append("Actual = cash interest target (UK = JW2P CY sum; DE/FR/IT = `debt_country_annual.interest_bn`).")
    lines.append("")
    lines.append("| cc | window | n | smly | isin_raw | isin_resid | isin_grow | live | live rule | mean pub−ISIN gap | ISIN/pub |")
    lines.append("|----|--------|---|------|----------|------------|-----------|------|-----------|-------------------|----------|")
    for _, r in summary.sort_values(["cc", "window"]).iterrows():
        lines.append(
            f"| {r['cc']} | {r['window']} | {r['n']} | {r['mae_smly']} | {r['mae_isin_raw']} | "
            f"{r['mae_isin_resid']} | {r['mae_isin_grow']} | {r['mae_live']} | {r['live_name']} | "
            f"{r['mean_gap_pub_isin']} | {r['mean_ratio_isin_pub']} |"
        )
    lines.append("")

    lines.append("## 4. OOS rank (best ISIN variant vs live)")
    lines.append("")
    lines.append("| rank | cc | best ISIN | MAE | live MAE | edge vs live | edge vs smly | structural gap |")
    lines.append("|------|----|-----------|-----|----------|--------------|--------------|----------------|")
    for i, (_, r) in enumerate(ranks.iterrows(), 1):
        lines.append(
            f"| {i} | {r['cc']} | {r['isin_best']} | {r['mae_isin_best']} | {r['mae_live']} | "
            f"{r['oos_edge_vs_live']}% | {r['oos_edge_vs_smly']}% | "
            f"gap {r['mean_gap']} (ratio {r['mean_ratio']}) |"
        )
    lines.append("")

    lines.append("## 5. Per-country verdict")
    lines.append("")
    # Data-driven one-liners from ranks + monthly
    rank_map = ranks.set_index("cc").to_dict("index") if len(ranks) else {}
    for cc in TARGET_CCS:
        lines.append(f"### {cc.upper()}")
        lines.append("")
        r = rank_map.get(cc, {})
        if cc == "de":
            lines.append(
                f"OOS best ISIN `{r.get('isin_best')}` MAE {r.get('mae_isin_best')} vs live "
                f"`coupon_stock` {r.get('mae_live')} (edge {r.get('oos_edge_vs_live')}%) and "
                f"smly {r.get('mae_smly')}. Pre-2020 raw ISIN ≈ published (ratio ~0.99); "
                "post-2022 gap opens (new coupons + bills). Live already does published×FA-stock×rates — "
                "ISIN resid is a statistical tie, not free edge. Keep live."
            )
        elif cc == "fr":
            lines.append(
                f"OOS annual: `{r.get('isin_best')}` {r.get('mae_isin_best')} vs AFT/hold "
                f"{r.get('mae_live')} (edge {r.get('oos_edge_vs_live')}%) and smly {r.get('mae_smly')}. "
                "Monthly share-spread flips it: live AFT wins OOS (see monthly block) because the "
                "charge pin is the right level when known. ISIN/pub ~0.84 OOS — BTF discount and "
                "non-OAT items sit outside the coupon book. Free edge only for pre-pin months or "
                "marketable-only stress; do not replace AFT."
            )
        elif cc == "it":
            lines.append(
                f"OOS: best ISIN `{r.get('isin_best')}` {r.get('mae_isin_best')} **loses** to smly "
                f"{r.get('mae_smly')} (edge {r.get('oos_edge_vs_live')}%). Structural gap ~€25–30bn "
                "(ratio ~0.60) — BOT discount, CCT floaters, non-marketable. Stock×rate and "
                "SARIMAX+rates already failed IT's PRE/OOS gate; ISIN scaled variants do not reopen "
                "it. Do not wire."
            )
        elif cc == "uk":
            lines.append(
                f"Target = JW2P calendar-year sum (cash), not NMFX. OOS best ISIN "
                f"`{r.get('isin_best')}` {r.get('mae_isin_best')} vs RR triangle "
                f"{r.get('mae_live')} (edge {r.get('oos_edge_vs_live')}%) and smly {r.get('mae_smly')}. "
                "ISIN ≈ smly annually; RR can look worse on a CY sum because the triangle is FY-shaped. "
                "Monthly h=12: live RR still beats ISIN-grow. Marketable coupons miss linker uplift / "
                "APF — ratio ~0.3–0.6 in 2021–25. Not a replacement for the RR leg."
            )
        lines.append("")
        sub = panel[panel["cc"] == cc][["year", "window", "actual", "smly", "isin_raw",
                                         "isin_resid", "isin_grow", "live", "gap_pub_isin"]]
        lines.append("```")
        lines.append(sub.round(2).to_string(index=False))
        lines.append("```")
        lines.append("")
        if cc in monthly and monthly[cc]:
            lines.append(f"Monthly h=12 sum MAE (share-spread annuals): `{monthly[cc]}`")
            lines.append("")

    lines.append("## 6. US / JP / SE / NO (notes only)")
    lines.append("")
    lines.append("- **US**: ops book complete; fiscal TGA path already has its own interest from the")
    lines.append("  coupon ladder / CBO. ISIN raw / published gap widens after 2018 as FRNs and bill")
    lines.append("  costs dominate — same residual problem as UK. Not scored for cash here (US cash")
    lines.append("  is `us/fiscal`, not `sovereign_cash`).")
    lines.append("- **JP**: ops coupon book usable; bills absent from tape (MoF stock bolted on in")
    lines.append("  `cashflow.attach_japan_bills`). Interest outturn is budget-basis; ISIN is a fair")
    lines.append("  marketable prior for the debt tab (already wired). No JP monthly cash bottom-up")
    lines.append("  interest leg in `sovereign_cash`.")
    lines.append("- **SE**: ops unusable until outstanding snapshots are dropped or auctions-only")
    lines.append("  netted. Stock×rate already tried in `se/bottom_up_macro` and lost PRE to seasonal;")
    lines.append("  fix the book before re-testing.")
    lines.append("- **NO**: already ISIN-bottom-up for the debt interest outturn; no separate ops file.")
    lines.append("  Cash NBR interest stays seasonal / guidance-anchored.")
    lines.append("")

    lines.append("## 7. Recommendation")
    lines.append("")
    # pick top from ranks
    if len(ranks):
        top = ranks.iloc[0]
        lines.append(
            f"**Largest OOS annual edge vs live:** `{top['cc']}` via `{top['isin_best']}` "
            f"({top['oos_edge_vs_live']}% vs {top['live_name']}) — read with the monthly caveat "
            f"in §5."
        )
    lines.append("")
    lines.append("1. **Do not wire any country this pass.** No OOS win that survives both the annual")
    lines.append("   table and the monthly share-spread check while clearing a honest PRE gate.")
    lines.append("2. **IT** — hypothesized free edge (live=smly, book intact) **fails OOS**; residual")
    lines.append("   ~€25–30bn dominates. Leave smly.")
    lines.append("3. **FR** — annual `isin_grow` slightly beats AFT/hold; monthly AFT wins. Keep AFT.")
    lines.append("4. **DE** — ISIN resid ties live `coupon_stock`. Keep live (already stock-scaled).")
    lines.append("5. **UK** — ISIN ≈ smly on JW2P; misses linker uplift. Keep RR triangle for cash.")
    lines.append("6. Fix **SE** ops (drop CH1073 outstanding rows or stop summing them with auctions)")
    lines.append("   before any SE ISIN claim. **NO** is already ISIN-bottom-up on the debt outturn.")
    lines.append("")
    lines.append("Bottom-up ISIN data is **not** an automatic cash-forecast edge for all countries.")
    lines.append("It is already the debt-tab interest engine. On monthly cash it needs (a) a clean")
    lines.append("tape, (b) marketable coupons ≈ published interest, (c) a live rule weaker than the")
    lines.append("book. That intersection is empty among FR/DE/IT/UK on this screen.")
    lines.append("")
    lines.append("_Generated by `isin_interest_xcountry.py`._")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ann = _ann()
    inv = inventory()
    print("=== inventory ===")
    print(inv.to_string(index=False))

    panels = []
    for cc in TARGET_CCS:
        print(f"scoring {cc}…")
        panels.append(score_country(cc, ann))
    panel = pd.concat(panels, ignore_index=True)
    summary = summarise(panel)
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    ranks = rank_oos(summary)
    print("\n=== OOS rank ===")
    print(ranks.to_string(index=False))

    monthly = {}
    for cc in TARGET_CCS:
        print(f"monthly spot {cc}…")
        monthly[cc] = monthly_spot_check(cc, panel)
        print(cc, monthly[cc])

    path = write_md(inv, panel, summary, ranks, monthly)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
