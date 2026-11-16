"""UK central-government cash framework: history identity, backtests, 12-month forward, outputs.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.uk.run            # cached inputs
  PYTHONPATH=. .venv/bin/python -m sovereign_cash.uk.run --refresh  # re-fetch ONS PUSF series

DMO tables (D2.2G bills, D2.2D tenders, D2.1E operations, D1C redemptions, D1D holdings) are read
from the uk_flows caches; refresh those with ``uk_flows/dmo.py``. Writes
``sovereign_cash/derived/uk_cash_monthly.parquet``, ``uk_cash_summary.json``,
``uk_cash_score.csv``, ``uk_cash_mm_panel.parquet`` and ``sovereign_cash/uk_cash_dashboard.html``.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from sovereign_cash import contract
from sovereign_cash.contract import fmt_bn, fmt_num
from sovereign_cash.uk import model as M
from sovereign_cash.uk import sources as src

DERIVED = contract.DERIVED
MONTHLY = DERIVED / "uk_cash_monthly.parquet"
SUMMARY = DERIVED / "uk_cash_summary.json"
SCORE = DERIVED / "uk_cash_score.csv"
MM = DERIVED / "uk_cash_mm_panel.parquet"
GBP = "£"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 3)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def g(v, dp: int = 1, plus: bool = False) -> str:
    return fmt_bn(v, GBP, dp, plus=plus)


def _hlist(hs: list[int]) -> str:
    return ", ".join(f"{h}m" for h in hs) if hs else "none"


def _fy_label(fy: int) -> str:
    return f"FY{fy}-{str(fy + 1)[2:]}"


def _sources(hist: pd.DataFrame, ons_ok: dict[str, bool], ons_meta: dict, checks: dict) -> list[dict]:
    def span(key):
        m = ons_meta.get(key)
        return f"{m['first']:%Y-%m}..{m['last']:%Y-%m}" if m else "missing"

    c = checks
    of, pf = c["o_fy"], c["p_fy"]
    base = "https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/{}/pusf"
    out = [
        {"name": f"ONS PUSF M98R CGNCR ex NRAM, B&B, NR ({span('cgncr_ex')})", "url": base.format("m98r"),
         "frequency": "monthly", "lag_days": 21,
         "verified": bool(ons_ok.get("m98r") and abs(c["cgncr_fy"] - c["cgncr_fy_dmo"]) < 0.15),
         "check": f"{of} sum {fmt_num(c['cgncr_fy'])} vs DMO outturn {fmt_num(c['cgncr_fy_dmo'])}"},
        {"name": f"ONS PUSF RUUW CGNCR ({span('cgncr')})", "url": base.format("ruuw"),
         "frequency": "monthly", "lag_days": 21,
         "verified": bool(ons_ok.get("ruuw") and abs(c["ruuw_minus_m98r_fy"]) < 3.0),
         "check": f"{of} RUUW − M98R {g(c['ruuw_minus_m98r_fy'], 2)} (NRAM, B&B, Network Rail)"},
        {"name": f"ONS PUSF ACUA CG national savings stock; change = NS&I net financing ({span('nsandi')})",
         "url": base.format("acua"), "frequency": "monthly", "lag_days": 21,
         "verified": bool(ons_ok.get("acua") and abs(c["nsandi_fy"] - c["nsandi_fy_dmo"]) < 1.5),
         "check": f"{of} change {fmt_num(c['nsandi_fy'])} vs DMO NS&I net + green {fmt_num(c['nsandi_fy_dmo'])}"},
        {"name": f"ONS PUSF ANTA/ANSZ/ANTB/ANTC financing of CGNCR ({span('fin_bgs')})",
         "url": base.format("anta"), "frequency": "monthly", "lag_days": 21,
         "verified": bool(all(ons_ok.get(k) for k in ("anta", "ansz", "antb", "antc")) and c["gap_2022"] < 2.5),
         "check": f"largest unexplained financing gap since 2022 {g(c['gap_2022'], 2)} (limit {g(2.5)})"},
        {"name": f"ONS PUSF BKPJ CG sterling T-bills ({span('tbill_level')})", "url": base.format("bkpj"),
         "frequency": "monthly", "lag_days": 21,
         "verified": bool(ons_ok.get("bkpj") and c["bills_vs_bkpj"] < 1.5),
         "check": f"largest monthly gap to D2.2G since 2015 {g(c['bills_vs_bkpj'], 2)}"},
        {"name": f"DMO D2.2G T-bill stock (through {c['bills_last']:%Y-%m})",
         "url": "https://www.dmo.gov.uk/data/treasury-bills/", "frequency": "monthly", "lag_days": 1,
         "verified": c["bills_vs_bkpj"] < 1.5,
         "check": (f"largest monthly gap to ONS BKPJ since 2015 {g(c['bills_vs_bkpj'], 2)}; debt-financing "
                   f"stock published at FY-end only")},
        {"name": f"DMO D2.2D T-bill tenders (last issue {c['tender_last']})",
         "url": "https://www.dmo.gov.uk/data/treasury-bills/", "frequency": "weekly", "lag_days": 1,
         "verified": bool(c["tender_mae"] is not None and c["tender_mae"] < 0.5),
         "check": (f"issued − maturing vs D2.2G monthly change since 2023: mean abs gap "
                   f"{g(c['tender_mae'], 2)}; pins {c['tender_pinned'] or 'no month'}")},
        {"name": "DMO D2.1E gilt operations (priced, settlement-dated)",
         "url": "https://www.dmo.gov.uk/data/gilt-market/gilt-operations/", "frequency": "daily", "lag_days": 1,
         "verified": abs(c["gilt_sales_fy"] - c["gilt_sales_fy_dmo"]) < 1.0,
         "check": f"{of} gross cash {fmt_num(c['gilt_sales_fy'])} vs DMO {fmt_num(c['gilt_sales_fy_dmo'])}"},
        {"name": "DMO D1C redeemed gilts + D1D government holdings (netted)",
         "url": "https://www.dmo.gov.uk/data/gilt-market/redemptions/", "frequency": "event", "lag_days": 1,
         "verified": bool(abs(c["red_fy"] - c["red_fy_dmo"]) < 1.0 and abs(c["red_pfy"] - c["red_pfy_dmo"]) < 1.0),
         "check": (f"{of} redemption cash {fmt_num(c['red_fy'])} vs DMO {fmt_num(c['red_fy_dmo'])}; {pf} ladder "
                   f"{fmt_num(c['red_pfy'])} vs remit {fmt_num(c['red_pfy_dmo'])}")},
        {"name": "BoE APF gilt holdings by ISIN",
         "url": "https://www.bankofengland.co.uk/markets/bank-of-england-asset-purchase-facility",
         "frequency": "event", "lag_days": 1, "verified": bool(c["apf_age_days"] is not None and c["apf_age_days"] < 45),
         "check": (f"last APF operation {c['apf_last']} ({c['apf_age_days']}d before build); APF-held "
                   f"redemptions in the 12m window {g(c['apf_red_12m'])}")},
        {"name": "DMO remit progress report (currentremit.pdf)",
         "url": "https://www.dmo.gov.uk/dmo_static_reports/currentremit.pdf", "frequency": "weekly",
         "lag_days": 1, "verified": bool(c["remit_total_ok"] and c["remit_age_days"] < 21 and abs(c["gilt_fy_vs_plan"]) < 1.0),
         "check": (f"as of {c['remit_asof']} ({c['remit_age_days']}d old); total {fmt_num(c['remit_total'])} vs remit "
                   f"{fmt_num(c['remit_plan'])}; FY gross (tape + remaining) − remit {g(c['gilt_fy_vs_plan'], 2, True)}")},
        {"name": f"DMO financing arithmetic, Annex C ({c['anchor_published']})",
         "url": "https://www.dmo.gov.uk/responsibilities/financing-remit/full-details/", "frequency": "fiscal event",
         "lag_days": 23, "verified": bool(c["arith_max_miss"] is not None and c["arith_max_miss"] <= 0.25),
         "check": (f"hand-entered (data/uk/dmr_anchors.csv); net cash identity closes on every complete row "
                   f"to {g(c['arith_max_miss'], 2)} (published rounding)")},
        {"name": f"DMO remit CGNCR(ex) vintages ({c['vint_span']})",
         "url": "https://www.dmo.gov.uk/responsibilities/financing-remit/full-details/", "frequency": "fiscal event",
         "lag_days": 0, "verified": bool(c["vintage_outturn_miss"] is not None and c["vintage_outturn_miss"] < 1.5),
         "check": (f"{c['n_vintages']} vintages hand-entered (data/uk/remit_vintages.csv); first-published outturns "
                   f"vs M98R FY sums: max miss {g(c['vintage_outturn_miss'], 2)} ({c['vintage_outturn_worst']})")},
        {"name": "SONIA and Bank Rate (BoE, via the long-end workbook)",
         "url": "https://www.bankofengland.co.uk/boeapps/database/", "frequency": "daily", "lag_days": 1,
         "verified": bool(c["sonia_age_days"] is not None and c["sonia_age_days"] < 10),
         "check": f"last obs {c['sonia_last']} ({c['sonia_age_days']}d before build)"},
    ]
    return out


def _arith_miss(anchors: pd.DataFrame) -> float | None:
    """Max |published end-March net cash − financing-arithmetic end| over rows with every field."""
    a = anchors[anchors["kind"].isin(["outturn", "plan"])]
    out = a[a["kind"] == "outturn"].set_index("fy")["net_cash_end_march"]
    miss = []
    need = ["cgncr_ex", "gilt_redemptions", "nsandi_net", "nsandi_green", "other_financing", "gilt_sales",
            "tbill_net_debtfin", "net_cash_end_march"]
    for r in a.itertuples(index=False):
        if any(pd.isna(getattr(r, k)) for k in need):
            continue
        y = int(r.fy[:4])
        prev = f"{y - 1}-{str(y)[2:]}"
        if prev not in out.index:
            continue
        sk = 0.0 if pd.isna(r.sukuk_redemption) else r.sukuk_redemption
        sp = 0.0 if pd.isna(r.secondary_purchase_fin) else r.secondary_purchase_fin
        end = (out[prev] + r.gilt_sales + r.tbill_net_debtfin + r.nsandi_net + r.nsandi_green
               + r.other_financing - r.cgncr_ex - r.gilt_redemptions - sk - sp)
        miss.append(abs(end - r.net_cash_end_march))
    return max(miss) if miss else None


def _vintage_outturn_check(hist: pd.DataFrame, vint: pd.DataFrame) -> tuple[float | None, str]:
    x = hist["cash_requirement_bn"]
    fy = pd.Series(M.fy_of(x.index), index=x.index)
    tot = x.groupby(fy).sum()[x.groupby(fy).size() == 12]
    o = vint[(vint["kind"] == "outturn") & vint["comparable"]].set_index("fy_start")["cgncr_ex"]
    d = (o - tot.reindex(o.index)).dropna()
    if d.empty:
        return None, ""
    k = d.abs().idxmax()
    return float(d.abs().max()), f"{_fy_label(int(k))} {fmt_num(o[k])} vs {fmt_num(tot[k])}"


def _skill_notes(t_cg: pd.DataFrame, cg_method: str, t_bl: pd.DataFrame, bill_method: str,
                 paired: pd.DataFrame, steer: dict) -> list[str]:
    def mae(t, w, h, m):
        r = t[(t["window"] == w) & (t["h"] == h) & (t["model"] == m)]
        return float(r["mae"].iloc[0]) if len(r) else float("nan")

    cgl, bll = M.label(cg_method), M.label(bill_method)
    edge = [h for h in M.HORIZONS if mae(t_cg, "OOS", h, cg_method) < (1 - M.EDGE_BAND) * mae(t_cg, "OOS", h, "smly")]
    lose_bl = [h for h in M.HORIZONS if mae(t_bl, "OOS", h, bill_method) > mae(t_bl, "OOS", h, "bill_flat")]
    out = [f"CGNCR model ({cgl}, chosen on PRE) beats same month last year by more than "
           f"{M.EDGE_BAND:.0%} out of sample at: {_hlist(edge)}.",
           f"Bill model ({bll}, chosen on PRE) loses to a flat stock out of sample at: {_hlist(lose_bl)}."]
    if not paired.empty:
        beats = {w: [int(r["h"]) for _, r in paired.iterrows()
                     if r["window"] == w and r[M.OFFICIAL] < r[cg_method]] for w in ("PRE", "OOS")}
        out.append(f"{M.label(M.OFFICIAL)} beats the CGNCR model at: {_hlist(beats['PRE'])} (PRE), "
                   f"{_hlist(beats['OOS'])} (OOS).")
    if steer["cgncr_source_in_fy"] == "bottom_up_remit":
        src_in = (f"bottom-up closed to remit "
                  f"(+£{steer.get('cgncr_live_pin_gap_bn', 0):.1f}bn on rest-of-FY)")
    elif steer["cgncr_source_in_fy"] == "bottom_up":
        src_in = "bottom-up components"
    elif steer["cgncr_source_in_fy"] == "plan":
        src_in = M.label(M.OFFICIAL)
    else:
        src_in = cgl
    out.append(f"Rule: a model replaces the default (OBR / remit CGNCR, flat stock for bills) only if it "
               f"beats it by more than {M.EDGE_BAND:.0%} on PRE at the horizon. Live in-FY YE still "
               f"closes to the published remit after Budget / revision. Now: CGNCR to March = {src_in} "
               f"({steer['months_left_in_fy']}m); bills after March = "
               f"{'model' if steer['bill_after_fy'] == 'model' else 'held flat'}.")
    if "cgncr_bottomup_ship" in steer:
        out.append(
            f"Bottom-up CGNCR PRE mean MAE {steer.get('cgncr_bottomup_pre_mean')} vs production "
            f"{steer.get('cgncr_bottomup_prod_pre_mean')} (edge {steer.get('cgncr_bottomup_edge')}); "
            f"{'SHIP' if steer['cgncr_bottomup_ship'] else 'NO-SHIP'}.")
    return out


def _anchor_info(anchors: pd.DataFrame) -> dict:
    a = anchors.copy()
    a["fy_start"] = a["fy"].str[:4].astype(int)
    o = a[a["kind"] == "outturn"].sort_values("fy_start").iloc[-1]
    p = a[a["kind"] == "plan"].sort_values("fy_start").iloc[-1]
    return {"o_key": o["fy"], "p_key": p["fy"], "o_fy": int(o["fy_start"]), "p_fy": int(p["fy_start"]),
            "published": sorted(a.loc[a["kind"].isin(["outturn", "plan"]), "published"].dropna().unique())}


def run(refresh: bool = False) -> dict:
    ons_ok = src.fetch_ons(refresh=refresh)
    ons = src.ons_monthly()
    anchors = src.dmr_anchors()
    vint = src.remit_vintages()
    hist = M.build_history()

    bt_cg = M.backtest_cgncr(hist["cash_requirement_bn"], vint)
    t_cg = M.score_table(bt_cg)
    cg_method = M.choose(t_cg, M.CGNCR_MODELS)
    paired = M.official_paired(bt_cg, cg_method)
    bt_bl = M.backtest_bills(hist, cg_method)
    t_bl = M.score_table(bt_bl)
    bill_method = M.choose(t_bl, M.BILL_MODELS + M.BILL_BENCH)
    bt_ns = M.backtest_cgncr(hist["of_nsandi_bn"])
    t_ns = M.score_table(bt_ns)
    ns_method = M.choose(t_ns, M.CGNCR_MODELS + M.CGNCR_BENCH)

    from sovereign_cash.uk import bottom_up_macro as BU

    bu_decision = BU.evaluate()
    bu_comp = BU.components()
    from sovereign_cash.cash_season import forward_months as _fwd_m
    bu_path = BU.forecast(bu_comp, hist.index.max(), h=_fwd_m(hist.index.max()))
    tenders = src.bill_tenders()
    tender_net = src.bill_tender_net(tenders)
    d22 = src.bills_stock()["bill_stock"].diff()
    tender_new = (d22[d22.index > hist.index.max()].dropna()
                  .combine_first(tender_net[tender_net.index > hist.index.max()]))
    fc, dmo, steer = M.forward(
        hist, anchors, cg_method, bill_method, ns_method,
        scores={"paired": paired, "t_bl": t_bl, "t_ns": t_ns}, tender_net=tender_new,
        bottomup=bu_path, use_bottomup=bool(bu_decision["ship"]))
    steer["cgncr_bottomup_ship"] = bool(bu_decision["ship"])
    steer["cgncr_bottomup_edge"] = bu_decision.get("edge")
    steer["cgncr_bottomup_pre_mean"] = bu_decision.get("bu_pre_mean")
    steer["cgncr_bottomup_prod_pre_mean"] = bu_decision.get("prod_pre_mean")
    steer["cgncr_bottomup_spec"] = bu_decision.get("spec")
    level, anchor_checks = M.anchor_buffer(hist, anchors)
    hist = hist.copy()
    hist["buffer_level_implied_bn"] = level
    hist["buffer_level_dmo_basis_bn"] = dmo["buffer_level_dmo_basis_bn"].reindex(hist.index)
    for a in anchor_checks:
        hist.loc[a["month"], "buffer_level_bn"] = a["buffer_anchor"]
        hist.loc[a["month"], "dmo_net_cash_bn"] = a["net_cash"]

    mm, mm_stats = M.money_market_panel(hist)

    df = pd.concat([hist, fc], axis=0, sort=False).sort_index()
    df.index.name = "month"
    df = df.reset_index()
    for c in contract.REQUIRED + contract.OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan
    for c in ("cgncr_from_plan", "gilts_assumed", "bills_assumed", "net_bills_pinned"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    df["is_forecast"] = df["is_forecast"].astype(bool)
    resid = contract.validate(df)

    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MONTHLY, index=False)
    score = pd.concat([t_cg.assign(target="cgncr_cum"), t_bl.assign(target="bill_stock_level"),
                       t_ns.assign(target="nsandi_cum")])
    score.to_csv(SCORE, index=False)
    if not mm.empty:
        mm.rename_axis("month").reset_index().to_parquet(MM, index=False)

    # Score list: chosen model vs best benchmark per window × h.
    def score_rows(t, model, benches, target):
        rows = []
        for (w, h), gg in t.groupby(["window", "h"]):
            gg = gg.set_index("model")
            if model not in gg.index:
                continue
            for b in benches:
                if b in gg.index:
                    rows.append({"target": target, "window": w, "h": int(h), "model_name": model,
                                 "model": float(gg.at[model, "mae"]), "benchmark_name": b,
                                 "benchmark": float(gg.at[b, "mae"]), "n": int(gg.at[model, "n"])})
        return rows

    score_list = (score_rows(t_cg, cg_method, M.CGNCR_BENCH, "cum CGNCR(ex) MAE £bn")
                  + score_rows(t_bl, bill_method, [b for b in M.BILL_BENCH + M.BILL_MODELS if b != bill_method],
                               "T-bill stock level MAE £bn"))
    ns_score = score_rows(t_ns, ns_method, [m for m in M.CGNCR_MODELS + M.CGNCR_BENCH if m != ns_method],
                          "cum NS&I net financing MAE £bn")

    ai = _anchor_info(anchors)
    a = anchors.set_index(["fy", "kind"])
    o_row = a.loc[(ai["o_key"], "outturn")]
    p_row = a.loc[(ai["p_key"], "plan")]
    o_months = pd.date_range(pd.Timestamp(ai["o_fy"], 4, 1), periods=12, freq="MS")
    p_months = pd.date_range(pd.Timestamp(ai["p_fy"], 4, 1), periods=12, freq="MS")
    fyo = hist.reindex(o_months)
    b = src.bills_stock()
    bk = ons["tbill_level"].diff()
    build = pd.Timestamp.now().normalize()
    apf = src._apf_ops()
    apf_last = apf["op_dt"].max() if len(apf) else None
    sonia_last = (pd.read_csv(src.SONIA_CSV, usecols=["date"], parse_dates=["date"])["date"].max()
                  if src.SONIA_CSV.exists() else None)
    vmiss, vworst = _vintage_outturn_check(hist, vint)
    tcmp = (tender_net - b["bill_stock"].diff()).loc["2023":].dropna()
    checks = {
        "o_fy": _fy_label(ai["o_fy"]), "p_fy": _fy_label(ai["p_fy"]),
        "cgncr_fy": float(fyo["cash_requirement_bn"].sum()), "cgncr_fy_dmo": float(o_row["cgncr_ex"]),
        "ruuw_minus_m98r_fy": float((fyo["cgncr_ruuw_bn"] - fyo["cash_requirement_bn"]).sum()),
        "nsandi_fy": float(fyo["of_nsandi_bn"].sum()),
        "nsandi_fy_dmo": float(o_row["nsandi_net"] + o_row["nsandi_green"]),
        "gap_2022": float(hist.loc["2022":, "of_unexplained_bn"].abs().max()),
        "bills_last": b.index.max(),
        "bills_vs_bkpj": float((b["bill_stock"].diff() - bk).loc["2015":].abs().max()),
        "tender_last": str(tenders["issue_dt"].max().date()) if len(tenders) else "missing",
        "tender_mae": float(tcmp.abs().mean()) if len(tcmp) else None,
        "tender_pinned": ", ".join(f"{m:%b-%y}" for m in tender_new.index),
        "gilt_sales_fy": float(fyo["bond_gross_bn"].sum()), "gilt_sales_fy_dmo": float(o_row["gilt_sales"]),
        "red_fy": float(fyo["bond_redemption_bn"].sum()), "red_fy_dmo": float(o_row["gilt_redemptions"]),
        "red_pfy": float(df.loc[df["month"].isin(p_months), "bond_redemption_bn"].sum()),
        "red_pfy_dmo": float(p_row["gilt_redemptions"]),
        "apf_last": str(apf_last.date()) if apf_last is not None else "missing",
        "apf_age_days": int((build - apf_last).days) if apf_last is not None else None,
        "apf_red_12m": float(fc["bond_redemption_cb_bn"].sum()),
        "remit_asof": steer["gilt_remit_asof"],
        "remit_age_days": int((build - pd.Timestamp(steer["gilt_remit_asof"])).days),
        "remit_total": steer["gilt_remit_total_bn"], "remit_plan": float(p_row["gilt_sales"]),
        "remit_total_ok": abs(steer["gilt_remit_total_bn"] - float(p_row["gilt_sales"])) < 0.05,
        "gilt_fy_vs_plan": steer["gilt_sales_fy_bn"] - steer["gilt_sales_plan_bn"],
        "arith_max_miss": _arith_miss(anchors),
        "anchor_published": ", ".join(ai["published"]),
        "vintage_outturn_miss": vmiss, "vintage_outturn_worst": vworst, "n_vintages": int(len(vint)),
        "vint_span": f"{_fy_label(int(vint['fy_start'].min()))}..{_fy_label(int(vint['fy_start'].max()))}",
        "sonia_last": str(sonia_last.date()) if sonia_last is not None else "missing",
        "sonia_age_days": int((build - sonia_last).days) if sonia_last is not None else None,
    }

    of = hist.loc["2010":]
    first = fc.index[0]
    cal = M.redemption_calendar(hist.index.max(), fc.index)
    e = steer
    remit_pm = e["gilt_remit_remaining_bn"] / max(e["gilt_remit_months_left"], 1)
    in_fy = ~fc["gilts_assumed"]
    fwd = {
        "window": f"{first:%Y-%m}..{fc.index[-1]:%Y-%m}",
        "fy_window": f"{first:%Y-%m}..{fc.index[in_fy][-1]:%Y-%m}" if in_fy.any() else "",
        "cgncr_model": cg_method,
        "bill_model": bill_method,
        "nsandi_model": ns_method,
        "cgncr_12m_bn": float(fc["cash_requirement_bn"].sum()),
        "cgncr_12m_model_bn": float(fc["cash_requirement_model_bn"].sum()),
        "cgncr_12m_smly_bn": float(fc["cgncr_smly_bn"].sum()),
        "cgncr_12m_official_profile_bn": float(fc["cash_requirement_fc_official_bn"].sum()),
        "cgncr_12m_bottomup_bn": float(fc["cash_requirement_bottomup_bn"].sum()),
        "cgncr_bottomup_ship": bool(bu_decision["ship"]),
        "cgncr_bottomup_edge": bu_decision.get("edge"),
        "cgncr_bottomup_pre": bu_decision.get("bu_pre"),
        "cgncr_bottomup_prod_pre": bu_decision.get("prod_pre"),
        "cgncr_bottomup_oos": bu_decision.get("bu_oos"),
        "cgncr_bottomup_prod_oos": bu_decision.get("prod_oos"),
        "cgncr_bottomup_spec": bu_decision.get("spec"),
        "cgncr_bottomup_reason": bu_decision.get("reason"),
        "gilt_gross_12m_bn": float(fc["bond_gross_bn"].sum()),
        "gilt_redemptions_12m_bn": float(fc["bond_redemption_bn"].sum()),
        "cb_held_redemptions_12m_bn": float(fc["bond_redemption_cb_bn"].sum()),
        "gilt_net_12m_bn": float(fc["net_bonds_bn"].sum()),
        "private_net_bonds_12m_bn": float(fc["net_bonds_private_bn"].sum()),
        "private_net_bonds_fy_bn": float(fc.loc[in_fy, "net_bonds_private_bn"].sum()),
        "private_net_bonds_assumed_bn": float(fc.loc[~in_fy, "net_bonds_private_bn"].sum()),
        "gilt_gross_assumed_bn": float(fc.loc[~in_fy, "bond_gross_bn"].sum()),
        "remit_remaining_bn": e["gilt_remit_remaining_bn"],
        "remit_window": e["gilt_remit_window"],
        "remit_per_month_bn": remit_pm,
        "nsandi_12m_bn": float(fc["of_nsandi_bn"].sum()),
        "remit_other_12m_bn": float(fc["remit_other_fin_bn"].sum()),
        "net_bills_12m_bn": float(fc["net_bills_bn"].sum()),
        "bill_stock_last_obs_bn": float(hist["bill_stock_bn"].iloc[-1]),
        "bill_stock_implied_end_bn": float(fc["bill_stock_implied_bn"].iloc[-1]),
        "buffer_change_12m_bn": float(fc["buffer_change_bn"].sum()),
        "redemption_calendar": cal,
        "end_march": steer,
        "monthly": [{"month": str(m.date()), "cgncr": r["cash_requirement_bn"],
                     "cgncr_model": r["cash_requirement_model_bn"],
                     "cgncr_official": r["cash_requirement_fc_official_bn"],
                     "cgncr_bottomup": r["cash_requirement_bottomup_bn"],
                     "gilt_gross": r["bond_gross_bn"],
                     "redemptions": r["bond_redemption_bn"], "redemptions_cb": r["bond_redemption_cb_bn"],
                     "gilt_net": r["net_bonds_bn"], "gilt_net_private": r["net_bonds_private_bn"],
                     "nsandi": r["of_nsandi_bn"], "remit_other": r["remit_other_fin_bn"],
                     "net_bills": r["net_bills_bn"], "net_bills_model": r["net_bills_model_bn"],
                     "net_bills_pinned": bool(r["net_bills_pinned"]),
                     "bill_stock_implied": r["bill_stock_implied_bn"], "buffer_change": r["buffer_change_bn"],
                     "buffer_level": r["buffer_level_implied_bn"], "assumed": bool(r["gilts_assumed"])}
                    for m, r in fc.iterrows()],
    }
    from sovereign_cash.uk import dashboard

    gp = e["next_revision_gap_parts_bn"]
    skill = _skill_notes(t_cg, cg_method, t_bl, bill_method, paired, steer)
    worst_anchor = max(anchor_checks, key=lambda r: abs(r["miss_bn"])) if anchor_checks else None
    closure = {
        "other_fin_mean_abs_bn_2010": float(of["other_fin_bn"].abs().mean()),
        "unexplained_mean_abs_bn_2010": float(of["of_unexplained_bn"].abs().mean()),
        "unexplained_max_abs_bn_2010": float(of["of_unexplained_bn"].abs().max()),
        "unexplained_max_abs_bn_2022": checks["gap_2022"],
        "worst_anchor_fy": _fy_label(worst_anchor["month"].year - 1) if worst_anchor else None,
        "worst_anchor_miss_bn": worst_anchor["miss_bn"] if worst_anchor else None,
    }
    if e["cgncr_source_in_fy"] == "bottom_up_remit":
        cg_src = "bottom-up closed to remit"
    elif e["cgncr_source_in_fy"] == "bottom_up":
        cg_src = "bottom-up components"
    elif e["cgncr_source_in_fy"] == "plan":
        cg_src = M.label(M.OFFICIAL)
    else:
        cg_src = M.label(cg_method)
    notes = [
        f"Next-revision gap {g(e['next_revision_gap_bn'], plus=True)} (± {g(e['gap_err_bn'])} out-of-sample "
        f"error): {dashboard.gap_parts_text(gp)}. NS&I at remit: "
        f"{g(e['next_revision_gap_nsandi_at_remit_bn'], plus=True)}; CGNCR on the model instead of {cg_src}: "
        f"{g(e['next_revision_gap_cgncr_model_bn'], plus=True)}.",
        f"Assumption: cash-management bills at end-March held at the last FY-end level "
        f"({g(e['cash_mgmt_bills_end_march_assumed_bn'])}); the DMO publishes the split at FY-end only.",
        f"Assumption: months after March use forward-plan gilt nominal × realised cash/nominal "
        f"({g(fwd['gilt_gross_assumed_bn'])} gross); no remit is published for them.",
        f"Bill path to March is solved so DMO net cash meets the remit's {g(e['net_cash_plan_end_march_bn'])}; "
        f"the solve is spread evenly over the free months"
        + (f"; {checks['tender_pinned']} pinned from weekly tenders." if checks["tender_pinned"] else "."),
        "Buffer = DMO net cash + cash-management bills; monthly proxy is CG net deposits with banks "
        "(ONS ANSZ + ANTB), pinned to the latest DMR end-March outturn.",
        "Gilt cash = clean price × nominal × index ratio (no accrued). Redemptions net out CRND/DMO holdings; "
        "APF-held redemptions are Exchequer cash paid to the Bank, so private net supply adds them back.",
        f"Covid months {M.COVID_EST[0]:%b-%Y}..{M.COVID_EST[1]:%b-%Y} are excluded from estimation and scoring.",
    ]
    summary = {
        "country": "UK",
        "ccy": "GBP",
        "asof": str(hist.index.max().date()),
        "sources": _sources(hist, ons_ok, ons.attrs["meta"], checks),
        "buffer": ("Exchequer liquid position managed by the DMO: DMO net cash position + "
                   "cash-management T-bill stock. Monthly proxy = CG net deposits with MFIs, "
                   "−(ANSZ+ANTB), pinned to the DMR end-March net cash outturn + (total D2.2G stock "
                   "− debt-financing stock). The forward buffer is on the DMO basis (remit items only) and "
                   "starts from the FY-to-date DMO-basis level."),
        "identity_resid_bn": float(resid),
        "closure": closure,
        "other_fin": {
            "definition": "residual of the identity; itemised in of_* columns",
            "forward": "NS&I (model chosen on PRE) + remit other financing − Sukuk − secondary-purchase financing",
            "mean_abs_bn_2010": closure["other_fin_mean_abs_bn_2010"],
            "max_abs_bn_2010": float(of["other_fin_bn"].abs().max()),
            "unexplained_mean_abs_bn_2010": closure["unexplained_mean_abs_bn_2010"],
            "unexplained_max_abs_bn_2010": closure["unexplained_max_abs_bn_2010"],
            "unexplained_max_abs_bn_2022": checks["gap_2022"],
            "fy_sums_bn": {str(k): {c: float(v) for c, v in r.items()} for k, r in
                           hist.loc["2015-04":].groupby(M.fy_of(hist.loc["2015-04":].index))
                           [["other_fin_bn", "of_nsandi_bn", "of_dep_by_others_bn", "of_nram_nr_bn",
                             "of_gilt_vs_ons_bn", "of_bill_vs_ons_bn", "of_unexplained_bn"]].sum().iterrows()},
        },
        "buffer_anchor_checks": anchor_checks,
        "score": score_list,
        "nsandi_score": ns_score,
        "official_benchmark": paired.to_dict("records") if not paired.empty else [],
        "skill_notes": skill,
        "score_note": (f"Walk-forward monthly origins. PRE = {M.PRE[0]:%Y-%m}..{M.PRE[1]:%Y-%m}; OOS = "
                       f"{M.OOS_START:%Y-%m}+; COVID = any origin whose targets or trailing {M.INPUT_MONTHS}m "
                       f"inputs touch the Covid months. Models are chosen on PRE only; the bill backtest uses "
                       f"realised gilt cash (calendar known). The remit-plan benchmark is scored where a "
                       f"published vintage covers every target month."),
        "money_markets": {"stats": mm_stats,
                          "note": "descriptive only; 3m T-bill tender yield − SONIA, monthly means"},
        "forward": fwd,
        "notes": notes,
    }
    summary = _clean(summary)
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    dashboard.build(df, summary, bt_cg, bt_bl, mm)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-fetch ONS PUSF series")
    a = ap.parse_args()
    s = run(refresh=a.refresh)
    f = s["forward"]
    e = f["end_march"]
    print(f"UK cash asof {s['asof']}  identity resid {s['identity_resid_bn']}bn")
    print(f"  CGNCR 12m {f['cgncr_12m_bn']} (in-FY source {e['cgncr_source_in_fy']}; model "
          f"{f['cgncr_12m_model_bn']}, official-profile {f['cgncr_12m_official_profile_bn']}, smly {f['cgncr_12m_smly_bn']})")
    print(f"  gilt net 12m {f['gilt_net_12m_bn']}  net bills {f['net_bills_12m_bn']}  "
          f"T-bill stock {f['bill_stock_last_obs_bn']} -> {f['bill_stock_implied_end_bn']} (after March: {e['bill_after_fy']})")
    print(f"  private net gilts 12m {f['private_net_bonds_12m_bn']} (to March {f['private_net_bonds_fy_bn']}, "
          f"assumed after {f['private_net_bonds_assumed_bn']})")
    print(f"  end-March {e['fy']}: net cash {e['net_cash_model_end_march_bn']} (plan "
          f"{e['net_cash_plan_end_march_bn']}); next revision gap {e['next_revision_gap_bn']} ± {e['gap_err_bn']} "
          f"{e['next_revision_gap_parts_bn']}; pinned {e['bill_pinned_months']}")


if __name__ == "__main__":
    main()
