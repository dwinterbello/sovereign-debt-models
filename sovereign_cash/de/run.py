"""Germany federal cash framework: history identity, deficit scores, Bubill/placement path, contract outputs, dashboard.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.de.run            # cached inputs
  PYTHONPATH=. .venv/bin/python -m sovereign_cash.de.run --refresh  # re-fetch BMF / Finanzagentur / Destatis
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sovereign_cash.contract import DERIVED, OPTIONAL, REQUIRED, fmt_bn, fmt_num
from sovereign_cash.core import cash_framework as C
from sovereign_cash.de import dashboard as D
from sovereign_cash.de import fetch as F
from sovereign_cash.de import model as M

EXTRA = ["cash_requirement_core_bn", "cash_requirement_core_lo_bn", "cash_requirement_core_hi_bn",
         "cash_requirement_sv_bn", "cash_requirement_sv_infrastruktur_bn", "cash_requirement_sv_bundeswehr_bn",
         "cash_requirement_sv_other_bn", "cash_requirement_bottomup_bn", "cash_requirement_core_bottomup_bn",
         "bond_redemption_cb_bn", "private_net_bonds_bn",
         "bubill_gross_bn", "bubill_redemption_bn", "net_bubills_bn", "mm_budget_gross_bn",
         "mm_budget_redemption_bn", "net_mm_budget_bn", "mm_budget_stock_bn", "other_fin_debt_bn",
         "other_fin_gap_bn", "coupon_stock_bn", "core_kassenmittel_bn", "fa_debt_identity_bn",
         "buffer_level_plan_bn", "cash_observed", "coupon_source", "fc_source"]
LANDS_ON = ("net money-market placements (Bubills follow the published calendar; after it "
            "hist α+β·need demeaned to hold the calendar-end stock)")
BUBA_ZERO_RATE_FROM = pd.Timestamp("2023-10-01")
TAX_PREPAY_MONTHS = (3, 6, 9, 12)
CB_SHARE_FLAG = 0.6
FUND_ROWS = [("core", "Core budget"), ("infrastruktur", "Infrastruktur und Klimaneutralität"),
             ("bundeswehr", "Bundeswehr"), ("other", "Other special funds (FMS, ITF, WSF, RSF)")]
MODEL_LABEL = {"smly": "same month last year", "prevyear_3": "prior-year outturn", "target_3": "plan, no haircut",
               "planhc_3": "plan × haircut (not PRE-gated)", "seas_add_3": "trailing level + profile",
               "gap0": "identity, gap 0", "gapseas": "identity, seasonal gap", "flat": "flat placements",
               "sx_level_exog": "component SARIMAX + HICP/NGDP (interest from ISIN book)"}


def eur(v, **kw) -> str:
    return fmt_bn(v, "€", **kw)


def mon(t) -> str:
    return f"{pd.Timestamp(t):%b-%Y}"


def choose_gap(bsc: pd.DataFrame) -> tuple[str, str]:
    pre = bsc[(bsc["window"] == "PRE") & bsc["model"].isin(["gap0", "gapseas"])]
    n = int(pre.loc[pre["h"] == 12, "n"].max()) if len(pre.loc[pre["h"] == 12]) else 0
    if n < C.MIN_PRE:
        return "gap0", f"PRE has {n} h=12 windows (< {C.MIN_PRE}), too few to choose; gap 0 by default"
    m = pre[pre["common"]].groupby("model")["mae"].mean()
    return m.idxmin(), f"lowest mean PRE placement MAE ({', '.join(f'{k} {v:.1f}' for k, v in m.items())})"


def vs_flat(bsc: pd.DataFrame, m: str) -> str:
    """Where the placement path beats / loses to flat placements, generated from the score table."""
    win, lose = [], []
    for (w, h), g in bsc[bsc["window"].isin(["PRE", "OOS"])].groupby(["window", "h"]):
        g = g.set_index("model")["mae"]
        if m in g and "flat" in g:
            (win if g[m] < g["flat"] else lose).append((w, int(h), g[m], g["flat"]))

    def fmt(xs):
        out = []
        for w in ("PRE", "OOS"):
            hs = [x for x in xs if x[0] == w]
            if hs:
                out.append(f"{w} h={', '.join(str(x[1]) for x in hs)}")
        return "; ".join(out)
    txt = f"beats flat placements at {fmt(win)}" if win else "beats flat placements at no horizon"
    if lose:
        worst = max(lose, key=lambda x: (x[0] == "OOS", x[1]))
        txt += (f"; loses at {fmt(lose)} ({worst[0]} h={worst[1]}: MAE {worst[2]:.1f} vs {worst[3]:.1f})")
    return txt


def _span(idx) -> str:
    idx = pd.DatetimeIndex(idx).dropna()
    return f"{idx.min():%Y-%m} → {idx.max():%Y-%m} ({len(idx)} months)" if len(idx) else "none"


def _fig_destatis(h: pd.DataFrame, des: pd.DataFrame | None) -> tuple[go.Figure | None, dict]:
    """Destatis Extrahaushalte YTD deficit vs YTD borrowing booked to the special funds (deficit +)."""
    if des is None or des.empty or "extra_fin_balance" not in des:
        return None, {}
    ex = -des["extra_fin_balance"]
    sv = h["cash_requirement_sv_bn"]
    sv_ytd = sv.groupby(sv.index.year).cumsum().reindex(ex.index)
    ex_m = C.monthly_from_ytd(ex)
    both = pd.concat([ex_m, sv], axis=1, keys=["ex", "sv"], sort=True).dropna()
    r = float(both["ex"].corr(both["sv"])) if len(both) > 3 else np.nan
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex.index, y=ex, name="Destatis Extrahaushalte, YTD deficit", line=dict(color=D.PURPLE, width=1.6)))
    fig.add_trace(go.Scatter(x=sv_ytd.index, y=sv_ytd, name="borrowing booked to the special funds, YTD",
                             line=dict(color=D.ORANGE, width=1.6, dash="dot")))
    D._style(fig, title="Special funds cross-check: Destatis extra-budget deficit vs fund borrowing (YTD)", ylab="EUR bn")
    info = {"ex_ytd": float(ex.iloc[-1]), "ex_month": ex.index[-1], "corr": r,
            "corr_span": (both.index.min(), both.index.max()) if len(both) else None}
    return fig, info


def matched_oos(bt: pd.DataFrame, model: str, others: list[str], h: int = 12) -> dict:
    """OOS MAE at horizon h on the production model's own origins (honest like-for-like)."""
    b = bt[(bt["h"] == h)].assign(window=C.window_of(bt[bt["h"] == h]))
    b = b[b["window"] == "OOS"]
    piv = b.assign(err=(b["fc"] - b["actual"]).abs()).pivot_table(index="origin", columns="model", values="err")
    if model not in piv:
        return {}
    keep = [m for m in dict.fromkeys([model] + others) if m in piv]
    p = piv[keep].dropna()
    return {"h": h, "n": int(len(p)), "origins": f"{p.index.min():%Y-%m} → {p.index.max():%Y-%m}" if len(p) else "",
            "mae": {m: round(float(p[m].mean()), 2) for m in keep}}


def redemption_calendar(ops: pd.DataFrame, months: pd.DatetimeIndex, r_ratio: float) -> list[dict]:
    coup = ops[ops["sec_type"].astype(str).isin(["Nominal", "Linker"])]
    st = coup.groupby(coup["isin"].astype(str)).agg(amt=("amount_issued", "sum"), mat=("maturity_dt", "first"),
                                                     desc=("sec_desc_long", "first"), cpn=("coupon", "first"))
    for isin, (mat, amt) in M.PRE_TAPE.items():
        if isin in st.index:
            st.at[isin, "amt"] += amt * 1000
    st["mat"] = pd.to_datetime(st["mat"])
    st = st[(st["mat"] >= months.min()) & (st["mat"] <= months.max() + pd.offsets.MonthEnd(0))].sort_values("mat")
    return [{"date": f"{r['mat']:%d-%b-%Y}", "isin": i, "month": r["mat"].to_period("M").to_timestamp(),
             "line": f"{r['desc']} {r['cpn']:g}% {r['mat']:%d-%b-%Y}", "bn": round(r["amt"] / 1000 * r_ratio, 1)}
            for i, r in st.iterrows()]


def run(refresh: bool = False) -> dict:
    if refresh:
        F.fetch_all(refresh=True, verbose=True)
    h, meta = M.build_history()
    core = meta["core"]
    cash = meta["cash_full"].dropna()
    hc = M.haircuts(cash, core, h["other_fin_gap_bn"])
    targets = M.soll_by_origin(core, hc)
    bt, sc, model, why, pre_winner = M.deficit_scores(cash, targets, hc)
    bb = M.placement_backtest(h, cash, model, targets)
    bsc = C.score_table(bb)
    gap_mode, gap_why = choose_gap(bsc)
    f, info = M.forward(h, cash, model, gap_mode, core, hc)

    cols = list(dict.fromkeys(
        list(REQUIRED[1:]) + [c for c in OPTIONAL if c in h or c in f] + EXTRA))
    frame = pd.concat([h, f], axis=0)
    frame = frame[[c for c in cols if c in frame.columns]].rename_axis("month").reset_index()
    frame["is_forecast"] = frame["is_forecast"].astype(bool)
    frame["cash_observed"] = frame["cash_observed"].fillna(False).astype(bool)

    bench = "smly"
    score = C.score_records(sc, model, bench)
    # planhc runs on fewer origins than the common sample: compare it to the benchmark on its own origins
    btw = bt.assign(window=C.window_of(bt), err=(bt["fc"] - bt["actual"]).abs())
    for r in score:
        g = btw[(btw["window"] == r["window"]) & (btw["h"] == r["h"])].pivot_table(index="origin", columns="model", values="err")
        if model in g and bench in g:
            p = g[[model, bench]].dropna()
            r.update(model=round(float(p[model].mean()), 2), benchmark=round(float(p[bench].mean()), 2), n=int(len(p)),
                     benchmark_sample=f"{model}'s own origins")
    bscore = C.score_records(bsc, gap_mode, "flat")

    last_cash = cash.index.max()
    last_fin = h.index.max()
    yr = last_cash.year
    fc12 = f.loc[f.index > last_cash]
    plan = info["plan"]
    tg = {yr: info["core_target_year_bn"], **{y: plan[y]["core"] for y in plan if y > yr}}
    by_model = {}
    for m in ["smly", "prevyear_3", "target_3", "planhc_3"]:
        p = C.forecast(cash, last_cash, 12, m, targets={**tg, "hc": info["core_haircut"]})
        if p is not None:
            by_model[m] = round(float(p.sum()), 1)
    oos = matched_oos(bt, model, [pre_winner, "smly", "target_3"])
    gap = h["other_fin_gap_bn"]
    gap_xc = gap[~gap.index.year.isin(M.COVID)]
    try:
        des = F.fetch_destatis(refresh=False)
    except Exception:  # noqa: BLE001 — cross-check only
        des = None

    def s12(c):
        return round(float(fc12[c].sum()), 1)

    # ------------------------------------------------------------ year: YTD, model, band, plan scenario
    yr_rows = frame[frame["month"].dt.year == yr].set_index("month")
    hist_yr = h[h.index.year == yr]
    ytd = {"core": float(cash[cash.index.year == yr].sum()),
           **{n: float(hist_yr[f"cash_requirement_sv_{n}_bn"].sum()) for n in M.SV_PLAN.values()},
           "other": float(hist_yr["cash_requirement_sv_other_bn"].sum())}
    fy_model = {"core": float(yr_rows["cash_requirement_core_bn"].sum()),
                **{n: float(yr_rows[f"cash_requirement_sv_{n}_bn"].sum()) for n in M.SV_PLAN.values()},
                "other": float(yr_rows["cash_requirement_sv_other_bn"].sum())}
    fy_plan = {"core": info["core_target_year_bn"], **{n: plan[yr][n] for n in M.SV_PLAN.values()},
               "other": ytd["other"]}
    plan_row = {"core": plan[yr]["core"], **{n: plan[yr][n] for n in M.SV_PLAN.values()}, "other": 0.0}
    sv_tot = float(yr_rows["cash_requirement_sv_bn"].sum())
    fy_total = float(yr_rows["cash_requirement_bn"].sum())
    core_lo = float(yr_rows["cash_requirement_core_lo_bn"].fillna(yr_rows["cash_requirement_core_bn"]).sum())
    core_hi = float(yr_rows["cash_requirement_core_hi_bn"].fillna(yr_rows["cash_requirement_core_bn"]).sum())
    fy_lo, fy_hi = fy_total - fy_model["core"] + core_lo, fy_total - fy_model["core"] + core_hi
    gap_plan = fy_total - plan[yr]["total"]
    gap_parts = {"core": fy_model["core"] - plan[yr]["core"],
                 "plan funds": sum(fy_model[n] - plan[yr][n] for n in M.SV_PLAN.values()),
                 "other funds": fy_model["other"]}

    dec = pd.Timestamp(yr, 12, 1)
    fdec = f.loc[dec] if dec in f.index else f.iloc[-1]
    forward = {
        "window": f"{fc12.index.min():%Y-%m} → {fc12.index.max():%Y-%m}",
        "model": model,
        "model_reason": why,
        "model_not_pre_gated": True,
        "live_core_path": (
            "bottom-up macro" if info.get("bottomup_macro_ship") else
            ("bottom-up v1" if info.get("bottomup_ship") else model)
        ),
        "cash_requirement_12m_bn": s12("cash_requirement_bn"),
        "cash_requirement_core_12m_bn": s12("cash_requirement_core_bn"),
        "cash_requirement_sv_12m_bn": s12("cash_requirement_sv_bn"),
        "cash_requirement_12m_official_bn": s12("cash_requirement_fc_official_bn"),
        "cash_requirement_core_12m_by_model_bn": by_model,
        "oos_matched_h12": oos,
        "core_haircut": info["core_haircut"], "core_haircut_years": info["core_haircut_years"],
        "core_haircut_band": info["core_haircut_band"],
        "core_haircut_table": {int(y): {"plan": round(r["plan"], 1), "outturn": round(r["outturn"], 1),
                                        "ratio": round(r["ratio"], 3),
                                        "gap": None if not np.isfinite(r["gap"]) else round(r["gap"], 1),
                                        "in_band": bool(r["in_band"]), "eligible": bool(r["eligible"])}
                               for y, r in hc.iterrows()},
        "soll_core_fin_balance_year_bn": info["core_target_year_bn"],
        "coupon_gross_12m_bn": s12("bond_gross_bn"),
        "coupon_gross_12m_by_source_bn": {k: round(float(v), 1) for k, v in
                                          fc12.groupby("coupon_source")["bond_gross_bn"].sum().items()},
        "coupon_redemptions_12m_bn": s12("bond_redemption_bn"),
        "net_bonds_12m_bn": s12("net_bonds_bn"),
        "cb_held_redemptions_12m_bn": s12("bond_redemption_cb_bn"),
        "cb_held_redemptions_modelled": True,
        "private_net_bonds_12m_bn": s12("private_net_bonds_bn"),
        "net_bills_12m_bn": s12("net_bills_bn"),
        "net_bubills_12m_bn": s12("net_bubills_bn"),
        "bubill_stock_last_actual_bn": round(float(h["bill_stock_bn"].iloc[-1]), 1),
        "bubill_stock_last_actual_month": f"{last_fin:%Y-%m}",
        "bubill_stock_end_year_bn": round(float(fdec["bill_stock_implied_bn"]), 1),
        "bubill_stock_implied_end_bn": round(float(f["bill_stock_implied_bn"].iloc[-1]), 1),
        "net_placements_last_bn": round(float(h["buffer_level_bn"].iloc[-1]), 1),
        "net_placements_end_year_bn": round(float(fdec["buffer_level_implied_bn"]), 1),
        "net_placements_end_year_plan_bn": round(float(fdec["buffer_level_plan_bn"]), 1),
        "net_placements_implied_end_bn": round(float(f["buffer_level_implied_bn"].iloc[-1]), 1),
        "net_placements_implied_end_plan_bn": round(float(f["buffer_level_plan_bn"].iloc[-1]), 1),
        "plan_year": yr,
        "plan_cash_requirement_bn": plan[yr]["total"],
        "plan_split_bn": {k: plan[yr][k] for k in ("core", "infrastruktur", "bundeswehr")},
        "plan_source": plan[yr]["source"] + " — total federal credit-market borrowing incl. special funds",
        "plan_next_year_bn": {k: plan[yr + 1][k] for k in ("core", "infrastruktur", "bundeswehr", "total")} if yr + 1 in plan else None,
        "cash_requirement_ytd_bn": round(sum(ytd.values()), 1),
        "ytd_through": f"{last_cash:%Y-%m}",
        "ytd_split_bn": {k: round(v, 1) for k, v in ytd.items()},
        "ytd_split_through": {"core": f"{last_cash:%Y-%m}", "special funds": info["sv_ytd_through"]},
        "model_full_year_bn": round(fy_total, 1),
        "model_full_year_lo_bn": round(fy_lo, 1),
        "model_full_year_hi_bn": round(fy_hi, 1),
        "model_full_year_core_bn": round(fy_model["core"], 1),
        "model_full_year_sv_bn": round(sv_tot, 1),
        "model_full_year_split_bn": {k: round(v, 1) for k, v in fy_model.items()},
        "plan_scenario_full_year_bn": round(sum(fy_plan.values()), 1),
        "model_minus_plan_bn": round(gap_plan, 1),
        "model_minus_plan_split_bn": {k: round(v, 1) for k, v in gap_parts.items()},
        "plan_gap_lands_on": LANDS_ON,
        **{k: v for k, v in info.items() if k not in ("origin_cash", "plan", "core_target_year_bn",
                                                     "core_haircut", "core_haircut_years", "core_haircut_band")},
    }

    # ------------------------------------------------------------ text pieces, all generated
    pre_n12 = sc[(sc.window == "PRE") & (sc.h == 12) & sc.common]["n"].max()
    oos_n12 = sc[(sc.window == "OOS") & (sc.h == 12) & sc.common]["n"].max()
    covid = f"{M.COVID[0]}–{M.COVID[-1]}"
    oos_txt = ", ".join(f"{k} {v:.1f}" for k, v in oos.get("mae", {}).items())
    score_note = (
        f"Core budget only (the special funds have no plan history). Walk-forward from monthly origins, cumulative "
        f"core cash requirement over the next h months. PRE = windows ending by {mon(C.PRE_END)} (common h=12 "
        f"n={int(pre_n12 or 0)}); OOS = origins from {mon(C.OOS_START)} (common h=12 n={int(oos_n12 or 0)}); windows "
        f"touching {covid} excluded. {why} OOS h=12 on {model}'s own {oos.get('n', 0)} origins "
        f"({oos.get('origins', '')}): {oos_txt}."
    )
    hc_used = [int(y) for y in info["core_haircut_years"]]
    hc_out = [int(y) for y, r in hc.iterrows() if r["in_band"] and not r["eligible"]]
    hc_out_txt = ", ".join(f"{y} (unexplained financing {eur(hc.at[y, 'gap'])})" for y in hc_out)
    lo, hi = info["core_haircut_band"]
    band_years = [int(y) for y, r in hc.iterrows() if r["in_band"] and y < yr]
    exe_f = info["sv_execution_ytd_by_fund"]
    destatis_fig, dinfo = _fig_destatis(h, des)
    mm = h[h["mm_budget_gross_bn"] > 0]
    ops = pd.read_parquet(M.OPS)
    desc = ops.drop_duplicates("isin").set_index(ops.drop_duplicates("isin")["isin"].astype(str))
    pre_tape = [f"{desc.at[i, 'sec_desc_long']} {desc.at[i, 'coupon']:g}% {pd.Timestamp(m):%d-%b-%Y} {eur(a)}"
                if i in desc.index else eur(a) for i, (m, a) in M.PRE_TAPE.items()]
    from issuance.qe_alloc import PEPP_CAP, PSPP_CAP
    fwin_all = f.loc[f.index > last_fin]
    share = fc12["bond_redemption_cb_bn"] / fc12["bond_redemption_bn"].where(fc12["bond_redemption_bn"] > 0)
    flagged = [f"{mon(m)} {v:.0%}" for m, v in share.items() if v > CB_SHARE_FLAG]
    cal_q = pd.Timestamp(info["bubill_calendar_end"])
    bub_cal_total = json.loads(M.BUBILL_CAL.read_text()).get("quarter_total_mn", 0) / 1000.0
    src12 = forward["coupon_gross_12m_by_source_bn"]

    notes = [
        f"Buffer = net money-market placements: the Finanzagentur stock 'Kassenverstärkungskredite und Geldanlagen' "
        f"with the sign flipped (negative = net cash-management borrowing). The Bundesbank pays 0% on federal deposits "
        f"from {mon(BUBA_ZERO_RATE_FROM)}, so the central-bank account is liquidity-neutral.",
        "Cash requirement = core budget (BMF Finanzierungssaldo, deficit +) + net borrowing booked to the special funds "
        "by the Finanzagentur. Fund spending is not observed monthly: cash a fund has borrowed but not yet spent sits "
        "in unexplained financing.",
        (f"Core, rest of {yr}: spending closes to its Soll {eur(info['core_spend_soll_bn'])} × {info['core_spend_ratio']:.2f}, "
         f"revenue to its Soll {eur(info['core_rev_soll_bn'])} × {info['core_rev_ratio']:.2f} "
         f"(each the mean of outturn ÷ plan over {', '.join(map(str, info['core_line_years']))}). "
         f"From {yr + 1} the draft core is scaled by the balance haircut {info['core_haircut']:.2f} "
         f"over {', '.join(map(str, hc_used))}; that band is {lo:.2f}–{hi:.2f}."
         if info.get("core_line_years") else
         f"Core forward = Soll {eur(info['core_target_year_bn'])} × haircut {info['core_haircut']:.2f} (mean outturn ÷ plan "
         f"over {', '.join(map(str, hc_used))}); band {lo:.2f}–{hi:.2f} = range over {', '.join(map(str, band_years))}.")
        + (f" Left out of the balance haircut: {hc_out_txt} — above {eur(M.HC_GAP_MAX)}, the core outturn misses cash "
           "that left the buffer." if hc_out else ""),
        (
            f"Special funds through {mon(info['sv_ytd_through'])}: "
            f"Infrastruktur {exe_f['infrastruktur']:.0%} / Bundeswehr {exe_f['bundeswehr']:.0%} of plan pace YTD; "
            f"BW profile={info.get('sv_profile', {}).get('bundeswehr')} "
            f"(scale_channel={ (info.get('sv_scale_channel') or {}).get('bundeswehr') }), "
            f"implied YE {eur((info.get('sv_implied_ye_bn') or {}).get('bundeswehr', 0))}; "
            f"infra={info.get('sv_profile', {}).get('infrastruktur')}. "
            f"Hist month shares only when the Wirtschaftsplan sits inside recent FA scale; "
            f"a defence / fund ramp (plan ≫ those years) sets YE = plan and flattens the residual — "
            f"do not freeze last year's Dec spike onto a new level. "
            f"Wirtschaftsplan is annual only — no published monthly Auszahlungsprofil. Other "
            f"funds forward 0: their {eur(ytd['other'])} this year is unscheduled repayment."
        ),
        f"Plan scenario = core Soll/draft without haircut + plan funds at the plan pace. Issuance is the same in both "
        f"scenarios, so the difference lands on placements (Dec-{yr}: model {eur(forward['net_placements_end_year_bn'])}, "
        f"plan {eur(forward['net_placements_end_year_plan_bn'])}).",
    ]
    if dinfo:
        cs = dinfo["corr_span"]
        notes.append(f"Destatis extra-budget deficit {eur(dinfo['ex_ytd'])} YTD to {mon(dinfo['ex_month'])}; its monthly "
                     f"change correlates {fmt_num(dinfo['corr'], 2)} with fund borrowing ({mon(cs[0])} → {mon(cs[1])}), so "
                     "it is a cross-check, not an input.")
    if len(mm):
        notes.append("Year-end money-market budget financing, repaid in January, counts as net bills: "
                     + ", ".join(f"{eur(v)} {mon(m)}" for m, v in mm["mm_budget_gross_bn"].items() if m.year >= yr - 3)
                     + ". Forward 0.")
    notes += [
        f"The identity closes by construction: other financing = Δ(Schuldschein + other loans) + an unexplained "
        f"remainder (sd {eur(gap_xc.std())}/month excluding {covid}). The Finanzagentur's own debt identity closes to "
        f"{eur(h['fa_debt_identity_bn'].abs().max(), dp=3)}.",
        f"Placement backtest (gap rule {MODEL_LABEL.get(gap_mode, gap_mode)}: {gap_why}): the path {vs_flat(bsc, gap_mode)}. "
        "Read the placement path as where cash lands, not as a level forecast beyond a month.",
    ]
    if meta["sv_transfers"]:
        notes.append("Debt moved between a fund and the core budget without a market transaction is not borrowing: "
                     + ", ".join(f"{t['label']} {eur(t['bn'])} ({mon(t['month'])})" for t in meta["sv_transfers"]) + ".")
    notes += [
        f"Coupons: auction tape to {pd.Timestamp(info['tape_last_op']):%d-%b-%Y}, then the published issuance calendar "
        f"to {pd.Timestamp(info['calendar_published_end']):%d-%b-%Y}, then a pattern sized on the borrowing plan; gross "
        f"× {info['gross_market_ratio']:.3f} (market share of tape volume). Redemptions = tape ladder"
        + (f" + tranches issued before the tape ({'; '.join(pre_tape)})" if pre_tape else "")
        + f" × {info['redemption_market_ratio']:.3f}. No regular buyback programme.",
        f"Bubills: tape to {pd.Timestamp(info['bubill_tape_last_op']):%d-%b-%Y}, then the published calendar "
        f"({eur(bub_cal_total)} gross to {mon(cal_q)}) × {info['bubill_gross_market_ratio']:.3f}; stock flat after.",
        f"Eurosystem-held redemptions are modelled: the ECB publishes country stock and maturity, not holdings by line, "
        f"so lines are allocated with caps of {PSPP_CAP:.0%} (PSPP) + {PEPP_CAP:.0%} (PEPP) per line, no reinvestment."
        + (f" Months where the modelled share of redemptions exceeds {CB_SHARE_FLAG:.0%} — {', '.join(flagged)} — "
           "sit near the caps and likely overstate runoff." if flagged else ""),
    ]

    mb = F.load_bmf_mb()
    sdds = F.parse_sdds_pdf(F.fetch_sdds_pdf())
    fa = meta["fa"]["stock"]
    cal, js = M.load_bubill_calendar()
    sources = [
        {"name": "BMF Monatsbericht 'Entwicklung des Bundeshaushalts' (core YTD, Soll)",
         "url": F.BMF + "Monatsberichte/", "frequency": "M", "lag_days": 25, "verified": True, "history": _span(mb.index)},
        {"name": "BMF Monatsbericht archive PDF (SDDS table)", "url": F.BMF + F.SDDS_PDF, "frequency": "M",
         "lag_days": 25, "verified": True, "history": _span(sdds.index)},
        {"name": "Finanzagentur Schuldenbericht (debt by use and instrument, gross/redemptions, placements)",
         "url": F.FA_URL, "frequency": "M", "lag_days": 30, "verified": True, "history": _span(fa.index)},
        {"name": "Finanzagentur auction tape", "url": "issuance/cache/ops_de.parquet",
         "frequency": "D", "lag_days": 1, "verified": True,
         "history": _span(ops["op_dt"].dt.to_period("M").dt.to_timestamp().unique())},
        {"name": "Finanzagentur quarterly issuance plan (Bubill calendar)", "url": info["bubill_calendar_url"],
         "frequency": "Q", "lag_days": 0, "verified": True,
         "history": f"{cal['op_dt'].min():%Y-%m-%d} → {cal['op_dt'].max():%Y-%m-%d}" if len(cal) else "none"},
        {"name": "Federal borrowing plan (core + special funds)", "url": "sovereign_cash/data/de/plan.json",
         "frequency": "A", "lag_days": 0, "verified": True, "history": ", ".join(map(str, sorted(plan)))},
        {"name": "Issuance forward calendar (published quarter, then plan-sized pattern)", "url": "issuance/forward.py",
         "frequency": "Q", "lag_days": 0, "verified": True, "history": "forward only"},
        {"name": "Eurosystem PSPP+PEPP holdings by line — modelled allocation", "url": "issuance/qe_alloc.py",
         "frequency": "M", "lag_days": 30, "verified": False, "history": "modelled, forward maturities"},
        {"name": "Destatis Bund core + Extrahaushalte YTD (cross-check)", "url": F.DESTATIS_URL, "frequency": "M",
         "lag_days": 60, "verified": des is not None, "history": _span(des.index) if des is not None else "none"},
    ]
    from sovereign_cash import scenarios as SX
    sx = SX.run("de")
    SX.save("de", sx)
    live_core = (
        "bottom-up macro (Soll open-year × hist/YTD blend; later smly×NGDP/wages + coupon×FA-stock interest)"
        if info.get("bottomup_macro_ship") else
        ("bottom-up v1 (Soll×hist then smly)" if info.get("bottomup_ship") else
         f"{model} (not PRE-gated — a priori debt-brake / special-fund pin)")
    )
    notes.append(
        "Macro scenario card: primary spending on HICP, receipts on NGDP (AMECO/WEO), interest carved from "
        "the ISIN book with +100bp from debt_repricing. Level SARIMAX does not clear PRE. "
        f"Live core path = {live_core}."
    )
    summary = {
        "country": "DE", "ccy": "EUR", "asof": f"{last_cash:%Y-%m}",
        "sources": sources,
        "buffer": ("Net money-market placements (Finanzagentur 'Kassenverstärkungskredite und Geldanlagen', sign "
                   "flipped); the Bundesbank account is liquidity-neutral at 0%. Bubills follow the published calendar, "
                   "then flat; placements flex."),
        "score": score, "score_note": score_note,
        "model": model,
        "model_reason": why,
        "model_label": ("not PRE-gated — a priori; score table only. Live core = bottom-up macro"
                        if info.get("bottomup_macro_ship") else
                        "not PRE-gated — a priori debt-brake / special-fund pin"),
        "live_core_path": live_core,
        "pre_winner": pre_winner,
        "bill_stock_score": bscore,
        "forward": forward, "notes": notes, "scenarios": sx,
    }
    C.write_outputs("de", frame, summary, DERIVED)

    # ---------------------------------------------------------------- dashboard
    F_ = forward
    win = f"{mon(fc12.index.min())} → {mon(fc12.index.max())}"
    lead = []
    src_txt = ", ".join(f"{k} {eur(v)}" for k, v in src12.items())
    lead.append((f"Private net duration supply, next 12 months ({win})", D.table([
        ["Coupon gross (market)", F_["coupon_gross_12m_bn"]],
        ["Coupon redemptions", F_["coupon_redemptions_12m_bn"]],
        ["Net bonds", F_["net_bonds_12m_bn"]],
        ["Eurosystem-held redemptions (modelled, no reinvestment)", F_["cb_held_redemptions_12m_bn"]],
        ["Private net bonds (net bonds + Eurosystem runoff)", F_["private_net_bonds_12m_bn"]],
    ], ["", "EUR bn"]) + f"<p class='sub'>Gross by source: {src_txt}. Market amounts, own-holding retentions excluded.</p>"))

    rows = []
    for k, lab in FUND_ROWS:
        thr = mon(last_cash) if k == "core" else mon(info["sv_ytd_through"])
        rng = f"{eur(core_lo)} – {eur(core_hi)}" if k == "core" else ""
        rows.append([lab, ytd[k], thr, plan_row[k], fy_model[k], rng, fy_plan[k]])
    rows.append(["Total federal", sum(ytd.values()), "", plan[yr]["total"], fy_total,
                 f"{eur(fy_lo)} – {eur(fy_hi)}", sum(fy_plan.values())])
    gp = F_["model_minus_plan_split_bn"]
    hc_rows = [[int(y), r["plan"], r["outturn"], f"{r['ratio']:.2f}", r["gap"] if np.isfinite(r["gap"]) else None,
                "centre + band" if r["eligible"] else ("band only" if r["in_band"] else "—")]
               for y, r in hc.iterrows() if y >= yr - 9]
    lead.append((f"Borrowing {yr}: year to date vs plan", D.table(
        rows, ["", "YTD", "through", f"plan {yr}", "model full year", "model range (haircut band)", "plan scenario"]) +
        f"<p class='sub'>Model − plan {eur(gap_plan, plus=True)}: core {eur(gp['core'], plus=True)}, plan funds "
        f"{eur(gp['plan funds'], plus=True)}, other funds {eur(gp['other funds'], plus=True)} (not in the plan). "
        f"A shortfall vs plan lands on {LANDS_ON}.</p>" +
        D.table(hc_rows, ["year", "core plan (final Soll)", "core outturn", "outturn ÷ plan", "unexplained financing",
                          "haircut use"])))

    lead.append(("Short-term path: Bubills and net placements (the buffer)", D.table([
        ["Bubill stock", F_["bubill_stock_last_actual_bn"], F_["bubill_stock_end_year_bn"], F_["bubill_stock_end_year_bn"],
         F_["bubill_stock_implied_end_bn"], F_["bubill_stock_implied_end_bn"]],
        ["Net money-market placements", F_["net_placements_last_bn"], F_["net_placements_end_year_bn"],
         F_["net_placements_end_year_plan_bn"], F_["net_placements_implied_end_bn"], F_["net_placements_implied_end_plan_bn"]],
    ], ["EUR bn", f"{mon(last_fin)} actual", f"Dec-{yr} model", f"Dec-{yr} plan", f"{mon(fc12.index.max())} model",
        f"{mon(fc12.index.max())} plan"]) +
        f"<p class='sub'>Bubills follow the published calendar to {mon(cal_q)} and are flat after it; net placements "
        "absorb everything else.</p>"))

    start = max(fwin_all.index.min(), pd.Timestamp.today().normalize().replace(day=1))
    fwin = fwin_all.loc[fwin_all.index >= start]
    cal_rows = [[mon(m), r["bond_redemption_bn"], r["bond_redemption_cb_bn"], r["bubill_gross_bn"], r["bubill_redemption_bn"],
                 "yes" if m.month in TAX_PREPAY_MONTHS else ""] for m, r in fwin.iterrows()]
    lines = redemption_calendar(ops, fwin.index, info["redemption_market_ratio"])
    lead.append((f"Dated calendar from {mon(start)}", D.table(
        cal_rows, ["month", "coupon redemptions", "of which Eurosystem (modelled)", "Bubill gross", "Bubill redemptions",
                   "income/corporate tax prepayment"]) +
        D.table([[x["date"], x["line"], x["bn"]] for x in lines], ["date", "maturing line", "market EUR bn"], left=2)))

    kpis = [
        ("Private net bonds, 12m", eur(F_["private_net_bonds_12m_bn"]),
         f"{win} · net bonds {eur(F_['net_bonds_12m_bn'])} + modelled Eurosystem runoff {eur(F_['cb_held_redemptions_12m_bn'])}"),
        (f"Borrowing YTD {yr}", eur(F_["cash_requirement_ytd_bn"]),
         f"core to {mon(last_cash)}, funds to {mon(info['sv_ytd_through'])} · plan {eur(plan[yr]['total'])}"),
        (f"Model full year {yr}", eur(fy_total),
         f"range {eur(fy_lo)}–{eur(fy_hi)} · vs plan {eur(gap_plan, plus=True)}"),
        ("Cash requirement, 12m", eur(F_["cash_requirement_12m_bn"]),
         f"{win} · plan scenario {eur(F_['cash_requirement_12m_official_bn'])}"),
        (f"Bubills, Dec-{yr}", eur(F_["bubill_stock_end_year_bn"]),
         f"{mon(last_fin)} actual {eur(F_['bubill_stock_last_actual_bn'])}"),
        ("Net placements", eur(F_["net_placements_last_bn"]),
         f"{mon(last_fin)} actual · Dec-{yr} model {eur(F_['net_placements_end_year_bn'])}, plan "
         f"{eur(F_['net_placements_end_year_plan_bn'])}"),
    ]
    extra = [x for x in [destatis_fig] if x is not None]
    keep = h.index.max() - pd.DateOffset(years=6)
    plc_hist = h["buffer_level_bn"]
    plc_model = pd.concat([plc_hist.iloc[-1:], f["buffer_level_implied_bn"]])
    plc_plan = pd.concat([plc_hist.iloc[-1:], f["buffer_level_plan_bn"]])
    stocks = [plc_hist[plc_hist.index >= keep].rename("net placements (actual)"),
              plc_model.rename("net placements, model scenario"), plc_plan.rename("net placements, plan scenario")]
    intro = (f"Inputs: BMF core budget to {mon(last_cash)} · Finanzagentur debt table to {mon(last_fin)} · auction tape "
             f"to {pd.Timestamp(info['tape_last_op']):%d-%b-%Y} · Bubill calendar to {mon(cal_q)}")
    D.build("de", "Germany — federal cash framework (Bubills + net placements)", intro,
            frame, summary, bill_label="Bubill", kpis=kpis, lead=lead, gap_col="other_fin_gap_bn",
            ytd_col="cash_requirement_bn", ytd_label=" — core budget + special funds",
            core_col="cash_requirement_core_bn", production=model, bill_score=bscore,
            bill_score_title="Net placement backtest — MAE (EUR bn)", bill_production=gap_mode,
            extra_stock=stocks, extra_figs=extra, scenario_html=SX.table_html("de", sx),
            stock_title="Short-term path: Bubill stock and net placements (model vs plan scenario)",
            ytd_band={"lo": "cash_requirement_core_lo_bn", "hi": "cash_requirement_core_hi_bn",
                      "base": "cash_requirement_core_bn", "name": f"{yr} core haircut band"},
            sv_cols={"Infrastruktur fund": "cash_requirement_sv_infrastruktur_bn",
                     "Bundeswehr fund": "cash_requirement_sv_bundeswehr_bn",
                     "other special funds": "cash_requirement_sv_other_bn"},
            cb_label="modelled Eurosystem runoff", score_models=[model, pre_winner, bench, "target_3"],
            ytd_years=min(5, yr - max(M.COVID)),
            model_legend="; ".join(f"{k} = {v}" for k, v in MODEL_LABEL.items()
                                   if k in {model, pre_winner, bench, "target_3"}) + ".")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    s = run(refresh=ap.parse_args().refresh)
    fw = s["forward"]
    print({k: s[k] for k in ("asof", "identity_resid_bn")},
          {k: fw[k] for k in ("model", "cash_requirement_12m_bn", "cash_requirement_core_12m_bn", "net_bonds_12m_bn",
                              "private_net_bonds_12m_bn", "net_bills_12m_bn", "bubill_stock_end_year_bn",
                              "net_placements_end_year_bn", "model_full_year_bn", "model_minus_plan_bn")})
