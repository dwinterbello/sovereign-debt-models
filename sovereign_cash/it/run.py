"""Italian central-government cash framework: history identity, fabbisogno backtest, 12-month path.

    Δ Treasury liquidity = −fabbisogno (state sector) + net BTP/CCTeu/CTZ/BTP€i/retail cash
                           + net BOT + other_fin

buffer          MEF month-end 'disponibilità liquide del Tesoro' = account at Banca d'Italia
                + money-market deployment (OPTES)
cash req.       MEF state-sector fabbisogno (final restatement where a later release gives one,
                else the provisional print); the backtest scores the first provisional print
bonds / bills   issuance tape ladder (``ladder.py``) + known events (``events.py``); redemptions from
                MEF's maturity schedule where a vintage covers the month; BOT only in bills
other_fin       residual, decomposed with Banca d'Italia Tavola 2 where it is published; forward =
                RRF loan part + other events − FX-global redemptions + baseline (backtested vs zero)
RRF             grants lower the fabbisogno; past grants (estimated from BdI EU loans at the plan ratio)
                are stripped from the forecast base, future grants added back in their month

Writes ``sovereign_cash/derived/it_cash_monthly.parquet``, ``it_cash_summary.json``,
``it_cash_backtest.parquet`` and ``sovereign_cash/it_cash_dashboard.html``.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.it.run            # cached raw files
  PYTHONPATH=. .venv/bin/python -m sovereign_cash.it.run --refresh  # re-fetch MEF / BdI
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sovereign_cash import contract
from sovereign_cash.it import events as ev
from sovereign_cash.it import ladder as lad
from sovereign_cash.it import model as M
from sovereign_cash.it import sources as src

DERIVED = contract.DERIVED
HIST_START = pd.Timestamp("2012-01-01")
BN = 1000.0


def load_inputs(refresh: bool = False) -> dict:
    rel = src.fetch_mef_fabbisogno(refresh=refresh)
    return {
        "releases": rel,
        "mef": src.mef_monthly(rel),
        "liq": src.fetch_liquidity(refresh=refresh),
        "bdi": src.fetch_bdi(refresh=refresh),
        "comp": src.fetch_composition(refresh=refresh),
        "scad": src.fetch_maturities(refresh=refresh),
        "scad_vintages": src.maturity_vintages(),
        "ops": lad.load_ops(),
    }


def history(inp: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    mef, liq, (t1, t2), comp = inp["mef"], inp["liq"], inp["bdi"], inp["comp"]
    ladder = lad.monthly_ladder(inp["ops"])
    asof = min(mef["fabb_mn"].dropna().index.max(), liq["total_mn"].dropna().index.max())
    idx = pd.date_range(HIST_START, asof, freq="MS")
    h = pd.DataFrame(index=idx)
    h["cash_requirement_bn"] = mef["fabb_mn"].reindex(idx) / BN
    h["cash_requirement_provisional_bn"] = mef["provisional_mn"].reindex(idx) / BN
    for c in ("bond_gross", "bill_gross", "bill_redemption"):
        h[f"{c}_bn"] = ladder[f"{c}_mn"].reindex(idx).fillna(0.0) / BN
    h["buyback_bn"] = ladder["buyback_mn"].reindex(idx).fillna(0.0) / BN
    h["bond_redemption_tape_bn"] = ladder["bond_maturing_mn"].reindex(idx).fillna(0.0) / BN
    h["bill_redemption_tape_bn"] = h["bill_redemption_bn"]
    # MEF's own schedule overrides the tape ladder where a vintage covers the month.
    mh = lad.mef_history(inp.get("scad_vintages")).reindex(idx)
    h["bond_redemption_mef_bn"] = mh["bond_mef_mn"] / BN
    h["bill_redemption_mef_bn"] = mh["bill_mef_mn"] / BN
    h["bond_redemption_resid_bn"] = h["bond_redemption_mef_bn"] - h["bond_redemption_tape_bn"]
    h["bill_redemption_resid_bn"] = h["bill_redemption_mef_bn"] - h["bill_redemption_tape_bn"]
    h["bond_redemption_bn"] = h["bond_redemption_mef_bn"].fillna(h["bond_redemption_tape_bn"]) + h["buyback_bn"]
    h["bill_redemption_bn"] = h["bill_redemption_mef_bn"].fillna(h["bill_redemption_tape_bn"])
    h["redemption_source"] = np.where(mh["bond_mef_mn"].notna(), "MEF schedule", "tape ladder")
    h["bond_redemption_cb_bn"] = lad.cb_redemptions(inp["ops"], idx) / BN
    h["net_bonds_bn"] = h["bond_gross_bn"] - h["bond_redemption_bn"]
    h["net_bills_bn"] = h["bill_gross_bn"] - h["bill_redemption_bn"]
    h["buffer_level_bn"] = liq["total_mn"].reindex(idx) / BN
    h["buffer_account_bn"] = liq["account_mn"].reindex(idx) / BN
    h["buffer_mm_bn"] = liq["mm_mn"].reindex(idx) / BN
    h["buffer_change_bn"] = (liq["total_mn"].diff() / BN).reindex(idx)
    h["other_fin_bn"] = (h["buffer_change_bn"] + h["cash_requirement_bn"] - h["net_bonds_bn"] - h["net_bills_bn"])
    h["bill_stock_bn"] = comp["bot_stock_mn"].reindex(idx) / BN if len(comp) else np.nan
    # Banca d'Italia decomposition of other_fin (exact where Tavola 2 is published):
    # other = liq gap + non-security GG financing + (BdI m/l securities − ladder bonds)
    #         + (BdI short securities − ladder BOT) + (state sector − general government fabbisogno)
    b = t2.reindex(idx).select_dtypes("number") / BN if len(t2) else pd.DataFrame(index=idx)
    if len(b.columns):
        h["fabb_ap_bdi_bn"] = b["fabb_ap"]
        h["fabb_ac_bdi_bn"] = (t1["fabb_ac"].reindex(idx) / BN) if len(t1) else np.nan
        h["other_fin_liq_gap_bn"] = h["buffer_change_bn"] + b["liq_change_neg"]
        h["other_fin_bdi_nonsec_bn"] = b[["deposits", "mfi_loans", "eu_loans", "other_liab"]].sum(axis=1, min_count=1)
        h["other_fin_eu_loans_bn"] = b["eu_loans"]
        h["other_fin_bond_gap_bn"] = b["mlt_securities"] - h["net_bonds_bn"]
        h["other_fin_bill_gap_bn"] = b["st_securities"] - h["net_bills_bn"]
        h["other_fin_perimeter_bn"] = h["cash_requirement_bn"] - b["fabb_ap"]
    h["is_forecast"] = False
    h = h.dropna(subset=["cash_requirement_bn", "buffer_change_bn"])
    return h, asof


def _append_level_bt(bt: pd.DataFrame, x: pd.Series, book: dict, name: str) -> pd.DataFrame:
    """Append annual-level × seasonal-shape rows (x in EUR mn) into the IT backtest."""
    from sovereign_cash.annual_macro import forecast_monthly

    y = (x / BN).dropna()
    rows = []
    last = y.index.max()
    lo, hi = M.EXCL
    for origin, tg in book.items():
        # targets are in bn; forecast then scale back to mn
        fc = forecast_monthly(y, origin, max(M.HORIZONS), {yr: v for yr, v in tg.items()})
        if fc is None:
            continue
        for h in M.HORIZONS:
            end = origin + pd.DateOffset(months=h)
            if end > last:
                continue
            act = y.loc[origin + pd.offsets.MonthBegin(1):end]
            if len(act) != h or act.isna().any():
                continue
            start = origin - pd.DateOffset(months=11)
            rows.append({"origin": origin, "h": h, "model": name,
                         "fc": float(fc.iloc[:h].sum()) * BN, "actual": float(act.sum()) * BN,
                         "touches_excl": bool(start <= hi and end >= lo)})
    if not rows:
        return bt
    return pd.concat([bt, pd.DataFrame(rows)[list(bt.columns)]], ignore_index=True)


def run_backtest(x: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Monthly candidates, annual component books, and monthly SARIMAX + macro exog.

    Annual books: tax on lag-1 NGDP, current spending on lag-1 CPI. Monthly SARIMAX
    (``sx_level_exog``): each line as a share of the lag-2 nominal GDP / HICP level, seasonal SARIMAX.
    MoM and YoY transforms lose PRE; FR's level SARIMAX ties seasonals and stays off.
    """
    from sovereign_cash.annual_macro import component_book
    from sovereign_cash.monthly_components import append_it_bt

    bt = M.backtest(x)
    for rule in ("unit", "floor"):
        book = {o: {yr: v for yr, v in tg.items()} for o, tg in component_book("it", rule=rule).items()}
        bt = _append_level_bt(bt, x, book, f"annual_comp_{rule}_3")
    bt = append_it_bt(bt, x, mode="level", with_exog=True, name="sx_level_exog")
    sc = M.score(bt)
    return bt, sc, M.choose(sc)


def other_fin_drift(hist: pd.DataFrame, months: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Forward other_fin baseline (EUR bn): backtested against zero on other_fin ex EU loans.

    Same walk-forward, windows and COVID screen as the fabbisogno; a non-zero baseline is used only if it
    beats zero on PRE mean MAE.
    """
    eu = hist["other_fin_eu_loans_bn"].fillna(0.0) if "other_fin_eu_loans_bn" in hist else 0.0
    y = ((hist["other_fin_bn"] - eu) * BN).asfreq("MS")
    bt = M.backtest(y, models=DRIFT_MODELS)
    sc = M.score(bt)
    choice = M.choose(sc)
    return M.forecast(y, len(y) - 1, choice, horizon=len(months)) / BN, sc, choice


DRIFT_MODELS = ("zero", "flat", "smly", "add_3")


def _alt_model(sc: pd.DataFrame, chosen: str) -> str:
    pre = sc[sc["window"] == "PRE"].groupby("model")["mae_bn"].mean().sort_values()
    return next(m for m in pre.index if m != chosen)


def forward(inp: dict, hist: pd.DataFrame, asof: pd.Timestamp, x: pd.Series, chosen: str,
            bt: pd.DataFrame, sc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.cash_season import forward_months as _fwd_m

    h_live = _fwd_m(asof)
    months = pd.date_range(asof + pd.offsets.MonthBegin(1), periods=h_live, freq="MS")
    xs = x.loc[:asof].asfreq("MS")
    o = len(xs) - 1
    # RRF grants lower the fabbisogno in the month they arrive. Strip the estimated grants from the base
    # months so a repeated month does not repeat last year's instalment; add known future grants back.
    eu = hist["other_fin_eu_loans_bn"] if "other_fin_eu_loans_bn" in hist else pd.Series(dtype=float)
    grants = ev.rrf_grants_history(eu).reindex(xs.index).fillna(0.0)
    xa = xs + grants * BN
    rrf = ev.rrf_split(months)
    alt = _alt_model(sc, chosen)

    def level_path(model: str, h: int) -> pd.Series | None:
        from sovereign_cash.annual_macro import component_book, forecast_monthly
        from sovereign_cash.monthly_components import live_path
        if model == "sx_level_exog":
            return live_path("it", asof, h, mode="level", with_exog=True, use_authority=True)
        if not model.startswith("annual_comp_"):
            return None
        rule = "unit" if "unit" in model else "floor"
        book = component_book("it", rule=rule)
        tg = book.get(asof)
        if tg is None and book:
            tg = book[max(book)]
        return forecast_monthly(xa / BN, asof, h, tg) if tg else None

    def fc_of(model: str) -> np.ndarray:
        path = level_path(model, h_live)
        if path is not None:
            return path.reindex(months).to_numpy() - rrf["grant_bn"].to_numpy()
        if model.startswith("annual_"):
            return M.forecast(xa, o, "smly", horizon=h_live) / BN - rrf["grant_bn"].to_numpy()
        return M.forecast(xa, o, model, horizon=h_live) / BN - rrf["grant_bn"].to_numpy()

    fc, smly, flat, alt_fc = fc_of(chosen), fc_of("smly"), fc_of("flat"), fc_of(alt)
    # Plan level × month shape for DFP years; the component model after them (below).
    n_ext = h_live
    ext_idx = months
    raw = level_path(chosen, n_ext)
    if raw is not None:
        # SARIMAX only scores to h=12; past that, calendar-repeat the last twelve months.
        if len(raw) < n_ext:
            pad = pd.Series(
                [float(raw.iloc[i % 12]) for i in range(len(raw), n_ext)],
                index=ext_idx[len(raw):],
            )
            raw = pd.concat([raw, pad])
        ext = raw.reindex(ext_idx) - ev.rrf_split(ext_idx)["grant_bn"]
    else:
        ext = (pd.Series(np.resize(M.forecast(xa, o, chosen if chosen != "sx_level_exog" else "smly") / BN,
                                   n_ext), index=ext_idx)
               - ev.rrf_split(ext_idx)["grant_bn"])
    plans = {y: v / BN for y, v in ev.plan_mn().items()}
    # After the year in progress the board runs on our component model, the path the Shock scenarios page shows
    # (cash-account lines on the macro paths, interest from the debt book, RRF grants as scheduled, the
    # fabbisogno's flows outside those lines). Before Nov 2026 the short-term model's last twelve months were
    # repeated past the DFP years (151bn a year for good). ``ext_source`` records which months came from where.
    ext_source = pd.Series("short-term model", index=ext_idx)
    after = ext_idx.year > asof.year
    # The DFP pins the year in progress only (its months are partly out-turned); later DFP years are shown
    # beside our forecast, not used for it.
    plans_all = dict(plans)
    plans = {y: v for y, v in plans.items() if y <= asof.year}
    if after.any():
        from sovereign_cash.scenarios.channels.sarimax import book_interest, our_path
        from sovereign_cash.scenarios.unit_shocks import _origin

        o_c = _origin("it")
        h_c = (ext_idx.max().year - o_c.year) * 12 + ext_idx.max().month - o_c.month
        own = hist[["cash_requirement_bn"]].assign(is_forecast=False)
        path, _ = our_path("it", o_c, h_c, book_interest("it"), board=own)
        ext = ext.copy()
        ext[after] = path["total"].reindex(ext_idx[after]).to_numpy()
        ext_source[after] = "component model"
    plan_fc = M.anchor_to_plan(ext, xs / BN, asof, plans).reindex(months)
    ladder = lad.monthly_ladder(inp["ops"])
    gross = lad.forward_bonds(asof, months, inp["ops"]) / BN
    red = lad.forward_redemptions(months, inp["scad"], ladder)
    drift, drift_sc, drift_model = other_fin_drift(hist, months)
    f = pd.DataFrame(index=months)
    f["is_forecast"] = True
    # Live path = plan level × chosen month shape for years the DFP covers; scored smly stays on the chart.
    f["cash_requirement_bn"] = plan_fc.to_numpy()
    f["cash_requirement_rrf_grant_bn"] = -rrf["grant_bn"].to_numpy()
    f["cash_requirement_smly_bn"] = smly
    f["cash_requirement_flat_bn"] = flat
    f["cash_requirement_alt_bn"] = alt_fc
    f["cash_requirement_model_bn"] = fc
    f["cash_requirement_plan_bn"] = plan_fc.to_numpy()
    f["cash_requirement_ext_source"] = ext_source.reindex(months).to_numpy()
    f["bond_gross_bn"] = gross.to_numpy()
    f["bond_gross_known_events_bn"] = gross.attrs["announced_mn"].to_numpy() / BN
    f["bond_gross_assumed_bn"] = gross.attrs["assumed_mn"].to_numpy() / BN
    f["bond_redemption_bn"] = red["bond_redemption_mn"].to_numpy() / BN
    f["bond_redemption_cb_bn"] = lad.cb_redemptions(inp["ops"], months).to_numpy() / BN
    f["net_bonds_bn"] = f["bond_gross_bn"] - f["bond_redemption_bn"]
    f["private_net_bonds_bn"] = f["net_bonds_bn"] + f["bond_redemption_cb_bn"]
    existing_bot = red["bill_redemption_mn"].to_numpy() / BN
    f["bill_redemption_existing_bn"] = existing_bot
    f["redemption_source"] = red["source"].to_numpy()
    f["other_fin_events_bn"] = ev.other_fin_bn(months).to_numpy()
    f["other_fin_rrf_loan_bn"] = rrf["loan_bn"].to_numpy()
    f["other_fin_fx_redemption_bn"] = -red["other_redemption_mn"].to_numpy() / BN
    f["other_fin_drift_bn"] = drift
    f["other_fin_bn"] = f["other_fin_events_bn"] + f["other_fin_fx_redemption_bn"] + f["other_fin_drift_bn"]
    # Past the board's forward calendar there are no auctions to count, and a zero gross line would
    # push every maturing BTP and the whole fabbisogno into BOTs (the bill stock passed 50% of GDP by
    # 2030). There, gross bonds roll redemptions and fund the requirement net of other financing, so
    # the BOT stock moves only with its seasonal rule and the liquidity floor.
    last_fwd = gross.attrs.get("forward_last_month")
    tail = months > pd.Timestamp(last_fwd) if last_fwd else np.zeros(len(months), bool)
    f["bond_gross_tail_bn"] = 0.0
    f.loc[tail, "bond_gross_tail_bn"] = (f["bond_redemption_bn"] + f["cash_requirement_bn"]
                                         - f["other_fin_bn"]).clip(lower=0.0)[tail]
    f["bond_gross_bn"] = f["bond_gross_bn"] + f["bond_gross_tail_bn"]
    f["bond_gross_assumed_bn"] = f["bond_gross_assumed_bn"] + f["bond_gross_tail_bn"]
    f["net_bonds_bn"] = f["bond_gross_bn"] - f["bond_redemption_bn"]
    f["private_net_bonds_bn"] = f["net_bonds_bn"] + f["bond_redemption_cb_bn"]
    lvl0 = float(hist["buffer_level_bn"].iloc[-1])
    flows = (-f["cash_requirement_bn"] + f["net_bonds_bn"] + f["other_fin_bn"]).to_numpy()
    floor, cap = M.account_band(hist["buffer_level_bn"], asof)
    kb = lad.known_bill_gross(asof, months, inp["ops"]) / BN
    f["bill_gross_known_bn"] = kb.to_numpy()
    mix = lad.bot_tenor_mix(inp["ops"], asof)
    seas = M.bot_seasonal_net(hist["bill_stock_bn"], asof)
    target = seas.reindex(months.month).to_numpy()

    def hold(fl):
        return M.hold_bills(lvl0, fl, floor, existing_bot, mix.to_dict(), target, kb.to_numpy())

    net, bg, br = hold(flows)
    # Liquidity is published before the fabbisogno print: forecast months with an observed level carry
    # the gap to the model as their own series (inside other_fin, so the identity holds).
    obs = (inp["liq"]["total_mn"] / BN).reindex(months).to_numpy()
    gap = np.zeros(len(months))
    prev = lvl0
    for i in range(len(months)):
        if np.isnan(obs[i]):
            break
        gap[i] = obs[i] - prev - (flows[i] + net[i])
        prev = obs[i]
    if gap.any():
        flows = flows + gap
        net, bg, br = hold(flows)
    f["other_fin_liquidity_reconcile_bn"] = gap
    f["other_fin_bn"] = f["other_fin_bn"] + gap
    f["buffer_level_rollover_bn"] = lvl0 + np.cumsum(flows)
    f["net_bills_bn"], f["bill_gross_bn"], f["bill_redemption_bn"] = net, bg, br
    f["bill_net_target_bn"] = target
    f["buffer_change_bn"] = flows + f["net_bills_bn"].to_numpy()
    f["buffer_level_implied_bn"] = lvl0 + f["buffer_change_bn"].cumsum()
    f["buffer_floor_bn"], f["buffer_cap_bn"] = floor, cap
    f["liquidity_surplus_bn"] = (f["buffer_level_implied_bn"] - cap).clip(lower=0.0)
    stock0 = hist["bill_stock_bn"].dropna()
    if len(stock0):
        f["bill_stock_implied_bn"] = float(stock0.iloc[-1]) + f["net_bills_bn"].cumsum()
    liq = inp["liq"]["total_mn"] / BN
    f["buffer_level_bn"] = liq.reindex(months)
    # 80% band on cumulative fabbisogno from the chosen model's PRE+OOS errors; with the BOT stock held,
    # a fabbisogno miss lands in liquidity, so the band sits on the liquidity path.
    b = bt[(bt["model"] == chosen)].assign(w=M.window_of(bt[bt["model"] == chosen]))
    b = b[b["w"] != ""]
    q = (b.assign(e=(b["fc"] - b["actual"]).abs() / BN).groupby("h")["e"].quantile(0.8))
    steps = np.arange(1, len(months) + 1, dtype=float)
    hx = q.index.to_numpy(dtype=float)
    hy = q.to_numpy()
    band = np.interp(steps, hx, hy, left=hy[0] if len(hy) else np.nan,
                     right=hy[-1] if len(hy) else np.nan)
    f["cum_cash_requirement_band80_bn"] = band
    f.index.name = "month"
    from sovereign_cash.it import bottom_up_macro as BU
    f, bu_info = BU.apply_to_forward(
        f, xs, asof, plans_bn=plans, rrf_grant_bn=rrf["grant_bn"])
    if bu_info.get("bottomup_ship"):
        # Bills / liquidity were sized on the production cash path; refresh the identity.
        flows = (-f["cash_requirement_bn"] + f["net_bonds_bn"] + f["other_fin_bn"]).to_numpy()
        f["buffer_change_bn"] = flows + f["net_bills_bn"].to_numpy()
        f["buffer_level_implied_bn"] = lvl0 + f["buffer_change_bn"].cumsum()
        f["buffer_level_rollover_bn"] = lvl0 + np.cumsum(flows)
        f["liquidity_surplus_bn"] = (f["buffer_level_implied_bn"] - cap).clip(lower=0.0)
    yr = asof.year
    ytd = float(xs[(xs.index.year == yr) & (xs.index <= asof)].sum()) / BN
    in_yr = f.index.year == yr
    plan = plans.get(yr)
    r1 = lambda v: round(float(v), 1)  # noqa: E731
    dsc = drift_sc[drift_sc["window"] == "PRE"].groupby("model")["mae_bn"].mean()
    sur = f["liquidity_surplus_bn"]
    items = gross.attrs["items"]
    info = {
        "model": chosen,
        "alt_model": alt,
        "origin": str(asof.date()),
        "buffer_origin_bn": round(lvl0, 2),
        "band_bn": [round(floor, 2), round(cap, 2)],
        "bot_stock_origin_bn": r1(stock0.iloc[-1]) if len(stock0) else None,
        "bot_stock_end_bn": r1(f["bill_stock_implied_bn"].iloc[-1]) if len(stock0) else None,
        "bot_rule": "BOT stock held at its current level plus its average calendar-month change (trailing five "
                    "years, demeaned); raised only to keep liquidity at the floor",
        "cash_requirement_12m_bn": r1(f["cash_requirement_bn"].sum()),
        "cash_requirement_12m_alt_bn": r1(np.nansum(alt_fc)),
        "cash_requirement_12m_smly_bn": r1(np.nansum(smly)),
        "cash_requirement_12m_band80_bn": r1(band[-1]),
        "cash_requirement_12m_plan_bn": r1(f["cash_requirement_plan_bn"].sum()),
        "rrf_grant_share": round(ev.rrf_grant_share(), 3),
        "rrf_grants_base_12m_bn": r1(grants.iloc[-12:].sum()),
        "rrf_grants_fwd_12m_bn": r1(rrf["grant_bn"].sum()),
        "rrf_loans_fwd_12m_bn": r1(rrf["loan_bn"].sum()),
        "plan_year": yr,
        "plan_cash_requirement_bn": r1(plan) if plan is not None else None,
        "plan_pct_gdp": ev.PLAN.get(yr, {}).get("pct_gdp"),
        "plan_next_year_cash_requirement_bn": r1(plans_all[yr + 1]) if yr + 1 in plans_all else None,
        "plan_source": ev.PLAN_SOURCE,
        "plan_vintage": ev.PLAN_VINTAGE,
        "plan_scored": False,
        "cash_requirement_ytd_bn": r1(ytd),
        "ytd_through": str(asof.date())[:7],
        "model_full_year_bn": r1(ytd + f.loc[in_yr, "cash_requirement_bn"].sum()),
        "alt_full_year_bn": r1(ytd + f.loc[in_yr, "cash_requirement_alt_bn"].sum()),
        "plan_minus_model_full_year_bn": (r1(plan - (ytd + f.loc[in_yr, "cash_requirement_bn"].sum()))
                                          if plan is not None else None),
        "net_bonds_12m_bn": r1(f["net_bonds_bn"].sum()),
        "bond_gross_12m_bn": r1(f["bond_gross_bn"].sum()),
        "bond_gross_known_events_12m_bn": r1(f["bond_gross_known_events_bn"].sum()),
        "bond_gross_assumed_12m_bn": r1(f["bond_gross_assumed_bn"].sum()),
        "bond_redemption_12m_bn": r1(f["bond_redemption_bn"].sum()),
        "cb_held_redemptions_12m_bn": r1(f["bond_redemption_cb_bn"].sum()),
        "cb_holdings_snapshot": lad.cb_redemptions(inp["ops"], months[:1]).attrs.get("snapshot"),
        "private_net_bonds_12m_bn": r1(f["private_net_bonds_bn"].sum()),
        "net_bills_12m_bn": r1(f["net_bills_bn"].sum()),
        "bill_gross_12m_bn": r1(f["bill_gross_bn"].sum()),
        "bot_tenor_mix": {int(k): round(float(v), 3) for k, v in mix.items()},
        "other_fin_12m_bn": r1(f["other_fin_bn"].sum()),
        "other_fin_fx_redemption_12m_bn": r1(f["other_fin_fx_redemption_bn"].sum()),
        "other_fin_drift_model": drift_model,
        "other_fin_drift_12m_bn": r1(f["other_fin_drift_bn"].sum()),
        "other_fin_drift_pre_mae_bn": {m: round(float(v), 2) for m, v in dsc.items()},
        "liquidity_reconcile_bn": {str(m.date())[:7]: r1(v) for m, v in
                                   f["other_fin_liquidity_reconcile_bn"].items() if v != 0},
        "known_events": ev.describe(inp["ops"]),
        "issues": [{"month": str(r.month.date())[:7], "label": r.label, "cash_bn": r1(r.cash_mn / BN),
                    "basis": r.basis, "kind": r.kind, "id": r.id} for r in items.itertuples(index=False)],
        "buffer_end_bn": r1(f["buffer_level_implied_bn"].iloc[-1]),
        "buffer_end_rollover_bn": r1(f["buffer_level_rollover_bn"].iloc[-1]),
        "buffer_max_bn": r1(f["buffer_level_implied_bn"].max()),
        "liquidity_surplus_max_bn": r1(sur.max()),
        "liquidity_surplus_max_month": str(sur.idxmax().date())[:7] if sur.max() > 0 else None,
        "liquidity_surplus_end_bn": r1(sur.iloc[-1]),
        "months_above_cap": [str(m.date())[:7] for m in f.index[sur > 0.05]],
        "path": [{"month": str(m.date()), "cash_requirement_bn": round(float(r.cash_requirement_bn), 1),
                  "cash_requirement_plan_bn": round(float(r.cash_requirement_plan_bn), 1),
                  "bond_gross_bn": round(float(r.bond_gross_bn), 1),
                  "bond_redemption_bn": round(float(r.bond_redemption_bn), 1),
                  "bond_redemption_cb_bn": round(float(r.bond_redemption_cb_bn), 1),
                  "net_bonds_bn": round(float(r.net_bonds_bn), 1), "net_bills_bn": round(float(r.net_bills_bn), 1),
                  "other_fin_bn": round(float(r.other_fin_bn), 1),
                  "buffer_rollover_bn": round(float(r.buffer_level_rollover_bn), 1),
                  "buffer_bn": round(float(r.buffer_level_implied_bn), 1)} for m, r in f.iterrows()],
        "bond_gross_tape_last_op": gross.attrs.get("tape_last_op"),
        "bond_gross_forward_last_month": gross.attrs.get("forward_last_month"),
        "redemption_schedule_asof": inp["scad"].attrs.get("asof") if len(inp["scad"]) else None,
        **{k: v for k, v in bu_info.items()},
        "cash_requirement_bottomup_12m_bn": r1(f["cash_requirement_bottomup_bn"].sum()),
    }
    return f, info


def _lag_days(rel: pd.DataFrame) -> int:
    p = rel[rel["kind"] == "provisional"]
    if p.empty:
        return 1
    return int(((p["pub_date"] - (p["month"] + pd.offsets.MonthEnd(0))).dt.days).median())


def summary(inp: dict, hist: pd.DataFrame, asof: pd.Timestamp, sc: pd.DataFrame, chosen: str,
            info: dict, resid: float) -> dict:
    mef, liq, (_t1, t2), comp = inp["mef"], inp["liq"], inp["bdi"], inp["comp"]
    rng = lambda i: f"{i.min():%Y-%m}→{i.max():%Y-%m} ({len(i)} months)"  # noqa: E731
    sources = [
        {"name": "MEF press releases — saldo del settore statale (fabbisogno), provisional + year-ago final",
         "url": "https://www.mef.gov.it/ufficio-stampa/comunicati/", "frequency": "monthly",
         "lag_days": _lag_days(inp["releases"]), "verified": True,
         "history": rng(mef["fabb_mn"].dropna().index)},
        {"name": "MEF Tesoro — Liquidità del Tesoro (conto disponibilità / depositi governativi + OPTES)",
         "url": "https://www.dt.mef.gov.it/it/debito_pubblico/gestione_liquidita/dati_liquidita/",
         "frequency": "monthly", "lag_days": 1, "verified": True, "history": rng(liq.index)},
        {"name": "Banca d'Italia — Finanza pubblica: fabbisogno e debito, Tavola 1 (AC) and Tavola 2 (AP by instrument)",
         "url": "https://www.bancaditalia.it/pubblicazioni/finanza-pubblica/", "frequency": "monthly",
         "lag_days": 45, "verified": bool(len(t2)), "history": rng(t2.index) if len(t2) else ""},
        {"name": "MEF Tesoro — Composizione dei Titoli di Stato (BOT stock)",
         "url": "https://www.dt.mef.gov.it/it/debito_pubblico/dati_statistici/composizione_titoli_stato/",
         "frequency": "monthly", "lag_days": 10, "verified": bool(len(comp)),
         "history": rng(comp.index) if len(comp) else ""},
        {"name": "MEF Tesoro — Titoli in scadenza nei prossimi 12 mesi (forward redemptions)",
         "url": src.SCAD_PAGE, "frequency": "monthly", "lag_days": 10, "verified": bool(len(inp["scad"])),
         "history": f"as of {inp['scad'].attrs.get('asof')}" if len(inp["scad"]) else ""},
        {"name": "Issuance tape (auctions, syndications, retail; MEF communiqués)",
         "url": "local", "frequency": "per auction", "lag_days": 0, "verified": True,
         "history": f"2000-01→{info.get('bond_gross_tape_last_op')}"},
        {"name": "Issuance forward (quarterly programme, auction pattern, repeated syndications)",
         "url": "local", "frequency": "per auction", "lag_days": 0, "verified": True,
         "history": f"through {info.get('bond_gross_forward_last_month')}"},
        {"name": f"Annual plan: state-sector fabbisogno (vintage {ev.PLAN_VINTAGE}, hand-entered)",
         "url": "https://www.rgs.mef.gov.it/VERSIONE-I/attivita_istituzionali/previsione/contabilita_e_finanza_pubblica/"
                "documento_di_finanza_pubblica/",
         "frequency": "annual", "lag_days": 0, "verified": True,
         "history": ", ".join(f"{y} {contract.fmt_bn(v / BN, '€')}" for y, v in ev.plan_mn().items())
                    + "; one vintage, not scored"},
        {"name": "Known events: " + "; ".join(e["name"] for e in ev.BOND_EVENTS + ev.RRF_EVENTS + ev.OTHER_FIN_EVENTS),
         "url": "https://www.mef.gov.it/ufficio-stampa/comunicati/", "frequency": "ad hoc", "lag_days": 0,
         "verified": True, "history": "hand-entered with sources"},
        {"name": "RRF grant / loan split", "url": "", "frequency": "static", "lag_days": 0, "verified": True,
         "history": ev.RRF_SOURCE},
        {"name": "Eurosystem PSPP + PEPP holdings by ISIN (estimated: ECB country stock allocated by maturity)",
         "url": "local", "frequency": "monthly", "lag_days": 30, "verified": True,
         "history": f"snapshot {info.get('cb_holdings_snapshot')}"},
    ]
    score = [{"window": r.window, "h": int(r.h), "model": r.model, "mae_bn": round(float(r.mae_bn), 3),
              "benchmark": "smly / flat", "n": int(r.n), "chosen": r.model == chosen}
             for r in sc.itertuples(index=False)]
    fb = contract.fmt_bn
    unexp = unexplained(hist)
    notes = [
        f"Fabbisogno model chosen on PRE (origins {M.PRE[0][:4]}–{M.PRE[1][:4]}) mean MAE: {M.label(chosen)}; OOS "
        f"(origins {M.OOS_START[:4]}→) reported only. Scored on the first provisional MEF print, COVID months "
        f"{M.EXCL[0]:%b-%Y}–{M.EXCL[1]:%b-%Y} screened out.",
        f"RRF grants lower the fabbisogno; loans are financing. Grants are estimated from Banca d'Italia EU loans "
        f"at the plan ratio ({info['rrf_grant_share']:.0%} grants). The base months carry "
        f"{fb(info['rrf_grants_base_12m_bn'], '€')} of past grants, stripped before forecasting; next 12m add "
        f"{fb(info['rrf_grants_fwd_12m_bn'], '€')} of grants to the fabbisogno and "
        f"{fb(info['rrf_loans_fwd_12m_bn'], '€')} of loans to other financing.",
        f"Cash path = {'component SARIMAX + authority HICP/NGDP' if chosen == 'sx_level_exog' else 'seasonal / annual-level'} "
        f"({M.label(chosen)}). Forward drivers: AMECO NGDP (DFP Tav. II.1.2.2 where published) and WEO CPI. "
        f"The DFP total pins the year in progress "
        f"({info['plan_year']}: {fb(info['plan_cash_requirement_bn'], '€')} = {info['plan_pct_gdp']}% of GDP, "
        f"vintage {ev.PLAN_VINTAGE}) for the year in progress. Later years use our component model, the same path as the Shock scenarios "
        f"page (cash-account lines on the macro paths, interest from the debt book, RRF grants only as scheduled). "
        f"Scenario card = model-implied Δ from ±1pp GDP/CPI; live path unchanged.",
        f"BOT: {info['bot_rule']}. Liquidity above the band (P90 of the last {M.BAND_MONTHS} month-ends) is a "
        f"financing surplus: peak {fb(info['liquidity_surplus_max_bn'], '€')}"
        + (f" in {pd.Timestamp(info['liquidity_surplus_max_month']):%b-%y}" if info['liquidity_surplus_max_month'] else "")
        + "; the Treasury would close it with smaller auctions, buybacks or fewer BOT.",
        f"Other-financing baseline: {M.label(info['other_fin_drift_model'])} (PRE mean MAE "
        + ", ".join(f"{M.label(m)} {v:.2f}" for m, v in sorted(info['other_fin_drift_pre_mae_bn'].items(),
                                                                 key=lambda kv: kv[1]))
        + f"bn), next 12m {fb(info['other_fin_drift_12m_bn'], '€', plus=True)}.",
        f"Gross bonds next 12m include {fb(info['bond_gross_known_events_12m_bn'], '€')} announced syndications / "
        f"retail and {fb(info['bond_gross_assumed_12m_bn'], '€')} assumed (last year's syndications and retail "
        "issues repeated 52 weeks on by the issuance forward).",
        f"Eurosystem-held redemptions are estimates (holdings by ISIN are not published; snapshot "
        f"{info['cb_holdings_snapshot']}); next 12m {fb(info['cb_held_redemptions_12m_bn'], '€')}.",
    ]
    if unexp:
        notes.append(f"Identity: history closes by construction (other financing is the residual). Against Banca "
                     f"d'Italia's instrument table ({unexp['from']}→{unexp['to']}), the part not explained by its "
                     f"non-security financing and perimeter has mean {fb(unexp['mean_bn'], '€', 2)} and sd "
                     f"{fb(unexp['sd_bn'], '€')} a month (last 24m sd {fb(unexp['sd_24m_bn'], '€')}), vs other "
                     f"financing sd {fb(unexp['other_sd_bn'], '€')}.")
    if info["liquidity_reconcile_bn"]:
        notes.append("Published liquidity runs ahead of the fabbisogno print: the gap to the model ("
                     + ", ".join(f"{pd.Timestamp(m):%b-%y} {fb(v, '€', plus=True)}"
                                 for m, v in info["liquidity_reconcile_bn"].items())
                     + ") is shown as its own series until the print explains it.")
    if mef.attrs.get("dropped_finals"):
        notes.append(f"{mef.attrs['dropped_finals']} parsed year-ago restatements dropped by the parse guard.")
    if info.get("bottomup_reason"):
        notes.append(
            f"Bottom-up BdI T1 baseline: {'SHIP' if info.get('bottomup_ship') else 'NO-SHIP'} — "
            f"{info['bottomup_reason']}. Alternate column cash_requirement_bottomup_bn "
            f"{'replaces' if info.get('bottomup_ship') else 'does not replace'} the live path.")
    return {
        "country": "IT", "ccy": "EUR", "asof": str(asof.date()), "sources": sources,
        "buffer": "MEF month-end 'disponibilità liquide del Tesoro': Treasury account(s) at Banca d'Italia "
                  "(conto disponibilità; 'saldo dei depositi governativi' from 2025) + money-market deployment (OPTES)",
        "cash_requirement": "MEF state-sector fabbisogno (saldo del settore statale), positive = borrowing need",
        "identity_resid_bn": resid, "identity_note": "history closes by construction: other_fin is the residual",
        "unexplained": unexp, "score": score, "score_note": "MAE of cumulative fabbisogno, EUR bn",
        "forward": {k: v for k, v in info.items() if k != "path"} | {"path": info["path"]},
        "notes": notes,
    }


def unexplained(hist: pd.DataFrame) -> dict | None:
    """other_fin not explained by Banca d'Italia: tape-vs-BdI securities gaps + MEF-vs-BdI liquidity gap."""
    cols = ["other_fin_liq_gap_bn", "other_fin_bond_gap_bn", "other_fin_bill_gap_bn"]
    if not all(c in hist for c in cols):
        return None
    d = hist.dropna(subset=cols)
    if d.empty:
        return None
    u = d[cols].sum(axis=1)
    return {"from": f"{d.index.min():%Y-%m}", "to": f"{d.index.max():%Y-%m}", "n": int(len(u)),
            "mean_bn": round(float(u.mean()), 2), "sd_bn": round(float(u.std()), 2),
            "sd_24m_bn": round(float(u.tail(24).std()), 2), "max_abs_bn": round(float(u.abs().max()), 1),
            "max_abs_month": f"{u.abs().idxmax():%Y-%m}",
            "other_sd_bn": round(float(d["other_fin_bn"].std()), 2)}


def run(refresh: bool = False, dashboard: bool = True, verbose: bool = True) -> dict:
    inp = load_inputs(refresh)
    hist, asof = history(inp)
    x = inp["mef"]["fabb_mn"].loc["2010-01":asof].asfreq("MS")
    bt, sc, chosen = run_backtest(inp["mef"]["fabb_rt_mn"].loc["2010-01":asof].asfreq("MS"))
    fwd, info = forward(inp, hist, asof, x, chosen, bt, sc)
    df = pd.concat([hist, fwd], axis=0, sort=False)
    df.index.name = "month"
    if "cash_requirement_bottomup_bn" in df.columns:
        hist_mask = ~df["is_forecast"].fillna(False)
        df.loc[hist_mask, "cash_requirement_bottomup_bn"] = df.loc[hist_mask, "cash_requirement_bn"]
    else:
        df["cash_requirement_bottomup_bn"] = df["cash_requirement_bn"]
    df = df.reset_index()
    cols = list(contract.REQUIRED) + [c for c in df.columns if c not in contract.REQUIRED]
    df = df[cols]
    resid = contract.validate(df)
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DERIVED / "it_cash_monthly.parquet", index=False)
    bt.to_parquet(DERIVED / "it_cash_backtest.parquet", index=False)
    from sovereign_cash import scenarios as SX
    sx = SX.run("it")
    SX.save("it", sx)
    summ = summary(inp, hist, asof, sc, chosen, info, resid)
    summ["scenarios"] = sx
    (DERIVED / "it_cash_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    if dashboard:
        from sovereign_cash.it import dashboard as dash

        dash.build(df, summ, sc)
    if verbose:
        tab = sc.pivot_table(index="model", columns=["window", "h"], values="mae_bn")
        for w in ("PRE", "OOS"):
            tab[(w, "mean")] = sc[sc["window"] == w].groupby("model")["mae_bn"].mean()
        tab = tab.sort_values(("PRE", "mean")).round(2)
        print(f"asof {asof:%Y-%m}  chosen {chosen}  identity resid {resid:.2e}bn")
        print(tab.to_string())
        print(json.dumps({k: v for k, v in info.items() if k != "path"}, indent=1, default=str))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch MEF / Banca d'Italia pages")
    ap.add_argument("--no-dashboard", action="store_true")
    a = ap.parse_args()
    run(refresh=a.refresh, dashboard=not a.no_dashboard)


if __name__ == "__main__":
    main()
