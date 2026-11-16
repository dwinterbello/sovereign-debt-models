"""Standalone page ``sovereign_cash/it_cash_dashboard.html`` (theme from us/fiscal/build_dashboard.py)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sovereign_cash.cash_season import (
    SEASON_CSS, bottomup_card, category_tape_html, default_specs,
    it_category_panel, season_html,
)
from sovereign_cash.contract import ROOT, fmt_bn, fmt_num
from sovereign_cash.it.model import label as model_label
from us.fiscal.build_dashboard import _T, BLUE, GREEN, GREY, ORANGE, PAGE_CSS, PURPLE, RED, _html, _kpi, _style
from us.fiscal.build_liquidity_dashboard import TABLE_CSS, _table

TABLE_CSS = TABLE_CSS + SEASON_CSS

OUT = ROOT / "it_cash_dashboard.html"
HIST_YEARS = 5
IDENTITY_QUARTERS = 12
LIGHT_PURPLE = "rgba(150,110,200,0.45)"


def eur(v, dp: int = 1, plus: bool = False) -> str:
    return fmt_bn(v, "€", dp, plus=plus)


def _mon(s) -> str:
    return f"{pd.Timestamp(s):%b-%y}"


def _top(fig: go.Figure, *, title: str, ylab: str = "EUR bn", height: int = 420) -> go.Figure:
    _style(fig, title=title, ylab=ylab, height=height)
    fig.update_layout(title=dict(y=0.985, yanchor="top"), margin=dict(t=82, b=40),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return fig


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.set_index("month")
    fc = d["is_forecast"].astype(bool)
    return d[~fc], d[fc]


# ── charts ──────────────────────────────────────────────────────────────────

def fig_plan(df: pd.DataFrame, fw: dict) -> go.Figure:
    """Cumulative fabbisogno in the plan year: actual YTD, model / alternative / plan-anchored, plan total."""
    yr = int(fw["plan_year"])
    h, f = _split(df)
    h, f = h[h.index.year == yr], f[f.index.year == yr]
    cum_h = h["cash_requirement_bn"].cumsum()
    base = float(cum_h.iloc[-1]) if len(cum_h) else 0.0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cum_h.index, y=cum_h, name=f"Actual {yr}", marker_color=GREY))
    for col, name, color, dash in (
            ("cash_requirement_bn", "Live (plan × shape)", BLUE, "solid"),
            ("cash_requirement_model_bn", f"Scored ({model_label(fw['model'])})", ORANGE, "dash"),
            ("cash_requirement_alt_bn", f"Alternative ({model_label(fw['alt_model'])})", PURPLE, "dot")):
        s = pd.concat([cum_h.tail(1), base + f[col].cumsum()])
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines", line=dict(color=color, width=2, dash=dash)))
    all_h = df.set_index("month")["cash_requirement_bn"]
    prev = all_h[all_h.index.year == yr - 1].cumsum()
    fig.add_trace(go.Scatter(x=prev.index + pd.DateOffset(years=1), y=prev, name=f"{yr - 1}",
                             mode="lines", line=dict(color=GREY, width=1, dash="dot")))
    if fw.get("plan_cash_requirement_bn") is not None:
        fig.add_hline(y=fw["plan_cash_requirement_bn"], line=dict(color=GREY, width=1.2, dash="dash"),
                      annotation_text=f"Plan {yr} {eur(fw['plan_cash_requirement_bn'])}",
                      annotation_position="bottom right")
    return _top(fig, title=f"{yr} state-sector fabbisogno, cumulative (EUR bn): actual, forecasts and plan")


def fig_supply(df: pd.DataFrame, fw: dict) -> go.Figure:
    _, f = _split(df)
    iss = pd.DataFrame(fw.get("issues", []))
    hover = {}
    if len(iss):
        for m, g in iss.groupby("month"):
            hover[m] = "<br>".join(f"{r.label}: {eur(r.cash_bn)} ({r.basis})" for r in g.itertuples(index=False))
    cd = [hover.get(f"{m:%Y-%m}", "") for m in f.index]
    regular = f["bond_gross_bn"] - f["bond_gross_known_events_bn"] - f["bond_gross_assumed_bn"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=f.index, y=regular, name="Auctions", marker_color=GREEN))
    fig.add_trace(go.Bar(x=f.index, y=f["bond_gross_known_events_bn"], name="Announced syndication / retail",
                         marker_color=PURPLE, customdata=cd, hovertemplate="%{y:.1f}<br>%{customdata}"))
    fig.add_trace(go.Bar(x=f.index, y=f["bond_gross_assumed_bn"], name="Assumed syndication / retail",
                         marker=dict(color=LIGHT_PURPLE, pattern=dict(shape="/")), customdata=cd,
                         hovertemplate="%{y:.1f}<br>%{customdata}"))
    fig.add_trace(go.Bar(x=f.index, y=-(f["bond_redemption_bn"] - f["bond_redemption_cb_bn"]),
                         name="Redemptions, private", marker_color=RED))
    fig.add_trace(go.Bar(x=f.index, y=-f["bond_redemption_cb_bn"], name="Redemptions, Eurosystem (est.)",
                         marker_color=ORANGE))
    fig.add_trace(go.Scatter(x=f.index, y=f["private_net_bonds_bn"], name="Private net supply",
                             mode="lines+markers", line=dict(color=_T["text"], width=1.8), marker=dict(size=4)))
    fig.update_layout(barmode="relative", bargap=0.15)
    return _top(fig, title="Bond supply next 12 months (EUR bn)")


def fig_buffer(df: pd.DataFrame) -> go.Figure:
    h, f = _split(df)
    h = h.loc[h.index >= h.index.max() - pd.DateOffset(years=HIST_YEARS)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h.index, y=h["buffer_account_bn"], name="Account at Banca d'Italia", marker_color=GREY,
                         opacity=0.5))
    fig.add_trace(go.Scatter(x=h.index, y=h["buffer_level_bn"], name="Liquidity (account + OPTES)",
                             mode="lines", line=dict(color=BLUE, width=2)))
    lvl0 = pd.Series([h["buffer_level_bn"].iloc[-1]], index=[h.index[-1]])
    path = pd.concat([lvl0, f["buffer_level_implied_bn"]])
    band = np.concatenate([[0.0], f["cum_cash_requirement_band80_bn"].to_numpy()])
    fig.add_trace(go.Scatter(x=list(path.index) + list(path.index[::-1]),
                             y=list(path + band) + list((path - band)[::-1]), fill="toself",
                             fillcolor="rgba(0,114,189,0.10)", line=dict(width=0), name="80% band",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=path.index, y=path, name="Forecast (BOT stock held)", mode="lines",
                             line=dict(color=BLUE, width=1.8, dash="dash")))
    roll = pd.concat([lvl0, f["buffer_level_rollover_bn"]])
    fig.add_trace(go.Scatter(x=roll.index, y=roll, name="BOT rolled at par", mode="lines",
                             line=dict(color=GREY, width=1.2, dash="dot")))
    obs = f["buffer_level_bn"].dropna()
    if len(obs):
        fig.add_trace(go.Scatter(x=obs.index, y=obs, name="Published, ahead of the fabbisogno print",
                                 mode="markers", marker=dict(color=BLUE, size=8, symbol="circle-open",
                                                             line=dict(width=2))))
    for y, lab in ((f["buffer_floor_bn"].iloc[0], "floor (P10)"), (f["buffer_cap_bn"].iloc[0], "cap (P90)")):
        fig.add_hline(y=y, line=dict(color=GREY, width=1, dash="dash"), annotation_text=lab,
                      annotation_position="bottom left", annotation_font_size=9)
    return _top(fig, title="Treasury liquidity, month-end (EUR bn): history and 12-month forecast", height=440)


def fig_bot(df: pd.DataFrame) -> go.Figure | None:
    h, f = _split(df)
    s = h["bill_stock_bn"].dropna()
    if s.empty or "bill_stock_implied_bn" not in f:
        return None
    s = s.loc[s.index >= s.index.max() - pd.DateOffset(years=HIST_YEARS)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="BOT outstanding", mode="lines", line=dict(color=BLUE, width=2)))
    fwd = pd.concat([s.tail(1), f["bill_stock_implied_bn"]])
    fig.add_trace(go.Scatter(x=fwd.index, y=fwd, name="Forecast", mode="lines", line=dict(color=BLUE, width=1.8,
                                                                                         dash="dash")))
    fig.add_trace(go.Bar(x=f.index, y=f["net_bills_bn"], name="Net BOT (forecast)", marker_color=GREEN, yaxis="y2",
                         opacity=0.6))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="net, EUR bn",
                                  zeroline=True, zerolinecolor=_T["chart-zero"]))
    _top(fig, title="BOT stock (EUR bn): held at its level plus the usual calendar-month pattern", height=400)
    fig.update_layout(margin=dict(r=52))
    return fig


def fig_bridge(df: pd.DataFrame) -> go.Figure:
    h, f = _split(df)
    l0 = float(h["buffer_level_bn"].iloc[-1])
    other = f["other_fin_bn"] - f["other_fin_rrf_loan_bn"] - f["other_fin_liquidity_reconcile_bn"]
    steps = [("−Fabbisogno", -float(f["cash_requirement_bn"].sum())),
             ("Bond gross", float(f["bond_gross_bn"].sum())),
             ("Bond redemptions", -float(f["bond_redemption_bn"].sum())),
             ("Net BOT", float(f["net_bills_bn"].sum())),
             ("RRF loans", float(f["other_fin_rrf_loan_bn"].sum())),
             ("Other", float(other.sum())),
             ("Gap to published", float(f["other_fin_liquidity_reconcile_bn"].sum()))]
    steps = [s for s in steps if abs(s[1]) >= 0.05]
    end = float(f["buffer_level_implied_bn"].iloc[-1])
    fig = go.Figure(go.Waterfall(
        x=[f"Liquidity {h.index[-1]:%b-%y}"] + [s[0] for s in steps] + [f"{f.index[-1]:%b-%y}"],
        y=[l0] + [s[1] for s in steps] + [end], measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        text=[fmt_num(l0)] + [fmt_num(s[1], plus=True) for s in steps] + [fmt_num(end)], textposition="outside",
        increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color=GREY, width=0.6))))
    return _style(fig, title="12-month bridge of Treasury liquidity (EUR bn)", ylab="EUR bn", height=420)


def fig_fabbisogno(df: pd.DataFrame, fw: dict) -> go.Figure:
    h, f = _split(df)
    h = h.loc[h.index >= h.index.max() - pd.DateOffset(years=HIST_YEARS)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h.index, y=h["cash_requirement_bn"], name="State sector (MEF)", marker_color=GREY))
    fig.add_trace(go.Bar(x=f.index, y=f["cash_requirement_bn"], name="Forecast (plan × shape)",
                         marker_color=BLUE))
    if "cash_requirement_model_bn" in f:
        fig.add_trace(go.Scatter(x=f.index, y=f["cash_requirement_model_bn"],
                                 name=f"Scored ({model_label(fw['model'])})", mode="lines",
                                 line=dict(color=ORANGE, width=1.4, dash="dash")))
    fig.add_trace(go.Scatter(x=f.index, y=f["cash_requirement_alt_bn"],
                             name=f"Alternative ({model_label(fw['alt_model'])})", mode="lines",
                             line=dict(color=PURPLE, width=1.4, dash="dot")))
    if "fabb_ac_bdi_bn" in h:
        fig.add_trace(go.Scatter(x=h.index, y=h["fabb_ac_bdi_bn"], name="Central government (Banca d'Italia)",
                                 mode="lines", line=dict(color=GREY, width=1)))
    fig.update_layout(bargap=0.15)
    return _top(fig, title="Monthly cash requirement (EUR bn, borrowing need +; RRF grants lower it)")


def fig_identity(df: pd.DataFrame) -> go.Figure:
    d = df.set_index("month").copy()
    d["gap"] = d.get("other_fin_liquidity_reconcile_bn", pd.Series(0.0, index=d.index)).fillna(0.0)
    d["other"] = d["other_fin_bn"] - d["gap"]
    d["neg_fabb"] = -d["cash_requirement_bn"]
    q = d.index.to_period("Q")
    cols = ["neg_fabb", "net_bonds_bn", "net_bills_bn", "other", "gap", "buffer_change_bn"]
    g = d[cols].groupby(q).sum()
    fc = d["is_forecast"].astype(bool).groupby(q).any()
    g = g.tail(IDENTITY_QUARTERS + 4)
    lab = [f"{p.year} Q{p.quarter}" + ("*" if fc[p] else "") for p in g.index]
    fig = go.Figure()
    for col, name, color in (("neg_fabb", "−Fabbisogno", RED), ("net_bonds_bn", "Net bonds", GREEN),
                             ("net_bills_bn", "Net BOT", BLUE), ("other", "Other financing", PURPLE),
                             ("gap", "Gap to published liquidity", GREY)):
        if g[col].abs().max() < 0.05:
            continue
        fig.add_trace(go.Bar(x=lab, y=g[col], name=name, marker_color=color))
    fig.add_trace(go.Scatter(x=lab, y=g["buffer_change_bn"], name="Δ liquidity", mode="lines+markers",
                             line=dict(color=_T["text"], width=1.6), marker=dict(size=5)))
    fig.update_layout(barmode="relative", bargap=0.2)
    return _top(fig, title="Identity by quarter: Δ liquidity = −fabbisogno + net bonds + net BOT + other "
                           "(EUR bn; * = includes forecast months)", height=440)


def fig_other(df: pd.DataFrame) -> go.Figure | None:
    h, _ = _split(df)
    if "other_fin_bond_gap_bn" not in h:
        return None
    d = h[h["fabb_ap_bdi_bn"].notna()].copy()
    d["eu"] = d["other_fin_eu_loans_bn"].fillna(0.0)
    d["nonsec_ex_eu"] = d["other_fin_bdi_nonsec_bn"] - d["eu"]
    yr = d.index.year
    n = d.groupby(yr).size()
    keep = n.index[(n == 12) | (n.index == n.index.max())]
    cols = [("eu", "EU loans (RRF, SURE)", ORANGE), ("nonsec_ex_eu", "Deposits, postal, other loans", GREY),
            ("other_fin_bond_gap_bn", "Bonds: Banca d'Italia − tape", GREEN),
            ("other_fin_bill_gap_bn", "BOT: Banca d'Italia − tape", BLUE),
            ("other_fin_perimeter_bn", "State sector − general government", RED),
            ("other_fin_liq_gap_bn", "Liquidity: MEF − Banca d'Italia", PURPLE)]
    y = d.groupby(yr)[[c for c, _, _ in cols] + ["other_fin_bn"]].sum().loc[keep]
    last = d.index.max()
    xl = [str(v) if n[v] == 12 else f"{v} (to {last:%b})" for v in y.index]
    fig = go.Figure()
    for c, name, color in cols:
        fig.add_trace(go.Bar(x=xl, y=y[c], name=name, marker_color=color))
    fig.add_trace(go.Scatter(x=xl, y=y["other_fin_bn"], name="Other financing, total", mode="lines+markers",
                             line=dict(color=_T["text"], width=1.6)))
    fig.update_layout(barmode="relative")
    return _top(fig, title="Other financing by year, split with Banca d'Italia's instrument table (EUR bn)",
                height=420)


# ── tables ──────────────────────────────────────────────────────────────────

def score_table(sc: pd.DataFrame, chosen: str) -> str:
    models = list(sc.loc[sc["window"] == "PRE"].groupby("model")["mae_bn"].mean().sort_values().index)
    lab = {m: model_label(m) + (" (chosen)" if m == chosen else "") for m in models}
    rows = []
    for (w, h), g in sc.groupby(["window", "h"]):
        r = {"window": w, "months ahead": str(int(h)), "n": int(g["n"].max())}
        for m in models:
            v = g.loc[g["model"] == m, "mae_bn"]
            r[lab[m]] = float(v.iloc[0]) if len(v) else np.nan
        rows.append(r)
    for w in ("PRE", "OOS"):
        g = sc[sc["window"] == w]
        r = {"window": w, "months ahead": "mean", "n": int(g["n"].max()) if len(g) else 0}
        r.update({lab[m]: float(g.loc[g["model"] == m, "mae_bn"].mean()) for m in models})
        rows.append(r)
    rows.sort(key=lambda r: (r["window"] != "PRE", r["months ahead"] == "mean",
                             int(r["months ahead"]) if r["months ahead"].isdigit() else 99))
    cols = ["window", "months ahead", "n"] + [lab[m] for m in models]
    return _table(rows, cols, best_low=cols[3:], fmt="{:,.2f}")


def path_table(df: pd.DataFrame) -> str:
    _, f = _split(df)
    f = f.assign(other_ex_gap=f["other_fin_bn"] - f["other_fin_liquidity_reconcile_bn"].fillna(0.0))
    spec = [("Fabbisogno", "cash_requirement_bn"), ("Bond gross", "bond_gross_bn"),
            ("of which assumed", "bond_gross_assumed_bn"), ("Redemptions", "bond_redemption_bn"),
            ("Eurosystem (est.)", "bond_redemption_cb_bn"), ("Net bonds", "net_bonds_bn"),
            ("Net BOT", "net_bills_bn"), ("BOT stock", "bill_stock_implied_bn"), ("Other fin.", "other_ex_gap"),
            ("Gap to published", "other_fin_liquidity_reconcile_bn"), ("Liquidity", "buffer_level_implied_bn"),
            ("Above band", "liquidity_surplus_bn")]
    spec = [(n, c) for n, c in spec if c in f and (f[c].abs().max() >= 0.05 or n in ("Net BOT", "Above band"))]
    rows = [{"month": _mon(m), **{n: fmt_num(r[c]) for n, c in spec}} for m, r in f.iterrows()]
    return _table(rows, ["month"] + [n for n, _ in spec])


def lumps_table(df: pd.DataFrame, fw: dict) -> str:
    _, f = _split(df)
    rows = []
    for e in fw.get("issues", []):
        basis = "announced" if e["basis"] == "announced" else "assumed: last year's issue repeated"
        if e["basis"] == "announced" and e["kind"] == "retail" and not e["id"].startswith("IT0"):
            basis = "announced; size assumed (mean of the last two)"
        rows.append({"month": _mon(e["month"]), "item": e["label"], "EUR bn": eur(e["cash_bn"]), "basis": basis})
    for e in fw.get("known_events", []):
        if e["kind"] != "rrf" or pd.Timestamp(e["settle"]) not in f.index:
            continue
        rows.append({"month": _mon(e["settle"]), "item": e["name"], "EUR bn": eur(e["cash_bn"]),
                     "basis": f"timing assumed; grant {eur(e['grant_bn'])} (fabbisogno) / loan "
                              f"{eur(e['loan_bn'])} (other financing), est."})
    for m, v in f["bond_redemption_bn"].nlargest(4).items():
        rows.append({"month": _mon(m), "item": "Bond redemptions (MEF schedule)", "EUR bn": eur(v),
                     "basis": f"Eurosystem-held {eur(f.at[m, 'bond_redemption_cb_bn'])} (est.)"})
    rows.sort(key=lambda r: pd.Timestamp(pd.to_datetime(r["month"], format="%b-%y")))
    return _table(rows, ["month", "item", "EUR bn", "basis"])


def scenario_card(summ: dict) -> str:
    """UK-style unit-shock table from the level+exog SARIMAX."""
    from sovereign_cash import scenarios as SX
    payload = summ.get("scenarios") or SX.load("it")
    if not payload:
        return "<div class='sub'>No scenario table (run refresh).</div>"
    return SX.table_html("it", payload)


# ── page ────────────────────────────────────────────────────────────────────

def build(df: pd.DataFrame, summ: dict, sc: pd.DataFrame) -> str:
    from sovereign_cash.horizon import clip_frame, programme_end
    # Charts: stop at last gross BTP month (past that net bonds = −redemptions only).
    end = programme_end(df)
    chart_df = clip_frame(df)
    fw = summ["forward"]
    chosen, alt = fw["model"], fw["alt_model"]
    h, f = _split(df)
    acct = h["buffer_account_bn"].tail(24)
    mm = h["buffer_mm_bn"].tail(24)
    yr = fw["plan_year"]
    asof = pd.Timestamp(summ["asof"])
    obs = f["buffer_level_bn"].dropna()
    unexp = summ.get("unexplained") or {}
    sur_m = fw.get("liquidity_surplus_max_month")
    lead = "".join([
        _kpi(f"Fabbisogno {yr} to {asof:%b}", eur(fw["cash_requirement_ytd_bn"]),
             f"plan {eur(fw['plan_cash_requirement_bn'])} for the year"),
        _kpi(f"Fabbisogno {yr} full year", eur(fw["model_full_year_bn"]),
             f"{model_label(chosen)} · {model_label(alt)} {eur(fw['alt_full_year_bn'])} · "
             f"plan − model {eur(fw['plan_minus_model_full_year_bn'], plus=True)}"),
        _kpi("Private net BTP supply 12m", eur(fw["private_net_bonds_12m_bn"], plus=True),
             f"net {eur(fw['net_bonds_12m_bn'], plus=True)} + Eurosystem runoff (est.) "
             f"{eur(fw['cb_held_redemptions_12m_bn'])}"),
        _kpi("Assumed gross supply 12m", eur(fw["bond_gross_assumed_12m_bn"]),
             f"repeats of last year's syndications / retail · announced {eur(fw['bond_gross_known_events_12m_bn'])}"),
    ])
    kp = "".join([
        _kpi("Treasury liquidity", eur(fw["buffer_origin_bn"]),
             f"end-{asof:%b-%y}" + (f" · published {_mon(obs.index[-1])} {eur(obs.iloc[-1])}" if len(obs) else "")),
        _kpi("Fabbisogno next 12m", eur(fw["cash_requirement_12m_bn"]),
             f"±{eur(fw['cash_requirement_12m_band80_bn'], 0)} (80%) · {model_label(alt)} "
             f"{eur(fw['cash_requirement_12m_alt_bn'])}"),
        _kpi("Net bonds next 12m", eur(fw["net_bonds_12m_bn"], plus=True),
             f"gross {eur(fw['bond_gross_12m_bn'], 0)} − redemptions {eur(fw['bond_redemption_12m_bn'], 0)}"),
        _kpi("Net BOT next 12m", eur(fw["net_bills_12m_bn"], plus=True),
             f"stock {eur(fw['bot_stock_origin_bn'], 0)} → {eur(fw['bot_stock_end_bn'], 0)}"),
        _kpi("Liquidity above band", eur(fw["liquidity_surplus_max_bn"]),
             (f"peak {_mon(sur_m + '-01')} · end {eur(fw['liquidity_surplus_end_bn'])} · cap "
              f"{eur(fw['band_bn'][1], 0)}") if sur_m else f"within band · cap {eur(fw['band_bn'][1], 0)}"),
        _kpi("Unexplained financing", eur(unexp.get("sd_bn")),
             f"monthly sd vs Banca d'Italia, {unexp.get('from', '')}→{unexp.get('to', '')}"),
    ])
    figs = [fig_buffer(chart_df), fig_bot(chart_df), fig_bridge(chart_df),
            fig_fabbisogno(chart_df, fw), fig_identity(chart_df), fig_other(chart_df)]
    figs = [g for g in figs if g is not None]
    lead_figs = [fig_plan(chart_df, fw), fig_supply(chart_df, fw)]
    lead_cards = [f"<div class='card'>{_html(g, first=(i == 0))}</div>" for i, g in enumerate(lead_figs)]
    cards = [f"<div class='card'>{_html(g, first=False)}</div>" for g in figs]
    notes = "".join(f"<li>{n}</li>" for n in summ["notes"])
    srcs = "".join(f"<li>{s['name']} — {s.get('history', '')}" + ("" if s["verified"] else " (unverified)") + "</li>"
                   for s in summ["sources"])
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bu = bottomup_card(summ)
    hz = f"through {end:%b-%Y} (last gross BTP month)" if end is not None else "through the live horizon"
    season = season_html(
        chart_df, cc="it", prefix="it",
        specs=default_specs("cash_requirement_bn", "Fabbisogno"),
        fy_start_month=1, currency="€", tall_key="cash",
        forecast_note=f"Forecast months follow the live fabbisogno path {hz}.",
    )
    try:
        cat_panel, cat_specs = it_category_panel()
        cat_html = category_tape_html(
            cat_panel, cc="it", prefix="it", specs=cat_specs, currency="€",
            note="DFP/BdI category leaves under the shipped fabbisogno bottom-up.")
    except Exception:  # noqa: BLE001
        cat_html = ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Italy — central-government cash (fabbisogno, Treasury liquidity, BOT)</title>
  <script src="https://cdn.plot.ly/plotly-3.7.0.min.js" charset="utf-8"></script>
  <style>{PAGE_CSS}{TABLE_CSS}</style>
</head>
<body>
  <header>
    <h1>Italy — central-government cash: fabbisogno, Treasury liquidity, BOT</h1>
    <div class="sub">{built} · fabbisogno through {asof:%b-%Y}. Δ Treasury liquidity = −fabbisogno + net bonds
      + net BOT + other financing.</div>
  </header>
  <div class="wrap">
    <h2 class="section">Headline</h2>
    <div class="kpi-row">{lead}</div>
    <div class="grid-2">{lead_cards[0]}{lead_cards[1]}</div>
    {bu}
    <h2 class="section">Through the year</h2>
    {season}
    {cat_html}
    <h2 class="section">Known and assumed lumps</h2>
    {lumps_table(df, fw)}
    <div class="kpi-row">{kp}</div>
    <h2 class="section">Treasury liquidity and BOT</h2>
    <div class="grid">{cards[0]}</div>
    <p class="sub">Swings are in OPTES (money-market deployment): over 24 months the Banca d'Italia account
      averaged {eur(acct.mean())} (sd {eur(acct.std())}) against OPTES sd {eur(mm.std())}.</p>
    <div class="grid-2">{''.join(cards[1:3])}</div>
    <h2 class="section">Fabbisogno and identity</h2>
    <div class="grid">{''.join(cards[3:])}</div>
    <h2 class="section">Fabbisogno backtest (MAE of cumulative fabbisogno, EUR bn)</h2>
    {score_table(sc, chosen)}
    <h2 class="section">Macro scenario (model-implied)</h2>
    <div class="card">{scenario_card(summ)}</div>
    <h2 class="section">12-month path (EUR bn)</h2>
    {path_table(df)}
    <h2 class="section">Notes</h2>
    <ul class="sub">{notes}</ul>
    <h2 class="section">Sources</h2>
    <ul class="sub">{srcs}</ul>
  </div>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")
    return str(OUT)
