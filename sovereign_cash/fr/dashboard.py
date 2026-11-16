"""France cash-framework page (own charts; table/score helpers and page CSS come from the shared DE builder)."""

from __future__ import annotations

import datetime as dt
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sovereign_cash.cash_season import (
    SEASON_CSS, bottomup_card, category_tape_html, default_specs,
    fr_category_panel, season_html,
)
from sovereign_cash.contract import fmt_bn
from sovereign_cash.de import dashboard as D
from sovereign_cash.de.dashboard import BLUE, GREEN, GREY, ORANGE, PAGE_CSS, PURPLE, RED, TABLE_CSS, _html, _kpi, _style
from sovereign_cash.investor import seasonal_role

LABELS = {
    "smly": "same month last year",
    "seas_add_3": "additive seasonal, 3y", "seas_add_5": "additive seasonal, 5y",
    "seas_mult_3": "multiplicative seasonal, 3y", "seas_mult_5": "multiplicative seasonal, 5y",
    "prevyear_3": "prior-year total, 3y shape", "prevyear_5": "prior-year total, 5y shape",
    "annual_ngdp_3": "annual GDP level × seasonal shape",
    "annual_comp_unit_3": "component annual (GDP/CPI pass-through) × seasonal shape",
    "annual_comp_floor_3": "component annual (floored slopes) × seasonal shape",
    "gap0": "identity, no gap", "gapseas": "identity + seasonal gap", "flat": "flat stock",
}
EXTRA_CSS = ("table.sc td.n { font-variant-numeric: tabular-nums; } p.sub { max-width: 1080px; }"
             + SEASON_CSS)


def label(m: str) -> str:
    return LABELS.get(m, m)


def eur(v, dp: int = 1, plus: bool = False) -> str:
    return fmt_bn(v, "€", dp, plus=plus)


def num(v, dp: int = 1) -> str:
    return fmt_bn(v, "", dp, unit="")


def table(rows: list[list], header: list[str], left: int = 1, titles: list[str | None] | None = None) -> str:
    """HTML table of pre-formatted cells; ``titles`` adds a hover title to the first cell of each row."""
    def cell(v, i, tag, title=None):
        cls = " class='l'" if i < left else " class='n'"
        t = f" title='{html.escape(title)}'" if title else ""
        return f"<{tag}{cls}{t}>{html.escape('—' if v is None else str(v))}</{tag}>"
    head = "<tr>" + "".join(cell(h, i, "th") for i, h in enumerate(header)) + "</tr>"
    body = "".join("<tr>" + "".join(cell(v, i, "td", (titles[j] if titles and i == 0 else None)) for i, v in enumerate(r))
                   + "</tr>" for j, r in enumerate(rows))
    return f"<table class='sc'>{head}{body}</table>"


def score_table(score: list[dict], production: str) -> str:
    rel = [{**r, "all_mae": {label(k): v for k, v in r.get("all_mae", {}).items()}} for r in score]
    return D.score_table(rel, label(production))


def _bar(fig: go.Figure, x, y, name: str, color: str, hover: str = "%{y:.1f}bn") -> None:
    y = pd.Series(y, dtype=float)
    if y.fillna(0).abs().sum() < 1e-9:
        return
    fig.add_bar(x=x, y=y, name=name, marker_color=color, hovertemplate=f"{name}: {hover}<extra></extra>")


def fig_sources_uses(frame: pd.DataFrame, back: int = 24) -> go.Figure:
    from sovereign_cash.horizon import clip_frame, programme_end
    end = programme_end(frame)
    d = clip_frame(frame).set_index("month")
    fc = d.index[d["is_forecast"].astype(bool)] if "is_forecast" in d.columns else d.index[:0]
    d = d[d.index >= (fc[0] if len(fc) else d.index.max()) - pd.DateOffset(months=back)]
    fig = go.Figure()
    _bar(fig, d.index, -d["cash_requirement_bn"], "State deficit (use, −)", RED)
    _bar(fig, d.index, d["net_bonds_bn"], "net OAT", BLUE)
    _bar(fig, d.index, d["net_bills_bn"], "net BTF", GREEN)
    _bar(fig, d.index, d["other_fin_bn"], "other financing", GREY)
    _bar(fig, d.index, -d["buffer_change_bn"], "Treasury placements (build −)", ORANGE)
    if "bond_redemption_cb_bn" in d.columns:
        cb = d["bond_redemption_cb_bn"].where(d["is_forecast"].astype(bool))
        if cb.fillna(0).abs().sum() > 0:
            fig.add_trace(go.Scatter(x=d.index, y=cb, name="Eurosystem-held OAT redemptions (est.)", mode="lines+markers",
                                     line=dict(color=PURPLE, width=1.6), marker=dict(size=4),
                                     hovertemplate="Eurosystem-held redemptions (est.): %{y:.1f}bn<extra></extra>"))
    if len(fc):
        fig.add_vline(x=fc[0] - pd.Timedelta(days=15), line_dash="dot", line_color=GREY)
    fig.update_layout(barmode="relative")
    note = f"; charts stop {end:%b-%Y} (last OAT gross month)" if end is not None else ""
    return _style(fig, title="Sources and uses, EUR bn (each month sums to zero; forecast after dotted line"
                  + note + ")",
                  height=460, ylab="EUR bn", rangeslider=True)


def fig_oat(frame: pd.DataFrame) -> go.Figure:
    from sovereign_cash.horizon import clip_frame, programme_end
    end = programme_end(frame)
    d = clip_frame(frame)
    d = d[d["is_forecast"].astype(bool)].set_index("month")
    bb = d["bond_buyback_bn"].fillna(0.0) if "bond_buyback_bn" in d.columns else pd.Series(0.0, index=d.index)
    fig = go.Figure()
    _bar(fig, d.index, d["bond_gross_bn"], "OAT gross", BLUE)
    _bar(fig, d.index, -(d["bond_redemption_bn"] - bb), "maturities (−)", RED)
    _bar(fig, d.index, -bb, "buybacks (−)", ORANGE)
    fig.add_trace(go.Scatter(x=d.index, y=d["net_bonds_bn"], name="net OAT", mode="lines+markers",
                             line=dict(color=GREY, width=1.6), marker=dict(size=4)))
    if "private_net_bonds_bn" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["private_net_bonds_bn"], name="private net OAT (est.)",
                                 mode="lines+markers", line=dict(color=PURPLE, width=1.6, dash="dot"), marker=dict(size=4)))
    fig.update_layout(barmode="relative")
    return _style(fig, title="OAT supply by month, EUR bn (through programme horizon)", ylab="EUR bn",
                  rangeslider=True)


def fig_btf(frame: pd.DataFrame, path_dec: float | None, plan_year: int) -> go.Figure:
    from sovereign_cash.horizon import clip_frame
    d = clip_frame(frame).set_index("month")
    d = d[d.index >= d.index.max() - pd.DateOffset(years=4)]
    act = d["bill_stock_bn"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=act.index, y=act, name="BTF outstanding (AFT)", line=dict(color=BLUE, width=1.8)))
    imp = d["bill_stock_implied_bn"].dropna()
    imp = imp[imp.index > act.index.max()] if len(act) else imp
    if len(imp):
        imp = pd.concat([act.iloc[-1:], imp])
        fig.add_trace(go.Scatter(x=imp.index, y=imp, name="BTF forward path", line=dict(color=ORANGE, width=1.8, dash="dash")))
    if path_dec is not None:
        fig.add_trace(go.Scatter(x=[pd.Timestamp(plan_year, 12, 1)], y=[path_dec], name=f"AFT end-{plan_year} target",
                                 mode="markers", marker=dict(color=RED, size=10, symbol="diamond")))
    return _style(fig, title="BTF outstanding, EUR bn", ylab="EUR bn", rangeslider=True)


def fig_ytd(frame: pd.DataFrame, yr: int, plan_fy: float | None, model_fy: float, band: float | None,
            years: int = 5) -> go.Figure:
    d = frame.set_index("month")
    s = d["cash_requirement_bn"]
    obs_all = d["cash_observed"].fillna(False).astype(bool)
    pal = [GREY, PURPLE, GREEN, BLUE, RED]
    fig = go.Figure()
    for i, y in enumerate(range(yr - years + 1, yr + 1)):
        sy = s[s.index.year == y].dropna()
        if sy.empty:
            continue
        cum = sy.cumsum()
        obs = obs_all.reindex(sy.index).fillna(False).to_numpy()
        x = sy.index.month
        c = pal[i % len(pal)]
        if obs.any():
            fig.add_trace(go.Scatter(x=x[obs], y=cum[obs], name=str(y), mode="lines+markers",
                                     line=dict(color=c, width=1.6), marker=dict(size=4)))
        if (~obs).any():
            k0 = max(int(np.where(~obs)[0][0]) - 1, 0)
            fig.add_trace(go.Scatter(x=x[k0:], y=cum.iloc[k0:], name=f"{y} model", mode="lines",
                                     line=dict(color=c, width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=[12], y=[model_fy], name=f"{yr} model full year" + (" ± OOS MAE" if band else ""),
                             mode="markers", marker=dict(color=RED, size=8),
                             error_y=dict(type="data", array=[band or 0], visible=bool(band), color=RED)))
    if plan_fy is not None:
        fig.add_trace(go.Scatter(x=[12], y=[plan_fy], name=f"{yr} plan", mode="markers",
                                 marker=dict(color=RED, size=10, symbol="diamond-open")))
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return _style(fig, title="Cumulative State deficit by calendar year, EUR bn (dashed = model)", ylab="EUR bn")


def build(path, title: str, intro: str, asof: str, kpis: list[tuple[str, str, str]], lead: list[tuple[str, str]],
          figs: list[go.Figure], score_html: str, bill_score_html: str, score_note: str, sources: list[dict],
          notes: list[str], scenario_html: str = "",
          frame: pd.DataFrame | None = None, summary: dict | None = None) -> None:
    charts = "".join(f'<div class="card">{_html(f, first=i == 0)}</div>' for i, f in enumerate(figs))
    kpi_html = "".join(_kpi(*k) for k in kpis)
    lead_html = "".join(f'<h2 class="section">{html.escape(t)}</h2>{body}' for t, body in lead)
    src = table([[s["name"], s["frequency"], s.get("history", ""), "yes" if s["verified"] else "no"] for s in sources],
                ["source", "freq", "coverage", "verified"])
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    sx = (f'<h2 class="section">Macro scenario (model-implied)</h2><div class="card">{scenario_html}</div>'
          if scenario_html else "")
    bu = bottomup_card(summary or {})
    season = ""
    if frame is not None:
        from sovereign_cash.horizon import clip_frame, programme_end
        end = programme_end(frame)
        fr_season = clip_frame(frame)
        horizon = f"through {end:%b-%Y} (OAT programme end)" if end is not None else "through the live horizon"
        season = ("<h2 class=\"section\">Through the year</h2>"
                  + season_html(fr_season, cc="fr", prefix="fr",
                                specs=default_specs("cash_requirement_bn", "State deficit"),
                                fy_start_month=1, currency="€", tall_key="cash",
                                forecast_note=f"Forecast months follow the live cash model {horizon}. "
                                "Past the last gross OAT month the identity is not a financing forecast."))
    try:
        cat_panel, cat_specs = fr_category_panel()
        cat_html = category_tape_html(
            cat_panel, cc="fr", prefix="fr", specs=cat_specs, currency="€",
            note="SMB four-line identity (recettes / primary / charge / spéciaux) under the shipped rules.")
    except Exception:  # noqa: BLE001
        cat_html = ""
    built = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-3.7.0.min.js" charset="utf-8"></script>
<style>{PAGE_CSS}{TABLE_CSS}{EXTRA_CSS}</style></head>
<body>
<header><h1>{html.escape(title)}</h1>
<div class="sub">built {built} · {html.escape(asof)} · {html.escape(intro)}</div></header>
<div class="wrap">
<div class="kpi-row">{kpi_html}</div>
{lead_html}
{bu}
{season}
{cat_html}
<h2 class="section">Charts</h2><div class="grid">{charts}</div>
<h2 class="section">Deficit model backtest — MAE of the cumulative deficit over h months, EUR bn</h2>
<p class="sub">{html.escape(score_note)} ◆ = PRE-selected seasonal model ({html.escape(seasonal_role("fr", summary or {}) or "live")}). Bold = lowest on the common sample.</p>
{score_html}
{sx}
<h2 class="section">BTF stock backtest — MAE, EUR bn</h2>
{bill_score_html}
<h2 class="section">Sources</h2>{src}
<h2 class="section">Notes</h2><ul class="notes">{notes_html}</ul>
</div></body></html>"""
    path.write_text(page, encoding="utf-8")
