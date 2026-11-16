"""Standalone cash-framework dashboard (shared by de and fr).

Reads the contract frame and summary already in memory and writes
``sovereign_cash/<cc>_cash_dashboard.html``.
"""

from __future__ import annotations

import datetime as dt
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
US = ROOT.parent / "us" / "fiscal"
if str(US) not in sys.path:
    sys.path.insert(0, str(US))
from build_dashboard import BLUE, GREEN, GREY, ORANGE, PAGE_CSS, PURPLE, RED, _html, _kpi, _style  # noqa: E402
from sovereign_cash.cash_season import (  # noqa: E402
    SEASON_CSS, bottomup_card, category_tape_html, default_specs,
    de_category_panel, fr_category_panel, season_html,
)

TABLE_CSS = """
table.sc { border-collapse: collapse; font: 11px/1.4 var(--mono); margin: 6px 0 10px; }
table.sc th, table.sc td { border: 1px solid var(--grid); padding: 3px 8px; text-align: right; }
table.sc th { color: var(--muted); font-weight: 600; }
table.sc td.l, table.sc th.l { text-align: left; }
table.sc td.best { font-weight: 700; color: var(--accent); }
table.sc th.prod, table.sc td.prod { background: rgba(255, 170, 0, 0.12); }
ul.notes { font-size: 11px; color: var(--muted); max-width: 1080px; }
p.lead { font-size: 13px; max-width: 1080px; }
""" + SEASON_CSS


def table(rows: list[list], header: list[str], left: int = 1) -> str:
    """Plain HTML table; the first ``left`` columns are left-aligned."""
    def cell(v, i, tag):
        cls = " class='l'" if i < left else ""
        txt = f"{v:,.1f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        return f"<{tag}{cls}>{html.escape(txt)}</{tag}>"
    head = "<tr>" + "".join(cell(h, i, "th") for i, h in enumerate(header)) + "</tr>"
    body = "".join("<tr>" + "".join(cell(v, i, "td") for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<table class='sc'>{head}{body}</table>"


COVID_YEARS = (2020, 2021)


def _fig_ytd(df: pd.DataFrame, col: str = "cash_requirement_bn", years: int = 6, label: str = "",
             band: dict | None = None, ylab: str = "bn") -> go.Figure:
    """Cumulative cash requirement by calendar year; months not observed are dashed.

    ``band`` = {lo, hi, base, name}: shaded range on the current year's unobserved months, with the
    ``base`` component of ``col`` swapped for the ``lo`` / ``hi`` columns.
    """
    fig = go.Figure()
    d = df.set_index("month")
    s = d[col]
    obs_all = d["cash_observed"].fillna(False).astype(bool) if "cash_observed" in d else ~d["is_forecast"].astype(bool)
    obs_mask = obs_all.reindex(s.index).fillna(False).to_numpy()
    cur_year = s.index[obs_mask].max().year if obs_mask.any() else s.index.max().year
    pal = [GREY, PURPLE, GREEN, ORANGE, BLUE, RED]
    for i, y in enumerate(range(cur_year - years + 1, cur_year + 1)):
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
            if band and y == cur_year and {band["lo"], band["hi"], band["base"]} <= set(d.columns):
                for j, bc in enumerate((band["lo"], band["hi"])):
                    by = (sy + (d[bc] - d[band["base"]]).reindex(sy.index).fillna(0.0)).cumsum()
                    fig.add_trace(go.Scatter(x=x[k0:], y=by.iloc[k0:], mode="lines", line=dict(width=0, color=c),
                                             fill="tonexty" if j else None, fillcolor="rgba(120,120,120,0.22)",
                                             name=band["name"], showlegend=bool(j)))
            fig.add_trace(go.Scatter(x=x[k0:], y=cum.iloc[k0:], name=f"{y} forecast", mode="lines",
                                     line=dict(color=c, width=1.6, dash="dash")))
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    return _style(fig, title=f"Cumulative cash requirement by year{label} (deficit +; dashed = not yet observed)",
                  ylab=ylab)


def _fig_identity(df: pd.DataFrame, gap_col: str | None, months: int = 60) -> go.Figure:
    d = df.set_index("month").iloc[-months:]
    fig = go.Figure()
    fig.add_bar(x=d.index, y=-d["cash_requirement_bn"], name="−cash requirement", marker_color=RED)
    fig.add_bar(x=d.index, y=d["net_bonds_bn"], name="net bonds", marker_color=BLUE)
    fig.add_bar(x=d.index, y=d["net_bills_bn"], name="net bills", marker_color=GREEN)
    fig.add_bar(x=d.index, y=d["other_fin_bn"], name="other financing", marker_color=GREY)
    fig.add_trace(go.Scatter(x=d.index, y=d["buffer_change_bn"], name="Δ buffer", mode="lines+markers",
                             line=dict(color=ORANGE, width=1.6), marker=dict(size=4)))
    if gap_col and gap_col != "other_fin_bn" and gap_col in d:
        fig.add_trace(go.Scatter(x=d.index, y=d[gap_col], name="unexplained part of other financing", mode="lines",
                                 line=dict(color=PURPLE, width=1.2, dash="dot")))
    fc0 = d.index[d["is_forecast"].astype(bool)]
    if len(fc0):
        fig.add_vline(x=fc0[0], line_dash="dot", line_color=GREY)
    fig.update_layout(barmode="relative")
    return _style(fig, title="Monthly identity: Δbuffer = −cash requirement + net bonds + net bills + other financing",
                  height=460, ylab="bn")


def _fig_stock(df: pd.DataFrame, label: str, extra: list[pd.Series] | None = None,
               official: pd.Series | None = None, title: str | None = None, ylab: str = "bn") -> go.Figure:
    from sovereign_cash.horizon import clip_frame, clip_series, programme_end
    end = programme_end(df)
    d = clip_frame(df).set_index("month")
    d = d[d.index >= d.index.max() - pd.DateOffset(years=6)]
    act = d["bill_stock_bn"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=act.index, y=act, name=f"{label} outstanding (actual)", line=dict(color=BLUE, width=1.8)))
    if "bill_stock_implied_bn" in d:
        imp = d["bill_stock_implied_bn"].dropna()
        last = act[act.index < imp.index.min()] if len(imp) else act
        if len(last):
            imp = pd.concat([last.iloc[-1:], imp])
        fig.add_trace(go.Scatter(x=imp.index, y=imp, name=f"{label} forward path", line=dict(color=ORANGE, width=1.8, dash="dash")))
    if official is not None and len(official):
        off = clip_series(official, end)
        fig.add_trace(go.Scatter(x=off.index, y=off.values, name=off.name, mode="lines+markers",
                                 line=dict(color=RED, width=1.2, dash="dot"), marker=dict(size=7, symbol="diamond")))
    for s, c in zip(extra or [], [GREEN, PURPLE, GREY]):
        ss = clip_series(s, end)
        dash = "dot" if "plan" in str(ss.name) else None
        fig.add_trace(go.Scatter(x=ss.index, y=ss.values, name=ss.name, line=dict(color=c, width=1.4, dash=dash)))
    return _style(fig, title=title or f"Short-term path: {label} stock", ylab=ylab)


def _fig_forward(df: pd.DataFrame, core_col: str | None = None, sv_cols: dict[str, str] | None = None,
                 cb_label: str = "Eurosystem-held redemptions", ylab: str = "bn") -> go.Figure:
    """Forward months as sources (+) vs uses (−), stacked; Δbuffer and private net bonds as lines."""
    from sovereign_cash.horizon import clip_frame, programme_end
    end = programme_end(df)
    d = clip_frame(df)
    d = d[d["is_forecast"].astype(bool)].set_index("month")
    fig = go.Figure()
    uses = []
    if core_col and core_col in d:
        uses.append(("core budget", d[core_col], RED))
        rest = d["cash_requirement_bn"] - d[core_col]
        for (name, col), c in zip((sv_cols or {}).items(), [ORANGE, "#c0703a", "#e8b04a"]):
            if col in d:
                uses.append((name, d[col], c))
                rest = rest - d[col]
        if not sv_cols or rest.abs().max() > 1e-6:
            uses.append(("other cash requirement" if sv_cols else "special funds", rest, ORANGE if not sv_cols else GREY))
    else:
        uses.append(("cash requirement", d["cash_requirement_bn"], RED))
    for name, s, c in uses:
        fig.add_bar(x=d.index, y=-s, name=f"use: {name}", marker_color=c)
    fig.add_bar(x=d.index, y=d["net_bonds_bn"], name="source: net bonds", marker_color=BLUE)
    fig.add_bar(x=d.index, y=d["net_bills_bn"], name="source: net bills", marker_color=GREEN)
    fig.add_bar(x=d.index, y=d["other_fin_bn"], name="source: other financing", marker_color=GREY,
                marker_pattern_shape="/")
    fig.add_trace(go.Scatter(x=d.index, y=d["buffer_change_bn"], name="Δ buffer (= sum of bars)", mode="lines+markers",
                             line=dict(color=ORANGE, width=1.6), marker=dict(size=4)))
    if "bond_redemption_cb_bn" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["net_bonds_bn"] + d["bond_redemption_cb_bn"],
                                 name=f"private net bonds (net bonds + {cb_label})", mode="lines",
                                 line=dict(color=PURPLE, width=1.6, dash="dot")))
    fig.update_layout(barmode="relative")
    note = f" through {end:%b-%Y}" if end is not None else ""
    return _style(fig, title=f"Forward months: sources (+) vs uses (−) of cash{note}", ylab=ylab)


def _fig_gap_profile(df: pd.DataFrame, gap_col: str, ylab: str = "bn") -> go.Figure:
    d = df[~df["is_forecast"].astype(bool)].set_index("month")[gap_col].dropna()
    d = d[~d.index.year.isin(COVID_YEARS)]
    g = d.groupby(d.index.month)
    last5 = d[d.index >= d.index.max() - pd.DateOffset(years=5)]
    fig = go.Figure()
    fig.add_bar(x=list(range(1, 13)), y=g.mean().reindex(range(1, 13)),
                name=f"mean {d.index.min():%Y}–{d.index.max():%Y} (whiskers ±1 sd)",
                marker_color=GREY, error_y=dict(type="data", array=g.std().reindex(range(1, 13)), thickness=0.8,
                                                width=3, color=GREY))
    fig.add_bar(x=list(range(1, 13)), y=last5.groupby(last5.index.month).mean().reindex(range(1, 13)),
                name="mean, last 5 years", marker_color=PURPLE)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    cov = "–".join(str(y) for y in (COVID_YEARS[0], COVID_YEARS[-1]))
    return _style(fig, title=f"Unexplained financing by calendar month, excluding {cov}", ylab=ylab)


def score_table(score: list[dict], production: str | None = None, models: list[str] | None = None) -> str:
    """MAE by window × h; the production model's column is shaded and marked ◆. ``models`` limits the columns."""
    if not score:
        return "<p>No score.</p>"
    have = {m for r in score for m in r.get("all_mae", {})}
    models = [m for m in dict.fromkeys(models) if m in have] if models else sorted(have)
    head = "<tr><th class='l'>window</th><th>h</th><th>n (prod.)</th>" + "".join(
        f"<th class='{'prod' if m == production else ''}'>{html.escape(m)}{' ◆' if m == production else ''}</th>"
        for m in models) + "</tr>"
    body = []
    for r in score:
        am = r.get("all_mae", {})
        best = min((m for m in models if m in am and am[m]["common"]), key=lambda m: am[m]["mae"], default=None)
        cells = []
        for m in models:
            v = am.get(m)
            txt = "" if v is None else f"{v['mae']:.1f}" + ("" if v["common"] else f"* (n={v['n']})")
            cls = " ".join(c for c in ("best" if m == best else "", "prod" if m == production else "") if c)
            cells.append(f"<td class='{cls}'>{txt}</td>")
        body.append(f"<tr><td class='l'>{r['window']}</td><td>{r['h']}</td><td>{r['n']}</td>{''.join(cells)}</tr>")
    return f"<table class='sc'>{head}{''.join(body)}</table>"


def build(cc: str, title: str, intro: str, frame: pd.DataFrame, summary: dict, *,
          bill_label: str, kpis: list[tuple[str, str, str]], lead: list[tuple[str, str]],
          gap_col: str | None = None, ytd_col: str = "cash_requirement_bn", ytd_label: str = "",
          core_col: str | None = None, production: str | None = None,
          bill_score: list[dict] | None = None, bill_score_title: str = "", bill_production: str | None = None,
          extra_stock: list[pd.Series] | None = None, official_stock: pd.Series | None = None,
          extra_figs: list[go.Figure] | None = None, model_legend: str = "",
          stock_title: str | None = None, ytd_band: dict | None = None, sv_cols: dict[str, str] | None = None,
          cb_label: str = "Eurosystem-held redemptions", score_models: list[str] | None = None,
          ytd_years: int = 6, scenario_html: str = "",
          cash_label: str = "Cash requirement", currency: str = "€") -> Path:
    out = ROOT / f"{cc}_cash_dashboard.html"
    from sovereign_cash.investor import seasonal_role
    role = seasonal_role(cc, summary)
    prod_legend = f"◆ = {production} ({role})." if production and role else "◆ = production model."
    ylab = f"{summary.get('ccy', '')} bn".strip()
    figs = [_fig_forward(frame, core_col, sv_cols, cb_label, ylab=ylab),
            _fig_stock(frame, bill_label, extra_stock, official_stock, title=stock_title, ylab=ylab),
            _fig_ytd(frame, ytd_col, years=ytd_years, label=ytd_label, band=ytd_band, ylab=ylab),
            _fig_identity(frame, gap_col)]
    figs[-1].update_layout(yaxis_title=ylab)
    if gap_col:
        figs.append(_fig_gap_profile(frame, gap_col, ylab=ylab))
    figs += list(extra_figs or [])
    charts = "".join(f'<div class="card">{_html(f, first=i == 0)}</div>' for i, f in enumerate(figs))
    kpi_html = "".join(_kpi(*k) for k in kpis)
    lead_html = "".join(f'<h2 class="section">{html.escape(t)}</h2>{body}' for t, body in lead)
    src_tbl = table([[s["name"], s["frequency"], s.get("lag_days", ""), "yes" if s["verified"] else "NO",
                      str(s.get("history", ""))] for s in summary["sources"]],
                    ["source", "freq", "lag d", "verified", "history"], left=1)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in summary.get("notes", []))
    bill_tbl = score_table(bill_score, bill_production) if bill_score else ""
    bu = bottomup_card(summary)
    from sovereign_cash.horizon import clip_frame as _clip
    season = season_html(
        _clip(frame), cc=cc, prefix=cc,
        specs=default_specs(ytd_col, cash_label),
        fy_start_month=1, currency=currency,
        tall_key="cash",
        forecast_note="Forecast months follow the live cash model through the coupon programme "
        "(charts stop when gross bond issuance goes to zero; past that the buffer path is not a forecast).",
    )
    cat_html = ""
    try:
        if cc == "de":
            cat_panel, cat_specs = de_category_panel()
            cat_html = category_tape_html(
                cat_panel, cc="de", prefix="de", specs=cat_specs, currency=currency,
                note="BMF Steuerarten / Ausgaben — history plus same-month-last-year beyond the origin.")
        elif cc == "fr":
            cat_panel, cat_specs = fr_category_panel()
            cat_html = category_tape_html(
                cat_panel, cc="fr", prefix="fr", specs=cat_specs, currency=currency,
                note="SMB four-line identity (recettes / primary / charge / spéciaux) under the shipped rules.")
    except Exception:  # noqa: BLE001
        cat_html = ""
    built = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-3.7.0.min.js" charset="utf-8"></script>'
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
{plotly_cdn}
<style>{PAGE_CSS}{TABLE_CSS}</style></head>
<body>
<header><h1>{html.escape(title)}</h1>
<div class="sub">built {built} · as of {summary['asof']} · {html.escape(intro)}</div></header>
<div class="wrap">
<div class="kpi-row">{kpi_html}</div>
{lead_html}
{bu}
<h2 class="section">Through the year</h2>
{season}
{cat_html}
<h2 class="section">Charts</h2><div class="grid">{charts}</div>
<h2 class="section">Deficit backtest — cumulative cash requirement MAE (bn)</h2>
<p class="sub">{html.escape(summary.get('score_note', ''))} {html.escape(model_legend)} {prod_legend}
Bold = lowest on the common-origin sample. * = scored on its own origins (not on the common sample).</p>
{score_table(summary.get('score', []), production, score_models)}
{(f'<h2 class="section">Macro scenario (model-implied)</h2><div class="card">{scenario_html}</div>'
  if scenario_html else '')}
{('<h2 class="section">' + html.escape(bill_score_title) + '</h2>' + bill_tbl) if bill_tbl else ''}
<h2 class="section">Sources</h2>{src_tbl}
<h2 class="section">Notes</h2><ul class="notes">{notes}</ul>
</div></body></html>"""
    out.write_text(page, encoding="utf-8")
    return out
