"""Standalone UK cash page: sovereign_cash/uk_cash_dashboard.html (plotly, us/fiscal theme).

Every number and sentence on the page is generated from the summary / monthly frame.
"""

from __future__ import annotations

import datetime as dt
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio

from sovereign_cash.cash_season import SEASON_CSS, category_tape_html, uk_category_panel
from sovereign_cash.contract import fmt_bn, fmt_num
from sovereign_cash.uk import model as M

REPO = Path(__file__).resolve().parents[2]
_FISCAL = REPO / "us" / "fiscal"
if str(_FISCAL) not in sys.path:
    sys.path.insert(0, str(_FISCAL))
from build_dashboard import (BLUE, GREEN, GREY, MATLAB, ORANGE, PAGE_CSS, PURPLE, RED,  # noqa: E402
                             _PLOT_CONFIG, _kpi, _style)

OUT = REPO / "sovereign_cash" / "uk_cash_dashboard.html"
GBP = "£"
ZOOM_MONTHS = 24
FC_OPACITY = 0.45
ASSUMED_PATTERN = dict(shape="/", fgopacity=0.6, size=6)

# One colour per series across the page.
C = {
    "cgncr": RED, "gilts": BLUE, "redeem_private": "#1F3B73", "redeem_apf": "#4DBEEE", "bills": GREEN,
    "bills_debtfin": "#3B6E1E", "other": PURPLE, "buffer": "#222222", "plan": ORANGE, "bench": GREY,
    "sonia": "#8C564B", "deposits": "#17BECF", "nram": "#BCBD22", "gilt_gap": "#9ECAE9",
    "bill_gap": "#B5DDA4", "unexplained": "#666666", "more": "#555555", "less": "#AAAAAA",
}

TABLE_CSS = """
table.t { border-collapse: collapse; font: 11px/1.4 var(--mono); margin: 6px 0 4px; }
table.t th, table.t td { border-bottom: 1px solid var(--grid); padding: 3px 8px; text-align: right; }
table.t th:first-child, table.t td:first-child { text-align: left; }
table.t td.win { font-weight: 700; }
ul.lead { margin: 6px 0 0 18px; padding: 0; font-size: 12px; line-height: 1.5; }
.tag { font: 600 9px var(--mono); color: var(--muted); border: 1px solid var(--line); padding: 0 4px; margin-left: 4px; }
.toggles { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; align-items: center; }
.toggles button { font: 600 11px/1 var(--mono); padding: 7px 12px; border: 1px solid var(--line);
  background: var(--panel-hi); color: var(--text); cursor: pointer; }
.toggles button.on { background: var(--text); color: var(--panel-hi); }
.toggles .chk { font: 600 11px/1 var(--mono); display: flex; align-items: center; gap: 6px; }
.season { height: 420px; } .season.tall { height: 500px; }
details.fold { margin-top: 18px; }
details.fold summary { font: 700 11px/1.3 var(--mono); cursor: pointer; letter-spacing: .04em;
  text-transform: uppercase; }
"""


def g(v, dp: int = 1, plus: bool = False) -> str:
    return fmt_bn(v, GBP, dp, plus=plus)


def n(v, dp: int = 1, plus: bool = False) -> str:
    return fmt_num(v, dp, plus=plus)


def _div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config=_PLOT_CONFIG)


def _fy_label(fy: int) -> str:
    return f"FY{fy}-{str(fy + 1)[2:]}"


def _zoom(fig: go.Figure, end: pd.Timestamp, months: int = ZOOM_MONTHS) -> None:
    fig.update_xaxes(range=[end - pd.DateOffset(months=months), end + pd.DateOffset(days=20)])


# --------------------------------------------------------------------------------------------
# Charts


def fig_identity(df: pd.DataFrame) -> go.Figure:
    from sovereign_cash.horizon import clip_frame, programme_end
    end = programme_end(df)
    d = clip_frame(df)
    d = d[d["month"] >= pd.Timestamp("2016-04-01")].copy()
    fig = go.Figure()
    series = (("cash_requirement_bn", "−CGNCR", C["cgncr"]), ("net_bonds_bn", "Net gilt cash", C["gilts"]),
              ("net_bills_bn", "Change in T-bill stock", C["bills"]),
              ("other_fin_bn", "Other financing (NS&I, deposits, gaps)", C["other"]))
    for fc in (False, True):
        h = d[d["is_forecast"] == fc]
        for col, name, colr in series:
            y = -h[col] if col == "cash_requirement_bn" else h[col]
            fig.add_bar(x=h["month"], y=y, name=name + (" (forecast)" if fc else ""), marker_color=colr,
                        opacity=FC_OPACITY if fc else 1.0, legendgroup=col, showlegend=not fc)
    fig.add_scatter(x=d["month"], y=d["buffer_change_bn"], name="Change in buffer",
                    mode="lines+markers", line=dict(color=C["buffer"], width=1.2), marker=dict(size=3))
    fig.update_layout(barmode="relative")
    note = f" (through {end:%b-%Y})" if end is not None else ""
    _style(fig, title="Cash identity, £bn a month: change in buffer = −CGNCR + net gilts + change in bills + other"
           + note,
           ylab="£bn a month", height=440, rangeslider=True)
    _zoom(fig, d["month"].max())
    return fig


def fig_gilts(df: pd.DataFrame, s: dict) -> go.Figure:
    fwd, e = s["forward"], s["forward"]["end_march"]
    fy = int(e["fy"][:4])
    d = df[df["month"] >= pd.Timestamp(fy, 4, 1)].copy()
    d = d[d["month"] <= df.loc[df["is_forecast"], "month"].max()]
    d["red_private"] = d["bond_redemption_bn"] - d["bond_redemption_cb_bn"]
    fig = go.Figure()
    kinds = (("act", ~d["is_forecast"]), ("fc", d["is_forecast"] & ~d["gilts_assumed"]),
             ("assumed", d["is_forecast"] & d["gilts_assumed"]))
    suffix = {"act": "", "fc": " (forecast)", "assumed": " (assumed, no remit)"}
    for k, mask in kinds:
        h = d[mask]
        if h.empty:
            continue
        op = 1.0 if k == "act" else FC_OPACITY
        pat = ASSUMED_PATTERN if k == "assumed" else None
        for col, name, colr, sign in (("bond_gross_bn", "Gross gilt sales", C["gilts"], 1),
                                      ("red_private", "Redemptions to private holders", C["redeem_private"], -1),
                                      ("bond_redemption_cb_bn", "Redemptions of APF-held gilts", C["redeem_apf"], -1)):
            fig.add_bar(x=h["month"], y=sign * h[col], name=name + suffix[k], marker_color=colr, opacity=op,
                        marker_pattern=pat, legendgroup=col, showlegend=(k == "act"))
    in_fy = d[d["month"] < pd.Timestamp(fy + 1, 4, 1)]
    cum = in_fy["bond_gross_bn"].cumsum()
    fig.add_scatter(x=in_fy["month"], y=cum, name=f"Cumulative {_fy_label(fy)} sales (rhs)", yaxis="y2",
                    mode="lines+markers", line=dict(color=C["gilts"], dash="dot"), marker=dict(size=4))
    fig.add_scatter(x=[in_fy["month"].min(), in_fy["month"].max()], y=[e["gilt_sales_plan_bn"]] * 2,
                    name=f"Remit {g(e['gilt_sales_plan_bn'])} (rhs)", yaxis="y2", mode="lines",
                    line=dict(color=C["plan"], dash="dash"))
    fig.update_layout(barmode="relative",
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="£bn cumulative", rangemode="tozero"))
    return _style(fig, title=f"Gilt cash by month, £bn: {_fy_label(fy)} sales vs remit, redemptions by holder",
                  ylab="£bn a month", rmargin=60, height=400)


def fig_cgncr(df: pd.DataFrame, e: dict) -> go.Figure:
    from sovereign_cash.horizon import clip_frame
    df = clip_frame(df)
    h = df[(~df["is_forecast"]) & (df["month"] >= df["month"].max() - pd.DateOffset(months=47))]
    f = df[df["is_forecast"]]
    plan_on = e["cgncr_source_in_fy"] == "plan" and e["cgncr_source_after_fy"] == "plan"
    fig = go.Figure()
    fig.add_bar(x=h["month"], y=h["cash_requirement_bn"], name="Actual", marker_color=C["cgncr"])
    hatch = dict(shape=np.where(f["gilts_assumed"], "/", "").tolist(), fgopacity=0.6, size=6)
    fig.add_bar(x=f["month"], y=f["cash_requirement_fc_official_bn"], name=M.label(M.OFFICIAL),
                marker_color=C["plan"], opacity=FC_OPACITY, marker_pattern=hatch, visible=plan_on)
    fig.add_bar(x=f["month"], y=f["cash_requirement_model_bn"], name="Seasonal model",
                marker_color=C["cgncr"], opacity=FC_OPACITY, marker_pattern=hatch, visible=not plan_on)
    fig.add_scatter(x=f["month"], y=f["cash_requirement_fc_official_bn"], name=M.label(M.OFFICIAL),
                    mode="markers+lines", line=dict(color=C["plan"], dash="dot"), marker=dict(size=5),
                    showlegend=False)
    fig.add_scatter(x=f["month"], y=f["cash_requirement_model_bn"], name="Seasonal model",
                    mode="markers", marker=dict(color=C["cgncr"], symbol="diamond-open", size=8),
                    showlegend=False)
    fig.add_scatter(x=f["month"], y=f["cgncr_smly_bn"], name="Same month last year",
                    mode="markers", marker=dict(color=C["bench"], symbol="circle-open", size=7))

    def _vis(plan: bool) -> list[bool]:
        return [True, plan, not plan, True, True, True]

    fig.update_layout(updatemenus=[dict(
        type="dropdown", direction="down", x=0, y=1.14, xanchor="left", yanchor="top",
        bgcolor="white", bordercolor="#ccc",
        buttons=[
            dict(label="OBR / remit", method="update", args=[{"visible": _vis(True)}]),
            dict(label="Seasonal model", method="update", args=[{"visible": _vis(False)}]),
        ],
        active=0 if plan_on else 1,
    )])
    return _style(fig, title="CGNCR (ex NRAM, B&B, Network Rail), £bn a month",
                  ylab="£bn a month")


def fig_cgncr_cum(df: pd.DataFrame, e: dict) -> go.Figure:
    fy = int(e["fy"][:4])
    x = df.loc[~df["is_forecast"]].set_index("month")["cash_requirement_bn"]
    origin = x.index.max()
    prof = M.month_profile(x, origin)
    months = pd.date_range(pd.Timestamp(fy, 4, 1), periods=12, freq="MS")
    lab = [f"{m:%b}" for m in months]
    allm = df.set_index("month")["cash_requirement_bn"]
    cur = allm.reindex(months)
    is_fc = df.set_index("month")["is_forecast"].reindex(months).fillna(True).astype(bool)
    cum = cur.cumsum()
    last = x.reindex(months - pd.DateOffset(years=1)).cumsum()
    plan = e["cgncr_ex_plan_bn"] * np.cumsum([prof[m.month] for m in months])
    fig = go.Figure()
    k = int((~is_fc).sum())
    fig.add_scatter(x=lab[:k], y=cum.iloc[:k], name=f"{_fy_label(fy)} actual", mode="lines+markers",
                    line=dict(color=C["cgncr"], width=2.4))
    fig.add_scatter(x=lab[max(k - 1, 0):], y=cum.iloc[max(k - 1, 0):], name="Forecast", mode="lines",
                    line=dict(color=C["cgncr"], dash="dash"))
    fig.add_scatter(x=lab, y=plan, name=f"OBR / remit {g(e['cgncr_ex_plan_bn'])}", mode="lines",
                    line=dict(color=C["plan"], dash="dot"))
    fig.add_scatter(x=lab, y=last, name=f"{_fy_label(fy - 1)} actual", mode="lines",
                    line=dict(color=C["bench"]))
    return _style(fig, title=f"CGNCR {_fy_label(fy)} cumulative, £bn: year to date vs plan profile and last year",
                  ylab="£bn cumulative")


def fig_bills(df: pd.DataFrame, bt_bl: pd.DataFrame, model: str, e: dict) -> go.Figure:
    from sovereign_cash.horizon import clip_frame
    df = clip_frame(df)
    start = df["month"].max() - pd.DateOffset(years=7)
    h = df[(~df["is_forecast"]) & (df["month"] >= start)]
    f = df[df["is_forecast"]]
    fig = go.Figure()
    fig.add_scatter(x=h["month"], y=h["bill_stock_bn"], name="Total stock (D2.2G)",
                    line=dict(color=C["bills"], width=2))
    deb = h.dropna(subset=["bill_stock_debtfin_bn"])
    fig.add_scatter(x=deb["month"], y=deb["bill_stock_debtfin_bn"], name="Debt-financing stock (FY-end)",
                    mode="markers", marker=dict(color=C["bills_debtfin"], size=7))
    b6 = bt_bl[(bt_bl["model"] == model) & (bt_bl["h"] == 6) & (bt_bl["target_end"] >= start)]
    fig.add_scatter(x=b6["target_end"], y=b6["fc"], name=f"Backtest, 6m ahead ({M.label(model)})",
                    mode="lines", line=dict(color=C["bills"], width=1, dash="dot"), visible="legendonly")
    last = h.iloc[[-1]]
    pre = pd.concat([last.assign(bill_stock_implied_bn=last["bill_stock_bn"]), f[~f["bills_assumed"]]])
    fig.add_scatter(x=pre["month"], y=pre["bill_stock_implied_bn"], name="To March: solved to plan net cash",
                    line=dict(color=C["bills"], width=2, dash="dash"))
    post = f[f["bills_assumed"]]
    if len(post):
        post = pd.concat([pre.iloc[[-1]], post])
        lab = "held flat" if e["bill_after_fy"] == "flat" else M.label(model)
        fig.add_scatter(x=post["month"], y=post["bill_stock_implied_bn"], name=f"After March: {lab} (assumption)",
                        line=dict(color=C["bills"], width=1.5, dash="dot"))
    pin = f[f["net_bills_pinned"]]
    if len(pin):
        fig.add_scatter(x=pin["month"], y=pin["bill_stock_implied_bn"], name="From weekly tenders",
                        mode="markers", marker=dict(color=C["bills"], size=8, symbol="square"))
    mar = pd.Timestamp(int(e["fy"][:4]) + 1, 3, 1)
    cm = e["cash_mgmt_bills_end_march_assumed_bn"]
    fig.add_scatter(x=[mar], y=[e["tbill_debtfin_stock_end_march_plan_bn"] + cm], mode="markers",
                    name=f"DMO plan end-March (debt-financing {g(e['tbill_debtfin_stock_end_march_plan_bn'])} "
                         f"+ cash-mgmt {g(cm)} assumed)",
                    marker=dict(color=C["plan"], size=12, symbol="star"))
    return _style(fig, title="T-bill stock, £bn: actual, path to the remit's end-March net cash, DMO plan",
                  ylab="£bn")


def fig_buffer(df: pd.DataFrame, s: dict) -> go.Figure:
    from sovereign_cash.horizon import clip_frame, programme_end
    e = s["forward"]["end_march"]
    # Drop post-programme buffer dump (gross gilts → 0, redemptions → buffer −£tr).
    d = clip_frame(df)
    d = d[d["month"] >= d["month"].max() - pd.DateOffset(years=6)]
    h, f = d[~d["is_forecast"]], d[d["is_forecast"]]
    fig = go.Figure()
    fig.add_scatter(x=h["month"], y=h["buffer_level_implied_bn"], name="ONS proxy, pinned at last FY-end",
                    line=dict(color=C["buffer"]))
    b = d.dropna(subset=["buffer_level_dmo_basis_bn"])
    path = pd.concat([b[["month", "buffer_level_dmo_basis_bn"]].rename(columns={"buffer_level_dmo_basis_bn": "y"}),
                      f[["month", "buffer_level_implied_bn"]].rename(columns={"buffer_level_implied_bn": "y"})])
    fig.add_scatter(x=path["month"], y=path["y"], name="DMO basis: year to date + forward",
                    line=dict(color=C["buffer"], dash="dash"))
    checks = {pd.Timestamp(r["month"]): r for r in s.get("buffer_anchor_checks", [])}
    a = d.dropna(subset=["buffer_level_bn"])
    txt = [f"proxy miss {g(checks[m]['miss_bn'], plus=True)}" if m in checks else "" for m in a["month"]]
    fig.add_scatter(x=a["month"], y=a["buffer_level_bn"], name="DMR outturn: net cash + cash-mgmt bills",
                    mode="markers", marker=dict(color=C["buffer"], size=9, symbol="diamond"),
                    text=txt, hovertemplate="%{y:.1f} %{text}<extra></extra>")
    m = f.dropna(subset=["dmo_net_cash_bn"])
    if len(m):
        fig.add_scatter(x=m["month"], y=m["buffer_level_implied_bn"], mode="markers",
                        name=f"End-March plan: net cash {g(e['net_cash_plan_end_march_bn'])} + cash-mgmt bills",
                        marker=dict(color=C["plan"], size=12, symbol="star"))
    o = h["month"].max()
    fig.add_annotation(x=o, y=e["buffer_dmo_basis_origin_bn"], ax=0, ay=-34, showarrow=True, arrowhead=0,
                       font=dict(size=9), text=f"basis gap {g(e['buffer_basis_gap_bn'], plus=True)}")
    end = programme_end(df)
    note = f" (through {end:%b-%Y})" if end is not None else ""
    return _style(fig, title="Exchequer buffer, £bn: DMO net cash + cash-management bills" + note, ylab="£bn")


def fig_gap(e: dict) -> go.Figure:
    p = e["next_revision_gap_parts_bn"]
    lab = {"cgncr": "CGNCR vs plan", "nsandi": "NS&I shortfall", "gilt_sales": "Gilt sales vs remit",
           "redemptions": "Redemptions vs remit", "other": "Other items", "published_rounding": "Published rounding"}
    keys = [k for k in lab if abs(p.get(k) or 0.0) >= 0.05]
    y = [p[k] for k in keys] + [0]
    fig = go.Figure(go.Waterfall(x=[lab[k] for k in keys] + ["Next revision"], y=y,
                                 measure=["relative"] * len(keys) + ["total"],
                                 text=[n(v, plus=True) for v in [p[k] for k in keys] + [e["next_revision_gap_bn"]]],
                                 textposition="outside",
                                 increasing=dict(marker_color=C["more"]), decreasing=dict(marker_color=C["less"]),
                                 totals=dict(marker_color=C["buffer"]), connector=dict(line=dict(color=GREY))))
    fig.add_annotation(x=1, y=1.0, xref="paper", yref="paper", xanchor="right", showarrow=False, font=dict(size=9),
                       text=f"± {g(e['gap_err_bn'])} out-of-sample error")
    return _style(fig, title=f"Extra financing at the next {e['fy']} remit revision, £bn (+ = more bills or gilts)",
                  ylab="£bn", height=360)


def fig_other(df: pd.DataFrame) -> go.Figure:
    h = df[(~df["is_forecast"]) & (df["month"] >= pd.Timestamp("2015-04-01"))].copy()
    h["fy"] = M.fy_of(pd.DatetimeIndex(h["month"]))
    cols = {"of_nsandi_bn": ("NS&I", C["other"]), "of_dep_by_others_bn": ("Deposits with CG (mainly DMADF)", C["deposits"]),
            "of_nram_nr_bn": ("NRAM, B&B, Network Rail", C["nram"]), "of_gilt_vs_ons_bn": ("Gilts: ONS − DMO", C["gilt_gap"]),
            "of_bill_vs_ons_bn": ("Bills: ONS − DMO", C["bill_gap"]), "of_unexplained_bn": ("Unexplained (ONS)", C["unexplained"])}
    grp = h.groupby("fy")
    tot = grp[list(cols)].sum()
    cnt = grp.size()
    x = [_fy_label(y) + (f" YTD ({cnt[y]}m)" if cnt[y] < 12 else "") for y in tot.index]
    fig = go.Figure()
    for c, (name, colr) in cols.items():
        fig.add_bar(x=x, y=tot[c], name=name, marker_color=colr)
    fig.update_layout(barmode="relative")
    return _style(fig, title="Other financing by financial year, £bn: what is outside gilts and bills", ylab="£bn")


def fig_mm(mm: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not mm.empty:
        m = mm[mm.index >= mm.index.max() - pd.DateOffset(years=6)]
        fig.add_bar(x=m.index, y=m["net_bills_bn"], name="Change in T-bill stock (£bn, lhs)", marker_color=C["bills"])
        fig.add_scatter(x=m.index, y=m["bill_3m_sonia_bp"], name="3m T-bill tender yield − SONIA (bp, rhs)",
                        yaxis="y2", line=dict(color=C["sonia"]))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="bp"))
    return _style(fig, title="T-bill supply and pricing: monthly change in stock vs 3m tender yield − SONIA",
                  ylab="£bn", rmargin=52)


# --------------------------------------------------------------------------------------------
# Tables


def _score_table(bt: pd.DataFrame, models: list[str], chosen: str, title: str) -> str:
    t = (bt[bt["window"].isin(["PRE", "COVID", "OOS"])].groupby(["window", "model", "h"])["abs_err"]
         .mean().unstack("h"))
    cnt = bt[bt["window"].isin(["PRE", "COVID", "OOS"])].groupby(["window", "h"]).size().unstack("h")
    hs = list(t.columns)
    out = [f"<h3 style='font:600 11px var(--mono)'>{html.escape(title)}</h3><table class='t'><tr><th>window · model</th>"
           + "".join(f"<th>{h}m</th>" for h in hs) + "</tr>"]
    for w in ("PRE", "OOS", "COVID"):
        if w not in t.index.get_level_values(0):
            continue
        sub = t.loc[w]
        best = {h: sub[h].idxmin() for h in hs}
        for mdl in models:
            if mdl not in sub.index:
                continue
            cells = "".join(f"<td class='{'win' if best[h] == mdl else ''}'>{n(sub.at[mdl, h], 2)}</td>" for h in hs)
            tag = "<span class='tag'>chosen on PRE</span>" if mdl == chosen else ""
            out.append(f"<tr><td title='{mdl}'>{w} · {html.escape(M.label(mdl))}{tag}</td>{cells}</tr>")
        out.append(f"<tr><td style='color:var(--muted)'>{w} · origins</td>"
                   + "".join(f"<td style='color:var(--muted)'>{int(cnt.at[w, h])}</td>" for h in hs) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _cal_table(cal: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{pd.Timestamp(r['date']):%d-%b-%y}</td><td style='text-align:left'>{html.escape(r['name'])}</td>"
        f"<td>{n(r['nominal_bn'])}</td><td>{n(r['gov_held_bn'])}</td><td>{n(r['apf_held_bn'])}</td>"
        f"<td>{n(r['cash_bn'])}</td><td>{n(r['private_cash_bn'])}</td></tr>" for r in cal)
    keys = ("nominal_bn", "gov_held_bn", "apf_held_bn", "cash_bn", "private_cash_bn")
    tot = {k: sum(r[k] for r in cal) for k in keys}
    rows += "<tr><td><b>total</b></td><td></td>" + "".join(f"<td><b>{n(tot[k])}</b></td>" for k in keys) + "</tr>"
    return ("<h3 style='font:600 11px var(--mono)'>Gilt redemptions, next 12 months, £bn</h3><table class='t'><tr>"
            "<th>date</th><th>gilt</th><th>nominal</th><th>CRND/DMO held</th><th>APF held</th>"
            "<th>Exchequer cash</th><th>to private</th></tr>" + rows + "</table>")


def _paired_table(rows: list[dict], model: str) -> str:
    if not rows:
        return ""
    body = "".join(f"<tr><td>{r['window']}</td><td>{int(r['h'])}m</td><td>{n(r[model], 2)}</td><td>{n(r['smly'], 2)}</td>"
                   f"<td>{n(r[M.OFFICIAL], 2)}</td><td>{int(r['n'])}</td></tr>" for r in rows)
    return ("<h3 style='font:600 11px var(--mono)'>CGNCR vs the remit plan published at each origin, MAE £bn "
            "(same origins in every column)</h3>"
            f"<table class='t'><tr><th>window</th><th>horizon</th><th>{html.escape(M.label(model))}</th>"
            f"<th>{M.label('smly')}</th><th>{M.label(M.OFFICIAL)}</th><th>origins</th></tr>{body}</table>")


# --------------------------------------------------------------------------------------------
# Page


GAP_LABELS = {"cgncr": "CGNCR", "nsandi": "NS&I", "gilt_sales": "gilt sales", "redemptions": "redemptions",
              "other": "other items", "published_rounding": "rounding"}


def gap_parts_text(parts: dict, min_abs: float = 0.05) -> str:
    """Non-trivial components of the revision gap, largest first."""
    items = sorted(((k, v) for k, v in parts.items() if v is not None and abs(v) >= min_abs),
                   key=lambda kv: -abs(kv[1]))
    return ", ".join(f"{GAP_LABELS.get(k, k)} {g(v, plus=True)}" for k, v in items) or "no component above £0.05bn"


def gap_err_text(e: dict) -> str:
    return (f"± {g(e['gap_err_bn'])} (CGNCR {g(e['gap_err_cgncr_bn'])}, NS&I {g(e['gap_err_nsandi_bn'])}; "
            f"out-of-sample MAE over {e['months_left_in_fy']}m)")


def _lead(s: dict) -> list[str]:
    fwd = s["forward"]
    e = fwd["end_march"]
    gp = e["next_revision_gap_parts_bn"]
    return [
        f"Remit: {g(fwd['remit_remaining_bn'])} left over {fwd['remit_window']} (~{g(fwd['remit_per_month_bn'])} a month); "
        f"{g(e['gilt_remit_sold_bn'])} of {g(e['gilt_remit_total_bn'])} sold to {e['gilt_remit_asof']}.",
        f"Next {e['fy']} revision: {g(e['next_revision_gap_bn'], plus=True)} {gap_err_text(e)} to hold end-March "
        f"net cash at {g(e['net_cash_plan_end_march_bn'])}: {gap_parts_text(gp)}. Lands on bills unless gilts are "
        f"added ({g(e.get('gilt_remit_unallocated_bn'))} of the remit unallocated).",
        f"Year to {e['ytd_through'][:7]}: CGNCR {g(e['cgncr_ex_ytd_bn'])} vs {g(e['cgncr_ex_plan_profile_ytd_bn'])} "
        f"on the plan profile; NS&I {g(e['nsandi_ytd_bn'])} vs {g(e['nsandi_remit_profile_ytd_bn'])}.",
    ]


def obr_table() -> str:
    """OBR EFO lines, one block at a time. Empty if the parse is not cached."""
    from sovereign_cash.uk import obr_tables as O

    d = O.load()
    if d.empty or "block" not in d.columns:
        return ""
    blocks = (("financing", "Financing"), ("receipts", "Receipts"), ("expenditure", "Spending"),
              ("welfare", "Welfare"), ("determinants", "Determinants"))
    years = sorted(int(y) for y in d["year"].unique())
    head = "".join(f"<th>{y}-{str(y + 1)[2:]}</th>" for y in years)
    opts, tables = [], []
    for i, (key, title) in enumerate(blocks):
        sub = d[d["block"] == key]
        if sub.empty:
            continue
        opts.append(f"<option value='{key}'>{html.escape(title)}</option>")
        order = sub.groupby("var", sort=False)["ord"].min().sort_values()
        wide = sub.pivot_table(index="var", columns="year", values="value", aggfunc="first")
        labels = sub.groupby("var")["label"].first()
        units = sub.groupby("var")["unit"].first()
        body = []
        for var in order.index:
            cells = []
            pct = str(units.get(var, "")).startswith("%")
            for y in years:
                v = wide.at[var, y] if y in wide.columns else float("nan")
                cells.append(f"<td>{n(v, 1) if pct else g(v)}</td>")
            weight = " style='font-weight:700'" if var == "cgncr_ex_nr_bn" else ""
            body.append(f"<tr><td{weight}>{html.escape(str(labels.get(var, var)))}</td>{''.join(cells)}</tr>")
        hidden = "" if i == 0 else " hidden"
        tables.append(f"<table class='t' data-obr='{key}'{hidden}><tr><th></th>{head}</tr>{''.join(body)}</table>")
    vint = html.escape(O.vintage())
    return (f"<div class='sub'>{vint}</div>"
            f"<select onchange=\"document.querySelectorAll('[data-obr]').forEach(t=>t.hidden=t.dataset.obr!==this.value)\">"
            f"{''.join(opts)}</select>{''.join(tables)}"
            f"<div class='sub'>Published forecast, £bn. Determinants are per cent. "
            f"CGNCR ex Network Rail is the remit total.</div>")


def scenario_table() -> str:
    """Unit shocks on the published ready reckoner. Empty if the workbook is absent."""
    from sovereign_cash.uk import ready_reckoner as R

    if not R.XLSX.exists():
        return ""
    shocks = (
        ("earnings", "Earnings +1%", "Earnings level"),
        ("gilts", "Gilt rates +1pp", "Long-term interest rates"),
        ("bank", "Bank Rate +1pp", "Bank Rate"),
        ("rpi", "RPI +1pp", "RPI inflation"),
        ("cpi", "September CPI +1pp", "Q3 CPI for benefits (zero floor)"),
    )
    years = R.YEARS[2:]
    head = "".join(f"<th>{html.escape(y)}</th>" for y in years)
    opts, tables = [], []
    for i, (key, title, name) in enumerate(shocks):
        fx = R.run(R.unit(name, years))
        by_line = fx.groupby(["line", "effect_fy"])["gbp_bn"].sum().unstack("effect_fy")
        keep = [ln for ln in by_line.index if float(by_line.loc[ln].abs().max()) >= 0.05]
        body = []
        for ln in keep:
            cells = "".join(f"<td>{g(by_line.loc[ln].get(y, float('nan')), plus=True)}</td>" for y in years)
            body.append(f"<tr><td>{html.escape(ln)}</td>{cells}</tr>")
        borrowing = R.psnb(fx)
        cells = "".join(f"<td>{g(borrowing.get(y, float('nan')), plus=True)}</td>" for y in years)
        body.append(f"<tr><td style='font-weight:700'>Public sector net borrowing</td>{cells}</tr>")
        opts.append(f"<option value='{key}'>{html.escape(title)}</option>")
        hidden = "" if i == 0 else " hidden"
        tables.append(f"<table class='t' data-rr='{key}'{hidden}><tr><th></th>{head}</tr>{''.join(body)}</table>")
    return (f"<div class='sub'>{html.escape(R.VINTAGE)}</div>"
            f"<select onchange=\"document.querySelectorAll('[data-rr]').forEach(t=>t.hidden=t.dataset.rr!==this.value)\">"
            f"{''.join(opts)}</select>{''.join(tables)}"
            f"<div class='sub'>Change in £bn from a +1 shock in every year from 2025-26. "
            f"Linear and additive, so a 2pp shock is twice this and shocks add. "
            f"The cash path stays the published forecast.</div>")


def _fy_of(month: pd.Timestamp) -> int:
    return int(month.year) if month.month >= 4 else int(month.year) - 1


def _axis_x(month: pd.Timestamp, cal: str) -> str:
    if cal == "fy" and month.month < 4:
        return f"2001-{month.month:02d}-01"
    return f"2000-{month.month:02d}-01"


def _gdp_by_month(months: pd.DatetimeIndex) -> pd.Series:
    path = REPO / "sovereign_cash" / "derived" / "debt_country_annual.parquet"
    if not path.exists():
        return pd.Series(np.nan, index=months)
    ann = pd.read_parquet(path)
    uk = ann.loc[ann["cc"].astype(str).eq("uk"), ["year", "gdp_bn"]].dropna()
    level = uk.set_index("year")["gdp_bn"].astype(float)
    out = pd.Series([float(level.get(int(m.year), np.nan)) for m in months], index=months)
    known = out.dropna()
    if not known.empty:
        out = out.ffill().bfill()
    return out


def _year_book(frame: pd.DataFrame, col: str, level: str | None, cal: str, gdp: pd.Series) -> dict:
    """One line per year. Stock is the within-year sum, except a level series which is the stock itself."""
    book = {"bn": {"flow": {}, "stock": {}}, "pct": {"flow": {}, "stock": {}}}
    years = frame["month"].map(lambda m: _fy_of(m) if cal == "fy" else int(m.year))
    work = frame.assign(_y=years)
    for year, part in work.groupby("_y"):
        part = part.sort_values("month")
        xs = [_axis_x(pd.Timestamp(m), cal) for m in part["month"]]

        def _clean(s: pd.Series) -> list:
            return [None if not np.isfinite(v) else float(v) for v in s.astype(float)]

        flow = _clean(part[col])
        stock = _clean(part[level] if level else part[col].astype(float).cumsum())
        g = [float(gdp.get(m, np.nan)) for m in part["month"]]

        def _share(vals: list) -> list:
            return [None if v is None or not np.isfinite(d) or d == 0 else 100.0 * v / d
                    for v, d in zip(vals, g)]

        book["bn"]["flow"][str(int(year))] = [xs, flow]
        book["bn"]["stock"][str(int(year))] = [xs, stock]
        book["pct"]["flow"][str(int(year))] = [xs, _share(flow)]
        book["pct"]["stock"][str(int(year))] = [xs, _share(stock)]
    return book


def season_payload(df: pd.DataFrame) -> dict:
    """Monthly cash lines overlaid by fiscal year and by calendar year, in £bn and as % of GDP."""
    from sovereign_cash.horizon import clip_frame
    d = clip_frame(df).sort_values("month").copy()
    months = pd.DatetimeIndex(d["month"])
    gdp = _gdp_by_month(months)
    gdp.index = months
    level = d["buffer_level_bn"].where(~d["is_forecast"], d["buffer_level_implied_bn"])
    d = d.assign(_buffer=level)
    specs = (
        ("cgncr", "CGNCR", "cash_requirement_bn", None),
        ("gilts", "Net gilt cash", "net_bonds_bn", None),
        ("bills", "Net bills", "net_bills_bn", None),
        ("other", "Other financing", "other_fin_bn", None),
        ("buffer", "Cash balance", "buffer_change_bn", "_buffer"),
    )
    views = {key: {cal: _year_book(d, col, lvl, cal, gdp) for cal in ("fy", "cy")}
             for key, _, col, lvl in specs}
    last = pd.Timestamp(d.loc[~d["is_forecast"], "month"].max())
    years = sorted({_fy_of(m) for m in months} | {int(m.year) for m in months})
    latest = max(years)
    return {
        "views": views,
        "order": [k for k, *_ in specs],
        "labels": {k: lab for k, lab, *_ in specs},
        "colors": {str(y): MATLAB[(latest - y) % len(MATLAB)] for y in years},
        "histEnd": {"fy": _axis_x(last, "fy"), "cy": _axis_x(last, "cy")},
        "current": {"fy": _fy_of(last), "cy": int(last.year)},
        "ticks": {
            "fy": [["2000-04-01", "2000-07-01", "2000-10-01", "2001-01-01"], ["Apr", "Jul", "Oct", "Jan"]],
            "cy": [["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"], ["Jan", "Apr", "Jul", "Oct"]],
        },
        "gdp": "calendar-year nominal GDP, held flat through the year",
    }


def season_html(df: pd.DataFrame) -> str:
    import json
    payload = json.dumps(season_payload(df), separators=(",", ":"))
    cards = "".join(
        f"<div class='card{' span-2' if k == 'cgncr' else ''}'>"
        f"<div id='uk-{k}' class='season{' tall' if k == 'cgncr' else ''}'></div></div>"
        for k in ("cgncr", "gilts", "bills", "other", "buffer")
    )
    return f"""
    <div class="toggles" id="uk-toggle">
      <button type="button" data-kind="stock" class="on">Stock</button>
      <button type="button" data-kind="flow">Flow</button>
      <button type="button" data-cal="fy" class="on">FY</button>
      <button type="button" data-cal="cy">CY</button>
      <button type="button" data-hist="3">3y</button>
      <button type="button" data-hist="5" class="on">5y</button>
      <button type="button" data-hist="10">10y</button>
      <button type="button" data-hist="0">All</button>
      <label class="chk"><input type="checkbox" id="uk-gdp"> % of GDP</label>
    </div>
    <p class="sub" id="uk-note"></p>
    <div class="grid-2">{cards}</div>
    <script>
    const UKS = {payload};
    const ukState = {{kind: "stock", cal: "fy", gdp: false, hist: 5}};
    function ukNote() {{
      const flow = ukState.kind === "flow";
      const fy = ukState.cal === "fy";
      const base = flow
        ? "Flow: the month's print, one line per year."
        : (fy ? "Stock: sum since April. The cash balance is the level, not a sum."
              : "Stock: sum since January. The cash balance is the level, not a sum.");
      const span = ukState.hist ? " Last " + ukState.hist + " years, plus the forecast." : " Full history, plus the forecast.";
      const unit = ukState.gdp ? " Share of " + UKS.gdp + "." : " £bn.";
      document.getElementById("uk-note").textContent = base + span
        + " The latest year is blue. The vertical rule is the last ONS print; after it the line is broken."
        + unit + " Forecast months follow the OBR / remit path.";
    }}
    function ukLayout(title, ylab, cal, cut) {{
      return {{
        title: {{text: title, font: {{size: 13}}, x: 0, xanchor: "left"}},
        margin: {{l: 58, r: 16, t: 36, b: 64}},
        height: title === "CGNCR" ? 500 : 420,
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        hovermode: "x unified",
        legend: {{orientation: "h", y: -0.22, x: 0, font: {{size: 10}}}},
        xaxis: {{tickmode: "array", tickvals: UKS.ticks[cal][0], ticktext: UKS.ticks[cal][1],
                 showgrid: false, title: {{text: cal === "fy" ? "Fiscal year" : "Calendar year"}}}},
        yaxis: {{title: {{text: ylab}}, zeroline: true, gridcolor: "#e6e6e6"}},
        shapes: [{{type: "line", xref: "x", yref: "paper", x0: cut, x1: cut, y0: 0, y1: 1,
                   line: {{color: "#000", width: 1.25}}}}],
        annotations: [{{x: cut, y: 1, yref: "paper", text: "forecast", showarrow: false,
                        xanchor: "left", font: {{size: 10, color: "#000"}}}}]
      }};
    }}
    function ukSplit(xs, ys, cut, current) {{
      if (!current) return [{{x: xs, y: ys, dash: "solid"}}];
      const a = {{x: [], y: [], dash: "solid"}}, b = {{x: [], y: [], dash: "dash"}};
      xs.forEach((x, i) => {{ (x <= cut ? a : b).x.push(x); (x <= cut ? a : b).y.push(ys[i]); }});
      if (a.x.length && b.x.length) {{ b.x.unshift(a.x[a.x.length - 1]); b.y.unshift(a.y[a.y.length - 1]); }}
      return [a, b].filter(t => t.x.length);
    }}
    function ukDraw() {{
      ukNote();
      const cal = ukState.cal, kind = ukState.kind, cut = UKS.histEnd[cal], cur = String(UKS.current[cal]);
      const ylab = ukState.gdp
        ? (kind === "flow" ? "Month, % of GDP" : (cal === "fy" ? "Since April, % of GDP" : "Since January, % of GDP"))
        : (kind === "flow" ? "Month, £bn" : (cal === "fy" ? "Since April, £bn" : "Since January, £bn"));
      UKS.order.forEach(key => {{
        const el = document.getElementById("uk-" + key);
        const block = (ukState.gdp ? UKS.views[key][cal].pct : UKS.views[key][cal].bn)[kind];
        const years = Object.keys(block).sort();
        const traces = [];
        years.forEach(y => {{
          const n = ukState.hist;
          if (n && Number(y) < Number(cur) - n + 1) return;
          const color = UKS.colors[y] || "#0072BD";
          const future = Number(y) > Number(cur);
          const parts = future ? [{{x: block[y][0], y: block[y][1], dash: "dash"}}]
                               : ukSplit(block[y][0], block[y][1], cut, y === cur);
          const name = (cal === "fy" ? y + "-" + String(Number(y) + 1).slice(2) : y);
          parts.forEach((p, i) => {{
            traces.push({{
              x: p.x, y: p.y, mode: "lines", name: i ? name + " forecast" : name,
              legendgroup: y, showlegend: i === 0,
              line: {{color: color, width: y === cur && p.dash === "solid" ? 2.4 : 1.5, dash: p.dash === "dash" ? "dash" : "solid"}},
              hovertemplate: name + " · %{{y:.1f}}<extra></extra>"
            }});
          }});
        }});
        Plotly.react(el, traces, ukLayout(UKS.labels[key], ylab, cal, cut), {{displayModeBar: false, responsive: true}});
      }});
    }}
    document.getElementById("uk-toggle").addEventListener("click", ev => {{
      const b = ev.target.closest("button");
      if (!b) return;
      if (b.dataset.kind) ukState.kind = b.dataset.kind;
      if (b.dataset.cal) ukState.cal = b.dataset.cal;
      if (b.dataset.hist !== undefined) ukState.hist = Number(b.dataset.hist);
      document.querySelectorAll("#uk-toggle button").forEach(x => {{
        x.classList.toggle("on", (x.dataset.kind && x.dataset.kind === ukState.kind)
          || (x.dataset.cal && x.dataset.cal === ukState.cal)
          || (x.dataset.hist !== undefined && Number(x.dataset.hist) === ukState.hist));
      }});
      ukDraw();
    }});
    document.getElementById("uk-gdp").addEventListener("change", ev => {{ ukState.gdp = ev.target.checked; ukDraw(); }});
    ukDraw();
    </script>
    """


def build(df: pd.DataFrame, summary: dict, bt_cg: pd.DataFrame, bt_bl: pd.DataFrame, mm: pd.DataFrame) -> Path:
    fwd = summary["forward"]
    e = fwd["end_march"]
    gp = e["next_revision_gap_parts_bn"]
    cl = summary.get("closure", {})
    figs = {"gap": fig_gap(e), "gilts": fig_gilts(df, summary), "cgncr": fig_cgncr(df, e),
            "cgncr_cum": fig_cgncr_cum(df, e), "buffer": fig_buffer(df, summary),
            "bills": fig_bills(df, bt_bl, fwd["bill_model"], e), "identity": fig_identity(df),
            "other": fig_other(df), "mm": fig_mm(mm)}
    p = {k: _div(f) for k, f in figs.items()}
    after = "held flat" if e["bill_after_fy"] == "flat" else M.label(fwd["bill_model"])
    kpis = "".join([
        _kpi(f"CGNCR to {e['ytd_through'][:7]}", g(e["cgncr_ex_ytd_bn"]),
             f"plan profile {g(e['cgncr_ex_plan_profile_ytd_bn'])} · FY {g(e['cgncr_ex_plan_bn'])}"),
        _kpi(f"Next {e['fy']} revision", g(e["next_revision_gap_bn"], plus=True),
             f"± {g(e['gap_err_bn'])} · NS&I {g(e['next_revision_gap_parts_bn']['nsandi'], plus=True)}"),
        _kpi(f"Private net gilts to March", g(fwd["private_net_bonds_fy_bn"]),
             f"sold {g(e['gilt_remit_sold_bn'])} of {g(e['gilt_remit_total_bn'])}"),
        _kpi("T-bill stock", f"{g(fwd['bill_stock_last_obs_bn'])} → {g(e['bill_stock_implied_end_march_bn'])}",
             f"end-March, solved to the remit · after March {after}"),
    ])
    lead = "".join(f"<li>{html.escape(x)}</li>" for x in _lead(summary))
    cg_models = list(M.CGNCR_MODELS + M.CGNCR_BENCH)
    bl_models = list(M.BILL_MODELS + M.BILL_BENCH)
    tables = (_score_table(bt_cg, cg_models, fwd["cgncr_model"], "Cumulative CGNCR, MAE £bn")
              + _paired_table(summary.get("official_benchmark", []), fwd["cgncr_model"])
              + _score_table(bt_bl, bl_models, fwd["bill_model"], "T-bill stock level, MAE £bn"))
    skill = "".join(f"<li>{html.escape(x)}</li>" for x in summary.get("skill_notes", []))
    src_rows = "".join(
        f"<tr><td>{html.escape(x['name'])}</td><td>{'yes' if x['verified'] else 'NO'}</td>"
        f"<td style='text-align:left'>{html.escape(x.get('check', ''))}</td></tr>" for x in summary["sources"])
    notes = "".join(f"<li>{html.escape(x)}</li>" for x in summary["notes"])
    try:
        cat_panel, cat_specs = uk_category_panel()
        cat_html = category_tape_html(
            cat_panel, cc="uk", prefix="uk", specs=cat_specs, currency="£",
            note="ONS/EFO category leaves under the shipped CGNCR bottom-up (calendar × ready-reckoner).")
    except Exception:  # noqa: BLE001
        cat_html = ""
    plotly_js = f"https://cdn.plot.ly/plotly-{plotly.offline.get_plotlyjs_version()}.min.js"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>UK central-government cash</title>
<script src="{plotly_js}" charset="utf-8"></script>
<style>{PAGE_CSS}{TABLE_CSS}{SEASON_CSS}</style></head><body>
<header><h1>UK central-government cash</h1>
<div class="sub">Through {summary['asof'][:7]} · ONS public sector finances, DMO remit, OBR March 2026 · built {dt.datetime.now():%Y-%m-%d %H:%M}</div>
</header>
<div class="wrap">
<div class="kpi-row">{kpis}</div>
<h2 class="section">Through the year</h2>
{season_html(df)}
{cat_html}
<h2 class="section">OBR</h2>
<div class="grid-2"><div class="card">{obr_table()}</div><div class="card">{scenario_table()}</div></div>
<h2 class="section">Financing</h2>
<div class="grid-2">
<div class="card span-2">{p['gilts']}<div class="sub">Translucent = forecast; hatched = after March, no remit published.</div></div>
<div class="card">{p['cgncr']}</div>
<div class="card">{p['gap']}</div>
<div class="card">{_cal_table(fwd['redemption_calendar'])}
<div class="sub">CRND/DMO holdings move no cash. APF-held redemptions are paid to the Bank.</div></div>
<div class="card">{p['bills']}</div>
<div class="card">{p['buffer']}</div>
<div class="card">{p['identity']}</div>
<div class="card">{p['other']}</div>
<div class="card span-2">{p['mm']}</div>
</div>
<details class="fold"><summary>Backtest</summary><div class="card">{tables}
<ul class="sub">{skill}</ul>
<div class="sub">{html.escape(summary['score_note'])}</div></div></details>
<details class="fold"><summary>Sources and method</summary><div class="card">
<table class="t"><tr><th>source</th><th>verified</th><th>check</th></tr>{src_rows}</table>
<ul class="sub">{notes}</ul></div></details>
</div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    return OUT
