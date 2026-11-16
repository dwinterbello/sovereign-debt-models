"""Sweden cash dashboard → ``sovereign_cash/se_cash_dashboard.html``.

One sign convention throughout: net borrowing requirement (NBR) and every financing line, + = borrowing.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
US_FISCAL = ROOT.parent / "us" / "fiscal"
if str(US_FISCAL) not in sys.path:
    sys.path.insert(0, str(US_FISCAL))
from build_dashboard import _T, BLUE, GREEN, GREY, ORANGE, PAGE_CSS, PURPLE, RED, _html, _kpi, _style  # noqa: E402
from build_liquidity_dashboard import TABLE_CSS, _table  # noqa: E402

from sovereign_cash.cash_season import (  # noqa: E402
    SEASON_CSS, bottomup_card, category_tape_html, default_specs,
    se_category_panel, season_html,
)
from sovereign_cash.contract import fmt_bn, fmt_num  # noqa: E402
from sovereign_cash.se import model  # noqa: E402

DERIVED = ROOT / "derived"
OUT = ROOT / "se_cash_dashboard.html"
HIST_START = pd.Timestamp("2016-01-01")
SYM = "SEK"
FIN_MONTHS = 24  # outcome months shown on the financing chart before the forward
ALLOC_LABEL = {"uniform": "equal shift per month", "err_share": "where months historically miss most",
               "abs_share": "in proportion to the model month", "hist_share": "historical share of the year"}


def _d(s: str) -> str:
    return f"{pd.Timestamp(s):%-d %b %Y}"


def fig_nbr(df: pd.DataFrame, rg: pd.DataFrame) -> go.Figure:
    h = df[(~df["is_forecast"]) & (df["month"] >= HIST_START)]
    f = df[df["is_forecast"]]
    r = rg[rg["month"] >= HIST_START]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h["month"], y=h["cash_requirement_bn"], name="Outcome", marker_color=GREY))
    fig.add_trace(go.Bar(x=f["month"], y=f["cash_requirement_bn"], name="Forward (Riksgälden annual, model month shape)",
                         marker_color=BLUE))
    fig.add_trace(go.Scatter(x=f["month"], y=-f["budget_balance_model_bn"], name="Seasonal model alone", mode="markers",
                             marker=dict(color=ORANGE, size=7, symbol="diamond")))
    fig.add_trace(go.Scatter(x=r["month"], y=-r["rg_forecast_bn"], name="Riksgälden monthly forecast (as published)",
                             mode="markers", marker=dict(color=RED, size=5)))
    fig = _style(fig, title="Net borrowing requirement, monthly (bn SEK, + = borrowing)", ylab="bn SEK", height=440,
                 rangeslider=True)
    fig.update_layout(barmode="overlay", bargap=0.1)
    return fig


def fig_rolling(df: pd.DataFrame, annual: dict, vintage: str) -> go.Figure:
    s = df.set_index("month")["cash_requirement_bn"]
    roll = s.rolling(12).sum()
    hist = ~df.set_index("month")["is_forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll[hist].index, y=roll[hist], name="12-month sum, outcome",
                             line=dict(color=_T["text"], width=1.6)))
    fig.add_trace(go.Scatter(x=roll[~hist].index, y=roll[~hist], name="12-month sum, forward",
                             line=dict(color=BLUE, width=1.6, dash="dot")))
    xs = [pd.Timestamp(int(yr), 12, 1) for yr in annual]
    fig.add_trace(go.Scatter(x=xs, y=[-v for v in annual.values()], mode="markers+text", name=f"Riksgälden calendar year ({vintage})",
                             text=[f"{yr}: {fmt_num(-v, 0)}" for yr, v in annual.items()], textposition="middle right",
                             marker=dict(color=RED, size=10, symbol="x-thin-open", line=dict(width=2))))
    fig = _style(fig, title="Net borrowing requirement, rolling 12 months (bn SEK); the forward at December is "
                            "Riksgälden's year less its YTD deviation", ylab="bn SEK", height=380)
    x1 = max([df["month"].max()] + xs) + pd.DateOffset(months=4)
    fig.update_xaxes(range=[pd.Timestamp("2008-01-01"), x1])
    return fig


def fig_financing(df: pd.DataFrame) -> go.Figure:
    f0 = df.loc[df["is_forecast"], "month"].min()
    d = df[df["month"] >= f0 - pd.DateOffset(months=FIN_MONTHS)]
    fig = go.Figure()
    for col, name, c, sgn in (("net_bonds_nominal_bn", "Nominal SGBs (incl. green)", BLUE, 1),
                              ("net_bonds_linker_bn", "Inflation-linked", "#17becf", 1),
                              ("net_bonds_fx_bn", "FX bonds", "#8c564b", 1),
                              ("net_bills_bn", "T-bills", ORANGE, 1),
                              ("other_fin_bn", "Other", PURPLE, 1),
                              ("buffer_change_bn", "Liquidity management (Δ borrowing)", GREEN, -1)):
        fig.add_trace(go.Bar(x=d["month"], y=sgn * d[col], name=name, marker_color=c))
    fig.add_trace(go.Scatter(x=d["month"], y=d["cash_requirement_bn"], name="Net borrowing requirement",
                             mode="lines+markers", line=dict(color=_T["text"], width=1.4), marker=dict(size=4)))
    fig.add_vline(x=f0 - pd.Timedelta(days=15), line_width=0.8, line_dash="dot", line_color=GREY)
    fig = _style(fig, title="Net borrowing by instrument (bn SEK, + = borrowing); bars sum to the line; forward right "
                            "of the dotted line", ylab="bn SEK", height=440)
    fig.update_layout(barmode="relative", bargap=0.15)
    return fig


def fig_stocks(df: pd.DataFrame, summ: dict) -> go.Figure:
    d = df[df["month"] >= pd.Timestamp("2022-01-01")]
    h, f = d[~d["is_forecast"]], d[d["is_forecast"]]
    fwd = summ["forward"]
    fig = go.Figure()
    plan = {"bill_stock_implied_bn": fwd["bill_stock"], "liq_mgmt_stock_implied_bn": fwd["liq_mgmt_stock"]}
    for col, name, c in (("bill_stock_implied_bn", "T-bills", ORANGE),
                         ("liq_mgmt_stock_implied_bn", "Liquidity management (borrowing +)", GREEN)):
        for k, v in plan[col].items():
            if k.startswith("riksgalden_plan_end_"):
                yr = int(k.split("_")[3])
                fig.add_trace(go.Scatter(x=[pd.Timestamp(yr, 12, 1)], y=[v], mode="markers+text",
                                         text=[f"plan {fmt_num(v, 0)}"], textposition="middle right", showlegend=False,
                                         marker=dict(color=c, size=14, symbol="diamond-open", line=dict(width=1.5))))
        fig.add_trace(go.Scatter(x=h["month"], y=h[col], name=f"{name}, outcome", line=dict(color=c, width=1.6)))
        fig.add_trace(go.Scatter(x=f["month"], y=f[col], name=f"{name}, forward", line=dict(color=c, width=1.6, dash="dot")))
    for key, c in (("bill_stock_check", ORANGE), ("liq_mgmt_stock_check", GREEN)):
        for k, v in summ[key].items():
            fig.add_trace(go.Scatter(x=[pd.Timestamp(int(k[-4:]), 12, 1)], y=[v["riksgalden_bn"]], mode="markers",
                                     marker=dict(color=c, size=9, symbol="circle-open"), showlegend=False,
                                     hovertext="Riksgälden year-end outcome"))
    bs = fwd["bill_stock"]
    fig.add_trace(go.Scatter(x=[pd.Timestamp(bs["snapshot_date"])], y=[bs["snapshot_bill_stock_bn"]], mode="markers",
                             name=f"T-bills outstanding {_d(bs['snapshot_date'])}",
                             marker=dict(color=ORANGE, size=9, symbol="star")))
    fig = _style(fig, title="T-bill and liquidity-management stocks, month end (bn SEK); circles = Riksgälden year-end "
                            "outcome, diamonds = plan", ylab="bn SEK", height=420)
    fig.update_xaxes(range=[pd.Timestamp("2022-01-01"), df["month"].max() + pd.DateOffset(months=6)])
    return fig


def fig_errors(bt: pd.DataFrame, rg: pd.DataFrame, chosen: str, df: pd.DataFrame) -> go.Figure:
    act = df.set_index("month")["budget_balance_bn"]
    b = bt[(bt["h"] == 1) & bt["model"].isin([chosen, "smly"])].pivot_table(index="target", columns="model", values="ae")
    r = (rg.set_index("month")["rg_forecast_bn"] - act).abs().dropna()
    fig = go.Figure()
    for col, name, c in ((chosen, "Seasonal model, 1 month ahead", BLUE), ("smly", "Same month last year", GREY)):
        fig.add_trace(go.Scatter(x=b.index, y=b[col].rolling(12, min_periods=6).mean(), name=name, line=dict(color=c, width=1.4)))
    fig.add_trace(go.Scatter(x=r.index, y=r.rolling(12, min_periods=6).mean(), name="Riksgälden published forecast",
                             line=dict(color=RED, width=1.4)))
    fig.add_vrect(x0=str(model.PRE[0].date()), x1=str(model.PRE[1].date()), fillcolor=GREY, opacity=0.08, line_width=0,
                  annotation_text="PRE", annotation_position="top left")
    fig.add_vrect(x0=str(model.OOS_START.date()), x1=str(act.index.max().date()), fillcolor=BLUE, opacity=0.05,
                  line_width=0, annotation_text="OOS", annotation_position="top left")
    return _style(fig, title="Rolling 12-month mean absolute error of the monthly NBR (bn SEK)", ylab="bn SEK", height=360)


def skill_rows(score: list[dict]) -> list[dict]:
    rows = []
    for win in ("PRE", "OOS"):
        rs = [r for r in score if r["window"] == win]
        if not rs:
            continue
        for r in rs:
            rows.append({"window": win, "forecaster": "Seasonal model", "horizon": f"{r['h']} month{'s' if r['h'] != 1 else ''}", "n": r["n"],
                         "MAE": r["model"]})
        rows.append({"window": win, "forecaster": "Same month last year", "horizon": "any", "n": rs[0]["n"],
                     "MAE": rs[0]["benchmark"]})
        if rs[0].get("riksgalden") is not None:
            rows.append({"window": win, "forecaster": "Riksgälden (latest report)", "horizon": "1–6 months",
                         "n": rs[0]["n_rg"], "MAE": rs[0]["riksgalden"]})
    return rows


def build() -> Path:
    import html as html_esc

    df = pd.read_parquet(DERIVED / "se_cash_monthly.parquet")
    summ = json.loads((DERIVED / "se_cash_summary.json").read_text())
    bt = pd.read_parquet(DERIVED / "se_backtest.parquet")
    rg = pd.read_parquet(DERIVED / "se_riksgalden_forecasts.parquet")
    fwd = summ["forward"]
    try:
        cat_panel, cat_specs = se_category_panel()
        cat_html = category_tape_html(
            cat_panel, cc="se", prefix="se", specs=cat_specs, currency="SEK",
            note="SCB/CH1089 NBR categories (explanatory). Live cash stays RG×seasonal.")
    except Exception:  # noqa: BLE001
        cat_html = ""
    chosen = summ["score"][0]["model_name"]
    yr = int(fwd["plan_year"])
    annual = {k.split("_")[2]: v for k, v in fwd.items() if k.startswith("riksgalden_annual_") and v is not None}
    roy, n12 = fwd[f"rest_of_{yr}"], fwd["next_12m"]
    bs, lm = fwd["bill_stock"], fwd["liq_mgmt_stock"]
    vintage = fwd["riksgalden_vintage"].split(" (")[0]
    ytd_m = f"{pd.Timestamp(fwd['ytd_through'] + '-01'):%b %Y}"
    lm_m = f"{pd.Timestamp(lm['last_outcome_month'] + '-01'):%b %Y}"
    roy_m = f"{pd.Timestamp(roy['window'][:7] + '-01'):%b}–{pd.Timestamp(roy['window'][-7:] + '-01'):%b}"
    timing = fwd.get("timing_items") or []
    tim_note = (" · ".join(f"{i['item']} {fmt_bn(i['amount_bn'], SYM, 0)} delayed, expected {i['expected_year']}"
                           for i in timing))
    gdp = fwd.get("gdp_bn")
    sup12 = n12["net_bonds_bn"] + n12["net_bills_bn"]
    nxt = fwd.get("next_outcome")
    kp = "".join([
        _kpi("Borrowing vs Riksgälden path, YTD", fmt_bn(fwd["borrowing_vs_riksgalden_path_ex_timing_bn"], SYM, plus=True),
             f"excluding timing items · as reported {fmt_bn(fwd['borrowing_vs_riksgalden_path_ytd_bn'], SYM, plus=True)}"
             + (f" · {tim_note}" if tim_note else "") + f" · release {_d(fwd['riksgalden_ytd_deviation_asof'])}"
             if fwd.get("riksgalden_ytd_deviation_asof") else "no deviation published"),
        _kpi(f"NBR {yr}: Riksgälden-consistent / plan",
             f"{fmt_bn(fwd['plan_consistent_full_year_bn'], SYM)} / {fmt_num(fwd['plan_cash_requirement_bn'], 0)}bn",
             f"plan = {vintage} · seasonal model alone {fmt_bn(fwd['pure_model_full_year_bn'], SYM)} · "
             f"YTD {fmt_bn(fwd['cash_requirement_ytd_bn'], SYM)} to {ytd_m}"),
        _kpi(f"NBR {roy_m} {yr} implied", fmt_bn(fwd["rest_of_year_implied_bn"], SYM),
             f"same months {yr - 1}: {fmt_bn(fwd['rest_of_year_same_months_last_year_bn'], SYM)}"),
        _kpi(f"Nominal SGB net, rest of {yr}", fmt_bn(roy["net_bonds_nominal_bn"], SYM, plus=True),
             f"gross {fmt_num(roy['bond_gross_nominal_bn'], 1)} − redemptions {fmt_num(roy['bond_redemption_nominal_bn'], 1)}"
             f" · next 12m {fmt_bn(n12['net_bonds_nominal_bn'], SYM, plus=True)}"),
        _kpi("Net bonds + bills, next 12m", fmt_bn(sup12, SYM, plus=True),
             (f"{fmt_num(100 * sup12 / gdp, 1)}% of GDP · " if gdp else "")
             + f"bonds {fmt_num(n12['net_bonds_bn'], 1)}, bills {fmt_num(n12['net_bills_bn'], 1)}"),
        _kpi("T-bill stock", f"{fmt_bn(bs['snapshot_bill_stock_bn'], SYM, 0)} → {fmt_num(bs[f'end_{yr}_path_bn'], 0)}bn",
             f"{_d(bs['snapshot_date'])} → end-{yr} (plan {fmt_num(bs[f'riksgalden_plan_end_{yr}_bn'], 0)})"),
        _kpi("Liquidity management", f"{fmt_bn(lm['last_outcome_implied_bn'], SYM, 0)} → {fmt_num(lm[f'end_{yr}_implied_bn'], 0)}bn",
             f"end-{lm_m} → end-{yr} (plan {fmt_num(lm[f'riksgalden_plan_end_{yr}_bn'], 0)}; "
             f"end-{yr - 1} {fmt_num(lm[f'end_{yr - 1}_riksgalden_bn'], 0)})"),
        _kpi("Next releases", _d(nxt["date"]) if nxt else "—",
             (f"{pd.Timestamp(nxt['month'] + '-01'):%B} outcome · " if nxt else "")
             + f"next report {_d(fwd['next_report'])} (<span class='days-to' data-date='{fwd['next_report']}'></span>)"),
    ])
    blk_cols = [("cash_requirement_bn", "Net borrowing requirement"), ("bond_gross_nominal_bn", "Nominal SGB gross"),
                ("bond_redemption_nominal_bn", "Nominal SGB redemptions"), ("net_bonds_nominal_bn", "Nominal SGB net"),
                ("net_bonds_linker_bn", "Inflation-linked net (cash redemptions)"), ("net_bonds_fx_bn", "FX bonds net"),
                ("bill_gross_bn", "T-bill gross"), ("net_bills_bn", "T-bill net"), ("other_fin_bn", "Other"),
                ("liquidity_mgmt_need_bn", "Liquidity management (Δ borrowing)")]
    c_roy, c_12 = f"rest of {yr} ({roy_m})", f"next 12m ({n12['window'][:7]} to {n12['window'][-7:]})"
    blk_rows = [{"line": lab, c_roy: fmt_num(roy[c], 1), c_12: fmt_num(n12[c], 1),
                 "next 12m, % GDP": fmt_num(100 * n12[c] / gdp, 2) if gdp else "—"} for c, lab in blk_cols]
    blk_html = _table(blk_rows, ["line", c_roy, c_12, "next 12m, % GDP"])
    score_html = _table(skill_rows(summ["score"]), ["window", "forecaster", "horizon", "n", "MAE"], fmt="{:,.2f}")
    ship_rows = [{"report": r["vintage"], "origin": r["origin"], "targets": r["targets"].replace("..", " to "), "n": r["n"],
                  "Riksgälden annual + model shape": r["shipped"], "model alone": r["model"],
                  "same month last year": r["smly"]} for r in summ["shipped_forward_score"]]
    ship_cols = ["report", "origin", "targets", "n", "Riksgälden annual + model shape", "model alone", "same month last year"]
    ship_html = _table(ship_rows, ship_cols, best_low=ship_cols[4:], fmt="{:,.2f}")
    al = summ["allocation"]
    al_rows = [{"how the gap is spread": ALLOC_LABEL.get(m, m) + (" (shipped)" if m == al["chosen"] else ""),
                "PRE": v.get("PRE"), "OOS": v.get("OOS")} for m, v in al["pre_oos_vs_outcome"].items()]
    al_html = _table(al_rows, ["how the gap is spread", "PRE", "OOS"], fmt="{:,.2f}")
    labels = summ.get("selection_labels", {})
    sel_rows = [{"variant": labels.get(k, k), "PRE": v.get("PRE"), "OOS": v.get("OOS")}
                for k, v in summ["selection_table"].items()]
    sel_html = _table(sel_rows, ["variant", "PRE", "OOS"], fmt="{:,.2f}")
    figs = [fig_nbr(df, rg), fig_rolling(df, annual, vintage), fig_financing(df), fig_errors(bt, rg, chosen, df),
            fig_stocks(df, summ)]
    cards = [f"<div class='card'>{_html(g, first=(i == 0))}</div>" for i, g in enumerate(figs)]
    notes = "".join(f"<li>{n}</li>" for n in summ["notes"])
    src = "".join(f"<li><a href='{s['url']}'>{s['name']}</a> · {s['frequency']} · lag ~{s['lag_days']}d</li>"
                  for s in summ["sources"])
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Sweden central-government cash — net borrowing and financing</title>
  <style>{PAGE_CSS}{TABLE_CSS}{SEASON_CSS}</style>
  <script src="https://cdn.plot.ly/plotly-3.7.0.min.js" charset="utf-8"></script>
</head>
<body>
  <header>
    <h1>Sweden central-government cash — net borrowing and financing</h1>
    <div class="sub">Built {built} · Riksgälden outcomes through {ytd_m} · plan {fwd['riksgalden_vintage']}.
      Sign: net borrowing requirement (NBR) and every financing line, + = borrowing. Buffer = Riksgälden's
      liquidity-management position. History identity residual {fmt_num(summ['identity_resid_bn'], 4)}bn.
      Forward through {fwd.get('horizon_end', fwd.get('window', '')[-7:])}.</div>
  </header>
  <div class="wrap">
    <div class="kpi-row">{kp}</div>
    <p class="sub">The gap to plan lands on {fwd['plan_gap_lands_on']}.</p>
    <h2 class="section">Rest of {yr} and next twelve months (bn SEK)</h2>
    {blk_html}
    <div class="card"><div class="sub">Cash path</div>
    <p style="font:12px/1.45 var(--mono);margin:6px 0"><b>SHIP — Riksgälden annual × PRE seasonal</b>
    · Category×macro fails the RG-anchored PRE gate (research column only)
    · {html_esc.escape(str((summ.get('bottom_up_macro') or {}).get('ship_reason', '')))}</p></div>
    <h2 class="section">Through the year</h2>
    {season_html(df, cc="se", prefix="se",
                 specs=default_specs("cash_requirement_bn", "Net borrowing requirement"),
                 fy_start_month=1, currency="SEK", tall_key="cash",
                 forecast_note="Forecast months follow Riksgälden annuals + seasonal shape through Dec-2030; "
                               "post-plan years carry last plan issuance.")}
    {cat_html}
    <h2 class="section">Net borrowing requirement</h2>
    <div class="grid">{cards[0]}{cards[1]}</div>
    <h2 class="section">Financing</h2>
    <div class="grid">{cards[2]}{cards[4]}</div>
    <h2 class="section">Forecast skill (walk-forward, MAE bn SEK)</h2>
    <p class="sub">{summ['skill_verdict']} {summ['score_note']}</p>
    {score_html}
    <p class="sub">{summ['shipped_forward_score_note']}</p>
    {ship_html}
    <p class="sub">Spreading the gap between Riksgälden's year and the model over the remaining months.
      {al['selection']}</p>
    {al_html}
    <p class="sub">Model variants ranked on PRE (mean MAE over 1–12 months); OOS for information.</p>
    {sel_html}
    <div class="grid">{cards[3]}</div>
    <h2 class="section">Notes</h2>
    <ul class="sub">{notes}</ul>
    <h2 class="section">Sources</h2>
    <ul class="sub">{src}</ul>
  </div>
  <script>
    document.querySelectorAll('.days-to').forEach(function (el) {{
      var d = Math.round((new Date(el.dataset.date + 'T00:00:00') - new Date().setHours(0, 0, 0, 0)) / 864e5);
      el.textContent = d > 0 ? 'in ' + d + ' days' : (d === 0 ? 'today' : 'published');
    }});
  </script>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
