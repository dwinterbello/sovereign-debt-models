"""Norway cash dashboard → ``sovereign_cash/no_cash_dashboard.html``.

One sign convention on the page: + = inflow to the government's accounts at Norges Bank (the cash buffer),
which is liquidity drained from banks. Levels (liquidity, outstanding debt) are plain levels.
"""

from __future__ import annotations

import datetime as dt
import html as _h
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from sovereign_cash.cash_season import (
    SEASON_CSS, category_tape_html, default_specs, no_category_panel, season_html,
)
from sovereign_cash.contract import fmt_bn, fmt_num

ROOT = Path(__file__).resolve().parents[1]
US_FISCAL = ROOT.parent / "us" / "fiscal"
if str(US_FISCAL) not in sys.path:
    sys.path.insert(0, str(US_FISCAL))
from build_dashboard import _T, BLUE, GREEN, GREY, ORANGE, PAGE_CSS, PURPLE, RED, _html, _kpi, _style  # noqa: E402
from build_liquidity_dashboard import TABLE_CSS, _table  # noqa: E402

DERIVED = ROOT / "derived"
OUT = ROOT / "no_cash_dashboard.html"
CCY = "NOK"
THIN_BEFORE = pd.Timestamp("2026-01-01")  # weekly points before this date, daily after
TEAL, BROWN, PINK, OLIVE, LIGHT = "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#c8c8c8"
TAB4 = [("oil_tax", "Petroleum tax", ORANGE), ("sdfi", "SDFI", RED), ("tax_net", "Taxes", BLUE),
        ("vat", "VAT", PURPLE), ("customs_excise", "Excise", GREY), ("nav", "Welfare (NAV)", GREEN),
        ("local_authorities", "Municipalities", BROWN), ("health", "Health trusts", PINK), ("other_net", "Other", OLIVE)]
H_LABEL = {"1bd": "1 business day", "5bd": "5 business days", "10bd": "10 business days", "20bd": "20 business days",
           "60bd": "60 business days", "m+1": "month +1", "m+2": "month +2", "m+3": "month +3"}
OPACITY = {"history": 1.0, "nb": 0.55, "scenario": 0.25}


def _bn(v, dp: int = 1, plus: bool = False) -> str:
    return fmt_bn(v, CCY, dp, plus=plus)


def _mon(s: str) -> str:
    return f"{pd.Timestamp(s + '-01' if len(s) == 7 else s):%b-%Y}"


def _win(w: str) -> str:
    a, b = w.split("..")
    return f"{pd.Timestamp(a + '-01'):%b}–{pd.Timestamp(b + '-01'):%b %Y}"


def _xs(idx) -> list[str]:
    return [f"{t:%Y-%m-%d}" for t in pd.DatetimeIndex(idx)]


def _thin(daily: pd.DataFrame, col: str, start: pd.Timestamp) -> pd.Series:
    s = daily.set_index("date")[col].dropna()
    s = s[s.index >= start]
    pre = s[s.index < THIN_BEFORE].resample("W-FRI").last().dropna()
    return pd.concat([pre, s[s.index >= THIN_BEFORE]]).round(2)


def _line(s: pd.Series, name: str, color: str, width: float = 1.4, dash: str | None = None) -> go.Scatter:
    return go.Scatter(x=_xs(s.index), y=s.tolist(), name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash), hovertemplate="%{y:,.1f}bn")


def _phase(df: pd.DataFrame) -> pd.Series:
    return pd.Series(["scenario" if sc else ("nb" if fc else "history")
                      for fc, sc in zip(df["is_forecast"], df["scenario"])], index=df.index)


def _shade(fig: go.Figure, df: pd.DataFrame) -> None:
    ph = _phase(df)
    half = pd.Timedelta(days=15)
    for key, label, color in (("nb", "Norges Bank path", BLUE), ("scenario", "scenario", GREY)):
        m = df.loc[ph == key, "month"]
        if len(m):
            fig.add_vrect(x0=f"{m.min() - half:%Y-%m-%d}", x1=f"{m.max() + half:%Y-%m-%d}", fillcolor=color,
                          opacity=0.08, line_width=0, annotation_text=label, annotation_position="top left",
                          annotation_font_size=9, annotation_font_color=GREY)


# ── figures ───────────────────────────────────────────────────────────────────────────


def fig_liquidity(daily: pd.DataFrame, fwd: dict) -> go.Figure:
    start = pd.Timestamp("2022-01-01")
    fig = go.Figure()
    fig.add_trace(_line(_thin(daily, "total_liquidity_bn", start), "Total", GREEN, 1.6))
    fig.add_trace(_line(_thin(daily, "structural_liquidity_bn", start), "Structural", _T["text"], 1.2))
    if "cb_certificates_outstanding_bn" in daily:
        c = _thin(daily, "cb_certificates_outstanding_bn", start)
        fig.add_trace(_line(c[c.abs() > 0.01], "Certificates", PURPLE, 1.1))
    if fwd.get("f_operations_active_365d"):
        for col, name, color in (("f_loans_bn", "F-loans", RED), ("f_deposits_bn", "F-deposits", ORANGE)):
            fig.add_trace(_line(_thin(daily, col, start), name, color, 1.0))
    fc = daily[daily["is_forecast"]].set_index("date")
    fig.add_trace(_line(fc["structural_liquidity_fc_bn"].dropna().round(2), "NB path: structural", _T["text"], 1.4, "dot"))
    if "structural_liquidity_incl_cbc_fc_bn" in fc:
        fig.add_trace(_line(fc["structural_liquidity_incl_cbc_fc_bn"].dropna().round(2), "NB path: incl. certificates",
                            GREEN, 1.6, "dot"))
    fig = _style(fig, title=f"Banking-system liquidity, {CCY} bn (weekly before {THIN_BEFORE:%Y}, daily after)",
                 ylab=f"{CCY} bn", height=420, rangeslider=True)
    fig.update_xaxes(range=[f"{fc.index.min() - pd.DateOffset(years=2):%Y-%m-%d}",
                            f"{fc.index.max() + pd.Timedelta(days=10):%Y-%m-%d}"])
    return fig


def fig_buffer(df: pd.DataFrame) -> go.Figure:
    """Δbuffer by source, + = inflow to the government accounts; bars sum to the line."""
    d = df[df["month"] >= pd.Timestamp("2022-01-01")].reset_index(drop=True)
    ph = _phase(d)
    op = [OPACITY[p] for p in ph]
    x = _xs(d["month"])
    hist = ph == "history"
    fiscal_fc = (-d["cash_requirement_bn"]).where(~hist)
    series = (("Fiscal ex-petroleum", -d["cash_nonpetroleum_bn"], BLUE),
              ("Petroleum tax + SDFI", -d["cash_petroleum_bn"], RED),
              ("Fiscal, NB-implied", fiscal_fc, "#5b7fa6"),
              ("Oil-fund FX", d["oil_fund_legs_bn"], PURPLE),
              ("NGB net", d["net_bonds_bn"], TEAL),
              ("T-bills net", d["net_bills_bn"], ORANGE),
              ("Other securities", d["securities_other_bn"], GREY))
    fig = go.Figure()
    for name, y, c in series:
        if y.fillna(0).abs().sum() == 0:
            continue
        fig.add_trace(go.Bar(x=x, y=y.round(2).tolist(), name=name, marker=dict(color=c, opacity=op),
                             hovertemplate="%{y:,.1f}bn"))
    fig.add_trace(go.Scatter(x=x, y=d["buffer_change_bn"].round(2).tolist(), name="Δ buffer", mode="lines",
                             line=dict(color=_T["text"], width=1.3), hovertemplate="%{y:,.1f}bn"))
    _shade(fig, d)
    fig = _style(fig, title=f"Government cash buffer change by source, {CCY} bn (+ = inflow to the accounts)",
                 ylab=f"{CCY} bn", height=440, rangeslider=True)
    fig.update_layout(barmode="relative", bargap=0.15)
    return fig


def fig_fiscal(df: pd.DataFrame) -> go.Figure:
    d = df[(~df["is_forecast"]) & (df["month"] >= pd.Timestamp("2022-01-01"))]
    x = _xs(d["month"])
    fig = go.Figure()
    for c, name, color in TAB4:
        fig.add_trace(go.Bar(x=x, y=(-d[f"liq_{c}_bn"]).round(2).tolist(), name=name, marker_color=color,
                             hovertemplate="%{y:,.1f}bn"))
    fig.add_trace(go.Scatter(x=x, y=(-d["cash_requirement_bn"]).round(2).tolist(), name="Total", mode="lines",
                             line=dict(color=_T["text"], width=1.3), hovertemplate="%{y:,.1f}bn"))
    fig = _style(fig, title=f"Fiscal flows through the government accounts, {CCY} bn (+ = receipts)",
                 ylab=f"{CCY} bn", height=440, rangeslider=True)
    fig.update_layout(barmode="relative", bargap=0.15)
    return fig


def fig_ladder(fwd: dict) -> go.Figure:
    lad = pd.DataFrame(fwd.get("ladder", []))
    fig = go.Figure()
    if not lad.empty:
        lab = [f"{r['instr']}" for _, r in lad.iterrows()]
        hover = [f"{r['isin']} · matures {r['maturity']}" for _, r in lad.iterrows()]
        for kind, color, name in (("bill", ORANGE, "T-bills, market"), ("bond", BLUE, "NGB, market")):
            fig.add_trace(go.Bar(x=lab, y=lad["market_bn"].where(lad["type"] == kind).round(2).tolist(), name=name,
                                 marker_color=color, customdata=hover, hovertemplate="%{customdata}<br>%{y:,.1f}bn"))
        fig.add_trace(go.Bar(x=lab, y=lad["own_bn"].round(2).tolist(), name="Government own holdings",
                             marker_color=LIGHT, customdata=hover, hovertemplate="%{customdata}<br>%{y:,.1f}bn"))
    fig = _style(fig, title=f"Outstanding NGBs and T-bills by maturity, {CCY} bn (as of {fwd.get('ladder_asof', '')})",
                 ylab=f"{CCY} bn", height=380)
    fig.update_layout(barmode="stack", bargap=0.25, legend=dict(y=1.02, yanchor="bottom"), margin=dict(t=78),
                      title=dict(y=0.985, yanchor="top"))
    fig.update_xaxes(tickangle=-45, type="category")
    return fig


def fig_errors(sc: pd.DataFrame, h: str = "20bd") -> go.Figure:
    d = sc[(sc["kind"] == "daily") & (sc["h"] == h)].sort_values("target").set_index("target")
    def roll(e: pd.Series) -> pd.Series:
        return e.abs().rolling(50, min_periods=20).mean().resample("W-FRI").last().dropna().round(2)

    fig = go.Figure()
    fig.add_trace(_line(roll(d["nb"] - d["actual"]), "Norges Bank", BLUE, 1.4))
    fig.add_trace(_line(roll(d["naive_ly"] - d["actual"]), "Same weekday a year earlier", GREY, 1.4))
    fig.add_trace(_line(roll(d["actual"]), "Flat", ORANGE, 1.0, "dot"))
    for x0, x1, label, color in (("2014-01-01", "2019-12-31", "PRE", GREY), ("2020-01-01", "2021-12-31", "not scored", RED),
                                 ("2022-01-01", f"{d.index.max():%Y-%m-%d}", "OOS", BLUE)):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0, annotation_text=label,
                      annotation_position="top left", annotation_font_size=9, annotation_font_color=GREY)
    return _style(fig, title=f"Structural-liquidity change over {H_LABEL[h]}: MAE over the last 50 vintages, {CCY} bn",
                  ylab=f"{CCY} bn", height=380, rangeslider=True)


# ── page ──────────────────────────────────────────────────────────────────────────────


def kpis(summ: dict) -> str:
    fwd = summ["forward"]
    v, gp, ngb, bl, nr = fwd["nb_vintage"], fwd["gpfg_fx"], fwd["ngb"], fwd["bills"], fwd.get("next_redemption")
    hz = fwd["nb_horizon"]
    path_end = pd.Timestamp(v["path_end"])
    out = []
    incl = fwd.get("structural_liquidity_incl_cb_certificates_year_end_fc_bn")
    out.append(_kpi("Banking-system liquidity",
                    f"{_bn(fwd['total_liquidity_now_bn'], 0)} → {fmt_num(incl, 0)}bn" if incl is not None
                    else _bn(fwd["total_liquidity_now_bn"], 0),
                    f"total {pd.Timestamp(fwd['structural_liquidity_now_date']):%d %b} → NB path incl. certificates "
                    f"{path_end:%d %b} · structural {fmt_num(fwd['structural_liquidity_now_bn'], 0)} → "
                    f"{fmt_num(fwd['structural_liquidity_year_end_fc_bn'], 0)} · certificates "
                    f"{fmt_num(fwd.get('cb_certificates_outstanding_bn'), 1)}"))
    out.append(_kpi(f"Δ buffer {_win(fwd['window'])}", _bn(hz["buffer_change_bn"], plus=True),
                    f"fiscal {fmt_num(-hz['cash_requirement_bn'], plus=True)} · NGB {fmt_num(hz['net_bonds_bn'], plus=True)}"
                    f" · bills {fmt_num(hz['net_bills_bn'], plus=True)} · oil-fund FX "
                    f"{fmt_num(hz['oil_fund_legs_bn'], plus=True)}"))
    per_day = abs(gp["announced_nok_mn_per_day"])
    out.append(_kpi(f"Oil-fund FX {_mon(gp['announced_month'])}",
                    f"NB {'buys' if gp['nb_buys_fx'] else 'sells'} FX: {CCY} {per_day:,}m/day",
                    ("NOK paid out of the government accounts" if gp["nb_buys_fx"]
                     else "NOK paid into the government accounts")
                    + f" · fund FX into the accounts {gp['ytd_year']} YTD {fmt_num(gp['oil_fund_legs_ytd_bn'], plus=True)}"))
    for g in ngb.get("quarter_guidance", []):
        per = g["path_bn"] / g["auctions"] if g["auctions"] else float("nan")
        prog = ngb.get("programme_bn") or [float("nan")] * 2
        out.append(_kpi(f"NGB Q{g['quarter']}-{g['year']} gross", f"{CCY} {g['bn'][0]:.0f}–{g['bn'][1]:.0f}bn",
                        f"{g['auctions']} auctions ≈ {fmt_num(per)}bn each · {ngb['year']}: issued "
                        f"{fmt_num(ngb['issued_ytd_bn'], 0)}, path {fmt_num(ngb['programme_path_bn'], 1)} of "
                        f"{prog[0]:.0f}–{prog[1]:.0f}"))
    if nr:
        bb = ", ".join(f"{pd.Timestamp(d):%d %b}" for d in nr["buyback_dates"]) or "none announced"
        out.append(_kpi(f"{nr['instr']} matures {pd.Timestamp(nr['maturity']):%d %b %Y}",
                        f"{CCY} {fmt_num(nr['low_bn'], 0)}–{fmt_num(nr['high_bn'], 0)}bn",
                        f"central {fmt_num(nr['central_bn'])} · market now {fmt_num(nr['market_now_bn'])} · expected "
                        f"buyback {fmt_num(nr['expected_buyback_bn'])} ({bb})"))
    aim = bl.get("aim_end_year_bn")
    out.append(_kpi("Market T-bills", f"{_bn(bl['market_stock_now_bn'], 0)} → {fmt_num(bl['stock_horizon_end_bn'], 0)}bn",
                    f"to {pd.Timestamp(fwd['window'].split('..')[1] + '-01'):%b-%Y} from the calendar "
                    f"({bl['calendar_auctions_left']} auctions)"
                    + (f" · NB aim {fmt_num(aim, 0)} (range {bl['aim_range_bn'][0]:.0f}–{bl['aim_range_bn'][1]:.0f})"
                       if aim is not None else "")))
    pm = fwd.get("partial_month")
    if pm:
        out.append(_kpi(f"{_mon(pm['month'])} so far", _bn(pm["buffer_change_bn"], plus=True),
                        f"Δ buffer · {pm['bdays_observed']}/{pm['bdays_in_month']} business days observed"))
    return "".join(out)


def decomposition(summ: dict) -> str:
    fwd = summ["forward"]
    hz = fwd["nb_horizon"]
    col = f"{_win(fwd['window'])} (Norges Bank path)"
    rows = [("Fiscal flows (NB-implied residual)", -hz["cash_requirement_bn"]),
            ("Oil-fund FX and transfer financing", hz["oil_fund_legs_bn"]),
            ("NGB gross", hz["bond_gross_bn"]),
            ("NGB redemptions and buybacks", -hz["bond_redemption_bn"]),
            ("T-bill gross", hz["bill_gross_bn"]),
            ("T-bill redemptions", -hz["bill_redemption_bn"]),
            ("Other securities", hz["securities_other_bn"]),
            ("Δ buffer", hz["buffer_change_bn"])]
    body = [{"line": k, col: fmt_num(v, plus=True)} for k, v in rows if not (k == "Other securities" and v == 0)]
    body.append({"line": "memo: NB structural-liquidity change (+ = paid to banks)",
                 col: fmt_num(hz["structural_change_fc_bn"], plus=True)})
    return _table(body, ["line", col])


def calendar_table(summ: dict) -> str:
    ev = summ["forward"].get("calendar_next", [])
    rows = [{"date": f"{pd.Timestamp(e['date']):%a %d %b %Y}", "event": e["event"], "instrument": e["instrument"],
             f"{CCY} bn": fmt_num(e["size_bn"], 1), "basis": e["basis"]} for e in ev]
    return _table(rows, ["date", "event", "instrument", f"{CCY} bn", "basis"]) if rows else ""


def score_tables(summ: dict) -> tuple[str, str, str]:
    rows = [{"window": r["window"], "horizon": H_LABEL.get(r["h"], r["h"]), "n": r["n"], "Norges Bank": r["model"],
             "a year earlier": r["benchmark"], "flat": r["flat"]} for r in summ["score"]]
    score = _table(rows, ["window", "horizon", "n", "Norges Bank", "a year earlier", "flat"],
                   best_low=["Norges Bank", "a year earlier", "flat"], fmt="{:,.1f}")
    nv = summ["cash_requirement_naive_score"]
    naive = _table([{"window": r["window"], "n": r["n"], "same month a year earlier": r["same_month_last_year"],
                     "3-year same-month mean": r["same_month_3y_mean"], "zero": r["zero"]} for r in nv],
                   ["window", "n", "same month a year earlier", "3-year same-month mean", "zero"],
                   best_low=["same month a year earlier", "3-year same-month mean", "zero"], fmt="{:,.1f}")
    oos = next((r for r in nv if r["window"] == "OOS"), None)
    verdict = ""
    if oos:
        m, z = oos["same_month_last_year"], oos["zero"]
        word = "beats" if m < 0.97 * z else ("ties" if m <= 1.03 * z else "does not beat")
        verdict = (f"Beyond Norges Bank's path, fiscal flows copy the same month a year earlier: OOS MAE {fmt_num(m)} "
                   f"vs {fmt_num(z)} for zero — it {word} zero, so those months are a scenario, not a forecast.")
    return score, naive, verdict


def build() -> Path:
    df = pd.read_parquet(DERIVED / "no_cash_monthly.parquet")
    df["scenario"] = df["scenario"].fillna(False).astype(bool) if "scenario" in df else False
    daily = pd.read_parquet(DERIVED / "no_liquidity_daily.parquet")
    sc = pd.read_parquet(DERIVED / "no_nb_forecast_score.parquet")
    summ = json.loads((DERIVED / "no_cash_summary.json").read_text())
    fwd = summ["forward"]
    v = fwd["nb_vintage"]
    xc = summ["structural_crosscheck_resid_bn"]
    try:
        cat_panel, cat_specs = no_category_panel()
        cat_html = category_tape_html(
            cat_panel, cc="no", prefix="no", specs=cat_specs, currency="NOK",
            note="Tab4 lines under the shipped beyond-NB macro rules (NB residual inside the vintage).")
    except Exception:  # noqa: BLE001
        cat_html = ""
    figs = [fig_liquidity(daily, fwd), fig_buffer(df), fig_fiscal(df), fig_ladder(fwd), fig_errors(sc)]
    cards = [f"<div class='card'>{_html(g, first=(i == 0))}</div>" for i, g in enumerate(figs)]
    score_html, naive_html, verdict = score_tables(summ)
    notes = "".join(f"<li>{_h.escape(n)}</li>" for n in summ["notes"])
    src = "".join(f"<li><a href='{s['url']}'>{_h.escape(s['name'])}</a> · {_h.escape(s['frequency'])}</li>"
                  for s in summ["sources"])
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Norway government cash and banking-system liquidity</title>
  <style>{PAGE_CSS}{TABLE_CSS}{SEASON_CSS}</style>
  <script src="https://cdn.plot.ly/plotly-3.7.0.min.js" charset="utf-8"></script>
</head>
<body>
  <header>
    <h1>Norway government cash and banking-system liquidity</h1>
    <div class="sub">{built} · liquidity statistics through {pd.Timestamp(summ['asof_daily']):%d %b %Y} · Norges Bank
      forecast vintage {pd.Timestamp(v['vintage']):%d %b %Y}, path to {pd.Timestamp(v['path_end']):%d %b %Y}.
      Buffer = Treasury accounts + oil-fund krone account at Norges Bank (level unpublished).
      Sign: + = inflow to the accounts = liquidity drained from banks. Identity closes by construction (max residual
      {fmt_num(summ['identity_resid_bn'], 2)}bn); Norges Bank's two datasets differ by a median
      {fmt_num(xc['median_abs'], 2)}bn a month. Cash forward through {fwd.get('horizon_end', '')}.</div>
  </header>
  <div class="wrap">
    <div class="kpi-row">{kpis(summ)}</div>
    <h2 class="section">Δ buffer over Norges Bank's forecast path ({CCY} bn, + = inflow)</h2>
    {decomposition(summ)}
    <h2 class="section">Next three months: auctions, buybacks, maturities</h2>
    {calendar_table(summ)}
    <h2 class="section">Through the year</h2>
    {season_html(df, cc="no", prefix="no",
                 specs=default_specs("cash_requirement_bn", "Cash requirement (Δ gov accounts)"),
                 fy_start_month=1, currency="NOK", tall_key="cash",
                 forecast_note="Inside Norges Bank's path: structural vintage. Beyond: Tab4×macro (SHIP) "
                               "through Dec-2030.")}
    {cat_html}
    <h2 class="section">Liquidity and the buffer</h2>
    <div class="grid">{cards[0]}{cards[1]}</div>
    <h2 class="section">Fiscal flows and outstanding debt</h2>
    <div class="grid">{cards[2]}{cards[3]}</div>
    <h2 class="section">Forecast skill ({CCY} bn MAE)</h2>
    <p class="sub">{_h.escape(summ['score_note'])} This page's own Δ buffer path is unscored.</p>
    {score_html}
    <div class="grid">{cards[4]}</div>
    <p class="sub">{_h.escape(verdict)}</p>
    {naive_html}
    <h2 class="section">Notes</h2>
    <ul class="sub">{notes}</ul>
    <h2 class="section">Sources</h2>
    <ul class="sub">{src}</ul>
  </div>
</body>
</html>"""
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
