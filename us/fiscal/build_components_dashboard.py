"""TGA components: per-bucket DTS flow forecasts and the financing bridge to the target.

Reads ``derived/`` (dts_bucket_daily, tga_forward_buckets, tga_forward_path,
tga_forward_summary, flow_bucket_score, flow_model_choice) and writes
``tga_components_dashboard.html``. ``sections_html`` is the body fragment used
by that page.

  PYTHONPATH=. .venv/bin/python us/fiscal/build_components_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from build_dashboard import (  # noqa: E402
    FORECAST_DASH,
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    RED,
    _T,
    _html,
    _kpi,
    _style,
    year_color,
)
from portal import theme  # noqa: E402
from build_liquidity_dashboard import TABLE_CSS, _read, _table  # noqa: E402

DERIVED = HERE / "derived"
OUT = HERE / "tga_components_dashboard.html"
SEASON_FROM = 2019
BRIDGE_HIST_YEARS = 3

# Weekly bridge panels: (column, title, color). Fed and other are derived aggregates.
BRIDGE_PANELS = (
    ("op_flow_bn", "Net operating flow", RED),
    ("net_coupons", "Net coupons", ORANGE),
    ("net_bills", "Announced net bills", BLUE),
    ("implied_bills_bn", "Implied bills to target", GREEN),
    ("fed_bn", "Fed rolls + purchases", PURPLE),
    ("other_fin_bn", "Buybacks + nonmarketables", GREY),
)

# Extra CSS for the bucket season toggles / wider table block on the merged cash page.
SECTIONS_CSS = """
    .wrap-components { max-width: 1480px; margin: 0 auto; padding: 10px 16px 36px; }
    .toggles { display: flex; gap: 6px; margin: 8px 0 4px; flex-wrap: wrap; }
    .toggles button { font: 600 11px/1 var(--mono); padding: 7px 12px; border: 1px solid var(--line);
      background: var(--panel-hi); color: var(--text); cursor: pointer; }
    .toggles button.on { background: var(--text); color: var(--panel-hi); }
    .toggles .chk { font: 600 11px/1 var(--mono); display: flex; align-items: center; gap: 6px;
      padding: 0 8px; color: var(--text); cursor: pointer; }
    .season { height: 460px; }
    .season.tall { height: 540px; }
"""

LABELS = {
    "dep_withheld": "Withheld income & employment taxes",
    "dep_nonwithheld": "Non-withheld individual taxes",
    "dep_corporate": "Corporate income taxes",
    "dep_customs": "Customs duties (rate × imports)",
    "dep_fed_earnings": "Fed earnings",
    "dep_other_tax": "Excise, estate, FUTA, other taxes",
    "dep_medicare_prem": "Medicare premiums",
    "dep_education": "Education receipts",
    "dep_other": "Other deposits",
    "wd_ssa": "Social Security / SSI",
    "wd_medicare": "Medicare",
    "wd_medicaid": "Medicaid",
    "wd_interest": "Interest on Treasury securities",
    "wd_refunds": "Tax refunds",
    "wd_defense": "Defense",
    "wd_va": "Veterans Affairs",
    "wd_salaries": "Federal salaries",
    "wd_opm": "OPM retirement & insurance",
    "wd_education": "Education outlays",
    "wd_unemployment": "Unemployment insurance",
    "wd_snap": "SNAP",
    "wd_hhs_other": "HHS other",
    "wd_other": "Other withdrawals",
}


def _gdp_label(gdp: pd.Series) -> str:
    raw = pd.read_csv(HERE / "data" / "cache" / "fred_gdp.csv", parse_dates=["date"])
    last = raw.dropna().iloc[-1]
    return f"{float(gdp.dropna().iloc[-1]):,.0f}bn SAAR, last quarter {pd.Timestamp(last['date']):%Y-%m-%d}, held flat after that"


def _nominal_gdp(index: pd.DatetimeIndex) -> pd.Series:
    """FRED nominal GDP, $bn SAAR, time-interpolated and held flat after the last quarter."""
    from tga_deficit import interpolate_to_daily

    raw = pd.read_csv(HERE / "data" / "cache" / "fred_gdp.csv", parse_dates=["date"])
    q = raw.set_index("date")["value"].astype(float)
    return interpolate_to_daily(q, pd.DatetimeIndex(index)).rename("gdp_saar_bn")


def _year_views(s: pd.Series, gdp: pd.Series | None = None, digits: int = 1) -> dict:
    """Daily flow and the within-year cumulative stock, on the FY and CY calendars.

    ``s`` is the daily print, $bn. The stock is the sum of those prints since 1 October
    (fiscal) or 1 January (calendar). The flow is the daily change in that stock.
    With ``gdp`` (nominal SAAR, $bn) the packed values are 100 × series / GDP.
    Each year is ``[day-of-year, values]``."""
    from tga_deficit import fy_pivot, us_fiscal_year

    s = s.dropna().sort_index().astype(float)
    if s.empty:
        return {cal: {kind: {} for kind in ("flow", "stock")} for cal in ("fy", "cy")}

    def _over(series: pd.Series) -> pd.Series:
        if gdp is None:
            return series
        return 100.0 * series / gdp.reindex(series.index)

    fy = pd.Series([us_fiscal_year(t) for t in s.index], index=s.index)

    def _pack(df: pd.DataFrame) -> dict:
        out = {}
        for year in df.columns:
            if int(year) < SEASON_FROM:
                continue
            col = df[year].dropna()
            if len(col) < 5:
                continue
            out[str(int(year))] = [col.index.astype(int).tolist(), [round(float(v), digits) for v in col]]
        return out

    def _cy_pivot(series: pd.Series) -> pd.DataFrame:
        tmp = pd.DataFrame({"y": series.index.year, "doy": series.index.dayofyear, "v": series.to_numpy()})
        parts = [pd.Series(sl["v"].to_numpy(), index=sl["doy"].to_numpy(), name=int(y))
                 for y, sl in tmp.groupby("y")]
        return pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()

    daily = _over(s)
    cy_stock = _over(s.groupby(s.index.year).cumsum())
    fy_stock = _over(s.groupby(fy).cumsum())
    return {
        "fy": {"flow": _pack(fy_pivot(daily)), "stock": _pack(fy_pivot(fy_stock))},
        "cy": {"flow": _pack(_cy_pivot(daily)), "stock": _pack(_cy_pivot(cy_stock))},
    }


def bucket_forecast(short: pd.DataFrame) -> pd.DataFrame:
    """Per-bucket path through the fiscal-path end (2029). The steer file stops at the refunding target."""
    longp = DERIVED / "fiscal_path_buckets.parquet"
    if not longp.exists():
        return short
    long = pd.read_parquet(longp)
    if long.empty or short.empty or long.index.max() <= short.index.max():
        return short
    cols = [c for c in short.columns if c in long.columns]
    return long[cols]


def _ticks(cal: str) -> tuple[list[int], list[str]]:
    start = pd.Timestamp("2025-10-01") if cal == "fy" else pd.Timestamp("2025-01-01")
    vals, labs = [], []
    for m in range(12):
        d = start + pd.DateOffset(months=m)
        vals.append(int((d - start).days + 1))
        labs.append(d.strftime("%b"))
    return vals, labs


def season_payload(panel: pd.DataFrame, fc: pd.DataFrame, buckets: list[str]) -> dict:
    """History in $mn, forecast buckets in $bn. Years after the last print are dashed through 2029."""
    from tga_deficit import fy_day_of_year, us_fiscal_year

    fc = bucket_forecast(fc)
    origin = panel.index.max()
    gdp = _nominal_gdp(panel.index.union(fc.index))
    views, pct = {}, {}

    def _add(key: str, s: pd.Series) -> None:
        views[key] = _year_views(s)
        pct[key] = _year_views(s, gdp, digits=3)

    for b in buckets:
        hist = panel[b].dropna() / 1000.0
        fut = fc[b].dropna() if b in fc.columns else pd.Series(dtype=float)
        fut = fut[fut.index > origin]
        s = pd.concat([hist, fut])
        _add(b, s[~s.index.duplicated(keep="first")].sort_index())
    wd_cols = [b for b in buckets if b.startswith("wd_")]
    for side, cols in (("dep_total", [b for b in buckets if b.startswith("dep_")]),
                       ("wd_total", wd_cols),
                       ("wd_total_xint", [b for b in wd_cols if b != "wd_interest"])):
        hist = panel[cols].sum(axis=1) / 1000.0
        fut = fc[cols].sum(axis=1)
        fut = fut[fut.index > origin]
        s = pd.concat([hist, fut])
        _add(side, s[~s.index.duplicated(keep="first")].sort_index())
    return {
        "views": views,
        "pct": pct,
        "colors": {y: year_color(int(y)) for y in
                   {y for side in views.values() for cal in side.values() for kind in cal.values() for y in kind}},
        "ticks": {cal: _ticks(cal) for cal in ("fy", "cy")},
        "current": {"fy": int(us_fiscal_year(origin)), "cy": int(origin.year)},
        "histEnd": {"fy": int(fy_day_of_year(origin)), "cy": int(origin.dayofyear)},
        "origin": origin.strftime("%Y-%m-%d"),
        "forecastEnd": fc.index.max().strftime("%Y-%m-%d") if len(fc) else origin.strftime("%Y-%m-%d"),
        "gdp": _gdp_label(gdp),
    }


def fig_bridge(path: pd.DataFrame, fc: pd.DataFrame, summ: dict) -> go.Figure:
    t = pd.Timestamp(summ["target_date"])
    p = path.loc[:t]
    f = fc.loc[:t]
    dep = float(f[[c for c in f.columns if c.startswith("dep_")]].sum().sum())
    wd = float(f[[c for c in f.columns if c.startswith("wd_")]].sum().sum())
    steps = [
        ("Net operating flow", dep - wd),
        ("Net coupons", float(p["net_coupons"].sum())),
        ("Announced net bills", float(p["net_bills"].sum())),
        ("Fed rolls", float(p["soma_bn"].sum())),
        ("Fed purchases maturing", float(p["soma_purchases_bn"].sum())),
        ("Buybacks", float(p["buyback_bn"].sum())),
        ("Nonmarketables", float(p["nonmkt_bn"].sum())),
        ("Implied bills to target", float(p["implied_bills_bn"].sum())),
    ]
    tga0 = summ["tga_origin_bn"]
    end = float(p["tga_target_bn"].iloc[-1])
    other = end - tga0 - sum(v for _, v in steps)
    if abs(other) > 0.5:
        steps.append(("Other", other))
    fig = go.Figure(go.Waterfall(
        x=["TGA " + summ["origin"]] + [s[0] for s in steps] + ["TGA " + summ["target_date"]],
        y=[tga0] + [s[1] for s in steps] + [end],
        measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        text=[f"{tga0:,.0f}"] + [f"{s[1]:+,.0f}" for s in steps] + [f"{end:,.0f}"], textposition="outside",
        increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color=GREY, width=0.6)),
    ))
    return _style(fig, title=f"Bridge from today's TGA to the {summ['target_date']} target ($bn)", ylab="$bn", height=520)


def fig_cumulative(path: pd.DataFrame, fc: pd.DataFrame, tga0: float) -> go.Figure:
    op = (fc[[c for c in fc.columns if c.startswith("dep_")]].sum(axis=1)
          - fc[[c for c in fc.columns if c.startswith("wd_")]].sum(axis=1)).cumsum()
    fig = go.Figure()
    for y, name, col in ((op, "Net operating flow", RED),
                         (path["net_coupons"].cumsum(), "Net coupons", ORANGE),
                         ((path["net_bills"] + path["implied_bills_bn"]).cumsum(), "Net bills (announced + implied)", BLUE),
                         (path["drift_bn"].cumsum(), "Financing drift (Fed rolls, buybacks, nonmkt)", PURPLE)):
        fig.add_trace(go.Scatter(x=y.index, y=y, name=name, mode="lines", line=dict(color=col, width=1.5)))
    fig.add_trace(go.Scatter(x=path.index, y=path["tga_target_bn"] - tga0, name="ΔTGA", mode="lines",
                             line=dict(color=_T["text"], width=2, dash="dot")))
    return _style(fig, title="Cumulative contributions to the TGA from the origin ($bn)", ylab="$bn", height=520)


def _bridge_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fed / other aggregates used by the weekly bridge panels."""
    out = df.copy()
    soma = out["soma_bn"] if "soma_bn" in out.columns else 0.0
    purch = out["soma_purchases_bn"] if "soma_purchases_bn" in out.columns else 0.0
    out["fed_bn"] = soma + purch
    bb = out["buyback_bn"] if "buyback_bn" in out.columns else 0.0
    nm = out["nonmkt_bn"] if "nonmkt_bn" in out.columns else 0.0
    out["other_fin_bn"] = bb + nm
    return out


def _bridge_history(origin: pd.Timestamp) -> pd.DataFrame:
    """Daily $bn history for bridge panels through the last DTS print."""
    start = origin - pd.Timedelta(days=BRIDGE_HIST_YEARS * 365 + 30)
    parts: dict[str, pd.Series] = {}

    buckets = DERIVED / "dts_bucket_daily.parquet"
    if buckets.exists():
        b = pd.read_parquet(buckets)
        if "dep_total" in b.columns and "wd_total" in b.columns:
            parts["op_flow_bn"] = (b["dep_total"] - b["wd_total"]) / 1000.0

    ladder = DERIVED / "auction_ladder.parquet"
    if ladder.exists():
        al = pd.read_parquet(ladder)
        for c in ("net_coupons", "net_bills"):
            if c in al.columns:
                parts[c] = al[c]

    recon = DERIVED / "financing_reconcile.parquet"
    if recon.exists():
        r = pd.read_parquet(recon)
        if "soma_net_bn" in r.columns:
            parts["fed_bn"] = r["soma_net_bn"]
        bb = r["buyback_bn"] if "buyback_bn" in r.columns else None
        nm = r["nonmkt"] if "nonmkt" in r.columns else (r["nonmkt_bn"] if "nonmkt_bn" in r.columns else None)
        if bb is not None or nm is not None:
            parts["other_fin_bn"] = (bb if bb is not None else 0.0) + (nm if nm is not None else 0.0)

    if not parts:
        return pd.DataFrame()
    hist = pd.DataFrame(parts).sort_index()
    hist.index = pd.DatetimeIndex(hist.index)
    hist["implied_bills_bn"] = 0.0
    return hist.loc[start:origin]


def _fiscal_path_forward(origin: pd.Timestamp) -> pd.DataFrame | None:
    """Longer unchanged fiscal-path forecast when present; else None."""
    fp = DERIVED / "fiscal_path.parquet"
    if not fp.exists():
        return None
    p = pd.read_parquet(fp)
    if isinstance(p.index, pd.MultiIndex):
        p = p.xs("unchanged")
    p = p[p.index > origin]
    if p.empty:
        return None
    return _bridge_derived(p)


def fig_bridge_weekly(path: pd.DataFrame, origin: pd.Timestamp) -> go.Figure:
    """2×3 small multiples: weekly $bn of each bridge component, history + pale forecast."""
    from plotly.subplots import make_subplots

    hist = _bridge_history(origin)
    fut = _fiscal_path_forward(origin)
    if fut is None or fut.empty:
        fut = _bridge_derived(path[path.index > origin])

    titles = [t for _, t, _ in BRIDGE_PANELS]
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=titles, shared_xaxes=True,
        vertical_spacing=0.10, horizontal_spacing=0.06,
    )
    for i, (col, _title, color) in enumerate(BRIDGE_PANELS):
        row, col_i = divmod(i, 3)
        row, col_i = row + 1, col_i + 1
        h = hist[col].dropna() if col in hist.columns else pd.Series(dtype=float)
        f = fut[col].dropna() if col in fut.columns else pd.Series(dtype=float)
        hw = h.resample("W-FRI").sum() if len(h) else h
        fw = f.resample("W-FRI").sum() if len(f) else f
        if len(hw):
            fig.add_trace(
                go.Scatter(
                    x=hw.index, y=hw, mode="lines", name="History",
                    legendgroup="hist", showlegend=(i == 0),
                    line=dict(color=color, width=1.4),
                ),
                row=row, col=col_i,
            )
        if len(fw):
            # Underlay + broken stroke (same pattern as cash-page forecasts).
            fig.add_trace(
                go.Scatter(
                    x=fw.index, y=fw, mode="lines", name="Forecast",
                    legendgroup="fc", showlegend=False,
                    line=dict(color=theme.rgba(color, 0.45), width=1.4),
                    hoverinfo="skip",
                ),
                row=row, col=col_i,
            )
            fig.add_trace(
                go.Scatter(
                    x=fw.index, y=fw, mode="lines", name="Forecast",
                    legendgroup="fc", showlegend=(i == 0),
                    line=dict(color=color, width=1.4, dash=FORECAST_DASH),
                ),
                row=row, col=col_i,
            )
        fig.add_vline(
            x=origin, line_color="#000000", line_width=1.0,
            row=row, col=col_i,
        )
        fig.add_hline(y=0, line_color=GREY, line_width=0.6, row=row, col=col_i)

    fig = _style(
        fig,
        title="Bridge components — weekly $bn (history vs forecast)",
        ylab="$bn / week",
        height=720,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0))
    fig.update_annotations(font=dict(size=11, color=_T["text"], family=theme.PLOT_FONT))
    fig.update_yaxes(title_text="")
    return fig


def bucket_table(panel: pd.DataFrame, fc: pd.DataFrame, summ: dict, score: list[dict] | None,
                 choice: dict | None) -> str:
    t = pd.Timestamp(summ["target_date"])
    origin = panel.index.max()
    f = fc.loc[:t]
    ly = panel.loc[origin + pd.Timedelta(days=1) - pd.Timedelta(weeks=52):t - pd.Timedelta(weeks=52)] / 1000.0
    lvl = panel.loc[:origin].tail(250).mean() / 1000.0
    sc = {r["bucket"]: r for r in (score or [])}
    rows = []
    for b in fc.columns:
        s = sc.get(b, {})
        rows.append({
            "bucket": LABELS.get(b, b), "forecast to target": float(f[b].sum()),
            "same window −52w": float(ly[b].sum()), "trailing level × days": float(lvl[b] * len(f)),
            "daily MAE PRE model": s.get("model_PRE", np.nan), "PRE flat": s.get("flat_PRE", np.nan),
            "OOS model": s.get("model_OOS", np.nan), "OOS flat": s.get("flat_OOS", np.nan),
            "uses": (choice or {}).get(b, "model"),
        })
    dep = [r for r, b in zip(rows, fc.columns) if b.startswith("dep_")]
    wd = [r for r, b in zip(rows, fc.columns) if b.startswith("wd_")]
    tot = lambda rs, k: float(sum(r[k] for r in rs))  # noqa: E731
    for name, rs, sign in (("Total receipts", dep, 1), ("Total outlays", wd, 1)):
        rows.append({"bucket": name, **{k: tot(rs, k) for k in ("forecast to target", "same window −52w",
                                                                  "trailing level × days")}, "uses": ""})
    rows.append({"bucket": "Net operating flow", **{k: tot(dep, k) - tot(wd, k) for k in
                                                     ("forecast to target", "same window −52w",
                                                      "trailing level × days")}, "uses": ""})
    cols = ["bucket", "forecast to target", "same window −52w", "trailing level × days", "daily MAE PRE model",
            "PRE flat", "OOS model", "OOS flat", "uses"]
    rows = [{c: ("—" if r.get(c) is None or (isinstance(r.get(c), float) and not np.isfinite(r[c])) else r[c])
             for c in cols} for r in rows]
    return _table(rows, cols, fmt="{:,.2f}")


def _season_html(payload: dict, buckets: list[str]) -> str:
    dep = [b for b in buckets if b.startswith("dep_")]
    wd = [b for b in buckets if b.startswith("wd_")]

    def cards(keys: list[tuple[str, str]], tall: bool = False) -> str:
        cls = "season tall" if tall else "season"
        span = " span-2" if tall else ""
        return "".join(
            f"<div class='card{span}'><div id='c-{k}' class='{cls}'></div></div>" for k, _ in keys
        )

    blob = json.dumps(payload, separators=(",", ":"))
    labels = {b: LABELS.get(b, b) for b in buckets}
    labels["dep_total"] = "All deposits"
    labels["wd_total"] = "All withdrawals"
    lab = json.dumps(labels, separators=(",", ":"))
    return f"""
    <h2 class="section">Deposits and withdrawals</h2>
    <p class="sub" id="season-note"></p>
    <div class="toggles" id="season-toggle">
      <button type="button" data-kind="flow">Flow</button>
      <button type="button" data-kind="stock" class="on">Stock</button>
      <button type="button" data-cal="fy" class="on">FY</button>
      <button type="button" data-cal="cy">CY</button>
      <button type="button" data-hist="3" class="on">3y</button>
      <button type="button" data-hist="5">5y</button>
      <button type="button" data-hist="10">10y</button>
      <button type="button" data-hist="0">All</button>
      <label class="chk"><input type="checkbox" id="dts-gdp"> % of GDP</label>
      <label class="chk"><input type="checkbox" id="dts-int" checked> Include interest</label>
    </div>
    <div class="grid-2">{cards([("dep_total", "")], tall=True)}{cards([(b, "") for b in dep])}</div>
    <h2 class="section">Withdrawals</h2>
    <div class="grid-2">{cards([("wd_total", "")], tall=True)}{cards([(b, "") for b in wd])}</div>
    <script>
    const DTS = {blob};
    const DTS_LABEL = {lab};
    const LAY = {json.dumps(theme.plotly_layout())};
    const dtsState = {{kind: "stock", cal: "fy", gdp: false, interest: true, hist: 3}};
    function dtsNote() {{
      const flow = dtsState.kind === "flow";
      const fy = dtsState.cal === "fy";
      const unit = dtsState.gdp
        ? " Shown as a share of nominal GDP (" + DTS.gdp + ")."
        : " Shown in $bn.";
      const base = flow
        ? (fy ? "Flow: the daily print, one line per fiscal year."
              : "Flow: the daily print, one line per calendar year.")
        : (fy ? "Stock: sum of daily prints since 1 October. The flow is the change in this line."
              : "Stock: sum of daily prints since 1 January. The flow is the change in this line.");
      const span = dtsState.hist ? " Last " + dtsState.hist + " years, plus the forecast." : " Full history, plus the forecast.";
      const intr = dtsState.interest
        ? " All withdrawals include interest on Treasury securities."
        : " All withdrawals exclude interest on Treasury securities.";
      document.getElementById("season-note").textContent = base + span
        + " The latest forecast year is blue; earlier years walk the MATLAB order backward. The vertical rule is the last DTS print."
        + " After it the line is the same colour, broken (4px on, 2px off) over a lighter continuous stroke, through "
        + DTS.forecastEnd + "."
        + unit + intr;
    }}
    function dtsRgba(hex, a) {{
      const h = hex.replace("#", "");
      const n = parseInt(h, 16);
      return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
    }}
    function dtsSplit(xs, ys, cut, current) {{
      if (!current) return [{{x: xs, y: ys, dash: "solid"}}];
      const a = {{x: [], y: [], dash: "solid"}}, b = {{x: [], y: [], dash: "dash"}};
      xs.forEach((x, i) => {{
        const t = x <= cut ? a : b;
        t.x.push(x); t.y.push(ys[i]);
      }});
      return [a, b].filter(t => t.x.length);
    }}
    function dtsLayout(el, ylab, cal, ticks, title, cut) {{
      const lay = JSON.parse(JSON.stringify(LAY));
      lay.title = {{text: title, font: {{size: 13, family: LAY.font.family, color: LAY.font.color}}, x: 0, xanchor: "left"}};
      lay.margin = {{l: 64, r: 16, t: 42, b: 78}};
      lay.height = el.classList.contains("tall") ? 540 : 460;
      lay.hovermode = "x unified";
      lay.legend = Object.assign({{}}, LAY.legend, {{orientation: "h", y: -0.22, x: 0}});
      lay.xaxis = Object.assign({{}}, LAY.xaxis, {{
        title: {{text: cal === "fy" ? "Fiscal year" : "Calendar year", font: LAY.xaxis.title.font}},
        tickmode: "array", tickvals: ticks[0], ticktext: ticks[1]
      }});
      lay.yaxis = Object.assign({{}}, LAY.yaxis, {{title: {{text: ylab, font: LAY.yaxis.title.font}}}});
      lay.shapes = [{{type: "line", xref: "x", yref: "paper", x0: cut, x1: cut, y0: 0, y1: 1,
        line: {{color: "#000000", width: 1.25}}}}];
      lay.annotations = [{{x: cut, y: 1, yref: "paper", text: "forecast", showarrow: false,
        xanchor: "left", font: {{size: 10, color: "#000000"}}}}];
      return lay;
    }}
    function dtsDraw() {{
      dtsNote();
      const cal = dtsState.cal, kind = dtsState.kind;
      const ticks = DTS.ticks[cal];
      const cut = DTS.histEnd[cal];
      const cur = String(DTS.current[cal]);
      Object.keys(DTS.views).forEach(key => {{
        const el = document.getElementById("c-" + key);
        if (!el) return;
        const src = (key === "wd_total" && !dtsState.interest) ? "wd_total_xint" : key;
        const book = dtsState.gdp ? DTS.pct : DTS.views;
        const block = book[src][cal][kind];
        const years = Object.keys(block).sort();
        const traces = [];
        const hover = dtsState.gdp ? "%{{y:.2f}}% of GDP" : "%{{y:.1f}}bn";
        years.forEach(y => {{
          const n = dtsState.hist;
          if (n && Number(y) <= Number(cur) && Number(y) < Number(cur) - n + 1) return;
          const color = DTS.colors[y] || "#0072BD";
          const forecastYear = Number(y) > Number(cur);
          const parts = forecastYear
            ? [{{x: block[y][0], y: block[y][1], dash: "dash"}}]
            : dtsSplit(block[y][0], block[y][1], cut, y === cur);
          parts.forEach((p, i) => {{
            const forecast = p.dash === "dash";
            const width = (y === cur && !forecast) ? 2.4 : 1.6;
            const name = forecastYear ? y + " forecast" : (i ? y + " forecast" : y);
            if (forecast) {{
              traces.push({{
                x: p.x, y: p.y, mode: "lines", name: name, legendgroup: y, showlegend: false,
                line: {{color: dtsRgba(color, 0.55), width: width}}, hoverinfo: "skip"
              }});
            }}
            traces.push({{
              x: p.x, y: p.y, mode: "lines", name: name,
              legendgroup: y, showlegend: i === 0,
              line: {{color: color, width: width, dash: forecast ? "{FORECAST_DASH}" : "solid"}},
              hovertemplate: y + " · " + hover + "<extra></extra>"
            }});
          }});
        }});
        const ylab = dtsState.gdp
          ? (kind === "flow" ? "Daily flow, % of GDP" : (cal === "fy" ? "FYTD stock, % of GDP" : "YTD stock, % of GDP"))
          : (kind === "flow" ? "Daily flow, $bn" : (cal === "fy" ? "FYTD stock, $bn" : "YTD stock, $bn"));
        Plotly.react(el, traces, dtsLayout(el, ylab, cal, ticks, DTS_LABEL[key] || key, cut),
          {{displayModeBar: false, responsive: true}});
      }});
    }}
    document.getElementById("season-toggle").addEventListener("click", ev => {{
      const b = ev.target.closest("button");
      if (!b) return;
      if (b.dataset.kind) dtsState.kind = b.dataset.kind;
      if (b.dataset.cal) dtsState.cal = b.dataset.cal;
      if (b.dataset.hist !== undefined) dtsState.hist = Number(b.dataset.hist);
      document.querySelectorAll("#season-toggle button").forEach(x => {{
        const on = (x.dataset.kind && x.dataset.kind === dtsState.kind)
          || (x.dataset.cal && x.dataset.cal === dtsState.cal)
          || (x.dataset.hist !== undefined && Number(x.dataset.hist) === dtsState.hist);
        x.classList.toggle("on", on);
      }});
      dtsDraw();
    }});
    document.getElementById("dts-gdp").addEventListener("change", ev => {{ dtsState.gdp = ev.target.checked; dtsDraw(); }});
    document.getElementById("dts-int").addEventListener("change", ev => {{ dtsState.interest = ev.target.checked; dtsDraw(); }});
    dtsDraw();
    </script>
    """


def sections_html() -> tuple[str, str]:
    """Body fragment + CSS for the financing bridge, bucket table and seasonals.

    Returns ``("", "")`` when derived inputs are missing (e.g. ``--skip-liquidity``).
    """
    needed = (
        "dts_bucket_daily.parquet",
        "tga_forward_buckets.parquet",
        "tga_forward_path.parquet",
        "tga_forward_summary.json",
    )
    if any(not (DERIVED / n).exists() for n in needed):
        return "", ""

    panel = _read("dts_bucket_daily.parquet")
    fc = _read("tga_forward_buckets.parquet")
    path = _read("tga_forward_path.parquet")
    summ = _read("tga_forward_summary.json")
    score = _read("flow_bucket_score.json")
    ch = _read("flow_model_choice.json")
    if panel is None or fc is None or path is None or summ is None:
        return "", ""
    choice = ch.get("choice") if ch else None
    import dts_flows

    buckets = [b for b in dts_flows.OP_BUCKETS if b in fc.columns]
    if not buckets:
        return "", ""
    panel = panel[buckets]
    fc = fc[buckets]
    t = pd.Timestamp(summ["target_date"])
    f = fc.loc[:t]
    dep = float(f[[b for b in buckets if b.startswith("dep_")]].sum().sum())
    wd = float(f[[b for b in buckets if b.startswith("wd_")]].sum().sum())
    kp = "".join([
        _kpi("TGA", f"${summ['tga_origin_bn']:,.0f}bn", summ["origin"]),
        _kpi("Target", f"${summ['target_bn']:,.0f}bn", summ["target_date"]),
        _kpi("Receipts to target", f"${dep:,.0f}bn", f"{len(f)} business days"),
        _kpi("Outlays to target", f"${wd:,.0f}bn", f"interest ${float(f['wd_interest'].sum()):,.0f}bn"),
        _kpi("Net operating flow", f"{dep - wd:+,.0f}bn", "receipts − outlays"),
        _kpi("Implied net bills", f"{summ['implied_net_bills_to_target_bn']:+,.0f}bn",
             f"private {summ.get('private_net_bills_to_target_bn', float('nan')):+,.0f}bn"),
    ])
    payload = season_payload(panel, fc, buckets)
    origin = pd.Timestamp(summ["origin"])
    figs = [
        fig_bridge(path, fc, summ),
        fig_cumulative(path, fc, summ["tga_origin_bn"]),
        fig_bridge_weekly(path, origin),
    ]
    cards = [f"<div class='card'>{_html(g, first=(i == 0))}</div>" for i, g in enumerate(figs)]
    season = _season_html(payload, buckets)
    body = f"""
    <div id="components" class="wrap-components">
    <h2 class="section">Table II components — bridge and bucket forecasts</h2>
    <p class="sub">Each Table II bucket is forecast as its trailing 250-business-day level × a gradient-boosted
      calendar shape (tax deadlines, 1st/3rd/15th pay dates under both weekend conventions, Social Security
      Wednesdays moved off holidays, biweekly federal payroll on salaries), refit monthly. Spike-heavy buckets
      (Medicare, VA, OPM, HHS) pin quiet days to the trailing median; SSA splits Benefits from SSI. A bucket uses
      its flat level where the calendar model lost on PRE (2022–23). Interest is the coupon schedule × trailing
      realised ratio. Financing: auction ladder as announced, Fed rolls and purchases from SOMA holdings by CUSIP,
      buybacks and nonmarketables at trailing pace; bills steered to the refunding target.</p>
    <div class="kpi-row">{kp}</div>
    <h2 class="section">Financing bridge to the target</h2>
    <div class="grid">{cards[0]}{cards[1]}</div>
    <h2 class="section">Bridge components over time</h2>
    <p class="sub">Weekly sums of the same bridge lines. History is Table II operating flow, the auction ladder
      (coupons / announced bills), and the financing reconcile (Fed term, buybacks, nonmarketables). Implied bills
      are zero until the steered path starts. Forecast is the unchanged fiscal-path scenario (pale broken stroke);
      the black vertical is the last DTS print.</p>
    <div class="grid">{cards[2]}</div>
    <h2 class="section">By bucket ($bn)</h2>
    <p class="sub">Forecast to {summ['target_date']} against the same window 52 weeks earlier and the trailing level
      run-rate. Daily MAE ($bn) from monthly origins, model vs flat level; PRE = 2022–23, OOS = 2024→.</p>
    {bucket_table(panel, fc, summ, score, choice)}
    {season}
    </div>
    """
    return body, SECTIONS_CSS + TABLE_CSS


def build() -> Path:
    import datetime as dt

    from build_dashboard import PAGE_CSS

    body, css = sections_html()
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not body:
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<title>US TGA components</title><style>{PAGE_CSS}</style></head><body>
<header><h1>US TGA components — flow forecasts by bucket</h1>
<div class="sub">{built} · Derived inputs missing; run the liquidity stack.</div></header>
</body></html>"""
    else:
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>US TGA components — flow forecasts by bucket</title>
  <style>{PAGE_CSS}{css}</style>
</head>
<body>
  <header>
    <h1>US TGA components — flow forecasts by bucket</h1>
    <div class="sub">{built} · Table II deposits and withdrawals, calendar shape × trailing level
      (customs = effective tariff rate × goods imports). Financing bridge steers bills to the
      refunding TGA target.</div>
  </header>
  {body}
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()
