"""MATLAB-style year-overlay season charts shared by country cash pages.

Same interaction model as the UK cash tab / US TGA components: stock vs flow,
calendar vs fiscal year (where defined), history window, optional % of GDP.
Forecast months after the last observed print are dashed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_FISCAL = REPO / "us" / "fiscal"
import sys

if str(_FISCAL) not in sys.path:
    sys.path.insert(0, str(_FISCAL))
from build_dashboard import MATLAB  # noqa: E402

DERIVED = Path(__file__).resolve().parent / "derived"

# Align with US cash path (through FY2030) and debt ``Y_END``.
HORIZON_END = pd.Timestamp("2030-12-01")

SEASON_CSS = """
.toggles { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; align-items: center; }
.toggles button { font: 600 11px/1 var(--mono); padding: 7px 12px; border: 1px solid var(--line);
  background: var(--panel-hi); color: var(--text); cursor: pointer; }
.toggles button.on { background: var(--text); color: var(--panel-hi); }
.toggles .chk { font: 600 11px/1 var(--mono); display: flex; align-items: center; gap: 6px; }
.season { height: 420px; } .season.tall { height: 500px; }
"""


def forward_months(origin: pd.Timestamp | None = None) -> int:
    """Months from the month after ``origin`` through ``HORIZON_END`` (min 12)."""
    o = pd.Timestamp(origin) if origin is not None else pd.Timestamp.today()
    o = (o + pd.offsets.MonthBegin(1)).normalize()
    if o > HORIZON_END:
        return 12
    return max(12, (HORIZON_END.year - o.year) * 12 + (HORIZON_END.month - o.month) + 1)


def _fy_of(month: pd.Timestamp, fy_start_month: int) -> int:
    m = pd.Timestamp(month)
    return int(m.year) if m.month >= fy_start_month else int(m.year) - 1


def _axis_x(month: pd.Timestamp, cal: str, fy_start_month: int) -> str:
    m = pd.Timestamp(month)
    if cal == "fy" and m.month < fy_start_month:
        return f"2001-{m.month:02d}-01"
    return f"2000-{m.month:02d}-01"


def _gdp_by_month(cc: str, months: pd.DatetimeIndex) -> pd.Series:
    path = DERIVED / "debt_country_annual.parquet"
    if not path.exists():
        return pd.Series(np.nan, index=months)
    ann = pd.read_parquet(path)
    sub = ann.loc[ann["cc"].astype(str).eq(cc), ["year", "gdp_bn"]].dropna()
    level = sub.set_index("year")["gdp_bn"].astype(float)
    out = pd.Series([float(level.get(int(m.year), np.nan)) for m in months], index=months)
    if out.notna().any():
        out = out.ffill().bfill()
    return out


def _year_book(frame: pd.DataFrame, col: str, level: str | None, cal: str,
               gdp: pd.Series, fy_start_month: int) -> dict:
    book = {"bn": {"flow": {}, "stock": {}}, "pct": {"flow": {}, "stock": {}}}
    years = frame["month"].map(lambda m: _fy_of(m, fy_start_month) if cal == "fy" else int(m.year))
    work = frame.assign(_y=years)
    for year, part in work.groupby("_y"):
        part = part.sort_values("month")
        xs = [_axis_x(pd.Timestamp(m), cal, fy_start_month) for m in part["month"]]

        def _clean(s: pd.Series) -> list:
            return [None if not np.isfinite(v) else float(v) for v in s.astype(float)]

        flow = _clean(part[col])
        stock = _clean(part[level] if level else part[col].astype(float).cumsum())
        g = []
        for m in part["month"]:
            v = gdp.reindex([pd.Timestamp(m)]).iloc[0] if len(gdp) else np.nan
            g.append(float(v) if pd.notna(v) else float("nan"))

        def _share(vals: list) -> list:
            return [None if v is None or not np.isfinite(d) or d == 0 else 100.0 * v / d
                    for v, d in zip(vals, g)]

        book["bn"]["flow"][str(int(year))] = [xs, flow]
        book["bn"]["stock"][str(int(year))] = [xs, stock]
        book["pct"]["flow"][str(int(year))] = [xs, _share(flow)]
        book["pct"]["stock"][str(int(year))] = [xs, _share(stock)]
    return book


def season_payload(df: pd.DataFrame, *, cc: str, specs: list[tuple[str, str, str, str | None]],
                   fy_start_month: int = 1, currency: str = "€",
                   forecast_note: str = "Forecast months follow the live cash model.") -> dict:
    """``specs`` = (key, label, flow_col, level_col_or_None)."""
    d = df.sort_values("month").copy()
    if "month" not in d.columns:
        d = d.rename_axis("month").reset_index()
    d = d.drop_duplicates(subset=["month"], keep="last")
    months = pd.DatetimeIndex(pd.to_datetime(d["month"]))
    gdp = _gdp_by_month(cc, months)
    gdp = pd.Series(gdp.to_numpy(), index=months)
    obs = (~d["is_forecast"].astype(bool)) if "is_forecast" in d.columns else pd.Series(True, index=d.index)
    last = pd.Timestamp(d.loc[obs.to_numpy(), "month"].max()) if obs.any() else pd.Timestamp(months.max())
    views = {key: {cal: _year_book(d, col, lvl, cal, gdp, fy_start_month) for cal in ("fy", "cy")}
             for key, _, col, lvl in specs}
    years = sorted({_fy_of(m, fy_start_month) for m in months} | {int(m.year) for m in months})
    latest = max(years) if years else int(last.year)
    fy_ticks = (
        [[f"2000-{fy_start_month:02d}-01",
          f"2000-{(fy_start_month + 2 - 1) % 12 + 1:02d}-01",
          f"2000-{(fy_start_month + 5 - 1) % 12 + 1:02d}-01",
          f"2000-{(fy_start_month + 8 - 1) % 12 + 1:02d}-01"],
         ["start", "+3m", "+6m", "+9m"]]
        if fy_start_month != 1 else
        [["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"], ["Jan", "Apr", "Jul", "Oct"]]
    )
    return {
        "views": views,
        "order": [k for k, *_ in specs],
        "labels": {k: lab for k, lab, *_ in specs},
        "colors": {str(y): MATLAB[(latest - y) % len(MATLAB)] for y in years},
        "histEnd": {
            "fy": _axis_x(last, "fy", fy_start_month),
            "cy": _axis_x(last, "cy", fy_start_month),
        },
        "current": {"fy": _fy_of(last, fy_start_month), "cy": int(last.year)},
        "ticks": {
            "fy": fy_ticks,
            "cy": [["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"],
                   ["Jan", "Apr", "Jul", "Oct"]],
        },
        "gdp": "calendar-year nominal GDP, held flat through the year",
        "currency": currency,
        "note": forecast_note,
        "fyStart": fy_start_month,
    }


def season_html(df: pd.DataFrame, *, cc: str, prefix: str,
                specs: list[tuple[str, str, str, str | None]],
                fy_start_month: int = 1, currency: str = "€",
                tall_key: str | None = None,
                forecast_note: str = "Forecast months follow the live cash model.") -> str:
    tall_key = tall_key or (specs[0][0] if specs else "")
    payload = json.dumps(
        season_payload(df, cc=cc, specs=specs, fy_start_month=fy_start_month,
                       currency=currency, forecast_note=forecast_note),
        separators=(",", ":"))
    cards = "".join(
        f"<div class='card{' span-2' if k == tall_key else ''}'>"
        f"<div id='{prefix}-{k}' class='season{' tall' if k == tall_key else ''}'></div></div>"
        for k, *_ in specs
    )
    fy_label = "FY" if fy_start_month != 1 else "CY"
    return f"""
    <div class="toggles" id="{prefix}-toggle">
      <button type="button" data-kind="stock" class="on">Stock</button>
      <button type="button" data-kind="flow">Flow</button>
      <button type="button" data-cal="fy" class="on">{fy_label}</button>
      <button type="button" data-cal="cy">CY</button>
      <button type="button" data-hist="3">3y</button>
      <button type="button" data-hist="5" class="on">5y</button>
      <button type="button" data-hist="10">10y</button>
      <button type="button" data-hist="0">All</button>
      <label class="chk"><input type="checkbox" id="{prefix}-gdp"> % of GDP</label>
    </div>
    <p class="sub" id="{prefix}-note"></p>
    <div class="grid-2">{cards}</div>
    <script>
    const {prefix.upper()}S = {payload};
    const {prefix}State = {{kind: "stock", cal: "fy", gdp: false, hist: 5}};
    function {prefix}Note() {{
      const S = {prefix.upper()}S, st = {prefix}State;
      const flow = st.kind === "flow";
      const fy = st.cal === "fy";
      const base = flow
        ? "Flow: the month's print, one line per year."
        : (fy ? "Stock: sum within the year. Levels use the published stock when available."
              : "Stock: sum since January.");
      const span = st.hist ? " Last " + st.hist + " years, plus the forecast." : " Full history, plus the forecast.";
      const unit = st.gdp ? " Share of " + S.gdp + "." : " " + S.currency + "bn.";
      document.getElementById("{prefix}-note").textContent = base + span
        + " The latest year is blue. The vertical rule is the last published month; after it the line is broken."
        + unit + " " + S.note;
    }}
    function {prefix}Layout(title, ylab, cal, cut) {{
      return {{
        title: {{text: title, font: {{size: 13}}, x: 0, xanchor: "left"}},
        margin: {{l: 58, r: 16, t: 36, b: 64}},
        height: title === {prefix.upper()}S.labels[{json.dumps(tall_key)}] ? 500 : 420,
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        hovermode: "x unified",
        legend: {{orientation: "h", y: -0.22, x: 0, font: {{size: 10}}}},
        xaxis: {{tickmode: "array", tickvals: {prefix.upper()}S.ticks[cal][0],
                 ticktext: {prefix.upper()}S.ticks[cal][1], showgrid: false,
                 title: {{text: cal === "fy" ? "Year" : "Calendar year"}}}},
        yaxis: {{title: {{text: ylab}}, zeroline: true, gridcolor: "#e6e6e6"}},
        shapes: [{{type: "line", xref: "x", yref: "paper", x0: cut, x1: cut, y0: 0, y1: 1,
                   line: {{color: "#000", width: 1.25}}}}],
        annotations: [{{x: cut, y: 1, yref: "paper", text: "forecast", showarrow: false,
                        xanchor: "left", font: {{size: 10, color: "#000"}}}}]
      }};
    }}
    function {prefix}Split(xs, ys, cut, current) {{
      if (!current) return [{{x: xs, y: ys, dash: "solid"}}];
      const a = {{x: [], y: [], dash: "solid"}}, b = {{x: [], y: [], dash: "dash"}};
      xs.forEach((x, i) => {{ (x <= cut ? a : b).x.push(x); (x <= cut ? a : b).y.push(ys[i]); }});
      if (a.x.length && b.x.length) {{ b.x.unshift(a.x[a.x.length - 1]); b.y.unshift(a.y[a.y.length - 1]); }}
      return [a, b].filter(t => t.x.length);
    }}
    function {prefix}Draw() {{
      {prefix}Note();
      const S = {prefix.upper()}S, st = {prefix}State;
      const cal = st.cal, kind = st.kind, cut = S.histEnd[cal], cur = String(S.current[cal]);
      const ylab = st.gdp
        ? (kind === "flow" ? "Month, % of GDP" : "Within year, % of GDP")
        : (kind === "flow" ? "Month, " + S.currency + "bn" : "Within year, " + S.currency + "bn");
      S.order.forEach(key => {{
        const el = document.getElementById("{prefix}-" + key);
        const block = (st.gdp ? S.views[key][cal].pct : S.views[key][cal].bn)[kind];
        const years = Object.keys(block).sort();
        const traces = [];
        years.forEach(y => {{
          const n = st.hist;
          if (n && Number(y) < Number(cur) - n + 1) return;
          const color = S.colors[y] || "#0072BD";
          const future = Number(y) > Number(cur);
          const parts = future ? [{{x: block[y][0], y: block[y][1], dash: "dash"}}]
                               : {prefix}Split(block[y][0], block[y][1], cut, y === cur);
          const name = String(y);
          parts.forEach((p, i) => {{
            traces.push({{
              x: p.x, y: p.y, mode: "lines", name: i ? name + " forecast" : name,
              legendgroup: y, showlegend: i === 0,
              line: {{color: color, width: y === cur && p.dash === "solid" ? 2.4 : 1.5,
                      dash: p.dash === "dash" ? "dash" : "solid"}},
              hovertemplate: name + " · %{{y:.1f}}<extra></extra>"
            }});
          }});
        }});
        Plotly.react(el, traces, {prefix}Layout(S.labels[key], ylab, cal, cut),
                     {{displayModeBar: false, responsive: true}});
      }});
    }}
    document.getElementById("{prefix}-toggle").addEventListener("click", ev => {{
      const b = ev.target.closest("button");
      if (!b) return;
      if (b.dataset.kind) {prefix}State.kind = b.dataset.kind;
      if (b.dataset.cal) {prefix}State.cal = b.dataset.cal;
      if (b.dataset.hist !== undefined) {prefix}State.hist = Number(b.dataset.hist);
      document.querySelectorAll("#{prefix}-toggle button").forEach(x => {{
        x.classList.toggle("on", (x.dataset.kind && x.dataset.kind === {prefix}State.kind)
          || (x.dataset.cal && x.dataset.cal === {prefix}State.cal)
          || (x.dataset.hist !== undefined && Number(x.dataset.hist) === {prefix}State.hist));
      }});
      {prefix}Draw();
    }});
    document.getElementById("{prefix}-gdp").addEventListener("change", ev => {{
      {prefix}State.gdp = ev.target.checked; {prefix}Draw();
    }});
    {prefix}Draw();
    </script>
    """


def default_specs(cash_col: str = "cash_requirement_bn",
                  cash_label: str = "Cash requirement") -> list[tuple[str, str, str, str | None]]:
    return [
        ("cash", cash_label, cash_col, None),
        ("bonds", "Net bonds", "net_bonds_bn", None),
        ("bills", "Net bills", "net_bills_bn", None),
        ("other", "Other financing", "other_fin_bn", None),
        ("buffer", "Buffer change", "buffer_change_bn", None),
    ]


def bottomup_card(summary: dict) -> str:
    """Compact ship / method card from summary forward keys."""
    fwd = summary.get("forward") or summary
    keys = (
        ("bottomup_ship", "bottom_up_ship", "cgncr_bottomup_ship", "bottomup_macro_ship"),
        ("bottomup_reason", "bottom_up_reason", "bottomup_macro_reason"),
        ("bottomup_pre_mean", "bottomup_macro_pre_mean"),
        ("bottomup_oos_mean", "bottomup_macro_oos_mean"),
    )
    ship = next((fwd.get(k) for k in keys[0] if fwd.get(k) is not None), None)
    reason = next((fwd.get(k) for k in keys[1] if fwd.get(k)), None)
    pre = next((fwd.get(k) for k in keys[2] if fwd.get(k) is not None), None)
    oos = next((fwd.get(k) for k in keys[3] if fwd.get(k) is not None), None)
    if ship is None and reason is None:
        return ""
    status = "SHIP — live cash path" if ship else "research only — production path unchanged"
    bits = [f"<b>{status}</b>"]
    if reason:
        bits.append(str(reason))
    # Scores stay off the charts; card text is the internal readout only.
    if pre is not None:
        bits.append(f"PRE mean MAE {float(pre):.2f}")
    if oos is not None:
        bits.append(f"OOS mean MAE {float(oos):.2f}")
    return ("<div class='card'><div class='sub'>Bottom-up cash model</div>"
            f"<p style='font:12px/1.45 var(--mono);margin:6px 0'>{' · '.join(bits)}</p></div>")


def category_tape_html(panel: pd.DataFrame | None, *, cc: str, prefix: str,
                       specs: list[tuple[str, str, str, str | None]],
                       currency: str = "€",
                       note: str = "Category flows under the shipped bottom-up rules.") -> str:
    """MATLAB year-overlay block for category leaves (no MAE on the chart)."""
    if panel is None or panel.empty or not specs:
        return ""
    have = {c for c in panel.columns}
    specs = [(k, lab, col, lvl) for k, lab, col, lvl in specs if col in have]
    if not specs:
        return ""
    return (
        f"<h2 class='section'>Category tape</h2>"
        f"<p class='sub'>{note}</p>"
        + season_html(panel, cc=cc, prefix=f"{prefix}cat", specs=specs,
                      fy_start_month=1, currency=currency, tall_key=specs[0][0],
                      forecast_note=note)
    )


def _panel_hist_fc(hist: pd.DataFrame, fc: pd.DataFrame,
                   start: str = "2016-01-01") -> pd.DataFrame:
    h = hist.loc[hist.index >= pd.Timestamp(start)].copy()
    # Forecast months win when a sparse history column trails past the
    # complete-row origin (SE lending/interest vs tax/spend).
    if len(fc):
        h = h.loc[~h.index.isin(fc.index)]
    h["is_forecast"] = False
    f = fc.copy()
    f["is_forecast"] = True
    out = pd.concat([h, f]).sort_index()
    out.index.name = "month"
    return out.reset_index()


def uk_category_panel() -> tuple[pd.DataFrame | None, list[tuple[str, str, str, str | None]]]:
    from sovereign_cash.uk import bottom_up_macro as bu

    if not getattr(bu, "SHIP", False):
        return None, []
    comp = bu.components()
    origin = pd.Timestamp(comp.index.max())
    h = forward_months(origin)
    paths = bu.category_paths(comp, origin, h=h)
    keys = [
        ("it", "Income tax", "it"),
        ("vat", "VAT", "vat"),
        ("ct", "Corporation tax", "ct"),
        ("nics", "NICs", "nics"),
        ("welfare", "Welfare", "welfare"),
        ("interest", "Debt interest", "interest"),
        ("rdel", "RDEL proxy", "rdel_proxy"),
        ("cap", "Capital", "cap"),
    ]
    cols = [c for _, _, c in keys if c in paths and c in comp.columns]
    if not cols:
        return None, []
    fut = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fc = pd.DataFrame({c: paths[c] for c in cols}, index=fut)
    panel = _panel_hist_fc(comp[cols], fc)
    specs = [(k, lab, c, None) for k, lab, c in keys if c in cols]
    return panel, specs


def de_category_panel() -> tuple[pd.DataFrame | None, list[tuple[str, str, str, str | None]]]:
    from sovereign_cash.de import bottom_up_macro as bu

    flows = bu.monthly_flows()
    keys = [
        ("lohn", "Lohnsteuer", "lohnsteuer"),
        ("ust", "Umsatzsteuer", "ust"),
        ("kst", "Körperschaftsteuer", "kst"),
        ("zins", "Zinsausgaben", "zins"),
        ("pers", "Personalausgaben", "personal"),
        ("zuw", "Zuweisungen", "zuweisungen"),
    ]
    cols = [c for _, _, c in keys if c in flows and flows[c].notna().sum() >= 24]
    if not cols:
        return None, []
    hist = pd.DataFrame({c: flows[c] for c in cols}).dropna(how="all")
    origin = hist.index.max()
    h = forward_months(origin)
    fut = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    rows = {}
    for c in cols:
        s = hist[c].dropna()
        rows[c] = [float(s.get(m - pd.DateOffset(months=12), np.nan)) for m in fut]
    fc = pd.DataFrame(rows, index=fut)
    panel = _panel_hist_fc(hist[cols], fc)
    specs = [(k, lab, c, None) for k, lab, c in keys if c in cols]
    return panel, specs


def fr_category_panel() -> tuple[pd.DataFrame | None, list[tuple[str, str, str, str | None]]]:
    from sovereign_cash.fr import bottom_up as bu
    from sovereign_cash.yoy_components import components

    hist = components("fr")
    cols_map = [
        ("rec", "Recettes", "recettes"),
        ("pri", "Primary spend", "primary"),
        ("int", "Charge d'intérêts", "interest"),
        ("spe", "Comptes spéciaux", "speciaux"),
    ]
    have = [c for _, _, c in cols_map if c in hist.columns]
    if len(have) < 2:
        return None, []
    origin = hist.dropna(how="all").index.max()
    h = forward_months(origin)
    fc = bu.forecast(origin, h, frame=hist)
    if fc is None:
        return None, []
    panel = _panel_hist_fc(hist[have], fc[have])
    specs = [(k, lab, c, None) for k, lab, c in cols_map if c in have]
    return panel, specs


def it_category_panel() -> tuple[pd.DataFrame | None, list[tuple[str, str, str, str | None]]]:
    from sovereign_cash.it import bottom_up_macro as bu

    if not getattr(bu, "SHIP", False):
        return None, []
    frame = bu._frame_bn()
    origin = frame.index.max()
    h = forward_months(origin)
    # Component smly needs a realised year-ago print; score path uses h≤12. Extend the
    # first year by chaining same-calendar-month values out to HORIZON_END.
    base = bu.forecast_components(origin, min(h, 12), frame=frame)
    if base is None or base.empty:
        return None, []
    keys = [
        ("tax", "Tax revenue", "tax_rev"),
        ("oth", "Other revenue", "other_rev"),
        ("pri", "Primary spend", "primary"),
        ("int", "Interest", "interest"),
        ("cap", "Capital", "cap_exp"),
        ("tre", "Treasury balance", "treasury_bal"),
    ]
    cols = [c for _, _, c in keys if c in frame.columns and c in base.columns]
    if not cols:
        return None, []
    fut = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    chain = pd.concat([frame[cols], base[cols]]).sort_index()
    rows = {c: [] for c in cols}
    for m in fut:
        if m in base.index:
            for c in cols:
                rows[c].append(float(base.at[m, c]))
        else:
            prev = m - pd.DateOffset(months=12)
            for c in cols:
                v = float(chain.at[prev, c]) if prev in chain.index else float("nan")
                rows[c].append(v)
            chain.loc[m] = [rows[c][-1] for c in cols]
    fc = pd.DataFrame(rows, index=fut)
    panel = _panel_hist_fc(frame[cols], fc)
    specs = [(k, lab, c, None) for k, lab, c in keys if c in cols]
    return panel, specs


def se_category_panel() -> tuple[pd.DataFrame | None, list[tuple[str, str, str, str | None]]]:
    """SCB/CH1089 NBR categories — explanatory tape (macro BU not shipped under RG pin)."""
    from sovereign_cash.se import bottom_up_macro as bu
    from sovereign_cash.se import build as SB

    d = SB.load()
    cats = bu.categories(d)
    keys = [
        ("tax", "Tax revenue", "tax_rev"),
        ("pri", "Primary spend", "primary_spend"),
        ("lend", "Net lending", "net_lending"),
        ("int", "Interest", "interest"),
    ]
    cols = [c for _, _, c in keys if c in cats and cats[c].notna().sum() >= 24]
    if not cols:
        return None, []
    hist = pd.DataFrame({c: cats[c] for c in cols}).dropna(how="all")
    origin = hist.dropna(how="any").index.max()
    h = forward_months(origin)
    drivers = bu.macro_drivers()
    debt_ye = bu.debt_stock_ye(d)
    auth = bu.load_authority()
    raw = bu.forecast_nbr(cats, drivers, debt_ye, origin, hmax=h, auth=auth)
    # Split forward NBR into category paths via the same rules used in forecast_nbr.
    tax = cats["tax_rev"].dropna()
    spend = cats["primary_spend"].dropna()
    use_scb = origin in tax.index and origin in spend.index and len(tax) >= 24
    fut = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    if use_scb:
        g_tax = bu._growth(bu._annual(tax.loc[:origin]), drivers, origin, ("ngdp",))
        if "wages" in drivers:
            g_w = bu._growth(bu._annual(tax.loc[:origin]), drivers, origin, ("wages",))
            g_tax = 0.5 * g_tax + 0.5 * g_w
        g_sp = bu._growth(bu._annual(spend.loc[:origin]), drivers, origin, ("cpi",))
        tax_fc = bu._authority_scale(tax.loc[:origin], origin, h, "tax_rev_bn", auth)
        if tax_fc is None:
            tax_fc = bu._smly_grown(tax.loc[:origin], origin, h, g_tax)
        sp_fc = bu._smly_grown(spend.loc[:origin], origin, h, g_sp)
        lend_fc = bu._lending_fc(cats["net_lending"].loc[:origin], origin, h, auth)
        inte_fc = bu._interest_monthly(origin, h, cats["interest"].loc[:origin],
                                       bu._eff_rate(bu._annual(cats["interest"]), debt_ye),
                                       debt_ye, drivers, auth)
        fc = pd.DataFrame({
            "tax_rev": tax_fc.reindex(fut),
            "primary_spend": sp_fc.reindex(fut),
            "net_lending": lend_fc.reindex(fut),
            "interest": inte_fc.reindex(fut),
        })
    else:
        fc = pd.DataFrame({
            c: [float(hist[c].get(m - pd.DateOffset(months=12), np.nan)) for m in fut]
            for c in cols
        }, index=fut)
    panel = _panel_hist_fc(hist[cols], fc[cols])
    specs = [(k, lab, c, None) for k, lab, c in keys if c in cols]
    return panel, specs


def no_category_panel() -> tuple[pd.DataFrame | None, list[tuple[str, str, str, str | None]]]:
    """Tab4 lines under the shipped beyond-NB macro rules."""
    from sovereign_cash.no import bottom_up_macro as bu
    from sovereign_cash.no import build as NB
    from sovereign_cash.no import fetch

    fl = NB.daily_flows(fetch.read("liquidity_statistics"))
    m = bu.monthly_tab4(fl)
    drivers = bu.macro_drivers()
    rules = bu.select_use_macro(m, drivers)
    keys = [
        ("vat", "VAT", "vat"),
        ("tax", "Tax net", "tax_net"),
        ("oil", "Oil tax", "oil_tax"),
        ("nav", "NAV", "nav"),
        ("hlth", "Health", "health"),
        ("loc", "Local authorities", "local_authorities"),
    ]
    cols = [c for _, _, c in keys if c in m.columns and m[c].notna().sum() >= 24]
    if not cols:
        return None, []
    hist = m[cols].dropna(how="all")
    origin = hist.dropna(how="any").index.max()
    h = forward_months(origin)
    fut = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    rows = {c: [] for c in cols}
    for month in fut:
        for c in cols:
            v = bu.forecast_component(m, drivers, c, month, origin, rules.get(c, "smly"))
            rows[c].append(float(v) if pd.notna(v) else float("nan"))
    fc = pd.DataFrame(rows, index=fut)
    panel = _panel_hist_fc(hist, fc)
    specs = [(k, lab, c, None) for k, lab, c in keys if c in cols]
    return panel, specs
