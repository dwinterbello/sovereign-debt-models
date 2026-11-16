"""Autumn Budget comparison — three charts + macro strip.

1. Additional borrowing: bills vs gilts (street vs Budget once filled)
2. Full remit composition incl. T-bills (before / street / Budget)
3. PSNB path (March OBR / street / Budget)
Plus key OBR macros (AWE wages, GDP, CPI, Bank Rate, gilt rates).

Data: ``sovereign_cash/data/uk/budget_compare.json`` → ``uk_budget_compare.html``.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "sovereign_cash" / "data" / "uk" / "budget_compare.json"
WEO = REPO / "sovereign_cash" / "data" / "debt" / "weo.parquet"
OUT = REPO / "sovereign_cash" / "uk_budget_compare.html"

sys.path.insert(0, str(REPO))
from portal import theme  # noqa: E402

theme.use_plotly()
BLUE, ORANGE, GREEN, PURPLE, GREY, RED = (
    theme.SERIES[0], theme.SERIES[1], theme.SERIES[4],
    theme.SERIES[3], theme.TOKENS["chart-zero"], theme.SERIES[6],
)
CYAN = theme.SERIES[5]

PAGE_CSS = theme.page_css() + """
body { margin: 0; color: var(--text); font: 12px/1.45 var(--sans); font-variant-numeric: tabular-nums; }
header { padding: 10px 16px; }
h1 { margin: 0; }
.sub { color: var(--muted); font-size: 11px; margin-top: 4px; max-width: 1080px; line-height: 1.45; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 8px 16px 36px; }
h2.section { font: 700 11px/1.3 var(--mono); margin: 20px 0 6px; color: var(--text);
  text-transform: uppercase; letter-spacing: 0.12em;
  border-bottom: 1px solid var(--grid); padding-bottom: 3px; }
.banner { border: 1px solid var(--line); background: var(--panel-hi); padding: 7px 12px; margin: 8px 0 10px;
  font: 600 12px/1.4 var(--mono); border-left: 4px solid """ + ORANGE + """; }
.banner.ok { border-left-color: """ + GREEN + """; }
.muted { color: var(--muted); font-size: 11px; margin: 2px 0 8px; max-width: 1080px; }
.card { margin: 0 0 8px; overflow: hidden; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
table.t { border-collapse: collapse; font: 11px/1.4 var(--mono); margin: 4px 0 8px; width: 100%; }
table.t th, table.t td { border-bottom: 1px solid var(--grid); padding: 3px 8px; text-align: right; }
table.t th:first-child, table.t td:first-child { text-align: left; }
table.t th { background: var(--th-bg); font-weight: 600; }
table.t td.na { color: var(--muted); }
table.t td.pos { color: """ + BLUE + """; }
table.t td.neg { color: """ + RED + """; }
"""

_PLOT = {"displayModeBar": False, "responsive": True}

SECTORS = [
    ("short", "Short", BLUE),
    ("medium", "Medium", CYAN),
    ("long", "Long", ORANGE),
    ("linker", "Index-linked", PURPLE),
    ("unallocated", "Unallocated", GREY),
    ("bills", "T-bills (debt-fin)", GREEN),
]

MACRO = [
    ("awe_pct", "Average weekly earnings %", 1),
    ("determinants_real_gdp", "Real GDP %", 1),
    ("cpi_pct", "CPI %", 1),
    ("bank_rate_pct", "Bank Rate %", 1),
    ("gilt_rate_pct", "Gilt rate %", 1),
    ("productivity_pct", "Trend productivity / hour %", 1),
]


def load() -> dict:
    return json.loads(DATA.read_text())


def imf_deficit_pct(years: list[str] | None = None) -> dict[str, float]:
    """IMF WEO general-government deficit, % GDP (=-balance). Calendar years.

    Not the same perimeter as OBR PSNB (FY, public sector), but the board's
    usual external check — ``sovereign_cash/data/debt/weo.parquet``.
    """
    import pandas as pd

    if not WEO.exists():
        return {}
    w = pd.read_parquet(WEO)
    uk = w[(w["country"].astype(str) == "uk") & (w["var"].astype(str) == "balance_pct")]
    out = {str(int(y)): round(-float(v), 2) for y, v in zip(uk["year"], uk["value"])}
    if years is not None:
        out = {y: out[y] for y in years if y in out}
    return out


def after_ready(d: dict) -> bool:
    a = d.get("after") or {}
    if a.get("status") == "pending":
        return False
    return (a.get("financing") or {}).get("gilt_sales") is not None


def _num(v, dp: int = 1) -> str:
    return "—" if v is None else f"{v:,.{dp}f}"


def _gbp(v, dp: int = 1, plus: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:,.{dp}f}"
    if plus and v > 0:
        s = "+" + s
    return "£" + s


def _style(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    # Titles live in the HTML <h2>; plotly only needs room for the legend.
    top = 56 if not title else 72
    lay = theme.plotly_layout(
        height=height, margin=dict(l=52, r=24, t=top, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    if title:
        lay["title"] = dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                            font=dict(size=12))
    else:
        lay["title"] = None
    fig.update_layout(**lay)
    return fig


def _div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config=_PLOT)


def _cols(d: dict) -> list[str]:
    """Column labels: always Street + Budget; Budget may be placeholder."""
    return ["Street", "Budget" if after_ready(d) else "Budget (pending)"]


def additional_borrowing(d: dict) -> go.Figure:
    """Δ financing vs Apr remit: bills vs gilts. Street now; Budget when filled."""
    st = d["street"]["financing_delta_vs_before"]
    af = d["after"]["financing"]
    bf = d["before"]["financing"]
    ready = after_ready(d)
    cols = _cols(d)

    street_bills = st.get("tbill_net_debtfin")
    street_gilts = st.get("gilt_sales")
    budget_bills = (None if not ready or af.get("tbill_net_debtfin") is None
                    else af["tbill_net_debtfin"] - bf["tbill_net_debtfin"])
    budget_gilts = (None if not ready or af.get("gilt_sales") is None
                    else af["gilt_sales"] - bf["gilt_sales"])

    fig = go.Figure()
    fig.add_bar(
        name="Δ T-bills", x=cols,
        y=[street_bills, budget_bills],
        marker_color=GREEN,
        text=[_gbp(street_bills, plus=True), _gbp(budget_bills, plus=True)],
        textposition="inside", insidetextanchor="middle",
    )
    fig.add_bar(
        name="Δ gilts", x=cols,
        y=[street_gilts, budget_gilts],
        marker_color=BLUE,
        text=[_gbp(street_gilts, plus=True), _gbp(budget_gilts, plus=True)],
        textposition="inside", insidetextanchor="middle",
    )
    if not ready:
        fig.add_annotation(
            x="Budget (pending)", y=0, text="awaiting remit",
            showarrow=False, yshift=28, font=dict(size=11, color=GREY),
        )
    fig.update_layout(barmode="stack", yaxis_title="£bn vs Apr remit")
    return _style(fig, "", height=360)


def _composition_series(d: dict, which: str) -> dict[str, float | None]:
    """Sector totals incl. T-bill debt-fin for before / street / after."""
    if which == "before":
        c = d["before"]["composition"]
        out = {k: c[k]["total"] for k, *_ in SECTORS if k != "bills"}
        out["bills"] = d["before"]["financing"]["tbill_net_debtfin"]
        return out
    if which == "street":
        c = d["street"]["composition_total"]
        out = {k: c.get(k) for k, *_ in SECTORS if k != "bills"}
        out["bills"] = d["street"]["financing"].get("tbill_net_debtfin")
        return out
    c = d["after"]["composition"]
    out = {k: (c.get(k) or {}).get("total") for k, *_ in SECTORS if k != "bills"}
    out["bills"] = d["after"]["financing"].get("tbill_net_debtfin")
    return out


def remit_composition(d: dict) -> go.Figure:
    """Full remit stack: gilt buckets + T-bills. Before / Street / Budget."""
    ready = after_ready(d)
    cols = ["Before (Apr)", "Street", "Budget" if ready else "Budget (pending)"]
    series = {
        "before": _composition_series(d, "before"),
        "street": _composition_series(d, "street"),
        "after": _composition_series(d, "after"),
    }
    fig = go.Figure()
    for key, name, color in SECTORS:
        ys = [series["before"][key], series["street"][key], series["after"][key]]
        fig.add_bar(
            name=name, x=cols, y=ys, marker_color=color,
            text=[_num(v) if v is not None else "" for v in ys],
            textposition="inside", insidetextanchor="middle",
        )
    if not ready:
        fig.add_annotation(
            x="Budget (pending)", y=0, text="awaiting remit",
            showarrow=False, yshift=28, font=dict(size=11, color=GREY),
        )
    fig.update_layout(barmode="stack", yaxis_title="£bn")
    return _style(fig, "", height=380)


def _fy_labs(years: list[str]) -> list[str]:
    return [f"FY{y}-{str(int(y) + 1)[2:]}" for y in years]


def _path(d: dict, var: str, title: str, unit: str, height: int = 340,
          street: bool = True) -> go.Figure:
    """March EFO vs optional street vs Budget (pending placeholder)."""
    years = ["2026", "2027", "2028", "2029", "2030"]
    labs = _fy_labs(years)
    b = d["before"]["obr"].get(var) or {}
    s = (d["street"].get("obr") or {}).get(var) or {}
    a = d["after"]["obr"].get(var) or {}
    ready = after_ready(d)
    fig = go.Figure()
    fig.add_scatter(
        x=labs, y=[b.get(y) for y in years], name="March EFO",
        mode="lines+markers", line=dict(color=GREY, width=2), marker=dict(size=7),
    )
    if street and any(s.get(y) is not None for y in years):
        fig.add_scatter(
            x=labs, y=[s.get(y) for y in years], name="Street median",
            mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dot"),
            marker=dict(size=7),
        )
    if ready and any(a.get(y) is not None for y in years):
        fig.add_scatter(
            x=labs, y=[a.get(y) for y in years], name="Budget EFO",
            mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=7),
        )
    else:
        fig.add_scatter(
            x=labs, y=[None] * len(labs), name="Budget EFO (pending)",
            mode="lines+markers", line=dict(color=BLUE, width=2, dash="dash"),
            marker=dict(size=7),
        )
    fig.update_layout(yaxis_title=unit)
    return _style(fig, title, height=height)


def _obr_gdp(d: dict, which: str) -> dict[str, float]:
    """OBR nominal GDP £bn (Table A.3 denominator) by FY key."""
    return (d.get(which) or {}).get("obr", {}).get("nominal_gdp_bn") or {}


def _psnb_pct(d: dict, which: str) -> dict[str, float | None]:
    """Published OBR PSNB % GDP, else PSNB £bn / OBR nominal GDP."""
    obr = (d.get(which) or {}).get("obr") or {}
    pct = obr.get("deficit_pct") or {}
    if pct:
        return {y: pct.get(y) for y in ["2025", "2026", "2027", "2028", "2029", "2030"]}
    bn = obr.get("psnb_bn") or {}
    gdp = obr.get("nominal_gdp_bn") or {}
    out = {}
    for y, v in bn.items():
        g = gdp.get(y)
        out[y] = None if v is None or not g else round(100.0 * float(v) / float(g), 2)
    return out


def _street_psnb_pct(d: dict) -> dict[str, float | None]:
    """Street PSNB £bn / Budget (else March) OBR nominal GDP → % GDP."""
    bn = (d.get("street") or {}).get("obr", {}).get("psnb_bn") or {}
    gdp = _obr_gdp(d, "after") or _obr_gdp(d, "before")
    out = {}
    for y, v in bn.items():
        g = gdp.get(y)
        out[y] = None if v is None or not g else round(100.0 * float(v) / float(g), 2)
    return out


def psnb_path(d: dict) -> go.Figure:
    """PSNB as % of OBR GDP — March / street / Budget."""
    years = ["2026", "2027", "2028", "2029", "2030"]
    labs = _fy_labs(years)
    b = _psnb_pct(d, "before")
    a = _psnb_pct(d, "after")
    s = _street_psnb_pct(d)
    ready = after_ready(d)
    fig = go.Figure()
    fig.add_scatter(
        x=labs, y=[b.get(y) for y in years], name="March EFO",
        mode="lines+markers", line=dict(color=GREY, width=2), marker=dict(size=7),
    )
    if any(s.get(y) is not None for y in years):
        fig.add_scatter(
            x=labs, y=[s.get(y) for y in years], name="Street median",
            mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dot"),
            marker=dict(size=7),
        )
    if ready and any(a.get(y) is not None for y in years):
        fig.add_scatter(
            x=labs, y=[a.get(y) for y in years], name="Budget EFO",
            mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=7),
        )
    else:
        fig.add_scatter(
            x=labs, y=[None] * len(labs), name="Budget EFO (pending)",
            mode="lines+markers", line=dict(color=BLUE, width=2, dash="dash"),
            marker=dict(size=7),
        )
    fig.update_layout(yaxis_title="% of OBR GDP")
    return _style(fig, "", height=360)


def cgncr_path(d: dict) -> go.Figure:
    """CGNCR ex NRAM/B&B/NR £bn — feeds the financing arithmetic."""
    return _path(d, "cgncr_ex_nr_bn", "", "£bn", height=360)


def financing_arithmetic(d: dict) -> go.Figure:
    """FY26-27 CGNCR / GFR / NFR / gilts / bills — Apr vs street vs Budget.

    Source: Budget 2026 Annex A Table A.1 (and Apr remit / MNI street).
    """
    ready = after_ready(d)
    bf = d["before"]["financing"]
    st = d["street"]["financing"]
    af = d["after"]["financing"]
    cols = ["Apr remit", "Street", "Budget" if ready else "Budget (pending)"]
    metrics = [
        ("cgncr_ex", "CGNCR ex NR", BLUE),
        ("gross_financing_requirement", "GFR", ORANGE),
        ("nfr", "NFR (DMO)", PURPLE),
        ("gilt_sales", "Gilt sales", CYAN),
        ("tbill_net_debtfin", "T-bills", GREEN),
    ]
    # Street has no NFR in financing — derive gilts+bills
    street_nfr = None
    if st.get("gilt_sales") is not None and st.get("tbill_net_debtfin") is not None:
        street_nfr = st["gilt_sales"] + st["tbill_net_debtfin"]

    fig = go.Figure()
    for key, name, color in metrics:
        if key == "nfr":
            ys = [bf.get("nfr"), street_nfr,
                  af.get("nfr") if ready else None]
        else:
            ys = [bf.get(key), st.get(key),
                  af.get(key) if ready else None]
        fig.add_bar(
            name=name, x=cols, y=ys, marker_color=color,
            text=[_num(v) if v is not None else "" for v in ys],
            textposition="outside",
        )
    fig.update_layout(barmode="group", yaxis_title="£bn")
    return _style(fig, "", height=400)


def financing_delta_arithmetic(d: dict) -> go.Figure:
    """Δ vs Apr remit for CGNCR / GFR / NFR / gilts / bills."""
    ready = after_ready(d)
    bf = d["before"]["financing"]
    st = d["street"]["financing"]
    af = d["after"]["financing"]
    st_d = d["street"]["financing_delta_vs_before"]
    cols = ["Street", "Budget" if ready else "Budget (pending)"]

    def _d(fin, key):
        if fin.get(key) is None or bf.get(key) is None:
            return None
        return round(fin[key] - bf[key], 1)

    street_nfr_d = None
    if st.get("gilt_sales") is not None and st.get("tbill_net_debtfin") is not None:
        street_nfr_d = round(
            (st["gilt_sales"] + st["tbill_net_debtfin"]) - bf["nfr"], 1
        )
    budget_nfr_d = _d(af, "nfr") if ready else None

    series = [
        ("CGNCR ex NR", [st_d.get("cgncr_ex"), _d(af, "cgncr_ex") if ready else None], BLUE),
        ("GFR", [st_d.get("gross_financing_requirement"),
                 _d(af, "gross_financing_requirement") if ready else None], ORANGE),
        ("NFR", [street_nfr_d, budget_nfr_d], PURPLE),
        ("Gilt sales", [st_d.get("gilt_sales"), _d(af, "gilt_sales") if ready else None], CYAN),
        ("T-bills", [st_d.get("tbill_net_debtfin"),
                     _d(af, "tbill_net_debtfin") if ready else None], GREEN),
    ]
    fig = go.Figure()
    for name, ys, color in series:
        fig.add_bar(
            name=name, x=cols, y=ys, marker_color=color,
            text=[_gbp(v, plus=True) if v is not None else "" for v in ys],
            textposition="outside",
        )
    fig.update_layout(barmode="group", yaxis_title="£bn vs Apr remit")
    return _style(fig, "", height=380)


def deficit_pct_path(d: dict) -> go.Figure:
    """OBR PSNB % GDP vs IMF WEO GG deficit % GDP; Budget placeholder."""
    years = ["2026", "2027", "2028", "2029", "2030"]
    labs = _fy_labs(years)
    b = d["before"]["obr"].get("deficit_pct") or {}
    a = d["after"]["obr"].get("deficit_pct") or {}
    imf = imf_deficit_pct(years)
    ready = after_ready(d)
    fig = go.Figure()
    fig.add_scatter(
        x=labs, y=[b.get(y) for y in years], name="OBR March (PSNB % GDP)",
        mode="lines+markers", line=dict(color=GREY, width=2), marker=dict(size=7),
    )
    if imf:
        fig.add_scatter(
            x=labs, y=[imf.get(y) for y in years], name="IMF WEO (GG deficit % GDP)",
            mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dot"),
            marker=dict(size=7),
        )
    if ready and any(a.get(y) is not None for y in years):
        fig.add_scatter(
            x=labs, y=[a.get(y) for y in years], name="OBR Budget (PSNB % GDP)",
            mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=7),
        )
    else:
        fig.add_scatter(
            x=labs, y=[None] * len(labs), name="OBR Budget (pending)",
            mode="lines+markers", line=dict(color=BLUE, width=2, dash="dash"),
            marker=dict(size=7),
        )
    fig.update_layout(yaxis_title="% of GDP")
    return _style(fig, "", height=360)


def macro_panel(d: dict) -> go.Figure:
    """Single 3×2 panel — March vs Budget macros, one screenshot."""
    specs = [
        ("awe_pct", "AWE %"),
        ("determinants_real_gdp", "Real GDP %"),
        ("cpi_pct", "CPI %"),
        ("bank_rate_pct", "Bank Rate %"),
        ("gilt_rate_pct", "Gilt rate %"),
        ("productivity_pct", "Trend productivity / hour %"),
    ]
    specs = [s for s in specs if d["before"]["obr"].get(s[0])]
    n = len(specs)
    cols = 2
    rows = max(1, (n + cols - 1) // cols)
    years = ["2026", "2027", "2028", "2029", "2030"]
    labs = _fy_labs(years)
    ready = after_ready(d)

    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=False,
        subplot_titles=[t for _, t in specs],
        vertical_spacing=0.11, horizontal_spacing=0.08,
    )
    for i, (key, _title) in enumerate(specs):
        r, c = divmod(i, cols)
        row, col = r + 1, c + 1
        b = d["before"]["obr"].get(key) or {}
        a = d["after"]["obr"].get(key) or {}
        show = i == 0
        fig.add_trace(
            go.Scatter(
                x=labs, y=[b.get(y) for y in years], name="March EFO",
                mode="lines+markers",
                line=dict(color=GREY, width=2), marker=dict(size=6),
                legendgroup="march", showlegend=show,
            ),
            row=row, col=col,
        )
        if ready and any(a.get(y) is not None for y in years):
            fig.add_trace(
                go.Scatter(
                    x=labs, y=[a.get(y) for y in years], name="Budget EFO",
                    mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=6),
                    legendgroup="budget", showlegend=show,
                ),
                row=row, col=col,
            )
        fig.update_yaxes(title_text="%", row=row, col=col, title_standoff=4)

    # Place each subplot title just right of that cell's y-axis (not centred / clipped).
    title_set = {t for _, t in specs}
    for i, ann in enumerate(fig.layout.annotations or []):
        if ann.text not in title_set:
            continue
        r, c = divmod(i, cols)
        # xaxis key: xaxis, xaxis2, … in row-major order matching subplot index.
        axis = fig.layout[f"xaxis{'' if i == 0 else i + 1}"]
        x0 = float(axis.domain[0])
        ann.update(
            x=x0 + 0.035, xanchor="left", xref="paper",
            font=dict(size=11, color=theme.TOKENS.get("chart-text", "#222")),
        )

    lay = theme.plotly_layout(
        height=190 * rows + 56,
        margin=dict(l=48, r=20, t=52, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.08, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    lay["title"] = None
    fig.update_layout(**lay)
    return fig


def macro_plots(d: dict) -> str:
    note = ("March vs Budget EFO (A.1 calendar GDP/CPI/AWE/productivity; A.3 FY Bank Rate / gilt). "
            "Productivity = trend output per hour. Street has no macro path.")
    return (f'<div class="card">{_div(macro_panel(d))}</div>'
            f'<div class="muted">{note}</div>')


def macro_table(d: dict) -> str:
    years = ["2026", "2027", "2028", "2029", "2030"]
    ready = after_ready(d)
    head = ("<table class='t'><tr><th>OBR determinant</th>"
            + "".join(f"<th colspan='{'2' if ready else '1'}'>FY{y}-{str(int(y)+1)[2:]}</th>" for y in years)
            + "</tr><tr><th></th>")
    for _ in years:
        head += "<th>March</th>"
        if ready:
            head += "<th>Budget</th>"
    head += "</tr>"
    rows = []
    for key, lab, dp in MACRO:
        b = d["before"]["obr"].get(key) or {}
        a = d["after"]["obr"].get(key) or {}
        if not b:
            continue
        cells = [f"<td>{lab}</td>"]
        for y in years:
            cells.append(f"<td>{_num(b.get(y), dp)}</td>")
            if ready:
                cells.append(f"<td class='{'na' if a.get(y) is None else ''}'>{_num(a.get(y), dp)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return head + "".join(rows) + "</table>"


def build(d: dict | None = None) -> Path:
    d = d or load()
    ready = after_ready(d)
    b, street, desk = d["before"], d["street"], d.get("desk") or {}
    if ready:
        banner = (f'<div class="banner ok">Budget vintage in · '
                  f'{d["after"].get("obr_vintage") or d["after"].get("remit_event")} · '
                  f'{d["after"].get("remit_date") or d["after"].get("obr_date")}</div>')
    else:
        banner = (f'<div class="banner">Budget column pending ({d["event_date"]}). '
                  f'Street = MNI median {street["asof"]}; Before = Apr remit / March EFO. '
                  f'Fill <code>after</code> in <code>budget_compare.json</code> and re-run.</div>')

    bill_d = street["financing_delta_vs_before"]["tbill_net_debtfin"]
    gilt_d = street["financing_delta_vs_before"]["gilt_sales"]
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>UK Budget — {d['event']}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{PAGE_CSS}</style>
</head><body>
<header>
  <h1>UK Budget — remit &amp; OBR</h1>
  <div class="sub">
    Before = {b['obr_vintage']} ({b['obr_date']}) + DMO remit {b['remit_date']}.
    Street = {street['source']}.
    Desk = {desk.get('source', '')}.
    Built {dt.datetime.now():%Y-%m-%d %H:%M}.
  </div>
</header>
<div class="wrap">
{banner}

<div class="muted">Street additional borrowing +£{bill_d + gilt_d:.1f}bn
(= bills {_gbp(bill_d, plus=True)} / gilts {_gbp(gilt_d, plus=True)});
desk £{desk.get('financing_lift_bn', {}).get('low', '?')}–{desk.get('financing_lift_bn', {}).get('high', '?')}bn primarily bill-funded.
Post-measures headroom ~{_gbp(street.get('headroom_bn'), 0)}
(desk pre-measures £{desk.get('headroom_pre_measures_bn', {}).get('low', '?')}–{desk.get('headroom_pre_measures_bn', {}).get('high', '?')}bn).</div>

<h2 class="section">1 · Financing arithmetic (FY2026-27)</h2>
<div class="muted">Budget Annex A Table A.1. CGNCR ex NR £{d['after']['financing']['cgncr_ex'] if ready else '—'}bn
(+£{(d['after']['financing']['cgncr_ex'] - b['financing']['cgncr_ex']) if ready else 0:.1f} vs Apr £{b['financing']['cgncr_ex']:.1f});
GFR £{d['after']['financing']['gross_financing_requirement'] if ready else '—'}bn
(+£{(d['after']['financing']['gross_financing_requirement'] - b['financing']['gross_financing_requirement']) if ready else 0:.1f} vs Apr £{b['financing']['gross_financing_requirement']:.1f});
NFR £{d['after']['financing']['nfr'] if ready else '—'}bn (+£{(d['after']['financing']['nfr'] - b['financing']['nfr']) if ready else 0:.1f}).
Soft vs PSNB (+£17.7bn) — inflation on IL gilts hits PSNB now, CGNCR later. Street CGNCR/GFR £{street['financing']['cgncr_ex']:.1f} / £{street['financing']['gross_financing_requirement']:.1f}.</div>
<div class="grid-2">
  <div class="card"><h2 class="section">Levels — CGNCR / GFR / NFR</h2>{_div(financing_arithmetic(d))}</div>
  <div class="card"><h2 class="section">Δ vs Apr remit</h2>{_div(financing_delta_arithmetic(d))}</div>
</div>
<div class="card"><h2 class="section">Δ bills vs gilts only</h2>{_div(additional_borrowing(d))}</div>

<h2 class="section">2 · Remit composition — gilts + T-bills</h2>
<div class="muted">Apr gilt buckets from Annex A; T-bills = debt-financing net.
Street gilt buckets = MNI median; unallocated residual to gilt total £{street['financing']['gilt_sales']:.1f}bn.</div>
<div class="card">{_div(remit_composition(d))}</div>

<h2 class="section">3 · Deficit &amp; cash requirement</h2>
<div class="muted">{street.get('obr_note', '')}
 OBR PSNB % GDP = Table A.9. CGNCR ex NR = Table 6.6 (financing arithmetic).
 Street CGNCR only scored for FY26–27 / FY27–28 in MNI.
 IMF WEO from <code>data/debt/weo.parquet</code> (GG overall balance, calendar year — not OBR PSNB FY perimeter).</div>
<div class="grid-2">
  <div class="card"><h2 class="section">PSNB % of OBR GDP</h2>{_div(psnb_path(d))}</div>
  <div class="card"><h2 class="section">CGNCR ex NR, £bn</h2>{_div(cgncr_path(d))}</div>
</div>
<div class="card"><h2 class="section">vs IMF WEO</h2>{_div(deficit_pct_path(d))}</div>

<h2 class="section">Key OBR macros (wage risk)</h2>
{macro_plots(d)}
{macro_table(d)}

<div class="muted" style="margin-top:18px">{d.get('note', '')}</div>
</div>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    return OUT


def main() -> None:
    path = build()
    d = load()
    print(f"wrote {path}  after={'ready' if after_ready(d) else 'pending'}")


if __name__ == "__main__":
    main()
