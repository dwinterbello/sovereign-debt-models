"""Interactive HTML: daily TGA-implied deficit as % of interpolated GDP."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
OUT = HERE / "tga_deficit_dashboard.html"

sys.path.append(str(HERE.parents[1]))
from portal import theme  # noqa: E402

theme.use_mpl()

_T = theme.TOKENS
MATLAB = theme.SERIES
BLUE = MATLAB[0]
ORANGE = MATLAB[1]
GREEN = MATLAB[4]
RED = MATLAB[6]
GREY = _T["chart-zero"]
PURPLE = MATLAB[3]
BESSENT = pd.Timestamp("2025-01-28")
YELLEN_BB = pd.Timestamp("2024-05-29")
DETAIL_START = pd.Timestamp("2022-01-01")
QRA_DATE = pd.Timestamp("2026-11-04")
TBAC_TGA_BN = 850.0
TBAC_TGA_SEP_BN = 950.0
YELLEN_TGA_BN = 550.0
CURRENT_LIQ_CAP_BN = 2.0
LONG_END_OPS_PER_YEAR = 33.0  # 10s–20s + 20s–30s combined
FY_YEARS = (2022, 2023, 2024, 2025, 2026)
YEAR_COLORS = {
    2022: MATLAB[5],
    2023: MATLAB[0],
    2024: MATLAB[3],
    2025: MATLAB[1],
    2026: MATLAB[6],
}

PAGE_CSS = theme.page_css() + """
body { margin: 0; color: var(--text); font: 12px/1.45 var(--sans); font-variant-numeric: tabular-nums; }
header { padding: 10px 16px; }
h1 { margin: 0; }
.sub { color: var(--muted); font-size: 11px; margin-top: 4px; max-width: 1080px; line-height: 1.45; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 10px 16px 36px; }
h2.section { font: 700 11px/1.3 var(--mono); margin: 22px 0 8px; color: var(--text);
  text-transform: uppercase; letter-spacing: 0.12em;
  border-bottom: 1px solid var(--grid); padding-bottom: 3px; }
.grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.grid-2 .span-2 { grid-column: 1 / -1; }
@media (max-width: 980px) { .grid-2 { grid-template-columns: 1fr; } }
.card { padding: 8px 10px; }
.kpi-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 6px; }
.kpi { border: 1px solid var(--line); border-top: 3px solid var(--accent); padding: 7px 10px; min-width: 148px; background: var(--panel-hi); }
.kpi .lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.45px; font-weight: 600; }
.kpi .val { font: 600 16px/1.2 var(--mono); margin-top: 2px; }
.kpi .note { font-size: 10px; color: var(--muted); margin-top: 2px; }
code { font-size: 10px; }
"""

_PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "tga_implied_deficit"},
}


def _style(
    fig: go.Figure,
    *,
    title: str,
    height: int = 420,
    ylab: str = "",
    rmargin: int = 16,
    rangeslider: bool = False,
) -> go.Figure:
    fig.update_layout(
        template=theme.use_plotly(default=False),
        title=dict(
            text=title,
            font=dict(size=12, color=_T["text"], family=theme.PLOT_FONT),
            x=0,
            xanchor="left",
            pad=dict(t=2, b=8),
            **(dict(y=0.985, yanchor="top") if rangeslider else {}),
        ),
        paper_bgcolor=_T["panel-hi"],
        plot_bgcolor=_T["plot-bg"],
        font=dict(family=theme.PLOT_FONT, color=_T["chart-text"], size=11),
        height=height,
        margin=dict(l=58, r=rmargin, t=78 if rangeslider else 44, b=40 if rangeslider else 62),
        legend=dict(
            orientation="h",
            # a bottom legend collides with the range slider
            **(dict(y=1.02, yanchor="bottom") if rangeslider else dict(y=-0.16)),
            x=0,
            font=dict(size=10, color=_T["chart-legend"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        yaxis_title=ylab,
    )
    axis = dict(
        showgrid=True,
        gridcolor=_T["chart-grid"],
        gridwidth=1,
        zeroline=True,
        zerolinecolor=_T["chart-zero"],
        zerolinewidth=1,
        showline=True,
        mirror=True,
        linecolor=_T["chart-axis"],
        linewidth=1,
        tickcolor=_T["chart-axis"],
        ticks="outside",
        ticklen=3.5,
        tickwidth=0.8,
        tickfont=dict(size=10, color=_T["chart-text"], family=theme.PLOT_FONT),
        automargin=True,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if rangeslider:
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08, bgcolor=_T["panel"]))
    return fig


def _event_lines(fig: go.Figure) -> go.Figure:
    fig.add_vline(
        x=YELLEN_BB,
        line_width=0.8,
        line_dash="dot",
        line_color=GREY,
        annotation_text="Yellen buybacks",
        annotation_position="top left",
        annotation_font_size=9,
        annotation_font_color=GREY,
    )
    fig.add_vline(
        x=BESSENT,
        line_width=0.9,
        line_dash="dash",
        line_color=PURPLE,
        annotation_text="Bessent",
        annotation_position="top right",
        annotation_font_size=9,
        annotation_font_color=PURPLE,
    )
    return fig


def _html(fig: go.Figure, *, first: bool) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if first else False,
        full_html=False,
        config=_PLOT_CONFIG,
    )


def _kpi(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="kpi"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="note">{note}</div></div>'
    )


def _load() -> pd.DataFrame:
    path = DERIVED / "tga_implied_deficit_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python us/fiscal/run_refresh.py")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _load_diag() -> pd.DataFrame | None:
    path = DERIVED / "ust_liquidity_fiscal_daily.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _load_buybacks_m() -> pd.DataFrame | None:
    path = DERIVED / "buybacks_monthly.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["month"]).sort_values("month")


def _load_buybacks_ops() -> pd.DataFrame | None:
    path = DERIVED / "buybacks_operations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["operation_date"])


def _since(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    return df[df[col] >= DETAIL_START].copy()


def _fig_12m(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    d = _since(panel.dropna(subset=["implied_deficit_12m_pct_gdp"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["implied_deficit_12m_pct_gdp"],
            mode="lines",
            name="TGA-implied 12m (Δdebt − ΔTGA)",
            line=dict(color=BLUE, width=1.6),
        )
    )
    if forecast and "implied_deficit_1d_bn" in forecast and "implied_deficit_1d_bn" in panel.columns:
        daily = panel.dropna(subset=["implied_deficit_1d_bn"]).set_index("date")["implied_deficit_1d_bn"]
        roll = _forward_12m(daily, forecast["implied_deficit_1d_bn"])
        fut = (100.0 * roll / _gdp_held(panel, roll.index)).dropna()
        if len(fut):
            _add_dated_forecast(fig, fut.index, fut, name="TGA-implied 12m forecast", color=BLUE, width=1.6)
    if "dts_fiscal_12m_pct_gdp" in d.columns:
        f = d.dropna(subset=["dts_fiscal_12m_pct_gdp"])
        fig.add_trace(
            go.Scatter(
                x=f["date"],
                y=f["dts_fiscal_12m_pct_gdp"],
                mode="lines",
                name="DTS Table II cash deficit 12m",
                line=dict(color=ORANGE, width=1.2),
            )
        )
        if forecast and "dts_fiscal_1d_bn" in forecast:
            daily = panel.dropna(subset=["dts_fiscal_1d_bn"]).set_index("date")["dts_fiscal_1d_bn"]
            roll = _forward_12m(daily, forecast["dts_fiscal_1d_bn"])
            fut = (100.0 * roll / _gdp_held(panel, roll.index)).dropna()
            if len(fut):
                _add_dated_forecast(fig, fut.index, fut, name="DTS cash 12m forecast", color=ORANGE, width=1.6)
    if "mts_deficit_12m_pct_gdp" in d.columns:
        m = d.dropna(subset=["mts_deficit_12m_pct_gdp"])
        fig.add_trace(
            go.Scatter(
                x=m["date"],
                y=m["mts_deficit_12m_pct_gdp"],
                mode="lines",
                name="MTS official 12m",
                line=dict(color=GREY, width=1.1, dash="dot"),
            )
        )
    return _event_lines(
        _style(
            fig,
            title="12-month deficit / interpolated GDP (2022– ). Identity vs DTS cash vs MTS.",
            ylab="% of GDP",
            height=480,
            rangeslider=True,
        )
    )


def _fig_dts_split_bn(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    d = _since(panel)
    fig = go.Figure()
    specs = [
        ("dts_fiscal_12m_bn", "DTS cash deficit", BLUE, 1.6),
        ("dts_primary_12m_bn", "DTS primary (ex interest)", ORANGE, 1.5),
        ("dts_interest_12m_bn", "DTS interest on Treasuries", RED, 1.4),
    ]
    for col, name, color, width in specs:
        if col not in d.columns:
            continue
        m = d.dropna(subset=[col])
        fig.add_trace(
            go.Scatter(x=m["date"], y=m[col], mode="lines", name=name, line=dict(color=color, width=width))
        )
        daily_col = col.replace("_12m_bn", "_1d_bn")
        if forecast and daily_col in forecast and daily_col in panel.columns:
            roll = _forward_12m(panel.set_index("date")[daily_col], forecast[daily_col])
            if len(roll):
                _add_dated_forecast(fig, roll.index, roll, name=f"{name} forecast", color=color, width=width)
    return _event_lines(
        _style(
            fig,
            title="Twelve-month DTS flows ($bn)",
            ylab="$bn",
            height=480,
            rangeslider=True,
        )
    )


def _fig_dts_split(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    d = _since(panel)
    fig = go.Figure()
    specs = [
        ("dts_fiscal_12m_pct_gdp", "DTS cash deficit", BLUE, 1.6),
        ("dts_primary_12m_pct_gdp", "DTS primary (ex interest)", ORANGE, 1.5),
        ("dts_interest_12m_pct_gdp", "DTS interest on Treasuries", RED, 1.4),
    ]
    for col, name, color, width in specs:
        if col not in d.columns:
            continue
        m = d.dropna(subset=[col])
        fig.add_trace(
            go.Scatter(x=m["date"], y=m[col], mode="lines", name=name, line=dict(color=color, width=width))
        )
        daily_col = col.replace("_12m_pct_gdp", "_1d_bn")
        if forecast and daily_col in forecast and daily_col in panel.columns:
            roll = _forward_12m(panel.set_index("date")[daily_col], forecast[daily_col])
            fut = (100.0 * roll / _gdp_held(panel, roll.index)).dropna()
            if len(fut):
                _add_dated_forecast(fig, fut.index, fut, name=f"{name} forecast", color=color, width=width)
    return _event_lines(
        _style(
            fig,
            title="Twelve-month DTS flows / nominal GDP",
            ylab="% of GDP",
            height=480,
            rangeslider=True,
        )
    )


def _fig_3m(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    from tga_deficit import LOOKBACK_12M, LOOKBACK_3M

    d = _since(panel.dropna(subset=["implied_deficit_3m_ann_pct_gdp"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["implied_deficit_3m_ann_pct_gdp"],
            mode="lines",
            name="3m annualized",
            line=dict(color=ORANGE, width=0.7),
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["implied_deficit_3m_ann_pct_gdp_ma20"],
            mode="lines",
            name="3m ann. 20d MA",
            line=dict(color=RED, width=1.3),
        )
    )
    if (
        forecast
        and forecast.get("debt_public_bn") is not None
        and forecast.get("tga_bn") is not None
        and "debt_public_bn" in panel.columns
    ):
        debt_h = panel.dropna(subset=["debt_public_bn"]).set_index("date")["debt_public_bn"]
        tga_h = panel.dropna(subset=["tga_bn"]).set_index("date")["tga_bn"]
        d_debt = _forward_level_change(debt_h, forecast["debt_public_bn"], LOOKBACK_3M)
        d_tga = _forward_level_change(tga_h, forecast["tga_bn"], LOOKBACK_3M)
        common = d_debt.index.intersection(d_tga.index)
        if len(common):
            ann = LOOKBACK_12M / LOOKBACK_3M
            pct = 100.0 * (d_debt.reindex(common) - d_tga.reindex(common)) * ann / _gdp_held(panel, common)
            pct = pct.dropna()
            if len(pct):
                _add_dated_forecast(fig, pct.index, pct, name="3m ann. forecast", color=ORANGE, width=0.9)
                hist = panel.dropna(subset=["implied_deficit_3m_ann_pct_gdp"]).set_index("date")[
                    "implied_deficit_3m_ann_pct_gdp"
                ]
                comb = pd.concat([hist, pct[pct.index > hist.index.max()]]).sort_index()
                comb = comb[~comb.index.duplicated(keep="last")]
                ma = comb.rolling(20, min_periods=10).mean()
                ma_fut = ma[ma.index > hist.index.max()].dropna()
                if len(ma_fut):
                    _add_dated_forecast(fig, ma_fut.index, ma_fut, name="3m ann. 20d MA forecast", color=RED, width=1.3)
    return _event_lines(
        _style(fig, title="3-month annualized TGA-implied deficit / GDP — recent run-rate", ylab="% of GDP")
    )


def _fig_tga(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    d = _since(panel.dropna(subset=["tga_bn"]))
    fig = go.Figure()
    fig.add_hrect(y0=550, y1=600, fillcolor=GREY, opacity=0.18, line_width=0, annotation_text="Yellen ~$550–600bn")
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["tga_bn"],
            mode="lines",
            name="TGA close",
            line=dict(color=GREEN, width=1.4),
        )
    )
    if forecast and forecast.get("tga_bn") is not None and len(forecast["tga_bn"]):
        tga = forecast["tga_bn"]
        _add_dated_forecast(fig, tga.index, tga, name="Forecast", color=GREEN, width=1.6)
    return _event_lines(
        _style(fig, title="TGA close. Grey band = Yellen-era cash target (~week of needs). Bessent is ~$900–950bn.", ylab="$bn")
    )


def _fig_components(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    from tga_deficit import LOOKBACK_12M

    d = _since(panel.dropna(subset=["d_debt_12m_bn", "d_tga_12m_bn"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["d_debt_12m_bn"],
            mode="lines",
            name="12m Δ debt held by public",
            line=dict(color=BLUE, width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["d_tga_12m_bn"],
            mode="lines",
            name="12m Δ TGA",
            line=dict(color=ORANGE, width=1.2),
        )
    )
    if forecast and forecast.get("debt_public_bn") is not None and forecast.get("tga_bn") is not None:
        debt_h = panel.dropna(subset=["debt_public_bn"]).set_index("date")["debt_public_bn"]
        tga_h = panel.dropna(subset=["tga_bn"]).set_index("date")["tga_bn"]
        d_debt = _forward_level_change(debt_h, forecast["debt_public_bn"], LOOKBACK_12M)
        d_tga = _forward_level_change(tga_h, forecast["tga_bn"], LOOKBACK_12M)
        if len(d_debt):
            _add_dated_forecast(fig, d_debt.index, d_debt, name="12m Δ debt forecast", color=BLUE, width=1.2)
        if len(d_tga):
            _add_dated_forecast(fig, d_tga.index, d_tga, name="12m Δ TGA forecast", color=ORANGE, width=1.2)
    return _style(fig, title="Identity pieces (implied deficit = Δdebt − ΔTGA)", ylab="$bn")


def _fig_interest(diag: pd.DataFrame) -> go.Figure:
    d = _since(diag.dropna(subset=["nipa_interest_pct_gdp"], how="all").copy())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["nipa_interest_pct_gdp"],
            mode="lines",
            name="NIPA federal interest SAAR / GDP",
            line=dict(color=RED, width=1.4),
        )
    )
    if "interest_public_12m_pct_gdp" in d.columns:
        m = d.dropna(subset=["interest_public_12m_pct_gdp"])
        fig.add_trace(
            go.Scatter(
                x=m["date"],
                y=m["interest_public_12m_pct_gdp"],
                mode="lines",
                name="Fiscal Data public-issues 12m / GDP",
                line=dict(color=ORANGE, width=1.1),
            )
        )
    return _event_lines(_style(fig, title="Interest payments / interpolated nominal GDP", ylab="% of GDP"))


def _fig_avg_rate(diag: pd.DataFrame) -> go.Figure:
    d = diag.dropna(subset=["avg_mkt_rate"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["avg_mkt_rate"],
            mode="lines",
            name="Avg rate on marketable debt",
            line=dict(color=BLUE, width=1.3),
        )
    )
    return _event_lines(_style(fig, title="Average coupon on marketable Treasury debt", ylab="%"))


def _fig_repo(diag: pd.DataFrame) -> go.Figure:
    d = _since(diag.dropna(subset=["sofr_iorb_bp"], how="all"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["sofr_iorb_bp"],
            mode="lines",
            name="SOFR − IORB/IOER",
            line=dict(color=BLUE, width=1.4),
            yaxis="y",
        )
    )
    if "sofr_tgcr_bp" in d.columns:
        m = d.dropna(subset=["sofr_tgcr_bp"])
        fig.add_trace(
            go.Scatter(
                x=m["date"],
                y=m["sofr_tgcr_bp"],
                mode="lines",
                name="SOFR − TGCR (specials vs GC)",
                line=dict(color=ORANGE, width=1.3),
                yaxis="y2",
            )
        )
    fig = _style(
        fig,
        title="Overnight repo, last 5y. Left: SOFR vs administered floor. Right: specials vs GC.",
        ylab="SOFR − floor (bp)",
        height=440,
        rmargin=58,
    )
    fig.update_layout(
        yaxis2=dict(
            title="SOFR − TGCR (bp)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
            zerolinecolor=theme.rgba(ORANGE, 0.35),
            tickfont=dict(size=9, color=ORANGE),
            title_font=dict(size=10, color=ORANGE),
        )
    )
    return _event_lines(fig)


def _fig_otr(diag: pd.DataFrame) -> go.Figure:
    d = _since(diag.dropna(subset=["otr_richness_30_bp"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["otr_richness_30_bp"],
            mode="lines",
            name="GSW 30y par − CMT 30y (OTR-tilted)",
            line=dict(color=BLUE, width=1.2),
        )
    )
    a = diag.dropna(subset=["ust30_auction_vs_gsw_bp"])
    if not a.empty:
        fig.add_trace(
            go.Scatter(
                x=a["date"],
                y=a["ust30_auction_vs_gsw_bp"],
                mode="markers",
                name="30y auction stop-out − GSW par",
                marker=dict(color=RED, size=7),
            )
        )
    return _event_lines(
        _style(
            fig,
            title="30y OTR vs OFTR (bp). Line: OFTR spline − CMT (>0 = CMT/OTR rich). Dots: auction concession (>0 = cheap to spline)",
            ylab="bp",
            height=460,
        )
    )


def _fig_buybacks(bb: pd.DataFrame) -> go.Figure:
    d = bb[bb["month"] >= pd.Timestamp("2023-01-01")].copy()
    fig = go.Figure()
    colors = {
        "cm_front_bn": GREY,
        "liq_20_30_bn": RED,
        "liq_10_20_bn": ORANGE,
        "liq_other_bn": BLUE,
    }
    names = {
        "cm_front_bn": "Cash management (1Mo–2Y)",
        "liq_20_30_bn": "Liquidity support 20Y–30Y",
        "liq_10_20_bn": "Liquidity support 10Y–20Y",
        "liq_other_bn": "Liquidity support other buckets",
    }
    for col, color in colors.items():
        if col not in d.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=d["month"],
                y=d[col],
                name=names[col],
                marker_color=color,
            )
        )
    fig.update_layout(barmode="stack")
    return _event_lines(
        _style(fig, title="Treasury buybacks, par accepted ($bn). Labels are Treasury's operation_type", ylab="$bn")
    )


def _fig_rrp(diag: pd.DataFrame) -> go.Figure:
    d = _since(diag.dropna(subset=["rrp_bn"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["rrp_bn"],
            mode="lines",
            name="ON RRP take-up",
            line=dict(color=GREEN, width=1.2),
        )
    )
    return _event_lines(_style(fig, title="Fed overnight reverse repo take-up", ylab="$bn"))


def _value_at_doy(wide: pd.DataFrame, year: int, doy: int) -> float:
    """Last observation on or before FY day-of-year in `year` (weekend-safe)."""
    if year not in wide.columns:
        return float("nan")
    col = wide[year].dropna()
    col = col.loc[col.index <= doy]
    if col.empty:
        return float("nan")
    return float(col.iloc[-1])


def _fy_tax_days() -> list[tuple[int, str]]:
    """Corporate/individual settlement dates on a non-leap FY calendar."""
    from tga_deficit import fy_day_of_year

    out = []
    for month, day, lab in ((4, 15, "15 Apr"), (6, 15, "15 Jun"), (9, 15, "15 Sep")):
        out.append((fy_day_of_year(pd.Timestamp(2026, month, day)), lab))
    return out


def _mpl_box(ax) -> None:
    ax.grid(True, color=_T["chart-grid"], lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(_T["chart-axis"])
        sp.set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors=_T["chart-text"], labelsize=8)


def _fy_month_ticks() -> tuple[list[int], list[str]]:
    start = pd.Timestamp(2025, 10, 1)  # non-leap FY (FY2026)
    ticks, labels = [], []
    for year, month, lab in (
        (2025, 10, "Oct"),
        (2025, 11, "Nov"),
        (2025, 12, "Dec"),
        (2026, 1, "Jan"),
        (2026, 2, "Feb"),
        (2026, 3, "Mar"),
        (2026, 4, "Apr"),
        (2026, 5, "May"),
        (2026, 6, "Jun"),
        (2026, 7, "Jul"),
        (2026, 8, "Aug"),
        (2026, 9, "Sep"),
    ):
        ticks.append(int((pd.Timestamp(year, month, 1) - start).days + 1))
        labels.append(lab)
    return ticks, labels


# Latest forecast fiscal year is MATLAB blue. Earlier years walk the order backward.
FORECAST_COLOR_YEAR = 2030


def year_color(year: int) -> str:
    return MATLAB[(FORECAST_COLOR_YEAR - int(year)) % len(MATLAB)]


# Short gaps so a daily wiggle stays on the stroke. The solid underlay is the same hue.
FORECAST_DASH = "4px,2px"


def _forecast_stroke(fig: go.Figure, x, y, *, name: str, color: str, width: float, legendgroup: str, showlegend: bool) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, legendgroup=legendgroup, showlegend=False,
        line=dict(color=theme.rgba(color, 0.55), width=width), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, legendgroup=legendgroup, showlegend=showlegend,
        line=dict(color=color, width=width, dash=FORECAST_DASH),
    ))


def _fig_fy_overlay(
    series: pd.Series, *, title: str, ylab: str, already_fytd: bool = False, origin: pd.Timestamp | None = None,
) -> go.Figure:
    """One line per fiscal year. After ``origin`` the line is a broken stroke.

    The window is the last three fiscal years that contain a print, plus every later year
    in ``series`` (the forecast through FY2030)."""
    from tga_deficit import fy_day_of_year, fy_pivot, fytd_cumsum, us_fiscal_year

    ytd = fy_pivot(series) if already_fytd else fytd_cumsum(series)
    origin = series.dropna().index.max() if origin is None else pd.Timestamp(origin)
    current = int(us_fiscal_year(origin))
    cut = int(fy_day_of_year(origin))
    years = [int(y) for y in ytd.columns if int(y) >= current - 2]
    fig = go.Figure()
    has_forecast = False
    for year in years:
        if year not in ytd.columns:
            continue
        col = ytd[year].dropna()
        if col.empty:
            continue
        color = year_color(year)
        if year > current:
            parts = [(col, True)]
        elif year < current:
            parts = [(col, False)]
        else:
            parts = []
            hist, fut = col[col.index <= cut], col[col.index >= cut]
            if len(hist):
                parts.append((hist, False))
            if len(fut) > 1:
                parts.append((fut, True))
        for i, (sl, forecast) in enumerate(parts):
            width = 2.4 if year == current and not forecast else 1.6
            if forecast:
                has_forecast = True
                _forecast_stroke(
                    fig, sl.index, sl, name=f"FY{year}", color=color, width=width,
                    legendgroup=str(year), showlegend=(i == 0),
                )
            else:
                fig.add_trace(go.Scatter(
                    x=sl.index, y=sl, mode="lines", name=f"FY{year}",
                    legendgroup=str(year), showlegend=(i == 0),
                    line=dict(color=color, width=width),
                ))
    fig = _style(fig, title=title, ylab=ylab, height=460)
    ticks, labels = _fy_month_ticks()
    fig.update_xaxes(title="US fiscal year (Oct–Sep)", tickmode="array", tickvals=ticks, ticktext=labels)
    if has_forecast:
        fig.add_vline(
            x=cut, line_color="#000000", line_width=1.25,
            annotation_text="forecast", annotation_position="top right",
            annotation_font_size=10, annotation_font_color="#000000",
        )
    return fig


def _splice(hist: pd.Series, fut: pd.Series | None) -> tuple[pd.Series, pd.Timestamp]:
    hist = hist.dropna().sort_index()
    origin = hist.index.max()
    if fut is None or fut.dropna().empty:
        return hist, origin
    fut = fut[fut.index > origin].dropna()
    if fut.empty:
        return hist, origin
    return pd.concat([hist, fut]).sort_index(), origin


def load_fy_forecast(panel: pd.DataFrame) -> dict[str, pd.Series] | None:
    """Operating-flow and TGA path after the last DTS print.

    Cash deficit = withdrawals − deposits, the Table II identity the page already plots.
    Buckets run through FY2030. The TGA level is the steered path and stops at its end.
    """
    buckets = DERIVED / "fiscal_path_buckets.parquet"
    if not buckets.exists():
        return None
    b = pd.read_parquet(buckets)
    origin = pd.Timestamp(panel["date"].max())
    dep = [c for c in b.columns if c.startswith("dep_") and c != "dep_total"]
    wd = [c for c in b.columns if c.startswith("wd_") and c != "wd_total"]
    if not dep or not wd:
        return None
    fiscal = -(b[dep].sum(axis=1) - b[wd].sum(axis=1))
    interest = b["wd_interest"] if "wd_interest" in b.columns else pd.Series(0.0, index=b.index)
    primary = fiscal - interest
    idx = pd.date_range(b.index.min(), b.index.max(), freq="D")

    def _flow(s: pd.Series) -> pd.Series:
        return s.reindex(idx).fillna(0.0)[lambda x: x.index > origin]

    out = {
        "dts_fiscal_1d_bn": _flow(fiscal),
        "dts_primary_1d_bn": _flow(primary),
        "dts_interest_1d_bn": _flow(interest),
        "dep_total_1d_bn": _flow(b[dep].sum(axis=1)),
        "wd_total_1d_bn": _flow(b[wd].sum(axis=1)),
    }
    fp_path = DERIVED / "fiscal_path.parquet"
    if fp_path.exists() and "tga_bn" in panel.columns:
        fp = pd.read_parquet(fp_path)
        if isinstance(fp.index, pd.MultiIndex):
            fp = fp.xs("unchanged")
        tga = fp["tga_bn"].dropna()
        tga = tga[tga.index > origin]
        if len(tga):
            cal = pd.date_range(tga.index.min(), tga.index.max(), freq="D")
            level = tga.reindex(cal).ffill()
            change = level.diff()
            last = float(panel.set_index("date")["tga_bn"].dropna().iloc[-1])
            change.iloc[0] = float(level.iloc[0]) - last
            out["tga_bn"] = level
            out["d_tga_1d_bn"] = change.fillna(0.0)
            # Stock identity: Δdebt ≈ DTS cash + ΔTGA (issues − redemptions). Implied = Δdebt − ΔTGA.
            dts = out["dts_fiscal_1d_bn"].reindex(level.index).fillna(0.0)
            d_tga = out["d_tga_1d_bn"].reindex(level.index).fillna(0.0)
            d_debt = dts + d_tga
            last_debt = float(panel.set_index("date")["debt_public_bn"].dropna().iloc[-1])
            out["debt_public_bn"] = last_debt + d_debt.cumsum()
            out["implied_deficit_1d_bn"] = d_debt - d_tga  # = dts on the path
    # Seasonal TGA (prior-2-FY shape) and ceiling-path TGA from debt_ceiling.project.
    ceil_path = DERIVED / "debt_ceiling_forward.parquet"
    if ceil_path.exists():
        c = pd.read_parquet(ceil_path)
        for src, key in (("tga_seasonal_bn", "tga_seasonal_bn"), ("tga_ceiling_bn", "tga_ceiling_bn")):
            if src in c.columns:
                s = c[src].dropna()
                s = s[s.index > origin]
                if len(s):
                    out[key] = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()
    return out


def _forward_12m(hist: pd.Series, fut: pd.Series) -> pd.Series:
    from tga_deficit import LOOKBACK_12M

    comb = pd.concat([hist.dropna(), fut[fut.index > hist.dropna().index.max()]]).sort_index()
    comb = comb.reindex(pd.date_range(comb.index.min(), comb.index.max(), freq="D")).fillna(0.0)
    roll = comb.rolling(LOOKBACK_12M, min_periods=LOOKBACK_12M).sum()
    return roll[roll.index > hist.dropna().index.max()]


def _forward_level_change(hist: pd.Series, fut: pd.Series, days: int) -> pd.Series:
    """``level[t] − level[t−days]`` past the last history print."""
    hist = hist.dropna().sort_index()
    if hist.empty:
        return pd.Series(dtype=float)
    origin = hist.index.max()
    fut = fut.dropna().sort_index()
    fut = fut[fut.index > origin]
    level = pd.concat([hist, fut]).sort_index()
    level = level[~level.index.duplicated(keep="last")]
    level = level.reindex(pd.date_range(level.index.min(), level.index.max(), freq="D")).ffill()
    ch = level - level.shift(int(days))
    return ch[ch.index > origin].dropna()


def _gdp_held(panel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Nominal GDP on ``index``, last print held flat past the panel (reindex alone is all-NaN)."""
    gdp = panel.set_index("date")["gdp_saar_bn"].dropna().sort_index()
    return gdp.reindex(gdp.index.union(index)).ffill().reindex(index)


def _add_dated_forecast(fig: go.Figure, x, y, *, name: str, color: str, width: float) -> None:
    if len(x) == 0:
        return
    _forecast_stroke(fig, x, y, name=name, color=color, width=width, legendgroup=name, showlegend=True)
    if getattr(fig, "_fc_marked", False):
        return
    fig.add_vline(x=pd.Timestamp(pd.Index(x)[0]), line_color="#000000", line_width=1.25)
    fig._fc_marked = True


def _load_bucket_totals() -> pd.DataFrame | None:
    """Daily deposit / withdrawal aggregates from Table II buckets ($bn)."""
    path = DERIVED / "dts_bucket_daily.parquet"
    if not path.exists():
        return None
    b = pd.read_parquet(path)
    if "dep_total" not in b.columns or "wd_total" not in b.columns:
        return None
    out = pd.DataFrame({
        "dep_total_bn": b["dep_total"] / 1000.0,
        "wd_total_bn": b["wd_total"] / 1000.0,
    })
    out.index = pd.DatetimeIndex(b.index)
    out.index.name = "date"
    return out.sort_index()


def _fig_flow_seasonal(
    hist: pd.Series,
    fut: pd.Series | None,
    *,
    title: str,
) -> go.Figure:
    """FYTD cumulative of a daily flow on the fiscal-year calendar (TGA seasonal style)."""
    s, origin = _splice(hist.dropna(), fut)
    return _fig_fy_overlay(s, title=title, ylab="$bn FYTD", origin=origin)


def _fig_tga_seasonal(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    """TGA *level* on the FY calendar — seasonals, not FYTD flow.

    Steered TBAC path is the broken forecast stroke. When present, the prior-2-FY
    median shape (anchored at the TBAC target) is overlaid as a second stroke.
    """
    from tga_deficit import fy_day_of_year, us_fiscal_year

    s, origin = _splice(
        panel.dropna(subset=["tga_bn"]).set_index("date")["tga_bn"],
        None if forecast is None else forecast.get("tga_bn"),
    )
    fig = _fig_fy_overlay(
        s,
        title="TGA balance by fiscal-year day ($bn)",
        ylab="$bn",
        already_fytd=True,
        origin=origin,
    )
    if forecast and forecast.get("tga_seasonal_bn") is not None:
        sea, _ = _splice(
            panel.dropna(subset=["tga_bn"]).set_index("date")["tga_bn"],
            forecast["tga_seasonal_bn"],
        )
        sea = sea[sea.index > origin]
        cur = int(us_fiscal_year(origin))
        sea = sea[[us_fiscal_year(t) == cur for t in sea.index]]
        if len(sea):
            _forecast_stroke(
                fig,
                [fy_day_of_year(t) for t in sea.index],
                sea.to_numpy(),
                name="seasonal shape (prior 2 FY)",
                color=PURPLE,
                width=1.8,
                legendgroup="seasonal",
                showlegend=True,
            )
    fig.add_hrect(
        y0=YELLEN_TGA_BN,
        y1=600,
        fillcolor=_T["chart-zero"],
        opacity=0.12,
        line_width=0,
        annotation_text="Yellen $550–600bn",
        annotation_font_size=10,
        annotation_font_color=_T["chart-legend"],
    )
    fig.add_hline(
        y=TBAC_TGA_SEP_BN,
        line_dash="dash",
        line_color=GREY,
        line_width=1,
        annotation_text="TBAC Sep $950bn",
        annotation_font_size=10,
        annotation_font_color=GREY,
    )
    fig.add_hline(
        y=TBAC_TGA_BN,
        line_dash="dot",
        line_color=GREY,
        line_width=1,
        annotation_text="TBAC Dec $850bn",
        annotation_font_size=10,
        annotation_font_color=GREY,
    )
    for x, _lab in _fy_tax_days():
        fig.add_vline(x=x, line_dash="dot", line_color=_T["dim"], line_width=0.8)
    last = panel.dropna(subset=["tga_bn"]).iloc[-1]
    fig.add_vline(
        x=int(fy_day_of_year(last["date"])),
        line_dash="dot",
        line_color=_T["chart-axis"],
        line_width=1,
        annotation_text=pd.Timestamp(last["date"]).strftime("%d %b"),
        annotation_font_size=10,
        annotation_font_color=_T["chart-axis"],
    )
    return fig


def _load_ceiling() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    """History, forward path, summary — None when debt_ceiling has not been run."""
    hist_p = DERIVED / "debt_limit_daily.parquet"
    fut_p = DERIVED / "debt_ceiling_forward.parquet"
    sum_p = DERIVED / "debt_ceiling_summary.json"
    if not hist_p.exists() or not fut_p.exists() or not sum_p.exists():
        return None, None, None
    import json

    hist = pd.read_parquet(hist_p)
    fut = pd.read_parquet(fut_p)
    summary = json.loads(sum_p.read_text())
    return hist, fut, summary


def _fig_ceiling_headroom(hist: pd.DataFrame, fut: pd.DataFrame, summary: dict) -> go.Figure:
    h = hist.dropna(subset=["headroom_bn"]).loc[DETAIL_START:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h.index, y=h["headroom_bn"], mode="lines", name="Headroom",
        line=dict(color=BLUE, width=1.5),
    ))
    if len(fut) and "headroom_bn" in fut.columns:
        f = fut["headroom_bn"].dropna()
        if len(f):
            _add_dated_forecast(fig, f.index, f, name="Headroom forecast", color=BLUE, width=1.5)
    contact = summary.get("contact_date")
    x_date = summary.get("x_date")
    if contact:
        fig.add_vline(x=pd.Timestamp(contact), line_color=ORANGE, line_width=1.4,
                      annotation_text="contact", annotation_font_size=10)
    if x_date:
        fig.add_vline(x=pd.Timestamp(x_date), line_color=RED, line_width=1.4,
                      annotation_text="X-date", annotation_font_size=10)
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    return _style(fig, title="Debt-limit headroom ($bn) — STL vs statutory limit", ylab="$bn", height=420)


def _fig_ceiling_tga(panel: pd.DataFrame, fut: pd.DataFrame, summary: dict) -> go.Figure:
    """Steered vs seasonal vs post-contact TGA paths."""
    hist = panel.dropna(subset=["tga_bn"]).set_index("date")["tga_bn"].loc[DETAIL_START:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", name="TGA", line=dict(color=GREEN, width=1.5)))
    specs = [
        ("tga_steered_bn", "Production (TBAC + bill reaction)", GREEN, 1.5),
        ("tga_seasonal_bn", "Pure seasonal overlay", PURPLE, 1.4),
        ("tga_ceiling_bn", "Ceiling (no net issuance)", ORANGE, 1.4),
    ]
    for col, name, color, width in specs:
        if col not in fut.columns:
            continue
        s = fut[col].dropna()
        if len(s):
            _add_dated_forecast(fig, s.index, s, name=name, color=color, width=width)
    contact = summary.get("contact_date")
    x_date = summary.get("x_date")
    if contact:
        fig.add_vline(x=pd.Timestamp(contact), line_color=ORANGE, line_width=1.2,
                      annotation_text="contact", annotation_font_size=10)
    if x_date:
        fig.add_vline(x=pd.Timestamp(x_date), line_color=RED, line_width=1.2,
                      annotation_text="X-date", annotation_font_size=10)
    fig.add_hline(y=TBAC_TGA_BN, line_dash="dot", line_color=GREY, line_width=1)
    return _style(fig, title="TGA scenarios — bill reaction vs seasonal vs debt-ceiling", ylab="$bn", height=420)


def _fig_gas(hist: pd.DataFrame) -> go.Figure | None:
    if "gas_net_bn" not in hist.columns or hist["gas_net_bn"].dropna().empty:
        return None
    g = hist.dropna(subset=["gas_net_bn"]).loc[DETAIL_START:]
    roll = g["gas_net_bn"].rolling(20, min_periods=5).sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g.index, y=g["gas_net_bn"], name="GAS net day", marker_color=theme.rgba(GREY, 0.45)))
    fig.add_trace(go.Scatter(x=roll.index, y=roll, mode="lines", name="GAS net 20d",
                             line=dict(color=ORANGE, width=1.4)))
    return _style(fig, title="Government Account Series net (Table IIIA, $bn) — EM book", ylab="$bn", height=360)


def _fytd(panel: pd.DataFrame, col: str, forecast: dict[str, pd.Series] | None, title: str, ylab: str) -> go.Figure:
    s, origin = _splice(panel.dropna(subset=[col]).set_index("date")[col], None if forecast is None else forecast.get(col))
    return _fig_fy_overlay(s, title=title, ylab=ylab, origin=origin)


def _fig_fytd_fiscal_bn(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd(panel, "dts_fiscal_1d_bn", forecast, "FYTD DTS cash deficit ($bn)", "FYTD $bn")


def _fig_fytd_primary_bn(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd(panel, "dts_primary_1d_bn", forecast, "FYTD DTS primary deficit, ex interest ($bn)", "FYTD $bn")


def _fig_fytd_interest_bn(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd(panel, "dts_interest_1d_bn", forecast, "FYTD DTS interest on Treasury securities ($bn)", "FYTD $bn")


def _fytd_pct(panel: pd.DataFrame, daily: str, forecast: dict[str, pd.Series] | None, title: str) -> go.Figure:
    """FYTD dollars over GDP. The forecast uses the last printed GDP, held flat."""
    from tga_deficit import fy_index

    hist = panel.dropna(subset=[daily]).set_index("date")[daily]
    fut = None if forecast is None else forecast.get(daily)
    s, origin = _splice(hist, fut)
    fytd = s.groupby(fy_index(s.index)).cumsum()
    return _fig_fy_overlay(
        100.0 * fytd / _gdp_held(panel, s.index), title=title, ylab="FYTD % of GDP", already_fytd=True, origin=origin,
    )


def _fig_fytd_fiscal_pct(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd_pct(panel, "dts_fiscal_1d_bn", forecast, "FYTD DTS cash deficit / nominal GDP")


def _fig_fytd_primary_pct(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd_pct(panel, "dts_primary_1d_bn", forecast, "FYTD DTS primary (ex interest) / nominal GDP")


def _fig_fytd_interest_pct(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd_pct(panel, "dts_interest_1d_bn", forecast, "FYTD DTS interest on Treasury securities / nominal GDP")


def _fig_fytd_tga(panel: pd.DataFrame, forecast: dict[str, pd.Series] | None = None) -> go.Figure:
    return _fytd(panel, "d_tga_1d_bn", forecast, "FYTD cumulative ΔTGA ($bn)", "FYTD $bn")


def _fig_fytd_buybacks(ops: pd.DataFrame) -> go.Figure:
    o = ops.dropna(subset=["operation_date", "accepted_bn"]).copy()
    total = o.set_index("operation_date")["accepted_bn"].groupby(level=0).sum().resample("D").sum().fillna(0.0)
    return _fig_fy_overlay(
        total,
        title="FYTD cumulative Treasury buybacks, all types ($bn)",
        ylab="FYTD $bn",
    )


def write_matplotlib_png(panel: pd.DataFrame, path: Path) -> Path:
    d = _since(panel.dropna(subset=["implied_deficit_12m_pct_gdp"]))
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=140)
    ax.plot(d["date"], d["implied_deficit_12m_pct_gdp"], color=BLUE, lw=1.6, label="TGA-implied 12m")
    if "dts_fiscal_12m_pct_gdp" in d.columns:
        f = d.dropna(subset=["dts_fiscal_12m_pct_gdp"])
        ax.plot(f["date"], f["dts_fiscal_12m_pct_gdp"], color=ORANGE, lw=1.2, label="DTS cash 12m")
    if "dts_primary_12m_pct_gdp" in d.columns:
        p = d.dropna(subset=["dts_primary_12m_pct_gdp"])
        ax.plot(p["date"], p["dts_primary_12m_pct_gdp"], color=RED, lw=1.1, label="DTS primary 12m")
    if "mts_deficit_12m_pct_gdp" in d.columns:
        m = d.dropna(subset=["mts_deficit_12m_pct_gdp"])
        ax.plot(m["date"], m["mts_deficit_12m_pct_gdp"], color=GREY, lw=1.1, ls=":", label="MTS official 12m")
    ax.axhline(0, color=_T["chart-axis"], lw=0.6)
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.8)
    ax.set_ylabel("% of GDP")
    ax.set_title("US deficit / interpolated GDP (daily, 2022–)")
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_fytd_png(panel: pd.DataFrame, path: Path) -> Path:
    from matplotlib.gridspec import GridSpec
    from tga_deficit import fy_pivot, fytd_cumsum

    ticks, labels = _fy_month_ticks()
    fig = plt.figure(figsize=(11.2, 10.4), dpi=140)
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.22)
    specs = [
        (fig.add_subplot(gs[0, :]), "tga_bn", True, True, "TGA balance $bn — FY overlay", "$bn"),
        (fig.add_subplot(gs[1, 0]), "dts_fiscal_1d_bn", False, False, "FYTD DTS cash deficit $bn", "$bn"),
        (fig.add_subplot(gs[1, 1]), "dts_primary_1d_bn", False, False, "FYTD DTS primary $bn", "$bn"),
        (fig.add_subplot(gs[2, 0]), "dts_interest_1d_bn", False, False, "FYTD DTS interest $bn", "$bn"),
        (fig.add_subplot(gs[2, 1]), "d_tga_1d_bn", False, False, "FYTD ΔTGA $bn", "$bn"),
    ]
    for ax, col, already, is_level, title, ylab in specs:
        if col not in panel.columns:
            ax.set_title(title)
            continue
        s = panel.dropna(subset=[col]).set_index("date")[col]
        ytd = fy_pivot(s) if already else fytd_cumsum(s)
        present = [c for c in FY_YEARS if c in ytd.columns]
        current = max(present) if present else None
        for year in FY_YEARS:
            if year not in ytd.columns:
                continue
            col_s = ytd[year].dropna()
            ax.plot(
                col_s.index,
                col_s.values,
                color=YEAR_COLORS[year],
                lw=2.0 if year == current else 1.1,
                label=f"FY{year}",
            )
        if is_level:
            ax.axhspan(YELLEN_TGA_BN, 600, color=_T["chart-zero"], alpha=0.12, zorder=0)
            ax.axhline(TBAC_TGA_SEP_BN, color=_T["chart-zero"], ls="--", lw=0.8)
            ax.axhline(TBAC_TGA_BN, color=_T["chart-zero"], ls=":", lw=0.8)
            for x, _lab in _fy_tax_days():
                ax.axvline(x, color=_T["dim"], ls=":", lw=0.7)
        else:
            ax.axhline(0, color=_T["chart-axis"], lw=0.5)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.legend(frameon=False, fontsize=7, ncol=3)
        _mpl_box(ax)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def extra_buyback_bn_yr(new_cap_bn: float) -> float:
    """Incremental long-end OFTR vs the standing $2bn cap, both sectors."""
    return max(new_cap_bn - CURRENT_LIQ_CAP_BN, 0.0) * LONG_END_OPS_PER_YEAR


def tga_runway_months(tga_bn: float, extra_bn_yr: float, floor_bn: float) -> float:
    if extra_bn_yr <= 0:
        return float("inf")
    return max(tga_bn - floor_bn, 0.0) / extra_bn_yr * 12.0


def _pm_ax(ax) -> None:
    ax.grid(True, color="0.92", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8, colors=_T["chart-axis"])
    ax.axhline(0, color=_T["chart-axis"], lw=0.5)


def _pm_end(ax, x, y, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
        fontweight="bold",
        clip_on=False,
    )


def _pm_fy_overlay(ax, panel: pd.DataFrame, col: str) -> None:
    from tga_deficit import fy_pivot

    s = panel.dropna(subset=[col]).set_index("date")[col]
    ytd = fy_pivot(s)
    ticks, labels = _fy_month_ticks()
    present = [c for c in FY_YEARS if c in ytd.columns]
    current = max(present) if present else None
    last_y = None
    for year in FY_YEARS:
        if year not in ytd.columns:
            continue
        ser = ytd[year].dropna()
        ax.plot(
            ser.index,
            ser.values,
            color=YEAR_COLORS[year],
            lw=2.15 if year == current else 1.05,
            label=f"FY{year}",
        )
        if year == current and len(ser):
            last_y = (ser.index[-1], float(ser.iloc[-1]), year)
    if last_y:
        _pm_end(ax, last_y[0], last_y[1], f"FY{last_y[2]} {last_y[1]:.2f}%", RED)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper left")
    _pm_ax(ax)


def write_pm_briefing(panel: pd.DataFrame, bb: pd.DataFrame | None, outdir: Path) -> list[Path]:
    """Single-purpose PNGs for a desk walkthrough."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    d = _since(panel)
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    series = [
        ("implied_deficit_12m_pct_gdp", "Stock identity (Δdebt − ΔTGA)", BLUE, 1.7),
        ("mts_deficit_12m_pct_gdp", "MTS official", GREY, 1.1),
        ("dts_fiscal_12m_pct_gdp", "DTS cash", ORANGE, 1.35),
        ("dts_primary_12m_pct_gdp", "DTS primary (ex interest)", RED, 1.45),
    ]
    last_pts = []
    for col, name, color, lw in series:
        if col not in d.columns:
            continue
        m = d.dropna(subset=[col])
        ls = ":" if col.startswith("mts") else "-"
        ax.plot(m["date"], m[col], color=color, lw=lw, ls=ls, label=name)
        last_pts.append((m["date"].iloc[-1], float(m[col].iloc[-1]), color, name.split()[0]))
    for x, y, color, _ in last_pts:
        _pm_end(ax, x, y, f"{y:.1f}%", color)
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.7)
    ax.set_ylabel("% of GDP")
    ax.set_title("Twelve-month deficit / nominal GDP", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_01_deficit_12m.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    _pm_fy_overlay(ax, panel, "dts_primary_fytd_pct_gdp")
    ax.set_ylabel("% of GDP")
    ax.set_title("FYTD primary deficit (ex interest) / nominal GDP", loc="left", fontsize=11)
    fig.tight_layout()
    p = outdir / "pm_02_fytd_primary.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    _pm_fy_overlay(ax, panel, "dts_interest_fytd_pct_gdp")
    ax.set_ylabel("% of GDP")
    ax.set_title("FYTD interest on Treasury securities / nominal GDP", loc="left", fontsize=11)
    fig.tight_layout()
    p = outdir / "pm_03_fytd_interest.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    t = _since(panel.dropna(subset=["tga_bn"]))
    ax.axhspan(550, 600, color=GREY, alpha=0.22, label="Yellen ~$550–600bn")
    ax.plot(t["date"], t["tga_bn"], color=GREEN, lw=1.5, label="TGA close")
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.7)
    last = t.iloc[-1]
    _pm_end(ax, last["date"], last["tga_bn"], f"${last['tga_bn']:.0f}bn", GREEN)
    ax.set_ylabel("$bn")
    ax.set_title("TGA closing balance", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_04_tga.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    if bb is not None and not bb.empty:
        b = bb[bb["month"] >= pd.Timestamp("2024-01-01")].copy()
        ax.bar(b["month"], b.get("cm_front_bn", 0), width=22, color=GREY, label="Cash management 1mo–2y")
        bottom = b.get("cm_front_bn", 0)
        ax.bar(b["month"], b.get("liq_20_30_bn", 0), width=22, color=RED, bottom=bottom, label="Liquidity support 20y–30y")
        bottom = bottom + b.get("liq_20_30_bn", 0)
        ax.bar(b["month"], b.get("liq_10_20_bn", 0), width=22, color=ORANGE, bottom=bottom, label="Liquidity support 10y–20y")
        ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.7)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylabel("$bn par")
    ax.set_title("Treasury buybacks by operation type", loc="left", fontsize=11)
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_05_buybacks.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    m = _since(panel.dropna(subset=["dts_primary_12m_pct_gdp", "dts_interest_12m_pct_gdp"]))
    ax.stackplot(
        m["date"],
        m["dts_primary_12m_pct_gdp"],
        m["dts_interest_12m_pct_gdp"],
        colors=[ORANGE, RED],
        labels=["12m primary", "12m interest"],
        alpha=0.85,
        lw=0,
    )
    ax.plot(m["date"], m["dts_fiscal_12m_pct_gdp"], color=BLUE, lw=1.2, label="12m cash (sum)")
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.7)
    last = m.iloc[-1]
    _pm_end(ax, last["date"], last["dts_fiscal_12m_pct_gdp"], f"cash {last['dts_fiscal_12m_pct_gdp']:.1f}%", BLUE)
    ax.set_ylabel("% of GDP")
    ax.set_title("Twelve-month cash deficit: primary and interest / nominal GDP", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_06_primary_vs_interest.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    from tga_deficit import LOOKBACK_12M

    last = panel.dropna(subset=["dts_primary_12m_bn", "tga_bn"]).iloc[-1]
    tga0 = float(last["tga_bn"])
    caps = (4.0, 6.0, 8.0)
    cap_colors = {4.0: ORANGE, 6.0: RED, 8.0: "#7b241c"}
    floors = ((TBAC_TGA_BN, ORANGE, "TBAC $850bn"), (YELLEN_TGA_BN, RED, "Yellen $550bn"))

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    x = list(range(len(caps)))
    width = 0.36
    for i, (floor, color, _lab) in enumerate(floors):
        xs = [xi + (i - 0.5) * width for xi in x]
        heights = [tga_runway_months(tga0, extra_buyback_bn_yr(cap), floor) for cap in caps]
        ax.bar(xs, heights, width=width, color=color, label=f"vs {_lab} floor")
        for xi, h in zip(xs, heights):
            ax.text(xi, h + 1.2, f"{h:.0f}m", ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"${c:.0f}bn/op\n(+${extra_buyback_bn_yr(c):.0f}bn/yr extra)" for c in caps]
    )
    ax.set_ylabel("months of TGA (incremental only)")
    ax.set_title("Incremental long-end buybacks vs TGA floors", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_07_buyback_runway.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    months = range(0, 37)
    asof = pd.Timestamp(last["date"])
    qra_m = (QRA_DATE - asof).days / 30.437
    ax.axhline(TBAC_TGA_BN, color=GREY, ls="--", lw=0.9, label="TBAC Dec $850bn")
    ax.axhline(YELLEN_TGA_BN, color=BLUE, ls="--", lw=0.9, label="Yellen $550bn")
    ax.axvline(qra_m, color=_T["chart-axis"], ls=":", lw=0.8)
    ax.text(qra_m + 0.4, tga0 - 20, "4 Nov", fontsize=8, color=_T["chart-axis"])
    for cap in caps:
        extra = extra_buyback_bn_yr(cap)
        path = [tga0 - extra / 12.0 * m for m in months]
        ax.plot(list(months), path, color=cap_colors[cap], lw=1.4, label=f"${cap:.0f}bn/op + ${extra:.0f}bn/yr extra")
    ax.set_xlim(0, 36)
    ax.set_ylabel("TGA $bn")
    ax.set_xlabel("Months from now")
    ax.set_title("TGA path under incremental long-end buybacks", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_08_tga_path_buybacks.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    end = pd.Timestamp(last["date"])
    start = end - pd.Timedelta(days=LOOKBACK_12M - 1)
    win = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    net_iss = float((win["dts_issues_bn"].fillna(0) - win["dts_redeem_bn"].fillna(0)).sum())
    bars = [
        ("Primary", float(last["dts_primary_12m_bn"]), ORANGE),
        ("Interest", float(last["dts_interest_12m_bn"]), RED),
        ("Cash deficit", float(last["dts_fiscal_12m_bn"]), BLUE),
        ("Net issuance", net_iss, GREY),
        ("ΔTGA", float(last["d_tga_12m_bn"]), GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=150)
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for i, (_name, val, color) in enumerate(bars):
        ax.text(i, val + (18 if val >= 0 else -40), f"${val:,.0f}bn", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, color=color, fontweight="bold")
    ax.set_ylabel("$bn, last 12m")
    ax.set_title("Last 12 months: primary, interest, cash, issuance, ΔTGA", loc="left", fontsize=11)
    _pm_ax(ax)
    fig.tight_layout()
    p = outdir / "pm_09_issued_the_interest.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    return written


def write_diagnostic_png(diag: pd.DataFrame, bb: pd.DataFrame | None, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), dpi=140)
    ax = axes[0, 0]
    d = _since(diag.dropna(subset=["nipa_interest_pct_gdp"]))
    ax.plot(d["date"], d["nipa_interest_pct_gdp"], color=RED, lw=1.3, label="NIPA interest/GDP")
    if "interest_public_12m_pct_gdp" in d.columns:
        m = d.dropna(subset=["interest_public_12m_pct_gdp"])
        ax.plot(m["date"], m["interest_public_12m_pct_gdp"], color=ORANGE, lw=1.0, label="Public-issues 12m/GDP")
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.8)
    ax.set_title("Fiscal: interest / GDP")
    ax.set_ylabel("% of GDP")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.9", lw=0.5)

    ax = axes[0, 1]
    d = _since(diag.dropna(subset=["sofr_iorb_bp"]))
    ax.plot(d["date"], d["sofr_iorb_bp"], color=BLUE, lw=1.2, label="SOFR−floor")
    ax.axhline(0, color=_T["chart-axis"], lw=0.5)
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.8)
    ax.set_title("Liquidity: overnight repo")
    ax.set_ylabel("SOFR−floor bp")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color="0.9", lw=0.5)
    if "sofr_tgcr_bp" in d.columns:
        ax2 = ax.twinx()
        m = d.dropna(subset=["sofr_tgcr_bp"])
        ax2.plot(m["date"], m["sofr_tgcr_bp"], color=ORANGE, lw=1.0, label="SOFR−TGCR")
        ax2.set_ylabel("SOFR−TGCR bp", color=ORANGE)
        ax2.tick_params(axis="y", labelcolor=ORANGE)
        ax2.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1, 0]
    d = _since(diag.dropna(subset=["otr_richness_30_bp"]))
    ax.plot(d["date"], d["otr_richness_30_bp"], color=BLUE, lw=1.1, label="GSW−CMT 30y")
    a = d.dropna(subset=["ust30_auction_vs_gsw_bp"])
    if not a.empty:
        ax.scatter(a["date"], a["ust30_auction_vs_gsw_bp"], s=12, c=RED, label="30y auction−GSW", zorder=3)
    ax.axhline(0, color=_T["chart-axis"], lw=0.5)
    ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.8)
    ax.set_title("Liquidity: 30y OTR vs OFTR (bp)")
    ax.set_ylabel("bp")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.9", lw=0.5)

    ax = axes[1, 1]
    if bb is not None and not bb.empty:
        d = bb[bb["month"] >= DETAIL_START]
        ax.bar(d["month"], d.get("cm_front_bn", 0), width=20, color=GREY, label="Cash mgmt")
        bottom = d.get("cm_front_bn", 0)
        ax.bar(d["month"], d.get("liq_20_30_bn", 0), width=20, color=RED, bottom=bottom, label="Liq 20–30y")
        bottom = bottom + d.get("liq_20_30_bn", 0)
        ax.bar(d["month"], d.get("liq_10_20_bn", 0), width=20, color=ORANGE, bottom=bottom, label="Liq 10–20y")
        ax.axvline(BESSENT, color=PURPLE, ls="--", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Buybacks $bn (Treasury labels)")
    ax.set_ylabel("$bn")
    ax.grid(True, color="0.9", lw=0.5, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def build(*, png_path: Path | None = None) -> Path:
    from tga_deficit import fy_day_of_year, fy_pivot, fytd_through_md, us_fiscal_year

    panel = _load()
    last = panel.dropna(subset=["implied_deficit_12m_pct_gdp"]).iloc[-1]
    asof_ts = pd.Timestamp(last["date"])
    asof = asof_ts.strftime("%Y-%m-%d")
    fy_now = us_fiscal_year(asof_ts)
    tga_wide = fy_pivot(panel.dropna(subset=["tga_bn"]).set_index("date")["tga_bn"])
    tga_prev = _value_at_doy(tga_wide, fy_now - 1, fy_day_of_year(asof_ts))
    tga_kpi_note = (
        f"FY{fy_now - 1} same FY-day ${tga_prev:,.0f}bn"
        if pd.notna(tga_prev)
        else "DTS Table I · Yellen ~$550–600bn"
    )
    mts_through = None
    mts_cache = HERE / "data" / "cache" / "fred_mtsds133fms.csv"
    if mts_cache.exists():
        mts_raw = pd.read_csv(mts_cache, parse_dates=["date"]).dropna()
        if not mts_raw.empty:
            mts_through = pd.Timestamp(mts_raw["date"].iloc[-1])
    mts_note = ""
    if pd.notna(last.get("mts_deficit_12m_pct_gdp")):
        thru = f" · through {mts_through.strftime('%b-%Y')}" if mts_through is not None else ""
        mts_note = f"MTS 12m {last['mts_deficit_12m_pct_gdp']:.1f}%{thru}"

    kpis = "".join(
        [
            _kpi("12m implied / GDP", f"{last['implied_deficit_12m_pct_gdp']:.1f}%", asof),
            _kpi("12m implied", f"${last['implied_deficit_12m_bn']:,.0f}bn", mts_note),
            _kpi("TGA close", f"${last['tga_bn']:,.0f}bn", tga_kpi_note),
            _kpi(
                "3m ann. / GDP",
                f"{last['implied_deficit_3m_ann_pct_gdp']:.1f}%",
                f"20d MA {last['implied_deficit_3m_ann_pct_gdp_ma20']:.1f}%",
            ),
        ]
    )
    if pd.notna(last.get("dts_fiscal_12m_pct_gdp")):
        kpis += _kpi(
            "12m DTS cash",
            f"${last['dts_fiscal_12m_bn']:,.0f}bn",
            f"primary ${last['dts_primary_12m_bn']:,.0f}bn · interest ${last['dts_interest_12m_bn']:,.0f}bn",
        )
        kpis += _kpi(
            "12m DTS cash / GDP",
            f"{last['dts_fiscal_12m_pct_gdp']:.1f}%",
            f"primary {last['dts_primary_12m_pct_gdp']:.1f}% · interest {last['dts_interest_12m_pct_gdp']:.1f}%",
        )
    if "dts_fiscal_1d_bn" in panel.columns:
        idx = panel.set_index("date")
        through = fytd_through_md(idx["dts_fiscal_1d_bn"], asof_ts.month, asof_ts.day)
        prim = fytd_through_md(idx["dts_primary_1d_bn"], asof_ts.month, asof_ts.day)
        inte = fytd_through_md(idx["dts_interest_1d_bn"], asof_ts.month, asof_ts.day)
        now_bn = float(through.get(fy_now, float("nan")))
        prev_bn = float(through.get(fy_now - 1, float("nan")))
        now_p = float(prim.get(fy_now, float("nan")))
        prev_p = float(prim.get(fy_now - 1, float("nan")))
        now_i = float(inte.get(fy_now, float("nan")))
        prev_i = float(inte.get(fy_now - 1, float("nan")))
        kpis += _kpi(
            f"FY{fy_now} TD cash",
            f"${now_bn:,.0f}bn" if pd.notna(now_bn) else "—",
            f"FY{fy_now - 1} same day ${prev_bn:,.0f}bn" if pd.notna(now_bn) and pd.notna(prev_bn) else "US FY Oct–Sep",
        )
        kpis += _kpi(
            f"FY{fy_now} TD primary",
            f"${now_p:,.0f}bn" if pd.notna(now_p) else "—",
            f"FY{fy_now - 1} same day ${prev_p:,.0f}bn" if pd.notna(now_p) and pd.notna(prev_p) else "ex Table II interest",
        )
        kpis += _kpi(
            f"FY{fy_now} TD interest",
            f"${now_i:,.0f}bn" if pd.notna(now_i) else "—",
            f"FY{fy_now - 1} same day ${prev_i:,.0f}bn" if pd.notna(now_i) and pd.notna(prev_i) else "Table II cash coupons",
        )

    forecast = load_fy_forecast(panel)
    flow_fc_end = tga_fc_end = None
    if forecast and forecast.get("tga_bn") is not None and len(forecast["tga_bn"]):
        tga_fc_end = pd.Timestamp(forecast["tga_bn"].index.max()).strftime("%d %b %Y")
    if forecast and forecast.get("dts_fiscal_1d_bn") is not None and len(forecast["dts_fiscal_1d_bn"]):
        flow_fc_end = pd.Timestamp(forecast["dts_fiscal_1d_bn"].index.max()).strftime("%d %b %Y")
    gdp_figs = [_fig_12m(panel, forecast), _fig_dts_split(panel, forecast), _fig_dts_split_bn(panel, forecast),
                _fig_3m(panel, forecast)]
    fytd_bn_figs: list[go.Figure] = [_fig_tga_seasonal(panel, forecast)]
    buckets = _load_bucket_totals()
    if buckets is not None:
        dep_fut = None if forecast is None else forecast.get("dep_total_1d_bn")
        wd_fut = None if forecast is None else forecast.get("wd_total_1d_bn")
        fytd_bn_figs += [
            _fig_flow_seasonal(
                buckets["dep_total_bn"],
                dep_fut,
                title="Deposits aggregate by fiscal-year day ($bn FYTD)",
            ),
            _fig_flow_seasonal(
                buckets["wd_total_bn"],
                wd_fut,
                title="Withdrawals aggregate by fiscal-year day ($bn FYTD)",
            ),
        ]
    fytd_pct_figs: list[go.Figure] = []
    if "dts_fiscal_1d_bn" in panel.columns:
        fytd_bn_figs += [
            _fig_fytd_fiscal_bn(panel, forecast),
            _fig_fytd_primary_bn(panel, forecast),
            _fig_fytd_interest_bn(panel, forecast),
            _fig_fytd_tga(panel, forecast),
        ]
        fytd_pct_figs = [
            _fig_fytd_fiscal_pct(panel, forecast),
            _fig_fytd_primary_pct(panel, forecast),
            _fig_fytd_interest_pct(panel, forecast),
        ]
    ops = _load_buybacks_ops()
    if ops is not None and not ops.empty:
        fytd_bn_figs.append(_fig_fytd_buybacks(ops))
    tga_figs = [_fig_tga(panel, forecast), _fig_components(panel, forecast)]

    diag = _load_diag()
    bb = _load_buybacks_m()
    extra_kpis = ""
    if diag is not None and not diag.empty:
        dlast = (
            diag.dropna(subset=["nipa_interest_pct_gdp"]).iloc[-1]
            if diag["nipa_interest_pct_gdp"].notna().any()
            else diag.iloc[-1]
        )
        sofr_bp = dlast.get("sofr_iorb_bp")
        otr_bp = dlast.get("otr_richness_30_bp")
        extra_kpis = "".join(
            [
                _kpi(
                    "NIPA interest / GDP",
                    f"{dlast['nipa_interest_pct_gdp']:.2f}%" if pd.notna(dlast.get("nipa_interest_pct_gdp")) else "—",
                    f"avg coupon {dlast['avg_mkt_rate']:.2f}%" if pd.notna(dlast.get("avg_mkt_rate")) else "",
                ),
                _kpi(
                    "SOFR − floor",
                    f"{sofr_bp:.1f} bp" if pd.notna(sofr_bp) else "—",
                    f"SOFR−TGCR {dlast['sofr_tgcr_bp']:.1f} bp" if pd.notna(dlast.get("sofr_tgcr_bp")) else "NY Fed",
                ),
                _kpi(
                    "30y OFTR−CMT",
                    f"{otr_bp:.1f} bp" if pd.notna(otr_bp) else "—",
                    "GSW par − DGS30",
                ),
            ]
        )
        if bb is not None and not bb.empty:
            post = bb[bb["month"] >= BESSENT]
            extra_kpis += _kpi(
                "Bessent buybacks",
                f"liq 20–30y ${post['liq_20_30_bn'].sum():.0f}bn",
                f"cash mgmt ${post['cm_front_bn'].sum():.0f}bn",
            )

    books_note = ""
    if mts_through is not None and "dts_fiscal_1d_bn" in panel.columns:
        win_end = (mts_through + pd.offsets.MonthEnd(0)).normalize()
        win_start = (mts_through - pd.DateOffset(months=11)).replace(day=1)
        win = panel[(panel["date"] >= win_start) & (panel["date"] <= win_end)]
        dts_al = float(win["dts_fiscal_1d_bn"].sum()) if not win.empty else float("nan")
        stock_row = panel.loc[panel["date"] <= win_end].dropna(subset=["implied_deficit_12m_bn"])
        stock_al = float(stock_row.iloc[-1]["implied_deficit_12m_bn"]) if not stock_row.empty else float("nan")
        mts_al = float(last["mts_deficit_12m_bn"])
        books_note = f"""
    <p class="sub">Three 12-month books (not a scrape error). DTS cash is the financing residual
      (public-debt cash issues − redemptions − ΔTGA) through {asof}. MTS is the unified budget
      deficit, last twelve months through {mts_through.strftime("%b-%Y")}. Par stock is
      Δ(debt held by the public) − ΔTGA — TIPS inflation and par vs proceeds sit here.
      Same MTS window {win_start.strftime("%b-%Y")}–{win_end.strftime("%b-%Y")}:
      DTS cash ${dts_al:,.0f}bn, MTS ${mts_al:,.0f}bn, par stock ${stock_al:,.0f}bn.
      Primary/interest split is DTS cash only.</p>"""

    all_figs = gdp_figs + fytd_bn_figs + fytd_pct_figs + tga_figs
    cards = [f"<div class='card'>{_html(f, first=(i == 0))}</div>" for i, f in enumerate(all_figs)]
    n0 = len(gdp_figs)
    n1 = n0 + len(fytd_bn_figs)
    n2 = n1 + len(fytd_pct_figs)
    gdp_html = "".join(cards[:n0])
    fytd_cards = list(cards[n0:n1])
    if fytd_cards:
        fytd_cards[0] = fytd_cards[0].replace("class='card'", "class='card span-2'", 1)
    fytd_bn_html = "".join(fytd_cards)
    fytd_pct_html = "".join(cards[n1:n2])
    tga_html = "".join(cards[n2:])
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>US cash — TGA / DTS</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <header>
    <h1>US cash — TGA / DTS</h1>
    <div class="sub">{built} · Daily DTS scrape from
      <code>fiscaldata.treasury.gov</code> (Table I TGA, Table II deposits/withdrawals, Table IIIC debt).
      Stock identity: implied deficit = 365d Δ(debt held by the public) − 365d ΔTGA.
      Flow identity: DTS cash deficit = (public-debt cash issues − redemptions) − ΔTGA;
      primary = that minus Table II <em>Interest on Treasury Securities</em>.
      Denominator is BEA quarterly nominal GDP (SAAR), linearly interpolated to calendar days
      and held flat after the last print. FYTD uses the US fiscal year (1 Oct – 30 Sep).
      MTS overlay is FRED MTSDS133FMS trailing 12m. Cash, not NIPA federal deficit.
      Table II buckets, the debt ceiling and funding liquidity are separate tabs.</div>
  </header>
  <div class="wrap">
    <div class="kpi-row">{kpis}{extra_kpis}</div>
    {books_note}
    <h2 class="section">Share of interpolated nominal GDP</h2>
    <div class="grid">{gdp_html}</div>
    <h2 class="section">FY seasonals — TGA level, deposits / withdrawals, FYTD $bn</h2>
    <p class="sub">Last three fiscal years. The latest forecast year is blue, then the MATLAB order backward
      (FY2030 blue, FY2029 orange, FY2028 yellow, FY2027 purple, FY2026 green, FY2025 cyan).
      The vertical rule is the last DTS print. After it the line is the same colour, broken, over a lighter
      continuous stroke so each day still shows. TGA is the cash balance level; deposits and withdrawals
      below it are FYTD cumulatives of the Table II aggregates. Operating flows run through
      {flow_fc_end or "FY2030"}; the TGA path through {tga_fc_end or "the steered cash balance"}.
      Faint dotted verticals on the TGA chart are tax dates (15 Apr / 15 Jun / 15 Sep).</p>
    <div class="grid-2">{fytd_bn_html}</div>
    <h2 class="section">FYTD / nominal GDP</h2>
    <p class="sub">Year-to-date dollar flow over annual SAAR GDP (not annualized).</p>
    <div class="grid-2">{fytd_pct_html}</div>
    <div class="grid">{tga_html}</div>
  </div>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")

    png = png_path or (DERIVED / "tga_implied_deficit_12m_pct_gdp.png")
    write_matplotlib_png(panel, png)
    print(f"Wrote {png}")
    if "dts_fiscal_1d_bn" in panel.columns:
        fpng = DERIVED / "dts_fytd_flows.png"
        write_fytd_png(panel, fpng)
        print(f"Wrote {fpng}")
    if diag is not None:
        dpng = DERIVED / "ust_liquidity_vs_fiscal.png"
        write_diagnostic_png(diag, bb, dpng)
        print(f"Wrote {dpng}")
    for p in write_pm_briefing(panel, bb, DERIVED / "pm_briefing"):
        print(f"Wrote {p}")
    return OUT


if __name__ == "__main__":
    build()
