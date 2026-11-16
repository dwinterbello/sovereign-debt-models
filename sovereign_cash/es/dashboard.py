"""Spain cash page: IGAE need + bond/Letras financing legs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "es_cash_dashboard.html"
DERIVED = ROOT / "derived"


def build() -> Path:
    from sovereign_cash.es import bottom_up_macro as BUM

    path = DERIVED / "es_cash_monthly.parquet"
    if not path.exists():
        from sovereign_cash.es import run as R
        R.run(dashboard=False, verbose=False)
    frame = pd.read_parquet(path)
    frame["month"] = pd.to_datetime(frame["month"])
    ev = BUM.evaluate()
    hist = frame.loc[~frame["is_forecast"].astype(bool)]
    fc = frame.loc[frame["is_forecast"].astype(bool)]
    src = str(fc["fc_source"].iloc[0]) if len(fc) and "fc_source" in fc else "forecast"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("Cash requirement (€bn)", "Net financing (€bn)"),
    )
    fig.add_trace(go.Bar(
        x=hist["month"], y=hist["cash_requirement_bn"], name="IGAE B.9",
        marker_color="#64748b"), row=1, col=1)
    if not fc.empty:
        fig.add_trace(go.Bar(
            x=fc["month"], y=fc["cash_requirement_bn"],
            name=f"Forecast ({src})", marker_color="#1f4e79"), row=1, col=1)
        if "cash_requirement_bottomup_bn" in fc and fc["cash_requirement_bottomup_bn"].notna().any():
            fig.add_trace(go.Scatter(
                x=fc["month"], y=fc["cash_requirement_bottomup_bn"],
                name="Bottom-up", mode="lines", line=dict(color="#c45c26", width=2)),
                row=1, col=1)

    for col, name, color in (
            ("net_bonds_bn", "Net Bonos", "#1f4e79"),
            ("net_bills_bn", "Net Letras", "#2a9d8f"),
            ("other_fin_bn", "Other fin", "#b08968"),
    ):
        if col not in frame.columns:
            continue
        fig.add_trace(go.Bar(
            x=hist["month"], y=hist[col], name=name, marker_color=color,
            legendgroup=name, showlegend=True), row=2, col=1)
        if not fc.empty:
            fig.add_trace(go.Bar(
                x=fc["month"], y=fc[col], name=name, marker_color=color,
                opacity=0.55, legendgroup=name, showlegend=False), row=2, col=1)

    fig.update_layout(
        title="Spain — Administración Central cash & financing",
        template="plotly_white", height=720, barmode="relative",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=50, r=20, t=80, b=40),
    )
    note = (
        f"Live rest-of-year cash: {ev['model']} (SHIP={ev.get('ship', BUM.SHIP)}). "
        f"PRE |err|/|year| {ev['pre_mean_ratio']:.3f} vs seas_add_3 {ev['prod_pre_mean_ratio']:.3f}. "
        f"Financing: Tesoro stock Δ + sheet-13 / ops; Letras flex against plan 09."
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Spain cash</title>
<style>
body{{font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:18px 24px;color:#1d2433}}
p.sub{{color:#4a5263;max-width:980px}}
</style></head><body>
<h1>Spain — IGAE central cash</h1>
<p class="sub">{note}</p>
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<p class="sub">Sources: IGAE AC B.9, Tesoro bulletins 01/09/13, BdE BE_12_5.3, ops_es.
Identity: Δbuffer = −cash + net Bonos + net Letras + other_fin.</p>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
