"""Issuance board theme — thin layer over the shared desk theme (portal/theme.py)."""

from portal import theme

MATLAB = theme.SERIES

COUNTRY_COLOR = {
    "UK": MATLAB[0],
    "DE": MATLAB[1],
    "FR": MATLAB[2],
    "IT": MATLAB[3],
    "US": MATLAB[4],
    "JP": MATLAB[5],
}

PAGE_CSS = theme.page_css() + """
body { margin:0; color:var(--text); font:12px/1.45 var(--sans); font-variant-numeric:tabular-nums; }
header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:14px;
         align-items:baseline; flex-wrap:wrap; background:var(--panel); }
h1 { font-size:13px; margin:0; font-weight:600; letter-spacing:.1px; }
.sub { color:var(--muted); font-size:11px; max-width:980px; }
nav { display:flex; gap:0; padding:0 16px; border-bottom:1px solid var(--line); background:var(--panel); }
nav button { background:none; border:none; color:var(--muted); padding:7px 11px; cursor:pointer;
             font-size:11px; border-bottom:2px solid transparent; font-family:inherit; }
nav button.on { color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }
main { display:grid; grid-template-columns:220px 1fr; gap:10px; padding:10px 16px 28px; }
aside { background:var(--panel); border:1px solid var(--line); padding:10px;
        height:fit-content; position:sticky; top:6px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
aside h2 { font-size:10px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted);
           margin:12px 0 4px; font-weight:600; border-bottom:1px solid var(--grid); padding-bottom:2px; }
aside h2:first-child { margin-top:0; }
select, input { width:100%; background:var(--panel); color:var(--text); border:1px solid var(--line);
                border-radius:0; padding:4px 6px; font-size:11px; font-family:inherit; }
.card { background:var(--panel-hi); border:1px solid var(--line); padding:10px 12px;
        margin-bottom:10px; box-shadow:0 1px 2px rgba(0,0,0,.04); overflow:hidden; }
.js-plotly-plot, .plotly-graph-div { width:100% !important; max-width:100%; }
.card h3 { margin:0 0 6px; font-size:12px; font-weight:600; color:var(--text);
           border-left:3px solid var(--accent); padding-left:8px; }
.note { color:var(--muted); font-size:11px; margin:0 0 6px; line-height:1.4; }
.kpi-row { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 10px; }
.kpi { border:1px solid var(--line); border-top:3px solid var(--accent); padding:7px 10px; min-width:132px; background:var(--panel-hi); }
.kpi .lbl { font-size:9px; color:var(--muted); text-transform:uppercase; letter-spacing:.45px; font-weight:600; }
.kpi .val { font:600 16px/1.2 var(--mono); margin-top:2px; }
label.inline { color:var(--muted); font-size:11px; display:flex; align-items:center; gap:6px;
               margin:0 0 4px; }
label.inline input { width:13px; height:13px; }
.tab { display:none; }
.tab.on { display:block; }
table.recon { border-collapse:collapse; width:100%; font-size:11px; }
table.recon th, table.recon td { padding:4px 6px; text-align:right; }
table.recon th:first-child, table.recon td:first-child { text-align:left; }
.chart { min-height:360px; width:100%; }
"""

_JS_COLORS = ", ".join(f'"{c}"' for c in MATLAB)
_JS_COUNTRY = ", ".join(f'"{k}":"{v}"' for k, v in COUNTRY_COLOR.items())

_T = theme.TOKENS
JS_THEME = theme.js_theme() + rf"""
const T = {{
  pos: THEME.pos,
  neg: THEME.neg,
  muted: THEME.tokens.faint,
  accent: THEME.accent,
  series: [{_JS_COLORS}],
  country: {{{_JS_COUNTRY}}},
  layout: Object.assign(JSON.parse(JSON.stringify(THEME.layout)), {{
    autosize:true,
    hovermode:"x unified",
    margin:{{l:58,r:14,t:38,b:52}},
    height:380,
    legend:{{orientation:"h", y:-0.20, font:{{size:10, color:THEME.tokens["chart-legend"]}}}},
  }})
}};
const layout = extra => Object.assign(JSON.parse(JSON.stringify(T.layout)), extra || {{}});
const countryColour = c => T.country[c] || T.series[0];
const seriesColour = i => T.series[i % T.series.length];
function resizeCharts() {{
  if (typeof Plotly === "undefined" || !Plotly.Plots) return;
  document.querySelectorAll(".js-plotly-plot").forEach(el => {{
    try {{ Plotly.Plots.resize(el); }} catch (e) {{}}
  }});
}}
window.addEventListener("resize", () => {{
  if (window.__issResize) clearTimeout(window.__issResize);
  window.__issResize = setTimeout(resizeCharts, 120);
}});
if (typeof ResizeObserver !== "undefined") {{
  let lastW = 0;
  new ResizeObserver(entries => {{
    const w = Math.round(entries[0].contentRect.width);
    if (w === lastW) return;
    lastW = w;
    if (window.__issResize) clearTimeout(window.__issResize);
    window.__issResize = setTimeout(resizeCharts, 120);
  }}).observe(document.documentElement);
}}
"""
