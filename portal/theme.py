"""Desk visual theme — the one colour scheme every board shares.

MATLAB figure look: flat grey page, white panels, 1px grey rules, Helvetica for text and a
monospace face for numbers, MATLAB series order, boxed Plotly axes, blue = positive / red =
negative. No gradients, shadows, glow or motion. Tables are dense monospaced numerics with
sticky headers and heat-coloured cells.

Light only. Builders consume:

* ``page_css()``        — ``@import`` + ``:root`` tokens + base page / card / table CSS
* ``css_vars()``        — the ``:root`` token block alone
* ``PLOTLY_TEMPLATE``   — registered Plotly template name (``use_plotly()`` makes it default)
* ``plotly_layout()``   — plain layout dict for JS-rendered charts; ``js_theme()`` wraps it
* ``heat_bg(z)``        — cell background for a signed score
* ``use_mpl()``         — matplotlib rcParams for PNG exhibits

``site/assets/theme.css`` is generated from this module:
``PYTHONPATH=. .venv/bin/python -m portal.theme``.
"""

from __future__ import annotations

import json
from pathlib import Path

# MATLAB R2014b+ default line order — the house colorway.
SERIES: tuple[str, ...] = (
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # cyan
    "#A2142F",  # dark red
    "#000000",  # black
)

FONT_SANS = '"Helvetica Neue",Helvetica,Arial,sans-serif'
FONT_MONO = '"IBM Plex Mono",Menlo,Consolas,ui-monospace,monospace'
PLOT_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
FONT_IMPORT = '@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap");'

TOKENS: dict[str, str] = {
    # surfaces
    "bg": "#ececec",          # page: MATLAB figure grey, flat
    "bg-grad-0": "#ececec",   # kept for old consumers; flat on purpose
    "bg-grad-1": "#ececec",
    "bg-grad-2": "#ececec",
    "panel": "#f5f5f5",       # header / rail / control strips
    "rail-0": "#f5f5f5",
    "rail-1": "#f5f5f5",
    "rail-2": "#f5f5f5",
    "panel-hi": "#ffffff",    # cards, plot area
    "plot-bg": "#ffffff",
    "th-bg": "#f3f3f3",
    "code-bg": "#f3f3f3",
    # lines
    "line": "#c4c4c4",
    "line-dark": "#8c8c8c",
    "grid": "#d9d9d9",
    "hair": "#e8e8e8",
    # text
    "text": "#111111",
    "muted": "#555555",
    "faint": "#777777",
    "dim": "#a0a0a0",
    # interaction
    "accent": "#0072BD",
    "link": "#0060a0",
    "hover": "#e8eef5",
    "row-hover": "#f2f6fa",
    "sel": "#d9e6f2",
    "extreme": "#fff8e6",
    # semantics (sign convention)
    "pos": SERIES[0],
    "neg": SERIES[6],
    "warn": SERIES[1],
    "flag": SERIES[2],
    "ok": "#2e7d32",
    "ok-bg": "#edf5ee",
    "fail": "#9b1c1c",
    "fail-bg": "#fee2e2",
    "warn-bg": "#fef3c7",
    "idle": "#4a5563",
    # chart chrome (JS_THEME)
    "chart-grid": "#d8d8d8",
    "chart-zero": "#888888",
    "chart-axis": "#333333",
    "chart-text": "#222222",
    "chart-legend": "#444444",
}

# Diverging heat, negative → positive. Endpoints are the pos/neg tokens.
HEAT: tuple[str, ...] = ("#A2142F", "#c75a6c", "#e8b4bd", "#fafbfc", "#b3d3ea", "#4d9ad0", "#0072BD")
HEAT_COLORSCALE: list[list] = [[i / (len(HEAT) - 1), c] for i, c in enumerate(HEAT)]
# Reversed: red for high (rich / hot), blue for low.
HEAT_COLORSCALE_R: list[list] = [[i / (len(HEAT) - 1), c] for i, c in enumerate(reversed(HEAT))]
# Sequential magnitude scale (counts, |corr|, shares).
SEQ: tuple[str, ...] = ("#fafbfc", "#c5d8e6", "#7fb0cf", "#2f7ba8", "#155e75")
SEQ_COLORSCALE: list[list] = [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]

_POS_RGB = (0, 114, 189)
_NEG_RGB = (162, 20, 47)


def rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def heat_bg(z: float | None, zmax: float = 3.0, max_alpha: float = 0.55) -> str:
    """Cell background for a signed value: blue positive, red negative, alpha ∝ |z|/zmax."""
    if z is None or z != z:
        return "transparent"
    a = min(abs(z) / zmax, 1.0) * max_alpha
    r, g, b = _POS_RGB if z >= 0 else _NEG_RGB
    return f"rgba({r},{g},{b},{a:.3f})"


def css_vars(selector: str = ":root") -> str:
    lines = [f"  --{k}:{v};" for k, v in TOKENS.items()]
    lines += [f"  --s{i + 1}:{c};" for i, c in enumerate(SERIES)]
    lines += [f"  --heat-{k}:{c};" for k, c in zip(("n3", "n2", "n1", "0", "p1", "p2", "p3"), HEAT)]
    lines += [f"  --sans:{FONT_SANS};", f"  --mono:{FONT_MONO};"]
    return selector + " {\n" + "\n".join(lines) + "\n}\n"


BASE_CSS = """
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:12px/1.45 var(--sans); font-variant-numeric:tabular-nums; -webkit-font-smoothing:antialiased; }
a { color:var(--link); }
header { background:var(--panel); border-bottom:1px solid var(--line); color:var(--text); }
h1 { font-family:var(--sans); font-size:13px; font-weight:700; letter-spacing:0; color:var(--text); }
.sub { color:var(--muted); }
.card, .panel { background:var(--panel-hi); border:1px solid var(--line); border-radius:0; box-shadow:none; }
.card h3, .panel h3 { color:var(--text); font-weight:700; }
.note { color:var(--muted); }
code { background:var(--code-bg); font-family:var(--mono); }
select, input, button { font-family:inherit; }
select, input { background:var(--panel-hi); color:var(--text); border:1px solid var(--line); border-radius:0; }
.pos, td.pos { color:var(--pos); }
.neg, td.neg { color:var(--neg); }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--hair); }
th { position:sticky; top:0; z-index:1; background:var(--th-bg); color:var(--muted); font-weight:600;
     border-bottom:1px solid var(--line); }
td { font-family:var(--mono); font-size:11px; color:var(--text); }
td:first-child, th:first-child { font-family:var(--sans); }
tbody tr:hover td { background:var(--row-hover); }
tr.extreme td { background:var(--extreme); }
td.hm, td.heat { text-align:center; }
td.na, tr.dim td { color:var(--dim); }
"""

# Class-level table spec for builders that want the full desk look (opt-in via class="dense").
DENSE_TABLE_CSS = """
table.dense { width:100%; border-collapse:collapse; font:11px/1.3 var(--mono); }
table.dense th, table.dense td { padding:4px 9px; border-bottom:1px solid var(--hair); text-align:right; white-space:nowrap; }
table.dense th { position:sticky; top:0; background:var(--th-bg); color:var(--muted); font-weight:600;
                 border-bottom:1px solid var(--line); z-index:1; }
table.dense td:first-child, table.dense th:first-child { text-align:left; font-weight:600; }
table.dense tbody tr:hover { background:var(--row-hover); }
"""


def page_css(*, fonts: bool = True) -> str:
    """Everything a board needs in one ``<style>`` prefix. Put it first: ``@import`` must lead."""
    return (FONT_IMPORT + "\n" if fonts else "") + css_vars() + BASE_CSS + DENSE_TABLE_CSS


# ---------------------------------------------------------------- Plotly

PLOTLY_TEMPLATE = "desk"


def axis_style() -> dict:
    t = TOKENS
    return dict(
        showgrid=True, gridcolor=t["chart-grid"], gridwidth=1,
        zeroline=True, zerolinecolor=t["chart-zero"], zerolinewidth=1,
        showline=True, mirror=True, linecolor=t["chart-axis"], linewidth=1,
        tickcolor=t["chart-axis"], tickfont=dict(size=10, color=t["chart-text"]),
        title=dict(font=dict(size=11, color=t["chart-text"])),
        automargin=True,
    )


def plotly_layout(**overrides) -> dict:
    """Plain layout dict (JSON-serialisable) — the ``JS_THEME.layout`` in Python."""
    t = TOKENS
    base = dict(
        paper_bgcolor=t["panel-hi"], plot_bgcolor=t["plot-bg"],
        font=dict(family=PLOT_FONT, size=11, color=t["chart-text"]),
        title=dict(font=dict(family=PLOT_FONT, size=12, color=t["text"])),
        colorway=list(SERIES),
        legend=dict(font=dict(size=10, color=t["chart-legend"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=t["panel-hi"], bordercolor=t["line"],
                        font=dict(family=PLOT_FONT, size=11, color=t["text"])),
        xaxis=axis_style(), yaxis=axis_style(),
        colorscale=dict(diverging=HEAT_COLORSCALE, sequential=SEQ_COLORSCALE),
    )
    base.update(overrides)
    return base


def plotly_template():
    import plotly.graph_objects as go

    lay = plotly_layout()
    tmpl = go.layout.Template(layout=lay)
    tmpl.layout.annotationdefaults = dict(font=dict(size=10, color=TOKENS["chart-text"]))
    tmpl.data.heatmap = [go.Heatmap(colorscale=HEAT_COLORSCALE)]
    tmpl.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    return tmpl


def use_plotly(default: bool = True) -> str:
    """Register the ``desk`` template (idempotent); optionally make it ``pio.templates.default``."""
    import plotly.io as pio

    if PLOTLY_TEMPLATE not in pio.templates:
        pio.templates[PLOTLY_TEMPLATE] = plotly_template()
    if default:
        pio.templates.default = PLOTLY_TEMPLATE
    return PLOTLY_TEMPLATE


def js_theme(name: str = "THEME") -> str:
    """``const THEME = {...}`` for JS-built charts: tokens, series, heat, layout."""
    payload = {
        "tokens": TOKENS, "series": list(SERIES), "heat": list(HEAT),
        "heatScale": HEAT_COLORSCALE, "heatScaleR": HEAT_COLORSCALE_R, "seqScale": SEQ_COLORSCALE,
        "pos": TOKENS["pos"], "neg": TOKENS["neg"], "accent": TOKENS["accent"], "muted": TOKENS["muted"],
        "text": TOKENS["text"], "grid": TOKENS["chart-grid"], "zero": TOKENS["chart-zero"],
        "font": PLOT_FONT, "layout": plotly_layout(),
        "tint": {k: rgba(TOKENS[k], a) for k, a in (("pos", 0.15), ("neg", 0.2), ("accent", 0.07))},
    }
    return f"const {name} = {json.dumps(payload, separators=(',', ':'))};\n{PLOTLY_PIN_JS}"


# With responsive:true, Plotly's window-resize handler relayouts {autosize:true}, which drops a
# fixed layout.height and re-sizes from the container; an unsized container collapses the plot to
# ~14px and its SVG spills over the next card. Pinning the container to layout.height prevents it.
PLOTLY_PIN_JS = """(function(){
  var P = window.Plotly; if (!P || P.__pinHeights) return; P.__pinHeights = true;
  function pin(gd){ var h = gd && gd.layout && gd.layout.height;
    if (h && (!gd.style.height || gd.dataset.pinH)) { gd.style.height = h + "px"; gd.dataset.pinH = "1"; } return gd; }
  ["newPlot", "react"].forEach(function(fn){ var orig = P[fn]; if (!orig) return;
    P[fn] = function(){ return orig.apply(this, arguments).then(pin); }; });
})();"""


# ---------------------------------------------------------------- matplotlib

MPL_RC: dict = {
    "figure.facecolor": TOKENS["panel-hi"],
    "axes.facecolor": TOKENS["plot-bg"],
    "savefig.facecolor": TOKENS["panel-hi"],
    "axes.edgecolor": TOKENS["chart-axis"],
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": TOKENS["chart-grid"],
    "grid.linewidth": 0.6,
    "axes.labelcolor": TOKENS["chart-text"],
    "axes.titlecolor": TOKENS["text"],
    "axes.titleweight": "semibold",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.prop_cycle": None,  # filled in use_mpl (needs cycler)
    "xtick.color": TOKENS["chart-axis"],
    "ytick.color": TOKENS["chart-axis"],
    "xtick.labelcolor": TOKENS["chart-text"],
    "ytick.labelcolor": TOKENS["chart-text"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "text.color": TOKENS["text"],
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
}


def mpl_rc() -> dict:
    from cycler import cycler

    rc = dict(MPL_RC)
    rc["axes.prop_cycle"] = cycler(color=list(SERIES))
    return rc


def use_mpl() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(mpl_rc())


# ---------------------------------------------------------------- site/assets/theme.css

THEME_CSS_PATH = Path(__file__).resolve().parents[1] / "site" / "assets" / "theme.css"


def site_css() -> str:
    head = ("/* GENERATED by portal/theme.py — do not edit. "
            "Regenerate: PYTHONPATH=. .venv/bin/python -m portal.theme */\n")
    return head + FONT_IMPORT + "\n" + css_vars() + DENSE_TABLE_CSS + """
.heat-pos-1 { background:rgba(0,114,189,.12); } .heat-pos-2 { background:rgba(0,114,189,.28); }
.heat-pos-3 { background:rgba(0,114,189,.50); }
.heat-neg-1 { background:rgba(162,20,47,.12); } .heat-neg-2 { background:rgba(162,20,47,.28); }
.heat-neg-3 { background:rgba(162,20,47,.50); }
"""


def write_site_css(path: Path = THEME_CSS_PATH) -> Path:
    path.write_text(site_css(), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_site_css())
