"""Render the Debt & supply tab (``sovereign_cash/debt_supply_dashboard.html``).

Dalio's "numbers behind the bond market" per country: debt, net issuance, rollovers, gross issuance and
debt service against revenue from 2015 to three forward years, the cross-country snapshot, stress
(+100bp, recession, holders rolling 90%), supply-demand sensitivities (private net / DV01 / rollovers /
bill share / CB / reinvestment / 15y→10s30s / plan gap), holders, and the panel test. Reads
``sovereign_cash.debt.build``, ``sovereign_cash.debt.sensitivities`` and ``sovereign_cash.debt.signal``.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.page
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portal.theme import FONT_IMPORT, PLOT_FONT, SERIES, TOKENS, axis_style, plotly_layout
from sovereign_cash.contract import MINUS
from sovereign_cash.investor_page import CSS, PLOTLY

HERE = Path(__file__).resolve().parent
DERIVED = HERE.parent / "derived"
OUT = HERE.parent / "debt_supply_dashboard.html"
ORDER = ("us", "uk", "de", "fr", "it", "es", "jp", "se", "no")
NAME = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy", "es": "Spain",
        "jp": "Japan", "se": "Sweden", "no": "Norway"}
SYM = {"USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥", "SEK": "SEK ", "NOK": "NOK "}
# Same MATLAB country order as the issuance / DV01 board.
COL = {"uk": SERIES[0], "de": SERIES[1], "fr": SERIES[2], "it": SERIES[3],
       "us": SERIES[4], "jp": SERIES[5], "se": SERIES[6], "no": SERIES[7],
       "es": SERIES[2]}
C_CURVE, C_UP, C_DOWN = SERIES[0], SERIES[6], SERIES[4]
C_NG, C_INFL, C_BOTH, C_AUTH = SERIES[5], SERIES[3], "#e67e22", SERIES[7]
XLAB = {"priv_tr": "Private net supply, trailing 12m", "priv_fw": "Private net supply, next 12m (foresight)",
        "cb_tr": "CB net purchases, trailing 12m", "roll": "Rollovers", "bills": "Bill share of stock",
        "int_rev": "Interest / revenue"}
YLAB = {"y10": "10y", "tp10": "10y − 3m", "s1030": "10s30s"}
SHOW_YEARS = range(2023, 2030)
EXTRA_CSS = """
body { background: #ececec; color: #111111; font: 12px/1.45 "Helvetica Neue", Helvetica, Arial, sans-serif; }
h1, h2, h3 { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; letter-spacing: 0; text-transform: none; color: #111111; }
h1 { font-size: 15px; font-weight: 600; }
h2 { font-size: 13px; font-weight: 600; color: #111111; border-bottom: 1px solid #c4c4c4; }
h3 { font: 600 12px/1.3 "Helvetica Neue", Helvetica, Arial, sans-serif; margin: 18px 0 4px; }
.chart { background: #fff; border: 1px solid #c4c4c4; }
table { display: block; max-width: 100%; overflow-x: auto; width: auto;
        font-family: "IBM Plex Mono", Menlo, Consolas, ui-monospace, monospace; font-size: 11px; }
th { background: #f3f3f3; }
.k div { background: #fff; border: 1px solid #c4c4c4; border-top: 3px solid #0072BD; }
.grid { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: start; }
.grid > div { min-width: 0; }
th.f, td.f { background: #fafbfd; color: #333333; }
.chart.s { height: 300px; }
"""


def _n(v, d=1, pct=False, sign=False) -> str:
    if v is None or (isinstance(v, (float, np.floating, int)) and not np.isfinite(v)):
        return "—"
    v = round(float(v), d) + 0.0
    s = f"{v:+,.{d}f}" if sign else f"{v:,.{d}f}"
    return s.replace("-", MINUS) + ("%" if pct else "")


def _money(v, ccy: str) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    sym, unit, d = SYM[ccy], "bn", 0
    if ccy == "JPY" or abs(v) >= 10_000:
        v, unit, d = v / 1000.0, "tn", 1
    neg = MINUS if v < 0 and round(abs(v), d) > 0 else ""
    return f"{sym}{neg}{abs(v):,.{d}f}{unit}"


def _usd(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"${v / 1000:,.2f}tn" if abs(v) >= 1000 else f"${v:,.0f}bn".replace("-", MINUS)


def _merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _axis(spec: dict | None = None) -> dict:
    ax = axis_style()
    if spec:
        _merge(ax, spec)
    return ax


def _fig(data, title, ytitle="", extra=None) -> dict:
    lay = plotly_layout(
        title={"text": title, "font": {"family": PLOT_FONT, "size": 12, "color": TOKENS["text"]}},
        margin={"l": 58, "r": 16, "t": 40, "b": 52},
        legend={"orientation": "h", "y": -0.2, "font": {"size": 10, "color": TOKENS["chart-legend"]},
                "bgcolor": "rgba(0,0,0,0)"},
        hovermode="x unified",
    )
    if ytitle:
        lay["yaxis"]["title"]["text"] = ytitle
    _merge(lay, extra or {})
    for k, v in list(lay.items()):
        if isinstance(v, dict) and (k == "xaxis" or k == "yaxis" or k.startswith(("xaxis", "yaxis"))):
            lay[k] = _axis(v)
    return {"data": data, "layout": lay}


def _dropdown(options: list[tuple[str, dict, str]]) -> dict:
    """One chart, several series sets. ``options`` is (label, figure, y-axis title)."""
    sets: list[tuple[int, dict]] = []
    for i, (_, fig, _) in enumerate(options):
        for tr in fig["data"]:
            row = dict(tr)
            row["visible"] = i == 0
            sets.append((i, row))
    data = [tr for _, tr in sets]
    buttons = []
    for i, (lab, _, unit) in enumerate(options):
        buttons.append({
            "label": lab, "method": "update",
            "args": [{"visible": [j == i for j, _ in sets]}, {"yaxis.title.text": unit}],
        })
    base = options[0][1]
    fig = _fig(data, "", options[0][2], {
        "updatemenus": [{
            "type": "dropdown", "direction": "down", "x": 0, "xanchor": "left", "y": 1.12, "yanchor": "bottom",
            "showactive": True, "buttons": buttons,
            "bgcolor": "#fff", "bordercolor": "#c4c4c4", "font": {"size": 12, "family": PLOT_FONT},
        }],
        "margin": {"l": 52, "r": 16, "t": 8, "b": 36},
        "shapes": base["layout"].get("shapes", []),
    })
    return fig


def _channels(rp: pd.DataFrame, gdp: pd.DataFrame) -> dict:
    """2029 central interest / published GDP. Each country is a tight pair: today, then +100bp."""
    parts = (("ch_locked", "Set coupons", SERIES[7]),
             ("ch_new", "New fixed", SERIES[0]),
             ("ch_bills", "Bills", SERIES[5]),
             ("ch_linker", "New linkers + uplift", SERIES[1]),
             ("ch_floater", "Floaters", SERIES[3]),
             ("ch_other", "Frozen gap", TOKENS["faint"]))
    ccs = [c for c in ("us", "uk", "de", "fr", "it", "es", "jp") if c in set(rp["cc"])]
    d = rp[(rp["scenario"] == "baseline") & (rp["year"] == rp["year"].max()) & rp["cc"].isin(ccs)]
    if d.empty or "ch_locked" not in d.columns:
        return _fig([], "2029 interest / GDP by channel", "% GDP")
    xs, ticks, notes = [], [], []
    slot = 0.0
    for cc in ccs:
        xs += [slot, slot + 0.72]
        ticks += ["today", "+100bp"]
        notes.append({"x": slot + 0.36, "y": 1.04, "xref": "x", "yref": "paper",
                      "text": NAME[cc], "showarrow": False,
                      "font": {"size": 11, "family": PLOT_FONT, "color": TOKENS["chart-text"]}})
        slot += 2.2
    data = []
    for col, lab, color in parts:
        y = []
        for cc in ccs:
            for case in ("current", "shock"):
                row = d[(d["cc"] == cc) & (d["case"] == case)]
                y.append(round(100 * float(row[col].iloc[0]) / float(gdp.at[int(row["year"].iloc[0]), cc]), 2) if len(row) else None)
        data.append({"type": "bar", "name": lab, "x": xs, "y": y, "width": 0.64,
                     "marker": {"color": color}})
    return _fig(data, "2029 interest / GDP, by what pays it", "% GDP",
                {"barmode": "relative", "bargap": 0,
                 "xaxis": {"tickmode": "array", "tickvals": xs, "ticktext": ticks,
                           "tickfont": {"size": 11}, "range": [-0.5, slot - 0.7]},
                 "margin": {"l": 46, "r": 12, "t": 48, "b": 78},
                 "annotations": notes,
                 "legend": {"orientation": "h", "y": -0.32}})


def _published_gdp(ca: pd.DataFrame) -> pd.DataFrame:
    """Nominal GDP, local bn. Outturn through the last history year, then the published forecast.

    US: CBO. UK: OECD Economic Outlook. Germany, France, Italy: the Commission's AMECO.
    Japan: Cabinet Office baseline. Years after that vintage continue on IMF WEO nominal growth.
    """
    y0 = int(ca.loc[~ca["forecast"], "year"].max())
    base = ca.pivot(index="year", columns="cc", values="gdp_bn").sort_index()
    growth = base / base.shift(1)
    pub = pd.DataFrame(index=base.index, columns=base.columns, dtype=float)

    def _overlay(cc: str, series: pd.Series) -> None:
        for y, v in series.dropna().items():
            y = int(y)
            if y > y0 and y in pub.index and np.isfinite(v) and v > 0:
                pub.at[y, cc] = float(v)

    oecd = pd.read_parquet(HERE.parent / "data" / "debt" / "oecd.parquet")
    ameco = pd.read_parquet(HERE.parent / "data" / "debt" / "ameco.parquet")
    cbo = pd.read_parquet(HERE.parent / "data" / "debt" / "cbo.parquet")
    cao = pd.read_parquet(HERE.parent / "data" / "debt" / "cao.parquet")
    for cc in ("us", "uk", "de", "fr", "it", "es", "jp"):
        o = oecd[(oecd["country"] == cc) & (oecd["var"] == "gdp_lcu_bn")].set_index("year")["value"]
        _overlay(cc, o)
    for cc in ("de", "fr", "it"):
        a = ameco[(ameco["country"] == cc) & (ameco["var"] == "gdp_lcu_bn")].set_index("year")["value"]
        _overlay(cc, a)
    us = cbo.loc[cbo["var"].eq("gdp_lcu_bn"), ["year", "value"]].set_index("year")["value"]
    if us.dropna().max() < 1000:
        us = us * 1000  # CBO stores trillions in the "USD bn" field
    _overlay("us", us)
    jp = cao[(cao["source"] == "cao_baseline") & (cao["var"] == "gdp_lcu_bn")].set_index("year")["value"]
    _overlay("jp", jp)
    out = base.copy()
    for cc in out.columns:
        for y in out.index:
            if int(y) <= y0:
                continue
            if np.isfinite(pub.at[y, cc]):
                out.at[y, cc] = pub.at[y, cc]
            elif int(y) - 1 in out.index and np.isfinite(growth.at[y, cc]):
                out.at[y, cc] = out.at[y - 1, cc] * growth.at[y, cc]
    return out


def _auth_paths(ca: pd.DataFrame, rp: pd.DataFrame, gdp: pd.DataFrame) -> dict:
    """Central interest. Curve, ±100bp, +2pp nominal GDP, +2pp inflation, both, authority.

    The dropdown switches the denominator: published GDP, or central revenue (the +2pp nominal
    case scales that denominator the same way)."""
    from sovereign_cash.debt.cashflow import components

    ccs = [c for c in ("us", "uk", "de", "fr", "it", "es", "jp") if (ca["cc"] == c).any()]
    y0 = int(ca.loc[~ca["forecast"], "year"].max())
    y1 = int(ca["year"].max())
    comp = components(y0, list(range(y0, y1 + 1)), lambda cc, y: 0.0).set_index(["cc", "year"])
    # 3×2 fits six peers; Japan makes seven — keep two rows (4×2) so domains stay in [0, 1].
    # A fixed 3×2 left the 7th axis at y∈[-0.48,-0.1]; Plotly then collapsed the left column.
    n = len(ccs)
    ncols = 3 if n <= 6 else 4
    nrows = max(1, (n + ncols - 1) // ncols)
    gap_x, gap_y = 0.06, 0.16
    width = (1 - gap_x * max(ncols - 1, 0)) / ncols
    height = (1 - gap_y * max(nrows - 1, 0) - 0.08) / nrows
    data, annotations, shapes = [], [], []
    layout: dict = plotly_layout(
        margin={"l": 52, "r": 12, "t": 72, "b": 28},
        legend={"orientation": "h", "y": 1.12, "x": 0,
                "font": {"size": 10, "color": TOKENS["chart-legend"]}, "bgcolor": "rgba(0,0,0,0)"},
        hovermode="closest",
    )
    layout["annotations"] = annotations
    layout["shapes"] = shapes

    def _trace(i, name, xs, ys, color, ax, ay, width=1.6, dash=None):
        line = {"color": color, "width": width}
        if dash:
            line["dash"] = dash
        data.append({
            "type": "scatter", "mode": "lines", "name": name, "x": xs, "y": ys,
            "line": line, "xaxis": ax, "yaxis": ay, "showlegend": i == 0,
        })

    for i, cc in enumerate(ccs):
        col, row = i % ncols, i // ncols
        x0 = col * (width + gap_x)
        y_bot = 1 - 0.02 - (row + 1) * height - row * gap_y
        ax = "x" if i == 0 else f"x{i + 1}"
        ay = "y" if i == 0 else f"y{i + 1}"
        d = ca[ca["cc"] == cc].set_index("year").sort_index()

        def bill(y, src):
            if y == y0 or src is d:
                return float(d.at[y, "interest_bn"])
            return float(src.at[y, "interest_bn"])

        def scale(y):
            return 1.02 ** (int(y) - y0)

        def lvl(y, interest, scale_n=1.0, revenue=False):
            if revenue:
                den = float(d.at[y, "revenue_bn"]) * scale_n if y in d.index else np.nan
            else:
                den = float(gdp.at[y, cc]) * scale_n
            if not np.isfinite(den) or den <= 0 or not np.isfinite(interest):
                return None
            return round(100.0 * float(interest) / den, 2)

        years = [int(y) for y in d.index if int(y) >= 2015 and np.isfinite(d.at[y, "interest_bn"])]
        fwd = [y for y in years if y >= y0]
        sh = rp[(rp["cc"] == cc) & (rp["scenario"] == "baseline") & (rp["case"] == "shock")].set_index("year")
        ez = rp[(rp["cc"] == cc) & (rp["scenario"] == "recession") & (rp["case"] == "ease")].set_index("year")
        ayrs = [y for y in years if y >= y0 and np.isfinite(d.at[y, "interest_authority_bn"])]
        infl_int = []
        for y in fwd:
            extra = 0.0
            if y > y0:
                rowc = comp.loc[(cc, y)]
                extra = float(rowc.linker_face_bn) * float(rowc.index_ratio) * 0.02
            infl_int.append(bill(y, d) + extra)
        specs = (
            ("Curve", years, [bill(y, d) for y in years], 1.0, C_CURVE, 2, None),
            ("+100bp", fwd, [bill(y, sh) for y in fwd], 1.0, C_UP, 1.6, None),
            ("−100bp", fwd, [bill(y, ez) for y in fwd], 1.0, C_DOWN, 1.6, None),
            ("+2pp nominal GDP", fwd, [bill(y, d) for y in fwd], None, C_NG, 1.4, "dash"),
            ("+2pp inflation, yields unchanged", fwd, infl_int, None, C_INFL, 1.4, "dot"),
            ("−100bp and +2pp nominal GDP", fwd, [bill(y, ez) for y in fwd], None, C_BOTH, 2, None),
        )
        for revenue, visible in ((False, True), (True, False)):
            for name, xs, ints, fixed, color, w, dash in specs:
                ys = [lvl(y, iv, fixed if fixed is not None else scale(y), revenue) for y, iv in zip(xs, ints)]
                _trace(i, name, xs, ys, color, ax, ay, w, dash)
                data[-1]["visible"] = visible
                data[-1]["showlegend"] = visible and i == 0
            auth_y = []
            for y in ayrs:
                if revenue and np.isfinite(d.at[y, "revenue_authority_bn"]) and d.at[y, "revenue_authority_bn"] > 0:
                    auth_y.append(round(100.0 * float(d.at[y, "interest_authority_bn"]) / float(d.at[y, "revenue_authority_bn"]), 2))
                elif not revenue:
                    auth_y.append(lvl(y, d.at[y, "interest_authority_bn"]))
                else:
                    auth_y.append(None)
            _trace(i, "Authority", ayrs, auth_y, C_AUTH, ax, ay, 1.3, "dash")
            data[-1]["visible"] = visible
            data[-1]["showlegend"] = visible and i == 0
        layout["xaxis" if i == 0 else f"xaxis{i + 1}"] = _axis({
            "domain": [x0, x0 + width], "anchor": ay, "dtick": 3,
        })
        layout["yaxis" if i == 0 else f"yaxis{i + 1}"] = _axis({
            "domain": [y_bot, y_bot + height], "anchor": ax,
            "title": {"text": "% of GDP"},
        })
        annotations.append({
            "text": NAME[cc], "x": x0 + width / 2, "xref": "paper", "y": y_bot + height, "yref": "paper",
            "showarrow": False, "yanchor": "bottom",
            "font": {"size": 12, "family": PLOT_FONT, "color": TOKENS["text"]},
        })
        shapes.append({
            "type": "line", "x0": y0 + 0.5, "x1": y0 + 0.5, "y0": 0, "y1": 1,
            "xref": ax, "yref": f"{ay} domain", "line": {"color": TOKENS["chart-zero"], "width": 1},
        })
    per = 7  # six scenarios plus the authority line, twice per country
    ypatch = {("yaxis" if i == 0 else f"yaxis{i + 1}") + ".title.text": "% of GDP" for i in range(len(ccs))}
    def _menu(label_gdp, label_rev):
        def _half(revenue: bool) -> list[bool]:
            return [((i % (per * 2)) >= per) if revenue else ((i % (per * 2)) < per) for i in range(len(data))]
        vis_gdp, vis_rev = _half(False), _half(True)
        show = [bool(data[i].get("showlegend")) or bool(data[i - per].get("showlegend")) for i in range(len(data))]
        def args(vis, unit):
            leg = [show[i] and vis[i] for i in range(len(data))]
            patch = {k: unit for k in ypatch}
            return [{"visible": vis, "showlegend": leg}, patch]
        return {
            "type": "dropdown", "direction": "down", "x": 0, "xanchor": "left", "y": 1.14, "yanchor": "bottom",
            "showactive": True, "bgcolor": "#fff", "bordercolor": "#c4c4c4",
            "font": {"size": 12, "family": PLOT_FONT},
            "buttons": [
                {"label": label_gdp, "method": "update", "args": args(vis_gdp, "% of GDP")},
                {"label": label_rev, "method": "update", "args": args(vis_rev, "% of revenue")},
            ],
        }
    layout["updatemenus"] = [_menu("Central interest / GDP", "Central interest / revenue")]
    layout["margin"]["t"] = 36
    return {"data": data, "layout": layout}


def _context_charts(y0: int) -> dict[str, dict]:
    """Debt/GDP, the primary balance and nominal GDP growth. History solid, WEO forecast dotted."""
    weo = pd.read_parquet(HERE.parent / "data" / "debt" / "weo.parquet")
    wide = {cc: weo[weo["country"] == cc].pivot_table(index="year", columns="var", values="value")
            for cc in ("us", "uk", "de", "fr", "it", "jp")}
    fred = _us_federal_debt_pct()
    charts = {}

    def lines(series_for, title, ytitle):
        data, shapes = [], [{
            "type": "line", "x0": y0 + 0.5, "x1": y0 + 0.5, "y0": 0, "y1": 1,
            "xref": "x", "yref": "y domain", "line": {"color": TOKENS["chart-zero"], "width": 1},
        }]
        for cc in ("us", "uk", "de", "fr", "it", "jp"):
            s = series_for(cc).dropna()
            if s.empty:
                continue
            hist, fwd = s[s.index <= y0], s[s.index >= y0]
            for part, dash, show in ((hist, "solid", True), (fwd, "dot", False)):
                if part.empty:
                    continue
                data.append({
                    "type": "scatter", "mode": "lines", "name": NAME[cc], "legendgroup": cc,
                    "showlegend": show, "x": [int(y) for y in part.index],
                    "y": [round(float(v), 2) for v in part],
                    "line": {"color": COL[cc], "width": 1.8, "dash": dash},
                })
        return _fig(data, title, ytitle, {"shapes": shapes, "margin": {"l": 48, "r": 16, "t": 36, "b": 36}})

    def debt(cc):
        w = wide[cc]["debt_pct"]
        if cc == "us" and fred is not None and y0 in fred.index and y0 in w.index:
            base = float(fred.loc[y0])
            out = fred[fred.index <= y0].copy()
            for y, v in w[w.index > y0].items():
                out.loc[int(y)] = base * float(v) / float(w.loc[y0])
            return out.sort_index()
        return w

    def primary(cc):
        return wide[cc]["primary_pct"]

    def ngdp(cc):
        w = wide[cc]
        return ((1 + w["real_growth"] / 100) * (1 + w["cpi"] / 100) - 1) * 100

    charts["c_debt"] = lines(debt, "Debt / GDP", "% of GDP")
    charts["c_prim"] = lines(primary, "Primary balance", "% of GDP")
    charts["c_ngdp"] = lines(ngdp, "Nominal GDP growth", "%")
    return charts


def _us_federal_debt_pct(series: str = "GFDEGDQ188S"):
    """Annual mean of a FRED debt / GDP series (default GFDEGDQ188S = total public debt; the outlook note
    asks for FYGFGDQ188S = debt held by the public, the base net interest is paid on). None if the fetch
    fails; the chart then uses WEO."""
    import io
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read()
    except Exception:
        return None
    q = pd.read_csv(io.BytesIO(raw), parse_dates=["observation_date"])
    q["year"] = q["observation_date"].dt.year
    s = q.groupby("year")[series].mean()
    s.index = s.index.astype(int)
    return s.dropna()


def _gg_gaps(ca: pd.DataFrame, rp: pd.DataFrame) -> dict:
    """2029 GG interest / revenue, gap versus the bill on today's curve."""
    from sovereign_cash.debt.cashflow import components

    y0 = int(ca.loc[~ca["forecast"], "year"].max())
    y1 = 2029
    comp = components(y0, [y1], lambda cc, y: 0.0).set_index(["cc", "year"])
    order = ["us", "jp", "it", "fr", "uk", "de"]
    labels = [NAME[c] for c in order]
    cols = {
        "ng": ("+2pp nominal GDP", C_NG),
        "infl": ("+2pp inflation, yields unchanged", C_INFL),
        "ai": ("−100bp and +2pp nominal GDP", C_BOTH),
        "up": ("+100bp", C_UP),
    }
    vals = {k: [] for k in cols}
    k = 1.02 ** (y1 - y0)
    for cc in order:
        d = ca[ca["cc"] == cc].set_index("year")
        gap = float(d.at[y0, "gg_interest_bn"] - d.at[y0, "interest_bn"])
        rev = float(d.at[y1, "gg_revenue_bn"])
        base_i = float(d.at[y1, "interest_bn"])
        ease_i = float(rp[(rp["cc"] == cc) & (rp["scenario"] == "recession") & (rp["case"] == "ease") & (rp["year"] == y1)]["interest_bn"].iloc[0])
        shock_i = float(rp[(rp["cc"] == cc) & (rp["scenario"] == "baseline") & (rp["case"] == "shock") & (rp["year"] == y1)]["interest_bn"].iloc[0])
        row = comp.loc[(cc, y1)]
        extra = float(row.linker_face_bn) * float(row.index_ratio) * 0.02

        def ratio(interest, revenue):
            return 100.0 * (interest + gap) / revenue

        base = ratio(base_i, rev)
        vals["ng"].append(round(ratio(base_i, rev * k) - base, 2))
        vals["infl"].append(round(ratio(base_i + extra, rev * k) - base, 2))
        vals["ai"].append(round(ratio(ease_i, rev * k) - base, 2))
        vals["up"].append(round(ratio(shock_i, rev) - base, 2))
    data = [{"type": "bar", "orientation": "h", "name": name, "y": labels, "x": vals[key],
             "marker": {"color": color}} for key, (name, color) in cols.items()]
    return _fig(data, "2029 GG interest / revenue", "vs current curves", {
        "barmode": "group", "xaxis": {"title": {"text": "vs current curves"}},
        "yaxis": {"autorange": "reversed"}, "margin": {"l": 70, "r": 20, "t": 36, "b": 80},
        "legend": {"orientation": "h", "y": -0.25},
    })


def peer_rows(ca: pd.DataFrame, y_fc: int, ccs: tuple[str, ...] = ORDER) -> list[dict]:
    """General-government peer ranking: the comparison the IMF, the Commission and rating agencies use.

    Central-government ratios are not comparable across countries: central governments keep very different shares
    of general-government revenue (UK ~100%, Germany ~28%), so a central ratio flatters a centralised state. The
    US is shown on the federal basis (state debt carries no federal guarantee).
    """
    raw = HERE.parent / "data" / "debt"
    w = pd.read_parquet(raw / "weo.parquet").pivot_table(index=["country", "year"], columns="var", values="value")
    z = pd.read_parquet(raw / "gov_zeros.parquet")
    y10 = {}
    for _, r in z.iterrows():
        m = list(r["maturity"])
        if r["kind"] == "zero" and "_" not in r["cc"] and 10.0 in m:
            y10[r["cc"]] = (float(r["zero"][m.index(10.0)]), pd.Timestamp(r["asof"]))
    y_end = min(int(ca["year"].max()), y_fc + 4)
    out = []
    for cc in ccs:
        d = ca[ca["cc"] == cc].set_index("year").sort_index()
        if y_fc not in d.index or y_end not in d.index:
            continue
        col = "interest_pct_rev" if cc == "us" else "gg_interest_pct_rev"
        # A coupon book whose modelled stock turns negative (Sweden) also gives negative interest: use the
        # debt office's own interest path for those years, keeping the sub-central part, as Exhibit 2 does.
        bad = d["forecast"].astype(bool) & (d["debt_bn"] <= 0) & d["interest_authority_bn"].notna()
        if bad.any():
            d = d.copy()
            sub = d["gg_interest_bn"] - d["interest_bn"]
            d.loc[bad, "interest_bn"] = d.loc[bad, "interest_authority_bn"]
            d.loc[bad, "gg_interest_bn"] = d.loc[bad, "interest_bn"] + sub[bad]
            d.loc[bad, "interest_pct_rev"] = 100 * d.loc[bad, "interest_bn"] / d.loc[bad, "revenue_bn"]
            d.loc[bad, "gg_interest_pct_rev"] = 100 * d.loc[bad, "gg_interest_bn"] / d.loc[bad, "gg_revenue_bn"]
            d.loc[bad, "debt_bn"] = np.nan
        rg_imf = np.nan
        # Norway is a large net creditor: its gross-debt implicit rate is not a funding cost
        if cc != "no" and cc in w.index.get_level_values(0):
            e = w.loc[cc].sort_index()
            g = ((1 + e["real_growth"] / 100) * (1 + e["cpi"] / 100) - 1) * 100
            r = 100 * (e["interest_pct"] / e["debt_pct"].shift(1)) * (1 + g / 100)
            rg_imf = float((r - g).loc[y_fc:y_end].mean())
        rb = 100 * d["interest_bn"] / d["debt_bn"].shift(1)
        gb = 100 * d["gdp_bn"].pct_change()
        out.append({
            "cc": cc, "basis": "federal" if cc == "us" else "general government",
            "ir_now": float(d.at[y_fc, col]), "ir_end": float(d.at[y_end, col]),
            "central_now": float(d.at[y_fc, "interest_pct_rev"]),
            "rev_share": float(100 * d.at[y_fc, "revenue_bn"] / d.at[y_fc, "gg_revenue_bn"])
            if np.isfinite(d.at[y_fc, "gg_revenue_bn"]) and d.at[y_fc, "gg_revenue_bn"] else np.nan,
            "debt_gg": float(w.loc[(cc, y_fc), "debt_pct"]) if (cc, y_fc) in w.index else np.nan,
            "rg_imf": rg_imf, "rg_book": float((rb - gb).loc[y_fc:y_end].mean()),
            "y10": y10.get(cc, (np.nan, None))[0], "y_end": y_end})
    return sorted(out, key=lambda r: -r["ir_end"])


def _peer_table(ca: pd.DataFrame, y_fc: int, ccs: tuple[str, ...] = ORDER) -> str:
    rows = peer_rows(ca, y_fc, ccs)
    if not rows:
        return ""
    ye = rows[0]["y_end"]
    body = "".join(
        f"<tr><td class='l'><b>{NAME[r['cc']]}</b>{' (federal)' if r['cc'] == 'us' else ''}</td>"
        f"<td><b>{_n(r['ir_now'], 1)}</b></td><td><b>{_n(r['ir_end'], 1)}</b></td>"
        f"<td>{_n(r['debt_gg'], 0)}</td><td>{_n(r['rg_imf'], 2)}</td><td>{_n(r['rg_book'], 2)}</td>"
        f"<td>{_n(r['y10'], 2)}</td><td style='color:#6b7385'>{_n(r['central_now'], 1)}</td>"
        f"<td style='color:#6b7385'>{_n(min(r['rev_share'], 100.0), 0)}</td></tr>" for r in rows)
    return (f"<table><tr><th class='l'>Ranked by {ye} interest / revenue</th><th>Interest / revenue<br>{y_fc}, %</th>"
            f"<th>Interest / revenue<br>{ye}, %</th><th>GG debt<br>% GDP (IMF)</th><th>r − g {y_fc}–{str(ye)[2:]}<br>IMF, pp</th>"
            f"<th>r − g {y_fc}–{str(ye)[2:]}<br>our book at market rates, pp</th><th>10y yield<br>%</th>"
            f"<th>Central interest /<br>central revenue {y_fc}, %</th><th>Central share of<br>GG revenue, %</th></tr>{body}</table>"
            "<p class='note' style='color:#6b7385;font-size:12.5px'>Peer comparison is on general government (the IMF, "
            "Commission and rating-agency basis); the US on the federal basis. Central-government ratios (grey) measure "
            "the issuer's own cash burden and drive supply and rollover, but are not comparable across countries: "
            "central governments keep very different shares of general-government revenue (last column), so a central "
            "ratio flatters a centralised state such as the UK. r − g: IMF implicit rate (interest over last year's "
            "debt) less nominal GDP growth; our book: central interest from the bond-by-bond book at today's curve "
            "over last year's debt, less nominal GDP growth.</p>")


def _load():
    r = {n: pd.read_parquet(DERIVED / f"{n}.parquet") for n in
         ("debt_snapshot", "debt_reinvestment", "debt_repricing", "debt_history", "debt_annual", "debt_country_annual")}
    r["summary"] = json.loads((DERIVED / "debt_summary.json").read_text())
    p = DERIVED / "debt_signal.json"
    r["signal"] = json.loads(p.read_text()) if p.exists() else None
    return r


def _country_panel(cc: str, c: pd.DataFrame, ccy: str, y_fc: int) -> tuple[dict, str]:
    d = c[(c["cc"] == cc)].set_index("year").sort_index()
    yrs = [str(y) for y in d.index]
    op = [0.45 if y >= y_fc else 1.0 for y in d.index]
    fig = _fig([
        {"type": "bar", "name": "Net issuance", "x": yrs, "y": list(d["net_issuance_pct_gdp"].round(2)),
         "marker": {"color": SERIES[0], "opacity": op}},
        {"type": "bar", "name": "Rollovers", "x": yrs, "y": list(d["rollovers_pct_gdp"].round(2)),
         "marker": {"color": SERIES[5], "opacity": op}},
        {"type": "scatter", "mode": "lines+markers", "name": "Central debt service / revenue (rhs)", "x": yrs,
         "y": list(d["debt_service_pct_rev"].round(1)), "yaxis": "y2", "line": {"color": SERIES[6]}},
        {"type": "scatter", "mode": "lines+markers", "name": "Central interest / revenue (rhs)", "x": yrs,
         "y": list(d["interest_pct_rev"].round(1)), "yaxis": "y2", "line": {"color": SERIES[6], "dash": "dot"}},
        {"type": "scatter", "mode": "lines+markers", "name": "GG interest / GG revenue (rhs, peer basis)", "x": yrs,
         "y": list(d["gg_interest_pct_rev"].round(1)) if "gg_interest_pct_rev" in d.columns else [],
         "yaxis": "y2", "line": {"color": SERIES[2]}}],
        f"{NAME[cc]}: gross issuance = net + rollovers (% GDP); shaded = forecast", "% GDP",
        {"barmode": "stack", "yaxis2": {"overlaying": "y", "side": "right", "title": {"text": "% revenue"},
                                        "rangemode": "tozero", "showgrid": False}})
    rows = (("Debt", "debt_bn", "debt_pct_gdp"), ("Net issuance", "net_issuance_bn", "net_issuance_pct_gdp"),
            ("Rollovers", "rollovers_bn", "rollovers_pct_gdp"), ("Gross issuance", "gross_issuance_bn", "gross_issuance_pct_gdp"),
            ("Interest", "interest_bn", "interest_pct_gdp"))
    yy = [y for y in SHOW_YEARS if y in d.index]
    head = "".join(f"<th class='{'f' if y >= y_fc else ''}'>{y}{'f' if y >= y_fc else ''}</th>" for y in yy)
    body = ""
    for lab, a, p in rows:
        body += f"<tr><td class='l'>{lab}</td>" + "".join(
            f"<td class='{'f' if y >= y_fc else ''}'>{_money(d.at[y, a], ccy)}<br><span style='color:#6b7385'>"
            f"{_n(d.at[y, p], 2 if p == 'interest_pct_gdp' else 1)}%</span></td>" for y in yy) + "</tr>"
    for lab, col in (("Interest / revenue, general government", "gg_interest_pct_rev"),
                     ("Central interest / central revenue (issuer)", "interest_pct_rev"),
                     ("Central debt service / central revenue (issuer)", "debt_service_pct_rev")):
        if col not in d.columns:
            continue
        body += f"<tr><td class='l'><b>{lab}</b></td>" + "".join(
            f"<td class='{'f' if y >= y_fc else ''}'><b>{_n(d.at[y, col], 0 if col == 'debt_service_pct_rev' else 1, pct=True)}</b></td>"
            for y in yy) + "</tr>"
    return fig, f"<table><tr><th class='l'>{ccy} · % GDP</th>{head}</tr>{body}</table>"


def render() -> str:
    R = _load()
    s = R["debt_snapshot"].set_index("cc").reindex(ORDER)
    rv, rp, h, ca = R["debt_reinvestment"], R["debt_repricing"], R["debt_history"], R["debt_country_annual"]
    h["month"] = pd.to_datetime(h["month"])
    sig, vint = R["signal"], R["summary"]["vintages"]
    y_fc = int(ca.loc[ca["forecast"], "year"].min())
    charts: dict[str, dict] = {}

    def usd(cc, v):
        return v * s.at[cc, "fx"] if v is not None and np.isfinite(v) else np.nan

    def pg(cc, v):
        return 100 * v / s.at[cc, "gdp_bn"] if v is not None and np.isfinite(v) else np.nan

    u = s.loc["us"]
    kpi = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in (
        ("US marketable debt", _usd(u["marketable_bn"])), ("US rollovers, 12m", _usd(u["rollovers_bn"])),
        ("US net issuance, 12m", _usd(u["net_issuance_fwd"])), ("US gross issuance, 12m", _usd(u["gross_issuance_fwd"])),
        ("US federal interest / revenue", _n(u["interest_pct_rev"], 1, pct=True)),
        ("US federal debt service / revenue", _n(u["debt_service_pct_rev"], 0, pct=True)),
        ("US debt per household", f"${u['debt_per_household_usd'] / 1000:,.0f}k"),
        ("US effective / marginal rate", f"{u['eff_rate']:.2f}% / {u['marginal_rate']:.2f}%")))

    # ---- cross-country snapshot
    gg_now = {}
    for cc in ORDER:
        x = ca[(ca["cc"] == cc) & (ca["year"] == y_fc)]
        if not x.empty:
            gg_now[cc] = float(x.iloc[0]["interest_pct_rev" if cc == "us" else "gg_interest_pct_rev"])
    rows = []
    for cc in ORDER:
        r = s.loc[cc]
        net, gross, roll = r["net_issuance_fwd"], r["gross_issuance_fwd"], r["rollovers_bn"]
        priv = net - r["fw_cb_net"] if cc != "us" else net
        bs, norm = r["bill_share_fwd"], r["bill_share_norm"]
        bill = f"{_n(100 * bs, 0)} / {_n(100 * norm, 0)}" if np.isfinite(bs) and np.isfinite(norm) and net > 0 else "—"
        debt = r["marketable_bn"]
        rows.append(
            f"<tr><td class='l'><b>{NAME[cc]}</b></td><td>{_usd(usd(cc, debt))}</td><td>{_n(pg(cc, debt), 0)}</td>"
            f"<td>{_usd(usd(cc, net))}</td><td>{_n(pg(cc, net), 1)}</td><td><b>{_n(pg(cc, priv), 1)}</b></td>"
            f"<td>{_n(pg(cc, roll), 1)}</td><td>{_usd(usd(cc, gross))}</td><td>{_n(pg(cc, gross), 1)}</td><td>{bill}</td>"
            f"<td><b>{_n(gg_now.get(cc), 1)}</b></td><td style='color:#6b7385'>{_n(r['interest_pct_rev'], 1)}</td>"
            f"<td><b>{_n(r['debt_service_pct_rev'], 0)}</b></td>"
            f"<td>{_n(r['eff_rate'], 2)} / {_n(r['marginal_rate'], 2)}</td>"
            f"<td>{_n(r['debt_per_household_usd'] / 1000, 0)}k</td></tr>")
    t_snap = f"""<table><tr><th class='l'>Next 12m</th><th>Marketable<br>debt $</th><th>% GDP</th><th>Net<br>issuance $</th>
<th>% GDP</th><th>Private net<br>supply % GDP</th><th>Rollovers<br>% GDP</th><th>Gross<br>issuance $</th><th>% GDP</th>
<th>Bill share of net<br>/ norm %</th><th>Interest / revenue %<br>GG (US federal)</th><th>Central interest /<br>central revenue %</th><th>Central debt service /<br>central revenue %</th>
<th>Effective / marginal<br>rate %</th><th>Debt per<br>household $</th></tr>{''.join(rows)}</table>"""

    # Interest / GDP uses the published nominal GDP. Debt / GDP stays on the model GDP.
    pub = _published_gdp(ca)
    for cc in pub.columns:
        sub = ca["cc"].eq(cc)
        for y, g in pub[cc].items():
            if not np.isfinite(g) or g <= 0:
                continue
            m = sub & ca["year"].eq(int(y))
            if not m.any():
                continue
            ca.loc[m, "interest_pct_gdp"] = 100.0 * ca.loc[m, "interest_bn"] / g
            ca.loc[m, "gg_interest_pct_gdp"] = 100.0 * ca.loc[m, "gg_interest_bn"] / g

    # ---- per-country panels
    panels = []
    for cc in ORDER:
        if ca[(ca["cc"] == cc)]["net_issuance_bn"].notna().sum() < 3:
            continue
        fig, tab = _country_panel(cc, ca, s.at[cc, "ccy"], y_fc)
        charts[f"c_{cc}"] = fig
        panels.append(f"<h3>{NAME[cc]}</h3><div class='grid'><div id='c_{cc}' class='chart s'></div><div>{tab}</div></div>")

    # ---- stress: curve, +100bp, −100bp, holders rolling 90%
    rows = []
    for cc in ORDER:
        d = rp[rp["cc"] == cc]
        if d.empty:
            continue
        g = d.set_index(["scenario", "case", "year"])
        yl = sorted(d["year"].unique())[-1]
        rein = rv[(rv["cc"] == cc) & (rv["reinvest"] == 0.9)]
        rr = rein.iloc[0] if len(rein) else None
        curve = g.loc[("baseline", "current", yl)]
        up = g.loc[("baseline", "shock", yl)]
        down = g.loc[("recession", "ease", yl)]
        gpub = float(pub.at[yl, cc])
        scale = 1.02 ** (yl - (y_fc - 1))
        def ig(row, denom=1.0):
            return 100.0 * float(row["interest_bn"]) / (gpub * denom)
        rows.append(
            f"<tr><td class='l'><b>{NAME[cc]}</b></td>"
            f"<td>{_n(ig(curve), 2)}</td>"
            f"<td>{_n(ig(up), 2)}</td>"
            f"<td>{_n(ig(down), 2)}</td>"
            f"<td>{_n(ig(curve, scale), 2)}</td>"
            f"<td>{_n(ig(down, scale), 2)}</td>"
            f"<td>{_n(curve['debt_service_pct_rev'], 0)}</td>"
            f"<td>{_n(up['debt_service_pct_rev'], 0)}</td>"
            f"<td>{_n(down['debt_service_pct_rev'], 0)}</td>"
            f"<td>{_usd(rr['new_money_usd_bn']) if rr is not None else '—'}</td>"
            f"<td>{_n(rr['new_money_pct_gdp'], 1) if rr is not None else '—'}</td>"
            f"<td class='{'pos' if rr is not None and rr['vs_private_buying_last_year'] > 1 else ''}'>"
            f"{_n(rr['vs_private_buying_last_year'], 2) if rr is not None else '—'}×</td></tr>")
    t_stress = f"""<table><tr><th class='l'></th><th colspan=5>Central interest / GDP {yl}, %</th><th colspan=3>Debt service / revenue {yl}, %</th>
<th colspan=3>Holders roll 90%: new private money, 12m</th></tr>
<tr><th class='l'></th><th>curve</th><th>+100bp</th><th>−100bp</th><th>+2pp nominal GDP</th><th>−100bp and +2pp nominal GDP</th>
<th>curve</th><th>+100bp</th><th>−100bp</th>
<th>USD</th><th>% GDP</th><th>× last yr's<br>private buying</th></tr>{''.join(rows)}</table>"""

    # ---- unified forecast vs the authorities' own paths
    rows = []
    for cc in ORDER:
        x = ca[(ca["cc"] == cc) & (ca["year"] == yl)]
        if x.empty:
            continue
        x = x.iloc[0]
        ia = 100 * x["interest_authority_bn"] / x["revenue_authority_bn"]
        rows.append(f"<tr><td class='l'><b>{NAME[cc]}</b></td><td><b>{_n(x['net_issuance_pct_gdp'], 1)}</b></td>"
                    f"<td>{_n(x['deficit_weo_pct_gdp'], 1)}</td><td>{_n(x['deficit_authority_pct_gdp'], 1)}</td>"
                    f"<td>{_n(x['net_issuance_auth_pct_gdp'], 1)}</td><td>{_n(x['debt_pct_gdp'], 0)}</td>"
                    f"<td>{_n(x['debt_auth_pct_gdp'], 0)}</td>"
                    f"<td><b>{_n(x['interest_pct_rev'] if cc == 'us' else x['gg_interest_pct_rev'], 1)}</b></td><td>{_n(ia, 1)}</td>"
                    f"<td>{_n(s.at[cc, 'new_bond_yield'], 2)}</td></tr>")
    t_auth = f"""<table><tr><th class='l'></th><th>Net issuance<br>% GDP</th><th>WEO deficit<br>% GDP</th>
<th>Authority deficit<br>% GDP</th><th>Net issuance on<br>authority deficit</th><th>Debt<br>% GDP</th>
<th>Debt on<br>authority deficit</th><th>Interest / revenue<br>ours, GG (US federal) %</th><th>Interest / revenue<br>authority (EU: IMF) %</th>
<th>New-bond yield<br>today %</th></tr>{''.join(rows)}</table>"""

    # ---- holders
    hc = [("h_dcb", "hn_dcb", "Central bank"), ("h_db", "hn_db", "Dom. banks"), ("h_dnb", "hn_dnb", "Dom. non-banks"),
          ("h_fo", "hn_fo", "Foreign official"), ("h_fb", "hn_fb", "Foreign banks"), ("h_fnb", "hn_fnb", "Foreign non-banks")]
    rows = []
    for cc in ORDER:
        r = s.loc[cc]
        tot = r["hn_total"]
        rows.append(f"<tr><td class='l'><b>{NAME[cc]}</b></td>" + "".join(f"<td>{_n(r[a], 0)}</td>" for a, _, _ in hc)
                    + "".join(f"<td class='{'neg' if r[n] < 0 else ''}'>{_n(100 * r[n] / tot, 0) if tot > 0 else '—'}</td>"
                              for _, n, _ in hc) + "</tr>")
    t_hold = f"""<table><tr><th class='l' rowspan=2>{s.at['us', 'holders_q']}</th><th colspan=6>Share held, %</th>
<th colspan=6>Share of last-4q net buying, %</th></tr><tr>{''.join(f'<th>{lab}</th>' for *_, lab in hc) * 2}</tr>
{''.join(rows)}</table>"""
    charts["c_hold"] = _fig([{"type": "bar", "name": lab, "x": [NAME[c] for c in ORDER],
                              "y": [round(float(s.at[c, a]), 1) for c in ORDER]} for a, _, lab in hc],
                            "Who holds the debt (% of general-government debt); EA central bank is PSPP/PEPP", "%",
                            {"barmode": "stack", "hovermode": "closest"})

    # ---- supply history
    def line_hist(col, title):
        data = []
        for cc in ORDER:
            d = h[(h["cc"] == cc) & (h["month"] >= "2012-01-01")].dropna(subset=[col]).sort_values("month")
            act, fc = d[~d["is_forecast"]], d[d["is_forecast"]]
            if not fc.empty and not act.empty:
                fc = pd.concat([act.tail(1), fc])
            for part, dash, show in ((act, "solid", True), (fc, "dot", False)):
                if part.empty:
                    continue
                # real dates (not "%Y-%m" categories) so Plotly thins the axis
                data.append({"type": "scatter", "mode": "lines", "name": NAME[cc], "legendgroup": cc, "showlegend": show,
                             "x": [f"{m:%Y-%m-%d}" for m in part["month"]], "y": [round(float(v), 2) for v in part[col]],
                             "line": {"color": COL[cc], "dash": dash, "width": 1.6}})
        return _fig(data, title, "% GDP", {
            "xaxis": {"type": "date", "tickformat": "%b\n%Y", "dtick": "M12", "ticklabelmode": "period"},
        })
    charts["c_flow"] = _dropdown([
        ("Private net supply, 12m", line_hist("private12_pct_gdp", "Private net supply"), "% of GDP"),
        ("Central-bank net purchases, 12m", line_hist("cb12_pct_gdp", "Central-bank net purchases"), "% of GDP"),
    ])

    def line_year(col, title, visible=True, nd=1):
        data = []
        for cc in ORDER:
            d = ca[(ca["cc"] == cc) & (ca["year"] >= 2015)].dropna(subset=[col]).sort_values("year")
            act, fc = d[d["year"] < y_fc], d[d["year"] >= y_fc - 1]
            for part, dash, show in ((act, "solid", True), (fc, "dot", False)):
                if part.empty:
                    continue
                data.append({"type": "scatter", "mode": "lines+markers", "name": NAME[cc], "legendgroup": cc,
                             "showlegend": show, "x": part["year"].astype(int).tolist(),
                             "y": [round(float(v), nd) for v in part[col]],
                             "line": {"color": COL[cc], "dash": dash, "width": 1.8}, "marker": {"size": 4},
                             "visible": visible})
        return _fig(data, title, "% revenue", {"xaxis": {"dtick": 2}})
    charts["c_ratio"] = _dropdown([
        ("General government interest / revenue (peer basis)", line_year("gg_interest_pct_rev", "", nd=1), "% of revenue"),
        ("General government interest / GDP (peer basis)", line_year("gg_interest_pct_gdp", "", nd=2), "% of GDP"),
        ("Central interest / revenue (issuer, not comparable across countries)", line_year("interest_pct_rev", "", nd=1), "% of revenue"),
        ("Central interest / GDP (issuer)", line_year("interest_pct_gdp", "", nd=2), "% of GDP"),
        ("Central debt service / revenue (issuer)", line_year("debt_service_pct_rev", ""), "% of revenue"),
        ("Central debt service ex bills / revenue (issuer)", line_year("debt_service_ex_bills_pct_rev", ""), "% of revenue"),
        ("General government debt service / revenue", line_year("gg_debt_service_pct_rev", ""), "% of revenue"),
        ("General government debt service ex bills / revenue", line_year("gg_debt_service_ex_bills_pct_rev", ""), "% of revenue"),
    ])
    charts["c_ch"] = _channels(rp, pub)
    ctx = _context_charts(y_fc - 1)
    charts["c_macro"] = _dropdown([
        ("Debt / GDP", ctx["c_debt"], "% of GDP"),
        ("Primary balance", ctx["c_prim"], "% of GDP"),
        ("Nominal GDP growth", ctx["c_ngdp"], "%"),
    ])
    charts["c_auth"] = _auth_paths(ca, rp, pub)
    charts["c_gap"] = _gg_gaps(ca, rp)
    sens = _load_sens()
    if sens is not None:
        charts.update(_sens_charts(sens))
    sens_html = _sens_section(sens)
    t_peer = _peer_table(ca, y_fc)
    js = "\n".join(f"Plotly.newPlot('{k}', {json.dumps(v['data'])}, {json.dumps(v['layout'])}, "
                   "{responsive: true, displaylogo: false});" for k, v in charts.items())
    src = " · ".join(html.escape(v) for v in sorted(set(vint.values())))
    drivers = _drivers_section()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Debt &amp; supply</title>
<script src="{PLOTLY}"></script><style>{FONT_IMPORT}{CSS}{EXTRA_CSS}</style></head><body><div class="wrap">
<h1>Debt &amp; supply</h1>
<div class='k'>{kpi}</div>
{drivers}
<h2>Peer ranking: general government</h2>{t_peer}
<h2>Debt, the primary balance, nominal GDP</h2>
<div id='c_macro' class='chart' style='height:420px'></div>
<h2>Interest and debt service</h2>
<div id='c_ratio' class='chart' style='height:420px'></div>
<div id='c_ch' class='chart' style='height:380px'></div>
<h2>2029 GG interest / revenue</h2>
<div id='c_gap' class='chart' style='height:420px'></div>
<h2>Central interest</h2>
<div id='c_auth' class='chart' style='height:740px'></div>
<h2>Next twelve months</h2>{t_snap}
{sens_html}
<h2>By country, 2023 → {int(ca['year'].max())}</h2>{''.join(panels)}
<h2>Against the authorities, {yl}</h2>{t_auth}
<h2>Stress</h2>{t_stress}
<h2>Holders</h2><div class='grid'><div id='c_hold' class='chart'></div><div>{t_hold}</div></div>
<h2>Supply and the central bank</h2><div id='c_flow' class='chart' style='height:420px'></div>
{_signal_table(sig)}
</div><script>{js}</script></body></html>"""


def _signal_table(sig: dict | None) -> str:
    if not sig:
        return ""
    t = pd.DataFrame(sig["tests"])
    rows = "".join(
        f"<tr><td class='l'>{YLAB[r['y']]}</td><td class='l'>{XLAB[r['x']]}</td><td class='l'>{'country' if r['fe'] == 'country' else 'two-way'}</td>"
        f"<td>{_n(r['beta'], 2, sign=True)}</td><td>{r['p_placebo']:.2f}</td><td>{_n(r['beta_pre'], 2, sign=True)}</td>"
        f"<td>{_n(r['beta_post'], 2, sign=True)}</td><td>{_n(r['oos_r2'], 2, sign=True)}</td>"
        f"<td class='l {'neg' if r['supports_thesis'] else 'pos' if r['passes'] else ''}'>"
        f"{'supports' if r['supports_thesis'] else 'opposite' if r['passes'] else ''}</td></tr>" for _, r in t.iterrows())
    return f"""<details><summary>Panel test: 36 regressions (next-12m Δ, bp; six markets 2012→)</summary><table>
<tr><th class='l'>Target</th><th class='l'>Regressor</th><th class='l'>FE</th><th>β</th><th>Placebo p</th><th>β 2012–19</th>
<th>β 2020→</th><th>OOS R²</th><th class='l'></th></tr>{rows}</table></details>"""


def _load_sens() -> dict | None:
    """Live tables from ``sensitivities``; fall back to the JSON stub if long_end / investor miss."""
    from sovereign_cash.debt import sensitivities as Sens

    try:
        return Sens.tables()
    except Exception:
        path = DERIVED / "debt_supply_sensitivities.json"
        if not path.is_file():
            return None
        raw = json.loads(path.read_text())
        out = {
            "peers": raw["peers"],
            "curve_slope_bp_per_pp": raw.get("curve_slope_bp_per_pp"),
            "curve_slope_t": raw.get("curve_slope_t"),
        }
        for key in ("next12", "path2029", "curve", "plan"):
            df = pd.DataFrame(raw.get(key) or [])
            out[key] = df.set_index("cc") if not df.empty and "cc" in df.columns else df
        return out


def _sens_charts(t: dict) -> dict[str, dict]:
    """Grouped / horizontal bars parallel to the interest/revenue scenario charts."""
    n12, p29, curve = t["next12"], t["path2029"], t["curve"]
    peers = [cc for cc in ("us", "uk", "de", "fr", "it", "se", "no") if cc in n12.index]
    labels = [NAME[cc] for cc in peers]

    def col(series_key, scale=1.0):
        return [round(float(n12.at[cc, series_key]) * scale, 2)
                if cc in n12.index and np.isfinite(n12.at[cc, series_key]) else None for cc in peers]

    # Next-12m absorption: private net vs QT-style stop vs reinvest haircut
    c_priv = _fig([
        {"type": "bar", "name": "Private net", "x": labels, "y": col("private_net_pct_gdp"),
         "marker": {"color": SERIES[0]}},
        {"type": "bar", "name": "SOMA-stop / QT private", "x": labels, "y": col("soma_stop_priv_pct_gdp"),
         "marker": {"color": SERIES[4]}},
        {"type": "bar", "name": "Reinvest 90% new money", "x": labels, "y": col("reinvest_90_pct_gdp"),
         "marker": {"color": SERIES[5]}},
        {"type": "bar", "name": "CB net (runoff −)", "x": labels, "y": col("cb_net_pct_gdp"),
         "marker": {"color": SERIES[3]}},
    ], "Next 12m private absorption", "% of GDP", {
        "barmode": "group", "margin": {"l": 52, "r": 16, "t": 40, "b": 80},
        "legend": {"orientation": "h", "y": -0.28},
    })

    # Reinvestment ladder
    c_rein = _fig([
        {"type": "bar", "name": "100% roll", "x": labels, "y": col("reinvest_100_pct_gdp"),
         "marker": {"color": SERIES[7]}},
        {"type": "bar", "name": "95%", "x": labels, "y": col("reinvest_95_pct_gdp"),
         "marker": {"color": SERIES[0]}},
        {"type": "bar", "name": "90%", "x": labels, "y": col("reinvest_90_pct_gdp"),
         "marker": {"color": SERIES[4]}},
    ], "Reinvestment haircut: new private money", "% of GDP", {
        "barmode": "group", "margin": {"l": 52, "r": 16, "t": 40, "b": 72},
        "legend": {"orientation": "h", "y": -0.22},
    })

    # Path shocks at terminal year
    yl_peers = [cc for cc in peers if cc in p29.index]
    yl_lab = [NAME[cc] for cc in yl_peers]
    c_path = _fig([
        {"type": "bar", "name": "Δ net if +100bp interest borrowed",
         "x": yl_lab,
         "y": [round(float(p29.at[cc, "d_net_borrow100_pct_gdp"]), 2) for cc in yl_peers],
         "marker": {"color": C_UP}},
        {"type": "bar", "name": "Δ debt recession",
         "x": yl_lab,
         "y": [round(float(p29.at[cc, "d_debt_recession_pp"]), 2) for cc in yl_peers],
         "marker": {"color": SERIES[3]}},
        {"type": "bar", "name": "Δ rollovers recession",
         "x": yl_lab,
         "y": [round(float(p29.at[cc, "d_roll_recession_pp"]), 2) for cc in yl_peers],
         "marker": {"color": SERIES[5]}},
    ], f"Path shocks ({int(p29['year'].iloc[0]) if len(p29) else ''})", "pp of GDP", {
        "barmode": "group", "margin": {"l": 52, "r": 16, "t": 40, "b": 80},
        "legend": {"orientation": "h", "y": -0.28},
    })

    # Long-end face → 10s30s
    long_ccs = [cc for cc in ("us", "uk", "de", "fr", "it", "jp")
                if cc in curve.index and np.isfinite(curve.at[cc, "long_bp"])]
    long_lab = [NAME[cc] for cc in long_ccs]
    c_long = _fig([
        {"type": "bar", "orientation": "h", "name": "15y+ face",
         "y": long_lab,
         "x": [round(float(curve.at[cc, "long_bp"]), 1) for cc in long_ccs],
         "marker": {"color": SERIES[0]}},
        {"type": "bar", "orientation": "h", "name": "All-tenor",
         "y": long_lab,
         "x": [round(float(curve.at[cc, "all_bp"]), 1)
               if np.isfinite(curve.at[cc, "all_bp"]) else None for cc in long_ccs],
         "marker": {"color": SERIES[5]}},
        {"type": "bar", "orientation": "h", "name": "Joint",
         "y": long_lab,
         "x": [round(float(curve.at[cc, "joint_bp"]), 1)
               if np.isfinite(curve.at[cc, "joint_bp"]) else None for cc in long_ccs],
         "marker": {"color": SERIES[3]}},
    ], "Implied next-12m Δ10s30s from private face vs mean", "bp", {
        "barmode": "group", "yaxis": {"autorange": "reversed"},
        "margin": {"l": 70, "r": 20, "t": 40, "b": 72},
        "legend": {"orientation": "h", "y": -0.22},
        "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1,
                    "xref": "x", "yref": "paper",
                    "line": {"color": TOKENS["chart-zero"], "width": 1}}],
    })
    return {"c_sens_priv": c_priv, "c_sens_rein": c_rein, "c_sens_path": c_path, "c_sens_long": c_long}


def _sens_section(t: dict | None) -> str:
    """Supply-demand scenario panels (presentation only — no engine re-run)."""
    if t is None:
        return ("<h2>Supply-demand scenarios</h2>"
                "<p class='sub'>No sensitivity tables (run "
                "<code>python -m sovereign_cash.debt.sensitivities</code>).</p>")
    n12, p29, curve, plan = t["next12"], t["path2029"], t["curve"], t["plan"]
    peers = [cc for cc in ("us", "uk", "de", "fr", "it", "se", "no") if cc in n12.index]

    def name(cc):
        return NAME.get(cc, cc.upper())

    rows = []
    for cc in peers:
        r = n12.loc[cc]
        bill = (f"{_n(100 * r['bill_share_fwd'], 0)} / {_n(100 * r['bill_share_norm'], 0)}"
                if np.isfinite(r["bill_share_fwd"]) and np.isfinite(r["bill_share_norm"]) else "—")
        dv = r["private_dv01_usd_mn"]
        rows.append(
            f"<tr><td class='l'><b>{name(cc)}</b></td>"
            f"<td class='l'>{html.escape(str(r['window']))}</td>"
            f"<td>{_n(r['private_net_pct_gdp'], 1)}</td>"
            f"<td>{_usd(r['private_net_usd_bn'])}</td>"
            f"<td>{_n(dv, 0) if np.isfinite(dv) else '—'}</td>"
            f"<td>{_n(r['rollovers_pct_gdp'], 1)}</td><td>{bill}</td>"
            f"<td>{_n(r['cb_net_pct_gdp'], 1, sign=True)}</td>"
            f"<td>{_n(r['cb01_pct_gdp'], 1)}</td>"
            f"<td>{_n(r['soma_stop_priv_pct_gdp'], 1)}</td>"
            f"<td>{_n(r['reinvest_100_pct_gdp'], 1)}</td>"
            f"<td>{_n(r['reinvest_95_pct_gdp'], 1)}</td>"
            f"<td>{_n(r['reinvest_90_pct_gdp'], 1)}</td>"
            f"<td class='{'pos' if np.isfinite(r['reinvest_90_vs_buy']) and r['reinvest_90_vs_buy'] > 1 else ''}'>"
            f"{_n(r['reinvest_90_vs_buy'], 2)}×</td></tr>")
    t_n12 = f"""<table><tr><th class='l'>Next 12m</th><th class='l'>Window</th>
<th>Private net<br>% GDP</th><th>Private net<br>$</th><th>Private DV01<br>$mn/bp</th>
<th>Rollovers<br>% GDP</th><th>Bill share of net<br>/ norm %</th>
<th>CB net<br>% GDP</th><th>CB 0–1y<br>% GDP</th><th>SOMA-stop private<br>% GDP</th>
<th>Reinvest 100%<br>new money % GDP</th><th>95%</th><th>90%</th>
<th>× last yr<br>private buying</th></tr>{''.join(rows)}</table>
<p class='sub'>US private net = gross net (SOMA rolls at auction); elsewhere private = net − CB. SOMA-stop adds
US coupon holdings maturing in 12m. Reinvestment haircuts are new private money needed if holders roll
100/95/90% of redemptions.</p>"""

    rows = []
    for cc in peers:
        if cc not in p29.index:
            continue
        r = p29.loc[cc]
        rows.append(
            f"<tr><td class='l'><b>{name(cc)}</b></td>"
            f"<td>{_n(r['debt_pct_gdp'], 0)}</td><td>{_n(r['net_pct_gdp'], 1)}</td>"
            f"<td>{_n(r['roll_pct_gdp'], 1)}</td>"
            f"<td class='{_cls_delta(r['d_net_borrow100_pct_gdp'])}'>"
            f"{_n(r['d_net_borrow100_pct_gdp'], 2, sign=True)}</td>"
            f"<td class='{_cls_delta(r['d_debt_recession_pp'])}'>"
            f"{_n(r['d_debt_recession_pp'], 1, sign=True)}</td>"
            f"<td class='{_cls_delta(r['d_roll_recession_pp'])}'>"
            f"{_n(r['d_roll_recession_pp'], 1, sign=True)}</td>"
            f"<td class='{_cls_delta(r['d_net_recession_pct_gdp'])}'>"
            f"{_n(r['d_net_recession_pct_gdp'], 2, sign=True)}</td></tr>")
    yl = int(p29["year"].iloc[0]) if len(p29) else ""
    t_path = f"""<table><tr><th class='l'>{yl} path</th><th>Debt<br>% GDP</th><th>Net issuance<br>% GDP</th>
<th>Rollovers<br>% GDP</th><th>Δ net if +100bp<br>interest borrowed</th>
<th>Δ debt<br>recession</th><th>Δ rollovers<br>recession</th><th>Δ net<br>recession</th></tr>
{''.join(rows)}</table>
<p class='sub'>Baseline path vs borrowed +100bp (extra net from interest) and the recession case in
<code>debt_repricing</code>. Deltas in percentage points of GDP.</p>"""

    slope = t.get("curve_slope_bp_per_pp")
    st = t.get("curve_slope_t")
    rows = []
    for cc in ("us", "uk", "de", "fr", "it", "jp", "se", "no"):
        if cc not in curve.index:
            continue
        r = curve.loc[cc]
        if not np.isfinite(r.get("long_fwd", np.nan)) and not np.isfinite(r.get("all_fwd", np.nan)):
            continue
        rows.append(
            f"<tr><td class='l'><b>{name(cc)}</b></td>"
            f"<td>{_n(r['long_fwd'], 2)}</td><td>{_n(r['long_mean'], 2)}</td>"
            f"<td class='{_cls_delta(r['long_dev'])}'>{_n(r['long_dev'], 2, sign=True)}</td>"
            f"<td class='{_cls_delta(r['long_bp'])}'><b>{_n(r['long_bp'], 1, sign=True)}</b></td>"
            f"<td>{_n(r['all_fwd'], 2)}</td>"
            f"<td class='{_cls_delta(r['all_bp'])}'>{_n(r['all_bp'], 1, sign=True)}</td>"
            f"<td class='{_cls_delta(r['joint_bp'])}'>{_n(r['joint_bp'], 1, sign=True)}</td></tr>")
    t_curve = f"""<table><tr><th class='l'>15y+ face → 10s30s</th><th>Long fwd<br>% GDP</th><th>Long mean<br>% GDP</th>
<th>vs mean</th><th>Implied Δ10s30s<br>bp (long)</th><th>All-tenor fwd<br>% GDP</th>
<th>Implied bp<br>(all-tenor)</th><th>Joint bp</th></tr>{''.join(rows)}</table>
<p class='sub'>Pooled long-end slope {_n(slope, 1)} bp per pp GDP (t {_n(st, 1)}). Face = private 15y+ gross less
CB purchases in the same buckets; implied bp = slope × (fwd − sample mean). Joint = long + short wedge in one fit.
SE/NO have no long-end tape here.</p>"""

    rows = []
    for cc in peers:
        if cc not in plan.index:
            continue
        r = plan.loc[cc]
        gap = r["gap_bn"]
        rows.append(
            f"<tr><td class='l'><b>{name(cc)}</b></td>"
            f"<td>{_n(r['plan_bn'], 1)}</td><td>{_n(r['model_bn'], 1)}</td>"
            f"<td class='{_cls_delta(gap)}'>{_n(gap, 1, sign=True)}</td>"
            f"<td class='l'>{html.escape(str(r.get('plan_src') or ''))}</td>"
            f"<td class='l' style='max-width:280px'>{html.escape(str(r.get('absorbs') or '—'))}</td>"
            f"<td class='l' style='max-width:320px'>{html.escape(str(r.get('lands_on') or '—'))}</td>"
            f"<td class='l' style='max-width:360px'>{html.escape(str(r.get('reserves') or '—'))}</td></tr>")
    t_plan = f"""<table><tr><th class='l'>Plan gap</th><th>Plan<br>local bn</th><th>Model<br>local bn</th>
<th>Model − plan</th><th class='l'>Plan source</th><th class='l'>CHANNEL absorber</th>
<th class='l'>Where this gap lands</th><th class='l'>Cash ↔ reserves</th></tr>
{''.join(rows)}</table>
<p class='sub'>Gap from country cash frameworks. CHANNEL absorber / reserves are institutional facts from
<code>investor.CHANNEL</code> (same as the overview liquidity table). Positive gap = model borrowing above plan.</p>"""

    return f"""<h2>Supply-demand scenarios</h2>
<p class='sub'>Presentation layer on existing derived outputs (snapshot, reinvestment, repricing, long-end drifts,
investor tracking / CHANNEL). No cash-engine re-run. Charts parallel the interest/revenue scenario block above.</p>
<div class='grid'><div id='c_sens_priv' class='chart' style='height:360px'></div>
<div id='c_sens_rein' class='chart' style='height:360px'></div></div>
<div class='grid'><div id='c_sens_path' class='chart' style='height:360px'></div>
<div id='c_sens_long' class='chart' style='height:360px'></div></div>
<h3>Private net, DV01, rollovers, bill share, CB runoff</h3>{t_n12}
<h3>Path shocks ({yl})</h3>{t_path}
<h3>Long-end face → 10s30s</h3>{t_curve}
<h3>Plan-gap landing + CHANNEL</h3>{t_plan}"""


def _cls_delta(v) -> str:
    if v is None or (isinstance(v, (float, np.floating, int)) and not np.isfinite(v)):
        return ""
    if abs(float(v)) < 1e-12:
        return ""
    return "pos" if float(v) > 0 else "neg"


def _drivers_section() -> str:
    """Plain-English how interest and revenue are built on this tab + cash board."""
    rows = [
        ("Interest on this tab",
         "Every country with an auction tape: bond-by-bond coupons, linker uplift, "
         "floating-rate resets, the bill ladder rolled on the forward curve, plus the "
         "gap between that marketable book and the published interest outturn (loans, "
         "APF, definition). New bonds are couponed at the issuance-mix forward yield."),
        ("UK cash interest",
         "Same ISIN + curve annual figure, spread across months with the usual ONS "
         "JW2P seasonal pattern, scaled so the trailing twelve months still match the "
         "published total (covers non-marketable pieces the gilt book does not see)."),
        ("Germany cash interest",
         "ISIN + curve annual × the Finance Ministry interest month pattern outside "
         "the open Soll year (open year still blends the ministry plan with outturn)."),
        ("France cash interest",
         "AFT published debt-service charge when the budget has already printed it; "
         "otherwise the ISIN + curve annual. Months follow same-month-last-year."),
        ("Italy cash interest",
         "ISIN + curve annual × Bank of Italy interest month shares."),
        ("UK revenue / spend (cash)",
         "OBR year levels for each tax and spend block, ONS month shares, then the "
         "OBR ready reckoner on live earnings, retail, CPI, output, equities and rates."),
        ("France revenue / spend (cash)",
         "Budget (PLF) category lines when published; otherwise hold last complete "
         "year (local refunds may grow with inflation). Soft-pinned to the official deficit."),
        ("Germany / Italy revenue (cash)",
         "Germany: open-year Soll blend; later years same-month-last-year with GDP on "
         "receipts and wages on social spending. Italy: same-month-last-year with GDP "
         "on tax and inflation on primary spending."),
        ("Spain",
         "Cash rest-of-year: IGAE AC tax/social × NGDP and primary spend × HICP "
         "(es.bottom_up_macro). Interest on this page is Bonos / Letras ISIN + Tesoro curve "
         "(GOV_CURVE). Multi-year fiscal path from EC / Eurostat."),
        ("Japan",
         "Cash rest-of-FY: MoF GA same-month-last-year (category×macro NO-SHIP; sample from "
         "FY2024 only). Interest on this page is JGB / T-bill ISIN + MoF curve. Fiscal path "
         "Cabinet Office then WEO. Screened 2020–25 interest MAE ~¥177bn vs ~¥417bn last year."),
        ("United States",
         "Daily Treasury cash buckets with a calendar model; COLA on indexed benefits. "
         "Interest on this tab is the marketable ISIN + curve book."),
    ]
    body = "".join(
        f"<tr><td class='l'><b>{html.escape(k)}</b></td><td class='l'>{html.escape(v)}</td></tr>"
        for k, v in rows
    )
    return f"""<h2>How the forecasts are built</h2>
<p class='sub'>Cash-need recipes (UK / DE / FR / IT / ES / NO) feed the financing board; this
Debt &amp; supply tab keeps interest on the live bond book and the yield curve. Full
category tables live on the <a href="methodology.html">Methodology</a> page.</p>
<table><tr><th class='l'>Piece</th><th class='l'>What drives it</th></tr>{body}</table>"""


def build() -> Path:
    html_out = render()
    OUT.write_text(html_out, encoding="utf-8")
    print(f"Wrote {OUT}")
    try:
        from sovereign_cash.debt import sensitivities as Sens
        stub = Sens.write_stub()
        print(f"Wrote {stub}")
    except Exception as e:
        print(f"sensitivities stub skipped: {e}")
    return OUT


if __name__ == "__main__":
    build()
