"""Render the sovereign-cash overview for rates investors (``sovereign_cash_overview.html``).

Leads with how far to trust each live cash path (``investor.live_cash_path``: what drives each forecast
month, its score, and any bottom-up candidate that is not live), then forward supply to the private market (% of GDP, USD, private
DV01), the fiscal year vs the official plan, the calendar, and the reserves channel.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from sovereign_cash import investor as iv
from sovereign_cash.contract import MINUS, fmt_bn

PLOTLY = "https://cdn.plot.ly/plotly-3.7.0.min.js"
CSS = """
body { font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1d2433; background: #fff; }
.wrap { padding: 16px 22px 40px; }
table { width: 100%; }
details { margin: 4px 0 8px; color: #4a5263; } summary { cursor: pointer; font-size: 12px; color: #3b4356; }
h1 { font: 600 15px/1.3 ui-monospace, Menlo, monospace; letter-spacing: .06em; text-transform: uppercase; margin: 0 0 4px; }
h2 { font: 600 12px/1.3 ui-monospace, Menlo, monospace; letter-spacing: .06em; text-transform: uppercase; margin: 26px 0 6px;
     color: #3b4356; border-bottom: 1px solid #d8dde6; padding-bottom: 4px; }
p.sub { color: #4a5263; max-width: 1150px; margin: 4px 0 8px; }
table { border-collapse: collapse; margin: 6px 0 4px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 9px; border-bottom: 1px solid #e6e9ef; text-align: right; vertical-align: top; }
th { font-weight: 600; background: #f5f7fa; font-size: 12px; }
td.l, th.l { text-align: left; }
td.w { max-width: 460px; text-align: left; color: #3b4356; }
.pos { color: #a3261b; } .neg { color: #1d6b3a; }
.b { display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 11px; font-weight: 600; white-space: nowrap; }
.b.ok { background: #e2f3e8; color: #1d6b3a; } .b.tie { background: #fdf3d8; color: #8a6200; }
.b.bad { background: #fbe4e1; color: #a3261b; } .b.anc { background: #e3ecfb; color: #1f4c9a; }
.b.hi { background: #e2f3e8; color: #1d6b3a; } .b.med { background: #fdf3d8; color: #8a6200; }
.b.lo { background: #fbe4e1; color: #a3261b; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.chart { height: 340px; border: 1px solid #e6e9ef; }
.k { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 2px; }
.k div { border: 1px solid #d8dde6; border-top: 3px solid #3b4356; padding: 6px 10px; min-width: 150px; }
.k b { display: block; font: 600 17px/1.2 ui-monospace, Menlo, monospace; } .k span { font-size: 11px; color: #6b7385; }
"""
BADGE = (("beats", "ok"), ("ties", "tie"), ("does not beat", "bad"), ("official anchor", "anc"),
         ("PRE vs prior path", "ok"))


def _badge(verdict: str | None) -> str:
    if not verdict or verdict == "—":
        return "—"
    cls = next((c for p, c in BADGE if verdict.startswith(p)), "tie")
    return f"<span class='b {cls}'>{html.escape(verdict)}</span>"


def _n(v, d=1, pct=False, sign=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s = f"{v:+,.{d}f}" if sign else f"{v:,.{d}f}"
    return s.replace("-", MINUS) + ("%" if pct else "")


def _m(v, cc: str, d: int = 1, plus: bool = False) -> str:
    return fmt_bn(v, iv.SYM[iv.CCY[cc]].strip(), d, plus=plus)


def _cls(v) -> str:
    return "" if v is None or not np.isfinite(v) or abs(v) < 1e-9 else ("pos" if v > 0 else "neg")


def render(refresh: bool = False) -> str:
    sc = iv.scalers(refresh=refresh)
    sup, panel = iv.supply(sc)
    track = iv.tracking()
    live = {cc: p for cc in iv.ORDER if (p := iv.live_cash_path(cc))}
    cal = iv.calendar()
    fx = sc.get("fx_usd_per_unit", {})
    TIER_CLS = {"HIGH": "hi", "MED": "med", "LOW": "lo"}

    # ---- lead: supply
    rows = []
    for _, r in sup.iterrows():
        cc = r["cc"]
        lp = live.get(cc, {})
        badge = _badge(lp.get("verdict"))
        nb_only = cc == "no" and all(g["key"] == "nb" for g in lp.get("segments", []))
        dv = (f"{r['dv01_12m']:,.0f} <span style='color:#6b7385'>({r['dv01_float_pct']:.0f}%)</span>"
              if r["dv01_12m"] is not None and np.isfinite(r["dv01_12m"]) else "—")
        rows.append(
            f"<tr><td class='l'><b>{iv.NAME[cc]}</b></td><td class='l'>{r['window']}"
            f"{' <i>(NB horizon)</i>' if nb_only else ''}</td>"
            f"<td>{_n(r['cash_req_pct_gdp'])}</td>"
            f"<td>{_n(r['net_bonds_pct_gdp'])}</td><td>{_n(r['private_net_bonds_pct_gdp'])}</td>"
            f"<td>{_n(r['net_bills_pct_gdp'], sign=True)}</td>"
            f"<td>{_n(100 * r['bill_share'], 0, pct=True) if r['net_bills'] >= 0 and r['net_bonds'] > 0 else 'n/m'}</td>"
            f"<td>{_m(r['net_bonds'], cc, 0)} / {_m(r['net_bills'], cc, 0, plus=True)}</td>"
            f"<td>{fmt_bn(r['net_bonds_usd'] + r['net_bills_usd'], '$', 0)}</td>"
            f"<td>{dv}</td><td class='l'>{badge}</td></tr>")
    lead = f"""<table><tr><th class='l'>Country</th><th class='l'>Window</th><th>Cash req.<br>% GDP</th>
<th>Net bonds<br>% GDP</th><th>Private net<br>bonds % GDP</th><th>Net bills<br>% GDP</th><th>Bill share of<br>net supply</th><th>Net bonds / bills<br>local</th>
<th>Net marketable<br>USD</th><th>Private DV01 issued<br>$mn/bp (% of float)</th><th class='l'>Live cash-req. path<br>vs benchmark OOS</th></tr>
{''.join(rows)}</table>
<details><summary>Definitions</summary><p class='sub'>Next twelve forecast months per country (windows differ by
data lag). Net bonds = gross coupon issuance − redemptions (cash); net bills = change in bill stock. Private net bonds = net
bonds + redemptions held by the central bank. Private DV01 = DV01 of forecast gross issuance net of the central-bank take,
from the issuance forward; % = share of today's privately held DV01. The last column is the verdict of the
source driving most of the window (trust table above). Norway: Norges Bank's structural-liquidity forecast inside
its vintage, the Tab4 bottom-up beyond it; % GDP on mainland GDP (SSB); single months go negative when a bullet NGB
matures. Gross issuance here and in the issuance forward agree within ±3% over a common twelve months (DE +6%:
4pp is Finanzagentur retention, scaled out of the cash framework; US months differ by settlement date). US net bills exclude Fed purchases and rolls.</p></details>"""

    # ---- charts
    charts = {}
    sup_bar = {"data": [
        {"type": "bar", "name": "Net bonds", "x": [iv.NAME[c] for c in sup["cc"]], "y": list(sup["net_bonds_pct_gdp"].round(2)),
         "marker": {"color": "#1f4c9a"}},
        {"type": "bar", "name": "Net bills", "x": [iv.NAME[c] for c in sup["cc"]], "y": list(sup["net_bills_pct_gdp"].round(2)),
         "marker": {"color": "#d9822b"}}],
        "layout": {"barmode": "relative", "title": {"text": "Net marketable supply, next 12 forecast months (% GDP)",
                                                     "font": {"size": 13}},
                   "margin": {"l": 45, "r": 10, "t": 40, "b": 40}, "legend": {"orientation": "h", "y": -0.15}}}
    charts["c_sup"] = sup_bar
    if not panel.empty:
        months = sorted(panel["month"].unique())[:13]
        pv = panel.assign(net=panel["bills_pct_gdp"] + panel["bonds_pct_gdp"])
        pv = pv[pv["month"].isin(months)].pivot_table(index="cc", columns="month", values="net")
        pv = pv.reindex([c for c in iv.ORDER if c in pv.index])
        z = [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in pv.values]
        charts["c_heat"] = {"data": [{"type": "heatmap", "z": z, "x": [f"{pd.Timestamp(m):%b-%y}" for m in pv.columns],
                                      "y": [iv.NAME[c] for c in pv.index], "colorscale": "RdBu", "reversescale": True,
                                      "zmid": 0, "colorbar": {"title": {"text": "% GDP"}}, "hoverongaps": False,
                                      "texttemplate": "%{z:.2f}"}],
                            "layout": {"title": {"text": "Monthly net bonds + net bills (% GDP)", "font": {"size": 13}},
                                       "margin": {"l": 70, "r": 10, "t": 40, "b": 40},
                                       "yaxis": {"autorange": "reversed"}}}

    # ---- tracking
    trows = []
    for t in track:
        cc = t["cc"]
        gdp = sc.get("gdp_bn", {}).get(cc, np.nan)
        trows.append(
            f"<tr><td class='l'><b>{iv.NAME[cc]}</b></td><td class='l'>{t['year']}</td>"
            f"<td>{_m(t['ytd'], cc)} <span style='color:#6b7385'>to {t['ytd_through']}</span></td>"
            f"<td>{_m(t['plan'], cc)}</td><td class='w'>{html.escape(t['plan_src'])}</td>"
            f"<td>{_m(t['model'], cc)}</td>"
            f"<td class='{_cls(t['gap'])}'>{_m(t['gap'], cc, plus=True)} ({_n(100 * t['gap'] / gdp, 2, sign=True)}% GDP)</td>"
            f"<td class='w'>{html.escape(t['lands_on'])}</td></tr>")
    tracking = f"""<table><tr><th class='l'>Country</th><th class='l'>Year</th><th>Year to date</th><th>Official plan</th>
<th class='l'>Plan source</th><th>Model full year</th><th>Model − plan</th><th class='l'>Where a gap lands</th></tr>
{''.join(trows)}</table>
<p class='sub'>Cash requirement, deficit +. Positive gap = borrowing above plan. Read with the trust table above.</p>"""

    # ---- calendar
    crow = []
    for c in sorted(cal, key=lambda x: (x["month"], -x["bn"])):
        if c["bn"] <= 0:
            continue
        gdp = sc.get("gdp_bn", {}).get(c["cc"], np.nan)
        crow.append(f"<tr><td class='l'>{pd.Timestamp(c['month']):%b-%y}</td><td class='l'>{iv.NAME[c['cc']]}</td>"
                    f"<td class='l'>{c['what']}</td><td>{_m(c['bn'], c['cc'])}</td>"
                    f"<td>{100 * c['bn'] / gdp:.2f}%</td></tr>")
    calendar = f"""<table><tr><th class='l'>Month</th><th class='l'>Country</th><th class='l'>Flow</th><th>Amount</th>
<th>% GDP</th></tr>{''.join(crow)}</table>
<p class='sub'>Per country, the two largest bond-redemption and cash-requirement months from this month on, above 0.05% of GDP.</p>"""

    # ---- channel + buffer notes
    chrows = "".join(
        f"<tr><td class='l'><b>{iv.NAME[cc]}</b></td><td class='w'>{html.escape(iv.CHANNEL[cc][0])}</td>"
        f"<td class='w'>{html.escape(iv.CHANNEL[cc][1])}</td></tr>" for cc in iv.ORDER if cc in set(sup["cc"]))
    bufrows = "".join(
        f"<tr><td class='l'><b>{iv.NAME[cc]}</b></td><td class='w' style='max-width:900px'>"
        f"{html.escape(str(iv.summary(cc).get('buffer', '')))}</td></tr>" for cc in iv.ORDER if cc != "us" and iv.summary(cc))

    # ---- KPI strip
    tot = sup[sup["cc"] != "no"]
    eu = tot[tot["cc"].isin(["de", "fr", "it"])]
    k = (f"<div><b>${(tot['net_bills_usd']).sum():,.0f}bn</b><span>net bills, 12m, ex NO</span></div>"
         f"<div><b>${(tot['net_bonds_usd']).sum():,.0f}bn</b><span>net bonds, 12m, ex NO</span></div>"
         f"<div><b>{fmt_bn(eu['net_bills'].sum(), '€', 0, plus=True)}</b><span>DE+FR+IT net bills, 12m</span></div>"
         f"<div><b>${eu['dv01_12m'].sum():,.0f}mn</b><span>DE+FR+IT private DV01 issued, 12m</span></div>")

    asof = ", ".join(f"{iv.NAME[cc]} {iv.summary(cc).get('asof', '')[:7]}" for cc in iv.ORDER if cc != "us" and iv.summary(cc))
    scripts = "".join(f"Plotly.newPlot('{k_}', {json.dumps(v['data'])}, {json.dumps(v['layout'])}, "
                      f"{{displayModeBar: false, responsive: true}});" for k_, v in charts.items())
    fytd_block = ""
    try:
        from sovereign_cash.fytd_page import png_data_uri
        uri = png_data_uri()
        if uri:
            fytd_block = (
                "<h2>FYTD cash requirement (% GDP)</h2>"
                "<p class='sub'>Printed YTD; dashed = model; current / next = this- and "
                "next-year plans. Cash requirement only — financing identity charts stop "
                "at each country's coupon programme.</p>"
                f"<figure style='margin:8px 0 16px'><img src='{uri}' alt='FYTD cash' "
                "style='width:100%;max-width:1200px;border:1px solid #e6e9ef'/></figure>"
                "<p class='sub'><a href='fytd_cash_dashboard.html'>Open FYTD panel</a> · "
                "Cash → FYTD tab.</p>"
            )
    except Exception:  # noqa: BLE001
        fytd_block = ""

    trust_rows = []
    for cc, lp in live.items():
        if lp.get("shipped_bu"):
            bu_cell = "shipped: in the live path (left)" + (
                f"<br><span style='color:#6b7385'>{html.escape(lp['seasonal_tag'])} = "
                f"{html.escape(lp['seasonal_role'] or '')}</span>" if lp.get("seasonal_tag") and not lp.get("seasonal_live") else "")
        elif lp.get("bu_detail"):
            tier = lp.get("bu_tier")
            bu_cell = ((f"<span class='b {TIER_CLS.get(tier, 'tie')}'>{html.escape(tier)}</span> " if tier else "")
                       + f"<span style='color:#6b7385'>{html.escape(lp['bu_detail'])}</span>")
        else:
            bu_cell = "—"
        trust_rows.append(
            f"<tr><td class='l'><b>{iv.NAME[cc]}</b></td>"
            f"<td class='w'>{html.escape(lp['basis'])}</td><td class='w'>{html.escape(lp['oos'])}</td>"
            f"<td class='l'>{_badge(lp.get('verdict'))}</td><td class='w'>{bu_cell}</td></tr>")
    trust = f"""<table><tr><th class='l'>Country</th><th class='l'>What drives the live cash requirement (next 12 forecast months)</th>
<th class='l'>Out-of-sample score, each source vs its benchmark</th><th class='l'>Verdict</th>
<th class='l'>Bottom-up research (only where not live)</th></tr>
{''.join(trust_rows)}</table>
<p class='sub'>Live path = what the charts and the supply table use, read month by month from each
country's forward. Each source is scored against the benchmark it was gated on: naive (same month last
year, last-year flows), the seasonal fallback it replaced, or the official forecast. Windows: PRE ends by
2019, OOS from 2022; never re-chosen on OOS. "Ties" = within 3%. The verdict is that of the source covering
most of the window. Component-model scores are year-ahead calendar-year totals ex interest on realised macro,
so they flatter a live forecast. Where the live path does not beat naive, the bill path is the identity
consequence of the naive cash requirement and the bond calendar, not a forecast. Bottom-up research: edge on
PRE against the live path; "worse" means the candidate loses and is not shipped.</p>"""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Sovereign cash — investor view</title>
<script src="{PLOTLY}"></script><style>{CSS}</style></head><body><div class="wrap">
<h1>Sovereign cash — supply, plan tracking, liquidity</h1>
<p class="sub">Last actual: {asof}. FX {fx.get('date', '')} (Norges Bank); GDP = trailing four quarters (Eurostat, ONS, BEA).</p>
<div class="k">{k}</div>
{fytd_block}
<h2>How far to trust each forward path</h2>{trust}
<h2>Next 12 months: what the market has to absorb</h2>{lead}
<div class="grid"><div id="c_sup" class="chart"></div><div id="c_heat" class="chart"></div></div>
<h2>Fiscal year vs official plan</h2>{tracking}
<h2>Calendar: next six months</h2>{calendar}
<h2>Liquidity channel and shock absorber</h2>
<table><tr><th class='l'>Country</th><th class='l'>Cash buffer and central-bank reserves</th><th class='l'>What absorbs a fiscal surprise</th></tr>{chrows}</table>
<h2>What "buffer" means in each country</h2><table>{bufrows}</table>
</div><script>{scripts}</script></body></html>"""
