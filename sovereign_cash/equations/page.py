"""Shock-layer sections of the Methodology page, and the helpers every section of that page is built from.

``shock_sections(pack)`` returns the shock-response part of the Methodology (identity → limits) as a list of
``(section_id, title, html)``; ``sovereign_cash.methodology.page`` composes it after the production-chain sections
into one document. The CSS, ``eq_block``, ``fig``, ``fold``, ``prose_math``, ``resolve_refs``, the audit tables,
``WORKED_JS`` and ``FIGURES_JS`` live here so both halves of the page share one style.

``equations.html`` is no longer a page of its own: ``redirect_html`` writes a stub that forwards old links to
``methodology.html``.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from sovereign_cash.equations.registry import EQUATIONS, Equation

# Local vendor first (copied next to this HTML on build); CDN is the fallback if vendor is missing.
VENDOR = "vendor"
KATEX_CDN = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.7.0.min.js"
CSS = """
:root { --ink:#1d2433; --muted:#5d667a; --rule:#d8dde6; --panel:#f5f7fa; --link:#2f6f8f; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink);
  background:#fff; font-variant-numeric: tabular-nums; }
.wrap { max-width: 1100px; padding: 24px 20px 64px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -.01em; }
h2 { font-size: 19px; margin: 40px 0 6px; padding-top: 14px; border-top: 1px solid var(--ink); }
h3 { font-size: 15.5px; margin: 22px 0 4px; }
p { max-width: 72ch; margin: 6px 0 10px; }
.lead { font-size: 16.5px; }
.muted { color: var(--muted); font-size: 13.5px; }
details.src { margin: -4px 0 14px; max-width: 72ch; color: var(--muted); font-size: 11.5px; }
details.src summary { cursor: pointer; display: inline-block; list-style: none; color: var(--muted); opacity: .75; }
details.src summary::-webkit-details-marker { display: none; }
details.src summary::before { content: "▸ "; }
details.src[open] summary::before { content: "▾ "; }
details.src .code { font-family: ui-monospace, Menlo, monospace; margin: 2px 0 0; word-break: break-all; }
details.fold { margin: 10px 0 16px; border: 1px solid var(--rule); border-radius: 6px; padding: 0 12px 4px; background: var(--panel); }
details.fold > summary { cursor: pointer; list-style: none; font-size: 13.5px; color: var(--muted); padding: 10px 0; font-weight: 600; }
details.fold > summary::-webkit-details-marker { display: none; }
details.fold > summary::before { content: "▸ "; color: var(--ink); }
details.fold[open] > summary::before { content: "▾ "; }
details.fold[open] > summary { border-bottom: 1px solid var(--rule); margin-bottom: 8px; }
@media print {
  details.src > .code, details.fold { display: block; }
  details.fold > summary { display: none; }
}
.intro { max-width: 72ch; margin: 4px 0 14px; color: var(--ink); }
.readfirst { max-width: 72ch; margin: 12px 0 18px; padding: 12px 14px; background: var(--panel); border-radius: 6px; }
.readfirst p { margin: 0 0 6px; }
.readfirst p:last-child { margin: 0; }
.eq { overflow-x: auto; padding: 10px 0; }
a { color: var(--link); }
nav.toc { columns: 2; column-gap: 32px; font-size: 14px; margin: 14px 0 0; }
nav.toc a { display: block; padding: 2px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 8px 0 14px; }
th, td { padding: 5px 9px; text-align: right; border-bottom: 1px solid var(--rule); vertical-align: top; }
th:first-child, td:first-child, td.l, th.l { text-align: left; }
thead th { color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--ink); }
.tablewrap { overflow-x: auto; }
.flag { background: #fff7e0; border-left: 3px solid #c99a1e; padding: 8px 12px; margin: 10px 0; max-width: 72ch; }
.worked { background: var(--panel); padding: 10px 14px; border-radius: 6px; margin: 8px 0 16px; }
.worked select, .worked input { font: inherit; }
.worked .controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: end; margin: 8px 0 10px; }
.worked .controls label { display: flex; flex-direction: column; gap: 2px; font-size: 12.5px; color: var(--muted); }
.worked .controls input[type=number] { width: 5.5rem; padding: 3px 6px; }
.sym sub, .sym sup { font-size: 0.75em; line-height: 0; }
.fig { margin: 10px 0 18px; }
.fig .plot { width: 100%; height: 340px; }
.fig .plot.tall { height: 470px; }
.fig .row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.fig .row.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.fig .row .plot { height: 380px; }
.fig .row .plot.tall { height: 430px; }
.fig figcaption { color: var(--muted); font-size: 13px; max-width: 80ch; margin-top: 4px; }
.fig figcaption b { color: var(--ink); }
@media (max-width: 900px) { .fig .row, .fig .row.two { grid-template-columns: 1fr; } }
@media (max-width: 700px) { nav.toc { columns: 1; } }
"""


FIGURES_JS = (Path(__file__).resolve().parent / "figures.js").read_text(encoding="utf-8")
VENDOR_SRC = Path(__file__).resolve().parent / "vendor"


def prose_math(text: str) -> str:
    """Turn TeX-ish sub/superscripts in already-escaped prose into HTML <sub>/<sup>."""
    base = r"[A-Za-zΔΣσεωΠΒÂÊÎÔÛà-üα-ωĀ-ž̂̄̃]+"
    # C^{a,b}_{c,d} ; R^ℓ_p ; _{…} ; ^{…} ; X^y ; X_y
    text = re.sub(
        rf"({base})\^\{{([^}}]+)\}}_\{{([^}}]+)\}}",
        r'<span class="sym">\1<sup>\2</sup><sub>\3</sub></span>', text)
    text = re.sub(
        rf"({base})\^([A-Za-z0-9ℓ]+)_([A-Za-z0-9ℓ]+)",
        r'<span class="sym">\1<sup>\2</sup><sub>\3</sub></span>', text)
    text = re.sub(
        rf"({base})_\{{([^}}]+)\}}",
        r'<span class="sym">\1<sub>\2</sub></span>', text)
    text = re.sub(
        rf"({base})\^\{{([^}}]+)\}}",
        r'<span class="sym">\1<sup>\2</sup></span>', text)
    text = re.sub(
        rf"({base})\^([A-Za-z0-9ℓ]+)",
        r'<span class="sym">\1<sup>\2</sup></span>', text)
    text = re.sub(
        rf"({base})_([A-Za-z0-9ℓ]+)",
        r'<span class="sym">\1<sub>\2</sub></span>', text)
    return text


FIG_NUMBERS: dict[str, int] = {}


def reset_figures() -> None:
    """Start figure numbering at 1. Call once at the top of every page render."""
    FIG_NUMBERS.clear()


def fig(label: str, ids: list[str] | str, caption: str, *, tall: bool = False, cap_id: str | None = None) -> str:
    """A figure drawn in the browser by ``figures.js`` from the pack and the engine.

    Figures are numbered 1, 2, 3… in the order ``fig`` is called, i.e. page order, so a number can never be typed
    out of sequence. Refer to one from any prose as ``@fig:label@``; ``render`` resolves it to "Figure N".
    """
    if label in FIG_NUMBERS:
        raise ValueError(f"duplicate figure label {label!r}")
    n = FIG_NUMBERS[label] = len(FIG_NUMBERS) + 1
    ids = [ids] if isinstance(ids, str) else ids
    plots = "".join(f"<div id='{i}' class='plot{' tall' if tall else ''}' role='img' aria-label='{esc(caption)}'></div>"
                    for i in ids)
    if len(ids) > 1:
        plots = f"<div class='row{' two' if len(ids) == 2 else ''}'>{plots}</div>"
    cid = f" id='{cap_id}'" if cap_id else ""
    return f"<figure class='fig'>{plots}<figcaption><b>Figure {n}.</b> <span{cid}>{caption}</span></figcaption></figure>"


def esc(x) -> str:
    return html.escape(str(x))


def num(v, dp: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    if f != f:
        return "—"
    return f"{f:,.{dp}f}".replace("-", "−")


def eq_block(eid: str | Equation) -> str:
    """One equation: title, lead-in, KaTeX formula, every symbol defined, code path fold.

    Takes an id from the shock registry or an ``Equation`` object from any registry (the production chain's).
    """
    e = EQUATIONS[eid] if isinstance(eid, str) else eid
    lead = f'<p class="intro">{prose_math(esc(e.intro))}</p>' if e.intro else ""
    return (f'<h3 id="eq-{e.id}">{esc(e.title)}</h3>{lead}<div class="eq" data-tex="{esc(e.latex)}"></div>'
            f"<p>{prose_math(esc(e.says))}</p>"
            f"<details class=\"src\"><summary>code</summary><p class=\"code\">{esc(e.code)}</p></details>")


def intro(text: str) -> str:
    return f'<p class="intro">{prose_math(esc(text))}</p>'


def fold(summary: str, body: str, *, open_: bool = False) -> str:
    """Collapse audit / parameter tables so the main path stays figure-led."""
    if not body:
        return ""
    return (f"<details class='fold'{' open' if open_ else ''}><summary>{esc(summary)}</summary>"
            f"{body}</details>")


def resolve_refs(html_text: str, sec_numbers: dict[str, int]) -> str:
    """Replace ``@fig:label@`` / ``@sec:id@`` with "Figure N" / "Section N"; an unknown label is a build error."""
    def sub(m: re.Match) -> str:
        kind, key = m.group(1), m.group(2)
        table = FIG_NUMBERS if kind == "fig" else sec_numbers
        if key not in table:
            raise KeyError(f"unresolved cross-reference @{kind}:{key}@")
        return f"{'Figure' if kind == 'fig' else 'Section'}&nbsp;{table[key]}"
    return re.sub(r"@(fig|sec):([a-z0-9_]+)@", sub, html_text)


def identity_table(pack: dict) -> str:
    rows = []
    for c in pack["countries"].values():
        terms = " ".join(("+" if ln["sign"] > 0 else "−") + " " + esc(ln["label"].lower()) for ln in c["lines"])
        rows.append(f"<tr><td>{esc(c['name'])}</td><td class='l'>{esc(c['measure'])}</td>"
                    f"<td class='l'>{terms.lstrip('+ ')}</td><td class='l'>{esc(c['period_kind'])}</td>"
                    f"<td class='l'>{esc(c['baseline']['source'])}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Measure</th>"
            "<th class='l'>Identity</th><th class='l'>Periods</th><th class='l'>Baseline</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def forecast_table(pack: dict) -> str:
    """Our baseline by year, line by line, beside our cash board and the authority's own plan (comparison only)."""
    rows = []
    for cc in pack["countries"]:
        c = pack["countries"].get(cc)
        if not c or "model_total" not in c["baseline"]:
            continue
        b = c["baseline"]
        stub = (c.get("history") or {}).get("stub") or {}
        auth = (b.get("authority") or {}).get("years", {})
        per = [ln for ln in c["lines"] if ln.get("perimeter")]
        for i, p in enumerate(c["periods"]):
            sl = stub.get("lines", {}) if i == 0 else {}
            full = b["total"][i] + (float(stub.get("total", 0.0)) if i == 0 else 0.0)
            pv = " ".join(f"{num(b['lines'][ln['id']][i] + sl.get(ln['id'], 0.0), 1)}" for ln in per) or "—"
            model = full - sum(b["lines"][ln["id"]][i] + sl.get(ln["id"], 0.0) for ln in per)
            bf = (b.get("board_full_year") or [None] * len(c["periods"]))[i]
            a = auth.get(p)
            rows.append(f"<tr><td>{esc(c['name'])}</td><td>{esc(p)}</td><td>{num(model, 1)}</td><td>{pv}</td>"
                        f"<td><b>{num(full, 1)}</b></td><td>{num(bf, 1) if bf is not None else '—'}</td>"
                        f"<td>{num(a, 1) if a is not None else '—'}</td>"
                        f"<td>{num(full - a, 1) if a is not None else '—'}</td></tr>")
    if not rows:
        return ""
    labels = "; ".join(f"{pack['countries'][cc]['name']}: {esc(pack['countries'][cc]['baseline']['authority']['label'])}"
                       for cc in pack["countries"]
                       if (pack["countries"][cc]["baseline"].get("authority") or {}).get("label"))
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th>Period</th><th>Modelled lines</th>"
            "<th>Perimeter P</th><th>Our baseline</th><th>Our cash board</th><th>Official forecast</th>"
            "<th>Ours − official</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='note'>Local currency bn, full periods (the first adds the months already out-turned). The official "
            "forecasts are not used anywhere in the baseline; they are here so the gap is visible. Official: "
            f"{labels}.</p>")


def limitations(pack: dict) -> str:
    """What a reviewer should hold against these numbers, and what is done about each."""
    my = (pack.get("model_selection") or {}).get("multi_year") or {}
    n2 = {cc: (my.get(cc, {}).get("final", {}).get("h2", {}) or {}).get("n") for cc in ("fr", "de", "it")}
    rec = pack.get("forecast_record") or {}
    rows = [
        ("Backtests are conditional on the macro that happened",
         "The test feeds each model the GDP and prices that actually came, so its errors leave out macro forecast error. "
         "Real-time errors will be larger.",
         "This is the right test for a scenario tool (the user sets the macro), and it is labelled as such.",
         "Score the archived forecasts below against outturn as years close: that is the real-time record."),
        ("Few origins, overlapping horizons",
         f"Two years ahead there are {n2.get('fr')} (France), {n2.get('de')} (Germany) and {n2.get('it')} (Italy) "
         "December origins, and neighbouring forecasts share years, so errors are correlated. Differences of a few bn "
         "are not significant.",
         "Choices are made on large differences only (Germany halved its error; Italy cut it by a quarter) and on "
         "theory; every candidate's score is published, including the ones not chosen.",
         "Add Diebold-Mariano tests with a HAC variance, and monthly origins."),
        ("Germany's rule rests on one post-reform year",
         "The share-persists rule is chosen because the 2025 reform of the debt brake removed the pull back; the "
         "evidence is 2025 alone (error −5.9bn against −16.4bn under the old rule).",
         "Stated on this page; the debt-brake-era rule stays in the code and in the test for comparison.",
         "Re-test each year as post-reform outturns arrive."),
        ("Legislated policy: static costings on a different baseline",
         "Every country carries a legislated-policy line on top of the model. The costings are the authorities' own and "
         "static, measured against the authority's baseline, not ours: a base-year measure is netted off by its share, "
         "which is approximate where the model reverts a share to its mean (US receipts), and where an authority budgets "
         "in nominal frames (Sweden) only ring-fenced spending is added. Budgets still before parliament are left out; "
         "Germany's line is the Finanzplan under laws already passed.",
         "Each measure, its source and the netting rule are on this page; the dashboard can switch the line off.",
         "Add each 2027 budget when it becomes law; earlier events' scheduled expiries where the authority publishes "
         "them."),
        ("Unit elasticity",
         "Receipts and spending move one for one with their driver level. Estimated free elasticities (France 0.68, "
         "Germany 1.01, Italy 1.11) are not distinguishable from 1, but short-run tax elasticities are often above 1.",
         "Unit elasticity is the textbook long-run value and the estimates do not reject it.",
         "Tax-by-tax elasticities with lags, as the authorities use, once the category data are long enough."),
        ("Interest depends on today's curve",
         "Interest comes from the ISIN book at today's forward rates; the curve shapes come from history since 2000.",
         "Yield shocks between the book's 100bp runs are within 0.011% of the interest bill; the curve shape's "
         "linearity is checked each build.",
         "Show the book's interest against the authorities' debt-service forecasts each vintage."),
        ("Perimeter lines are simple",
         "Germany's special funds run at their year-to-date rate; Italy's flows outside the cash-account lines are a "
         "three-year same-month average.",
         "They are separate lines, labelled, and no shock acts on them.",
         "Model the special funds from their legislated envelopes."),
        ("Annual models for six countries",
         "The UK, US, Spain, Sweden, Norway and Japan are forecast by annual share models (the long histories are "
         "annual), so their within-year path is spread evenly; only the UK uses its months of the year in progress.",
         "Same economics and the same exact shock channels as the monthly models; tested one to three years ahead "
         "from every year since 2005.",
         "Monthly component models where monthly cash data exist (UK CGNCR, US Monthly Treasury Statement)."),
        ("OBR's reckoner is from March 2025",
         "OBR's latest published ready reckoner is consistent with its March 2025 forecast; it has not published one "
         "since.",
         "It is a comparison only: the UK's default responses come from our model.",
         "Load OBR's next reckoner when it is published."),
        ("Point forecasts only",
         "The dashboard shows no uncertainty band around the baseline.",
         "The backtest above gives its size: about 20 to 30bn at one to two years.",
         "Fan charts from the backtest errors by horizon."),
        ("Drivers move independently",
         "A GDP shock does not move yields, and a yield shock does not move GDP; the presets set combinations by hand.",
         "Stated with each preset.",
         "A small macro block linking growth, inflation and yields."),
    ]
    body = "".join(f"<tr><td class='l'><b>{esc(a)}</b></td><td class='l'>{esc(b)}</td><td class='l'>{esc(c)}</td>"
                   f"<td class='l'>{esc(d)}</td></tr>" for a, b, c, d in rows)
    record = ""
    if rec:
        record = (f"<p>Forecast record: {rec.get('n_builds', 0)} builds archived since {esc(rec.get('since', '—'))} "
                  f"(sovereign_cash/derived/forecast_vintages.jsonl). Each year's forecasts are scored against outturn "
                  f"when the year closes; the first full year to score is {esc(rec.get('first_scorable', '—'))}.</p>")
    return ("<p>What a reviewer should hold against these numbers, and what is done about each.</p>"
            "<div class='tablewrap'><table><thead><tr><th class='l'>Limitation</th><th class='l'>What it means</th>"
            "<th class='l'>How it is handled now</th><th class='l'>Next</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>{record}")


def policy_table(pack: dict) -> str:
    """Each country's legislated-policy line: the named measures and the net, by year, with the sources."""
    out = []
    for c in pack["countries"].values():
        p = c.get("policy")
        if not p:
            continue
        head = "".join(f"<th>{esc(x)}</th>" for x in c["periods"])
        rows = [f"<tr><td class='l'>{esc(g['label'])}{'' if g.get('other') else ''}</td><td class='l'>{esc(g['head'])}</td>"
                + "".join(f"<td>{num(v, 1)}</td>" for v in g["values"]) + "</tr>" for g in p["groups"]]
        rows.append("<tr class='tot'><td class='l'><b>Net, all measures</b></td><td></td>"
                    + "".join(f"<td><b>{num(v, 1)}</b></td>" for v in p["total"]) + "</tr>")
        wp = c["baseline"].get("with_policy") or []
        stub = ((c.get("history") or {}).get("stub") or {}).get("total", 0.0) or 0.0
        rows.append("<tr><td class='l'>Our model, no policy change</td><td></td>" + "".join(
            f"<td>{num(v + (stub if i == 0 else 0.0), 1)}</td>" for i, v in enumerate(c["baseline"]["total"])) + "</tr>")
        rows.append("<tr><td class='l'><b>Our model + legislated policy</b></td><td></td>" + "".join(
            f"<td><b>{num(v + (stub if i == 0 else 0.0), 1)}</b></td>" for i, v in enumerate(wp)) + "</tr>")
        src = "; ".join(f"<a href='{esc(s['url'])}'>{esc(s['title'])}</a>" for s in p["sources"])
        out.append(f"<h3>{esc(c['name'])}</h3><div class='tablewrap'><table><thead><tr><th class='l'>{esc(c['unit'])}, "
                   f"effect on {esc(c['measure'].lower())}</th><th class='l'>Head</th>{head}</tr></thead>"
                   f"<tbody>{''.join(rows)}</tbody></table></div>"
                   f"<p class='note'>{p['n_measures']} measures; base year {esc(p.get('base_period_label', ''))}. "
                   f"Sources: {src}. Full periods (the UK's first year includes its months of outturn).</p>")
    if not out:
        return ""
    return ("".join(out) + "<p class='note'>Included: measures in law, costed by the authority that costed them. Not "
            "included: budgets still before parliament (the 2027 budgets in France, Italy, Sweden and Norway), tariffs "
            "and other executive actions (US), and announced spending beyond the plans listed. Germany's defence and "
            "infrastructure lines are the cabinet's Finanzplan under laws already passed; its 2027 budget law is before "
            "the Bundestag. Spain has had no budget law since 2023, so only the pension reform's scheduled contribution "
            "rises enter. Norway budgets one year at a time, so its 2026 loosening is held. Sweden: general spending "
            "reforms are not added, because Sweden sets spending in nominal frames while our model already grows "
            "spending with nominal GDP; tax measures, defence and Ukraine support are.</p>")


def bridge_table(pack: dict) -> str:
    """Our forecast against each authority's, line by line (scenarios/bridge.py)."""
    br = pack.get("authority_bridge") or {}
    out = []
    lab = {"receipts": "Receipts", "primary": "Spending excluding interest", "interest": "Interest"}
    for cc, b in br.items():
        c = pack["countries"].get(cc, {})
        if b.get("total_only"):
            cells = "".join(f"<td>{num(a, 1) if a is not None else '—'}</td><td>{num(o, 1)}</td>"
                            for a, o in zip(b["authority"], b["ours"]))
            out.append(f"<tr><td class='l'>{esc(c.get('name', cc))}</td><td class='l' colspan='2'>{esc(b['label'])} "
                       f"vs ours (total only; the plan has no line split we can match)</td>{cells}</tr>")
            continue
        head = "".join(f"<th colspan='2'>{esc(r['period'])}</th>" for r in b["rows"])
        sub = "".join("<th>bn</th><th>% GDP</th>" for _ in b["rows"])
        body = [f"<tr><td class='l'><b>{esc(b['label'])}</b></td>"
                + "".join(f"<td><b>{num(r['authority'], 1)}</b></td><td></td>" for r in b["rows"]) + "</tr>"]
        for k in ("receipts", "primary", "interest"):
            body.append(f"<tr><td class='l'>{lab[k]}: forecast</td>" + "".join(
                f"<td>{num(r['lines'][k]['forecast'], 1)}</td><td>{num(r['lines'][k]['forecast_pct_gdp'], 2)}</td>"
                for r in b["rows"]) + "</tr>")
        body.append("<tr><td class='l'>Definitions (difference already in the last outturn year, grown with GDP)</td>"
                    + "".join(f"<td>{num(sum(r['lines'][k]['definitions'] for k in r['lines']), 1)}</td><td></td>"
                              for r in b["rows"]) + "</tr>")
        body.append("<tr><td class='l'><b>Ours (with legislated policy)</b></td>"
                    + "".join(f"<td><b>{num(r['ours'], 1)}</b></td><td></td>" for r in b["rows"]) + "</tr>")
        reasons = "".join(f"<li><b>{lab[k]}.</b> {esc(t)}</li>" for k, t in b["reasons"].items())
        out.append(f"<h3>{esc(c.get('name', cc))}: against {esc(b['label'])}</h3><div class='tablewrap'><table>"
                   f"<thead><tr><th class='l'>{esc(b['unit'])}, effect on the deficit</th>{head}</tr>"
                   f"<tr><th></th>{sub}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
                   f"<ul class='note'>{reasons}</ul><p class='note'>Source: <a href='{esc(b['url'])}'>{esc(b['source'])}</a>. "
                   f"Rows add up exactly: authority + receipts + spending + interest + definitions = ours. Base year "
                   f"{esc(b.get('base_year'))}.</p>")
    totals = [x for x in out if x.startswith("<tr>")]
    blocks = [x for x in out if not x.startswith("<tr>")]
    tot = ""
    if totals:
        tot = ("<h3>France, Germany, Italy: against the government's plan</h3><div class='tablewrap'><table><tbody>"
               + "".join(totals) + "</tbody></table></div><p class='note'>Pairs per year: plan, ours. The plans are "
               "the governments' own financing figures for the years they cover.</p>")
    return "".join(blocks) + tot


def crosscheck_table(pack: dict) -> str:
    """Each country's policy numbers against an independent source (data/policy/checks.json)."""
    path = Path(__file__).resolve().parents[1] / "data" / "policy" / "checks.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    names = {cc: c["name"] for cc, c in pack["countries"].items()}
    rows = "".join(
        f"<tr><td class='l'>{esc(names.get(r['cc'], r['cc']))}</td><td class='l'>{esc(r['item'])}</td>"
        f"<td class='l'>{esc(r['ours'])}</td><td class='l'>{esc(r['other'])}</td>"
        f"<td class='l'><a href='{esc(r['url'])}'>{esc(r['source'])}</a></td><td class='l'>{esc(r['verdict'])}</td></tr>"
        for r in d["rows"])
    return (f"<h3>Cross-checks</h3><p>{esc(d['note'])} Checked {esc(d['checked'])}.</p><div class='tablewrap'><table>"
            "<thead><tr><th class='l'>Country</th><th class='l'>What</th><th class='l'>Ours</th><th class='l'>Other source</th>"
            "<th class='l'>Source</th><th class='l'>Verdict</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def own_table(pack: dict) -> str:
    """The annual models' walk-forward test: each line's share, persist against revert, and the measure ex interest."""
    oa = (pack.get("model_selection") or {}).get("own_annual") or {}
    rows = []
    for cc, r in oa.items():
        if "lines" not in r:
            continue
        nm = pack["countries"].get(cc, {}).get("name", cc)
        for line, v in r["lines"].items():
            sc = v["scores"]
            rows.append(f"<tr><td>{esc(nm)}</td><td class='l'>{esc(line)}</td>"
                        + "".join(f"<td>{num((sc.get(m) or {}).get('mean_pct_gdp'), 2)}</td>"
                                  for m in ("persist", "revert", "persist_gdp"))
                        + f"<td class='l'><b>{esc({'persist': 'persist', 'revert': 'revert', 'persist_gdp': 'persist, share of GDP', 'fiscal_drag': 'IT+NIC fiscal drag (ε=1.3 frozen)'}[v['chosen']])}</b></td></tr>")
        t = r.get("total_ex_interest") or {}
        if t:
            rows.append(f"<tr><td>{esc(nm)}</td><td class='l'><i>measure ex interest, as used</i></td>"
                        f"<td colspan='3' class='l'>MAE {' / '.join(num(t[f'h{k}']['mae'], 1) for k in (1, 2, 3))} bn at 1/2/3 years; "
                        f"bias {' / '.join(num(t[f'h{k}']['bias'], 1) for k in (1, 2, 3))}; {r.get('n_origins')} origins from "
                        f"{r.get('first_origin')}</td><td></td></tr>")
    if not rows:
        return ""
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Share</th>"
            "<th>Persist</th><th>Revert to mean</th><th>Spending: share of GDP</th><th class='l'>Chosen</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='note'>Mean absolute error one to three years ahead (averaged), % of GDP, every origin since 2005, "
            "given the GDP and prices that happened, Covid years excluded. Receipts are a share of GDP; spending a share "
            "of the price level unless keeping its share of GDP is better. The default (persist) is replaced only where "
            "another rule is at least 10% better. UK receipts also score a frozen-threshold fiscal-drag rule (IT+NIC at "
            "ε=1.3 to earnings while thresholds are frozen under prior law; OBR EFO Oct 2026 para 4.14), shipped when it "
            "does not lose to persist. sovereign_cash/scenarios/channels/own_annual.py.</p>")


def multi_year_table(pack: dict) -> str:
    """The walk-forward test behind METHOD: one to three calendar years ahead."""
    my = (pack.get("model_selection") or {}).get("multi_year")
    if not my:
        return ""
    names = {"final": "In use", "current": "Share persists everywhere (before Nov 2026)",
             "combo": "Debt-brake era: German spending reverts", "mr": "Every share reverts",
             "naive": "Last year's share of GDP, every line", "ngdp": "Spending on nominal GDP"}
    rows = []
    for cc in ("fr", "de", "it"):
        r = my.get(cc)
        if not r:
            continue
        nm = pack["countries"].get(cc, {}).get("name", cc)
        for spec, label in names.items():
            sc = r.get(spec)
            if not sc or (spec == "combo" and cc != "de"):
                continue
            rows.append(f"<tr><td>{esc(nm)}</td><td class='l'>{'<b>' + label + '</b>' if spec == 'final' else label}</td>"
                        + "".join(f"<td>{num(sc[f'h{k}']['mae_total_ex_covid'], 1)}</td>" for k in (1, 2, 3))
                        + "".join(f"<td>{num(sc[f'h{k}']['bias_total_ex_covid'], 1)}</td>" for k in (1, 2, 3))
                        + f"<td>{sc['h2']['n']}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>How shares are carried forward</th>"
            "<th>MAE 1y</th><th>2y</th><th>3y</th><th>Bias 1y</th><th>2y</th><th>3y</th><th>Origins</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='note'>Cash requirement excluding interest, EUR bn, calendar years one to three ahead of each December "
            "origin from 2013, given the macro that happened; 2020 and 2021 excluded as target years. Positive bias = "
            "forecast deficit too high. 'In use' is the production model itself. Germany's in-use rule scores worse over "
            "2013-24, when the debt brake pulled spending back after each crisis, and better on 2025, the first year "
            "after the reform (error −12.9bn against −23.4bn). sovereign_cash/experiments/multi_year_backtest.py.</p>")


def sarimax_table(pack: dict) -> str:
    rows = []
    for cc in ("fr", "de", "it"):
        c = pack["countries"].get(cc)
        if not c:
            continue
        for comp, f in c["model"]["fits"].items():
            if f is None:
                rows.append(f"<tr><td>{esc(c['name'])}</td><td class='l'>{esc(comp)}</td>"
                            "<td colspan='6' class='l'>did not fit; same month last year used</td></tr>")
                continue
            p = f["params"]
            ar = next((v for k, v in p.items() if k.startswith("ar.L1")), None)
            ma = next((v for k, v in p.items() if k.startswith("ma.S.L12")), None)
            t = (f["beta"] / f["beta_se"]) if f.get("beta") is not None and f.get("beta_se") else None
            spec = {"ratio": "share of level", "loglevel": "log level", "yoy": "YoY rate"}.get(f.get("spec", "yoy"), "YoY rate")
            rows.append(f"<tr><td>{esc(c['name'])}</td><td class='l'>{esc(comp)}</td>"
                        f"<td class='l'>{esc(f['exog'] or 'none')}{' (' + spec + ')' if f['exog'] and f['exog'] != 'rates' else ''}</td><td>{num(f['beta'], 4)}</td>"
                        f"<td>{num(f['beta_se'], 4)}</td><td>{num(t, 1)}</td><td>{num(ar)}</td><td>{num(ma)}</td>"
                        f"<td>{f['nobs']}</td><td class='l'>{esc(f['sample'][0][:7])} to {esc(f['sample'][1][:7])}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Component</th>"
            "<th class='l'>Driver x</th><th>β</th><th>s.e.</th><th>t</th><th>φ (AR1)</th>"
            "<th>Θ (seasonal MA)</th><th>Months</th><th class='l'>Sample</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='muted'>Driven lines are shares of a level, so they have no β: the response is proportional by "
            "construction. Interest has β, bn per pp of the effective rate.</p>")


def selection_tables(pack: dict) -> str:
    ms = pack.get("model_selection")
    if not ms:
        return "<p class='muted'>Backtest results not in this build.</p>"
    res, used = ms["results"], ms["in_use"]
    names = {"yoy": "Growth rate (before)", "loglevel": "Log level", "ratio": "Share of level"}
    rows = []
    for cc in ("fr", "de", "it"):
        r = res.get(cc)
        if not r or "error" in r:
            continue
        nm = pack["countries"].get(cc, {}).get("name", cc)
        for spec in ("yoy", "loglevel", "ratio"):
            sc = r["scores"].get(spec) or {}
            pre, oos = sc.get("PRE", {}), sc.get("OOS", {})
            v = r["verdict"].get(spec, {})
            rel = v.get("pre_mean_vs_yoy")
            mark = "in use" if used.get(cc) == spec else ""
            if spec == "yoy":
                mark = "replaced"
            rows.append(f"<tr><td>{esc(nm)}</td><td class='l'>{names[spec]}</td>"
                        + "".join(f"<td>{num(pre.get(h), 1)}</td>" for h in ("1", "3", "6", "12"))
                        + "".join(f"<td>{num(oos.get(h), 1)}</td>" for h in ("1", "3", "6", "12"))
                        + f"<td>{'' if rel is None else f'{100 * rel:+.0f}%'}</td><td class='l'><b>{mark}</b></td></tr>")
    t1 = ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Specification</th>"
          "<th>PRE h1</th><th>h3</th><th>h6</th><th>h12</th><th>OOS h1</th><th>h3</th><th>h6</th><th>h12</th>"
          "<th>PRE mean vs before</th><th class='l'></th></tr></thead>"
          f"<tbody>{''.join(rows)}</tbody></table></div>")
    erows = []
    for cc in ("fr", "de", "it"):
        r = res.get(cc)
        if not r or "error" in r:
            continue
        for comp, e in (r.get("elasticities") or {}).items():
            if not e:
                continue
            erows.append(f"<tr><td>{esc(pack['countries'].get(cc, {}).get('name', cc))}</td><td class='l'>{esc(comp)}</td>"
                         f"<td class='l'>{'nominal GDP' if e['driver'] == 'ngdp' else 'HICP'}</td>"
                         f"<td>{num(e['beta'], 2)}</td><td>{num(e['se'], 2)}</td><td>{num(e['t_vs_0'], 1)}</td>"
                         f"<td>{num(e['t_vs_1'], 1)}</td></tr>")
    t2 = ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Component</th><th class='l'>Level of</th>"
          "<th>Elasticity</th><th>s.e.</th><th>t vs 0</th><th>t vs 1</th></tr></thead>"
          f"<tbody>{''.join(erows)}</tbody></table></div>")
    gate = esc(res.get("_gate", {}).get("rule", "").replace("(decided before running):", "").strip())
    gate += " The gate chose levels for France and Italy; Germany uses levels on theory."
    return ("<h3>Which specification, and why</h3><p>Mean absolute error of the cumulative cash requirement, bn, "
            "walk-forward every third month with realised drivers for every specification, on the origins all three "
            "share. PRE = windows ending by 2019 (2021 for Italy); OOS = origins from 2022, reported only.</p>"
            + t1 + f"<p class='muted'>Gate, set before running: {gate}</p>"
            + "<h3>Elasticities estimated freely (log level on log level)</h3>" + t2
            + "<p class='muted'>No receipts elasticity is distinguishable from one (|t vs 1| below 1.1), which is "
              "what the share-of-level specification imposes.</p>")


def curve_table(pack: dict) -> str:
    cf = pack.get("curve_factors")
    if not cf:
        return ""
    rows = []
    for m, f in cf["markets"].items():
        if "error" in f:
            rows.append(f"<tr><td>{esc(m.upper())}</td><td class='l' colspan='13'>{esc(f['error'])}</td></tr>")
            continue
        used_by = ", ".join(c.upper() for c, mm in cf["source_for"].items() if mm == m)
        ex = f["explained"]
        rows.append(f"<tr><td>{esc(m.upper())}</td><td class='l'>{esc(used_by)}</td>"
                    f"<td class='l'>{esc(f['sample'][0][:7])} to {esc(f['sample'][1][:7])}</td>"
                    f"<td>{100 * ex[0]:.0f}% / {100 * ex[1]:.0f}% / {100 * ex[2]:.0f}%</td>"
                    + "".join(f"<td>{100 * w:.0f}</td>" for w in f["w3m"]) + "</tr>")
    grid = next(f["grid"] for f in cf["markets"].values() if "grid" in f)
    head = "".join(f"<th>{g:g}y</th>" for g in grid)
    rows10 = []
    for m, f in cf["markets"].items():
        if "error" in f:
            continue
        rows10.append(f"<tr><td>{esc(m.upper())}</td>" + "".join(f"<td>{100 * w:.0f}</td>" for w in f["w10y"]) + "</tr>")
    return ("<div class='tablewrap'><table><thead><tr><th>Curve</th><th class='l'>Used for</th><th class='l'>Sample</th>"
            f"<th>Variance: level / slope / curvature</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='muted'>Right-hand columns: bp at each maturity for +100bp at 3 months and 0 at 10 years. "
            "Germany's, the UK's and Japan's histories start at 6 months or 1 year, so their 3-month point is the "
            "shortest maturity in the history.</p>"
            f"<div class='tablewrap'><table><thead><tr><th>Curve</th>{head}</tr></thead><tbody>{''.join(rows10)}</tbody></table></div>"
            "<p class='muted'>bp at each maturity for +100bp at 10 years and 0 at 3 months.</p>")


def channel_table(pack: dict, types: tuple[str, ...]) -> str:
    rows = []
    for c in pack["countries"].values():
        for ch in c["channels"]:
            if ch["type"] not in types:
                continue
            param = ""
            if ch["type"] == "elasticity":
                param = f"ε = {num(ch['elasticity'])}; base {', '.join(num(b, 1) for b in ch['base'])}"
            elif ch["type"] == "matrix":
                param = f"{ch['kind']}: " + ", ".join(ch.get("determinants", []))
            elif ch["type"] == "linear":
                pu = ch["per_unit"]
                param = "; ".join(f"{m}: " + ", ".join(num(v, 4) for v in vals) for m, vals in pu.items())
            src = f"<a href='{esc(ch['url'])}'>{esc(ch['source'])}</a>" if ch.get("url") else esc(ch["source"])
            rows.append(f"<tr><td>{esc(c['name'])}</td><td class='l'>{esc(ch['driver'])} → {esc(ch['line'])}</td>"
                        f"<td class='l'>{esc(ch['badge'])}</td><td class='l'>{param}</td><td class='l'>{src}</td></tr>")
    if not rows:
        return ""
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Channel</th><th class='l'>Kind</th>"
            "<th class='l'>Parameters (per unit of shock)</th><th class='l'>Source</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def rates_benchmark(pack: dict) -> str:
    """Our ISIN book at +100bp next to the published rule, where both exist."""
    rows = []
    for c in pack["countries"].values():
        book = next((ch for ch in c["channels"] if ch["id"].endswith("_rates_book")), None)
        if not book:
            continue
        i100 = book["knots"].index(100.0)
        ours = list(book["values"][i100])
        for ch in c["channels"]:
            if ch["driver"] == "rates" and ch.get("method_group") == "book" and ch["type"] == "matrix":
                sign = next(ln["sign"] for ln in c["lines"] if ln["id"] == ch["line"])
                x = [100.0 * w for w in ch.get("exposure", [1.0] * len(ch["tau"]))]
                ours = [o + sign * sum(x[j] * ch["coef"][j][p] for j in range(len(x)))
                        for p, o in enumerate(ours)]
        pub = [0.0] * len(c["periods"])
        has_pub = False
        for ch in c["channels"]:
            if ch["driver"] != "rates" or ch.get("method_group") != "authority":
                continue
            has_pub = True
            sign = next(ln["sign"] for ln in c["lines"] if ln["id"] == ch["line"])
            if ch["type"] == "linear":
                vals = [100.0 * v for v in ch["per_unit"]["level"]]
            else:  # matrix, rate kind: +100bp every year from the first
                x = [100.0 * w for w in ch.get("exposure", [1.0] * len(ch["tau"]))]
                vals = [sum(x[j] * ch["coef"][j][p] for j in range(len(x))) for p in range(len(c["periods"]))]
            pub = [a + sign * v for a, v in zip(pub, vals)]
        cells = "".join(f"<td>{num(o, 1)}</td><td>{num(p, 1) if has_pub else '—'}</td>" for o, p in zip(ours, pub))
        rows.append(f"<tr><td>{esc(c['name'])} ({esc(c['unit'])})</td>{cells}</tr>")
    head = "".join(f"<th>P{i + 1} ours</th><th>P{i + 1} published</th>" for i in range(4))
    return (f"<div class='tablewrap'><table><thead><tr><th>Country</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='muted'>P1 to P4 are each country's four periods (see the identity table). Both sides start on the "
            "same date and leave out interest on the extra borrowing. Published: AFT triangle (France), OBR ready reckoner, "
            "long rates and Bank Rate (UK), CBO rules of thumb (US). Ours: the ISIN book, plus interest on QE reserves "
            "for the UK at OBR's own Bank Rate coefficient. What is left is each office's own debt projection and "
            "vintage: OBR's reckoner dates from March 2025.</p>")


def checks_table(pack: dict) -> str:
    rows = []
    for cc, r in (pack.get("checks") or {}).items():
        if cc == "_curve_linearity":
            if "error" in r:
                rows.append(f"<tr><td>All</td><td class='l' colspan='2'>curve check failed: {esc(r['error'])}</td></tr>")
                continue
            for c2, x in r.items():
                nm = pack["countries"].get(c2, {}).get("name", c2)
                rows.append(f"<tr><td>{esc(nm)}</td><td class='l'>ISIN book re-run on three curve shapes vs adding the "
                            f"3m and 10y responses ({x['pct_of_interest']:.2f}% of the interest bill)</td>"
                            f"<td>{x['max_abs_err_bn']:.1e}</td></tr>")
            continue
        if cc == "_rates_interpolation":
            if "error" in r:
                rows.append(f"<tr><td>All</td><td class='l' colspan='2'>rates check failed: {esc(r['error'])}</td></tr>")
                continue
            for c2, x in r.items():
                nm = pack["countries"].get(c2, {}).get("name", c2)
                rows.append(f"<tr><td>{esc(nm)}</td><td class='l'>ISIN book re-run at +50, +150, +250bp vs the straight "
                            f"line between runs ({x['pct_of_interest']:.2f}% of the interest bill)</td>"
                            f"<td>{x['max_abs_err_bn']:.1e}</td></tr>")
            continue
        name = pack["countries"].get(cc, {}).get("name", cc)
        if "error" in r:
            rows.append(f"<tr><td>{esc(name)}</td><td class='l' colspan='2'>check failed: {esc(r['error'])}</td></tr>")
            continue
        if cc == "uk":
            for d, x in r.items():
                rows.append(f"<tr><td>{esc(name)}</td><td class='l'>{esc(d)} every year vs ready_reckoner.run</td>"
                            f"<td>{x['max_abs_err_bn']:.1e}</td></tr>")
            continue
        for d, e in r.get("vs_published_table_bn", {}).items():
            rows.append(f"<tr><td>{esc(name)}</td><td class='l'>{esc(d)}: published unit-shock table</td><td>{e:.1e}</td></tr>")
        for x in r.get("vs_full_model_bn", []):
            what = "a year (growth)" if x.get("mode", "growth") == "growth" else "one-off level"
            rows.append(f"<tr><td>{esc(name)}</td><td class='l'>full SARIMAX re-run, NGDP {x['ngdp_pp']:+.2f} and CPI "
                        f"{x['cpi_pp']:+.2f} together, {what}</td><td>{x['max_abs_err']:.1e}</td></tr>")
    return ("<div class='tablewrap'><table><thead><tr><th>Country</th><th class='l'>Test</th>"
            "<th>Largest gap, bn</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def assumptions(pack: dict) -> str:
    items = []
    for c in pack["countries"].values():
        for ch in c["channels"]:
            if ch.get("badge") == "Assumed elasticity":
                items.append(f"<li>{esc(c['name'])}: {esc(ch['driver'])} → {esc(ch['line'])}, ε = {num(ch['elasticity'], 2)}</li>")
    if not items:
        return ""
    return ("<div class='flag'><b>Assumed, not estimated.</b> These responses have no shipped model behind them. "
            "They are stated so they can be challenged, and each has a 0 to 2× multiplier on the dashboard."
            f"<ul>{''.join(items)}</ul></div>")


WORKED_JS = """
const PACK = JSON.parse(document.getElementById('pack').textContent);
__ENGINE__
const sel = document.getElementById('wk-country');
const DRV = PACK.drivers || {};
const lineLabel = (c, id) => (c.lines.find((l) => l.id === id) || {}).label || id;
const drvLabel = (d) => (DRV[d] || {}).label || d;
for (const [cc, c] of Object.entries(PACK.countries)) sel.appendChild(new Option(c.name, cc));
function num(id, fallback) {
  const el = document.getElementById(id);
  const v = el ? parseFloat(el.value) : NaN;
  return Number.isFinite(v) ? v : fallback;
}
function shocksFor(c) {
  const ngdpMode = (c.modes?.ngdp || ['growth']).includes('growth') ? 'growth' : 'level';
  const s = {
    ngdp: { size: num('wk-ngdp', 1), mode: ngdpMode },
    cpi: { size: num('wk-cpi', 1), mode: 'growth' },
    rates: { size: num('wk-rates', 100) },
  };
  if (c.modes?.brent || c.channels.some((ch) => ch.driver === 'brent'))
    s.brent = { size: num('wk-brent', 10) };
  if (document.getElementById('wk-feedback')?.checked) s.feedback = true;
  return s;
}
function describe(s, unit) {
  const bits = [];
  if (s.ngdp) bits.push(`nominal GDP ${s.ngdp.size >= 0 ? '+' : ''}${s.ngdp.size}${s.ngdp.mode === 'growth' ? 'pp/yr' : '%'}`);
  if (s.cpi) bits.push(`inflation ${s.cpi.size >= 0 ? '+' : ''}${s.cpi.size}pp/yr`);
  if (s.rates) bits.push(`yields ${s.rates.size >= 0 ? '+' : ''}${s.rates.size}bp`);
  if (s.brent) bits.push(`Brent ${s.brent.size >= 0 ? '+' : ''}${s.brent.size}%`);
  bits.push(s.feedback ? 'feedback on' : 'no feedback');
  return `${unit}; ${bits.join(', ')}.`;
}
function worked() {
  const cc = sel.value, c = PACK.countries[cc];
  const s = shocksFor(c);
  document.getElementById('wk-brent-wrap').style.display =
    (c.modes?.brent || c.channels.some((ch) => ch.driver === 'brent')) ? '' : 'none';
  const r = evaluate(PACK, cc, s);
  const f = (v) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3);
  let rows = '';
  const pairCount = {};
  for (const ch of c.channels) pairCount[`${ch.driver}|${ch.line}`] = (pairCount[`${ch.driver}|${ch.line}`] || 0) + 1;
  for (const ch of c.channels) {
    const chosen = (c.default_methods || {})[ch.driver];
    if (ch.method_group && ch.method_group !== chosen) continue;
    if (['r3m', 'r10y'].includes(ch.driver)) continue;
    if (!c.modes?.[ch.driver]?.includes(s[ch.driver]?.mode) && ['ngdp','cpi'].includes(ch.driver) && c.modes?.[ch.driver]) continue;
    if (!(ch.driver in s) && ch.driver !== 'rates') continue;
    const d = channelDelta(ch, s, c.periods.length, c.month_period || null);
    const sign = c.lines.find((l) => l.id === ch.line).sign;
    let name = `${drvLabel(ch.driver)} → ${lineLabel(c, ch.line)}`;
    if (pairCount[`${ch.driver}|${ch.line}`] > 1)
      name += ` (${ch.badge || ch.method_group || 'alt'})`;
    rows += `<tr><td class="l">${name}</td><td class="l">${ch.badge || ''}</td><td>${sign > 0 ? '+1' : '−1'}</td>` +
            d.map((v) => `<td>${f(sign * v)}</td>`).join('') + '</tr>';
  }
  rows += `<tr><td class="l"><b>Total, as the dashboard shows</b></td><td></td><td></td>` + r.delta.map((v) => `<td><b>${f(v)}</b></td>`).join('') + '</tr>';
  document.getElementById('wk-table').innerHTML = `<thead><tr><th class="l">Channel</th><th class="l">Kind</th><th>σ</th>${c.periods.map((p) => `<th>${p}</th>`).join('')}</tr></thead><tbody>${rows}</tbody>`;
  document.getElementById('wk-unit').textContent = describe(s, c.unit);
}
sel.onchange = worked;
for (const id of ['wk-ngdp','wk-cpi','wk-rates','wk-brent','wk-feedback']) {
  const el = document.getElementById(id); if (el) el.addEventListener('input', worked);
}
worked();
document.querySelectorAll('[data-tex]').forEach((el) => {
  if (typeof katex !== 'undefined') katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false });
});
"""


def shock_sections(pack: dict) -> list[tuple[str, str, str]]:
    """Shock-response sections (identity → limits). Caller owns ``FIG_NUMBERS`` and section numbering.

    Must be called after any preceding ``fig(...)`` calls so figure numbers continue the page sequence.
    """
    tri = ["fr", "de", "it"]
    worked = (
        intro("Pick a country and set the shocks: each row is one channel on its own; the rows add up to what "
              "the dashboard shows for the same inputs. Do this once before reading the country models.")
        + "<div class='worked'>"
        "<div class='controls'>"
        "<label>Country <select id='wk-country'></select></label>"
        "<label>Nominal GDP <input id='wk-ngdp' type='number' step='0.25' value='1'></label>"
        "<label>Inflation (pp) <input id='wk-cpi' type='number' step='0.25' value='1'></label>"
        "<label>Yields (bp) <input id='wk-rates' type='number' step='25' value='100'></label>"
        "<label id='wk-brent-wrap'>Brent (%) <input id='wk-brent' type='number' step='5' value='10'></label>"
        "<label style='flex-direction:row;align-items:center;gap:6px'>"
        "<input id='wk-feedback' type='checkbox'> Interest on extra debt</label>"
        "</div>"
        "<p class='muted' id='wk-unit'></p><div class='tablewrap'><table id='wk-table'></table></div></div>"
        "<p>σ is the channel's sign in the identity (@sec:identity@).</p>"
    )
    return [
        ("identity", "The borrowing identity",
         eq_block("identity")
         + fig("identity", "fig-identity",
               "France, stagflation preset: each line's change times its sign σ, stacked; the diamonds are ΔB, the "
               "total the dashboard shows. The bars add up to the diamonds exactly.",
               cap_id="cap-identity")
         + fold("Identity for each country (measure, signs, baseline source)", identity_table(pack))),
        ("gap", "The size of the shock",
         eq_block("gap")
         + fig("gap", "fig-gap",
               "Driver gap g for a 1% shock. A one-off level shift holds the driver 1% above baseline from "
               "the first month; a shock every year compounds, so the gap grows to 4.1% after four years.")),
        ("worked", "Check one yourself", worked),
        ("sarimax", "France, Germany, Italy: the baseline model",
         eq_block("perimeter") + eq_block("sarimax")
         + fig("shares", [f"fig-shares-{c}" for c in tri], tall=True,
               caption="Each driven line's share ω of its driver, last full outturn year = 100, outturn then "
                       "forecast (shaded). 'Persists' carries the latest share forward; 'held' keeps last "
                       "year's; Germany's spending carries forward with the Covid months left out of the fit.")
         + fold("Specification choice and fitted coefficients",
                selection_tables(pack) + "<h3>Fitted coefficients</h3>" + sarimax_table(pack))
         + "<h3>Walk-forward test of the multi-year forecast</h3>"
         + "<p>The chosen rules are scored one to three calendar years ahead against the alternatives, given the "
           "macro that happened. @fig:bt@ gives the error by horizon and @fig:err@ the two-year error year by "
           "year.</p>"
         + fig("bt", [f"fig-bt-{c}" for c in tri],
               "Multi-year test: mean absolute error of the cash requirement excluding interest, one to three "
               "calendar years ahead of each December from 2013, given the macro that happened, Covid years "
               "excluded.")
         + fig("err", [f"fig-err-{c}" for c in tri],
               "The same test, two years ahead, year by year: forecast minus outturn. Grey bands are 2020 "
               "and 2021. A good model sits near zero and does not stay on one side.")
         + fold("Walk-forward scores by rule", multi_year_table(pack))
         + "<h3>The resulting baseline</h3>"
         + "<p>With the lines and the perimeter P in place, @fig:forecast@ shows every country's baseline B "
           "beside the official forecast (never used). France, Germany and Italy are the monthly model; the "
           "other panels are the annual models of @sec:own@.</p>"
         + fig("forecast", [f"fig-forecast-{c}" for c in pack["countries"]],
               "Our forecast (blue) after the outturn (black), our forecast plus legislated policy (green, "
               "built in @sec:policy@), against our cash board (orange, dashed, France, Germany, Italy) and "
               "the official forecast (purple diamonds). Official forecasts are shown, never used; after the "
               "year in progress each board runs on our forecast, so those two lines meet.")
         + fold("Baseline by year: modelled lines, perimeter P, official forecast", forecast_table(pack))),
        ("level", "France, Germany, Italy: the response to a shock",
         eq_block("level")
         + fig("level", "fig-level",
               "France's receipts: a one-off 1% higher GDP level moves them by the same amount every year "
               "(the line is a share of the level); 1pp faster growth every year compounds.")),
        ("own", "United Kingdom, United States, Spain, Sweden, Norway, Japan: annual models",
         eq_block("own")
         + fig("own", ["fig-own-receipts", "fig-own-primary"],
               "Walk-forward test for the annual models: mean absolute error of each line one to three years "
               "ahead, % of GDP, persist against revert. The chosen rule is the solid bar.")
         + fold("Persist vs revert scores by country", own_table(pack))),
        ("elasticity", "Spain, Sweden, Norway, Japan: the response to a shock",
         eq_block("elasticity")
         + fig("elasticity", "fig-elasticity",
               "Response as a power of the level, (1+g/100)^ε − 1, not a straight line, so large shocks are "
               "exact. Diamonds: the elasticities in use, at a 5% shock.")
         + fold("Elasticities in use", assumptions(pack) + channel_table(pack, ("elasticity",)))),
        ("matrix", "United Kingdom: OBR's published reckoner, for comparison",
         eq_block("matrix")
         + fig("matrix", ["fig-matrix-a", "fig-matrix-b"],
               "Two of OBR's ready-reckoner matrices as used: a shock in year s (rows) moves the line in year t "
               "(columns). The persistent rates matrix carries a shock into every later year.")
         + fold("Ready-reckoner channel parameters", channel_table(pack, ("matrix",)))),
        ("linear", "United States and published triangles, for comparison",
         eq_block("linear")
         + fig("linear", "fig-linear",
               "Per-unit responses by year, as published (CBO rules of thumb, AFT's +100bp triangle), moved to "
               "the same start date as our own models.", tall=True)
         + fold("Published per-unit responses", channel_table(pack, ("linear",)))),
        ("policy", "Legislated policy on top of the model",
         eq_block("policy")
         + fig("policy", [f"fig-policy-{c}" for c, v in pack["countries"].items() if v.get("policy")],
               "Each legislated measure's effect on borrowing beyond what the model already carries, stacked: "
               "above zero adds to borrowing. The six largest are named; the rest are summed as other tax and "
               "other spending. Black diamonds: the net, which is the gap between the two forecast lines in "
               "@fig:forecast@.", tall=True)
         + fold("Policy measures by country", policy_table(pack))
         + fold("Cross-checks against independent sources", crosscheck_table(pack))),
        ("bridge", "Our forecast against each authority's",
         eq_block("bridge")
         + fig("bridge", [f"fig-bridge-{c}" for c, v in (pack.get("authority_bridge") or {}).items()
                          if not v.get("total_only")],
               "Our forecast minus the authority's, by line, each year: receipts, spending excluding interest, "
               "interest, and the definitional difference already present in the last outturn year. Above "
               "zero: that line makes our deficit larger than theirs. Diamonds: the total gap.")
         + fold("Bridge by country and year", bridge_table(pack))),
        ("book", "Bond yields: the debt book",
         eq_block("book")
         + fig("book", [f"fig-book-{c}" for c in tri],
               "The ISIN book is re-run at each marker (−200 to +300bp); the dashboard reads between markers "
               "along the lines. Each line is one year; later years are steeper as more of the debt has been "
               "refinanced at the shifted rates. The lines are nearly straight, so reading between markers "
               "costs little (checked in @sec:checks@).")
         + fold("Our model against published rules, +100bp", rates_benchmark(pack))),
        ("curve", "Bond yields: the shape of the curve",
         eq_block("curve")
         + fig("curve", ["fig-curve-3m", "fig-curve-10y"],
               "How the rest of the curve moves when only the 3-month or only the 10-year point is set, from "
               "the history of daily curve moves since 2000 in each market.")
         + fold("Conditional-mean curve weights", curve_table(pack)
                + "<p>Pinning only the first two components instead (curvature forced to zero) bends the curve "
                  "away from the anchors: +100bp at both gave +137bp at 2 to 3 years in the US, and a "
                  "3-month-only move sent Japan's 30-year down 142bp. The conditional version keeps the anchors "
                  "and stays within 73 to 117bp.</p>")),
        ("feedback", "Interest on the extra debt",
         eq_block("feedback")
         + fig("feedback", "fig-feedback",
               "United Kingdom, yields +100bp and nominal GDP 1% lower: the same shock with F off (blue) and on "
               "(orange). The gap between the bars is F; it compounds because each year's extra borrowing is "
               "borrowed again.",
               cap_id="cap-feedback")),
        ("checks", "How we know the dashboard is the model",
         intro("Every pack build re-runs the full models and compares them with the dashboard's arithmetic. The "
               "browser engine is the same code as the Python engine; the test suite holds them to 1e-9 on "
               "hundreds of random shocks. @fig:checks@ shows the largest gap for each country.")
         + fig("checks", "fig-checks",
               "Largest gap between the dashboard and a full re-run of each model, on random shocks "
               "(log scale). Rounding error, ten million times below the test tolerance.")
         + fold("Parity checks by country", checks_table(pack))),
        ("limits", "What this does not do",
         fold("Limitations and what is done about each", limitations(pack))),
    ]


def redirect_html(target: str = "methodology.html", anchor: str = "") -> str:
    """Stub for a retired page: forwards old links (and their #anchors) to the Methodology."""
    href = f"{target}#{anchor}" if anchor else target
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Methodology</title>
<meta http-equiv="refresh" content="0; url={esc(href)}"/>
<script>location.replace({json.dumps(href)} + ({json.dumps(not anchor)} ? location.hash : ""));</script>
</head><body><p>This page has moved to <a href="{esc(href)}">Methodology</a>.</p></body></html>"""
