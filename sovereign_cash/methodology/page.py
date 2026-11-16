"""Methodology: production forecast chain + shock-response sections in one document.

``render(pack)`` composes ``methodology.registry`` sections then ``equations.page.shock_sections``.
``equations.html`` is only a redirect stub (written by ``web.pages`` / ``build``).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from sovereign_cash.equations.page import (
    CSS,
    FIGURES_JS,
    KATEX_CDN,
    PLOTLY_CDN,
    VENDOR,
    WORKED_JS,
    eq_block,
    esc,
    fold,
    prose_math,
    redirect_html,
    reset_figures,
    resolve_refs,
    shock_sections,
)
from sovereign_cash.methodology.registry import EQUATIONS, SECTIONS, _ORDER
from sovereign_cash.scenarios.pack import PACK_PATH

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "methodology.html"
ALIAS = HERE / "bottom_up_methods.html"
ENGINE_JS = HERE / "web" / "engine.js"


def _engine_inline() -> str:
    """engine.js without ``export`` so it runs in the page script."""
    return re.sub(r"^export\s+", "", ENGINE_JS.read_text(encoding="utf-8"), flags=re.M)


def _table(headers: list[str], rows: list[list[str]], *, right: set[int] | None = None) -> str:
    right = right or set()
    head = "".join(
        f"<th class='r'>{h}</th>" if i in right else f"<th>{h}</th>" for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f"<td class='r'>{c}</td>" if i in right else f"<td>{c}</td>" for i, c in enumerate(r)
        ) + "</tr>"
        for r in rows
    )
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _primary_methods_table() -> str:
    from sovereign_cash.scenarios import primary as P

    try:
        P.write(P.build())
    except Exception:  # noqa: BLE001
        pass
    path = P.PATH if P.PATH.exists() else HERE / "derived" / "primary_path.json"
    if not path.exists():
        return ("<p class='muted'>primary_path.json not built — run "
                "<code>python -m sovereign_cash.scenarios.primary</code>.</p>")
    d = json.loads(path.read_text())
    names = {
        "us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy",
        "es": "Spain", "se": "Sweden", "no": "Norway", "jp": "Japan",
    }
    rows = []
    for cc, c in d.get("countries", {}).items():
        m = c.get("methods") or {}
        rec = "fiscal_drag" if cc == "uk" else m.get("receipts", "—")
        pri = m.get("primary", "—")
        w = c.get("wedge_pct")
        w_s = "—" if w is None else f"{w:+.2f}".replace("-", "−")
        rows.append([
            names.get(cc, cc), esc(c.get("measure", "")), str(c.get("origin_period", "")),
            rec, pri, w_s, str(c.get("policy_n_measures", "")),
        ])
    return (
        _table(
            ["Country", "Measure", "T", "Receipts rule", "Primary rule", "ω, pp GDP", "# policy measures"],
            rows, right={2, 5, 6},
        )
        + "<p class='note'>T = last complete outturn. ω held at last common outturn vs AMECO GG. "
        "UK receipts always fiscal_drag in production. Source: derived/primary_path.json.</p>"
    )


def _uk_blocks() -> str:
    return _table(
        ["Block ℓ", "Driver d", "β source"],
        [
            ["Income tax / NICs", "Average earnings", "OBR ready reckoner"],
            ["VAT", "Retail spending", "OBR ready reckoner"],
            ["Corporation tax", "Whole-economy output", "OBR ready reckoner"],
            ["Other tax", "Retail + UK equities", "OBR ready reckoner"],
            ["Welfare", "CPI", "OBR ready reckoner"],
            ["Departmental / capital", "Output (modest)", "OBR ready reckoner"],
            ["Debt interest", "ISIN book + rates triangle", "Book + OBR"],
            ["Financing residual", "Seasonal; soft remit pin", "DMO remit when present"],
        ],
    )


def _us_scores() -> str:
    return (
        _table(
            ["Metric", "h=1", "h=5", "h=10", "h=20 bdays"],
            [
                ["Cum. net op. flow MAE $bn — model / last-year / flat (OOS)",
                 "9.1 / 27.3 / 30.1", "18.3 / 31.2 / 65.5", "24.6 / 43.7 / 107.2", "33.9 / 60.8 / 196.3"],
                ["TGA level MAE $bn — steer_hold / flat / oracle financing",
                 "—", "23.8 / 49.9 / 18.3", "53.2 / 75.6 / 24.6", "87.3 / 102.8 / 33.9"],
            ],
        )
        + "<p class='note'>PRE 2022–23 origins; OOS 2024→; weekly. steer_target is conditional — not the quoted forecast.</p>"
    )


# Callables evaluated in render() so the hold table always rebuilds primary_path.json.
EXTRAS: dict[str, str | Callable[[], str]] = {
    "chain": (
        "<div class='flag'><b>Scope.</b> Sections 1–"
        f"{len(SECTIONS)} are the production forecast chain (Cash, US, Debt, Supply, primary exhibits). "
        "The shock-response half below recomputes the same pack the dashboard uses — do not mix the two MAE sets. "
        "It is the dashboard's aggregate model, not the note's: the note's scenarios run the line model of "
        "@sec:scenlines@. By default the dashboard moves receipts and spending on aggregate own models (fitted "
        "share-of-level models for France, Germany and Italy; annual own-model elasticities for the UK, the US, Spain, "
        "Sweden, Norway and Japan) and interest on the ISIN book; the OBR reckoner and CBO's per-unit responses "
        "(including Fed remittances that fall as rates rise) are its alternative 'authority' method. The note's line model instead moves each budget line on its driver, puts corporate tax on a "
        "common profit path, uprates 80% of spending a year after prices, lets linkers accrue, pays no Fed remittance "
        "while the Fed carries a deferred asset, moves the BoJ's payment to the Treasury and adds the UK's Bank Rate cost on reserves.</div>"
    ),
    "hold": lambda: fold("Shipped share rules, measures and wedges", _primary_methods_table()),
    "cashuk": fold("UK category × driver map", _uk_blocks()),
    "ustga": fold("Scored MAEs (OOS)", _us_scores()),
    "newmoney": (
        "<p class='note'>Shock-slider arithmetic continues in <a href='#part-shock'>Shock response</a> "
        "(identity, gap, FR/DE/IT monthly shares, annual own models, policy, bridge, book, curve, feedback).</p>"
    ),
}


def _extra(eid: str) -> str:
    v = EXTRAS.get(eid, "")
    return v() if callable(v) else v


def _pack_json(pack: dict) -> str:
    return json.dumps(pack, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def render(pack: dict | None = None) -> str:
    """One Methodology document: production chain, then shock sections; pack JSON embedded for the vintage gate."""
    if pack is None:
        if not PACK_PATH.exists():
            raise FileNotFoundError(f"scenario pack missing: {PACK_PATH}")
        pack = json.loads(PACK_PATH.read_text())

    missing = [eid for eid in _ORDER if eid not in EQUATIONS]
    if missing:
        raise RuntimeError(f"methodology _ORDER ids not in EQUATIONS: {missing}")

    reset_figures()
    prod: list[tuple[str, str, str]] = []
    for sid, title, eids in SECTIONS:
        parts = []
        for eid in eids:
            parts.append(eq_block(EQUATIONS[eid]))
            parts.append(_extra(eid))
        prod.append((sid, title, "".join(parts)))

    shock = shock_sections(pack)
    all_secs = prod + shock
    sec_numbers: dict[str, int] = {sid: i + 1 for i, (sid, _, _) in enumerate(all_secs)}
    # Equation ids resolve to their enclosing section number (@sec:deficit@ → wedge section).
    for sid, _, eids in SECTIONS:
        n = sec_numbers[sid]
        for eid in eids:
            sec_numbers[eid] = n

    toc = "".join(
        f"<a href='#{sid}'>{sec_numbers[sid]}. {esc(title)}</a>" for sid, title, _ in all_secs
    )
    body_parts = []
    for i, (sid, title, html_body) in enumerate(all_secs):
        n = sec_numbers[sid]
        if i == len(prod):
            body_parts.append("<div id='part-shock'></div>")
        body_parts.append(f"<h2 id='{sid}'>{n}. {esc(title)}</h2>{html_body}")
    body = resolve_refs("".join(body_parts), sec_numbers)

    lead = (
        "<p class='lead'>One document for the sovereign financing chain and the shock-response arithmetic. "
        "Each section: intro → formula → every symbol defined → code path (and a folded audit table where one ships). "
        f"Production chain is sections 1–{len(SECTIONS)}; shock response starts at "
        f"<a href='#part-shock'>section {len(SECTIONS) + 1}</a>. "
        "Beyond T+3 the GG primary is the IMF WEO primary, except the US, whose line model runs to FY2030; interest / "
        "Commission / OBR / CBO totals stay comparators. Scenarios come in two forms: the note's and the debt board's "
        "(heatmap, repression, downturn lines) run one line model for growth, inflation and rates "
        "(<a href='#scenlines'>Section 10</a>); the dashboard's sliders run the aggregate channels of the "
        "shock-response half. Where they differ, Section 10 is the one the note quotes.</p>"
    )
    readfirst = resolve_refs(
        """<div class='readfirst'>
<p><b>Read this first.</b> Multi-year primary / debt / interest exhibits: @sec:shares@–@sec:debtpath@.
Live Cash tab: @sec:cashid@–@sec:cashjp@. US tab: @sec:usflow@–@sec:ustga@.
Note scenarios (heatmap, repression, downturn lines): @sec:scenlines@. Shock sliders on the dashboard map to
<a href='#part-shock'>Shock response</a>.</p>
</div>""",
        sec_numbers,
    )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Methodology</title>
<link rel="stylesheet" href="{VENDOR}/katex.min.css">
<script src="{VENDOR}/katex.min.js"></script>
<script src="{VENDOR}/plotly.min.js"></script>
<style>{CSS}</style>
</head><body><div class="wrap">
<h1>Methodology</h1>
{lead}
{readfirst}
<nav class="toc">{toc}</nav>
{body}
<p class="muted" style="margin-top:48px">Generated by <code>sovereign_cash.methodology.page</code>.
<a href="#">Back to top</a></p>
</div>
<script type="application/json" id="pack">{_pack_json(pack)}</script>
<script>
{FIGURES_JS}
{WORKED_JS.replace("__ENGINE__", _engine_inline())}
(async function bootMethodology() {{
  function loadScript(src) {{
    return new Promise((resolve, reject) => {{
      const s = document.createElement("script");
      s.src = src; s.onload = () => resolve(); s.onerror = () => reject(new Error(src));
      document.head.appendChild(s);
    }});
  }}
  function loadCss(href) {{
    return new Promise((resolve) => {{
      const l = document.createElement("link");
      l.rel = "stylesheet"; l.href = href; l.onload = () => resolve(); l.onerror = () => resolve();
      document.head.appendChild(l);
    }});
  }}
  if (typeof katex === "undefined") {{
    await loadCss("{KATEX_CDN}/katex.min.css");
    try {{ await loadScript("{KATEX_CDN}/katex.min.js"); }} catch (e) {{ console.error(e); }}
  }}
  if (typeof Plotly === "undefined") {{
    try {{ await loadScript("{PLOTLY_CDN}"); }} catch (e) {{ console.error(e); }}
  }}
  document.querySelectorAll("[data-tex]").forEach((el) => {{
    if (typeof katex !== "undefined")
      katex.render(el.dataset.tex, el, {{ displayMode: true, throwOnError: false }});
  }});
  if (typeof drawMethodologyFigures === "function") drawMethodologyFigures();
}})();
</script>
</body></html>"""


def build(pack: dict | None = None) -> Path:
    html_text = render(pack)
    OUT.write_text(html_text, encoding="utf-8")
    ALIAS.write_text(redirect_html("methodology.html"), encoding="utf-8")
    print(f"Wrote {OUT} (alias {ALIAS.name} → redirect)")
    return OUT


if __name__ == "__main__":
    build()
