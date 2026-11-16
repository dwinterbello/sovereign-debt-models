"""One board over cash, the debt stock, market supply and the US fiscal stack.

Pages stay where their silos write them. Same-directory cash and debt pages are
framed by name. The US fiscal pages and the issuance board are one directory up,
which is also where publish places them. Writes ``financing_dashboard.html``.

US is its own top tab (Cash / Flows / Ceiling / Liquidity / Report). Other
countries stay under Cash. The old combined ``US funding`` tab is folded in.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.build_financing
"""
from __future__ import annotations

from pathlib import Path

from sovereign_cash.build_hub import COUNTRIES, HERE

OUT = HERE / "financing_dashboard.html"
REPO = HERE.parent
US = REPO / "us" / "fiscal"

CSS = """
html, body { height: 100%; margin: 0; }
body { font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1d2433; }
.hub { display: flex; flex-direction: column; height: 100%; }
nav { display: flex; gap: 2px; padding: 6px 12px 0; background: #f5f7fa; }
nav.main { border-bottom: 1px solid #d8dde6; }
nav.sub { display: none; border-bottom: 1px solid #d8dde6; padding-top: 0; }
nav.sub.on { display: flex; }
nav a { font: 600 11px/1 ui-monospace, Menlo, monospace; letter-spacing: .06em; text-transform: uppercase;
  color: #6b7385; padding: 8px 12px; border: 1px solid transparent; border-bottom: none; text-decoration: none; cursor: pointer; }
nav.sub a { text-transform: none; letter-spacing: 0; font-weight: 600; }
nav a.on { color: #1d2433; border-color: #d8dde6; background: #fff; margin-bottom: -1px; }
main { flex: 1; position: relative; }
main iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; display: none; }
main iframe.on { display: block; }
"""


def _us_tabs() -> list[tuple[str, str, str]]:
    """US sub-tabs that exist on disk."""
    cand = [
        ("us-cash", "Cash", "tga_deficit_dashboard.html"),
        ("us-flows", "Flows", "tga_components_dashboard.html"),
        ("us-ceiling", "Ceiling", "tga_ceiling_dashboard.html"),
        ("us-liq", "Liquidity", "liquidity_dashboard.html"),
        ("us-report", "Report", "report/US_Fiscal_Funding_Report_selfcontained.html"),
    ]
    out = []
    for key, lab, rel in cand:
        if (US / rel).exists():
            out.append((key, lab, f"../us/fiscal/{rel}"))
    return out


def items() -> list[tuple[str, str, str | None, list[tuple[str, str, str]]]]:
    """(group, label, src or None, children). A child is (key, label, src)."""
    cash: list[tuple[str, str, str]] = []
    if (HERE / "fytd_cash_dashboard.html").exists():
        cash.append(("fytd", "FYTD", "fytd_cash_dashboard.html"))
    for cc, name, _ in COUNTRIES[1:]:
        src = f"{cc}_cash_dashboard.html"
        if (HERE / src).exists():
            cash.append((cc, name, src))
    # Spain is thin_cash / IGAE (not in COUNTRIES — no full runner).
    if (HERE / "es_cash_dashboard.html").exists() and not any(c[0] == "es" for c in cash):
        cash.append(("es", "Spain", "es_cash_dashboard.html"))
    out: list[tuple[str, str, str | None, list[tuple[str, str, str]]]] = [
        ("overview", "Overview", "sovereign_cash_overview.html", []),
    ]
    if (HERE / "shock_dashboard.html").exists():
        out.append(("shocks", "Shock scenarios", "shock_dashboard.html", []))
    # equations.html is a redirect to methodology.html, never a tab of its own
    if (HERE / "methodology.html").exists():
        out.append(("methods", "Methodology", "methodology.html", []))
    elif (HERE / "bottom_up_methods.html").exists():
        out.append(("methods", "Methodology", "bottom_up_methods.html", []))
    us = _us_tabs()
    if us:
        out.append(("us", "US", None, us))
    if cash:
        out.append(("cash", "Cash", None, cash))
    if (HERE / "debt_supply_dashboard.html").exists():
        out.append(("debt", "Debt stock", "debt_supply_dashboard.html", []))
    if (REPO / "issuance" / "issuance_dv01_dashboard.html").exists():
        out.append(("supply", "Market supply", "../issuance/issuance_dv01_dashboard.html", []))
    return out


def _leaves(rows: list[tuple[str, str, str | None, list]]) -> list[tuple[str, str, str, str]]:
    """(key, group, label, src) for every frame."""
    leaves = []
    for group, label, src, kids in rows:
        if src:
            leaves.append((group, group, label, src))
        for key, lab, ksrc in kids:
            leaves.append((key, group, lab, ksrc))
    return leaves


def build() -> Path:
    rows = items()
    leaves = _leaves(rows)
    main = "".join(
        f'<a data-g="{g}" href="#{g}">{label}</a>' for g, label, _, _ in rows
    )
    subs = []
    for group, _, _, kids in rows:
        if not kids:
            continue
        links = "".join(f'<a data-k="{k}" data-g="{group}" href="#{k}">{lab}</a>' for k, lab, _ in kids)
        subs.append(f'<nav class="sub" id="sub-{group}">{links}</nav>')
    frames = "".join(
        f'<iframe id="f-{k}" data-src="{src}" title="{lab}"></iframe>' for k, _, lab, src in leaves
    )
    default = {g: (kids[0][0] if kids else g) for g, _, _, kids in rows}
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Sovereign financing</title><style>{CSS}</style></head>
<body><div class="hub">
<nav class="main">{main}</nav>
{''.join(subs)}
<main>{frames}</main>
</div>
<script>
const DEFAULT = {default!r};
const GROUP = { {k: g for k, g, _, _ in leaves}!r };
function show(raw) {{
  let k = raw && GROUP[raw] ? raw : (raw && DEFAULT[raw] ? DEFAULT[raw] : "overview");
  const g = GROUP[k];
  document.querySelectorAll("nav.main a").forEach(a => a.classList.toggle("on", a.dataset.g === g));
  document.querySelectorAll("nav.sub").forEach(n => n.classList.toggle("on", n.id === "sub-" + g));
  document.querySelectorAll("nav.sub a").forEach(a => a.classList.toggle("on", a.dataset.k === k));
  document.querySelectorAll("main iframe").forEach(f => {{
    const on = f.id === "f-" + k;
    if (on && !f.getAttribute("src")) f.src = f.dataset.src;
    f.classList.toggle("on", on);
  }});
}}
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
show(location.hash.slice(1));
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT} ({', '.join(k for k, _, _, _ in leaves)})")
    return OUT


if __name__ == "__main__":
    build()
