"""One US board: tabs over cash, Table II flows, ceiling, funding liquidity and the report.

Each tab is the standalone page in an iframe (loaded on first open), so every page
still builds and opens on its own. Writes ``us_fiscal_dashboard.html``.

  PYTHONPATH=. .venv/bin/python us/fiscal/build_us_hub.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from build_dashboard import PAGE_CSS  # noqa: E402

OUT = HERE / "us_fiscal_dashboard.html"
TABS = (
    ("cash", "Cash", "tga_deficit_dashboard.html"),
    ("flows", "Flows", "tga_components_dashboard.html"),
    ("ceiling", "Ceiling", "tga_ceiling_dashboard.html"),
    ("liquidity", "Liquidity", "liquidity_dashboard.html"),
    ("report", "Report", "report/US_Fiscal_Funding_Report_selfcontained.html"),
)

HUB_CSS = """
html, body { height: 100%; }
body { margin: 0; display: flex; flex-direction: column; }
nav { display: flex; gap: 2px; padding: 6px 12px 0; border-bottom: 1px solid var(--line); background: var(--panel-hi); }
nav a { font: 600 11px/1 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  padding: 8px 12px; border: 1px solid transparent; border-bottom: none; text-decoration: none; cursor: pointer; }
nav a.on { color: var(--text); border-color: var(--line); background: var(--bg, #fff); margin-bottom: -1px; }
nav .sp { flex: 1; }
nav a.pdf { text-transform: none; }
main { flex: 1; position: relative; }
main iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; display: none; }
main iframe.on { display: block; }
"""


def build() -> Path:
    present = [(k, label, src) for k, label, src in TABS if (HERE / src).exists()]
    links = "".join(f'<a data-t="{k}" href="#{k}">{label}</a>' for k, label, _ in present)
    frames = "".join(f'<iframe id="f-{k}" data-src="{src}" title="{label}"></iframe>' for k, label, src in present)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>US fiscal & funding</title>
  <style>{PAGE_CSS}{HUB_CSS}</style>
</head>
<body>
  <nav>{links}<span class="sp"></span><a class="pdf" href="report/US_Fiscal_Funding_Report.pdf" target="_blank">Report PDF</a></nav>
  <main>{frames}</main>
<script>
const keys = {[k for k, _, _ in present]!r};
function show(k) {{
  if (!keys.includes(k)) k = keys[0];
  document.querySelectorAll('nav a[data-t]').forEach(a => a.classList.toggle('on', a.dataset.t === k));
  document.querySelectorAll('main iframe').forEach(f => {{
    const on = f.id === 'f-' + k;
    if (on && !f.src) f.src = f.dataset.src;
    f.classList.toggle('on', on);
  }});
}}
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1));
</script>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()
