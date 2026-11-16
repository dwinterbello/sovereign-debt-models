"""Sovereign cash board: overview + one tab per country page that exists.

The US is an overview row only (from the supply bridge); its pages live on the
us_fiscal board, and a `../` iframe would not survive portal bundling.

Overview rows come from each `<cc>_cash_monthly.parquet` (next twelve forecast months
summed) and `<cc>_cash_summary.json` (buffer definition, identity residual, scoring
note). Country tabs iframe the standalone `<cc>_cash_dashboard.html`, loaded on first
open. Writes ``sovereign_cash/sovereign_cash_dashboard.html``.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.build_hub
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
OUT = HERE / "sovereign_cash_dashboard.html"
OVERVIEW = HERE / "sovereign_cash_overview.html"
COUNTRIES = (("us", "US", None), ("uk", "UK", "£"), ("de", "Germany", "€"), ("fr", "France", "€"),
             ("it", "Italy", "€"), ("se", "Sweden", "SEK "), ("no", "Norway", "NOK "))
FLOWS = ("cash_requirement_bn", "net_bonds_bn", "net_bills_bn", "other_fin_bn", "buffer_change_bn")

CSS = """
body { font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1d2433; }
html, body { height: 100%; }
.hub { display: flex; flex-direction: column; height: 100%; }
nav { display: flex; gap: 2px; padding: 6px 12px 0; border-bottom: 1px solid #d8dde6; background: #f5f7fa; }
nav a { font: 600 11px/1 ui-monospace, Menlo, monospace; letter-spacing: .06em; text-transform: uppercase; color: #6b7385;
  padding: 8px 12px; border: 1px solid transparent; border-bottom: none; text-decoration: none; cursor: pointer; }
nav a.on { color: #1d2433; border-color: #d8dde6; background: #fff; margin-bottom: -1px; }
main { flex: 1; position: relative; }
main iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; display: none; }
main iframe.on { display: block; }
.wrap { padding: 18px 24px; max-width: 1300px; }
table { border-collapse: collapse; margin: 10px 0 18px; }
th, td { padding: 5px 10px; border-bottom: 1px solid #e6e9ef; text-align: right; vertical-align: top; }
th { font-weight: 600; background: #f5f7fa; }
td.l, th.l { text-align: left; max-width: 520px; }
p.sub { color: #4a5263; max-width: 1100px; }
"""


def _forward(cc: str) -> dict | None:
    p = DERIVED / f"{cc}_cash_monthly.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    f = d[d["is_forecast"].astype(bool)].sort_values("month").head(12)
    if f.empty:
        return None
    out = {c: float(f[c].sum()) for c in FLOWS if c in f}
    out["window"] = f"{pd.Timestamp(f['month'].iloc[0]):%b-%y}–{pd.Timestamp(f['month'].iloc[-1]):%b-%y}"
    out["asof"] = f"{pd.Timestamp(d.loc[~d['is_forecast'].astype(bool), 'month'].max()):%b-%y}"
    for c in ("bill_stock_implied_bn", "bill_stock_bn"):
        if c in f and f[c].notna().any():
            out["bill_stock_end"] = float(f[c].dropna().iloc[-1])
            break
    return out


def _summary(cc: str) -> dict:
    p = DERIVED / f"{cc}_cash_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def tabs() -> list[tuple[str, str, str]]:
    out = [("overview", "Overview", OVERVIEW.name)]
    if (HERE / "methodology.html").exists():
        out.append(("methods", "Methodology", "methodology.html"))
    elif (HERE / "bottom_up_methods.html").exists():
        out.append(("methods", "Methodology", "bottom_up_methods.html"))
    if (HERE / "debt_supply_dashboard.html").exists():
        out.append(("debt", "Debt & supply", "debt_supply_dashboard.html"))
    for cc, name, _ in COUNTRIES[1:]:
        if (HERE / f"{cc}_cash_dashboard.html").exists():
            out.append((cc, name, f"{cc}_cash_dashboard.html"))
    if (HERE / "es_cash_dashboard.html").exists() and not any(t[0] == "es" for t in out):
        out.append(("es", "Spain", "es_cash_dashboard.html"))
    return out


def overview(refresh: bool = False) -> Path:
    from sovereign_cash.investor_page import render

    OVERVIEW.write_text(render(refresh=refresh), encoding="utf-8")
    return OVERVIEW


def build(refresh: bool = False) -> Path:
    overview(refresh)
    t = tabs()
    links = "".join(f'<a data-t="{k}" href="#{k}">{label}</a>' for k, label, _ in t)
    frames = "".join(f'<iframe id="f-{k}" data-src="{src}" title="{label}"></iframe>' for k, label, src in t)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Sovereign cash</title><style>{CSS}</style></head>
<body><div class="hub"><nav>{links}</nav><main>{frames}</main></div>
<script>
const keys = {[k for k, _, _ in t]!r};
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
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT} ({', '.join(k for k, _, _ in t)})")
    return OUT


if __name__ == "__main__":
    build()
