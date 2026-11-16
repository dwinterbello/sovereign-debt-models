"""Write the shock dashboard and the Methodology page from the scenario pack.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.web.pages

Each page is self-contained (pack and engine inlined) so the site publishes it
like any other board. The engine source is the same file the parity tests run.
``equations.html`` is kept only as a redirect to ``methodology.html``.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from sovereign_cash.scenarios.pack import PACK_PATH

WEB = Path(__file__).resolve().parent
SC = WEB.parent
OUT_DASHBOARD = SC / "shock_dashboard.html"
OUT_METHODOLOGY = SC / "methodology.html"
OUT_METHODOLOGY_ALIAS = SC / "bottom_up_methods.html"
OUT_EQUATIONS = SC / "equations.html"  # redirect stub only
OUT_VENDOR = SC / "vendor"
VENDOR_SRC = SC / "equations" / "vendor"


def engine_inline() -> str:
    """engine.js with ``export`` dropped so it runs inside the page's own module script."""
    src = (WEB / "engine.js").read_text(encoding="utf-8")
    return re.sub(r"^export\s+", "", src, flags=re.M)


def _pack_json(pack: dict) -> str:
    # safe inside <script type="application/json">: no "</" sequences
    return json.dumps(pack, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def _embedded_vintage(html: str) -> tuple[str, str]:
    m = re.search(r'<script type="application/json" id="pack">(.*?)</script>', html, re.S)
    if not m:
        raise RuntimeError("pack JSON missing from page")
    p = json.loads(m.group(1).replace("<\\/", "</"))
    return p["built_at"], p["model_vintage"]


def render(template: str, pack: dict, **extra: str) -> str:
    out = template.replace("__PACK__", _pack_json(pack)).replace("__ENGINE__", engine_inline())
    for k, v in extra.items():
        out = out.replace(f"__{k}__", v)
    if "__PACK__" in out or "__ENGINE__" in out:
        raise RuntimeError("template placeholders left unfilled")
    return out


def _ensure_vendor_assets() -> None:
    """Fetch KaTeX / Plotly into equations/vendor when the JS/CSS files are missing."""
    import urllib.request

    VENDOR_SRC.mkdir(parents=True, exist_ok=True)
    assets = {
        "katex.min.js": "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js",
        "katex.min.css": "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.css",
        "plotly.min.js": "https://cdn.plot.ly/plotly-3.7.0.min.js",
    }
    for name, url in assets.items():
        dest = VENDOR_SRC / name
        if dest.is_file() and dest.stat().st_size > 1000:
            continue
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())


def _sync_vendor() -> Path | None:
    """Copy KaTeX/Plotly next to methodology.html so the page does not depend on a CDN."""
    try:
        _ensure_vendor_assets()
    except Exception as exc:  # noqa: BLE001 — CDN optional; page has a JS fallback
        print(f"vendor fetch skipped: {exc}")
    if not VENDOR_SRC.is_dir():
        return None
    if not (VENDOR_SRC / "katex.min.js").is_file():
        return None
    if OUT_VENDOR.exists():
        shutil.rmtree(OUT_VENDOR)
    shutil.copytree(VENDOR_SRC, OUT_VENDOR, ignore=shutil.ignore_patterns("*.ttf"))
    return OUT_VENDOR


def build(pack: dict | None = None) -> list[Path]:
    from sovereign_cash.equations import page as eq_page
    from sovereign_cash.methodology import page as meth_page

    pack = pack or json.loads(PACK_PATH.read_text())
    tpl = (WEB / "shock_dashboard.template.html").read_text(encoding="utf-8")
    OUT_DASHBOARD.write_text(render(tpl, pack), encoding="utf-8")
    OUT_METHODOLOGY.write_text(meth_page.render(pack), encoding="utf-8")
    OUT_METHODOLOGY_ALIAS.write_text(eq_page.redirect_html("methodology.html"), encoding="utf-8")
    OUT_EQUATIONS.write_text(eq_page.redirect_html("methodology.html"), encoding="utf-8")
    vendor = _sync_vendor()
    # Gate: both boards must certify the same pack (CIO ship-stopper Nov-2026).
    want = (pack["built_at"], pack["model_vintage"])
    for path in (OUT_DASHBOARD, OUT_METHODOLOGY):
        got = _embedded_vintage(path.read_text(encoding="utf-8"))
        if got != want:
            raise RuntimeError(f"{path.name} pack {got} != source {want}")
    out = [OUT_DASHBOARD, OUT_METHODOLOGY, OUT_METHODOLOGY_ALIAS, OUT_EQUATIONS]
    if vendor:
        out.append(vendor)
    return out


if __name__ == "__main__":
    for p in build():
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"wrote {p}/ ({n} files)")
        else:
            print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
