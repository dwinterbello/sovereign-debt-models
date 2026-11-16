"""Sovereign financing dashboard pages, served locally (no internet needed).

    python serve_site.py              # http://127.0.0.1:8765/ (opens the browser)
    python serve_site.py --build      # rebuild every page from the current data first (offline), then serve
    python serve_site.py --port 9000 --no-browser

One tabbed hub (sovereign_cash/financing_dashboard.html): overview, shock scenarios, methodology, US cash / flows /
debt ceiling, cash by country (UK, DE, FR, IT, ES, SE, NO, FYTD) and debt stock. The pages are plain HTML files that
you can also open directly; the local server avoids the browser's file:// restrictions on the embedded pages, and it
serves the Plotly charting library from this folder: the pages ask for it on cdn.plot.ly, which a machine without
internet cannot reach. Each page gets the Plotly version it was built with (site_assets/plotly-<version>.min.js, or
the copy installed with requirements.txt).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HUB = "sovereign_cash/financing_dashboard.html"
ASSETS = ROOT / "site_assets"
CDN = re.compile(rb"https://cdn\.plot\.ly/plotly-([0-9][0-9.]*)\.min\.js")


def _plotly_js(version: str) -> bytes | None:
    """The Plotly bundle a page was built with: shipped in site_assets/, else the one installed with requirements."""
    f = ASSETS / f"plotly-{version}.min.js"
    if f.exists():
        return f.read_bytes()
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs().encode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def build() -> None:
    """US pages first (the hub picks up whichever US pages exist), then every model and page, hub last."""
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""), "MPLBACKEND": "Agg"}
    for cmd in ([sys.executable, "us/fiscal/run_refresh.py"],
                [sys.executable, "-m", "sovereign_cash.run_all"]):
        print("$ python " + " ".join(cmd[1:]), flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode not in (0, 2):        # run_all returns 2 when an optional page failed
            print(f"  exited {r.returncode}; serving the pages that are there")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].split("#")[0]
        if path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/" + HUB)
            self.end_headers()
            return
        if path.startswith("/_plotly/") and path.endswith(".js"):
            js = _plotly_js(path[len("/_plotly/"):-3])
            if js is None:
                self.send_error(404, "plotly not installed")
                return
            self._send(js, "application/javascript")
            return
        f = (ROOT / path.lstrip("/")).resolve()
        if f.suffix == ".html" and f.is_file() and ROOT in f.parents:
            self._send(CDN.sub(rb"/_plotly/\1.js", f.read_bytes()), "text/html; charset=utf-8")
            return
        super().do_GET()

    def log_message(self, fmt, *args):  # quiet: one line per page, not per asset
        if self.path.endswith(".html"):
            sys.stderr.write(f"  {self.address_string()} {self.path}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="rebuild the pages from the current data first")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if not (ROOT / HUB).exists():
        print(f"{HUB} is missing: run  python serve_site.py --build")
        return 1
    handler = functools.partial(_Handler, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Sovereign financing pages: {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
