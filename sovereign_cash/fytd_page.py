"""Cross-country FYTD cash panel for the Sovereign financing board.

Same ``cash_requirement_bn`` frames the country cash dashboards use; this page is
the Ex1c multi-country %GDP view (printed YTD, model remainder, plan / next levels).
Built into ``fytd_cash_dashboard.html`` and embedded on the overview.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.fytd_page
"""
from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_HTML = HERE / "fytd_cash_dashboard.html"
OUT_PNG = HERE / "derived" / "fytd_cash.png"


def build() -> Path:
    from sovereign_cash.debt.note.fytd_deficit import build_panel, render, summary_table_html

    HERE.joinpath("derived").mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    png, caption, _ = render(panel, OUT_PNG)
    table = summary_table_html(panel)
    # Exhibit 1b (UK PSNB / ES B.9 accrued meters) stays in the debt note only —
    # FR/IT/DE have no monthly GG series, so the two-panel strip on this board was noise.

    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    built = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>FYTD cash — sovereign financing</title>
<style>
body {{ font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1d2433; background: #fff; }}
.wrap {{ padding: 16px 22px 36px; max-width: 1280px; }}
h1 {{ font: 600 15px/1.3 ui-monospace, Menlo, monospace; letter-spacing: .06em; text-transform: uppercase; margin: 0 0 4px; }}
h2 {{ font: 600 12px/1.3 ui-monospace, Menlo, monospace; letter-spacing: .06em; text-transform: uppercase; margin: 22px 0 6px;
     color: #3b4356; border-bottom: 1px solid #d8dde6; padding-bottom: 4px; }}
p.sub {{ color: #4a5263; max-width: 1100px; margin: 4px 0 10px; }}
figure {{ margin: 8px 0 18px; }}
figure img {{ width: 100%; height: auto; display: block; border: 1px solid #e6e9ef; }}
table {{ border-collapse: collapse; margin: 6px 0 12px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 9px; border-bottom: 1px solid #e6e9ef; text-align: right; }}
th {{ font-weight: 600; background: #f5f7fa; font-size: 12px; }}
td.l, th.l {{ text-align: left; }}
.yr {{ color: #6b7385; font-size: 11px; }}
</style></head>
<body><div class="wrap">
<h1>FYTD cash requirement</h1>
<p class="sub">Built {built}. Same monthly cash frames as the country Cash tabs.
<b>current / next</b> = this- and next-year authority plan levels; <b>model</b> (dashed)
= shipped rest-of-FY path. Cash requirement only — financing identity charts stop at
the coupon programme horizon.</p>
<figure><img src="data:image/png;base64,{b64}" alt="FYTD cash % GDP"/></figure>
<p class="sub">{caption}</p>
<h2>Live vs plan</h2>
{table}
</div></body></html>"""
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT_HTML} + {png}")
    return OUT_HTML


def png_data_uri() -> str | None:
    """Overview embed helper — rebuilds if missing."""
    if not OUT_PNG.exists():
        build()
    if not OUT_PNG.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(OUT_PNG.read_bytes()).decode("ascii")


if __name__ == "__main__":
    build()
