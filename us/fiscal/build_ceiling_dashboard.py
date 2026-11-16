"""US debt ceiling / non-marketables page: STL headroom, GAS / EM, contact and X-date.

Also carries the TGA / buybacks cash-tools note that used to sit on the merged
cash page. Writes ``tga_ceiling_dashboard.html``.

  PYTHONPATH=. .venv/bin/python us/fiscal/build_ceiling_dashboard.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_dashboard import (  # noqa: E402
    PAGE_CSS, _fig_buybacks, _fig_ceiling_headroom, _fig_ceiling_tga, _fig_gas,
    _html, _kpi, _load, _load_buybacks_m, _load_ceiling,
)

OUT = HERE / "tga_ceiling_dashboard.html"


def build() -> Path:
    panel = _load()
    ceil_hist, ceil_fut, ceil_sum = _load_ceiling()
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if ceil_hist is None or ceil_fut is None or ceil_sum is None:
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<title>US debt ceiling</title><style>{PAGE_CSS}</style></head><body>
<header><h1>US debt ceiling — STL / GAS / EM</h1>
<div class="sub">{built} · Run <code>debt_ceiling</code> to refresh.</div></header>
</body></html>"""
        OUT.write_text(html, encoding="utf-8")
        print(f"Wrote {OUT} (empty)")
        return OUT

    contact = ceil_sum.get("contact_date") or "—"
    x_date = ceil_sum.get("x_date") or "—"
    days_c = ceil_sum.get("days_to_contact")
    kpis = "".join([
        _kpi("STL headroom", f"${ceil_sum['headroom_bn']:,.0f}bn",
             f"STL ${ceil_sum['stl_bn']:,.0f}bn / limit ${ceil_sum['limit_bn']:,.0f}bn"),
        _kpi("EM capacity", f"${ceil_sum['em_capacity_bn']:,.0f}bn",
             str(ceil_sum.get("em_source", "intragov at-limit drop"))),
        _kpi("Contact date", contact,
             f"in {days_c}d" if days_c is not None else "STL path vs limit"),
        _kpi("X-date", x_date,
             f"TGA floor ${ceil_sum.get('tga_floor_bn', 50):.0f}bn · post-contact runway"),
    ])
    figs = [
        _fig_ceiling_headroom(ceil_hist, ceil_fut, ceil_sum),
        _fig_ceiling_tga(panel, ceil_fut, ceil_sum),
    ]
    gas = _fig_gas(ceil_hist)
    if gas is not None:
        figs.append(gas)
    bb = _load_buybacks_m()
    if bb is not None and not bb.empty:
        figs.append(_fig_buybacks(bb))
    cards = "".join(f"<div class='card'>{_html(f, first=(i == 0))}</div>" for i, f in enumerate(figs))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>US debt ceiling — STL / GAS / EM</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <header>
    <h1>US debt ceiling — STL, GAS / EM, contact and X-date</h1>
    <div class="sub">{built} · Debt subject to the limit from DTS Table IIIC (public + intragov − not-subject).
      Forward STL = last print + fiscal-path net marketable issuance + trailing 60d intragov pace.
      Contact = first day projected headroom ≤ 0. After contact, net issuance stops and the TGA runs on the
      rollover path; extraordinary-measures capacity is the peak-to-trough intragovernmental drop in the last
      at-limit episode (GAS / CSRDF / G-Fund etc.), floored at ${ceil_sum.get('em_floor_bn', 250):.0f}bn.
      X-date = runway (TGA at contact + EM) spent down to the ${ceil_sum.get('tga_floor_bn', 50):.0f}bn TGA floor.
      Seasonal TGA on the path chart is the prior-two-FY median same-day shape anchored at the TBAC target.</div>
  </header>
  <div class="wrap">
    <div class="kpi-row">{kpis}</div>
    <div class="grid">{cards}</div>
    <h2 class="section">TGA stockpile and buybacks</h2>
    <p class="sub">Yellen cash policy was about a week of outflows, ~$550–600bn. Bessent Q3 2026 refunding
      (TBAC) assumes <strong>$950bn end-Sep</strong> and <strong>$850bn end-Dec</strong>. 12-month ΔTGA is
      <em>up</em>, so the stockpile was built, not drained by buybacks. The May 2024 program had two books:
      <em>cash management</em> (seasonal, spend TGA / cut bills around tax dates) and
      <em>liquidity support</em> (weekly OFTR; amounts treated as a borrowing need — issuance-financed,
      not a TGA draw). Using the ~$1tn TGA to fund larger long-end OFTR ops is a 2026 overlay, not the
      original design. Rebuilding the TGA after a draw requires more issuance later.</p>
  </div>
</body>
</html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()
