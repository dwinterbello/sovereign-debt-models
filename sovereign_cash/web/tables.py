"""HTML tables for the country boards (presentation only; numbers come from the models)."""
from __future__ import annotations

import html

import numpy as np


def unit_shock_table(cc: str, payload: dict, *, ccy: str = "€") -> str:
    """UK-style selectable unit-shock tables for a ``unit_shocks.run`` payload."""
    if not payload.get("ok"):
        return ""
    years = [int(y) for y in payload["years"]]
    head = "".join(f"<th>{y}</th>" for y in years)
    opts, tables = [], []
    base = payload["base"]
    for i, sh in enumerate(payload["shocks"]):
        delta = sh["delta"]
        cells = "".join(f"<td>{_fmt(delta.get(str(y)), ccy)}</td>" for y in years)
        body = [f"<tr><td style='font-weight:700'>Δ cash requirement</td>{cells}</tr>"]
        base_cells = "".join(
            f"<td>{_fmt(base.get(str(y)), ccy, plus=False)}</td>" for y in years
        )
        body.append(f"<tr><td>Baseline (model)</td>{base_cells}</tr>")
        opts.append(f"<option value='{sh['key']}'>{html.escape(sh['title'])}</option>")
        hidden = "" if i == 0 else " hidden"
        tables.append(
            f"<table class='t' data-sx='{cc}-{sh['key']}'{hidden}>"
            f"<tr><th></th>{head}</tr>{''.join(body)}</table>"
        )
    drv = payload["drivers"]
    src = (f"{html.escape(drv['ngdp_source'])}; {html.escape(drv['cpi_source'])}; "
           f"{html.escape(drv.get('rates_source', 'rates'))} · origin {payload['origin']}")
    return (
        f"<div class='sub'>{src}</div>"
        f"<select onchange=\"document.querySelectorAll('[data-sx^=\\'{cc}-\\']')"
        f".forEach(t=>t.hidden=t.dataset.sx!=='{cc}-'+this.value)\">"
        f"{''.join(opts)}</select>{''.join(tables)}"
        f"<div class='sub'>{html.escape(payload['note'])}</div>"
    )


def _fmt(v, ccy: str, plus: bool = True) -> str:
    from sovereign_cash.contract import fmt_bn

    if v is None or not np.isfinite(v):
        return "—"
    return fmt_bn(float(v), ccy, plus=plus)
