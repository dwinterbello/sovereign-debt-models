"""Legislated policy on top of our model: costed measures, added as a fixed line to the no-policy-change forecast.

Our component models carry each line's share of GDP or of the price level forward from the last outturn: a
no-policy-change forecast. Measures already in force in that outturn are inside its shares. What the model cannot
know is the legislated change *after* the base year: measures announced at a later event, and the phasing of
earlier ones (a tax cut that grows, a temporary fund that winds down). For each costed measure m with cost c_{m,t}
(effect on the measure, + adds to borrowing):

    policy_t = sum_m ( c_{m,t} - c_{m,0} * g_t )        g_t = X_t / X_0, X the line's driver (GDP or prices)

c_{m,0} is the measure's effect in the model's base year (0 for an event after it), grown as the model grows the
line it sits in, so a measure already in the base is counted only by how far it moves off its share. The line is
fixed in money terms as costed: no shock moves it, and costings are the authorities' own (static, before
behavioural feedback to the macro). Sources and every measure's numbers are in data/policy/{cc}.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "policy"
TOP = 6  # named items in the stacked chart; the rest are summed as other tax / other spending
_RP = {"tax": "receipts", "spend": "primary"}
LINE_IDS = {"uk": _RP, "us": _RP, "de": {"tax": "revenue", "spend": "primary"},
            "fr": {"tax": "recettes", "spend": "primary"}, "it": {"tax": "tax_rev", "spend": "primary"},
            "es": _RP, "se": _RP, "no": _RP, "jp": _RP}
COUNTRIES = tuple(LINE_IDS)


UK_FIRST = 2026  # the UK fiscal year in progress (2026-27) when Budget 2026 was costed


def _uk() -> list[dict]:
    """The UK's decided policy since the October 2024 forecast, £bn on PSNB (+ adds), as the OBR assesses it.

    Taxes: the OBR's total for every event (Budgets 2025 and 2026 as HMT's scorecard measures plus the OBR's
    adjustments to them). Departmental spending: the Government's plans (DEL) as levels, which already carry every
    event's departmental decisions. Annually managed spending: each event's decisions. Indirect effects: the OBR's.
    """
    d = json.loads((DATA / "uk" / "obr_decisions.json").read_text())
    sc = {k: json.loads((DATA / "uk" / f"{k}.json").read_text()) for k in ("budget_2025", "budget_2026")}
    ame = d["ame_keys"]
    out = []
    for ev in d["events"]:
        y0 = ev["first_fy"]
        meta = {"event": ev["event"], "url": ev["url"], "scale_first": y0 < UK_FIRST}
        if ev.get("scorecard"):
            card = sc[ev["scorecard"]]
            ys = [_fy(y) for y in card["years"]]
            tax = [0.0] * len(ys)
            for m in card["measures"]:
                head = "tax" if m["head"].startswith("Tax") else "spend"
                vals = [-v / 1000.0 for v in m["values_m"]]  # table: + Exchequer gain; on PSNB: minus, £bn
                if head == "tax":
                    tax = [a + b for a, b in zip(tax, vals)]
                elif not (ev.get("ame_from_scorecard") and any(k in m["name"] for k in ame)):
                    continue  # departmental: inside the DEL plans
                out.append({"label": m["short"], "name": m["name"], "head": head,
                            "by_year": dict(zip(ys, vals)), **meta})
            adj = [o - t for o, t in zip(ev["obr_tax"], tax)]
            out.append({"label": f"{ev['event'].split(' (')[0]}: OBR adjustments to HMT's tax measures",
                        "name": f"The OBR's tax total for the event less HMT's scorecard tax measures ({ev['source']})",
                        "head": "tax", "by_year": dict(zip(ys, adj)), **meta})
        for lab, (head, vals) in ev["components"].items():
            ys = list(range(y0, y0 + len(vals)))
            out.append({"label": lab, "name": f"{lab} ({ev['source']})", "head": head,
                        "by_year": dict(zip(ys, vals)), **meta})
    dp = d["del_plans"]
    lv = [a + b + c for a, b, c in zip(dp["rdel"], dp["cdel"], dp["unfunded_pensions"])]
    out.append({"label": "Departmental spending plans (Spending Review 2025, Budget 2026)", "name": dp["source"],
                "head": "spend", "event": "Government departmental spending plans (DEL), as of Budget 2026",
                "url": dp["url"], "scale_first": True, "level": True,
                "by_year": dict(zip(range(dp["first_fy"], dp["first_fy"] + len(lv)), lv))})
    return out


def obr_check() -> dict:
    """Each event's tax items add up to the OBR's published tax total (£bn): {event: max abs gap}."""
    d = json.loads((DATA / "uk" / "obr_decisions.json").read_text())
    items = _uk()
    tax_tot = {"ab24": [-1.2, -24.0, -34.8, -39.1, -39.7, -41.2], "ss25": [-0.1, -1.0, -1.4, -1.6, -1.9, -2.2],
               "sf26": [0.1, 0.6, 0.6, 0.6, 0.5, 0.5]}
    gaps = {}
    for ev in d["events"]:
        tot = ev.get("obr_tax") or tax_tot[ev["key"]]
        ys = range(ev["first_fy"], ev["first_fy"] + len(tot))
        mine = [sum(m["by_year"].get(y, 0.0) for m in items if m["event"] == ev["event"] and m["head"] == "tax")
                for y in ys]
        gaps[ev["key"]] = max(abs(a - b) for a, b in zip(mine, tot))
    return gaps


def _fy(label: str) -> int:
    return int(label[:4])


def raw(cc: str) -> list[dict]:
    """Every measure as {label, head ('tax'/'spend'), event, by_year {year: effect on the measure, local bn}}.

    Years are the model's period years (UK fiscal year 2026 = 2026-27, US fiscal year, Germany calendar year).
    """
    out = []
    if cc == "uk":
        return _uk()
    if cc == "us":
        return _us()
    return _generic(cc)


def _us() -> list[dict]:
    """P.L. 119-21 (OBBBA) by title, plus P.L. 119-98 (Secure America) direct outlays."""
    out = []
    d = json.loads((DATA / "us" / "pl119_21.json").read_text())
    ys, t = d["years"], d["deficit_by_title"]
    tax = [-v for v in d["totals"]["revenues"]]  # revenue loss adds to the deficit
    total = d["totals"]["deficit"]
    groups = {
        "Tax provisions (all titles)": ("tax", tax),
        "Defence (Armed Services)": ("spend", t["II. Armed Services"]),
        "Border and immigration enforcement": ("spend", [a + b for a, b in zip(
            t["IX. Homeland Security and Governmental Affairs"], t["X. Judiciary"])]),
        "Medicaid, Medicare and student loans": ("spend", [f + h - x for f, h, x in zip(
            t["VII. Finance"], t["VIII. Health, Education, Labor and Pensions"], tax)]),
        "SNAP and farm programmes": ("spend", t["I. Agriculture"]),
    }
    named = [sum(v[i] for _, v in groups.values()) for i in range(len(ys))]
    groups["Other titles (spectrum, energy, other)"] = ("spend", [a - b for a, b in zip(total, named)])
    for lab, (head, vals) in groups.items():
        out.append({"label": lab, "name": lab, "head": head, "event": d["event"],
                    "by_year": {y: v / 1000.0 for y, v in zip(ys, vals)}})
    saa = json.loads((DATA / "us" / "pl119_98.json").read_text())
    out.append({"label": "Secure America Act (P.L. 119-98): border / ICE direct spending",
                "name": saa["source"], "head": "spend", "event": saa["event"],
                "by_year": {y: v / 1000.0 for y, v in zip(saa["years"], saa["outlays"])},
                "url": saa.get("url")})
    return out


def _generic(cc: str) -> list[dict]:
    """data/policy/{cc}/measures.json: items with values by year as effects on borrowing (+ adds) or, for a
    spending plan ('kind': 'level'), the planned level itself (the netting rule then nets off the base year's)."""
    path = DATA / cc / "measures.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    out = []
    for it in d["items"]:
        vals = {int(y): float(v) for y, v in it.get("values", {}).items() if v is not None}
        if it.get("residual"):  # the published total less the named budget-law items
            tot = {int(y): float(v) for y, v in d["total_ex_interest"].items()}
            named = [x for x in d["items"] if not x.get("residual") and not x.get("extra")]
            vals = {y: t - sum(float(x["values"].get(str(y), 0.0)) for x in named) for y, t in tot.items()}
        if it.get("fiscal_april"):  # fiscal year April-March into calendar years
            fy = dict(vals)
            last = max(fy)
            vals = {y: 0.25 * fy.get(y - 1, fy[y]) + 0.75 * fy[y] for y in fy if y - 1 in fy}
            vals[last + 1] = fy[last]
        out.append({"label": it["label"], "name": it.get("source", it["label"]), "head": it["head"],
                    "event": it.get("event", d.get("event", "")), "by_year": vals,
                    "url": it.get("url", d.get("url")), "grow": it.get("grow", True),
                    "level": it.get("kind") == "level"})
    return out


def sources(cc: str) -> list[dict]:
    if cc == "uk":
        d = json.loads((DATA / "uk" / "obr_decisions.json").read_text())
        return [{"title": ev["source"], "url": ev["url"]} for ev in d["events"]] + [
            {"title": "OBR EFO October 2026, Table A.7 (departmental spending plans)", "url": d["del_plans"]["url"]}]
    if cc == "us":
        d21 = json.loads((DATA / "us" / "pl119_21.json").read_text())
        d98 = json.loads((DATA / "us" / "pl119_98.json").read_text())
        return [
            {"title": "CBO, Estimated Budgetary Effects of Public Law 119-21 (July 2025)", "url": d21["url"]},
            {"title": "CBO / Congress.gov: Public Law 119-98 Secure America Act", "url": d98["url"]},
        ]
    path = DATA / cc / "measures.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    seen, out = set(), []
    for it in d["items"]:
        url = it.get("url", d.get("url"))
        if url and url not in seen:
            seen.add(url)
            out.append({"title": (it.get("source") or d.get("source", "")).split(":")[0], "url": url})
    return out



def build(cc: str, years: list[int], base_year: int, first_share: float = 1.0,
          growth: dict[str, pd.Series] | None = None) -> dict | None:
    """The policy line for the pack, local bn on the measure (+ adds to borrowing) for each of ``years``.

    ``base_year``: the model's last outturn period. ``growth``: {'tax': X, 'spend': X} driver levels by period
    year (the line each head sits in). ``first_share``: the share of the first period still to forecast (the UK's
    year in progress); a base-year measure's increment that year is scaled by it, a later event's is not (its
    measures start after the outturn months).
    """
    if cc not in LINE_IDS:
        return None
    ms = raw(cc)
    if not ms:
        return None
    items = []
    for m in ms:
        gk = "spend" if m["head"] == "indirect" else m["head"]
        x = growth[gk] if growth and m.get("grow", True) else None
        c0 = m["by_year"].get(base_year, 0.0)
        vals, raws = [], []
        for i, y in enumerate(years):
            g = float(x[y] / x[base_year]) if x is not None else 1.0
            r = m["by_year"].get(y, m["by_year"][max(m["by_year"])] if y > max(m["by_year"]) else 0.0)
            v = r - c0 * g
            if i == 0 and m.get("scale_first", c0 != 0.0):
                v *= first_share
            vals.append(v)
            raws.append(v if m.get("level") else r)  # a planned level is shown as its change on the base
        items.append({**m, "values": vals, "raw": raws})
    # chart groups: same label summed across events, the largest TOP by peak size named, the rest by head
    by = {}
    for m in items:
        k = (m["label"], m["head"])
        if k not in by:
            by[k] = {"label": m["label"], "head": m["head"], "values": [0.0] * len(years), "n": 0,
                     "events": [], "detail": m["name"], "_peak": 0.0}
        peak = max(abs(v) for v in m["raw"])
        if peak > by[k]["_peak"]:  # the group's description is its largest measure's
            by[k]["detail"], by[k]["_peak"] = m["name"], peak
        by[k]["values"] = [a + b for a, b in zip(by[k]["values"], m["raw"])]
        by[k]["n"] += 1
        if m["event"] not in by[k]["events"]:
            by[k]["events"].append(m["event"])
    ranked = sorted(by.values(), key=lambda g: -max(abs(v) for v in g["values"]))
    named, rest = ranked[:TOP], ranked[TOP:]
    groups = [{"id": f"p{i}", **{k: v for k, v in g.items() if k != "_peak"}} for i, g in enumerate(named)]
    for head, lab in (("tax", "Other tax measures"), ("spend", "Other spending measures"),
                      ("indirect", "Other indirect effects")):
        rs = [g for g in rest if g["head"] == head]
        if rs:
            groups.append({"id": f"other_{head}", "label": lab, "head": head, "other": True,
                           "values": [sum(g["values"][i] for g in rs) for i in range(len(years))],
                           "n": sum(g["n"] for g in rs), "events": sorted({e for g in rs for e in g["events"]})})
    total = [sum(m["values"][i] for m in items) for i in range(len(years))]
    # what the authority's costings say, as published, less what the model already carries from the base year
    netting = [total[i] - sum(m["raw"][i] for m in items) for i in range(len(years))]
    if any(abs(v) > 1e-9 for v in netting):
        groups.append({"id": "netting", "label": "Already in our base year (netted off)", "head": "netting",
                       "netting": True, "values": netting, "n": 0, "events": [],
                       "detail": "The part of these decisions already in the base year's outturn, which the model "
                                 "carries forward with its shares; it is not added twice"})
    ids = LINE_IDS[cc]
    # on each line in its own units: receipts up when a tax measure lowers borrowing; spending up with it
    by_line = {ids["tax"]: [-sum(m["values"][i] for m in items if m["head"] == "tax") for i in range(len(years))],
               ids["spend"]: [sum(m["values"][i] for m in items if m["head"] in ("spend", "indirect"))
                              for i in range(len(years))]}
    return {"total": total, "groups": groups, "by_line": by_line, "n_measures": len(items),
            "events": sorted({m["event"] for m in items}), "base_period": base_year, "sources": sources(cc),
            "published": [sum(m["raw"][i] for m in items) for i in range(len(years))],
            "measures": [{"label": m["label"], "head": m["head"], "event": m["event"],
                          "values": [round(v, 4) for v in m["values"]], "raw": [round(v, 4) for v in m["raw"]]}
                         for m in items]}
