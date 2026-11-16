"""The one primary-balance engine: every annual exhibit and the debt board read the primary deficit from here.

    primary_p = q_p P_p + Pol^spend_p − r_p Y_p − Pol^tax_p          (central / published measure, local bn)
    primary_gg_p = primary_p / GDP_p + ω                                (general-government basis, % GDP)

r, q: receipts share of nominal GDP and primary spending share of the price level, carried forward from the last
complete outturn by the rule the walk-forward test picked (``own_annual.METHOD``; France, Germany and Italy
annualise the same cash-account lines their monthly SARIMAX models forecast). Pol: the legislated measures in
``policy/overlay`` (static costings, fixed in money terms).

ω (forecast only): general-government less measure wedge at the last common outturn year, held forward.
Outturn ``primary_gg`` is AMECO GG primary (Commission) when available — never the held-ω rewrite of
history (that put FR/DE/UK/IT 2019 ~2pp off AMECO and contaminated every forecast QA gap).

US: ``us.fiscal`` DTS bucket path with every line put on a driver (``us_dts_adjust``: FY2026 one-offs out, CBO
growth for Social Security and Medicare, nominal GDP elsewhere), rebased to FRED FY levels at the origin so the
AMECO wedge stays valid (``dts_fy``); it runs to FY2030 (``US_DTS_TO_YEND``) rather than handing over to WEO.

Years at or before the origin are outturns. The share / cash / DTS rules are tested one to three years out.
Beyond T+3 the GG primary deficit is the IMF WEO primary (``weo_pct``); policy overlays are dropped there
(already in WEO). ``held_pct`` remains only as a fallback when a WEO cell is missing. Interest stays on
the ISIN / forward book — WEO interest is never an input.

Japan inside the tested window still forces primary spend onto GDP (``persist_gdp``): receipts-on-Y /
spend-on-P persistence manufactured a surplus by T+3 that ``held_pct`` used to lock — an unscored
artefact. That force stops the handoff year from being poisoned; the medium-term path is WEO for
every panel country, not a Japan special case.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.scenarios.primary      # writes derived/primary_path.json
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels import own_annual as OA

log = logging.getLogger(__name__)
SC = Path(__file__).resolve().parents[1]
DEBT = SC / "data" / "debt"
PATH = SC / "derived" / "primary_path.json"
COUNTRIES = ("us", "uk", "de", "fr", "it", "es", "se", "no", "jp")
# measures already on the general-government basis carry no wedge
GG_MEASURE = ("es", "se", "no", "jp")
Y_END = 2030  # the debt board's horizon; the authority macro paths end here
TESTED_H = 3  # horizon of the walk-forward share / cash test; beyond it GG primary = WEO
# Beyond TESTED_H: GG primary deficit = −WEO primary_pct (no policy overlay — WEO already has it).
# Full panel — short-horizon cash/share/DTS stays ours; medium-term primary is the authority.
WEO_PRIMARY_HOLD = frozenset(COUNTRIES)
# Inside the tested window: force spend on GDP so Y−P drift cannot invent a surplus before the handoff.
PERSIST_GDP_PRIMARY = frozenset({"jp", "es"})  # Spain: same failure as Japan (no budget since 2023, so no policy
# overlay to offset it): spend-on-P persistence took the primary from 0.0% (2025) to a 2.6% surplus by 2028, then a
# 2.5pp cliff at the WEO handoff; Commission and IMF have Spain near balance for 2026-27


# Current year: anchored to an official general-government estimate made late in the year, so headline gaps do not
# rest on our own nowcast. The anchor's difference from the engine fades linearly to zero at the WEO handoff, keeping
# the engine's year-on-year dynamics. Our cash tracker (fytd_deficit) stays the in-year risk signal in the note.
#   de, it, es: European Commission autumn 2026 (AMECO, 15 Nov 2026): balance + gross interest (D.41), as the engine
#   fr: the government's revised 5.4% (PLF 2027), wider than the Commission's 5.1% and confirmed by our cash tracker;
#       interest from AMECO
#   uk: OBR October 2026 Budget, primary deficit FY2026-27 (net interest: a conservative primary)
#   jp: IMF WEO October 2026, general-government primary (the Cabinet Office's excludes social security funds)
#   us: FY2026 is outturn on the DTS path; no anchor
ANCHOR_YEAR = 2026
FR_GOV_DEFICIT_2026 = 5.4


def anchor_primary_pct(cc: str, y: int = ANCHOR_YEAR) -> tuple[float, str] | None:
    """Official GG primary deficit (% GDP, + = deficit) for the current year, and its source; None = no anchor."""
    if cc in ("de", "it", "es", "fr"):
        a = _ameco(cc)
        if y not in a.index:
            return None
        intr = float(a.at[y, "interest_gross_pct"])
        if cc == "fr":
            return FR_GOV_DEFICIT_2026 - intr, "government 5.4% (PLF 2027) less EC interest"
        return -float(a.at[y, "balance_pct"]) - intr, "European Commission autumn 2026"
    if cc == "uk":
        o = pd.read_parquet(DEBT / "obr.parquet")
        r = o[(o["var"] == "primary_deficit_pct") & (o["year"] == y)]
        return (float(r["value"].iloc[-1]), "OBR Oct 2026 Budget") if len(r) else None
    if cc == "jp":
        w = weo_primary_deficit_pct(cc)
        return (float(w[y]), "IMF WEO Oct 2026") if y in w.index and np.isfinite(w[y]) else None
    return None


# The 2025 act's tax measures on top of the DTS path count them twice (the DTS levels already carry them).
US_DTS_TAX_OVERLAY = False
# The US line model (DTS buckets, every line on a driver) runs through FY2030, the note's horizon. Handing FY2030 to the
# IMF WEO primary instead put a 1pp+ step between FY2029 and FY2030 that no driver explains.
US_DTS_TO_YEND = True
# Last AMECO year treated as outturn. Fixed rather than taken from the clock, so a rebuild in January does not turn
# AMECO's forecast for the year just ended into outturn. Move to 2026 when AMECO publishes the 2026 outturn (spring 2027).
AMECO_OUTTURN_LAST = 2025


@lru_cache(maxsize=16)
def _ameco(cc: str) -> pd.DataFrame:
    a = pd.read_parquet(DEBT / "ameco.parquet")
    return a[a["country"] == cc].pivot_table(index="year", columns="var", values="value").astype(float)


def gg_primary_pct(cc: str) -> pd.Series:
    """AMECO general-government primary deficit, % GDP, outturn years (+ = deficit)."""
    a = _ameco(cc)
    last = AMECO_OUTTURN_LAST
    s = (-a["balance_pct"] - a["interest_gross_pct"]).dropna()
    return s.loc[:last]


@lru_cache(maxsize=16)
def weo_primary_deficit_pct(cc: str) -> pd.Series:
    """IMF WEO primary deficit, % GDP (+ = deficit). WEO publishes primary_pct as a surplus."""
    w = pd.read_parquet(DEBT / "weo.parquet")
    s = w[(w["country"] == cc) & (w["var"] == "primary_pct")].set_index("year")["value"].astype(float)
    return (-s).dropna()


def wedge(cc: str, T: int) -> tuple[float, int | None]:
    """ω: general-government primary deficit less the measure's, % GDP, at the last common outturn year ≤ T.

    Fiscal-year measures (UK April–March, US October–September) are compared with the calendar year that carries
    nine of their twelve months, i.e. the same label."""
    if cc in GG_MEASURE:
        return 0.0, None
    h = OA.history(cc)
    Y, _ = OA.levels(cc, list(h.index))
    own = 100.0 * (h["primary"] - h["receipts"]) / Y.reindex(h.index)
    gg = gg_primary_pct(cc)
    common = [y for y in own.index if y <= T and y in gg.index and np.isfinite(own[y])]
    if not common:
        return float("nan"), None
    y = max(common)
    return float(gg[y] - own[y]), int(y)


def _policy(cc: str, years: list[int], T: int) -> dict | None:
    from sovereign_cash.policy import overlay

    if cc not in overlay.COUNTRIES:
        return None
    span = list(range(T, max(years) + 1))
    Y, P = OA.levels(cc, span)
    spend = Y if OA.method(cc, "primary") == "persist_gdp" else P
    return overlay.build(cc, years, T, 1.0, growth={"tax": Y, "spend": spend})


def path(cc: str, years=None) -> pd.DataFrame:
    """Annual table for ``years`` (default: 2015 → Y_END) by period year, local bn unless ``_pct``.

    Columns: receipts, primary_spend, interest (own_annual's, informational), policy_tax (+ raises receipts),
    policy_spend (+ raises spending), primary_deficit, gdp, primary_deficit_pct, wedge_pct, primary_gg_pct,
    is_forecast.

    Outturn rows come from ``own_annual.history``; forecast rows from ``own_annual.forecast`` (future only).
    ``primary_gg_pct`` on outturn years is AMECO GG primary when present; on forecast years it is the measure
    plus the held origin wedge.
    """
    OA.load_methods()
    h = OA.history(cc)
    T = int(h.index.max())
    if years is None:
        years = list(range(max(2015, int(h.index.min())), Y_END + 1))
    years = [int(y) for y in years]
    hist_years = [y for y in years if y <= T]
    fut = [y for y in years if y > T]
    Y, _ = OA.levels(cc, years)
    out = pd.DataFrame(index=pd.Index(years, name="year"), dtype=float)
    out["gdp"] = pd.to_numeric(Y.reindex(years), errors="coerce")
    out["policy_tax"] = 0.0
    out["policy_spend"] = 0.0
    out["rule"] = ["outturn" if y <= T else "share" if y <= T + TESTED_H else "held_pct" for y in years]
    out["is_forecast"] = [y > T for y in years]

    # --- outturn: published measure history ---
    if hist_years:
        hh = h.reindex(hist_years)
        out.loc[hist_years, "receipts"] = pd.to_numeric(hh["receipts"], errors="coerce")
        out.loc[hist_years, "primary_spend"] = pd.to_numeric(hh["primary"], errors="coerce")
        out.loc[hist_years, "interest"] = pd.to_numeric(hh["interest"], errors="coerce")

    # --- forecast: US = DTS bucket path (rebased to FRED); else share rules ---
    info: dict = {"methods": {line: OA.method(cc, line) for line in ("receipts", "primary")}}
    pol = None
    if fut:
        dts_years: list[int] = []
        if cc == "us":
            from sovereign_cash.scenarios.channels import us_dts as UD

            # DTS only inside the tested window (through Y_END when US_DTS_TO_YEND). Passing a missing FY used to
            # return None and drop the whole US path onto share rules before the WEO handoff.
            if UD.available():
                near = [y for y in fut if y <= (Y_END if US_DTS_TO_YEND else T + TESTED_H)]
                rb = UD.rebased(h, T, near) if near else None
                if rb is not None and len(rb):
                    dts_years = [int(y) for y in rb.index]
                    out.loc[dts_years, "receipts"] = pd.to_numeric(rb["receipts"], errors="coerce")
                    out.loc[dts_years, "primary_spend"] = pd.to_numeric(rb["primary_spend"], errors="coerce")
                    out.loc[dts_years, "interest"] = pd.to_numeric(rb["interest"], errors="coerce")
                    out.loc[dts_years, "rule"] = "dts_fy"
                    info["methods"] = {"receipts": "dts_fy", "primary": "dts_fy",
                                       "source": "us.fiscal fiscal_path_buckets × FRED origin"}
        rest = [y for y in fut if y not in dts_years]
        if rest:
            fc, _, info_oa = OA.forecast(cc, rest)
            out.loc[rest, "receipts"] = pd.to_numeric(fc["receipts"], errors="coerce")
            out.loc[rest, "primary_spend"] = pd.to_numeric(fc["primary"], errors="coerce")
            out.loc[rest, "interest"] = pd.to_numeric(fc["interest"], errors="coerce")
            # Japan: OA.forecast load_methods() restores P-driver persist; rewrite spend onto GDP so
            # Y−P drift cannot invent a surplus before held_pct / weo_pct.
            if cc in PERSIST_GDP_PRIMARY and T in h.index and np.isfinite(h.at[T, "primary"]):
                q_y = float(h.at[T, "primary"]) / float(out.at[T, "gdp"]) if T in out.index else float(
                    h.at[T, "primary"]) / float(OA.levels(cc, [T])[0][T])
                for y in rest:
                    out.at[y, "primary_spend"] = q_y * float(out.at[y, "gdp"])
                info_oa = {**info_oa, "methods": {**(info_oa.get("methods") or {}),
                                                  "primary": "persist_gdp",
                                                  "primary_note": "forced: P-driver manufactures surplus into WEO handoff"}}
            if not dts_years:
                info = info_oa
            # Beyond T+3: provisional held residual (fallback if WEO cell missing); overwritten by weo_pct below.
            hold_y = T + TESTED_H
            if hold_y in out.index and any(y > hold_y for y in rest):
                base_pct = float(
                    (out.at[hold_y, "primary_spend"] - out.at[hold_y, "receipts"]) / out.at[hold_y, "gdp"])
                for y in rest:
                    if y > hold_y:
                        out.at[y, "primary_spend"] = out.at[y, "receipts"] + base_pct * out.at[y, "gdp"]
                        out.at[y, "rule"] = "held_pct"
        pol = _policy(cc, fut, T)
        if pol is not None:
            from sovereign_cash.policy.overlay import LINE_IDS

            ids = LINE_IDS[cc]
            out.loc[fut, "policy_tax"] = pol["by_line"][ids["tax"]]
            out.loc[fut, "policy_spend"] = pol["by_line"][ids["spend"]]
            if cc == "us" and dts_years and not US_DTS_TAX_OVERLAY:
                # The DTS levels already carry the 2025 act's tax effects (corporate, refunds, withholding), and CBO's
                # FY26->27 tax increment is mostly the timing of the TCJA-expiry baseline, not new cash. Keep the
                # spending measures (Medicaid / SNAP cuts phase in after the base year); drop the tax increment.
                out.loc[dts_years, "policy_tax"] = 0.0

        # Medium-term: GG primary = WEO (panel-wide). Policy already in WEO; interest stays on the ISIN book.
        # Measure primary is set so measure% + held ω = WEO (ω=0 on GG measures).
        if cc in WEO_PRIMARY_HOLD:
            weo = weo_primary_deficit_pct(cc)
            w0, _ = wedge(cc, T)
            w0 = float(w0) if np.isfinite(w0) else 0.0
            for y in fut:
                if y <= T + TESTED_H or y in dts_years:
                    continue
                if y not in weo.index or not np.isfinite(weo[y]):
                    continue
                out.at[y, "policy_tax"] = 0.0
                out.at[y, "policy_spend"] = 0.0
                measure_pct = float(weo[y]) - w0
                target = measure_pct / 100.0 * float(out.at[y, "gdp"])
                out.at[y, "primary_spend"] = float(out.at[y, "receipts"]) + target
                out.at[y, "rule"] = "weo_pct"

    out["primary_deficit"] = (out["primary_spend"] + out["policy_spend"] - out["receipts"] - out["policy_tax"])
    out["primary_deficit_pct"] = 100.0 * out["primary_deficit"] / out["gdp"]

    w, wy = wedge(cc, T)
    w_held = w if np.isfinite(w) else 0.0
    anchor_info = None
    Y = ANCHOR_YEAR
    if Y in fut and out.at[Y, "rule"] not in ("weo_pct", "dts_fy"):
        anc = anchor_primary_pct(cc, Y)
        if anc is not None:
            target, src = anc
            delta = target - (float(out.at[Y, "primary_deficit_pct"]) + w_held)
            handoff = T + TESTED_H + 1
            for y in fut:
                if y < Y or out.at[y, "rule"] in ("weo_pct", "dts_fy"):
                    continue
                wgt = 1.0 if y == Y else max(0.0, (handoff - y) / (handoff - Y))
                out.at[y, "primary_spend"] = float(out.at[y, "primary_spend"]) + wgt * delta / 100.0 * float(out.at[y, "gdp"])
            out["primary_deficit"] = (out["primary_spend"] + out["policy_spend"] - out["receipts"] - out["policy_tax"])
            out["primary_deficit_pct"] = 100.0 * out["primary_deficit"] / out["gdp"]
            anchor_info = {"year": Y, "target_pct": round(target, 3), "engine_delta_pct": round(delta, 3), "source": src}
    gg = gg_primary_pct(cc)
    wedge_col = pd.Series(w_held, index=out.index, dtype=float)
    gg_col = out["primary_deficit_pct"] + w_held
    for y in hist_years:
        if y in gg.index and np.isfinite(gg[y]):
            # year-specific ω so history primary_gg ≡ AMECO (Commission), not the held origin wedge
            wedge_col.at[y] = float(gg[y] - out.at[y, "primary_deficit_pct"])
            gg_col.at[y] = float(gg[y])
    out["wedge_pct"] = wedge_col
    out["primary_gg_pct"] = gg_col
    try:
        from sovereign_cash.scenarios.channels import sarimax as SX
        measure = {**getattr(SX, "MEASURE", {}), **OA.MEASURE}.get(cc, OA.MEASURE.get(cc, cc))
    except Exception:  # noqa: BLE001
        measure = OA.MEASURE.get(cc, cc)
    out.attrs.update({
        "cc": cc, "origin_period": T, "measure": measure, "methods": info["methods"],
        "wedge_pct": None if not np.isfinite(w) else round(float(w), 4), "wedge_year": wy,
        "policy_events": (pol or {}).get("events", []), "policy_n_measures": (pol or {}).get("n_measures", 0),
        "anchor": anchor_info,
    })
    return out


def build(countries=COUNTRIES) -> dict:
    out = {"countries": {}, "errors": {}}
    for cc in countries:
        try:
            p = path(cc)
        except Exception as exc:  # noqa: BLE001 - one country must not sink the file
            log.exception("primary: %s failed", cc)
            out["errors"][cc] = f"{type(exc).__name__}: {exc}"
            continue
        rows = {}
        for y, r in p.iterrows():
            row = {}
            for k, v in r.items():
                if k == "is_forecast":
                    row[k] = bool(v)
                elif k == "rule":
                    row[k] = str(v)
                else:
                    row[k] = None if not np.isfinite(float(v)) else round(float(v), 4)
            rows[str(y)] = row
        out["countries"][cc] = {**p.attrs, "years": rows}
    return out


def write(res: dict | None = None, path_: Path = PATH) -> Path:
    res = build() if res is None else res
    path_.parent.mkdir(exist_ok=True)
    path_.write_text(json.dumps(res, indent=1))
    return path_


if __name__ == "__main__":
    res = build()
    write(res)
    for cc, c in res["countries"].items():
        ys = c["years"]
        row = " ".join(f"{y}:{v['primary_gg_pct']:+.1f}" for y, v in ys.items() if int(y) >= 2025 and v["primary_gg_pct"] is not None)
        print(f"{cc}: origin {c['origin_period']} wedge {c['wedge_pct']} ({c['wedge_year']}) {c['methods']} "
              f"| GG primary deficit % GDP {row}")
    for cc, e in res["errors"].items():
        print(cc, "ERROR", e)
