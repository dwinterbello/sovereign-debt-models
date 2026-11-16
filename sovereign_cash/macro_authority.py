"""Authority macro drivers for the monthly component SARIMAX and the annual share engine.

History is Eurostat (monthly HICP YoY, quarterly NGDP YoY held flat). Forward years default
to AMECO NGDP and WEO CPI. Country overlays:

  FR  DG Trésor RESF (real growth + CPI → NGDP ≈ real + cpi) on the RESF years — not AMECO/WEO.
      AFT publishes financing and charge de la dette, not the macro scenario.
  IT  DFP Tav. II.1.2.2 nominal GDP on the plan years.
  all Own bottom-up CPI annual (``own_inflation.SOURCE``) under ``cpi_override_gate(cc)`` —
      default mode ``full`` (each country's BU path through 2030). Realisation check can flip
      to ``weo`` if post-forecast prints fail; ``near`` (h≤6 splice) is research-only.
      Forecast-year NGDP = WEO real × own CPI, except RESF (FR: NGDP+CPI) / DFP (IT: NGDP) pins.
      Beyond 2030: WEO. See ``experiments/own_cpi_override_wire.md``, ``own_cpi_vs_weo_vintages.md``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.monthly_components import monthly_drivers

DEBT = Path(__file__).resolve().parent / "data" / "debt"
FR_AUTH = Path(__file__).resolve().parent / "data" / "fr" / "authority.json"

# Own-CPI fiscal override, every country with an own path.
# Default FULL_BU=True: ship each country's full BU annual (close enough vs WEO on the
# origin-honest replays in own_cpi_vs_weo_vintages.md — EA/US/JP/NO/SE; UK n=1 valid).
# Realisation fail still drops to WEO. near splice kept as FALLBACK_MODE if FULL_BU is False.
FULL_BU = True
REALISE_H = 6          # months after last print (near-mode / realisation window)
REALISE_MIN_N = 3      # printed post-forecast months needed to judge realisation
MAE_BAND = 0.15        # pp — own may be this much worse than WEO-flat and still pass
# back-compat names (UK-only wire, 2026-11-08 morning)
UK_OVERRIDE_CC = "uk"
UK_REALISE_H, UK_REALISE_MIN_N, UK_MAE_BAND = REALISE_H, REALISE_MIN_N, MAE_BAND


def override_countries() -> tuple[str, ...]:
    """Countries the override can apply to: every own_inflation source (path presence decides the mode)."""
    from sovereign_cash import own_inflation as OI
    return tuple(OI.SOURCE)


def first_forecast_year() -> int:
    return int(pd.Timestamp.today().year)


def _ameco_ngdp_yoy(cc: str) -> pd.Series:
    ameco = pd.read_parquet(DEBT / "ameco.parquet")
    g = (ameco[(ameco["country"] == cc) & (ameco["var"] == "gdp_lcu_bn")]
         .sort_values("year").set_index("year")["value"].astype(float))
    return (g / g.shift(1) - 1.0) * 100.0


def _weo_cpi_yoy(cc: str) -> pd.Series:
    weo = pd.read_parquet(DEBT / "weo.parquet")
    return (weo[(weo["country"] == cc) & (weo["var"] == "cpi")]
            .set_index("year")["value"].astype(float).sort_index())


def _weo_real_yoy(cc: str) -> pd.Series:
    weo = pd.read_parquet(DEBT / "weo.parquet")
    return (weo[(weo["country"] == cc) & (weo["var"] == "real_growth")]
            .set_index("year")["value"].astype(float).sort_index())


def _dfp_ngdp_yoy() -> pd.Series:
    """Italy DFP nominal GDP YoY for years the table covers."""
    from sovereign_cash.it import events as ev
    ameco = _ameco_ngdp_yoy("it")
    out = {}
    years = sorted(ev.PLAN)
    prev = None
    ameco_gdp = pd.read_parquet(DEBT / "ameco.parquet")
    g = (ameco_gdp[(ameco_gdp["country"] == "it") & (ameco_gdp["var"] == "gdp_lcu_bn")]
         .set_index("year")["value"].astype(float))
    for y in years:
        level = ev.PLAN[y]["gdp_mn"] / 1000.0  # bn
        if prev is None:
            base = float(g.get(y - 1, np.nan))
            if not np.isfinite(base) or base <= 0:
                out[y] = float(ameco.get(y, np.nan))
            else:
                out[y] = 100.0 * (level / base - 1.0)
        else:
            out[y] = 100.0 * (level / prev - 1.0)
        prev = level
    return pd.Series(out)


def _resf_fr() -> tuple[pd.Series, pd.Series, str]:
    """DG Trésor RESF: CPI as printed; NGDP ≈ real growth + CPI."""
    raw = json.loads(FR_AUTH.read_text())["macro"]
    ngdp, cpi = {}, {}
    for y, row in raw["years"].items():
        yi = int(y)
        cpi[yi] = float(row["cpi"])
        ngdp[yi] = float(row["real_growth"]) + float(row["cpi"])
    return pd.Series(ngdp), pd.Series(cpi), raw["source"]


def official_macro(cc: str) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """Published authority macro as comparator (RESF / DFP / WEO), not the desk driver."""
    if cc == "fr" and FR_AUTH.exists():
        ngdp, cpi, src = _resf_fr()
        return (ngdp.sort_index(), cpi.sort_index(),
                {"ngdp": f"{src} (real + CPI)", "cpi": src})
    ngdp = _ameco_ngdp_yoy(cc)
    cpi = _weo_cpi_yoy(cc)
    src = {"ngdp": "AMECO nominal GDP", "cpi": "IMF WEO CPI"}
    if cc == "it":
        dfp = _dfp_ngdp_yoy()
        ngdp = ngdp.copy()
        for y, v in dfp.items():
            if np.isfinite(v):
                ngdp.loc[int(y)] = float(v)
        src["ngdp"] = "AMECO nominal GDP + DFP Tav. II.1.2.2"
    return ngdp.dropna(), cpi.dropna(), src


# ---------------------------------------------------------------- own-CPI override, all countries (h≤6 realisation gate)

def _node(cc: str) -> dict:
    from sovereign_cash import own_inflation as OI
    if not OI.DERIVED.exists():
        return {}
    return json.loads(OI.DERIVED.read_text()).get(cc) or {}


@lru_cache(maxsize=16)
def _official_monthly(cc: str) -> pd.Series:
    """Published monthly YoY on the series the own path targets (own_inflation.HISTORY); empty on failure."""
    from sovereign_cash import own_inflation as OI
    try:
        s = OI._history(cc)  # noqa: SLF001 — same series the path was built on
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)
    if s is None or not len(s):
        return pd.Series(dtype=float)
    s = s.astype(float).sort_index()
    return s[s.index <= pd.Timestamp.today() + pd.offsets.MonthEnd(0)]


@lru_cache(maxsize=16)
def _monthly_own(cc: str) -> pd.Series:
    """Published + forecast monthly YoY on the own path's basis."""
    from sovereign_cash import own_inflation as OI
    node = _node(cc)
    if not node:
        return pd.Series(dtype=float)
    hist = _official_monthly(cc)
    fc = pd.Series({pd.Timestamp(f"{k}-01") + pd.offsets.MonthEnd(0): float(v)
                    for k, v in (node.get("forecast_monthly") or {}).items()}, dtype=float)
    if fc.empty:
        return hist.sort_index()
    fc.index = OI._me(fc.index)  # noqa: SLF001
    out = pd.concat([hist, fc[~fc.index.isin(hist.index)]]).sort_index()
    return out.astype(float)


def cpi_override_gate(cc: str) -> dict:
    """Own-CPI fiscal override status for ``cc``.

    Modes
      full  — default when ``FULL_BU`` (ship each country's BU annual through END_YEAR).
      near  — research fallback if ``FULL_BU`` is False: own months within REALISE_H of last
              print, WEO fill beyond.
      weo   — realisation failed (n≥REALISE_MIN_N prints and own MAE > WEO-flat + MAE_BAND).
      off   — no own path on disk for ``cc`` (or ``cc`` has no own source).
    """
    from sovereign_cash import own_inflation as OI

    if cc not in OI.SOURCE:
        return {"mode": "off", "reason": f"no own inflation source for {cc}", "n_printed": 0}
    node = _node(cc)
    own_ann = OI.annual_cpi(cc)
    if not node or own_ann.empty:
        return {"mode": "off", "reason": f"no own {cc} path", "n_printed": 0}

    first_fc = pd.Timestamp(node["first_forecast"]) if node.get("first_forecast") else None
    hist_last = pd.Timestamp(node["history_last"]) if node.get("history_last") else None
    monthly = _monthly_own(cc)
    official = _official_monthly(cc)
    weo = _weo_cpi_yoy(cc)
    if hist_last is None and len(official):
        hist_last = official.index.max()

    printed = []
    if first_fc is not None and len(monthly) and len(official):
        fc_m = {pd.Timestamp(f"{k}-01") + pd.offsets.MonthEnd(0): float(v)
                for k, v in (node.get("forecast_monthly") or {}).items()}
        for m, act in official.items():
            if m < first_fc.to_period("M").to_timestamp(how="end"):
                continue
            if m not in fc_m or not np.isfinite(fc_m[m]):
                continue
            y = int(m.year)
            weo_m = float(weo.loc[y]) if y in weo.index else float("nan")
            printed.append({"month": m.strftime("%Y-%m"), "actual": float(act), "own": float(fc_m[m]), "weo_flat": weo_m})

    base = {
        "cc": cc, "n_printed": len(printed), "printed": printed,
        "hist_last": str(hist_last.date()) if hist_last is not None else None,
        "first_forecast": str(first_fc.date()) if first_fc is not None else None,
        "realise_h": REALISE_H, "history_ok": bool(len(official)), "full_bu": FULL_BU,
    }
    n = len(printed)
    if n >= REALISE_MIN_N:
        own_mae = float(np.mean([abs(r["own"] - r["actual"]) for r in printed]))
        weo_rows = [abs(r["weo_flat"] - r["actual"]) for r in printed if np.isfinite(r["weo_flat"])]
        weo_mae = float(np.mean(weo_rows)) if weo_rows else float("nan")
        ok = (not np.isfinite(weo_mae)) or own_mae <= weo_mae + MAE_BAND
        if not ok:
            return {**base, "mode": "weo", "own_mae": own_mae, "weo_mae": weo_mae,
                    "reason": (f"realisation fail: n={n} own MAE {own_mae:.3f} > WEO-flat "
                               f"{weo_mae:.3f} + {MAE_BAND}")}
        return {**base, "mode": "full", "own_mae": own_mae, "weo_mae": weo_mae,
                "reason": (f"realisation pass: n={n} house MAE {own_mae:.3f} ≤ WEO-flat "
                           f"{weo_mae:.3f} + {MAE_BAND}; house CPI through {OI.END_YEAR}")}
    if FULL_BU:
        return {**base, "mode": "full",
                "reason": (f"house CPI annual through {OI.END_YEAR} "
                           f"(n_printed={n} < {REALISE_MIN_N}, awaiting realisation check)")}
    return {**base, "mode": "near",
            "reason": (f"n_printed={n} < {REALISE_MIN_N} — own months within {REALISE_H} of last print, "
                       f"WEO fill beyond (awaiting realisation)"
                       + ("" if base["history_ok"] else "; official history unavailable, published months of the open year fall to WEO"))}


@lru_cache(maxsize=16)
def own_cpi_annual(cc: str) -> tuple[pd.Series, dict]:
    """Calendar-year CPI for ``cc`` under the override gate. Empty series → caller keeps WEO."""
    from sovereign_cash import own_inflation as OI

    gate = cpi_override_gate(cc)
    mode = gate["mode"]
    if mode in ("off", "weo"):
        return pd.Series(dtype=float), gate
    own = OI.annual_cpi(cc)
    weo = _weo_cpi_yoy(cc)
    if mode == "full":
        return own.copy(), {**gate, "annual_source": f"own full through {OI.END_YEAR}"}

    # near: own monthly within REALISE_H of last print; WEO annual as flat fill beyond
    monthly = _monthly_own(cc)
    hist_last = pd.Timestamp(gate["hist_last"]) if gate.get("hist_last") else (
        monthly.dropna().index.max() if len(monthly.dropna()) else None)
    if hist_last is None or monthly.empty:
        return pd.Series(dtype=float), {**gate, "annual_source": "fallback empty"}
    cut = (hist_last + pd.offsets.MonthEnd(REALISE_H)).normalize()
    out = {}
    for y in range(first_forecast_year() - 1, OI.END_YEAR + 1):
        vals = []
        for m in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="ME"):
            if m in monthly.index and np.isfinite(monthly.loc[m]) and m <= cut:
                vals.append(float(monthly.loc[m]))
            elif y in weo.index and np.isfinite(weo.loc[y]):
                vals.append(float(weo.loc[y]))
            elif y in own.index and np.isfinite(own.loc[y]):
                vals.append(float(own.loc[y]))
        if len(vals) >= 10:
            out[y] = float(np.mean(vals))
    for y, v in own.items():  # closed published years
        if int(y) < first_forecast_year() and int(y) not in out:
            out[int(y)] = float(v)
    s = pd.Series(out, dtype=float).sort_index()
    return s, {**gate, "annual_source": f"own ≤ {cut.date()} (h≤{REALISE_H}), WEO fill beyond", "cut": str(cut.date())}


# back-compat wrappers (UK-only wire)
def uk_cpi_override_gate() -> dict:
    return cpi_override_gate(UK_OVERRIDE_CC)


def uk_own_cpi_annual() -> tuple[pd.Series, dict]:
    return own_cpi_annual(UK_OVERRIDE_CC)


def _apply_own_cpi(cc: str, ngdp: pd.Series, cpi: pd.Series, src: dict[str, str],
                   pin_cpi_years: frozenset[int] = frozenset(), pin_ngdp_years: frozenset[int] = frozenset(),
                   ) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """Replace forecast CPI with the gated own path and compound WEO real into NGDP.

    ``pin_cpi_years`` / ``pin_ngdp_years`` keep an authority overlay (FR RESF, IT DFP) on those years.
    """
    from sovereign_cash import own_inflation as OI

    own, gate = own_cpi_annual(cc)
    mode = gate.get("mode", "off")
    if own.empty or mode in ("off", "weo"):
        return ngdp, cpi, {**src, "cpi_override": gate.get("reason", "off"), "cpi_override_mode": mode}
    real = _weo_real_yoy(cc)
    weo_cpi = _weo_cpi_yoy(cc)
    ngdp, cpi = ngdp.copy(), cpi.copy()
    y0 = first_forecast_year()
    applied = []
    for y in range(y0, OI.END_YEAR + 1):
        if y in pin_cpi_years or y not in own.index or not np.isfinite(own.loc[y]):
            continue
        cpi.loc[y] = float(own.loc[y])
        applied.append(y)
        if y in pin_ngdp_years or y not in real.index or not np.isfinite(real.loc[y]):
            continue
        ngdp.loc[y] = ((1 + float(real.loc[y]) / 100) * (1 + float(own.loc[y]) / 100) - 1) * 100
    # beyond own horizon keep WEO CPI / WEO-compounded NGDP
    for y in weo_cpi.index:
        if y <= OI.END_YEAR or int(y) in pin_cpi_years:
            continue
        cpi.loc[int(y)] = float(weo_cpi.loc[y])
        if y in real.index and int(y) not in pin_ngdp_years:
            ngdp.loc[int(y)] = ((1 + float(real.loc[y]) / 100) * (1 + float(weo_cpi.loc[y]) / 100) - 1) * 100
    label = OI.source_label(cc)
    ngdp_years = [y for y in applied if y not in pin_ngdp_years]

    def _span(ys: list[int]) -> str:
        return f"{min(ys)}–{max(ys)}" if ys else "none"

    ngdp_src = f"IMF WEO real growth compounded with {cc.upper()} house CPI ({mode}, {_span(ngdp_years)})"
    # Opaque mode tag only — do not echo leaf / print-pack / module paths into client packs.
    cpi_src = f"{label} [{mode}: house path through {OI.END_YEAR}]"
    # an authority overlay (RESF / DFP) stays first in its label — tests and boards key on it
    if pin_ngdp_years:
        ngdp_src = src["ngdp"] + (f"; {ngdp_src}" if ngdp_years else "")
    if pin_cpi_years:
        cpi_src = src["cpi"] + (f"; {cpi_src} on {_span(applied)}" if applied else "")
    out_src = {
        **src, "ngdp": ngdp_src, "cpi": cpi_src,
        "cpi_override": gate.get("reason", ""), "cpi_override_mode": mode, "cpi_override_cut": gate.get("cut"),
        "cpi_override_n_printed": gate.get("n_printed", 0), "cpi_override_years": applied,
        "cpi_override_ngdp_years": ngdp_years,
    }
    return ngdp.sort_index(), cpi.sort_index(), out_src


def annual_authority(cc: str) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    """Calendar-year NGDP and CPI YoY for the chain's driver path."""
    ngdp = _ameco_ngdp_yoy(cc)
    cpi = _weo_cpi_yoy(cc)
    src = {"ngdp": "AMECO nominal GDP", "cpi": "IMF WEO CPI"}
    pin_cpi: frozenset[int] = frozenset()
    pin_ngdp: frozenset[int] = frozenset()
    if cc == "fr" and FR_AUTH.exists():
        r_ngdp, r_cpi, r_src = _resf_fr()
        ngdp, cpi = ngdp.copy(), cpi.copy()
        for y, v in r_ngdp.items():
            ngdp.loc[int(y)] = float(v)
        for y, v in r_cpi.items():
            cpi.loc[int(y)] = float(v)
        src = {"ngdp": f"{r_src} (real + CPI)", "cpi": r_src}
        pin_cpi = pin_ngdp = frozenset(int(y) for y in r_cpi.index)
    elif cc == "it":
        dfp = _dfp_ngdp_yoy()
        ngdp = ngdp.copy()
        for y, v in dfp.items():
            if np.isfinite(v):
                ngdp.loc[int(y)] = float(v)
        src["ngdp"] = "AMECO nominal GDP + DFP Tav. II.1.2.2"
        pin_ngdp = frozenset(int(y) for y, v in dfp.items() if np.isfinite(v))
    if cc in override_countries():
        ngdp, cpi, src = _apply_own_cpi(cc, ngdp, cpi, src, pin_cpi_years=pin_cpi, pin_ngdp_years=pin_ngdp)
    return ngdp.dropna(), cpi.dropna(), src


def driver_path(cc: str, origin: pd.Timestamp, h: int = 24,
                shock: dict[str, float] | None = None) -> dict[str, pd.Series]:
    """Monthly NGDP / HICP YoY through origin+h, authority past the last print.

    ``shock`` adds a constant pp to future months only: ``{"ngdp": 1.0}`` or
    ``{"hicp": 1.0}``. History is never shocked (real-time).
    """
    shock = shock or {}
    hist = monthly_drivers(cc)
    ngdp_a, cpi_a, src = annual_authority(cc)
    from sovereign_cash.interest_split import annual_effective_rate
    rates_a = annual_effective_rate(cc)
    annual = {"ngdp": ngdp_a, "cpi": cpi_a, "rates": rates_a}
    src = {**src, "rates": "debt interest / debt (authority path on forecast years)"}
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out: dict = {}
    from sovereign_cash.monthly_components import MACRO_LAG
    for key, annual_key, shock_key in (
            ("ngdp_yoy", "ngdp", "ngdp"),
            ("hicp_yoy", "cpi", "hicp"),
            ("rates", "rates", "rates"),
    ):
        if key not in hist and annual_key == "rates":
            from sovereign_cash.interest_split import rates_monthly
            hser = rates_monthly(cc, through=months.max())
        else:
            hser = hist[key]
        ann = annual[annual_key]
        vals = []
        for m in months:
            if m in hser.index and np.isfinite(hser.loc[m]):
                v = float(hser.loc[m])
            else:
                y = int(m.year)
                if y in ann.index and np.isfinite(ann.loc[y]):
                    v = float(ann.loc[y])
                elif len(ann.loc[:y]):
                    v = float(ann.loc[:y].iloc[-1])
                else:
                    v = float(hser.dropna().iloc[-1])
            v = v + float(shock.get(shock_key, 0.0))
            vals.append(v)
        raw = pd.Series(vals, index=months, name=key)
        full = hser.copy()
        for m, v in raw.items():
            full.loc[m] = v
        full = full.sort_index()
        out[key] = full.shift(MACRO_LAG).rename(key)
    out["_source"] = src
    out["_shock"] = shock
    return out
