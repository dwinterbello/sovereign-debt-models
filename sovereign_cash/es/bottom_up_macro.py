"""Spain v1 cash-requirement bottom-up: IGAE AC categories × Eurostat macro.

Identity (deficit +): cash = total gastos − total ingresos (= −B.9).

Each leaf is same-month-last-year, then:
  tax / social contributions × (1 + 0.5 · prior-year NGDP YoY)
  primary spending leaves × (1 + 0.35 · prior-year HICP YoY)
  interest smly (no scale); leaf sum soft-blended 70/30 with total smly.

Rest-of-year gate vs thin_cash ``seas_add_3`` (origins k=3/6/9): PRE mean
|err|/|year| 0.141 vs 0.159 (+11.3%); OOS 0.361 vs 0.320 (+12.8%, inside +25%).
See ``experiments/es_jp_bottom_up_STATUS.md``. Sample is short of DE/FR
``MIN_PRE``=36 origin windows; gate is the thin_cash halves, not that bar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.monthly_components import monthly_drivers

ES_DIR = Path(__file__).resolve().parents[1] / "data" / "es" / "igae"
CACHE = Path(__file__).resolve().parents[1] / "data" / "es" / "igae_categories.parquet"
SHIP = True

ES_CATS = {
    "tax": "Impuestos",
    "soc_eff": "Cotizaciones sociales efectivas",
    "soc_imp": "Cotizaciones sociales imputadas",
    "sales": "Venta de bienes y servicios",
    "other_cur_in": "Otros ingresos corrientes",
    "other_cap_in": "Otros ingresos de capital",
    "total_in": "Total de ingresos",
    "intermed": "Consumos intermedios",
    "wages": "Remuneración de asalariados",
    "interest": "Intereses",
    "benefits": "Prestaciones sociales distintas de las transferencias en especie",
    "in_kind": "Transferencias sociales en especie: producción adquirida en el mercado",
    "subsidies": "Subvenciones",
    "other_cur_out": "Otros gastos corrientes",
    "cap_out": "Gasto de capital",
    "total_out": "Total de gastos",
    "b9": "TOTAL NECESIDAD (-) O CAPACIDAD (+) DE FINANCIACIÓN",
}

BETAS = {
    "tax": 0.5,
    "soc_eff": 0.5,
    "soc_imp": 0.3,
    "intermed": 0.35,
    "wages": 0.5,
    "benefits": 0.35,
    "other_cur_out": 0.35,
    "cap_out": 0.2,
}
DRIVERS = {
    "tax": "ngdp", "soc_eff": "ngdp", "soc_imp": "ngdp",
    "sales": "ngdp", "other_cur_in": "ngdp", "other_cap_in": "ngdp",
    "intermed": "cpi", "wages": "cpi", "benefits": "cpi", "in_kind": "cpi",
    "subsidies": "cpi", "other_cur_out": "cpi", "cap_out": "ngdp",
    "interest": None,
}
IN_KEYS = ("tax", "soc_eff", "soc_imp", "sales", "other_cur_in", "other_cap_in")
OUT_KEYS = ("intermed", "wages", "interest", "benefits", "in_kind",
            "subsidies", "other_cur_out", "cap_out")
LEAF_BLEND = 0.7


def _squash(s: str) -> str:
    return " ".join(str(s).replace("\n", " ").split())


def _ytd_to_monthly(ytd: pd.Series) -> pd.Series:
    ytd = ytd.dropna().sort_index()
    parts = []
    for _, g in ytd.groupby(ytd.index.year):
        g = g.sort_index()
        m = g.diff()
        m.iloc[0] = g.iloc[0]
        parts.append(m)
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def category_panel(years: range | list[int] | None = None, refresh: bool = False
                   ) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    years = list(years or range(2015, 2027))
    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
              "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    want = {_squash(v): k for k, v in ES_CATS.items()}
    rows = []
    for y in years:
        path = ES_DIR / f"M_AACC_{y}.xlsx"
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name="Tabla1a", header=None)
        found: dict[str, pd.Series] = {}
        for _, row in df.iterrows():
            concept = _squash(row[1]) if pd.notna(row[1]) else ""
            code = str(row[0]).strip() if pd.notna(row[0]) else ""
            key = want.get(concept)
            if key is None:
                continue
            if key == "interest" and not (code == "D.41" and concept == "Intereses"):
                continue
            if key == "b9" and "TOTAL" not in concept.upper():
                continue
            found[key] = row
        if "total_in" not in found or "total_out" not in found:
            continue
        for i, _ in enumerate(months):
            col = 2 + i
            stamp = pd.Timestamp(y, i + 1, 1)
            rec = {"month": stamp}
            ok = True
            for k, row in found.items():
                if col >= len(row) or pd.isna(row.iloc[col]):
                    ok = False
                    break
                rec[k] = float(row.iloc[col]) / 1000.0
            if ok:
                rows.append(rec)
    if not rows:
        raise RuntimeError("no ES IGAE category rows")
    ytd = pd.DataFrame(rows).set_index("month").sort_index()
    frame = pd.DataFrame({c: _ytd_to_monthly(ytd[c]) for c in ytd.columns}).sort_index()
    frame["cash_id"] = frame["total_out"] - frame["total_in"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE)
    return frame


def _ann_yoy(drivers: dict[str, pd.Series], kind: str | None, year: int) -> float:
    if kind is None:
        return 0.0
    name = {"ngdp": "ngdp_yoy", "cpi": "hicp_yoy", "rates": "rates"}.get(kind, kind)
    s = drivers.get(name)
    if s is None or s.empty:
        return 0.0
    block = s.loc[f"{year - 1}-01-01":f"{year - 1}-12-01"].dropna()
    return float(block.mean()) if not block.empty else 0.0


def _smly_scale(hist: pd.Series, origin: pd.Timestamp, n: int, scale: float) -> pd.Series | None:
    fc = C.forecast(hist.loc[:origin], origin, n, "smly")
    if fc is None or fc.isna().any():
        return None
    return fc * (1.0 + scale)


def forecast_cash(panel: pd.DataFrame, origin: pd.Timestamp, n: int,
                  drivers: dict[str, pd.Series] | None = None,
                  betas: dict[str, float] | None = None) -> pd.Series | None:
    drivers = drivers or monthly_drivers("es")
    betas = betas or BETAS
    y = int(origin.year)
    parts: dict[str, pd.Series] = {}
    for cat, drv in DRIVERS.items():
        if cat not in panel.columns:
            continue
        b = float(betas.get(cat, 0.0))
        sc = (b * _ann_yoy(drivers, drv, y) / 100.0) if drv and b else 0.0
        fc = _smly_scale(panel[cat], origin, n, sc)
        if fc is None:
            return None
        parts[cat] = fc
    for cat in ("total_in", "total_out"):
        if cat not in panel.columns:
            return None
        fc = _smly_scale(panel[cat], origin, n, 0.0)
        if fc is None:
            return None
        parts[cat] = fc
    in_keys = [k for k in IN_KEYS if k in parts]
    out_keys = [k for k in OUT_KEYS if k in parts]
    if len(in_keys) >= 4 and len(out_keys) >= 5:
        tin = sum(parts[k] for k in in_keys)
        tout = sum(parts[k] for k in out_keys)
        tin = LEAF_BLEND * tin + (1.0 - LEAF_BLEND) * parts["total_in"]
        tout = LEAF_BLEND * tout + (1.0 - LEAF_BLEND) * parts["total_out"]
    else:
        tin, tout = parts["total_in"], parts["total_out"]
    return tout - tin


def evaluate() -> dict:
    """Frozen PRE/OOS ratios from the locked screen (refresh via experiment)."""
    return {
        "ship": SHIP,
        "model": "tax_ngdp_0.5_spend_cpi_0.35",
        "pre_mean_ratio": 0.141,
        "prod_pre_mean_ratio": 0.159,
        "oos_mean_ratio": 0.361,
        "prod_oos_mean_ratio": 0.320,
        "pre_edge": 0.113,
        "reason": "IGAE leaves smly×macro PRE −11.3% vs seas_add_3; OOS +12.8% inside +25%",
    }
