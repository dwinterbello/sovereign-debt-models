"""Spain / Japan category×macro screen vs thin_cash baselines.

Spain: IGAE Administración Central Tabla1a category YTD → monthly. Identity
  cash = total_gastos − total_ingresos (= −B.9). Drivers via Eurostat ES
  HICP / NGDP (``monthly_components.monthly_drivers``).

Japan: MoF GA tax / rev_ex_bond / exp from cached workbooks. Sample starts
  FY2024 — one complete holdout year. Not PRE-gateable.

Gate (Spain, same rest-of-year meter as thin_cash): mean |err|/|year| on
origins k=3,6,9 must beat ``seas_add_3`` by ≥5% on PRE half (fy≤2019) and
not worsen the OOS half (fy≥2022) by >25%. Japan: beat ``smly`` on the
single scored FY if possible; report only.

Writes ``es_jp_bottom_up_STATUS.md`` + ``es_jp_bottom_up_numbers.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash import thin_cash as TC
from sovereign_cash.core import cash_framework as C
from sovereign_cash.monthly_components import monthly_drivers

HERE = Path(__file__).resolve().parent
ES_DIR = HERE.parent / "data" / "es" / "igae"
OUT_MD = HERE / "es_jp_bottom_up_STATUS.md"
OUT_JSON = HERE / "es_jp_bottom_up_numbers.json"

# IGAE concept keys (exact CONCEPTOS match after whitespace squash).
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
    "interest": "Intereses",  # expenditure D.41 (exact; receipt is "* Intereses")
    "benefits": "Prestaciones sociales distintas de las transferencias en especie",
    "in_kind": "Transferencias sociales en especie: producción adquirida en el mercado",
    "subsidies": "Subvenciones",
    "other_cur_out": "Otros gastos corrientes",
    "cap_out": "Gasto de capital",
    "total_out": "Total de gastos",
    "b9": "TOTAL NECESIDAD (-) O CAPACIDAD (+) DE FINANCIACIÓN",
}

# Receipts grow with NGDP; primary spend with HICP; interest with rates or smly.
ES_DRIVER = {
    "tax": "ngdp",
    "soc_eff": "ngdp",
    "soc_imp": "ngdp",
    "sales": "ngdp",
    "other_cur_in": "ngdp",
    "other_cap_in": "ngdp",
    "intermed": "cpi",
    "wages": "cpi",
    "benefits": "cpi",
    "in_kind": "cpi",
    "subsidies": "cpi",
    "other_cur_out": "cpi",
    "cap_out": "ngdp",
    "interest": None,
}

REL_EDGE = 0.05
OOS_TOL = 0.25


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


def es_category_panel(years: range | list[int] | None = None) -> pd.DataFrame:
    """Monthly €bn by category from cached M_AACC_*.xlsx Tabla1a."""
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
            # Expenditure interest is D.41 + exact "Intereses"; receipt is "* Intereses".
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
                rec[k] = float(row.iloc[col]) / 1000.0  # mn → bn
            if ok:
                rows.append(rec)
    if not rows:
        raise RuntimeError("no ES IGAE category rows")
    ytd = pd.DataFrame(rows).set_index("month").sort_index()
    out = {c: _ytd_to_monthly(ytd[c]) for c in ytd.columns}
    frame = pd.DataFrame(out).sort_index()
    # Identity check: cash ≈ total_out − total_in ≈ −b9
    frame["cash_id"] = frame["total_out"] - frame["total_in"]
    if "b9" in frame:
        frame["cash_b9"] = -frame["b9"]
    return frame


def _ann_yoy(drivers: dict[str, pd.Series], kind: str, year: int) -> float:
    """Calendar-year mean of lag-1 YoY for year ``year`` (honest at Jan origin)."""
    name = {"ngdp": "ngdp_yoy", "cpi": "hicp_yoy", "rates": "rates"}.get(kind, kind)
    s = drivers.get(name)
    if s is None or s.empty:
        return 0.0
    # use prior calendar year mean (known by March of ``year``)
    block = s.loc[f"{year - 1}-01-01":f"{year - 1}-12-01"].dropna()
    if block.empty:
        return 0.0
    return float(block.mean())


def _smly_scale(hist: pd.Series, origin: pd.Timestamp, n: int, scale: float) -> pd.Series | None:
    fc = C.forecast(hist.loc[:origin], origin, n, "smly")
    if fc is None or fc.isna().any():
        return None
    return fc * (1.0 + scale)


def es_forecast_cash(panel: pd.DataFrame, origin: pd.Timestamp, n: int,
                     betas: dict[str, float], drivers: dict[str, pd.Series]
                     ) -> pd.Series | None:
    """Category smly × (1+β·driver); cash = Σ out − Σ in (interest out carved)."""
    y = int(origin.year)
    # If rest-of-year crosses into next calendar year, still use this year's scale
    # (thin_cash rest-of-FY stays inside calendar year for ES).
    scales = {}
    for cat, drv in ES_DRIVER.items():
        if cat not in panel.columns:
            continue
        b = float(betas.get(cat, 0.0))
        if drv is None or b == 0.0:
            scales[cat] = 0.0
        else:
            scales[cat] = b * _ann_yoy(drivers, drv, y) / 100.0

    parts = {}
    for cat in list(ES_DRIVER) + ["total_in", "total_out"]:
        if cat not in panel.columns:
            continue
        sc = scales.get(cat, 0.0)
        fc = _smly_scale(panel[cat], origin, n, sc)
        if fc is None:
            return None
        parts[cat] = fc

    # Prefer leaf identity when leaves present; else total_in/out.
    in_keys = [k for k in ("tax", "soc_eff", "soc_imp", "sales", "other_cur_in", "other_cap_in")
               if k in parts]
    out_keys = [k for k in ("intermed", "wages", "interest", "benefits", "in_kind",
                            "subsidies", "other_cur_out", "cap_out") if k in parts]
    if len(in_keys) >= 4 and len(out_keys) >= 5:
        tin = sum(parts[k] for k in in_keys)
        tout = sum(parts[k] for k in out_keys)
        # residual other income/spend vs totals — hold via total − leaves if available
        if "total_in" in parts and "total_out" in parts:
            # soft: blend leaf sum toward total smly to absorb unmatched lines
            tin = 0.7 * tin + 0.3 * parts["total_in"]
            tout = 0.7 * tout + 0.3 * parts["total_out"]
    else:
        tin, tout = parts["total_in"], parts["total_out"]
    return tout - tin


def rest_errors_es_bu(panel: pd.DataFrame, betas: dict[str, float],
                      drivers: dict[str, pd.Series]) -> pd.DataFrame:
    y = panel["cash_id"].dropna()
    start = 1
    rows = []
    fys = sorted({TC.fy_of(t, start) for t in y.index})
    for fy in fys:
        block = TC._fy_block(y, fy, start)
        if block is None:
            continue
        year_total = float(block.sum())
        if abs(year_total) < 1e-9:
            continue
        for k in TC.ORIGIN_MONTHS:
            origin = TC.month_ts(fy, k, start)
            if origin not in y.index:
                continue
            n = 12 - k
            fc = es_forecast_cash(panel.loc[:origin], origin, n, betas, drivers)
            if fc is None or len(fc) != n or fc.isna().any():
                continue
            act_idx = [TC.month_ts(fy, m, start) for m in range(k + 1, 13)]
            act = float(y.loc[act_idx].sum())
            err = float(fc.sum()) - act
            rows.append({"fy": fy, "k": k, "err": err, "ratio": abs(err) / abs(year_total)})
    return pd.DataFrame(rows)


def score_halves(err: pd.DataFrame) -> dict:
    return {
        "pre": TC.mean_ratio(err, fy_max=2019),
        "oos": TC.mean_ratio(err, fy_min=2022),
        "all": TC.mean_ratio(err),
        "n_pre": int(((err["fy"] <= 2019).sum()) if not err.empty else 0),
        "n_oos": int(((err["fy"] >= 2022).sum()) if not err.empty else 0),
    }


def screen_es() -> dict:
    panel = es_category_panel()
    drivers = monthly_drivers("es")
    base_y = panel["cash_id"]
    base_err = TC.rest_errors(base_y, "es", TC.ES_MODEL)
    base = score_halves(base_err)

    specs = {
        "smly_leaves": {k: 0.0 for k in ES_DRIVER},
        "tax_ngdp_0.5_spend_cpi_0.35": {
            **{k: 0.0 for k in ES_DRIVER},
            "tax": 0.5, "soc_eff": 0.5, "soc_imp": 0.3,
            "intermed": 0.35, "wages": 0.5, "benefits": 0.35,
            "other_cur_out": 0.35, "cap_out": 0.2,
        },
        "tax_ngdp_1_spend_cpi_1": {
            **{k: 0.0 for k in ES_DRIVER},
            "tax": 1.0, "soc_eff": 1.0,
            "intermed": 1.0, "wages": 1.0, "benefits": 1.0,
            "other_cur_out": 1.0, "cap_out": 0.5,
        },
        "totals_ngdp_cpi": {},  # special: scale total_in by ngdp, total_out by cpi
    }

    results = {"baseline_seas_add_3": base}
    best = None
    for name, betas in specs.items():
        if name == "totals_ngdp_cpi":
            # hack via custom loop
            err_rows = []
            y = base_y
            start = 1
            for fy in sorted({TC.fy_of(t, start) for t in y.index}):
                block = TC._fy_block(y, fy, start)
                if block is None:
                    continue
                year_total = float(block.sum())
                if abs(year_total) < 1e-9:
                    continue
                for k in TC.ORIGIN_MONTHS:
                    origin = TC.month_ts(fy, k, start)
                    if origin not in y.index:
                        continue
                    n = 12 - k
                    g_in = 0.5 * _ann_yoy(drivers, "ngdp", int(origin.year)) / 100.0
                    g_out = 0.35 * _ann_yoy(drivers, "cpi", int(origin.year)) / 100.0
                    tin = _smly_scale(panel["total_in"], origin, n, g_in)
                    tout = _smly_scale(panel["total_out"], origin, n, g_out)
                    if tin is None or tout is None:
                        continue
                    fc = tout - tin
                    act = float(y.loc[[TC.month_ts(fy, m, start) for m in range(k + 1, 13)]].sum())
                    err = float(fc.sum()) - act
                    err_rows.append({"fy": fy, "k": k, "err": err,
                                     "ratio": abs(err) / abs(year_total)})
            err = pd.DataFrame(err_rows)
        else:
            err = rest_errors_es_bu(panel, betas, drivers)
        sc = score_halves(err)
        results[name] = sc
        if np.isfinite(sc["pre"]) and np.isfinite(base["pre"]):
            edge = (base["pre"] - sc["pre"]) / base["pre"]
            oos_ok = (not np.isfinite(sc["oos"]) or not np.isfinite(base["oos"])
                      or sc["oos"] <= base["oos"] * (1.0 + OOS_TOL))
            clear = edge >= REL_EDGE and oos_ok
            sc["pre_edge"] = edge
            sc["clears"] = clear
            if clear and (best is None or sc["pre"] < best["pre"]):
                best = {"name": name, **sc}

    # also score baseline identity vs published cash_requirement
    pub = TC.load("es")
    id_mae = float((panel["cash_id"].reindex(pub.index) - pub).abs().mean())
    ship = best is not None
    return {
        "panel_months": int(panel.shape[0]),
        "identity_mae_vs_pub_bn": id_mae,
        "results": results,
        "best": best,
        "ship": ship,
        "reason": (f"SHIP {best['name']} PRE edge {best['pre_edge']:.1%}"
                   if ship else "NO-SHIP: no category×macro clears ≥5% PRE vs seas_add_3 "
                   "with OOS ≤+25%"),
    }


def screen_jp() -> dict:
    from sovereign_cash.debt.note.monthly_primary import jp_panel_from_disk, jp_cash_tax_corrected

    frame = jp_panel_from_disk()
    cash = jp_cash_tax_corrected()
    # drivers: WEO/AMECO annual NGDP YoY as monthly constant (Eurostat JP flaky)
    ameco = pd.read_parquet(HERE.parent / "data" / "debt" / "ameco.parquet")
    gdp = (ameco[(ameco.country == "jp") & (ameco["var"] == "gdp_lcu_bn")]
           .set_index("year")["value"].sort_index().astype(float))
    ngdp_yoy = gdp.pct_change() * 100.0

    base_err = TC.rest_errors(cash, "jp", TC.JP_MODEL)
    base = {
        "all": TC.mean_ratio(base_err),
        "n": int(len(base_err)),
        "by_fy": base_err.groupby("fy")["ratio"].mean().to_dict() if not base_err.empty else {},
    }

    def jp_bu(origin, n, b_tax=0.5, b_exp=0.35):
        fy = TC.fy_of(origin, 4)
        g = float(ngdp_yoy.get(fy - 1, 0.0) or 0.0) / 100.0
        tax = _smly_scale(frame["tax_bn"], origin, n, b_tax * g) if "tax_bn" in frame else None
        rev = _smly_scale(frame["rev_ex_bond_bn"], origin, n, b_tax * g)
        exp = _smly_scale(frame["exp_bn"], origin, n, b_exp * g)
        if rev is None or exp is None:
            return None
        # tax overlay: replace tax share of rev if tax path exists
        if tax is not None:
            # rev_ex = tax + other; scale other with tax path residual
            other_hist = frame["rev_ex_bond_bn"] - frame["tax_bn"]
            other = _smly_scale(other_hist, origin, n, 0.0)
            if other is None:
                return None
            rev = tax + other
        return exp - rev

    rows = []
    start = 4
    for fy in sorted({TC.fy_of(t, start) for t in cash.index}):
        block = TC._fy_block(cash, fy, start)
        if block is None:
            continue
        year_total = float(block.sum())
        for k in TC.ORIGIN_MONTHS:
            origin = TC.month_ts(fy, k, start)
            if origin not in cash.index:
                continue
            n = 12 - k
            fc = jp_bu(origin, n)
            if fc is None or len(fc) != n:
                continue
            act = float(cash.loc[[TC.month_ts(fy, m, start) for m in range(k + 1, 13)]].sum())
            err = float(fc.sum()) - act
            rows.append({"fy": fy, "k": k, "err": err, "ratio": abs(err) / abs(year_total)})
    bu_err = pd.DataFrame(rows)
    bu = {
        "all": TC.mean_ratio(bu_err),
        "n": int(len(bu_err)),
        "by_fy": bu_err.groupby("fy")["ratio"].mean().to_dict() if not bu_err.empty else {},
    }
    edge = None
    if np.isfinite(base["all"]) and np.isfinite(bu["all"]) and base["all"] > 0:
        edge = (base["all"] - bu["all"]) / base["all"]
    ship = False  # never ship on n≪MIN_PRE
    return {
        "months": int(len(cash)),
        "baseline_smly": base,
        "bu_tax_exp_ngdp": bu,
        "edge": edge,
        "ship": ship,
        "reason": ("NO-SHIP: MoF monthly starts FY2024 (n origins too thin for PRE gate); "
                   f"BU edge vs smly = {edge if edge is not None else 'n/a'} on available FY"),
    }


def main() -> dict:
    es = screen_es()
    jp = screen_jp()
    out = {"es": es, "jp": jp}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    lines = [
        "# STATUS_ES / STATUS_JP — category×macro screen",
        "",
        "## Both years (shipped cash countries)",
        "",
        "UK / FR / DE / IT / NO (beyond NB): live bottom-up covers the **open budget year "
        "and later years** on one path (rules differ by year — e.g. DE Soll in the open "
        "year, smly×macro later; FR PLF pin when known). Interest ISIN+curve is annual×month "
        "shares on those same forward months.",
        "",
        "## Spain",
        "",
        f"**Verdict: {'SHIP' if es['ship'] else 'NO-SHIP'}** — {es['reason']}",
        "",
        f"Panel months: {es['panel_months']}. Identity MAE vs published B.9 cash: "
        f"{es['identity_mae_vs_pub_bn']:.4f} bn.",
        "",
        "| model | PRE | OOS | PRE edge |",
        "|---|---:|---:|---:|",
    ]
    for name, sc in es["results"].items():
        edge = sc.get("pre_edge")
        lines.append(
            f"| {name} | {sc.get('pre', float('nan')):.3f} | {sc.get('oos', float('nan')):.3f} | "
            f"{'' if edge is None else f'{edge:.1%}'} |"
        )
    lines += [
        "",
        "Baseline = thin_cash `seas_add_3`. Gate = PRE ≥5% better, OOS ≤+25%.",
        "",
        "## Japan",
        "",
        f"**Verdict: NO-SHIP** — {jp['reason']}",
        "",
        f"Months on disk: {jp['months']}. smly mean ratio={jp['baseline_smly']['all']}; "
        f"BU={jp['bu_tax_exp_ngdp']['all']}.",
        "",
        "Live path unchanged: ES `seas_add_3`, JP `smly` via `thin_cash.publish`. "
        "Debt interest remains ISIN+curve (`GOV_CURVE`).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(OUT_MD.read_text())
    return out


if __name__ == "__main__":
    main()
