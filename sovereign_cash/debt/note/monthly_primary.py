"""Monthly central primary / cash series for Spain (IGAE) and Japan (MoF).

Spain: IGAE Administración Central SEC-2010 monthly (YTD → monthly).
  B.9 = financing capacity/need; primary ≈ B.9 + D.41 interest.

Japan: MoF 国庫歳入歳出状況 general-account month files.
  Cash need = expenditure − (total revenue − 公債金 − 前年度剰余金受入), ¥bn.
  Balance (primary proxy) = −cash need (no clean GA interest carve in the sheet).

Caches under ``sovereign_cash/data/{es,jp}/``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
ES_DIR = REPO / "sovereign_cash" / "data" / "es" / "igae"
JP_DIR = REPO / "sovereign_cash" / "data" / "jp" / "mof_revexp"
ES_CACHE = REPO / "sovereign_cash" / "data" / "es" / "igae_ac_monthly.parquet"
JP_CACHE = REPO / "sovereign_cash" / "data" / "jp" / "mof_ga_monthly.parquet"

UA = {"User-Agent": "Mozilla/5.0"}
IGAE = "https://www.igae.pap.hacienda.gob.es"
MOF = "https://www.mof.go.jp/policy/budget/report/revenue_and_expenditure"


def _get(url: str) -> bytes:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.content


def _ytd_to_monthly(ytd: pd.Series) -> pd.Series:
    ytd = ytd.dropna().sort_index()
    parts = []
    for _, g in ytd.groupby(ytd.index.year):
        g = g.sort_index()
        m = g.diff()
        m.iloc[0] = g.iloc[0]
        parts.append(m)
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


# ── Spain ──────────────────────────────────────────────────────────────

def fetch_es_igae(years: range | list[int] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Monthly Administración Central B.9 / interest / primary, €bn."""
    ES_DIR.mkdir(parents=True, exist_ok=True)
    # 2015 is the first year the same Tabla1a file parses. seas_add_3 needs three
    # complete years before its first scored origin (2018).
    years = list(years or range(2015, 2027))
    rows = []
    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
              "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    for y in years:
        path = ES_DIR / f"M_AACC_{y}.xlsx"
        if refresh or not path.exists() or path.stat().st_size < 1000:
            url = (f"{IGAE}/sitios/igae/es-ES/Contabilidad/ContabilidadNacional/"
                   f"Publicaciones/Documents/AACC-M/M_AACC_{y}.xlsx")
            path.write_bytes(_get(url))
        df = pd.read_excel(path, sheet_name="Tabla1a", header=None)
        hdr = df.iloc[6].tolist()
        # concept column is 1; month cols start at 2
        b9 = interest = None
        for _, row in df.iterrows():
            concept = str(row[1]) if pd.notna(row[1]) else ""
            code = str(row[0]) if pd.notna(row[0]) else ""
            # TOTAL B.9 only — skip the DANA impact footnote row (also says capacidad/necesidad).
            if code.strip().startswith("B.9") and "TOTAL" in concept.upper():
                b9 = row
            # Prefer expenditure-side D.41 interest (row label "Intereses", not the * Intereses receipt)
            if code.strip() == "D.41" and concept.strip() == "Intereses":
                interest = row
        if b9 is None:
            continue
        for i, _mon in enumerate(months):
            col = 2 + i
            if col >= len(b9) or pd.isna(b9.iloc[col]):
                continue
            # units = millions of euros
            b9_bn = float(b9.iloc[col]) / 1000.0
            int_bn = (
                float(interest.iloc[col]) / 1000.0
                if interest is not None and pd.notna(interest.iloc[col]) else np.nan
            )
            rows.append({
                "month": pd.Timestamp(y, i + 1, 1),
                "b9_ytd_bn": b9_bn,
                "interest_ytd_bn": int_bn,
            })
    if not rows:
        raise RuntimeError("no ES IGAE rows")
    ytd = pd.DataFrame(rows).set_index("month").sort_index()
    b9_m = _ytd_to_monthly(ytd["b9_ytd_bn"])
    int_m = _ytd_to_monthly(ytd["interest_ytd_bn"])
    out = pd.DataFrame({
        "b9_bn": b9_m,
        "interest_bn": int_m,
        "primary_bn": b9_m + int_m,  # surplus(+)
        "cash_requirement_bn": -b9_m,  # deficit(+)
    })
    out.index.name = "month"
    ES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ES_CACHE)
    return out


def es_monthly(refresh: bool = False) -> pd.DataFrame:
    if ES_CACHE.exists() and not refresh:
        return pd.read_parquet(ES_CACHE)
    return fetch_es_igae(refresh=refresh)


# ── Japan ──────────────────────────────────────────────────────────────

def _jp_fy(year: int, month: int) -> int:
    """Japanese fiscal year label (Apr Y → Mar Y+1 uses Y)."""
    return year if month >= 4 else year - 1


def _jp_reiwa(year: int) -> int:
    return year - 2018


def _jp_urls_for(year: int, month: int) -> list[str]:
    fy = _jp_fy(year, month)
    rw = _jp_reiwa(year)
    name = f"{rw:02d}{month:02d}e_fy{fy}.xlsx"
    return [
        f"{MOF}/fy{fy}/{name}",
        f"{MOF}/fy{fy}/{rw:02d}{month:02d}_fy{fy}.xlsx",
    ]


def _num(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, str):
        x = x.strip().replace(",", "").replace("−", "-").replace("△", "-")
        if x in {"", "-", "－", "—"}:
            return 0.0
        return float(x)
    return float(x)


def _parse_jp_file(path: Path, month: pd.Timestamp) -> dict | None:
    try:
        rev = pd.read_excel(path, sheet_name="一般歳入", header=None)
        exp = pd.read_excel(path, sheet_name="一般歳出", header=None)
    except Exception:
        return None
    # Revenue: col4 = 本月分. Total row 合計; 公債金; 前年度剰余金受入
    total_rev = bond = carry = tax = None
    for _, row in rev.iterrows():
        c0 = str(row[0]) if pd.notna(row[0]) else ""
        c1 = str(row[1]) if pd.notna(row[1]) else ""
        c0s = c0.replace("　", "")
        if "合計" in c0s:
            total_rev = _num(row[4])
        if "租税及印紙" in c0s:
            tax = _num(row[4])
        if "公債金" in c1:
            bond = _num(row[4])
        if "前年度剰余金" in c1 or "前年度剰余金" in c0:
            carry = _num(row[4])
    total_exp = None
    for _, row in exp.iterrows():
        c0 = str(row[0]) if pd.notna(row[0]) else ""
        if "合計" in c0.replace("　", ""):
            total_exp = _num(row[2])  # 本月分
            break
    if total_rev is None or total_exp is None:
        return None
    bond = 0.0 if bond is None else bond
    carry = 0.0 if carry is None else carry
    # sheet unit = 千円 → ¥bn
    scale = 1e-6
    rev_ex = (total_rev - bond - carry) * scale
    exp_bn = total_exp * scale
    cash = exp_bn - rev_ex
    return {
        "month": month,
        "rev_ex_bond_bn": rev_ex,
        "exp_bn": exp_bn,
        "bond_bn": bond * scale,
        "cash_requirement_bn": cash,
        "tax_bn": (0.0 if tax is None else tax) * scale,
        "primary_bn": -cash,  # overall GA balance ex bond proceeds
    }


# MoF 決算 一般会計税収, 億円. The March monthly report books only part of it;
# the rest is attributed back when the July settlement is published.
JP_TAX_SETTLEMENT_OKU = {2024: 752_321, 2025: 842_226}


def jp_panel_from_disk() -> pd.DataFrame:
    """Reparse cached MoF workbooks (no network). Includes monthly tax."""
    import re
    rows = []
    for path in sorted(JP_DIR.glob("*.xlsx")):
        m = re.fullmatch(r"(\d{2})(\d{2})e_fy(\d{4})", path.stem)
        if not m:
            continue
        rw, mo = int(m.group(1)), int(m.group(2))
        stamp = pd.Timestamp(2018 + rw, mo, 1)
        parsed = _parse_jp_file(path, stamp)
        if parsed:
            rows.append(parsed)
    if not rows:
        raise RuntimeError("no cached JP MoF workbooks")
    out = pd.DataFrame(rows).set_index("month").sort_index()
    return out[~out.index.duplicated(keep="last")]


def jp_cash_tax_corrected() -> pd.Series:
    """GA cash after grossing monthly tax up to the settlement.

    cash falls by tax × (settlement / monthly sum − 1). A FY with no
    settlement yet uses the latest settled ratio.
    """
    frame = jp_panel_from_disk()
    cash = frame["cash_requirement_bn"].astype(float)
    tax = frame["tax_bn"].astype(float)

    def fy_of(ts: pd.Timestamp) -> int:
        return int(ts.year if ts.month >= 4 else ts.year - 1)

    fys = pd.Index([fy_of(ts) for ts in cash.index])
    ratios: dict[int, float] = {}
    for fy, oku in JP_TAX_SETTLEMENT_OKU.items():
        booked = float(tax[fys == fy].sum())
        if booked > 0:
            ratios[fy] = (oku * 0.1) / booked  # 億円 → ¥bn
    if not ratios:
        return cash
    live_ratio = ratios[max(ratios)]
    out = cash.copy()
    for ts, fy in zip(cash.index, fys):
        r = ratios.get(int(fy), live_ratio)
        out.loc[ts] = float(cash.loc[ts]) - float(tax.loc[ts]) * (r - 1.0)
    return out


def fetch_jp_mof(start: str = "2024-01", end: str | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """Pull MoF GA monthly files into a cash / primary panel (¥bn)."""
    JP_DIR.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m")
    months = pd.period_range(start, end, freq="M")
    rows = []
    for per in months:
        y, m = per.year, per.month
        stamp = pd.Timestamp(y, m, 1)
        # prefer already-cached matching file
        rw = _jp_reiwa(y)
        fy = _jp_fy(y, m)
        candidates = [
            JP_DIR / f"{rw:02d}{m:02d}e_fy{fy}.xlsx",
            JP_DIR / f"{rw:02d}{m:02d}_fy{fy}.xlsx",
        ]
        path = next((p for p in candidates if p.exists() and p.stat().st_size > 1000), None)
        if path is None or refresh:
            for url in _jp_urls_for(y, m):
                try:
                    blob = _get(url)
                except Exception:
                    continue
                path = JP_DIR / Path(url).name
                path.write_bytes(blob)
                break
        if path is None or not path.exists():
            continue
        parsed = _parse_jp_file(path, stamp)
        if parsed:
            rows.append(parsed)
    if not rows:
        raise RuntimeError("no JP MoF monthly rows")
    out = pd.DataFrame(rows).set_index("month").sort_index()
    JP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(JP_CACHE)
    return out


def jp_monthly(refresh: bool = False) -> pd.DataFrame:
    if JP_CACHE.exists() and not refresh:
        return pd.read_parquet(JP_CACHE)
    return fetch_jp_mof(refresh=refresh)


def primary_series(cc: str, refresh: bool = False) -> pd.Series:
    """Surplus(+) monthly primary / balance proxy, local bn."""
    if cc == "es":
        return es_monthly(refresh=refresh)["primary_bn"].dropna()
    if cc == "jp":
        return jp_monthly(refresh=refresh)["primary_bn"].dropna()
    raise KeyError(cc)


def cash_series(cc: str, refresh: bool = False) -> pd.Series:
    """Deficit(+) monthly cash requirement, local bn."""
    if cc == "es":
        return es_monthly(refresh=refresh)["cash_requirement_bn"].dropna()
    if cc == "jp":
        return jp_monthly(refresh=refresh)["cash_requirement_bn"].dropna()
    raise KeyError(cc)


if __name__ == "__main__":
    es = fetch_es_igae(refresh=True)
    print("ES", es.tail(3))
    print("ES LTM primary €bn", float(es["primary_bn"].tail(12).sum()))
    jp = fetch_jp_mof(start="2024-01", refresh=False)
    print("JP months", len(jp), "last", jp.index.max())
    print(jp.tail(3))
