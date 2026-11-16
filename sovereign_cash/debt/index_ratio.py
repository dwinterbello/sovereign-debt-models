"""Linker index ratios from the issuer's coefficient files.

Germany: Finanzagentur daily indexation coefficients (HICP ex tobacco).
Italy: MEF BTP€i coefficients (cumulative) and BTP Italia coefficients (reset each coupon;
the file's ratio is the indexed principal today). The BTP Italia file names the cum ISIN;
the ex ISIN in the auction tape shares the maturity.
Japan: MoF indexation coefficient by issue number, mapped to ``JPIL10y#n``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "data" / "debt" / "index_coef"
_IT_MONTH = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _on(series: pd.Series, asof: pd.Timestamp, fallback: bool = False) -> float | None:
    s = series.dropna().sort_index()
    prior = s[s.index <= asof]
    if not prior.empty:
        return float(prior.iloc[-1])
    if fallback and not s.empty:
        return float(s.iloc[0])
    return None


def _de(asof: pd.Timestamp, fallback: bool = False) -> dict[str, float]:
    files = sorted(HERE.glob("de_bj*.xlsx"))
    if not files:
        return {}
    x = pd.read_excel(files[-1], header=None)
    dates = pd.to_datetime(x[1], errors="coerce", format="mixed")
    out = {}
    for col in range(2, x.shape[1]):
        isin = re.search(r"DE\d{10}", str(x.iloc[9, col]))
        if not isin:
            continue
        vals = pd.to_numeric(x[col], errors="coerce")
        vals.index = dates
        v = _on(vals, asof, fallback)
        if v is not None:
            out[isin.group(0)] = v
    return out


def _jp(asof: pd.Timestamp, fallback: bool = False) -> dict[str, float]:
    files = sorted(HERE.glob("jp_keisuu_*.xls"))
    if not files:
        return {}
    x = pd.read_excel(files[-1], header=None)
    dates = pd.to_datetime(x[0], errors="coerce", format="mixed")
    out = {}
    for col in range(2, x.shape[1]):
        n = re.search(r"Issue No\.(\d+)", str(x.iloc[3, col]))
        if not n:
            continue
        vals = pd.to_numeric(x[col], errors="coerce")
        vals.index = dates
        v = _on(vals, asof, fallback)
        if v is not None:
            out[f"JPIL10y#{int(n.group(1))}"] = v
    return out


def _it_file(path: Path, asof: pd.Timestamp, fallback: bool = False) -> tuple[str | None, pd.Timestamp | None, float | None]:
    text = path.read_text(encoding="latin-1", errors="replace")
    isin = re.search(r"(IT\d{10})", text)
    mat = re.search(r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})", text)
    rows = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 3 or not parts[1]:
            continue
        d = pd.to_datetime(parts[1], format="%d-%b-%y", errors="coerce")
        if pd.isna(d):
            m = re.match(r"(\d{2})-([a-z]{3})-(\d{2})", parts[1], re.I)
            if not m:
                continue
            mon = _IT_MONTH.get(m.group(2).lower())
            if not mon:
                continue
            d = pd.Timestamp(2000 + int(m.group(3)), mon, int(m.group(1)))
        cell = parts[2]
        raw = cell.replace(",", ".") if "," in cell and "." not in cell else cell.replace(",", "")
        v = pd.to_numeric(raw, errors="coerce")
        if pd.notna(d) and pd.notna(v):
            rows[pd.Timestamp(d)] = float(v)
    coef = _on(pd.Series(rows), asof, fallback) if rows else None
    maturity = pd.to_datetime(mat.group(2), dayfirst=True) if mat else None
    return (isin.group(1) if isin else None), maturity, coef


def _it(asof: pd.Timestamp, fallback: bool = False) -> dict[str, float]:
    out = {}
    by_mat: dict[pd.Timestamp, float] = {}
    for folder in (HERE / "it", HERE / "it_italia"):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.csv"):
            isin, mat, coef = _it_file(path, asof, fallback)
            if coef is None:
                continue
            if isin:
                out[isin] = coef
            if mat is not None:
                by_mat[pd.Timestamp(mat).normalize()] = coef
    out["_by_maturity"] = by_mat  # type: ignore[assignment]
    return out


def ratios(asof: pd.Timestamp | None = None, fallback: bool = False) -> dict[str, float]:
    """Index ratio by the auction-tape key, on ``asof``.

    ``fallback`` uses the first published day when the file starts after ``asof``
    (the Japan and BTP€i files on disk are the current month).
    """
    asof = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
    out = {}
    out.update(_de(asof, fallback))
    out.update(_jp(asof, fallback))
    it = _it(asof, fallback)
    by_mat = it.pop("_by_maturity", {})
    out.update({k: v for k, v in it.items() if not str(k).startswith("_")})
    # BTP Italia files name the cum ISIN. The tape's ex line has the same maturity.
    try:
        from sovereign_cash.debt.isin_book import outstanding
        bk = outstanding(asof)
        it_link = bk[(bk["country"] == "IT") & (bk["sec_type"] == "Linker")]
        for r in it_link.itertuples(index=False):
            key = str(r.key)
            if key in out:
                continue
            mat = pd.Timestamp(r.maturity_dt).normalize()
            if mat in by_mat:
                out[key] = by_mat[mat]
    except Exception:
        pass
    return out


def country_ratios(book: pd.DataFrame, asof: pd.Timestamp, fallback: bool = False) -> dict[str, float]:
    """Face-weighted index ratio for linkers that have a coefficient. Others stay at 1."""
    got = ratios(asof, fallback)
    out = {}
    link = book[book["sec_type"] == "Linker"].copy()
    link["cc"] = link["country"].str.lower()
    for cc, g in link.groupby("cc"):
        w = g["face_mn"].astype(float)
        r = g["key"].astype(str).map(got).fillna(1.0).astype(float)
        if float(w.sum()) <= 0:
            continue
        out[str(cc)] = float((r * w).sum() / w.sum())
    return out


def boj_tbill_bn(path: Path | None = None) -> float:
    """BoJ outright T-bill holdings, yen billion. The tmei file is face purchased, 100 million yen."""
    files = sorted((path.parent if path else HERE).glob("boj_tmei_*.xlsx")) if path is None else [path]
    if not files:
        return float("nan")
    x = pd.read_excel(files[-1], header=None)
    amt = pd.to_numeric(x[4], errors="coerce")
    # header row contains the label; data are the numeric issue rows
    return float(amt.sum() * 0.1)
