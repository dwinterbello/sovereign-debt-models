"""Curve shapes for non-parallel shocks, from the covariance of monthly zero-curve changes.

Two sliders set the 3-month and 10-year shifts. Every other maturity moves by its
expected change given those two, from the same covariance matrix the principal
components come from (all factors kept at their conditional means):

    s(t) = Sigma[t, a] Sigma[a, a]^-1 [x3m, x10y]',   a = {3m, 10y}

Pinning only the first two components instead (curvature forced to zero) bends the
curve away from the anchors: +100bp at both gave +137bp at 2-3 years in the US and
a 3m-only move sent the Japanese 30-year down 142bp, so it is reported, not used.

Loadings come from each market's own history where the repo has a full curve
(US: Fed GSW Svensson; UK: Bank of England; DE: Bundesbank; JP: MoF), and from
the nearest full curve otherwise (euro area and Nordics use Germany's).

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.scenarios.curve_pca   # estimate and cache
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
SC = Path(__file__).resolve().parents[1]
CACHE = SC / "data" / "debt" / "curve_factors.json"
GRID = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], float)
START = "2000-01-01"
SOURCE_FOR = {"us": "us", "uk": "uk", "de": "de", "jp": "jp",
              "fr": "de", "it": "de", "es": "de", "se": "de", "no": "de"}


def _on_grid(mats: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Linear in maturity, flat beyond the ends (the short end of DE/UK is flat below its first point)."""
    ok = np.isfinite(vals)
    return np.interp(GRID, mats[ok], vals[ok])


def _svensson(p: pd.Series, t: np.ndarray) -> np.ndarray:
    b0, b1, b2, b3, t1, t2 = (float(p[k]) for k in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"))
    x1, x2 = t / t1, t / t2
    f1 = (1 - np.exp(-x1)) / x1
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))


def history_us() -> pd.DataFrame:
    from sovereign_cash.debt.gov_curve import GSW, _get
    raw = _get(GSW, timeout=180).decode("utf-8", "replace").splitlines()
    start = next(i for i, ln in enumerate(raw) if ln.startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(raw[start:])), low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    keys = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    df[keys] = df[keys].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Date", *keys]).set_index("Date").loc[START:]
    m = df[keys].resample("ME").last().dropna()
    return pd.DataFrame([_svensson(r, GRID) for _, r in m.iterrows()], index=m.index, columns=GRID)


def history_uk() -> pd.DataFrame:
    from sovereign_cash.debt.gov_curve import BOE
    c = pd.read_parquet(BOE, columns=["date", "maturity_years", "rate", "curve", "measure", "segment"])
    c = c[(c["curve"] == "nominal") & (c["measure"] == "spot") & (c["segment"] == "full")]
    c = c[c["date"] >= START]
    wide = c.pivot_table(index="date", columns="maturity_years", values="rate").resample("ME").last()
    mats = wide.columns.to_numpy(float)
    return pd.DataFrame([_on_grid(mats, r.to_numpy(float)) for _, r in wide.iterrows()],
                        index=wide.index, columns=GRID).dropna()


def history_de() -> pd.DataFrame:
    from correlations.g4.sources import parse_bundesbank
    from sovereign_cash.debt.gov_curve import BBK, _get
    stem = "+".join(f"R{t:02d}XX" for t in range(1, 31))
    wide = parse_bundesbank(_get(BBK.format(stem=stem) + f"&startPeriod={START}", timeout=300).decode("utf-8", "replace"))
    cols = {}
    for col in wide.columns:
        hit = [p for p in str(col).split(".") if p.startswith("R") and p[1:3].isdigit()]
        if hit:
            cols[col] = int(hit[0][1:3])
    wide = wide[list(cols)].rename(columns=cols).sort_index(axis=1)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.resample("ME").last().dropna(how="all")
    mats = wide.columns.to_numpy(float)
    return pd.DataFrame([_on_grid(mats, r.to_numpy(float)) for _, r in wide.iterrows()],
                        index=wide.index, columns=GRID).dropna()


MOF_ALL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"


def history_jp() -> pd.DataFrame:
    from correlations.g4.sources import parse_mof
    from sovereign_cash.debt.gov_curve import _get
    wide = parse_mof(_get(MOF_ALL, timeout=180).decode("utf-8", "replace"))
    wide.index = pd.to_datetime(wide.index)
    wide = wide.loc[START:].resample("ME").last().dropna(how="all")
    mats = np.array([int(str(c).split("_")[1]) for c in wide.columns], float)
    order = np.argsort(mats)
    wide = wide.iloc[:, order]
    mats = mats[order]
    # par yields; at these levels zero minus par is a few bp and does not move the change factors
    return pd.DataFrame([_on_grid(mats, r.to_numpy(float)) for _, r in wide.iterrows()],
                        index=wide.index, columns=GRID).dropna()


HISTORY = {"us": history_us, "uk": history_uk, "de": history_de, "jp": history_jp}


def estimate(levels: pd.DataFrame) -> dict:
    """PCA of monthly changes (bp). Signs: level positive on average, slope rising with maturity."""
    d = levels.diff().dropna() * 100.0
    cov = np.cov(d.to_numpy().T)
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    L1, L2, L3 = vec[:, 0], vec[:, 1], vec[:, 2]
    if L1.mean() < 0:
        L1 = -L1
    if L2[-1] - L2[0] < 0:
        L2 = -L2
    if L3[len(L3) // 2] < 0:
        L3 = -L3
    share = val / val.sum()
    return {"grid": GRID.tolist(), "L1": L1.tolist(), "L2": L2.tolist(), "L3": L3.tolist(), "cov": cov.tolist(),
            "explained": share[:3].tolist(), "n_months": int(len(d)),
            "sample": [str(d.index.min().date()), str(d.index.max().date())],
            "sd_bp": np.sqrt(np.diag(cov)).tolist()}


def build_cache() -> dict:
    out = {"source_for": SOURCE_FOR, "markets": {}}
    for mkt, fn in HISTORY.items():
        try:
            out["markets"][mkt] = estimate(fn())
            log.info("curve factors %s: explained %s", mkt, out["markets"][mkt]["explained"])
        except Exception as exc:  # noqa: BLE001 - one market's fetch must not stop the others
            log.warning("curve factors %s failed: %s", mkt, exc)
            out["markets"][mkt] = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" in out["markets"].get("jp", {}):
        out["source_for"] = {**SOURCE_FOR, "jp": "us"}
    CACHE.write_text(json.dumps(out, indent=1))
    return out


def load() -> dict:
    return json.loads(CACHE.read_text())


ANCHORS = (0.25, 10.0)


def weights(factors: dict) -> np.ndarray:
    """(len(grid), 2): expected change at each maturity per bp at the 3m and 10y anchors."""
    g = np.asarray(factors["grid"])
    cov = np.asarray(factors["cov"])
    ia = [int(np.argmin(np.abs(g - a))) for a in ANCHORS]
    saa = cov[np.ix_(ia, ia)]
    if abs(np.linalg.det(saa)) < 1e-9 * np.trace(saa) ** 2:
        raise ValueError("3m and 10y changes are collinear in this history")
    return cov[:, ia] @ np.linalg.inv(saa)


def shape(factors: dict, x3m: float, x10y: float):
    """The curve shift in bp as a function of maturity (years), flat beyond the grid."""
    g = np.asarray(factors["grid"])
    s = weights(factors) @ np.array([x3m, x10y], float)

    def f(t):
        return np.interp(np.asarray(t, float), g, s)

    f.knots = s.tolist()  # type: ignore[attr-defined]
    return f


def shape_two_pc(factors: dict, x3m: float, x10y: float) -> np.ndarray:
    """The rejected alternative (level + slope only), kept for the Methodology page comparison."""
    g = np.asarray(factors["grid"])
    L1, L2 = np.asarray(factors["L1"]), np.asarray(factors["L2"])
    M = np.array([[np.interp(a, g, L1), np.interp(a, g, L2)] for a in ANCHORS])
    a, b = np.linalg.solve(M, np.array([x3m, x10y], float))
    return a * L1 + b * L2


def factors_for(cc: str, cache: dict | None = None) -> tuple[dict, str]:
    cache = cache or load()
    mkt = cache["source_for"].get(cc, "de")
    return cache["markets"][mkt], mkt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    c = build_cache()
    for m, f in c["markets"].items():
        if "error" in f:
            print(m, "ERROR", f["error"])
            continue
        print(m, f["sample"], "n", f["n_months"], "explained", [round(x, 3) for x in f["explained"]])
        print("   L1", [round(x, 2) for x in f["L1"]])
        print("   L2", [round(x, 2) for x in f["L2"]])
        for x3, x10 in ((100, 100), (100, 0), (0, 100)):
            sh = shape(f, x3, x10)
            print(f"   3m {x3:+} 10y {x10:+}: conditional", [round(float(v)) for v in sh(GRID)])
            print(f"   {'':17}two-PC     ", [round(float(v)) for v in shape_two_pc(f, x3, x10)])
