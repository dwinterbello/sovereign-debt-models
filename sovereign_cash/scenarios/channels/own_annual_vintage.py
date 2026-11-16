"""Share-engine MAE under point-in-time WEO macro vs realised macro.

Same persist/revert rules and origins as ``own_annual.backtest``, but the GDP/price
path at origin ``o`` is the latest IMF WEO vintage dated on or before 31 Dec of ``o``
(CPI + real growth → nominal GDP growth ≈ real + CPI). Realised-macro MAE is the
incumbent number; the vintage-macro MAE is the forecast-skill companion.

Writes ``derived/own_annual_vintage_score.json``.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.scenarios.channels.own_annual_vintage
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels import own_annual as OA

SC = OA.SC
DEBT = OA.DEBT
DERIVED = OA.SC / "derived"
VINT = DEBT / "weo_vintages" / "weo_vintages.parquet"
OUT = DERIVED / "own_annual_vintage_score.json"
CCS = OA.COUNTRIES


def _vintages() -> pd.DataFrame:
    if not VINT.exists():
        raise FileNotFoundError(VINT)
    v = pd.read_parquet(VINT)
    v["vintage_date"] = pd.to_datetime(v["vintage_date"])
    return v


def _macro_at(cc: str, origin: int, vint: pd.DataFrame, years: list[int]) -> tuple[pd.Series, pd.Series] | None:
    """Nominal GDP index (origin=100) and price index from WEO vintage ≤ origin year-end."""
    cut = pd.Timestamp(origin, 12, 31)
    sub = vint[(vint["country"] == cc) & (vint["vintage_date"] <= cut)]
    if sub.empty:
        return None
    vd = sub["vintage_date"].max()
    one = sub[sub["vintage_date"] == vd]
    cpi = one[one["var"] == "cpi"].set_index("year")["value"].astype(float)
    real = one[one["var"] == "real_growth"].set_index("year")["value"].astype(float)
    # build levels from origin backwards/forwards using growth rates
    Y = pd.Series(dtype=float)
    P = pd.Series(dtype=float)
    if origin not in cpi.index or origin not in real.index:
        return None
    Y.loc[origin] = 100.0
    P.loc[origin] = 100.0
    for y in range(origin + 1, max(years) + 1):
        if y not in cpi.index or y not in real.index:
            break
        g = float(real[y]) / 100.0 + float(cpi[y]) / 100.0
        Y.loc[y] = Y.loc[y - 1] * (1 + g)
        P.loc[y] = P.loc[y - 1] * (1 + float(cpi[y]) / 100.0)
    for y in range(origin - 1, min(years) - 1, -1):
        if y + 1 not in cpi.index or y + 1 not in real.index:
            break
        g = float(real[y + 1]) / 100.0 + float(cpi[y + 1]) / 100.0
        if abs(1 + g) < 1e-9:
            break
        Y.loc[y] = Y.loc[y + 1] / (1 + g)
        P.loc[y] = P.loc[y + 1] / (1 + float(cpi[y + 1]) / 100.0)
    return Y.sort_index(), P.sort_index()


def score_cc(cc: str, vint: pd.DataFrame) -> dict:
    h = OA.history(cc)
    yrs = list(h.index)
    Y_r, P_r = OA.levels(cc, yrs)
    rows = []
    for o in [y for y in yrs if OA.FIRST_ORIGIN <= y < yrs[-1] and y not in OA.COVID]:
        need = [o + k for k in (1, 2, 3) if o + k in h.index]
        if not need:
            continue
        mac = _macro_at(cc, o, vint, [o] + need)
        if mac is None:
            continue
        Y_v, P_v = mac
        # Index was built at origin=100; scale onto realised levels so shares × X stay in local bn.
        if o not in Y_r.index or o not in P_r.index or o not in Y_v.index or o not in P_v.index:
            continue
        Y_v = Y_v / float(Y_v.loc[o]) * float(Y_r.loc[o])
        P_v = P_v / float(P_v.loc[o]) * float(P_r.loc[o])
        for line, X_r, X_v, m in (

                ("receipts", Y_r, Y_v, OA.method(cc, "receipts")),
                ("primary", P_r if OA.method(cc, "primary") != "persist_gdp" else Y_r,
                 P_v if OA.method(cc, "primary") != "persist_gdp" else Y_v,
                 OA.method(cc, "primary")),
        ):
            if m == "fiscal_drag":
                continue  # UK drag needs earnings vintages we do not have here
            share = (h[line] / X_r.reindex(h.index)).dropna()
            if o not in share.index:
                continue
            for kind, X in (("realised", X_r), ("vintage", X_v)):
                if any(y not in X.index for y in need):
                    continue
                mm = "persist" if m == "persist_gdp" else m
                fc = OA.carry(share.loc[:o], o, len(need), mm)
                for s, y in zip(fc, need):
                    if y not in h.index or y not in X.index or not np.isfinite(X[y]):
                        continue
                    hat = float(s) * float(X[y])
                    act = float(h.at[y, line])
                    rows.append({"origin": o, "year": y, "h": y - o, "line": line,
                                 "kind": kind, "err": hat - act})
    if not rows:
        return {"cc": cc, "error": "no vintage rows"}
    df = pd.DataFrame(rows)
    out: dict = {"cc": cc, "method": {ln: OA.method(cc, ln) for ln in ("receipts", "primary")}}
    for kind in ("realised", "vintage"):
        sub = df[df["kind"] == kind]
        out[kind] = {
            "mae": round(float(sub["err"].abs().mean()), 3),
            "n": int(len(sub)),
            "by_h": {int(h): round(float(g["err"].abs().mean()), 3)
                     for h, g in sub.groupby("h")},
        }
    if out.get("realised") and out.get("vintage"):
        r, v = out["realised"]["mae"], out["vintage"]["mae"]
        out["vintage_over_realised"] = round(v / r, 3) if r else None
        out["note"] = (
            f"Vintage-macro MAE is {out['vintage_over_realised']}× realised-macro MAE on the same "
            f"share rules — the page's headline share MAE is conditional on perfect macro."
        )
    return out


def build(countries=CCS) -> dict:
    vint = _vintages()
    res = {"built_at": pd.Timestamp.utcnow().isoformat() + "Z", "countries": {}}
    for cc in countries:
        try:
            res["countries"][cc] = score_cc(cc, vint)
        except Exception as exc:  # noqa: BLE001
            res["countries"][cc] = {"cc": cc, "error": f"{type(exc).__name__}: {exc}"}
    ratios = [c["vintage_over_realised"] for c in res["countries"].values()
              if c.get("vintage_over_realised")]
    res["pool_vintage_over_realised"] = round(float(np.mean(ratios)), 3) if ratios else None
    res["verdict"] = (
        f"Across {len(ratios)} countries, vintage-WEO macro raises share-line MAE to "
        f"{res['pool_vintage_over_realised']}× the realised-macro figure on average. "
        f"Ship rules stay on realised-macro (stated on the Methodology page); this file is the "
        f"forecast-skill companion."
    )
    DERIVED.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = build()
    print(r["verdict"])
    for cc, c in r["countries"].items():
        print(cc, c.get("error") or c.get("note"), c.get("realised"), c.get("vintage"))
