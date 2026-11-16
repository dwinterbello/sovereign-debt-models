"""Does the flow demand score line up with yields?

Score and pillars from flow_score.parquet, oriented + = more demand, so every expected
correlation is negative (yields fall, spreads tighten, gilts richen to swaps).

Matched targets by sleeve (plan gate):
  score       → lvl10 / rel10
  score_int   → lvl10 (intermediate ASW / 2s10s when a clean series exists later)
  score_long  → UK asw20; FR/IT sprde (Bund ASW proxy); else s1030 / lvl10

Horizons:
  d3     3m change ending at t (descriptive)
  f3l2   3m change from t+2 to t+5 (first month the full score is known)

PRE ≤ 2023-12 and OOS after. A sleeve ships only if f3l2 clears |t|≥2 on its
matched primary target in the full sample, or beats a no-signal baseline on PRE.

  PYTHONPATH=. .venv/bin/python issuance/experiments/flow_score_test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from issuance.experiments.supply_metric_test import _fred, _monthly, targets
from issuance.paths import CACHE

SIGNALS = (
    "score", "score_int", "score_long", "score_xpos",
    "auction", "auction_int", "auction_long",
    "foreign", "foreign_int", "foreign_long",
    "official", "positioning",
)

# Primary matched price per score sleeve; secondary used when primary missing for a country.
MATCHED = {
    "score": ("lvl10", "rel10"),
    "score_int": ("lvl10", "rel10"),
    "score_long": ("asw20", "sprde", "s1030", "lvl10"),
}
PRE_END = pd.Timestamp("2023-12-01")
T_GATE = 2.0


def signals() -> pd.DataFrame:
    d = pd.read_parquet(CACHE / "flow_score.parquet")
    d["month"] = pd.to_datetime(d["month"])
    s = d[d["component"].isin(["_score", "_pillar"])]
    w = s.pivot_table(index=["country", "month"], columns="pillar", values="z").reset_index()
    have = [c for c in ("auction", "foreign", "official") if c in w.columns]
    if have:
        w["score_xpos"] = w[have].mean(axis=1)
    return w


def target_panel() -> pd.DataFrame:
    t = targets()
    jp = _monthly(_fred("IRLTLT01JPM156N"))
    t = pd.concat([t, pd.DataFrame({"country": "JP", "target": "lvl10", "month": jp.index, "y": jp.to_numpy()})])
    lvl = t[t["target"] == "lvl10"].pivot_table(index="month", columns="country", values="y").dropna(how="all")
    rel = lvl.sub(lvl.mean(axis=1), axis=0).stack().rename("y").reset_index()
    rel["target"] = "rel10"
    t = pd.concat([t, rel[["country", "target", "month", "y"]]], ignore_index=True)
    t = t.sort_values(["country", "target", "month"])
    g = t.groupby(["country", "target"])["y"]
    t["d3"] = g.diff(3)
    t["f3l2"] = g.shift(-5) - g.shift(-2)
    return t


def _pooled(sub: pd.DataFrame, x: str, h: str) -> dict:
    sub = sub[["country", "month", x, h]].dropna()
    if sub["country"].nunique() < 1 or len(sub) < 30:
        return {}
    dm = sub.groupby("country")[[x, h]].transform(lambda v: v - v.mean())
    r = float(np.corrcoef(dm[x], dm[h])[0, 1])
    per = {c: round(float(np.corrcoef(g[x], g[h])[0, 1]), 2)
           for c, g in sub.groupby("country") if len(g) > 24}
    out = {"r": round(r, 3), "neg": f"{sum(v < 0 for v in per.values())}/{len(per)}", "n": len(sub)}
    if h == "f3l2":
        q = sub[sub["month"].dt.month % 3 == 0]
        dq = q.groupby("country")[[x, h]].transform(lambda v: v - v.mean())
        if len(q) > 20:
            fit = sm.OLS(dq[h], sm.add_constant(dq[x])).fit(
                cov_type="cluster", cov_kwds={"groups": q["month"].factorize()[0]})
            out["beta_bp"] = round(float(fit.params[x]) * 100, 1)
            out["t"] = round(float(fit.tvalues[x]), 2)
    out["per"] = per
    return out


def _matched_frame(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """One row per country×month on the first available matched target for that country."""
    prefs = MATCHED.get(signal, ("lvl10",))
    parts = []
    for c, g in df.groupby("country"):
        picked = None
        for tgt in prefs:
            # Country-specific: asw20 is UK-only; sprde is FR/IT; s1030 US/DE/UK.
            if tgt == "asw20" and c != "UK":
                continue
            if tgt == "sprde" and c not in ("FR", "IT"):
                continue
            if tgt == "s1030" and c not in ("US", "DE", "UK"):
                continue
            sub = g[g["target"] == tgt]
            if not sub.empty and sub[signal].notna().any():
                picked = sub.copy()
                picked["matched"] = tgt
                break
        if picked is not None:
            parts.append(picked)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    df = signals().merge(target_panel(), on=["country", "month"])
    df = df[df["month"] >= start]
    if end is not None:
        df = df[df["month"] <= end]
    rows = []
    for tgt, sub in df.groupby("target"):
        for x in SIGNALS:
            if x not in sub:
                continue
            for h in ("d3", "f3l2"):
                got = _pooled(sub, x, h)
                if got:
                    rows.append({"target": tgt, "signal": x, "h": h, "scope": "target", **got})
    # Matched-price gate rows for the three scores.
    for x in ("score", "score_int", "score_long"):
        if x not in df.columns:
            continue
        m = _matched_frame(df, x)
        if m.empty:
            continue
        for h in ("d3", "f3l2"):
            got = _pooled(m, x, h)
            if got:
                rows.append({"target": "matched", "signal": x, "h": h, "scope": "matched", **got})
    return pd.DataFrame(rows)


def gate(res: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ship a sleeve score only if matched f3l2 |t|≥2, or PRE matched |t| beats 0 with right sign."""
    full = run() if res is None else res
    pre = run(end=PRE_END.strftime("%Y-%m-%d"))
    rows = []
    for sig in ("score", "score_int", "score_long"):
        f = full[(full["signal"] == sig) & (full["target"] == "matched") & (full["h"] == "f3l2")]
        p = pre[(pre["signal"] == sig) & (pre["target"] == "matched") & (pre["h"] == "f3l2")]
        ft = float(f["t"].iloc[0]) if len(f) and "t" in f.columns and pd.notna(f["t"].iloc[0]) else np.nan
        pt = float(p["t"].iloc[0]) if len(p) and "t" in p.columns and pd.notna(p["t"].iloc[0]) else np.nan
        # Demand ↑ → yields/spreads ↓, so t should be negative.
        ship = bool((np.isfinite(ft) and abs(ft) >= T_GATE and ft < 0)
                    or (np.isfinite(pt) and pt < -1.0))
        rows.append({
            "signal": sig,
            "t_full": None if not np.isfinite(ft) else round(ft, 2),
            "t_pre": None if not np.isfinite(pt) else round(pt, 2),
            "r_full": None if f.empty else float(f["r"].iloc[0]),
            "ship": ship,
            "reason": ("|t|≥2 right sign" if np.isfinite(ft) and abs(ft) >= T_GATE and ft < 0
                       else ("PRE t<-1" if np.isfinite(pt) and pt < -1.0
                             else "fail")),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", 200)
    res = run()
    print(res.drop(columns=["per"], errors="ignore").to_string(index=False))
    print("\nmatched sleeve gate (PRE ≤ 2023-12)")
    print(gate(res).to_string(index=False))
    print("\nsplit: 2010-2018 vs 2019+ (lvl10, rel10)")
    for lo, hi in (("2010-01-01", "2018-12-01"), ("2019-01-01", "2026-12-01")):
        df = signals().merge(target_panel(), on=["country", "month"])
        df = df[(df["month"] >= lo) & (df["month"] <= hi)]
        r = pd.DataFrame([{"target": t, "signal": x, "h": h, **_pooled(s, x, h)}
                          for t, s in df[df["target"].isin(["lvl10", "rel10"])].groupby("target")
                          for x in SIGNALS if x in s for h in ("d3", "f3l2")])
        print(lo[:4], "-", hi[:4])
        print(r.drop(columns=["per"], errors="ignore").to_string(index=False))
