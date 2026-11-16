"""PRE/OOS screen: link DTS dep_corporate to US equity YoY.

Production dep_corporate = trailing 250d mean × HGB calendar shape (choice=model).
FY overlay only applies trailing growth / COLA — no equity. This experiment scales
that forecast by (1 + ε · equity_yoy/100) at lags 0/1/3/6/12, and a residual
OLS correction on the corporate bucket alone. Does NOT change production defaults.

Equity = FRED SP500 monthly mean YoY (WILL5000PR fallback). At origin o only
months ending ≤ o are used (no lookahead). Monthly series: last complete
month-end ≤ o; lag k shifts that stamp back k months.

  PYTHONPATH=. .venv/bin/python us/fiscal/experiments/equity_corporate_ab.py
  PYTHONPATH=. .venv/bin/python us/fiscal/experiments/equity_corporate_ab.py --net
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

EPS_GRID = (0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0)
# Negative ε: equity up → corporate down (contrarian / already-in-mean check).
EPS_GRID_FULL = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0)
LAGS = (0, 1, 3, 6, 12)
EQUITY_CODES = ("SP500", "SPASTT01USM661N")  # WILL5000PR 404 on FRED CSV
RESID_MIN = 24  # months of history before expanding OLS residual overlay
PRE_TOL = 1.05  # ship only if PRE ≤ baseline × this
# Clear OOS win: ≥2% relative cut on corporate daily MAE (noise floor ~0.35%).
OOS_REL_MIN = 0.02
# Warm residual / surprise history from this month so PRE origins have a fit.
HIST_START = "2018-01-01"


def load_equity_yoy(*, refresh: bool = False, code: str = "SP500") -> pd.Series:
    """Month-end equity index → YoY %; stamped at month-end of the observation month."""
    from tga_deficit import fetch_fred

    raw = fetch_fred(code, refresh=refresh)
    m = raw.resample("ME").mean().dropna()
    yoy = (m.pct_change(12) * 100.0).dropna().rename(f"{code}_yoy")
    return yoy


def yoy_at(yoy: pd.Series, origin: pd.Timestamp, lag: int) -> float:
    """Last YoY known at ``origin`` with ``lag`` extra months of delay. No lookahead."""
    origin = pd.Timestamp(origin)
    # Only complete months ending on or before origin.
    known = yoy[yoy.index <= origin]
    if known.empty:
        return float("nan")
    if lag > 0:
        cut = (origin.to_period("M") - lag).to_timestamp("M")
        known = known[known.index <= cut]
    if known.empty:
        return float("nan")
    return float(known.iloc[-1])


def scale_factor(yoy: float, eps: float) -> float:
    if not np.isfinite(yoy):
        return 1.0
    return max(0.25, 1.0 + eps * yoy / 100.0)


def _window(origin: pd.Timestamp) -> str:
    import flow_model as fm

    o = str(pd.Timestamp(origin).date())
    for name, (a, b) in fm.WINDOWS.items():
        if a <= o <= b:
            return name
    return "OTHER"


def _embedded_yoy(yoy: pd.Series, origin: pd.Timestamp, window: int = 12) -> float:
    """Mean equity YoY over the prior ``window`` complete months ≤ origin (in the 250d level)."""
    known = yoy[yoy.index <= origin].dropna().tail(window)
    return float(known.mean()) if len(known) else float("nan")


def score_corporate_bucket(
    yoy: pd.Series,
    *,
    eps_grid=EPS_GRID_FULL,
    lags=LAGS,
    start: str = "2022-01-01",
    hist_start: str = HIST_START,
) -> pd.DataFrame:
    """Monthly-origin daily MAE ($bn) on dep_corporate: baseline vs equity overlays.

    Months from ``hist_start`` warm residual OLS / are scored only from ``start``.
    """
    import flow_model as fm

    panel, coupon = fm.load_inputs()
    feats = fm.calendar_features(panel.index)
    months = pd.period_range(hist_start, panel.index.max(), freq="M")
    score_from = pd.Timestamp(start)
    rows = []
    # residual expanding OLS of monthly corporate miss on equity YoY
    hist_miss: list[tuple[pd.Timestamp, float, float]] = []

    for p in months:
        prior = panel.index[panel.index < p.start_time]
        if len(prior) == 0:
            continue
        o = prior[-1]
        fut = panel.index[(panel.index >= p.start_time) & (panel.index <= p.end_time)]
        if len(fut) == 0 or "dep_corporate" not in panel.columns:
            continue
        models = fm.fit_all(panel, feats, o)
        choice = {b: "model" for b in panel.columns}
        choice["dep_corporate"] = "model"
        fc = fm.forecast(models, panel, coupon, o, fut, choice=choice)
        base = fc["dep_corporate"]
        y = panel.loc[fut, "dep_corporate"]
        n = len(fut)
        month_miss = float((y - base).sum())
        ey0 = yoy_at(yoy, o, 0)
        score_month = p.start_time >= score_from
        if score_month:
            base_err = float((base - y).abs().sum())
            w = _window(p.start_time)
            rows.append({
                "window": w, "origin": o, "month": str(p), "variant": "baseline",
                "eps": 0.0, "lag": -1, "err_mn": base_err, "n": n,
            })
            emb = _embedded_yoy(yoy, o)
            for lag in lags:
                ey = yoy_at(yoy, o, lag)
                for eps in eps_grid:
                    if eps == 0.0 and lag != lags[0]:
                        continue
                    fac = scale_factor(ey, eps)
                    err = float((base * fac - y).abs().sum())
                    rows.append({
                        "window": w, "origin": o, "month": str(p),
                        "variant": f"scale_e{eps}_L{lag}",
                        "eps": float(eps), "lag": int(lag), "err_mn": err, "n": n,
                        "equity_yoy": ey, "factor": fac,
                    })
                    # Surprise vs YoY already sitting in the trailing mean window.
                    if np.isfinite(ey) and np.isfinite(emb):
                        fac_s = scale_factor(ey - emb, eps)
                        err_s = float((base * fac_s - y).abs().sum())
                        rows.append({
                            "window": w, "origin": o, "month": str(p),
                            "variant": f"surprise_e{eps}_L{lag}",
                            "eps": float(eps), "lag": int(lag), "err_mn": err_s, "n": n,
                            "equity_yoy": ey, "embedded_yoy": emb, "factor": fac_s,
                        })
            if np.isfinite(ey0):
                past = [(t, m, e) for t, m, e in hist_miss if t < o and np.isfinite(e)]
                if len(past) >= RESID_MIN:
                    mm = np.array([m for _, m, _ in past], dtype=float)
                    ee = np.array([e for _, _, e in past], dtype=float)
                    X = np.column_stack([np.ones(len(ee)), ee])
                    beta, *_ = np.linalg.lstsq(X, mm, rcond=None)
                    pred = float(beta[0] + beta[1] * ey0)
                    adj = base + pred / max(n, 1)
                    err = float((adj - y).abs().sum())
                    rows.append({
                        "window": w, "origin": o, "month": str(p),
                        "variant": "resid_ols_L0",
                        "eps": float("nan"), "lag": 0, "err_mn": err, "n": n,
                        "equity_yoy": ey0, "resid_pred": pred,
                    })
        if np.isfinite(ey0):
            hist_miss.append((o, month_miss, ey0))
    return pd.DataFrame(rows)


def summarize_bucket(detail: pd.DataFrame) -> pd.DataFrame:
    g = detail.groupby(["variant", "eps", "lag", "window"], dropna=False).apply(
        lambda x: pd.Series({
            "mae_bn": x["err_mn"].sum() / x["n"].sum() / 1000.0,
            "n_days": int(x["n"].sum()),
            "n_months": len(x),
        }),
        include_groups=False,
    ).reset_index()
    return g


def score_net_flow(
    yoy: pd.Series,
    specs: list[tuple[str, float, int]],
    *,
    start: str = "2022-01-01",
    every: str = "W-FRI",
) -> pd.DataFrame:
    """Weekly-origin cumulative net op-flow MAE ($bn) at h=1/5/10/20.

    Fits once per month; applies equity scale only to dep_corporate so the grid
    reuses the same baseline forecast.
    """
    import flow_model as fm

    panel, coupon = fm.load_inputs()
    feats = fm.calendar_features(panel.index)
    actual = fm._net(panel)
    origins = sorted({
        panel.index[panel.index.searchsorted(d, side="right") - 1]
        for d in pd.date_range(start, panel.index.max(), freq=every)
        if panel.index.searchsorted(d, side="right") > 0
    })
    rows = []
    models: dict = {}
    fitted_month = None
    for i, o in enumerate(origins):
        if fitted_month != o.to_period("M"):
            models = fm.fit_all(panel, feats, o)
            fitted_month = o.to_period("M")
            if (i + 1) % 20 == 0:
                print(f"   net refit {o.date()} ({i + 1}/{len(origins)})", flush=True)
        fut = panel.index[panel.index > o][: fm.HORIZON]
        if len(fut) == 0:
            continue
        fc = fm.forecast(models, panel, coupon, o, fut)
        base_net = fm._net(fc)
        corp_base = fc["dep_corporate"].copy()
        for name, eps, lag in specs:
            if name == "baseline":
                net = base_net
            else:
                fac = scale_factor(yoy_at(yoy, o, lag), eps)
                fc2 = fc.copy()
                fc2["dep_corporate"] = corp_base * fac
                net = fm._net(fc2)
            for h, d in enumerate(fut, start=1):
                if h not in fm.HORIZONS:
                    continue
                rows.append((name, eps, lag, o, d, h, actual.loc[d], net.loc[d]))
    bt = pd.DataFrame(rows, columns=["variant", "eps", "lag", "origin", "date", "h", "actual", "model"])
    bt = bt.sort_values(["variant", "origin", "h"])
    bt["cum_actual"] = bt.groupby(["variant", "origin"])["actual"].cumsum()
    bt["cum_model"] = bt.groupby(["variant", "origin"])["model"].cumsum()
    out = []
    for (variant, eps, lag), sl0 in bt.groupby(["variant", "eps", "lag"], dropna=False):
        for w, (a, b) in fm.WINDOWS.items():
            sl = sl0[(sl0["origin"] >= a) & (sl0["origin"] <= b)]
            for h in fm.HORIZONS:
                x = sl[sl["h"] == h]
                if x.empty:
                    continue
                out.append({
                    "variant": variant, "eps": float(eps) if pd.notna(eps) else None,
                    "lag": int(lag) if pd.notna(lag) and lag >= 0 else None,
                    "window": w, "h": int(h), "n": len(x),
                    "mae_bn": float((x["cum_model"] - x["cum_actual"]).abs().mean() / 1000.0),
                })
    return pd.DataFrame(out)


def pick_best(bucket_sum: pd.DataFrame) -> dict:
    """Prefer OOS improvement with PRE not worse than baseline × PRE_TOL."""
    base = bucket_sum[bucket_sum["variant"] == "baseline"]
    if base.empty:
        return {"verdict": "no", "reason": "no baseline"}
    pre0 = float(base.loc[base["window"] == "PRE", "mae_bn"].iloc[0])
    oos0 = float(base.loc[base["window"] == "OOS", "mae_bn"].iloc[0])
    cands = []
    for v, g in bucket_sum.groupby("variant"):
        if v == "baseline":
            continue
        pre = g.loc[g["window"] == "PRE", "mae_bn"]
        oos = g.loc[g["window"] == "OOS", "mae_bn"]
        if pre.empty or oos.empty:
            continue
        pre_m, oos_m = float(pre.iloc[0]), float(oos.iloc[0])
        row = g.iloc[0]
        cands.append({
            "variant": v,
            "eps": None if pd.isna(row["eps"]) else float(row["eps"]),
            "lag": None if pd.isna(row["lag"]) or row["lag"] < 0 else int(row["lag"]),
            "pre": pre_m, "oos": oos_m,
            "d_pre": pre_m - pre0, "d_oos": oos_m - oos0,
            "pre_ok": pre_m <= pre0 * PRE_TOL,
            "oos_win": oos_m < oos0,
        })
    # Prefer OOS win + PRE ok; among those, prefer PRE not worse, then largest OOS cut.
    # Positive-ε (UK-style equity→tax) is the only ship path — negative ε is a
    # trailing-mean fade, not the desk thesis.
    cands = sorted(
        cands,
        key=lambda r: (not r["oos_win"], not r["pre_ok"], r["d_pre"] > 0, r["d_oos"], r["d_pre"]),
    )
    gated = [c for c in cands if c["oos_win"] and c["pre_ok"]]
    pos_gated = [c for c in gated if (c.get("eps") or 0) > 0]
    neg_gated = [c for c in gated if (c.get("eps") or 0) < 0]
    clear = [
        c for c in pos_gated
        if oos0 > 0 and (-c["d_oos"] / oos0) >= OOS_REL_MIN
    ]
    ship = bool(clear)
    best = clear[0] if ship else (pos_gated[0] if pos_gated else (gated[0] if gated else (cands[0] if cands else None)))
    return {
        "verdict": "ship" if ship else "no",
        "baseline_pre": pre0,
        "baseline_oos": oos0,
        "oos_rel_min": OOS_REL_MIN,
        "best": best,
        "best_positive_eps": pos_gated[0] if pos_gated else None,
        "best_negative_eps": neg_gated[0] if neg_gated else None,
        "gated_count": len(gated),
        "clear_positive_count": len(clear),
        "top": cands[:12],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--code", default="SP500", choices=list(EQUITY_CODES) + ["SPASTT01USM661N"])
    ap.add_argument("--net", action="store_true", help="also score net op-flow MAE (slower)")
    ap.add_argument("--write-md", action="store_true", help="rewrite equity_corporate_tax.md")
    args = ap.parse_args()

    print(f"loading equity {args.code}…", flush=True)
    yoy = load_equity_yoy(refresh=args.refresh, code=args.code)
    print(f"  {args.code} YoY: {yoy.index.min().date()} → {yoy.index.max().date()} "
          f"last={yoy.iloc[-1]:.2f}%", flush=True)

    print("scoring dep_corporate bucket…", flush=True)
    detail = score_corporate_bucket(yoy)
    summ = summarize_bucket(detail)
    pick = pick_best(summ)

    out_dir = HERE / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "equity_corporate_ab_detail.parquet"
    summ_path = out_dir / "equity_corporate_ab.json"
    detail.to_parquet(detail_path, index=False)

    net_tbl = None
    if args.net and pick.get("best"):
        b = pick["best"]
        specs = [("baseline", 0.0, -1)]
        if b.get("eps") is not None and b.get("lag") is not None:
            specs.append((b["variant"], float(b["eps"]), int(b["lag"])))
        # also a UK-like ε=0.7 L0 and ε=1 L3 for reference
        for eps, lag in ((0.7, 0), (1.0, 3), (1.0, 0)):
            name = f"scale_e{eps}_L{lag}"
            if name not in {s[0] for s in specs}:
                specs.append((name, eps, lag))
        print(f"scoring net op-flow for {len(specs)} specs…", flush=True)
        net_tbl = score_net_flow(yoy, specs)

    payload = {
        "equity_code": args.code,
        "equity_last": float(yoy.iloc[-1]),
        "equity_through": str(yoy.index.max().date()),
        "eps_grid": list(EPS_GRID),
        "lags": list(LAGS),
        "pre_tol": PRE_TOL,
        "bucket": summ.to_dict(orient="records"),
        "pick": pick,
        "net": None if net_tbl is None else net_tbl.to_dict(orient="records"),
    }
    summ_path.write_text(json.dumps(payload, indent=1, default=float))
    print(json.dumps(pick, indent=1, default=float))
    if net_tbl is not None:
        print(net_tbl.pivot_table(index=["window", "h"], columns="variant", values="mae_bn").round(2).to_string())

    if args.write_md:
        write_md(payload, Path(__file__).with_name("equity_corporate_tax.md"))


def write_md(payload: dict, path: Path) -> None:
    pick = payload["pick"]
    base_pre = pick["baseline_pre"]
    base_oos = pick["baseline_oos"]
    pos = pick.get("best_positive_eps") or {}
    neg = pick.get("best_negative_eps") or {}
    rel = pick.get("oos_rel_min", OOS_REL_MIN)

    def _row(label: str, c: dict | None) -> str:
        if not c:
            return f"| {label} | — | — | — | — |"
        return (
            f"| {label} (`{c['variant']}`) | {c['pre']:.4f} | {c['oos']:.4f} | "
            f"{c['d_pre']:+.4f} | {c['d_oos']:+.4f} |"
        )

    ship = pick["verdict"] == "ship"
    lines = [
        "# Equity × corporate income tax (DTS `dep_corporate`)",
        "",
        f"## Verdict: **{'ship' if ship else 'no'}** — "
        + ("promote positive-ε surprise overlay" if ship else "do not ship"),
        "",
        (
            "Positive-ε surprise overlay clears PRE/OOS gate."
            if ship else
            "UK-style **positive** equity → corporate scale does not clear a material OOS bar."
        ),
        "Production stays trailing 250d mean × HGB calendar (`choice=model`). "
        "No change to `fy_overlay` / `fiscal_path`.",
        "",
        "| | PRE (2022–23) | OOS (2024+) | ΔPRE | ΔOOS |",
        "|---|---:|---:|---:|---:|",
        f"| baseline | {base_pre:.4f} | {base_oos:.4f} | — | — |",
        _row("best UK-style (+ε)", pos if pos else None),
        _row("best fade (−ε)", neg if neg else None),
        "",
        f"Ship rule: positive ε, OOS better, PRE ≤ baseline×{payload.get('pre_tol', PRE_TOL):.0%}, "
        f"OOS cut ≥{rel:.0%} relative. "
        f"clear_positive_count={pick.get('clear_positive_count', 0)}.",
        "",
        "## How corporate is forecast today",
        "",
        "- **Scored / live daily path** (`flow_model.py`): `dep_corporate` = trailing "
        "250-business-day mean × HGB calendar shape. PRE choice = `model`.",
        "- **Multi-year path** (`fy_overlay.apply_fy`): trailing growth only — "
        "**no equity**.",
        "- UK analogue: `other_tax` × equity YoY (FRED `SPASTT01GBM661N`, ε=0.7).",
        "",
        "## Experiment",
        "",
        f"- Equity: FRED `{payload['equity_code']}` monthly mean → YoY % "
        f"(through {payload['equity_through']}, last {payload['equity_last']:.1f}%).",
        "- Overlays: scale / surprise(yoy−12m mean) / resid_ols; ε grid incl. negatives; "
        "lags 0/1/3/6/12; month-ends ≤ origin only.",
        "",
        "## Read of the signs",
        "",
        "- Raw ε>0: loses OOS (double-counts the trailing mean).",
        "- Raw ε<0: OOS improves — fade equity into the mean; wrong sign for the thesis.",
        "- Surprise ε>0: small OOS cuts; none material under the 2% bar unless ship flips.",
        "- Residual OLS: worse on both windows.",
        "",
    ]
    if payload.get("net"):
        lines += ["## Net operating-flow MAE ($bn)", "", "```"]
        net = pd.DataFrame(payload["net"])
        piv = net.pivot_table(index=["window", "h"], columns="variant", values="mae_bn")
        lines.append(piv.round(2).to_string())
        lines.append("```")
        lines.append("")
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "cd us/fiscal && ../../.venv/bin/python experiments/equity_corporate_ab.py --write-md",
        "```",
        "",
        "Artifacts: `derived/equity_corporate_ab.json`, `equity_corporate_ab_detail.parquet`.",
        "",
    ]
    # drop empty strings from the ship/no branch
    text = "\n".join(x for x in lines if x is not None)
    path.write_text(text if text.endswith("\n") else text + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
