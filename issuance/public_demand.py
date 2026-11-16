"""Net official-sector demand out of gross issuance.

UK APF is ISIN-exact (BoE operations). US SOMA is CUSIP-exact (NY Fed monthly).
EA PSPP+PEPP ages last month's ISIN book and places only the flow
(`qe_alloc.allocate_isin`). The first month's flow equals that month's
opening net (cumsum starts there). Later months are stock diffs.
Tenor flows are the ISIN solution aggregated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from issuance.paths import CACHE
from issuance.qe_alloc import (
    PROGRAM_RULES,
    AllocResult,
    allocate_isin,
    phase_of,
    fit_row,
    result_frame,
)
from issuance.tenor import BUCKET_MID, BUCKET_ORDER, attach_tenor, bucket_from_years

ISSUER_CAP = 0.33  # PSPP issue-share limit (PEPP default is 0.50 in qe_alloc)


def _apf_monthly() -> pd.DataFrame:
    path = CACHE / "apf_operations.parquet"
    if not path.exists():
        return pd.DataFrame()
    ops = pd.read_parquet(path)
    ops = attach_tenor(ops)
    if "amount" in ops.columns and "amount_issued" not in ops.columns:
        ops["amount_issued"] = pd.to_numeric(ops["amount"], errors="coerce")
    ops["month"] = pd.to_datetime(ops["op_dt"]).dt.to_period("M").dt.to_timestamp()
    if "tenor_bucket" not in ops.columns:
        ops["tenor_bucket"] = ops.get("resid_mat_yrs", pd.Series(dtype=float)).map(bucket_from_years)
    g = ops.groupby(["month", "isin", "tenor_bucket"], dropna=False)["amount_issued"].sum().reset_index()
    g = g.rename(columns={"amount_issued": "official_mn"})
    g["country"] = "UK"
    g["program"] = "APF"
    return g


def _parse_ecb_country_panel() -> pd.DataFrame:
    """Long panel: program, country, month, holdings_mn, wam_yrs."""
    path = CACHE / "ecb_qe_holdings.parquet"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(path)
    if raw.empty:
        return raw
    keep = raw[raw.get("country", pd.Series(dtype=str)).isin(["DE", "FR", "IT"])].copy()
    if keep.empty:
        return keep
    keep["month"] = pd.to_datetime(keep["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    # fetch_ecb_qe writes holdings_mn = cumsum(monthly net). Never reuse `value`
    # (that is the monthly flow) as if it were the stock.
    if "holdings_mn" in keep.columns:
        keep["holdings_mn"] = pd.to_numeric(keep["holdings_mn"], errors="coerce")
    elif "value" in keep.columns and keep.get("metric", pd.Series(dtype=str)).eq("stock").any():
        keep["holdings_mn"] = pd.to_numeric(keep["value"], errors="coerce")
    else:
        keep["holdings_mn"] = pd.to_numeric(keep.get("value"), errors="coerce")
    if "wam_yrs" in keep.columns:
        keep["wam_yrs"] = pd.to_numeric(keep["wam_yrs"], errors="coerce")
    elif "wam" in keep.columns:
        keep["wam_yrs"] = pd.to_numeric(keep["wam"], errors="coerce")
    else:
        keep["wam_yrs"] = np.nan
    # Latest WAM xlsx is a snapshot; broadcast last known WAM backward if needed.
    wam_path = CACHE / "ecb_pspp_wam_raw.parquet"
    if wam_path.exists():
        wam_raw = pd.read_parquet(wam_path)
        wam_long = _wam_from_xlsx(wam_raw)
        if not wam_long.empty:
            keep = keep.merge(wam_long, on=["country", "month"], how="left", suffixes=("", "_x"))
            if "wam_yrs_x" in keep.columns:
                keep["wam_yrs"] = keep["wam_yrs"].fillna(keep["wam_yrs_x"])
    return keep.dropna(subset=["month", "holdings_mn"])


def _wam_from_xlsx(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["country", "month", "wam_yrs"])
    # Try to find a country column and date columns or a date column + WAM.
    cols = [str(c).lower() for c in raw.columns]
    country_col = None
    for c in raw.columns:
        if str(c).lower() in {"country", "jurisdiction", "issuer"}:
            country_col = c
            break
    if country_col is None:
        country_col = raw.columns[0]
    date_cols = [c for c in raw.columns if c != country_col]
    long = raw.melt(id_vars=[country_col], value_vars=date_cols, var_name="date", value_name="wam_yrs")
    long = long.rename(columns={country_col: "country"})
    long["country"] = long["country"].astype(str).str.upper().str[:2]
    long["month"] = pd.to_datetime(long["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    long["wam_yrs"] = pd.to_numeric(long["wam_yrs"], errors="coerce")
    return long.dropna(subset=["month", "wam_yrs"])[["country", "month", "wam_yrs"]]


def allocate_tenor(target_stock: float, target_wam: float,
                   outstanding_by_bucket: dict[str, float],
                   prev: dict[str, float] | None = None) -> dict[str, float]:
    """Allocate a country stock across tenor buckets so the WAM matches.

    Minimise distance to the previous allocation (or to outstanding weights),
    subject to sum h = S, sum h * mid = S * WAM, 0 <= h <= cap * outstanding.
    """
    buckets = [b for b in BUCKET_ORDER if outstanding_by_bucket.get(b, 0) > 0]
    if not buckets or target_stock <= 0:
        return {b: 0.0 for b in BUCKET_ORDER}
    mids = np.array([BUCKET_MID[b] for b in buckets], dtype=float)
    outst = np.array([outstanding_by_bucket[b] for b in buckets], dtype=float)
    cap = np.minimum(outst, ISSUER_CAP * outst if outst.sum() > 0 else outst)
    # If the official stock exceeds the capped universe, relax the cap.
    if cap.sum() < target_stock:
        cap = outst.copy()
    if prev:
        x0 = np.array([prev.get(b, 0.0) for b in buckets], dtype=float)
    else:
        x0 = outst / outst.sum() * target_stock
    x0 = np.clip(x0, 0, cap)
    if x0.sum() > 0:
        x0 = x0 / x0.sum() * min(target_stock, cap.sum())

    def objective(x):
        return float(np.sum((x - x0) ** 2))

    cons = [
        {"type": "eq", "fun": lambda x: np.sum(x) - target_stock},
        {"type": "eq", "fun": lambda x: np.dot(x, mids) - target_stock * target_wam},
    ]
    bounds = [(0.0, float(c)) for c in cap]
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-8})
    alloc = res.x if res.success else x0
    # If the WAM constraint is infeasible (common at the short end of QT), drop it.
    if not res.success:
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - target_stock}]
        res2 = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-8})
        alloc = res2.x if res2.success else x0
    out = {b: 0.0 for b in BUCKET_ORDER}
    for b, v in zip(buckets, alloc):
        out[b] = max(0.0, float(v))
    return out


_OUTSTANDING_CACHE: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
_ISIN_STOCK_CACHE: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}


def _isin_stock(ops: pd.DataFrame, country: str, asof,
                resid_lo: float = 0.0, resid_hi: float = np.inf) -> pd.DataFrame:
    key = (country, pd.Timestamp(asof), float(resid_lo), float(resid_hi))
    if key in _ISIN_STOCK_CACHE:
        return _ISIN_STOCK_CACHE[key]
    g = ops[(ops["country"] == country) & (ops["op_dt"] <= asof) & ops["isin"].notna()].copy()
    if not g.empty and "sec_type" in g.columns:
        g = g[~g["sec_type"].astype(str).isin(["Bill", "Foreign"])]
    if g.empty:
        empty = pd.DataFrame(columns=["isin", "outstanding", "maturity_dt", "resid", "tenor_bucket"])
        _ISIN_STOCK_CACHE[key] = empty
        return empty
    last = g.groupby("isin").agg(
        outstanding=("amount_issued", "sum"),
        maturity_dt=("maturity_dt", "last"),
    ).reset_index()
    last = last[last["outstanding"] > 0]
    last["resid"] = (pd.to_datetime(last["maturity_dt"]) - pd.Timestamp(asof)).dt.days / 365.25
    last = last[(last["resid"] >= resid_lo) & (last["resid"] < resid_hi)]
    last["tenor_bucket"] = last["resid"].map(bucket_from_years)
    _ISIN_STOCK_CACHE[key] = last
    return last


def _outstanding_buckets(ops: pd.DataFrame, country: str, asof) -> dict[str, float]:
    key = (country, pd.Timestamp(asof))
    if key in _OUTSTANDING_CACHE:
        return _OUTSTANDING_CACHE[key]
    last = _isin_stock(ops, country, asof)
    if last.empty:
        _OUTSTANDING_CACHE[key] = {}
        return {}
    out = last.groupby("tenor_bucket")["outstanding"].sum().to_dict()
    _OUTSTANDING_CACHE[key] = out
    return out


def _ea_isin(ops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ISIN-level PSPP/PEPP book + per-month fit diagnostics."""
    panel = _parse_ecb_country_panel()
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    fits = []
    prev_hold: dict[tuple[str, str], dict[str, float]] = {}
    prev_target: dict[tuple[str, str], float] = {}
    for (country, program), grp in panel.groupby(["country", "program"]):
        grp = grp.sort_values("month").drop_duplicates("month", keep="last")
        rules = PROGRAM_RULES.get(str(program), PROGRAM_RULES["PSPP"])
        last_wam = 7.0
        key = (country, program)
        wam_hist: list[float] = []
        for rec in grp.itertuples(index=False):
            wam = rec.wam_yrs if pd.notna(getattr(rec, "wam_yrs", np.nan)) else last_wam
            if pd.notna(wam):
                last_wam = float(wam)
            stock = float(rec.holdings_mn)
            universe = _isin_stock(
                ops, country, rec.month,
                resid_lo=rules["resid_lo"], resid_hi=rules["resid_hi"],
            )
            res: AllocResult = allocate_isin(
                stock, last_wam, universe,
                prev=prev_hold.get(key),
                cap=rules["cap"],
                buy_lo=rules.get("buy_lo"),
                buy_hi=rules.get("buy_hi"),
                prev_wams=wam_hist,
                prev_target=prev_target.get(key),
                phase=phase_of(str(program), rec.month),
            )
            prev_target[key] = stock
            if pd.notna(res.achieved_wam):
                wam_hist.append(float(res.achieved_wam))
                wam_hist = wam_hist[-4:]
            prev_hold[key] = {str(i): float(h) for i, h in zip(res.isins, res.holdings) if h > 1e-8}
            frame = result_frame(res, rec.month, country, program)
            if not frame.empty:
                frames.append(frame)
            fits.append(fit_row(res, rec.month, country, program, n_live=len(universe)))
    isin = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not isin.empty:
        isin = isin.sort_values(["country", "program", "isin", "month"])
        isin["official_mn"] = isin.groupby(["country", "program", "isin"])["official_stock_mn"].diff()
        first = ~isin.duplicated(["country", "program", "isin"])
        isin.loc[first, "official_mn"] = isin.loc[first, "official_stock_mn"]
        isin["wam_yrs"] = isin["wam_target"]
    fit = pd.DataFrame(fits)
    if not isin.empty and not fit.empty:
        # Runoff months re-spread the book to the published WAM; nothing was traded.
        run = fit.loc[fit["mode"] == "runoff", ["country", "program", "month"]].assign(_run=True)
        isin = isin.merge(run, on=["country", "program", "month"], how="left")
        isin.loc[isin["_run"].fillna(False).astype(bool), "official_mn"] = 0.0
        isin = isin.drop(columns="_run")
    return isin, fit


def _tenor_from_isin(isin: pd.DataFrame) -> pd.DataFrame:
    if isin is None or isin.empty:
        return pd.DataFrame()
    g = isin.groupby(["country", "program", "month", "tenor_bucket"], dropna=False).agg(
        official_stock_mn=("official_stock_mn", "sum"),
        official_mn=("official_mn", "sum"),
        wam_yrs=("wam_yrs", "first") if "wam_yrs" in isin.columns else ("wam_target", "first"),
    ).reset_index()
    return g


def official_isin_flow(ops: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Kept for callers. EA path now writes ISIN holdings directly."""
    path = CACHE / "official_isin_flow.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return official if official is not None else pd.DataFrame()


def _soma_monthly() -> pd.DataFrame:
    from issuance.fetch_soma import soma_flow
    return soma_flow()


def _boj_monthly(ops: pd.DataFrame) -> pd.DataFrame:
    """BoJ JGB holdings by issue -> monthly official flow. A bond maturing out of the book is a
    redemption, not a sale, so matured rows carry no flow. Bucket is the residual at month end."""
    path = CACHE / "boj_isin_flow.parquet"
    if not path.exists():
        return pd.DataFrame()
    f = pd.read_parquet(path)
    f = f[~f["matured"].astype(bool)].copy()
    jp = ops[ops["country"].astype(str).eq("JP") & ops["maturity_dt"].notna()] if "country" in ops.columns else ops.iloc[0:0]
    mats = jp.groupby(jp["isin"].astype(str))["maturity_dt"].max()
    linker = set(jp.loc[jp["sec_type"].astype(str).eq("Linker"), "isin"].astype(str))
    f["maturity_dt"] = f["isin"].map(mats)
    f = f[f["maturity_dt"].notna()]
    f["month"] = pd.to_datetime(f["month"])
    end = f["month"] + pd.offsets.MonthEnd(0)
    f["resid_mat_yrs"] = (pd.to_datetime(f["maturity_dt"]) - end).dt.days / 365.25
    f = f[f["resid_mat_yrs"] > 0]
    f["tenor_bucket"] = f["resid_mat_yrs"].map(bucket_from_years)
    return pd.DataFrame({
        "country": "JP", "program": "BoJ", "month": f["month"], "isin": f["isin"],
        "tenor_bucket": f["tenor_bucket"], "official_mn": f["flow_mn"].astype(float),
        "official_stock_mn": f["holdings_mn"].astype(float), "maturity_dt": f["maturity_dt"],
        "resid_mat_yrs": f["resid_mat_yrs"],
        "sec_type": f["isin"].map(lambda s: "Linker" if s in linker else "Nominal"),
    })


def official_flow(ops: pd.DataFrame) -> pd.DataFrame:
    _OUTSTANDING_CACHE.clear()
    _ISIN_STOCK_CACHE.clear()
    apf = _apf_monthly()
    soma = _soma_monthly()
    boj = _boj_monthly(ops)
    ea_isin, fit = _ea_isin(ops)
    isin_parts = [
        p for p in (
            apf if (apf is not None and not apf.empty and "isin" in apf.columns) else pd.DataFrame(),
            soma if (soma is not None and not soma.empty and "isin" in soma.columns) else pd.DataFrame(),
            boj,
            ea_isin,
        ) if p is not None and not p.empty
    ]
    if isin_parts:
        isin = pd.concat(isin_parts, ignore_index=True)
        isin.to_parquet(CACHE / "official_isin_flow.parquet", index=False)
    else:
        isin = pd.DataFrame()
    if not fit.empty:
        fit.to_parquet(CACHE / "official_qe_fit.parquet", index=False)
    tenor_parts = []
    if not ea_isin.empty:
        tenor_parts.append(_tenor_from_isin(ea_isin))
    if apf is not None and not apf.empty:
        tenor_parts.append(apf.groupby(
            ["country", "program", "month", "tenor_bucket"], dropna=False
        )["official_mn"].sum().reset_index())
    if not boj.empty:
        tenor_parts.append(boj.groupby(
            ["country", "program", "month", "tenor_bucket"], dropna=False
        ).agg(official_mn=("official_mn", "sum"), official_stock_mn=("official_stock_mn", "sum")).reset_index())
    if soma is not None and not soma.empty:
        tenor_parts.append(soma.groupby(
            ["country", "program", "month", "tenor_bucket"], dropna=False
        )["official_mn"].sum().reset_index())
    tenor_parts = [p for p in tenor_parts if p is not None and not p.empty]
    if not tenor_parts:
        return pd.DataFrame()
    out = pd.concat(tenor_parts, ignore_index=True)
    out.to_parquet(CACHE / "official_flow.parquet", index=False)
    return out


def net_private(tenor_flow: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    if official is None or official.empty:
        tenor_flow = tenor_flow.copy()
        tenor_flow["official_mn"] = 0.0
        tenor_flow["private_mn"] = tenor_flow["gross_mn"]
        return tenor_flow
    off = official[official.get("tenor_bucket", pd.Series(dtype=str)).astype(str) != "non_central"]
    off = off.groupby(["country", "month", "tenor_bucket"], dropna=False)["official_mn"].sum().reset_index()
    merged = tenor_flow.merge(off, on=["country", "month", "tenor_bucket"], how="left")
    merged["official_mn"] = merged["official_mn"].fillna(0.0)
    merged["private_mn"] = merged["gross_mn"] - merged["official_mn"]
    merged.to_parquet(CACHE / "flow_tenor_private.parquet", index=False)
    return merged
