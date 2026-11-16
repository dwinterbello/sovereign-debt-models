"""UK central-government cash: history identity, CGNCR seasonal model, implied T-bill stock.

The DMO neutralises Exchequer flows daily with T-bills, DMO repo/reverse repo and its balance at
the Bank, so the object here is the T-bill stock and the DMO's cash position, not reserves.

History identity (monthly, £bn, inflows to the buffer positive)

    Δbuffer = −CGNCR(ex NRAM, B&B, NR) + net gilt cash + ΔT-bill stock + other_fin

buffer        CG net deposits with MFIs, monthly from ONS: −(ANSZ + ANTB). This is the DMO's net
              cash position plus the cash-management T-bill stock (the DMO counts cash-management
              bills as an offset to its cash, not as financing). End-March anchors: DMR net cash
              position + (total D2.2G stock − debt-financing stock).
gilts         DMO D2.1E priced operations: clean × index ratio; redemptions D1C nominal ×
              tranche-weighted issue ratio, net of CRND/DMO holdings. The APF-held part
              (bond_redemption_cb_bn) is Exchequer cash paid to the Bank, not to private holders.
bills         D2.2G total stock outstanding (incl. cash-management bills, excl. collateral bills).
other_fin     NS&I (ΔACUA) + deposits with CG by other sectors (ANTC, mainly DMADF) + NRAM/NR
              (RUUW − M98R) + ONS-vs-DMO gilt and bill measurement gaps + the ONS financing gap
              (everything the ONS financing split does not itemise: NILO, coin, reserves).

Forecast: CGNCR and NS&I from month-of-year models chosen on PRE origins only; gilts from the remit
/ forward calendar and the redemption ladder; forward other_fin = NS&I + the remit's other items.
In the current FY the bill path is solved so the DMO net cash position lands on the remit's
end-March plan; what that takes beyond the planned debt-financing bills is next_revision_gap_bn.
After March the bill model chosen on PRE runs. Δbuffer falls out of the identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.uk import sources as src

PRE = (pd.Timestamp("2012-01-01"), pd.Timestamp("2019-12-01"))
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-01"))
OOS_START = pd.Timestamp("2022-01-01")
COVID_EST = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-03-01"))  # excluded from estimation
HORIZONS = (1, 3, 6, 12)
FC_MONTHS = 12  # backtest / score horizon cap; live forward uses cash_season.forward_months

MODEL_LABELS = {
    "seasonal_add5": "seasonal, 5y month effects",
    "seasonal_add3": "seasonal, 3y month effects",
    "profile_mult5": "FY share profile, 5y",
    "smly": "same month last year",
    "trail_mean": "trailing 12m mean",
    "official_vintage": "OBR CGNCR ex Network Rail",
    "bill_seasonal": "seasonal Δ bills",
    "bill_beta": "Δ bills on financing need",
    "bill_absorb": "bills absorb the need",
    "bill_flat": "flat stock",
}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


CGNCR_MODELS = ("seasonal_add5", "seasonal_add3", "profile_mult5")
CGNCR_BENCH = ("smly", "trail_mean")
OFFICIAL = "official_vintage"
BILL_MODELS = ("bill_seasonal", "bill_beta", "bill_absorb")
BILL_BENCH = ("bill_flat",)
INPUT_MONTHS = 12  # every model reads at most the trailing 12 months as level/smly inputs
# Sovereign Sukuk 0.333% matured 22-Jul-2026 (DMO remit 2026-27 Annex C: £0.5bn).
SUKUK_MONTH = {2026: pd.Timestamp("2026-07-01")}


def fy_of(month: pd.Timestamp | pd.DatetimeIndex):
    return month.year - (month.month < 4)


# --------------------------------------------------------------------------------------------
# History


def build_history() -> pd.DataFrame:
    ons = src.ons_monthly()
    bills = src.bills_stock()
    ops = src.gilt_ops_cash()
    ops["month"] = ops["settle"].dt.to_period("M").dt.to_timestamp()
    red = src.gilt_redemptions(ops)

    sales = ops[ops["mechanism"].isin(src.CASH_IN)].groupby("month")["cash_bn"].sum()
    reverse = -ops[ops["mechanism"].isin(src.CASH_OUT)].groupby("month")["cash_bn"].sum()
    redeem = red.groupby("month")["cash_bn"].sum()
    redeem_cb = red.groupby("month")["cb_cash_bn"].sum()

    last = ons["cgncr_ex"].dropna().index.max()
    idx = pd.date_range("2000-05-01", last, freq="MS")
    df = pd.DataFrame(index=idx)
    df.index.name = "month"
    df["cash_requirement_bn"] = ons["cgncr_ex"].reindex(idx)
    df["cgncr_ruuw_bn"] = ons["cgncr"].reindex(idx)
    df["bond_gross_bn"] = sales.reindex(idx).fillna(0.0)
    df["bond_redemption_bn"] = redeem.reindex(idx).fillna(0.0) + reverse.reindex(idx).fillna(0.0)
    df["net_bonds_bn"] = df["bond_gross_bn"] - df["bond_redemption_bn"]
    df["bond_redemption_cb_bn"] = redeem_cb.reindex(idx).fillna(0.0)
    df["net_bonds_private_bn"] = df["net_bonds_bn"] + df["bond_redemption_cb_bn"]
    df["bill_stock_bn"] = bills["bill_stock"].reindex(idx)
    df["net_bills_bn"] = bills["bill_stock"].diff().reindex(idx)
    df["bill_gross_bn"] = bills["bill_gross"].reindex(idx)
    df["bill_redemption_bn"] = df["bill_gross_bn"] - df["net_bills_bn"]
    df["bill_stock_debtfin_bn"] = bills["bill_stock_debtfin"].reindex(idx)

    df["buffer_change_bn"] = -(ons["fin_dep_mfi"] + ons["fin_loans_mfi"]).reindex(idx)
    df["of_nsandi_bn"] = ons["nsandi"].diff().reindex(idx)
    df["of_dep_by_others_bn"] = ons["fin_dep_cg"].reindex(idx)
    df["of_nram_nr_bn"] = -(df["cgncr_ruuw_bn"] - df["cash_requirement_bn"])
    df["of_gilt_vs_ons_bn"] = ons["fin_bgs"].reindex(idx) - df["net_bonds_bn"]
    df["of_bill_vs_ons_bn"] = ons["tbill_level"].diff().reindex(idx) - df["net_bills_bn"]
    df["other_fin_bn"] = (df["buffer_change_bn"] + df["cash_requirement_bn"]
                          - df["net_bonds_bn"] - df["net_bills_bn"])
    named = ["of_nsandi_bn", "of_dep_by_others_bn", "of_nram_nr_bn", "of_gilt_vs_ons_bn",
             "of_bill_vs_ons_bn"]
    df["of_unexplained_bn"] = df["other_fin_bn"] - df[named].sum(axis=1)
    df = df.dropna(subset=["cash_requirement_bn", "net_bills_bn", "buffer_change_bn"])
    df["is_forecast"] = False
    return df


def anchor_buffer(hist: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Cumulate Δbuffer, pinned at the latest end-March outturn (DMR net cash + cash-mgmt bills).

    Returns the level and a check of every other outturn anchor against the cumulated proxy.
    """
    out = anchors[anchors["kind"] == "outturn"].copy()
    rows = []
    for r in out.itertuples(index=False):
        m = pd.Timestamp(int(r.fy[:4]) + 1, 3, 1)
        if m not in hist.index or pd.isna(hist.at[m, "bill_stock_debtfin_bn"]):
            continue
        cm = hist.at[m, "bill_stock_bn"] - hist.at[m, "bill_stock_debtfin_bn"]
        rows.append({"month": m, "net_cash": float(r.net_cash_end_march), "cash_mgmt_bills": cm,
                     "buffer_anchor": float(r.net_cash_end_march) + cm})
    cum = hist["buffer_change_bn"].cumsum()
    if not rows:
        return cum - cum.iloc[-1], []
    pin = max(rows, key=lambda x: x["month"])
    level = cum - cum[pin["month"]] + pin["buffer_anchor"]
    checks = [{**r, "proxy": float(level[r["month"]]),
               "miss_bn": float(level[r["month"]] - r["buffer_anchor"])} for r in rows]
    return level, checks


# --------------------------------------------------------------------------------------------
# CGNCR models


def _est_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return ~((index >= COVID_EST[0]) & (index <= COVID_EST[1]))


def _trail_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """False where the trailing 12-month window touches a Covid month."""
    return ~((index >= COVID_EST[0]) & (index <= COVID_EST[1] + pd.DateOffset(months=11)))


def _fy_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """False for every month of a financial year that contains a Covid month."""
    bad = {fy_of(COVID_EST[0]), fy_of(COVID_EST[1])}
    return ~pd.Index(fy_of(index)).isin(bad)


def cgncr_forecast(x: pd.Series, origin: pd.Timestamp, method: str, h: int = FC_MONTHS) -> np.ndarray:
    """Monthly CGNCR forecasts for origin+1..origin+h using data through origin only."""
    s = x.loc[:origin].dropna()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    if method == "smly":
        return np.array([s.get(m - pd.DateOffset(years=1 if k < 12 else 2), np.nan)
                         for k, m in enumerate(months)], dtype=float)
    level = s.iloc[-12:].mean()
    if method == "trail_mean":
        return np.full(h, level)
    if method.startswith("seasonal_add"):
        n = int(method[-1])
        tm = s.rolling(12).mean()
        dev = (s - tm)[_trail_mask(s.index)].dropna()
        out = []
        for m in months:
            d = dev[dev.index.month == m.month].iloc[-n:]
            out.append(level + (d.mean() if len(d) else 0.0))
        return np.array(out)
    if method.startswith("profile_mult"):
        n = int(method[-1])
        fy = pd.Series(fy_of(s.index), index=s.index)
        tot = s.groupby(fy).transform("sum")
        cnt = s.groupby(fy).transform("size")
        share = (s / tot)[(cnt == 12) & _fy_mask(s.index)]
        annual = s.iloc[-12:].sum()
        out = []
        for m in months:
            sh = share[share.index.month == m.month].iloc[-n:]
            out.append(annual * (sh.median() if len(sh) else 1 / 12))
        return np.array(out)
    raise ValueError(method)


# Remit / prior complete FY above this → do not freeze hist month shares onto the
# residual (same class as DE SV scale channel). Live CGNCR is bottom-up; this
# guards the official plan×profile column if a remit ramp lands.
PLAN_SCALE_CHANNEL = 1.20


def month_profile(x: pd.Series, origin: pd.Timestamp, n: int = 5) -> dict[int, float]:
    """Median month-of-year share of the FY total over the last ``n`` complete, Covid-free FYs,
    normalised to sum to one."""
    s = x.loc[:origin].dropna()
    fy = pd.Series(fy_of(s.index), index=s.index)
    tot = s.groupby(fy).transform("sum")
    cnt = s.groupby(fy).transform("size")
    share = (s / tot)[(cnt == 12) & _fy_mask(s.index)]
    prof = {mo: float(share[share.index.month == mo].iloc[-n:].median()) for mo in range(1, 13)}
    z = sum(prof.values())
    return {k: v / z for k, v in prof.items()}


def _prior_fy_outturn(x: pd.Series, fy: int) -> float | None:
    """Complete prior financial-year CGNCR total, or None if incomplete."""
    months = pd.date_range(pd.Timestamp(fy - 1, 4, 1), periods=12, freq="MS")
    v = x.reindex(months)
    if v.isna().any():
        return None
    return float(v.sum())


def plan_profile(x: pd.Series, origin: pd.Timestamp, plan: dict[int, float],
                 h: int = FC_MONTHS) -> np.ndarray:
    """Annual plan × month profile; the FY in progress gets (plan − year-to-date) spread over its
    remaining months in proportion to the profile. NaN where the FY has no plan.

    Scale channel: if plan / prior complete FY ≥ ``PLAN_SCALE_CHANNEL``, spread the
    residual uniformly — do not freeze hist FY shares onto a remit ramp.
    """
    s = x.loc[:origin].dropna()
    prof = month_profile(x, origin)
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = []
    for m in months:
        f = fy_of(m)
        if f not in plan:
            out.append(np.nan)
            continue
        fy_months = pd.date_range(pd.Timestamp(f, 4, 1), periods=12, freq="MS")
        done = [k for k in fy_months if k <= origin]
        ytd = float(s.reindex(done).sum()) if done else 0.0
        rest = [k for k in fy_months if k > origin]
        residual = plan[f] - ytd
        prior = _prior_fy_outturn(s, f)
        scale = (prior is not None and prior > 0
                 and float(plan[f]) >= PLAN_SCALE_CHANNEL * prior)
        if scale or not rest:
            out.append(residual / len(rest) if rest else np.nan)
        else:
            w = np.array([prof[k.month] for k in rest])
            out.append(residual * prof[m.month] / w.sum())
    return np.array(out)


def cgncr_annual_plan(anchors: pd.DataFrame, extra: dict[int, float] | None = None) -> dict[int, float]:
    """DMO plan and illustrative CGNCR(ex) by FY start. Years the remit does not publish take the OBR
    CGNCR ex Network Rail (Table 5.4). A published remit year is kept: it is that OBR figure, or a
    later revision of it."""
    plan: dict[int, float] = {}
    for r in anchors.itertuples(index=False):
        if r.kind in ("plan", "illustrative") and pd.notna(r.cgncr_ex):
            plan.setdefault(int(str(r.fy)[:4]), float(r.cgncr_ex))
    if extra is None:
        from sovereign_cash.uk.obr_tables import cgncr_ex_nr
        extra = cgncr_ex_nr()
    for fy, v in extra.items():
        plan.setdefault(int(fy), float(v))
    return plan


def official_profile(x: pd.Series, origin: pd.Timestamp, anchors: pd.DataFrame,
                     h: int = FC_MONTHS) -> np.ndarray:
    """Annual CGNCR(ex) × month profile. Remit years first, then OBR Table 5.4."""
    return plan_profile(x, origin, cgncr_annual_plan(anchors), h)


def info_date(origin: pd.Timestamp) -> pd.Timestamp:
    """When the origin month's PSF is out (~21 days after month end)."""
    return origin + pd.offsets.MonthBegin(1) + pd.Timedelta(days=20)


def vintage_plan(vint: pd.DataFrame, asof: pd.Timestamp) -> dict[int, float]:
    """Latest comparable remit CGNCR(ex) per FY published on or before ``asof`` (outturns excluded:
    by then the FY is in the data)."""
    v = vint[(vint["vintage_date"] <= asof) & vint["comparable"] & (vint["kind"] != "outturn")]
    return {int(f): float(g["cgncr_ex"].iloc[-1]) for f, g in v.groupby("fy_start")}


def official_vintage_forecast(x: pd.Series, origin: pd.Timestamp, vint: pd.DataFrame,
                              h: int = FC_MONTHS) -> np.ndarray:
    return plan_profile(x, origin, vintage_plan(vint, info_date(origin)), h)


def backtest_cgncr(x: pd.Series, vint: pd.DataFrame | None = None,
                   models: tuple[str, ...] = CGNCR_MODELS + CGNCR_BENCH) -> pd.DataFrame:
    """Walk-forward cumulative h-month sums. With ``vint`` the remit-vintage benchmark is added as
    OFFICIAL (NaN where no comparable vintage covers every target month)."""
    rows = []
    x = x.dropna()
    origins = x.index[(x.index >= pd.Timestamp("2008-01-01"))]
    meths = list(models) + ([OFFICIAL] if vint is not None else [])
    for o in origins:
        fut = x.loc[o + pd.offsets.MonthBegin(1):].iloc[:FC_MONTHS]
        for meth in meths:
            fc = (official_vintage_forecast(x, o, vint) if meth == OFFICIAL
                  else cgncr_forecast(x, o, meth))
            for h in HORIZONS:
                if len(fut) < h:
                    continue
                f = fc[:h]
                rows.append({"origin": o, "h": h, "model": meth,
                             "fc": float(f.sum()) if np.isfinite(f).all() else np.nan,
                             "actual": float(fut.iloc[:h].sum()), "target_end": fut.index[h - 1]})
    bt = pd.DataFrame(rows)
    bt["abs_err"] = (bt["fc"] - bt["actual"]).abs()
    bt["window"] = _window(bt)
    return bt


def _window(bt: pd.DataFrame) -> pd.Series:
    """PRE / OOS need both the targets and the inputs Covid-free; anything touching
    COVID_EST (targets in the window, or the trailing INPUT_MONTHS read at the origin) is COVID."""
    w = pd.Series("other", index=bt.index)
    inputs_hit = (bt["origin"] >= COVID_EST[0]) & \
        (bt["origin"] - pd.DateOffset(months=INPUT_MONTHS - 1) <= COVID_EST[1])
    targets_hit = (bt["target_end"] >= COVID_EST[0]) & (bt["origin"] < COVID_EST[1])
    hit = inputs_hit | targets_hit
    in_pre = (bt["origin"] >= PRE[0]) & (bt["origin"] <= PRE[1])
    w[in_pre & ~hit] = "PRE"
    w[(bt["origin"] >= OOS_START) & ~hit] = "OOS"
    w[hit & (bt["origin"] >= PRE[0])] = "COVID"
    return w


def score_table(bt: pd.DataFrame) -> pd.DataFrame:
    t = (bt[bt["window"].isin(["PRE", "COVID", "OOS"]) & (bt["model"] != OFFICIAL)]
         .groupby(["window", "h", "model"])
         .agg(mae=("abs_err", "mean"), n=("abs_err", "count")).reset_index())
    return t


def official_paired(bt: pd.DataFrame, model: str, bench: str = "smly") -> pd.DataFrame:
    """MAE of the chosen model, smly and the remit-vintage benchmark on the origins where the
    benchmark exists (same sample for all three)."""
    piv = bt[bt["window"].isin(["PRE", "OOS"])].pivot_table(
        index=["window", "h", "origin"], columns="model", values="abs_err")
    if OFFICIAL not in piv:
        return pd.DataFrame()
    piv = piv[[model, bench, OFFICIAL]].dropna()
    out = piv.groupby(level=["window", "h"]).agg(["mean"])
    out.columns = [c[0] for c in out.columns]
    out["n"] = piv.groupby(level=["window", "h"]).size()
    return out.reset_index()


def choose(t: pd.DataFrame, models: tuple[str, ...]) -> str:
    pre = t[(t["window"] == "PRE") & t["model"].isin(models)]
    return str(pre.groupby("model")["mae"].mean().idxmin())


EDGE_BAND = 0.03  # a model must beat the default (plan / flat stock) by more than this on PRE


def mae_at(h_mae: dict[int, float], h: float) -> float:
    """Cumulative-error MAE at horizon ``h`` by linear interpolation over the scored horizons
    (clamped to the scored range)."""
    hs = sorted(k for k, v in h_mae.items() if np.isfinite(v))
    if not hs:
        return float("nan")
    return float(np.interp(h, hs, [h_mae[k] for k in hs]))


def table_mae(t: pd.DataFrame, window: str, model: str) -> dict[int, float]:
    r = t[(t["window"] == window) & (t["model"] == model)]
    return {int(h): float(m) for h, m in zip(r["h"], r["mae"])}


def paired_mae(paired: pd.DataFrame, window: str, col: str) -> dict[int, float]:
    if paired.empty or col not in paired:
        return {}
    r = paired[paired["window"] == window]
    return {int(h): float(m) for h, m in zip(r["h"], r[col])}


def model_beats(model_mae: dict[int, float], default_mae: dict[int, float], h: float,
                band: float = EDGE_BAND) -> bool:
    """True when the model's PRE MAE at ``h`` is more than ``band`` below the default's."""
    m, d = mae_at(model_mae, h), mae_at(default_mae, h)
    return bool(np.isfinite(m) and np.isfinite(d) and m < (1 - band) * d)


# --------------------------------------------------------------------------------------------
# T-bill stock models


def bill_forecast(hist: pd.DataFrame, origin: pd.Timestamp, method: str, need: np.ndarray,
                  h: int = FC_MONTHS) -> np.ndarray:
    """Monthly Δ T-bill stock for origin+1..origin+h.

    need = forecast CGNCR − net gilt cash − NS&I (the gap bills + buffer must fill).
    """
    hh = hist.loc[:origin]
    hh = hh[_est_mask(hh.index)].iloc[-96:]
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    if method == "bill_flat":
        return np.zeros(h)
    seas = hh["net_bills_bn"].groupby(hh.index.month).apply(lambda v: v.iloc[-5:].mean())
    if method == "bill_seasonal":
        return np.array([seas.get(m.month, 0.0) for m in months])
    if method == "bill_absorb":
        # bills take the whole need except the buffer's average month-of-year swing
        bseas = hh["buffer_change_bn"].groupby(hh.index.month).apply(lambda v: v.iloc[-5:].mean())
        return np.array([need[k] + bseas.get(m.month, 0.0) for k, m in enumerate(months)])
    if method == "bill_beta":
        alpha, beta = bill_beta_fit(hh)
        return np.array([alpha[m.month - 1] + beta * need[k] for k, m in enumerate(months)])
    raise ValueError(method)


def bill_beta_fit(hh: pd.DataFrame) -> tuple[np.ndarray, float]:
    """OLS Δbills = α_month + β·(CGNCR − net gilts − NS&I) on the supplied (pre-origin) rows."""
    hn = hh["cash_requirement_bn"] - hh["net_bonds_bn"] - hh["of_nsandi_bn"]
    dummies = np.eye(12)[hh.index.month - 1]
    X = np.column_stack([dummies, hn.to_numpy()])
    y = hh["net_bills_bn"].to_numpy()
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return coef[:12], float(coef[12])


def backtest_bills(hist: pd.DataFrame, cg_method: str) -> pd.DataFrame:
    """Bill stock level at h, origins monthly. CGNCR is the chosen model's forecast; gilts and
    redemptions are the realised ladder (calendar known in advance); NS&I trailing 12m mean."""
    x = hist["cash_requirement_bn"]
    rows = []
    for o in hist.index[hist.index >= pd.Timestamp("2008-01-01")]:
        fut = hist.loc[o + pd.offsets.MonthBegin(1):].iloc[:FC_MONTHS]
        if fut.empty:
            continue
        cg = cgncr_forecast(x, o, cg_method)[:len(fut)]
        ns = hist.loc[:o, "of_nsandi_bn"].iloc[-12:].mean()
        need = cg - fut["net_bonds_bn"].to_numpy() - ns
        s0 = hist.at[o, "bill_stock_bn"]
        for meth in BILL_MODELS + BILL_BENCH:
            d = bill_forecast(hist, o, meth, need, h=len(fut))
            path = s0 + np.cumsum(d)
            for h in HORIZONS:
                if len(fut) < h:
                    continue
                rows.append({"origin": o, "h": h, "model": meth, "fc": float(path[h - 1]),
                             "actual": float(fut["bill_stock_bn"].iloc[h - 1]),
                             "target_end": fut.index[h - 1]})
    bt = pd.DataFrame(rows)
    bt["abs_err"] = (bt["fc"] - bt["actual"]).abs()
    bt["window"] = _window(bt)
    return bt


# --------------------------------------------------------------------------------------------
# Forward


def forward_gilts(hist: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Gross gilt cash and redemptions by month for the forecast window.

    Settled operations already on the tape are used as is. The rest of the current FY is the
    remit's remaining cash (DMO progress report) spread over the forward plan's nominal by
    settlement month; later months are plan nominal × the FY's realised cash/nominal ratio.
    """
    from issuance import uk_remit

    ops = src.gilt_ops_cash()
    ops["month"] = ops["settle"].dt.to_period("M").dt.to_timestamp()
    rp = uk_remit.remit_progress()
    remit_asof = pd.Timestamp(rp["asof"])
    fy_end = pd.Timestamp(rp["fy_end"])
    remaining = (rp["total_planned"] - rp["sales_to_date"]) / 1000.0
    # The progress report counts by operation date (T, settlement T+1).
    op_day = ops["settle"] - pd.offsets.BDay(1)
    after = ops[(op_day > remit_asof) & ops["mechanism"].isin(src.CASH_IN)]
    remaining -= float(after["cash_bn"].sum())
    known = ops[ops["mechanism"].isin(src.CASH_IN)].groupby("month")["cash_bn"].sum()

    plan = src.forward_gilt_plan()
    last_tape = ops["settle"].max()
    plan = plan[plan["op_dt"] > last_tape]
    fy_now = ops[(ops["settle"] > fy_end - pd.DateOffset(years=1)) & ops["mechanism"].isin(src.CASH_IN)]
    ratio = float(fy_now["cash_bn"].sum() / (fy_now["nominal_mn"].sum() / 1000.0)) if len(fy_now) else 1.0
    in_fy = plan["month"] <= fy_end
    nom_fy = plan.loc[in_fy].groupby("month")["nominal_bn"].sum()
    scale = remaining / nom_fy.sum() if nom_fy.sum() > 0 else ratio
    gross = (nom_fy * scale).add(plan.loc[~in_fy].groupby("month")["nominal_bn"].sum() * ratio,
                                 fill_value=0.0)
    gross = gross.add(known[known.index >= months[0]], fill_value=0.0)

    asof = hist.index.max() + pd.offsets.MonthEnd(0)
    red = src.forward_redemptions(pd.Timestamp(asof))
    out = pd.DataFrame(index=months)
    out["bond_gross_bn"] = gross.reindex(months).fillna(0.0)
    out["bond_redemption_bn"] = red.groupby("month")["cash_bn"].sum().reindex(months).fillna(0.0)
    out["bond_redemption_cb_bn"] = red.groupby("month")["cb_cash_bn"].sum().reindex(months).fillna(0.0)
    out["net_bonds_bn"] = out["bond_gross_bn"] - out["bond_redemption_bn"]
    out["net_bonds_private_bn"] = out["net_bonds_bn"] + out["bond_redemption_cb_bn"]
    left = pd.date_range(remit_asof + pd.offsets.MonthBegin(1), fy_end, freq="MS")
    out.attrs.update(remit_asof=str(remit_asof.date()), remit_remaining_bn=remaining,
                     remit_total_bn=rp["total_planned"] / 1000.0, remit_sold_bn=rp["sales_to_date"] / 1000.0,
                     remit_unallocated_bn=rp.get("unallocated", np.nan) / 1000.0,
                     remit_months_left=len(left),
                     remit_window=f"{left[0]:%b-%y}..{left[-1]:%b-%y}" if len(left) else "",
                     cash_per_nominal=ratio, tape_end=str(last_tape.date()))
    return out


def redemption_calendar(origin: pd.Timestamp, months: pd.DatetimeIndex) -> list[dict]:
    """Gilts redeeming in the forecast window: nominal, CRND/DMO and APF holdings, cash split."""
    red = src.forward_redemptions(pd.Timestamp(origin + pd.offsets.MonthEnd(0)))
    red = red[red["month"].isin(months)].sort_values("maturity_dt")
    return [{"month": str(r.month.date()), "date": str(r.maturity_dt.date()), "name": r.name,
             "nominal_bn": r.amount_nominal_mn / 1000.0, "gov_held_bn": r.gov_held_mn / 1000.0,
             "apf_held_bn": r.apf_held_mn / 1000.0, "cash_bn": r.cash_bn, "cb_cash_bn": r.cb_cash_bn,
             "private_cash_bn": r.cash_bn - r.cb_cash_bn} for r in red.itertuples(index=False)]


def remit_other_monthly(planrow: pd.Series, months: pd.DatetimeIndex, fy: int) -> pd.Series:
    """Remit financing items outside gilts, bills and NS&I, as monthly Δ DMO net cash: other
    financing (coinage, unhedged reserves) less the prior-year secondary-purchase financing, spread
    evenly over the FY, and the Sukuk redemption in its month (spread if the month is unknown)."""
    fy_months = pd.date_range(pd.Timestamp(fy, 4, 1), periods=12, freq="MS")
    def num(k):
        v = planrow.get(k, np.nan)
        return 0.0 if pd.isna(v) else float(v)
    even = (num("other_financing") - num("secondary_purchase_fin")) / 12.0
    sukuk = num("sukuk_redemption")
    sm = SUKUK_MONTH.get(fy)
    out = pd.Series(0.0, index=months)
    for m in months:
        if m in fy_months:
            out[m] = even - (sukuk if sm is not None and m == sm else (0.0 if sm is not None else sukuk / 12.0))
    return out


def end_march_closure(net_cash_start: float, plan: dict, fy: dict) -> dict:
    """DMO financing arithmetic for the FY with bills at plan; the shortfall against the plan's
    end-March net cash is what the next remit revision must add (bills or gilts).

    plan / fy keys: cgncr, gilt_sales, redemptions, nsandi, other (other financing − Sukuk −
    secondary-purchase financing); plan also: tbill_debtfin, net_cash_end.
    """
    nc_bills_at_plan = (net_cash_start + fy["gilt_sales"] + plan["tbill_debtfin"] + fy["nsandi"]
                        + fy["other"] - fy["cgncr"] - fy["redemptions"])
    gap = plan["net_cash_end"] - nc_bills_at_plan
    rounding = plan["net_cash_end"] - (net_cash_start + plan["gilt_sales"] + plan["tbill_debtfin"]
                                       + plan["nsandi"] + plan["other"] - plan["cgncr"] - plan["redemptions"])
    parts = {"cgncr": fy["cgncr"] - plan["cgncr"],
             "nsandi": plan["nsandi"] - fy["nsandi"],
             "gilt_sales": plan["gilt_sales"] - fy["gilt_sales"],
             "redemptions": fy["redemptions"] - plan["redemptions"],
             "other": plan["other"] - fy["other"],
             "published_rounding": rounding}
    return {"gap": gap, "tbill_debtfin_required": plan["tbill_debtfin"] + gap, "parts": parts}


def solve_bills(d: np.ndarray, in_fy: np.ndarray, s0: float, target_stock: float,
                fixed: np.ndarray | None = None) -> np.ndarray:
    """Shift the free FY months of a monthly Δbill path evenly so the stock hits ``target_stock`` at
    the last FY month. ``fixed`` months (already observed, e.g. from tenders) and months after the
    FY are left alone."""
    d = np.asarray(d, dtype=float).copy()
    fixed = np.zeros(len(d), dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
    free = np.asarray(in_fy, dtype=bool) & ~fixed
    n = int(free.sum())
    if n:
        d[free] += (target_stock - s0 - d[in_fy].sum()) / n
    return d


def _fy_key(fy: int) -> str:
    return f"{fy}-{str(fy + 1)[2:]}"


def forward(hist: pd.DataFrame, anchors: pd.DataFrame, cg_method: str, bill_method: str,
            ns_method: str = "trail_mean", scores: dict | None = None,
            tender_net: pd.Series | None = None,
            bottomup: np.ndarray | None = None,
            use_bottomup: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """12-month forward frame, the FY-to-date DMO-basis buffer path, and the end-March steer.

    ``scores`` (optional: ``paired`` remit-vintage table, ``t_bl`` bill and ``t_ns`` NS&I score
    tables) sets the defaults: CGNCR follows the published plan × profile unless the model beats the
    remit vintage by more than EDGE_BAND on PRE at the horizon (months left in the FY; 12 after it);
    after March the bill stock is held flat unless the bill model beats flat by EDGE_BAND on PRE at
    h=12. ``tender_net`` pins forecast months whose weekly tenders are complete.
    ``bottomup`` / ``use_bottomup`` write ``cash_requirement_bottomup_bn`` and, when the PRE ship
    gate passes, replace live ``cash_requirement_bn`` (bills still solve to the remit end-March).
    """
    from sovereign_cash.cash_season import forward_months as _fwd_m

    scores = scores or {}
    origin = hist.index.max()
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=_fwd_m(origin), freq="MS")
    x = hist["cash_requirement_bn"]
    fc = pd.DataFrame(index=months)
    fc.index.name = "month"

    fy = fy_of(origin) if origin.month != 3 else fy_of(origin) + 1
    fy_months = pd.date_range(pd.Timestamp(fy, 4, 1), periods=12, freq="MS")
    in_fy = fc.index.isin(fy_months)
    months_left = int(in_fy.sum())

    h_live = len(months)
    cg_model = cgncr_forecast(x, origin, cg_method, h=h_live)
    cg_plan = official_profile(x, origin, anchors, h=h_live)
    paired = scores.get("paired", pd.DataFrame())
    pm, pp = paired_mae(paired, "PRE", cg_method), paired_mae(paired, "PRE", OFFICIAL)
    src_in = "model" if (not pp or model_beats(pm, pp, months_left)) else "plan"
    src_after = "model" if (not pp or model_beats(pm, pp, FC_MONTHS)) else "plan"
    use_plan = np.where(in_fy, src_in == "plan", src_after == "plan") & np.isfinite(cg_plan)
    fc["cash_requirement_model_bn"] = cg_model
    fc["cash_requirement_fc_official_bn"] = cg_plan
    hybrid = np.where(use_plan, cg_plan, cg_model)
    if bottomup is None:
        fc["cash_requirement_bottomup_bn"] = np.nan
    else:
        bu = np.asarray(bottomup, dtype=float)
        if len(bu) != len(fc):
            raise ValueError(f"bottomup length {len(bu)} != forward months {len(fc)}")
        fc["cash_requirement_bottomup_bn"] = bu
    if use_bottomup and bottomup is not None:
        # BU only inside the current remit FY. Past March it loses the January
        # self-assessment seasonal (next-FY sum ~£48bn vs remit ~£145bn /
        # seasonal ~£136bn) and would understate supply into debt/build.
        fc["cash_requirement_bn"] = np.where(in_fy, bu, hybrid)
        src_in = "bottom_up"
        # after-FY source stays model/plan hybrid (src_after already set)
        fc["cgncr_from_plan"] = np.where(in_fy, False, use_plan)
    else:
        fc["cash_requirement_bn"] = hybrid
        fc["cgncr_from_plan"] = use_plan
    fc["cgncr_smly_bn"] = cgncr_forecast(x, origin, "smly", h=h_live)
    g = forward_gilts(hist, months)
    fc = fc.join(g)
    fc["gilts_assumed"] = ~in_fy  # beyond the current remit: forward-plan nominal, no remit yet
    fc["of_nsandi_bn"] = cgncr_forecast(hist["of_nsandi_bn"], origin, ns_method, h=h_live)

    a = anchors.set_index(["fy", "kind"])
    planrow = a.loc[(_fy_key(fy), "plan")]
    ytd_idx = [m for m in fy_months if m <= origin]
    fc["remit_other_fin_bn"] = remit_other_monthly(planrow, months, fy)
    fc["other_fin_bn"] = fc["of_nsandi_bn"] + fc["remit_other_fin_bn"]

    def p(k):
        v = planrow.get(k, np.nan)
        return 0.0 if pd.isna(v) else float(v)
    plan = {"cgncr": p("cgncr_ex"), "gilt_sales": p("gilt_sales"), "redemptions": p("gilt_redemptions"),
            "nsandi": p("nsandi_net") + p("nsandi_green"),
            "other": p("other_financing") - p("sukuk_redemption") - p("secondary_purchase_fin"),
            "tbill_debtfin": p("tbill_net_debtfin"), "net_cash_end": p("net_cash_end_march")}
    prev_key = (_fy_key(fy - 1), "outturn")
    net_cash_start = (float(a.loc[prev_key, "net_cash_end_march"]) if prev_key in a.index
                      else plan["net_cash_end"] - p("financing_adjustment"))
    H = hist.reindex(ytd_idx)
    # Live financing board closes to the published remit. Soft-pin α on the BU leaf
    # is for scored PRE accuracy only — after Budget / remit revision, YE = plan.
    live_pin = False
    live_pin_gap = 0.0
    if use_bottomup and months_left > 0 and np.isfinite(plan["cgncr"]):
        ytd_cg0 = float(H["cash_requirement_bn"].sum()) if ytd_idx else 0.0
        rest = float(fc.loc[in_fy, "cash_requirement_bn"].sum())
        live_pin_gap = plan["cgncr"] - ytd_cg0 - rest
        if abs(live_pin_gap) > 1e-9:
            fc.loc[in_fy, "cash_requirement_bn"] = (
                fc.loc[in_fy, "cash_requirement_bn"].to_numpy() + live_pin_gap / months_left
            )
            live_pin = True
            src_in = "bottom_up_remit"
            fc.loc[in_fy, "cgncr_from_plan"] = True
    F = fc[in_fy]
    fy_tot = {"cgncr": H["cash_requirement_bn"].sum() + F["cash_requirement_bn"].sum(),
              "gilt_sales": H["bond_gross_bn"].sum() + F["bond_gross_bn"].sum(),
              "redemptions": H["bond_redemption_bn"].sum() + F["bond_redemption_bn"].sum(),
              "nsandi": H["of_nsandi_bn"].sum() + F["of_nsandi_bn"].sum(),
              "other": plan["other"]}
    clo = end_march_closure(net_cash_start, plan, fy_tot)
    cg_model_fy = float(H["cash_requirement_bn"].sum() + fc.loc[in_fy, "cash_requirement_model_bn"].sum())
    clo_model = end_march_closure(net_cash_start, plan, {**fy_tot, "cgncr": cg_model_fy})

    mar_prev = pd.Timestamp(fy, 3, 1)
    debtfin_prev = float(hist.at[mar_prev, "bill_stock_debtfin_bn"])
    cm_prev = float(hist.at[mar_prev, "bill_stock_bn"]) - debtfin_prev
    cm_end = cm_prev  # cash-management bills at end-March held at last FY-end (in-year split unpublished)
    stock_target = debtfin_prev + clo["tbill_debtfin_required"] + cm_end
    s0 = float(hist["bill_stock_bn"].iloc[-1])
    need = (fc["cash_requirement_bn"] - fc["net_bonds_bn"] - fc["of_nsandi_bn"]).to_numpy()
    d_model = bill_forecast(hist, origin, bill_method, need, h=len(need))
    fc["net_bills_model_bn"] = d_model
    t_bl = scores.get("t_bl")
    bill_after = "model"
    if t_bl is not None and not t_bl.empty:
        beats = model_beats(table_mae(t_bl, "PRE", bill_method), table_mae(t_bl, "PRE", "bill_flat"), FC_MONTHS)
        bill_after = "model" if beats else "flat"
    d = d_model.copy()
    if bill_after == "flat":
        d[~in_fy] = 0.0
    tn = pd.Series(dtype=float) if tender_net is None else tender_net
    pinned = fc.index.isin(tn.index)
    if pinned.any():
        d[pinned] = tn.reindex(fc.index[pinned]).to_numpy()
    fc["net_bills_pinned"] = pinned
    fc["bills_assumed"] = ~in_fy
    fc["net_bills_bn"] = solve_bills(d, in_fy, s0, stock_target, fixed=pinned)
    fc["bill_gross_bn"] = np.nan
    fc["bill_redemption_bn"] = np.nan
    fc["buffer_change_bn"] = (-fc["cash_requirement_bn"] + fc["net_bonds_bn"]
                              + fc["net_bills_bn"] + fc["other_fin_bn"])
    fc["bill_stock_implied_bn"] = s0 + fc["net_bills_bn"].cumsum()
    fc["is_forecast"] = True

    # DMO-basis buffer: end-March outturn net cash + cash-management bills, rolled with the identity
    # but with other_fin restricted to the remit's items (NS&I + other financing − Sukuk − secondary).
    ytd_other = remit_other_monthly(planrow, pd.DatetimeIndex(ytd_idx), fy) if ytd_idx else pd.Series(dtype=float)
    dmo = pd.DataFrame(index=pd.DatetimeIndex([mar_prev] + ytd_idx, name="month"))
    d_ytd = (-H["cash_requirement_bn"] + H["net_bonds_bn"] + H["net_bills_bn"] + H["of_nsandi_bn"]
             + ytd_other.reindex(H.index).fillna(0.0)) if ytd_idx else pd.Series(dtype=float)
    dmo["buffer_level_dmo_basis_bn"] = net_cash_start + cm_prev + pd.concat(
        [pd.Series([0.0], index=[mar_prev]), d_ytd]).cumsum()
    dmo["remit_other_fin_bn"] = ytd_other.reindex(dmo.index)
    b0 = float(dmo["buffer_level_dmo_basis_bn"].iloc[-1])
    fc["buffer_level_implied_bn"] = b0 + fc["buffer_change_bn"].cumsum()
    mar = pd.Timestamp(fy + 1, 3, 1)
    fc["dmo_net_cash_bn"] = np.nan
    if mar in fc.index:
        fc.at[mar, "dmo_net_cash_bn"] = fc.at[mar, "buffer_level_implied_bn"] - cm_end

    proxy, _ = anchor_buffer(hist, anchors)
    basis_items = ["of_dep_by_others_bn", "of_nram_nr_bn", "of_gilt_vs_ons_bn", "of_bill_vs_ons_bn",
                   "of_unexplained_bn"]
    basis = {c.replace("of_", "").replace("_bn", ""): float(H[c].sum()) for c in basis_items} if ytd_idx else {}
    basis["less_remit_other_ytd"] = -float(ytd_other.sum()) if ytd_idx else 0.0

    prof = month_profile(x, origin)
    ns_prof = month_profile(hist["of_nsandi_bn"], origin)
    done_share = sum(prof[m.month] for m in ytd_idx)
    ns_done_share = sum(ns_prof[m.month] for m in ytd_idx)
    rest_off = fc.loc[in_fy, "cash_requirement_fc_official_bn"].sum()
    ytd_cg = float(H["cash_requirement_bn"].sum())
    ytd_ns = float(H["of_nsandi_bn"].sum())
    steer = {
        "fy": _fy_key(fy),
        "ytd_through": str(origin.date()),
        "cgncr_ex_plan_bn": plan["cgncr"],
        "cgncr_ex_ytd_bn": ytd_cg,
        "cgncr_ex_plan_profile_ytd_bn": plan["cgncr"] * done_share,
        "cgncr_ex_fc_fy_bn": float(fy_tot["cgncr"]),
        "cgncr_ex_model_fy_bn": cg_model_fy,
        "cgncr_source_in_fy": src_in,
        "cgncr_source_after_fy": src_after,
        "cgncr_live_pin_to_remit": live_pin,
        "cgncr_live_pin_gap_bn": round(float(live_pin_gap), 3),
        "months_left_in_fy": months_left,
        "next_revision_gap_cgncr_model_bn": clo_model["gap"],
        "bill_after_fy": bill_after,
        "bill_pinned_months": [str(m.date()) for m in fc.index[pinned]],
        "cgncr_ex_official_profile_fy_bn": float(ytd_cg + rest_off),
        "nsandi_remit_bn": plan["nsandi"],
        "nsandi_ytd_bn": ytd_ns,
        "nsandi_remit_profile_ytd_bn": plan["nsandi"] * ns_done_share,
        "nsandi_fy_model_bn": float(fy_tot["nsandi"]),
        "nsandi_method": ns_method,
        "gilt_sales_fy_bn": float(fy_tot["gilt_sales"]),
        "gilt_sales_plan_bn": plan["gilt_sales"],
        "redemptions_fy_bn": float(fy_tot["redemptions"]),
        "redemptions_plan_bn": plan["redemptions"],
        "remit_other_fy_bn": plan["other"],
        "net_cash_start_bn": net_cash_start,
        "net_cash_plan_end_march_bn": plan["net_cash_end"],
        "net_cash_model_end_march_bn": float(fc.at[mar, "dmo_net_cash_bn"]) if mar in fc.index else None,
        "next_revision_gap_bn": clo["gap"],
        "next_revision_gap_parts_bn": clo["parts"],
        "next_revision_gap_nsandi_at_remit_bn": clo["gap"] - clo["parts"]["nsandi"],
        "tbill_debtfin_plan_bn": plan["tbill_debtfin"],
        "tbill_debtfin_required_bn": clo["tbill_debtfin_required"],
        "tbill_debtfin_stock_end_march_plan_bn": debtfin_prev + plan["tbill_debtfin"],
        "tbill_debtfin_stock_end_march_required_bn": debtfin_prev + clo["tbill_debtfin_required"],
        "cash_mgmt_bills_end_march_assumed_bn": cm_end,
        "bill_stock_implied_end_march_bn": float(fc.at[mar, "bill_stock_implied_bn"]) if mar in fc.index else None,
        "buffer_proxy_origin_bn": float(proxy.iloc[-1]),
        "buffer_dmo_basis_origin_bn": b0,
        "buffer_basis_gap_bn": float(proxy.iloc[-1]) - b0,
        "buffer_basis_gap_parts_bn": basis,
        "note": ("End-March closes inside the identity: forward other_fin = NS&I (model) + remit other "
                 "items; FY bills are solved so DMO net cash = plan. next_revision_gap_bn = debt-financing "
                 "bills needed beyond plan = extra bills/gilts at the next remit revision (+ = more financing)."),
    }
    steer.update({f"gilt_{k}": v for k, v in g.attrs.items()})
    hh = hist[_est_mask(hist.index)].iloc[-96:]
    steer["bill_beta_at_origin"] = bill_beta_fit(hh)[1]
    # Error band on the revision gap: OOS MAE of each driver over the months left, interpolated.
    t_ns = scores.get("t_ns")
    ns_err = mae_at(table_mae(t_ns, "OOS", ns_method), months_left) if t_ns is not None else float("nan")
    if src_in == "bottom_up":
        cg_err = float("nan")
    else:
        cg_col = OFFICIAL if src_in == "plan" else cg_method
        cg_err = mae_at(paired_mae(paired, "OOS", cg_col), months_left)
    steer["gap_err_nsandi_bn"] = ns_err
    steer["gap_err_cgncr_bn"] = cg_err
    steer["gap_err_bn"] = float(np.sqrt(np.nansum([ns_err ** 2, cg_err ** 2]))) \
        if np.isfinite(ns_err) or np.isfinite(cg_err) else float("nan")
    return fc, dmo, steer


def money_market_panel(hist: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mm = src.sonia_bank_monthly()
    if mm.empty:
        return pd.DataFrame(), {"note": "SONIA workbook missing"}
    p = hist[["net_bills_bn", "bill_stock_bn", "buffer_change_bn"]].join(mm, how="outer")
    p = p.loc[p.index >= hist.index.min()]
    p["bill_3m_yield_pct"] = src.bill_yield_monthly().reindex(p.index)
    p["bill_3m_sonia_bp"] = (p["bill_3m_yield_pct"] - p["sonia_pct"]) * 100.0
    p["d_sonia_bank_bp"] = p["sonia_bank_bp"].diff()
    p["d3_bill_stock_bn"] = p["bill_stock_bn"].diff(3)
    p["d3_sonia_bank_bp"] = p["sonia_bank_bp"].diff(3)
    stats = {}
    corr_cols = ["net_bills_bn", "bill_stock_bn", "buffer_change_bn", "d_sonia_bank_bp",
                 "d3_bill_stock_bn", "d3_sonia_bank_bp"]
    for name, sub in (("2000+", p), ("2009-2021", p.loc["2009":"2021"]), ("2022+", p.loc["2022":])):
        sub = sub.dropna(subset=corr_cols)
        if len(sub) < 12:
            continue
        q = sub.iloc[::3]
        stats[name] = {
            "n": int(len(sub)),
            "corr_dbill_dspread": float(sub["net_bills_bn"].corr(sub["d_sonia_bank_bp"])),
            "corr_d3bill_d3spread": float(q["d3_bill_stock_bn"].corr(q["d3_sonia_bank_bp"])),
            "n_3m": int(len(q)),
            "corr_dbuffer_dspread": float(sub["buffer_change_bn"].corr(sub["d_sonia_bank_bp"])),
        }
    return p, stats
