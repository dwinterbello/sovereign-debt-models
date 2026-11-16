"""Germany: federal cash framework (history identity, deficit model, Bubill path, scores).

The buffer
    The Bundesbank pays 0% on domestic government deposits from 1 Oct 2023, so federal cash
    sits in money-market placements, not at the central bank; the Bundesbank account is
    liquidity-neutral. The observable buffer is the Finanzagentur stock of
    ``Kassenverstärkungskredite und Geldanlagen`` with the sign flipped (net placements =
    Geldanlagen − cash-management borrowing). The Bundesbank balance is not published monthly; its
    change lands in ``other_fin_gap_bn``. Net placements are the flex variable in the forecast:
    Bubills follow the published quarterly calendar; after it the stock is held at the
    calendar-end level on average while a hist month-dummy + β·need path (demeaned) lets
    cash seasonals show in Bubills rather than dumping every residual onto placements.

History identity (EUR bn, inflows to the buffer +), Jan-2011 → last Finanzagentur month:

    Δnet placements = −cash requirement + net coupon bonds + net bills + other_fin
    cash requirement = core budget + special funds with own borrowing authority
    net bills        = net Bubills + Δ money-market budget financing (year-end bridge, repaid in January)
    other_fin        = Δ(Schuldschein + other loans)   [debt]  + gap

core               −monthly change of the core-budget YTD Finanzierungssaldo (BMF), deficit +.
special funds      Δ debt of FMS, ITF, WSF, Bundeswehr, Infrastruktur/Klimaneutralität, RSF
                   (Finanzagentur), i.e. their net borrowing; transfers to/from the core budget
                   without a market transaction are excluded.
bonds, Bubills     Finanzagentur gross borrowing / redemptions (market, own holdings excluded).
gap                Bundesbank account, core Kassenmittel timing, coin, reserves, special-fund
                   cash borrowed but not yet spent. Reported, never zeroed.

Forecast (from the month after the last Finanzagentur month to 12 months after the last BMF month):
core               observed BMF months, then the open budget year closes spending to its Soll and
                   revenue to its Soll, each times its own outturn/plan ratio. A later year in the
                   window keeps the balance haircut on that year's draft. Years whose unexplained
                   financing exceeds ``HC_GAP_MAX`` are left out of the balance haircut (their core
                   outturn misses cash that left the buffer) but stay in its band.
special funds      Infrastruktur + Bundeswehr: when the Wirtschaftsplan sits inside the scale of
                   recent complete FA years, rest-of-year uses hist month shares (BW is H2-heavy).
                   When plan is a scale channel (≫ those years — defence / fund ramp), YE = plan
                   and the residual is flat: do not freeze last year's Dec spike onto a new
                   level. Else flat YTD pace if no usable shares. Other funds (FMS/ITF/WSF/RSF) 0.
coupons            Finanzagentur tape where the month is on it, then the issuance forward calendar
                   (``issuance/cache/forecast_priced.parquet``: published quarterly calendar, then a
                   pattern sized on the borrowing plan), scaled to market placement by the
                   trailing ratio of Finanzagentur market gross to tape volume; redemptions = the
                   tape's maturity ladder (plus pre-tape tranches) scaled the same way.
scenarios          'model' (above) and 'plan' (core Soll/draft without haircut + special funds at
                   plan pace). Issuance is the same in both, so the difference lands on placements.
Bubills            tape, then the published quarterly calendar (gross, maturities); after it
                   α_month + β·(cash − coupons − other debt) from hist OLS, demeaned so the
                   stock ends where the calendar left it. Money-market budget financing 0.
other debt, gap    trailing-12m mean of Δ(Schuldschein + other loans); gap = backtest-selected rule.
placements         close the identity (the flex variable).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.de import fetch as F

COUPON = ["bund", "bobl", "schatz", "linker", "green", "extra_iss", "other_sec"]
OTHER_DEBT = ["ssd", "other_loans"]
SV = ["sv_fms", "sv_itf", "sv_wsf", "sv_bw", "sv_sik", "sv_rsf"]
SV_PLAN = {"sv_sik": "infrastruktur", "sv_bw": "bundeswehr"}
START = pd.Timestamp("2011-01-01")
OPS = F.ROOT.parent / "issuance" / "cache" / "ops_de.parquet"
FWD = F.ROOT.parent / "issuance" / "cache" / "forecast_priced.parquet"
PLAN = F.DATA / "plan.json"
BUBILL_CAL = F.DATA / "bubill_calendar.json"
COVID = (2020, 2021)
HC_MIN_PLAN = 20.0
# |annual unexplained financing| above this (bn) means the core outturn misses cash that left the
# buffer (2024: −37bn spread over every month while Kassenmittel ran down), so the year's
# outturn ÷ plan is not an execution rate.
HC_GAP_MAX = 20.0
SV_LABEL = {"sv_fms": "FMS", "sv_itf": "ITF", "sv_wsf": "WSF", "sv_bw": "Bundeswehr",
            "sv_sik": "Infrastruktur und Klimaneutralität", "sv_rsf": "RSF"}
# Tranches issued before the auction tape starts (1999), EUR bn of issuance volume not on it.
# 6.50% Bund 4 Jul 2027: Finanzagentur factsheet issuance volume 13.75bn; the tape carries only
# the 2.5bn own-holding reopening of 16 Apr 2020.
PRE_TAPE = {"DE0001135044": ("2027-07-04", 11.25)}


def _monthly(ytd: pd.DataFrame) -> pd.DataFrame:
    out = ytd.diff()
    jan = ytd.index.month == 1
    out[jan] = ytd[jan]
    return out


def load_plan() -> dict[int, dict]:
    return {int(k): v for k, v in json.loads(PLAN.read_text())["years"].items()}


# --------------------------------------------------------------------------------------
# Inputs


def core_budget() -> pd.DataFrame:
    """Core-budget YTD (EUR bn) 2011 →: SDDS archive PDF to Nov-2016, Monatsbericht pages after.

    A December missing from both is filled with the full-year Ist the next year's issues carry.
    """
    sdds = F.parse_sdds_pdf(F.fetch_sdds_pdf())
    mb = F.load_bmf_mb()
    keep = ["ytd_expenditure", "ytd_revenue", "ytd_fin_balance", "kassenmittel_pos"]
    df = pd.concat([sdds[[c for c in keep if c in sdds]], mb[[c for c in keep + ["soll_fin_balance", "soll_year"] if c in mb]]])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["source"] = np.where(df.index.isin(mb.index), "bmf_mb", "sdds_pdf")
    for yr in range(df.index.min().year, df.index.max().year):
        dec = pd.Timestamp(yr, 12, 1)
        if dec in df.index and pd.notna(df.at[dec, "ytd_fin_balance"]):
            continue
        nxt = mb[(mb.index.year == yr + 1)]
        if nxt.empty or "ist_prev_fin_balance" not in nxt:
            continue
        nxt = nxt.dropna(subset=["ist_prev_fin_balance"])
        if nxt.empty:
            continue
        r = nxt.iloc[-1]
        df.loc[dec, ["ytd_expenditure", "ytd_revenue", "ytd_fin_balance"]] = [
            r.get("ist_prev_expenditure"), r.get("ist_prev_revenue"), r.get("ist_prev_fin_balance")]
        df.loc[dec, "source"] = "bmf_mb (next-year Ist)"
    return df.sort_index()


def soll_final(core: pd.DataFrame) -> dict[int, float]:
    """Last printed −Soll Finanzierungssaldo of each year (incl. supplementary budgets)."""
    s = core.dropna(subset=["soll_fin_balance", "soll_year"])
    s = s[s["soll_year"].astype(int) == s.index.year]
    return {int(y): -float(g["soll_fin_balance"].iloc[-1]) for y, g in s.groupby(s.index.year)}


def haircuts(cash: pd.Series, core: pd.DataFrame, gap: pd.Series | None = None) -> pd.DataFrame:
    """Core outturn ÷ final Soll per completed year.

    ``in_band`` = non-Covid with plan ≥ HC_MIN_PLAN; ``eligible`` (central haircut) additionally
    needs |annual unexplained financing| ≤ HC_GAP_MAX when ``gap`` is given.
    """
    sf = soll_final(core)
    rows = []
    for y, plan in sf.items():
        blk = cash.loc[f"{y}-01-01":f"{y}-12-01"]
        if len(blk) != 12 or blk.isna().any():
            continue
        out = float(blk.sum())
        g = gap.loc[f"{y}-01-01":f"{y}-12-01"] if gap is not None else None
        gsum = float(g.sum()) if g is not None and len(g) == 12 else np.nan
        in_band = y not in COVID and plan >= HC_MIN_PLAN
        rows.append({"year": y, "plan": plan, "outturn": out, "ratio": out / plan if plan else np.nan,
                     "gap": gsum, "in_band": in_band,
                     "eligible": in_band and not (np.isfinite(gsum) and abs(gsum) > HC_GAP_MAX)})
    cols = ["plan", "outturn", "ratio", "gap", "in_band", "eligible"]
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame(columns=cols)


def haircut_at(hc: pd.DataFrame, year: int) -> tuple[float, list[int]]:
    """Mean outturn/plan over eligible years before ``year``; 1.0 when none."""
    e = hc[(hc.index < year) & hc["eligible"].astype(bool)] if not hc.empty else hc
    return (float(e["ratio"].mean()), list(map(int, e.index))) if len(e) else (1.0, [])


def soll_line_maps(mb: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Last printed spending and revenue Soll of each year."""
    out = {}
    for col in ("soll_expenditure", "soll_revenue"):
        by_year = {}
        if col not in mb:
            out[col] = by_year
            continue
        for yr, g in mb.groupby(mb.index.year):
            g = g.dropna(subset=[col, "soll_year"])
            g = g[g["soll_year"] == yr]
            if len(g):
                by_year[int(yr)] = float(g.iloc[-1][col])
        out[col] = by_year
    return out


def line_ratio(monthly: pd.Series, plan_by_year: dict[int, float], year: int, k: int = 5
               ) -> tuple[float, float, float, list[int]]:
    """Mean, low and high of outturn/plan on complete years before ``year``. Covid is left out."""
    ratios, years = [], []
    for yr in range(year - 1, year - 12, -1):
        if yr in COVID or yr not in plan_by_year or plan_by_year[yr] <= 0:
            continue
        block = monthly.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if len(block) != 12 or block.isna().any():
            continue
        ratios.append(float(block.sum()) / plan_by_year[yr])
        years.append(yr)
        if len(ratios) == k:
            break
    if not ratios:
        return 1.0, 1.0, 1.0, []
    return float(np.mean(ratios)), float(min(ratios)), float(max(ratios)), years


def open_year_core(exp: pd.Series, rev: pd.Series, origin: pd.Timestamp, h: int,
                   plan_e: float, plan_r: float,
                   re: tuple[float, float, float], rr: tuple[float, float, float],
                   ) -> dict[str, pd.Series] | None:
    """Rest of the budget year: spending and revenue each close to Soll × their own ratio.

    The low path is the smaller deficit (low spending ratio, high revenue ratio). Months in a
    later calendar year are left to the caller.
    """
    from sovereign_cash.yoy_components import _month_share, close_to_plan

    sh_e, sh_r = _month_share(exp, origin), _month_share(rev, origin)
    if sh_e is None or sh_r is None:
        return None
    paths = {}
    for name, er, rr_ in (
        ("central", re[0], rr[0]),
        ("lo", re[1], rr[2]),
        ("hi", re[2], rr[1]),
    ):
        fe = close_to_plan(exp, plan_e, er, origin, h, sh_e)
        fr = close_to_plan(rev, plan_r, rr_, origin, h, sh_r)
        paths[name] = fe - fr
    return paths


def haircut_band(hc: pd.DataFrame, year: int) -> tuple[float, float]:
    """(min, max) outturn/plan over in-band years before ``year`` (incl. those left out of the centre)."""
    col = "in_band" if "in_band" in hc else "eligible"
    e = hc[(hc.index < year) & hc[col].astype(bool)] if not hc.empty else hc
    if not len(e):
        c = haircut_at(hc, year)[0]
        return c, c
    return float(e["ratio"].min()), float(e["ratio"].max())


def soll_by_origin(core: pd.DataFrame, hc: pd.DataFrame | None = None) -> dict[pd.Timestamp, dict]:
    """Annual target known at each origin: −Soll Finanzierungssaldo of that year (+ haircut 'hc')."""
    out = {}
    if "soll_fin_balance" not in core:
        return out
    for t, r in core.iterrows():
        if pd.notna(r.get("soll_fin_balance")) and pd.notna(r.get("soll_year")) and int(r["soll_year"]) == t.year:
            d = {t.year: -float(r["soll_fin_balance"])}
            if hc is not None:
                d["hc"] = haircut_at(hc, t.year)[0]
            out[t] = d
    return out


# --------------------------------------------------------------------------------------
# History


def build_history() -> tuple[pd.DataFrame, dict]:
    fa = F.load_finanzagentur()
    stock, gm, rm = fa["stock"], _monthly(fa["gross_ytd"]), _monthly(fa["redemption_ytd"])
    core = core_budget()
    cash = -C.monthly_from_ytd(core["ytd_fin_balance"])
    idx = pd.date_range(START, stock.index.max(), freq="MS")
    h = pd.DataFrame(index=idx)
    h["cash_requirement_core_bn"] = cash.reindex(idx)
    # Debt moved between a special fund and the core budget without a market transaction (WSF
    # wind-up into the core, Jan-2024: WSF −72.7, core +76.9, total +5.5) is not borrowing.
    dsv = stock[SV].diff()
    big = dsv.abs().gt(20) & stock["total"].diff().abs().lt(0.25 * dsv.abs().max(axis=1)).to_numpy()[:, None]
    transfers = [{"month": f"{dsv.index[i]:%Y-%m}", "fund": SV[j], "label": SV_LABEL[SV[j]],
                  "bn": round(float(dsv.iat[i, j]), 1)}
                 for i, j in zip(*np.where(big.to_numpy()))]
    dsv = dsv.mask(big, 0.0)
    h["cash_requirement_sv_bn"] = dsv.sum(axis=1).reindex(idx)
    for col, name in SV_PLAN.items():
        h[f"cash_requirement_sv_{name}_bn"] = dsv[col].reindex(idx)
    h["cash_requirement_sv_other_bn"] = h["cash_requirement_sv_bn"] - sum(
        h[f"cash_requirement_sv_{n}_bn"].fillna(0.0) for n in SV_PLAN.values())
    h["cash_requirement_bn"] = h["cash_requirement_core_bn"] + h["cash_requirement_sv_bn"]
    h["bond_gross_bn"] = gm[COUPON].sum(axis=1).reindex(idx)
    h["bond_redemption_bn"] = rm[COUPON].sum(axis=1).reindex(idx)
    h["net_bonds_bn"] = h["bond_gross_bn"] - h["bond_redemption_bn"]
    h["bubill_gross_bn"] = gm["bubill"].reindex(idx)
    h["bubill_redemption_bn"] = rm["bubill"].reindex(idx)
    h["net_bubills_bn"] = h["bubill_gross_bn"] - h["bubill_redemption_bn"]
    h["mm_budget_gross_bn"] = gm["mm_budget"].reindex(idx).fillna(0.0)
    h["mm_budget_redemption_bn"] = rm["mm_budget"].reindex(idx).fillna(0.0)
    h["net_mm_budget_bn"] = stock["mm_budget"].diff().reindex(idx)
    h["bill_gross_bn"] = h["bubill_gross_bn"] + h["mm_budget_gross_bn"]
    h["bill_redemption_bn"] = h["bubill_redemption_bn"] + h["mm_budget_redemption_bn"]
    h["net_bills_bn"] = h["net_bubills_bn"] + h["net_mm_budget_bn"]
    h["buffer_level_bn"] = -stock["kvk_net"].reindex(idx)
    h["buffer_change_bn"] = (-stock["kvk_net"]).diff().reindex(idx)
    h["other_fin_debt_bn"] = stock[OTHER_DEBT].sum(axis=1).diff().reindex(idx)
    h["other_fin_bn"] = h["buffer_change_bn"] + h["cash_requirement_bn"] - h["net_bonds_bn"] - h["net_bills_bn"]
    h["other_fin_gap_bn"] = h["other_fin_bn"] - h["other_fin_debt_bn"]
    h["bill_stock_bn"] = stock["bubill"].reindex(idx)
    h["mm_budget_stock_bn"] = stock["mm_budget"].reindex(idx)
    h["coupon_stock_bn"] = stock[COUPON].sum(axis=1).reindex(idx)
    h["core_kassenmittel_bn"] = core["kassenmittel_pos"].reindex(idx)
    dtot = stock["total"].diff().reindex(idx)
    h["fa_debt_identity_bn"] = dtot - (h["net_bonds_bn"] + h["net_bills_bn"] + h["other_fin_debt_bn"])
    h = h.dropna(subset=["cash_requirement_core_bn", "net_bonds_bn", "buffer_change_bn"])
    h["is_forecast"] = False
    h["cash_observed"] = True
    h["fc_source"] = "history"
    meta = {"core": core, "fa": fa, "cash_full": cash, "sv_transfers": transfers}
    return h, meta


# --------------------------------------------------------------------------------------
# Scores


def pre_verdict(bt: pd.DataFrame, model: str, rival: str) -> dict:
    """PRE MAE of ``model`` vs ``rival`` on ``model``'s own origins, per horizon."""
    b = bt.assign(window=C.window_of(bt), err=(bt["fc"] - bt["actual"]).abs())
    b = b[b["window"] == "PRE"]
    out = {}
    for h, g in b.groupby("h"):
        p = g.pivot_table(index="origin", columns="model", values="err")
        if model in p and rival in p:
            q = p[[model, rival]].dropna()
            if len(q):
                out[int(h)] = {"n": int(len(q)), model: float(q[model].mean()), rival: float(q[rival].mean())}
    return out


def deficit_scores(cash: pd.Series, targets: dict, hc: pd.DataFrame | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """Core cash-requirement models; production = PRE winner, or the plan-anchored model a priori.

    Also scores ``sx_level_exog`` (component SARIMAX + HICP/NGDP, interest from the
    ISIN book). It can win the PRE comparison for the note, but production stays
    ``planhc_3`` while the debt-brake / special-fund ramp is in the plan.

    Returns (backtest, scores, production model, reason, PRE winner among data-selectable models).
    """
    from sovereign_cash.monthly_components import append_sx_bt

    models = C.model_list(has_target=True, has_haircut=True)
    bt = C.backtest(cash.dropna(), models, targets_asof=targets)
    bt = append_sx_bt(bt, "de", mode="level", with_exog=True, name="sx_level_exog")
    sc = C.score_table(bt)
    pre_winner, pre_why = C.choose(sc)
    pre = sc[(sc["window"] == "PRE") & (sc["model"] == "planhc_3") & (sc["h"] == 12)]
    n_plan = int(pre["n"].max()) if len(pre) else 0
    if n_plan >= C.MIN_PRE and pre_winner.startswith("planhc"):
        return bt, sc, pre_winner, pre_why, pre_winner
    model = "planhc_3"
    v = pre_verdict(bt, model, pre_winner)
    lost = [h for h, r in v.items() if r[model] > r[pre_winner]]
    h12 = v.get(12)
    res = (f"In PRE ({n_plan} h=12 windows, below the {C.MIN_PRE} needed to select) it loses to {pre_winner} at "
           f"{'every horizon' if len(lost) == len(v) else f'{len(lost)} of {len(v)} horizons'}"
           + (f" (h=12 MAE {h12[model]:.1f} vs {h12[pre_winner]:.1f} on the same windows)" if h12 else "") + ".")
    over = []
    if hc is not None and not hc.empty:
        over = [int(y) for y, r in hc.iterrows() if r["plan"] > 0 and r["outturn"] < 0]
    if over:
        res += f" The {over[0]}–{over[-1]} plans printed deficits that closed as surpluses, which is what the haircut corrects."
    why = (f"{model} = Soll × execution haircut, not PRE-gated (chosen a priori): trailing models cannot see "
           f"the debt-brake reform and special-fund ramp the plan carries. {res} OOS is reported, not used "
           f"for the choice. Once the bottom-up ship clears, {model} is only the seasonal fallback: the live open-year "
           f"core is the bottom-up path and later years the component model.")
    return bt, sc, model, why, pre_winner


def placement_backtest(h: pd.DataFrame, cash: pd.Series, model: str, targets: dict) -> pd.DataFrame:
    """Net placements at t+h implied at origin t, scored against the Finanzagentur stock.

    Mirrors the forward: core from the production model (target as known at t), plan funds
    (Infrastruktur, Bundeswehr) and other debt at their trailing-12m pace, other funds 0, coupon
    gross = same month last year, coupon
    redemptions realised (the maturity schedule; buybacks/own-holding sales leak in), Bubills flat.
    Variants: gap = 0 ('gap0') or month-of-year mean of the last 5 years ('gapseas'). Benchmark:
    flat placements ('flat').
    """
    rows = []
    for t in h.index:
        if t < START + pd.DateOffset(years=5):
            continue
        fc = C.forecast(cash.dropna(), t, 12, model, targets=targets.get(t))
        if fc is None:
            continue
        past = h.loc[:t]
        plan_sv = sum(past[f"cash_requirement_sv_{n}_bn"].fillna(0.0) for n in SV_PLAN.values())
        sv12 = float(plan_sv.iloc[-12:].mean())
        od12 = float(past["other_fin_debt_bn"].iloc[-12:].mean())
        gseas = past["other_fin_gap_bn"].iloc[-60:].groupby(past.index[-60:].month).mean()
        s0 = float(h.at[t, "buffer_level_bn"])
        cum = {"gap0": s0, "gapseas": s0}
        for i, m in enumerate(fc.index, start=1):
            if m not in h.index:
                break
            ly = m - pd.DateOffset(years=1)
            gross = float(h.at[ly, "bond_gross_bn"]) if ly in h.index else np.nan
            nb = gross - float(h.at[m, "bond_redemption_bn"])
            base = -(fc[m] + sv12) + nb + od12
            cum["gap0"] += base
            cum["gapseas"] += base + float(gseas.get(m.month, 0.0))
            if i in C.HORIZONS:
                act = float(h.at[m, "buffer_level_bn"])
                for k, v in cum.items():
                    rows.append({"origin": t, "h": i, "model": k, "fc": v, "actual": act})
                rows.append({"origin": t, "h": i, "model": "flat", "fc": s0, "actual": act})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Forward


def _settle(s: pd.Series) -> pd.Series:
    return (s + pd.offsets.BDay(2)).dt.to_period("M").dt.to_timestamp()


def _market_ratio(ops: pd.DataFrame, gross_fa: pd.Series, red_fa: pd.Series, asof: pd.Timestamp,
                  classes: list[str], months_g: int = 12, months_r: int = 24) -> tuple[float, float]:
    """Finanzagentur market gross / redemptions ÷ the tape's volume / ladder over trailing windows."""
    sel = ops[ops["sec_type"].astype(str).isin(classes)]
    vol = sel[sel["op_type"].astype(str) != "Own Holdings"]
    m0 = asof - pd.DateOffset(months=months_g - 1)
    st = _settle(vol["op_dt"])
    tape = vol[(st >= m0) & (st <= asof)]["amount_issued"].sum() / 1000
    fa = gross_fa.loc[m0:asof].sum()
    g_ratio = float(fa / tape) if tape > 0 else 1.0
    r0 = asof - pd.DateOffset(months=months_r - 1)
    lad = C.ladder_monthly(sel, asof + pd.offsets.MonthEnd(0), pd.date_range(r0, asof, freq="MS"), classes).sum()
    r_ratio = float(red_fa.loc[r0:asof].sum() / lad) if lad > 0 else 1.0
    return g_ratio, r_ratio


def coupon_forward(ops: pd.DataFrame, months: pd.DatetimeIndex, asof_fin: pd.Timestamp,
                   h: pd.DataFrame) -> pd.DataFrame:
    classes = ["Nominal", "Linker"]
    coup = ops[ops["sec_type"].astype(str).isin(classes) & (ops["op_type"].astype(str) != "Own Holdings")]
    tape = coup.groupby(_settle(coup["op_dt"]))["amount_issued"].sum() / 1000.0
    last_op = ops["op_dt"].max()
    fwd = pd.read_parquet(FWD)
    fwd = fwd[(fwd["country"] == "DE") & (fwd["op_dt"] > last_op) & (fwd["sec_type"].astype(str) != "Bill")]
    cal = fwd.groupby(_settle(fwd["op_dt"]))["amount_mn"].sum() / 1000.0
    src = fwd.assign(m=_settle(fwd["op_dt"])).groupby("m")["source"].agg(
        lambda s: "issuance calendar" if (s.astype(str) == "calendar").any() else "pattern sized on plan")
    g_ratio, r_ratio = _market_ratio(ops, h["bond_gross_bn"], h["bond_redemption_bn"], asof_fin, classes)
    ladder = C.ladder_monthly(ops[ops["sec_type"].astype(str).isin(classes)], last_op, months, classes)
    pre = pd.Series(0.0, index=months)
    for isin, (mat, amt) in PRE_TAPE.items():
        m = pd.Timestamp(mat).to_period("M").to_timestamp()
        if m in pre.index:
            pre[m] += amt
    out = pd.DataFrame(index=months)
    out["tape_bn"] = tape.reindex(months).fillna(0.0)
    out["calendar_bn"] = cal.reindex(months).fillna(0.0)
    out["bond_gross_bn"] = (out["tape_bn"] + out["calendar_bn"]) * g_ratio
    out["bond_redemption_bn"] = (ladder + pre) * r_ratio
    out["pre_tape_bn"] = pre
    out["coupon_source"] = ["tape" if out.at[m, "tape_bn"] > 0 else src.get(m, "none") for m in months]
    cal_rows = fwd[fwd["source"].astype(str) == "calendar"]
    out.attrs.update({"calendar_published_end": str(cal_rows["op_dt"].max().date()) if len(cal_rows) else None,"gross_market_ratio": g_ratio, "redemption_market_ratio": r_ratio,
                      "tape_last_op": str(last_op.date()),
                      "calendar_end": str(fwd["op_dt"].max().date()) if not fwd.empty else None,
                      "pre_tape_tranches_bn": {k: v[1] for k, v in PRE_TAPE.items()}})
    return out


def load_bubill_calendar() -> tuple[pd.DataFrame, dict]:
    js = json.loads(BUBILL_CAL.read_text())
    cal = pd.DataFrame(js["auctions"], columns=["op_dt", "isin", "maturity_dt", "amount_issued"])
    cal["op_dt"] = pd.to_datetime(cal["op_dt"])
    cal["maturity_dt"] = pd.to_datetime(cal["maturity_dt"])
    cal["sec_type"] = "Bill"
    cal["op_type"] = "Calendar"
    return cal, js


def bubill_beta_fit(hh: pd.DataFrame) -> tuple[np.ndarray, float]:
    """OLS ΔBubills = α_month + β·(cash − coupons − other-debt fin) on pre-origin rows."""
    need = hh["cash_requirement_bn"] - hh["net_bonds_bn"] - hh["other_fin_debt_bn"]
    dummies = np.eye(12)[hh.index.month - 1]
    X = np.column_stack([dummies, need.to_numpy()])
    y = hh["net_bubills_bn"].to_numpy()
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return coef[:12], float(coef[12])


def bubill_forward(ops: pd.DataFrame, months: pd.DatetimeIndex, asof_fin: pd.Timestamp,
                   h: pd.DataFrame, need: pd.Series | None = None) -> pd.DataFrame:
    """Tape, published quarterly calendar, then hist α+β·need demeaned to hold the calendar-end stock."""
    bills = ops[ops["sec_type"].astype(str) == "Bill"]
    cal, js = load_bubill_calendar()
    last_bill = bills["op_dt"].max()
    cal = cal[cal["op_dt"] > last_bill]
    g_ratio, r_ratio = _market_ratio(ops, h["bubill_gross_bn"], h["bubill_redemption_bn"], asof_fin,
                                     ["Bill"], months_g=12, months_r=12)
    tape = bills[bills["op_type"].astype(str) != "Own Holdings"]
    tape_m = tape.groupby(_settle(tape["op_dt"]))["amount_issued"].sum() / 1000.0
    cal_m = cal.groupby(_settle(cal["op_dt"]))["amount_issued"].sum() / 1000.0
    tape_end = _settle(pd.Series([last_bill])).iloc[0]
    cal_end = pd.Timestamp(js["quarter_end"]).to_period("M").to_timestamp()
    known = (tape_m.reindex(months).fillna(0.0) + cal_m.reindex(months).fillna(0.0)) * g_ratio
    fixed = known.where(months <= max(tape_end, cal_end))
    both = pd.concat([bills[["op_dt", "isin", "maturity_dt", "amount_issued", "sec_type"]],
                      cal[["op_dt", "isin", "maturity_dt", "amount_issued", "sec_type"]]], ignore_index=True)
    ladder = C.ladder_monthly(both, pd.Timestamp("2100-01-01"), months, ["Bill"]) * r_ratio
    shares = C.bill_tenor_shares(ops, last_bill, ["Bill"])
    target = pd.Series(0.0, index=months)
    alpha, beta = bubill_beta_fit(h.loc[:asof_fin].iloc[-96:])
    post = months > max(tape_end, cal_end)
    if post.any():
        nd = (need if need is not None else pd.Series(0.0, index=months)).reindex(months).fillna(0.0)
        raw = pd.Series([float(alpha[m.month - 1]) + beta * float(nd[m]) for m in months], index=months)
        raw = raw.where(post, 0.0)
        # Hold the stock at the calendar-end level: free-window nets sum to 0.
        raw.loc[post] = raw.loc[post] - float(raw.loc[post].mean())
        target.loc[post] = raw.loc[post]
    gross, red, net = C.roll_bills(target, ladder, shares, on_tape=fixed.fillna(0.0), fixed=fixed)
    src = np.where(months <= tape_end, "tape",
                   np.where(months <= cal_end, "calendar", "hist need (flat end stock)"))
    out = pd.DataFrame({"bubill_gross_bn": gross, "bubill_redemption_bn": red, "net_bubills_bn": net,
                        "bubill_source": src}, index=months)
    out.attrs.update({"bubill_gross_market_ratio": g_ratio, "bubill_redemption_market_ratio": r_ratio,
                      "bubill_tape_last_op": str(last_bill.date()), "bubill_calendar_end": str(cal_end.date()),
                      "bubill_calendar_source": js["source"], "bubill_calendar_url": js["url"],
                      "bubill_beta": round(beta, 3),
                      "bill_tenor_shares": {int(k): round(v, 3) for k, v in shares.items()}})
    return out


# Plan / max(hist complete-year total) above this → scale channel: do not freeze
# hist month shares onto a new annual level (defence / Sondervermögen ramps).
SV_SCALE_CHANNEL = 1.20


def _sv_complete_years(hist: pd.Series, origin: pd.Timestamp, k: int = 5
                       ) -> list[tuple[int, float, np.ndarray]]:
    """Complete calendar years usable for fund seasonality: (year, total, month shares)."""
    out: list[tuple[int, float, np.ndarray]] = []
    for yr in range(int(origin.year) - 1, int(origin.year) - 12, -1):
        b = hist.loc[f"{yr}-01-01":f"{yr}-12-01"]
        if len(b) != 12 or b.isna().any():
            continue
        tot = float(b.sum())
        if abs(tot) < 1.0:
            continue
        sh = b.to_numpy(dtype=float) / tot
        if not (0.15 <= float(sh[:8].sum()) <= 0.85):
            continue
        out.append((yr, tot, sh))
        if len(out) == k:
            break
    return out


def _sv_month_shares(hist: pd.Series, origin: pd.Timestamp, k: int = 5) -> pd.Series | None:
    """Mean calendar-month share of annual fund borrowing over complete years.

    Wirtschaftspläne (HG Anlage) publish annual totals only — no monthly
    Auszahlungsprofil. FA / BMF Monatsbericht borrowing is the observable
    seasonality. Skip ramp/wind-down years (Aug YTD share outside 15–85%).
    Needs ≥2 usable years; else None (caller keeps a flat pace).
    """
    years = _sv_complete_years(hist, origin, k=k)
    if len(years) < 2:
        return None
    return pd.Series(np.mean([sh for _, _, sh in years], axis=0), index=range(1, 13))


def _sv_scale_channel(plan_bn: float, hist: pd.Series, origin: pd.Timestamp) -> bool:
    """True when this year's Wirtschaftsplan is outside the scale of share-years.

    Hist Dec spikes (~30% of BW in 2023–25) are a within-regime shape. Applying
    them after a defence / fund ramp freezes last year's profile onto a new
    annual level — wrong. Drop to plan residual, flat.
    """
    years = _sv_complete_years(hist, origin)
    if len(years) < 2 or plan_bn <= 0:
        return False
    peak = max(tot for _, tot, _ in years)
    return float(plan_bn) >= SV_SCALE_CHANNEL * peak


def _sv_allocate(total: float, months: list[pd.Timestamp],
                 shares: pd.Series | None) -> list[float]:
    """Spread ``total`` over ``months`` by hist shares (renormalised) or evenly."""
    if not months:
        return []
    if shares is None or total == 0.0:
        return [total / len(months)] * len(months)
    w = np.asarray([float(shares.loc[m.month]) for m in months], dtype=float)
    s = float(w.sum())
    if s <= 1e-12:
        return [total / len(months)] * len(months)
    return list(total * w / s)


def sv_forward(h: pd.DataFrame, months: pd.DatetimeIndex, plan: dict[int, dict]) -> tuple[pd.DataFrame, dict]:
    """Special-fund net borrowing; other funds 0.

    Plan funds (Infrastruktur, Bundeswehr):
    - Scale channel (Wirtschaftsplan ≫ recent complete FA years): YE = plan,
      rest-of-year residual flat — do not freeze hist month shares onto the new
      level (defence / Sondervermögen ramps).
    - Stable scale + ≥2 share years: hist month profile (BW H2-heavy); implied YE
      = YTD / hist elapsed share; residual on that profile.
    - Else: flat YTD pace.
    Official scenario always closes to the Wirtschaftsplan annual (same shape
    rule as the model for the residual). Other funds (FMS/ITF/WSF/RSF) forward 0.
    """
    last = h.index.max()
    yr = last.year
    elapsed = last.month
    names = list(SV_PLAN.values())
    ytd = {n: float(h.loc[f"{yr}-01-01":last, f"cash_requirement_sv_{n}_bn"].sum()) for n in names}
    plan_yr = sum(plan[yr][n] for n in names)
    exe_f = {n: ytd[n] / (plan[yr][n] * elapsed / 12.0) if plan[yr][n] else 1.0 for n in names}
    exe = sum(ytd.values()) / (plan_yr * elapsed / 12.0) if plan_yr and elapsed else 1.0
    f = pd.DataFrame(index=months)
    profile_label: dict[str, str] = {}
    implied_ye = {}
    scale_ch: dict[str, bool] = {}
    for n in names:
        series = h[f"cash_requirement_sv_{n}_bn"].dropna()
        plan_n = float(plan[yr][n])
        channel = _sv_scale_channel(plan_n, series, last)
        scale_ch[n] = channel
        sh = None if channel else _sv_month_shares(series, last)
        rest_m = [m for m in months if m.year == yr]
        later_by_yr: dict[int, list[pd.Timestamp]] = {}
        for m in months:
            if m.year != yr:
                later_by_yr.setdefault(m.year, []).append(m)
        if channel:
            # Wirtschaftsplan is the annual channel — put it in; shape not last year's.
            implied = plan_n
            sh = None
            profile_label[n] = "plan_residual_flat"
        elif sh is not None and elapsed >= 1:
            elapsed_share = float(sh.loc[1:elapsed].sum())
            implied = (ytd[n] / elapsed_share) if elapsed_share > 0.05 else ytd[n] * 12.0 / elapsed
            profile_label[n] = "hist_month_share"
        else:
            implied = ytd[n] * 12.0 / elapsed if elapsed else 0.0
            sh = None
            profile_label[n] = "flat_ytd_pace"
        implied_ye[n] = implied
        rem_total = implied - ytd[n]
        mod_rest = _sv_allocate(rem_total, rest_m, sh)
        off_rest = _sv_allocate(plan_n - ytd[n], rest_m, sh)
        mod_map = {m: v for m, v in zip(rest_m, mod_rest)}
        off_map = {m: v for m, v in zip(rest_m, off_rest)}
        for y2, ms in later_by_yr.items():
            p_ann = plan[y2][n] if y2 in plan else plan_n
            # Later year: scale channel again vs that year's plan; else keep shares.
            sh2 = None if _sv_scale_channel(float(p_ann), series, last) else sh
            for m, v in zip(ms, _sv_allocate(p_ann * exe_f[n], ms, sh2)):
                mod_map[m] = v
            for m, v in zip(ms, _sv_allocate(p_ann, ms, sh2)):
                off_map[m] = v
        f[f"cash_requirement_sv_{n}_bn"] = [mod_map[m] for m in months]
        f[f"sv_official_{n}_bn"] = [off_map[m] for m in months]
    f["sv_plan_funds_bn"] = sum(f[f"cash_requirement_sv_{n}_bn"] for n in names)
    f["sv_other_funds_bn"] = 0.0
    f["cash_requirement_sv_bn"] = f["sv_plan_funds_bn"] + f["sv_other_funds_bn"]
    f["sv_official_bn"] = sum(f[f"sv_official_{n}_bn"] for n in names)
    info = {"sv_ytd_bn": {k: round(v, 1) for k, v in ytd.items()}, "sv_ytd_through": f"{last:%Y-%m}",
            "sv_plan_year_bn": round(plan_yr, 1), "sv_execution_ytd": round(exe, 3),
            "sv_execution_ytd_by_fund": {k: round(v, 3) for k, v in exe_f.items()},
            "sv_other_funds_bn_per_month": 0.0,
            "sv_profile": profile_label,
            "sv_scale_channel": scale_ch,
            "sv_implied_ye_bn": {k: round(v, 1) for k, v in implied_ye.items()},
            "sv_plan_pace_rest_of_year_bn_per_month": round((plan_yr - sum(ytd.values())) / max(12 - elapsed, 1), 2),
            "sv_model_pace_rest_of_year_bn_per_month": round(sum(ytd.values()) / elapsed, 2) if elapsed else 0.0}
    return f, info


def forward(h: pd.DataFrame, cash: pd.Series, model: str, gap_mode: str, core: pd.DataFrame,
            hc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.cash_season import HORIZON_END, forward_months as _fwd_m

    plan = load_plan()
    last_cash = cash.dropna().index.max()
    asof_fin = h.index.max()
    h_live = _fwd_m(last_cash)
    end = max(last_cash + pd.DateOffset(months=h_live), HORIZON_END)
    months = pd.date_range(asof_fin + pd.offsets.MonthBegin(1), end, freq="MS")
    yr = last_cash.year
    soll_now = -float(core.loc[:last_cash, "soll_fin_balance"].dropna().iloc[-1])
    hc_now, hc_years = haircut_at(hc, yr)
    hc_lo, hc_hi = haircut_band(hc, yr)
    tg_plan = {yr: soll_now, **{y: plan[y]["core"] for y in plan if y > yr}}
    k = model.rsplit("_", 1)[1] if model[-1].isdigit() else "3"
    planish = model.startswith(("planhc", "target"))
    fc = C.forecast(cash.dropna(), last_cash, h_live, model,
                    targets={**tg_plan, "hc": hc_now} if planish else None)
    off = C.forecast(cash.dropna(), last_cash, h_live, f"target_{k}", targets=tg_plan)
    band = {b: C.forecast(cash.dropna(), last_cash, h_live, f"planhc_{k}",
                          targets={**tg_plan, "hc": v})
            for b, v in (("lo", hc_lo), ("hi", hc_hi))}
    exp = C.monthly_from_ytd(core["ytd_expenditure"].dropna())
    rev = C.monthly_from_ytd(core["ytd_revenue"].dropna())
    mb = F.load_bmf_mb()
    known = mb.loc[:last_cash].dropna(subset=["soll_expenditure", "soll_revenue", "soll_year"])
    known = known[known["soll_year"] == yr]
    line_info: dict = {}
    if not known.empty and fc is not None:
        plan_e = float(known.iloc[-1]["soll_expenditure"])
        plan_r = float(known.iloc[-1]["soll_revenue"])
        maps = soll_line_maps(mb)
        re = line_ratio(exp, maps["soll_expenditure"], yr)
        rr = line_ratio(rev, maps["soll_revenue"], yr)
        paths = open_year_core(exp, rev, last_cash, h_live, plan_e, plan_r, re, rr)
        if paths is not None:
            for m in list(fc.index):
                if m.year != yr:
                    continue
                fc.loc[m] = float(paths["central"].loc[m])
                for b in ("lo", "hi"):
                    if band.get(b) is not None and m in band[b].index:
                        band[b].loc[m] = float(paths[b].loc[m])
            line_info = {
                "core_spend_soll_bn": round(plan_e, 1),
                "core_rev_soll_bn": round(plan_r, 1),
                "core_spend_ratio": round(re[0], 3),
                "core_rev_ratio": round(rr[0], 3),
                "core_line_years": re[3],
            }
    # Beyond the year in progress the core budget runs on our component model (the Shock scenarios page's lines:
    # receipts on nominal GDP, spending on prices with no debt-brake reversion, interest from the debt book). The
    # band keeps its width around it. The special funds stay on their run-rate model below.
    from sovereign_cash.scenarios.channels.sarimax import later_years

    ours = later_years("de", months, yr, column="cash_requirement")
    if fc is not None and len(ours):
        fc = fc.copy()
        for m, v in ours.items():
            shift = float(v) - float(fc.get(m, v))
            fc.loc[m] = float(v)
            for b in ("lo", "hi"):
                if band.get(b) is not None and m in band[b].index:
                    band[b].loc[m] = float(band[b].loc[m]) + shift
        fc = fc.sort_index()
    ops = pd.read_parquet(OPS)
    cf = coupon_forward(ops, months, asof_fin, h)
    svf, sv_info = sv_forward(h, months, plan)
    f = pd.DataFrame(index=months)
    f["cash_requirement_core_bn"] = [float(cash[m]) if m <= last_cash else float(fc[m]) for m in months]
    for b, s in band.items():
        f[f"cash_requirement_core_{b}_bn"] = [float(cash[m]) if m <= last_cash else float(s[m]) for m in months]
    f["cash_observed"] = [m <= last_cash for m in months]
    f["cash_requirement_sv_bn"] = svf["cash_requirement_sv_bn"]
    for n in SV_PLAN.values():
        f[f"cash_requirement_sv_{n}_bn"] = svf[f"cash_requirement_sv_{n}_bn"]
    f["cash_requirement_sv_other_bn"] = svf["sv_other_funds_bn"]
    f["cash_requirement_bn"] = f["cash_requirement_core_bn"] + f["cash_requirement_sv_bn"]
    # later_years already folded legislated policy into the core path for outer years.
    f["policy_bn"] = 0.0
    if len(ours):
        f.loc[ours.index, "policy_bn"] = 1.0
    f["cash_requirement_fc_official_bn"] = [
        (float(cash[m]) if m <= last_cash else float(off[m])) + float(svf.at[m, "sv_official_bn"]) for m in months]
    f["bond_gross_bn"] = cf["bond_gross_bn"]
    f["bond_redemption_bn"] = cf["bond_redemption_bn"]
    f["net_bonds_bn"] = f["bond_gross_bn"] - f["bond_redemption_bn"]
    f["bond_redemption_cb_bn"] = C.cb_redemptions("DE", months)
    f["private_net_bonds_bn"] = f["net_bonds_bn"] + f["bond_redemption_cb_bn"]
    # Post-calendar Bubills see cash − coupons − trailing other-debt fin (gap stays on placements).
    od0 = float(h["other_fin_debt_bn"].iloc[-12:].mean())
    bill_need = f["cash_requirement_bn"] - f["net_bonds_bn"] - od0
    bf = bubill_forward(ops, months, asof_fin, h, need=bill_need)
    for c in ("bubill_gross_bn", "bubill_redemption_bn", "net_bubills_bn"):
        f[c] = bf[c]
    f["mm_budget_gross_bn"] = 0.0
    f["mm_budget_redemption_bn"] = 0.0
    f["net_mm_budget_bn"] = 0.0
    f["bill_gross_bn"] = f["bubill_gross_bn"] + f["mm_budget_gross_bn"]
    f["bill_redemption_bn"] = f["bubill_redemption_bn"] + f["mm_budget_redemption_bn"]
    f["net_bills_bn"] = f["net_bubills_bn"] + f["net_mm_budget_bn"]
    f["other_fin_debt_bn"] = float(h["other_fin_debt_bn"].iloc[-12:].mean())
    gseas = h["other_fin_gap_bn"].iloc[-60:].groupby(h.index[-60:].month).mean()
    f["other_fin_gap_bn"] = [float(gseas.get(m.month, 0.0)) if gap_mode == "gapseas" else 0.0 for m in months]
    f["other_fin_bn"] = f["other_fin_debt_bn"] + f["other_fin_gap_bn"]
    f["buffer_change_bn"] = -f["cash_requirement_bn"] + f["net_bonds_bn"] + f["net_bills_bn"] + f["other_fin_bn"]
    f["bill_stock_implied_bn"] = float(h["bill_stock_bn"].iloc[-1]) + f["net_bubills_bn"].cumsum()
    f["buffer_level_implied_bn"] = float(h["buffer_level_bn"].iloc[-1]) + f["buffer_change_bn"].cumsum()
    # plan scenario: same issuance, cash requirement at the official profile
    plan_change = f["buffer_change_bn"] + f["cash_requirement_bn"] - f["cash_requirement_fc_official_bn"]
    f["buffer_level_plan_bn"] = float(h["buffer_level_bn"].iloc[-1]) + plan_change.cumsum()
    f["coupon_source"] = cf["coupon_source"]
    f["fc_source"] = [
        ("core observed (BMF); " if m <= last_cash else
         ("core spending plan and revenue plan; " if line_info and m.year == yr else
          "core component model (as the Shock scenarios page); " if m.year > yr else f"core {model}; "))
        + f"special funds YTD run rate; coupons {cf.at[m, 'coupon_source']}; "
        + f"Bubills {bf.at[m, 'bubill_source']}" for m in months]
    f["is_forecast"] = True
    # Bottom-up: v1 (Soll×hist then smly) first; macro replaces live only if its PRE ship clears.
    from sovereign_cash.de.bottom_up import apply_to_forward as apply_v1
    from sovereign_cash.de.bottom_up_macro import apply_to_forward as apply_macro
    f, bu_info = apply_v1(f, cash, core, last_cash)
    f, mac_info = apply_macro(f, cash, core, last_cash)
    live_src = None
    if mac_info.get("bottomup_macro_ship"):
        live_src = "core bottom-up macro Soll×blend then NGDP/CPI priors; "
    elif bu_info.get("bottomup_ship"):
        live_src = "core bottom-up spend/rev Soll×ratio then smly; "
    if live_src is not None:
        f["buffer_level_implied_bn"] = float(h["buffer_level_bn"].iloc[-1]) + f["buffer_change_bn"].cumsum()
        plan_change = f["buffer_change_bn"] + f["cash_requirement_bn"] - f["cash_requirement_fc_official_bn"]
        f["buffer_level_plan_bn"] = float(h["buffer_level_bn"].iloc[-1]) + plan_change.cumsum()
        f["fc_source"] = [
            ("core observed (BMF); " if m <= last_cash else
             "core component model (as the Shock scenarios page); " if m.year > yr else live_src)
            + f"special funds YTD run rate; coupons {cf.at[m, 'coupon_source']}; "
            + f"Bubills {bf.at[m, 'bubill_source']}" for m in months]
    info = {"origin_cash": str(last_cash.date()), "origin_financing": str(asof_fin.date()),
            "gap_mode": gap_mode, "core_target_year_bn": soll_now, "core_haircut": round(hc_now, 3),
            "core_haircut_years": hc_years, "core_haircut_band": [round(hc_lo, 3), round(hc_hi, 3)],
            "plan": plan, **line_info, **sv_info, **cf.attrs, **bf.attrs, **bu_info, **mac_info}
    return f, info
