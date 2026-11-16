"""France: State cash framework (history identity, deficit model, BTF path, scores).

The buffer
    The Treasury account at the Banque de France has been held at EUR 120-160mn since May 2023,
    with any surplus placed in the market (repo / deposits): the account is liquidity-neutral and
    its Δ is 0. Before May 2023 the account was not pinned, and neither its balance nor Treasury
    placements, correspondents' deposits (local authorities, public bodies) or other treasury
    resources are published monthly in machine-readable form, so in history they sit together
    in ``other_fin_bn`` and buffer_change_bn = 0 by construction, flagged ``account_pinned``.

History identity (EUR bn, inflows +), Jan-2009 → last month with both SMB and AFT data:

    0 = −cash requirement + net OAT + net BTF + other_fin

cash requirement   −monthly change of the SMB YTD "solde général d'exécution" (State deficit +; not
                   the besoin de financement, which adds amortisation and other treasury needs).
net OAT            Δ outstanding negotiable debt with original maturity > 1y (AFT/INSEE, face;
                   OATi/OAT€i at indexed value), as published. Months whose net exceeds gross by
                   more than ``JUMP_BN`` are flagged, not patched.
net BTF            Δ outstanding ≤1y (BTF), face. Gross = AFT BTF auctions; redemptions = gross − net.
other_fin          everything else.

OAT gross is the tape with each auction's amount replaced by the AFT auction-history total
(competitive + non-competitive) where the history file has it.

Forward (12 months after the last SMB month):
cash requirement   PLF tax/spend categories × RESF macro when ``fr.bottom_up_macro.SHIP``;
                   else v1 SMB-component seas when ``fr.bottom_up.SHIP``; else PRE-selected
                   seasonal. Any calendar year with a published State deficit (PLF/LFI solde
                   budgétaire hors FMI = déficit à financer in plan.json) is closed to that
                   total (plan level × month shape). ``cash_requirement_bottomup_bn`` holds
                   the unanchored bottom-up path; ``cash_requirement_seas_bn`` the seasonal.
OAT gross          tape months; the month of the last tape auction = max(tape, programme target);
                   then ``issuance.forward.fr_month_targets``.
OAT redemptions    AFT outstanding by line (nets buybacks done; linkers × index ratio at maturity)
                   + forward buybacks = gross × (1 − net/gross ratio run this year), each year's
                   buybacks taken pro rata out of the next year's maturities.
other_fin          plan year: (plan-implied full year − realised YTD) / remaining months, plan-implied
                   = deficit − (net programme − amortisation) − BTF change; next year: AFT 'autres
                   ressources' / 12. Plus the mean-zero month-of-year profile if the PRE gap rule picks it.
BTF                observed months from AFT/INSEE; tape gross; otherwise the stock steps evenly to the
                   AFT year-end level (prior Dec stock + AFT BTF change; next year's change after that).
                   Treasury placements (buffer_change_bn) absorb the difference to the financing need,
                   so they show how far the model's financing need departs from the AFT path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import fetch as F

OPS = F.ROOT.parent / "issuance" / "cache" / "ops_fr.parquet"
OFFICIAL = F.ROOT.parent / "issuance" / "cache" / "official_isin_flow.parquet"
COUPON = ["Nominal", "Linker"]
START = pd.Timestamp("2009-01-01")
PIN_START = pd.Timestamp("2023-05-01")
JUMP_BN = 10.0
RECON_TOL_BN = 5.0


def load_ops() -> pd.DataFrame:
    """AFT tape with OAT auction amounts set to the auction-history total (incl. non-competitive)."""
    ops = pd.read_parquet(OPS)
    hist = F.load_mlt_auctions()
    ops.attrs["onc_added_bn"] = {}
    if hist.empty:
        return ops
    tot = hist.groupby([hist["isin"].astype(str), hist["op_dt"].dt.normalize()])["total_mn"].sum()
    m = ops["op_type"].astype(str).eq("Auction") & ~ops["sec_type"].astype(str).eq("Bill")
    keys = list(zip(ops.loc[m, "isin"].astype(str), ops.loc[m, "op_dt"].dt.normalize()))
    new = pd.Series([tot.get(k, np.nan) for k in keys], index=ops.index[m])
    upd = new.notna() & (new - ops.loc[m, "amount_issued"]).abs().gt(0.5)
    diff = (new - ops.loc[m, "amount_issued"])[upd]
    ops.loc[diff.index, "amount_issued"] = new[upd]
    yr = ops.loc[diff.index, "op_dt"].dt.year
    ops.attrs["onc_added_bn"] = {int(y): round(float(v) / 1000, 2) for y, v in diff.groupby(yr).sum().items()}
    ops.attrs["hist_file"] = hist.attrs.get("file")
    return ops


def _ops_monthly(ops: pd.DataFrame, classes: list[str]) -> pd.Series:
    o = ops[ops["sec_type"].astype(str).isin(classes) & (ops["amount_issued"] > 0)
            & (ops["op_type"].astype(str) != "Buyback")]
    when = o["issue_dt"].fillna(o["op_dt"] + pd.offsets.BDay(2))
    return o.groupby(when.dt.to_period("M").dt.to_timestamp())["amount_issued"].sum() / 1000.0


def stock_jumps(mlt: pd.Series, gross: pd.Series) -> list[dict]:
    """Months whose Δ OAT stock exceeds gross issuance by more than JUMP_BN (flag only)."""
    out = []
    d = mlt.diff()
    for t, v in d.dropna().items():
        g = float(gross.get(t, 0.0))
        if v > g + JUMP_BN:
            p = t - pd.offsets.MonthBegin(1)
            out.append({"month": f"{t:%Y-%m}", "net_bn": round(float(v), 1), "gross_bn": round(g, 1),
                        "prior_month": f"{p:%Y-%m}", "prior_net_bn": round(float(d.get(p, np.nan)), 1),
                        "prior_gross_bn": round(float(gross.get(p, 0.0)), 1)})
    return out


def build_history() -> tuple[pd.DataFrame, dict]:
    smb = F.load_smb()
    debt = F.load_debt()
    ops = load_ops()
    cash = -C.monthly_from_ytd(smb["solde_ytd"])
    end = min(cash.dropna().index.max(), debt.index.max())
    idx = pd.date_range(START, end, freq="MS")
    gross = _ops_monthly(ops, COUPON)
    mlt = debt["mlt"]
    h = pd.DataFrame(index=idx)
    h["cash_requirement_bn"] = cash.reindex(idx)
    h["net_bonds_bn"] = mlt.diff().reindex(idx)
    h["bond_gross_bn"] = gross.reindex(idx).fillna(0.0)
    h["bond_redemption_bn"] = h["bond_gross_bn"] - h["net_bonds_bn"]
    h["net_bills_bn"] = debt["btf"].diff().reindex(idx)
    h["bill_gross_bn"] = _ops_monthly(ops, ["Bill"]).reindex(idx).fillna(0.0)
    h["bill_redemption_bn"] = h["bill_gross_bn"] - h["net_bills_bn"]
    h["buffer_change_bn"] = 0.0
    h["other_fin_bn"] = h["buffer_change_bn"] + h["cash_requirement_bn"] - h["net_bonds_bn"] - h["net_bills_bn"]
    h["bill_stock_bn"] = debt["btf"].reindex(idx)
    h["oat_stock_bn"] = mlt.reindex(idx)
    h["indexed_stock_bn"] = debt["indexed"].reindex(idx)
    h["account_pinned"] = h.index >= PIN_START
    h = h.dropna(subset=["cash_requirement_bn", "net_bonds_bn", "net_bills_bn"])
    h["is_forecast"] = False
    h["cash_observed"] = True
    h["fc_source"] = "history"
    dec = debt["btf"].where(debt.index.month == 12).shift(1).ffill()
    chk = (debt["btf_ytd_change"] - (debt["btf"] - dec)).dropna()
    return h, {"smb": smb, "debt": debt, "ops": ops, "cash_full": cash, "mlt": mlt,
               "oat_jumps": stock_jumps(mlt, gross),
               "btf_ytd_check_max_bn": float(chk.abs().max()) if len(chk) else None}


def deficit_scores(cash: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Seasonal models plus component-annual level × seasonal shape (US-style split).

    Receipts on lag-1 nominal GDP, spending on lag-1 CPI; deficit = identity. Slope rules
    ``unit`` (pass-through) and ``floor`` (b≥0) compete; neither has beaten seasonals on PRE.
    """
    from sovereign_cash.annual_macro import component_book

    cash = cash.dropna()
    bt = C.backtest(cash, C.model_list(has_target=False))
    for rule in ("unit", "floor"):
        book = component_book("fr", rule=rule)
        bt = pd.concat([bt, C.backtest(cash, [f"annual_comp_{rule}_3"], targets_asof=book)],
                       ignore_index=True)
    sc = C.score_table(bt)
    model, why = C.choose(sc)
    if model.startswith("annual_comp_"):
        rule = "unit" if "unit" in model else "floor"
        why = (f"component annual level ({rule}) × seasonal month shape, {why}; "
               "receipts on GDP, spending on CPI, deficit = identity")
    pre = sc[(sc["window"] == "PRE") & sc["common"]]
    full = pre.groupby("model")["h"].nunique()
    means = pre[pre["model"].isin(full[full == len(C.HORIZONS)].index)].groupby("model")["mae"].mean().sort_values()
    if len(means) > 1 and means.iloc[1] - means.iloc[0] < 0.25:
        why += f"; runner-up '{means.index[1]}' is {means.iloc[1] - means.iloc[0]:.2f}bn behind on PRE"
    return bt, sc, model, why


def bill_backtest(h: pd.DataFrame, cash: pd.Series, model: str) -> pd.DataFrame:
    """BTF stock at t+h implied at origin t vs the AFT stock.

    Known at t: deficit model path; OAT redemptions treated as known (realised gross − net, so
    buybacks and indexation leak in); OAT gross = same month last year; other_fin = 0
    ('gap0') or its last-5-year month-of-year mean ('gapseas'). Benchmark: flat stock.
    """
    rows = []
    for t in h.index:
        if t < START + pd.DateOffset(years=5):
            continue
        fc = C.forecast(cash.dropna(), t, 12, model)
        if fc is None:
            continue
        past = h.loc[:t]
        gseas = past["other_fin_bn"].iloc[-60:].groupby(past.index[-60:].month).mean()
        s0 = float(h.at[t, "bill_stock_bn"])
        cum = {"gap0": s0, "gapseas": s0}
        for i, m in enumerate(fc.index, start=1):
            if m not in h.index:
                break
            ly = m - pd.DateOffset(years=1)
            nb = float(h.at[ly, "bond_gross_bn"]) - float(h.at[m, "bond_redemption_bn"])
            base = fc[m] - nb
            cum["gap0"] += base
            cum["gapseas"] += base - float(gseas.get(m.month, 0.0))
            if i in C.HORIZONS:
                act = float(h.at[m, "bill_stock_bn"])
                for k, v in cum.items():
                    rows.append({"origin": t, "h": i, "model": k, "fc": v, "actual": act})
                rows.append({"origin": t, "h": i, "model": "flat", "fc": s0, "actual": act})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Forward pieces


def _oat_kind(desc: str) -> str:
    d = str(desc).upper().replace("€", "E")
    if "OATEI" in d or "OATEUROI" in d:
        return "oatei"
    if "OATI" in d:
        return "oati"
    return "oat"


def oat_ladder(months: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Redemptions per line from AFT outstanding (nets buybacks done), linkers at indexed value.

    Encours drops a line the day it matures, so a cash-history cutoff before the page as-of
    misses mid-month maturities. Those are spliced from ``ops_fr`` (net issued face).
    """
    enc = F.load_aft_encours()
    enc = enc[(enc["maturity"] >= months.min()) & (enc["kind"].isin(["oat", "oatei", "oati"]))].copy()
    if OPS.exists():
        ops = load_ops()
        coup = ops[ops["sec_type"].astype(str).isin(COUPON) & ops["maturity_dt"].notna()]
        g = coup.groupby(coup["isin"].astype(str)).agg(
            amt=("amount_issued", "sum"), maturity=("maturity_dt", "first"),
            desc=("sec_desc_long", "first"))
        g["maturity"] = pd.to_datetime(g["maturity"])
        today = pd.Timestamp.today().normalize()
        alive = set(enc["isin"].astype(str))
        miss = g[(g["maturity"] >= months.min()) & (g["maturity"] <= today)
                 & (g["amt"] > 1) & ~g.index.isin(alive)]
        if len(miss):
            past = pd.DataFrame({
                "isin": miss.index.astype(str), "label": miss["desc"].astype(str).to_numpy(),
                "kind": [_oat_kind(x) for x in miss["desc"]],
                "maturity": miss["maturity"].to_numpy(),
                "nominal_bn": (miss["amt"] / 1000.0).to_numpy(),
            })
            enc = pd.concat([enc, past], ignore_index=True)
    enc["coef"], notes = 1.0, {}
    for i, r in enc[enc["kind"].isin(["oatei", "oati"])].iterrows():
        c, how = F.oatei_coef(r["maturity"], r["kind"])
        enc.at[i, "coef"] = c
        notes[r["isin"]] = how
    enc["redemption_bn"] = enc["nominal_bn"] * enc["coef"]
    enc["month"] = enc["maturity"].dt.to_period("M").dt.to_timestamp()
    return enc, notes


def cb_holdings(enc: pd.DataFrame) -> pd.DataFrame:
    """Estimated Eurosystem (PSPP+PEPP) holdings per line at the last snapshot, scaled to AFT outstanding.

    The ISIN allocation is sized on gross issued; it is scaled pro rata to outstanding net of
    buybacks, capped at that outstanding, and converted to cash at the line's index ratio."""
    out = enc[["isin", "month", "nominal_bn", "coef"]].copy()
    out["cb_face_bn"] = 0.0
    if not OFFICIAL.exists():
        return out
    x = pd.read_parquet(OFFICIAL, columns=["month", "isin", "country", "program", "official_stock_mn", "outstanding_mn"])
    x = x[x["country"] == "FR"].copy()
    x["month"] = pd.to_datetime(x["month"])
    snap = x["month"].max()
    x = x[x["month"] == snap]
    g = x.groupby(x["isin"].astype(str)).agg(face=("official_stock_mn", "sum"), out=("outstanding_mn", "max")) / 1000.0
    for i, r in out.iterrows():
        k = str(r["isin"])
        if k not in g.index:
            continue
        face, model_out = float(g.at[k, "face"]), float(g.at[k, "out"])
        scale = min(1.0, r["nominal_bn"] / model_out) if model_out > 0 else 1.0
        out.at[i, "cb_face_bn"] = min(face * scale, r["nominal_bn"])
    out["cb_cash_bn"] = out["cb_face_bn"] * out["coef"]
    out.attrs["snapshot"] = f"{snap:%Y-%m}"
    return out


def buyback_forward(months: pd.DatetimeIndex, targets: dict[pd.Timestamp, float], ratio: float,
                    enc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Monthly buyback cash and the reduction of maturities it implies (next-year lines, pro rata)."""
    bb = pd.Series({m: targets.get(m, 0.0) * (1.0 - 1.0 / ratio) for m in months}).reindex(months).fillna(0.0)
    cut = pd.Series(0.0, index=months)
    for yr, amt in bb.groupby(bb.index.year).sum().items():
        nxt = enc[enc["maturity"].dt.year == yr + 1]
        if nxt.empty or amt <= 0:
            continue
        w = nxt["redemption_bn"] / nxt["redemption_bn"].sum()
        for m, x in (w * amt).groupby(nxt["month"]).sum().items():
            if m in cut.index:
                cut[m] += x
    return bb, cut


def programme(ops: pd.DataFrame, plan: dict) -> dict:
    """AFT net programme progress for the plan year from plan.json and the (ONC-corrected) tape."""
    yr = max(int(y) for y, v in plan.items() if v.get("net_mlt_done_share") is not None)
    p = plan[str(yr)]
    asof = pd.Timestamp(p["net_mlt_done_asof"])
    c = ops[(ops["sec_type"].astype(str) != "Bill") & (ops["amount_issued"] > 0)]
    gross = float(c[(c["op_dt"].dt.year == yr) & (c["op_dt"] <= asof)]["amount_issued"].sum()) / 1000
    done = p["net_mlt_issuance"] * p["net_mlt_done_share"]
    return {"year": yr, "asof": asof, "programme_bn": p["net_mlt_issuance"], "done_share": p["net_mlt_done_share"],
            "net_done_bn": done, "gross_bn": gross, "buybacks_bn": gross - done, "ratio": gross / done,
            "net_left_bn": p["net_mlt_issuance"] - done}


def coupon_forward(ops: pd.DataFrame, months: pd.DatetimeIndex, plan: dict, enc: pd.DataFrame) -> pd.DataFrame:
    from issuance.forward import fr_month_targets

    prog = programme(ops, plan)
    tape = _ops_monthly(ops, COUPON)
    last_op = ops[ops["sec_type"].astype(str) != "Bill"]["op_dt"].max()
    targets = {p.to_timestamp(): v / 1000.0 for p, v in fr_month_targets(ops).items()}
    last_tape_month = (last_op + pd.offsets.BDay(2)).to_period("M").to_timestamp()
    out = pd.DataFrame(index=months)
    out["tape_bn"] = tape.reindex(months).fillna(0.0)
    out["programme_bn"] = [targets.get(m, np.nan) if m >= last_tape_month else np.nan for m in months]
    gross = []
    for m in months:
        if m < last_tape_month:
            gross.append(out.at[m, "tape_bn"])
        elif m == last_tape_month:
            gross.append(max(out.at[m, "tape_bn"], np.nan_to_num(out.at[m, "programme_bn"])))
        else:
            gross.append(np.nan_to_num(out.at[m, "programme_bn"]))
    out["bond_gross_bn"] = gross
    lad = enc.groupby("month")["redemption_bn"].sum().reindex(months).fillna(0.0)
    fwd_targets = {m: out.at[m, "bond_gross_bn"] for m in months if m >= last_tape_month}
    bb, cut = buyback_forward(months, fwd_targets, prog["ratio"], enc)
    out["maturities_bn"] = lad - cut
    out["buyback_bn"] = bb
    out["bond_redemption_bn"] = out["maturities_bn"] + out["buyback_bn"]
    out.attrs.update({"tape_last_op": last_op, "last_tape_month": last_tape_month, "programme": prog,
                      "buybacks_forward_by_year_bn": {int(y): float(v) for y, v in bb.groupby(bb.index.year).sum().items()}})
    return out


def btf_supplement(months: pd.DatetimeIndex, ops: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Hand-entered BTF auctions not yet on the tape; any auction date already on the tape is dropped."""
    js = F.load_json("btf_supplement.json") or {"auctions": []}
    on_tape = set(ops[ops["sec_type"].astype(str) == "Bill"]["op_dt"].dt.normalize())
    s = pd.Series(0.0, index=months)
    used = []
    for a in js["auctions"]:
        if pd.Timestamp(a["op_dt"]).normalize() in on_tape:
            continue
        m = pd.Timestamp(a["settle_dt"]).to_period("M").to_timestamp()
        if m in s.index:
            s[m] += a["total_mn"] / 1000.0
            used.append(a)
    return s, used


def btf_path(need: pd.Series, ladder: pd.Series, shares: dict[int, float], on_tape: pd.Series,
             fixed: pd.Series, observed: dict[pd.Timestamp, float], s0: float,
             anchors: dict[pd.Timestamp, float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Gross, redemptions, net BTF month by month.

    Observed months take the AFT stock change (gross = tape); tape / supplement months take the
    tape gross. Free months provisionally absorb the financing need (cash − OAT − other_fin) so
    cash seasonals flow into BTF; a uniform shift inside each stretch to the next AFT year-end
    anchor then lands the stock on that level (UK ``solve_bills`` pattern). Past the last anchor
    free months keep the demeaned need (cash seasonal, YE stock held). Gross ≥ 0; untaped gross
    rolls into m+k at the tenor shares.
    """
    net = pd.Series(0.0, index=need.index)
    free = pd.Series(True, index=need.index)
    red0 = ladder.astype(float).copy()
    for m in need.index:
        if m in observed:
            net[m] = float(observed[m])
            free[m] = False
        elif pd.notna(fixed.get(m, np.nan)):
            net[m] = float(fixed[m]) - float(red0[m])
            free[m] = False
        else:
            net[m] = float(need[m])

    # Uniform shift on free months inside each window so stock hits the next YE anchor.
    # Past the last anchor: demean the absorb path by calendar year (hold YE stock; keep seasonals).
    months = list(need.index)
    stock = float(s0)
    i = 0
    while i < len(months):
        m = months[i]
        if not free[m]:
            stock += float(net[m])
            i += 1
            continue
        nxt = [a for a in sorted(anchors) if a >= m]
        if not nxt:
            # rest of this calendar year (or to horizon): demean free nets to 0
            yr = m.year
            win = [x for x in months[i:] if x.year == yr]
            free_ix = [x for x in win if free[x]]
            if free_ix:
                mu = float(sum(net[x] for x in free_ix)) / len(free_ix)
                for x in free_ix:
                    net[x] = float(net[x]) - mu
            for x in win:
                stock += float(net[x])
            i = months.index(win[-1]) + 1
            continue
        A = nxt[0]
        win = [x for x in months if m <= x <= A]
        free_ix = [x for x in win if free[x]]
        pinned_sum = float(sum(net[x] for x in win if not free[x]))
        free_sum = float(sum(net[x] for x in free_ix))
        n = len(free_ix)
        if n:
            shift = (float(anchors[A]) - stock - pinned_sum - free_sum) / n
            for x in free_ix:
                net[x] = float(net[x]) + shift
        for x in win:
            stock += float(net[x])
        i = months.index(A) + 1

    red = ladder.astype(float).copy()
    gross = pd.Series(0.0, index=need.index)
    for i, m in enumerate(need.index):
        if m in observed:
            gross.iloc[i] = float(on_tape[m])
            red.iloc[i] = gross.iloc[i] - float(net[m])
            net[m] = float(net[m])
        elif pd.notna(fixed.get(m, np.nan)):
            gross.iloc[i] = float(fixed[m])
            net[m] = gross.iloc[i] - float(red.iloc[i])
        else:
            gross.iloc[i] = max(float(net[m]) + float(red.iloc[i]), 0.0)
            net[m] = gross.iloc[i] - float(red.iloc[i])
        new = max(gross.iloc[i] - float(on_tape.get(m, 0.0)), 0.0)
        for k, w in shares.items():
            if i + k < len(red):
                red.iloc[i + k] += new * w
    return gross, red, net


def plan_other_fin(p: dict) -> float | None:
    """Other financing implied by a plan year: deficit − (net programme − amortisation) − BTF change."""
    if p.get("deficit_to_finance") is None or p.get("amortisation") is None:
        return None
    return p["deficit_to_finance"] - (p["net_mlt_issuance"] - p["amortisation"]) - p["btf_change"]


def forward(h: pd.DataFrame, cash: pd.Series, model: str, gap_mode: str, debt: pd.DataFrame,
            mlt: pd.Series, ops: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sovereign_cash.cash_season import HORIZON_END, forward_months as _fwd_m

    plan = F.load_json("plan.json")["years"]
    last_cash = cash.dropna().index.max()
    asof_fin = h.index.max()
    h_live = _fwd_m(last_cash)
    end = max(last_cash + pd.DateOffset(months=h_live), HORIZON_END)
    months = pd.date_range(asof_fin + pd.offsets.MonthBegin(1), end, freq="MS")
    if model.startswith("annual_comp_"):
        from sovereign_cash.annual_macro import component_book
        rule = "unit" if "unit" in model else "floor"
        book = component_book("fr", rule=rule)
        tg = book.get(last_cash) or book.get(max(book)) if book else None
        seas_fc = C.forecast(cash.dropna(), last_cash, h_live, model, targets=tg)
    elif model.startswith("annual_"):
        from sovereign_cash.annual_macro import macro as _macro, targets_asof
        keys = ("ngdp",) if "ngdp" in model else ("ngdp", "cpi")
        tg = targets_asof(cash.dropna(), _macro("fr"), last_cash, keys)
        seas_fc = C.forecast(cash.dropna(), last_cash, h_live, model, targets=tg)
    else:
        seas_fc = C.forecast(cash.dropna(), last_cash, h_live, model)
    from sovereign_cash.fr import bottom_up as BU
    from sovereign_cash.fr import bottom_up_macro as BUM
    # v2 PLF×RESF macro when shipped; else v1 component seas (also gated); else seasonal.
    if BUM.SHIP:
        bu_fc = BUM.path(last_cash, h_live)
        bu_live = True
    else:
        bu_fc = BU.path(last_cash, h_live)
        bu_live = bool(BU.SHIP and bu_fc is not None)
    shape_fc = bu_fc if (bu_live and bu_fc is not None) else seas_fc
    plans = {int(y): float(p["deficit_to_finance"])
             for y, p in plan.items() if p.get("deficit_to_finance") is not None}
    # Bottom-up macro already soft-pins to the PLF solde inside BUM (α=1 when the
    # vintage is known). Do not double-anchor here.
    if BUM.SHIP and bu_fc is not None:
        fc = shape_fc
    elif plans:
        fc = C.anchor_to_plan(shape_fc, cash.dropna(), last_cash, plans)
    else:
        fc = shape_fc
    # Beyond the year in progress the board runs on our component model, the path the Shock scenarios page
    # shows (SMB lines on the RESF macro paths, interest from the debt book). No PLF total is pinned there.
    from sovereign_cash.scenarios.channels.sarimax import later_years

    ours = later_years("fr", months, last_cash.year)  # no-policy path + legislated overlay
    if len(ours):
        fc = fc.copy()
        for m, v in ours.items():
            fc.loc[m] = float(v)
        fc = fc.sort_index()
    enc, coef_notes = oat_ladder(months)
    enc = enc[enc["month"] <= months.max()].copy()
    cf = coupon_forward(ops, months, plan, enc)
    cb = cb_holdings(enc)
    obs = [m for m in months if m in debt.index and pd.notna(debt.at[m, "btf"]) and pd.notna(mlt.get(m))]
    f = pd.DataFrame(index=months)
    f["cash_requirement_bn"] = [float(cash[m]) if m <= last_cash else float(fc[m]) for m in months]
    f["cash_requirement_seas_bn"] = [float(cash[m]) if m <= last_cash else float(seas_fc[m]) for m in months]
    f["cash_requirement_bottomup_bn"] = [
        float(cash[m]) if m <= last_cash else (float(bu_fc[m]) if bu_fc is not None and m in bu_fc.index else np.nan)
        for m in months]
    # Outer years already include the policy overlay via later_years; non-zero
    # policy_bn marks that so Ex1c does not add the overlay a second time.
    f["policy_bn"] = 0.0
    if len(ours):
        f.loc[ours.index, "policy_bn"] = 1.0  # flag only (magnitude lives in cash_requirement)
    f["cash_observed"] = [m <= last_cash for m in months]
    f["financing_observed"] = [m in obs for m in months]
    f["bond_gross_bn"] = cf["bond_gross_bn"]
    f["bond_redemption_bn"] = cf["bond_redemption_bn"]
    f["bond_buyback_bn"] = cf["buyback_bn"]
    for m in obs:
        f.at[m, "bond_gross_bn"] = float(cf.at[m, "tape_bn"])
        f.at[m, "bond_redemption_bn"] = float(cf.at[m, "tape_bn"]) - float(mlt[m] - mlt[m - pd.offsets.MonthBegin(1)])
        f.at[m, "bond_buyback_bn"] = np.nan
    f["net_bonds_bn"] = f["bond_gross_bn"] - f["bond_redemption_bn"]
    f["bond_redemption_cb_bn"] = cb.groupby("month")["cb_cash_bn"].sum().reindex(months).fillna(0.0)
    f["private_net_bonds_bn"] = f["net_bonds_bn"] + f["bond_redemption_cb_bn"]

    yr = last_cash.year
    gseas = h["other_fin_bn"].iloc[-60:].groupby(h.index[-60:].month).mean()
    gseas = gseas - gseas.mean()
    seas = pd.Series([float(gseas.get(m.month, 0.0)) if gap_mode == "gapseas" else 0.0 for m in months], index=months)
    rem = [m for m in months if m.year == yr]
    seas[rem] = seas[rem] - seas[rem].mean()
    realised = float(h.loc[h.index.year == yr, "other_fin_bn"].sum())
    p_of = plan_other_fin(plan[str(yr)])
    base, of_basis = pd.Series(0.0, index=months), {}
    for y in sorted(set(months.year)):
        in_y = [m for m in months if m.year == y]
        py = plan.get(str(y)) or plan[max(plan)]
        if y == yr and p_of is not None:
            base[in_y] = (p_of - realised) / len(in_y)
            of_basis[y] = "plan_remaining"
        else:
            base[in_y] = py["other_resources"] / 12.0
            of_basis[y] = "other_resources"
    f["other_fin_aft_bn"] = base
    f["other_fin_seas_bn"] = seas
    f["other_fin_bn"] = base + seas

    need = f["cash_requirement_bn"] - f["net_bonds_bn"] - f["other_fin_bn"]
    bills = ops[ops["sec_type"].astype(str) == "Bill"]
    last_bill = bills["op_dt"].max()
    tape_b = _ops_monthly(ops, ["Bill"]).reindex(months).fillna(0.0)
    supp, supp_used = btf_supplement(months, ops)
    tape_end = (last_bill + pd.offsets.BDay(2)).to_period("M").to_timestamp()
    fixed_end = max([tape_end] + [pd.Timestamp(a["settle_dt"]).to_period("M").to_timestamp() for a in supp_used])
    fixed = (tape_b + supp).where(months <= fixed_end)
    ladder = C.ladder_monthly(bills, last_bill, months, ["Bill"])
    shares = C.bill_tenor_shares(ops, last_bill, ["Bill"])
    s0 = float(h["bill_stock_bn"].iloc[-1])
    anchors, lvl = {}, float(debt.at[pd.Timestamp(yr - 1, 12, 1), "btf"])
    for y in (yr, yr + 1):
        py = plan.get(str(y))
        if py and py.get("btf_change") is not None:
            lvl += py["btf_change"]
            anchors[pd.Timestamp(y, 12, 1)] = lvl
    gross, red, net = btf_path(need, ladder, shares, tape_b, fixed,
                               {m: float(debt.at[m, "btf"] - debt.at[m - pd.offsets.MonthBegin(1), "btf"]) for m in obs},
                               s0, anchors)
    f["bill_gross_bn"], f["bill_redemption_bn"], f["net_bills_bn"] = gross, red, net
    f["buffer_change_bn"] = f["net_bills_bn"] - need
    f["buffer_level_implied_bn"] = f["buffer_change_bn"].cumsum()
    f["bill_stock_implied_bn"] = s0 + f["net_bills_bn"].cumsum()
    f["bill_stock_bn"] = debt["btf"].reindex(months)
    f["oat_stock_bn"] = mlt.reindex(months)
    f["fc_source"] = [
        ("cash observed (SMB); " if m <= last_cash else (
            "cash component model (as the Shock scenarios page); " if m in ours.index else
            "cash PLF×RESF categories; " if (BUM.SHIP and bu_fc is not None) else
            ("cash bottom-up components × PLF/LFI deficit; " if (bu_live and bu_fc is not None and m.year in plans) else
             "cash bottom-up components; " if (bu_live and bu_fc is not None) else
             "cash PLF/LFI deficit × seasonal shape; " if m.year in plans else f"cash {model}; ")))
        + ("OAT/BTF observed (AFT stock)" if m in obs else
           f"OAT {'tape' if m < cf.attrs['last_tape_month'] else 'programme'}; "
           f"BTF {'tape' if m <= tape_end else ('tape+supplement' if pd.notna(fixed.get(m)) else 'need→YE anchor')}")
        for m in months]
    f["is_forecast"] = True
    info = {"origin_cash": last_cash, "origin_financing": asof_fin, "financing_observed_months": obs,
            "gap_mode": gap_mode, "bill_tenor_shares": shares, "btf_tape_last_op": last_bill,
            "btf_fixed_end": fixed_end, "btf_supplement_used": supp_used, "btf_anchors": anchors,
            "btf_supplement_bn": float(supp.sum()),
            "other_fin_basis": of_basis, "other_fin_realised_ytd_bn": realised,
            "other_fin_plan_fy_bn": p_of,
            "deficit_plan_years": sorted(plans), "enc": enc, "cb": cb, "coef_notes": coef_notes,
            "plan": plan, "bottom_up_ship": bool(bu_live and bu_fc is not None),
            "bottom_up_macro_ship": bool(BUM.SHIP and bu_fc is not None),
            "bottom_up_available": bu_fc is not None, **cf.attrs}
    try:
        from sovereign_cash.fr.bottom_up_macro import ship_decision as _bum_ship
        v = _bum_ship()
        info["bottomup_pre_mean"] = v.get("pre_bu")
        info["bottomup_oos_mean"] = v.get("oos_bu")
        info["bottomup_pre_production_mean"] = v.get("pre_prod")
        info["bottomup_oos_production_mean"] = v.get("oos_prod")
        info["bottomup_reason"] = v.get("reason")
    except Exception:  # noqa: BLE001
        pass
    return f, info


def oat_reconcile(frame: pd.DataFrame, plan: dict, yr: int, debt: pd.DataFrame, ops: pd.DataFrame,
                  enc_all: pd.DataFrame | None = None) -> dict | None:
    """Calendar-year net OAT (Δ stock, history + forward) vs programme − amortisation ± indexation.

    indexation = accrual on the indexed stock (trailing inflation-reference growth) + uplift on
    linker issuance (auction index coefficients) − indexation paid on linkers maturing in the year
    (face issued × (ratio at maturity − 1), an upper bound: buybacks are ignored)."""
    p = plan.get(str(yr))
    if not p or p.get("amortisation") is None:
        return None
    d = frame.set_index("month")
    dec0 = pd.Timestamp(yr - 1, 12, 1)
    dec1 = pd.Timestamp(yr, 12, 1)
    if dec0 not in debt.index or dec1 not in d.index:
        return None
    net = float(d.loc[(d.index > dec0) & (d.index <= dec1), "net_bonds_bn"].sum())
    hist = F.load_mlt_auctions()
    lk = hist[(hist["op_dt"].dt.year == yr) & (hist["coef"] > 1.0)]
    uplift = float((lk["total_mn"] * (lk["coef"] - 1.0)).sum()) / 1000
    g = F.oatei_coef(pd.Timestamp(yr, 12, 31), "oatei")[1]
    try:
        growth = float(g.split("× ")[1].split("/yr")[0]) - 1.0
    except (IndexError, ValueError):
        growth = 0.02
    accrual = float(debt.at[dec0, "indexed"]) * growth
    c = ops[ops["sec_type"].astype(str).eq("Linker") & (ops["amount_issued"] > 0)]
    lines = c.groupby(c["isin"].astype(str)).agg(face=("amount_issued", "sum"), mat=("maturity_dt", "first"),
                                                 desc=("sec_desc_long", "first"))
    lines = lines[pd.to_datetime(lines["mat"]).dt.year == yr]
    paid = 0.0
    for k, r in lines.iterrows():
        kind = "oatei" if "€" in str(r["desc"]) else "oati"
        coef = F.oatei_coef(pd.Timestamp(r["mat"]), kind)[0]
        paid += float(r["face"]) / 1000 * (coef - 1.0)
    idx = accrual + uplift - paid
    expected = p["net_mlt_issuance"] - p["amortisation"] + idx
    return {"year": yr, "net_oat_bn": net, "programme_bn": p["net_mlt_issuance"], "amortisation_bn": p["amortisation"],
            "amortisation_vintage": p.get("amortisation_vintage", ""), "indexation_bn": idx,
            "accrual_bn": accrual, "uplift_bn": uplift, "indexation_paid_bn": paid, "expected_bn": expected,
            "gap_bn": net - expected, "within_tol": abs(net - expected) <= RECON_TOL_BN, "tol_bn": RECON_TOL_BN}
