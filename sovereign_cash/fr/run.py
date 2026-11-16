"""France State cash framework: history identity, deficit scores, BTF path, contract outputs, dashboard.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.fr.run            # cached inputs
  PYTHONPATH=. .venv/bin/python -m sovereign_cash.fr.run --refresh  # re-fetch INSEE, AFT outstanding, index ratios
"""

from __future__ import annotations

import argparse

import pandas as pd

from sovereign_cash.contract import DERIVED, OPTIONAL, REQUIRED
from sovereign_cash.core import cash_framework as C
from sovereign_cash.fr import dashboard as P
from sovereign_cash.fr import fetch as F
from sovereign_cash.fr import model as M
from sovereign_cash.fr.dashboard import eur, label, num

EXTRA = ["oat_stock_bn", "indexed_stock_bn", "account_pinned", "cash_observed", "financing_observed",
         "cash_requirement_seas_bn", "cash_requirement_bottomup_bn",
         "bond_buyback_bn", "bond_redemption_cb_bn", "private_net_bonds_bn", "other_fin_aft_bn",
         "other_fin_seas_bn", "fc_source"]
LANDS_ON = "BTF (AFT revises the BTF change)"
CIT_MONTHS = (3, 6, 9, 12)


def choose_gap(bsc: pd.DataFrame) -> tuple[str, str]:
    pre = bsc[(bsc["window"] == "PRE") & bsc["common"] & bsc["model"].isin(["gap0", "gapseas"])]
    if pre.empty or pre.loc[pre["h"] == 12, "n"].max() < C.MIN_PRE:
        return "gap0", "PRE too short for the gap rule; gap = 0 by default"
    m = pre.groupby("model")["mae"].mean()
    return m.idxmin(), "lowest mean PRE BTF-stock MAE"


def _beats(bsc: pd.DataFrame, m: str) -> list[str]:
    w = []
    for (win, h), g in bsc[bsc["window"].isin(["PRE", "OOS"])].groupby(["window", "h"]):
        g = g.set_index("model")["mae"]
        if m in g and "flat" in g and g[m] < g["flat"]:
            w.append(f"{win} h={h}")
    return w


def _span(idx) -> str:
    idx = pd.DatetimeIndex(idx).dropna()
    return f"{idx.min():%b-%Y} → {idx.max():%b-%Y}" if len(idx) else "none"


def _band(sc: pd.DataFrame, model: str, h_rem: int) -> tuple[float | None, int | None]:
    """OOS MAE of the production model at the first scored horizon ≥ the months left in the year."""
    hs = [h for h in C.HORIZONS if h >= h_rem] or [max(C.HORIZONS)]
    g = sc[(sc["window"] == "OOS") & (sc["model"] == model) & (sc["h"] == hs[0])]
    return (float(g["mae"].iloc[0]), hs[0]) if len(g) else (None, None)


def run(refresh: bool = False) -> dict:
    if refresh:
        F.load_smb(refresh=True)
        F.load_debt(refresh=True)
        F.load_aft_encours(refresh=True)
        for k in F.COEF_PAGES:
            try:
                F.fetch_coef_file(k)
            except Exception as e:  # noqa: BLE001 — a failed fetch keeps the cached file
                print(f"FR index coefficients {k}: {e}")
    h, meta = M.build_history()
    cash = meta["cash_full"].dropna()
    debt, ops = meta["debt"], meta["ops"]
    bt, sc, model, why = M.deficit_scores(cash)
    bb = M.bill_backtest(h, cash, model)
    bsc = C.score_table(bb)
    gap_mode, gap_why = choose_gap(bsc)
    f, info = M.forward(h, cash, model, gap_mode, debt, meta["mlt"], ops)

    cols = list(dict.fromkeys(
        list(REQUIRED[1:]) + [c for c in OPTIONAL if c in h or c in f] + EXTRA))
    frame = pd.concat([h, f], axis=0)
    frame = frame[[c for c in cols if c in frame.columns]].rename_axis("month").reset_index()
    frame["is_forecast"] = frame["is_forecast"].astype(bool)
    for c in ("cash_observed", "financing_observed"):
        frame[c] = frame[c].fillna(c == "financing_observed").astype(bool)

    bench = "smly" if model != "smly" else "prevyear_3"
    score = C.score_records(sc, model, bench)
    bscore = C.score_records(bsc, gap_mode, "flat")

    last_cash = cash.index.max()
    yr = last_cash.year
    plan = info["plan"]
    p_yr = plan[str(yr)]
    p_nx = plan.get(str(yr + 1)) or {}
    prog = info["programme"]
    fc12 = f.loc[f.index > last_cash]
    fd = frame.set_index("month")
    yrs = fd[fd.index.year == yr]
    last_debt = debt["btf"].dropna().index.max()
    dec, dec_prev = pd.Timestamp(yr, 12, 1), pd.Timestamp(yr - 1, 12, 1)
    plan_btf_end = float(debt.at[dec_prev, "btf"] + p_yr["btf_change"])
    model_fy = float(yrs["cash_requirement_bn"].sum())
    band, band_h = _band(sc, model, 12 - last_cash.month)
    trail12 = float(cash.iloc[-12:].sum())
    other_needs = p_yr["besoin"] - p_yr["amortisation"] - p_yr["deficit_to_finance"]
    nx_def_plus = (p_nx["besoin"] - p_nx["amortisation"]) if p_nx.get("amortisation") is not None else None
    btf_dec = float(f.at[dec, "bill_stock_implied_bn"]) if dec in f.index else None
    place_dec = float(f.loc[:dec, "buffer_change_bn"].sum())
    place_act = float(f.loc[:last_debt, "buffer_change_bn"].sum())
    fixed_end = info["btf_fixed_end"]
    btf_fixed = float(f.at[fixed_end, "bill_stock_implied_bn"]) if fixed_end in f.index and fixed_end > last_debt else None
    recon = M.oat_reconcile(frame, plan, yr, debt, ops)
    net_fy = float(yrs["net_bonds_bn"].sum())
    of_fy = float(yrs["other_fin_bn"].sum())
    p_net = p_yr["net_mlt_issuance"] - p_yr["amortisation"]
    p_of = info["other_fin_plan_fy_bn"]
    decomp = None
    if btf_dec is not None and p_of is not None:
        decomp = {"net_oat": net_fy - p_net, "deficit": -(model_fy - p_yr["deficit_to_finance"]),
                  "other_fin": of_fy - p_of, "btf": btf_dec - plan_btf_end}
    cb = info["cb"]
    enc = info["enc"]
    bbf = info["buybacks_forward_by_year_bn"]

    def s12(c):
        return round(float(fc12[c].fillna(0).sum()), 1)

    forward = {
        "window": f"{fc12.index.min():%Y-%m} → {fc12.index.max():%Y-%m}",
        "model": model,
        "model_reason": why,
        "cash_requirement_12m_bn": s12("cash_requirement_bn"),
        "cash_requirement_trailing_12m_bn": round(trail12, 1),
        "oat_gross_12m_bn": s12("bond_gross_bn"),
        "oat_redemptions_12m_bn": s12("bond_redemption_bn"),
        "oat_buybacks_12m_bn": s12("bond_buyback_bn"),
        "net_bonds_12m_bn": s12("net_bonds_bn"),
        "cb_held_redemptions_12m_bn": s12("bond_redemption_cb_bn"),
        "cb_held_basis": f"estimated: Eurosystem ISIN allocation (snapshot {cb.attrs.get('snapshot')}) scaled to AFT "
                         "outstanding net of buybacks, linkers at index ratio",
        "private_net_bonds_12m_bn": s12("private_net_bonds_bn"),
        "net_bills_12m_bn": s12("net_bills_bn"),
        "other_fin_12m_bn": s12("other_fin_bn"),
        "placements_12m_bn": s12("buffer_change_bn"),
        "btf_stock_last_actual_bn": round(float(debt.at[last_debt, "btf"]), 1),
        "btf_stock_last_actual_month": f"{last_debt:%Y-%m}",
        "btf_stock_from_auctions_bn": None if btf_fixed is None else round(btf_fixed, 1),
        "btf_stock_from_auctions_month": f"{fixed_end:%Y-%m}",
        "btf_stock_end_year_bn": None if btf_dec is None else round(btf_dec, 1),
        "btf_stock_implied_end_bn": round(float(f["bill_stock_implied_bn"].iloc[-1]), 1),
        "plan_btf_stock_end_year_bn": round(plan_btf_end, 1),
        "placements_dec_bn": round(place_dec, 1),
        "placements_dec_decomposition_bn": None if decomp is None else {k: round(v, 1) for k, v in decomp.items()},
        "btf_anchors": {f"{k:%Y-%m}": round(v, 1) for k, v in info["btf_anchors"].items()},
        "plan_year": yr,
        "plan_cash_requirement_bn": p_yr["deficit_to_finance"],
        "plan_source": p_yr["deficit_vintage"],
        "plan_besoin_bn": p_yr["besoin"],
        "plan_amortisation_bn": p_yr["amortisation"],
        "plan_other_needs_bn": round(other_needs, 1),
        "plan_net_mlt_issuance_bn": p_yr["net_mlt_issuance"],
        "plan_other_resources_bn": p_yr["other_resources"],
        "plan_other_fin_implied_bn": None if p_of is None else round(p_of, 1),
        "plan_next_year": {**p_nx, "deficit_plus_other_needs_bn": None if nx_def_plus is None else round(nx_def_plus, 1)},
        "cash_requirement_ytd_bn": round(float(cash[cash.index.year == yr].sum()), 1),
        "ytd_through": f"{last_cash:%Y-%m}",
        "model_full_year_bn": round(model_fy, 1),
        "model_full_year_oos_mae_bn": None if band is None else round(band, 1),
        "model_full_year_oos_mae_h": band_h,
        "plan_gap_lands_on": LANDS_ON,
        "programme": {k: (str(v.date()) if isinstance(v, pd.Timestamp) else round(v, 4) if isinstance(v, float) else v)
                      for k, v in prog.items()},
        "oat_reconciliation": None if recon is None else {k: (round(v, 2) if isinstance(v, float) else v)
                                                          for k, v in recon.items()},
        "oat_stock_jumps": meta["oat_jumps"],
        "onc_added_bn": ops.attrs.get("onc_added_bn", {}),
        "other_fin_basis": info["other_fin_basis"],
        "other_fin_realised_ytd_bn": round(info["other_fin_realised_ytd_bn"], 1),
        "gap_mode": gap_mode,
        "buybacks_forward_by_year_bn": {k: round(v, 1) for k, v in bbf.items()},
        "btf_tape_last_op": str(info["btf_tape_last_op"].date()),
        "oat_tape_last_op": str(info["tape_last_op"].date()),
        "btf_supplement_used": info["btf_supplement_used"],
        "financing_observed_months": [f"{m:%Y-%m}" for m in info["financing_observed_months"]],
        "oat_ladder_lines": [{"isin": r["isin"], "line": r["label"], "maturity": str(r["maturity"].date()),
                              "nominal_bn": round(r["nominal_bn"], 3), "coef": round(r["coef"], 4),
                              "redemption_bn": round(r["redemption_bn"], 3),
                              "cb_cash_bn": round(float(cb.at[i, "cb_cash_bn"]), 3)} for i, r in enc.iterrows()],
        "index_ratio_basis": info["coef_notes"],
    }
    pre_n12 = sc[(sc.window == "PRE") & (sc.h == 12) & sc.common]["n"].max()
    oos_n12 = sc[(sc.window == "OOS") & (sc.h == 12) & sc.common]["n"].max()
    score_note = (
        f"Walk-forward, monthly origins. PRE = windows ending by Dec-2019 (h=12 n={int(pre_n12)}); OOS = origins "
        f"from Jan-2022 (h=12 n={int(oos_n12)}); windows touching 2020-21 excluded. Production: {label(model)}, "
        f"lowest mean PRE MAE. At h=12 every trailing-level model forecasts the trailing 12-month sum, so h=12 "
        f"scores tie by construction."
    )

    # ---------------------------------------------------------------- page text (generated)
    coef_file = F.coef_file("oatei").name
    coef_to = F.coef_asof("oatei")
    jumps = meta["oat_jumps"]
    notes = [
        f"Cash requirement = SMB State deficit (solde général d'exécution, special accounts included). The AFT besoin "
        f"({eur(p_yr['besoin'])}) adds amortisation ({eur(p_yr['amortisation'])}) and other treasury needs "
        f"({eur(other_needs)}).",
        (
            f"Cash path = bottom-up SMB components (recettes/primary {label('seas_mult_3')}, AFT charge × month share, "
            f"speciaux moy) then closed to the published State deficit (PLF/LFI solde hors FMI = déficit à "
            f"financer); {yr} = {eur(p_yr['deficit_to_finance'])} ({p_yr['deficit_vintage']}). "
            if info.get("bottom_up_ship") else
            f"Cash path = annual level × seasonal month shape. Calendar years with a published State deficit "
            f"(PLF/LFI solde budgétaire hors FMI = déficit à financer) are closed to that total; {yr} = "
            f"{eur(p_yr['deficit_to_finance'])} ({p_yr['deficit_vintage']}). Years without one keep "
            f"{label(model)}. "
        )
        + f"AFT publishes besoin / amortisation / BTF / charge; the deficit split is in the "
        f"PLF financing table (update {pd.Timestamp(p_yr['update_date']):%d-%b-%Y}).",
        "Banque de France account pinned since May-2023: Treasury placements sit inside other financing on history "
        "and are shown separately on forecast months.",
    ]
    for j in jumps[-2:]:
        notes.append(f"AFT/INSEE OAT stock {pd.Timestamp(j['month']):%b-%Y}: net {eur(j['net_bn'])} vs gross "
                     f"{eur(j['gross_bn'])}, after {pd.Timestamp(j['prior_month']):%b-%Y} net {eur(j['prior_net_bn'])} vs "
                     f"gross {eur(j['prior_gross_bn'])}. Kept as published; the swing sits in other financing.")
    if recon:
        notes.append(
            f"{yr} net OAT {eur(recon['net_oat_bn'])} vs programme − amortisation ± indexation "
            f"{eur(recon['expected_bn'])} (indexation {eur(recon['indexation_bn'], plus=True)}): gap "
            f"{eur(recon['gap_bn'], plus=True)}"
            + ("." if recon["within_tol"] else
               f", beyond the {eur(recon['tol_bn'])} tolerance. The stock grew more than the programme arithmetic "
               f"allows; candidates are buybacks of {yr} lines after the {recon['amortisation_vintage']} amortisation "
               "figure and the stock swing above."))
    if ops.attrs.get("onc_added_bn", {}).get(yr):
        notes.append(f"OAT gross uses AFT auction-history totals incl. non-competitive bids "
                     f"({eur(ops.attrs['onc_added_bn'][yr])} above the tape in {yr}).")
    notes.append(f"Linker index ratios: AFT file {coef_file} (to {coef_to:%d-%b-%Y}), extrapolated at trailing "
                 "inflation-reference growth.")
    if info["btf_supplement_used"]:
        notes.append(f"BTF auctions added by hand (not yet on the tape): "
                     + ", ".join(f"{pd.Timestamp(a['op_dt']):%d-%b} {eur(a['total_mn'] / 1000)}" for a in info["btf_supplement_used"]) + ".")
    beats = _beats(bsc, gap_mode)
    notes.append(f"BTF forward steps to the AFT year-end level rather than to the identity: the identity path "
                 f"({label(gap_mode)}) beats a flat stock only at {', '.join(beats) if beats else 'no horizon'}.")

    sources = [
        {"name": "SMB State deficit via INSEE BDM (YTD solde général d'exécution)",
         "url": "https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/001717255", "frequency": "M", "lag_days": 40,
         "verified": True, "history": _span(meta["smb"]["solde_ytd"].dropna().index)},
        {"name": "AFT negotiable debt outstanding via INSEE BDM (BTF / >1y / indexed)",
         "url": "https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/001711532", "frequency": "M", "lag_days": 30,
         "verified": True, "history": _span(debt["btf"].dropna().index)},
        {"name": "AFT outstanding by line (OAT / OAT€i / OATi)", "url": F.AFT_ENCOURS.format(kind="oat"),
         "frequency": "D", "lag_days": 1, "verified": True,
         "history": f"snapshot {pd.Timestamp((F.DATA / 'aft_encours_oat.html').stat().st_mtime, unit='s'):%d-%b-%Y}"},
        {"name": "AFT linker index coefficients", "url": F.AFT + F.COEF_PAGES["oatei"][0], "frequency": "M",
         "lag_days": 0, "verified": True, "history": f"to {coef_to:%d-%b-%Y}"},
        {"name": "AFT auctions and syndications (tape + auction history)", "url": "https://www.aft.gouv.fr",
         "frequency": "D", "lag_days": 1, "verified": True,
         "history": f"OAT to {info['tape_last_op']:%d-%b-%Y}, BTF to {info['btf_tape_last_op']:%d-%b-%Y}"},
        {"name": "PLF/AFT financement (plan.json): déficit à financer, besoin, BTF, charge",
         "url": "https://www.budget.gouv.fr/", "frequency": "A", "lag_days": 0, "verified": True,
         "history": f"PLF/AFT {pd.Timestamp(p_yr['update_date']):%d-%b-%Y}"},
        {"name": "Eurosystem PSPP+PEPP holdings by ISIN (estimated allocation)",
         "url": "issuance/cache/official_isin_flow.parquet", "frequency": "M", "lag_days": 30, "verified": False,
         "history": f"snapshot {cb.attrs.get('snapshot')}"},
    ]
    from sovereign_cash import scenarios as SX
    sx = SX.run("fr")
    SX.save("fr", sx)
    notes.append(
        "Authority stack: DG Trésor RESF (growth/CPI) + PLF category totals as year pins where known + "
        "AFT charge / +100bp triangle (interest). "
        + ("Live cash = PLF×RESF category path (no LFI solde re-pin). "
           if info.get("bottom_up_macro_ship") or info.get("bottom_up_ship") else
           "Live cash = plan deficit × PRE seasonal shape (component SARIMAX loses PRE). ")
        + "Scenario card = model-implied Δ off that path."
    )
    forward["bottom_up_ship"] = bool(info.get("bottom_up_ship"))
    forward["bottomup_macro_ship"] = bool(info.get("bottom_up_macro_ship"))
    for k in ("bottomup_pre_mean", "bottomup_oos_mean",
              "bottomup_pre_production_mean", "bottomup_oos_production_mean",
              "bottomup_reason"):
        if info.get(k) is not None:
            forward[k] = info[k]
    forward["cash_requirement_bottomup_12m_bn"] = (
        None if "cash_requirement_bottomup_bn" not in f.columns else
        round(float(fc12["cash_requirement_bottomup_bn"].sum()), 1)
    )
    summary = {
        "country": "FR", "ccy": "EUR", "asof": f"{last_cash:%Y-%m}",
        "sources": sources,
        "buffer": ("Banque de France Treasury account pinned at EUR 120-160mn since May-2023 (surplus placed in "
                   "market): liquidity-neutral. BTF is the flex; forward BTF follows the AFT year-end path and "
                   "Treasury placements (buffer_change_bn) take the model's departure from it."),
        "score": score, "score_note": score_note,
        "bill_stock_score": bscore,
        "forward": forward, "notes": notes, "scenarios": sx,
    }
    C.write_outputs("fr", frame, summary, DERIVED)

    # ---------------------------------------------------------------- dashboard
    F_ = forward
    today = pd.Timestamp.today().normalize().replace(day=1)
    band_txt = f" ± {eur(band)}" if band else ""
    kpis = [
        ("Private net OAT, next 12m (est.)", eur(F_["private_net_bonds_12m_bn"]),
         f"net {eur(F_['net_bonds_12m_bn'])} + Eurosystem-held {eur(F_['cb_held_redemptions_12m_bn'])}"),
        (f"State deficit {yr} to {last_cash:%b}", eur(F_["cash_requirement_ytd_bn"]),
         f"plan full year {eur(p_yr['deficit_to_finance'])} · {p_yr['deficit_vintage']}"),
        (f"Model full year {yr}", eur(model_fy), f"{band_txt.strip()} OOS MAE at h={band_h}" if band else label(model)),
        ("State deficit, trailing 12m", eur(trail12), f"to {last_cash:%b-%Y}"),
        (f"Placements at Dec-{yr}", eur(place_dec, plus=True),
         f"model need vs AFT path; BTF held on AFT {eur(plan_btf_end)}"),
        (f"BTF {last_debt:%b-%Y}", eur(F_["btf_stock_last_actual_bn"]),
         f"end-{fixed_end:%b} from auctions {eur(btf_fixed)}" if btf_fixed is not None else "AFT/INSEE"),
    ]

    ytd_rows = []
    for y in range(yr - 4, yr):
        sy = cash[cash.index.year == y]
        if len(sy) == 12:
            ytd_rows.append([str(y), eur(sy[sy.index.month <= last_cash.month].sum()), eur(sy.sum()),
                             eur(sy[sy.index.month > last_cash.month].sum(), plus=True)])
    rem_model = model_fy - F_["cash_requirement_ytd_bn"]
    ytd_rows.append([f"{yr} model", eur(F_["cash_requirement_ytd_bn"]), eur(model_fy) + band_txt, eur(rem_model, plus=True)])
    ytd_rows.append([f"{yr} plan", eur(F_["cash_requirement_ytd_bn"]), eur(p_yr["deficit_to_finance"]),
                     eur(p_yr["deficit_to_finance"] - F_["cash_requirement_ytd_bn"], plus=True)])
    mo = f"{last_cash:%b}"
    lead = [(f"State deficit: year to {mo} vs full year", P.table(
        ytd_rows, ["year", f"Jan–{mo}", "full year", f"rest of year"]))]

    prog_rows = [
        ["net programme (net of buybacks)", eur(prog["programme_bn"])],
        [f"done at {prog['asof']:%d-%b-%Y} (AFT)", f"{prog['done_share']:.0%} = {eur(prog['net_done_bn'])}"],
        ["gross issued to that date", eur(prog["gross_bn"])],
        ["implied buybacks", eur(prog["buybacks_bn"])],
        ["net left", eur(prog["net_left_bn"])],
        [f"forward buybacks rest of {yr}", eur(bbf.get(yr, 0.0))],
    ]
    if p_nx:
        prog_rows.append([f"{yr + 1} net programme", eur(p_nx["net_mlt_issuance"])])
    lead.append((f"OAT programme {prog['year']}", P.table(prog_rows, ["", "EUR"])))

    btf_rows = [["BTF stock", eur(F_["btf_stock_last_actual_bn"]), eur(btf_fixed) if btf_fixed is not None else "—",
                 eur(btf_dec), eur(plan_btf_end)],
                ["Treasury placements, cumulative", eur(place_act), "—", eur(place_dec), "—"]]
    hdr = ["", f"{last_debt:%b-%Y} actual", f"end-{fixed_end:%b} from auctions", f"Dec-{yr} path", f"Dec-{yr} AFT"]
    txt = ""
    if decomp:
        txt = (f"<p class='sub'>BTF absorbs the monthly financing need then shifts evenly onto the AFT year-end "
               f"level; placements take the residual. Placements at Dec-{yr} "
               f"{eur(place_dec, plus=True)} = net OAT vs plan {eur(decomp['net_oat'], plus=True)} + deficit vs plan "
               f"{eur(decomp['deficit'], plus=True)} + other financing vs plan {eur(decomp['other_fin'], plus=True)}"
               + (f" + BTF vs AFT {eur(decomp['btf'], plus=True)}" if abs(decomp["btf"]) >= 0.05 else "") + ".</p>")
    lead.append((f"BTF vs the AFT end-{yr} target", P.table(btf_rows, hdr) + txt))

    cal, fut = [], f[f.index >= today]
    for m, r in fut.iterrows():
        tax = f"15-{m:%b}" if m.month in CIT_MONTHS else ""
        bbv = 0.0 if pd.isna(r["bond_buyback_bn"]) else float(r["bond_buyback_bn"])
        cal.append([f"{m:%b-%Y}", num(r["bond_gross_bn"]), num(r["bond_redemption_bn"] - bbv), num(bbv), num(r["bond_redemption_cb_bn"]), num(r["bill_gross_bn"]),
                    num(r["bill_redemption_bn"]), num(r["bill_stock_implied_bn"]), tax])
    lad = enc[enc["maturity"] >= today]
    lad_rows = [[f"{r['maturity']:%d-%b-%Y}", r["label"], num(r["nominal_bn"], 2), f"{r['coef']:.4f}",
                 num(r["redemption_bn"], 2), num(float(cb.at[i, "cb_cash_bn"]), 2)] for i, r in lad.iterrows()]
    lead.append(("Calendar, EUR bn", P.table(cal, ["month", "OAT gross", "maturities", "buybacks",
                                                   "Eurosystem-held (est.)", "BTF gross", "BTF redemptions",
                                                   "BTF stock", "corporate tax"])
                 + P.table(lad_rows, ["maturity", "line", "outstanding (face)", "index ratio", "redemption (cash)",
                                      "Eurosystem-held (est.)"], left=2, titles=list(lad["isin"]))))

    figs = [P.fig_sources_uses(frame), P.fig_btf(frame, plan_btf_end, yr),
            P.fig_ytd(frame, yr, p_yr["deficit_to_finance"], model_fy, band), P.fig_oat(frame)]
    asof = (f"deficit to {last_cash:%b-%Y} · debt stock to {last_debt:%b-%Y} · OAT auctions to "
            f"{info['tape_last_op']:%d-%b-%Y} · BTF auctions to {info['btf_tape_last_op']:%d-%b-%Y}")
    P.build(DERIVED.parent / "fr_cash_dashboard.html", "France — State cash framework (BTF flex)",
            "SMB (INSEE) · AFT debt, outstanding by line, auctions · Banque de France account pinned since May-2023",
            asof, kpis, lead, figs, P.score_table(score, model), P.score_table(bscore, gap_mode), score_note,
            sources, notes, scenario_html=SX.table_html("fr", sx),
            frame=frame, summary=summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    s = run(refresh=ap.parse_args().refresh)
    fw = s["forward"]
    print({k: s[k] for k in ("asof", "identity_resid_bn")},
          {k: fw[k] for k in ("model", "cash_requirement_12m_bn", "net_bonds_12m_bn", "private_net_bonds_12m_bn",
                              "net_bills_12m_bn", "btf_stock_end_year_bn", "plan_btf_stock_end_year_bn",
                              "placements_12m_bn", "model_full_year_bn", "oat_reconciliation")})
