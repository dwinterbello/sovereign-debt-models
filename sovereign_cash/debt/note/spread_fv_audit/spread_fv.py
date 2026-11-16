"""Fiscal fair value for euro-area 10y spreads to Bunds (nine countries against Germany). Self-contained audit version.

    pip install -r requirements.txt
    python spread_fv.py                      # reads inputs/, writes outputs/

Inputs (inputs/, all public or derived from public data):
  ecb_irs_m.csv     ECB SDW, IRS.M.<CC>.L.L40.CI.0000.EUR.N.Z: long-term (10y) interest rates for convergence
                    purposes, monthly, % p.a. Spread = country minus Germany, bp; annual = calendar-year mean
                    (2026 = January to latest month).
  eurostat_gg.csv   Eurostat (21 Oct 2026 release), general government: gross debt (gov_10dd_edpt1 GD), net lending
                    (B9), interest payable (D41PAY), all % of GDP; total revenue % of GDP (gov_10a_main TR); nominal
                    GDP, EUR mn (nama_10_gdp B1GQ). 1995-2025, AT BE DE ES FI FR IE IT NL PT. Greece is excluded
                    (2012 debt restructuring: default, not fiscal, pricing).
  ameco_gg.csv      European Commission AMECO (autumn 2026): the same variables for DE FR IT ES, used only for 2026
                    (observation) and 2026-27 (fair value), as Eurostat ends in 2025.
  note_scenarios.json  The note's scenario grid: change in interest / revenue and primary deficit by 2030.
  note_paths.json   From the note's own model: general-government interest / revenue in 2026 and 2030 and today's
                    primary balance for France, Italy and Germany, and the latest OAT-Bund level.

Method:
  Pooled OLS on annual data, 1999-2026, nine countries against Germany, no country effects (levels
  are to be explained by fundamentals). Regimes 1999-2007, 2008-09, 2010-12, 2013-21, 2022-26 get their own
  intercept and their own slope on the main fiscal variable. Preferred specification:
      spread_it = a_regime + b_regime * (interest/revenue_i - interest/revenue_DE) + c * (primary_i - primary_DE) + e
  Fair value = fitted spread under 2022-26 pricing; band = 80% prediction interval. HAC (Newey-West, 2 lags) standard
  errors for the coefficients. Robustness: debt / GDP instead of interest / revenue, own primary balance instead of
  the gap to Germany, the debt-stabilising fiscal gap, a debt spline above 90% of GDP, lagged fundamentals (reverse
  causality), post-2013 sample with year effects, leave-one-country-out.

Outputs (outputs/): results.json, regression_tables.txt, fair_value.png.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
IN, OUT = HERE / "inputs", HERE / "outputs"
CTRY = {c.lower(): c for c in ("FR", "IT", "ES", "BE", "NL", "AT", "PT", "IE", "FI")}
NAME = {"fr": "France", "it": "Italy", "es": "Spain", "be": "Belgium", "nl": "Netherlands", "at": "Austria",
        "pt": "Portugal", "ie": "Ireland", "fi": "Finland"}
CORE = ("fr", "it", "es")
TPI = "R4 2022-26"
SPECS = {
    "preferred: int/rev gap + primary gap": "spread ~ C(regime) + C(regime):d_ir + d_pb - 1",
    "debt/GDP gap + primary gap": "spread ~ C(regime) + C(regime):d_debt + d_pb - 1",
    "int/rev gap + own primary": "spread ~ C(regime) + C(regime):d_ir + pb - 1",
    "int/rev gap + own fiscal gap": "spread ~ C(regime) + C(regime):d_ir + gap - 1",
    "int/rev gap + fiscal gap by regime": "spread ~ C(regime) + C(regime):d_ir + C(regime):gap - 1",
    "int/rev gap + own primary + debt>90 spline": "spread ~ C(regime) + C(regime):d_ir + pb + C(regime):spl - 1",
    "lagged: int/rev gap(t-1) + primary gap(t-1)": "spread ~ C(regime) + C(regime):d_ir_l1 + d_pb_l1 - 1",
}


def regime(y: int) -> str:
    return ("R0 1999-07" if y <= 2007 else "R1 2008-09" if y <= 2009 else "R2 2010-12" if y <= 2012
            else "R3 2013-21" if y <= 2021 else TPI)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.read_csv(IN / "ecb_irs_m.csv", usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    y = y.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
    y.index = pd.PeriodIndex(y.index, freq="M")
    S = y[list(CTRY.values())].sub(y["DE"], axis=0) * 100
    e = pd.read_csv(IN / "eurostat_gg.csv")
    e = e.assign(country=e.geo.str.lower()).rename(columns={"GD": "debt_pct", "B9": "balance_pct",
                                                            "D41PAY": "interest_gross_pct", "TR": "revenue_pct",
                                                            "GDP": "gdp_lcu_bn"})
    e = e[e.year <= 2025].set_index(["country", "year"])[["debt_pct", "balance_pct", "interest_gross_pct",
                                                          "revenue_pct", "gdp_lcu_bn"]]
    am = pd.read_csv(IN / "ameco_gg.csv").pivot_table(index=["country", "year"], columns="var", values="value")
    am = am[am.index.get_level_values("year") >= 2025]
    # 2026-27 from AMECO, growth chained onto Eurostat's 2025 GDP level (AMECO GDP is in national currency bn)
    g = am["gdp_lcu_bn"].groupby(level=0).pct_change()
    am = am[am.index.get_level_values("year") >= 2026].copy()
    for c in am.index.get_level_values(0).unique():
        lvl = float(e.loc[(c, 2025), "gdp_lcu_bn"])
        for y in (2026, 2027):
            if (c, y) in am.index:
                lvl *= 1 + float(g.loc[(c, y)])
                am.loc[(c, y), "gdp_lcu_bn"] = lvl
    a = pd.concat([e, am[e.columns]]).sort_index()
    a["pb"] = a["balance_pct"] + a["interest_gross_pct"]
    a["ir"] = 100 * a["interest_gross_pct"] / a["revenue_pct"]
    Y = a["gdp_lcu_bn"]
    g = Y.groupby(level=0).pct_change()
    d1 = a["debt_pct"].groupby(level=0).shift(1)
    r = a["interest_gross_pct"] * Y / (d1 * Y.groupby(level=0).shift(1))
    a["gap"] = (r - g) / (1 + g) * d1 - a["pb"]          # debt-stabilising primary minus actual, % GDP
    a["spl"] = (a["debt_pct"] - 90).clip(lower=0)
    rows = []
    for c, code in CTRY.items():
        d, de = a.loc[c].copy(), a.loc["de"]
        for v, k in (("debt_pct", "d_debt"), ("pb", "d_pb"), ("ir", "d_ir")):
            d[k] = d[v] - de[v]
        d["spread"] = S[code].groupby(S.index.year).mean().reindex(d.index)
        d["cc"], d["year"] = c, d.index
        rows.append(d)
    P = pd.concat(rows).reset_index(drop=True).sort_values(["cc", "year"])
    P["d_ir_l1"] = P.groupby("cc")["d_ir"].shift(1)
    P["d_pb_l1"] = P.groupby("cc")["d_pb"].shift(1)
    P["regime"] = P["year"].map(regime)
    return P, S


def predict(r, row: dict, reg: str = TPI) -> tuple[float, float, float]:
    x = dict(row, regime=reg, spread=0.0)
    f = r.get_prediction(pd.DataFrame([x])).summary_frame(alpha=0.2)
    return tuple(round(float(f[k].iloc[0]), 1) for k in ("mean", "obs_ci_lower", "obs_ci_upper"))



def figure(res: dict, A: pd.DataFrame, S: pd.DataFrame, path: Path) -> None:
    """Exhibit: A. France and Italy against fair value; B. France: equilibrium, premium, scenarios."""
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.28})
    for c, col in (("fr", "#1f4e79"), ("it", "#0f766e")):
        d = A[(A.cc == c) & (A.year >= 2005)]
        a1.plot(d.year, d.spread, color=col, lw=2.2, label=f"{NAME[c]}: actual (annual average)")
        a1.plot(d.year, d.fit, color=col, lw=1.4, ls="--", label=f"{NAME[c]}: fiscal fair value")
        last = float(S[CTRY[c]].iloc[-1])
        a1.scatter([2026.83], [last], color=col, s=55, zorder=5, edgecolor="white")
        a1.annotate(f"{last:.0f}", (2026.83, last), xytext=(6, 0), textcoords="offset points", color=col,
                    fontweight="bold", va="center", fontsize=9)
    a1.set_xlim(2005, 2028)
    a1.set_ylabel("10y spread to Bunds, bp")
    a1.legend(frameon=False, fontsize=8.2, loc="upper right")
    a1.set_title("A. Spreads against fiscal fair value")
    a1.text(0.01, 0.97, f"Dots: {S.index[-1]} monthly average", transform=a1.transAxes, fontsize=8, color="#555",
            va="top")
    p, d = res["preferred"], res["oat_bund_decomposition_bp"]
    fr26 = res["specs"]["preferred: int/rev gap + primary gap"]["fair_tpi_pricing"]["fr_2026"]
    rows = [("Equilibrium 2026 (post-2022 pricing)", fr26, "#1f4e79"),
            ("Equilibrium 2030, today's primaries", p["fr_2030_tpi_pricing"], "#1f4e79"),
            ("2010-12 crisis pricing, 2027 fiscal", p["fr_2027_crisis_pricing_2010_12"], "#b91c1c")]
    y = 4.6
    for lab, (m, lo, hi), cc in rows:
        a2.plot([max(lo, 0), hi], [y, y], color=cc, lw=7, alpha=0.25, solid_capstyle="butt")
        a2.scatter([m], [y], color=cc, s=60, zorder=4)
        a2.text(m, y + 0.3, f"{lab}: {m:.0f}bp ({max(lo, 0):.0f}-{hi:.0f})", ha="center", fontsize=8.4, color=cc)
        y -= 1.1
    for (lab, (lo, hi)), yy in zip({"Scenario: legislated consolidation (premium halved)": (110, 140),
                                    "Scenario: budget fails (2010-12 regime pricing)": (210, 300)}.items(), (1.2, 0.2)):
        a2.barh(yy, hi - lo, left=lo, height=0.42, color="#9ca3af")
        a2.text((lo + hi) / 2, yy + 0.32, f"{lab} ({lo}-{hi})", va="bottom", ha="center", fontsize=8.4)
    now = d["now"]
    a2.axvline(now, color="#111827", lw=1.2, ls=":")
    pr = res["market_implied_crisis_probability"]["range"]
    a2.text(now + 3, 5.35, f"OAT-Bund {now:.0f}bp ({d['asof']}): premium ~{d['france_specific_premium'][0]}-"
            f"{d['france_specific_premium'][1]}bp; implied probability of crisis-regime pricing ~{pr[0]:.0%}-{pr[1]:.0%}", fontsize=8.2)
    a2.set_yticks([])
    a2.set_xlim(0, 400)
    a2.set_ylim(-0.4, 5.7)
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("OAT-Bund 10y, bp")
    a2.set_title("B. France: equilibrium, premium and scenarios")
    pr_ = res["preferred"]
    fig.text(0.01, -0.07,
             f"Pooled OLS, nine euro-area countries against Germany (AT BE ES FI FR IE IT NL PT), annual 1999-2026 "
             f"(n={res['inputs']['n_obs']}). Spread on the gap in general-government interest / revenue and in the primary "
             f"balance; pricing differs by regime (1999-2007, 2008-09, 2010-12, 2013-21, 2022-26). Since 2022 each pp of "
             f"interest / revenue gap adds {pr_['slope_int_rev_bp_per_pp_2022_26']:.0f}bp (country-clustered se "
             f"{res['country_clustered_se_slope_2022_26']:.1f}). Bands: 80% prediction intervals. Premium = OAT-Bund minus "
             f"equilibrium minus the euro-area-wide move (~{res['oat_bund_decomposition_bp']['euro_area_common_move_mean_7_since_jan_aug']:.0f}bp, the average widening since the summer in the seven other "
             "panel countries ex France and Italy). Sources: ECB long-term "
             "convergence yields; Eurostat; EC AMECO autumn 2026 for 2026-27.", fontsize=7.0, color="#555", wrap=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    P, S = load()
    paths = json.loads((IN / "note_paths.json").read_text())
    A = P[(P.year >= 1999) & (P.year <= 2026)].dropna(subset=["spread"])
    res, tables = {"specs": {}, "inputs": {"last_month": str(S.index[-1]), "n_obs": int(len(A))}}, []
    fits = {}
    for name, f in SPECS.items():
        d = A.dropna(subset=["d_ir_l1"]) if "l1" in f else A.dropna(subset=["gap"]) if "gap" in f else A
        ols = smf.ols(f, d).fit()
        hac = smf.ols(f, d).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        fits[name] = ols
        tables.append(f"===== {name}   n={int(ols.nobs)}  R2={ols.rsquared:.3f}  AIC={ols.aic:.1f}\n"
                      + hac.summary2().tables[1].round(3).to_string() + "\n")
        fv = {}
        for c in CORE:
            for yr in (2026, 2027):
                row = P[(P.cc == c) & (P.year == yr)].iloc[0].to_dict()
                try:
                    fv[f"{c}_{yr}"] = predict(ols, row)
                except Exception:  # noqa: BLE001  (lagged spec has no 2027 lag issue; guard anyway)
                    pass
        res["specs"][name] = {"r2": round(ols.rsquared, 3), "aic": round(ols.aic, 1), "n": int(ols.nobs),
                              "fair_tpi_pricing": fv}
    pref = fits["preferred: int/rev gap + primary gap"]
    res["preferred"] = {
        "slope_int_rev_bp_per_pp_2022_26": round(float(pref.params[f"C(regime)[{TPI}]:d_ir"]), 2),
        "slope_primary_bp_per_pp": round(float(pref.params["d_pb"]), 2)}
    # crisis pricing, and 2030 on the note's interest paths: today's primaries and the IMF plan primaries
    for c in ("fr", "it"):
        row = P[(P.cc == c) & (P.year == 2027)].iloc[0].to_dict()
        res["preferred"][f"{c}_2027_crisis_pricing_2010_12"] = predict(pref, row, "R2 2010-12")
        d_ir30 = paths["ir_2030"][c] - paths["ir_2030"]["de"]
        res["preferred"][f"{c}_2030_tpi_pricing"] = predict(pref, {"d_ir": d_ir30,
                                                                    "d_pb": paths["pb_now"][c] - paths["pb_now"]["de"]})
        res["preferred"][f"{c}_2030_tpi_pricing_plan_primary"] = predict(
            pref, {"d_ir": d_ir30, "d_pb": paths["pb_plan_2030"][c] - paths["pb_plan_2030"]["de"]})
    # the note's scenarios through the preferred model: change in fiscal equilibrium, bp
    b_ir = float(pref.params[f"C(regime)[{TPI}]:d_ir"]); b_pb = float(pref.params["d_pb"])
    sc = json.loads((IN / "note_scenarios.json").read_text())
    res["scenario_equilibrium_change_bp"] = {
        k: {c: round(b_ir * (v["d_ir"][c] - v["d_ir"]["de"]) - b_pb * (v["d_pd"][c] - v["d_pd"]["de"]), 1)
            for c in CORE} for k, v in sc.items() if not k.startswith("_")}
    # full cross-section in the latest Eurostat year: actual against fair value
    last = P[(P.year == 2025)].dropna(subset=["d_ir", "spread"])
    res["cross_section_2025"] = {r.cc: {"actual": round(float(r.spread), 1), "fair": predict(pref, r._asdict())[0]}
                                 for r in last.itertuples(index=False)}
    # post-2013 with year effects; leave one country out
    B = A[A.year >= 2013]
    yfe = smf.ols("spread ~ d_ir + d_pb + C(year)", B).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    res["post2013_year_effects"] = {"d_ir": round(float(yfe.params["d_ir"]), 2), "se": round(float(yfe.bse["d_ir"]), 2),
                                    "r2": round(float(yfe.rsquared), 3)}
    res["leave_one_out_slope_2022_26"] = {
        f"drop_{c}": round(float(smf.ols(SPECS["preferred: int/rev gap + primary gap"], A[A.cc != c]).fit()
                                 .params[f"C(regime)[{TPI}]:d_ir"]), 2) for c in CTRY}
    clu = smf.ols(SPECS["preferred: int/rev gap + primary gap"], A).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(A["cc"])[0]})
    res["country_clustered_se_slope_2022_26"] = round(float(clu.bse[f"C(regime)[{TPI}]:d_ir"]), 2)
    resid = A.assign(e=pref.resid)
    res["residual_ar1_by_country"] = {c: round(float(g.e.autocorr()), 2) for c, g in resid.groupby("cc")}
    # decomposition of today's OAT-Bund (back-of-envelope, labelled as such)
    m = S.loc["2026-01":]
    # euro-area common move: the average move since Jan-Aug in the seven panel countries other than France and Italy
    # (Spain alone, ~20bp, is the high end: it trades inside its fair value and is itself exposed to contagion)
    peers = [c for c in ("AT", "BE", "ES", "FI", "IE", "NL", "PT") if c in m.columns]
    moves = {c: float(m[c].iloc[-1] - m[c].loc["2026-01":"2026-08"].mean()) for c in peers}
    common = float(np.mean(list(moves.values())))
    fr_fv = [v["fair_tpi_pricing"]["fr_2026"][0] for k, v in res["specs"].items()
             if not k.startswith("debt") and "fr_2026" in v["fair_tpi_pricing"]]
    now = paths["oat_bund_now_bp"]
    res["oat_bund_decomposition_bp"] = {
        "now": now, "asof": paths["oat_bund_asof"],
        "fiscal_equilibrium_2026_range_across_int_rev_specs": [round(min(fr_fv)), round(max(fr_fv))],
        "fiscal_equilibrium_2030_preferred": res["preferred"]["fr_2030_tpi_pricing"][0],
        "euro_area_common_move_mean_7_since_jan_aug": round(common, 1),
        "euro_area_moves_since_jan_aug": {c.lower(): round(v, 1) for c, v in moves.items()},
        "euro_area_common_move_median_7": round(float(np.median(list(moves.values()))), 1),
        "france_specific_premium": [round(now - max(max(fr_fv), res["preferred"]["fr_2030_tpi_pricing"][0]) - common),
                                    round(now - min(fr_fv) - common)],
        "note": "premium = now - fiscal equilibrium (highest: 2030 path; lowest: 2026 across int/rev specs) - "
                "euro-area common move (mean move since Jan-Aug of AT BE ES FI IE NL PT); a back-of-envelope split, "
                "not a regression output"}
    # crisis pricing is identified mostly on the 2010-12 periphery (IE IT ES PT): re-estimate on the core five alone
    core = A[A.cc.isin(["at", "be", "fi", "fr", "nl"])]
    pref_core = smf.ols(SPECS["preferred: int/rev gap + primary gap"], core).fit()
    row27 = P[(P.cc == "fr") & (P.year == 2027)].iloc[0].to_dict()
    res["preferred"]["fr_2027_crisis_pricing_core_sample"] = predict(pref_core, row27, "R2 2010-12")
    # market-implied probability of crisis-regime pricing: now - common = p * crisis + (1 - p) * equilibrium
    crisis = res["preferred"]["fr_2027_crisis_pricing_2010_12"][0]
    eqs = (min(fr_fv), res["preferred"]["fr_2030_tpi_pricing"][0])
    res["market_implied_crisis_probability"] = {
        "range": [round((now - common - e) / (crisis - e), 2) for e in sorted(eqs, reverse=True)],
        "crisis_pricing_bp": crisis, "equilibria_bp": [round(e) for e in eqs],
        "range_core_sample_crisis": [round((now - common - e) / (res["preferred"]["fr_2027_crisis_pricing_core_sample"][0]
                                            - e), 2) for e in sorted(eqs, reverse=True)],
        "note": "illustrative two-state mixture, after removing the euro-area common move"}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    (OUT / "regression_tables.txt").write_text("\n".join(tables))
    figure(res, A.assign(fit=pref.fittedvalues), S, OUT / "fair_value.png")
    print(json.dumps({k: res[k] for k in ("inputs", "preferred", "post2013_year_effects", "leave_one_out_slope_2022_26",
                                          "country_clustered_se_slope_2022_26", "oat_bund_decomposition_bp",
                                          "market_implied_crisis_probability", "scenario_equilibrium_change_bp",
                                          "cross_section_2025")}, indent=1))
    for k, v in res["specs"].items():
        print(f"{k:45s} R2={v['r2']:.2f}  FR26 {v['fair_tpi_pricing'].get('fr_2026')}  "
              f"IT26 {v['fair_tpi_pricing'].get('it_2026')}  ES26 {v['fair_tpi_pricing'].get('es_2026')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
