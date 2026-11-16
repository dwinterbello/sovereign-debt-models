"""The formulas, one per channel type and identity, listed in the order the Methodology page teaches them.

``intro`` is the one-to-two sentence lead-in (what is derived, what the next figure proves); ``says`` defines every
symbol in ``latex``. ``@fig:label@`` and ``@sec:id@`` are resolved to "Figure N" / "Section N" by page.render, so
numbers are never typed by hand. Code paths are repo-relative.

Symbol table (one meaning per letter across the page):
  B borrowing measure; L line; σ sign; F interest on extra debt; P perimeter; X driver level; ω share of a driver;
  g driver gap (%); s slider value; τ months in force; Π price level; Y nominal GDP; r interest rate (%);
  Δr yield shift (bp); ε elasticity; p period; m month; c, ℓ line; T last outturn.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Equation:
    id: str
    title: str
    latex: str
    says: str
    code: str
    intro: str = ""


_ORDER = ["identity", "gap", "perimeter", "sarimax", "level", "own", "elasticity", "matrix", "linear",
          "policy", "bridge", "book", "curve", "feedback"]

_EQ = [
    Equation(
        "identity", "What is being shocked",
        r"\Delta B_p=\sum_{\ell} \sigma_\ell\,\Delta L_{\ell,p}\;+\;F_p",
        "p is a forecast period (a calendar or fiscal year; the first may be part-year). B is the borrowing measure "
        "(cash requirement, PSNB, deficit) and ΔB_p its change from baseline in p. ℓ indexes the lines of the "
        "country's identity (receipts, primary spending, interest, and any other line it carries); ΔL_{ℓ,p} is the "
        "change in line ℓ in p from the channels below, and σ_ℓ its sign: +1 for spending and interest, −1 for "
        "receipts. F_p is interest on the extra debt, zero unless the feedback switch is on (derived in "
        "@sec:feedback@).",
        "sovereign_cash/scenarios/engine.py: evaluate",
        "Every number on the shock dashboard is a change in one borrowing measure, built from signed line changes. "
        "@fig:identity@ shows one preset adding up exactly to ΔB."),
    Equation(
        "gap", "How big the shock is, month by month",
        r"g_m=\begin{cases} s & \text{one-off level, } \tau_m>0\\ 100\left[(1+s/100)^{\tau_m/12}-1\right] & \text{every year, } \tau_m>0\\ 0 & \tau_m\le 0\end{cases}",
        "m indexes months and g_m is the percentage gap between the shocked driver's level and its baseline in month "
        "m. s is the slider value for that driver (in % for a level shift, in pp a year for a shock every year). "
        "τ_m is the number of months the shock has been in force by month m, after any lag; before onset (τ_m ≤ 0) "
        "the gap is zero. A one-off level shift holds the driver s% above baseline; a shock every year compounds.",
        "sovereign_cash/scenarios/engine.py: level_gap",
        "Each slider becomes a month-by-month gap g between the shocked and the baseline driver, and every channel "
        "below consumes g. @fig:gap@ plots g for a one-off level shift and for a shock every year."),
    Equation(
        "perimeter", "France, Germany, Italy: what the baseline is",
        r"B_p=\sum_{\ell}\sigma_\ell\,\hat L_{\ell,p}+P_p,\qquad \Delta P_p=0",
        "B_p is the borrowing measure in period p and σ_ℓ the line signs of @sec:identity@. L̂_{ℓ,p} is our forecast "
        "of line ℓ: receipts and spending from the component models below, interest from our ISIN debt book (the "
        "debt build's projection, in the line's usual month pattern). Italy's cash board uses this same path for the "
        "years after the DFP. No authority total is copied in; the authorities' plans are shown beside it for "
        "comparison only. P_p is the part of the published measure outside the modelled lines, forecast on its own: "
        "Germany's special funds (the Germany board's run-rate model) and, for Italy, the State sector borrowing "
        "requirement's flows outside the Bank of Italy cash-account lines (the same month's average over the last "
        "three full years). France's SMB is the measure, so France has no P. No shock acts on P (ΔP = 0). In outturn "
        "P is measured, so history and forecast are on one definition.",
        "sovereign_cash/scenarios/channels/sarimax.py: perimeter",
        "The baseline for France, Germany and Italy is our own line-by-line forecast plus a perimeter term P that no "
        "shock touches. The next equation specifies the lines; the resulting baseline (modelled lines plus P) is "
        "@fig:forecast@, after the share chart and the walk-forward test."),
    Equation(
        "sarimax", "France, Germany, Italy: component models",
        r"y_{c,t}=\omega_{c,t}\,X_{c,t-2};\quad \text{share: }(1-\phi L)(1-L^{12})\,\omega_{c,t}=(1+\Theta L^{12})\varepsilon_{c,t};\quad \text{hold: } \omega_{c,t}=\omega_{c,t-12}",
        "c indexes the driven lines and t months. y_{c,t} is the line's cash flow and X_{c,t} its driver's level, "
        "lagged two months: receipts on nominal GDP (seasonally adjusted, interpolated from quarters), primary "
        "spending on the HICP index, Italy's capital spending and other revenue on nominal GDP. ω_{c,t} is the line's "
        "share of that level. L is the lag operator (L ω_t = ω_{t−1}), φ the non-seasonal autoregressive coefficient, "
        "Θ the seasonal moving-average coefficient and ε_{c,t} the innovation. How the share is carried forward was "
        "chosen on a multi-year walk-forward test: 'share' lets a change in the share persist (France, and receipts "
        "everywhere); 'share, Covid out' does the same with March 2020 to December 2021 left out of the fit (German "
        "spending: since the 2025 reform the debt brake no longer pulls it back, and 2025 fits best this way); "
        "'hold' keeps last year's share month by month (Italy's capital spending and other revenue). Italy's other "
        "revenue is modelled without the EU RRF grants, which are added back only as still scheduled (the final "
        "instalment), so grants that have ended are not carried forward. Interest is not a share: it takes the "
        "ISIN book (@sec:book@) and, in the fit table, a coefficient β in bn per pp of the effective rate. Yield "
        "shocks go through the book.",
        "sovereign_cash/monthly_components.py: _fit_level, forecast",
        "Each driven line is a share of a macro driver, and the share is forecast by a seasonal time-series model. "
        "@fig:shares@ shows the shares carried forward; the specification choice, fitted coefficients and "
        "walk-forward tests follow it."),
    Equation(
        "level", "France, Germany, Italy: why a shock to the level is exact",
        r"\Delta y_{c,m}=\hat\omega_{c,m}\,\Delta X_{c,m-2}=\hat y_{c,m}\,\frac{g_{m-2}}{100}",
        "ŷ_{c,m} is the model's baseline forecast of line c in month m and ω̂_{c,m} the share forecast behind it, "
        "so ŷ = ω̂ X̂ with X̂ the baseline driver level. ΔX_{c,m−2} = X̂_{c,m−2} g_{m−2}/100 is the driver's change "
        "implied by the gap g of @sec:gap@, two months earlier (the lag). The share forecast does not depend on the "
        "driver, so moving the driver's level by g% moves the line by g% of its own forecast: a one-off level "
        "shift persists, growth compounds.",
        "sovereign_cash/scenarios/channels/sarimax.py",
        "Because the share forecast ignores the driver, a shock to the driver's level moves the line in proportion "
        "and needs no refit. @fig:level@ shows France's receipts: a one-off shift persists and growth compounds."),
    Equation(
        "own", "United Kingdom, United States, Spain, Sweden, Norway, Japan: annual component models",
        r"R_p=\omega^{R}_p\,Y_p,\quad Q_p=\omega^{Q}_p\,\Pi_p,\quad I_p=I_T\,\frac{\hat I^{\text{book}}_p}{\hat I^{\text{book}}_T};\qquad \omega_p=\begin{cases}\omega_T & \text{persist}\\ \bar\omega+\rho^{\,p-T}(\omega_T-\bar\omega) & \text{revert}\end{cases}",
        "R_p is receipts, Q_p primary spending and I_p interest in period p; T is the last outturn period. Receipts "
        "are a share ω^R of nominal GDP Y and primary spending a share ω^Q of the price level Π, as in the France, "
        "Germany and Italy models, at the frequency each country's long history exists: UK borrowing by fiscal year "
        "from ONS (monthly since 1997), the US unified deficit by fiscal year (OMB/Treasury), and general government "
        "for Spain, Sweden, Norway and Japan (AMECO, since 1995). Interest grows from its last outturn I_T by the "
        "growth of our ISIN debt book's projected interest, Î^book. Each share ω (written without superscript "
        "when the line does not matter) either persists at its last outturn ω_T or reverts to its mean ω̄ at "
        "persistence ρ, estimated per line and chosen on a walk-forward test one to three years ahead (table "
        "below): persist unless revert is at least 10% better. The OBR reckoner and CBO rules of thumb remain "
        "selectable as published comparisons.",
        "sovereign_cash/scenarios/channels/own_annual.py: forecast, backtest",
        "The other six countries are forecast by annual share models; persist or revert is chosen by walk-forward "
        "error. @fig:own@ scores the two rules for each line."),
    Equation(
        "elasticity", "Spain, Sweden, Norway, Japan: elasticities",
        r"\Delta L_{\ell,p}=\bar L_{\ell,p}\;\frac{1}{12}\sum_{m\in p}\left[\left(1+\frac{g_m}{100}\right)^{\varepsilon_\ell}-1\right]",
        "L̄_{ℓ,p} is line ℓ's baseline in period p, m runs over the 12 months of p, and g_m is the driver gap of "
        "@sec:gap@. The line responds with constant elasticity ε_ℓ to the driver's level, month by month. ε comes "
        "from the country's own model where one ships (Spain's category betas, Norway's oil pass-through) and is a "
        "stated assumption otherwise.",
        "sovereign_cash/scenarios/channels/annual.py",
        "Where a line responds to its driver with constant elasticity, the response is a power of the level, so "
        "large shocks stay exact. @fig:elasticity@ plots the curve and marks the elasticities in use."),
    Equation(
        "matrix", "United Kingdom: OBR ready reckoner",
        r"\Delta L_{\ell,p}=\sum_{d}\sum_{j\le p} C^{\ell,d}_{j,p}\;w_j\,x_{d,j},\qquad x_{d,j}=\begin{cases} g_{\text{end of }j} & \text{level determinants}\\ s & \text{rate determinants, every year}\\ s\cdot\mathbb 1[j=\text{first}] & \text{rate determinants, one-off}\end{cases}",
        "d indexes the determinants a driver moves together and j the year of the shock. x_{d,j} is the shock to "
        "determinant d in year j: the driver gap g at the end of j (@sec:gap@) for level determinants, or the slider "
        "value s for rate determinants, which are persistent. C^{ℓ,d}_{j,p} is OBR's coefficient: the effect on "
        "line ℓ in year p of a one-unit change in determinant d in year j. w_j pro-rates the first year to the "
        "months left. OBR's CGNCR financing loop is left out: interest on extra borrowing is the dashboard's own "
        "switch (@sec:feedback@). Interest rates are persistent determinants: shifted every year in either mode.",
        "sovereign_cash/scenarios/channels/obr.py; sovereign_cash/uk/ready_reckoner.py",
        "OBR's published ready reckoner is a selectable alternative to our own UK model. @fig:matrix@ shows two of "
        "its matrices as the dashboard reads them."),
    Equation(
        "linear", "United States, and published triangles: per-unit responses",
        r"\Delta L_{\ell,p}=s\,R^{\ell}_{p}",
        "s is the slider value and R^ℓ_p the published response of line ℓ in period p per unit of shock, moved to "
        "the dashboard's start date: CBO's rules of thumb (divided by their 0.1pp step, debt-service lines dropped, "
        "each year moved later by the gap between January 2026 and our start) and AFT's +100bp triangle (year k "
        "after the shock costs the triangle at k, spread over its months from our start).",
        "sovereign_cash/scenarios/channels/cbo.py; channels/sarimax.py (AFT)",
        "CBO's rules of thumb and AFT's triangle are selectable published responses, linear in the shock. "
        "@fig:linear@ shows the per-unit paths after moving them to our start date."),
    Equation(
        "policy", "Legislated policy on top of the model",
        r"B^{\text{pol}}_p=B_p+\sum_{m}\Bigl(c_{m,p}-c_{m,0}\,\frac{X_{h(m),p}}{X_{h(m),0}}\Bigr)",
        "p is a forecast period, 0 the last outturn year and m indexes legislated measures. B_p is our model's "
        "no-policy-change forecast (@sec:sarimax@, @sec:own@): each line's share of GDP or of prices carried forward "
        "from year 0. B^pol_p is the forecast with legislated policy. c_{m,p} is measure m's costed effect on "
        "borrowing (+ adds) in p, from the authority that costed it. A measure already in force in year 0 is inside "
        "the shares, so it counts only by how far it moves off them: its year-0 effect c_{m,0} grown as the model "
        "grows its line h(m), where X_{h,p} is the level that drives line h (nominal GDP for receipts, the price "
        "level for spending). Measures announced after year 0 count in full (c_{m,0} = 0). The line is fixed in "
        "money as costed: no shock moves it, and the costings are static (before any feedback through the "
        "economy).",
        "sovereign_cash/policy/overlay.py: build",
        "Costed legislated measures are layered on the no-policy-change forecast, netting what the shares already "
        "carry. @fig:policy@ stacks each country's measures; their net is the gap between the two forecast lines "
        "in @fig:forecast@."),
    Equation(
        "bridge", "Our forecast against the authority's: the bridge",
        r"B^{\text{ours}}_p-B^{\text{auth}}_p=\sum_{k}\sigma_k\bigl(\hat L_{k,p}-A_{k,p}\bigr),\qquad \sigma_k\bigl(\hat L_{k,p}-A_{k,p}\bigr)=\underbrace{\sigma_k\,d_k\,\frac{Y_p}{Y_0}}_{\text{definitions}}+\underbrace{f_{k,p}}_{\text{forecast}}",
        "k runs over the three lines receipts, primary spending and interest, with σ_k as in @sec:identity@. "
        "B^ours_p is our forecast with legislated policy (@sec:policy@), B^auth_p the budgetary authority's own "
        "total for the same period, L̂_{k,p} our line k and A_{k,p} the authority's, split on the authority's "
        "published lines. Lines are not always defined identically (our UK and US interest is gross, the "
        "authorities' net), so each line's difference is split into d_k, the difference already present in the last "
        "outturn year 0 (our outturn less the authority's), grown with nominal GDP Y_p/Y_0, and the forecast "
        "difference f_{k,p} left after it. The pieces add to the total gap exactly. France, Germany and Italy are "
        "compared on the total only: their plans have no line split we can match.",
        "sovereign_cash/scenarios/bridge.py: build",
        "The bridge splits every bn between our forecast and the authority's into a definitional part and a "
        "forecast part, line by line. @fig:bridge@ stacks the parts; the diamonds are the total gap."),
    Equation(
        "book", "Rates: the ISIN book",
        r"I_p(\Delta r)=\sum_{i\in\text{book}} c_i N_{i,p}+\sum_{k\in\text{new}} N_{k,p}\,f_{k}(\Delta r),\qquad \Delta I_p(s)=\Delta I_p(r_a)+\frac{s-r_a}{r_b-r_a}\left[\Delta I_p(r_b)-\Delta I_p(r_a)\right]",
        "I_p is interest in period p given a parallel shift Δr (bp) of the government zero curve. i indexes bonds "
        "already outstanding, with locked coupon c_i and notional N_{i,p} outstanding in p; k indexes new issues, "
        "with notional N_{k,p}, priced at the forward par yield f_k(Δr) off the zero curve shifted by Δr. The book "
        "is run at −200, −100, 0, +100, +200, +300bp. For a slider value s (bp) between two runs r_a ≤ s ≤ r_b, "
        "ΔI_p(s) is interpolated in a straight line from the changes ΔI_p(r_a) and ΔI_p(r_b) against the 0bp run. "
        "How a shift at one maturity spreads across the curve is the next section's equation.",
        "sovereign_cash/debt/build.py: project; sovereign_cash/scenarios/channels/debt_book.py",
        "A yield shock reprices every bond issued after it, so interest is computed from the bond-by-bond book "
        "at fixed shifts and read between them. @fig:book@ shows interest against the shift, one line per year."),
    Equation(
        "curve", "Bond yields: the 3-month and 10-year points",
        r"\delta(u)=\Sigma_{u,a}\,\Sigma_{a,a}^{-1}\begin{pmatrix}x_{3m}\\x_{10y}\end{pmatrix},\qquad a=\{3m,\,10y\}",
        "u is a maturity and δ(u) the change in its zero rate (bp). x_{3m} and x_{10y} are the moves in the two "
        "anchor points that the user sets. Σ is the covariance of monthly changes in zero rates since 2000, the "
        "matrix the principal components come from: Σ_{u,a} holds each maturity's covariances with the anchors "
        "and Σ_{a,a} the anchors' own 2×2 covariance. Every maturity moves by its expected change given the two "
        "anchors: the anchors are hit exactly and level, slope and curvature all sit at their conditional means. "
        "The ISIN book (@sec:book@) is run on these shapes at −200…+300bp in each anchor; the two anchors' "
        "effects add.",
        "sovereign_cash/scenarios/curve_pca.py: weights, shape",
        "A shock at one point has to move the rest of the curve, and the conditional expectation from history "
        "does it. @fig:curve@ shows the curve that follows a 3-month-only and a 10-year-only move."),
    Equation(
        "feedback", "Interest on the extra debt",
        r"F_t=\frac{k_t\,(S_{t-1}+\tfrac12 D_t)}{1-\tfrac12 k_t},\qquad S_t=S_{t-1}+D_t+F_t,\qquad k_t=\frac{r_t+\Delta r/100}{100}",
        "t is a forecast period (one year). D_t is the scenario's own change in borrowing in t (the ΔB of "
        "@sec:identity@ before F), S_t the cumulative extra debt at the end of t (S_0 = 0) and F_t the interest on "
        "it. r_t is the marginal new-issue rate in percent and Δr the yield shock in bp, so k_t is the rate as a "
        "fraction. Debt carried in, S_{t−1}, pays the full year at k_t; the year's new borrowing, including the "
        "feedback itself, pays half a year, which is why F_t appears on both sides and the denominator is "
        "1 − k_t/2.",
        "sovereign_cash/scenarios/engine.py: _feedback",
        "Extra borrowing is itself financed at the marginal new-issue rate, and that interest is borrowed again. "
        "@fig:feedback@ is the same shock with feedback off and on; the gap between the bars is F."),
]

EQUATIONS: dict[str, Equation] = {e.id: e for e in sorted(_EQ, key=lambda e: _ORDER.index(e.id))}
