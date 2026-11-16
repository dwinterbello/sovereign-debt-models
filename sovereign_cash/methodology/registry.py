"""Methodology formulas: every component of the sovereign financing chain, Equations-page style.

Shock-channel formulas (gap, SARIMAX response, reckoner matrices, feedback) stay on
``sovereign_cash.equations.registry``. This registry is the production forecast chain the
debt board, primary exhibits and Cash / US / Supply tabs read — not the slider arithmetic.
"""
from __future__ import annotations

from sovereign_cash.equations.registry import Equation

SECTIONS: list[tuple[str, str, list[str]]] = [
    ('chain', 'One chain, four horizons', ['chain']),
    ('macro', 'Macro drivers: nominal GDP, prices, earnings', ['macro', 'fiscalyear']),
    ('shares', 'Receipts and primary spending as shares, T+1 to T+3', ['shares', 'carry', 'gate']),
    ('drag', 'UK receipts: frozen thresholds', ['drag']),
    ('wedge', 'Primary deficit, legislated policy and the general-government wedge', ['deficit', 'ggwedge']),
    ('hold', 'After T+3: GG primary from WEO', ['hold']),
    ('interest', 'Interest from the ISIN book', ['interest', 'bookrate']),
    ('financing', 'Net issuance and the financing wedge', ['financing']),
    ('debtpath', 'Debt-to-GDP arithmetic', ['debtpath', 'repricing', 'pbstar']),
    ('scenlines', 'Scenarios: one line model for growth, inflation and rates', ['scenlines', 'cycreal', 'cycus', 'cycuk', 'cycprice', 'cycrates']),
    ('cashid', 'Near-term cash: identity, scorer, selection', ['cashid', 'cashmodels', 'cashscore', 'shiprule']),
    ('cashuk', 'United Kingdom: CGNCR(ex)', ['cashuk', 'cashukbu']),
    ('cashfr', 'France: central-government cash requirement', ['cashfr']),
    ('cashde', 'Germany: Federal core budget and special funds', ['cashde', 'cashdelater']),
    ('cashit', 'Italy: fabbisogno del settore statale', ['cashit', 'cashitbu']),
    ('cashthin', 'Spain, Sweden, Norway, Japan', ['cashes', 'cashse', 'cashno', 'cashjp']),
    ('usflow', 'United States: daily operating flows', ['usflow', 'uslevel', 'usinterest', 'usdtsadj']),
    ('ustga', 'United States: Treasury cash, financing and reserves', ['ustga', 'ussteer', 'usdrift', 'usreserves']),
    ('usbridge', 'United States: monthly supply bridge', ['usbridge', 'usbills', 'usaccrete']),
    ('issuance', 'Net issuance and private DV01', ['dv01', 'privatedv01', 'usbilldv01']),
    ('absorb', 'Absorption and new private money', ['absorb', 'newmoney']),
]

_EQ = [
    Equation(
        'chain', 'What is forecast at each horizon',
        r"""\hat D_p=\begin{cases}\sum_{m\in p}\hat C_m & \text{months }1\ldots 12\text{ (cash boards, by line)}\\[2pt]\hat q_p X_p+\mathrm{Pol}^{Q}_{p}-\hat r_p Y_p-\mathrm{Pol}^{R}_{p} & T+1\le p\le T+3\\[2pt]\pi^{\mathrm{WEO}}_{p}\,Y_p & p>T+3\end{cases}""",
        'p is a period (calendar year, or fiscal year for the UK and US); m a month; T the last complete outturn period. D̂_p is the forecast primary deficit (+ = deficit). Ĉ_m is the monthly cash requirement from the country cash model (@sec:cashid@ onward), built line by line where the country publishes lines. r̂_p and q̂_p are the receipts and primary-spending shares carried forward from T (@sec:shares@), Y_p nominal GDP and X_p the price level (@sec:macro@). Pol^{Q}_{p} and Pol^{R}_{p} are legislated spending and tax measures, static authority costings net of the base year (one section covers them, @sec:deficit@) — applied only inside the tested window. Beyond T+3 the GG primary is the IMF WEO primary deficit π^{WEO}_p (@sec:hold@), except the US, whose line model (@sec:usflow@) runs to FY2030; policy overlays are dropped there because WEO already embeds them. Interest, debt and issuance follow from D̂_p through the ISIN book (@sec:interest@ to @sec:debtpath@) — WEO interest is never an input.',
        'sovereign_cash/scenarios/primary.py: path; sovereign_cash/debt/note/build.py: _chain_table',
        'The debt board, the cash boards and the shock dashboard share one chain. Each horizon has one model: cash/share/DTS where the walk-forward test runs (h≤3), WEO primary beyond (the US line model to FY2030). Scenarios run one line model for growth, inflation and rates (@sec:scenlines@). Interest stays on the ISIN book. The table below lists the handoffs; the rest of this page derives each row.'),
    Equation(
        'macro', 'Nominal GDP and the price level',
        r"""Y_p=\begin{cases}Y^{AMECO}_p & p\le T_A\\ Y_{p-1}\bigl(1+\tfrac{\hat g^{Y}_p}{100}\bigr) & p>T_A\end{cases},\qquad \hat g^{Y}_p=\begin{cases}g^{RESF}_p+\pi^{RESF}_p & \text{FR, RESF years}\\ g^{DFP}_p & \text{IT, DFP years}\\ g^{AMECO}_p & \text{AMECO forecast years}\\ g^{real,WEO}_p+\pi^{WEO}_p & \text{beyond}\end{cases},\qquad X_p=X_{p-1}\bigl(1+\tfrac{\pi_p}{100}\bigr),\ X_{1995}=1,\quad \pi_p=\begin{cases}\pi^{RESF}_p & \text{FR, RESF years}\\ \pi^{\mathrm{BU}}_p & \text{FULL\_BU, all countries}\\ \pi^{WEO}_p & \text{beyond 2030 / fail}\end{cases}""",
        "Y_p is nominal GDP in local currency, X_p the price index. Through the year before the house CPI window the GDP level is AMECO's; after it the level compounds at ĝ^{Y}_p: RESF real+CPI (FR plan years), DFP nominal (IT plan years), else IMF WEO real × our house CPI path through 2030. Forecast NGDP = WEO real × that CPI except RESF/DFP pins. Realisation fail (≥3 post-forecast prints, house MAE > WEO-flat + 0.15pp) drops CPI to WEO. WEO CPI beyond 2030. UK earnings E_p (fiscal drag) are ONS AWE then OBR.",
        'sovereign_cash/macro_authority.py: annual_authority, cpi_override_gate; sovereign_cash/scenarios/channels/own_annual.py: macro_cy',
        "House CPI is the price driver through 2030; WEO supplies real growth and the beyond-2030 / fail fallback."),
    Equation(
        'fiscalyear', 'Fiscal-year labelling',
        r"""Y^{FY}_{y}=\tfrac34\,Y_{y}+\tfrac14\,Y_{y+1}\ \text{(UK, Apr }y\text{–Mar }y{+}1),\qquad Y^{FY}_{y}=\tfrac14\,Y_{y-1}+\tfrac34\,Y_{y}\ \text{(US, Oct }y{-}1\text{–Sep }y)""",
        'Y^{FY}_{y} is the fiscal-year value of a calendar series Y. The UK fiscal year April–March is labelled by the calendar year carrying nine of its twelve months and takes 3/4 of that year and 1/4 of the next; the US fiscal year October–September is labelled by the year carrying nine months and takes 1/4 of the prior year. The same weights convert the price level and earnings; every other country is calendar-year.',
        'sovereign_cash/scenarios/channels/own_annual.py: weights, to_period, period_months',
        'The UK and US publish fiscal-year borrowing. Their macro drivers are calendar series, so each is moved to the fiscal year by month count before any share is formed.'),
    Equation(
        'shares', 'Receipts to GDP, spending to prices',
        r"""r_p=\frac{R_p}{Y_p},\qquad q_p=\frac{Q_p}{X_p}\ \ (\text{or }\frac{Q_p}{Y_p}\text{ under persist-GDP}),\qquad \hat R_{T+h}=\hat r_{T+h}\,Y_{T+h},\quad \hat Q_{T+h}=\hat q_{T+h}\,X_{T+h}""",
        "R_p is receipts and Q_p primary spending in the country's borrowing measure (federal deficit for the US; PSNB ex banks for the UK; general-government deficit for ES/SE/NO/JP; for FR/DE/IT the same measure as their cash boards). r_p is the receipts share of nominal GDP and q_p primary spending relative to the price level, so spending is real in the no-policy case; Sweden's primary is the one line carried as a share of GDP (persist-GDP), where the test chose it. R̂, Q̂ for h = 1…3 are the carried shares times the macro path of @sec:macro@. Origins since 2005; Covid years 2020–21 are excluded from fitting and scoring. The walk-forward test is conditional on realised GDP and prices at each origin (macro path known; share rule is what is scored).",
        'sovereign_cash/scenarios/channels/own_annual.py: history, levels, forecast, backtest',
        'Beyond the cash window the engine does not forecast cash lines; it forecasts two shares and multiplies by the macro path. A share is a stationary object a walk-forward test can grade, a cash level is not.'),
    Equation(
        'carry', 'Persist or revert',
        r"""\text{persist: } \hat r_{T+h}=r_T;\qquad \text{revert: } \hat r_{T+h}=\mu+\rho^{h}\,(r_T-\mu),\quad \mu=\overline{r}_{t\notin\text{Covid}},\ \ \rho=\operatorname{clip}_{[0,1]}\frac{\sum_t (r_{t-1}-\mu)(r_t-\mu)}{\sum_t (r_{t-1}-\mu)^2}""",
        'r_T is the share at the last outturn, h the horizon in periods. Under persist the share is held. Under revert it decays to the sample mean μ (years 2020–21 left out) at annual persistence ρ, the no-intercept AR(1) slope of the demeaned share, clipped to [0, 1] and set to 1 when fewer than five year pairs exist. The same two rules apply to q with the price-level share.',
        'sovereign_cash/scenarios/channels/own_annual.py: carry, _rho_mean',
        'Two carry rules only. Anything richer (trend, cyclical adjustment, authority path) failed the walk-forward test or cannot be tested on twenty annual origins.'),
    Equation(
        'gate', 'Which rule ships, per country and line',
        r"""\text{revert if }\ \overline{\mathrm{MAE}}^{\,h=1..3}_{\text{revert}}<(1-0.10)\,\overline{\mathrm{MAE}}^{\,h=1..3}_{\text{persist}},\ \text{else persist},\qquad \mathrm{MAE}=\frac1n\sum_{o}\bigl|\hat L_{o+h}-L_{o+h}\bigr|""",
        'The walk-forward test refits from every origin year o from 2005 to T−1 and scores the line level L (receipts or primary spending, local bn) one to three years out given realised GDP and prices at o; target years in 2020–21 are dropped. Revert replaces persist only if its mean MAE over h = 1…3 is more than REVERT_GAIN = 10% better. The headline share MAE on the Methodology page is therefore conditional on realised macro; the companion score under point-in-time WEO vintages is ``derived/own_annual_vintage_score.json`` (UK/US/ES/SE/NO/JP — the own-annual set; FR/DE/IT use component models, not this file).',
        'sovereign_cash/scenarios/channels/own_annual.py: backtest, method, REVERT_GAIN; sovereign_cash/scenarios/channels/own_annual_vintage.py → derived/own_annual_vintage_score.json; derived/own_annual_backtest.json',
        "The choice is made on the level error of the line, not on the headline deficit: a share rule that wins the headline by cancelling another line's error is not evidence for the rule."),
    Equation(
        'drag', 'Income tax and NICs under frozen thresholds',
        r"""\hat R_{T+h}=(1-s_T)\,R_T\,\frac{Y_{T+h}}{Y_T}\;+\;s_T\,R_T\prod_{t=T+1}^{T+h}\phi_t,\qquad \phi_t=\begin{cases}\bigl(E_t/E_{t-1}\bigr)^{\varepsilon} & 2021\le t\le 2027\ (\varepsilon=1.3)\\ Y_t/Y_{t-1} & \text{otherwise}\end{cases}""",
        's_T is the income-tax-plus-NICs share of receipts at the origin (OBR authority file; 0.435 fallback), E_t average earnings on the fiscal-year basis, Y_t nominal GDP. Through FY2027-28 — the last year the Autumn Statement 2022 freeze would have run under prior law — IT+NICs compound with earnings at elasticity ε = 1.3 (OBR EFO October 2026 ¶4.14: +1pp earnings ≈ £7bn under frozen thresholds against ≈ £4.5bn if CPI-indexed, ratio 4.5/7 = 0.84 for the indexed counterfactual). After that the elevated level is unit-elastic to GDP, so the Budget 2025 extension enters only through the policy line. The rest of receipts is unit-elastic to GDP.',
        'sovereign_cash/scenarios/channels/own_annual.py: receipts_fiscal_drag, itnic_eps, itnic_share, EPS_FROZEN',
        'A frozen personal allowance makes income tax grow faster than earnings. Persist on a share of GDP misses that by construction, so the UK receipts line is the one place the chain carries a legislated elasticity instead of a carried share. Backtest: freeze-era bias −0.97% of GDP under persist, −0.76% under this rule; on freeze-era and full-sample MAE it loses to persist (34.0 vs 31.5 £bn; 0.713 vs 0.688% of GDP) and ships as a structural prior, not on the gate.'),
    Equation(
        'deficit', 'Primary deficit with legislated policy',
        r"""D_p=\hat Q_p+\mathrm{Pol}^{Q}_{p}-\hat R_p-\mathrm{Pol}^{R}_{p}""",
        "D_p is the primary deficit in the country's measure (+ = deficit). Pol^{Q}_{p} and Pol^{R}_{p} are the authority's static costings of measures legislated by the origin, summed by line and expressed net of the base year, so a measure already in the T outturn is not counted twice; a positive tax measure raises receipts. In the US DTS years the tax measures of the 2025 reconciliation act are not added (US_DTS_TAX_OVERLAY = False): the Treasury cash levels already carry them, and CBO's FY26→27 tax increment is mostly the timing of its TCJA-expiry baseline; the spending measures (Medicaid and SNAP cuts phasing in after the base year) stay. The policy overlay — sources, signs, per-country event counts and its cross-checks against the authority — is one section, @sec:deficit@, shared with the shock dashboard.",
        'sovereign_cash/scenarios/primary.py: path, _policy; sovereign_cash/policy/overlay.py',
        'Shares give the no-policy path; policy is added on top as the authorities costed it. Keeping the two apart is what lets the authority bridge of the shock-dashboard bridge (Equations) say how much of a gap to the OBR or CBO is macro and how much is measures.'),
    Equation(
        'ggwedge', 'From the measure to the general-government primary balance',
        r"""\pi_p=100\,\frac{D_p}{Y_p}+\omega,\qquad \omega=\pi^{AMECO}_{T^*}-100\,\frac{D_{T^*}}{Y_{T^*}}""",
        "π_p is the general-government primary deficit as a share of GDP, the quantity the debt arithmetic of @sec:debtpath@ needs. The measure D is central-government or federal for US/UK/DE/FR/IT, so ω bridges it to the GG balance: the gap between AMECO's GG primary balance and the measure at the last year both are published (T*), held constant. For ES/SE/NO/JP the measure is already GG and ω = 0. The current ω per country is in the table under this section.",
        'sovereign_cash/scenarios/primary.py: wedge, gg_primary_pct, GG_MEASURE',
        'A cash or federal measure and a Maastricht primary balance differ by local government, social security and accrual timing. The wedge is observed, not modelled: it is the last published gap, held.'),
    Equation(
        'hold', 'WEO primary after the tested horizon',
        r"""\pi_p=\pi^{\mathrm{WEO}}_{p}\quad(p>T+3),\qquad \hat Q_p=\hat R_p+\frac{\pi^{\mathrm{WEO}}_{p}-\omega}{100}\,Y_p,\quad \mathrm{Pol}^{Q}_{p}=\mathrm{Pol}^{R}_{p}=0""",
        "Beyond T+3 the GG primary deficit is the IMF WEO primary (rule weo_pct). The US is the exception: its DTS line model, every line on a driver (@sec:usflow@), runs to FY2030 (primary.US_DTS_TO_YEND), because handing FY2030 to WEO put a step of more than 1pp of GDP between FY2029 and FY2030 that no driver explains. The measure primary is set to π^{WEO}−ω so that after the held GG wedge (@sec:ggwedge@) the stamped GG primary equals WEO; ω=0 where the measure is already GG. Share / cash / DTS rules are tested only one to three years out (TESTED_H = 3); freezing the T+3 residual (held_pct) was an unscored convenience that manufactured artefacts (Japan P-driver surplus) and dressed authority consolidation gaps as house views. held_pct remains only if a WEO cell is missing. Policy overlays are zeroed in those years — WEO already includes them. Japan still forces primary spend onto GDP inside the tested window (PERSIST_GDP_PRIMARY) so the handoff year is not poisoned by Y−P drift. Interest stays on the ISIN book. Horizon 2030 (Y_END).",
        'sovereign_cash/scenarios/primary.py: path, TESTED_H, Y_END, WEO_PRIMARY_HOLD, PERSIST_GDP_PRIMARY, weo_primary_deficit_pct',
        'Short-horizon primary is ours; medium-term primary is the authority. The interest edge is the forward curve on the ISIN book, not a better primary path.'),
    Equation(
        'interest', 'Interest from the ISIN book',
        r"""I_m=\sum_{i\in\text{fixed}} c_i N_i\,\mathbb 1[m\in\text{pay}_i]\;+\;\sum_{j\in\text{float}} (\text{ref}_m+s_j)\,N_j\;+\;\sum_{k\in\text{bills}} N_k\,\bigl(1-P_k\bigr)\mathbb 1[m=\text{mat}_k]\;+\;\sum_{l\in\text{linkers}} c_l N_l\,\frac{\mathrm{CPI}_{m-\text{lag}}}{\mathrm{CPI}_{\text{base},l}}\;+\;\sum_{n\in\text{new}} y(m_n,\tau_n)\,N_n\;+\;\bar I^{NM}""",
        "I_m is cash interest in month m. For every ISIN i in the book: N the outstanding nominal (less the central bank's holding where the measure is private), c the coupon, pay_i the payment months. Floaters reset on the reference rate ref_m plus spread s_j; bills pay their discount 1 − P_k at maturity; index-linked coupons scale with the CPI ratio at the issue's lag (RPI for gilts). New issues n needed to fund the cash requirement are priced at the forward par yield y(m_n, τ_n) of the government zero curve at their issue month and tenor (next equation), tenor mix from the trailing year's issuance. Ī^{NM} is the non-marketable residual (retail, loans, intra-government) frozen at its last outturn.",
        'sovereign_cash/debt/isin_book.py; sovereign_cash/debt/cashflow.py; sovereign_cash/isin_curve_interest.py',
        'Interest is not a ratio times a stock: every bond in the book is rolled month by month, with locked coupons on the existing stock and the curve only on what has to be issued. The yield shock channels of the shock part (@sec:interest@) change y and ref and nothing else.'),
    Equation(
        'bookrate', 'Forward par yield on the government zero curve',
        r"""y(m,\tau)=\frac{1-P(m,\tau)}{\sum_{k=1}^{\tau} P(m,k)},\qquad P(m,k)=\frac{P(0,m+k)}{P(0,m)}""",
        "P(0, t) is today's discount factor from the fitted government zero curve (Nelson–Siegel–Svensson on the country's benchmark yields); P(m, k) the forward discount factor for a k-year cash flow issued at m. y(m, τ) is the par coupon that prices a τ-year bond at 100 when issued in month m — annual coupons, so the formula is annual. Beyond the last liquid tenor the curve is held flat.",
        'sovereign_cash/debt/gov_curve.py; sovereign_cash/isin_curve_interest.py',
        'Issuing at the forward par yield is the arbitrage-free way to book a coupon that does not yet exist; it means the interest path moves with the whole curve, not with one headline yield.'),
    Equation(
        'financing', 'Net issuance',
        r"""NI_p=D_p+I_p+W_p,\qquad NI^{priv}_{p}=NI_p-\Delta H^{CB}_{p},\qquad \text{gross}_p=NI_p+\text{redemptions}_p+\text{buybacks}_p""",
        "NI_p is net marketable issuance; D_p the primary deficit and I_p interest from the sections above; W_p the financing wedge — the part of the published financing requirement outside the deficit (lending, financial transactions, cash-buffer change, FX and timing), carried from the authority's financing table where there is one and from its trailing outturn otherwise. ΔH^{CB}_{p} is the central bank's net purchase (negative under run-off), so NI^{priv} is what the private sector must take. Gross issuance adds redemptions from the ISIN book's maturity calendar and announced buybacks.",
        'sovereign_cash/debt/build.py: history, forward; sovereign_cash/core/cash_framework.py: ladder_monthly, cb_redemptions',
        'Net issuance is an identity on the pieces already derived; the only new input is the financing wedge, which no model in this chain claims to forecast.'),
    Equation(
        'debtpath', 'Debt ratio recursion',
        r"""d_{t+1}=d_t\,\frac{1+r_t}{1+g}-pb,\qquad pb=-\pi\ \ (\text{primary surplus}>0)""",
        "d_t is general-government gross debt to GDP, r_t the effective nominal interest rate on the stock, g the nominal growth rate (the engine's 2026–29 compound rate of nominal GDP, @sec:macro@) and pb the primary surplus in % of GDP. The 'now' primary is the engine's year-T+1 value; the 'plan' primary is its 2029 value; 'halfway' moves half of the way from now to the debt-stabilising pb* two equations down. The scenario grid (rate shock, growth shock, primary) is tabulated under this section; ten years from the snapshot.",
        'sovereign_cash/debt/note/debt_paths.py: sim, SCENARIOS; sovereign_cash/debt/dalio_note.py: sustain; sovereign_cash/debt/debt_path_score.py → derived/debt_path_score.json',
        'Walk-forward score of the recursion (realised g, held pb, reprice ρ) lives in ``derived/debt_path_score.json`` (``debt/debt_path_score.py``): PRE origins ≤2019, OOS ≥2022, Covid dropped. Ship set is the countries that beat flat and last-Δ on PRE and clear the OOS band (currently DE/JP/ES). Board path charts for other countries (including the US and France lead paths) remain scenario arithmetic on the engine primary; the JSON is the falsifiable statement for the recursion itself.'),
    Equation(
        'repricing', 'How fast the effective rate reaches the marginal rate',
        r"""r_{t+1}=r_t+\rho\,\bigl(r^{m}+\Delta-r_t\bigr),\qquad \rho=1-\bigl(1-s_{3y}\bigr)^{1/3}""",
        "r^{m} is today's marginal rate — the ISIN book's issuance-weighted yield on the government curve — and Δ the scenario shock in rate terms. s_{3y} is the share of the stock that reprices within three years, from the book's maturity and floater schedule, so ρ is the equivalent constant annual repricing fraction. r_0 is GG interest to GDP divided by debt to GDP in the last complete outturn year on the debt panel (not the forecast-year row).",
        'sovereign_cash/debt/dalio_note.py: sustain; derived/debt_repricing.parquet',
        'A rate shock reaches the interest bill only as the book turns over. Reading ρ off the ISIN book, not a stylised average maturity, is what makes the US and Italy diverge in the fan.'),
    Equation(
        'pbstar', 'Debt-stabilising primary balance',
        r"""pb^{*}=\frac{r^{m}-g}{1+g}\,d_0""",
        "pb^{*} is the primary surplus that holds d constant once the whole stock has repriced to r^{m}. The 'adjustment' columns on the debt board are pb^{*} − pb_{now} and pb^{*} − pb_{plan}.",
        'sovereign_cash/debt/dalio_note.py: sustain (pbs)',
        "The gap between pb^{*} and the engine's primary balance is the sustainability gap; the board ranks countries on it as a cross-section, not a predicted break probability."),
    Equation(
        'scenlines', 'Every scenario leg on the same budget lines',
        r"""\Delta D_{c,y}=\underbrace{\sum_{\ell}\Delta Q^{real}_{\ell,y}-\sum_{\ell}\Delta R^{real}_{\ell,y}}_{\text{growth, real part}}+\underbrace{\pi\,Q_{y}\,x_{y-1}-R_{y}\,x_{y}}_{\text{price leg}}-\underbrace{\Delta R^{rates}_{y}}_{\text{rate-linked receipts}},\qquad \Delta I_{c,y}=\Delta I^{book}_{y}(b)+r^{m}\,(\tilde D_{y-1}-D_{y-1})+L\,(x_{y}-x_{y-1})""",
        "ΔD_{c,y} is the change in country c's primary deficit in year y against baseline (+ = more borrowing), the same function for every scenario in the note's grid, the repression chart and the downturn line of the scenario-interest chart. A nominal-growth shock of g pp a year for two years (level gap g, then 2g, no catch-up) splits into a real part (REAL_SHARE = 0.75; @sec:scenlines@) and a price part x_y = 0.25 × gap on the price leg; an inflation shock is the price leg alone with x_y the price-level gap; a rates shock moves only the receipts that rates move. ℓ are budget lines: US federal Daily Treasury Statement buckets, UK OBR ready-reckoner lines, elsewhere personal income tax, social contributions, corporate tax, indirect taxes and current primary spending. Interest ΔI: the ISIN book re-run on the shifted curve (calendar years through 2030 for every country, straight-line between the −100 / 0 / +100bp runs), the marginal rate r^m on the extra debt, and index-linked debt L accruing the extra inflation at once. US general government: state and local receipts move with nominal GDP and their spending matches them (balanced budgets), so only the interest / revenue denominator moves. 'Half spent' adds half of an upturn's receipts windfall to spending. Baseline interest: the debt board's path, scaled to the reviewed interest / revenue in 2026 and 2030.",
        'sovereign_cash/scenarios/cyclical.py: lines, price_lines, rates_lines, response_all; sovereign_cash/debt/note/scenario_grid.py: run_country, book_rates, linker_stock; sovereign_cash/debt/note/rebuild_ex1_gap.py: _growth_borrowing',
        'One model carries growth, inflation and rates, so a downturn, a reflation and a sell-off differ only in which drivers move, not in which model answers. The shock dashboard (Equations) still runs the aggregate channels.'),
    Equation(
        'cycreal', 'Growth, real part: OECD elasticities by tax and for benefits, corporate tax on profits',
        r"""\Delta R^{real}_{t,y}=R_{y}\,w_t\,\varepsilon_t\,\gamma_y\ (t\in\{PIT,SSC,IND\}),\qquad \Delta R_{CIT,y}=R_y\,w_{CIT}\,\bigl(s\,\Pi_{y}+(1-s)\,\Pi_{y-1}\bigr),\qquad \Delta Q^{real}_{y}=Q^{cur}_{y}\,\tilde\varepsilon_{Q}\,\gamma_y,\qquad \gamma_y=0.75\,\mathrm{gap}_y""",
        "For DE, FR, IT, ES and JP (general government). w_t is the tax's share of general-government revenue and ε_t its elasticity to the output gap (Price, Dang and Botev 2015, OECD Economics Department WP 1275, Tables 8-10). Corporate tax follows the common nominal profit path Π_y used for the US and UK (−10 / −20 / −15 / −12% in the canonical downturn, capped at 5 × the nominal gap; mirrored in an upturn) with a tax-to-profit elasticity of 1: the OECD gap elasticities average the cycle and understate sharp recessions. s is the share paid in the year: France 0.35 (acomptes on last year's profit, solde in May: PLF Voies et moyens), Japan 0.5 (interim payment on half the prior year's tax), 1 elsewhere. ε̃_Q is the OECD current-spending elasticity with its unemployment part rebased to today's spending share where it has moved: Spain 3.75% of current primary spending (OECD's 2002-11 weight 6.26%), France 3.4% (ESSPROS 2024 early estimates: unemployment benefits 1.5% and 1.75% of GDP). The tests hold the implied budget-balance semi-elasticity within 0.1 of OECD Table 10 (Spain back to 0.63 once rebased).",
        'sovereign_cash/scenarios/cyclical.py: OECD, _oecd, profit_path, cp_elasticity, balance_semi_elasticity; sovereign_cash/tests/test_cyclical.py',
        'Where no line-level national reckoner exists, the OECD estimates are the country-specific elasticities the European Commission uses for its cyclical adjustment; corporate tax follows the same profit path as the US and UK so the downturn is comparable across countries.'),
    Equation(
        'cycus', 'Growth, real part: US federal lines on their drivers',
        r"""\Delta u_y=-0.65\,\gamma_y,\quad \Delta\mathrm{WB}_y=0.9\,\gamma_y,\quad \Delta R_{NW}=R_{NW}\bigl[(1+\gamma_y)(1+e_y)^{0.7}-1\bigr],\quad \Delta R_{CIT}=1.5\,R_{CIT}\,\Pi_y-\kappa_y\,K\,s\,\tau,\quad \Delta R_{CUS}=2.5\,R_{CUS}\,\gamma_y,\quad \Delta Q_{UI}=Q_{UI}\Bigl[\bigl(\tfrac{u+\Delta u_y}{u}\bigr)^{1.54}(1+\Delta\mathrm{WB}_y)-1\Bigr]""",
        "US federal lines from the DTS. Unemployment rises by Okun (0.65pp per 1% of real gap; 2001: +2pp on a ~3% gap); withheld tax on the wage bill (elasticity 1.0; DTS 2014-26 ex Covid 0.73, statutory ~1.1); non-withheld tax on nominal GDP and the S&P 500 (e_y: −12 / −25 / −15 / −10% in the canonical downturn; elasticity 0.7 from DTS FY2018-26 and the FY2009 April payments); corporate tax 1.5 × profits Π_y (FY2000-03 −36%, FY2007-09 −63%) less the AI-capex expensing cushion (κ_y capex cut × K $550bn × 60% expensed × 21%: lower capex means fewer deductions, more tax); customs on real imports (2.5 × the real gap); unemployment insurance from log(UI / wage bill) on log u (DTS 2014-26, 1.54, R² 0.68); SNAP +5% and Medicaid +1% per pp of unemployment; refunds and early Social Security claims rise. Equity and profit paths are nominal, so the price leg does not add to the lines they drive. Stimulus (1% of GDP in year 2) is a switch, off in the grid. Ex equities the federal response is on CBO's automatic-stabiliser scale; CBO does not count equity-driven capital-gains swings as cyclical.",
        'sovereign_cash/scenarios/cyclical.py: _us; sovereign_cash/scenarios/channels/us_dts.py: fy_table',
        'The US budget is unusually exposed to equities and corporate profits; a line model is the only way to show where an AI-led downturn lands.'),
    Equation(
        'cycuk', 'Growth, real part: the OBR ready reckoner, line by line',
        r"""\Delta B^{UK}_{y}=\sum_{d}\sum_{j\le y}c_{d,j\to y}\,z_{d,j},\qquad z=\{\text{earnings},\ \text{employment},\ \text{unemployment},\ \text{consumption},\ \text{profits},\ \text{equities},\ \text{imports},\ \text{housing},\ \text{GDP}\}""",
        "c_{d,j→y} is the OBR ready reckoner coefficient (March 2025 EFO) for determinant d shocked in fiscal year j on each receipt and spending line in year y; z_{d,j} the determinant's move: real wage bill 0.9 × γ less employment (0.4 × γ), unemployment Okun 0.4 on a 34m labour force (input in millions), consumption, GDP and imports with γ, profits and equities on the common nominal paths, house prices 1.5 × γ, transactions 4 × γ. Debt interest and the CGNCR financing loop are left to the ISIN book. Checked against the OBR's cyclical-adjustment rule (0.5 contemporaneous, 0.2 lagged per 1% of output gap).",
        'sovereign_cash/scenarios/cyclical.py: _uk_all; sovereign_cash/uk/ready_reckoner.py: run',
        "The UK is the one country with a published line-level reckoner; capital gains and stamp duty carry the equity channel a year later."),
    Equation(
        'cycprice', 'The price leg, common to growth and inflation',
        r"""\Delta R^{px}_{y}=R_{y}\,x_{y},\qquad \Delta Q^{px}_{y}=\pi\,Q_{y}\,x_{y-1},\qquad \pi=0.8,\qquad \Delta I^{link}_{y}=L\,(x_{y}-x_{y-1})""",
        "x_y is the price-level gap: 0.25 × the nominal gap in a growth scenario, the inflation path (infl, then 2 × infl) in reflation. Receipts follow their nominal bases in the year. Primary spending follows a year later (benefit and pension uprating on last year's inflation, pay settlements and budgets set ahead) for the share π that is price-linked within the horizon. π = 0.8 is calibrated to CBO's inflation rule of thumb (outlays 44 / 95 / 156bn per pp in years 2-4 against ~62bn a year for full lagged indexation) and fits the EU spending mix; a test holds the US price leg within a factor of two of CBO each year. US lines take the same rule line by line. The UK runs the OBR reckoner's price determinants (CPI, RPI, earnings, consumption, profits, GDP, imports, house prices; September CPI uprating, the triple lock, frozen thresholds) and, because the reckoner holds departmental spending (DEL, ~half of primary spending) fixed in cash, adds DEL at the same π a year later. In a growth scenario, lines already on nominal paths (US corporate and non-withheld tax, UK profits, corporate tax elsewhere) are left out of the price leg. Index-linked debt L (US, UK, France: the issuer's stock with uplift; Germany, Italy, Spain, Japan: the ISIN tape's linker face) accrues the higher inflation at once, as interest. Fiscal drag from unindexed thresholds in IT, ES and JP and Japan's pension macro slide are not modelled.",
        'sovereign_cash/scenarios/cyclical.py: _price, price_lines, PRICE_PASS_THROUGH, PRICE_LAG_YEARS, UK_DEL_SHARE; sovereign_cash/debt/note/scenario_grid.py: linker_stock',
        'Inflation helps the primary for a year or two because receipts reprice before spending does, while linkers cost at once; the same leg answers the price part of every growth scenario, so reflation and nominal growth cannot disagree.'),
    Equation(
        'cycrates', 'Rates: the receipts that rates move',
        r"""\Delta R^{rates}_{UK,y}=\sum_{d\in\{\text{Bank Rate},\ \text{deposit rate}\}}c_{d,\cdot\to y}\,\tfrac{b}{100},\qquad \Delta R^{rates}_{JP,y}=\max\bigl(-P^{BoJ},\,-\mathrm{ER}\cdot\tfrac{b}{10^4}\bigr)\ (\text{paid the May after the fiscal year}),\qquad \Delta R^{rates}_{US}=\Delta R^{rates}_{EA}=0""",
        "b is the parallel shift in basis points. UK: OBR reckoner Bank Rate and deposit-rate lines (interest and dividend receipts, self-assessment tax on savings income, the Bank's reserves). Japan: the BoJ's payment to the Treasury (¥1.83tn for FY2025) falls as interest on excess reserves ER (~¥430tn) rises, floored at zero (P^{BoJ} ~¥1tn once the June 2026 hike is in); a cut raises it in full. US: none within the horizon: the Fed remits nothing while it carries a deferred asset (~$244bn, Q1 2026), so neither a rise nor a 75bp cut reaches the Treasury before 2030. Euro area: none (national central banks carry losses forward). Interest itself is the ISIN book.",
        'sovereign_cash/scenarios/cyclical.py: rates_lines, BOJ_EXCESS_RESERVES_BN, BOJ_REMIT_BASE_BN',
        'Rates move the primary only through a few receipt lines; most of a sell-off works through interest, which is the bond-by-bond book.'),
    Equation(
        'cashid', 'Monthly cash identity',
        r"""C_m=\sum_{\ell}\sigma_\ell\,L_{\ell,m},\qquad C^{(h)}_{o}=\sum_{k=1}^{h}C_{o+k}""",
        "C_m is the country's monthly cash requirement (+ = need to borrow), L_{ℓ,m} its published lines and σ_ℓ the sign (+1 spending, interest, lending; −1 receipts). C^{(h)}_{o} is the cumulative requirement over the h months after origin o, the object every cash model is scored on. Country measures: UK CGNCR ex Network Rail; France central-government cash requirement; Germany Federal core cash requirement plus special funds; Italy state-sector fabbisogno; Spain IGAE central-administration B.9 (deficit +); Sweden Riksgälden net borrowing requirement; Norway Norges Bank structural liquidity; Japan MoF general account.",
        'sovereign_cash/core/cash_framework.py; country modules sovereign_cash/{uk,fr,de,it,es,se,no,jp}/',
        "Everything inside twelve months runs on published monthly cash. Each country's lines, model family and ship state differ and are given in turn; the identity, scorer and selection rule are shared."),
    Equation(
        'cashmodels', 'The trailing-level model family',
        r"""\text{smly: }\hat C_m=C_{m-12};\quad \text{seas\_add}_K:\ \hat C_m=\frac{\Lambda}{12}+\bar s_{\mathrm{mo}(m)};\quad \text{seas\_mult}_K:\ \hat C_m=\Lambda\,\bar p_{\mathrm{mo}(m)};\quad \text{target}_K:\ \hat C_m=\bar s_{\mathrm{mo}(m)}+\frac{A_y-\mathrm{YTD}_y-\sum_{m'\in\text{rest}}\bar s_{\mathrm{mo}(m')}}{|\text{rest}|}""",
        "Λ is the trailing twelve-month sum through the origin. s̄_{mo} is the mean over the last K complete calendar years of the month's deviation from its year mean, p̄_{mo} the mean of the month's share of the year total (K ∈ {3, 5}). target_K pins the open year's remaining months to an annual A_y (plan, prior-year total under prevyear_K, or plan × an execution haircut hc under planhc_K) while keeping the seasonal shape. Beyond twelve months smly reuses its own forecast. Every model reads at most the trailing twelve months as its level input.",
        'sovereign_cash/core/cash_framework.py: forecast, seasonal_dev, seasonal_share, model_list',
        'The candidate set is deliberately small. Monthly cash is a seasonal pattern around a slowly moving level; SARIMAX on the headline cash series and macro regressions on it were tried for every country and lost (the FR/DE/IT share models of the Equations SARIMAX response work on lines divided by a driver, not on cash levels).'),
    Equation(
        'cashscore', 'Scorer and model selection',
        r"""\mathrm{MAE}_{w,h}=\frac1{n}\sum_{o\in w}\bigl|\hat C^{(h)}_{o}-C^{(h)}_{o}\bigr|,\qquad \hat\theta=\arg\min_{\theta}\ \frac14\sum_{h\in\{1,3,6,12\}}\mathrm{MAE}_{\mathrm{PRE},h}(\theta)\quad\text{if } n_{\mathrm{PRE},12}\ge 36,\ \text{else smly}""",
        'Walk-forward from every monthly origin o. PRE is the set of windows whose target months end by December 2019; OOS the origins from January 2022; any window touching 2020–21 is dropped. The model θ is the lowest mean PRE MAE over the four horizons among models scored at all four; if fewer than 36 PRE windows exist at h = 12 the default smly is used and the summary says so (DE core: planhc_3 chosen a priori instead, see @sec:cashde@). OOS is reported, never used to choose. The benchmark column on every cash board is smly.',
        'sovereign_cash/core/cash_framework.py: backtest, score_table, choose, MIN_PRE, HORIZONS',
        'Selection is on the pre-Covid window only; the out-of-sample column is report-only. At h = 12 every trailing-level model forecasts the trailing twelve-month sum, so h = 12 scores tie by construction and the selection is effectively on h = 1, 3, 6.'),
    Equation(
        'shiprule', 'When a bottom-up path replaces the production model',
        r"""\text{SHIP}\iff \overline{\mathrm{MAE}}_{\mathrm{PRE}}^{BU}\le(1-0.05)\,\overline{\mathrm{MAE}}_{\mathrm{PRE}}^{prod}\ \wedge\ \#\{h:\mathrm{MAE}^{BU}_{\mathrm{PRE},h}>\mathrm{MAE}^{prod}_{\mathrm{PRE},h}\}\le 1\ \wedge\ \overline{\mathrm{MAE}}_{\mathrm{OOS}}^{BU}\le 1.25\,\overline{\mathrm{MAE}}_{\mathrm{OOS}}^{prod}""",
        'BU is the category × macro bottom-up path for the country, prod the production model selected above. Means are over h ∈ {1, 3, 6, 12} on the same windows. PRE edge ≥5% (REL_EDGE) and ≤1 worse horizon, and OOS mean ≤ (1+OOS_BAND)×production with OOS_BAND = 0.25, are enforced for FR/DE/IT; the UK uses EDGE_BAND = 3% on PRE at the horizon against the remit hybrid. Where the gate fails, the candidate stays as a research column.',
        'sovereign_cash/{fr,de,it}/bottom_up_macro.py: REL_EDGE, OOS_BAND; sovereign_cash/uk/bottom_up_macro.py: SHIP, rejects; sovereign_cash/uk/model.py: EDGE_BAND; derived/*_cash_summary.json',
        'A richer model has to earn its place on the pre-Covid window. Where it failed, the candidate stays as a research column on the board and the production path is the seasonal model; this page says which.'),
    Equation(
        'cashuk', 'Production CGNCR(ex): seasonal month effects, closed to the remit',
        r"""\hat C_m=\bar C_{12}+\frac13\sum_{j=1}^{3}\Bigl(C_{m-12j}-\bar C_{12}(m-12j)\Bigr),\qquad \sum_{m\in\mathrm{FY}}\hat C_m\to \mathrm{CGNCR}^{remit}_{\mathrm{FY}}\ \text{once published}""",
        "C̄_{12} is the trailing twelve-month mean at the origin and C̄_{12}(m−12j) the rolling mean at the same month j years earlier, so the bracket is that month's deviation from its own run-rate; the last three are averaged (seasonal_add3). Months 2020-03 to 2021-03 are excluded from the deviations. Within the current fiscal year the path is closed to the DMO remit CGNCR vintage known at the origin after each Budget or revision (additive adjustment over the remaining months); bills after March are held flat and NS&I follows profile_mult5. Rule for the model to replace the default (remit profile / flat stock): beat it by more than EDGE_BAND = 3% on PRE at the horizon; it does so at 6m, so the live CGNCR to March is the seasonal model.",
        'sovereign_cash/uk/model.py: cgncr_forecast, forward, EDGE_BAND; derived/uk_cash_summary.json',
        "The UK's production path is a seasonal model, not the bottom-up. The table below gives the scores that decide it; the next equation is the candidate that did not ship."),
    Equation(
        'cashukbu', 'Bottom-up candidate: OBR categories × ready-reckoner elasticities (NO-SHIP)',
        r"""\hat L_{\ell,m}=L_{\ell,m-12}\,\bigl(1+\beta_\ell\,x_{\ell}\bigr)+\alpha\,\frac{\mathrm{CGNCR}^{remit}_{\mathrm{FY}}-\sum_{m'\in\mathrm{FY}}\hat C_{m'}}{|\text{rest of FY}|},\qquad \mathrm{CGNCR}=\text{welfare}+\text{RDEL}+I+\text{capital}-(\text{IT}+\text{NICs}+\text{VAT}+\text{CT}+\text{other tax}+\text{nontax})+\text{fin. transactions}""",
        "Each category ℓ is last year's same month scaled by its ready-reckoner elasticity β_ℓ times the determinant's YoY x_ℓ known at the origin: welfare 1.5 × CPI, RDEL 0.25 × GVA, capital 0.2 × GVA, other HMRC taxes 0.75 × retail + 0.7 × equity, nontax calendar-only; interest is the OBR debt-interest triangle on Bank Rate and gilt yields, additive; financial transactions are mapped to ONS PUSF lines on seasonal_add3 with PRE-gated overlays; January/February self-assessment takes baseline + 0.35 × last year's excess × the disposal-year equity ratio. α = 0.12 is a soft pin to the remit. PRE mean MAE 9.83 vs production 8.99 (edge −9.3%): NO-SHIP; the bottom-up is a research column on the UK board.",
        'sovereign_cash/uk/bottom_up_macro.py: SPEC, forecast; sovereign_cash/uk/bottom_up.py: ship_rule',
        "The desk's categorical model is written down so a reader can see what was tried and why the seasonal model still ships. Its full spec dictionary is in the fold under this section."),
    Equation(
        'cashfr', 'PLF categories × RESF macro, pinned to the PLF balance (SHIP)',
        r"""\hat L_{\ell,y}=\begin{cases}\mathrm{PLF}_{\ell,y} & \text{vintage known at origin}\\ L_{\ell,y-1}\,(1+\hat\pi^{cpi}_{y}) & \ell=\text{remboursements locaux}\\ L_{\ell,y-1} & \text{otherwise}\end{cases},\quad \hat L_{\ell,m}=\hat L_{\ell,y}\,\bar p^{\,5y}_{\ell,\mathrm{mo}(m)},\quad \hat C_m=\sum_\ell\sigma_\ell\hat L_{\ell,m}+\alpha\,\frac{S^{PLF}_{y}-\sum_{m'\in y}\hat C_{m'}}{12}""",
        'Lines ℓ: crédits, charge de la dette, CAS pensions, remboursements locaux, PSR, less IR, IS, TVA, accises, autres fiscales, non-fiscal, less the solde of comptes spéciaux and budgets annexes (deficit +). An open year with a PLF catégorie vintage known at the origin (deposit ≈ October of y−1) takes that annual; otherwise most categories hold the last complete SMB/PLF year — same-year RESF and OLS variants failed PRE — except remboursements locaux, which grows with the CPI prior. Charge = AFT when known (2024 exécution beat PLF on OOS), else the ISIN book + curve annual (@sec:interest@). Month shape p̄ is the five-year mean SMB share by category; interest months are same-month-last-year. α = SOFT_PIN_ALPHA = 1 closes the sum to the PLF solde S^{PLF}_{y} when known. PRE mean MAE 7.44 vs the seasonal model seas_mult_3 8.13 (+8.5%), worse at one horizon, OOS 8.04 vs 14.74: SHIP, so seas_mult_3 is now only the seasonal fallback / scoreboard.',
        'sovereign_cash/fr/bottom_up_macro.py; sovereign_cash/yoy_components.py: SPEC; derived/fr_cash_summary.json',
        'France is the cleanest bottom-up: the budget law gives every line an annual, the SMB gives the month shape, and the one macro elasticity that survived PRE is CPI on local-authority refunds.'),
    Equation(
        'cashde', 'Open year: Soll × execution ratios (SHIP)',
        r"""\hat A^{spend}_{y}=\mathrm{Soll}^{spend}_{y}\bigl(0.8\,\overline{\mathrm{Ist/Soll}}^{hist}+0.2\,\mathrm{Ist/Soll}^{YTD}_{y}\bigr),\qquad \hat A^{rev}_{y}=\mathrm{Soll}^{rev}_{y}\bigl(0.95\,\overline{\mathrm{Ist/Soll}}^{hist}+0.05\,\mathrm{Ist/Soll}^{YTD}_{y}\bigr),\qquad \hat C_m=\sum_\ell\sigma_\ell\,\hat A_{\ell,y}\,\bar p_{\ell,\mathrm{mo}(m)}""",
        "Soll is the Federal budget plan (BMF Monatsbericht), Ist the outturn. For the open year each side is the plan times an execution ratio: the historical mean Ist/Soll weighted 80% (spending) or 95% (revenue) with the year-to-date ratio for the rest; the remaining months take the category's month shares p̄. Revenue leans on history because the YTD ratio is dominated by tax-timing noise. Special funds (Sondervermögen) have no plan history and are carried separately on their published path. Core PRE mean MAE 6.18 vs 6.61 for the prior live path (+6.5%), worse at 0 of 4 horizons, OOS 8.96: SHIP. The scoreboard model planhc_3 (Soll × execution haircut) is chosen a priori, not PRE-gated — the 2017–19 plans printed deficits that closed as surpluses, and the trailing models cannot see the debt-brake reform.",
        'sovereign_cash/de/bottom_up_macro.py; derived/de_cash_summary.json (bottomup_macro_authority)',
        'Germany publishes a monthly plan-versus-outturn table, so the open year is a question of how much of the plan gets executed, not of what the plan is.'),
    Equation(
        'cashdelater', 'Later years: same month last year scaled by macro',
        r"""\hat L^{rev}_{m}=L^{rev}_{m-12}\bigl(1+1.0\,\hat g^{Y}_{y-1}\bigr),\quad \hat L^{pri}_{m}=L^{pri}_{m-12}\bigl(1+w_{soz}\,0.35\,\hat g^{W}_{y-1}\bigr),\quad \hat I_y=I^{book}_{y-1}\,\frac{B^{FA}_{y}}{B^{FA}_{y-1}}\bigl(1+2.0\,\Delta r_{y-1}\bigr),\ \ \hat I_m=\hat I_y\,\bar p^{\,I}_{\mathrm{mo}(m)}""",
        "g^{Y} is nominal GDP growth and g^{W} Eurostat compensation growth, both lagged one year (known at the origin); w_{soz} is the social-spending share of primary spending from the BMF Aufgabenbereiche table. Revenue is unit-elastic to nominal GDP; primary spending carries 0.35 × wage growth on its social share. Interest is the debt book's annual (@sec:interest@) grown with the Finanzagentur stock B^{FA} and a 2.0 × change in effective rates, spread by the carve-out month share. Leaf-level elasticities (Lohnsteuer, USt, KSt) are audit-only: Steuerarten start in 2017 and could not be fitted on PRE.",
        'sovereign_cash/de/bottom_up_macro.py; derived/de_cash_summary.json',
        'Past the plan year the German path is the same trailing-level logic as everywhere else, with the three elasticities that have an a-priori value and a PRE test behind them.'),
    Equation(
        'cashit', 'Production: same month last year, DFP plan allocated by month (production)',
        r"""\hat C_m=\begin{cases}\dfrac{C_{m-12}}{\sum_{m'\in y}C_{m'-12}}\;\mathrm{PLAN}_y & y\in\{2026,2027\}\\[8pt] C_{m-12} & \text{otherwise}\end{cases},\qquad \mathrm{PLAN}_y=\frac{\text{settore statale}\ \%\mathrm{GDP}_y}{100}\times\mathrm{GDP}^{DFP}_{y}""",
        "C is the state-sector fabbisogno (Banca d'Italia). For the years the DFP covers the annual is the plan — DFP 2026 Tav. II.1.2.5: 5.6% of GDP in 2026, 4.8% in 2027, on Tav. II.1.2.2 nominal GDP (2,313 and 2,373 EUR bn) — spread over months by last year's month shares; the BOT stock is held at its current level. Other years are same-month-last-year, the scoreboard model (OOS h = 1 MAE 5.26bn). The scored smly stays on the chart beside the live plan path.",
        'sovereign_cash/it/run.py; sovereign_cash/it/events.py: PLAN, plan_mn; derived/it_cash_summary.json',
        "Italy's production path is the plan on a seasonal shape. The macro bottom-up below is scored on every board but does not ship."),
    Equation(
        'cashitbu', 'Bottom-up candidate: BdI categories × macro (NO-SHIP)',
        r"""\hat L^{tax}_{m}=L^{tax}_{m-12}\bigl(1+0.55\,\operatorname{clip}_{\pm 2.5pp}\hat g^{Y}_{y-1}\bigr),\quad \hat L^{pri}_{m}=L^{pri}_{m-12}\bigl(1+0.35\,\hat\pi^{cpi}_{y-1}\bigr),\quad \hat L^{cap}_{m}=L^{cap}_{m-12}\bigl(1+0.2\,\hat g^{Y}_{y-1}\bigr),\quad \hat L^{treas}_{m}=0.94\,L^{treas}_{m-12}""",
        'Lines: tax revenue (β 0.55 on lag-1 nominal GDP growth, clipped at ±2.5pp — the PRE support), other revenue smly, primary spending (0.35 × lag-1 HICP), capital spending (0.2 × GDP), treasury balances scaled 0.94, interest from the ISIN book + curve. PRE mean MAE 6.76 vs production 5.52 (−22.5%; h = 12 14.0 vs 6.8 drives it), OOS 11.20 vs 11.27: NO-SHIP. The OOS diagnosis attributes the twelve-month miss to treasury balances (~41bn), capital spending (~28) and tax under the clip (~22) in the 2022–24 nominal boom.',
        'sovereign_cash/it/bottom_up_macro.py; derived/it_cash_summary.json (bottomup_*)',
        'The Italian categories do not add information over the plan-on-seasonal path on the pre-Covid window; what was tried and rejected is listed in the fold.'),
    Equation(
        'cashes', 'Spain: IGAE leaves × prior-year macro, blended with the total (SHIP, thin gate)',
        r"""\hat L_{\ell,m}=L_{\ell,m-12}\bigl(1+\beta_\ell\,x_{y-1}\bigr),\qquad \hat C_m=0.7\sum_\ell\sigma_\ell\hat L_{\ell,m}+0.3\,C_{m-12},\qquad \beta=\{\text{tax }0.5,\ \text{soc. contributions }0.5/0.3,\ \text{wages }0.5,\ \text{intermediate/benefits/other }0.35,\ \text{capital }0.2\}""",
        "C is IGAE Administración Central B.9 (deficit +), monthly from year-to-date. Revenue leaves take prior-year nominal GDP growth, spending leaves prior-year HICP, interest smly unscaled; the leaf sum is blended 70/30 with the total's smly (LEAF_BLEND). The gate is rest-of-year |error|/|year| from March/June/September origins against the thin-sample model seas_add_3: PRE 0.141 vs 0.159 (+11.3%), OOS 0.361 vs 0.320 (+12.8%, inside the +25% band). The sample is short of the 36-window bar used for DE/FR; the halves fy ≤ 2019 / fy ≥ 2022 are the gate.",
        'sovereign_cash/es/bottom_up_macro.py: BETAS, DRIVERS, LEAF_BLEND; sovereign_cash/thin_cash.py',
        'Spain has leaves but a short monthly history, so the gate matches the sample it has, and the page states that the path is thinner than the others.'),
    Equation(
        'cashse', 'Sweden: five-year same-month mean with half a level shift, under the Riksgälden annual',
        r"""\hat C_m=\frac15\sum_{j=1}^{5}C_{m-12j}+0.5\Bigl(\bar C_{12}-\frac1{60}\sum_{j=1}^{5}\sum_{k=0}^{11}C_{m-12j-k}\Bigr),\qquad \sum_{m\in y}\hat C_m\to \mathrm{NBR}^{RG}_{y}""",
        "C is the Riksgälden net borrowing requirement. The month shape is the five-year same-month mean plus half of the shift between the trailing twelve-month mean and the five-year mean (seas_k5_a0.5_mean, chosen on PRE 2012–19); the level is Riksgälden's latest annual forecast and its rest-of-year total, so the model supplies shape only. PRE MAE 11.3 vs 12.3 smly (bn SEK, monthly balance), Riksgälden's own 8.5 on its 1–6-month horizon. Two bottom-ups were tested under the RG anchor — CH1089 component seasonals (anchored PRE 10.57) and SCB categories × macro (10.35) against production 10.05 — both NO-SHIP.",
        'sovereign_cash/se/; derived/se_cash_summary.json (bottom_up, bottom_up_macro)',
        "Sweden's authority publishes a monthly outcome forecast that beats every model here; the production path takes its level and adds only a month shape."),
    Equation(
        'cashno', 'Norway: Norges Bank inside its window, categories × macro beyond (SHIP)',
        r"""\hat C_m=\begin{cases}\mathrm{NB}_m & m\le \text{end of published NB path}\\[2pt]\sum_\ell \hat L_{\ell,m} & \text{beyond}\end{cases},\qquad \hat L^{oil}_{m}=L^{oil}_{m-12}\,\frac{\mathrm{Brent}_{y}}{\mathrm{Brent}_{y-1}},\quad \hat L^{main}_{m}=L^{main}_{m-12}+0.5\,\hat a_\ell(\text{expanding OLS on }g^{Y}_{main})""",
        'C is the change in structural liquidity (NOK bn), the Norges Bank forecast NB_m every vintage since 2014 (PRE MAE 1.52 at one business day vs 9.27 for the same weekday a year earlier). Beyond the NB path: petroleum tax and SDFI are same-month-last-year times the Brent YoY ratio (unit multiplier); VAT, net tax, NAV, health and local-authority lines add an expanding-window OLS increment on mainland GDP growth, shrunk by 0.5; customs/excise and other net stay smly. Beyond-NB PRE mean MAE 5.95 vs smly 6.73 (January-only targets, n = 6): SHIP. Beating NB inside its own window is disallowed.',
        'sovereign_cash/no/; derived/no_cash_summary.json (bottom_up); derived/no_bottomup_macro_score.json',
        "Norway is the one country where an outside forecast is simply adopted, because it is the central bank's own liquidity management and nothing published beats it inside its horizon."),
    Equation(
        'cashjp', 'Japan: same month last year on the MoF general account',
        r"""\hat C_m=C_{m-12}\quad(\text{fiscal year April–March; MoF monthly file from FY2024})""",
        "C is MoF general-account cash after the tax gross-up. One complete holdout fiscal year exists; on it smly beats a run-rate and a one-year additive seasonal from three origins. The category × macro candidate loses on that year and is NO-SHIP. Japan's longer horizon is the annual share engine (@sec:shares@).",
        'sovereign_cash/thin_cash.py: JP_MODEL; sovereign_cash/jp/',
        "With fourteen months of monthly data the only model that fits is last year's month."),
    Equation(
        'usflow', 'Daily operating flows by bucket',
        r"""\hat y_{b}(d\,|\,o)=\ell_b(o)\,s_b\bigl(\mathrm{cal}(d)\bigr),\qquad \hat F^{(h)}_{o}=\sum_{k=1}^{h}\sum_b \sigma_b\,\hat y_b(o+k\,|\,o)""",
        "b indexes the Daily Treasury Statement Table II buckets (withheld, non-withheld, corporate and customs deposits; salaries, defense, HHS, SSA, interest, refunds and other withdrawals), d a business day, o the origin. s_b is the bucket's calendar shape from a histogram-gradient-boosting fit on calendar features (day of month, week, month, payroll phase, holiday-rolled payment rules such as SSA's Wednesday schedule), ℓ_b the level (next equation). Per bucket the PRE-gated choice is model / flat / spike / customs / ssa_split (flow_model_choice.json). F̂^{(h)} is the cumulative net operating flow, the scored object, with σ_b = +1 for deposits and −1 for withdrawals.",
        'us/fiscal/flow_model.py; us/fiscal/calendar_rules.py; us/fiscal/derived/flow_model_choice.json',
        'The US has daily cash, so the near-term model is daily: a calendar shape times a trailing level per bucket. The scores against flat and last-year are in the table under this section.'),
    Equation(
        'uslevel', 'The trailing level',
        r"""\ell_b(o)=\frac{\sum_{d\in W_{250}(o)} y_b(d)}{\sum_{d\in W_{250}(o)} s_b(\mathrm{cal}(d))}""",
        "W_{250}(o) is the 250-business-day window ending at the origin (LEVEL_WINDOW). The level is the ratio of realised to shape-implied flow over the window, so a bucket that has run above its calendar for a year is carried at that pace. Flat and last-year are the two benchmarks; 'wins' in the score table means the model beats both on the window.",
        'us/fiscal/flow_model.py: LEVEL_WINDOW',
        'Levels drift with the economy and with tax law; a year of history is the shortest window that spans every seasonal and the longest that forgets a tax-season anomaly. Beyond the current fiscal year the trailing level is not used as is: every line goes onto a driver (below), so one-offs and a single year of fast growth are not compounded.'),
    Equation(
        'usinterest', 'Interest withdrawals',
        r"""\hat y_{int}(d)=\kappa\sum_{i:\ \mathrm{pay}_i=d} c_i N_i,\qquad \kappa=\frac{\sum_{d\in W}y_{int}(d)}{\sum_{d\in W}\sum_{i:\mathrm{pay}_i=d}c_iN_i}""",
        "Coupon payments are read from the Treasury securities schedule (every CUSIP's coupon c_i, outstanding N_i and payment date), including coupons on projected issuance so the path does not decay through the year; κ is the trailing ratio of DTS interest withdrawals to scheduled coupons, which absorbs non-marketable and intragovernmental interest.",
        'us/fiscal/flow_model.py; us/fiscal/auction_ladder.py: coupon_schedule_asof',
        'Interest is the one bucket with a published calendar, so its shape is the schedule, not a fit.'),
    Equation(
        'usdtsadj', 'Every US line on a driver: one-offs out, CBO growth for the big programmes',
        r"""\hat y^{adj}_{b,y}=\begin{cases}(y_{b,T}-E^{CBP}_{T})\prod(1+g^{Y})+\text{tail} & b=\text{other outlays}\\ \bigl(y_{b,T-1}(1+\rho)+\phi\,S_T\bigr)(1+\rho)^{y-T} & b=\text{refunds}\\ y_{b,T}(1.10)\prod(1+g^{Y}) & b=\text{corporate}\\ 12\,\bar y^{3m}_{b}(1.04)^{y-T} & b=\text{customs}\\ y_{b,T}(1+g^{WB})^{y-T} & b=\text{withheld}\\ y_{b,T}\prod e^{\ln(1+g^{Y})+0.7(\ln(1+g^{SP})-\ln 1.07)} & b=\text{non-withheld}\\ y_{b,T}\prod(1+g^{CBO}_{b}) & b\in\{\text{Social Security, Medicare}\}\\ y_{b,T}\prod(1+g^{Y}+\delta_b) & \text{every other line}\end{cases}""",
        "The trailing-level path (@sec:usflow@) carries FY2026's one-offs forward and compounds trailing growth (defence +15%, Medicare +13% in FY2027). Before us_dts.rebased uses the buckets' growth, every line is put on a driver: E^{CBP}_T, the IEEPA tariff refunds in Customs and Border Protection withdrawals (Supreme Court, 20 February 2026; ~$147bn over the ~$11bn normal in FY2026, a $35bn tail in FY2027), come out of other outlays; the refund surge S_T from deductions backdated to TY2025 recurs at φ = 60% (the 2026 W-4 does not apply the new deductions by default), trend ρ = 5%; corporate tax on its post-reconciliation run-rate; customs on the last three full months after IEEPA collections stopped; withheld tax on the wage bill; non-withheld tax on nominal GDP and the FY-average S&P 500; Social Security and Medicare on CBO's February 2026 baseline growth; Medicaid at nominal GDP + 1.5pp and SNAP at nominal GDP (the 2025 act's cuts come through the policy overlay, not twice); veterans at nominal GDP + 2pp; defence and every other line at nominal GDP. The line model runs through FY2030 (primary.US_DTS_TO_YEND), so there is no step at the WEO handoff. us_dts_adjust.log() lists every change; the note's 2027 range moves the refund, tariff, corporate and Medicaid assumptions to the ends of their evidence. Cross-check: the general-government primary deficit for 2027 (~2.6% of GDP) sits close to CBO's federal 2.3% before its tariff update plus the ~0.25% of GDP a year the August update adds.",
        'sovereign_cash/scenarios/channels/us_dts_adjust.py: adjust, params; sovereign_cash/scenarios/channels/us_dts.py: fy_table; sovereign_cash/debt/note/framework_sync.py: us_baseline_stats',
        'A trailing level is the right near-term forecast and the wrong multi-year one when the trailing year holds a court-ordered refund programme, a tax-law catch-up and a spending ramp that does not repeat.'),
    Equation(
        'ustga', 'Treasury General Account path',
        r"""\mathrm{TGA}_{t+h}=\mathrm{TGA}_t+\sum_{k=1}^{h}\Bigl(\hat F_{t+k}+\hat N_{t+k}+\hat\delta_{t+k}\Bigr)""",
        "F̂ is the net operating flow of @sec:usflow@, N̂ net financing — public auction cash from the ladder (total less SOMA add-ons, times price plus accrued) less maturities and coupons rolled to the next Fed business day, less buyback cash, plus bills (below) — and δ̂ the SOMA drift (below). Score 'steer_hold' MAE $bn at h = 5/10/20 business days: 23.8 / 53.2 / 87.3 against flat 49.9 / 75.6 / 102.8 and oracle financing 18.3 / 24.6 / 33.9; beyond a week the error is financing, not flows.",
        'us/fiscal/tga_forecast.py; us/fiscal/auction_ladder.py; us/fiscal/derived/tga_backtest_score.json',
        'The cash balance is an identity on flows and financing. The two-week-plus error is the bill decision Treasury has not yet made, so bills are steered to its own stated target.'),
    Equation(
        'ussteer', 'Bill steering to the refunding target',
        r"""\hat N^{bills}_{M}=\mathrm{TGA}^{\ast}_{M}-\mathrm{TGA}_{M^-}-\sum_{d\in M}\bigl(\hat F_d+\hat N^{coupon}_{d}+\hat\delta_d\bigr),\qquad \mathrm{TGA}^{\ast}=\{\text{30 Sep 2026: 950},\ 31\text{ Dec 2026: }850\}\ \$\text{bn}""",
        "TGA* is the quarter-end cash balance assumed in the latest Treasury refunding (TGA_TARGETS, hand-maintained each February/May/August/November). Net bills over the period to the next target are set to deliver it given flows, coupon financing and drift; with no target at least fourteen days out the path holds the origin level and the summary says so (target_source). 'steer_target' uses the realised quarter-end and is conditional — never quoted as a forecast.",
        'us/fiscal/tga_forecast.py: TGA_TARGETS, steer_hold',
        'Rolling bills at constant size loses to a flat TGA at twenty days (129 vs 103); steering them to the announced target is what closes the gap.'),
    Equation(
        'usdrift', 'SOMA drift',
        r"""\hat\delta_d=\underbrace{\sum_{i\in\mathrm{SOMA}} N_i\,\mathbb 1[\mathrm{mat}_i=d]}_{\text{Fed-held maturing}}-\underbrace{\min(\text{cap}_M,\ \text{maturing}_M)}_{\text{QT run-off}}+\text{purchases}_d-\text{buybacks}_d""",
        "DTS Table IIIB equals public auction cash plus SOMA add-ons less maturities at par less buyback cash plus nonmarketables (daily residual sd $0.43bn). The gap to the public ladder is the Fed's holdings by CUSIP: securities it bought in the market that roll at auction, less the monthly QT cap charged from the pre-month snapshot, plus reserve-management bill purchases read from week-on-week CUSIP deltas net of add-ons. h = 63 MAE 22–25bn each year 2023–26 against 25–52 for a trailing mean.",
        'us/fiscal/financing.py: drift_components',
        'The drift is an identity on SOMA holdings, and it decides the sign of the bill call even where it is second-order to the TGA level.'),
    Equation(
        'usreserves', 'Reserves and the funding curve',
        r"""R_{t+h}=R_t-\Delta\mathrm{TGA}-\Delta\mathrm{ONRRP}+\Delta A^{Fed},\qquad \mathrm{SOFR}-\mathrm{IORB}=\max_{\text{soft}}\Bigl(\text{floor},\ c+e^{a-b\,R/\mathrm{GDP}}\Bigr)+\text{calendar}""",
        "R is bank reserves; the first identity rolls the H.4.1 Wednesday level with the TGA path, ON RRP and the Fed's balance-sheet run-off (holiday Wednesdays take the prior business day). The reserve-demand curve is a soft maximum of the ON RRP floor and an exponential in reserves to GDP, capped at the SRF, with a calendar block scaled by scarcity; fitted on 2023, it scores 2024→ at 3.35bp vs 4.48 for a constant. Monthly SERFF averages ship a random walk (PRE picks it); the curve is shown, not used.",
        'us/fiscal/money_markets.py; us/fiscal/reserve_demand.py; us/fiscal/serff.py',
        'Near $3tn the slope is about 1bp per $100bn, so the TGA forecast barely moves SOFR; the curve is on the board to show where the slope steepens, not to trade funding.'),
    Equation(
        'usbridge', 'Monthly supply bridge',
        r"""\Delta\mathrm{TGA}_M=-\mathrm{Def}_M+NC_M+NB_M+\delta_M,\qquad \bigl|\text{residual}\bigr|\le\$1\text{bn}""",
        'By settlement month M and scenario: Def the deficit (operating flows), NC net coupon cash, NB announced plus implied net bills, δ the SOMA drift. The bridge raises if the identity misses by more than $1bn (IDENTITY_TOL) in any month. Year two is a maturity wall, not a forecast: net coupons fall from ~1,163 to ~770 as 2023–25 issuance matures and the bill share of net issuance goes 18% → ~55%; coupons use the two-years-ago template and the TGA is held at the last target.',
        'us/fiscal/supply_bridge.py; issuance/fiscal_bridge.py',
        'The bridge hands the daily US path to the issuance board at monthly frequency, with the identity enforced rather than assumed.'),
    Equation(
        'usbills', 'Private net bills',
        r"""NB^{priv}_{M}=NB^{ladder}_{M}+\mathrm{mat}^{Fed}_{M}-\mathrm{purch}^{Fed}_{M},\qquad \mathrm{mat}^{Fed}_{M}=\text{snapshot holdings maturing}+\text{later purchases maturing}""",
        "Private net bills are the ladder's net bills plus Fed-held bills maturing — both the snapshot holdings and bills bought after it — less Fed purchases. Until 2026-10-01 the second roll term was missing, which charged the whole purchase pace (~$265bn/yr) to the public: twelve-month private net bills read +80bn, the correct figure is +272bn. Once bought bills start maturing (~3 months) purchases are funded by add-ons the ladder already excludes, so private ≈ ladder net bills.",
        'us/fiscal/supply_bridge.py; us/fiscal/financing.py',
        "Who absorbs bills depends on the Fed's roll, which has to be tracked bill by bill."),
    Equation(
        'usaccrete', 'Bill par stock accretion',
        r"""S_{M+1}=S_M+NB^{par}_{M},\qquad \Delta S^{disc}_{d}\approx S_d\,\frac{y^{13w}_{d}}{252}""",
        'S is the bill par stock (private or public-issued) by settlement month; the discount accretes at roughly the stock times the 13-week yield per business day, so par and cash differ by the accreted discount. The private and public-issued stocks feed the US bill DV01 forward of @sec:dv01@.',
        'us/fiscal/supply_bridge.py',
        'Bills are issued at a discount; the stock that matters for DV01 is par, and the difference is this term.'),
    Equation(
        'dv01', 'DV01 of an issue',
        r"""\mathrm{DV01}=\frac{D_{mod}\times P^{dirty}\times N}{10^{4}},\qquad P^{dirty}=\sum_{k}\frac{c/f}{(1+y/f)^{k}}+\frac{100}{(1+y/f)^{n}}+\text{accrued}""",
        'D_{mod} is modified duration, P^{dirty} the dirty price per 100 at the auction yield y with coupon c and frequency f, N face value. Issue DV01 is summed by settlement month and tenor bucket, in local currency and USD. Bills are priced at their discount yield to maturity.',
        'issuance/dv01.py: dirty_price, dv01_local_mn',
        'Issuance is compared across countries in rate risk, not in notional; DV01 is the common unit of every supply chart on the issuance board.'),
    Equation(
        'privatedv01', 'Private DV01 forward',
        r"""\mathrm{DV01}^{priv}_{M}=\bigl(\mathrm{DV01}^{coupon}_{M}+\mathrm{DV01}^{buyback}_{M}\bigr)-\mathrm{DV01}^{official}_{M},\qquad \mathrm{DV01}^{official}_{M}=\overline{\mathrm{DV01}^{CB}}_{3m}\ \ (\text{US bills}),\ \text{announced QT path otherwise}""",
        "Coupon DV01 is the priced issuance calendar extended from the last published month at the current tenor mix; buyback DV01 is negative (bonds retired, `float_rv.buyback_forward`); official DV01 is central-bank absorption — the trailing quarter's pace for US reserve-management bills, the announced run-off path for the APF, Eurosystem and BoJ. Until 2026-10-01 buyback DV01 was hard-coded zero while history netted buybacks out, leaving forecast private DV01 ~$10mn/month too high.",
        'issuance/forward.py: extend_published; issuance/qt_forward.py',
        "The private sector's rate risk to absorb is what remains after the central bank, and the forward has to net the same items history nets."),
    Equation(
        'usbilldv01', 'US bill DV01 scales with the stock',
        r"""\mathrm{DV01}^{bills}_{M}=\mathrm{DV01}^{bills}_{M-12}\times\frac{S^{pub}_{M}}{S^{pub}_{M-12}}""",
        'S^{pub} is the public-issued bill par stock from the supply bridge (@sec:usbridge@). Gross bill DV01 is about stock/12 × 1bp at a constant term mix, so it scales with the stock, not with net bills; the year-ago month carries the seasonal.',
        'issuance/forward.py: extend_published (bill_stock_ratio)',
        'A bill programme that rolls every month has DV01 proportional to what is outstanding, which is why the forward takes the stock ratio rather than the net-issuance sign.'),
    Equation(
        'absorb', 'Absorption: what private holders must add',
        r"""A_{12}=NI_{12}+\mathrm{RO}^{CB}_{12},\qquad \text{history: }\ \mathrm{priv}_{12}=\mathrm{net}_{12}-\mathrm{cb}_{12}""",
        "A_{12} is the change in private holdings over the next twelve months: net issuance (@sec:financing@) plus central-bank run-off RO^{CB} (Fed-held maturities not rolled, APF/PSPP/PEPP redemptions not reinvested, BoJ taper). In history the same object is net issuance less the central bank's net purchases cb_{12}, so the forward and the history are one definition. Non-US central banks do not roll: their maturities reach the market through the net CB flow.",
        'sovereign_cash/debt/build.py: history, forward; sovereign_cash/debt/rebuild_ex8_absorption.py',
        "Absorption is the number the debt note's Exhibit 8 ranks countries on; it is net issuance seen from the buyer's side."),
    Equation(
        'newmoney', 'New private money under reinvestment haircuts',
        r"""\mathrm{New}_{\rho}=NI^{priv}_{12}+(1-\rho)\,\mathrm{Mat}^{priv}_{12},\qquad \rho\in\{1.00,\ 0.95,\ 0.90\},\qquad \mathrm{Mat}^{priv}_{12}=\text{bills}+\text{coupons maturing}\le 1y-\text{CB-held}""",
        "ρ is the share of maturing private holdings assumed reinvested (REINVEST). Private maturities are bills plus 0–1y coupons less the central bank's holdings of them. At ρ = 1 new money equals private net supply; each 5% of non-reinvestment adds that share of the roll. The ratio to last year's elastic buying (foreign official, funds, banks, households from the holder tables) is the stress gauge on the board.",
        'sovereign_cash/debt/build.py: REINVEST, _elastic_buying_bn',
        'Rollover risk is a reinvestment assumption; the board shows three so the reader sees how much of the stress is the roll and how much is new supply.'),
]

EQUATIONS: dict[str, Equation] = {e.id: e for e in _EQ}
_ORDER: list[str] = ['chain', 'macro', 'fiscalyear', 'shares', 'carry', 'gate', 'drag', 'deficit', 'ggwedge', 'hold', 'interest', 'bookrate', 'financing', 'debtpath', 'repricing', 'pbstar', 'scenlines', 'cycreal', 'cycus', 'cycuk', 'cycprice', 'cycrates', 'cashid', 'cashmodels', 'cashscore', 'shiprule', 'cashuk', 'cashukbu', 'cashfr', 'cashde', 'cashdelater', 'cashit', 'cashitbu', 'cashes', 'cashse', 'cashno', 'cashjp', 'usflow', 'uslevel', 'usinterest', 'usdtsadj', 'ustga', 'ussteer', 'usdrift', 'usreserves', 'usbridge', 'usbills', 'usaccrete', 'dv01', 'privatedv01', 'usbilldv01', 'absorb', 'newmoney']
_listed = sorted(set(i for _, _, ids in SECTIONS for i in ids))
assert _listed == sorted(EQUATIONS), (set(_listed) ^ set(EQUATIONS))
assert len(_ORDER) == len(EQUATIONS)
