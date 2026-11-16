# How the sovereign cash forecasts work

Each country answers the same question: **how much cash does the government need this month, and how will it fund it?** The funding side (bonds, bills, cash buffer) is mostly accounting once the cash need is known. What differs is how that cash need is built.

There is no single “macro model” shared across countries. There are six separate recipes. Some grow tax and spending lines off wages, prices, or GDP. Some mostly copy the official budget year and only decide the shape of months. Sweden does not use a category build at all on the live path.

---

## United Kingdom

The UK forecast is the closest thing here to a real bottom-up macro model. It is **not** a free-form fit on our own history alone.

**Levels** come from the Office for Budget Responsibility’s Economic and Fiscal Outlook (how much income tax, VAT, welfare, etc. they expect in each fiscal year).

**Month shape** comes from ONS public-finance history (the usual January/April/etc. pattern for each line).

**Sensitivities** come from the OBR **ready reckoner**: a published spreadsheet that says, for a 1% rise in earnings / CPI / GDP / etc., how many pounds each tax or spending line moves. We apply those published coefficients to live drivers (AWE, retail sales, CPI, GVA, UK share prices, Bank Rate, gilt yields). We did not invent those elasticities from a private regression — the OBR did; we use their table.

In short:

- Income tax and NICs move with average earnings.
- VAT moves with retail spending.
- Corporation tax moves with whole-economy output (stand-in for profits).
- Other taxes (stamp, CGT, IHT, and similar) move with retail spending and UK equity prices.
- Welfare moves with CPI.
- Departmental and capital spending move modestly with output.
- Debt interest is the gilt/bill book plus the yield curve (coupons, linker uplift, bill ladder, new issue at forward par), spread with the usual JW2P month pattern and scaled so the trailing year still matches published interest (covers non-marketable / APF).

A few financing odds-and-ends stay mostly seasonal. If the DMO has published a remit for the central-government net cash requirement, the path is pulled about 12% of the way toward that remit.

So: **OBR year levels + ONS month shape + OBR ready-reckoner × live macro, lightly tied to the remit; interest from the ISIN book and the curve.**

---

## France

France looks like a category model on paper (income tax, corporate tax, VAT, excise, other tax, non-tax, ministerial spending, pensions, local refunds, payroll taxes, debt interest). In practice the live forecast is much simpler.

When the budget (PLF) has published a figure for a category for that year, **that figure is used**. When it has not, the category is held flat at the last complete year. The one exception is refunds to local government, which are allowed to grow with inflation.

Those annual category totals are then turned into months using the average monthly pattern of the last five years of the Situation Mensuelle Budgétaire (the monthly state budget outturn). Debt interest uses the AFT published charge when known, otherwise the ISIN + curve annual; months follow last year’s calendar. If the budget also states the annual deficit to be financed, the whole year is forced to match that deficit.

What this is **not**: a live model where VAT tracks GDP every month, or corporate tax tracks equity prices, or ministerial spending tracks CPI. Those relationships exist in the budget documents as background elasticities, and they were tested as automatic growth rules. They made the pre-2020 forecast worse, so they are not applied. Open years without a PLF line simply hold.

So France live is: **use the official category table when you have it; otherwise hold last year; spread with seasonal months; force the year to the official deficit when that is published.**

Research that scaled a pure seasonal path by inflation (and a bit of European equity on the corporate-tax share) beat a bare seasonal. It does **not** beat this live PLF build (worse on the pre-2020 window). So “make France bottom-up like the UK” is still open — but the seasonal×CPI overlay is not a drop-in upgrade over what is already live.

---

## Germany

Germany’s core federal cash need is built from revenue, primary spending, and interest.

In the current budget year, revenue and spending start from the Finance Ministry’s Soll (planned) path, blended with a little of what has already come in year-to-date. In later years, the forecast mostly repeats the same month last year, then scales revenue up with nominal GDP and scales the social part of primary spending with wages.

Interest in later years is the ISIN + curve annual (coupons, uplift, bills, new issue) times the Finance Ministry interest month pattern.

Finer tax lines (wage tax, VAT, corporate tax, and so on) are recorded for monitoring. They are not what drives the monthly cash engine. Extra-budgetary funds (Sondervermögen) follow their own published annual pace.

There is also a coarser “stick to the financing plan” option used as a scoreboard pin for the debt-brake / funds ramp; that is not the main live core path when the bottom-up macro build is on.

So Germany live (core) is: **this year’s ministry plan blended with outturn; later years same-month-last-year scaled by GDP on receipts and wages on social spending; interest from the debt book.**

Using the later-years rule (same-month-last-year × macro) **inside** the open budget year as well was tried: it fails the pre-2020 gate. The ministry Soll is kept as the open-year engine, not only as a comparison line.

---

## Italy

Italy builds the state borrowing requirement from Bank of Italy monthly lines: tax revenue, other revenue, primary spending, interest, capital spending, and the Treasury’s own balance, plus a small residual against the MEF total.

Each line starts as “same month last year.” Then:

- Tax revenue is scaled with last year’s nominal GDP growth (about half a percent of receipts for each percent of GDP).
- Primary spending is scaled with euro-area / Italian consumer price inflation (about a third of a percent for each percent of CPI).
- Capital spending gets a smaller GDP scale.
- The Treasury balance is slightly shrunk versus a pure repeat of last year.
- Interest is the ISIN + curve annual times Bank of Italy interest month shares (fair OOS monthly beats same-month-last-year by ~5%).

The multi-year DFP/DEF documents set the official narrative and some financing context. They do not allocate the monthly path category-by-category in the live engine (that approach failed out of sample).

So Italy live is: **last year’s months, with tax and investment nudged by GDP and primary spending nudged by inflation.**

---

## Spain

Spain’s monthly cash need is IGAE Administración Central B.9 (deficit +). Live rest-of-year path is **IGAE category leaves × macro** (`es.bottom_up_macro`, SHIP): tax and social contributions scale with prior-year NGDP YoY (β≈0.5); primary spending leaves with prior-year HICP (β≈0.35); interest same-month-last-year; leaf sum soft-blended with total ingresos/gastos. Beats the old `seas_add_3` seasonal by ~11% on pre-2020 rest-of-year |err|/|year|; OOS is a bit worse but inside the +25% band. Sample is thinner than the DE/FR 36-origin PRE bar — gate is the thin_cash halves.

Debt & supply still carries EC / Eurostat fiscal paths for multi-year debt. Marketable interest is Bonos / Letras ISIN + Tesoro curve (`GOV_CURVE`).

## Japan

Japan has a **thin** monthly cash series (MoF general-account from FY2024 only) on the board via `thin_cash` (`smly`). Category×tax/exp × NGDP was tried and **loses** on that one holdout year — NO-SHIP. No PRE gate is possible until more fiscal years land.

Debt & supply: marketable interest from the JGB / T-bill ISIN book + MoF/market curve (`GOV_CURVE`). Fiscal paths follow Cabinet Office then WEO. Point-in-time ISIN+curve interest 2020–25 ≈ ¥177bn MAE vs ≈ ¥417bn same-year-last-year.

---

## Norway

Inside the horizon of Norges Bank’s published liquidity forecast, the model does not invent a better cash need: it uses the official forecast residual.

Beyond that horizon it builds from Norges Bank’s Tab4 spending and revenue lines:

- Petroleum taxes and SDFI move with oil prices (Brent).
- Mainland lines (VAT, other tax, NAV, health, local government, and so on) move with mainland GDP, but only with half pass-through so the path does not overreact.
- Some smaller lines stay as same-month-last-year.

So Norway live is: **trust Norges Bank while they publish; after that, oil lines follow oil, mainland lines follow GDP gently.**

---

## Sweden

Sweden does **not** run a live bottom-up category model.

The live cash path takes the Debt Office’s (Riksgälden) annual borrowing need — or the rest of the year once some months are known — and spreads it with the production seasonal pattern. Budget and KI annuals exist and were tried as category drivers; under the Debt Office year total they barely change the score, and the category build never beat the seasonal path by enough to replace it.

So Sweden live is: **official annual cash need × seasonal months.**

---

## What “macro linked” means here

- **UK** and **Italy** (and Norway beyond the central bank) really do move categories with prices, wages, output, or oil.
- **France** mostly copies the budget year; macro shows up in the PLF numbers the government already chose, not as an automatic monthly rule on each tax.
- **Germany** uses GDP and wages mainly on the outer years after the Soll year.
- **Sweden** is seasonal on an official annual total.

That is why France can feel inconsistent next to the UK: the UK multiplies seasonal months by earnings, CPI, retail, and equities; France pins or holds annual PLF categories and only seasons the months. Research on “scale French seasonal spending by inflation” is about moving France closer to the UK/Italy pattern. It is not what the live French path does today.
