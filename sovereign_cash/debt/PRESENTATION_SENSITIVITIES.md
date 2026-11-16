# Debt & supply — sensitivity presentation

Audience: rates PM who already knows TGA-style cash. Spirit: same as the shipped
interest/revenue scenario block (today vs +100bp / recession / haircuts), not a new
forecast engine.

**No country `*/model.py` edits.** Numbers come from existing derived outputs +
`investor.CHANNEL`.

---

## Information architecture

| Layer | Lives on | Job in ≤10s |
|---|---|---|
| **Interest / revenue scenarios** | Debt tab (already shipped) | “What does the interest bill do under curve / GDP / inflation?” |
| **Supply-demand scenarios** | Debt tab, after Next-12m snapshot | “What does the *market* have to absorb, and under which shock?” |
| **Liquidity CHANNEL** | Overview + plan-gap table | “Where does a fiscal surprise land, and does cash move reserves?” |
| **Panel test** | Debt tab `<details>` | “Does supply predict yields? (mostly no on levels; modest on 10s30s)” |

Loop the page is making legible:

```
fiscal deficit → cash need → bond/bill mix → private net / DV01
    → holders (CB runoff, foreign, domestic) → rates / curve (15y+ face → 10s30s)
    → interest bill → next deficit
```

Plan-gap landing and reinvestment haircuts sit on the cash-need → private-net leg.
Recession / +100bp / QT-style shocks sit on the path and absorption legs.

---

## Scenario matrix

| Scenario | Input | Output shown | Source |
|---|---|---|---|
| **Baseline next-12m** | Live snapshot | Private net % GDP / $, private DV01, rollovers, bill share vs norm, CB net, CB 0–1y | `debt_snapshot` + `private_dv01` |
| **SOMA-stop / QT** | US coupon holdings maturing in 12m hit private; elsewhere = private net | SOMA-stop private % GDP | snapshot `cb_01_bn` |
| **Reinvest 100 / 95 / 90** | Holders roll that share of redemptions | New private money % GDP; 90% × last-year private buying | `debt_reinvestment` |
| **+100bp interest borrowed** | Shock interest financed by extra net | Δ net issuance pp GDP at terminal year | `debt_repricing` `shock_borrowed` |
| **Recession** | Path case in repricing | Δ debt, Δ rollovers, Δ net pp GDP | `debt_repricing` recession |
| **15y+ face → 10s30s** | Private long face vs sample mean × pooled slope | Implied next-12m Δ10s30s bp (long / all-tenor / joint) | `long_end.drifts` |
| **Plan gap** | Model − official cash requirement | Gap bn + where it lands | cash summaries + `tracking()` |
| **CHANNEL** | Institutional facts | Reserves path + shock absorber | `investor.CHANNEL` |

Interest/revenue (+100bp, −100bp, +2pp NGDP, inflation, authority) stays in the
upper block; do not duplicate those bars here — link by proximity on the page.

---

## What each panel answers (≤10s)

### Charts (grid under “Supply-demand scenarios”)

1. **Next 12m private absorption** — Who needs financing capacity this year?
   Private net vs QT-style stop vs 90% reinvest vs CB runoff.
2. **Reinvestment haircut ladder** — How much new private money if rolls slip to 95/90?
3. **Path shocks (terminal year)** — Does +100bp borrowed matter for net? Does recession
   lift debt/rollovers (usually yes) or net (often ≈0 on this path)?
4. **Implied Δ10s30s** — Is long-end private face rich/cheap vs its mean, in bp of curve?

### Tables

5. **Next-12m detail** — Same as chart (1) with DV01 $, bill share / norm, CB 0–1y.
6. **Path shocks detail** — Levels + deltas for debt / net / rollovers.
7. **Long-end face → 10s30s** — Fwd, mean, deviation, implied bp (SE/NO blank: no tape).
8. **Plan-gap + CHANNEL** — Model−plan, CHANNEL absorber, live landing note, cash↔reserves.

---

## CHANNEL wiring (complete)

Every peer row carries:

| Field | Meaning |
|---|---|
| `reserves` | Does the cash buffer move CB reserves? (`CHANNEL[cc][0]`) |
| `absorbs` | Which instrument takes a fiscal surprise? (`CHANNEL[cc][1]`) |
| `lands_on` | Live tracking note when present; else `absorbs` |

| CC | Absorber (CHANNEL) | Typical gap landing |
|---|---|---|
| US | Bills (TBAC 15–20%) | No cash-plan card; TGA / bill share |
| UK | T-bills / DMO net cash | Bills to remit end-March; residual → remit revision (NS&I) |
| DE | Money-market placements, then Bubills | SV vs plan → placements |
| FR | BTF stock | AFT revises BTF change |
| IT | BOT + Treasury liquidity band | Liquidity / BOT |
| SE | Liquidity mgmt + T-bills | Liquidity management |
| NO | None (oil fund) | NGB programme fixed |

---

## Page placement

Debt tab order (relevant slice):

1. Interest and debt service / 2029 GG interest / Central interest *(existing)*
2. Next twelve months snapshot
3. **Supply-demand scenarios** (charts + four tables) ← this work
4. By country panels / Against authorities / Stress / Holders / Supply & CB / Panel test

Overview keeps its own “Liquidity channel and shock absorber” table; debt plan-gap
reuses the same CHANNEL strings so the two tabs do not diverge.

---

## Rebuild

```bash
PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.sensitivities   # JSON stub
PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.page            # HTML + stub
```

Outputs:

- `sovereign_cash/debt_supply_dashboard.html`
- `sovereign_cash/derived/debt_supply_sensitivities.json`

---

## Design-only (not shipped overnight)

- Interactive loop diagram (deficit → … → interest → deficit) as SVG.
- Per-country waterfall: cash need → bonds / bills / other → private net.
- Foreign vs domestic absorption split under haircuts (needs holder flow join).
- Live QT pace slider (SOMA-stop is a discrete scenario, not a continuous control).
- SE/NO long-end face once a 15y+ tape exists.
- Wiring these panels onto the conviction hub overview strip (debt tab is enough for PM use).

---

## Shipped vs design-only

| Item | Status |
|---|---|
| `sensitivities.py` tables (next12, path2029, curve, plan+CHANNEL) | **Shipped** |
| Debt HTML charts ×4 + tables ×4 | **Shipped** |
| CHANNEL on plan-gap (`reserves` / `absorbs` / `lands_on`) | **Shipped** |
| Interest/revenue scenario block | **Already shipped** (0b17ae37) |
| Loop SVG / waterfalls / QT slider / hub strip | Design-only (above) |
| Country forecast engines | **Untouched** |
