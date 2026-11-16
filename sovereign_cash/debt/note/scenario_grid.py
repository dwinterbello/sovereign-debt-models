"""Rates ±75bp × nominal growth ±2pp: who grows out, who breaks on rates, who breaks in a downturn.

General-government debt / GDP and interest / revenue in 2030 against the note's baseline. One model for the primary in
every scenario (``scenarios.cyclical``), interest on the ISIN book (``scenarios.channels.debt_book``):

* growth: nominal GDP 2pp a year below / above baseline for two years (level −2% in the first year, −4% from the
  second, no catch-up). The real part (75%) moves each budget line on its own driver: US federal DTS lines (withheld and
  non-withheld tax on wages and equities, corporate tax on profits net of the AI-capex expensing cushion, customs on
  imports, UI / SNAP / Medicaid on unemployment), state and local assumed to balance; the UK on the OBR ready reckoner
  line by line; the other five on OECD output-gap elasticities by tax and for benefit spending (Price, Dang and Botev
  2015), corporate tax at up to 1.7x the OECD elasticity in a downturn. The price part (25%) is the common price leg below.
* inflation (reflation case, and the price part of growth): receipts follow nominal bases in the year, 80% of primary
  spending follows prices a year later (the UK on the OBR reckoner, departmental budgets uprated like everyone else's);
  index-linked debt accrues the higher inflation at once (interest).
* rates: parallel ±75bp, permanent; extra interest from the ISIN book re-run on the shifted curve, calendar years
  through 2030 for every country, plus interest on the extra borrowing at the marginal rate; the rate-linked receipts
  move the primary (UK interest receipts and tax on savings income, OBR reckoner; Japan the BoJ's payment to the
  Treasury, floored at zero; none for the Fed, which carries a deferred asset, or the euro-area central banks).
* policy asymmetry: an upturn in which half the receipts windfall is spent, against a downturn in which the
  stabilisers run in full.

Baseline: the note's primary path (``note_numbers.json``, US on the framework via ``framework_sync``) and the debt
board's interest path (``debt_country_annual``), scaled to the reviewed interest / revenue in 2026 and 2030. US general
government: state and local receipts move with nominal GDP and their spending matches them (balanced), so only the
interest / revenue denominator moves. Only differences from baseline are reported.

    PYTHONPATH=. python -m sovereign_cash.debt.note.scenario_grid
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTE = Path(__file__).resolve().parent
FIG = NOTE / "figures"
SC = NOTE.parents[1]
PACK = SC / "derived" / "scenario_pack.json"
NUMBERS = NOTE / "note_numbers.json"
OUT_JSON = NOTE / "scenario_grid.json"
CC = ("us", "uk", "de", "fr", "it", "es", "jp")
NM = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy", "es": "Spain", "jp": "Japan"}
Y0, Y1 = 2026, 2030
GROWTH_PP, RATES_BP = 2.0, 75.0
# central / GG revenue already GG for these measures
GG_MEASURE = {"uk", "es", "jp"}


def _period_year(cc: str, label: str) -> int:
    """Calendar year a pack period mostly falls in (US FY ends Sep, UK FY ends Mar)."""
    s = str(label)
    if cc == "us":
        return int(s.replace("FY", ""))
    if cc == "uk":
        return int(s.split("-")[0])  # 2026-27 → mostly 2026 (Apr–Dec)
    return int(s[:4])


def _engine(pack: dict, cc: str, driver: str, size: float, mode: str | None = None) -> dict[int, float]:
    """Engine delta (local bn, + = more borrowing) by calendar year for one driver."""
    from sovereign_cash.scenarios import engine as E

    shock = {driver: {"size": size, "mode": mode}} if mode else {driver: {"size": size}}
    r = E.evaluate(pack, cc, shock)
    d = r["delta_by_driver"].get(driver, [0.0] * len(r["periods"]))
    return {_period_year(cc, p): float(v) for p, v in zip(r["periods"], d)}


def _level_path(pack: dict, cc: str, driver: str, size_pp: float) -> dict[int, float]:
    """Two years of ±size_pp growth: level gap size in the first shock year, 2×size from the second."""
    one = _engine(pack, cc, driver, size_pp, "level")
    two = _engine(pack, cc, driver, 2 * size_pp, "level")
    out = {}
    for y in range(Y0, Y1 + 1):
        src = one if y <= Y0 + 1 else two
        if y in src:
            out[y] = src[y]
        else:  # hold the last year's response, grown with the level
            out[y] = src[max(src)]
    return out


def _gap(y: int, size_pp: float) -> float:
    return 0.0 if y <= Y0 else (size_pp if y == Y0 + 1 else 2 * size_pp) / 100


def _inputs() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    N = json.loads(NUMBERS.read_text())
    pack = json.loads(PACK.read_text())
    a = pd.read_parquet(SC / "derived" / "debt_country_annual.parquet").set_index(["cc", "year"])
    s = pd.read_parquet(SC / "derived" / "debt_snapshot.parquet").set_index("cc")
    return N, pack, a, s


def _book(cc: str, N: dict, a: pd.DataFrame, s: pd.DataFrame) -> dict:
    d0 = N.get(f"d0_{cc}") or float(s.loc[cc, "gg_debt_pct"])
    rm = s.loc[cc, "marginal_rate"]
    if not np.isfinite(rm):
        rm = s.loc[cc, "new_bond_yield"] if np.isfinite(s.loc[cc, "new_bond_yield"]) else s.loc[cc, "y10"]
    share = 1.0 if cc in GG_MEASURE else float(N[f"rev_share_{cc}"]) / 100
    return {"d0": float(d0), "rm": float(rm) / 100, "share": share}


_DEBT_BOOK = None
BOOK_GRID_BP = (-200.0, -100.0, 0.0, 100.0, 200.0)


def debt_book():
    """The ISIN book re-run at a few parallel shifts (calendar years); built once per process."""
    global _DEBT_BOOK
    if _DEBT_BOOK is None:
        from sovereign_cash.scenarios.channels.debt_book import DebtBook
        _DEBT_BOOK = DebtBook(grid_bp=BOOK_GRID_BP)
    return _DEBT_BOOK


def book_rates(cc: str, bp: float, years: list[int]) -> dict[int, float]:
    """Extra interest (local bn) from a parallel ``bp`` shift, by calendar year, straight-line between grid runs."""
    if bp == 0.0:
        return {y: 0.0 for y in years}
    B = debt_book()
    knots = [k for k in B.grid]
    if not min(knots) <= bp <= max(knots):
        raise ValueError(f"{bp}bp is outside the ISIN-book runs {min(knots)}..{max(knots)}bp; extend BOOK_GRID_BP")
    lo = max(k for k in knots if k <= bp)
    hi = min(k for k in knots if k >= bp)
    base = B.interest(cc, 0.0)
    f = lambda k: (B.interest(cc, k) - base).reindex(years).fillna(0.0)  # noqa: E731
    if hi == lo:
        d = f(lo)
    else:
        w = (bp - lo) / (hi - lo)
        d = (1 - w) * f(lo) + w * f(hi)
    return {y: float(d.get(y, 0.0)) for y in years}


def linker_stock(cc: str) -> float:
    """Index-linked debt, local bn: the issuer's stock incl. uplift (US, UK, FR), else the ISIN tape's linker face."""
    from sovereign_cash.debt import build as BLD
    from sovereign_cash.debt import isin_book as IB

    up = BLD._uplift(debt_book().inputs.src, cc)
    if up["il_now"] > 0:
        return float(up["il_now"])
    o = IB.outstanding()
    m = (o["country"].str.lower() == cc) & (o["sec_type"] == "Linker")
    return float(o.loc[m, "face_mn"].sum()) / 1000.0


def _price_gap_path(g_pp: float, infl_pp: float):
    from sovereign_cash.scenarios import cyclical as CY

    return lambda k: (1 - CY.REAL_SHARE) * CY.gap(k, g_pp) + CY._price_gap(k, infl_pp)


def _base_interest(cc: str, N: dict, a: pd.DataFrame, years: list[int], rev: dict) -> dict[int, float]:
    """Debt board GG interest path, scaled log-linearly so interest / revenue hits the reviewed 2026 and 2030 values."""
    I = {y: float(a.loc[(cc, y), "gg_interest_bn"]) for y in years}
    lo, hi = N.get(f"ir_now_{cc}"), N.get(f"ir_end_{cc}")
    if cc == "us" or lo is None or hi is None:  # US: the reviewed ratios are federal; keep the board's GG path
        return I
    f0 = lo / (100 * I[Y0] / rev[Y0])
    f1 = hi / (100 * I[Y1] / rev[Y1])
    return {y: I[y] * f0 ** ((Y1 - y) / (Y1 - Y0)) * f1 ** ((y - Y0) / (Y1 - Y0)) for y in years}


def run_country(cc: str, N: dict, pack: dict, a: pd.DataFrame, s: pd.DataFrame,
                g_pp: float = 0.0, r_bp: float = 0.0, infl_pp: float = 0.0, spend_windfall: float = 0.0) -> dict:
    """Baseline and shocked GG paths 2026→2030; returns 2030 differences."""
    B = _book(cc, N, a, s)
    years = list(range(Y0, Y1 + 1))
    gdp = {y: float(a.loc[(cc, y), "gdp_bn"]) for y in years}
    rev = {y: float(a.loc[(cc, y), "gg_revenue_bn"]) for y in years}
    pd_pct = {y: float(N.get(f"pb_gg_{cc}_{y}") or 0.0) for y in years}  # primary deficit, % GDP (+ = deficit)
    # one model for the primary in every scenario: growth, inflation and rates legs on the same lines
    # (scenarios.cyclical); interest on the ISIN book
    from sovereign_cash.scenarios import cyclical as CY
    cyc = CY.response_all(cc, years, Y0, g_pp=g_pp, infl_pp=infl_pp, r_bp=r_bp)  # GG (US federal), local bn
    rt = book_rates(cc, r_bp, years)
    ri = CY.rates_interest(cc, r_bp, years, Y0)  # UK: Bank Rate on reserves (APF) and NS&I, outside the gilt book
    gap = {y: _gap(y, g_pp + infl_pp) for y in years}
    px = _price_gap_path(g_pp, infl_pp)
    IL = linker_stock(cc) if (g_pp or infl_pp) else 0.0
    Ib = _base_interest(cc, N, a, years, rev)

    D = B["d0"] / 100 * gdp[Y0]
    Ds = D
    out_b, out_s = {}, {}
    for y in years[1:]:
        k = y - Y0
        I = Ib[y]
        PD = pd_pct[y] / 100 * gdp[y]
        D_new = D + I + PD
        # shocked
        d_rev = cyc[y]["d_rev"]  # receipts change (+ = more receipts) is −borrowing
        d_spend = cyc[y]["d_spend"]
        if d_rev > 0 and spend_windfall:  # the windfall is spent, not forgone: receipts stay, spending rises
            d_spend += spend_windfall * d_rev
        if cc == "us":  # state and local: receipts with nominal GDP, spending matches them (balanced budgets)
            sl = rev[y] * (1 - B["share"]) * gap[y]
            d_rev += sl
            d_spend += sl
        dI = (rt[y] + ri[y] + (B["rm"] + r_bp / 1e4) * (Ds - D)  # book, Bank Rate (UK), extra debt
              + IL * (px(k) - px(k - 1)))                       # linker accrual
        Is = I + dI
        PDs = PD - d_rev + d_spend
        Ds_new = Ds + Is + PDs
        G, Gs = gdp[y], gdp[y] * (1 + gap[y])
        R, Rs = rev[y], rev[y] + d_rev
        out_b[y] = {"debt": 100 * D_new / G, "ir": 100 * I / R, "pd": 100 * PD / G}
        out_s[y] = {"debt": 100 * Ds_new / Gs, "ir": 100 * Is / Rs, "pd": 100 * PDs / Gs}
        D, Ds = D_new, Ds_new
    b, z = out_b[Y1], out_s[Y1]
    return {"d_debt": z["debt"] - b["debt"], "d_ir": z["ir"] - b["ir"], "d_pd": z["pd"] - b["pd"],
            "base_debt": b["debt"], "base_ir": b["ir"], "base_pd": b["pd"]}


SCEN = {  # name → (growth pp, rates bp, inflation pp, windfall spent)
    "rates_up": (0, RATES_BP, 0, 0), "rates_dn": (0, -RATES_BP, 0, 0),
    "down": (-GROWTH_PP, 0, 0, 0), "up": (GROWTH_PP, 0, 0, 0),
    "down_ease": (-GROWTH_PP, -RATES_BP, 0, 0), "down_tp": (-GROWTH_PP, RATES_BP, 0, 0),
    "up_hike": (GROWTH_PP, RATES_BP, 0, 0), "up_ease": (GROWTH_PP, -RATES_BP, 0, 0),
    "up_spent": (GROWTH_PP, 0, 0, 0.5),
    "reflate": (0, RATES_BP, GROWTH_PP, 0),
}


def table() -> pd.DataFrame:
    N, pack, a, s = _inputs()
    rows = []
    for cc in CC:
        for name, (g, rb, inf, w) in SCEN.items():
            r = run_country(cc, N, pack, a, s, g_pp=g, r_bp=rb, infl_pp=inf, spend_windfall=w)
            rows.append({"cc": cc, "scenario": name, **r})
    return pd.DataFrame(rows)


def numbers(T: pd.DataFrame | None = None) -> dict:
    T = table() if T is None else T
    out = {}
    for _, r in T.iterrows():
        for k in ("d_debt", "d_ir", "d_pd"):
            out[f"sg_{r.scenario}_{k}_{r.cc}"] = float(r[k])
    for cc in CC:
        dn = out[f"sg_down_d_debt_{cc}"]
        upsp = out[f"sg_up_spent_d_debt_{cc}"]
        out[f"sg_asym_{cc}"] = dn + upsp  # >0: downturn costs more than the (half-banked) upturn gives
    return out


def figure(T: pd.DataFrame | None = None, out: Path | None = None) -> Path:
    T = table() if T is None else T
    out = out or FIG / "ex13_scenario_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [("rates_up", "Rates\n+75bp"), ("down", "Nominal GDP\n−4%"), ("down_tp", "NGDP −4%\nrates +75bp"),
            ("down_ease", "NGDP −4%\nrates −75bp"), ("up_hike", "NGDP +4%\nrates +75bp"),
            ("up_spent", "NGDP +4%,\nhalf spent"), ("reflate", "Inflation +2pp\nrates +75bp")]
    order = ["fr", "us", "it", "jp", "uk", "es", "de"]
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"wspace": 0.08})
    for ax, (k, title) in zip(axes, (("d_debt", "A. Debt / GDP in 2030, pp vs baseline"),
                                     ("d_ir", "B. Interest / revenue in 2030, pp vs baseline"))):
        M = np.array([[float(T[(T.cc == cc) & (T.scenario == c)][k].iloc[0]) for c, _ in cols] for cc in order])
        lim = np.nanmax(np.abs(M))
        ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                lab = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "−")
                ax.text(j, i, lab, ha="center", va="center", fontsize=8.6,
                        color="white" if abs(v) > 0.6 * lim else "#222")
        ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=7.6)
        ax.set_yticks(range(len(order)), [NM[c] for c in order] if ax is axes[0] else [""] * len(order), fontsize=9)
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold", pad=34)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0)
    fig.text(0.01, 0.01,
             "Growth: nominal GDP 2pp a year below/above baseline in 2027–28 (4% by 2028), no catch-up; each budget line on its own driver "
             "(US federal lines, UK OBR reckoner, OECD elasticities elsewhere, corporate tax stressed in a downturn). Prices "
             "(reflation, and a quarter of growth): receipts in the year, 80% of primary spending a year later, linkers accrue. "
             "Rates: parallel, permanent; ISIN book re-run through 2030, marginal rate on extra borrowing, BoJ payment to the "
             "Treasury. 'Half spent': half the upturn's receipts windfall is spent. Red = worse. US is general government.",
             fontsize=6.6, color="#555", wrap=True)
    fig.suptitle("Rates ±75bp and nominal GDP ±4%: change against baseline in 2030", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.12)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


REP = {"rep_rate": (0, 100.0, 0, 0), "rep_infl": (0, 0, GROWTH_PP, 0), "rep": (0, 100.0, GROWTH_PP, 0)}


def repression() -> pd.DataFrame:
    """Financial repression on the same model: +100bp with +2pp inflation for two years, and each leg alone."""
    N, pack, a, s = _inputs()
    rows = []
    for cc in CC:
        for name, (g, rb, inf, w) in REP.items():
            rows.append({"cc": cc, "scenario": name,
                         **run_country(cc, N, pack, a, s, g_pp=g, r_bp=rb, infl_pp=inf, spend_windfall=w)})
    return pd.DataFrame(rows)


def repression_figure(R: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or FIG / "ex7b_repression_grid.png"
    order = ["us", "jp", "it", "fr", "es", "de", "uk"]
    v = lambda sc, cc: float(R[(R.cc == cc) & (R.scenario == sc)]["d_ir"].iloc[0])  # noqa: E731
    rate = np.array([v("rep_rate", c) for c in order])
    tot = np.array([v("rep", c) for c in order])
    infl = tot - rate
    debt = np.array([float(R[(R.cc == c) & (R.scenario == "rep")]["d_debt"].iloc[0]) for c in order])
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.4), gridspec_kw={"wspace": 0.22})
    x = np.arange(len(order))
    a1.bar(x, rate, color="#4a6fa5", label="From the nominal curve (+100bp, ISIN book)")
    a1.bar(x, infl, bottom=np.where(infl >= 0, rate, 0), color="#b33b5c",
           label="From inflation (+2pp: receipts, spending a year later, linkers)")
    for i, t in enumerate(tot):
        a1.text(i, max(rate[i], rate[i] + infl[i], 0) + 0.06, f"{t:+.1f}".replace("-", "−"), ha="center", fontsize=9,
                fontweight="bold")
    a1.axhline(0, color="#333", lw=0.8)
    a1.set_xticks(x, [NM[c] for c in order])
    a1.set_title("A. Interest / revenue in 2030, pp vs baseline", loc="left", fontsize=10, fontweight="bold")
    a1.legend(fontsize=7.6, frameon=False, loc="upper right")
    a2.bar(x, debt, color="#64748b")
    for i, t in enumerate(debt):
        a2.text(i, t - 0.4, f"{t:+.1f}".replace("-", "−"), ha="center", va="top", fontsize=9, color="white",
                fontweight="bold")
    a2.axhline(0, color="#333", lw=0.8)
    a2.set_xticks(x, [NM[c] for c in order])
    a2.set_title("B. Debt / GDP in 2030, pp vs baseline", loc="left", fontsize=10, fontweight="bold")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Inflation +2pp for two years with yields +100bp: change against baseline in 2030 "
                 "interest / revenue and debt / GDP", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Same model as the scenario heatmap: receipts follow prices in the year, 80% of primary spending "
             "a year later, index-linked debt accrues the extra inflation; interest on the ISIN book re-run on the "
             "shifted curve. General government (US incl. states).", fontsize=6.8, color="#555")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.14)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    T = table()
    pd.set_option("display.width", 200)
    for k in ("d_debt", "d_ir", "d_pd"):
        print(k)
        print(T.pivot(index="cc", columns="scenario", values=k).round(2).to_string())
    print(T.groupby("cc")[["base_debt", "base_ir", "base_pd"]].first().round(1))
    R = repression()
    out = numbers(T)
    for _, r in R.iterrows():
        for k in ("d_debt", "d_ir", "d_pd"):
            out[f"sg_{r.scenario}_{k}_{r.cc}"] = float(r[k])
    out.update(us_vs_weo())
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(figure(T), repression_figure(R), OUT_JSON)


def us_vs_weo() -> dict[str, float]:
    """US debt and interest in 2030 on our primary path vs the IMF WEO primary (which the debt board issues on)."""
    from sovereign_cash.scenarios import primary as P

    N, pack, a, s = _inputs()
    w = P.weo_primary_deficit_pct("us")
    N2 = {**N, **{f"pb_gg_us_{y}": float(w[y]) for y in range(Y0 + 1, Y1 + 1)}}
    B = _book("us", N, a, s)
    gdp = {y: float(a.loc[("us", y), "gdp_bn"]) for y in (Y1 - 1, Y1)}
    rev = float(a.loc[("us", Y1), "gg_revenue_bn"])
    # extra debt at end-2029 (bn): sum of the primary gap 2027-29; 2030 interest is charged on it (run_country's convention)
    gap29 = sum((N2[f"pb_gg_us_{y}"] - N[f"pb_gg_us_{y}"]) / 100 * float(a.loc[("us", y), "gdp_bn"])
                for y in range(Y0 + 1, Y1))
    gap30 = gap29 + (N2[f"pb_gg_us_{Y1}"] - N[f"pb_gg_us_{Y1}"]) / 100 * gdp[Y1]
    return {"us_weo_debt_gap_2030": 100 * gap30 / gdp[Y1],
            "us_weo_ir_gap_gg_2030": 100 * B["rm"] * gap29 / rev,
            "us_weo_ir_gap_fed_2030": 100 * B["rm"] * gap29 / (rev * B["share"])}


if __name__ == "__main__":
    main()
