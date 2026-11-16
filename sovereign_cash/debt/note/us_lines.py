"""Exhibit 15: the US federal cash deficit line by line, from the framework.

Model      the DTS bucket path as run (``us_dts.fy_table(adjusted=False)``).
Baseline   the same path with FY2026 one-offs out (``scenarios.channels.us_dts_adjust``).
Recession  each line on its driver (``scenarios.cyclical``, canonical -2pp x 2 years); + Fed cuts: ISIN-book interest
           at a -100bp average curve shift (scenario pack ``us_rates_book``); + stimulus: ``cyclical.STIMULUS_PCT_GDP``.

Basis: DTS cash, federal, fiscal years; interest is DTS coupon interest. Writes figures/ex15_us_lines.png,
us_lines.json and us_lines_assumptions.md.

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.note.us_lines
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NOTE = Path(__file__).resolve().parent
SC = NOTE.parents[1]
FIG = NOTE / "figures"
OUT_JSON = NOTE / "us_lines.json"
OUT_MD = NOTE / "us_lines_assumptions.md"
YEARS = [2026, 2027, 2028, 2029]
Y0 = 2026
FED_SHIFT_BP = -100.0


def _gdp() -> pd.Series:
    a = pd.read_parquet(SC / "derived" / "debt_country_annual.parquet").set_index(["cc", "year"])
    return pd.Series({y: float(a.loc[("us", y), "gdp_bn"]) for y in YEARS})


def _totals(T: pd.DataFrame, gdp: pd.Series, extra: dict | None = None, shift: dict | None = None) -> dict:
    deps = [c for c in T.columns if c.startswith("dep_") and c != "dep_total"]
    wdp = [c for c in T.columns if c.startswith("wd_") and c not in ("wd_total", "wd_interest")]
    out = {"deficit_pct": [], "primary_pct": []}
    for y in YEARS:
        r, p, i = float(T.loc[y, deps].sum()), float(T.loc[y, wdp].sum()), float(T.at[y, "wd_interest"])
        e = extra.get(y, {}) if extra else {}
        r += e.get("rec", 0.0)
        p += e.get("spend", 0.0)
        i += e.get("int", 0.0)
        g = float(gdp[y]) * (1 + (shift or {}).get(y, 0.0))
        out["deficit_pct"].append(100 * (p + i - r) / g)
        out["primary_pct"].append(100 * (p - r) / g)
    return out


def run() -> dict:
    from sovereign_cash.scenarios import cyclical as C
    from sovereign_cash.scenarios.channels import us_dts as U
    from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ

    raw, adj = U.fy_table(adjusted=False), U.fy_table()
    gdp = _gdp()
    L = C.lines("us", -C.CANON_PP, YEARS[1:], Y0)
    Ls = C.lines("us", -C.CANON_PP, YEARS[1:], Y0, stimulus=True)
    shift = {y: C.gap(y - Y0, -C.CANON_PP) for y in YEARS}

    def ex(Lx, fed=False):
        out = {}
        pack = json.loads((SC / "derived" / "scenario_pack.json").read_text())
        ch = next(c for c in pack["countries"]["us"]["channels"] if c["id"] == "us_rates_book")
        k = [float(x) for x in ch["knots"]].index(FED_SHIFT_BP)
        for i, y in enumerate(YEARS[1:]):
            d = Lx.get(y, {})
            out[y] = {"rec": sum(v for kk, v in d.items() if C.is_receipt(kk)),
                      "spend": sum(v for kk, v in d.items() if not C.is_receipt(kk)),
                      "int": float(ch["values"][k][i]) if fed else 0.0}
        return out

    tot = {"model": _totals(raw, gdp), "baseline": _totals(adj, gdp),
           "recession": _totals(adj, gdp, ex(L), shift),
           "recession_fed": _totals(adj, gdp, ex(L, fed=True), shift),
           "recession_fed_stimulus": _totals(adj, gdp, ex(Ls, fed=True), shift)}
    g28 = float(gdp[2028]) * (1 + shift[2028])
    fed28 = ex(Ls, fed=True)[2028]["int"]
    contrib = {k: (-v if C.is_receipt(k) else v) * 100 / g28 for k, v in Ls[2028].items()}
    contrib["wd_interest"] = fed28 * 100 / g28
    return {"basis": "DTS cash, federal, fiscal years; interest = DTS coupon interest", "years": YEARS,
            "totals": tot, "adjustments": ADJ.log(), "recession_contrib_fy28_pct": contrib}


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sovereign_cash.scenarios import cyclical as C

    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    sty = [("model", "DTS model as run", "#9ca3af", "--"), ("baseline", "Baseline, FY26 one-offs out", "#111827", "-"),
           ("recession", "Recession", "#dc2626", "-"), ("recession_fed", "+ Fed cuts", "#f59e0b", "-"),
           ("recession_fed_stimulus", "+ Fed cuts + stimulus", "#7c3aed", "-")]
    for k, lab, c, ls in sty:
        a1.plot(YEARS, res["totals"][k]["deficit_pct"], color=c, ls=ls, lw=1.5 if k == "model" else 2, marker="o",
                ms=3.5, label=lab)
    a1.set_xticks(YEARS, [f"FY{str(y)[2:]}" for y in YEARS])
    a1.set_ylabel("% of GDP")
    a1.set_title("A. Federal cash deficit, % of GDP", loc="left", fontsize=10, fontweight="bold")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.grid(axis="y", color="#e5e7eb", lw=0.6)
    c = res["recession_contrib_fy28_pct"]
    groups = [("Non-withheld tax (equities)", ["dep_nonwithheld"]), ("Corporate tax", ["dep_corporate"]),
              ("Withheld tax (wages)", ["dep_withheld"]), ("Customs, other receipts", ["dep_customs", "dep_other_tax", "dep_other"]),
              ("Refunds", ["wd_refunds"]), ("UI, SNAP, Medicaid", ["wd_unemployment", "wd_snap", "wd_medicaid"]),
              ("Social Security, COLA lines", ["wd_ssa", "wd_va", "wd_opm"]), ("Stimulus", ["stimulus"]),
              ("Interest (Fed cuts)", ["wd_interest"])]
    vals = [sum(c.get(x, 0.0) for x in xs) for _, xs in groups]
    yy = np.arange(len(vals))[::-1]
    a2.barh(yy, vals, color=["#dc2626" if v > 0 else "#059669" for v in vals], height=0.62)
    for yi, v in zip(yy, vals):
        a2.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8)
    a2.set_yticks(yy, [g for g, _ in groups])
    a2.axvline(0, color="#111827", lw=0.8)
    a2.set_xlim(min(vals) - 0.3, max(vals) + 0.35)
    a2.set_title(f"B. FY28, recession + Fed cuts + stimulus vs baseline, pp of GDP (net {sum(vals):+.1f})", loc="left",
                 fontsize=10, fontweight="bold")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, -0.02, "Source: Daily Treasury Statement buckets and the us.fiscal path; scenarios.channels.us_dts_adjust "
             "(baseline) and scenarios.cyclical (recession: nominal GDP 2/4/4% below baseline FY27-29, unemployment "
             f"+{C.OKUN_US * C.REAL_SHARE * 4:.1f}pp, S&P 500 -30% peak to trough, profits -20%, AI capex -30%). Federal cash "
             "basis; interest is DTS coupon interest.", fontsize=7, color="#4b5563", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_md(res: dict, path: Path) -> None:
    names = {"model": "DTS model as run", "baseline": "Baseline (FY26 one-offs out)", "recession": "Recession",
             "recession_fed": "Recession + Fed cuts", "recession_fed_stimulus": "Recession + Fed cuts + stimulus"}
    L = ["# US federal deficit, line by line", "", f"Basis: {res['basis']}. Built from `scenarios.channels.us_dts_adjust`"
         " and `scenarios.cyclical`; their docstrings and named constants carry every assumption and source.", "",
         "| Deficit, % of GDP | " + " | ".join(f"FY{y}" for y in YEARS) + " |", "|---|" + "---:|" * len(YEARS)]
    for k, n in names.items():
        L.append(f"| {n} | " + " | ".join(f"{v:.1f}" for v in res["totals"][k]["deficit_pct"]) + " |")
    L += ["", "| Primary, % of GDP | " + " | ".join(f"FY{y}" for y in YEARS) + " |", "|---|" + "---:|" * len(YEARS)]
    for k, n in names.items():
        L.append(f"| {n} | " + " | ".join(f"{v:.1f}" for v in res["totals"][k]["primary_pct"]) + " |")
    L += ["", "## Baseline adjustments ($bn by forecast FY: model -> adjusted)", ""]
    for a in res["adjustments"]:
        L.append(f"- **{a['line']}**: {' / '.join(f'{m:.0f}->{v:.0f}' for m, v in zip(a['model'], a['adjusted']))}. "
                 f"{a['why']}")
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    res = run()
    OUT_JSON.write_text(json.dumps(res, indent=1))
    figure(res, FIG / "ex15_us_lines.png")
    write_md(res, OUT_MD)
    for k, v in res["totals"].items():
        print(f"{k:24s} deficit% {[round(x, 2) for x in v['deficit_pct']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
