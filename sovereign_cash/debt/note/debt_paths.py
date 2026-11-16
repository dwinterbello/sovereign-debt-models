"""Exhibits 5 and 9: general-government debt / GDP paths on the sustain identity, d' = d(1 + r)/(1 + g) − pb.

Inputs are ``debt.dalio_note.sustain``'s per-country table: d0 (IMF GG debt, origin year), g (our GDP path
2026-29), r0 (GG interest / debt), rm (market rate on new issuance), rho (repricing speed), and the primary
engine's pb now / plan (``scenarios.primary``: cash/share/DTS through T+3, IMF WEO primary beyond).
Scenario variants shift r_m and g around that primary baseline — ready reckoners, not a second primary model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTE = Path(__file__).resolve().parent
FIG = NOTE / "figures"
SC = NOTE.parents[1]
Y0 = 2026
CC6 = ("us", "uk", "de", "fr", "it", "jp")
NM = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy", "jp": "Japan"}
ML = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#A2142F"]
PAL = dict(zip(CC6, ML))
# name → (Δ market rate bp, Δ nominal growth pp, which primary)
SCENARIOS = {
    "baseline": (0.0, 0.0, "pb_now"),
    "plan": (0.0, 0.0, "pb_plan"),
    "soft": (-50.0, 0.5, "pb_now"),
    "political": (100.0, -0.5, "pb_now"),
    "stagflation": (150.0, -1.0, "pb_now"),
}
LABEL = {
    "baseline": "baseline — engine primary (cash/share/DTS → IMF WEO)",
    "plan": "2030 IMF WEO primary held (not agency plan)",
    "soft": "soft (−50bp, +0.5pp g)",
    "political": "political failure (+100bp, −0.5pp g)",
    "stagflation": "stagflation (+150bp, −1pp g)",
}


def _r0(cc: str) -> float:
    a = pd.read_parquet(SC / "derived" / "debt_country_annual.parquet").set_index(["cc", "year"])
    s = pd.read_parquet(SC / "derived" / "debt_snapshot.parquet").set_index("cc")
    return float(a.loc[(cc, Y0), "gg_interest_pct_gdp"]) / float(s.loc[cc, "gg_debt_pct"])


def sim(T: pd.DataFrame, cc: str, shock_bp: float, dg_pp: float, pb: float, horizon: int = 10) -> list[float]:
    d, r = float(T.loc[cc, "d0"]), _r0(cc)
    g = float(T.loc[cc, "g"]) / 100 + dg_pp / 100
    rm, rho = float(T.loc[cc, "rm"]) / 100, float(T.loc[cc, "rho"])
    out = [d]
    for _ in range(horizon):
        r += rho * (rm + shock_bp / 1e4 - r)
        d = d * (1 + r) / (1 + g) - pb
        out.append(d)
    return out


def paths(sustain: dict | None = None, horizon: int = 10) -> pd.DataFrame:
    """Long table: cc, scenario, year, debt_pct."""
    from sovereign_cash.debt.dalio_note import sustain as _sustain
    T = (sustain or _sustain(horizon))["T"]
    rows = []
    for cc in CC6 + ("es",):
        if cc not in T.index:
            continue
        for name, (sh, dg, which) in SCENARIOS.items():
            p = sim(T, cc, sh, dg, float(T.loc[cc, which]), horizon)
            rows += [{"cc": cc, "scenario": name, "year": Y0 + i, "debt_pct": v} for i, v in enumerate(p)]
    return pd.DataFrame(rows)


def endpoints(P: pd.DataFrame) -> pd.DataFrame:
    """debt/GDP at the origin and the end year by cc × scenario."""
    y1 = int(P["year"].max())
    w = P.pivot_table(index="cc", columns="scenario", values="debt_pct", aggfunc="last")
    w0 = P[P["year"] == Y0].groupby("cc")["debt_pct"].first()
    w.insert(0, "d0", w0)
    w.attrs["y_end"] = y1
    return w


def _style():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans"],
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.55,
        "axes.edgecolor": "#262626", "axes.linewidth": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _stagger_ends(ends: list[tuple[float, str, str]], gap: float) -> list[tuple[float, float, str, str]]:
    """ends: (y_data, text, color) → (y_data, y_label, text, color) with non-overlapping labels."""
    ends = sorted(ends, key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    out = []
    for (y0, txt, c), y in zip(ends, ys):
        out.append((y0, y, txt, c))
    return out


def figure_ex5(P: pd.DataFrame, out: Path | None = None) -> Path:
    """Baseline (solid) and 2030-WEO-primary (dotted) debt/GDP paths, all six books."""
    out = out or (FIG / "ex5_debt_paths.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    _style()
    y1 = int(P["year"].max())
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_title(
        f"Debt / GDP, general government, {Y0} → {y1}",
        fontsize=10.5, loc="left", fontweight="bold", pad=8,
    )
    ends = []
    for cc in CC6:
        b = P[(P.cc == cc) & (P.scenario == "baseline")].set_index("year")["debt_pct"]
        p = P[(P.cc == cc) & (P.scenario == "plan")].set_index("year")["debt_pct"]
        if b.empty:
            continue
        ax.plot(b.index, b.values, color=PAL[cc], lw=2.05, solid_capstyle="round", label=NM[cc], zorder=3)
        ax.plot(p.index, p.values, color=PAL[cc], lw=1.35, ls=(0, (1.15, 1.05)), solid_capstyle="round", zorder=2)
        ends.append((float(b.iloc[-1]),
                     f"{b.iloc[0]:.0f}\u2192{b.iloc[-1]:.0f} (WEO {p.iloc[-1]:.0f})", PAL[cc]))
    placed = _stagger_ends(ends, gap=7.5)
    for y0, y, txt, c in placed:
        if abs(y - y0) > 1.5:
            ax.plot([y1, y1], [y0, y], color=c, lw=0.45, alpha=0.5, zorder=1)
        ax.annotate(txt, (y1, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.0, color=c)
    ax.set_xlim(Y0 - 0.3, y1 + 3.2)
    all_y = P[P.scenario.isin(("baseline", "plan")) & P.cc.isin(CC6)]["debt_pct"]
    lo = float(all_y.min()) - 6
    hi = max(float(all_y.max()), max(e[1] for e in placed)) + 8
    ax.set_ylim(lo, hi)
    ax.set_ylabel("% of GDP, general government", fontsize=9)
    ax.tick_params(labelsize=8)
    # Country colours + style key (solid = engine path; dotted = 2030 WEO primary held).
    from matplotlib.lines import Line2D
    country_handles = [Line2D([0], [0], color=PAL[c], lw=2.0, label=NM[c]) for c in CC6]
    style_handles = [
        Line2D([0], [0], color="#333", lw=2.0, label="Solid: engine primary (cash/share/DTS \u2192 IMF WEO)"),
        Line2D([0], [0], color="#333", lw=1.4, ls=(0, (1.15, 1.05)),
               label="Dotted: 2030 IMF WEO primary held (not OBR/CBO/AFT)"),
    ]
    leg1 = ax.legend(handles=country_handles, fontsize=8.0, ncol=6, loc="upper center",
                     bbox_to_anchor=(0.5, -0.06), columnspacing=1.2, handlelength=1.6)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=7.6, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), columnspacing=1.4, handlelength=2.4)
    import textwrap
    fig.text(
        0.01, 0.012,
        textwrap.fill(
            "Scenario arithmetic on the engine primary — not a scored path probability "
            "(recursion ships in DE/JP/ES only). Neither line is a national agency plan "
            "(OBR / CBO / AFT / CAO): solid = evolving engine primary (cash/share/DTS through T+3, "
            "IMF WEO primary beyond); dotted = hold the 2030 IMF WEO primary constant. "
            "Effective rate reprices toward the market rate at the book's own speed; "
            "g = our nominal GDP 2026–29.",
            width=128,
        ),
        fontsize=6.7, color="#555",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    return out


def figure_ex9(P: pd.DataFrame, ccs=("us", "fr"), out: Path | None = None) -> Path:
    """Political-failure and stagflation paths against baseline for the two flow cases."""
    out = out or (FIG / "ex9_political_paths.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    _style()
    y1 = int(P["year"].max())
    fig, axes = plt.subplots(1, len(ccs), figsize=(11.0, 4.4), sharey=False)
    styles = (
        ("baseline", "-", 2.1),
        ("soft", (0, (4.0, 1.6)), 1.35),
        ("political", (0, (1.2, 0.9, 3.2, 0.9)), 1.55),
        ("stagflation", (0, (1.1, 1.1)), 1.55),
    )
    for ax, cc in zip(np.atleast_1d(axes), ccs):
        ends = []
        for name, ls, lw in styles:
            s = P[(P.cc == cc) & (P.scenario == name)].set_index("year")["debt_pct"]
            ax.plot(s.index, s.values, color=PAL[cc], ls=ls, lw=lw, label=LABEL[name],
                    solid_capstyle="round", zorder=3)
            ends.append((float(s.iloc[-1]), f"{s.iloc[-1]:.0f}", "#222"))
        placed = _stagger_ends(ends, gap=5.5)
        for y0, y, txt, c in placed:
            if abs(y - y0) > 1.0:
                ax.plot([y1, y1], [y0, y], color=PAL[cc], lw=0.4, alpha=0.45, zorder=1)
            ax.annotate(txt, (y1, y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.0, color=c)
        ax.set_title(NM[cc], fontsize=11, loc="left", fontweight="bold", pad=6)
        ax.set_xlim(Y0 - 0.3, y1 + 1.8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.2, loc="upper left", handlelength=2.4)
    np.atleast_1d(axes)[0].set_ylabel("% of GDP, general government", fontsize=9)
    fig.text(
        0.01, 0.01,
        "Scenario arithmetic on the engine primary — not a scored path probability; US and France outside the "
        "recursion ship set (DE, JP, ES).",
        fontsize=7.0, color="#555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    return out


def main() -> None:
    P = paths()
    print(endpoints(P).round(0).to_string())
    print(figure_ex5(P), figure_ex9(P))


if __name__ == "__main__":
    main()
