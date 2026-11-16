"""Rate and repression shocks on the peer basis: general-government revenue (US federal).

The repricing book is central government. To compare countries, the central shock is scaled to general
government by the ratio of general-government to central interest in the same year (Länder, local and
social-security debt reprice too), and divided by general-government revenue. The US stays federal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sovereign_cash.debt.note import rebuild_ex7_repression as r7

NOTE = Path(__file__).resolve().parent
SC = NOTE.parents[1]
# debt.build writes sovereign_cash/derived. sovereign_cash/debt/derived is an
# older copy: it has no Spain row, so .at["es"] killed the note (KeyError) on
# every run after ES joined the annual table.
CA = SC / "derived" / "debt_country_annual.parquet"


def repricing_path() -> Path:
    """Parquet ``debt.build`` just wrote, else the legacy debt/derived copy."""
    canonical = SC / "derived" / "debt_repricing.parquet"
    legacy = SC / "debt" / "derived" / "debt_repricing.parquet"
    if canonical.is_file():
        return canonical
    return legacy
Y = 2030
CORE = ("us", "jp", "it", "fr", "uk", "de", "es")
NM = {"us": "US (federal)", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy",
      "jp": "Japan", "es": "Spain"}
CH = (("ch_new", "New fixed (roll + net)", "#1f6fb4"), ("ch_bills", "Bills (3m tenor)", "#e8833a"),
      ("ch_linker", "Linkers", "#8c8c8c"), ("ch_floater", "Floaters", "#c7c7c7"))
GREY = "#666666"


def _base() -> pd.DataFrame:
    a = pd.read_parquet(CA)
    a = a[a["year"] == Y].set_index("cc")
    k = (a["gg_interest_bn"] / a["interest_bn"]).where(a.index != "us", 1.0)
    rev = a["gg_revenue_bn"].where(a.index != "us", a["revenue_bn"])
    i0 = a["gg_interest_bn"].where(a.index != "us", a["interest_bn"])
    return pd.DataFrame({"k": k, "rev": rev, "i0": i0})


def rate_table() -> pd.DataFrame:
    rp = pd.read_parquet(repricing_path())
    rp = rp[(rp["scenario"] == "baseline") & (rp["year"] == Y)]
    cur = rp[rp["case"] == "current"].set_index("cc")
    sh = rp[rp["case"] == "shock"].set_index("cc")
    b = _base()
    rows = []
    for cc in CORE:
        # A country on the annual table but missing from this repricing file
        # used to raise KeyError and fail the whole sovereign_cash job.
        if cc not in b.index or cc not in cur.index or cc not in sh.index:
            continue
        k, rev, i0 = b.at[cc, "k"], b.at[cc, "rev"], b.at[cc, "i0"]
        d = {c: 100 * k * (sh.at[cc, c] - cur.at[cc, c]) / rev for c, _, _ in CH}
        dc = 100 * k * (sh.at[cc, "interest_bn"] - cur.at[cc, "interest_bn"]) / rev
        rows.append({"cc": cc, "ir0": 100 * i0 / rev, "d_pp": dc,
                     "d_pp_central": 100 * (sh.at[cc, "interest_bn"] - cur.at[cc, "interest_bn"]) / cur.at[cc, "revenue_bn"],
                     "d_gdp": 100 * k * (sh.at[cc, "interest_bn"] - cur.at[cc, "interest_bn"]) / cur.at[cc, "gdp_bn"], **d})
    return pd.DataFrame(rows).set_index("cc")


def repression_table() -> pd.DataFrame:
    """Ex7 on general-government revenue: central legs scaled by k, revenue and its inflation lift by GG/central."""
    t = r7.combined_table()
    a = pd.read_parquet(CA)
    a = a[a["year"] == Y].set_index("cc")
    b = _base()
    rows = []
    for cc in t.index:
        if cc not in b.index or cc not in a.index:
            continue
        k = b.at[cc, "k"]
        R0 = b.at[cc, "rev"]
        I0 = b.at[cc, "i0"]
        rc = a.at[cc, "revenue_bn"]
        dR = t.at[cc, "d_revenue_bn"] * (R0 / rc if cc != "us" else 1.0)
        dIr, dIi = k * t.at[cc, "dI_rate_bn"], k * t.at[cc, "dI_infl_bn"]
        ir0 = 100 * I0 / R0
        c_rate = 100 * (I0 + dIr) / R0 - ir0
        comb = 100 * (I0 + dIr + dIi) / (R0 + dR) - ir0
        rows.append({"cc": cc, "ir0": ir0, "d_ir_pp": comb, "c_rate_pp": c_rate, "c_infl_pp": comb - c_rate})
    return pd.DataFrame(rows).set_index("cc")


def _style():
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def rate_chart(out: Path | None = None) -> Path:
    out = out or (NOTE / "dm_debt_outlook_rate_shock_gg.png")
    _style()
    t = rate_table().sort_values("d_pp", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(t))
    for i, cc in enumerate(t.index):
        a1.plot([t.at[cc, "ir0"], t.at[cc, "ir0"] + t.at[cc, "d_pp"]], [i, i], color="#bbbbbb", lw=2, zorder=1)
        a1.scatter(t.at[cc, "ir0"], i, s=46, facecolor="white", edgecolor="#555555", lw=1.6, zorder=2)
        a1.scatter(t.at[cc, "ir0"] + t.at[cc, "d_pp"], i, s=46, color="#c0392b", zorder=3)
        a1.annotate(f"{t.at[cc, 'ir0'] + t.at[cc, 'd_pp']:.1f}", (t.at[cc, "ir0"] + t.at[cc, "d_pp"], i),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color="#333")
    a1.set_yticks(y, [NM[c] for c in t.index], fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel(f"Interest / revenue {Y}, %", fontsize=9)
    a1.set_title("A. Levels: curve (open) and +100bp (filled)", fontsize=9.5, loc="left", fontweight="bold")
    a1.set_xlim(0, float((t["ir0"] + t["d_pp"]).max()) * 1.15)
    left = np.zeros(len(t))
    for c, lab, col in CH:
        v = t[c].to_numpy(float)
        a2.barh(y, v, left=left, color=col, height=0.6, label=lab, edgecolor="white", linewidth=1)
        left += v
    for i, cc in enumerate(t.index):
        a2.annotate(f"+{t.at[cc, 'd_pp']:.1f}pp", (left[i], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#333")
    a2.set_yticks(y, [NM[c] for c in t.index], fontsize=9)
    a2.invert_yaxis()
    a2.set_xlim(0, left.max() * 1.18)
    a2.set_xlabel(f"+100bp: Δ interest / revenue by {Y}, pp", fontsize=9)
    a2.set_title("B. Which part of the book takes the lift", fontsize=9.5, loc="left", fontweight="bold")
    a2.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(f"Rate shock: +100bp on the curve, interest / general-government revenue by {Y} (US federal)",
                 fontsize=10.5, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.01, "Bond-by-bond central book repriced at +100bp, scaled to general government by the ratio "
             "of general-government to central interest\n(Germany ×1.20, France ×1.10, UK ×1.09), over "
             "general-government revenue. US = federal.", fontsize=7.4, color=GREY)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    return out


def repression_chart(out: Path | None = None) -> Path:
    out = out or (NOTE / "dm_debt_outlook_repression_gg.png")
    ir = repression_table()
    r7.chart(ir, out, basis="gg")
    return out
