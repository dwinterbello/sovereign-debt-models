"""QA: comprehensive ours-vs-authority bench (extension of Exhibit A1).

Desk-only — not a published note exhibit. Under ``figures/qa/``:

* history consistency dots (2019, 2025 outturn) against Commission / IMF on the
  same panel as the forecast
* forecast gaps kept first-class: |gap|>1pp without a clean history match is
  ``needs_conviction``, not dismissed as noise
* IMF interest / overall / IR plotted gross-equivalent (WEO net + OECD netting) —
  like-for-like vs our GG gross, never raw net beside gross

    MPLCONFIGDIR=.mplconfig PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.note.qa_authority_bench
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sovereign_cash.debt.note import service_rev as SR

NOTE = Path(__file__).resolve().parent
SC = NOTE.parents[1]
OUT = NOTE / "figures" / "qa"
PANEL = ("de", "jp", "us", "fr", "uk", "es", "it")
NM = SR.NM
Y_FC = (2026, 2027, 2030)
Y_HIST = (2019, 2025)  # paired cells only (gaps() drops unpaired ours/authority)
GAP_SOFT = 1.0  # pp — above this needs a reason
Y_FRAME = Y_HIST + Y_FC

# High-conviction reasons already argued in the note (else flag stays needs_conviction).
CONVICTION: dict[tuple[str, str], str] = {
    # Primary beyond T+3 is WEO panel-wide — 2030 primary gaps vs IMF should be ~0; near-term gaps are cash/share/DTS.
    ("us", "primary_deficit"): "h≤3 DTS bucket path (COLA+volume on SSA/VA/OPM) rebased to FRED; beyond T+3 = WEO",
    ("jp", "interest_gdp"): "residual after OECD gross-up is curve / CAO path vs WEO",
    ("jp", "primary_deficit"): "h≤3 persist_gdp (P-driver would invent a surplus); beyond T+3 = WEO like the panel",
    ("uk", "interest_gdp"): "gilt sell-off since Commission/OBR cut-off; note reads us above consensus on purpose",
}


def _style() -> None:
    SR._style()


def _piv(raw: pd.DataFrame, var: str) -> pd.DataFrame:
    return raw.loc[raw["var"] == var].pivot_table(index="country", columns="year", values="value")


def _get(df: pd.DataFrame, cc: str, y: int) -> float:
    if cc not in df.index or y not in df.columns:
        return float("nan")
    v = df.at[cc, y]
    return float(v) if np.isfinite(v) else float("nan")


def frame() -> pd.DataFrame:
    """Wide frame: ``{metric}_{source}_{year}`` including history years."""
    ca = pd.read_parquet(SC / "derived" / "debt_country_annual.parquet")
    ameco = pd.read_parquet(SC / "data" / "debt" / "ameco.parquet")
    weo = pd.read_parquet(SC / "data" / "debt" / "weo.parquet")
    a_ig, a_bal, a_debt, a_rev = (_piv(ameco, v) for v in
                                  ("interest_gross_pct", "balance_pct", "debt_pct", "revenue_pct"))
    w_ig, w_pri, w_bal, w_debt, w_rev = (_piv(weo, v) for v in
                                         ("interest_pct", "primary_pct", "balance_pct", "debt_pct", "revenue_pct"))
    # AMECO implied primary deficit (+) = −(balance + interest_gross)
    a_pri = -(a_bal + a_ig)

    o_ig = ca.pivot_table(index="cc", columns="year", values="gg_interest_pct_gdp")
    o_pri = ca.pivot_table(index="cc", columns="year", values="primary_gg_pct_gdp")
    o_rev_i = ca.pivot_table(index="cc", columns="year", values="gg_interest_pct_rev")
    # Book GG debt stock by year (WEO in annual, AMECO revenue overlay does not touch debt).
    # Do NOT use debt_snapshot here for history: snap stamps calendar-year WEO (2026 while
    # Y_HIST includes 2025) and invented false |g|>1 vs 2025 Commission/IMF.
    ann = pd.read_parquet(SC / "derived" / "debt_annual.parquet")
    o_debt = ann.pivot_table(index="country", columns="year", values="gg_debt_pct")

    from sovereign_cash.debt.note import debt_paths as DP

    paths = DP.paths()
    base = paths[paths["scenario"] == "baseline"].pivot(index="cc", columns="year", values="debt_pct")

    years = sorted(set(Y_FRAME))
    rows = []
    for cc in PANEL:
        spot, fwd = SR._ours_10y(cc)
        r: dict = {"cc": cc, "auth_10y": SR._auth_long_rate(cc),
                   "ours_10y_spot": spot, "ours_10y_2030": fwd}
        for y in years:
            r[f"ig_ours_{y}"] = _get(o_ig, cc, y)
            r[f"pri_ours_{y}"] = _get(o_pri, cc, y)
            ig, pri = r[f"ig_ours_{y}"], r[f"pri_ours_{y}"]
            r[f"ov_ours_{y}"] = (pri + ig) if np.isfinite(pri) and np.isfinite(ig) else np.nan
            r[f"ir_ours_{y}"] = _get(o_rev_i, cc, y)
            if y in base.columns and cc in base.index and np.isfinite(base.at[cc, y]):
                r[f"debt_ours_{y}"] = float(base.at[cc, y])
            elif y in Y_HIST:
                r[f"debt_ours_{y}"] = _get(o_debt, cc, y)
            else:
                r[f"debt_ours_{y}"] = np.nan

            r[f"ig_ec_{y}"] = _get(a_ig, cc, y)
            r[f"pri_ec_{y}"] = _get(a_pri, cc, y)
            r[f"ov_ec_{y}"] = -_get(a_bal, cc, y)
            r[f"debt_ec_{y}"] = _get(a_debt, cc, y)
            ec_ig, ec_rv = _get(a_ig, cc, y), _get(a_rev, cc, y)
            r[f"ir_ec_{y}"] = (100.0 * ec_ig / ec_rv) if np.isfinite(ec_ig) and ec_rv else np.nan

            r[f"ig_imf_{y}"] = _get(w_ig, cc, y)
            r[f"pri_imf_{y}"] = -_get(w_pri, cc, y)
            r[f"ov_imf_{y}"] = -_get(w_bal, cc, y)
            r[f"debt_imf_{y}"] = _get(w_debt, cc, y)
            wi, wr = _get(w_ig, cc, y), _get(w_rev, cc, y)
            r[f"ir_imf_{y}"] = (100.0 * wi / wr) if np.isfinite(wi) and wr else np.nan
            # Like-for-like gross: IMF publishes net; add OECD gross−net proxy (same as A1).
            net = SR._netting_pp(cc, min(y, 2027))
            r[f"ig_imf_x_{y}"] = (wi + net) if np.isfinite(wi) and np.isfinite(net) else np.nan
            r[f"ov_imf_x_{y}"] = (r[f"ov_imf_{y}"] + net) if np.isfinite(r[f"ov_imf_{y}"]) and np.isfinite(net) else np.nan
            r[f"ir_imf_x_{y}"] = (100.0 * r[f"ig_imf_x_{y}"] / wr) if np.isfinite(r[f"ig_imf_x_{y}"]) and wr else np.nan

        r["ig_ours_2016"] = _get(o_ig, cc, 2016)
        rows.append(r)
    return pd.DataFrame(rows).set_index("cc")


def gaps(b: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long gaps with ``family``, netting adj, history anchor, and ``flag``."""
    b = frame() if b is None else b
    specs: list[tuple] = []
    # history — IMF interest/overall on gross-equivalent (net + OECD netting), never raw net
    for y in Y_HIST:
        specs += [
            ("interest_gdp", y, f"ig_ours_{y}", f"ig_ec_{y}", "Commission", "history"),
            ("interest_gdp", y, f"ig_ours_{y}", f"ig_imf_x_{y}", "IMF", "history"),
            ("primary_deficit", y, f"pri_ours_{y}", f"pri_ec_{y}", "Commission", "history"),
            ("primary_deficit", y, f"pri_ours_{y}", f"pri_imf_{y}", "IMF", "history"),
            ("overall_deficit", y, f"ov_ours_{y}", f"ov_ec_{y}", "Commission", "history"),
            ("overall_deficit", y, f"ov_ours_{y}", f"ov_imf_x_{y}", "IMF", "history"),
            ("debt_gdp", y, f"debt_ours_{y}", f"debt_ec_{y}", "Commission", "history"),
            ("debt_gdp", y, f"debt_ours_{y}", f"debt_imf_{y}", "IMF", "history"),
            ("interest_rev", y, f"ir_ours_{y}", f"ir_ec_{y}", "Commission", "history"),
            ("interest_rev", y, f"ir_ours_{y}", f"ir_imf_x_{y}", "IMF", "history"),
        ]
    # forecast — AMECO ends 2027; IMF interest/IR/overall = gross-equivalent only
    specs += [
        ("interest_gdp", 2026, "ig_ours_2026", "ig_ec_2026", "Commission", "forecast"),
        ("interest_gdp", 2026, "ig_ours_2026", "ig_imf_x_2026", "IMF", "forecast"),
        ("interest_gdp", 2027, "ig_ours_2027", "ig_ec_2027", "Commission", "forecast"),
        ("interest_gdp", 2027, "ig_ours_2027", "ig_imf_x_2027", "IMF", "forecast"),
        ("interest_gdp", 2030, "ig_ours_2030", "ig_imf_x_2030", "IMF", "forecast"),
        ("interest_rev", 2026, "ir_ours_2026", "ir_ec_2026", "Commission", "forecast"),
        ("interest_rev", 2026, "ir_ours_2026", "ir_imf_x_2026", "IMF", "forecast"),
        ("interest_rev", 2027, "ir_ours_2027", "ir_ec_2027", "Commission", "forecast"),
        ("interest_rev", 2027, "ir_ours_2027", "ir_imf_x_2027", "IMF", "forecast"),
        ("interest_rev", 2030, "ir_ours_2030", "ir_imf_x_2030", "IMF", "forecast"),
        ("primary_deficit", 2026, "pri_ours_2026", "pri_ec_2026", "Commission", "forecast"),
        ("primary_deficit", 2026, "pri_ours_2026", "pri_imf_2026", "IMF", "forecast"),
        ("primary_deficit", 2027, "pri_ours_2027", "pri_imf_2027", "IMF", "forecast"),
        ("primary_deficit", 2027, "pri_ours_2027", "pri_ec_2027", "Commission", "forecast"),
        ("primary_deficit", 2030, "pri_ours_2030", "pri_imf_2030", "IMF", "forecast"),
        ("overall_deficit", 2026, "ov_ours_2026", "ov_ec_2026", "Commission", "forecast"),
        ("overall_deficit", 2026, "ov_ours_2026", "ov_imf_x_2026", "IMF", "forecast"),
        ("overall_deficit", 2027, "ov_ours_2027", "ov_ec_2027", "Commission", "forecast"),
        ("overall_deficit", 2027, "ov_ours_2027", "ov_imf_x_2027", "IMF", "forecast"),
        ("overall_deficit", 2030, "ov_ours_2030", "ov_imf_x_2030", "IMF", "forecast"),
        ("debt_gdp", 2026, "debt_ours_2026", "debt_ec_2026", "Commission", "forecast"),
        ("debt_gdp", 2026, "debt_ours_2026", "debt_imf_2026", "IMF", "forecast"),
        ("debt_gdp", 2027, "debt_ours_2027", "debt_ec_2027", "Commission", "forecast"),
        ("debt_gdp", 2027, "debt_ours_2027", "debt_imf_2027", "IMF", "forecast"),
        ("debt_gdp", 2030, "debt_ours_2030", "debt_imf_2030", "IMF", "forecast"),
        ("y10", 2030, "ours_10y_2030", "auth_10y", "OECD/CAO", "rates"),
    ]

    rows = []
    for metric, year, oc, ac, src, family in specs:
        for cc in b.index:
            if oc not in b.columns or ac not in b.columns:
                continue
            o, a = b.at[cc, oc], b.at[cc, ac]
            if not (np.isfinite(o) and np.isfinite(a)):
                continue
            gap = float(o) - float(a)
            # Authority cells ``*_imf_x_*`` are already gross (= net + OECD netting). Record the
            # proxy for the log; do not subtract it again from the gap.
            net = SR._netting_pp(cc, min(year, 2027)) if src == "IMF" and metric in (
                "interest_gdp", "interest_rev", "overall_deficit") else float("nan")
            if year < 2027 and src == "IMF" and np.isfinite(net):
                net_y = SR._netting_pp(cc, year)
                if np.isfinite(net_y):
                    net = net_y
            gap_x = gap if ("_imf_x_" in ac and np.isfinite(gap)) else float("nan")
            rows.append({
                "cc": cc, "name": NM[cc], "metric": metric, "year": int(year),
                "ours": round(float(o), 3), "authority": round(float(a), 3),
                "source": src, "family": family, "gap_pp": round(gap, 3),
                "netting_pp": round(net, 3) if np.isfinite(net) else np.nan,
                "gap_ex_netting_pp": round(gap_x, 3) if np.isfinite(gap_x) else np.nan,
            })
    g = pd.DataFrame(rows)
    return _flag(g)


def _flag(g: pd.DataFrame) -> pd.DataFrame:
    """Attach flag + hist_gap_2019 + conviction_note. Gaps vs IMF interest are already gross."""
    hist = g[(g["family"] == "history") & (g["year"] == 2019)].set_index(
        ["cc", "metric", "source"])["gap_pp"]
    flags, hist_gaps, notes = [], [], []
    for r in g.itertuples():
        key = (r.cc, r.metric, r.source)
        h = float(hist[key]) if key in hist.index else float("nan")
        hist_gaps.append(round(h, 3) if np.isfinite(h) else np.nan)
        note = CONVICTION.get((r.cc, r.metric), "")
        notes.append(note)

        if r.family == "history":
            flags.append("ok" if abs(r.gap_pp) <= GAP_SOFT else "basis_fail")
            continue
        if r.family == "rates":
            flags.append("rates_vintage" if abs(r.gap_pp) > GAP_SOFT else "ok")
            continue

        score = r.gap_pp
        if abs(score) <= GAP_SOFT:
            flags.append("ok")
        elif np.isfinite(h) and abs(h) > GAP_SOFT and np.sign(h) == np.sign(score):
            flags.append("basis_contaminated")
        elif note:
            flags.append("conviction")
        else:
            flags.append("needs_conviction")
    g = g.copy()
    g["hist_gap_2019"] = hist_gaps
    g["flag"] = flags
    g["conviction_note"] = notes
    return g


# Year = colour; source = shape. Same year shares one colour across Ours / Commission / IMF.
# One hue per year — no two oranges.
YEAR_COLOR = {
    2019: "#4a4a4a",   # charcoal
    2025: "#2a9d8f",   # teal
    2026: "#e9c46a",   # gold
    2027: "#e76f51",   # coral
    2030: "#1f6fb4",   # blue
}
# Ours ●, Commission ×, IMF ◇
SHAPE = {"Ours": ("o", True), "Commission": ("x", False), "IMF": ("D", False)}


def _scatter_panel(ax, b: pd.DataFrame, *, title: str, ylab: str,
                   ours: dict[int, str],
                   auth: list[tuple[str, int, str]],
                   annotate_year: int | None = None) -> None:
    """Year→colour, source→shape. ``auth`` rows are ``(col, year, "Commission"|"IMF")``."""
    x = np.arange(len(b))
    # nest points by year so same-year Ours/Commission/IMF sit in one cluster
    years = sorted(set(ours) | {y for _, y, _ in auth})
    n_y = max(len(years), 1)
    width = 0.22
    seen: set[str] = set()
    plotted_cols: list[str] = []

    for i, y in enumerate(years):
        color = YEAR_COLOR.get(y, "#333333")
        base = -0.5 * width * (n_y - 1) + i * width
        slots: list[tuple[str, str, str]] = []
        if y in ours:
            slots.append((ours[y], "Ours", f"Ours {y}"))
        for col, ay, src in auth:
            if ay == y:
                slots.append((col, src, f"{src} {y}"))
        for j, (col, src, label) in enumerate(slots):
            if col not in b.columns:
                continue
            marker, filled = SHAPE.get(src, ("o", True))
            dx = base + (j - 0.5 * (len(slots) - 1)) * 0.055
            lab = label if label not in seen else None
            if marker == "x":
                # unfilled marker: colour must be `color=` (edgecolor is ignored + warns)
                ax.scatter(x + dx, b[col], s=72, marker=marker, zorder=3, linewidth=2.0,
                           color=color, label=lab)
            elif filled:
                ax.scatter(x + dx, b[col], s=70, marker=marker, zorder=3, linewidth=1.6,
                           facecolor=color, edgecolor=color, label=lab)
            else:
                ax.scatter(x + dx, b[col], s=64, marker=marker, zorder=3, linewidth=1.8,
                           facecolor="white", edgecolor=color, label=lab)
            seen.add(label)
            plotted_cols.append(col)
            if annotate_year == y and src == "Ours":
                for k, cc in enumerate(b.index):
                    v = b.at[cc, col]
                    if np.isfinite(v):
                        ax.annotate(f"{v:.1f}", (x[k] + dx, v), xytext=(0, 6),
                                    textcoords="offset points", ha="center",
                                    fontsize=6.5, color=color)

    ax.set_xticks(x, [NM[c] for c in b.index], fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    vals = [b[c].to_numpy(float) for c in plotted_cols if c in b.columns]
    stack = np.concatenate([v[np.isfinite(v)] for v in vals if len(v)]) if vals else np.array([0.0])
    if len(stack):
        lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
        pad = max(0.12 * (hi - lo), 0.35)
        ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.grid(True, color="#eeeeee", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=6.0, ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))


def _rates_panel(ax, b: pd.DataFrame) -> None:
    x = np.arange(len(b))
    w = 0.22
    ax.bar(x - w, b["auth_10y"], width=w * 0.95, color="#d9d9d9", edgecolor="white",
           linewidth=1.5, label="CAO (JP) / OECD EO long rate", zorder=1)
    ax.bar(x, b["ours_10y_spot"], width=w * 0.95, color=YEAR_COLOR[2026], edgecolor="white",
           linewidth=1.5, label="Ours spot 10y", zorder=1)
    ax.bar(x + w, b["ours_10y_2030"], width=w * 0.95, color=YEAR_COLOR[2030], edgecolor="white",
           linewidth=1.5, label="Ours 2030 fwd 10y", zorder=1)
    for i, cc in enumerate(b.index):
        for dx, col in ((-w, "auth_10y"), (0.0, "ours_10y_spot"), (w, "ours_10y_2030")):
            v = b.at[cc, col]
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (x[i] + dx, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.2, color="#333333")
    ax.set_xticks(x, [NM[c] for c in b.index], fontsize=8)
    ax.set_ylabel("%", fontsize=8)
    ymax = np.nanmax(b[["auth_10y", "ours_10y_spot", "ours_10y_2030"]].to_numpy(float))
    ax.set_ylim(0, ymax * 1.22)
    ax.yaxis.grid(True, color="#eeeeee", zorder=0)
    ax.set_title("10-year yield: OECD / Cabinet Office vs our government curve (%)", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=6.2, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))


def figure_hist(out: Path | None = None, b: pd.DataFrame | None = None) -> Path:
    """2019 + 2025: year=colour, source=shape (● Ours / × Commission / ◇ IMF)."""
    out = out or (OUT / "authority_bench_hist.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    _style()
    b = frame() if b is None else b
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 11.2))
    ax = axes.ravel()
    _scatter_panel(ax[0], b, title="Interest / GDP, general government (%)",
                   ylab="% of GDP", annotate_year=2025,
                   ours={2019: "ig_ours_2019", 2025: "ig_ours_2025"},
                   auth=[("ig_ec_2019", 2019, "Commission"), ("ig_ec_2025", 2025, "Commission"),
                         ("ig_imf_x_2019", 2019, "IMF"), ("ig_imf_x_2025", 2025, "IMF")])
    _scatter_panel(ax[1], b, title="Primary deficit — basis must match before forecast",
                   ylab="% of GDP (deficit +)", annotate_year=2025,
                   ours={2019: "pri_ours_2019", 2025: "pri_ours_2025"},
                   auth=[("pri_ec_2019", 2019, "Commission"), ("pri_ec_2025", 2025, "Commission"),
                         ("pri_imf_2019", 2019, "IMF"), ("pri_imf_2025", 2025, "IMF")])
    _scatter_panel(ax[2], b, title="Overall deficit, % of GDP",
                   ylab="% of GDP (deficit +)", annotate_year=2025,
                   ours={2019: "ov_ours_2019", 2025: "ov_ours_2025"},
                   auth=[("ov_ec_2019", 2019, "Commission"), ("ov_ec_2025", 2025, "Commission"),
                         ("ov_imf_x_2019", 2019, "IMF"), ("ov_imf_x_2025", 2025, "IMF")])
    _scatter_panel(ax[3], b, title="Debt / GDP — book WEO stock by year vs Commission / IMF",
                   ylab="% of GDP", annotate_year=2025,
                   ours={2019: "debt_ours_2019", 2025: "debt_ours_2025"},
                   auth=[("debt_ec_2019", 2019, "Commission"), ("debt_ec_2025", 2025, "Commission"),
                         ("debt_imf_2019", 2019, "IMF"), ("debt_imf_2025", 2025, "IMF")])
    _scatter_panel(ax[4], b, title="Interest / revenue (%)",
                   ylab="% of revenue", annotate_year=2025,
                   ours={2019: "ir_ours_2019", 2025: "ir_ours_2025"},
                   auth=[("ir_ec_2019", 2019, "Commission"), ("ir_ec_2025", 2025, "Commission"),
                         ("ir_imf_x_2019", 2019, "IMF"), ("ir_imf_x_2025", 2025, "IMF")])
    ax[5].axis("off")
    ax[5].text(0.02, 0.85,
               "Encoding: colour = year, shape = source\n"
               "  o Ours   x Commission   D IMF\n\n"
               "History panel = like-for-like basis check.\n"
               "• Interest / overall / IR vs IMF = WEO net +\n"
               "  OECD gross−net (same measure as Ours/AMECO).\n"
               "• Interest vs Commission should be ~0 both years.\n"
               "• Primary/overall |g|>1pp = basis_fail — fix the\n"
               "  outturn map before reading 2030 fiscal gaps as a view.\n"
               "• Unpaired cells dropped (e.g. JP AMECO 2025 interest).\n"
               "• Debt ours = annual WEO stock (not snap calendar year).",
               fontsize=8.5, va="top", family="sans-serif", color="#333")

    fig.suptitle("QA history — 2019 / 2025 ours vs Commission / IMF",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.05, hspace=0.38, wspace=0.18)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def figure_forecast(out: Path | None = None, b: pd.DataFrame | None = None) -> Path:
    """Forecast only (2026/27/30) — year=colour, source=shape."""
    out = out or (OUT / "authority_bench_forecast.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    _style()
    b = frame() if b is None else b
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.6))
    ax = axes.ravel()

    _scatter_panel(ax[0], b, title="Interest / GDP, general government (%)",
                   ylab="% of GDP", annotate_year=2030,
                   ours={2026: "ig_ours_2026", 2027: "ig_ours_2027", 2030: "ig_ours_2030"},
                   auth=[("ig_ec_2026", 2026, "Commission"), ("ig_ec_2027", 2027, "Commission"),
                         ("ig_imf_x_2026", 2026, "IMF"), ("ig_imf_x_2027", 2027, "IMF"),
                         ("ig_imf_x_2030", 2030, "IMF")])
    _scatter_panel(ax[1], b, title="Primary deficit, % of GDP",
                   ylab="% of GDP (deficit +)", annotate_year=2030,
                   ours={2026: "pri_ours_2026", 2027: "pri_ours_2027", 2030: "pri_ours_2030"},
                   auth=[("pri_ec_2026", 2026, "Commission"), ("pri_ec_2027", 2027, "Commission"),
                         ("pri_imf_2026", 2026, "IMF"), ("pri_imf_2027", 2027, "IMF"),
                         ("pri_imf_2030", 2030, "IMF")])
    _scatter_panel(ax[2], b, title="Overall deficit, % of GDP",
                   ylab="% of GDP (deficit +)", annotate_year=2030,
                   ours={2026: "ov_ours_2026", 2027: "ov_ours_2027", 2030: "ov_ours_2030"},
                   auth=[("ov_ec_2026", 2026, "Commission"), ("ov_ec_2027", 2027, "Commission"),
                         ("ov_imf_x_2026", 2026, "IMF"), ("ov_imf_x_2027", 2027, "IMF"),
                         ("ov_imf_x_2030", 2030, "IMF")])
    _scatter_panel(ax[3], b, title="Debt / GDP (%)",
                   ylab="% of GDP", annotate_year=2030,
                   ours={2026: "debt_ours_2026", 2027: "debt_ours_2027", 2030: "debt_ours_2030"},
                   auth=[("debt_ec_2026", 2026, "Commission"), ("debt_ec_2027", 2027, "Commission"),
                         ("debt_imf_2026", 2026, "IMF"), ("debt_imf_2027", 2027, "IMF"),
                         ("debt_imf_2030", 2030, "IMF")])
    _scatter_panel(ax[4], b, title="Interest / revenue (%)",
                   ylab="% of revenue", annotate_year=2030,
                   ours={2026: "ir_ours_2026", 2027: "ir_ours_2027", 2030: "ir_ours_2030"},
                   auth=[("ir_ec_2026", 2026, "Commission"), ("ir_ec_2027", 2027, "Commission"),
                         ("ir_imf_x_2026", 2026, "IMF"), ("ir_imf_x_2027", 2027, "IMF"),
                         ("ir_imf_x_2030", 2030, "IMF")])
    _rates_panel(ax[5], b)

    fig.suptitle("Forecasts compared: ours (circles), European Commission (crosses), IMF (diamonds); colour = year",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.text(
        0.01, 0.006,
        "General government. IMF interest, overall balance and interest / revenue moved to a gross-interest basis with "
        "the OECD gross-net interest gap, for comparability with the Commission and our interest (gross). "
        "Sources: our model; European Commission AMECO (autumn 2026); IMF WEO (October 2026); OECD; Cabinet Office.",
        fontsize=6.2, color="#666666",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.05, hspace=0.36, wspace=0.18)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def figure(out: Path | None = None, b: pd.DataFrame | None = None) -> Path:
    """Backward-compatible alias: write the forecast panel path used by older callers."""
    b = frame() if b is None else b
    figure_hist(OUT / "authority_bench_hist.png", b)
    return figure_forecast(out or (OUT / "authority_bench_forecast.png"), b)


def _heatmap(g: pd.DataFrame, *, value_col: str, title: str, out: Path,
             vmax: float | None = None, metrics: list[str] | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _style()
    g = g.copy()
    if metrics is not None:
        g = g[g["metric"].isin(metrics)]
    g["cell"] = g["metric"] + " " + g["year"].astype(str) + "\n" + g["source"]
    order = []
    for metric in (metrics or list(dict.fromkeys(g["metric"]))):
        for year in sorted(g["year"].unique()):
            for src in ("Commission", "IMF", "OECD/CAO"):
                lab = f"{metric} {year}\n{src}"
                if lab in set(g["cell"]):
                    order.append(lab)
    mat = g.pivot_table(index="name", columns="cell", values=value_col)
    mat = mat.reindex(index=[NM[c] for c in PANEL if NM[c] in mat.index], columns=order)
    fig_w = max(7.0, 0.48 * max(len(order), 1) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    data = np.ma.masked_invalid(mat.to_numpy(float))
    filled = data.filled(np.nan)
    if vmax is None:
        vmax = np.nanmax(np.abs(filled)) if np.isfinite(filled).any() else 1.0
        vmax = max(float(vmax), 0.5)
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=8)
    ax.set_xticks(range(len(mat.columns)), mat.columns, fontsize=5.8, rotation=90)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.6,
                        color="#111" if abs(v) < 0.55 * vmax else "#fff")
    ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="pp")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.30)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def gaps_figures(g: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    hist = g[g["family"] == "history"]
    fc = g[g["family"].isin(("forecast", "definition", "rates"))]
    # IMF interest ex-netting overlay on like-for-like forecast interest
    imf_x = g[(g["metric"] == "interest_gdp") & (g["source"] == "IMF") & (g["family"] == "forecast")].copy()
    imf_x = imf_x.dropna(subset=["gap_ex_netting_pp"])
    imf_x["source"] = "IMF ex-netting"
    imf_x["gap_pp"] = imf_x["gap_ex_netting_pp"]
    like_fc = pd.concat([
        fc[fc["metric"].isin(("interest_gdp", "interest_rev")) & (fc["source"] == "Commission")],
        fc[(fc["metric"] == "interest_gdp") & (fc["source"] == "IMF")],
        imf_x,
    ], ignore_index=True)
    return {
        "gaps_hist": _heatmap(
            hist, value_col="gap_pp", out=out_dir / "authority_gaps_hist.png", vmax=2.5,
            metrics=["interest_gdp", "primary_deficit", "overall_deficit", "debt_gdp", "interest_rev"],
            title="History consistency (pp) — 2019 / 2025 ours vs authority  "
                  "(|g|>1 on primary = basis_fail; interest vs IMF use netting)",
        ),
        "gaps_like": _heatmap(
            like_fc, value_col="gap_pp", out=out_dir / "authority_gaps_like.png", vmax=2.0,
            metrics=["interest_gdp", "interest_rev"],
            title="Forecast interest gaps (pp) — Commission gross + IMF raw/ex-netting (±2)",
        ),
        "gaps_fiscal": _heatmap(
            fc[fc["metric"].isin(("primary_deficit", "overall_deficit", "debt_gdp"))],
            value_col="gap_pp", out=out_dir / "authority_gaps_fiscal.png", vmax=None,
            metrics=["primary_deficit", "overall_deficit", "debt_gdp"],
            title="Forecast fiscal-path gaps (pp) — |g|>1 needs conviction or is basis_contaminated "
                  "(see authority_flags.csv)",
        ),
        "gaps": _heatmap(
            g[g["family"].isin(("history", "forecast"))],
            value_col="gap_pp", out=out_dir / "authority_gaps.png", vmax=None,
            title="All history+forecast gaps (pp) — read flags, not colours alone",
        ),
    }


def build(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    b = frame()
    g = gaps(b)
    csv = out_dir / "authority_gaps.csv"
    g.to_csv(csv, index=False)
    flags = g[g["flag"].isin(("needs_conviction", "basis_fail", "basis_contaminated", "conviction"))]
    flags_path = out_dir / "authority_flags.csv"
    flags.sort_values(["flag", "metric", "year", "cc"]).to_csv(flags_path, index=False)
    hist = figure_hist(out_dir / "authority_bench_hist.png", b)
    fc = figure_forecast(out_dir / "authority_bench_forecast.png", b)
    # keep a stable name pointing at the forecast panel
    bench = out_dir / "authority_bench.png"
    bench.write_bytes(fc.read_bytes())
    out = {"bench_hist": hist, "bench_forecast": fc, "bench": bench,
           "csv": csv, "flags": flags_path}
    out.update(gaps_figures(g, out_dir))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    paths = build(a.out)
    for k, p in paths.items():
        print(f"{k}: {p} ({p.stat().st_size / 1024:.0f} KB)")
    g = pd.read_csv(paths["csv"])
    print("\nHistory |g|>1 (basis check):")
    for y in Y_HIST:
        hist = g[(g.family == "history") & (g.year == y)]
        bad = hist[hist.gap_pp.abs() > GAP_SOFT]
        print(f"  [{y}]")
        if bad.empty:
            print("    none")
            continue
        for r in bad.itertuples():
            extra = f"  ex-netting {r.gap_ex_netting_pp:+.1f}" if pd.notna(r.gap_ex_netting_pp) else ""
            print(f"    {r.flag:18s} {r.name:8s} {r.metric:16s} {r.source:12s} {r.gap_pp:+.1f}{extra}")
    print("\nForecast flags (fiscal gaps matter):")
    for flag in ("needs_conviction", "basis_contaminated", "conviction", "basis_fail"):
        sub = g[(g.family == "forecast") & (g.flag == flag)]
        if sub.empty:
            continue
        print(f"  [{flag}] n={len(sub)}")
        for r in sub.sort_values("gap_pp", key=lambda s: s.abs(), ascending=False).head(12).itertuples():
            note = f" — {r.conviction_note}" if isinstance(r.conviction_note, str) and r.conviction_note else ""
            print(f"    {r.name:8s} {r.metric:16s} {r.year} {r.source:12s} {r.gap_pp:+.1f} "
                  f"(hist2019 {r.hist_gap_2019}){note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
