"""Exhibit 2: 2×2 of debt service and interest / revenue — central (issuer) and general government
(US federal on both GG panels). Exhibit A1 lives in ``qa_authority_bench.figure_forecast`` (multi-panel
ours vs Commission / IMF); ``benchmark()`` still writes the legacy Bloomberg-layout cross-check.

Peer ranking uses general government, as the IMF, the Commission and the rating agencies do: central governments
keep very different shares of general-government revenue (UK ~100%, Germany ~28%), so a central ratio is not
comparable across countries.

Debt service is interest plus principal due (bills once, coupons inside the year).
Sweden's coupon-book stock goes negative after 2026, which left one forecast point.
The forecast uses the authority interest path and rolls the last actual bill stock
plus the bonds that mature. History is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

NOTE = Path(__file__).resolve().parent
CA = Path(__file__).resolve().parents[1] / "derived" / "debt_country_annual.parquet"

Y0, Y1 = 2015, 2030
PANEL = ("us", "uk", "de", "fr", "it", "es", "jp", "se")
NM = {
    "us": "US", "uk": "UK", "de": "Germany", "fr": "France",
    "it": "Italy", "es": "Spain", "jp": "Japan", "se": "Sweden",
}
PAL = {
    "us": "#1f77b4", "uk": "#d62728", "de": "#2ca02c", "fr": "#9467bd",
    "it": "#ff7f0e", "es": "#8c564b", "jp": "#e377c2", "se": "#7f7f7f",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333", "xtick.color": "#333", "ytick.color": "#333",
        "text.color": "#222", "axes.labelcolor": "#222",
    })


def _sweden_forecast(ca: pd.DataFrame) -> pd.DataFrame:
    """Replace Sweden's negative-stock forecast. History rows are left as stored."""
    from sovereign_cash import investor as iv

    d = ca[ca["cc"] == "se"].sort_values("year").copy()
    hist = ~d["forecast"].astype(bool)
    d.loc[~hist, "interest_bn"] = d.loc[~hist, "interest_authority_bn"]
    d["interest_pct_rev"] = 100.0 * d["interest_bn"] / d["revenue_bn"]
    last = d.loc[hist, "rollovers_bills_bn"].dropna()
    bills = float(last.iloc[-1]) if len(last) else 0.0
    se = iv.monthly("se").set_index("month").sort_index()
    for i, r in d.loc[~hist].iterrows():
        y = int(r["year"])
        bonds = float(se.loc[se.index.year == y, "bond_redemption_bn"].sum())
        interest = float(r["interest_authority_bn"])
        d.at[i, "debt_service_pct_rev"] = 100.0 * (interest + bills + bonds) / float(r["revenue_bn"])
    return d


def _panel_frame(ca: pd.DataFrame) -> pd.DataFrame:
    se = _sweden_forecast(ca)
    return pd.concat([ca[ca["cc"] != "se"], se], ignore_index=True)


def _series(ca: pd.DataFrame, cc: str, col: str) -> pd.DataFrame:
    d = ca[(ca["cc"] == cc) & (ca["year"] >= Y0) & (ca["year"] <= Y1)].sort_values("year")
    return d.dropna(subset=[col])


def _plot_hist_fc(ax, years, vals, forecast, color, label=None) -> None:
    years = np.asarray(years)
    vals = np.asarray(vals, float)
    forecast = np.asarray(forecast, bool)
    if (~forecast).any():
        ax.plot(years[~forecast], vals[~forecast], color=color, lw=1.7, label=label)
    if forecast.any():
        h = np.where(~forecast)[0]
        f = np.where(forecast)[0]
        if len(h) and len(f):
            ax.plot([years[h[-1]], *years[f]], [vals[h[-1]], *vals[f]], color=color, lw=1.7, ls=":")
        else:
            ax.plot(years[f], vals[f], color=color, lw=1.7, ls=":")


def _end_labels(ax, ends, gap: float) -> None:
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    # Keep the stack inside the axes: if it overflows the top, shift it down.
    top = ax.get_ylim()[1]
    if ys and ys[-1] > top - gap * 0.3:
        ys = [y - (ys[-1] - (top - gap * 0.3)) for y in ys]
    for (x, _, txt, c), y in zip(ends, ys):
        ax.annotate(txt, (x, y), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=c, va="center")


def _panel(ax, ca: pd.DataFrame, col: str, title: str, ylabel: str) -> list:
    ends = []
    seen = []
    for cc in PANEL:
        d = _series(ca, cc, col)
        if d.empty:
            continue
        fc = d["forecast"].astype(bool).to_numpy()
        _plot_hist_fc(ax, d["year"], d[col], fc, PAL[cc], NM[cc])
        seen.append(d[col].to_numpy(float))
        ends.append((float(d["year"].iloc[-1]), float(d[col].iloc[-1]),
                     f"{d[col].iloc[-1]:.0f}", PAL[cc]))
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(Y0, Y1 + 2.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(labelsize=8)
    if ends:
        vals = np.concatenate(seen)
        lo = min(0.0, float(vals.min()))
        hi = float(vals.max())
        pad = max(4.0, (hi - lo) * 0.06)
        ax.set_ylim(lo - pad * 0.15, hi + pad)
        _end_labels(ax, ends, max(1.2, (hi - lo) * 0.045))
    return ends


def render(out: Path | None = None) -> Path:
    """2×2: central / GG debt service (top) and central / GG interest (bottom). US GG = federal."""
    out = out or (NOTE / "dm_debt_outlook_service_rev.png")
    _style()
    ca = _panel_frame(pd.read_parquet(CA))
    # US has no separate GG issuer book — federal is the peer-comparable basis on both GG panels.
    ca = ca.copy()
    ca["peer_ds"] = np.where(ca["cc"].eq("us"), ca["debt_service_pct_rev"], ca["gg_debt_service_pct_rev"])
    ca["peer_ir"] = np.where(ca["cc"].eq("us"), ca["interest_pct_rev"], ca["gg_interest_pct_rev"])
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    _panel(axes[0, 0], ca, "debt_service_pct_rev",
           f"A. Central debt service / central revenue to {Y1}", "% of central revenue")
    _panel(axes[0, 1], ca, "peer_ds",
           f"B. GG debt service / GG revenue to {Y1} (US federal)", "% of revenue")
    _panel(axes[1, 0], ca, "interest_pct_rev",
           f"C. Central interest / central revenue to {Y1}", "% of central revenue")
    _panel(axes[1, 1], ca, "peer_ir",
           f"D. GG interest / GG revenue to {Y1} (US federal)", "% of revenue")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=8, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.text(
        0.01, 0.01,
        "Top: debt service = interest + principal due (bills once, coupons inside the year). Bottom: interest only. "
        "Left column = central (issuer refinance). Right column = general government; US = federal on both.\n"
        "Solid = history; dotted = coupon book at market forwards; net issuance from IMF WEO. "
        "Central ratios are not peer-comparable (UK centre ~100% of GG revenue, Germany ~28%).",
        fontsize=7.2, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    return out


BENCH = ("de", "jp", "us", "fr", "uk", "es", "it")  # Bloomberg chart order
SNAP = Path(__file__).resolve().parents[1] / "derived" / "debt_snapshot.parquet"


def _fwd_start_2030(asof: pd.Timestamp | None = None) -> float:
    """Years from ``asof`` (default today) to mid-2030 — start argument for ``forward_par(..., 10y)``."""
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()
    return max((pd.Timestamp("2030-06-30") - asof).days / 365.25, 0.0)


# Back-compat for importers / notebooks that read the old constant.
_FWD_START_2030 = _fwd_start_2030()


def _auth_long_rate(cc: str) -> float:
    """Fiscal / EO long-rate assumption: CAO 名目長期金利 for Japan (2030); OECD IRL 2027 else."""
    from sovereign_cash.debt.build import _s
    from sovereign_cash.debt.sources import load

    src = load()
    if cc == "jp":
        s = _s(src, "jp", "cao_baseline", "irl")
        if 2030 in s.index and np.isfinite(s[2030]):
            return float(s[2030])
    s = _s(src, cc, "oecd", "irl")
    for y in (2027, 2026, 2025):
        if y in s.index and np.isfinite(s[y]):
            return float(s[y])
    return float("nan")


def _ours_10y(cc: str) -> tuple[float, float]:
    """Spot 10y par and the 10y forward that lands in 2030, both off the same government zero curve.

    The snapshot ``y10`` is a monthly average and falls back to the OECD IRL projection where there is no
    market series (Spain), which made "ours spot" equal the OECD number it is compared against.
    """
    spot = fwd = float("nan")
    try:
        from sovereign_cash.debt.gov_curve import forward_par, load as load_zeros
        z = load_zeros()
        asof = pd.Timestamp(z.loc[z["cc"].eq(cc), "asof"].iloc[0]) if (z["cc"] == cc).any() else None
        spot = float(forward_par(cc, 0.0, [10.0])[0])
        fwd = float(forward_par(cc, _fwd_start_2030(asof), [10.0])[0])
    except KeyError:
        # No government zero curve for this cc (e.g. SE/NO) — leave NaN.
        pass
    return spot, fwd


def _jp_cao_interest_pct_gdp(year: int = 2030) -> float:
    """CAO baseline interest / GDP (central), for the Japan attribution."""
    ca = pd.read_parquet(CA)
    j = ca.loc[ca["cc"].eq("jp") & (ca["year"] == year)]
    if j.empty or not np.isfinite(j.iloc[0].get("interest_authority_bn", np.nan)):
        return float("nan")
    return float(100.0 * j.iloc[0]["interest_authority_bn"] / j.iloc[0]["gdp_bn"])


def _netting_pp(cc: str, year: int = 2027) -> float:
    """OECD EO general-government gross minus net interest, % of GDP: the size of the IMF's netting."""
    from sovereign_cash.debt.build import _s
    from sovereign_cash.debt.sources import load

    src = load()
    g = _s(src, cc, "oecd", "gdp_lcu_bn")
    gi = _s(src, cc, "oecd", "interest_gross_bn")
    ni = _s(src, cc, "oecd", "interest_net_bn")
    if year not in g.index or year not in gi.index or year not in ni.index:
        return float("nan")
    return float(100.0 * (gi[year] - ni[year]) / g[year])


def _us_fed_vs_imf(year: int) -> tuple[float, float]:
    """US federal net interest / GDP (ours) and IMF GG net interest / GDP in `year`."""
    ca = pd.read_parquet(CA)
    r = ca.loc[ca["cc"].eq("us") & (ca["year"] == year)]
    w = pd.read_parquet(Path(__file__).resolve().parents[2] / "data" / "debt" / "weo.parquet")
    w = w[(w["var"] == "interest_pct") & (w["country"] == "us") & (w["year"] == year)]
    fed = float(r.iloc[0]["interest_pct_gdp"]) if len(r) else float("nan")
    imf = float(w.iloc[0]["value"]) if len(w) else float("nan")
    return fed, imf


def bench_attribution(b: pd.DataFrame | None = None) -> list[str]:
    """Country-by-country read of Exhibit A1 (interest gaps + rate panel), from live numbers."""
    b = bench_frame() if b is None else b
    d27 = b["o27"] - b["ec27"]
    d30 = b["o30"] - b["imf30"]
    rg = b["ours_10y_2030"] - b["auth_10y"]
    jp_cao = _jp_cao_interest_pct_gdp(2030)
    jp_ours_c = float("nan")
    ca = pd.read_parquet(CA)
    jr = ca.loc[ca["cc"].eq("jp") & (ca["year"] == 2030)]
    if len(jr) and np.isfinite(jr.iloc[0].get("interest_bn", np.nan)):
        jp_ours_c = float(100.0 * jr.iloc[0]["interest_bn"] / jr.iloc[0]["gdp_bn"])
    net = {cc: _netting_pp(cc, 2027) for cc in b.index}
    us_fed25, us_imf25 = _us_fed_vs_imf(2025)

    def nt(cc: str) -> str:
        return f"{float(net[cc]):.1f}pp"

    def pp(x: float) -> str:
        return f"{x:+.1f}pp".replace("-", "\u2212")

    def mag(x: float) -> str:
        return f"{abs(x):.1f}pp"

    def pct(x: float) -> str:
        return f"{x:.1f}%"

    return [
        "Reading Exhibit A1 country by country. Top: interest / GDP against the Commission (AMECO, gross interest "
        "payable, 2027) and the IMF (WEO primary minus overall balance, i.e. net of interest received for every "
        "country, 2030). Our general-government path is our central coupon book plus the base-year sub-central "
        "residual, so the gap to the IMF carries the IMF's netting (OECD gross minus net, 2027) on top of any rate "
        "difference. Bottom: Cabinet Office (Japan) or OECD Economic Outlook (others) 10y projection against our spot "
        "10y and 10y forward into 2030. The OECD and the Commission had earlier cut-offs, so much of the bar gap is "
        "the curve marked to today, not a more cautious assumption; and the 2030 bill depends on the whole curve "
        "from 2026, bills included, not on one 10y point.",
        f"Germany — near the Commission ({pp(float(d27['de']))} in 2027) and {mag(float(d30['de']))} above the IMF, "
        f"less than the IMF's netting ({nt('de')}): on a gross basis we sit at or below the IMF. OECD long rate "
        f"{pct(float(b.at['de', 'auth_10y']))} vs our 2030 forward {pct(float(b.at['de', 'ours_10y_2030']))} "
        f"({pp(float(rg['de']))}).",
        f"France — Commission-close in 2027 ({pp(float(d27['fr']))}); above the IMF by {mag(float(d30['fr']))} in "
        f"2030, of which netting is only {nt('fr')}. Largest rate gap in the panel: OECD "
        f"{pct(float(b.at['fr', 'auth_10y']))} vs spot {pct(float(b.at['fr', 'ours_10y_spot']))} and 2030 forward "
        f"{pct(float(b.at['fr', 'ours_10y_2030']))} ({pp(float(rg['fr']))}). Our net issuance is the IMF's, so the "
        "2030 gap to the IMF is the curve, not the deficit path or a definition mismatch.",
        f"Italy — essentially on the Commission ({pp(float(d27['it']))}) and {mag(float(d30['it']))} above the IMF "
        f"(netting {nt('it')}). BTP forward {pct(float(b.at['it', 'ours_10y_2030']))} sits {mag(float(rg['it']))} over "
        "the OECD long rate, but the book already carries a high average coupon, so the interest / GDP gap stays tight.",
        f"Spain — below the Commission in 2027 ({pp(float(d27['es']))}), {mag(float(d30['es']))} above the IMF by "
        f"2030, most of which is netting ({nt('es')}). Spot 10y matches the OECD "
        f"({pct(float(b.at['es', 'ours_10y_spot']))} vs {pct(float(b.at['es', 'auth_10y']))}); the 2030 forward "
        f"({pct(float(b.at['es', 'ours_10y_2030']))}) adds the rest.",
        f"UK — above both ({pp(float(d27['uk']))} vs Commission, {pp(float(d30['uk']))} vs IMF, netting {nt('uk')}). "
        f"Gilt spot {pct(float(b.at['uk', 'ours_10y_spot']))} and 2030 forward "
        f"{pct(float(b.at['uk', 'ours_10y_2030']))} vs OECD {pct(float(b.at['uk', 'auth_10y']))} "
        f"({pp(float(rg['uk']))}). The OBR and the Commission condition on market forwards too, so the gap is the "
        "sell-off since their cut-offs (mark-to-market), not a more cautious assumption. Beyond coupon roll, the UK "
        "bill also moves with RPI uplift on index-linked gilts and Bank Rate on QE reserves.",
        f"Japan — {pp(float(d27['jp']))} vs Commission, {pp(float(d30['jp']))} vs IMF on the GG chart. Against the "
        f"Cabinet Office's own general-account interest the gap is smaller: our central interest / GDP "
        f"~{pct(jp_ours_c)} vs CAO baseline ~{pct(jp_cao)} in 2030. CAO baseline long rate "
        f"{pct(float(b.at['jp', 'auth_10y']))} (the July spot held for FY2026, then a model path) vs our MoF 10y "
        f"forward {pct(float(b.at['jp', 'ours_10y_2030']))} ({pp(float(rg['jp']))}), plus coupon roll off BoJ-era "
        f"stock. The IMF gap is mostly netting ({nt('jp')} in 2027): WEO nets general-government interest receipts "
        "(FX reserves, FILP loans, pension reserve funds); we net only coupons on BoJ-held JGBs. That netting is "
        "gross of the interest the BoJ pays on reserves, which already exceeds its JGB income (FY2025), so it "
        "overstates the remittance offset; and history is gross, so the 2025→26 step is partly a basis break.",
        f"US — chart is federal net interest ({pct(float(b.at['us', 'o30']))} in 2030) against IMF "
        f"general-government net interest ({pct(float(b.at['us', 'imf30']))}). The perimeter is small on the IMF's "
        f"basis: in 2025 the IMF series ({pct(us_imf25)}) sits on federal net interest ({pct(us_fed25)}), because the "
        "IMF excludes imputed interest on unfunded pension liabilities and nets interest received. The OECD and "
        "Commission gross series include both and run ~1.5pp higher, so they are not a like-for-like check. The "
        f"{mag(float(d30['us']))} shortfall to the IMF is a forecast difference — the IMF's interest path rises faster "
        f"than ours — not perimeter. Rate panel: OECD {pct(float(b.at['us', 'auth_10y']))} vs our forward "
        f"{pct(float(b.at['us', 'ours_10y_2030']))} ({pp(float(rg['us']))}).",
    ]


def bench_numbers(b: pd.DataFrame | None = None) -> dict:
    """Flat key map for the A1 prose (same keys as peer_basis.numbers), from live bench_frame()."""
    b = bench_frame() if b is None else b
    d27 = (b["o27"] - b["ec27"]).round(1)
    d30 = (b["o30"] - b["imf30"]).round(1)
    out: dict = {}
    for cc in b.index:
        out[f"bench_d27_{cc}"], out[f"bench_d30_{cc}"] = float(d27[cc]), float(d30[cc])
        out[f"bench_o30_{cc}"] = float(b.at[cc, "o30"])
    out["bench_close_ec27"] = ", ".join(NM[c] for c in ("de", "fr", "it") if abs(d27[c]) <= 0.2)
    out["bench_rank_2030"] = ", ".join(f"{NM[c]} {v:.1f}" for c, v in b["o30"].sort_values(ascending=False).items())
    return out


def bench_frame() -> pd.DataFrame:
    """Interest / GDP plus 10y rate assumptions (ours vs authority / OECD)."""
    sc = Path(__file__).resolve().parents[2]  # sovereign_cash/ (the repo reaches it via the debt/data symlink)
    ca = pd.read_parquet(CA)
    ca["ig"] = ca["gg_interest_pct_gdp"]  # general government for every country, the US included (as the IMF and AMECO)
    o = ca.pivot_table(index="cc", columns="year", values="ig")
    a = pd.read_parquet(sc / "data" / "debt" / "ameco.parquet")
    a = a[a["var"] == "interest_gross_pct"].pivot_table(index="country", columns="year", values="value")
    w = pd.read_parquet(sc / "data" / "debt" / "weo.parquet")
    w = w[w["var"] == "interest_pct"].pivot_table(index="country", columns="year", values="value")
    rows = []
    for cc in BENCH:
        spot, fwd = _ours_10y(cc)
        rows.append({
            "cc": cc,
            "o16": o.at[cc, 2016], "o26": o.at[cc, 2026], "o27": o.at[cc, 2027],
            "o30": o.at[cc, 2030], "ec27": a.at[cc, 2027], "imf30": w.at[cc, 2030],
            "auth_10y": _auth_long_rate(cc),
            "ours_10y_spot": spot,
            "ours_10y_2030": fwd,
        })
    return pd.DataFrame(rows).set_index("cc")


def benchmark(out: Path | None = None) -> Path:
    out = out or (NOTE / "dm_debt_outlook_interest_gdp_bench.png")
    _style()
    b = bench_frame()
    x = np.arange(len(b))
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11.4, 7.2), gridspec_kw={"height_ratios": [1.15, 0.95], "hspace": 0.38})
    # --- interest / GDP
    ax.bar(x, b["o16"], width=0.55, color="#d9d9d9", edgecolor="white", linewidth=2, label="Ours 2016 (bar)",
           zorder=1)
    ax.scatter(x - 0.12, b["o26"], s=60, facecolor="white", edgecolor="#e8833a", linewidth=2, zorder=3,
               label="Ours 2026")
    ax.scatter(x - 0.12, b["o30"], s=60, facecolor="white", edgecolor="#1f6fb4", linewidth=2, zorder=3,
               label="Ours 2030")
    ax.scatter(x + 0.14, b["ec27"], s=55, marker="x", color="#222222", linewidth=1.6, zorder=3,
               label="Commission 2027 (gross)")
    ax.scatter(x + 0.14, b["imf30"], s=55, marker="D", facecolor="none", edgecolor="#222222", linewidth=1.4,
               zorder=3, label="IMF 2030 (net of interest received)")
    for i, cc in enumerate(b.index):
        ax.annotate(f"{b.at[cc, 'o30']:.1f}", (x[i] - 0.12, b.at[cc, "o30"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.5, color="#333333")
    ax.set_xticks(x, [NM[c] + (" (federal)" if c == "us" else "") for c in b.index], fontsize=9)
    ax.set_ylabel("% of GDP", fontsize=9)
    ax.set_ylim(0, max(b[["o16", "o26", "o30", "ec27", "imf30"]].max()) * 1.18)
    ax.yaxis.grid(True, color="#eeeeee", zorder=0)
    ax.set_title("Interest / GDP: ours against consensus (general government; US federal)", fontsize=10,
                 loc="left", fontweight="bold")
    ax.legend(fontsize=7.2, ncol=5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    # --- 10y rate assumptions (why the interest paths diverge)
    w = 0.22
    axr.bar(x - w, b["auth_10y"], width=w * 0.95, color="#d9d9d9", edgecolor="white", linewidth=1.5,
            label="CAO (Japan) / OECD EO long rate", zorder=1)
    axr.bar(x, b["ours_10y_spot"], width=w * 0.95, color="#e8833a", edgecolor="white", linewidth=1.5,
            label="Ours spot 10y", zorder=1)
    axr.bar(x + w, b["ours_10y_2030"], width=w * 0.95, color="#1f6fb4", edgecolor="white", linewidth=1.5,
            label="Ours 2030 fwd 10y", zorder=1)
    for i, cc in enumerate(b.index):
        for dx, col in ((-w, "auth_10y"), (0.0, "ours_10y_spot"), (w, "ours_10y_2030")):
            v = b.at[cc, col]
            if np.isfinite(v):
                axr.annotate(f"{v:.1f}", (x[i] + dx, v), xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=6.5, color="#333333")
    axr.set_xticks(x, [NM[c] for c in b.index], fontsize=9)
    axr.set_ylabel("%", fontsize=9)
    ymax = np.nanmax(b[["auth_10y", "ours_10y_spot", "ours_10y_2030"]].to_numpy(float))
    axr.set_ylim(0, ymax * 1.22)
    axr.yaxis.grid(True, color="#eeeeee", zorder=0)
    axr.set_title("10y yield: Cabinet Office / OECD projections vs our government curve", fontsize=10,
                  loc="left", fontweight="bold")
    axr.legend(fontsize=7.2, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.text(
        0.01, 0.01,
        "Top: Bloomberg Economics layout (bars = 2016, circles = projections). Commission = AMECO gross interest "
        "payable (EDP). IMF = WEO primary minus overall balance: net of interest received, all countries; US excludes "
        "imputed pension interest,\nso IMF US GG ≈ federal net. Japan forward subtracts coupons on BoJ-held JGBs, "
        "gross of BoJ reserve interest (upper bound on the remittance offset; history is gross). "
        "Bottom: Japan = CAO baseline-case\nnominal long rate FY2030; others = OECD Economic Outlook projection for "
        "2027 (third-party, earlier cut-off). Ours spot and 2030 = 10y par on the government zero curve "
        "(2030 = forward starting mid-2030).",
        fontsize=6.8, color="#666666",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, hspace=0.42)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    return out
