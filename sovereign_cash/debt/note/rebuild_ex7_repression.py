"""Rebuild Exhibit 7 — inflationary repression as **one** combined shock.

Case (history-scaled): **+100bp nominal** and **+2pp inflation** for three years.
That is Δi − Δπ ≈ **−100bp real** — inflation runs ahead of the nominal selloff
(Reinhart–Sbrancia postwar signature was negative reals, not Fisher pass-through).

Metric: Δ interest / revenue by **2030** (same liability horizon as Ex2–6 / roll shares), stacked as
  (1) from the nominal curve  (+100bp; ``shock_nominal``)
  (2) from inflation          (+2pp uplift + receipts; 2× the shipped +1pp sens)

Patches ``Sovereign_Debt_Outlook.html`` / PDF and copies the PDF to Desktop.
"""
from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from weasyprint import HTML as WHTML
except ImportError:  # charts do not need it
    WHTML = None

REPO = Path(__file__).resolve().parents[3]
from sovereign_cash.debt.note.paths import NOTE, OUT_HTML, OUT_PDF
HTML_P = OUT_HTML
PDF = OUT_PDF
DESK = Path.home() / "Desktop" / "Sovereign_Debt_Outlook.pdf"
DERIVED = REPO / "sovereign_cash" / "derived"
DERIVED_DEBT = REPO / "sovereign_cash" / "debt" / "derived"
RP_P = DERIVED_DEBT / "debt_repricing.parquet"
if not RP_P.is_file():
    RP_P = DERIVED / "debt_repricing.parquet"

NM = {
    "us": "US", "uk": "UK", "de": "Germany", "fr": "France",
    "it": "Italy", "jp": "Japan",
}
CORE = ("us", "uk", "jp", "fr", "it", "de")
C_RATE, C_INFL = "#4a6fa5", "#b33b5c"
GREY = "#94a3b8"
Y_SHOCK = 2030  # match Ex2–6 / Conditional-flows +100bp; not 2026+3
# Sens table is +1pp; scale linearly for the +2pp inflation leg.
INFL_PP = 2.0
INFL_SCALE = INFL_PP / 1.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def splice_figure(html: str, exhibit: str, png: Path, caption: str) -> str:
    n_h2 = html.count("<h2>")
    cm = re.search(
        rf"<figcaption[^>]*>\s*<b>{re.escape(exhibit)}\.?</b>.*?</figcaption>",
        html, re.S,
    )
    if not cm:
        raise SystemExit(f"missing {exhibit}")
    fs = html.rfind("<figure>", 0, cm.start())
    fe = html.find("</figure>", cm.end()) + len("</figure>")
    if html[fs:fe].count("<figure>") != 1:
        raise SystemExit(f"{exhibit} slice not single figure")
    b64 = base64.b64encode(png.read_bytes()).decode()
    repl = (f"<figure><img src='data:image/png;base64,{b64}' style='width:100%'/>"
            f"<figcaption>{caption}</figcaption></figure>")
    out = html[:fs] + repl + html[fe:]
    if out.count("<h2>") != n_h2:
        raise SystemExit(f"{exhibit} splice changed h2 count")
    return out


def _gdp_lift(board: pd.Series, debt_bn: float, gdp_bn: float) -> float:
    """NGDP level lift under the board's +1pp NGDP path; scale outside."""
    D, dD, g0 = debt_bn, float(board["d_debt_bn"]), gdp_bn
    s = float(board["d_debt_pct_gdp_nom_shocked"])
    if abs(D + dD) < 1e-9:
        den = s / 100.0 + D / g0
        return (D / den) / g0 - 1.0 if den else 0.0
    rhs = s + 100.0 * D / g0
    g1 = 100.0 * (D + dD) / rhs if rhs else g0
    return g1 / g0 - 1.0


def combined_table() -> pd.DataFrame:
    """Δ interest/revenue under +100bp nominal + +2pp inflation (≈ −100bp real)."""
    rp = pd.read_parquet(RP_P)
    rp = rp[(rp["scenario"] == "baseline") & (rp["year"] == Y_SHOCK)]
    cur = rp[rp["case"] == "current"].set_index("cc")
    rate = rp[rp["case"] == "shock_nominal"].set_index("cc")  # +100 nom

    sens = pd.read_parquet(DERIVED / "debt_bu_inflation_sens.parquet")
    board = sens.query("layer == 'board' and shock == 'cpi1_w1' and year == @Y_SHOCK").set_index("cc")
    ship = sens.query("layer == 'shipped' and shock == 'cpi1_w1' and year == @Y_SHOCK").set_index("cc")
    uk_rpi = sens.query("layer == 'board' and shock == 'rpi1_uk' and year == @Y_SHOCK and cc == 'uk'")

    rows = []
    for cc in CORE:
        I0 = float(cur.at[cc, "interest_bn"])
        R0 = float(cur.at[cc, "revenue_bn"])
        I_rate = float(rate.at[cc, "interest_bn"])
        b = board.loc[cc]
        dI_1 = float(b["d_interest_bn"])
        if cc == "uk" and len(uk_rpi):
            dI_1 = float(uk_rpi.iloc[0]["d_interest_bn"])
        dR_ship = float(ship.at[cc, "d_receipts_bn"]) if cc in ship.index else np.nan
        lift_1 = _gdp_lift(b, float(cur.at[cc, "debt_bn"]), float(cur.at[cc, "gdp_bn"]))
        if np.isfinite(dR_ship) and abs(dR_ship) > 0.05:
            dR_1, src = dR_ship, "shipped_receipts"
        else:
            dR_1, src = R0 * lift_1, "weo_ngdp"
        dI_infl = INFL_SCALE * dI_1
        dR = INFL_SCALE * dR_1

        ir0 = 100.0 * I0 / R0
        c_rate = 100.0 * I_rate / R0 - ir0
        ir_comb = 100.0 * (I_rate + dI_infl) / (R0 + dR) - ir0
        c_infl = ir_comb - c_rate
        rows.append({
            "cc": cc, "ir0": ir0, "d_ir_pp": ir_comb,
            "c_rate_pp": c_rate, "c_infl_pp": c_infl,
            "dI_rate_bn": I_rate - I0, "dI_infl_bn": dI_infl, "d_revenue_bn": dR,
            "revenue_src": src,
        })
    out = pd.DataFrame(rows).set_index("cc")
    assert np.allclose(out["c_rate_pp"] + out["c_infl_pp"], out["d_ir_pp"], atol=1e-9)
    return out


def chart(ir: pd.DataFrame, out: Path, basis: str = "central") -> None:
    gg = basis == "gg"
    peers = sorted(
        [c for c in CORE if c in ir.index],
        key=lambda c: float(ir.at[c, "d_ir_pp"]), reverse=True,
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    fig.suptitle(
        f"Inflationary repression: +100bp nominal + +{INFL_PP:.0f}pp inflation "
        f"(≈ −100bp real): Δ interest / {'GG ' if gg else ''}revenue by {Y_SHOCK}",
        fontsize=10.5, fontweight="bold", x=0.01, ha="left", y=0.98,
    )
    y = np.arange(len(peers))
    rate = np.array([float(ir.at[c, "c_rate_pp"]) for c in peers])
    infl = np.array([float(ir.at[c, "c_infl_pp"]) for c in peers])
    net = rate + infl

    for i in range(len(peers)):
        r, f = rate[i], infl[i]
        left = 0.0
        if r > 0:
            ax.barh(i, r, left=left, color=C_RATE, height=0.62)
            left += r
        if f > 0:
            ax.barh(i, f, left=left, color=C_INFL, height=0.62)
        left = 0.0
        if r < 0:
            ax.barh(i, r, left=left, color=C_RATE, height=0.62)
            left += r
        if f < 0:
            ax.barh(i, f, left=left, color=C_INFL, height=0.62)
        ax.plot(net[i], i, "o", color="#111", ms=5, zorder=4)
        ax.text(net[i] + (0.08 if net[i] >= 0 else -0.08), i,
                f"{net[i]:+.2f}", va="center",
                ha="left" if net[i] >= 0 else "right", fontsize=8.5, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([NM[c] for c in peers])
    ax.invert_yaxis()
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel(f"Δ interest / {'general-government ' if gg else ''}revenue by {Y_SHOCK}, pp")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_RATE),
        plt.Rectangle((0, 0), 1, 1, color=C_INFL),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#111", markersize=5),
    ]
    fig.legend(
        handles,
        ["From +100bp nominal",
         f"From +{INFL_PP:.0f}pp inflation (uplift + receipts)",
         "Net (combined)"],
        loc="lower center", ncol=3, fontsize=7.5, frameon=False,
        bbox_to_anchor=(0.5, 0.86),
    )
    fig.text(
        0.01, 0.02,
        "Shock: +100bp on the zeros and +2pp CPI/RPI for three years (Δi−Δπ ≈ −100bp real) — "
        "closer to postwar repression (negative reals) than Fisher pass-through.\n"
        + ("Central book scaled to general government by the ratio of general-government to central interest, "
           "over general-government revenue; US federal. Inflation leg = 2× the +1pp sensitivity."
           if gg else "Inflation leg = 2× the shipped +1pp BU sens. US/JP revenue = WEO×NGDP."),
        fontsize=6.8, color=GREY, ha="left", va="bottom",
    )
    xs = np.concatenate([rate, infl, net, [0.0]])
    pad = max(0.35, 0.10 * float(np.nanmax(np.abs(xs))))
    ax.set_xlim(float(np.nanmin(xs)) - pad, float(np.nanmax(xs)) + pad)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.19)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


LEAD_PUNCH = (
    "<p><b>Financial repression does not work this time: at today’s debt stocks a free curve "
    "means even −100bp real (+100bp nom / +2pp infl) still raises interest/revenue — "
    "the vulnerable need productivity and a primary, not an inflation exit.</b></p>"
)
LEAD_SUPPORT = (
    "<p>Postwar liquidation needed capped nominals and captive buyers (Reinhart–Sbrancia). "
    "Exhibit 7 is the open-market case: the inflation receipt windfall loses to roll on a large book "
    "— JP/US/FR hardest; UK/DE least via duration. Soft path (Ex6) helps the interest path; "
    "it does not close France’s structural gap. Escape hatch that scales with the stock is "
    "<i>g</i> (r − g) plus primary delivery.</p>"
)


def patch_prose(html: str, ir: pd.DataFrame) -> str:
    uk, us, jp = ir.loc["uk"], ir.loc["us"], ir.loc["jp"]
    body = f"""
<p><b>Inflationary financial repression here is +100bp on nominal yields and +2pp inflation for three years (about −100bp real) — inflation runs ahead of the selloff, in the spirit of the postwar negative-real episodes (Reinhart–Sbrancia), not Fisher pass-through.</b> Exhibit 7 is that joint Δ interest/revenue by {Y_SHOCK} (same horizon as the rest of the liability path): steel = curve, coral = inflation (uplift + receipts), black = net.</p>
<p>For the UK the combined hit is <b>+{uk['d_ir_pp']:.2f}pp</b> (curve +{uk['c_rate_pp']:.2f}, inflation {uk['c_infl_pp']:+.2f}). The inflation leg is linker uplift less the CPI receipt offset; whether CPI alone helps the primary still depends on wages — matching pay is roughly neutral, flat pay worsens the primary need by ~£14bn / 0.4% of GDP at +1pp (scale roughly with the inflation shock).</p>
<p>The United States nets <b>{us['d_ir_pp']:+.2f}pp</b> (curve +{us['c_rate_pp']:.2f}, inflation {us['c_infl_pp']:+.2f}). Japan’s curve leg still dominates (+{jp['c_rate_pp']:.2f}); JGBi uplift stays small versus revenue. Rankings on the curve leg match the Conditional-flows +100bp duration story; the inflation leg is what repression adds.</p>
<p>That is why repression fails as a cashflow strategy now: the curve leg (free markets, high stocks, fast roll) dominates the inflation receipt windfall everywhere except where duration is long — and even there the net interest/revenue ratio still rises.
See the Lead for the high-level read.</p>
"""
    html2, n = re.subn(
        r"(<h2>Inflation exits</h2>)\s*.*?(?=<figure>)",
        r"\1\n" + body,
        html,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit(f"prose splice failed (n={n})")

    # Lead punchline: strip any prior repression open, then put punch + support first.
    html2 = re.sub(
        r"(<h2>Lead</h2>)\s*"
        r"(?:<p><b>Financial repression does not work this time:.*?</b></p>\s*)?"
        r"(?:<p>Postwar liquidation needed capped nominals.*?</p>\s*)?"
        r"(?:<p><b>Financial repression does not clear the cashflow meter this time\.</b>.*?</p>\s*)?"
        r"(?:<p><b>Who is most vulnerable therefore needs higher productivity.*?</p>\s*)?",
        r"\1\n" + LEAD_PUNCH + "\n" + LEAD_SUPPORT + "\n",
        html2,
        count=1,
        flags=re.S,
    )
    if LEAD_PUNCH not in html2:
        raise SystemExit("Lead punchline missing after patch")
    return html2


def main() -> None:
    ir = combined_table()
    png = NOTE / "dm_debt_outlook_mtm.png"
    chart(ir, png)
    print(ir.round(3).to_string())
    print("wrote", png)

    caption = (
        "<b>Exhibit 7.</b> Inflationary repression — +100bp nominal <i>and</i> "
        f"+{INFL_PP:.0f}pp inflation × 3y (≈ −100bp real). "
        f"Δ interest/revenue by {Y_SHOCK}. Steel = curve; coral = inflation "
        "(uplift + receipts, 2× the +1pp BU sens); black = net. "
        f"UK <b>+{ir.at['uk', 'd_ir_pp']:.2f}pp</b>; "
        f"US <b>{ir.at['us', 'd_ir_pp']:+.2f}pp</b>; "
        f"Japan <b>{ir.at['jp', 'd_ir_pp']:+.2f}pp</b>."
    )
    html = HTML_P.read_text()
    html = splice_figure(html, "Exhibit 7", png, caption)
    html = patch_prose(html, ir)
    HTML_P.write_text(html)
    print("patched", HTML_P)

    WHTML(filename=str(HTML_P)).write_pdf(str(PDF))
    shutil.copy2(PDF, DESK)
    print("pdf", PDF, "→", DESK)


if __name__ == "__main__":
    main()


def figure(out: Path | None = None):
    """Note-build adapter. Returns (png, combined interest table)."""
    out = Path(out) if out else (NOTE / "figures" / "ex7_repression.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    ir = combined_table()
    chart(ir, out)
    return out, ir
