"""Monthly accrued / politics print meters — thin layer on top of cash Ex1c.

Not a second cash stack. Product = outturn + authority profile + light seasonal
nowcast + surprise. CGNCR / SMB / NKA financing paths stay in Ex1c.

Phase 1: UK PSNB ex banks vs OBR (annual + PSF-day YTD profile stub).
Phase 2: ES IGAE AC B.9 (deficit +) vs prior-year pace / seas_add_3.
Spike: FR / IT / DE monthly GG — see ``spike_fr_it_de()``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
from sovereign_cash.debt.note.paths import NOTE, OUT_HTML, OUT_PDF
DERIVED_DEBT = REPO / "sovereign_cash" / "debt" / "derived"
UK_PROFILE = REPO / "sovereign_cash" / "data" / "uk" / "obr_psnb_profile_ytd.json"
PF = REPO / "uk_flows" / "output" / "public_finances.parquet"
OBR = REPO / "sovereign_cash" / "data" / "debt" / "obr_lines.parquet"
SPIKE_PATH = NOTE / "print_meters_spike.json"

PRE = (pd.Timestamp("2012-04-01"), pd.Timestamp("2019-03-01"))
# Ship UK seasonal only if PRE h=1 MAE beats the hist-profile benchmark by this band.
EDGE = 0.03


def _gdp(cc: str, year: int) -> float:
    ann = pd.read_parquet(DERIVED_DEBT / "debt_country_annual.parquet")
    r = ann[(ann["cc"] == cc) & (ann["year"] == year)]
    if r.empty:
        r = ann[ann["cc"] == cc].sort_values("year").iloc[[-1]]
    return float(r["gdp_bn"].iloc[0])


# ── UK PSNB ─────────────────────────────────────────────────────────────

def uk_psnb_monthly() -> pd.Series:
    """PSNB ex banks, deficit (+), £bn, month-start index."""
    pf = pd.read_parquet(PF)
    s = pf[pf["series"].eq("psnb_ex") & pf["frequency"].eq("monthly")].copy()
    s["month"] = pd.to_datetime(s["date"]).dt.to_period("M").dt.to_timestamp()
    # PUSF stores borrowing as negative.
    return (-s.set_index("month")["value"].sort_index() / 1000.0).astype(float)


def uk_obr_annual_psnb() -> dict[int, float]:
    """OBR EFO PSNB by FY-start year (2026 → FY2026-27)."""
    obr = pd.read_parquet(OBR)
    rows = obr[obr["var"].eq("psnb_bn")][["year", "value"]].drop_duplicates()
    return {int(r.year): float(r.value) for r in rows.itertuples(index=False)}


def uk_profile_stub() -> dict:
    if not UK_PROFILE.exists():
        return {}
    return json.loads(UK_PROFILE.read_text())


def _fy_of_uk(ts: pd.Timestamp) -> int:
    return int(ts.year if ts.month >= 4 else ts.year - 1)


def _uk_month_shares(x: pd.Series, origin: pd.Timestamp, n: int = 5) -> dict[int, float]:
    from sovereign_cash.uk.model import month_profile

    return month_profile(x.loc[:origin].dropna(), origin, n=n)


def uk_plan_monthly_path(through: pd.Timestamp | None = None) -> pd.Series:
    """Monthly OBR-implied PSNB path for the open FY, £bn.

    Hist month shares × OBR annual, then rescaled so cumulative through the stub
    ``through`` month equals ``ytd_profile_bn`` when the stub covers that FY.
    """
    x = uk_psnb_monthly().dropna()
    origin = through or x.index.max()
    fy = _fy_of_uk(origin)
    annual = uk_obr_annual_psnb().get(fy)
    if annual is None or not np.isfinite(annual):
        raise KeyError(f"no OBR PSNB for FY{fy}")
    months = pd.date_range(pd.Timestamp(fy, 4, 1), periods=12, freq="MS")
    prof = _uk_month_shares(x, origin)
    raw = pd.Series([annual * prof[m.month] for m in months], index=months)
    stub = uk_profile_stub()
    if (
        stub.get("fy") == f"{fy}-{str(fy + 1)[-2:]}"
        and stub.get("ytd_profile_bn") is not None
        and stub.get("through")
    ):
        th = pd.Timestamp(str(stub["through"]) + "-01")
        done = months[months <= th]
        if len(done):
            scale = float(stub["ytd_profile_bn"]) / float(raw.loc[done].sum())
            raw.loc[done] = raw.loc[done] * scale
            rest = months[months > th]
            if len(rest):
                rem = annual - float(stub["ytd_profile_bn"])
                w = raw.loc[rest].to_numpy(float)
                if w.sum() != 0:
                    raw.loc[rest] = rem * w / w.sum()
                else:
                    raw.loc[rest] = rem / len(rest)
    return raw


def uk_seas_add_forecast(origin: pd.Timestamp, h: int = 1, k: int = 3) -> pd.Series | None:
    from sovereign_cash.core.cash_framework import forecast

    y = uk_psnb_monthly().dropna()
    return forecast(y, origin, h, f"seas_add_{k}")


def uk_profile_forecast(origin: pd.Timestamp, h: int = 1,
                        annual_override: float | None = None) -> pd.Series | None:
    """Next h months = remaining hist-share slice of the FY annual total.

    Live: OBR annual. PRE score: realised FY total when known at year-end (for
    origins inside a completed FY), else trailing-12 × (12 / months_elapsed) is
    not used — skip. OBR vintages only start mid-2020s, so PRE needs outturn
    annuals or the profile MAE is NaN and the gate is vacuous.
    """
    y = uk_psnb_monthly().dropna()
    fy = _fy_of_uk(origin)
    annual = annual_override
    if annual is None:
        annual = uk_obr_annual_psnb().get(fy)
    if annual is None:
        # Honest PRE stand-in: prior complete FY total (known at this FY's start).
        prev = pd.date_range(pd.Timestamp(fy - 1, 4, 1), periods=12, freq="MS")
        block = y.reindex(prev)
        if not block.notna().all():
            return None
        annual = float(block.sum())
    try:
        prof = _uk_month_shares(y, origin)
    except Exception:
        return None
    months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fy_months = pd.date_range(pd.Timestamp(fy, 4, 1), periods=12, freq="MS")
    done = [m for m in fy_months if m <= origin]
    ytd = float(y.reindex(done).sum()) if done else 0.0
    rest = [m for m in fy_months if m > origin]
    w = np.array([prof[m.month] for m in rest], dtype=float)
    if not len(rest) or w.sum() == 0:
        return None
    rem = float(annual) - ytd
    by_m = {m: rem * prof[m.month] / w.sum() for m in rest}
    return pd.Series([by_m.get(m, np.nan) for m in months], index=months)


def score_uk_h1(pre: tuple[pd.Timestamp, pd.Timestamp] = PRE) -> pd.DataFrame:
    """Walk-forward h=1 MAE: seas_add_3 vs hist-profile remainder vs smly."""
    from sovereign_cash.core.cash_framework import forecast

    y = uk_psnb_monthly().dropna()
    rows = []
    origins = y.loc[pre[0]:pre[1]].index
    for o in origins:
        if o + pd.DateOffset(months=1) not in y.index:
            continue
        actual = float(y.loc[o + pd.DateOffset(months=1)])
        seas = forecast(y, o, 1, "seas_add_3")
        smly = forecast(y, o, 1, "smly")
        prof = uk_profile_forecast(o, 1)
        for name, fc in (("seas_add_3", seas), ("smly", smly), ("profile", prof)):
            if fc is None or not np.isfinite(fc.iloc[0]):
                continue
            rows.append({
                "origin": o, "model": name,
                "err": float(fc.iloc[0] - actual),
                "ae": abs(float(fc.iloc[0] - actual)),
            })
    return pd.DataFrame(rows)


def uk_ship_model(score: pd.DataFrame | None = None) -> dict:
    """Choose seas_add_3 only if PRE MAE beats profile by EDGE; else profile."""
    score = score if score is not None else score_uk_h1()
    if score.empty:
        return {"model": "profile", "ship_seas": False, "reason": "empty score"}
    mae = score.groupby("model")["ae"].mean()
    prof = float(mae.get("profile", np.nan))
    seas = float(mae.get("seas_add_3", np.nan))
    ship = bool(np.isfinite(seas) and np.isfinite(prof) and seas < prof * (1.0 - EDGE))
    return {
        "model": "seas_add_3" if ship else "profile",
        "ship_seas": ship,
        "mae_seas": seas,
        "mae_profile": prof,
        "mae_smly": float(mae.get("smly", np.nan)),
        "reason": (
            f"seas PRE MAE {seas:.3f} beats profile {prof:.3f} by ≥{EDGE:.0%}"
            if ship else
            f"seas PRE MAE {seas:.3f} does not beat profile {prof:.3f} by {EDGE:.0%}; ship profile"
        ),
    }


def uk_live_state() -> dict:
    """Live PSNB print card: YTD vs OBR stub / hist path, next-month nowcast."""
    y = uk_psnb_monthly().dropna()
    last = y.index.max()
    fy = _fy_of_uk(last)
    fy_months = pd.date_range(pd.Timestamp(fy, 4, 1), periods=12, freq="MS")
    done = fy_months[fy_months <= last]
    ytd = float(y.reindex(done).sum())
    plan = uk_plan_monthly_path(last)
    plan_td = float(plan.reindex(done).sum())
    stub = uk_profile_stub()
    official_gap = stub.get("ytd_gap_bn") if stub.get("through") == f"{last:%Y-%m}" else None
    ship = uk_ship_model()
    if ship["model"] == "seas_add_3":
        fc = uk_seas_add_forecast(last, 1, 3)
    else:
        fc = uk_profile_forecast(last, 1)
    next_m = (last + pd.offsets.MonthBegin(1)) if fc is not None else None
    gdp = _gdp("uk", fy)
    return {
        "cc": "uk",
        "meter": "psnb_ex",
        "through": f"{last:%Y-%m}",
        "fy": f"{fy}-{str(fy + 1)[-2:]}",
        "ytd_bn": ytd,
        "plan_td_bn": plan_td,
        "gap_bn": ytd - plan_td,
        "official_gap_bn": official_gap,
        "annual_plan_bn": float(uk_obr_annual_psnb().get(fy, np.nan)),
        "ytd_gdp": 100.0 * ytd / gdp,
        "plan_td_gdp": 100.0 * plan_td / gdp,
        "next_month": f"{next_m:%Y-%m}" if next_m is not None else None,
        "next_bn": float(fc.iloc[0]) if fc is not None else None,
        "ship": ship,
        "plan_path": plan,
        "outturn": y.reindex(done),
    }


# ── ES B.9 ──────────────────────────────────────────────────────────────

def es_b9_deficit_monthly() -> pd.Series:
    """IGAE AC financing need, deficit (+), €bn (= −B.9)."""
    from sovereign_cash.debt.note.monthly_primary import es_monthly

    return es_monthly()["cash_requirement_bn"].astype(float).dropna().sort_index()


def es_prior_year_pace(origin: pd.Timestamp) -> float | None:
    """Same calendar month last year."""
    y = es_b9_deficit_monthly()
    prev = origin - pd.DateOffset(years=1)
    if prev not in y.index:
        return None
    return float(y.loc[prev])


def es_seas_forecast(origin: pd.Timestamp, h: int = 1) -> pd.Series | None:
    from sovereign_cash.core.cash_framework import forecast

    return forecast(es_b9_deficit_monthly(), origin, h, "seas_add_3")


def score_es_h1(pre_end: pd.Timestamp = pd.Timestamp("2022-12-01")) -> pd.DataFrame:
    from sovereign_cash.core.cash_framework import forecast

    y = es_b9_deficit_monthly()
    rows = []
    for o in y.loc[:pre_end].index:
        tgt = o + pd.DateOffset(months=1)
        if tgt not in y.index or o.year < 2016:
            continue
        actual = float(y.loc[tgt])
        seas = forecast(y, o, 1, "seas_add_3")
        smly = forecast(y, o, 1, "smly")
        for name, fc in (("seas_add_3", seas), ("smly", smly)):
            if fc is None or not np.isfinite(fc.iloc[0]):
                continue
            rows.append({
                "origin": o, "model": name,
                "err": float(fc.iloc[0] - actual),
                "ae": abs(float(fc.iloc[0] - actual)),
            })
    return pd.DataFrame(rows)


def es_ship_model(score: pd.DataFrame | None = None) -> dict:
    score = score if score is not None else score_es_h1()
    if score.empty:
        return {"model": "smly", "ship_seas": False, "reason": "empty score"}
    mae = score.groupby("model")["ae"].mean()
    seas = float(mae.get("seas_add_3", np.nan))
    smly = float(mae.get("smly", np.nan))
    ship = bool(np.isfinite(seas) and np.isfinite(smly) and seas < smly * (1.0 - EDGE))
    return {
        "model": "seas_add_3" if ship else "smly",
        "ship_seas": ship,
        "mae_seas": seas,
        "mae_smly": smly,
        "reason": (
            f"seas PRE MAE {seas:.3f} beats smly {smly:.3f}"
            if ship else
            f"seas does not beat smly; ship smly"
        ),
    }


def es_live_state() -> dict:
    from sovereign_cash.core.cash_framework import forecast

    y = es_b9_deficit_monthly()
    last = y.index.max()
    ytd = float(y.loc[f"{last.year}-01-01":last].sum())
    prior = y.loc[f"{last.year - 1}-01-01":f"{last.year - 1}-{last.month:02d}-01"]
    prior_td = float(prior.sum()) if len(prior) == last.month else float("nan")
    ship = es_ship_model()
    fc = forecast(y, last, 1, ship["model"])
    gdp = _gdp("es", last.year)
    return {
        "cc": "es",
        "meter": "igae_ac_b9",
        "through": f"{last:%Y-%m}",
        "fy": str(last.year),
        "ytd_bn": ytd,
        "prior_td_bn": prior_td,
        "gap_bn": ytd - prior_td if np.isfinite(prior_td) else float("nan"),
        "ytd_gdp": 100.0 * ytd / gdp,
        "prior_td_gdp": 100.0 * prior_td / gdp if np.isfinite(prior_td) else float("nan"),
        "next_month": f"{(last + pd.offsets.MonthBegin(1)):%Y-%m}",
        "next_bn": float(fc.iloc[0]) if fc is not None else None,
        "ship": ship,
        "outturn_ytd": y.loc[f"{last.year}-01-01":last],
        "prior_ytd": prior if len(prior) else pd.Series(dtype=float),
    }


# ── Spike FR / IT / DE ──────────────────────────────────────────────────

def spike_fr_it_de() -> dict:
    """Whether monthly GG / national-accounts print meters exist in-repo."""
    findings = {
        "fr": {
            "usable_monthly_gg": False,
            "have": "SMB State cash (solde général) — Ex1c; GG Maastricht annual in plan.json / Ex1d",
            "missing": "No stable monthly GG/Maastricht path in sovereign_cash/data/fr",
            "action": "Keep Ex1d annual politics; do not invent monthly Maastricht",
        },
        "it": {
            "usable_monthly_gg": False,
            "have": "BdI/MEF fabbisogno settore statale monthly (cash-ish) — Ex1c",
            "missing": "Maastricht / ind. netto is quarterly / annual; not a monthly print meter here",
            "action": "No Phase-2 IT accrued engine",
        },
        "de": {
            "usable_monthly_gg": False,
            "have": "BMF Finanzierungssaldo + Sondervermögen Kredit (cash); Destatis extra-budget weak cross-check (corr ~0.22)",
            "missing": "No clean monthly Maastricht GG series wired for print surprise",
            "action": "No Phase-2 DE accrued engine",
        },
        "verdict": (
            "FR/IT/DE: no usable monthly GG print meter in-repo. "
            "Phase 2 stays ES B.9 only; FR politics remains Ex1d annual."
        ),
    }
    SPIKE_PATH.write_text(json.dumps(findings, indent=2))
    return findings


# ── Exhibit 1b ──────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def render(out: Path | None = None) -> tuple[Path, str, str]:
    """Two-panel print meters: UK PSNB, ES B.9."""
    out = out or (NOTE / "dm_debt_outlook_print.png")
    _style()
    uk = uk_live_state()
    es = es_live_state()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.18, wspace=0.28)
    fig.text(0.07, 0.92, "Print meters — accrued deficit vs authority path",
             fontsize=12.5, weight="bold", color="#111", ha="left")

    # UK
    ax = axes[0]
    ot = uk["outturn"]
    plan = uk["plan_path"]
    mi = np.arange(1, len(ot) + 1)
    ax.plot(mi, 100.0 * ot.cumsum().to_numpy() / _gdp("uk", _fy_of_uk(ot.index.max())),
            color="#8b1e1e", lw=1.8, marker="o", ms=3.5, label="PSNB YTD")
    pmi = np.arange(1, 13)
    ax.plot(pmi, 100.0 * plan.cumsum().to_numpy() / _gdp("uk", _fy_of_uk(ot.index.max())),
            color="#8b1e1e", lw=1.2, ls=(0, (1.4, 1.5)), label="OBR path")
    gap = uk["official_gap_bn"] if uk["official_gap_bn"] is not None else uk["gap_bn"]
    ax.set_title(
        f"UK PSNB  {uk['through']}   YTD £{uk['ytd_bn']:.0f}bn "
        f"({gap:+.0f} vs OBR)",
        loc="left", fontsize=9, color="#8b1e1e",
    )
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks([1, 6, 12])
    ax.set_xlabel("Month of FY", fontsize=8)
    ax.set_ylabel("% GDP", fontsize=8)
    ax.axhline(0, color="#333", lw=0.4)
    nxt = uk["next_bn"]
    ship = uk["ship"]["model"]
    ax.text(0.02, 0.02, f"next ({ship}): £{nxt:.1f}bn" if nxt is not None else "",
            transform=ax.transAxes, fontsize=7.2, color="#444")

    # ES
    ax = axes[1]
    ot = es["outturn_ytd"]
    pr = es["prior_ytd"]
    gdp = _gdp("es", int(es["fy"]))
    if len(ot):
        ax.plot(np.arange(1, len(ot) + 1),
                100.0 * ot.cumsum().to_numpy() / gdp,
                color="#c45c26", lw=1.8, marker="o", ms=3.5, label="B.9 deficit YTD")
    if len(pr):
        ax.plot(np.arange(1, len(pr) + 1),
                100.0 * pr.cumsum().to_numpy() / gdp,
                color="#c45c26", lw=1.15, alpha=0.35, label="prior FY")
    gap_es = es["gap_bn"]
    ax.set_title(
        f"ES IGAE AC B.9  {es['through']}   YTD €{es['ytd_bn']:.0f}bn "
        f"({gap_es:+.0f} vs prior FY)" if np.isfinite(gap_es) else
        f"ES IGAE AC B.9  {es['through']}",
        loc="left", fontsize=9, color="#c45c26",
    )
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks([1, 6, 12])
    ax.set_xlabel("Month of calendar year", fontsize=8)
    ax.axhline(0, color="#333", lw=0.4)
    nxt = es["next_bn"]
    ship = es["ship"]["model"]
    ax.text(0.02, 0.02, f"next ({ship}): €{nxt:.1f}bn" if nxt is not None else "",
            transform=ax.transAxes, fontsize=7.2, color="#444")

    fig.text(
        0.07, 0.04,
        "Accrued / national-accounts print meters — not financing. "
        "UK: PSNB ex banks vs OBR (PSF-day YTD profile stub + hist month shares). "
        "ES: IGAE Administración Central B.9 deficit. "
        "Cash / CGNCR / SMB stays Exhibit 1c.",
        fontsize=7.4, color="#555", ha="left", va="center",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # Prose block for the note
    uk_gap = uk["official_gap_bn"] if uk["official_gap_bn"] is not None else uk["gap_bn"]
    prose = (
        f"<h3>Print meters (accrued)</h3>\n"
        f"<p>Exhibit 1b tracks the monthly <i>accrued / politics</i> meters that move the "
        f"print — not the cash path in Exhibit 1c. "
        f"<b>UK PSNB</b> YTD £{uk['ytd_bn']:.0f}bn through {uk['through']} "
        f"(<span class=\"gap\">{uk_gap:+.0f}bn vs OBR March profile</span>; "
        f"October EFO annual PSNB £133bn); "
        f"next-month nowcast £{uk['next_bn']:.1f}bn on <code>{uk['ship']['model']}</code> "
        f"({uk['ship']['reason']}). "
        f"<b>ES IGAE AC B.9</b> YTD €{es['ytd_bn']:.0f}bn "
        f"({es['gap_bn']:+.0f} vs same months last year); "
        f"next €{es['next_bn']:.1f}bn on <code>{es['ship']['model']}</code>. "
        f"FR/IT/DE have no usable monthly GG series here — Ex1d stays annual for France.</p>\n"
    )
    caption = (
        "<b>Exhibit 1b.</b> Accrued print meters, cumulative % of GDP. "
        "UK: PSNB ex banks vs OBR path (PSF YTD stub). "
        "ES: IGAE Administración Central B.9 deficit vs prior year. "
        "Not the cash / financing perimeter (Exhibit 1c)."
    )
    return out, caption, prose


def splice_into_html(html_path: Path, png: Path, caption: str, prose: str) -> None:
    """Insert Ex1b figure + prose immediately before the FYTD cash h3."""
    import re

    html = html_path.read_text()
    n_h2 = html.count("<h2>")
    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    block = (
        f"{prose}"
        f"<figure><img src='data:image/png;base64,{b64}' style='width:100%;max-width:100%'/>"
        f"<figcaption>{caption}</figcaption></figure>\n"
    )
    # Drop a previous Ex1b block if present.
    html = re.sub(
        r"<h3>Print meters \(accrued\)</h3>.*?</figure>\s*",
        "",
        html,
        count=1,
        flags=re.S,
    )
    marker = "<h3>FYTD cash requirement vs authority plan</h3>"
    if marker not in html:
        raise SystemExit("FYTD cash h3 not found for Ex1b splice")
    html2 = html.replace(marker, block + marker, 1)
    if "Exhibit 1b." not in html2:
        raise SystemExit("Ex1b caption missing after splice")
    if html2.count("<h2>") != n_h2:
        raise SystemExit("Ex1b splice changed h2 count")
    html_path.write_text(html2)


def build(html_path: Path | None = None) -> dict:
    """Score, render, splice, write spike. Returns live summary."""
    html_path = html_path or (OUT_HTML)
    uk_score = score_uk_h1()
    es_score = score_es_h1()
    uk_ship = uk_ship_model(uk_score)
    es_ship = es_ship_model(es_score)
    spike = spike_fr_it_de()
    png, cap, prose = render()
    if html_path.exists():
        splice_into_html(html_path, png, cap, prose)
    summary = {
        "uk_ship": uk_ship,
        "es_ship": es_ship,
        "uk": {k: v for k, v in uk_live_state().items()
               if k not in ("plan_path", "outturn")},
        "es": {k: v for k, v in es_live_state().items()
               if k not in ("outturn_ytd", "prior_ytd")},
        "spike": spike["verdict"],
        "png": str(png),
    }
    (NOTE / "print_meters_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


if __name__ == "__main__":
    s = build()
    print(json.dumps({k: s[k] for k in ("uk_ship", "es_ship", "spike")}, indent=2))
    print("uk ytd", s["uk"]["ytd_bn"], "gap", s["uk"].get("official_gap_bn") or s["uk"]["gap_bn"])
    print("es ytd", s["es"]["ytd_bn"], "gap", s["es"]["gap_bn"])
    print("wrote", s["png"])
