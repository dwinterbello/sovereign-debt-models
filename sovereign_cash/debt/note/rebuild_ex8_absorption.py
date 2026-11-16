"""Rebuild Exhibits 8a / 8c: the job and the absorption gaps.

  8b  investor-base stock (rebuild_ex8_holders.py)
  8a  next vs previous Δ holdings (net + CB runoff)
  8c  gaps: last-4Q repeats | elastic −ELASTIC_CUT | JP fills by selling peers

Evidence maps — country-specific. A sector is price-insensitive ("inelastic") when its buying is sticky
absorption (no forward predictive link to the next price move, and/or same-window buying into cheapening).
Pooled EA bank tests are not used for the bank role: BSI bank flows are scored DE / FR / IT separately.
  UK  banks + ICPF inelastic; OFI + overseas + households elastic.
      Banks (`uk_banks_f4q`): no forward link to y10 / s1030 / asw20 in any window; same-window co-move
      with s1030 +0.42 / asw20 +0.24 (buy into curve steepening / ASW cheapening — LCR / reserve
      substitutes under QT). OFI: forward link to y10 (t 2.5–4.7, OOS R² 0.07–0.20).
  US  insurers + pensions inelastic; funds/dealers/HH/RoW/banks elastic
  JP  insurers/trust/mutual/agri inelastic; city/regional/foreign elastic
  EA  insurers + pension inelastic; funds/HH/RoW elastic. Banks by country (BSI 12m, 2010+):
      IT inelastic — no forward link; co-move with Bund-spread widening (+0.14; stress tercile +0.09).
      FR elastic — no forward link; co-move ~0 vs y10 / spread (−0.09 / −0.04); not a selloff bid.
      DE elastic — no forward link; co-move with y10 −0.17 (sells into selloffs).
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

from issuance.experiments.buyer_tape import ROLE, Z1_LINES
from issuance.paths import CACHE
from issuance import ecb_holders as eh
from issuance import uk_holders as ukh
from sovereign_cash.debt import sensitivities as sens

REPO = Path(__file__).resolve().parents[3]
NOTE = Path(__file__).resolve().parent
OUT_HTML = NOTE / "Sovereign_Debt_Outlook.html"
OUT_PDF = NOTE / "Sovereign_Debt_Outlook.pdf"
HTML_P = OUT_HTML
PDF = OUT_PDF
DESK = Path.home() / "Desktop" / "Sovereign_Debt_Outlook.pdf"
SNAP = REPO / "sovereign_cash" / "derived" / "debt_snapshot.parquet"
JP_OUT = CACHE / "jp_outbound_debt.parquet"

PEERS = ("us", "uk", "de", "fr", "it", "es", "jp")


def _ea_banks_inelastic() -> tuple[str, ...]:
    """From ``issuance.experiments.bank_role_country`` (per-country BSI/ONS tests)."""
    try:
        from issuance.experiments.bank_role_country import inelastic_ea
        return inelastic_ea()
    except Exception:
        return ("it",)  # last scored call: IT selloff co-move; FR/DE none


EA_BANKS_INELASTIC = _ea_banks_inelastic()
DEST_PEERS = ("us", "uk", "de", "fr", "it")
NM = {
    "us": "US", "uk": "UK", "de": "Germany", "fr": "France",
    "it": "Italy", "es": "Spain", "jp": "Japan",
}
# Display: inelastic stack + elastic stack (realized absorption).
# Stress: elastic demand component runs CUT below last-4Q pace (inelastic sticky).
ELASTIC_CUT = 0.20
ELASTIC_KEEP = 1.0 - ELASTIC_CUT

BUCKETS = ("inelastic", "elastic")
C_NEED = "#111111"
C_GAP = "#E69F00"
C_INE = "#1d4ed8"
C_ELA = "#15803d"
C_ELA_CUT = "#86efac"  # elastic after haircut (panel C)
C_BEFORE = "#64748b"
C_AFTER = "#b91c1c"
GREY = "#94a3b8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _private(net: float, cb: float, cc: str) -> float:
    if not np.isfinite(cb):
        cb = 0.0
    return float(net) if cc == "us" else float(net) - float(cb)


def _jp_stack(gdp: float) -> dict[str, float]:
    js = pd.read_parquet(CACHE / "jsda_investor.parquet")
    js["month"] = pd.to_datetime(js["month"])
    last = js["month"].max()
    cut = last - pd.DateOffset(months=11)
    sub = js[
        js["month"].between(cut, last)
        & js["maturity"].isin(("medium", "long", "super_long"))
    ]
    by = sub.groupby("investor")["net_purchase_jpy_bn"].sum()
    ine = ela = 0.0
    detail: dict[str, float] = {}
    for k, role in ROLE.items():
        v = 100.0 * float(by.get(k, 0.0)) / gdp
        detail[k] = v
        if role == "liability matcher":
            ine += max(v, 0.0)
        else:
            ela += max(v, 0.0)
    return {
        "inelastic": ine, "elastic": ela, "_detail": detail,
        "_tape": f"JSDA coupon {cut:%Y-%m}→{last:%Y-%m}",
    }


def _ea_stack(cc: str, gdp: float) -> dict[str, float]:
    f = eh.shss_panel()
    f = f[f["measure"].eq("F") & f["country"].eq(cc.upper())]
    qs = sorted(f["quarter_end"].unique())[-4:]
    by = f[f["quarter_end"].isin(qs)].groupby("sector")["value_eur_bn"].sum()

    def g(*names: str) -> float:
        return sum(float(by.get(n, 0.0)) for n in names)

    # Per-country BSI bank evidence (not a pooled EA rule): inelastic only where banks buy into
    # cheapening / spread widening without a forward price link. See module docstring.
    banks_inelastic = cc in EA_BANKS_INELASTIC
    banks = g("banks") if "banks" in by.index else g("mfi_ex_cb")
    ine = max(g("insurers"), 0.0) + max(g("pension"), 0.0)
    if banks_inelastic:
        ine += max(banks, 0.0)
    ela_parts = {
        "funds": g("funds", "mmf", "fvc", "ofi"),
        "households": g("households", "nfc"),
        "row": g("row"),
    }
    if not banks_inelastic:
        ela_parts = {"banks": banks, **ela_parts}
    ela = sum(max(v, 0.0) for v in ela_parts.values())
    return {
        "inelastic": 100.0 * ine / gdp,
        "elastic": 100.0 * ela / gdp,
        "banks": 100.0 * max(banks, 0.0) / gdp,
        "banks_inelastic": banks_inelastic,
        "_detail": {**{k: 100.0 * v / gdp for k, v in ela_parts.items()},
                    "banks": 100.0 * banks / gdp,
                    "insurers": 100.0 * g("insurers") / gdp,
                    "pension": 100.0 * g("pension") / gdp},
        "_tape": f"SHSS F {qs[0]:%Y-%m}→{qs[-1]:%Y-%m}",
    }


def _uk_stack(gdp: float) -> dict[str, float]:
    # Banks: no forward predictive link, but buy into curve/ASW cheapening → sticky with ICPF.
    # OFI + overseas are the gilt price-sensitive pair (OFI forward-links y10).
    q = ukh.quarterly_panel()
    lastq = q.index.max()
    u = q.loc[lastq - pd.DateOffset(months=9): lastq]

    def s(col: str) -> float:
        return float(u[col].sum()) if col in u.columns else 0.0

    banks, icpf = s("banks"), s("icpf")
    ofi, overseas, hh = s("ofi"), s("overseas"), s("households")
    return {
        "inelastic": 100.0 * (max(banks, 0.0) + max(icpf, 0.0)) / gdp,
        "elastic": 100.0 * (max(ofi, 0.0) + max(overseas, 0.0) + max(hh, 0.0)) / gdp,
        "banks": 100.0 * max(banks, 0.0) / gdp,
        "banks_inelastic": True,
        "_detail": {
            "banks": 100.0 * banks / gdp, "icpf": 100.0 * icpf / gdp,
            "ofi": 100.0 * ofi / gdp, "overseas": 100.0 * overseas / gdp,
            "households": 100.0 * hh / gdp,
        },
        "_tape": f"ONS {u.index.min():%Y-%m}→{u.index.max():%Y-%m}",
    }


def _us_stack(gdp: float) -> dict[str, float]:
    z = pd.read_parquet(CACHE / "us_z1_sectors.parquet")
    z = z[z["table"].eq("F")].copy()
    z["quarter"] = pd.to_datetime(z["quarter"])
    qs = sorted(z["quarter"].unique())[-4:]
    ine = ela = 0.0
    detail: dict[str, float] = {}
    for buyer, sleeve, code, role in Z1_LINES:
        if sleeve != "all" or role == "official":
            continue
        v = 100.0 * float(
            z[z["series_code"].eq(code) & z["quarter"].isin(qs)]["value_usd_bn"].sum()
        ) / gdp
        detail[buyer] = v
        if role == "liability matcher":
            ine += max(v, 0.0)
        else:
            ela += max(v, 0.0)
    return {
        "inelastic": ine, "elastic": ela, "_detail": detail,
        "_tape": f"Z.1 F.210 {qs[0]:%Y-%m}→{qs[-1]:%Y-%m}",
    }


def _banks_from_detail(detail: dict) -> float:
    """US/Japan: the bank lines of the buyer tape (positive buying only), % GDP."""
    keys = ("bank", "city", "regional", "shinkin", "depository")
    return sum(max(float(v), 0.0) for k, v in detail.items() if any(x in k.lower() for x in keys))


def absorption_stack(cc: str, gdp: float) -> dict[str, float]:
    """Last-4Q realized private buying by evidence role, % GDP."""
    if cc == "jp":
        return _jp_stack(gdp)
    if cc in ("de", "fr", "it", "es"):
        return _ea_stack(cc, gdp)
    if cc == "uk":
        return _uk_stack(gdp)
    if cc == "us":
        return _us_stack(gdp)
    raise KeyError(cc)


# build.py back-compat: "inelastic capacity" alone is the wrong object now.
def inelastic_stack(cc: str, gdp: float) -> dict[str, float]:
    st = absorption_stack(cc, gdp)
    return {
        "banks": 0.0, "insurers": 0.0, "pensions": 0.0, "other": float(st["inelastic"]),
        "_tape": st["_tape"],
    }


elastic_stack = inelastic_stack
ELASTIC_REINVEST = 1.00
# build.py BUCKETS import — keep a harmless triple so old callers don't crash.
# Actual Ex8 chart uses BUCKETS = inelastic/elastic above.


def table() -> pd.DataFrame:
    snap = pd.read_parquet(SNAP).set_index("cc")
    n12 = sens.next12()
    rows = []
    for cc in PEERS:
        r = snap.loc[cc]
        gdp = float(r["gdp_bn"])
        fw_net = float(r["net_issuance_fwd"])
        tr_net = float(r["net_issuance_tr"])
        fw_cb = float(r["fw_cb_net"]) if np.isfinite(r["fw_cb_net"]) else 0.0
        tr_cb = float(r["tr_cb_net"]) if np.isfinite(r["tr_cb_net"]) else 0.0
        if tr_cb and abs(fw_cb) > 2.5 * abs(tr_cb):  # broken forward CB leg (Spain: no ladder CB path) → trailing runoff
            fw_cb = tr_cb
        # Net issuance = gross − redemptions (issuer). Not "private net".
        net_pct = 100.0 * fw_net / gdp
        tr_net_pct = 100.0 * tr_net / gdp
        fw_run = max(-100.0 * fw_cb / gdp, 0.0) if cc != "us" else 0.0
        tr_run = max(-100.0 * tr_cb / gdp, 0.0) if cc != "us" else 0.0
        # Δ private holdings = net − ΔCB (= net + runoff when CB sells).
        hold_fw = 100.0 * _private(fw_net, fw_cb, cc) / gdp
        hold_tr = 100.0 * _private(tr_net, tr_cb, cc) / gdp
        window = str(r["fw_window"])
        if cc in n12.index:
            hold_fw = float(n12.at[cc, "private_net_pct_gdp"])
            window = str(n12.at[cc, "window"])
        st = absorption_stack(cc, gdp)
        ine = float(st["inelastic"])
        ela = float(st["elastic"])
        need_pct = hold_fw
        cover_full = ine + ela
        cover_cut = ine + ELASTIC_KEEP * ela
        gap_full = need_pct - cover_full
        gap_cut = need_pct - cover_cut
        # Share of next holdings that inelastic cannot cover (elastic dependence).
        resid = max(need_pct - ine, 0.0)
        elas_dep = 100.0 * resid / need_pct if need_pct > 0 else np.nan
        # Sensitivities: force the same bank classification in every country (house map is country-specific:
        # UK + IT banks inelastic on country evidence; FR/DE/US/JP banks elastic). Bounds = banks inelastic everywhere vs elastic everywhere.
        banks = float(st.get("banks", _banks_from_detail(st.get("_detail", {}))))
        b_in = bool(st.get("banks_inelastic", False))
        ine_bin = ine if b_in else ine + banks      # banks price-insensitive everywhere
        ine_bout = ine - banks if b_in else ine     # banks price-sensitive everywhere
        elas_dep_broad = 100.0 * max(need_pct - ine_bin, 0.0) / need_pct if need_pct > 0 else np.nan
        elas_dep_strict = 100.0 * max(need_pct - ine_bout, 0.0) / need_pct if need_pct > 0 else np.nan
        rows.append({
            "cc": cc,
            "net_pct": net_pct,
            "tr_net_pct": tr_net_pct,
            "cb_pct": 100.0 * fw_cb / gdp,
            "cb_runoff_pct": fw_run,
            "tr_cb_runoff_pct": tr_run,
            "tr_pct": hold_tr,
            "fw_pct": hold_fw,
            "need_pct": need_pct,
            "inelastic": ine,
            "elastic": ela,
            "absorbed": cover_full,
            "cover_cut": cover_cut,
            "gap_full": gap_full,
            "gap_cut": gap_cut,
            "elas_dep": elas_dep,
            "elas_dep_broad": elas_dep_broad,
            "elas_dep_strict": elas_dep_strict,
            "banks": banks,
            "step": need_pct - hold_tr,
            "step_vs_mix": gap_full,
            "elast_pct": ine,
            "elast_raw": cover_full,
            "short_pct": max(gap_cut, 0.0),
            "hair": need_pct / cover_cut if cover_cut > 0 else np.nan,
            "tape": st["_tape"],
            "fw_window": window,
            "tr_window": str(r["tr_window"]),
            "detail": st.get("_detail", {}),
        })
    return pd.DataFrame(rows).set_index("cc")


def _jp_outbound() -> pd.DataFrame:
    """BoJ regional debt-securities stocks (IIP) and last-4Q Net(Assets) flows."""
    if not JP_OUT.exists():
        raise FileNotFoundError(
            f"missing {JP_OUT}; build from BoJ regbp_q_en.zip + iip_cy_en.zip"
        )
    return pd.read_parquet(JP_OUT)


def _jp_repatriation(df: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Allocate Japan's C-gap across US/UK/DE/FR/IT by IIP debt-stock weights.

    Hit on destination = share × max(jp gap_cut, 0) converted at FX into % of
    destination GDP. Flow column is last-4Q Japan Net(Assets) as % dest GDP
    (positive = Japan still buying there).
    """
    outb = _jp_outbound()
    stocks = outb[outb["kind"].eq("stock") & outb["dest"].isin(DEST_PEERS)].set_index("dest")
    flows = outb[outb["kind"].eq("flow_4q") & outb["dest"].isin(DEST_PEERS)].set_index("dest")
    w = stocks["bn_jpy"].clip(lower=0.0)
    share = w / w.sum()
    jp_gap = max(float(df.at["jp", "gap_cut"]), 0.0)
    jp_gdp = float(snap.loc["jp", "gdp_bn"])
    jp_fx = float(snap.loc["jp", "fx"])
    rows = []
    for cc in DEST_PEERS:
        hit_jp_pp = jp_gap * float(share[cc])
        hit_usd = (hit_jp_pp / 100.0) * jp_gdp * jp_fx
        dest_usd = float(snap.loc[cc, "gdp_bn"]) * float(snap.loc[cc, "fx"])
        hit_pp = 100.0 * hit_usd / dest_usd
        flow_bn = float(flows.at[cc, "bn_jpy"])
        flow_usd = flow_bn * jp_fx
        flow_pp = 100.0 * flow_usd / dest_usd
        gap0 = float(df.at[cc, "gap_cut"])
        rows.append({
            "cc": cc,
            "share": float(share[cc]),
            "stock_bn_jpy": float(w[cc]),
            "flow_pp_dest": flow_pp,
            "hit_pp": hit_pp,
            "gap_before": gap0,
            "gap_after": gap0 + hit_pp,
        })
    # Japan fills: gap goes to zero by construction of the allocation.
    rows.append({
        "cc": "jp",
        "share": 1.0,
        "stock_bn_jpy": float(outb[outb["kind"].eq("stock") & outb["dest"].eq("total")]["bn_jpy"].iloc[0]),
        "flow_pp_dest": np.nan,
        "hit_pp": -jp_gap,
        "gap_before": float(df.at["jp", "gap_cut"]),
        "gap_after": 0.0,
    })
    meta = {
        "flow_start": str(flows["start"].iloc[0]) if "start" in flows.columns else "",
        "flow_end": str(flows["end"].iloc[0]) if "end" in flows.columns else "",
        "stock_year": str(stocks["end"].iloc[0]),
    }
    res = pd.DataFrame(rows).set_index("cc")
    res.attrs["meta"] = meta
    return res


def _order(df: pd.DataFrame) -> list[str]:
    return list(df.sort_values("gap_cut", ascending=False).index)


def chart_job(df: pd.DataFrame, out: Path) -> None:
    """Exhibit 8a — previous vs next Δ holdings (net + CB runoff)."""
    order = _order(df)
    y = np.arange(len(order))
    h = 0.32
    tr_net = np.array([float(df.at[c, "tr_net_pct"]) for c in order])
    tr_run = np.array([float(df.at[c, "tr_cb_runoff_pct"]) for c in order])
    fw_net = np.array([float(df.at[c, "net_pct"]) for c in order])
    fw_run = np.array([float(df.at[c, "cb_runoff_pct"]) for c in order])
    C_NET_P, C_RUN_P = "#94a3b8", "#cbd5e1"
    C_NET_N, C_RUN_N = "#9a3412", "#ea580c"
    cut_pp = int(round(100 * ELASTIC_CUT))

    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    ax.set_title(
        "Private net absorption requirement: trailing 12m vs forward 12m",
        fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    ax.barh(y + h / 2, tr_net, height=h, color=C_NET_P, label="Trailing net issuance")
    ax.barh(y + h / 2, tr_run, height=h, left=tr_net, color=C_RUN_P, label="Trailing CB runoff")
    ax.barh(y - h / 2, fw_net, height=h, color=C_NET_N, label="Forward net issuance")
    ax.barh(y - h / 2, fw_run, height=h, left=fw_net, color=C_RUN_N, label="Forward CB runoff")

    for i in range(len(order)):
        prev = tr_net[i] + tr_run[i]
        nxt = fw_net[i] + fw_run[i]
        ax.text(prev + 0.15, y[i] + h / 2, f"{prev:.1f}", va="center", fontsize=7.5, color="#64748b")
        ax.text(nxt + 0.15, y[i] - h / 2, f"{nxt:.1f}", va="center", fontsize=8,
                color="#111", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([NM[c] for c in order], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(np.nanmax(tr_net + tr_run)), float(np.nanmax(fw_net + fw_run))) * 1.18)
    ax.set_xlabel("% of GDP", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        fontsize=7.5, frameon=False, ncol=4,
    )
    fig.text(
        0.01, 0.015,
        "Δ private holdings = net issuance + CB runoff. Muted = trailing 12m; saturated = forward 12m. "
        f"Country order matches Exhibit 8c (ranked by gap under a {cut_pp}% elastic shortfall).",
        fontsize=6.2, color=GREY, ha="left", va="bottom",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def chart_gaps(df: pd.DataFrame, out: Path) -> None:
    """Exhibit 8c — three gaps: last-4Q, elastic cut, JP sells foreign books."""
    order = _order(df)
    y = np.arange(len(order))
    cut_pp = int(round(100 * ELASTIC_CUT))
    snap = pd.read_parquet(SNAP).set_index("cc")
    rep = _jp_repatriation(df, snap)

    g_full = np.array([float(df.at[c, "gap_full"]) for c in order])
    g_cut = np.array([float(df.at[c, "gap_cut"]) for c in order])
    # countries outside Japan's IIP destinations (Spain) take no repatriation hit: gap stays at the elastic cut
    g_jp = np.array([float(rep.at[c, "gap_after"]) if c in rep.index else float(df.at[c, "gap_cut"]) for c in order])

    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    ax.set_title(
        f"Private absorption shortfall under three demand scenarios",
        fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    ax.axvline(0, color="#cbd5e1", lw=0.8, zorder=0)

    h = 0.22
    ax.barh(y + h, g_full, height=h, color="#94a3b8",
            label="Last-4Q demand repeats (inelastic + elastic)")
    ax.barh(y, g_cut, height=h, color=C_GAP,
            label=f"Elastic demand −{cut_pp}% vs last-4Q (inelastic sticky)")
    ax.barh(y - h, g_jp, height=h, color=C_AFTER,
            label="Japan closes its shortfall via foreign-book sales")

    for i in range(len(order)):
        for yi, val, col in (
            (y[i] + h, g_full[i], "#64748b"),
            (y[i], g_cut[i], "#b45309"),
            (y[i] - h, g_jp[i], "#991b1b"),
        ):
            ax.text(
                val + (0.12 if val >= 0 else -0.12), yi, f"{val:+.1f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=6.8, color=col,
                fontweight="bold" if abs(val) >= 0.5 else "normal",
            )

    ax.set_yticks(y)
    ax.set_yticklabels([NM[c] for c in order], fontsize=10)
    ax.invert_yaxis()
    span = max(float(np.nanmax(np.abs(np.concatenate([g_full, g_cut, g_jp])))), 1.0)
    ax.set_xlim(-span * 1.35, span * 1.45)
    ax.set_xlabel("% of GDP", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        fontsize=7.2, frameon=False, ncol=1,
    )

    meta = rep.attrs.get("meta", {})
    fig.text(
        0.01, 0.015,
        "Shortfall = forward Δ holdings − assumed private cover. "
        f"Amber: elastic demand component {cut_pp}% below its last-4Q pace; inelastic held at last-4Q. "
        "Red: Japan allocates its amber shortfall across foreign debt books by IIP stock weights "
        f"(BoJ IIP end-{meta.get('stock_year', '2025')}, FX-scaled) — base case is domestic GPIF/trust. "
        "Ranked by amber gap.",
        fontsize=5.8, color=GREY, ha="left", va="bottom",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def _figure_html(png: Path, caption: str) -> str:
    b64 = base64.b64encode(png.read_bytes()).decode()
    return (
        f"<figure><img src='data:image/png;base64,{b64}' style='width:100%'/>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _replace_figure(html: str, exhibit: str, block: str,
                    after: str | None = None) -> str:
    cm = re.search(
        rf"<figcaption[^>]*>\s*<b>Exhibit {re.escape(exhibit)}\.?</b>.*?</figcaption>",
        html, re.S,
    )
    if cm:
        fs = html.rfind("<figure>", 0, cm.start())
        fe = html.find("</figure>", cm.end()) + len("</figure>")
        return html[:fs] + block + html[fe:]
    if after:
        am = re.search(
            rf"<figcaption[^>]*>\s*<b>Exhibit {re.escape(after)}\.?</b>.*?</figcaption>",
            html, re.S,
        )
        if am:
            fe = html.find("</figure>", am.end()) + len("</figure>")
            return html[:fe] + "\n" + block + html[fe:]
    raise SystemExit(f"Exhibit {exhibit} missing")


def splice(html: str, png_job: Path, png_gaps: Path, df: pd.DataFrame) -> str:
    jp, uk_, fr = df.loc["jp"], df.loc["uk"], df.loc["fr"]
    cut = int(round(100 * ELASTIC_CUT))
    snap = pd.read_parquet(SNAP).set_index("cc")
    rep = _jp_repatriation(df, snap)
    cap_a = (
        "<b>Exhibit 8a.</b> Private net absorption requirement — trailing vs forward "
        "12m Δ holdings (net issuance + CB runoff), % of GDP. "
        f"Forward: Japan {jp['fw_pct']:.1f}, France {fr['fw_pct']:.1f}, "
        f"Italy {df.at['it', 'fw_pct']:.1f}."
    )
    cap_c = (
        "<b>Exhibit 8c.</b> Private absorption shortfall under three scenarios. "
        f"Last-4Q demand repeats; elastic demand component {cut}% below last-4Q "
        f"(inelastic sticky) — peer shorts <b>FR {fr['gap_cut']:+.1f}</b> / "
        f"<b>IT {df.at['it', 'gap_cut']:+.1f}</b>; "
        f"Japan closes its {jp['gap_cut']:+.1f} shortfall via foreign-book sales "
        f"(FR {rep.at['fr', 'gap_before']:+.1f}→{rep.at['fr', 'gap_after']:+.1f}, "
        f"IT {rep.at['it', 'gap_before']:+.1f}→{rep.at['it', 'gap_after']:+.1f}). "
        f"Base case: JP via GPIF/trust. UK {uk_['elas_dep']:.0f}% dependent on price-sensitive buyers "
        f"vs FR {fr['elas_dep']:.0f}% / IT {df.at['it', 'elas_dep']:.0f}%."
    )
    html = _replace_figure(html, "8a", _figure_html(png_job, cap_a))
    html = _replace_figure(html, "8c", _figure_html(png_gaps, cap_c), after="8a")
    return html


def patch_prose(html: str, df: pd.DataFrame) -> str:
    jp, fr, uk_ = df.loc["jp"], df.loc["fr"], df.loc["uk"]
    cut = int(round(100 * ELASTIC_CUT))
    keep = int(round(100 * ELASTIC_KEEP))
    ranked = sorted(PEERS, key=lambda c: float(df.at[c, "gap_cut"]), reverse=True)
    snap = pd.read_parquet(SNAP).set_index("cc")
    rep = _jp_repatriation(df, snap)

    lead = (
        "<h2>Absorption — foreign sleeve and who clears the book</h2>\n"
        "<p><b>Europe is the foreign-selloff risk; France and Italy are the clearing risk.</b> "
        "Exhibit 8b: non-euro RoW is ~34% of the French stock and ~30% of Germany’s "
        "(SHSS), vs Japan ~15% and a BoJ-heavy book. "
        "Exhibit 8a: forward Δ holdings = net issuance + CB runoff. "
        f"Exhibit 8c: if the elastic demand component runs {cut}% below its last-4Q pace "
        f"(inelastic sticky), peer shorts are <b>FR {fr['gap_cut']:+.1f}</b> / "
        f"<b>IT {df.at['it', 'gap_cut']:+.1f}</b> "
        f"(~{fr['elas_dep']:.0f}% / ~{df.at['it', 'elas_dep']:.0f}% elastic-dependent). "
        f"Japan’s larger shortfall ({jp['gap_cut']:+.1f}) is GPIF/trust at home; "
        f"if instead it sold foreign books, FR {rep.at['fr', 'gap_before']:+.1f}→"
        f"{rep.at['fr', 'gap_after']:+.1f}, IT {rep.at['it', 'gap_before']:+.1f}→"
        f"{rep.at['it', 'gap_after']:+.1f}. "
        f"<b>The UK is the least dependent on price-sensitive buyers</b>: {uk_['elas_dep']:.0f}% of its "
        f"next-12m need, against {fr['elas_dep']:.0f}% for France and {df.at['it', 'elas_dep']:.0f}% for Italy. "
        f"UK banks sit with insurers and pension funds on the price-insensitive side: no forward predictive "
        f"link to yields or ASW, but same-window buying into curve steepening / ASW cheapening "
        f"(LCR / reserve substitutes under QT — the selloff bid, not a clearing price). Italian banks sit "
        f"on the same side (no forward link; buying rises with Bund-spread widening). French and German "
        f"banks sit with funds and foreign buyers on the price-sensitive side (no forward link and no "
        f"selloff co-movement — DE banks sell into selloffs). Move every country's banks onto the same side of that split and the UK is still "
        f"the least dependent ({uk_['elas_dep_broad']:.0f}–{uk_['elas_dep_strict']:.0f}% vs France "
        f"{fr['elas_dep_broad']:.0f}–{fr['elas_dep_strict']:.0f}% and Italy "
        f"{df.at['it', 'elas_dep_broad']:.0f}–{df.at['it', 'elas_dep_strict']:.0f}%). "
        "UK buying over the last four quarters came from banks "
        f"({uk_['detail'].get('banks', float('nan')):.1f}% of GDP), insurers and pension funds "
        f"({uk_['detail'].get('icpf', float('nan')):.1f}), other financials ({uk_['detail'].get('ofi', float('nan')):.1f}) "
        f"and overseas ({uk_['detail'].get('overseas', float('nan')):.1f}) — not the foreign official sector.</p>\n"
    )
    # Stop before the absorption table or first exhibit figure — a bare (?=<figure>) can
    # miss when the section opens with a table (or never rewrite if the h2 title drifted).
    m = re.search(
        r"<h2>Absorption[^<]*</h2>.*?(?=(?:<table>|<figure>))",
        html, re.S,
    )
    if m:
        html = html[:m.start()] + lead + html[m.end():]

    for pat in (
        r"<p>Exhibit 8a\..*?</p>\s*(?=<table>)",
        r"<p>Next-12m private coupon DV01.*?</p>\s*",
        r"<p><b>External vulnerability is who must refinance.*?</p>\s*",
        r"<p>Non-euro-area holders of euro government.*?</p>\s*",
        r"<p><b>France and Italy are the vulnerable absorbers\.</b>.*?</p>\s*",
        r"<p><b>The question is vulnerability.*?</p>\s*",
    ):
        html = re.sub(pat, "", html, count=1, flags=re.S)

    # Order: 8b → 8a → 8c
    figs = {
        lab: re.search(
            rf"<figure>\s*<img[^>]*>\s*<figcaption[^>]*>\s*<b>Exhibit {lab}\.?</b>"
            r".*?</figcaption>\s*</figure>",
            html, re.S,
        )
        for lab in ("8a", "8b", "8c")
    }
    if all(figs.values()):
        blocks = [figs[k].group(0) for k in ("8b", "8a", "8c")]
        lo = min(figs[k].start() for k in figs)
        hi = max(figs[k].end() for k in figs)
        html = html[:lo] + "\n".join(blocks) + html[hi:]

    old_tab = re.search(
        r"<tr><th>Country</th><th>(?:Private net next-12m|New money 100%|"
        r"Private net need|Prev private net|Net issuance|Δ holdings)[^<]*</th>.*?</th></tr>\s*"
        r"(?:<tr>.*?</tr>)+",
        html, re.S,
    )
    if old_tab:
        body = [
            "<tr><th>Country</th><th>Δ holdings</th><th>Inelastic</th>"
            "<th>Elastic</th><th>Price-sensitive share, %</th>"
            f"<th>Gap @{keep}% elastic</th></tr>"
        ]
        for cc in ranked:
            lab = {"us": "US", "uk": "UK", "de": "DE", "fr": "FR", "it": "IT", "jp": "JP"}[cc]
            r = df.loc[cc]
            body.append(
                f"<tr><td>{lab}</td><td>{r['need_pct']:.1f}</td>"
                f"<td>{r['inelastic']:.1f}</td><td>{r['elastic']:.1f}</td>"
                f"<td>{r['elas_dep']:.0f}</td>"
                f"<td>{r['gap_cut']:+.1f}</td></tr>"
            )
        html = html[:old_tab.start()] + "".join(body) + html[old_tab.end():]

    mult_line = (
        f"Absorption (elastic −{cut}%).</b> Peer stress = FR/IT "
        f"({df.at['fr', 'gap_cut']:+.1f} / {df.at['it', 'gap_cut']:+.1f}pp); "
        f"JP gap {df.at['jp', 'gap_cut']:+.1f} filled via GPIF/trust, not peers. "
        "Europe foreign sleeve = Exhibit 8b."
    )
    html = re.sub(
        r"Absorption (?:multiples|\(flow\)|\(inelastic cover\)|"
        r"\(elastic −\d+%\))\.</b>.*?</li>",
        mult_line + "</li>",
        html, count=1, flags=re.S,
    )
    html = re.sub(
        r"<li><b>Absorption\.</b>.*?</li>",
        "<li><b>Absorption.</b> Stock: Europe foreign-selloff risk (8b — FR ~34% / DE ~30% "
        f"non-euro RoW). Flow: FR/IT peer shorts if elastic −{cut}% vs last-4Q "
        f"({df.at['fr', 'gap_cut']:+.1f} / {df.at['it', 'gap_cut']:+.1f}pp); "
        f"JP gap ({df.at['jp', 'gap_cut']:+.1f}) is GPIF/trust, not peers.</li>",
        html, count=1, flags=re.S,
    )
    html = re.sub(
        r"FR/IT (?:most exposed|gaps) at \d+% elastic[^.<]*\.",
        f"FR/IT gaps if elastic −{cut}% "
        f"{df.at['fr', 'gap_cut']:+.1f} / {df.at['it', 'gap_cut']:+.1f}pp "
        f"(JP GPIF-fillable); Europe foreign sleeve 8b.",
        html, count=1,
    )
    # A base that already carries the "gaps if elastic" form: refresh its numbers.
    html = re.sub(
        r"FR/IT gaps if elastic −\d+% [+\-−]?[\d.]+ / [+\-−]?[\d.]+pp",
        f"FR/IT gaps if elastic −{cut}% {df.at['fr', 'gap_cut']:+.1f} / {df.at['it', 'gap_cut']:+.1f}pp",
        html,
    )
    return html


def main() -> None:
    df = table()
    cols = ["need_pct", "inelastic", "elastic", "elas_dep", "gap_full", "gap_cut"]
    print(df[cols].round(2).sort_values("gap_cut", ascending=False).to_string())
    png_job = NOTE / "dm_debt_outlook_absorption.png"
    png_gaps = NOTE / "dm_debt_outlook_absorption_gaps.png"
    chart_job(df, png_job)
    chart_gaps(df, png_gaps)
    print("wrote", png_job)
    print("wrote", png_gaps)
    html = HTML_P.read_text()
    html = splice(html, png_job, png_gaps, df)
    html = patch_prose(html, df)
    HTML_P.write_text(html)
    WHTML(filename=str(HTML_P)).write_pdf(str(PDF))
    import os

    if os.environ.get("NOTE_NO_DESKTOP"):  # unattended runs: a Desktop copy can block on a macOS privacy prompt
        print("pdf", PDF.stat().st_size)
        return
    try:
        shutil.copy2(PDF, DESK)
    except PermissionError:
        import subprocess
        subprocess.run(["cp", str(PDF), str(DESK)], check=False)
    print("pdf", PDF.stat().st_size)


if __name__ == "__main__":
    main()


def figure(out_a: Path | None = None, out_g: Path | None = None):
    """Note-build adapter. Returns absorption chart, gaps chart, table."""
    out_a = Path(out_a) if out_a else (NOTE / "figures" / "ex8a_absorption.png")
    out_g = Path(out_g) if out_g else (NOTE / "figures" / "ex8c_absorption_gaps.png")
    out_a.parent.mkdir(parents=True, exist_ok=True)
    df = table()
    chart_job(df, out_a)
    chart_gaps(df, out_g)
    return out_a, out_g, df
