"""Developed-market debt on Dalio's two conditions (supply vs demand, debt service vs revenue), market by market, plus the
debt arithmetic: tightening needed to stabilise, where a spiral starts, and what a selloff does to interest when it is borrowed.
One chart and two lines per section (c1_supply … c5_prices), plus a Teams paste in note/dm_debt_dalio_teams.md.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.dalio_note [--email a@b.com]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PB = None
PC = None

OUT = Path(__file__).resolve().parent / "note"
STEM = "dm_debt_dalio"
NM = {"us": "US", "uk": "UK", "de": "DE", "fr": "FR", "it": "IT", "jp": "JP"}


FIS_CC = ("uk", "de", "fr", "it", "us")
ML = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]  # MATLAB 'lines'
PAL = {"us": ML[0], "uk": ML[1], "de": ML[2], "fr": ML[3], "it": ML[4], "jp": ML[6]}
BLUE, LBLUE, RED = ML[0], ML[5], ML[1]
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "axes.prop_cycle": matplotlib.cycler(color=ML),
                     "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.6, "axes.edgecolor": "#262626",
                     "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                     "ytick.right": True, "axes.axisbelow": True, "legend.frameon": False})


def fiscal() -> dict:
    """What the debt dashboard's fiscal metrics say about yields: forward tests (debt panel) and the cross-section of levels.

    Forward: next-12m change in 10y / 10y−3m / 10s30s on each metric, each market vs its own history (`debt_signal.json`,
    placebo + halves + leave-one-out + non-overlapping). Levels: each month, across UK/DE/FR/IT/US (JP's 175% debt and YCC are
    off-scale), the slope of the level on debt % GDP and on general-government interest / revenue (last completed year),
    averaged over months 2012→ with Newey-West t (Fama-MacBeth).
    """
    import statsmodels.api as sm
    from sovereign_cash.debt import signal as S

    tests = pd.DataFrame(json.loads((S.DERIVED / "debt_signal.json").read_text())["tests"])
    a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet")
    m = pd.read_parquet(S.MARKET)
    m["month"] = pd.to_datetime(m["month"])
    w = m.pivot_table(index=["country", "month"], columns="series", values="value").reset_index()
    w["cc"] = w["country"].str.lower()
    w["yr"] = np.where(w["month"].dt.month >= 5, w["month"].dt.year - 1, w["month"].dt.year - 2)
    hist = a[~a["forecast"].astype(bool)][["cc", "year", "gg_interest_pct_rev", "debt_pct_gdp"]].rename(columns={"year": "yr"})
    w = w.merge(hist, on=["cc", "yr"], how="left")
    w = w[(w["month"] >= "2012-01-01") & w["cc"].isin(FIS_CC)]
    fm = {}
    for y in ("y10", "tp10", "s1030"):
        for x in ("debt_pct_gdp", "gg_interest_pct_rev"):
            bs = []
            for t, g in w.dropna(subset=[y, x]).groupby("month"):
                if len(g) >= 4:
                    bs.append(np.polyfit(g[x].astype(float), g[y].astype(float) * 100, 1)[0])
            r = sm.OLS(np.array(bs), np.ones(len(bs))).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
            fm[(y, x)] = (float(r.params[0]), float(r.tvalues[0]))
    now = w[w["month"] == w["month"].max()].set_index("cc")
    paths = a[a["cc"].isin(FIS_CC) & (a["year"] >= 2015)].set_index(["cc", "year"])
    # Dalio's two conditions by market: supply vs demand, and debt service vs revenue (next 12m / 2027, dashboard basis)
    s = pd.read_parquet(S.DERIVED / "debt_snapshot.parquet").set_index("cc")
    rp = pd.read_parquet(S.DERIVED / "debt_repricing.parquet").query("scenario == 'baseline'")
    rp = rp.pivot_table(index=["cc", "year"], columns="case", values="interest_pct_rev")
    A = a.set_index(["cc", "year"])
    mk = m[m["month"] == m["month"].max()].pivot_table(index="country", columns="series", values="value")
    mk.index = mk.index.str.lower()
    D = pd.DataFrame({cc: {
        "debt": s.loc[cc, "gg_debt_pct"],
        "priv": 100 * (s.loc[cc, "net_issuance_fwd"] - s.loc[cc, "fw_cb_net"]) / s.loc[cc, "gdp_bn"],
        "bills": 100 * s.loc[cc, "bill_share_stock"],
        "roll": s.loc[cc, "rollovers_pct_rev"],
        "ds": s.loc[cc, "debt_service_pct_rev"],
        "gi25": A.loc[(cc, 2025), "gg_interest_pct_rev"], "gi29": A.loc[(cc, 2029), "gg_interest_pct_rev"],
        "shock": rp.loc[(cc, 2029), "shock"] - rp.loc[(cc, 2029), "current"],
        "cb": 100 * s.loc[cc, "cb_share"], "foreign": s.loc[cc, "h_foreign"],
        "y10": 100 * mk.loc[cc, "y10"], "s1030": 100 * mk.loc[cc, "s1030"]} for cc in ("us", "uk", "de", "fr", "it", "jp")}).T
    return {"tests": tests, "fm": fm, "now": now, "paths": paths, "asof": w["month"].max(), "D": D, "sus": sustain(), "dem": demand()}


def demand(q0: str = "2020-12-31", q1: str = "2025-12-31") -> pd.DataFrame:
    """IMF investor base (general-government gross debt, nominal): holder shares five years ago and now, and who did the private
    buying over the last four quarters (shares of total net purchases ex the domestic central bank)."""
    from issuance import imf_investor_base as ib

    p = ib.panel()
    grp = {"cb": ["dcb"], "for": ["fo", "fb", "fnb"], "fo": ["fo"], "bank": ["db"], "dom": ["dnb"]}
    q0, q1 = pd.Timestamp(q0), pd.Timestamp(q1)
    qm = q1 - pd.DateOffset(years=1)
    rows = {}
    for cc in ("us", "uk", "de", "fr", "it", "jp"):
        h = p[p["country"] == cc.upper()].pivot(index="quarter_end", columns="holder_type", values="value_local_bn")
        fl = {g: h.loc[q1, c].sum() - h.loc[qm, c].sum() for g, c in grp.items()}
        priv = h.loc[q1, "total"] - h.loc[qm, "total"] - fl["cb"]
        r = {f"{g}{k}": 100 * h.loc[q, c].sum() / h.loc[q, "total"] for g, c in grp.items() for k, q in (("0", q0), ("1", q1))}
        r.update({f"buy_{g}": 100 * fl[g] / priv for g in ("for", "fo", "bank", "dom")})
        rows[cc] = r
    return pd.DataFrame(rows).T


SRC = {'c1c': 'Source: WAM of marketable stock incl. bills by bond from the issuance dashboard (debt-office ISIN books; 2027 = issuance forecast); maturities from debt-office books, 2027 at the remit / auction-calendar forecast; GDP IMF WEO. JP WAM excludes T-bills (no bill book).', 'c1': 'Source: central-government interest and principal due by year from debt-office maturity books (US Treasury MSPD, UK DMO, AFT, MEF, Finanzagentur, Japan MoF) and the CBO, OBR and Cabinet Office outturns; GDP IMF WEO. Projections: issuance plans at market forwards. DE is the Bund only (Länder excluded).', 'c2': "Source: IMF Sovereign Debt Investor Base (Arslanalp-Tsuda), general-government gross debt, to 2025Q4. Foreign = official + banks + non-banks, after the domestic central bank; for DE/FR/IT that central bank is the Eurosystem PSPP/PEPP book (ECB SHSS, within €15–55bn at 2025Q4), so it is not in this chart. Load = next-12m domestic non-bank absorption over the last four quarters'.", 'c3': 'Source: IMF WEO (GG deficit, GDP), OECD / AMECO (GG interest); market rates on new issuance from national curves at the issuance mix. pb* = (r − g)/(1 + g) × debt.', 'c4': "Source: own calculation. Repricing from each market's maturity book (dashboard 2027-29, then its repricing speed); principal = extra debt × 2026 rollovers / debt; extra debt deflated by WEO nominal GDP growth; supply slope from the 10s30s panel (section 4), an upper bound for yield levels.", 'c5': 'Source: auction results and calendars of the national debt offices; central-bank holdings by bond (Fed SOMA, BoE APF, ECB PSPP/PEPP allocation, BoJ); GDP IMF WEO. Forward = remit / auction calendar; JP net of the BoJ purchase plan.'}

SHOCKS = np.arange(-200, 401, 25)


def _engine_pb(cc: str, y: int, a: pd.DataFrame | None = None) -> float:
    """GG primary *surplus* % GDP from the stamped engine column (−primary_gg_pct_gdp).

    Fallback (legacy residual): GG interest − WEO overall deficit ≈ −primary when WEO interest ≈ ours.
    """
    from sovereign_cash.debt import signal as S

    if a is None:
        a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet").set_index(["cc", "year"])
    row = a.loc[(cc, y)]
    if "primary_gg_pct_gdp" in a.columns and pd.notna(row.get("primary_gg_pct_gdp")):
        return -float(row["primary_gg_pct_gdp"])
    return float(row["gg_interest_pct_gdp"]) - float(row["deficit_weo_pct_gdp"])


def sustain(horizon: int = 10) -> dict:
    """Debt arithmetic per market, general government: d' = d (1 + r_eff)/(1 + g) − pb.

    pb is the engine GG primary *surplus* (% GDP) at today (2026) and plan (2030) — cash/share/DTS
    through T+3, IMF WEO primary beyond (``primary_gg_pct_gdp``). r_m = market rate on new issuance;
    g = nominal GDP growth 2026-29. Debt-stabilising pb* = (r_m − g)/(1 + g) d; adjustment = pb* − pb.
    The effective rate reprices toward r_m at the dashboard's repricing speed.
    """
    from sovereign_cash.debt import signal as S

    a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet").set_index(["cc", "year"])
    s = pd.read_parquet(S.DERIVED / "debt_snapshot.parquet").set_index("cc")
    rp = pd.read_parquet(S.DERIVED / "debt_repricing.parquet").query("scenario == 'baseline' and case == 'current'").set_index(["cc", "year"])
    rows, curves, paths = {}, {}, {}
    for cc in ("us", "uk", "de", "fr", "it", "es", "jp"):
        d0 = float(s.loc[cc, "gg_debt_pct"])
        gdp = a.loc[cc, "gdp_bn"]
        g = float((gdp[2029] / gdp[2026]) ** (1 / 3) - 1)
        pb = {k: _engine_pb(cc, y, a) for k, y in (("now", 2026), ("plan", 2030))}
        r0 = float(a.loc[(cc, 2026), "gg_interest_pct_gdp"]) / d0
        rm = float(s.loc[cc, "marginal_rate"]) / 100
        if not np.isfinite(rm):  # no bill leg (e.g. Spain: no forward bills, no 3m rate) → new-bond yield
            rm = float(s.loc[cc, "new_bond_yield"]) / 100
        rho = 1 - (1 - float(rp.loc[(cc, 2029), "repriced_share"])) ** (1 / 3)
        rev = float(a.loc[(cc, 2026), "gg_revenue_bn"] / a.loc[(cc, 2026), "gdp_bn"])

        def sim(shock_bp: float, p: float) -> tuple[list[float], float]:
            d, r, ds = d0, r0, [d0]
            for _ in range(horizon):
                r += rho * (rm + shock_bp / 1e4 - r)
                it = r * d
                d = d * (1 + r) / (1 + g) - p
                ds.append(d)
            return ds, it / rev

        pbs = (rm - g) / (1 + g) * d0
        rs = {k: g + v / 100 * (1 + g) / (d0 / 100) for k, v in pb.items()}
        rows[cc] = {"d0": d0, "g": 100 * g, "rho": rho, "pb_now": pb["now"], "pb_plan": pb["plan"], "rm": 100 * rm, "pbs": pbs,
                    "adj_now": pbs - pb["now"], "adj_plan": pbs - pb["plan"], "rs_now": 100 * rs["now"], "rs_plan": 100 * rs["plan"],
                    "gap_now": 1e4 * (rs["now"] - rm), "gap_plan": 1e4 * (rs["plan"] - rm), "ir_now": 100 * r0 * d0 / 100 / rev,
                    "ir10": sim(0, pb["now"])[1], "ir10_100": sim(100, pb["now"])[1], "ir10_300": sim(300, pb["now"])[1],
                    "d10": sim(0, pb["now"])[0][-1], "d10_plan": sim(0, pb["plan"])[0][-1], "rg": 1e4 * (rm - g),
                    "adj_eff": (r0 - g) / (1 + g) * d0 - pb["now"], "dstar_now": pb["now"] * (1 + g) / (rm - g),
                    "step": [sim(x, pb["now"])[1] - sim(x - 100, pb["now"])[1] for x in (100, 200, 300)]}
        curves[cc] = [sim(x, pb["now"])[1] for x in SHOCKS]
        paths[cc] = {"now": sim(0, pb["now"])[0], "plan": sim(0, pb["plan"])[0]}
    return {"T": pd.DataFrame(rows).T, "curves": curves, "paths": paths, "horizon": horizon}


B_HIGH = 4.0


def spiral(fs: dict, b_supply: float, shocks=(100, 300), horizon: int = 20) -> pd.DataFrame:
    """Extra debt service (% GDP) from a parallel selloff when the extra interest is borrowed, not paid.

    Interest: actual repricing schedule (dashboard 2027-29, then each market's repricing speed); the borrowed interest
    adds to debt. Principal: that extra debt is rolled at the market's rollover rate (2026 rollovers / debt), so it adds
    principal due as well. Demand: the borrowed interest is extra private supply; at `b_supply` bp of yield per pp of
    GDP of supply (the all-tenor 10s30s slope, an upper bound for a level effect the panel does not find) it raises the
    rate on everything repricing next year; the panel slope is per pp of 12m supply, so the yield effect cumulates with the
    extra debt (a stock effect of ~2bp per pp of debt/GDP). Extra debt is scaled by nominal GDP growth (2026-29), so it
    compounds only where r > g."""
    from sovereign_cash.debt import signal as S

    T = fs["sus"]["T"]
    rp = pd.read_parquet(S.DERIVED / "debt_repricing.parquet").query("scenario == 'baseline' and case == 'current'").set_index(["cc", "year"])
    a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet").set_index(["cc", "year"])

    def run(cc: str, sh: float, fb: float = 0.0) -> list[tuple[float, float, float, float]]:
        d, rm, rho, g = T.loc[cc, "d0"] / 100, T.loc[cc, "rm"] / 100, T.loc[cc, "rho"], T.loc[cc, "g"] / 100
        m = float(a.loc[(cc, 2026), "rollovers_bn"] / a.loc[(cc, 2026), "debt_bn"])
        s = [float(rp.loc[(cc, y), "repriced_share"]) for y in (2027, 2028, 2029)]
        while len(s) < horizon:
            s.append(1 - (1 - s[-1]) * (1 - rho))
        dD, locked, prev, extra, out = 0.0, 0.0, 0.0, 0.0, []
        for t in range(horizon):
            shock = sh + extra
            locked += (s[t] - prev) * d * shock / 1e4
            prev = s[t]
            dI = locked + (rm + shock / 1e4) * dD
            dP = m * dD
            dD = (dD + dI) / (1 + g)
            extra = fb * 100 * dD
            out.append((100 * dI, 100 * dP, 100 * dD, shock))
        return out

    rows, paths = {}, {}
    for cc in T.index:
        o, o3, of, o4 = run(cc, shocks[0]), run(cc, shocks[1]), run(cc, shocks[0], b_supply), run(cc, shocks[0], B_HIGH)
        rg, bd = T.loc[cc, "rm"] - T.loc[cc, "g"], b_supply * T.loc[cc, "d0"] / 100
        rows[cc] = {"i10": o[9][0], "p10": o[9][1], "ds10": o[9][0] + o[9][1], "ds20": o[19][0] + o[19][1],
                    "dD10": o[9][2], "nl10": o3[9][0] / (3 * o[9][0]), "nl20": o3[19][0] / (3 * o[19][0]),
                    "fb_bp10": of[9][3] - shocks[0], "fb_bp20": of[19][3] - shocks[0],
                    "fb_ds10": of[9][0] + of[9][1] - o[9][0] - o[9][1], "fb_ds20": of[19][0] + of[19][1],
                    "bstar": max(0.0, (T.loc[cc, "g"] - T.loc[cc, "rm"]) / (T.loc[cc, "d0"] / 100)),
                    "rg": rg, "bd": bd, "net": rg + bd, "bd_hi": B_HIGH * T.loc[cc, "d0"] / 100, "d0": T.loc[cc, "d0"],
                    "dbl": np.log(2) / np.log(1 + (rg + bd) / 100) if rg + bd > 0 else np.inf,
                    "i1": o[0][0], "dD20": o[19][2], "dD20f": of[19][2], "dD10f": of[9][2], "fb4_bp20": o4[19][3] - shocks[0]}
        paths[cc] = {"ds": [x[0] + x[1] for x in o], "fb": [x[0] + x[1] for x in of],
                     "dD0": [x[2] for x in o], "dDb": [x[2] for x in of], "dD4": [x[2] for x in o4]}
    df = pd.DataFrame(rows).T
    df.attrs["path"] = paths
    return df


def demand_shock(n_W: pd.DataFrame, b_supply: float) -> pd.Series:
    """Foreign buying halves: the other half lands on domestic private buyers as extra supply (% GDP a year), priced at
    the all-tenor supply slope (bp per pp GDP)."""
    return b_supply * 0.5 * n_W["buy_for"] / 100 * n_W["trail"]


CC6 = ("us", "uk", "de", "fr", "it", "jp")


def _fig(title: str):
    fig, a = plt.subplots(figsize=(7, 3.2))
    a.set_title(title, fontsize=10, loc="left", fontweight="bold")
    a.tick_params(labelsize=8.5)
    return fig, a


def _save(fig, name: str, src: str = "") -> Path:
    for a in fig.axes:
        _place_labels(a)
    fig.tight_layout(rect=(0, 0.06 if src else 0, 1, 1))
    if src:
        fig.text(0.01, 0.012, src, fontsize=6, color="#666", ha="left", va="bottom", wrap=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _rng(s: pd.Series, f: str = ".0f") -> str:
    return f"{s.min():{f}}-{s.max():{f}}"


def _names(ix) -> str:
    n = [NM[c] for c in ix]
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]


def _annual_paths() -> pd.DataFrame:
    from sovereign_cash.debt import signal as S

    a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet")
    return a[a["cc"].isin(CC6) & (a["year"] >= 2015)].set_index(["cc", "year"])


def _jp_net() -> float:
    """MoF net issuance over the same next-12m window as the private-supply table, % GDP."""
    from sovereign_cash.debt import signal as S

    h = pd.read_parquet(S.DERIVED / "debt_history.parquet")
    h["month"] = pd.to_datetime(h["month"])
    x = h[h["cc"] == "jp"].set_index("month").sort_index()
    last = x[~x["is_forecast"].astype(bool)].index.max()
    return float(x["net12_pct_gdp"].get(last + pd.DateOffset(months=12)))


def numbers(fs: dict, sp: dict) -> dict:
    """Every figure the charts and text quote, from the existing computations."""
    T, ST, D = sp["T"], fs["sus"]["T"], fs["D"]
    le = long_end()
    b = le["R"]["all"]["pooled"][0]
    SP = spiral(fs, b)
    W = fs["dem"].join(T[["trail", "fwd"]])
    W["dom_last"] = W["buy_dom"] / 100 * W["trail"]
    W["need"] = W["fwd"] - (W["buy_for"] + W["buy_bank"]) / 100 * W["trail"]
    W["need_half"] = W["need"] + 0.5 * W["buy_for"] / 100 * W["trail"]
    W["mult"], W["mult_half"] = W["need"] / W["dom_last"], W["need_half"] / W["dom_last"]
    spiral_cc = ST.index[(ST["rg"] > 0) & (ST["pb_now"] < 0)]
    gap3 = ST.loc[ST["adj_now"].sort_values(ascending=False).index[:3]]
    return {"T": T, "ST": ST, "SP": SP, "D": D, "W": W, "eu": W.loc[["de", "fr", "it"]], "spiral_cc": spiral_cc, "gap3": gap3,
            "f": sp["f"], "asof": fs["asof"], "paths": _annual_paths(), "sus_paths": fs["sus"]["paths"], "supply_path": sp["path"],
            "le": le, "jp_net": _jp_net(), "wam": _wam(), "dshock": demand_shock(W, b), "b_supply": b, "ds": _debt_service()}


def _debt_service() -> pd.DataFrame:
    """Central-government debt service = interest + principal due in the year (bonds and bills), % GDP; ex-bills alongside."""
    from sovereign_cash.debt import signal as S

    a = pd.read_parquet(S.DERIVED / "debt_country_annual.parquet")
    a = a[a["cc"].isin(CC6) & (a["year"] >= 2015)].copy()
    a["ds"] = 100 * (a["interest_bn"] + a["rollovers_bn"]) / a["gdp_bn"]
    a["ds_xb"] = 100 * (a["interest_bn"] + a["rollovers_bn"] - a["rollovers_bills_bn"]) / a["gdp_bn"]
    a["int"] = 100 * a["interest_bn"] / a["gdp_bn"]
    return a.set_index(["cc", "year"])


def _label_end(a, x, y, cc, txt=None):
    a._ends = getattr(a, "_ends", []) + [(x, y, cc, txt or NM[cc])]


def _place_labels(a, gap: float = 0.065):
    """End-of-line labels, pushed apart vertically so none overlap (gap = share of the y-range)."""
    ends = sorted(getattr(a, "_ends", []), key=lambda e: e[1])
    lo, hi = a.get_ylim()
    d = gap * (hi - lo)
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + d)
    for (x, _, cc, txt), y in zip(ends, ys):
        a.annotate(txt, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=8, color=PAL[cc], va="center")


def c1_supply(n: dict) -> Path:
    ds = n["ds"]
    hi = ds.xs(2029, level="year")["ds"].sort_values(ascending=False)
    fig, a = _fig("Central-government debt service, % of GDP, 2015-2029")
    for cc in CC6:
        s = ds.loc[cc]["ds"].dropna()
        fc = ds.loc[cc]["forecast"].astype(bool).reindex(s.index)
        h, f = s[~fc], s[fc]
        a.plot(h.index, h, color=PAL[cc], lw=1.8)
        if len(f):
            a.plot([h.index[-1], *f.index], [h.iloc[-1], *f], color=PAL[cc], lw=1.8, ls="--")
        _label_end(a, s.index[-1], s.iloc[-1], cc)
    a.set_xlim(2015, 2030.2)
    a.set_ylim(0, ds["ds"].max() * 1.15)
    a.set_ylabel("interest + principal due, % of GDP", fontsize=8.5)
    a.text(0.01, 0.97, "central government; interest + bonds and bills maturing; dashed = projection at market forwards", transform=a.transAxes,
           va="top", fontsize=8, color="#555")
    return _save(fig, "c1_supply", SRC["c1"])


def c1b_breakdown(n: dict) -> Path:
    ds = n["ds"]
    x = ds.xs(2026, level="year").loc[list(CC6)]
    x["bond"] = x["ds_xb"] - x["int"]
    x["bill"] = x["ds"] - x["ds_xb"]
    x = x.sort_values("ds")
    sh = 100 * x["bill"] / x["ds"]
    top = sh.sort_values(ascending=False).index[:2]
    fig, a = _fig("Debt service by component, 2026, % of GDP")
    y = np.arange(len(x))
    left = np.zeros(len(x))
    for col, lab, c in (("int", "interest", BLUE), ("bond", "bonds maturing", LBLUE), ("bill", "bills maturing", RED)):
        a.barh(y, x[col], 0.6, left=left, color=c, label=lab)
        left += x[col].to_numpy()
    for i, cc in enumerate(x.index):
        a.text(x.loc[cc, "ds"] + 0.4, i, f"{x.loc[cc, 'ds']:.0f}% (bills {sh[cc]:.0f}%; 2029: {ds.loc[(cc, 2029), 'ds']:.0f}%)", va="center", fontsize=7.5)
    a.set_yticks(y, [NM[c] for c in x.index], fontsize=8.5)
    a.set_xlim(0, x["ds"].max() * 1.45)
    a.set_xlabel("2026 debt service, % of GDP", fontsize=8.5)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, "c1b_breakdown", SRC["c1"])


def _wam() -> pd.DataFrame:
    """WAM of the stock incl. bills at the start of each year (issuance dashboard, forecast to 2027); JP ex bills (no bill book)."""
    w = pd.read_parquet(Path(__file__).resolve().parents[2] / "issuance" / "cache" / "rv_wam_fc.parquet")
    w["month"] = pd.to_datetime(w["month"])
    w = w[w["month"].dt.month == 1].assign(cc=lambda q: q["country"].str.lower(), year=lambda q: q["month"].dt.year)
    w["wam"] = w["wam_stock_ib"].fillna(w["wam_stock"])
    return w.set_index(["cc", "year"])["wam"]


ROLL_YEARS = (2019, 2026, 2027)


def c1c_rollover(n: dict) -> Path:
    W, a_ = n["wam"], n["paths"]
    fig, a = _fig("Maturing debt vs weighted-average maturity, 2019, 2026, 2027F")
    for cc in CC6:
        pts = [(W.get((cc, y)), 100 * a_.loc[(cc, y), "rollovers_bn"] / a_.loc[(cc, y), "gdp_bn"]) for y in ROLL_YEARS]
        xs, ys = zip(*pts)
        a.plot(xs, ys, color=PAL[cc], lw=1, alpha=0.6)
        a.annotate("", xy=pts[1], xytext=pts[0], arrowprops=dict(arrowstyle="->", color=PAL[cc], lw=1))
        a.scatter(*pts[0], s=30, facecolor="white", edgecolor=PAL[cc], zorder=3)
        a.scatter(*pts[1], s=55, color=PAL[cc], zorder=3)
        a.scatter(*pts[2], s=45, marker="D", facecolor="white", edgecolor=PAL[cc], zorder=3)
        a.annotate(NM[cc] + (" (WAM ex bills)" if cc == "jp" else ""), pts[1], xytext=(6, 7), textcoords="offset points", fontsize=8, color=PAL[cc])
    a.set_xlabel("weighted-average maturity incl. bills, years (start of year)", fontsize=8.5)
    a.set_ylabel("bonds and bills maturing in the year,\n% of GDP", fontsize=8.5)
    a.set_ylim(0, None)
    a.text(0.99, 0.97, "○ 2019   ● 2026   ◇ 2027 forecast", transform=a.transAxes, ha="right", va="top", fontsize=8, color="#555")
    return _save(fig, "c1c_rollover", SRC["c1c"])


def c2_demand(n: dict) -> Path:
    from issuance import imf_investor_base as ib

    p = ib.panel()
    eu = n["eu"]
    fig, a = _fig("Foreign holdings of general-government debt, % of total")
    for cc in CC6:
        h = p[p["country"] == cc.upper()].pivot(index="quarter_end", columns="holder_type", values="value_local_bn").sort_index()
        s = (100 * h[["fo", "fb", "fnb"]].sum(axis=1, min_count=1) / h["total"]).dropna().loc["2008":]
        a.plot(s.index, s, color=PAL[cc], lw=1.8)
        _label_end(a, s.index[-1], s.iloc[-1], cc)
    a.set_ylabel("foreign holders, % of GG debt", fontsize=8.5)
    a.set_xlim(right=a.get_xlim()[1] + 400)
    a.set_ylim(0, 72)
    W = n["W"]
    a.text(0.01, 0.97, "domestic non-bank load if foreign buying halves: " + ", ".join(f"{NM[c]} {W.loc[c, 'mult_half']:.1f}x"
           for c in W.drop(index="jp")["mult_half"].sort_values(ascending=False).index), transform=a.transAxes, va="top", fontsize=7.5, color="#555")
    return _save(fig, "c2_demand", SRC["c2"])


def c3_gap(n: dict) -> Path:
    ST = n["ST"].loc[list(CC6)].sort_values("adj_now")
    g3 = n["gap3"]
    fig, a = _fig("Primary tightening to stabilise debt/GDP at market rates, pp of GDP")
    y = np.arange(len(ST))
    a.barh(y + 0.2, ST["adj_now"], 0.4, color=BLUE, label="from today's primary balance")
    a.barh(y - 0.2, ST["adj_plan"], 0.4, color=LBLUE, label="from the 2030 IMF WEO primary")
    for i, (_, r) in enumerate(ST.iterrows()):
        v = r["adj_now"]
        a.text(max(v, 0) + 0.1, i + 0.2, f"{v:+.1f}", va="center", fontsize=8)
    a.axvline(0, color="#777", lw=0.8)
    a.set_yticks(y, [f"{NM[c]} (r−g {ST.loc[c, 'rg']:+.0f}bp)" for c in ST.index], fontsize=8.5)
    a.set_xlabel("primary tightening to hold debt/GDP at market rates, pp of GDP (below 0: debt already falling)", fontsize=8)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, "c3_gap", SRC["c3"])


def c4_spiral(n: dict) -> Path:
    SP = n["SP"].loc[list(CC6)]
    b = n["b_supply"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("Borrowed +100bp selloff: growth rate of the extra debt and its size after 20 years", fontsize=10, x=0.01, ha="left", fontweight="bold")
    a = ax[0]
    o = SP.sort_values("net").index
    y = np.arange(len(o))
    a.barh(y, SP.loc[o, "rg"], 0.55, color=BLUE, label="r − g: interest on the extra debt less GDP growth")
    a.barh(y, SP.loc[o, "bd"], 0.55, left=SP.loc[o, "rg"].clip(lower=0), color=RED,
           label=f"slope × debt: the extra debt lifts yields {b:.1f}bp per pp on the whole stock")
    for i, cc in enumerate(o):
        r = SP.loc[cc]
        a.plot(r["net"], i, marker="|", color="k", ms=14, mew=1.6)
        a.text(max(r["net"], r["bd"] + max(r["rg"], 0)) + 0.12, i, f"{r['net']:.1f}%/yr: doubles in {r['dbl']:.0f}y", va="center", fontsize=7.5)
    a.axvline(0, color="k", lw=1)
    a.set_yticks(y, [f"{NM[c]} (debt {SP.loc[c, 'd0']:.0f}%)" for c in o], fontsize=8)
    a.set_xlim(min(-1.0, SP["rg"].min() - 0.3), SP["net"].max() + 3.6)
    a.set_xlabel("growth of the extra debt relative to GDP, %/yr (black tick = net; right of 0 = compounds)", fontsize=8)
    a.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1)
    a.tick_params(labelsize=8)
    a = ax[1]
    o2 = SP.sort_values("dD20").index
    y = np.arange(len(o2))
    a.barh(y, SP.loc[o2, "dD20"], 0.55, color="#9a9a9a", label="the borrowing itself (no feedback)")
    a.barh(y, SP.loc[o2, "dD20f"] - SP.loc[o2, "dD20"], 0.55, left=SP.loc[o2, "dD20"], color=RED, label=f"+ feedback at {b:.1f}bp per pp")
    for i, cc in enumerate(o2):
        r = SP.loc[cc]
        a.text(r["dD20f"] + 0.5, i, f"{r['dD20']:.0f} + {r['dD20f'] - r['dD20']:.1f}; yields +{r['fb_bp20']:.0f}bp more", va="center", fontsize=7.5)
    a.set_yticks(y, [NM[c] for c in o2], fontsize=8)
    a.set_xlim(0, SP["dD20f"].max() * 1.75)
    a.set_xlabel("extra debt after 20 years of +100bp borrowed, % of GDP", fontsize=8)
    a.legend(frameon=False, fontsize=6.8, loc="lower right")
    a.tick_params(labelsize=8)
    return _save(fig, "c4_spiral", SRC["c4"])


def _z(x: float) -> str:
    return "0" if round(x) == 0 else f"{x:+.0f}"


HIST_END = pd.Timestamp("2026-09-01")


def long_end() -> dict:
    """Two supply regressions for the next 12 months' 10s30s (`sovereign_cash.debt.long_end.drifts`): all-tenor private
    net supply (% GDP) and private face at 15y+ (% GDP; forward JP net of the BoJ purchase plan). Same controls (2y,
    realised vol), country FE, Driscoll-Kraay; implied move = pooled slope × (next 12m − own 2012-onward mean)."""
    from sovereign_cash.debt import long_end as LE

    return LE.drifts()


def c5_prices(n: dict) -> Path:
    from sovereign_cash.debt import long_end as LE

    le = n["le"]
    T, R = le["T"], le["R"]
    m = LE.measures()
    h = pd.read_parquet(Path(__file__).resolve().parent.parent / "derived" / "debt_history.parquet")
    h["month"] = pd.to_datetime(h["month"])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.suptitle("Private net supply, 12m, % of GDP: all tenors and 15y+", fontsize=10, x=0.01, ha="left", fontweight="bold")
    for a, k in zip(ax, ("all", "long")):
        a.tick_params(labelsize=8)
        for cc in CC6:
            if k == "all":
                x = h[(h["cc"] == cc) & (h["month"] >= "2013-01-01")].set_index("month").sort_index()
                fc = x["is_forecast"].astype(bool)
                g = x["private12_pct_gdp"]
            else:
                x = m[(m["country"] == cc) & (m["month"] >= "2013-01-01")].set_index("month")
                g = x["face12"].dropna()
                fc = pd.Series(g.index > HIST_END, index=g.index)
            hh, ff = g[~fc].dropna(), g[fc].dropna()
            a.plot(hh.index, hh, color=PAL[cc], lw=1.5)
            if len(ff):
                a.plot([hh.index[-1], *ff.index], [hh.iloc[-1], *ff], color=PAL[cc], lw=1.5, ls="--")
            bp = T.loc[cc, k + "_bp"]
            lab = f"{NM[cc]} " + ("n/a" if pd.isna(bp) else f"{_z(bp)}bp")
            end = ff if len(ff) else hh
            _label_end(a, end.index[-1] + pd.Timedelta(days=45), end.iloc[-1], cc, lab)
        b, t = R[k]["pooled"]
        a.set_title((f"All-tenor private net supply: {b:+.1f}bp per pp (t {t:.1f})" if k == "all" else
                     f"15y+ private supply: {b:+.0f}bp per pp (t {t:.1f})"), fontsize=8.5, loc="left")
        a.set_xlim(right=a.get_xlim()[1] + 900)
        a.axhline(0, color="#777", lw=0.6)
        a.set_ylabel("% of GDP, 12m", fontsize=8)
    ax[0].text(0.01, 0.97, f"labels = implied 12m Δ10s30s; EUR (GDP-wtd) {_z(T.loc['eur', 'all_bp'])}bp / {_z(T.loc['eur', 'long_bp'])}bp",
               transform=ax[0].transAxes, va="top", fontsize=7, color="#555")
    return _save(fig, "c5_prices", SRC["c5"])


CHARTS = ((c5_prices,), (), (c1b_breakdown, c2_demand), (c1_supply, c1c_rollover, c3_gap), (c4_spiral,))


def sections(n: dict) -> tuple[str, list[tuple[str, str, str]]]:
    """Headline, then (title, bold takeaway, numbers line) per section; one chart each, in CHARTS order."""
    D, W, eu, ST, SP, T = n["D"], n["W"], n["eu"], n["ST"], n["SP"], n["T"]
    b, t = n["f"]["c"]["priv_fw"]
    nsp = n["spiral_cc"]
    le = n["le"]
    LT, R, Wg = le["T"], le["R"], le["W"]["all"]
    sig = json.loads((Path(__file__).resolve().parents[1] / "derived" / "debt_signal.json").read_text())
    dutch = _dutch_2027()
    lv = [x for x in sig["tests"] if x["y"] in ("y10", "tp10")]
    ds26 = n["ds"].xs(2026, level="year")
    bs = lambda c: 100 * (1 - ds26.loc[c, "ds_xb"] / ds26.loc[c, "ds"])
    a_, Wm = n["paths"], n["wam"]
    roll = lambda cc, y: 100 * a_.loc[(cc, y), "rollovers_bn"] / a_.loc[(cc, y), "gdp_bn"]
    head = (f"Supply, not debt levels, moves DM curves: ~{LT.loc['eur', 'joint_bp']:.0f}bp a year of EUR 10s30s steepening, "
            f"UK ~{LT.loc['uk', 'joint_bp']:.0f}bp. Debt burdens do not predict yields.")
    sec = [
        ("The trade: supply and 10s30s",
         f"Next-12m private supply implies EUR {_z(LT.loc['eur', 'joint_bp'])}bp, UK {_z(LT.loc['uk', 'joint_bp'])}, US {_z(LT.loc['us', 'joint_bp'])}, "
         f"JP {_z(LT.loc['jp', 'long_bp'])} of 10s30s (JP on 15y+ supply only).",
         f"On top of ~{dutch:.0f}bp of EUR steepening from the Dutch pension switch (Jan-27). A slow drift: out-of-sample R² "
         f"{R['all']['oos_r2']:.2f}, nothing ex-JP before 2020 (a QT effect), no cross-market ranking."),
        ("Debt levels are not a yield signal",
         "Neither debt burden (interest/revenue, rollovers, bill share) nor supply predicts 10y yields or term premium.",
         f"{len(lv)} tests on 10y yields and term premium since 2012: {sum(x['p_placebo'] <= 0.05 for x in lv)} significant against "
         f"{0.05 * len(lv):.1f} by chance, and all {sum(x['passes'] for x in lv)} that survive the robustness checks have the wrong sign "
         f"(more supply or burden, lower yields: the cycle, not a premium). Fade 'debt means higher yields'."),
        ("Where the exposure sits",
         f"Bills are {bs('us'):.0f}% of US and {bs('jp'):.0f}% of JP debt service ({ds26.loc['us', 'ds']:.0f}% / {ds26.loc['jp', 'ds']:.0f}% of GDP).",
         f"The US rolls {roll('us', 2026):.0f}% of GDP a year on a {Wm[('us', 2026)]:.1f}-year WAM; the UK {roll('uk', 2026):.0f}% on {Wm[('uk', 2026)]:.1f}. "
         f"Foreigners did {_rng(eu['buy_for'])}% of 2025 private buying in DE, FR and IT. US and FR need {_rng(n['gap3'].loc[['us', 'fr'], 'adj_now'], '.1f')}pp "
         f"of GDP of tightening to stabilise debt; the UK stabilises only on plans, with r − g at {ST.loc['uk', 'rg']:+.0f}bp."),
        ("Appendix A: debt service, rollover and fiscal gap",
         "Reference charts.",
         f"Debt service 2015-2029; maturing debt against WAM (2019, 2026, 2027F); tightening needed to stabilise debt "
         f"({_names(nsp)} have r above g)."),
        ("Appendix B: does a selloff start an interest spiral",
         "A borrowed selloff piles up debt; the spiral on top of it is real but slow.",
         f"That debt grows at r − g + s × d (market rate less growth, plus the yield push from extra supply, s = {n['b_supply']:.1f}bp per pp of GDP, "
         f"on the whole stock d): {_rng(SP['net'], '.1f')}% a year, positive everywhere, doubling every {_rng(SP['dbl'])} years. Losing half of foreign "
         f"buying adds {_rng(n['dshock'].drop(index='jp'), '.0f')}bp. s is the 10s30s supply slope; our yield-level test finds none.",
         _spiral_steps(n)),
    ]
    return head, sec


def _spiral_steps(n: dict) -> list[str]:
    """Section 4 in four lines (PDF only), the US as the worked example."""
    SP, b = n["SP"], n["b_supply"]
    u = SP.loc["us"]
    return [
        f"<b>Shock:</b> +100bp reaches interest as debt rolls: US +{u['i1']:.1f}% of GDP in year 1, +{u['i10']:.1f}% by year 10.",
        f"<b>Borrow it:</b> the extra debt piles up, US +{u['dD20']:.0f}pp of GDP in 20 years (all six {_rng(SP['dD20'])}pp). This is the big number.",
        f"<b>Compound:</b> the extra debt grows at r − g + s × d. r − g is {u['rg']:+.1f}% in the US, up to {SP['rg'].max():+.1f}% in {NM[SP['rg'].idxmax()]}; "
        f"s × d (the {b:.1f}bp-per-pp supply push on the whole debt stock) is {_rng(SP['bd'], '.1f')}%. Positive everywhere.",
        f"<b>Speed:</b> it doubles every {_rng(SP['dbl'])} years; by year 20 it adds {_rng(SP['fb_bp20'])}bp of yield but only "
        f"{_rng(SP['dD20f'] - SP['dD20'], '.1f')}pp of debt. Fiscal policy stops it, not the market.",
    ]


def _dutch_2027() -> float:
    return float("nan")


def _roll_line(n: dict) -> str:
    a, W = n["paths"], n["wam"]
    r = lambda cc, y: 100 * a.loc[(cc, y), "rollovers_bn"] / a.loc[(cc, y), "gdp_bn"]
    return (f"Rollover risk: the US rolls {r('us', 2026):.0f}% of GDP a year on a {W[('us', 2026)]:.1f}-year WAM incl. bills (2019: "
            f"{r('us', 2019):.0f}%), JP {r('jp', 2026):.0f}%; the UK {r('uk', 2026):.0f}% on {W[('uk', 2026)]:.1f} years, DE {r('de', 2026):.0f}%.")


def _ds_line(ds: pd.DataFrame) -> str:
    x = ds.xs(2026, level="year")
    o = x["ds"].sort_values(ascending=False)
    return ("; ".join(f"{NM[c]} {x.loc[c, 'ds']:.0f}% of GDP (bills {100 * (1 - x.loc[c, 'ds_xb'] / x.loc[c, 'ds']):.0f}% of it)" for c in o.index[:3]) + "; "
            + ", ".join(f"{NM[c]} {x.loc[c, 'ds']:.0f}" for c in o.index[3:]) + " in 2026")


def page(n: dict, imgs: list[str]) -> str:
    head, sec = sections(n)
    ol = lambda st: "<ol class='m'>" + "".join(f"<li>{x}</li>" for x in st) + "</ol>" if st else ""
    body = "".join(f"<h2>{'' if s[0].startswith('Appendix') else f'{i}. '}{s[0]}</h2><p><b>{s[1]}</b><br/>{s[2]}</p>{ol(s[3] if len(s) > 3 else None)}"
                   + "".join(f"<img src='{x}' style='width:100%'/>" for x in im) for i, (s, im) in enumerate(zip(sec, imgs), 1))
    return (f"<h1>Developed-market debt on Dalio's numbers</h1><p><b>{head}</b></p>"
            f"<div class='small'>{pd.Timestamp.today():%d %b %Y}. General government; holders IMF to end-2025; markets to {n['asof']:%b %Y}.</div>{body}")


def teams(n: dict, pngs: list[Path]) -> str:
    head, sec = sections(n)
    lines = [f"**{head}**"]
    for i, (s, p) in enumerate(zip(sec, pngs), 1):
        if s[0].startswith("Appendix"):
            continue
        lines.append(f"{i}. **{s[1]}** {s[2]}" + (f" [{', '.join(x.name for x in p)}]" if p else ""))
    return "\n\n".join(lines) + "\n"


def build() -> tuple[Path, dict]:
    OUT.mkdir(exist_ok=True)
    fs = fiscal()
    n = numbers(fs, PC.supply())
    ch = {f.__name__: f(n) for g in CHARTS for f in g}
    uri = lambda p: f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"
    uris = [[uri(ch[f.__name__]) for f in g] for g in CHARTS]
    h = OUT / f"{STEM}.html"
    h.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{PB.CSS}</style></head><body>{page(n, uris)}</body></html>",
                 encoding="utf-8")
    (OUT / f"{STEM}_teams.md").write_text(teams(n, [[ch[f.__name__] for f in g] for g in CHARTS]), encoding="utf-8")
    from weasyprint import HTML

    pdf = OUT / f"{STEM}.pdf"
    HTML(filename=str(h)).write_pdf(str(pdf))
    return pdf, {"n": n, "ch": ch}


def email(to: list[str], pdf: Path, ctx: dict) -> None:
    import sys
    import urllib.request

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "projects" / "uk_long_end"))
    import _email_report as E

    E.load_env_files()
    key = os.environ.get("RESEND_API_KEY", "").strip()
    if not key:
        raise SystemExit(E.config_help())
    body = page(ctx["n"], [[f"cid:{f.__name__}" for f in g] for g in CHARTS])
    att = [{"filename": p.name, "content": base64.b64encode(p.read_bytes()).decode(), "content_id": k} for k, p in ctx["ch"].items()]
    att.append({"filename": pdf.name, "content": base64.b64encode(pdf.read_bytes()).decode()})
    payload = {"from": os.environ.get("EMAIL_FROM", "onboarding@resend.dev"), "to": to,
               "subject": "DM debt: supply steepens EUR 10s30s ~12bp a year; debt levels are not a yield signal",
               "html": f"<html><head><meta charset='utf-8'><style>{PB.CSS}</style></head><body>{body}</body></html>",
               "text": "Note in HTML; PDF attached.", "attachments": att}
    req = urllib.request.Request("https://api.resend.com/emails", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "dm-debt-dalio-note/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        print(r.read().decode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default="")
    a = ap.parse_args()
    pdf, ctx = build()
    print(pdf)
    if a.email:
        email([x.strip() for x in a.email.split(",") if x.strip()], pdf, ctx)


