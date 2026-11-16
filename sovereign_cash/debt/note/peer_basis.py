"""Peer-basis numbers for the note: general-government interest / revenue ranking (US federal), r − g on the
IMF and on our book, OECD net liabilities, OAT–Bund / BTP–Bund betas, and the IMF / Commission cross-check table.

Every number comes from the same data the exhibits use: derived/debt_country_annual.parquet (our book, central and
general government), data/debt/ameco.parquet (Commission), data/debt/weo.parquet (IMF), data/debt/gov_zeros.parquet
(10y), issuance/cache/egb_daily.parquet (spreads: FR−DE / IT−DE 10y, not BdF TEC10 − BBK).
``numbers()`` feeds note_numbers.json; ``peer_table()`` and ``crosscheck_table()`` are the two HTML tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.debt import page as dpage
from sovereign_cash.debt.note import service_rev

NOTE = Path(__file__).resolve().parent
SC = NOTE.parents[1]
REPO = SC.parent
PANEL = ("us", "uk", "de", "fr", "it", "es", "jp", "se")
NM = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy", "es": "Spain", "jp": "Japan",
      "se": "Sweden", "no": "Norway"}


def spreads() -> dict:
    e = pd.read_parquet(REPO / "issuance" / "cache" / "egb_daily.parquet")
    y = e[e["series"] == "y10"].pivot_table(index="date", columns="country", values="value").sort_index()
    y.index = pd.to_datetime(y.index)
    s = pd.DataFrame({"oat": 100 * (y["FR"] - y["DE"]), "btp": 100 * (y["IT"] - y["DE"])}).dropna()
    w = s.resample("W-FRI").last().diff().dropna()

    def beta(a, b):
        x = w.loc[a:b]
        if len(x) < 8:  # history missing (e.g. the BdI BTP series came back truncated): no beta rather than a crash
            return float("nan"), float("nan")
        return float(np.polyfit(x["oat"], x["btp"], 1)[0]), float(x["oat"].corr(x["btp"]))
    lv = s.resample("W-FRI").last()
    jun24 = (lv.loc["2024-07-05"] - lv.loc["2024-06-07"]) if {pd.Timestamp("2024-06-07"), pd.Timestamp("2024-07-05")} <= set(lv.index) \
        else pd.Series({"oat": np.nan, "btp": np.nan})
    last = s.index.max()
    lvl = s.loc[last] - s.loc[:"2026-06-30"].iloc[-1]
    return {"b_since_jun": beta("2026-06-30", last), "b_2425": beta("2024-01-01", "2025-12-31"),
            "b_long": beta("2012-01-01", last), "jun24_ratio": float(jun24["btp"] / jun24["oat"]),
            "lvl_ratio": float(lvl["btp"] / lvl["oat"]), "lvl_oat": float(lvl["oat"]), "lvl_btp": float(lvl["btp"]),
            "asof": last, "oat_last": float(s["oat"].iloc[-1]), "oat_peak": float(s.loc["2026-10-01":, "oat"].max())}


def data() -> dict:
    ca = pd.read_parquet(SC / "derived" / "debt_country_annual.parquet")
    y_fc = int(ca.loc[ca["forecast"], "year"].min())
    peers = dpage.peer_rows(ca, y_fc, PANEL)
    a = pd.read_parquet(SC / "data" / "debt" / "ameco.parquet").pivot_table(
        index=["country", "year"], columns="var", values="value")
    w = pd.read_parquet(SC / "data" / "debt" / "weo.parquet").pivot_table(
        index=["country", "year"], columns="var", values="value")
    am = (100 * a["interest_gross_pct"] / a["revenue_pct"])
    we = (100 * w["interest_pct"] / w["revenue_pct"])
    nf = pd.read_csv(SC / "data" / "debt" / "oecd_net_fin_liab.csv")
    iso = {"DEU": "de", "ESP": "es", "FRA": "fr", "GBR": "uk", "ITA": "it", "JPN": "jp", "SWE": "se", "USA": "us"}
    nf = nf[nf["REF_AREA"].isin(iso)].assign(cc=lambda d: d["REF_AREA"].map(iso))
    net = nf[nf["TIME_PERIOD"] == y_fc].set_index("cc")["OBS_VALUE"].astype(float).to_dict()
    net_vint = "OECD " + str(nf["STRUCTURE_NAME"].iloc[0]) if "STRUCTURE_NAME" in nf else "OECD Economic Outlook"
    return {"ca": ca, "y_fc": y_fc, "peers": {r["cc"]: r for r in peers}, "rank": [r["cc"] for r in peers],
            "ameco": am, "weo": we, "w": w, "net": net, "net_vint": net_vint,
            "bench": service_rev.bench_frame()}


def _f(v, d=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}".replace("-", "−")


def peer_table(D: dict) -> str:
    """Exhibit 2's ranking table (general government, US federal; central ratios in grey)."""
    return dpage._peer_table(D["ca"], D["y_fc"], PANEL)


def crosscheck_table(D: dict) -> str:
    """Appendix A1 table: ours vs Commission (AMECO gross) and IMF (WEO net) interest / revenue."""
    ca, y_fc = D["ca"], D["y_fc"]
    rows = []
    for cc in D["rank"]:
        p = D["peers"][cc]
        g = ca[(ca["cc"] == cc)].set_index("year")
        col = "interest_pct_rev" if cc == "us" else "gg_interest_pct_rev"
        ours27 = float(g.at[2027, col]) if 2027 in g.index else np.nan
        am27 = float(D["ameco"].get((cc, 2027), np.nan))
        we27 = float(D["weo"].get((cc, 2027), np.nan))
        we30 = float(D["weo"].get((cc, 2030), np.nan))
        ig30 = float(g.at[2030, "gg_interest_pct_gdp"]) if 2030 in g.index else np.nan
        rows.append(f"<tr><td>{NM[cc]}{' (federal)' if cc == 'us' else ''}</td><td><b>{_f(ours27)}</b></td>"
                    f"<td>{_f(am27)}</td><td>{_f(we27)}</td><td>{_f(we30)}</td><td>{_f(ig30, 2)}</td>"
                    f"<td>{_f(p['debt_gg'], 0)}</td><td>{_f(p['rg_imf'], 2)}</td><td>{_f(p['rg_book'], 2)}</td></tr>")
    return (
        "<table><tr><th>Ranked by 2030</th><th>Ours 2027, %</th><th>Commission 2027, %</th><th>IMF 2027, %</th>"
        "<th>IMF 2030, %</th><th>Ours GG interest / GDP 2030, %</th><th>GG debt % GDP (IMF)</th>"
        f"<th>r − g {y_fc}–30, IMF</th><th>r − g {y_fc}–30, our book</th></tr>" + "".join(rows) + "</table>")


def numbers(D: dict, S: dict | None = None) -> dict:
    """Flat key → value map for the note text (peer ranking, UK framing, consensus deltas, spreads)."""
    P, N, b = D["peers"], D["net"], D["bench"]
    ye = P["uk"]["y_end"]
    out: dict = {"y_fc": D["y_fc"], "y_end": ye, "net_vint": D["net_vint"]}
    for cc, r in P.items():
        out[f"ir_now_{cc}"] = r["ir_now"]
        out[f"ir_end_{cc}"] = r["ir_end"]
        out[f"ir_move_{cc}"] = r["ir_end"] - r["ir_now"]
        out[f"debt_gg_{cc}"] = r["debt_gg"]
        out[f"rg_imf_{cc}"] = r["rg_imf"]
        out[f"rg_book_{cc}"] = r["rg_book"]
        out[f"central_ir_{cc}"] = r["central_now"]
        out[f"rev_share_{cc}"] = min(r["rev_share"], 100.0) if np.isfinite(r["rev_share"]) else np.nan
        out[f"y10_{cc}"] = r["y10"]
    for cc, v in N.items():
        out[f"net_liab_{cc}"] = v
    out["rank_top4"] = ", ".join(f"{NM[c]}{' (federal)' if c == 'us' else ''} {P[c]['ir_end']:.1f}%" for c in D["rank"][:4])
    moves = sorted(((c, P[c]["ir_end"] - P[c]["ir_now"]) for c in PANEL if c in P), key=lambda x: -x[1])
    out["ir_moves_ex_uk"] = ", ".join(f"{NM[c]} {d:+.1f}" for c, d in moves if c not in ("uk", "se"))
    out["gilt_bund_bp"] = 100 * (P["uk"]["y10"] - P["de"]["y10"])
    jg = D["ca"][D["ca"]["cc"] == "jp"].set_index("year")
    out["jp_gg_ir_now"] = float(jg["gg_interest_pct_rev"].get(D["y_fc"], np.nan))
    out["jp_gg_ir_2030"] = float(jg["gg_interest_pct_rev"].get(2030, np.nan))
    # Net of public financial assets (OECD NFL; GPIF on the asset side): scale interest / revenue
    # by net liabilities / gross GG debt at the forecast origin. Gross coupons are unchanged;
    # the scale is the stock netting applied to the burden ratio.
    gross_jp = float(P["jp"]["debt_gg"]) if "jp" in P and np.isfinite(P["jp"].get("debt_gg", np.nan)) else np.nan
    net_jp = float(N.get("jp", np.nan))
    scale = (net_jp / gross_jp) if np.isfinite(gross_jp) and gross_jp > 0 and np.isfinite(net_jp) else np.nan
    out["jp_net_scale"] = scale
    out["jp_gross_debt"] = gross_jp
    out["jp_gg_ir_now_net"] = out["jp_gg_ir_now"] * scale if np.isfinite(scale) else np.nan
    out["jp_gg_ir_2030_net"] = out["jp_gg_ir_2030"] * scale if np.isfinite(scale) else np.nan
    out["jp_gg_ir_move_net"] = (
        out["jp_gg_ir_2030_net"] - out["jp_gg_ir_now_net"]
        if np.isfinite(out["jp_gg_ir_2030_net"]) and np.isfinite(out["jp_gg_ir_now_net"]) else np.nan
    )
    jc = jg["interest_pct_rev"]
    out["jp_cg_ir_now"] = float(jc.get(D["y_fc"], np.nan))
    out["jp_cg_ir_2030"] = float(jc.get(2030, np.nan))
    out["jp_cg_ir_now_net"] = out["jp_cg_ir_now"] * scale if np.isfinite(scale) else np.nan
    out["jp_cg_ir_2030_net"] = out["jp_cg_ir_2030"] * scale if np.isfinite(scale) else np.nan
    out["jp_cg_ir_move_net"] = (
        out["jp_cg_ir_2030_net"] - out["jp_cg_ir_now_net"]
        if np.isfinite(out["jp_cg_ir_2030_net"]) and np.isfinite(out["jp_cg_ir_now_net"]) else np.nan
    )
    out["fr_ameco_ir_2025"] = float(D["ameco"].get(("fr", 2025), np.nan))
    out["fr_ameco_ir_2027"] = float(D["ameco"].get(("fr", 2027), np.nan))
    out.update(service_rev.bench_numbers(b))
    if S:
        out.update({"oat_last_bp": S["oat_last"], "oat_asof": f"{S['asof']:%d %b %Y}", "oat_peak_bp": S["oat_peak"],
                    "btp_beta_since_jun": S["b_since_jun"][0], "btp_corr_since_jun": S["b_since_jun"][1],
                    "btp_beta_2425": S["b_2425"][0], "btp_beta_long": S["b_long"][0],
                    "btp_ratio_jun24": S["jun24_ratio"], "btp_ratio_level": S["lvl_ratio"],
                    "oat_move_since_jun_bp": S["lvl_oat"], "btp_move_since_jun_bp": S["lvl_btp"]})
    return out
