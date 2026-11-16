"""Full 2011 EA clearing map vs France 2026 — calibrated on public series in this repo.

Layers
------
1. Spreads (daily 10y vs Bund) — ``issuance/cache/egb_daily.parquet``
2. Foreign sleeve (quarterly) — IMF Arslanalp–Tsuda + ECB QSA non-resident share
3. Domestic bank absorption (monthly) — ECB BSI MFIs ex NCB, domestic GG debt
4. Auction cover — FR/IT ``ops_*.parquet``; ES Tesoro ``historico_subastas.xlsx``
5. Official bid (weekly) — ECB ILM securities held for monetary-policy purposes
   (SMP-era proxy; country stock only from end-2012 transparency)
6. Bank equity stress — iShares EURO STOXX Banks (EXV1.DE) via Yahoo
7. National holder tapes — BdF DET (FR), BoI TCCE0200 (IT), BdE be2244 (ES, 2016+)
8. Event markers — SMP / LTRO / OMT / FR 2026 tape gates

Run: ``PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.note.analogue_2011_map``
Writes ``analogue_2011_map.json``, chart PNGs under ``note/``, cache under
``issuance/cache/analogue_2011/``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from issuance import ecb_holders as eh
from issuance import imf_investor_base as imf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CACHE = REPO / "issuance" / "cache" / "analogue_2011"
OPS = REPO / "issuance" / "cache"
ES_XLSX = REPO / "data" / "raw" / "es" / "tesoro" / "historico_subastas.xlsx"
EGB = REPO / "issuance" / "cache" / "egb_daily.parquet"
OUT_JSON = HERE / "analogue_2011_map.json"
CHART_PATH = HERE / "dm_debt_outlook_analogue_2011.png"
CHART_ABSORB = HERE / "dm_debt_outlook_analogue_absorb.png"
CHART_HOLDERS = HERE / "dm_debt_outlook_analogue_holders_ites.png"
CHART_BDF = HERE / "dm_debt_outlook_analogue_bdf_det.png"

_ES_MONTHS = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}

EVENTS_2011 = [
    ("2010-05-10", "SMP start"),
    ("2011-08-08", "SMP reactivated (IT/ES)"),
    ("2011-11-12", "Berlusconi resigns"),
    ("2011-11-17", "FR–Bund peak 208bp"),
    ("2011-12-21", "3y LTRO #1"),
    ("2012-02-29", "3y LTRO #2"),
    ("2012-07-26", "Draghi ‘whatever it takes’"),
    ("2012-09-06", "OMT announced"),
]
EVENTS_2026 = [
    ("2026-06-30", "OAT–Bund baseline 66bp"),
    ("2026-10-22", "Tape ~175bp (Europe-problem gate)"),
]


# ---------------------------------------------------------------- series
def spreads_daily() -> pd.DataFrame:
    egb = pd.read_parquet(EGB)
    y10 = egb[egb["series"] == "y10"].pivot_table(index="date", columns="country", values="value")
    y10.index = pd.to_datetime(y10.index)
    out = pd.DataFrame({
        "FR": (y10["FR"] - y10["DE"]) * 100,
        "IT": (y10["IT"] - y10["DE"]) * 100,
        "ES": (y10["ES"] - y10["DE"]) * 100,
    }).dropna(how="all")
    return out.sort_index()


def imf_foreign_private() -> pd.DataFrame:
    w = imf.wide().copy()
    w["quarter_end"] = pd.to_datetime(w["quarter_end"])
    w["fpriv"] = w["fb"] + w["fnb"]
    w["d4_fpriv_gdp"] = w.groupby("country")["fpriv"].transform(lambda s: s - s.shift(4)) / w["gdp"] * 100
    w["fpriv_share"] = 100 * w["fpriv"] / w["total"]
    w["db_share"] = 100 * w["db"] / w["total"]
    w["dcb_share"] = 100 * w["dcb"] / w["total"]
    return w


def qsa_nonres() -> pd.DataFrame:
    q = eh.qsa_raw()
    le = q[q["measure"] == "LE"]
    w = le.pivot_table(index=["country", "quarter_end"], columns="sector", values="value_eur_bn")
    w["nonres_share"] = 100 * w["nonres"] / w["total"]
    return w.reset_index()


def bsi_banks() -> pd.DataFrame:
    return eh.bsi_panel()


def fr_it_cover(cc: str, start: str, end: str) -> dict:
    ops = pd.read_parquet(OPS / f"ops_{cc}.parquet")
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["btc"] = pd.to_numeric(ops["btc"], errors="coerce")
    if cc == "it":
        ops.loc[ops["btc"] > 20, "btc"] = ops.loc[ops["btc"] > 20, "btc"] / 100.0
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    sub = ops[(ops["op_dt"] >= start) & (ops["op_dt"] <= end) & ops["btc"].notna() & (ops["amount_issued"] > 0)]
    if "resid_mat_yrs" in sub.columns:
        ml = sub[pd.to_numeric(sub["resid_mat_yrs"], errors="coerce") >= 5]
    else:
        ml = sub
    if ml.empty:
        ml = sub
    sw = float((ml["btc"] * ml["amount_issued"]).sum() / ml["amount_issued"].sum()) if ml["amount_issued"].sum() else np.nan
    return {
        "n": int(len(ml)),
        "median": float(ml["btc"].median()) if len(ml) else None,
        "size_weighted": sw,
        "p10": float(ml["btc"].quantile(0.1)) if len(ml) else None,
    }


def es_cover_from_tesoro(years: list[int]) -> pd.DataFrame:
    """Tesoro historico_subastas: Excel mis-years the FECHA column — replace year with sheet year."""
    rows = []
    if not ES_XLSX.exists():
        return pd.DataFrame(columns=["op_dt", "sleeve", "btc", "issued", "bid"])
    xl = pd.ExcelFile(ES_XLSX)
    for year in years:
        name = str(year) if year != 2015 else "2015 "
        if name not in xl.sheet_names and str(year) not in xl.sheet_names:
            continue
        sheet = name if name in xl.sheet_names else str(year)
        df = pd.read_excel(ES_XLSX, sheet_name=sheet, header=None)
        for _, row in df.iterrows():
            raw = pd.to_datetime(row[0], errors="coerce")
            if pd.isna(raw):
                continue
            d = pd.Timestamp(year=year, month=int(raw.month), day=min(int(raw.day), 28 if raw.month == 2 else 30))
            try:
                d = pd.Timestamp(year=year, month=int(raw.month), day=int(raw.day))
            except ValueError:
                continue
            for sleeve, bid_i, adj_i in (("short", 2, 4), ("long", 8, 10)):
                bid = pd.to_numeric(row[bid_i], errors="coerce")
                bid2 = pd.to_numeric(row[bid_i + 1], errors="coerce")
                adj = pd.to_numeric(row[adj_i], errors="coerce")
                if pd.isna(adj) or adj <= 0:
                    continue
                total_bid = (bid if pd.notna(bid) else 0) + (bid2 if pd.notna(bid2) else 0)
                if total_bid <= 0:
                    continue
                rows.append({"op_dt": d, "sleeve": sleeve, "bid": float(total_bid),
                             "issued": float(adj), "btc": float(total_bid / adj), "year": year})
    return pd.DataFrame(rows)


def es_cover_stats(start: str, end: str) -> dict:
    a = es_cover_from_tesoro(list(range(2010, 2014)) + list(range(2024, 2027)))
    if a.empty:
        return {"n": 0, "median": None, "size_weighted": None, "p10": None}
    sub = a[(a["op_dt"] >= start) & (a["op_dt"] <= end) & (a["sleeve"] == "long")]
    if sub.empty:
        return {"n": 0, "median": None, "size_weighted": None, "p10": None}
    sw = float((sub["btc"] * sub["issued"]).sum() / sub["issued"].sum())
    return {"n": int(len(sub)), "median": float(sub["btc"].median()),
            "size_weighted": sw, "p10": float(sub["btc"].quantile(0.1))}


def ilm_weekly() -> pd.DataFrame:
    p = CACHE / "ilm_monpol_weekly.parquet"
    if p.exists():
        return pd.read_parquet(p)
    csv = CACHE / "ilm_monpol_sec.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["date", "stock_mn", "d_bn"])
    ilm = pd.read_csv(csv)
    ilm["date"] = pd.to_datetime(ilm["TIME_PERIOD"] + "-5", format="%G-W%V-%u", errors="coerce")
    ilm["stock_mn"] = pd.to_numeric(ilm["OBS_VALUE"])
    ilm = ilm.dropna(subset=["date"]).sort_values("date")
    ilm["d_bn"] = ilm["stock_mn"].diff() / 1000.0
    return ilm[["date", "stock_mn", "d_bn"]]


def smp_end2012_by_issuer() -> dict:
    """End-2012 nominal SMP stock by issuer (first transparency print). FR was not purchased."""
    fallback = {"IT": 102.8, "ES": 44.3, "IE": 14.2, "GR": 33.9, "PT": 22.8, "FR": 0.0, "total": 218.0}
    p = CACHE / "SMP_breakdown_history.csv"
    if not p.exists():
        return fallback
    # Rows look like: Italy,Nominal amount,102.8,89.7,... (2012 is first numeric col)
    name = {"Ireland": "IE", "Greece": "GR", "Spain": "ES", "Italy": "IT", "Portugal": "PT"}
    out: dict[str, float] = {}
    for ln in p.read_text(encoding="utf-8-sig").splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 3:
            continue
        if parts[0] in name and parts[1].lower().startswith("nominal"):
            try:
                out[name[parts[0]]] = float(parts[2])
            except ValueError:
                continue
    if len(out) < 3:
        return fallback
    out["FR"] = 0.0
    out["total"] = round(sum(v for k, v in out.items() if k != "total"), 1)
    return out


def banks_etf() -> pd.Series:
    p = CACHE / "exv1_banks.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_parquet(p)["close"]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def bdf_detention_panel() -> pd.DataFrame:
    p = CACHE / "bdf_detention_comptespublics.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    d.index = pd.to_datetime(d.index)
    return d.sort_index()


def bdf_detention_snapshot(quarters: list[str]) -> dict:
    """Quarters as '2011Q4' → period start 2011-10-01 (BdF DET is quarterly)."""
    d = bdf_detention_panel()
    if d.empty:
        return {}
    qmap = {"Q1": "-01-01", "Q2": "-04-01", "Q3": "-07-01", "Q4": "-10-01"}
    out: dict = {
        "source": "Banque de France DET via comptespublics.fr/series/detention-dette-etat/donnees.csv (+ Webstat DET2 OAT amounts)",
    }
    keys = (
        "nonres_pct", "nonres_oat_pct", "banques_oat_pct", "banque_centrale_oat_pct",
        "assurances_retraite_oat_pct", "encours_oat_md", "nonres_md",
    )
    for q in quarters:
        y, qq = q[:4], q[4:]
        ts = pd.Timestamp(y + qmap[qq])
        if ts not in d.index:
            continue
        r = d.loc[ts]
        out[q] = {k: round(float(r[k]), 2) for k in keys if k in d.columns and pd.notna(r[k])}
    out["read"] = (
        "FR nonres OAT share fell 61.8% (2010Q1) → 54.5% (2011Q1) → 55.2% (2011Q4) — a mild "
        "foreign step-back, not the IT/ES collapse. Stock nonres grew with the stock. Banks on "
        "OAT stuck ~13%. Central-bank OAT share 2.5% → 3.2% in 2011 vs 18–21% in 2025-26."
    )
    return out


def oat_bund_official_block() -> dict:
    p = CACHE / "oat_bund_official_bp.parquet"
    if not p.exists():
        return {}
    s = pd.read_parquet(p)["oat_bund_bp"]
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().dropna()
    peak_2011 = s.loc["2011"]
    return {
        "source": "BdF TEC10 (Webstat) − Bundesbank BBSIS 10y federal zero",
        "2011_peak_bp": round(float(peak_2011.max()), 1),
        "2011_peak_date": str(peak_2011.idxmax().date()),
        "2026_06_30_bp": round(float(s.loc[: "2026-06-30"].iloc[-1]), 1) if len(s.loc[: "2026-06-30"]) else None,
        "2026_10_22_bp": round(float(s.loc["2026-10-22"]), 1) if "2026-10-22" in s.index else None,
        "2026_10_23_bp": round(float(s.loc["2026-10-23"]), 1) if "2026-10-23" in s.index else None,
        "note": "Desk tape ~175 on 22-Oct was ahead of the official close. Use official for the map.",
    }


def eba_domestic_block() -> dict:
    p = CACHE / "eba_2013_domestic_sovereign_share.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    shares = {cc: {d: v[2] for d, v in dates.items()} for cc, dates in raw.items()}
    return {
        "source": "EBA 2013 EU-wide Transparency Summary Report Table 2 (matched to 2011 stress-test Dec-2010)",
        "domestic_bank_share_pct_of_sovereign_held_by_eba_sample": shares,
        "read": (
            "IT domestic-bank share of EBA-sample sovereign rose 59%→68%→75% (Dec-10→Dec-11→Dec-12) "
            "as foreign left. ES 78→83→86. FR 61→59→69 — FR banks did not absorb in 2011 the way "
            "IT/ES did; the jump is 2012."
        ),
    }


def boi_tcce0200() -> pd.DataFrame:
    """BoI public-debt holders by creditor (TCCE0200) — monthly, face €mn. Matches IMF GG totals."""
    p = CACHE / "boi_TCCE0200.csv"
    if not p.exists():
        return pd.DataFrame()
    raw = pd.read_csv(p, sep=";")
    raw["date"] = pd.to_datetime(raw["DATA_OSS"])
    raw = raw.set_index("date").sort_index()
    cols = {
        "boi_bn": "FPI_FP.M.IT.S13.MGD.S121.101.112.FAV.EUR.EDP",
        "mfi_bn": "FPI_FP.M.IT.S13.MGD.S12BI1.101.112.FAV.EUR.EDP",
        "ofi_bn": "FPI_FP.M.IT.S13.MGD.S12BI2.101.112.FAV.EUR.EDP",
        "oth_bn": "FPI_FP.M.IT.S13.MGD.SBI1.101.112.FAV.EUR.EDP",
        "nonres_bn": "FPI_FP.M.IT.S13.MGD.S2.101.112.FAV.EUR.EDP",
        "nonres_sec_bn": "FPI_FP.M.IT.S13.F3.S2.101.112.FAV.EUR.EDP",
        "mfi_sec_bn": "FPI_FP.M.IT.S13.F3.S12BI1.101.112.FAV.EUR.EDP",
        "total_bn": "FPI_FP.M.IT.S13.MGD.SBI3.101.112.FAV.EUR.EDP",
        "nonres_pct": "FPI_FP.M.IT.S13.MGD.S2.101.112.FAV.PT.EDP",
        "mfi_pct": "FPI_FP.M.IT.S13.MGD.S12BI1.101.112.FAV.PT.EDP",
        "boi_pct": "FPI_FP.M.IT.S13.MGD.S121.101.112.FAV.PT.EDP",
    }
    h = raw[[c for c in cols.values() if c in raw.columns]].apply(pd.to_numeric, errors="coerce")
    h.columns = [k for k, c in cols.items() if c in raw.columns]
    for c in ("boi_bn", "mfi_bn", "ofi_bn", "oth_bn", "nonres_bn", "nonres_sec_bn", "mfi_sec_bn", "total_bn"):
        if c in h.columns:
            h[c] = h[c] / 1000.0  # €mn → €bn
    if "nonres_pct" not in h.columns and "nonres_bn" in h.columns:
        h["nonres_pct"] = 100.0 * h["nonres_bn"] / h["total_bn"]
    if "mfi_pct" not in h.columns and "mfi_bn" in h.columns:
        h["mfi_pct"] = 100.0 * h["mfi_bn"] / h["total_bn"]
    out_p = CACHE / "boi_tcce0200_holders.parquet"
    h.to_parquet(out_p)
    return h


def bde_be2244() -> pd.DataFrame:
    """BdE BE.22.44 LET+BYO holdings by sector — monthly face €, from Jan-2016 only."""
    p = CACHE / "be2244.csv"
    if not p.exists():
        return pd.DataFrame()
    text = p.read_text(encoding="latin-1")
    rows = []
    for ln in text.splitlines():
        m = re.match(r'^"?([A-Z]{3}) (\d{4})"?,(.*)$', ln)
        if not m or m.group(1) not in _ES_MONTHS:
            continue
        vals = [pd.to_numeric(x.replace('"', "").replace(",", ""), errors="coerce")
                for x in m.group(3).split(",")]
        rows.append((pd.Timestamp(year=int(m.group(2)), month=_ES_MONTHS[m.group(1)], day=1), vals))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    # cols: 0 total, 1 deposit+CB, 2 CB, 3 deposit ex-CB, 4 other fin, … 12 RoW
    names = {
        0: "total_bn", 1: "deposit_incl_cb_bn", 2: "cb_bn", 3: "banks_ex_cb_bn",
        4: "other_fin_bn", 5: "funds_bn", 6: "insurance_bn", 7: "pf_bn",
        8: "other_ofi_bn", 9: "nfc_bn", 10: "hh_bn", 11: "gg_bn", 12: "nonres_bn",
    }
    keep = {i: n for i, n in names.items() if i < df.shape[1]}
    out = df[list(keep)].copy()
    out.columns = list(keep.values())
    out = out / 1e9  # face euros → €bn
    out["nonres_pct"] = 100.0 * out["nonres_bn"] / out["total_bn"]
    out["banks_ex_cb_pct"] = 100.0 * out["banks_ex_cb_bn"] / out["total_bn"]
    out["cb_pct"] = 100.0 * out["cb_bn"] / out["total_bn"]
    out.to_parquet(CACHE / "bde_be2244_holders.parquet")
    return out


def _holder_row(df: pd.DataFrame, date: str, fields: list[str]) -> dict:
    if df.empty or date not in df.index:
        # nearest prior month-end / period
        prior = df.loc[:date] if len(df) else df
        if prior.empty:
            return {}
        r = prior.iloc[-1]
        used = str(prior.index[-1].date())
    else:
        r = df.loc[date]
        used = date
    out = {"asof": used}
    for f in fields:
        if f in df.columns and pd.notna(r[f]):
            out[f] = round(float(r[f]), 1)
    return out


# ---------------------------------------------------------------- build
def _at(sp: pd.Series, date: str) -> float | None:
    s = sp.dropna()
    if s.empty:
        return None
    ts = pd.Timestamp(date)
    if ts in s.index:
        return float(s.loc[ts])
    # nearest prior
    prior = s.loc[:ts]
    return float(prior.iloc[-1]) if len(prior) else None


def build() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    sp = spreads_daily()
    imf_w = imf_foreign_private()
    qsa = qsa_nonres()
    bsi = bsi_banks()
    ilm = ilm_weekly()
    banks = banks_etf()
    smp_stock = smp_end2012_by_issuer()

    def imf_row(cc: str, q: str) -> dict:
        s = imf_w[(imf_w["country"] == cc) & (imf_w["quarter_end"] == pd.Timestamp(q))]
        if s.empty:
            return {}
        r = s.iloc[0]
        return {
            "d4_fpriv_gdp": round(float(r["d4_fpriv_gdp"]), 2) if pd.notna(r["d4_fpriv_gdp"]) else None,
            "fpriv_share": round(float(r["fpriv_share"]), 1),
            "db_share": round(float(r["db_share"]), 1),
            "dcb_share": round(float(r["dcb_share"]), 1),
        }

    def qsa_row(cc: str, q: str) -> float | None:
        s = qsa[(qsa["country"] == cc) & (qsa["quarter_end"] == pd.Timestamp(q))]
        if s.empty or pd.isna(s.iloc[0]["nonres_share"]):
            return None
        return round(float(s.iloc[0]["nonres_share"]), 1)

    def bsi_window(cc: str, a: str, b: str) -> dict:
        s = bsi[bsi["country"] == cc].set_index("month").sort_index()
        w = s.loc[a:b]
        return {
            "flow_sum_bn": round(float(w["flow_eur_bn"].sum()), 1) if len(w) else None,
            "stock_start_bn": round(float(w["value_eur_bn"].iloc[0]), 0) if len(w) else None,
            "stock_end_bn": round(float(w["value_eur_bn"].iloc[-1]), 0) if len(w) else None,
        }

    # peaks
    peaks = {
        "FR_2011": {"bp": round(float(sp["FR"].loc["2011"].max()), 1),
                    "date": str(sp["FR"].loc["2011"].idxmax().date())},
        "IT_2011": {"bp": round(float(sp["IT"].loc["2011"].max()), 1),
                    "date": str(sp["IT"].loc["2011"].idxmax().date())},
        "ES_2011": {"bp": round(float(sp["ES"].loc["2011"].max()), 1),
                    "date": str(sp["ES"].loc["2011"].idxmax().date())},
        "FR_2026_06_30": round(_at(sp["FR"], "2026-06-30") or 66.0, 1),
        "FR_2026_09_25": round(_at(sp["FR"], "2026-09-25") or 100.0, 1),
        "FR_live_tape_note_bp": 175,
    }

    # ILM H2-11
    h2 = ilm[(ilm["date"] >= "2011-08-01") & (ilm["date"] <= "2011-12-31")] if len(ilm) else ilm
    ilm_stats = {
        "stock_end_2010_bn": round(float(ilm[ilm["date"] <= "2010-12-31"].iloc[-1]["stock_mn"] / 1000), 1) if len(ilm) else None,
        "stock_end_2011_bn": round(float(ilm[ilm["date"] <= "2011-12-31"].iloc[-1]["stock_mn"] / 1000), 1) if len(ilm) else None,
        "stock_end_2012_bn": round(float(ilm[ilm["date"] <= "2012-12-31"].iloc[-1]["stock_mn"] / 1000), 1) if len(ilm) else None,
        "h2_2011_sum_delta_bn": round(float(h2["d_bn"].sum()), 1) if len(h2) else None,
        "h2_2011_max_weekly_bn": round(float(h2["d_bn"].max()), 1) if len(h2) else None,
        "note": "ILM A070100 = all monetary-policy securities (SMP + CBPP…). Country SMP stock only from end-2012 transparency; FR was never an SMP issuer.",
    }

    bank_dd = {}
    if len(banks):
        j11 = banks.loc["2011-05":"2011-07"].max()
        n11 = banks.loc["2011-11":"2011-12"].min()
        bank_dd["2011"] = {"may_jul_max": round(float(j11), 2), "nov_dec_min": round(float(n11), 2),
                           "drawdown_pct": round(100 * (n11 / j11 - 1), 1)}
        if len(banks.loc["2026-06":"2026-09"]):
            j26 = banks.loc["2026-06":"2026-06"].max()
            n26 = banks.loc["2026-06":"2026-09"].iloc[-1]
            bank_dd["2026"] = {"jun_max": round(float(j26), 2), "sep_last": round(float(n26), 2),
                               "move_pct": round(100 * (n26 / j26 - 1), 1)}

    it_h = boi_tcce0200()
    es_h = bde_be2244()
    it_fields = ["nonres_pct", "mfi_pct", "boi_pct", "nonres_bn", "mfi_bn", "total_bn"]
    es_fields = ["nonres_pct", "banks_ex_cb_pct", "cb_pct", "nonres_bn", "banks_ex_cb_bn", "total_bn"]
    it_snap = {
        d: _holder_row(it_h, d, it_fields)
        for d in ("2010-12-31", "2011-06-30", "2011-09-30", "2011-12-31", "2012-03-31",
                  "2012-12-31", "2025-12-31", "2026-06-30")
    }
    es_snap = {
        d: _holder_row(es_h, d, es_fields)
        for d in ("2016-01-01", "2020-01-01", "2025-12-01", "2026-08-01")
    }
    # ES 2011 crisis: BdE sector tape starts 2016 — use IMF (matches BoI TCCE for IT 1:1)
    es_imf_2011 = {
        q: imf_row("ES", q) for q in ("2010-12-31", "2011-06-30", "2011-09-30", "2011-12-31", "2012-03-31")
    }

    bdf = bdf_detention_snapshot(
        ["2010Q1", "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1", "2025Q4", "2026Q1", "2026Q2"]
    )
    official = oat_bund_official_block()
    eba = eba_domestic_block()
    if official.get("2026_10_23_bp") is not None:
        peaks["FR_official_2026_10_23"] = official["2026_10_23_bp"]
    if official.get("2026_10_22_bp") is not None:
        peaks["FR_official_2026_10_22"] = official["2026_10_22_bp"]

    payload = {
        "asof": "2026-10-24",
        "verdict": (
            "FR 2026 = IT/ES-2011 buyer-stack geometry at FR-2011 spread levels. "
            f"Official OAT–Bund {official.get('2026_10_23_bp', 166)}bp (23-Oct, BdF TEC−BBK) sits in the "
            f"FR-2011 peak zone ({official.get('2011_peak_bp', 208)} on {official.get('2011_peak_date', '2011-11-17')}), "
            "not IT-2011 (~550). BdF DET: FR foreign OAT share only drifted 62→55 in 2011 (mild). "
            "BoI TCCE0200: IT nonres 38.3%→34.8%→31.5% (Jun-11→Dec-11→Mar-12) while MFI 25.4%→29.0%. "
            "IMF ES foreign-private 33%→17% (Jun-11→Mar-12). EBA: FR banks absorbed in 2012, not 2011. "
            "SMP never bought FR (€0 end-2012). Live risk = IT/ES-2011 foreign blink + bank clearing "
            "that France largely skipped in 2011."
        ),
        "sources": {
            "spreads": "issuance/cache/egb_daily.parquet",
            "imf": "issuance.imf_investor_base.wide",
            "qsa_nonres": "ECB QSA S13 F3 non-resident liabilities",
            "bsi_banks": "ECB BSI MFIs ex NCB domestic GG debt (monthly stocks + transactions)",
            "cover_fr_it": "issuance/cache/ops_{fr,it}.parquet",
            "cover_es": "data/raw/es/tesoro/historico_subastas.xlsx (sheet year overrides Excel FECHA year)",
            "smp_weekly_proxy": "ECB ILM.W.U2.C.A070100.U2.EUR",
            "smp_country_stock": "ECB SMP_breakdown_history.csv (end-year from 2012)",
            "banks_equity": "Yahoo EXV1.DE (iShares EURO STOXX Banks)",
            "bdf_detention": "Banque de France DET via comptespublics.fr (+ Webstat DET2 OAT)",
            "boi_holders": "Banca d'Italia TCCE0200 (Debito AP: settori detentori) via a2a Infostat",
            "bde_holders": "Banco de España BE.22.44 LET+BYO tenencias (from Jan-2016); 2011 via IMF",
            "oat_bund_official": "BdF TEC10 − Bundesbank BBSIS 10y federal zero",
            "eba_domestic": "EBA 2013 Transparency Summary Report Table 2",
        },
        "spreads_bp": peaks,
        "imf_foreign_private": {
            "IT": {q: imf_row("IT", q) for q in ("2011-06-30", "2011-09-30", "2011-12-31", "2012-03-31", "2025-12-31")},
            "ES": {q: imf_row("ES", q) for q in ("2011-06-30", "2011-09-30", "2011-12-31", "2012-03-31", "2025-12-31")},
            "FR": {q: imf_row("FR", q) for q in ("2011-06-30", "2011-09-30", "2011-12-31", "2012-03-31", "2025-12-31")},
        },
        "qsa_nonres_share_pct": {
            "IT": {q: qsa_row("IT", q) for q in ("2012-03-31", "2012-06-30", "2025-12-31", "2026-03-31")},
            "ES": {q: qsa_row("ES", q) for q in ("2010-12-31", "2011-06-30", "2011-12-31", "2012-03-31", "2025-12-31", "2026-03-31")},
            "FR": {q: qsa_row("FR", q) for q in ("2010-12-31", "2011-06-30", "2011-12-31", "2012-03-31", "2025-12-31", "2026-03-31")},
        },
        "bsi_domestic_bank_absorption": {
            "IT_H2_2011": bsi_window("IT", "2011-07", "2011-12"),
            "ES_H2_2011": bsi_window("ES", "2011-07", "2011-12"),
            "FR_H2_2011": bsi_window("FR", "2011-07", "2011-12"),
            "FR_2025H2": bsi_window("FR", "2025-07", "2025-12"),
            "FR_2026_YTD": bsi_window("FR", "2026-01", "2026-08"),
            "DE_H2_2011": bsi_window("DE", "2011-07", "2011-12"),
        },
        "auction_cover": {
            "FR_H2_2011": fr_it_cover("fr", "2011-07-01", "2011-12-31"),
            "FR_Aug_Oct_2026": fr_it_cover("fr", "2026-08-01", "2026-10-31"),
            "IT_H2_2011": fr_it_cover("it", "2011-07-01", "2011-12-31"),
            "ES_H2_2011_long": es_cover_stats("2011-07-01", "2011-12-31"),
            "ES_2011_long": es_cover_stats("2011-01-01", "2011-12-31"),
        },
        "smp": {
            "weekly_ilm": ilm_stats,
            "end_2012_nominal_bn_by_issuer": smp_stock,
            "france_purchased": False,
        },
        "banks_equity_exv1": bank_dd,
        "events_2011": EVENTS_2011,
        "events_2026": EVENTS_2026,
        "gaps_remaining": [
            "No public daily/weekly foreign OAT flow (BdF DET quarterly; BoI TCCE monthly; BdE be2244 monthly from 2016).",
            "BdE BE.22.44 holder tape starts Jan-2016 — ES 2011 crisis uses IMF (bit-identical to BoI TCCE totals for IT).",
            "No free CDS panel (Markit/WRDS/Bloomberg only) — EXV1 equity ETF remains the stress proxy.",
            "Weekly SMP by country during 2011 is confidential ECB research data; public = ILM aggregate + end-2012 country stock.",
            "EBA sovereign template Dec-2011 is annual snapshots, not a monthly path.",
            "VALM0100 is securities *issued* by residents, not holders — do not use it as a detention tape.",
        ],
        "bdf_detention": bdf,
        "oat_bund_official": official,
        "eba_domestic_sovereign": eba,
        "boi_tcce0200": {
            "source": "Banca d'Italia TCCE0200 (Debito delle Amministrazioni pubbliche: settori detentori)",
            "unit": "€bn face / % of GG debt",
            "note": "Totals match IMF Arslanalp–Tsuda GG stock (€1,921bn Dec-2010). Nonres = S2; MFI = other resident MFIs ex BoI.",
            "snapshots": it_snap,
            "read": (
                "IT nonres share 38.3% (Jun-11) → 34.8% (Dec-11) → 31.5% (Mar-12); "
                "MFI 25.4% → 26.0% → 29.0%. Securities nonres share 46.6% → 42.1% → 37.7%. "
                "That is the foreign blink + bank clearing FR did not run in 2011."
            ),
        },
        "bde_be2244": {
            "source": "Banco de España BE.22.44 (LET+BYO tenencias a vencimiento)",
            "unit": "€bn face / % of LET+BYO",
            "coverage": "2016-01 → present (methodology break; no pre-2016 sector tape on WebBE)",
            "snapshots": es_snap,
            "crisis_2011_via_imf": es_imf_2011,
            "read": (
                "Live ES nonres share of LET+BYO ~46% (2016) → 51% (Aug-26); CB 9% → 20%. "
                "For 2011 use IMF: foreign-private share 33% (Jun-11) → 26% (Dec-11) → 17% (Mar-12)."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    _charts(sp, ilm, banks, bsi, payload, it_h=it_h, es_h=es_h)
    return payload


def _charts(
    sp: pd.DataFrame,
    ilm: pd.DataFrame,
    banks: pd.Series,
    bsi: pd.DataFrame,
    payload: dict,
    it_h: pd.DataFrame | None = None,
    es_h: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=False,
                            gridspec_kw={"height_ratios": [1.35, 1.0, 1.0]})
    ax = axes[0]
    w = sp.loc["2010-01":"2012-12"]
    for c, col in (("IT", "#b33"), ("ES", "#c80"), ("FR", "#06c")):
        ax.plot(w.index, w[c], label=c, lw=1.1 if c != "FR" else 1.6, alpha=0.85 if c != "FR" else 1)
    ax.axhline(175, color="#06c", ls=":", lw=0.9, label="FR 2026 tape ~175")
    ax.axhline(208, color="#06c", ls="--", lw=0.7, alpha=0.7, label="FR peak 17-Nov-11 (208)")
    for d, lab in EVENTS_2011:
        ts = pd.Timestamp(d)
        if w.index.min() <= ts <= w.index.max():
            ax.axvline(ts, color="#999", lw=0.5, alpha=0.5)
    ax.set_ylabel("10y vs Bund (bp)")
    ax.set_title("2011 map — spreads (IT/ES clearing vs FR sympathy)")
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    ax.set_ylim(0, 600)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    if len(ilm):
        il = ilm[(ilm["date"] >= "2010-01-01") & (ilm["date"] <= "2012-12-31")]
        ax.fill_between(il["date"], 0, il["d_bn"].clip(lower=0), color="#444", alpha=0.35, label="ILM Δ (SMP-era proxy, €bn/wk)")
        ax.plot(il["date"], il["stock_mn"] / 1000, color="#000", lw=1.0, label="ILM stock €bn (right scale transformed)")
        ax.set_ylabel("Weekly Δ €bn")
        ax.set_title("Official bid — ILM monetary-policy securities (SMP reactivation Aug-11)")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(il["date"], il["stock_mn"] / 1000, color="#000", lw=1.0)
        ax2.set_ylabel("Stock €bn")

    ax = axes[2]
    for cc, color in (("IT", "#b33"), ("ES", "#c80"), ("FR", "#06c")):
        s = bsi[bsi["country"] == cc].set_index("month").sort_index()
        f = s.loc["2010-01":"2012-12", "flow_eur_bn"].rolling(3, min_periods=1).sum()
        ax.plot(f.index, f.values, color=color, lw=1.2, label=f"{cc} BSI bank flow 3m sum")
    ax.axhline(0, color="#000", lw=0.5)
    ax.set_ylabel("€bn (3m sum)")
    ax.set_title("Domestic bank absorption — ECB BSI (ex NCB) of domestic GG debt")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # absorption comparison panel
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rows = []
    for cc in ("IT", "ES", "FR"):
        for q, lab in (("2011-09-30", "2011Q3"), ("2011-12-31", "2011Q4"), ("2012-03-31", "2012Q1")):
            r = payload["imf_foreign_private"][cc].get(q, {})
            if r.get("d4_fpriv_gdp") is not None:
                rows.append({"cc": cc, "lab": lab, "v": r["d4_fpriv_gdp"]})
    d = pd.DataFrame(rows)
    if len(d):
        x = np.arange(len(d["lab"].unique()))
        width = 0.25
        for i, cc in enumerate(("IT", "ES", "FR")):
            y = [d[(d.cc == cc) & (d.lab == lab)]["v"].values[0]
                 if len(d[(d.cc == cc) & (d.lab == lab)]) else 0 for lab in d["lab"].unique()]
            ax.bar(x + (i - 1) * width, y, width, label=cc, color={"IT": "#b33", "ES": "#c80", "FR": "#06c"}[cc])
        ax.set_xticks(x)
        ax.set_xticklabels(list(d["lab"].unique()))
    ax.axhline(0, color="#000", lw=0.6)
    ax.set_ylabel("IMF Δ4q foreign-private holdings / GDP (pp)")
    ax.set_title("Foreign sleeve blink — IT/ES 2011 vs FR flat")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART_ABSORB, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # BdF DET (rebuild from cache)
    bdf = bdf_detention_panel()
    if len(bdf):
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        sub = bdf.loc["2009":"2013"]
        ax.plot(sub.index, sub["nonres_oat_pct"], color="#06c", lw=1.6, label="FR nonres OAT %")
        ax.plot(sub.index, sub["banques_oat_pct"], color="#444", lw=1.1, label="FR banks OAT %")
        ax.plot(sub.index, sub["banque_centrale_oat_pct"], color="#080", lw=1.1, label="BdF OAT %")
        ax.set_ylabel("% of OAT stock")
        ax.set_title("BdF DET — France OAT holders through 2011 (mild foreign drift)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(CHART_BDF, dpi=140, bbox_inches="tight")
        plt.close(fig)

    # IT BoI + ES BdE/IMF national holder tapes
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=False)
    ax = axes[0]
    if it_h is not None and len(it_h):
        w = it_h.loc["2010-01":"2013-06"].dropna(subset=["nonres_pct"], how="all")
        ax.plot(w.index, w["nonres_pct"], color="#b33", lw=1.5, label="IT nonres % (BoI TCCE0200)")
        if "mfi_pct" in w.columns:
            ax.plot(w.index, w["mfi_pct"], color="#b33", lw=1.0, ls="--", label="IT MFI %")
    # FR BdF overlay on same axes (quarterly → resample plot)
    if len(bdf):
        fr = bdf.loc["2010":"2013", "nonres_oat_pct"].dropna()
        ax.plot(fr.index, fr.values, color="#06c", lw=1.4, label="FR nonres OAT % (BdF DET)")
    # ES IMF foreign-private share quarterly markers
    imf_w = imf_foreign_private()
    es = imf_w[(imf_w["country"] == "ES") & (imf_w["quarter_end"] >= "2010-01-01")
               & (imf_w["quarter_end"] <= "2013-06-30")]
    if len(es):
        ax.plot(es["quarter_end"], es["fpriv_share"], color="#c80", lw=1.4, marker="o",
                ms=3.5, label="ES foreign-private % (IMF; BdE sector tape from 2016)")
    ax.set_ylabel("% of GG / OAT stock")
    ax.set_title("National holder tapes — IT (BoI) / ES (IMF) / FR (BdF) through the 2011 blink")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(10, 70)

    ax = axes[1]
    if es_h is not None and len(es_h):
        w = es_h.loc["2016-01":]
        ax.plot(w.index, w["nonres_pct"], color="#c80", lw=1.4, label="ES nonres % LET+BYO (BdE 22.44)")
        ax.plot(w.index, w["cb_pct"], color="#080", lw=1.0, label="ES CB %")
        ax.plot(w.index, w["banks_ex_cb_pct"], color="#c80", lw=1.0, ls="--", label="ES banks ex-CB %")
    if it_h is not None and len(it_h):
        w = it_h.loc["2016-01":"2026-08"].dropna(subset=["nonres_pct"])
        ax.plot(w.index, w["nonres_pct"], color="#b33", lw=1.2, label="IT nonres % (BoI)")
        if "boi_pct" in w.columns:
            ax.plot(w.index, w["boi_pct"], color="#080", lw=1.0, ls=":", label="BoI %")
    if len(bdf):
        fr = bdf.loc["2016":, "nonres_oat_pct"].dropna()
        ax.plot(fr.index, fr.values, color="#06c", lw=1.3, label="FR nonres OAT %")
    ax.set_ylabel("% of stock")
    ax.set_title("Live tape — ES BdE sector holdings (2016+) vs IT BoI / FR BdF")
    ax.legend(frameon=False, fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART_HOLDERS, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = build()
    print(json.dumps({
        "verdict": p["verdict"],
        "boi_it": p.get("boi_tcce0200", {}).get("snapshots", {}).get("2011-12-31"),
        "bde_es_live": p.get("bde_be2244", {}).get("snapshots", {}).get("2026-08-01"),
        "es_imf_2011q4": p.get("bde_be2244", {}).get("crisis_2011_via_imf", {}).get("2011-12-31"),
        "spreads": p["spreads_bp"],
        "wrote": str(OUT_JSON),
        "charts": [str(CHART_PATH), str(CHART_ABSORB), str(CHART_BDF), str(CHART_HOLDERS)],
    }, indent=2))


if __name__ == "__main__":
    main()
