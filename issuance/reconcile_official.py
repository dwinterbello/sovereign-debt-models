"""Stack gross issuance vs an independent official total, by country, period and type.

Each check compares what the issuance stack carries (``cache/ops_<cc>.parquet``, plus
``forecast_priced.parquet`` for a plan year that is not over) to a publication that the
ops fetcher does *not* read:

  US  DTS Table IIIA marketable issues (us/fiscal cache)       vs ops by settlement year
  UK  DMO D2.1A auction results + DMO syndication list         vs ops auctions + syndications, FY
  DE  BMF Schuldenbericht "Umlaufvolumen" (cumulative to Dec)  vs ops (DFA Emissionshistorie)
  FR  AFT besoins/ressources: net MLT issuance done to date    vs ops MLT net of buybacks, YTD
  IT  Banca d'Italia "Aste dal 2002" auction archive           vs ops auctions (MEF), ex syndications
  ES  Tesoro sheet 13 effective issuance + press gross totals  vs ops at effective cash (capped at par)
  JP  MoF FY plan, periodic auctions ex TBs and FY adjustment  vs ops (+ forecast) by FY
  SE  Riksgälden CH926 gross allocations by type               vs ops Auction + Syndication (same tape:
      checks the ISIN join keeps matured series, not an independent total)

A missing official file yields ``status = "missing"`` (tests skip); any row past its
country tolerance is ``FAIL``. ``python -m issuance.reconcile_official`` exits 1 on a FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE, DERIVED, RAW, REPO

# Target tolerance as a share of the official figure. Every country should pass these.
TOL_PCT = {
    "US": 0.02,  # DTS is cash at settlement: bills at discount (~1%)
    "UK": 0.005,  # same DMO nominal, different report
    "DE": 0.03,  # Umlauf flow is face to 2024, issue-price value from 2025; both incl. retention
    "FR": 0.03,
    "IT": 0.03,
    "ES": 0.03,
    "JP": 0.02,
    "SE": 0.005,  # gross allocations, same CH926 tape
}
# Temporary gates for countries whose fetchers are being fixed. Remove an entry once the
# country passes TOL_PCT; the report always shows the target verdict as well.
TOL_PCT_TEMP = {
    "UK": 0.05,
    # ops drop a second same-type line on one auction day (2015 BTP€i, 2020/2024 BOT, May-25 BTP€i)
    "IT": 0.25,
}
# Absolute floor on the band. Keep it under one auction of the smallest gated line: ES
# linker auctions are €0.5–0.9bn, and a €500mn floor passed a double-counted one (+6.7%).
ABS_FLOOR_MN = {"US": 5_000.0, "UK": 50.0, "DE": 500.0, "FR": 500.0, "IT": 500.0,
                "ES": 100.0, "JP": 50_000.0, "SE": 1_000.0}
ABS_FLOOR_TEMP_MN: dict[str, float] = {}
# Before FY2008-09 D2.1E books non-competitive allotments (+£20–50mn per auction) and DMO
# taps ("Outright", e.g. 6-Aug-1999 £400mn 6% 2028) that D2.1A omits; D2.1E ties to D1A stock,
# so those FYs (FY1999-00 +£460mn, FY2001-02 +£75mn) are reported but not target-gated.
UK_TARGET_FROM_FY = 2008

US_DTS = REPO / "us" / "fiscal" / "data" / "cache" / "dts_table3a.parquet"
UK_D21A = RAW / "uk" / "D2_1A.pdf"
UK_SYND = REPO / "uk_flows" / "cache" / "dmo" / "syndications.csv"
UK_TENDERS = REPO / "uk_flows" / "cache" / "dmo" / "tenders.xls"
DE_SB = REPO / "sovereign_cash" / "data" / "de" / "schuldenbericht_dt.xlsx"
FR_PLAN = REPO / "sovereign_cash" / "data" / "fr" / "plan.json"
IT_BDI = RAW / "mef" / "bdi" / "storico_aste" / "Aste dal 2002 al 2025.xlsx"
ES_T13 = RAW / "es" / "tesoro" / "13_net_financing.xlsx"
JP_PLAN = RAW / "jp" / "highlight251226.pdf"
SE_AUCTIONS = REPO / "sovereign_cash" / "data" / "se" / "auctions.csv"

OFFICIAL = {"US": US_DTS, "UK": UK_D21A, "DE": DE_SB, "FR": FR_PLAN, "IT": IT_BDI,
            "ES": ES_T13, "JP": JP_PLAN, "SE": SE_AUCTIONS}
COUNTRIES = tuple(OFFICIAL)


def _ops(cc: str) -> pd.DataFrame | None:
    p = CACHE / f"ops_{cc.lower()}.parquet"
    if not p.exists():
        return None
    o = pd.read_parquet(p)
    if "is_forecast" in o.columns:
        o = o[~o["is_forecast"].fillna(False).astype(bool)]
    o = o.copy()
    o["op_dt"] = pd.to_datetime(o["op_dt"])
    if "issue_dt" in o.columns:
        o["issue_dt"] = pd.to_datetime(o["issue_dt"])
    o["amount_issued"] = pd.to_numeric(o["amount_issued"], errors="coerce").fillna(0.0)
    o["sec_type"] = o["sec_type"].astype(str)
    o["op_type"] = o["op_type"].astype(str)
    return o


def _fy(dt: pd.Series) -> pd.Series:
    return dt.dt.year.where(dt.dt.month >= 4, dt.dt.year - 1)


def _row(cc, check, period, group, stack, official, note=""):
    return {"country": cc, "check": check, "period": str(period), "group": group,
            "stack_mn": float(stack), "official_mn": float(official), "note": note}


# --------------------------------------------------------------------------- US
def check_us(ops: pd.DataFrame) -> list[dict]:
    d = pd.read_parquet(US_DTS)
    d = d[(d["transaction_type"] == "Issues") & (d["security_market"] == "Marketable")].copy()
    d["v"] = pd.to_numeric(d["transaction_today_amt"], errors="coerce").fillna(0.0)
    d["dt"] = pd.to_datetime(d["record_date"])
    last = d["dt"].max()
    grp = {"Bills": "Bill", "Notes": "Coupon", "Bonds": "Coupon"}  # TIPS/FRN sit inside Notes
    d["group"] = d["security_type"].map(grp)
    d = d[d["group"].notna()]
    d["year"] = d["dt"].dt.year
    off = d.groupby(["year", "group"])["v"].sum()

    g = ops[ops["amount_issued"] > 0].copy()
    g["dt"] = g["issue_dt"].fillna(g["op_dt"])
    g = g[(g["dt"] >= d["dt"].min()) & (g["dt"] <= last)]
    g["group"] = g["sec_type"].map({"Bill": "Bill"}).fillna("Coupon")
    g["year"] = g["dt"].dt.year
    stk = g.groupby(["year", "group"])["amount_issued"].sum()
    out = []
    for (y, k), v in off.items():
        period = y if y < last.year else f"{y}-YTD {last.date()}"
        out.append(_row("US", "dts_iiia", period, k, stk.get((y, k), 0.0), v,
                        "settlement year; DTS cash"))
    return out


# --------------------------------------------------------------------------- UK
def _uk_d21a() -> pd.DataFrame:
    import pypdf

    pat = re.compile(r"^(\d{2}-[A-Z][a-z]{2}-\d{4}) (.+?\d{4})\s+([\d,]+\.\d{2})")
    rows = []
    for page in pypdf.PdfReader(str(UK_D21A)).pages:
        for line in (page.extract_text() or "").splitlines():
            m = pat.match(line)
            if m:
                rows.append({"dt": pd.Timestamp(m.group(1)), "name": m.group(2),
                             "mn": float(m.group(3).replace(",", ""))})
    return pd.DataFrame(rows)


def check_uk(ops: pd.DataFrame) -> list[dict]:
    a = _uk_d21a()
    a["group"] = a["name"].str.contains("Index-linked", case=False).map({True: "Linker", False: "Nominal"})
    a["kind"] = "auction"
    a["dt"] = a["dt"] + pd.offsets.BDay(1)  # ops carry the settlement date (T+1)
    parts = [a[["dt", "group", "mn", "kind"]]]
    last = a["dt"].max()
    if UK_TENDERS.exists():
        t = pd.read_excel(UK_TENDERS, header=None, skiprows=5)
        t = pd.DataFrame({"dt": pd.to_datetime(t[8], errors="coerce"), "name": t[1].astype(str),
                          "mn": pd.to_numeric(t[2], errors="coerce")}).dropna()
        t["group"] = t["name"].str.contains("Index-linked", case=False).map({True: "Linker", False: "Nominal"})
        t["kind"] = "auction"
        parts.append(t[["dt", "group", "mn", "kind"]])
    if UK_SYND.exists():
        s = pd.read_csv(UK_SYND, encoding="latin-1")
        s.columns = ["dt", "name", "mn", "price", "yield"][: len(s.columns)]
        s["dt"] = pd.to_datetime(s["dt"], format="%d %b %Y", errors="coerce")
        s["mn"] = pd.to_numeric(s["mn"].astype(str).str.replace(",", ""), errors="coerce")
        s["group"] = s["name"].str.contains("Index-linked", case=False).map({True: "Linker", False: "Nominal"})
        s["kind"] = "syndication"
        parts.append(s[["dt", "group", "mn", "kind"]].dropna())
    off = pd.concat(parts, ignore_index=True)
    first_fy = int(_fy(pd.Series([a["dt"].min()])).iloc[0]) + 1  # first FY fully covered
    off = off[off["dt"] <= last]
    off["fy"] = _fy(off["dt"])
    want = {"Auction": "auction", "Tender": "auction", "Syndication": "syndication"}
    g = ops[ops["op_type"].isin(want) & ops["sec_type"].isin(["Nominal", "Linker"])
            & (ops["amount_issued"] > 0) & (ops["op_dt"] <= last)].copy()
    g["kind"] = g["op_type"].map(want)
    g["fy"] = _fy(g["op_dt"])
    stk = g.groupby(["fy", "sec_type", "kind"])["amount_issued"].sum()
    offg = off.groupby(["fy", "group", "kind"])["mn"].sum()
    last_fy = int(_fy(pd.Series([last])).iloc[0])
    out = []
    for (fy, k, kind), v in offg.items():
        if fy < first_fy:
            continue
        period = f"FY{fy}-{str(fy + 1)[2:]}" + (f" to {last.date()}" if fy == last_fy else "")
        r = _row("UK", f"dmo_{kind}", period, k, stk.get((fy, k, kind), 0.0), v,
                 "nominal; ops Auction+Tender vs D2.1A + gilt tenders" if kind == "auction"
                 else "nominal")
        if fy < UK_TARGET_FROM_FY:
            r["in_target"] = False
            r["note"] += "; pre-FY2008 D2.1E incl. non-competitive allotments and taps, not target-gated"
        out.append(r)
    return out


# --------------------------------------------------------------------------- DE
def check_de(ops: pd.DataFrame) -> list[dict]:
    raw = pd.read_excel(DE_SB, sheet_name="rpgBruttokreditaufnahme Gesamt", header=None)
    hdr = raw.iloc[13]
    lab = raw[0].astype(str).str.strip()
    start = int(lab[lab == "Umlaufvolumen"].index[0])
    block = raw.iloc[start:start + 25]
    blab = block[0].astype(str).str.strip()

    def line(name: str) -> pd.Series:
        r = block[blab == name]
        return pd.to_numeric(r.iloc[0, 1:], errors="coerce") / 1e6 if len(r) else pd.Series(dtype=float)

    conv, bubill = line("Konventionelle Bundeswertpapiere"), line("Unverzinsliche Schatzanweisungen des Bundes")
    linker, green = line("Inflationsindexierte Bundeswertpapiere"), line("Grüne Bundeswertpapiere")
    dates = pd.to_datetime(hdr.iloc[1:], format="%d.%m.%Y", errors="coerce")
    series = {"Bill": bubill, "Nominal": conv - bubill + green.fillna(0.0), "Linker": linker}
    last = dates.max()
    pick = [i for i, dt in dates.items() if pd.notna(dt) and ((dt.month == 12 and dt.year >= 2015) or dt == last)]

    g = ops[ops["amount_issued"] > 0].copy()
    g = g[g["sec_type"].isin(["Bill", "Nominal", "Linker"])].sort_values("op_dt")
    g["settle"] = g["op_dt"] + pd.offsets.BDay(2)  # Umlauf counts at value date (T+2)
    g["year"] = g["settle"].dt.year
    # OH taps (into own holdings) carry no price: value them at the line's nearest auction price.
    px = pd.to_numeric(g["price"], errors="coerce")
    px = px.groupby(g["isin"]).transform(lambda s: s.ffill().bfill()).fillna(100.0)
    g["cash"] = g["amount_issued"] * px / 100.0
    # Umlaufvolumen flows are face (round €mn) through 2024 and issue-price value from 2025;
    # both include retention (Eigenbestand is netted on its own line). Detect per column.
    face_lines = [conv, bubill, linker, green]
    out = []
    for i in pick:
        dt = dates[i]
        vals = pd.Series([s.get(i) for s in face_lines], dtype=float).dropna()
        face = bool(((vals - vals.round(-1)).abs() < 0.5).all())
        col = "amount_issued" if face else "cash"
        stk = g[(g["year"] == dt.year) & (g["settle"] <= dt)].groupby("sec_type")[col].sum()
        period = dt.year if dt.month == 12 else f"{dt.year}-YTD {dt.date()}"
        for k, s in series.items():
            v = s.get(i)
            if pd.isna(v):
                continue
            out.append(_row("DE", "schuldenbericht_umlauf", period, k, stk.get(k, 0.0), v,
                            "Umlaufvolumen incl. retention; green in Nominal; "
                            + ("face" if face else "volume × avg price")))
    return out


# --------------------------------------------------------------------------- FR
def check_fr(ops: pd.DataFrame) -> list[dict]:
    plan = json.loads(FR_PLAN.read_text())["years"]
    out = []
    for y, p in plan.items():
        share, asof = p.get("net_mlt_done_share"), p.get("net_mlt_done_asof")
        if share is None or asof is None or p.get("net_mlt_issuance") is None:
            continue
        asof = pd.Timestamp(asof)
        g = ops[ops["sec_type"].isin(["Nominal", "Linker"]) & (ops["op_dt"].dt.year == int(y))
                & (ops["op_dt"] <= asof)]
        out.append(_row("FR", "aft_net_mlt_done", f"{y}-YTD {asof.date()}", "MLT",
                        g["amount_issued"].sum(), 1000.0 * share * p["net_mlt_issuance"],
                        "AFT % done × programme (rounded share); net of buybacks"))
    return out


# --------------------------------------------------------------------------- IT
def check_it(ops: pd.DataFrame) -> list[dict]:
    b = pd.read_excel(IT_BDI, header=None, skiprows=4)
    b["dt"] = pd.to_datetime(b[0], errors="coerce")
    b = b[b["dt"].notna()]
    grp = {"BOT": "Bill", "CCT": "FRN", "CCTEU": "FRN", "BTPI": "Linker",
           "BTP": "Nominal", "CTZ": "Nominal", "BTPGREEN": "Nominal"}
    b["group"] = b[8].astype(str).str.strip().str.upper().map(grp)
    b["mn"] = pd.to_numeric(b[13], errors="coerce")
    b["year"] = b["dt"].dt.year
    off = b.dropna(subset=["group"]).groupby(["year", "group"])["mn"].sum()
    g = ops[(ops["op_type"] == "Auction") & (ops["amount_issued"] > 0)].copy()
    g["year"] = g["op_dt"].dt.year
    stk = g.groupby(["year", "sec_type"])["amount_issued"].sum()
    return [_row("IT", "bdi_auctions", y, k, stk.get((y, k), 0.0), v, "auctions incl. supplementary")
            for (y, k), v in off.items() if y >= 2015]


# --------------------------------------------------------------------------- ES
# Gross issuance press totals (MINECO / Moncloa), effective € mn, Letras + B&O incl. linkers.
ES_PRESS_TOTAL_MN = {2022: 232_570.0, 2023: 252_000.0, 2024: 259_341.0}
_ES_MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
              "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


def _es_table13() -> list[tuple[int, pd.Timestamp, bool, dict]]:
    """Cumulative effective issuance per year from Tesoro sheet 13 (``acumulados`` tabs).

    Returns (year, cut date, full year, {Bill, Nominal, Linker} € mn). Columns: Letras 2–5,
    Bonos y Obligaciones 6–10, B&O indexados 11; RESTO (loans, other) is left out.
    """
    x = pd.ExcelFile(ES_T13)
    out = []
    for sheet in x.sheet_names:
        r = pd.read_excel(x, sheet_name=sheet, header=None)
        head = " ".join(str(v) for v in r.iloc[:3, 1])
        ym = re.search(r"EN (20\d{2})", head)
        if not ym or not re.search(r"acumulad", head, re.I):
            continue
        year, month, best = int(ym.group(1)), None, None
        for _, row in r.iterrows():
            lab = row.iloc[1]
            key = str(lab).strip().lower()
            if key in _ES_MONTHS:
                month = _ES_MONTHS[key]
                dt = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
            elif key == "total año":
                dt, month = pd.Timestamp(year, 12, 31), 12
            elif isinstance(lab, (int, float)) and pd.notna(lab) and month:
                dt = pd.Timestamp(year, month, int(lab))
            else:
                continue
            if pd.isna(pd.to_numeric(row.iloc[13], errors="coerce")):
                continue
            v = pd.to_numeric(row.iloc[2:12], errors="coerce").fillna(0.0).to_numpy()
            best = (dt, {"Bill": v[0:4].sum(), "Nominal": v[4:9].sum(), "Linker": v[9]})
        if best:
            out.append((year, best[0], best[0] >= pd.Timestamp(year, 12, 31), best[1]))
    return out


def _es_effective(o: pd.DataFrame) -> pd.Series:
    """Ops nominal → Tesoro effective cash, capped at nominal (sheet 13 note 1).

    Bills: act/360 simple discount at the marginal yield. Bonds: clean price off yield,
    annual coupons. Linkers: real price × index ratio is above par, so the cap binds.
    """
    mat = pd.to_datetime(o["maturity_dt"])
    resid = pd.to_numeric(o.get("resid_mat_yrs"), errors="coerce")
    mat = mat.fillna(o["op_dt"] + pd.to_timedelta((resid * 365.25).round(), unit="D"))
    y = pd.to_numeric(o["yield"], errors="coerce") / 100.0
    c = pd.to_numeric(o["coupon"], errors="coerce").fillna(0.0) / 100.0
    days = (mat - o["op_dt"]).dt.days

    def clean(yy, cc, tt):
        if pd.isna(yy) or pd.isna(tt) or tt <= 0:
            return 100.0
        yrs = tt / 365.25
        n = int(np.ceil(yrs - 1e-9))
        f = yrs - (n - 1)
        times = f + np.arange(n)
        return 100 * ((cc * (1 + yy) ** -times).sum() + (1 + yy) ** -times[-1] - cc * (1 - f))

    bill = o["sec_type"].eq("Bill")
    px = pd.Series([clean(a, b, d) for a, b, d in zip(y, c, days)], index=o.index)
    px[bill] = (100 / (1 + y * days / 360))[bill].fillna(100.0)
    px[o["sec_type"].eq("Linker")] = 100.0
    return o["amount_issued"] * px.clip(upper=100.0) / 100.0


def check_es(ops: pd.DataFrame) -> list[dict]:
    g = ops[(ops["amount_issued"] > 0) & ops["sec_type"].isin(["Bill", "Nominal", "Linker"])].copy()
    g["settle"] = g["issue_dt"].fillna(g["op_dt"] + pd.offsets.BDay(2))
    g["eff"] = _es_effective(g)
    g["year"] = g["settle"].dt.year
    out = []
    for y, v in ES_PRESS_TOTAL_MN.items():
        out.append(_row("ES", "tesoro_press_gross", y, "Total", g.loc[g["year"] == y, "eff"].sum(), v,
                        "effective cash; Letras + B&O incl. syndications and linkers"))
    for y, cut, full, ref in _es_table13():
        sub = g[(g["year"] == y) & (g["settle"] <= cut)]
        stk = sub.groupby("sec_type")["eff"].sum()
        period = y if full else f"{y}-YTD {cut.date()}"
        for k, v in ref.items():
            out.append(_row("ES", "tesoro_t13_gross", period, k, stk.get(k, 0.0), v,
                            "sheet 13 effective cash by settlement date; ex RESTO"))
    return out


# --------------------------------------------------------------------------- JP
def _jp_plan() -> dict[str, float]:
    """Market issuance by periodic auctions ex TBs and ex the between-FY adjustment (¥ mn).

    The plan's periodic-auction line = calendar base + NPC II + "adjustment between fiscal
    years" (front-loaded / deferred issuance in the accounting adjustment period), which
    is not auctioned inside the FY's April–March calendar.
    """
    import pypdf

    text = "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(str(JP_PLAN)).pages)
    num = r"([\d.]+)"
    per = re.search(rf"Market Issuance\s*by Periodic Auctions\s+{num}\s+{num}", text)
    tbs = re.search(rf"\nTBs\s+{num}\s+{num}", text)
    sgn = r"(-?[\d,]+\.\d)"
    adj = re.search(rf"Adjustment between\s*fiscal years\s+{sgn}\s+{sgn}\s+{sgn}", text)
    if not per or not tbs or not adj:
        raise ValueError(f"JP plan table not found in {JP_PLAN.name}")
    adj_bn = [float(adj.group(i).replace(",", "")) for i in (2, 3)]  # FY-1 suppl, FY initial
    fy = int(re.search(r"FY(\d{4}) JGB Issuance Plan", text).group(1))
    return {f"FY{fy - 1} suppl": (float(per.group(1)) - float(tbs.group(1))) * 1e6 - adj_bn[0] * 1e3,
            f"FY{fy} initial": (float(per.group(2)) - float(tbs.group(2))) * 1e6 - adj_bn[1] * 1e3}


def check_jp(ops: pd.DataFrame) -> list[dict]:
    plan = _jp_plan()
    g = ops[(ops["sec_type"] != "Bill") & (ops["amount_issued"] > 0)].copy()
    g["fy"] = _fy(g["op_dt"])
    last = g["op_dt"].max()
    fc_path = CACHE / "forecast_priced.parquet"
    fc = pd.DataFrame()
    if fc_path.exists():
        fc = pd.read_parquet(fc_path)
        fc = fc[(fc["country"] == "JP") & (fc["sec_type"].astype(str) != "Bill")].copy()
        fc["op_dt"] = pd.to_datetime(fc["op_dt"])
        fc = fc[fc["op_dt"] > last]
        fc["fy"] = _fy(fc["op_dt"])
    out = []
    for key, v in plan.items():
        fy = int(key[2:6])
        stack = g.loc[g["fy"] == fy, "amount_issued"].sum()
        note = "ops"
        if not fc.empty and pd.Timestamp(f"{fy + 1}-03-31") > last:
            stack += fc.loc[fc["fy"] == fy, "gross_mn"].sum()
            note = f"ops to {last.date()} + forecast_priced"
        out.append(_row("JP", "mof_plan_ex_tbs", key, "Coupon", stack, v,
                        note + "; incl. NPC II, LE; plan ex between-FY adjustment"))
    return out


# --------------------------------------------------------------------------- SE
def check_se(ops: pd.DataFrame) -> list[dict]:
    """Gross outright auctions + syndications (CH926 type O, plus syndicated switches) vs
    ops Auction + Syndication.

    Riksgälden's debt-change table is net borrowing, which needs a redemption ladder the
    ops book does not carry for matured lines; gross is the like-for-like basis.
    """
    a = pd.read_csv(SE_AUCTIONS)
    a["dt"] = pd.to_datetime(a["date"], errors="coerce")
    a["mn"] = pd.to_numeric(a["allocated_mn"], errors="coerce").abs()
    synd = a["auction_type"].astype(str).str.contains("Syndikering", case=False)
    a = a[a["dt"].notna() & (a["type"].eq("O") | synd) & (a["mn"] > 0)]
    up = a["series"].astype(str).str.upper().str.strip()
    a["group"] = np.where(up.str.startswith("STB"), "Bill",
                          np.where(up.str.startswith("SGB IL"), "Linker", "Nominal"))
    a["year"] = a["dt"].dt.year
    last = a["dt"].max()
    off = a.groupby(["year", "group"])["mn"].sum()
    g = ops[ops["op_type"].isin(["Auction", "Syndication"]) & (ops["amount_issued"] > 0)
            & (ops["op_dt"] <= last)].copy()
    g["year"] = g["op_dt"].dt.year
    stk = g.groupby(["year", "sec_type"])["amount_issued"].sum()
    out = []
    for (y, k), v in off.items():
        if y < 2015:
            continue
        period = y if y < last.year else f"{y}-YTD {last.date()}"
        out.append(_row("SE", "rg_gross_auctions", period, k, stk.get((y, k), 0.0), v,
                        "CH926 outright + syndicated allocations; not independent of ops"))
    return out


CHECKS = {"US": check_us, "UK": check_uk, "DE": check_de, "FR": check_fr, "IT": check_it,
          "ES": check_es, "JP": check_jp, "SE": check_se}


def _grade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_mn"] = df["stack_mn"] - df["official_mn"]
    df["delta_pct"] = df["delta_mn"] / df["official_mn"].abs().where(df["official_mn"].abs() > 0)
    floor = df["country"].map(ABS_FLOOR_MN)
    floor_eff = df["country"].map(lambda c: ABS_FLOOR_TEMP_MN.get(c, ABS_FLOOR_MN[c]))
    tgt = df["country"].map(TOL_PCT)
    eff = df["country"].map(lambda c: TOL_PCT_TEMP.get(c, TOL_PCT[c]))
    off = df["official_mn"].abs()
    band_eff = (eff * off).clip(lower=floor_eff)
    band_tgt = (tgt * off).clip(lower=floor)
    # Reported tolerances are the bands applied, floor included.
    df["tol_mn"] = band_eff
    df["tol_pct"] = band_eff / off.where(off > 0)
    df["tol_target_pct"] = band_tgt / off.where(off > 0)
    ok_eff = df["delta_mn"].abs() <= band_eff
    ok_tgt = df["delta_mn"].abs() <= band_tgt
    df["status"] = ok_eff.map({True: "ok", False: "FAIL"})
    df["in_target"] = df["in_target"].fillna(True).astype(bool) if "in_target" in df else True
    df["status_target"] = ok_tgt.map({True: "ok", False: "FAIL"}).where(df["in_target"], "ungated")
    return df


def run(countries=COUNTRIES) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {}
    for cc in countries:
        src = OFFICIAL[cc]
        ops = _ops(cc)
        if not Path(src).exists():
            meta[cc] = {"status": "missing", "official": str(src)}
            continue
        if ops is None or ops.empty:
            meta[cc] = {"status": "no_ops"}
            continue
        r = CHECKS[cc](ops)
        meta[cc] = {"status": "ok", "official": str(Path(src).relative_to(REPO)), "n": len(r)}
        rows += r
    df = _grade(pd.DataFrame(rows)) if rows else pd.DataFrame()
    for cc, m in meta.items():
        if m["status"] != "ok" or df.empty:
            continue
        sub = df[df["country"] == cc]
        m["n_fail"] = int((sub["status"] == "FAIL").sum())
        m["n_fail_target"] = int((sub["status_target"] == "FAIL").sum())
        m["worst_pct"] = round(float(sub["delta_pct"].abs().max()), 4) if len(sub) else None
        m["status"] = "FAIL" if m["n_fail"] else "ok"
    return df, meta


def write(df: pd.DataFrame, meta: dict) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(CACHE / "reconcile_official.parquet", index=False)
    (DERIVED / "reconcile_official.json").write_text(json.dumps(
        {"countries": meta, "tol_pct": TOL_PCT, "tol_pct_temp": TOL_PCT_TEMP,
         "rows": df.round(4).to_dict("records") if not df.empty else []}, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", type=str.upper, default=list(COUNTRIES))
    ap.add_argument("--target", action="store_true", help="gate on TOL_PCT, ignoring TOL_PCT_TEMP")
    ap.add_argument("--all-rows", action="store_true")
    a = ap.parse_args(argv)
    df, meta = run(a.only)
    write(df, meta)
    key = "status_target" if a.target else "status"
    pd.set_option("display.width", 200)
    if not df.empty:
        show = df if a.all_rows else df[(df[key] == "FAIL") | df.groupby("country").cumcount(ascending=False).lt(4)]
        cols = ["country", "check", "period", "group", "stack_mn", "official_mn", "delta_mn",
                "delta_pct", "tol_pct", "tol_target_pct", "status", "status_target"]
        print(show[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}"))
    for cc, m in meta.items():
        print(f"{cc}: {m}")
    fails = df[df[key] == "FAIL"] if not df.empty else df
    if len(fails):
        print(f"\nFAIL: {len(fails)} rows over tolerance in {sorted(fails['country'].unique())}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
