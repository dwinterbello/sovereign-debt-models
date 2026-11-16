"""Euro-area holder sectors of DE / FR / IT general-government debt securities (ECB SHS + BSI + QSA).

Sources (ECB Data Portal SDMX API, raw CSV cached under `cache/ecb_holders/`, parsed offline):

  SHSS  Securities Holdings Statistics by sector (DSD ESTAT NA_SEC), 2021Q1→, quarterly. Holders are
        euro-area residents (REF_AREA U2) by sector, issuer = country general government (COUNTERPART
        S13), debt securities F3, all maturities T. Stocks LE at market (M) and face (F) value,
        transactions F (market). Sectors: S1 total economy (incl. Eurosystem), S121 central bank
        (Eurosystem), S122 deposit-takers, S123 MMF, S124 non-MMF investment funds, S125A FVCs,
        S12P OFI (= S124 + S125A + other non-MFI non-ICPF), S12Q ICPF (= S128 insurers + S129 pension
        funds), S11 NFCs, S13 general government (S1311 central govt), S1M households + NPISH.
        Identities S1 = S11 + S12 + S13 + S1M and S12 = S121 + S122 + S123 + S12P + S12Q hold to the cent.
  SHS   the older public SHS dataflow (DSD ECB_SHS6), 2013Q4–2022Q1: holder area I8, stocks at market
        value only, short (S) + long (L) original maturity summed. No central bank and no investment-fund
        split: S12T (MFIs ex central banks = S122 + S123), S12P, S12Q, S11, S13, S1M, S1N
        (non-sectorised), S1ZK (all euro-area residents ex central banks).
  QSA   quarterly sector accounts: country S13 debt-security liabilities, unconsolidated, market value
        (VALUATION S), LE and F, counterpart W0 (all holders) and W1 (non-residents of the country).
        The total-outstanding denominator, and the revaluation used to turn SHS stock changes into
        transactions before 2021.
  BSI   monthly: MFIs ex NCB (BS_REP_SECTOR A) holdings of debt securities (A30) of the domestic (U6)
        general government (2100), stocks (DATA_TYPE 1) and transactions (4), 1997-09→.

Splice. On the five overlap quarters 2021Q1–2022Q1 SHS and SHSS agree sector by sector (see
`splice_check()`); SHSS is used from 2021Q1, SHS before. Pre-2021 transactions are not published, so
they are estimated as ΔH − H₋₁·r with r the quarter's revaluation rate of the whole stock from QSA
((ΔLE − F) / LE₋₁) — exact if every sector holds the same price mix; `splice_check()` scores it
against published SHSS transactions. The Eurosystem is not in SHS: before 2021 'eurosystem' =
PSPP + PEPP holdings (book value, `cache/ecb_qe_holdings.parquet`) × the country's 2021Q1 ratio of
SHSS S121 (market value) to that book value (DE 0.83: PSPP-DE includes non-S13 agencies; FR 1.05;
IT 1.33: Banca d'Italia's non-QE portfolios). The ratio drifts (IT 1.33 → 1.23 over 2021), so the
pre-2021 Eurosystem, euro-area total and rest-of-world residual carry that error; with no series
before PSPP they start in 2015Q1.

Publication (established from the API's `updatedAfter`, Oct-2026): SHSS 2026Q2 entered between
25 Aug and 1 Sep → a quarter is public in the 3rd month after its quarter-end month (Q2 → Sep).
QSA 2026Q1 entered ~7–12 Jul → 4th month (Q1 → Jul). BSI Aug-2026 entered 25–28 Sep → the next
month. Each row of `metrics()` is stamped at the month its *last* input became public and
forward-filled; `experiments/ecb_holders_test.py` adds one more month of lag on top.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from issuance.paths import CACHE

API = "https://data-api.ecb.europa.eu/service/data"
DIR = CACHE / "ecb_holders"
COUNTRIES = ("DE", "FR", "IT", "ES")
_C = "+".join(COUNTRIES)
SHSS_SECTORS = ("S1", "S11", "S12", "S121", "S122", "S123", "S124", "S125A", "S128", "S129", "S12P", "S12Q",
                "S13", "S1311", "S1M")
SHS_SECTORS = ("S11", "S12P", "S12Q", "S12T", "S13", "S1M", "S1N", "S1ZK")
KEYS = {
    "shss": f"SHSS/Q.N.U2.{_C}.{'+'.join(SHSS_SECTORS)}.S13.N.A.LE+F.F3.T._Z.XDC._T.M+F.V.N._T",
    "shs": f"SHS/Q.I8.{_C}.{'+'.join(SHS_SECTORS)}.S13._X.LE.F3.S+L.EUR.M.A.A.A._X._X.V._T.ZZZ",
    "qsa": f"QSA/Q.N.{_C}.W0+W1.S13.S1.N.L.LE+F.F3.T._Z.XDC._T.S.V.N._T",
    "bsi": f"BSI/M.{_C}.N.A.A30.A.1+4.U6.2100.Z01.E",
}
SOURCE = {"shss": "ECB SHSS (sector), market value", "shs": "ECB SHS (I8, pre-2021), market value",
          "qsa": "ECB QSA S13 F3 liabilities, market value", "bsi": "ECB BSI MFIs ex NCB, domestic GG debt",
          "qe": "ECB PSPP+PEPP holdings (book value)", "derived": "derived"}

# Canonical sectors. SHSS codes map 1:1; SHS codes onto the coarser ones it has.
SECTOR_LABEL = {
    "total": "Total outstanding (QSA)", "ea": "Euro-area holders, all", "eurosystem": "Eurosystem",
    "ea_ex_cb": "Euro-area holders ex Eurosystem", "banks": "Banks (deposit-takers)", "mmf": "Money-market funds",
    "mfi_ex_cb": "Banks + MMF", "funds": "Investment funds", "ofi": "Other financial (OFI, incl. funds)",
    "icpf": "Insurers + pension funds", "insurers": "Insurers", "pension": "Pension funds",
    "households": "Households", "nfc": "Non-financial corporations", "government": "General government",
    "nonres": "Non-residents of the issuer country (QSA)", "row": "Non-euro-area (residual)",
}
SHSS_MAP = {"S1": "ea", "S121": "eurosystem", "S122": "banks", "S123": "mmf", "S124": "funds", "S12P": "ofi",
            "S12Q": "icpf", "S128": "insurers", "S129": "pension", "S1M": "households", "S11": "nfc",
            "S13": "government"}
SHS_MAP = {"S12T": "mfi_ex_cb", "S12P": "ofi", "S12Q": "icpf", "S1M": "households", "S11": "nfc",
           "S13": "government", "S1ZK": "ea_ex_cb"}
# Sectors that sum to euro-area holders (SHSS); the absorption table splits them this way.
EA_PARTS = ("eurosystem", "mfi_ex_cb", "icpf", "ofi", "households", "nfc", "government")
SHSS_START = pd.Timestamp("2021-03-31")
SHSS_PUBLIC_MONTHS = 3
QSA_PUBLIC_MONTHS = 4
BSI_PUBLIC_MONTHS = 1


# ---------------------------------------------------------------- fetch
def _url(name: str) -> str:
    return f"{API}/{KEYS[name]}?format=csvdata"


def fetch(refresh: bool = False) -> dict[str, Path]:
    """Cached CSV per dataset; downloads when absent or `refresh`. A failed download keeps the cache."""
    from issuance.http import get_bytes

    DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in KEYS:
        path = DIR / f"{name}.csv"
        if refresh or not path.exists():
            try:
                payload = get_bytes(_url(name), timeout=180, min_bytes=500)
                if not payload.startswith(b"KEY,"):
                    raise RuntimeError(f"ECB {name}: not SDMX csvdata ({payload[:80]!r})")
                path.write_bytes(payload)
            except Exception:
                if not path.exists():
                    raise
        out[name] = path
    return out


def _read(name: str, path: Path | None = None) -> pd.DataFrame:
    path = DIR / f"{name}.csv" if path is None else path
    if not path.exists():
        fetch()
    d = pd.read_csv(path, dtype=str, low_memory=False)
    d["OBS_VALUE"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    mult = pd.to_numeric(d.get("UNIT_MULT", 6), errors="coerce").fillna(6)
    d["value_eur_bn"] = d["OBS_VALUE"] * 10.0 ** (mult - 9)
    return d.dropna(subset=["value_eur_bn"])


def _quarter_end(label: pd.Series) -> pd.Series:
    return pd.PeriodIndex(label.str.replace("-", ""), freq="Q").to_timestamp(how="end").normalize()


# ---------------------------------------------------------------- panels
def shss_raw(path: Path | None = None) -> pd.DataFrame:
    """country, quarter_end, code, sector, measure (LE / F), valuation (M / F), value_eur_bn."""
    d = _read("shss", path)
    d = d[(d["REF_AREA"] == "U2") & (d["COUNTERPART_SECTOR"] == "S13") & (d["INSTR_ASSET"] == "F3")
          & (d["MATURITY"] == "T") & d["COUNTERPART_AREA"].isin(COUNTRIES)]
    return pd.DataFrame({"country": d["COUNTERPART_AREA"].to_numpy(), "quarter_end": _quarter_end(d["TIME_PERIOD"]),
                         "code": d["REF_SECTOR"].to_numpy(), "sector": d["REF_SECTOR"].map(SHSS_MAP).to_numpy(),
                         "measure": d["STO"].to_numpy(), "valuation": d["VALUATION"].to_numpy(),
                         "value_eur_bn": d["value_eur_bn"].to_numpy()})


def shs_raw(path: Path | None = None) -> pd.DataFrame:
    """Old SHS: country, quarter_end, code, sector, measure 'LE', value_eur_bn (short + long summed)."""
    d = _read("shs", path)
    d = d[(d["ISSUER_SECTOR"] == "S13") & (d["INSTR_ASSET"] == "F3") & d["ISSUER_AREA"].isin(COUNTRIES)]
    d = d.assign(country=d["ISSUER_AREA"], quarter_end=_quarter_end(d["TIME_PERIOD"]), code=d["HOLDER_SECTOR"])
    # A sector missing one maturity in a quarter is incomplete, not zero.
    g = d.groupby(["country", "quarter_end", "code"]).agg(value_eur_bn=("value_eur_bn", "sum"),
                                                         n=("MATURITY", "nunique")).reset_index()
    g = g[(g["n"] == 2) | (g["code"] == "S1N")]
    g["sector"] = g["code"].map(SHS_MAP)
    g["measure"] = "LE"
    return g.drop(columns="n")


def qsa_raw(path: Path | None = None) -> pd.DataFrame:
    """country, quarter_end, sector ('total' W0 / 'nonres' W1), measure, value_eur_bn."""
    d = _read("qsa", path)
    d = d[(d["REF_SECTOR"] == "S13") & (d["ACCOUNTING_ENTRY"] == "L") & d["REF_AREA"].isin(COUNTRIES)]
    return pd.DataFrame({"country": d["REF_AREA"].to_numpy(), "quarter_end": _quarter_end(d["TIME_PERIOD"]),
                         "sector": d["COUNTERPART_AREA"].map({"W0": "total", "W1": "nonres"}).to_numpy(),
                         "measure": d["STO"].to_numpy(), "value_eur_bn": d["value_eur_bn"].to_numpy()})


def bsi_raw(path: Path | None = None) -> pd.DataFrame:
    """country, month, measure (LE stocks / F transactions), value_eur_bn."""
    d = _read("bsi", path)
    d = d[(d["BS_ITEM"] == "A30") & (d["COUNT_AREA"] == "U6") & (d["BS_COUNT_SECTOR"] == "2100")
          & d["REF_AREA"].isin(COUNTRIES)]
    return pd.DataFrame({"country": d["REF_AREA"].to_numpy(),
                         "month": pd.PeriodIndex(d["TIME_PERIOD"], freq="M").to_timestamp(),
                         "measure": d["DATA_TYPE"].map({"1": "LE", "4": "F"}).to_numpy(),
                         "value_eur_bn": d["value_eur_bn"].to_numpy()})


def bsi_panel(path: Path | None = None) -> pd.DataFrame:
    """Domestic MFIs' (ex NCB) holdings of domestic general-government debt: country, month, value_eur_bn
    (stock) and flow_eur_bn (transactions, revaluation-free)."""
    r = bsi_raw(path)
    w = r.pivot_table(index=["country", "month"], columns="measure", values="value_eur_bn").reset_index()
    w = w.rename(columns={"LE": "value_eur_bn", "F": "flow_eur_bn"})
    w["source"] = SOURCE["bsi"]
    return w[["country", "month", "value_eur_bn", "flow_eur_bn", "source"]].sort_values(["country", "month"]) \
        .reset_index(drop=True)


def _qe_quarterly() -> pd.DataFrame:
    """PSPP + PEPP holdings by country at quarter end (book value, EUR bn)."""
    p = CACHE / "ecb_qe_holdings.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["country", "quarter_end", "value_eur_bn"])
    q = pd.read_parquet(p)
    q = q[q["country"].isin(COUNTRIES)].copy()
    q["month"] = pd.to_datetime(q["month"])
    m = q.groupby(["country", "month"])["holdings_mn"].sum().reset_index()
    out = []
    for c, g in m.groupby("country"):
        s = g.set_index("month")["holdings_mn"].sort_index().asfreq("MS").ffill()
        s = s[s.index.month.isin([3, 6, 9, 12])]
        out.append(pd.DataFrame({"country": c, "quarter_end": s.index + pd.offsets.MonthEnd(0),
                                 "value_eur_bn": s.to_numpy() / 1e3}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["country", "quarter_end", "value_eur_bn"])


def shss_panel(shss_path: Path | None = None, shs_path: Path | None = None,
               qsa_path: Path | None = None, qe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Spliced long panel: country, quarter_end, sector, measure (LE / F), value_eur_bn, source.

    LE market value. SHSS from 2021Q1, SHS before (stocks, with F estimated from the QSA revaluation
    rate). Adds 'total' and 'nonres' (QSA), 'eurosystem' before 2021 (PSPP + PEPP), the canonical
    aggregates 'mfi_ex_cb' and 'ea_ex_cb', and 'row' = total − euro-area holders."""
    cols = ["country", "quarter_end", "sector", "measure", "value_eur_bn", "source"]
    new = shss_raw(shss_path)
    new = new[(new["valuation"] == "M") & new["sector"].notna()]
    new = new.assign(source=SOURCE["shss"])[cols]
    w = new.pivot_table(index=["country", "quarter_end", "measure"], columns="sector", values="value_eur_bn")
    agg = pd.DataFrame({"mfi_ex_cb": w["banks"] + w["mmf"], "ea_ex_cb": w["ea"] - w["eurosystem"]})
    agg.columns.name = "sector"
    agg = agg.stack().rename("value_eur_bn").reset_index()
    new = pd.concat([new, agg.assign(source=SOURCE["shss"])[cols]], ignore_index=True)

    qsa = qsa_raw(qsa_path).assign(source=SOURCE["qsa"])
    old = shs_raw(shs_path)
    old = old[old["sector"].notna() & (old["quarter_end"] < SHSS_START)].assign(source=SOURCE["shs"])
    qe = _qe_quarterly() if qe is None else qe
    s121 = new[(new["sector"] == "eurosystem") & (new["measure"] == "LE") & (new["quarter_end"] == SHSS_START)]
    ratio = s121.set_index("country")["value_eur_bn"] / qe[qe["quarter_end"] == SHSS_START].set_index("country")[
        "value_eur_bn"]
    qe_old = qe[qe["quarter_end"] < SHSS_START].copy()
    qe_old["value_eur_bn"] = qe_old["value_eur_bn"] * qe_old["country"].map(ratio)
    qe_old = qe_old.dropna(subset=["value_eur_bn"]).assign(sector="eurosystem", measure="LE", source=SOURCE["qe"])
    old = pd.concat([old[cols], qe_old[cols]], ignore_index=True)
    ow = old.pivot_table(index=["country", "quarter_end"], columns="sector", values="value_eur_bn")
    if "eurosystem" in ow:
        # No Eurosystem series before PSPP (2015Q1): euro-area total and the residual start there.
        ea = (ow["ea_ex_cb"] + ow["eurosystem"]).dropna().rename("value_eur_bn")
        old = pd.concat([old, ea.reset_index().assign(sector="ea", measure="LE", source=SOURCE["shs"])[cols]],
                        ignore_index=True)
    old = pd.concat([old, _implied_flows(old, qsa)], ignore_index=True)

    out = pd.concat([old, new, qsa[cols]], ignore_index=True)
    tot = qsa[qsa["sector"] == "total"].set_index(["country", "quarter_end", "measure"])["value_eur_bn"]
    ea = out[out["sector"] == "ea"].set_index(["country", "quarter_end", "measure"])["value_eur_bn"]
    row = (tot - ea).dropna().rename("value_eur_bn").reset_index()
    out = pd.concat([out, row.assign(sector="row", source=SOURCE["derived"])[cols]], ignore_index=True)
    return out.sort_values(["country", "sector", "measure", "quarter_end"]).reset_index(drop=True)


def reval_rate(qsa: pd.DataFrame) -> pd.Series:
    """Quarterly revaluation rate of the whole stock: (ΔLE − F) / LE₋₁, index (country, quarter_end)."""
    t = qsa[qsa["sector"] == "total"].pivot_table(index=["country", "quarter_end"], columns="measure",
                                                 values="value_eur_bn").sort_index()
    le_prev = t.groupby(level="country")["LE"].shift(1)
    return ((t["LE"] - le_prev - t["F"]) / le_prev).rename("r")


def _implied_flows(stocks: pd.DataFrame, qsa: pd.DataFrame) -> pd.DataFrame:
    """F ≈ H − H₋₁·(1 + r) per sector, r from `reval_rate` (consecutive quarters only)."""
    r = reval_rate(qsa)
    out = []
    for (c, s), g in stocks[stocks["measure"] == "LE"].groupby(["country", "sector"]):
        h = g.set_index("quarter_end")["value_eur_bn"].sort_index()
        h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="QE"))
        rr = r.xs(c, level="country").reindex(h.index) if c in r.index.get_level_values(0) else np.nan
        f = (h - h.shift(1) * (1 + rr)).dropna()
        out.append(pd.DataFrame({"country": c, "quarter_end": f.index, "sector": s, "measure": "F",
                                 "value_eur_bn": f.to_numpy(), "source": g["source"].iloc[0] + " (F implied)"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=stocks.columns)


def wide(p: pd.DataFrame | None = None, measure: str = "LE") -> pd.DataFrame:
    """country × quarter_end, one column per sector."""
    p = shss_panel() if p is None else p
    return p[p["measure"] == measure].pivot_table(index=["country", "quarter_end"], columns="sector",
                                                  values="value_eur_bn").reset_index()


# ---------------------------------------------------------------- checks
def sector_sums(path: Path | None = None) -> pd.DataFrame:
    """SHSS identities per country-quarter-measure: S1 vs S11+S12+S13+S1M, S12 vs its five parts,
    S12Q vs S128+S129 (gaps in EUR bn)."""
    r = shss_raw(path)
    r = r[r["valuation"] == "M"]
    w = r.pivot_table(index=["country", "quarter_end", "measure"], columns="code", values="value_eur_bn")
    return pd.DataFrame({
        "gap_s1": w["S1"] - w[["S11", "S12", "S13", "S1M"]].sum(axis=1),
        "gap_s12": w["S12"] - w[["S121", "S122", "S123", "S12P", "S12Q"]].sum(axis=1),
        "gap_icpf": w["S12Q"] - w[["S128", "S129"]].sum(axis=1),
        "s1": w["S1"],
    }).reset_index()


def splice_check(shss_path: Path | None = None, shs_path: Path | None = None, qsa_path: Path | None = None,
                 qe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Overlap 2021Q1–2022Q1 per country × sector: SHS vs SHSS stock (mean gap, % of SHSS), and the
    implied-transaction estimate vs published SHSS F (corr, mean abs gap), plus PSPP+PEPP vs S121."""
    new = shss_raw(shss_path)
    new = new[(new["valuation"] == "M")]
    nw = new.pivot_table(index=["country", "quarter_end", "measure"], columns="code", values="value_eur_bn")
    nw["S12T"], nw["S1ZK"] = nw["S122"] + nw["S123"], nw["S1"] - nw["S121"]
    old = shs_raw(shs_path).pivot_table(index=["country", "quarter_end"], columns="code", values="value_eur_bn")
    qsa = qsa_raw(qsa_path)
    rows = []
    for code in ("S12T", "S12P", "S12Q", "S1M", "S11", "S13", "S1ZK"):
        a = old[code].rename("shs")
        b = nw.xs("LE", level="measure")[code].rename("shss")
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        # implied F from SHSS stocks vs the published F (2021Q2→)
        st = nw.xs("LE", level="measure")[code].reset_index().rename(columns={code: "value_eur_bn"})
        st = st.assign(sector=code, measure="LE", source="shss")
        imp = _implied_flows(st, qsa).set_index(["country", "quarter_end"])["value_eur_bn"]
        pub = nw.xs("F", level="measure")[code]
        f = pd.concat([imp.rename("imp"), pub.rename("pub")], axis=1, join="inner").dropna()
        for c in COUNTRIES:
            jj, ff = j.xs(c, level="country") if c in j.index.get_level_values(0) else j.iloc[:0], \
                f.xs(c, level="country") if c in f.index.get_level_values(0) else f.iloc[:0]
            rows.append({"country": c, "code": code, "n_overlap": len(jj),
                         "shss_bn": jj["shss"].mean(), "gap_bn": (jj["shs"] - jj["shss"]).mean(),
                         "gap_pct": 100 * ((jj["shs"] - jj["shss"]) / jj["shss"]).mean(),
                         "n_flow": len(ff), "flow_corr": ff["imp"].corr(ff["pub"]) if len(ff) > 3 else np.nan,
                         "flow_mae_bn": (ff["imp"] - ff["pub"]).abs().mean(), "flow_abs_bn": ff["pub"].abs().mean()})
    qe = _qe_quarterly() if qe is None else qe
    s121 = nw.xs("LE", level="measure")["S121"].rename("s121")
    j = pd.concat([qe.set_index(["country", "quarter_end"])["value_eur_bn"].rename("qe"), s121], axis=1,
                  join="inner").dropna()
    for c in COUNTRIES:
        jj = j.xs(c, level="country") if c in j.index.get_level_values(0) else j.iloc[:0]
        rows.append({"country": c, "code": "S121 vs PSPP+PEPP", "n_overlap": len(jj), "shss_bn": jj["s121"].mean(),
                     "gap_bn": (jj["qe"] - jj["s121"]).mean(),
                     "gap_pct": 100 * ((jj["qe"] - jj["s121"]) / jj["s121"]).mean()})
    return pd.DataFrame(rows)


def coverage(p: pd.DataFrame | None = None, b: pd.DataFrame | None = None) -> pd.DataFrame:
    p = shss_panel() if p is None else p
    b = bsi_panel() if b is None else b
    rows = []
    for (src, m), g in p.groupby(["source", "measure"]):
        rows.append({"source": src, "measure": m, "first": g["quarter_end"].min().date(),
                     "last": g["quarter_end"].max().date(), "sectors": " ".join(sorted(g["sector"].unique()))})
    rows.append({"source": SOURCE["bsi"], "measure": "LE+F", "first": b["month"].min().date(),
                 "last": b["month"].max().date(), "sectors": "domestic MFIs ex NCB"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- publication stamping
def shss_public_month(quarter_end) -> pd.Timestamp:
    """SHS(S): public in the 3rd month after the quarter-end month (Q2 → September)."""
    return (pd.Timestamp(quarter_end).to_period("M") + SHSS_PUBLIC_MONTHS).to_timestamp()


def qsa_public_month(quarter_end) -> pd.Timestamp:
    """QSA: public in the 4th month after the quarter-end month (Q1 → July)."""
    return (pd.Timestamp(quarter_end).to_period("M") + QSA_PUBLIC_MONTHS).to_timestamp()


def bsi_public_month(month) -> pd.Timestamp:
    """BSI: the month after the data month (published ~4 weeks after month end)."""
    return (pd.Timestamp(month).to_period("M") + BSI_PUBLIC_MONTHS).to_timestamp()


# ---------------------------------------------------------------- metrics
METRICS: dict[str, dict] = {
    "ecb_bank_share": {
        "label": "Euro-area bank share", "unit": "% of outstanding", "sign": -1, "fmt": ".1f", "lag": "SHS+QSA",
        "what": "Euro-area banks + MMFs (SHSS S122+S123; SHS S12T before 2021) ÷ total outstanding (QSA), market "
                "value. Banks are a less price-elastic, carry/regulation-driven holder (Fang–Hardy–Lewis; "
                "Koijen et al. 2021 low elasticity for banks): a larger bank share = more demand already placed."},
    "ecb_bank_d4q": {
        "label": "Euro-area bank share, 4q change", "unit": "pp", "sign": -1, "fmt": ".1f", "lag": "SHS+QSA",
        "what": "Four-quarter change in the euro-area bank share: banks taking a rising share of the stock."},
    "ecb_icpf_share": {
        "label": "Insurer + pension share", "unit": "% of outstanding", "sign": -1, "fmt": ".1f", "lag": "SHS+QSA",
        "what": "ICPF (S12Q) holdings ÷ total outstanding. Liability-matching, the least price-elastic private "
                "holder (Koijen–Yogo; Koijen–Koulischer–Nguyen–Yogo 2021 insurers/pensions inelastic)."},
    "ecb_icpf_f4q": {
        "label": "Insurer + pension 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "SHS+QSA",
        "what": "ICPF transactions over four quarters ÷ GDP (pre-2021 estimated from stocks net of the QSA "
                "revaluation rate). Inelastic buying = demand."},
    "ecb_funds_f4q": {
        "label": "Investment-fund 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "SHS+QSA",
        "what": "Non-MFI, non-ICPF financial (S12P, ~95% investment funds) transactions over four quarters ÷ GDP. "
                "Funds are the most elastic euro-area holder (KKNY): they buy when bonds are cheap, so the "
                "theory sign (−) is the demand reading; a mean-reversion reading would show as 'wrong sign'."},
    "ecb_hh_f4q": {
        "label": "Household 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "SHS+QSA",
        "what": "Households + NPISH (S1M) transactions over four quarters ÷ GDP (retail BTP Valore / Italia "
                "programmes post-2022)."},
    "ecb_row_share": {
        "label": "Non-euro-area share (residual)", "unit": "% of outstanding", "sign": 1, "fmt": ".1f",
        "lag": "SHS+QSA",
        "what": "1 − euro-area holders (incl. Eurosystem) ÷ total outstanding. Foreign holders are the most "
                "price-elastic (KKNY: elasticity of the rest of the world ≫ banks/ICPF; Fang–Hardy–Lewis: "
                "foreign nonbanks carry the marginal absorption), so a high share = demand that leaves first "
                "and more future supply pressure (+). Pre-2021 the Eurosystem is PSPP+PEPP only."},
    "ecb_row_f4q": {
        "label": "Non-euro-area 4q net purchases", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "SHS+QSA",
        "what": "Total net issuance (QSA F) less euro-area transactions, four quarters ÷ GDP. Foreign buying = "
                "demand (−)."},
    "ecb_ea_ex_cb_f4q": {
        "label": "Euro-area private 4q net purchases", "unit": "% of GDP", "sign": 1, "fmt": ".2f",
        "lag": "SHS+QSA",
        "what": "Euro-area holders ex Eurosystem, four-quarter transactions ÷ GDP: net supply the private "
                "domestic euro-area balance sheet had to absorb (+, like `absorb`)."},
    "bsi_bank_gdp": {
        "label": "Domestic bank holdings / GDP", "unit": "% of GDP", "sign": -1, "fmt": ".1f", "lag": "BSI",
        "what": "Domestic MFIs (ex NCB) holdings of domestic general-government debt securities ÷ GDP (BSI). "
                "Bank–sovereign nexus level."},
    "bsi_bank_f3m": {
        "label": "Domestic bank buying, 3m", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "BSI",
        "what": "Domestic MFIs' net purchases of domestic GG debt over 3 months (BSI transactions, "
                "revaluation-free) ÷ GDP. More bank buying = more demand (Altavilla–Pagano–Simonelli: domestic "
                "banks buy home debt in stress)."},
    "bsi_bank_f12m": {
        "label": "Domestic bank buying, 12m", "unit": "% of GDP", "sign": -1, "fmt": ".2f", "lag": "BSI",
        "what": "As the 3m measure over 12 months."},
}


def _gdp_quarterly() -> dict[str, pd.Series]:
    from issuance import market

    g = market.load_gdp()
    return {c: x.set_index("quarter")["gdp_bn"].sort_index() for c, x in g[g["country"].isin(COUNTRIES)].groupby("country")}


def _gdp_at(months: pd.DatetimeIndex, q: pd.Series) -> pd.Series:
    from issuance.headline import gdp_monthly

    return gdp_monthly(pd.DatetimeIndex(months), q)


def quarterly_metrics(p: pd.DataFrame | None = None, gdp: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """country, quarter_end, metric, value, public (month the inputs are all public), before ffill.
    GDP in % of GDP metrics is the annualised quarterly value public by the stamp month."""
    p = shss_panel() if p is None else p
    gdp = _gdp_quarterly() if gdp is None else gdp
    le, fl = wide(p, "LE"), wide(p, "F")
    rows = []
    for c in COUNTRIES:
        L = le[le["country"] == c].drop(columns="country").set_index("quarter_end").sort_index()
        F = fl[fl["country"] == c].drop(columns="country").set_index("quarter_end").sort_index()
        if L.empty or "total" not in L:
            continue
        idx = pd.date_range(L.index.min(), L.index.max(), freq="QE")
        L, F = L.reindex(idx), F.reindex(idx)
        tot = L["total"].where(L["total"] > 0)
        f4 = F.rolling(4, min_periods=4).sum()
        public = pd.DatetimeIndex([max(shss_public_month(q), qsa_public_month(q)) for q in idx])
        g = _gdp_at(public, gdp[c]) if c in gdp else pd.Series(np.nan, index=public)
        gv = g.to_numpy()
        bank = 100 * L["mfi_ex_cb"] / tot
        vals = {
            "ecb_bank_share": bank,
            "ecb_bank_d4q": bank - bank.shift(4),
            "ecb_icpf_share": 100 * L["icpf"] / tot,
            "ecb_icpf_f4q": 100 * f4["icpf"] / gv,
            "ecb_funds_f4q": 100 * f4["ofi"] / gv,
            "ecb_hh_f4q": 100 * f4["households"] / gv,
            "ecb_row_share": 100 * L["row"] / tot,
            "ecb_row_f4q": 100 * f4["row"] / gv,
            "ecb_ea_ex_cb_f4q": 100 * f4["ea_ex_cb"] / gv,
        }
        for k, v in vals.items():
            rows.append(pd.DataFrame({"country": c, "quarter_end": idx, "metric": k, "value": v.to_numpy(),
                                      "public": public}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["country", "quarter_end", "metric", "value", "public"])
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])


def monthly_metrics(b: pd.DataFrame | None = None, gdp: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """BSI metrics: country, month (data), metric, value, public."""
    b = bsi_panel() if b is None else b
    gdp = _gdp_quarterly() if gdp is None else gdp
    rows = []
    for c, g in b.groupby("country"):
        if c not in gdp:
            continue
        s = g.set_index("month").sort_index().asfreq("MS")
        public = pd.DatetimeIndex([bsi_public_month(m) for m in s.index])
        gv = _gdp_at(public, gdp[c]).to_numpy()
        vals = {"bsi_bank_gdp": 100 * s["value_eur_bn"] / gv,
                "bsi_bank_f3m": 100 * s["flow_eur_bn"].rolling(3, min_periods=3).sum() / gv,
                "bsi_bank_f12m": 100 * s["flow_eur_bn"].rolling(12, min_periods=12).sum() / gv}
        for k, v in vals.items():
            rows.append(pd.DataFrame({"country": c, "month": s.index, "metric": k, "value": v.to_numpy(),
                                      "public": public}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["country", "month", "metric", "value", "public"])
    return out.dropna(subset=["value"])


def _stamp(df: pd.DataFrame, key: str, end: pd.Timestamp) -> pd.DataFrame:
    """Rows stamped at `public`, forward-filled monthly to `end`; when two data periods share a public
    month the later one is what was public."""
    out = []
    df = df[df["public"] <= end]
    for (m, c), g in df.groupby(["metric", "country"]):
        s = g.sort_values(key).groupby("public")["value"].last()
        idx = pd.date_range(s.index.min(), end, freq="MS")
        s = s.reindex(idx).ffill()
        out.append(pd.DataFrame({"country": c, "month": idx, "metric": m, "value": s.to_numpy(), "phase": "history"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def metrics(today: pd.Timestamp | None = None, p: pd.DataFrame | None = None, b: pd.DataFrame | None = None,
            gdp: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Monthly rows in `headline.panel()` shape (country, month, metric, value, phase='history'), each
    value at the month it became public, forward-filled through the last complete month before `today`."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    end = today.to_period("M").to_timestamp() - pd.DateOffset(months=1)
    gdp = _gdp_quarterly() if gdp is None else gdp
    parts = [_stamp(quarterly_metrics(p, gdp), "quarter_end", end), _stamp(monthly_metrics(b, gdp), "month", end)]
    parts = [x for x in parts if not x.empty]
    cols = ["country", "month", "metric", "value", "phase"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols].sort_values(["metric", "country", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------- QT-era absorption
ABSORB_PARTS = {"eurosystem": "Eurosystem", "mfi_ex_cb": "Banks+MMF", "icpf": "ICPF", "ofi": "Funds/OFI",
                "households": "Households", "other_ea": "Other EA (NFC, govt)", "row": "Rest of world"}


def absorption(p: pd.DataFrame | None = None, start: str = "2022-12-31") -> pd.DataFrame:
    """Four-quarter net issuance (QSA transactions) split into holder transactions, EUR bn, for every
    Q4 from `start` plus the latest available four quarters. other_ea = EA total − the named EA sectors;
    row = total − EA (so the parts sum to the total exactly)."""
    p = shss_panel() if p is None else p
    f = wide(p, "F")
    rows = []
    for c, g in f.groupby("country"):
        g = g.drop(columns="country").set_index("quarter_end").sort_index()
        g = g[g.index >= SHSS_START]
        r4 = g.rolling(4, min_periods=4).sum().dropna(subset=["total", "ea"])
        ends = [q for q in r4.index if q >= pd.Timestamp(start) and q.month == 12]
        if len(r4) and r4.index[-1] not in ends:
            ends.append(r4.index[-1])
        for q in ends:
            x = r4.loc[q]
            named = ["eurosystem", "mfi_ex_cb", "icpf", "ofi", "households"]
            row = {"country": c, "four_q_to": q.to_period("Q").strftime("%YQ%q"), "net_issuance": x["total"]}
            for k in named:
                row[k] = x[k]
            row["other_ea"] = x["ea"] - x[named].sum()
            row["row"] = x["total"] - x["ea"]
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    fetch()
    print(coverage().to_string(index=False))
    print(splice_check().round(2).to_string(index=False))
    print(absorption().round(1).to_string(index=False))
