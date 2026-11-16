"""Who holds the outstanding debt.

These are stocks. The sectors add to the published debt stock:
Eurostat (DE, annual), BdF DET (FR OAT, quarterly — no public monthly nonres),
BoI TCCE0200 (IT, monthly), BdE BE.22.44 (ES LET+BYO, monthly from 2016),
ONS (UK, quarterly), Treasury OFS-2 (US, quarterly). They do not add to that
year's gross issuance.
"""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

from issuance.paths import CACHE

_ES_MONTHS = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}
ES_URL = "https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/be2244.csv"
FR_DET_URL = "https://comptespublics.fr/series/detention-dette-etat/donnees.csv"

EUROSTAT = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/gov_10dd_ggd/"
    "A.GD.S1+S121+S122_S123+S124-S129+S11+S14_S15+S2.S13.TOTAL.MIO_EUR.DE+FR+IT"
)
HOLDERS = {
    "S121": "central_bank",
    "S122_S123": "banks",
    "S124-S129": "other_financial",
    "S11": "nonfinancial",
    "S14_S15": "households",
    "S2": "rest_of_world",
}
ONS = {
    "nizc": ("insurance_pension", "economy/grossdomesticproductgdp", "ukea"),
    "nntw": ("banks_and_boe", "economy/grossdomesticproductgdp", "ukea"),
    "njsa": ("other_financial", "economy/grossdomesticproductgdp", "ukea"),
    "nldu": ("rest_of_world", "economy/grossdomesticproductgdp", "ukea"),
    "nisu": ("households", "economy/grossdomesticproductgdp", "ukea"),
    "nkky": ("nonfinancial", "economy/grossdomesticproductgdp", "ukea"),
    "njfk": ("local_government", "economy/grossdomesticproductgdp", "ukea"),
    "nkeq": ("public_corporations", "economy/grossdomesticproductgdp", "ukea"),
    "nyxr": ("domestic_total", "economy/grossdomesticproductgdp", "ukea"),
    "j8mv": ("gilt_stock_face", "economy/governmentpublicsectorandtaxes/publicsectorfinance", "pusf"),
}
US_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v1/accounting/tb/ofs2_estimated_ownership_treasury_securities"
)
US_HOLDERS = {
    "Depository Institutions": "banks",
    "U.S. Savings Bonds": "savings_bonds",
    "Pension Funds - Private": "private_pensions",
    "Pension Funds - State And Local Governments": "public_pensions",
    "Insurance Companies": "insurance",
    "Mutual Funds": "mutual_funds",
    "State And Local Governments": "state_local",
    "Foreign And International": "rest_of_world",
    "Other Investors": "other_investors",
    "Federal Reserve And Government Accounts": "fed_and_government",
}
US_TOTAL = "Total Public Debt"
IT_URL = (
    "https://a2a.bancaditalia.it/infostat/dataservices/export/EN/CSV/DATA/"
    "CUBE/BANKITALIA/DIFF/TCCE0200"
)
IT_F3 = {
    "S121": "central_bank",
    "S12BI1": "banks",
    "S12BI2": "other_financial",
    "SBI1": "other_residents",
    "S2": "rest_of_world",
}


def _eurostat() -> pd.DataFrame:
    response = requests.get(EUROSTAT, params={"format": "SDMX-CSV", "startPeriod": "2005"}, timeout=90)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text))
    raw = raw[raw["sector2"].isin(list(HOLDERS) + ["S1"])].copy()
    raw["holder"] = raw["sector2"].map(HOLDERS).fillna("domestic_total")
    raw["amount_mn"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    out = raw.rename(columns={"geo": "country", "TIME_PERIOD": "period"})
    out["period"] = out["period"].astype(int).astype(str)
    out["freq"] = "A"
    return out[["country", "period", "freq", "holder", "amount_mn"]]


def _residual(frame: pd.DataFrame) -> pd.DataFrame:
    """DE and FR publish the financial sectors and the domestic total, not households."""
    parts = ["central_bank", "banks", "other_financial", "nonfinancial", "households"]
    rows = []
    for (country, period), grp in frame.groupby(["country", "period"]):
        domestic = grp.loc[grp["holder"].eq("domestic_total"), "amount_mn"]
        if domestic.empty:
            continue
        known = grp.loc[grp["holder"].isin(parts), "amount_mn"].sum()
        gap = float(domestic.iloc[0]) - float(known)
        if gap > 1:
            rows.append({
                "country": country, "period": period, "freq": "A",
                "holder": "other_domestic", "amount_mn": gap,
            })
    if not rows:
        return frame
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def _ons_one(cdid: str, topic: str, dataset: str) -> pd.DataFrame:
    url = f"https://www.ons.gov.uk/{topic}/timeseries/{cdid}/{dataset}/data"
    response = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    quarters = response.json().get("quarters") or []
    rows = []
    for row in quarters:
        value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.isna(value):
            continue
        rows.append({"period": row["date"], "amount_mn": float(value)})
    return pd.DataFrame(rows)


def _us() -> pd.DataFrame:
    response = requests.get(
        US_URL,
        params={"page[size]": 1000, "fields": "end_of_month,securities_owner,securities_bil_amt"},
        timeout=60,
    )
    response.raise_for_status()
    raw = pd.DataFrame(response.json()["data"])
    raw["holder"] = raw["securities_owner"].map(US_HOLDERS)
    raw["amount_mn"] = pd.to_numeric(raw["securities_bil_amt"], errors="coerce") * 1000
    totals = raw[raw["securities_owner"].eq(US_TOTAL)][["end_of_month", "amount_mn"]].rename(
        columns={"end_of_month": "period", "amount_mn": "stock_mn"}
    )
    out = raw[raw["holder"].notna()].copy()
    out["country"] = "US"
    out["freq"] = "Q"
    out["period"] = out["end_of_month"]
    out = out.merge(totals, on="period", how="left")
    return out[["country", "period", "freq", "holder", "amount_mn", "stock_mn"]]


def _italy() -> pd.DataFrame:
    """Monthly debt securities of general government, by holder. Banca d'Italia TCCE0200."""
    response = requests.get(IT_URL, timeout=90)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    raw = pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])), sep=";")
    frames = []
    for code, holder in IT_F3.items():
        col = next(c for c in raw.columns if f".F3.{code}." in c and c.endswith(".EUR.EDP"))
        part = raw[["DATA_OSS", col]].rename(columns={"DATA_OSS": "period", col: "amount_mn"})
        part["holder"] = holder
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["amount_mn"] = pd.to_numeric(out["amount_mn"], errors="coerce")
    out["period"] = pd.to_datetime(out["period"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    complete = out.groupby("period")["amount_mn"].apply(lambda s: s.notna().all())
    out = out[out["period"].isin(complete[complete].index)]
    out["country"] = "IT"
    out["freq"] = "M"
    out["stock_mn"] = out.groupby("period")["amount_mn"].transform("sum")
    return out[["country", "period", "freq", "holder", "amount_mn", "stock_mn"]]


def _spain() -> pd.DataFrame:
    """Monthly LET+BYO holdings by sector. Banco de España BE.22.44 (from Jan-2016).

    Column 4 ('otras entidades financieras') is the parent of funds / insurance / PF /
    other OFI (5–8) — use 2+3+4+9+10+11+12 so parts sum to total.
    """
    response = requests.get(ES_URL, timeout=90)
    response.raise_for_status()
    text = response.content.decode("latin-1")
    rows = []
    for ln in text.splitlines():
        m = re.match(r'^"?([A-Z]{3}) (\d{4})"?,(.*)$', ln)
        if not m or m.group(1) not in _ES_MONTHS:
            continue
        vals = [pd.to_numeric(x.replace('"', "").replace(",", ""), errors="coerce")
                for x in m.group(3).split(",")]
        if len(vals) < 13:
            continue
        rows.append((pd.Timestamp(year=int(m.group(2)), month=_ES_MONTHS[m.group(1)], day=1), vals))
    if not rows:
        return pd.DataFrame(columns=["country", "period", "freq", "holder", "amount_mn", "stock_mn"])
    # face euros → €mn; holders: CB, banks ex-CB, other fin (ex-MFI), NFC, HH, GG, RoW
    map_cols = {
        2: "central_bank", 3: "banks", 4: "other_financial",
        9: "nonfinancial", 10: "households", 11: "government", 12: "rest_of_world",
    }
    frames = []
    for ts, vals in rows:
        stock = float(vals[0]) / 1e6  # € → €mn
        for i, holder in map_cols.items():
            frames.append({
                "country": "ES", "period": ts.strftime("%Y-%m-%d"), "freq": "M",
                "holder": holder, "amount_mn": float(vals[i]) / 1e6, "stock_mn": stock,
            })
    return pd.DataFrame(frames)


def _france_det() -> pd.DataFrame:
    """Quarterly OAT holders. BdF DET via comptespublics (no public monthly nonres share)."""
    response = requests.get(FR_DET_URL, timeout=90)
    response.raise_for_status()
    raw = pd.read_csv(io.BytesIO(response.content))

    def _q_period(p: str) -> pd.Timestamp:
        # comptespublics uses "2026-T2"; older extracts used ISO dates
        m = re.match(r"^(\d{4})-T([1-4])$", str(p).strip())
        if m:
            return pd.Timestamp(year=int(m.group(1)), month=3 * int(m.group(2)) - 2, day=1)
        return pd.Timestamp(p)

    raw["periode"] = raw["periode"].map(_q_period)
    wide = raw.pivot_table(index="periode", columns="serie", values="valeur", aggfunc="last")
    # OAT stock €bn → €mn; sector % of OAT → amounts
    pct_map = {
        "banque_centrale_oat_pct": "central_bank",
        "banques_oat_pct": "banks",
        "assurances_retraite_oat_pct": "insurance_pension",
        "opc_monetaires_oat_pct": "mmf",
        "autres_financiers_oat_pct": "other_financial",
        "administrations_oat_pct": "government",
        "menages_oat_pct": "households",
        "societes_non_financieres_oat_pct": "nonfinancial",
        "secteur_non_determine_oat_pct": "other_domestic",
        "nonres_oat_pct": "rest_of_world",
    }
    frames = []
    for ts, row in wide.iterrows():
        stock_bn = row.get("encours_oat_md")
        if pd.isna(stock_bn) or stock_bn <= 0:
            continue
        stock = float(stock_bn) * 1000.0
        for serie, holder in pct_map.items():
            pct = row.get(serie)
            if pd.isna(pct):
                continue
            frames.append({
                "country": "FR", "period": ts.strftime("%Y-%m-%d"), "freq": "Q",
                "holder": holder, "amount_mn": stock * float(pct) / 100.0, "stock_mn": stock,
            })
    return pd.DataFrame(frames)


def _uk() -> pd.DataFrame:
    frames = []
    for cdid, (holder, topic, dataset) in ONS.items():
        part = _ons_one(cdid, topic, dataset)
        if part.empty:
            continue
        part["country"] = "UK"
        part["freq"] = "Q"
        part["holder"] = holder
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["country", "period", "freq", "holder", "amount_mn"])
    return pd.concat(frames, ignore_index=True)[["country", "period", "freq", "holder", "amount_mn"]]


def _asof(row: pd.Series) -> str:
    period = str(row["period"])
    if row["freq"] == "A":
        return f"{period}-12-31"
    year, quarter = period.split(" Q")
    day = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[quarter]
    return f"{year}-{day}"


def _attach_stock(frame: pd.DataFrame, total_name: str) -> pd.DataFrame:
    """Sectors plus a residual equal the published stock. The residual is labeled."""
    totals = frame[frame["holder"].eq(total_name)][["country", "period", "freq", "amount_mn"]].rename(
        columns={"amount_mn": "stock_mn"}
    )
    parts = frame[~frame["holder"].isin([total_name, "gilt_stock_face"])].copy()
    domestic_parts = parts[~parts["holder"].eq("rest_of_world")]
    summed = domestic_parts.groupby(["country", "period"], as_index=False)["amount_mn"].sum()
    gap = totals.merge(summed, on=["country", "period"], how="left")
    gap["resid"] = gap["stock_mn"] - gap["amount_mn"].fillna(0)
    rows = gap.loc[gap["resid"].abs() > 1, ["country", "period", "freq", "resid", "stock_mn"]].rename(
        columns={"resid": "amount_mn"}
    )
    rows["holder"] = "other_domestic"
    rows["amount_mn"] = rows["amount_mn"].astype(float)
    out = pd.concat([parts, rows[parts.columns.intersection(rows.columns)]], ignore_index=True)
    if "stock_mn" not in out.columns:
        out["stock_mn"] = pd.NA
    out = out.drop(columns=["stock_mn"], errors="ignore").merge(
        totals[["country", "period", "stock_mn"]], on=["country", "period"], how="left"
    )
    return out


def fetch_holders() -> pd.DataFrame:
    euro = _eurostat()
    euro["stock_mn"] = pd.NA
    domestic = euro[euro["holder"].eq("domestic_total")][["country", "period", "amount_mn"]].rename(
        columns={"amount_mn": "domestic_mn"}
    )
    foreign = euro[euro["holder"].eq("rest_of_world")][["country", "period", "amount_mn"]].rename(
        columns={"amount_mn": "foreign_mn"}
    )
    stock = domestic.merge(foreign, on=["country", "period"], how="outer")
    stock["stock_mn"] = stock["domestic_mn"].fillna(0) + stock["foreign_mn"].fillna(0)
    euro = _residual(euro)
    euro = euro[~euro["holder"].isin(["domestic_total"])]
    euro = euro.drop(columns=["stock_mn"], errors="ignore").merge(
        stock[["country", "period", "stock_mn"]], on=["country", "period"], how="left"
    )
    # IT/FR/ES have national higher-frequency tapes; keep Eurostat for DE only.
    euro = euro[euro["country"].eq("DE")]
    italy = _italy()
    spain = _spain()
    france = _france_det()
    uk = _attach_stock(_uk(), "domestic_total")
    # ONS domestic_total excludes the rest of the world. The stock is both.
    row_foreign = uk[uk["holder"].eq("rest_of_world")][["country", "period", "amount_mn"]].rename(
        columns={"amount_mn": "foreign_mn"}
    )
    uk = uk.merge(row_foreign, on=["country", "period"], how="left")
    uk["stock_mn"] = uk["stock_mn"].fillna(0) + uk["foreign_mn"].fillna(0)
    uk = uk.drop(columns=["foreign_mn"])
    us = _us()
    out = pd.concat([euro, italy, spain, france, uk, us], ignore_index=True)
    out["source"] = out["country"].map({
        "UK": "ONS financial accounts, British government securities, quarterly",
        "US": "Treasury OFS-2, total public debt, quarterly",
        "IT": "Banca d'Italia TCCE0200, government debt securities, monthly",
        "ES": "Banco de España BE.22.44 LET+BYO tenencias, monthly from 2016",
        "DE": "Eurostat gov_10dd_ggd, general government debt, annual",
        "FR": "Banque de France DET (comptespublics), OAT holders, quarterly",
    })
    out["asof"] = [
        p if len(str(p)) == 10 and str(p)[4] == "-" else _asof(pd.Series({"period": p, "freq": f}))
        for p, f in zip(out["period"], out["freq"])
    ]
    out["share"] = out["amount_mn"] / out["stock_mn"].replace(0, pd.NA)
    check = out.groupby(["country", "period"])["amount_mn"].sum()
    stock_check = out.groupby(["country", "period"])["stock_mn"].first()
    gap = (check - stock_check).abs()
    bad = gap[gap > 500].index
    if len(bad):
        out = out.set_index(["country", "period"]).drop(index=bad, errors="ignore").reset_index()
    # Tenth-of-a-billion rounding in OFS-2. Fold it into the residual holder.
    check = out.groupby(["country", "period"])["amount_mn"].sum()
    stock_check = out.groupby(["country", "period"])["stock_mn"].first()
    dust = (stock_check - check).rename("dust")
    dust = dust[dust.abs().between(0.05, 500)]
    if len(dust):
        out = out.merge(dust.reset_index(), on=["country", "period"], how="left")
        plug = out["holder"].isin(["other_investors", "other_domestic"]) & out["dust"].notna()
        out.loc[plug, "amount_mn"] = out.loc[plug, "amount_mn"] + out.loc[plug, "dust"]
        out = out.drop(columns=["dust"])
        out["share"] = out["amount_mn"] / out["stock_mn"].replace(0, pd.NA)
    check = out.groupby(["country", "period"])["amount_mn"].sum()
    stock_check = out.groupby(["country", "period"])["stock_mn"].first()
    gap = (check - stock_check).abs()
    print(f"  holders: {out.groupby('country').period.nunique().to_dict()}")
    print(f"  stock gap max mn: {gap.groupby(level=0).max().round(1).to_dict()}")
    out.to_parquet(CACHE / "holders.parquet", index=False)
    return out
