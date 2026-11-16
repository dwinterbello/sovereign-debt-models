"""Cross-border flows into government bonds: who buys each sovereign from abroad.

One tidy file, ``issuance/cache/xborder_flows.parquet``:
country, period (month start; quarter start for Q), series, value, unit, freq, source.

Sources
- EA / DE / FR / IT: ECB data portal, dataflow ``BPS`` (b.o.p., 17 dims incl. TYPE_ENTITY).
  ``BP6`` is still listed but returns no series. Euro area is ``U2`` (changing composition;
  ``I9`` stops at 2025-12 when Bulgaria joined, ``I10`` starts there). National rows use
  counterpart ``W1`` = all non-residents, including other euro-area countries.
- IT stock: Banca d'Italia TCCE0200 rest-of-world (S2) holdings, face value (EDP), read from
  ``holders.parquet`` (written by fetch_holders); the month-on-month change is a flow proxy.
- US: TIC SLT Table 3 (country holdings + net LT Treasury sales to foreigners, from 2023-02),
  legacy Form S ``s1_globl.txt`` for net LT Treasuries before 2023-02, and the old
  Major Foreign Holders history ``mfhhis01.txt`` for holdings before 2020.
- UK: BoE IADB ``LPMB9P6`` (monthly non-resident net acquisitions of central government gilts),
  plus the quarterly ONS overseas gilt transactions (NEXL) already in
  ``projects/uk_long_end/workbook/66_sector_flows_quarterly.csv``.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from issuance.http import get_bytes, session
from issuance.paths import CACHE, RAW, REPO

OUT = CACHE / "xborder_flows.parquet"
RAW_DIR = RAW / "xborder"
COLUMNS = ["country", "period", "series", "value", "unit", "freq", "source"]

ECB_API = "https://data-api.ecb.europa.eu/service/data/BPS"
# (series id, REF_SECTOR, COUNTERPART_SECTOR, ACCOUNTING_ENTRY, INSTR_ASSET, MATURITY)
ECB_SERIES = {
    "ecb_bop_liab_govt_lt_debt": ("S13", "S1", "L", "F3", "L"),
    "ecb_bop_liab_govt_st_debt": ("S13", "S1", "L", "F3", "S"),
    "ecb_bop_liab_all_lt_debt": ("S1", "S1", "L", "F3", "L"),
    "ecb_bop_assets_all_lt_debt": ("S1", "S1", "A", "F3", "L"),
    "ecb_bop_assets_foreign_govt": ("S1", "S13", "A", "F", "_Z"),
}
ECB_AREAS = {"U2": "EA", "DE": "DE", "FR": "FR", "IT": "IT"}
ECB_EA_ONLY = {"ecb_bop_assets_foreign_govt"}
# no U2 series for this one; I10 = fixed EA21 composition, backfilled from 2013
ECB_EA_AREA = {"ecb_bop_assets_foreign_govt": "I10"}

TIC = "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents"
TIC_TABLE3 = f"{TIC}/slt_table3.txt"
TIC_FORM_S = f"{TIC}/s1_globl.txt"
TIC_MFH_HIST = f"{TIC}/mfhhis01.txt"
TIC_TOTALS = {"99996": "total", "99990": "official", "99991": "private"}
TIC_COUNTRIES = {
    "Japan": "japan", "China, Mainland": "china", "United Kingdom": "uk",
    "Belgium": "belgium", "Luxembourg": "luxembourg", "Cayman Islands": "cayman",
    "Canada": "canada", "Ireland": "ireland", "France": "france",
    "Switzerland": "switzerland", "Taiwan": "taiwan",
}
# mfhhis01 names that differ from SLT Table 3
MFH_ALIASES = {"China, Mainland": ("China, Mainland", "Mainland China", "China"),
               "Grand Total": ("Grand Total",)}

BOE_IADB = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
BOE_CODE = "LPMB9P6"
UK_WORKBOOK = REPO / "projects" / "uk_long_end" / "workbook" / "66_sector_flows_quarterly.csv"


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=COLUMNS)


def _raw(name: str, url: str, force: bool, **kw) -> bytes:
    """Download to RAW/xborder; on failure fall back to the last copy."""
    dest = RAW_DIR / name
    try:
        return get_bytes(url, dest=dest, **kw)
    except Exception:
        if dest.exists() and not force:
            print(f"  {name}: fetch failed, using cached raw")
            return dest.read_bytes()
        raise


# ---------------------------------------------------------------- ECB

def ecb_key(area: str, ref_sector: str, cp_sector: str, entry: str, instr: str, mat: str) -> str:
    return ".".join(["M", "N", area, "W1", ref_sector, cp_sector, "T", entry, "FA", "P",
                     instr, mat, "EUR", "_T", "M", "N", "ALL"])


def _ecb(force: bool) -> pd.DataFrame:
    parts = []
    for series, dims in ECB_SERIES.items():
        for area, country in ECB_AREAS.items():
            if series in ECB_EA_ONLY and area != "U2":
                continue
            if area == "U2":
                area = ECB_EA_AREA.get(series, area)
            key = ecb_key(area, *dims)
            payload = _raw(f"ecb_BPS.{key}.csv", f"{ECB_API}/{key}?format=csvdata", force,
                           timeout=120)
            raw = pd.read_csv(io.BytesIO(payload))
            mult = pd.to_numeric(raw.get("UNIT_MULT", 6), errors="coerce").fillna(6)
            value = pd.to_numeric(raw["OBS_VALUE"], errors="coerce") * 10.0 ** (mult - 9)
            part = pd.DataFrame({
                "country": country,
                "period": pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m"),
                "series": series,
                "value": value,
                "unit": "bn_eur",
                "freq": "M",
                "source": f"ECB BPS.{key}",
            }).dropna(subset=["value"])
            parts.append(part)
            print(f"  ECB {country:2s} {series}: {len(part)} obs to {part['period'].max():%Y-%m}")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- Italy (BdI stock)

def _national_row(country: str, series_prefix: str, src: str, freq: str = "M") -> pd.DataFrame:
    """Rest-of-world stock (+ MoM/QoQ change) from ``holders.parquet``."""
    path = CACHE / "holders.parquet"
    if not path.exists():
        from issuance.fetch_holders import fetch_holders
        fetch_holders()
    holders = pd.read_parquet(path)
    row = holders[(holders["country"] == country) & (holders["holder"] == "rest_of_world")].copy()
    if row.empty:
        return pd.DataFrame(columns=["country", "period", "series", "value", "unit", "freq", "source"])
    row["period"] = pd.to_datetime(row["period"])
    if freq == "M":
        row["period"] = row["period"].dt.to_period("M").dt.to_timestamp()
    else:
        row["period"] = row["period"].dt.to_period("Q").dt.to_timestamp(how="end")
    row = row.dropna(subset=["amount_mn"]).sort_values("period").drop_duplicates("period")
    stock = pd.DataFrame({
        "country": country, "period": row["period"],
        "series": f"{series_prefix}_holdings", "value": row["amount_mn"] / 1e3,
        "unit": "bn_eur", "freq": freq, "source": src,
    })
    if freq == "M":
        gap = row["period"].diff().dt.days.between(27, 32)
    else:
        gap = row["period"].diff().dt.days.between(80, 100)
    flow = stock.assign(
        series=f"{series_prefix}_net_change",
        value=stock["value"].diff(),
        source=src + "; period change",
    )[gap.values]
    out = pd.concat([stock, flow.dropna(subset=["value"])], ignore_index=True)
    print(f"  {country} rest-of-world ({freq}): {len(stock)} obs to {stock['period'].max():%Y-%m}")
    return out


def _italy_row() -> pd.DataFrame:
    return _national_row(
        "IT", "bdi_row",
        "Banca d'Italia TCCE0200 F3 S2 (rest of world), face value", "M",
    )


def _spain_row() -> pd.DataFrame:
    return _national_row(
        "ES", "bde_row",
        "Banco de España BE.22.44 LET+BYO rest of world, face value", "M",
    )


def _france_row() -> pd.DataFrame:
    return _national_row(
        "FR", "bdf_det_row",
        "Banque de France DET OAT nonres (comptespublics), quarterly", "Q",
    )


# ---------------------------------------------------------------- US TIC

def _tic_num(text: str) -> float:
    text = str(text).strip().strip('"').replace(",", "")
    return float(text) if re.fullmatch(r"-?\d+(\.\d+)?", text) else float("nan")


def parse_slt_table3(text: str) -> pd.DataFrame:
    """Rows with a YYYY-MM date in col 3; returns country, code, period, the seven numbers."""
    names = ["for_treas_pos", "for_treas_net", "for_lt_treas_pos", "for_lt_treas_net",
             "for_lt_treas_valchg", "for_st_treas_pos", "for_st_treas_net"]
    rows = []
    for line in text.splitlines():
        cells = line.split("\t")
        if len(cells) < 10 or not re.fullmatch(r"\d{4}-\d{2}", cells[2].strip()):
            continue
        rec = {"name": cells[0].strip(), "code": cells[1].strip(),
               "period": pd.Timestamp(cells[2].strip() + "-01")}
        rec.update({n: _tic_num(c) for n, c in zip(names, cells[3:10])})
        rows.append(rec)
    return pd.DataFrame(rows)


def parse_form_s_total(text: str) -> pd.DataFrame:
    """Legacy Form S grand total: gross purchases [1] minus gross sales [7] of T-bonds & notes."""
    rows = []
    for line in text.splitlines():
        cells = line.split("\t")
        if len(cells) < 15 or cells[0].strip() != "Grand Total":
            continue
        if not re.fullmatch(r"\d{4}-\d{2}", cells[2].strip()):
            continue
        rows.append({"period": pd.Timestamp(cells[2].strip() + "-01"),
                     "net_mn": _tic_num(cells[3]) - _tic_num(cells[9])})
    return pd.DataFrame(rows)


_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def parse_mfh_history(text: str) -> pd.DataFrame:
    """mfhhis01.txt: yearly blocks, month row then 'Country' year row. A repeated month
    (2000-2011 benchmark breaks: 'Jun Jun', 'Mar Mar') appears newest-first; the caller
    keeps the first occurrence, i.e. the post-benchmark value."""
    lines = text.splitlines()
    rows = []
    i = 0
    while i < len(lines) - 1:
        months = [c.strip() for c in lines[i].split("\t") if c.strip()]
        nxt = [c.strip() for c in lines[i + 1].split("\t")]
        if months and all(m in _MONTHS for m in months) and nxt and nxt[0] == "Country":
            years = [c for c in nxt[1:] if c]
            ok = len(years) >= len(months)
            j = i + 2
            while j < len(lines) and not lines[j].strip().startswith("Of which"):
                cells = [c.strip().strip('"') for c in lines[j].split("\t")]
                if ok and cells and cells[0] and len(cells) > len(months):
                    name = re.sub(r"\s+\d+/$", "", cells[0])
                    for k, m in enumerate(months):
                        val = _tic_num(cells[k + 1])
                        if val == val:
                            rows.append({"name": name,
                                         "period": pd.Timestamp(int(years[k]), _MONTHS[m], 1),
                                         "bn": val})
                if cells and cells[0] == "Grand Total":
                    break
                j += 1
            i = j
        i += 1
    return pd.DataFrame(rows)


def _tic(force: bool) -> pd.DataFrame:
    t3 = parse_slt_table3(_raw("tic_slt_table3.txt", TIC_TABLE3, force, timeout=120)
                          .decode("utf-8", errors="replace"))
    src3 = f"TIC SLT Table 3 {TIC_TABLE3}"
    out = []

    def add(frame, col, series, scale, src):
        f = frame.dropna(subset=[col])
        out.append(pd.DataFrame({"country": "US", "period": f["period"], "series": series,
                                 "value": f[col] * scale, "unit": "bn_usd", "freq": "M",
                                 "source": src}))

    for code, tag in TIC_TOTALS.items():
        sub = t3[t3["code"] == code]
        add(sub, "for_lt_treas_net", f"tic_net_lt_treasuries{'' if tag == 'total' else '_' + tag}",
            1e-3, src3 + f" code {code} for_lt_treas_net")
        add(sub, "for_treas_net", f"tic_net_treasuries_incl_bills{'' if tag == 'total' else '_' + tag}",
            1e-3, src3 + f" code {code} for_treas_net")
        add(sub, "for_treas_pos", f"tic_holdings_{tag}", 1e-3, src3 + f" code {code} for_treas_pos")
    for name, tag in TIC_COUNTRIES.items():
        sub = t3[t3["name"] == name]
        add(sub, "for_treas_pos", f"tic_holdings_{tag}", 1e-3, src3 + f" '{name}' for_treas_pos")
        add(sub, "for_lt_treas_net", f"tic_net_lt_treasuries_{tag}", 1e-3,
            src3 + f" '{name}' for_lt_treas_net")
    slt = pd.concat(out, ignore_index=True)

    # legacy Form S before SLT transactions begin
    first_net = slt.loc[slt["series"] == "tic_net_lt_treasuries", "period"].min()
    form_s = parse_form_s_total(_raw("tic_s1_globl.txt", TIC_FORM_S, force, timeout=180)
                                .decode("utf-8", errors="replace"))
    form_s = form_s[form_s["period"] < first_net]
    legacy = pd.DataFrame({"country": "US", "period": form_s["period"],
                           "series": "tic_net_lt_treasuries", "value": form_s["net_mn"] / 1e3,
                           "unit": "bn_usd", "freq": "M",
                           "source": f"TIC Form S {TIC_FORM_S} Grand Total col[1]-col[7]"})

    # MFH history before SLT Table 3 holdings begin
    mfh = parse_mfh_history(_raw("tic_mfhhis01.txt", TIC_MFH_HIST, force, timeout=120)
                            .decode("utf-8", errors="replace"))
    hist = []
    targets = {**{n: f"tic_holdings_{t}" for n, t in TIC_COUNTRIES.items()},
               "Grand Total": "tic_holdings_total"}
    for name, series in targets.items():
        aliases = MFH_ALIASES.get(name, (name,))
        first = slt.loc[slt["series"] == series, "period"].min()
        sub = mfh[mfh["name"].isin(aliases) & (mfh["period"] < first)]
        sub = sub.drop_duplicates("period", keep="first")
        hist.append(pd.DataFrame({"country": "US", "period": sub["period"], "series": series,
                                  "value": sub["bn"], "unit": "bn_usd", "freq": "M",
                                  "source": f"TIC Major Foreign Holders history {TIC_MFH_HIST}"}))
    frame = pd.concat([slt, legacy, *hist], ignore_index=True)
    for s in ("tic_net_lt_treasuries", "tic_holdings_japan", "tic_holdings_total"):
        p = frame.loc[frame["series"] == s, "period"]
        print(f"  TIC {s}: {len(p)} obs {p.min():%Y-%m}..{p.max():%Y-%m}")
    return frame


# ---------------------------------------------------------------- UK

def _boe_monthly(force: bool) -> pd.DataFrame:
    """IADB HTML table (the CSV export answers with the page shell)."""
    dest = RAW_DIR / f"boe_{BOE_CODE}.html"
    params = {"Datefrom": "01/Jan/1990", "Dateto": "now", "SeriesCodes": BOE_CODE,
              "CSVF": "TN", "UsingCodes": "Y", "VPD": "Y", "VFD": "N", "html.x": "66"}
    try:
        s = session()
        s.headers["Accept"] = "text/html,application/xhtml+xml,*/*;q=0.8"
        s.cookies.set("boeconsent", "necessary", domain="www.bankofengland.co.uk")
        resp = s.get(BOE_IADB, params=params, timeout=120)
        resp.raise_for_status()
        if "Access denied" in resp.text or BOE_CODE not in resp.text:
            raise RuntimeError("BoE IADB refused")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_text(resp.text, encoding="utf-8")
        html = resp.text
    except Exception:
        if force or not dest.exists():
            raise
        print("  BoE: fetch failed, using cached raw")
        html = dest.read_text(encoding="utf-8")
    table = next(t for t in pd.read_html(io.StringIO(html))
                 if str(t.columns[0]).strip().lower() == "date"
                 and any(BOE_CODE in str(c) for c in t.columns))
    col = next(c for c in table.columns if BOE_CODE in str(c))
    period = pd.to_datetime(table["Date"], format="%d %b %y").dt.to_period("M").dt.to_timestamp()
    out = pd.DataFrame({"country": "UK", "period": period, "series": "boe_overseas_gilt_net_purchases",
                        "value": pd.to_numeric(table[col], errors="coerce") / 1e3,
                        "unit": "bn_gbp", "freq": "M",
                        "source": f"BoE IADB {BOE_CODE} (non-resident net acquisitions of central "
                                  "government sterling gilts, NSA)"}).dropna(subset=["value"])
    print(f"  BoE {BOE_CODE}: {len(out)} obs to {out['period'].max():%Y-%m}")
    return out


def _uk_quarterly() -> pd.DataFrame:
    if not UK_WORKBOOK.exists():
        return _frame([])
    wb = pd.read_csv(UK_WORKBOOK)
    yq = wb["quarter"].str.extract(r"(\d{4}) Q(\d)").astype(int)
    period = pd.to_datetime(dict(year=yq[0], month=3 * yq[1] - 2, day=1))
    out = pd.DataFrame({"country": "UK", "period": period, "series": "ons_overseas_gilt_tx_q",
                        "value": pd.to_numeric(wb["overseas"], errors="coerce"),
                        "unit": "bn_gbp", "freq": "Q",
                        "source": "ONS UKEA NEXL via projects/uk_long_end/workbook/"
                                  "66_sector_flows_quarterly.csv"}).dropna(subset=["value"])
    print(f"  ONS NEXL (uk_long_end): {len(out)} quarters to {out['period'].max():%Y-%m}")
    return out


# ---------------------------------------------------------------- main

def fetch_xborder(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parts = [_ecb(force), _italy_row(), _spain_row(), _france_row(),
             _tic(force), _boe_monthly(force), _uk_quarterly()]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)[COLUMNS]
    out["period"] = pd.to_datetime(out["period"])
    out["value"] = out["value"].astype(float)
    out = (out.sort_values(["country", "series", "period"])
              .drop_duplicates(["country", "period", "series"], keep="first")
              .reset_index(drop=True))
    out.to_parquet(OUT, index=False)
    print(f"  wrote {OUT.relative_to(REPO)}: {len(out)} rows, {out['series'].nunique()} series")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    fetch_xborder(force=ap.parse_args().force)
