"""Riksgälden raw data → ``sovereign_cash/data/se/``.

Sources (all public, no key):
  QlikView ``Webb_Upplaning.qvw``      CH1089  budget balance / net borrowing requirement, monthly from 1994,
                                               split primary / net lending to agencies / interest (mn SEK)
                                       CH1044  central-government debt change by instrument (net borrowing at
                                               nominal amount) and debt dispositions (FX, inflation
                                               compensation, money-market assets), monthly from 2002 (SEK)
  QlikView ``webb auktionsresultat.qvw`` CH926 every auction since 1996 (SGB, SGBi, T-bills, switches)
  QlikView ``webb_upplaning_daglig.qvw`` CH1073 outstanding securities (maturity + amount), today only
  MFN press-release feed (mfn.se/all/a/riksgalden.json)  monthly outcome releases with Riksgälden's own
                                               monthly forecast or the deviation from it, 2001→
"""

from __future__ import annotations

import json
import re
from itertools import zip_longest
from pathlib import Path

import pandas as pd
import requests

from sovereign_cash.se.qlik import Qlik

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "se"
MFN_URL = "https://mfn.se/all/a/riksgalden.json"
SV_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
             "okt": 10, "nov": 11, "dec": 12}

NBR_ROWS = {  # CH1089 row label → column (mn SEK; components carry the net-borrowing-requirement sign)
    "Budgetsaldo": "budget_balance",
    "Nettolånebehov": "nbr",
    "Primärt saldo": "nbr_primary",
    "Nettoutlåning till myndigheter m.fl.": "nbr_net_lending",
    "Räntor på statsskulden": "nbr_interest",
}
DEBT_ROWS = {  # CH1044 row index → column (SEK); labels asserted on parse
    1: ("STATSSKULD", "debt"),
    5: ("Statsskuldens förändring", "d_debt"),
    7: ("Nettolånebehov", "nbr_debt_table"),
    8: ("Affärsdagsjustering m.m.", "bday_adj"),
    9: ("varav: Skulddispositioner", "dispositions"),
    10: ("Valutakurseffekter, förändring", "fx_valuation"),
    11: ("Upplupen inflationskompensation, förändring", "inflation_comp"),
    12: ("Förvaltningstillgångar, förändring", "mm_assets"),
    13: ("Särskilda skulddispositioner", "special_disp"),
    18: ("varav: Nettoupplåning, nominellt belopp", "net_borrowing"),
    20: ("Statsobligationer", "nb_nominal_bonds"),
    21: ("Reala statsobligationer", "nb_linkers"),
    22: ("Gröna obligationer", "nb_green"),
    23: ("Publika obligationer, utländsk valuta", "nb_fx_bonds"),
    24: ("Övriga kapitalmarknadsskulder", "nb_other_capital"),
    26: ("Statsskuldväxlar", "nb_bills"),
    27: ("Likviditetesförvaltning", "nb_liquidity_mgmt"),
    28: ("Säkerheter", "nb_collateral"),
    29: ("Privatmarknad", "nb_retail"),
    32: ("Derivatmarknad", "nb_derivatives"),
}


def num(s: str) -> float:
    s = (s or "").replace("\xa0", "").replace(" ", "").replace(",", ".").strip()
    if s in ("", "-", "–"):
        return float("nan")
    return float(s)


def month_label(s: str) -> pd.Timestamp:
    m, y = s.strip().split()
    return pd.Timestamp(int(y), SV_MONTHS[m[:3].lower()], 1)


def _columns_to_frame(cols: list[list[str]], rows: dict[int, tuple[str, str]] | None = None,
                      labels: dict[str, str] | None = None) -> pd.DataFrame:
    lab = [x.strip() for x in cols[0]]
    recs = []
    for c in cols[1:]:
        try:
            m = month_label(c[0])
        except (ValueError, KeyError):
            continue
        r = {"month": m}
        if rows:
            for i, (want, name) in rows.items():
                if lab[i] != want:
                    raise ValueError(f"row {i}: expected {want!r}, got {lab[i]!r}")
                r[name] = num(c[i])
        if labels:
            for i, l in enumerate(lab):
                if l in labels and labels[l] not in r:
                    r[labels[l]] = num(c[i])
        recs.append(r)
    return pd.DataFrame(recs).sort_values("month").reset_index(drop=True)


def fetch_nbr() -> pd.DataFrame:
    cols = Qlik().table("Document\\BM11", "Document\\CH1089")
    (RAW / "qlik_ch1089_nbr.json").write_text(json.dumps(cols, ensure_ascii=False))
    df = _columns_to_frame(cols, labels=NBR_ROWS)
    df.to_csv(RAW / "nbr_monthly.csv", index=False)
    return df


def fetch_debt() -> pd.DataFrame:
    cols = Qlik().table("Document\\BM11", "Document\\CH1044")
    (RAW / "qlik_ch1044_debt_change.json").write_text(json.dumps(cols, ensure_ascii=False))
    df = _columns_to_frame(cols, rows=DEBT_ROWS)
    df.to_csv(RAW / "debt_change_monthly.csv", index=False)
    return df


def fetch_auctions() -> pd.DataFrame:
    head, rows = Qlik(view="webb auktionsresultat.qvw").rows("Document\\BM03", "Document\\CH926")
    df = pd.DataFrame(rows, columns=head)
    df = df[df["Date"].str.match(r"\d{4}-\d{2}-\d{2}")].drop_duplicates()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"]),
        "type": df["Type"],
        "auction_type": df["Auktionstyp"],
        "series": df["Serie / Maturity"],
        "offered_mn": df["Offered, max"].map(num),
        "tendered_mn": df["Tendered"].map(num),
        "allocated_mn": df["Allocated volume"].map(num),
        "avg_yield": df["Average yield"].map(num),
    }).sort_values("date").reset_index(drop=True)
    out.to_csv(RAW / "auctions.csv", index=False)
    return out


def fetch_outstanding() -> pd.DataFrame:
    head, rows = Qlik(view="webb_upplaning_daglig.qvw").rows("Document\\BM02", "Document\\CH1073")
    df = pd.DataFrame(rows, columns=head)
    df = df[df.iloc[:, 4].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
    problem = outstanding_problem(df)
    if problem:
        # Keep the last good list: a bad download (several days summed, rows repeated) would otherwise
        # flow straight into the debt build's redemption schedule.
        print(f"  se: outstanding securities rejected ({problem}); keeping the previous file")
        return pd.read_csv(RAW / "outstanding_securities.csv") if (RAW / "outstanding_securities.csv").exists() else df
    df.to_csv(RAW / "outstanding_securities.csv", index=False)
    return df


def outstanding_problem(df: pd.DataFrame) -> str:
    """Why a CH1073 outstanding-securities list is not usable, or "" if it is: one row per ISIN, and the SEK
    total within 20% of Riksgälden's published central-government debt (CH1044, latest month)."""
    if "ISIN" in df and df["ISIN"].duplicated().any():
        return f"{int(df['ISIN'].duplicated().sum())} repeated ISINs"
    col = "Current exchange rate incl. inflation compensation"
    debt_csv = RAW / "debt_change_monthly.csv"
    if col in df and debt_csv.exists():
        amt = pd.to_numeric(df[col].astype(str).str.replace("[\\s\u00a0\u202f]", "", regex=True), errors="coerce")
        debt = float(pd.read_csv(debt_csv)["debt"].dropna().iloc[-1])
        ratio = float(amt.sum()) / debt if debt else float("nan")
        if not 0.8 <= ratio <= 1.2:
            return f"total is {ratio:.1f}x the published debt"
    return ""


def fetch_mfn(max_items: int = 6000) -> list[dict]:
    items, off = [], 0
    while off < max_items:
        r = requests.get(f"{MFN_URL}?limit=100&offset={off}", timeout=60)
        r.raise_for_status()
        js = r.json()
        it = js.get("items", [])
        items += it
        if not it or not js.get("next_url"):
            break
        off += 100
    keep = [{"date": i["content"]["publish_date"], "lang": i["properties"].get("lang"),
             "title": i["content"]["title"], "preamble": i["content"].get("preamble") or "",
             "text": i["content"].get("text") or "", "url": i.get("url")} for i in items]
    (RAW / "mfn").mkdir(parents=True, exist_ok=True)
    (RAW / "mfn" / "riksgalden_releases.json").write_text(json.dumps(keep, ensure_ascii=False))
    return keep


def fetch_all() -> dict[str, int]:
    RAW.mkdir(parents=True, exist_ok=True)
    n = {}
    n["nbr_months"] = len(fetch_nbr())
    n["debt_months"] = len(fetch_debt())
    n["auctions"] = len(fetch_auctions())
    n["outstanding"] = len(fetch_outstanding())
    n["mfn_releases"] = len(fetch_mfn())
    (RAW / "fetch_log.json").write_text(json.dumps({"fetched": pd.Timestamp.now(tz="UTC").isoformat(), **n}))
    return n


# --------------------------------------------------------------------------------------
# Press releases → Riksgälden's monthly forecast

_SV_FULL = {"januari": 1, "februari": 2, "mars": 3, "april": 4, "maj": 5, "juni": 6, "juli": 7, "augusti": 8,
            "september": 9, "oktober": 10, "november": 11, "december": 12}
_NUM = r"(-?\d[\d\s\xa0]*(?:,\d+)?)"


def _bn(s: str) -> float:
    return float(re.sub(r"[\s\xa0]", "", s).replace(",", "."))


def _grid_rows(text: str) -> list[list[str]]:
    """Rows of a +---+ grid table, with wrapped cell fragments joined by a space."""
    rows, cur = [], None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("+"):
            if cur is not None:
                rows.append([c.strip() for c in cur])
            cur = None
        elif s.startswith("|"):
            cells = s.strip("|").split("|")
            cur = cells if cur is None else [f"{a} {b}" for a, b in zip_longest(cur, cells, fillvalue="")]
    if cur is not None:
        rows.append([c.strip() for c in cur])
    return rows


def _cell_bn(s: str) -> float | None:
    s = re.sub(r"[\s\xa0]", "", s).replace(",", ".").replace("−", "-")
    return float(s) / 1000.0 if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def parse_table(text: str) -> dict | None:
    """Budgetsaldo row of the release table (mn SEK → bn): outcome, forecast and the cumulative
    deviation since the latest forecast report ("Ack. diff.", footnote: sum of monthly forecast
    differences since the latest forecast)."""
    rows = _grid_rows(text)
    head = next((r for r in rows if any(c.startswith("Prognos") for c in r) and any("Utfall" in c for c in r)), None)
    bal = next((r for r in rows if r and r[0].startswith("Budgetsaldo")), None)
    if head is None or bal is None:
        return None
    col = {}
    for i, c in enumerate(head):
        key = "outcome" if c.startswith("Utfall") and "12" not in c else (
            "forecast" if c.startswith("Prognos") else ("cum_dev" if c.startswith("Ack") else None))
        if key and key not in col:
            col[key] = i
    if "outcome" not in col or "forecast" not in col:
        return None
    get = {k: _cell_bn(bal[i]) if i < len(bal) else None for k, i in col.items()}
    if get["outcome"] is None or get["forecast"] is None:
        return None
    return get


def parse_release(date: str, text: str) -> dict | None:
    """Outcome and Riksgälden forecast (bn SEK, surplus +) from one Swedish monthly release.

    The release table is exact (mn SEK) and wins; the body text (rounded, and phrased many ways) is the
    fallback for releases without a table (most of 2001–2017)."""
    tab = parse_table(text)
    t = re.sub(r"\s+", " ", text)
    m = re.search(r"(överskott|underskott) (?:på|om) " + _NUM + r" miljard(?:er)? kronor i (" +
                  "|".join(_SV_FULL) + r")\b", t, re.I)
    if not m:
        return None
    sgn = 1.0 if m.group(1).lower() == "överskott" else -1.0
    out = sgn * _bn(m.group(2))
    mon = _SV_FULL[m.group(3).lower()]
    pub = pd.Timestamp(date[:10])
    year = pub.year - 1 if mon > pub.month else pub.year
    if tab is not None and abs(tab["outcome"] - out) <= 1.0:
        return {"month": pd.Timestamp(year, mon, 1), "published": pub, "outcome_bn": tab["outcome"],
                "rg_forecast_bn": tab["forecast"], "rg_cum_dev_bn": tab.get("cum_dev"), "parsed_from": "table"}
    rest = t[m.end():m.end() + 400]
    fc = None
    lvl = re.search(r"prognos(?:en)?(?: för \w+)?(?: från [\wåäö ]{1,30}?)?,? (?:som )?(?:var|med|på|om) (?:ett )?"
                    r"(?:marginellt |litet |mindre |större )?(överskott|underskott)? ?(?:på |om )?" + _NUM + r" miljard",
                    rest, re.I)
    dev = re.search(_NUM + r" miljard(?:er)? kronor (högre|lägre|större|mindre|mer|bättre|sämre) än "
                    r"(?:Riksgäldens |vår |i vår )?prognos", rest, re.I)
    same = re.search(r"i (?:linje|nivå) med (?:Riksgäldens |vår )?prognos", rest, re.I)
    if lvl:
        fsgn = sgn if not lvl.group(1) else (1.0 if lvl.group(1).lower() == "överskott" else -1.0)
        fc = fsgn * _bn(lvl.group(2))
    elif dev:
        d = _bn(dev.group(1))
        word = dev.group(2).lower()
        if word in ("bättre", "sämre"):
            fc = out - d if word == "bättre" else out + d
        else:  # compares the stated amount (surplus or deficit) with its forecast
            amt = abs(out) - d if word in ("högre", "större", "mer") else abs(out) + d
            fc = sgn * amt
    elif same:
        fc = out
    return {"month": pd.Timestamp(year, mon, 1), "published": pub, "outcome_bn": out, "rg_forecast_bn": fc,
            "rg_cum_dev_bn": None, "parsed_from": "text"}


_EN_FULL = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12}
_MONTH_WORD = {**_SV_FULL, **_EN_FULL}
_D = r"(\d{1,2}) (\w+),? (\d{4})"


def _date(day: str, month: str, year: str) -> pd.Timestamp | None:
    m = _MONTH_WORD.get(month.lower())
    return pd.Timestamp(int(year), m, int(day)) if m else None


def next_dates(releases: list[dict] | None = None) -> dict:
    """Next monthly outcome and next forecast report as announced in the latest outcome release
    ("Utfallet för september 2026 publiceras den 7 oktober 2026", "... ny prognos ... är den 26 november 2026")."""
    if releases is None:
        p = RAW / "mfn" / "riksgalden_releases.json"
        releases = json.loads(p.read_text()) if p.exists() else []
    out: dict = {}
    for r in sorted(releases, key=lambda x: x["date"], reverse=True):
        t = re.sub(r"\s+", " ", r.get("text") or "")
        if "outcome" not in out:
            m = (re.search(r"Utfallet för (\w+) (\d{4}) publiceras den " + _D, t)
                 or re.search(r"outcome for (\w+),? (\d{4}) will be published on " + _D, t, re.I))
            if m and _MONTH_WORD.get(m.group(1).lower()):
                d = _date(*m.group(3, 4, 5))
                if d is not None:
                    out["outcome"] = {"month": pd.Timestamp(int(m.group(2)), _MONTH_WORD[m.group(1).lower()], 1),
                                      "date": d, "release": r["date"][:10]}
        if "report" not in out:
            m = (re.search(r"ny prognos över [^.]*? är den " + _D, t)
                 or re.search(r"publishing a new forecast [^.]*? is " + _D, t, re.I))
            if m:
                d = _date(*m.group(1, 2, 3))
                if d is not None:
                    out["report"] = {"date": d, "release": r["date"][:10]}
        if len(out) == 2:
            break
    return out


def riksgalden_forecasts(releases: list[dict] | None = None) -> pd.DataFrame:
    if releases is None:
        releases = json.loads((RAW / "mfn" / "riksgalden_releases.json").read_text())
    recs = []
    for r in releases:
        if r["lang"] != "sv":
            continue
        p = parse_release(r["date"], r["preamble"] + " " + r["text"])
        if p:
            recs.append(p)
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["month", "published", "outcome_bn", "rg_forecast_bn", "rg_cum_dev_bn",
                                     "parsed_from"])
    return df.sort_values("published").drop_duplicates("month", keep="last").sort_values("month").reset_index(drop=True)


if __name__ == "__main__":
    print(fetch_all())
