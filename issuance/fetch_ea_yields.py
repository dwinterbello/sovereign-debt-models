"""Daily French and Italian government bond yields, % p.a.

    PYTHONPATH=. .venv/bin/python -m issuance.fetch_ea_yields

FR  Banque de France Webstat, Taux de l'Échéance Constante (TEC n, daily constant-maturity OAT
    yield interpolated from benchmark OATs, computed by Euronext), series
    FM.D.FR.EUR.FR2.BB.FRMOYTEC{n}.HSTA, from 2004-11-03 (TEC30 from 2006-10-02), published the
    next morning. The observations API needs `Authorization: Apikey <key>`: a free personal key
    from an account at webstat.banque-france.fr (Mon compte → API). Set `BDF_WEBSTAT_API_KEY`, or
    put `BDF_WEBSTAT_API_KEY=<key>` in issuance/webstat.env (gitignored). Without a key the fetch
    is skipped and a raw download from an earlier keyed run is parsed if one is cached.
    Days after the last TEC observation (all days, when there is no TEC at all) come from CNBC
    FR{n}Y-FR closes, labelled in `source`.
IT  Banca d'Italia BDS cube BMK0200, gross yield to maturity of the MTS benchmark BTP (the most
    traded of the latest issues; 3y from 1989, 5y 1988, 10y 1991, 30y 1993). The cube is
    refreshed with the monthly "Mercato finanziario" bulletin, so it ends about a month back.
    There is no 2y BTP benchmark: the 2y CTZ benchmark (MFN_BMK.D.001A…9) ends Aug-2021 when CTZ
    issuance was suspended, so Italy's short leg is the 3y.
    Days after the last BdI observation come from CNBC (ts-api.cnbc.com, IT{n}Y-IT, daily
    closes about two years back): a market-data vendor, not an official series, kept apart by
    the `source` column and never used inside the BdI window.

Raw downloads are cached under cache/ea_yields/ and parsed offline; a failed download parses the
cached file instead.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

from issuance.paths import CACHE

RAW = CACHE / "ea_yields"
RAW.mkdir(parents=True, exist_ok=True)

BDF_URL = ("https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/json/"
           "?where=series_key%20IN%20(%22{key}%22)&order_by=time_period_start"
           "&select=series_key,time_period_start,time_period_end,obs_value")
BDF_KEY = "FM.D.FR.EUR.FR2.BB.FRMOYTEC{t}.HSTA"
BDF_KEY_ENV = "BDF_WEBSTAT_API_KEY"
BDF_KEY_FILE = Path(__file__).resolve().parent / "webstat.env"
FR_TENORS = (2, 5, 10, 20, 30)

BDI_URL = "https://a2a.bancaditalia.it/infostat/dataservices/export/EN/CSV/DATA/CUBE/BANKITALIA/DIFF/BMK0200"
BDI_SERIES = {"MFN_BMK.D.020.922.0.EUR.203": "y3", "MFN_BMK.D.020.922.0.EUR.205": "y5",
              "MFN_BMK.D.020.922.0.EUR.210": "y10", "MFN_BMK.D.020.922.0.EUR.230": "y30",
              "MFN_BMK.D.001A.922.0.EUR.9": "ctz2"}
IT_TENORS = (3, 5, 10, 30)

CNBC_URL = "https://ts-api.cnbc.com/harmony/app/charts/1Y.json?symbol={sym}"
# CNBC bars are stamped D 04:00 UTC and close at the end of day D; it also emits a near-flat
# Saturday bar, dropped. Shift 0 maximises the daily-change match against BdI
# (`cnbc_lag_check`: corr 0.65-0.70 at 0 vs 0.08-0.18 at -1).
CNBC_SHIFT_DAYS = 0

SRC_BDF = "BdF Webstat TEC{t} (FM.D.FR.EUR.FR2.BB.FRMOYTEC{t}.HSTA)"
SRC_BDI = "BdI BDS BMK0200 MTS benchmark BTP {t}y"
SRC_CNBC = "CNBC IT{t}Y-IT close (tail after BdI)"
SRC_CNBC_FR = "CNBC FR{t}Y-FR close (vendor; days without a BdF TEC)"


class NoApiKey(RuntimeError):
    pass


def _get(url: str, path, headers: dict | None = None, timeout: int = 90, offline: bool = False) -> bytes:
    if offline:
        return path.read_bytes()
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0 (issuance research)"}, timeout=timeout)
        r.raise_for_status()
        head = r.content[:200].lstrip().lower()
        if head.startswith((b"<!doctype html", b"<html")) and not str(path).endswith((".html", ".htm")):
            # a maintenance / error page served with 200 (BdI, 14 Nov 2026): keep the last good cache
            print(f"  ea_yields: {url} returned an HTML page, keeping {path.name}")
            if path.exists():
                return path.read_bytes()
            raise requests.RequestException("HTML page instead of data")
        path.write_bytes(r.content)
        return r.content
    except requests.RequestException:
        if not path.exists():
            raise
        return path.read_bytes()


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _clean(s: pd.Series, today: pd.Timestamp | None = None) -> pd.Series:
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s[s.index <= (_today() if today is None else today)]


# ------------------------------------------------------------------ parsers (offline)
def parse_bdf_json(raw: str | bytes, key: str | None = None) -> pd.Series:
    """Webstat Opendatasoft JSON export (a list of records; the `/records` shape
    {"results": [...]} also accepted) with series_key, a date in time_period_start /
    time_period_end / time_period (ISO date or datetime) and obs_value (number or string) → % by
    date, rows of `key` only when given."""
    j = json.loads(raw)
    recs = j.get("results", []) if isinstance(j, dict) else j
    if not recs:
        return pd.Series(dtype=float)
    d = pd.DataFrame(recs)
    if key is not None and "series_key" in d:
        d = d[d["series_key"] == key]
    col = next(c for c in ("time_period_start", "time_period_end", "time_period") if c in d)
    idx = pd.to_datetime(d[col].astype(str).str[:10], format="%Y-%m-%d")
    v = pd.to_numeric(d["obs_value"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return _clean(pd.Series(v.to_numpy(), index=idx))


def api_key() -> str | None:
    """`BDF_WEBSTAT_API_KEY` from the environment, else from issuance/webstat.env."""
    k = os.environ.get(BDF_KEY_ENV, "").strip()
    if k:
        return k
    if BDF_KEY_FILE.exists():
        for line in BDF_KEY_FILE.read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip() == BDF_KEY_ENV and val.strip():
                return val.strip().strip("'\"")
    return None


def parse_bdi_csv(text: str) -> pd.DataFrame:
    """BMK0200 multicube CSV (`;`, quoted, DATA_OSS yyyy/mm/dd, decimal point or comma; weekends
    blank) → wide frame of % by date, columns per `BDI_SERIES`."""
    d = pd.read_csv(io.StringIO(text.lstrip("\ufeff")), sep=";", dtype=str)
    idx = pd.to_datetime(d["DATA_OSS"], format="%Y/%m/%d")
    out = pd.DataFrame(index=idx)
    for key, name in BDI_SERIES.items():
        if key in d:
            out[name] = pd.to_numeric(d[key].str.replace(",", ".", regex=False), errors="coerce").to_numpy()
    out = out.dropna(how="all").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[out.index <= _today()]


def _bdi_text(raw: bytes) -> str:
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            return z.read(name).decode("utf-8", "ignore")
    return raw.decode("utf-8", "ignore")


def parse_cnbc(raw: str | bytes, shift_days: int = CNBC_SHIFT_DAYS) -> pd.Series:
    """CNBC harmony chart JSON → daily close % by (shifted) trade date."""
    j = json.loads(raw)
    bars = (j.get("barData") or {}).get("priceBars") or []
    if not bars:
        return pd.Series(dtype=float)
    d = pd.DataFrame(bars)
    idx = pd.to_datetime(d["tradeTime"].astype(str).str[:8], format="%Y%m%d") + pd.Timedelta(days=shift_days)
    s = pd.Series(pd.to_numeric(d["close"], errors="coerce").to_numpy(), index=idx)
    return _clean(s[s.index.dayofweek < 5])


# ------------------------------------------------------------------ fetchers
def bdf_tec(t: int, offline: bool = False) -> pd.Series:
    """TEC t, % by date. No key: the raw download of an earlier keyed run, else `NoApiKey`."""
    key = BDF_KEY.format(t=t)
    path = RAW / f"bdf_tec{t}.json"
    k = api_key()
    if k is None and not offline:
        if not path.exists():
            raise NoApiKey(f"no {BDF_KEY_ENV} (env or issuance/webstat.env) and no cached TEC{t}")
        offline = True
    hdr = {"Authorization": f"Apikey {k}", "User-Agent": "Mozilla/5.0 (issuance research)"}
    raw = _get(BDF_URL.format(key=key), path, hdr, offline=offline)
    return parse_bdf_json(raw, key)


def bdi_bmk(offline: bool = False) -> pd.DataFrame:
    return parse_bdi_csv(_bdi_text(_get(BDI_URL, RAW / "bdi_bmk0200.zip", timeout=180, offline=offline)))


def cnbc(sym: str, offline: bool = False) -> pd.Series:
    return parse_cnbc(_get(CNBC_URL.format(sym=sym), RAW / f"cnbc_{sym}.json", offline=offline))


def splice_tail(official: pd.Series, tail: pd.Series) -> pd.DataFrame:
    """Official series, then tail values dated strictly after its last observation. Frame
    date, value, origin ('official' | 'tail')."""
    official = official.dropna()
    last = official.index.max() if len(official) else pd.Timestamp.min
    t = tail[pd.DatetimeIndex(tail.index) > last].dropna() if len(tail) else tail
    return pd.concat([pd.DataFrame({"date": official.index, "value": official.to_numpy(), "origin": "official"}),
                      pd.DataFrame({"date": t.index, "value": t.to_numpy(), "origin": "tail"})], ignore_index=True)


def daily(offline: bool = False) -> pd.DataFrame:
    """Long frame country, series (y2/y3/y5/y10/y20/y30), date, value (%), source. A tenor whose
    sources all fail is absent. `offline` parses the raw cache only."""
    rows = []
    for t in FR_TENORS:
        try:
            off = bdf_tec(t, offline)
        except Exception as exc:  # noqa: BLE001
            print(f"  ea_yields: FR TEC{t} skipped: {exc}")
            off = pd.Series(dtype=float)
        try:
            tail = cnbc(f"FR{t}Y-FR", offline)
        except Exception as exc:  # noqa: BLE001
            print(f"  ea_yields: CNBC FR{t}Y failed: {exc}")
            tail = pd.Series(dtype=float)
        sp = splice_tail(off, tail)
        if sp.empty:
            continue
        sp["source"] = [SRC_BDF.format(t=t) if o == "official" else SRC_CNBC_FR.format(t=t) for o in sp["origin"]]
        rows.append(sp.drop(columns="origin").assign(country="FR", series=f"y{t}"))
    try:
        bdi = bdi_bmk(offline)
    except Exception as exc:  # noqa: BLE001
        print(f"  ea_yields: BdI BMK0200 failed: {exc}")
        bdi = pd.DataFrame()
    for t in IT_TENORS:
        off = bdi[f"y{t}"].dropna() if f"y{t}" in bdi else pd.Series(dtype=float)
        try:
            tail = cnbc(f"IT{t}Y-IT", offline)
        except Exception as exc:  # noqa: BLE001
            print(f"  ea_yields: CNBC IT{t}Y failed: {exc}")
            tail = pd.Series(dtype=float)
        sp = splice_tail(off, tail)
        if sp.empty:
            continue
        sp["source"] = [SRC_BDI.format(t=t) if o == "official" else SRC_CNBC.format(t=t) for o in sp["origin"]]
        rows.append(sp.drop(columns="origin").assign(country="IT", series=f"y{t}"))
    if not rows:
        return pd.DataFrame(columns=["country", "series", "date", "value", "source"])
    out = pd.concat(rows, ignore_index=True)[["country", "series", "date", "value", "source"]]
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["country", "series", "date"]).reset_index(drop=True)


def cnbc_lag_check(official: pd.Series, raw_json: str | bytes) -> pd.DataFrame:
    """Corr of daily changes and mean abs level gap (bp) between CNBC and an official daily
    series for date shifts −2..+1 days."""
    rows = []
    for sh in (-2, -1, 0, 1):
        c = parse_cnbc(raw_json, shift_days=sh)
        j = pd.concat([official.rename("o"), c.rename("c")], axis=1, join="inner").dropna()
        dj = j.diff().dropna()
        rows.append({"shift": sh, "n": len(j), "corr_d": dj["o"].corr(dj["c"]),
                     "mad_bp": 100 * (j["o"] - j["c"]).abs().mean(), "max_bp": 100 * (j["o"] - j["c"]).abs().max()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = daily()
    print(d.groupby(["country", "series", "source"])["date"].agg(["min", "max", "count"]).to_string())
