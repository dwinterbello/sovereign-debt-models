"""Free public fetchers for the G4 rates lab, cached as wide CSV under ``g4/cache/``.

Every feed is idempotent and incremental: a refresh asks the source only for the
tail it has not seen (FRED ``cosd``, SDMX ``startPeriod``, MoF current-month
file), merges it over the cache with the new print winning, and never shrinks
the cache. A failed fetch keeps the cached history and is reported; a feed with
neither network nor cache raises. Nothing here fills across days.

    PYTHONPATH=. ../.venv/bin/python -m g4.sources --refresh     # from correlations/

Sources (none needs a key)
  FRED            UST par CMT (DGS*), EFFR / SOFR / target band, US macro, WALCL, FX
  Fed GSW         feds200628.csv — daily-refit Svensson zero curve 1-30y (forwards)
  ECB data API    AAA + all-issuer euro area spot curves, DFR, EUR STR / EONIA, HICP,
                  unemployment, Eurosystem total assets
  Bundesbank      BBSIS daily Svensson term structure of listed Federal securities
  MoF Japan       jgbcme_all.csv + jgbcme.csv constant-maturity JGB yields
  BoJ             stat-search API — uncollateralised overnight call rate (TONA)
"""

from __future__ import annotations

import argparse
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("G4_CACHE_DIR", HERE / "cache"))
TIMEOUT = (10, 120)
AGENT = "system-g4-rv-lab/1.0 (public data)"
OVERLAP_DAYS = 45  # re-request this much tail so revisions overwrite the cache

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata&detail=dataonly"
BBK_URL = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/{key}?format=csv&lang=en"
MOF_ALL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
MOF_CUR = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
BOJ_URL = "https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=en&db={db}&code={code}"
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


class FetchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# cache primitives
# ---------------------------------------------------------------------------------------------

def _offline() -> bool:
    return os.environ.get("G4_OFFLINE", "") not in ("", "0")


def cache_path(source: str, name: str) -> Path:
    return CACHE / source / f"{name}.csv"


def read_wide(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def merge_wide(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New observations overwrite old on the same date; old history is never dropped."""
    if old is None or old.empty:
        out = new.copy()
    elif new is None or new.empty:
        out = old.copy()
    else:
        out = new.combine_first(old)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    return out


def _age_h(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, date_format="%Y-%m-%d")
    tmp.replace(path)


def _start_after(old: pd.DataFrame) -> pd.Timestamp | None:
    if old is None or old.empty:
        return None
    return old.index.max() - pd.Timedelta(days=OVERLAP_DAYS)


def _get(url: str, *, session: requests.Session | None = None, ua: bool = True) -> str:
    # FRED tarpits custom agents; everyone else gets a descriptive one.
    getter = session.get if (session is not None and ua) else requests.get
    r = getter(url, timeout=TIMEOUT)
    if r.status_code != 200 or not r.text.strip():
        raise FetchError(f"HTTP {r.status_code} for {url[:140]}")
    return r.text


# ---------------------------------------------------------------------------------------------
# parsers (pure; unit-tested on offline fixtures)
# ---------------------------------------------------------------------------------------------

def parse_fred(text: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise FetchError(f"FRED {name}: unexpected payload")
    df.columns = ["date", name] + list(df.columns[2:])
    out = pd.DataFrame({name: pd.to_numeric(df[name], errors="coerce").to_numpy()},
                       index=pd.to_datetime(df["date"], errors="coerce"))
    return out[out.index.notna()].dropna()


def parse_ecb(text: str, names: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if not {"KEY", "TIME_PERIOD", "OBS_VALUE"} <= set(df.columns):
        raise FetchError(f"ECB: unexpected columns {list(df.columns)[:6]}")
    df["KEY"] = df["KEY"].astype(str)
    if names:
        df["KEY"] = df["KEY"].map(lambda k: names.get(k, names.get(k.split(".", 1)[-1], k)))
    tp = df["TIME_PERIOD"].astype(str)
    # monthly periods "2024-05" → month end; daily stays daily
    dates = pd.to_datetime(tp, format="%Y-%m-%d", errors="coerce")
    monthly = dates.isna() & tp.str.fullmatch(r"\d{4}-\d{2}")
    dates[monthly] = pd.PeriodIndex(tp[monthly], freq="M").to_timestamp(how="end").normalize()
    weekly = dates.isna() & tp.str.contains("W")
    if weekly.any():
        dates[weekly] = pd.to_datetime(tp[weekly] + "-5", format="%G-W%V-%u", errors="coerce")
    df["date"] = dates
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    wide = df.dropna(subset=["date"]).pivot_table(index="date", columns="KEY", values="OBS_VALUE", aggfunc="last")
    wide.columns.name = None
    return wide.sort_index()


def parse_bundesbank(text: str, names: dict[str, str] | None = None) -> pd.DataFrame:
    lines = [ln for ln in text.lstrip("\ufeff").splitlines() if ln.strip()]
    if not lines:
        raise FetchError("Bundesbank: empty payload")
    sep = ";" if lines[0].count(";") > lines[0].count(",") else ","
    head = [c.strip().strip('"') for c in lines[0].split(sep)]
    rows = [ln for ln in lines[1:] if re.match(r'^"?\d{4}-\d{2}-\d{2}', ln)]
    if not rows:
        raise FetchError("Bundesbank: no data rows")
    df = pd.read_csv(io.StringIO("\n".join(rows)), sep=sep, header=None, names=head, dtype=str)
    val_cols = [c for c in head[1:] if c and not c.endswith("_FLAGS")]
    out = pd.DataFrame(index=pd.to_datetime(df[head[0]].str.strip('"'), errors="coerce"))
    for c in val_cols:
        v = df[c].str.replace(",", ".", regex=False) if sep == ";" else df[c]
        key = c.replace("BBSIS.", "")
        out[(names or {}).get(key, key)] = pd.to_numeric(v, errors="coerce").to_numpy()
    out = out[out.index.notna()].dropna(how="all")
    return out.sort_index()


def parse_mof(text: str) -> pd.DataFrame:
    lines = text.lstrip("\ufeff").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str)
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    idx = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    out = pd.DataFrame(index=idx)
    for c in df.columns[1:]:
        m = re.fullmatch(r"(\d+)Y", c.strip())
        if m:
            out[f"jgb_{int(m.group(1))}"] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce").to_numpy()
    return out[out.index.notna()].dropna(how="all").sort_index()


def parse_boj(text: str) -> tuple[pd.DataFrame, str | None]:
    lines = text.splitlines()
    nxt = None
    for ln in lines:
        if ln.startswith("NEXTPOSITION"):
            v = ln.split(",", 1)[1].strip() if "," in ln else ""
            nxt = v or None
        if ln.startswith("STATUS") and not ln.strip().endswith("200"):
            raise FetchError(f"BoJ: {ln}")
    start = next((i for i, ln in enumerate(lines) if ln.startswith("SERIES_CODE")), None)
    if start is None:
        raise FetchError("BoJ: no data block")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str)
    df["date"] = pd.to_datetime(df["SURVEY_DATES"], format="%Y%m%d", errors="coerce")
    df["v"] = pd.to_numeric(df["VALUES"].replace("null", np.nan), errors="coerce")
    wide = df.dropna(subset=["date"]).pivot_table(index="date", columns="SERIES_CODE", values="v", aggfunc="last")
    wide.columns.name = None
    return wide.sort_index(), nxt


def parse_gsw(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), low_memory=False)
    zero = [c for c in df.columns if re.fullmatch(r"SVENY\d\d", c)]
    par = [c for c in df.columns if re.fullmatch(r"SVENPY\d\d", c)]
    out = df[zero + par].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(df["Date"], errors="coerce")
    out.columns = [f"ustz_{int(c[5:])}" for c in zero] + [f"ustgp_{int(c[6:])}" for c in par]
    return out[out.index.notna()].dropna(how="all").sort_index()


# ---------------------------------------------------------------------------------------------
# feed registry
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Feed:
    name: str                     # cache file stem
    source: str                   # fred | ecb | bbk | mof | boj | gsw
    spec: tuple = ()              # source-specific
    names: dict = field(default_factory=dict)
    max_age_h: float = 20.0
    label: str = ""


UST_PAR = {"DGS3MO": "ust_0.25", "DGS6MO": "ust_0.5", "DGS1": "ust_1", "DGS2": "ust_2", "DGS3": "ust_3",
           "DGS5": "ust_5", "DGS7": "ust_7", "DGS10": "ust_10", "DGS20": "ust_20", "DGS30": "ust_30"}
FRED_MACRO = {
    # policy / money market
    "EFFR": "usd_effr", "SOFR": "usd_sofr", "DFEDTARU": "usd_fftu", "DFEDTARL": "usd_fftl",
    "IORB": "usd_iorb", "RRPONTSYD": "usd_rrp_bn",
    # inflation / expectations
    "CPIAUCSL": "us_cpi", "CPILFESL": "us_core_cpi", "T10YIE": "us_be10", "T5YIFR": "us_5y5y_infl",
    # labour / activity
    "PAYEMS": "us_payrolls", "UNRATE": "us_unrate", "ICSA": "us_claims", "INDPRO": "us_indpro",
    # balance sheet / fiscal cash
    "WALCL": "us_fed_assets", "WTREGEN": "us_tga_w",
    # FX / cross-market
    "DTWEXBGS": "usd_broad", "DEXJPUS": "jpy_per_usd", "DEXUSEU": "usd_per_eur", "DEXUSUK": "usd_per_gbp",
    # Japan / euro area long macro via FRED (OECD MEI where native feeds are short)
    "LRUN64TTJPM156S": "jp_unrate", "JPNPROINDMISMEI": "jp_indpro", "CPALTT01JPM659N": "jp_cpi_yoy_oecd",
    "JPNASSETS": "jp_boj_assets",
}

ECB_AAA = {f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{t}Y": f"eur_{t}" for t in range(1, 31)}
ECB_ALL = {f"YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_{t}Y": f"eurall_{t}" for t in range(1, 31)}
BBK_BUND = {f"D.I.ZST.ZI.EUR.S1311.B.A604.R{t:02d}XX.R.A.A._Z._Z.A": f"bund_{t}" for t in range(1, 31)}


def _ecb_key(names: dict[str, str]) -> tuple[str, str]:
    """Collapse full YC keys that differ only in the last dimension into one SDMX '+' query."""
    keys = list(names)
    flow = keys[0].split(".")[0]
    stems = {".".join(k.split(".")[1:-1]) for k in keys}
    if len(stems) != 1:
        raise ValueError("ECB feed keys must share every dimension but the last")
    stem = stems.pop()
    return flow, f"{stem}.{'+'.join(k.split('.')[-1] for k in keys)}"


FEEDS: tuple[Feed, ...] = (
    *(Feed(f"fred_{code.lower()}", "fred", (code,), {code: nm}, label=nm) for code, nm in UST_PAR.items()),
    *(Feed(f"fred_{code.lower()}", "fred", (code,), {code: nm}, max_age_h=20 if code[0] in "DES" else 72, label=nm)
      for code, nm in FRED_MACRO.items()),
    Feed("gsw_zero", "gsw", (), max_age_h=72, label="Fed GSW Svensson zero curve"),
    Feed("ecb_aaa_spot", "ecb", _ecb_key(ECB_AAA), {k.split(".", 1)[1]: v for k, v in ECB_AAA.items()},
         label="ECB AAA spot 1-30y"),
    Feed("ecb_all_spot", "ecb", _ecb_key(ECB_ALL), {k.split(".", 1)[1]: v for k, v in ECB_ALL.items()},
         label="ECB all-issuer spot 1-30y"),
    Feed("ecb_money", "ecb_multi", (
        ("FM", "D.U2.EUR.4F.KR.DFR.LEV", "ecb_dfr"),
        ("EST", "B.EU000A2X2A25.WT", "eur_estr"),
        ("EON", "D.EONIA_TO.RATE", "eur_eonia"),
    ), label="ECB DFR / EUR STR / EONIA"),
    Feed("ecb_macro", "ecb_multi", (
        ("ICP", "M.U2.N.000000.4.ANR", "ea_hicp_yoy"),
        ("ICP", "M.U2.N.XEF000.4.ANR", "ea_core_yoy"),
        ("ICP", "M.U2.Y.000000.3.INX", "ea_hicp_sa"),
        ("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T", "ea_unrate"),
        ("STS", "M.I9.Y.PROD.NS0020.4.000", "ea_indpro"),
        ("ILM", "W.U2.C.T000000.Z5.Z01", "ea_ecb_assets"),
    ), max_age_h=72, label="ECB macro"),
    Feed("bbk_bund", "bbk", ("+".join(k.split(".")[8] for k in BBK_BUND),), dict(BBK_BUND),
         label="Bundesbank Svensson Bund curve 1-30y"),
    Feed("mof_jgb", "mof", (), label="MoF JGB constant maturity 1-40y"),
    Feed("boj_tona", "boj", ("FM01", "STRDCLUCON"), {"STRDCLUCON": "jpy_tona"}, label="BoJ TONA"),
)
BY_NAME = {f.name: f for f in FEEDS}


# ---------------------------------------------------------------------------------------------
# fetch implementations
# ---------------------------------------------------------------------------------------------

def _fetch_fred(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    code = feed.spec[0]
    url = FRED_URL.format(code=code)
    s = _start_after(old)
    if s is not None:
        url += f"&cosd={s:%Y-%m-%d}"
    return parse_fred(_get(url, ua=False), feed.names[code])


def _fetch_ecb(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    flow, key = feed.spec
    url = ECB_URL.format(flow=flow, key=key)
    s = _start_after(old)
    if s is not None:
        url += f"&startPeriod={s:%Y-%m-%d}"
    return parse_ecb(_get(url, session=session), feed.names)


def _fetch_ecb_multi(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    parts = []
    for flow, key, nm in feed.spec:
        url = ECB_URL.format(flow=flow, key=key)
        if old is not None and nm in old.columns and old[nm].notna().any():
            s = old[nm].dropna().index.max() - pd.Timedelta(days=OVERLAP_DAYS * 3)
            url += f"&startPeriod={s:%Y-%m-%d}" if flow in ("FM", "EST", "EON") else f"&startPeriod={s:%Y-%m}"
        try:
            w = parse_ecb(_get(url, session=session))
        except FetchError as e:
            if "HTTP 404" in str(e):  # discontinued / no new obs in window (EONIA)
                continue
            raise
        if w.shape[1]:
            parts.append(w.iloc[:, [0]].rename(columns={w.columns[0]: nm}))
    return pd.concat(parts, axis=1, sort=True) if parts else pd.DataFrame()


def _fetch_bbk(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    stem = "D.I.ZST.ZI.EUR.S1311.B.A604"
    key = f"{stem}.{feed.spec[0]}.R.A.A._Z._Z.A"
    url = BBK_URL.format(key=key)
    s = _start_after(old)
    if s is not None:
        url += f"&startPeriod={s:%Y-%m-%d}"
    return parse_bundesbank(_get(url, session=session), feed.names)


def _fetch_mof(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    cur = parse_mof(_get(MOF_CUR, session=session))
    need_all = old is None or old.empty or (cur.index.min() - old.index.max()).days > 5
    if need_all:
        return merge_wide(parse_mof(_get(MOF_ALL, session=session)), cur)
    return cur


def _fetch_boj(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    db, code = feed.spec
    s = _start_after(old)
    start = f"{s:%Y%m}" if s is not None else "199801"
    parts, pos = [], None
    for _ in range(200):
        url = BOJ_URL.format(db=db, code=code) + f"&startDate={start}"
        if pos:
            url += f"&startPosition={pos}"
        w, pos = parse_boj(_get(url, session=session))
        parts.append(w)
        if not pos:
            break
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.rename(columns=feed.names)


def _fetch_gsw(feed: Feed, old: pd.DataFrame, session) -> pd.DataFrame:
    return parse_gsw(_get(GSW_URL, session=session))


_FETCH = {"fred": _fetch_fred, "ecb": _fetch_ecb, "ecb_multi": _fetch_ecb_multi, "bbk": _fetch_bbk,
          "mof": _fetch_mof, "boj": _fetch_boj, "gsw": _fetch_gsw}


def update_feed(feed: Feed, *, refresh: bool = False, session: requests.Session | None = None) -> dict:
    """Refresh one feed into its cache. Returns a status row; raises only if no data at all."""
    path = cache_path(feed.source.split("_")[0], feed.name)
    old = read_wide(path)
    status = {"feed": feed.name, "label": feed.label, "path": str(path.relative_to(CACHE.parent)
                                                                    if CACHE.parent in path.parents else path)}
    if not refresh and not old.empty and _age_h(path) < feed.max_age_h:
        status.update(action="cached", rows=len(old), last=str(old.index.max().date()))
        return status
    if _offline():
        if old.empty:
            raise FetchError(f"{feed.name}: offline and no cache")
        status.update(action="offline", rows=len(old), last=str(old.index.max().date()))
        return status
    try:
        new = _FETCH[feed.source](feed, old, session or requests.Session())
        merged = merge_wide(old, new)
        if len(merged) < len(old):
            raise FetchError(f"{feed.name}: merge shrank cache {len(old)}→{len(merged)}")
        _write(path, merged)
        status.update(action="fetched", rows=len(merged), new_rows=len(merged) - len(old),
                      last=str(merged.index.max().date()) if len(merged) else None)
    except Exception as err:  # noqa: BLE001 - one feed never kills the refresh
        if old.empty:
            raise FetchError(f"{feed.name}: {err}") from err
        status.update(action="fetch_failed_kept_cache", error=str(err)[:200], rows=len(old),
                      last=str(old.index.max().date()))
    return status


def refresh_all(*, refresh: bool = False, workers: int = 6, only: list[str] | None = None) -> list[dict]:
    feeds = [f for f in FEEDS if not only or f.name in only]
    session = requests.Session()
    session.headers.update({"User-Agent": AGENT})
    out = []
    with ThreadPoolExecutor(max_workers=min(workers, 6)) as pool:
        futs = {pool.submit(update_feed, f, refresh=refresh, session=session): f for f in feeds}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                out.append(fut.result())
            except Exception as err:  # noqa: BLE001
                out.append({"feed": f.name, "label": f.label, "action": "FAILED", "error": str(err)[:300]})
    return sorted(out, key=lambda r: r["feed"])


def load(name: str) -> pd.DataFrame:
    """Cached wide frame for a feed (no network)."""
    f = BY_NAME[name]
    return read_wide(cache_path(f.source.split("_")[0], f.name))


def load_fred(names: list[str] | None = None) -> pd.DataFrame:
    """All cached FRED series as one wide frame, columns = friendly names."""
    parts = []
    for f in FEEDS:
        if f.source != "fred":
            continue
        nm = next(iter(f.names.values()))
        if names and nm not in names:
            continue
        w = load(f.name)
        if not w.empty:
            parts.append(w.iloc[:, [0]].rename(columns={w.columns[0]: nm}))
    return pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore cache age; fetch the tail")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    rows = refresh_all(refresh=args.refresh, only=args.only)
    for r in rows:
        print(f"{r['feed']:<22} {r.get('action',''):<24} rows={r.get('rows','')!s:<7} last={r.get('last','')} "
              f"{r.get('error','')}")
    bad = [r for r in rows if r.get("action") == "FAILED"]
    if bad:
        raise SystemExit(f"{len(bad)} feed(s) failed with no cache: {[r['feed'] for r in bad]}")


if __name__ == "__main__":
    main()
