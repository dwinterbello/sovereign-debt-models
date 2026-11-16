"""IMF WEO vintages (April / October releases) for CPI and real growth, one long table.

``data/debt/weo.parquet`` stays the single live vintage the chain runs on. This module archives the
*history* of WEO releases so WEO can be scored as a forecaster (experiments/own_cpi_vs_weo_vintages.py).

Sources, in order
  1. DBnomics mirror of the IMF WEO database, one dataset per release: ``IMF/WEO:YYYY-MM`` for
     2008-04 → 2025-04 (2011-10 was never mirrored). Series ``{ISO3}.{SUBJECT}.pcent_change``.
  2. IMF SDMX 3.0 (api.imf.org, no key): dataflows ``IMF.RES/WEO_{YYYY}_{APR|OCT}_VINTAGE`` (archived
     releases from 2025-10) plus the live ``IMF.RES/WEO`` flow, whose release label is inferred from the
     pull date (Apr-release between April and September, Oct-release otherwise) and cross-checked against
     ``weo.parquet``. Older ``WEO`` flow versions (6.0.0 = the April-2025 release) are deduplicated
     against DBnomics on values.
  imf.org file archives (WEO*all.xls / WEOhistorical.xlsx) sit behind Akamai and return 403 to every
  non-browser client; not worked around.

Columns: country, vintage_date (release month, first day), vintage_label (YYYY-MM), vintage_source,
year, var (cpi | real_growth), value. Raw JSON per vintage is kept under ``weo_vintages/raw/``.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.weo_vintages [--force]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "debt"
DIR = RAW / "weo_vintages"
RAW_DIR = DIR / "raw"
PARQ = DIR / "weo_vintages.parquet"
CC = ("us", "uk", "de", "fr", "it", "es", "jp", "se", "no")
ISO3 = {"us": "USA", "uk": "GBR", "de": "DEU", "fr": "FRA", "it": "ITA", "es": "ESP",
        "jp": "JPN", "se": "SWE", "no": "NOR"}
INV = {v: k for k, v in ISO3.items()}
VARS = {"PCPIPCH": "cpi", "NGDP_RPCH": "real_growth"}
COLS = ["country", "vintage_date", "vintage_label", "vintage_source", "year", "var", "value"]
UA = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}
DBN = "https://api.db.nomics.world/v22"
SDMX = "https://api.imf.org/external/sdmx/3.0"
SDMX_JSON = {"Accept": "application/vnd.sdmx.data+json;version=2.0.0", "User-Agent": "curl/8.7.1"}
SDMX_STRUCT = {"Accept": "application/vnd.sdmx.structure+json;version=2.0.0", "User-Agent": "curl/8.7.1"}


def _get(url: str, headers: dict | None = None, timeout: int = 120, **kw) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers=headers or UA, **kw)
    r.raise_for_status()
    return r


def _label_to_date(label: str) -> pd.Timestamp:
    return pd.Timestamp(f"{label}-01")


# ---------------------------------------------------------------- DBnomics (2008-04 → 2025-04)
def dbnomics_vintages() -> list[str]:
    js = _get(f"{DBN}/datasets/IMF?limit=500").json()
    docs = js["datasets"]["docs"] if isinstance(js["datasets"], dict) and "docs" in js["datasets"] else js["datasets"]
    out = []
    for d in docs:
        m = re.fullmatch(r"WEO:(\d{4})-(\d{2})", d["code"])
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}")
    return sorted(out)


def fetch_dbnomics(label: str, force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"dbnomics_WEO_{label}.json"
    if cache.exists() and not force:
        js = json.loads(cache.read_text())
    else:
        dims = json.dumps({"weo-country": list(ISO3.values()), "weo-subject": list(VARS)})
        js = _get(f"{DBN}/series/IMF/WEO:{label}", params={"dimensions": dims, "observations": 1, "limit": 200}).json()
        cache.write_text(json.dumps(js))
    rows = []
    for s in js["series"]["docs"]:
        dim = s["dimensions"]
        cc = INV.get(dim.get("weo-country"))
        var = VARS.get(dim.get("weo-subject"))
        if cc is None or var is None:
            continue
        for p, v in zip(s["period"], s["value"]):
            if v is None or v == "NA":
                continue
            try:
                yr = int(str(p)[:4])
            except ValueError:
                continue
            rows.append((cc, _label_to_date(label), label, "dbnomics IMF/WEO", yr, var, float(v)))
    return pd.DataFrame(rows, columns=COLS)


# ---------------------------------------------------------------- IMF SDMX 3.0 (2025-10 →, live flow)
def sdmx_flows() -> list[tuple[str, str, str | None]]:
    """(dataflow id, version, label-or-None) for every WEO flow under IMF.RES; None = live flow."""
    js = _get(f"{SDMX}/structure/dataflow/IMF.RES/*/*?detail=allstubs", headers=SDMX_STRUCT).json()
    out = []
    for d in js["data"]["dataflows"]:
        m = re.fullmatch(r"WEO_(\d{4})_(APR|OCT)_VINTAGE", d["id"])
        if m:
            out.append((d["id"], d["version"], f"{m.group(1)}-{'04' if m.group(2) == 'APR' else '10'}"))
        elif d["id"] == "WEO":
            out.append((d["id"], d["version"], None))
    return out


def live_label(today: pd.Timestamp | None = None) -> str:
    """Release the live WEO flow carries, by calendar: April release Apr–Sep, October release otherwise."""
    t = today or pd.Timestamp.today()
    if 4 <= t.month <= 9:
        return f"{t.year}-04"
    return f"{t.year}-10" if t.month >= 10 else f"{t.year - 1}-10"


def fetch_sdmx(flow: str, version: str, label: str | None, source: str, force: bool = False) -> pd.DataFrame:
    """One SDMX flow version → long rows. ``label`` None (unlabelled flow) leaves vintage_date NaT."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"sdmx_{flow}_{version}.json"
    if cache.exists() and not force:
        js = json.loads(cache.read_text())
    else:
        key = "+".join(ISO3.values()) + "." + "+".join(VARS) + ".A"
        js = _get(f"{SDMX}/data/dataflow/IMF.RES/{flow}/{version}/{key}",
                  headers=SDMX_JSON, params={"dimensionAtObservation": "TIME_PERIOD"}).json()
        cache.write_text(json.dumps(js))
    ds = js["data"]["dataSets"][0]
    st = js["data"]["structures"][0]
    dims = st["dimensions"]["series"]
    tdim = st["dimensions"]["observation"][0]["values"]
    rows = []
    for k, s in ds["series"].items():
        idx = [int(x) for x in k.split(":")]
        keyvals = {dims[i]["id"]: dims[i]["values"][idx[i]]["id"] for i in range(len(idx))}
        cc = INV.get(keyvals.get("COUNTRY"))
        var = VARS.get(keyvals.get("INDICATOR"))
        if cc is None or var is None:
            continue
        for o, v in s["observations"].items():
            if v[0] is None:
                continue
            yr = int(str(tdim[int(o)].get("value") or tdim[int(o)].get("id"))[:4])
            rows.append((cc, _label_to_date(label) if label else pd.NaT, label or "", source, yr, var, float(v[0])))
    return pd.DataFrame(rows, columns=COLS)


def _same_vintage(a: pd.DataFrame, b: pd.DataFrame, tol: float = 0.02) -> bool:
    """Two pulls are the same release if every shared (country, var, year) agrees within DBnomics rounding."""
    m = a.merge(b, on=["country", "var", "year"], suffixes=("_a", "_b"))
    if len(m) < 50:
        return False
    return bool((m["value_a"] - m["value_b"]).abs().max() <= tol)


def _check_live_against_weo_parquet(live: pd.DataFrame) -> str:
    p = RAW / "weo.parquet"
    if not p.exists():
        return "weo.parquet absent — live label unverified"
    w = pd.read_parquet(p)
    w = w[w["var"].isin(VARS.values())][["country", "var", "year", "value"]]
    m = live.merge(w, on=["country", "var", "year"], suffixes=("_v", "_live"))
    if m.empty:
        return "no overlap with weo.parquet"
    gap = float((m["value_v"] - m["value_live"]).abs().max())
    return f"live flow vs weo.parquet max |Δ| {gap:.3f} on {len(m)} cells"


# ---------------------------------------------------------------- assemble
def refresh_weo_vintages(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Pull every reachable vintage (cached per release), dedupe, write ``weo_vintages.parquet``."""
    frames: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    # 1. DBnomics
    try:
        labels = dbnomics_vintages()
    except Exception as e:  # noqa: BLE001
        labels = sorted(re.findall(r"dbnomics_WEO_(\d{4}-\d{2})\.json", " ".join(p.name for p in RAW_DIR.glob("*.json")))) if RAW_DIR.exists() else []
        notes.append(f"DBnomics catalogue unavailable ({e}); using cached vintages {labels}")
    for lab in labels:
        try:
            d = fetch_dbnomics(lab, force=force)
            if len(d):
                frames[lab] = d
        except Exception as e:  # noqa: BLE001
            notes.append(f"DBnomics {lab} failed: {e}")
    # 2. IMF SDMX 3.0
    try:
        flows = sdmx_flows()
    except Exception as e:  # noqa: BLE001
        flows = []
        notes.append(f"IMF SDMX catalogue unavailable ({e})")
    live_frames = []
    for flow, ver, lab in sorted(flows, key=lambda t: (t[2] or "9999", t[1])):
        try:
            if lab is not None:
                d = fetch_sdmx(flow, ver, lab, f"imf sdmx3 {flow}/{ver}", force=force)
                if lab in frames and _same_vintage(frames[lab], d):
                    continue  # DBnomics already has this release
                frames[lab] = d  # IMF's own copy wins on a label collision (more decimals)
            else:
                live_frames.append((flow, ver))
        except Exception as e:  # noqa: BLE001
            notes.append(f"SDMX {flow}/{ver} failed: {e}")
    # live / unlabelled WEO flow versions: dedupe on values, label the newest by calendar
    live_frames.sort(key=lambda t: [int(x) for x in t[1].split(".")])
    for i, (flow, ver) in enumerate(live_frames):
        is_newest = i == len(live_frames) - 1
        lab = live_label() if is_newest else None
        try:
            d = fetch_sdmx(flow, ver, lab, f"imf sdmx3 {flow}/{ver}" + (" (live, label inferred)" if is_newest else ""), force=force)
        except Exception as e:  # noqa: BLE001
            notes.append(f"SDMX {flow}/{ver} failed: {e}")
            continue
        dup = next((k for k, f in frames.items() if _same_vintage(f, d)), None)
        if dup is not None:
            notes.append(f"SDMX WEO/{ver} duplicates vintage {dup}; dropped")
            continue
        if lab is None:
            notes.append(f"SDMX WEO/{ver} matches no known release and is not the live flow; dropped")
            continue
        notes.append(f"SDMX WEO/{ver} labelled {lab} by calendar; " + _check_live_against_weo_parquet(d))
        if lab in frames:
            notes.append(f"label {lab} already present from an explicit vintage; live flow kept under that label")
        frames[lab] = d
    if not frames:
        if PARQ.exists():
            notes.append("no vintage fetched; keeping cache")
            if verbose:
                print("\n".join(notes))
            return pd.read_parquet(PARQ)
        raise RuntimeError("no WEO vintage could be fetched: " + "; ".join(notes))
    out = pd.concat(frames.values(), ignore_index=True)
    out["year"] = out["year"].astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).drop_duplicates(["country", "vintage_label", "year", "var"]).sort_values(
        ["vintage_date", "country", "var", "year"]).reset_index(drop=True)
    DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PARQ, index=False)
    (DIR / "refresh_notes.json").write_text(json.dumps({
        "pulled": pd.Timestamp.today().strftime("%Y-%m-%d"), "n_vintages": int(out["vintage_label"].nunique()),
        "vintages": sorted(out["vintage_label"].unique().tolist()), "notes": notes}, indent=1))
    if verbose:
        print(f"weo_vintages: {out['vintage_label'].nunique()} vintages, {len(out)} rows → {PARQ}")
        for n in notes:
            print("  ", n)
    return out


def load(refresh: bool = False) -> pd.DataFrame:
    if PARQ.exists() and not refresh:
        return pd.read_parquet(PARQ)
    return refresh_weo_vintages(force=refresh)


def pivot(var: str = "cpi") -> pd.DataFrame:
    """country × vintage_label × year → value, for one variable."""
    d = load()
    return d[d["var"] == var].pivot_table(index=["country", "vintage_label"], columns="year", values="value")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download every vintage (caches are per release)")
    a = ap.parse_args()
    d = refresh_weo_vintages(force=a.force)
    print(d.groupby("vintage_label").agg(n=("value", "size"), y0=("year", "min"), y1=("year", "max"),
                                         src=("vintage_source", "first")).to_string())
