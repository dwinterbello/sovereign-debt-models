"""One data pull for the sovereign-debt models. Run it where there is internet; carry one zip into the sandbox.

On a machine with internet (this folder, after `pip install -r requirements.txt`):

    python pull_data.py                         # every source, then data_pull_<yyyymmdd_hhmm>.zip in this folder
    python pull_data.py --only us issuance      # a subset of steps (python pull_data.py --list shows them)
    python pull_data.py --full                  # zip every data file, not only the ones this pull changed

In the sandbox (the same folder, no internet):

    python pull_data.py --unpack data_pull_20261120_0900.zip
    python -m sovereign_cash.run_all            # rerun the models on the new data

Each step re-downloads its sources into the folders the models read (sovereign_cash/data, issuance/cache,
us/fiscal/data, correlations/g4/cache, ...). The `countries` step is the models' own refresh, so it also reruns the
models; that is harmless, the sandbox reruns them anyway. A source that fails keeps its previous file and the pull
carries on: the summary at the end, and the zip's manifest, list what failed.

The zip holds data files only (never .py code): every file under this folder whose modification time is after the
pull started (or all of them with --full), plus `pull_manifest.json` (steps, status, files). --unpack refuses
anything that is code or would land outside this folder.

Fixed inputs, not refreshed by this pull: house CPI paths (sovereign_cash/derived/own_inflation_*), UK RPI
(sovereign_cash/_vendor_data), BoE gilt curves and DMO operations (uk_flows/output), UK long-end tables
(projects/uk_long_end/workbook), NY Fed dealer positions (flow/output/nyfed.parquet).
"""
from __future__ import annotations

import argparse
import io
import re
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", "venv", ".git"}
CODE = {".py", ".pyc", ".pyd", ".so", ".dll", ".exe", ".bat", ".cmd", ".ps1", ".sh"}
NOT_DATA = {"README.md", "CLAUDE.md", "requirements.txt", "pull_manifest.json"}
SPREAD_IN = ROOT / "sovereign_cash" / "debt" / "note" / "spread_fv_audit" / "inputs"
# lines the fetchers print when a source fails but the step carries on (they keep the old file)
SOURCE_FAIL = re.compile(r"(?i)(fetch_failed|\bfailed\b|unavailable|ProxyError|ConnectionError|Max retries exceeded)")
EA = ("AT", "BE", "DE", "ES", "FI", "FR", "IE", "IT", "NL", "PT")


def _spread_inputs() -> None:
    """ECB 10y convergence yields, Eurostat general government, AMECO: the spread fair-value model's inputs.

    Each file is replaced only if the new download has the same shape as the old one (same columns, every
    country, no shorter history); otherwise the old file stays and the step reports it.
    """
    import pandas as pd
    import requests

    problems = []

    # ECB Data Portal: IRS.M.<CC>.L.L40.CI.0000.EUR.N.Z, the raw csvdata download (same layout as the shipped file)
    url = ("https://data-api.ecb.europa.eu/service/data/IRS/M." + "+".join(EA) +
           ".L.L40.CI.0000.EUR.N.Z?format=csvdata")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        new = pd.read_csv(io.StringIO(r.text))
        old = pd.read_csv(SPREAD_IN / "ecb_irs_m.csv")
        ok = ({"REF_AREA", "TIME_PERIOD", "OBS_VALUE"} <= set(new.columns) and set(EA) <= set(new.REF_AREA)
              and new.TIME_PERIOD.max() >= old.TIME_PERIOD.max())
        if ok:
            (SPREAD_IN / "ecb_irs_m.csv").write_text(r.text, encoding="utf-8")
            print(f"  ECB 10y yields: to {new.TIME_PERIOD.max()}")
        else:
            problems.append("ECB yields: unexpected shape, kept the old file")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ECB yields: {exc}")

    # Eurostat: GD / B9 / D41PAY (gov_10dd_edpt1), TR (gov_10a_main), all % GDP; B1GQ (nama_10_gdp), EUR mn
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    geo = "&".join(f"geo={g}" for g in EA)
    asks = [("gov_10dd_edpt1", "unit=PC_GDP&sector=S13&na_item=GD&na_item=B9&na_item=D41PAY"),
            ("gov_10a_main", "unit=PC_GDP&sector=S13&na_item=TR"),
            ("nama_10_gdp", "unit=CP_MEUR&na_item=B1GQ")]
    try:
        parts = []
        for ds, q in asks:
            r = requests.get(f"{base}{ds}?format=JSON&lang=EN&sinceTimePeriod=1995&{geo}&{q}", timeout=120)
            r.raise_for_status()
            parts.append(_jsonstat(r.json()))
        long = pd.concat(parts)
        long["na_item"] = long["na_item"].replace({"B1GQ": "GDP"})
        long["year"] = long["time"].astype(int)
        new = (long.pivot_table(index=["geo", "year"], columns="na_item", values="value", aggfunc="first")
               .reset_index()[["geo", "year", "GD", "B9", "D41PAY", "TR", "GDP"]].sort_values(["geo", "year"]))
        old = pd.read_csv(SPREAD_IN / "eurostat_gg.csv")
        both = old.merge(new, on=["geo", "year"], suffixes=("_o", "_n"))
        ok = (set(EA) <= set(new.geo) and new.year.max() >= old.year.max() and len(both) >= 0.9 * len(old)
              and (both.GD_o - both.GD_n).abs().median() < 2)          # revisions are small; a unit mix-up is not
        if ok:
            new.to_csv(SPREAD_IN / "eurostat_gg.csv", index=False)
            print(f"  Eurostat general government: {new.year.min()}-{new.year.max()}")
        else:
            problems.append("Eurostat: unexpected shape or values, kept the old file")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Eurostat: {exc}")

    # AMECO: from the copy the countries step just refreshed (sovereign_cash/data/debt/ameco.parquet)
    try:
        a = pd.read_parquet(ROOT / "sovereign_cash" / "data" / "debt" / "ameco.parquet")
        a = a[a.country.isin(["de", "fr", "it", "es"])]
        new = a[["country", "year", "var", "value", "unit", "vintage"]].sort_values(["country", "var", "year"])
        old = pd.read_csv(SPREAD_IN / "ameco_gg.csv")
        if set(old["var"]) <= set(new["var"]) and new.year.max() >= old.year.max():
            new.to_csv(SPREAD_IN / "ameco_gg.csv", index=False)
            print(f"  AMECO: {new.vintage.iloc[0]}")
        else:
            problems.append("AMECO: unexpected shape, kept the old file")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"AMECO: {exc}")

    if problems:
        raise RuntimeError("; ".join(problems))


MACRO_PANEL = ROOT / "correlations" / "derived" / "macro_panel_monthly.csv"
# the five panel columns the UK / Norway macro blocks read: ONS time series (path, dataset) and FRED Brent
ONS_SERIES = {"cpi_all": "/economy/inflationandpriceindices/timeseries/d7bt/mm23",
              "retail_sales_index": "/businessindustryandtrade/retailindustry/timeseries/j5ek/drsi",
              "gdp_gva_index": "/economy/grossdomesticproductgdp/timeseries/ecy2/mgdp",
              "awe_total_index": "/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9/lms"}


def _macro_panel() -> None:
    """UK CPI (D7BT), retail sales volume (J5EK), monthly GVA (ECY2), AWE total pay (KAB9, rebased to the panel's
    index) and Brent (FRED DCOILBRENTEU, monthly mean): correlations/derived/macro_panel_monthly.csv.

    A column is replaced only where the new series agrees with the old one over their overlap (revisions are small);
    otherwise the old column stays and the step reports it.
    """
    import pandas as pd
    import requests

    old = pd.read_csv(MACRO_PANEL, parse_dates=["date"]).set_index("date")
    new, problems = old.copy(), []
    for col, path in ONS_SERIES.items():
        try:
            r = requests.get(f"https://www.ons.gov.uk{path}/data", timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            m = r.json()["months"]
            s = pd.Series([float(x["value"]) for x in m],
                          index=pd.to_datetime([x["date"] for x in m], format="%Y %b") + pd.offsets.MonthEnd(0))
            o = old[col].dropna()
            both = s.index.intersection(o.index)
            if col == "awe_total_index":              # KAB9 is GBP per week; the panel holds it as an index
                s = s * float((o[both] / s[both]).median())
            if len(both) < 24 or (s[both] - o[both]).abs().median() > 0.5:
                problems.append(f"{col}: does not match the existing series, kept")
                continue
            new = new.reindex(new.index.union(s.index))
            new[col] = s.combine_first(new[col])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{col}: {str(exc)[:120]}")
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU", timeout=60)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["date", "v"]
        d["v"] = pd.to_numeric(d["v"], errors="coerce")
        s = d.dropna().set_index(pd.to_datetime(d.dropna()["date"]))["v"].resample("ME").mean()
        o = old["brent_usd"].dropna()
        both = s.index.intersection(o.index)
        if len(both) < 24 or (s[both] - o[both]).abs().median() > 1.0:
            problems.append("brent_usd: does not match the existing series, kept")
        else:
            new = new.reindex(new.index.union(s.index))
            new["brent_usd"] = s.combine_first(new["brent_usd"])
    except Exception as exc:  # noqa: BLE001
        problems.append(f"brent_usd: {str(exc)[:120]}")
    new.sort_index().reset_index().rename(columns={"index": "date"}).to_csv(MACRO_PANEL, index=False,
                                                                            date_format="%Y-%m-%d")
    print("  macro panel to", new.dropna(how="all").index.max().date())
    if problems:
        raise RuntimeError("; ".join(problems))


def _jsonstat(js: dict):
    """A JSON-stat 2.0 dataset (Eurostat) as a long frame: one column per dimension, plus value."""
    import pandas as pd

    ids, sizes = js["id"], js["size"]
    cats = []
    for d in ids:
        idx = js["dimension"][d]["category"]["index"]
        cats.append({v: k for k, v in idx.items()} if isinstance(idx, dict) else dict(enumerate(idx)))
    rows = []
    for k, v in js["value"].items():
        k, pos = int(k), []
        for s in reversed(sizes):
            pos.append(k % s)
            k //= s
        pos.reverse()
        rows.append({**{d: cats[i][pos[i]] for i, d in enumerate(ids)}, "value": v})
    return pd.DataFrame(rows)


# name, what it pulls, command (run from this folder) or function. Order matters: the countries step (the models'
# own refresh) reads what issuance, us and curves have just written.
STEPS = [
    ("issuance", "bond auctions and buybacks (UK, DE, FR, IT, ES, US, JP, SE), BoJ / APF / SOMA / ECB holdings, "
                 "investor holders, FX, euro-area government yields",
     [PY, "-m", "issuance.run_refresh", "--force", "--skip-dashboard"]),
    ("us", "US Treasury daily statement, auctions, buybacks, debt limit, FRED, NY Fed rates, ACM term premium",
     [PY, "us/fiscal/run_refresh.py", "--force", "--skip-dashboard"]),
    ("curves", "Bundesbank Bund curve, MoF JGB, ECB curves and macro, BoJ TONA, FRED, Fed GSW zero curve",
     [PY, "-c", "from correlations.g4.sources import refresh_all\n"
                "for r in refresh_all(refresh=True): print(' ', r.get('feed'), r.get('action'), r.get('error', ''))"]),
    ("monthly", "Spain IGAE and Japan MoF monthly primary balance files",
     [PY, "-c", "import sys\nfrom sovereign_cash.debt.note.monthly_primary import fetch_es_igae, fetch_jp_mof\nbad = 0\n"
                "for f in (fetch_es_igae, fetch_jp_mof):\n"
                "    try:\n        f(refresh=True)\n"
                "    except Exception as e:\n        print('FAILED', f.__name__, str(e)[:200]); bad = 1\n"
                "sys.exit(bad)"]),
    ("weo", "IMF WEO vintages (for scoring the WEO as a forecaster)",
     [PY, "-c", "from sovereign_cash.debt.sources import refresh_weo_vintages\nrefresh_weo_vintages(force=True)"]),
    ("countries", "every country's cash-model sources (UK, DE, FR, IT, ES, SE, NO, JP) and the debt board's "
                  "(IMF WEO, AMECO, OECD, CBO, ...), then the models' own rebuild",
     [PY, "-m", "sovereign_cash.run_all", "--refresh"]),
    ("macro_panel", "UK CPI, retail sales, monthly GVA, AWE (ONS) and Brent (FRED) for the UK / Norway macro blocks",
     _macro_panel),
    ("spread_fv", "ECB 10y yields, Eurostat general government, AMECO: the spread fair-value inputs", _spread_inputs),
]


def _data_files(since: float | None) -> list[Path]:
    out = []
    for root, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for nm in names:
            p = Path(root) / nm
            rel = p.relative_to(ROOT)
            if (p.suffix.lower() in CODE or nm in NOT_DATA or nm.startswith("data_pull_")
                    or (len(rel.parts) == 1 and nm.startswith("pull_log"))):
                continue
            if since is None or p.stat().st_mtime >= since:
                out.append(p)
    return out


def pull(only: list[str] | None, full: bool) -> int:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
           "SOVDEBT_PULL": "1"}   # lets the models apply their cache-age refetch rules (a normal run reads the caches)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = ROOT / f"pull_log_{stamp}.txt"
    start = time.time() - 2
    status = []
    with log_path.open("w", encoding="utf-8") as log:
        for name, what, how in STEPS:
            if only and name not in only:
                continue
            print(f"\n=== {name}: {what}", flush=True)
            log.write(f"\n=== {name}: {what}\n")
            t0, ok, note = time.time(), True, ""
            if callable(how):
                try:
                    how()
                except Exception as exc:  # noqa: BLE001
                    ok, note = False, str(exc)[:400]
                log.write(note + "\n")
            else:
                p = subprocess.Popen(how, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
                tail, fails = [], []
                for line in p.stdout:
                    print("  " + line.rstrip(), flush=True)
                    log.write(line)
                    log.flush()
                    tail = (tail + [line.rstrip()])[-6:]
                    if SOURCE_FAIL.search(line):
                        fails.append(line.strip()[:160])
                ok = p.wait() == 0
                note = (f"exit {p.returncode}: " + " | ".join(tail)[-400:]) if not ok else (
                    f"{len(fails)} source(s) failed and kept their old files, e.g. {fails[0]}" if fails else "")
            state = "FAILED" if not ok else ("partial" if note else "ok")
            status.append({"step": name, "ok": ok, "state": state, "seconds": round(time.time() - t0), "note": note})
            print(f"--- {name}: {state} ({time.time() - t0:,.0f}s)", flush=True)
    files = _data_files(None if full else start)
    zpath = ROOT / f"data_pull_{stamp}.zip"
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "full": full, "steps": status,
                "files": {p.relative_to(ROOT).as_posix(): p.stat().st_size for p in files}}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("pull_manifest.json", json.dumps(manifest, indent=1))
    print("\nsummary")
    for s in status:
        print(f"  {s['step']:<10} {s['state']:<8} {s['seconds']:>5}s  {s['note'][:160]}")
    mb = zpath.stat().st_size / 1e6
    print(f"\n{len(files):,} data files -> {zpath.name} ({mb:,.0f} MB); log: {log_path.name}")
    print("A step that failed kept its previous files; the zip still holds everything that did update.")
    print("ok = every source updated; partial = some sources failed and kept their previous files; FAILED = the step "
          "stopped (its earlier sources may still have updated).")
    return 0 if all(s["state"] == "ok" for s in status) else 1


def unpack(zpath: Path) -> int:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        bad = [n for n in names if n.startswith(("/", "\\")) or ".." in Path(n).parts or ":" in n
               or Path(n).suffix.lower() in CODE]
        if bad:
            print(f"refused: {len(bad)} entries are code or point outside this folder, e.g. {bad[:3]}")
            return 1
        man = json.loads(z.read("pull_manifest.json")) if "pull_manifest.json" in names else {}
        for n in names:
            if n == "pull_manifest.json":
                continue
            dst = ROOT / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as out:
                out.write(src.read())
    (ROOT / "pull_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"unpacked {len(names) - 1:,} data files from {zpath.name} (pulled {man.get('created', '?')})")
    for s in man.get("steps", []):
        if s.get("state", "ok" if s["ok"] else "FAILED") != "ok":
            print(f"  note: step {s['step']} was {s.get('state', 'FAILED')} in that pull; what failed is from the "
                  f"previous pull ({s['note'][:120]})")
    print("next: python -m sovereign_cash.run_all")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="steps to run (default: all)")
    ap.add_argument("--full", action="store_true", help="zip every data file, not only those this pull changed")
    ap.add_argument("--unpack", type=Path, help="lay a data_pull_*.zip into this folder (the sandbox side)")
    ap.add_argument("--list", action="store_true", help="list the steps")
    a = ap.parse_args()
    if a.list:
        for name, what, _ in STEPS:
            print(f"  {name:<10} {what}")
        return 0
    if a.unpack:
        return unpack(a.unpack)
    unknown = set(a.only or []) - {s[0] for s in STEPS}
    if unknown:
        ap.error(f"unknown step(s): {', '.join(sorted(unknown))}; see --list")
    return pull(a.only, a.full)


if __name__ == "__main__":
    raise SystemExit(main())
