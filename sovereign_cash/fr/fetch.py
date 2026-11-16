"""France: raw sources for the central-government cash framework, cached under ``sovereign_cash/data/fr/``.

insee_smb    INSEE BDM dataflow ``COMPTES-ETAT``: the Direction du Budget "Situation mensuelle
             budgétaire de l'État" (SMB) as monthly YTD series (EUR mn) — solde général
             d'exécution (001717255), dépenses (001717256), recettes (001717257), solde des
             comptes spéciaux (001717258). Jan-2009 →, ~5-week lag. Same numbers as the SMB PDF;
             budget.gouv.fr itself is behind Incapsula and answers scripted clients with 403.
insee_debt   INSEE BDM dataflow ``DETTE-NEGOCIABLE-ETAT`` (AFT monthly bulletin): negotiable
             debt outstanding, EUR mn, month-end — total (001711531), ≤1y original maturity =
             BTF (001711532), >1y = OAT (001711533), inflation-indexed (001738854), YTD changes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fr"
BDM = "https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/"

SMB_IDS = {
    "001717255": "solde_ytd",
    "001717256": "depenses_ytd",
    "001717257": "recettes_ytd",
    "001717258": "comptes_speciaux_ytd",
}
DEBT_IDS = {
    "001711531": "debt_total",
    "001711532": "btf",
    "001711533": "mlt",
    "001738853": "fixed",
    "001738854": "indexed",
    "001738856": "btf_ytd_change",
    "001738857": "mlt_ytd_change",
}


def _parse_sdmx(text: str) -> pd.DataFrame:
    cols = {}
    for s in re.finditer(r"<Series ([^>]*)>(.*?)</Series>", text, re.S):
        idb = re.search(r'IDBANK="(\d+)"', s.group(1)).group(1)
        obs = re.findall(r'TIME_PERIOD="([^"]*)" OBS_VALUE="([^"]*)"', s.group(2))
        ser = pd.Series({pd.Period(p, "M").to_timestamp(): float(v) for p, v in obs if v not in ("", "NaN")})
        cols[idb] = ser.sort_index()
    return pd.DataFrame(cols)


def _fetch(ids: dict[str, str], name: str, refresh: bool) -> pd.DataFrame:
    dest = DATA / f"insee_{name}.xml"
    if dest.exists() and not refresh:
        pass
    else:
        from issuance.http import get_bytes

        try:
            payload = get_bytes(BDM + "+".join(ids), timeout=90, min_bytes=1000)
            if b"<Series " not in payload:
                raise RuntimeError(f"INSEE BDM returned no series for {name}")
            DATA.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
        except Exception as e:  # noqa: BLE001 — keep cache on refresh failure
            if dest.exists() and b"<Series " in dest.read_bytes():
                print(f"FR INSEE {name}: fetch failed ({e}); keeping cache")
            else:
                raise
    df = _parse_sdmx(dest.read_text(encoding="utf-8"))
    missing = set(ids) - set(df.columns)
    if missing:
        raise RuntimeError(f"INSEE {name}: missing idbanks {sorted(missing)}")
    return df.rename(columns=ids)[list(ids.values())] / 1000.0  # EUR bn


def load_smb(refresh: bool = False) -> pd.DataFrame:
    return _fetch(SMB_IDS, "smb", refresh)


def load_debt(refresh: bool = False) -> pd.DataFrame:
    return _fetch(DEBT_IDS, "debt", refresh)


AFT_ENCOURS = "https://www.aft.gouv.fr/en/encours-detaille-{kind}"
ENCOURS_KINDS = ("oat", "oatei", "oati")
AFT = "https://www.aft.gouv.fr"
RAW_AFT = ROOT.parent / "data" / "raw" / "aft" / "KeyFigures"
# index-coefficient files: AFT key-figures page, cached copy under data/fr/, fallback raw tree
COEF_PAGES = {"oatei": ("/en/oateuroi-key-figures", "coef_oatei", RAW_AFT / "OATeuroi"),
              "oati": ("/en/oatis-key-figures", "coef_oati", RAW_AFT / "OATis")}


def _aft_encours_ok(payload: bytes) -> bool:
    """True if the HTML has a parseable outstanding table (not a Cloudflare interstitial).

    Challenge markers live in the head; ``Code ISIN`` is in the table body (often past 8 KB
    of Eulerian / Drupal head), so scan the whole payload for the column header.
    """
    head = payload[:4000].lower()
    if b"just a moment" in head or b"cf-browser-verification" in head:
        return False
    return b"code isin" in payload.lower()


def load_aft_encours(refresh: bool = False) -> pd.DataFrame:
    """AFT outstanding by line (nominal, EUR bn) for OAT / OAT€i / OATi.

    The live pages are current to the latest settled auction and net every buyback done; a line
    can stay listed for a few days after it matures. Cloudflare blocks plain clients, so the
    fetch impersonates a browser; the HTML is cached under data/fr/. On refresh, a 403/challenge
    keeps the cached HTML (same contract as ``fetch_coef_file`` / ``debt.sources``) so
    ``run_all --refresh`` does not fail the FR country when AFT flaps. A response without an
    outstanding table never overwrites a good cache.
    """
    import io

    from issuance.http import get_bytes

    rows = []
    DATA.mkdir(parents=True, exist_ok=True)
    for kind in ENCOURS_KINDS:
        dest = DATA / f"aft_encours_{kind}.html"
        if dest.exists() and not refresh:
            pass
        else:
            try:
                payload = get_bytes(
                    AFT_ENCOURS.format(kind=kind), timeout=60, impersonate=True, min_bytes=2000
                )
                if not _aft_encours_ok(payload):
                    raise RuntimeError(f"AFT encours {kind}: Cloudflare challenge or empty table")
                dest.write_bytes(payload)
            except Exception as e:  # noqa: BLE001 — keep cache on AFT/Cloudflare flap
                if dest.exists() and _aft_encours_ok(dest.read_bytes()):
                    print(f"FR AFT encours {kind}: fetch failed ({e}); keeping cache")
                else:
                    print(f"FR AFT encours {kind}: fetch failed ({e}); no usable cache, skipping kind")
                    continue
        if not dest.exists() or not _aft_encours_ok(dest.read_bytes()):
            continue
        tabs = pd.read_html(io.StringIO(dest.read_text(encoding="utf-8")), flavor="lxml")
        t = pd.concat([x for x in tabs if "Code ISIN" in x.columns and len(x)], ignore_index=True)
        t = t.drop_duplicates("Code ISIN")
        for _, r in t.iterrows():
            lab = str(r["Libellé"])
            mat = pd.to_datetime(re.search(r"(\d{1,2} \w+ \d{4})\s*$", lab).group(1), format="%d %B %Y")
            rows.append({"isin": r["Code ISIN"], "label": lab, "kind": kind, "maturity": mat,
                         "nominal_bn": float(r["Encours (€)"]) / 1e9})
    if not rows:
        raise RuntimeError("AFT encours: no kinds available (fetch failed and no usable cache)")
    return pd.DataFrame(rows).sort_values("maturity").reset_index(drop=True)


def fetch_coef_file(kind: str) -> Path:
    """Download the latest AFT daily index-coefficient file for ``kind`` into data/fr/ (network)."""
    from issuance.http import get_bytes

    page, pat, _ = COEF_PAGES[kind]
    html = get_bytes(AFT + page, timeout=60, impersonate=True, min_bytes=2000).decode("utf-8", "ignore")
    links = sorted(set(re.findall(rf'href="([^"]*\d{{4}}-\d{{2}}_{pat}[^"]*\.xlsx?)"', html)))
    if not links:
        raise RuntimeError(f"AFT {kind}: no coefficient file linked on {page}")
    link = links[-1]
    dest = DATA / link.rsplit("/", 1)[-1]
    if not dest.exists():
        dest.write_bytes(get_bytes(AFT + link, timeout=60, impersonate=True, min_bytes=2000))
    return dest


def coef_file(kind: str) -> Path:
    """Latest coefficient file on disk (data/fr/ or the raw AFT tree), by its YYYY-MM prefix."""
    _, pat, raw = COEF_PAGES[kind]
    files = [f for d in (DATA, raw) for f in d.glob(f"*_{pat}*.xls*") if re.match(r"\d{4}-\d{2}_", f.name)]
    if not files:
        raise FileNotFoundError(f"no {pat} file")
    return max(files, key=lambda f: (f.name[:7], f.stat().st_mtime))


def oatei_coef(maturity: pd.Timestamp, kind: str = "oatei") -> tuple[float, str]:
    """Index ratio of a linker at maturity: last published daily coefficient, carried forward at the
    trailing-12m growth of the daily inflation reference; floored at 1 (par floor)."""
    f = coef_file(kind)
    x = pd.read_excel(f, header=None)
    mats = pd.to_datetime(x.iloc[3, 2:], errors="coerce")
    col = [c for c, m in zip(x.columns[2:], mats) if pd.notna(m) and m.normalize() == maturity.normalize()]
    body = x.iloc[6:].copy()
    body[0] = pd.to_datetime(body[0], errors="coerce")
    body = body.dropna(subset=[0]).set_index(0)
    if not col:
        return 1.0, f"{f.name}: no column for {maturity.date()}, face used"
    s = pd.to_numeric(body[col[0]], errors="coerce").dropna()
    ref = pd.to_numeric(body[1], errors="coerce").dropna()
    d0 = s.index.max()
    g = float(ref.loc[d0] / ref.loc[:d0 - pd.DateOffset(years=1)].iloc[-1])
    c = float(s.loc[d0]) * g ** ((maturity - d0).days / 365.25)
    return max(c, 1.0), f"{f.name}: {float(s.loc[d0]):.5f} at {d0.date()} × {g:.4f}/yr to {maturity.date()}"


def coef_asof(kind: str = "oatei") -> pd.Timestamp:
    """Last date carried by the latest coefficient file."""
    x = pd.read_excel(coef_file(kind), header=None, usecols=[0])
    return pd.to_datetime(x.iloc[6:, 0], errors="coerce").max()


def load_mlt_auctions() -> pd.DataFrame:
    """AFT OAT auction history (latest ``*_hist_mlt.xlsx``): total issued incl. non-competitive (ONC), EUR mn."""
    files = sorted(RAW_AFT.joinpath("OAT").glob("*_hist_mlt.xlsx"))
    if not files:
        return pd.DataFrame(columns=["isin", "op_dt", "total_mn", "coef"])
    x = pd.read_excel(files[-1], header=0)
    x = x.iloc[:, [1, 5, 11, 14]]
    x.columns = ["op_dt", "isin", "total_mn", "coef"]
    x["op_dt"] = pd.to_datetime(x["op_dt"], errors="coerce")
    x["total_mn"] = pd.to_numeric(x["total_mn"], errors="coerce")
    x["coef"] = pd.to_numeric(x["coef"], errors="coerce").fillna(1.0)
    x = x.dropna(subset=["op_dt", "isin", "total_mn"])
    x.attrs["file"] = files[-1].name
    return x


def load_json(name: str) -> dict | None:
    import json

    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


if __name__ == "__main__":
    print(load_smb(refresh=True).tail(3))
    print(load_debt(refresh=True).tail(3))
