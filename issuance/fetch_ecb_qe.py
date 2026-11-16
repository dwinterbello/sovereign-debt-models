"""ECB PSPP / PEPP country holdings and WAM (the public WAMS constraint)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from issuance.http import get_bytes
from issuance.paths import CACHE, RAW

ECB = "https://www.ecb.europa.eu"
FILES = {
    "pspp_breakdown": "/mopo/pdf/PSPP_breakdown_history.csv",
    "pepp_public": "/mopo/pdf/PEPP_public_sector_securities_breakdown_history.csv",
}

ISO2 = {
    "austria": "AT", "belgium": "BE", "cyprus": "CY", "germany": "DE",
    "estonia": "EE", "spain": "ES", "finland": "FI", "france": "FR",
    "ireland": "IE", "italy": "IT", "lithuania": "LT", "luxembourg": "LU",
    "latvia": "LV", "malta": "MT", "netherlands": "NL", "the netherlands": "NL",
    "portugal": "PT", "slovenia": "SI", "slovakia": "SK", "greece": "GR",
    "supranationals": "SUPRA", "supranational": "SUPRA", "total": "TOTAL",
}


def _iso(name: object) -> str:
    text = str(name).strip()
    if text.upper() in ISO2.values():
        return text.upper()
    return ISO2.get(text.lower(), "")


def _section_frames(text: str) -> list[tuple[str, pd.DataFrame]]:
    """Split an ECB titled CSV into (title, wide-dataframe) pairs."""
    lines = text.splitlines()
    sections: list[tuple[str, pd.DataFrame]] = []
    i = 0
    while i < len(lines):
        first = lines[i].split(",")[0].strip()
        if not first or first.lower() in ISO2 or first[:1].isdigit():
            i += 1
            continue
        title = first
        j = i + 1
        while j < len(lines):
            cell0 = lines[j].split(",")[0].strip()
            has_date = any(ch.isdigit() for ch in lines[j][:24])
            if cell0 and not has_date and not _iso(cell0) and cell0.lower() not in ISO2:
                title = cell0
                j += 1
                continue
            break
        while j < len(lines) and lines[j].count(",") < 5:
            j += 1
        if j >= len(lines):
            break
        header = lines[j]
        rows = [header]
        k = j + 1
        while k < len(lines):
            cell = lines[k].split(",")[0].strip()
            if not cell:
                k += 1
                continue
            if cell.lower() in ISO2 or _iso(cell):
                rows.append(lines[k])
                k += 1
                continue
            break
        if len(rows) > 2:
            df = pd.read_csv(io.StringIO("\n".join(rows)))
            sections.append((title, df))
        i = k
    return sections


def _num(val) -> float:
    """ECB PSPP cells use quoted thousands commas: '-9,418' / '435,323'."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    if isinstance(val, str):
        text = val.strip().replace("\u00a0", "").replace(" ", "").replace(",", "")
        if not text or text in {"-", "n/a", "na", "nan"}:
            return float("nan")
        val = text
    return float(pd.to_numeric(val, errors="coerce"))


def _is_cum_col(col) -> bool:
    return "cumul" in str(col).lower()


def _melt_safe(df: pd.DataFrame, program: str, metric: str) -> pd.DataFrame:
    country_col = df.columns[0]
    date_cols = list(df.columns[1:])
    records = []
    published = []
    for _, row in df.iterrows():
        country = _iso(row[country_col])
        if not country:
            continue
        for col in date_cols:
            if str(col).lower().startswith("unnamed"):
                continue
            if _is_cum_col(col):
                published.append({
                    "program": program,
                    "country": country,
                    "published_stock_mn": _num(row[col]),
                })
                continue
            label = str(col).strip()
            dt = pd.NaT
            if len(label) == 6 and label[3] == "-":
                dt = pd.to_datetime(label, format="%b-%y", errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(label, errors="coerce", dayfirst=True)
            if pd.isna(dt):
                continue
            if getattr(dt, "year", 2000) < 1990:
                continue
            records.append({
                "program": program,
                "country": country,
                "date": dt,
                "value": _num(row[col]),
                "metric": metric,
            })
    out = pd.DataFrame.from_records(records)
    out.attrs["published"] = pd.DataFrame.from_records(published)
    return out


def _classify(title: str) -> str | None:
    t = title.lower()
    if "eligible" in t:
        return None
    if "wam" in t:
        return "wam"
    if "net purchase" in t or "monthly" in t:
        return "net"
    if "cumulative" in t or "holding" in t:
        return "stock"
    return None


def _align_published_stock(net: pd.DataFrame, max_gap: float = 50.0) -> pd.DataFrame:
    """Move the last stock onto the published cumulative when the gap is dust.

    A multi-billion hole is missing history, not a comma. Leave that alone
    and keep the WARN. Gaps of a few million are the thousands-comma residue.
    """
    if net.empty or "published_stock_mn" not in net.columns:
        return net
    out = net.sort_values(["program", "country", "date"]).copy()
    last_idx = out.groupby(["program", "country"], sort=False).tail(1).index
    pub_v = pd.to_numeric(out.loc[last_idx, "published_stock_mn"], errors="coerce")
    hold_v = pd.to_numeric(out.loc[last_idx, "holdings_mn"], errors="coerce")
    delta = pub_v - hold_v
    mask = delta.notna() & (delta.abs() > 1e-6) & (delta.abs() <= max_gap)
    if not mask.any():
        return out
    out.loc[delta.index[mask], "holdings_mn"] = pub_v[mask]
    out.loc[delta.index[mask], "value"] = (
        pd.to_numeric(out.loc[delta.index[mask], "value"], errors="coerce") + delta[mask]
    )
    print(f"  aligned {int(mask.sum())} last stocks to published cumulative (|gap|<={max_gap:.0f})")
    return out


def fetch_ecb_qe(force: bool = False) -> pd.DataFrame:
    parts = []
    for key, rel in FILES.items():
        dest = RAW / "ecb" / f"{key}.csv"
        if force or not dest.exists():
            get_bytes(ECB + rel, dest=dest, timeout=120)
        text = dest.read_text(encoding="utf-8", errors="replace")
        program = "PSPP" if "pspp" in key else "PEPP"
        for title, wide in _section_frames(text):
            kind = _classify(title)
            if kind is None:
                continue
            melted = _melt_safe(wide, program, kind)
            parts.append(melted)
            print(f"  {program} {kind}: {len(melted)} cells from '{title[:48]}'")

    published_parts = []
    clean = []
    for p in parts:
        pub = p.attrs.pop("published", None)
        if pub is not None and not getattr(pub, "empty", True):
            published_parts.append(pub)
        p.attrs.clear()
        clean.append(p)
    parts = clean
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if panel.empty:
        return panel

    net = panel[panel["metric"] == "net"].copy()
    wam = panel[panel["metric"] == "wam"].copy()
    stock = panel[panel["metric"] == "stock"].copy()

    if not net.empty:
        net = net.dropna(subset=["date", "value"]).sort_values(["program", "country", "date"])
        net["holdings_mn"] = net.groupby(["program", "country"])["value"].cumsum()
        net = net.dropna(subset=["holdings_mn"])
    elif not stock.empty:
        net = stock.rename(columns={"value": "holdings_mn"})
    if net is None or net.empty:
        return panel

    net["month"] = pd.to_datetime(net["date"]).dt.to_period("M").dt.to_timestamp()
    if not wam.empty:
        wam = wam.rename(columns={"value": "wam_yrs"})
        wam["month"] = pd.to_datetime(wam["date"]).dt.to_period("M").dt.to_timestamp()
        wam_m = wam.dropna(subset=["wam_yrs"]).groupby(
            ["program", "country", "month"], as_index=False
        )["wam_yrs"].last()
        net = net.merge(wam_m, on=["program", "country", "month"], how="left")
        net = net.sort_values(["program", "country", "month"])
        net["wam_yrs"] = net.groupby(["program", "country"])["wam_yrs"].ffill()
    elif "wam_yrs" not in net.columns:
        net["wam_yrs"] = pd.NA

    if published_parts:
        pub = pd.concat(published_parts, ignore_index=True)
        pub = pub.dropna(subset=["published_stock_mn"])
        last = net.sort_values("date").groupby(["program", "country"], as_index=False).tail(1)
        chk = last.merge(pub, on=["program", "country"], how="left")
        bad = chk[chk["published_stock_mn"].notna()
                  & ((chk["holdings_mn"] - chk["published_stock_mn"]).abs() > 2.0)]
        for rec in bad.itertuples(index=False):
            print(f"  WARN {rec.program} {rec.country}: cumsum {rec.holdings_mn:.0f} "
                  f"vs published {rec.published_stock_mn:.0f}")
        net = net.merge(pub, on=["program", "country"], how="left")
        net = _align_published_stock(net)

    net.to_parquet(CACHE / "ecb_qe_holdings.parquet", index=False)
    return net
