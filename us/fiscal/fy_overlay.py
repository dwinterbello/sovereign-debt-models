"""Fiscal-year level for the DTS path, then the gradient-boosted calendar shapes the year.

The scored daily model is unchanged: trailing 250-day level times the boosted shape.
The live multi-year path replaces that level with a fiscal-year profile.

Indexed lines take the statutory COLA. The share is the line's weight in its DTS
bucket over the last complete fiscal year, so a mixed bucket is not all indexed:

  SSA benefits and SSI                         full CPI-W COLA (January; SSI pays 31 Dec)
  VA benefits                                  full COLA (compensation and pension)
  Military retirement                          full COLA
  Civil-service retirement                     half full COLA, half the FERS diet COLA
                                               (the DTS line does not split CSRS from FERS)

The rest of the bucket, and every bucket with no indexed line, steps once a year by its
trailing 250-day growth. Years past the last announced COLA repeat that COLA.
The boosted shape is renormalised inside each fiscal year so the year's mean equals
this level.
"""

from __future__ import annotations

import pandas as pd

# January of the calendar year the COLA takes effect. 2027 announced 14 Oct 2026.
COLA_PCT: dict[int, float] = {
    2015: 1.7, 2016: 0.0, 2017: 0.3, 2018: 2.0, 2019: 2.8, 2020: 1.6, 2021: 1.3,
    2022: 5.9, 2023: 8.7, 2024: 3.2, 2025: 2.5, 2026: 2.8, 2027: 3.6,
}
# category -> full CPI-W, or the CSRS/FERS blend
INDEXED = {
    "SSA - Benefits Payments": "full",
    "SSA - Supplemental Security Income": "full",
    "VA - Benefits": "full",
    "DoD - Military Retirement": "full",
    "OPM - Civil Serv Retirement & Disability": "blend",
}
VOLUME_CAP = 0.03
ANCHOR_YEAR = 2026  # January index = 1


def cola_pct(year: int) -> float:
    if year in COLA_PCT:
        return COLA_PCT[year]
    last = max(COLA_PCT)
    return COLA_PCT[last] if year > last else COLA_PCT[min(COLA_PCT)]


def fers_diet(cola: float) -> float:
    """FERS annuity COLA: full below 2%, 2% from 2 to 3, CPI-W minus 1pp above 3."""
    if cola >= 3.0:
        return cola - 1.0
    if cola >= 2.0:
        return 2.0
    return cola


def january_index(year: int) -> float:
    """Benefit index in January, January of ANCHOR_YEAR = 1."""
    if year >= ANCHOR_YEAR:
        x = 1.0
        for y in range(ANCHOR_YEAR + 1, year + 1):
            x *= 1.0 + cola_pct(y) / 100.0
        return x
    x = 1.0
    for y in range(year + 1, ANCHOR_YEAR + 1):
        x /= 1.0 + cola_pct(y) / 100.0
    return x


def fy_of(ts: pd.Timestamp) -> int:
    """Fiscal year ending 30 September."""
    return int(ts.year + 1 if ts.month >= 10 else ts.year)


def _fy_mean_index(fy: int, kind: str) -> float:
    # Oct-Dec sit on the prior January's COLA; Jan-Sep on this calendar year's.
    prev, cur = january_index(fy - 1), january_index(fy)
    if kind == "blend":
        # half CSRS (full), half FERS (diet), on the step from the anchor
        def _blend(year: int) -> float:
            if year <= ANCHOR_YEAR:
                return january_index(year)
            full = 1.0
            diet = 1.0
            for y in range(ANCHOR_YEAR + 1, year + 1):
                c = cola_pct(y)
                full *= 1.0 + c / 100.0
                diet *= 1.0 + fers_diet(c) / 100.0
            return 0.5 * full + 0.5 * diet
        prev, cur = _blend(fy - 1), _blend(fy)
    return (3.0 * prev + 9.0 * cur) / 12.0


def cola_ratio(fy: int, base_fy: int, kind: str) -> float:
    b = _fy_mean_index(base_fy, kind)
    return _fy_mean_index(fy, kind) / b if b else 1.0


def cola_shares(asof: pd.Timestamp) -> dict[str, dict[str, float]]:
    """Indexed share of each DTS bucket in the last complete fiscal year."""
    import dts_flows

    raw = pd.read_parquet(dts_flows.RAW_PATH)
    raw = raw.copy()
    raw["record_date"] = pd.to_datetime(raw["record_date"])
    last = asof.year if asof.month >= 10 else asof.year - 1
    start, end = pd.Timestamp(last - 1, 10, 1), pd.Timestamp(last, 9, 30)
    w = raw[(raw["record_date"] >= start) & (raw["record_date"] <= end)]
    w = w[w["transaction_type"] == "Withdrawals"]
    g = w.groupby("transaction_catg")["amt_mn"].sum()
    out: dict[str, dict[str, float]] = {}
    for cat, rule in INDEXED.items():
        if cat not in g.index:
            continue
        bucket = dts_flows.classify(cat, "Withdrawals")
        slot = out.setdefault(bucket, {"full": 0.0, "blend": 0.0, "total": 0.0})
        slot[rule] += float(g[cat])
    totals = w.groupby(w["transaction_catg"].map(lambda c: dts_flows.classify(c, "Withdrawals")))["amt_mn"].sum()
    shares = {}
    for bucket, slot in out.items():
        tot = float(totals.get(bucket, 0.0))
        if tot <= 0:
            continue
        shares[bucket] = {"full": slot["full"] / tot, "blend": slot["blend"] / tot}
    return shares


def _bucket_growth(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Trailing 250-day growth, clipped. Same clip as the scored long path."""
    hist = panel[panel.index <= origin]
    now = hist.tail(250).mean()
    ago = hist[hist.index <= origin - pd.DateOffset(years=1)].tail(250).mean()
    return (now / ago.where(ago.abs() > 1e-6) - 1.0).fillna(0.0).clip(-0.15, 0.25)


def volume_drift(panel: pd.DataFrame, bucket: str, origin: pd.Timestamp) -> float:
    """Beneficiary drift: median fiscal-year growth minus the COLA already in that year, capped."""
    y = panel[bucket]
    fy = y.index.year + (y.index.month >= 10).astype(int)
    sums = y.groupby(fy).sum()
    last = origin.year if origin.month >= 10 else origin.year - 1
    excess = []
    for fy_end in range(last - 2, last + 1):
        if fy_end not in sums.index or fy_end - 1 not in sums.index or sums[fy_end - 1] == 0:
            continue
        yoy = sums[fy_end] / sums[fy_end - 1] - 1.0
        embedded = cola_ratio(fy_end, fy_end - 1, "full") - 1.0
        excess.append(yoy - embedded)
    if not excess:
        return 0.0
    return float(min(VOLUME_CAP, max(0.0, pd.Series(excess).median())))


def apply_fy(fc: pd.DataFrame, panel: pd.DataFrame, origin: pd.Timestamp,
             shares: dict[str, dict[str, float]] | None = None,
             growth: pd.Series | None = None) -> pd.DataFrame:
    """Rescale a trailing-level forecast so each fiscal year hits the overlaid level."""
    if fc.empty:
        return fc
    shares = cola_shares(origin) if shares is None else shares
    g = _bucket_growth(panel, origin) if growth is None else growth
    mid = origin - pd.tseries.offsets.BDay(125)
    base = fy_of(pd.Timestamp(mid))
    out = fc.copy()
    fys = pd.Series([fy_of(pd.Timestamp(t)) for t in fc.index], index=fc.index)
    for b in fc.columns:
        if b == "wd_interest" or b not in panel.columns:
            continue
        sh = shares.get(b, {"full": 0.0, "blend": 0.0})
        vol = volume_drift(panel, b, origin) if sh["full"] + sh["blend"] else 0.0
        gg = float(g.get(b, 0.0))
        rest = 1.0 - sh["full"] - sh["blend"]
        lvl = float(panel.loc[panel.index <= origin, b].tail(250).mean())
        for fy, idx in fys.groupby(fys).groups.items():
            n = int(fy) - base
            nom = (1.0 + gg) ** n
            full = (1.0 + vol) ** n * cola_ratio(int(fy), base, "full")
            blend = (1.0 + vol) ** n * cola_ratio(int(fy), base, "blend")
            factor = rest * nom + sh["full"] * full + sh["blend"] * blend
            block = out.loc[idx, b]
            mean = float(block.mean())
            if mean == 0.0 or not pd.notna(lvl):
                continue
            out.loc[idx, b] = block / mean * lvl * factor
    return out
