"""Japan forward supply: MoF FY2026 JGB issuance plan on the MoF auction calendar.

Sizes are the plan's per-auction offering (calendar base, FY2026 initial of 26 Dec 2025; the
3 Jun 2026 supplementary revision moved only the fiscal-year adjustment, not auction sizes),
plus each product's trailing-year Non-Price Competitive II take, which the plan books on top.
Dates are the MoF calendar where posted (about three months ahead), else the same product's
auction a year earlier, same day of the month (the business-day roll keeps it in its month). FY2027 (from Apr 2027) has no plan yet and repeats
FY2026 sizes. Lines: a product opens a new issue in the calendar months it did over the last two
years, otherwise it reopens its current issue. Liquidity enhancement amounts are MoF's quarterly
release per residual-maturity zone (the plan's ¥13.5tn FY figure is not binding: Oct–Dec 2026 was
cut to ¥2.7tn from ¥3.6tn), repeated past the release; each zone is spread over live issues in it.

T-Discount Bills (`bill_rows`): last year's auction days at trailing sizes, not the plan (see there).
Plan figures are MoF fiscal years (Apr–Mar); monthly panels summed by calendar year do not match them.

Not modelled: floating-rate JGBs (6m T-bill + auction spread, 2y first, from Jan 2027 at the
earliest). MoF has set no size or date; terms come about two months before the first issue.
"""

from __future__ import annotations

import pandas as pd

from issuance.tenor import bucket_from_years

PLAN_ASOF = pd.Timestamp("2025-12-26")
PLAN_MN = {
    "2y": 2_800_000.0, "5y": 2_500_000.0, "10y": 2_600_000.0, "20y": 700_000.0,
    "30y": 600_000.0, "40y": 300_000.0, "10y IL": 250_000.0, "GX10y": 250_000.0, "GX5y": 250_000.0,
}
PLAN_LEA_FY_MN = 13_500_000.0
PLAN_FY = 2026
# Products the plan does not auction monthly; the year-earlier tape can sit in the wrong month
# (FY2025 sold a GX5y in January, FY2026 sells them in May and November only).
PLAN_MONTHS = {"40y": {5, 7, 9, 11, 1, 3}, "10y IL": {5, 8, 11, 2}, "GX10y": {8, 2}, "GX5y": {5, 11}}
YEARS = {"2y": 2, "5y": 5, "10y": 10, "20y": 20, "30y": 30, "40y": 40, "10y IL": 10, "GX10y": 10, "GX5y": 5}
ISIN_STEM = {"10y IL": "JPIL10y", "GX10y": "JPGX10y", "GX5y": "JPGX5y"}
COUPON = tuple(PLAN_MN)


def new_maturity(product: str, day: pd.Timestamp) -> pd.Timestamp:
    """2y JGBs mature on the 1st of the month two years on; the rest on the 20th of the last
    Mar/Jun/Sep/Dec before the full term (a 10y issued in Oct 2026 matures 20 Sep 2036)."""
    end = pd.Timestamp(day) + pd.DateOffset(years=YEARS[product])
    if product == "2y":
        return end.replace(day=1)
    if product == "40y":
        return pd.Timestamp(year=end.year if end.month > 3 else end.year - 1, month=3, day=20)
    m = end.month - 1
    q = m - m % 3 or 12
    year = end.year if m >= 3 else end.year - 1
    return pd.Timestamp(year=year, month=q, day=20)


def _tape(ops: pd.DataFrame) -> pd.DataFrame:
    o = ops[ops["country"].astype(str).eq("JP")].copy() if "country" in ops.columns else ops.copy()
    o = o[(o["amount_issued"] > 0) & o["op_dt"].notna()]
    o["op_dt"] = pd.to_datetime(o["op_dt"])
    return o.sort_values("op_dt")


def npc2_share(tape: pd.DataFrame, asof: pd.Timestamp) -> dict[str, float]:
    a = tape[(tape["op_type"].astype(str) == "Auction") & (tape["op_dt"] > asof - pd.DateOffset(years=1))]
    out = {}
    for p, g in a.groupby("product"):
        npc2 = pd.to_numeric(g.get("npc2_mn"), errors="coerce").fillna(0.0).sum()
        base = (g["amount_issued"].sum() - npc2)
        out[str(p)] = float(npc2 / base) if base > 0 else 0.0
    return out


def new_issue_months(tape: pd.DataFrame, asof: pd.Timestamp) -> dict[str, set[int]]:
    a = tape[tape["op_type"].astype(str) == "Auction"]
    first = a.groupby("isin").agg(product=("product", "first"), first=("op_dt", "min"))
    first = first[first["first"] > asof - pd.DateOffset(years=2)]
    return {str(p): set(g["first"].dt.month) for p, g in first.groupby("product")}


def _calendar(asof: pd.Timestamp) -> pd.DataFrame:
    from issuance.jp_calendar import load

    cal = load()
    return cal[cal["date"] > asof]


def _schedule(tape: pd.DataFrame, asof: pd.Timestamp, horizon: pd.Timestamp) -> pd.DataFrame:
    """Every forward auction (date, product, source); calendar months replace the shifted tape."""
    cal = _calendar(asof)
    cal_months = set(cal["date"].dt.to_period("M"))
    cal = cal[~cal["product"].astype(str).str.startswith("LEA")]
    # MoF drops past dates from its calendar, so the first month it lists is partial: there the
    # calendar replaces the shifted tape only for the products it names.
    first = cal["date"].min().to_period("M") if len(cal) else None
    cal_first = set(cal.loc[cal["date"].dt.to_period("M") == first, "product"].astype(str))
    hist = tape[tape["op_type"].astype(str) == "Auction"][["op_dt", "product"]].drop_duplicates()
    shifted = hist.copy()
    shifted["date"] = shifted["op_dt"] + pd.DateOffset(years=1)
    shifted = shifted[(shifted["date"] > asof) & (shifted["date"] <= horizon)]
    sm = shifted["date"].dt.to_period("M")
    partial = (sm == first) & ~shifted["product"].astype(str).isin(cal_first)
    shifted = shifted[~sm.isin(cal_months) | partial]
    off_plan = [p in PLAN_MONTHS and d.month not in PLAN_MONTHS[p] for p, d in zip(shifted["product"], shifted["date"])]
    shifted = shifted[[not x for x in off_plan]]
    rows = [(r.date, r.product, "calendar") for r in cal.itertuples() if r.date <= horizon]
    rows += [(r.date, r.product, "plan") for r in shifted.itertuples()]
    have = {(p, d.to_period("M")) for d, p, _ in rows}
    for p, months in PLAN_MONTHS.items():
        days = hist.loc[hist["product"].astype(str).eq(p), "op_dt"].dt.day
        dom = int(days.median()) if len(days) else 15
        for m in pd.period_range(asof.to_period("M") + 1, horizon.to_period("M"), freq="M"):
            if m.month in months and m not in cal_months and (p, m) not in have:
                day = pd.Timestamp(year=m.year, month=m.month, day=min(dom, m.days_in_month))
                rows.append((day + pd.offsets.BDay(0), p, "plan"))
    out = pd.DataFrame(rows, columns=["date", "product", "source"])
    return out.sort_values("date").reset_index(drop=True)


def lea_events(asof: pd.Timestamp, horizon: pd.Timestamp) -> pd.DataFrame:
    """(date, zone, amount_mn, basis) per forward LEA auction. Amounts are MoF's quarterly
    release; months past it repeat the latest announced quarter month for month."""
    from issuance.jp_calendar import load, load_lea

    ann = {pd.Period(m, "M"): z for m, z in load_lea().items()}
    if not ann:
        return pd.DataFrame(columns=["date", "zone", "amount_mn", "basis"])
    q0 = max(ann) - 2
    cal = load()
    cal = cal[cal["product"].astype(str).str.startswith("LEA")].copy()
    cal["zone"] = cal["product"].str.extract(r"LEA ([\d.]+-[\d.]+)y")[0]
    cal["m"] = cal["date"].dt.to_period("M")
    dom = cal.groupby("zone")["date"].apply(lambda d: int(d.dt.day.median())).to_dict()
    rows = []
    for m in pd.period_range(asof.to_period("M"), horizon.to_period("M"), freq="M"):
        zones, basis = ann.get(m), "announced"
        if zones is None:
            if m < q0:
                continue
            zones, basis = ann.get(q0 + (m - q0).n % 3, {}), "repeat"
        for zone, amt in zones.items():
            if amt <= 0:
                continue
            hit = cal.loc[cal["m"].eq(m) & cal["zone"].eq(zone), "date"]
            if len(hit):
                day = hit.iloc[0]
            else:
                d = dom.get(zone, 22)
                day = pd.Timestamp(year=m.year, month=m.month, day=min(d, m.days_in_month)) + pd.offsets.BDay(0)
            if day > asof and day <= horizon:
                rows.append((day, zone, float(amt), basis))
    return pd.DataFrame(rows, columns=["date", "zone", "amount_mn", "basis"]).sort_values("date")


def _lea_rows(tape: pd.DataFrame, day: pd.Timestamp, zone: str, amount: float) -> list[dict]:
    """Spread one zone's amount over live 2y–40y issues in that residual range, weighted by what
    each took in liquidity auctions in the year to the tape's end; issues with no take (new, or
    just rolled into the zone) get a quarter of the median take."""
    lo, hi = (float(x) for x in zone.split("-"))
    end = tape["op_dt"].max()
    a = tape[(tape["op_type"].astype(str) == "Auction") & tape["product"].astype(str).isin(["2y", "5y", "10y", "20y", "30y", "40y"])]
    w = a.groupby("isin").agg(mat=("maturity_dt", "max"))
    lea = tape[(tape["op_type"].astype(str) == "Liquidity Enhancement") & (tape["op_dt"] > end - pd.DateOffset(years=1))]
    took = lea.groupby("isin")["amount_issued"].sum()
    w["w"] = took.reindex(w.index).fillna(0.0)
    w["w"] = w["w"].where(w["w"] > 0, 0.25 * took.median() if len(took) else 1.0)
    w = w[_in_zone(w, day, lo, hi)]
    if w.empty:
        return []
    w["w"] = w["w"] / w["w"].sum()
    return [{"isin": isin, "tenor_bucket": bucket_from_years((pd.Timestamp(r.mat) - day).days / 365.25),
             "op_dt": day, "amount_mn": amount * r.w, "source": "lea", "sec_type": "Nominal",
             "maturity_dt": pd.Timestamp(r.mat), "op_type": "Liquidity Enhancement"} for isin, r in w.iterrows()]


def _in_zone(w: pd.DataFrame, day: pd.Timestamp, lo: float, hi: float) -> pd.Series:
    yrs = (pd.to_datetime(w["mat"]) - day).dt.days / 365.25
    return (yrs >= max(lo, 1.0)) & (yrs < hi)


def jp_rows(ops: pd.DataFrame, asof: pd.Timestamp, horizon: pd.Timestamp) -> list[dict]:
    tape = _tape(ops)
    if tape.empty:
        return []
    npc2 = npc2_share(tape, asof)
    newm = new_issue_months(tape, asof)
    auctions = tape[tape["op_type"].astype(str) == "Auction"]
    current = {p: g.iloc[-1] for p, g in auctions.groupby("product")}
    live: dict[str, dict] = {}
    for p, r in current.items():
        live[str(p)] = {"isin": r["isin"], "maturity_dt": pd.Timestamp(r["maturity_dt"]),
                        "coupon": r["coupon"], "yield": r["yield"]}
    rows = []
    for e in lea_events(asof, horizon).itertuples():
        rows += _lea_rows(tape, pd.Timestamp(e.date), e.zone, e.amount_mn)
    for s in _schedule(tape, asof, horizon).itertuples():
        day = pd.Timestamp(s.date)
        p = s.product
        if p not in PLAN_MN:
            continue
        cur = live.get(p)
        if cur is None or day.month in newm.get(p, set()) and day > asof:
            mat = new_maturity(p, day)
            cur = {"isin": f"next {ISIN_STEM.get(p, 'JP' + p)} {mat:%y-%m}", "maturity_dt": mat,
                   "coupon": cur["yield"] if cur else None, "yield": cur["yield"] if cur else None}
            live[p] = cur
        yrs = (cur["maturity_dt"] - day).days / 365.25
        rows.append({
            "isin": cur["isin"], "tenor_bucket": bucket_from_years(yrs), "op_dt": day,
            "amount_mn": PLAN_MN[p] * (1.0 + npc2.get(p, 0.0)), "source": s.source,
            "sec_type": "Linker" if p == "10y IL" else "Nominal", "maturity_dt": cur["maturity_dt"],
            "coupon_hint": cur["coupon"], "yield_hint": cur["yield"], "op_type": "Auction",
        })
    return sorted(rows, key=lambda r: r["op_dt"])


BILL_SIZE_N = 3


def bill_rows(ops: pd.DataFrame, asof: pd.Timestamp, horizon: pd.Timestamp) -> pd.DataFrame:
    """Forward T-Discount Bill auctions: each tenor's auction days in the 52 weeks to `asof`,
    moved on 52 weeks at a time (keeps the weekday and the weekly 3m cadence), each sized at the
    mean of that tenor's last BILL_SIZE_N auctions. Tenors not auctioned in the last year drop.
    The plan's TB line (1y + 6m) cannot size these: 1y bills are TB and FB jointly and the 3m
    bills are FB, which no plan publishes; MoF announces each size a week ahead."""
    cols = ["op_dt", "product", "amount_mn", "resid_mat_yrs", "yield", "price", "source"]
    tape = _tape(ops)
    b = tape[tape["sec_type"].astype(str).eq("Bill") & (tape["op_dt"] <= asof)]
    if b.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for p, g in b.groupby(b["product"].astype(str)):
        g = g.sort_values("op_dt")
        days = g.loc[g["op_dt"] > asof - pd.Timedelta(weeks=52), "op_dt"]
        if days.empty:
            continue
        last = g.tail(BILL_SIZE_N)
        size = float(last["amount_issued"].mean())
        yrs = float(pd.to_numeric(last["resid_mat_yrs"], errors="coerce").mean())
        y = float(pd.to_numeric(last["yield"], errors="coerce").iloc[-1])
        px = 100.0 / (1.0 + y / 100.0 * yrs)
        k = 1
        while True:
            fwd = days + pd.Timedelta(weeks=52 * k)
            fwd = fwd[(fwd > asof) & (fwd <= horizon)]
            if fwd.empty and (days + pd.Timedelta(weeks=52 * k)).min() > horizon:
                break
            rows += [(d, p, size, yrs, y, px, "trailing") for d in fwd]
            k += 1
    return pd.DataFrame(rows, columns=cols).sort_values("op_dt").reset_index(drop=True)


def fy_totals(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["fy"] = d["op_dt"].map(lambda t: t.year if t.month >= 4 else t.year - 1)
    return d.groupby(["fy", "op_type"])["amount_mn"].sum().unstack().div(1e6)
