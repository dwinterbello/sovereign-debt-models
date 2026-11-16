"""Forward issuance from the published programme, then line patterns past it.

Windows that have a number:
  UK  DMO D5D dates and gilts, the minutes' syndications and tenders. FY26-27 is
      sized in cash to the DMO remit report by sector and method (auctions,
      syndications, unallocated tenders), exactly. Jan–Mar auction dates go to
      the sectors that still have auctions left. Apr–Sep 2027 repeats Apr–Sep
      2026 cash by sector and method.
  DE  the 24 Sep 2026 Finanzagentur Q4 calendar, ISIN and size as printed.
      The second bond of a multi-ISIN auction is unnamed until the week before.
  US  each month equals the Aug 2026 refunding table, then the TBAC table, then
      that same coupon grid. Bills stay out.
  FR  AFT next-issuance communiqués (ISIN, volume range, auction day) displace the
      programme pattern on those auction days; otherwise meets the €310bn (2026) /
      €340bn (2027) net programmes exactly, booked gross at this year's gross-to-net
      ratio, in last year's monthly shape.
  IT  BdI offering announcements (ISIN, size range, auction day) displace the programme
      pattern for those lines; past the announced window continues the Q4 programme mix.
      The H2 guidance is not spent by reopening the 2025 book.
  ES  BdE BON PDF lines (ISIN + resolución day) and the Tesoro annual calendar dates
      mark announced auctions; sizes from the line's last tap when BdE names the ISIN.
      Otherwise: lines tapped in the last six months, trailing 12-month coupon pace,
      first/third-Thursday days. Letras stay in the cash framework, not this coupon paint.

Past those windows the tenor mix of the programme is rolled forward on open lines.
That stretch is labeled pattern. Annex D of the UK remit is not a gilt forecast
and is not used.

US rows are synthetic (there is no CUSIP yet). They are not priced as a 3% par
bond, and they are not given the round tenor. A new 10y/30y is issued in
February, May, August and November and clears 10y / 30y by a few days; the
reopenings come back under the boundary (about 9.93y and 29.93y). The 20y
stays under 20y in both months. Coupon and yield are the latest auction of
that product. TIPS stay under the same boundaries, at the real yield.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.tenor import bucket_from_years
from issuance.uk_offers import UK_ANNOUNCED

UK_REMIT_MN = {
    "short": 95_000.0,
    "medium": 76_000.0,
    "long": 22_400.0,
    "IL": 23_000.0,
    "unallocated": 29_800.0,
}
UK_FY_END = pd.Timestamp("2027-03-31")

# DMO D5D outright calendar, pulled 30 Sep 2026, with four November rows
# replaced or added from the DMO feed on 2026-11-11. 5⅜% 2056 was issued at
# £1,000mn on 10 Nov (the 30 Sep residual was £1,604.25mn). 0⅛% 2028 is
# £1,000mn on 11 Nov (not on the 30 Sep D5D sheet). 4⅛% 2033 is £4,000mn on
# 12 Nov (was the prior tap, £4,999.999mn). 5¼% 2041 is £3,250mn on 17 Nov
# (D5D had not sized it; the median £5,000mn fill is no longer the announced size).
# January–March names are still "Conventional" / "Index-linked" and are not
# given another gilt. Remit revision of 28 Oct 2026 (Budget): the 12 Jan and 23 Feb
# index-linked auctions and the 16 Feb conventional auction are cancelled; Q4 is 7
# conventional (7, 26, 27 Jan; 2, 10 Feb; 4, 23 Mar) and 2 index-linked (9 Feb, 9 Mar).
# (date, isin, bucket, amount or None, sec)
UK_CALENDAR = [
    ("2026-09-29", "GB00BWBR1N39", "7-10y", 5_312.5, "Nominal"),
    ("2026-10-06", "GB00BT7HZZ68", "7-10y", 1_250.0, "Linker"),
    ("2026-10-14", "GB00BWBR1P52", "3-7y", 5_000.0, "Nominal"),
    ("2026-10-20", "GB00BWBR1N39", "7-10y", 5_312.5, "Nominal"),
    ("2026-10-22", "GB00BNNGP551", "3-7y", 1_250.0, "Linker"),
    ("2026-10-27", "GB00BWGDRG14", "3-7y", 5_527.749, "Nominal"),
    ("2026-11-03", "GB00BVP99905", "10-15y", 3_250.0, "Nominal"),
    *UK_ANNOUNCED,
    ("2026-12-01", "GB00BWGDRG14", "3-7y", 5_527.749, "Nominal"),
    ("2026-12-08", "GB00BWBR1N39", "7-10y", 5_312.5, "Nominal"),
    ("2026-12-10", "GB00BWBR1P52", "3-7y", 5_000.0, "Nominal"),
    ("2027-01-07", "", "", None, "Nominal"),
    ("2027-01-26", "", "", None, "Nominal"),
    ("2027-01-27", "", "", None, "Nominal"),
    ("2027-02-02", "", "", None, "Nominal"),
    ("2027-02-09", "", "", None, "Linker"),
    ("2027-02-10", "", "", None, "Nominal"),
    ("2027-03-04", "", "", None, "Nominal"),
    ("2027-03-09", "", "", None, "Linker"),
    ("2027-03-23", "", "", None, "Nominal"),
]

# Syndications. Dates and gilts from the minutes; sizes are set by the cash fit to the remit
# (`uk_remit`). 24 Aug 2026 minutes: the Q3 syndication is a linker in November, most support for
# a new 2043; Q3 has no conventional syndication. The 28 Oct remit revision cancelled one long
# syndication and added a medium one (medium £33.8bn in three deals, long £12.5bn in two, IL £7.8bn
# in two): after Apr 2036s, Jun 2041s, Jul IL 2038s and Sep 2056s, one medium and one long deal are
# left for Jan–Mar. Neither is named yet: a new 20y long in January, a new medium (2039) in March.
# Apr–Sep 2027 has no remit;
# it repeats the FY26-27 months (new 10y in April, 15y in June, linker in July, long in September).
# (date, isin, bucket, nominal mn, sec, maturity, coupon, yield, source)
UK_SYNDICATIONS = [
    ("2026-11-24", "NEW IL 2043", "15-20y", 4_000.0, "Linker", "2043-11-22", 2.0, 2.15, "syndication"),
    ("2027-01-19", "NEW 2047", "20-30y", 6_000.0, "Nominal", "2047-07-31", 5.5, 5.5, "syndication"),
    ("2027-03-02", "NEW 2039", "10-15y", 10_000.0, "Nominal", "2039-07-31", 5.0, 5.1, "syndication"),
    ("2027-04-13", "NEW 10y 2037", "10-15y", 15_000.0, "Nominal", "2037-07-31", 4.875, 4.9, "syndication pattern"),
    ("2027-06-08", "NEW 15y 2042", "10-15y", 9_000.0, "Nominal", "2042-01-31", 5.25, 5.3, "syndication pattern"),
    ("2027-07-13", "NEW IL 2043", "15-20y", 4_750.0, "Linker", "2043-11-22", 2.0, 2.15, "syndication pattern"),
    ("2027-09-07", "GB00BT7J0241", "20-30y", 4_250.0, "Nominal", "2056-01-31", 5.375, 5.82, "syndication pattern"),
]

# Q4 auction menus by DMO sector. Short and medium repeat the Oct–Dec gilts; long and IL are the
# gilts the 24 Aug minutes put forward for Jan–Mar (4¼% 2049 / 4¼% 2046 / 4¾% 2043; IL 2038, 2049).
UK_Q4_MENU = {
    "short": ["GB00BWBR1P52", "GB00BWGDRG14", "GB00BVP99780"],
    "medium": ["GB00BWBR1N39", "GB00BVP99905", "GB00BVP99897"],
    "long": ["GB00B39R3707", "GB00B128DP45", "GB00BPJJKP77"],
    "IL": ["GB00BMY62Z61", "GB00BT7J0134", "GB00BT7HZZ68", "GB00BNNGP551"],
}
# Typical outright size by sector. Only breaks the tie when Q4 quotas exceed the dated slots.
_UK_TYPICAL_MN = {"short": 5_000.0, "medium": 4_500.0, "long": 2_500.0, "IL": 1_250.0}
# Gilt tenders. Q3 dates are the 24 Aug minutes (7 Oct, 13 Oct long, 29 Oct, 19 Nov) plus the fifth
# programmatic tender the 28 Oct remit revision adds on 3 Dec (a conventional gilt, not named: booked
# long, where the cancelled auctions were); Q4 is two a
# month, the cadence the DMO has run since the unallocated pot was created. Cash is the unallocated
# remainder in equal parts. FY27-28 has none here: the tape carries FY26-27's tenders as auctions,
# so the H1 replay already repeats them.
UK_TENDERS = [
    ("2026-10-07", "short"), ("2026-10-13", "long"), ("2026-10-29", "medium"), ("2026-11-19", "short"), ("2026-12-03", "long"),
    ("2027-01-13", "medium"), ("2027-01-28", "long"), ("2027-02-03", "short"),
    ("2027-02-24", "medium"), ("2027-03-10", "long"), ("2027-03-17", "short"),
]
UK_TENDER_ISIN = {"short": "GB00BWGDRG14", "medium": "GB00BWBR1N39", "long": "GB00B39R3707"}

# Finanzagentur press release 8, 24 Sep 2026. Million euros. An empty ISIN is the
# second bond of a multi-ISIN auction; that ISIN is announced the week before.
# The 20 Oct green is on the calendar with no volume yet, so it is not booked.
DE_CALENDAR = [
    ("2026-09-30", "DE000BU2Z072", "7-10y", 5_500.0),
    ("2026-10-06", "DE000BU22155", "1-3y", 6_000.0),
    ("2026-10-07", "DE000BU27022", "7-10y", 4_000.0),
    ("2026-10-13", "DE000BU25075", "3-7y", 5_000.0),
    ("2026-10-14", "DE000BU2D012", "20-30y", 1_500.0),
    ("2026-10-14", "", "20-30y", 1_000.0),
    ("2026-10-27", "DE000BU22155", "1-3y", 5_000.0),
    ("2026-10-28", "DE000BU2Z072", "7-10y", 5_500.0),
    ("2026-11-03", "DE000BU25075", "3-7y", 5_000.0),
    ("2026-11-04", "DE000BU2F009", "10-15y", 1_000.0),
    ("2026-11-04", "", "20-30y", 1_000.0),
    ("2026-11-11", "DE000BU2D012", "20-30y", 1_500.0),
    ("2026-11-11", "", "20-30y", 1_000.0),
    ("2026-11-17", "DE000BU22155", "1-3y", 5_000.0),
    ("2026-11-18", "DE000BU27022", "7-10y", 3_000.0),
    ("2026-11-25", "DE000BU2Z072", "7-10y", 5_000.0),
    ("2026-12-01", "DE000BU25075", "3-7y", 4_500.0),
    ("2026-12-08", "DE000BU22155", "1-3y", 4_000.0),
]

# Aug 2026 refunding, then the 5 Aug 2026 TBAC illustrative quarter, then the same grid.
# One TIPS maturity per month, matching the refunding statement.
_US_ROW = ("2y", "3y", "5y", "7y", "10y", "20y", "30y", "tips", "frn")
US_MONTHS = {
    "2026-10": (69, 58, 70, 44, 39, 13, 22, 26, 30, "5y", "refunding"),
    "2026-11": (69, 58, 70, 44, 42, 16, 25, 19, 28, "10y", "tbac"),
    "2026-12": (69, 58, 70, 44, 39, 13, 22, 24, 28, "5y", "tbac"),
    "2027-01": (69, 58, 70, 44, 39, 13, 22, 21, 30, "10y", "tbac"),
    "2027-02": (69, 58, 70, 44, 42, 16, 25, 9, 28, "30y", "steady"),
    "2027-03": (69, 58, 70, 44, 39, 13, 22, 19, 28, "10y", "steady"),
    "2027-04": (69, 58, 70, 44, 39, 13, 22, 26, 30, "5y", "steady"),
    "2027-05": (69, 58, 70, 44, 42, 16, 25, 19, 28, "10y", "steady"),
    "2027-06": (69, 58, 70, 44, 39, 13, 22, 24, 28, "5y", "steady"),
    "2027-07": (69, 58, 70, 44, 39, 13, 22, 21, 30, "10y", "steady"),
    "2027-08": (69, 58, 70, 44, 42, 16, 25, 8, 28, "30y", "steady"),
    "2027-09": (69, 58, 70, 44, 39, 13, 22, 19, 28, "10y", "steady"),
}
_US_YEARS = {"2y": 2, "3y": 3, "5y": 5, "7y": 7, "10y": 10, "20y": 20, "30y": 30, "frn": 2}
_US_DAY = {"2y": 22, "3y": 8, "5y": 23, "7y": 24, "10y": 9, "20y": 15, "30y": 10, "tips": 17, "frn": 23}
# Original 10y and 30y issues. Reopenings the other eight months sit just
# under the bucket boundary; booking them at 10.0 and 30.0 dumps every month
# into 10-15y and 30y+.
_US_NEW_ISSUE_MONTHS = {2, 5, 8, 11}
_US_NEW_MONTHS = {
    ("Nominal", "10y"): _US_NEW_ISSUE_MONTHS, ("Nominal", "20y"): _US_NEW_ISSUE_MONTHS,
    ("Nominal", "30y"): _US_NEW_ISSUE_MONTHS,
    ("Linker", "5y"): {4, 10}, ("Linker", "10y"): {1, 7}, ("Linker", "30y"): {2},
}
# Residuals measured on the 2026 tape (settlement vs the 15th maturity).
_US_RESID = {
    ("Nominal", "10y", True): 10.01,
    ("Nominal", "10y", False): 9.93,
    ("Nominal", "20y", True): 19.92,
    ("Nominal", "20y", False): 19.92,
    ("Nominal", "30y", True): 30.01,
    ("Nominal", "30y", False): 29.93,
    ("Linker", "5y", True): 4.98,
    ("Linker", "5y", False): 4.98,
    ("Linker", "10y", True): 9.90,
    ("Linker", "10y", False): 9.90,
    ("Linker", "30y", True): 29.90,
    ("Linker", "30y", False): 29.90,
}
_US_QUOTE_BAND = {
    ("Nominal", "2y"): (1.6, 2.5),
    ("Nominal", "3y"): (2.6, 3.5),
    ("Nominal", "5y"): (4.5, 5.6),
    ("Nominal", "7y"): (6.5, 7.6),
    ("Nominal", "10y"): (9.4, 10.3),
    ("Nominal", "20y"): (19.2, 20.3),
    ("Nominal", "30y"): (29.2, 30.3),
    ("Linker", "5y"): (4.2, 5.6),
    ("Linker", "10y"): (8.5, 10.3),
    ("Linker", "30y"): (28.0, 30.2),
}

# AFT: 2026 programme €310bn net of buybacks, 85% done at end-September; 2027 programme €340bn
# net (29 Sep 2026).
FR_PROGRAMME_MN = 310_000.0
FR_DONE_SHARE = 0.85
FR_PROGRAMME_ASOF = pd.Timestamp("2026-09-30")
FR_PROGRAMME_NEXT_MN = 340_000.0
IT_H2_MN = 137_500.0  # published midpoint; not a licence to reopen stale lines
IT_LIVE_DAYS = 180
# Forward book end: rolls with the calendar so every country keeps 12 forward months past its latest outturn
# (a fixed Sep-2027 end left Japan with 11 once October 2026 printed). Never earlier than the published horizon.
HORIZON = max(pd.Timestamp("2027-09-30"),
              (pd.Timestamp.today().normalize() + pd.DateOffset(months=12)) + pd.offsets.MonthEnd(0))


def _dmo_bucket(years: float, linker: bool) -> str:
    if linker:
        return "IL"
    if years < 7:
        return "short"
    if years <= 15:
        return "medium"
    return "long"


def _uk_issued(ops: pd.DataFrame) -> dict[str, float]:
    ops = ops[ops["sec_type"].astype(str) != "Bill"]
    fy = ops[(ops["op_dt"] >= "2026-04-01") & (ops["op_dt"] <= UK_FY_END) & (ops["amount_issued"] > 0)]
    out = {k: 0.0 for k in UK_REMIT_MN}
    for row in fy.itertuples(index=False):
        years = row.resid_mat_yrs if pd.notna(row.resid_mat_yrs) else 0.0
        key = _dmo_bucket(float(years), str(row.sec_type) == "Linker")
        out[key] = out.get(key, 0.0) + float(row.amount_issued)
    return out


# A new 30y is residual-bucketed in 20-30y within weeks of issue, so an empty
# 30y+ book is that neighbourhood, not "any open line".
_BUCKET_FALLBACK = {"30y+": ("20-30y", "15-20y")}
_UK_SLEEVE = {
    "short": ("3-7y", "1-3y"),
    "medium": ("7-10y", "10-15y"),
    # 24 Aug 2026 minutes: the one long auction is 5⅜% 2056, or 4¼% 2046 / 4¼% 2049.
    # All three are 20–30y. 4% 2063 was mentioned for a tender and left uncommitted.
    # Ultra (30y+) is not part of the long book.
    "long": ("20-30y", "15-20y"),
    "unallocated": ("3-7y", "7-10y", "10-15y", "15-20y", "20-30y", "30y+"),
    # 24 Aug 2026 GEMM minutes: IL auctions are 2031 / 2035 / 2038 / 2049,
    # and the November syndication is a 2043-area linker, not the 2050s.
    "IL": ("3-7y", "7-10y", "10-15y", "15-20y"),
}
# Q3 2026-27 conventional auctions in those minutes: five short, four medium, one long.
# The 19 Jan 2026 annual GEMM meeting said this shorter mix should continue, with
# caution on the long proportion. The unused long sleeve is not a catch-up.
_UK_AUCTION_WEIGHT = {"short": 5.0, "medium": 4.0, "long": 1.0}


def _pick_line(opened: pd.DataFrame, bucket, linker: bool | None, amount: float | None = None) -> str:
    """Least-recent line that can actually take `amount`.

    Stays inside the requested buckets. A line whose own taps are far smaller
    than the slice (a rump gilt, a mini-tender) is skipped when a full-size
    line exists. The chosen line goes to the back of the queue.
    """
    if opened is None or opened.empty or "tenor_bucket" not in opened.columns:
        return ""
    want = (bucket,) if isinstance(bucket, str) else tuple(bucket)
    pool = opened
    if linker is True:
        pool = pool[pool["sec_type"].astype(str).eq("Linker")]
    elif linker is False:
        pool = pool[~pool["sec_type"].astype(str).eq("Linker")]
    have = set(pool["tenor_bucket"].astype(str))
    chosen = [b for b in want if b in have]
    if not chosen:
        for b in want:
            for alt in _BUCKET_FALLBACK.get(b, ()):
                if alt in have:
                    chosen = [alt]
                    break
            if chosen:
                break
    if not chosen:
        return ""
    pool = pool[pool["tenor_bucket"].astype(str).isin(chosen)]
    if pool.empty:
        return ""
    # A line that has already reached the stock this tenor finishes at is done.
    if "room_mn" in pool.columns:
        room = pd.to_numeric(pool["room_mn"], errors="coerce")
        med = pd.to_numeric(pool["median_size_mn"], errors="coerce") if "median_size_mn" in pool.columns else 0.0
        floor = (0.4 * med.fillna(0)).clip(lower=1.0)
        pool = pool[room.fillna(float("inf")) + 1e-9 >= floor]
        if pool.empty:
            return ""
    if amount and "median_size_mn" in pool.columns:
        med = pd.to_numeric(pool["median_size_mn"], errors="coerce")
        if med.notna().any():
            able = pool[med.fillna(0) >= 0.35 * float(amount)]
            pool = able if not able.empty else pool.loc[[med.idxmax()]]
    pool = pool.sort_values("last_tap")
    isin = str(pool.iloc[0]["isin"])
    nxt = pd.Timestamp(opened["last_tap"].max())
    if pd.isna(nxt):
        nxt = pd.Timestamp("1970-01-01")
    opened.loc[opened["isin"].eq(isin), "last_tap"] = nxt + pd.Timedelta(days=1)
    return isin


def _line_bucket(opened: pd.DataFrame, isin: str, fallback: str) -> str:
    if not isin or opened is None or opened.empty or "isin" not in opened.columns:
        return fallback
    hit = opened.loc[opened["isin"].eq(isin), "tenor_bucket"]
    return str(hit.iloc[0]) if len(hit) else fallback


def _place(opened, buckets, linker, amount, day, source, sec_type: str = "") -> dict:
    want = (buckets,) if isinstance(buckets, str) else tuple(buckets)
    isin = _pick_line(opened, want, linker, amount)
    return _row(isin, _line_bucket(opened, isin, want[0]), day, amount, source, sec_type)


def _tap_cap(opened: pd.DataFrame, isin: str) -> float:
    """Largest tap this line has actually printed, else twice its median."""
    if opened is None or opened.empty or not isin:
        return 0.0
    hit = opened.loc[opened["isin"].eq(isin)]
    if hit.empty:
        return 0.0
    row = hit.iloc[0]

    def _num(col: str) -> float:
        if col not in hit.columns:
            return 0.0
        val = pd.to_numeric(row[col], errors="coerce")
        return float(val) if pd.notna(val) and float(val) > 0 else 0.0

    p90 = _num("p90_size_mn")
    if p90 > 0:
        return p90
    med = _num("median_size_mn")
    return med * 2.0 if med > 0 else 0.0


def _spread(opened, buckets, linker, amount, day, source, sec_type: str = "") -> list[dict]:
    """A bucket-month is several taps. Each line takes at most its own large print."""
    want = (buckets,) if isinstance(buckets, str) else tuple(buckets)
    left = float(amount)
    rows = []
    seen: set[str] = set()
    for _ in range(24):
        if left <= 1:
            break
        isin = _pick_line(opened, want, linker, None)
        if not isin or isin in seen:
            break
        seen.add(isin)
        cap = _tap_cap(opened, isin)
        # A line with no usable print must not be treated as uncapped.
        if cap <= 1:
            continue
        take = min(left, cap)
        rows.append(_row(isin, _line_bucket(opened, isin, want[0]), day, take, source, sec_type))
        left -= take
    if left > 1:
        rows.append(_row("", want[0], day, left, source, sec_type))
    return rows


def _row(isin: str, bucket: str, day, amount: float, source: str, sec_type: str = "") -> dict:
    return {
        "isin": isin or f"NEW {bucket}",
        "tenor_bucket": bucket,
        "op_dt": pd.Timestamp(day),
        "amount_mn": float(amount),
        "source": source,
        "sec_type": sec_type,
    }


def _us_identity(isin: str) -> tuple[str, str]:
    """Token and security type from a synthetic US label. Empty when it is not one."""
    text = str(isin or "")
    if not text.startswith("US "):
        return "", ""
    rest = text[3:].strip()
    low = rest.lower()
    if low == "frn":
        return "frn", "FRN"
    if low.startswith("tips"):
        return rest.split()[-1].lower(), "Linker"
    return low, "Nominal"


def us_forward_years(token: str, when: pd.Timestamp, sec_type: str) -> float | None:
    """Residual years a US forward row should carry so it lands in the tape's bucket."""
    from issuance.tenor import years_from_label

    token = str(token or "").lower()
    kind = "FRN" if sec_type.upper() == "FRN" or token == "frn" else ("Linker" if sec_type == "Linker" else "Nominal")
    if kind == "FRN":
        return 2.0
    new = pd.Timestamp(when).month in _US_NEW_ISSUE_MONTHS
    pinned = _US_RESID.get((kind, token, new))
    if pinned is not None:
        return pinned
    return 2.0 if token == "frn" else years_from_label(token)


def tape_quote(ops: pd.DataFrame, sec_type: str, token: str) -> tuple[float, float] | None:
    """Coupon and yield from the latest auction in this product's residual band."""
    band = _US_QUOTE_BAND.get((sec_type, str(token).lower()))
    if ops is None or ops.empty or band is None or "resid_mat_yrs" not in ops.columns:
        return None
    sub = ops
    if "sec_type" in sub.columns:
        sub = sub[sub["sec_type"].astype(str).eq(sec_type)]
    if "country" in sub.columns:
        sub = sub[sub["country"].astype(str).eq("US")]
    yrs = pd.to_numeric(sub["resid_mat_yrs"], errors="coerce")
    sub = sub[(yrs > band[0]) & (yrs <= band[1])]
    if "coupon" not in sub.columns:
        return None
    sub = sub[pd.to_numeric(sub["coupon"], errors="coerce").notna()]
    if sub.empty:
        return None
    row = sub.sort_values("op_dt").iloc[-1]
    coupon = float(row["coupon"])
    yld = float(row["yield"]) if "yield" in sub.columns and pd.notna(row["yield"]) else coupon
    return coupon, yld


def _stat(opened: pd.DataFrame, isin: str, col: str) -> float:
    if opened is None or opened.empty or not isin or col not in opened.columns:
        return 0.0
    hit = opened.loc[opened["isin"].eq(isin)]
    if hit.empty:
        return 0.0
    val = pd.to_numeric(hit.iloc[0][col], errors="coerce")
    return float(val) if pd.notna(val) and float(val) > 0 else 0.0


def _own_tap(opened: pd.DataFrame, isin: str) -> float:
    """The size this line actually prints. Median, else its large print."""
    med = _stat(opened, isin, "median_size_mn")
    if med > 1:
        return med
    return _stat(opened, isin, "p90_size_mn")


def _room(opened: pd.DataFrame, isin: str) -> float:
    if opened is None or opened.empty or "room_mn" not in opened.columns or not isin:
        return float("inf")
    hit = opened.loc[opened["isin"].astype(str).eq(str(isin)), "room_mn"]
    if hit.empty:
        return float("inf")
    val = pd.to_numeric(hit.iloc[0], errors="coerce")
    return float(val) if pd.notna(val) else float("inf")


def _debit_room(opened: pd.DataFrame, isin: str, take: float) -> None:
    if opened is None or opened.empty or "room_mn" not in opened.columns or not isin or take <= 0:
        return
    idx = opened.index[opened["isin"].astype(str).eq(str(isin))]
    opened.loc[idx, "room_mn"] = pd.to_numeric(opened.loc[idx, "room_mn"], errors="coerce") - float(take)


def _spawn_line(opened: pd.DataFrame, bucket: str, linker: bool) -> str:
    """A new line of this tenor, capped at the stock the last one finished at."""
    if opened is None or opened.empty:
        return ""
    caps = opened.attrs.get("line_cap") or {}
    taps = opened.attrs.get("line_tap") or {}
    sec = "Linker" if linker else "Nominal"
    cap = caps.get((sec, str(bucket)))
    tap = taps.get((sec, str(bucket)))
    if not cap or not tap or float(tap) < 1 or float(cap) < 1:
        return ""
    prefix = f"next IL {bucket}" if linker else f"next {bucket}"
    n = int(opened["isin"].astype(str).str.startswith(prefix).sum()) + 1
    isin = prefix if n == 1 else f"{prefix} {n}"
    row = {c: pd.NA for c in opened.columns}
    row.update({
        "isin": isin,
        "tenor_bucket": str(bucket),
        "sec_type": sec,
        "median_size_mn": float(tap),
        "p90_size_mn": float(tap),
        "room_mn": float(cap),
        "stock_mn": 0.0,
        "last_tap": pd.Timestamp("1970-01-01"),
        "name": "new " + isin[len("next "):],
        "open": True,
    })
    opened.loc[int(opened.index.max()) + 1] = pd.Series(row)
    return isin


_CAP_ORDER = ("1-3y", "3-7y", "7-10y", "10-15y", "15-20y", "20-30y", "30y+")


def _terminal_book(ops: pd.DataFrame, asof: pd.Timestamp, largest: bool = False) -> tuple[dict, dict]:
    """Median stock a line of this tenor finished at, and the tap it was built with.

    Finished means first issued from 2015 and quiet for nine months. A tenor with
    fewer than three such lines borrows the nearest tenor that has them, looking
    longer first: a new 7-10y is the 10y shelf near £40bn, not the 5y shelf.
    """
    from issuance.tenor import bucket_from_years

    if ops is None or ops.empty or "maturity_dt" not in ops.columns:
        return {}, {}
    book = ops[(ops["amount_issued"] > 0) & ops["isin"].notna()].copy()
    if "sec_type" in book.columns:
        book = book[book["sec_type"].astype(str).isin(["Nominal", "Linker"])]
    if book.empty:
        return {}, {}
    book["maturity_dt"] = pd.to_datetime(book["maturity_dt"])
    rows = []
    for isin, grp in book.groupby(book["isin"].astype(str)):
        grp = grp.sort_values("op_dt")
        first, last = grp.iloc[0], grp.iloc[-1]
        if pd.isna(first["maturity_dt"]) or pd.isna(first["op_dt"]):
            continue
        orig = (first["maturity_dt"] - first["op_dt"]).days / 365.25
        med = float(grp["amount_issued"].median())
        rows.append({
            "sec": str(last["sec_type"]),
            "bucket": bucket_from_years(orig),
            "first": first["op_dt"],
            "last": last["op_dt"],
            "n": int(len(grp)),
            "stock": float(grp["amount_issued"].sum()),
            "tap": med,
        })
    if not rows:
        return {}, {}
    built = pd.DataFrame(rows)
    if largest:
        # Benchmarks that have grown past the finished median: the shelf is the largest line
        # of the tenor still being tapped in the last three years.
        fin = built[((asof - built["last"]).dt.days <= 3 * 365) & (built["n"] >= 2) & (built["tap"] >= 500)]
    else:
        fin = built[
            (built["first"] >= "2015-01-01")
            & ((asof - built["last"]).dt.days > 270)
            & (built["n"] >= 4)
            & (built["tap"] >= 500)
        ]
    raw_cap, raw_tap, raw_n = {}, {}, {}
    for (sec, bucket), grp in fin.groupby(["sec", "bucket"]):
        raw_n[(sec, bucket)] = int(len(grp))
        raw_cap[(sec, bucket)] = float(grp["stock"].max() if largest else grp["stock"].median())
        raw_tap[(sec, bucket)] = float(grp["tap"].median())

    def _borrow(sec: str, bucket: str, store: dict) -> float:
        if raw_n.get((sec, bucket), 0) >= 3 and (sec, bucket) in store:
            return store[(sec, bucket)]
        if bucket not in _CAP_ORDER:
            return 0.0
        i = _CAP_ORDER.index(bucket)
        for step in range(1, len(_CAP_ORDER)):
            for j in (i + step, i - step):
                if j < 0 or j >= len(_CAP_ORDER):
                    continue
                key = (sec, _CAP_ORDER[j])
                if raw_n.get(key, 0) >= 3 and key in store:
                    return store[key]
        return 0.0

    caps, taps = {}, {}
    secs = {k[0] for k in raw_n}
    for sec in secs:
        for bucket in _CAP_ORDER:
            cap = _borrow(sec, bucket, raw_cap)
            tap = _borrow(sec, bucket, raw_tap)
            if cap > 1:
                caps[(sec, bucket)] = cap
            if tap > 1:
                taps[(sec, bucket)] = tap
    return caps, taps


def _arm_uk(opened: pd.DataFrame, ops: pd.DataFrame, asof: pd.Timestamp, largest: bool = False) -> pd.DataFrame:
    """Room left on each line before it reaches the stock its tenor finishes at."""
    if opened is None or opened.empty or "stock_mn" not in opened.columns:
        return opened
    caps, taps = _terminal_book(ops, asof, largest=largest)
    if not caps:
        return opened
    out = opened.copy()
    out.attrs["line_cap"] = caps
    out.attrs["line_tap"] = taps
    issue = {}
    if ops is not None and not ops.empty and "maturity_dt" in ops.columns:
        from issuance.tenor import bucket_from_years

        book = ops[ops["isin"].notna() & (ops["amount_issued"] > 0)]
        for isin, grp in book.groupby(book["isin"].astype(str)):
            grp = grp.sort_values("op_dt")
            first = grp.iloc[0]
            if pd.isna(first.get("maturity_dt")) or pd.isna(first.get("op_dt")):
                continue
            orig = (pd.Timestamp(first["maturity_dt"]) - pd.Timestamp(first["op_dt"])).days / 365.25
            sec = "Linker" if str(first.get("sec_type")) == "Linker" else "Nominal"
            issue[str(isin)] = (sec, bucket_from_years(orig))
    rooms = []
    for row in out.itertuples(index=False):
        sec = "Linker" if str(getattr(row, "sec_type", "")) == "Linker" else "Nominal"
        key = issue.get(str(row.isin), (sec, str(getattr(row, "tenor_bucket", ""))))
        cap = caps.get(key) or caps.get((sec, str(getattr(row, "tenor_bucket", ""))))
        stock = float(getattr(row, "stock_mn", 0) or 0)
        rooms.append(max(0.0, float(cap) - stock) if cap else float("inf"))
    out["room_mn"] = rooms
    return out


def _cap_by_sleeve(opened: pd.DataFrame, ops: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """No line outgrows the largest line of its residual sleeve tapped in the last three years.

    The original-maturity shelf lets a 2048 OAT (issued as a 31y) grow to the 2041's €43bn, a
    line that is now medium-term; the long end has never carried more than its largest long line.
    """
    if opened is None or opened.empty or "room_mn" not in opened.columns or ops is None or ops.empty:
        return opened
    from issuance.forward_audit import sleeve_of

    book = ops[(ops["amount_issued"] > 0) & ops["isin"].notna()
               & ops["sec_type"].astype(str).isin(["Nominal", "Linker"])]
    per = book.sort_values("op_dt").groupby(book["isin"].astype(str)).agg(
        stock=("amount_issued", "sum"), last=("op_dt", "max"),
        bucket=("tenor_bucket", "last"), sec=("sec_type", "last"))
    per["sleeve"] = sleeve_of(per["bucket"].astype(str), per["sec"].astype(str))
    recent = per[per["last"] >= asof - pd.DateOffset(years=3)]
    caps = recent.groupby("sleeve")["stock"].max().to_dict()
    out = opened.copy()
    sleeve = sleeve_of(out["tenor_bucket"].astype(str), out["sec_type"].astype(str))
    stock = pd.to_numeric(out["stock_mn"], errors="coerce").fillna(0.0)
    cap = sleeve.map(caps)
    room = (cap - stock).clip(lower=0.0)
    out["room_mn"] = np.minimum(pd.to_numeric(out["room_mn"], errors="coerce").fillna(np.inf),
                                room.fillna(np.inf))
    line_cap = dict(out.attrs.get("line_cap") or {})
    for (sec, bucket), v in list(line_cap.items()):
        s = sleeve_of(pd.Series([bucket]), pd.Series([sec])).iloc[0]
        if s in caps:
            line_cap[(sec, bucket)] = min(v, caps[s])
    out.attrs["line_cap"] = line_cap
    return out


def _own_gap(opened: pd.DataFrame, isin: str, tighten: int) -> pd.Timedelta:
    days = _stat(opened, isin, "median_gap_days") or 28.0
    days = min(120.0, max(14.0, days)) / max(1, tighten)
    return pd.Timedelta(days=max(7.0, days))


def _fill_to_amount(opened, buckets, linker, amount, start, end, source, sec_type: str = "") -> list[dict]:
    """Spend `amount` as each line's own tap, on its own gap. Do not inflate the tap."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    want = (buckets,) if isinstance(buckets, str) else tuple(buckets)
    if amount <= 1 or start > end:
        return []
    pool_ids = []
    if opened is not None and not opened.empty and "tenor_bucket" in opened.columns:
        view = opened
        if linker is True:
            view = view[view["sec_type"].astype(str).eq("Linker")]
        elif linker is False:
            view = view[~view["sec_type"].astype(str).eq("Linker")]
        have = set(view["tenor_bucket"].astype(str))
        chosen = [b for b in want if b in have] or [
            alt for b in want for alt in _BUCKET_FALLBACK.get(b, ()) if alt in have
        ][:1]
        if chosen:
            pool_ids = [
                str(i) for i in view.loc[view["tenor_bucket"].astype(str).isin(chosen), "isin"]
                if _own_tap(opened, str(i)) > 1
            ]
    capped = opened is not None and not opened.empty and "room_mn" in opened.columns
    if capped:
        pool_ids = [i for i in pool_ids if _room(opened, i) >= max(1.0, 0.4 * _own_tap(opened, i))]
    if not pool_ids and not capped:
        return [_row("", want[0], end, amount, source, sec_type)]

    months = pd.date_range(start.replace(day=1), end, freq="MS")
    months = [m for m in months if (m + pd.offsets.MonthEnd(0)) >= start]
    if not months:
        months = [start]
    # A bucket whose monthly share is far below its lines' own tap is tapped every few months at a
    # normal size, not every month in slivers.
    taps_now = [_own_tap(opened, i) for i in pool_ids] if pool_ids else []
    pool_tap = float(np.median(taps_now)) if taps_now else 0.0
    per_month = float(amount) / len(months)
    if pool_tap > 1 and per_month < 0.6 * pool_tap and len(months) > 1:
        step = int(min(len(months), max(1, round(pool_tap / per_month))))
        months = months[step - 1::step] or [months[-1]]
    # Equal month budgets. The sleeve is not spent in the first week it is due.
    budget = float(amount) / len(months)
    rows = []
    cursor = 0
    placed = 0.0
    for i, month in enumerate(months):
        left_m = budget if i < len(months) - 1 else max(0.0, float(amount) - placed)
        month_end = min(end, month + pd.offsets.MonthEnd(0))
        day = max(start, month + pd.Timedelta(days=6))
        if day > month_end:
            day = month_end
        guard = 0
        taps: dict[str, int] = {}
        # A gilt prints about twice in a month. Once every line of this tenor is
        # at its shelf, the next line of that tenor opens. Cash is not left on
        # a synthetic lump.
        limit = max(8, len(pool_ids) * 6 + 12)
        while left_m > 1 and guard < limit:

            def _open(i: str) -> bool:
                size_i = _own_tap(opened, i)
                return size_i > 1 and _room(opened, i) >= max(1.0, 0.4 * size_i)

            if capped:
                live = [i for i in pool_ids if _open(i)]
                if not live:
                    spawned = _spawn_line(opened, want[0], linker is True)
                    if not spawned:
                        break
                    pool_ids.append(spawned)
                    live = [spawned]
                fresh = [i for i in live if taps.get(i, 0) < 2]
                isin = (fresh or live)[cursor % len(fresh or live)]
            else:
                if not pool_ids:
                    break
                isin = pool_ids[cursor % len(pool_ids)]
            cursor += 1
            guard += 1
            size = _own_tap(opened, isin)
            room = _room(opened, isin)
            if size <= 1 or room < max(1.0, 0.4 * size):
                continue
            # A sliver waits for next month's budget (the last month takes whatever is left).
            if left_m < 0.4 * size and i < len(months) - 1:
                break
            take = min(left_m, size, room)
            when = min(month_end, day)
            rows.append(_row(isin, _line_bucket(opened, isin, want[0]), when, take, source, sec_type))
            opened.loc[opened["isin"].astype(str).eq(isin), "last_tap"] = pd.Timestamp(when)
            _debit_room(opened, isin, take)
            taps[isin] = taps.get(isin, 0) + 1
            left_m -= take
            placed += take
            nxt = day + pd.Timedelta(days=4)
            day = nxt if nxt <= month_end else month_end
    if float(amount) - placed > 1:
        rows.append(_row("", want[0], end, float(amount) - placed, source, sec_type))
    return rows


def _uk_tape(ops: pd.DataFrame) -> pd.DataFrame:
    """Gilt outright and syndicated sales on the tape, with DMO sector and cash."""
    from issuance.uk_remit import dmo_sector, index_ratios

    cols = ["op_dt", "isin", "maturity_dt", "price", "amount_issued", "op_type", "sector", "cash", "linker"]
    need = {"op_dt", "isin", "maturity_dt", "price", "amount_issued", "op_type", "sec_type"}
    if ops is None or ops.empty or not need <= set(ops.columns):
        return pd.DataFrame(columns=cols).astype({"op_dt": "datetime64[ns]", "cash": float})
    t = ops[ops["country"].astype(str).eq("UK")] if "country" in ops.columns else ops
    t = t[(t["sec_type"].astype(str) != "Bill") & (t["amount_issued"] > 0)
          & t["op_type"].astype(str).isin(["Auction", "Syndication", "Tender"])].copy()
    if "is_forecast" in t.columns:
        t = t[~t["is_forecast"].astype(bool)]
    ratios = index_ratios()
    t["linker"] = t["sec_type"].astype(str).eq("Linker")
    years = (pd.to_datetime(t["maturity_dt"]) - t["op_dt"]).dt.days / 365.25
    t["sector"] = [dmo_sector(y, lk) for y, lk in zip(years, t["linker"])]
    ratio = [ratios.get(str(i), 1.0) if lk else 1.0 for i, lk in zip(t["isin"].astype(str), t["linker"])]
    t["cash"] = t["amount_issued"] * t["price"] / 100.0 * ratio
    return t[cols]


def uk_plan(ops: pd.DataFrame, asof: pd.Timestamp) -> tuple[list[dict], dict]:
    """UK forward rows before the cash fit, and the cash each (period, sector, method) must raise.

    FY26-27 (to the remit report's FY end): auctions raise what the report says auctions still
    require, by sector; syndications raise planned less done; tenders split the unallocated
    portion. Jan–Mar auction dates are given to sectors so each holds the auctions the report says
    remain. FY27-28 H1 has no remit: each sector raises in auctions+tenders, and in syndications,
    what it raised in Apr–Sep of FY26-27.
    """
    from issuance.uk_remit import SECTORS, bond_price, dmo_sector, index_ratios, remit_progress

    rp = remit_progress()
    report = rp["asof"]
    fy_end = rp["fy_end"]
    ratios = index_ratios()
    tape = _uk_tape(ops)
    seen = set(zip(tape["op_dt"].dt.strftime("%Y-%m-%d"), tape["isin"].astype(str)))
    last = tape.dropna(subset=["price"]).sort_values("op_dt").groupby(tape["isin"].astype(str)).last()

    def maturity(isin, mat=None):
        if mat:
            return pd.Timestamp(mat)
        return pd.Timestamp(last.at[isin, "maturity_dt"]) if isin in last.index else None

    def sector(isin, sec, day, mat=None):
        m = maturity(isin, mat)
        years = (m - pd.Timestamp(day)).days / 365.25 if m is not None else 10.0
        return dmo_sector(years, sec == "Linker")

    rows: list[dict] = []

    def add(isin, bucket, day, amount, source, sec, method, mat=None, cp=None, yy=None):
        when = pd.Timestamp(day)
        # Strict `<` on the cutoff day. `when <= asof` dropped 0⅛% 2028 on
        # 11 Nov 2026: DMO still showed AmountOnOffer £1,000mn and D2.1E had
        # no result, so the line chart stopped on the 8 Oct tap (£2,500mn).
        # The remit report's sales to date include operations on its own date, so those go too.
        key = (when.strftime("%Y-%m-%d"), isin)
        if (when.normalize() < pd.Timestamp(asof).normalize() or when.normalize() <= pd.Timestamp(report).normalize()
                or when > HORIZON or key in seen):
            return
        m = maturity(isin, mat)
        years = (m - when).days / 365.25 if m is not None else 10.0
        sec_name = dmo_sector(years, sec == "Linker")
        row = _row(isin, bucket or bucket_from_years(years), when, amount, source, sec)
        if mat:
            row.update(maturity_dt=pd.Timestamp(mat), coupon_hint=cp, yield_hint=yy)
        px = float(last.at[isin, "price"]) if isin in last.index else float("nan")
        if not np.isfinite(px):
            px = bond_price(cp if cp is not None else 4.0, yy if yy is not None else 4.5, years)
        ratio = ratios.get(isin, 1.0) if sec == "Linker" else 1.0
        period = "fy" if when <= fy_end else "next"
        if period == "next" and method == "tender":
            method = "auction"
        row["_cash"] = px / 100.0 * ratio
        row["_key"] = (period, "all" if method == "tender" else sec_name, method)
        row["_fixed"] = method == "auction" and when <= asof + pd.Timedelta(days=8)
        if method == "tender":
            row["amount_mn"] = 1_000.0 / row["_cash"]
        rows.append(row)

    named = [(d, i, b, float(a), s) for d, i, b, a, s in UK_CALENDAR if a is not None and i]
    for day, isin, bucket, amount, sec in named:
        add(isin, bucket, day, amount, "calendar", sec, "auction")

    have = {s: 0 for s in SECTORS}
    named_cash = {s: 0.0 for s in SECTORS}
    for day, isin, _, amount, sec in named:
        if pd.Timestamp(day) > report:
            s = sector(isin, sec, day)
            have[s] += 1
            px = float(last.at[isin, "price"]) if isin in last.index else 100.0
            named_cash[s] += amount * px / 100.0
    quota = {s: max(rp["auctions_remaining"][s] - have[s], 0) for s in SECTORS}
    slots = {
        "Nominal": [d for d, i, _, a, s in UK_CALENDAR if s == "Nominal" and not i and pd.Timestamp(d) > report],
        "Linker": [d for d, i, _, a, s in UK_CALENDAR if s == "Linker" and not i and pd.Timestamp(d) > report],
    }
    for sec, group in (("Nominal", ("short", "medium", "long")), ("Linker", ("IL",))):
        q = {s: quota[s] for s in group}
        n = len(slots[sec])

        def per_auction(s, k):
            return (rp["auction_required"][s] - named_cash[s]) / max(k, 1) / _UK_TYPICAL_MN[s]

        # More remaining auctions than dated slots: the sector whose auctions would be the
        # smallest against its usual size gives one up. Fewer: the one that would be largest gains.
        while sum(q.values()) > n:
            s = min((s for s in group if q[s] > 0), key=lambda s: per_auction(s, q[s]))
            q[s] -= 1
        while sum(q.values()) < n:
            s = max(group, key=lambda s: per_auction(s, q[s]))
            q[s] += 1
        used = {s: 0 for s in group}
        for k, day in enumerate(slots[sec]):
            s = max(group, key=lambda s: q[s] * (k + 1) / n - used[s])
            menu = UK_Q4_MENU[s]
            isin = menu[used[s] % len(menu)]
            used[s] += 1
            add(isin, "", day, _UK_TYPICAL_MN[s], "pattern", sec, "auction")

    # Apr–Sep 2027 auctions: each Apr–Sep 2026 auction again, 52 weeks on, so each sector keeps
    # its count and the cash fit does not double the taps. The gilt is a current one of that
    # sector and maturity bucket: the Oct–Dec calendar gilts first, then the menu. Not the 2056.
    h1_prev = tape[(tape["op_dt"] >= pd.Timestamp(f"{fy_end.year - 1}-04-01"))
                   & (tape["op_dt"] <= pd.Timestamp(f"{fy_end.year - 1}-09-30"))
                   & tape["op_type"].astype(str).ne("Syndication")]
    pool: dict[str, list[str]] = {}
    for day, isin, _, _, sec in named:
        if isin != "GB00BT7J0241" and day >= "2026-10-01":
            pool.setdefault(sector(isin, sec, day), []).append(isin)
    for s, menu in UK_Q4_MENU.items():
        pool.setdefault(s, [])
        pool[s] += [m for m in menu if m not in pool[s]]
    turn: dict[tuple, int] = {}
    for prev in h1_prev.sort_values("op_dt").itertuples(index=False):
        # The DMO's ACTUAL_DATE is settlement, T+1; the calendar rows are auction days.
        when = pd.Timestamp(prev.op_dt) + pd.Timedelta(days=364) - pd.offsets.BDay(1)
        s = str(prev.sector)
        want = _years_bucket(prev.maturity_dt, prev.op_dt)
        same = [i for i in pool.get(s, []) if maturity(i) is not None and _years_bucket(maturity(i), when) == want]
        cands = same or [i for i in pool.get(s, []) if maturity(i) is not None]
        if not cands:
            continue
        # The gilt that has rested longest; the tape almost never re-taps a gilt inside 14 days.
        k = turn.get((s, want), 0)
        turn[(s, want)] = k + 1
        last_use = {i: max((pd.Timestamp(r["op_dt"]) for r in rows if r["isin"] == i and pd.Timestamp(r["op_dt"]) <= when),
                           default=pd.Timestamp("1900-01-01")) for i in cands}
        isin = max(cands, key=lambda i: (last_use[i] < when - pd.Timedelta(days=14), -last_use[i].value,
                                         -((cands.index(i) - k) % len(cands))))
        sec = "Linker" if bool(prev.linker) else "Nominal"
        add(isin, "", when, float(prev.amount_issued), "pattern", sec, "auction")
    for day, isin, bucket, amount, sec, mat, cp, yy, source in UK_SYNDICATIONS:
        add(isin, bucket, day, amount, source, sec, "syndication", mat, cp, yy)
    # A tender is not of a gilt auctioned within ten days either side.
    for day, s in UK_TENDERS:
        d = pd.Timestamp(day)
        busy = {r["isin"] for r in rows if abs((pd.Timestamp(r["op_dt"]) - d).days) <= 10}
        cands = [UK_TENDER_ISIN[s]] + [m for m in UK_Q4_MENU[s] if m != UK_TENDER_ISIN[s]]
        isin = next((m for m in cands if m not in busy), cands[0])
        add(isin, "", day, 1_000.0, "tender", "Nominal", "tender")

    lag = pd.Timedelta(days=1)
    tender_days = {pd.Timestamp(d) + lag for d, _ in UK_TENDERS}
    after = tape[tape["op_dt"] > report + lag]
    is_tender = after["op_dt"].isin(tender_days)
    targets: dict[tuple, float] = {}
    for s in SECTORS:
        auc = after[~is_tender & after["op_type"].astype(str).ne("Syndication") & after["sector"].eq(s)]
        syn = after[after["op_type"].astype(str).eq("Syndication") & after["sector"].eq(s)]
        targets[("fy", s, "auction")] = rp["auction_required"][s] - float(auc["cash"].sum())
        targets[("fy", s, "syndication")] = max(
            rp["synd_planned"][s] - rp["synd_done"][s] - float(syn["cash"].sum()), 0.0)
    targets[("fy", "all", "tender")] = rp["unallocated"] - float(after[is_tender]["cash"].sum())
    h1_start = pd.Timestamp(f"{fy_end.year - 1}-04-01")
    h1 = tape[(tape["op_dt"] >= h1_start) & (tape["op_dt"] <= pd.Timestamp(f"{fy_end.year - 1}-09-30"))]
    for s in SECTORS if not h1.empty else ():
        part = h1[h1["sector"].eq(s)]
        synd = part["op_type"].astype(str).eq("Syndication")
        targets[("next", s, "auction")] = float(part[~synd]["cash"].sum())
        targets[("next", s, "syndication")] = float(part[synd]["cash"].sum())
    return rows, targets


def _fit_cash(rows: list[dict], targets: dict) -> list[dict]:
    """Scale the unannounced rows of each group so the group raises its target cash exactly."""
    for key, target in targets.items():
        grp = [r for r in rows if r["_key"] == key]
        fixed = sum(r["amount_mn"] * r["_cash"] for r in grp if r["_fixed"])
        free = [r for r in grp if not r["_fixed"]]
        cash = sum(r["amount_mn"] * r["_cash"] for r in free)
        scale = max(target - fixed, 0.0) / cash if cash > 0 else 0.0
        for r in free:
            r["amount_mn"] *= scale
    return [r for r in rows if r["amount_mn"] > 1.0]


def _years_bucket(maturity, day) -> str:
    from issuance.tenor import bucket_from_years

    if pd.isna(maturity):
        return ""
    return bucket_from_years((pd.Timestamp(maturity) - pd.Timestamp(day)).days / 365.25)


def _uk_shelf(rows: list[dict], ops: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """Modelled rows (tenders, unnamed auctions, FY27-28 repeats) leave a gilt once it would outgrow
    the largest gilt of its sector tapped in the last three years: first for the menu gilt with most
    room, else a new gilt. Cash is kept, so the remit fit still holds."""
    from issuance.uk_remit import dmo_sector, index_ratios

    tape = _uk_tape(ops)
    if tape.empty:
        return rows
    tape = tape.assign(isin=tape["isin"].astype(str))
    ratios = index_ratios()
    stock = tape.groupby("isin")["amount_issued"].sum().to_dict()
    last = tape.sort_values("op_dt").groupby("isin").last()
    sector = {i: dmo_sector((pd.Timestamp(r.maturity_dt) - asof).days / 365.25, bool(r.linker))
              for i, r in last.iterrows() if pd.notna(r.maturity_dt)}
    live = [i for i, r in last.iterrows() if r.op_dt >= asof - pd.DateOffset(years=3) and i in sector]
    cap: dict[str, float] = {}
    for i in live:
        cap[sector[i]] = max(cap.get(sector[i], 0.0), stock.get(i, 0.0))

    def cash_of(i: str) -> float:
        if i not in last.index or pd.isna(last.at[i, "price"]):
            return 1.0
        ratio = ratios.get(i, 1.0) if bool(last.at[i, "linker"]) else 1.0
        return float(last.at[i, "price"]) / 100.0 * ratio

    fresh: dict[str, int] = {}
    for r in sorted(rows, key=lambda x: x["op_dt"]):
        i = str(r["isin"])
        s = sector.get(i)
        c = cap.get(s) if s else None
        if r["source"] not in ("pattern", "tender") or c is None or stock.get(i, 0.0) + r["amount_mn"] <= c:
            stock[i] = stock.get(i, 0.0) + r["amount_mn"]
            continue
        cash = r["amount_mn"] * r.get("_cash", cash_of(i))
        # Only a gilt of the same maturity bucket takes the overflow: an IL 2035 tap moved
        # to the IL 2049 would book a 23y linker's DV01 for a 9y one.
        room = {m: c - stock.get(m, 0.0) for m in UK_Q4_MENU.get(s, [])
                if m in sector and m != i and _years_bucket(last.at[m, "maturity_dt"], r["op_dt"]) == r["tenor_bucket"]}
        best = max(room, key=room.get) if room else None
        if best is not None and room[best] >= cash / cash_of(best):
            new, amount = best, cash / cash_of(best)
        else:
            k = fresh.get(s, 1)
            new = f"NEW {s} gilt" + (f" {k}" if k > 1 else "")
            if stock.get(new, 0.0) + cash > c:
                k += 1
                fresh[s] = k
                new = f"NEW {s} gilt {k}"
            amount = cash
        r["isin"], r["amount_mn"] = new, amount
        stock[new] = stock.get(new, 0.0) + amount
    return rows


def _uk(ops: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """D5D dates and gilts, the minutes' syndications and tenders, sized to the remit in cash.

    Sizes D5D has announced (auctions within a week) are kept; every other row in a group is
    scaled by one factor so the group meets its cash target. Nothing past 30 years residual.
    """
    rows, targets = uk_plan(ops, asof)
    out = _uk_shelf(_fit_cash(rows, targets), ops, asof)
    for r in out:
        for k in ("_cash", "_key", "_fixed"):
            r.pop(k, None)
    return out


def _de(ops: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    rows = []
    seen = set(zip(ops["op_dt"].dt.strftime("%Y-%m-%d"), ops["isin"].astype(str)))
    for day, isin, bucket, amount in DE_CALENDAR:
        if pd.Timestamp(day) <= asof:
            continue
        if isin and (day, isin) in seen:
            continue
        # An empty ISIN is the unnamed second bond. It is not assigned to a line.
        rows.append(_row(isin, bucket, day, amount, "calendar"))
    # 2027 is not published (the annual preview comes in December). January–September
    # repeats the auctions Germany actually held in January–September 2026, one year
    # on, so December stays the thin month it is and the year does not collapse to Q4.
    if ops is not None and not ops.empty:
        hist = ops[(ops["op_dt"] >= "2026-01-01") & (ops["op_dt"] < "2026-10-01") & (ops["amount_issued"] > 0)]
        if "sec_type" in hist.columns:
            hist = hist[hist["sec_type"].astype(str) != "Bill"]
        scale = de_2027_scale(ops)
        launch = ops[ops["amount_issued"] > 0].groupby(ops["isin"].astype(str))["op_dt"].min()
        coupons = ops[ops["sec_type"].astype(str).eq("Nominal")] if "sec_type" in ops.columns else ops
        mats = pd.to_datetime(coupons.groupby(coupons["isin"].astype(str))["maturity_dt"].first()).dropna()
        for row in hist.itertuples(index=False):
            when = pd.Timestamp(row.op_dt) + pd.Timedelta(days=364)
            if when <= pd.Timestamp("2026-12-31") or when > HORIZON or when <= asof:
                continue
            bucket = str(getattr(row, "tenor_bucket", "") or "")
            mat = getattr(row, "maturity_dt", pd.NaT)
            if bucket in ("", "nan", "None"):
                years = (pd.Timestamp(mat) - when).days / 365.25 if pd.notna(mat) else 10.0
                bucket = bucket_from_years(years)
            # Off-the-run reopenings (line over a year old, or a long line's taps after its
            # launch) reopen the same Bund, and stop inside its last year. Cycle benchmarks and
            # long launches roll to their 2027 successor: a finished Schatz does not take another year.
            born = launch.get(str(row.isin), pd.Timestamp(row.op_dt))
            long_line = pd.notna(mat) and (pd.Timestamp(mat) - born).days / 365.25 >= 15
            if pd.notna(mat) and (pd.Timestamp(row.op_dt) - born > pd.Timedelta(days=365)
                                  or (long_line and pd.Timestamp(row.op_dt) > born)):
                if pd.Timestamp(mat) - when <= pd.Timedelta(days=365):
                    continue
                r = _row(str(row.isin), bucket_from_years((pd.Timestamp(mat) - when).days / 365.25),
                         when, float(row.amount_issued) * scale, "pattern")
                r["maturity_dt"] = pd.Timestamp(mat)
                rows.append(r)
                continue
            # The successor may already be on the tape (Feb-35 → Feb-36 Bund): reopen it.
            if pd.notna(mat):
                near = ((mats - (pd.Timestamp(mat) + pd.Timedelta(days=364))).abs() <= pd.Timedelta(days=45)) & (
                    mats.index.map(lambda i: launch.get(i, pd.Timestamp.min) > born))
                if near.any() and (mats[near].iloc[0] - when) > pd.Timedelta(days=365):
                    r = _row(near.idxmax(), bucket_from_years((mats[near].iloc[0] - when).days / 365.25),
                             when, float(row.amount_issued) * scale, "pattern")
                    r["maturity_dt"] = mats[near].iloc[0]
                    rows.append(r)
                    continue
            kind = str(getattr(row, "sec_desc", "") or "bond")
            tag = (pd.Timestamp(mat) + pd.Timedelta(days=364)).strftime("%y-%m") if pd.notna(mat) else str(row.isin)[-4:]
            r = _row(f"next {kind} {tag}", bucket, when, float(row.amount_issued) * scale, "pattern")
            if pd.notna(mat):
                r["maturity_dt"] = pd.Timestamp(mat) + pd.Timedelta(days=364)
            rows.append(r)
    return rows


# Federal credit market borrowing, €mn: core budget + Sondervermögen Infrastruktur und
# Klimaneutralität + Sondervermögen Bundeswehr. 2026 Soll (98.0 + 58.1 + 25.5); 2027
# Regierungsentwurf, cabinet 6 Jul 2026, Bundestag 21/7300 (118.7 + 54.9 + 30.0).
DE_NET_BORROWING = {2026: 181_600.0, 2027: 203_600.0}


def de_2027_scale(ops: pd.DataFrame) -> float:
    """2027 coupon gross ÷ 2026 coupon gross, from the budget draft and the redemption wall.

    Gross coupon 2027 = gross 2026 (tape Jan–Sep + Q4 calendar) + Δ net borrowing + Δ coupon
    redemptions, bill stock held. A projection: the Finanzagentur plan (December) replaces it.
    """
    d = ops[ops["country"].astype(str).eq("DE")] if "country" in ops.columns else ops
    c = d[d["sec_type"].astype(str) != "Bill"]
    from issuance.isin_flow import market_ops

    sold = market_ops(c)
    g26 = float(sold[(sold["op_dt"].dt.year == 2026) & (sold["amount_issued"] > 0)]["amount_issued"].sum())
    g26 += sum(a for d_, _, _, a in DE_CALENDAR if a and pd.Timestamp(d_).year == 2026
               and pd.Timestamp(d_) > sold["op_dt"].max())
    stock = c.groupby(c["isin"].astype(str)).agg(amt=("amount_issued", "sum"), mat=("maturity_dt", "first"))
    red = {y: float(stock[pd.to_datetime(stock["mat"]).dt.year == y]["amt"].sum()) for y in (2026, 2027)}
    if g26 <= 0:
        return 1.0
    g27 = g26 + DE_NET_BORROWING[2027] - DE_NET_BORROWING[2026] + red[2027] - red[2026]
    return g27 / g26


def _on_the_run(ops: pd.DataFrame, sec_type: str, token: str) -> str:
    """Latest auctioned CUSIP in this product's residual band. Empty if none."""
    band = _US_QUOTE_BAND.get((sec_type, token))
    if ops is None or ops.empty or band is None or "resid_mat_yrs" not in ops.columns or "isin" not in ops.columns:
        return ""
    sub = ops
    if "sec_type" in sub.columns:
        sub = sub[sub["sec_type"].astype(str).eq(sec_type)]
    yrs = pd.to_numeric(sub["resid_mat_yrs"], errors="coerce")
    sub = sub[(yrs > band[0]) & (yrs <= band[1])]
    if "amount_issued" in sub.columns:
        sub = sub[pd.to_numeric(sub["amount_issued"], errors="coerce").fillna(0) > 0]
    sub = sub[sub["isin"].notna()]
    if sub.empty:
        return ""
    isin = str(sub.sort_values("op_dt").iloc[-1]["isin"])
    return "" if isin.startswith("US ") or isin.startswith("NEW ") else isin


# Overview keeps this many model days so a TBAC day-of-month (30y on the 10th)
# can be moved onto Treasury's close (12 Nov 2026). Dropping at `op_dt > asof`
# removed the row before that move, and the overview strip then had no donor.
_US_LOOKBACK_DAYS = 21


def _treasury_closes(asof: pd.Timestamp) -> dict[tuple[str, tuple[str, int]], pd.Timestamp]:
    """TreasuryDirect auction close by (YYYY-MM, product). Empty if the feed is down."""
    try:
        from scripts import site_events
    except ImportError:
        from issuance import _td_events as site_events
    start = (pd.Timestamp(asof).normalize() - pd.Timedelta(days=_US_LOOKBACK_DAYS)).date()
    try:
        rows = site_events._td_announced(start)
    except Exception:  # noqa: BLE001 — offline build keeps the model day
        return {}
    out: dict[tuple[str, tuple[str, int]], pd.Timestamp] = {}
    for r in rows or []:
        kind = str(r.get("security_type") or "")
        if kind in ("Bill", "CMB"):
            continue
        day = str(r.get("auction_date") or "")[:10]
        if len(day) != 10:
            continue
        prod = site_events._us_product(kind, str(r.get("security_term") or ""))
        out[(day[:7], prod)] = pd.Timestamp(day)
    return out


def _us(asof: pd.Timestamp, ops: pd.DataFrame | None = None, announced: dict | None = None) -> list[dict]:
    """US coupon grid.

    ``announced`` maps (YYYY-MM, (sleeve, years)) to Treasury's close. None
    fetches TreasuryDirect. Pass {} to stay on the model day (tests, feed down).
    A close already behind ``asof`` is omitted. A model day inside 21 days is
    kept when that product has no close yet, so publish can retime it.
    """
    try:
        from scripts import site_events
    except ImportError:
        from issuance import _td_events as site_events
    if announced is None:
        announced = _treasury_closes(asof)
    cut = pd.Timestamp(asof).normalize()
    lookback = cut - pd.Timedelta(days=_US_LOOKBACK_DAYS)
    rows = []
    # 7y is the September coupon still ahead of a 23 Sep tape. Skip if asof is past it.
    if asof < pd.Timestamp("2026-09-24"):
        rows.append(_row("US 7y", bucket_from_years(7), "2026-09-24", 44_000.0, "refunding"))
    issued: set[tuple[str, str]] = set()
    for key, spec in US_MONTHS.items():
        *sizes, tips_tenor, source = spec
        year, month = key.split("-")
        for name, size in zip(_US_ROW, sizes):
            if name == "tips":
                tenor, sec, label = tips_tenor, "Linker", f"US TIPS {tips_tenor}"
            elif name == "frn":
                tenor, sec, label = "frn", "FRN", "US frn"
            else:
                tenor, sec, label = name, "Nominal", f"US {name}"
            when = pd.Timestamp(f"{year}-{month}-01")
            years = us_forward_years(tenor, when, sec) or _US_YEARS.get(tenor, 10)
            day = min(_US_DAY[name], 28)
            # A reopening continues the on-the-run CUSIP until the first forward new issue of
            # that product; after it, the reopening is of that new (synthetic) issue.
            new = when.month in _US_NEW_MONTHS.get((sec, tenor), ())
            if new and when > asof:
                issued.add((sec, tenor))
            synthetic = new or (sec, tenor) in issued or name in {"2y", "3y", "5y", "7y", "frn"}
            live = "" if synthetic else _on_the_run(ops, sec, tenor)
            kind = "TIPS" if sec == "Linker" else ("FRN" if sec == "FRN" else "Note")
            prod = site_events._us_product(kind, label)
            model = pd.Timestamp(f"{year}-{month}-{day:02d}")
            close = announced.get((f"{year}-{month}", prod))
            if close is not None:
                if close < cut or close > HORIZON:
                    continue
                op, src = close, "treasury"
            else:
                if model < lookback or model > HORIZON:
                    continue
                op, src = model, source
            rows.append(_row(
                live or label, bucket_from_years(years), op.strftime("%Y-%m-%d"),
                float(size) * 1000.0, src, sec,
            ))
    return rows


def _pattern_months(opened: pd.DataFrame, ops: pd.DataFrame, asof: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, monthly: float, source: str, mix_days: int = 365) -> list[dict]:
    """The published monthly pace, spent as each line's own tap across the window."""
    recent = ops[(ops["op_dt"] > asof - pd.Timedelta(days=mix_days)) & (ops["amount_issued"] > 0)]
    if "sec_type" in recent.columns:
        recent = recent[~recent["sec_type"].astype(str).isin(["Bill", "Foreign", "Buyback"])]
    # Linkers and nominals share tenor buckets; mixing on the bucket alone hands nominal cash to linker lines.
    linker = recent["sec_type"].astype(str).eq("Linker") if "sec_type" in recent.columns else pd.Series(False, index=recent.index)
    mix = recent.groupby([recent["tenor_bucket"].astype(str), linker.rename("linker")])["amount_issued"].sum()
    total = float(mix.sum()) or 1.0
    cursor = pd.Timestamp(start.year, start.month, 1)
    last = pd.Timestamp(end.year, end.month, 1)
    n_months = 0
    while cursor <= last:
        n_months += 1
        cursor = cursor + pd.DateOffset(months=1)
    rows = []
    for (bucket, is_linker), weight in mix.items():
        rows.extend(_fill_to_amount(
            opened, str(bucket), bool(is_linker), monthly * n_months * float(weight) / total,
            start, end, source, "Linker" if is_linker else "",
        ))
    return rows


def _es_coupon_pace(ops: pd.DataFrame, asof: pd.Timestamp) -> float:
    """Trailing-12m mean coupon gross (€mn/month). Bills out."""
    if ops is None or ops.empty:
        return 17_000.0
    o = ops[ops["country"].astype(str).eq("ES")] if "country" in ops.columns else ops
    o = o[(o["amount_issued"] > 0) & (o["sec_type"].astype(str) != "Bill")
          & (o["op_dt"] > asof - pd.Timedelta(days=365)) & (o["op_dt"] <= asof)]
    if "op_type" in o.columns:
        o = o[o["op_type"].astype(str).ne("Syndication")]
    months = o["op_dt"].dt.to_period("M").nunique()
    if months < 6:
        return 17_000.0
    return float(o["amount_issued"].sum()) / months


def _es_auction_day(when: pd.Timestamp, medium: bool) -> pd.Timestamp:
    """Tesoro: Thursdays — shorter/linker mid-month, longer early-month (2025–26 tape)."""
    return _nth_thursday(when, 3 if medium else 1)


def _es_last_tap(ops: pd.DataFrame, isin: str) -> float:
    if ops is None or ops.empty or "isin" not in ops.columns:
        return float("nan")
    o = ops[ops["isin"].astype(str).eq(isin) & (ops["amount_issued"] > 0)]
    if "op_type" in o.columns:
        o = o[o["op_type"].astype(str).ne("Syndication")]
    if o.empty:
        return float("nan")
    return float(o.sort_values("op_dt").iloc[-1]["amount_issued"])


def _es_announced(asof: pd.Timestamp, ops: pd.DataFrame) -> list[dict]:
    """BdE BON PDFs past ``asof``: ISIN + resolución day; size = last tap of that line."""
    from issuance.paths import CACHE
    from issuance.tenor import bucket_from_years

    path = CACHE / "es_bde_announcements.parquet"
    if not path.exists():
        return []
    a = pd.read_parquet(path)
    if a.empty or "op_dt" not in a.columns:
        return []
    a["op_dt"] = pd.to_datetime(a["op_dt"])
    a = a[a["op_dt"].notna() & (a["op_dt"] > asof)].copy()
    if a.empty:
        return []
    rows = []
    for r in a.itertuples(index=False):
        mat = pd.Timestamp(r.maturity_dt) if pd.notna(getattr(r, "maturity_dt", pd.NaT)) else pd.NaT
        years = (mat - pd.Timestamp(r.op_dt)).days / 365.25 if pd.notna(mat) else float("nan")
        bucket = bucket_from_years(years)
        amt = _es_last_tap(ops, str(r.isin))
        if not (pd.notna(amt) and amt > 0):
            amt = 1_500.0
        sec = str(getattr(r, "sec_type", None) or "Nominal")
        row = _row(str(r.isin), bucket, r.op_dt, float(amt), "calendar", sec)
        if pd.notna(mat):
            row["maturity_dt"] = mat
        rows.append(row)
    return rows


def _es_calendar_dates(asof: pd.Timestamp) -> set[pd.Timestamp]:
    from issuance.paths import CACHE

    path = CACHE / "es_tesoro_calendar.parquet"
    if not path.exists():
        return set()
    c = pd.read_parquet(path)
    if c.empty:
        return set()
    c["op_dt"] = pd.to_datetime(c["op_dt"])
    return {pd.Timestamp(d).normalize() for d in c.loc[c["op_dt"] > asof, "op_dt"]}


def _es_drop_pattern_covered(rows: list[dict], announced: list[dict]) -> list[dict]:
    if not announced:
        return rows
    ann_isins = {str(r["isin"]) for r in announced}
    ann_days = {pd.Timestamp(r["op_dt"]).normalize() for r in announced}
    out = []
    for r in rows:
        if str(r.get("source")) != "pattern":
            out.append(r)
            continue
        if str(r["isin"]) in ann_isins:
            continue
        if pd.Timestamp(r["op_dt"]).normalize() in ann_days:
            continue
        out.append(r)
    return out


def _es(ops: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """Bonos/Obligaciones forward: live lines × trailing coupon pace to HORIZON."""
    start = asof.to_period("M").to_timestamp()
    pace = _es_coupon_pace(ops, asof)
    auc_ops = ops[ops["op_type"].astype(str).ne("Syndication")] if "op_type" in ops.columns else ops
    live = opened
    if opened is not None and not opened.empty and "last_tap" in opened.columns:
        live = opened[pd.to_datetime(opened["last_tap"]) >= asof - pd.Timedelta(days=180)].copy()
    if live is None or live.empty:
        return []
    live = _cap_by_sleeve(_arm_uk(live, ops, asof, largest=True), ops, asof)
    rows = _pattern_months(live, auc_ops, asof, start, HORIZON, pace, "pattern", mix_days=365)
    synd = _syndications_repeat(ops, asof, "ES", start)
    mats = (ops.dropna(subset=["maturity_dt"]).groupby(ops["isin"].astype(str))["maturity_dt"].first()
            if ops is not None and not ops.empty and "maturity_dt" in ops.columns
            else pd.Series(dtype="datetime64[ns]"))
    merged: dict[tuple, dict] = {}
    for row in rows:
        when = pd.Timestamp(row["op_dt"])
        mat = mats.get(str(row["isin"]))
        resid = (pd.Timestamp(mat) - when).days / 365.25 if mat is not None and pd.notna(mat) else None
        linker = row.get("sec_type") == "Linker"
        bucket = str(row["tenor_bucket"])
        if resid is not None:
            linker_or_short = linker or resid < 8.5
        else:
            linker_or_short = linker or bucket in ("0-1y", "1-3y", "3-7y")
        row["op_dt"] = _es_auction_day(when, linker_or_short)
        key = (row["isin"], row["op_dt"])
        if key in merged:
            merged[key]["amount_mn"] += row["amount_mn"]
        else:
            merged[key] = row
    out = list(merged.values())
    cur = asof.to_period("M")
    done = _coupon_tape_in_month(ops, "ES", cur, asof)
    if done > 0:
        _fit_rest_of_month(out, cur, asof, pace, done)
    announced = _es_announced(asof, ops)
    out = _es_drop_pattern_covered(out, announced)
    out += announced
    cal_days = _es_calendar_dates(asof)
    for row in out:
        if str(row.get("source")) == "pattern" and pd.Timestamp(row["op_dt"]).normalize() in cal_days:
            row["source"] = "calendar"
    return [r for r in out if r["amount_mn"] > 0 and r["op_dt"] > asof] + synd


def _fr_announced(asof: pd.Timestamp) -> list[dict]:
    """AFT next-issuance communiqués (ISIN + volume range + auction day) past ``asof``."""
    from issuance.paths import CACHE
    from issuance.tenor import bucket_from_years

    path = CACHE / "fr_aft_announcements.parquet"
    if not path.exists():
        return []
    a = pd.read_parquet(path)
    if a.empty:
        return []
    a["op_dt"] = pd.to_datetime(a["op_dt"])
    a = a[(a["op_dt"] > asof) & a["sec_type"].astype(str).ne("Bill")].copy()
    rows = []
    for r in a.itertuples(index=False):
        mat = pd.Timestamp(r.maturity_dt) if pd.notna(r.maturity_dt) else pd.NaT
        years = (mat - pd.Timestamp(r.op_dt)).days / 365.25 if pd.notna(mat) else float("nan")
        bucket = bucket_from_years(years)
        row = _row(str(r.isin), bucket, r.op_dt, float(r.amount_mn), "calendar", str(r.sec_type))
        if pd.notna(mat):
            row["maturity_dt"] = mat
        rows.append(row)
    return rows


def _fr_drop_pattern_covered(rows: list[dict], announced: list[dict]) -> list[dict]:
    """Announced auction days displace the whole programme pattern on that Thursday."""
    if not announced:
        return rows
    ann_days = {pd.Timestamp(r["op_dt"]).normalize() for r in announced}
    ann_isins = {str(r["isin"]) for r in announced}
    out = []
    for r in rows:
        if str(r.get("source")) != "pattern":
            out.append(r)
            continue
        if str(r["isin"]) in ann_isins:
            continue
        if pd.Timestamp(r["op_dt"]).normalize() in ann_days:
            continue
        out.append(r)
    return out


def _fr(ops: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    # AFT names the Thursday, not the OAT. The pace is the last 92 days. The
    # lines are the ones tapped in the last six months, not the 540-day book.
    # From the current month: its auctions still to come are fitted to the month target less the
    # tape below. Starting next month dropped them (the 15 Oct 2026 medium-term auction).
    start = asof.to_period("M").to_timestamp()
    targets = fr_month_targets(ops)
    # Syndicated new lines are their own rows; auctions raise the rest of each month.
    synd = _syndications_repeat(ops, asof, "FR", start)
    for r in synd:
        m = pd.Timestamp(r["op_dt"]).to_period("M")
        if m in targets:
            targets[m] = max(targets[m] - r["amount_mn"], 0.0)
    auc_ops = ops[ops["op_type"].astype(str).ne("Syndication")] if "op_type" in ops.columns else ops
    # Fill at the programme's own coupon pace, so the line rooms bind on the cash that is booked;
    # the month rescale below only reshapes (December is one auction).
    window = [v for k, v in targets.items() if start.to_period("M") <= k <= HORIZON.to_period("M")]
    if window:
        monthly = float(np.mean(window))
    else:
        recent = ops[(ops["op_dt"] > asof - pd.Timedelta(days=92)) & (ops["amount_issued"] > 0)
                     & (ops["sec_type"].astype(str) != "Bill")]
        monthly = float(recent["amount_issued"].sum()) / 3.0
    live = opened
    if opened is not None and not opened.empty and "last_tap" in opened.columns:
        live = opened[pd.to_datetime(opened["last_tap"]) >= asof - pd.Timedelta(days=180)].copy()
    # Each OAT has room to the stock its tenor finishes at; past that AFT opens the next line.
    live = _cap_by_sleeve(_arm_uk(live, ops, asof, largest=True), ops, asof)
    rows = _pattern_months(live, auc_ops, asof, start, HORIZON, monthly, "pattern", mix_days=365)
    by_month: dict[pd.Period, list] = {}
    for row in rows:
        by_month.setdefault(pd.Timestamp(row["op_dt"]).to_period("M"), []).append(row)
    for month, group in by_month.items():
        if month not in targets:
            continue
        got = sum(r["amount_mn"] for r in group) or 1.0
        for row in group:
            row["amount_mn"] *= targets[month] / got
    mats = ops.dropna(subset=["maturity_dt"]).groupby(ops["isin"].astype(str))["maturity_dt"].first() \
        if "maturity_dt" in ops.columns else pd.Series(dtype="datetime64[ns]")
    merged: dict[tuple, dict] = {}
    for row in rows:
        when = pd.Timestamp(row["op_dt"])
        mat = mats.get(str(row["isin"]))
        resid = (pd.Timestamp(mat) - when).days / 365.25 if mat is not None and pd.notna(mat) else None
        row["op_dt"] = _fr_auction_day(when, str(row["tenor_bucket"]), row.get("sec_type") == "Linker", resid)
        key = (row["isin"], row["op_dt"])
        if key in merged:
            merged[key]["amount_mn"] += row["amount_mn"]
        else:
            merged[key] = row
    out = list(merged.values())
    cur = asof.to_period("M")
    if cur in targets:
        done = _coupon_tape_in_month(ops, "FR", cur, asof)
        if done > 0:
            _fit_rest_of_month(out, cur, asof, targets[cur], done)
    # A remainder far under the line's own tap is not an auction: it joins the largest
    # line of the same kind at that auction.
    for row in sorted(out, key=lambda r: r["amount_mn"]):
        own = _own_tap(live, str(row["isin"]))
        if own <= 1 or row["amount_mn"] >= 0.3 * own:
            continue
        peers = [r for r in out if r is not row and r["op_dt"] == row["op_dt"] and r.get("amount_mn", 0) > 0
                 and (r.get("sec_type") == "Linker") == (row.get("sec_type") == "Linker")]
        if not peers:
            continue
        max(peers, key=lambda r: r["amount_mn"])["amount_mn"] += row["amount_mn"]
        row["amount_mn"] = 0.0
    announced = _fr_announced(asof)
    out = _fr_drop_pattern_covered(out, announced)
    out += announced
    return [r for r in out if r["amount_mn"] > 0] + synd


def _nth_thursday(month: pd.Timestamp, n: int) -> pd.Timestamp:
    first = pd.Timestamp(month.year, month.month, 1)
    return first + pd.Timedelta(days=(3 - first.weekday()) % 7 + 7 * (n - 1))


def _fr_auction_day(when: pd.Timestamp, bucket: str, linker: bool, resid: float | None = None) -> pd.Timestamp:
    """AFT: long-term OATs on the first Thursday, medium-term and OAT€i on the third; December is one long auction.

    On the 2024-26 tape every nominal OAT under 8.5y residual went to the medium-term auction.
    """
    if when.month == 12:
        return _nth_thursday(when, 1)
    medium = linker or (resid < 8.5 if resid is not None else bucket in ("0-1y", "1-3y", "3-7y"))
    return _nth_thursday(when, 3 if medium else 1)


def fr_month_targets(ops: pd.DataFrame) -> dict[pd.Period, float]:
    """Gross OAT issuance per forward month that meets the net programmes exactly.

    Net left in 2026 is the programme less the share AFT reports done; 2027 is the 2027 programme.
    Gross = net × (tape gross YTD ÷ net done), the buyback share already run this year. Months
    take the shape of the last full year on the tape, so December stays one auction.
    """
    c = ops[(ops["sec_type"].astype(str) != "Bill") & (ops["amount_issued"] > 0)]
    if "country" in c.columns:
        c = c[c["country"].astype(str).eq("FR")]
    year = FR_PROGRAMME_ASOF.year
    ytd = float(c[(c["op_dt"].dt.year == year) & (c["op_dt"] <= FR_PROGRAMME_ASOF)]["amount_issued"].sum())
    net_done = FR_PROGRAMME_MN * FR_DONE_SHARE
    gross = ytd / net_done if ytd > 0 else 1.0
    prior = c[c["op_dt"].dt.year == year - 1].groupby(c["op_dt"].dt.month)["amount_issued"].sum()
    if prior.sum() <= 0 or prior.index.nunique() < 12:
        prior = pd.Series(1.0, index=range(1, 13))
    shape = prior / prior.sum()
    out = {}
    rest = [m for m in range(FR_PROGRAMME_ASOF.month + 1, 13)]
    left = FR_PROGRAMME_MN - net_done
    for m in rest:
        out[pd.Period(f"{year}-{m:02d}", "M")] = gross * left * shape[m] / shape[rest].sum()
    for m in range(1, 13):
        out[pd.Period(f"{year + 1}-{m:02d}", "M")] = gross * FR_PROGRAMME_NEXT_MN * shape[m]
    return out


# MEF quarterly programme, 28 Sep 2026. Q4 reopenings plus the new 5y (minimum
# outstanding €10bn over its life). Weights are the last tap, in €bn.
# Oct–Dec gross is the €59–69bn range, midpoint €64bn. Longer, CCTeu and
# linkers are "reserve": they print at their trailing-year monthly volume inside
# that total, and the named bonds take the rest. 2027 is the 2026 Jan–Sep coupon
# pace on the tape, named taps in the trailing nominal mix. With syndications and
# retail BTPs supplemented (it_supplement.json) the tape's Jan–Aug coupon gross is
# €260.0bn against MEF's stated €261.8bn medium-long term; MEF's Jan–Sep €290.8bn
# is the fallback.
_IT_Q4 = (
    ("IT0005729733", "1-3y", 3.00),
    ("IT0005716839", "1-3y", 3.50),
    ("IT 5y 2032", "3-7y", 3.50),
    ("IT0005722845", "3-7y", 3.50),
    ("IT0005729931", "7-10y", 4.00),
)
_IT_Q4_MN = 64_000.0
# MEF 25-Sep-2026: BTP Valore + Valore Insieme, both 5y, placed 19–23 Oct-2026, settlement assumed
# T+2. Inside the €59–69bn (a medium-long issue), so the named auctions share what is left.
# Size until it prints: mean of the last two Valore issues on the tape.
_IT_Q4_RETAIL = ("2026-10-23", "2031-10-27")


def _it_retail_q4(ops: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, list[dict]]:
    """Q4 retail BTP: (size inside the Q4 total, forecast row if not yet on the tape)."""
    day, mat = (pd.Timestamp(d) for d in _IT_Q4_RETAIL)
    if ops is None or ops.empty or "source" not in ops.columns:
        return 0.0, []
    valore = ops["source"].astype(str).str.contains("Valore", case=False) & (ops["amount_issued"] > 0)
    done = ops[valore & (pd.to_datetime(ops["op_dt"]).dt.to_period("M") == day.to_period("M"))]
    if len(done):
        return float(done["amount_issued"].sum()), []
    if day <= asof:
        return 0.0, []
    past = ops[valore]
    if past.empty:
        return 0.0, []
    size = float(past.sort_values("op_dt")["amount_issued"].tail(2).mean())
    row = _row("IT BTP Valore 2031", "3-7y", day, size, "announced retail")
    row["maturity_dt"] = mat
    return size, [row]
_IT_2027_MONTH_MN = (261_800.0 + 29_000.0) / 9.0


def _it_coupon_pace(ops: pd.DataFrame) -> float:
    if ops is None or ops.empty:
        return _IT_2027_MONTH_MN
    if "country" in ops.columns:
        ops = ops[ops["country"] == "IT"]
    c = ops[(ops["sec_type"].astype(str) != "Bill") & (ops["amount_issued"] > 0)
            & ops["op_type"].astype(str).ne("Syndication")
            & (ops["op_dt"] >= "2026-01-01") & (ops["op_dt"] < "2026-10-01")]
    months = c["op_dt"].dt.to_period("M").nunique()
    return float(c["amount_issued"].sum()) / 9.0 if months >= 6 else _IT_2027_MONTH_MN


# Finished nominal BTP stock since 2018, €mn. A benchmark is replaced around here.
_IT_CAP = {"1-3y": 14_500.0, "3-7y": 16_500.0, "7-10y": 18_800.0, "10-15y": 18_800.0,
           "15-20y": 14_000.0, "20-30y": 14_800.0, "30y+": 14_800.0}


_IT_LONG = ("15-20y", "20-30y", "30y+")
# Typical single auction and finished line, €mn: BTP€i, CCTeu, long BTP.
_IT_RESERVE_TAP = {"Linker": 1_000.0, "FRN": 1_500.0, "": 1_250.0}
_IT_RESERVE_CAP = {"Linker": 12_000.0, "FRN": 15_000.0}


def _it_mix(ops: pd.DataFrame, asof: pd.Timestamp) -> tuple[dict, dict]:
    """Trailing-year tape split: monthly € per reserve line, and the named nominal bucket shares.

    MEF's quarterly programme names the 3y-10y benchmarks only; long BTPs, CCTeu and BTP€i are
    "reserve", which in practice means they keep printing at the mid- and end-month auctions.
    """
    if ops is None or ops.empty:
        return {}, {}
    o = ops[ops["country"] == "IT"] if "country" in ops.columns else ops
    o = o[(o["amount_issued"] > 0) & (o["op_dt"] > asof - pd.Timedelta(days=365)) & (o["op_dt"] <= asof)]
    sec = o["sec_type"].astype(str)
    o = o[~sec.isin(["Bill", "Foreign", "Buyback"]) & o["op_type"].astype(str).ne("Syndication")]
    sec = o["sec_type"].astype(str)
    bucket = o["tenor_bucket"].astype(str)
    reserve = {}
    for (kind, label), mask in (
        (("Linker", "BTP€i"), sec.eq("Linker")),
        (("FRN", "CCTeu"), sec.eq("FRN")),
        (("", "long"), ~sec.isin(["Linker", "FRN"]) & bucket.isin(_IT_LONG)),
    ):
        for b, amt in o[mask].groupby(bucket[mask])["amount_issued"].sum().items():
            reserve[(kind, label, str(b))] = float(amt) / 12.0
    named = o[~sec.isin(["Linker", "FRN"]) & ~bucket.isin(_IT_LONG)]
    shares = named.groupby(named["tenor_bucket"].astype(str))["amount_issued"].sum()
    shares = (shares / shares.sum()).to_dict() if shares.sum() > 0 else {}
    return reserve, shares


def _it_months(schedule: list[tuple], asof: pd.Timestamp, stock: dict[str, float]) -> list[dict]:
    """Each named bond takes its own tap until the stock its tenor finishes at. Then the next one starts.

    schedule: (month, named € for the month, bonds as (isin, bucket, tap €bn)).
    """
    live = [(m, float(per), bonds) for m, per, bonds in schedule if m + pd.offsets.MonthEnd(0) > asof and per > 1]
    if not live:
        return []
    built = dict(stock)
    generation: dict[str, int] = {}
    rows = []
    for month, per, bonds in live:
        left = per
        for isin, bucket, tap_bn in bonds:
            if left <= 1:
                break
            generation.setdefault(isin, 0)
            tap = tap_bn * 1000.0
            cap = _IT_CAP.get(bucket, 16_000.0)
            tag = isin[-4:] if isin.startswith("IT0") else isin.removeprefix("IT ").strip()
            ident = isin if generation[isin] == 0 else f"next {tag} {generation[isin]}"
            room = cap - built.get(ident, 0.0)
            guard = 0
            while room < max(1.0, 0.4 * tap) and guard < 6:
                generation[isin] = generation[isin] + 1
                ident = f"next {tag} {generation[isin]}"
                built.setdefault(ident, 0.0)
                room = cap - built[ident]
                guard += 1
            take = min(left, tap, max(room, 0.0))
            if take <= 1:
                continue
            rows.append(_row(ident, bucket, _it_auction_day(month, bucket, ""), take, "pattern"))
            built[ident] = built.get(ident, 0.0) + take
            left -= take
        # The month is larger than the named taps. The extra is the next bond, not a fatter tap;
        # a remainder under half a tap goes on the month's last tap instead of opening a line.
        if 1 < left < 1_750.0 and rows and rows[-1]["op_dt"].month == month.month:
            rows[-1]["amount_mn"] += left
            built[rows[-1]["isin"]] = built.get(rows[-1]["isin"], 0.0) + left
            left = 0.0
        guard = 0
        while left > 1 and guard < 8:
            bucket = "3-7y"
            tap = 3_500.0
            cap = _IT_CAP[bucket]
            generation[bucket] = generation.get(bucket, 0) + 1
            ident = f"next {bucket} {generation[bucket]}"
            built.setdefault(ident, 0.0)
            take = min(left, tap, cap - built[ident])
            if take <= 1:
                break
            rows.append(_row(ident, bucket, _it_auction_day(month, bucket, ""), take, "pattern"))
            built[ident] += take
            left -= take
            guard += 1
    return rows


def _it_auction_day(month: pd.Timestamp, bucket: str, kind: str) -> pd.Timestamp:
    """MEF: mid-month 3y/7y/long BTPs; end-month 5y/10y and CCTeu; BTP Short and BTP€i around the 25th."""
    if kind == "Linker":
        day = 25
    elif kind == "FRN" or bucket in ("3-7y", "10-15y"):
        day = 28
    else:
        day = 12
    return pd.Timestamp(month.year, month.month, day)


def _it_2027_bonds(shares: dict, named_mn: float) -> tuple:
    """2027 named taps in the trailing nominal bucket mix, on the Q4 bonds (or a new one where none)."""
    out = []
    for b, sh in sorted(shares.items()):
        held = [(i, bb) for i, bb, _ in _IT_Q4 if bb == b]
        held = held or [(f"IT {b} new", b)]
        for i, bb in held:
            out.append((i, bb, sh * named_mn / 1000.0 / len(held)))
    return tuple(out) or _IT_Q4


def _it_syndications(ops: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    """2027 syndications and retail BTPs: last year's, a year on, each on a new line.

    Q4 2026 gets no repeat: MEF's €59–69bn covers every medium-long instrument; the one
    announced Q4 issue (BTP Valore, `_it_retail_q4`) sits inside it.
    """
    return _syndications_repeat(ops, asof, "IT", pd.Timestamp("2027-01-01"))


def _coupon_tape_in_month(ops: pd.DataFrame, country: str, month: pd.Period, asof: pd.Timestamp,
                          skip=None) -> float:
    """Coupon gross already on the tape in `month` (operations up to `asof`)."""
    if ops is None or ops.empty:
        return 0.0
    o = ops[(ops["amount_issued"] > 0) & (ops["sec_type"].astype(str) != "Bill") & (ops["op_dt"] <= asof)
            & (ops["op_dt"].dt.to_period("M") == month)]
    if "country" in o.columns:
        o = o[o["country"].astype(str).eq(country)]
    if skip is not None:
        o = o[~skip(o)]
    return float(o["amount_issued"].sum())


def _fit_rest_of_month(rows: list[dict], month: pd.Period, asof: pd.Timestamp, total: float, done: float,
                       pick=None) -> None:
    """Scale the month's operations still to come so tape + forecast = `total`.

    Forecast rows dated up to `asof` are dropped later and the tape stands in for them, so a
    month target spread before that drop would double count whatever the tape sold above them.
    """
    fwd = [r for r in rows if r["op_dt"] > asof and pd.Timestamp(r["op_dt"]).to_period("M") == month
           and (pick is None or pick(r))]
    got = sum(r["amount_mn"] for r in fwd)
    if got <= 0:
        return
    k = max(total - done, 0.0) / got
    for r in fwd:
        r["amount_mn"] *= k


def _syndications_repeat(ops: pd.DataFrame, asof: pd.Timestamp, country: str, start: pd.Timestamp) -> list[dict]:
    """The trailing year's syndications, 52 weeks on, each a new line of the same original term."""
    if ops is None or ops.empty or "op_type" not in ops.columns:
        return []
    o = ops[ops["country"] == country] if "country" in ops.columns else ops
    o = o[o["op_type"].astype(str).eq("Syndication") & (o["amount_issued"] > 0)
          & (o["op_dt"] > asof - pd.Timedelta(days=365)) & (o["op_dt"] <= asof)]
    rows = []
    for r in o.itertuples(index=False):
        day = pd.Timestamp(r.op_dt) + pd.Timedelta(days=364)
        if day < start or day <= asof or day > HORIZON:
            continue
        sec = str(r.sec_type) if str(r.sec_type) in ("Linker", "FRN") else ""
        row = _row(f"next synd {r.tenor_bucket} {day:%Y-%m}", str(r.tenor_bucket), day,
                   float(r.amount_issued), "syndication pattern", sec)
        if pd.notna(getattr(r, "maturity_dt", pd.NaT)):
            row["maturity_dt"] = pd.Timestamp(r.maturity_dt) + pd.Timedelta(days=364)
        rows.append(row)
    return rows


def _it_announced(asof: pd.Timestamp) -> list[dict]:
    """BdI offering announcements (ISIN + size range + auction day) past ``asof``.

    Bills stay out of the coupon paint — same cut as the rest of the forward. Size is the
    midpoint of the published min/max range.
    """
    from issuance.paths import CACHE
    from issuance.tenor import bucket_from_years

    path = CACHE / "it_bdi_announcements.parquet"
    if not path.exists():
        return []
    a = pd.read_parquet(path)
    if a.empty:
        return []
    a["op_dt"] = pd.to_datetime(a["op_dt"])
    a = a[(a["op_dt"] > asof) & a["sec_type"].astype(str).ne("Bill")].copy()
    rows = []
    for r in a.itertuples(index=False):
        mat = pd.Timestamp(r.maturity_dt) if pd.notna(r.maturity_dt) else pd.NaT
        years = (mat - pd.Timestamp(r.op_dt)).days / 365.25 if pd.notna(mat) else float("nan")
        bucket = bucket_from_years(years)
        row = _row(str(r.isin), bucket, r.op_dt, float(r.amount_mn), "calendar", str(r.sec_type))
        if pd.notna(mat):
            row["maturity_dt"] = mat
        rows.append(row)
    return rows


def _it_drop_pattern_covered(rows: list[dict], announced: list[dict]) -> list[dict]:
    """Announced ISINs and same-month reserve synthetics displace the programme pattern."""
    if not announced:
        return rows
    ann_isins = {str(r["isin"]) for r in announced}
    ann_month_kind: set[tuple] = set()
    for r in announced:
        kind = str(r.get("sec_type") or "")
        if kind in {"Linker", "FRN"}:
            ann_month_kind.add((pd.Timestamp(r["op_dt"]).to_period("M"), kind))
        elif kind == "Nominal" and str(r.get("tenor_bucket")) in _IT_LONG:
            ann_month_kind.add((pd.Timestamp(r["op_dt"]).to_period("M"), "long"))
    out = []
    for r in rows:
        if str(r.get("source")) != "pattern":
            out.append(r)
            continue
        isin = str(r["isin"])
        if isin in ann_isins:
            continue
        kind = str(r.get("sec_type") or "")
        mp = pd.Timestamp(r["op_dt"]).to_period("M")
        if kind in {"Linker", "FRN"} and (mp, kind) in ann_month_kind:
            continue
        if isin.startswith("IT long") and (mp, "long") in ann_month_kind:
            continue
        # Synthetic "IT BTP€i …" / "IT CCTeu …" labels from the reserve loop.
        if isin.startswith("IT BTP€i") and (mp, "Linker") in ann_month_kind:
            continue
        if isin.startswith("IT CCTeu") and (mp, "FRN") in ann_month_kind:
            continue
        out.append(r)
    return out


def _it(ops: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    stock = {}
    if ops is not None and not ops.empty and "isin" in ops.columns:
        book = ops[ops["amount_issued"] > 0]
        stock = book.groupby(book["isin"].astype(str))["amount_issued"].sum().to_dict()
    reserve, shares = _it_mix(ops, asof)
    res_month = sum(reserve.values())
    # One stock across both windows, so a bond filled in Q4 is not filled again in 2027.
    q4_months = list(pd.date_range("2026-10-01", "2026-12-01", freq="MS"))
    y_months = list(pd.date_range("2027-01-01", "2027-09-01", freq="MS"))
    retail_mn, retail_rows = _it_retail_q4(ops, asof)
    per_q4 = (_IT_Q4_MN - retail_mn) / len(q4_months)
    pace = _it_coupon_pace(ops)
    # The reserve never takes more than half a month; the named programme is the anchor.
    scale_q4 = min(1.0, 0.5 * per_q4 / res_month) if res_month > 0 else 0.0
    scale_y = min(1.0, 0.5 * pace / res_month) if res_month > 0 else 0.0
    named_q4 = per_q4 - res_month * scale_q4
    named_y = pace - res_month * scale_y
    bonds_y = _it_2027_bonds(shares, named_y)
    schedule = [(m, named_q4, _IT_Q4) for m in q4_months] + [(m, named_y, bonds_y) for m in y_months]
    rows = _it_months(schedule, asof, stock)
    # Reserve lines print at a normal auction size when their monthly share has built up to it,
    # and a full line hands over to the next one.
    months = q4_months + y_months
    for (kind, label, b), amt in reserve.items():
        base, flow = _IT_RESERVE_TAP.get(kind, 1_250.0), amt * max(scale_q4, scale_y)
        # A line running at ~a tap a month prints monthly, not double every other month.
        tap = flow if flow >= 0.75 * base else base
        cap = _IT_RESERVE_CAP.get(kind, _IT_CAP.get(b, 15_000.0))
        carry, built, gen = 0.0, 0.0, 0
        for j, m in enumerate(months):
            carry += amt * (scale_q4 if m in q4_months else scale_y)
            if carry < tap and j < len(months) - 1:
                continue
            # Under half a tap at the horizon is the next auction, after the horizon.
            if carry <= 1 or carry < 0.5 * tap:
                continue
            if built + carry > cap:
                gen, built = gen + 1, 0.0
            ident = f"IT {label} {b}" + (f" #{gen + 1}" if gen else "")
            rows.append(_row(ident, b, _it_auction_day(m, b, kind), carry, "pattern", kind))
            built += carry
            carry = 0.0
    announced = _it_announced(asof)
    rows = _it_drop_pattern_covered(rows, announced)
    rows += announced
    # Reserve carry that has not reached a tap is not sold that month; the named bonds take it,
    # so the Q4 auctions plus the retail issue are MEF's programme exactly.
    def is_named(r: dict) -> bool:
        return r.get("source") == "pattern" and r.get("sec_type", "") == "" and not str(r["isin"]).startswith("IT long")

    def is_retail(o: pd.DataFrame) -> pd.Series:
        return o["source"].astype(str).str.contains("Valore", case=False) if "source" in o.columns else pd.Series(False, index=o.index)

    for m in q4_months:
        if m + pd.offsets.MonthEnd(0) <= asof:
            continue
        mp = m.to_period("M")
        done = _coupon_tape_in_month(ops, "IT", mp, asof, skip=is_retail)
        later = [r for r in rows if r["op_dt"] > asof and r["op_dt"].to_period("M") == mp]
        fixed = sum(r["amount_mn"] for r in later if not is_named(r))
        named_mn = sum(r["amount_mn"] for r in later if is_named(r))
        if named_mn > 0 and abs(per_q4 - done - fixed - named_mn) > 1:
            _fit_rest_of_month(rows, mp, asof, per_q4 - fixed, done, pick=is_named)
    rows += retail_rows + _it_syndications(ops, asof)
    return [r for r in rows if r["op_dt"] > asof and float(r.get("amount_mn") or 0) > 0]


def price_paint(paint: pd.DataFrame, ops: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """DV01 and residual years on forward rows. ISIN maturity when we have it, else the bucket mid."""
    from issuance.dv01 import attach_dv01
    from issuance.tenor import BUCKET_MID, years_from_label

    if paint is None or paint.empty:
        return pd.DataFrame()
    out = paint.copy()
    meta = lines.set_index("isin") if not lines.empty and "isin" in lines.columns else pd.DataFrame()
    years, coupon, yld, sec, ccy, buckets, unpriced = [], [], [], [], [], [], []
    ccy_of = {"UK": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR", "US": "USD", "JP": "JPY"}
    last = ops.sort_values("op_dt").groupby("isin").tail(1).set_index("isin") if not ops.empty else pd.DataFrame()
    for row in out.itertuples(index=False):
        isin = str(row.isin)
        mat = meta.at[isin, "maturity_dt"] if isin in getattr(meta, "index", []) and "maturity_dt" in meta.columns else pd.NaT
        if pd.isna(mat) and pd.notna(getattr(row, "maturity_dt", pd.NaT)):
            mat = row.maturity_dt
        st = str(meta.at[isin, "sec_type"]) if isin in getattr(meta, "index", []) and "sec_type" in meta.columns else ""
        hinted = str(getattr(row, "sec_type", "") or "")
        if hinted and hinted.lower() not in {"", "nan", "none"}:
            st = hinted
        # A synthetic "US 10y" has no CUSIP. The label year is not the residual
        # the tape will show, and a 3% par assumption is not the auction yield.
        token, kind = _us_identity(isin)
        if kind:
            st = kind
        known = us_forward_years(token, pd.Timestamp(row.op_dt), st) if token else None
        if known is None and token:
            known = 2.0 if token == "frn" else years_from_label(token)
        if known is not None:
            yrs = float(known)
        elif pd.notna(mat):
            yrs = (pd.Timestamp(mat) - pd.Timestamp(row.op_dt)).days / 365.25
        elif str(isin).startswith("NEW ") and not lines.empty and "maturity_dt" in lines.columns:
            sib = lines[lines["tenor_bucket"].astype(str).eq(str(row.tenor_bucket))] if "tenor_bucket" in lines.columns else lines.iloc[0:0]
            if "open" in sib.columns:
                sib = sib[sib["open"].astype(bool)]
            sib_yrs = (pd.to_datetime(sib["maturity_dt"]) - pd.Timestamp(row.op_dt)).dt.days / 365.25
            sib_yrs = sib_yrs[sib_yrs > 0.5]
            yrs = float(sib_yrs.median()) if len(sib_yrs) else BUCKET_MID.get(row.tenor_bucket, 10.0)
        else:
            yrs = BUCKET_MID.get(row.tenor_bucket, 10.0)
        if yrs <= 0:
            yrs = float(known) if known is not None else BUCKET_MID.get(row.tenor_bucket, 10.0)
        if token == "frn" or st.upper() == "FRN":
            st = "FRN"
        cp = float(last.at[isin, "coupon"]) if isin in getattr(last, "index", []) and "coupon" in last.columns and pd.notna(last.at[isin, "coupon"]) else 3.0
        yy = float(last.at[isin, "yield"]) if isin in getattr(last, "index", []) and "yield" in last.columns and pd.notna(last.at[isin, "yield"]) else cp
        on_tape = isin in getattr(last, "index", []) and "coupon" in last.columns and pd.notna(last.at[isin, "coupon"])
        if not on_tape and pd.notna(getattr(row, "coupon_hint", np.nan)):
            cp = float(row.coupon_hint)
            yy = float(row.yield_hint) if pd.notna(getattr(row, "yield_hint", np.nan)) else cp
        quote = tape_quote(ops, "Linker" if st == "Linker" else "Nominal", token) if token and st != "FRN" else None
        if quote is not None:
            cp, yy = quote
        if not (0 <= cp <= 15):
            cp = 3.0
        if not (-2 <= yy <= 15):
            yy = cp
        unpriced.append(not on_tape and quote is None and pd.isna(getattr(row, "coupon_hint", np.nan)))
        yrs = min(float(yrs), 50.0)
        years.append(yrs)
        coupon.append(cp)
        yld.append(yy)
        sec.append(st or "Nominal")
        ccy.append(ccy_of.get(str(row.country), "EUR"))
        # A known maturity fixes the bucket at the auction date; a line's tape bucket goes stale.
        buckets.append(bucket_from_years(yrs) if token or (known is None and pd.notna(mat)) else row.tenor_bucket)
    out["resid_mat_yrs"] = years
    out["tenor_bucket"] = buckets
    out["coupon"] = coupon
    out["yield"] = yld
    out["sec_type"] = sec
    out["ccy"] = ccy
    # A line with no tape print and no hint (a new gilt, a next OAT) is a par bond at the market curve.
    miss = np.array(unpriced, dtype=bool)
    if miss.any() and not ops.empty and {"yield", "maturity_dt", "coupon"} <= set(ops.columns):
        from issuance.pricing import curve_yield, market_curve

        curve = market_curve(ops)
        for c in out.loc[miss, "country"].astype(str).unique():
            m = miss & out["country"].astype(str).eq(c).to_numpy()
            y = curve_yield(curve, c, out.loc[m, "sec_type"].astype(str).eq("Linker").to_numpy(),
                            pd.to_datetime(out.loc[m, "op_dt"]).to_numpy(), out.loc[m, "resid_mat_yrs"].to_numpy(dtype=float))
            out.loc[m, "yield"] = y
            out.loc[m, "coupon"] = y
    out["gross_mn"] = out["amount_mn"]
    out["month"] = pd.to_datetime(out["op_dt"]).dt.to_period("M").dt.to_timestamp()
    return attach_dv01(out)


def monthly_forecast(priced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if priced is None or priced.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = priced.copy()
    g["dur"] = g["gross_mn"] * g["resid_mat_yrs"]
    month = g.groupby(["country", "month"], as_index=False).agg(
        coupon_dv01_usd=("dv01_usd_mn", "sum"),
        face_mn=("gross_mn", "sum"),
        duration_issued_mn_y=("dur", "sum"),
    )
    month["wam_issuance"] = month["duration_issued_mn_y"] / month["face_mn"].replace(0, pd.NA)
    tenor = g.groupby(["country", "month", "tenor_bucket"], as_index=False).agg(
        dv01_usd_mn=("dv01_usd_mn", "sum"),
        gross_mn=("gross_mn", "sum"),
        dur=("dur", "sum"),
    )
    tenor["wam_issuance"] = tenor["dur"] / tenor["gross_mn"].replace(0, pd.NA)
    tenor["coupon_issued_mn"] = tenor["gross_mn"]
    return month, tenor


def _fold_remainder(supply: pd.DataFrame, month: pd.DataFrame, tenor: pd.DataFrame, last: pd.Series) -> pd.DataFrame:
    """Add forecast operations dated inside a country's last tape month (the rest of the current
    month, or auctions the tape has not caught up with) to that month, and record the amount in
    `fc_remainder_dv01` so readers can treat the month as incomplete history."""
    from issuance.rv_metrics import BELLY, FRONT, LONG, MIDLONG, ULTRA

    supply = supply.copy()
    supply["fc_remainder_dv01"] = 0.0
    rem = month[[r.month == last.get(r.country) for r in month.itertuples()]]
    if rem.empty:
        return supply
    legs = {"dv01_front": FRONT, "dv01_belly": BELLY, "dv01_midlong": MIDLONG, "dv01_long": LONG, "dv01_ultra": ULTRA}
    for r in rem.itertuples():
        i = supply.index[(supply["country"] == r.country) & (supply["month"] == r.month)]
        if not len(i):
            continue
        add = float(r.coupon_dv01_usd)
        for col in ("coupon_dv01_usd", "gross_dv01_usd", "net_coupon_dv01_usd", "private_dv01_usd"):
            if col in supply:
                supply.loc[i, col] = supply.loc[i, col].fillna(0.0) + add
        if "coupon_gross_mn" in supply:
            supply.loc[i, "coupon_gross_mn"] = supply.loc[i, "coupon_gross_mn"].fillna(0.0) + float(r.face_mn)
        tb = tenor[(tenor["country"] == r.country) & (tenor["month"] == r.month)]
        for col, names in legs.items():
            if col in supply:
                supply.loc[i, col] = supply.loc[i, col].fillna(0.0) + float(tb.loc[tb["tenor_bucket"].isin(names), "dv01_usd_mn"].sum())
        supply.loc[i, "fc_remainder_dv01"] = add
    hit = supply["fc_remainder_dv01"] > 0
    if hit.any() and {"dv01_front", "dv01_belly", "dv01_long"} <= set(supply.columns):
        s = supply.loc[hit]
        supply.loc[hit, "dv01_2s10s"] = s["dv01_belly"] - s["dv01_front"]
        supply.loc[hit, "dv01_10s30s"] = s["dv01_long"] - s["dv01_belly"]
        supply.loc[hit, "dv01_5s30s"] = s["dv01_front"] + s["dv01_long"] - 2.0 * s["dv01_belly"]
    return supply


def extend_published(month: pd.DataFrame, tenor: pd.DataFrame) -> None:
    """Append forecast months onto the published supply, WAM and tenor panels."""
    from issuance.paths import CACHE
    from issuance.rv_metrics import BELLY, FRONT, LONG, MIDLONG, ULTRA

    if month.empty or not (CACHE / "rv_supply.parquet").exists():
        return
    supply = pd.read_parquet(CACHE / "rv_supply.parquet")
    wam = pd.read_parquet(CACHE / "rv_wam.parquet")
    ten = pd.read_parquet(CACHE / "dv01_tenor.parquet")
    supply["month"] = pd.to_datetime(supply["month"])
    wam["month"] = pd.to_datetime(wam["month"])
    ten["month"] = pd.to_datetime(ten["month"])
    last = supply.groupby("country")["month"].max()
    supply = _fold_remainder(supply, month, tenor, last)
    extra = month[month.apply(lambda r: r["month"] > last.get(r["country"], pd.Timestamp("1900-01-01")), axis=1)].copy()
    if extra.empty:
        return
    from issuance.qt_forward import qt_path
    qt = qt_path(extra[["country", "month"]])
    extra = extra.merge(qt, on=["country", "month"], how="left")
    extra["official_dv01_usd"] = extra["official_dv01_usd"].fillna(0.0)
    extra["official_stock_delta_mn"] = extra["official_stock_delta_mn"].fillna(0.0)
    extra = extra.sort_values(["country", "month"])
    for col in ("bill_dv01_usd", "official_bill_dv01_usd"):
        extra[col] = _same_month_last_year(supply, extra, col)
    jp_bills = _jp_bills_fc(extra[["country", "month"]])
    if not jp_bills.empty:
        jp = extra["country"].eq("JP")
        extra.loc[jp, "bill_dv01_usd"] = extra.loc[jp, "month"].map(
            jp_bills.set_index("month")["dv01_usd_mn"]).fillna(0.0).to_numpy()
    from issuance.fiscal_bridge import apply_us
    extra, bridge_info = apply_us(supply, extra, tenor)
    bill_ratio = {pd.Timestamp(k): v for k, v in bridge_info["bill_stock_ratio"].items()}
    extra["net_coupon_dv01_usd"] = extra["coupon_dv01_usd"] + extra["buyback_dv01_usd"]
    extra["private_dv01_usd"] = extra["net_coupon_dv01_usd"] - extra["official_dv01_usd"]
    buckets = tenor.merge(extra[["country", "month"]], on=["country", "month"])
    def _b(names, col):
        sub = buckets[buckets["tenor_bucket"].isin(names)]
        return sub.groupby(["country", "month"])["dv01_usd_mn"].sum().rename(col)
    curve = pd.concat([
        _b(FRONT, "dv01_front"), _b(BELLY, "dv01_belly"), _b(MIDLONG, "dv01_midlong"),
        _b(LONG, "dv01_long"), _b(ULTRA, "dv01_ultra"),
    ], axis=1).reset_index()
    extra = extra.merge(curve, on=["country", "month"], how="left")
    for col in ("dv01_front", "dv01_belly", "dv01_midlong", "dv01_long", "dv01_ultra"):
        extra[col] = extra[col].fillna(0.0)
    extra["dv01_2s10s"] = extra["dv01_belly"] - extra["dv01_front"]
    extra["dv01_10s30s"] = extra["dv01_long"] - extra["dv01_belly"]
    extra["dv01_5s30s"] = extra["dv01_front"] + extra["dv01_long"] - 2.0 * extra["dv01_belly"]
    both = pd.concat([supply, extra], ignore_index=True).sort_values(["country", "month"])
    rolls = {
        "coupon_dv01_usd_12m": ("coupon_dv01_usd", 12),
        "private_dv01_usd_3m": ("private_dv01_usd", 3),
        "private_dv01_usd_12m": ("private_dv01_usd", 12),
        "official_dv01_usd_3m": ("official_dv01_usd", 3),
        "official_dv01_usd_12m": ("official_dv01_usd", 12),
        "dv01_2s10s_3m": ("dv01_2s10s", 3),
        "dv01_2s10s_12m": ("dv01_2s10s", 12),
        "dv01_10s30s_3m": ("dv01_10s30s", 3),
        "dv01_10s30s_12m": ("dv01_10s30s", 12),
        "dv01_5s30s_3m": ("dv01_5s30s", 3),
    }
    for name, (col, n) in rolls.items():
        both[name] = both.groupby("country")[col].transform(lambda s, n=n: s.rolling(n, min_periods=1).sum())
    de = both.loc[both["country"].eq("DE"), ["month", "private_dv01_usd"]].drop_duplicates("month")
    de = de.rename(columns={"private_dv01_usd": "de_private_dv01_usd"})
    both = both.drop(columns=[c for c in ("de_private_dv01_usd", "priv_vs_de", "priv_vs_de_3m", "priv_vs_de_12m") if c in both.columns])
    both = both.merge(de, on="month", how="left")
    both["priv_vs_de"] = both["private_dv01_usd"] - both["de_private_dv01_usd"]
    both.loc[both["country"].eq("DE"), "priv_vs_de"] = 0.0
    both["priv_vs_de_3m"] = both.groupby("country")["priv_vs_de"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    both["priv_vs_de_12m"] = both.groupby("country")["priv_vs_de"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    from issuance.rv_metrics import add_bill_view
    both["official_bill_dv01_usd"] = both["official_bill_dv01_usd"].fillna(0.0)
    both["bill_dv01_usd"] = both["bill_dv01_usd"].fillna(0.0)
    both = add_bill_view(both)
    fc_months = set(zip(extra["country"], extra["month"]))
    is_fc = both.apply(lambda r: (r["country"], r["month"]) in fc_months, axis=1) | (both["fc_remainder_dv01"].fillna(0.0) > 0)
    for col in ("coupon_dv01_usd_12m_z", "private_dv01_z", "private_dv01_usd_3m_z", "private_dv01_usd_12m_z",
                "imbalance_z", "demand_z"):
        for c in (col, col + "_ib"):
            if c in both.columns:
                both.loc[is_fc, c] = pd.NA
    both.to_parquet(CACHE / "rv_supply_fc.parquet", index=False)

    wam_extra = extra[["country", "month", "wam_issuance", "face_mn", "duration_issued_mn_y", "official_stock_delta_mn"]].rename(columns={"face_mn": "coupon_issued_mn"})
    if "bill_issued_mn" in wam.columns:
        from issuance.fiscal_bridge import scaled_last_year
        for col in ("bill_issued_mn", "bill_duration_mn_y"):
            wam_extra[col] = _same_month_last_year(wam, wam_extra, col)
            s = scaled_last_year(wam, wam_extra[wam_extra["country"] == "US"], col, bill_ratio)
            wam_extra.loc[s.index, col] = s
        if not jp_bills.empty:
            jp = wam_extra["country"].eq("JP")
            for col, src in (("bill_issued_mn", "gross_mn"), ("bill_duration_mn_y", "duration_mn_y")):
                wam_extra.loc[jp, col] = wam_extra.loc[jp, "month"].map(
                    jp_bills.set_index("month")[src]).fillna(0.0).to_numpy()
        wam_extra["coupon_issued_mn_ib"] = wam_extra["coupon_issued_mn"] + wam_extra["bill_issued_mn"]
        wam_extra["duration_issued_mn_y_ib"] = wam_extra["duration_issued_mn_y"] + wam_extra["bill_duration_mn_y"]
        wam_extra["wam_issuance_ib"] = wam_extra["duration_issued_mn_y_ib"] / wam_extra["coupon_issued_mn_ib"].replace(0.0, pd.NA)
    # A month already on the tape with no coupon yet (bills only) is in both: keep its stock, take the forward issuance.
    wboth = (pd.concat([wam, wam_extra], ignore_index=True)
             .groupby(["country", "month"], as_index=False, sort=True).first())
    parts = []
    for _, sub in wboth.groupby("country"):
        sub = sub.sort_values("month").copy()
        for sfx in ("", "_ib"):
            if "wam_stock" + sfx not in sub.columns:
                continue
            for col in ("wam_stock", "wam_public", "wam_private"):
                sub[col + sfx] = sub[col + sfx].ffill()
            if "official_mn" + sfx in sub.columns and "official_stock_delta_mn" in sub.columns:
                delta = sub["official_stock_delta_mn"].fillna(0.0)
                sub["official_mn" + sfx] = sub["official_mn" + sfx].ffill() + delta.cumsum()
                if "outstanding_mn" + sfx in sub.columns:
                    stock = sub["outstanding_mn" + sfx].ffill()
                    sub["private_mn" + sfx] = (stock - sub["official_mn" + sfx]).clip(lower=0)
                    sub["official_share" + sfx] = sub["official_mn" + sfx] / stock.replace(0, pd.NA)
            dur, face = sub["duration_issued_mn_y" + sfx], sub["coupon_issued_mn" + sfx]
            for n, col in ((3, "wam_issuance_3m"), (12, "wam_issuance_12m")):
                d = dur.rolling(n, min_periods=1).sum()
                a = face.rolling(n, min_periods=1).sum()
                sub[col + sfx] = d / a.replace(0.0, pd.NA)
            sub["duration_issued_3m" + sfx] = dur.rolling(3, min_periods=1).sum()
            sub["duration_issued_12m" + sfx] = dur.rolling(12, min_periods=1).sum()
            sub["wam_issuance_gap" + sfx] = sub["wam_issuance" + sfx] - sub["wam_stock" + sfx]
        parts.append(sub)
    pd.concat(parts, ignore_index=True).to_parquet(CACHE / "rv_wam_fc.parquet", index=False)

    ten_c = ten[ten.get("sec_type", pd.Series("", index=ten.index)).astype(str) != "Bill"] if "sec_type" in ten.columns else ten
    rem_months = set(zip(supply.loc[supply["fc_remainder_dv01"] > 0, "country"], supply.loc[supply["fc_remainder_dv01"] > 0, "month"]))
    rem_rows = tenor[[(c, m) in rem_months for c, m in zip(tenor["country"], tenor["month"])]]
    t_extra = pd.concat([rem_rows, buckets], ignore_index=True)
    t_both = pd.concat([
        ten_c[["country", "month", "tenor_bucket", "dv01_usd_mn", "gross_mn"]],
        t_extra[["country", "month", "tenor_bucket", "dv01_usd_mn", "gross_mn"]],
    ], ignore_index=True)
    t_both.to_parquet(CACHE / "dv01_tenor_fc.parquet", index=False)
    cols = ["country", "month", "tenor_bucket", "dv01_usd_mn", "gross_mn"]
    if "sec_type" in ten.columns:
        bills = ten[ten["sec_type"].astype(str) == "Bill"][cols]
        bills_fc = _shift_year(bills, extra[["country", "month"]])
        if bill_ratio and not bills_fc.empty:
            # Chained rows already carry earlier ratios; compound month by month.
            us = bills_fc["country"].eq("US")
            for m in sorted(bill_ratio):
                for k in range(0, 36, 12):
                    sel = us & bills_fc["month"].eq(m + pd.DateOffset(months=k))
                    bills_fc.loc[sel, ["dv01_usd_mn", "gross_mn"]] *= bill_ratio[m]
        if not jp_bills.empty:
            bills_fc = pd.concat([bills_fc[bills_fc["country"].ne("JP")],
                                  jp_bills.assign(country="JP", tenor_bucket="0-1y")[cols]], ignore_index=True)
        pd.concat([bills, bills_fc], ignore_index=True).to_parquet(CACHE / "dv01_tenor_bills_fc.parquet", index=False)
    wcols = ["country", "month", "tenor_bucket", "wam_issuance", "coupon_issued_mn"]
    wt = tenor.drop(columns=["gross_mn"], errors="ignore")
    if (CACHE / "rv_wam_tenor.parquet").exists():
        hist_wt = pd.read_parquet(CACHE / "rv_wam_tenor.parquet")
        hist_wt["month"] = pd.to_datetime(hist_wt["month"])
        wt = pd.concat([hist_wt, wt[wcols]], ignore_index=True)
    wt.to_parquet(CACHE / "rv_wam_tenor_fc.parquet", index=False)
    if (CACHE / "rv_wam_tenor_ib.parquet").exists():
        hist_ib = pd.read_parquet(CACHE / "rv_wam_tenor_ib.parquet")
        hist_ib["month"] = pd.to_datetime(hist_ib["month"])
        front_fc = _shift_year(hist_ib[hist_ib["tenor_bucket"].eq("0-1y")], extra[["country", "month"]])
        coup_fc = tenor[~tenor["tenor_bucket"].eq("0-1y")][wcols]
        pd.concat([hist_ib, coup_fc, front_fc], ignore_index=True).to_parquet(
            CACHE / "rv_wam_tenor_ib_fc.parquet", index=False)


def _same_month_last_year(hist: pd.DataFrame, extra: pd.DataFrame, col: str) -> pd.Series:
    """Value of `col` twelve months earlier, chaining through forecast months past a year out."""
    if col not in hist.columns:
        return pd.Series(0.0, index=extra.index)
    known = {(c, pd.Timestamp(m)): v for c, m, v in zip(hist["country"], hist["month"], hist[col]) if pd.notna(v)}
    recent = (hist.sort_values("month").groupby("country")[col]
              .apply(lambda s: float(s.dropna().tail(12).mean()) if s.notna().any() else 0.0).to_dict())
    out = []
    for c, m in extra[["country", "month"]].sort_values(["country", "month"]).itertuples(index=False):
        v = float(known.get((c, pd.Timestamp(m) - pd.DateOffset(months=12)), recent.get(c, 0.0)))
        known[(c, pd.Timestamp(m))] = v
        out.append(v)
    s = pd.Series(out, index=extra.sort_values(["country", "month"]).index)
    return s.reindex(extra.index)


def _jp_bills_fc(months: pd.DataFrame) -> pd.DataFrame:
    """JP forward bills per month (jp_forward.bill_rows), DV01 priced like the tape:
    month, gross_mn, dv01_usd_mn, duration_mn_y. Empty when there is no JP bill tape."""
    from issuance.dv01 import attach_dv01
    from issuance.jp_forward import bill_rows
    from issuance.paths import CACHE

    out_cols = ["month", "gross_mn", "dv01_usd_mn", "duration_mn_y"]
    jm = pd.to_datetime(months.loc[months["country"].eq("JP"), "month"])
    path = CACHE / "ops_jp.parquet"
    if jm.empty or not path.exists():
        return pd.DataFrame(columns=out_cols)
    ops = pd.read_parquet(path)
    asof = pd.Timestamp(pd.to_datetime(ops["op_dt"]).max())
    r = bill_rows(ops, asof, jm.max() + pd.offsets.MonthEnd(0))
    if r.empty:
        return pd.DataFrame(columns=out_cols)
    r["month"] = r["op_dt"].dt.to_period("M").dt.to_timestamp()
    r = r[r["month"].isin(set(jm))]
    priced = attach_dv01(pd.DataFrame({
        "month": r["month"].to_numpy(), "gross_mn": r["amount_mn"].to_numpy(), "tenor_bucket": "0-1y",
        "resid_mat_yrs": r["resid_mat_yrs"].to_numpy(), "coupon": 0.0, "yield": r["yield"].to_numpy(),
        "price": r["price"].to_numpy(), "ccy": "JPY", "sec_type": "Bill"}))
    priced["duration_mn_y"] = priced["gross_mn"] * priced["resid_mat_yrs"]
    return priced.groupby("month", as_index=False)[["gross_mn", "dv01_usd_mn", "duration_mn_y"]].sum()


def _shift_year(rows: pd.DataFrame, months: pd.DataFrame) -> pd.DataFrame:
    """Rows for each forecast month copied from the same month a year earlier (chained)."""
    if rows.empty or months.empty:
        return rows.iloc[0:0]
    rows = rows.copy()
    rows["month"] = pd.to_datetime(rows["month"])
    parts = []
    pool = rows
    for c, m in months.drop_duplicates().sort_values(["country", "month"]).itertuples(index=False):
        src = pool[(pool["country"] == c) & (pool["month"] == pd.Timestamp(m) - pd.DateOffset(months=12))]
        if src.empty:
            continue
        new = src.assign(month=pd.Timestamp(m))
        parts.append(new)
        pool = pd.concat([pool, new], ignore_index=True)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0]


def project(country: str, ops: pd.DataFrame, opened: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    if country == "UK":
        rows = _uk(ops, opened, asof)
    elif country == "DE":
        rows = _de(ops, opened, asof)
    elif country == "US":
        rows = _us(asof, ops)
    elif country == "FR":
        rows = _fr(ops, opened, asof)
    elif country == "IT":
        rows = _it(ops, opened, asof)
    elif country == "ES":
        rows = _es(ops, opened, asof)
    elif country == "JP":
        from issuance.jp_forward import jp_rows

        rows = jp_rows(ops, asof, HORIZON)
    else:
        rows = []
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["op_dt"] = pd.to_datetime(out["op_dt"]).map(lambda d: business_day(d, country))
    # UK and US keep an auction dated on the origin. `>` dropped the 0⅛% 2028
    # gilt for the whole of 11 Nov 2026 while DMO still listed it, and dropped
    # the US 30y on 12 Nov 2026 (Treasury close that day, model day the 10th):
    # `_us` emits the close, then this filter deleted `op_dt == asof` before
    # the overview strip was built. auctionSizes still showed the coupon
    # because the chart retime runs on a different table. Other countries
    # drop `op_dt == asof` inside their builders; widening those filters
    # reopens a same-day row they already chose to omit.
    origin = pd.Timestamp(asof).normalize()
    if country in ("UK", "US"):
        out = out[out["op_dt"] >= origin]
    else:
        out = out[out["op_dt"] > asof]
    out["country"] = country
    return out


def _closed(day: pd.Timestamp, country: str) -> bool:
    if day.weekday() >= 5 or (day.month, day.day) in {(12, 24), (12, 25), (12, 26), (12, 31), (1, 1)}:
        return True
    if country == "US":
        from pandas.tseries.holiday import USFederalHolidayCalendar

        return day in USFederalHolidayCalendar().holidays(day - pd.Timedelta(days=1), day + pd.Timedelta(days=1))
    from dateutil.easter import easter

    sunday = pd.Timestamp(easter(day.year))
    return day in (sunday - pd.Timedelta(days=2), sunday + pd.Timedelta(days=1))


def business_day(day: pd.Timestamp, country: str) -> pd.Timestamp:
    """A modelled date on a weekend or market holiday moves to the previous open day, or the next one
    if that would leave the month (the month is what the panels book)."""
    d = pd.Timestamp(day)
    back = d
    while _closed(back, country):
        back -= pd.Timedelta(days=1)
    if back.month == d.month:
        return back
    fwd = d
    while _closed(fwd, country):
        fwd += pd.Timedelta(days=1)
    return fwd


def reconcile(country: str, ops: pd.DataFrame, priced: pd.DataFrame) -> str:
    """One generated sentence per country: what the forward is anchored to, with today's numbers.

    `priced` is that country's rows of forecast_priced (op_dt, source, sec_type, amount_mn)."""
    p = priced if priced is not None else pd.DataFrame()
    day = pd.to_datetime(p["op_dt"]) if len(p) else pd.Series(dtype="datetime64[ns]")
    if country == "UK":
        from issuance.uk_remit import remit_progress

        rp = remit_progress()
        left = rp["total_planned"] - rp["sales_to_date"]
        auc = sum(rp["auction_required"].values())
        synd = sum(max(rp["synd_planned"][s] - rp["synd_done"][s], 0.0) for s in rp["synd_planned"])
        sect = ", ".join(f"{s} {rp['auction_required'][s] / 1e3:.1f}" for s in rp["auction_required"])
        return (f"UK: DMO remit report {rp['asof']:%d %b %Y}, £{left / 1e3:.1f}bn cash left to {rp['fy_end']:%b %Y}: "
                f"auctions £{auc / 1e3:.1f}bn ({sect}), syndications £{synd / 1e3:.1f}bn, unallocated "
                f"£{rp['unallocated'] / 1e3:.1f}bn by tender. The forward raises exactly that cash by sector (nominal is "
                f"higher where gilts price below par). After {rp['fy_end']:%b %Y} there is no remit: the forward repeats "
                f"the same months a year earlier, cash by sector and method.")
    if country == "DE":
        cal = [d_ for d_, _, _, a in DE_CALENDAR if a]
        cal_mn = sum(a for _, _, _, a in DE_CALENDAR if a)
        scale = de_2027_scale(ops)
        return (f"DE: Finanzagentur quarterly calendar to {pd.Timestamp(max(cal)):%d %b %Y} (€{cal_mn / 1e3:.1f}bn, "
                f"sizes published); 2027 is the 2026 gross × {scale:.3f}, from federal net borrowing "
                f"€{DE_NET_BORROWING[2026] / 1e3:.1f}bn (2026 budget) → €{DE_NET_BORROWING[2027] / 1e3:.1f}bn (2027 draft) "
                f"plus the change in coupon redemptions. A projection until the Finanzagentur's annual plan.")
    if country == "US":
        src = pd.Series([v[-1] for v in US_MONTHS.values()], index=list(US_MONTHS))
        label = {"refunding": "the latest refunding statement", "tbac": "TBAC's illustrative quarter",
                 "steady": "the same grid held"}
        spans = []
        for s, g in src.groupby(src, sort=False):
            a, b = pd.Timestamp(g.index[0]), pd.Timestamp(g.index[-1])
            spans.append(f"{label.get(s, s)} ({a:%b %Y}" + (f"–{b:%b %Y})" if b != a else ")"))
        return "US: auction sizes from " + ", then ".join(spans) + "; dates on Treasury's auction pattern."
    if country == "FR":
        coupons = ops[ops["sec_type"].astype(str) != "Bill"]
        year = FR_PROGRAMME_ASOF.year
        ytd = float(coupons[(coupons["op_dt"].dt.year == year) & (coupons["op_dt"] <= FR_PROGRAMME_ASOF)
                            & (coupons["amount_issued"] > 0)]["amount_issued"].sum())
        net_done = FR_PROGRAMME_MN * FR_DONE_SHARE
        left = FR_PROGRAMME_MN - net_done
        ratio = ytd / net_done if net_done else float("nan")
        n_ann = int(p["source"].astype(str).eq("calendar").sum()) if len(p) else 0
        atxt = f"; {n_ann} lines from live AFT next-issuance communiqués" if n_ann else ""
        return (f"FR: AFT programme €{FR_PROGRAMME_MN / 1e3:.0f}bn net for {year} ({FR_DONE_SHARE:.0%} done at "
                f"{FR_PROGRAMME_ASOF:%d %b}, €{left / 1e3:.1f}bn left) and €{FR_PROGRAMME_NEXT_MN / 1e3:.0f}bn for "
                f"{year + 1}, booked gross at this year's gross-to-net ratio ({ratio:.2f}, i.e. buybacks) in last "
                f"year's monthly shape{atxt}. A month already partly auctioned gets only what its target has left.")
    if country == "IT":
        it = ops[(ops["sec_type"].astype(str) != "Bill") & (ops["amount_issued"] > 0)]
        if "country" in it.columns:
            it = it[it["country"].astype(str).eq("IT")]
        ytd = float(it[(it["op_dt"] >= "2026-01-01") & (it["op_dt"] < "2026-09-01")]["amount_issued"].sum())
        retail = p[p["source"].astype(str).eq("announced retail")]["amount_mn"].sum() if len(p) else 0.0
        named = len([b for b in _IT_Q4 if b[2]])
        rtxt = (f", of which BTP Valore €{retail / 1e3:.1f}bn in {pd.Timestamp(_IT_Q4_RETAIL[0]):%b} (sized at the "
                "mean of the last two Valore issues until it prints)") if retail > 0 else ""
        n_ann = int(p["source"].astype(str).eq("calendar").sum()) if len(p) else 0
        atxt = f"; {n_ann} lines from live BdI offering announcements" if n_ann else ""
        return (f"IT: Q4 is MEF's quarterly programme, €{_IT_Q4_MN / 1e3:.0f}bn (midpoint of the published range){rtxt}{atxt}; "
                f"long BTPs, CCTeu and BTP€i at their trailing-year monthly volume, the {named} named bonds take the rest. "
                f"2027 is the 2026 Jan–Sep coupon pace in the trailing mix, plus last year's syndications and retail BTPs "
                f"a year on. Tape Jan–Aug coupon gross €{ytd / 1e3:.1f}bn against MEF's stated €261.8bn medium-long "
                f"(syndications and retail BTPs added from MEF communiqués).")
    if country == "ES":
        pace = _es_coupon_pace(ops, pd.Timestamp.today().normalize())
        n = int(p["amount_mn"].sum() / 1e3) if len(p) else 0
        n_ann = int(p["source"].astype(str).eq("calendar").sum()) if len(p) else 0
        atxt = (f"; {n_ann} lines on Tesoro calendar / BdE BON announcements"
                if n_ann else "")
        return (f"ES: Tesoro Bonos/Obligaciones at the trailing-12m coupon pace (€{pace / 1e3:.1f}bn/month) on "
                f"lines tapped in the last six months, first/third-Thursday auction days through "
                f"{HORIZON:%b %Y}{atxt}. Forward coupon gross in the paint €{n}bn. Letras stay in the cash "
                f"framework (net financing plan), not this coupon calendar.")
    return ""
