"""Monthly BTP / BOT cash ladder from the issuance tape, plus the forward ladder.

History (EUR mn, settlement month):
  gross       auction + syndication + retail cash at the allotment price (``issuance/cache/ops_it.parquet``
              with ``issuance.fetch_it.supplement_it`` applied: syndications and retail BTPs from MEF
              communiqués; then ``events.supplement`` for announced issues not yet on either).
              Foreign-currency globals (non-IT ISINs) are left out: they sit in other_fin.
  redemption  MEF 'Titoli in scadenza nei prossimi 12 mesi' (first month of the vintage published at the
              end of the previous month; BTP€i at indexed capital, 'Altre emissioni' excluded) where a
              vintage exists, else the tape ladder: everything issued on the tape by maturity month, at
              par, net of buybacks. Cash buybacks also redeem cash in their settlement month.
  exchanges   cash neutral at settlement. The issued leg's par is added to the line's maturity; the
              bought-back leg (rarely on the tape) reduces it when present. Legs above EXCHANGE_MAX_MN
              are rejected as tape errors.
  bills       BOT only: CTZ (24m zero coupons) are booked with bonds, as MEF and Banca d'Italia do.

Settlement: issue_dt, except when it is before the auction (reopenings carry the original issue date)
or more than SETTLE_MAX_DAYS after it (bad issue_dt on old syndications), then auction + 2 business days.
Rows with no maturity date are dropped.

Known gaps, all of which land in other_fin: lines issued before 2000 mature unseen through 2031;
BTP Italia before 2022 and BTP€i indexation are missing from the tape ladder (not from MEF months);
lines whose taps are missing from the tape (Jul/Aug-2026 lines are 3.3 / 5.1bn short of MEF).

Forward (asof = last month with the identity observed):
  bond gross   tape (incl. known events) for months already auctioned; ``issuance/cache/forecast_priced.parquet``
               (``issuance/forward.py``) after the last real tape auction, settlement = auction + 2 bdays.
  redemptions  latest MEF 12-month schedule ex 'Altre' (FX globals, booked in other_fin); tape beyond.
  bills        existing BOT maturities, then new BOT rolled at the trailing-12m tenor mix
               (``bot_tenor_mix``, ``model.hold_bills``).
  tagging      ``forward_bonds.attrs['items']``: syndications / retail settling in forward months, 'announced'
               (on the tape, a known event, or an announced forward row) or 'assumed' (forward.py's repeat of
               last year's syndications; a repeat whose size equals a past retail issue is tagged retail).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.it import events as ev

REPO = Path(__file__).resolve().parents[2]
OPS = REPO / "issuance" / "cache" / "ops_it.parquet"
FWD = REPO / "issuance" / "cache" / "forecast_priced.parquet"
QE_ISIN = REPO / "issuance" / "cache" / "official_isin_flow.parquet"
CASH_OPS = ("Auction", "Syndicate", "Syndication")
BILL_MAX_DAYS = 400
SETTLE_MAX_DAYS = 30
EXCHANGE_PAIR_DAYS = 3
# Exchange legs above this are tape errors, not exchanges (14.1bn of a 15.25bn CTZ in Nov-2023; 8bn of
# BTP 2051 in Oct-2020). Booking the CTZ one adds a 14bn May-2024 redemption Banca d'Italia does not see.
EXCHANGE_MAX_MN = 5000.0


def load_ops(path: Path = OPS, known_events: bool = True) -> pd.DataFrame:
    ops = pd.read_parquet(path)
    try:
        from issuance.fetch_it import supplement_it

        ops = supplement_it(ops)
    except Exception:  # noqa: BLE001 — the tape alone is still a ladder
        pass
    return ev.supplement(ops) if known_events else ops


def _month(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").dt.to_timestamp()


def classify(ops: pd.DataFrame) -> pd.DataFrame:
    """Settlement date, BOT flag, cash amount (EUR mn), exchange-leg flag per tape row."""
    o = ops.copy()
    for c in ("op_dt", "issue_dt", "maturity_dt"):
        o[c] = pd.to_datetime(o[c], errors="coerce")
    o = o[o["maturity_dt"].notna()].copy()
    t2 = o["op_dt"] + pd.offsets.BDay(2)
    lag = (o["issue_dt"] - o["op_dt"]).dt.days
    o["settle"] = o["issue_dt"].where((lag >= 0) & (lag <= SETTLE_MAX_DAYS), t2)
    tenor_days = (o["maturity_dt"] - o["settle"]).dt.days
    o["is_bill"] = o["sec_type"].astype(str).eq("Bill") & (tenor_days.fillna(0) <= BILL_MAX_DAYS)
    isin = o["isin"].astype("string").fillna("IT")
    o["domestic"] = isin.str.startswith("IT") | o["isin"].isna()
    px = pd.to_numeric(o["price"], errors="coerce").where(lambda s: (s > 50) & (s < 200), 100.0).fillna(100.0)
    o["cash_mn"] = pd.to_numeric(o["amount_issued"], errors="coerce").fillna(0.0) * px / 100.0
    o["op_type"] = o["op_type"].astype(str)
    o["known_event"] = o["source"].astype(str).eq(ev.KNOWN_EVENT) if "source" in o else False
    # A buyback within a few days of an exchange's issued leg is that exchange's bought-back leg.
    exch_dt = o.loc[o["op_type"].eq("Exchange"), "op_dt"].dropna().to_numpy()
    bb = o["op_type"].eq("Buyback")
    o["exchange_leg"] = o["op_type"].eq("Exchange")
    if len(exch_dt):
        near = o.loc[bb, "op_dt"].map(
            lambda d: bool(np.any(np.abs((exch_dt - np.datetime64(d)) / np.timedelta64(1, "D")) <= EXCHANGE_PAIR_DAYS)))
        o.loc[near[near].index, "exchange_leg"] = True
    return o


def monthly_ladder(ops: pd.DataFrame) -> pd.DataFrame:
    """Gross cash / tape redemptions per month for bonds and bills (EUR mn)."""
    o = classify(ops)
    o = o[o["domestic"]]
    issue = o[o["op_type"].isin(CASH_OPS) & (o["amount_issued"] > 0)]
    cash_bb = o[o["op_type"].eq("Buyback") & ~o["exchange_leg"]]
    exch = o[o["exchange_leg"] & (o["amount_issued"] <= EXCHANGE_MAX_MN)]
    book = pd.concat([issue, cash_bb, exch])
    bonds, bills = book[~book["is_bill"]], book[book["is_bill"]]
    out = pd.DataFrame({
        "bond_gross_mn": issue[~issue["is_bill"]].groupby(_month(issue.loc[~issue["is_bill"], "settle"]))["cash_mn"].sum(),
        "bill_gross_mn": issue[issue["is_bill"]].groupby(_month(issue.loc[issue["is_bill"], "settle"]))["cash_mn"].sum(),
        "bond_maturing_mn": bonds.groupby(_month(bonds["maturity_dt"]))["amount_issued"].sum(),
        "bill_redemption_mn": bills.groupby(_month(bills["maturity_dt"]))["amount_issued"].sum(),
        "buyback_mn": -cash_bb.groupby(_month(cash_bb["settle"]))["cash_mn"].sum(),
        "exchange_issued_mn": exch[exch["amount_issued"] > 0].groupby(
            _month(exch.loc[exch["amount_issued"] > 0, "settle"]))["amount_issued"].sum(),
    }).fillna(0.0)
    out["bond_redemption_mn"] = out["bond_maturing_mn"] + out["buyback_mn"]
    out.index.name = "month"
    return out.sort_index()


def mef_history(vintages: pd.DataFrame) -> pd.DataFrame:
    """Per month, MEF scheduled maturities from the vintage dated the end of the previous month (EUR mn).

    bond = BTP + BTP€i + BTP Italia + CCTeu (ex BOT, ex 'Altre'); other = 'Altre emissioni'.
    """
    if vintages is None or vintages.empty:
        return pd.DataFrame(columns=["bond_mef_mn", "bill_mef_mn", "other_mef_mn", "vintage"])
    v = vintages[vintages["month"] == vintages["vintage"] + pd.offsets.MonthBegin(1)]
    v = v.sort_values("vintage").drop_duplicates("month", keep="last").set_index("month")
    return pd.DataFrame({"bond_mef_mn": v["total_mn"] - v["bot_mn"] - v["other_mn"], "bill_mef_mn": v["bot_mn"],
                         "other_mef_mn": v["other_mn"], "vintage": v["vintage"]}).sort_index()


def forward_bonds(asof: pd.Timestamp, months: pd.DatetimeIndex, ops: pd.DataFrame,
                  fwd_path: Path = FWD) -> pd.Series:
    """Coupon-bond gross cash per forward month: tape where already auctioned, forward.py after."""
    o = classify(ops)
    o = o[o["domestic"] & o["op_type"].isin(CASH_OPS) & (o["amount_issued"] > 0) & ~o["is_bill"]]
    tape = o[o["settle"] > asof + pd.offsets.MonthEnd(0)]
    last_tape = o.loc[~o["known_event"], "op_dt"].max()
    f = pd.DataFrame()
    if fwd_path.exists():
        f = pd.read_parquet(fwd_path)
        f = f[(f["country"] == "IT") & (pd.to_datetime(f["op_dt"]) > last_tape)].copy()
        f["settle"] = pd.to_datetime(f["op_dt"]) + pd.offsets.BDay(2)
    # A known event that issuance/forward.py already books (same maturity month, same settlement
    # month) is counted once, from the forward.
    if len(f) and "maturity_dt" in f.columns:
        fkeys = set(zip(_month(pd.to_datetime(f["maturity_dt"], errors="coerce")), _month(f["settle"])))
        dup = tape["known_event"] & pd.Series(
            [k in fkeys for k in zip(_month(tape["maturity_dt"]), _month(tape["settle"]))], index=tape.index)
        tape = tape[~dup]
    tape_m = tape.groupby(_month(tape["settle"]))["cash_mn"].sum()
    out = pd.Series(0.0, index=months)
    for k, v in tape_m.items():
        if k in out.index:
            out[k] += v
    retail_sizes = _retail_sizes(o)
    items = []
    synd = tape["known_event"] | tape["op_type"].isin(("Syndicate", "Syndication"))
    for r in tape[synd].itertuples(index=False):
        kind = "retail" if _is_retail(r) else "syndication"
        items.append(_item(r.settle, r.maturity_dt, r.cash_mn, "announced", kind, _desc(r), r.isin))
    if len(f):
        px = pd.to_numeric(f.get("price", pd.Series(np.nan, index=f.index)), errors="coerce").fillna(100.0)
        f["cash_mn"] = f["amount_mn"] * px / 100.0
        fm = f.groupby(_month(f["settle"]))["cash_mn"].sum()
        for k, v in fm.items():
            if k in out.index:
                out[k] += v
        out.attrs["forward_last_month"] = str(fm.index.max().date()) if len(fm) else None
        srcs = f["source"].astype(str) if "source" in f else pd.Series("", index=f.index)
        for r, s in zip(f.itertuples(index=False), srcs):
            mat = pd.to_datetime(getattr(r, "maturity_dt", pd.NaT), errors="coerce")
            if s.startswith("announced"):
                items.append(_item(r.settle, mat, r.cash_mn, "announced", "retail" if "retail" in s else "syndication",
                                   "BTP Valore" if "valore" in str(r.isin).lower() else "", r.isin))
            elif s == "syndication pattern":
                name = next((n for v, n in retail_sizes if abs(r.amount_mn - v) < 1.0), "")
                items.append(_item(r.settle, mat, r.cash_mn, "assumed", "retail" if name else "syndication",
                                   name or str(getattr(r, "sec_type", "") or ""), r.isin))
    it = pd.DataFrame(items, columns=["month", "settle", "maturity", "cash_mn", "basis", "kind", "label", "id"])
    it = it[it["month"].isin(months)].sort_values("settle").reset_index(drop=True)
    out.attrs["tape_last_op"] = str(last_tape.date())
    out.attrs["items"] = it
    for b in ("announced", "assumed"):
        out.attrs[f"{b}_mn"] = it[it["basis"] == b].groupby("month")["cash_mn"].sum().reindex(months).fillna(0.0)
    out.attrs["known_events_mn"] = out.attrs["announced_mn"]
    return out


def _retail_sizes(o: pd.DataFrame) -> list[tuple[float, str]]:
    """(size, name) of every retail issue on the tape: forward.py repeats them as 'syndication pattern'."""
    return [(float(r.amount_issued), _desc(r)) for r in o[o["amount_issued"] > 0].itertuples(index=False)
            if _is_retail(r)]


def _is_retail(r) -> bool:
    t = f"{getattr(r, 'source', '')} {getattr(r, 'sec_desc', '')}".lower()
    return any(k in t for k in ("valore", "btp italia", "btp più", "btp piu", "retail"))


def _desc(r) -> str:
    t = f"{getattr(r, 'source', '')} {getattr(r, 'sec_desc', '')}".lower()
    if "valore" in t:
        return "BTP Valore"
    if "btp italia" in t:
        return "BTP Italia"
    if "green" in t:
        return "BTP Green"
    return "BTP€i" if str(getattr(r, "sec_type", "")) == "Linker" else ""


def _item(settle, maturity, cash_mn, basis, kind, desc, ident) -> dict:
    mat = pd.Timestamp(maturity) if pd.notna(maturity) else None
    desc = {"Linker": "BTP€i", "FRN": "CCTeu", "Nominal": "BTP", "": "BTP"}.get(desc, desc)
    label = f"{desc} {mat:%b-%Y}" if mat is not None else desc
    return {"month": pd.Timestamp(settle).to_period("M").to_timestamp(), "settle": pd.Timestamp(settle),
            "maturity": mat, "cash_mn": float(cash_mn), "basis": basis, "kind": kind,
            "label": f"{label} ({kind})", "id": str(ident)}


def known_bill_gross(asof: pd.Timestamp, months: pd.DatetimeIndex, ops: pd.DataFrame) -> pd.Series:
    """BOT cash already auctioned for forward months (NaN where the month's auctions are not on the tape)."""
    o = classify(ops)
    o = o[~o["known_event"]]
    b = o[o["is_bill"] & o["op_type"].isin(CASH_OPS) & (o["amount_issued"] > 0)
          & (o["settle"] > asof + pd.offsets.MonthEnd(0))]
    out = pd.Series(np.nan, index=months)
    if b.empty:
        return out
    g = b.groupby(_month(b["settle"]))["cash_mn"].sum()
    # A month counts as known only once the tape runs to its end.
    last = o["op_dt"].max()
    for k, v in g.items():
        if k in out.index and last >= k + pd.offsets.MonthEnd(0) - pd.Timedelta(days=3):
            out[k] = v
    return out


def bot_tenor_mix(ops: pd.DataFrame, asof: pd.Timestamp, months_back: int = 12) -> pd.Series:
    """Share of BOT gross by tenor in whole months over the trailing window."""
    o = classify(ops)
    b = o[o["is_bill"] & o["op_type"].isin(CASH_OPS) & (o["amount_issued"] > 0)
          & (o["settle"] > asof - pd.DateOffset(months=months_back)) & (o["settle"] <= asof + pd.offsets.MonthEnd(0))]
    if b.empty:
        return pd.Series({6: 0.5, 12: 0.5})
    ten = ((b["maturity_dt"] - b["settle"]).dt.days / 30.44).round().clip(lower=1).astype(int)
    s = b.groupby(ten)["amount_issued"].sum()
    return s / s.sum()


def forward_redemptions(months: pd.DatetimeIndex, scad: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    """Bond and BOT redemptions per forward month: MEF schedule (ex 'Altre') in its window, tape beyond."""
    out = pd.DataFrame(index=months, columns=["bond_redemption_mn", "bill_redemption_mn", "other_redemption_mn",
                                              "source"])
    s = scad.set_index("month") if len(scad) else pd.DataFrame()
    for mth in months:
        if mth in s.index:
            r = s.loc[mth]
            out.loc[mth] = [float(r["total_mn"] - r["bot_mn"] - r["other_mn"]), float(r["bot_mn"]),
                            float(r["other_mn"]), "MEF schedule"]
        else:
            b = ladder["bond_maturing_mn"].get(mth, 0.0)
            bl = ladder["bill_redemption_mn"].get(mth, 0.0)
            out.loc[mth] = [float(b), float(bl), 0.0, "tape ladder"]
    cols = ["bond_redemption_mn", "bill_redemption_mn", "other_redemption_mn"]
    out[cols] = out[cols].astype(float)
    return out


def isin_maturities(ops: pd.DataFrame) -> pd.Series:
    o = ops.assign(maturity_dt=pd.to_datetime(ops["maturity_dt"], errors="coerce")).dropna(subset=["isin", "maturity_dt"])
    return o.groupby(o["isin"].astype(str))["maturity_dt"].max()


def cb_redemptions(ops: pd.DataFrame, months: pd.DatetimeIndex, path: Path = QE_ISIN) -> pd.Series:
    """Eurosystem PSPP + PEPP holdings of Italian central-government ISINs maturing per month (EUR mn).

    Holdings at the end of the previous month (``issuance/qe_alloc.py`` allocation by ISIN); months past
    the last holdings snapshot use that snapshot (no reinvestment since Jul-2023 / Jan-2025). Non-IT ISINs
    (FX globals) and the non-central bucket are excluded. NaN before the first snapshot.
    """
    out = pd.Series(np.nan, index=months)
    if not path.exists():
        return out
    q = pd.read_parquet(path, columns=["month", "isin", "country", "program", "official_stock_mn", "resid_yrs"])
    q = q[(q["country"] == "IT") & q["program"].isin(["PSPP", "PEPP"])]
    q = q[q["isin"].astype(str).str.match(r"^IT\d")].copy()
    q["month"] = pd.to_datetime(q["month"])
    mat = isin_maturities(ops)
    q["maturity_dt"] = q["isin"].astype(str).map(mat)
    fb = q["month"] + pd.to_timedelta(q["resid_yrs"] * 365.25, unit="D")
    q["maturity_dt"] = q["maturity_dt"].fillna(fb)
    q["mat_m"] = _month(q["maturity_dt"])
    hold = q.groupby(["month", "mat_m"])["official_stock_mn"].sum()
    snaps = hold.index.get_level_values(0).unique().sort_values()
    for m in months:
        prev = snaps[snaps < m]
        if not len(prev):
            continue
        snap = prev[-1]
        out[m] = float(hold.loc[snap].get(m, 0.0))
    out.attrs["snapshot"] = str(snaps.max().date()) if len(snaps) else None
    return out
