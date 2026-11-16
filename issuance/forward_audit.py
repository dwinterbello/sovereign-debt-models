"""Line-by-line audit of the forward supply calendar.

Two tables:
  sleeve  per country x sleeve: last 12m on the tape against the forward 12m (face, auctions, size)
  line    per forward line: its taps against its own history (or its sleeve's), its end size against
          what lines of that tenor finish at, tap spacing, and what pins it down

Basis says what fixes the row, and grades conviction before any flag:
  announced   date, line and size, or date and line, published (DE/UK quarterly calendars, the US
              refunding statement, UK syndications)
  programme   the total is an official programme (UK remit, FR net programme, MEF quarterly range,
              US size guidance); line and date are modelled
  extrapolated beyond any published programme (DE 2027 repeats 2026, UK FY27-28 repeats FY26-27 H1,
              IT 2027 at the 2026 pace)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.paths import CACHE

SLEEVE = {"0-1y": "Short", "1-3y": "Short", "3-7y": "Short", "7-10y": "Medium", "10-15y": "Medium",
          "15-20y": "Long", "20-30y": "Long", "30y+": "Long"}
NON_COUPON = ("Bill", "Foreign", "Buyback")
# Published horizon of each country's programme; rows after it are extrapolated.
PROGRAMME_END = {"UK": "2027-03-31", "DE": "2026-12-31", "FR": "2027-12-31", "IT": "2026-12-31", "US": "2027-06-30", "JP": "2027-03-31"}
ANNOUNCED = {"calendar", "refunding", "syndication", "treasury"}
TAP_HI, TAP_LO, SIZE_HI, MIN_GAP_DAYS = 1.6, 0.5, 1.1, 7
GRADE = {"announced": "High", "programme": "Medium", "projection": "Low", "extrapolated": "Low"}
# Past the programme but the year's total is scaled to an official fiscal projection (not an
# issuance plan): DE 2027 to the Bundeshaushalt 2027 draft. Line and date stay modelled.
PROJECTION_END = {"DE": "2027-12-31"}
DOWNGRADE = {"High": "Medium", "Medium": "Low", "Low": "Low"}


def sleeve_of(bucket: pd.Series, sec: pd.Series) -> pd.Series:
    out = bucket.astype(str).map(SLEEVE)
    return out.where(sec.astype(str) != "Linker", "Index-linked")


def _tape(ops: pd.DataFrame) -> pd.DataFrame:
    from issuance.isin_flow import market_ops
    ops = market_ops(ops)
    o = ops[(ops["amount_issued"] > 0) & ~ops["sec_type"].astype(str).isin(NON_COUPON)].copy()
    o["op_dt"] = pd.to_datetime(o["op_dt"])
    # Forward US rows are announced sizes; SOMA add-ons sit on top, so compare the public take.
    if "official_take_mn" in o.columns:
        o["amount_issued"] = o["amount_issued"] - pd.to_numeric(o["official_take_mn"], errors="coerce").fillna(0.0).clip(lower=0)
    o["sleeve"] = sleeve_of(o["tenor_bucket"], o["sec_type"])
    return o


def _forward() -> pd.DataFrame:
    p = pd.read_parquet(CACHE / "forecast_priced.parquet")
    p = p[p["gross_mn"].fillna(0) > 0].copy()
    p["op_dt"] = pd.to_datetime(p["op_dt"])
    p["sleeve"] = sleeve_of(p["tenor_bucket"], p["sec_type"])
    return p


def _basis(country: str, source: str, isin: str, when: pd.Timestamp) -> str:
    if when > pd.Timestamp(PROGRAMME_END.get(country, "2026-12-31")):
        if source == "pattern" and when <= pd.Timestamp(PROJECTION_END.get(country, "1900-01-01")):
            return "projection"
        return "extrapolated"
    real = not (isin.startswith(("NEW", "next", "fc:")) or isin == "")
    if source in ANNOUNCED and (real or country == "US"):
        return "announced"
    return "programme"


def sleeve_audit(ops: pd.DataFrame, fwd: pd.DataFrame | None = None) -> pd.DataFrame:
    tape = _tape(ops)
    fwd = _forward() if fwd is None else fwd
    rows = []
    for c, o in tape.groupby("country"):
        end = o["op_dt"].max()
        trail = o[o["op_dt"] > end - pd.DateOffset(years=1)]
        prior = o[(o["op_dt"] <= end - pd.DateOffset(years=1)) & (o["op_dt"] > end - pd.DateOffset(years=2))]
        f = fwd[(fwd["country"] == c) & (fwd["op_dt"] > end) & (fwd["op_dt"] <= end + pd.DateOffset(years=1))]
        for s in ("Short", "Medium", "Long", "Index-linked"):
            a, b, g = trail[trail["sleeve"] == s], prior[prior["sleeve"] == s], f[f["sleeve"] == s]
            # The tape books a syndication or a split auction as several rows; count auction days.
            n_a, n_g = a["op_dt"].dt.normalize().nunique(), g["op_dt"].dt.normalize().nunique()
            rows.append({
                "country": c, "sleeve": s, "tape_end": end,
                "prior_bn": b["amount_issued"].sum() / 1e3, "trail_bn": a["amount_issued"].sum() / 1e3,
                "fwd_bn": g["gross_mn"].sum() / 1e3, "trail_days": n_a, "fwd_days": n_g,
                "trail_per_day_bn": a["amount_issued"].sum() / 1e3 / n_a if n_a else np.nan,
                "fwd_per_day_bn": g["gross_mn"].sum() / 1e3 / n_g if n_g else np.nan,
            })
    out = pd.DataFrame(rows)
    out["fwd_vs_trail"] = out["fwd_bn"] / out["trail_bn"].where(out["trail_bn"] > 0)
    return out


def _auctions(tape: pd.DataFrame) -> pd.DataFrame:
    """One row per auction: the tape's next-day supplementary (IT), PAOF (UK) and split rows join their auction."""
    t = tape.sort_values(["country", "isin", "op_dt"]).copy()
    new = t.groupby(["country", "isin"])["op_dt"].diff().dt.days.fillna(99) > 3
    t["auction"] = new.cumsum()
    return t.groupby("auction").agg(
        country=("country", "first"), isin=("isin", "first"), op_dt=("op_dt", "first"),
        amount_issued=("amount_issued", "sum"), tenor_bucket=("tenor_bucket", "last"),
        sec_type=("sec_type", "last"), sleeve=("sleeve", "last"), op_type=("op_type", "first"))


NOT_AUCTION_OP = ("Syndication", "Tender", "Liquidity Enhancement")
NOT_AUCTION_SRC = {"syndication", "syndication pattern", "tender", "lea"}


def line_audit(ops: pd.DataFrame, fwd: pd.DataFrame | None = None) -> pd.DataFrame:
    tape = _auctions(_tape(ops))
    fwd = _forward() if fwd is None else fwd
    end = tape.groupby("country")["op_dt"].max()
    # Sleeve tap size: the trailing year's median auction. Line ceiling: the largest line of that
    # sleeve still being tapped in the last three years (older finished lines are smaller).
    # Tap sizes compare auctions with auctions: a syndication or tender is several taps in one.
    auc = tape[~tape["op_type"].astype(str).isin(NOT_AUCTION_OP)]
    trail = auc[auc["op_dt"] > auc["country"].map(end) - pd.DateOffset(years=1)]
    sleeve_tap = trail.groupby(["country", "sleeve"])["amount_issued"].median()
    per_line = tape.groupby(["country", "isin"]).agg(
        stock_now=("amount_issued", "sum"), last_tap=("op_dt", "max"),
        bucket=("tenor_bucket", "last"), sec=("sec_type", "last"))
    per_auc = auc.groupby(["country", "isin"])["amount_issued"].agg(["size", "median"])
    per_line["n_taps"] = per_auc["size"].reindex(per_line.index).fillna(0)
    per_line["med_tap"] = per_auc["median"].reindex(per_line.index)
    # A new line of a tenor replaces that tenor's benchmark: the line that took the most in the
    # trailing year. Its taps, not the sleeve's off-the-run median, are the reference.
    vol = trail.groupby(["country", "tenor_bucket", "sec_type", "isin"])["amount_issued"].agg(["sum", "median"])
    bench = vol.sort_values("sum").groupby(level=[0, 1, 2]).tail(1).droplevel(3)["median"]
    per_line["sleeve"] = sleeve_of(per_line["bucket"], per_line["sec"])
    recent = per_line[per_line["last_tap"] >= per_line.index.get_level_values(0).map(end) - pd.DateOffset(years=3)]
    finished = recent.groupby(["country", "sleeve"])["stock_now"].max()
    rows, per_row = [], []
    for (c, isin), g in fwd.groupby(["country", "isin"]):
        g = g.sort_values("op_dt")
        sleeve = g["sleeve"].iloc[0]
        hist = per_line.loc[(c, isin)] if (c, isin) in per_line.index else None
        own = hist is not None and hist["n_taps"] >= 2
        new_line = str(isin).startswith(("NEW", "next"))
        sec_key = "Linker" if str(g["sec_type"].iloc[0]) == "Linker" else "Nominal"
        bench_tap = bench.get((c, str(g["tenor_bucket"].iloc[0]), sec_key), np.nan) if new_line else np.nan
        if own:
            ref_tap, ref_kind = float(hist["med_tap"]), "own"
        elif pd.notna(bench_tap):
            ref_tap, ref_kind = float(bench_tap), "benchmark"
        else:
            ref_tap, ref_kind = float(sleeve_tap.get((c, sleeve), np.nan)), "sleeve"
        stock_now = float(hist["stock_now"]) if hist is not None else 0.0
        fwd_face = float(g["gross_mn"].sum())
        sold = g[~g["source"].astype(str).isin(NOT_AUCTION_SRC)]
        avg_tap = float(sold["gross_mn"].sum()) / len(sold) if len(sold) else fwd_face / len(g)
        cap = float(finished.get((c, sleeve), np.nan))
        # A JGB liquidity enhancement auction reopens a line days after its own auction by design.
        gaps = g.loc[g["source"].astype(str) != "lea", "op_dt"].diff().dt.days.dropna()
        min_gap = float(gaps.min()) if len(gaps) else np.nan
        flags = []
        src = set(g["source"].astype(str))
        bases = pd.Series([_basis(c, str(s), str(isin), d) for s, d in zip(g["source"].astype(str), g["op_dt"])],
                          index=g.index)
        tap_ratio = avg_tap / ref_tap if ref_tap and ref_tap > 0 else np.nan
        # DE 2027 rows copy a 2026 auction.
        tap_checked = c != "US" and len(sold) > 0 and not (c == "DE" and src == {"pattern"})
        if tap_checked and pd.notna(tap_ratio) and not TAP_LO <= tap_ratio <= TAP_HI:
            flags.append(f"tap {tap_ratio:.1f}x {ref_kind} median")
        size_ratio = (stock_now + fwd_face) / cap if cap and cap > 0 else np.nan
        # Every tap announced by the issuer: the line's growth is theirs, not the model's.
        if c != "US" and pd.notna(size_ratio) and size_ratio > SIZE_HI and not bases.eq("announced").all():
            flags.append(f"ends {size_ratio:.2f}x largest live line")
        if c not in ("US", "UK") and not (c == "DE" and src == {"pattern"}) and pd.notna(min_gap) and min_gap < MIN_GAP_DAYS:
            flags.append(f"taps {min_gap:.0f}d apart")
        grades = bases.map(GRADE)
        if flags:
            grades = grades.map(DOWNGRADE)
        dv = g["dv01_usd_mn"].astype(float)
        by_grade = {k: float(dv[grades == k].sum()) for k in ("High", "Medium", "Low")}
        per_row.append(pd.DataFrame({"country": c, "isin": isin, "op_dt": g["op_dt"], "sleeve": sleeve,
                                     "amount_mn": g["gross_mn"].astype(float), "source": g["source"].astype(str),
                                     "basis": bases, "grade": grades}))
        rows.append({
            "country": c, "isin": isin, "sleeve": sleeve, "bucket": g["tenor_bucket"].iloc[-1],
            "sources": ", ".join(sorted(set(g["source"].astype(str)))),
            "basis": ", ".join(f"{k} {n}" for k, n in bases.value_counts().items()),
            "grade": max(by_grade, key=by_grade.get) if dv.sum() > 0 else grades.iloc[0],
            "dv01_high": by_grade["High"], "dv01_medium": by_grade["Medium"], "dv01_low": by_grade["Low"],
            "first": g["op_dt"].min(), "last": g["op_dt"].max(), "fwd_taps": len(g),
            "fwd_face_mn": fwd_face, "fwd_avg_tap_mn": avg_tap, "ref_tap_mn": ref_tap,
            "ref": ref_kind, "tap_ratio": tap_ratio, "stock_now_mn": stock_now,
            "stock_end_mn": stock_now + fwd_face, "largest_live_mn": cap, "size_ratio": size_ratio,
            "min_gap_days": min_gap, "fwd_dv01_usd": float(g["dv01_usd_mn"].sum()),
            "flags": "; ".join(flags),
        })
    out = pd.DataFrame(rows).sort_values(["country", "sleeve", "first"])
    out.attrs["rows"] = pd.concat(per_row, ignore_index=True) if per_row else pd.DataFrame()
    return out


def _bill_label(days: pd.Series) -> pd.Series:
    w = (days / 7).round()
    return np.where(days < 300, w.astype("Int64").astype(str) + "w", (days / 365).round().astype("Int64").astype(str) + "y")


def auction_sizes(ops: pd.DataFrame, rows: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """One row per auction (line x day), tape then forward, for the size-over-time dot plots.

    Bills have no forward model; their forward is last year's same auctions moved on 52 weeks."""
    names = ops.dropna(subset=["sec_desc_long"]).groupby("isin")["sec_desc_long"].last()
    tape = _auctions(_tape(ops))
    end = tape.groupby("country")["op_dt"].max()
    start = tape["country"].map(end) - pd.DateOffset(years=years)
    hist = tape[tape["op_dt"] > start].assign(phase="history", basis="tape", grade="")
    hist = hist.rename(columns={"amount_issued": "amount_mn"})
    hist["label"] = hist["isin"].map(names).fillna(hist["isin"])
    hist = hist[["country", "op_dt", "sleeve", "label", "amount_mn", "phase", "basis", "grade"]]

    b = ops[(ops["sec_type"].astype(str) == "Bill") & (ops["amount_issued"] > 0)
            & (ops["op_type"].astype(str) != "Buyback")].copy()
    b["op_dt"] = pd.to_datetime(b["op_dt"]).dt.normalize()
    b["days"] = (pd.to_datetime(b["maturity_dt"]) - b["op_dt"]).dt.days
    b = b[b["days"] > 0]
    b["label"] = "Bill " + pd.Series(_bill_label(b["days"]), index=b.index)
    b = b.groupby(["country", "op_dt", "label"], as_index=False)["amount_issued"].sum()
    b_end = b.groupby("country")["op_dt"].max()
    b = b[b["op_dt"] > b["country"].map(b_end) - pd.DateOffset(years=years)]
    bills = b.rename(columns={"amount_issued": "amount_mn"}).assign(
        sleeve="Bills", phase="history", basis="tape", grade="")

    fwd = pd.DataFrame()
    if not rows.empty:
        f = rows.copy()
        f["label"] = f["isin"].map(names).fillna(f["isin"])
        f = f.groupby(["country", "op_dt", "sleeve", "label"], as_index=False).agg(
            amount_mn=("amount_mn", "sum"), basis=("basis", "first"), grade=("grade", "first"))
        fwd = f.assign(phase="forward")
    horizon = fwd.groupby("country")["op_dt"].max() if not fwd.empty else pd.Series(dtype="datetime64[ns]")
    rep = []
    for c, g in b.groupby("country"):
        last = b_end[c]
        lag = g[g["op_dt"] > last - pd.Timedelta(days=364)].copy()
        stop = horizon.get(c, last + pd.Timedelta(days=364))
        k = 1
        while not lag.empty and last + pd.Timedelta(days=1) <= stop and k <= 3:
            r = lag.assign(op_dt=lag["op_dt"] + pd.Timedelta(days=364 * k))
            rep.append(r[(r["op_dt"] > last) & (r["op_dt"] <= stop)])
            k += 1
    if rep:
        rb = pd.concat(rep).rename(columns={"amount_issued": "amount_mn"})
        rb = rb.assign(sleeve="Bills", phase="forward", basis="repeat last year", grade="Low")
        fwd = pd.concat([fwd, rb[["country", "op_dt", "sleeve", "label", "amount_mn", "phase", "basis", "grade"]]])
    out = pd.concat([hist, bills, fwd], ignore_index=True)
    out = out[out["amount_mn"] > 0]
    out["amount_mn"] = out["amount_mn"].astype(float).round(1)
    return out.sort_values(["country", "op_dt"]).reset_index(drop=True)


def grade_mix(lines: pd.DataFrame) -> pd.DataFrame:
    """Forward DV01 share by conviction grade (auction by auction), per country."""
    g = lines.groupby("country")[["dv01_high", "dv01_medium", "dv01_low"]].sum()
    g.columns = ["High", "Medium", "Low"]
    return g.div(g.sum(axis=1), axis=0).fillna(0.0)


def write_audit(ops: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fwd = _forward()
    sl, ln = sleeve_audit(ops, fwd), line_audit(ops, fwd)
    sl.to_parquet(CACHE / "forward_audit_sleeve.parquet", index=False)
    rows = ln.attrs.pop("rows", pd.DataFrame())
    ln.to_parquet(CACHE / "forward_audit_line.parquet", index=False)
    auction_sizes(ops, rows).to_parquet(CACHE / "auction_sizes.parquet", index=False)
    mix = grade_mix(ln)
    print("  forward DV01 by grade", mix.round(2).to_dict("index"))
    print("  flagged lines", ln[ln["flags"] != ""].groupby("country").size().to_dict())
    return {"sleeve": sl, "line": ln}
