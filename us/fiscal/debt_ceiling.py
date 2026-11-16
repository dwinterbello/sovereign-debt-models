"""Debt subject to the limit, extraordinary measures (GAS), contact / X-date.

DTS Table IIIC publishes the statutory debt limit and the stock categories that
rebuild debt subject to the limit (STL). Table IIIA's Government Account Series
is the intragovernmental book Treasury suspends / redeems under extraordinary
measures (EM). Neither is in the TBAC-steered TGA path.

Contact date: project STL forward on the fiscal path's net marketable issuance
plus the trailing intragovernmental pace; first day STL ≥ limit.

After contact, net marketable issuance stops. Cash runway = TGA + EM headroom,
spent at the path's daily cash need (op deficit − coupon redemptions that still
roll). X-date = first day runway ≤ TGA floor.

EM headroom is bottom-up from the last debt-issuance-suspension episode:
peak-to-trough drop in Intragovernmental Holdings while headroom was pinned at
the limit (GAS / CSRDF / G-Fund etc. freed into borrowing capacity), floored at
a desk minimum. SLGS suspension is inside that intragov move.

Seasonal TGA overlay: prior-two-FY median same-day shape anchored at the TBAC
target — a pure seasonal counterfactual beside the production path, which after
the waypoint uses `tga_forecast`'s total-bills reaction (seasonal gap + rolling Δ).

  PYTHONPATH=. .venv/bin/python -m us.fiscal.debt_ceiling
  PYTHONPATH=us/fiscal:. .venv/bin/python debt_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import financing  # noqa: E402
from tga_deficit import (  # noqa: E402
    DERIVED,
    DTS_DEBT,
    _fiscaldata_pages,
    _to_float,
    fetch_tga_mn,
    fy_day_of_year,
    us_fiscal_year,
)

# Desk floor when the last suspension sample is thin ($bn).
EM_FLOOR_BN = 250.0
# TGA not drained below this in the X-date path ($bn) — operational minimum.
TGA_FLOOR_BN = 50.0
# Headroom ($bn) treated as "at the limit" for EM calibration windows.
AT_LIMIT_BN = 5.0


def fetch_debt_limit_raw(*, refresh: bool = False) -> pd.DataFrame:
    return _fiscaldata_pages(
        DTS_DEBT,
        fields="record_date,debt_catg,close_today_bal",
        cache_name="dts_debt_subject_to_limit.csv",
        refresh=refresh,
    )


def parse_debt_limit_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Daily STL / statutory limit / headroom / intragov ($bn)."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["record_date"])
    df["v"] = df["close_today_bal"].map(_to_float) / 1000.0
    df["debt_catg"] = df["debt_catg"].astype(str).str.strip()

    def _day(g: pd.DataFrame) -> pd.Series:
        pub = float(g.loc[g["debt_catg"] == "Debt Held by the Public", "v"].sum())
        ig = float(g.loc[g["debt_catg"] == "Intragovernmental Holdings", "v"].sum())
        notsubj = float(g.loc[g["debt_catg"] == "Debt Not Subject to Limit", "v"].sum())
        less = float(g.loc[g["debt_catg"] == "Less Debt Not Subject to Limit", "v"].sum())
        other = float(
            g.loc[
                g["debt_catg"].isin(
                    ["Other Debt Subject to Limit", "Plus Other Debt Subject to Limit"]
                ),
                "v",
            ].sum()
        )
        limit = float(g.loc[g["debt_catg"] == "Statutory Debt Limit", "v"].sum())
        stl = pub + ig - notsubj - less + other
        return pd.Series(
            {
                "debt_public_stl_bn": pub,
                "intragov_bn": ig,
                "debt_not_subject_bn": notsubj + less,
                "other_subject_bn": other,
                "stl_bn": stl,
                "limit_bn": limit if limit > 0 else np.nan,
                "headroom_bn": (limit - stl) if limit > 0 else np.nan,
            }
        )

    return df.groupby("date", sort=True).apply(_day, include_groups=False)


def gas_daily(iiia: pd.DataFrame) -> pd.DataFrame:
    """Government Account Series issues / redemptions / net ($bn)."""
    d = iiia.copy()
    d["date"] = pd.to_datetime(d["record_date"])
    d["amt"] = pd.to_numeric(d["transaction_today_amt"], errors="coerce").fillna(0.0) / 1000.0
    d = d[d["security_type"] == "Government Account Series"]
    if d.empty:
        return pd.DataFrame(columns=["gas_issues_bn", "gas_redemptions_bn", "gas_net_bn"])
    iss = d.loc[d["transaction_type"] == "Issues"].groupby("date")["amt"].sum()
    red = d.loc[d["transaction_type"] == "Redemptions"].groupby("date")["amt"].sum()
    out = pd.DataFrame({"gas_issues_bn": iss, "gas_redemptions_bn": red}).fillna(0.0)
    out["gas_net_bn"] = out["gas_issues_bn"] - out["gas_redemptions_bn"]
    return out.sort_index()


def em_capacity_from_history(limit_daily: pd.DataFrame) -> dict:
    """Peak-to-trough intragov drop while headroom was pinned at the limit ($bn)."""
    d = limit_daily.dropna(subset=["headroom_bn", "intragov_bn"]).sort_index()
    at = d["headroom_bn"] <= AT_LIMIT_BN
    # Contiguous runs of at-limit days ending after 2018 (modern EM toolkit).
    runs: list[pd.DataFrame] = []
    if at.any():
        grp = (~at).cumsum()
        for _, g in d[at].groupby(grp[at]):
            if len(g) >= 20 and g.index.min() >= pd.Timestamp("2018-01-01"):
                runs.append(g)
    drops = []
    for g in runs:
        drops.append(
            {
                "start": str(g.index.min().date()),
                "end": str(g.index.max().date()),
                "days": int(len(g)),
                "ig_drop_bn": float(g["intragov_bn"].max() - g["intragov_bn"].min()),
            }
        )
    best = max((r["ig_drop_bn"] for r in drops), default=0.0)
    capacity = max(EM_FLOOR_BN, best)
    return {
        "em_capacity_bn": capacity,
        "em_floor_bn": EM_FLOOR_BN,
        "em_from_intragov_drop_bn": best,
        "em_calibration_runs": drops[-3:],
        "em_source": (
            f"intragov peak-to-trough ${best:.0f}bn in last at-limit run"
            if best >= EM_FLOOR_BN
            else f"desk floor ${EM_FLOOR_BN:.0f}bn (last run drop ${best:.0f}bn)"
        ),
    }


def seasonal_tga_path(
    tga_hist: pd.Series,
    steered: pd.Series,
    target_date: pd.Timestamp,
    target_bn: float,
) -> pd.Series:
    """Steered path through the TBAC date; then prior-2-FY median same-day shape."""
    steered = steered.dropna().sort_index()
    if steered.empty:
        return steered
    hist = tga_hist.dropna().sort_index()
    # Build FY-day → level panel for the two complete FYs before the target's FY.
    tgt_fy = us_fiscal_year(target_date)
    frames = []
    for fy in (tgt_fy - 2, tgt_fy - 1):
        sl = hist[(hist.index.map(us_fiscal_year) == fy)]
        if sl.empty:
            continue
        frames.append(pd.Series(sl.to_numpy(), index=sl.index.map(fy_day_of_year), name=fy))
    if not frames:
        return steered.copy()
    panel = pd.concat(frames, axis=1)
    med = panel.median(axis=1)
    # Anchor: difference from the FY-day of the target date.
    anchor_doy = int(fy_day_of_year(target_date))
    if anchor_doy not in med.index or pd.isna(med.loc[anchor_doy]):
        # nearest available doy
        if med.dropna().empty:
            return steered.copy()
        anchor_doy = int(med.dropna().index[np.argmin(np.abs(med.dropna().index.to_numpy() - anchor_doy))])
    base = float(med.loc[anchor_doy])
    shape = med - base  # 0 at target FY-day

    out = steered.copy()
    after = out.index[out.index > target_date]
    for dt in after:
        doy = int(fy_day_of_year(dt))
        if doy in shape.index and pd.notna(shape.loc[doy]):
            out.loc[dt] = target_bn + float(shape.loc[doy])
        else:
            # hold last seasonal point
            out.loc[dt] = out.loc[:dt].iloc[-2] if len(out.loc[:dt]) > 1 else target_bn
    # Through target date keep the steered path (lands on target_bn).
    return out


def _net_issuance_bn(path: pd.DataFrame) -> pd.Series:
    """Daily change in marketable debt face proxy from the fiscal path ($bn)."""
    bills = path["net_bills_total_bn"] if "net_bills_total_bn" in path.columns else path.get(
        "net_bills", pd.Series(0.0, index=path.index)
    )
    coupons = path["net_coupons"] if "net_coupons" in path.columns else pd.Series(0.0, index=path.index)
    return bills.fillna(0.0) + coupons.fillna(0.0)


def project(
    limit_daily: pd.DataFrame,
    fiscal_path: pd.DataFrame,
    tga_hist: pd.Series,
    *,
    em_info: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Forward STL / contact / X-date / seasonal TGA on the unchanged fiscal scenario."""
    hist = limit_daily.dropna(subset=["stl_bn", "limit_bn"]).sort_index()
    if hist.empty:
        raise RuntimeError("empty debt-limit history")
    origin = hist.index.max()
    last = hist.iloc[-1]
    em_info = em_info or em_capacity_from_history(hist)
    em_cap = float(em_info["em_capacity_bn"])

    fp = fiscal_path.copy()
    if isinstance(fp.index, pd.MultiIndex):
        fp = fp.xs("unchanged")
    fp = fp[fp.index > origin].sort_index()
    if fp.empty:
        raise RuntimeError("fiscal path has no days past the debt-limit origin")

    # Trailing 60-business-day intragov daily change.
    ig = hist["intragov_bn"].dropna()
    ig_d = ig.diff().dropna().iloc[-60:]
    ig_pace = float(ig_d.mean()) if len(ig_d) else 0.0

    net = _net_issuance_bn(fp)
    stl = float(last["stl_bn"]) + (net + ig_pace).cumsum()
    limit = float(last["limit_bn"])
    headroom = limit - stl

    # Production steered TGA on the path.
    steered = fp["tga_bn"].astype(float)
    # TBAC target from path summary / first date where steered flattens — use tga_forecast targets.
    import tga_forecast as tf

    tgt = tf._next_target(origin)
    t_date, t_val = tgt if tgt is not None else (tf._quarter_end_after(origin), float(steered.iloc[0]))
    seasonal = seasonal_tga_path(tga_hist, steered, t_date, t_val)

    # Contact: first projected day with headroom ≤ 0.
    contact_idx = headroom[headroom <= 0]
    contact = pd.Timestamp(contact_idx.index[0]) if len(contact_idx) else None

    # Post-contact: no net marketable issuance → STL held at limit. TGA drains on
    # op_flow + drift only. Do NOT splice tga_rollover: that path still carries
    # net_coupons / net_bills and overstates cash under the ceiling by hundreds of $bn.
    tga_ceil = steered.copy()
    runway = pd.Series(np.nan, index=fp.index)
    x_date = None
    summary_burn = None
    tga_at = None
    if contact is not None:
        op = (fp["op_flow_bn"].astype(float).fillna(0.0)
              if "op_flow_bn" in fp.columns else pd.Series(0.0, index=fp.index))
        drift = (fp["drift_bn"].astype(float).fillna(0.0)
                 if "drift_bn" in fp.columns else pd.Series(0.0, index=fp.index))
        cash = (op + drift).reindex(fp.index).fillna(0.0)
        pre = steered.loc[steered.index < contact]
        if len(pre):
            tga_at = float(pre.iloc[-1])
        else:
            tga_at = float(tga_hist.dropna().iloc[-1])
        post = fp.index >= contact
        tga_ceil.loc[post] = tga_at + cash.loc[post].cumsum()
        # Cash need ≈ −ΔTGA on the no-issuance path when TGA is falling, else op deficit.
        # Daily burn after contact: path-mean operating deficit (surplus days net in).
        # Conditional-on-outflow means burn too fast vs the usual 6–9m EM runway.
        if "op_flow_bn" in fp.columns:
            burn = max(0.0, -float(fp["op_flow_bn"].mean()))
        else:
            burn = max(0.0, -float(tga_ceil.diff().mean()))
        burn = max(burn, 0.5)
        resources = tga_at + em_cap
        after = fp.index[fp.index >= contact]
        rem = resources
        for dt in after:
            rem = rem - burn
            runway.loc[dt] = rem
            if x_date is None and rem <= TGA_FLOOR_BN:
                x_date = pd.Timestamp(dt)
                break
        summary_burn = burn

    out = pd.DataFrame(
        {
            "stl_bn": stl,
            "limit_bn": limit,
            "headroom_bn": headroom,
            "net_issuance_bn": net,
            "intragov_pace_bn": ig_pace,
            "tga_steered_bn": steered,
            "tga_seasonal_bn": seasonal,
            "tga_ceiling_bn": tga_ceil,
            "runway_bn": runway,
        },
        index=fp.index,
    )
    out.index.name = "date"

    summary = {
        "origin": str(origin.date()),
        "stl_bn": float(last["stl_bn"]),
        "limit_bn": limit,
        "headroom_bn": float(last["headroom_bn"]),
        "intragov_bn": float(last["intragov_bn"]),
        "intragov_pace_bn_day": ig_pace,
        "tga_bn": float(tga_hist.dropna().iloc[-1]),
        "target_date": str(t_date.date()),
        "target_bn": float(t_val),
        "contact_date": str(contact.date()) if contact is not None else None,
        "x_date": str(x_date.date()) if x_date is not None else None,
        "tga_at_contact_bn": tga_at,
        "tga_floor_bn": TGA_FLOOR_BN,
        "post_contact_burn_bn_day": float(summary_burn) if contact is not None else None,
        **em_info,
    }
    if contact is not None:
        summary["days_to_contact"] = int((contact - origin).days)
    if x_date is not None and contact is not None:
        summary["days_contact_to_x"] = int((x_date - contact).days)
    return out, summary


def run(*, refresh: bool = False, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    raw = fetch_debt_limit_raw(refresh=refresh)
    daily = parse_debt_limit_daily(raw)
    daily.to_parquet(DERIVED / "debt_limit_daily.parquet")

    iiia = financing.fetch_iiia(refresh=refresh)
    gas = gas_daily(iiia)
    gas.to_parquet(DERIVED / "gas_daily.parquet")

    em_info = em_capacity_from_history(daily)
    tga = fetch_tga_mn(refresh=False) / 1000.0

    fp_path = DERIVED / "fiscal_path.parquet"
    if not fp_path.exists():
        raise FileNotFoundError(f"Missing {fp_path}; run fiscal_path first")
    fp = pd.read_parquet(fp_path)
    forward, summary = project(daily, fp, tga, em_info=em_info)
    forward.to_parquet(DERIVED / "debt_ceiling_forward.parquet")
    # Attach recent GAS net onto the history file for the board.
    hist = daily.copy()
    hist = hist.join(gas.reindex(hist.index), how="left")
    hist.to_parquet(DERIVED / "debt_limit_daily.parquet")
    (DERIVED / "debt_ceiling_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if verbose:
        c = summary.get("contact_date") or "none"
        x = summary.get("x_date") or "none"
        print(
            f"debt ceiling: STL ${summary['stl_bn']:.0f}bn / limit ${summary['limit_bn']:.0f}bn "
            f"(headroom ${summary['headroom_bn']:.0f}bn)  EM ${summary['em_capacity_bn']:.0f}bn  "
            f"contact {c}  X-date {x}"
        )
    return forward, summary


if __name__ == "__main__":
    run(refresh="--refresh" in sys.argv)
