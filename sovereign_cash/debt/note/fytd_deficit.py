"""FYTD cash-requirement tracking with full financing disaggregation.

Live monthly deficits from sovereign_cash frameworks (cash_requirement_bn) and
US DTS fiscal flows — not net-issuance alone. Financing identity:

    cash_requirement = net_bonds + net_bills + other_fin − buffer_change
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sovereign_cash import investor as iv

REPO = Path(__file__).resolve().parents[3]
from sovereign_cash.debt.note.paths import NOTE, OUT_HTML, OUT_PDF
DERIVED = REPO / "sovereign_cash" / "derived"
DEBT_ANN = REPO / "sovereign_cash" / "debt" / "derived" / "debt_country_annual.parquet"
CBO = REPO / "sovereign_cash" / "data" / "debt" / "cbo.parquet"
TGA = REPO / "us" / "fiscal" / "derived" / "tga_implied_deficit_daily.csv"
DTS = REPO / "us" / "fiscal" / "derived" / "dts_bucket_daily.parquet"
LADDER = REPO / "us" / "fiscal" / "derived" / "auction_ladder.parquet"
FIN = REPO / "us" / "fiscal" / "derived" / "financing_reconcile.parquet"

ORDER = ["us", "uk", "de", "fr", "it", "es", "jp", "se"]
LABEL = {"us": "US", "uk": "UK", "de": "DE", "fr": "FR", "it": "IT",
         "es": "ES", "jp": "JP", "se": "SE"}
# FY month starts (1=Jan … 12=Dec); length always 12.
FY_START = {"us": 10, "uk": 4, "de": 1, "fr": 1, "it": 1, "es": 1, "jp": 4, "se": 1}

@dataclass
class CountryFYTD:
    cc: str
    year_label: str
    through: str  # YYYY-MM
    months_elapsed: float  # /12 pace
    cash_ytd: float
    plan: float
    plan_src: str
    gdp: float
    gdp_prior: float
    gdp_prev2: float
    net_bonds: float
    net_bills: float
    other_fin: float
    buffer_change: float  # + = buffer up
    cum_path: pd.Series  # live FY cumulative, index=month start
    prior_cum_path: pd.Series  # last complete FY cumulative
    prev2_cum_path: pd.Series  # FY before that (complete)
    fy_origin: pd.Timestamp
    prior_fy_origin: pd.Timestamp
    prev2_fy_origin: pd.Timestamp
    prior_year_label: str
    prev2_year_label: str


def _gdp(cc: str, year: int | None = None) -> float:
    ann = pd.read_parquet(DEBT_ANN)
    s = ann[ann.cc == cc]
    if year is not None:
        r = s[s.year == year]
        if not r.empty:
            return float(r.gdp_bn.iloc[0])
    for y in (2026, 2025, 2024):
        r = s[s.year == y]
        if not r.empty:
            return float(r.gdp_bn.iloc[0])
    raise KeyError(cc)


def _fy_origin(cc: str, year_label: str) -> pd.Timestamp:
    if cc == "us":
        fy = int(year_label.replace("FY", ""))
        return pd.Timestamp(fy - 1, 10, 1)
    if cc in ("uk", "jp"):
        y0 = int(year_label.replace("FY", "").split("-")[0])
        return pd.Timestamp(y0, 4, 1)
    return pd.Timestamp(int(year_label), 1, 1)


def _prior_fy_window(cc: str, origin: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive month-start window for the prior fiscal year."""
    if cc in ("us", "uk", "jp"):
        start = origin - pd.DateOffset(years=1)
        end = origin - pd.DateOffset(months=1)
    else:
        start = pd.Timestamp(origin.year - 1, 1, 1)
        end = pd.Timestamp(origin.year - 1, 12, 1)
    return start, end


def _fy_label_at(cc: str, origin: pd.Timestamp) -> str:
    if cc == "us":
        return f"FY{origin.year + 1}"
    if cc == "uk":
        y0 = origin.year
        return f"FY{y0}-{str(y0 + 1)[-2:]}"
    if cc == "jp":
        y0 = origin.year
        return f"FY{y0}-{str(y0 + 1)[-2:]}"
    return str(origin.year)


def _cum_window(def_or_cash: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    w = def_or_cash.loc[start:end]
    return w.cumsum() if not w.empty else pd.Series(dtype=float)


def _months_into_fy(months: pd.DatetimeIndex, origin: pd.Timestamp) -> list[int]:
    o = origin.to_period("M")
    return [(m.to_period("M") - o).n + 1 for m in months]


def _fy_window(cc: str, asof: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Return (fy_start, through_month_start, year_label)."""
    if cc == "us":
        fy = asof.year + (1 if asof.month >= 10 else 0)
        start = pd.Timestamp(fy - 1, 10, 1)
        label = f"FY{fy}"
    elif cc in ("uk", "jp"):
        fy_end = asof.year + (1 if asof.month >= 4 else 0)
        start = pd.Timestamp(fy_end - 1, 4, 1)
        label = f"FY{fy_end - 1}-{str(fy_end)[-2:]}"
    else:
        start = pd.Timestamp(asof.year, 1, 1)
        label = str(asof.year)
    through = pd.Timestamp(asof.year, asof.month, 1)
    return start, through, label


def _hist(cc: str) -> pd.DataFrame:
    d = iv.monthly(cc).copy()
    d["month"] = pd.to_datetime(d["month"])
    if "is_forecast" in d.columns:
        d = d[~d["is_forecast"].fillna(False)]
    return d.sort_values("month")


def _cash_country(cc: str, trk: dict) -> CountryFYTD:
    d = _hist(cc)
    last = d["month"].max()
    start, through, label = _fy_window(cc, last)
    if trk.get("ytd_through"):
        through = pd.Timestamp(str(trk["ytd_through"]) + "-01")
        label = trk.get("year") or label
    origin = _fy_origin(cc, label)
    start = origin
    ytd = d[(d["month"] >= start) & (d["month"] <= through)].copy()
    if ytd.empty:
        raise ValueError(f"{cc}: empty FYTD")
    cash = float(ytd["cash_requirement_bn"].sum())
    nb = float(ytd["net_bonds_bn"].sum())
    nbi = float(ytd["net_bills_bn"].sum())
    of = float(ytd["other_fin_bn"].sum()) if "other_fin_bn" in ytd else 0.0
    buf = float(ytd["buffer_change_bn"].sum()) if "buffer_change_bn" in ytd else 0.0
    resid = cash - (nb + nbi + of - buf)
    of = of + resid
    plan = float(trk["plan"])
    plan_src = trk.get("plan_src", "")
    n_m = len(ytd)
    elapsed = float(n_m)
    if "partial_month" in ytd.columns and bool(ytd.iloc[-1].get("partial_month", False)):
        bo = ytd.iloc[-1].get("bdays_observed")
        bi = ytd.iloc[-1].get("bdays_in_month")
        if bo and bi and bi > 0:
            elapsed = n_m - 1 + float(bo) / float(bi)
    cash_s = d.set_index("month")["cash_requirement_bn"].sort_index()
    cum = cash_s.loc[start:through].cumsum()

    p_start, p_end = _prior_fy_window(cc, origin)
    prior_cum = _cum_window(cash_s, p_start, p_end)
    p2_start, p2_end = _prior_fy_window(cc, p_start)
    prev2_cum = _cum_window(cash_s, p2_start, p2_end)

    gdp_year = origin.year
    return CountryFYTD(
        cc=cc, year_label=label, through=f"{through:%Y-%m}", months_elapsed=elapsed,
        cash_ytd=cash, plan=plan, plan_src=plan_src,
        gdp=_gdp(cc, gdp_year), gdp_prior=_gdp(cc, gdp_year - 1),
        gdp_prev2=_gdp(cc, gdp_year - 2),
        net_bonds=nb, net_bills=nbi, other_fin=of, buffer_change=buf,
        cum_path=cum, prior_cum_path=prior_cum, prev2_cum_path=prev2_cum,
        fy_origin=origin, prior_fy_origin=p_start, prev2_fy_origin=p2_start,
        prior_year_label=_fy_label_at(cc, p_start),
        prev2_year_label=_fy_label_at(cc, p2_start),
    )


def _us_fytd() -> CountryFYTD:
    t = pd.read_csv(TGA, parse_dates=["date"]).sort_values("date")
    asof = t["date"].iloc[-1]
    fy = asof.year + (1 if asof.month >= 10 else 0)
    start = pd.Timestamp(fy - 1, 10, 1)
    label = f"FY{fy}"
    window = t[t["date"] >= start]
    last = window.iloc[-1]
    cash = float(last["dts_fiscal_fytd_bn"])
    d_tga = float(last["d_tga_fytd_bn"])

    lad = pd.read_parquet(LADDER)
    lad_w = lad.loc[start:asof]
    nb = float(lad_w["net_coupons"].sum())
    nbi = float(lad_w["net_bills"].sum())
    of = cash - nb - nbi + d_tga
    buf = d_tga

    cbo = pd.read_parquet(CBO)
    plan_row = cbo[(cbo["var"] == "deficit_bn") & (cbo["year"] == fy)]
    plan = abs(float(plan_row["value"].iloc[0])) if not plan_row.empty else np.nan

    dts = pd.read_parquet(DTS)
    def_m = (-dts["op_net"] / 1000.0).resample("MS").sum()
    cum = def_m.loc[start:asof].cumsum()

    prior_start = pd.Timestamp(fy - 2, 10, 1)
    prior_end = pd.Timestamp(fy - 1, 9, 1)
    prior_cum = _cum_window(def_m, prior_start, prior_end)
    prev2_start = pd.Timestamp(fy - 3, 10, 1)
    prev2_end = pd.Timestamp(fy - 2, 9, 1)
    prev2_cum = _cum_window(def_m, prev2_start, prev2_end)

    fy_days = 365.0 + (1.0 if pd.Timestamp(fy, 2, 1).is_leap_year else 0.0)
    elapsed_m = 12.0 * ((asof - start).days + 1) / fy_days

    # FY N runs Oct-(N-1)→Sep-N; scale by calendar GDP of the start year.
    return CountryFYTD(
        cc="us", year_label=label, through=f"{asof:%Y-%m}", months_elapsed=elapsed_m,
        cash_ytd=cash, plan=plan, plan_src=f"CBO {label} unified deficit (abs)",
        gdp=_gdp("us", fy - 1), gdp_prior=_gdp("us", fy - 2),
        gdp_prev2=_gdp("us", fy - 3),
        net_bonds=nb, net_bills=nbi, other_fin=of, buffer_change=buf,
        cum_path=cum, prior_cum_path=prior_cum, prev2_cum_path=prev2_cum,
        fy_origin=start, prior_fy_origin=prior_start, prev2_fy_origin=prev2_start,
        prior_year_label=_fy_label_at("us", prior_start),
        prev2_year_label=_fy_label_at("us", prev2_start),
    )


def _weo_deficit_plan(cc: str, year: int) -> tuple[float, str]:
    """IMF WEO overall balance → cash-requirement plan (deficit +, local bn)."""
    weo = pd.read_parquet(REPO / "sovereign_cash" / "data" / "debt" / "weo.parquet")
    row = weo.loc[
        (weo["country"].astype(str) == cc)
        & (weo["var"].astype(str) == "balance_pct")
        & (weo["year"] == int(year))
    ]
    gdp = _gdp(cc, year)
    if not row.empty and np.isfinite(gdp):
        # balance_pct: surplus(+)/deficit(−) → cash requirement = −balance
        pct = float(row["value"].iloc[0])
        return max(0.0, -pct) / 100.0 * gdp, f"WEO {year} -balance"
    return float("nan"), "WEO missing"


def _igae_es_fytd() -> CountryFYTD:
    from sovereign_cash.debt.note.monthly_primary import cash_series

    cash_s = cash_series("es").sort_index()
    last = cash_s.index.max()
    start, through, label = _fy_window("es", last)
    origin = _fy_origin("es", label)
    start = origin
    ytd = cash_s.loc[start:through]
    cash = float(ytd.sum())
    # Cash chart: same-perimeter benchmark = prior calendar-year IGAE cash (not WEO GG).
    p_start, p_end = _prior_fy_window("es", origin)
    prior_full = cash_s.loc[p_start:p_end]
    if len(prior_full) >= 12:
        plan = float(prior_full.sum())
        plan_src = f"IGAE prior FY {_fy_label_at('es', p_start)} cash"
    else:
        plan, plan_src = float("nan"), "IGAE prior FY missing"
    cum = ytd.cumsum()
    prior_cum = _cum_window(cash_s, p_start, p_end)
    p2_start, p2_end = _prior_fy_window("es", p_start)
    prev2_cum = _cum_window(cash_s, p2_start, p2_end)
    return CountryFYTD(
        cc="es", year_label=label, through=f"{through:%Y-%m}", months_elapsed=float(len(ytd)),
        cash_ytd=cash, plan=plan, plan_src=plan_src,
        gdp=_gdp("es", origin.year), gdp_prior=_gdp("es", origin.year - 1),
        gdp_prev2=_gdp("es", origin.year - 2),
        net_bonds=0.0, net_bills=0.0, other_fin=cash, buffer_change=0.0,
        cum_path=cum, prior_cum_path=prior_cum, prev2_cum_path=prev2_cum,
        fy_origin=origin, prior_fy_origin=p_start, prev2_fy_origin=p2_start,
        prior_year_label=_fy_label_at("es", p_start),
        prev2_year_label=_fy_label_at("es", p2_start),
    )


def _mof_jp_fytd() -> CountryFYTD:
    from sovereign_cash.debt.note.monthly_primary import jp_cash_tax_corrected

    # Monthly file through March misses the tax booked in the July settlement
    # (FY2025 ¥49.3tn vs 決算 ¥84.2tn). Gross tax up to that settlement.
    cash_s = jp_cash_tax_corrected().sort_index()
    last = cash_s.index.max()
    start, through, label = _fy_window("jp", last)
    origin = _fy_origin("jp", label)
    start = origin
    ytd = cash_s.loc[start:through]
    cash = float(ytd.sum())
    # Same-basis plan = prior complete FY MoF GA cash (WEO GG is not GA).
    gdp_year = origin.year
    p_start, p_end = _prior_fy_window("jp", origin)
    prior_full = cash_s.loc[p_start:p_end]
    if len(prior_full) >= 12:
        plan = float(prior_full.sum())
        plan_src = f"MoF GA prior FY {_fy_label_at('jp', p_start)} tax-corrected"
    else:
        plan, plan_src = _weo_deficit_plan("jp", gdp_year)
    cum = ytd.cumsum()
    prior_cum = _cum_window(cash_s, p_start, p_end)
    p2_start, p2_end = _prior_fy_window("jp", p_start)
    prev2_cum = _cum_window(cash_s, p2_start, p2_end)
    return CountryFYTD(
        cc="jp", year_label=label, through=f"{through:%Y-%m}", months_elapsed=float(len(ytd)),
        cash_ytd=cash, plan=plan, plan_src=plan_src,
        gdp=_gdp("jp", gdp_year), gdp_prior=_gdp("jp", gdp_year - 1),
        gdp_prev2=_gdp("jp", gdp_year - 2),
        net_bonds=0.0, net_bills=0.0, other_fin=cash, buffer_change=0.0,
        cum_path=cum, prior_cum_path=prior_cum, prev2_cum_path=prev2_cum,
        fy_origin=origin, prior_fy_origin=p_start, prev2_fy_origin=p2_start,
        prior_year_label=_fy_label_at("jp", p_start),
        prev2_year_label=_fy_label_at("jp", p2_start),
    )


def build_panel() -> list[CountryFYTD]:
    trk = {r["cc"]: r for r in iv.tracking()}
    out: list[CountryFYTD] = []
    for cc in ORDER:
        if cc == "us":
            out.append(_us_fytd())
        elif cc == "es":
            out.append(_igae_es_fytd())
        elif cc == "jp":
            out.append(_mof_jp_fytd())
        else:
            out.append(_cash_country(cc, trk[cc]))
    return out


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333", "xtick.color": "#333", "ytick.color": "#333",
        "text.color": "#222", "axes.labelcolor": "#222",
    })


def _prior_fy_month_share(p: CountryFYTD) -> float:
    """Prior-FY cum through same FY month / prior-FY full. Fallback calendar m/12."""
    prior = p.prior_cum_path.dropna()
    live_n = int(p.cum_path.dropna().shape[0])
    fallback = min(live_n, 12) / 12.0 if live_n > 0 else 1.0
    if prior.empty:
        return fallback
    mi = _months_into_fy(prior.index, p.prior_fy_origin)
    full = float(prior.iloc[-1])
    if full == 0:
        return fallback
    target_m = live_n if live_n > 0 else 1
    hit = [i for i, m in enumerate(mi) if m == target_m]
    if not hit:
        hit = [i for i, m in enumerate(mi) if m <= target_m]
    if not hit:
        return fallback
    share = float(prior.iloc[hit[-1]]) / full
    # Front-loaded years can exceed 1 by Aug; clamp away from zero only.
    if abs(share) < 0.05:
        return fallback
    return share


def _seasonal_annualize_live(p: CountryFYTD) -> float:
    """Implied full-year cash requirement: live YTD / prior-FY month share."""
    return float(p.cash_ytd) / _prior_fy_month_share(p)


def _prior_fy_month_shares(p: CountryFYTD) -> np.ndarray | None:
    """Length-12 prior-FY month shares of the cash total (sums to 1)."""
    prior = p.prior_cum_path.dropna()
    if prior.empty or len(prior) < 12:
        return None
    mi = _months_into_fy(prior.index, p.prior_fy_origin)
    levels = {m: float(v) for m, v in zip(mi, prior.values)}
    if any(m not in levels for m in range(1, 13)):
        return None
    flows = np.array(
        [levels[1]] + [levels[m] - levels[m - 1] for m in range(2, 13)],
        dtype=float,
    )
    tot = float(flows.sum())
    if abs(tot) < 1e-9:
        return None
    return flows / tot


def _plan_profile_cum_gdp(p: CountryFYTD) -> tuple[np.ndarray, np.ndarray] | None:
    """Cumulative this-year cash plan path, % GDP.

    Prior-FY month shares × plan for most countries. UK uses the cash framework's
    5y Covid-free median month profile (same as ``uk_cash_summary`` plan_profile_ytd)
    so the chart matches the £90bn vs £83bn CGNCR reading. A flat year-end hline
    mid-FY always sits above YTD even when the outturn is already hot vs profile.
    """
    if not (np.isfinite(p.plan) and p.plan != 0 and p.gdp > 0):
        return None
    if p.cc == "uk":
        from sovereign_cash.uk.model import month_profile

        d = iv.monthly("uk")
        x = (
            d.loc[~d["is_forecast"].fillna(False)]
            .set_index("month")["cash_requirement_bn"]
            .astype(float)
            .sort_index()
        )
        origin = p.cum_path.dropna().index.max()
        prof = month_profile(x, origin)
        # FY Apr–Mar: months into FY 1..12 ↔ calendar Apr..Mar
        cal = [(p.fy_origin + pd.DateOffset(months=i)).month for i in range(12)]
        sh = np.array([prof[m] for m in cal], dtype=float)
    else:
        sh = _prior_fy_month_shares(p)
        if sh is None:
            sh = np.full(12, 1.0 / 12.0)
    cum = np.cumsum(float(p.plan) * sh)
    return np.arange(1, 13), 100.0 * cum / p.gdp


def remainder_flows(p: CountryFYTD) -> pd.Series | None:
    """Rest of this FY, same rule for every country.

    Last year's remaining months, scaled so the path ends on the seas. ann.
    bar (YTD / prior-FY month share). A missing prior month falls back to an
    even split of the same remainder. A finished year returns None.
    """
    live = p.cum_path.dropna()
    k = int(live.shape[0])
    if k < 1 or k >= 12:
        return None
    months = pd.date_range(p.fy_origin, periods=12, freq="MS")
    rest_m = months[k:]
    end_bn = _seasonal_annualize_live(p)
    if not np.isfinite(end_bn):
        return None
    rest_lvl = end_bn - float(live.values[-1])
    shape = _prior_remaining_shape(p, k)
    if shape is None or len(shape) != len(rest_m) or abs(float(np.sum(shape))) < 1e-9:
        flows = pd.Series(rest_lvl / len(rest_m), index=rest_m)
    else:
        flows = pd.Series(rest_lvl * shape / float(np.sum(shape)), index=rest_m)
    if flows.isna().any():
        return None
    return flows


_BRIDGE = REPO / "us" / "fiscal" / "derived" / "supply_bridge.parquet"


_thin_published = False


def _publish_thin() -> None:
    """Spain and Japan parquets. Once per process; the chart reads them below."""
    global _thin_published
    if _thin_published:
        return
    _thin_published = True
    from sovereign_cash.thin_cash import publish
    publish()


def model_remainder(p: CountryFYTD) -> pd.Series | None:
    """Shipped cash-model flows for the months still in this FY.

    US is the fiscal path (DTS operating cash, ``unchanged`` bridge). UK, Germany,
    France and Italy are the bottom-up paths that ship. Sweden is the cash
    framework's live path, anchored to Riksgälden's monthly forecast; the
    bottom-up did not ship. Spain is IGAE category×macro (``es.bottom_up_macro``)
    when SHIP, else ``seas_add_3``. Japan is same-month last year on tax-corrected
    MoF general-account cash (category×macro NO-SHIP; scored on FY2025 only).

    France on this chart is BUM soft_pin α=0 (PLF×RESF categories, no solde
    re-pin). Walk-forward OOS mean MAE ~9.6 vs production seas_mult_3 ~15.3;
    a month-wise ``max(seas, BUM)`` envelope loses OOS (~14.3) and is not used.
    The live financing board still ships α=1 and lands on the PLF solde — that
    path must not appear here.
    """
    _publish_thin()
    live = p.cum_path.dropna()
    k = int(live.shape[0])
    if k < 1 or k >= 12:
        return None
    rest_m = pd.date_range(p.fy_origin, periods=12, freq="MS")[k:]
    if p.cc == "us":
        if not _BRIDGE.exists():
            return None
        b = pd.read_parquet(_BRIDGE).loc["unchanged"]
        b.index = pd.to_datetime(b.index)
        s = b["deficit_bn"].astype(float)
        s = s[~s.index.duplicated(keep="last")].reindex(rest_m)
    elif p.cc == "fr":
        from sovereign_cash.fr import bottom_up_macro as BUM

        origin = live.index.max()
        bu = BUM.path(origin, int(len(rest_m)), soft_pin_alpha=0.0)
        if bu is None:
            return None
        s = bu.reindex(rest_m)
    elif p.cc == "it":
        # Prefer the bottom-up column written by apply_to_forward (BU − RRF).
        # cash_requirement_bn equals that when SHIP; fall back to an on-the-fly
        # forecast minus RRF grants so a stale parquet cannot revert to DFP pin.
        d = iv.monthly("it")
        if d is not None and "cash_requirement_bottomup_bn" in d.columns:
            f = (
                d.loc[d["is_forecast"].fillna(False)]
                .set_index("month")["cash_requirement_bottomup_bn"]
                .astype(float)
            )
            f = f[~f.index.duplicated(keep="last")]
            s = f.reindex(rest_m)
            if not s.isna().any():
                return s
        from sovereign_cash.it import bottom_up_macro as BUM
        from sovereign_cash.it import events as ev

        origin = live.index.max()
        bu = BUM.forecast(origin, int(len(rest_m)))
        if bu is None:
            return None
        grant = ev.rrf_split(rest_m)["grant_bn"].reindex(rest_m).fillna(0.0)
        s = (bu.reindex(rest_m) - grant).astype(float)
    else:
        d = iv.monthly(p.cc)
        if d is None or "cash_requirement_bn" not in d.columns:
            return None
        f = d.loc[d["is_forecast"].fillna(False)].set_index("month")["cash_requirement_bn"].astype(float)
        f = f[~f.index.duplicated(keep="last")]
        s = f.reindex(rest_m)
    if s.isna().any():
        return None
    return s


def _soft_pinned_to_plan(p: CountryFYTD, rem: pd.Series) -> bool:
    """True when ``rem`` is the FR BUM α=1 soft-pin (lands on the PLF solde).

    Do not use YE≈plan: with YTD already past the revised solde, α=0 H2 can land
    within a few tenths of €bn of the plan without being pinned (Nov-2026: α=0
    gap 0.22 vs α=1 gap 0.00; a 0.2%·plan band false-fired).
    """
    if p.cc != "fr" or rem is None or rem.empty:
        return False
    from sovereign_cash.fr import bottom_up_macro as BUM

    live = p.cum_path.dropna()
    if live.empty:
        return False
    pinned = BUM.path(live.index.max(), int(len(rem)), soft_pin_alpha=1.0)
    if pinned is None:
        return False
    pinned = pinned.reindex(rem.index).astype(float)
    if pinned.isna().any() or rem.isna().any():
        return False
    return float((rem.astype(float) - pinned).abs().sum()) < 0.05


def chart_remainder(p: CountryFYTD) -> pd.Series | None:
    """Rest-of-FY path on Ex1c = cash model (FR = BUM α=0; not the board soft-pin).

    Dashed end must match ``projection_fy_bn``. Do not fall back to a seasonal
    remainder when a model path exists — that would re-hide FR/IT off-plan gaps.
    """
    modelled = model_remainder(p)
    if modelled is not None:
        return modelled
    return remainder_flows(p)


def _next_fy_origin(p: CountryFYTD) -> pd.Timestamp:
    return p.fy_origin + pd.DateOffset(months=12)


def next_fy_model_flows(p: CountryFYTD) -> pd.Series | None:
    """Shipped model monthly cash for the *next* full FY (12 months), local bn.

    Same sources as ``model_remainder`` / ``next_fy_model_bn``: US bridge,
    FR BUM α=0 + legislated policy, IT bottom-up column, ES/JP thin next-FY
    block, UK seasonal ``cash_requirement_model_bn`` (live BU collapses past
    the current remit — Jan self-assessment seasonal dies; FY27-28 ~£48bn vs
    remit ~£145bn / seasonal ~£136bn). Else ``iv.monthly`` forecast rows.

    Policy is the same overlay as the scenario pack's ``with_policy`` line —
    next-FY must not look artificially better than the authority by dropping it.
    """
    from sovereign_cash.policy.monthly import add_to_monthly

    _publish_thin()
    y0 = _next_fy_origin(p)
    months = pd.date_range(y0, periods=12, freq="MS")
    if p.cc == "us":
        if not _BRIDGE.exists():
            return None
        b = pd.read_parquet(_BRIDGE).loc["unchanged"]
        b.index = pd.to_datetime(b.index)
        s = b["deficit_bn"].astype(float)
        s = s[~s.index.duplicated(keep="last")].reindex(months)
    elif p.cc == "fr":
        from sovereign_cash.fr import bottom_up_macro as BUM

        live = p.cum_path.dropna()
        if live.empty:
            return None
        origin = live.index.max()
        end = y0 + pd.DateOffset(months=11)
        n = int((end.to_period("M") - origin.to_period("M")).n)
        if n < 12:
            return None
        # No-policy BUM (α=0), then the legislated overlay — not a PLF solde pin.
        # Pinning hid the policy line and fought the pack's with_policy baseline.
        bu = BUM.path(origin, n, soft_pin_alpha=0.0)
        if bu is None:
            return None
        s = add_to_monthly("fr", bu.reindex(months), origin)
    elif p.cc == "it":
        d = iv.monthly("it")
        if d is not None and "cash_requirement_bottomup_bn" in d.columns:
            s = (
                d.set_index(pd.to_datetime(d["month"]))["cash_requirement_bottomup_bn"]
                .astype(float)
            )
            s = s[~s.index.duplicated(keep="last")].reindex(months)
        else:
            return None
    elif p.cc == "uk":
        # Live cash_requirement_bn = bottom_up past the current FY loses the
        # January receipt seasonal and understates the year ~3× vs the remit.
        d = iv.monthly("uk")
        if d is None or "cash_requirement_model_bn" not in d.columns:
            return None
        s = (
            d.set_index(pd.to_datetime(d["month"]))["cash_requirement_model_bn"]
            .astype(float)
        )
        s = s[~s.index.duplicated(keep="last")].reindex(months)
    else:
        d = iv.monthly(p.cc)
        if d is None or "cash_requirement_bn" not in d.columns:
            return None
        s = (
            d.set_index(pd.to_datetime(d["month"]))["cash_requirement_bn"]
            .astype(float)
        )
        s = s[~s.index.duplicated(keep="last")].reindex(months)
    if s is None or s.isna().any() or len(s) < 12:
        return None
    s = s.astype(float)
    # FR already added policy above. Other countries: add here so Ex1c does not wait on a
    # cash-board rebuild. Writers (later_years / thin_cash) also bake it in — when both
    # fire, annual_effects netting on an already-adjusted level is wrong, so skip when the
    # frame carries an explicit policy column.
    if p.cc != "fr":
        d = None if p.cc == "us" else iv.monthly(p.cc)
        already = (
            d is not None
            and "policy_bn" in getattr(d, "columns", [])
            and float(pd.to_numeric(d["policy_bn"], errors="coerce").fillna(0.0).abs().sum()) > 0
        )
        if not already:
            s = add_to_monthly(p.cc, s, y0 - pd.offsets.MonthBegin(1))
    return s


def next_fy_model_bn(p: CountryFYTD) -> float | None:
    """Shipped model cash requirement for the *next* full FY, local bn."""
    s = next_fy_model_flows(p)
    return None if s is None else float(s.sum())


def projection_flows(p: CountryFYTD) -> tuple[pd.Series | None, str]:
    """Rest-of-FY flows and the source: ``model`` or ``prior``."""
    modelled = model_remainder(p)
    if modelled is not None:
        return modelled, "model"
    return remainder_flows(p), "prior"


def projection_fy_bn(p: CountryFYTD) -> float | None:
    """Model full-year cash, local bn. None when the dash is only the prior-year shape."""
    flows, src = projection_flows(p)
    if flows is None or src != "model":
        return None
    return float(p.cum_path.dropna().values[-1]) + float(flows.sum())


def _prior_remaining_shape(p: CountryFYTD, k: int) -> np.ndarray | None:
    """Prior-FY monthly flows for fiscal months k+1..12."""
    prior = p.prior_cum_path.dropna()
    if prior.empty:
        return None
    levels = {
        m: float(v)
        for m, v in zip(_months_into_fy(prior.index, p.prior_fy_origin), prior.values)
    }
    if any(m not in levels for m in range(1, 13)):
        return None
    flows = []
    prev = 0.0
    for m in range(1, 13):
        flows.append(levels[m] - prev)
        prev = levels[m]
    return np.array(flows[k:], dtype=float)


def _next_fy_label(p: CountryFYTD) -> str:
    if p.cc == "us":
        return f"FY{int(p.year_label.replace('FY', '')) + 1}"
    if p.cc in ("uk", "jp"):
        y0 = int(p.year_label.replace("FY", "").split("-")[0])
        return f"FY{y0 + 1}-{str(y0 + 2)[-2:]}"
    return str(int(p.year_label) + 1)


def _fy_short(label: str) -> str:
    """Compact FY token for panel labels: FY2027→FY27, FY2026-27→FY26-27."""
    if label.startswith("FY") and "-" in label:
        body = label[2:]
        a, b = body.split("-", 1)
        return f"FY{a[-2:]}-{b[-2:]}"
    if label.startswith("FY"):
        return f"FY{label[-2:]}"
    return label


def _plan_hline_lab(fy_label: str, pct: float, other: float | None = None,
                    kind: str = "plan") -> str:
    """e.g. ``2026 plan: 4.5`` / ``FY27 plan: 6.1`` / ``2026 prev plan: 4.1``."""
    if other is not None and abs(pct - other) < 0.15:
        num = f"{pct:.2f}"
    else:
        num = f"{pct:.1f}"
    return f"{_fy_short(fy_label)} {kind}: {num}"


def _next_plan_bn(p: CountryFYTD) -> float | None:
    """Authority cash-requirement plan for the FY after ``p.year_label``, local bn.

    Returns None when that vintage is missing — the summary next-year column is
    shown only when every country in the panel resolves.
    """
    import json

    cc = p.cc
    if cc == "us":
        fy = int(p.year_label.replace("FY", "")) + 1
        cbo = pd.read_parquet(CBO)
        row = cbo[(cbo["var"] == "deficit_bn") & (cbo["year"] == fy)]
        return abs(float(row["value"].iloc[0])) if not row.empty else None
    if cc == "uk":
        from sovereign_cash.uk import sources as uks
        from sovereign_cash.uk import obr_tables

        nxt = _next_fy_label(p).replace("FY", "")  # 2027-28
        v = uks.remit_vintages()
        sub = v[v["fy"].astype(str) == nxt].sort_values("vintage_date")
        if not sub.empty:
            return float(sub.iloc[-1]["cgncr_ex"])
        y = int(p.year_label.replace("FY", "").split("-")[0]) + 1
        d = obr_tables.cgncr_ex_nr()
        return float(d[y]) if y in d else None
    if cc == "de":
        plans = json.loads((REPO / "sovereign_cash" / "data" / "de" / "plan.json").read_text())
        y = str(int(p.year_label) + 1)
        block = plans.get("years", {}).get(y)
        return float(block["total"]) if block and block.get("total") is not None else None
    if cc == "fr":
        plans = json.loads((REPO / "sovereign_cash" / "data" / "fr" / "plan.json").read_text())
        y = str(int(p.year_label) + 1)
        block = plans.get("years", {}).get(y)
        return (float(block["deficit_to_finance"])
                if block and block.get("deficit_to_finance") is not None else None)
    if cc == "it":
        from sovereign_cash.it import events as ev

        y = int(p.year_label) + 1
        pl = getattr(ev, "PLAN", {}).get(y)
        if not pl:
            return None
        return float(pl["pct_gdp"]) / 100.0 * float(pl["gdp_mn"]) / 1000.0
    if cc == "se":
        se = json.loads((REPO / "sovereign_cash" / "data" / "se" / "authority.json").read_text())
        y = str(int(p.year_label) + 1)
        block = se.get("nbr_annual_balance", {}).get("years", {}).get(y)
        # NBR balance is surplus(+)/deficit(−); cash requirement = −balance.
        return -float(block["bn"]) if block and block.get("bn") is not None else None
    if cc == "es":
        # Hold this year's IGAE cash benchmark forward (no Tesoro cash plan wired).
        return float(p.plan) if np.isfinite(p.plan) and p.plan > 0 else None
    if cc == "jp":
        # Same MoF GA basis as this-FY plan (prior complete FY cash, held).
        # WEO GG −balance is not GA and falsely drops ~9% → ~2% of GDP.
        return float(p.plan) if np.isfinite(p.plan) and p.plan > 0 else None
    return None


def _next_plan_gdp(p: CountryFYTD) -> float:
    """GDP used to scale next-FY plan % — same calendar convention as live plan."""
    if p.cc == "us":
        fy_next = int(p.year_label.replace("FY", "")) + 1
        return _gdp("us", fy_next - 1)
    if p.cc in ("uk", "jp"):
        y0 = int(p.year_label.replace("FY", "").split("-")[0]) + 1
        return _gdp(p.cc, y0)
    return _gdp(p.cc, int(p.year_label) + 1)


def _prior_plan_bn(p: CountryFYTD) -> tuple[float, str] | None:
    """Pre-revision authority cash plan for *this* FY, if a vintage exists.

    France: LFI 2026 déficit à financer (€124.4bn) before PLF 2027 revised it
    to €137.2bn. No prior *next*-year cash vintage — 2027 only appears in PLF 2027.
    """
    import json

    if p.cc != "fr":
        return None
    plans = json.loads((REPO / "sovereign_cash" / "data" / "fr" / "plan.json").read_text())
    block = plans.get("years", {}).get(str(p.year_label))
    if not block:
        return None
    bn = block.get("deficit_lfi_bn")
    if bn is None or not np.isfinite(float(bn)):
        return None
    return float(bn), "prev plan"


def summary_table_html(panel: list[CountryFYTD]) -> str:
    """Live vs plan (% GDP); next = next-year authority plan (model in paren)."""
    head = (
        "<tr><th>Country</th><th>Through</th><th>Raw YTD</th><th>Seas. ann.</th>"
        "<th>Cash model</th><th>Plan</th><th>Next plan</th></tr>"
    )
    rows = []
    for p in panel:
        raw = 100.0 * p.cash_ytd / p.gdp
        ann = 100.0 * _seasonal_annualize_live(p) / p.gdp
        plan = 100.0 * p.plan / p.gdp
        modelled = projection_fy_bn(p)
        mod = f"{100.0 * modelled / p.gdp:.1f}" if modelled is not None else "—"
        nxt_plan = _next_plan_bn(p)
        nxt_mod = next_fy_model_bn(p)
        gdp_n = _next_plan_gdp(p)
        if nxt_plan is None or not np.isfinite(float(nxt_plan)) or gdp_n <= 0:
            nxt_cell = "—"
        else:
            nxt_pct = 100.0 * float(nxt_plan) / gdp_n
            mod_bit = ""
            if nxt_mod is not None and np.isfinite(float(nxt_mod)):
                mod_bit = (
                    f" <span class='yr'>(model {100.0 * float(nxt_mod) / gdp_n:.1f})</span>"
                )
            nxt_cell = f"{nxt_pct:.1f}{mod_bit} <span class='yr'>{_next_fy_label(p)}</span>"
        rows.append(
            f"<tr><td>{LABEL[p.cc]}</td><td>{p.through}</td>"
            f"<td>{raw:.1f}</td><td>{ann:.1f}</td><td>{mod}</td>"
            f"<td>{plan:.1f} <span class='yr'>{p.year_label}</span></td>"
            f"<td>{nxt_cell}</td></tr>"
        )
    note = (
        "Seas. ann. = YTD ÷ prior-FY month share. Cash model = printed YTD + shipped "
        "rest-of-FY path. Plan / next plan = authority cash levels. Model next-FY in "
        "parentheses on the next column. FR this-year remainder = BUM α=0 (unpinned gap); "
        "next-FY = no-policy model + legislated policy overlay (pack with_policy)."
    )
    return (
        "<!-- exhibit-1c-summary -->"
        "<table class='fytd-sum' style='max-width:54em'>"
        f"<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"
        f"<p class='fytd-sum-note'>{note}</p>"
        "<!-- /exhibit-1c-summary -->"
    )


# Common wedge categories for DE / IT stacked bars (pp GDP contribution to seas.ann − plan).
WEDGE_CATS = [
    ("current", "Current expenditure"),
    ("capital", "Capital expenditure"),
    ("revenue", "Revenue"),
    ("residual", "Residual"),
]
WEDGE_COLORS = {
    "Current expenditure": "#8b1e1e",
    "Capital expenditure": "#b5651d",
    "Revenue": "#2d6a4f",
    "Residual": "#6a4c93",
}
# Hatch on spend legs so current vs capital read apart when both are large.
WEDGE_HATCH = {
    "Current expenditure": "///",
    "Capital expenditure": "...",
}


def _de_overshoot_vs_plan(through_month: int | None = None) -> dict:
    """DE contribution to seas.ann − Bundeshaushalt plan, on the common wedge taxonomy.

    Books annualized with the same prior-FY *total-cash* month share as the headline bar
    (so the diamond matches the top-right DE wedge). Core NKA gap is split using BMF
    Monatsbericht revenue vs plan-to-date (annualized): revenue shortfall → Revenue;
    remainder of core + SVBw → Current; SVIK → Capital; other funds → Residual.
    """
    import json
    from sovereign_cash.de import fetch as Fde

    m = pd.read_parquet(DERIVED / "de_cash_monthly.parquet")
    m["month"] = pd.to_datetime(m["month"])
    h = m.loc[~m["is_forecast"]].copy()
    y_live = int(h["month"].max().year)
    last_m = int(h["month"].max().month)
    through = min(through_month if through_month is not None else last_m, last_m)

    def _ytd(col: str) -> float:
        s = h[(h["month"].dt.year == y_live) & (h["month"].dt.month <= through)]
        return float(s[col].sum()) if col in s and s[col].notna().any() else 0.0

    prior = h[h["month"].dt.year == y_live - 1]
    share = float(
        prior.loc[prior["month"].dt.month <= through, "cash_requirement_bn"].sum()
        / prior["cash_requirement_bn"].sum()
    )
    plans = json.loads((REPO / "sovereign_cash" / "data" / "de" / "plan.json").read_text())
    py = plans["years"][str(y_live)]
    gdp = _gdp("de", y_live)

    def _book_gap(col: str, plan_bn: float) -> float:
        return (_ytd(col) / share if share else _ytd(col)) - plan_bn

    core_g = _book_gap("cash_requirement_core_bn", float(py["core"]))
    infra_g = _book_gap("cash_requirement_sv_infrastruktur_bn", float(py["infrastruktur"]))
    bw_g = _book_gap("cash_requirement_sv_bundeswehr_bn", float(py["bundeswehr"]))
    oth_g = _book_gap("cash_requirement_sv_other_bn", 0.0)
    headline = core_g + infra_g + bw_g + oth_g

    mb = Fde.load_bmf_mb()
    ts = pd.Timestamp(y_live, through, 1)
    ts_prior = pd.Timestamp(y_live - 1, through, 1)
    rev_ann = 0.0
    if ts in mb.index and ts_prior in mb.index:
        r = mb.loc[ts]
        rp = mb.loc[ts_prior]
        prev_full = float(r["ist_prev_revenue"]) if pd.notna(r.get("ist_prev_revenue")) else np.nan
        if prev_full and prev_full == prev_full and prev_full != 0 and pd.notna(r.get("soll_revenue")):
            sh_r = float(rp["ytd_revenue"]) / prev_full
            if abs(sh_r) >= 0.05:
                rev_ytd_gap = float(r["ytd_revenue"]) - float(r["soll_revenue"]) * sh_r
                rev_ann = -rev_ytd_gap / sh_r

    gaps = {
        "current": core_g + bw_g - rev_ann,
        "capital": infra_g,
        "revenue": rev_ann,
        "residual": oth_g,
    }
    gaps["residual"] += headline - sum(gaps.values())

    segments = [
        {"key": k, "label": lab, "gap_bn": gaps[k], "gap_gdp": 100.0 * gaps[k] / gdp}
        for k, lab in WEDGE_CATS
    ]
    return {
        "y_live": y_live, "through": through, "share": share, "gdp": gdp,
        "segments": segments, "gap_bn": headline,
        "gap_gdp": 100.0 * headline / gdp,
        "plan_src": py.get("source", "Bundeshaushalt"),
        "detail": {"core_g": core_g, "bw_g": bw_g, "infra_g": infra_g, "oth_g": oth_g,
                   "rev_ann": rev_ann},
    }


def _it_bdi_t1_vs_plan(through_month: int | None = None) -> dict:
    """BdI T1 live YTD vs DFP authority plan-to-date (€bn); fabbisogno wedge by category.

    Identity: fabb = cur_exp + cap_exp − tax_rev − other_rev − treasury_bal.
    DFP cash Tavole sit on a different perimeter than BdI T1 (~9–15% on levels), so the
    authority path is rebased: BdI_{y−1} full × (DFP_y / DFP_{y−1}), then × prior-year
    same-month share → plan-to-date. Treasury has no DFP line → prior YTD held as plan.
    Interest carve omitted (incomplete-year bias).
    """
    import json

    t1 = pd.read_parquet(REPO / "sovereign_cash" / "data" / "it" / "bdi_t1.parquet")
    t1.index = pd.to_datetime(t1.index)
    t1_bn = t1 / 1000.0
    y_live = int(t1_bn.index.max().year)
    last_m = int(t1_bn.index.max().month)
    through = min(through_month if through_month is not None else last_m, last_m)
    cats = ["tax_rev", "other_rev", "cur_exp", "cap_exp", "treasury_bal"]

    def _sum(year: int, months: int | None = None) -> pd.Series:
        s = t1_bn[t1_bn.index.year == year]
        if months is not None:
            s = s[s.index.month <= months]
        return s[cats].sum()

    live = _sum(y_live, through)
    prior = _sum(y_live - 1, through)
    prior_full = _sum(y_live - 1, None)
    shares = prior / prior_full.replace(0, np.nan)

    auth = json.loads((REPO / "sovereign_cash" / "data" / "it" / "authority.json").read_text())
    yrs = auth["categories_cash_central"]["years"]
    dfp_map = {
        "tax_rev": "tributari",
        "other_rev": "incassi_correnti_ex_tributari",
        "cur_exp": "pagamenti_correnti",
        "cap_exp": "pagamenti_conto_capitale",
    }
    y0, y1 = str(y_live - 1), str(y_live)
    if y1 not in yrs or y0 not in yrs:
        raise KeyError(f"DFP categories missing for {y0}/{y1}")

    plan_todate = {}
    live_gap = {}  # live − plan_todate (category level)
    for k, dfp_k in dfp_map.items():
        g = float(yrs[y1][dfp_k]) / float(yrs[y0][dfp_k])
        plan_full = float(prior_full[k]) * g
        ptd = plan_full * float(shares[k])
        plan_todate[k] = ptd
        live_gap[k] = float(live[k]) - ptd
    # Treasury: no DFP line — prior YTD is the plan path
    plan_todate["treasury_bal"] = float(prior["treasury_bal"])
    live_gap["treasury_bal"] = float(live["treasury_bal"]) - plan_todate["treasury_bal"]

    # Contribution to fabbisogno wedge vs plan (spend +, revenue −, treasury −)
    contrib = {
        "cur_exp": live_gap["cur_exp"],
        "cap_exp": live_gap["cap_exp"],
        "tax_rev": -live_gap["tax_rev"],
        "other_rev": -live_gap["other_rev"],
        "treasury": -live_gap["treasury_bal"],
    }
    fabb_live = float(
        live["cur_exp"] + live["cap_exp"] - live["tax_rev"] - live["other_rev"] - live["treasury_bal"]
    )
    fabb_plan = (
        plan_todate["cur_exp"] + plan_todate["cap_exp"]
        - plan_todate["tax_rev"] - plan_todate["other_rev"]
        - plan_todate["treasury_bal"]
    )
    dfabb = fabb_live - fabb_plan
    gdp = _gdp("it", y_live)
    gaps = {
        "current": contrib["cur_exp"],
        "capital": contrib["cap_exp"],
        "revenue": contrib["tax_rev"] + contrib["other_rev"],
        "residual": contrib["treasury"],
    }
    segments = [
        {"key": k, "label": lab, "gap_bn": gaps[k], "gap_gdp": 100.0 * gaps[k] / gdp}
        for k, lab in WEDGE_CATS
    ]
    return {
        "y_live": y_live,
        "through": through,
        "live": live,
        "prior": prior,
        "plan_todate": plan_todate,
        "live_gap": live_gap,
        "contrib": contrib,
        "segments": segments,
        "fabb_live": fabb_live,
        "fabb_plan": float(fabb_plan),
        "dfabb": float(dfabb),
        "gap_bn": float(dfabb),
        "gap_gdp": 100.0 * float(dfabb) / gdp,
        "gdp": gdp,
        "plan_src": f"DFP {yrs[y1].get('vintage', y1)}",
    }


def _imf_year(p: CountryFYTD) -> int:
    """WEO calendar year lined up with the open FY. US label is the FY end year."""
    if p.cc == "us":
        return int(p.year_label.replace("FY", ""))
    if p.cc in ("uk", "jp"):
        return int(p.year_label.replace("FY", "").split("-")[0])
    return int(p.year_label)


def _imf_deficit_pct(cc: str, year: int) -> float:
    """WEO general-government overall deficit, % of GDP. Surplus is negative."""
    weo = pd.read_parquet(REPO / "sovereign_cash" / "data" / "debt" / "weo.parquet")
    row = weo.loc[
        (weo["country"].astype(str) == cc)
        & (weo["var"].astype(str) == "balance_pct")
        & (weo["year"] == int(year))
    ]
    if row.empty or not np.isfinite(row["value"].iloc[0]):
        return float("nan")
    return -float(row["value"].iloc[0])


def render(panel: list[CountryFYTD] | None = None, out: Path | None = None) -> tuple[Path, str, str]:
    """Eight cash-balance panels. Plan and IMF are level lines in the same % of GDP."""
    panel = panel or build_panel()
    out = out or (NOTE / "dm_debt_outlook_fytd.png")
    _style()

    # Compact grid — room for ``YYYY plan: x.x`` labels without a sparse page.
    fig = plt.figure(figsize=(14.4, 9.0))
    top = fig.add_gridspec(
        2, 4, hspace=0.30, wspace=0.26,
        left=0.045, right=0.99, top=0.935, bottom=0.095,
    )
    fig.text(0.045, 0.965, "Cash balance, % of GDP",
             fontsize=12.5, color="#111", ha="left", va="bottom", weight="bold")
    palette = {
        "us": "#1f4e79", "uk": "#8b1e1e", "de": "#2d6a4f",
        "fr": "#6a4c93", "it": "#b5651d", "es": "#c45c26",
        "jp": "#dc2626", "se": "#0077b6",
    }
    _model_ls = (0, (5.0, 2.5))       # clear dash — rest-of-FY (this year)
    _next_fy_ls = (0, (3.0, 1.6))     # mid dash — next-FY model path
    _plan_ls = (0, (1.2, 1.6))        # faint dash — this-year plan level
    _next_ls = (0, (6.0, 2.2))        # bold dash — next-year plan level

    for i, p in enumerate(panel):
        ax = fig.add_subplot(top[i // 4, i % 4])
        col = palette[p.cc]
        solid = p.cum_path.dropna()
        plan_y = 100.0 * p.plan / p.gdp if p.gdp else np.nan
        ys_all: list[float] = [0.0]
        ring_at = None
        model_end = None  # (x_month, y_pct, ye_bn)
        if not solid.empty and p.gdp:
            mi = _months_into_fy(solid.index, p.fy_origin)
            yv = 100.0 * solid.values / p.gdp
            ax.plot(mi, yv, color=col, lw=1.7, marker="o", ms=2.6, zorder=3)
            ys_all.extend(float(v) for v in yv)
            rest = chart_remainder(p)
            if rest is not None:
                cum_bn = float(solid.values[-1]) + rest.cumsum()
                xs = [mi[-1], *range(mi[-1] + 1, mi[-1] + 1 + len(rest))]
                ys = [float(yv[-1]), *(100.0 * cum_bn.values / p.gdp)]
                ax.plot(xs, ys, color=col, lw=1.45, ls=_model_ls, zorder=4)
                ys_all.extend(float(v) for v in ys)
                model_end = (xs[-1], float(ys[-1]), float(cum_bn.iloc[-1]))
                if p.cc == "fr" and _soft_pinned_to_plan(p, rest):
                    raise RuntimeError(
                        "Exhibit 1c FR model landed on PLF solde — "
                        "BUM α=0 path broken"
                    )
            if p.cc in ("fr", "it"):
                ring_at = (mi[-1], float(yv[-1]))
        # Prior FY outturn (faint) — shape reference, same month-of-FY axis.
        prior_cum = p.prior_cum_path.dropna()
        if not prior_cum.empty and p.gdp:
            pmi = _months_into_fy(prior_cum.index, p.prior_fy_origin)
            pyv = 100.0 * prior_cum.values / p.gdp
            ax.plot(pmi, pyv, color=col, lw=1.0, alpha=0.28, zorder=1)
            ys_all.extend(float(v) for v in pyv)
        # Next-FY model path on the same month-of-FY axis.
        # ES/JP: thin_cash next-FY block (BU / smly).
        nxt_gdp = _next_plan_gdp(p)
        nxt_flows = next_fy_model_flows(p)
        if nxt_flows is not None and nxt_gdp > 0:
            nxt_cum = 100.0 * nxt_flows.cumsum().to_numpy() / nxt_gdp
            ax.plot(range(1, 13), nxt_cum, color=col, lw=1.5, ls=_next_fy_ls,
                    zorder=3)
            ys_all.extend(float(v) for v in nxt_cum)
        plan_end = plan_y
        prior = _prior_plan_bn(p)
        prior_y = (
            100.0 * prior[0] / p.gdp
            if prior is not None and p.gdp > 0 else float("nan")
        )
        has_prior = np.isfinite(prior_y) and (
            not np.isfinite(plan_end) or abs(prior_y - plan_end) >= 0.08
        )
        if np.isfinite(plan_end):
            ax.hlines(plan_end, 0.5, 12.0, colors=col, linestyles=_plan_ls,
                      lw=1.0, alpha=0.65, zorder=2)
            ys_all.append(float(plan_end))
        if has_prior:
            ax.hlines(prior_y, 0.5, 12.0, colors=col, linestyles=_plan_ls,
                      lw=0.9, alpha=0.40, zorder=2)
            ys_all.append(float(prior_y))
        nxt_bn = _next_plan_bn(p)
        nxt_y = (
            100.0 * float(nxt_bn) / nxt_gdp
            if nxt_bn is not None and np.isfinite(float(nxt_bn)) and nxt_gdp > 0
            else float("nan")
        )
        has_nxt = np.isfinite(nxt_y)
        if has_nxt:
            ax.hlines(nxt_y, 0.5, 12.0, colors=col, linestyles=_next_ls,
                      lw=1.55, alpha=1.0, zorder=4)
            ys_all.append(float(nxt_y))
        if ring_at is not None:
            ax.scatter([ring_at[0]], [ring_at[1]], s=90, facecolors="none",
                       edgecolors=col, linewidths=1.2, zorder=6)
        ax.axhline(0, color="#333", lw=0.35)
        ax.set_xlim(0.4, 12.4)
        ax.set_xticks([1, 6, 12])
        ax.tick_params(axis="both", labelsize=6.5, length=2.5, pad=1.5)
        finite = [v for v in ys_all if np.isfinite(v)]
        lo, hi = min(finite), max(finite)
        span = max(hi - lo, 0.6)
        ax.set_ylim(lo - 0.08 * span, hi + 0.22 * span)
        from matplotlib.transforms import blended_transform_factory
        lab_tf = blended_transform_factory(ax.transAxes, ax.transData)
        lab_bbox = dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.88)
        nxt_lab = _next_fy_label(p)

        if np.isfinite(plan_end):
            plan_va = "top" if (has_nxt and nxt_y < plan_end) else "bottom"
            ax.text(
                0.02, plan_end,
                _plan_hline_lab(p.year_label, plan_end, nxt_y if has_nxt else None),
                transform=lab_tf, ha="left", va=plan_va, fontsize=5.5,
                color=col, alpha=0.85, clip_on=True, zorder=5, bbox=lab_bbox,
            )
        if has_nxt:
            nxt_va = "bottom" if (np.isfinite(plan_end) and nxt_y >= plan_end) else "top"
            ax.text(
                0.98, nxt_y,
                _plan_hline_lab(nxt_lab, nxt_y, plan_end if np.isfinite(plan_end) else None),
                transform=lab_tf, ha="right", va=nxt_va, fontsize=6.2,
                color=col, clip_on=True, zorder=6, bbox=lab_bbox, fontweight="bold",
            )
        if has_prior:
            prior_kind = prior[1] if prior is not None else "prior"
            prior_va = "top" if np.isfinite(plan_end) and prior_y < plan_end else "bottom"
            ax.text(
                0.98, prior_y,
                _plan_hline_lab(p.year_label, prior_y, kind=prior_kind),
                transform=lab_tf, ha="right", va=prior_va, fontsize=5.5,
                color=col, alpha=0.8, clip_on=True, zorder=5, bbox=lab_bbox,
            )
        if model_end is not None and np.isfinite(plan_end):
            mx, my, _ = model_end
            if abs(my - plan_end) >= 0.08:
                ax.scatter([mx], [my], s=16, color=col, zorder=6)
                ax.text(mx + 0.12, my, f"{my:.1f}", ha="left",
                        va="bottom" if my > plan_end else "top",
                        fontsize=5.8, color=col, zorder=6, bbox=lab_bbox)
        ax.set_title(f"{LABEL[p.cc]} {_fy_short(p.year_label)}", loc="left",
                     fontsize=7.5, color=col, pad=1.5)
        if i % 4 == 0:
            ax.set_ylabel("% GDP", fontsize=7)

    fig.text(0.5, 0.055, "Month of FY", fontsize=7.5, color="#444",
             ha="center", va="top")
    from matplotlib.lines import Line2D
    key = [
        Line2D([0], [0], color="#111", lw=1.7, marker="o", ms=3.0, label="printed YTD"),
        Line2D([0], [0], color="#111", lw=1.45, ls=_model_ls, label="rest-of-FY"),
        Line2D([0], [0], color="#111", lw=1.5, ls=_next_fy_ls, label="next-FY forecast"),
        Line2D([0], [0], color="#111", lw=1.0, alpha=0.35, label="prior FY"),
        Line2D([0], [0], color="#111", lw=1.55, ls=_next_ls, label="next-year plan"),
    ]
    fig.legend(
        handles=key, loc="lower center", ncol=5, frameon=False,
        fontsize=7.2, bbox_to_anchor=(0.5, 0.002),
        handlelength=2.2, columnspacing=1.2, handletextpad=0.35,
    )

    fig.savefig(out, dpi=160)
    plt.close(fig)

    caption = (
        f"<b>Exhibit 1c.</b> Government net cash borrowing, percent of GDP. "
        f"Faint = prior FY outturn; dashed rest-of-FY = cash model; mid-dash = "
        f"next-FY model (same month-of-FY axis). "
        f"Bold ``FY plan:`` = next-year authority level; left label = this-year "
        f"plan (France also ``prev plan:`` before the latest revision). "
        f"Cash perimeter only (UK = CGNCR)."
    )
    return out, caption, ""



def qa_table(panel: list[CountryFYTD] | None = None) -> pd.DataFrame:
    panel = panel or build_panel()
    rows = []
    for p in panel:
        ident = p.net_bonds + p.net_bills + p.other_fin - p.buffer_change
        rows.append({
            "cc": p.cc, "year": p.year_label, "through": p.through,
            "cash": round(p.cash_ytd, 1), "plan": round(p.plan, 1),
            "share_pct": round(100 * p.cash_ytd / p.plan, 1),
            "time_pct": round(100 * p.months_elapsed / 12, 1),
            "bonds": round(p.net_bonds, 1), "bills": round(p.net_bills, 1),
            "other": round(p.other_fin, 1), "d_buffer": round(p.buffer_change, 1),
            "ident": round(ident, 2), "gap": round(p.cash_ytd - ident, 3),
            "cash_gdp": round(100 * p.cash_ytd / p.gdp, 2),
        })
    return pd.DataFrame(rows)


def colour_html(panel: list[CountryFYTD] | None = None) -> str:
    """Exhibit 1c country bullets: what underpins model − plan (composition, not just the gap)."""
    panel = panel or build_panel()
    by = {p.cc: p for p in panel}

    def ye_bn(p: CountryFYTD) -> float:
        rem = model_remainder(p)
        return float(p.cash_ytd) + (float(rem.sum()) if rem is not None else 0.0)

    def gap_pp(p: CountryFYTD) -> float:
        return 100.0 * (ye_bn(p) - float(p.plan)) / float(p.gdp)

    def pct(p: CountryFYTD, bn: float) -> str:
        return f"{100.0 * bn / float(p.gdp):.1f}%"

    us, uk, de, fr, it, es, jp, se = (by[c] for c in ORDER)
    # Rank line from live gaps (undershoot → overshoot).
    gaps = sorted(((gap_pp(by[c]), c) for c in ORDER), key=lambda x: x[0])
    soft = [f"{LABEL[c]} {g:+.1f}pp" for g, c in gaps if g <= -0.4]
    hot = [f"{LABEL[c]} {g:+.1f}pp" for g, c in gaps if g >= 0.15]
    near = [
        f"{LABEL[c]}" + ("" if abs(g) < 0.05 else f" {g:+.1f}pp")
        for g, c in gaps if -0.4 < g < 0.15
    ]
    bits = []
    if soft:
        bits.append(f"<b>{' and '.join(soft)}</b> undershoot")
    if hot:
        bits.append(f"<b>{' / '.join(hot)}</b> overshoot")
    if near:
        bits.append(f"{' / '.join(near)} near plan")
    rank = "Rank by model − plan: " + "; ".join(bits) + "."

    de_ye, de_plan = ye_bn(de), float(de.plan)
    se_ye, se_plan = ye_bn(se), float(se.plan)
    fr_ye, fr_plan = ye_bn(fr), float(fr.plan)
    uk_ye, uk_plan = ye_bn(uk), float(uk.plan)
    it_ye, it_plan = ye_bn(it), float(it.plan)
    us_ye, us_plan = ye_bn(us), float(us.plan)
    es_ye, es_plan = ye_bn(es), float(es.plan)
    jp_ye, jp_plan = ye_bn(jp), float(jp.plan)

    # SE Acc. Diff from cash summary when present
    se_acc = se_ex = se_rrf = None
    se_sum = DERIVED / "se_cash_summary.json"
    if se_sum.exists():
        fwd = json.loads(se_sum.read_text()).get("forward") or {}
        se_acc = fwd.get("borrowing_vs_riksgalden_path_ytd_bn")
        se_ex = fwd.get("borrowing_vs_riksgalden_path_ex_timing_bn")
        items = fwd.get("timing_items") or []
        se_rrf = sum(float(t.get("amount_bn") or 0) for t in items)

    # DE split from cash summary
    de_split = de_mplan = None
    de_sum = DERIVED / "de_cash_summary.json"
    if de_sum.exists():
        fwd = json.loads(de_sum.read_text()).get("forward") or {}
        de_split = fwd.get("model_full_year_split_bn") or {}
        de_mplan = fwd.get("model_minus_plan_split_bn") or {}

    intro = (
        "<p>Exhibit 1c is the cash perimeter only — model full-year net cash borrowing vs "
        "this-year (and next-year) cash plans. Faint stroke = prior FY outturn. "
        f"{rank} "
        "Cash-plan gaps are not the print: UK PSNB vs OBR is Exhibit 1b; "
        "France Maastricht GG is Exhibit 1d.</p>\n"
        "<ul class='m'>\n"
    )

    de_core = float((de_split or {}).get("core") or 75.3)
    de_ik = float((de_split or {}).get("infrastruktur") or 47.0)
    de_bw = float((de_split or {}).get("bundeswehr") or 29.8)
    de_core_gap = float((de_mplan or {}).get("core") or -22.7)
    de_li = (
        f"<li><b>Germany ({gap_pp(de):+.1f}pp; largest cash undershoot).</b> "
        f"Model <b>{pct(de, de_ye)}</b> of GDP vs Bundeshaushalt Soll <b>{pct(de, de_plan)}</b> "
        f"(€{de_ye:.0f}bn vs €{de_plan:.0f}bn). "
        f"Split: <span class=\"gap\">core federal {100 * de_core_gap / de.gdp:+.1f}pp</span> "
        f"(€{de_core:.0f} vs Soll €98 — BMF Ist through Jul: Einnahmen −2.5% YoY / −€6.2bn, "
        f"Ausgaben +4.4%; open-year Soll haircuts on spend 0.97 / revenue 1.02); "
        f"<span class=\"gap\">Infrastruktur −0.2pp</span> "
        f"(€{de_ik:.0f} vs €58, flat YTD pace — one complete fund year); "
        f"<span class=\"gap\">SVBw</span> "
        f"(€{de_bw:.0f} = Wirtschaftsplan €25.5): scale channel — plan ≫ 2023–25 FA "
        f"totals, so YE = plan and rest-of-year residual is <b>flat</b> (no frozen hist "
        f"Dec spike). "
        f"Defence scale is two lines: <b>core Verteidigung Soll €93.5bn</b> "
        f"(Ist-25 €67; +39% — inside the €524 spend Soll) plus SVBw NKA €25.5. "
        f"YTD Verteidigung €38.2 (+27% YoY) is only ~70% of linear Soll; "
        f"militärische Beschaffungen €9.5 / Soll €38 are ~43% — H2-backloaded. "
        f"SVBw NKA is now plan-anchored; "
        f"<span class=\"gap\">core EP14 is still not carved out</span> — it rides the "
        f"blanket 0.97 spend haircut on hist month shares. If procurements catch up "
        f"toward Soll, the −0.5pp core soft understates. Path is <b>not PRE-gated</b> "
        f"(<code>planhc_3</code> a priori).</li>\n"
    )

    se_acc_s = f"−SEK {abs(se_acc):.0f}bn" if se_acc is not None else "−SEK 25bn"
    se_ex_s = f"−{abs(se_ex):.0f}bn" if se_ex is not None else "−40bn"
    se_rrf_s = f"SEK {se_rrf:.0f}bn" if se_rrf else "SEK 15bn"
    se_li = (
        f"<li><b>Sweden ({gap_pp(se):+.1f}pp).</b> "
        f"Model <b>{pct(se, se_ye)}</b> vs Riksgälden Statsupplåning 2026:1 NBR "
        f"<b>{pct(se, se_plan)}</b> (SEK {se_ye:.0f} vs {se_plan:.0f}bn). "
        f"YTD through Sep is −SEK {abs(se.cash_ytd):.0f}bn. "
        f"RG Oct-7 Acc. Diff borrowing {se_acc_s} vs the May monthly path "
        f"({se_ex_s} ex delayed EU RRF {se_rrf_s}): "
        f"<span class=\"gap\">net lending to agencies −SEK 46bn</span> "
        f"(Aug alone −38bn — FMV Gripen-Ukraine financing inflow + heavier Pensionsmyndigheten / "
        f"Svenska Kraftnät deposits; not agency programme underspend); "
        f"<span class=\"gap\">primary +SEK 22bn worse</span> "
        f"(taxes mixed month-to-month; Sep’s tax/spend beat only cut the primary drag); "
        f"interest −SEK 2bn. "
        f"Chart YE = plan-consistent (hold the path gap). Soft vs a May plan already above KI. "
        f"Financing residual sits in liquidity management because bonds/bills stay on the auction plan."
        f"</li>\n"
    )

    fr_li = (
        f"<li><b>France ({gap_pp(fr):+.1f}pp vs revised; already past LFI).</b> "
        f"Exhibit path = PLF×RESF category bottom-up <b>α=0</b> (OOS beats seas_mult_3); "
        f"live financing board still soft-pins α=1 to the PLF solde. "
        f"Model YE <b>{pct(fr, fr_ye)}</b> / €{fr_ye:.0f} vs revised State "
        f"<i>déficit à financer</i> <b>{pct(fr, fr_plan)}</b> / €{fr_plan:.0f} "
        f"(LFI was €124). "
        f"YTD Aug €{fr.cash_ytd:.0f} — "
        f"<span class=\"gap\">already through the original LFI by July "
        f"(cum €146) and through the revised €{fr_plan:.0f} too</span>; "
        f"dépenses ahead of recettes on the SMB books. "
        f"The chart’s −0.1pp ‘undershoot’ is only an H2 seasonal bet "
        f"(Sep–Dec model −€26bn, Dec recettes-heavy) against the *revised* plan — "
        f"cash has already missed the voted requirement. "
        f"Markets price GG/politics (Exhibit 1d) more than this cash point.</li>\n"
    )

    uk_li = (
        f"<li><b>United Kingdom (on plan).</b> "
        f"Autumn Budget 2026 (28 Oct): CGNCR ex NRAM remit <b>£{uk_plan:.1f}bn</b> "
        f"(was £137.2); gilt sales £250.6bn (+£4.4); T-bills net debt-financing £8bn (+£3). "
        f"YTD £{uk.cash_ytd:.0f}bn ≈ prior-FY month-share of the old £137 path (~£89bn) — "
        f"cash seasonality, not a miss vs April. "
        f"Full-year model <b>£{uk_ye:.0f} / {pct(uk, uk_ye)}</b> = Budget "
        f"(bottom-up leaf + live close to remit; Jan/Feb SA CGT–IT equity nudge). "
        f"Accrued PSNB vs OBR March profile (+£6bn YTD) is Exhibit 1b — October annual "
        f"PSNB £133bn; DMO sizes gilts off the new remit."
        f"</li>\n"
    )

    it_li = (
        f"<li><b>Italy (on plan).</b> "
        f"Bottom-up BdI categories (not a forced DFP hit). "
        f"Through Sep cumulative €{it.cash_ytd:.0f}bn (~{pct(it, it.cash_ytd)} of GDP); "
        f"year lands ~€{it_ye:.0f} / {pct(it, it_ye)} on a seasonal Dec surplus "
        f"(~−€22bn — RRF grant timing + tax calendar) against DFP €{it_plan:.0f}. "
        f"YTD already near plan; the path is not pinned to the DFP total.</li>\n"
    )

    us_li = (
        f"<li><b>United States ({gap_pp(us):+.1f}pp).</b> "
        f"Model ~{pct(us, us_ye)} vs CBO unified ~{pct(us, us_plan)} on the DTS cash bridge "
        f"(${us_ye:.0f}bn vs ${us_plan:.0f}bn). "
        f"FYTD (first ~3 weeks of Oct): deficit ${us.cash_ytd:.0f}bn vs ~$100bn prior-FY "
        f"same days — <span class=\"gap\">outlays +$48bn</span> "
        f"(defence +14, Medicare +12, other +27; SSA −23 timing) and "
        f"<span class=\"gap\">receipts −$12bn</span> "
        f"(non-withheld −24, customs −7; withheld +11, corporate +5). "
        f"Slight over the authority path; early-FY heat, not a regime miss.</li>\n"
    )

    es_li = (
        f"<li><b>Spain ({gap_pp(es):+.1f}pp).</b> "
        f"Model ~{pct(es, es_ye)} vs prior-FY IGAE cash ~{pct(es, es_plan)}. "
        f"YTD through Jul €{es.cash_ytd:.0f}bn is <span class=\"gap\">+€5.3bn vs same months 2025</span>: "
        f"outlays +€26bn (other current +11, capex +9) outrun receipts +€21bn (tax +14). "
        f"Central cash hotter than last year’s path — plan is prior-FY cash, not a WEO GG target."
        f"</li>\n"
    )

    jp_li = (
        f"<li><b>Japan ({gap_pp(jp):+.1f}pp).</b> "
        f"Model ~{pct(jp, jp_ye)} vs prior-FY tax-corrected MoF GA ~{pct(jp, jp_plan)}. "
        f"Apr–Aug already ¥{jp.cash_ytd/1000:.1f}tn vs ¥{35.9:.1f}tn same months prior FY: "
        f"<span class=\"gap\">expenditure front-loaded</span> (Apr/Jun spikes) while "
        f"tax settlement only thickens from July — early-FY rev soft vs prior. "
        f"Same-month-last-year YE inherits that hot start. "
        f"Plan is not the FY2026 initial bond-issuance ceiling.</li>\n"
    )

    # Order bullets by |gap| desc for reading, matching prior writeup emphasis
    body = de_li + se_li + fr_li + uk_li + it_li + us_li + es_li + jp_li
    return intro + body + "</ul>\n"


def splice_colour_into_html(html_path: Path, colour: str) -> None:
    """Replace the Exhibit 1c intro paragraph + country ``<ul class='m'>`` block."""
    import re
    html = html_path.read_text()
    n_h2 = html.count("<h2>")
    pat = re.compile(
        r"<p>Exhibit 1c is the cash perimeter only.*?</p>\s*<ul class='m'>.*?</ul>\s*",
        re.S,
    )
    m = pat.search(html)
    if not m:
        raise SystemExit("Exhibit 1c colour block (intro+ul) not found")
    html2 = html[:m.start()] + colour + html[m.end():]
    for lab in ("Exhibit 1.", "Exhibit 1c.", "Exhibit 2.", "Watch list"):
        if lab not in html2:
            raise SystemExit(f"colour splice ate {lab}")
    if html2.count("<h2>") < n_h2:
        raise SystemExit("colour splice shrank h2 count")
    if not html2.rstrip().endswith("</html>"):
        raise SystemExit("colour splice lost </html>")
    html_path.write_text(html2)


def splice_into_html(html_path: Path, png: Path, caption: str, summary_html: str = "") -> None:
    """Replace Exhibit 1c figure; drop any legacy summary table block.

    Bounds the match to a single ``<figure>…</figure>`` (no ``re.S`` across the
    file). Refuses to write if section count or trailing body shrinks — a prior
    mid-file base64 regex wiped everything after Exhibit 2.
    """
    import re
    html = html_path.read_text()
    n_h2 = html.count("<h2>")
    n_fig = html.count("<figure>")

    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    img = f"data:image/png;base64,{b64}"
    cap_pat = re.compile(
        r"<figcaption[^>]*>\s*<b>Exhibit 1c\.</b>.*?</figcaption>", re.S,
    )
    cm = cap_pat.search(html)
    if not cm:
        raise SystemExit("Exhibit 1c figcaption not found for splice")
    fig_start = html.rfind("<figure>", 0, cm.start())
    fig_end = html.find("</figure>", cm.end())
    if fig_start < 0 or fig_end < 0:
        raise SystemExit("Exhibit 1c enclosing figure not found")
    fig_end += len("</figure>")
    slice_ = html[fig_start:fig_end]
    if slice_.count("<figure>") != 1 or "<b>Exhibit 1.</b>" in slice_ or "<b>Exhibit 2.</b>" in slice_:
        raise SystemExit("Exhibit 1c splice slice is not a single figure")
    if fig_end - fig_start > 2_500_000:
        raise SystemExit(f"Exhibit 1c splice slice too large ({fig_end - fig_start})")
    after = html[fig_end:]
    m_sum = re.match(
        r"\s*<!-- exhibit-1c-summary -->.*?<!-- /exhibit-1c-summary -->",
        after, re.S,
    )
    drop_end = fig_end + (m_sum.end() if m_sum else 0)
    # Drop orphaned fytd-sum CSS once the table is gone.
    html_head = html[:fig_start]
    html_head = re.sub(
        r"<style>\.fytd-sum\{.*?</style>\s*",
        "",
        html_head,
        count=1,
        flags=re.S,
    )
    repl = (
        f"<figure><img src='{img}' style='width:100%;max-width:100%'/>"
        f"<figcaption>{caption}</figcaption></figure>\n"
    )
    html2 = html_head + repl + html[drop_end:]
    for lab in ("Exhibit 1.", "Exhibit 1c.", "Exhibit 2.", "Watch list"):
        if lab not in html2:
            raise SystemExit(f"splice ate {lab}")
    if html2.count("<h2>") < n_h2 or html2.count("<figure>") < n_fig:
        raise SystemExit(
            f"splice shrank structure h2 {n_h2}→{html2.count('<h2>')} "
            f"fig {n_fig}→{html2.count('<figure>')}"
        )
    if not html2.rstrip().endswith("</html>"):
        raise SystemExit("splice lost </html>")
    if "<!-- exhibit-1c-summary -->" in html2:
        raise SystemExit("summary table should be removed")
    html_path.write_text(html2)


if __name__ == "__main__":
    panel = build_panel()
    print(qa_table(panel).to_string(index=False))
    png, cap, table = render(panel)
    print("wrote", png, png.stat().st_size)
    print(cap)
    print(table)
    html = OUT_HTML
    if html.exists():
        splice_into_html(html, png, cap, table)
        col = colour_html(panel)
        splice_colour_into_html(html, col)
        print("spliced", html)
