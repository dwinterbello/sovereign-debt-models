"""Forward official stock under published QT guidance.

Each country keeps the same object the history already uses.

UK history is APF operations, not maturities. From the 17 September 2026
market notice, gilt auctions are paused and the £20bn-a-year sales are a
channel to the DMO, not to the private market. Private official DV01 on the
forecast is therefore zero. Maturing APF gilts still leave the official
*stock*; they are not booked again as private supply, because the DMO remit
already refinances them.

US history is the SOMA coupon stock. The 16 September 2026 directive rolls
all Treasury principal at auction and puts agency principal into bills.
Coupon face stock does not run off. The rollover is an add-on on top of the
announced auction size, and the forward books announced sizes, so the public
take is the whole forward row: official DV01 on the forecast is zero. History
nets the same way (gross is total accepted less SOMA add-ons; the add-ons are
taken out of the SOMA holdings change in `rv_metrics.supply_panel`).

Germany, France and Italy are the PSPP+PEPP allocated book. APP and PEPP
maturities are not reinvested, so a holding leaves the stock in the month
its residual maturity ends and is not replaced. That flow is official DV01
at a one-month duration. It is not an active sale. Spain has no ISIN-level
book: the published country PSPP+PEPP stock's trailing-12m pace is spread
flat and priced the same way (1m residual).

Japan is the BoJ holdings-by-issue book. The BoJ still buys, on the published
reduction plan, so forward official DV01 is positive and private supply is gross
issuance less those purchases.
"""

from __future__ import annotations

import pandas as pd

from issuance.paths import CACHE

EA = ("DE", "FR", "IT")
# Spain has no ISIN-level PSPP/PEPP book; forward uses the published country stock pace.
ES_SOURCE = "ecb_es_stock_pace"
UK_SOURCE = "apf_auctions_paused"
US_SOURCE = "fed_rollover"
EA_SOURCE = "ecb_no_reinvestment"
EA_HALT_SOURCE = "ecb_full_reinvestment"  # stock freeze: PEPP+PSPP principal reinvested
EA_PARTIAL_SOURCE = "ecb_partial_reinvestment"  # share of maturities replaced; rest still run off
ES_HALT_SOURCE = "ecb_es_full_reinvestment"
ES_PARTIAL_SOURCE = "ecb_es_partial_reinvestment"
JP_SOURCE = "boj_purchase_plan"
# Realistic EA policy case used by the note: slow APP/PEPP wind-down, not a full QT halt
# and not a restart of net structural buying. Share of principal reinvested (0–1).
EA_PARTIAL_REINVEST = 0.5


def _month_start(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _ea_redemptions(horizon: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / "official_isin_flow.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["country", "month", "official_stock_delta_mn", "official_face_mn"])
    isin = pd.read_parquet(path)
    isin = isin[isin["country"].isin(EA)].copy()
    if isin.empty or "resid_yrs" not in isin.columns:
        return pd.DataFrame(columns=["country", "month", "official_stock_delta_mn", "official_face_mn"])
    isin["month"] = pd.to_datetime(isin["month"])
    last = isin.sort_values("month").groupby(["country", "program", "isin"], as_index=False).tail(1)
    last["resid_yrs"] = pd.to_numeric(last["resid_yrs"], errors="coerce")
    last["official_stock_mn"] = pd.to_numeric(last["official_stock_mn"], errors="coerce")
    last = last[(last["resid_yrs"] > 0) & (last["official_stock_mn"] > 1)]
    snap = last["month"].max()
    last["mature"] = last.apply(
        lambda r: _month_start(r["month"] + pd.DateOffset(days=int(round(float(r["resid_yrs"]) * 365.25)))),
        axis=1,
    )
    # resid_yrs > 0 at the snapshot means a same-month maturity is still held
    last = last[(last["mature"] >= snap) & (last["mature"] <= horizon)]
    if last.empty:
        return pd.DataFrame(columns=["country", "month", "official_stock_delta_mn", "official_face_mn"])
    g = last.groupby(["country", "mature"], as_index=False)["official_stock_mn"].sum()
    g = g.rename(columns={"mature": "month", "official_stock_mn": "face"})
    g["official_stock_delta_mn"] = -g["face"]
    g["official_face_mn"] = -g["face"]
    return g[["country", "month", "official_stock_delta_mn", "official_face_mn"]]


def _uk_maturities(horizon: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / "apf_operations.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["country", "month", "official_stock_delta_mn"])
    apf = pd.read_parquet(path, columns=["op_dt", "isin", "maturity_dt", "apf_stock"])
    apf["op_dt"] = pd.to_datetime(apf["op_dt"])
    apf["maturity_dt"] = pd.to_datetime(apf["maturity_dt"], errors="coerce")
    last = apf.sort_values("op_dt").groupby("isin", as_index=False).tail(1)
    last["apf_stock"] = pd.to_numeric(last["apf_stock"], errors="coerce")
    asof = last["op_dt"].max()
    last = last[(last["apf_stock"] > 1) & last["maturity_dt"].notna()]
    last = last[(last["maturity_dt"] > asof) & (last["maturity_dt"] <= horizon)]
    if last.empty:
        return pd.DataFrame(columns=["country", "month", "official_stock_delta_mn"])
    last["month"] = last["maturity_dt"].map(_month_start)
    g = last.groupby("month", as_index=False)["apf_stock"].sum()
    g["country"] = "UK"
    g["official_stock_delta_mn"] = -g["apf_stock"]
    return g[["country", "month", "official_stock_delta_mn"]]


def _es_stock_pace(months: pd.Series) -> pd.DataFrame:
    """Spain: no ISIN aging book — next-12m official face = trailing-12m Δ(PSPP+PEPP) / 12.

    DV01 matches the EA ISIN path: redemptions priced at one-month residual (private
    supply absorbs the face; duration risk is not the maturity).
    """
    cols = ["country", "month", "official_dv01_usd", "official_face_mn", "official_stock_delta_mn"]
    path = CACHE / "ecb_qe_holdings.parquet"
    if not path.exists() or months is None or len(months) == 0:
        return pd.DataFrame(columns=cols)
    h = pd.read_parquet(path)
    h = h[h["country"].astype(str).str.upper().eq("ES")].copy()
    if h.empty:
        return pd.DataFrame(columns=cols)
    h["month"] = pd.to_datetime(h["month"]).map(_month_start)
    stock = h.groupby("month", as_index=True)["holdings_mn"].sum().sort_index()
    pace = float(stock.diff().dropna().tail(12).sum()) / 12.0  # negative under QT
    if not (pace < 0):
        return pd.DataFrame(columns=cols)
    from issuance.dv01 import attach_dv01
    from issuance.fetch_fx import fetch_fx

    rows = []
    for m in sorted(set(pd.to_datetime(months).map(_month_start))):
        # Same sign convention as EA ISIN redemptions: negative face leaving the stock.
        rows.append({"country": "ES", "month": m, "official_face_mn": pace,
                     "official_stock_delta_mn": pace, "gross_mn": pace,
                     "ccy": "EUR", "resid_mat_yrs": 1.0 / 12.0, "coupon": 3.0,
                     "tenor_bucket": "0-1y"})
    priced = attach_dv01(pd.DataFrame(rows), fetch_fx(force=False))
    priced["official_dv01_usd"] = priced["dv01_usd_mn"].astype(float)
    return priced[cols]


def _jp_purchases(months: pd.Series) -> pd.DataFrame:
    """BoJ forward: the published purchase plan (¥0.2tn a quarter less to Jan–Mar 2027), held at its
    last quarter after it. Purchases are priced at the trailing six months' DV01 per yen actually
    bought; maturing BoJ holdings leave the stock only."""
    cols = ["country", "month", "official_dv01_usd", "official_face_mn", "official_stock_delta_mn"]
    plan_path, isin_path = CACHE / "boj_purchase_plan.parquet", CACHE / "official_isin_flow.parquet"
    supply_path = CACHE / "rv_supply.parquet"
    if not plan_path.exists() or not isin_path.exists() or not supply_path.exists():
        return pd.DataFrame(columns=cols)
    plan = pd.read_parquet(plan_path)
    plan["month"] = pd.to_datetime(plan["month"])
    planned = plan.set_index("month")["planned_mn"].astype(float)
    isin = pd.read_parquet(isin_path)
    isin = isin[isin["country"].astype(str).eq("JP")].copy()
    isin["month"] = pd.to_datetime(isin["month"])
    sup = pd.read_parquet(supply_path)
    sup = sup[sup["country"].eq("JP")].copy()
    sup["month"] = pd.to_datetime(sup["month"])
    flow = isin.groupby("month")["official_mn"].sum()
    both = sup.set_index("month")["official_dv01_usd"].to_frame().join(flow.rename("face")).dropna().tail(6)
    per_mn = float(both["official_dv01_usd"].sum() / both["face"].sum()) if both["face"].sum() > 0 else 0.0
    last = isin.sort_values("month").groupby("isin").tail(1)
    last = last[(pd.to_numeric(last["official_stock_mn"], errors="coerce") > 0) & last["maturity_dt"].notna()]
    mature = last.groupby(pd.to_datetime(last["maturity_dt"]).map(_month_start))["official_stock_mn"].sum()
    rows = []
    for m in sorted(set(pd.to_datetime(months).map(_month_start))):
        buy = float(planned.get(m, planned.iloc[-1] if len(planned) else 0.0))
        rows.append({"country": "JP", "month": m, "official_dv01_usd": buy * per_mn, "official_face_mn": buy,
                     "official_stock_delta_mn": buy - float(mature.get(m, 0.0))})
    return pd.DataFrame(rows, columns=cols)


def qt_path(countries_months: pd.DataFrame, *, ea_reinvest: bool | float = False) -> pd.DataFrame:
    """One row per forecast country-month.

    official_dv01_usd is the private-supply adjustment (negative official
    flow raises private DV01). official_stock_delta_mn is the change in the
    official face stock, which is not the same number for the UK.

    ea_reinvest controls PSPP+PEPP / ES stock-pace runoff only:
      False / 0.0 — shipped path (no reinvestment; APP/PEPP wind-down).
      True  / 1.0 — full principal reinvestment / stock freeze (halt ceiling).
      0 < x < 1   — replace share ``x`` of maturing face; residual still runs off
                    (slowed structural wind-down; not a restart of net buying).
    """
    if isinstance(ea_reinvest, bool):
        share = 1.0 if ea_reinvest else 0.0
    else:
        share = float(ea_reinvest)
        if not (0.0 <= share <= 1.0):
            raise ValueError(f"ea_reinvest share must be in [0, 1], got {share}")

    base = countries_months[["country", "month"]].drop_duplicates().copy()
    base["month"] = pd.to_datetime(base["month"]).map(_month_start)
    if base.empty:
        return base
    horizon = base["month"].max() + pd.offsets.MonthEnd(0)
    ea = _ea_redemptions(horizon)
    uk_stock = _uk_maturities(horizon)

    from issuance.dv01 import attach_dv01
    from issuance.fetch_fx import fetch_fx

    rows = base.copy()
    rows["official_dv01_usd"] = 0.0
    rows["official_face_mn"] = 0.0
    rows["official_stock_delta_mn"] = 0.0
    rows["qt_source"] = ""
    rows.loc[rows["country"].eq("UK"), "qt_source"] = UK_SOURCE
    rows.loc[rows["country"].eq("US"), "qt_source"] = US_SOURCE
    rows.loc[rows["country"].isin(EA), "qt_source"] = EA_SOURCE
    rows.loc[rows["country"].eq("ES"), "qt_source"] = ES_SOURCE
    es = _es_stock_pace(rows.loc[rows["country"].eq("ES"), "month"])
    if not es.empty:
        rows = rows.merge(es, on=["country", "month"], how="left", suffixes=("", "_es"))
        for col in ("official_dv01_usd", "official_face_mn", "official_stock_delta_mn"):
            rows[col] = rows[col] + rows[col + "_es"].fillna(0)
        rows = rows.drop(columns=[c + "_es" for c in ("official_dv01_usd", "official_face_mn", "official_stock_delta_mn")])
    if not uk_stock.empty:
        uk_stock["month"] = pd.to_datetime(uk_stock["month"])
        rows = rows.merge(uk_stock, on=["country", "month"], how="left", suffixes=("", "_uk"))
        rows["official_stock_delta_mn"] = rows["official_stock_delta_mn"].fillna(0) + rows["official_stock_delta_mn_uk"].fillna(0)
        rows = rows.drop(columns=["official_stock_delta_mn_uk"])

    if not ea.empty:
        ea["month"] = pd.to_datetime(ea["month"])
        priced_in = ea.rename(columns={"official_face_mn": "gross_mn"}).copy()
        priced_in["ccy"] = "EUR"
        priced_in["resid_mat_yrs"] = 1.0 / 12.0
        priced_in["coupon"] = 3.0
        priced_in["tenor_bucket"] = "0-1y"
        priced = attach_dv01(priced_in, fetch_fx(force=False))
        ea = ea.merge(
            priced[["country", "month", "dv01_usd_mn"]],
            on=["country", "month"], how="left",
        )
        rows = rows.merge(
            ea[["country", "month", "official_face_mn", "official_stock_delta_mn", "dv01_usd_mn"]],
            on=["country", "month"], how="left", suffixes=("", "_ea"),
        )
        rows["official_dv01_usd"] = rows["official_dv01_usd"] + rows["dv01_usd_mn"].fillna(0)
        rows["official_face_mn"] = rows["official_face_mn"] + rows["official_face_mn_ea"].fillna(0)
        rows["official_stock_delta_mn"] = rows["official_stock_delta_mn"] + rows["official_stock_delta_mn_ea"].fillna(0)
        rows = rows.drop(columns=["official_face_mn_ea", "official_stock_delta_mn_ea", "dv01_usd_mn"])
    jp = _jp_purchases(rows.loc[rows["country"].eq("JP"), "month"])
    if not jp.empty:
        rows = rows.merge(jp, on=["country", "month"], how="left", suffixes=("", "_jp"))
        for col in ("official_dv01_usd", "official_face_mn", "official_stock_delta_mn"):
            rows[col] = rows[col] + rows[col + "_jp"].fillna(0)
        rows = rows.drop(columns=[c + "_jp" for c in ("official_dv01_usd", "official_face_mn", "official_stock_delta_mn")])
        rows.loc[rows["country"].eq("JP"), "qt_source"] = JP_SOURCE

    if share > 0:
        ea_mask = rows["country"].isin(EA) | rows["country"].eq("ES")
        keep = 1.0 - share  # share reinvested → that fraction of runoff disappears
        for col in ("official_dv01_usd", "official_face_mn", "official_stock_delta_mn"):
            rows.loc[ea_mask, col] = rows.loc[ea_mask, col] * keep
        if share >= 1.0 - 1e-12:
            rows.loc[rows["country"].isin(EA), "qt_source"] = EA_HALT_SOURCE
            rows.loc[rows["country"].eq("ES"), "qt_source"] = ES_HALT_SOURCE
        else:
            rows.loc[rows["country"].isin(EA), "qt_source"] = EA_PARTIAL_SOURCE
            rows.loc[rows["country"].eq("ES"), "qt_source"] = ES_PARTIAL_SOURCE
    return rows


def qt_halt_delta(countries_months: pd.DataFrame | None = None,
                  *, partial_share: float = EA_PARTIAL_REINVEST) -> pd.DataFrame:
    """Baseline runoff face vs partial reinvestment vs halt, one row per EA/ES country.

    Returns columns: country, runoff_face_mn (baseline private-supply face, positive
    when QT adds private supply), partial_face_mn, halt_face_mn (=0),
    delta_face_mn (= halt − runoff), delta_partial_face_mn (= partial − runoff).
    """
    if countries_months is None:
        months = pd.date_range("2026-10-01", periods=12, freq="MS")
        countries_months = pd.DataFrame(
            [(c, m) for c in (*EA, "ES") for m in months],
            columns=["country", "month"],
        )
    base = qt_path(countries_months, ea_reinvest=False)
    part = qt_path(countries_months, ea_reinvest=partial_share)
    halt = qt_path(countries_months, ea_reinvest=True)
    g0 = base.groupby("country", as_index=False)["official_face_mn"].sum()
    gp = part.groupby("country", as_index=False)["official_face_mn"].sum()
    g1 = halt.groupby("country", as_index=False)["official_face_mn"].sum()
    out = (g0.merge(gp, on="country", suffixes=("_base", "_partial"))
             .merge(g1, on="country"))
    out = out.rename(columns={
        "official_face_mn_base": "runoff_face_mn",
        "official_face_mn_partial": "partial_face_mn",
        "official_face_mn": "halt_face_mn",
    })
    # official_face_mn is negative under runoff; private supply relief = −Δ stock
    out["delta_face_mn"] = out["halt_face_mn"] - out["runoff_face_mn"]
    out["delta_partial_face_mn"] = out["partial_face_mn"] - out["runoff_face_mn"]
    out["runoff_eur_bn"] = (-out["runoff_face_mn"]) / 1000.0
    # delta_*_face_mn is positive when private supply falls (official runoff shrinks)
    out["partial_relief_eur_bn"] = out["delta_partial_face_mn"] / 1000.0
    out["partial_share"] = float(partial_share)
    return out
