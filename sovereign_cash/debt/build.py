"""Sovereign debt supply / demand: stock, rollovers, holders, debt service, scenarios.

Market debt is the central government's marketable book: coupon face by residual-maturity bucket and
central-bank holdings (`issuance/cache/float_share_fc.parquet`, history and the published forward),
bill stocks (`issuance/cache/bill_supply_monthly.parquet`) and the forward cash frameworks
(`sovereign_cash.investor.monthly`, US = fiscal supply bridge). Japan has no bill series: its bill
stock is backed out of the OECD refinancing share (``RFSH``) and labelled so. Sweden and Norway come
from their cash frameworks only (no coupon ladder).

Rollovers follow Dalio's convention: marketable debt that matures inside twelve months, i.e. the bill
stock (counted once) plus coupons in the 0-1y bucket. Gross issuance = rollovers + net issuance.

Debt service is central-government: interest and revenue from the fiscal authority where it publishes
them (CBO federal, OBR public-sector receipts and net debt interest, Cabinet Office general account),
Eurostat S1311 outturns otherwise, carried forward on the general-government growth of the EC / IMF
forecasts. Every ratio states its basis on the page.

Repricing: locked coupons stay on the ISIN. New bonds price off the government zero curve
(forward par). Bills with a maturity pay the 0–12m curve until they roll. Japan has no bill
maturity split, so the whole MoF stock pays the 3-month forward. The gap between that marketable
bill and the 2025 central outturn is frozen, in currency, and a rate shock does not move it.
``real_shock_bp`` is separate from the nominal shock. Reinvestment: of maturing *privately held* debt, a share r is rolled by its holders;
new private money needed = net issuance − central-bank net purchases + (1 − r) × private maturities.
Recession: the country's own 2007→2009 swing in the overall balance (% GDP, IMF) added for two
years, and its 2008→2009 revenue fall.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.build [--refresh]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.debt import sources as S

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DERIVED = HERE.parent / "derived"
FLOAT = REPO / "issuance" / "cache" / "float_share_fc.parquet"
BILLS = REPO / "issuance" / "cache" / "bill_supply_monthly.parquet"
MARKET = REPO / "issuance" / "cache" / "market_monthly.parquet"
LINES = REPO / "issuance" / "cache" / "lines.parquet"
SCALERS = HERE.parent / "data" / "scalers.json"

CC = S.CC
NAME = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy", "es": "Spain",
        "jp": "Japan", "se": "Sweden", "no": "Norway"}
CCY = {"us": "USD", "uk": "GBP", "de": "EUR", "fr": "EUR", "it": "EUR", "es": "EUR",
       "jp": "JPY", "se": "SEK", "no": "NOK"}
LADDER = ("us", "uk", "de", "fr", "it", "es", "jp")
Y0 = pd.Timestamp.today().year - 1  # last outturn year; forecasts start after it
REINVEST = (1.00, 0.95, 0.90)
YEARS = range(2000, 2032)
# Which fiscal authority anchors the central-government forecast, and its basis.
AUTHORITY = {"us": "CBO (federal, FY Oct–Sep)", "uk": "OBR (public sector, FY Apr–Mar)",
             "jp": "Cabinet Office (general account, FY Apr–Mar)", "de": "EC forecast + Eurostat S1311",
             "fr": "EC forecast + Eurostat S1311", "it": "EC forecast + Eurostat S1311",
             "es": "EC forecast + Eurostat S1311",
             "se": "EC forecast + Eurostat S1311", "no": "EC forecast + Eurostat S1311"}
# What happens to the central bank's maturing holdings over the next year (issuance.qt_forward).
CB_POLICY = {"us": "SOMA rolls all Treasury principal at auction (16 Sep 2026 directive)",
             "uk": "APF maturities not reinvested; sales go to the DMO, not the market",
             "de": "APP and PEPP maturities not reinvested", "fr": "APP and PEPP maturities not reinvested",
             "it": "APP and PEPP maturities not reinvested", "es": "APP and PEPP maturities not reinvested",
             "jp": "BoJ purchases on the published reduction path, below maturities",
             "se": "Riksbank holdings running off", "no": "Norges Bank holds no NGBs for policy"}


# ---------------------------------------------------------------- helpers
def _s(src: pd.DataFrame, cc: str, source: str, var: str) -> pd.Series:
    x = src[(src["country"] == cc) & (src["source"] == source) & (src["var"] == var)]
    return x.groupby("year")["value"].last().sort_index()


def no_debt(src: pd.DataFrame) -> pd.DataFrame:
    """Norway marketable bonds and bills by quarter end (Norges Bank key figures), NOK bn."""
    x = src[(src["source"] == "nb_debt")]
    if x.empty:
        return pd.DataFrame(columns=["bonds_mkt_bn", "bills_mkt_bn"])
    d = x.assign(t=pd.to_datetime(x["basis"])).pivot_table(index="t", columns="var", values="value")
    return d[["bonds_mkt_bn", "bills_mkt_bn"]].dropna().sort_index()


def _fx() -> dict[str, float]:
    sc = json.loads(SCALERS.read_text()) if SCALERS.exists() else {}
    fx = dict(sc.get("fx_usd_per_unit", {}))
    fx.setdefault("JPY", np.nan)
    return fx


def _jpy_fx(src: pd.DataFrame) -> float:
    """USD per JPY from IMF GDP in USD over OECD GDP in yen, latest common year."""
    usd, jpy = _s(src, "jp", "weo", "gdp_usd_bn"), _s(src, "jp", "oecd", "gdp_lcu_bn")
    y = max(set(usd.index) & set(jpy.index) & set(range(2000, pd.Timestamp.today().year)))
    return float(usd[y] / jpy[y])


def _extend(base: pd.Series, growth: pd.Series) -> pd.Series:
    """Carry `base` past its last year on the year-on-year growth of `growth`."""
    out = base.dropna().copy()
    if out.empty:
        return out
    g = growth.dropna()
    for y in sorted(g.index):
        if y > out.index.max() and (y - 1) in g.index and (y - 1) in out.index and g[y - 1] != 0:
            out[y] = out[y - 1] * g[y] / g[y - 1]
    return out.sort_index()


# ---------------------------------------------------------------- annual panel
def annual(src: pd.DataFrame) -> pd.DataFrame:
    from sovereign_cash.macro_authority import annual_authority

    rows = []
    for cc in CC:
        gdp = _s(src, cc, "oecd", "gdp_lcu_bn")
        if gdp.empty:
            gdp = _s(src, cc, "ameco", "gdp_lcu_bn")
        # Outturn levels through Y0; forward NGDP YoY from annual_authority (WEO real × house CPI).
        ngdp_a, _, _ = annual_authority(cc)
        nom = pd.Series({Y0: 1.0}, dtype=float)
        for y in range(Y0 + 1, max(YEARS) + 1):
            g = float(ngdp_a.get(y, np.nan)) if y in ngdp_a.index else np.nan
            if not np.isfinite(g):
                g = float(_s(src, cc, "weo", "real_growth").get(y, np.nan)) + float(
                    _s(src, cc, "weo", "cpi").get(y, np.nan))
            if not np.isfinite(g):
                break
            nom.loc[y] = float(nom.loc[y - 1]) * (1.0 + g / 100.0)
        gdp = _extend(gdp[gdp.index <= Y0], nom)
        d = pd.DataFrame(index=pd.Index(YEARS, name="year"))
        d["gdp_lcu_bn"] = gdp
        d["gdp_usd_bn"] = _s(src, cc, "weo", "gdp_usd_bn")
        d["pop_mn"] = _s(src, cc, "weo", "pop_mn")
        for v in ("debt_pct", "revenue_pct", "balance_pct", "primary_pct"):
            d[f"gg_{v}"] = _s(src, cc, "weo", v)
        # Commission (AMECO) is the GG peer basis through its horizon (2027). Prefer it over WEO/OECD
        # wherever both exist so history vs Commission is identical, not OECD−AMECO noise.
        for v, col in (("revenue_pct", "gg_revenue_pct"), ("debt_pct", "gg_debt_pct")):
            a = _s(src, cc, "ameco", v)
            if not a.empty:
                d[col] = a.combine_first(d[col])
        # GG gross interest: AMECO UYIGE × our GDP (exact % match), else OECD bn, else WEO net path
        ameco_ig = (_s(src, cc, "ameco", "interest_gross_pct") / 100 * gdp).dropna()
        gi = _s(src, cc, "oecd", "interest_gross_bn")
        if not ameco_ig.empty:
            gi = ameco_ig.combine_first(gi)
        elif gi.empty:
            gi = ameco_ig
        gi = _extend(gi, (_s(src, cc, "weo", "interest_pct").clip(lower=0.05) * gdp).dropna())
        d["gg_interest_bn"] = gi
        d["gg_interest_pct_rev"] = 100 * d["gg_interest_bn"] / (d["gg_revenue_pct"] / 100 * d["gdp_lcu_bn"])
        gdebt = d["gg_debt_pct"] / 100 * d["gdp_lcu_bn"]
        d["gg_eff_rate"] = 100 * d["gg_interest_bn"] / ((gdebt + gdebt.shift()) / 2)
        d["irl"] = _s(src, cc, "oecd", "irl")
        d["irs"] = _s(src, cc, "oecd", "irs")
        # central government revenue / interest
        gg_rev = d["gg_revenue_pct"] / 100 * d["gdp_lcu_bn"]
        if cc in ("us", "uk"):
            hr, hi = _s(src, cc, f"{cc}_hist", "revenue_bn"), _s(src, cc, f"{cc}_hist", "interest_net_bn")
        if cc == "us":
            rev = _s(src, cc, "cbo", "revenue_bn").combine_first(hr)
            intr = _s(src, cc, "cbo", "interest_net_bn").combine_first(hi)
            d["deficit_official_bn"] = -_s(src, cc, "cbo", "deficit_bn")
            d["eff_rate_official"] = _s(src, cc, "cbo", "eff_rate")
            d["cb_holdings_official_bn"] = _s(src, cc, "cbo", "cb_holdings_bn")
            d["debt_official_bn"] = _s(src, cc, "cbo", "debt_bn")
        elif cc == "uk":
            rev, intr = _s(src, cc, "obr", "revenue_ex_int_bn"), _s(src, cc, "obr", "interest_net_bn")
            # ONS outturns before the OBR table, receipts rescaled to the OBR's non-interest basis at the overlap
            ov = rev.index.intersection(hr.index)
            if len(ov):
                rev = rev.combine_first(hr * float(rev[ov[0]] / hr[ov[0]]))
            intr = intr.combine_first(hi)
            d["deficit_official_bn"] = _s(src, cc, "obr", "deficit_bn")
            d["debt_official_bn"] = _s(src, cc, "obr", "psnd_bn")
        elif cc == "jp":
            # tax only: CAO "other receipts" are mostly prior-year surpluses carried forward (FY2020 −¥15.6tn,
            # FY2021 +¥22.2tn, FY2023 +¥22.1tn against ¥5–7tn before Covid), not revenue
            rev = _s(src, cc, "cao_baseline", "tax_cg_bn")
            intr = _s(src, cc, "cao_baseline", "interest_cg_bn")
            d["deficit_official_bn"] = -(_s(src, cc, "cao_baseline", "balance_pct") / 100
                                         * _s(src, cc, "cao_baseline", "gdp_lcu_bn"))
            d["eff_rate_official"] = _s(src, cc, "cao_baseline", "eff_rate")
            d["debt_official_bn"] = (_s(src, cc, "cao_baseline", "debt_pct") / 100
                                     * _s(src, cc, "cao_baseline", "gdp_lcu_bn"))
        else:
            rev = _extend(_s(src, cc, "eurostat", "revenue_cg_bn"), gg_rev.dropna())
            intr = _extend(_s(src, cc, "eurostat", "interest_cg_bn"), d["gg_interest_bn"].dropna())
            if cc == "no":
                # S1311 interest is ~3x the coupons on government debt; use the bottom-up cash interest
                bu = _s(src, cc, "nb_interest", "interest_cg_bn")
                intr = bu if len(bu) else intr
            d["deficit_official_bn"] = -(_s(src, cc, "ameco", "balance_pct") / 100 * d["gdp_lcu_bn"])
        d["cg_revenue_bn"], d["cg_interest_bn"] = rev, intr
        d["cg_interest_pct_rev"] = 100 * d["cg_interest_bn"] / d["cg_revenue_bn"]
        d["country"] = cc
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- monthly market book
def monthly_book() -> pd.DataFrame:
    """Coupon stock, CB holdings, 0-1y coupons, bill stock by month (local bn), 2012 → forward."""
    f = pd.read_parquet(FLOAT)
    f["month"] = pd.to_datetime(f["month"])
    g = f.groupby(["country", "month"]).agg(coupon_bn=("stock_mn", "sum"), cb_bn=("official_face", "sum"),
                                            phase=("phase", "first")).reset_index()
    short = f[f["tenor_bucket"] == "0-1y"].groupby(["country", "month"]).agg(
        coupon_01_bn=("stock_mn", "sum"), cb_01_bn=("official_face", "sum")).reset_index()
    g = g.merge(short, on=["country", "month"], how="left")
    for c in ("coupon_bn", "cb_bn", "coupon_01_bn", "cb_01_bn"):
        g[c] /= 1e3
    b = pd.read_parquet(BILLS)[["country", "month", "bill_stock_bn"]]
    b["month"] = pd.to_datetime(b["month"])
    g = g.merge(b, on=["country", "month"], how="left")
    g["country"] = g["country"].str.lower()
    # The bond book is for the ladder countries only. The issuance float model also carries Sweden since 14 Nov
    # (today's CH1073 lines rolled back, no 0-1y bucket in most months); SE and NO stay on their cash frameworks
    # (``_bills_dec`` + bond redemptions), or their rollovers come out NaN and their debt off the CH1073 face.
    g = g[g["country"].isin(LADDER)]
    g = g.sort_values(["country", "month"])
    # bills: carry the last published level through the coupon forward
    g["bill_stock_bn"] = g.groupby("country")["bill_stock_bn"].ffill()
    g["bill_source"] = np.where(g["bill_stock_bn"].notna(), "published", None)
    return _es_bills(g.reset_index(drop=True))


def jp_mof_bills(refresh: bool = False) -> dict:
    """MoF Central Government Debt: short-term JGBs (≤1y) + Financing Bills, ¥bn.

    Source: https://www.mof.go.jp/english/jgbs/reference/gbb/suii.xls (億円 → ×0.1 = ¥bn).
    Both roll inside a year at money-market rates — the bill block for repricing."""
    path = HERE.parent / "data" / "debt" / "jp_mof_outstanding.xls"
    url = "https://www.mof.go.jp/english/jgbs/reference/gbb/suii.xls"
    if refresh or not path.is_file():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    d = pd.read_excel(path, header=None, engine="xlrd")
    # Column 2 holds the English subcategory; last numeric column is the latest quarter.
    def _row(label: str) -> float:
        for i in range(len(d)):
            cell = d.iloc[i, 2]
            if isinstance(cell, str) and label in cell:
                for c in range(d.shape[1] - 1, 2, -1):
                    v = pd.to_numeric(d.iloc[i, c], errors="coerce")
                    if np.isfinite(v):
                        return float(v) * 0.1  # 億円 → ¥bn
        raise KeyError(label)
    # Short-term under General Bonds (col 2, first match); Financing Bills label is in col 0.
    short = fb = None
    seen_short = 0
    for i in range(len(d)):
        row_txt = " ".join(str(d.iloc[i, c]) for c in range(min(3, d.shape[1])) if pd.notna(d.iloc[i, c]))
        if "Financing Bills" in row_txt:
            for c in range(d.shape[1] - 1, 2, -1):
                v = pd.to_numeric(d.iloc[i, c], errors="coerce")
                if np.isfinite(v):
                    fb = float(v) * 0.1
                    break
        if "Short-term (one year or less)" in row_txt and "Borrowings" not in row_txt and "Short-term (one year or less)" in str(d.iloc[i, 2]):
            # General Bonds short-term is the first such row (FILP short-term follows at 0)
            for c in range(d.shape[1] - 1, 2, -1):
                v = pd.to_numeric(d.iloc[i, c], errors="coerce")
                if np.isfinite(v):
                    if seen_short == 0:
                        short = float(v) * 0.1
                    seen_short += 1
                    break
    if short is None or fb is None:
        raise RuntimeError(f"MoF outstanding parse failed: short={short} fb={fb}")
    # latest header date
    hdr = d.iloc[2]
    asof = pd.NaT
    for c in range(d.shape[1] - 1, 2, -1):
        if pd.notna(hdr.iloc[c]) and "20" in str(hdr.iloc[c]):
            asof = pd.to_datetime(str(hdr.iloc[c]).split("\n")[-1], errors="coerce")
            break
    return {"bills_bn": short + fb, "short_jgb_bn": short, "financing_bills_bn": fb,
            "asof": asof, "source": "MoF Central Government Debt (short-term JGBs + Financing Bills)"}


ES_OUTSTANDING = REPO / "issuance" / "cache" / "es_outstanding.parquet"


def es_tesoro_bills() -> dict:
    """Letras outstanding, EUR bn, from the latest Tesoro 'valores en circulación' snapshot (issuance cache)."""
    o = pd.read_parquet(ES_OUTSTANDING)
    asof = pd.Timestamp(o["asof"].max())
    o = o[pd.to_datetime(o["asof"]) == asof]
    bills = float(o.loc[o["sec_type"].astype(str).eq("Bill"), "stock_mn"].sum()) / 1e3
    return {"bills_bn": bills, "asof": asof, "source": "Tesoro valores en circulación (Letras)"}


def _es_bills(book: pd.DataFrame, src: pd.DataFrame | None = None) -> pd.DataFrame:
    """Spain bill stock: no monthly Letras series in the bill book, so the Tesoro's latest published outstanding
    is held through the book, as Japan's MoF figure (``_jp_bills``). Without it Spain's marketable stock,
    rollovers and effective rate were NaN. No cache or no Letras: the book is returned unchanged."""
    es = book["country"] == "es"
    if not es.any() or book.loc[es, "bill_stock_bn"].notna().any():
        return book
    try:
        m = es_tesoro_bills()
    except Exception:  # noqa: BLE001 — issuance cache not built: Spain stays without bills
        return book
    if not (np.isfinite(m["bills_bn"]) and m["bills_bn"] > 0):
        return book
    book.loc[es, "bill_stock_bn"] = m["bills_bn"]
    book.loc[es, "bill_source"] = m["source"]
    return book


def _jp_bills(book: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """JP bill stock: MoF published short-term JGBs + Financing Bills; OECD RFSH only as fallback."""
    jp = book["country"] == "jp"
    try:
        m = jp_mof_bills()
        book.loc[jp, "bill_stock_bn"] = m["bills_bn"]
        book.loc[jp, "bill_source"] = m["source"]
        return book
    except Exception:
        pass
    rf = _s(src, "jp", "oecd", "refi_share")
    yr = book.loc[jp, "month"].dt.year.clip(upper=rf.index.max())
    share = yr.map(rf).astype(float)
    share = share.where(share < 1, share / 100)
    c, c01 = book.loc[jp, "coupon_bn"], book.loc[jp, "coupon_01_bn"]
    book.loc[jp, "bill_stock_bn"] = ((share * c - c01) / (1 - share)).clip(lower=0)
    book.loc[jp, "bill_source"] = "implied from OECD refinancing share"
    return book


# ---------------------------------------------------------------- monthly flows
US_LADDER = REPO / "us" / "fiscal" / "derived" / "auction_ladder.parquet"
US_BRIDGE = REPO / "us" / "fiscal" / "derived" / "supply_bridge.parquet"


def us_dts_years(bridge: pd.DataFrame | None = None, soma_bills_bn: float | None = None) -> pd.DataFrame:
    """Calendar years fully covered by the DTS supply bridge, $bn.

    Net issuance matches the monthly flow (coupon cash plus private net bills). Cash interest is
    the coupon schedule scaled by the realised ratio plus the off-coupon-day mean. Bills rolling
    in the year are the private bill stock at the previous December plus the SOMA bill book at
    the path's origin (the Fed's bill book is not grown here). Empty if no complete year."""
    if bridge is None:
        if not US_BRIDGE.exists():
            return pd.DataFrame()
        bridge = pd.read_parquet(US_BRIDGE)
        meta = US_BRIDGE.with_name("supply_bridge.json")
        if soma_bills_bn is None and meta.exists():
            soma_bills_bn = json.loads(meta.read_text())["origin_stocks_bn"]["soma_bills_bn"]
    if isinstance(bridge.index, pd.MultiIndex):
        bridge = bridge.loc["unchanged"]
    if "interest_cash_bn" not in bridge.columns:
        return pd.DataFrame()
    soma = 0.0 if soma_bills_bn is None else float(soma_bills_bn)
    rows = []
    for y, g in bridge.groupby(bridge.index.year):
        if len(g) < 12:
            continue
        prev = pd.Timestamp(int(y) - 1, 12, 1)
        bills = float(bridge.at[prev, "private_bill_stock_bn"]) + soma if prev in bridge.index else np.nan
        rows.append({"year": int(y),
                     "net_issuance_bn": float(g["net_coupons_bn"].sum() + g["private_net_bills_bn"].sum()),
                     "interest_cash_bn": float(g["interest_cash_bn"].sum()),
                     "bills_bn": bills})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def _uk_post_remit(d: pd.DataFrame) -> pd.DataFrame:
    """Months past the current remit carry only gilts the DMO has already scheduled (a fraction of a
    year's sales). A remit is sized so gilts fund the cash requirement less NS&I and other financing with
    bills flat, so set net gilts to that there."""
    p = DERIVED / "uk_cash_monthly.parquet"
    if not p.exists():
        return d
    u = pd.read_parquet(p).set_index("month")
    ass = u["gilts_assumed"].astype(bool) & u["is_forecast"].astype(bool)
    if not ass.any():
        return d
    need = (u["cash_requirement_bn"] - u["other_fin_bn"].fillna(0.0))[ass]
    d = d.set_index("month")
    idx = need.index.intersection(d.index)
    d.loc[idx, "net_bonds_bn"] = need[idx]
    d.loc[idx, "net_bills_bn"] = 0.0
    if "bond_gross_bn" in d:  # the assumed months are inside the horizon: gross = net + redemptions
        d.loc[idx, "bond_gross_bn"] = need[idx] + d.loc[idx, "bond_redemption_bn"].fillna(0.0)
    return d.reset_index()


def plan_horizon(d: pd.DataFrame) -> pd.Timestamp | None:
    """Last month a cash framework's forward path still issues bonds.

    `investor.monthly(naive_tail=True)` extends the cash requirement past the agency plan, but
    bond gross issuance there is zero while the ISIN redemption schedule keeps running, so net
    bonds collapse to −redemptions (FR: −9% of GDP a year by 2029). Months past this horizon are
    a maturity schedule, not a supply forecast, and must not enter any 12m flow series.
    Returns None when the frame carries no gross column (US bridge, JP book)."""
    if "bond_gross_bn" not in d or d["bond_gross_bn"].isna().all():
        return None
    fc = d[d["is_forecast"].astype(bool) & (d["bond_gross_bn"].fillna(0.0) > 0)]
    if fc.empty:
        return None
    return pd.Timestamp(fc["month"].max())


def flows_monthly(book: pd.DataFrame) -> pd.DataFrame:
    """Net bonds, net bills, bond redemptions and central-bank net coupon purchases by month (local bn).

    UK/DE/FR/IT/SE/NO: the cash frameworks (cash basis, history + forecast). US: the auction ladder
    (public cash, SOMA add-ons out) for history and the fiscal supply bridge forward. JP: change in
    the MoF-plan coupon book for history *and* forecast (bills 0) — not GA cash-requirement smly.
    ES joins UK/DE/FR/IT on the Tesoro cash frame (``net_bonds`` / ``net_bills``), not book Δ and
    not IGAE cash need. Central-bank purchases: change in official coupon face.
    """
    from sovereign_cash import investor as iv

    rows = []
    for cc in CC:
        if cc == "us":
            lad = pd.read_parquet(US_LADDER)
            lad.index = pd.to_datetime(lad.index)
            m = lad.resample("MS").sum()
            br = pd.read_parquet(US_BRIDGE).loc["unchanged"]
            br.index = pd.to_datetime(br.index)
            # the bridge's origin month is whole-month (actuals to date + path); the ladder's is partial
            start = br.index[0]
            h = m[m.index < start]
            d = pd.DataFrame({"month": h.index, "is_forecast": False,
                              "net_bonds_bn": (h["net_coupons"]).values, "net_bills_bn": h["net_bills"].values,
                              "bond_redemption_bn": (h["red_coupon"] + h["red_frn"] + h["red_tips"]).values})
            f = br
            d = pd.concat([d, pd.DataFrame({"month": f.index, "is_forecast": True,
                                            "net_bonds_bn": f["net_coupons_bn"].values,
                                            "net_bills_bn": f["private_net_bills_bn"].values,
                                            "bond_redemption_bn": f["coupon_red_bn"].values})])
        elif cc == "jp":
            # MoF auction-plan coupon book (issuance.jp_forward → float_share_fc): net = Δ coupon stock.
            # Do NOT splice jp_cash_monthly cash_requirement (GA smly) — that is fiscal cash, not JGB net
            # issuance, and was inventing a ~5pp-of-GDP forward step against a ~1pp trail.
            b = book[book["country"] == cc].set_index("month")
            d = pd.DataFrame({"month": b.index, "is_forecast": (b["phase"] != "history").values,
                              "net_bonds_bn": b["coupon_bn"].diff().values, "net_bills_bn": 0.0,
                              "bond_redemption_bn": np.nan})
        else:
            # UK/DE/FR/IT/ES/SE/NO cash frameworks — net_bonds from the financing identity.
            # ES must not use coupon-book Δ (float stock monthly nets are near-noise vs Tesoro).
            x = iv.monthly(cc, naive_tail=True)
            if x is None:
                raise RuntimeError(f"investor.monthly({cc!r}) returned None")
            d = x[["month", "is_forecast", "net_bonds_bn", "net_bills_bn", "bond_redemption_bn"]].copy()
            # gross bonds mark the framework's plan horizon: past it the naive tail carries the ISIN
            # redemption schedule with no issuance, so net bonds = −redemptions (see `plan_horizon`)
            d["bond_gross_bn"] = x["bond_gross_bn"].values if "bond_gross_bn" in x else np.nan
            if cc == "uk":
                d = _uk_post_remit(d)
        d["cc"] = cc
        bk = book[book["country"] == cc].set_index("month")
        b = bk["cb_bn"]
        nz = b[b > 0]
        # A holdings series that opens at a large level (BoJ from Mar-2013) is missing before, not zero.
        if len(nz) and nz.iloc[0] > 0.02 * bk.loc[nz.index[0], "coupon_bn"]:
            b = b.where(b.index >= nz.index[0])
        start = b.first_valid_index()
        if start is None:
            d["cb_net_bn"] = 0.0
        else:
            d["cb_net_bn"] = d["month"].map(b.diff())
            after = d["month"] > start
            d.loc[after, "cb_net_bn"] = d.loc[after, "cb_net_bn"].fillna(0.0)
        if cc == "us":
            d["cb_net_bn"] = np.where(d["is_forecast"], 0.0, d["cb_net_bn"])
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["month"] = pd.to_datetime(out["month"])
    return out.sort_values(["cc", "month"]).reset_index(drop=True)


def _window(fl: pd.DataFrame, cc: str, forecast: bool) -> pd.DataFrame:
    d = fl[(fl["cc"] == cc) & (fl["is_forecast"] == forecast)].sort_values("month")
    return d.head(12) if forecast else d.tail(12)


# ---------------------------------------------------------------- snapshot per country
def _yield(cc: str, series: str, src: pd.DataFrame) -> float:
    m = pd.read_parquet(MARKET)
    x = m[(m["country"] == cc.upper()) & (m["series"] == series)].sort_values("month")
    if len(x):
        return float(x["value"].iloc[-1])
    k = {"y10": "irl", "r3m": "irs"}[series]
    s = _s(src, cc, "oecd", k)
    return float(s.get(pd.Timestamp.today().year, s.iloc[-1])) if len(s) else np.nan


def _ladder_shares(cc: str, asof: pd.Timestamp) -> tuple[float, float]:
    """Share of live coupon face maturing in 12–24m and 24–36m (open-line snapshot)."""
    li = pd.read_parquet(LINES)
    li = li[(li["country"] == cc.upper()) & (li["sec_type"] != "Bill")]
    li = li[pd.to_datetime(li["maturity_dt"]) > asof]
    if li.empty:
        return np.nan, np.nan
    mo = (pd.to_datetime(li["maturity_dt"]) - asof).dt.days / 30.44
    tot = li["stock_mn"].sum()
    return (float(li.loc[(mo > 12) & (mo <= 24), "stock_mn"].sum() / tot),
            float(li.loc[(mo > 24) & (mo <= 36), "stock_mn"].sum() / tot))


def curve(cc: str, r3m: float, y10: float) -> tuple[np.ndarray, np.ndarray]:
    """Today's par curve (3m, 2y/3y, 10y, 20y, 30y where published), read as zero rates; SE/NO 3m + 10y."""
    m = pd.read_parquet(MARKET)
    m = m[m["country"] == cc.upper()].sort_values("month").groupby("series")["value"].last()
    pts = {0.25: r3m, 10.0: y10}
    for k, t in (("y2", 2.0), ("y3", 3.0), ("y20", 20.0), ("y30", 30.0)):
        if np.isfinite(m.get(k, np.nan)):
            pts[t] = float(m[k])
    pts = {t: v for t, v in pts.items() if np.isfinite(v)}
    x = np.array(sorted(pts))
    return x, np.array([pts[t] for t in x])


def forward(cv: tuple[np.ndarray, np.ndarray], start: float, tenor: np.ndarray | float) -> np.ndarray:
    """Forward rate (%) for `tenor` years starting `start` years ahead. Zero curve = monotone cubic through the
    published points (exact there, continuous forwards; a linear one turns the 2y/10y kinks into 50bp forward
    jumps), flat beyond its ends."""
    from scipy.interpolate import PchipInterpolator
    x, z = cv
    f = PchipInterpolator(x, z, extrapolate=False) if len(x) > 2 else (lambda t: np.interp(t, x, z))
    zt = lambda t: np.where(np.asarray(t) <= x[0], z[0], np.where(np.asarray(t) >= x[-1], z[-1], f(np.clip(t, x[0], x[-1]))))
    tenor = np.atleast_1d(np.asarray(tenor, dtype=float))
    out = (zt(start + tenor) * (start + tenor) - (float(zt(start)) * start if start > 0 else 0.0)) / tenor
    return out if out.size > 1 else float(out[0])


TENORS = np.arange(1, 41)


ISIN = REPO / "issuance" / "cache" / "dv01_isin.parquet"


def realised_ytd(cc: str, year: int) -> tuple[float, float]:
    """Gross-weighted auction yield of nominal + FRN bonds issued so far in `year` and the mean 3m rate over its
    published months (%); NaN where the bottom-up tape or market panel has none (SE, NO)."""
    yb = r3 = np.nan
    if ISIN.exists():
        d = pd.read_parquet(ISIN, columns=["country", "month", "sec_type", "gross_mn", "yield_avg", "coupon"])
        d = d[(d["country"] == cc.upper()) & (pd.to_datetime(d["month"]).dt.year == year)
              & d["sec_type"].isin(["Nominal", "FRN"]) & (d["gross_mn"] > 0)]
        y = d["yield_avg"].fillna(d["coupon"])
        ok = y.notna()
        if ok.any():
            yb = float(np.average(y[ok], weights=d.loc[ok, "gross_mn"]))
    m = pd.read_parquet(MARKET)
    m = m[(m["country"] == cc.upper()) & (m["series"] == "r3m") & (pd.to_datetime(m["month"]).dt.year == year)]
    if len(m.dropna(subset=["value"])):
        r3 = float(m["value"].mean())
    return yb, r3


def tenor_mix(cc: str, sched: dict) -> np.ndarray:
    """Share of new bond issuance by original maturity (years 1..40, bins centred on the integer). Bond-book
    countries: lines tapped in the last two years, maturity from the last tap, weighted by stock. NO: NGB
    volumes issued in the last two years. SE: outstanding lines by original maturity (no tap history)."""
    li = pd.read_parquet(LINES)
    li = li[(li["country"] == cc.upper()) & (li["sec_type"] != "Bill")]
    now = pd.Timestamp.today()
    tap = pd.to_datetime(li["last_tap"])
    li = li[(tap >= now - pd.DateOffset(years=2)) & (pd.to_datetime(li["maturity_dt"]) > now)]
    if len(li):
        t = (pd.to_datetime(li["maturity_dt"]) - pd.to_datetime(li["last_tap"])).dt.days / 365.25
        w = li["stock_mn"].values
    else:
        r = sched.get(f"{cc}_tenor", pd.Series(dtype=float))
        t, w = r.index.values.astype(float), r.values
    if not len(w) or np.sum(w) <= 0:
        return np.r_[np.zeros(9), 1.0, np.zeros(len(TENORS) - 10)]
    p = np.histogram(np.clip(np.round(t), 1, TENORS[-1]), bins=np.arange(0.5, TENORS[-1] + 1.5), weights=w)[0]
    return p / p.sum()


def _issue_slice(cc: str, months: int = 12) -> pd.DataFrame:
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    ops = pd.read_parquet(path, columns=["op_dt", "sec_type", "amount_issued"])
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    cut = ops["op_dt"].max() - pd.DateOffset(months=months)
    return ops[(ops["op_dt"] >= cut) & (ops["amount_issued"] > 0) & (~ops["sec_type"].isin(["Bill", "Foreign"]))]


def linker_issue_share(cc: str, months: int = 12) -> float:
    """Face share of coupon issuance that was Linker over the last ``months`` (ops tape)."""
    g = _issue_slice(cc, months)
    if g.empty:
        return 0.0
    return float(g.loc[g["sec_type"] == "Linker", "amount_issued"].sum() / g["amount_issued"].sum())


def floater_issue_share(cc: str, months: int = 12) -> float:
    """Face share of coupon issuance that was a floater. Those reset at the bill yield plus the spread."""
    g = _issue_slice(cc, months)
    if g.empty:
        return 0.0
    return float(g.loc[g["sec_type"] == "FRN", "amount_issued"].sum() / g["amount_issued"].sum())


def linker_marginal_yield(cc: str, fallback_nom: float) -> float:
    """Latest linker auction yield (%); else nominal − 2 (same fallback as ``issuance.pricing.market_curve``)."""
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    if not path.is_file():
        return fallback_nom - 2.0
    ops = pd.read_parquet(path, columns=["op_dt", "sec_type", "yield", "amount_issued"])
    g = ops[(ops["sec_type"] == "Linker") & ops["yield"].notna() & (ops["amount_issued"] > 0)]
    if g.empty:
        return fallback_nom - 2.0
    g = g.sort_values("op_dt")
    return float(g.iloc[-1]["yield"])


def _new_bond_yield(cc: str, r3m: float, y10: float, mix: np.ndarray) -> float:
    """Today's par yield on new bonds at the issuance tenor mix. Government curve where we have one."""
    gov = _gov_par(cc, 0.0, TENORS, 0.0) if cc in GOV_CURVE else None
    if gov is not None:
        return float(mix @ gov)
    return float(mix @ forward(curve(cc, r3m, y10), 0.0, TENORS))


def _fwd_frameworks(cc: str) -> dict:
    from sovereign_cash import investor as iv

    d = iv.monthly(cc)
    if d is None:
        return {}
    f = d[d["is_forecast"]].head(12)
    h = d[~d["is_forecast"]].tail(12)
    out = {"fw_window": f"{f['month'].iloc[0]:%b-%y}–{f['month'].iloc[-1]:%b-%y}" if len(f) else "",
           "fw_months": len(f)}
    for k, c in (("net_bonds", "net_bonds_bn"), ("net_bills", "net_bills_bn"), ("bond_red", "bond_redemption_bn"),
                 ("cash_req", "cash_requirement_bn")):
        out[f"fw_{k}"] = float(f[c].sum()) if c in f else np.nan
        out[f"tr_{k}"] = float(h[c].sum()) if c in h and len(h) == 12 else np.nan
    for c in ("bill_stock_bn", "bill_stock_implied_bn"):
        if c in d and d.loc[~d["is_forecast"], c].notna().any():
            out["fw_bill_stock_now"] = float(d.loc[~d["is_forecast"], c].dropna().iloc[-1])
            break
    if cc == "se" and "debt_bn" in d:
        out["se_debt_bn"] = float(d.loc[~d["is_forecast"], "debt_bn"].dropna().iloc[-1])
    return out


def snapshot(book: pd.DataFrame, fl: pd.DataFrame, ann: pd.DataFrame, src: pd.DataFrame,
             holders: pd.DataFrame, sched: dict) -> pd.DataFrame:
    fx = _fx()
    fx["JPY"] = _jpy_fx(src)
    rows = []
    for cc in CC:
        a = ann[ann["country"] == cc].set_index("year")
        # Stock ratios on Y0 (last outturn), not calendar year — calendar year is a WEO forecast
        # and was being compared to Y0/2025 authorities in QA.
        yr = Y0 if Y0 in a.index else pd.Timestamp.today().year
        r = {"cc": cc, "name": NAME[cc], "ccy": CCY[cc], "fx": fx.get(CCY[cc], np.nan), "authority": AUTHORITY[cc],
             "cb_policy": CB_POLICY[cc]}
        gdp = float(a.loc[yr, "gdp_lcu_bn"])
        r["gdp_bn"] = gdp
        r["gg_debt_pct"] = float(a.loc[yr, "gg_debt_pct"])
        r["gg_debt_year"] = int(yr)
        fw = _fwd_frameworks(cc)
        if cc in LADDER:
            b = book[book["country"] == cc].set_index("month")
            now = b.index[b["phase"] == "history"].max()
            fwd = min(now + pd.DateOffset(months=12), b.index.max())
            n, f_ = b.loc[now], b.loc[fwd]
            r.update(asof=f"{now:%b-%y}", coupon_bn=n["coupon_bn"], bills_bn=n["bill_stock_bn"],
                     bill_source=n["bill_source"], cb_bn=n["cb_bn"], coupon_01_bn=n["coupon_01_bn"],
                     cb_01_bn=n["cb_01_bn"])
            off = _official(src, cc)
            r["marketable_bn"] = (float(off.iloc[-1]) if len(off) else r["coupon_bn"] + r["bills_bn"]) + _uplift(src, cc)["now"]
            # MoF Financing Bills sit outside the JGB total; keep marketable ≥ coupon + bills
            if cc == "jp" and np.isfinite(r["bills_bn"]):
                r["marketable_bn"] = max(float(r["marketable_bn"]), float(r["coupon_bn"]) + float(r["bills_bn"]))
            r["fw_cb_net"] = 0.0 if cc == "us" else f_["cb_bn"] - n["cb_bn"]
            r["fw_cb_check"] = True  # compared with the trailing runoff once tr_cb_net is known (below)
            s12, s24 = _ladder_shares(cc, now)
            r["coupon_12_24_bn"], r["coupon_24_36_bn"] = s12 * n["coupon_bn"], s24 * n["coupon_bn"]
        else:
            debt, bills = fw.get("se_debt_bn", np.nan), fw.get("fw_bill_stock_now", np.nan)
            if cc == "no":
                nd = no_debt(src)
                if len(nd):
                    debt, bills = float(nd.iloc[-1].sum()), float(nd["bills_mkt_bn"].iloc[-1])
            r.update(asof="", bills_bn=bills, bill_source="published",
                     coupon_01_bn=np.nan, cb_bn=np.nan, cb_01_bn=0.0, fw_cb_net=0.0,
                     coupon_12_24_bn=np.nan, coupon_24_36_bn=np.nan, marketable_bn=debt)
            r["coupon_bn"] = debt - r["bills_bn"] if np.isfinite(debt) else np.nan
        tr, fwd_ = _window(fl, cc, False), _window(fl, cc, True)
        r["tr_window"] = f"{tr['month'].iloc[0]:%b-%y}–{tr['month'].iloc[-1]:%b-%y}"
        r["fw_window"] = f"{fwd_['month'].iloc[0]:%b-%y}–{fwd_['month'].iloc[-1]:%b-%y}"
        r["fw_months"] = len(fwd_)
        for k, w in (("tr", tr), ("fw", fwd_)):
            num = lambda c: pd.to_numeric(w[c], errors="coerce").astype(float)  # noqa: E731 — pd.NA rows (JP bills unknown)
            r[f"{k}_net_bonds"], r[f"{k}_net_bills"] = float(num("net_bonds_bn").sum()), float(num("net_bills_bn").sum())
            r[f"{k}_bond_red"] = float(num("bond_redemption_bn").sum(min_count=1))
        r["tr_cb_net"] = float(tr["cb_net_bn"].sum())
        if r.pop("fw_cb_check", False) and r["tr_cb_net"] and abs(r["fw_cb_net"]) > 2.5 * abs(r["tr_cb_net"]):
            r["fw_cb_net"] = r["tr_cb_net"]  # forward CB leg missing from the ladder (Spain) → trailing runoff
        if not np.isfinite(r["coupon_01_bn"]):
            r["coupon_01_bn"] = r["fw_bond_red"]
        if cc in LADDER and cc not in ("jp", "es"):  # JP and ES bills are one level held flat: no history
            hb = b.loc["2012-01-01":"2019-12-31"]
            r["bill_share_norm"] = float((hb["bill_stock_bn"] / (hb["coupon_bn"] + hb["bill_stock_bn"])).median())
        else:
            r["bill_share_norm"] = np.nan
        r["net_issuance_fwd"] = r["fw_net_bonds"] + r["fw_net_bills"]
        r["net_issuance_tr"] = r["tr_net_bonds"] + r["tr_net_bills"]
        r["bill_share_fwd"] = r["fw_net_bills"] / r["net_issuance_fwd"] if r["net_issuance_fwd"] > 0 and cc != "jp" else np.nan
        r["bill_share_tr"] = r["tr_net_bills"] / r["net_issuance_tr"] if r["net_issuance_tr"] > 0 and cc != "jp" else np.nan
        r["bill_share_stock"] = r["bills_bn"] / r["marketable_bn"] if np.isfinite(r.get("marketable_bn", np.nan)) else np.nan
        r["rollovers_bn"] = r["bills_bn"] + r["coupon_01_bn"]
        r["gross_issuance_fwd"] = r["rollovers_bn"] + r["net_issuance_fwd"]
        r["cb_share"] = r["cb_bn"] / r["coupon_bn"] if np.isfinite(r.get("cb_bn", np.nan)) else np.nan
        # central-government revenue and interest, next twelve months ≈ current (fiscal) year
        r["revenue_bn"] = float(a.loc[yr, "cg_revenue_bn"]) if np.isfinite(a.loc[yr, "cg_revenue_bn"]) else np.nan
        iy = yr - 1 if cc in BUDGET_BASIS else yr
        r["interest_bn"] = float(a.loc[iy, "cg_interest_bn"]) if np.isfinite(a.loc[iy, "cg_interest_bn"]) else np.nan
        r["revenue_next_bn"] = float(a.loc[yr + 1, "cg_revenue_bn"])
        r["interest_pct_rev"] = 100 * r["interest_bn"] / r["revenue_bn"]
        r["rollovers_pct_rev"] = 100 * r["rollovers_bn"] / r["revenue_bn"]
        r["debt_service_pct_rev"] = r["interest_pct_rev"] + r["rollovers_pct_rev"]
        mk = r.get("marketable_bn", np.nan)
        r["eff_rate"] = 100 * r["interest_bn"] / mk if np.isfinite(mk) and mk > 0 else np.nan
        off = a.get("eff_rate_official")
        if off is not None and np.isfinite(off.get(yr, np.nan)):
            r["eff_rate_official"] = float(off[yr])
        r["y10"], r["r3m"] = _yield(cc, "y10", src), _yield(cc, "r3m", src)
        bs = r["bills_bn"] / r["gross_issuance_fwd"] if r["gross_issuance_fwd"] > 0 else np.nan
        r["new_bond_yield"] = _new_bond_yield(cc, r["r3m"], r["y10"], tenor_mix(cc, sched))
        bill_leg = bs * r["r3m"] if np.isfinite(bs) and bs > 0 else 0.0
        r["marginal_rate"] = bill_leg + (1 - (bs if np.isfinite(bs) else 0.0)) * r["new_bond_yield"]
        r["debt_per_capita_usd"] = (1e3 * r["gg_debt_pct"] / 100 * gdp * r["fx"] / float(a.loc[yr, "pop_mn"]))
        # holders (IMF investor base, general government, nominal)
        h = holders[holders["country"] == cc.upper()]
        if len(h):
            q = h["quarter_end"].max()
            last = h[h["quarter_end"] == q].set_index("holder_type")
            prev = h[h["quarter_end"] == q - pd.DateOffset(years=1)]
            prev = prev.set_index("holder_type") if len(prev) else None
            r["holders_q"] = f"{q:%Y}Q{(q.month - 1) // 3 + 1}"
            for t in ("dcb", "db", "dnb", "fo", "fb", "fnb"):
                r[f"h_{t}"] = float(last.loc[t, "share_pct"]) if t in last.index else np.nan
                if prev is not None and t in prev.index and t in last.index:
                    r[f"hn_{t}"] = float(last.loc[t, "value_local_bn"] - prev.loc[t, "value_local_bn"])
            r["h_foreign"] = r.get("h_fo", 0) + r.get("h_fb", 0) + r.get("h_fnb", 0)
            tot_net = sum(r.get(f"hn_{t}", 0.0) for t in ("dcb", "db", "dnb", "fo", "fb", "fnb"))
            r["hn_total"] = tot_net
            r["hn_private"] = tot_net - r.get("hn_dcb", 0.0)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scenarios
def _elastic_buying_bn(r: pd.Series, haircut: float = 1.0) -> float:
    """Last-4Q realized private buyer mix (evidence roles), local bn.

    Inelastic + elastic sleeves that absorbed last year — not an inelastic-only
    shortfall. ``haircut`` scales the total mix. Falls back to NaN if missing.
    """
    cc = str(r["cc"])
    gdp = float(r["gdp_bn"])
    try:
        from sovereign_cash.debt.note.rebuild_ex8_absorption import absorption_stack
        st = absorption_stack(cc, gdp)
        pct = haircut * (max(float(st["inelastic"]), 0.0) + max(float(st["elastic"]), 0.0))
        return pct / 100.0 * gdp if pct > 0 else np.nan
    except Exception:
        return np.nan


def reinvestment(snap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in snap.iterrows():
        cb01 = r["cb_01_bn"] if np.isfinite(r["cb_01_bn"]) else 0.0
        priv_mat = r["rollovers_bn"] - cb01
        # non-US central banks do not roll: their maturities reach the market through fw_cb_net
        priv_supply = r["net_issuance_fwd"] - (r["fw_cb_net"] if r["cc"] != "us" else 0.0)
        # Legacy whole-book maturity haircut kept for vs_private_buying_last_year.
        # Elastic stress: private-net need ÷ (elastic stack × reinvestment share).
        elast_raw = _elastic_buying_bn(r, haircut=1.0)
        for rr in REINVEST:
            need = priv_supply + (1 - rr) * priv_mat
            elast_h = elast_raw * rr if np.isfinite(elast_raw) else np.nan
            rows.append({"cc": r["cc"], "reinvest": rr, "private_maturities_bn": priv_mat,
                         "private_net_supply_bn": priv_supply, "new_money_bn": need,
                         "new_money_usd_bn": need * r["fx"], "new_money_pct_gdp": 100 * need / r["gdp_bn"],
                         "vs_private_buying_last_year": need / r["hn_private"]
                         if r.get("hn_private", 0) and r["hn_private"] > 0 else np.nan,
                         # Exhibit 8a definition: private net / (elastic × rr)
                         "vs_elastic_buying_last_year": priv_supply / elast_h
                         if np.isfinite(elast_h) and elast_h > 0 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- history for charts
def history(book: pd.DataFrame, fl: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """Monthly 12m net issuance, private net supply, bill share and CB share, % GDP."""
    rows = []
    for cc in CC:
        d = fl[fl["cc"] == cc].sort_values("month")
        hz = plan_horizon(d)
        if hz is not None:
            d = d[d["month"] <= hz]
        d = d.set_index("month")
        g = ann[ann["country"] == cc].set_index("year")["gdp_lcu_bn"]
        gm = d.index.year.map(g).astype(float)
        net12 = (d["net_bonds_bn"] + d["net_bills_bn"]).rolling(12).sum()
        cb12 = d["cb_net_bn"].rolling(12).sum()
        out = pd.DataFrame({"cc": cc, "month": d.index, "is_forecast": d["is_forecast"].values,
                            "net12_pct_gdp": (100 * net12 / gm).values,
                            "bills12_pct_gdp": (100 * d["net_bills_bn"].rolling(12).sum() / gm).values,
                            "cb12_pct_gdp": (100 * cb12 / gm).values,
                            "private12_pct_gdp": (100 * (net12 - cb12) / gm).values})
        b = book[book["country"] == cc].set_index("month")
        if len(b):
            mk = b["coupon_bn"] + b["bill_stock_bn"]
            gb = b.index.year.map(g).astype(float)
            ext = pd.DataFrame({"bill_share_stock": (b["bill_stock_bn"] / mk).values,
                                "cb_share": (b["cb_bn"] / b["coupon_bn"]).values,
                                "rollover_pct_gdp": (100 * (b["bill_stock_bn"] + b["coupon_01_bn"]) / gb).values},
                               index=b.index)
            out = out.merge(ext, left_on="month", right_index=True, how="left")
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# Average household size: Census CPS 2024 (US), ONS Families and households 2024 (UK), 2020 census (JP),
# Eurostat ilc_lvph01 2025 (rest).
# CAO's current and later fiscal years are on the budget basis, whose interest carries a rate cushion
# (FY2026 ¥13.0tn vs FY2025 settled ¥9.1tn): the snapshot reads the last settled year.
BUDGET_BASIS = ("jp",)
HH_SIZE = {"us": 2.51, "uk": 2.36, "jp": 2.21, "de": 2.0, "fr": 2.1, "it": 2.2, "es": 2.5,
           "se": 2.0, "no": 2.0}
FIRST_YEAR = 2015


OFFICIAL = ("us", "fr")  # marketable stock incl. inflation uplift from the issuer's statement (MSPD, INSEE/AFT)


def _official(src: pd.DataFrame, cc: str, var: str = "marketable_bn") -> pd.Series:
    o = src[(src["source"] == "official_stock") & (src["country"] == cc) & (src["var"] == var)]
    return pd.Series(o["value"].values, index=pd.to_datetime(o["basis"] + "-01")).sort_index()


def _uplift(src: pd.DataFrame, cc: str) -> dict:
    """Index-linked stock and uplift (local bn). Lines and coupon books carry linkers at par: the UK uplift is
    rebuilt per gilt from the DMO (exact to D1A) and added to the book ("dec" year ends, "now"); US / FR debt
    is the issuer's own stock, which already includes it. "il_now" = linkers incl. uplift today; `infl(y)` =
    the accrual rate: WEO CPI, plus for the UK today's RPI − CPI wedge (RPI-linked until 2030)."""
    weo = _s(src, cc, "weo", "cpi")
    out = {"dec": pd.Series(dtype=float), "now": 0.0, "il_now": 0.0}
    wedge = 0.0
    if cc == "uk":
        u = src[src["source"] == "uk_uplift"]
        g = lambda v: u[u["var"] == v].set_index("year")["value"]
        now, il, ry = g("il_uplift_now_bn"), g("il_nominal_now_bn"), g("rpi_yoy")
        if len(now):
            out.update(dec=g("il_uplift_bn"), now=float(now.iloc[0]), il_now=float(il.iloc[0] + now.iloc[0]))
            wedge = float(ry.iloc[0]) - float(weo.get(int(ry.index[0]), np.nan)) if len(ry) else 0.0
            wedge = wedge if np.isfinite(wedge) else 0.0
    elif cc in OFFICIAL:
        il = _official(src, cc, "linker_bn")
        out["il_now"] = float(il.iloc[-1]) if len(il) else 0.0
    out["infl"] = lambda y: (float(weo.get(y, weo.iloc[-1])) + wedge) / 100 if len(weo) else 0.0
    return out


def _bills_dec(src: pd.DataFrame) -> dict[str, pd.Series]:
    """Year-end bill stock where there is no bond book: SE cash framework, NO Norges Bank key figures."""
    from sovereign_cash import investor as iv

    out = {}
    se = iv.monthly("se")
    se = se[~se["is_forecast"].astype(bool)].set_index("month")["bill_stock_implied_bn"].dropna()
    out["se"] = se[se.index.month == 12].groupby(se.index[se.index.month == 12].year).last()
    nd = no_debt(src)["bills_mkt_bn"]
    out["no"] = nd[nd.index.month == 12].groupby(nd.index[nd.index.month == 12].year).last()
    return out


def country_annual(book: pd.DataFrame, fl: pd.DataFrame, ann: pd.DataFrame, y_last: int,
                   asof: pd.Timestamp, snap: pd.DataFrame, bills_dec: dict | None = None,
                   src: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dalio's table per country by calendar year to `y_last`: marketable debt at year end (today's stock
    rolled on net issuance; index-linked uplift not included), net issuance, rollovers (bills + coupons due
    inside the year, at its start), central-government interest and revenue (authority outturns). Without a
    bond book (SE, NO) rollovers = bills at the prior year end + that year's bond redemptions."""
    bills_dec = bills_dec or {}
    rows = []
    stock0 = snap.set_index("cc")["marketable_bn"]
    for cc in CC:
        b = book[book["country"] == cc].set_index("month").sort_index()
        f = fl[fl["cc"] == cc].set_index("month").sort_index()
        a = ann[ann["country"] == cc].set_index("year")
        mk = (b["coupon_bn"] + b["bill_stock_bn"].fillna(0)) if len(b) else pd.Series(dtype=float)
        net_m = f["net_bonds_bn"] + f["net_bills_bn"]
        for y in range(FIRST_YEAR, y_last + 1):
            dec = pd.Timestamp(y, 12, 1)
            r = {"cc": cc, "year": y, "forecast": bool(dec > asof)}
            off = _official(src, cc) if src is not None else pd.Series(dtype=float)
            if len(off):
                a_m = min(asof, off.index.max())
                anchor = float(off[a_m])
            else:
                a_m = asof
                anchor = float(mk[asof]) if asof in mk.index else float(stock0.get(cc, np.nan))
            if np.isfinite(anchor) and len(net_m) and net_m.index.min() <= min(dec, a_m):
                # Anchored on the latest stock and rolled on net issuance both ways, so Δdebt = net issuance:
                # some books update in steps (UK −£185bn in Jul-25) and the forward book adds no new issues.
                lo, hi = min(dec, a_m), max(dec, a_m)
                roll = net_m[(net_m.index > lo) & (net_m.index <= hi)].sum()
                r["debt_bn"] = anchor + float(roll if dec > a_m else -roll)
            if len(off) and dec <= a_m and dec in off.index:
                r["debt_bn"] = float(off[dec])
            if src is not None and "debt_bn" in r:
                up = _uplift(src, cc)
                if dec <= a_m:
                    r["debt_bn"] += float(up["dec"].get(y, 0.0))
                else:  # today's uplift, accrued to the year end
                    r["debt_bn"] += up["now"] + up["il_now"] * up["infl"](y) * (12 - a_m.month) / 12
            if len(b):
                start = b.loc[:pd.Timestamp(y - 1, 12, 1)]
                if len(start) and start.index[-1].year == y - 1:
                    r["rollovers_bills_bn"] = float(start["bill_stock_bn"].fillna(0).iloc[-1])
                    r["rollovers_bn"] = r["rollovers_bills_bn"] + float(start["coupon_01_bn"].iloc[-1])
            fy = f[f.index.year == y]
            if not len(b) and len(fy) == 12 and cc in bills_dec and np.isfinite(bills_dec[cc].get(y - 1, np.nan)):
                r["rollovers_bills_bn"] = float(bills_dec[cc][y - 1])
                r["rollovers_bn"] = r["rollovers_bills_bn"] + float(fy["bond_redemption_bn"].sum())
            if len(fy) == 12:
                r["net_issuance_bn"] = float((fy["net_bonds_bn"] + fy["net_bills_bn"]).sum())
            r["interest_bn"] = a["cg_interest_bn"].get(y, np.nan)
            r["revenue_bn"] = a["cg_revenue_bn"].get(y, np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


# Rolling today's par stock back on cash net issuance misses SOMA add-ons, inflation uplift and bill discount:
# US Dec-2015 came out $16.8tn against $13.4tn published. US / FR history is the issuer's stock (OFFICIAL).

SE_SECURITIES = HERE.parent / "data" / "se" / "outstanding_securities.csv"


def redemption_schedule(src: pd.DataFrame) -> dict[str, pd.Series]:
    """Bond redemptions by calendar year of today's outstanding lines, local bn: issuance lines (US/UK/DE/FR/
    IT/JP), Riksgälden's securities list (SE), Norges Bank primary-market volumes (NO)."""
    out = {}
    li = pd.read_parquet(LINES)
    li = li[li["sec_type"] != "Bill"].assign(y=lambda d: pd.to_datetime(d["maturity_dt"]).dt.year)
    for cc, g in li.groupby("country"):
        out[cc.lower()] = g.groupby("y")["stock_mn"].sum() / 1e3
    se_bad = ""
    if SE_SECURITIES.exists():
        from sovereign_cash.se.fetch import outstanding_problem
        se_bad = outstanding_problem(pd.read_csv(SE_SECURITIES))
        if se_bad:
            print(f"  debt: Riksgälden securities list unusable ({se_bad}); Sweden without a redemption schedule")
    if SE_SECURITIES.exists() and not se_bad:
        se = pd.read_csv(SE_SECURITIES)
        se = se[~se["Security/ Market"].isin(["STB", "CP"])]
        # SEK value incl. FX / inflation compensation; thousands separated by U+00A0, which arrow-backed \s misses
        amt = pd.to_numeric(se["Current exchange rate incl. inflation compensation"].astype(str)
                            .str.replace("[\\s\u00a0\u202f]", "", regex=True), errors="coerce")
        assert amt.notna().all(), "Riksgälden securities list: unparsed amounts"
        mat = pd.to_datetime(se["Maturity date"], errors="coerce")
        out["se"] = (amt / 1e9).groupby(mat.dt.year).sum()
        tenor = ((mat - pd.to_datetime(se["Issue date"], errors="coerce")).dt.days / 365.25).round().clip(lower=1)
        out["se_tenor"] = (amt / 1e9).groupby(tenor).sum()
    no = src[(src["source"] == "nb_interest") & (src["var"] == "redemption_sched_bn")]
    out["no"] = no.groupby("year")["value"].sum()
    nt = src[(src["source"] == "nb_interest") & (src["var"] == "issued_2y_by_tenor_bn")]
    out["no_tenor"] = nt.groupby("year")["value"].sum()
    return out


Y_END = 2030  # match US cash path through FY2030 / calendar 2030
DEFICIT_FREE = ("no",)
COVID = (2020, 2021)  # NGBs fund market-making and state lending; the oil fund covers the deficit


def _gap(hist: pd.DataFrame, ann: pd.DataFrame, y0: int) -> pd.DataFrame:
    """Net issuance minus the WEO general-government deficit, pp GDP, outturn years ex-Covid."""
    a = ann.set_index(["country", "year"])
    h = hist[(hist["year"] <= y0) & ~hist["year"].isin(COVID) & ~hist["cc"].isin(DEFICIT_FREE)].copy()
    k = list(zip(h["cc"], h["year"]))
    h["gap"] = 100 * h["net_issuance_bn"].values / a.loc[k, "gdp_lcu_bn"].values + a.loc[k, "gg_balance_pct"].values
    return h.dropna(subset=["gap"]).pivot(index="year", columns="cc", values="gap")


def gap_persistence(hist: pd.DataFrame, ann: pd.DataFrame, y0: int) -> float:
    """Pooled one-year autocorrelation of the gap around each country's mean (0.26 on 2015–25, n 49; two-year
    −0.18): a framework year's gap is mostly transitory."""
    p = _gap(hist, ann, y0)
    d = p - p.mean()
    x, y = d.shift(1).stack(), d.stack()
    j = x.index.intersection(y.index)
    j = [i for i in j if i[0] - 1 in p.index]  # consecutive years only (Covid years are absent)
    x, y = x[j].values, y[j].values
    return float(np.clip(x @ y / (x @ x), 0.0, 1.0)) if len(x) > 10 else 0.0


GOV_CURVE = ("us", "uk", "de", "jp", "fr", "it", "es")
_LEGACY: dict[int, pd.DataFrame] = {}


_COMP: dict[int, pd.DataFrame] = {}


def _book(y0: int) -> pd.DataFrame:
    from sovereign_cash.debt.cashflow import book
    return book(y0)


def _marketable(y0: int, src: pd.DataFrame | None) -> pd.DataFrame | None:
    if y0 in _COMP:
        return _COMP[y0]
    from sovereign_cash.debt.cashflow import attach_japan_bills, components
    years = list(range(y0, Y_END + 1))

    def infl(cc: str, year: int) -> float:
        if src is None:
            return 0.0
        return float(_uplift(src, cc)["infl"](year))

    frame = components(y0, years, infl)
    try:
        frame = attach_japan_bills(frame, float(jp_mof_bills()["bills_bn"]))
    except Exception:
        pass
    _COMP[y0] = frame
    return frame


def _legacy(y0: int) -> pd.DataFrame:
    if y0 not in _LEGACY:
        from sovereign_cash.debt.isin_book import legacy_cash
        _LEGACY[y0] = legacy_cash(y0)
    return _LEGACY[y0]


def _gov_par(cc: str, start: float, tenors, shock_bp: float) -> np.ndarray | None:
    from sovereign_cash.debt.gov_curve import forward_par
    try:
        return np.asarray(forward_par(cc, start, tenors, shock_bp), float)
    except KeyError:
        return None


def project(hist: pd.DataFrame, fl: pd.DataFrame, ann: pd.DataFrame, snap: pd.DataFrame, sched: dict,
            y0: int, shock_bp: float = 0.0, recession: bool = False, src: pd.DataFrame | None = None,
            net_add: dict[str, dict[int, float]] | None = None,
            bill_share_override: dict[str, float] | None = None,
            real_shock_bp: float | None = None, ease: bool = False,
            bu_fiscal: pd.DataFrame | None = None) -> pd.DataFrame:
    """One set of assumptions for every country, years y0+1 .. Y_END (y0 = last outturn year).

    Net issuance: y0+1 from the monthly flows (outturn + cash framework); y0+2 the framework's next twelve
    months (calendar y0+2 if Q4 repeats); later years = the IMF WEO general-government deficit plus the gap
    between net issuance and that deficit (central vs general government, cash vs accrual, debt-management
    operations), reverting from y0+2's gap to its 2015–25 mean ex-Covid at the pooled estimated persistence.
    NO: share of GDP held. GDP: WEO nominal growth after y0. Revenue: y0 central-government
    outturn grown with cash bottom-up receipt categories when ``bu_fiscal`` has a shipped path for that
    country (UK/DE/FR/IT); else WEO general-government revenue. Rates: for US, UK, DE, JP, FR and IT the new-bond yield
    is the issuance-mix forward par off the government zero curve (Fed GSW, BoE, Bundesbank, MoF bootstrapped,
    TradingView cash bootstrapped) and the bill rate is the 3-month yield on that same curve; a shock is added
    to the zeros before the forward. UK linkers use the BoE real zero curve. SE and NO stay on the published
    par knots. Interest: the old coupon book is the ISIN tape outstanding at end-y0, scaled so y0 coupon
    interest plus bills equals the outturn (the scale carries linker uplift and anything the tape does not
    itemise); redemptions follow that book's maturity ladder, scaled onto the model's coupon stock. New bonds
    carry the forward par. Recession: the country's own
    2007→09 WEO balance swing added to net issuance in y0+2 and y0+3, its 2009 revenue fall in y0+2 and half
    of it in y0+3. ``ease`` is a parallel −100bp on the government curve, breakevens unchanged, on the
    baseline primary path.

    ``net_add[cc][year]`` (local bn) is extra net issuance that year — used to iterate a borrow-the-interest
    feedback without changing the baseline deficit path. ``bill_share_override[cc]`` replaces the snapshot
    bill share (JP sensitivity when the stock is OECD-implied)."""
    rows = []
    if real_shock_bp is None:
        real_shock_bp = shock_bp
    rho = gap_persistence(hist, ann, y0)
    comp = _marketable(y0, src)
    for cc in CC:
        h = hist[hist["cc"] == cc].set_index("year")
        a = ann[ann["country"] == cc].set_index("year")
        r = snap.set_index("cc").loc[cc]
        D0, Dm1 = h.at[y0, "debt_bn"], h.at[y0 - 1, "debt_bn"]
        if np.isfinite(h.at[y0 + 1, "debt_bn"]) and np.isfinite(h.at[y0 + 1, "net_issuance_bn"]):
            D0 = h.at[y0 + 1, "debt_bn"] - h.at[y0 + 1, "net_issuance_bn"]  # forward chains on the y0+1 level
        I0, R0 = h.at[y0, "interest_bn"], h.at[y0, "revenue_bn"]
        if not all(np.isfinite([D0, Dm1, I0, R0])):
            continue
        gdp = a["gdp_lcu_bn"]
        ggrev = a["gg_revenue_pct"] / 100 * gdp
        # Overall deficit % for issuance sizing. Default = −WEO/AMECO GG balance; overlay engine
        # GG primary + origin interest/GDP so medium-term net issuance tracks Π + book interest,
        # not a second authority overall path. Interest itself is still the ISIN loop below.
        dfc = (-a["gg_balance_pct"]).copy()
        try:
            from sovereign_cash.scenarios import primary as P

            p_eng = P.path(cc)
            i_pct0 = 100.0 * float(I0) / float(gdp[y0])
            for y in range(y0 + 1, Y_END + 1):
                if y in p_eng.index and np.isfinite(p_eng.at[y, "primary_gg_pct"]):
                    dfc.loc[y] = float(p_eng.at[y, "primary_gg_pct"]) + i_pct0
        except Exception:  # noqa: BLE001 — keep WEO overall if the engine path fails
            pass
        if bill_share_override is not None and cc in bill_share_override:
            bs = float(bill_share_override[cc])
        else:
            bs = r["bills_bn"] / r["marketable_bn"] if np.isfinite(r["bills_bn"]) else 0.0
        r3 = r["r3m"] / 100 if np.isfinite(r["r3m"]) else r["marginal_rate"] / 100
        r3_0 = a["irs"].get(y0, np.nan)
        r3_0 = r3_0 / 100 if np.isfinite(r3_0) else r3
        old_avg0 = (1 - bs) * (Dm1 + D0) / 2
        c_old = max((I0 - r3_0 * bs * (Dm1 + D0) / 2) / old_avg0, 0.0) if old_avg0 > 0 else 0.0
        leg_c = comp[comp["cc"] == cc].set_index("year") if cc in GOV_CURVE and comp is not None else None
        # Non-marketable residual: outturn minus the ISIN bill at the y0 short rate. A rate shock
        # does not move it, and it is not spread back onto the coupons.
        bp = shock_bp
        bp_real = real_shock_bp
        if ease:
            bp = bp_real = -100.0
        # par-knot countries (no fitted zero curve): the shift at each issuance tenor and at 3m
        from sovereign_cash.debt.gov_curve import shift_at
        sh_t = shift_at(bp, TENORS) / 1e4
        sh = float(shift_at(bp, 0.25)[()]) / 1e4
        rsh = float(shift_at(bp_real, 10.0)[()]) / 1e4
        book_uplift = True
        bill_locked = r3_0
        if leg_c is not None and y0 in leg_c.index:
            z = leg_c.loc[y0]
            frn0 = float(z["frn_face_bn"]) * (r3_0 + float(z["frn_spread"]))
            # Japan publishes no bill maturity, so the bill rate is whatever reconciles the outturn
            # to the coupon book. If linker uplift does not fit inside that outturn, it is not in
            # the interest line (the debt stock still accrues it).
            if int(z["bill_rows"]) == 0 and float(z["bill_face_bn"]) > 0:
                room = I0 - float(z["coupon_bn"]) - frn0
                implied = (room - float(z["uplift_bn"])) / float(z["bill_face_bn"])
                if implied < 0:
                    book_uplift = False
                    bill_locked = max(room / float(z["bill_face_bn"]), 0.0)
                else:
                    bill_locked = implied
            uplift0 = float(z["uplift_bn"]) if book_uplift else 0.0
            mkt0 = float(z["coupon_bn"]) + uplift0 + frn0 + float(z["bill_face_bn"]) * bill_locked
            # UK: accrual uplift can exceed cash JW2P (end-2021). Clip residual at 0
            # so a timing gap does not subtract from next year (isin_interest_bugfix.md).
            # Other countries: keep signed residual (marketable overstatement is real).
            residual = max(I0 - mkt0, 0.0) if cc == "uk" else (I0 - mkt0)
            bk = _book(y0)
            face0 = float(bk[(bk["country"] == cc.upper()) & (bk["sec_type"] != "Bill")]["face_mn"].sum()) / 1000.0
            face_scale = (1 - bs) * D0 / face0 if face0 > 0 else np.nan
        else:
            residual, face_scale = 0.0, np.nan

        net = {y0 + 1: h.at[y0 + 1, "net_issuance_bn"], y0 + 2: float(r["net_issuance_fwd"])}
        hy = [y for y in range(FIRST_YEAR, y0 + 1) if y not in COVID and np.isfinite(h.at[y, "net_issuance_bn"])]
        gap_hist = float(np.mean([100 * h.at[y, "net_issuance_bn"] / gdp[y] - dfc[y] for y in hy]))
        gap2 = 100 * net[y0 + 2] / gdp[y0 + 2] - dfc[y0 + 2]
        for y in range(y0 + 3, Y_END + 1):
            if cc in DEFICIT_FREE:
                net[y] = net[y0 + 2] * gdp[y] / gdp[y0 + 2]
                continue
            net[y] = (dfc[y] + gap_hist + rho ** (y - y0 - 2) * (gap2 - gap_hist)) / 100 * gdp[y]
        # US years the DTS path covers in full replace the WEO deficit. The partial current year
        # stays on the monthly flows already in y0+1. Rate shocks keep this primary path.
        dts = us_dts_years() if cc == "us" else pd.DataFrame()
        if len(dts):
            for y in range(y0 + 2, Y_END + 1):
                if y in dts.index:
                    net[y] = float(dts.at[y, "net_issuance_bn"])
        # Revenue growth: cash bottom-up receipt categories when shipped (UK/DE/FR/IT);
        # else WEO general-government revenue share path. Interest stays on the ISIN book;
        # cash DI / AFT charge / BMF carve sit in interest_cash_bu_bn only.
        rev = {y: R0 * ggrev[y] / ggrev[y0] for y in range(y0 + 1, Y_END + 1)}
        rev_src = "weo_gg"
        bu_cash_int: dict[int, float] = {}
        if bu_fiscal is not None and cc in ("uk", "de", "fr", "it"):
            from sovereign_cash.debt.bu_fiscal import interest_cash_path, receipts_index
            idx = receipts_index(bu_fiscal, cc, y0, range(y0 + 1, Y_END + 1))
            if idx is not None and any(np.isfinite(v) for v in idx.values()):
                for y, m in idx.items():
                    if np.isfinite(m):
                        rev[y] = R0 * m
                rev_src = "cash_bu"
            bu_cash_int = interest_cash_path(bu_fiscal, cc, range(y0 + 1, Y_END + 1))
        if recession:
            swing = float(a.loc[2007, "gg_balance_pct"] - a.loc[2008:2010, "gg_balance_pct"].min())
            fall = min(float(ggrev[2009] / ggrev[2008] - 1), 0.0) if np.isfinite(ggrev.get(2009, np.nan)) else -0.05
            for y, k in ((y0 + 2, 1.0), (y0 + 3, 0.5)):
                net[y] += swing / 100 * gdp[y]
                rev[y] *= 1 + k * fall

        # Bonds: the old book keeps its implied coupon and runs off on today's lines; new bonds are issued at the
        # market forward for their year and tenor, carry their coupon to maturity, and roll inside the horizon
        # by the issuance tenor mix. Bills: maturity ladder at each bill's issue tenor (4w–52w), else forward 3m.
        # index-linked uplift accrues on the linker share of debt (incl. uplift) at the latest inflation
        up = _uplift(src, cc) if src is not None else {"il_now": 0.0, "infl": lambda y: 0.0}
        il_share = up["il_now"] / r["marketable_bn"] if r["marketable_bn"] > 0 else 0.0
        red = sched.get(cc, pd.Series(dtype=float))
        mix = tenor_mix(cc, sched)
        cv = curve(cc, r["r3m"], r["y10"])
        w_linker = linker_issue_share(cc)  # bottom-up 12m face share of Linker auctions
        w_frn = min(floater_issue_share(cc), max(1.0 - w_linker, 0.0))
        y_linker0 = linker_marginal_yield(cc, float(r["y10"])) / 100.0
        today = pd.Timestamp.today()
        t_now = today.year + (today.dayofyear - 0.5) / 365.25
        # the current year: realised auction yields and 3m rates for the part already done, forwards for the rest
        done = today.dayofyear / 365.25
        yb_ytd, r3_ytd = realised_ytd(cc, today.year)
        D, old = D0, (1 - bs) * D0
        new, cost = 0.0, 0.0
        vint = {}  # issue year -> gross new bonds
        add = (net_add or {}).get(cc, {})
        for y in range(y0 + 1, Y_END + 1):
            start = max(y + 0.5 - t_now, 0.0)
            gov = _gov_par(cc, start, TENORS, bp) if cc in GOV_CURVE else None
            if gov is not None:
                yb_n = float(mix @ gov) / 100.0
                yr3 = float(_gov_par(cc, start, 0.25, bp)[0]) / 100.0
            else:
                yb_n = float(mix @ (forward(cv, start, TENORS) / 100 + sh_t))
                yr3 = float(forward(cv, start, 0.25)) / 100 + sh
            # Real shock is its own input. The default is the same number of basis points as the
            # nominal shock, which holds breakevens unchanged. Pass 0 to put the whole move in breakevens.
            real = _gov_par(f"{cc}_real", start, TENORS, bp_real) if cc in GOV_CURVE else None
            yb_l = float(mix @ real) / 100.0 if real is not None else y_linker0 + rsh
            yb_f = yr3 + (float(leg_c.at[y0, "frn_spread"]) if leg_c is not None and y0 in leg_c.index else 0.0)
            yb = (1.0 - w_linker - w_frn) * yb_n + w_linker * yb_l + w_frn * yb_f
            if y == today.year:
                if np.isfinite(yb_ytd):
                    # realised YTD is nominal+FRN only; keep linker share on the live real yield
                    yb = ((1.0 - w_linker - w_frn) * (done * yb_ytd / 100 + (1 - done) * yb_n)
                          + w_linker * yb_l + w_frn * yb_f)
                if np.isfinite(r3_ytd):
                    yr3 = done * r3_ytd / 100 + (1 - done) * yr3
            if leg_c is not None and y in leg_c.index and np.isfinite(face_scale):
                mat = float(leg_c.at[y, "redeem_bn"]) * face_scale
            elif y == y0 + 1 and np.isfinite(h.at[y, "rollovers_bn"]):
                mat = h.at[y, "rollovers_bn"] - bs * D
            else:
                mat = float(red.get(y, 0.0))
            m_new = sum(g * mix[y - s - 1] for s, g in vint.items() if 0 < y - s <= len(TENORS))
            accrual = il_share * D * up["infl"](y) if y > y0 + 1 else 0.0  # y0+1 debt already carries its accrual
            borrowed = float(add.get(y, 0.0))
            Dn = D + net[y] + accrual + borrowed
            old_n = max(old - mat, 0.0)
            new_n = max((1 - bs) * Dn - old_n, 0.0)
            gross = max(new_n - new + m_new, 0.0)
            cost_n = cost * (1 - m_new / new if new > 0 else 0.0) + gross * yb
            new_int = (cost + cost_n) / 2
            model_bills = bs * (D + Dn) / 2
            if leg_c is not None and y in leg_c.index:
                z = leg_c.loc[y]
                from sovereign_cash.debt.cashflow import bill_ladder
                bills = _book(y0)
                bills = bills[(bills["country"] == cc.upper()) & (bills["sec_type"] == "Bill")]

                def _rate(start: float, tenor: float, _cc=cc, _bp=bp) -> float:
                    got = _gov_par(_cc, start, tenor, _bp)
                    return float(got[0]) / 100.0 if got is not None else yr3

                tape_face = float(bills["face_mn"].sum()) / 1000.0 if len(bills) else 0.0
                if int(z["bill_rows"]) > 0 and tape_face > 0:
                    # Ladder on tape face; scale to model bills. MoF/stock face may exceed tape (JP).
                    tape_int = bill_ladder(bills, y, y0, _rate)
                    bill_int = tape_int * (model_bills / tape_face)
                elif y == y0 + 1:
                    # No maturity dates: the stock rolls uniformly through the first forecast year.
                    tape_int = float(z["bill_face_bn"]) * (0.5 * bill_locked + 0.5 * yr3)
                    bill_int = tape_int * (model_bills / float(z["bill_face_bn"])) if float(z["bill_face_bn"]) > 0 else model_bills * yr3
                else:
                    tape_int = float(z["bill_face_bn"]) * yr3
                    bill_int = tape_int * (model_bills / float(z["bill_face_bn"])) if float(z["bill_face_bn"]) > 0 else model_bills * yr3
                frn_int = float(z["frn_face_bn"]) * (yr3 + float(z["frn_spread"]))
                uplift_i = float(z["uplift_bn"]) if book_uplift else 0.0
                infl = float(up["infl"](y)) if book_uplift else 0.0
                new_lk = w_linker * (new + new_n) / 2 * infl
                old_int = float(z["coupon_bn"]) + uplift_i + new_lk + frn_int + residual
                denom = yb if yb else 1.0
                w_fix = (1.0 - w_linker - w_frn) * yb_n / denom
                w_l = w_linker * yb_l / denom
                w_f = w_frn * yb_f / denom
                ch_locked = float(z["coupon_bn"])
                ch_new = new_int * w_fix
                ch_linker = uplift_i + new_lk + new_int * w_l
                ch_floater = frn_int + new_int * w_f
                ch_other = residual
            else:
                bill_int = yr3 * model_bills
                old_int = c_old * (old + old_n) / 2
                ch_locked, ch_new, ch_linker, ch_floater, ch_other = old_int, new_int, 0.0, 0.0, 0.0
            interest = old_int + new_int + bill_int
            roll_bills = bs * D
            if len(dts) and y in dts.index and np.isfinite(dts.at[y, "bills_bn"]):
                # TGA cash interest is not the federal interest outlay (trust-fund credits never
                # hit the DTS line), so only the bill stock and the net issuance come across.
                roll_bills = float(dts.at[y, "bills_bn"])
            elif (cc == "us" and len(dts) and y > int(dts.index.max())
                  and np.isfinite(dts["bills_bn"].iloc[-1])):
                # Bridge ends mid-horizon (complete years through ~y0+3). Falling back to the
                # snap-time bill share after a DTS stock run-up invents a downward kink in
                # debt-service / revenue (2029→2030). Hold the last DTS bill/debt ratio.
                y_last = int(dts.index.max())
                # D here is start-of-year debt; the last DTS year row's debt is prior end =
                # this loop's D when y == y_last+1, else scale from stored row.
                d_last = None
                for prev in reversed(rows):
                    if prev["cc"] == cc and prev["year"] == y_last:
                        d_last = float(prev["debt_bn"])
                        break
                if d_last is not None and d_last > 0:
                    roll_bills = float(dts.at[y_last, "bills_bn"]) / d_last * D
            rows.append({"cc": cc, "year": y, "debt_bn": Dn, "net_issuance_bn": net[y] + borrowed,
                         "borrowed_interest_bn": borrowed,
                         "rollovers_bn": roll_bills + mat + m_new, "rollovers_new_bn": m_new, "rollovers_bills_bn": roll_bills, "il_accrual_bn": accrual,
                         "interest_bn": interest, "revenue_bn": rev[y], "gdp_bn": gdp[y],
                         "revenue_source": rev_src,
                         "interest_cash_bu_bn": float(bu_cash_int[y]) if y in bu_cash_int else np.nan,
                         "ch_locked": ch_locked, "ch_new": ch_new, "ch_bills": bill_int,
                         "ch_linker": ch_linker, "ch_floater": ch_floater, "ch_other": ch_other,
                         "new_bond_yield": 100 * yb, "bill_rate": 100 * yr3,
                         "repriced_share": (new_n + bs * Dn) / Dn})
            vint[y] = gross
            D, old, new, cost = Dn, old_n, new_n, cost_n
    out = pd.DataFrame(rows)
    out["scenario"] = "recession" if recession or ease else "baseline"
    if ease:
        out["case"] = "ease"
    elif shock_bp and real_shock_bp == 0:
        out["case"] = "shock_nominal"
    elif shock_bp:
        out["case"] = "shock"
    else:
        out["case"] = "current"
    return _ratios(out)


def project_borrowed(hist: pd.DataFrame, fl: pd.DataFrame, ann: pd.DataFrame, snap: pd.DataFrame, sched: dict,
                     y0: int, shock_bp: float = 100.0, src: pd.DataFrame | None = None,
                     bill_share_override: dict[str, float] | None = None,
                     max_iter: int = 12, tol_bn: float = 1e-3,
                     real_shock_bp: float | None = None,
                     bu_fiscal: pd.DataFrame | None = None) -> pd.DataFrame:
    """+``shock_bp`` with the incremental interest bill borrowed each year (fixed primary path).

    Iterate: net_add[y] = I_shock[y] − I_base[y] until the interest path is stable. Primary deficits are
    unchanged; only the extra coupon is debt-financed. No yield→supply feedback (that is ``dalio_note.spiral``)."""
    if real_shock_bp is None:
        real_shock_bp = shock_bp
    base = project(hist, fl, ann, snap, sched, y0, 0.0, False, src, bill_share_override=bill_share_override,
                   real_shock_bp=0.0, bu_fiscal=bu_fiscal)
    bI = base.set_index(["cc", "year"])["interest_bn"]
    net_add: dict[str, dict[int, float]] = {}
    out = base
    for _ in range(max_iter):
        out = project(hist, fl, ann, snap, sched, y0, shock_bp, False, src, net_add=net_add,
                      bill_share_override=bill_share_override, real_shock_bp=real_shock_bp,
                      bu_fiscal=bu_fiscal)
        nxt: dict[str, dict[int, float]] = {}
        delta = 0.0
        for (cc, y), I in out.set_index(["cc", "year"])["interest_bn"].items():
            extra = float(I - bI.get((cc, y), I))
            nxt.setdefault(cc, {})[int(y)] = extra
            prev = net_add.get(cc, {}).get(int(y), 0.0)
            delta = max(delta, abs(extra - prev))
        net_add = nxt
        if delta < tol_bn:
            break
    out = out.copy()
    out["case"] = "shock_nominal_borrowed" if real_shock_bp == 0 else "shock_borrowed"
    return out


def _ratios(out: pd.DataFrame, gdp: pd.Series | None = None) -> pd.DataFrame:
    out["gross_issuance_bn"] = out["net_issuance_bn"] + out["rollovers_bn"]
    out["interest_pct_rev"] = 100 * out["interest_bn"] / out["revenue_bn"]
    out["rollovers_pct_rev"] = 100 * out["rollovers_bn"] / out["revenue_bn"]
    out["debt_service_pct_rev"] = out["interest_pct_rev"] + out["rollovers_pct_rev"]
    if "rollovers_bills_bn" in out:
        out["debt_service_ex_bills_pct_rev"] = 100 * (out["interest_bn"] + out["rollovers_bn"]
                                                      - out["rollovers_bills_bn"]) / out["revenue_bn"]
    if "gg_revenue_bn" in out:
        out["gg_interest_pct_rev"] = 100 * out["gg_interest_bn"] / out["gg_revenue_bn"]
        out["gg_interest_pct_gdp"] = 100 * out["gg_interest_bn"] / out["gdp_bn"]
        out["gg_debt_service_pct_rev"] = 100 * (out["gg_interest_bn"] + out["rollovers_bn"]) / out["gg_revenue_bn"]
        out["gg_debt_service_ex_bills_pct_rev"] = 100 * (out["gg_interest_bn"] + out["rollovers_bn"]
                                                         - out["rollovers_bills_bn"]) / out["gg_revenue_bn"]
    g = out["gdp_bn"] if "gdp_bn" in out else gdp
    for c in ("debt_bn", "net_issuance_bn", "rollovers_bn", "gross_issuance_bn", "interest_bn"):
        out[c.replace("_bn", "_pct_gdp")] = 100 * out[c] / g
    return out


def attach_engine_primary(cya: pd.DataFrame) -> pd.DataFrame:
    """Stamp the share-engine GG primary onto the debt board (Ex1 / sustain / residual_pb).

    ``primary_gg_pct_gdp`` is a *deficit* (+ = deficit), matching ``scenarios.primary``.
    Forecast-year ``revenue_source`` is labelled ``primary_engine`` once the engine row exists —
    the interest / debt path still uses the ISIN book; the primary balance for the note identity
    is this column, not WEO interest − deficit.
    """
    from sovereign_cash.scenarios import primary as P

    out = cya.copy()
    out["primary_gg_pct_gdp"] = np.nan
    out["primary_rule"] = pd.Series(pd.NA, index=out.index, dtype="string")
    # Always regenerate: a stale primary_path.json previously suppressed path() and published yesterday's handoff.
    P.write(P.build())
    j = json.loads(P.PATH.read_text())
    for cc, g in out.groupby("cc"):
        c = (j.get("countries") or {}).get(cc)
        if not c:
            try:
                p = P.path(cc)
            except Exception:
                continue
            years = {str(y): {"primary_gg_pct": float(p.at[y, "primary_gg_pct"]),
                              "rule": str(p.at[y, "rule"])} for y in p.index}
            methods = p.attrs.get("methods") or {}
        else:
            years = c.get("years") or {}
            methods = c.get("methods") or {}
        rule = "+".join(f"{k}:{v}" for k, v in sorted(methods.items())) if methods else "engine"
        for i, y in zip(g.index, g["year"]):
            row = years.get(str(int(y)))
            if not row or row.get("primary_gg_pct") is None:
                continue
            out.at[i, "primary_gg_pct_gdp"] = float(row["primary_gg_pct"])
            out.at[i, "primary_rule"] = str(row.get("rule") or rule)
            if int(y) >= 2027 and pd.notna(out.at[i, "primary_gg_pct_gdp"]):
                out.at[i, "revenue_source"] = "primary_engine"
    return out


def combine(hist: pd.DataFrame, base: pd.DataFrame, ann: pd.DataFrame, y0: int) -> pd.DataFrame:
    """Outturns to y0; y0+1 keeps its flows-based debt, net issuance and rollovers with projected interest and
    revenue; later years are the projection. The authority's own interest / revenue path is kept alongside."""
    out = hist[hist["year"] <= y0 + 1].copy()
    out["gdp_bn"] = [float(ann.set_index(["country", "year"])["gdp_lcu_bn"].get((c, y), np.nan))
                     for c, y in zip(out["cc"], out["year"])]
    out = out.set_index(["cc", "year"])
    b = base.set_index(["cc", "year"])
    i1 = b.index[b.index.get_level_values(1) == y0 + 1].intersection(out.index)
    out.loc[i1, ["interest_bn", "revenue_bn"]] = b.loc[i1, ["interest_bn", "revenue_bn"]].values
    if "interest_cash_bu_bn" in b.columns:
        out["interest_cash_bu_bn"] = np.nan
        out.loc[i1, "interest_cash_bu_bn"] = b.loc[i1, "interest_cash_bu_bn"].astype(float).values
    if "revenue_source" in b.columns:
        out["revenue_source"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[i1, "revenue_source"] = b.loc[i1, "revenue_source"].astype("string").values
    carry = ["debt_bn", "net_issuance_bn", "rollovers_bn", "rollovers_new_bn",
             "rollovers_bills_bn", "il_accrual_bn", "interest_bn", "revenue_bn", "gdp_bn",
             "new_bond_yield", "bill_rate"]
    for extra in ("revenue_source", "interest_cash_bu_bn"):
        if extra in b.columns:
            carry.append(extra)
    later = b[b.index.get_level_values(1) > y0 + 1][carry].assign(forecast=True)
    out = pd.concat([out, later]).reset_index().sort_values(["cc", "year"]).reset_index(drop=True)
    a = ann.set_index(["country", "year"])
    out["interest_authority_bn"] = [a["cg_interest_bn"].get((c, y), np.nan) for c, y in zip(out["cc"], out["year"])]
    out["revenue_authority_bn"] = [a["cg_revenue_bn"].get((c, y), np.nan) for c, y in zip(out["cc"], out["year"])]
    out["deficit_weo_pct_gdp"] = [-a["gg_balance_pct"].get((c, y), np.nan) for c, y in zip(out["cc"], out["year"])]
    out["deficit_authority_pct_gdp"] = [100 * a["deficit_official_bn"].get((c, y), np.nan) / a["gdp_lcu_bn"].get((c, y), np.nan)
                                        for c, y in zip(out["cc"], out["year"])]
    # the same projection on the authority's deficit path: net issuance moves by the change in the
    # authority-minus-WEO deficit gap since y0+2 (the first year the WEO rule takes over)
    out["net_issuance_auth_bn"] = np.nan
    out["debt_auth_bn"] = np.nan
    for cc, g in out.groupby("cc"):
        g = g.set_index("year")
        wedge = g["deficit_authority_pct_gdp"] - g["deficit_weo_pct_gdp"]
        if cc in DEFICIT_FREE or not np.isfinite(wedge.get(y0 + 2, np.nan)):
            continue
        adj = ((wedge - wedge[y0 + 2]) / 100 * g["gdp_bn"]).where(g.index > y0 + 2, 0.0)
        adj = adj[g.index >= y0 + 2]
        cover = adj.notna().cumprod().astype(bool)
        adj = adj[cover]
        idx = out.index[(out["cc"] == cc) & out["year"].isin(adj.index)]
        out.loc[idx, "net_issuance_auth_bn"] = (g.loc[adj.index, "net_issuance_bn"] + adj).values
        out.loc[idx, "debt_auth_bn"] = (g.loc[adj.index, "debt_bn"] + adj.cumsum()).values
    out["net_issuance_auth_pct_gdp"] = 100 * out["net_issuance_auth_bn"] / out["gdp_bn"]
    out["debt_auth_pct_gdp"] = 100 * out["debt_auth_bn"] / out["gdp_bn"]
    # General government, comparable across countries. Central revenue is a third of the total in
    # FR / DE / US / JP (social security, Länder, states) and nearly all of it in the UK.
    # After y0, GG interest is the central bill plus the y0 gap (OECD gross GG minus central net
    # for the US and the UK; Eurostat central is already gross). The gap mixes interest the
    # government receives with the state/local perimeter. Neither piece has a maturity tape.
    # Japan: BoJ ISIN coupons as ``cb_coupon_bn`` memo only (not subtracted from GG: GFSM puts BoJ
    # outside GG; remittance is property income, not negative interest).
    # GG revenue = AMECO share of GDP when present, else WEO (see annual()). Same perimeter as
    # gross GG interest so interest/rev is Commission-comparable.
    from sovereign_cash.debt.cashflow import CB_NET_CC, cb_coupon_path

    out["gg_revenue_bn"] = [a["gg_revenue_pct"].get((c, y), np.nan) / 100 * g for c, y, g in
                            zip(out["cc"], out["year"], out["gdp_bn"])]
    out["gg_interest_bn"] = np.nan
    out["cb_coupon_bn"] = np.nan
    for cc, g in out.groupby("cc"):
        gi = a["gg_interest_bn"].xs(cc, level=0).dropna()
        g = g.set_index("year")
        if not np.isfinite(gi.get(y0, np.nan)) or not np.isfinite(g["interest_bn"].get(y0, np.nan)):
            continue
        path = gi[gi.index <= y0].reindex(g.index)
        fc = g.index > y0
        # States, social security and agencies do not roll on the sovereign curve.
        # Their y0 residual stays put; only the marketable central bill moves.
        path[fc] = g.loc[fc, "interest_bn"] + (gi[y0] - g.at[y0, "interest_bn"])
        if cc in CB_NET_CC:
            cb = cb_coupon_path(cc, y0, [int(y) for y in g.index if int(y) > y0])
            if len(cb):
                # Memo only — do not subtract. GFSM: BoJ outside GG; remittance = property income.
                m = (out["cc"] == cc) & (out["year"] > y0)
                out.loc[m, "cb_coupon_bn"] = out.loc[m, "year"].map(cb).to_numpy(float)
        out.loc[out["cc"] == cc, "gg_interest_bn"] = path.to_numpy(float)
    return _ratios(out)


# ---------------------------------------------------------------- run
@dataclass
class Inputs:
    """Everything ``project`` needs, loaded once. ``prepare_inputs`` writes nothing."""

    src: pd.DataFrame
    ann: pd.DataFrame
    book: pd.DataFrame
    fl: pd.DataFrame
    sched: dict
    snap: pd.DataFrame
    rein: pd.DataFrame
    hist: pd.DataFrame
    asof: pd.Timestamp
    y0: int
    ca_hist: pd.DataFrame
    bu_tab: pd.DataFrame
    bu_meta: dict

    def project(self, shock_bp: float = 0.0, **kw) -> pd.DataFrame:
        """``project`` on these inputs; ``kw`` as ``project`` (recession, ease, real_shock_bp, ...)."""
        return project(self.ca_hist, self.fl, self.ann, self.snap, self.sched, self.y0, shock_bp,
                       kw.pop("recession", False), self.src, bu_fiscal=self.bu_tab, **kw)


def prepare_inputs(refresh: bool = False) -> Inputs:
    from issuance import imf_investor_base as ib

    from sovereign_cash.debt import gov_curve as _curves
    if refresh or (_curves.stale() and __import__("os").environ.get("SOVDEBT_PULL") == "1"):
        _curves.build()
    src = S.load(refresh)
    ann = annual(src)
    book = _jp_bills(monthly_book(), src)
    holders = ib.panel()
    fl = flows_monthly(book)
    sched = redemption_schedule(src)
    snap = snapshot(book, fl, ann, src, holders, sched)
    rein = reinvestment(snap)
    hist = history(book, fl, ann)
    asof = pd.Timestamp(fl.loc[~fl["is_forecast"].astype(bool) & (fl["cc"] == "uk"), "month"].max())
    y0 = Y0
    ca_hist = country_annual(book, fl, ann, y0 + 1, asof, snap, _bills_dec(src), src)
    from sovereign_cash.debt import bu_fiscal as _buf
    bu_tab, bu_meta = _buf.build(y0=y0)
    return Inputs(src, ann, book, fl, sched, snap, rein, hist, asof, y0, ca_hist, bu_tab, bu_meta)


def run(refresh: bool = False) -> dict:
    from sovereign_cash.debt import bu_fiscal as _buf

    inp = prepare_inputs(refresh)
    src, ann, book, fl, sched = inp.src, inp.ann, inp.book, inp.fl, inp.sched
    snap, rein, hist, asof, y0, ca_hist = inp.snap, inp.rein, inp.hist, inp.asof, inp.y0, inp.ca_hist
    _buf.write(inp.bu_tab, inp.bu_meta)
    bu_tab, bu_meta = inp.bu_tab, inp.bu_meta
    kw = dict(bu_fiscal=bu_tab)
    runs = [project(ca_hist, fl, ann, snap, sched, y0, sb, rc, src, **kw)
            for rc in (False, True) for sb in (0.0, 100.0)]
    runs.append(project(ca_hist, fl, ann, snap, sched, y0, 100.0, False, src, real_shock_bp=0.0, **kw))
    runs.append(project_borrowed(ca_hist, fl, ann, snap, sched, y0, 100.0, src, **kw))
    runs.append(project_borrowed(ca_hist, fl, ann, snap, sched, y0, 100.0, src, real_shock_bp=0.0, **kw))
    runs.append(project(ca_hist, fl, ann, snap, sched, y0, 0.0, False, src, ease=True, **kw))
    rep = pd.concat(runs, ignore_index=True)
    rep = rep[rep["year"] > y0].reset_index(drop=True)
    cya = combine(ca_hist, runs[0], ann, y0)
    cya = attach_engine_primary(cya)
    # snapshot ratios on the same basis as the table's current year
    cur = cya[cya["year"] == y0 + 1].set_index("cc")
    snap = snap.set_index("cc")
    snap["interest_bn"], snap["revenue_bn"] = cur["interest_bn"], cur["revenue_bn"]
    snap["interest_pct_rev"] = 100 * snap["interest_bn"] / snap["revenue_bn"]
    snap["rollovers_pct_rev"] = 100 * snap["rollovers_bn"] / snap["revenue_bn"]
    snap["debt_service_pct_rev"] = snap["interest_pct_rev"] + snap["rollovers_pct_rev"]
    snap["eff_rate"] = 100 * snap["interest_bn"] / snap["marketable_bn"]
    snap["gap_persistence"] = gap_persistence(ca_hist, ann, y0)
    snap = snap.reset_index()
    pop = ann[ann["year"] == asof.year].set_index("country")["pop_mn"]
    debt_usd = snap["marketable_bn"].fillna(snap["gg_debt_pct"] / 100 * snap["gdp_bn"]) * snap["fx"]
    snap["debt_per_household_usd"] = 1e3 * debt_usd / (snap["cc"].map(pop) / snap["cc"].map(HH_SIZE))
    DERIVED.mkdir(exist_ok=True)
    cya.to_parquet(DERIVED / "debt_country_annual.parquet", index=False)
    ann.to_parquet(DERIVED / "debt_annual.parquet", index=False)
    snap.to_parquet(DERIVED / "debt_snapshot.parquet", index=False)
    rein.to_parquet(DERIVED / "debt_reinvestment.parquet", index=False)
    rep.to_parquet(DERIVED / "debt_repricing.parquet", index=False)
    hist.to_parquet(DERIVED / "debt_history.parquet", index=False)
    fl.to_parquet(DERIVED / "debt_flows_monthly.parquet", index=False)
    vint = src.groupby("source")["vintage"].last().to_dict()
    summary = {"asof": pd.Timestamp.today().strftime("%Y-%m-%d"), "vintages": vint,
               "bu_fiscal": bu_meta}
    try:
        from sovereign_cash.debt.gov_curve import load as _gov
        curves = _gov()
        summary["gov_curve"] = {
            str(r.cc): {"asof": pd.Timestamp(r.asof).strftime("%Y-%m-%d"), "source": str(r.source)}
            for r in curves.itertuples()
        }
    except Exception:
        pass
    (DERIVED / "debt_summary.json").write_text(json.dumps(summary, indent=1))
    return {"src": src, "annual": ann, "book": book, "snapshot": snap, "reinvestment": rein, "repricing": rep,
            "history": hist, "flows": fl, "summary": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    out = run(ap.parse_args().refresh)
    s = out["snapshot"]
    pd.set_option("display.width", 250)
    cols = ["cc", "marketable_bn", "bills_bn", "rollovers_bn", "net_issuance_fwd", "gross_issuance_fwd",
            "bill_share_fwd", "bill_share_norm", "cb_share", "interest_pct_rev", "rollovers_pct_rev",
            "debt_service_pct_rev", "eff_rate", "marginal_rate", "y10"]
    print(s[cols].round(2).to_string())
