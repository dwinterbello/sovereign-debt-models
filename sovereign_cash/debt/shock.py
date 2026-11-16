"""+100bp selloff channels from the government zero curve.

Interest: ``build.project`` / ``project_borrowed``. Locked ISIN coupons stay put. The shock is
added to the zeros before the forward par. ``real_shock_bp`` is separate.

Holder MTM: each live ISIN discounted on today's government zeros, then on the curve shifted
100bp. Linkers use the real curve and the issuer's index coefficient (UK uplift rebuild,
US and France indexed stock, Finanzagentur, MEF, MoF). Private holdings are SOMA by ISIN,
the APF by ISIN, the Bank of Japan by issue, and the ECB country stock placed on ISINs.
Japan's bill stock has no ISINs. The BoJ outright T-bill file is zero, so that stock is private.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sovereign_cash.debt import build as B
from sovereign_cash.debt import sources as S

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = REPO / "issuance" / "cache"
DERIVED = HERE.parent / "derived"
CC_MAP = {"us": "US", "uk": "UK", "de": "DE", "fr": "FR", "it": "IT", "jp": "JP"}


def _ops() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(CACHE.glob("ops_*.parquet"))]
    ops = pd.concat(frames, ignore_index=True)
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["maturity_dt"] = pd.to_datetime(ops["maturity_dt"])
    ops["isin"] = ops["isin"].astype(str)
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    return ops


def live_stock(ops: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Outstanding face by country × ISIN × sec_type at ``asof`` (sum of taps still alive)."""
    asof = asof or ops["op_dt"].max()
    live = ops[(ops["op_dt"] <= asof) & (ops["maturity_dt"] > asof) & (ops["amount_issued"] > 0)]
    g = live.groupby(["country", "isin", "sec_type"], as_index=False).agg(
        face_mn=("amount_issued", "sum"),
        maturity_dt=("maturity_dt", "last"),
        coupon=("coupon", "last"),
    )
    g["resid"] = (g["maturity_dt"] - asof).dt.days / 365.25
    g["asof"] = asof
    return g


def stock_dv01(stock: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    """USD mn per bp by country × sec_type from live ISIN face."""
    from issuance.float_rv import CCY, _fx_on
    from issuance.pricing import curve_yield, dv01_mn, market_curve

    asof = pd.Timestamp(stock["asof"].iloc[0])
    curve = market_curve(ops, end=asof)
    rows = []
    for (cc, st), g in stock.groupby(["country", "sec_type"]):
        resid = np.clip(g["resid"].to_numpy(dtype=float), 0.02, 50.0)
        linker = np.full(len(g), st == "Linker")
        yy = curve_yield(curve, cc, linker, asof, resid)
        if st == "Bill":
            cp = np.zeros(len(g))
            yy = np.where(np.isfinite(yy), yy, 3.0)
        else:
            # ops coupon is percent; AFT sometimes writes junk (e.g. 20361) — same guard as float_rv
            cp = g["coupon"].to_numpy(dtype=float)
            yy = np.where(np.isfinite(yy) & (yy >= -2) & (yy <= 25), yy, 3.0)
            cp = np.where(np.isfinite(cp) & (cp >= 0) & (cp <= 20), cp, yy)
        fx = _fx_on(pd.Series([asof] * len(g)), pd.Series([CCY[cc]] * len(g)))
        dv = dv01_mn(g["face_mn"].to_numpy(dtype=float), resid, cp, yy, fx)
        rows.append({
            "cc": cc.lower(), "sec_type": st,
            "face_bn": float(g["face_mn"].sum()) / 1000.0,
            "dv01_usd_mn": float(dv.sum()),
            "wam": float(np.average(resid, weights=g["face_mn"].to_numpy())),
            "source": "ops ISIN stock",
        })
    return pd.DataFrame(rows)


def _published_bills() -> pd.DataFrame:
    """Bill stock (local bn) from the bills tape; JP from MoF short-term JGBs + Financing Bills."""
    rows = []
    b = pd.read_parquet(CACHE / "bill_supply_monthly.parquet")
    b["month"] = pd.to_datetime(b["month"])
    for CC in ("US", "UK", "DE", "FR", "IT"):
        h = b[(b["country"] == CC) & (b["phase"] == "history")].sort_values("month").iloc[-1]
        rows.append({"cc": CC.lower(), "bills_bn": float(h["bill_stock_bn"]),
                     "month": h["month"], "source": "bill_supply_monthly"})
    jp = B.jp_mof_bills()
    rows.append({"cc": "jp", "bills_bn": jp["bills_bn"], "month": jp["asof"], "source": jp["source"]})
    return pd.DataFrame(rows)


def _bill_wam_from_ops(dv: pd.DataFrame) -> float:
    w = dv[dv["sec_type"] == "Bill"]
    if w.empty or w["face_bn"].sum() <= 0:
        return 0.25
    return float(np.average(w["wam"], weights=w["face_bn"]))


def fill_missing_bill_dv01(dv: pd.DataFrame, ops: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Add Bill DV01 for countries with no live bill ISINs in ops (UK, JP) from published stock × WAM."""
    from issuance.float_rv import CCY

    pub = _published_bills().set_index("cc")
    wam_ref = _bill_wam_from_ops(dv)
    snap = snap.set_index("cc")
    extra = []
    for cc in ("uk", "jp"):
        if ((dv["cc"] == cc) & (dv["sec_type"] == "Bill")).any():
            continue
        if cc not in pub.index:
            continue
        face_bn = float(pub.loc[cc, "bills_bn"])
        fx = float(snap.loc[cc, "fx"]) if cc in snap.index else {"uk": 1.27, "jp": 0.0067}[cc]
        # ZCB approximation: DV01 USD mn/bp ≈ face_mn × fx × wam / 10000 * 100? 
        # dv01_mn(face_mn, years, 0, y, fx) with face in local million.
        from issuance.pricing import dv01_mn
        face_mn = face_bn * 1000.0
        dv01 = float(dv01_mn(np.array([face_mn]), np.array([wam_ref]), np.array([0.0]),
                             np.array([2.0]), np.array([fx]))[0])
        extra.append({
            "cc": cc, "sec_type": "Bill", "face_bn": face_bn, "dv01_usd_mn": dv01,
            "wam": wam_ref, "source": f"published stock × ops bill WAM {wam_ref:.2f}y ({pub.loc[cc, 'source']})",
        })
    return pd.concat([dv, pd.DataFrame(extra)], ignore_index=True) if extra else dv


def private_scale(snap: pd.DataFrame) -> dict[str, float]:
    """Private / gross on the coupon book (1 − CB share); bills left at 1 (SOMA bills not stripped here)."""
    snap = snap.set_index("cc")
    out = {}
    for cc in CC_MAP:
        cb = snap.loc[cc, "cb_share"] if cc in snap.index else np.nan
        out[cc] = float(1.0 - cb) if np.isfinite(cb) else 1.0
    return out


def _bond_price(resid: float, coupon_pct: float, freq: int, zfn) -> float:
    """Clean price per 1 of par from a continuously compounded zero curve, percent yields."""
    step = 1.0 / freq
    resid = max(float(resid), 1.0 / 365.0)
    if resid < step - 1e-9:
        return float(np.exp(-float(zfn(resid)) / 100.0 * resid))
    times = np.arange(step, resid + 1e-9, step)
    dfs = np.exp(-zfn(times) / 100.0 * times)
    c = 0.0 if not np.isfinite(coupon_pct) else float(coupon_pct) / 100.0
    return float(np.sum(c * step * dfs) + dfs[-1])


def _apf_share() -> pd.Series:
    """APF nominal / gilt nominal, by ISIN, from the live stock file.

    Purchases minus sales leave matured gilts in the portfolio. The stock file does not.
    Index-linked rows in that file are zero: the monetary-policy portfolio holds none.
    """
    path = REPO / "projects" / "uk_long_end" / "workbook" / "35_stock_apf_freefloat_by_isin.csv"
    df = pd.read_csv(path)
    nom = pd.to_numeric(df["nominal_mn"], errors="coerce")
    held = pd.to_numeric(df["apf_mn"], errors="coerce").fillna(0.0) + pd.to_numeric(df["official_mn"], errors="coerce").fillna(0.0)
    share = (held / nom).clip(0, 1)
    share.index = df["isin"].astype(str)
    return share.groupby(level=0).mean()


def _soma_by_isin() -> pd.Series:
    s = pd.read_parquet(REPO / "issuance" / "cache" / "soma_holdings.parquet")
    last = s[s["asof"] == s["asof"].max()]
    return last.groupby(last["isin"].astype(str))["official_stock_mn"].sum()


def _cb_by_isin() -> pd.Series:
    """Central-bank face, million local, by the auction-tape key.

    Japan is the BoJ issue file. Germany, France and Italy are the published PSPP and PEPP
    country stocks placed on ISINs (``official_isin_flow``).
    """
    path = CACHE / "official_isin_flow.parquet"
    if not path.is_file():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df = df[df["country"].isin(["DE", "FR", "IT", "JP"])].copy()
    df["month"] = pd.to_datetime(df["month"])
    last = df["month"].groupby(df["country"]).transform("max")
    snap = df[df["month"].eq(last)]
    held = pd.to_numeric(snap["official_stock_mn"], errors="coerce").fillna(0.0)
    return held.groupby(snap["isin"].astype(str)).sum()


def mtm_table(shock_bp: float = 100.0, real_shock_bp: float | None = None) -> pd.DataFrame:
    """Holder mark-to-market from discounting each ISIN on the government zero curve.

    Nominals and bills use the nominal curve. Linkers use the real curve and the issuer's
    index coefficient. Floaters are priced to the next reset (a quarter). ``real_shock_bp``
    defaults to ``shock_bp``. Private holdings are SOMA by ISIN, the APF by ISIN, the BoJ
    by issue, and the ECB country stock placed on ISINs.
    """
    from sovereign_cash.debt.gov_curve import FREQ, _curve

    if real_shock_bp is None:
        real_shock_bp = shock_bp
    from sovereign_cash.debt.index_ratio import ratios as isin_ratios
    from sovereign_cash.debt.isin_book import outstanding
    soma_all = pd.read_parquet(REPO / "issuance" / "cache" / "soma_holdings.parquet")
    soma_asof = soma_all["asof"].max()
    # The end-2025 bill book has already rolled. Match SOMA to the book on the same day.
    bk = outstanding(pd.Timestamp(soma_asof))
    from sovereign_cash.debt.cashflow import _index_ratios
    ratios = _index_ratios(2025)
    ratios.update(isin_ratios(pd.Timestamp(soma_asof)))
    soma = _soma_by_isin()
    apf = _apf_share()
    cb_isin = _cb_by_isin()
    soma_all = soma_all[soma_all["asof"] == soma_asof]
    soma_bill_mn = float(soma_all.loc[soma_all["sec_type"] == "Bill", "official_stock_mn"].sum())
    soma_bill_matched = 0.0
    snap = pd.read_parquet(DERIVED / "debt_snapshot.parquet").set_index("cc")
    ours = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    gdp = ours[ours["year"] == 2025].set_index("cc")["gdp_bn"]
    curves = {}

    def z(name, bp):
        key = (name, bp)
        if key not in curves:
            got = _curve(name, bp)
            curves[key] = None if got is None else got[0]
        return curves[key]

    rows = []
    for cc_up, g in bk.groupby("country"):
        cc = str(cc_up).lower()
        if cc not in CC_MAP:
            continue
        freq = FREQ.get(cc, 2)
        fx = float(snap.at[cc, "fx"])
        gdp_usd = float(gdp[cc]) * fx
        zn, zn1 = z(cc, 0.0), z(cc, shock_bp)
        zr, zr1 = z(f"{cc}_real", 0.0), z(f"{cc}_real", real_shock_bp)
        if zn is None:
            continue
        for r in g.itertuples(index=False):
            resid = max(float(r.resid_yrs), 1.0 / 365.0)
            st = r.sec_type
            if st == "Linker" and zr is not None:
                p0 = _bond_price(resid, r.coupon_pct, freq, zr)
                p1 = _bond_price(resid, r.coupon_pct, freq, zr1 if zr1 is not None else zr)
            elif st == "FRN":
                p0 = _bond_price(min(resid, 0.25), 0.0, freq, zn)
                p1 = _bond_price(min(resid, 0.25), 0.0, freq, zn1)
            else:
                cp = 0.0 if st == "Bill" else r.coupon_pct
                p0 = _bond_price(resid, cp, freq, zn)
                p1 = _bond_price(resid, cp, freq, zn1)
            key = str(getattr(r, "key"))
            face_bn = float(r.face_mn) / 1000.0
            if st == "Linker":
                face_bn *= float(ratios.get(key, ratios.get(cc, 1.0)))
            loss_local_bn = face_bn * (p0 - p1)
            matched = False
            if cc == "us" and key in soma.index:
                held = float(soma[key])
                priv = max(float(r.face_mn) - held, 0.0) / float(r.face_mn)
                src = "SOMA by ISIN"
                matched = True
                if st == "Bill":
                    soma_bill_matched += min(held, float(r.face_mn))
            elif cc == "uk" and key in apf.index:
                priv = 1.0 - float(apf[key])
                src = "APF stock by ISIN"
                matched = True
            elif cc in ("us", "uk"):
                # The Fed and the APF are known bond by bond. A bond absent from that list is private.
                priv, src = 1.0, "not held by the central bank"
                matched = False
            elif key in cb_isin.index and float(r.face_mn) > 0:
                share = min(float(cb_isin[key]) / float(r.face_mn), 1.0)
                priv = 1.0 - share
                src = "BoJ by issue" if cc == "jp" else "ECB stock placed on ISINs"
                matched = True
            else:
                priv, src = 1.0, "not on the central-bank ISIN file"
            loss_usd_bn = loss_local_bn * priv * fx
            rows.append({
                "cc": cc, "sec_type": st, "face_bn": float(r.face_mn) / 1000.0,
                "dv01_usd_mn": loss_usd_bn / shock_bp * 1000.0,
                "private_dv01_usd_mn": loss_usd_bn / shock_bp * 1000.0,
                "mtm_pct_gdp": 100.0 * loss_usd_bn / gdp_usd,
                "wam": resid, "source": src,
                "private_share": priv,
            })
    # SOMA bills whose CUSIP is not on the tape are still Fed-owned. Take them off the unmatched bill stock.
    loose = soma_bill_mn - soma_bill_matched
    if loose > 1.0:
        pool = sum(r["face_bn"] for r in rows if r["cc"] == "us" and r["sec_type"] == "Bill" and r["source"] != "SOMA by ISIN")
        if pool > 0:
            haircut = min(loose / 1000.0 / pool, 1.0)
            for r in rows:
                if r["cc"] == "us" and r["sec_type"] == "Bill" and r["source"] != "SOMA by ISIN":
                    keep = 1.0 - haircut
                    r["dv01_usd_mn"] *= keep
                    r["private_dv01_usd_mn"] *= keep
                    r["mtm_pct_gdp"] *= keep
                    r["private_share"] *= keep
                    r["source"] = "SOMA bills, tape has no CUSIP"
    # The MoF bill stock has no ISINs. The BoJ outright T-bill file (tmei) is zero, so the
    # stock is private. Underwriting of financing bills is not in that file.
    jb = B.jp_mof_bills()
    face = float(jb["bills_bn"])
    zn, zn1 = z("jp", 0.0), z("jp", shock_bp)
    if face > 0 and zn is not None and zn1 is not None and "jp" in gdp.index:
        p0 = _bond_price(0.25, 0.0, FREQ.get("jp", 2), zn)
        p1 = _bond_price(0.25, 0.0, FREQ.get("jp", 2), zn1)
        fx = float(snap.at["jp", "fx"])
        gdp_usd = float(gdp["jp"]) * fx
        loss_usd = face * (p0 - p1) * fx
        rows.append({
            "cc": "jp", "sec_type": "Bill", "face_bn": face,
            "dv01_usd_mn": loss_usd / shock_bp * 1000.0,
            "private_dv01_usd_mn": loss_usd / shock_bp * 1000.0,
            "mtm_pct_gdp": 100.0 * loss_usd / gdp_usd,
            "wam": 0.25,
            "source": "MoF bill stock, 3-month life; BoJ outright T-bills are zero",
            "private_share": 1.0,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    grouped = []
    for (cc, st, src), d in out.groupby(["cc", "sec_type", "source"]):
        grouped.append({
            "cc": cc, "sec_type": st, "source": src,
            "face_bn": float(d["face_bn"].sum()),
            "dv01_usd_mn": float(d["dv01_usd_mn"].sum()),
            "private_dv01_usd_mn": float(d["private_dv01_usd_mn"].sum()),
            "mtm_pct_gdp": float(d["mtm_pct_gdp"].sum()),
            "wam": float(np.average(d["wam"], weights=d["face_bn"].clip(lower=1e-9))),
        })
    return pd.DataFrame(grouped)


def interest_shock(shock_bp: float = 100.0) -> pd.DataFrame:
    """2029 GG interest/revenue under baseline, +shock, +shock with interest borrowed."""
    src = S.load(False)
    ann = B.annual(src)
    book = B._jp_bills(B.monthly_book(), src)
    fl = B.flows_monthly(book)
    sched = B.redemption_schedule(src)
    from issuance import imf_investor_base as ib
    snap = B.snapshot(book, fl, ann, src, ib.panel(), sched)
    hist = B.history(book, fl, ann)
    asof = pd.Timestamp(fl.loc[~fl["is_forecast"].astype(bool) & (fl["cc"] == "uk"), "month"].max())
    y0 = B.Y0
    ca = B.country_annual(book, fl, ann, y0 + 1, asof, snap, B._bills_dec(src), src)

    base = B.project(ca, fl, ann, snap, sched, y0, 0.0, False, src)
    shk = B.project(ca, fl, ann, snap, sched, y0, shock_bp, False, src)
    bor = B.project_borrowed(ca, fl, ann, snap, sched, y0, shock_bp, src)

    ours = pd.read_parquet(DERIVED / "debt_country_annual.parquet")
    snap_p = snap.set_index("cc") if "cc" in snap.columns else pd.read_parquet(DERIVED / "debt_snapshot.parquet").set_index("cc")

    def gg(path: pd.DataFrame, cc: str, year: int) -> float:
        b = ours[(ours["cc"] == cc) & (ours["year"] == 2025)].iloc[0]
        ggrev = float(ours[(ours["cc"] == cc) & (ours["year"] == year)]["gg_revenue_bn"].iloc[0])
        I = float(path[(path["cc"] == cc) & (path["year"] == year)]["interest_bn"].iloc[0])
        # Frozen residual: GG moves only with the marketable central bill.
        # Scaled: the old 1:1 gross-up, which also shocks states and social security.
        g0, i0 = float(b["gg_interest_bn"]), float(b["interest_bn"])
        return 100.0 * (I + g0 - i0) / ggrev, 100.0 * g0 * I / i0 / ggrev

    rows = []
    for cc in ("us", "uk", "jp", "fr", "it", "de"):
        b, bs = gg(base, cc, 2029)
        s, ss = gg(shk, cc, 2029)
        r, rs = gg(bor, cc, 2029)
        rows.append({
            "cc": cc,
            "base": b, "shock": s, "borrowed": r,
            "d100": s - b, "d100_borrow": r - b,
            "d100_scaled": ss - bs, "d100_borrow_scaled": rs - bs,
            "bills": float(snap_p.loc[cc, "bill_share_stock"]),
            "bill_source": snap_p.loc[cc, "bill_source"],
            "rshare": float(shk[(shk["cc"] == cc) & (shk["year"] == 2029)]["repriced_share"].iloc[0]),
        })
    return pd.DataFrame(rows).set_index("cc")


def run() -> dict:
    mtm = mtm_table(100.0)
    interest = interest_shock(100.0)
    piv = mtm.pivot_table(index="cc", columns="sec_type", values="mtm_pct_gdp", aggfunc="sum").fillna(0.0)
    piv["total"] = piv.sum(axis=1)
    return {"interest": interest, "mtm": mtm, "mtm_by_type": piv}


if __name__ == "__main__":
    out = run()
    print(out["interest"].round(2))
    print(out["mtm_by_type"].round(2))
    print(out["mtm"][["cc", "sec_type", "face_bn", "dv01_usd_mn", "mtm_pct_gdp", "wam", "source"]]
          .sort_values(["cc", "sec_type"]).to_string(index=False))
