"""Who takes the debt, and who buys when the yield rises. Country by country.

The yield impact is those two facts. A z-score of aggregate demand is not a substitute: the
sector that absorbs the supply is often not the sector that leans against the selloff.

Each country keeps its own tape.

  UK     ONS transactions: ICPF, OFI, overseas, banks, the APF, households.
  JP     JSDA dealer-customer outright transactions, by issue type: medium (2–5y) against
         the 2y, long (10y) against the 10y, super-long (20/30/40y) against the 30y and against
         10s30s. The IMF nominal change stays beside it.
  US     Treasury coupon-auction allotments; the quarterly change in foreign official custody,
         bank Treasuries and Treasury money funds; OFS-2 par holdings, the quarterly change
         (Fed, banks, foreign, mutual funds, insurers, private and public pensions).
  DE/FR/IT  ECB securities-holdings transactions, insurers and pension funds separate.

Three measurements, each the one the literature reports, re-estimated on these tapes.

  Absorption   sector flow on the tape's own total flow (Fang, Hardy and Lewis, RFS 2025,
               equation 2). Their advanced-economy panel puts 0.36 of a debt increase on
               domestic nonbanks and 0.36 on foreign nonbanks. Those are panel coefficients.
  Price        percent of GDP bought per 100bp, at the 2y, the 10y, the 30y and 10s30s,
               this quarter and the next. The Bank of England preferred-habitat results
               put insurers and pension funds on the long end. The same regression on the
               10y, in 2016–2021 and 2022–, is the bought-per-100bp slope.
  Reversion    the next quarter's 10s30s move after the long-habitat sector bought, and after
               it sold. The 10y half-life is an AR(1) on the demeaned quarterly level from 2012.

Arslanalp and Tsuda (2012) froze a 2008Q3–2011Q4 correlation into a risk score. Nothing here
uses that score. Flows and yields are jointly determined. This is not a Koijen–Yogo elasticity.
The named buyer is the leave-one-out partial association: one sector's flow against the yield,
the rest of that tape held fixed (Frisch–Waugh). A kitchen-sink regression of the yield on
every sector is not run. The gate is a Student-t on that one coefficient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.paths import CACHE, REPO

MARKET = REPO / "issuance" / "cache" / "market_monthly.parquet"
YIELDS = ("us", "uk", "de", "fr", "it", "jp")
ERAS = (("2016–2021", "2016-01-01", "2021-12-31"), ("2022–", "2022-01-01", "2099-01-01"))
MIN_N = 12

UK_SECTORS = (
    ("icpf", "ICPF"), ("ofi", "OFI"), ("overseas", "Overseas"),
    ("banks", "Banks"), ("apf", "APF"), ("households", "Households"),
)
EA_SECTORS = (
    ("eurosystem", "Eurosystem"), ("mfi_ex_cb", "Banks"), ("funds", "Funds"),
    ("insurers", "Insurers"), ("pension", "Pension funds"), ("ofi", "OFI"),
    ("pension", "Pension funds"), ("ofi", "OFI"), ("households", "Households"), ("row", "Non-euro-area"),
)
US_AUCTION = (
    ("depository", "Banks (auction)"), ("pension_ins", "Pensions and insurers (auction)"),
    ("inv_funds", "Funds (auction)"), ("foreign", "Foreign (auction)"),
    ("dealers", "Dealers (auction)"), ("soma", "SOMA (auction)"),
)
JSDA_INV = (
    ("city_banks", "City banks"), ("regional_banks", "Regional banks"),
    ("trust_banks", "Trust banks"), ("insurers", "Insurers"),
    ("inv_trusts", "Investment trusts"), ("foreigners", "Foreigners"),
    ("individuals", "Individuals"), ("mutual_aid", "Mutual aid"),
)
JSDA_MATCH = {"medium": "dy2", "long": "dy10", "super_long": "dy30"}
US_OFS = (
    ("fed_and_government", "Fed"), ("banks", "Banks"), ("rest_of_world", "Foreign"),
    ("mutual_funds", "Mutual funds"), ("insurance", "Insurers"),
    ("private_pensions", "Private pensions"), ("public_pensions", "Public pensions"),
)
IMF_SECTORS = (
    ("dcb", "Central bank"), ("db", "Banks"), ("dnb", "Domestic nonbanks"),
    ("fo", "Foreign official"), ("fb", "Foreign banks"), ("fnb", "Foreign nonbanks"),
)


def _rows(cc, quarter, sector, label, flow, tape, share=np.nan, supply_bn=np.nan, match_tenor=None) -> dict:
    return {"cc": cc, "quarter_end": pd.Timestamp(quarter), "sector": sector, "label": label,
            "flow_bn": float(flow) if pd.notna(flow) else np.nan, "share": share,
            "supply_bn": float(supply_bn) if pd.notna(supply_bn) else np.nan, "tape": tape,
            "match_tenor": match_tenor if match_tenor else ""}


def _uk() -> pd.DataFrame:
    from issuance import uk_holders as uk
    q = uk.quarterly_panel()
    rows = []
    for sector, label in UK_SECTORS:
        if sector not in q.columns:
            continue
        for dt, v in q[sector].items():
            supply = q.at[dt, "total"] if "total" in q.columns else np.nan
            rows.append(_rows("uk", dt, sector, label, v, "ONS transactions", supply_bn=supply))
    return pd.DataFrame(rows)


def _ea() -> pd.DataFrame:
    from issuance import ecb_holders as ec
    p = ec.shss_panel()
    want = [s for s, _ in EA_SECTORS]
    f = p[(p["measure"] == "F") & p["sector"].isin(want + ["total"])]
    lab = dict(EA_SECTORS)
    tot = f[f["sector"] == "total"].set_index(["country", "quarter_end"])["value_eur_bn"]
    rows = []
    for r in f[f["sector"].isin(want)].itertuples(index=False):
        supply = tot.get((r.country, r.quarter_end), np.nan)
        rows.append(_rows(r.country.lower(), r.quarter_end, r.sector, lab[r.sector], r.value_eur_bn,
                          "ECB transactions", supply_bn=supply))
    return pd.DataFrame(rows)


def _us() -> pd.DataFrame:
    rows = []
    path = CACHE / "us_allotments.parquet"
    if path.exists():
        a = pd.read_parquet(path)
        a = a[a["security_type"].astype(str).str.upper().ne("BILL")].copy()
        a = a[pd.to_numeric(a["term_years"], errors="coerce").ge(1)].copy()
        a["term_years"] = pd.to_numeric(a["term_years"])
        a["match"] = np.where(a["term_years"] <= 3, "dy2", np.where(a["term_years"] <= 10, "dy10", "dy30"))
        a["quarter_end"] = pd.to_datetime(a["issue_date"]) + pd.offsets.QuarterEnd(0)
        lab = dict(US_AUCTION)
        g = a.groupby(["quarter_end", "investor_class", "match"], as_index=False)["amount_bn"].sum()
        issued = (a.drop_duplicates(["cusip", "issue_date"])
                  .groupby(["quarter_end", "match"])["total_bn"].sum())
        for r in g.itertuples(index=False):
            if r.investor_class not in lab:
                continue
            issued_q = float(issued.get((r.quarter_end, r.match), np.nan))
            share = r.amount_bn / issued_q if issued_q else np.nan
            rows.append(_rows("us", r.quarter_end, r.investor_class, lab[r.investor_class], r.amount_bn,
                              "Treasury auction allotments", share, issued_q, match_tenor=r.match))
        soma = (a.drop_duplicates(["cusip", "issue_date"])
                .groupby(["quarter_end", "match"])[["soma_bn", "total_bn"]].sum())
        for (dt, match), v in soma.iterrows():
            rows.append(_rows("us", dt, "soma", "SOMA (auction)", v["soma_bn"],
                              "Treasury auction allotments",
                              v["soma_bn"] / v["total_bn"] if v["total_bn"] else np.nan,
                              v["total_bn"], match_tenor=match))
    fred = (
        ("fred_WMTSECL1.csv", "WMTSECL1", 1e-3, "custody", "Foreign official (custody)"),
        ("fred_TNMACBW027SBOG.csv", "TNMACBW027SBOG", 1.0, "banks_h8", "Banks (H.8)"),
        ("fred_WRMFNS.csv", "WRMFNS", 1.0, "mmf", "Money funds"),
    )
    for name, col, scale, sector, label in fred:
        p = CACHE / "us_demand" / name
        if not p.exists():
            continue
        s = pd.read_csv(p, parse_dates=["observation_date"]).set_index("observation_date")[col]
        s = pd.to_numeric(s, errors="coerce") * scale
        s = s.where(s > 0)
        q = s.resample("QE").last().diff()
        for dt, v in q.items():
            rows.append(_rows("us", dt, sector, label, v, "Balance-sheet change"))
    return pd.DataFrame(rows)


def _imf(countries=("jp", "se", "no")) -> pd.DataFrame:
    from issuance import imf_investor_base as ib
    w = ib.wide().sort_values(["country", "quarter_end"])
    lab = dict(IMF_SECTORS)
    rows = []
    for cc in countries:
        g = w[w["country"] == cc.upper()]
        if g.empty:
            continue
        for sector, label in IMF_SECTORS:
            if sector not in g.columns:
                continue
            d = g[sector].diff()
            supply = g["total"].diff() if "total" in g.columns else pd.Series(np.nan, index=g.index)
            for dt, v, s in zip(g["quarter_end"], d, supply):
                rows.append(_rows(cc, dt, sector, label, v, "IMF nominal change", supply_bn=s))
    return pd.DataFrame(rows)


def _jsda() -> pd.DataFrame:
    """Dealer-customer JGB flow by issue type. super-long vs the 30y, long vs the 10y, medium vs the 2y."""
    path = CACHE / "jsda_investor.parquet"
    if not path.exists():
        return pd.DataFrame()
    p = pd.read_parquet(path)
    p["month"] = pd.to_datetime(p["month"])
    want = dict(JSDA_INV)
    p = p[p["investor"].isin(set(want) | {"regional2"}) & p["maturity"].isin(JSDA_MATCH)].copy()
    p["investor"] = p["investor"].replace({"regional2": "regional_banks"})
    p["quarter_end"] = p["month"] + pd.offsets.QuarterEnd(0)
    complete = p.groupby("quarter_end")["month"].max()
    complete = complete[complete.dt.month.isin([3, 6, 9, 12])].index
    p = p[p["quarter_end"].isin(complete)]
    g = p.groupby(["quarter_end", "investor", "maturity"], as_index=False)["net_purchase_jpy_bn"].sum()
    rows = [_rows("jp", r.quarter_end, r.investor, want[r.investor], r.net_purchase_jpy_bn,
                  "JSDA transactions", match_tenor=JSDA_MATCH[r.maturity])
            for r in g.itertuples(index=False)]
    rows.extend(_boj_rows())
    return pd.DataFrame(rows)


BOJ_MATCH = {"1-3y": "dy2", "3-5y": "dy2", "5-10y": "dy10", "10y+": "dy30", "10-25y": "dy30", "25y+": "dy30"}


def _boj_rows() -> list:
    """BoJ outright JGB purchases, ¥bn, on the same tenor grid as JSDA."""
    path = CACHE / "boj_operations.parquet"
    if not path.exists():
        return []
    b = pd.read_parquet(path)
    b = b[b["bucket"].isin(BOJ_MATCH)].copy()
    b["quarter_end"] = pd.to_datetime(b["date"]) + pd.offsets.QuarterEnd(0)
    b["match"] = b["bucket"].map(BOJ_MATCH)
    g = b.groupby(["quarter_end", "match"], as_index=False)["accepted_mn"].sum()
    return [_rows("jp", r.quarter_end, "boj", "BoJ", r.accepted_mn / 1000.0,
                   "JSDA transactions", match_tenor=r.match) for r in g.itertuples(index=False)]


JPX_INV = (
    ("city_regional_banks", "City and regional banks"),
    ("trust_banks", "Trust banks"),
    ("life_nonlife_insurers", "Insurers"),
    ("investment_trusts", "Investment trusts"),
    ("foreigners", "Foreigners"),
    ("individuals", "Individuals"),
    ("securities_cos", "Securities companies"),
    ("proprietary", "Dealers"),
)
JPX_MATCH = {"jgb10y": "dy10", "mini_jgb20y": "dy30"}


def _tic() -> pd.DataFrame:
    """TIC SLT net purchases of long-term Treasuries, official against private. $bn.

    Country rows are custodians and are not a second contest. The split starts in 2023-02,
    so a 12-quarter window exists only at the end of the sample. A quarter needs three months.
    """
    path = CACHE / "xborder_flows.parquet"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(path)
    want = {
        "tic_net_lt_treasuries_official": ("official", "Foreign official"),
        "tic_net_lt_treasuries_private": ("private", "Foreign private"),
    }
    d = d[d["series"].isin(want)].copy()
    if d.empty:
        return pd.DataFrame()
    d["quarter_end"] = pd.to_datetime(d["period"]) + pd.offsets.QuarterEnd(0)
    g = d.groupby(["quarter_end", "series"], as_index=False).agg(flow_bn=("value", "sum"), n=("value", "size"))
    g = g[g["n"] >= 3]
    rows = []
    for r in g.itertuples(index=False):
        sector, label = want[r.series]
        rows.append(_rows("us", r.quarter_end, sector, label, r.flow_bn, "TIC long-term"))
    return pd.DataFrame(rows)


def _jpx() -> pd.DataFrame:
    """Weekly JPX JGB futures net buying, summed to the quarter. 10y futures vs the 10y, mini 20y vs the 30y."""
    path = CACHE / "jpx_jgb_futures_investor.parquet"
    if not path.exists():
        return pd.DataFrame()
    j = pd.read_parquet(path)
    j = j[(j["unit"] == "contracts") & j["contract"].isin(JPX_MATCH) & j["investor"].isin(dict(JPX_INV))].copy()
    j["quarter_end"] = pd.to_datetime(j["week_end"]) + pd.offsets.QuarterEnd(0)
    g = j.groupby(["quarter_end", "investor", "contract"], as_index=False)["net"].sum()
    lab = dict(JPX_INV)
    rows = [_rows("jp", r.quarter_end, r.investor, lab[r.investor], r.net,
                   "JPX JGB futures", match_tenor=JPX_MATCH[r.contract])
            for r in g.itertuples(index=False)]
    return pd.DataFrame(rows)


def _ofs() -> pd.DataFrame:
    """Treasury OFS-2 par holdings, quarterly change. Face, so the change is not a price revaluation."""
    path = CACHE / "holders.parquet"
    if not path.exists():
        return pd.DataFrame()
    h = pd.read_parquet(path)
    h = h[(h["country"] == "US") & h["holder"].isin(dict(US_OFS))].copy()
    h["quarter_end"] = pd.to_datetime(h["period"])
    lab = dict(US_OFS)
    rows = []
    for holder, g in h.groupby("holder"):
        g = g.sort_values("quarter_end")
        supply = g.drop_duplicates("quarter_end").set_index("quarter_end")["stock_mn"].diff() / 1000.0
        d = g["amount_mn"].diff() / 1000.0
        for dt, v, s in zip(g["quarter_end"], d, supply.reindex(g["quarter_end"]).to_numpy()):
            rows.append(_rows("us", dt, holder, lab[holder], v, "OFS-2 par change", supply_bn=s))
    return pd.DataFrame(rows)


def flows() -> pd.DataFrame:
    parts = []
    for fn in (_uk, _ea, _us, _imf, _jsda, _ofs, _jpx, _tic):
        try:
            parts.append(fn())
        except Exception:
            continue
    out = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    return out.sort_values(["cc", "sector", "quarter_end"]).reset_index(drop=True)


def _yields() -> pd.DataFrame:
    """Quarter-end yield changes in basis points. y2, y10, y30 and 10s30s where the curve has them."""
    m = pd.read_parquet(MARKET)
    keep = m["series"].isin(("y2", "y3", "y10", "y30", "s1030")) & m["country"].str.lower().isin(YIELDS)
    m = m.loc[keep].copy()
    m["month"] = pd.to_datetime(m["month"])
    m["quarter_end"] = m["month"] + pd.offsets.QuarterEnd(0)
    q = (m.sort_values("month")
         .groupby(["country", "series", "quarter_end"], as_index=False)["value"].last())
    q["cc"] = q["country"].str.lower()
    q["dy"] = q.groupby(["cc", "series"])["value"].diff() * 100.0
    wide = q.pivot_table(index=["cc", "quarter_end"], columns="series", values="dy")
    wide = wide.rename(columns={c: f"d{c}" for c in wide.columns})
    lvl = q[q["series"] == "y10"].set_index(["cc", "quarter_end"])["value"]
    wide["y10"] = lvl
    return wide.reset_index()


def _gdp() -> pd.DataFrame:
    from issuance import imf_investor_base as ib
    w = ib.wide()[["country", "quarter_end", "gdp"]].copy()
    w["cc"] = w["country"].str.lower()
    w["quarter_end"] = pd.to_datetime(w["quarter_end"])
    return w[["cc", "quarter_end", "gdp"]]


def _corr(flow: np.ndarray, dy: np.ndarray) -> tuple[float, float, int]:
    m = np.isfinite(flow) & np.isfinite(dy)
    n = int(m.sum())
    if n < MIN_N or np.nanstd(dy[m]) == 0 or np.nanstd(flow[m]) == 0:
        return np.nan, np.nan, n
    x, y = dy[m], flow[m]
    var = float(np.dot(x - x.mean(), x - x.mean()))
    if var <= 0:
        return np.nan, np.nan, n
    slope = float(np.dot(x - x.mean(), y - y.mean()) / var) * 100.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr, slope, n


def response(fl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per country, sector and era: corr(flow, Δ10y). `per_100bp` is % of GDP, or percentage points of the auction for US allotments."""
    fl = _cash(flows() if fl is None else fl)
    y = _yields()
    g = _gdp()
    d = fl.merge(y, on=["cc", "quarter_end"], how="inner").merge(g, on=["cc", "quarter_end"], how="left")
    d["flow_pct"] = np.where(d["gdp"] > 0, 100.0 * d["flow_bn"] / d["gdp"], np.nan)
    auction = d["tape"].eq("Treasury auction allotments")
    d["response"] = np.where(auction, d["share"] * 100.0, d["flow_pct"])
    d["unit"] = np.where(auction, "pp of the auction", "% GDP")
    if "match_tenor" in d.columns:
        matched = d["match_tenor"].fillna("").astype(str)
        d = d[matched.eq("") | matched.eq("dy10")]
    rows = []
    for (cc, sector), sub in d.groupby(["cc", "sector"]):
        label = sub["label"].iloc[0]
        tape = sub["tape"].iloc[0]
        unit = sub["unit"].iloc[0]
        for era, a, b in ERAS:
            w = sub[(sub["quarter_end"] >= a) & (sub["quarter_end"] <= b)]
            corr, slope, n = _corr(w["response"].to_numpy(float), w["dy10"].to_numpy(float))
            rows.append({"cc": cc, "sector": sector, "label": label, "tape": tape, "era": era,
                         "n": n, "corr": corr, "per_100bp": slope, "unit": unit})
    return pd.DataFrame(rows)


def marginal(fl: pd.DataFrame | None = None, n_quarters: int = 4) -> pd.DataFrame:
    """Largest net buyer over the last n quarters, and its share of gross positive buying."""
    fl = _cash(flows() if fl is None else fl)
    g = _gdp()
    rows = []
    for (cc, tape), sub in fl.groupby(["cc", "tape"]):
        end = sub["quarter_end"].max()
        start = end - pd.DateOffset(months=3 * n_quarters) + pd.Timedelta(days=1)
        w = sub[sub["quarter_end"] >= start]
        bought = w.groupby(["sector", "label", "tape"], as_index=False)["flow_bn"].sum()
        if bought.empty:
            continue
        pos = bought.loc[bought["flow_bn"] > 0, "flow_bn"].sum()
        i = bought["flow_bn"].idxmax()
        top = bought.loc[i]
        gdp = g[(g["cc"] == cc) & (g["quarter_end"] == end)]
        gdp_v = float(gdp["gdp"].iloc[0]) if len(gdp) else np.nan
        rows.append({
            "cc": cc, "tape": tape, "quarter_end": end, "sector": top["sector"], "label": top["label"],
            "flow_pct_gdp": 100.0 * float(top["flow_bn"]) / gdp_v if gdp_v and gdp_v > 0 else np.nan,
            "share_of_gross": float(top["flow_bn"]) / pos if pos > 0 and top["flow_bn"] > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def half_life() -> pd.DataFrame:
    """AR(1) half-life of the demeaned quarterly 10y, in quarters, from 2012."""
    y = _yields()
    y = y[y["quarter_end"] >= "2012-01-01"]
    rows = []
    for cc, sub in y.groupby("cc"):
        v = sub.sort_values("quarter_end")["y10"].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < MIN_N:
            rows.append({"cc": cc, "phi": np.nan, "half_life_q": np.nan, "n": len(v)})
            continue
        x = v - v.mean()
        a, b = x[1:], x[:-1]
        var = float(np.dot(b, b))
        phi = float(np.dot(a, b) / var) if var else np.nan
        hl = float(np.log(0.5) / np.log(phi)) if np.isfinite(phi) and 0 < phi < 1 else np.nan
        rows.append({"cc": cc, "phi": phi, "half_life_q": hl, "n": int(len(v))})
    return pd.DataFrame(rows)


def _beta(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    """OLS slope of y on x, with an intercept. Fang, Hardy and Lewis (2025) equation (2)."""
    m = np.isfinite(y) & np.isfinite(x)
    n = int(m.sum())
    if n < MIN_N:
        return np.nan, n
    xx, yy = x[m], y[m]
    xc, yc = xx - xx.mean(), yy - yy.mean()
    var = float(np.dot(xc, xc))
    if var <= 0:
        return np.nan, n
    return float(np.dot(xc, yc) / var), n


# Their advanced-economy panel, not a number this tape is required to match.
FANG_AE_NONBANK = 0.36

ABS_ERAS = (("2012–2019", "2012-01-01", "2019-12-31"), ("2022–", "2022-01-01", "2099-01-01"))
TENORS = ("dy2", "dy10", "dy30", "ds1030")
HABITAT = {"uk": "icpf", "de": "icpf", "fr": "icpf", "it": "icpf", "jp": "dnb"}


def absorption(fl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of the tape's own supply change taken by each sector.

    The regression is sector flow on the same tape's total flow. Sectors that do not partition
    the total leave a residual, so the coefficients need not sum to one. US custody and bank
    balance sheets have no supply total on the tape and are left out.
    """
    fl = _cash(flows() if fl is None else fl)
    rows = []
    d = fl.dropna(subset=["supply_bn", "flow_bn"])
    for (cc, sector), sub in d.groupby(["cc", "sector"]):
        label, tape = sub["label"].iloc[0], sub["tape"].iloc[0]
        for era, a, b in ABS_ERAS:
            w = sub[(sub["quarter_end"] >= a) & (sub["quarter_end"] <= b)]
            slope, n = _beta(w["flow_bn"].to_numpy(float), w["supply_bn"].to_numpy(float))
            rows.append({"cc": cc, "sector": sector, "label": label, "tape": tape, "era": era,
                         "n": n, "marginal": slope})
    return pd.DataFrame(rows)


def impact(fl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Percent of GDP bought per 100bp, at each tenor, this quarter and the next.

    The 2y against 10s30s is the preferred-habitat cut: insurers and pension funds
    belong on the long end. Auction allotments are a share of the auction and stay out.
    The regression is jointly determined.
    """
    fl = _cash(flows() if fl is None else fl)
    y = _yields()
    g = _gdp()
    d = fl[fl["tape"].ne("Treasury auction allotments")].merge(
        y, on=["cc", "quarter_end"], how="inner").merge(g, on=["cc", "quarter_end"], how="left")
    d["flow_pct"] = np.where(d["gdp"] > 0, 100.0 * d["flow_bn"] / d["gdp"], np.nan)
    d = d.sort_values(["cc", "sector", "quarter_end"])
    for tenor in TENORS:
        if tenor not in d.columns:
            continue
        d[f"next_{tenor}"] = d.groupby(["cc", "sector"])[tenor].shift(-1)
    rows = []
    for (cc, sector), sub in d.groupby(["cc", "sector"]):
        label, tape = sub["label"].iloc[0], sub["tape"].iloc[0]
        split = "match_tenor" in sub.columns and sub["match_tenor"].fillna("").astype(str).str.len().gt(0).any()
        for era, a, b in ERAS:
            era_rows = sub[(sub["quarter_end"] >= a) & (sub["quarter_end"] <= b)]
            for tenor in TENORS:
                if tenor not in era_rows.columns:
                    continue
                w = era_rows
                if split:
                    bucket = "dy30" if tenor == "ds1030" else tenor
                    w = era_rows[era_rows["match_tenor"].fillna("").astype(str).eq(bucket)]
                if w.empty:
                    continue
                w = w.sort_values("quarter_end")
                flow = w["flow_pct"].to_numpy(float)
                corr, per, n = _corr(flow, w[tenor].to_numpy(float))
                _, nxt, n2 = _corr(flow, w[tenor].shift(-1).to_numpy(float))
                rows.append({"cc": cc, "sector": sector, "label": label, "tape": tape, "era": era,
                             "tenor": tenor, "n": n, "corr": corr, "per_100bp": per,
                             "per_100bp_next": nxt, "n_next": n2})
    return pd.DataFrame(rows)


def reversion(fl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Next-quarter 10s30s change after the long-habitat sector bought, and after it sold.

    UK, Germany, France and Italy use insurers and pension funds. Japan uses domestic nonbanks,
    which is the IMF residual and is not an ICPF series. The US tape has no net ICPF flow.
    """
    fl = _cash(flows() if fl is None else fl)
    y = _yields()
    if "ds1030" not in y.columns:
        return pd.DataFrame()
    rows = []
    for cc, sector in HABITAT.items():
        if cc in ("de", "fr", "it"):
            parts = fl[(fl["cc"] == cc) & fl["sector"].isin(["insurers", "pension"])]
            sub = parts.groupby("quarter_end", as_index=False)["flow_bn"].sum()
            sub["cc"] = cc
            sub["label"] = "Insurers and pension funds"
            sector = "insurers+pension"
        else:
            sub = fl[(fl["cc"] == cc) & (fl["sector"] == sector)][
                ["cc", "quarter_end", "flow_bn", "label"]].copy()
        sub = sub.merge(y[["cc", "quarter_end", "ds1030"]], on=["cc", "quarter_end"], how="inner")
        sub = sub[sub["quarter_end"] >= "2016-01-01"].sort_values("quarter_end")
        if sub.empty:
            continue
        sub["next"] = sub["ds1030"].shift(-1)
        for side, mask in (("bought", sub["flow_bn"] > 0), ("sold", sub["flow_bn"] < 0)):
            w = sub.loc[mask, "next"].dropna()
            rows.append({"cc": cc, "sector": sector, "label": sub["label"].iloc[0], "side": side,
                         "n": int(len(w)), "next_s1030_bp": float(w.mean()) if len(w) else np.nan})
    return pd.DataFrame(rows)


AUCTION = "Treasury auction allotments"


def _roll4(s: pd.Series) -> pd.Series:
    return s.rolling(4, min_periods=4).sum()


def tracker(fl: pd.DataFrame | None = None, start: str = "2016-01-01") -> pd.DataFrame:
    """Rolling four-quarter net buying, through the last quarter on each tape.

    `level` is percent of GDP, except US coupon auctions, where it is the percent of
    the four quarters' issuance. `dy10_4q` is the 10y change over those four quarters,
    in basis points. A window that spans more than 400 days is dropped. Quarters that
    have not ended are dropped. GDP is carried forward from the last published quarter.
    """
    fl = _cash(flows() if fl is None else fl)
    y = _yields().sort_values(["cc", "quarter_end"]).copy()
    for col in ("dy2", "dy3", "dy10", "dy30", "ds1030"):
        out = f"{col}_4q"
        if col not in y.columns:
            y[out] = np.nan
            continue
        y[out] = y.groupby("cc")[col].transform(_roll4)
    g = _gdp().sort_values(["cc", "quarter_end"])
    g["gdp"] = g.groupby("cc")["gdp"].ffill()
    parts = []
    d = fl.dropna(subset=["flow_bn"]).sort_values(["cc", "tape", "sector", "quarter_end"])
    if "match_tenor" not in d.columns:
        d["match_tenor"] = ""
    d["match_tenor"] = d["match_tenor"].fillna("").astype(str)
    long = d[d["match_tenor"].eq("dy30")].assign(match_tenor="ds1030")
    d = pd.concat([d, long], ignore_index=True)
    for (_, _, _, _), sub in d.groupby(["cc", "tape", "sector", "match_tenor"]):
        sub = sub.copy()
        span = sub["quarter_end"].diff(3)
        sub["flow_4q"] = _roll4(sub["flow_bn"])
        sub["supply_4q"] = _roll4(sub["supply_bn"])
        sub.loc[span > pd.Timedelta(days=400), ["flow_4q", "supply_4q"]] = np.nan
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    ycols = ["cc", "quarter_end"] + [c for c in y.columns if c.endswith("_4q")]
    out = out[out["quarter_end"] >= start].merge(y[ycols], on=["cc", "quarter_end"], how="left")
    out = out.merge(g, on=["cc", "quarter_end"], how="left")
    out = out.sort_values(["cc", "sector", "quarter_end"])
    out["gdp"] = out.groupby(["cc", "sector"])["gdp"].ffill()
    out = out[out["quarter_end"] <= pd.Timestamp.today().normalize()]
    out["flow_4q_pct"] = np.where(out["gdp"] > 0, 100.0 * out["flow_4q"] / out["gdp"], np.nan)
    auction = out["tape"].eq(AUCTION)
    issued = out["supply_4q"].where(out["supply_4q"] > 0)
    out["level"] = np.where(auction, 100.0 * out["flow_4q"] / issued, out["flow_4q_pct"])
    out["unit"] = np.where(auction, "% of coupon auctions", "% of GDP")
    pos = (out["flow_4q"].where(out["flow_4q"] > 0)
           .groupby([out["cc"], out["tape"], out["match_tenor"], out["quarter_end"]]).transform("sum"))
    out["share_of_gross"] = np.where(pos > 0, out["flow_4q"] / pos, np.nan)
    out["tenor"] = np.where(out["match_tenor"].eq(""), "dy10", out["match_tenor"])
    out["dy_4q"] = np.nan
    for tenor, name in (("dy2", "dy2_4q"), ("dy3", "dy3_4q"), ("dy10", "dy10_4q"),
                        ("dy30", "dy30_4q"), ("ds1030", "ds1030_4q")):
        if name in out.columns:
            out.loc[out["tenor"].eq(tenor), "dy_4q"] = out.loc[out["tenor"].eq(tenor), name]
    keep = ["cc", "tape", "sector", "label", "tenor", "quarter_end", "flow_4q", "flow_4q_pct",
            "level", "unit", "share_of_gross", "dy10_4q", "dy_4q"]
    return out[keep].sort_values(["cc", "tape", "sector", "quarter_end"]).reset_index(drop=True)


CURVE = ("dy2", "dy3", "dy10", "dy30", "ds1030")
CURVE_NAME = {"dy2": "2y", "dy3": "3y", "dy10": "10y", "dy30": "30y", "ds1030": "10s30s"}
BUYER_WINDOW = 12


def _cash(fl: pd.DataFrame) -> pd.DataFrame:
    """Quantity charts are currency. Futures contracts stay on the heatmap only."""
    if "tape" not in fl.columns:
        return fl
    return fl[fl["tape"] != "JPX JGB futures"]


def _fwl(y: np.ndarray, x: np.ndarray, Z: np.ndarray, min_n: int = 8):
    """Partial slope of y on x given controls Z. Returns beta, t, partial corr, df, and the two residual series."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    Z = np.atleast_2d(np.asarray(Z, float))
    if Z.shape[0] == 1 and Z.shape[1] == len(y):
        Z = Z.T
    elif Z.shape[0] != len(y):
        Z = Z.T
    k = Z.shape[1]
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)
    n = int(m.sum())
    df = n - k - 2
    rx = np.full(len(y), np.nan)
    ry = np.full(len(y), np.nan)
    if n < max(min_n, k + 4) or df < 3:
        return np.nan, np.nan, np.nan, df, rx, ry
    def _resid(v, C):
        A = np.column_stack([np.ones(len(v)), C])
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    ry_m = _resid(y[m], Z[m])
    rx_m = _resid(x[m], Z[m])
    if np.nanstd(rx_m) == 0 or np.nanstd(ry_m) == 0:
        return np.nan, np.nan, np.nan, df, rx, ry
    ssx = float(np.dot(rx_m, rx_m))
    beta = float(np.dot(rx_m, ry_m) / ssx)
    err = ry_m - beta * rx_m
    se = float(np.sqrt((np.dot(err, err) / df) / ssx))
    t = beta / se if se > 0 else np.nan
    pcorr = float(np.corrcoef(rx_m, ry_m)[0, 1])
    rx[m] = rx_m
    ry[m] = ry_m
    return beta, t, pcorr, df, rx, ry


def _lead_t(tstats: np.ndarray, dfs: np.ndarray) -> tuple[int, float, float, bool] | None:
    """Winner by partial t. Clears when its t beats the Student 5% point and the next sector by 1."""
    from scipy.stats import t as student_t
    finite = np.isfinite(tstats)
    if int(finite.sum()) < 2:
        return None
    order = np.argsort(np.where(finite, tstats, -np.inf))
    i1, i2 = int(order[-1]), int(order[-2])
    t1, t2 = float(tstats[i1]), float(tstats[i2])
    df = int(dfs[i1]) if np.isfinite(dfs[i1]) else 1
    crit = float(student_t.ppf(0.975, max(df, 1)))
    identified = bool(np.isfinite(t1) and t1 > crit and t1 - t2 > 1)
    return i1, t1, t2, identified


def _same_sign(a: float, b: float, c: float) -> bool:
    """True when three correlations are finite and share a sign."""
    vals = (a, b, c)
    if not all(np.isfinite(vals)):
        return False
    return all(v > 0 for v in vals) or all(v < 0 for v in vals)


def _pair(flow: np.ndarray, dy: np.ndarray, ctrl: np.ndarray | None = None, min_n: int = 8) -> float:
    """Correlation of a flow with a yield change. With `ctrl`, the yield is residualised on the front of the curve first."""
    m = np.isfinite(flow) & np.isfinite(dy)
    if ctrl is not None:
        m = m & np.isfinite(ctrl)
    if int(m.sum()) < min_n or np.nanstd(flow[m]) == 0 or np.nanstd(dy[m]) == 0:
        return np.nan
    y = dy[m].astype(float)
    f = flow[m].astype(float)
    if ctrl is not None:
        x = ctrl[m].astype(float)
        xc, yc = x - x.mean(), y - y.mean()
        var = float(np.dot(xc, xc))
        if var <= 0:
            return np.nan
        y = yc - xc * (float(np.dot(xc, yc)) / var)
        if np.nanstd(y) == 0:
            return np.nan
    return float(np.corrcoef(f, y)[0, 1])


def _lead(corrs: np.ndarray, n: int) -> tuple[int, float, float, bool] | None:
    """Winner of one yield, and whether the win clears a noise correlation.

    One sector against one yield. The other sectors are the null, not extra regressors.
    Fisher z: the leader has to clear 5% on its own and lead the next sector by one
    standard error. A full sector-by-tenor matrix is not estimated.
    """
    c = np.asarray(corrs, float)
    ok = np.isfinite(c)
    if int(ok.sum()) < 2 or n <= 3:
        return None
    order = np.argsort(np.where(ok, c, -np.inf))
    i1, i2 = int(order[-1]), int(order[-2])
    r1, r2 = float(c[i1]), float(c[i2])
    if not np.isfinite(r1):
        return None
    se = 1.0 / np.sqrt(n - 3)
    z1 = float(np.arctanh(np.clip(r1, -0.999, 0.999)))
    z2 = float(np.arctanh(np.clip(r2, -0.999, 0.999))) if np.isfinite(r2) else -np.inf
    identified = bool(z1 > 1.96 * se and z1 - z2 > se)
    return i1, r1, r2, identified


def buyer_path(fl: pd.DataFrame | None = None, window: int = BUYER_WINDOW) -> pd.DataFrame:
    """Who leads each yield, quarter by quarter.

    Where the tape has no maturity, the sector's total buying is correlated with each yield.
    That is the UK grid's assignment. Where the tape records the issue type, only that bucket
    is correlated with its yield: JSDA medium with the 2y, long with the 10y, super-long with
    the 30y and with 10s30s. BoJ outright purchases sit on that same grid. JPX futures are
    a separate tape: the 10y contract against the 10y, the mini 20y against the 30y.
    `basis` level holds the rest of the tape fixed. `basis` habitat also holds the 2y fixed.
    The coefficient is the leave-one-out partial slope, one sector at a time. `stable` means
    that sector's partial residual keeps its sign in both halves of the window.
    """
    fl = flows() if fl is None else fl
    y = _yields()
    d = fl.merge(y, on=["cc", "quarter_end"], how="inner")
    d = d[d["quarter_end"] <= pd.Timestamp.today().normalize()]
    if "match_tenor" not in d.columns:
        d["match_tenor"] = ""
    d["match_tenor"] = d["match_tenor"].fillna("").astype(str)
    auction = d["tape"].eq(AUCTION)
    d["flow"] = np.where(auction, d["share"], d["flow_bn"])
    rows = []
    present = [t for t in CURVE if t in d.columns]
    for (cc, tape), sub in d.groupby(["cc", "tape"]):
        measured = sub["match_tenor"].str.len().gt(0).any()
        if measured:
            curve = sub[sub["match_tenor"].eq("dy30")].assign(match_tenor="ds1030")
            sub = pd.concat([sub, curve], ignore_index=True)
        panels, labels = {}, {}
        keys = ["sector", "match_tenor"] if measured else ["sector"]
        for key, gsec in sub.groupby(keys):
            gsec = gsec.sort_values("quarter_end").drop_duplicates("quarter_end")
            sector = key[0] if measured else key
            panels[key] = gsec.set_index("quarter_end")
            labels[sector] = gsec["label"].iloc[0]
        if measured:
            sectors_for = {t: [s for (s, mt) in panels if mt == t] for t in present}
        else:
            sectors_for = {t: list(labels) for t in present}
            if {"insurers", "pension", "icpf"} <= set(labels):
                for t in sectors_for:
                    sectors_for[t] = [s for s in sectors_for[t] if s != "icpf"]
        idx = sorted(set().union(*[p.index for p in panels.values()]))
        streak = {}
        for tenor in present:
            sectors = sectors_for.get(tenor, [])
            if len(sectors) < 2 or tenor not in present:
                continue
            for i in range(window - 1, len(idx)):
                sl = idx[i - window + 1:i + 1]
                if sl[-1] - sl[0] > pd.Timedelta(days=int(window / 4 * 400)):
                    continue
                n_here = 0
                for sector in sectors:
                    key = (sector, tenor) if measured else sector
                    panel = panels.get(key)
                    if panel is not None and sl[-1] in panel.index and np.isfinite(panel.at[sl[-1], "flow"]):
                        n_here += 1
                if n_here < 2:
                    continue
                flows_m, yield_row = [], None
                for sector in sectors:
                    panel_key = (sector, tenor) if measured else sector
                    panel = panels.get(panel_key)
                    if panel is None:
                        flows_m.append(np.full(len(sl), np.nan))
                        continue
                    w = panel.reindex(sl)
                    flows_m.append(w["flow"].to_numpy(float))
                    if yield_row is None:
                        yield_row = w
                if yield_row is None or tenor not in yield_row.columns:
                    continue
                F = np.column_stack(flows_m)
                dy = yield_row[tenor].to_numpy(float)
                front = "dy2" if "dy2" in yield_row.columns else "dy3"
                front_vec = yield_row[front].to_numpy(float) if front in yield_row.columns else None
                fitted = {}
                for j, sector in enumerate(sectors):
                    oth = np.delete(F, j, axis=1)
                    others = np.where(np.any(np.isfinite(oth), axis=1), np.nansum(oth, axis=1), np.nan)
                    level = _fwl(dy, F[:, j], others)
                    if front_vec is not None and tenor != front:
                        hab = _fwl(dy, F[:, j], np.column_stack([others, front_vec]))
                    else:
                        hab = (np.nan,) * 6
                    fitted[sector] = {"level": level, "habitat": hab}
                for basis in ("level", "habitat"):
                    tstats = np.array([fitted[s][basis][1] for s in sectors], float)
                    pcorrs = np.array([fitted[s][basis][2] for s in sectors], float)
                    dfs = np.array([fitted[s][basis][3] for s in sectors], float)
                    won = _lead_t(tstats, dfs)
                    if won is None:
                        continue
                    i1, t1, t2, identified = won
                    sector = sectors[i1]
                    order = np.argsort(np.where(np.isfinite(tstats), tstats, -np.inf))
                    key = (tenor, basis)
                    prev, nprev = streak.get(key, (None, 0))
                    nprev = nprev + 1 if sector == prev else 1
                    streak[key] = (sector, nprev)
                    _b, _t, _p, _df, rx, ry = fitted[sector][basis]
                    mid = len(sl) // 2
                    h1 = float(np.nansum(rx[:mid] * ry[:mid]))
                    h2 = float(np.nansum(rx[mid:] * ry[mid:]))
                    rows.append({
                        "cc": cc, "tape": tape, "tenor": tenor, "tenor_name": CURVE_NAME[tenor],
                        "basis": basis, "quarter_end": sl[-1], "sector": sector, "label": labels[sector],
                        "second_label": labels[sectors[int(order[-2])]],
                        "corr": float(pcorrs[i1]), "second": float(pcorrs[int(order[-2])]),
                        "t": t1, "identified": identified,
                        "stable": _same_sign(h1, h2, t1), "lead_q": nprev,
                        "n": window, "measured": measured,
                    })
    return pd.DataFrame(rows)


def _ols_t(y: np.ndarray, x: np.ndarray, min_n: int = 8) -> tuple[float, float]:
    m = np.isfinite(y) & np.isfinite(x)
    n = int(m.sum())
    if n < min_n or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return np.nan, np.nan
    xc = x[m] - np.mean(x[m])
    yc = y[m] - np.mean(y[m])
    ssx = float(np.dot(xc, xc))
    if ssx <= 0:
        return np.nan, np.nan
    beta = float(np.dot(xc, yc) / ssx)
    err = yc - beta * xc
    df = n - 2
    se = float(np.sqrt((np.dot(err, err) / df) / ssx))
    return beta, (beta / se if se > 0 else np.nan)


def _size_weighted_shock(F: np.ndarray):
    """Leave-one-out residuals, weighted by mean absolute flow before the last quarter."""
    eps = np.full_like(F, np.nan)
    for j in range(F.shape[1]):
        oth = np.delete(F, j, axis=1)
        others = np.where(np.any(np.isfinite(oth), axis=1), np.nansum(oth, axis=1), np.nan)
        *_, rx, _ry = _fwl(F[:, j], F[:, j], others)
        eps[:, j] = rx
    hist = np.abs(F[:-1])
    size = np.array([
        float(np.nanmean(hist[:, j])) if np.isfinite(hist[:, j]).any() else 0.0
        for j in range(hist.shape[1])
    ])
    if float(size.sum()) <= 0:
        return None
    wts = size / size.sum()
    zw = np.where(np.isfinite(eps), eps * wts, np.nan)
    z = np.nansum(zw, axis=1)
    z = np.where(np.any(np.isfinite(zw), axis=1), z, np.nan)
    return z, wts * eps[-1]


def granular(fl: pd.DataFrame | None = None, window: int = BUYER_WINDOW) -> pd.DataFrame:
    """Rolling granular demand against supply, one slope per yield.

    A tape with no maturity split has one shock, and that shock is regressed on each yield
    change: 2y, 3y where the curve has it, 10y, 30y, 10s30s. A tape that records the issue
    type is estimated inside the bucket, and the super-long bucket is also regressed on
    10s30s. Supply is that bucket's issuance when the tape has it, otherwise the tape total.
    The named sector is the largest absolute contribution to the last quarter's shock.
    """
    fl = _cash(flows() if fl is None else fl)
    yld = _yields()
    g = _gdp().sort_values(["cc", "quarter_end"])
    g["gdp"] = g.groupby("cc")["gdp"].ffill()
    if "match_tenor" not in fl.columns:
        fl = fl.copy()
        fl["match_tenor"] = ""
    fl = fl.copy()
    fl["match_tenor"] = fl["match_tenor"].fillna("").astype(str)
    long = fl[fl["match_tenor"].eq("dy30")].assign(match_tenor="ds1030")
    fl = pd.concat([fl, long], ignore_index=True)
    agg = fl.groupby(
        ["cc", "tape", "quarter_end", "sector", "label", "match_tenor"], as_index=False
    ).agg(flow_bn=("flow_bn", "sum"), supply_bn=("supply_bn", "max"), share=("share", "mean"))
    agg = agg.merge(g[["cc", "quarter_end", "gdp"]], on=["cc", "quarter_end"], how="left")
    agg["gdp"] = agg.groupby(["cc", "sector"])["gdp"].ffill()
    agg = agg[agg["quarter_end"] <= pd.Timestamp.today().normalize()]
    agg["flow_pct"] = np.where(agg["gdp"] > 0, 100.0 * agg["flow_bn"] / agg["gdp"], np.nan)
    agg["supply_pct"] = np.where(agg["gdp"] > 0, 100.0 * agg["supply_bn"] / agg["gdp"], np.nan)
    out = []
    for (cc, tape), sub in agg.groupby(["cc", "tape"]):
        yc = yld[yld["cc"] == cc].set_index("quarter_end")
        measured = sub["match_tenor"].str.len().gt(0).any()
        buckets = [t for t in CURVE if t in set(sub["match_tenor"])] if measured else [""]
        for bucket in buckets:
            block = sub if bucket == "" else sub[sub["match_tenor"].eq(bucket)]
            labels = dict(block.groupby("sector")["label"].first())
            sectors = [s for s in labels if block.loc[block["sector"].eq(s), "flow_pct"].notna().sum() >= window]
            if len(sectors) < 2:
                continue
            wide = (block[block["sector"].isin(sectors)]
                    .pivot_table(index="quarter_end", columns="sector", values="flow_pct", aggfunc="sum")
                    .reindex(columns=sectors).sort_index())
            supply = block.groupby("quarter_end")["supply_pct"].max()
            tenors = [bucket] if bucket else [t for t in CURVE if t in yc.columns and yc[t].notna().any()]
            if not tenors:
                tenors = ["dy10"]
            idx = list(wide.index)
            for i in range(window - 1, len(idx)):
                sl = idx[i - window + 1:i + 1]
                if sl[-1] - sl[0] > pd.Timedelta(days=int(window / 4 * 400)):
                    continue
                shock = _size_weighted_shock(wide.loc[sl, sectors].to_numpy(float))
                if shock is None:
                    continue
                z, contrib = shock
                if not np.isfinite(contrib).any():
                    continue
                j1 = int(np.nanargmax(np.abs(np.where(np.isfinite(contrib), contrib, np.nan))))
                for tenor in tenors:
                    series = (yc[tenor].reindex(sl).to_numpy(float)
                              if tenor in yc.columns else np.full(len(sl), np.nan))
                    beta, tstat = _ols_t(series, z)
                    dy = float(yc[tenor].get(sl[-1], np.nan)) if tenor in yc.columns else np.nan
                    out.append({
                        "cc": cc, "tape": tape, "tenor": tenor, "tenor_name": CURVE_NAME[tenor],
                        "quarter_end": sl[-1],
                        "supply_pct": float(supply.get(sl[-1], np.nan)),
                        "z_pct": float(z[-1]) if np.isfinite(z[-1]) else np.nan,
                        "dy": dy,
                        "beta": beta, "t": tstat,
                        "sector": sectors[j1], "label": labels[sectors[j1]],
                        "contrib": float(contrib[j1]) if np.isfinite(contrib[j1]) else np.nan,
                    })
    return pd.DataFrame(out)
