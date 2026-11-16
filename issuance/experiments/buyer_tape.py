"""Gilt holder test, on each buyer tape this repo actually has.

The gilt version is `uk_flows.behaviour.holder_mix` plus `experiments.uk_holders_test`:

  liability matchers (ICPF) are the price-insensitive holder;
  other financial and overseas are the price-sensitive holder;
  the official sector is not a behavioural holder.

The measurement, identical to `uk_holders_test.co_move` and `evidence._fit`:

  co_move   correlation of the 12-month sum of net purchases with the 12-month
            change in the yield. Positive means the buyer added while the
            sleeve cheapened.
  forward   purchases × −1 (the gilt purchase sign: more buying is less
            pressure), expanding percentile, yield change from t+1 to t+4,
            trailing-12m change as the control. Verdict is `evidence.verdict`.

Roles are assigned the gilt way, before the regression. A liability matcher
who buys a cheapening is still a liability matcher; ICPF itself co-moves
+0.34 with the 10y. The forward test asks whether their buying sets the
next yield. On gilts, OFI does (10y forward link) and ICPF does not.

Tapes scored here:

  JP   JSDA outright purchases by investor and maturity.
  US   Z.1 F.210 transactions. Bills versus other Treasuries where published.
       No 10y versus 30y split. Auction allotments, Fed custody, H.8 and
       retail money funds stay in `us_demand_evidence`.
  EA   SHSS transactions, holder by original maturity (bills versus bonds),
       from 2021. Too short for the forward fit; the co-move is reported.
  UK   BoE MFI (ex central bank) gilt transactions by residual maturity.
       Fifteen buckets from 2016; the 2009–2015 form is three buckets and is
       scored on its own. MQ5 insurer, pension and trust transactions through
       2018. DMO GEMM counterparty codes are still unpublished. FSPS pension
       stocks are levels, not transactions, and are not scored.
  US   Primary-dealer net outright positions (FR 2004), week-on-week change
       by residual maturity from 2013. A position change, not a settlement
       tape. The 11y+ bucket is one series until 2022 and then 11–21y and 21y+.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import evidence as ev
from issuance.fetch_jsda import DURATION
from issuance.habitat import JSDA_SLEEVE, SLEEVE_YIELD
from issuance.paths import CACHE, REPO

OUT = CACHE / "buyer_tape.parquet"
# F.210 Treasury asset, plus the bill / coupon split where the Fed publishes one.
# Private pensions are the parent line. State-and-local pensions are the defined-benefit line.
Z1_LINES = (
    ("life_insurance", "all", "FU543061105.Q", "liability matcher"),
    ("life_insurance", "bills", "FU543061115.Q", "liability matcher"),
    ("life_insurance", "coupon", "FU543061125.Q", "liability matcher"),
    ("pc_insurance", "all", "FU513061105.Q", "liability matcher"),
    ("pc_insurance", "bills", "FU513061115.Q", "liability matcher"),
    ("pc_insurance", "coupon", "FU513061125.Q", "liability matcher"),
    ("private_pension_funds", "all", "FU573061105.Q", "liability matcher"),
    ("slg_pension_funds", "all", "FU223061143.Q", "liability matcher"),
    ("money_market_funds", "all", "FU633061105.Q", "elastic"),
    ("money_market_funds", "bills", "FU633061110.Q", "elastic"),
    ("money_market_funds", "coupon", "FU633061120.Q", "elastic"),
    ("mutual_funds", "all", "FU653061105.Q", "elastic"),
    ("mutual_funds", "bills", "FU653061113.Q", "elastic"),
    ("mutual_funds", "coupon", "FU653061125.Q", "elastic"),
    ("etfs", "all", "FU563061103.Q", "elastic"),
    ("hedge_funds", "all", "FU623061103.Q", "elastic"),
    ("broker_dealers", "all", "FU663061105.Q", "elastic"),
    ("households", "all", "FU153061105.Q", "elastic"),
    ("rest_of_world", "all", "FU263061105.Q", "elastic"),
    ("rest_of_world", "bills", "FU263061110.Q", "elastic"),
    ("rest_of_world", "coupon", "FU263061120.Q", "elastic"),
    ("us_banks", "all", "FU763061100.Q", "elastic"),
    ("federal_reserve", "all", "FU713061103.Q", "official"),
    ("federal_reserve", "bills", "FU713061113.Q", "official"),
    ("federal_reserve", "coupon", "FU713061125.Q", "official"),
)
Z1_YIELD = {"bills": "r3m", "coupon": "y10", "all": "y10"}
# Gilt roles. Official is excluded from the behavioural split.
ROLE = {
    "insurers": "liability matcher",
    "trust_banks": "liability matcher",
    "mutual_aid": "liability matcher",
    "agri_fin": "liability matcher",
    "city_banks": "elastic",
    "regional_banks": "elastic",
    "foreigners": "elastic",
    "dealers": "elastic",
    "inv_trusts": "elastic",
}
LAG = 1


def _yields() -> pd.DataFrame:
    px = pd.read_parquet(CACHE / "market_monthly.parquet")
    px["month"] = pd.to_datetime(px["month"])
    return px.pivot_table(index=["country", "month"], columns="series", values="value")


def _jsda() -> pd.DataFrame:
    js = pd.read_parquet(CACHE / "jsda_investor.parquet")
    js = js[js["investor"].astype(str).isin(ROLE)].copy()
    js["month"] = pd.to_datetime(js["month"]).dt.to_period("M").dt.to_timestamp()
    js["sleeve"] = js["maturity"].map(JSDA_SLEEVE)
    js = js.dropna(subset=["sleeve"])
    js["dv01"] = js["net_purchase_jpy_bn"] * js["maturity"].map(DURATION).astype(float) * 1e-4
    return js.groupby(["sleeve", "investor", "month"], as_index=False)["dv01"].sum()


def _flow12(s: pd.Series) -> pd.Series:
    return s.sort_index().asfreq("MS").rolling(12, min_periods=12).sum()


def _co(flow: pd.Series, dy: pd.Series, start: str, end: str | None) -> tuple[float, int]:
    j = pd.DataFrame({"f": flow, "dy": dy}).dropna()
    j = j[j.index >= start]
    if end:
        j = j[j.index <= end]
    if len(j) < 16 or j["f"].std() == 0 or j["dy"].std() == 0:
        return np.nan, len(j)
    return float(j["f"].corr(j["dy"])), len(j)


def _quarter_end(flow: pd.Series) -> pd.Series:
    """F.210 quarter-start stamp → the quarter-end month, summed over four quarters."""
    s = flow.sort_index()
    s.index = (pd.to_datetime(s.index) + pd.offsets.QuarterEnd(0)).to_period("M").to_timestamp()
    s = s.groupby(level=0).sum()
    return s.rolling(4, min_periods=4).sum()


def _forward(flow: pd.Series, yld: pd.Series, country: str) -> dict:
    """evidence._fit with the gilt purchase sign. y is the yield, in percent."""
    v = (-flow).asfreq("MS")
    x = ev.pressure_pct(v)
    yld = yld.sort_index().asfreq("MS")
    y = yld.shift(-(LAG + 3)) - yld.shift(-LAG)
    y12 = yld - yld.shift(12)
    df = pd.DataFrame({"month": x.index, "x": x.to_numpy(), "y": y.reindex(x.index).to_numpy(),
                       "y12": y12.reindex(x.index).to_numpy(), "country": country, "lag": LAG}).dropna()
    got = ev._fit(df, 3)
    if not got:
        return {"verdict": "untested", "n_obs": len(df)}
    got["verdict"] = ev.verdict(got["beta_bp"], got["t"], got["loo_min_t"], got["p_chance"],
                                 got["loo_max_t"], got["oos_r2"])
    return got


def _pack(country: str, sleeve: str, buyer: str, role: str, tape: str, yname: str,
          flow: pd.Series, yld: pd.Series, y10: pd.Series, forward) -> dict:
    gap = flow.dropna().index.to_series().diff().dt.days.median()
    k = 4 if gap and gap > 60 else 12
    rec = {"country": country, "sleeve": sleeve, "buyer": buyer, "role": role,
           "tape": tape, "yield": yname}
    for win, start, end in (("2010+", "2010-01-01", None),
                            ("2010–18", "2010-01-01", "2018-12-01"),
                            ("2019+", "2019-01-01", None)):
        r, n = _co(flow, yld.diff(k), start, end)
        rec[f"r_{win}"] = None if r != r else round(r, 2)
        rec[f"n_{win}"] = n
    fwd = forward(flow, yld, country)
    rec["verdict"] = fwd.get("verdict", "untested")
    rec["t"] = None if fwd.get("t") is None else round(float(fwd["t"]), 2)
    rec["beta_bp"] = None if fwd.get("beta_bp") is None else round(float(fwd["beta_bp"]), 2)
    r10, _ = _co(flow, y10.diff(k), "2010-01-01", None)
    rec["r_y10"] = None if r10 != r10 else round(r10, 2)
    return rec


def _z1(wide: pd.DataFrame) -> list[dict]:
    path = CACHE / "us_z1_sectors.parquet"
    if not path.exists():
        return []
    z = pd.read_parquet(path)
    z = z[z["table"].eq("F")].copy()
    z["quarter"] = pd.to_datetime(z["quarter"])
    us = wide.xs("US")
    y10 = us["y10"].sort_index()
    # Mutual-fund bill and coupon lines are the whole book before 2010:Q4.
    late = {"FU653061113.Q", "FU653061125.Q"}
    out = []
    for buyer, sleeve, code, role in Z1_LINES:
        sub = z[z["series_code"].eq(code)]
        if sub.empty:
            continue
        if code in late:
            sub = sub[sub["quarter"] >= "2010-10-01"]
        q = sub.set_index("quarter")["value_usd_bn"]
        flow = _quarter_end(q)
        yname = Z1_YIELD[sleeve]
        yld = us[yname].sort_index()
        # Quarterly observations only, so the percentile is not filled with empty months.
        yld_q = yld.copy()
        yld_q.index = pd.to_datetime(yld_q.index).to_period("M").to_timestamp()
        yld_q = yld_q.reindex(flow.index)
        y10_q = y10.copy()
        y10_q.index = pd.to_datetime(y10_q.index).to_period("M").to_timestamp()
        y10_q = y10_q.reindex(flow.index)
        out.append(_pack("US", sleeve, buyer, role, "Z.1 F.210 transactions", yname,
                         flow, yld_q, y10_q, _forward_q))
    return out


def _forward_q(flow: pd.Series, yld: pd.Series, country: str) -> dict:
    """Same fit as `_forward`, on quarter-end months. Lag is one quarter."""
    v = -flow.sort_index()
    x = ev.pressure_pct(v)
    yld = yld.sort_index()
    y = yld.shift(-2) - yld.shift(-1)
    y12 = yld - yld.shift(4)
    df = pd.DataFrame({"month": x.index, "x": x.to_numpy(), "y": y.reindex(x.index).to_numpy(),
                       "y12": y12.reindex(x.index).to_numpy(), "country": country, "lag": LAG}).dropna()
    got = ev._fit(df, 3)
    if not got:
        return {"verdict": "untested", "n_obs": len(df)}
    got["verdict"] = ev.verdict(got["beta_bp"], got["t"], got["loo_min_t"], got["p_chance"],
                                 got["loo_max_t"], got["oos_r2"])
    return got


EA_ROLE = {
    "insurers": "liability matcher", "pension": "liability matcher",
    "banks": "elastic", "mmf": "elastic", "funds": "elastic", "fvc": "elastic",
    "households": "elastic", "nfc": "elastic",
}
EA_MATURITY = {"S": "bills", "L": "coupon", "T": "all", "LS": "near"}
EA_YIELD = {"bills": "r3m", "coupon": "y10", "all": "y10", "near": "r3m"}
UK_BANK_SLEEVE = {
    "lt_1y": "bills",
    "1-2y": "int", "2-3y": "int", "3-4y": "int", "4-5y": "int", "5-6y": "int",
    "6-7y": "int", "7-8y": "int", "8-9y": "int", "9-10y": "int",
    "10-15y": "long", "15-20y": "long",
    "20-25y": "ultra", "25-30y": "ultra", "30y+": "ultra",
}
UK_MQ5_SLEEVE = {"0-15y": "int", "0-5y": "int", "5-15y": "long", "over_15y_undated": "ultra"}
UK_MQ5_ROLE = {
    "mq5_insurer_long_term": "liability matcher",
    "mq5_insurer_general": "liability matcher",
    "mq5_pension_self_administered": "liability matcher",
    "mq5_investment_trust": "elastic",
    "mq5_unit_trust": "elastic",
}


def _month(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp()
    return out.groupby(level=0).sum().sort_index()


def _ea(wide: pd.DataFrame) -> list[dict]:
    path = CACHE / "ecb_holders" / "shss_sector_maturity.parquet"
    if not path.exists():
        return []
    from issuance.ecb_holders import SHSS_MAP
    e = pd.read_parquet(path)
    e = e[e["sto"].eq("F") & e["maturity"].isin(EA_MATURITY)].copy()
    e["buyer"] = e["sector"].map(lambda s: "fvc" if s == "S125A" else SHSS_MAP.get(s))
    e = e[e["buyer"].isin(EA_ROLE)]
    e["quarter"] = pd.PeriodIndex(e["quarter"], freq="Q").to_timestamp()
    out = []
    for (country, buyer, mat), sub in e.groupby(["country", "buyer", "maturity"]):
        if country not in wide.index.get_level_values(0):
            continue
        sleeve = EA_MATURITY[mat]
        flow = _quarter_end(sub.set_index("quarter")["value"])
        mkt = wide.xs(country)
        yname = EA_YIELD[sleeve]
        if yname not in mkt:
            continue
        yld = _month(mkt[yname]).reindex(flow.index)
        y10 = _month(mkt["y10"]).reindex(flow.index)
        out.append(_pack(country, sleeve, buyer, EA_ROLE[buyer],
                         "ECB SHSS transactions, original maturity", yname,
                         flow, yld, y10, _forward_q))
    return out


def _uk(wide: pd.DataFrame) -> list[dict]:
    path = CACHE / "uk_holders_tenor.parquet"
    if not path.exists() or "UK" not in wide.index.get_level_values(0):
        return []
    u = pd.read_parquet(path)
    u = u[u["measure"].eq("flow")].copy()
    u["date"] = pd.to_datetime(u["date"])
    mkt = wide.xs("UK")
    out = []
    banks = {
        "uk_banks": ("boe_mfi_ex_cb_long", "boe_mfi_ex_cb_short"),
        "uk_banks_il": ("boe_mfi_ex_cb_il_long", "boe_mfi_ex_cb_il_short"),
    }
    fine = u[u["sector"].isin(sum(banks.values(), ())) & u["maturity_bucket"].isin(UK_BANK_SLEEVE)
             & u["date"].ge("2016-02-01")]
    for buyer, sectors in banks.items():
        sub = fine[fine["sector"].isin(sectors)].copy()
        sub["sleeve"] = sub["maturity_bucket"].map(UK_BANK_SLEEVE)
        for sleeve, g in sub.groupby("sleeve"):
            flow = _flow12(_month(g.set_index("date")["value"]))
            yname = SLEEVE_YIELD[sleeve]
            yld = _month(mkt[yname])
            y10 = _month(mkt["y10"])
            out.append(_pack("UK", sleeve, buyer, "elastic",
                             "BoE MFI ex-CB gilt transactions", yname,
                             flow, yld, y10, _forward))
    mq = u[u["sector"].isin(UK_MQ5_ROLE) & u["maturity_bucket"].isin(UK_MQ5_SLEEVE)]
    for (buyer, bucket), g in mq.groupby(["sector", "maturity_bucket"]):
        sleeve = UK_MQ5_SLEEVE[bucket]
        flow = _quarter_end(_month(g.set_index("date")["value"]))
        yname = SLEEVE_YIELD[sleeve]
        yld = _month(mkt[yname]).reindex(flow.index)
        y10 = _month(mkt["y10"]).reindex(flow.index)
        out.append(_pack("UK", sleeve, buyer, UK_MQ5_ROLE[buyer],
                         "ONS MQ5 transactions, through 2018", yname,
                         flow, yld, y10, _forward_q))
    coarse_map = {"lt_5y": ("lt_5y", "y2"), "5-25y": ("5-25y", "y10"), "25y+": ("25y+", "y30")}
    coarse = u[u["sector"].isin(banks["uk_banks"]) & u["maturity_bucket"].isin(coarse_map)
               & u["date"].lt("2016-02-01")]
    for bucket, g in coarse.groupby("maturity_bucket"):
        sleeve, yname = coarse_map[bucket]
        flow = _flow12(_month(g.set_index("date")["value"]))
        yld = _month(mkt[yname])
        y10 = _month(mkt["y10"])
        out.append(_pack("UK", sleeve, "uk_banks", "elastic",
                         "BoE MFI ex-CB, three buckets through 2015", yname,
                         flow, yld, y10, _forward))
    return out


DEALER_INT = ("PDPOSGSC-L2", "PDPOSGSC-G2L3", "PDPOSGSC-G3L6", "PDPOSGSC-G6L7", "PDPOSGSC-G7L11")
DEALER_TIPS_INT = ("PDPOSTIPS-L2", "PDPOSTIPS-G2", "PDPOSTIPS-G6L11")


def _dealer_month(levels: pd.DataFrame) -> pd.Series:
    """12-month change in the net position. Index is the publication month."""
    last = levels.sort_index().groupby(levels.index.to_period("M")).last()
    last.index = last.index.to_timestamp()
    return last.diff(12)


def _dealers(wide: pd.DataFrame) -> list[dict]:
    path = REPO / "flow" / "output" / "nyfed.parquet"
    if not path.exists() or "US" not in wide.index.get_level_values(0):
        return []
    n = pd.read_parquet(path, columns=["date", "keyid", "value"])
    n["date"] = pd.to_datetime(n["date"])
    w = n.pivot_table(index="date", columns="keyid", values="value", aggfunc="last").sort_index()
    us = wide.xs("US")
    y10 = _month(us["y10"])
    books: dict[str, dict[str, pd.Series]] = {"us_dealers": {}, "us_dealers_tips": {}}
    if "PDPOSGS-B" in w:
        books["us_dealers"]["bills"] = w["PDPOSGS-B"]
    have = [k for k in DEALER_INT if k in w]
    if len(have) == len(DEALER_INT):
        books["us_dealers"]["0-11y"] = w[list(have)].sum(axis=1, min_count=len(have))
    if "PDPOSGSC-G11" in w:
        books["us_dealers"]["11y+"] = w["PDPOSGSC-G11"].dropna()
    if {"PDPOSGSC-G11L21", "PDPOSGSC-G21"} <= set(w.columns):
        books["us_dealers"]["11-21y"] = w["PDPOSGSC-G11L21"].dropna()
        books["us_dealers"]["21y+"] = w["PDPOSGSC-G21"].dropna()
    tips = [k for k in DEALER_TIPS_INT if k in w]
    if len(tips) == len(DEALER_TIPS_INT):
        books["us_dealers_tips"]["0-11y"] = w[list(tips)].sum(axis=1, min_count=len(tips))
    if "PDPOSTIPS-G11" in w:
        books["us_dealers_tips"]["11y+"] = w["PDPOSTIPS-G11"]
    yname_of = {"bills": "r3m", "0-11y": "y2", "11y+": "y30", "11-21y": "y10", "21y+": "y30"}
    out = []
    for buyer, sleeves in books.items():
        for sleeve, level in sleeves.items():
            flow = _dealer_month(level.dropna())
            yname = yname_of[sleeve]
            out.append(_pack("US", sleeve, buyer, "elastic",
                             "FR 2004 net outright, 12m position change", yname,
                             flow, _month(us[yname]), y10, _forward))
    return out


def build() -> pd.DataFrame:
    wide = _yields()
    rows = []
    js = _jsda()
    for (sleeve, buyer), sub in js.groupby(["sleeve", "investor"]):
        flow = _flow12(sub.set_index("month")["dv01"])
        yname = SLEEVE_YIELD[sleeve]
        yld = wide.xs("JP")[yname].sort_index()
        y10 = wide.xs("JP")["y10"].sort_index()
        rows.append(_pack("JP", sleeve, buyer, ROLE[buyer], "JSDA outright, valuation-free",
                          yname, flow, yld, y10, _forward))
    rows.extend(_z1(wide))
    rows.extend(_ea(wide))
    rows.extend(_uk(wide))
    rows.extend(_dealers(wide))
    return pd.DataFrame(rows).sort_values(["country", "role", "sleeve", "buyer"]).reset_index(drop=True)


def write() -> pd.DataFrame:
    out = build()
    if not out.empty:
        out.to_parquet(OUT, index=False)
    return out
