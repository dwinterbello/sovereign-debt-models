"""Who has to own each tenor. Not a score.

A sleeve is a habitat. The buyer is whoever is structurally stuck with that
maturity, and the number is their own flow, never an average with the other
pillars.

  bills   under 1y     foreign official, money funds, JSDA foreigners
  int     1y to 10y    banks (Japan), funds and foreign private (US auctions)
  long    10y to 20y   pensions and trust banks
  ultra   over 20y     insurers

Supply and official DV01 are tenor-true for all six countries (float book).
Private habitat is tenor-true only where a tape splits it: JSDA by investor
and maturity, US auction allotments by class and term. Euro-area holder
statistics do not split 10y from 30y, so DE/FR/IT private cells stay empty.
UK auction cover by residual maturity is an auction fact, not a buyer.

Positive flow = that buyer added DV01. Official is signed the same way
(purchases positive, QT negative). Private supply is paper the market had to
take. They are shown side by side. They are not combined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from issuance.fetch_jsda import DURATION, BANKS, REAL_MONEY
from issuance.paths import CACHE

SLEEVES = ("bills", "int", "long", "ultra")
# Residual-maturity buckets on the float book.
FLOAT_SLEEVE = {
    "0-1y": "bills",
    "1-3y": "int", "3-7y": "int", "7-10y": "int",
    "10-15y": "long", "15-20y": "long",
    "20-30y": "ultra", "30y+": "ultra",
}
JSDA_SLEEVE = {"tbills": "bills", "medium": "int", "long": "long", "super_long": "ultra", "zero": "long"}
JP_FOREIGN = ("foreigners",)
JP_DEALER = ("dealers", "others")
# Norinchukin-type accounts sit with real money, not with city banks.
JP_REAL = REAL_MONEY + ("agri_fin",)
US_LEG = {
    "depository": "bank",
    "inv_funds": "fund",
    "foreign": "foreign",
    "pension_ins": "pension",
    "dealers": "dealer",
}
# Which leg is the habitat buyer of that sleeve. The others are still published.
HABITAT_LEG = {
    "bills": "foreign",
    "int": "bank",
    "long": "pension",
    "ultra": "insurer",
}
# US int habitat is funds, not banks. Override per country where the buyer differs.
# US bills are a money-fund habitat. Auction "foreign" on bills is not that buyer, so the
# cell stays empty rather than borrowing a primary-market class.
HABITAT_LEG_CC = {("US", "bills"): "",
                  ("US", "int"): "fund", ("US", "long"): "pension", ("US", "ultra"): "pension",
                  ("JP", "int"): "bank", ("JP", "long"): "pension", ("JP", "ultra"): "insurer",
                  ("JP", "bills"): "foreign"}


def _z(s: pd.Series) -> pd.Series:
    mu = s.rolling(36, min_periods=24).mean()
    sd = s.rolling(36, min_periods=24).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-4, 4)


def _month(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s.groupby(level=0).sum()


def _rows(country: str, sleeve: str, leg: str, flow: pd.Series, source: str) -> pd.DataFrame:
    flow = flow.dropna().sort_index()
    if flow.empty:
        return pd.DataFrame()
    z = _z(flow.rolling(3, min_periods=3).sum())
    df = pd.DataFrame({"flow": flow, "z": z}).dropna(subset=["z"]).reset_index()
    if "month" not in df.columns:
        df = df.rename(columns={df.columns[0]: "month"})
    return df.assign(country=country, sleeve=sleeve, leg=leg, source=source)


def _jp() -> pd.DataFrame:
    path = CACHE / "jsda_investor.parquet"
    if not path.exists():
        return pd.DataFrame()
    js = pd.read_parquet(path)
    js = js[js["investor"].astype(str).ne("total")].copy()
    js["month"] = pd.to_datetime(js["month"]).dt.to_period("M").dt.to_timestamp()
    js["sleeve"] = js["maturity"].map(JSDA_SLEEVE)
    js = js.dropna(subset=["sleeve"])
    dur = js["maturity"].map(DURATION).astype(float)
    # ¥bn face × years × 1e-4 = ¥bn per bp.
    js["dv01"] = js["net_purchase_jpy_bn"] * dur * 1e-4
    legs = {"bank": BANKS, "real": JP_REAL, "foreign": JP_FOREIGN, "dealer": JP_DEALER,
            "insurer": ("insurers",), "pension": ("trust_banks", "mutual_aid")}
    out = []
    for sleeve, sub in js.groupby("sleeve"):
        for leg, names in legs.items():
            s = sub[sub["investor"].isin(names)].groupby("month")["dv01"].sum()
            out.append(_rows("JP", sleeve, leg, s, "JSDA OTC outright, face × fixed duration"))
    return pd.concat([p for p in out if len(p)], ignore_index=True) if out else pd.DataFrame()


def _us() -> pd.DataFrame:
    path = CACHE / "us_allotments.parquet"
    if not path.exists():
        return pd.DataFrame()
    al = pd.read_parquet(path)
    al["term_years"] = pd.to_numeric(al["term_years"], errors="coerce")
    al = al[al["term_years"].notna() & (al["amount_bn"].notna())].copy()
    al["sleeve"] = np.where(al["term_years"] <= 1, "bills",
                            np.where(al["term_years"] <= 10, "int",
                                     np.where(al["term_years"] <= 20, "long", "ultra")))
    al["month"] = pd.to_datetime(al["issue_date"]).dt.to_period("M").dt.to_timestamp()
    al["leg"] = al["investor_class"].map(US_LEG)
    al = al.dropna(subset=["leg"])
    out = []
    for (sleeve, leg), sub in al.groupby(["sleeve", "leg"]):
        s = sub.groupby("month")["amount_bn"].sum()
        out.append(_rows("US", sleeve, leg, s, "Treasury auction allotment, $bn face, primary only"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _book() -> pd.DataFrame:
    """Private supply and official DV01 by sleeve, every country. USD mn per bp."""
    path = CACHE / "float_share.parquet"
    if not path.exists():
        return pd.DataFrame()
    f = pd.read_parquet(path)
    f["month"] = pd.to_datetime(f["month"]).dt.to_period("M").dt.to_timestamp()
    f["sleeve"] = f["tenor_bucket"].astype(str).map(FLOAT_SLEEVE)
    f = f.dropna(subset=["sleeve"])
    g = f.groupby(["country", "month", "sleeve"], as_index=False)[["flow_dv01", "official_dv01"]].sum()
    out = []
    for (c, sleeve), sub in g.groupby(["country", "sleeve"]):
        sub = sub.set_index("month").sort_index()
        out.append(_rows(c, sleeve, "supply", sub["flow_dv01"], "private coupon DV01 issued"))
        # official_dv01 on the float book is a stock. The flow is the change.
        out.append(_rows(c, sleeve, "official", sub["official_dv01"].diff(), "official DV01 change"))
    return pd.concat([p for p in out if p is not None and len(p)], ignore_index=True)


def _uk_auction() -> pd.DataFrame:
    path = CACHE / "rv_demand.parquet"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(path)
    d = d[d["country"].eq("UK")].copy()
    d["month"] = pd.to_datetime(d["month"])
    out = []
    for col, sleeve in (("btc_int", "int"), ("btc_long", "long")):
        if col not in d.columns:
            continue
        s = d.set_index("month")[col].dropna()
        # Cover is already a level; z it on the 3m mean inside _rows via the rolling sum,
        # which is the wrong operator for a ratio. Z the monthly cover directly.
        if s.empty:
            continue
        z = _z(s)
        df = pd.DataFrame({"month": z.index, "flow": s.reindex(z.index), "z": z}).dropna(subset=["z"])
        out.append(df.assign(country="UK", sleeve=sleeve, leg="auction",
                             source="DMO cover, residual maturity"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build() -> pd.DataFrame:
    parts = [p for p in (_jp(), _us(), _book(), _uk_auction()) if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["habitat"] = [
        HABITAT_LEG_CC.get((c, s), HABITAT_LEG.get(s)) == leg
        for c, s, leg in zip(out["country"], out["sleeve"], out["leg"])
    ]
    return out.sort_values(["country", "sleeve", "leg", "month"]).reset_index(drop=True)


# Sleeve yield the buyer would be responding to. 2y stands for the intermediate, 10y for
# the 10-20y issue, 30y for the ultra, the 3-month rate for bills.
SLEEVE_YIELD = {"bills": "r3m", "int": "y2", "long": "y10", "ultra": "y30"}
JP_PRICE = ("foreigners", "city_banks", "regional_banks", "insurers", "trust_banks",
            "inv_trusts", "mutual_aid", "agri_fin", "dealers")
# A null on these names is a result. A null on everyone else is omitted.
INSENSITIVE_NAMES = {"foreigners", "foreign", "insurers", "insurer", "trust_banks", "pension",
                     "mutual_aid", "agri_fin", "fund", "official", "city_banks", "regional_banks"}


def _hac_t(flow: pd.Series, dy: pd.Series, min_q: int = 16) -> tuple[float, int] | None:
    import statsmodels.api as sm

    df = pd.DataFrame({"f": flow, "dy": dy}).dropna()
    df = df[df.index.month % 3 == 0]
    if len(df) < min_q or float(df["f"].std()) == 0 or float(df["dy"].std()) == 0:
        return None
    res = sm.OLS(df["f"], sm.add_constant(df["dy"] * 100)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    t = float(res.tvalues["dy"])
    return t, int(res.nobs)


def _windows(flow: pd.Series, dy: pd.Series) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, mask in (("full", None), ("pre", flow.index <= "2018-12-01"), ("post", flow.index >= "2019-01-01")):
        f = flow if mask is None else flow[mask]
        hit = _hac_t(f, dy.reindex(f.index), 16 if name == "full" else 12)
        out[name] = None if hit is None else hit[0]
        if name == "full":
            out["n"] = None if hit is None else hit[1]
    return out


def _stance(t: float | None, pre: float | None, post: float | None) -> str:
    halves = [h for h in (pre, post) if h is not None]
    if t is not None and halves and t >= 2 and all(h >= 1.5 for h in halves):
        return "sensitive"
    if t is not None and halves and t <= -2 and all(h <= -1.5 for h in halves):
        return "rally"
    # The two regimes sit at different levels, so the pooled slope washes out.
    if pre is not None and post is not None and pre <= -2 and post <= -2 and (t is None or abs(t) < 1.5):
        return "rally"
    if pre is not None and post is not None and pre >= 2 and post >= 2 and (t is None or abs(t) < 1.5):
        return "sensitive"
    if (t is not None and abs(t) < 1.5
            and (pre is None or abs(pre) < 1.5) and (post is None or abs(post) < 1.5)):
        return "insensitive"
    return ""


def price_response() -> pd.DataFrame:
    """Who changes their buying when the sleeve's own yield has moved.

    Flow over three months on the yield change over those same three months.
    Quarterly months only, HAC(2). Sensitive = they buy after the sleeve cheapens,
    in both halves. Rally = they buy after it has already richened. Insensitive =
    no detectable response. Official is tagged policy: a rally slope there is QE timing.
    US rows are auction shares, so auction size does not look like demand.
    """
    path = CACHE / "market_monthly.parquet"
    if not path.exists():
        return pd.DataFrame()
    px = pd.read_parquet(path)
    px["month"] = pd.to_datetime(px["month"])
    wide = px.pivot_table(index=["country", "month"], columns="series", values="value")
    rows: list[dict] = []

    def add(country: str, sleeve: str, buyer: str, flow: pd.Series, kind: str) -> None:
        series = SLEEVE_YIELD[sleeve]
        try:
            y = wide.xs(country)[series].sort_index()
        except KeyError:
            return
        flow = flow.sort_index()
        flow.index = pd.to_datetime(flow.index)
        w = _windows(flow, y.diff(3))
        stance = _stance(w.get("full"), w.get("pre"), w.get("post"))
        if kind == "official":
            stance = "policy" if stance == "rally" else ""
        if stance == "insensitive" and (buyer not in INSENSITIVE_NAMES or (w.get("n") or 0) < 40):
            return
        if not stance:
            return
        rows.append({"country": country, "sleeve": sleeve, "buyer": buyer, "stance": stance,
                     "t": None if w.get("full") is None else round(float(w["full"]), 2),
                     "t_pre": None if w.get("pre") is None else round(float(w["pre"]), 2),
                     "t_post": None if w.get("post") is None else round(float(w["post"]), 2),
                     "n": w.get("n"), "kind": kind})

    js_path = CACHE / "jsda_investor.parquet"
    if js_path.exists():
        js = pd.read_parquet(js_path)
        js = js[js["investor"].astype(str).isin(JP_PRICE)].copy()
        js["month"] = pd.to_datetime(js["month"]).dt.to_period("M").dt.to_timestamp()
        js["sleeve"] = js["maturity"].map(JSDA_SLEEVE)
        js = js.dropna(subset=["sleeve"])
        js["dv01"] = js["net_purchase_jpy_bn"] * js["maturity"].map(DURATION).astype(float) * 1e-4
        for (sleeve, inv), sub in js.groupby(["sleeve", "investor"]):
            flow = sub.groupby("month")["dv01"].sum().rolling(3, min_periods=3).sum()
            add("JP", sleeve, inv, flow, "secondary")

    us_path = CACHE / "us_allotments.parquet"
    if us_path.exists():
        al = pd.read_parquet(us_path)
        al["term_years"] = pd.to_numeric(al["term_years"], errors="coerce")
        al = al[al["term_years"].notna() & al["amount_bn"].notna()].copy()
        al["sleeve"] = np.where(al["term_years"] <= 1, "bills",
                                np.where(al["term_years"] <= 10, "int",
                                         np.where(al["term_years"] <= 20, "long", "ultra")))
        al["month"] = pd.to_datetime(al["issue_date"]).dt.to_period("M").dt.to_timestamp()
        al["buyer"] = al["investor_class"].map(US_LEG)
        al = al.dropna(subset=["buyer"])
        for (sleeve, buyer), sub in al.groupby(["sleeve", "buyer"]):
            tot = al[al["sleeve"] == sleeve].groupby("month")["amount_bn"].sum()
            share = (sub.groupby("month")["amount_bn"].sum() / tot).rolling(3, min_periods=3).mean()
            add("US", sleeve, buyer, share, "primary")

    book = CACHE / "habitat.parquet"
    if book.exists():
        h = pd.read_parquet(book)
        h["month"] = pd.to_datetime(h["month"])
        off = h[h["leg"].eq("official")]
        for (country, sleeve), sub in off.groupby(["country", "sleeve"]):
            flow = sub.set_index("month")["flow"].sort_index().rolling(3, min_periods=3).sum()
            add(country, sleeve, "official", flow, "official")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["stance", "country", "sleeve", "buyer"]).reset_index(drop=True)


def write() -> pd.DataFrame:
    out = build()
    if not out.empty:
        out.to_parquet(CACHE / "habitat.parquet", index=False)
    resp = price_response()
    if not resp.empty:
        resp.to_parquet(CACHE / "habitat_price.parquet", index=False)
    return out
