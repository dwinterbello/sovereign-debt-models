"""Market prices beside the supply data: monthly-average 2y/10y/30y par yields per country, plus
the relative measures each country trades on. Complete calendar months only.

    PYTHONPATH=. .venv/bin/python -m issuance.market

Sources, freshest first; a source that fails keeps the cached series:
  US  FRED DGS2/DGS10/DGS30 (Treasury par yields, daily)
  DE  Bundesbank par curve via correlations/g4 cache (daily); FRED OECD long rate as fallback
  UK  par yields computed from the BoE fitted nominal spot curve (continuously compounded zeros on
      a 0.5y grid; reproduces the BoE's published 10y/20y par within a few bp), daily
  JP  MoF JGB benchmark yields (daily, 1974 onward)
  FR  Banque de France TEC 2/10/20/30 (constant-maturity OAT, daily from 2004-11; 30y from 2006-10;
      needs a free Webstat API key, `BDF_WEBSTAT_API_KEY`); days after it, or every day without
      a key, from CNBC closes (about two years back), labelled in `source`
  IT  Banca d'Italia BMK0200 MTS benchmark BTP 3/10/30y (daily from 1989-93, refreshed with the
      monthly bulletin so about a month behind); days after its last observation from CNBC closes,
      labelled in `source`. No 2y: the CTZ benchmark ended Aug-2021, so Italy's short leg is the 3y.
  FR, IT  OECD long-term (10y) rate via FRED fills months before the daily series (FR before
      2004-11; IT's OECD rate is the BdI 10y benchmark monthly mean to the bp). See fetch_ea_yields.
  ES  OECD long-term (10y) rate via FRED only (monthly; no free daily Bonos curve), so its 10y
      runs a month or two behind and its rvol10 is the DE proxy.

Derived: s1030 = 30y − 10y; spread_de = 10y − Bund 10y (an OECD-filled month against the OECD Bund
rate, so both legs are one measure); rel10 = 10y − mean of the UK/DE/US/JP 10y (FR/IT are daily
too but stay out of the peer mean: adding them would make it half euro-area and redefine every
country's rel10);
asw20 = UK 20y gilt zero-coupon spot − SONIA OIS 20y spot. Writes cache/market_monthly.parquet
(long: country, month, series, value, source, last_obs).

Term-premium inputs (`MARKET_META` describes each):
  r3m      3-month rate, % (US 3m bill DTB3; UK SONIA OIS 3m spot from the BoE curves, OECD 3m
           interbank before 2009; DE/FR/IT euro 3m interbank (OECD via FRED); JP OECD 3m interbank,
           call rate where that is missing). Monthly average.
  rx10_3, rx10_12  excess return of the 10y over the h months *starting* in the row's month, %:
           rx ≈ y_t·h/12 − D_t·(y_end − y_t) − r3m_t·h/12, D_t the modified duration of a 10y
           semi-annual par bond at y_t, y_end the (10 − h/12)y yield at t+h interpolated linearly
           between the short leg and the 10y (roll-down; the short leg is the 2y, IT's 3y). Months
           before the short leg starts have no roll-down. Monthly average yields, no convexity.
           Positive = the bond beat cash.
  tp10     10y term-premium proxy, %. US: ACM 10y term premium (NY Fed). UK: 10y gilt par − 10y
           SONIA OIS par (bond over the swap-implied path; the OIS rate still carries its own term
           premium). DE, JP, FR: 10y − 2y (no free 10y OIS). IT: 10y − 3y BTP. FR/IT months
           before their own short leg: 10y − Bund 2y, labelled in `source`.
           Definitions differ by country; pooled tests remove each country's mean only.
  rvol10   realised vol of daily 10y changes over the trailing 3 calendar months, bp annualised
           (√252). Daily markets only; FR/IT months before their own daily 10y carry the DE value,
           labelled as a euro-area proxy.
  move     ICE BofA MOVE index (Yahoo ^MOVE), US, monthly average, when Yahoo answers.
Nominal GDP (quarterly, annualised, local-currency bn) is cached beside it in
cache/gdp_quarterly.parquet for `headline.dur_stock_gdp`.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from issuance.http import session
from issuance.paths import CACHE, REPO

OUT = CACHE / "market_monthly.parquet"
G4 = REPO / "correlations" / "g4" / "cache"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_CACHE = CACHE / "fred"
MOF_ALL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
MOF_NOW = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
OECD = {"FR": "IRLTLT01FRM156N", "IT": "IRLTLT01ITM156N", "DE": "IRLTLT01DEM156N", "ES": "IRLTLT01ESM156N"}
START = "2000-01-01"
# rel10 peer mean. FR/IT have daily 10y too but are left out: three of six peers would be euro
# area, and every country's rel10 would change definition.
PEERS = ("UK", "DE", "US", "JP")
DAILY = PEERS + ("FR", "IT")
EURO = ("DE", "FR", "IT", "ES")
# Short leg for roll-down and the slope proxy, (series, maturity in years).
SHORT = {"IT": ("y3", 3.0)}
# Daily series written to the shared daily-yield caches, per country.
EA_DAILY = {"FR": ("y2", "y10", "y30"), "IT": ("y3", "y10", "y30")}
DAILY_CACHES = ((CACHE / "supply_daily_yields.parquet", "2000-01-01"),
                (CACHE / "market_daily_yields.parquet", "2009-06-01"))
RX_H = (3, 12)
RVOL_MONTHS = 3
RVOL_MIN_OBS = 40
ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
ACM_CACHE = REPO / "us" / "fiscal" / "data" / "cache" / "acm_daily.parquet"
GDP_OUT = CACHE / "gdp_quarterly.parquet"
# FRED id, factor to annualised local-currency bn.
GDP_FRED = {"US": ("GDP", 1.0), "UK": ("UKNGDP", 4e-3), "DE": ("CPMNACSCAB1GQDE", 4e-3),
            "FR": ("CPMNACSCAB1GQFR", 4e-3), "IT": ("CPMNACSCAB1GQIT", 4e-3),
            "ES": ("CPMNACSCAB1GQES", 4e-3), "JP": ("JPNNGDP", 1.0)}
TP10_SOURCE = {"US": "ACM 10y term premium (NY Fed)", "UK": "10y gilt par − 10y SONIA OIS par",
               "DE": "10y − 2y (no free 10y OIS)", "JP": "10y − 2y (no free 10y OIS)",
               "FR": "10y − 2y TEC (no free 10y OIS)", "IT": "10y − 3y BTP benchmark (no 2y BTP benchmark)",
               "ES": "10y (OECD monthly) − Bund 2y (no free daily Bonos curve)"}
MARKET_META = {
    "rx10_3": {"label": "10y excess return, 3m", "unit": "bp", "scale": 100.0,
               "what": "Return of the 10y over the next 3 months minus the 3m rate: carry − duration × yield change "
                       "(with roll-down from the 2y–10y line) − cash. Positive = bond beat cash."},
    "rx10_12": {"label": "10y excess return, 12m", "unit": "bp", "scale": 100.0,
                "what": "As the 3m excess return, over the next 12 months."},
    "tp10": {"label": "10y term premium proxy", "unit": "bp", "scale": 100.0,
             "what": "US ACM 10y term premium; UK 10y gilt − 10y SONIA OIS; DE/JP/FR 10y − 2y; IT 10y − 3y."},
    "rvol10": {"label": "10y realised vol, 3m", "unit": "bp/yr", "scale": 1.0,
               "what": "Standard deviation of daily 10y changes over the trailing three months, annualised. "
                       "FR/IT months before their own daily 10y use the DE value."},
    "r3m": {"label": "3m rate", "unit": "%", "scale": 1.0,
            "what": "US 3m bill; UK SONIA OIS 3m; euro 3m interbank; JP 3m interbank."},
    "move": {"label": "MOVE index", "unit": "index", "scale": 1.0,
             "what": "ICE BofA MOVE (implied Treasury vol), monthly average."},
}


def _fred(series: str) -> pd.Series:
    # FRED stalls on the shared session's browser headers; a bare client answers in <1s.
    own = FRED_CACHE / f"{series}.csv"
    try:
        r = requests.get(FRED.format(series), timeout=30)
        r.raise_for_status()
        text = r.text
        FRED_CACHE.mkdir(parents=True, exist_ok=True)
        own.write_text(text)
    except requests.RequestException:
        cached = next((p for p in (own, G4 / "fred" / f"fred_{series.lower()}.csv") if p.exists()), None)
        if cached is None:
            raise
        text = cached.read_text()
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna()


def _mof() -> pd.DataFrame:
    s = session()
    frames = []
    for url in (MOF_ALL, MOF_NOW):
        r = s.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", "ignore")), skiprows=1)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
        frames.append(df.dropna(subset=["Date"]))
    d = pd.concat(frames).drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    return d.apply(pd.to_numeric, errors="coerce")


def par_from_spot(spot: pd.DataFrame, t: float) -> pd.Series:
    """Semi-annual par yield at maturity t from continuously compounded zero rates (% p.a.) on a
    0.5y grid (columns = maturity in years). The BoE often omits the 0.5y point; a missing short
    end takes the first published rate. Dates missing anything else up to t are dropped."""
    grid = np.arange(0.5, t + 1e-9, 0.5)
    cols = [c for c in spot.columns if np.any(np.isclose(c, grid))]
    if len(cols) != len(grid):
        return pd.Series(dtype=float)
    z = spot[cols]
    lead = z.notna().cumsum(axis=1).eq(0)
    z = z.mask(lead, z.bfill(axis=1)).dropna() / 100
    df = np.exp(-z.to_numpy() * grid)
    return pd.Series(200 * (1 - df[:, -1]) / df.sum(axis=1), index=z.index)


def _uk_curves() -> dict[str, pd.Series]:
    path = REPO / "uk_flows" / "output" / "curves.parquet"
    c = pd.read_parquet(path, columns=["date", "curve", "measure", "segment", "maturity_years", "rate"])
    c = c[(c["measure"] == "spot") & c["curve"].isin(["nominal", "ois"])]
    short = c[(c["curve"] == "ois") & ((c["maturity_years"] - 0.25).abs() < 1e-4)]
    c = c[c["segment"] == "full"]
    nom = c[c["curve"] == "nominal"].pivot_table(index="date", columns="maturity_years", values="rate", observed=True)
    out = {f"y{t}": par_from_spot(nom, t) for t in (2, 10, 20, 30)}
    sub = c[(c["maturity_years"] - 20).abs() < 1e-6].pivot_table(index="date", columns="curve", values="rate", observed=True)
    if {"nominal", "ois"} <= set(sub.columns):
        out["asw20"] = (sub["nominal"] - sub["ois"]).dropna()
    ois = c[c["curve"] == "ois"].pivot_table(index="date", columns="maturity_years", values="rate", observed=True)
    if len(ois):
        out["ois10"] = par_from_spot(ois, 10)
    if len(short):
        # the full segment stops publishing the 3m point in later years; the short-end file has it
        se = short[short["segment"] == "short_end"].groupby("date")["rate"].mean()
        fu = short[short["segment"] == "full"].groupby("date")["rate"].mean()
        out["r3m"] = se.combine_first(fu).dropna()
    return {k: v for k, v in out.items() if len(v)}


def mod_duration(y_pct, years: float = 10.0) -> np.ndarray:
    """Modified duration of a semi-annual par bond at yield y (% p.a.); negative yields allowed."""
    y = np.asarray(y_pct, dtype=float) / 100
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (1 - (1 + y / 2) ** (-2 * years)) / y
    return np.where(np.abs(y) < 1e-8, years, d)


def excess_return(y10: pd.Series, r: pd.Series, h: int, y2: pd.Series | None = None,
                  short_years: float = 2.0) -> pd.Series:
    """Excess return (%) of a 10y par bond bought at month t and held h months, over the h-month
    cash return at r(t): y_t·h/12 − D_t·(y_end − y_t) − r_t·h/12. y_end is the (10 − h/12)y yield at
    t+h, linear between the short leg `y2` (maturity `short_years`) and the 10y when given
    (roll-down), else the 10y. Indexed by the start month t, so the value at t is only known at t+h."""
    idx = pd.date_range(min(y10.index.min(), r.index.min()), max(y10.index.max(), r.index.max()), freq="MS")
    y = y10.reindex(idx)
    end = y.shift(-h)
    if y2 is not None:
        end = end - (h / 12) / (10 - short_years) * (end - y2.reindex(idx).shift(-h))
    d = pd.Series(mod_duration(y.to_numpy()), index=idx)
    return (y * h / 12 - d * (end - y) - r.reindex(idx) * h / 12).dropna()


def realised_vol(s: pd.Series, months: int = RVOL_MONTHS, min_obs: int = RVOL_MIN_OBS) -> pd.Series:
    """Per calendar month, the annualised (√252) bp standard deviation of daily changes dated in
    that month and the `months` − 1 before it; uses nothing after the month's last day."""
    s = s.dropna().sort_index()
    d = s.diff().dropna() * 100
    key = d.index.to_period("M").to_timestamp()
    g = pd.DataFrame({"n": 1.0, "s": d.to_numpy(), "q": d.to_numpy() ** 2}, index=key).groupby(level=0).sum()
    g = g.asfreq("MS", fill_value=0.0)
    w = g.rolling(months, min_periods=months).sum()
    var = (w["q"] - w["s"] ** 2 / w["n"]) / (w["n"] - 1)
    return np.sqrt(var.clip(lower=0) * 252).where(w["n"] >= min_obs).dropna()


def _acm_tp10() -> pd.Series:
    """ACM 10y term premium, daily. Download; the us/fiscal cache if the NY Fed does not answer."""
    try:
        r = requests.get(ACM_URL, headers={"User-Agent": "Mozilla/5.0 (issuance research)"}, timeout=60)
        r.raise_for_status()
        d = pd.read_excel(io.BytesIO(r.content), sheet_name="ACM Daily")
        d["date"] = pd.to_datetime(d["DATE"], format="%d-%b-%Y")
        return pd.to_numeric(d.set_index("date")["ACMTP10"], errors="coerce").dropna()
    except Exception:  # noqa: BLE001
        if not ACM_CACHE.exists():
            raise
        return pd.read_parquet(ACM_CACHE)["ACMTP10"].dropna()


def _move() -> pd.Series:
    import yfinance as yf

    d = yf.download("^MOVE", start=START, progress=False, auto_adjust=False)
    if d is None or d.empty:
        return pd.Series(dtype=float)
    close = d["Close"]
    close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna()


def ea_series(frame: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    """`fetch_ea_yields.daily()` frame → (country, series, source, daily values). The source is
    the earliest row's (the official series); a later vendor tail is recorded in
    `attrs['tail_from']` / `attrs['tail_source']` so monthly rows that use it say so."""
    out = []
    for (c, name), g in frame.groupby(["country", "series"], sort=True):
        g = g.sort_values("date")
        s = pd.Series(g["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(g["date"])).dropna()
        src = g["source"].iloc[0]
        tail = g[g["source"] != src]
        if len(tail):
            s.attrs = {"tail_from": pd.Timestamp(tail["date"].min()), "tail_source": tail["source"].iloc[0]}
        out.append((c, name, src, s))
    return out


def _ea_sets() -> list[tuple[str, str, str, pd.Series]]:
    from issuance import fetch_ea_yields

    d = fetch_ea_yields.daily()
    return ea_series(d[d["series"].isin(["y2", "y3", "y10", "y20", "y30"])])


def _daily_sets() -> list[tuple[str, str, str, pd.Series]]:
    """(country, series, source, daily-or-monthly values)."""
    got: list[tuple[str, str, str, pd.Series]] = []

    def add(country, name, source, fn):
        try:
            s = fn()
            if s is not None and len(s):
                got.append((country, name, source, s))
        except Exception as exc:  # noqa: BLE001 - a dead source keeps the cached series
            print(f"  market: {country} {name} from {source} failed: {exc}")

    for t, sid in ((2, "DGS2"), (10, "DGS10"), (30, "DGS30")):
        add("US", f"y{t}", f"FRED {sid}", lambda sid=sid: _fred(sid))
    bund = G4 / "bbk" / "bbk_bund.csv"
    if bund.exists():
        b = pd.read_csv(bund, index_col=0, parse_dates=True)
        for t in (2, 10, 30):
            if f"bund_{t}" in b:
                add("DE", f"y{t}", "Bundesbank", lambda t=t: b[f"bund_{t}"].dropna())
    for c, sid in OECD.items():
        add(c, "y10_oecd", f"FRED {sid}", lambda sid=sid: _fred(sid))
    try:
        uk = _uk_curves()
        for k, s in uk.items():
            add("UK", k, "BoE fitted curves", lambda s=s: s)
    except Exception as exc:  # noqa: BLE001
        print(f"  market: UK curves failed: {exc}")
    try:
        jp = _mof()
        for t in (2, 10, 30):
            add("JP", f"y{t}", "MoF JGB yields", lambda t=t: jp[f"{t}Y"].dropna())
    except Exception as exc:  # noqa: BLE001
        print(f"  market: MoF JGB yields failed: {exc}")
    try:
        for c, name, source, s in _ea_sets():
            add(c, name, source, lambda s=s: s)
    except Exception as exc:  # noqa: BLE001
        print(f"  market: FR/IT daily yields failed: {exc}")
    add("US", "r3m", "FRED DTB3", lambda: _fred("DTB3"))
    try:
        eur = _fred("IR3TIB01DEM156N")
        for c in EURO:
            add(c, "r3m", "FRED IR3TIB01DEM156N (euro 3m interbank)", lambda: eur)
    except Exception as exc:  # noqa: BLE001
        print(f"  market: euro 3m failed: {exc}")
    add("UK", "r3m_fb", "FRED IR3TIB01GBM156N", lambda: _fred("IR3TIB01GBM156N"))
    add("JP", "r3m", "FRED IR3TIB01JPM156N", lambda: _fred("IR3TIB01JPM156N"))
    add("JP", "r3m_fb", "FRED IRSTCI01JPM156N (call rate)", lambda: _fred("IRSTCI01JPM156N"))
    add("US", "acm_tp10", "ACM (NY Fed)", _acm_tp10)
    add("US", "move", "Yahoo ^MOVE", _move)
    return got


def _gdp_build() -> pd.DataFrame:
    rows = []
    for c, (sid, k) in GDP_FRED.items():
        try:
            s = _fred(sid)
        except Exception as exc:  # noqa: BLE001
            print(f"  market: GDP {c} ({sid}) failed: {exc}")
            continue
        q = s.index.to_period("Q").to_timestamp()
        rows.append(pd.DataFrame({"country": c, "quarter": q, "gdp_bn": s.to_numpy() * k, "source": f"FRED {sid}"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["country", "quarter", "gdp_bn", "source"])


def write_gdp() -> pd.DataFrame:
    """Quarterly nominal GDP, annualised, local-currency bn; a country whose fetch fails keeps its cache."""
    new = _gdp_build()
    if GDP_OUT.exists():
        old = pd.read_parquet(GDP_OUT)
        new = pd.concat([new, old[~old["country"].isin(set(new["country"]))]], ignore_index=True)
    if not new.empty:
        new.to_parquet(GDP_OUT, index=False)
    return new


def load_gdp() -> pd.DataFrame:
    if not GDP_OUT.exists():
        return pd.DataFrame(columns=["country", "quarter", "gdp_bn", "source"])
    d = pd.read_parquet(GDP_OUT)
    d["quarter"] = pd.to_datetime(d["quarter"])
    return d


def build(today: pd.Timestamp | None = None) -> pd.DataFrame:
    # Only complete months: a part-month average would compare a few days with a full month.
    cutoff = (pd.Timestamp.today() if today is None else pd.Timestamp(today)).to_period("M").to_timestamp()
    rows = []
    vols: dict[str, tuple[pd.Series, str, pd.Timestamp]] = {}
    for country, name, source, s in _daily_sets():
        tail_from, tail_src = s.attrs.get("tail_from"), s.attrs.get("tail_source")
        s = s[s.index >= START]
        m = s.groupby(s.index.to_period("M").to_timestamp()).mean()
        m = m[m.index < cutoff]
        src = pd.Series(source, index=m.index, dtype=object)
        if tail_from is not None:
            src[m.index >= pd.Timestamp(tail_from).to_period("M").to_timestamp()] = (
                f"{source}; {tail_src} from {pd.Timestamp(tail_from):%Y-%m-%d}")
        rows.append(pd.DataFrame({"country": country, "month": m.index, "series": name, "value": m.to_numpy(),
                                  "source": src.to_numpy(), "last_obs": s.index.max()}))
        if name == "y10" and country in DAILY:
            v = realised_vol(s)
            tail = "" if tail_from is None else f"; {tail_src} from {pd.Timestamp(tail_from):%Y-%m-%d}"
            vols[country] = (v[v.index < cutoff], f"{source}{tail}, daily changes", s.index.max())
    for c, (v, src, lo) in vols.items():
        rows.append(pd.DataFrame({"country": c, "month": v.index, "series": "rvol10", "value": v.to_numpy(),
                                  "source": src, "last_obs": lo}))
    if "DE" in vols:
        # Months before a euro market's own daily 10y carry the DE value, labelled as a proxy.
        v, src, lo = vols["DE"]
        for e in EURO:
            if e == "DE":
                continue
            own = vols[e][0].index.min() if e in vols and len(vols[e][0]) else pd.Timestamp.max
            p = v[v.index < own]
            rows.append(pd.DataFrame({"country": e, "month": p.index, "series": "rvol10", "value": p.to_numpy(),
                                      "source": f"DE {src} (euro-area proxy)", "last_obs": lo}))
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    # A fallback short rate fills only months before the primary source starts: splicing a call
    # rate onto a 3m rate at the ragged end would read as a jump in the cash leg.
    fb = d[d["series"] == "r3m_fb"].assign(series="r3m")
    first_r = d[d["series"] == "r3m"].groupby("country")["month"].min()
    fb = fb[[m < first_r.get(c, pd.Timestamp.max) for c, m in zip(fb["country"], fb["month"])]]
    d = pd.concat([d[d["series"] != "r3m_fb"], fb], ignore_index=True)
    # The OECD rate fills only months the daily 10y does not cover (FR before 2004-11; DE where the
    # Bundesbank curve is missing; all of FR/IT if their daily sources fail).
    oecd = d[d["series"] == "y10_oecd"].assign(series="y10")
    de_oecd = oecd[oecd["country"] == "DE"].set_index("month")["value"]
    de_oecd_last = oecd.loc[oecd["country"] == "DE", "last_obs"].max()
    have = set(zip(d.loc[d["series"] == "y10", "country"], d.loc[d["series"] == "y10", "month"]))
    oecd = oecd[[(c, m) not in have for c, m in zip(oecd["country"], oecd["month"])]]
    d = pd.concat([d[d["series"] != "y10_oecd"], oecd], ignore_index=True)
    wide = d.pivot_table(index=["country", "month"], columns="series", values="value").reset_index()
    extra = []
    if {"y10", "y30"} <= set(wide.columns):
        w = wide.dropna(subset=["y10", "y30"])
        extra.append(pd.DataFrame({"country": w["country"], "month": w["month"], "series": "s1030",
                                   "value": w["y30"] - w["y10"], "source": "derived"}))
    y10 = wide.pivot_table(index="month", columns="country", values="y10")
    if "DE" in y10:
        sp = y10.sub(y10["DE"], axis=0).drop(columns="DE")
        # FR/IT 10y are benchmark yields (TEC, MTS benchmark, or the OECD rate built from them); the
        # Bundesbank par curve runs 12bp above the OECD Bund benchmark rate on average (38bp in
        # 2009), so FR/IT spread against the OECD Bund rate and both legs stay one measure.
        if len(de_oecd):
            for c in [c for c in sp.columns if c in EURO]:
                sp[c] = y10[c] - de_oecd.reindex(y10.index)
        sp = sp.stack().rename("value").reset_index()
        extra.append(sp.assign(series="spread_de", source="derived"))
    peers = [c for c in PEERS if c in y10]
    if len(peers) == len(PEERS):
        mean = y10[peers].dropna().mean(axis=1)
        rel = y10.sub(mean, axis=0).stack().rename("value").reset_index()
        extra.append(rel.assign(series="rel10", source="derived"))

    def col(c: str, s: str) -> pd.Series | None:
        if s not in wide:
            return None
        v = wide.loc[wide["country"] == c].set_index("month")[s].dropna()
        return v if len(v) else None

    for c in wide["country"].unique():
        short, short_years = SHORT.get(c, ("y2", 2.0))
        y10c, y2c, rc = col(c, "y10"), col(c, short), col(c, "r3m")
        if y10c is not None and rc is not None:
            for h in RX_H:
                rx = excess_return(y10c, rc, h, y2c, short_years=short_years)
                src = pd.Series("derived: 10y, 3m rate" + (f", {short[1:]}y roll-down" if y2c is not None
                                                           else ", no roll-down (no 2y)"), index=rx.index, dtype=object)
                if y2c is not None:
                    # Months before the short leg starts (FR before its TEC 2y or CNBC 2y) keep the
                    # 10y-only excess return rather than losing their history.
                    rx = rx[rx.index >= y2c.index.min()]
                    flat = excess_return(y10c, rc, h)
                    flat = flat[flat.index < y2c.index.min()]
                    src = pd.concat([pd.Series("derived: 10y, 3m rate, no roll-down (before the short leg)",
                                               index=flat.index, dtype=object), src.reindex(rx.index)])
                    rx = pd.concat([flat, rx])
                extra.append(pd.DataFrame({"country": c, "month": rx.index, "series": f"rx10_{h}",
                                           "value": rx.to_numpy(), "source": src.to_numpy()}))
        tp_src = pd.Series(dtype=object)
        if c == "US":
            tp = col(c, "acm_tp10")
        elif c == "UK":
            o = col(c, "ois10")
            tp = None if y10c is None or o is None else y10c - o
        else:
            tp = None if y10c is None or y2c is None else (y10c - y2c).dropna()
            b2 = col("DE", "y2")
            if c in EURO and c != "DE" and y10c is not None and b2 is not None:
                # Months before the own short leg: 10y − Bund 2y (the earlier euro-area definition).
                first = tp.index.min() if tp is not None and len(tp) else pd.Timestamp.max
                fb = (y10c - b2).dropna()
                fb = fb[fb.index < first]
                if len(fb):
                    tp_src = pd.Series("10y − Bund 2y (before the own short leg)", index=fb.index, dtype=object)
                    tp = fb if tp is None else pd.concat([fb, tp])
        if tp is not None and len(tp.dropna()):
            tp = tp.dropna().sort_index()
            src = tp_src.reindex(tp.index).fillna(TP10_SOURCE.get(c, "derived"))
            extra.append(pd.DataFrame({"country": c, "month": tp.index, "series": "tp10", "value": tp.to_numpy(),
                                       "source": src.to_numpy()}))
    last = d.groupby(["country", "series"])["last_obs"].max()
    out = pd.concat([d] + extra, ignore_index=True).dropna(subset=["value"])
    y10_last = {c: last.get((c, "y10"), pd.NaT) for c in out["country"].unique()}

    def oldest(*v):
        return pd.Series(v, dtype="datetime64[ns]").min()

    def stamp(c: str, s: str):
        # A derived series is only as fresh as its oldest input.
        if s == "s1030":
            return oldest(last.get((c, "y10"), pd.NaT), last.get((c, "y30"), pd.NaT))
        if s == "spread_de":
            leg = de_oecd_last if c in EURO and len(de_oecd) else last.get(("DE", "y10"), pd.NaT)
            return oldest(y10_last.get(c, pd.NaT), leg)
        if s == "rel10":
            return oldest(y10_last.get(c, pd.NaT), *(y10_last.get(p, pd.NaT) for p in PEERS))
        if s == "rvol10" and c in EURO and c not in vols:
            return y10_last.get("DE", pd.NaT)
        if s.startswith("rx10_"):
            return oldest(y10_last.get(c, pd.NaT), last.get((c, "r3m"), pd.NaT))
        if s == "tp10":
            leg = {"US": last.get(("US", "acm_tp10"), pd.NaT), "UK": last.get(("UK", "ois10"), pd.NaT)}.get(
                c, last.get((c, SHORT.get(c, ("y2",))[0]), pd.NaT))
            return leg if c == "US" else oldest(y10_last.get(c, pd.NaT), leg)
        return last.get((c, s), pd.NaT)

    keys = out[["country", "series"]].drop_duplicates()
    stamps = {(c, s): stamp(c, s) for c, s in zip(keys["country"], keys["series"])}
    out["last_obs"] = [stamps[(c, s)] for c, s in zip(out["country"], out["series"])]
    return out.sort_values(["country", "series", "month"]).reset_index(drop=True)


def merge_daily(old: pd.DataFrame, ea: pd.DataFrame, start: str) -> pd.DataFrame:
    """Replace the FR/IT rows of a daily-yield cache (country, series, date, value[, source]) with
    `ea` from `start`, for the series in `EA_DAILY`; every other row is kept as it was."""
    want = ea[[s in EA_DAILY.get(c, ()) for c, s in zip(ea["country"], ea["series"])]]
    want = want[want["date"] >= start]
    if want.empty:
        return old
    got = set(zip(want["country"], want["series"]))
    keep = old[[(c, s) not in got for c, s in zip(old["country"], old["series"])]]
    out = pd.concat([keep, want[["country", "series", "date", "value", "source"]]], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def write_daily(ea: pd.DataFrame | None = None) -> None:
    """FR/IT daily yields into the shared daily caches (supply_events, experiments/auction_window).
    Their own refreshes also pull FR/IT through `_daily_sets` (filtered on `DAILY`)."""
    if ea is None:
        from issuance import fetch_ea_yields

        ea = fetch_ea_yields.daily(offline=True)
    for path, start in DAILY_CACHES:
        if not path.exists():
            continue
        merge_daily(pd.read_parquet(path), ea, start).to_parquet(path, index=False)


def write() -> pd.DataFrame:
    try:
        write_gdp()
    except Exception as exc:  # noqa: BLE001 - GDP keeps its cache
        print(f"  market: GDP cache not refreshed: {exc}")
    new = build()
    try:
        write_daily()
    except Exception as exc:  # noqa: BLE001 - the daily caches keep their rows
        print(f"  market: FR/IT daily caches not refreshed: {exc}")
    if OUT.exists():
        old = pd.read_parquet(OUT)
        if not new.empty:
            keep = old[~old.set_index(["country", "series"]).index.isin(new.set_index(["country", "series"]).index)]
            new = pd.concat([new, keep], ignore_index=True)
        else:
            new = old
    if not new.empty:
        new.to_parquet(OUT, index=False)
    return new


def load() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=["country", "month", "series", "value", "source", "last_obs"])
    d = pd.read_parquet(OUT)
    d["month"] = pd.to_datetime(d["month"])
    return d


if __name__ == "__main__":
    out = write()
    last = out.sort_values("month").groupby(["country", "series"]).tail(1)
    print(last[["country", "series", "month", "value", "source", "last_obs"]].to_string(index=False))
