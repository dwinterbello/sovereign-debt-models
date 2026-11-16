"""Government spot curves and the forward par yield of a bond issued later.

UK  Bank of England nominal (and real) continuously compounded zeros, already in
    ``uk_flows/output/curves.parquet``.
US  Fed GSW Svensson continuously compounded zeros (``SVENY``).
DE  Bundesbank BBSIS fitted zero rates on listed Federal securities, 1y–30y.
JP  MoF constant-maturity par yields, bootstrapped.
FR, IT  TradingView cash par yields (``TVC:FR*`` / ``TVC:IT*``), bootstrapped. Spot only;
    the 50y print is dropped (it sits through the 30y).

A forward rate is not a separate market. ``forward_par`` discounts with these zeros and
returns the par yield, in the market's coupon frequency, of a bond issued ``start`` years
from now. ``shock_bp`` is added to the zero curve before that calculation: a number shifts every
maturity alike, a callable ``f(maturity_years) -> bp`` shifts each by its own amount (curve shapes).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE.parent / "data" / "debt" / "gov_zeros.parquet"
TV_PAR = HERE.parent / "data" / "debt" / "tv_par.parquet"
LOG = HERE.parent / "data" / "debt" / "gov_curve_log.json"
BOE = REPO / "uk_flows" / "output" / "curves.parquet"
GSW = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
BBK = ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
       "D.I.ZST.ZI.EUR.S1311.B.A604.{stem}.R.A.A._Z._Z.A?format=csv&lang=en")
MOF = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
TV = "https://scanner.tradingview.com/global/scan"

# coupon frequency of the par quote
FREQ = {"uk": 2, "us": 2, "jp": 2, "de": 1, "fr": 1, "it": 1, "es": 1}
TENORS_Y = {
    "fr": [1 / 12, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
    "it": [1 / 12, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
    "es": [1 / 12, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
}


def _get(url: str, data: bytes | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0",
                                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _pchip(x: np.ndarray, y: np.ndarray):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.r_[True, np.diff(x) > 1e-9]
    x, y = x[keep], y[keep]
    if len(x) < 2:
        raise ValueError("curve has fewer than two points")
    f = PchipInterpolator(x, y, extrapolate=False)

    def z(t):
        t = np.asarray(t, dtype=float)
        return np.where(t <= x[0], y[0], np.where(t >= x[-1], y[-1], f(np.clip(t, x[0], x[-1]))))

    return z, x, y


def zeros_from_par(mat: np.ndarray, par_pct: np.ndarray, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap a par curve (percent) into continuously compounded zeros (percent).

    The par quote is the annual coupon rate; payments are ``rate / freq``. Dates under one year are
    discounted as a simple money-market yield. From one year on, the par curve is interpolated onto
    the payment grid and bootstrapped.
    """
    mat = np.asarray(mat, float)
    par = np.asarray(par_pct, float) / 100.0
    yld, _, _ = _pchip(mat, par)
    step = 1.0 / freq
    tmax = max(float(np.nanmax(mat)), 30.0)
    times: list[float] = []
    dfs: list[float] = []
    t = step
    while t <= tmax + 1e-9:
        c = float(yld(min(t, float(np.nanmax(mat)))))
        if t < 1.0 - 1e-8:
            df = 1.0 / (1.0 + c * t)
        else:
            acc = float(np.sum(dfs))
            df = (1.0 - c * step * acc) / (1.0 + c * step)
            if not np.isfinite(df) or df <= 0:
                df = (dfs[-1] if dfs else 1.0) * float(np.exp(-c * step))
        times.append(round(t, 10))
        dfs.append(float(df))
        t = round(t + step, 10)
    tt = np.asarray(times)
    zz = -np.log(np.asarray(dfs)) / tt * 100.0
    return tt, zz


def _boe(kind: str = "nominal") -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    c = pd.read_parquet(BOE, columns=["date", "maturity_years", "rate", "curve", "measure", "segment"])
    c = c[(c["curve"] == kind) & (c["measure"] == "spot") & (c["segment"] == "full")]
    asof = pd.Timestamp(c["date"].max())
    g = c[c["date"] == asof].sort_values("maturity_years")
    return asof, g["maturity_years"].to_numpy(float), g["rate"].to_numpy(float)


def _gsw() -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    raw = _get(GSW).decode("utf-8", "replace")
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date"))
    df = pd.read_csv(__import__("io").StringIO("\n".join(lines[start:])), low_memory=False)
    zero = [c for c in df.columns if len(c) == 7 and c.startswith("SVENY")]
    df = df[["Date", *zero]].apply(lambda s: pd.to_datetime(s) if s.name == "Date" else pd.to_numeric(s, errors="coerce"))
    df = df.dropna(subset=["Date"]).sort_values("Date")
    row = df.dropna(subset=zero, how="all").iloc[-1]
    mats = np.array([int(c[5:]) for c in zero], float)
    return pd.Timestamp(row["Date"]), mats, row[zero].to_numpy(float)


def _bbk() -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    stem = "+".join(f"R{t:02d}XX" for t in range(1, 31))
    text = _get(BBK.format(stem=stem) + "&startPeriod=2026-09-01").decode("utf-8", "replace")
    from correlations.g4.sources import parse_bundesbank
    wide = parse_bundesbank(text)
    # columns are the BBSIS keys; maturity is the Rxx token
    asof = wide.dropna(how="all").index.max()
    row = wide.loc[asof]
    mats, vals = [], []
    for col, val in row.items():
        token = str(col)
        hit = [p for p in token.split(".") if p.startswith("R") and p[1:3].isdigit()]
        if not hit or not np.isfinite(val):
            continue
        mats.append(int(hit[0][1:3]))
        vals.append(float(val))
    order = np.argsort(mats)
    return pd.Timestamp(asof), np.asarray(mats, float)[order], np.asarray(vals, float)[order]


def _mof() -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    from correlations.g4.sources import parse_mof
    wide = parse_mof(_get(MOF).decode("utf-8", "replace"))
    asof = wide.dropna(how="all").index.max()
    row = wide.loc[asof].dropna()
    mats = np.array([int(c.split("_")[1]) for c in row.index], float)
    order = np.argsort(mats)
    return pd.Timestamp(asof), mats[order], row.to_numpy(float)[order]


def _tv_symbols() -> list[tuple[str, float, str]]:
    out = []
    for cc, tenors in TENORS_Y.items():
        for t in tenors:
            label = f"{int(round(t * 12)):02d}MY" if t < 1 else f"{int(t):02d}Y"
            out.append((cc, t, f"TVC:{cc.upper()}{label}"))
    return out


def fetch_tradingview() -> pd.DataFrame:
    """One scanner call for the France and Italy cash par curves. Appends a dated vintage."""
    spec = _tv_symbols()
    body = json.dumps({"symbols": {"tickers": [s for _, _, s in spec]}, "columns": ["close"]}).encode()
    data = json.loads(_get(TV, data=body))
    got = {row["s"]: row["d"][0] for row in data.get("data", [])}
    fetched = pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows = []
    for cc, t, code in spec:
        if code not in got or got[code] is None:
            continue
        rows.append({"fetched": fetched, "cc": cc, "maturity": t, "par": float(got[code]), "symbol": code})
    if len(rows) < 10:
        raise RuntimeError(f"TradingView returned {len(rows)} quotes")
    new = pd.DataFrame(rows)
    TV_PAR.parent.mkdir(parents=True, exist_ok=True)
    if TV_PAR.is_file():
        old = pd.read_parquet(TV_PAR)
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(TV_PAR, index=False)
    return new[new["fetched"] == fetched]


def _tv_latest() -> dict[str, tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    q = fetch_tradingview()
    out = {}
    for cc, g in q.groupby("cc"):
        g = g.sort_values("maturity")
        out[cc] = (pd.Timestamp(g["fetched"].iloc[0]).normalize(), g["maturity"].to_numpy(float), g["par"].to_numpy(float))
    return out


def auction_nominal_asof(cc: str, asof: pd.Timestamp | None = None,
                         lookback_months: int = 18
                         ) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """Par nominal yields from the latest bond auction at each tenor ≤ ``asof``.

    Used for Spain (no MoF/TV cash strip in the live cache) and as a PIT fallback
    in the ISIN+curve interest screen. Bills excluded; yield band −1…15.
    """
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    if not path.is_file():
        raise RuntimeError(f"no ops tape for {cc}")
    ops = pd.read_parquet(path, columns=["op_dt", "maturity_dt", "yield", "sec_type", "amount_issued"])
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["maturity_dt"] = pd.to_datetime(ops["maturity_dt"])
    ops["yield"] = pd.to_numeric(ops["yield"], errors="coerce")
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(ops["op_dt"].max())
    cut = asof - pd.DateOffset(months=lookback_months)
    g = ops[(ops["op_dt"] <= asof) & (ops["op_dt"] >= cut) & (ops["amount_issued"] > 0)
            & (ops["sec_type"] == "Nominal") & ops["yield"].between(-1, 15)].copy()
    if g.empty:
        raise RuntimeError(f"{cc} has no nominal auctions ≤ {asof.date()}")
    g["tenor"] = ((g["maturity_dt"] - g["op_dt"]).dt.days / 365.25).round().astype(int)
    g = g[(g["tenor"] >= 1) & (g["tenor"] <= 40)].sort_values("op_dt")
    last = g.groupby("tenor").tail(1).sort_values("tenor")
    if len(last) < 3:
        raise RuntimeError(f"{cc} auction curve has {len(last)} tenors at {asof.date()}")
    return (pd.Timestamp(g["op_dt"].max()).normalize(),
            last["tenor"].to_numpy(float), last["yield"].to_numpy(float))


def _auction_real(cc: str) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """Par real yields from the latest linker auction at each tenor in the last 18 months.

    A five-year average mixes the 2021 negative-yield prints into today's curve. Reopenings of
    old low-coupon lines also print a point away from the par bond, so a knot more than 0.6pp
    off the median of its neighbours is dropped and the rest is stored at that median.
    """
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    ops = pd.read_parquet(path, columns=["op_dt", "maturity_dt", "yield", "sec_type", "amount_issued"])
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["maturity_dt"] = pd.to_datetime(ops["maturity_dt"])
    ops["yield"] = pd.to_numeric(ops["yield"], errors="coerce")
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    cut = ops["op_dt"].max() - pd.DateOffset(months=18)
    g = ops[(ops["sec_type"] == "Linker") & (ops["op_dt"] >= cut) & (ops["amount_issued"] > 0)
            & ops["yield"].between(-1, 8)].copy()
    tenor = ((g["maturity_dt"] - g["op_dt"]).dt.days / 365.25).round().astype(int)
    g = g.assign(tenor=tenor)
    g = g[(g["tenor"] >= 2) & (g["tenor"] <= 40)].sort_values("op_dt")
    if g.empty or g["tenor"].nunique() < 3:
        raise RuntimeError(f"{cc} linker auctions span {0 if g.empty else g['tenor'].nunique()} tenors")
    last = g.groupby("tenor").tail(1).sort_values("tenor")
    mat = last["tenor"].to_numpy(float)
    par = last["yield"].to_numpy(float)
    smooth = np.array([np.median(par[max(0, i - 2):i + 3]) for i in range(len(par))])
    keep = np.abs(par - smooth) <= 0.6
    if keep.sum() < 3 or mat[keep].max() - mat[keep].min() < 8:
        raise RuntimeError(f"{cc} linker curve has no usable span after the neighbour screen")
    return pd.Timestamp(g["op_dt"].max()).normalize(), mat[keep], smooth[keep]


def _spot_par(zfn, tenor: float, freq: int) -> float:
    step = 1.0 / freq
    pay = np.arange(step, tenor + 1e-9, step)
    if len(pay) == 0:
        return float("nan")
    dfs = np.exp(-zfn(pay) / 100.0 * pay)
    c = (1.0 - float(dfs[-1])) / (step * float(np.sum(dfs)))
    return 100.0 * c


def _auction_knots(cc: str) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """Latest linker-auction real yield at each tenor in the last 18 months. One point is enough:
    it becomes a breakeven subtracted from the nominal zero curve, not a curve of its own."""
    path = REPO / "issuance" / "cache" / f"ops_{cc}.parquet"
    ops = pd.read_parquet(path, columns=["op_dt", "maturity_dt", "yield", "sec_type", "amount_issued"])
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    ops["maturity_dt"] = pd.to_datetime(ops["maturity_dt"])
    ops["yield"] = pd.to_numeric(ops["yield"], errors="coerce")
    ops["amount_issued"] = pd.to_numeric(ops["amount_issued"], errors="coerce")
    cut = ops["op_dt"].max() - pd.DateOffset(months=18)
    g = ops[(ops["sec_type"] == "Linker") & (ops["op_dt"] >= cut) & (ops["amount_issued"] > 0)
            & ops["yield"].between(-1, 8)].copy()
    tenor = ((g["maturity_dt"] - g["op_dt"]).dt.days / 365.25).round().astype(int)
    g = g.assign(tenor=tenor)
    g = g[(g["tenor"] >= 2) & (g["tenor"] <= 40)].sort_values("op_dt")
    if g.empty:
        raise RuntimeError(f"{cc} has no linker auction in 18 months")
    last = g.groupby("tenor").tail(1).sort_values("tenor")
    return pd.Timestamp(g["op_dt"].max()).normalize(), last["tenor"].to_numpy(float), last["yield"].to_numpy(float)


def _breakeven_real(rows: list[dict]) -> list[dict]:
    """Real zeros = nominal zeros minus the breakeven implied by linker auctions.

    The nominal curve supplies the shape. Each auction supplies the breakeven at its tenor.
    One tenor (Japan) is a flat breakeven. No auction inside 18 months (Germany) stays absent.
    """
    by = {r["cc"]: r for r in rows}
    made = []
    for cc in ("fr", "it", "de", "jp"):
        if cc not in by:
            continue
        try:
            asof, mat, par = _auction_knots(cc)
        except Exception:
            continue
        nom = by[cc]
        zfn = _pchip(np.asarray(nom["maturity"], float), np.asarray(nom["zero"], float))[0]
        freq = FREQ[cc]
        pts = []
        for t, rp in zip(mat, par):
            sp = _spot_par(zfn, float(t), freq)
            if np.isfinite(sp):
                pts.append((float(t), sp - float(rp)))
        if not pts:
            continue
        pts.sort()
        bm = np.array([p[0] for p in pts])
        bv = np.array([p[1] for p in pts])
        grid = np.asarray(nom["maturity"], float)
        be = np.interp(grid, bm, bv)
        label = ("nominal zero minus a single-tenor linker breakeven" if bm.max() - bm.min() < 5
                 else "nominal zero minus linker-auction breakevens")
        made.append(_row(f"{cc}_real", asof, grid, np.asarray(nom["zero"], float) - be, "zero", label))
    return made


def _previous() -> pd.DataFrame | None:
    if CACHE.is_file():
        return pd.read_parquet(CACHE)
    return None


def _kept(old: pd.DataFrame | None, *ccs: str) -> list[dict]:
    if old is None:
        return []
    g = old[old["cc"].isin(ccs)]
    return g.to_dict("records")


def build(countries: tuple[str, ...] = ("uk", "us", "de", "jp", "fr", "it", "es")) -> pd.DataFrame:
    """Latest curve per country. A source that fails keeps its previous row."""
    old = _previous()
    rows: list[dict] = []
    log: dict = {"fetched": pd.Timestamp.now().isoformat(timespec="seconds"), "ok": [], "kept": [], "error": {}}

    def take(name: str, ccs: tuple[str, ...], producer):
        try:
            made = producer()
            if not made:
                raise RuntimeError("empty")
            rows.extend(made)
            log["ok"].append(name)
        except Exception as e:
            kept = _kept(old, *ccs)
            rows.extend(kept)
            if kept:
                log["kept"].append(name)
            log["error"][name] = f"{type(e).__name__}: {e}"

    def uk():
        asof, m, z = _boe("nominal")
        made = [_row("uk", asof, m, z, "zero", "BoE nominal spot zero")]
        _, rm, rz = _boe("real")
        made.append(_row("uk_real", asof, rm, rz, "zero", "BoE real spot zero"))
        return made

    def _fed_curve(url: str, prefix: str) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
        import io
        import re
        raw = _get(url).decode("utf-8", "replace")
        lines = raw.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Date"))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), low_memory=False)
        cols = [c for c in df.columns if re.fullmatch(prefix + r"\d\d", c)]
        df = df[["Date", *cols]].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        row = df.dropna(subset=["Date"]).dropna(subset=cols, how="all").iloc[-1]
        mats = np.array([int(c[len(prefix):]) for c in cols], float)
        return pd.Timestamp(row["Date"]), mats, row[cols].to_numpy(float)

    def us():
        asof, m, z = _gsw()
        made = [_row("us", asof, m, z, "zero", "Fed GSW SVENY zero")]
        pasof, pm, pp = _fed_curve(GSW, "SVENPY")
        basof, bm, bb = _fed_curve(
            "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv", "BKEVENPY")
        common = np.array(sorted(set(pm).intersection(bm)), float)
        real_par = np.array([pp[pm == t][0] - bb[bm == t][0] for t in common])
        tm, tz = zeros_from_par(common, real_par, FREQ["us"])
        made.append(_row("us_real", min(pasof, basof), tm, tz, "zero",
                         "GSW nominal par minus par breakeven, bootstrapped"))
        return made

    def de():
        asof, m, z = _bbk()
        return [_row("de", asof, m, z, "zero", "Bundesbank BBSIS zero")]

    def jp():
        asof, m, p = _mof()
        zz_m, zz = zeros_from_par(m, p, FREQ["jp"])
        return [_row("jp", asof, zz_m, zz, "zero", "MoF JGB par, bootstrapped")]

    def linker_reals():
        made = []
        for cc in ("de", "fr", "it", "jp"):
            try:
                asof, mat, par = _auction_real(cc)
            except Exception:
                continue
            tm, tz = zeros_from_par(mat, par, FREQ[cc])
            made.append(_row(f"{cc}_real", asof, tm, tz, "zero",
                             "linker-auction par yields, bootstrapped"))
        if not made:
            raise RuntimeError("no linker auction curves")
        return made

    def tv():
        made = []
        for cc, (asof, mat, par) in _tv_latest().items():
            if cc not in countries:
                continue
            m, z = zeros_from_par(mat, par, FREQ[cc])
            made.append(_row(cc, asof, m, z, "zero", "TradingView cash par, bootstrapped"))
        return made

    take("uk", ("uk", "uk_real"), uk)
    take("us", ("us", "us_real"), us)
    take("linker_real", ("de_real", "fr_real", "it_real", "jp_real"), linker_reals)
    take("de", ("de",), de)
    take("jp", ("jp",), jp)
    take("tradingview", ("fr", "it", "es"), tv)

    def es_auction():
        asof, mat, par = auction_nominal_asof("es")
        zm, zz = zeros_from_par(mat, par, FREQ["es"])
        return [_row("es", asof, zm, zz, "zero", "Tesoro auction par, bootstrapped")]

    # Prefer TV when it returned ES; else auction tape (always available offline).
    if not any(r["cc"] == "es" for r in rows) and "es" in countries:
        take("es_auction", ("es",), es_auction)
    fresh = _breakeven_real(rows)
    if fresh:
        replaced = {r["cc"] for r in fresh}
        rows = [r for r in rows if r["cc"] not in replaced] + fresh
        log["ok"].append("breakeven_real")
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"no government curve: {log['error']}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    LOG.write_text(json.dumps(log, indent=1))
    return out


def _row(cc, asof, mat, zero, kind, source) -> dict:
    return {"cc": cc, "asof": pd.Timestamp(asof), "maturity": [float(x) for x in np.asarray(mat, float)],
            "zero": [float(x) for x in np.asarray(zero, float)], "kind": kind, "source": source}


def age_hours() -> float:
    import datetime as _dt
    if not CACHE.is_file():
        return 1e9
    return (_dt.datetime.now() - _dt.datetime.fromtimestamp(CACHE.stat().st_mtime)).total_seconds() / 3600


def stale(max_age_h: float = 18.0) -> bool:
    return age_hours() >= max_age_h


def load(max_age_h: float = 18.0) -> pd.DataFrame:
    if stale(max_age_h) and (__import__("os").environ.get("SOVDEBT_PULL") == "1" or not CACHE.is_file()):
        return build()
    return pd.read_parquet(CACHE)


def shift_at(shock_bp, t) -> np.ndarray:
    """The shock in bp at maturities ``t`` (years): a number is a parallel shift, a callable a curve shape."""
    t = np.asarray(t, float)
    if callable(shock_bp):
        return np.asarray(shock_bp(t), float) * np.ones_like(t)
    return np.full(t.shape, float(shock_bp))


def _curve(cc: str, shock_bp=0.0):
    d = load()
    g = d[d["cc"] == cc]
    if g.empty:
        return None
    # parquet may have stored arrays as objects or lists
    mat = np.asarray(g["maturity"].iloc[0], float)
    zero = np.asarray(g["zero"].iloc[0], float) + shift_at(shock_bp, mat) / 100.0
    return _pchip(mat, zero)[0], str(g["source"].iloc[0]), pd.Timestamp(g["asof"].iloc[0])


def forward_par(cc: str, start: float, tenor, shock_bp: float = 0.0) -> np.ndarray:
    """Par yield (percent) of a bond issued in ``start`` years, maturity ``start + tenor``."""
    got = _curve(cc, shock_bp)
    if got is None:
        raise KeyError(cc)
    zfn, _, _ = got
    freq = FREQ.get(cc.replace("_real", ""), 2)
    step = 1.0 / freq
    tenor = np.atleast_1d(np.asarray(tenor, float))

    def df(t):
        t = np.asarray(t, float)
        return np.exp(-zfn(t) / 100.0 * t)

    out = np.empty(len(tenor))
    for i, T in enumerate(tenor):
        pay = np.arange(step, T + 1e-9, step)
        if len(pay) == 0 or T < step - 1e-9:
            # simple money-market yield from the forward discount
            d0 = 1.0 if start <= 0 else float(df(start))
            d1 = float(df(start + T))
            out[i] = 100.0 * (d0 / d1 - 1.0) / T
            continue
        dfs = df(pay) if start <= 0 else df(start + pay) / float(df(start))
        acc = float(np.sum(dfs))
        # c is the annual coupon rate; each payment is c * step
        c = (1.0 - float(dfs[-1])) / (step * acc)
        out[i] = 100.0 * c
    return out if out.size > 1 else out


def bill_yield(cc: str, start: float, shock_bp: float = 0.0) -> float:
    """3-month yield (percent) off the same zero curve."""
    return float(forward_par(cc, start, 0.25, shock_bp)[0])


if __name__ == "__main__":
    df = build()
    for cc in ("uk", "us", "de", "jp", "fr", "it"):
        spot = forward_par(cc, 0.0, [2, 10, 30])
        fwd = forward_par(cc, 2.0, [10])
        print(f"{cc:4} {df.loc[df.cc==cc,'source'].iloc[0]:42}  asof {df.loc[df.cc==cc,'asof'].iloc[0].date()}"
              f"  2s {spot[0]:.2f}  10s {spot[1]:.2f}  30s {spot[2]:.2f}  2y10y {fwd[0]:.2f}")
