"""Reserve-demand curve: SOFR − IORB as a function of reserves, plus calendar.

    g(x)  = c + exp(a − b·x)                          demand curve, x = reserves / scale (%)
    core  = softmin(cap, softmax(floor, g))
    s_t   = core + w_t·(Σ γ_k cal_k,t + δ·coupon_t) + u_t,   w_t = σ((g − floor)/3bp)

floor_t = ON RRP rate − IORB: while the facility holds cash, money funds lend
there instead of below it, so the spread cannot fall through the floor
whatever reserves are. cap_t = top of range − IORB once the Standing Repo
Facility exists (Jul-2021), +400bp before. Month-end, quarter-end and
settlement pressure only show up when reserves are not abundant, so the
calendar block is scaled by w (0 on the floor, 1 when scarce). δ prices net
coupon settlement ($100bn units) as collateral supply. Soft-L1 loss so
Sep-2019 informs the curvature without owning it.

Two tests:
  stability   fit 2018-04→2023-12, score 2024-01→ on realised reserves
  forecast    expanding refit monthly; reserves at t+h from the TGA flow
              model; variants 'level' (curve + decayed residual) and 'delta'
              (today's spread + curve change + calendar change) against a
              random walk with and without calendar.

  PYTHONPATH=. .venv/bin/python us/fiscal/reserve_demand.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from flow_model import calendar_features
from tga_deficit import fetch_fred, interpolate_to_daily

HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

TRAIN = ("2018-04-02", "2023-12-31")
TEST = ("2024-01-01", "2099-12-31")
SRF_START = pd.Timestamp("2021-07-28")
PRE_SRF_CAP_BP = 400.0
CAL = ["month_end", "first_bd", "quarter_end", "year_end", "mid_month", "corp_tax"]
HORIZONS = (1, 5, 10, 20)
IC_WINDOW = 60  # bdays; trailing median residual for the intercept-corrected forecast
FC_VARIANTS = ("level", "level_ic", "delta", "rw", "rw_cal")
SCALES = {"gdp": "GDP", "deposits": "DPSACBW027SBOG"}  # FRED: nominal GDP SAAR; commercial bank deposits


def load_panel(scale: str = "gdp", liquidity: str = "reserves") -> pd.DataFrame:
    """`liquidity`: 'reserves', or 'reserves_rrp' (reserves + ON RRP: idle MMF cash that lends into repo)."""
    mm = pd.read_parquet(DERIVED / "money_markets_daily.parquet")
    mm = mm[mm["sofr"].notna() & mm["iorb"].notna() & mm["reserves_bn"].notna()].copy()
    mm["gdp_bn"] = interpolate_to_daily(fetch_fred("GDP"), mm.index)
    if scale == "gdp":
        mm["scale_bn"] = mm["gdp_bn"]
    else:
        mm["scale_bn"] = interpolate_to_daily(fetch_fred(SCALES[scale]), mm.index)
    mm["liq_bn"] = mm["reserves_bn"] + (mm["rrp_bn"].fillna(0.0) if liquidity == "reserves_rrp" else 0.0)
    mm["x"] = 100.0 * mm["liq_bn"] / mm["scale_bn"]
    mm["y"] = mm["sofr_iorb_bp"]
    mm["floor"] = (100.0 * (mm["rrp_rate"] - mm["iorb"])).ffill().fillna(-10.0)
    cap = 100.0 * (mm["target_hi"] - mm["iorb"])
    mm["cap"] = np.where(mm.index >= SRF_START, cap, mm["floor"] + PRE_SRF_CAP_BP)
    mm = mm.join(calendar_dummies(mm.index))
    lad_path = DERIVED / "auction_ladder.parquet"
    mm["coupon_settle"] = 0.0
    if lad_path.exists():
        lad = pd.read_parquet(lad_path)
        mm["coupon_settle"] = lad["net_coupons"].reindex(mm.index).fillna(0.0) / 100.0
    return mm


def calendar_dummies(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = calendar_features(idx)
    out = pd.DataFrame(index=idx)
    out["month_end"] = (f["bdom_rev"] == 1).astype(float)
    out["first_bd"] = (f["bdom"] == 1).astype(float)
    out["quarter_end"] = ((f["bdom_rev"] == 1) & f["month"].isin([3, 6, 9, 12])).astype(float)
    out["year_end"] = ((f["bdom_rev"] == 1) & (f["month"] == 12)).astype(float)
    out["mid_month"] = f["pay15_foll"].astype(float)
    out["corp_tax"] = (f["dist_corp"] == 0).astype(float)
    return out


def attach_shifters(mm: pd.DataFrame) -> pd.DataFrame:
    """Repo-demand shifters, % of GDP, as known on each date (publication-lagged).

    debt     marketable debt held by the public (DTS) — collateral to finance
    dealer   primary dealer net Treasury position (FR 2004, public +8d)
    pd_repo  primary dealer Treasury repo borrowing (FR 2004, public +8d)
    """
    from tga_deficit import fetch_debt_public_mn

    mm = mm.copy()
    try:
        debt = fetch_debt_public_mn() / 1000.0
        mm["z_debt"] = 100.0 * debt.reindex(mm.index, method="ffill") / mm["gdp_bn"]
    except Exception as exc:  # noqa: BLE001
        print(f"   debt shifter unavailable: {exc}")
    pd_path = HERE.parents[1] / "flow" / "output" / "nyfed.parquet"
    if pd_path.exists():
        db = pd.read_parquet(pd_path, columns=["date", "keyid", "value"])
        for key, name in (("PDPOSGST-TOT", "z_dealer"), ("PDSORA-UTSETTOT", "z_pd_repo")):
            s = db[db["keyid"] == key].groupby("date")["value"].last().sort_index() / 1000.0
            mm[name] = 100.0 * s.reindex(mm.index, method="ffill") / mm["gdp_bn"]
    return mm


class Curve:
    SOFT = 2.0  # bp; smoothness of the floor / cap kinks

    def __init__(self, shifters: tuple[str, ...] = ()):
        self.shifters = tuple(shifters)
        self.params: np.ndarray | None = None

    @staticmethod
    def demand(p: np.ndarray, x) -> np.ndarray:
        return p[-2] + np.exp(np.clip(p[0] - p[1] * np.asarray(x, dtype=float), -20, 8))

    def _theta(self) -> np.ndarray:
        k = len(self.shifters)
        return self.params[2 + len(CAL):2 + len(CAL) + k]

    def parts(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(core level, calendar block) for rows of d."""
        p = self.params
        g = self.demand(p, d["x"].to_numpy())
        if self.shifters:
            z = d[list(self.shifters)].to_numpy() - self.z_ref
            g = g + z @ self._theta()
        lo, hi, s = d["floor"].to_numpy(), d["cap"].to_numpy(), self.SOFT
        v = lo + s * np.logaddexp(0.0, (g - lo) / s)
        core = hi - s * np.logaddexp(0.0, (hi - v) / s)
        w = 1.0 / (1.0 + np.exp(-(g - lo) / 3.0))
        cal = w * (d[CAL].to_numpy() @ p[2:2 + len(CAL)] + p[-1] * d["coupon_settle"].to_numpy())
        return core, cal

    def predict(self, d: pd.DataFrame, calendar: bool = True) -> np.ndarray:
        core, cal = self.parts(d)
        return core + cal if calendar else core

    def fit(self, d: pd.DataFrame) -> "Curve":
        if self.shifters:
            d = d.dropna(subset=list(self.shifters))
            self.z_ref = d[list(self.shifters)].mean().to_numpy()
        y = d["y"].to_numpy()

        def resid(p):
            self.params = p
            return self.predict(d) - y

        k = len(self.shifters)
        p0 = np.r_[np.log(15.0) + 0.6 * 8.0, 0.6, np.zeros(len(CAL)), np.zeros(k), -20.0, 0.0]
        lb = np.r_[-np.inf, 0.01, np.full(len(CAL), -30.0), np.full(k, -50.0), -60.0, -20.0]
        ub = np.r_[np.inf, 5.0, np.full(len(CAL), 60.0), np.full(k, 50.0), 10.0, 20.0]
        self.params = least_squares(resid, p0, loss="soft_l1", f_scale=3.0, bounds=(lb, ub)).x
        return self

    def to_dict(self) -> dict:
        p = self.params
        out = {"a": float(p[0]), "b": float(p[1]), **{k: float(v) for k, v in zip(CAL, p[2:2 + len(CAL)])}}
        if self.shifters:
            out.update({f"theta_{s}": float(v) for s, v in zip(self.shifters, self._theta())})
        out.update({"c": float(p[-2]), "coupon_per_100bn": float(p[-1])})
        return out

    def reserves_for_spread(self, spread_bp: float, scale_bn: float) -> float:
        """Reserves ($bn) at which the demand curve g reaches `spread_bp` (ex-calendar)."""
        a, b, c = self.params[0], self.params[1], self.params[-2]
        if spread_bp <= c:
            return np.nan
        return (a - np.log(spread_bp - c)) / b / 100.0 * scale_bn

    def slope_bp_per_100bn(self, x: float, scale_bn: float) -> float:
        """dg/dR at x, bp per $100bn of reserves (negative: fewer reserves, higher spread)."""
        a, b = self.params[0], self.params[1]
        return float(-b * np.exp(a - b * x) * 100.0 / scale_bn * 100.0)


def score_stability(mm: pd.DataFrame, curve: Curve) -> dict:
    out = {}
    for name, (a, b) in (("train", TRAIN), ("test", TEST)):
        d = mm.loc[a:b].dropna(subset=list(curve.shifters))
        e = curve.predict(d) - d["y"].to_numpy()
        out[name] = {"n": len(d), "mae_bp": float(np.abs(e).mean()), "bias_bp": float(e.mean()),
                     "mae_const_bp": float(np.abs(d["y"] - d["y"].mean()).mean())}
    return out


def score_forecast(mm: pd.DataFrame, shifters: tuple[str, ...] = (), start: str = TEST[0]
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-refit h-day forecasts with reserves from the TGA flow model.

    R̂_{t+h} = R_{t+h} − (operating-flow forecast error to t+h): the financing
    leg is announced ~1 week ahead and fixed by the refunding beyond, so it
    is treated as known, and ON RRP / Fed assets as realised. The only
    reserves error is therefore the flow model's — the honest part of what
    this system contributes.
    """
    bt_path = DERIVED / "flow_backtest.parquet"
    if not bt_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    bt = pd.read_parquet(bt_path).sort_values(["origin", "h"])
    bt["flow_err_bn"] = (bt.groupby("origin")["model"].cumsum() - bt.groupby("origin")["actual"].cumsum()) / 1000.0
    bt = bt[bt["h"].isin(HORIZONS) & (bt["origin"] >= start)]
    rows = []
    curve, fitted, rho = None, None, 0.0
    for o, g in bt.groupby("origin"):
        if o not in mm.index:
            continue
        if fitted != o.to_period("M"):
            hist = mm[mm.index <= o].dropna(subset=list(shifters))
            curve = Curve(shifters).fit(hist)
            u_hist = hist["y"] - curve.predict(hist)
            rho = float(np.clip(u_hist.tail(500).autocorr(1), 0.0, 0.999))
            fitted = o.to_period("M")
        oo = mm.loc[[o]]
        core_o, cal_o = curve.parts(oo)
        u_o = mm.at[o, "y"] - (core_o + cal_o)[0]
        h_o = mm[mm.index <= o].tail(IC_WINDOW)
        u_bar = float(np.median(h_o["y"].to_numpy() - curve.predict(h_o)))
        for _, r in g.iterrows():
            d = r["date"]
            if d not in mm.index:
                continue
            dd = mm.loc[[d]].copy()
            dd["x"] = 100.0 * (mm.at[d, "liq_bn"] - r["flow_err_bn"]) / mm.at[d, "scale_bn"]
            core_d, cal_d = curve.parts(dd)
            y_o = mm.at[o, "y"]
            rows.append({
                "origin": o, "h": int(r["h"]), "date": d, "actual": mm.at[d, "y"],
                "level": core_d[0] + cal_d[0] + rho ** r["h"] * u_o,
                "level_ic": core_d[0] + cal_d[0] + u_bar + rho ** r["h"] * (u_o - u_bar),
                "delta": y_o + (core_d[0] - core_o[0]) + (cal_d[0] - cal_o[0]),
                "rw": y_o,
                "rw_cal": y_o + (cal_d[0] - cal_o[0]),
            })
    f = pd.DataFrame(rows)
    s = f.groupby("h").apply(lambda x: pd.Series(
        {"n": len(x), **{c: (x[c] - x["actual"]).abs().mean() for c in FC_VARIANTS},
         **{f"bias_{c}": (x[c] - x["actual"]).mean() for c in ("level", "level_ic")}}),
        include_groups=False)
    return s, f


PRODUCTION = {"scale": "gdp", "shifters": (), "liquidity": "reserves"}


def run(scale: str = PRODUCTION["scale"], shifters: tuple[str, ...] = PRODUCTION["shifters"],
        verbose: bool = True, liquidity: str = PRODUCTION["liquidity"]) -> tuple[Curve, dict]:
    mm = load_panel(scale, liquidity)
    if shifters:
        mm = attach_shifters(mm)
        mm = mm.dropna(subset=list(shifters))
    production = (scale == PRODUCTION["scale"] and tuple(shifters) == tuple(PRODUCTION["shifters"])
                  and liquidity == PRODUCTION["liquidity"])
    extra = [] if liquidity == "reserves" else [liquidity]
    tag = "" if production else "_" + "_".join([scale, *extra, *shifters])
    stab_curve = Curve(shifters).fit(mm.loc[TRAIN[0]:TRAIN[1]])
    stab = score_stability(mm, stab_curve)
    fc, f = score_forecast(mm, shifters)
    full = Curve(shifters).fit(mm)
    if production:
        mm["fit_full"] = full.predict(mm)
        mm["curve_full_excal"] = full.predict(mm, calendar=False)
        mm["fit_stability"] = stab_curve.predict(mm)
        mm[["y", "x", "floor", "cap", "reserves_bn", "scale_bn", "fit_stability", "fit_full", "curve_full_excal"]] \
            .to_parquet(DERIVED / "reserve_demand_fit.parquet")
        if not f.empty:
            f.to_parquet(DERIVED / "reserve_demand_forecast_bt.parquet", index=False)
    last = mm.iloc[-1]
    report = {
        "scale": scale, "shifters": list(shifters), "train_params": stab_curve.to_dict(), "full_params": full.to_dict(),
        "stability": stab, "forecast": fc.reset_index().to_dict(orient="records") if not fc.empty else [],
        "now": {"date": str(mm.index[-1].date()), "reserves_bn": float(last["reserves_bn"]),
                "x": float(last["x"]), "slope_bp_per_100bn": full.slope_bp_per_100bn(last["x"], last["scale_bn"]),
                "reserves_at_0bp": full.reserves_for_spread(0.0, last["scale_bn"]),
                "reserves_at_5bp": full.reserves_for_spread(5.0, last["scale_bn"])},
    }
    (DERIVED / f"reserve_demand{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    if verbose:
        print(f"[{scale} {shifters}] full params:", {k: round(v, 3) for k, v in full.to_dict().items()})
        print("stability:", json.dumps(stab, default=lambda v: round(v, 2)))
        print("now:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in report["now"].items()})
        if not fc.empty:
            print(fc.round(2).to_string())
    return full, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=PRODUCTION["scale"], choices=list(SCALES))
    ap.add_argument("--shifters", nargs="*", default=list(PRODUCTION["shifters"]),
                    choices=["z_debt", "z_dealer", "z_pd_repo"])
    ap.add_argument("--liquidity", default=PRODUCTION["liquidity"], choices=["reserves", "reserves_rrp"])
    a = ap.parse_args()
    run(a.scale, tuple(a.shifters), liquidity=a.liquidity)
