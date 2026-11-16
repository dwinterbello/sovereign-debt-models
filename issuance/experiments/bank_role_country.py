"""Per-country bank role for absorption: forward link vs selloff co-move.

Pooled DE+FR+IT bank tests hide the split. Score each country alone:

  inelastic  no forward predictive link, and same-window buying rises with cheapening
             (UK: s1030 / asw20; IT: Bund spread; FR/DE: neither)

    PYTHONPATH=. .venv/bin/python -m issuance.experiments.bank_role_country

Writes ``cache/bank_role_country.parquet`` and prints the call used by
``sovereign_cash.debt.note.rebuild_ex8_absorption.EA_BANKS_INELASTIC``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from issuance import ecb_holders as eh
from issuance import evidence as ev
from issuance import market
from issuance import uk_holders as uh
from issuance.paths import CACHE

OUT = CACHE / "bank_role_country.parquet"
LAG = 1
WINDOW = "2010+"


def _fwd(x: pd.Series, y: pd.Series, y12: pd.Series, h: int) -> dict:
    df = pd.DataFrame({"x": x, "y": y, "y12": y12}).dropna()
    if len(df) < ev.MIN_OBS:
        return {"n_obs": len(df), "t": np.nan, "beta_bp": np.nan, "p_chance": np.nan,
                "oos_r2": np.nan, "verdict": "short"}
    q = df.reset_index(names="month").assign(country="X", lag=LAG)
    got = ev._fit(q, h)
    if not got:
        return {"n_obs": len(df), "t": np.nan, "beta_bp": np.nan, "p_chance": np.nan,
                "oos_r2": np.nan, "verdict": "short"}
    return {
        "n_obs": int(got["n_obs"]), "t": float(got["t"]), "beta_bp": float(got["beta_bp"]),
        "p_chance": float(got["p_chance"]) if np.isfinite(got["p_chance"]) else np.nan,
        "oos_r2": float(got["oos_r2"]) if np.isfinite(got["oos_r2"]) else np.nan,
        "verdict": ev.verdict(got["beta_bp"], got["t"], got["loo_min_t"], got["p_chance"],
                             got["loo_max_t"], got["oos_r2"], ev.T_BAR),
    }


def _series(mkt: pd.DataFrame, country: str, name: str) -> pd.Series:
    g = mkt[(mkt["country"] == country) & (mkt["series"] == name)]
    return g.set_index("month")["value"].sort_index().asfreq("MS")


def _co_move(buy: pd.Series, y: pd.Series, start: str = "2010-01-01") -> tuple[float, int]:
    d12 = y - y.shift(12)
    j = pd.concat({"buy": buy, "d12": d12}, axis=1, sort=False).dropna()
    j = j[j.index >= start]
    if len(j) <= 12:
        return np.nan, len(j)
    return float(j["buy"].corr(j["d12"])), len(j)


def _stress_co(buy: pd.Series, spread: pd.Series, start: str = "2010-01-01") -> dict:
    """corr(buy, 12m Δspread) overall / high-stress tercile / rest."""
    d12 = spread - spread.shift(12)
    s = ev.pressure_pct(spread) / 10.0
    j = pd.concat({"buy": buy, "d12": d12, "s": s}, axis=1, sort=False).dropna()
    j = j[j.index >= start]
    out = {"co_r": np.nan, "co_hi": np.nan, "co_rest": np.nan, "n_co": len(j),
           "n_hi": 0, "n_rest": 0}
    if len(j) <= 12:
        return out
    out["co_r"] = float(j["buy"].corr(j["d12"]))
    hi = j[j["s"] >= ev.TERCILE]
    rest = j[j["s"] < ev.TERCILE]
    out["n_hi"], out["n_rest"] = len(hi), len(rest)
    if len(hi) > 12:
        out["co_hi"] = float(hi["buy"].corr(hi["d12"]))
    if len(rest) > 12:
        out["co_rest"] = float(rest["buy"].corr(rest["d12"]))
    return out


def ea_bank_rows(mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    """DE/FR/IT BSI 12m bank buying: forward + selloff co-move, country by country."""
    mkt = market.load() if mkt is None else mkt
    rows = eh.metrics()
    bank = rows[(rows["metric"] == "bsi_bank_f12m") & (rows["phase"] == "history")]
    sign = eh.METRICS["bsi_bank_f12m"]["sign"]
    out = []
    for c in ("DE", "FR", "IT"):
        raw = bank[bank["country"] == c].set_index("month")["value"].sort_index().asfreq("MS")
        x = ev.pressure_pct(raw * sign)
        targets = [("y10", "y10")]
        if c != "DE":
            targets.append(("spread_de", "spread_de"))
        for tgt, series in targets:
            y = _series(mkt, c, series)
            if tgt == "spread_de":
                co = _stress_co(raw, y)
            else:
                r, n = _co_move(raw, y)
                co = {"co_r": r, "co_hi": np.nan, "co_rest": np.nan, "n_co": n,
                      "n_hi": np.nan, "n_rest": np.nan}
            fw = ev.forward_changes(mkt[mkt["country"] == c], lags=(LAG,))
            for h in (3, 12):
                sub = fw[(fw["target"] == tgt) & (fw["h"] == h) & (fw["lag"] == LAG)]
                if sub.empty:
                    continue
                yy = sub.set_index("month")["y"]
                y12 = sub.set_index("month")["y12"]
                fit = _fwd(x.reindex(yy.index), yy, y12, h)
                out.append({
                    "country": c, "metric": "bsi_bank_f12m", "target": tgt,
                    "window": WINDOW, "horizon": h, **co, **fit,
                })
    return pd.DataFrame(out)


def uk_bank_from_cache() -> pd.DataFrame:
    """UK bank rows from ``uk_holders_evidence`` (quarterly co_move is the house number)."""
    p = CACHE / "uk_holders_evidence.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    b = df[df["metric"].eq("uk_banks_f4q")].copy()
    rows = []
    for _, r in b[b["test"].eq("main")].iterrows():
        rows.append({
            "country": "UK", "metric": "uk_banks_f4q", "target": r["target"],
            "window": r["window"], "horizon": int(r["horizon"]),
            "co_r": np.nan, "co_hi": np.nan, "co_rest": np.nan, "n_co": np.nan,
            "n_hi": np.nan, "n_rest": np.nan,
            "n_obs": r.get("n_obs"), "t": r.get("t"), "beta_bp": r.get("beta_bp"),
            "p_chance": r.get("p_chance"), "oos_r2": r.get("oos_r2"),
            "verdict": r.get("verdict"),
        })
    co = b[b["test"].eq("co_move")]
    for _, r in co.iterrows():
        rows.append({
            "country": "UK", "metric": "uk_banks_f4q", "target": r["target"],
            "window": r.get("window", WINDOW), "horizon": 0,
            "co_r": r.get("r12"), "co_hi": np.nan, "co_rest": np.nan,
            "n_co": r.get("n_obs"), "n_hi": np.nan, "n_rest": np.nan,
            "n_obs": r.get("n_obs"), "t": np.nan, "beta_bp": np.nan,
            "p_chance": np.nan, "oos_r2": np.nan, "verdict": "co_move",
        })
    return pd.DataFrame(rows)


def call(df: pd.DataFrame) -> dict[str, str]:
    """Bank role per country from the scored panel."""
    roles = {}
    # UK: inelastic if no forward link on y10/s1030/asw20 and co_move s1030 > 0
    uk = df[df["country"].eq("UK")]
    uk_fwd = uk[(uk["horizon"] > 0) & uk["target"].isin(("y10", "s1030", "asw20"))]
    uk_co = uk[(uk["horizon"] == 0) & uk["target"].eq("s1030")]
    no_fwd = uk_fwd.empty or (uk_fwd["verdict"] != "forward link").all()
    co = float(uk_co["co_r"].iloc[0]) if len(uk_co) else np.nan
    roles["UK"] = "inelastic" if (no_fwd and np.isfinite(co) and co >= 0.10) else "elastic"

    for c, tgt in (("DE", "y10"), ("FR", "spread_de"), ("IT", "spread_de")):
        sub = df[df["country"].eq(c)]
        fwd = sub[(sub["horizon"] == 12) & sub["target"].isin(("y10", "spread_de"))]
        co_row = sub[(sub["horizon"] == 12) & sub["target"].eq(tgt)]
        no_fwd = fwd.empty or (fwd["verdict"] != "forward link").all()
        co = float(co_row["co_r"].iloc[0]) if len(co_row) else np.nan
        # selloff absorption: positive co with cheapening/spread widen
        roles[c] = "inelastic" if (no_fwd and np.isfinite(co) and co >= 0.10) else "elastic"
    return roles


def inelastic_ea(roles: dict[str, str] | None = None) -> tuple[str, ...]:
    if roles is None:
        if OUT.exists():
            roles = call(pd.read_parquet(OUT))
        else:
            roles = call(build())
    return tuple(c.lower() for c in ("DE", "FR", "IT") if roles.get(c) == "inelastic")


def build() -> pd.DataFrame:
    return pd.concat([ea_bank_rows(), uk_bank_from_cache()], ignore_index=True)


def write() -> pd.DataFrame:
    res = build()
    res["computed"] = pd.Timestamp.today().normalize()
    res.to_parquet(OUT, index=False)
    return res


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    res = write()
    roles = call(res)
    print("=== EA banks country-by-country (BSI 12m) ===")
    ea = res[res["country"].isin(("DE", "FR", "IT")) & res["horizon"].eq(12)]
    print(ea[["country", "target", "co_r", "co_hi", "t", "verdict", "n_obs"]].round(3).to_string(index=False))
    print("\n=== UK banks (from uk_holders_evidence) ===")
    uk = res[res["country"].eq("UK")]
    print(uk[["target", "horizon", "co_r", "t", "verdict"]].round(3).to_string(index=False))
    print("\n=== CALL ===")
    for c, r in roles.items():
        print(f"  {c}: {r}")
    print("EA_BANKS_INELASTIC =", inelastic_ea(roles))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
