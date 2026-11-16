"""Investor view across the country cash frameworks: supply, plan tracking, skill, calendar.

Reads every `<cc>_cash_monthly.parquet` / `<cc>_cash_summary.json`, the US supply
bridge, and the issuance private-DV01 forward (`issuance/cache/float_share_fc.parquet`,
gross issuance DV01 net of the central-bank take). Scales to USD and % of GDP with
`data/scalers.json` (Norges Bank EXR cross rates; Eurostat / ONS / FRED nominal GDP,
trailing four quarters), refreshed on `refresh=True` and kept on a failed fetch.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sovereign_cash.contract import fmt_bn, fmt_num

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DERIVED = HERE / "derived"
SCALERS = HERE / "data" / "scalers.json"
BRIDGE = REPO / "us" / "fiscal" / "derived" / "supply_bridge.parquet"
US_GDP = REPO / "us" / "fiscal" / "data" / "cache" / "fred_gdp.csv"
FLOAT = REPO / "issuance" / "cache" / "float_share_fc.parquet"

CCY = {"us": "USD", "uk": "GBP", "de": "EUR", "fr": "EUR", "it": "EUR", "es": "EUR",
       "jp": "JPY", "se": "SEK", "no": "NOK"}
NAME = {"us": "US", "uk": "UK", "de": "Germany", "fr": "France", "it": "Italy", "es": "Spain",
        "jp": "Japan", "se": "Sweden", "no": "Norway"}
SYM = {"USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥", "SEK": "SEK ", "NOK": "NOK "}
# Overview / conviction board. Spain and Japan cash models feed debt/issuance via
# monthly() but are not yet full financing frameworks on this board.
ORDER = ("us", "uk", "de", "fr", "it", "se", "no")

# What the cash buffer does to central-bank reserves, and which instrument absorbs a
# fiscal surprise. Institutional facts, not model output.
CHANNEL = {
    "us": ("TGA at the Fed: every $ of TGA build drains reserves one-for-one.", "Bills (TBAC 15–20% share guidance)"),
    "uk": ("Exchequer sterilised by the DMO (bills, repo); no direct reserves channel.",
           "T-bill stock for debt financing, set at fiscal events; in-year surprises hit DMO net cash first"),
    "de": ("Bundesbank pays 0% on federal deposits since Oct-2023, so cash is placed in the market: "
           "no reserves channel, a money-market supply channel.", "Money-market placements first, then Bubills"),
    "fr": ("Treasury account at the BdF pinned at ~€0.1bn since May-2023; surplus placed in repo.",
           "BTF stock (AFT revises the BTF change in the financing table)"),
    "it": ("Treasury account at BdI + OPTES deposits: account moves change euro-area excess liquidity.",
           "BOT, and the Treasury liquidity balance (range ~35–70bn)"),
    "es": ("Treasury account at Banco de España; cash need cleared mainly by bond and letra issuance.",
           "Bonos/Obligaciones and Letras; no full monthly financing framework yet"),
    "jp": ("MoF general-account cash; issuance programme is annual, not a monthly buffer identity.",
           "JGBs clear the cash need on the thin seasonal path (bills held flat in the overlay)"),
    "se": ("Riksgälden liquidity management (repos, deposits, bills) offsets the budget day to day.",
           "Liquidity-management instruments and T-bill stock; SGB plan revised at the Feb/Jun/Oct reports"),
    "no": ("Government accounts sit at Norges Bank: fiscal flows drive structural liquidity, which Norges Bank "
           "offsets with F-loans/F-deposits; NOWA sits on the policy rate.",
           "None: petroleum-fund transfers cover the non-oil deficit; NGB issuance is a fixed programme"),
}


# ---------------------------------------------------------------- scalers
def _nb_fx() -> dict[str, float]:
    url = ("https://data.norges-bank.no/api/data/EXR/B.USD+EUR+SEK+GBP.NOK.SP"
           "?format=csv&lastNObservations=1&locale=en")
    rows = list(csv.DictReader(io.StringIO(requests.get(url, timeout=30).text), delimiter=";"))
    nok = {r["BASE_CUR"]: float(r["OBS_VALUE"]) / (10 ** int(r["UNIT_MULT"])) for r in rows}
    usd = nok["USD"]
    return {"USD": 1.0, "NOK": 1 / usd, "EUR": nok["EUR"] / usd, "SEK": nok["SEK"] / usd, "GBP": nok["GBP"] / usd,
            "date": rows[0]["TIME_PERIOD"]}


def _gdp() -> dict[str, float]:
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/namq_10_gdp?format=JSON"
           "&geo=DE&geo=FR&geo=IT&geo=SE&geo=NO&unit=CP_MNAC&s_adj=SCA&na_item=B1GQ&sinceTimePeriod=2024-Q1")
    d = requests.get(url, timeout=60).json()
    geo = {v: k for k, v in d["dimension"]["geo"]["category"]["index"].items()}
    tix = d["dimension"]["time"]["category"]["index"]
    n = len(tix)
    vals: dict[str, dict[int, float]] = {}
    for k, v in d["value"].items():
        k = int(k)
        vals.setdefault(geo[k // n], {})[k % n] = v
    out = {g.lower(): sum(v[i] for i in sorted(v)[-4:]) / 1e3 for g, v in vals.items()}
    q = requests.get("https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/ybha/qna/data",
                     timeout=60).json()["quarters"]
    out["uk"] = sum(float(r["value"]) for r in q[-4:]) / 1e3
    g = pd.read_csv(US_GDP)
    out["us"] = float(g.iloc[-1, -1])
    # Norway on mainland GDP: total GDP is ~25% offshore petroleum, which the oil fund, not NGBs, finances
    q = {"query": [{"code": "Makrost", "selection": {"filter": "item", "values": ["bnpb.nr23_9fn"]}},
                   {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Priser"]}},
                   {"code": "Tid", "selection": {"filter": "top", "values": ["4"]}}],
         "response": {"format": "json-stat2"}}
    out["no"] = sum(requests.post("https://data.ssb.no/api/v0/en/table/09190", json=q, timeout=60).json()["value"]) / 1e3
    return out


def scalers(refresh: bool = False) -> dict:
    old = json.loads(SCALERS.read_text()) if SCALERS.exists() else {}
    if not refresh and old:
        return old
    new = dict(old)
    for key, fn in (("fx_usd_per_unit", _nb_fx), ("gdp_bn", _gdp)):
        try:
            new[key] = fn()
        except Exception as e:  # noqa: BLE001
            print(f"scalers: {key} fetch failed ({e}); keeping cached")
    SCALERS.parent.mkdir(parents=True, exist_ok=True)
    SCALERS.write_text(json.dumps(new, indent=1))
    return new


# ---------------------------------------------------------------- inputs
def monthly(cc: str, naive_tail: bool = False) -> pd.DataFrame | None:
    if cc == "us":
        if not BRIDGE.exists():
            return None
        b = pd.read_parquet(BRIDGE).loc["unchanged"].copy()
        b.index = pd.to_datetime(b.index)
        b = b.iloc[1:]  # origin month is partial history
        return pd.DataFrame({
            "month": b.index, "is_forecast": True, "cash_requirement_bn": b["deficit_bn"].values,
            "net_bonds_bn": b["net_coupons_bn"].values, "net_bills_bn": b["private_net_bills_bn"].values,
            "bond_redemption_bn": b["coupon_red_bn"].values, "buffer_change_bn": b["d_tga_bn"].values,
            # SOMA rolls all maturing Treasuries at auction as add-ons, already outside net coupons
            "bond_redemption_cb_bn": 0.0})
    p = DERIVED / f"{cc}_cash_monthly.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["month"] = pd.to_datetime(d["month"])
    d["is_forecast"] = d["is_forecast"].astype(bool)
    if "cash_observed" in d:
        # a month whose cash requirement is published but whose financing is not (DE) is history
        # for the forward window, as in the country's own summary
        d["is_forecast"] &= ~d["cash_observed"].fillna(False).astype(bool)
    if cc == "no" and not naive_tail:
        # beyond Norges Bank's vintage the cash requirement is a same-month-last-year copy
        # that does not beat zero OOS; it is not shown as a forecast
        d = d[~(d["is_forecast"] & d["fc_source"].astype(str).str.startswith("same_month"))]
    return d.sort_values("month").reset_index(drop=True)


def summary(cc: str) -> dict:
    p = DERIVED / f"{cc}_cash_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- live vs seasonal labels
# Every page that names a cash model reads these, so a PRE-selected seasonal tag kept in a summary
# (``forward.model``, ``cgncr_model``, ``production_model``) is never shown as live once a bottom-up ships.
def bu_ship(cc: str, s: dict) -> bool:
    """Whether the country's bottom-up path is shipped by its PRE gate (summary flags only)."""
    fw = s.get("forward") or {}
    if cc == "uk":
        return bool(fw.get("cgncr_bottomup_ship"))
    if cc in ("de", "fr"):
        return bool(fw.get("bottomup_macro_ship") or fw.get("bottomup_ship") or fw.get("bottom_up_ship"))
    if cc == "it":
        return bool(fw.get("bottomup_ship"))
    if cc == "se":
        return bool((s.get("bottom_up_macro") or {}).get("ship") or (s.get("bottom_up") or {}).get("ship"))
    if cc == "no":
        return bool((s.get("bottom_up") or {}).get("ship"))
    return False


def seasonal_tag(cc: str, s: dict) -> str | None:
    """The PRE-selected (or a-priori scoreboard) seasonal model the summary carries, if any."""
    fw = s.get("forward") or {}
    tag = {"uk": fw.get("cgncr_model"), "de": fw.get("model"), "fr": fw.get("model"), "it": fw.get("model"),
           "se": (s.get("bottom_up") or s.get("bottom_up_macro") or {}).get("production_model"),
           "no": "smly" if s else None}.get(cc)
    return str(tag) if tag else None


def seasonal_role(cc: str, s: dict) -> str | None:
    """What the seasonal tag does on the live path: a live role, or 'seasonal fallback / scoreboard (not live)'."""
    if not seasonal_tag(cc, s):
        return None
    if bu_ship(cc, s):
        return {"de": "seasonal fallback / scoreboard, not live: open-year core runs on the bottom-up, later years on "
                      "the component model",
                "fr": "seasonal fallback / scoreboard, not live: open year runs on the PLF×RESF bottom-up, later years "
                      "on the component model",
                "no": "fallback beyond Norges Bank's vintage, not live: the Tab4 bottom-up ships there",
                }.get(cc, "seasonal fallback / scoreboard, not live: the bottom-up ships")
    return {"uk": "live: CGNCR to end-March (OBR profile after)",
            "de": "live: core seasonal path (bottom-up not shipped)",
            "fr": "live: seasonal shape under the PLF deficit (bottom-up not shipped)",
            "it": "live: month shape under the DFP annual pin (component model for later years)",
            "se": "live: month shape under the Riksgälden annual NBR",
            "no": "live beyond Norges Bank's vintage (bottom-up not shipped)"}.get(cc)


def seasonal_label(cc: str, s: dict) -> str:
    """'<tag> (<role>)' for tables and captions; empty when the country carries no seasonal tag."""
    tag, role = seasonal_tag(cc, s), seasonal_role(cc, s)
    return f"{tag} ({role})" if tag else ""


def private_dv01() -> pd.DataFrame:
    """Monthly forward private duration supply by country: USD mn/bp, and the private DV01 float."""
    if not FLOAT.exists():
        return pd.DataFrame()
    f = pd.read_parquet(FLOAT)
    f = f[f["phase"] != "history"]
    g = f.groupby(["country", "month"], as_index=False)[["flow_dv01", "private_stock_dv01"]].sum()
    g["cc"] = g["country"].str.lower()
    g["month"] = pd.to_datetime(g["month"])
    return g


# ---------------------------------------------------------------- tables
@dataclass
class Row:
    cc: str
    window: str
    n_months: int
    cash_req: float
    net_bonds: float
    net_bills: float
    redemptions: float
    private_net_bonds: float
    dv01_12m: float | None
    dv01_float_pct: float | None


def supply(sc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward 12m per country plus a common-calendar monthly panel (% GDP)."""
    dv = private_dv01()
    rows, panel = [], []
    for cc in ORDER:
        d = monthly(cc)
        if d is None:
            continue
        f = d[d["is_forecast"]].head(12)
        if f.empty:
            continue
        gdp = sc.get("gdp_bn", {}).get(cc, np.nan)
        dvc = dv[dv["cc"] == cc].sort_values("month").head(12) if not dv.empty else pd.DataFrame()
        rows.append(Row(
            cc, f"{f['month'].iloc[0]:%b-%y}–{f['month'].iloc[-1]:%b-%y}", len(f),
            f["cash_requirement_bn"].sum(), f["net_bonds_bn"].sum(), f["net_bills_bn"].sum(),
            f["bond_redemption_bn"].sum() if "bond_redemption_bn" in f else np.nan,
            (f["net_bonds_bn"] + f["bond_redemption_cb_bn"].fillna(0)).sum() if "bond_redemption_cb_bn" in f else np.nan,
            float(dvc["flow_dv01"].sum()) if len(dvc) else None,
            float(100 * dvc["flow_dv01"].sum() / dvc["private_stock_dv01"].iloc[0]) if len(dvc) else None,
        ).__dict__)
        for _, r in f.iterrows():
            panel.append({"cc": cc, "month": r["month"],
                          "bills_pct_gdp": 100 * r["net_bills_bn"] / gdp,
                          "bonds_pct_gdp": 100 * r["net_bonds_bn"] / gdp,
                          "cash_req_pct_gdp": 100 * r["cash_requirement_bn"] / gdp})
    out = pd.DataFrame(rows)
    fx = sc.get("fx_usd_per_unit", {})
    for c in ("cash_req", "net_bonds", "net_bills", "private_net_bonds"):
        out[f"{c}_usd"] = [v * fx.get(CCY[cc], np.nan) for cc, v in zip(out["cc"], out[c])]
        out[f"{c}_pct_gdp"] = [100 * v / sc.get("gdp_bn", {}).get(cc, np.nan) for cc, v in zip(out["cc"], out[c])]
    out["bill_share"] = out["net_bills"] / (out["net_bills"] + out["net_bonds"])
    return out, pd.DataFrame(panel)


def tracking() -> list[dict]:
    """Year-to-date cash requirement vs the official plan, and what the model implies for the rest."""
    out = []
    u = summary("uk").get("forward", {}).get("end_march", {})
    if u:
        out.append({"cc": "uk", "year": f"FY{u['fy']}", "ytd": u["cgncr_ex_ytd_bn"], "ytd_through": u["ytd_through"][:7],
                    "plan": u["cgncr_ex_plan_bn"], "plan_src": "OBR / DMO remit CGNCR (ex NRAM etc.)",
                    "model": u["cgncr_ex_model_fy_bn"],
                    "lands_on": f"bills solved to plan net cash {fmt_bn(u['net_cash_plan_end_march_bn'], '£')} end-March; "
                                f"next remit revision {fmt_bn(u.get('next_revision_gap_bn', 0), '£', plus=True)} financing "
                                f"(NS&I {fmt_num(u.get('next_revision_gap_parts_bn', {}).get('nsandi', 0), plus=True)})"})
    have = {r["cc"] for r in out}
    for cc in ORDER:
        fw = summary(cc).get("forward", {})
        if cc in have or fw.get("plan_cash_requirement_bn") is None:
            continue
        out.append({"cc": cc, "year": str(fw.get("plan_year", "")), "ytd": fw["cash_requirement_ytd_bn"],
                    "ytd_through": str(fw.get("ytd_through", ""))[:7], "plan": fw["plan_cash_requirement_bn"],
                    "plan_src": fw.get("plan_source", ""), "model": fw["model_full_year_bn"],
                    "lands_on": fw.get("plan_gap_lands_on", "bills / cash buffer")})
    out.sort(key=lambda r: ORDER.index(r["cc"]))
    for r in out:
        r["gap"] = r["model"] - r["plan"]
    return out


def _verdict(m: float, b: float, vs: str = "naive") -> str:
    return (f"beats {vs}" if m < 0.97 * b else (f"ties {vs}" if m <= 1.03 * b else f"does not beat {vs}"))


MULTI_YEAR = DERIVED / "multi_year_backtest.json"

# What drives each forecast month of ``cash_requirement_bn``, keyed by ``_seg_key``. Seasonal tags appear
# here only where that model is what the board actually forecasts with.
SEGMENT = {
    "us": {"dts": "DTS flow model (calendar shape × trailing level), TGA steered to refunding target"},
    "uk": {"model": "{cgncr_model} seasonal CGNCR model to end-March",
           "official": "OBR / DMO official CGNCR monthly profile after March"},
    "de": {"bu": "Bundeshaushalt Soll × Ist/Soll blend (bottom-up, open year)",
           "seasonal": "{model} Soll × haircut (seasonal fallback)",
           "comp": "component model (lines as shares of the GDP / price level, interest from the debt book)"},
    "fr": {"bu": "PLF×RESF categories soft-pinned to the PLF solde (bottom-up, open year)",
           "seasonal": "{model} seasonal × PLF deficit (seasonal fallback)",
           "comp": "component model (SMB lines on the RESF GDP / price levels, interest from the debt book, "
                   "legislated overlay)"},
    "it": {"plan": "DFP total × same-month-last-year shape (open year)",
           "comp": "component model (cash-account lines on the macro paths, interest from the debt book, RRF grants)"},
    "se": {"rg": "Riksgälden annual NBR × PRE-selected month shape ({production_model})"},
    "no": {"nb": "Norges Bank structural-liquidity residual (inside its vintage)",
           "bu": "Tab4 categories × Brent / mainland GDP (bottom-up, beyond Norges Bank's vintage)",
           "smly": "same month last year"},
}
SEGMENT_SUFFIX = {"de": "special funds at their YTD run rate"}
SEGMENT_TAG = {"bu": "BU", "comp": "component", "seasonal": "seasonal fallback", "plan": "DFP pin",
               "model": "seasonal", "official": "OBR profile", "nb": "NB", "smly": "smly"}


def _seg_key(cc: str, r: pd.Series) -> str:
    src = str(r.get("fc_source", ""))
    if cc == "us":
        return "dts"
    if cc == "uk":
        return "official" if bool(r.get("cgncr_from_plan")) else "model"
    if cc in ("de", "fr"):
        if "component model" in src:
            return "comp"
        return "bu" if ("bottom-up" in src or "PLF×RESF" in src) else "seasonal"
    if cc == "it":
        return "comp" if r.get("cash_requirement_ext_source") == "component model" else "plan"
    if cc == "se":
        return "rg"
    if cc == "no":
        return "nb" if src.startswith("norges_bank") else ("bu" if src.startswith("bottom_up") else "smly")
    return "model"


def segments(cc: str, n: int = 12) -> list[dict]:
    """Runs of the next n forecast months by what drives their cash requirement (same months as ``supply``)."""
    d = monthly(cc)
    if d is None:
        return []
    out: list[dict] = []
    for _, r in d[d["is_forecast"]].head(n).iterrows():
        k = _seg_key(cc, r)
        if out and out[-1]["key"] == k:
            out[-1]["last"], out[-1]["n"] = r["month"], out[-1]["n"] + 1
        else:
            out.append({"key": k, "first": r["month"], "last": r["month"], "n": 1})
    return out


def _mean(d: dict | None) -> float | None:
    v = [float(x) for x in (d or {}).values() if isinstance(x, (int, float))]
    return float(np.mean(v)) if v else None


def _by_h_mean(by_h: dict | None, key: str) -> float | None:
    return _mean({h: r.get(key) for h, r in (by_h or {}).items()})


def _seg_score(cc: str, key: str, s: dict) -> tuple[str, str] | None:
    """(score text, verdict) for one live segment, against the benchmark that segment was gated on."""
    fw = s.get("forward") or {}
    if key == "dts":
        p = REPO / "us" / "fiscal" / "derived" / "fiscal_path_score.json"
        r = next((x for x in json.loads(p.read_text()).get("score", [])
                  if x.get("window") == "OOS" and x.get("h") == 250), None) if p.exists() else None
        return (f"250-bday cum. op-flow MAE {r['model']:.0f} vs last-year {r['last_year']:.0f}",
                _verdict(r["model"], r["last_year"])) if r else None
    if cc == "uk" and key == "model":
        r = next((x for x in s.get("score", []) if x.get("window") == "OOS" and x.get("h") == 12
                  and x.get("benchmark_name") == "smly"), None)
        return (f"h=12 MAE {r['model']:.1f} vs smly {r['benchmark']:.1f}", _verdict(r["model"], r["benchmark"])) if r else None
    if cc == "uk" and key == "official":
        r = next((x for x in s.get("official_benchmark", []) if x.get("window") == "OOS" and x.get("h") == 12), None)
        return (f"OBR vintage h=12 MAE {r['official_vintage']:.1f} vs seasonal {r[fw.get('cgncr_model', '')]:.1f} "
                f"(n={r['n']})", "official anchor") if r and fw.get("cgncr_model") in r else None
    if key == "seasonal":
        r = next((x for x in s.get("score", []) if x.get("window") == "OOS" and x.get("h") == 12
                  and isinstance(x.get("model"), (int, float))), None)
        return (f"seasonal fallback h=12 MAE {r['model']:.1f} vs {r.get('benchmark_name', '')} {r['benchmark']:.1f}",
                _verdict(r["model"], r["benchmark"])) if r else None
    if key == "comp":
        my = json.loads(MULTI_YEAR.read_text()).get(cc, {}).get("summary", {}) if MULTI_YEAR.exists() else {}
        f_, n_ = (my.get("final") or {}).get("h1"), (my.get("naive") or {}).get("h1")
        if not f_ or not n_:
            return None
        m, b = f_["mae_total_ex_covid"], n_["mae_total_ex_covid"]
        return (f"year-ahead annual ex-interest MAE {m:.1f} vs naive share {b:.1f} (n={f_['n']}, realised macro)",
                _verdict(m, b) + " (realised macro)")
    if cc == "fr" and key == "bu":
        bu, prod = fw.get("bottomup_oos_mean"), fw.get("bottomup_oos_production_mean")
        pre, pre_p = fw.get("bottomup_pre_mean"), fw.get("bottomup_pre_production_mean")
        if bu is None or prod is None:
            return None
        return (f"mean MAE h1–12 OOS {bu:.1f} vs seasonal fallback {fw.get('model', '')} {prod:.1f}"
                + (f"; PRE {pre:.1f} vs {pre_p:.1f}" if pre is not None and pre_p is not None else ""),
                _verdict(bu, prod, "seasonal fallback"))
    if cc == "de" and key == "bu":
        # The macro BU was gated on the prior BU (v1), and v1 on the Soll × haircut fallback.
        v1, ph = _by_h_mean(fw.get("bottomup_oos_by_h"), "bottomup"), _by_h_mean(fw.get("bottomup_oos_by_h"), "planhc")
        mac, mac_v1 = fw.get("bottomup_macro_oos_mean"), _by_h_mean(fw.get("bottomup_macro_oos_by_h"), "production")
        if v1 is None or ph is None:
            return None
        if fw.get("bottomup_macro_ship") and mac is not None and mac_v1 is not None:
            txt = (f"open-year core mean MAE h1–12 OOS {mac:.1f} vs prior BU {mac_v1:.1f}; prior BU {v1:.1f} vs "
                   f"seasonal fallback {fw.get('model', '')} {ph:.1f}")
            verd = _verdict(v1, ph, "seasonal fallback") if mac <= 1.03 * mac_v1 else _verdict(mac, mac_v1, "prior BU")
        else:
            txt = f"open-year core mean MAE h1–12 OOS {v1:.1f} vs seasonal fallback {fw.get('model', '')} {ph:.1f}"
            verd = _verdict(v1, ph, "seasonal fallback")
        return txt, verd
    if cc == "it" and key == "plan":
        r = next((x for x in s.get("score", []) if x.get("window") == "OOS" and x.get("h") == 12
                  and x.get("model") == "smly"), None)
        return (f"level = DFP plan; shape = smly (h=12 MAE {r['mae_bn']:.1f}, the naive itself)" if r
                else "level = DFP plan; shape = smly", "official anchor")
    if cc == "se":
        r = next((x for x in s.get("shipped_forward_score", []) if x.get("vintage") == "pooled"), None)
        return (f"shipped path MAE {r['shipped']:.1f} vs smly {r['smly']:.1f} (n={r['n']} months, model alone "
                f"{r['model']:.1f})", _verdict(r["shipped"], r["smly"])) if r else None
    if cc == "no" and key == "nb":
        sc = s.get("score", [])
        r = (next((x for x in sc if x.get("window") == "OOS" and x.get("h") == "m+1"), None)
             or next((x for x in sc if x.get("window") == "OOS" and x.get("h") == "20bd"), None))
        return (f"NB {r['h']} MAE {r['model']:.1f} vs {r['benchmark']:.1f}", "official anchor") if r else None
    if cc == "no" and key == "bu":
        bu = s.get("bottom_up") or {}
        o, p = bu.get("oos_mean_mae") or {}, bu.get("pre_mean_mae") or {}
        n = ((bu.get("oos_beyond_nb") or {}).get("12") or {}).get("n")
        if o.get("bottom_up") is None or o.get("production_smly") is None:
            return None
        return (f"beyond-NB mean MAE OOS {o['bottom_up']:.1f} vs smly {o['production_smly']:.1f}"
                + (f" (n={n})" if n else "")
                + (f"; PRE {p['bottom_up']:.1f} vs {p['production_smly']:.1f}" if p.get("bottom_up") is not None else ""),
                _verdict(o["bottom_up"], o["production_smly"]))
    return None


def _horizon_mean(block: dict | None, key: str) -> float | None:
    vals = [float(row[key]) for row in (block or {}).values()
            if isinstance(row, dict) and isinstance(row.get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def _de_live_skill(fw: dict) -> dict | None:
    """Live German core is bottom-up macro. planhc_3 vs smly is the score-table residual.

    planhc_3 is not PRE-gated (17 h=12 windows, loses to prevyear_3). Its OOS win
    against same-month-last-year is not the forecast on this page.
    """
    if fw.get("live_core_path") != "bottom-up macro" or not fw.get("bottomup_macro_ship"):
        return None
    pre, prod = fw.get("bottomup_macro_pre_mean"), fw.get("bottomup_macro_pre_production_mean")
    oos = fw.get("bottomup_macro_oos_mean")
    oos_prior = _horizon_mean(fw.get("bottomup_macro_oos_by_h"), "production")
    if not all(isinstance(v, (int, float)) for v in (pre, prod, oos, oos_prior)):
        return None
    return {
        "cc": "de",
        "basis": "bottom-up macro (live core). planhc_3 is the score-table residual, not this forecast",
        "oos": (
            f"PRE mean MAE {pre:.1f} vs prior path {prod:.1f} (the gate). "
            f"OOS mean {oos:.1f} vs prior path {oos_prior:.1f}, reported, not used to choose. "
            "planhc_3 vs smly is not this path"
        ),
        "verdict": "PRE vs prior path",
    }


def _bu_research(cc: str, s: dict) -> tuple[str | None, str | None]:
    """(tier, detail) of the bottom-up candidate where it is not the live path. Edge > 0 = BU better."""
    fw = s.get("forward") or {}

    def _edge(bu, prod) -> str:
        e = (prod - bu) / prod
        return f"BU {abs(100 * e):.0f}% {'better' if e > 0 else 'worse'}"

    if cc == "uk":
        pre, prod = _mean(fw.get("cgncr_bottomup_pre")), _mean(fw.get("cgncr_bottomup_prod_pre"))
        oos, prod_o = _mean(fw.get("cgncr_bottomup_oos")), _mean(fw.get("cgncr_bottomup_prod_oos"))
        if pre is None or prod is None:
            return None, None
        return "LOW", (f"ONS/EFO categories × ready-reckoner (soft remit pin α=0.12): PRE mean MAE {pre:.1f} vs live "
                       f"{prod:.1f} ({_edge(pre, prod)}); OOS {oos:.1f} vs {prod_o:.1f}. "
                       f"{fw.get('cgncr_bottomup_reason', '')}")
    if cc == "it":
        pre, prod = fw.get("bottomup_pre_mean"), fw.get("bottomup_pre_production_mean")
        oos, prod_o = fw.get("bottomup_oos_mean"), fw.get("bottomup_oos_production_mean")
        if pre is None or prod is None:
            return None, None
        return "LOW", (f"BdI Tavola 1 × NGDP/HICP: PRE mean MAE {pre:.1f} vs live {prod:.1f} ({_edge(pre, prod)}); "
                       f"OOS {oos:.1f} vs {prod_o:.1f}. Fails the PRE gate.")
    if cc == "se":
        b = s.get("bottom_up_macro") or s.get("bottom_up") or {}
        pre, oos = b.get("pre_mean_mae_anchored") or {}, b.get("oos_mean_mae_anchored") or {}
        if pre.get("bottom_up") is None:
            return None, None
        return "LOW", (f"SCB categories × macro, RG-anchored: PRE mean MAE {pre['bottom_up']:.1f} vs live "
                       f"{pre['production']:.1f} ({_edge(pre['bottom_up'], pre['production'])}); OOS "
                       f"{oos.get('bottom_up', float('nan')):.1f} vs {oos.get('production', float('nan')):.1f}. "
                       "Fails the PRE gate.")
    if cc in ("de", "fr", "no"):
        reason = " ".join(str(x or "") for x in (fw.get("bottomup_reason"), fw.get("bottomup_macro_reason"),
                                                 (s.get("bottom_up") or {}).get("ship_reason")))
        return "LOW", ("Bottom-up shipped by its PRE gate but drives no month in the window"
                       + (" (FORCE_LIVE quarantine)" if "FORCE_LIVE" in reason else "") + ".")
    return None, None


def live_cash_path(cc: str) -> dict:
    """What drives the live ``cash_requirement_bn`` over the next twelve forecast months, and its score.

    ``basis`` names each source with its months; ``oos`` scores each against the benchmark it was gated on;
    ``verdict`` is the verdict of the source covering most of the window. ``bu_tier`` / ``bu_detail`` describe a
    bottom-up candidate only when it is not live.
    """
    if cc == "us":
        us_path = REPO / "us" / "fiscal" / "derived" / "fiscal_path_score.json"
        if not us_path.exists():
            return {}
        sc = json.loads(us_path.read_text())
        r = next((x for x in sc.get("score", []) if x.get("window") == "OOS" and x.get("h") == 250), None)
        if not r:
            return {}
        return {
            "cc": "us",
            "basis": "DTS flow model (calendar shape × trailing level), TGA steered to refunding target",
            "segments": [], "oos": f"250-bday cum. op-flow MAE {r['model']:.0f} vs last-year {r['last_year']:.0f}",
            "verdict": _verdict(r["model"], r["last_year"]), "shipped_bu": False, "bu_ship_flag": False,
            "bu_tier": None, "bu_detail": None, "seasonal_tag": None, "seasonal_role": None, "seasonal_live": False,
        }
    if cc == "de":
        live = _de_live_skill(summary(cc).get("forward") or {})
        if live:
            segs = segments(cc)
            live = {**live, "segments": segs, "shipped_bu": any(g["key"] == "bu" for g in segs),
                    "bu_ship_flag": True, "bu_tier": None, "bu_detail": None,
                    "seasonal_tag": seasonal_tag(cc, summary(cc)), "seasonal_role": seasonal_role(cc, summary(cc)),
                    "seasonal_live": False}
            return live
    s = summary(cc)
    segs = segments(cc)
    if not segs:
        return {}
    fw = s.get("forward") or {}
    fmt = {"cgncr_model": fw.get("cgncr_model", ""), "model": fw.get("model", ""),
           "production_model": (s.get("bottom_up") or {}).get("production_model", "PRE-selected")}
    parts, scores = [], []
    for g in segs:
        win = (f"{g['first']:%b-%y}" if g["n"] == 1 else f"{g['first']:%b-%y}–{g['last']:%b-%y}")
        g["label"] = SEGMENT[cc][g["key"]].format(**fmt)
        parts.append(f"{g['label']} [{win}]")
        sc_ = _seg_score(cc, g["key"], s)
        g["oos"], g["verdict"] = sc_ if sc_ else (None, None)
        if sc_:
            tag = SEGMENT_TAG.get(g["key"])
            scores.append(f"{tag}: {sc_[0]}" if tag and len(segs) > 1 else sc_[0])
    basis = "; then ".join(parts) + (f"; {SEGMENT_SUFFIX[cc]}" if cc in SEGMENT_SUFFIX else "")
    main = max(segs, key=lambda g: (g["n"], -segs.index(g)))
    shipped = any(g["key"] == "bu" for g in segs)
    bu_tier, bu_detail = (None, None)
    flag = bu_ship(cc, s)
    if not shipped and cc != "us" and (cc in ("uk", "it", "se") or flag):
        bu_tier, bu_detail = _bu_research(cc, s)
    tag, role = seasonal_tag(cc, s), seasonal_role(cc, s)
    return {"cc": cc, "basis": basis, "segments": segs, "oos": "; ".join(scores) or "—",
            "verdict": main.get("verdict") or "—", "shipped_bu": shipped, "bu_ship_flag": flag,
            "bu_tier": bu_tier, "bu_detail": bu_detail, "seasonal_tag": tag, "seasonal_role": role,
            "seasonal_live": bool(tag) and not flag}


def skill() -> list[dict]:
    """What the forward cash requirement is, and whether it beats its benchmark OOS (from ``live_cash_path``)."""
    return [{k: p[k] for k in ("cc", "basis", "oos", "verdict")} for p in map(live_cash_path, ORDER) if p]


def conviction() -> list[dict]:
    """Per-country cash-path conviction for the overview (BU ship / authority)."""
    out = []
    for cc in ORDER:
        s = summary(cc)
        fw = s.get("forward") or {}
        bu = s.get("bottom_up_macro") or s.get("bottom_up") or {}
        if cc == "us":
            out.append({"cc": cc, "tier": "HIGH",
                        "path": "DTS Table II buckets × calendar; TGA steered to TBAC",
                        "edge": "OOS beats last-year / flat on operating flow",
                        "note": "Open refunding target is hand-maintained"})
            continue
        ship = (fw.get("cgncr_bottomup_ship") if cc == "uk"
                else fw.get("bottomup_macro_ship", fw.get("bottom_up_ship",
                           fw.get("bottomup_ship", bu.get("ship")))))
        if ship is None and cc == "no":
            ship = bu.get("ship")
        if cc == "se":
            ship_score = s.get("shipped_forward_score") or []
            pooled = next((r for r in ship_score if r.get("vintage") == "pooled"), None)
            edge = None
            if pooled and pooled.get("smly"):
                edge = (pooled["smly"] - pooled["shipped"]) / pooled["smly"]
            out.append({
                "cc": cc, "tier": "HIGH",
                "path": "Riksgälden annual NBR × PRE-selected month shape (authority)",
                "edge": (f"OOS MAE {pooled['shipped']:.1f} vs smly {pooled['smly']:.1f} "
                         f"({100 * edge:.0f}% under smly)" if pooled and edge is not None else "official anchor"),
                "note": "Category×macro fails RG-anchored PRE gate — not shipped",
            })
            continue
        if cc == "de":
            pre = fw.get("bottomup_macro_pre_mean") or fw.get("bottomup_pre_mean")
            prod = fw.get("bottomup_macro_pre_production_mean") or fw.get("bottomup_pre_production_mean")
            oos = fw.get("bottomup_macro_oos_mean") or fw.get("bottomup_oos_mean")
            e = ((prod - pre) / prod) if pre and prod else None
            out.append({
                "cc": cc, "tier": "HIGH" if ship else "MED",
                "path": "Open year: Bundeshaushalt Soll × Ist/Soll blend; later: smly×macro",
                "edge": (f"PRE {pre:.1f} vs prod {prod:.1f} ({100 * e:.0f}%); OOS {oos:.1f}"
                         if e is not None else str(fw.get("bottomup_macro_reason") or "")[:80]),
                "note": "Soll open-year is printed at origin — not a MoF-free call",
            })
            continue
        if cc == "no":
            pre = (bu.get("pre_mean_mae") or {}).get("bottom_up")
            prod = (bu.get("pre_mean_mae") or {}).get("production_smly")
            oos = (bu.get("oos_mean_mae") or {}).get("bottom_up")
            e = ((prod - pre) / prod) if pre and prod else None
            out.append({
                "cc": cc, "tier": "HIGH" if ship else "MED",
                "path": "Inside NB vintage: NB residual; beyond: Tab4×macro (SHIP)",
                "edge": (f"beyond-NB PRE {pre:.1f} vs smly {prod:.1f} ({100 * e:.0f}%); OOS {oos:.1f}"
                         if e is not None else bu.get("ship_reason", "")),
                "note": "Does not try to beat NB inside its published window",
            })
            continue
        if cc == "uk":
            pre = fw.get("cgncr_bottomup_pre") or {}
            prod = fw.get("cgncr_bottomup_prod_pre") or {}
            oos = fw.get("cgncr_bottomup_oos") or {}
            pre_m = float(np.mean(list(pre.values()))) if pre else None
            prod_m = float(np.mean(list(prod.values()))) if prod else None
            oos_m = float(np.mean(list(oos.values()))) if oos else None
            e = ((prod_m - pre_m) / prod_m) if pre_m and prod_m else fw.get("cgncr_bottomup_edge")
            out.append({
                "cc": cc, "tier": "HIGH" if ship and e is not None and e >= 0 else ("MED" if ship else "LOW"),
                "path": "ONS/EFO categories × ready-reckoner (soft remit pin α=0.12)",
                "edge": (f"PRE mean {pre_m:.1f} vs prod {prod_m:.1f} ({100 * e:.0f}%); OOS {oos_m:.1f}"
                         if e is not None and pre_m is not None else str(fw.get("cgncr_bottomup_reason", ""))[:80]),
                "note": "Research column only (SHIP=False)" if not ship else "Soft-pin α=0.12",
            })
            continue
        pre = fw.get("bottomup_pre_mean") or fw.get("bottomup_macro_pre_mean")
        prod = fw.get("bottomup_pre_production_mean") or fw.get("bottomup_macro_pre_production_mean")
        oos = fw.get("bottomup_oos_mean") or fw.get("bottomup_macro_oos_mean")
        e = ((prod - pre) / prod) if pre and prod else None
        tier = "HIGH" if (ship and e is not None and e >= 0.06) else ("HIGH" if ship and e and e >= 0.05 else ("MED" if ship else "LOW"))
        if cc == "it" and e is not None and e >= 0.06:
            tier = "HIGH"
        path = {
            "fr": "PLF×RESF categories (no LFI solde re-pin)",
            "it": "BdI Tavola 1 × NGDP/HICP (DFP share off)",
        }.get(cc, "bottom-up macro")
        out.append({
            "cc": cc, "tier": tier if ship else "LOW",
            "path": path,
            "edge": (f"PRE {pre:.1f} vs prod {prod:.1f} ({100 * e:.0f}%); OOS {oos:.1f}"
                     if e is not None else str(fw.get("bottomup_reason") or bu.get("ship_reason") or "")[:80]),
            "note": "SHIP" if ship else "research only",
        })
    return out


def calendar(n_months: int = 6, top: int = 2, min_pct_gdp: float = 0.05) -> list[dict]:
    """Largest bond-redemption and cash-requirement months in the next n months, per country.

    Redemptions come from the maturity schedule, so they are listed past any forecast
    horizon (Norway beyond Norges Bank's vintage); cash requirement only where forecast.
    """
    out = []
    now = pd.Timestamp.today().to_period("M").to_timestamp()
    end = now + pd.DateOffset(months=n_months)
    gdp = scalers().get("gdp_bn", {})
    for cc in ORDER:
        d = monthly(cc)
        full = monthly(cc, naive_tail=True)
        if d is None:
            continue
        floor = max(0.05, min_pct_gdp / 100 * gdp.get(cc, 0) or 0)
        for src, col, label in ((full, "bond_redemption_bn", "bond redemption"),
                                (d, "cash_requirement_bn", "cash requirement")):
            f = src[src["is_forecast"] & (src["month"] >= now) & (src["month"] < end)]
            if col not in f:
                continue
            for _, r in f[f[col] > floor].nlargest(top, col).iterrows():
                out.append({"cc": cc, "month": r["month"], "what": label, "bn": float(r[col])})
    return out
