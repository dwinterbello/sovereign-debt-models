"""Hand-entered, sourced constants: the annual plan anchor and known events not yet on the tape.

Every entry carries its source and vintage. Bond events are appended to the issuance tape as
tape-like rows (``source`` = ``KNOWN_EVENT``) unless the tape already carries the same line settling
in the same month; other_fin events go straight into the forward ``other_fin_bn``.
"""

from __future__ import annotations

import pandas as pd

KNOWN_EVENT = "known_event"

# DFP 2026, Tavola II.1.2.5 'Saldi di cassa': settore statale −5.6% of GDP (2026), −4.8% (2027);
# Tavola II.1.2.2 nominal GDP (tendenziale) 2,313,094 / 2,372,795 EUR mn. Published to 0.1% of GDP,
# so each level is ±1.2bn. The Dec-2025 Linee guida della gestione del debito pubblico 2026 put 2026
# at ~125bn (DPB 2026 vintage).
PLAN = {
    2026: {"pct_gdp": 5.6, "gdp_mn": 2_313_094.0},
    2027: {"pct_gdp": 4.8, "gdp_mn": 2_372_795.0},
}
PLAN_SOURCE = ("DFP 2026 (Documento di finanza pubblica, CdM 22-Apr-2026), Tav. II.1.2.5 settore statale "
               "% of GDP × Tav. II.1.2.2 nominal GDP; sovereign_cash/data/it/plan/DFP26-file-unico.pdf")
PLAN_VINTAGE = "2026-04-22"


def plan_mn() -> dict[int, float]:
    return {y: round(v["pct_gdp"] / 100.0 * v["gdp_mn"], 0) for y, v in PLAN.items()}


# Syndications and retail issues announced by MEF but not on the issuance tape.
BOND_EVENTS = [
    {"name": "BTP Green 4.40% 30-Oct-2038 (syndication)", "op_dt": "2026-09-22", "issue_dt": "2026-09-29",
     "maturity_dt": "2038-10-30", "amount_issued": 8000.0, "price": 99.591, "sec_type": "Nominal",
     "assumption": None,
     "source": "MEF comunicato 99, 22-Sep-2026 (8bn at 99.591, settlement 29-Sep-2026)"},
    {"name": "BTP Valore + BTP Valore Insieme 5y (retail, double issue)", "op_dt": "2026-10-23",
     "issue_dt": "2026-10-27", "maturity_dt": "2031-10-27", "amount_issued": None, "price": 100.0,
     "sec_type": "Nominal",
     "assumption": "size = mean of the last two BTP Valore issues on the tape (Oct-2025, Mar-2026); settlement "
                   "assumed T+2 after the 23-Oct close (Oct-2025 issue: close Fri 24-Oct, settled 28-Oct)",
     "source": "MEF comunicato 104, 25-Sep-2026 (placement 19–23 Oct-2026, 5y, at par)"},
]
VALORE_SOURCES = ("MEF BTP Valore Oct-2025", "MEF BTP Valore Mar-2026")
VALORE_FALLBACK_MN = (16570.0 + 16222.584) / 2.0

# Italy's RRF envelope after the 8-Dec-2023 amendment of the Council Implementing Decision: 194.4bn =
# 122.6bn loans + 71.8bn grants. Grants are budget revenue (they lower the state-sector fabbisogno);
# loans are financing (Banca d'Italia Tavola 2 'EU loans'). Instalments are split at this ratio.
RRF_LOANS_BN = 122.6
RRF_GRANTS_BN = 71.8
RRF_SOURCE = "Council Implementing Decision on Italy's PNRR, amended 8-Dec-2023 (122.6bn loans + 71.8bn grants)"
RRF_START = pd.Timestamp("2021-08-01")
RRF_MIN_LOAN_BN = 0.5


def rrf_grant_share() -> float:
    return RRF_GRANTS_BN / (RRF_GRANTS_BN + RRF_LOANS_BN)


# RRF instalments not yet received (EUR bn, by month); split into grant (fabbisogno) and loan (other_fin).
RRF_EVENTS = [
    {"name": "RRF 10th (final) instalment", "month": "2026-12-01", "amount_bn": 28.4,
     "assumption": "requested 29-Sep-2026; past instalments paid 3–9 months after request and the RRF "
                   "regulation sets 31-Dec-2026 as the last payment date",
     "source": "Ministro Foti / Italpress 29-Sep-2026; Cabina di regia PNRR 23-Sep-2026 (28.4bn)"},
]
OTHER_FIN_EVENTS: list[dict] = []


def valore_size(ops: pd.DataFrame) -> float:
    src = ops["source"].astype(str)
    v = ops.loc[src.isin(VALORE_SOURCES), "amount_issued"]
    return float(v.mean()) if len(v) == len(VALORE_SOURCES) else VALORE_FALLBACK_MN


def bond_event_rows(ops: pd.DataFrame) -> pd.DataFrame:
    """Tape-like rows for BOND_EVENTS that the tape does not already carry."""
    rows = []
    if "source" in ops:
        ops = ops[ops["source"].astype(str) != KNOWN_EVENT]
    mat = pd.to_datetime(ops["maturity_dt"], errors="coerce")
    settle = pd.to_datetime(ops["issue_dt"], errors="coerce").fillna(pd.to_datetime(ops["op_dt"], errors="coerce"))
    for e in BOND_EVENTS:
        m, s = pd.Timestamp(e["maturity_dt"]), pd.Timestamp(e["issue_dt"])
        dup = (mat == m) & (settle.dt.to_period("M") == s.to_period("M")) & (ops["amount_issued"] > 0)
        if dup.any():
            continue
        amt = e["amount_issued"] if e["amount_issued"] is not None else valore_size(ops)
        rows.append({"country": "IT", "ccy": "EUR", "op_dt": pd.Timestamp(e["op_dt"]), "issue_dt": s,
                     "maturity_dt": m, "isin": f"IT_EVENT_{m:%Y%m%d}", "amount_issued": float(amt),
                     "price": e["price"], "sec_type": e["sec_type"], "op_type": "Syndication",
                     "source": KNOWN_EVENT, "sec_desc": e["name"]})
    return pd.DataFrame(rows)


def supplement(ops: pd.DataFrame) -> pd.DataFrame:
    if "source" in ops:
        ops = ops[ops["source"].astype(str) != KNOWN_EVENT]
    ev = bond_event_rows(ops)
    if ev.empty:
        return ops
    return pd.concat([ops, ev.reindex(columns=ops.columns)], ignore_index=True)


def other_fin_bn(months: pd.DatetimeIndex) -> pd.Series:
    """Non-security events: other_fin events plus the loan part of RRF instalments (EUR bn)."""
    out = pd.Series(0.0, index=months)
    for e in OTHER_FIN_EVENTS:
        m = pd.Timestamp(e["month"])
        if m in out.index:
            out[m] += e["amount_bn"]
    return out + rrf_split(months)["loan_bn"]


def rrf_split(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward RRF instalments by month: grant part (lowers fabbisogno) and loan part (other_fin), EUR bn."""
    out = pd.DataFrame(0.0, index=months, columns=["grant_bn", "loan_bn"])
    g = rrf_grant_share()
    for e in RRF_EVENTS:
        m = pd.Timestamp(e["month"])
        if m in out.index:
            out.loc[m, "grant_bn"] += e["amount_bn"] * g
            out.loc[m, "loan_bn"] += e["amount_bn"] * (1.0 - g)
    return out


def rrf_grants_history(eu_loans_bn: pd.Series) -> pd.Series:
    """Estimated RRF grant receipts per month (EUR bn) from the loan part Banca d'Italia records.

    A month with EU loans above ``RRF_MIN_LOAN_BN`` since the RRF began is an instalment month; its grant
    part = loans × grants / loans in the plan envelope. Other months (and SURE repayments) are 0.
    """
    s = pd.to_numeric(eu_loans_bn, errors="coerce").fillna(0.0)
    on = (s.index >= RRF_START) & (s > RRF_MIN_LOAN_BN)
    return (s * RRF_GRANTS_BN / RRF_LOANS_BN).where(on, 0.0)


def describe(ops: pd.DataFrame) -> list[dict]:
    """Event list for the summary JSON (sizes resolved)."""
    added = bond_event_rows(ops)
    out = []
    for e in BOND_EVENTS:
        amt = e["amount_issued"] if e["amount_issued"] is not None else valore_size(ops)
        on_tape = not (len(added) and (added["maturity_dt"] == pd.Timestamp(e["maturity_dt"])).any())
        out.append({"name": e["name"], "kind": "bond gross", "settle": e["issue_dt"],
                    "cash_bn": round(amt * e["price"] / 100.0 / 1000.0, 2), "already_on_tape": on_tape,
                    "assumption": e["assumption"], "source": e["source"]})
    g = rrf_grant_share()
    for e in RRF_EVENTS:
        out.append({"name": e["name"], "kind": "rrf", "settle": e["month"], "cash_bn": e["amount_bn"],
                    "grant_bn": round(e["amount_bn"] * g, 2), "loan_bn": round(e["amount_bn"] * (1 - g), 2),
                    "already_on_tape": False, "assumption": e["assumption"], "source": e["source"]})
    for e in OTHER_FIN_EVENTS:
        out.append({"name": e["name"], "kind": "other_fin", "settle": e["month"], "cash_bn": e["amount_bn"],
                    "already_on_tape": False, "assumption": e["assumption"], "source": e["source"]})
    return out
