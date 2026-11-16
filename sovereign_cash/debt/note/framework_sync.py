"""Carry framework corrections made after the reviewed note build into ``note_numbers.json``.

The reviewed note is the 10 Nov 2026 build. Rebuilding the whole chain today would move every country with three
days of data and code changes, so only the corrected keys are replaced, with the framework's corrected values on
today's data. The published value is today's full framework value, reviewed again in the 13 Nov stress test; the audit
block splits it into the reviewed 10 Nov value, the data update since (uncorrected today minus reviewed) and the
correction (corrected minus uncorrected, today), so nothing moves without a named source.

  us_primary   ``scenarios.channels.us_dts_adjust`` (FY2026 one-offs out of the DTS path: IEEPA tariff refunds,
               the refund catch-up, corporate and customs run-rates, withheld on the wage bill, non-withheld on
               equities; spending lines on CBO growth or nominal GDP instead of trailing growth), the US line model
               through FY2030 (``primary.US_DTS_TO_YEND``) and ``scenarios.primary.US_DTS_TAX_OVERLAY`` (the 2025 act's tax measures not added on
               top of DTS levels that already carry them). Keys: pb_gg_us_Y, pb_meas_us_Y for the DTS years.

Debt paths (``d_*``) and the primary gaps (``adj_*``) hold the current primary (pb_now, FY2026 outturn), so they do
not move. Run before ``scenario_grid`` and the docx builders:

    PYTHONPATH=. .venv/bin/python -m sovereign_cash.debt.note.framework_sync
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

NOTE = Path(__file__).resolve().parent
NUMBERS = NOTE / "note_numbers.json"


def us_primary_delta() -> dict[int, dict[str, float]]:
    from sovereign_cash.scenarios import primary as P
    from sovereign_cash.scenarios.channels import us_dts as U
    from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ

    def run(corrected: bool):
        ADJ.ENABLED, P.US_DTS_TAX_OVERLAY = corrected, not corrected
        U.fy_table.cache_clear()
        t = P.path("us")
        return t[["primary_gg_pct", "primary_deficit_pct", "rule"]]

    try:
        old, new = run(False), run(True)
    finally:
        ADJ.ENABLED, P.US_DTS_TAX_OVERLAY = True, False
        U.fy_table.cache_clear()
    yrs = [y for y in new.index if new.at[y, "rule"] == "dts_fy"]
    return {int(y): {"gg": float(new.at[y, "primary_gg_pct"]), "meas": float(new.at[y, "primary_deficit_pct"]),
                     "gg_uncorrected": float(old.at[y, "primary_gg_pct"]),
                     "gg_corr": float(new.at[y, "primary_gg_pct"] - old.at[y, "primary_gg_pct"])}
            for y in yrs}


# Ranges for the US 2027 primary: one-offs and run-rates at the ends of their evidence (us_dts_adjust constants)
US_LOW = {"REFUND_PERSIST": 0.25, "CBP_TAIL_BN": 20.0, "CORP_FIRST_GROWTH": 0.20, "VA_OVER_NGDP": 0.0,
          "CUSTOMS_SCALE": 1.05, "OTHER_RECEIPTS_OVER_NGDP": 0.01}
US_HIGH = {"REFUND_PERSIST": 0.9, "CBP_TAIL_BN": 50.0, "CORP_FIRST_GROWTH": 0.0, "MEDICAID_OVER_NGDP": 0.03,
           "CUSTOMS_SCALE": 0.8, "OTHER_RECEIPTS_OVER_NGDP": -0.01}
# CBO cross-check on CBO's own basis. Our federal measure subtracts gross interest (FRED FYOINT includes interest paid to
# the trust funds, $1,216bn in FY2025 against CBO's net $970bn), so it is not CBO's primary deficit; the GG wedge (0.88)
# absorbs that. Like for like: CBO's federal primary vs our GG primary less the state-and-local gap, measured in FY2025
# (AMECO GG 2.72% of GDP against CBO's actual federal primary of $805bn).
CBO_FY25_PRIMARY_BN = 805.0     # CBO, The Budget and Economic Outlook: 2026 to 2036, By the Numbers (Feb 2026)
CBO_FY27_PRIMARY_PCT = 2.3      # ibid., $779bn
CBO_TARIFF_PRIMARY_BN = 700.0   # CBO, Aug 2026 tariff update: $0.9tn of deficits over 2027-36, ~$0.7tn of it primary


def us_baseline_stats() -> dict[str, float]:
    """The US 2027 range, and where the narrowing comes from (receipts vs spending, % of GDP)."""
    from sovereign_cash.scenarios import primary as P
    from sovereign_cash.scenarios.channels import us_dts as U
    from sovereign_cash.scenarios.channels import us_dts_adjust as ADJ

    def run(kw):
        old = {k: getattr(ADJ, k) for k in kw}
        try:
            for k, v in kw.items():
                setattr(ADJ, k, v)
            U.fy_table.cache_clear()
            return P.path("us")
        finally:
            for k, v in old.items():
                setattr(ADJ, k, v)
            U.fy_table.cache_clear()

    base, lo, hi = run({}), run(US_LOW), run(US_HIGH)
    out = {"pb_gg_us_2027_lo": float(lo.at[2027, "primary_gg_pct"]),
           "pb_gg_us_2027_hi": float(hi.at[2027, "primary_gg_pct"]),
           "pb_gg_us_2030_lo": float(lo.at[2030, "primary_gg_pct"]),
           "pb_gg_us_2030_hi": float(hi.at[2030, "primary_gg_pct"])}
    sl = float(base.at[2025, "primary_gg_pct"]) - 100 * CBO_FY25_PRIMARY_BN / float(base.at[2025, "gdp"])
    out["us_sl_gap_2025"] = sl
    out["pb_fed_cbo_us_2027"] = float(base.at[2027, "primary_gg_pct"]) - sl
    out["cbo_pb_2027"] = CBO_FY27_PRIMARY_PCT
    out["cbo_pb_2027_tariff"] = CBO_FY27_PRIMARY_PCT + 100 * CBO_TARIFF_PRIMARY_BN / 10 / float(base.at[2027, "gdp"])
    for y in (2026, 2027, 2029):
        g = float(base.at[y, "gdp"])
        out[f"us_rec_pct_{y}"] = 100 * float(base.at[y, "receipts"] + base.at[y, "policy_tax"]) / g
        out[f"us_spend_pct_{y}"] = 100 * float(base.at[y, "primary_spend"] + base.at[y, "policy_spend"]) / g
    return out


def sync() -> dict:
    N = json.loads(NUMBERS.read_text())
    meta = N.get("_framework_sync", {})
    base = meta.get("base", {})  # reviewed values, kept so the sync is idempotent
    d = us_primary_delta()
    log = []
    for y, dd in d.items():
        for k, v in ((f"pb_gg_us_{y}", dd["gg"]), (f"pb_meas_us_{y}", dd["meas"])):
            if k not in N:
                continue
            b = base.setdefault(k, N[k])
            N[k] = v
            row = {"key": k, "reviewed": round(b, 4), "now": round(v, 4)}
            if k.startswith("pb_gg"):
                lab = "data_update" if y <= 2029 else "data_and_horizon_update"  # FY2030: WEO -> line model
                row.update({lab: round(dd["gg_uncorrected"] - b, 4), "correction_today": round(dd["gg_corr"], 4)})
            log.append(row)
    N.update(us_baseline_stats())
    N["_framework_sync"] = {"date": date.today().isoformat(), "base": base, "us_primary": log,
                            "source": "scenarios.channels.us_dts_adjust + scenarios.primary.US_DTS_TAX_OVERLAY"}
    NUMBERS.write_text(json.dumps(N, indent=1, default=str))
    return N["_framework_sync"]


if __name__ == "__main__":
    out = sync()
    for r in out["us_primary"]:
        print(r)
