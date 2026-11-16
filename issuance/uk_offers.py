"""DMO-announced gilt nominals.

Kept out of ``issuance.forward`` so the desk and the publish-time splice can
read the offers without importing pandas. Amounts are £mn of nominal.
"""

from __future__ import annotations

# (date, isin, bucket, amount_mn, sector)
# 10 Nov 2056 issued £1,000mn. 11 Nov 2028 offer £1,000mn (DMO upcoming,
# maturity 2028-01-31). 12 Nov 2033 offer £4,000mn. 17 Nov 2041 offer £3,250mn.
UK_ANNOUNCED = (
    ("2026-11-10", "GB00BT7J0241", "20-30y", 1_000.0, "Nominal"),
    ("2026-11-11", "GB00BMBL1G81", "1-3y", 1_000.0, "Nominal"),
    ("2026-11-12", "GB00BVP99780", "3-7y", 4_000.0, "Nominal"),
    ("2026-11-17", "GB00BVP99897", "10-15y", 3_250.0, "Nominal"),
)

UK_ANNOUNCED_DATES = frozenset(row[0] for row in UK_ANNOUNCED)
UK_ANNOUNCED_BY_DATE = {row[0]: row for row in UK_ANNOUNCED}


def offer_mn(day: str) -> float | None:
    """Announced nominal for an auction date, or None when DMO has not sized it."""
    row = UK_ANNOUNCED_BY_DATE.get(str(day)[:10])
    if row is None:
        return None
    return float(row[3])
