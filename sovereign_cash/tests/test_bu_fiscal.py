"""Cash bottom-up → debt revenue bridge."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt import bu_fiscal as BF


def test_build_writes_receipts_for_ship_countries():
    tab, meta = BF.build(y0=2025, y_end=2030)
    assert isinstance(tab, pd.DataFrame)
    for cc in ("uk", "de", "fr", "it"):
        if not meta.get(cc, {}).get("ship"):
            continue
        rec = tab[(tab["cc"] == cc) & (tab["category"] == "receipts_total")]
        assert len(rec) >= 2, (cc, meta[cc])
        idx = BF.receipts_index(tab, cc, 2025, range(2026, 2030))
        assert idx is not None
        assert any(np.isfinite(v) for v in idx.values())


def test_receipts_index_none_without_base():
    empty = pd.DataFrame(columns=["cc", "year", "category", "value_bn", "kind"])
    assert BF.receipts_index(empty, "uk", 2025, range(2026, 2028)) is None
