from pathlib import Path

import pandas as pd
import pytest

from issuance.forward import _de

OPS = Path(__file__).resolve().parents[1] / "cache" / "ops_de.parquet"


@pytest.mark.skipif(not OPS.exists(), reason="DE tape cache")
def test_long_reopenings_keep_their_bund():
    ops = pd.read_parquet(OPS)
    ops["op_dt"] = pd.to_datetime(ops["op_dt"])
    rows = pd.DataFrame(_de(ops, pd.DataFrame(), pd.Timestamp("2026-10-01")))
    rows = rows[pd.to_datetime(rows["op_dt"]) >= "2027-01-01"]
    tape_mat = ops.groupby(ops["isin"].astype(str))["maturity_dt"].first()
    real = rows[rows["isin"].str.startswith("DE")]
    # Reopened lines keep the tape maturity and never take a tap inside their last year.
    assert (pd.to_datetime(real["maturity_dt"]) == pd.to_datetime(real["isin"].map(tape_mat))).all()
    assert ((pd.to_datetime(real["maturity_dt"]) - pd.to_datetime(real["op_dt"])).dt.days > 365).all()
    # The 2056 30y is reopened, not replaced by a fictional 2057.
    assert "DE000BU2D012" in set(real["isin"])
    long_new = rows[rows["isin"].str.startswith("next") & rows["isin"].str.contains("15y|20y|30y")]
    assert len(long_new) <= 3
