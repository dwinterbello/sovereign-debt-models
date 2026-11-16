import pandas as pd

from issuance.fetch_it import SUPPLEMENT, supplement_it
from issuance.schema import coerce_ops


def _tape() -> pd.DataFrame:
    return coerce_ops(pd.DataFrame({
        "op_dt": pd.to_datetime(["2023-06-09", "2026-02-12"]),
        "isin": [pd.NA, "IT0005689994"],
        "maturity_dt": pd.to_datetime(["2025-11-13", "2033-03-15"]),
        "amount_issued": [18191.09, 3000.0],
        "sec_type": ["Nominal", "Nominal"],
        "op_type": ["Auction", "Auction"],
        "tenor_bucket": ["1-3y", "3-7y"],
    }), "IT")


def test_supplement_replaces_isinless_valore_and_is_idempotent():
    sup = pd.read_json(SUPPLEMENT, dtype={"isin": str})
    once = supplement_it(_tape())
    assert once["isin"].notna().all()
    valore = once[once["isin"].eq("IT0005547408")]
    assert len(valore) == 1 and valore["maturity_dt"].iloc[0] == pd.Timestamp("2027-06-13")
    assert len(once) == 1 + len(sup)
    twice = supplement_it(once)
    assert len(twice) == len(once)


def test_supplement_rows_are_syndications_with_real_maturities():
    out = supplement_it(_tape())
    add = out[out["op_type"].eq("Syndication")]
    assert add["maturity_dt"].notna().all()
    assert (add["maturity_dt"] > add["op_dt"]).all()
    assert set(add["sec_type"]) <= {"Nominal", "Linker"}
    assert add["isin"].is_unique
