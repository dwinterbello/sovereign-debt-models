import pandas as pd
import pytest

from issuance.forward import _coupon_tape_in_month, _fit_rest_of_month, _it_retail_q4


def _ops(rows):
    return pd.DataFrame(rows, columns=["country", "op_dt", "amount_issued", "sec_type", "source"]).assign(
        op_dt=lambda d: pd.to_datetime(d["op_dt"]))


def test_rest_of_month_takes_only_what_the_target_has_left():
    asof = pd.Timestamp("2026-10-01")
    ops = _ops([("FR", "2026-10-01", 12_000.0, "Nominal", ""), ("FR", "2026-10-01", 3_000.0, "Bill", ""),
                ("FR", "2026-09-17", 15_000.0, "Nominal", "")])
    done = _coupon_tape_in_month(ops, "FR", asof.to_period("M"), asof)
    assert done == 12_000.0
    rows = [{"op_dt": pd.Timestamp("2026-10-01"), "amount_mn": 9_000.0},   # dropped later: the tape stands in
            {"op_dt": pd.Timestamp("2026-10-15"), "amount_mn": 14_000.0},
            {"op_dt": pd.Timestamp("2026-11-05"), "amount_mn": 20_000.0}]
    _fit_rest_of_month(rows, asof.to_period("M"), asof, 23_200.0, done)
    assert rows[1]["amount_mn"] == pytest.approx(11_200.0)
    assert rows[0]["amount_mn"] == 9_000.0 and rows[2]["amount_mn"] == 20_000.0
    _fit_rest_of_month(rows, asof.to_period("M"), asof, 10_000.0, done)
    assert rows[1]["amount_mn"] == 0.0  # the tape already sold more than the target


def test_it_retail_is_booked_once_inside_q4():
    past = [("IT", "2025-10-24", 16_570.0, "Nominal", "MEF BTP Valore Oct-2025"),
            ("IT", "2026-03-06", 16_222.584, "Nominal", "MEF BTP Valore Mar-2026"),
            ("IT", "2026-09-30", 3_000.0, "Nominal", "BdI")]
    size, rows = _it_retail_q4(_ops(past), pd.Timestamp("2026-09-30"))
    assert size == pytest.approx((16_570.0 + 16_222.584) / 2)
    assert len(rows) == 1 and rows[0]["op_dt"] == pd.Timestamp("2026-10-23") and rows[0]["tenor_bucket"] == "3-7y"
    printed = past + [("IT", "2026-10-23", 18_000.0, "Nominal", "MEF BTP Valore Oct-2026")]
    size, rows = _it_retail_q4(_ops(printed), pd.Timestamp("2026-10-30"))
    assert size == 18_000.0 and rows == []
