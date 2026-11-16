import pandas as pd
import pytest

from issuance import positioning
from issuance.paths import CACHE


def test_summarise_z_and_percentile():
    dates = pd.date_range("2023-01-06", periods=150, freq="W-FRI")
    s = pd.DataFrame({"panel": "p", "series": "a", "label": "A", "unit": "u",
                      "date": dates, "value": range(150)})
    out = positioning.summarise(s)
    row = out.iloc[0]
    assert row["value"] == 149
    assert row["pct"] == 100
    assert row["z"] > 1.5


def test_series_schema_and_no_duplicates():
    if not (CACHE / "jp_flows_weekly.parquet").exists():
        pytest.skip("jp flows cache absent")
    s = positioning.build_series()
    assert list(s.columns) == positioning.COLS
    assert not s.duplicated(["panel", "series", "date"]).any()
    assert {"jp_weekly", "jp_investor"} <= set(s["panel"])


def test_jp_weekly_is_four_week_sum_in_trillions():
    path = CACHE / "jp_flows_weekly.parquet"
    if not path.exists():
        pytest.skip("jp flows cache absent")
    raw = pd.read_parquet(path).sort_values("week_end")
    s = positioning.jp_weekly()
    last = s[s["series"] == "a_lt_net"].iloc[-1]
    assert last["value"] == pytest.approx(raw["a_lt_net"].tail(4).sum() / 1000)
