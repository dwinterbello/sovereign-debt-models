import numpy as np
import pandas as pd
import pytest

from issuance import flow_score as fs
from issuance.demand import sleeve_of_years


def test_zscore_has_no_lookahead():
    s = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2020-01-01", periods=60, freq="MS"))
    z = fs._zscore(s)
    bumped = s.copy()
    bumped.iloc[-1] = 1e6
    assert fs._zscore(bumped).iloc[:-1].equals(z.iloc[:-1])
    assert z.iloc[: fs.Z_MIN - 1].isna().all()


def test_sleeve_of_years():
    assert sleeve_of_years(5.0) == "int"
    assert sleeve_of_years(10.0) == "int"
    assert sleeve_of_years(10.1) == "long"
    assert sleeve_of_years(1.5) == "short"
    s = sleeve_of_years(pd.Series([1.0, 5.0, 15.0]))
    assert list(s) == ["short", "int", "long"]


def test_signs_point_to_demand():
    comp = fs.components()
    if comp.empty:
        pytest.skip("caches absent")
    signs = comp.groupby("component")["sign"].first()
    assert signs.get("tail", -1) == -1 and signs.get("dealer", -1) == -1
    assert signs.get("cover", 1) == 1 and signs.get("nonres", 1) == 1


def test_score_is_mean_of_scored_pillars_and_blank_without_data():
    out = fs.build()
    if out.empty:
        pytest.skip("caches absent")
    last = out[out["month"] == out["month"].max()]
    pil = last[(last["component"] == "_pillar") & last["pillar"].isin(fs.SCORED)].groupby("country")["z"].mean()
    score = last[(last["component"] == "_score") & (last["pillar"] == "score")].set_index("country")["z"]
    assert (score - pil.reindex(score.index)).abs().max() < 1e-12
    eu = last[(last["country"].isin(["DE", "FR", "IT", "UK"])) & (last["pillar"] == "positioning")]
    assert eu.empty


def test_sleeve_scores_present_when_auction_split_exists():
    out = fs.build()
    if out.empty:
        pytest.skip("caches absent")
    last = out[(out["component"] == "_score") & (out["month"] == out["month"].max())]
    pillars = set(last["pillar"])
    assert "score" in pillars
    # US/JP always have int/long auction or JSDA; at least one sleeve score beyond all.
    assert pillars & {"score_int", "score_long"}
    assert "sleeve" in out.columns
