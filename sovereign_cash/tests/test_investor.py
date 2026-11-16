import numpy as np
import pandas as pd

from sovereign_cash import investor as iv


def test_germany_skill_is_the_live_core_not_planhc():
    rows = {r["cc"]: r for r in iv.skill()}
    de = rows.get("de")
    if not de:
        return
    assert de["verdict"] == "PRE vs prior path"
    assert "beats naive" not in de["verdict"]
    assert "not this forecast" in de["basis"]
    assert "planhc_3 vs smly is not this path" in de["oos"]
    assert "PRE mean MAE" in de["oos"]


def test_verdict_band():
    assert iv._verdict(37.3, 37.8) == "ties naive"
    assert iv._verdict(19.5, 18.8) == "does not beat naive"
    assert iv._verdict(10.0, 20.0) == "beats naive"


def test_norway_drops_naive_months():
    d = iv.monthly("no")
    if d is None:
        return
    f = d[d["is_forecast"]]
    assert not f["fc_source"].astype(str).str.startswith("same_month").any()


def test_supply_scaled_consistently():
    sc = iv.scalers()
    if not sc:
        return
    sup, panel = iv.supply(sc)
    for _, r in sup.iterrows():
        gdp = sc["gdp_bn"][r["cc"]]
        assert np.isclose(r["net_bonds_pct_gdp"], 100 * r["net_bonds"] / gdp)
        assert np.isclose(r["net_bills_usd"], r["net_bills"] * sc["fx_usd_per_unit"][iv.CCY[r["cc"]]])
    assert set(panel["cc"]) <= set(iv.ORDER)
    assert pd.api.types.is_datetime64_any_dtype(panel["month"])


def test_published_cash_month_is_not_forward(tmp_path, monkeypatch):
    months = pd.date_range("2026-07-01", periods=4, freq="MS")
    pd.DataFrame({"month": months, "is_forecast": [False, True, True, True],
                  "cash_observed": [True, True, False, False], "cash_requirement_bn": 1.0,
                  "net_bonds_bn": 1.0, "net_bills_bn": 0.0}).to_parquet(tmp_path / "de_cash_monthly.parquet")
    monkeypatch.setattr(iv, "DERIVED", tmp_path)
    d = iv.monthly("de")
    assert d.loc[d["is_forecast"], "month"].min() == pd.Timestamp("2026-09-01")


def test_tracking_gap_is_model_minus_plan():
    for t in iv.tracking():
        assert np.isclose(t["gap"], t["model"] - t["plan"])
