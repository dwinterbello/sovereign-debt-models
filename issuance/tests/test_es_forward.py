"""Spain Tesoro coupon forward paint (Bonos/Obligaciones)."""
from __future__ import annotations

import pandas as pd
import pytest

from issuance.isin_flow import load_ops
from issuance.paths import CACHE


@pytest.mark.skipif(not (CACHE / "ops_es.parquet").exists(), reason="ops_es missing")
def test_es_project_fills_horizon():
    from issuance.forward import HORIZON, project

    ops = load_ops()
    es = ops[ops["country"] == "ES"]
    assert not es.empty
    lines = pd.read_parquet(CACHE / "lines.parquet")
    opened = lines[(lines["country"] == "ES") & (lines["open"].astype(bool))]
    asof = pd.to_datetime(es["op_dt"]).max()
    paint = project("ES", es, opened, asof)
    assert not paint.empty
    assert paint["op_dt"].min() > asof
    assert paint["op_dt"].max() <= HORIZON + pd.Timedelta(days=5)
    # ~12 months of coupon at ~€15–20bn/month
    total = float(paint["amount_mn"].sum())
    assert 100_000 < total < 400_000
    assert paint["country"].eq("ES").all()


@pytest.mark.skipif(not (CACHE / "forecast_priced.parquet").exists(), reason="run write_lines first")
def test_es_in_forecast_priced_after_refresh():
    p = pd.read_parquet(CACHE / "forecast_priced.parquet")
    if "ES" not in set(p["country"].astype(str)):
        pytest.skip("ES not yet in forecast_priced — run issuance lines refresh")
    es = p[p["country"].astype(str).eq("ES")]
    assert len(es) > 20
    assert es["amount_mn"].sum() > 50_000
