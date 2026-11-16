"""Stack gross issuance must reconcile to an independent official total in every country.

Skips only when the official file (or the country's ops cache) is absent. Tolerances live
in ``issuance.reconcile_official``; ``TOL_PCT_TEMP`` entries are temporary gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from issuance import reconcile_official as R


@pytest.mark.parametrize("cc", R.COUNTRIES)
def test_stack_matches_official_total(cc):
    if not Path(R.OFFICIAL[cc]).exists():
        pytest.skip(f"official file missing: {R.OFFICIAL[cc]}")
    df, meta = R.run([cc])
    if meta[cc]["status"] == "no_ops":
        pytest.skip(f"no ops_{cc.lower()}.parquet")
    assert not df.empty, f"{cc}: official file present but no comparable periods"
    bad = df[df["status"] == "FAIL"]
    assert bad.empty, (
        f"{cc}: {len(bad)} periods over tolerance\n"
        + bad[["check", "period", "group", "stack_mn", "official_mn", "delta_pct", "tol_pct"]].to_string(index=False)
    )


def test_reported_tolerance_is_the_band_applied():
    import pandas as pd

    df = R._grade(pd.DataFrame([
        R._row("ES", "t13", 2025, "Linker", 7607.0, 7126.0),
        R._row("ES", "t13", 2025, "Bill", 102_700.0, 102_711.0),
        R._row("UK", "x", 2025, "Linker", 1010.0, 1000.0),
    ]))
    for col, tol in (("status", "tol_pct"), ("status_target", "tol_target_pct")):
        ok = df[df[col] == "ok"]
        assert (ok["delta_pct"].abs() <= ok[tol] + 1e-12).all(), ok
    assert df.loc[0, "status_target"] == "FAIL"


def test_es_bulletins_dedupe_across_label_languages():
    from issuance import fetch_es

    if not (fetch_es.RAW_ES / "11_auctions.xlsx").exists():
        pytest.skip("ES bulletin 11 missing")
    o = fetch_es.parse_auctions()
    d = o[o["op_dt"] == "2025-12-04"]
    if d.empty:
        pytest.skip("4-Dec-2025 auction not in the bulletins")
    assert not d.duplicated(["sec_type", "resid_mat_yrs", "amount_issued"]).any(), d


def test_temporary_gates_stay_looser_than_target():
    for cc, tol in R.TOL_PCT_TEMP.items():
        assert tol >= R.TOL_PCT[cc], cc
