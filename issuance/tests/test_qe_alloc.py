import numpy as np
import pandas as pd

from issuance.qe_alloc import (
    FLOW_SHARE,
    WAM_TOL,
    allocate_isin,
    _extreme_wam,
    _wam_feasible_stock,
    project_box_affine,
)
from issuance.public_demand import allocate_tenor
from issuance.tenor import BUCKET_MID, BUCKET_ORDER


def _universe(resids, outst=None):
    n = len(resids)
    if outst is None:
        outst = [100.0] * n
    return pd.DataFrame({
        "isin": [f"X{i:02d}" for i in range(n)],
        "outstanding": outst,
        "resid": resids,
    })


def test_allocate_isin_matches_stock_and_wam():
    u = _universe([2.0, 5.0, 8.0, 12.0, 20.0], [80, 120, 100, 90, 70])
    res = allocate_isin(150.0, 8.0, u, cap=0.50)
    assert abs(res.achieved_stock - 150.0) < 1e-3
    assert abs(res.achieved_wam - 8.0) < 1e-3
    assert res.slack < 1e-6
    assert res.ok
    assert (res.holdings <= 0.50 * res.outstanding + 1e-6).all()


def test_allocate_isin_respects_cap_and_reports_slack():
    u = _universe([5.0, 6.0, 7.0], [100, 100, 100])
    res = allocate_isin(200.0, 6.0, u, cap=0.33)
    assert abs(res.achieved_stock - 99.0) < 1e-3
    assert abs(res.slack - 101.0) < 1e-3
    assert abs(res.achieved_wam - 6.0) < 0.05


def test_allocate_isin_empty_universe():
    res = allocate_isin(100.0, 7.0, pd.DataFrame())
    assert res.achieved_stock == 0.0
    assert res.slack == 100.0


def test_allocate_isin_smooth_vs_previous():
    u = _universe([3.0, 7.0, 15.0], [200, 200, 200])
    first = allocate_isin(120.0, 8.0, u, cap=0.40)
    prev = {str(i): float(h) for i, h in zip(first.isins, first.holdings)}
    second = allocate_isin(110.0, 8.1, u, prev=prev, cap=0.40)
    assert abs(second.achieved_wam - 8.1) < 1e-2
    fresh = allocate_isin(110.0, 8.1, u, cap=0.40)
    turn_prev = float(np.abs(second.holdings - first.holdings).sum())
    turn_fresh = float(np.abs(fresh.holdings - first.holdings).sum())
    assert turn_prev <= turn_fresh + 1e-6


def test_project_box_affine_unit():
    h0 = np.array([10.0, 10.0, 10.0])
    y = np.array([2.0, 8.0, 20.0])
    lo = np.zeros(3)
    hi = np.array([20.0, 20.0, 20.0])
    h, ok = project_box_affine(h0, y, 30.0, 30.0 * 8.0, lo, hi)
    assert ok
    assert abs(h.sum() - 30.0) < 1e-6
    assert abs(np.dot(h, y) / 30.0 - 8.0) < 1e-6


def test_allocator_matches_stock_and_wam():
    outstanding = {b: 100.0 for b in BUCKET_ORDER}
    target_stock = 200.0
    target_wam = 8.0
    alloc = allocate_tenor(target_stock, target_wam, outstanding)
    total = sum(alloc.values())
    wam = sum(alloc[b] * BUCKET_MID[b] for b in BUCKET_ORDER) / total
    assert abs(total - target_stock) < 1e-3
    assert abs(wam - target_wam) < 0.15


def test_aging_buys_only_in_window():
    u = _universe([0.4, 5.0, 12.0, 25.0], [200, 200, 200, 200])
    prev = {"X00": 40.0, "X01": 40.0, "X02": 40.0, "X03": 0.0}
    res = allocate_isin(150.0, 9.0, u, prev=prev, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    by = {str(i): float(h) for i, h in zip(res.isins, res.holdings)}
    assert by["X00"] <= 40.0 + 1e-6
    assert by["X01"] >= 40.0 - 1e-6
    assert by["X02"] >= 40.0 - 1e-6
    assert abs(res.achieved_stock - 150.0) < 1e-3
    assert abs(res.achieved_wam - 9.0) < WAM_TOL
    assert res.mode.startswith("purchase")
    assert res.turnover_mn + 1e-6 >= abs(res.flow_mn)


def test_aging_sales_do_not_buy():
    u = _universe([3.0, 8.0, 20.0], [200, 200, 200])
    prev = {"X00": 50.0, "X01": 50.0, "X02": 50.0}
    res = allocate_isin(120.0, 8.0, u, prev=prev, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    by = {str(i): float(h) for i, h in zip(res.isins, res.holdings)}
    assert abs(res.achieved_stock - 120.0) < 1e-3
    assert abs(res.achieved_wam - 8.0) < WAM_TOL
    assert res.flow_mn < 0
    if res.mode == "sale":
        assert by["X00"] <= 50.0 + 1e-6
        assert by["X01"] <= 50.0 + 1e-6
        assert by["X02"] <= 50.0 + 1e-6


def test_aging_turnover_near_flow_when_no_reshuffle():
    u = _universe([5.0, 10.0, 20.0], [300, 300, 300])
    prev = {"X00": 30.0, "X01": 30.0, "X02": 30.0}
    res = allocate_isin(120.0, 11.67, u, prev=prev, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    if res.mode == "purchase":
        assert abs(res.turnover_mn - abs(res.flow_mn)) < 1e-2


def test_sale_spreads_across_isins():
    u = _universe([4.0, 8.0, 12.0, 20.0], [400, 400, 400, 400])
    prev = {f"X{i:02d}": 80.0 for i in range(4)}
    res = allocate_isin(280.0, 10.0, u, prev=prev, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    assert abs(res.achieved_stock - 280.0) < 1e-3
    assert abs(res.achieved_wam - 10.0) < WAM_TOL
    assert res.n_changed >= 2


def test_large_sale_cannot_use_one_isin():
    u = _universe([3.0, 6.0, 9.0, 12.0, 20.0], [800] * 5)
    prev = {f"X{i:02d}": 200.0 for i in range(5)}
    res = allocate_isin(800.0, 10.0, u, prev=prev, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    assert abs(res.achieved_stock - 800.0) < 1e-3
    assert abs(res.achieved_wam - 10.0) < WAM_TOL
    assert res.n_changed >= 4
    assert float(np.max(np.abs(res.holdings - 200.0))) <= FLOW_SHARE * 200.0 + 1.0


def test_extreme_wam_respects_aged_lo():
    y = np.array([2.0, 10.0, 20.0])
    lo = np.array([40.0, 40.0, 40.0])
    hi = np.array([80.0, 80.0, 80.0])
    w_lo, w_hi = _extreme_wam(y, lo, hi, 150.0)
    expect_lo = (40 * 2 + 40 * 10 + 40 * 20 + 30 * 2) / 150.0
    expect_hi = (40 * 2 + 40 * 10 + 40 * 20 + 30 * 20) / 150.0
    assert abs(w_lo - expect_lo) < 1e-9
    assert abs(w_hi - expect_hi) < 1e-9


def test_cap_fill_shrinks_until_published_wam_reachable():
    """Cap-fill of a long-heavy universe pins WAM at ~14.5y. Published 6.0
    is only reachable by holding less than Σ cap·o (drop the longs)."""
    u = _universe([2.0, 5.0, 20.0], [100.0, 100.0, 400.0])
    res = allocate_isin(198.0, 6.0, u, cap=0.33)
    assert abs(res.achieved_wam - 6.0) < WAM_TOL
    assert res.achieved_stock < 99.0  # strictly below cap-fill 33+33+132
    assert res.slack > 198.0 - 99.0
    assert res.ok
    y = np.array([2.0, 5.0, 20.0])
    hi = 0.33 * np.array([100.0, 100.0, 400.0])
    feas = _wam_feasible_stock(y, hi, 6.0, 198.0)
    assert abs(res.achieved_stock - feas) < 1e-2


def test_purchase_window_does_not_leave_wam_on_the_table():
    """<1y paper cannot be bought. S* must still be able to hit W_pub
    on the 1–31y names, via shrink or reshuffle — not sit 0.3y long."""
    u = _universe([0.4, 2.0, 5.0, 20.0], [200.0, 200.0, 200.0, 800.0])
    prev = {"X00": 40.0, "X01": 66.0, "X02": 66.0, "X03": 200.0}
    res = allocate_isin(400.0, 6.0, u, prev=prev, cap=0.33, buy_lo=1.0, buy_hi=31.0)
    assert abs(res.achieved_wam - 6.0) <= WAM_TOL
    assert res.ok


def test_hits_published_wam_when_box_allows():
    u = _universe([4.0, 8.0, 20.0], [400, 400, 400])
    prev = {"X00": 80.0, "X01": 80.0, "X02": 80.0}
    res = allocate_isin(270.0, 11.0, u, prev=prev, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    assert abs(res.achieved_stock - 270.0) < 1e-3
    assert abs(res.achieved_wam - 11.0) < WAM_TOL
    assert res.ok


def test_live_wam_and_spread():
    from issuance.paths import CACHE
    path = CACHE / "official_qe_fit.parquet"
    if not path.exists() or "n_changed" not in pd.read_parquet(path).columns:
        return
    fit = pd.read_parquet(path)
    fit["month"] = pd.to_datetime(fit["month"])
    last = fit.sort_values("month").groupby(["country", "program"]).tail(1)
    last = last[last["country"].isin(["DE", "FR", "IT"])]
    core = last[last["achieved_stock_mn"] > 50_000]
    if core.empty:
        return
    assert float(core["wam_error"].abs().max()) < 0.08
    sub = fit[fit["month"] >= fit["month"].max() - pd.DateOffset(months=17)]
    if sub.empty:
        return
    assert float(sub["wam_error"].abs().quantile(0.9)) < 0.15
    big = sub[sub["flow_mn"].abs() > 2000]
    if not big.empty:
        assert float(big["n_changed"].median()) >= 4
    huge = sub[sub["flow_mn"].abs() > 5000]
    if not huge.empty:
        assert float(huge["n_changed"].min()) >= 2
        assert float(huge["n_changed"].median()) >= 5
    de24 = fit[(fit["country"] == "DE") & (fit["program"] == "PSPP")]
    de24 = de24[de24["month"].dt.year == 2024]
    if not de24.empty:
        assert float(de24["wam_error"].abs().median()) < 0.01
        assert float(de24["wam_error"].abs().max()) < 0.01
    hist = fit[(fit["country"] == "DE") & (fit["program"] == "PSPP")]
    hist = hist[hist["month"].dt.year.between(2017, 2024)]
    if not hist.empty:
        assert float(hist["wam_error"].abs().max()) < 0.01


def test_runoff_books_no_purchases_and_keeps_wam():
    from issuance.qe_alloc import phase_of
    assert phase_of("PSPP", "2023-06-01") == "partial" and phase_of("PSPP", "2023-07-01") == "runoff"
    assert phase_of("PEPP", "2024-12-01") == "partial" and phase_of("PEPP", "2025-01-01") == "runoff"
    u = _universe([3.0, 8.0, 20.0], [400, 400, 400])
    prev = {"X00": 60.0, "X01": 60.0, "X02": 60.0, "GONE": 30.0}
    # Published stock sits above the aged book (cap-bound, non-central): no buying into the cap.
    res = allocate_isin(250.0, 9.0, u, prev=prev, cap=0.33, prev_target=260.0, phase="runoff")
    assert res.mode == "runoff" and res.flow_mn == 0.0
    assert abs(res.achieved_stock - 180.0) < 1e-6
    assert abs(res.slack - 70.0) < 1e-6
    assert abs(res.achieved_wam - 9.0) < WAM_TOL


def test_partial_reinvestment_caps_purchases():
    u = _universe([3.0, 8.0, 20.0], [400, 400, 400])
    prev = {"X00": 60.0, "X01": 60.0, "X02": 60.0, "GONE": 30.0}
    # Published stock −20 while 30 matured: at most 10 was bought.
    res = allocate_isin(240.0, 9.0, u, prev=prev, cap=0.33, prev_target=260.0, phase="partial")
    assert res.achieved_stock <= 190.0 + 1e-6
    assert res.slack >= 50.0 - 1e-6


def test_opening_book_stays_inside_purchase_window():
    u = _universe([0.4, 5.0, 12.0, 40.0], [200, 200, 200, 200])
    res = allocate_isin(100.0, 8.0, u, prev=None, cap=0.50, buy_lo=1.0, buy_hi=31.0)
    by = {str(i): float(h) for i, h in zip(res.isins, res.holdings)}
    assert by["X00"] < 1e-6
    assert by["X03"] < 1e-6
    assert abs(res.achieved_wam - 8.0) <= WAM_TOL


def test_sale_widens_share_instead_of_one_isin():
    """20% cannot reach a short WAM. The box that can is the next share, not one ISIN."""
    from issuance.qe_alloc import _flow_cap_box
    aged = np.array([100.0, 100.0, 100.0])
    lo = np.zeros(3)
    hi = np.array([200.0, 200.0, 200.0])
    y = np.array([2.0, 10.0, 30.0])
    lo2, hi2 = _flow_cap_box(aged, lo, hi, 180.0, 120.0, 12.0, y)
    width = np.maximum(aged - lo2, hi2 - aged)
    assert float(width.max()) <= 0.50 * 120.0 + 1e-6
    assert float(width.max()) > FLOW_SHARE * 120.0 + 1e-6
