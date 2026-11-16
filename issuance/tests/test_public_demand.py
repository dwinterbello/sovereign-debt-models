from issuance.public_demand import allocate_tenor
from issuance.tenor import BUCKET_MID, BUCKET_ORDER


def test_allocator_matches_stock_and_wam():
    outstanding = {b: 100.0 for b in BUCKET_ORDER}
    target_stock = 200.0
    target_wam = 8.0
    alloc = allocate_tenor(target_stock, target_wam, outstanding)
    total = sum(alloc.values())
    wam = sum(alloc[b] * BUCKET_MID[b] for b in BUCKET_ORDER) / total
    assert abs(total - target_stock) < 1e-3
    assert abs(wam - target_wam) < 0.15


def test_allocator_respects_zero_universe():
    alloc = allocate_tenor(100.0, 7.0, {})
    assert sum(alloc.values()) == 0
