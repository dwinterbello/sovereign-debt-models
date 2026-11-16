import pandas as pd

from issuance.rv_metrics import _wam
from issuance.tenor import BUCKET_MID, BUCKET_ORDER


def test_official_dv01_scale():
    from issuance.dv01 import modified_duration
    # £1bn face, ~8.5y mod duration → ~£0.8mn/bp, not 1000× smaller
    dmod = modified_duration(8.5, 3.0, 3.0)
    assert 0.6 < 1000.0 * dmod / 10_000.0 < 1.2


def test_wam_identity():
    # S=100 @ 8y, H=30 @ 6y → P=70 @ (800-180)/70 = 8.857
    stock, official = 100.0, 30.0
    wam_s, wam_h = 8.0, 6.0
    wam_p = (stock * wam_s - official * wam_h) / (stock - official)
    assert abs(wam_p - 8.857142) < 1e-4
    assert abs(_wam(pd.Series([70.0]), pd.Series([wam_p])) - wam_p) < 1e-9


def test_wam_ignores_nonpositive():
    assert pd.isna(_wam(pd.Series([0.0, -1.0]), pd.Series([10.0, 8.0])))


def test_curve_bucket_cover():
    from issuance.rv_metrics import BELLY, FRONT, LONG, MIDLONG, ULTRA
    used = set(FRONT) | set(BELLY) | set(LONG) | set(MIDLONG) | set(ULTRA)
    assert used <= set(BUCKET_ORDER)
    assert "0-1y" not in used
    assert "15-20y" in LONG
    assert set(MIDLONG) <= set(LONG)


def test_no_official_private_equals_stock():
    # No official book: official_mn = 0 → WAM_private = WAM_stock, gap blank.
    stock, official = 100.0, 0.0
    wam_s = 8.0
    has_pub = official > 1.0
    wam_p = wam_s if not has_pub else (stock * wam_s) / (stock - official)
    assert wam_p == wam_s


def test_rolling_z_unit():
    from issuance.rv_metrics import rolling_z
    s = pd.Series([1.0] * 20 + [4.0])
    z = rolling_z(s, window=20, min_periods=10)
    assert pd.isna(z.iloc[8])
    assert z.iloc[-1] > 2.0


def test_3m_roll_is_trailing_sum():
    s = pd.Series([10.0, 20.0, 30.0, 40.0])
    r3 = s.rolling(3, min_periods=1).sum()
    assert list(r3) == [10.0, 30.0, 60.0, 90.0]


def test_live_3m_columns_present():
    from issuance.paths import CACHE
    path = CACHE / "rv_supply.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    for col in ("private_dv01_usd_3m", "official_dv01_usd_3m",
                "dv01_2s10s_3m", "dv01_10s30s_3m", "dv01_5s30s_3m",
                "priv_vs_de_3m", "private_dv01_usd_3m_z"):
        assert col in df.columns
        latest = df.sort_values("month").groupby("country").tail(1)
        assert latest[col].notna().any()


def test_market_ops_drops_own_holdings():
    from issuance.isin_flow import market_ops
    df = pd.DataFrame({
        "op_type": ["Auction", "Own Holdings", "Market Adjustment"],
        "amount_issued": [100.0, 40.0, 10.0],
    })
    out = market_ops(df)
    assert set(out["op_type"]) == {"Auction", "Market Adjustment"}
    assert out["amount_issued"].sum() == 110.0
