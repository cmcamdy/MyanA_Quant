"""动量指标测试 (RSI, KDJ, WR, CCI)"""

import pytest
import pandas as pd
import numpy as np

from analysis.indicators.momentum import rsi, kdj, wr, cci


@pytest.fixture
def close_series():
    return pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


@pytest.fixture
def ohlcv_df():
    return pd.DataFrame({
        'open': [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'high': [10.5, 11.5, 12.5, 11.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        'low': [9.5, 10.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
        'close': [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'volume': [1000] * 10,
    })


class TestRSI:

    def test_rsi_basic(self, close_series):
        result = rsi(close_series, period=14)
        assert f'rsi_14' in result.columns
        assert len(result) == len(close_series)

    def test_rsi_default_period(self, close_series):
        result = rsi(close_series)
        assert 'rsi_14' in result.columns

    def test_rsi_range(self, close_series):
        result = rsi(close_series, period=5)
        valid = result['rsi_5'].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_monotone_up(self):
        close = pd.Series(range(1, 100), dtype=float)
        result = rsi(close, period=14)
        assert result['rsi_14'].iloc[-1] > 90

    def test_rsi_monotone_down(self):
        close = pd.Series(range(50, 1, -1), dtype=float)
        result = rsi(close, period=14)
        # 单调递减序列RSI应接近0
        assert result['rsi_14'].iloc[-1] < 10

    def test_rsi_constant_input(self):
        close = pd.Series([10.0] * 30)
        result = rsi(close, period=14)
        # 常数序列无涨跌，RSI=50
        valid = result['rsi_14'].dropna()
        np.testing.assert_allclose(valid.values, 50.0, atol=1e-10)

    def test_rsi_custom_period(self, close_series):
        result = rsi(close_series, period=5)
        assert 'rsi_5' in result.columns


class TestKDJ:

    def test_kdj_basic(self, ohlcv_df):
        result = kdj(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        assert 'k' in result.columns
        assert 'd' in result.columns
        assert 'j' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_kdj_j_formula(self, ohlcv_df):
        result = kdj(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        expected_j = 3 * result['k'] - 2 * result['d']
        np.testing.assert_allclose(result['j'].values, expected_j.values, atol=1e-10)

    def test_kdj_k_range(self, ohlcv_df):
        result = kdj(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], n=5)
        valid_k = result['k'].dropna()
        # K值通常在0-100之间，但J值可以超出
        assert (valid_k >= -20).all() and (valid_k <= 120).all()

    def test_kdj_flat_price(self):
        """价格不波动时RSV=50，KDJ趋向50"""
        high = pd.Series([10.5] * 20)
        low = pd.Series([9.5] * 20)
        close = pd.Series([10.0] * 20)
        result = kdj(high, low, close, n=9)
        # 稳定后K和D应接近50
        assert abs(result['k'].iloc[-1] - 50) < 1
        assert abs(result['d'].iloc[-1] - 50) < 1

    def test_kdj_custom_params(self, ohlcv_df):
        result = kdj(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], n=5, m1=2, m2=2)
        assert 'k' in result.columns

    def test_kdj_zero_range(self):
        """最高价等于最低价时不报错"""
        high = pd.Series([10.0] * 10)
        low = pd.Series([10.0] * 10)
        close = pd.Series([10.0] * 10)
        result = kdj(high, low, close, n=5)
        assert not result['k'].isna().all()


class TestWR:

    def test_wr_basic(self, ohlcv_df):
        result = wr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        assert 'wr_5' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_wr_default_period(self, ohlcv_df):
        result = wr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        assert 'wr_14' in result.columns

    def test_wr_range(self, ohlcv_df):
        result = wr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        valid = result['wr_5'].dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_wr_at_high(self):
        """收盘价等于最高价时WR=0"""
        high = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        low = pd.Series([8.0, 10.0, 12.0, 14.0, 16.0])
        close = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        result = wr(high, low, close, period=5)
        # 最后一根：close=18, HH=18, LL=8, WR = (18-18)/(18-8)*(-100) = 0
        assert abs(result['wr_5'].iloc[-1] - 0.0) < 1e-10

    def test_wr_at_low(self):
        """收盘价等于最低价时WR=-100"""
        high = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        low = pd.Series([8.0, 10.0, 12.0, 14.0, 16.0])
        close = pd.Series([8.0, 10.0, 12.0, 14.0, 16.0])
        result = wr(high, low, close, period=5)
        # 最后一根：close=16, HH=18, LL=16, WR = (18-16)/(18-16)*(-100) = -100
        # 但 LL=8（5周期最低），所以 WR = (18-16)/(18-8)*(-100) = -20
        # 改用 close=LL 的情况测试
        close2 = pd.Series([8.0, 8.0, 8.0, 8.0, 8.0])
        result2 = wr(high, low, close2, period=5)
        # close=8=LL, WR = (18-8)/(18-8)*(-100) = -100
        assert abs(result2['wr_5'].iloc[-1] - (-100.0)) < 1e-10

    def test_wr_zero_range(self):
        """最高=最低时不报错"""
        high = pd.Series([10.0] * 10)
        low = pd.Series([10.0] * 10)
        close = pd.Series([10.0] * 10)
        result = wr(high, low, close, period=5)
        assert not result['wr_5'].isna().all()


class TestCCI:

    def test_cci_basic(self, ohlcv_df):
        result = cci(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        assert 'cci_5' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_cci_default_period(self, ohlcv_df):
        result = cci(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        assert 'cci_14' in result.columns

    def test_cci_constant_input(self):
        high = pd.Series([10.5] * 30)
        low = pd.Series([9.5] * 30)
        close = pd.Series([10.0] * 30)
        result = cci(high, low, close, period=5)
        # 常数输入CCI应为NaN（MAD=0）
        valid = result['cci_5'].dropna()
        # MAD=0 → 分母为0 → NaN
        assert len(valid) == 0 or result['cci_5'].iloc[-1] != result['cci_5'].iloc[-1]

    def test_cci_custom_period(self, ohlcv_df):
        result = cci(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=3)
        assert 'cci_3' in result.columns

    def test_cci_unbounded(self, ohlcv_df):
        """CCI是无界指标，可正可负"""
        result = cci(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        valid = result['cci_5'].dropna()
        # 不应被限制在特定范围内
        assert valid.max() > 0 or valid.min() < 0
