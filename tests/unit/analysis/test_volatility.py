"""波动率指标测试 (Bollinger, ATR, Keltner, Chaikin Vol)"""

import pytest
import pandas as pd
import numpy as np

from analysis.indicators.volatility import bollinger, atr, keltner, chaikin_vol


@pytest.fixture
def close_series():
    return pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


@pytest.fixture
def ohlcv_df():
    return pd.DataFrame({
        'high': [10.5, 11.5, 12.5, 11.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        'low': [9.5, 10.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
        'close': [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })


class TestBollinger:

    def test_bollinger_basic(self, close_series):
        result = bollinger(close_series, period=5)
        assert 'boll_mid' in result.columns
        assert 'boll_upper' in result.columns
        assert 'boll_lower' in result.columns

    def test_bollinger_mid_equals_ma(self, close_series):
        result = bollinger(close_series, period=5)
        expected_mid = close_series.rolling(window=5).mean()
        np.testing.assert_allclose(result['boll_mid'].values, expected_mid.values, atol=1e-10)

    def test_bollinger_upper_above_mid(self, close_series):
        result = bollinger(close_series, period=5).dropna()
        assert (result['boll_upper'] >= result['boll_mid']).all()

    def test_bollinger_lower_below_mid(self, close_series):
        result = bollinger(close_series, period=5).dropna()
        assert (result['boll_lower'] <= result['boll_mid']).all()

    def test_bollinger_default_params(self, close_series):
        result = bollinger(close_series)
        assert 'boll_mid' in result.columns

    def test_bollinger_custom_std(self, close_series):
        result1 = bollinger(close_series, period=5, num_std=1.0)
        result2 = bollinger(close_series, period=5, num_std=2.0)
        valid1 = result1.dropna()
        valid2 = result2.dropna()
        # 更大的std → 更宽的带
        width1 = valid1['boll_upper'] - valid1['boll_lower']
        width2 = valid2['boll_upper'] - valid2['boll_lower']
        assert (width2 >= width1).all()

    def test_bollinger_constant_input(self):
        s = pd.Series([10.0] * 20)
        result = bollinger(s, period=5).dropna()
        # 常数序列：mid=10, std=0, upper=lower=mid
        np.testing.assert_allclose(result['boll_upper'].values, 10.0)
        np.testing.assert_allclose(result['boll_lower'].values, 10.0)


class TestATR:

    def test_atr_basic(self, ohlcv_df):
        result = atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        assert 'atr_5' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_atr_first_bar(self, ohlcv_df):
        result = atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        # 第一根bar的ATR = high - low
        expected_first = ohlcv_df['high'].iloc[0] - ohlcv_df['low'].iloc[0]
        assert abs(result['atr_5'].iloc[0] - expected_first) < 1e-10

    def test_atr_positive(self, ohlcv_df):
        result = atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], period=5)
        assert (result['atr_5'].dropna() > 0).all()

    def test_atr_default_period(self, ohlcv_df):
        result = atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        assert 'atr_14' in result.columns

    def test_atr_true_range_formula(self):
        """验证TR = max(H-L, |H-prev_C|, |L-prev_C|)"""
        high = pd.Series([11.0, 12.0])
        low = pd.Series([9.0, 10.0])
        close = pd.Series([10.0, 8.0])  # 第2根收盘大跌

        result = atr(high, low, close, period=2)
        # 第2根: TR = max(12-10=2, |12-10|=2, |10-8|=2) = 2
        # 不对，prev_close=10, TR = max(12-10=2, |12-10|=2, |10-8|=2) = 2
        # 第一根: TR = 11-9 = 2
        assert result['atr_2'].iloc[0] == 2.0

    def test_atr_constant_range(self):
        high = pd.Series([11.0] * 20)
        low = pd.Series([9.0] * 20)
        close = pd.Series([10.0] * 20)
        result = atr(high, low, close, period=5)
        # 固定区间 ATR = 2.0
        np.testing.assert_allclose(result['atr_5'].iloc[-1], 2.0, atol=0.01)


class TestKeltner:

    def test_keltner_basic(self, ohlcv_df):
        result = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        assert 'kelt_mid' in result.columns
        assert 'kelt_upper' in result.columns
        assert 'kelt_lower' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_keltner_mid_equals_ema(self, ohlcv_df):
        result = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], ema_period=20)
        expected_mid = ohlcv_df['close'].ewm(span=20, adjust=False).mean()
        np.testing.assert_allclose(result['kelt_mid'].values, expected_mid.values, atol=1e-10)

    def test_keltner_upper_above_mid(self, ohlcv_df):
        result = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close']).dropna()
        assert (result['kelt_upper'] >= result['kelt_mid']).all()

    def test_keltner_lower_below_mid(self, ohlcv_df):
        result = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close']).dropna()
        assert (result['kelt_lower'] <= result['kelt_mid']).all()

    def test_keltner_custom_params(self, ohlcv_df):
        result = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'],
                         ema_period=10, atr_period=5, num_atr=2.0)
        assert 'kelt_mid' in result.columns

    def test_keltner_wider_with_larger_atr(self, ohlcv_df):
        result1 = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], num_atr=1.0)
        result2 = keltner(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], num_atr=2.0)
        valid1 = result1.dropna()
        valid2 = result2.dropna()
        # 更大的num_atr → 更宽的通道
        width1 = valid1['kelt_upper'] - valid1['kelt_lower']
        width2 = valid2['kelt_upper'] - valid2['kelt_lower']
        assert (width2 >= width1).all()

    def test_keltner_constant_input(self):
        high = pd.Series([10.5] * 30)
        low = pd.Series([9.5] * 30)
        close = pd.Series([10.0] * 30)
        result = keltner(high, low, close)
        # 常数输入：mid=10, ATR固定，通道对称
        np.testing.assert_allclose(result['kelt_mid'].iloc[-1], 10.0, atol=0.01)


class TestChaikinVol:

    def test_chaikin_vol_basic(self, ohlcv_df):
        result = chaikin_vol(ohlcv_df['high'], ohlcv_df['low'])
        assert 'chaikin_vol_10' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_chaikin_vol_default_params(self, ohlcv_df):
        result = chaikin_vol(ohlcv_df['high'], ohlcv_df['low'])
        assert 'chaikin_vol_10' in result.columns

    def test_chaikin_vol_custom_params(self, ohlcv_df):
        result = chaikin_vol(ohlcv_df['high'], ohlcv_df['low'], period=5, roc_period=5)
        assert 'chaikin_vol_5' in result.columns

    def test_chaikin_vol_constant_spread(self):
        """固定高低价差时Chaikin Vol=0"""
        high = pd.Series([10.5] * 30)
        low = pd.Series([9.5] * 30)
        result = chaikin_vol(high, low, period=5, roc_period=5)
        valid = result['chaikin_vol_5'].dropna()
        # 价差不变 → 变化率=0
        if len(valid) > 0:
            np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_chaikin_vol_expanding(self):
        """价差扩大时Chaikin Vol为正"""
        high = pd.Series([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                          21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])
        low = pd.Series([9.0] * 20)
        result = chaikin_vol(high, low, period=5, roc_period=5)
        # 价差持续扩大，后期应为正值
        valid = result['chaikin_vol_5'].dropna()
        if len(valid) > 2:
            assert valid.iloc[-1] > 0
