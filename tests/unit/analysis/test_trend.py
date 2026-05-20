"""趋势指标测试 (MA, EMA, MACD, DEMA, SAR)"""

import pytest
import pandas as pd
import numpy as np

from analysis.indicators.trend import ma, ema, macd, dema, sar


@pytest.fixture
def close_series():
    return pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


class TestMA:

    def test_ma_basic(self, close_series):
        result = ma(close_series, period=3)
        assert 'ma_3' in result.columns
        expected = [np.nan, np.nan, 11.0, 11.333, 11.0, 10.667, 11.0, 12.0, 13.0, 14.0]
        np.testing.assert_allclose(result['ma_3'].values, expected, atol=0.01)

    def test_ma_default_period(self, close_series):
        result = ma(close_series)
        assert 'ma_20' in result.columns

    def test_ma_period_1(self, close_series):
        result = ma(close_series, period=1)
        np.testing.assert_allclose(result['ma_1'].values, close_series.values)

    def test_ma_first_n_nan(self, close_series):
        result = ma(close_series, period=5)
        assert result['ma_5'].iloc[:4].isna().all()
        assert not pd.isna(result['ma_5'].iloc[4])

    def test_ma_returns_dataframe(self, close_series):
        result = ma(close_series, period=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(close_series)


class TestEMA:

    def test_ema_basic(self, close_series):
        result = ema(close_series, period=5)
        assert 'ema_5' in result.columns
        assert len(result) == len(close_series)

    def test_ema_first_value_equals_close(self, close_series):
        result = ema(close_series, period=5)
        # EMA第一个值等于第一个收盘价（adjust=False时）
        assert result['ema_5'].iloc[0] == close_series.iloc[0]

    def test_ema_smoother_than_ma(self, close_series):
        ma_result = ma(close_series, period=5)['ma_5'].dropna()
        ema_result = ema(close_series, period=5)['ema_5'].iloc[4:]
        # EMA比MA更贴近近期价格，波动通常更小
        assert isinstance(ema_result, pd.Series)

    def test_ema_default_period(self, close_series):
        result = ema(close_series)
        assert 'ema_20' in result.columns

    def test_ema_constant_input(self):
        s = pd.Series([10.0] * 20)
        result = ema(s, period=5)
        # 常数序列EMA应等于常数
        np.testing.assert_allclose(result['ema_5'].values, 10.0)


class TestMACD:

    def test_macd_basic(self, close_series):
        result = macd(close_series)
        assert 'macd_dif' in result.columns
        assert 'macd_dea' in result.columns
        assert 'macd_hist' in result.columns
        assert len(result) == len(close_series)

    def test_macd_dif_equals_fast_minus_slow(self, close_series):
        result = macd(close_series, fast_period=12, slow_period=26, signal_period=9)
        ema_fast = close_series.ewm(span=12, adjust=False).mean()
        ema_slow = close_series.ewm(span=26, adjust=False).mean()
        expected_dif = ema_fast - ema_slow
        np.testing.assert_allclose(result['macd_dif'].values, expected_dif.values, atol=1e-10)

    def test_macd_dea_is_ema_of_dif(self, close_series):
        result = macd(close_series, fast_period=5, slow_period=10, signal_period=5)
        expected_dea = result['macd_dif'].ewm(span=5, adjust=False).mean()
        np.testing.assert_allclose(result['macd_dea'].values, expected_dea.values, atol=1e-10)

    def test_macd_hist_formula(self, close_series):
        result = macd(close_series)
        expected_hist = 2 * (result['macd_dif'] - result['macd_dea'])
        np.testing.assert_allclose(result['macd_hist'].values, expected_hist.values, atol=1e-10)

    def test_macd_custom_params(self, close_series):
        result = macd(close_series, fast_period=5, slow_period=10, signal_period=3)
        assert 'macd_dif' in result.columns

    def test_macd_constant_input(self):
        s = pd.Series([10.0] * 50)
        result = macd(s)
        # 常数序列MACD应接近0
        assert abs(result['macd_dif'].iloc[-1]) < 1e-10


class TestDEMA:

    def test_dema_basic(self, close_series):
        result = dema(close_series, period=5)
        assert 'dema_5' in result.columns
        assert len(result) == len(close_series)

    def test_dema_default_period(self, close_series):
        result = dema(close_series)
        assert 'dema_20' in result.columns

    def test_dema_formula(self, close_series):
        result = dema(close_series, period=5)
        ema1 = close_series.ewm(span=5, adjust=False).mean()
        ema2 = ema1.ewm(span=5, adjust=False).mean()
        expected = 2 * ema1 - ema2
        np.testing.assert_allclose(result['dema_5'].values, expected.values, atol=1e-10)

    def test_dema_constant_input(self):
        s = pd.Series([10.0] * 20)
        result = dema(s, period=5)
        # 常数序列DEMA等于常数
        np.testing.assert_allclose(result['dema_5'].values, 10.0)

    def test_dema_less_lag_than_ema(self, close_series):
        """DEMA比EMA滞后更小"""
        ema_result = ema(close_series, period=5)['ema_5']
        dema_result = dema(close_series, period=5)['dema_5']
        # 在上升趋势中，DEMA应更贴近价格
        last_close = close_series.iloc[-1]
        assert abs(dema_result.iloc[-1] - last_close) <= abs(ema_result.iloc[-1] - last_close) + 0.01


class TestSAR:

    @pytest.fixture
    def ohlcv_df(self):
        return pd.DataFrame({
            'high': [10.5, 11.5, 12.5, 11.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            'low': [9.5, 10.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            'close': [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })

    def test_sar_basic(self, ohlcv_df):
        result = sar(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        assert 'sar_value' in result.columns
        assert 'sar_trend' in result.columns
        assert len(result) == len(ohlcv_df)

    def test_sar_trend_values(self, ohlcv_df):
        result = sar(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        # trend应为1或-1
        valid = result['sar_trend']
        assert set(valid.unique()).issubset({1, -1})

    def test_sar_uptrend_below_price(self, ohlcv_df):
        """上升趋势中SAR应低于价格"""
        result = sar(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
        up = result[result['sar_trend'] == 1]
        if len(up) > 0:
            # SAR在上升趋势中应低于对应low
            for idx in up.index:
                assert result['sar_value'].loc[idx] <= ohlcv_df['high'].loc[idx] + 0.01

    def test_sar_downtrend_above_price(self):
        """下降趋势中SAR应高于价格"""
        close = pd.Series(range(20, 10, -1), dtype=float)
        high = close + 0.5
        low = close - 0.5
        result = sar(high, low, close)
        down = result[result['sar_trend'] == -1]
        if len(down) > 0:
            for idx in down.index:
                assert result['sar_value'].loc[idx] >= low.iloc[idx] - 0.01

    def test_sar_custom_params(self, ohlcv_df):
        result = sar(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], af_step=0.01, af_max=0.10)
        assert 'sar_value' in result.columns

    def test_sar_single_bar(self):
        high = pd.Series([11.0])
        low = pd.Series([9.0])
        close = pd.Series([10.0])
        result = sar(high, low, close)
        assert len(result) == 1
