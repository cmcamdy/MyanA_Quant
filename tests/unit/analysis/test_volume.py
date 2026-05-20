"""成交量指标测试 (VWAP, OBV, MFI)"""

import pytest
import pandas as pd
import numpy as np

from analysis.indicators.volume import vwap, obv, mfi


@pytest.fixture
def daily_df():
    return pd.DataFrame({
        'high': [10.5, 11.5, 12.5],
        'low': [9.5, 10.5, 11.5],
        'close': [10.0, 11.0, 12.0],
        'volume': [1000, 2000, 1500],
    }, index=pd.date_range('2024-01-02', periods=3))


class TestVWAP:

    def test_vwap_basic(self, daily_df):
        result = vwap(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'])
        assert 'vwap' in result.columns
        assert len(result) == len(daily_df)

    def test_vwap_first_bar(self, daily_df):
        result = vwap(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'])
        tp0 = (10.5 + 9.5 + 10.0) / 3
        assert abs(result['vwap'].iloc[0] - tp0) < 1e-10

    def test_vwap_daily_cumulative(self, daily_df):
        result = vwap(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'])
        # 日线数据每天一个bar，每天VWAP=typical_price
        tp1 = (11.5 + 10.5 + 11.0) / 3
        assert abs(result['vwap'].iloc[1] - tp1) < 1e-10

    def test_vwap_intraday_reset(self):
        """日内VWAP在次日开盘时重置"""
        dates = pd.to_datetime(['2024-01-02 10:00', '2024-01-02 14:00', '2024-01-03 10:00'])
        df = pd.DataFrame({
            'high': [10.5, 11.5, 12.5],
            'low': [9.5, 10.5, 11.5],
            'close': [10.0, 11.0, 12.0],
            'volume': [1000, 2000, 1500],
        }, index=dates)

        result = vwap(df['high'], df['low'], df['close'], df['volume'])
        # 第3根bar是新的一天，VWAP应重置
        tp3 = (12.5 + 11.5 + 12.0) / 3
        assert abs(result['vwap'].iloc[2] - tp3) < 1e-10

    def test_vwap_constant_price(self, daily_df):
        df = daily_df.copy()
        df['high'] = 10.0
        df['low'] = 10.0
        df['close'] = 10.0
        result = vwap(df['high'], df['low'], df['close'], df['volume'])
        np.testing.assert_allclose(result['vwap'].values, 10.0, atol=1e-10)

    def test_vwap_single_bar(self):
        high = pd.Series([11.0])
        low = pd.Series([9.0])
        close = pd.Series([10.0])
        volume = pd.Series([1000])
        result = vwap(high, low, close, volume)
        tp = (11.0 + 9.0 + 10.0) / 3
        assert abs(result['vwap'].iloc[0] - tp) < 1e-10


class TestOBV:

    def test_obv_basic(self, daily_df):
        result = obv(daily_df['close'], daily_df['volume'])
        assert 'obv' in result.columns
        assert len(result) == len(daily_df)

    def test_obv_first_bar_zero(self, daily_df):
        result = obv(daily_df['close'], daily_df['volume'])
        # 第一根bar无前值，diff=NaN，sign=0，OBV=0
        assert result['obv'].iloc[0] == 0.0

    def test_obv_monotone_up(self):
        close = pd.Series([10.0, 11.0, 12.0, 13.0])
        volume = pd.Series([1000, 2000, 1500, 3000])
        result = obv(close, volume)
        # 持续上涨：OBV = 0 + 2000 + 1500 + 3000 = 6500
        assert result['obv'].iloc[-1] == 6500.0

    def test_obv_monotone_down(self):
        close = pd.Series([13.0, 12.0, 11.0, 10.0])
        volume = pd.Series([1000, 2000, 1500, 3000])
        result = obv(close, volume)
        # 持续下跌：OBV = 0 - 2000 - 1500 - 3000 = -6500
        assert result['obv'].iloc[-1] == -6500.0

    def test_obv_flat_price(self):
        close = pd.Series([10.0, 10.0, 10.0, 10.0])
        volume = pd.Series([1000, 2000, 1500, 3000])
        result = obv(close, volume)
        # 价格不变：OBV始终为0
        np.testing.assert_allclose(result['obv'].values, 0.0, atol=1e-10)


class TestMFI:

    def test_mfi_basic(self, daily_df):
        result = mfi(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'])
        assert 'mfi_14' in result.columns
        assert len(result) == len(daily_df)

    def test_mfi_default_period(self, daily_df):
        result = mfi(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'])
        assert 'mfi_14' in result.columns

    def test_mfi_custom_period(self, daily_df):
        result = mfi(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'], period=5)
        assert 'mfi_5' in result.columns

    def test_mfi_range(self, daily_df):
        result = mfi(daily_df['high'], daily_df['low'], daily_df['close'], daily_df['volume'], period=2)
        valid = result['mfi_2'].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_mfi_monotone_up(self):
        high = pd.Series(range(10, 30), dtype=float)
        low = pd.Series(range(9, 29), dtype=float)
        close = pd.Series(range(10, 30), dtype=float)
        volume = pd.Series([1000] * 20, dtype=float)
        result = mfi(high, low, close, volume, period=5)
        # 持续上涨MFI应接近100
        assert result['mfi_5'].iloc[-1] > 90

    def test_mfi_monotone_down(self):
        high = pd.Series(range(30, 10, -1), dtype=float)
        low = pd.Series(range(29, 9, -1), dtype=float)
        close = pd.Series(range(30, 10, -1), dtype=float)
        volume = pd.Series([1000] * 20, dtype=float)
        result = mfi(high, low, close, volume, period=5)
        # 持续下跌MFI应接近0
        assert result['mfi_5'].iloc[-1] < 10
