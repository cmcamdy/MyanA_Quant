"""IndicatorSet 批量计算器测试"""

import pytest
import pandas as pd
import numpy as np

from analysis.indicator_set import IndicatorSet


@pytest.fixture
def ohlcv_df():
    return pd.DataFrame({
        'open': [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'high': [10.5, 11.5, 12.5, 11.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        'low': [9.5, 10.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
        'close': [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'volume': [1000, 1100, 1200, 1100, 1000, 1100, 1200, 1300, 1400, 1500],
    }, index=pd.date_range('2024-01-01', periods=10))


class TestIndicatorSetAdd:

    def test_add_single(self):
        iset = IndicatorSet()
        iset.add('ma', period=5)
        assert len(iset.specs) == 1
        assert iset.specs[0] == {'name': 'ma', 'period': 5}

    def test_add_chaining(self):
        iset = IndicatorSet()
        result = iset.add('ma', period=5).add('rsi', period=14)
        assert result is iset
        assert len(iset.specs) == 2

    def test_add_unknown_raises(self):
        iset = IndicatorSet()
        with pytest.raises(ValueError, match="未知指标"):
            iset.add('unknown_indicator')

    def test_add_duplicate(self):
        iset = IndicatorSet()
        iset.add('ma', period=5).add('ma', period=5)
        assert len(iset.specs) == 2  # 允许添加重复，compute时去重

    def test_clear(self):
        iset = IndicatorSet()
        iset.add('ma', period=5).add('rsi')
        iset.clear()
        assert len(iset.specs) == 0


class TestIndicatorSetCompute:

    def test_compute_ma(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('ma', period=3)
        result = iset.compute(ohlcv_df)
        assert 'ma_3' in result.columns
        assert 'close' in result.columns  # 原始列保留

    def test_compute_multiple(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('ma', period=3).add('ma', period=5).add('rsi', period=5)
        result = iset.compute(ohlcv_df)
        assert 'ma_3' in result.columns
        assert 'ma_5' in result.columns
        assert 'rsi_5' in result.columns

    def test_compute_preserves_original(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('ma', period=3)
        result = iset.compute(ohlcv_df)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns

    def test_compute_dedup(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('ma', period=3).add('ma', period=3)
        result = iset.compute(ohlcv_df)
        # ma_3列只出现一次
        assert list(result.columns).count('ma_3') == 1

    def test_compute_empty_set(self, ohlcv_df):
        iset = IndicatorSet()
        result = iset.compute(ohlcv_df)
        # 无指标时返回原始DataFrame
        assert list(result.columns) == list(ohlcv_df.columns)

    def test_compute_macd(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('macd')
        result = iset.compute(ohlcv_df)
        assert 'macd_dif' in result.columns
        assert 'macd_dea' in result.columns
        assert 'macd_hist' in result.columns

    def test_compute_kdj(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('kdj', n=9)
        result = iset.compute(ohlcv_df)
        assert 'k' in result.columns
        assert 'd' in result.columns
        assert 'j' in result.columns

    def test_compute_bollinger(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('bollinger', period=5)
        result = iset.compute(ohlcv_df)
        assert 'boll_mid' in result.columns

    def test_compute_atr(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('atr', period=5)
        result = iset.compute(ohlcv_df)
        assert 'atr_5' in result.columns

    def test_compute_vwap(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('vwap')
        result = iset.compute(ohlcv_df)
        assert 'vwap' in result.columns

    def test_compute_all_indicators(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('ma', period=5).add('ema', period=5).add('macd').add('rsi', period=5)
        iset.add('kdj').add('bollinger', period=5).add('atr', period=5).add('vwap')
        result = iset.compute(ohlcv_df)
        # 验证所有指标列都存在
        expected_cols = ['ma_5', 'ema_5', 'macd_dif', 'macd_dea', 'macd_hist',
                         'rsi_5', 'k', 'd', 'j', 'boll_mid', 'boll_upper', 'boll_lower',
                         'atr_5', 'vwap']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_dema(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('dema', period=5)
        result = iset.compute(ohlcv_df)
        assert 'dema_5' in result.columns

    def test_compute_sar(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('sar')
        result = iset.compute(ohlcv_df)
        assert 'sar_value' in result.columns
        assert 'sar_trend' in result.columns

    def test_compute_wr(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('wr', period=5)
        result = iset.compute(ohlcv_df)
        assert 'wr_5' in result.columns

    def test_compute_cci(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('cci', period=5)
        result = iset.compute(ohlcv_df)
        assert 'cci_5' in result.columns

    def test_compute_keltner(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('keltner')
        result = iset.compute(ohlcv_df)
        assert 'kelt_mid' in result.columns
        assert 'kelt_upper' in result.columns
        assert 'kelt_lower' in result.columns

    def test_compute_chaikin_vol(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('chaikin_vol', period=5)
        result = iset.compute(ohlcv_df)
        assert 'chaikin_vol_5' in result.columns

    def test_compute_obv(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('obv')
        result = iset.compute(ohlcv_df)
        assert 'obv' in result.columns

    def test_compute_mfi(self, ohlcv_df):
        iset = IndicatorSet()
        iset.add('mfi', period=5)
        result = iset.compute(ohlcv_df)
        assert 'mfi_5' in result.columns
