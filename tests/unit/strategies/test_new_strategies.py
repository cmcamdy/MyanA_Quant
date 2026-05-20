"""新增策略单元测试"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from strategies.base import SignalType
from strategies.examples.sar import SARStrategy
from strategies.examples.rsi import RSIStrategy
from strategies.examples.bollinger import BollingerStrategy


def _make_ohlcv_df(n_rows=200):
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='B')
    close = 10.0 + np.cumsum(np.random.randn(n_rows) * 0.1)
    return pd.DataFrame({
        'open': close + np.random.randn(n_rows) * 0.05,
        'high': close + abs(np.random.randn(n_rows) * 0.1),
        'low': close - abs(np.random.randn(n_rows) * 0.1),
        'close': close,
        'volume': np.random.randint(1e6, 1e8, n_rows).astype(float),
        'amount': np.random.uniform(1e8, 1e10, n_rows),
    }, index=dates)


class TestSARStrategy:
    def test_on_init(self):
        strat = SARStrategy()
        from strategies.base import Context, Portfolio
        df = _make_ohlcv_df()
        ctx = Context(bar=df.iloc[0], bars=df.iloc[:1], portfolio=Portfolio(),
                      current_time=df.index[0], symbol="TEST")
        strat.on_init(ctx)
        assert len(strat.indicator_specs) == 1
        assert strat.indicator_specs[0]['name'] == 'sar'


class TestRSIStrategy:
    def test_on_init(self):
        strat = RSIStrategy(period=14)
        from strategies.base import Context, Portfolio
        df = _make_ohlcv_df()
        ctx = Context(bar=df.iloc[0], bars=df.iloc[:1], portfolio=Portfolio(),
                      current_time=df.index[0], symbol="TEST")
        strat.on_init(ctx)
        assert len(strat.indicator_specs) == 1
        assert strat.indicator_specs[0]['name'] == 'rsi'

    def test_custom_params(self):
        strat = RSIStrategy(period=6, oversold=20, overbought=80)
        assert strat.period == 6
        assert strat.oversold == 20
        assert strat.overbought == 80


class TestBollingerStrategy:
    def test_on_init(self):
        strat = BollingerStrategy(period=20)
        from strategies.base import Context, Portfolio
        df = _make_ohlcv_df()
        ctx = Context(bar=df.iloc[0], bars=df.iloc[:1], portfolio=Portfolio(),
                      current_time=df.index[0], symbol="TEST")
        strat.on_init(ctx)
        assert len(strat.indicator_specs) == 1
        assert strat.indicator_specs[0]['name'] == 'bollinger'

    def test_custom_params(self):
        strat = BollingerStrategy(period=10, num_std=1.5)
        assert strat.period == 10
        assert strat.num_std == 1.5
