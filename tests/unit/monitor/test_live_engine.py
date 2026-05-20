"""LiveEngine 单元测试"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from monitor.config import MonitorConfig
from monitor.live_engine import LiveEngine
from strategies.base import Strategy, Signal, SignalType, Context


def _make_df(n=20, start_price=10.0, drift=0.1):
    """生成合成OHLCV数据"""
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = [start_price + i * drift + np.sin(i / 3) * 0.5 for i in range(n)]
    return pd.DataFrame({
        'open': close,
        'high': [c * 1.02 for c in close],
        'low': [c * 0.98 for c in close],
        'close': close,
        'volume': [10000 + i * 100 for i in range(n)],
    }, index=dates)


class BuyFirstBarStrategy(Strategy):
    """第一根bar买入"""
    def on_init(self, ctx):
        pass

    def on_bar(self, ctx):
        if ctx.current_time == ctx.bars.index[0]:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol)
        return None


class TestLiveEngineProcessSingleBar:

    def test_single_buy_signal(self):
        """处理单个bar产生买入信号"""
        config = MonitorConfig(
            symbols=["TEST.SZ"],
            freq="1d",
            poll_interval=60,
            commission=0,
            slippage=0,
        )
        strategy = BuyFirstBarStrategy()
        engine = LiveEngine(config, strategy)

        # 手动设置内部状态（模拟 initialize 后）
        from strategies.base import Portfolio
        engine._portfolio = Portfolio(cash=100000.0, equity=100000.0)
        engine._open_trades = {}
        engine._state = __import__("monitor.state", fromlist=["MonitorState"]).MonitorState(
            monitor_id="test", cash=100000.0,
        )
        engine._alert_manager = __import__("monitor.alerts", fromlist=["AlertManager"]).AlertManager()
        engine._recent_signals = []
        engine._last_known = {}

        df = _make_df(5)
        engine._historical_data = {"TEST.SZ": df}

        bar_data = {"TEST.SZ": df.iloc[0]}
        engine._process_single_bar(df.index[0], ["TEST.SZ"], {"TEST.SZ": df.iloc[:1]})

        pos = engine._portfolio.get_position("TEST.SZ")
        assert not pos.is_empty
        assert pos.quantity > 0

    def test_no_signal_for_subsequent_bars(self):
        """后续bar不产生新信号"""
        config = MonitorConfig(
            symbols=["TEST.SZ"],
            freq="1d",
            poll_interval=60,
            commission=0,
            slippage=0,
        )
        strategy = BuyFirstBarStrategy()
        engine = LiveEngine(config, strategy)

        from strategies.base import Portfolio
        from monitor.state import MonitorState
        from monitor.alerts import AlertManager

        engine._portfolio = Portfolio(cash=100000.0, equity=100000.0)
        engine._open_trades = {}
        engine._state = MonitorState(monitor_id="test", cash=100000.0)
        engine._alert_manager = AlertManager()
        engine._recent_signals = []
        engine._last_known = {}

        df = _make_df(5)
        engine._historical_data = {"TEST.SZ": df}

        # Process first bar (buy)
        engine._process_single_bar(df.index[0], ["TEST.SZ"], {"TEST.SZ": df.iloc[:1]})

        # Process second bar (no signal)
        engine._process_single_bar(df.index[1], ["TEST.SZ"], {"TEST.SZ": df.iloc[1:2]})

        # Should still have only one position
        pos = engine._portfolio.get_position("TEST.SZ")
        assert not pos.is_empty
        # Equity should update
        assert len(engine._state.equity_history) == 2


class TestLiveEngineStatePersistence:

    def test_state_saved_after_process(self):
        """处理后状态应正确更新"""
        config = MonitorConfig(
            symbols=["TEST.SZ"],
            freq="1d",
            poll_interval=60,
            commission=0,
            slippage=0,
        )
        strategy = BuyFirstBarStrategy()
        engine = LiveEngine(config, strategy)

        from strategies.base import Portfolio
        from monitor.state import MonitorState
        from monitor.alerts import AlertManager

        engine._portfolio = Portfolio(cash=100000.0, equity=100000.0)
        engine._open_trades = {}
        engine._state = MonitorState(monitor_id="test", cash=100000.0)
        engine._alert_manager = AlertManager()
        engine._recent_signals = []
        engine._last_known = {}

        df = _make_df(5)
        engine._historical_data = {"TEST.SZ": df}

        engine._process_single_bar(df.index[0], ["TEST.SZ"], {"TEST.SZ": df.iloc[:1]})

        # State should reflect the buy
        assert "TEST.SZ" in engine._state.positions
        assert engine._state.positions["TEST.SZ"]["quantity"] > 0
        assert len(engine._state.equity_history) == 1
