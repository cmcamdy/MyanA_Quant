"""组合回测引擎测试"""

import pytest
import pandas as pd
import numpy as np

from strategies.base import Strategy, Signal, SignalType, Context
from strategies.portfolio_engine import (
    PortfolioEngine, PortfolioContext, MultiStrategy,
    EqualWeightAllocation, CustomAllocation, RebalanceConfig,
)
from strategies.engine import BacktestConfig


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
    """第一根bar买入，持有到结束"""
    def on_init(self, ctx):
        pass

    def on_bar(self, ctx):
        if ctx.current_time == ctx.bars.index[0]:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol)
        return None


class BuyFirstBarMulti(MultiStrategy):
    """多标的：第一根bar全部买入"""
    def on_init_multi(self, ctx):
        pass

    def on_bar_multi(self, ctx):
        signals = []
        if ctx.current_time == list(ctx.historical.values())[0].index[0]:
            for sym in ctx.bars:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
        return signals


class TestEqualWeightAllocation:

    def test_equal_split(self):
        alloc = EqualWeightAllocation()
        w = alloc.allocate("A", None, None, 10.0, ["A", "B", "C"])
        assert abs(w - 1.0 / 3) < 1e-10

    def test_single_symbol(self):
        alloc = EqualWeightAllocation()
        w = alloc.allocate("A", None, None, 10.0, ["A"])
        assert abs(w - 1.0) < 1e-10


class TestCustomAllocation:

    def test_custom_weights(self):
        alloc = CustomAllocation({"A": 0.6, "B": 0.4})
        assert abs(alloc.allocate("A", None, None, 10.0, []) - 0.6) < 1e-10
        assert abs(alloc.allocate("B", None, None, 10.0, []) - 0.4) < 1e-10

    def test_missing_symbol_returns_zero(self):
        alloc = CustomAllocation({"A": 0.6})
        assert alloc.allocate("B", None, None, 10.0, []) == 0.0


class TestPortfolioEngine:

    def test_single_symbol_matches_backtest_engine(self):
        """单标的时结果应与 BacktestEngine 一致"""
        from strategies.engine import BacktestEngine
        df = _make_df(20)
        data = {"000001.SZ": df}

        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        result_single = engine.run(BuyFirstBarStrategy(), df, symbol="000001.SZ")

        pengine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result_multi = pengine.run(BuyFirstBarStrategy(), data)

        assert abs(result_single.total_return - result_multi.total_return) < 0.01

    def test_multi_symbol_with_legacy_strategy(self):
        """普通 Strategy 在多标的上逐个调用"""
        data = {"A": _make_df(20, 10.0), "B": _make_df(20, 20.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(BuyFirstBarStrategy(), data)
        # 买入并持有，权益应增长
        assert result.total_return > 0
        assert result.per_symbol_equity is not None
        assert "A" in result.per_symbol_equity

    def test_multi_symbol_with_multi_strategy(self):
        """MultiStrategy 一次返回多标的信号"""
        data = {"A": _make_df(20, 10.0), "B": _make_df(20, 20.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(BuyFirstBarMulti(), data)
        # 两个标的都买入并持有
        assert result.total_return > 0
        assert result.per_symbol_trades is not None

    def test_per_symbol_equity_tracked(self):
        data = {"A": _make_df(10, 10.0), "B": _make_df(10, 20.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(BuyFirstBarMulti(), data)
        assert "A" in result.per_symbol_equity
        assert "B" in result.per_symbol_equity
        assert len(result.per_symbol_equity["A"]) == 10

    def test_per_symbol_trades_tracked(self):
        data = {"A": _make_df(10, 10.0), "B": _make_df(10, 20.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(BuyFirstBarMulti(), data)
        assert "A" in result.per_symbol_trades
        assert "B" in result.per_symbol_trades

    def test_portfolio_equity_curve_length(self):
        data = {"A": _make_df(10, 10.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(BuyFirstBarStrategy(), data)
        assert len(result.equity_curve) == 10

    def test_allocation_policy_respected(self):
        data = {"A": _make_df(10, 10.0), "B": _make_df(10, 20.0)}
        alloc = CustomAllocation({"A": 1.0, "B": 0.0})
        engine = PortfolioEngine(
            BacktestConfig(commission=0, slippage=0),
            allocation=alloc,
        )
        result = engine.run(BuyFirstBarMulti(), data)
        # B 分配权重为0，不应买入
        b_trades = result.per_symbol_trades.get("B", [])
        assert len(b_trades) == 0

    def test_empty_signals_produce_no_trades(self):
        """无信号策略不产生交易"""
        class HoldStrategy(Strategy):
            def on_init(self, ctx):
                pass
            def on_bar(self, ctx):
                return None

        data = {"A": _make_df(10, 10.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(HoldStrategy(), data)
        assert result.total_trades == 0


class TestRebalance:

    def test_weekly_rebalance_generates_trades(self):
        """按周再平衡应产生交易"""
        data = {"A": _make_df(30, 10.0), "B": _make_df(30, 20.0)}
        rebalance_cfg = RebalanceConfig(frequency='W', record_trades=True)
        engine = PortfolioEngine(
            BacktestConfig(commission=0, slippage=0),
            rebalance=rebalance_cfg,
        )
        result = engine.run(BuyFirstBarMulti(), data)
        # 应有再平衡产生的交易
        assert result.total_trades > 0

    def test_drift_rebalance_triggers(self):
        """漂移超过阈值应触发再平衡"""
        data = {"A": _make_df(30, 10.0, drift=0.5), "B": _make_df(30, 20.0, drift=-0.1)}
        rebalance_cfg = RebalanceConfig(drift_threshold=0.01, record_trades=True)
        engine = PortfolioEngine(
            BacktestConfig(commission=0, slippage=0),
            rebalance=rebalance_cfg,
        )
        result = engine.run(BuyFirstBarMulti(), data)
        # 漂移再平衡应触发交易
        assert result.total_trades > 0

    def test_no_rebalance_by_default(self):
        """不设置 RebalanceConfig 时不触发再平衡"""
        data = {"A": _make_df(10, 10.0), "B": _make_df(10, 20.0)}
        engine = PortfolioEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(BuyFirstBarMulti(), data)
        # Without rebalance, per-symbol trades should be empty (buy-and-hold, no sell)
        assert len(result.per_symbol_trades["A"]) == 0
        assert len(result.per_symbol_trades["B"]) == 0
        # Both positions should still be held (equity > initial cash allocation)
        assert result.per_symbol_equity["A"].iloc[-1] > 0
        assert result.per_symbol_equity["B"].iloc[-1] > 0
