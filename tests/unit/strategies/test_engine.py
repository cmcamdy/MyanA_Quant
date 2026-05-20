"""回测引擎测试"""

import pytest
import pandas as pd
import numpy as np

from strategies.base import Strategy, Signal, SignalType, Context
from strategies.engine import BacktestEngine, BacktestConfig
from strategies.result import BacktestResult


def _make_price_df(n=50, start_price=10.0, trend=0.1):
    """生成测试价格数据"""
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = [start_price + trend * i for i in range(n)]
    return pd.DataFrame({
        'open': close,
        'high': [c * 1.02 for c in close],
        'low': [c * 0.98 for c in close],
        'close': close,
        'volume': [10000] * n,
    }, index=dates)


class BuyOnBar2Strategy(Strategy):
    """第2根bar买入，第5根bar卖出"""

    def on_init(self, context):
        self._bar_count = 0

    def on_bar(self, context):
        self._bar_count += 1
        if self._bar_count == 2:
            return Signal(type=SignalType.BUY, symbol=context.symbol)
        if self._bar_count == 5:
            return Signal(type=SignalType.SELL, symbol=context.symbol)
        return None


class AlwaysHoldStrategy(Strategy):
    """永远持有"""

    def on_init(self, context):
        pass

    def on_bar(self, context):
        return None


class BuyAndHoldStrategy(Strategy):
    """买入持有"""

    def on_init(self, context):
        pass

    def on_bar(self, context):
        pos = context.portfolio.get_position(context.symbol)
        if pos.is_empty:
            return Signal(type=SignalType.BUY, symbol=context.symbol)
        return None


class SellImmediatelyStrategy(Strategy):
    """有持仓就卖出"""

    def on_init(self, context):
        pass

    def on_bar(self, context):
        pos = context.portfolio.get_position(context.symbol)
        if not pos.is_empty:
            return Signal(type=SignalType.SELL, symbol=context.symbol)
        return None


@pytest.fixture
def price_df():
    return _make_price_df(n=30, start_price=10.0, trend=0.1)


@pytest.fixture
def engine():
    return BacktestEngine(config=BacktestConfig(
        initial_capital=100000.0,
        commission=0.0003,
        slippage=0.0001,
    ))


class TestBacktestEngine:

    def test_hold_strategy(self, engine, price_df):
        result = engine.run(AlwaysHoldStrategy(), price_df, symbol='000001.SZ')
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0
        assert abs(result.total_return) < 1e-6  # 无交易，收益为0

    def test_buy_and_sell(self, engine, price_df):
        result = engine.run(BuyOnBar2Strategy(), price_df, symbol='000001.SZ')
        assert result.total_trades == 1
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price is not None
        assert trade.exit_price is not None

    def test_buy_and_hold(self, engine, price_df):
        result = engine.run(BuyAndHoldStrategy(), price_df, symbol='000001.SZ')
        assert result.total_trades == 0  # 只有买入没有卖出
        assert result.total_return != 0

    def test_equity_curve_length(self, engine, price_df):
        result = engine.run(AlwaysHoldStrategy(), price_df, symbol='000001.SZ')
        assert len(result.equity_curve) == len(price_df)

    def test_commission_deducted(self, price_df):
        """手续费会降低收益"""
        engine_no_fee = BacktestEngine(config=BacktestConfig(
            initial_capital=100000.0, commission=0.0, slippage=0.0,
        ))
        engine_with_fee = BacktestEngine(config=BacktestConfig(
            initial_capital=100000.0, commission=0.001, slippage=0.0,
        ))
        strat = BuyOnBar2Strategy()
        r1 = engine_no_fee.run(strat, price_df, symbol='000001.SZ')
        strat2 = BuyOnBar2Strategy()
        r2 = engine_with_fee.run(strat2, price_df, symbol='000001.SZ')
        # 有手续费时收益更低
        assert r2.total_return <= r1.total_return

    def test_slippage_applied(self, price_df):
        """滑点影响成交价"""
        engine_no_slip = BacktestEngine(config=BacktestConfig(
            initial_capital=100000.0, commission=0.0, slippage=0.0,
        ))
        engine_with_slip = BacktestEngine(config=BacktestConfig(
            initial_capital=100000.0, commission=0.0, slippage=0.01,
        ))
        strat1 = BuyOnBar2Strategy()
        r1 = engine_no_slip.run(strat1, price_df, symbol='000001.SZ')
        strat2 = BuyOnBar2Strategy()
        r2 = engine_with_slip.run(strat2, price_df, symbol='000001.SZ')
        # 滑点使买入价更高、卖出价更低
        assert r2.total_return <= r1.total_return

    def test_result_has_metrics(self, engine, price_df):
        result = engine.run(BuyAndHoldStrategy(), price_df, symbol='000001.SZ')
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)

    def test_result_summary(self, engine, price_df):
        result = engine.run(BuyAndHoldStrategy(), price_df, symbol='000001.SZ')
        summary = result.summary()
        assert isinstance(summary, str)
        assert "总收益率" in summary

    def test_initial_capital_preserved(self, engine, price_df):
        result = engine.run(AlwaysHoldStrategy(), price_df, symbol='000001.SZ')
        assert result.initial_capital == 100000.0


class TestBacktestConfig:

    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100000.0
        assert cfg.commission == 0.0003
        assert cfg.slippage == 0.0001

    def test_custom(self):
        cfg = BacktestConfig(initial_capital=50000.0, commission=0.001)
        assert cfg.initial_capital == 50000.0
        assert cfg.commission == 0.001

    def test_from_yaml_missing_file(self):
        cfg = BacktestConfig.from_yaml(path="/nonexistent/settings.yaml")
        assert cfg.initial_capital == 100000.0

    def test_from_yaml_exists(self):
        cfg = BacktestConfig.from_yaml(path="config/settings.yaml")
        assert cfg.initial_capital == 100000.0


class TestInitialPositions:

    def test_initial_position_cash_deducted(self, price_df):
        """初始持仓应从现金中扣除成本"""
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0, slippage=0,
            initial_positions={"TEST": {"avg_cost": 10.0, "quantity": 1000}},
        )
        engine = BacktestEngine(config)
        result = engine.run(AlwaysHoldStrategy(), price_df, symbol="TEST")
        # 现金 = 100000 - 10*1000 = 90000
        assert abs(result.equity_curve['cash'].iloc[0] - 90000.0) < 1.0

    def test_initial_position_market_value_tracked(self, price_df):
        """初始持仓市值随价格变化"""
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0, slippage=0,
            initial_positions={"TEST": {"avg_cost": 10.0, "quantity": 1000}},
        )
        engine = BacktestEngine(config)
        result = engine.run(AlwaysHoldStrategy(), price_df, symbol="TEST")
        # 价格上涨，权益应增长
        assert result.total_return > 0

    def test_initial_position_can_be_sold(self, price_df):
        """策略可以卖出初始持仓"""
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0, slippage=0,
            initial_positions={"TEST": {"avg_cost": 10.0, "quantity": 1000}},
        )
        engine = BacktestEngine(config)
        result = engine.run(SellImmediatelyStrategy(), price_df, symbol="TEST")
        # SellImmediately 应卖出初始持仓
        assert result.total_trades >= 1

    def test_strategy_start_delays_signals(self, price_df):
        """strategy_start 之前不生成信号"""
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0, slippage=0,
            initial_positions={"TEST": {"avg_cost": 10.0, "quantity": 1000}},
        )
        engine = BacktestEngine(config)
        # 策略在最后一根bar才开始，之前只跟踪持仓
        last_date = price_df.index[-1]
        result = engine.run(
            SellImmediatelyStrategy(), price_df,
            symbol="TEST", strategy_start=last_date,
        )
        # 只在最后一天才卖出
        assert result.total_trades <= 1

    def test_strategy_start_equity_curve_length(self, price_df):
        """strategy_start 不影响权益曲线长度"""
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0, slippage=0,
            initial_positions={"TEST": {"avg_cost": 10.0, "quantity": 1000}},
        )
        engine = BacktestEngine(config)
        mid_date = price_df.index[len(price_df) // 2]
        result = engine.run(
            AlwaysHoldStrategy(), price_df,
            symbol="TEST", strategy_start=mid_date,
        )
        assert len(result.equity_curve) == len(price_df)

    def test_no_initial_positions_default(self, price_df):
        """不设初始持仓时行为不变"""
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        result = engine.run(AlwaysHoldStrategy(), price_df, symbol="TEST")
        assert result.total_trades == 0
