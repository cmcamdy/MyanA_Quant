"""风控模块测试"""

import pytest
import pandas as pd
import numpy as np

from strategies.risk import RiskConfig, RiskManager
from strategies.base import Signal, SignalType, Portfolio, Position


@pytest.fixture
def portfolio_with_position():
    """持有一只股票的组合"""
    p = Portfolio(cash=50000, equity=100000)
    pos = Position(symbol="000001.SZ", quantity=1000, avg_cost=10.0, market_value=10000)
    p.positions["000001.SZ"] = pos
    return p


@pytest.fixture
def bar_data():
    return {"000001.SZ": pd.Series({"close": 10.0, "high": 10.5, "low": 9.5})}


class TestRiskConfig:

    def test_defaults_all_none(self):
        cfg = RiskConfig()
        assert cfg.stop_loss_pct is None
        assert cfg.take_profit_pct is None
        assert cfg.max_positions is None

    def test_custom_values(self):
        cfg = RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.15, max_positions=3)
        assert cfg.stop_loss_pct == 0.05
        assert cfg.take_profit_pct == 0.15
        assert cfg.max_positions == 3

    def test_needs_atr_false(self):
        cfg = RiskConfig(stop_loss_pct=0.05)
        assert not cfg.needs_atr

    def test_needs_atr_true(self):
        cfg = RiskConfig(stop_loss_atr_multiplier=2.0)
        assert cfg.needs_atr


class TestRiskManagerStopLoss:

    def test_fixed_pct_stop_loss_triggered(self, portfolio_with_position, bar_data):
        cfg = RiskConfig(stop_loss_pct=0.05)
        rm = RiskManager(cfg)
        # 价格跌到 9.4 < 10 * 0.95 = 9.5 → 触发止损
        bar_data["000001.SZ"]["close"] = 9.4
        pos = portfolio_with_position.get_position("000001.SZ")
        pos.market_value = 9400
        signals = rm.check_exits(portfolio_with_position, bar_data)
        assert len(signals) == 1
        assert signals[0].type == SignalType.SELL
        assert signals[0].symbol == "000001.SZ"
        assert "止损" in signals[0].reason

    def test_fixed_pct_stop_loss_not_triggered(self, portfolio_with_position, bar_data):
        cfg = RiskConfig(stop_loss_pct=0.05)
        rm = RiskManager(cfg)
        # 价格 10.0 > 10 * 0.95 = 9.5 → 不触发
        signals = rm.check_exits(portfolio_with_position, bar_data)
        assert len(signals) == 0

    def test_atr_stop_loss_triggered(self, portfolio_with_position):
        cfg = RiskConfig(stop_loss_atr_multiplier=2.0)
        rm = RiskManager(cfg)
        bar_data = {"000001.SZ": pd.Series({"close": 7.9})}
        atr = {"000001.SZ": 1.0}
        pos = portfolio_with_position.get_position("000001.SZ")
        pos.market_value = 7900
        # 止损价 = 10 - 2.0*1.0 = 8.0, 当前 7.9 < 8.0
        signals = rm.check_exits(portfolio_with_position, bar_data, atr)
        assert len(signals) == 1
        assert "ATR止损" in signals[0].reason


class TestRiskManagerTrailingStop:

    def test_trailing_stop_triggered(self, portfolio_with_position):
        cfg = RiskConfig(trailing_stop_pct=0.05)
        rm = RiskManager(cfg)
        rm.register_trailing_stop("000001.SZ", 10.0)
        # 追踪止损位 = 10 * 0.95 = 9.5
        # 价格涨到 12 → 追踪位更新为 12 * 0.95 = 11.4
        bar_data = {"000001.SZ": pd.Series({"close": 12.0})}
        pos = portfolio_with_position.get_position("000001.SZ")
        pos.market_value = 12000
        rm.check_exits(portfolio_with_position, bar_data)
        # 然后价格回落到 11.0 < 11.4 → 触发
        bar_data2 = {"000001.SZ": pd.Series({"close": 11.0})}
        pos.market_value = 11000
        signals = rm.check_exits(portfolio_with_position, bar_data2)
        assert len(signals) == 1
        assert "追踪止损" in signals[0].reason

    def test_trailing_stop_not_triggered_during_rise(self, portfolio_with_position):
        cfg = RiskConfig(trailing_stop_pct=0.05)
        rm = RiskManager(cfg)
        rm.register_trailing_stop("000001.SZ", 10.0)
        bar_data = {"000001.SZ": pd.Series({"close": 11.0})}
        pos = portfolio_with_position.get_position("000001.SZ")
        pos.market_value = 11000
        signals = rm.check_exits(portfolio_with_position, bar_data)
        assert len(signals) == 0


class TestRiskManagerTakeProfit:

    def test_fixed_pct_take_profit_triggered(self, portfolio_with_position):
        cfg = RiskConfig(take_profit_pct=0.15)
        rm = RiskManager(cfg)
        # 价格涨到 11.6 >= 10 * 1.15 = 11.5
        bar_data = {"000001.SZ": pd.Series({"close": 11.6})}
        pos = portfolio_with_position.get_position("000001.SZ")
        pos.market_value = 11600
        signals = rm.check_exits(portfolio_with_position, bar_data)
        assert len(signals) == 1
        assert "止盈" in signals[0].reason

    def test_take_profit_not_triggered(self, portfolio_with_position, bar_data):
        cfg = RiskConfig(take_profit_pct=0.15)
        rm = RiskManager(cfg)
        # 价格 10.0 < 11.5 → 不触发
        signals = rm.check_exits(portfolio_with_position, bar_data)
        assert len(signals) == 0

    def test_atr_take_profit_triggered(self, portfolio_with_position):
        cfg = RiskConfig(take_profit_atr_multiplier=3.0)
        rm = RiskManager(cfg)
        atr = {"000001.SZ": 1.0}
        # 目标 = 10 + 3.0*1.0 = 13.0, 当前 13.5 > 13.0
        bar_data = {"000001.SZ": pd.Series({"close": 13.5})}
        pos = portfolio_with_position.get_position("000001.SZ")
        pos.market_value = 13500
        signals = rm.check_exits(portfolio_with_position, bar_data, atr)
        assert len(signals) == 1


class TestRiskManagerPositionLimits:

    def test_max_positions_limits_buys(self):
        cfg = RiskConfig(max_positions=1)
        rm = RiskManager(cfg)
        portfolio = Portfolio(cash=50000, equity=100000)
        portfolio.positions["000001.SZ"] = Position(symbol="000001.SZ", quantity=1000, avg_cost=10.0, market_value=10000)
        signals = [
            Signal(type=SignalType.BUY, symbol="000002.SZ"),
            Signal(type=SignalType.SELL, symbol="000001.SZ"),
        ]
        bar_data = {
            "000001.SZ": pd.Series({"close": 10.0}),
            "000002.SZ": pd.Series({"close": 20.0}),
        }
        filtered = rm.check_signals(signals, portfolio, bar_data)
        # 买入被过滤，卖出保留
        assert len(filtered) == 1
        assert filtered[0].type == SignalType.SELL

    def test_within_limits_passes_through(self):
        cfg = RiskConfig(max_positions=5)
        rm = RiskManager(cfg)
        portfolio = Portfolio(cash=100000, equity=100000)
        signals = [Signal(type=SignalType.BUY, symbol="000001.SZ")]
        bar_data = {"000001.SZ": pd.Series({"close": 10.0})}
        filtered = rm.check_signals(signals, portfolio, bar_data)
        assert len(filtered) == 1


class TestRiskManagerDrawdownLimit:

    def test_max_drawdown_triggers_liquidation(self):
        cfg = RiskConfig(max_portfolio_drawdown=0.10)
        rm = RiskManager(cfg)
        portfolio = Portfolio(cash=50000, equity=90000)
        portfolio.positions["000001.SZ"] = Position(symbol="000001.SZ", quantity=1000, avg_cost=10.0, market_value=10000)
        rm._peak_equity = 100000
        bar_data = {"000001.SZ": pd.Series({"close": 10.0})}
        signals = rm.check_exits(portfolio, bar_data)
        assert len(signals) == 1
        assert "最大回撤" in signals[0].reason


class TestRiskManagerDailyLoss:

    def test_daily_loss_limit_blocks_trading(self):
        cfg = RiskConfig(daily_loss_limit_pct=0.05)
        rm = RiskManager(cfg)
        portfolio = Portfolio(cash=90000, equity=94000)
        rm._daily_start_equity = 100000
        rm._last_date = pd.Timestamp("2024-01-02")
        signals = [Signal(type=SignalType.BUY, symbol="000001.SZ")]
        bar_data = {"000001.SZ": pd.Series({"close": 10.0})}
        current_time = pd.Timestamp("2024-01-02 10:00")
        filtered = rm.check_signals(signals, portfolio, bar_data, current_time=current_time)
        # 亏损 6% > 5%，买入被阻止
        assert len(filtered) == 0

    def test_new_day_resets(self):
        cfg = RiskConfig(daily_loss_limit_pct=0.05)
        rm = RiskManager(cfg)
        portfolio = Portfolio(cash=90000, equity=94000)
        rm._daily_start_equity = 100000
        rm._last_date = pd.Timestamp("2024-01-01")
        signals = [Signal(type=SignalType.BUY, symbol="000001.SZ")]
        bar_data = {"000001.SZ": pd.Series({"close": 10.0})}
        current_time = pd.Timestamp("2024-01-02")
        filtered = rm.check_signals(signals, portfolio, bar_data, current_time=current_time)
        # 新的一天，重置
        assert len(filtered) == 1


class TestRiskManagerReset:

    def test_reset_clears_state(self):
        cfg = RiskConfig(trailing_stop_pct=0.05)
        rm = RiskManager(cfg)
        rm.register_trailing_stop("000001.SZ", 10.0)
        rm._peak_equity = 100000
        rm._daily_start_equity = 95000
        rm.reset()
        assert rm._peak_equity == 0.0
        assert len(rm._trailing_stops) == 0
        assert rm._daily_start_equity == 0.0
