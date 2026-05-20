"""回测结果与绩效指标测试"""

import pytest
import pandas as pd
import numpy as np

from strategies.result import BacktestResult, TradeRecord, calc_performance_metrics


@pytest.fixture
def simple_equity():
    """简单递增权益曲线"""
    return pd.Series([100000, 101000, 102000, 103000, 104000, 105000],
                     index=pd.date_range('2024-01-01', periods=6))


@pytest.fixture
def loss_equity():
    """递减权益曲线"""
    return pd.Series([100000, 99000, 98000, 97000, 96000, 95000],
                     index=pd.date_range('2024-01-01', periods=6))


@pytest.fixture
def drawdown_equity():
    """有回撤的权益曲线"""
    return pd.Series([100000, 110000, 105000, 95000, 100000, 108000],
                     index=pd.date_range('2024-01-01', periods=6))


class TestCalcPerformanceMetrics:

    def test_total_return_positive(self, simple_equity):
        metrics = calc_performance_metrics(simple_equity)
        assert abs(metrics['total_return'] - 0.05) < 1e-10

    def test_total_return_negative(self, loss_equity):
        metrics = calc_performance_metrics(loss_equity)
        assert abs(metrics['total_return'] - (-0.05)) < 1e-10

    def test_annual_return(self, simple_equity):
        metrics = calc_performance_metrics(simple_equity, periods_per_year=252)
        assert metrics['annual_return'] > 0

    def test_sharpe_positive_for_gains(self, simple_equity):
        metrics = calc_performance_metrics(simple_equity)
        assert metrics['sharpe_ratio'] > 0

    def test_max_drawdown_zero(self, simple_equity):
        metrics = calc_performance_metrics(simple_equity)
        assert metrics['max_drawdown'] == 0.0

    def test_max_drawdown_with_loss(self, drawdown_equity):
        metrics = calc_performance_metrics(drawdown_equity)
        # 峰值110000 → 谷值95000, 回撤 = (110000-95000)/110000 ≈ 13.6%
        assert abs(metrics['max_drawdown'] - (15000 / 110000)) < 1e-6

    def test_max_drawdown_duration(self, drawdown_equity):
        metrics = calc_performance_metrics(drawdown_equity)
        assert metrics['max_drawdown_duration'] > 0

    def test_single_bar(self):
        eq = pd.Series([100000])
        metrics = calc_performance_metrics(eq)
        assert metrics['total_return'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0

    def test_constant_equity(self):
        eq = pd.Series([100000] * 10)
        metrics = calc_performance_metrics(eq)
        assert metrics['total_return'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0


class TestTradeRecord:

    def test_basic_record(self):
        t = TradeRecord(
            symbol='000001.SZ',
            entry_time=pd.Timestamp('2024-01-01'),
            exit_time=pd.Timestamp('2024-01-05'),
            entry_price=10.0,
            exit_price=12.0,
            quantity=100,
            pnl=200.0,
            commission=30.0,
        )
        assert t.pnl == 200.0
        assert t.symbol == '000001.SZ'

    def test_open_position(self):
        t = TradeRecord(
            symbol='000001.SZ',
            entry_time=pd.Timestamp('2024-01-01'),
            exit_time=None,
            entry_price=10.0,
            exit_price=None,
            quantity=100,
        )
        assert t.exit_time is None
        assert t.exit_price is None


class TestBacktestResult:

    def test_summary(self):
        result = BacktestResult(
            total_return=0.15,
            annual_return=0.30,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            max_drawdown_duration=10,
            win_rate=0.6,
            profit_loss_ratio=2.0,
            total_trades=10,
            equity_curve=pd.DataFrame(),
        )
        summary = result.summary()
        assert "15.00%" in summary
        assert "1.50" in summary
