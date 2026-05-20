"""可视化模块测试"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，不需要GUI

from visualization.candlestick import plot_kline
from visualization.equity import plot_equity, plot_drawdown
from visualization.trade import plot_trade_pnl, plot_trade_hold_period
from strategies.result import BacktestResult, TradeRecord


@pytest.fixture
def ohlcv_df():
    n = 60
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = [10 + i * 0.1 + np.sin(i / 5) for i in range(n)]
    return pd.DataFrame({
        'open': close,
        'high': [c * 1.02 for c in close],
        'low': [c * 0.98 for c in close],
        'close': close,
        'volume': [10000 + i * 100 for i in range(n)],
        'ma_5': pd.Series(close).rolling(5).mean().values,
        'ma_20': pd.Series(close).rolling(20).mean().values,
    }, index=dates)


@pytest.fixture
def backtest_result():
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    equity = [100000 + i * 500 + np.sin(i / 3) * 2000 for i in range(60)]
    equity_curve = pd.DataFrame({
        'equity': equity,
        'cash': [e * 0.3 for e in equity],
        'market_value': [e * 0.7 for e in equity],
    }, index=dates)

    trades = [
        TradeRecord('000001.SZ', dates[5], dates[15], 10.0, 12.0, 1000, 2000.0, 36.0),
        TradeRecord('000001.SZ', dates[20], dates[35], 11.5, 10.0, 1000, -1500.0, 34.5),
        TradeRecord('000001.SZ', dates[40], dates[55], 10.5, 13.0, 1000, 2500.0, 36.5),
    ]

    return BacktestResult(
        total_return=0.15, annual_return=0.30, sharpe_ratio=1.5,
        max_drawdown=0.08, max_drawdown_duration=10,
        win_rate=0.67, profit_loss_ratio=2.0, total_trades=3,
        equity_curve=equity_curve, trades=trades,
        initial_capital=100000.0,
    )


# ─── K线图测试 ───

class TestPlotKline:

    def test_mpl_returns_figure(self, ohlcv_df):
        fig = plot_kline(ohlcv_df, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plotly_returns_figure(self, ohlcv_df):
        pytest.importorskip('plotly')
        import plotly.graph_objects as go
        fig = plot_kline(ohlcv_df, engine='plotly')
        assert isinstance(fig, go.Figure)

    def test_with_indicators(self, ohlcv_df):
        fig = plot_kline(ohlcv_df, indicators=['ma_5', 'ma_20'], engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_trades(self, ohlcv_df, backtest_result):
        fig = plot_kline(ohlcv_df, trades=backtest_result.trades, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_title(self, ohlcv_df):
        fig = plot_kline(ohlcv_df, title='测试K线', engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_volume(self, ohlcv_df):
        df_no_vol = ohlcv_df.drop(columns=['volume'])
        fig = plot_kline(df_no_vol, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_missing_indicator_ignored(self, ohlcv_df):
        fig = plot_kline(ohlcv_df, indicators=['nonexistent'], engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)


# ─── 权益曲线测试 ───

class TestPlotEquity:

    def test_mpl_returns_figure(self, backtest_result):
        fig = plot_equity(backtest_result, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plotly_returns_figure(self, backtest_result):
        pytest.importorskip('plotly')
        import plotly.graph_objects as go
        fig = plot_equity(backtest_result, engine='plotly')
        assert isinstance(fig, go.Figure)

    def test_with_benchmark(self, backtest_result):
        dates = backtest_result.equity_curve.index
        bm = pd.DataFrame({'close': [10 + i * 0.05 for i in range(len(dates))]}, index=dates)
        fig = plot_equity(backtest_result, benchmark_df=bm, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotDrawdown:

    def test_mpl_returns_figure(self, backtest_result):
        fig = plot_drawdown(backtest_result, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plotly_returns_figure(self, backtest_result):
        pytest.importorskip('plotly')
        import plotly.graph_objects as go
        fig = plot_drawdown(backtest_result, engine='plotly')
        assert isinstance(fig, go.Figure)


# ─── 交易分布图测试 ───

class TestPlotTradePnl:

    def test_mpl_returns_figure(self, backtest_result):
        fig = plot_trade_pnl(backtest_result, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plotly_returns_figure(self, backtest_result):
        pytest.importorskip('plotly')
        import plotly.graph_objects as go
        fig = plot_trade_pnl(backtest_result, engine='plotly')
        assert isinstance(fig, go.Figure)

    def test_no_trades(self):
        result = BacktestResult(
            total_return=0, annual_return=0, sharpe_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            win_rate=0, profit_loss_ratio=0, total_trades=0,
            equity_curve=pd.DataFrame({'equity': [100000]}, index=pd.date_range('2024-01-01', periods=1)),
            trades=[],
        )
        fig = plot_trade_pnl(result, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotTradeHoldPeriod:

    def test_mpl_returns_figure(self, backtest_result):
        fig = plot_trade_hold_period(backtest_result, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plotly_returns_figure(self, backtest_result):
        pytest.importorskip('plotly')
        import plotly.graph_objects as go
        fig = plot_trade_hold_period(backtest_result, engine='plotly')
        assert isinstance(fig, go.Figure)

    def test_no_trades(self):
        result = BacktestResult(
            total_return=0, annual_return=0, sharpe_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            win_rate=0, profit_loss_ratio=0, total_trades=0,
            equity_curve=pd.DataFrame({'equity': [100000]}, index=pd.date_range('2024-01-01', periods=1)),
            trades=[],
        )
        fig = plot_trade_hold_period(result, engine='matplotlib')
        assert isinstance(fig, matplotlib.figure.Figure)
