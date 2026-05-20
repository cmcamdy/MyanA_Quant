"""交易分布图"""

from typing import Optional

import pandas as pd
import numpy as np


def plot_trade_pnl(result, engine: str = 'matplotlib'):
    """
    盈亏柱状图

    Args:
        result: BacktestResult
        engine: 'matplotlib' 或 'plotly'

    Returns:
        Figure
    """
    if engine == 'plotly':
        return _plot_trade_pnl_plotly(result)
    return _plot_trade_pnl_mpl(result)


def plot_trade_hold_period(result, engine: str = 'matplotlib'):
    """
    持有期散点图

    Args:
        result: BacktestResult
        engine: 'matplotlib' 或 'plotly'

    Returns:
        Figure
    """
    if engine == 'plotly':
        return _plot_trade_hold_plotly(result)
    return _plot_trade_hold_mpl(result)


def _get_closed_trades(result):
    """获取已平仓交易"""
    return [t for t in result.trades if t.exit_time is not None and t.exit_price is not None]


def _plot_trade_pnl_mpl(result):
    import matplotlib.pyplot as plt

    from .candlestick import _setup_chinese_font
    _setup_chinese_font()

    trades = _get_closed_trades(result)
    if not trades:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, '无交易记录', ha='center', va='center', fontsize=14)
        return fig

    pnls = [t.pnl for t in trades]
    colors = ['#388e3c' if p > 0 else '#d32f2f' for p in pnls]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('交易序号')
    ax.set_ylabel('盈亏金额')
    ax.set_title(f'交易盈亏分布 (胜率 {result.win_rate:.0%})')

    plt.tight_layout()
    return fig


def _plot_trade_pnl_plotly(result):
    import plotly.graph_objects as go

    trades = _get_closed_trades(result)
    if not trades:
        fig = go.Figure()
        fig.add_annotation(text='无交易记录', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    pnls = [t.pnl for t in trades]
    colors = ['#388e3c' if p > 0 else '#d32f2f' for p in pnls]

    fig = go.Figure(go.Bar(
        x=list(range(len(pnls))), y=pnls,
        marker_color=colors, opacity=0.8,
    ))
    fig.update_layout(
        title=f'交易盈亏分布 (胜率 {result.win_rate:.0%})',
        xaxis_title='交易序号', yaxis_title='盈亏金额',
        height=500, template='plotly_white',
    )
    return fig


def _plot_trade_hold_mpl(result):
    import matplotlib.pyplot as plt

    from .candlestick import _setup_chinese_font
    _setup_chinese_font()

    trades = _get_closed_trades(result)
    if not trades:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, '无交易记录', ha='center', va='center', fontsize=14)
        return fig

    hold_days = [(t.exit_time - t.entry_time).days for t in trades]
    returns = [(t.exit_price / t.entry_price - 1) * 100 for t in trades]
    colors = ['#388e3c' if r > 0 else '#d32f2f' for r in returns]

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(hold_days, returns, c=colors, alpha=0.6, s=60, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('持有天数')
    ax.set_ylabel('收益率 (%)')
    ax.set_title('持有期 vs 收益率')

    plt.tight_layout()
    return fig


def _plot_trade_hold_plotly(result):
    import plotly.graph_objects as go

    trades = _get_closed_trades(result)
    if not trades:
        fig = go.Figure()
        fig.add_annotation(text='无交易记录', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    hold_days = [(t.exit_time - t.entry_time).days for t in trades]
    returns = [(t.exit_price / t.entry_price - 1) * 100 for t in trades]
    colors = ['#388e3c' if r > 0 else '#d32f2f' for r in returns]

    fig = go.Figure(go.Scatter(
        x=hold_days, y=returns, mode='markers',
        marker=dict(color=colors, size=10, opacity=0.7),
        text=[f'盈亏: {t.pnl:.0f}' for t in trades],
    ))
    fig.update_layout(
        title='持有期 vs 收益率',
        xaxis_title='持有天数', yaxis_title='收益率 (%)',
        height=500, template='plotly_white',
    )
    return fig
