"""权益曲线 + 回撤图"""

from typing import Optional

import pandas as pd
import numpy as np


def plot_equity(
    result,
    benchmark_df: Optional[pd.DataFrame] = None,
    engine: str = 'matplotlib',
):
    """
    绘制权益曲线和回撤图

    Args:
        result: BacktestResult
        benchmark_df: 可选基准收盘价 DataFrame (含 'close' 列, DatetimeIndex)
        engine: 'matplotlib' 或 'plotly'

    Returns:
        Figure
    """
    if engine == 'plotly':
        return _plot_equity_plotly(result, benchmark_df)
    return _plot_equity_mpl(result, benchmark_df)


def plot_drawdown(result, engine: str = 'matplotlib'):
    """
    绘制回撤曲线，高亮最大回撤区间

    Args:
        result: BacktestResult
        engine: 'matplotlib' 或 'plotly'

    Returns:
        Figure
    """
    if engine == 'plotly':
        return _plot_drawdown_plotly(result)
    return _plot_drawdown_mpl(result)


def _calc_drawdown(equity: pd.Series):
    """计算回撤序列和最大回撤区间"""
    cummax = equity.cummax()
    drawdown = (cummax - equity) / cummax
    # 找最大回撤区间
    end_idx = drawdown.idxmax()
    peak_idx = equity[:end_idx].idxmax()
    return drawdown, peak_idx, end_idx


def _plot_equity_mpl(result, benchmark_df):
    import matplotlib.pyplot as plt

    from .candlestick import _setup_chinese_font
    _setup_chinese_font()

    equity = result.equity_curve['equity']
    drawdown, _, _ = _calc_drawdown(equity)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # 权益曲线
    ax1.plot(equity.index, equity.values, label='策略权益', color='#1565c0', linewidth=1.2)

    # 基准对比
    if benchmark_df is not None and 'close' in benchmark_df.columns:
        bm = benchmark_df['close']
        bm_normalized = bm / bm.iloc[0] * result.initial_capital
        # 对齐日期
        common = equity.index.intersection(bm_normalized.index)
        if len(common) > 0:
            ax1.plot(common, bm_normalized[common], label='基准', color='gray', linewidth=1, alpha=0.7)

    ax1.set_ylabel('权益')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(f'总收益 {result.total_return:.2%} | 夏普 {result.sharpe_ratio:.2f} | 最大回撤 {result.max_drawdown:.2%}')

    # 回撤曲线
    ax2.fill_between(drawdown.index, 0, -drawdown.values * 100, color='#d32f2f', alpha=0.4)
    ax2.plot(drawdown.index, -drawdown.values * 100, color='#d32f2f', linewidth=0.8)
    ax2.set_ylabel('回撤 (%)')
    ax2.set_xlabel('日期')

    plt.tight_layout()
    return fig


def _plot_equity_plotly(result, benchmark_df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    equity = result.equity_curve['equity']
    drawdown, _, _ = _calc_drawdown(equity)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name='策略权益',
        line=dict(color='#1565c0', width=1.2),
    ), row=1, col=1)

    if benchmark_df is not None and 'close' in benchmark_df.columns:
        bm = benchmark_df['close']
        bm_normalized = bm / bm.iloc[0] * result.initial_capital
        common = equity.index.intersection(bm_normalized.index)
        if len(common) > 0:
            fig.add_trace(go.Scatter(
                x=common, y=bm_normalized[common], name='基准',
                line=dict(color='gray', width=1),
            ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=-drawdown.values * 100, name='回撤',
        fill='tozeroy', fillcolor='rgba(211,47,47,0.3)',
        line=dict(color='#d32f2f', width=0.8),
    ), row=2, col=1)

    fig.update_layout(
        title=f'总收益 {result.total_return:.2%} | 夏普 {result.sharpe_ratio:.2f} | 最大回撤 {result.max_drawdown:.2%}',
        height=700, template='plotly_white',
        yaxis2=dict(ticksuffix='%'),
    )
    return fig


def _plot_drawdown_mpl(result):
    import matplotlib.pyplot as plt

    from .candlestick import _setup_chinese_font
    _setup_chinese_font()

    equity = result.equity_curve['equity']
    drawdown, peak_idx, end_idx = _calc_drawdown(equity)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, 0, -drawdown.values * 100, color='#d32f2f', alpha=0.3)
    ax.plot(drawdown.index, -drawdown.values * 100, color='#d32f2f', linewidth=0.8)

    # 高亮最大回撤区间
    ax.axvspan(peak_idx, end_idx, alpha=0.2, color='red', label='最大回撤区间')
    ax.set_ylabel('回撤 (%)')
    ax.set_xlabel('日期')
    ax.legend()
    ax.set_title(f'最大回撤 {result.max_drawdown:.2%}')

    plt.tight_layout()
    return fig


def _plot_drawdown_plotly(result):
    import plotly.graph_objects as go

    equity = result.equity_curve['equity']
    drawdown, peak_idx, end_idx = _calc_drawdown(equity)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=-drawdown.values * 100, name='回撤',
        fill='tozeroy', fillcolor='rgba(211,47,47,0.3)',
        line=dict(color='#d32f2f', width=0.8),
    ))

    fig.add_vrect(x0=peak_idx, x1=end_idx, fillcolor='red', opacity=0.15,
                  annotation_text='最大回撤', annotation_position='top left')

    fig.update_layout(
        title=f'最大回撤 {result.max_drawdown:.2%}',
        height=400, template='plotly_white',
        yaxis=dict(ticksuffix='%'),
    )
    return fig
