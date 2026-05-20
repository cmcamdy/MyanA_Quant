"""K线图 + 指标叠加 + 买卖点标注"""

from typing import List, Optional

import pandas as pd
import numpy as np


def _setup_chinese_font():
    """配置matplotlib中文字体"""
    import matplotlib
    import platform
    system = platform.system()
    if system == 'Darwin':
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    elif system == 'Windows':
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False


def plot_kline(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    trades: Optional[list] = None,
    title: Optional[str] = None,
    engine: str = 'matplotlib',
):
    """
    绘制K线图

    Args:
        df: OHLCV DataFrame (DatetimeIndex, 包含指标列)
        indicators: 要叠加的指标列名列表
        trades: TradeRecord 列表，标注买卖点
        title: 图表标题
        engine: 'matplotlib' 或 'plotly'

    Returns:
        matplotlib Figure 或 plotly Figure
    """
    if engine == 'plotly':
        return _plot_kline_plotly(df, indicators, trades, title)
    return _plot_kline_mpl(df, indicators, trades, title)


def _plot_kline_mpl(df, indicators, trades, title):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch

    _setup_chinese_font()

    # 判断成交量子图是否需要
    has_volume = 'volume' in df.columns
    fig, axes = plt.subplots(
        2 if has_volume else 1, 1,
        figsize=(14, 8),
        gridspec_kw={'height_ratios': [3, 1]} if has_volume else None,
        sharex=True,
    )
    if not has_volume:
        axes = [axes]

    ax = axes[0]
    dates = df.index

    # 绘制蜡烛图
    width = 0.6
    for i, (idx, row) in enumerate(df.iterrows()):
        color = '#d32f2f' if row['close'] >= row['open'] else '#388e3c'
        # 影线
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)
        # 实体
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        rect = plt.Rectangle((i - width / 2, body_bottom), width, body_height,
                              facecolor=color if row['close'] >= row['open'] else color,
                              edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    # 叠加指标线
    if indicators:
        for col in indicators:
            if col in df.columns:
                style = '--' if col.startswith('boll_') else '-'
                alpha = 0.5 if col.startswith('boll_') else 0.9
                ax.plot(range(len(df)), df[col].values, label=col, linestyle=style, alpha=alpha, linewidth=1)

        # 布林带填充
        upper_cols = [c for c in indicators if c.startswith('boll_upper')]
        lower_cols = [c for c in indicators if c.startswith('boll_lower')]
        for uc, lc in zip(upper_cols, lower_cols):
            ax.fill_between(range(len(df)), df[uc].values, df[lc].values, alpha=0.1, color='blue')

    # 标注买卖点
    if trades:
        from strategies.result import TradeRecord
        date_to_idx = {d: i for i, d in enumerate(dates)}
        for t in trades:
            if t.entry_time in date_to_idx:
                idx = date_to_idx[t.entry_time]
                ax.annotate('B', xy=(idx, df['low'].iloc[idx]),
                            xytext=(0, -15), textcoords='offset points',
                            color='green', fontsize=9, fontweight='bold', ha='center')
            if t.exit_time and t.exit_time in date_to_idx:
                idx = date_to_idx[t.exit_time]
                ax.annotate('S', xy=(idx, df['high'].iloc[idx]),
                            xytext=(0, 10), textcoords='offset points',
                            color='red', fontsize=9, fontweight='bold', ha='center')

    ax.set_ylabel('价格')
    if title:
        ax.set_title(title)
    if indicators:
        ax.legend(loc='upper left', fontsize=8)

    # 成交量子图
    if has_volume:
        ax_vol = axes[1]
        colors = ['#d32f2f' if c >= o else '#388e3c' for c, o in zip(df['close'], df['open'])]
        ax_vol.bar(range(len(df)), df['volume'].values, color=colors, width=0.6, alpha=0.7)
        ax_vol.set_ylabel('成交量')
        ax_vol.set_yscale('log')

    # X轴日期
    n_ticks = min(12, len(df))
    tick_positions = np.linspace(0, len(df) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in tick_positions], rotation=30, fontsize=8)

    plt.tight_layout()
    return fig


def _plot_kline_plotly(df, indicators, trades, title):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_volume = 'volume' in df.columns
    rows = 2 if has_volume else 1
    row_heights = [0.75, 0.25] if has_volume else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights,
                        vertical_spacing=0.03)

    # 蜡烛图
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线',
    ), row=1, col=1)

    # 指标叠加
    if indicators:
        for col in indicators:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], mode='lines',
                    name=col, line=dict(dash='dash' if col.startswith('boll_') else 'solid'),
                ), row=1, col=1)

        # 布林带填充
        upper_cols = [c for c in indicators if c.startswith('boll_upper')]
        lower_cols = [c for c in indicators if c.startswith('boll_lower')]
        for uc, lc in zip(upper_cols, lower_cols):
            fig.add_trace(go.Scatter(
                x=pd.concat([df.index, df.index[::-1]]),
                y=pd.concat([df[uc], df[lc][::-1]]),
                fill='toself', fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(0,0,0,0)'), name='布林带',
            ), row=1, col=1)

    # 买卖点
    if trades:
        buy_times, buy_prices = [], []
        sell_times, sell_prices = [], []
        date_to_price = dict(zip(df.index, df['low']))
        date_to_price_high = dict(zip(df.index, df['high']))
        for t in trades:
            if t.entry_time in date_to_price:
                buy_times.append(t.entry_time)
                buy_prices.append(date_to_price[t.entry_time] * 0.98)
            if t.exit_time and t.exit_time in date_to_price_high:
                sell_times.append(t.exit_time)
                sell_prices.append(date_to_price_high[t.exit_time] * 1.02)

        if buy_times:
            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode='markers+text',
                marker=dict(symbol='triangle-up', color='green', size=12),
                text=['B'] * len(buy_times), textposition='bottom center',
                name='买入',
            ), row=1, col=1)
        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode='markers+text',
                marker=dict(symbol='triangle-down', color='red', size=12),
                text=['S'] * len(sell_times), textposition='top center',
                name='卖出',
            ), row=1, col=1)

    # 成交量
    if has_volume:
        colors = ['#d32f2f' if c >= o else '#388e3c' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], marker_color=colors, name='成交量', opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        title=title or '', xaxis_rangeslider_visible=False,
        height=700, template='plotly_white',
    )
    return fig
