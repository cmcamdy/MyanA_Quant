# 可视化模块 (`src/visualization/`)

## 模块概述

可视化模块提供回测结果的图形化展示，支持 Matplotlib（静态图，适合保存）和 Plotly（交互图，适合研究）两种渲染引擎。所有函数自动处理缺失数据和空交易场景。

## 模块架构

```
src/visualization/
├── __init__.py       # 统一导出
├── candlestick.py    # K线图 + 指标叠加 + 买卖点标注
├── equity.py         # 权益曲线 + 回撤图
└── trade.py          # 交易盈亏柱状图 + 持有期散点图
```

## 函数总览

| 函数 | 说明 | 引擎支持 |
|------|------|----------|
| `plot_kline` | K线图，支持指标叠加和买卖点标注 | matplotlib, plotly |
| `plot_equity` | 权益曲线 + 回撤子图，可选基准对比 | matplotlib, plotly |
| `plot_drawdown` | 独立回撤曲线，高亮最大回撤区间 | matplotlib, plotly |
| `plot_trade_pnl` | 交易盈亏柱状图（盈利绿/亏损红） | matplotlib, plotly |
| `plot_trade_hold_period` | 持有期 vs 收益率散点图 | matplotlib, plotly |

## 使用示例

### K线图

```python
from visualization import plot_kline

fig = plot_kline(
    df,                                    # OHLCV DataFrame (DatetimeIndex)
    indicators=['ma_5', 'ma_20'],          # 叠加的指标列名
    trades=result.trades,                  # TradeRecord 列表，标注买卖点
    title='000807.SZ K线图',
    engine='matplotlib',                   # 'matplotlib' 或 'plotly'
)
```

自动检测 `volume` 列决定是否画成交量子图。布林带指标会自动填充上下轨区域。缺失的 indicator 列会被静默忽略。

### 权益曲线

```python
from visualization import plot_equity

fig = plot_equity(
    result,                                # BacktestResult
    benchmark_df=bm_df,                    # 可选，含 'close' 列的 DataFrame
    engine='matplotlib',
)
```

上半区: 权益曲线 + 可选基准对比。下半区: 回撤百分比。

### 回撤图

```python
from visualization import plot_drawdown

fig = plot_drawdown(result, engine='matplotlib')
```

回撤曲线 + 高亮最大回撤区间（红色阴影）。

### 交易盈亏

```python
from visualization import plot_trade_pnl

fig = plot_trade_pnl(result, engine='matplotlib')
```

盈亏柱状图，盈利绿色，亏损红色。无交易时显示"无交易记录"。

### 持有期散点图

```python
from visualization import plot_trade_hold_period

fig = plot_trade_hold_period(result, engine='matplotlib')
```

散点图: x=持有天数, y=收益率(%)，盈利绿色，亏损红色。

## 保存与展示

```python
# Matplotlib — 保存为图片
fig = plot_equity(result, engine='matplotlib')
fig.savefig('equity.png', dpi=150, bbox_inches='tight')

# Plotly — 交互式展示
fig = plot_equity(result, engine='plotly')
fig.show()

# Plotly — 保存为 HTML
fig.write_html('equity.html')
```
