# 技术指标模块 (`src/analysis/`)

## 模块概述

技术指标模块提供 16 个常用金融技术指标的计算实现，以及一个批量计算器 `IndicatorSet`。所有指标函数接受 `pd.Series` 输入，返回 `pd.DataFrame`，与 Pandas 生态无缝集成。

## 模块架构

```
src/analysis/
├── indicator_set.py      # IndicatorSet 批量指标计算器
└── indicators/
    ├── trend.py          # 趋势类: MA, EMA, MACD, DEMA, SAR
    ├── momentum.py       # 动量类: RSI, KDJ, WR, CCI
    ├── volatility.py     # 波动率类: 布林带, ATR, Keltner, Chaikin Vol
    └── volume.py         # 量价类: VWAP, OBV, MFI
```

## 指标总览

| 类别 | 指标 | 函数 | 输出列 | 说明 |
|------|------|------|--------|------|
| 趋势 | 简单移动平均 | `ma(close, period=20)` | `ma_{period}` | 均线 |
| 趋势 | 指数移动平均 | `ema(close, period=20)` | `ema_{period}` | 指数加权均值 |
| 趋势 | MACD | `macd(close, fast=12, slow=26, signal=9)` | `macd_dif`, `macd_dea`, `macd_hist` | DIF/DEA/柱状图 |
| 趋势 | 双指数移动平均 | `dema(close, period=20)` | `dema_{period}` | 消除EMA滞后: 2*EMA - EMA(EMA) |
| 趋势 | 抛物线指标 | `sar(high, low, close, af_step=0.02, af_max=0.20)` | `sar_value`, `sar_trend` | 趋势跟踪止损，迭代算法 |
| 动量 | RSI | `rsi(close, period=14)` | `rsi_{period}` | Wilder 平滑法，单边上涨 RSI=100 |
| 动量 | KDJ | `kdj(high, low, close, n=9, m1=3, m2=3)` | `k`, `d`, `j` | 随机指标，一字板 RSV=50 |
| 动量 | 威廉 %R | `wr(high, low, close, period=14)` | `wr_{period}` | 无平滑超买超卖，范围 [-100, 0] |
| 动量 | CCI | `cci(high, low, close, period=14)` | `cci_{period}` | 无界动量幅度，MAD=0 时 NaN |
| 波动率 | 布林带 | `bollinger(close, period=20, num_std=2.0)` | `boll_mid`, `boll_upper`, `boll_lower` | 中轨/上轨/下轨 |
| 波动率 | ATR | `atr(high, low, close, period=14)` | `atr_{period}` | 真实波幅均值，首根K线 TR=H-L |
| 波动率 | 肯特纳通道 | `keltner(high, low, close, ema_period=20, atr_period=10, num_atr=1.5)` | `kelt_mid`, `kelt_upper`, `kelt_lower` | 与布林带配合识别 squeeze |
| 波动率 | 佳庆波动率 | `chaikin_vol(high, low, period=10, roc_period=10)` | `chaikin_vol_{period}` | 波动率变化速率 |
| 量价 | VWAP | `vwap(high, low, close, volume)` | `vwap` | 成交量加权均价，日内自动重置 |
| 量价 | 能量潮 | `obv(close, volume)` | `obv` | 涨加量、跌减量，反映资金流向 |
| 量价 | 资金流量指标 | `mfi(high, low, close, volume, period=14)` | `mfi_{period}` | 量价结合动量，成交量版 RSI [0,100] |

## IndicatorSet 批量计算器

通过注册表机制管理所有指标，支持链式调用和自动去重。回测引擎通过此接口为策略预计算指标。

```python
from analysis.indicator_set import IndicatorSet

iset = IndicatorSet()
iset.add('ma', period=5)       # 链式调用
   .add('ma', period=20)
   .add('macd')
   .add('rsi', period=14)
   .add('bollinger')

df_result = iset.compute(df)   # 返回原始列 + 所有指标列
iset.clear()                    # 清空规格
iset.specs                      # 查看已注册的指标规格
```

**自动去重**: 相同指标名+参数组合只计算一次。

**指标注册表** (`_INDICATOR_REGISTRY`): 维护指标名→(函数, 需要的列) 映射，支持所有 16 个指标。

## 使用示例

```python
import pandas as pd
from analysis.indicators.trend import ma, macd
from analysis.indicators.momentum import rsi, kdj
from analysis.indicators.volatility import bollinger, atr
from analysis.indicators.volume import obv, mfi

# 单独计算
df_ma = ma(df['close'], period=20)
df_macd = macd(df['close'])
df_rsi = rsi(df['close'], period=14)

# 批量计算
iset = IndicatorSet()
iset.add('ma', period=5).add('ma', period=20).add('rsi', period=14)
df_all = iset.compute(df)
# df_all 包含原始列 + ma_5, ma_20, rsi_14
```
