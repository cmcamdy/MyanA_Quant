# 因子分析模块 (`src/factors/`)

## 模块概述

因子分析模块提供量化选股的核心工具：因子定义、因子评估（IC/IR/分位分析）和因子筛选（复合评分与排名）。支持单因子分析和多因子批量分析，可与其他模块无缝集成。

## 模块架构

```
src/factors/
├── base.py       # Factor 基类 + FactorResult 结果数据类
├── factors.py    # 6 个基础因子实现
├── alpha101.py   # 13 个 Alpha101 因子实现（波动率/价量/突破/动量/反转）
├── analyzer.py   # FactorAnalyzer 因子评估器
└── screener.py   # FactorScreener 因子筛选器
```

## 核心组件

### Factor 基类

所有因子必须实现 `compute` 方法，返回与输入 DataFrame 索引一致的 `pd.Series`。

```python
from factors.base import Factor, FactorResult

class MyFactor(Factor):
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值，返回与 df 索引一致的 Series"""
        ...
```

### FactorResult — 因子分析结果

```python
@dataclass
class FactorResult:
    factor_name: str           # 因子名称
    values: pd.Series          # 因子值序列
    ic: float                  # 信息系数 (Spearman 秩相关)
    ir: float                  # 信息比率 (IC均值/IC标准差)
    quantile_returns: Dict[int, float]  # 各分位平均远期收益
```

## 内置因子

| 因子 | 类名 | 公式 | 默认参数 |
|------|------|------|----------|
| 动量因子 | `MomentumFactor` | `(close - close.shift(p)) / close.shift(p)` | period=20 |
| 波动率因子 | `VolatilityFactor` | `close.pct_change().rolling(p).std()` | period=20 |
| 换手率因子 | `TurnoverFactor` | `vol_rolling_mean / vol_rolling_mean.shift(1)` | period=20 |
| 反转因子 | `ReversalFactor` | `-close.pct_change(p)` | period=5 |
| 价量相关因子 | `PriceVolumeFactor` | `price_pct.corr(volume_pct).rolling(p)` | period=10 |
| 乖离率因子 | `BiasFactor` | `(close - MA) / MA` | period=20 |

### Alpha101 因子（`src/factors/alpha101.py`）

基于 WorldQuant Alpha101 论文选取的高 IC 因子，仅需 OHLCV + amount 数据。
原始公式中的截面排名 rank() 用时间序列排名 ts_rank() 替代，截面标准化可在选股层面统一处理。

#### 波动率类

| 因子 | 类名 | 公式 | 默认参数 | 与基础因子的关系 |
|------|------|------|----------|------------------|
| 波动率异常因子 | `Alpha001Factor` | `ts_rank(std(close - MA(close, n)), m)` | mean_window=5, rank_window=20 | VolatilityFactor 的相对变化版 |
| 偏度反转因子 | `SkewReversalFactor` | `-ts_rank(skew(returns), n)` | window=20, rank_window=20 | 三阶矩（不对称性），VolatilityFactor 是二阶矩 |
| 峰度过滤因子 | `KurtFilterFactor` | `-ts_rank(kurt(returns), n)` | window=20, rank_window=20 | 四阶矩（厚尾），与 VolatilityFactor + SkewReversalFactor 形成「分布三件套」 |

#### 价量类

| 因子 | 类名 | 公式 | 默认参数 | 与基础因子的关系 |
|------|------|------|----------|------------------|
| 量价趋势因子 | `Alpha005Factor` | `ts_rank(close*volume, n) 的 n 日变化` | window=10 | 度量资金流趋势方向，PriceVolumeFactor 只看相关性 |
| 开盘收盘反转因子 | `Alpha014Factor` | `corr(close-open, close-close.shift(1), n)` | window=10 | 区分日内与隔夜反转，ReversalFactor 不区分 |
| 日内效率因子 | `Alpha015Factor` | `sum(close-open, n) / sum(high-low, n)` | window=20 | 衡量趋势质量（净涨幅/总振幅），全新维度 |

#### 突破类

| 因子 | 类名 | 公式 | 默认参数 | 与基础因子的关系 |
|------|------|------|----------|------------------|
| 新高突破因子 | `Alpha023Factor` | `close >= rolling_max(close, n)` → 1/0 | window=20 | 离散事件信号，MomentumFactor 是连续信号 |
| OBV 动量交叉因子 | `Alpha054Factor` | `ts_rank(OBV - MA(OBV, n), m)` | obv_ma_window=20, rank_window=20 | 度量 OBV 偏离强度，obv_direction 只看方向 |

#### 动量类

| 因子 | 类名 | 公式 | 默认参数 | 与基础因子的关系 |
|------|------|------|----------|------------------|
| 上涨胜率因子 | `Alpha084Factor` | `mean(close > prev_close, n)` | window=20 | 度量方向一致性，MomentumFactor 度量涨幅大小 |
| 衰减线性动量因子 | `DecayLinearMomFactor` | `decay_linear(daily_return, n)` | window=10 | 近期权重更大的动量，对趋势加速更敏感 |

#### 反转类

| 因子 | 类名 | 公式 | 默认参数 | 与基础因子的关系 |
|------|------|------|----------|------------------|
| 下影线反转因子 | `Alpha033Factor` | `ts_rank((low-close)/(high-low), n)` | window=20 | 利用 K 线形态，ReversalFactor 只用收盘价 |
| 缩量反转因子 | `Alpha041Factor` | `ts_rank(Δclose², n) * ts_rank(-volume, n)` | window=20 | 缩量急跌后反转信号，引入量能过滤 |
| 短期 Z-Score 反转因子 | `ZscoreReversalFactor` | `-(close-MA)/std` | window=5 | 标准化后的反转，在不同波动率环境下可比 |

```python
from factors import Alpha001Factor, SkewReversalFactor, ZscoreReversalFactor

# 使用默认参数
alpha001 = Alpha001Factor()
values = alpha001.compute(df)  # pd.Series, name='alpha001_5_20'

# 自定义参数
skew = SkewReversalFactor(window=10, rank_window=20)
zscore = ZscoreReversalFactor(window=10)
```

### 辅助函数

`alpha101.py` 导出两个辅助函数，也可在自定义因子中复用：

| 函数 | 说明 |
|------|------|
| `_ts_rank(series, window)` | 时间序列百分位排名：当前值在过去 window 期中的排名位置 (0~1) |
| `_decay_linear(series, window)` | 线性衰减加权平均：越近的数据权重越大 [1,2,...,n]/sum |

## 配置示例

在 `config/portfolio.yaml` 的 `screening.factors` 中可直接使用新因子：

```yaml
screening:
  factors:
    momentum: {period: 20}
    volatility: {period: 20}
    alpha001: {mean_window: 5, rank_window: 20}
    skew_reversal: {window: 20}
    alpha015: {window: 20}
    alpha023: {window: 20}
    decay_linear_mom: {window: 10}
    zscore_reversal: {window: 5}
  weights:
    momentum_20: 0.15
    alpha001_5_20: 0.10
    skew_reversal_20: 0.10
    alpha015_20: 0.10
    alpha023_20: 0.05
    decay_linear_mom_10: 0.15
    zscore_reversal_5: 0.10
    volatility_20: -0.10
    ma_alignment: 0.15
```

```python
from factors import MomentumFactor, VolatilityFactor, BiasFactor

momentum = MomentumFactor(period=20)
values = momentum.compute(df)  # pd.Series, name='momentum_20'
```

## FactorAnalyzer — 因子评估器

提供因子计算、IC/IR 计算、分位分析和完整分析流程。

```python
from factors import FactorAnalyzer

analyzer = FactorAnalyzer(forward_period=1)  # 远期收益周期

# 完整分析单个因子
result = analyzer.analyze(MomentumFactor(period=20), df)
print(f"IC: {result.ic:.4f}, IR: {result.ir:.4f}")
print(f"分位收益: {result.quantile_returns}")

# 单独使用各功能
factor_values = analyzer.compute_factor(MomentumFactor(), df)
ic = analyzer.compute_ic(factor_values, df)
quantile_returns = analyzer.quantile_analysis(factor_values, df, n_quantiles=5)

# 批量分析多个因子
summary_df = analyzer.analyze_batch(
    [MomentumFactor(), VolatilityFactor(), BiasFactor()],
    df,
)
# DataFrame 列: factor_name, ic, ir, q1_return, ..., q5_return
```

**IC (信息系数)**: 使用 Spearman 秩相关系数衡量因子值与远期收益的单调相关性。

**IR (信息比率)**: IC 均值 / IC 标准差，衡量因子预测能力的稳定性。通过滚动窗口计算逐期 IC 序列后求得。

**分位分析**: 将因子值按分位数分组，计算每组的平均远期收益，验证因子的单调性。

### cross_section_ic — 截面 IC

```python
from factors.analyzer import cross_section_ic

# 因子值面板和收益率面板，列为股票代码，索引为日期
ic_series = cross_section_ic(factor_df, return_df)
# 各时间截面的 IC 序列
```

## FactorScreener — 因子筛选器

基于复合因子评分的选股工具，支持加权评分、排名和条件筛选。

```python
from factors import FactorScreener

screener = FactorScreener(
    factors=[MomentumFactor(period=20), VolatilityFactor(period=20)],
    weights={'momentum_20': 0.6, 'volatility_20': 0.4},  # 可选，默认等权
)

# 复合评分 (z-score 标准化后加权求和)
scores = screener.score(df)  # pd.Series, name='composite_score'

# 排名选股
top_stocks = screener.rank(df, top_n=10)  # 返回前 N 名股票代码

# 条件筛选
filtered = screener.filter(df, condition=lambda enriched: enriched['momentum_20'] > 0)
```

**评分逻辑**: 每个因子先做 z-score 标准化，再按权重加权求和，最后除以权重绝对值之和归一化。
