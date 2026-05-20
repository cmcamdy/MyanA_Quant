# 因子分析模块 (`src/factors/`)

## 模块概述

因子分析模块提供量化选股的核心工具：因子定义、因子评估（IC/IR/分位分析）和因子筛选（复合评分与排名）。支持单因子分析和多因子批量分析，可与其他模块无缝集成。

## 模块架构

```
src/factors/
├── base.py      # Factor 基类 + FactorResult 结果数据类
├── factors.py   # 6 个内置因子实现
├── analyzer.py  # FactorAnalyzer 因子评估器
└── screener.py  # FactorScreener 因子筛选器
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
