# 报告生成模块 (`src/reports/`)

## 模块概述

报告生成模块将回测结果转化为自包含的 HTML 报告，包含绩效指标、权益曲线、回撤曲线、交易记录和盈亏分布图。图表以 SVG 格式内嵌，无需外部依赖即可浏览。

## ReportGenerator

```python
from reports import ReportGenerator
from strategies.engine import BacktestEngine

# 运行回测
result = BacktestEngine().run(strategy, df, symbol="000807.SZ")

# 生成报告
generator = ReportGenerator(title="MA交叉策略回测报告")
html = generator.generate(result, output_path="report.html")  # 保存并返回 HTML
html = generator.to_html(result)                               # 只返回 HTML 字符串
```

## 报告内容

生成的 HTML 报告包含以下五个部分：

### 1. 绩效指标

| 指标 | 说明 |
|------|------|
| 总收益率 | 整个回测期间的总收益 |
| 年化收益率 | 按年化折算的收益率 |
| 夏普比率 | 风险调整后收益 |
| 最大回撤 | 从峰值到谷底的最大跌幅 |
| 最大回撤持续 | 回撤恢复所需天数 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利/平均亏损 |
| 总交易次数 | 完整交易对数 |
| 初始资金 | 回测起始资金 |

### 2. 权益曲线

SVG 折线图，展示权益总值和现金随时间的变化。

### 3. 回撤曲线

SVG 面积图，红色阴影标注回撤幅度，直观展示回撤时段。

### 4. 交易记录

HTML 表格，包含标的、入场/出场时间、入场/出场价格、数量、盈亏和手续费。最多显示前 100 条交易。

### 5. 交易盈亏分布

SVG 柱状图，盈利绿色、亏损红色，直观展示每笔交易的盈亏情况。
