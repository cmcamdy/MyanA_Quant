# 策略与回测模块 (`src/strategies/`)

## 模块概述

策略模块是系统的核心，提供策略定义框架、回测引擎、风控管理、组合策略、参数优化和多标的组合回测能力。采用事件驱动架构：策略通过 `on_bar` 逐根 K 线产生信号，引擎负责信号执行和组合跟踪。

## 模块架构

```
src/strategies/
├── base.py              # Strategy 基类, Signal, Position, Portfolio, Context
├── engine.py            # BacktestEngine 向量化回测引擎
├── result.py            # BacktestResult, TradeRecord 绩效指标
├── risk.py              # RiskManager 风控（止损/止盈/仓位限制）
├── sizers.py            # PositionSizer 仓位管理
├── composite.py         # CompositeStrategy 组合策略
├── portfolio_engine.py  # PortfolioEngine 多标的组合回测
├── optimizer.py         # Optimizer 参数优化
└── examples/            # 12 个内置策略示例
    ├── ma_cross.py      # 双均线交叉
    ├── macd.py          # MACD 交叉
    ├── sar.py           # SAR 趋势翻转
    ├── rsi.py           # RSI 超买超卖
    ├── kdj.py           # KDJ 随机指标
    ├── wr.py            # 威廉 %R
    ├── cci.py           # CCI 商品通道
    ├── bollinger.py     # 布林带
    ├── keltner.py       # 肯特纳通道
    ├── obv.py           # OBV 能量潮
    ├── mfi.py           # MFI 资金流量
    └── vwap.py          # VWAP 成交量加权价
```

## 核心组件

### Strategy 基类

所有策略必须继承 `Strategy` 并实现 `on_init` + `on_bar`。可选覆盖 `score` 返回连续观点分数，供 Meta Strategy 使用。

```python
from strategies.base import Strategy, Signal, SignalType, Context

class MyStrategy(Strategy):
    def on_init(self, ctx: Context) -> None:
        """注册指标依赖，回测开始前调用"""
        self.register_indicator('ma', period=5)
        self.register_indicator('ma', period=20)

    def on_bar(self, ctx: Context) -> Optional[Signal]:
        """逐 bar 生成交易信号"""
        if ctx.bar['ma_5'] > ctx.bar['ma_20']:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol, strength=0.8)
        return None

    def on_finish(self, ctx: Context) -> None:
        """回测结束后调用（可选）"""
        pass

    def score(self, ctx: Context) -> float:
        """连续观点分数 [-1, 1]，-1=强烈看空, 0=中性, 1=强烈看多"""
        # 默认实现根据 on_bar 信号映射
        # 子类可覆盖以提供更精细的分数
        ...
```

### Signal — 交易信号

```python
Signal(
    type=SignalType.BUY,     # BUY / SELL / HOLD
    symbol="000807.SZ",
    price=None,               # 信号价格（仅参考）
    quantity=None,            # 建议数量（由 sizer 决定）
    strength=1.0,             # 信号强度 [0, 1]，影响仓位比例
    reason="",                # 信号原因描述
)
```

### Context — 逐 bar 上下文

```python
ctx.bar          # pd.Series — 当前 bar 数据（含指标列）
ctx.bars         # pd.DataFrame — 截至当前的所有 bar
ctx.portfolio    # Portfolio — 当前组合状态
ctx.current_time # pd.Timestamp — 当前时间
ctx.symbol       # str — 当前标的代码
```

### BacktestEngine — 回测引擎

向量化回测引擎：先批量预计算所有指标，再逐 bar 驱动策略产生信号。

```python
from strategies.engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.0003,        # 手续费率
    slippage=0.0001,          # 滑点
    benchmark="000300.SH",
)

# 从 YAML 加载配置
config = BacktestConfig.from_yaml("config/settings.yaml")

engine = BacktestEngine(config)
result = engine.run(strategy, df, symbol="000807.SZ")
```

**回测流程**:
1. `strategy.on_init(ctx)` — 注册指标依赖
2. `IndicatorSet.compute(df)` — 向量化预计算所有指标
3. 逐 bar 循环：风控检查 → 更新持仓市值 → 构建上下文 → 生成信号 → 风控过滤 → 执行交易
4. 计算绩效指标

### BacktestResult — 回测结果

```python
result.total_return           # 总收益率
result.annual_return          # 年化收益率
result.sharpe_ratio           # 夏普比率
result.max_drawdown           # 最大回撤
result.max_drawdown_duration  # 最大回撤持续天数
result.win_rate               # 胜率
result.profit_loss_ratio      # 盈亏比
result.total_trades           # 总交易次数
result.equity_curve           # DataFrame: equity/cash/market_value
result.trades                 # List[TradeRecord]
result.per_symbol_equity      # 分标的权益
result.per_symbol_trades      # 分标的交易

print(result.summary())       # 打印绩效摘要
```

## 内置策略示例

共 12 个策略，覆盖趋势、动量、波动率和量价四大类：

| 类别 | 策略 | on_bar 触发条件 | score() 含义 |
|------|------|----------------|-------------|
| 趋势 | `MACrossStrategy(fast=5, slow=20)` | 金叉买入/死叉卖出 | MA 偏离度 (fast-slow)/close |
| 趋势 | `SARStrategy()` | SAR 趋势翻转 | 趋势方向 × 距离权重 |
| 趋势 | `MACDStrategy(fast=12, slow=26, signal=9)` | DIF 上穿/下穿 DEA | (DIF-DEA)/close |
| 动量 | `RSIStrategy(period=14, oversold=30, overbought=70)` | RSI 超卖买入/超买卖出 | (50-RSI)/50 |
| 动量 | `KDJStrategy(n=9, m1=3, m2=3, oversold=20, overbought=80)` | K/D 交叉 + J 值极值 | (50-J)/50 |
| 动量 | `WRStrategy(period=14, oversold=-80, overbought=-20)` | WR 超卖买入/超买卖出 | (-50-WR)/50 |
| 动量 | `CCIStrategy(period=14, oversold=-100, overbought=100)` | CCI 通道外极值 | -CCI/100 |
| 波动 | `BollingerStrategy(period=20)` | 触及下轨买入/上轨卖出 | 1-2×%B |
| 波动 | `KeltnerStrategy(ema=20, atr=10, num_atr=1.5)` | 触及下轨买入/上轨卖出 | 1-2×%K |
| 量价 | `OBVStrategy(ma_period=20)` | OBV 穿越其均线 | OBV 偏离 MA 比率 |
| 量价 | `MFIStrategy(period=14, oversold=20, overbought=80)` | MFI 超卖买入/超买卖出 | (50-MFI)/50 |
| 量价 | `VWAPStrategy()` | 价格低于 VWAP 买入/高于卖出 | -gap/vwap |

## 仓位管理

```python
from strategies.sizers import FixedSizer, AllInSizer

# 固定比例（80% 资金买入）
sizer = FixedSizer(percent=0.8)
engine = BacktestEngine(position_sizer=sizer)

# 全仓
sizer = AllInSizer()
```

`FixedSizer` 受 `signal.strength` 影响：实际比例 = percent × strength

## 风控模块

```python
from strategies.risk import RiskConfig, RiskManager

risk_cfg = RiskConfig(
    # 止损
    stop_loss_pct=0.05,                # 5% 固定止损
    stop_loss_atr_multiplier=2.0,      # ATR 止损: price <= avg_cost - multiplier*atr
    trailing_stop_pct=0.03,            # 3% 追踪止损
    trailing_stop_atr_multiplier=3.0,  # ATR 追踪止损
    # 止盈
    take_profit_pct=0.15,              # 15% 固定止盈
    take_profit_atr_multiplier=4.0,    # ATR 止盈: price >= avg_cost + multiplier*atr
    # 仓位限制
    max_positions=3,                   # 最大持仓数量
    max_per_symbol_weight=0.3,         # 单标的最大权重
    # 回撤限制
    max_portfolio_drawdown=0.10,       # 组合最大回撤 10% 清仓
    # 日亏损限制
    daily_loss_limit=5000.0,           # 日亏损金额上限
    daily_loss_limit_pct=0.02,         # 日亏损比例上限
    atr_period=14,                     # ATR 计算周期
)

engine = BacktestEngine(risk_manager=RiskManager(risk_cfg))
```

**RiskManager 方法**:
- `check_exits(portfolio, bar_data, atr_values)` — 检查止损/止盈，返回卖出信号
- `check_signals(signals, portfolio, bar_data, atr_values, current_time)` — 过滤买入信号
- `register_trailing_stop(symbol, entry_price, atr_at_entry)` — 注册追踪止损
- `remove_trailing_stop(symbol)` — 移除追踪止损

## 组合策略

```python
from strategies.composite import CompositeStrategy

composite = CompositeStrategy(
    strategies=[strategy_a, strategy_b, strategy_c],
    mode='unanimous',  # 'unanimous' / 'any' / 'majority'
)
```

- `unanimous`: 所有子策略信号一致才触发
- `any`: 任一子策略触发即执行
- `majority`: 超过半数子策略一致才触发

## 多标的组合回测

```python
from strategies.portfolio_engine import PortfolioEngine, EqualWeightAllocation, MultiStrategy

# 多标的策略
class MyMultiStrategy(MultiStrategy):
    def on_init_multi(self, ctx):
        self.register_indicator('ma', period=5)

    def on_bar_multi(self, ctx) -> List[Signal]:
        signals = []
        for sym, bar in ctx.bars.items():
            if bar.get('ma_5', 0) > bar.get('ma_20', 0):
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
        return signals

# 运行
data = {"000001.SZ": df1, "600036.SH": df2, "000807.SZ": df3}
engine = PortfolioEngine(allocation=EqualWeightAllocation())
result = engine.run(strategy, data)
# result.per_symbol_equity — 分标的权益曲线
# result.per_symbol_trades — 分标的交易记录
```

**时间轴对齐**: 所有标的索引取并集，缺失 bar 用最近收盘价更新市值但不生成信号。

**分配策略**: `EqualWeightAllocation` (等权 1/N) 和 `CustomAllocation` (自定义权重)。

## 参数优化

```python
from strategies.optimizer import Optimizer, ParamRange

optimizer = Optimizer(
    engine=BacktestEngine(),
    strategy_factory=lambda p: MACrossStrategy(fast_period=p['fast'], slow_period=p['slow']),
    param_ranges=[
        ParamRange(name='fast', values=[3, 5, 7]),
        ParamRange(name='slow', start=15, stop=30, step=5),
    ],
    objective='sharpe',   # 'sharpe' / 'return' / callable
    top_n=5,              # 保留前 N 名的完整 BacktestResult
)

results = optimizer.run(df, symbol="000001.SZ")
# results[0].params — 最优参数组合
# results[0].objective_value — 目标函数值
# results[0].backtest_result — 完整回测结果（仅 top_n 内保留）
```

**自定义目标函数**: `lambda result: -result.max_drawdown`
