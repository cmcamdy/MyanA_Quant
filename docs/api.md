# API 参考文档

## 数据模块 (`src/data/`)

### KlineData

```python
from data.base import KlineData

kline = KlineData(
    symbol="000807.SZ",
    freq="1d",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    df=df,  # DataFrame with DatetimeIndex, columns: open/high/low/close/volume
)

kline.open    # Series
kline.high    # Series
kline.low     # Series
kline.close   # Series
kline.volume  # Series
kline.amount  # Series or None
```

### DataProvider

```python
from data.base import DataProvider

class DataProvider(ABC):
    def get_kline(self, symbol, start, end, freq='1d', adjust='qfq') -> KlineData
    def get_stock_list(self) -> pd.DataFrame
    def normalize_symbol(self, symbol: str) -> str
    def supports_freq(self, freq: str) -> bool
    @property
    def supported_frequencies(self) -> List[str]
    @property
    def name(self) -> str
```

**AkShareProvider**: 支持 `1d`, `1w`, `1m`。无需注册。

**BaostockProvider**: 支持 `5min`, `15min`, `30min`, `60min`, `1d`, `1w`, `1m`。每次调用内部 login/logout。

### ParquetStorage

```python
from data.storage import ParquetStorage

storage = ParquetStorage(data_dir="./data")

storage.save(kline)                               # 保存，自动合并去重
df = storage.load(symbol, freq, start, end)       # 加载，可选日期范围过滤
symbols = storage.list_symbols(market="SH")        # 列出已存储标的
date_range = storage.get_date_range(symbol, freq)  # 获取数据日期范围 -> (start, end) or None
```

存储路径格式: `{data_dir}/{market}/{code}/{freq}.parquet`，如 `data/SZ/000807/1d.parquet`

### DataManager

```python
from data.manager import DataManager

mgr = DataManager(data_dir="./data")

# 获取数据（优先缓存，缺失时自动从数据源下载）
kline = mgr.get_kline("000807.SZ", start="2024-01-01", end="2024-12-31", freq="1d")

# 指定数据源
kline = mgr.get_kline("000807.SZ", start="2024-01-01", end="2024-12-31", source="baostock")

# 增量更新
new_data = mgr.update_kline("000807.SZ", freq="1d")

# 列出已缓存标的
symbols = mgr.list_cached_symbols(market="SH")

# 获取股票列表
stock_list = mgr.get_stock_list(source="akshare")
```

数据源自动选择规则:

| 频率 | 优先数据源 | 备选 |
|------|-----------|------|
| 1d | AkShare | Baostock |
| 1w | AkShare | Baostock |
| 1m | AkShare | Baostock |
| 5min/15min/30min/60min | Baostock | — |

### IndustryLookup

```python
from data import IndustryLookup

lk = IndustryLookup(data_dir="./data")

# 按行业查股票代码列表
stocks = lk.get_stocks("N78公共设施管理业")
# → ["000430.SZ", "000558.SZ", ...]

# 按行业查股票详情（代码+名称）
details = lk.get_stock_details("N78公共设施管理业")
# → DataFrame with columns: [code, name]

# 反查：股票代码→行业
industry = lk.get_industry("600874.SH")
# → "D46水的生产和供应业"

# 列出所有行业名称
industries = lk.list_industries()
# → ["A01农业", "A02林业", ..., 83个行业]
```

数据源: Baostock 证监会行业分类，通过 `scripts/download_industry_stock.py` 下载，存储于 `data/industry-stock/industry_stock.csv`。

---

## 深度学习预测模块 (`src/dl/`)

### DLConfig

```python
from dl.config import DLConfig

config = DLConfig(
    data_dir="./data",        # parquet 根目录
    window=120,               # 回看窗口天数
    horizon=1,                # 预测未来几天 (1=次日)
    train_ratio=0.7,          # 训练集比例
    val_ratio=0.15,           # 验证集比例
    hidden_dim=256,           # 隐藏层维度
    num_classes=6,            # 分类数
    dropout=0.3,              # Dropout
    batch_size=64,
    learning_rate=1e-3,
    epochs=50,
    early_stopping_patience=5,
    checkpoint_dir="./checkpoints",
    resume=False,             # 断点续训
    incremental=False,        # 增量训练
)
```

6 分类标签 (`CLASS_LABELS`): 大跌(<-5%), 中跌(-5%~-2%), 小跌(-2%~0%), 小涨(0%~2%), 中涨(2%~5%), 大涨(>5%)

### FeatureBuilder

```python
from dl.feature_builder import FeatureBuilder

builder = FeatureBuilder(config)

# 构建单只股票样本
features, labels = builder.build_stock("600036.SH")
# features: [N, 31, window], labels: [N]

# 构建多只股票样本
features, labels, symbol_list = builder.build_all(["600036.SH", "600519.SH"])

# 标准化参数保存/加载
builder.save_scaler("checkpoints/scaler.npz")
builder.load_scaler("checkpoints/scaler.npz")
```

特征列: 6 原始列 (OHLCV+A) + 25 技术指标列 = 31 列，由 `IndicatorSet` 自动计算。

### PricePredictor

```python
from dl.model import PricePredictor

model = PricePredictor(
    input_dim=3720,    # num_features × window
    hidden_dim=256,
    num_classes=6,
    dropout=0.3,
)

# 前向传播
logits = model(x)  # x: [B, 31, 120] → logits: [B, 6]
probs = torch.softmax(logits, dim=1)
```

### Trainer

```python
from dl.trainer import Trainer

trainer = Trainer(config)

# 完整训练
trainer.train(symbols=["600036.SH", "600519.SH", ...])

# 断点续训
trainer.resume_train(symbols=[...])

# 增量训练 (降低学习率 fine-tune)
trainer.incremental_train(symbols=[...])

# 回归测试
result = trainer.evaluate(symbols=[...])
result['accuracy']              # float
result['classification_report'] # str
result['confusion_matrix']      # np.ndarray [6, 6]

# 单只股票预测
pred = trainer.predict("600036.SH")
pred['probabilities']      # [6] 各分类概率
pred['predicted_class']    # int (0-5)
pred['label']              # str 如 "小涨 (0%~2%)"
```

---

## 技术指标模块 (`src/analysis/`)

### 单个指标函数

```python
from analysis.indicators.trend import ma, ema, macd, dema, sar
from analysis.indicators.momentum import rsi, kdj, wr, cci
from analysis.indicators.volatility import bollinger, atr, keltner, chaikin_vol
from analysis.indicators.volume import vwap, obv, mfi
```

#### `ma(close, period=20) → DataFrame`
- 输出列: `ma_{period}`

#### `ema(close, period=20) → DataFrame`
- 输出列: `ema_{period}`

#### `macd(close, fast=12, slow=26, signal=9) → DataFrame`
- 输出列: `macd_dif`, `macd_dea`, `macd_hist`

#### `dema(close, period=20) → DataFrame`
- 输出列: `dema_{period}`
- DEMA = 2*EMA - EMA(EMA)，消除EMA滞后

#### `sar(high, low, close, af_step=0.02, af_max=0.20) → DataFrame`
- 输出列: `sar_value`（止损位）, `sar_trend`（1=上升, -1=下降）
- 迭代算法，追踪趋势方向并提供动态止损位

#### `rsi(close, period=14) → DataFrame`
- 输出列: `rsi_{period}`
- 使用 Wilder 平滑 (ewm alpha=1/period)
- 单调上涨时 RSI=100，无变化时 RSI=50

#### `kdj(high, low, close, n=9, m1=3, m2=3) → DataFrame`
- 输出列: `k`, `d`, `j`
- 一字板 (high==low) 时 RSV=50

#### `wr(high, low, close, period=14) → DataFrame`
- 输出列: `wr_{period}`
- 范围 [-100, 0]，低于 -80 超卖，高于 -20 超买
- HH=LL 时默认 WR=-50

#### `cci(high, low, close, period=14) → DataFrame`
- 输出列: `cci_{period}`
- 无界指标，衡量价格偏离统计均值的幅度
- MAD=0 时输出 NaN

#### `bollinger(close, period=20, num_std=2.0) → DataFrame`
- 输出列: `boll_mid`, `boll_upper`, `boll_lower`

#### `atr(high, low, close, period=14) → DataFrame`
- 输出列: `atr_{period}`
- 首根K线 TR = high - low

#### `keltner(high, low, close, ema_period=20, atr_period=10, num_atr=1.5) → DataFrame`
- 输出列: `kelt_mid`, `kelt_upper`, `kelt_lower`
- Mid=EMA(close), Upper/Lower = Mid ± num_atr*ATR
- 与布林带配合可识别 squeeze 形态

#### `chaikin_vol(high, low, period=10, roc_period=10) → DataFrame`
- 输出列: `chaikin_vol_{period}`
- SMA(H-L) 的变化率，衡量波动率扩张/收缩

#### `vwap(high, low, close, volume) → DataFrame`
- 输出列: `vwap`
- 日内数据按自然日重置累计值

#### `obv(close, volume) → DataFrame`
- 输出列: `obv`
- 涨加量、跌减量、平不变，反映资金流向

#### `mfi(high, low, close, volume, period=14) → DataFrame`
- 输出列: `mfi_{period}`
- 范围 [0, 100]，成交量版 RSI
- neg_mf=0 时 MFI=100

### IndicatorSet 批量计算

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
```

可用指标名: `ma`, `ema`, `macd`, `dema`, `sar`, `rsi`, `kdj`, `wr`, `cci`, `bollinger`, `atr`, `keltner`, `chaikin_vol`, `vwap`, `obv`, `mfi`

---

## 策略回测模块 (`src/strategies/`)

### Strategy 基类

```python
from strategies.base import Strategy, Signal, SignalType, Context

class MyStrategy(Strategy):
    def on_init(self, ctx: Context) -> None:
        """注册指标依赖，在回测开始前调用"""
        self.register_indicator('ma', period=5)
        self.register_indicator('ma', period=20)

    def on_bar(self, ctx: Context) -> Optional[Signal]:
        """逐 bar 生成交易信号"""
        if ctx.bar['ma_5'] > ctx.bar['ma_20']:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol, strength=0.8)
        return None

    def on_finish(self, ctx: Context) -> None:
        """回测结束后调用（可选重写）"""
        pass
```

### Signal

```python
Signal(
    type=SignalType.BUY,     # BUY / SELL / HOLD
    symbol="000807.SZ",
    price=None,               # 信号价格（仅参考，不用于执行）
    quantity=None,            # 建议数量（可选，由 sizer 决定）
    strength=1.0,             # 信号强度 [0, 1]，影响仓位比例
    reason="",                # 信号原因描述
)
```

### Context

```python
ctx.bar          # pd.Series — 当前 bar 数据
ctx.bars         # pd.DataFrame — 截至当前的所有 bar（含指标列）
ctx.portfolio    # Portfolio — 当前组合状态
ctx.current_time # pd.Timestamp — 当前时间
ctx.symbol       # str — 当前标的代码
```

### Portfolio / Position

```python
portfolio = Portfolio(cash=100000, equity=100000)

pos = portfolio.get_position("000807.SZ")  # 自动创建空仓位
pos.is_empty     # True（quantity == 0）
pos.pnl          # 未实现盈亏
pos.quantity      # 持仓数量
pos.avg_cost      # 平均成本
pos.market_value  # 市值
```

### BacktestEngine

```python
from strategies.engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.0003,        # 手续费率（按名义金额）
    slippage=0.0001,          # 滑点（价格偏移比例）
    benchmark="000300.SH",
)

# 从 YAML 加载配置
config = BacktestConfig.from_yaml("config/settings.yaml")

engine = BacktestEngine(config)
result = engine.run(strategy, df, symbol="000807.SZ")
```

### BacktestResult

```python
result.total_return           # 总收益率
result.annual_return          # 年化收益率
result.sharpe_ratio           # 夏普比率
result.max_drawdown           # 最大回撤
result.max_drawdown_duration  # 最大回撤持续天数
result.win_rate               # 胜率
result.profit_loss_ratio      # 盈亏比
result.total_trades           # 总交易次数
result.equity_curve           # DataFrame: equity/cash/market_value, DatetimeIndex
result.trades                 # List[TradeRecord]
result.initial_capital        # 初始资金
result.benchmark              # 基准代码
result.per_symbol_equity      # Optional[Dict[str, pd.Series]] — 分标的权益
result.per_symbol_trades      # Optional[Dict[str, List[TradeRecord]]] — 分标的交易

print(result.summary())       # 打印绩效摘要
```

### TradeRecord

```python
TradeRecord(
    symbol="000807.SZ",
    entry_time=pd.Timestamp("2024-01-05"),
    exit_time=pd.Timestamp("2024-01-15"),
    entry_price=10.50,
    exit_price=11.20,
    quantity=1000,
    pnl=680.0,
    commission=63.0,
)
```

### 仓位管理

```python
from strategies.sizers import FixedSizer, AllInSizer

# 固定比例（80% 资金买入）
sizer = FixedSizer(percent=0.8)
engine = BacktestEngine(position_sizer=sizer)

# 全仓
sizer = AllInSizer()
```

`FixedSizer` 受 `signal.strength` 影响：实际比例 = percent * strength

### 组合策略

```python
from strategies.composite import CompositeStrategy

composite = CompositeStrategy(
    strategies=[strategy_a, strategy_b, strategy_c],
    mode='unanimous',  # 'unanimous' / 'any' / 'majority'
)

result = engine.run(composite, df, symbol="000807.SZ")
```

- `unanimous`: 所有子策略信号一致才触发
- `any`: 任一子策略触发即执行（买卖信号不能同时出现）
- `majority`: 超过半数子策略一致才触发

### 风控模块

```python
from strategies.risk import RiskConfig, RiskManager
from strategies.engine import BacktestEngine

risk_cfg = RiskConfig(
    # 止损
    stop_loss_pct=0.05,                # 固定百分比止损（5%）
    stop_loss_atr_multiplier=2.0,      # ATR止损：price <= avg_cost - multiplier * atr
    trailing_stop_pct=0.03,            # 追踪止损（3%）
    trailing_stop_atr_multiplier=3.0,  # ATR追踪止损
    # 止盈
    take_profit_pct=0.15,              # 固定百分比止盈（15%）
    take_profit_atr_multiplier=4.0,    # ATR止盈：price >= avg_cost + multiplier * atr
    # 仓位限制
    max_positions=3,                   # 最大持仓数量
    max_per_symbol_weight=0.3,         # 单标的最大权重
    # 回撤限制
    max_portfolio_drawdown=0.10,       # 组合最大回撤（超出清仓）
    # 日亏损限制
    daily_loss_limit=5000.0,           # 日亏损金额上限
    daily_loss_limit_pct=0.02,         # 日亏损比例上限
    # ATR
    atr_period=14,                     # ATR计算周期
)

engine = BacktestEngine(risk_manager=RiskManager(risk_cfg))
result = engine.run(strategy, df, symbol="000001.SZ")
```

**RiskConfig 属性**:
- `needs_atr` → `bool`：是否需要 ATR 指标（配置了任何 ATR 相关止损/止盈时为 True）

**RiskManager 方法**:
- `check_exits(portfolio, bar_data, atr_values) → List[Signal]` — 检查止损/止盈，返回卖出信号列表
- `check_signals(signals, portfolio, bar_data, atr_values, current_time) → List[Signal]` — 过滤买入信号
- `register_trailing_stop(symbol, entry_price, atr_at_entry)` — 注册追踪止损
- `remove_trailing_stop(symbol)` — 移除追踪止损
- `reset()` — 重置内部状态

### 组合回测引擎

#### MultiStrategy

```python
from strategies.portfolio_engine import MultiStrategy, PortfolioContext

class MyMultiStrategy(MultiStrategy):
    def on_init_multi(self, ctx: PortfolioContext) -> None:
        """注册指标（同 Strategy.register_indicator）"""
        self.register_indicator('ma', period=5)

    def on_bar_multi(self, ctx: PortfolioContext) -> List[Signal]:
        """返回多标的信号列表"""
        signals = []
        for sym, bar in ctx.bars.items():
            if bar.get('ma_5', 0) > bar.get('ma_20', 0):
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
        return signals

    def on_finish_multi(self, ctx: PortfolioContext) -> None:
        """回测结束回调（可选）"""
        pass
```

**PortfolioContext**:
```python
ctx.bars           # Dict[str, pd.Series] — {symbol: 当前bar}
ctx.historical     # Dict[str, pd.DataFrame] — {symbol: 截至当前的所有bar}
ctx.portfolio      # Portfolio — 当前组合状态
ctx.current_time   # pd.Timestamp — 当前时间
ctx.bar            # Optional[pd.Series] — 便捷属性，第一个标的的bar
ctx.symbol         # Optional[str] — 便捷属性，第一个标的代码
```

#### AllocationPolicy

```python
from strategies.portfolio_engine import EqualWeightAllocation, CustomAllocation

# 等权分配：1/N
alloc = EqualWeightAllocation()

# 自定义权重
alloc = CustomAllocation({"000001.SZ": 0.5, "600036.SH": 0.3, "000807.SZ": 0.2})
```

#### PortfolioEngine

```python
from strategies.portfolio_engine import PortfolioEngine

engine = PortfolioEngine(
    config=BacktestConfig(commission=0.0003, slippage=0.0001),
    allocation=EqualWeightAllocation(),
    risk_manager=RiskManager(risk_cfg),
)

data = {
    "000001.SZ": df1,
    "600036.SH": df2,
    "000807.SZ": df3,
}

# 支持普通 Strategy（逐标的调用 on_bar）和 MultiStrategy（一次调用 on_bar_multi）
result = engine.run(strategy, data)

result.per_symbol_equity   # Dict[str, pd.Series] — 分标的权益曲线
result.per_symbol_trades   # Dict[str, List[TradeRecord]] — 分标的交易记录
```

**时间轴对齐**: 所有标的索引取并集，缺失 bar 用最近收盘价更新市值但不生成信号。

### 参数优化器

```python
from strategies.optimizer import Optimizer, ParamRange, OptimizationResult
from strategies.optimizer import sharpe_objective, total_return_objective, results_to_dataframe

optimizer = Optimizer(
    engine=BacktestEngine(BacktestConfig(commission=0, slippage=0)),
    strategy_factory=lambda p: MACrossStrategy(
        fast_period=p['fast'],
        slow_period=p['slow'],
    ),
    param_ranges=[
        ParamRange(name='fast', values=[3, 5, 7]),
        ParamRange(name='slow', start=15, stop=30, step=5),
    ],
    objective='sharpe',   # 'sharpe' / 'return' / callable
    top_n=5,              # 保留前N名的完整 BacktestResult
)

results = optimizer.run(df, symbol="000001.SZ")
```

**ParamRange**:
```python
# 离散值
pr = ParamRange(name='period', values=[5, 10, 20])

# 连续范围
pr = ParamRange(name='threshold', start=0.01, stop=0.05, step=0.01)

pr.iter_values()  # → List[Any]
```

**OptimizationResult**:
```python
result.params            # Dict[str, Any] — 参数组合
result.objective_value   # float — 目标函数值
result.backtest_result   # Optional[BacktestResult] — 仅 top_n 内保留
```

**目标函数**:
- `'sharpe'` → `result.sharpe_ratio`
- `'return'` → `result.total_return`
- 自定义 callable: `lambda result: -result.max_drawdown`

**辅助函数**:
- `sharpe_objective(result)` — 夏普比率目标
- `total_return_objective(result)` — 总收益率目标
- `results_to_dataframe(results)` — 转为 DataFrame，含参数列 + objective 列

---

## 可视化模块 (`src/visualization/`)

所有函数支持 `engine='matplotlib'`（返回 `matplotlib.figure.Figure`）和 `engine='plotly'`（返回 `plotly.graph_objects.Figure`）。

### plot_kline

```python
from visualization import plot_kline

fig = plot_kline(
    df,                                    # OHLCV DataFrame (DatetimeIndex)
    indicators=['ma_5', 'ma_20'],          # 叠加的指标列名
    trades=result.trades,                  # TradeRecord 列表，标注买卖点
    title='000807.SZ K线图',               # 标题
    engine='matplotlib',                   # 'matplotlib' 或 'plotly'
)
```

自动检测 `volume` 列决定是否画成交量子图。布林带指标会自动填充上下轨区域。缺失的 indicator 列会被静默忽略。

### plot_equity

```python
from visualization import plot_equity

fig = plot_equity(
    result,                                # BacktestResult
    benchmark_df=bm_df,                    # 可选，含 'close' 列的 DataFrame
    engine='matplotlib',
)
```

上半区: 权益曲线 + 可选基准对比。下半区: 回撤百分比。

### plot_drawdown

```python
from visualization import plot_drawdown

fig = plot_drawdown(result, engine='matplotlib')
```

回撤曲线 + 高亮最大回撤区间（红色阴影）。

### plot_trade_pnl

```python
from visualization import plot_trade_pnl

fig = plot_trade_pnl(result, engine='matplotlib')
```

盈亏柱状图，盈利绿色，亏损红色。无交易时显示"无交易记录"。

### plot_trade_hold_period

```python
from visualization import plot_trade_hold_period

fig = plot_trade_hold_period(result, engine='matplotlib')
```

散点图: x=持有天数, y=收益率(%)，盈利绿色，亏损红色。无交易时显示"无交易记录"。

---

## 数据下载脚本 (`scripts/download_daily.py`)

```bash
# 全量下载（默认从 2000-01-01 开始）
python scripts/download_daily.py

# 增量更新（只下载本地缺失的日期段）
python scripts/download_daily.py --incremental

# 指定股票和日期
python scripts/download_daily.py --symbols 000807.SZ,600036.SH --start 2020-01-01

# 自定义数据目录
python scripts/download_daily.py --data-dir /path/to/data

# 复权方式: qfq(前复权,默认) / hfq(后复权) / 不复权
python scripts/download_daily.py --adjust hfq

# 重试和延迟
python scripts/download_daily.py --retry 5 --delay 0.5
```

参数说明:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--incremental` | False | 增量更新模式 |
| `--symbols` | None | 指定股票代码，逗号分隔 |
| `--data-dir` | ./data | 数据存储目录 |
| `--start` | 2000-01-01 | 全量下载起始日期 |
| `--end` | 今天 | 结束日期 |
| `--adjust` | qfq | 复权方式: qfq/hfq/空 |
| `--retry` | 3 | 单只股票失败重试次数 |
| `--delay` | 0.3 | 下载间隔秒数 |

---

## 实时监控模块 (`src/monitor/`)

### MonitorConfig

```python
from monitor import MonitorConfig

config = MonitorConfig(
    symbols=["600036.SH", "000001.SZ"],  # 监控标的列表
    freq="1d",                            # 数据频率
    poll_interval=60,                     # 轮询间隔(秒)
    initial_capital=100000.0,
    commission=0.0003,
    slippage=0.0001,
    state_dir="./data/monitor",           # 状态文件目录
    data_dir="./data",                    # DataManager数据目录
    data_source=None,                     # 数据源(None=自动)
    adjust="qfq",
    initial_positions=None,               # Dict[str, {avg_cost, quantity}]
    dashboard_refresh=5,                  # 仪表盘刷新间隔(秒)
    alert_on_signal=True,
    alert_on_risk_trigger=True,
    alert_on_equity_change_pct=0.05,      # 权益变动超过5%告警
)
```

### LiveEngine

```python
from monitor import LiveEngine
from strategies.examples.ma_cross import MACrossStrategy
from strategies.risk import RiskConfig

strategy = MACrossStrategy(fast_period=5, slow_period=20)
risk_config = RiskConfig(stop_loss_pct=0.05, trailing_stop_pct=0.03)

engine = LiveEngine(
    config=config,
    strategy=strategy,
    risk_config=risk_config,
)

# 启动监控（默认恢复上次状态）
engine.start(resume=True)

# 优雅停止
engine.stop()
```

`start()` 运行主循环：轮询数据 → 驱动策略 → 执行信号 → 更新仪表盘 → 保存状态。Ctrl+C 触发优雅退出。

支持普通 `Strategy`（逐标的调用 `on_bar`）和 `MultiStrategy`（一次调用 `on_bar_multi`）。

### MonitorState

```python
from monitor import MonitorState

# 从 Portfolio 创建
state = MonitorState.from_portfolio(portfolio, monitor_id="test")

# 保存/加载（原子写入，防崩溃）
state.save("./data/monitor/test.json")
loaded = MonitorState.load("./data/monitor/test.json")

# 重建 Portfolio
portfolio = state.to_portfolio()

# RiskManager 状态持久化
state.extract_risk_state(risk_manager)  # 从 RiskManager 提取
state.apply_risk_state(risk_manager)    # 恢复到 RiskManager
```

### AlertManager

```python
from monitor import AlertManager, AlertEvent, AlertCallback, LoggingCallback

# 自定义告警回调
class DingTalkCallback:
    def __call__(self, event: AlertEvent) -> None:
        # 发送钉钉/微信/邮件通知
        ...

alerts = AlertManager(
    on_signal=True,
    on_risk_trigger=True,
    equity_change_threshold=0.05,
    callbacks=[LoggingCallback(), DingTalkCallback()],
    log_file="./data/monitor/alerts.log",
)
```

**AlertEvent** 属性: `timestamp`, `event_type`("signal"/"risk_trigger"/"equity_change"/"trade"), `symbol`, `details`, `severity`("info"/"warning"/"critical")

### ConsoleDashboard

终端仪表盘，自动在 `LiveEngine.start()` 中显示：

```
============================================
  Strategy Monitor - MACrossStrategy
  Symbols: 600036.SH, 000001.SZ | Freq: 1d
  Poll: every 60s | Uptime: 2h 15m
============================================

  PORTFOLIO
  Equity: 105,230.45  Cash: 52,115.23  Return: +5.23%

  POSITIONS
  Symbol       Qty     AvgCost   Price    PnL       PnL%
  600036.SH    800     38.50     40.12    +1,296    +4.21%

  RECENT SIGNALS (last 5)
  14:30  600036.SH  BUY   (MA金叉)

  RISK ALERTS (last 3)
  (none)

  EQUITY
  ▂▃▃▄▅▅▆▇▆▇██▇▇█

  Polls: 42 | Last: 2026-05-20 14:30:00
============================================
```

### 监控脚本 (`scripts/monitor_strategy.py`)

```bash
# 基本用法
python scripts/monitor_strategy.py \
    --symbols 600036.SH,000001.SZ \
    --freq 1d --interval 60

# 带风控
python scripts/monitor_strategy.py \
    --symbols 600036.SH \
    --stop-loss 0.05 --trailing-stop 0.03 --take-profit 0.15

# 不恢复上次状态（重新开始）
python scripts/monitor_strategy.py \
    --symbols 600036.SH --no-resume

# 指定数据源
python scripts/monitor_strategy.py \
    --symbols 600036.SH --source baostock
```

参数说明:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | 必填 | 逗号分隔的标的代码 |
| `--freq` | 1d | 数据频率 |
| `--interval` | 60 | 轮询间隔(秒) |
| `--capital` | 100000 | 初始资金 |
| `--fast` | 5 | MA快线周期 |
| `--slow` | 20 | MA慢线周期 |
| `--stop-loss` | 0 | 止损比例 |
| `--trailing-stop` | 0 | 追踪止损比例 |
| `--take-profit` | 0 | 止盈比例 |
| `--max-positions` | 0 | 最大持仓数(0=不限) |
| `--source` | auto | 数据源 |
| `--no-resume` | False | 不恢复上次状态 |
