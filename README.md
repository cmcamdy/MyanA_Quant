# 量化交易分析系统 (MyanAQuant)

用于策略研究、回测和分析的量化交易分析系统，不包含实盘交易功能。

## 项目结构

```
MyanAQuant/
├── src/                          # 源代码
│   ├── data/                     # 数据获取模块
│   │   ├── base.py               # KlineData, DataProvider 基类
│   │   ├── storage.py            # ParquetStorage 本地存储
│   │   ├── industry.py           # IndustryLookup 行业分类查询
│   │   ├── manager.py            # DataManager 统一数据管理
│   │   └── providers/            # 数据源提供商
│   │       ├── akshare.py        # AkShare 数据源（日线/周线/月线）
│   │       └── baostock.py       # Baostock 数据源（分钟线/日线/周线/月线）
│   ├── analysis/                 # 技术指标模块
│   │   ├── indicator_set.py      # IndicatorSet 批量指标计算器
│   │   └── indicators/           # 指标实现
│   │       ├── trend.py          # MA, EMA, MACD, DEMA, SAR
│   │       ├── momentum.py       # RSI, KDJ, WR, CCI
│   │       ├── volatility.py     # 布林带, ATR, Keltner, Chaikin Vol
│   │       └── volume.py         # VWAP, OBV, MFI
│   ├── strategies/               # 策略回测模块
│   │   ├── base.py               # Strategy ABC, Signal, Position, Portfolio
│   │   ├── engine.py             # BacktestEngine 向量化回测引擎
│   │   ├── portfolio_engine.py   # PortfolioEngine 多标的组合回测
│   │   ├── risk.py               # RiskManager 风控（止损/止盈/仓位限制）
│   │   ├── optimizer.py          # Optimizer 参数优化（网格搜索）
│   │   ├── result.py             # BacktestResult, TradeRecord, 绩效指标
│   │   ├── sizers.py             # PositionSizer, FixedSizer, AllInSizer
│   │   ├── composite.py          # CompositeStrategy 组合策略
│   │   └── examples/             # 策略示例
│   │       └── ma_cross.py       # 双均线交叉策略
│   ├── dl/                        # 深度学习预测模块
│   │   ├── config.py              # DLConfig 配置
│   │   ├── feature_builder.py     # FeatureBuilder 特征工程
│   │   ├── dataset.py             # StockDataset PyTorch数据集
│   │   ├── model.py               # PricePredictor 网络
│   │   └── trainer.py             # Trainer 训练/评估/预测
│   └── visualization/            # 可视化模块
│       ├── candlestick.py        # K线图 + 指标叠加 + 买卖点标注
│       ├── equity.py             # 权益曲线 + 回撤图
│       └── trade.py              # 交易盈亏柱状图 + 持有期散点图
├── scripts/                      # 脚本
│   ├── download_daily.py         # A股日线批量下载脚本
│   ├── download_industry_stock.py # 行业-股票映射下载脚本
│   ├── demo_dl_predict.py         # 深度学习涨跌预测 Demo
│   └── watchdog_download.py      # 下载守护脚本（卡住自动重启）
├── tests/                        # 测试 (276 用例)
│   └── unit/                     # 单元测试
│       ├── data/                 # 数据模块测试
│       ├── analysis/             # 指标模块测试
│       ├── dl/                   # 深度学习模块测试
│       ├── strategies/           # 策略模块测试
│       └── visualization/        # 可视化模块测试
├── config/
│   └── settings.yaml             # 系统配置
├── docs/                         # 文档
├── requirements.txt
└── setup.py
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> Baostock 需要额外安装: `pip install baostock`

### 获取数据

```python
from data.manager import DataManager

mgr = DataManager(data_dir="./data")

# 获取日线数据（优先本地缓存，缺失时自动从数据源下载）
kline = mgr.get_kline("000807.SZ", start="2024-01-01", end="2024-12-31", freq="1d")
df = kline.df  # DataFrame with OHLCV columns, DatetimeIndex
```

### 批量下载数据

```bash
# 全量下载所有A股日线
python scripts/download_daily.py

# 增量更新（只下载本地缺失的日期段）
python scripts/download_daily.py --incremental

# 只下载指定股票
python scripts/download_daily.py --symbols 000807.SZ,600036.SH

# 定时任务：每个交易日 18:00 增量更新
# crontab: 0 18 * * 1-5 cd /path/to/MyanAQuant && python scripts/download_daily.py --incremental
```

### 下载行业分类数据

```bash
# 下载行业-股票映射（证监会行业分类）
python scripts/download_industry_stock.py

# 输出到 data/industry-stock/industry_stock.csv 和 industry_summary.txt
```

### 查询行业分类

```python
from data import IndustryLookup

lk = IndustryLookup("./data")

# 按行业查股票
stocks = lk.get_stocks("N78公共设施管理业")  # → ["000430.SZ", "000558.SZ", ...]

# 按行业查详情（含名称）
details = lk.get_stock_details("N78公共设施管理业")  # → DataFrame(code, name)

# 反查：股票→行业
industry = lk.get_industry("600874.SH")  # → "D46水的生产和供应业"

# 列出所有行业
industries = lk.list_industries()  # → 83个行业列表
```

### 计算技术指标

```python
from analysis.indicator_set import IndicatorSet

iset = IndicatorSet()
iset.add('ma', period=5).add('ma', period=20).add('macd').add('rsi', period=14)
df_with_indicators = iset.compute(df)
# 新增列: ma_5, ma_20, macd_dif, macd_dea, macd_hist, rsi_14
```

可用指标：`ma`, `ema`, `macd`, `dema`, `sar`, `rsi`, `kdj`, `wr`, `cci`, `bollinger`, `atr`, `keltner`, `chaikin_vol`, `vwap`, `obv`, `mfi`

### 编写策略并回测

```python
from strategies.base import Strategy, Signal, SignalType
from strategies.engine import BacktestEngine, BacktestConfig

class MACrossStrategy(Strategy):
    def on_init(self, ctx):
        self.register_indicator('ma', period=5)
        self.register_indicator('ma', period=20)

    def on_bar(self, ctx):
        ma5 = ctx.bar.get('ma_5')
        ma20 = ctx.bar.get('ma_20')
        if pd.isna(ma5) or pd.isna(ma20):
            return None
        prev_ma5 = ctx.bars['ma_5'].iloc[-2]
        prev_ma20 = ctx.bars['ma_20'].iloc[-2]
        if prev_ma5 <= prev_ma20 and ma5 > ma20:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol)
        if prev_ma5 >= prev_ma20 and ma5 < ma20:
            return Signal(type=SignalType.SELL, symbol=ctx.symbol)
        return None

# 运行回测
engine = BacktestEngine(BacktestConfig(initial_capital=100000))
result = engine.run(MACrossStrategy(), df, symbol="000807.SZ")
print(result.summary())
```

### 可视化

```python
from visualization import plot_kline, plot_equity, plot_drawdown, plot_trade_pnl

# K线图 + 指标叠加 + 买卖点标注
fig = plot_kline(df_with_indicators, indicators=['ma_5', 'ma_20'], trades=result.trades, engine='matplotlib')
fig.savefig('kline.png')

# 权益曲线 + 回撤
fig = plot_equity(result, engine='plotly')
fig.show()

# 交易盈亏分布
fig = plot_trade_pnl(result, engine='matplotlib')
fig.savefig('trades.png')
```

支持 `engine='matplotlib'`（静态图，适合保存）和 `engine='plotly'`（交互图，适合研究）。

### 深度学习涨跌预测

```python
from dl import DLConfig, Trainer

# 配置
config = DLConfig(
    data_dir="./data",
    window=120,          # 回看120天
    horizon=1,           # 预测次日涨跌
    hidden_dim=256,
    epochs=50,
)

# 训练
trainer = Trainer(config)
trainer.train(symbols=["600036.SH", "600519.SH", ...])

# 回归测试
result = trainer.evaluate()
print(f"准确率: {result['accuracy']:.4f}")
print(result['classification_report'])

# 预测
pred = trainer.predict("600036.SH")
print(pred['label'])  # 如 "小涨 (0%~2%)"
```

6 分类输出: 大跌(<-5%) / 中跌(-5%~-2%) / 小跌(-2%~0%) / 小涨(0%~2%) / 中涨(2%~5%) / 大涨(>5%)

## 模块说明

### 数据获取模块 (`src/data/`)

| 类 | 说明 |
|---|---|
| `KlineData` | K线数据容器，含 symbol/freq/start/end/df，提供 ohlcv 属性 |
| `DataProvider` | 数据源抽象基类，定义 get_kline/get_stock_list/normalize_symbol 接口 |
| `AkShareProvider` | AkShare 数据源，支持日线/周线/月线，无需注册 |
| `BaostockProvider` | Baostock 数据源，支持 5min/15min/30min/60min/日线/周线/月线 |
| `ParquetStorage` | Parquet 格式本地存储，层级目录 `{market}/{code}/{freq}.parquet`，自动合并去重 |
| `IndustryLookup` | 行业分类查询，基于证监会行业分类，支持行业→股票、股票→行业双向查询 |
| `DataManager` | 统一入口，cache-aside 模式，按频率自动选择数据源，支持增量更新 |

### 深度学习预测模块 (`src/dl/`)

| 类 | 说明 |
|---|---|
| `DLConfig` | 配置数据类，含窗口/预测周期/模型参数/训练参数 |
| `FeatureBuilder` | 特征工程，从 parquet 构建 OHLCV+指标特征和涨跌标签，z-score 标准化 |
| `StockDataset` | PyTorch Dataset 包装 |
| `PricePredictor` | 全连接分类网络 (Linear+LayerNorm+ReLU+Dropout+Linear)，6 分类涨跌预测 |
| `Trainer` | 训练器，支持训练/断点续训/增量训练/评估/单只预测 |

### 技术指标模块 (`src/analysis/`)

| 指标 | 函数 | 输出列名 | 说明 |
|------|------|----------|------|
| 移动平均线 | `ma(close, period)` | `ma_{period}` | 简单移动平均 |
| 指数移动平均 | `ema(close, period)` | `ema_{period}` | 指数移动平均 |
| MACD | `macd(close, fast, slow, signal)` | `macd_dif`, `macd_dea`, `macd_hist` | DIF/DEA/柱状图 |
| 双指数移动平均 | `dema(close, period)` | `dema_{period}` | 消除EMA滞后 |
| 抛物线指标 | `sar(high, low, close, af_step, af_max)` | `sar_value`, `sar_trend` | 趋势跟踪止损 |
| RSI | `rsi(close, period)` | `rsi_{period}` | Wilder 平滑，单边上涨时 RSI=100 |
| KDJ | `kdj(high, low, close, n, m1, m2)` | `k`, `d`, `j` | 随机指标 |
| 威廉指标 | `wr(high, low, close, period)` | `wr_{period}` | 无平滑超买超卖，范围 [-100, 0] |
| CCI | `cci(high, low, close, period)` | `cci_{period}` | 无界动量幅度 |
| 布林带 | `bollinger(close, period, num_std)` | `boll_mid`, `boll_upper`, `boll_lower` | 中轨/上轨/下轨 |
| ATR | `atr(high, low, close, period)` | `atr_{period}` | 真实波幅均值 |
| 肯特纳通道 | `keltner(high, low, close, ema_period, atr_period, num_atr)` | `kelt_mid`, `kelt_upper`, `kelt_lower` | 与布林带配合识别 squeeze |
| 佳庆波动率 | `chaikin_vol(high, low, period, roc_period)` | `chaikin_vol_{period}` | 波动率变化速率 |
| VWAP | `vwap(high, low, close, volume)` | `vwap` | 成交量加权均价，日内自动重置 |
| 能量潮 | `obv(close, volume)` | `obv` | 量价方向，反映资金流向 |
| 资金流量指标 | `mfi(high, low, close, volume, period)` | `mfi_{period}` | 量价结合动量，成交量版RSI |

`IndicatorSet` 支持链式调用批量计算，自动去重：
```python
iset = IndicatorSet()
iset.add('ma', period=5).add('ma', period=20).add('rsi', period=14)
df_result = iset.compute(df)
```

### 策略回测模块 (`src/strategies/`)

| 类 | 说明 |
|---|---|
| `Strategy` | 策略抽象基类，子类实现 `on_init`（注册指标）和 `on_bar`（生成信号） |
| `Signal` / `SignalType` | 交易信号，含 type/symbol/price/quantity/strength/reason |
| `Position` / `Portfolio` | 持仓和组合状态跟踪 |
| `Context` | 逐 bar 上下文，含当前 bar、历史 bars、组合状态 |
| `BacktestEngine` | 向量化回测引擎：指标预计算 + 逐 bar 信号 + 组合跟踪 |
| `BacktestConfig` | 回测配置（初始资金/手续费/滑点/基准），支持从 YAML 加载 |
| `BacktestResult` | 回测结果，含绩效指标/权益曲线/交易记录/分标的明细，提供 `summary()` |
| `PositionSizer` | 仓位管理协议，内置 `FixedSizer`/`AllInSizer` |
| `CompositeStrategy` | 组合策略，支持 unanimous/any/majority 三种信号合并模式 |
| `RiskManager` | 风控管理器：固定/ATR止损、追踪止损、固定/ATR止盈、仓位限制、回撤限制、日亏损限制 |
| `PortfolioEngine` | 多标的组合回测引擎，支持普通 Strategy 和 MultiStrategy |
| `Optimizer` | 参数优化器：网格搜索、自定义目标函数、结果排名 |

**回测流程：**
1. `strategy.on_init(ctx)` — 注册指标依赖
2. `IndicatorSet.compute(df)` — 向量化预计算所有指标
3. 逐 bar 循环：风控检查 → 更新持仓市值 → 构建上下文 → 生成信号 → 风控过滤 → 执行交易
4. 计算绩效指标：总收益率/年化/夏普/最大回撤/胜率/盈亏比

### 风控模块 (`src/strategies/risk.py`)

```python
from strategies.risk import RiskConfig, RiskManager
from strategies.engine import BacktestEngine

risk_cfg = RiskConfig(
    stop_loss_pct=0.05,           # 5% 固定止损
    trailing_stop_pct=0.03,       # 3% 追踪止损
    take_profit_pct=0.15,         # 15% 固定止盈
    max_positions=3,              # 最多持有3只
    max_portfolio_drawdown=0.10,  # 组合最大回撤10%清仓
)
engine = BacktestEngine(risk_manager=RiskManager(risk_cfg))
result = engine.run(strategy, df, symbol="000001.SZ")
```

### 多标的组合回测 (`src/strategies/portfolio_engine.py`)

```python
from strategies.portfolio_engine import PortfolioEngine, EqualWeightAllocation

data = {
    "000001.SZ": df1,
    "600036.SH": df2,
    "000807.SZ": df3,
}
engine = PortfolioEngine()
result = engine.run(strategy, data)
# result.per_symbol_equity — 分标的权益
# result.per_symbol_trades — 分标的交易记录
```

### 参数优化 (`src/strategies/optimizer.py`)

```python
from strategies.optimizer import Optimizer, ParamRange

optimizer = Optimizer(
    engine=BacktestEngine(),
    strategy_factory=lambda p: MACrossStrategy(fast_period=p['fast'], slow_period=p['slow']),
    param_ranges=[
        ParamRange(name='fast', start=3, stop=10, step=1),
        ParamRange(name='slow', start=15, stop=30, step=5),
    ],
    objective='sharpe',
    top_n=5,
)
results = optimizer.run(df, symbol="000001.SZ")
# results[0] — 最优参数组合
```

### 可视化模块 (`src/visualization/`)

| 函数 | 说明 |
|------|------|
| `plot_kline(df, indicators, trades, title, engine)` | K线图，支持指标叠加和买卖点标注 |
| `plot_equity(result, benchmark_df, engine)` | 权益曲线 + 回撤子图，可选基准对比 |
| `plot_drawdown(result, engine)` | 独立回撤曲线，高亮最大回撤区间 |
| `plot_trade_pnl(result, engine)` | 交易盈亏柱状图（盈利绿/亏损红） |
| `plot_trade_hold_period(result, engine)` | 持有期 vs 收益率散点图 |

## 配置说明

配置文件 `config/settings.yaml`：

```yaml
data_sources:
  akshare:
    enabled: true

storage:
  type: "local"
  path: "./data"

backtest:
  initial_capital: 100000
  commission: 0.0003    # 手续费率
  slippage: 0.0001      # 滑点
  benchmark: "000300.SH"

logging:
  level: "INFO"
  file: "./logs/trading.log"
```

## 技术栈

- **语言**: Python 3.9+
- **数据源**: Baostock（主力）, AkShare（日线备用）
- **数据分析**: Pandas, NumPy
- **可视化**: Matplotlib, Plotly
- **存储**: Parquet
- **回测**: 自研回测引擎（向量化指标 + 逐 bar 信号）

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个模块测试
pytest tests/unit/strategies/ -v

# 跳过需要 plotly 的测试
pytest tests/ -v -k "not plotly"
```
