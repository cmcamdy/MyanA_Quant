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
│   │   ├── base.py               # Strategy ABC (含 score 方法), Signal, Position, Portfolio
│   │   ├── engine.py             # BacktestEngine 向量化回测引擎
│   │   ├── portfolio_engine.py   # PortfolioEngine 多标的组合回测
│   │   ├── risk.py               # RiskManager 风控（止损/止盈/仓位限制）
│   │   ├── optimizer.py          # Optimizer 参数优化（网格搜索/遗传/贝叶斯）
│   │   ├── result.py             # BacktestResult, TradeRecord, 绩效指标
│   │   ├── sizers.py             # PositionSizer, FixedSizer, AllInSizer
│   │   ├── composite.py          # CompositeStrategy 组合策略
│   │   ├── examples/             # 策略示例 (12个)
│   │       ├── ma_cross.py       # 双均线交叉策略
│   │       ├── macd.py           # MACD 交叉策略
│   │       ├── sar.py            # SAR 抛物线策略
│   │       ├── rsi.py            # RSI 超买超卖策略
│   │       ├── kdj.py            # KDJ 随机指标策略
│   │       ├── wr.py             # 威廉 %R 策略
│   │       ├── cci.py            # CCI 商品通道策略
│   │       ├── bollinger.py      # 布林带策略
│   │       ├── keltner.py        # 肯特纳通道策略
│   │       ├── obv.py            # OBV 能量潮策略
│   │       ├── mfi.py            # MFI 资金流量策略
│   │       └── vwap.py           # VWAP 成交量加权价策略
│   ├── dl/                        # 深度学习模块
│   │   ├── config.py              # DLConfig 配置（回归模式）
│   │   ├── feature_builder.py     # FeatureBuilder 特征工程
│   │   ├── dataset.py             # StockDataset PyTorch数据集
│   │   ├── model.py               # PricePredictor Transformer 回归网络
│   │   ├── trainer.py             # Trainer 训练/评估/预测
│   │   ├── dl_strategy.py         # DLStrategy 桥接回测引擎
│   │   ├── meta_config.py         # MetaConfig 学习型组合策略配置
│   │   ├── meta_model.py          # MetaModel 策略注意力加权网络
│   │   ├── meta_signal_collector.py # MetaSignalCollector 策略信号采集
│   │   ├── meta_trainer.py        # MetaTrainer 训练/评估
│   │   └── meta_strategy.py       # MetaStrategy 桥接回测引擎
│   ├── rl/                        # 强化学习模块
│   │   ├── config.py              # RLConfig 配置
│   │   ├── data_adapter.py        # DataAdapter per-stock→panel数据适配
│   │   ├── factor_panel.py        # FactorPanelBuilder 296因子计算
│   │   ├── inference.py           # RLInference ONNX推理
│   │   ├── rl_strategy.py         # RLStrategy 截面选股策略(MultiStrategy)
│   │   ├── env_factory.py         # EnvFactory Gym环境创建
│   │   └── trainer.py             # RLTrainer PPO训练编排
│   ├── stock_selection/            # 五维选股模块
│   │   ├── tencent_fetcher.py     # 腾讯财经API数据获取（异步+同步）
│   │   ├── jys_scorer.py          # 五维评分引擎（技术面+估值+盈利+安全+分红）
│   │   ├── jys_screener.py        # 选股筛选器（对接portfolio pipeline）
│   │   └── dividend_override.py   # 股息率人工修正数据
│   └── visualization/            # 可视化模块
│       ├── candlestick.py        # K线图 + 指标叠加 + 买卖点标注
│       ├── equity.py             # 权益曲线 + 回撤图
│       └── trade.py              # 交易盈亏柱状图 + 持有期散点图
├── web/                          # Web应用
│   └── app.py                    # Streamlit五维选股可视化系统
├── scripts/                      # 脚本
│   ├── download_daily.py         # A股日线批量下载脚本
│   ├── download_industry_stock.py # 行业-股票映射下载脚本
│   ├── demo_portfolio.py         # Portfolio组合回测 Demo（含因子选股/策略搜索）
│   ├── demo_jys_screen.py        # JYS五维选股 Demo
│   ├── demo_dl_predict.py         # 深度学习收益率预测 Demo
│   ├── demo_meta_strategy.py      # Meta Strategy 学习型组合策略 Demo
│   ├── demo_rl_portfolio.py       # RL 截面选股组合回测 Demo
│   └── watchdog_download.py      # 下载守护脚本（卡住自动重启）
├── tests/                        # 测试 (566 用例)
│   └── unit/                     # 单元测试
│       ├── data/                 # 数据模块测试
│       ├── analysis/             # 指标模块测试
│       ├── dl/                   # 深度学习模块测试
│       ├── rl/                   # 强化学习模块测试
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

### 深度学习收益率预测

```bash
# 训练 + 预测
python3 scripts/demo_dl_predict.py --start-date 2013-01-01

# 训练 + 回测
python3 scripts/demo_dl_predict.py --backtest --start-date 2013-01-01
```

```python
from dl import DLConfig, Trainer, DLStrategy
from strategies.engine import BacktestEngine

# 配置（回归模式：预测未来收益率）
config = DLConfig(
    data_dir="./data",
    start_date="2013-01-01",    # 只用2013年以后的数据
    window=60,
    horizon=1,
    epochs=300,
    loss_type="huber",
    buy_threshold=0.005,        # 预测收益率 > 0.5% 买入
    sell_threshold=-0.005,      # 预测收益率 < -0.5% 卖出
)

# 训练
trainer = Trainer(config)
trainer.train(symbols=["600036.SH", "600519.SH", ...])

# 评估
result = trainer.evaluate()
print(f"MSE: {result['mse']:.6f}")
print(f"方向准确率: {result['direction_accuracy']:.4f}")

# 预测
pred = trainer.predict("600036.SH")
print(f"预测收益率: {pred['predicted_return']:+.2%}")
print(f"方向: {pred['direction']}")

# 接入回测
strategy = DLStrategy(config)
result = BacktestEngine().run(strategy, df, symbol="600036.SH")
```

### Meta Strategy: 学习型组合策略

用注意力机制学习多个子策略的权重，动态决定看多/看空倾向。

```bash
# 训练 + 查看策略权重
python3 scripts/demo_meta_strategy.py --start-date 2013-01-01

# 训练 + 回测
python3 scripts/demo_meta_strategy.py --backtest --start-date 2013-01-01
```

```python
from dl import MetaConfig, MetaTrainer, MetaStrategy
from strategies.examples.ma_cross import MACrossStrategy
from strategies.examples.sar import SARStrategy
from strategies.examples.rsi import RSIStrategy
from strategies.examples.bollinger import BollingerStrategy

# 配置
config = MetaConfig(
    strategies=[
        MACrossStrategy(fast_period=5, slow_period=20),
        SARStrategy(),
        RSIStrategy(period=14),
        BollingerStrategy(period=20),
    ],
    start_date="2013-01-01",
    window=20,
    epochs=100,
)

# 训练
trainer = MetaTrainer(config)
trainer.train(symbols)

# 评估 — 查看各策略权重
result = trainer.evaluate()
for i, strat in enumerate(config.strategies):
    w = result['avg_strategy_weights'][i]
    print(f"{type(strat).__name__}: {w:.4f}")

# 接入回测
meta = MetaStrategy(config)
result = BacktestEngine().run(meta, df, symbol="000807.SZ")
```

**工作原理**:

每个子策略输出连续观点分数 `score()` (范围 -1 到 +1)，MetaModel 通过注意力机制学习各策略权重，输出预测收益率。

| 策略 | score() 含义 |
|------|-------------|
| MACross | `(fast_ma - slow_ma) / close` 偏离度 |
| SAR | 趋势方向 × 距离权重 |
| RSI | `(50 - RSI) / 50` |
| Bollinger | `1 - 2×%B` (下轨=+1, 上轨=-1) |

### 强化学习截面选股

```bash
# 安装 RL 依赖
pip install onnxruntime>=1.17
pip install git+https://github.com/yupoet/aurumq-rl.git   # 训练需要

# 构建面板 (数据准备, 只需运行一次)
python3 scripts/demo_rl_portfolio.py --build-panel

# 训练 RL 模型
python3 scripts/demo_rl_portfolio.py --train --universe all --timesteps 100000

# 使用预训练模型回测
python3 scripts/demo_rl_portfolio.py --backtest --top-k 10 --rebalance-freq M
```

```python
from rl import RLConfig, RLStrategy
from strategies.portfolio_engine import PortfolioEngine, EqualWeightAllocation, RebalanceConfig

config = RLConfig(
    onnx_model_path="./checkpoints/rl/policy.onnx",
    inference_top_k=10,
    rebalance_freq="M",
)

strategy = RLStrategy(config)
engine = PortfolioEngine(
    allocation=EqualWeightAllocation(),
    rebalance=RebalanceConfig(frequency="M"),
)
result = engine.run(strategy, data)
print(f"Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
```

### 五维评分选股

基于 JYSstock_analyzer 的沪深300五维评分选股（技术面30% + 估值25% + 盈利质量30% + 安全性10% + 分红5%），使用腾讯财经API获取实时基本面数据。

```bash
# 安装异步依赖（推荐，300只3秒完成 vs 同步73秒）
pip install aiohttp

# 全量沪深300选股，显示Top 10
python3 scripts/demo_jys_screen.py

# 指定参数
python3 scripts/demo_jys_screen.py --top 5 --detail
python3 scripts/demo_jys_screen.py --codes 601318,600519,600036
python3 scripts/demo_jys_screen.py --min-score 50 --max-pe 25
```

```python
from stock_selection import JYSScreener, JYSScorer

# 一键选股：获取数据 → 五维评分 → 排名筛选
screener = JYSScreener(top_n=10, min_score=40)
top_df, all_df = screener.screen_all()
print(top_df[['symbol', 'name', 'composite_score', 'grade']])

# 单只股票评分
scorer = JYSScorer()
result = scorer.calculate_score({
    'code': '601318', 'name': '中国平安', 'price': 54.0,
    'change_pct': 2.5, 'momentum_20d': 8.0, 'turnover_rate': 1.5,
    'pe_ratio': 7.4, 'pb_ratio': 0.96, 'roe': 13.0, 'dividend_yield': 2.36,
})
print(f"总分: {result.total_score} [{result.grade}]")
print(f"技术={result.tech_score}/30 估值={result.valuation_score}/25 "
      f"盈利={result.profit_score}/30 安全={result.safety_score}/10 分红={result.dividend_score}/5")
```

**五维评分体系** (满分100分):

| 维度 | 满分 | 子项 |
|------|------|------|
| 技术面 | 30 | 日涨跌幅(10) + 20日动量(15) + 换手率(5) |
| 估值面 | 25 | PE(10) + PB(10) + PR市赚率(5) |
| 盈利质量 | 30 | ROE(15) + 利润增长(15) |
| 安全性 | 10 | PB安全边际(3) + 股息稳定性(3) + 换手率波动(4) |
| 分红 | 5 | 股息率 |

> 详细文档: [docs/stock_selection.md](docs/stock_selection.md)

### 五维选股可视化

```bash
# 安装 Web 依赖
pip install streamlit aiohttp

# 启动可视化系统
streamlit run web/app.py
```

支持沪深300/A股全量/港股通批量筛选、单股评分、五维雷达图、评级分布等可视化。

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

### 深度学习模块 (`src/dl/`)

| 类 | 说明 |
|---|---|
| `DLConfig` | DL 配置，含 start_date/end_date 日期范围、window/horizon、损失函数、信号阈值 |
| `FeatureBuilder` | 特征工程，从 parquet 构建 OHLCV+指标特征，z-score 标准化，支持日期过滤 |
| `StockDataset` | PyTorch Dataset 包装 (float32 回归标签) |
| `PricePredictor` | Transformer Decoder 回归网络，输出预测收益率 |
| `Trainer` | 训练器，支持训练/断点续训/增量训练/评估(MSE/方向准确率)/单只预测 |
| `DLStrategy` | DL→回测桥接，按 buy/sell 阈值生成 Signal |
| `MetaConfig` | Meta 配置，含子策略列表、日期范围、信号阈值 |
| `MetaModel` | 策略注意力加权网络，Conv1D 时序编码 + Multi-Head Attention，权重可解读 |
| `MetaSignalCollector` | 策略信号采集器，运行子策略 score() 构建连续特征序列 |
| `MetaTrainer` | Meta 训练器，训练后打印策略权重分布 |
| `MetaStrategy` | Meta→回测桥接，滚动窗口维护 score 历史，推理输出 Signal |

### 强化学习模块 (`src/rl/`)

基于 PPO 的截面选股模块，与 DL 模块并行独立。使用 aurumq-rl 的 Deep-Sets 策略网络 + 296 因子库 (Alpha101 + GTJA191)，推理仅需 ONNX Runtime (~50MB)。

| 类 | 说明 |
|---|---|
| `RLConfig` | RL 配置，含宇宙/因子/环境/PPO超参/推理/信号生成 |
| `DataAdapter` | per-stock parquet → panel parquet 转换，桥接 myana-quant 存储与 aurumq-rl 格式 |
| `FactorPanelBuilder` | 调用 aurumq-rl 因子库计算 296 因子，与 IndicatorSet 完全解耦 |
| `RLInference` | ONNX Runtime CPU 推理封装，无需 PyTorch |
| `RLStrategy` | `MultiStrategy` 子类，截面选股适配器，配合 PortfolioEngine 组合回测 |
| `EnvFactory` | 创建 aurumq-rl Gym 环境 (StockPickingEnv / PortfolioWeightEnv) |
| `RLTrainer` | SB3 PPO 训练编排，含面板构建/因子计算/环境创建/训练/ONNX导出 |

```bash
# 构建面板 → 训练 → 回测
python3 scripts/demo_rl_portfolio.py --build-panel --train --backtest
```

> 详细文档: [docs/rl.md](docs/rl.md)

### 五维选股模块 (`src/stock_selection/`)
| `JYSScorer` | 五维评分引擎，技术面30+估值25+盈利质量30+安全性10+分红5，满分100 |
| `JYSScoreResult` | 评分结果，含总分/评级(A+~D)/各维度子分/明细/原始数据 |
| `JYSScreener` | 选股筛选器，组合TencentFetcher+JYSScorer，输出与StockScreener兼容的DataFrame |

**评分体系**: 技术面(日涨跌幅+20日动量+换手率) + 估值面(PE+PB+PR市赚率) + 盈利质量(ROE+利润增长) + 安全性(PB安全边际+股息稳定性+换手率波动) + 分红(股息率)。详见 [docs/stock_selection.md](docs/stock_selection.md)。

### 策略模块 (`src/strategies/`)

| 类 | 说明 |
|---|---|
| `Strategy` | 策略抽象基类，子类实现 `on_init` + `on_bar`，可选覆盖 `score` 返回连续观点分数 |
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
| `DynamicAllocation` | 动态权重分配，供 RL 策略每轮再平衡时更新模型输出权重 |
| `Optimizer` | 参数优化器：网格搜索/遗传/贝叶斯，自定义目标函数 |

**内置策略示例** (`src/strategies/examples/`):

| 策略 | on_bar 触发条件 | score() 含义 |
|------|----------------|-------------|
| `MACrossStrategy` | 金叉/死叉 | MA 偏离度 |
| `MACDStrategy` | DIF/DEA 交叉 | (DIF-DEA)/close |
| `SARStrategy` | SAR 趋势翻转 | 趋势方向 × 距离 |
| `RSIStrategy` | RSI 超买/超卖 | (50-RSI)/50 |
| `KDJStrategy` | K/D 交叉 + J 值极值 | (50-J)/50 |
| `WRStrategy` | WR 超买/超卖 | (-50-WR)/50 |
| `CCIStrategy` | CCI 通道外极值 | -CCI/100 |
| `BollingerStrategy` | 触及上下轨 | 1-2×%B |
| `KeltnerStrategy` | 触及上下轨 | 1-2×%K |
| `OBVStrategy` | OBV 穿越均线 | OBV 偏离 MA |
| `MFIStrategy` | MFI 超买/超卖 | (50-MFI)/50 |
| `VWAPStrategy` | 价格偏离 VWAP | -gap/vwap |

**回测流程：**
1. `strategy.on_init(ctx)` — 注册指标依赖
2. `IndicatorSet.compute(df)` — 向量化预计算所有指标
3. 逐 bar 循环：风控检查 → 更新持仓市值 → 构建上下文 → 生成信号 → 风控过滤 → 执行交易
4. 计算绩效指标：总收益率/年化/夏普/最大回撤/胜率/盈亏比

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
- **数据源**: Baostock（主力）, AkShare（日线备用）, 腾讯财经API（实时基本面/选股）
- **数据分析**: Pandas, NumPy
- **深度学习**: PyTorch (Transformer Decoder, Conv1D + Attention)
- **强化学习**: Stable-Baselines3 (PPO), ONNX Runtime, aurumq-rl (Deep-Sets, 296因子)
- **异步并发**: aiohttp (五维选股, 3秒/300只)
- **可视化**: Matplotlib, Plotly, Streamlit
- **存储**: Parquet
- **回测**: 自研回测引擎（向量化指标 + 逐 bar 信号）

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个模块测试
pytest tests/unit/strategies/ -v
pytest tests/unit/dl/ -v
pytest tests/unit/rl/ -v

# 跳过需要 plotly 的测试
pytest tests/ -v -k "not plotly"
```
