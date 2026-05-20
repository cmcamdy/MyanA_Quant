# 开发文档

## 项目概述

量化交易分析系统，用于策略研究、回测和分析，不包含实盘交易功能。

## 模块开发状态

| 模块 | 状态 | 测试 | 说明 |
|------|------|------|------|
| `src/data/` | 已完成 | 138 用例 | 数据获取与存储 |
| `src/analysis/` | 已完成 | 55 用例 | 技术指标计算 |
| `src/strategies/` | 已完成 | 110 用例 | 策略回测引擎 + 风控 + 组合回测 + 参数优化 |
| `src/visualization/` | 已完成 | 13+5 用例 | 图表可视化 |
| `src/factors/` | 已完成 | 31 用例 | 因子分析 |
| `src/data/cleaner.py` | 已完成 | 45 用例 | 数据清洗 |
| `src/reports/` | 已完成 | 18 用例 | HTML报告生成 |
| `src/monitor/` | 已完成 | 27 用例 | 实时策略监控 |
| `scripts/` | 已完成 | — | 数据批量下载 |

## 架构设计

### 数据流

```
数据源 (AkShare / Baostock)
    ↓ DataManager (cache-aside)
本地存储 (Parquet: {market}/{code}/{freq}.parquet)
    ↓ DataFrame (OHLCV)
IndicatorSet.compute() → DataFrame + 指标列
    ↓
BacktestEngine / PortfolioEngine
    ↓ RiskManager (止损/止盈/仓位限制)
    ↓ BacktestResult
    ↓
Optimizer (参数网格搜索)
    ↓
可视化 (plot_kline / plot_equity / plot_trade_pnl)
```

### 设计原则

- **ABC + 协议**: DataProvider 用抽象基类约束接口，PositionSizer 用 Protocol 定义协议
- **函数式指标**: 每个指标是纯函数，输入 Series 输出 DataFrame，无副作用
- **懒加载**: `__init__.py` 使用 `__getattr__` 延迟导入数据源类，避免缺少依赖时全局报错
- **向量化优先**: 指标通过 pandas/numpy 向量化预计算，回测循环只做 O(1) 工作

## 各模块关键实现

### 数据模块 (`src/data/`)

**数据源优先级**: 日线优先 AkShare（速度快），分钟线只能用 Baostock。`DataManager` 按 `FREQ_SOURCE_PRIORITY` 自动选择。

**ParquetStorage 存储路径**:
```
data/
├── SH/
│   ├── 600036/
│   │   ├── 1d.parquet
│   │   └── 5min.parquet
│   └── 603083/
│       └── 1d.parquet
└── SZ/
    └── 000807/
        └── 1d.parquet
```

**save 时自动合并**: 如果本地已有数据，会 concat 后去重（以 index 为准），保留最新值。

**Baostock 登录生命周期**: `BaostockProvider` 在每次 `get_kline` 调用内部 login/logout，不持有长连接。

**代码格式转换**: AkShare 用 `000001` 格式，Baostock 用 `sh.600000` 格式，显示统一为 `000001.SZ`。各 Provider 的 `normalize_symbol` 负责转换。

**行业分类查询** (`src/data/industry.py` -- `IndustryLookup`):
- 数据源: Baostock `query_stock_industry()`，证监会行业分类（83 个行业，~5200 只股票）
- 存储: `data/industry-stock/industry_stock.csv`（CSV 格式，列: industry, code, name）
- 懒加载: 首次调用时读取 CSV，后续复用内存缓存
- 支持行业→股票（`get_stocks`）、股票→行业（`get_industry`）双向查询

### 深度学习预测模块 (`src/dl/`)

**模型结构**: Linear → LayerNorm → ReLU → Dropout → Linear，6 分类涨跌预测。输入展平后维度 = num_features × window（默认 31×120=3720）。

**特征工程** (`FeatureBuilder`):
- 从本地 parquet 读取 OHLCV，通过 `IndicatorSet` 计算 18 种技术指标，共 31 个特征列
- 滑动窗口构建样本：每个交易日取前 window 天的 [31, window] 特征矩阵
- 标签生成：未来 horizon 天收益率按 6 个区间映射（<-5%, -5%~-2%, -2%~0%, 0%~2%, 2%~5%, >5%）
- z-score 标准化：按训练集计算均值/标准差，保存为 scaler.npz 供推理时复用
- NaN/Inf 处理：指标初始几行用 0 填充

**训练器** (`Trainer`):
- 完整训练：FeatureBuilder 构建数据 → 按时间顺序划分 train/val/test → 训练 + 早停 → 保存最佳模型
- 断点续训：加载 checkpoint 中的模型权重和 optimizer 状态，从断点 epoch 继续
- 增量训练：降低学习率（×0.1）fine-tune 已有模型
- 回归测试：加载最佳模型在测试集上评估，输出 accuracy/classification_report/confusion_matrix
- 单只预测：构建特征 → softmax → 6 分类概率

**Checkpoint 内容**: 模型权重、optimizer 状态、epoch、best_val_loss、scaler 参数、训练元信息（特征维度、股票列表）

### 指标模块 (`src/analysis/`)

**指标注册机制**: `IndicatorSet._INDICATOR_REGISTRY` 维护名称→(函数, 必需列) 的映射。`Strategy.register_indicator` 声明依赖，引擎统一预计算。

**RSI Wilder 平滑**: 使用 `ewm(alpha=1/period, min_periods=1)` 实现。边界情况：单调上涨时 avg_loss=0，RSI 应为 100 而非 50。

**KDJ 零范围处理**: 当 `high_n == low_n`（一字板）时，RSV 默认为 50。

**VWAP 日内重置**: 通过 `groupby(df.index.date).cumsum()` 按自然日分组，确保每日重新计算。

**ATR 首根K线**: 首根K线的 TR 直接取 `high - low`，避免 NaN。

**DEMA 双指数平滑**: DEMA = 2*EMA - EMA(EMA)，消除单次EMA滞后。两次 `ewm` 调用实现。

**SAR 抛物线指标**: 唯一的迭代式指标，逐 bar 维护 AF（加速因子）、EP（极值点）、SAR 值。价格穿越 SAR 时翻转方向，SAR 不能进入前两根 bar 的价格范围。输出 `sar_value`（止损位）和 `sar_trend`（1=上升/-1=下降）。

**WR 威廉指标**: RSV 的原始无平滑版本，范围 [-100, 0]。当 HH=LL 时默认 WR=-50。

**CCI 商品通道指标**: TP=(H+L+C)/3，CCI = (TP-SMA)/0.015*MAD。无界指标，MAD=0 时输出 NaN。

**Keltner 通道**: 复用现有 `atr()` 函数，Mid=EMA(close)，Upper/Lower = Mid ± num_atr * ATR。

**Chaikin 波动率**: SMA(H-L) 的变化率，衡量波动率扩张/收缩。

**OBV 能量潮**: `sign(close.diff()) * volume` 的累加，首根 bar diff=NaN 填充为 0。

**MFI 资金流量指标**: 结构类似 RSI，但用 TP*Volume 替代纯价格涨跌。neg_mf=0 时 MFI=100。

### 策略模块 (`src/strategies/`)

**回测引擎流程** (`BacktestEngine.run`):
1. 调用 `strategy.on_init(ctx)` 注册指标（必须在预计算之前）
2. `IndicatorSet.compute(df)` 向量化计算所有指标
3. 若配置了 RiskManager 且需要 ATR，自动注册并计算 ATR
4. 逐 bar 循环:
   - 更新持仓市值 (`pos.market_value = quantity * close`)
   - RiskManager.check_exits() 检查止损/止盈 → 强制平仓
   - 构建上下文 (`Context`)
   - `strategy.on_bar(ctx)` 生成信号
   - RiskManager.check_signals() 过滤信号（仓位/回撤/日亏损限制）
   - 信号类型为 BUY 且空仓 → 买入；SELL 且持仓 → 卖出
   - 计算手续费（按名义金额）和滑点（价格偏移）
5. `strategy.on_finish(ctx)`
6. `calc_performance_metrics()` 计算绩效

**滑点模型**: 买入价 = close * (1 + slippage)，卖出价 = close * (1 - slippage)

**手续费**: 双向收取，按名义金额乘以费率

**仓位管理**: `PositionSizer` 协议，`FixedSizer` 支持 strength 加权（signal.strength * percent * cash）

**组合策略** (`CompositeStrategy`):
- `unanimous`: 所有子策略一致才发信号
- `any`: 任一子策略发信号即触发（但买卖不能同时出现）
- `majority`: 过半数子策略一致才发信号

### 风控模块 (`src/strategies/risk.py`)

**RiskConfig** 配置项:
- 固定止损: `stop_loss_pct` — `price <= avg_cost * (1 - pct)`
- ATR止损: `stop_loss_atr_multiplier` — `price <= avg_cost - multiplier * atr`
- 追踪止损: `trailing_stop_pct` — `price <= peak_price * (1 - pct)`，每 bar 更新 peak
- ATR追踪: `trailing_stop_atr_multiplier` — `price <= peak_price - multiplier * atr_at_peak`
- 固定止盈: `take_profit_pct` — `price >= avg_cost * (1 + pct)`
- ATR止盈: `take_profit_atr_multiplier` — `price >= avg_cost + multiplier * atr`
- 仓位限制: `max_positions` 阻止新买入，`max_per_symbol_weight` 限制单标的权重
- 回撤限制: `max_portfolio_drawdown` — 超出则清仓全部持仓
- 日亏损: `daily_loss_limit` / `daily_loss_limit_pct` — 超出则当日不再交易

**RiskManager 拦截流程**:
1. `check_exits()` — 在策略信号之前，检查所有持仓的止损/止盈条件
2. `check_signals()` — 在策略信号之后，过滤不符合仓位/回撤/日亏损限制的买入信号
3. `register_trailing_stop()` / `remove_trailing_stop()` — 买入时注册追踪，卖出时移除

**ATR 自动注册**: 引擎检测 RiskConfig.needs_atr，若策略未注册 ATR 则自动添加。

### 组合回测引擎 (`src/strategies/portfolio_engine.py`)

**MultiStrategy**: 继承 Strategy，新增 `on_init_multi()` 和 `on_bar_multi()`。普通 Strategy 在多标的上逐个调用 on_bar，MultiStrategy 一次调用返回多标的信号列表。

**时间轴对齐**: 取所有标的索引的并集。某标的某 bar 缺失时，用 last_known 收盘价更新市值但不生成信号。

**资金分配**: `EqualWeightAllocation` 等 1/N 权重，`CustomAllocation` 自定义权重字典。买入时 `quantity = sizer.compute_quantity() * allocation_pct`。

**分标的跟踪**: `per_symbol_equity` 和 `per_symbol_trades` 按标的记录权益和交易。

**再平衡** (`RebalanceConfig`): 支持时间触发 (`frequency='W'/'M'/'Q'`) 和漂移触发 (`drift_threshold`)。再平衡时卖出全部持仓，按目标权重重新买入。`record_trades` 控制是否产生交易记录。

### 参数优化器 (`src/strategies/optimizer.py`)

**ParamRange**: 支持离散值列表 (`values`) 和连续范围 (`start/stop/step`)，`iter_values()` 生成参数值列表。

**Optimizer**: 笛卡尔积生成参数网格 → 策略工厂创建实例 → 逐个运行回测 → 按目标函数降序排列。支持 `'sharpe'`、`'return'` 字符串目标或自定义 callable。top_N 保留完整 BacktestResult，其余只保留参数和目标值。

**GeneticOptimizer**: 种群进化搜索。轮盘赌选择、单点交叉、均匀变异、精英保留。参数去重后按目标排序。

**BayesianOptimizer**: 高斯过程代理模型 + Expected Improvement 采集函数。离散参数编码到 [0,1] 区间。sklearn 可用时使用 Matern 核 GP，否则退化为随机搜索。

### 可视化模块 (`src/visualization/`)

**双引擎**: matplotlib 生成静态图（适合保存 PNG/SVG），plotly 生成交互图（适合 Jupyter/浏览器）

**K线图**: matplotlib 手绘蜡烛图（Rectangle + vline），plotly 用 `go.Candlestick`。自动检测 volume 列决定是否画成交量子图。

**中文字体**: `_setup_chinese_font()` 按 macOS/Windows/Linux 选择对应字体，避免中文乱码。

**买卖点标注**: 买入用绿色 "B" 标注在 low 下方，卖出用红色 "S" 标注在 high 上方。

**布林带填充**: 上下轨之间用半透明区域填充，plotly 用 `fill='toself'` 实现。

### 因子分析模块 (`src/factors/`)

**Factor ABC**: `compute(df) -> pd.Series` 纯函数接口。内置因子: MomentumFactor, VolatilityFactor, TurnoverFactor, ReversalFactor, PriceVolumeFactor, BiasFactor。

**FactorAnalyzer**: IC 分析 (Spearman 秩相关)、IR (IC均值/IC标准差)、分位数分析 (5 分位收益差异)、批量分析 (`analyze_batch`)。复合因子用 z-score 标准化后等权加总。

**FactorScreener**: 基于因子得分排序 (`score`)、选股 (`rank`)、筛选 (`filter`)。

### 数据清洗模块 (`src/data/cleaner.py`)

**DataCleaner** 核心方法:
- `detect_missing(df)` → 布尔 DataFrame 标记缺失位置
- `detect_outliers(df, method='mad'/'zscore'/'iqr', threshold)` → 仅检测 OHLCV 列，NaN 不标记为异常
- `fill_missing(df, method='ffill'/'bfill'/'interpolate'/'mean', limit)` → 填充缺失值
- `replace_outliers(df, method, replacement='clip'/'nan'/'median')` → 替换异常值
- `validate_ohlcv(df)` → 检查 high>=open/close/low、volume>=0、索引单调、无重复
- `clean(df, ...)` → 流水线: 检测异常→替换为NaN→填充缺失
- `quality_report(df, ...)` → DataQualityReport 包含总行数、缺失计数、异常计数、校验错误

**DataQualityReport**: `summary()` 方法生成文本摘要。

### HTML报告模块 (`src/reports/`)

**ReportGenerator**: 将 BacktestResult 生成自包含 HTML 文件。内嵌 SVG 图表（权益曲线、回撤、交易盈亏），使用 matplotlib Agg 后端 + StringIO 渲染。CSS 和图表全部内联，无外部依赖。

### 实时监控模块 (`src/monitor/`)

**LiveEngine**: 实时策略监控引擎，复用 Strategy/MultiStrategy ABC。主循环: 轮询最新数据 → 驱动策略生成信号 → 执行交易 → 更新仪表盘 → 保存状态。

**DataPoller**: 封装 DataManager.update_kline()，跟踪 last_processed_time，只返回新增 bar。网络异常时 log warning 并跳过，下次重试。

**ConsoleDashboard**: 终端仪表盘，用 ANSI 控制码原地重绘。显示组合概览、持仓详情、最近信号/风控告警、权益迷你图 (Unicode block 字符)。刷新频率独立于轮询频率。

**AlertManager**: 告警管理器，支持自定义 AlertCallback 协议。内置 LoggingCallback 写 Python logging。告警类型: 信号触发、风控触发、权益显著变动。告警记录保留最近 50 条。

**MonitorState**: 状态持久化，将 Portfolio、RiskManager 内部状态 (追踪止损、峰值权益等)、open trades、权益/交易历史序列化为 JSON。原子写入 (.tmp + rename) 防崩溃。保存时自动截断 equity_history (500条) 和 trade_history (200条)。

**RiskManager 状态访问**: `get_state()` / `set_state()` 方法导出/恢复内部可变状态，使 LiveEngine 的重启恢复成为可能。

**策略初始化持仓**: BacktestConfig.initial_positions 支持设置预持仓 (avg_cost + quantity)，BacktestEngine/PortfolioEngine 的 `strategy_start` 参数控制策略信号开始时间，之前只跟踪市值。

## 测试

```bash
pytest tests/ -v                    # 全部
pytest tests/unit/data/ -v          # 数据模块
pytest tests/unit/analysis/ -v      # 指标模块
pytest tests/unit/strategies/ -v    # 策略模块
pytest tests/unit/visualization/ -v # 可视化
pytest tests/unit/factors/ -v       # 因子分析
pytest tests/unit/monitor/ -v       # 实时监控
```

测试使用合成数据，不依赖外部服务（数据源用 mock）。

## 已知限制

- 简单滑点模型：固定百分比偏移，非真实订单簿模拟

## 后续规划

（原6项规划已全部完成）
