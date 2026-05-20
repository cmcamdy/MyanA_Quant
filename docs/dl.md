# 深度学习模块 (`src/dl/`)

## 模块概述

深度学习模块提供两个核心功能：
1. **DL 预测器**: 基于 Transformer Decoder 的股价收益率预测网络
2. **Meta Strategy**: 基于注意力机制的学习型策略组合，动态学习多个子策略的权重

## 模块架构

```
src/dl/
├── config.py                # DLConfig 深度学习配置（回归模式）
├── feature_builder.py       # FeatureBuilder 特征工程
├── dataset.py               # StockDataset PyTorch 数据集
├── model.py                 # PricePredictor Transformer 回归网络
├── trainer.py               # Trainer 训练/评估/预测
├── dl_strategy.py           # DLStrategy 桥接回测引擎
├── meta_config.py           # MetaConfig 学习型组合策略配置
├── meta_model.py            # MetaModel 策略注意力加权网络
├── meta_signal_collector.py # MetaSignalCollector 策略信号采集
├── meta_trainer.py          # MetaTrainer 训练/评估
└── meta_strategy.py         # MetaStrategy 桥接回测引擎
```

## DL 预测器

### 工作流程

1. `FeatureBuilder` 从 Parquet 数据构建 OHLCV + 25 个技术指标特征，z-score 标准化
2. `PricePredictor` (Transformer Decoder) 接收 `[B, 31, window]` 的特征窗口，输出预测收益率
3. `Trainer` 负责训练、断点续训、增量训练、评估和单只股票预测
4. `DLStrategy` 将预测结果按阈值转换为 BUY/SELL 信号，桥接回测引擎

### DLConfig

```python
from dl.config import DLConfig

config = DLConfig(
    data_dir="./data",        # parquet 根目录
    window=60,                # 回看窗口天数
    horizon=1,                # 预测未来几天 (1=次日)
    train_ratio=0.7,          # 训练集比例
    val_ratio=0.15,           # 验证集比例
    hidden_dim=256,           # 隐藏层维度
    dropout=0.3,              # Dropout
    batch_size=64,
    learning_rate=1e-3,
    epochs=50,
    early_stopping_patience=5,
    checkpoint_dir="./checkpoints",
    loss_type="huber",        # 'mse' / 'mae' / 'huber'
    buy_threshold=0.005,      # 预测收益率 > 0.5% 买入
    sell_threshold=-0.005,    # 预测收益率 < -0.5% 卖出
    resume=False,             # 断点续训
    incremental=False,        # 增量训练
)
```

### FeatureBuilder

```python
from dl.feature_builder import FeatureBuilder

builder = FeatureBuilder(config)

# 构建单只股票样本
features, labels = builder.build_stock("600036.SH")
# features: [N, 31, window], labels: [N] (归一化收益率)

# 构建多只股票样本
features, labels, symbol_list = builder.build_all(["600036.SH", "600519.SH"])

# 标准化参数保存/加载
builder.save_scaler("checkpoints/scaler.npz")
builder.load_scaler("checkpoints/scaler.npz")
```

特征列: 6 原始列 (OHLCV+A) + 25 技术指标列 = 31 列，由 `IndicatorSet` 自动计算。标签为归一化收益率（基于滚动窗口的 z-score）。

### PricePredictor — Transformer 回归网络

```python
from dl.model import PricePredictor

model = PricePredictor(
    input_dim=3720,    # num_features × window
    hidden_dim=256,
    num_classes=6,     # 历史兼容参数，回归模式忽略
    dropout=0.3,
)

# 前向传播
logits = model(x)  # x: [B, 31, 120] → logits: [B, 1] (预测收益率)
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

# 评估
result = trainer.evaluate(symbols=[...])
result['mse']                # float
result['rmse']               # float
result['direction_accuracy'] # float — 方向预测准确率
result['correlation']        # float — 预测与实际相关系数

# 单只股票预测
pred = trainer.predict("600036.SH")
pred['predicted_return']     # float — 预测收益率
pred['direction']            # str — "up" / "down" / "neutral"
```

### DLStrategy — 桥接回测

```python
from dl.dl_strategy import DLStrategy

strategy = DLStrategy(config)
result = BacktestEngine().run(strategy, df, symbol="600036.SH")
```

按 `buy_threshold` / `sell_threshold` 将预测收益率转换为交易信号。

## Meta Strategy — 学习型组合策略

### 工作原理

1. 每个子策略输出连续观点分数 `score()` (范围 -1 到 +1)
2. `MetaSignalCollector` 运行所有子策略，构建 score 时间序列
3. `MetaModel` 通过 Conv1D 时序编码 + Multi-Head Attention 学习各策略权重
4. 输出预测收益率，策略权重可解读
5. `MetaStrategy` 桥接回测引擎

```
子策略 score 序列 → Conv1D 编码 → Multi-Head Attention → 加权融合 → 预测收益率
```

### MetaConfig

```python
from dl.meta_config import MetaConfig

config = MetaConfig(
    strategies=[
        MACrossStrategy(fast_period=5, slow_period=20),
        SARStrategy(),
        RSIStrategy(period=14),
        BollingerStrategy(period=20),
        # ... 最多 12 个内置策略
    ],
    data_dir="./data",
    checkpoint_dir="./checkpoints/meta",
    start_date="2013-01-01",
    end_date=None,
    window=20,               # score 回看窗口
    horizon=1,               # 预测周期
    hidden_dim=32,           # 注意力网络隐藏维度
    epochs=100,
    batch_size=32,
    loss_type="huber",       # 'mse' / 'mae' / 'huber'
    early_stopping_patience=3,
    buy_threshold=0.005,
    sell_threshold=-0.005,
)
```

### MetaModel — 策略注意力加权网络

```python
from dl.meta_model import MetaModel

model = MetaModel(
    num_strategies=12,       # 子策略数量
    window=20,               # 回看窗口
    hidden_dim=32,           # 隐藏维度
    num_heads=4,             # 注意力头数
)
# 输入: [B, window, num_strategies] — score 序列
# 输出: [B, 1] — 预测收益率
```

**独立编码器 + 策略嵌入**: 每个策略有独立的 Conv1D 编码器 + 可学习嵌入，避免注意力坍缩到单一策略。

### MetaTrainer

```python
from dl.meta_trainer import MetaTrainer

trainer = MetaTrainer(config)
trainer.train(symbols)

# 评估 — 查看各策略权重
result = trainer.evaluate(symbols)
result['mse']
result['rmse']
result['direction_accuracy']
result['avg_strategy_weights']  # List[float] — 各策略平均权重

for i, strat in enumerate(config.strategies):
    w = result['avg_strategy_weights'][i]
    print(f"{type(strat).__name__}: {w:.4f}")
```

### MetaStrategy — 桥接回测

```python
from dl.meta_strategy import MetaStrategy

meta = MetaStrategy(config)
result = BacktestEngine().run(meta, df, symbol="000807.SZ")
```

内部维护滚动窗口的 score 历史，每根 bar 推理输出交易信号。

## Demo 脚本

### demo_dl_predict.py — DL 收益率预测

```bash
# 训练 + 预测
python3 scripts/demo_dl_predict.py --start-date 2013-01-01

# 训练 + 回测
python3 scripts/demo_dl_predict.py --backtest --start-date 2013-01-01
```

### demo_meta_strategy.py — Meta Strategy

```bash
# 训练 + 查看策略权重
python3 scripts/demo_meta_strategy.py --start-date 2013-01-01

# 训练 + 回测
python3 scripts/demo_meta_strategy.py --backtest --start-date 2013-01-01
```

默认使用证监会行业分类中有色金属板块 (`C32有色金属冶炼和压延加工业`) 的股票数据。
