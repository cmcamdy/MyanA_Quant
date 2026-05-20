# 强化学习模块 (`src/rl/`)

## 模块概述

强化学习模块提供基于 PPO 的截面选股能力，与 DL 模块（单股回归预测）并行独立：

| 维度 | DL 模块 | RL 模块 |
|------|---------|---------|
| 范式 | 单股回归预测 | 截面选股排序 |
| 模型 | Transformer Decoder | SB3 PPO (Deep-Sets) |
| 输入 | per-stock `[31, 120]` 滑窗 | 截面 panel `(N_stocks, N_factors)` |
| 输出 | 预测收益率 → 阈值买卖信号 | 个股权重/优先分数 |
| 回测引擎 | BacktestEngine (单股) | PortfolioEngine (多股组合) |
| 因子 | 18 技术指标 (IndicatorSet) | 296 因子 (Alpha101 + GTJA191) |
| 推理依赖 | PyTorch | ONNX Runtime (~50MB) |

## 模块架构

```
src/rl/
├── __init__.py          # 模块导出 (lazy import)
├── config.py            # RLConfig 配置
├── data_adapter.py      # per-stock parquet → panel parquet 转换
├── factor_panel.py      # 调用 aurumq-rl 因子库计算 296 因子
├── inference.py         # ONNX Runtime CPU 推理
├── rl_strategy.py       # RLStrategy(MultiStrategy) 截面选股适配器
├── env_factory.py       # 创建 aurumq-rl Gym 环境
└── trainer.py           # SB3 PPO 训练编排
```

## 依赖安装

```bash
# 推理 (轻量, ~50MB)
pip install onnxruntime>=1.17

# 训练 (需 aurumq-rl 完整依赖)
pip install git+https://github.com/yupoet/aurumq-rl.git
# 或 clone 到 vendor/ 目录
git clone --depth 1 https://github.com/yupoet/aurumq-rl.git vendor/aurumq-rl
```

## 数据适配

RL 模块的核心数据格式差异由 `DataAdapter` 解决：

| myana-quant 存储 | aurumq-rl 期望 |
|---|---|
| `data/{sh\|sz}/{code}/1d.parquet` 每股一个文件 | 单文件面板 `(ts_code, trade_date, close, pct_chg, vol, ...)` |
| 列: `open, high, low, close, volume, amount` | 列: `ts_code, trade_date, close, pct_chg, vol` + 因子列 |

```python
from rl import RLConfig, DataAdapter

config = RLConfig(
    data_dir="./data",
    panel_dir="./data/panels",
    universe="all",         # "all" / "csi300" / json路径
    start_date="2020-01-01",
    end_date="2025-01-01",
)

adapter = DataAdapter(config)
panel_path = adapter.build_panel()           # 构建 panel.parquet
panel_df = adapter.load_panel()              # 加载已构建的面板
```

## 因子计算

使用 aurumq-rl 自带的 296 因子库（Alpha101 107 个 + GTJA191 191 个），与 IndicatorSet 完全解耦：

```python
from rl import FactorPanelBuilder

builder = FactorPanelBuilder(config)
panel_with_factors = builder.compute(panel_df)
# 输出: panel_with_factors.parquet (原始列 + 296 因子列)
```

因子筛选通过 `factor_prefixes` 配置：
```python
config = RLConfig(
    factor_prefixes=["alpha_*", "gtja_*"],  # 默认: 全部 296 因子
    # 或精确选择:
    # factor_prefixes=["alpha_001", "gtja_005"],
)
```

## RLConfig 配置

```python
from rl import RLConfig

config = RLConfig(
    # ── 数据 ──
    data_dir="./data",
    panel_dir="./data/panels",
    start_date="2020-01-01",
    end_date="2025-01-01",

    # ── 宇宙 ──
    universe="all",          # "all" / "csi300" / json路径
    max_stocks=300,          # 最大股票数量
    min_history=250,         # 最低交易天数

    # ── 因子 ──
    factor_prefixes=["alpha_*", "gtja_*"],

    # ── 环境 ──
    env_type="stock_picking",   # "stock_picking" / "portfolio_weight"
    top_k=10,                   # 选股数量
    reward_fn="sharpe",         # "simple_return" / "sharpe" / "sortino" / "mean_variance"
    lookback=20,                # 观测回看天数
    forward_period=5,           # 前向收益天数
    cost_bps=10.0,              # 交易成本(基点)

    # ── 训练 (PPO) ──
    total_timesteps=1_000_000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,

    # ── 推理 ──
    onnx_model_path="./checkpoints/rl/policy.onnx",
    inference_top_k=10,
    rebalance_freq="M",         # "W" / "M" / "Q"

    # ── 信号生成 ──
    signal_mode="top_k",        # "top_k" / "quantile"
    score_quantile_buy=0.8,     # quantile 模式: 80 分位买入
    score_quantile_sell=0.2,    # quantile 模式: 20 分位卖出
)
```

## 训练

```python
from rl import RLTrainer

trainer = RLTrainer(config)
output_dir = trainer.train(symbols=None)  # None = 按 config.universe 解析
# 输出:
#   checkpoints/rl/ppo_model.zip    — SB3 模型
#   checkpoints/rl/policy.onnx      — ONNX 推理模型
#   checkpoints/rl/metadata.json    — 训练元数据
```

训练流程:
1. `DataAdapter.build_panel()` — 构建 panel parquet
2. `FactorPanelBuilder.compute()` — 计算 296 因子
3. `EnvFactory.create_train_val()` — 创建训练/验证环境
4. SB3 PPO 训练 (使用 `PerStockEncoderPolicy` Deep-Sets 架构)
5. 导出 ONNX 模型 + 保存元数据

## 推理与回测

```python
from rl import RLStrategy
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
```

### 信号生成模式

**top_k 模式** (默认): 分数最高的 k 只股票生成 BUY，已持仓但不在 top-k 的生成 SELL。

**quantile 模式**: 分数超过 80 分位 → BUY，低于 20 分位 → SELL。

### 权重模式

使用 `PortfolioWeightEnv` 训练的模型输出连续权重，配合 `DynamicAllocation`：

```python
from strategies.portfolio_engine import DynamicAllocation

allocation = DynamicAllocation()
# RLStrategy 在每轮再平衡时调用 allocation.update_weights()
engine = PortfolioEngine(allocation=allocation)
```

## RLStrategy 设计

`RLStrategy` 继承 `MultiStrategy`，与 `PortfolioEngine` 配合：

- 只在再平衡周期（周/月/季）生成信号，非 per-bar
- `on_bar_multi` 流程: 检查再平衡日 → 构建截面观测 → ONNX 推理 → 选股 → 生成信号
- 观测构建: 从 `ctx.historical` 中提取最近 `lookback` 天的截面特征

## 环境类型

### StockPickingEnv (离散选股)
- 观测: 全截面因子值
- 动作: 每只股票的优先分数 [0, 1]
- 奖励: top-k 股票的前向收益 - 交易成本
- A股约束: 涨跌停/ST/停牌/60日IPO保护/单行业30%上限

### PortfolioWeightEnv (连续权重)
- 观测: 因子值 + 当前持仓权重
- 动作: 连续权重 (投影到单纯形 + 单股/行业上限)
- 奖励: 可选 simple_return / sharpe / sortino / mean_variance

## Demo 脚本

```bash
# 构建面板 (数据准备, 只需运行一次)
python3 scripts/demo_rl_portfolio.py --build-panel

# 训练 RL 模型
python3 scripts/demo_rl_portfolio.py --train

# 使用预训练 ONNX 模型回测
python3 scripts/demo_rl_portfolio.py --backtest

# 全流程
python3 scripts/demo_rl_portfolio.py --build-panel --train --backtest

# 指定参数
python3 scripts/demo_rl_portfolio.py --train --universe all --timesteps 100000
python3 scripts/demo_rl_portfolio.py --backtest --top-k 10 --rebalance-freq M
```

## 与 DL 模块的关系

| 对比项 | DL 模块 | RL 模块 |
|--------|---------|---------|
| 文件位置 | `src/dl/` | `src/rl/` |
| 策略基类 | `Strategy` (单股) | `MultiStrategy` (多股) |
| 回测引擎 | `BacktestEngine` | `PortfolioEngine` |
| 因子系统 | `IndicatorSet` (18指标) | aurumq-rl (296因子) |
| 数据格式 | per-stock parquet | panel parquet |
| 训练框架 | PyTorch 原生 | SB3 PPO |
| 推理方式 | PyTorch 模型加载 | ONNX Runtime |
| 代码耦合 | 无 | 无 |

两个模块完全独立，不共享代码也不互相引用，可以独立演进。
