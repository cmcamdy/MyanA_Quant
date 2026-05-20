"""强化学习模块配置"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RLConfig:
    # ── 数据 ──
    data_dir: str = "./data"
    panel_dir: str = "./data/panels"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # ── 宇宙 ──
    universe: str = "all"           # "all" / "csi300" / json路径
    max_stocks: int = 300
    min_history: int = 250          # 最低交易天数

    # ── 因子 ──
    factor_prefixes: List[str] = field(
        default_factory=lambda: ["alpha_*", "gtja_*"]
    )

    # ── 环境 ──
    env_type: str = "stock_picking"  # "stock_picking" / "portfolio_weight"
    lookback: int = 20               # 观测回看天数
    top_k: int = 10                  # stock_picking: 选股数量
    reward_fn: str = "sharpe"        # "simple_return" / "sharpe" / "sortino" / "mean_variance"
    forward_period: int = 5          # 前向收益计算天数
    cost_bps: float = 10.0           # 交易成本(基点)

    # ── 训练 (PPO) ──
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # ── 日志 ──
    wandb_project: Optional[str] = None
    tensorboard_log: Optional[str] = "./logs/rl"
    verbose: int = 1

    # ── 路径 ──
    checkpoint_dir: str = "./checkpoints/rl"

    # ── 推理 ──
    onnx_model_path: Optional[str] = None
    inference_top_k: int = 10
    rebalance_freq: str = "M"        # "W" / "M" / "Q"

    # ── 信号生成 ──
    signal_mode: str = "top_k"       # "top_k" / "quantile" / "threshold"
    score_quantile_buy: float = 0.8
    score_quantile_sell: float = 0.2
