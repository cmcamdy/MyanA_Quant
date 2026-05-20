"""Meta Strategy 模块配置"""

from dataclasses import dataclass, field
from typing import List, Optional

from strategies.base import Strategy


@dataclass
class MetaConfig:
    # 子策略列表
    strategies: List[Strategy] = field(default_factory=list)

    # 数据
    data_dir: str = "./data"
    start_date: Optional[str] = None   # 数据起始日期, 如 "2013-01-01"
    end_date: Optional[str] = None     # 数据截止日期, 如 "2026-01-01"
    window: int = 20       # 信号历史窗口
    horizon: int = 1       # 预测未来 N 天收益
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # 模型
    hidden_dim: int = 32
    dropout: float = 0.1

    # 训练
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 10
    loss_type: str = "huber"
    huber_delta: float = 1.0

    # 路径
    checkpoint_dir: str = "./checkpoints/meta"

    # 断点续训
    resume: bool = False

    # 回测信号阈值
    buy_threshold: float = 0.002
    sell_threshold: float = -0.002

    @property
    def num_strategies(self) -> int:
        return len(self.strategies)
