"""深度学习收益率预测模块配置"""

from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_INDICATORS = [
    ('ma', {'period': 5}),
    ('ma', {'period': 10}),
    ('ma', {'period': 20}),
    ('ma', {'period': 60}),
    ('ema', {'period': 12}),
    ('ema', {'period': 26}),
    ('macd', {}),
    ('rsi', {'period': 6}),
    ('rsi', {'period': 14}),
    ('kdj', {}),
    ('bollinger', {}),
    ('atr', {'period': 14}),
    ('obv', {}),
    ('mfi', {'period': 14}),
    ('wr', {'period': 14}),
    ('cci', {'period': 14}),
    ('vwap', {}),
    ('sar', {}),
]


@dataclass
class DLConfig:
    # 数据
    data_dir: str = "./data"
    window: int = 120
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # 特征
    indicators: Optional[List[tuple]] = None

    # 模型
    hidden_dim: int = 256
    dropout: float = 0.3

    # Transformer
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256

    # 训练
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 5
    loss_type: str = "huber"  # "mse" | "huber" | "mae"
    huber_delta: float = 1.0  # Huber loss 的 delta (归一化标签后，1.0 = 1个标准差)

    # 路径
    checkpoint_dir: str = "./checkpoints"

    # 断点续训
    resume: bool = False

    # 增量训练
    incremental: bool = False
    incremental_lr_factor: float = 0.1

    # 回测信号阈值
    buy_threshold: float = 0.005   # 预测收益率 > 此值生成 BUY
    sell_threshold: float = -0.005  # 预测收益率 < 此值生成 SELL

    def get_indicators(self):
        return self.indicators if self.indicators is not None else DEFAULT_INDICATORS
