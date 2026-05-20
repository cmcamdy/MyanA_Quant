"""深度学习涨跌预测模块配置"""

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

# 涨跌分类标签
CLASS_LABELS = [
    '大跌 (<-5%)',
    '中跌 (-5%~-2%)',
    '小跌 (-2%~0%)',
    '小涨 (0%~2%)',
    '中涨 (2%~5%)',
    '大涨 (>5%)',
]

# 涨跌幅区间边界
CLASS_BOUNDS = [-float('inf'), -0.05, -0.02, 0.0, 0.02, 0.05, float('inf')]


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
    num_classes: int = 6
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

    # 路径
    checkpoint_dir: str = "./checkpoints"

    # 断点续训
    resume: bool = False

    # 增量训练
    incremental: bool = False
    incremental_lr_factor: float = 0.1

    def get_indicators(self):
        return self.indicators if self.indicators is not None else DEFAULT_INDICATORS
