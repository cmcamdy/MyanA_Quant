"""PyTorch Dataset 包装"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """股票预测数据集

    Args:
        features: [N, num_features, window] numpy 数组
        labels: [N] numpy 数组 (float32 收益率)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features) if isinstance(features, np.ndarray) else features
        self.labels = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
