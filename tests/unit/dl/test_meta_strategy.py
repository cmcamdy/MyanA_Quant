"""MetaStrategy 集成测试"""

import pytest
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.meta_config import MetaConfig
from dl.meta_trainer import MetaTrainer
from dl.meta_strategy import MetaStrategy
from strategies.examples.ma_cross import MACrossStrategy
from strategies.examples.sar import SARStrategy


class TestMetaTrainer:
    def test_create_model(self):
        strategies = [MACrossStrategy(), SARStrategy()]
        config = MetaConfig(strategies=strategies, window=10)
        trainer = MetaTrainer(config)
        # 2 strategies + 3 market features = 5 channels
        model = trainer._create_model(num_channels=5)
        assert isinstance(model, torch.nn.Module)
        x = torch.randn(1, 5, 10)
        pred, weights = model(x)
        assert pred.shape == (1,)
        assert weights.shape == (1, 5)

    def test_create_criterion(self):
        from dl.meta_trainer import _create_criterion
        assert isinstance(_create_criterion("mse"), torch.nn.MSELoss)
        assert isinstance(_create_criterion("huber"), torch.nn.HuberLoss)
        with pytest.raises(ValueError):
            _create_criterion("unknown")


class TestMetaStrategy:
    def test_init(self):
        strategies = [MACrossStrategy(), SARStrategy()]
        config = MetaConfig(strategies=strategies, window=10)
        meta = MetaStrategy(config)
        assert meta.config.num_strategies == 2
        assert meta.buy_threshold == 0.002
