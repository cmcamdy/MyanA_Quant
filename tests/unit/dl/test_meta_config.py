"""MetaConfig 单元测试"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.meta_config import MetaConfig
from strategies.examples.ma_cross import MACrossStrategy
from strategies.examples.sar import SARStrategy


class TestMetaConfig:
    def test_default_values(self):
        config = MetaConfig()
        assert config.window == 20
        assert config.horizon == 1
        assert config.hidden_dim == 32
        assert config.num_strategies == 0

    def test_with_strategies(self):
        strategies = [MACrossStrategy(), SARStrategy()]
        config = MetaConfig(strategies=strategies)
        assert config.num_strategies == 2

    def test_custom_values(self):
        config = MetaConfig(window=30, horizon=5, hidden_dim=64, loss_type="mse")
        assert config.window == 30
        assert config.horizon == 5
        assert config.hidden_dim == 64
        assert config.loss_type == "mse"
