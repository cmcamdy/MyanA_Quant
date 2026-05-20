"""DLConfig 单元测试"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.config import DLConfig, DEFAULT_INDICATORS


class TestDLConfig:
    def test_default_values(self):
        config = DLConfig()
        assert config.window == 120
        assert config.horizon == 1
        assert config.hidden_dim == 256
        assert config.dropout == 0.3
        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.loss_type == "huber"
        assert config.buy_threshold == 0.005
        assert config.sell_threshold == -0.005

    def test_custom_values(self):
        config = DLConfig(window=60, horizon=5, hidden_dim=512, epochs=100, loss_type="mse")
        assert config.window == 60
        assert config.horizon == 5
        assert config.hidden_dim == 512
        assert config.epochs == 100
        assert config.loss_type == "mse"

    def test_get_indicators_default(self):
        config = DLConfig()
        indicators = config.get_indicators()
        assert indicators == DEFAULT_INDICATORS
        assert len(indicators) == 18

    def test_get_indicators_custom(self):
        custom = [('ma', {'period': 5}), ('rsi', {'period': 14})]
        config = DLConfig(indicators=custom)
        assert config.get_indicators() == custom

    def test_thresholds(self):
        config = DLConfig(buy_threshold=0.01, sell_threshold=-0.01)
        assert config.buy_threshold == 0.01
        assert config.sell_threshold == -0.01
