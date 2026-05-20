"""DLConfig 单元测试"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.config import DLConfig, DEFAULT_INDICATORS, CLASS_LABELS, CLASS_BOUNDS


class TestDLConfig:
    def test_default_values(self):
        config = DLConfig()
        assert config.window == 120
        assert config.horizon == 1
        assert config.num_classes == 6
        assert config.hidden_dim == 256
        assert config.dropout == 0.3
        assert config.epochs == 50
        assert config.batch_size == 64

    def test_custom_values(self):
        config = DLConfig(window=60, horizon=5, hidden_dim=512, epochs=100)
        assert config.window == 60
        assert config.horizon == 5
        assert config.hidden_dim == 512
        assert config.epochs == 100

    def test_get_indicators_default(self):
        config = DLConfig()
        indicators = config.get_indicators()
        assert indicators == DEFAULT_INDICATORS
        assert len(indicators) == 18  # 18 种指标配置

    def test_get_indicators_custom(self):
        custom = [('ma', {'period': 5}), ('rsi', {'period': 14})]
        config = DLConfig(indicators=custom)
        assert config.get_indicators() == custom


class TestClassLabels:
    def test_label_count(self):
        assert len(CLASS_LABELS) == 6

    def test_bounds_count(self):
        assert len(CLASS_BOUNDS) == 7  # n+1 个边界

    def test_bounds_monotonic(self):
        for i in range(len(CLASS_BOUNDS) - 1):
            assert CLASS_BOUNDS[i] < CLASS_BOUNDS[i + 1]
