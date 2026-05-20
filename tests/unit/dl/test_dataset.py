"""StockDataset 单元测试"""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.dataset import StockDataset


class TestStockDataset:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        features = np.random.randn(100, 31, 120).astype(np.float32)
        labels = np.random.randint(0, 6, size=100).astype(np.int64)
        return features, labels

    def test_len(self, sample_data):
        features, labels = sample_data
        ds = StockDataset(features, labels)
        assert len(ds) == 100

    def test_getitem(self, sample_data):
        features, labels = sample_data
        ds = StockDataset(features, labels)
        feat, lab = ds[0]
        assert isinstance(feat, torch.Tensor)
        assert isinstance(lab, torch.Tensor)
        assert feat.shape == (31, 120)
        assert lab.shape == ()

    def test_feature_dtype(self, sample_data):
        features, labels = sample_data
        ds = StockDataset(features, labels)
        feat, _ = ds[0]
        assert feat.dtype == torch.float32

    def test_label_dtype(self, sample_data):
        features, labels = sample_data
        ds = StockDataset(features, labels)
        _, lab = ds[0]
        assert lab.dtype == torch.int64

    def test_label_range(self, sample_data):
        features, labels = sample_data
        ds = StockDataset(features, labels)
        for i in range(len(ds)):
            _, lab = ds[i]
            assert 0 <= lab.item() <= 5
