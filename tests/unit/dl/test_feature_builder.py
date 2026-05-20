"""FeatureBuilder 单元测试"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.config import DLConfig
from dl.feature_builder import FeatureBuilder


def _make_ohlcv_df(n_rows=500):
    """生成合成 OHLCV DataFrame"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='B')
    close = 10.0 + np.cumsum(np.random.randn(n_rows) * 0.1)
    df = pd.DataFrame({
        'open': close + np.random.randn(n_rows) * 0.05,
        'high': close + abs(np.random.randn(n_rows) * 0.1),
        'low': close - abs(np.random.randn(n_rows) * 0.1),
        'close': close,
        'volume': np.random.randint(1e6, 1e8, n_rows).astype(float),
        'amount': np.random.uniform(1e8, 1e10, n_rows),
    }, index=dates)
    return df


class TestFeatureBuilder:
    @pytest.fixture
    def config(self):
        return DLConfig(window=60, horizon=1)

    @pytest.fixture
    def builder(self, config):
        return FeatureBuilder(config)

    def test_make_labels(self, builder):
        close = pd.Series([100, 101, 103, 99, 95, 105, 108, 107, 96, 102], dtype=float)
        labels = builder._make_labels(close)
        assert len(labels) == len(close)
        # 回归标签: 连续收益率
        assert labels.dtype == np.float32

    def test_make_labels_values(self, builder):
        close = pd.Series([100.0, 110.0], dtype=float)
        labels = builder._make_labels(close)
        # horizon=1, close[0]=100, future_close[0]=110, return=0.1
        assert abs(labels[0] - 0.1) < 1e-5

    def test_make_labels_horizon(self):
        config = DLConfig(window=60, horizon=5)
        builder = FeatureBuilder(config)
        close = pd.Series([100, 101, 103, 99, 95, 105, 108, 107, 96, 102, 110, 115], dtype=float)
        labels = builder._make_labels(close)
        assert len(labels) == len(close)
        # 末尾 horizon 个标签应为 NaN（无未来数据）
        for i in range(len(labels) - 5, len(labels)):
            assert np.isnan(labels[i])

    def test_normalize_features(self, builder):
        np.random.seed(42)
        features = np.random.randn(50, 10, 60).astype(np.float32)
        normalized = builder.normalize_features(features, fit=True)
        assert normalized.shape == features.shape
        # 标准化后每个特征的均值应接近 0
        mean_per_feat = normalized.mean(axis=(0, 2))
        assert np.abs(mean_per_feat).max() < 0.1

    def test_normalize_features_no_refit(self, builder):
        np.random.seed(42)
        features = np.random.randn(50, 10, 60).astype(np.float32)
        builder.normalize_features(features, fit=True)
        # 用同一 scaler 标准化新数据
        features2 = np.random.randn(20, 10, 60).astype(np.float32)
        normalized2 = builder.normalize_features(features2, fit=False)
        assert normalized2.shape == features2.shape

    def test_save_load_scaler(self, builder, tmp_path):
        np.random.seed(42)
        features = np.random.randn(50, 10, 60).astype(np.float32)
        builder.normalize_features(features, fit=True)

        path = str(tmp_path / "scaler.npz")
        builder.save_scaler(path)
        builder2 = FeatureBuilder(DLConfig(window=60))
        builder2.load_scaler(path)
        np.testing.assert_array_almost_equal(builder._scaler_mean, builder2._scaler_mean)
        np.testing.assert_array_almost_equal(builder._scaler_std, builder2._scaler_std)

    @patch.object(FeatureBuilder, '_load_parquet')
    def test_build_stock(self, mock_load, builder):
        mock_load.return_value = _make_ohlcv_df(500)
        result = builder.build_stock("600036.SH")
        assert result is not None
        features, labels = result
        assert features.ndim == 3
        assert features.shape[1] > 6  # 原始6列 + 指标列
        assert features.shape[2] == 60  # window
        assert len(labels) == features.shape[0]
        # 回归标签: float32 连续值
        assert labels.dtype == np.float32

    @patch.object(FeatureBuilder, '_load_parquet')
    def test_build_stock_insufficient_data(self, mock_load, builder):
        mock_load.return_value = _make_ohlcv_df(50)  # 少于 window
        result = builder.build_stock("600036.SH")
        assert result is None

    @patch.object(FeatureBuilder, '_load_parquet')
    def test_build_stock_missing_data(self, mock_load, builder):
        mock_load.return_value = None
        result = builder.build_stock("999999.SH")
        assert result is None
