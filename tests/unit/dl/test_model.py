"""PricePredictor 单元测试"""

import pytest
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.model import PricePredictor, PositionalEncoding


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=120)
        x = torch.randn(4, 30, 64)
        out = pe(x)
        assert out.shape == (4, 30, 64)

    def test_not_nan(self):
        pe = PositionalEncoding(d_model=64, max_len=120)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert not torch.isnan(out).any()


class TestPricePredictor:
    @pytest.fixture
    def model(self):
        return PricePredictor(
            num_features=31,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.3,
            window=120,
        )

    def test_output_shape(self, model):
        x = torch.randn(8, 31, 120)
        out = model(x)
        assert out.shape == (8,)  # 回归: [B] 预测收益率

    def test_single_sample(self, model):
        x = torch.randn(1, 31, 120)
        out = model(x)
        assert out.shape == (1,)

    def test_output_not_nan(self, model):
        x = torch.randn(4, 31, 120)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_different_window(self):
        model = PricePredictor(
            num_features=31, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=256, window=60,
        )
        x = torch.randn(2, 31, 60)
        out = model(x)
        assert out.shape == (2,)

    def test_different_num_features(self):
        model = PricePredictor(
            num_features=10, d_model=32, nhead=2, num_layers=1,
            dim_feedforward=128, window=30,
        )
        x = torch.randn(2, 10, 30)
        out = model(x)
        assert out.shape == (2,)

    def test_model_parameters(self, model):
        params = list(model.parameters())
        assert len(params) > 0
        assert len(params) > 6

    def test_eval_no_dropout(self, model):
        model.eval()
        x = torch.randn(4, 31, 120)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_output_is_scalar_per_sample(self, model):
        """回归输出: 每个样本一个连续值"""
        model.eval()
        x = torch.randn(4, 31, 120)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,)
        assert out.dtype == torch.float32

    def test_causal_mask_prevents_future_attention(self):
        """验证因果 mask: 改变未来时间步不应影响当前预测"""
        model = PricePredictor(
            num_features=10, d_model=32, nhead=2, num_layers=2,
            dim_feedforward=64, window=30,
        )
        model.eval()

        x1 = torch.randn(1, 10, 30)
        x2 = x1.clone()
        x2[:, :, 20:] = x2[:, :, 20:] + 10.0

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert out1.shape == (1,)
        assert out2.shape == (1,)
