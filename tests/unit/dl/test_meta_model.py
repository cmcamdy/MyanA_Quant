"""MetaModel 单元测试"""

import pytest
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.meta_model import MetaModel


class TestMetaModel:
    @pytest.fixture
    def model(self):
        return MetaModel(num_strategies=4, window=20, hidden_dim=32, dropout=0.1)

    def test_output_shape(self, model):
        x = torch.randn(8, 4, 20)  # 4 strategies, window=20
        pred, weights = model(x)
        assert pred.shape == (8,)
        assert weights.shape == (8, 4)

    def test_single_sample(self, model):
        x = torch.randn(1, 4, 20)
        pred, weights = model(x)
        assert pred.shape == (1,)
        assert weights.shape == (1, 4)

    def test_attention_weights_sum_to_one(self, model):
        model.eval()
        x = torch.randn(4, 4, 20)
        with torch.no_grad():
            _, weights = model(x)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_attention_weights_non_negative(self, model):
        model.eval()
        x = torch.randn(4, 4, 20)
        with torch.no_grad():
            _, weights = model(x)
        assert (weights >= -1e-6).all()

    def test_different_num_strategies(self):
        model = MetaModel(num_strategies=2, window=10, hidden_dim=16)
        x = torch.randn(4, 2, 10)
        pred, weights = model(x)
        assert pred.shape == (4,)
        assert weights.shape == (4, 2)

    def test_predict_method(self, model):
        x = torch.randn(2, 4, 20)
        pred = model.predict(x)
        assert pred.shape == (2,)

    def test_output_not_nan(self, model):
        x = torch.randn(4, 4, 20)
        pred, weights = model(x)
        assert not torch.isnan(pred).any()
        assert not torch.isnan(weights).any()

    def test_eval_no_dropout(self, model):
        model.eval()
        x = torch.randn(2, 4, 20)
        with torch.no_grad():
            out1, w1 = model(x)
            out2, w2 = model(x)
        assert torch.allclose(out1, out2)
        assert torch.allclose(w1, w2)

    def test_strategy_prior_initialization(self):
        """先验权重应直接影响初始注意力分布"""
        prior = [4, 2, 2, 1]
        model_with_prior = MetaModel(num_strategies=4, window=20, hidden_dim=32, strategy_prior=prior)
        model_with_prior.eval()

        x = torch.randn(1, 4, 20)
        with torch.no_grad():
            _, weights = model_with_prior(x)

        # 先验偏置使 softmax 初始输出接近 prior 分布
        # 高先验策略(4)的权重应明显高于低先验策略(1)
        assert weights[0, 0] > weights[0, 3], \
            f"prior[0]=4 的权重 {weights[0,0]:.4f} 应大于 prior[3]=1 的权重 {weights[0,3]:.4f}"

        # 无先验时偏置为0，权重应接近均匀
        model_uniform = MetaModel(num_strategies=4, window=20, hidden_dim=32)
        model_uniform.eval()
        with torch.no_grad():
            _, weights_uniform = model_uniform(x)
        # 无先验时 attn_prior_bias 全为0
        assert torch.allclose(model_uniform.attn_prior_bias, torch.zeros(4), atol=1e-6)
