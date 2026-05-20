"""Meta Model: 策略观点时序编码 + 注意力加权

架构:
  对每个子策略的连续 score 序列 [1, W]，用独立 Conv1D 编码时序特征，
  加入可学习的策略嵌入向量区分不同策略，
  再通过带先验偏置的 Attention 学习策略权重，加权组合后输出预测收益率。

  注意力权重可直接解读为各策略的相对重要性。
  先验偏置确保初始权重匹配用户指定的策略重要性，训练中可调整。
"""

from typing import Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaModel(nn.Module):
    """策略注意力加权模型

    输入: [B, num_strategies, window]
      每个策略占1个通道 (连续 score 值)

    输出: (pred: [B], attn_weights: [B, num_strategies])
      pred 为预测收益率, attn_weights 为各策略注意力权重

    strategy_prior: 策略先验权重，直接作为注意力偏置的初始化。
      高先验策略在 softmax 之前有更大的 logit，因此初始注意力权重更高。
      偏置是可学习参数，训练后可以调整。
    """

    def __init__(
        self,
        num_strategies: int,
        window: int = 20,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        strategy_prior: Optional[list] = None,
    ):
        super().__init__()
        self.num_strategies = num_strategies
        self.hidden_dim = hidden_dim

        # 每个策略独立的时序编码器: [1, W] → [hidden_dim]
        self.temporal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            for _ in range(num_strategies)
        ])

        # 策略嵌入: 为每个策略提供可学习的身份向量，打破对称性
        self.strategy_embedding = nn.Embedding(num_strategies, hidden_dim)

        # 策略注意力: 可学习的 query 在 K 个策略上做 attention
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # 先验偏置: 加在 attention logits 上，初始化为 log(prior)
        # 这样初始 softmax(logits + bias) = prior 分布
        if strategy_prior is not None:
            prior = np.array(strategy_prior, dtype=np.float32)
            prior = prior / prior.sum()  # 归一化
            init_bias = np.log(prior + 1e-8).astype(np.float32)
        else:
            init_bias = np.zeros(num_strategies, dtype=np.float32)
        self.attn_prior_bias = nn.Parameter(torch.from_numpy(init_bias))

        # 输出头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: [B, num_strategies, window]

        Returns:
            pred: [B] 预测收益率
            attn_weights: [B, num_strategies] 策略注意力权重
        """
        B = x.size(0)

        # 每个策略独立编码 + 加入策略嵌入
        features = []
        strategy_ids = torch.arange(self.num_strategies, device=x.device)
        for i in range(self.num_strategies):
            # [B, 1, W] → [B, hidden_dim, 1] → [B, hidden_dim]
            feat = self.temporal_encoders[i](x[:, i:i + 1, :]).squeeze(-1)
            # 加入策略嵌入
            feat = feat + self.strategy_embedding(strategy_ids[i])
            features.append(feat)

        # [B, K, hidden_dim]
        features = torch.stack(features, dim=1)

        # 手动 scaled dot-product attention (带先验偏置)
        query = self.attn_query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        keys = self.attn_key(features)               # [B, K, hidden_dim]

        # attention scores: [B, 1, K]
        scale = math.sqrt(self.hidden_dim)
        scores = torch.bmm(query, keys.transpose(1, 2)) / scale

        # 加入先验偏置: [K] → [B, 1, K]
        scores = scores + self.attn_prior_bias.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1).squeeze(1)  # [B, K]
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # [B, hidden_dim]

        # 输出
        pred = self.head(context).squeeze(-1)  # [B]

        return pred, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回预测值 (推理用)"""
        pred, _ = self.forward(x)
        return pred
