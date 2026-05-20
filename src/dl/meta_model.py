"""Meta Model: 策略观点时序编码 + 注意力加权

架构:
  对每个子策略的连续 score 序列 [1, W]，用独立 Conv1D 编码时序特征，
  加入可学习的策略嵌入向量区分不同策略，
  再通过 Multi-Head Attention 学习策略权重，加权组合后输出预测收益率。

  注意力权重可直接解读为各策略的相对重要性。
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MetaModel(nn.Module):
    """策略注意力加权模型

    输入: [B, num_strategies, window]
      每个策略占1个通道 (连续 score 值)

    输出: (pred: [B], attn_weights: [B, num_strategies])
      pred 为预测收益率, attn_weights 为各策略注意力权重

    strategy_prior: 可选的策略先验权重，用于初始化策略嵌入幅度，
      让高先验策略的嵌入更大，自然吸引更多注意力。
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
        if strategy_prior is not None:
            # 用先验权重缩放嵌入初始化幅度: 高先验 → 更大嵌入 → 更高注意力
            prior = np.array(strategy_prior, dtype=np.float32)
            prior = prior / prior.sum()  # 归一化
            scale = (prior * num_strategies) ** 0.5  # 均值=1, 高先验>1, 低先验<1
            with torch.no_grad():
                for i in range(num_strategies):
                    self.strategy_embedding.weight[i] *= scale[i]

        # 策略注意力: 可学习的 query 在 K 个策略上做 attention
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=1,
            dropout=dropout, batch_first=True,
        )

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

        # Attention: query attends over K strategy features
        query = self.attn_query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        context, attn_weights = self.attn(query, features, features)
        context = context.squeeze(1)  # [B, hidden_dim]
        attn_weights = attn_weights.squeeze(1)  # [B, K]

        # 输出
        pred = self.head(context).squeeze(-1)  # [B]

        return pred, attn_weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回预测值 (推理用)"""
        pred, _ = self.forward(x)
        return pred
