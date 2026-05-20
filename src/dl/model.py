"""价格涨跌预测网络: Transformer Decoder + 因果时序 Mask

输入设计
--------
对于第 t 天，取前 window 天(默认120)的行情数据，每根 bar 含 31 个特征，
组成 [31, window] 的矩阵，转置为 [window, 31] 的时序序列输入 Transformer。

特征构成 (31列):
  原始OHLCV+A (6列):  open, high, low, close, volume, amount
  技术指标 (25列):     ma_5, ma_10, ma_20, ma_60, ema_12, ema_26,
                       macd_dif, macd_dea, macd_hist, rsi_6, rsi_14,
                       k, d, j, boll_mid, boll_upper, boll_lower,
                       atr_14, obv, mfi_14, wr_14, cci_14, vwap,
                       sar_value, sar_trend

  序列示意 (转置后):
             feature0 feature1 ... feature30
    day1      10.1     10.5     ... 1.0
    day2      10.3     10.6     ... 1.0
    ...       ...      ...      ... ...
    day120    12.5     12.8     ... -1.0

  经 z-score 标准化后，通过 Linear 投影到 d_model 维，
  加入位置编码，输入 Transformer Decoder。

  因果 Mask: 第 t 天只能 attend 到 ≤t 的位置，防止看到未来数据。

输出设计
--------
6 分类涨跌预测，输出 logits 经 softmax 得到概率分布:

  [0] 大跌 (<-5%)
  [1] 中跌 (-5% ~ -2%)
  [2] 小跌 (-2% ~ 0%)
  [3] 小涨 (0% ~ 2%)
  [4] 中涨 (2% ~ 5%)
  [5] 大涨 (>5%)

  取 Transformer 最后一个时间步的输出，经 LayerNorm + Dropout + Linear 映射到 6 维 logits。

  示例输出:
    logits = model(x)                    # [B, 6]
    probs  = softmax(logits, dim=1)      # [B, 6]
    pred   = probs.argmax(dim=1)         # [B] 预测分类
    # 如 probs = [0.02, 0.11, 0.34, 0.38, 0.13, 0.02] → pred=3 (小涨)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码 (固定，不可学习)"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PricePredictor(nn.Module):
    """Transformer Decoder 分类网络

    结构:
      [B, num_features, window] → transpose → [B, window, num_features]
      Linear(num_features, d_model) → PositionalEncoding
      TransformerDecoderLayer × num_layers (带因果 mask)
      取最后时间步 → LayerNorm → Dropout → Linear(d_model, num_classes)

    Args:
        num_features: 每个时间步的特征数 (31)
        d_model: Transformer 内部维度
        nhead: 多头注意力头数
        num_layers: Transformer Decoder 层数
        dim_feedforward: FFN 中间层维度
        num_classes: 分类数
        dropout: Dropout 概率
        window: 时间步数 (用于预生成因果 mask)
    """

    def __init__(
        self,
        num_features: int = 31,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 6,
        dropout: float = 0.3,
        window: int = 120,
    ):
        super().__init__()
        self.d_model = d_model
        self.window = window

        # 特征投影
        self.input_proj = nn.Linear(num_features, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=window + 1, dropout=dropout * 0.3)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

        # 预生成因果 mask
        self._causal_mask: torch.Tensor | None = None

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果 mask: 上三角为 True (被遮蔽)"""
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: [B, num_features, window]

        Returns:
            logits: [B, num_classes]
        """
        # [B, num_features, window] → [B, window, num_features]
        x = x.transpose(1, 2)

        # 投影 + 位置编码
        x = self.input_proj(x)  # [B, window, d_model]
        x = self.pos_encoder(x)

        # 因果 mask
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # Transformer Decoder (memory=x, self-attention with causal mask)
        out = self.decoder(tgt=x, memory=x, tgt_mask=causal_mask)  # [B, window, d_model]

        # 取最后时间步
        out = out[:, -1, :]  # [B, d_model]

        # 输出
        out = self.norm(out)
        out = self.drop(out)
        logits = self.fc_out(out)  # [B, num_classes]

        return logits
