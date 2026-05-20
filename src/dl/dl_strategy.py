"""DL 策略桥接: 将深度学习收益率预测接入回测引擎"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

from strategies.base import Strategy, Context, Signal, SignalType
from .config import DLConfig
from .feature_builder import FeatureBuilder
from .model import PricePredictor

logger = logging.getLogger("dl.strategy")


class DLStrategy(Strategy):
    """基于深度学习收益率预测的交易策略

    在每个 on_bar 中:
      1. 从 ctx.bars 提取最近 window 天的特征
      2. 模型推理得到预测收益率
      3. 根据阈值生成 BUY/SELL/HOLD 信号
      4. 预测幅度映射为 signal.strength (用于仓位计算)
    """

    def __init__(
        self,
        config: DLConfig,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.buy_threshold = buy_threshold or config.buy_threshold
        self.sell_threshold = sell_threshold or config.sell_threshold
        self._builder = FeatureBuilder(config)
        self._model: Optional[PricePredictor] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._feature_cols: Optional[list] = None
        self._loaded = False

    def _ensure_loaded(self):
        """延迟加载模型和 scaler"""
        if self._loaded:
            return

        checkpoint_dir = __import__('pathlib').Path(self.config.checkpoint_dir)

        # 加载 scaler
        scaler_path = checkpoint_dir / "scaler.npz"
        if not scaler_path.exists():
            raise FileNotFoundError(f"未找到 scaler: {scaler_path}，请先训练模型")
        self._builder.load_scaler(str(scaler_path))

        # 推断 num_features
        from pathlib import Path
        meta_path = checkpoint_dir / "train_meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            num_features = meta.get('num_features', 31)
        else:
            num_features = 31

        # 加载模型
        self._model = PricePredictor(
            num_features=num_features,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            window=self.config.window,
        ).to(self._device)

        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"未找到模型: {best_path}，请先训练模型")
        checkpoint = torch.load(best_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

        self._loaded = True
        logger.info(f"DLStrategy 模型加载完成, device={self._device}")

    def on_init(self, context: Context) -> None:
        # DL 策略自行计算指标，不需要引擎预计算
        pass

    def on_bar(self, context: Context) -> Optional[Signal]:
        self._ensure_loaded()

        bars = context.bars
        if len(bars) < self.config.window:
            return None

        # 取最近 window 天的数据
        recent = bars.iloc[-self.config.window:]

        # 构建特征
        try:
            feature_vector = self._build_features_from_bars(recent)
        except Exception as e:
            logger.debug(f"特征构建失败: {e}")
            return None

        if feature_vector is None:
            return None

        # 推理
        with torch.no_grad():
            sample = torch.from_numpy(feature_vector).unsqueeze(0).to(self._device)
            pred_return = self._model(sample).cpu().numpy()[0]

        # 生成信号
        if pred_return > self.buy_threshold:
            strength = min(pred_return / 0.05, 1.0)  # 5% 对应满仓
            return Signal(
                type=SignalType.BUY,
                symbol=context.symbol,
                strength=max(strength, 0.2),
                reason=f"DL预测收益率={pred_return:.4f}",
            )
        elif pred_return < self.sell_threshold:
            strength = min(abs(pred_return) / 0.05, 1.0)
            return Signal(
                type=SignalType.SELL,
                symbol=context.symbol,
                strength=max(strength, 0.2),
                reason=f"DL预测收益率={pred_return:.4f}",
            )

        return Signal(
            type=SignalType.HOLD,
            symbol=context.symbol,
            reason=f"DL预测收益率={pred_return:.4f} (在阈值内)",
        )

    def _build_features_from_bars(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """从 bars DataFrame 构建 [num_features, window] 的特征矩阵"""
        # 计算指标
        df_with_ind = self._compute_indicators(df)

        # 选取特征列
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if self._feature_cols is not None:
            feature_cols.extend(self._feature_cols)
        else:
            original_cols = {'open', 'high', 'low', 'close', 'volume', 'amount'}
            ind_cols = [c for c in df_with_ind.columns if c not in original_cols]
            feature_cols.extend(ind_cols)
            self._feature_cols = ind_cols

        # 对齐列
        available = [c for c in feature_cols if c in df_with_ind.columns]
        if len(available) < len(feature_cols):
            logger.debug(f"特征列缺失: 期望 {len(feature_cols)}, 实际 {len(available)}")
            return None

        values = df_with_ind[available].fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(np.float32)

        # [window, num_features] → [num_features, window]
        feature_matrix = values.T

        # 标准化
        feature_matrix = self._builder.normalize_features(
            feature_matrix[np.newaxis], fit=False
        )[0]

        return feature_matrix

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        from analysis.indicator_set import IndicatorSet

        iset = IndicatorSet()
        for name, params in self.config.get_indicators():
            iset.add(name, **params)
        return iset.compute(df)
