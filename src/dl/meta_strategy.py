"""MetaStrategy: 学习型组合策略，桥接回测引擎

在每个 on_bar 中:
  1. 运行每个子策略的 score() 获取连续观点分数
  2. 维护滚动窗口的分数历史
  3. 喂入 MetaModel 推理 → 得到预测收益率 + 策略权重
  4. 按阈值生成 BUY/SELL/HOLD 信号
"""

import logging
from collections import deque
from typing import List, Optional

import numpy as np

from strategies.base import Strategy, Context, Signal, SignalType
from .meta_config import MetaConfig
from .meta_model import MetaModel
from .meta_signal_collector import MetaSignalCollector

logger = logging.getLogger("dl.meta_strategy")


class MetaStrategy(Strategy):
    """学习型组合策略

    用注意力机制学习子策略权重，动态决定看多/看空倾向。

    信号模式:
      - fixed: 绝对阈值，pred > buy_threshold → BUY, pred < sell_threshold → SELL
      - adaptive: 自适应百分位，pred 高于历史 buy_percentile 分位 → BUY，
        低于 sell_percentile 分位 → SELL
    """

    def __init__(
        self,
        config: MetaConfig,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.buy_threshold = buy_threshold or config.buy_threshold
        self.sell_threshold = sell_threshold or config.sell_threshold
        self.strategies = config.strategies
        self._model: Optional[MetaModel] = None
        self._collector: Optional[MetaSignalCollector] = None
        self._device = None
        self._score_history: Optional[deque] = None
        self._pred_history: Optional[deque] = None
        self._bar_count = 0
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from pathlib import Path

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_dir = Path(self.config.checkpoint_dir)

        # 加载 scaler
        scaler_path = checkpoint_dir / "meta_scaler.npz"
        if not scaler_path.exists():
            raise FileNotFoundError(f"未找到 scaler: {scaler_path}，请先训练模型")
        self._collector = MetaSignalCollector(self.config)
        self._collector.load_scaler(str(scaler_path))

        # 加载模型 (通道数 = 策略数 + 3个市场特征)
        num_channels = self.config.num_strategies + 3
        prior = self.config.strategy_prior
        if prior is not None:
            prior = list(prior) + [1.0, 1.0, 1.0]  # 市场特征均匀先验
        self._model = MetaModel(
            num_strategies=num_channels,
            window=self.config.window,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            strategy_prior=prior,
        ).to(self._device)

        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"未找到模型: {best_path}，请先训练模型")
        import torch as th
        checkpoint = th.load(best_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

        # 初始化分数历史
        self._score_history = deque(maxlen=self.config.window)
        self._pred_history = deque(maxlen=self.config.adaptive_window)

        self._loaded = True
        logger.info(f"MetaStrategy 加载完成, {self.config.num_strategies} 个子策略")

    def on_init(self, context: Context) -> None:
        # 注册所有子策略的指标依赖
        for strat in self.strategies:
            strat.on_init(context)
            for spec in strat.indicator_specs:
                self.register_indicator(spec['name'], **{k: v for k, v in spec.items() if k != 'name'})

    def on_bar(self, context: Context) -> Optional[Signal]:
        self._ensure_loaded()

        # 运行每个子策略的 score()
        step_scores = [strat.score(context) for strat in self.strategies]

        # 市场上下文特征
        close = context.bar.get('close')
        volume = context.bar.get('volume')
        if close is not None and len(context.bars) >= 2:
            prev_close = context.bars['close'].iloc[-2]
            ret_1d = (close - prev_close) / prev_close if prev_close > 0 else 0.0
        else:
            ret_1d = 0.0

        if close is not None and len(context.bars) >= 6:
            close_5d = context.bars['close'].iloc[-6]
            ret_5d = (close - close_5d) / close_5d if close_5d > 0 else 0.0
        else:
            ret_5d = 0.0

        if volume is not None and len(context.bars) >= 2:
            prev_vol = context.bars['volume'].iloc[-2]
            vol_chg = (volume - prev_vol) / prev_vol if prev_vol > 0 else 0.0
            vol_chg = max(-1.0, min(5.0, vol_chg))
        else:
            vol_chg = 0.0

        # 记录到历史 (策略分数 + 市场特征)
        step_row = step_scores + [ret_1d, ret_5d, vol_chg]
        self._score_history.append(step_row)

        # 窗口不足则不产生信号
        if len(self._score_history) < self.config.window:
            return None

        # 构建特征: [K+3, window]
        K = len(self.strategies)
        total_channels = K + 3
        window = self.config.window
        features = np.zeros((total_channels, window), dtype=np.float32)

        for t, row in enumerate(self._score_history):
            for j, val in enumerate(row):
                features[j, t] = val

        # 标准化
        features = self._collector.normalize_features(features[np.newaxis], fit=False)[0]

        # 推理
        import torch
        with torch.no_grad():
            sample = torch.from_numpy(features).unsqueeze(0).to(self._device)
            pred, attn_weights = self._model(sample)
            pred_return = self._collector.denormalize_labels(pred.cpu().numpy())[0]
            weights = attn_weights.cpu().numpy()[0]

        # 诊断日志: 前几次预测详细输出，之后每50个bar输出一次
        self._bar_count += 1
        if self._bar_count <= 5 or self._bar_count % 50 == 0:
            logger.info(
                f"Bar#{self._bar_count} Meta预测={pred_return:.6f}, "
                f"模式={self.config.signal_mode}, "
                f"权重={np.round(weights, 3).tolist()}"
            )

        # 记录预测值用于自适应信号
        self._pred_history.append(pred_return)

        # 生成信号
        if self.config.signal_mode == "adaptive":
            return self._adaptive_signal(pred_return, weights, context)
        else:
            return self._fixed_signal(pred_return, weights, context)

    def _adaptive_signal(self, pred_return: float, weights: np.ndarray, context: Context) -> Signal:
        """自适应百分位信号: 预测值在历史中的相对位置决定信号"""
        if len(self._pred_history) < 20:
            return Signal(
                type=SignalType.HOLD, symbol=context.symbol,
                reason=f"Meta预测={pred_return:.4f}, 历史不足",
            )

        hist = np.array(self._pred_history)
        buy_line = np.percentile(hist, self.config.buy_percentile * 100)
        sell_line = np.percentile(hist, self.config.sell_percentile * 100)

        if pred_return >= buy_line:
            # 在历史中排名越高，信号越强
            rank = np.searchsorted(np.sort(hist), pred_return) / len(hist)
            strength = max(rank, 0.2)
            return Signal(
                type=SignalType.BUY, symbol=context.symbol,
                strength=strength,
                reason=f"Meta预测={pred_return:.4f}, 高于{self.config.buy_percentile:.0%}分位({buy_line:.4f})",
            )
        elif pred_return <= sell_line:
            rank = 1.0 - np.searchsorted(np.sort(hist), pred_return) / len(hist)
            strength = max(rank, 0.2)
            return Signal(
                type=SignalType.SELL, symbol=context.symbol,
                strength=strength,
                reason=f"Meta预测={pred_return:.4f}, 低于{self.config.sell_percentile:.0%}分位({sell_line:.4f})",
            )

        return Signal(
            type=SignalType.HOLD, symbol=context.symbol,
            reason=f"Meta预测={pred_return:.4f}, 分位区间[{sell_line:.4f}, {buy_line:.4f}]",
        )

    def _fixed_signal(self, pred_return: float, weights: np.ndarray, context: Context) -> Signal:
        """固定阈值信号"""
        if pred_return > self.buy_threshold:
            strength = min(pred_return / 0.05, 1.0)
            return Signal(
                type=SignalType.BUY, symbol=context.symbol,
                strength=max(strength, 0.2),
                reason=f"Meta预测={pred_return:.4f}, 权重={np.round(weights, 3).tolist()}",
            )
        elif pred_return < self.sell_threshold:
            strength = min(abs(pred_return) / 0.05, 1.0)
            return Signal(
                type=SignalType.SELL, symbol=context.symbol,
                strength=max(strength, 0.2),
                reason=f"Meta预测={pred_return:.4f}, 权重={np.round(weights, 3).tolist()}",
            )

        return Signal(
            type=SignalType.HOLD, symbol=context.symbol,
            reason=f"Meta预测={pred_return:.4f}",
        )
