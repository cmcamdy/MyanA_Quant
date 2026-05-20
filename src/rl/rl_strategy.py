"""RL 截面选股策略

继承 MultiStrategy, 通过 ONNX 推理实现截面选股,
与 PortfolioEngine 配合进行多股组合回测。

核心设计:
- 只在再平衡周期 (周/月/季) 生成信号, 非 per-bar
- top-k 模式: 分数最高的 k 只股票 BUY, 已持仓但不在 top-k 的 SELL
- quantile 模式: 分数超过 80 分位 BUY, 低于 20 分位 SELL
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from strategies.base import Signal, SignalType
from strategies.portfolio_engine import MultiStrategy, PortfolioContext
from .config import RLConfig
from .inference import RLInference

logger = logging.getLogger(__name__)


class RLStrategy(MultiStrategy):
    """RL 截面选股策略, 使用 ONNX 模型推理"""

    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        self._inference: Optional[RLInference] = None
        self._symbol_index: Optional[Dict[str, int]] = None
        self._last_rebalance: Optional[pd.Timestamp] = None

    def _ensure_loaded(self):
        """懒加载 ONNX 模型"""
        if self._inference is not None:
            return

        if self.config.onnx_model_path is None:
            raise ValueError("onnx_model_path not set in RLConfig")

        self._inference = RLInference(self.config.onnx_model_path)
        logger.info("RL model loaded from: %s", self.config.onnx_model_path)

    def on_init_multi(self, context: PortfolioContext) -> None:
        """初始化: 加载 ONNX 模型, 构建股票索引"""
        self._ensure_loaded()

        symbols = list(context.historical.keys())
        self._symbol_index = {sym: i for i, sym in enumerate(symbols)}
        logger.info("RLStrategy initialized with %d symbols", len(symbols))

    def on_bar_multi(self, context: PortfolioContext) -> List[Signal]:
        """每 bar 调用: 仅在再平衡日生成信号"""
        if not self._should_rebalance(context.current_time):
            return []

        # 构建截面观测
        obs = self._build_observation(context)
        if obs is None:
            return []

        # ONNX 推理
        try:
            scores = self._inference.predict(obs)
        except Exception as e:
            logger.warning("Inference failed: %s", e)
            return []

        # 将分数映射到股票代码
        sym_scores = self._map_scores(scores)

        # 生成信号
        signals = self._generate_signals(sym_scores, context)

        self._last_rebalance = context.current_time
        return signals

    def _should_rebalance(self, current_time: pd.Timestamp) -> bool:
        """判断是否到达再平衡时间点"""
        if self._last_rebalance is None:
            return True

        freq_map = {"W": "W", "M": "MS", "Q": "QS"}
        freq = freq_map.get(self.config.rebalance_freq, "MS")
        periods = pd.date_range(
            start=self._last_rebalance, end=current_time, freq=freq
        )
        return len(periods) > 1

    def _build_observation(self, context: PortfolioContext) -> Optional[np.ndarray]:
        """从 PortfolioContext 构建截面观测张量

        使用 ctx.historical 中每只股票最近的 close/volume 等数据,
        组装为 ONNX 模型期望的输入格式。
        """
        if not context.historical:
            return None

        symbols = list(context.historical.keys())
        n_stocks = len(symbols)

        # 使用 close 价格作为基础特征 (可扩展)
        features_per_stock = 2  # close_pct, vol_pct
        obs = np.zeros((1, n_stocks * features_per_stock), dtype=np.float32)

        for i, sym in enumerate(symbols):
            hist = context.historical[sym]
            if len(hist) < 2:
                continue

            latest = hist.iloc[-1]
            prev = hist.iloc[-2]

            # 价格变化率
            if prev["close"] > 0:
                obs[0, i * features_per_stock] = (
                    latest["close"] / prev["close"] - 1.0
                )

            # 成交量变化率
            vol_col = "vol" if "vol" in hist.columns else "volume"
            if vol_col in hist.columns and prev[vol_col] > 0:
                obs[0, i * features_per_stock + 1] = (
                    latest[vol_col] / prev[vol_col] - 1.0
                )

        return obs

    def _map_scores(self, raw_scores: np.ndarray) -> Dict[str, float]:
        """将模型输出的分数映射到股票代码"""
        flat = raw_scores.flatten()
        if self._symbol_index is None:
            return {}

        index_to_sym = {v: k for k, v in self._symbol_index.items()}
        result = {}
        for idx, score in enumerate(flat):
            sym = index_to_sym.get(idx)
            if sym is not None:
                result[sym] = float(score)
        return result

    def _generate_signals(
        self,
        scores: Dict[str, float],
        context: PortfolioContext,
    ) -> List[Signal]:
        """根据分数和信号模式生成交易信号"""
        if not scores:
            return []

        if self.config.signal_mode == "top_k":
            return self._signals_top_k(scores, context)
        elif self.config.signal_mode == "quantile":
            return self._signals_quantile(scores, context)
        else:
            return self._signals_top_k(scores, context)

    def _signals_top_k(
        self,
        scores: Dict[str, float],
        context: PortfolioContext,
    ) -> List[Signal]:
        """top-k 模式: 买入分数最高的 k 只, 卖出不在 top-k 的持仓"""
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_symbols: Set[str] = {sym for sym, _ in ranked[: self.config.inference_top_k]}

        held: Set[str] = {
            sym
            for sym, pos in context.portfolio.positions.items()
            if not pos.is_empty
        }

        signals: List[Signal] = []

        # 买入 top-k 中未持仓的
        for sym in top_symbols:
            if context.portfolio.get_position(sym).is_empty:
                strength = min(max(scores[sym], 0.0), 1.0) if scores[sym] > 0 else 0.5
                signals.append(
                    Signal(type=SignalType.BUY, symbol=sym, strength=strength)
                )

        # 卖出不在 top-k 的持仓
        for sym in held - top_symbols:
            signals.append(Signal(type=SignalType.SELL, symbol=sym))

        return signals

    def _signals_quantile(
        self,
        scores: Dict[str, float],
        context: PortfolioContext,
    ) -> List[Signal]:
        """分位数模式: 高分位买入, 低分位卖出"""
        if not scores:
            return []

        score_values = list(scores.values())
        buy_threshold = np.percentile(score_values, self.config.score_quantile_buy * 100)
        sell_threshold = np.percentile(score_values, self.config.score_quantile_sell * 100)

        held: Set[str] = {
            sym
            for sym, pos in context.portfolio.positions.items()
            if not pos.is_empty
        }

        signals: List[Signal] = []

        for sym, score in scores.items():
            if score >= buy_threshold and context.portfolio.get_position(sym).is_empty:
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=sym,
                        strength=min(max(score / max(score_values), 0.0), 1.0),
                    )
                )
            elif score <= sell_threshold and sym in held:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))

        return signals
