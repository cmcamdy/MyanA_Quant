"""SAR 抛物线策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class SARStrategy(Strategy):
    """SAR 抛物线策略：趋势翻转时产生信号

    sar_trend 从负转正 → BUY (翻多)
    sar_trend 从正转负 → SELL (翻空)
    趋势持续中 → HOLD
    """

    def __init__(self):
        super().__init__()
        self._prev_trend: Optional[float] = None

    def on_init(self, context: Context) -> None:
        self.register_indicator('sar')

    def on_bar(self, context: Context) -> Optional[Signal]:
        sar_trend = context.bar.get('sar_trend')
        if sar_trend is None or pd.isna(sar_trend):
            self._prev_trend = None
            return None

        if self._prev_trend is None:
            self._prev_trend = sar_trend
            return None

        signal = None
        if self._prev_trend <= 0 and sar_trend > 0:
            signal = Signal(type=SignalType.BUY, symbol=context.symbol)
        elif self._prev_trend > 0 and sar_trend <= 0:
            signal = Signal(type=SignalType.SELL, symbol=context.symbol)

        self._prev_trend = sar_trend
        return signal

    def score(self, context: Context) -> float:
        """SAR 趋势方向 + 价格与 SAR 距离"""
        sar_val = context.bar.get('sar_value')
        sar_trend = context.bar.get('sar_trend')
        close = context.bar.get('close')

        if any(v is None or pd.isna(v) for v in [sar_val, sar_trend, close]) or close < 1e-8:
            return 0.0

        # 趋势方向: 1 或 -1
        direction = 1.0 if sar_trend > 0 else -1.0

        # 价格与 SAR 的偏离程度，归一化
        distance = (close - sar_val) / close
        distance = max(-0.1, min(0.1, distance))

        # 综合分数: 方向 * (0.5 + 距离权重)
        return direction * (0.5 + abs(distance) * 5.0)
