"""RSI 超买超卖策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class RSIStrategy(Strategy):
    """RSI 超买超卖策略：RSI < 超卖阈值买入，RSI > 超买阈值卖出"""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init(self, context: Context) -> None:
        self.register_indicator('rsi', period=self.period)

    def on_bar(self, context: Context) -> Optional[Signal]:
        rsi_col = f'rsi_{self.period}'
        rsi = context.bar.get(rsi_col)
        if rsi is None or pd.isna(rsi):
            return None

        if rsi < self.oversold:
            strength = (self.oversold - rsi) / self.oversold
            return Signal(
                type=SignalType.BUY, symbol=context.symbol,
                strength=min(strength, 1.0),
            )
        elif rsi > self.overbought:
            strength = (rsi - self.overbought) / (100.0 - self.overbought)
            return Signal(
                type=SignalType.SELL, symbol=context.symbol,
                strength=min(strength, 1.0),
            )
        return None

    def score(self, context: Context) -> float:
        """RSI 连续分数: 以50为中轴，越低越看多，越高越看空"""
        rsi_col = f'rsi_{self.period}'
        rsi = context.bar.get(rsi_col)
        if rsi is None or pd.isna(rsi):
            return 0.0
        # RSI 50 → 0, 0 → +1 (极度看多), 100 → -1 (极度看空)
        return (50.0 - rsi) / 50.0
