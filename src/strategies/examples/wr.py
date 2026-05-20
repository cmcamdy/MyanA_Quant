"""威廉指标策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class WRStrategy(Strategy):
    """威廉 %R 策略：低于 -80 超卖买入，高于 -20 超买卖出"""

    def __init__(self, period: int = 14, oversold: float = -80.0, overbought: float = -20.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init(self, context: Context) -> None:
        self.register_indicator('wr', period=self.period)

    def on_bar(self, context: Context) -> Optional[Signal]:
        wr_col = f'wr_{self.period}'
        wr_val = context.bar.get(wr_col)
        if wr_val is None or pd.isna(wr_val):
            return None

        if wr_val < self.oversold:
            strength = (self.oversold - wr_val) / abs(self.oversold)
            return Signal(type=SignalType.BUY, symbol=context.symbol, strength=min(strength, 1.0))
        if wr_val > self.overbought:
            strength = (wr_val - self.overbought) / abs(self.overbought)
            return Signal(type=SignalType.SELL, symbol=context.symbol, strength=min(strength, 1.0))

        return None

    def score(self, context: Context) -> float:
        wr_col = f'wr_{self.period}'
        wr_val = context.bar.get(wr_col)
        if wr_val is None or pd.isna(wr_val):
            return 0.0
        # WR: -50 为中性，-100 → +1 (超卖看多)，0 → -1 (超买卖空)
        return (-50.0 - wr_val) / 50.0
