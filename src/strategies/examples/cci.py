"""CCI 商品通道指标策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class CCIStrategy(Strategy):
    """CCI 策略：CCI 低于 -100 超卖买入，高于 +100 超买卖出"""

    def __init__(self, period: int = 14, oversold: float = -100.0, overbought: float = 100.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init(self, context: Context) -> None:
        self.register_indicator('cci', period=self.period)

    def on_bar(self, context: Context) -> Optional[Signal]:
        cci_col = f'cci_{self.period}'
        cci_val = context.bar.get(cci_col)
        if cci_val is None or pd.isna(cci_val):
            return None

        if cci_val < self.oversold:
            strength = (self.oversold - cci_val) / abs(self.oversold)
            return Signal(type=SignalType.BUY, symbol=context.symbol, strength=min(strength, 1.0))
        if cci_val > self.overbought:
            strength = (cci_val - self.overbought) / self.overbought
            return Signal(type=SignalType.SELL, symbol=context.symbol, strength=min(strength, 1.0))

        return None

    def score(self, context: Context) -> float:
        cci_col = f'cci_{self.period}'
        cci_val = context.bar.get(cci_col)
        if cci_val is None or pd.isna(cci_val):
            return 0.0
        # CCI: 0 为中性，负值看多，正值看空；典型范围 ±200，映射到 [-1, 1]
        return max(-1.0, min(1.0, -cci_val / 100.0))
