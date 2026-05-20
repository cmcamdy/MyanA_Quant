"""MFI 资金流量指标策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class MFIStrategy(Strategy):
    """MFI 策略：MFI 低于 20 超卖买入，高于 80 超买卖出"""

    def __init__(self, period: int = 14, oversold: float = 20.0, overbought: float = 80.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init(self, context: Context) -> None:
        self.register_indicator('mfi', period=self.period)

    def on_bar(self, context: Context) -> Optional[Signal]:
        mfi_col = f'mfi_{self.period}'
        mfi_val = context.bar.get(mfi_col)
        if mfi_val is None or pd.isna(mfi_val):
            return None

        if mfi_val < self.oversold:
            strength = (self.oversold - mfi_val) / self.oversold
            return Signal(type=SignalType.BUY, symbol=context.symbol, strength=min(strength, 1.0))
        if mfi_val > self.overbought:
            strength = (mfi_val - self.overbought) / (100.0 - self.overbought)
            return Signal(type=SignalType.SELL, symbol=context.symbol, strength=min(strength, 1.0))

        return None

    def score(self, context: Context) -> float:
        mfi_col = f'mfi_{self.period}'
        mfi_val = context.bar.get(mfi_col)
        if mfi_val is None or pd.isna(mfi_val):
            return 0.0
        return (50.0 - mfi_val) / 50.0
