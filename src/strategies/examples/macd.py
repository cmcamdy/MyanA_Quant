"""MACD 策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class MACDStrategy(Strategy):
    """MACD 策略：DIF 上穿 DEA 买入，下穿卖出"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def on_init(self, context: Context) -> None:
        self.register_indicator('macd', fast_period=self.fast_period,
                                slow_period=self.slow_period, signal_period=self.signal_period)

    def on_bar(self, context: Context) -> Optional[Signal]:
        if len(context.bars) < 2:
            return None

        curr_dif = context.bars['macd_dif'].iloc[-1]
        curr_dea = context.bars['macd_dea'].iloc[-1]
        prev_dif = context.bars['macd_dif'].iloc[-2]
        prev_dea = context.bars['macd_dea'].iloc[-2]

        if pd.isna(curr_dif) or pd.isna(curr_dea) or pd.isna(prev_dif) or pd.isna(prev_dea):
            return None

        if prev_dif <= prev_dea and curr_dif > curr_dea:
            return Signal(type=SignalType.BUY, symbol=context.symbol)
        if prev_dif >= prev_dea and curr_dif < curr_dea:
            return Signal(type=SignalType.SELL, symbol=context.symbol)

        return None

    def score(self, context: Context) -> float:
        dif = context.bar.get('macd_dif')
        dea = context.bar.get('macd_dea')
        close = context.bar.get('close')

        if any(v is None or pd.isna(v) for v in [dif, dea, close]) or close < 1e-8:
            return 0.0

        gap = (dif - dea) / close
        return max(-1.0, min(1.0, gap / 0.03))
