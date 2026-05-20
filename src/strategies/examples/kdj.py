"""KDJ 随机指标策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class KDJStrategy(Strategy):
    """KDJ 策略：K 上穿 D 且 J < 超卖阈值买入，K 下穿 D 且 J > 超买阈值卖出"""

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3,
                 oversold: float = 20.0, overbought: float = 80.0):
        super().__init__()
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.oversold = oversold
        self.overbought = overbought

    def on_init(self, context: Context) -> None:
        self.register_indicator('kdj', n=self.n, m1=self.m1, m2=self.m2)

    def on_bar(self, context: Context) -> Optional[Signal]:
        if len(context.bars) < 2:
            return None

        curr_k = context.bars['k'].iloc[-1]
        curr_d = context.bars['d'].iloc[-1]
        curr_j = context.bars['j'].iloc[-1]
        prev_k = context.bars['k'].iloc[-2]
        prev_d = context.bars['d'].iloc[-2]

        if pd.isna(curr_k) or pd.isna(curr_d) or pd.isna(curr_j) or pd.isna(prev_k) or pd.isna(prev_d):
            return None

        if prev_k <= prev_d and curr_k > curr_d and curr_j < self.oversold:
            strength = (self.oversold - curr_j) / self.oversold
            return Signal(type=SignalType.BUY, symbol=context.symbol, strength=min(strength, 1.0))
        if prev_k >= prev_d and curr_k < curr_d and curr_j > self.overbought:
            strength = (curr_j - self.overbought) / (100.0 - self.overbought)
            return Signal(type=SignalType.SELL, symbol=context.symbol, strength=min(strength, 1.0))

        return None

    def score(self, context: Context) -> float:
        j = context.bar.get('j')
        if j is None or pd.isna(j):
            return 0.0
        return (50.0 - j) / 50.0
