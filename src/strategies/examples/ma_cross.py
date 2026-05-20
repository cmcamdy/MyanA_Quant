"""双均线交叉策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class MACrossStrategy(Strategy):
    """双均线交叉策略：金叉买入，死叉卖出"""

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_init(self, context: Context) -> None:
        self.register_indicator('ma', period=self.fast_period)
        self.register_indicator('ma', period=self.slow_period)

    def on_bar(self, context: Context) -> Optional[Signal]:
        fast_col = f'ma_{self.fast_period}'
        slow_col = f'ma_{self.slow_period}'

        if len(context.bars) < 2:
            return None

        curr_fast = context.bars[fast_col].iloc[-1]
        curr_slow = context.bars[slow_col].iloc[-1]
        prev_fast = context.bars[fast_col].iloc[-2]
        prev_slow = context.bars[slow_col].iloc[-2]

        if pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return None

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return Signal(type=SignalType.BUY, symbol=context.symbol)
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return Signal(type=SignalType.SELL, symbol=context.symbol)

        return None

    def score(self, context: Context) -> float:
        """MA 偏离度: (fast - slow) / close, 标准化到 [-1, 1]"""
        fast_col = f'ma_{self.fast_period}'
        slow_col = f'ma_{self.slow_period}'

        fast = context.bar.get(fast_col)
        slow = context.bar.get(slow_col)
        close = context.bar.get('close')

        if any(v is None or pd.isna(v) for v in [fast, slow, close]) or close < 1e-8:
            return 0.0

        gap = (fast - slow) / close
        # 典型偏离在 ±0.05 以内，映射到 [-1, 1]
        return max(-1.0, min(1.0, gap / 0.03))
