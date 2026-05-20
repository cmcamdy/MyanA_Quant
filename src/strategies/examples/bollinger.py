"""布林带策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class BollingerStrategy(Strategy):
    """布林带策略：价格触及下轨买入，触及上轨卖出"""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__()
        self.period = period
        self.num_std = num_std

    def on_init(self, context: Context) -> None:
        self.register_indicator('bollinger', period=self.period, num_std=self.num_std)

    def on_bar(self, context: Context) -> Optional[Signal]:
        close = context.bar.get('close')
        lower = context.bar.get('boll_lower')
        upper = context.bar.get('boll_upper')
        mid = context.bar.get('boll_mid')

        if any(v is None or pd.isna(v) for v in [close, lower, upper, mid]):
            return None

        band_width = upper - mid
        if band_width < 1e-8:
            return None

        if close <= lower:
            strength = (mid - close) / band_width
            return Signal(
                type=SignalType.BUY, symbol=context.symbol,
                strength=min(strength, 1.0),
            )
        elif close >= upper:
            strength = (close - mid) / band_width
            return Signal(
                type=SignalType.SELL, symbol=context.symbol,
                strength=min(strength, 1.0),
            )
        return None

    def score(self, context: Context) -> float:
        """布林带位置: 价格在下轨附近 → +1 (看多反弹), 上轨附近 → -1 (看空回落)"""
        close = context.bar.get('close')
        lower = context.bar.get('boll_lower')
        upper = context.bar.get('boll_upper')
        mid = context.bar.get('boll_mid')

        if any(v is None or pd.isna(v) for v in [close, lower, upper, mid]):
            return 0.0

        band_width = upper - lower
        if band_width < 1e-8:
            return 0.0

        # %B 指标: 0 = 下轨, 0.5 = 中轨, 1 = 上轨
        pct_b = (close - lower) / band_width
        # 映射: 0 → +1 (超卖看多), 1 → -1 (超买卖空), 0.5 → 0
        return 1.0 - 2.0 * pct_b
