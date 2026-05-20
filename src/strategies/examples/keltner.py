"""肯特纳通道策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class KeltnerStrategy(Strategy):
    """肯特纳通道策略：价格触及下轨买入，触及上轨卖出"""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, num_atr: float = 1.5):
        super().__init__()
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.num_atr = num_atr

    def on_init(self, context: Context) -> None:
        self.register_indicator('keltner', ema_period=self.ema_period,
                                atr_period=self.atr_period, num_atr=self.num_atr)

    def on_bar(self, context: Context) -> Optional[Signal]:
        close = context.bar.get('close')
        lower = context.bar.get('kelt_lower')
        upper = context.bar.get('kelt_upper')
        mid = context.bar.get('kelt_mid')

        if any(v is None or pd.isna(v) for v in [close, lower, upper, mid]):
            return None

        half_width = upper - mid
        if half_width < 1e-8:
            return None

        if close <= lower:
            strength = (mid - close) / half_width
            return Signal(type=SignalType.BUY, symbol=context.symbol, strength=min(strength, 1.0))
        if close >= upper:
            strength = (close - mid) / half_width
            return Signal(type=SignalType.SELL, symbol=context.symbol, strength=min(strength, 1.0))

        return None

    def score(self, context: Context) -> float:
        close = context.bar.get('close')
        lower = context.bar.get('kelt_lower')
        upper = context.bar.get('kelt_upper')

        if any(v is None or pd.isna(v) for v in [close, lower, upper]):
            return 0.0

        band_width = upper - lower
        if band_width < 1e-8:
            return 0.0

        pct = (close - lower) / band_width
        return 1.0 - 2.0 * pct
