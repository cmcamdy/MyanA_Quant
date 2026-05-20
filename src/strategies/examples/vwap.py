"""VWAP 成交量加权平均价策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class VWAPStrategy(Strategy):
    """VWAP 策略：价格低于 VWAP 买入（均值回归），高于 VWAP 卖出"""

    def on_init(self, context: Context) -> None:
        self.register_indicator('vwap')

    def on_bar(self, context: Context) -> Optional[Signal]:
        close = context.bar.get('close')
        vwap_val = context.bar.get('vwap')

        if any(v is None or pd.isna(v) for v in [close, vwap_val]) or vwap_val < 1e-8:
            return None

        gap_pct = (close - vwap_val) / vwap_val

        if gap_pct < -0.01:
            strength = min(-gap_pct / 0.03, 1.0)
            return Signal(type=SignalType.BUY, symbol=context.symbol, strength=strength)
        if gap_pct > 0.01:
            strength = min(gap_pct / 0.03, 1.0)
            return Signal(type=SignalType.SELL, symbol=context.symbol, strength=strength)

        return None

    def score(self, context: Context) -> float:
        close = context.bar.get('close')
        vwap_val = context.bar.get('vwap')

        if any(v is None or pd.isna(v) for v in [close, vwap_val]) or vwap_val < 1e-8:
            return 0.0

        gap = (close - vwap_val) / vwap_val
        return max(-1.0, min(1.0, -gap / 0.02))
