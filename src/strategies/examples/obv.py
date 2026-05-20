"""OBV 能量潮策略"""

from typing import Optional

import pandas as pd

from ..base import Strategy, Context, Signal, SignalType


class OBVStrategy(Strategy):
    """OBV 策略：OBV 上穿其 MA 买入，下穿卖出"""

    def __init__(self, ma_period: int = 20):
        super().__init__()
        self.ma_period = ma_period

    def on_init(self, context: Context) -> None:
        self.register_indicator('obv')

    def on_bar(self, context: Context) -> Optional[Signal]:
        obv = context.bar.get('obv')
        if obv is None or pd.isna(obv):
            return None

        if len(context.bars) < self.ma_period:
            return None

        obv_series = context.bars['obv']
        obv_ma = obv_series.rolling(window=self.ma_period).mean().iloc[-1]
        prev_obv_ma = obv_series.rolling(window=self.ma_period).mean().iloc[-2]
        prev_obv = obv_series.iloc[-2]

        if pd.isna(obv_ma) or pd.isna(prev_obv_ma):
            return None

        if prev_obv <= prev_obv_ma and obv > obv_ma:
            return Signal(type=SignalType.BUY, symbol=context.symbol)
        if prev_obv >= prev_obv_ma and obv < obv_ma:
            return Signal(type=SignalType.SELL, symbol=context.symbol)

        return None

    def score(self, context: Context) -> float:
        obv = context.bar.get('obv')
        if obv is None or pd.isna(obv):
            return 0.0

        if len(context.bars) < self.ma_period:
            return 0.0

        obv_series = context.bars['obv']
        obv_ma = obv_series.rolling(window=self.ma_period).mean().iloc[-1]
        if pd.isna(obv_ma) or abs(obv_ma) < 1e-8:
            return 0.0

        gap = (obv - obv_ma) / abs(obv_ma)
        return max(-1.0, min(1.0, gap / 0.1))
