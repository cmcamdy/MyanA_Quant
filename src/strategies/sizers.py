"""仓位管理器"""

from typing import Protocol

from .base import Signal, SignalType, Portfolio


class PositionSizer(Protocol):
    def compute_quantity(
        self, signal: Signal, portfolio: Portfolio, current_price: float
    ) -> float:
        ...


class FixedSizer:
    """固定比例下单"""

    def __init__(self, percent: float = 1.0):
        self.percent = percent

    def compute_quantity(
        self, signal: Signal, portfolio: Portfolio, current_price: float
    ) -> float:
        if signal.type == SignalType.BUY:
            available = portfolio.cash * self.percent * signal.strength
            return available / current_price if current_price > 0 else 0.0
        elif signal.type == SignalType.SELL:
            pos = portfolio.get_position(signal.symbol)
            return pos.quantity * self.percent * signal.strength
        return 0.0


class AllInSizer(FixedSizer):
    """全仓交易"""

    def __init__(self):
        super().__init__(percent=1.0)
