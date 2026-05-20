"""组合策略：合并多个子策略信号"""

from typing import List, Optional

from .base import Strategy, Context, Signal, SignalType


class CompositeStrategy(Strategy):
    """合并多个子策略信号

    Args:
        strategies: 子策略列表
        mode: 合并模式
            'unanimous' - 所有子策略一致才发信号
            'any' - 任一子策略发信号即触发
            'majority' - 多数子策略一致才发信号
    """

    def __init__(self, strategies: List[Strategy], mode: str = 'unanimous'):
        super().__init__()
        self._strategies = strategies
        self._mode = mode

    def on_init(self, context: Context) -> None:
        # 先初始化子策略，让它们注册指标
        for s in self._strategies:
            s.on_init(context)
        # 再收集所有子策略声明的指标
        for s in self._strategies:
            for spec in s.indicator_specs:
                self.register_indicator(**spec)

    def on_bar(self, context: Context) -> Optional[Signal]:
        signals = []
        for s in self._strategies:
            sig = s.on_bar(context)
            if sig is not None:
                signals.append(sig)

        if not signals:
            return None

        buy_votes = sum(1 for s in signals if s.type == SignalType.BUY)
        sell_votes = sum(1 for s in signals if s.type == SignalType.SELL)
        total = len(self._strategies)

        if self._mode == 'unanimous':
            if buy_votes == total:
                return Signal(type=SignalType.BUY, symbol=context.symbol)
            if sell_votes == total:
                return Signal(type=SignalType.SELL, symbol=context.symbol)

        elif self._mode == 'any':
            if buy_votes > 0 and sell_votes == 0:
                return Signal(type=SignalType.BUY, symbol=context.symbol)
            if sell_votes > 0 and buy_votes == 0:
                return Signal(type=SignalType.SELL, symbol=context.symbol)

        elif self._mode == 'majority':
            if buy_votes > total / 2:
                return Signal(type=SignalType.BUY, symbol=context.symbol)
            if sell_votes > total / 2:
                return Signal(type=SignalType.SELL, symbol=context.symbol)

        return None
