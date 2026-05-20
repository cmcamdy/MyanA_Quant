"""组合策略：合并多个子策略信号"""

from collections import defaultdict
from typing import Dict, List, Optional

from .base import Strategy, Context, Signal, SignalType
from .portfolio_engine import MultiStrategy, PortfolioContext


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


class CompositeMultiStrategy(MultiStrategy):
    """合并多个多标的子策略信号

    Args:
        strategies: 多标的是策略列表
        mode: 合并模式
            'unanimous' - 所有子策略对同一标的一致才发信号
            'any' - 任一子策略对同一标的发信号即触发（无反对信号）
            'majority' - 多数子策略对同一标的一致才发信号
    """

    def __init__(self, strategies: List[MultiStrategy], mode: str = 'any'):
        super().__init__()
        self._strategies = strategies
        self._mode = mode

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        for s in self._strategies:
            s.on_init_multi(ctx)
        for s in self._strategies:
            for spec in s.indicator_specs:
                self.register_indicator(**spec)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        # 收集所有子策略的信号
        all_signals: List[List[Signal]] = []
        for s in self._strategies:
            sigs = s.on_bar_multi(ctx)
            all_signals.append(sigs if sigs else [])

        # 按标的分组投票
        sym_buy: Dict[str, int] = defaultdict(int)
        sym_sell: Dict[str, int] = defaultdict(int)
        total = len(self._strategies)

        for sigs in all_signals:
            for sig in sigs:
                if sig.type == SignalType.BUY:
                    sym_buy[sig.symbol] += 1
                elif sig.type == SignalType.SELL:
                    sym_sell[sig.symbol] += 1

        # 汇总所有出现过的标的
        all_symbols = set(sym_buy) | set(sym_sell)
        result = []

        for sym in all_symbols:
            b = sym_buy.get(sym, 0)
            s = sym_sell.get(sym, 0)

            if self._mode == 'unanimous':
                if b == total:
                    result.append(Signal(type=SignalType.BUY, symbol=sym))
                elif s == total:
                    result.append(Signal(type=SignalType.SELL, symbol=sym))

            elif self._mode == 'any':
                if b > 0 and s == 0:
                    result.append(Signal(type=SignalType.BUY, symbol=sym))
                elif s > 0 and b == 0:
                    result.append(Signal(type=SignalType.SELL, symbol=sym))

            elif self._mode == 'majority':
                if b > total / 2:
                    result.append(Signal(type=SignalType.BUY, symbol=sym))
                elif s > total / 2:
                    result.append(Signal(type=SignalType.SELL, symbol=sym))

        return result
