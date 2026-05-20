"""组合策略测试"""

import pytest
import pandas as pd

from strategies.base import Strategy, Signal, SignalType, Context
from strategies.composite import CompositeStrategy


class AlwaysBuyStrategy(Strategy):
    def on_init(self, context): pass
    def on_bar(self, context):
        return Signal(type=SignalType.BUY, symbol=context.symbol)


class AlwaysSellStrategy(Strategy):
    def on_init(self, context): pass
    def on_bar(self, context):
        return Signal(type=SignalType.SELL, symbol=context.symbol)


class NeverSignalStrategy(Strategy):
    def on_init(self, context): pass
    def on_bar(self, context):
        return None


@pytest.fixture
def ctx():
    return Context(
        bar=pd.Series({'close': 10.0}),
        bars=pd.DataFrame({'close': [10.0]}),
        portfolio=None,
        current_time=pd.Timestamp('2024-01-01'),
        symbol='000001.SZ',
    )


class TestCompositeStrategyUnanimous:

    def test_all_buy(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), AlwaysBuyStrategy()], mode='unanimous'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig.type == SignalType.BUY

    def test_all_sell(self, ctx):
        cs = CompositeStrategy(
            [AlwaysSellStrategy(), AlwaysSellStrategy()], mode='unanimous'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig.type == SignalType.SELL

    def test_disagree_returns_none(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), AlwaysSellStrategy()], mode='unanimous'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig is None

    def test_one_silent_one_buy(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), NeverSignalStrategy()], mode='unanimous'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig is None  # 不是一致同意


class TestCompositeStrategyAny:

    def test_any_buy(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), NeverSignalStrategy()], mode='any'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig.type == SignalType.BUY

    def test_conflicting_returns_none(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), AlwaysSellStrategy()], mode='any'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig is None  # 同时有买和卖，冲突

    def test_all_silent_returns_none(self, ctx):
        cs = CompositeStrategy(
            [NeverSignalStrategy(), NeverSignalStrategy()], mode='any'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig is None


class TestCompositeStrategyMajority:

    def test_majority_buy(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), AlwaysBuyStrategy(), NeverSignalStrategy()],
            mode='majority'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig.type == SignalType.BUY

    def test_no_majority(self, ctx):
        cs = CompositeStrategy(
            [AlwaysBuyStrategy(), AlwaysSellStrategy(), NeverSignalStrategy()],
            mode='majority'
        )
        cs.on_init(ctx)
        sig = cs.on_bar(ctx)
        assert sig is None


class TestCompositeStrategyIndicators:

    def test_merges_indicator_specs(self, ctx):
        class MAStrat(Strategy):
            def on_init(self, c): self.register_indicator('ma', period=5)
            def on_bar(self, c): return None

        class RSIStrat(Strategy):
            def on_init(self, c): self.register_indicator('rsi', period=14)
            def on_bar(self, c): return None

        cs = CompositeStrategy([MAStrat(), RSIStrat()])
        # on_init会调用子策略的on_init并注册指标
        cs.on_init(ctx)
        specs = cs.indicator_specs
        assert len(specs) == 2
        names = [s['name'] for s in specs]
        assert 'ma' in names
        assert 'rsi' in names
