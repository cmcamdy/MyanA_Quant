"""策略基类测试"""

import pytest
import pandas as pd

from strategies.base import Strategy, Signal, SignalType, Context, Portfolio, Position


class TestSignalType:

    def test_values(self):
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"


class TestSignal:

    def test_basic_signal(self):
        s = Signal(type=SignalType.BUY, symbol="000001.SZ")
        assert s.type == SignalType.BUY
        assert s.symbol == "000001.SZ"
        assert s.price is None
        assert s.strength == 1.0

    def test_signal_with_details(self):
        s = Signal(type=SignalType.SELL, symbol="000001.SZ", price=10.5, quantity=100, strength=0.5, reason="止损")
        assert s.price == 10.5
        assert s.quantity == 100
        assert s.strength == 0.5
        assert s.reason == "止损"


class TestPosition:

    def test_empty_position(self):
        pos = Position(symbol="000001.SZ")
        assert pos.is_empty is True
        assert pos.pnl == 0.0

    def test_position_with_holdings(self):
        pos = Position(symbol="000001.SZ", quantity=100, avg_cost=10.0, market_value=1200.0)
        assert pos.is_empty is False
        assert pos.pnl == 200.0  # 1200 - 10*100

    def test_position_loss(self):
        pos = Position(symbol="000001.SZ", quantity=100, avg_cost=10.0, market_value=800.0)
        assert pos.pnl == -200.0


class TestPortfolio:

    def test_empty_portfolio(self):
        p = Portfolio()
        assert p.cash == 0.0
        assert p.equity == 0.0
        assert len(p.positions) == 0

    def test_get_position_creates_if_missing(self):
        p = Portfolio()
        pos = p.get_position("000001.SZ")
        assert isinstance(pos, Position)
        assert pos.symbol == "000001.SZ"
        assert "000001.SZ" in p.positions

    def test_get_position_existing(self):
        p = Portfolio()
        pos1 = p.get_position("000001.SZ")
        pos1.quantity = 100
        pos2 = p.get_position("000001.SZ")
        assert pos2.quantity == 100  # 同一个对象

    def test_portfolio_with_cash(self):
        p = Portfolio(cash=100000.0)
        assert p.cash == 100000.0


class TestStrategy:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Strategy()

    def test_register_indicator(self):
        class TestStrat(Strategy):
            def on_init(self, ctx):
                self.register_indicator('ma', period=5)
            def on_bar(self, ctx):
                return None

        s = TestStrat()
        s.on_init(None)
        assert len(s.indicator_specs) == 1
        assert s.indicator_specs[0] == {'name': 'ma', 'period': 5}

    def test_register_multiple_indicators(self):
        class TestStrat(Strategy):
            def on_init(self, ctx):
                self.register_indicator('ma', period=5)
                self.register_indicator('ma', period=20)
                self.register_indicator('rsi', period=14)
            def on_bar(self, ctx):
                return None

        s = TestStrat()
        s.on_init(None)
        assert len(s.indicator_specs) == 3

    def test_on_finish_default(self):
        class TestStrat(Strategy):
            def on_init(self, ctx): pass
            def on_bar(self, ctx): return None

        s = TestStrat()
        # on_finish有默认空实现，不应报错
        s.on_finish(None)


class TestContext:

    def test_context_creation(self):
        bar = pd.Series({'close': 10.0})
        bars = pd.DataFrame({'close': [10.0]})
        portfolio = Portfolio(cash=100000.0)

        ctx = Context(
            bar=bar, bars=bars, portfolio=portfolio,
            current_time=pd.Timestamp('2024-01-01'), symbol='000001.SZ'
        )
        assert ctx.symbol == '000001.SZ'
        assert ctx.portfolio.cash == 100000.0
