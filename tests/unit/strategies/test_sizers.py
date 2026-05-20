"""仓位管理器测试"""

import pytest

from strategies.base import Signal, SignalType, Portfolio, Position
from strategies.sizers import FixedSizer, AllInSizer


@pytest.fixture
def portfolio():
    p = Portfolio(cash=100000.0)
    p.positions['000001.SZ'] = Position(symbol='000001.SZ', quantity=1000, avg_cost=10.0, market_value=10000.0)
    return p


@pytest.fixture
def buy_signal():
    return Signal(type=SignalType.BUY, symbol='000001.SZ')


@pytest.fixture
def sell_signal():
    return Signal(type=SignalType.SELL, symbol='000001.SZ')


class TestFixedSizer:

    def test_buy_full_position(self, portfolio, buy_signal):
        sizer = FixedSizer(percent=1.0)
        qty = sizer.compute_quantity(buy_signal, portfolio, current_price=10.0)
        # 100000 / 10 = 10000
        assert qty == 10000.0

    def test_buy_half_position(self, portfolio, buy_signal):
        sizer = FixedSizer(percent=0.5)
        qty = sizer.compute_quantity(buy_signal, portfolio, current_price=10.0)
        assert qty == 5000.0

    def test_sell_full_position(self, portfolio, sell_signal):
        sizer = FixedSizer(percent=1.0)
        qty = sizer.compute_quantity(sell_signal, portfolio, current_price=10.0)
        assert qty == 1000.0  # 全部持仓

    def test_sell_half_position(self, portfolio, sell_signal):
        sizer = FixedSizer(percent=0.5)
        qty = sizer.compute_quantity(sell_signal, portfolio, current_price=10.0)
        assert qty == 500.0

    def test_buy_with_strength(self, portfolio, buy_signal):
        buy_signal.strength = 0.5
        sizer = FixedSizer(percent=1.0)
        qty = sizer.compute_quantity(buy_signal, portfolio, current_price=10.0)
        assert qty == 5000.0  # 100000 * 1.0 * 0.5 / 10

    def test_sell_with_strength(self, portfolio, sell_signal):
        sell_signal.strength = 0.3
        sizer = FixedSizer(percent=1.0)
        qty = sizer.compute_quantity(sell_signal, portfolio, current_price=10.0)
        assert qty == 300.0  # 1000 * 1.0 * 0.3

    def test_zero_price(self, portfolio, buy_signal):
        sizer = FixedSizer(percent=1.0)
        qty = sizer.compute_quantity(buy_signal, portfolio, current_price=0.0)
        assert qty == 0.0

    def test_hold_returns_zero(self, portfolio):
        signal = Signal(type=SignalType.HOLD, symbol='000001.SZ')
        sizer = FixedSizer(percent=1.0)
        qty = sizer.compute_quantity(signal, portfolio, current_price=10.0)
        assert qty == 0.0


class TestAllInSizer:

    def test_buy_all_in(self, portfolio, buy_signal):
        sizer = AllInSizer()
        qty = sizer.compute_quantity(buy_signal, portfolio, current_price=10.0)
        assert qty == 10000.0

    def test_sell_all_in(self, portfolio, sell_signal):
        sizer = AllInSizer()
        qty = sizer.compute_quantity(sell_signal, portfolio, current_price=10.0)
        assert qty == 1000.0

    def test_is_fixed_sizer_subclass(self):
        assert issubclass(AllInSizer, FixedSizer)
