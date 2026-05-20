"""MonitorState 单元测试"""

import os
import tempfile

import pandas as pd
import pytest

from monitor.state import MonitorState
from strategies.base import Portfolio, Position


class TestMonitorStateSaveLoad:

    def test_roundtrip(self):
        state = MonitorState(
            monitor_id="test",
            cash=50000.0,
            positions={"600036.SH": {"quantity": 100, "avg_cost": 38.5, "market_value": 4000}},
            open_trades={"600036.SH": {"symbol": "600036.SH", "entry_time": "2026-01-01", "entry_price": 38.5, "quantity": 100}},
            last_processed_time={"600036.SH": "2026-01-10"},
            equity_history=[{"time": "2026-01-10", "equity": 54000}],
            trade_history=[],
            peak_equity=54000,
            trailing_stops={"600036.SH": 37.0},
            atr_at_peak={"600036.SH": 1.5},
            daily_start_equity=53500,
            last_date="2026-01-10",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)
            loaded = MonitorState.load(path)
            assert loaded.monitor_id == "test"
            assert loaded.cash == 50000.0
            assert "600036.SH" in loaded.positions
            assert loaded.positions["600036.SH"]["quantity"] == 100
            assert loaded.peak_equity == 54000
            assert loaded.trailing_stops["600036.SH"] == 37.0
            assert loaded.last_date == "2026-01-10"

    def test_atomic_write(self):
        """保存应该是原子的（写 .tmp 再 rename）"""
        state = MonitorState(monitor_id="atomic", cash=100000.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)
            assert os.path.exists(path)
            # .tmp 文件不应残留
            assert not os.path.exists(path + ".tmp")

    def test_equity_history_truncation(self):
        """保存时 equity_history 只保留最近500条"""
        state = MonitorState(
            monitor_id="trunc",
            cash=100000.0,
            equity_history=[{"time": f"2026-01-{i:02d}", "equity": 100000 + i} for i in range(1, 600)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            state.save(path)
            loaded = MonitorState.load(path)
            assert len(loaded.equity_history) == 500


class TestMonitorStatePortfolio:

    def test_from_portfolio(self):
        portfolio = Portfolio(cash=50000.0, equity=100000.0)
        pos = portfolio.get_position("600036.SH")
        pos.quantity = 100
        pos.avg_cost = 38.5
        pos.market_value = 4000.0
        portfolio.equity = portfolio.cash + pos.market_value

        state = MonitorState.from_portfolio(portfolio, "test")
        assert state.cash == 50000.0
        assert "600036.SH" in state.positions
        assert state.positions["600036.SH"]["quantity"] == 100

    def test_to_portfolio(self):
        state = MonitorState(
            monitor_id="test",
            cash=50000.0,
            positions={"600036.SH": {"quantity": 100, "avg_cost": 38.5, "market_value": 4000}},
        )
        portfolio = state.to_portfolio()
        assert portfolio.cash == 50000.0
        pos = portfolio.get_position("600036.SH")
        assert pos.quantity == 100
        assert pos.avg_cost == 38.5
        assert portfolio.equity == 54000.0

    def test_roundtrip_portfolio(self):
        portfolio = Portfolio(cash=50000.0, equity=100000.0)
        pos = portfolio.get_position("600036.SH")
        pos.quantity = 100
        pos.avg_cost = 38.5
        pos.market_value = 4000.0

        state = MonitorState.from_portfolio(portfolio, "rt")
        restored = state.to_portfolio()
        assert abs(restored.cash - portfolio.cash) < 1e-10
        assert abs(restored.get_position("600036.SH").quantity - 100) < 1e-10
        assert abs(restored.get_position("600036.SH").avg_cost - 38.5) < 1e-10


class TestMonitorStateRiskState:

    def test_risk_state_roundtrip(self):
        from strategies.risk import RiskManager, RiskConfig

        rm = RiskManager(RiskConfig(trailing_stop_pct=0.05))
        rm._peak_equity = 105000.0
        rm._trailing_stops = {"600036.SH": 40.0}
        rm._atr_at_peak = {"600036.SH": 1.5}
        rm._daily_start_equity = 104000.0
        rm._last_date = pd.Timestamp("2026-01-10")

        state = MonitorState(monitor_id="risk_test", cash=50000.0)
        state.extract_risk_state(rm)
        assert state.peak_equity == 105000.0
        assert state.trailing_stops["600036.SH"] == 40.0

        # 恢复到新的 RiskManager
        rm2 = RiskManager(RiskConfig(trailing_stop_pct=0.05))
        state.apply_risk_state(rm2)
        assert rm2._peak_equity == 105000.0
        assert rm2._trailing_stops["600036.SH"] == 40.0
        assert rm2._daily_start_equity == 104000.0
        assert rm2._last_date == pd.Timestamp("2026-01-10")
