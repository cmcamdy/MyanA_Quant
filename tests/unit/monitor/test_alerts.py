"""AlertManager 单元测试"""

import pytest
import pandas as pd

from monitor.alerts import AlertManager, AlertEvent, LoggingCallback
from strategies.base import Signal, SignalType


class TestAlertManager:

    def test_signal_fired(self):
        alerts = AlertManager(on_signal=True)
        sig = Signal(type=SignalType.BUY, symbol="600036.SH", reason="MA金叉")
        alerts.on_signal_fired(sig, pd.Timestamp("2026-01-10 14:30"))
        events = alerts.get_recent_events()
        assert len(events) == 1
        assert events[0].event_type == "signal"
        assert events[0].symbol == "600036.SH"
        assert "BUY" in events[0].details

    def test_signal_disabled(self):
        alerts = AlertManager(on_signal=False)
        sig = Signal(type=SignalType.BUY, symbol="600036.SH")
        alerts.on_signal_fired(sig, pd.Timestamp("2026-01-10"))
        assert len(alerts.get_recent_events()) == 0

    def test_risk_triggered(self):
        alerts = AlertManager(on_risk_trigger=True)
        sig = Signal(type=SignalType.SELL, symbol="600036.SH", reason="追踪止损")
        alerts.on_risk_triggered(sig, pd.Timestamp("2026-01-10"))
        events = alerts.get_recent_events()
        assert len(events) == 1
        assert events[0].severity == "critical"

    def test_risk_disabled(self):
        alerts = AlertManager(on_risk_trigger=False)
        sig = Signal(type=SignalType.SELL, symbol="600036.SH", reason="止损")
        alerts.on_risk_triggered(sig, pd.Timestamp("2026-01-10"))
        assert len(alerts.get_recent_events()) == 0

    def test_equity_change_alert(self):
        alerts = AlertManager(equity_change_threshold=0.05)
        alerts.on_equity_change(100000.0, pd.Timestamp("2026-01-10"))
        # 5% drop
        alerts.on_equity_change(95000.0, pd.Timestamp("2026-01-11"))
        events = alerts.get_recent_events()
        assert len(events) == 1
        assert events[0].event_type == "equity_change"

    def test_equity_change_no_alert_below_threshold(self):
        alerts = AlertManager(equity_change_threshold=0.05)
        alerts.on_equity_change(100000.0, pd.Timestamp("2026-01-10"))
        # Only 1% change
        alerts.on_equity_change(99000.0, pd.Timestamp("2026-01-11"))
        assert len(alerts.get_recent_events()) == 0

    def test_recent_events_limit(self):
        alerts = AlertManager(on_signal=True)
        for i in range(60):
            sig = Signal(type=SignalType.BUY, symbol="TEST")
            alerts.on_signal_fired(sig, pd.Timestamp("2026-01-10"))
        events = alerts.get_recent_events(n=100)
        assert len(events) == 50  # max_recent = 50

    def test_callback_invoked(self):
        received = []
        alerts = AlertManager(callbacks=[lambda e: received.append(e)])
        sig = Signal(type=SignalType.BUY, symbol="600036.SH")
        alerts.on_signal_fired(sig, pd.Timestamp("2026-01-10"))
        assert len(received) == 1

    def test_callback_error_does_not_crash(self):
        def bad_callback(e):
            raise RuntimeError("bad")

        alerts = AlertManager(callbacks=[bad_callback])
        sig = Signal(type=SignalType.BUY, symbol="600036.SH")
        alerts.on_signal_fired(sig, pd.Timestamp("2026-01-10"))  # should not raise
        assert len(alerts.get_recent_events()) == 1


class TestAlertEvent:

    def test_creation(self):
        event = AlertEvent(
            timestamp=pd.Timestamp("2026-01-10"),
            event_type="signal",
            symbol="600036.SH",
            details="BUY (MA金叉)",
            severity="info",
        )
        assert event.event_type == "signal"
        assert event.severity == "info"
