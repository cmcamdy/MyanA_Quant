"""告警管理器"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import pandas as pd

from strategies.base import Signal, SignalType
from strategies.result import TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """告警事件"""
    timestamp: pd.Timestamp
    event_type: str       # "signal", "risk_trigger", "equity_change", "trade"
    symbol: str
    details: str
    severity: str         # "info", "warning", "critical"


class AlertCallback(Protocol):
    """告警回调协议"""
    def __call__(self, event: AlertEvent) -> None: ...


class LoggingCallback:
    """内置回调：写 Python logging"""

    def __init__(self, log: Optional[logging.Logger] = None):
        self._logger = log or logger

    def __call__(self, event: AlertEvent) -> None:
        msg = f"[{event.timestamp}] {event.severity.upper()} {event.event_type} {event.symbol}: {event.details}"
        if event.severity == "critical":
            self._logger.critical(msg)
        elif event.severity == "warning":
            self._logger.warning(msg)
        else:
            self._logger.info(msg)


class AlertManager:
    """告警管理器"""

    def __init__(
        self,
        on_signal: bool = True,
        on_risk_trigger: bool = True,
        equity_change_threshold: Optional[float] = None,
        callbacks: Optional[List[AlertCallback]] = None,
        log_file: Optional[str] = None,
    ):
        self._on_signal = on_signal
        self._on_risk_trigger = on_risk_trigger
        self._equity_change_threshold = equity_change_threshold
        self._callbacks: List[AlertCallback] = list(callbacks or [])
        self._log_file = log_file
        self._last_equity: Optional[float] = None
        self._recent_events: List[AlertEvent] = []
        self._max_recent = 50

        if log_file:
            self._file_handler = logging.FileHandler(log_file, encoding="utf-8")
            self._file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(message)s")
            )
            logger.addHandler(self._file_handler)

    def on_signal_fired(self, signal: Signal, current_time: pd.Timestamp) -> None:
        """策略信号触发"""
        if not self._on_signal:
            return
        action = signal.type.value
        reason = f" ({signal.reason})" if signal.reason else ""
        severity = "info" if signal.type == SignalType.HOLD else "warning"
        event = AlertEvent(
            timestamp=current_time,
            event_type="signal",
            symbol=signal.symbol,
            details=f"{action}{reason}",
            severity=severity,
        )
        self._fire(event)

    def on_risk_triggered(self, signal: Signal, current_time: pd.Timestamp) -> None:
        """风控触发"""
        if not self._on_risk_trigger:
            return
        event = AlertEvent(
            timestamp=current_time,
            event_type="risk_trigger",
            symbol=signal.symbol,
            details=signal.reason or "风控触发",
            severity="critical",
        )
        self._fire(event)

    def on_equity_change(self, equity: float, current_time: pd.Timestamp) -> None:
        """权益变化检查"""
        if self._equity_change_threshold is None:
            return
        if self._last_equity is not None and self._last_equity > 0:
            change_pct = abs(equity - self._last_equity) / self._last_equity
            if change_pct >= self._equity_change_threshold:
                direction = "+" if equity > self._last_equity else "-"
                event = AlertEvent(
                    timestamp=current_time,
                    event_type="equity_change",
                    symbol="PORTFOLIO",
                    details=f"权益变动 {direction}{change_pct:.1%} (当前 {equity:,.0f})",
                    severity="warning",
                )
                self._fire(event)
        self._last_equity = equity

    def on_trade_executed(self, trade: TradeRecord, current_time: pd.Timestamp) -> None:
        """交易执行"""
        pnl_str = f", PnL={trade.pnl:+.2f}" if trade.exit_time is not None else ""
        event = AlertEvent(
            timestamp=current_time,
            event_type="trade",
            symbol=trade.symbol,
            details=f"qty={trade.quantity:.0f}, price={trade.entry_price:.2f}{pnl_str}",
            severity="info",
        )
        self._fire(event)

    def get_recent_events(self, n: int = 10) -> List[AlertEvent]:
        """获取最近N条告警事件"""
        return self._recent_events[-n:]

    def _fire(self, event: AlertEvent) -> None:
        """分发告警到所有回调"""
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_recent:
            self._recent_events = self._recent_events[-self._max_recent:]
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning("Alert callback error: %s", e)
