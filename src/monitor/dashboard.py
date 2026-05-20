"""终端仪表盘"""

import shutil
import time
from typing import Dict, List, Optional

import pandas as pd

from strategies.base import Portfolio, Position
from strategies.base import Signal
from monitor.config import MonitorConfig


_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


class ConsoleDashboard:
    """终端仪表盘，用ANSI控制码原地重绘"""

    def __init__(self, refresh_interval: int = 5):
        self._refresh_interval = refresh_interval
        self._last_draw_time: float = 0.0

    def should_draw(self) -> bool:
        """是否到了刷新时间"""
        now = time.time()
        if now - self._last_draw_time >= self._refresh_interval:
            self._last_draw_time = now
            return True
        return False

    def draw(
        self,
        strategy_name: str,
        portfolio: Portfolio,
        positions_detail: Dict[str, Dict],
        recent_signals: List[Dict],
        recent_alerts: List[Dict],
        equity_history: List[Dict],
        config: MonitorConfig,
        poll_count: int,
        last_poll_time: Optional[pd.Timestamp],
        uptime_seconds: float,
    ) -> None:
        """重绘仪表盘"""
        lines = []
        lines.append(self._format_header(strategy_name, config, uptime_seconds))
        lines.append("")
        lines.append(self._format_portfolio(portfolio))
        lines.append("")
        lines.append(self._format_positions(portfolio, positions_detail))
        lines.append("")
        lines.append(self._format_signals(recent_signals))
        lines.append("")
        lines.append(self._format_alerts(recent_alerts))
        lines.append("")
        lines.append(self._format_equity_sparkline(equity_history))
        lines.append("")
        lines.append(self._format_footer(poll_count, last_poll_time))
        lines.append("=" * 60)

        output = "\n".join(lines)
        self._clear_screen()
        print(output)

    def clear(self) -> None:
        """清屏"""
        self._clear_screen()
        print("Monitor stopped.")

    def _clear_screen(self) -> None:
        print("\033[H\033[J", end="")

    def _format_header(self, strategy_name: str, config: MonitorConfig, uptime: float) -> str:
        symbols_str = ", ".join(config.symbols[:3])
        if len(config.symbols) > 3:
            symbols_str += f" +{len(config.symbols) - 3}"
        uptime_str = self._format_duration(uptime)
        header = "=" * 60
        title = f"  Strategy Monitor - {strategy_name}"
        info = f"  Symbols: {symbols_str} | Freq: {config.freq}"
        poll = f"  Poll: every {config.poll_interval}s | Uptime: {uptime_str}"
        return f"{header}\n{title}\n{info}\n{poll}\n{header}"

    def _format_portfolio(self, portfolio: Portfolio) -> str:
        initial = 100000.0
        ret = (portfolio.equity - initial) / initial if initial > 0 else 0.0
        ret_str = f"{ret:+.2%}"
        lines = [
            "  PORTFOLIO",
            f"  Equity: {portfolio.equity:>12,.2f}  Cash: {portfolio.cash:>12,.2f}  Return: {ret_str}",
        ]
        return "\n".join(lines)

    def _format_positions(
        self, portfolio: Portfolio, positions_detail: Dict[str, Dict]
    ) -> str:
        lines = ["  POSITIONS"]
        active = {s: p for s, p in portfolio.positions.items() if not p.is_empty}
        if not active:
            lines.append("  (no positions)")
            return "\n".join(lines)

        lines.append(f"  {'Symbol':<12} {'Qty':>6} {'AvgCost':>8} {'Price':>8} {'PnL':>10} {'PnL%':>7}")
        for sym, pos in active.items():
            detail = positions_detail.get(sym, {})
            current_price = detail.get("current_price", 0.0)
            pnl = pos.pnl
            pnl_pct = pnl / (pos.avg_cost * pos.quantity) * 100 if pos.avg_cost * pos.quantity > 0 else 0.0
            lines.append(
                f"  {sym:<12} {pos.quantity:>6.0f} {pos.avg_cost:>8.2f} "
                f"{current_price:>8.2f} {pnl:>+10.0f} {pnl_pct:>+6.1f}%"
            )
        return "\n".join(lines)

    def _format_signals(self, recent_signals: List[Dict]) -> str:
        lines = ["  RECENT SIGNALS (last 5)"]
        if not recent_signals:
            lines.append("  (none)")
            return "\n".join(lines)
        for sig in recent_signals[-5:]:
            t = sig.get("time", "")
            sym = sig.get("symbol", "")
            action = sig.get("type", "")
            reason = sig.get("reason", "")
            reason_str = f"  ({reason})" if reason else ""
            lines.append(f"  {t}  {sym:<12} {action}{reason_str}")
        return "\n".join(lines)

    def _format_alerts(self, recent_alerts: List[Dict]) -> str:
        lines = ["  RISK ALERTS (last 3)"]
        if not recent_alerts:
            lines.append("  (none)")
            return "\n".join(lines)
        for alert in recent_alerts[-3:]:
            t = alert.get("time", "")
            sym = alert.get("symbol", "")
            details = alert.get("details", "")
            lines.append(f"  {t}  {sym:<12} {details}")
        return "\n".join(lines)

    def _format_equity_sparkline(self, equity_history: List[Dict]) -> str:
        lines = ["  EQUITY"]
        if len(equity_history) < 2:
            lines.append("  (insufficient data)")
            return "\n".join(lines)

        values = [h.get("equity", 0) for h in equity_history[-20:]]
        sparkline = self._make_sparkline(values)
        lines.append(f"  {sparkline}")
        return "\n".join(lines)

    def _format_footer(self, poll_count: int, last_poll_time: Optional[pd.Timestamp]) -> str:
        last_str = str(last_poll_time) if last_poll_time else "N/A"
        return f"  Polls: {poll_count} | Last: {last_str}"

    @staticmethod
    def _make_sparkline(values: List[float]) -> str:
        """将数值序列转为 Unicode 迷你图"""
        if not values:
            return ""
        vmin = min(values)
        vmax = max(values)
        vrange = vmax - vmin
        if vrange == 0:
            return _SPARKLINE_CHARS[-1] * len(values)
        result = []
        for v in values:
            idx = int((v - vmin) / vrange * (len(_SPARKLINE_CHARS) - 1))
            idx = max(0, min(idx, len(_SPARKLINE_CHARS) - 1))
            result.append(_SPARKLINE_CHARS[idx])
        return "".join(result)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
