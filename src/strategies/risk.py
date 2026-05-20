"""风控管理模块

在信号生成和执行之间进行风控检查，包括：
- 固定百分比止损/止盈
- ATR 止损/止盈
- 追踪止损
- 仓位限制（最大持仓数、单标的权重）
- 组合最大回撤限制
- 日亏损限制
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pandas as pd

from .base import Signal, SignalType, Portfolio, Position


@dataclass
class RiskConfig:
    """风控配置"""

    # 止损
    stop_loss_pct: Optional[float] = None
    stop_loss_atr_multiplier: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    trailing_stop_atr_multiplier: Optional[float] = None

    # 止盈
    take_profit_pct: Optional[float] = None
    take_profit_atr_multiplier: Optional[float] = None

    # 仓位限制
    max_positions: Optional[int] = None
    max_per_symbol_weight: Optional[float] = None
    max_portfolio_drawdown: Optional[float] = None

    # 日亏损限制
    daily_loss_limit: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None

    # ATR 参数
    atr_period: int = 14

    @property
    def needs_atr(self) -> bool:
        """是否需要 ATR 数据"""
        return (
            self.stop_loss_atr_multiplier is not None
            or self.trailing_stop_atr_multiplier is not None
            or self.take_profit_atr_multiplier is not None
        )


class RiskManager:
    """风控管理器"""

    def __init__(self, config: RiskConfig):
        self._config = config
        self._peak_equity: float = 0.0
        self._trailing_stops: Dict[str, float] = {}
        self._atr_at_peak: Dict[str, float] = {}
        self._daily_start_equity: float = 0.0
        self._last_date: Optional[pd.Timestamp] = None

    def reset(self) -> None:
        """重置风控状态（每次回测开始时调用）"""
        self._peak_equity = 0.0
        self._trailing_stops = {}
        self._atr_at_peak = {}
        self._daily_start_equity = 0.0
        self._last_date = None

    def check_exits(
        self,
        portfolio: Portfolio,
        bar_data: Dict[str, pd.Series],
        atr_values: Optional[Dict[str, float]] = None,
    ) -> List[Signal]:
        """检查所有持仓是否触发止损/止盈，返回自动卖出信号"""
        atr_values = atr_values or {}
        exit_signals: List[Signal] = []

        # 更新追踪止损峰值
        for sym, pos in portfolio.positions.items():
            if pos.is_empty:
                continue
            current_price = bar_data[sym]['close'] if sym in bar_data else pos.market_value / pos.quantity if pos.quantity > 0 else 0
            self._update_trailing_peak(sym, current_price, atr_values.get(sym))

        # 检查各持仓止损/止盈
        for sym, pos in portfolio.positions.items():
            if pos.is_empty:
                continue
            if sym not in bar_data:
                continue
            current_price = bar_data[sym]['close']
            atr = atr_values.get(sym)

            sig = self._check_stop_loss(sym, pos, current_price, atr)
            if sig is not None:
                exit_signals.append(sig)
                continue

            sig = self._check_trailing_stop(sym, pos, current_price, atr)
            if sig is not None:
                exit_signals.append(sig)
                continue

            sig = self._check_take_profit(sym, pos, current_price, atr)
            if sig is not None:
                exit_signals.append(sig)

        # 检查组合最大回撤
        dd_signals = self._check_drawdown_limit(portfolio)
        if dd_signals:
            exit_signals.extend(dd_signals)

        return exit_signals

    def check_signals(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        bar_data: Dict[str, pd.Series],
        atr_values: Optional[Dict[str, float]] = None,
        current_time: Optional[pd.Timestamp] = None,
    ) -> List[Signal]:
        """对策略信号进行风控过滤"""
        # 检查日亏损限制
        if current_time is not None and self._check_daily_loss(portfolio, current_time):
            return [s for s in signals if s.type == SignalType.SELL]

        filtered = self._check_position_limits(signals, portfolio)
        return filtered

    # ─── 内部方法 ───

    def _update_trailing_peak(self, symbol: str, current_price: float, atr: Optional[float]) -> None:
        """更新追踪止损的峰值价格"""
        current_stop = self._trailing_stops.get(symbol)
        if current_stop is None:
            # 首次建仓后初始化
            return

        # 追踪止损的峰值 = 止损位 / (1 - pct) 或 峰值 + multiplier * atr
        # 简化：直接追踪价格的最高点
        if self._config.trailing_stop_pct is not None:
            # 从止损位反推峰值
            peak = current_stop / (1 - self._config.trailing_stop_pct)
            if current_price > peak:
                new_stop = current_price * (1 - self._config.trailing_stop_pct)
                self._trailing_stops[symbol] = new_stop
        elif self._config.trailing_stop_atr_multiplier is not None and atr is not None:
            peak = current_stop + self._config.trailing_stop_atr_multiplier * self._atr_at_peak.get(symbol, atr)
            if current_price > peak:
                self._trailing_stops[symbol] = current_price - self._config.trailing_stop_atr_multiplier * atr
                self._atr_at_peak[symbol] = atr

    def _check_stop_loss(
        self, symbol: str, pos: Position, current_price: float, atr: Optional[float]
    ) -> Optional[Signal]:
        """固定百分比止损 + ATR 止损"""
        if self._config.stop_loss_pct is not None:
            stop_price = pos.avg_cost * (1 - self._config.stop_loss_pct)
            if current_price <= stop_price:
                return Signal(type=SignalType.SELL, symbol=symbol, reason=f"止损 (跌破 {self._config.stop_loss_pct:.0%})")

        if self._config.stop_loss_atr_multiplier is not None and atr is not None:
            stop_price = pos.avg_cost - self._config.stop_loss_atr_multiplier * atr
            if current_price <= stop_price:
                return Signal(type=SignalType.SELL, symbol=symbol, reason=f"ATR止损 ({self._config.stop_loss_atr_multiplier}x ATR)")

        return None

    def _check_trailing_stop(
        self, symbol: str, pos: Position, current_price: float, atr: Optional[float]
    ) -> Optional[Signal]:
        """追踪止损"""
        stop = self._trailing_stops.get(symbol)
        if stop is None:
            return None

        if current_price <= stop:
            return Signal(type=SignalType.SELL, symbol=symbol, reason="追踪止损")

        return None

    def _check_take_profit(
        self, symbol: str, pos: Position, current_price: float, atr: Optional[float]
    ) -> Optional[Signal]:
        """固定百分比止盈 + ATR 止盈"""
        if self._config.take_profit_pct is not None:
            target = pos.avg_cost * (1 + self._config.take_profit_pct)
            if current_price >= target:
                return Signal(type=SignalType.SELL, symbol=symbol, reason=f"止盈 (涨超 {self._config.take_profit_pct:.0%})")

        if self._config.take_profit_atr_multiplier is not None and atr is not None:
            target = pos.avg_cost + self._config.take_profit_atr_multiplier * atr
            if current_price >= target:
                return Signal(type=SignalType.SELL, symbol=symbol, reason=f"ATR止盈 ({self._config.take_profit_atr_multiplier}x ATR)")

        return None

    def _check_position_limits(self, signals: List[Signal], portfolio: Portfolio) -> List[Signal]:
        """仓位限制过滤"""
        filtered = []
        current_count = sum(1 for p in portfolio.positions.values() if not p.is_empty)

        for sig in signals:
            if sig.type == SignalType.BUY:
                # 最大持仓数
                if self._config.max_positions is not None:
                    already_held = not portfolio.get_position(sig.symbol).is_empty
                    if not already_held and current_count >= self._config.max_positions:
                        continue

                # 单标的权重限制
                if self._config.max_per_symbol_weight is not None and portfolio.equity > 0:
                    pos = portfolio.get_position(sig.symbol)
                    weight = pos.market_value / portfolio.equity
                    if weight >= self._config.max_per_symbol_weight:
                        continue

            filtered.append(sig)

        return filtered

    def _check_drawdown_limit(self, portfolio: Portfolio) -> List[Signal]:
        """组合最大回撤限制"""
        if self._config.max_portfolio_drawdown is None:
            return []

        self._peak_equity = max(self._peak_equity, portfolio.equity)
        if self._peak_equity == 0:
            return []

        drawdown = (self._peak_equity - portfolio.equity) / self._peak_equity
        if drawdown >= self._config.max_portfolio_drawdown:
            signals = []
            for sym, pos in portfolio.positions.items():
                if not pos.is_empty:
                    signals.append(Signal(type=SignalType.SELL, symbol=sym, reason="组合最大回撤限制"))
            return signals

        return []

    def _check_daily_loss(self, portfolio: Portfolio, current_time: pd.Timestamp) -> bool:
        """检查日亏损限制"""
        current_date = current_time.normalize()

        if self._last_date is None or current_date != self._last_date:
            self._last_date = current_date
            self._daily_start_equity = portfolio.equity
            return False

        loss = self._daily_start_equity - portfolio.equity

        if self._config.daily_loss_limit is not None and loss >= self._config.daily_loss_limit:
            return True

        if self._config.daily_loss_limit_pct is not None and self._daily_start_equity > 0:
            loss_pct = loss / self._daily_start_equity
            if loss_pct >= self._config.daily_loss_limit_pct:
                return True

        return False

    def register_trailing_stop(self, symbol: str, entry_price: float, atr: Optional[float] = None) -> None:
        """建仓时注册追踪止损初始位"""
        if self._config.trailing_stop_pct is not None:
            self._trailing_stops[symbol] = entry_price * (1 - self._config.trailing_stop_pct)
        elif self._config.trailing_stop_atr_multiplier is not None and atr is not None:
            self._trailing_stops[symbol] = entry_price - self._config.trailing_stop_atr_multiplier * atr
            self._atr_at_peak[symbol] = atr

    def remove_trailing_stop(self, symbol: str) -> None:
        """平仓时移除追踪止损"""
        self._trailing_stops.pop(symbol, None)
        self._atr_at_peak.pop(symbol, None)

    def get_state(self) -> dict:
        """导出内部状态用于持久化"""
        return {
            "peak_equity": self._peak_equity,
            "trailing_stops": dict(self._trailing_stops),
            "atr_at_peak": dict(self._atr_at_peak),
            "daily_start_equity": self._daily_start_equity,
            "last_date": self._last_date.isoformat() if self._last_date is not None else None,
        }

    def set_state(self, state: dict) -> None:
        """从持久化恢复内部状态"""
        self._peak_equity = state.get("peak_equity", 0.0)
        self._trailing_stops = state.get("trailing_stops", {})
        self._atr_at_peak = state.get("atr_at_peak", {})
        self._daily_start_equity = state.get("daily_start_equity", 0.0)
        last_date_str = state.get("last_date")
        self._last_date = pd.Timestamp(last_date_str) if last_date_str else None
