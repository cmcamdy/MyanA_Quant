"""监控状态持久化"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from strategies.base import Portfolio, Position


@dataclass
class MonitorState:
    """监控运行时状态，支持序列化到JSON以实现重启恢复"""

    monitor_id: str = ""
    cash: float = 0.0
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    open_trades: Dict[str, Dict] = field(default_factory=dict)
    last_processed_time: Dict[str, str] = field(default_factory=dict)
    equity_history: List[Dict] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    # RiskManager 状态
    peak_equity: float = 0.0
    trailing_stops: Dict[str, float] = field(default_factory=dict)
    atr_at_peak: Dict[str, float] = field(default_factory=dict)
    daily_start_equity: float = 0.0
    last_date: Optional[str] = None

    def save(self, path: str) -> None:
        """原子写入状态到JSON文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "MonitorState":
        """从JSON文件加载状态"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    def to_portfolio(self) -> Portfolio:
        """从状态重建 Portfolio"""
        portfolio = Portfolio(cash=self.cash, equity=self.cash)
        for sym, pos_data in self.positions.items():
            pos = Position(
                symbol=sym,
                quantity=pos_data.get("quantity", 0.0),
                avg_cost=pos_data.get("avg_cost", 0.0),
                market_value=pos_data.get("market_value", 0.0),
            )
            portfolio.positions[sym] = pos
        portfolio.equity = portfolio.cash + sum(
            p.market_value for p in portfolio.positions.values()
        )
        return portfolio

    @classmethod
    def from_portfolio(cls, portfolio: Portfolio, monitor_id: str) -> "MonitorState":
        """从 Portfolio 创建初始状态"""
        positions = {}
        for sym, pos in portfolio.positions.items():
            if not pos.is_empty:
                positions[sym] = {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                }
        return cls(
            monitor_id=monitor_id,
            cash=portfolio.cash,
            positions=positions,
            peak_equity=portfolio.equity,
        )

    def apply_risk_state(self, risk_manager) -> None:
        """恢复 RiskManager 内部状态"""
        risk_manager.set_state({
            "peak_equity": self.peak_equity,
            "trailing_stops": dict(self.trailing_stops),
            "atr_at_peak": dict(self.atr_at_peak),
            "daily_start_equity": self.daily_start_equity,
            "last_date": self.last_date,
        })

    def extract_risk_state(self, risk_manager) -> None:
        """从 RiskManager 提取内部状态"""
        state = risk_manager.get_state()
        self.peak_equity = state.get("peak_equity", 0.0)
        self.trailing_stops = state.get("trailing_stops", {})
        self.atr_at_peak = state.get("atr_at_peak", {})
        self.daily_start_equity = state.get("daily_start_equity", 0.0)
        self.last_date = state.get("last_date")

    def _to_dict(self) -> dict:
        return {
            "monitor_id": self.monitor_id,
            "cash": self.cash,
            "positions": self.positions,
            "open_trades": self.open_trades,
            "last_processed_time": self.last_processed_time,
            "equity_history": self.equity_history[-500:],  # 只保留最近500条
            "trade_history": self.trade_history[-200:],    # 只保留最近200条
            "peak_equity": self.peak_equity,
            "trailing_stops": self.trailing_stops,
            "atr_at_peak": self.atr_at_peak,
            "daily_start_equity": self.daily_start_equity,
            "last_date": self.last_date,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "MonitorState":
        return cls(
            monitor_id=data.get("monitor_id", ""),
            cash=data.get("cash", 0.0),
            positions=data.get("positions", {}),
            open_trades=data.get("open_trades", {}),
            last_processed_time=data.get("last_processed_time", {}),
            equity_history=data.get("equity_history", []),
            trade_history=data.get("trade_history", []),
            peak_equity=data.get("peak_equity", 0.0),
            trailing_stops=data.get("trailing_stops", {}),
            atr_at_peak=data.get("atr_at_peak", {}),
            daily_start_equity=data.get("daily_start_equity", 0.0),
            last_date=data.get("last_date"),
        )
