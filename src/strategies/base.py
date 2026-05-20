"""策略基类与核心类型定义"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    type: SignalType
    symbol: str
    price: Optional[float] = None
    quantity: Optional[float] = None
    strength: float = 1.0
    reason: str = ""


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_value: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.quantity == 0.0

    @property
    def pnl(self) -> float:
        return self.market_value - self.avg_cost * self.quantity


@dataclass
class Portfolio:
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]


@dataclass
class Context:
    bar: pd.Series
    bars: pd.DataFrame
    portfolio: Portfolio
    current_time: pd.Timestamp
    symbol: str


class Strategy(ABC):
    """策略抽象基类

    子类实现 on_init 注册指标依赖，on_bar 产生交易信号。
    可选实现 score 返回连续观点分数（-1 到 1），用于 Meta Strategy。
    """

    def __init__(self):
        self._indicator_specs: List[Dict[str, Any]] = []

    @abstractmethod
    def on_init(self, context: Context) -> None:
        pass

    @abstractmethod
    def on_bar(self, context: Context) -> Optional[Signal]:
        pass

    def on_finish(self, context: Context) -> None:
        pass

    def score(self, context: Context) -> float:
        """返回当前 bar 的连续观点分数

        -1.0 = 强烈看空, 0.0 = 中性, 1.0 = 强烈看多
        默认实现根据 on_bar 信号映射，子类可覆盖以提供更精细的分数。
        """
        sig = self.on_bar(context)
        if sig is None or sig.type == SignalType.HOLD:
            return 0.0
        s = sig.strength if sig.strength else 1.0
        if sig.type == SignalType.BUY:
            return min(s, 1.0)
        elif sig.type == SignalType.SELL:
            return -min(s, 1.0)
        return 0.0

    def register_indicator(self, name: str, **params) -> None:
        self._indicator_specs.append({'name': name, **params})

    @property
    def indicator_specs(self) -> List[Dict[str, Any]]:
        return list(self._indicator_specs)
