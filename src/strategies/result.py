"""回测结果与绩效指标"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pandas as pd
import numpy as np


@dataclass
class TradeRecord:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    equity_curve: pd.DataFrame
    trades: List[TradeRecord] = field(default_factory=list)
    initial_capital: float = 0.0
    benchmark: str = ""
    benchmark_return: float = 0.0
    per_symbol_equity: Optional[Dict[str, pd.Series]] = None
    per_symbol_trades: Optional[Dict[str, list]] = None

    def summary(self) -> str:
        lines = [
            f"总收益率:   {self.total_return:.2%}",
            f"年化收益率: {self.annual_return:.2%}",
            f"夏普比率:   {self.sharpe_ratio:.2f}",
            f"最大回撤:   {self.max_drawdown:.2%}",
            f"回撤持续:   {self.max_drawdown_duration} 天",
            f"胜率:       {self.win_rate:.2%}",
            f"盈亏比:     {self.profit_loss_ratio:.2f}",
            f"总交易次数: {self.total_trades}",
        ]
        return "\n".join(lines)


def calc_performance_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> dict:
    """从权益曲线计算绩效指标"""
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    total_return = (final / initial) - 1

    n_periods = len(equity_curve)
    if n_periods <= 1:
        return {
            'total_return': total_return,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
        }

    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # 日收益率
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() - risk_free_rate / periods_per_year) / returns.std() * np.sqrt(periods_per_year)

    # 最大回撤
    cummax = equity_curve.cummax()
    drawdown = (cummax - equity_curve) / cummax
    max_drawdown = drawdown.max()

    # 最大回撤持续天数
    is_dd = drawdown > 0
    dd_durations = []
    current_duration = 0
    for val in is_dd:
        if val:
            current_duration += 1
        else:
            if current_duration > 0:
                dd_durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        dd_durations.append(current_duration)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_dd_duration,
    }
