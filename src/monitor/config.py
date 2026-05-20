"""策略实时监控模块配置"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MonitorConfig:
    """监控配置"""
    symbols: List[str] = field(default_factory=list)
    freq: str = "1d"
    poll_interval: int = 60              # 轮询间隔(秒)
    initial_capital: float = 100000.0
    commission: float = 0.0003
    slippage: float = 0.0001
    state_dir: str = "./data/monitor"
    data_dir: str = "./data"
    data_source: Optional[str] = None    # None = 自动选择
    adjust: str = "qfq"
    initial_positions: Optional[Dict[str, Dict[str, float]]] = None
    dashboard_refresh: int = 5           # 仪表盘刷新间隔(秒)
    alert_on_signal: bool = True
    alert_on_risk_trigger: bool = True
    alert_on_equity_change_pct: Optional[float] = 0.05
