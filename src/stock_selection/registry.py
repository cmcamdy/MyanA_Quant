"""选股方法注册表

统一入口: create_screener(method, **config) 创建选股筛选器实例。
新增方法只需调用 register_screener(name, factory) 注册。
"""

from typing import Callable, Dict

from .base import ScreenerBase


_SCREENER_FACTORIES: Dict[str, Callable] = {}


def register_screener(name: str, factory: Callable) -> None:
    """注册选股方法

    Args:
        name: 方法名 (如 "jys", "factor", "trend")
        factory: 工厂函数，接受 **config -> ScreenerBase 实例
    """
    _SCREENER_FACTORIES[name] = factory


def create_screener(method: str, **config) -> ScreenerBase:
    """创建选股筛选器实例

    Args:
        method: 方法名
        **config: 传递给工厂函数的配置参数

    Raises:
        ValueError: 未注册的方法名
    """
    if method not in _SCREENER_FACTORIES:
        available = ", ".join(sorted(_SCREENER_FACTORIES.keys()))
        raise ValueError(f"未知选股方法: {method}，可选: {available}")
    return _SCREENER_FACTORIES[method](**config)


def available_methods() -> list:
    """返回已注册的方法名列表"""
    return sorted(_SCREENER_FACTORIES.keys())


# ---- 注册内置方法 ----

def _register_defaults() -> None:
    from .jys_screener import JYSScreener

    register_screener("jys", lambda **cfg: JYSScreener(
        top_n=cfg.get("top_n", 10),
        min_score=cfg.get("min_score", 40),
        max_pe=cfg.get("max_pe", 30),
        min_turnover=cfg.get("min_turnover", 0.3),
        min_price=cfg.get("min_price", 1.0),
        max_concurrent=cfg.get("max_concurrent", 20),
        market=cfg.get("market", "all"),
    ))

    from .factor_adapter import FactorScreenerAdapter

    register_screener("factor", lambda **cfg: FactorScreenerAdapter(
        top_n=cfg.get("top_n", 10),
        min_bars=cfg.get("min_bars", 60),
        factors=cfg.get("factors", {}),
        indicator_factors=cfg.get("indicator_factors"),
        weights=cfg.get("weights"),
        filters=cfg.get("filters"),
        data_dir=cfg.get("data_dir", "./data"),
    ))

    from .trend_screener import TrendScreener

    register_screener("trend", lambda **cfg: TrendScreener(
        top_n=cfg.get("top_n", 10),
        min_score=cfg.get("min_score", 50),
        market=cfg.get("market", "csi300"),
        benchmark=cfg.get("benchmark", "sh000001"),
        max_concurrent=cfg.get("max_concurrent", 20),
        sma_periods=cfg.get("sma_periods", [50, 150, 200]),
        min_criteria_pass=cfg.get("min_criteria_pass", 7),
    ))


_register_defaults()
