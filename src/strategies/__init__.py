"""策略测试模块"""

from .base import Strategy, Signal, SignalType, Context, Portfolio, Position
from .engine import BacktestEngine
from .result import BacktestResult

__all__ = [
    'Strategy',
    'Signal',
    'SignalType',
    'Context',
    'Portfolio',
    'Position',
    'BacktestEngine',
    'BacktestResult',
]


def __getattr__(name):
    _lazy_map = {
        'BacktestConfig': '.engine',
        'FixedSizer': '.sizers',
        'AllInSizer': '.sizers',
        'CompositeStrategy': '.composite',
        'MACrossStrategy': '.examples.ma_cross',
        'TradeRecord': '.result',
        'calc_performance_metrics': '.result',
        'RiskConfig': '.risk',
        'RiskManager': '.risk',
        'PortfolioEngine': '.portfolio_engine',
        'PortfolioContext': '.portfolio_engine',
        'MultiStrategy': '.portfolio_engine',
        'EqualWeightAllocation': '.portfolio_engine',
        'CustomAllocation': '.portfolio_engine',
        'RebalanceConfig': '.portfolio_engine',
        'Optimizer': '.optimizer',
        'ParamRange': '.optimizer',
        'OptimizationResult': '.optimizer',
        'sharpe_objective': '.optimizer',
        'total_return_objective': '.optimizer',
        'results_to_dataframe': '.optimizer',
        'GeneticOptimizer': '.optimizer',
        'BayesianOptimizer': '.optimizer',
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
