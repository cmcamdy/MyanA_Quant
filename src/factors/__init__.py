"""因子分析模块"""

from typing import Any


def __getattr__(name: str) -> Any:
    _lazy_map = {
        'Factor': '.base',
        'FactorResult': '.base',
        'MomentumFactor': '.factors',
        'VolatilityFactor': '.factors',
        'TurnoverFactor': '.factors',
        'ReversalFactor': '.factors',
        'PriceVolumeFactor': '.factors',
        'BiasFactor': '.factors',
        'FactorAnalyzer': '.analyzer',
        'FactorScreener': '.screener',
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Factor',
    'FactorResult',
    'MomentumFactor',
    'VolatilityFactor',
    'TurnoverFactor',
    'ReversalFactor',
    'PriceVolumeFactor',
    'BiasFactor',
    'FactorAnalyzer',
    'FactorScreener',
]
