"""因子分析模块"""

from typing import Any


def __getattr__(name: str) -> Any:
    _lazy_map = {
        # 基础因子
        'Factor': '.base',
        'FactorResult': '.base',
        'MomentumFactor': '.factors',
        'VolatilityFactor': '.factors',
        'TurnoverFactor': '.factors',
        'ReversalFactor': '.factors',
        'PriceVolumeFactor': '.factors',
        'BiasFactor': '.factors',
        # Alpha101 因子
        'Alpha001Factor': '.alpha101',
        'SkewReversalFactor': '.alpha101',
        'KurtFilterFactor': '.alpha101',
        'Alpha005Factor': '.alpha101',
        'Alpha014Factor': '.alpha101',
        'Alpha015Factor': '.alpha101',
        'Alpha023Factor': '.alpha101',
        'Alpha054Factor': '.alpha101',
        'Alpha084Factor': '.alpha101',
        'DecayLinearMomFactor': '.alpha101',
        'Alpha033Factor': '.alpha101',
        'Alpha041Factor': '.alpha101',
        'ZscoreReversalFactor': '.alpha101',
        # 基础设施
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
    # 基础因子
    'MomentumFactor',
    'VolatilityFactor',
    'TurnoverFactor',
    'ReversalFactor',
    'PriceVolumeFactor',
    'BiasFactor',
    # Alpha101 因子 — 波动率类
    'Alpha001Factor',
    'SkewReversalFactor',
    'KurtFilterFactor',
    # Alpha101 因子 — 价量类
    'Alpha005Factor',
    'Alpha014Factor',
    'Alpha015Factor',
    # Alpha101 因子 — 突破类
    'Alpha023Factor',
    'Alpha054Factor',
    # Alpha101 因子 — 动量类
    'Alpha084Factor',
    'DecayLinearMomFactor',
    # Alpha101 因子 — 反转类
    'Alpha033Factor',
    'Alpha041Factor',
    'ZscoreReversalFactor',
    # 基础设施
    'FactorAnalyzer',
    'FactorScreener',
]
