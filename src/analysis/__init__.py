"""技术分析模块"""

from .indicator_set import IndicatorSet

__all__ = ['IndicatorSet']


def __getattr__(name):
    _indicator_map = {
        'ma': '.indicators.trend',
        'ema': '.indicators.trend',
        'macd': '.indicators.trend',
        'dema': '.indicators.trend',
        'sar': '.indicators.trend',
        'rsi': '.indicators.momentum',
        'kdj': '.indicators.momentum',
        'wr': '.indicators.momentum',
        'cci': '.indicators.momentum',
        'bollinger': '.indicators.volatility',
        'atr': '.indicators.volatility',
        'keltner': '.indicators.volatility',
        'chaikin_vol': '.indicators.volatility',
        'vwap': '.indicators.volume',
        'obv': '.indicators.volume',
        'mfi': '.indicators.volume',
    }
    if name in _indicator_map:
        import importlib
        module = importlib.import_module(_indicator_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
