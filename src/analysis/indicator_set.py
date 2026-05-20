"""指标批量计算器"""

from typing import List, Dict, Any
import pandas as pd

from .indicators.trend import ma as _ma, ema as _ema, macd as _macd, dema as _dema, sar as _sar
from .indicators.momentum import rsi as _rsi, kdj as _kdj, wr as _wr, cci as _cci
from .indicators.volatility import bollinger as _bollinger, atr as _atr, keltner as _keltner, chaikin_vol as _chaikin_vol
from .indicators.volume import vwap as _vwap, obv as _obv, mfi as _mfi


# 指标名 → (函数, 需要的列)
_INDICATOR_REGISTRY = {
    'ma': (_ma, ['close']),
    'ema': (_ema, ['close']),
    'macd': (_macd, ['close']),
    'dema': (_dema, ['close']),
    'sar': (_sar, ['high', 'low', 'close']),
    'rsi': (_rsi, ['close']),
    'kdj': (_kdj, ['high', 'low', 'close']),
    'wr': (_wr, ['high', 'low', 'close']),
    'cci': (_cci, ['high', 'low', 'close']),
    'bollinger': (_bollinger, ['close']),
    'atr': (_atr, ['high', 'low', 'close']),
    'keltner': (_keltner, ['high', 'low', 'close']),
    'chaikin_vol': (_chaikin_vol, ['high', 'low']),
    'vwap': (_vwap, ['high', 'low', 'close', 'volume']),
    'obv': (_obv, ['close', 'volume']),
    'mfi': (_mfi, ['high', 'low', 'close', 'volume']),
}


class IndicatorSet:
    """批量计算多个技术指标

    Usage:
        iset = IndicatorSet()
        iset.add('ma', period=5).add('ma', period=20).add('macd')
        result = iset.compute(df)
    """

    def __init__(self):
        self._specs: List[Dict[str, Any]] = []

    def add(self, name: str, **params) -> 'IndicatorSet':
        """添加指标到计算列表，支持链式调用"""
        if name not in _INDICATOR_REGISTRY:
            raise ValueError(f"未知指标: {name}，可选: {list(_INDICATOR_REGISTRY.keys())}")
        self._specs.append({'name': name, **params})
        return self

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标，返回原始列+指标列"""
        result_parts = [df.copy()]
        computed_keys = set()

        for spec in self._specs:
            name = spec['name']
            params = {k: v for k, v in spec.items() if k != 'name'}

            # 去重：相同指标+参数只计算一次
            key = str(spec)
            if key in computed_keys:
                continue
            computed_keys.add(key)

            func, required_cols = _INDICATOR_REGISTRY[name]
            series_args = {col: df[col] for col in required_cols}
            indicator_df = func(**series_args, **params)
            # 确保指标结果与原始DataFrame对齐索引
            indicator_df.index = df.index
            result_parts.append(indicator_df)

        return pd.concat(result_parts, axis=1)

    @property
    def specs(self) -> List[Dict[str, Any]]:
        return list(self._specs)

    def clear(self) -> 'IndicatorSet':
        self._specs.clear()
        return self
