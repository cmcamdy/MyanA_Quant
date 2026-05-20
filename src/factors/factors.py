"""具体因子实现"""

import numpy as np
import pandas as pd

from .base import Factor


class MomentumFactor(Factor):
    """动量因子：过去 period 日的价格变化率

    公式: (close - close.shift(period)) / close.shift(period)
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        shifted = close.shift(self.period)
        result = (close - shifted) / shifted
        result.name = f'momentum_{self.period}'
        return result


class VolatilityFactor(Factor):
    """波动率因子：过去 period 日的收益率滚动标准差

    公式: close.pct_change().rolling(period).std()
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        result = close.pct_change().rolling(self.period).std()
        result.name = f'volatility_{self.period}'
        return result


class TurnoverFactor(Factor):
    """换手率因子：过去 period 日的平均成交量相对变化

    公式: volume.rolling(period).mean() / volume.rolling(period).mean().shift(1)
    处理除零：分母为零时结果为 NaN
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = df['volume']
        rolling_mean = volume.rolling(self.period).mean()
        shifted_mean = rolling_mean.shift(1)
        # 处理除零：分母为零或接近零时设为 NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            result = rolling_mean / shifted_mean
        result = result.where(shifted_mean != 0, np.nan)
        result.name = f'turnover_{self.period}'
        return result


class ReversalFactor(Factor):
    """反转因子：短期价格变化的相反数

    公式: -close.pct_change(period)
    """

    def __init__(self, period: int = 5) -> None:
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        result = -close.pct_change(self.period)
        result.name = f'reversal_{self.period}'
        return result


class PriceVolumeFactor(Factor):
    """价量相关因子：过去 period 日价格变化率与成交量变化率的滚动相关系数

    公式: close.pct_change() 与 volume.pct_change() 的 rolling(period).corr()
    """

    def __init__(self, period: int = 10) -> None:
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        volume = df['volume']
        price_chg = close.pct_change()
        vol_chg = volume.pct_change()
        result = price_chg.rolling(self.period).corr(vol_chg)
        result.name = f'price_volume_{self.period}'
        return result


class BiasFactor(Factor):
    """乖离率因子：价格偏离均线的程度

    公式: (close - close.rolling(period).mean()) / close.rolling(period).mean()
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        ma = close.rolling(self.period).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (close - ma) / ma
        result = result.where(ma != 0, np.nan)
        result.name = f'bias_{self.period}'
        return result
