"""趋势类技术指标"""

import pandas as pd
import numpy as np


def ma(close: pd.Series, period: int = 20) -> pd.DataFrame:
    """简单移动平均线"""
    return pd.DataFrame({f'ma_{period}': close.rolling(window=period).mean()})


def ema(close: pd.Series, period: int = 20) -> pd.DataFrame:
    """指数移动平均线"""
    return pd.DataFrame({f'ema_{period}': close.ewm(span=period, adjust=False).mean()})


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """MACD指标

    Returns:
        DataFrame with columns: macd_dif, macd_dea, macd_hist
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    hist = 2 * (dif - dea)

    return pd.DataFrame({
        'macd_dif': dif,
        'macd_dea': dea,
        'macd_hist': hist,
    })


def dema(close: pd.Series, period: int = 20) -> pd.DataFrame:
    """双指数移动平均线 (Double EMA)

    DEMA = 2 * EMA(close, period) - EMA(EMA(close, period), period)
    消除单次 EMA 的滞后。

    Returns:
        DataFrame with column: dema_{period}
    """
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    dema_val = 2 * ema1 - ema2
    return pd.DataFrame({f'dema_{period}': dema_val})


def sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.DataFrame:
    """抛物线指标 (Parabolic SAR)

    追踪趋势方向并提供动态止损位。

    Returns:
        DataFrame with columns: sar_value, sar_trend (1=上升, -1=下降)
    """
    n = len(close)
    sar_val = np.empty(n)
    trend_arr = np.empty(n)

    # 初始化：前两根bar判断方向
    if n < 2:
        sar_val[:] = close.iloc[0] if n > 0 else np.nan
        trend_arr[:] = 1 if n > 0 else 0
        return pd.DataFrame({'sar_value': sar_val, 'sar_trend': trend_arr})

    is_up = close.iloc[1] >= close.iloc[0]
    trend_arr[0] = 1
    trend_arr[1] = 1 if is_up else -1

    af = af_step
    if is_up:
        ep = high.iloc[:2].max()
        sar_val[0] = low.iloc[:2].min()
        sar_val[1] = sar_val[0]
    else:
        ep = low.iloc[:2].min()
        sar_val[0] = high.iloc[:2].max()
        sar_val[1] = sar_val[0]

    for i in range(2, n):
        prev_sar = sar_val[i - 1]
        prev_trend = trend_arr[i - 1]

        # 计算新 SAR
        new_sar = prev_sar + af * (ep - prev_sar)

        if prev_trend == 1:  # 上升趋势
            # SAR 不能高于前两根bar的最低价
            new_sar = min(new_sar, low.iloc[i - 1], low.iloc[i - 2])
            if low.iloc[i] < new_sar:
                # 翻转为下降
                trend_arr[i] = -1
                sar_val[i] = ep
                ep = low.iloc[i]
                af = af_step
            else:
                trend_arr[i] = 1
                sar_val[i] = new_sar
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:  # 下降趋势
            # SAR 不能低于前两根bar的最高价
            new_sar = max(new_sar, high.iloc[i - 1], high.iloc[i - 2])
            if high.iloc[i] > new_sar:
                # 翻转为上升
                trend_arr[i] = 1
                sar_val[i] = ep
                ep = high.iloc[i]
                af = af_step
            else:
                trend_arr[i] = -1
                sar_val[i] = new_sar
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)

    return pd.DataFrame({'sar_value': sar_val, 'sar_trend': trend_arr})
