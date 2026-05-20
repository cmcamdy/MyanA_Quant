"""波动率类技术指标"""

import pandas as pd
import numpy as np


def bollinger(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """布林带

    Returns:
        DataFrame with columns: boll_mid, boll_upper, boll_lower
    """
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    return pd.DataFrame({
        'boll_mid': mid,
        'boll_upper': upper,
        'boll_lower': lower,
    })


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """平均真实波幅

    Returns:
        DataFrame with column: atr_{period}
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # 首根bar无前收盘价，TR用 high - low
    tr.iloc[0] = high.iloc[0] - low.iloc[0]

    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({f'atr_{period}': atr_val})


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    num_atr: float = 1.5,
) -> pd.DataFrame:
    """肯特纳通道 (Keltner Channel)

    Mid = EMA(close), Upper/Lower = Mid ± num_atr * ATR
    与布林带配合可识别 squeeze 形态。

    Returns:
        DataFrame with columns: kelt_mid, kelt_upper, kelt_lower
    """
    mid = close.ewm(span=ema_period, adjust=False).mean()
    atr_df = atr(high, low, close, period=atr_period)
    atr_val = atr_df.iloc[:, 0]
    upper = mid + num_atr * atr_val
    lower = mid - num_atr * atr_val

    return pd.DataFrame({
        'kelt_mid': mid,
        'kelt_upper': upper,
        'kelt_lower': lower,
    })


def chaikin_vol(
    high: pd.Series,
    low: pd.Series,
    period: int = 10,
    roc_period: int = 10,
) -> pd.DataFrame:
    """佳庆波动率 (Chaikin Volatility)

    衡量高低价差（波动率）的变化速率。
    低值表示波动率收缩（可能突破），高值表示波动率扩张。

    Returns:
        DataFrame with column: chaikin_vol_{period}
    """
    hl_spread = high - low
    hl_sma = hl_spread.rolling(window=period).mean()
    hl_sma_prev = hl_sma.shift(roc_period)
    cv = (hl_sma - hl_sma_prev) / hl_sma_prev.replace(0, np.nan) * 100
    return pd.DataFrame({f'chaikin_vol_{period}': cv})
