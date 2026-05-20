"""动量类技术指标"""

import pandas as pd
import numpy as np


def rsi(close: pd.Series, period: int = 14) -> pd.DataFrame:
    """相对强弱指标 (Wilder平滑法)

    Returns:
        DataFrame with column: rsi_{period}
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    # avg_loss=0 时RS为无穷大，RSI=100；avg_gain和avg_loss都为0时RSI=50
    rsi_val = rsi_val.where(avg_loss > 0, 100.0)
    rsi_val = rsi_val.where((avg_gain > 0) | (avg_loss > 0), 50.0)

    return pd.DataFrame({f'rsi_{period}': rsi_val})


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> pd.DataFrame:
    """KDJ随机指标

    Returns:
        DataFrame with columns: k, d, j
    """
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()

    denom = high_n - low_n
    # 分母为0时RSV取50（标准做法）
    rsv = pd.Series(
        np.where(denom == 0, 50, (close - low_n) / denom * 100),
        index=close.index,
    )

    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({'k': k, 'd': d, 'j': j})


def wr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """威廉指标 (Williams %R)

    无平滑的超买超卖振荡器，范围 [-100, 0]。
    低于 -80 为超卖，高于 -20 为超买。

    Returns:
        DataFrame with column: wr_{period}
    """
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    denom = hh - ll
    wr_val = pd.Series(
        np.where(denom == 0, -50.0, (hh - close) / denom * (-100)),
        index=close.index,
    )
    return pd.DataFrame({f'wr_{period}': wr_val})


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """商品通道指标 (Commodity Channel Index)

    衡量价格偏离统计均值的幅度，无界指标。

    Returns:
        DataFrame with column: cci_{period}
    """
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=period).mean()
    tp_mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_val = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
    return pd.DataFrame({f'cci_{period}': cci_val})
