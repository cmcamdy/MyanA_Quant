"""成交量类技术指标"""

import pandas as pd
import numpy as np


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """成交量加权平均价

    日内数据按日重置累计值；日线数据为典型价格加权。

    Returns:
        DataFrame with column: vwap
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    if isinstance(tp_vol.index, pd.DatetimeIndex):
        # 按日分组累计
        dates = tp_vol.index.date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = volume.cumsum()

    vwap_val = cum_tp_vol / cum_vol.replace(0, np.nan)

    return pd.DataFrame({'vwap': vwap_val})


def obv(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """能量潮指标 (On-Balance Volume)

    涨加量、跌减量、平不变，反映资金流向。

    Returns:
        DataFrame with column: obv
    """
    direction = np.sign(close.diff()).fillna(0.0)
    obv_val = (direction * volume).cumsum()
    return pd.DataFrame({'obv': obv_val})


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """资金流量指标 (Money Flow Index)

    成交量版 RSI，量价结合的动量振荡器，范围 [0, 100]。

    Returns:
        DataFrame with column: mfi_{period}
    """
    tp = (high + low + close) / 3
    mf = tp * volume

    delta_tp = tp.diff()
    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)

    pos_sum = pos_mf.rolling(window=period, min_periods=1).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=1).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi_val = 100 - 100 / (1 + mfr)
    mfi_val = mfi_val.where(neg_sum > 0, 100.0)
    mfi_val = mfi_val.where((pos_sum > 0) | (neg_sum > 0), 50.0)

    return pd.DataFrame({f'mfi_{period}': mfi_val})
