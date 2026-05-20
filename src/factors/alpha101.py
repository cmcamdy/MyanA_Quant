"""Alpha101 因子实现

基于 WorldQuant Alpha101 论文，选取与 myana-quant 现有因子互补的高 IC 因子。
所有因子仅需 OHLCV + amount 数据，无额外数据源依赖。

关于截面排名 vs 时序排名
────────────────────────
Alpha101 原始公式中的 rank() 是截面排名（cross-sectional rank），
即同一时刻对所有股票取排名。本模块中因子按单只股票计算，因此用
时间序列排名 ts_rank() 替代。截面标准化可在选股层面（FactorScreener
或 StockScreener）统一处理——这也是 aurumq-rl 等成熟框架的标准做法。

因子分类
────────
波动率类:  Alpha001Factor, SkewReversalFactor, KurtFilterFactor
价量类:    Alpha005Factor, Alpha014Factor, Alpha015Factor
突破类:    Alpha023Factor, Alpha054Factor
动量类:    Alpha084Factor, DecayLinearMomFactor
反转类:    Alpha033Factor, Alpha041Factor, ZscoreReversalFactor
"""

import numpy as np
import pandas as pd

from .base import Factor


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """时间序列排名：当前值在过去 window 期中的百分位排名 (0~1)

    等价于 Alpha101 中的 ts_rank / rank 操作在单股票上的时序版本。
    值为 1.0 表示当前值是窗口内最大值，0.5 表示中位数，0.0 表示最小值。
    """
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    """线性衰减加权移动平均

    权重从最新到最旧线性递减：[1, 2, ..., window] / sum(weights)
    越近的数据权重越大，比等权移动平均更能捕捉近期趋势变化。
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


# ─────────────────────────────────────────────
# 波动率类因子
# ─────────────────────────────────────────────

class Alpha001Factor(Factor):
    """Alpha001 — 波动率异常因子

    公式
        ts_rank( std(close - ts_mean(close, mean_window)), rank_window )

    含义
        收盘价偏离短期均线的标准差，在较长窗口中的百分位排名。
        高值表示近期波动率异常放大，可能预示趋势转折或加速。

    与 VolatilityFactor 的关系
        VolatilityFactor 是绝对波动率（收益率标准差），
        本因子关注波动率的相对变化（是否处于近期高位），
        对横盘后突破的捕捉更敏感。
    """

    def __init__(self, mean_window: int = 5, rank_window: int = 20) -> None:
        self.mean_window = mean_window
        self.rank_window = rank_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        deviation = close - close.rolling(self.mean_window).mean()
        vol_of_deviation = deviation.rolling(self.rank_window).std()
        result = _ts_rank(vol_of_deviation, self.rank_window)
        result.name = f'alpha001_{self.mean_window}_{self.rank_window}'
        return result


class SkewReversalFactor(Factor):
    """偏度反转因子

    公式
        -ts_rank( skew(returns, window), rank_window )

    含义
        收益率分布正偏度（右偏 = 少数大阳线 + 多数小阴线）通常出现在
        上涨末期；负号取反意味着高偏度 → 低因子值 → 看跌。
        学术研究表明 A 股偏度反转效应显著（Greene & Smart, 1999）。

    与 VolatilityFactor 的关系
        波动率是二阶矩，偏度是三阶矩（不对称性）。
        同样 20% 波动率的股票，正偏和负偏的后续走势可能截然不同。
    """

    def __init__(self, window: int = 20, rank_window: int = 20) -> None:
        self.window = window
        self.rank_window = rank_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change()
        skew = returns.rolling(self.window).skew()
        result = -_ts_rank(skew, self.rank_window)
        result.name = f'skew_reversal_{self.window}'
        return result


class KurtFilterFactor(Factor):
    """峰度过滤因子

    公式
        -ts_rank( kurtosis(returns, window), rank_window )

    含义
        高峰度表示极端收益事件频繁（厚尾），统计上预示后续波动率放大。
        负号取反：高峰度 → 低因子值 → 过滤掉尾部风险大的标的。

    三因子互补
        VolatilityFactor (二阶矩) + SkewReversalFactor (三阶矩)
        + KurtFilterFactor (四阶矩) = 完整的收益率分布刻画。
    """

    def __init__(self, window: int = 20, rank_window: int = 20) -> None:
        self.window = window
        self.rank_window = rank_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change()
        kurt = returns.rolling(self.window).kurt()
        result = -_ts_rank(kurt, self.rank_window)
        result.name = f'kurt_filter_{self.window}'
        return result


# ─────────────────────────────────────────────
# 价量类因子
# ─────────────────────────────────────────────

class Alpha005Factor(Factor):
    """Alpha005 — 量价趋势因子

    公式
        ts_rank(close * volume, window) 的 window 日变化量

    含义
        成交金额（close * volume ≈ amount）的时序排名变化。
        排名上升表示资金加速流入，排名下降表示资金撤离。

    与 PriceVolumeFactor 的关系
        PriceVolumeFactor 只度量价量的相关系数（方向一致性），
        本因子直接度量资金流的趋势方向和强度。
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        volume = df['volume']
        cv_rank = _ts_rank(close * volume, self.window)
        result = cv_rank - cv_rank.shift(self.window)
        result.name = f'alpha005_{self.window}'
        return result


class Alpha014Factor(Factor):
    """Alpha014 — 开盘收盘反转因子

    公式
        corr(close - open, close - delay(close), window)

    含义
        日内涨幅（close - open）与隔夜+日内总涨幅（close - prev_close）
        的滚动相关性。正相关说明趋势延续（日内方向与隔夜一致），
        负相关说明趋势反转。

    与 ReversalFactor 的关系
        ReversalFactor 只看收盘价变化（-pct_change），
        本因子区分日内波动与隔夜跳空，能捕捉日内反转模式。
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        open_ = df['open']
        intraday = close - open_            # 日内涨跌
        total_chg = close - close.shift(1)  # 隔夜 + 日内总涨跌
        result = intraday.rolling(self.window).corr(total_chg)
        result.name = f'alpha014_{self.window}'
        return result


class Alpha015Factor(Factor):
    """Alpha015 — 日内效率因子

    公式
        sum(close - open, window) / sum(high - low, window)

    含义
        日内净涨幅占总振幅的比例。值接近 1 表示全天单边上涨，
        值接近 -1 表示单边下跌，值接近 0 表示多空拉锯。
        高效率 = 趋势明确，低效率 = 震荡无方向——衡量「趋势质量」。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        numerator = (close - open_).rolling(self.window).sum()
        denominator = (high - low).rolling(self.window).sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
        result = result.where(denominator != 0, np.nan)
        result.name = f'alpha015_{self.window}'
        return result


# ─────────────────────────────────────────────
# 突破类因子
# ─────────────────────────────────────────────

class Alpha023Factor(Factor):
    """Alpha023 — 新高突破因子

    公式
        close >= rolling_max(close, window) → 1.0, 否则 0.0

    含义
        当日收盘价是否创 window 日新高。突破信号是趋势启动的经典确认，
        但单独使用胜率不高，需配合量能或其他因子过滤假突破。

    与 MomentumFactor 的关系
        动量是连续信号（涨幅大小），突破是离散事件信号（是否创新高）。
        创新高但动量弱的股票可能是假突破，两个因子配合使用效果更好。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        rolling_max = close.rolling(self.window).max()
        result = (close >= rolling_max).astype(float)
        result.name = f'alpha023_{self.window}'
        return result


class Alpha054Factor(Factor):
    """Alpha054 — OBV 动量交叉因子

    公式
        ts_rank(OBV - ts_mean(OBV, obv_ma_window), rank_window)

    含义
        OBV（能量潮指标）偏离其均值的程度在时间窗口中的百分位排名。
        OBV 持续高于均值且排名上升 → 资金持续流入 → 看多；
        OBV 持续低于均值且排名下降 → 资金持续流出 → 看空。

    与 obv_direction 的关系
        StockScreener 中的 obv_direction 只看方向（+1/-1/0），
        本因子度量 OBV 偏离的强度和趋势，信息量更丰富。
    """

    def __init__(self, obv_ma_window: int = 20, rank_window: int = 20) -> None:
        self.obv_ma_window = obv_ma_window
        self.rank_window = rank_window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        volume = df['volume']
        # OBV: 上涨日累加成交量，下跌日累减成交量
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
        deviation = obv - obv.rolling(self.obv_ma_window).mean()
        result = _ts_rank(deviation, self.rank_window)
        result.name = f'alpha054_{self.obv_ma_window}'
        return result


# ─────────────────────────────────────────────
# 动量类因子
# ─────────────────────────────────────────────

class Alpha084Factor(Factor):
    """Alpha084 — 上涨胜率因子

    公式
        ts_mean(close > delay(close), window)
        即过去 window 日中上涨日的占比

    含义
        0.7 表示 70% 的交易日收阳。高胜率 = 强势趋势，低胜率 = 弱势或震荡。

    与 MomentumFactor 的关系
        动量度量涨幅大小，胜率度量方向一致性。
        一个涨幅大但胜率低的股票可能是被少数大阳线拉动，趋势不稳健；
        胜率高但涨幅小则是小步慢涨，趋势更可持续。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        up = (close > close.shift(1)).astype(float)
        result = up.rolling(self.window).mean()
        result.name = f'alpha084_{self.window}'
        return result


class DecayLinearMomFactor(Factor):
    """衰减线性动量因子

    公式
        decay_linear(close / delay(close) - 1, window)
        即对日收益率做线性衰减加权平均

    含义
        越近的交易日权重越大，能更快响应趋势变化。
        在趋势加速阶段表现优于等权动量。

    与 MomentumFactor 的关系
        MomentumFactor = (P_t - P_{t-n}) / P_{t-n}，等价于等权动量。
        本因子对近期收益赋予更高权重，可以理解为「最近比过去更重要」的动量。
        当近几天涨幅突然加速时，本因子会比 MomentumFactor 更快上升。
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        daily_return = close.pct_change()
        result = _decay_linear(daily_return, self.window)
        result.name = f'decay_linear_mom_{self.window}'
        return result


# ─────────────────────────────────────────────
# 反转类因子
# ─────────────────────────────────────────────

class Alpha033Factor(Factor):
    """Alpha033 — 下影线反转因子

    公式
        ts_rank((low - close) / (high - low), window)

    含义
        下影线长度占振幅比例的时序排名。
        长下影线（低排名）= 收盘接近最低价，空方主导；
        短下影线 / 无下影线（高排名）= 收盘远离最低价，下方有支撑。
        极端低值（收盘紧贴最低价）在短期内倾向于反转上涨。

    注意
        当 high == low（一字板）时分母为 0，此时因子值为 NaN。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        high = df['high']
        low = df['low']
        denominator = high - low
        with np.errstate(divide='ignore', invalid='ignore'):
            lower_shadow = (low - close) / denominator
        lower_shadow = lower_shadow.where(denominator > 1e-8, np.nan)
        result = _ts_rank(lower_shadow, self.window)
        result.name = f'alpha033_{self.window}'
        return result


class Alpha041Factor(Factor):
    """Alpha041 — 缩量反转因子

    公式
        ts_rank((close - delay(close))^2, window) * ts_rank(-volume, window)

    含义
        价格变化幅度大（高 rank）且成交量缩小（对 volume 取负 → 低 rank 看多）
        时因子值高。缩量大涨或缩量急跌后的反转尤其值得关注——
        缩量说明市场参与度低，卖方力量已经衰竭或买方尚未全面进场。

    典型场景
        缩量急跌后出现缩量小阳线 → 卖方衰竭 → 因子值升高 → 反弹信号。
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        volume = df['volume']
        price_move_sq = (close - close.shift(1)) ** 2
        result = _ts_rank(price_move_sq, self.window) * _ts_rank(-volume, self.window)
        result.name = f'alpha041_{self.window}'
        return result


class ZscoreReversalFactor(Factor):
    """短期 Z-Score 反转因子

    公式
        -(close - ts_mean(close, window)) / ts_std(close, window)

    含义
        当前价格偏离短期均值的标准差倍数，取负号。
        因子值高 = 价格显著低于均值（超卖）→ 看多反弹；
        因子值低 = 价格显著高于均值（超买）→ 看空回落。

    与 ReversalFactor 的关系
        ReversalFactor = -pct_change(period)，是简单反转。
        本因子用 z-score 标准化后衡量偏离程度，在不同波动率环境下可比：
        同样 3% 的跌幅，在低波动股票是显著超卖，在高波动股票只是正常波动。
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        with np.errstate(divide='ignore', invalid='ignore'):
            result = -(close - mean) / std
        result = result.where(std > 1e-8, np.nan)
        result.name = f'zscore_reversal_{self.window}'
        return result
