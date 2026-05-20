"""Minervini Stage 2 趋势评分引擎

核心组件:
  - Phase 分类器 (4阶段: Base/Uptrend/Distribution/Downtrend)
  - Minervini 8条件趋势模板验证 (7/8通过)
  - VCP (波动收缩形态) 检测
  - 相对强度计算 (vs 基准指数)
  - 成交量突破检测
  - ATR 止损 + 仓位计算
  - 线性评分引擎

参考: https://github.com/RyanJHamby/stock-screener
      Mark Minervini 《Trade Like a Stock Market Wizard》
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class TrendScoreResult:
    """趋势评分结果"""
    symbol: str
    name: str
    market: str
    price: float
    total_score: int          # 0-100 综合分

    # Phase
    phase: str = ""           # Base / Uptrend / Distribution / Downtrend

    # Minervini 趋势模板
    criteria_passed: int = 0
    criteria_total: int = 8
    criteria_details: Dict[str, bool] = field(default_factory=dict)

    # VCP
    vcp_detected: bool = False
    vcp_contractions: int = 0
    vcp_quality: int = 0      # 0-100

    # 相对强度
    relative_strength: float = 0.0   # 0-10

    # 成交量
    volume_breakout: bool = False
    volume_ratio: float = 0.0

    # SMA
    sma_50: Optional[float] = None
    sma_150: Optional[float] = None
    sma_200: Optional[float] = None

    # ATR 止损
    atr_stop_loss: float = 0.0
    position_size_pct: float = 0.0

    # 信号
    buy_signal: bool = False
    sell_signal: bool = False


# ---------------------------------------------------------------------------
# Phase 分类
# ---------------------------------------------------------------------------

def classify_phase(closes: np.ndarray, sma_50: float, sma_150: float,
                   sma_200: float, sma_50_slope: float,
                   sma_200_slope: float) -> str:
    """4阶段分类

    优先级: Downtroke > Uptrend > Distribution > Base

    Returns:
        "Base" / "Uptrend" / "Distribution" / "Downtrend"
    """
    price = closes[-1]
    empty = any(v is None or v == 0 for v in [sma_50, sma_150, sma_200])
    if empty or len(closes) < 10:
        return "Base"

    # Phase 4: 下降趋势
    if price < sma_50 and price < sma_200 and sma_50 < sma_200:
        return "Downtrend"

    # Phase 2: 上升趋势
    if (price > sma_50 > sma_150 > sma_200
            and sma_50_slope > 0 and sma_200_slope > 0):
        return "Uptrend"

    # Phase 3: 派发 (价格过度偏离)
    if sma_50 > sma_200 and price > sma_50 * 1.25:
        return "Distribution"

    return "Base"


# ---------------------------------------------------------------------------
# Minervini 趋势模板
# ---------------------------------------------------------------------------

def check_minervini_template(
    price: float,
    sma_50: float,
    sma_150: float,
    sma_200: float,
    sma_200_slope: float,
    high_52w: float,
    low_52w: float,
    phase: str,
) -> Dict[str, bool]:
    """Minervini 8条件趋势模板验证

    7/8 通过视为趋势确认。

    Returns:
        {criterion_name: passed}
    """
    criteria = {}

    # 1. 价格 > 150 SMA 且 > 200 SMA
    criteria["price_above_long_sma"] = price > sma_150 and price > sma_200

    # 2. 150 SMA > 200 SMA
    criteria["sma150_above_sma200"] = sma_150 > sma_200

    # 3. 200 SMA 至少1个月趋势向上
    criteria["sma200_trending_up"] = sma_200_slope > 0

    # 4. 50 SMA > 150 SMA (级联排列)
    criteria["sma50_above_sma150"] = sma_50 > sma_150

    # 5. 价格 > 50 SMA
    criteria["price_above_sma50"] = price > sma_50

    # 6. 价格 >= 52周最低的 130%
    criteria["price_above_30pct_low"] = (low_52w > 0
                                         and price >= low_52w * 1.30)

    # 7. 价格在52周最高点的 75% 以内
    criteria["price_within_25pct_high"] = (high_52w > 0
                                           and price >= high_52w * 0.75)

    # 8. 当前处于 Phase 2
    criteria["in_phase2"] = phase == "Uptrend"

    return criteria


# ---------------------------------------------------------------------------
# VCP 波动收缩形态
# ---------------------------------------------------------------------------

def detect_vcp(closes: np.ndarray, volumes: np.ndarray,
               window: int = 60) -> Tuple[bool, int, int]:
    """检测 VCP (Volatility Contraction Pattern)

    VCP 特征: 连续缩量回撤，每次回撤幅度小于前一次。

    Args:
        closes: 收盘价序列
        volumes: 成交量序列
        window: 检测窗口 (交易日)

    Returns:
        (detected, contraction_count, quality 0-100)
    """
    if len(closes) < window:
        return False, 0, 0

    recent_closes = closes[-window:]
    recent_volumes = volumes[-window:] if len(volumes) >= window else np.array([])

    # 寻找波动峰谷
    peaks = []
    troughs = []
    for i in range(1, len(recent_closes) - 1):
        if recent_closes[i] > recent_closes[i - 1] and recent_closes[i] > recent_closes[i + 1]:
            peaks.append((i, recent_closes[i]))
        elif recent_closes[i] < recent_closes[i - 1] and recent_closes[i] < recent_closes[i + 1]:
            troughs.append((i, recent_closes[i]))

    if len(troughs) < 2:
        return False, 0, 0

    # 计算每次回撤幅度 (从最近的高点到谷底)
    contractions = []
    for trough_idx, trough_price in troughs:
        # 找这个谷底之前的最近高点
        prev_peaks = [(pi, pp) for pi, pp in peaks if pi < trough_idx]
        if prev_peaks:
            _, peak_price = prev_peaks[-1]
            drawdown = (peak_price - trough_price) / peak_price * 100
            if drawdown > 1:  # 至少1%回撤才算
                contractions.append(drawdown)

    if len(contractions) < 2:
        return False, 0, 0

    # VCP条件: 回撤幅度逐步缩小
    shrinking = all(contractions[i] > contractions[i + 1]
                    for i in range(len(contractions) - 1))

    if not shrinking:
        return False, len(contractions), 0

    # 缩量确认: 后半段成交量低于前半段
    volume_dry = False
    if len(recent_volumes) >= window:
        vol_first_half = np.mean(recent_volumes[:window // 2])
        vol_second_half = np.mean(recent_volumes[window // 2:])
        volume_dry = vol_second_half < vol_first_half * 0.9

    # 评分
    quality = min(100, len(contractions) * 25)
    if volume_dry:
        quality = min(100, quality + 20)
    # 接近52周高点加分
    if closes[-1] >= np.max(recent_closes) * 0.95:
        quality = min(100, quality + 15)

    return True, len(contractions), quality


# ---------------------------------------------------------------------------
# 相对强度
# ---------------------------------------------------------------------------

def compute_relative_strength(closes: np.ndarray,
                              benchmark_closes: np.ndarray) -> float:
    """计算相对强度 (0-10分)

    RS = (stock_price / benchmark_price) 的63日线性回归斜率，
    映射到 0-10 分: 斜率0.3→10分, 0.0→5分, -0.3→0分

    Args:
        closes: 股票收盘价
        benchmark_closes: 基准指数收盘价

    Returns:
        RS 分数 (0-10)
    """
    period = min(63, len(closes) - 1, len(benchmark_closes) - 1)
    if period < 10:
        return 5.0  # 数据不足给中性分

    stock_recent = closes[-period - 1:]
    bench_recent = benchmark_closes[-period - 1:]

    # 计算价格比率
    rs_ratio = stock_recent / bench_recent

    # 线性回归斜率
    x = np.arange(len(rs_ratio), dtype=float)
    y = rs_ratio
    if np.std(y) == 0:
        return 5.0

    slope = np.polyfit(x, y, 1)[0]
    # 归一化斜率
    normalized_slope = slope / np.mean(rs_ratio) * len(rs_ratio)

    # 映射到0-10
    rs_score = 5.0 + normalized_slope * 16.67
    return round(max(0.0, min(10.0, rs_score)), 2)


# ---------------------------------------------------------------------------
# 成交量突破
# ---------------------------------------------------------------------------

def detect_volume_breakout(volumes: np.ndarray,
                           avg_period: int = 20,
                           threshold: float = 1.5) -> Tuple[bool, float]:
    """检测成交量突破

    Args:
        volumes: 成交量序列
        avg_period: 均量计算周期
        threshold: 突破阈值 (当前量 >= 均量的多少倍)

    Returns:
        (breakout, volume_ratio)
    """
    if len(volumes) < avg_period + 1:
        return False, 0.0

    avg_vol = np.mean(volumes[-(avg_period + 1):-1])
    current_vol = volumes[-1]

    if avg_vol <= 0:
        return False, 0.0

    ratio = current_vol / avg_vol
    return ratio >= threshold, round(ratio, 2)


# ---------------------------------------------------------------------------
# ATR 止损 + 仓位
# ---------------------------------------------------------------------------

def compute_atr_stop(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14,
                     multiplier: float = 2.0,
                     risk_per_trade: float = 0.01) -> Tuple[float, float]:
    """计算 ATR 止损价位和仓位比例

    Args:
        highs/lowers/closes: 价格序列
        period: ATR 周期
        multiplier: ATR 止损倍数
        risk_per_trade: 单笔风险占总资金比例

    Returns:
        (stop_loss_price, position_size_pct)
    """
    if len(closes) < period + 1:
        return 0.0, 0.0

    # True Range
    prev_closes = closes[-(period + 1):-1]
    curr_highs = highs[-period:]
    curr_lows = lows[-period:]

    tr = np.maximum(
        curr_highs - curr_lows,
        np.maximum(
            np.abs(curr_highs - prev_closes),
            np.abs(curr_lows - prev_closes),
        ),
    )
    atr = np.mean(tr)

    price = closes[-1]
    stop_loss = price - multiplier * atr

    # 仓位 = 单笔风险 / (价格 - 止损价)
    if price > stop_loss > 0:
        risk_per_share = price - stop_loss
        position_pct = (risk_per_trade * price) / risk_per_share * 100
        position_pct = min(position_pct, 100.0)
    else:
        position_pct = 0.0

    return round(stop_loss, 2), round(position_pct, 2)


# ---------------------------------------------------------------------------
# 评分引擎
# ---------------------------------------------------------------------------

class TrendScorer:
    """Minervini Stage 2 趋势评分器

    Args:
        sma_periods: SMA 周期列表
        min_criteria_pass: 趋势模板最少通过条件数
        atr_period: ATR 周期
        atr_multiplier: ATR 止损倍数
        risk_per_trade: 单笔风险比例
    """

    def __init__(
        self,
        sma_periods: Optional[List[int]] = None,
        min_criteria_pass: int = 7,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.sma_periods = sma_periods or [50, 150, 200]
        self.min_criteria_pass = min_criteria_pass
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    def score_batch(
        self,
        stocks: List[Dict],
        history: Dict[str, pd.DataFrame],
        benchmark_closes: Optional[np.ndarray] = None,
    ) -> List[TrendScoreResult]:
        """批量评分"""
        return [
            self.calculate_score(s, history, benchmark_closes)
            for s in stocks
        ]

    def calculate_score(
        self,
        stock: Dict,
        history: Dict[str, pd.DataFrame],
        benchmark_closes: Optional[np.ndarray] = None,
    ) -> TrendScoreResult:
        """计算单只股票趋势评分"""
        code = stock.get("code", "")
        name = stock.get("name", "")
        market = stock.get("market", "A")
        price = stock.get("price", 0) or 0

        # 基础结果
        result = TrendScoreResult(
            symbol=code,
            name=name,
            market=market,
            price=price,
            total_score=0,
        )

        # 获取 OHLCV 数据
        ohlcv = history.get(code)
        if ohlcv is None or len(ohlcv) < 30:
            return result

        closes = ohlcv["close"].values.astype(float)
        highs = ohlcv["high"].values.astype(float)
        lows = ohlcv["low"].values.astype(float)
        volumes = ohlcv["volume"].values.astype(float)

        # ---- SMA 计算 ----
        sma_vals = {}
        sma_slopes = {}
        for period in self.sma_periods:
            if len(closes) >= period:
                sma = np.mean(closes[-period:])
                sma_vals[period] = sma
                # 5日斜率 (百分比/日)
                if len(closes) >= period + 5:
                    sma_5d_ago = np.mean(closes[-(period + 5):-5])
                    sma_slopes[period] = (sma - sma_5d_ago) / sma_5d_ago / 5 if sma_5d_ago > 0 else 0
                else:
                    sma_slopes[period] = 0
            else:
                sma_vals[period] = None
                sma_slopes[period] = 0

        result.sma_50 = sma_vals.get(50)
        result.sma_150 = sma_vals.get(150)
        result.sma_200 = sma_vals.get(200)

        # ---- Phase 分类 ----
        result.phase = classify_phase(
            closes,
            sma_vals.get(50, 0) or 0,
            sma_vals.get(150, 0) or 0,
            sma_vals.get(200, 0) or 0,
            sma_slopes.get(50, 0),
            sma_slopes.get(200, 0),
        )

        # ---- Minervini 趋势模板 ----
        high_52w = np.max(closes) if len(closes) > 0 else 0
        low_52w = np.min(closes) if len(closes) > 0 else 0

        criteria = check_minervini_template(
            price=price,
            sma_50=sma_vals.get(50, 0) or 0,
            sma_150=sma_vals.get(150, 0) or 0,
            sma_200=sma_vals.get(200, 0) or 0,
            sma_200_slope=sma_slopes.get(200, 0),
            high_52w=high_52w,
            low_52w=low_52w,
            phase=result.phase,
        )
        result.criteria_details = criteria
        result.criteria_passed = sum(criteria.values())

        # ---- VCP 检测 ----
        vcp_detected, vcp_count, vcp_quality = detect_vcp(closes, volumes)
        result.vcp_detected = vcp_detected
        result.vcp_contractions = vcp_count
        result.vcp_quality = vcp_quality

        # ---- 相对强度 ----
        if benchmark_closes is not None and len(benchmark_closes) > 10:
            result.relative_strength = compute_relative_strength(closes, benchmark_closes)
        else:
            result.relative_strength = 5.0

        # ---- 成交量突破 ----
        breakout, vol_ratio = detect_volume_breakout(volumes)
        result.volume_breakout = breakout
        result.volume_ratio = vol_ratio

        # ---- ATR 止损 ----
        stop_loss, pos_size = compute_atr_stop(
            highs, lows, closes,
            period=self.atr_period,
            multiplier=self.atr_multiplier,
            risk_per_trade=self.risk_per_trade,
        )
        result.atr_stop_loss = stop_loss
        result.position_size_pct = pos_size

        # ---- 综合评分 (0-100) ----
        score = 0

        # Phase 得分 (0-25)
        phase_scores = {"Uptrend": 25, "Base": 10, "Distribution": 5, "Downtrend": 0}
        score += phase_scores.get(result.phase, 0)

        # 趋势模板通过度 (0-25)
        template_pct = result.criteria_passed / result.criteria_total
        score += int(template_pct * 25)

        # 相对强度 (0-15)
        score += int(result.relative_strength / 10 * 15)

        # VCP (0-15)
        if result.vcp_detected:
            score += min(15, 10 + result.vcp_contractions * 2)
        elif result.vcp_contractions >= 2:
            score += 5

        # 成交量突破 (0-10)
        if result.volume_breakout:
            score += 10
        elif result.volume_ratio > 1.2:
            score += 5

        # 52周位置 (0-10): 价格接近52周高分越多得分越高
        if high_52w > 0:
            pct_from_high = (high_52w - price) / high_52w * 100
            if pct_from_high < 5:
                score += 10
            elif pct_from_high < 10:
                score += 7
            elif pct_from_high < 20:
                score += 4

        result.total_score = min(100, score)

        # ---- 买卖信号 ----
        # 买入信号: Phase 2 + 趋势模板通过 + 综合分 >= 50
        result.buy_signal = (
            result.phase == "Uptrend"
            and result.criteria_passed >= self.min_criteria_pass
            and result.total_score >= 50
        )

        # 卖出信号: Phase 4 或 Phase 3 + 趋势模板不通过
        result.sell_signal = (
            result.phase == "Downtrend"
            or (result.phase == "Distribution" and result.criteria_passed < 4)
        )

        return result
