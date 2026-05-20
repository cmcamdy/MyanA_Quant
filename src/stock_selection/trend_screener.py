"""Minervini Stage 2 趋势选股筛选器

基于 Minervini 趋势模板 + VCP形态 + 相对强度 的综合选股方法。
使用 TencentFetcher 获取实时行情 + 历史K线 (SMA/volume 计算)。

参考: https://github.com/RyanJHamby/stock-screener
      Mark Minervini 《Trade Like a Stock Market Wizard》
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import ScreenerBase
from .tencent_fetcher import TencentFetcher, load_codes_by_market, _format_code
from .trend_scorer import TrendScorer, TrendScoreResult

logger = logging.getLogger(__name__)


class TrendScreener(ScreenerBase):
    """Minervini Stage 2 趋势选股筛选器

    Args:
        top_n: 返回前N只 (默认10)
        min_score: 最低综合分 (默认50)
        market: 股票池 "all"/"a"/"hk"/"csi300" (推荐 csi300，SMA200数据量大)
        benchmark: 基准指数代码 (默认 sh000001)
        max_concurrent: 异步并发数 (默认20)
        sma_periods: SMA周期列表 (默认 [50, 150, 200])
        min_criteria_pass: Minervini模板最少通过条件数 (默认7)
    """

    def __init__(
        self,
        top_n: int = 10,
        min_score: int = 50,
        market: str = "csi300",
        benchmark: str = "sh000001",
        max_concurrent: int = 20,
        sma_periods: Optional[List[int]] = None,
        min_criteria_pass: int = 7,
    ):
        self.top_n = top_n
        self.min_score = min_score
        self.market = market
        self.benchmark = benchmark
        self.max_concurrent = max_concurrent
        self.sma_periods = sma_periods or [50, 150, 200]
        self.min_criteria_pass = min_criteria_pass
        self._fetcher = TencentFetcher(max_concurrent=max_concurrent)
        self._scorer = TrendScorer(
            sma_periods=self.sma_periods,
            min_criteria_pass=self.min_criteria_pass,
        )

    def screen_all(self, codes: Optional[List[str]] = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行趋势选股

        Args:
            codes: 股票代码列表，默认按 market 配置加载

        Returns:
            (top_df, all_df): 入选 + 全部候选评分 DataFrame
        """
        if codes is None:
            codes = load_codes_by_market(self.market)
        if not codes:
            logger.error("无可用股票代码列表")
            return pd.DataFrame(), pd.DataFrame()

        logger.info(f"趋势选股: 开始分析 {len(codes)} 只股票 (市场={self.market})")

        # 1. 获取实时行情
        stocks = self._fetcher.fetch_all_sync(codes)
        if not stocks:
            logger.error("数据获取失败")
            return pd.DataFrame(), pd.DataFrame()

        # 2. 获取历史 OHLCV (需要足够数据计算 SMA200)
        valid_codes = [s["code"] for s in stocks if s.get("code")]
        days_needed = max(self.sma_periods) + 30
        logger.info(f"获取 {len(valid_codes)} 只股票的 {days_needed} 日OHLCV数据...")
        history = self._fetcher.fetch_history_ohlcv_sync(valid_codes, days=days_needed)
        logger.info(f"OHLCV数据: {len(history)}/{len(valid_codes)} 只成功")

        # 3. 获取基准指数数据
        benchmark_closes = None
        logger.info(f"获取基准指数 {self.benchmark} 历史...")
        benchmark_df = self._fetcher.fetch_index_history_sync(self.benchmark, days=days_needed)
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_closes = benchmark_df["close"].values.astype(float)
        else:
            logger.warning("基准指数数据获取失败，相对强度将使用默认值")

        # 4. 评分
        all_results = self._scorer.score_batch(stocks, history, benchmark_closes)

        # 5. 过滤 + 排序
        filtered = [r for r in all_results if r.total_score >= self.min_score]
        filtered.sort(key=lambda r: r.total_score, reverse=True)

        # 6. Top N
        top_results = filtered[:self.top_n]

        # 7. 转 DataFrame
        top_df = self._results_to_df(top_results)
        all_df = self._results_to_df(filtered)

        # 统计
        phase_counts = {}
        for r in filtered:
            phase_counts[r.phase] = phase_counts.get(r.phase, 0) + 1
        buy_count = sum(1 for r in filtered if r.buy_signal)

        logger.info(
            f"趋势选股: 分析 {len(stocks)} 只 → 过滤后 {len(filtered)} 只 "
            f"(Phase: {phase_counts}) → 入选 {len(top_results)} 只, 买入信号: {buy_count}"
        )

        return top_df, all_df

    def _results_to_df(self, results: List[TrendScoreResult]) -> pd.DataFrame:
        """评分结果转 DataFrame"""
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            rows.append({
                "symbol": _format_code(r.symbol) if r.symbol else "",
                "name": r.name,
                "market": r.market,
                "composite_score": r.total_score,
                "phase": r.phase,
                "criteria_passed": r.criteria_passed,
                "criteria_total": r.criteria_total,
                "vcp_detected": r.vcp_detected,
                "vcp_contractions": r.vcp_contractions,
                "vcp_quality": r.vcp_quality,
                "relative_strength": r.relative_strength,
                "volume_breakout": r.volume_breakout,
                "volume_ratio": r.volume_ratio,
                "sma_50": r.sma_50,
                "sma_150": r.sma_150,
                "sma_200": r.sma_200,
                "price": r.price,
                "atr_stop_loss": r.atr_stop_loss,
                "position_size_pct": r.position_size_pct,
                "buy_signal": r.buy_signal,
                "sell_signal": r.sell_signal,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            df.index.name = "rank"
        return df
