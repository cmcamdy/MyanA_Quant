"""JYS五维选股筛选器 — 对接 portfolio pipeline

组合 TencentFetcher + JYSScorer 完成"获取实时数据→五维打分→排名筛选"全流程。
输出格式与现有 StockScreener.screen() 兼容（返回含 symbol + composite_score 的 DataFrame）。
支持A股全量 + 港股通。
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .tencent_fetcher import TencentFetcher, load_codes_by_market, _format_code
from .jys_scorer import JYSScorer, JYSScoreResult

logger = logging.getLogger(__name__)


class JYSScreener:
    """JYS五维选股筛选器

    Args:
        top_n: 返回前N只 (默认10)
        min_score: 最低综合分 (默认40)
        max_pe: 最大PE (默认30)
        min_turnover: 最低换手率% (默认0.3)
        min_price: 最低股价 (默认1.0元/港元)
        max_concurrent: 异步并发数 (默认20)
        market: 股票池 "all"(A股+港股通) / "a" / "hk" / "csi300"
    """

    def __init__(
        self,
        top_n: int = 10,
        min_score: int = 40,
        max_pe: float = 30,
        min_turnover: float = 0.3,
        min_price: float = 1.0,
        max_concurrent: int = 20,
        market: str = "all",
    ):
        self.top_n = top_n
        self.min_score = min_score
        self.max_concurrent = max_concurrent
        self.market = market
        self._fetcher = TencentFetcher(max_concurrent=max_concurrent)
        self._scorer = JYSScorer(config={
            "max_pe_ratio": max_pe,
            "min_turnover_rate": min_turnover,
            "min_price": min_price,
            "min_strength_score": min_score,
        })

    def _results_to_df(self, results: List[JYSScoreResult], all_results: List[JYSScoreResult]) -> pd.DataFrame:
        """将评分结果转为 DataFrame"""
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            d = r.stock_data
            row = {
                "symbol": _format_code(d.get("code", "")) if d.get("code") else "",
                "name": r.name,
                "market": r.market,
                "composite_score": r.total_score,
                "grade": r.grade,
                "price": d.get("price", 0),
                "pe_ratio": d.get("pe_ratio"),
                "pb_ratio": d.get("pb_ratio"),
                "roe": d.get("roe"),
                "profit_growth": d.get("profit_growth"),
                "dividend_yield": d.get("dividend_yield", 0),
                "change_pct": d.get("change_pct", 0),
                "momentum_20d": d.get("momentum_20d", 0),
                "turnover_rate": d.get("turnover_rate", 0),
                "tech_score": r.tech_score,
                "valuation_score": r.valuation_score,
                "profit_score": r.profit_score,
                "safety_score": r.safety_score,
                "dividend_score": r.dividend_score,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def screen(self, codes: Optional[List[str]] = None) -> pd.DataFrame:
        """选股筛选，返回 Top N 结果

        Args:
            codes: 股票代码列表，默认按market配置加载

        Returns:
            DataFrame，含 symbol, name, composite_score, grade 等列
        """
        top, _ = self.screen_all(codes)
        return top

    def screen_all(self, codes: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """选股筛选，返回 (Top N, 全部候选)

        Args:
            codes: 股票代码列表，默认按market配置加载

        Returns:
            (top_df, all_df) 两个 DataFrame
        """
        if codes is None:
            codes = load_codes_by_market(self.market)
        if not codes:
            logger.error("无可用股票代码列表")
            return pd.DataFrame(), pd.DataFrame()

        logger.info(f"JYS五维选股: 开始分析 {len(codes)} 只股票 (市场={self.market})")

        # 获取数据
        stocks = self._fetcher.fetch_all_sync(codes)
        if not stocks:
            logger.error("数据获取失败")
            return pd.DataFrame(), pd.DataFrame()

        # 五维评分
        all_results = self._scorer.score_batch(stocks)

        # 过滤+排序
        filtered = self._scorer.select_top(stocks, top_n=len(stocks))

        # Top N
        top_results = filtered[:self.top_n]

        # 转 DataFrame
        top_df = self._results_to_df(top_results, all_results)
        all_df = self._results_to_df(filtered, all_results)

        if not top_df.empty:
            top_df = top_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            top_df.index = top_df.index + 1
            top_df.index.name = "rank"

        if not all_df.empty:
            all_df = all_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            all_df.index = all_df.index + 1
            all_df.index.name = "rank"

        a_count = sum(1 for r in filtered if r.market == "A")
        hk_count = sum(1 for r in filtered if r.market == "HK")
        logger.info(
            f"JYS五维选股: 分析 {len(stocks)} 只 → "
            f"过滤后 {len(filtered)} 只(A:{a_count} HK:{hk_count}) → 入选 {len(top_results)} 只"
        )

        return top_df, all_df
