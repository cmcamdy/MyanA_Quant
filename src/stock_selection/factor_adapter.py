"""因子筛选器适配器

将 strategies.screener.StockScreener 适配到 ScreenerBase 接口。
StockScreener 依赖本地 parquet 数据 + baostock，适合历史回测场景。
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base import ScreenerBase


class FactorScreenerAdapter(ScreenerBase):
    """因子筛选器适配器

    包装 strategies.screener.StockScreener，使其符合 ScreenerBase 接口。

    Args:
        top_n: 返回前N只
        min_bars: 最低K线数量
        factors: 因子配置 {name: {params}}
        indicator_factors: 指标因子配置 {name: bool}
        weights: 复合评分权重
        filters: 规则过滤 {column: {min/max}}
        data_dir: 数据目录
    """

    def __init__(
        self,
        top_n: int = 10,
        min_bars: int = 60,
        factors: Optional[Dict] = None,
        indicator_factors: Optional[Dict] = None,
        weights: Optional[Dict] = None,
        filters: Optional[Dict] = None,
        data_dir: str = "./data",
    ):
        from strategies.screener import StockScreener, ScreeningConfig
        from data.storage import ParquetStorage

        storage = ParquetStorage(data_dir)
        self._screener = StockScreener(
            config=ScreeningConfig(
                factors=factors or {},
                indicator_factors=indicator_factors,
                weights=weights,
                filters=filters,
                top_n=top_n,
                min_bars=min_bars,
            ),
            storage=storage,
            data_dir=data_dir,
        )

    def screen_all(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行因子筛选

        Args:
            symbols: 标的列表
            start_date: 数据起始日
            end_date: 数据结束日
        """
        top, all_df = self._screener.screen_all(
            symbols=symbols or [],
            start_date=start_date,
            end_date=end_date,
        )
        if "name" not in top.columns:
            top["name"] = ""
        if "name" not in all_df.columns:
            all_df["name"] = ""
        return top, all_df
