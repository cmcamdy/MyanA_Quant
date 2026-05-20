"""因子筛选器"""

from typing import Callable, Dict, List, Optional

import pandas as pd

from .base import Factor


class FactorScreener:
    """因子筛选器：复合因子评分与排名

    Args:
        factors: 因子实例列表
        weights: 各因子的权重字典，key 为因子 compute 后的列名。
                 若为 None 则等权。
    """

    def __init__(
        self,
        factors: List[Factor],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.factors = factors
        self.weights = weights

    def score(self, df: pd.DataFrame) -> pd.Series:
        """计算复合因子得分

        对每个因子先做 z-score 标准化，再按权重加权求和。

        Args:
            df: 行情数据，可含 'symbol' 列用于多股票场景

        Returns:
            复合得分序列，索引与 df 一致
        """
        scores = pd.Series(0.0, index=df.index)
        weight_sum = 0.0

        for factor in self.factors:
            factor_values = factor.compute(df)
            col_name = factor_values.name

            # z-score 标准化
            mean = factor_values.mean()
            std = factor_values.std()
            if std == 0 or pd.isna(std):
                normalized = pd.Series(0.0, index=df.index)
            else:
                normalized = (factor_values - mean) / std

            # 确定权重
            if self.weights is not None and col_name in self.weights:
                w = self.weights[col_name]
            else:
                w = 1.0

            scores = scores + normalized * w
            weight_sum += abs(w)

        if weight_sum > 0:
            scores = scores / weight_sum

        scores.name = 'composite_score'
        return scores

    def rank(self, df: pd.DataFrame, top_n: int = 10) -> List[str]:
        """按复合得分排名，返回 top N 股票代码

        Args:
            df: 行情数据，必须包含 'symbol' 列
            top_n: 返回前 N 名

        Returns:
            股票代码列表
        """
        scores = self.score(df)

        if 'symbol' in df.columns:
            # 多股票场景：按 symbol 分组取最后一期得分
            result_df = pd.DataFrame({
                'symbol': df['symbol'],
                'score': scores,
            })
            # 取每个股票最后一行的得分
            last_scores = result_df.groupby('symbol')['score'].last()
            ranked = last_scores.sort_values(ascending=False)
            return ranked.head(top_n).index.tolist()
        else:
            # 单股票场景：返回得分最高的 top_n 行索引
            ranked = scores.sort_values(ascending=False)
            return ranked.head(top_n).index.tolist()

    def filter(
        self,
        df: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
    ) -> pd.DataFrame:
        """根据因子条件筛选股票

        Args:
            df: 行情数据
            condition: 接受因子计算后的 DataFrame（含因子列），
                       返回布尔 Series

        Returns:
            筛选后的 DataFrame
        """
        # 先计算所有因子并附加到 df 上
        enriched = df.copy()
        for factor in self.factors:
            factor_values = factor.compute(df)
            enriched[factor_values.name] = factor_values

        mask = condition(enriched)
        return enriched[mask]
