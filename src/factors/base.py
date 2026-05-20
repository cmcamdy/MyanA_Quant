"""因子基类与结果数据类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class FactorResult:
    """因子分析结果

    Attributes:
        factor_name: 因子名称
        values: 因子值序列，与输入DataFrame索引一致
        ic: 信息系数 (Information Coefficient)，Spearman秩相关
        ir: 信息比率 (Information Ratio)，IC均值/IC标准差
        quantile_returns: 各分位的平均远期收益
    """
    factor_name: str
    values: pd.Series
    ic: float
    ir: float
    quantile_returns: Dict[int, float]


class Factor(ABC):
    """因子抽象基类

    所有具体因子必须实现 compute 方法，返回与输入 DataFrame
    具有相同索引的 pd.Series。
    """

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值

        Args:
            df: 包含至少 'close' 列的行情数据，可选 'volume' 列

        Returns:
            因子值序列，索引与 df 一致
        """
        ...
