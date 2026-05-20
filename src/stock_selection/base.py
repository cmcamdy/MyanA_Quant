"""选股筛选器抽象基类

所有选股方法统一接口: screen_all() 返回 (top_df, all_df)，
每个 DataFrame 至少包含 symbol, composite_score 列。

新增选股方法只需继承 ScreenerBase 并实现 screen_all()，
然后在 registry.py 中注册即可。
"""

from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd


class ScreenerBase(ABC):
    """选股筛选器抽象基类

    子类需实现:
        screen_all(**kwargs) -> (top_df, all_df)

    返回约定:
        - top_df: 入选标的 DataFrame，至少含 symbol, composite_score
        - all_df: 全部候选评分 DataFrame，至少含 symbol, composite_score
        - composite_score: 综合得分，越高越好
    """

    @abstractmethod
    def screen_all(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行选股筛选

        Returns:
            (top_df, all_df): 入选标的 + 全部候选评分
        """
        ...

    def screen(self, **kwargs) -> pd.DataFrame:
        """便捷方法: 仅返回入选标的"""
        top, _ = self.screen_all(**kwargs)
        return top
