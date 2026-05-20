"""行业分类查询模块"""

import pandas as pd
from pathlib import Path
from typing import List, Optional


class IndustryLookup:
    """行业-股票映射查询"""

    def __init__(self, data_dir: str = "./data"):
        self._csv_path = Path(data_dir) / "industry-stock" / "industry_stock.csv"
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self._csv_path, dtype=str)
        return self._df

    def get_stocks(self, industry, *, per_industry: int = 0) -> List[str]:
        """根据行业名称获取所属股票代码列表

        Args:
            industry: 行业名称或列表，如 "C32有色金属冶炼和压延加工业"
                      或 ["C32有色金属冶炼和压延加工业", "N78公共设施管理业"]
            per_industry: 每个行业最多取多少只，0=不限

        Returns:
            股票代码列表（已去重保序）
        """
        df = self._load()
        if isinstance(industry, str):
            industry = [industry]
        matched = df[df['industry'].isin(industry)]
        codes = matched['code'].tolist()
        if per_industry > 0:
            grouped = matched.groupby('industry')
            selected = []
            for _, grp in grouped:
                selected.extend(grp['code'].tolist()[:per_industry])
            return list(dict.fromkeys(selected))
        return list(dict.fromkeys(codes))

    def get_stock_details(self, industry: str) -> pd.DataFrame:
        """根据行业名称获取所属股票详情（代码+名称）

        Args:
            industry: 行业名称，如 "N78公共设施管理业"

        Returns:
            DataFrame with columns: [code, name]
        """
        df = self._load()
        return df[df['industry'] == industry][['code', 'name']].reset_index(drop=True)

    def list_industries(self) -> List[str]:
        """列出所有行业名称"""
        df = self._load()
        return sorted(df['industry'].unique().tolist())

    def get_industry(self, code: str) -> Optional[str]:
        """根据股票代码获取所属行业

        Args:
            code: 股票代码，如 "600874.SH"

        Returns:
            行业名称，未找到返回 None
        """
        df = self._load()
        matched = df[df['code'] == code]
        if matched.empty:
            return None
        return matched.iloc[0]['industry']
