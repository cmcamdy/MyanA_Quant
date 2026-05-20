"""因子分析器"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .base import Factor, FactorResult


class FactorAnalyzer:
    """因子分析器

    提供因子计算、IC/IR计算、分位分析等功能。

    Args:
        forward_period: 远期收益计算周期，默认1
    """

    def __init__(self, forward_period: int = 1) -> None:
        self.forward_period = forward_period

    def _forward_returns(self, df: pd.DataFrame) -> pd.Series:
        """计算远期收益"""
        close = df['close']
        return close.shift(-self.forward_period) / close - 1

    def compute_factor(self, factor: Factor, df: pd.DataFrame) -> pd.Series:
        """计算单个因子值

        Args:
            factor: 因子实例
            df: 行情数据

        Returns:
            因子值序列
        """
        return factor.compute(df)

    def compute_ic(self, factor_values: pd.Series, df: pd.DataFrame) -> float:
        """计算信息系数 (IC)

        使用 Spearman 秩相关系数衡量因子值与远期收益的单调相关性。

        Args:
            factor_values: 因子值序列
            df: 行情数据（用于计算远期收益）

        Returns:
            Spearman 秩相关系数
        """
        forward_ret = self._forward_returns(df)
        # 对齐索引，剔除 NaN
        valid = factor_values.notna() & forward_ret.notna()
        fv = factor_values[valid]
        fr = forward_ret[valid]
        if len(fv) < 3:
            return np.nan
        corr, _ = spearmanr(fv, fr)
        return float(corr) if not np.isnan(corr) else 0.0

    def compute_ir(self, ic_series: pd.Series) -> float:
        """计算信息比率 (IR)

        IR = IC均值 / IC标准差

        Args:
            ic_series: IC时间序列

        Returns:
            信息比率
        """
        if len(ic_series) < 2 or ic_series.std() == 0:
            return np.nan
        return float(ic_series.mean() / ic_series.std())

    def quantile_analysis(
        self,
        factor_values: pd.Series,
        df: pd.DataFrame,
        n_quantiles: int = 5,
    ) -> Dict[int, float]:
        """分位分析

        将因子值分成 n_quantiles 个分位，计算每个分位的平均远期收益。

        Args:
            factor_values: 因子值序列
            df: 行情数据
            n_quantiles: 分位数量，默认5

        Returns:
            各分位的平均远期收益字典 {分位编号: 平均收益}
        """
        forward_ret = self._forward_returns(df)
        valid = factor_values.notna() & forward_ret.notna()
        fv = factor_values[valid].copy()
        fr = forward_ret[valid].copy()

        if len(fv) < n_quantiles:
            return {i + 1: np.nan for i in range(n_quantiles)}

        try:
            labels = pd.qcut(fv, n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            return {i + 1: np.nan for i in range(n_quantiles)}

        unique_labels = sorted(labels.unique())
        result: Dict[int, float] = {}
        for i in range(n_quantiles):
            if i in unique_labels:
                mask = labels == i
                result[i + 1] = float(fr[mask].mean())
            else:
                result[i + 1] = np.nan
        return result

    def analyze(self, factor: Factor, df: pd.DataFrame) -> FactorResult:
        """完整分析单个因子

        Args:
            factor: 因子实例
            df: 行情数据

        Returns:
            FactorResult 包含因子值、IC、IR 和分位收益
        """
        factor_values = self.compute_factor(factor, df)
        ic = self.compute_ic(factor_values, df)

        # 构建 IC 序列来计算 IR：使用滚动窗口计算逐期 IC
        forward_ret = self._forward_returns(df)
        valid = factor_values.notna() & forward_ret.notna()
        if valid.sum() < 10:
            ir = np.nan
        else:
            # 使用滑动窗口计算 IC 序列
            window = min(60, max(20, valid.sum() // 3))
            ic_list = []
            fv_valid = factor_values[valid]
            fr_valid = forward_ret[valid]
            for i in range(window, len(fv_valid)):
                fv_window = fv_valid.iloc[i - window:i]
                fr_window = fr_valid.iloc[i - window:i]
                if fv_window.std() > 0 and fr_window.std() > 0:
                    corr, _ = spearmanr(fv_window, fr_window)
                    if not np.isnan(corr):
                        ic_list.append(corr)
            if len(ic_list) >= 2:
                ic_series = pd.Series(ic_list)
                ir = self.compute_ir(ic_series)
            else:
                ir = np.nan

        quantile_returns = self.quantile_analysis(factor_values, df)
        factor_name = factor_values.name if factor_values.name else type(factor).__name__

        return FactorResult(
            factor_name=factor_name,
            values=factor_values,
            ic=ic,
            ir=ir,
            quantile_returns=quantile_returns,
        )

    def analyze_batch(
        self,
        factors: List[Factor],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """批量分析多个因子

        Args:
            factors: 因子实例列表
            df: 行情数据

        Returns:
            DataFrame，列为 [factor_name, ic, ir, q1_return, ..., q5_return]
        """
        rows = []
        for factor in factors:
            result = self.analyze(factor, df)
            row = {
                'factor_name': result.factor_name,
                'ic': result.ic,
                'ir': result.ir,
            }
            for q, ret in result.quantile_returns.items():
                row[f'q{q}_return'] = ret
            rows.append(row)
        return pd.DataFrame(rows)


def cross_section_ic(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
) -> pd.Series:
    """计算截面 IC

    对每个时间截面，计算因子值与收益率之间的 Spearman 秩相关系数。

    Args:
        factor_df: 因子值面板，列为股票代码，索引为日期
        return_df: 收益率面板，列为股票代码，索引为日期

    Returns:
        各时间截面的 IC 序列
    """
    common_idx = factor_df.index.intersection(return_df.index)
    common_cols = factor_df.columns.intersection(return_df.columns)

    ic_list = []
    for idx in common_idx:
        fv = factor_df.loc[idx, common_cols]
        rv = return_df.loc[idx, common_cols]
        valid = fv.notna() & rv.notna()
        if valid.sum() < 3:
            ic_list.append(np.nan)
            continue
        corr, _ = spearmanr(fv[valid], rv[valid])
        ic_list.append(corr)

    return pd.Series(ic_list, index=common_idx, name='cross_section_ic')
