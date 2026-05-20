"""选股模块：因子筛选 + 指标因子 + 规则过滤"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.indicator_set import IndicatorSet
from data.storage import ParquetStorage
from data.manager import DataManager
from factors.factors import (
    MomentumFactor, VolatilityFactor, TurnoverFactor,
    ReversalFactor, PriceVolumeFactor, BiasFactor,
)

FACTOR_CLASSES = {
    'momentum': MomentumFactor,
    'volatility': VolatilityFactor,
    'turnover': TurnoverFactor,
    'reversal': ReversalFactor,
    'price_volume': PriceVolumeFactor,
    'bias': BiasFactor,
}


@dataclass
class ScreeningConfig:
    factors: Dict[str, Dict] = field(default_factory=dict)
    indicator_factors: Optional[Dict[str, bool]] = None
    weights: Optional[Dict[str, float]] = None
    filters: Optional[Dict[str, Dict]] = None
    top_n: int = 10
    min_bars: int = 60


class StockScreener:
    """多因子选股筛选器"""

    def __init__(self, config: ScreeningConfig, storage: ParquetStorage,
                 data_dir: str = "./data"):
        self._config = config
        self._storage = storage
        self._mgr = DataManager(data_dir)

    def screen(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = []
        for sym in symbols:
            factors = self._load_and_compute(sym, start_date, end_date)
            if factors is not None:
                factors['symbol'] = sym
                rows.append(factors)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # 规则过滤
        df_filtered = self._apply_filters(df)
        if df_filtered.empty:
            return df_filtered

        # z-score 归一化 + 加权打分
        df_scored = self._compute_composite_score(df_filtered)

        # 排序
        df_scored = df_scored.sort_values('composite_score', ascending=False).reset_index(drop=True)
        df_scored.index = df_scored.index + 1
        df_scored.index.name = 'rank'

        return df_scored.head(self._config.top_n)

    def screen_all(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple:
        """返回 (top_n结果, 全部候选评分)"""
        rows = []
        for sym in symbols:
            factors = self._load_and_compute(sym, start_date, end_date)
            if factors is not None:
                factors['symbol'] = sym
                rows.append(factors)

        if not rows:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(rows)

        # 记录被规则过滤前的数量
        no_data_count = len(symbols) - len(rows)

        # 规则过滤
        df_filtered = self._apply_filters(df)
        if df_filtered.empty:
            return pd.DataFrame(), df

        # z-score 归一化 + 加权打分
        df_scored = self._compute_composite_score(df_filtered)

        # 排序
        df_scored = df_scored.sort_values('composite_score', ascending=False).reset_index(drop=True)
        df_scored.index = df_scored.index + 1
        df_scored.index.name = 'rank'

        top = df_scored.head(self._config.top_n)
        return top, df_scored

    def _load_and_compute(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[Dict[str, float]]:
        df = self._storage.load(symbol, '1d')
        # 本地无数据时尝试下载
        if df is None or len(df) < self._config.min_bars:
            try:
                start = pd.Timestamp(start_date) if start_date else pd.Timestamp('2020-01-01')
                end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
                kline = self._mgr.get_kline(symbol, start=start, end=end, freq='1d', source='baostock')
                df = kline.df if kline and kline.df is not None and len(kline.df) > 0 else None
            except Exception:
                df = None
        if df is None or len(df) < self._config.min_bars:
            return None
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        if len(df) < self._config.min_bars:
            return None

        result = {}

        # 1. Factor 类因子
        for name, params in self._config.factors.items():
            cls = FACTOR_CLASSES.get(name)
            if cls is None:
                continue
            factor = cls(**params)
            series = factor.compute(df)
            val = series.iloc[-1]
            if pd.notna(val):
                result[series.name] = val

        # 2. IndicatorSet 指标因子
        ind_factors = self._config.indicator_factors or {}
        if any(ind_factors.values()):
            result.update(self._compute_indicator_factors(df, ind_factors))

        return result

    def _compute_indicator_factors(
        self, df: pd.DataFrame, enabled: Dict[str, bool],
    ) -> Dict[str, float]:
        iset = IndicatorSet()
        need_ma = enabled.get('ma_alignment', False)
        need_atr = enabled.get('atr_price_ratio', False)
        need_obv = enabled.get('obv_direction', False)
        need_vol = enabled.get('avg_volume', False)

        if need_ma:
            for p in [5, 10, 20, 60]:
                iset.add('ma', period=p)
        if need_atr:
            iset.add('atr', period=14)
        if need_obv:
            iset.add('obv')

        enriched = iset.compute(df) if iset.specs else df
        result = {}
        lookback = min(20, len(enriched) - 1)

        if need_ma:
            result['ma_alignment'] = self._ma_alignment(enriched)

        if need_atr and 'atr_14' in enriched.columns:
            close = enriched['close'].iloc[-1]
            atr = enriched['atr_14'].iloc[-1]
            if pd.notna(atr) and close > 0:
                result['atr_price_ratio'] = atr / close

        if need_obv and 'obv' in enriched.columns:
            obv_now = enriched['obv'].iloc[-1]
            obv_prev = enriched['obv'].iloc[-lookback - 1] if len(enriched) > lookback else enriched['obv'].iloc[0]
            if pd.notna(obv_now) and pd.notna(obv_prev):
                result['obv_direction'] = 1.0 if obv_now > obv_prev else (-1.0 if obv_now < obv_prev else 0.0)

        if need_vol and 'volume' in enriched.columns:
            result['avg_volume'] = enriched['volume'].rolling(20).mean().iloc[-1]

        return result

    @staticmethod
    def _ma_alignment(df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        ma5 = last.get('ma_5')
        ma10 = last.get('ma_10')
        ma20 = last.get('ma_20')
        ma60 = last.get('ma_60')
        vals = [ma5, ma10, ma20, ma60]
        if any(v is None or pd.isna(v) for v in vals):
            return 0.0
        pairs = [(ma5, ma10), (ma10, ma20), (ma20, ma60)]
        return sum(1 for a, b in pairs if a > b) / len(pairs)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filters = self._config.filters
        if not filters:
            return df
        mask = pd.Series(True, index=df.index)
        for col, cond in filters.items():
            if col not in df.columns:
                continue
            if 'min' in cond:
                mask &= (df[col] >= cond['min'])
            if 'max' in cond:
                mask &= (df[col] <= cond['max'])
        return df[mask].reset_index(drop=True)

    def _compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        factor_cols = [c for c in df.columns if c != 'symbol']
        if not factor_cols:
            df['composite_score'] = 0.0
            return df

        weights = self._config.weights or {}

        # z-score 归一化
        normalized = pd.DataFrame(index=df.index)
        for col in factor_cols:
            series = df[col]
            std = series.std()
            if std == 0 or pd.isna(std):
                normalized[col] = 0.0
            else:
                normalized[col] = (series - series.mean()) / std

        # 加权求和
        score = pd.Series(0.0, index=df.index)
        weight_sum = 0.0
        for col in factor_cols:
            w = weights.get(col, 1.0)
            score += normalized[col] * w
            weight_sum += abs(w)

        df['composite_score'] = score / weight_sum if weight_sum > 0 else score
        return df
