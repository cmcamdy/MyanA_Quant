"""策略信号采集器: 在历史数据上运行子策略，收集连续观点分数序列"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.indicator_set import IndicatorSet
from strategies.base import Strategy, Context, Signal, SignalType, Portfolio

from .meta_config import MetaConfig

logger = logging.getLogger("dl.meta_signal_collector")


class MetaSignalCollector:
    """在历史数据上运行多个策略，采集连续观点分数序列"""

    def __init__(self, config: MetaConfig):
        self.config = config
        self.strategies = config.strategies
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._label_mean: Optional[float] = None
        self._label_std: Optional[float] = None

    def collect_scores(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """在单只股票数据上运行所有策略，返回连续分数 DataFrame

        Args:
            df: OHLCV DataFrame (DatetimeIndex)
            symbol: 标的代码

        Returns:
            DataFrame 包含 strat_0 ~ strat_{K-1} 列 (连续分数) + close 列
        """
        K = len(self.strategies)

        # 1. 初始化策略并收集所有指标依赖
        init_ctx = Context(
            bar=df.iloc[0], bars=df.iloc[:1],
            portfolio=Portfolio(cash=100000, equity=100000),
            current_time=df.index[0], symbol=symbol,
        )
        all_specs = []
        for strat in self.strategies:
            strat.on_init(init_ctx)
            all_specs.extend(strat.indicator_specs)

        # 2. 预计算所有指标
        iset = IndicatorSet()
        seen = set()
        for spec in all_specs:
            key = (spec['name'], tuple(sorted((k, v) for k, v in spec.items() if k != 'name')))
            if key not in seen:
                iset.add(spec['name'], **{k: v for k, v in spec.items() if k != 'name'})
                seen.add(key)

        df_with_ind = iset.compute(df) if iset.specs else df.copy()

        # 3. 逐 bar 采集分数
        score_records = []
        for i in range(len(df_with_ind)):
            row = df_with_ind.iloc[i]
            current_time = pd.Timestamp(df_with_ind.index[i])

            ctx = Context(
                bar=row,
                bars=df_with_ind.iloc[:i + 1],
                portfolio=Portfolio(cash=100000, equity=100000),
                current_time=current_time,
                symbol=symbol,
            )

            record = {}
            for j, strat in enumerate(self.strategies):
                record[f'strat_{j}'] = strat.score(ctx)
            score_records.append(record)

        scores_df = pd.DataFrame(score_records, index=df_with_ind.index)
        scores_df['close'] = df_with_ind['close'].values

        return scores_df

    def _make_labels(self, close: pd.Series) -> np.ndarray:
        """计算未来 horizon 天收益率"""
        future_close = close.shift(-self.config.horizon)
        returns = ((future_close - close) / close).values.astype(np.float32)
        return returns

    def build_stock(self, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """构建单只股票的样本

        Returns:
            (features: [N, K, window], labels: [N]) 或 None
        """
        df = self._load_parquet(symbol)
        if df is None:
            return None

        # 重置策略状态 (每次 build_stock 用全新策略实例)
        strategies = [type(s)(**_get_strategy_args(s)) for s in self.strategies]
        temp_config = MetaConfig(
            strategies=strategies,
            window=self.config.window,
            horizon=self.config.horizon,
        )
        collector = MetaSignalCollector(temp_config)
        scores_df = collector.collect_scores(df, symbol=symbol)

        if len(scores_df) < self.config.window + self.config.horizon:
            logger.warning(f"{symbol}: 数据不足")
            return None

        # 特征列: strat_0 ~ strat_{K-1}
        K = len(self.strategies)
        feature_cols = [f'strat_{j}' for j in range(K)]

        values = scores_df[feature_cols].fillna(0.0).values.astype(np.float32)
        close = scores_df['close']
        labels = self._make_labels(close)

        window = self.config.window
        n_samples = len(values) - window - self.config.horizon + 1
        if n_samples <= 0:
            return None

        features = np.zeros((n_samples, K, window), dtype=np.float32)
        valid_labels = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            t = i + window
            features[i] = values[i:t].T
            valid_labels[i] = labels[t - 1]

        # 过滤 NaN
        mask = ~np.isnan(valid_labels)
        if mask.sum() == 0:
            return None

        return features[mask], valid_labels[mask]

    def build_all(
        self, symbols: Optional[List[str]] = None, normalize_features: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, int, int]]]:
        """构建多只股票的样本"""
        if symbols is None:
            from data.storage import ParquetStorage
            storage = ParquetStorage(str(Path(self.config.data_dir)))
            symbols = storage.list_symbols()

        all_features = []
        all_labels = []
        all_symbols = []
        stock_boundaries = []
        offset = 0

        for i, sym in enumerate(symbols):
            result = self.build_stock(sym)
            if result is None:
                continue
            feat, lab = result
            all_features.append(feat)
            all_labels.append(lab)
            all_symbols.extend([sym] * len(lab))
            stock_boundaries.append((sym, offset, offset + len(lab)))
            offset += len(lab)
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(symbols)} 只股票, 累计样本 {offset}")

        if not all_features:
            raise ValueError("没有可用的训练数据")

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logger.info(f"共 {len(symbols)} 只股票, 有效样本 {len(labels)}")

        if normalize_features:
            features = self.normalize_features(features)

        return features, labels, all_symbols, stock_boundaries

    def _load_parquet(self, symbol: str) -> Optional[pd.DataFrame]:
        storage_root = Path(self.config.data_dir)
        if '.SH' in symbol.upper():
            market, code = 'sh', symbol.split('.')[0]
        elif '.SZ' in symbol.upper():
            market, code = 'sz', symbol.split('.')[0]
        else:
            code = symbol
            market = 'sh' if symbol.startswith('6') else 'sz'
        path = storage_root / market / code / '1d.parquet'
        if not path.exists():
            return None
        df = pd.read_parquet(path)

        # 日期过滤
        if self.config.start_date is not None or self.config.end_date is not None:
            if isinstance(df.index, pd.DatetimeIndex):
                if self.config.start_date:
                    df = df[df.index >= pd.Timestamp(self.config.start_date)]
                if self.config.end_date:
                    df = df[df.index <= pd.Timestamp(self.config.end_date)]

        return df

    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit or self._scaler_mean is None:
            self._scaler_mean = features.mean(axis=(0, 2), keepdims=True)
            self._scaler_std = features.std(axis=(0, 2), keepdims=True)
            self._scaler_std[self._scaler_std < 1e-8] = 1.0
        return (features - self._scaler_mean) / self._scaler_std

    def normalize_labels(self, labels: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit or self._label_mean is None:
            self._label_mean = float(labels.mean())
            self._label_std = float(labels.std())
            if self._label_std < 1e-8:
                self._label_std = 1.0
        return ((labels - self._label_mean) / self._label_std).astype(np.float32)

    def denormalize_labels(self, labels: np.ndarray) -> np.ndarray:
        if self._label_mean is None or self._label_std is None:
            raise ValueError("标签归一化参数未初始化")
        return labels * self._label_std + self._label_mean

    def save_scaler(self, path: str):
        data = {'mean': self._scaler_mean, 'std': self._scaler_std}
        if self._label_mean is not None:
            data['label_mean'] = np.array([self._label_mean])
            data['label_std'] = np.array([self._label_std])
        np.savez(path, **data)

    def load_scaler(self, path: str):
        data = np.load(path)
        self._scaler_mean = data['mean']
        self._scaler_std = data['std']
        if 'label_mean' in data:
            self._label_mean = float(data['label_mean'])
            self._label_std = float(data['label_std'])


def _get_strategy_args(strategy: Strategy) -> dict:
    """从策略实例提取构造参数 (用于克隆)"""
    import dataclasses
    if dataclasses.is_dataclass(strategy):
        return {f.name: getattr(strategy, f.name) for f in dataclasses.fields(strategy)}
    import inspect
    sig = inspect.signature(strategy.__init__)
    args = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if hasattr(strategy, name):
            args[name] = getattr(strategy, name)
        elif param.default is not inspect.Parameter.empty:
            args[name] = param.default
    return args
