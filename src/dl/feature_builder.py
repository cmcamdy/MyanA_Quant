"""特征工程：从本地 parquet 构建模型输入特征和标签"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CLASS_BOUNDS, DLConfig

logger = logging.getLogger("dl.feature_builder")


class FeatureBuilder:
    """从本地 parquet 构建模型输入特征和标签"""

    def __init__(self, config: DLConfig):
        self.config = config
        self.storage_root = Path(config.data_dir)
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._indicator_columns: Optional[List[str]] = None

    @property
    def num_features(self) -> int:
        """每个时间步的特征数"""
        if self._indicator_columns is not None:
            return 6 + len(self._indicator_columns)
        # 估算: 6 原始列 + 默认指标输出列数
        return 31

    def _load_parquet(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载单只股票日线 parquet"""
        if '.SH' in symbol.upper():
            market, code = 'sh', symbol.split('.')[0]
        elif '.SZ' in symbol.upper():
            market, code = 'sz', symbol.split('.')[0]
        else:
            code = symbol
            market = 'sh' if symbol.startswith('6') else 'sz'

        path = self.storage_root / market / code / '1d.parquet'
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """对原始 OHLCV DataFrame 计算技术指标"""
        from analysis.indicator_set import IndicatorSet

        iset = IndicatorSet()
        for name, params in self.config.get_indicators():
            iset.add(name, **params)

        result = iset.compute(df)
        # 提取新增的指标列名（排除原始 6 列）
        original_cols = {'open', 'high', 'low', 'close', 'volume', 'amount'}
        indicator_cols = [c for c in result.columns if c not in original_cols]
        self._indicator_columns = indicator_cols
        return result

    def _make_labels(self, close: pd.Series) -> np.ndarray:
        """根据涨跌幅生成分类标签"""
        # 未来 horizon 天的收益率
        future_close = close.shift(-self.config.horizon)
        returns = (future_close - close) / close
        returns = returns.values

        labels = np.full(len(returns), -1, dtype=np.int64)
        for i, (lo, hi) in enumerate(zip(CLASS_BOUNDS[:-1], CLASS_BOUNDS[1:])):
            mask = (returns > lo) & (returns <= hi)
            labels[mask] = i
        return labels

    def build_stock(self, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """构建单只股票的全部样本

        Returns:
            (features: [N, num_features, window], labels: [N]) 或 None
        """
        df = self._load_parquet(symbol)
        if df is None:
            logger.warning(f"{symbol}: parquet 不存在，跳过")
            return None

        df = self._compute_indicators(df)
        df = df.fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0)

        close = df['close']
        labels = self._make_labels(close)

        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if self._indicator_columns:
            feature_cols.extend(self._indicator_columns)

        values = df[feature_cols].values.astype(np.float32)

        window = self.config.window
        n_samples = len(values) - window - self.config.horizon + 1
        if n_samples <= 0:
            logger.warning(f"{symbol}: 数据不足 ({len(values)} 行, 需 {window + self.config.horizon})")
            return None

        features = np.zeros((n_samples, len(feature_cols), window), dtype=np.float32)
        valid_labels = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            t = i + window
            features[i] = values[i:t].T
            valid_labels[i] = labels[t - 1]

        # 过滤无效标签
        mask = valid_labels >= 0
        if mask.sum() == 0:
            logger.warning(f"{symbol}: 无有效标签")
            return None

        return features[mask], valid_labels[mask]

    def build_all(
        self, symbols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, int, int]]]:
        """构建多只股票的样本

        Args:
            symbols: 股票列表，None 则遍历 data/ 下所有

        Returns:
            (features, labels, symbol_per_sample, stock_boundaries)
            stock_boundaries: [(symbol, start_idx, end_idx), ...] 每只股票的样本范围
        """
        if symbols is None:
            from data.storage import ParquetStorage
            storage = ParquetStorage(str(self.storage_root))
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

        # 标准化
        features = self.normalize_features(features)

        return features, labels, all_symbols, stock_boundaries

    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """z-score 标准化

        Args:
            features: [N, num_features, window]
            fit: True=计算并保存均值/标准差, False=用已有参数
        """
        if fit or self._scaler_mean is None:
            # 对每个特征在整个数据集上计算均值和标准差
            # features shape: [N, F, T] → 按 F 维度计算
            self._scaler_mean = features.mean(axis=(0, 2), keepdims=True)
            self._scaler_std = features.std(axis=(0, 2), keepdims=True)
            self._scaler_std[self._scaler_std < 1e-8] = 1.0

        return (features - self._scaler_mean) / self._scaler_std

    def save_scaler(self, path: str):
        """保存标准化参数"""
        np.savez(path, mean=self._scaler_mean, std=self._scaler_std)

    def load_scaler(self, path: str):
        """加载标准化参数"""
        data = np.load(path)
        self._scaler_mean = data['mean']
        self._scaler_std = data['std']
