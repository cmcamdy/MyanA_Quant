"""因子面板构建器

使用 aurumq-rl 的 296 因子库 (Alpha101 + GTJA191) 对面板数据计算因子,
与 myana-quant 的 IndicatorSet 完全解耦。

依赖: aurumq-rl 需已安装或位于 vendor/ 目录
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import RLConfig

logger = logging.getLogger(__name__)

# 确保 aurumq-rl 可导入
_VENDOR_PATH = Path(__file__).resolve().parent.parent.parent / "vendor" / "aurumq-rl" / "src"
if _VENDOR_PATH.exists() and str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))


class FactorPanelBuilder:
    """使用 aurumq-rl 因子库计算面板因子"""

    def __init__(self, config: RLConfig):
        self.config = config
        self._registries_loaded = False

    def _ensure_registries(self):
        """懒加载因子注册表"""
        if self._registries_loaded:
            return

        try:
            from aurumq_rl.factors import ALPHA101_REGISTRY, GTJA191_REGISTRY
            self._alpha101_registry = ALPHA101_REGISTRY
            self._gtja191_registry = GTJA191_REGISTRY
            self._registries_loaded = True
            logger.info(
                "Factor registries loaded: %d Alpha101, %d GTJA191",
                len(ALPHA101_REGISTRY), len(GTJA191_REGISTRY),
            )
        except ImportError as e:
            raise ImportError(
                "aurumq-rl not found. Install with: "
                "pip install git+https://github.com/yupoet/aurumq-rl.git "
                f"or clone to vendor/aurumq-rl/. Error: {e}"
            ) from e

    def compute(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """对面板数据计算因子

        Args:
            panel_df: DataAdapter 产出的面板 DataFrame
                列含 ts_code, trade_date, close, pct_chg, vol 等

        Returns:
            添加了因子列的面板 DataFrame
        """
        self._ensure_registries()

        factors_path = Path(self.config.panel_dir) / "panel_with_factors.parquet"
        if factors_path.exists():
            logger.info("Factor panel already exists: %s", factors_path)
            return pd.read_parquet(factors_path)

        # 转换为 polars (aurumq-rl 因子库是 polars-native)
        import polars as pl
        pl_df = pl.from_pandas(panel_df)

        # 按配置的 prefix 筛选因子
        entries = self._get_factor_entries()
        logger.info("Computing %d factors...", len(entries))

        # 逐因子计算
        new_cols = {}
        failed = 0
        for entry in entries:
            try:
                result = entry.impl(pl_df)
                if isinstance(result, pl.Series):
                    new_cols[entry.id] = result.to_pandas()
                elif isinstance(result, pl.Expr):
                    # 延迟表达式: 需要在 select 上下文中执行
                    computed = pl_df.select(result)
                    if computed.width == 1:
                        new_cols[entry.id] = computed.to_pandas().iloc[:, 0]
                    else:
                        for c in computed.columns:
                            new_cols[c] = computed[c].to_pandas()
            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.warning("Factor %s failed: %s", entry.id, e)

        if failed > 5:
            logger.warning("... %d more factors failed (total %d)", failed - 5, failed)

        # 添加因子列到原 DataFrame
        result = panel_df.copy()
        for col_name, col_data in new_cols.items():
            if col_name not in result.columns:
                result[col_name] = col_data.values if hasattr(col_data, 'values') else col_data

        # 处理 NaN/Inf
        factor_cols = [c for c in result.columns if c not in
                       ["ts_code", "trade_date", "open", "high", "low",
                        "close", "pct_chg", "vol", "amount"]]
        for col in factor_cols:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            result[col] = result[col].fillna(0.0)

        # 保存
        factors_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(factors_path, index=False)
        logger.info(
            "Factor panel saved: %s (%d factors computed, %d failed)",
            factors_path, len(new_cols), failed,
        )

        return result

    def _get_factor_entries(self) -> list:
        """按 factor_prefixes 配置筛选因子条目"""
        import fnmatch

        entries = []
        all_ids = set()

        for prefix in self.config.factor_prefixes:
            if prefix in ("alpha_*", "alpha101"):
                for fid, entry in self._alpha101_registry.items():
                    if fid not in all_ids:
                        entries.append(entry)
                        all_ids.add(fid)
            elif prefix in ("gtja_*", "gtja191"):
                for fid, entry in self._gtja191_registry.items():
                    if fid not in all_ids:
                        entries.append(entry)
                        all_ids.add(fid)
            elif "*" in prefix:
                # 通配符匹配
                for fid, entry in {**self._alpha101_registry, **self._gtja191_registry}.items():
                    if fnmatch.fnmatch(fid, prefix) and fid not in all_ids:
                        entries.append(entry)
                        all_ids.add(fid)
            else:
                # 精确匹配
                for registry in [self._alpha101_registry, self._gtja191_registry]:
                    if prefix in registry and prefix not in all_ids:
                        entries.append(registry[prefix])
                        all_ids.add(prefix)

        return entries
