"""数据适配器：将 myana-quant 的 per-stock parquet 转换为 aurumq-rl 的面板格式

myana-quant 存储:  data/{sh|sz}/{code}/1d.parquet  (每股一个文件, DatetimeIndex)
aurumq-rl 期望:   单文件面板  (ts_code, trade_date, close, pct_chg, vol, ...)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.storage import ParquetStorage
from .config import RLConfig

logger = logging.getLogger(__name__)


class DataAdapter:
    """per-stock parquet → panel parquet 转换器"""

    def __init__(self, config: RLConfig):
        self.config = config
        self._storage = ParquetStorage(config.data_dir)

    def build_panel(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ) -> Path:
        """从 per-stock 存储构建面板 parquet

        Args:
            symbols: 股票列表, None 则按 config.universe 解析
            force: 是否强制重建

        Returns:
            面板 parquet 路径
        """
        panel_path = Path(self.config.panel_dir) / "panel.parquet"
        if panel_path.exists() and not force:
            logger.info("Panel already exists: %s (use force=True to rebuild)", panel_path)
            return panel_path

        if symbols is None:
            symbols = self._resolve_symbols()

        logger.info("Building panel from %d symbols...", len(symbols))

        frames: List[pd.DataFrame] = []
        skipped = 0
        for sym in symbols:
            df = self._storage.load(sym, "1d")
            if df is None:
                skipped += 1
                continue

            if self.config.start_date:
                df = df[df.index >= pd.Timestamp(self.config.start_date)]
            if self.config.end_date:
                df = df[df.index <= pd.Timestamp(self.config.end_date)]

            if len(df) < self.config.min_history:
                skipped += 1
                continue

            frame = self._to_panel_row(df, sym)
            frames.append(frame)

        if skipped:
            logger.info("Skipped %d symbols (missing data or insufficient history)", skipped)

        if not frames:
            raise ValueError("No valid stock data found for panel construction")

        panel = pd.concat(frames, ignore_index=True)

        # 限制股票数量
        unique_codes = panel["ts_code"].nunique()
        if unique_codes > self.config.max_stocks:
            # 按交易天数排序, 保留流动性最好的
            code_counts = panel.groupby("ts_code").size().nlargest(self.config.max_stocks)
            panel = panel[panel["ts_code"].isin(code_counts.index)]

        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(panel_path, index=False)

        logger.info(
            "Panel saved: %s (%d rows, %d stocks)",
            panel_path, len(panel), panel["ts_code"].nunique(),
        )
        return panel_path

    def load_panel(self) -> pd.DataFrame:
        """加载已有的面板 parquet"""
        panel_path = Path(self.config.panel_dir) / "panel.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(
                f"Panel not found: {panel_path}. Run build_panel() first."
            )
        return pd.read_parquet(panel_path)

    def _to_panel_row(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """将单股 DataFrame 转为面板行格式

        myana-quant: DatetimeIndex, columns=[open, high, low, close, volume, amount]
        aurumq-rl:   columns=[ts_code, trade_date, open, high, low, close, pct_chg, vol, amount]
        """
        result = df.copy()
        result["ts_code"] = symbol              # e.g. "600000.SH"
        result["trade_date"] = result.index.strftime("%Y-%m-%d")
        result["pct_chg"] = result["close"].pct_change(fill_method=None).fillna(0.0)
        result = result.rename(columns={"volume": "vol"})

        # 确保列顺序: aurumq-rl 必需列在前, 其余在后
        required = ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol", "amount"]
        extra = [c for c in result.columns if c not in required]
        result = result[required + extra].reset_index(drop=True)

        return result

    def _resolve_symbols(self) -> List[str]:
        """根据 config.universe 解析股票列表"""
        universe = self.config.universe

        if universe == "all":
            return self._storage.list_symbols()

        if universe == "csi300":
            return self._load_index_constituents("csi300")

        # 尝试作为 JSON 文件路径
        json_path = Path(universe)
        if json_path.suffix == ".json" and json_path.exists():
            return self._load_index_constituents_from_file(json_path)

        # 兜底: 列出全部
        logger.warning("Unknown universe '%s', falling back to 'all'", universe)
        return self._storage.list_symbols()

    def _load_index_constituents(self, index_name: str) -> List[str]:
        """从本地 JSON 加载指数成分股"""
        json_path = Path(self.config.data_dir) / f"{index_name}_codes.json"
        if json_path.exists():
            return self._load_index_constituents_from_file(json_path)

        # 无成分股文件则取全部, 限制数量
        logger.warning("Index file not found: %s, using all symbols", json_path)
        all_sym = self._storage.list_symbols()
        return all_sym[: self.config.max_stocks]

    @staticmethod
    def _load_index_constituents_from_file(path: Path) -> List[str]:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "codes" in data:
            return data["codes"]
        raise ValueError(f"Unexpected format in {path}")
