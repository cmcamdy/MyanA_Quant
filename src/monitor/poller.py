"""数据轮询器"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from data.manager import DataManager

logger = logging.getLogger(__name__)


class DataPoller:
    """定期轮询 DataManager 获取最新数据"""

    def __init__(
        self,
        data_manager: DataManager,
        symbols: List[str],
        freq: str = "1d",
        poll_interval: int = 60,
        source: Optional[str] = None,
        adjust: str = "qfq",
    ):
        self._data_manager = data_manager
        self._symbols = symbols
        self._freq = freq
        self._poll_interval = poll_interval
        self._source = source
        self._adjust = adjust
        self._last_processed: Dict[str, pd.Timestamp] = {}
        self._warmup_done: Dict[str, bool] = {s: False for s in symbols}

    def warmup(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """首次加载历史数据作为策略预热"""
        try:
            kline = self._data_manager.get_kline(
                symbol, start=start, end=end, freq=self._freq,
                source=self._source, adjust=self._adjust,
            )
            if kline is not None and not kline.df.empty:
                self._last_processed[symbol] = kline.df.index[-1]
                self._warmup_done[symbol] = True
                return kline.df
        except Exception as e:
            logger.warning("Warmup error for %s: %s", symbol, e)
        return None

    def poll(self) -> Dict[str, pd.DataFrame]:
        """拉取所有标的新增数据，返回 {symbol: new_bars_df}"""
        new_data: Dict[str, pd.DataFrame] = {}
        for symbol in self._symbols:
            try:
                result = self._poll_symbol(symbol)
                if result is not None and not result.empty:
                    new_data[symbol] = result
            except Exception as e:
                logger.warning("Poll error for %s: %s", symbol, e)
        return new_data

    def _poll_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """拉取单个标的新增数据"""
        if not self._warmup_done.get(symbol, False):
            return None

        kline = self._data_manager.update_kline(
            symbol, freq=self._freq, adjust=self._adjust,
        )
        if kline is None or kline.df.empty:
            return None

        last = self._last_processed.get(symbol)
        if last is not None:
            new_bars = kline.df[kline.df.index > last]
        else:
            new_bars = kline.df

        if not new_bars.empty:
            self._last_processed[symbol] = new_bars.index[-1]
            return new_bars

        return None

    def set_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        """恢复状态时设置已处理时间"""
        self._last_processed[symbol] = timestamp

    @property
    def poll_interval(self) -> int:
        return self._poll_interval

    @property
    def last_processed(self) -> Dict[str, pd.Timestamp]:
        return dict(self._last_processed)
