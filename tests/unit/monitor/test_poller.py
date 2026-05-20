"""DataPoller 单元测试"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from monitor.poller import DataPoller
from data.base import KlineData


def _make_df(n=10, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq='D')
    close = [100 + i for i in range(n)]
    return pd.DataFrame({
        'open': close,
        'high': [c + 1 for c in close],
        'low': [c - 1 for c in close],
        'close': close,
        'volume': [10000] * n,
    }, index=dates)


class TestDataPollerWarmup:

    def test_warmup_loads_data(self):
        dm = MagicMock()
        df = _make_df(10)
        dm.get_kline.return_value = KlineData(
            symbol="600036.SH", freq="1d",
            start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-10"),
            df=df,
        )
        poller = DataPoller(dm, ["600036.SH"], freq="1d")
        result = poller.warmup("600036.SH", "2024-01-01", "2024-01-10")
        assert result is not None
        assert len(result) == 10
        assert poller._last_processed.get("600036.SH") == df.index[-1]

    def test_warmup_error_returns_none(self):
        dm = MagicMock()
        dm.get_kline.side_effect = RuntimeError("network error")
        poller = DataPoller(dm, ["600036.SH"], freq="1d")
        result = poller.warmup("600036.SH", "2024-01-01", "2024-01-10")
        assert result is None


class TestDataPollerPoll:

    def test_poll_returns_new_bars(self):
        dm = MagicMock()
        poller = DataPoller(dm, ["600036.SH"], freq="1d")

        # Set warmup done
        poller._warmup_done["600036.SH"] = True
        poller._last_processed["600036.SH"] = pd.Timestamp("2024-01-10")

        # update_kline returns data including new bars
        new_df = _make_df(5, start="2024-01-08")  # overlap + new
        dm.update_kline.return_value = KlineData(
            symbol="600036.SH", freq="1d",
            start=pd.Timestamp("2024-01-08"), end=pd.Timestamp("2024-01-12"),
            df=new_df,
        )

        result = poller.poll()
        assert "600036.SH" in result
        # Only bars after 2024-01-10 should be returned
        assert len(result["600036.SH"]) == 2  # 2024-01-09 and 2024-01-10 are <= last_processed
        # Actually, the filter is > last_processed, so only 2024-01-11 and 2024-01-12
        # Wait: _make_df(5, start="2024-01-08") gives 01-08 to 01-12
        # last_processed = 2024-01-10, so new bars = 01-11, 01-12 = 2 bars

    def test_poll_no_new_data(self):
        dm = MagicMock()
        dm.update_kline.return_value = None
        poller = DataPoller(dm, ["600036.SH"], freq="1d")
        poller._warmup_done["600036.SH"] = True

        result = poller.poll()
        assert len(result) == 0

    def test_poll_not_warmup_returns_empty(self):
        dm = MagicMock()
        poller = DataPoller(dm, ["600036.SH"], freq="1d")
        # warmup not done
        result = poller.poll()
        assert len(result) == 0

    def test_poll_error_skips_symbol(self):
        dm = MagicMock()
        dm.update_kline.side_effect = RuntimeError("network error")
        poller = DataPoller(dm, ["600036.SH"], freq="1d")
        poller._warmup_done["600036.SH"] = True

        result = poller.poll()
        assert len(result) == 0

    def test_set_last_processed(self):
        dm = MagicMock()
        poller = DataPoller(dm, ["600036.SH"], freq="1d")
        ts = pd.Timestamp("2024-01-10")
        poller.set_last_processed("600036.SH", ts)
        assert poller._last_processed["600036.SH"] == ts
