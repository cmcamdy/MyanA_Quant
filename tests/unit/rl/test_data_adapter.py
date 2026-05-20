"""DataAdapter 测试"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from rl.config import RLConfig
from rl.data_adapter import DataAdapter


@pytest.fixture
def tmp_data(tmp_path):
    """创建临时 per-stock parquet 数据"""
    from data.storage import ParquetStorage
    from data.base import KlineData

    storage = ParquetStorage(str(tmp_path / "data"))
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    for code in ["600000.SH", "000001.SZ", "600036.SH"]:
        df = pd.DataFrame({
            "open": 10.0 + np.arange(300) * 0.1,
            "high": 10.5 + np.arange(300) * 0.1,
            "low": 9.5 + np.arange(300) * 0.1,
            "close": 10.0 + np.arange(300) * 0.1,
            "volume": (1000000 + np.arange(300) * 1000).astype(int),
            "amount": 10000000.0 + np.arange(300) * 10000,
        }, index=dates)
        storage.save(KlineData(symbol=code, freq="1d", start=None, end=None, df=df))

    return tmp_path


class TestDataAdapter:

    def test_build_panel(self, tmp_data):
        config = RLConfig(
            data_dir=str(tmp_data / "data"),
            panel_dir=str(tmp_data / "panels"),
            universe="all",
            min_history=100,
        )
        adapter = DataAdapter(config)
        panel_path = adapter.build_panel()

        assert panel_path.exists()
        df = adapter.load_panel()
        assert "ts_code" in df.columns
        assert "trade_date" in df.columns
        assert "close" in df.columns
        assert "pct_chg" in df.columns
        assert "vol" in df.columns
        assert df["ts_code"].nunique() == 3

    def test_panel_schema(self, tmp_data):
        config = RLConfig(
            data_dir=str(tmp_data / "data"),
            panel_dir=str(tmp_data / "panels"),
            universe="all",
            min_history=100,
        )
        adapter = DataAdapter(config)
        adapter.build_panel()
        df = adapter.load_panel()

        # 检查 aurumq-rl 必需列
        required = ["ts_code", "trade_date", "close", "pct_chg", "vol"]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

        # 检查 ts_code 格式
        for code in df["ts_code"].unique():
            assert "." in code, f"Invalid ts_code format: {code}"

    def test_build_panel_cached(self, tmp_data):
        config = RLConfig(
            data_dir=str(tmp_data / "data"),
            panel_dir=str(tmp_data / "panels"),
            universe="all",
            min_history=100,
        )
        adapter = DataAdapter(config)
        path1 = adapter.build_panel()
        path2 = adapter.build_panel()  # 第二次应使用缓存
        assert path1 == path2

    def test_build_panel_force(self, tmp_data):
        config = RLConfig(
            data_dir=str(tmp_data / "data"),
            panel_dir=str(tmp_data / "panels"),
            universe="all",
            min_history=100,
        )
        adapter = DataAdapter(config)
        adapter.build_panel()
        # force rebuild
        path = adapter.build_panel(force=True)
        assert path.exists()

    def test_pct_chg_computed(self, tmp_data):
        config = RLConfig(
            data_dir=str(tmp_data / "data"),
            panel_dir=str(tmp_data / "panels"),
            universe="all",
            min_history=100,
        )
        adapter = DataAdapter(config)
        adapter.build_panel()
        df = adapter.load_panel()

        # pct_chg 应该是数值, 不全为 0
        assert df["pct_chg"].dtype in [float, "float32", "float64"]
        # 大部分行的 pct_chg 不为 0 (首行 fillna(0) 除外)
        nonzero = (df["pct_chg"] != 0.0).sum()
        assert nonzero > 0

    def test_date_filter(self, tmp_data):
        config = RLConfig(
            data_dir=str(tmp_data / "data"),
            panel_dir=str(tmp_data / "panels"),
            universe="all",
            min_history=10,
            start_date="2023-06-01",
            end_date="2023-12-31",
        )
        adapter = DataAdapter(config)
        adapter.build_panel(force=True)
        df = adapter.load_panel()

        dates = pd.to_datetime(df["trade_date"])
        assert dates.min() >= pd.Timestamp("2023-06-01")
        assert dates.max() <= pd.Timestamp("2023-12-31")
