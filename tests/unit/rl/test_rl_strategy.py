"""RLStrategy 测试 (使用 mock 推理)"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from rl.config import RLConfig
from rl.rl_strategy import RLStrategy
from strategies.base import SignalType
from strategies.portfolio_engine import PortfolioContext, Portfolio


@pytest.fixture
def config():
    return RLConfig(
        onnx_model_path="/tmp/mock_model.onnx",
        inference_top_k=2,
        rebalance_freq="M",
        signal_mode="top_k",
    )


@pytest.fixture
def mock_context():
    """构建 mock PortfolioContext"""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    symbols = ["600000.SH", "000001.SZ", "600036.SH"]
    historical = {}
    for sym in symbols:
        df = pd.DataFrame({
            "open": 10.0 + np.arange(100) * 0.1,
            "high": 10.5 + np.arange(100) * 0.1,
            "low": 9.5 + np.arange(100) * 0.1,
            "close": 10.0 + np.arange(100) * 0.1,
            "volume": (1000000 + np.arange(100) * 1000).astype(int),
        }, index=dates)
        historical[sym] = df

    portfolio = Portfolio(cash=100000.0, equity=100000.0)
    bars = {sym: historical[sym].iloc[-1] for sym in symbols}
    current_time = dates[-1]

    return PortfolioContext(
        bars=bars,
        historical=historical,
        portfolio=portfolio,
        current_time=current_time,
        bar=bars[symbols[0]],
        symbol=symbols[0],
    )


class TestRLStrategy:

    def test_should_rebalance_first_time(self, config, mock_context):
        strategy = RLStrategy(config)
        # 首次应触发再平衡
        assert strategy._should_rebalance(mock_context.current_time) is True

    def test_should_not_rebalance_same_period(self, config, mock_context):
        strategy = RLStrategy(config)
        strategy._last_rebalance = mock_context.current_time
        # 同一月内不应再平衡
        next_day = mock_context.current_time + pd.Timedelta(days=1)
        assert strategy._should_rebalance(next_day) is False

    def test_signals_top_k(self, config, mock_context):
        strategy = RLStrategy(config)

        # Mock inference
        mock_inference = MagicMock()
        # 3 只股票, 600036.SH 和 000001.SZ 分数最高
        mock_inference.predict.return_value = np.array([[0.3, 0.8, 0.6]])
        strategy._inference = mock_inference
        strategy._symbol_index = {
            "600000.SH": 0, "000001.SZ": 1, "600036.SH": 2,
        }

        scores = {"600000.SH": 0.3, "000001.SZ": 0.8, "600036.SH": 0.6}
        signals = strategy._signals_top_k(scores, mock_context)

        # top-2 = 000001.SZ (0.8), 600036.SH (0.6)
        buy_symbols = {s.symbol for s in signals if s.type == SignalType.BUY}
        assert "000001.SZ" in buy_symbols
        assert "600036.SH" in buy_symbols
        assert "600000.SH" not in buy_symbols

    def test_signals_top_k_with_holdings(self, config, mock_context):
        strategy = RLStrategy(config)

        # 持有 600000.SH, 但它不在 top-2
        mock_context.portfolio.get_position("600000.SH").quantity = 100
        mock_context.portfolio.get_position("600000.SH").avg_cost = 10.0
        mock_context.portfolio.get_position("600000.SH").market_value = 1000.0

        scores = {"600000.SH": 0.3, "000001.SZ": 0.8, "600036.SH": 0.6}
        signals = strategy._signals_top_k(scores, mock_context)

        # 应卖出 600000.SH
        sell_signals = [s for s in signals if s.type == SignalType.SELL]
        assert any(s.symbol == "600000.SH" for s in sell_signals)

    def test_signals_quantile(self, config, mock_context):
        config.signal_mode = "quantile"
        config.score_quantile_buy = 0.8
        config.score_quantile_sell = 0.2
        strategy = RLStrategy(config)

        # 持有低分股票
        mock_context.portfolio.get_position("600000.SH").quantity = 100
        mock_context.portfolio.get_position("600000.SH").avg_cost = 10.0
        mock_context.portfolio.get_position("600000.SH").market_value = 1000.0

        scores = {"600000.SH": 0.1, "000001.SZ": 0.9, "600036.SH": 0.5}
        signals = strategy._signals_quantile(scores, mock_context)

        buy_symbols = {s.symbol for s in signals if s.type == SignalType.BUY}
        sell_symbols = {s.symbol for s in signals if s.type == SignalType.SELL}

        # 高分位买入
        assert "000001.SZ" in buy_symbols
        # 低分位卖出
        assert "600000.SH" in sell_symbols

    def test_build_observation(self, config, mock_context):
        strategy = RLStrategy(config)
        obs = strategy._build_observation(mock_context)

        assert obs is not None
        assert obs.dtype == np.float32
        # 3 stocks * 2 features
        assert obs.shape == (1, 6)
