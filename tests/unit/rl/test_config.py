"""RLConfig 测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from rl.config import RLConfig


def test_default_config():
    config = RLConfig()
    assert config.data_dir == "./data"
    assert config.env_type == "stock_picking"
    assert config.top_k == 10
    assert config.reward_fn == "sharpe"
    assert config.total_timesteps == 1_000_000
    assert config.signal_mode == "top_k"
    assert config.inference_top_k == 10
    assert config.rebalance_freq == "M"
    assert config.gamma == 0.99
    assert config.clip_range == 0.2


def test_custom_config():
    config = RLConfig(
        data_dir="/tmp/test_data",
        universe="csi300",
        max_stocks=50,
        top_k=5,
        total_timesteps=100_000,
        signal_mode="quantile",
        score_quantile_buy=0.9,
        score_quantile_sell=0.1,
    )
    assert config.universe == "csi300"
    assert config.max_stocks == 50
    assert config.top_k == 5
    assert config.total_timesteps == 100_000
    assert config.signal_mode == "quantile"
    assert config.score_quantile_buy == 0.9


def test_factor_prefixes_default():
    config = RLConfig()
    assert "alpha_*" in config.factor_prefixes
    assert "gtja_*" in config.factor_prefixes
