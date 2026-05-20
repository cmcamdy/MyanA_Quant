"""Gym 环境工厂

桥接 myana-quant 数据到 aurumq-rl 的 Gymnasium 环境,
支持 StockPickingEnv 和 PortfolioWeightEnv。

依赖: aurumq-rl, gymnasium, stable-baselines3
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import RLConfig

logger = logging.getLogger(__name__)

# 确保 aurumq-rl 可导入
_VENDOR_PATH = Path(__file__).resolve().parent.parent.parent / "vendor" / "aurumq-rl" / "src"
if _VENDOR_PATH.exists() and str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))


class EnvFactory:
    """创建 aurumq-rl gym 环境"""

    def __init__(self, config: RLConfig):
        self.config = config

    def create_train_val(
        self,
        panel_df,
        train_ratio: float = 0.8,
    ) -> Tuple[list, object]:
        """创建训练和验证环境

        Args:
            panel_df: 含因子的面板 DataFrame (或路径)
            train_ratio: 训练集比例

        Returns:
            (train_envs, val_env) 元组
        """
        # 按日期分割
        if isinstance(panel_df, (str, Path)):
            import pandas as pd
            panel_df = pd.read_parquet(panel_df)

        dates = sorted(panel_df["trade_date"].unique())
        split_idx = int(len(dates) * train_ratio)
        train_dates = set(dates[:split_idx])
        val_dates = set(dates[split_idx:])

        train_df = panel_df[panel_df["trade_date"].isin(train_dates)]
        val_df = panel_df[panel_df["trade_date"].isin(val_dates)]

        logger.info(
            "Train/Val split: %d / %d dates, %d / %d rows",
            len(train_dates), len(val_dates), len(train_df), len(val_df),
        )

        # 保存临时面板文件供 FactorPanelLoader 读取
        tmp_dir = Path(self.config.panel_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        train_path = tmp_dir / "train_panel.parquet"
        val_path = tmp_dir / "val_panel.parquet"
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

        # 创建训练环境 (n_envs 个并行)
        train_envs = self._make_vec_env(str(train_path), is_train=True)

        # 创建验证环境 (单个)
        val_env = self._make_env(str(val_path), is_train=False)

        return train_envs, val_env

    def _make_vec_env(self, panel_path: str, is_train: bool = True) -> list:
        """创建多个并行环境"""
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env(rank):
            def _init():
                return self._make_env(panel_path, is_train=is_train)
            return _init

        n = self.config.n_envs
        envs = SubprocVecEnv([make_env(i) for i in range(n)])
        return envs

    def _make_env(self, panel_path: str, is_train: bool = True):
        """创建单个 aurumq-rl 环境"""
        try:
            from aurumq_rl.data_loader import FactorPanelLoader
            from aurumq_rl.env import StockPickingEnv, StockPickingConfig
            from aurumq_rl.portfolio_weight_env import PortfolioWeightEnv, PortfolioWeightConfig
            from aurumq_rl.reward_functions import (
                simple_return_reward,
                sharpe_reward,
                sortino_reward,
                mean_variance_reward,
            )
        except ImportError as e:
            raise ImportError(
                "aurumq-rl not found. Install with: "
                "pip install git+https://github.com/yupoet/aurumq-rl.git. "
                f"Error: {e}"
            ) from e

        # 加载面板
        loader = FactorPanelLoader(panel_path)
        panel = loader.load_panel()

        # 选择 reward 函数
        reward_fns = {
            "simple_return": simple_return_reward,
            "sharpe": sharpe_reward,
            "sortino": sortino_reward,
            "mean_variance": mean_variance_reward,
        }
        reward_fn = reward_fns.get(self.config.reward_fn, sharpe_reward)

        if self.config.env_type == "stock_picking":
            env_config = StockPickingConfig(
                top_k=self.config.top_k,
                forward_period=self.config.forward_period,
                cost_bps=self.config.cost_bps,
            )
            env = StockPickingEnv(
                panel=panel,
                config=env_config,
                reward_fn=reward_fn,
            )
        elif self.config.env_type == "portfolio_weight":
            env_config = PortfolioWeightConfig(
                reward_type=self.config.reward_fn,
            )
            env = PortfolioWeightEnv(
                panel=panel,
                config=env_config,
                reward_fn=reward_fn,
            )
        else:
            raise ValueError(f"Unknown env_type: {self.config.env_type}")

        return env
