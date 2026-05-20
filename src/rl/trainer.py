"""RL 训练编排器

使用 SB3 PPO 训练 aurumq-rl 环境,
训练完成后导出 ONNX 模型供推理使用。

完整流程:
  DataAdapter.build_panel() → FactorPanelBuilder.compute()
  → EnvFactory.create() → PPO.train() → export ONNX

依赖: aurumq-rl, gymnasium, stable-baselines3
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import RLConfig
from .data_adapter import DataAdapter
from .factor_panel import FactorPanelBuilder
from .env_factory import EnvFactory

logger = logging.getLogger(__name__)

# 确保 aurumq-rl 可导入
_VENDOR_PATH = Path(__file__).resolve().parent.parent.parent / "vendor" / "aurumq-rl" / "src"
if _VENDOR_PATH.exists() and str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))


class RLTrainer:
    """RL 训练编排器"""

    def __init__(self, config: RLConfig):
        self.config = config

    def train(self, symbols: Optional[List[str]] = None) -> Path:
        """执行完整训练流程

        Args:
            symbols: 股票列表, None 则按 config.universe 解析

        Returns:
            模型输出目录路径
        """
        # Step 1: 构建面板
        logger.info("=== Step 1: Building panel ===")
        adapter = DataAdapter(self.config)
        panel_path = adapter.build_panel(symbols)
        panel_df = adapter.load_panel()

        # Step 2: 计算因子
        logger.info("=== Step 2: Computing factors ===")
        builder = FactorPanelBuilder(self.config)
        panel_with_factors = builder.compute(panel_df)

        # Step 3: 创建环境
        logger.info("=== Step 3: Creating environments ===")
        factory = EnvFactory(self.config)
        train_envs, val_env = factory.create_train_val(panel_with_factors)

        # Step 4: 训练
        logger.info("=== Step 4: Training PPO ===")
        model = self._create_model(train_envs)
        model.learn(total_timesteps=self.config.total_timesteps)

        # Step 5: 保存模型
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_dir / "ppo_model"))
        logger.info("SB3 model saved: %s", ckpt_dir / "ppo_model.zip")

        # Step 6: 导出 ONNX
        logger.info("=== Step 5: Exporting ONNX ===")
        self._export_onnx(model, ckpt_dir)

        # Step 7: 保存元数据
        self._save_metadata(ckpt_dir, symbols)

        logger.info("Training complete. Output: %s", ckpt_dir)
        return ckpt_dir

    def _create_model(self, train_envs):
        """创建 SB3 PPO 模型"""
        try:
            from aurumq_rl.policy import PerStockEncoderPolicy
            policy = PerStockEncoderPolicy
            logger.info("Using PerStockEncoderPolicy (Deep-Sets)")
        except ImportError:
            policy = "MlpPolicy"
            logger.warning("aurumq-rl policy not found, falling back to MlpPolicy")

        from stable_baselines3 import PPO

        model = PPO(
            policy=policy,
            env=train_envs,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            verbose=self.config.verbose,
            tensorboard_log=self.config.tensorboard_log,
        )
        return model

    def _export_onnx(self, model, ckpt_dir: Path):
        """导出 ONNX 模型"""
        try:
            from aurumq_rl.onnx_export import export_sb3_policy_to_onnx
            onnx_path = str(ckpt_dir / "policy.onnx")
            export_sb3_policy_to_onnx(model, onnx_path)
            logger.info("ONNX model exported: %s", onnx_path)
        except ImportError:
            logger.warning(
                "aurumq-rl export not available. "
                "Install aurumq-rl for ONNX export support."
            )
        except Exception as e:
            logger.warning("ONNX export failed: %s", e)

    def _save_metadata(self, ckpt_dir: Path, symbols: Optional[List[str]]):
        """保存训练元数据"""
        metadata = {
            "config": {
                "env_type": self.config.env_type,
                "top_k": self.config.top_k,
                "reward_fn": self.config.reward_fn,
                "total_timesteps": self.config.total_timesteps,
                "universe": self.config.universe,
                "factor_prefixes": self.config.factor_prefixes,
            },
            "symbols_count": len(symbols) if symbols else "all",
        }
        meta_path = ckpt_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
