"""ONNX Runtime 推理封装

使用 aurumq-rl 导出的 policy.onnx 进行 CPU 推理,
仅需 onnxruntime, 不依赖 PyTorch。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RLInference:
    """基于 ONNX Runtime 的 RL 策略推理

    加载 aurumq-rl 的 export_sb3_policy_to_onnx() 产出的
    policy.onnx + metadata.json, 执行截面选股推理。
    """

    def __init__(self, onnx_path: str):
        self._onnx_path = Path(onnx_path)
        self._metadata: Optional[Dict] = None
        self._session = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._obs_shape: Optional[tuple] = None

    def _ensure_loaded(self):
        if self._session is not None:
            return

        if not self._onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._onnx_path}")

        # 加载元数据
        meta_path = self._onnx_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
            self._obs_shape = tuple(self._metadata.get("obs_shape", []))
            logger.info("Model metadata: %s", self._metadata)

        # 创建 ONNX Runtime 会话
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(self._onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # 获取输入输出名称
        input_info = self._session.get_inputs()[0]
        self._input_name = input_info.name
        output_info = self._session.get_outputs()[0]
        self._output_name = output_info.name

        logger.info(
            "ONNX session created: input=%s, output=%s",
            self._input_name, self._output_name,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """执行推理

        Args:
            obs: 观测张量, shape 取决于模型 (通常为 [1, n_stocks * n_factors])

        Returns:
            action scores, shape [1, n_stocks]
        """
        self._ensure_loaded()

        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        elif obs.ndim == 2 and obs.shape[0] != 1:
            # 批量推理: 保持原样
            pass

        result = self._session.run(
            [self._output_name],
            {self._input_name: obs.astype(np.float32)},
        )
        return result[0]

    def predict_scores(self, obs: np.ndarray) -> Dict[int, float]:
        """推理并返回每只股票的分数

        Returns:
            {stock_index: score} 字典
        """
        raw = self.predict(obs)
        scores = raw.flatten()
        return {i: float(s) for i, s in enumerate(scores)}

    @property
    def metadata(self) -> Optional[Dict]:
        if self._metadata is None:
            self._ensure_loaded()
        return self._metadata

    @property
    def obs_shape(self) -> Optional[tuple]:
        if self._obs_shape is None:
            self._ensure_loaded()
        return self._obs_shape
