"""Trainer 单元测试"""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from dl.config import DLConfig
from dl.trainer import Trainer


class TestTrainer:
    @pytest.fixture
    def config(self, tmp_path):
        return DLConfig(
            window=30,
            horizon=1,
            hidden_dim=64,
            epochs=2,
            batch_size=8,
            early_stopping_patience=3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            data_dir=str(tmp_path / "data"),
        )

    @pytest.fixture
    def trainer(self, config):
        return Trainer(config)

    def test_create_model(self, trainer):
        model = trainer._create_model(num_features=10)
        assert isinstance(model, torch.nn.Module)
        # 输入 [1, 10, 30] → 输出 [1, 6]
        x = torch.randn(1, 10, 30)
        out = model(x)
        assert out.shape == (1, 6)

    def test_split_dataset(self, trainer):
        from dl.dataset import StockDataset
        features = np.random.randn(100, 10, 30).astype(np.float32)
        labels = np.random.randint(0, 6, size=100).astype(np.int64)
        dataset = StockDataset(features, labels)

        # 模拟 5 只股票，每只 20 个样本
        stock_boundaries = [
            ("A", 0, 20), ("B", 20, 40), ("C", 40, 60), ("D", 60, 80), ("E", 80, 100),
        ]

        train_ds, val_ds, test_ds = trainer._split_dataset(dataset, stock_boundaries)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_train_epoch(self, trainer):
        from dl.dataset import StockDataset
        from torch.utils.data import DataLoader

        features = np.random.randn(50, 10, 30).astype(np.float32)
        labels = np.random.randint(0, 6, size=50).astype(np.int64)
        dataset = StockDataset(features, labels)
        loader = DataLoader(dataset, batch_size=8)

        model = trainer._create_model(num_features=10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        loss, acc = trainer._train_epoch(model, loader, optimizer, criterion)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 1

    def test_evaluate(self, trainer):
        from dl.dataset import StockDataset
        from torch.utils.data import DataLoader

        features = np.random.randn(50, 10, 30).astype(np.float32)
        labels = np.random.randint(0, 6, size=50).astype(np.int64)
        dataset = StockDataset(features, labels)
        loader = DataLoader(dataset, batch_size=8)

        model = trainer._create_model(num_features=10)
        criterion = torch.nn.CrossEntropyLoss()

        loss, acc = trainer._evaluate(model, loader, criterion)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1

    def test_save_load_checkpoint(self, trainer, tmp_path):
        model = trainer._create_model(num_features=10)
        trainer.model = model
        trainer._best_val_loss = 1.5

        ckpt_dir = Path(trainer.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer._save_checkpoint(epoch=5, is_best=True)

        best_path = ckpt_dir / "best_model.pt"
        assert best_path.exists()

        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 5
        assert checkpoint['best_val_loss'] == 1.5
