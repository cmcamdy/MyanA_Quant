"""训练器：训练/断点续训/增量训练/评估

数据划分策略：每只股票内部按时间顺序划分
  对每只股票，按时间顺序取前 train_ratio 为训练，中间 val_ratio 为验证，剩余为测试。
  所有股票的数据都会出现在三个集合中，保证样本多样性。
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from .config import DLConfig
from .dataset import StockDataset
from .feature_builder import FeatureBuilder
from .model import PricePredictor

logger = logging.getLogger("dl.trainer")


def _create_criterion(loss_type: str) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=0.01)
    else:
        raise ValueError(f"未知 loss_type: {loss_type}, 可选: mse, mae, huber")


class Trainer:
    """深度学习训练器 (回归模式)"""

    def __init__(self, config: DLConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.builder = FeatureBuilder(config)
        self.model: Optional[PricePredictor] = None
        self._best_val_loss = float('inf')

    def _create_model(self, num_features: int) -> PricePredictor:
        model = PricePredictor(
            num_features=num_features,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            window=self.config.window,
        )
        return model.to(self.device)

    def _split_dataset(self, dataset: StockDataset, stock_boundaries: List[Tuple[str, int, int]]):
        """每只股票内部按时间顺序划分 train/val/test"""
        train_indices = []
        val_indices = []
        test_indices = []

        for sym, start, end in stock_boundaries:
            n = end - start
            train_end = start + int(n * self.config.train_ratio)
            val_end = start + int(n * (self.config.train_ratio + self.config.val_ratio))
            train_indices.extend(range(start, train_end))
            val_indices.extend(range(train_end, val_end))
            test_indices.extend(range(val_end, end))

        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)

        logger.info(
            f"数据划分 (按时间): train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)} | "
            f"股票数: {len(stock_boundaries)}"
        )
        return train_ds, val_ds, test_ds

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer, criterion, pbar=None):
        model.train()
        total_loss = 0.0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            pred = model(features)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            total += features.size(0)

            if pbar is not None:
                pbar.set_postfix(loss=f"{total_loss / total:.6f}")
                pbar.update(features.size(0))

        return total_loss / total

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader, criterion, pbar=None):
        model.eval()
        total_loss = 0.0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            pred = model(features)
            loss = criterion(pred, labels)

            total_loss += loss.item() * features.size(0)
            total += features.size(0)

            if pbar is not None:
                pbar.set_postfix(loss=f"{total_loss / total:.6f}")
                pbar.update(features.size(0))

        return total_loss / total

    def train(self, symbols: Optional[List[str]] = None):
        """完整训练流程"""
        logger.info(f"开始训练 (回归模式), 设备: {self.device}")

        # 构建数据
        features, labels, _, stock_boundaries = self.builder.build_all(symbols)
        num_features = features.shape[1]
        logger.info(f"特征维度: {num_features} × {self.config.window} = {num_features * self.config.window}")

        # 创建数据集并划分
        dataset = StockDataset(features, labels)
        train_ds, val_ds, test_ds = self._split_dataset(dataset, stock_boundaries)

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size)

        # 创建模型
        self.model = self._create_model(num_features)
        criterion = _create_criterion(self.config.loss_type)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        start_epoch = 0
        # 断点续训
        if self.config.resume:
            start_epoch, self._best_val_loss = self._load_for_training(optimizer)

        # 训练循环
        patience_counter = 0
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        epoch_pbar = tqdm(range(start_epoch, self.config.epochs), desc="训练", unit="epoch")
        for epoch in epoch_pbar:
            # 训练
            train_total = len(train_loader.dataset)
            train_pbar = tqdm(total=train_total, desc=f"Epoch {epoch + 1} 训练", leave=False, unit="样本")
            train_loss = self._train_epoch(self.model, train_loader, optimizer, criterion, pbar=train_pbar)
            train_pbar.close()

            # 验证
            val_total = len(val_loader.dataset)
            val_pbar = tqdm(total=val_total, desc=f"Epoch {epoch + 1} 验证", leave=False, unit="样本")
            val_loss = self._evaluate(self.model, val_loader, criterion, pbar=val_pbar)
            val_pbar.close()

            epoch_pbar.set_postfix(t_loss=f"{train_loss:.6f}", v_loss=f"{val_loss:.6f}")
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

            # 保存最佳模型
            is_best = val_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # 早停
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"早停: 连续 {patience_counter} 个 epoch 验证损失未改善")
                break

        elapsed = time.time() - start_time
        logger.info(f"训练完成, 耗时 {elapsed:.0f}s, 最佳 val_loss={self._best_val_loss:.6f}")

        # 保存 scaler
        self.builder.save_scaler(str(checkpoint_dir / "scaler.npz"))

        # 保存配置
        self._save_config_meta(checkpoint_dir, symbols)

    def resume_train(self, symbols: Optional[List[str]] = None):
        """断点续训"""
        self.config.resume = True
        self.train(symbols)

    def incremental_train(self, symbols: Optional[List[str]] = None):
        """增量训练: 降低学习率 fine-tune"""
        self.config.incremental = True
        self.config.learning_rate *= self.config.incremental_lr_factor
        logger.info(f"增量训练, 学习率调整为 {self.config.learning_rate:.6f}")
        self.train(symbols)

    def evaluate(self, symbols: Optional[List[str]] = None) -> Dict:
        """回归测试: 加载最佳模型，在测试集上评估"""
        checkpoint_dir = Path(self.config.checkpoint_dir)

        # 加载 scaler
        self.builder.load_scaler(str(checkpoint_dir / "scaler.npz"))

        # 构建数据（不做 fit）
        features, labels, _, stock_boundaries = self.builder.build_all(symbols)
        features = self.builder.normalize_features(features, fit=False)

        num_features = features.shape[1]
        dataset = StockDataset(features, labels)
        _, _, test_ds = self._split_dataset(dataset, stock_boundaries)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size)

        # 加载最佳模型
        self.model = self._create_model(num_features)
        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"未找到最佳模型: {best_path}")
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info("已加载最佳模型")

        # 预测
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for feat, lab in test_loader:
                feat = feat.to(self.device)
                pred = self.model(feat)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(lab.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 回归指标
        mse = float(np.mean((all_preds - all_labels) ** 2))
        mae = float(np.mean(np.abs(all_preds - all_labels)))
        rmse = float(np.sqrt(mse))
        # 方向准确率: 预测与真实同号的比例
        direction_acc = float(np.mean((all_preds * all_labels) > 0))
        # 相关系数
        corr = float(np.corrcoef(all_preds, all_labels)[0, 1]) if len(all_preds) > 1 else 0.0

        result = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_acc,
            'correlation': corr,
            'predictions': all_preds,
            'true_labels': all_labels,
        }

        logger.info(f"测试集 MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
        logger.info(f"方向准确率={direction_acc:.4f}, 相关系数={corr:.4f}")

        return result

    def predict(self, symbol: str, date: Optional[str] = None) -> Dict:
        """单只股票预测

        Args:
            symbol: 股票代码
            date: 指定日期（None 则用最新数据）

        Returns:
            {'predicted_return': float, 'direction': str}
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)

        # 加载 scaler
        self.builder.load_scaler(str(checkpoint_dir / "scaler.npz"))

        # 构建特征
        result = self.builder.build_stock(symbol)
        if result is None:
            raise ValueError(f"无法构建 {symbol} 的特征")

        features, _ = result
        features = self.builder.normalize_features(features, fit=False)

        # 选取指定日期或最后一个样本
        if date is not None:
            df = self.builder._load_parquet(symbol)
            if df is not None and date in df.index.strftime('%Y-%m-%d').tolist():
                idx = list(df.index.strftime('%Y-%m-%d')).index(date)
                sample_idx = idx - self.config.window
                if sample_idx < 0 or sample_idx >= len(features):
                    sample_idx = len(features) - 1
            else:
                sample_idx = len(features) - 1
        else:
            sample_idx = len(features) - 1

        sample = torch.from_numpy(features[sample_idx:sample_idx + 1]).to(self.device)

        # 加载模型
        num_features = features.shape[1]
        if self.model is None:
            self.model = self._create_model(num_features)
            best_path = checkpoint_dir / "best_model.pt"
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 预测
        with torch.no_grad():
            pred = self.model(sample).cpu().numpy()[0]

        direction = "涨" if pred > 0 else "跌"
        return {
            'predicted_return': float(pred),
            'direction': direction,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self._best_val_loss,
        }
        suffix = 'best_model' if is_best else f'checkpoint_epoch{epoch}'
        path = Path(self.config.checkpoint_dir) / f"{suffix}.pt"
        torch.save(checkpoint, path)

    def _load_for_training(self, optimizer):
        """加载断点用于续训，返回 (start_epoch, best_val_loss)"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        # 优先加载 best_model，其次找最新的 checkpoint
        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
            if not checkpoints:
                logger.warning("未找到断点文件，从头开始训练")
                return 0, float('inf')
            best_path = checkpoints[-1]

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载 scaler
        scaler_path = checkpoint_dir / "scaler.npz"
        if scaler_path.exists():
            self.builder.load_scaler(str(scaler_path))

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"从 epoch {start_epoch} 续训, best_val_loss={best_val_loss:.6f}")
        return start_epoch, best_val_loss

    def _save_config_meta(self, checkpoint_dir: Path, symbols):
        """保存训练元信息"""
        meta = {
            'window': self.config.window,
            'horizon': self.config.horizon,
            'hidden_dim': self.config.hidden_dim,
            'num_features': self.builder.num_features,
            'loss_type': self.config.loss_type,
            'train_symbols': symbols or 'all',
        }
        path = checkpoint_dir / "train_meta.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
