"""Meta Strategy 训练器"""

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

from .meta_config import MetaConfig
from .meta_model import MetaModel
from .meta_signal_collector import MetaSignalCollector
from .dataset import StockDataset

logger = logging.getLogger("dl.meta_trainer")


def _create_criterion(loss_type: str, huber_delta: float = 1.0) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"未知 loss_type: {loss_type}")


class MetaTrainer:
    """Meta Strategy 训练器"""

    def __init__(self, config: MetaConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collector = MetaSignalCollector(config)
        self.model: Optional[MetaModel] = None
        self._best_val_loss = float('inf')
        self._last_attn_weights: Optional[np.ndarray] = None

    NUM_MKT_FEATURES = 3  # ret1d, ret5d, volchg

    def _create_model(self, num_channels: Optional[int] = None) -> MetaModel:
        """创建模型

        Args:
            num_channels: 总通道数 (策略+市场特征), 默认 num_strategies + 3
        """
        if num_channels is None:
            num_channels = self.config.num_strategies + self.NUM_MKT_FEATURES

        # 先验偏置: 策略通道用用户指定先验, 市场特征通道用均匀
        prior = self.config.strategy_prior
        if prior is not None:
            mkt_prior = [1.0] * self.NUM_MKT_FEATURES
            prior = list(prior) + mkt_prior

        model = MetaModel(
            num_strategies=num_channels,
            window=self.config.window,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            strategy_prior=prior,
        )
        return model.to(self.device)

    def _split_dataset(self, dataset, stock_boundaries):
        """按时间划分，返回 (train_ds, val_ds, test_ds, train_idx, val_idx, test_idx)"""
        train_idx, val_idx, test_idx = [], [], []
        for sym, start, end in stock_boundaries:
            n = end - start
            t_end = start + int(n * self.config.train_ratio)
            v_end = start + int(n * (self.config.train_ratio + self.config.val_ratio))
            train_idx.extend(range(start, t_end))
            val_idx.extend(range(t_end, v_end))
            test_idx.extend(range(v_end, end))

        return (
            Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx),
            train_idx, val_idx, test_idx,
        )

    def _train_epoch(self, model, loader, optimizer, criterion, pbar=None):
        model.train()
        total_loss = 0.0
        total = 0
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            pred, _ = model(features)
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
    def _evaluate(self, model, loader, criterion, pbar=None):
        model.eval()
        total_loss = 0.0
        total = 0
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            pred, _ = model(features)
            loss = criterion(pred, labels)
            total_loss += loss.item() * features.size(0)
            total += features.size(0)
            if pbar is not None:
                pbar.set_postfix(loss=f"{total_loss / total:.6f}")
                pbar.update(features.size(0))
        return total_loss / total

    def train(self, symbols: Optional[List[str]] = None):
        """完整训练流程"""
        logger.info(f"开始 Meta 训练, {self.config.num_strategies} 个子策略, 设备: {self.device}")

        # 采集信号 + 构建特征 (不做内部标准化)
        features, labels, _, stock_boundaries = self.collector.build_all(
            symbols, normalize_features=False
        )
        logger.info(f"特征维度: {features.shape[1]} × {features.shape[2]}, 样本数: {len(labels)}")

        # 数据集划分
        dataset = StockDataset(features, labels)
        train_ds, val_ds, test_ds, train_idx, val_idx, test_idx = self._split_dataset(
            dataset, stock_boundaries
        )

        # 在训练集上 fit 特征标准化 (避免数据泄露)
        train_features = features[train_idx]
        self.collector.normalize_features(train_features, fit=True)
        features = self.collector.normalize_features(features, fit=False)

        # 在训练集上 fit 标签归一化
        train_labels = labels[train_idx]
        self.collector.normalize_labels(train_labels, fit=True)
        labels = self.collector.normalize_labels(labels, fit=False)

        # 用归一化后的数据重建 dataset
        dataset = StockDataset(features, labels)
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size)

        # 模型 (通道数 = 策略数 + 市场特征数)
        num_channels = features.shape[1]
        self.model = self._create_model(num_channels=num_channels)
        criterion = _create_criterion(self.config.loss_type, self.config.huber_delta)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
        )

        start_epoch = 0
        if self.config.resume:
            start_epoch, self._best_val_loss = self._load_for_training(optimizer)

        # 训练循环
        patience_counter = 0
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        epoch_pbar = tqdm(range(start_epoch, self.config.epochs), desc="Meta训练", unit="epoch")
        for epoch in epoch_pbar:
            train_pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1} 训练", leave=False)
            train_loss = self._train_epoch(self.model, train_loader, optimizer, criterion, pbar=train_pbar)
            train_pbar.close()

            val_pbar = tqdm(total=len(val_loader.dataset), desc=f"Epoch {epoch+1} 验证", leave=False)
            val_loss = self._evaluate(self.model, val_loader, criterion, pbar=val_pbar)
            val_pbar.close()

            epoch_pbar.set_postfix(t_loss=f"{train_loss:.6f}", v_loss=f"{val_loss:.6f}")
            logger.info(f"Epoch {epoch+1}/{self.config.epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

            scheduler.step(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"早停: 连续 {patience_counter} epoch 未改善")
                break

        elapsed = time.time() - start_time
        logger.info(f"训练完成, 耗时 {elapsed:.0f}s, best_val_loss={self._best_val_loss:.6f}")

        # 保存 scaler
        self.collector.save_scaler(str(checkpoint_dir / "meta_scaler.npz"))

        # 保存元信息
        self._save_meta(checkpoint_dir, symbols)

        # 打印策略权重
        self._print_strategy_weights()

    def evaluate(self, symbols: Optional[List[str]] = None) -> Dict:
        """评估"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        self.collector.load_scaler(str(checkpoint_dir / "meta_scaler.npz"))

        features, labels, _, stock_boundaries = self.collector.build_all(symbols, normalize_features=False)
        features = self.collector.normalize_features(features, fit=False)
        labels_raw = labels.copy()
        labels = self.collector.normalize_labels(labels, fit=False)

        dataset = StockDataset(features, labels)
        _, _, test_ds, _, _, test_idx = self._split_dataset(dataset, stock_boundaries)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size)

        self.model = self._create_model()
        best_path = checkpoint_dir / "best_model.pt"
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        all_preds = []
        all_labels = []
        all_weights = []
        with torch.no_grad():
            for feat, lab in test_loader:
                feat = feat.to(self.device)
                pred, weights = self.model(feat)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(lab.numpy())
                all_weights.append(weights.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights, axis=0)

        # 反归一化
        all_preds = self.collector.denormalize_labels(all_preds)
        all_labels = self.collector.denormalize_labels(all_labels)

        mse = float(np.mean((all_preds - all_labels) ** 2))
        mae = float(np.mean(np.abs(all_preds - all_labels)))
        rmse = float(np.sqrt(mse))
        direction_acc = float(np.mean((all_preds * all_labels) > 0))
        corr = float(np.corrcoef(all_preds, all_labels)[0, 1]) if len(all_preds) > 1 else 0.0

        # 平均策略权重
        avg_weights = all_weights.mean(axis=0)

        result = {
            'mse': mse, 'rmse': rmse, 'mae': mae,
            'direction_accuracy': direction_acc,
            'correlation': corr,
            'predictions': all_preds,
            'true_labels': all_labels,
            'avg_strategy_weights': avg_weights,
        }

        logger.info(f"测试 MSE={mse:.6f}, RMSE={rmse:.6f}, 方向准确率={direction_acc:.4f}")
        logger.info(f"平均策略权重: {avg_weights}")
        return result

    def predict(self, signal_features: np.ndarray) -> Dict:
        """单样本预测

        Args:
            signal_features: [K*2, window] 策略信号特征

        Returns:
            {'predicted_return': float, 'direction': str, 'strategy_weights': list}
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        self.collector.load_scaler(str(checkpoint_dir / "meta_scaler.npz"))

        if self.model is None:
            self.model = self._create_model()
            best_path = checkpoint_dir / "best_model.pt"
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 标准化
        feat = self.collector.normalize_features(signal_features[np.newaxis], fit=False)
        sample = torch.from_numpy(feat).to(self.device)

        with torch.no_grad():
            pred, weights = self.model(sample)
            pred_return = self.collector.denormalize_labels(pred.cpu().numpy())[0]
            attn_weights = weights.cpu().numpy()[0]

        return {
            'predicted_return': float(pred_return),
            'direction': '涨' if pred_return > 0 else '跌',
            'strategy_weights': attn_weights.tolist(),
        }

    def get_strategy_weights(self) -> Optional[np.ndarray]:
        """获取最近一次前向传播的策略注意力权重"""
        return self._last_attn_weights

    def _print_strategy_weights(self):
        """打印策略权重"""
        if self.model is None:
            return
        self.model.eval()
        # 用一个全零输入触发 (shape: [1, num_strategies, window])
        dummy = torch.zeros(1, self.config.num_strategies, self.config.window).to(self.device)
        with torch.no_grad():
            _, weights = self.model(dummy)
        w = weights.cpu().numpy()[0]
        logger.info("策略注意力权重 (全零输入):")
        for i, strat in enumerate(self.config.strategies):
            logger.info(f"  {type(strat).__name__}: {w[i]:.4f}")

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
        checkpoint_dir = Path(self.config.checkpoint_dir)
        best_path = checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
            if not checkpoints:
                logger.warning("未找到断点文件")
                return 0, float('inf')
            best_path = checkpoints[-1]

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        scaler_path = checkpoint_dir / "meta_scaler.npz"
        if scaler_path.exists():
            self.collector.load_scaler(str(scaler_path))

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"从 epoch {start_epoch} 续训, best_val_loss={best_val_loss:.6f}")
        return start_epoch, best_val_loss

    def _save_meta(self, checkpoint_dir: Path, symbols):
        meta = {
            'num_strategies': self.config.num_strategies,
            'strategy_names': [type(s).__name__ for s in self.config.strategies],
            'window': self.config.window,
            'horizon': self.config.horizon,
            'hidden_dim': self.config.hidden_dim,
            'train_symbols': symbols or 'all',
        }
        path = checkpoint_dir / "train_meta.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
