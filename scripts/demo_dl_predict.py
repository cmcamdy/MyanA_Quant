#!/usr/bin/env python3
"""深度学习涨跌预测 Demo

用法:
    python3 scripts/demo_dl_predict.py
    python3 scripts/demo_dl_predict.py --industry "C32有色金属冶炼和压延加工业"
    python3 scripts/demo_dl_predict.py --industry "J66货币金融服务"
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dl.config import DLConfig, CLASS_LABELS
from dl.trainer import Trainer

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("demo")


def get_symbols_by_industry(industry_name: str) -> list:
    """通过行业名称获取股票列表"""
    from data.industry import IndustryLookup
    lk = IndustryLookup(str(PROJECT_ROOT / "data"))
    symbols = lk.get_stocks(industry_name)
    if not symbols:
        logger.error(f"行业 '{industry_name}' 未找到或无股票")
        sys.exit(1)
    return symbols


def main():
    parser = argparse.ArgumentParser(description='深度学习涨跌预测 Demo')
    parser.add_argument('--industry', type=str, default='C32有色金属冶炼和压延加工业',
                        help='行业名称（从 industry-stock 数据中选取）')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=1)
    args = parser.parse_args()

    # 按行业选股
    symbols = get_symbols_by_industry(args.industry)
    logger.info(f"行业 '{args.industry}' 共 {len(symbols)} 只股票: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")

    config = DLConfig(
        data_dir=str(PROJECT_ROOT / "data"),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "demo"),
        window=args.window,
        horizon=args.horizon,
        hidden_dim=128,
        epochs=args.epochs,
        batch_size=32,
        early_stopping_patience=3,
    )

    logger.info("=" * 60)
    logger.info("深度学习涨跌预测 Demo")
    logger.info(f"行业: {args.industry}")
    logger.info(f"股票数: {len(symbols)}")
    logger.info(f"配置: window={config.window}, horizon={config.horizon}, epochs={config.epochs}")
    logger.info("=" * 60)

    # 训练
    trainer = Trainer(config)
    logger.info("\n--- 开始训练 ---")
    trainer.train(symbols)

    # 回归测试
    logger.info("\n--- 回归测试 ---")
    result = trainer.evaluate(symbols)
    print(f"\n测试集准确率: {result['accuracy']:.4f}")
    print(f"\n分类报告:\n{result['classification_report']}")
    print(f"混淆矩阵:\n{result['confusion_matrix']}")

    # 单只股票预测（选一个有数据的股票）
    pred_symbol = None
    for sym in symbols:
        test_path = Path(config.data_dir) / ('sh' if sym.endswith('.SH') else 'sz') / sym.split('.')[0] / '1d.parquet'
        if test_path.exists():
            pred_symbol = sym
            break

    if pred_symbol:
        logger.info(f"\n--- 单只股票预测: {pred_symbol} ---")
        pred = trainer.predict(pred_symbol)
        print(f"\n{pred_symbol} 预测结果:")
        print(f"  预测分类: {pred['label']}")
        print(f"  各分类概率:")
        for i, (label, prob) in enumerate(zip(CLASS_LABELS, pred['probabilities'])):
            bar = "#" * int(prob * 50)
            print(f"    {label}: {prob:.3f} {bar}")
    else:
        logger.warning("没有找到有数据的股票用于预测演示")


if __name__ == "__main__":
    main()
