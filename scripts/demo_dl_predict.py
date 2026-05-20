#!/usr/bin/env python3
"""深度学习收益率预测 Demo

用法:
    python3 scripts/demo_dl_predict.py
    python3 scripts/demo_dl_predict.py --industry "C32有色金属冶炼和压延加工业"
    python3 scripts/demo_dl_predict.py --industry "J66货币金融服务" --backtest
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dl.config import DLConfig
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


def run_predict(args):
    """训练 + 预测模式"""
    symbols = get_symbols_by_industry(args.industry)
    logger.info(f"行业 '{args.industry}' 共 {len(symbols)} 只股票: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")

    config = DLConfig(
        data_dir=str(PROJECT_ROOT / "data"),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "demo"),
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        horizon=args.horizon,
        hidden_dim=128,
        epochs=args.epochs,
        batch_size=32,
        loss_type=args.loss,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        early_stopping_patience=3,
    )

    logger.info("=" * 60)
    logger.info("深度学习收益率预测 Demo (回归模式)")
    logger.info(f"行业: {args.industry}")
    logger.info(f"股票数: {len(symbols)}")
    logger.info(f"数据范围: {config.start_date or '最早'} ~ {config.end_date or '最新'}")
    logger.info(f"配置: window={config.window}, horizon={config.horizon}, epochs={config.epochs}, loss={config.loss_type}")
    logger.info(f"信号阈值: buy>{config.buy_threshold:.4f}, sell<{config.sell_threshold:.4f}")
    logger.info("=" * 60)

    # 训练
    trainer = Trainer(config)
    logger.info("\n--- 开始训练 ---")
    trainer.train(symbols)

    # 回归评估
    logger.info("\n--- 回归测试 ---")
    result = trainer.evaluate(symbols)
    print(f"\n测试集 MSE: {result['mse']:.6f}")
    print(f"测试集 RMSE: {result['rmse']:.6f}")
    print(f"测试集 MAE: {result['mae']:.6f}")
    print(f"方向准确率: {result['direction_accuracy']:.4f}")
    print(f"相关系数: {result['correlation']:.4f}")

    # 单只股票预测
    pred_symbol = None
    for sym in symbols:
        test_path = Path(config.data_dir) / ('sh' if sym.endswith('.SH') else 'sz') / sym.split('.')[0] / '1d.parquet'
        if test_path.exists():
            pred_symbol = sym
            break

    if pred_symbol:
        logger.info(f"\n--- 单只股票预测: {pred_symbol} ---")
        pred = trainer.predict(pred_symbol)
        pct = pred['predicted_return'] * 100
        direction = pred['direction']
        print(f"\n{pred_symbol} 预测结果:")
        print(f"  预测收益率: {pct:+.2f}%")
        print(f"  方向: {direction}")
        if pct > config.buy_threshold * 100:
            print(f"  信号: BUY (超过买入阈值 {config.buy_threshold*100:.2f}%)")
        elif pct < config.sell_threshold * 100:
            print(f"  信号: SELL (低于卖出阈值 {config.sell_threshold*100:.2f}%)")
        else:
            print(f"  信号: HOLD (在阈值范围内)")
    else:
        logger.warning("没有找到有数据的股票用于预测演示")


def run_backtest(args):
    """训练 + 回测模式"""
    from data.storage import ParquetStorage
    from dl.dl_strategy import DLStrategy
    from strategies.engine import BacktestEngine, BacktestConfig

    symbols = get_symbols_by_industry(args.industry)

    config = DLConfig(
        data_dir=str(PROJECT_ROOT / "data"),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "demo"),
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        horizon=args.horizon,
        hidden_dim=128,
        epochs=args.epochs,
        batch_size=32,
        loss_type=args.loss,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        early_stopping_patience=3,
    )

    # 先训练
    trainer = Trainer(config)
    logger.info("--- 训练模型 ---")
    trainer.train(symbols)

    # 选一只股票做回测
    storage = ParquetStorage(str(PROJECT_ROOT / "data"))
    test_symbol = None
    for sym in symbols:
        df = storage.load(sym, '1d')
        if df is not None and len(df) > config.window + 60:
            test_symbol = sym
            break

    if test_symbol is None:
        logger.error("没有找到足够数据的股票用于回测")
        return

    logger.info(f"\n--- 回测: {test_symbol} ---")
    df = storage.load(test_symbol, '1d')

    # 创建 DL 策略
    strategy = DLStrategy(config)

    # 运行回测
    bt_config = BacktestConfig.from_yaml(str(PROJECT_ROOT / "config" / "settings.yaml"))
    engine = BacktestEngine(config=bt_config)
    result = engine.run(strategy, df, symbol=test_symbol)

    print(f"\n{'='*40}")
    print(f"DL 策略回测结果: {test_symbol}")
    print(f"{'='*40}")
    print(f"总收益率:   {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率:   {result.sharpe_ratio:.2f}")
    print(f"最大回撤:   {result.max_drawdown:.2%}")
    print(f"胜率:       {result.win_rate:.2%}")
    print(f"盈亏比:     {result.profit_loss_ratio:.2f}")
    print(f"交易次数:   {result.total_trades}")


def main():
    parser = argparse.ArgumentParser(description='深度学习收益率预测 Demo')
    parser.add_argument('--industry', type=str, default='C32有色金属冶炼和压延加工业',
                        help='行业名称')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--loss', type=str, default='huber', choices=['mse', 'mae', 'huber'],
                        help='损失函数类型')
    parser.add_argument('--buy-threshold', type=float, default=0.005,
                        help='买入阈值 (预测收益率超过此值则买入)')
    parser.add_argument('--sell-threshold', type=float, default=-0.005,
                        help='卖出阈值 (预测收益率低于此值则卖出)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='数据起始日期 (如 2013-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='数据截止日期 (如 2026-01-01)')
    parser.add_argument('--backtest', action='store_true',
                        help='训练后运行回测')
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
