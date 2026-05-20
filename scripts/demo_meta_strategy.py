#!/usr/bin/env python3
"""Meta Strategy Demo: 学习型组合策略

用法:
    python3 scripts/demo_meta_strategy.py
    python3 scripts/demo_meta_strategy.py --backtest
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dl.meta_config import MetaConfig
from dl.meta_trainer import MetaTrainer

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("demo_meta")


def build_strategies():
    """构建子策略组合"""
    from strategies.examples.ma_cross import MACrossStrategy
    from strategies.examples.sar import SARStrategy
    from strategies.examples.rsi import RSIStrategy
    from strategies.examples.bollinger import BollingerStrategy
    from strategies.examples.macd import MACDStrategy
    from strategies.examples.kdj import KDJStrategy
    from strategies.examples.wr import WRStrategy
    from strategies.examples.cci import CCIStrategy
    from strategies.examples.keltner import KeltnerStrategy
    from strategies.examples.obv import OBVStrategy
    from strategies.examples.mfi import MFIStrategy
    from strategies.examples.vwap import VWAPStrategy

    return [
        MACrossStrategy(fast_period=5, slow_period=20),
        SARStrategy(),
        RSIStrategy(period=14, oversold=30, overbought=70),
        BollingerStrategy(period=20),
        MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        KDJStrategy(n=9, m1=3, m2=3, oversold=20, overbought=80),
        WRStrategy(period=14, oversold=-80, overbought=-20),
        CCIStrategy(period=14, oversold=-100, overbought=100),
        KeltnerStrategy(ema_period=20, atr_period=10, num_atr=1.5),
        OBVStrategy(ma_period=20),
        MFIStrategy(period=14, oversold=20, overbought=80),
        VWAPStrategy(),
    ]


def build_strategy_prior():
    """策略先验权重: 用户根据经验设定，模型训练后可调整

    顺序与 build_strategies() 一一对应。
    值越大 → 初始注意力权重越高。
    不设置或全1则退化为均匀初始化。
    """
    return [
        # MA, SAR, RSI, Bollinger — 经典策略，先验较高
        3, 2, 2, 2,
        # MACD — 趋势确认利器，先验最高
        4,
        # KDJ, WR, CCI — 动量类，中等
        1.5, 1.5, 1.5,
        # Keltner, OBV, MFI, VWAP — 辅助参考
        1, 1, 1, 1,
    ]


def get_symbols(args) -> list:
    """根据参数获取股票列表"""
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        logger.info(f"指定股票 {len(symbols)} 只: {symbols}")
        return symbols
    from data.industry import IndustryLookup
    lk = IndustryLookup(str(PROJECT_ROOT / "data"))
    symbols = lk.get_stocks(args.industry)
    if not symbols:
        logger.error(f"行业 '{args.industry}' 未找到或无股票")
        sys.exit(1)
    logger.info(f"行业 '{args.industry}' 共 {len(symbols)} 只股票")
    return symbols


def run_train(args):
    strategies = build_strategies()
    strategy_prior = build_strategy_prior()
    symbols = get_symbols(args)

    config = MetaConfig(
        strategies=strategies,
        strategy_prior=strategy_prior,
        data_dir=str(PROJECT_ROOT / "data"),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "meta"),
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        horizon=args.horizon,
        hidden_dim=64,
        epochs=args.epochs,
        batch_size=32,
        loss_type=args.loss,
        early_stopping_patience=10,
    )

    logger.info("=" * 60)
    logger.info("Meta Strategy: 学习型组合策略")
    logger.info(f"子策略: {[type(s).__name__ for s in strategies]}")
    if strategy_prior:
        logger.info(f"先验权重: {dict(zip([type(s).__name__ for s in strategies], strategy_prior))}")
    logger.info(f"数据范围: {config.start_date or '最早'} ~ {config.end_date or '最新'}")
    logger.info(f"窗口: {config.window}, 预测周期: {config.horizon}, patience: {config.early_stopping_patience}")
    logger.info("=" * 60)

    # 训练
    trainer = MetaTrainer(config)
    trainer.train(symbols)

    # 评估
    logger.info("\n--- 回归评估 ---")
    result = trainer.evaluate(symbols)
    print(f"\n测试集 MSE: {result['mse']:.6f}")
    print(f"测试集 RMSE: {result['rmse']:.6f}")
    print(f"方向准确率: {result['direction_accuracy']:.4f}")
    print(f"相关系数: {result['correlation']:.4f}")
    print(f"\n策略权重:")
    for i, strat in enumerate(strategies):
        w = result['avg_strategy_weights'][i]
        bar = "#" * int(w * 50)
        print(f"  {type(strat).__name__:20s}: {w:.4f} {bar}")


def run_backtest(args):
    from data.storage import ParquetStorage
    from dl.meta_strategy import MetaStrategy
    from strategies.engine import BacktestEngine, BacktestConfig

    strategies = build_strategies()
    strategy_prior = build_strategy_prior()
    symbols = get_symbols(args)

    config = MetaConfig(
        strategies=strategies,
        strategy_prior=strategy_prior,
        data_dir=str(PROJECT_ROOT / "data"),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "meta"),
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        horizon=args.horizon,
        hidden_dim=64,
        epochs=args.epochs,
        batch_size=32,
        loss_type=args.loss,
        early_stopping_patience=10,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        signal_mode=args.signal_mode,
    )

    # 先训练
    trainer = MetaTrainer(config)
    logger.info("--- 训练 Meta 模型 ---")
    trainer.train(symbols)

    # 选股票回测
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

    meta = MetaStrategy(config)
    bt_config = BacktestConfig.from_yaml(str(PROJECT_ROOT / "config" / "settings.yaml"))
    engine = BacktestEngine(config=bt_config)
    result = engine.run(meta, df, symbol=test_symbol)

    print(f"\n{'='*40}")
    print(f"Meta Strategy 回测结果: {test_symbol}")
    print(f"{'='*40}")
    print(f"总收益率:   {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率:   {result.sharpe_ratio:.2f}")
    print(f"最大回撤:   {result.max_drawdown:.2%}")
    print(f"胜率:       {result.win_rate:.2%}")
    print(f"盈亏比:     {result.profit_loss_ratio:.2f}")
    print(f"交易次数:   {result.total_trades}")


def main():
    parser = argparse.ArgumentParser(description='Meta Strategy: 学习型组合策略')
    parser.add_argument('--industry', type=str, default='C32有色金属冶炼和压延加工业',
                        help='行业名称 (与 --symbols 二选一)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='直接指定股票列表, 逗号分隔 (如 601600.SH,000807.SZ)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--loss', type=str, default='huber', choices=['mse', 'mae', 'huber'])
    parser.add_argument('--start-date', type=str, default=None,
                        help='数据起始日期 (如 2013-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='数据截止日期 (如 2026-01-01)')
    parser.add_argument('--buy-threshold', type=float, default=0.002,
                        help='买入阈值 (fixed模式下生效, 默认0.002=0.2%%)')
    parser.add_argument('--sell-threshold', type=float, default=-0.002,
                        help='卖出阈值 (fixed模式下生效, 默认-0.002=-0.2%%)')
    parser.add_argument('--signal-mode', type=str, default='adaptive',
                        choices=['fixed', 'adaptive'],
                        help='信号模式: fixed=固定阈值, adaptive=自适应百分位(推荐)')
    parser.add_argument('--backtest', action='store_true')
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
