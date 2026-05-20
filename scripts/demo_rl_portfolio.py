#!/usr/bin/env python3
"""RL 组合选股回测演示

用法:
    # 构建面板 (数据准备, 只需运行一次)
    python3 scripts/demo_rl_portfolio.py --build-panel

    # 训练 RL 模型 (需 aurumq-rl 已安装)
    python3 scripts/demo_rl_portfolio.py --train

    # 使用预训练 ONNX 模型回测
    python3 scripts/demo_rl_portfolio.py --backtest

    # 全流程
    python3 scripts/demo_rl_portfolio.py --build-panel --train --backtest

    # 指定参数
    python3 scripts/demo_rl_portfolio.py --train --universe all --timesteps 100000
    python3 scripts/demo_rl_portfolio.py --backtest --top-k 10 --rebalance-freq M
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from rl.config import RLConfig


def run_build_panel(args):
    """构建面板 parquet"""
    from rl.data_adapter import DataAdapter

    config = RLConfig(
        data_dir=str(PROJECT_ROOT / "data"),
        panel_dir=str(PROJECT_ROOT / "data/panels"),
        universe=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        max_stocks=args.max_stocks,
    )
    adapter = DataAdapter(config)
    panel_path = adapter.build_panel(force=args.force)
    print(f"Panel saved: {panel_path}")

    # 验证 schema
    df = adapter.load_panel()
    print(f"  Rows: {len(df)}, Stocks: {df['ts_code'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")


def run_train(args):
    """训练 RL 模型"""
    from rl.trainer import RLTrainer

    config = RLConfig(
        data_dir=str(PROJECT_ROOT / "data"),
        panel_dir=str(PROJECT_ROOT / "data/panels"),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints/rl"),
        universe=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        env_type=args.env_type,
        reward_fn=args.reward,
        top_k=args.top_k,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
    )
    trainer = RLTrainer(config)
    output_dir = trainer.train()
    print(f"Training complete. Output: {output_dir}")


def run_backtest(args):
    """使用 ONNX 模型回测"""
    from data.storage import ParquetStorage
    from rl.rl_strategy import RLStrategy
    from strategies.engine import BacktestConfig
    from strategies.portfolio_engine import (
        EqualWeightAllocation,
        PortfolioEngine,
        RebalanceConfig,
    )

    onnx_path = args.model or str(PROJECT_ROOT / "checkpoints/rl/policy.onnx")
    if not Path(onnx_path).exists():
        print(f"ONNX model not found: {onnx_path}")
        print("Run --train first, or specify --model path")
        return

    config = RLConfig(
        data_dir=str(PROJECT_ROOT / "data"),
        onnx_model_path=onnx_path,
        inference_top_k=args.top_k,
        rebalance_freq=args.rebalance_freq,
        signal_mode=args.signal_mode,
    )

    # 加载数据
    storage = ParquetStorage(str(PROJECT_ROOT / "data"))
    symbols = storage.list_symbols()
    if args.universe != "all":
        symbols = symbols[:args.max_stocks]

    data = {}
    for sym in symbols:
        df = storage.load(sym, "1d")
        if df is not None and len(df) > 60:
            if args.start_date:
                df = df[df.index >= pd.Timestamp(args.start_date)]
            if args.end_date:
                df = df[df.index <= pd.Timestamp(args.end_date)]
            data[sym] = df

    if not data:
        print("No data loaded. Check data directory.")
        return

    print(f"Loaded {len(data)} stocks for backtest")

    # 回测
    strategy = RLStrategy(config)
    bt_config = BacktestConfig.from_yaml(str(PROJECT_ROOT / "config/settings.yaml"))
    engine = PortfolioEngine(
        config=bt_config,
        allocation=EqualWeightAllocation(),
        rebalance=RebalanceConfig(frequency=args.rebalance_freq),
    )
    result = engine.run(strategy, data)

    # 输出结果
    print(f"\n{'='*50}")
    print(f"Backtest Results ({len(data)} stocks)")
    print(f"{'='*50}")
    print(f"Total Return:   {result.total_return:.2%}")
    print(f"Annual Return:  {result.annual_return:.2%}")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:   {result.max_drawdown:.2%}")
    print(f"Win Rate:       {result.win_rate:.2%}")
    print(f"Total Trades:   {result.total_trades}")


def main():
    parser = argparse.ArgumentParser(description="RL Portfolio Demo")
    parser.add_argument("--build-panel", action="store_true", help="Build panel parquet from per-stock data")
    parser.add_argument("--train", action="store_true", help="Train RL model")
    parser.add_argument("--backtest", action="store_true", help="Run backtest with ONNX model")
    parser.add_argument("--universe", default="all", help="Stock universe: all / csi300")
    parser.add_argument("--max-stocks", type=int, default=300, help="Max stocks in universe")
    parser.add_argument("--env-type", default="stock_picking", choices=["stock_picking", "portfolio_weight"])
    parser.add_argument("--reward", default="sharpe", choices=["simple_return", "sharpe", "sortino", "mean_variance"])
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel training environments")
    parser.add_argument("--top-k", type=int, default=10, help="Number of stocks to select")
    parser.add_argument("--rebalance-freq", default="M", choices=["W", "M", "Q"], help="Rebalance frequency")
    parser.add_argument("--signal-mode", default="top_k", choices=["top_k", "quantile"], help="Signal generation mode")
    parser.add_argument("--start-date", default=None, help="Data start date")
    parser.add_argument("--end-date", default=None, help="Data end date")
    parser.add_argument("--model", default=None, help="Path to ONNX model")
    parser.add_argument("--force", action="store_true", help="Force rebuild panel")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.build_panel:
        run_build_panel(args)
    if args.train:
        run_train(args)
    if args.backtest:
        run_backtest(args)

    if not any([args.build_panel, args.train, args.backtest]):
        parser.print_help()


if __name__ == "__main__":
    main()
