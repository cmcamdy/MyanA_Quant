"""实时策略监控 CLI

Usage:
    python scripts/monitor_strategy.py \
        --symbols 600036.SH,000001.SZ \
        --freq 1d --interval 60 \
        --strategy ma --fast 5 --slow 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monitor import LiveEngine, MonitorConfig
from strategies.examples.ma_cross import MACrossStrategy
from strategies.risk import RiskConfig


def main():
    parser = argparse.ArgumentParser(description="策略实时监控")
    parser.add_argument("--symbols", required=True, help="逗号分隔的标的代码")
    parser.add_argument("--freq", default="1d", help="数据频率 (1d/1w/5min/15min/30min/60min)")
    parser.add_argument("--interval", type=int, default=60, help="轮询间隔(秒)")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0003, help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0001, help="滑点")
    parser.add_argument("--strategy", default="ma", choices=["ma"], help="策略类型")
    parser.add_argument("--fast", type=int, default=5, help="MA快线周期")
    parser.add_argument("--slow", type=int, default=20, help="MA慢线周期")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="止损比例 (如0.05)")
    parser.add_argument("--trailing-stop", type=float, default=0.0, help="追踪止损比例")
    parser.add_argument("--take-profit", type=float, default=0.0, help="止盈比例")
    parser.add_argument("--max-positions", type=int, default=0, help="最大持仓数 (0=不限)")
    parser.add_argument("--data-dir", default="./data", help="数据目录")
    parser.add_argument("--source", default=None, help="数据源 (akshare/baostock/auto)")
    parser.add_argument("--no-resume", action="store_true", help="不恢复上次状态")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    config = MonitorConfig(
        symbols=symbols,
        freq=args.freq,
        poll_interval=args.interval,
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        data_dir=args.data_dir,
        data_source=args.source,
    )

    strategy = MACrossStrategy(fast_period=args.fast, slow_period=args.slow)

    risk_config = None
    if args.stop_loss > 0 or args.trailing_stop > 0 or args.take_profit > 0 or args.max_positions > 0:
        risk_config = RiskConfig(
            stop_loss_pct=args.stop_loss if args.stop_loss > 0 else None,
            trailing_stop_pct=args.trailing_stop if args.trailing_stop > 0 else None,
            take_profit_pct=args.take_profit if args.take_profit > 0 else None,
            max_positions=args.max_positions if args.max_positions > 0 else None,
        )

    engine = LiveEngine(config, strategy, risk_config=risk_config)
    engine.start(resume=not args.no_resume)


if __name__ == "__main__":
    main()
