"""持仓继承回测示例

功能：从已有持仓开始，用策略管理后续交易。
设置初始资金，在某天以指定价格买入一定数量，然后在之后的某天开始用策略进行回测。

示例：
    python scripts/position_carry_forward.py \\
        --symbol 600219.SH --cost 6.36 --quantity 12000 \\
        --entry-date 2026-03-31 --strategy-start 2026-04-01
"""

import argparse
import sys
from pathlib import Path

# 确保 src 目录在搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from data.manager import DataManager
from strategies.base import Strategy, Signal, SignalType
from strategies.engine import BacktestEngine, BacktestConfig
from strategies.risk import RiskConfig, RiskManager


class MACrossStrategy(Strategy):
    """双均线交叉策略"""

    def __init__(self, fast_period=5, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_init(self, ctx):
        self.register_indicator('ma', period=self.fast_period)
        self.register_indicator('ma', period=self.slow_period)

    def on_bar(self, ctx):
        fast_col = f'ma_{self.fast_period}'
        slow_col = f'ma_{self.slow_period}'
        ma_fast = ctx.bar.get(fast_col)
        ma_slow = ctx.bar.get(slow_col)
        if ma_fast is None or ma_slow is None or pd.isna(ma_fast) or pd.isna(ma_slow):
            return None
        if len(ctx.bars) < 2:
            return None
        prev_fast = ctx.bars[fast_col].iloc[-2]
        prev_slow = ctx.bars[slow_col].iloc[-2]
        if pd.isna(prev_fast) or pd.isna(prev_slow):
            return None
        if prev_fast <= prev_slow and ma_fast > ma_slow:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol)
        if prev_fast >= prev_slow and ma_fast < ma_slow:
            return Signal(type=SignalType.SELL, symbol=ctx.symbol)
        return None


class SARStrategy(Strategy):
    """SAR抛物线策略"""

    def on_init(self, ctx):
        self.register_indicator('sar')

    def on_bar(self, ctx):
        sar_val = ctx.bar.get('sar_value')
        sar_trend = ctx.bar.get('sar_trend')
        if sar_val is None or sar_trend is None or pd.isna(sar_val) or pd.isna(sar_trend):
            return None
        if sar_trend > 0:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol)
        else:
            return Signal(type=SignalType.SELL, symbol=ctx.symbol)


def main():
    parser = argparse.ArgumentParser(description="持仓继承回测")
    parser.add_argument("--symbol", required=True, help="股票代码，如 600219.SH")
    parser.add_argument("--cost", type=float, required=True, help="成本价")
    parser.add_argument("--quantity", type=float, required=True, help="持仓数量")
    parser.add_argument("--entry-date", required=True, help="买入日期，如 2026-03-31")
    parser.add_argument("--strategy-start", required=True, help="策略开始日期，如 2026-04-01")
    parser.add_argument("--capital", type=float, default=100000.0, help="初始资金（默认10万）")
    parser.add_argument("--strategy", default="ma", choices=["ma", "sar"], help="策略类型")
    parser.add_argument("--fast", type=int, default=5, help="快均线周期")
    parser.add_argument("--slow", type=int, default=20, help="慢均线周期")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="止损比例，如0.05表示5%%")
    parser.add_argument("--data-dir", default="./data", help="数据目录")
    parser.add_argument("--source", default="baostock", choices=["akshare", "baostock"], help="数据源")
    args = parser.parse_args()

    # 1. 获取数据（提前3个月用于指标预热）
    entry_dt = pd.Timestamp(args.entry_date)
    start_dt = entry_dt - pd.Timedelta(days=90)
    end_dt = pd.Timestamp.today()

    print(f"正在获取 {args.symbol} 数据 ({start_dt.date()} ~ {end_dt.date()})...")
    mgr = DataManager(data_dir=args.data_dir)
    kline = mgr.get_kline(args.symbol, start=start_dt.to_pydatetime(),
                          end=end_dt.to_pydatetime(), freq="1d",
                          source=args.source)
    df = kline.df
    print(f"获取到 {len(df)} 条数据，日期范围: {df.index[0].date()} ~ {df.index[-1].date()}")

    # 2. 配置回测
    initial_positions = {args.symbol: {"avg_cost": args.cost, "quantity": args.quantity}}
    config = BacktestConfig(
        initial_capital=args.capital,
        commission=0.0003,
        slippage=0.0001,
        initial_positions=initial_positions,
    )

    risk_mgr = None
    if args.stop_loss > 0:
        risk_cfg = RiskConfig(stop_loss_pct=args.stop_loss)
        risk_mgr = RiskManager(risk_cfg)

    engine = BacktestEngine(config, risk_manager=risk_mgr)

    # 3. 选择策略
    if args.strategy == "ma":
        strategy = MACrossStrategy(fast_period=args.fast, slow_period=args.slow)
        print(f"策略: 双均线交叉 (MA{args.fast}/MA{args.slow})")
    else:
        strategy = SARStrategy()
        print("策略: SAR抛物线")

    # 4. 运行回测
    print(f"初始持仓: {args.symbol} 成本={args.cost} 数量={int(args.quantity)}")
    print(f"持仓成本: {args.cost * args.quantity:.2f}")
    print(f"策略开始: {args.strategy_start}")
    print("-" * 50)

    result = engine.run(strategy, df, symbol=args.symbol,
                        strategy_start=args.strategy_start)

    # 5. 输出结果
    print(result.summary())
    print("-" * 50)

    # 持仓详情
    pos = result.equity_curve
    print(f"最终权益: {pos['equity'].iloc[-1]:.2f}")
    print(f"初始现金: {args.capital:.2f}")
    print(f"持仓成本: {args.cost * args.quantity:.2f}")

    # 交易明细
    if result.trades:
        print(f"\n交易明细 ({len(result.trades)} 笔):")
        for i, t in enumerate(result.trades, 1):
            direction = "买入" if t.quantity > 0 else "卖出"
            print(f"  {i}. {t.entry_time.date()} {direction} "
                  f"价格={t.entry_price:.2f} 数量={int(t.quantity)}")
            if t.exit_time is not None:
                pnl_str = f"PnL={t.pnl:.2f}" if t.pnl else ""
                print(f"     平仓: {t.exit_time.date()} 价格={t.exit_price:.2f} {pnl_str}")


if __name__ == "__main__":
    main()
