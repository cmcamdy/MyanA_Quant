#!/usr/bin/env python3
"""Minervini Stage 2 趋势选股独立运行脚本

基于 Minervini 趋势模板 + VCP形态 + 相对强度 的趋势选股。
使用腾讯财经API获取实时行情和历史K线数据。

用法:
    # 沪深300趋势选股 (默认，SMA200需要足够历史数据)
    python3 scripts/demo_trend_screen.py

    # 仅A股
    python3 scripts/demo_trend_screen.py --market a

    # 指定股票
    python3 scripts/demo_trend_screen.py --codes 601318,600519,600036

    # Top 5 + 评分明细
    python3 scripts/demo_trend_screen.py --top 5 --detail

    # 调整筛选参数
    python3 scripts/demo_trend_screen.py --min-score 60 --benchmark sz399006
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_selection import TrendScreener
from stock_selection.tencent_fetcher import load_codes_by_market

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("trend_screen")


def main():
    parser = argparse.ArgumentParser(description='Minervini Stage 2 趋势选股')
    parser.add_argument('--market', type=str, default='csi300',
                        choices=['all', 'a', 'hk', 'csi300'],
                        help='股票池 (默认csi300，SMA200需要大量历史数据)')
    parser.add_argument('--top', type=int, default=10, help='显示前N只 (默认10)')
    parser.add_argument('--min-score', type=int, default=50, help='最低综合分 (默认50)')
    parser.add_argument('--benchmark', type=str, default='sh000001',
                        help='基准指数 (sh000001=上证/sz399001=深证/sz399006=创业板)')
    parser.add_argument('--codes', type=str, default=None,
                        help='指定股票代码(逗号分隔, 如 601318,600519)')
    parser.add_argument('--concurrent', type=int, default=20, help='异步并发数 (默认20)')
    parser.add_argument('--min-criteria', type=int, default=7,
                        help='Minervini模板最少通过条件数 (默认7)')
    parser.add_argument('--detail', action='store_true', help='显示评分明细')
    args = parser.parse_args()

    codes = None
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
        logger.info(f"指定股票: {codes}")

    market_label = {
        "all": "A股+港股通", "a": "A股", "hk": "港股通", "csi300": "沪深300",
    }.get(args.market, args.market)

    logger.info("=" * 60)
    logger.info(f"Minervini Stage 2 趋势选股: {market_label}")
    logger.info(f"参数: top={args.top}, min_score={args.min_score}, "
                f"benchmark={args.benchmark}, min_criteria={args.min_criteria}")
    logger.info("=" * 60)

    t0 = time.time()

    screener = TrendScreener(
        top_n=args.top,
        min_score=args.min_score,
        market=args.market,
        benchmark=args.benchmark,
        max_concurrent=args.concurrent,
        min_criteria_pass=args.min_criteria,
    )
    top_df, all_df = screener.screen_all(codes=codes)

    elapsed = time.time() - t0

    # 输出结果
    print(f"\n{'='*80}")
    print(f"  Minervini 趋势选股结果 — {market_label} (耗时 {elapsed:.1f}s)")
    print(f"{'='*80}")

    if top_df.empty:
        print("  无符合条件的标的")
        return

    # 摘要
    total = len(all_df)
    pool_size = len(load_codes_by_market(args.market)) if codes is None else len(codes)
    buy_count = len(all_df[all_df['buy_signal'] == True]) if 'buy_signal' in all_df.columns else 0
    print(f"  分析: {pool_size} 只 → 过滤后: {total} 只 → 入选: {len(top_df)} 只, 买入信号: {buy_count} 只")
    print()

    # Phase 分布
    if 'phase' in all_df.columns:
        phase_counts = all_df['phase'].value_counts()
        print(f"  --- Phase 分布 ---")
        for phase in ["Uptrend", "Base", "Distribution", "Downtrend"]:
            count = phase_counts.get(phase, 0)
            bar = "█" * count
            label = {"Uptrend": "上升", "Base": "底基", "Distribution": "派发", "Downtrend": "下降"}.get(phase, phase)
            print(f"  {label:<6} ({phase:<12}): {count:>3} {bar}")
        print()

    # Top N
    print(f"  {'排名':>4} {'代码':<13} {'名称':<8} {'市场':>4} {'总分':>4} "
          f"{'Phase':>10} {'模板':>4} {'VCP':>3} {'RS':>5} {'放量':>4} "
          f"{'买入':>4} {'止损':>8} {'仓位%':>6}")
    print(f"  {'-'*100}")

    for idx, row in top_df.iterrows():
        phase = row.get('phase', '?')
        short_phase = {"Uptrend": "Up", "Base": "Base", "Distribution": "Dist", "Downtrend": "Down"}.get(phase, phase)
        vcp = "V" if row.get('vcp_detected') else "-"
        vol_bk = "Y" if row.get('volume_breakout') else "-"
        buy = "B" if row.get('buy_signal') else "-"
        mkt = row.get('market', 'A')
        stop = f"{row['atr_stop_loss']:.2f}" if row.get('atr_stop_loss', 0) > 0 else "N/A"
        pos = f"{row['position_size_pct']:.1f}" if row.get('position_size_pct', 0) > 0 else "N/A"
        rs = f"{row.get('relative_strength', 0):.1f}"
        criteria = f"{row.get('criteria_passed', 0)}/{row.get('criteria_total', 8)}"

        print(f"  {idx:>4} {row['symbol']:<13} {row['name']:<8} {mkt:>4} "
              f"{row['composite_score']:>4} {short_phase:>10} {criteria:>4} {vcp:>3} "
              f"{rs:>5} {vol_bk:>4} {buy:>4} {stop:>8} {pos:>6}")

    # 评分明细 (可选)
    if args.detail and not all_df.empty:
        print(f"\n  --- 全部候选评分 ---")
        print(f"  {'排名':>4} {'代码':<13} {'名称':<8} {'总分':>4} {'Phase':>10} "
              f"{'模板':>4} {'VCP':>3} {'RS':>5} {'SMA50':>8} {'SMA200':>8}")
        print(f"  {'-'*85}")
        for idx, row in all_df.head(30).iterrows():
            phase = row.get('phase', '?')
            short_phase = {"Uptrend": "Up", "Base": "Base", "Distribution": "Dist", "Downtrend": "Down"}.get(phase, phase)
            vcp = "V" if row.get('vcp_detected') else "-"
            criteria = f"{row.get('criteria_passed', 0)}/{row.get('criteria_total', 8)}"
            rs = f"{row.get('relative_strength', 0):.1f}"
            sma50 = f"{row['sma_50']:.2f}" if pd.notna(row.get('sma_50')) else "N/A"
            sma200 = f"{row['sma_200']:.2f}" if pd.notna(row.get('sma_200')) else "N/A"

            print(f"  {idx:>4} {row['symbol']:<13} {row['name']:<8} "
                  f"{row['composite_score']:>4} {short_phase:>10} {criteria:>4} "
                  f"{vcp:>3} {rs:>5} {sma50:>8} {sma200:>8}")

    print()


if __name__ == "__main__":
    main()
