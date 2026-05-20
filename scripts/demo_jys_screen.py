#!/usr/bin/env python3
"""JYS五维选股独立运行脚本

支持 A股全量 + 港股通 选股分析。

用法:
    # A股+港股通全量分析 (默认)
    python3 scripts/demo_jys_screen.py

    # 仅A股
    python3 scripts/demo_jys_screen.py --market a

    # 仅港股通
    python3 scripts/demo_jys_screen.py --market hk

    # 仅沪深300
    python3 scripts/demo_jys_screen.py --market csi300

    # 指定股票
    python3 scripts/demo_jys_screen.py --codes 601318,600519,00700,09988

    # Top 5 + 评分明细
    python3 scripts/demo_jys_screen.py --top 5 --detail

    # 调整筛选参数
    python3 scripts/demo_jys_screen.py --min-score 50 --max-pe 25
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_selection import JYSScreener, JYSScorer, TencentFetcher
from stock_selection.tencent_fetcher import load_codes_by_market

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("jys_screen")


def main():
    parser = argparse.ArgumentParser(description='JYS五维选股: A股+港股通基本面评分筛选')
    parser.add_argument('--market', type=str, default='all',
                        choices=['all', 'a', 'hk', 'csi300'],
                        help='股票池: all(A股+港股通) / a(仅A股) / hk(仅港股通) / csi300(沪深300)')
    parser.add_argument('--top', type=int, default=10, help='显示前N只 (默认10)')
    parser.add_argument('--min-score', type=int, default=40, help='最低综合分 (默认40)')
    parser.add_argument('--max-pe', type=float, default=30, help='最大PE (默认30)')
    parser.add_argument('--min-turnover', type=float, default=0.3, help='最低换手率%% (默认0.3)')
    parser.add_argument('--codes', type=str, default=None, help='指定股票代码(逗号分隔, 如 601318,00700)')
    parser.add_argument('--concurrent', type=int, default=20, help='异步并发数 (默认20)')
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
    logger.info(f"JYS五维选股: {market_label}基本面评分")
    logger.info(f"参数: top={args.top}, min_score={args.min_score}, max_pe={args.max_pe}, "
                f"min_turnover={args.min_turnover}, concurrent={args.concurrent}")
    logger.info("=" * 60)

    t0 = time.time()

    screener = JYSScreener(
        top_n=args.top,
        min_score=args.min_score,
        max_pe=args.max_pe,
        min_turnover=args.min_turnover,
        max_concurrent=args.concurrent,
        market=args.market,
    )
    top_df, all_df = screener.screen_all(codes=codes)

    elapsed = time.time() - t0

    # 输出结果
    print(f"\n{'='*75}")
    print(f"  JYS五维选股结果 — {market_label} (耗时 {elapsed:.1f}s)")
    print(f"{'='*75}")

    if top_df.empty:
        print("  无符合条件的标的")
        return

    # 摘要
    total = len(all_df)
    a_count = len(all_df[all_df['market'] == 'A']) if 'market' in all_df.columns else total
    hk_count = len(all_df[all_df['market'] == 'HK']) if 'market' in all_df.columns else 0
    pool_size = len(load_codes_by_market(args.market)) if codes is None else len(codes)
    print(f"  分析: {pool_size} 只 → 过滤后: {total} 只", end="")
    if hk_count > 0:
        print(f" (A股:{a_count} 港股:{hk_count})", end="")
    print(f" → 入选: {len(top_df)} 只")
    print()

    # 打印 Top N
    print(f"  {'排名':>4} {'代码':<13} {'名称':<8} {'市场':>4} {'总分':>4} {'评级':>3} "
          f"{'技术':>4} {'估值':>4} {'盈利':>4} {'安全':>4} {'分红':>4} "
          f"{'PE':>7} {'PB':>6} {'ROE':>6} {'股息率':>6}")
    print(f"  {'-'*95}")

    for idx, row in top_df.iterrows():
        pe_str = f"{row['pe_ratio']:.1f}" if pd.notna(row.get('pe_ratio')) else "N/A"
        pb_str = f"{row['pb_ratio']:.2f}" if pd.notna(row.get('pb_ratio')) else "N/A"
        roe_str = f"{row['roe']:.1f}" if pd.notna(row.get('roe')) else "N/A"
        div_str = f"{row['dividend_yield']:.2f}" if pd.notna(row.get('dividend_yield')) else "0.00"
        mkt = row.get('market', 'A')
        print(f"  {idx:>4} {row['symbol']:<13} {row['name']:<8} {mkt:>4} "
              f"{row['composite_score']:>4} {row['grade']:>3} "
              f"{row['tech_score']:>4} {row['valuation_score']:>4} {row['profit_score']:>4} "
              f"{row['safety_score']:>4} {row['dividend_score']:>4} "
              f"{pe_str:>7} {pb_str:>6} {roe_str:>6} {div_str:>6}")

    # 评分明细 (可选)
    if args.detail and not all_df.empty:
        print(f"\n  --- 全部候选评分 ---")
        print(f"  {'排名':>4} {'代码':<13} {'名称':<8} {'市场':>4} {'总分':>4} {'评级':>3} "
              f"{'PE':>7} {'PB':>6} {'动量20d':>8} {'换手率':>6}")
        print(f"  {'-'*75}")
        for idx, row in all_df.head(30).iterrows():
            pe_str = f"{row['pe_ratio']:.1f}" if pd.notna(row.get('pe_ratio')) else "N/A"
            pb_str = f"{row['pb_ratio']:.2f}" if pd.notna(row.get('pb_ratio')) else "N/A"
            mom_str = f"{row['momentum_20d']:.2f}%" if pd.notna(row.get('momentum_20d')) else "N/A"
            to_str = f"{row['turnover_rate']:.2f}%" if pd.notna(row.get('turnover_rate')) else "N/A"
            mkt = row.get('market', 'A')
            print(f"  {idx:>4} {row['symbol']:<13} {row['name']:<8} {mkt:>4} "
                  f"{row['composite_score']:>4} {row['grade']:>3} "
                  f"{pe_str:>7} {pb_str:>6} {mom_str:>8} {to_str:>6}")

    # 市场分布
    if not all_df.empty and 'market' in all_df.columns:
        market_counts = all_df['market'].value_counts()
        print(f"\n  --- 市场分布 ---")
        for mkt, count in market_counts.items():
            label = "A股" if mkt == "A" else "港股通"
            bar = "█" * count
            print(f"  {label:<6}: {count:>3} {bar}")

    # 评级分布
    if not all_df.empty and 'grade' in all_df.columns:
        print(f"\n  --- 评级分布 ---")
        grade_counts = all_df['grade'].value_counts().sort_index()
        for grade, count in grade_counts.items():
            bar = "█" * count
            print(f"  {grade:>3}: {count:>3} {bar}")

    print()


if __name__ == "__main__":
    main()
