#!/usr/bin/env python3
"""A股行业-股票映射下载脚本

用法:
    python3 scripts/download_industry_stock.py

    # 自定义数据目录
    python3 scripts/download_industry_stock.py --data-dir /path/to/data

输出:
    data/industry-stock/industry_stock.csv     全量映射表 (industry,code,name)
    data/industry-stock/industry_summary.txt   人类可读的行业汇总
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import baostock as bs
import pandas as pd

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("download_industry_stock")


def convert_code(bs_code: str) -> str:
    """baostock格式 sh.600000 → 显示格式 600000.SH"""
    market, num = bs_code.split(".")
    return f"{num}.{market.upper()}"


def download_industry_stock() -> pd.DataFrame:
    """从Baostock获取全量行业分类，返回 DataFrame[code, name, industry]"""
    rs = bs.query_stock_industry()
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())

    if not data:
        logger.error(f"获取行业数据失败: {rs.error_msg}")
        sys.exit(1)

    df = pd.DataFrame(data, columns=rs.fields)
    logger.info(f"原始记录数: {len(df)}")

    # 代码格式转换
    df['code'] = df['code'].apply(convert_code)
    df = df.rename(columns={'code_name': 'name'})

    # 过滤行业为空的记录
    before = len(df)
    df = df[df['industry'].str.strip() != '']
    logger.info(f"过滤行业为空: {before} → {len(df)} (去除 {before - len(df)})")

    # 去重：同一股票取最新记录
    df = df.drop_duplicates(subset=['code'], keep='last')

    df = df[['industry', 'code', 'name']].sort_values(['industry', 'code']).reset_index(drop=True)
    return df


def write_csv(df: pd.DataFrame, output_dir: Path):
    """写入 CSV"""
    path = output_dir / "industry_stock.csv"
    df.to_csv(path, index=False, encoding='utf-8')
    logger.info(f"已写入 {path} ({len(df)} 行)")


def write_summary(df: pd.DataFrame, output_dir: Path):
    """写入人类可读的行业汇总"""
    path = output_dir / "industry_summary.txt"
    lines = []
    for industry, group in df.groupby('industry', sort=True):
        lines.append(f"{industry} ({len(group)}只)")
        for _, row in group.iterrows():
            lines.append(f"  {row['code']} {row['name']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding='utf-8')
    industry_count = df['industry'].nunique()
    logger.info(f"已写入 {path} ({industry_count} 个行业)")


def main():
    parser = argparse.ArgumentParser(description='A股行业-股票映射下载')
    parser.add_argument('--data-dir', type=str, default=None, help='数据存储目录')
    args = parser.parse_args()

    data_dir = Path(args.data_dir or str(PROJECT_ROOT / "data"))
    output_dir = data_dir / "industry-stock"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("登录 Baostock ...")
    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"登录失败: {lg.error_msg}")
        sys.exit(1)

    try:
        df = download_industry_stock()
        write_csv(df, output_dir)
        write_summary(df, output_dir)
        logger.info(f"完成: {df['industry'].nunique()} 个行业, {len(df)} 只股票")
    finally:
        bs.logout()
        logger.info("已登出 Baostock")


if __name__ == '__main__':
    main()
