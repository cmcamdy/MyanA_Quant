#!/usr/bin/env python3
"""A股日线数据批量下载脚本

用法:
    # 全量下载（首次运行）
    python scripts/download_daily.py

    # 增量更新（从本地最后日期续传）
    python scripts/download_daily.py --incremental

    # 只下载指定股票
    python scripts/download_daily.py --symbols 000807.SZ,600036.SH

    # 自定义数据目录
    python scripts/download_daily.py --data-dir /path/to/data

    # 忽略断点，从头开始
    python scripts/download_daily.py --no-resume

断点续传:
    下载过程中每10只股票自动保存进度到 data/.download_checkpoint.json
    中断后再次运行会自动跳过已完成的股票继续下载
    全部成功后断点文件自动清理；有失败时保留断点，下次继续
    运行参数（日期、复权方式等）变化时断点自动失效

作为定时任务:
    # crontab -e
    # 每个交易日 18:00 增量更新
    0 18 * * 1-5 cd /path/to/lianghua && python scripts/download_daily.py --incremental >> logs/download.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目 src 到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import baostock as bs
import pandas as pd

from data.base import KlineData
from data.storage import ParquetStorage

# ─── 日志配置 ───

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("download_daily")


# ─── 辅助函数 ───

def get_all_a_stocks() -> list:
    """从Baostock获取全部A股代码，返回 [(baostock_code, display_code), ...]"""
    rs = bs.query_stock_basic()
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)
    # 只保留A股
    df = df[df['type'] == '1']

    result = []
    for code in df['code'].tolist():
        # baostock格式 sh.600000 → 显示格式 600000.SH
        market, num = code.split('.')
        display = f"{num}.{market.upper()}"
        result.append((code, display))
    return result


def download_one(
    bs_code: str,
    display_code: str,
    storage: ParquetStorage,
    start_date: str,
    end_date: str,
    adjust: str = '2',
) -> dict:
    """下载单只股票日线，返回结果统计"""
    result = {'code': display_code, 'status': 'ok', 'rows': 0, 'msg': ''}

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag=adjust,
        )

        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())

        if not data:
            result['status'] = 'empty'
            result['msg'] = rs.error_msg or '无数据'
            return result

        df = pd.DataFrame(data, columns=rs.fields)
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime').sort_index()
        keep = [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount'] if c in df.columns]
        df = df[keep]

        # 过滤掉全为NaN的行
        df = df.dropna(subset=['close'])

        if df.empty:
            result['status'] = 'empty'
            return result

        kline = KlineData(
            symbol=display_code, freq='1d',
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d'),
            df=df,
        )
        storage.save(kline)
        result['rows'] = len(df)

    except Exception as e:
        result['status'] = 'error'
        result['msg'] = str(e)

    return result


def get_incremental_start(storage: ParquetStorage, display_code: str) -> str:
    """获取增量更新的起始日期（本地最后日期的下一天）"""
    dr = storage.get_date_range(display_code, '1d')
    if dr is None:
        return '2000-01-01'
    next_day = dr[1] + pd.Timedelta(days=1)
    # 如果本地数据已是最近，跳过
    if next_day.date() >= datetime.now().date():
        return None
    return next_day.strftime('%Y-%m-%d')


# ─── 断点续传 ───

CHECKPOINT_FILENAME = ".download_checkpoint.json"


def checkpoint_path(data_dir: str) -> Path:
    return Path(data_dir) / CHECKPOINT_FILENAME


def save_checkpoint(data_dir: str, done_codes: set, run_params: dict):
    """保存断点：已完成股票集合 + 运行参数"""
    cp = checkpoint_path(data_dir)
    cp.write_text(json.dumps({
        'params': run_params,
        'done': sorted(done_codes),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }, ensure_ascii=False))


def load_checkpoint(data_dir: str, run_params: dict) -> Optional[set]:
    """加载断点，返回已完成股票集合；参数不匹配或文件不存在则返回 None"""
    cp = checkpoint_path(data_dir)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text())
        if data.get('params') != run_params:
            logger.info("断点文件参数与当前运行不一致，忽略断点，从头开始")
            return None
        done = set(data.get('done', []))
        logger.info(f"发现断点文件，已完成 {len(done)} 只股票，将从断点继续")
        return done
    except Exception as e:
        logger.warning(f"读取断点文件失败: {e}，从头开始")
        return None


def remove_checkpoint(data_dir: str):
    cp = checkpoint_path(data_dir)
    if cp.exists():
        cp.unlink()


# ─── 主流程 ───

def main():
    parser = argparse.ArgumentParser(description='A股日线数据批量下载')
    parser.add_argument('--incremental', action='store_true', help='增量更新（只下载本地缺失的日期段）')
    parser.add_argument('--symbols', type=str, default=None, help='指定股票代码，逗号分隔 (如 000807.SZ,600036.SH)')
    parser.add_argument('--data-dir', type=str, default=None, help='数据存储目录')
    parser.add_argument('--start', type=str, default='2000-01-01', help='全量下载起始日期 (默认 2000-01-01)')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (默认今天)')
    parser.add_argument('--adjust', type=str, default='qfq', choices=['qfq', 'hfq', ''], help='复权方式')
    parser.add_argument('--retry', type=int, default=3, help='单只股票下载失败重试次数')
    parser.add_argument('--delay', type=float, default=0.3, help='每只股票下载间隔秒数（避免请求过快）')
    parser.add_argument('--no-resume', action='store_true', help='忽略断点文件，从头开始下载')
    args = parser.parse_args()

    # 数据目录
    data_dir = args.data_dir or str(PROJECT_ROOT / "data")
    storage = ParquetStorage(data_dir)

    # 日期
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')
    adjust_map = {'qfq': '2', 'hfq': '1', '': '3'}
    adjust_flag = adjust_map[args.adjust]

    # 登录 Baostock
    logger.info("登录 Baostock ...")
    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"登录失败: {lg.error_msg}")
        sys.exit(1)

    try:
        # 获取股票列表
        if args.symbols:
            stock_list = []
            for sym in args.symbols.split(','):
                sym = sym.strip()
                if sym.startswith('6'):
                    bs_code = f"sh.{sym.split('.')[0]}"
                else:
                    bs_code = f"sz.{sym.split('.')[0]}"
                stock_list.append((bs_code, sym))
        else:
            logger.info("获取A股列表 ...")
            stock_list = get_all_a_stocks()

        total = len(stock_list)
        logger.info(f"共 {total} 只股票，模式: {'增量更新' if args.incremental else '全量下载'}")
        logger.info(f"日期范围: {args.start if not args.incremental else '按股票增量'} ~ {end_date}, 复权: {args.adjust}, 重试: {args.retry}, 间隔: {args.delay}s")
        logger.info(f"数据目录: {data_dir}")

        # 断点续传
        run_params = {
            'incremental': args.incremental,
            'start': args.start,
            'end': end_date,
            'adjust': args.adjust,
            'symbols': args.symbols,
        }
        done_codes = set()
        if not args.no_resume:
            loaded = load_checkpoint(data_dir, run_params)
            if loaded is not None:
                done_codes = loaded

        # 统计
        success = 0
        skipped = 0
        empty = 0
        failed = 0
        total_rows = 0

        start_time = time.time()
        checkpoint_counter = 0

        for i, (bs_code, display_code) in enumerate(stock_list, 1):
            # 断点续传：跳过已完成的股票
            if display_code in done_codes:
                continue

            # 增量模式：确定起始日期
            if args.incremental:
                inc_start = get_incremental_start(storage, display_code)
                if inc_start is None:
                    skipped += 1
                    if skipped <= 5 or skipped % 500 == 0:
                        logger.debug(f"[{i}/{total}] {display_code} 已是最新，跳过 (累计跳过 {skipped})")
                    continue
                dl_start = inc_start
            else:
                dl_start = args.start

            logger.info(f"[{i}/{total}] 开始下载 {display_code} ({dl_start} ~ {end_date})")

            # 重试逻辑
            for attempt in range(1, args.retry + 1):
                r = download_one(bs_code, display_code, storage, dl_start, end_date, adjust_flag)

                if r['status'] == 'ok':
                    success += 1
                    total_rows += r['rows']
                    done_codes.add(display_code)
                    # 每下载成功一只就保存断点
                    checkpoint_counter += 1
                    if checkpoint_counter % 10 == 0:
                        save_checkpoint(data_dir, done_codes, run_params)
                    # 每50只或首只或最后一只打印进度
                    if i <= 3 or i % 50 == 0 or i == total:
                        elapsed = time.time() - start_time
                        remaining = total - len(done_codes)
                        speed = len(done_codes) / elapsed if elapsed > 0 else 0
                        eta = remaining / speed if speed > 0 else 0
                        logger.info(f"[{len(done_codes)}/{total}] {display_code} +{r['rows']}行 | 成功 {success}, 跳过 {skipped}, 空 {empty}, 失败 {failed} | {elapsed:.0f}s, {speed:.1f}只/s, ETA {eta/60:.0f}min")
                    break

                elif r['status'] == 'empty':
                    empty += 1
                    done_codes.add(display_code)
                    if empty <= 3 or i % 200 == 0:
                        logger.info(f"[{len(done_codes)}/{total}] {display_code} 无数据: {r['msg']} (累计空 {empty})")
                    break

                else:  # error
                    if attempt < args.retry:
                        logger.warning(f"[{i}/{total}] {display_code} 第{attempt}次失败: {r['msg']}, 重试...")
                        time.sleep(1)
                    else:
                        failed += 1
                        logger.error(f"[{i}/{total}] {display_code} {args.retry}次重试均失败: {r['msg']}")

            time.sleep(args.delay)

        # 汇总
        elapsed = time.time() - start_time
        save_checkpoint(data_dir, done_codes, run_params)
        logger.info("=" * 50)
        logger.info(f"下载完成: 成功 {success}, 跳过(已是最新) {skipped}, 无数据 {empty}, 失败 {failed}")
        logger.info(f"总行数: {total_rows}, 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        logger.info(f"数据目录: {data_dir}")

        # 全部完成后删除断点文件
        if failed == 0:
            remove_checkpoint(data_dir)
            logger.info("断点文件已清理")
        else:
            logger.info(f"存在 {failed} 只失败，断点文件保留，再次运行将从断点继续（加 --no-resume 可忽略断点）")

    finally:
        bs.logout()
        logger.info("已登出 Baostock")


if __name__ == '__main__':
    main()
