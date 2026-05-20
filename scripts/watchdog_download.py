#!/usr/bin/env python3
"""下载守护脚本：监控 download_daily.py 运行状态，卡住自动重启续传

用法:
    python3 scripts/watchdog_download.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "download.log"
SCRIPT = PROJECT_ROOT / "scripts" / "download_daily.py"
CHECK_INTERVAL = 60       # 检查间隔（秒）
STUCK_THRESHOLD = 600     # 判定卡住的阈值（秒），10分钟无日志更新
MAX_RESTARTS = 20         # 最大重启次数，防止无限循环


def get_log_mtime() -> float:
    """返回日志文件最后修改时间，不存在则返回0"""
    if not LOG_FILE.exists():
        return 0.0
    return LOG_FILE.stat().st_mtime


def is_download_done() -> bool:
    """检查日志中是否出现下载完成的标志"""
    if not LOG_FILE.exists():
        return False
    text = LOG_FILE.read_text(errors="ignore")
    return "下载完成:" in text


def find_pid() -> int:
    """查找正在运行的 download_daily.py 进程PID"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "download_daily.py"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        # 排除自身
        my_pid = os.getpid()
        for p in pids:
            if int(p) != my_pid:
                return int(p)
    except Exception:
        pass
    return None


def kill_process(pid: int):
    """先 SIGTERM，等3秒不行再 SIGKILL"""
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.3)
            try:
                os.kill(pid, 0)
            except OSError:
                return
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def start_download() -> int:
    """启动下载脚本，返回PID"""
    log_f = open(LOG_FILE, "a")
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT)],
        stdout=log_f,
        stderr=log_f,
    )
    print(f"[{time.strftime('%H:%M:%S')}] 启动下载进程 PID={proc.pid}")
    return proc.pid


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 守护启动，每 {CHECK_INTERVAL}s 检查，{STUCK_THRESHOLD}s 无更新视为卡住")
    print(f"  日志: {LOG_FILE}")
    print(f"  脚本: {SCRIPT}")

    restarts = 0
    pid = find_pid()
    if pid:
        print(f"[{time.strftime('%H:%M:%S')}] 发现已运行的下载进程 PID={pid}")
    else:
        pid = start_download()
        restarts += 1

    last_mtime = get_log_mtime()

    while True:
        time.sleep(CHECK_INTERVAL)

        # 检查是否下载完成
        if is_download_done():
            print(f"[{time.strftime('%H:%M:%S')}] 下载已完成，守护退出")
            break

        # 检查进程是否还活着
        current_pid = find_pid()
        if current_pid is None:
            if restarts >= MAX_RESTARTS:
                print(f"[{time.strftime('%H:%M:%S')}] 已重启 {MAX_RESTARTS} 次，停止守护")
                break
            print(f"[{time.strftime('%H:%M:%S')}] 下载进程已退出，重启...")
            pid = start_download()
            restarts += 1
            last_mtime = get_log_mtime()
            continue

        # 检查日志是否更新
        now = time.time()
        current_mtime = get_log_mtime()

        if current_mtime > last_mtime:
            # 日志有更新，正常
            last_mtime = current_mtime
        elif now - last_mtime > STUCK_THRESHOLD:
            # 日志超过阈值没更新，判定卡住
            if restarts >= MAX_RESTARTS:
                print(f"[{time.strftime('%H:%M:%S')}] 已重启 {MAX_RESTARTS} 次，停止守护")
                break
            print(f"[{time.strftime('%H:%M:%S')}] 日志 {STUCK_THRESHOLD}s 无更新，判定卡住，杀掉 PID={current_pid} 并重启")
            kill_process(current_pid)
            time.sleep(2)
            pid = start_download()
            restarts += 1
            last_mtime = get_log_mtime()


if __name__ == "__main__":
    main()
