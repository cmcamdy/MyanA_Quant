"""腾讯财经API数据获取（异步+同步双模式）

支持市场: A股 (sh/sz) + 港股通 (hk)
数据源: 腾讯财经实时行情 + 历史K线
参考: https://github.com/stevenwxz/JYSstock_analyzer
"""

import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_REALTIME_URL = "https://qt.gtimg.cn/q={symbol}"
_HISTORY_URL = (
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    "?param={symbol},day,,,{days},qfq&_var=kline_dayqfq"
)
_INDEX_URL = "https://qt.gtimg.cn/q=sh000001,sz399001,sz399006"

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/17.2",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0",
]

_HEADERS = {"Referer": "https://gu.qq.com/"}

# 股票池缓存
_CSI300_CODES: Optional[List[str]] = None
_A_SHARE_CODES: Optional[List[str]] = None
_HK_CONNECT_CODES: Optional[List[str]] = None

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# 代码格式转换
# ---------------------------------------------------------------------------

def _code_to_symbol(code: str) -> str:
    """代码转腾讯格式: sh6xxxxx / sz0xxxxx / hk00700"""
    if code.startswith(("sh", "sz", "hk")):
        return code
    c = code.replace(".SH", "").replace(".SZ", "").replace(".HK", "")
    # 港股: 5位数字以0开头
    if len(c) == 5 and c.startswith("0"):
        return f"hk{c}"
    # A股
    if c.startswith(("6", "9")):
        return f"sh{c}"
    return f"sz{c}"


def _symbol_to_code(symbol: str) -> str:
    """腾讯格式转原始代码"""
    if symbol.startswith("hk"):
        return symbol.replace("hk", "")
    return symbol.replace("sh", "").replace("sz", "")


def _format_code(symbol: str) -> str:
    """腾讯格式或原始代码转标准格式: 601318.SH / 00700.HK"""
    if symbol.startswith("hk"):
        code = symbol.replace("hk", "")
        return f"{code}.HK"
    if symbol.startswith("sh"):
        code = symbol.replace("sh", "")
        return f"{code}.SH"
    if symbol.startswith("sz"):
        code = symbol.replace("sz", "")
        return f"{code}.SZ"
    # 纯数字代码
    code = symbol.replace(".SH", "").replace(".SZ", "").replace(".HK", "")
    if len(code) == 5 and code.startswith("0"):
        return f"{code}.HK"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _is_hk(symbol: str) -> bool:
    """判断是否为港股"""
    return symbol.startswith("hk") or ".HK" in symbol or (len(symbol) == 5 and symbol.startswith("0"))


# ---------------------------------------------------------------------------
# 股票池加载
# ---------------------------------------------------------------------------

def _load_cached_json(cache_path: Path, min_count: int = 100) -> Optional[List[str]]:
    """尝试从本地JSON缓存加载"""
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                codes = json.load(f)
            if isinstance(codes, list) and len(codes) >= min_count:
                return codes
        except Exception:
            pass
    return None


def _save_cache(cache_path: Path, codes: List[str]) -> None:
    """保存到本地JSON缓存"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(codes, f)


def load_csi300_codes() -> List[str]:
    """加载沪深300成分股代码列表 (约300只)

    优先级: 本地JSON缓存 > akshare在线获取
    """
    global _CSI300_CODES
    if _CSI300_CODES is not None:
        return _CSI300_CODES

    cached = _load_cached_json(_DATA_DIR / "csi300_codes.json", min_count=100)
    if cached:
        _CSI300_CODES = cached
        return cached

    try:
        import akshare as ak
        df = ak.index_stock_cons_csindex(symbol="000300")
        for col in ["品种代码", "成分券代码", "code"]:
            if col in df.columns:
                codes = [str(c) for c in df[col].tolist()]
                break
        else:
            codes = [str(c) for c in df.iloc[:, 2].tolist()] if len(df.columns) > 2 else []
        if not codes or len(codes) < 100:
            raise ValueError(f"获取的成分股数量异常: {len(codes)}")
        _CSI300_CODES = codes
        _save_cache(_DATA_DIR / "csi300_codes.json", codes)
        logger.info(f"从akshare获取沪深300成分股 {len(codes)} 只，已缓存")
        return codes
    except Exception as e:
        logger.warning(f"akshare获取沪深300成分股失败: {e}")
        return []


def load_a_share_codes() -> List[str]:
    """加载A股全量股票代码列表 (约5000只)

    优先级: 本地JSON缓存 > akshare在线获取
    """
    global _A_SHARE_CODES
    if _A_SHARE_CODES is not None:
        return _A_SHARE_CODES

    cached = _load_cached_json(_DATA_DIR / "a_share_codes.json", min_count=1000)
    if cached:
        _A_SHARE_CODES = cached
        return cached

    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        for col in ["代码", "code"]:
            if col in df.columns:
                codes = [str(c).zfill(6) for c in df[col].tolist()]
                break
        else:
            codes = [str(c).zfill(6) for c in df.iloc[:, 1].tolist()]
        if not codes or len(codes) < 1000:
            raise ValueError(f"获取的A股数量异常: {len(codes)}")
        _A_SHARE_CODES = codes
        _save_cache(_DATA_DIR / "a_share_codes.json", codes)
        logger.info(f"从akshare获取A股全量 {len(codes)} 只，已缓存")
        return codes
    except Exception as e:
        logger.warning(f"akshare获取A股全量失败: {e}")
        # fallback到沪深300
        return load_csi300_codes()


def load_hk_connect_codes() -> List[str]:
    """加载港股通成分股代码列表 (约800只)

    优先级: 本地JSON缓存 > akshare在线获取
    """
    global _HK_CONNECT_CODES
    if _HK_CONNECT_CODES is not None:
        return _HK_CONNECT_CODES

    cached = _load_cached_json(_DATA_DIR / "hk_connect_codes.json", min_count=100)
    if cached:
        _HK_CONNECT_CODES = cached
        return cached

    try:
        import akshare as ak
        df = ak.stock_hk_ggt_components_em()
        # 港股代码列名可能不同
        for col in ["代码", "股票代码", "code"]:
            if col in df.columns:
                codes = [str(c).zfill(5) for c in df[col].tolist()]
                break
        else:
            codes = [str(c).zfill(5) for c in df.iloc[:, 0].tolist()]
        if not codes or len(codes) < 50:
            raise ValueError(f"获取的港股通数量异常: {len(codes)}")
        _HK_CONNECT_CODES = codes
        _save_cache(_DATA_DIR / "hk_connect_codes.json", codes)
        logger.info(f"从akshare获取港股通 {len(codes)} 只，已缓存")
        return codes
    except Exception as e:
        logger.warning(f"akshare获取港股通失败: {e}")
        return []


def load_all_codes() -> List[str]:
    """加载 A股全量 + 港股通 代码列表 (约5800只)"""
    a_codes = load_a_share_codes()
    hk_codes = load_hk_connect_codes()
    return a_codes + hk_codes


def load_codes_by_market(market: str) -> List[str]:
    """按市场加载股票代码

    Args:
        market: "all" / "a" / "hk" / "csi300"
    """
    if market == "csi300":
        return load_csi300_codes()
    if market == "a":
        return load_a_share_codes()
    if market == "hk":
        return load_hk_connect_codes()
    # "all"
    return load_all_codes()


# ---------------------------------------------------------------------------
# 解析函数 — A股
# ---------------------------------------------------------------------------

def _parse_realtime_a(raw: str, symbol: str) -> Optional[Dict]:
    """解析A股实时行情

    关键字段 (2025年实测):
        [1] 名称, [3] 价格, [4] 前收盘, [32] 涨跌幅
        [38] 换手率, [39] PE, [44] 总市值(亿), [46] PB
        [64] 股息率%(近似，部分股票需override修正)
        [65] ROE%(财报值，比PB/PE推导更准确)
    """
    try:
        match = re.search(r'="(.+)"', raw)
        if not match:
            return None
        fields = match.group(1).split("~")
        if len(fields) < 60:
            return None

        code = _symbol_to_code(symbol)
        name = fields[1] if len(fields) > 1 else ""
        price = float(fields[3]) if fields[3] else 0
        prev_close = float(fields[4]) if fields[4] else 0

        # PE
        pe = None
        for idx in [39, 43]:
            try:
                val = float(fields[idx]) if len(fields) > idx and fields[idx] else 0
                if 0 < val < 1000:
                    pe = val
                    break
            except (ValueError, IndexError):
                continue

        # PB: 字段[46]
        pb = None
        try:
            val = float(fields[46]) if len(fields) > 46 and fields[46] else 0
            if val > 0:
                pb = val
        except (ValueError, IndexError):
            pass

        # 换手率: 字段[38]
        turnover_rate = 0.0
        try:
            turnover_rate = float(fields[38]) if len(fields) > 38 and fields[38] else 0
        except (ValueError, IndexError):
            pass

        # 涨跌幅: 字段[32]
        change_pct = 0.0
        try:
            change_pct = float(fields[32]) if fields[32] else 0
        except (ValueError, IndexError):
            pass

        # 总市值(亿): 字段[44]
        market_cap = 0.0
        try:
            market_cap = float(fields[44]) if len(fields) > 44 and fields[44] else 0
        except (ValueError, IndexError):
            pass

        # 成交额: 字段[57]
        turnover_amount = 0.0
        try:
            turnover_amount = float(fields[57]) if len(fields) > 57 and fields[57] else 0
        except (ValueError, IndexError):
            pass

        # 成交量: 字段[6]
        volume = 0.0
        try:
            volume = float(fields[6]) if fields[6] else 0
        except (ValueError, IndexError):
            pass

        # ROE: 字段[65]为财报ROE（比PB/PE推导更准确），fallback到推导
        roe = None
        try:
            val = float(fields[65]) if len(fields) > 65 and fields[65] else 0
            if val > 0:
                roe = val
        except (ValueError, IndexError):
            pass
        if roe is None and pb and pe and pe > 0:
            roe = (pb / pe) * 100

        # 股息率%: 字段[64]近似值（部分股票不够准确，由override表修正）
        dividend_yield = 0.0
        try:
            val = float(fields[64]) if len(fields) > 64 and fields[64] else 0
            if 0 < val < 15:  # 合理性检查：A股股息率通常<15%
                dividend_yield = val
        except (ValueError, IndexError):
            pass

        # 每股股息: 通过股息率和价格反算
        dividend_per_share = 0.0
        if dividend_yield > 0 and price > 0:
            dividend_per_share = round(dividend_yield / 100 * price, 4)

        return {
            "code": code,
            "name": name,
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "pe_ratio": pe,
            "pb_ratio": pb,
            "turnover_rate": turnover_rate,
            "market_cap": market_cap,
            "total_shares": market_cap * 1e8 / price / 1e4 if market_cap > 0 and price > 0 else 0,
            "dividend_per_share": dividend_per_share,
            "dividend_yield": dividend_yield,
            "roe": roe,
            "profit_growth": None,  # A股无此API字段
            "turnover": turnover_amount,
            "volume": volume,
            "market": "A",
            "currency": "CNY",
        }
    except Exception as e:
        logger.debug(f"解析A股实时行情失败 {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# 解析函数 — 港股
# ---------------------------------------------------------------------------

def _parse_realtime_hk(raw: str, symbol: str) -> Optional[Dict]:
    """解析港股实时行情

    关键字段 (2025年实测，与A股有显著差异):
        [1] 名称, [3] 价格, [4] 前收盘, [6] 成交量(股)
        [32] 涨跌幅, [39] 动态PE, [44] 总市值(亿港元)
        [47] 股息率%, [51] 净利润增长%, [57] 静态PE
        [58] PB, [72] 每股股息(港元)
    """
    try:
        match = re.search(r'="(.+)"', raw)
        if not match:
            return None
        fields = match.group(1).split("~")
        if len(fields) < 60:
            return None

        code = _symbol_to_code(symbol)
        name = fields[1] if len(fields) > 1 else ""
        price = float(fields[3]) if fields[3] else 0
        prev_close = float(fields[4]) if fields[4] else 0

        # PE: 字段[39]动态 / [57]静态
        pe = None
        for idx in [39, 57]:
            try:
                val = float(fields[idx]) if len(fields) > idx and fields[idx] else 0
                if 0 < val < 1000:
                    pe = val
                    break
            except (ValueError, IndexError):
                continue

        # PB: 字段[58] (A股是[46]，港股移位)
        pb = None
        try:
            val = float(fields[58]) if len(fields) > 58 and fields[58] else 0
            if val > 0:
                pb = val
        except (ValueError, IndexError):
            pass

        # 股息率%: 字段[47] (港股API可靠！)
        dividend_yield = 0.0
        try:
            dividend_yield = float(fields[47]) if len(fields) > 47 and fields[47] else 0
            if dividend_yield < 0 or dividend_yield > 20:
                dividend_yield = 0.0
        except (ValueError, IndexError):
            pass

        # 每股股息: 字段[72]
        dividend_per_share = 0.0
        try:
            dividend_per_share = float(fields[72]) if len(fields) > 72 and fields[72] else 0
        except (ValueError, IndexError):
            pass

        # 净利润增长%: 字段[51] (港股独有)
        profit_growth = None
        try:
            val = float(fields[51]) if len(fields) > 51 and fields[51] else 0
            if val != 0:
                profit_growth = val
        except (ValueError, IndexError):
            pass

        # 涨跌幅: 字段[32]
        change_pct = 0.0
        try:
            change_pct = float(fields[32]) if fields[32] else 0
        except (ValueError, IndexError):
            pass

        # 总市值(亿港元): 字段[44]
        market_cap = 0.0
        try:
            market_cap = float(fields[44]) if len(fields) > 44 and fields[44] else 0
        except (ValueError, IndexError):
            pass

        # 成交量(股): 字段[6]
        volume = 0.0
        try:
            volume = float(fields[6]) if fields[6] else 0
        except (ValueError, IndexError):
            pass

        # 成交额: 字段[37] (港元)
        turnover_amount = 0.0
        try:
            turnover_amount = float(fields[37]) if len(fields) > 37 and fields[37] else 0
        except (ValueError, IndexError):
            pass

        # 换手率: 港股API字段[38]始终为0，需手动计算
        turnover_rate = 0.0
        if market_cap > 0 and price > 0 and volume > 0:
            total_shares_calc = market_cap * 1e8 / price  # 总股本(股)
            if total_shares_calc > 0:
                turnover_rate = (volume / total_shares_calc) * 100
                if turnover_rate > 100:  # 合理性检查
                    turnover_rate = 0.0

        # ROE推导 (同A股逻辑)
        roe = None
        if pb and pe and pe > 0:
            roe = (pb / pe) * 100

        return {
            "code": code,
            "name": name,
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "pe_ratio": pe,
            "pb_ratio": pb,
            "turnover_rate": round(turnover_rate, 2),
            "market_cap": market_cap,
            "total_shares": market_cap * 1e8 / price / 1e4 if market_cap > 0 and price > 0 else 0,
            "dividend_per_share": dividend_per_share,
            "dividend_yield": dividend_yield,
            "roe": roe,
            "profit_growth": profit_growth,
            "turnover": turnover_amount,
            "volume": volume,
            "market": "HK",
            "currency": "HKD",
        }
    except Exception as e:
        logger.debug(f"解析港股实时行情失败 {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# 统一解析入口
# ---------------------------------------------------------------------------

def _parse_realtime(raw: str, symbol: str) -> Optional[Dict]:
    """根据symbol前缀分派到A股/港股解析"""
    if symbol.startswith("hk"):
        return _parse_realtime_hk(raw, symbol)
    return _parse_realtime_a(raw, symbol)


# ---------------------------------------------------------------------------
# 历史K线解析 (A股/港股通用)
# ---------------------------------------------------------------------------

def _parse_history(raw: str, symbol: str) -> Optional[Dict]:
    """解析腾讯历史K线数据，计算20日动量

    港股K线JSON中键名为"day"而非"qfqday"，代码已兼容。
    """
    try:
        prefix = "kline_dayqfq="
        idx = raw.find(prefix)
        if idx < 0:
            return None
        json_str = raw[idx + len(prefix):]
        data = json.loads(json_str)

        stock_data = data.get("data", {})
        day_kline = None
        for key in stock_data:
            if "day" in stock_data[key]:
                day_kline = stock_data[key]["day"]
                break
            if "qfqday" in stock_data[key]:
                day_kline = stock_data[key]["qfqday"]
                break

        if not day_kline or len(day_kline) < 2:
            return None

        closes = [float(k[2]) for k in day_kline if len(k) > 2]
        if len(closes) < 2:
            return None

        momentum_days = min(20, len(closes) - 1)
        current = closes[-1]
        past = closes[-1 - momentum_days]
        momentum_20d = ((current - past) / past) * 100 if past > 0 else 0

        return {
            "momentum_20d": round(momentum_20d, 2),
            "history_days": len(closes),
        }
    except Exception as e:
        logger.debug(f"解析历史数据失败 {symbol}: {e}")
        return None


def _parse_history_ohlcv(raw: str, symbol: str) -> Optional[pd.DataFrame]:
    """解析腾讯历史K线数据为 OHLCV DataFrame

    Returns:
        DataFrame with columns: date, open, close, high, low, volume
        或 None（解析失败）
    """
    try:
        prefix = "kline_dayqfq="
        idx = raw.find(prefix)
        if idx < 0:
            return None
        json_str = raw[idx + len(prefix):]
        data = json.loads(json_str)

        stock_data = data.get("data", {})
        day_kline = None
        for key in stock_data:
            if "day" in stock_data[key]:
                day_kline = stock_data[key]["day"]
                break
            if "qfqday" in stock_data[key]:
                day_kline = stock_data[key]["qfqday"]
                break

        if not day_kline or len(day_kline) < 2:
            return None

        rows = []
        for k in day_kline:
            if len(k) < 6:
                continue
            try:
                rows.append({
                    "date": k[0],
                    "open": float(k[1]),
                    "close": float(k[2]),
                    "high": float(k[3]),
                    "low": float(k[4]),
                    "volume": float(k[5]) if k[5] else 0,
                })
            except (ValueError, IndexError):
                continue

        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"解析OHLCV数据失败 {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# 同步获取器 (requests)
# ---------------------------------------------------------------------------

class _SyncFetcher:
    """同步方式获取腾讯财经数据 (fallback)"""

    def __init__(self, retry_times: int = 3, delay_range: tuple = (0.3, 1.0)):
        self.retry_times = retry_times
        self.delay_range = delay_range

    def fetch_realtime(self, codes: List[str]) -> List[Dict]:
        import requests

        results = []
        batch_size = 50
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            symbols = ",".join(_code_to_symbol(c) for c in batch)
            url = _REALTIME_URL.format(symbol=symbols)
            try:
                resp = requests.get(url, headers={**_HEADERS, "User-Agent": random.choice(_USER_AGENTS)},
                                    timeout=15)
                resp.encoding = "gbk"
                lines = resp.text.strip().split(";")
                for line in lines:
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    sym_part = line.split("=")[0].strip()
                    sym = sym_part.replace("v_", "")
                    data = _parse_realtime(line, sym)
                    if data:
                        results.append(data)
            except Exception as e:
                logger.warning(f"同步批量获取实时行情失败 (批次{i // batch_size}): {e}")
            time.sleep(random.uniform(*self.delay_range))
        return results

    def fetch_history(self, codes: List[str], days: int = 30) -> Dict[str, Dict]:
        import requests

        results = {}
        for code in codes:
            symbol = _code_to_symbol(code)
            url = _HISTORY_URL.format(symbol=symbol, days=days)
            for attempt in range(self.retry_times):
                try:
                    resp = requests.get(url, headers={**_HEADERS, "User-Agent": random.choice(_USER_AGENTS)},
                                        timeout=15)
                    data = _parse_history(resp.text, symbol)
                    if data:
                        results[code] = data
                    break
                except Exception as e:
                    if attempt < self.retry_times - 1:
                        time.sleep(0.5 * (2 ** attempt))
                    else:
                        logger.debug(f"获取历史数据失败 {code}: {e}")
        return results

    def fetch_history_ohlcv(self, codes: List[str], days: int = 230) -> Dict[str, pd.DataFrame]:
        """获取多只股票的 OHLCV 历史K线"""
        import requests

        results = {}
        for code in codes:
            symbol = _code_to_symbol(code)
            url = _HISTORY_URL.format(symbol=symbol, days=days)
            for attempt in range(self.retry_times):
                try:
                    resp = requests.get(url, headers={**_HEADERS, "User-Agent": random.choice(_USER_AGENTS)},
                                        timeout=15)
                    df = _parse_history_ohlcv(resp.text, symbol)
                    if df is not None and not df.empty:
                        results[code] = df
                    break
                except Exception as e:
                    if attempt < self.retry_times - 1:
                        time.sleep(0.5 * (2 ** attempt))
                    else:
                        logger.debug(f"获取OHLCV数据失败 {code}: {e}")
        return results


# ---------------------------------------------------------------------------
# 异步获取器
# ---------------------------------------------------------------------------

class TencentFetcher:
    """腾讯财经API数据获取器 (异步+同步，支持A股+港股通)

    Usage:
        fetcher = TencentFetcher(max_concurrent=20)
        # 异步 — 全量(A股+港股通)
        stocks = await fetcher.fetch_all()
        # 异步 — 仅港股通
        stocks = await fetcher.fetch_all(market="hk")
        # 同步
        stocks = fetcher.fetch_all_sync(market="a")
    """

    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self._sync_fetcher = _SyncFetcher()

    async def _get_session(self):
        import aiohttp
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
        )
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=15)
        return aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def _fetch_single_realtime(self, session, symbol: str, semaphore) -> Optional[Dict]:
        import aiohttp
        async with semaphore:
            url = _REALTIME_URL.format(symbol=symbol)
            headers = {**_HEADERS, "User-Agent": random.choice(_USER_AGENTS)}
            for attempt in range(3):
                try:
                    async with session.get(url, headers=headers) as resp:
                        text = await resp.text(encoding="gbk", errors="replace")
                        return _parse_realtime(text, symbol)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                    else:
                        logger.debug(f"获取实时行情失败 {symbol}: {e}")
                        return None

    async def fetch_realtime_batch(self, codes: List[str]) -> List[Dict]:
        """异步批量获取实时行情"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with await self._get_session() as session:
            symbols = [_code_to_symbol(c) for c in codes]
            tasks = [self._fetch_single_realtime(session, s, semaphore) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    async def _fetch_single_history(self, session, symbol: str, days: int, semaphore) -> Optional[Dict]:
        import aiohttp
        async with semaphore:
            url = _HISTORY_URL.format(symbol=symbol, days=days)
            headers = {**_HEADERS, "User-Agent": random.choice(_USER_AGENTS)}
            for attempt in range(3):
                try:
                    async with session.get(url, headers=headers) as resp:
                        text = await resp.text(errors="replace")
                        return _parse_history(text, symbol)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                    else:
                        logger.debug(f"获取历史数据失败 {symbol}: {e}")
                        return None

    async def fetch_history_batch(self, codes: List[str], days: int = 30) -> Dict[str, Dict]:
        """异步批量获取历史K线"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with await self._get_session() as session:
            symbols = [_code_to_symbol(c) for c in codes]
            tasks = [self._fetch_single_history(session, s, days, semaphore) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for code, result in zip(codes, results):
            if isinstance(result, dict):
                output[code] = result
        return output

    async def fetch_all(self, codes: List[str] = None, market: str = "all") -> List[Dict]:
        """异步全量获取: 实时行情 + 历史动量

        Args:
            codes: 指定代码列表 (优先于market)
            market: "all" / "a" / "hk" / "csi300"
        """
        if codes is None:
            codes = load_codes_by_market(market)
        if not codes:
            logger.warning("无可用股票代码列表")
            return []

        logger.info(f"开始获取 {len(codes)} 只股票数据 (异步, 并发={self.max_concurrent})")
        t0 = time.time()

        # 阶段1: 实时行情
        realtime = await self.fetch_realtime_batch(codes)
        valid_codes = [s["code"] for s in realtime if s.get("code")]
        logger.info(f"实时行情: {len(valid_codes)}/{len(codes)} 只成功")

        # 阶段2: 历史动量
        history = await self.fetch_history_batch(valid_codes, days=30)
        logger.info(f"历史动量: {len(history)}/{len(valid_codes)} 只成功")

        # 合并
        for stock in realtime:
            code = stock.get("code", "")
            hist = history.get(code)
            if hist:
                stock.update(hist)
            else:
                stock["momentum_20d"] = 0.0
                stock["history_days"] = 0

        elapsed = time.time() - t0
        logger.info(f"数据获取完成: {len(realtime)} 只, 耗时 {elapsed:.1f}s")
        return realtime

    # ---- 同步包装器 ----

    def fetch_all_sync(self, codes: List[str] = None, market: str = "all") -> List[Dict]:
        """同步全量获取 (自动选择异步或纯同步)"""
        try:
            import aiohttp  # noqa: F401
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            if loop is not None and loop.is_running():
                logger.info("检测到运行中的事件循环，使用同步模式")
                return self._fetch_all_pure_sync(codes, market)
            return asyncio.run(self.fetch_all(codes, market))
        except ImportError:
            logger.info("aiohttp未安装，使用同步模式")
            return self._fetch_all_pure_sync(codes, market)

    def _fetch_all_pure_sync(self, codes: List[str] = None, market: str = "all") -> List[Dict]:
        """纯同步获取 (无需aiohttp)"""
        if codes is None:
            codes = load_codes_by_market(market)
        if not codes:
            return []

        logger.info(f"开始获取 {len(codes)} 只股票数据 (同步模式)")
        t0 = time.time()

        realtime = self._sync_fetcher.fetch_realtime(codes)
        valid_codes = [s["code"] for s in realtime if s.get("code")]

        history = self._sync_fetcher.fetch_history(valid_codes, days=30)

        for stock in realtime:
            code = stock.get("code", "")
            hist = history.get(code)
            if hist:
                stock.update(hist)
            else:
                stock["momentum_20d"] = 0.0
                stock["history_days"] = 0

        elapsed = time.time() - t0
        logger.info(f"数据获取完成(同步): {len(realtime)} 只, 耗时 {elapsed:.1f}s")
        return realtime

    # ---- OHLCV 历史数据 (趋势选股用) ----

    def fetch_history_ohlcv_sync(self, codes: List[str], days: int = 230) -> Dict[str, pd.DataFrame]:
        """同步获取多只股票的 OHLCV 历史K线

        Args:
            codes: 股票代码列表
            days: 历史天数 (默认230, 足够计算SMA200)

        Returns:
            {code: DataFrame(columns=date,open,close,high,low,volume)}
        """
        return self._sync_fetcher.fetch_history_ohlcv(codes, days)

    def fetch_index_history_sync(self, index_code: str = "sh000001", days: int = 230) -> Optional[pd.DataFrame]:
        """同步获取指数历史K线

        Args:
            index_code: 指数代码 (如 sh000001=上证, sz399001=深证, sz399006=创业板)
            days: 历史天数

        Returns:
            OHLCV DataFrame 或 None
        """
        import requests

        url = _HISTORY_URL.format(symbol=index_code, days=days)
        headers = {**_HEADERS, "User-Agent": random.choice(_USER_AGENTS)}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            return _parse_history_ohlcv(resp.text, index_code)
        except Exception as e:
            logger.warning(f"获取指数历史数据失败 {index_code}: {e}")
            return None
