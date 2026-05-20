#!/usr/bin/env python3
"""Portfolio Engine Demo: 多标的组合回测

用法:
    # 使用默认yaml配置
    python3 scripts/demo_portfolio.py

    # 选股筛选
    python3 scripts/demo_portfolio.py --screen

    # 策略搜索
    python3 scripts/demo_portfolio.py --search

    # 全流程: 选股 → 策略搜索 → 回测
    python3 scripts/demo_portfolio.py --screen --search --plot

    # 多策略投票
    python3 scripts/demo_portfolio.py --strategy macd rsi kdj --vote-mode majority

    # 多行业选股 (每行业限N只)
    python3 scripts/demo_portfolio.py --industry C32... J66... --per-industry 3

    # 命令行覆盖yaml参数
    python3 scripts/demo_portfolio.py --strategy macd --start-date 2021-01-01
"""

import argparse
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.storage import ParquetStorage
from data.manager import DataManager
from data.industry import IndustryLookup
from strategies.engine import BacktestConfig
from strategies.result import calc_performance_metrics
from strategies.portfolio_engine import (
    PortfolioEngine, MultiStrategy, PortfolioContext,
    EqualWeightAllocation, IndustryBalancedAllocation, CustomAllocation, RebalanceConfig,
)
from strategies.composite import CompositeMultiStrategy
from strategies.base import Signal, SignalType

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("demo_portfolio")

DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "portfolio.yaml")


# ---------------------------------------------------------------------------
# 策略定义
# ---------------------------------------------------------------------------

class MACrossMulti(MultiStrategy):
    """多标的双均线交叉策略"""

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('ma', period=self.fast_period)
        self.register_indicator('ma', period=self.slow_period)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        fast_col = f'ma_{self.fast_period}'
        slow_col = f'ma_{self.slow_period}'
        for sym, bars in ctx.historical.items():
            if len(bars) < 2:
                continue
            curr_fast = bars[fast_col].iloc[-1]
            curr_slow = bars[slow_col].iloc[-1]
            prev_fast = bars[fast_col].iloc[-2]
            prev_slow = bars[slow_col].iloc[-2]
            if any(pd.isna(v) for v in [curr_fast, curr_slow, prev_fast, prev_slow]):
                continue
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class RSIMulti(MultiStrategy):
    """多标的RSI超买超卖策略"""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('rsi', period=self.period)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        rsi_col = f'rsi_{self.period}'
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            rsi = bars[rsi_col].iloc[-1]
            if pd.isna(rsi):
                continue
            if rsi < self.oversold:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif rsi > self.overbought:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class MACDMulti(MultiStrategy):
    """多标的MACD策略：DIF上穿DEA买入，下穿卖出"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('macd', fast_period=self.fast_period,
                                slow_period=self.slow_period, signal_period=self.signal_period)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bars in ctx.historical.items():
            if len(bars) < 2:
                continue
            curr_dif = bars['macd_dif'].iloc[-1]
            curr_dea = bars['macd_dea'].iloc[-1]
            prev_dif = bars['macd_dif'].iloc[-2]
            prev_dea = bars['macd_dea'].iloc[-2]
            if any(pd.isna(v) for v in [curr_dif, curr_dea, prev_dif, prev_dea]):
                continue
            if prev_dif <= prev_dea and curr_dif > curr_dea:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif prev_dif >= prev_dea and curr_dif < curr_dea:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class BollingerMulti(MultiStrategy):
    """多标的布林带策略：触及下轨买入，触及上轨卖出"""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__()
        self.period = period
        self.num_std = num_std

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('bollinger', period=self.period, num_std=self.num_std)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            close = bars['close'].iloc[-1]
            lower = bars['boll_lower'].iloc[-1]
            upper = bars['boll_upper'].iloc[-1]
            if any(pd.isna(v) for v in [close, lower, upper]):
                continue
            if close <= lower:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif close >= upper:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class KDJMulti(MultiStrategy):
    """多标的KDJ策略：K上穿D且J超卖买入，K下穿D且J超买卖出"""

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3,
                 oversold: float = 20.0, overbought: float = 80.0):
        super().__init__()
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.oversold = oversold
        self.overbought = overbought

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('kdj', n=self.n, m1=self.m1, m2=self.m2)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bars in ctx.historical.items():
            if len(bars) < 2:
                continue
            curr_k = bars['k'].iloc[-1]
            curr_d = bars['d'].iloc[-1]
            curr_j = bars['j'].iloc[-1]
            prev_k = bars['k'].iloc[-2]
            prev_d = bars['d'].iloc[-2]
            if any(pd.isna(v) for v in [curr_k, curr_d, curr_j, prev_k, prev_d]):
                continue
            if prev_k <= prev_d and curr_k > curr_d and curr_j < self.oversold:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif prev_k >= prev_d and curr_k < curr_d and curr_j > self.overbought:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class SARMulti(MultiStrategy):
    """多标的SAR抛物线策略：趋势翻转时产生信号"""

    def __init__(self):
        super().__init__()
        self._prev_trends: dict = {}

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('sar')
        for sym in ctx.bars:
            self._prev_trends[sym] = None

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bar in ctx.bars.items():
            sar_trend = bar.get('sar_trend')
            if sar_trend is None or pd.isna(sar_trend):
                self._prev_trends[sym] = None
                continue
            prev = self._prev_trends.get(sym)
            if prev is not None:
                if prev <= 0 and sar_trend > 0:
                    signals.append(Signal(type=SignalType.BUY, symbol=sym))
                elif prev > 0 and sar_trend <= 0:
                    signals.append(Signal(type=SignalType.SELL, symbol=sym))
            self._prev_trends[sym] = sar_trend
        return signals


class WRMulti(MultiStrategy):
    """多标的威廉%R策略：低于-80超卖买入，高于-20超买卖出"""

    def __init__(self, period: int = 14, oversold: float = -80.0, overbought: float = -20.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('wr', period=self.period)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        wr_col = f'wr_{self.period}'
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            wr_val = bars[wr_col].iloc[-1]
            if pd.isna(wr_val):
                continue
            if wr_val < self.oversold:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif wr_val > self.overbought:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class CCIMulti(MultiStrategy):
    """多标的CCI策略：CCI低于-100超卖买入，高于+100超买卖出"""

    def __init__(self, period: int = 14, oversold: float = -100.0, overbought: float = 100.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('cci', period=self.period)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        cci_col = f'cci_{self.period}'
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            cci_val = bars[cci_col].iloc[-1]
            if pd.isna(cci_val):
                continue
            if cci_val < self.oversold:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif cci_val > self.overbought:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class KeltnerMulti(MultiStrategy):
    """多标的凯尔特纳通道策略：触及下轨买入，触及上轨卖出"""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, num_atr: float = 1.5):
        super().__init__()
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.num_atr = num_atr

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('keltner', ema_period=self.ema_period,
                                atr_period=self.atr_period, num_atr=self.num_atr)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            close = bars['close'].iloc[-1]
            lower = bars['kelt_lower'].iloc[-1]
            upper = bars['kelt_upper'].iloc[-1]
            if any(pd.isna(v) for v in [close, lower, upper]):
                continue
            if close <= lower:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif close >= upper:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class OBVMulti(MultiStrategy):
    """多标的OBV能量潮策略：OBV上穿其MA买入，下穿卖出"""

    def __init__(self, ma_period: int = 20):
        super().__init__()
        self.ma_period = ma_period

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('obv')

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bars in ctx.historical.items():
            if len(bars) < self.ma_period:
                continue
            obv = bars['obv'].iloc[-1]
            if pd.isna(obv):
                continue
            obv_series = bars['obv']
            obv_ma = obv_series.rolling(window=self.ma_period).mean().iloc[-1]
            prev_obv_ma = obv_series.rolling(window=self.ma_period).mean().iloc[-2]
            prev_obv = obv_series.iloc[-2]
            if pd.isna(obv_ma) or pd.isna(prev_obv_ma):
                continue
            if prev_obv <= prev_obv_ma and obv > obv_ma:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif prev_obv >= prev_obv_ma and obv < obv_ma:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class MFIMulti(MultiStrategy):
    """多标的MFI资金流量策略：MFI低于20超卖买入，高于80超买卖出"""

    def __init__(self, period: int = 14, oversold: float = 20.0, overbought: float = 80.0):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('mfi', period=self.period)

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        mfi_col = f'mfi_{self.period}'
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            mfi_val = bars[mfi_col].iloc[-1]
            if pd.isna(mfi_val):
                continue
            if mfi_val < self.oversold:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif mfi_val > self.overbought:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


class VWAPMulti(MultiStrategy):
    """多标的VWAP策略：价格低于VWAP买入，高于VWAP卖出"""

    def on_init_multi(self, ctx: PortfolioContext) -> None:
        self.register_indicator('vwap')

    def on_bar_multi(self, ctx: PortfolioContext) -> list:
        signals = []
        for sym, bars in ctx.historical.items():
            if len(bars) < 1:
                continue
            close = bars['close'].iloc[-1]
            vwap_val = bars['vwap'].iloc[-1]
            if any(pd.isna(v) for v in [close, vwap_val]) or vwap_val < 1e-8:
                continue
            gap_pct = (close - vwap_val) / vwap_val
            if gap_pct < -0.01:
                signals.append(Signal(type=SignalType.BUY, symbol=sym))
            elif gap_pct > 0.01:
                signals.append(Signal(type=SignalType.SELL, symbol=sym))
        return signals


STRATEGY_MAP = {
    'ma':       ('MACross',  MACrossMulti),
    'rsi':      ('RSI',      RSIMulti),
    'macd':     ('MACD',     MACDMulti),
    'bollinger':('Bollinger', BollingerMulti),
    'kdj':      ('KDJ',      KDJMulti),
    'sar':      ('SAR',      SARMulti),
    'wr':       ('WR',       WRMulti),
    'cci':      ('CCI',      CCIMulti),
    'keltner':  ('Keltner',  KeltnerMulti),
    'obv':      ('OBV',      OBVMulti),
    'mfi':      ('MFI',      MFIMulti),
    'vwap':     ('VWAP',     VWAPMulti),
}


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def merge_config(yaml_cfg: dict, args) -> dict:
    """合并yaml配置和命令行参数, 命令行优先"""
    cfg = dict(yaml_cfg)
    # 命令行显式指定的参数覆盖yaml
    for key in ['strategy', 'vote_mode', 'start_date', 'end_date', 'plot', 'plot_engine', 'min_bars', 'per_industry']:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    # CLI --symbols 覆盖 yaml symbols/industry
    if getattr(args, 'symbols', None):
        cfg['symbols'] = [s.strip() for s in args.symbols.split(',')]
    # CLI --industry 覆盖 yaml industry
    if getattr(args, 'industry', None):
        cfg['industry'] = args.industry
    # CLI --weights 覆盖 yaml allocation/weights
    if getattr(args, 'weights', None):
        cfg['allocation'] = 'custom'
        cfg['weights'] = {p.split(':')[0].strip(): float(p.split(':')[1])
                          for p in args.weights.split(',')}
    # CLI --rebalance / --drift
    if getattr(args, 'rebalance', None):
        cfg['rebalance'] = {'frequency': args.rebalance}
    elif getattr(args, 'drift', None):
        cfg['rebalance'] = {'drift': args.drift}
    # CLI --screen / --search
    if getattr(args, 'screen', None):
        cfg.setdefault('screening', {})['enabled'] = True
    if getattr(args, 'search', None):
        cfg.setdefault('strategy_search', {})['enabled'] = True
    return cfg


def build_strategy(cfg: dict):
    names = cfg.get('strategy', 'ma')
    if isinstance(names, str):
        names = [names]

    for name in names:
        if name not in STRATEGY_MAP:
            raise ValueError(f"未知策略: {name}, 可选: {list(STRATEGY_MAP.keys())}")

    vote_mode = cfg.get('vote_mode', 'any')
    if vote_mode not in ('unanimous', 'any', 'majority'):
        raise ValueError(f"未知投票模式: {vote_mode}, 可选: unanimous / any / majority")

    # 构建子策略列表
    strategies = []
    labels = []
    for name in names:
        label, cls = STRATEGY_MAP[name]
        params = dict(cfg.get('strategy_params', {}))
        oversold = cfg.get('oversold')
        overbought = cfg.get('overbought')

        if oversold is not None and 'oversold' not in params:
            params['oversold'] = oversold
        if overbought is not None and 'overbought' not in params:
            params['overbought'] = overbought

        sig = inspect.signature(cls.__init__)
        valid_keys = {p.name for p in sig.parameters.values() if p.name != 'self'}
        params = {k: v for k, v in params.items() if k in valid_keys}

        strategies.append(cls(**params))
        labels.append(label)

    # 单策略直接返回，多策略用 CompositeMultiStrategy
    if len(strategies) == 1:
        return labels[0], strategies[0]

    combined_label = '+'.join(labels)
    strategy = CompositeMultiStrategy(strategies, mode=vote_mode)
    return combined_label, strategy


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def get_symbols(cfg: dict) -> tuple:
    """返回 (symbols, industry_map)，industry_map 为 标的→行业 映射"""
    if cfg.get('symbols'):
        symbols = cfg['symbols']
        logger.info(f"指定股票 {len(symbols)} 只: {symbols}")
        # 手动指定标的时，尝试从行业表补全映射
        lk = IndustryLookup(str(PROJECT_ROOT / "data"))
        industry_map = {sym: lk.get_industry(sym) or 'default' for sym in symbols}
        return symbols, industry_map
    industry = cfg.get('industry')
    if not industry:
        logger.error("未指定 symbols 或 industry")
        sys.exit(1)
    # 取全行业股票，per_industry 留到 screening 后再截取
    lk = IndustryLookup(str(PROJECT_ROOT / "data"))
    symbols = lk.get_stocks(industry)
    if not symbols:
        logger.error(f"行业 '{industry}' 未找到或无股票")
        sys.exit(1)
    industry_disp = industry if isinstance(industry, str) else ', '.join(industry)
    logger.info(f"行业 [{industry_disp}] 共 {len(symbols)} 只股票")
    # 构建标的→行业映射
    industry_map = {sym: lk.get_industry(sym) or 'unknown' for sym in symbols}
    return symbols, industry_map


def load_data(symbols: list, min_bars: int = 60,
              start_date: str = None, end_date: str = None) -> dict:
    storage = ParquetStorage(str(PROJECT_ROOT / "data"))
    mgr = DataManager(str(PROJECT_ROOT / "data"))
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp('2020-01-01')
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    data = {}
    skipped = []
    for sym in symbols:
        df = storage.load(sym, '1d')
        # 本地无数据时尝试下载
        if df is None:
            logger.info(f"本地无数据，尝试下载 {sym} ...")
            for attempt in range(3):
                try:
                    kline = mgr.get_kline(sym, start=start, end=end, freq='1d', source='baostock')
                    df = kline.df if kline and kline.df is not None and len(kline.df) > 0 else None
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"下载 {sym} 失败 (第{attempt+1}次): {e}, 重试...")
                        time.sleep(1)
                    else:
                        logger.warning(f"下载 {sym} 失败: {e}")
                    df = None
        if df is None:
            skipped.append(sym)
            continue
        # 时间范围过滤
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        if len(df) >= min_bars:
            data[sym] = df
        else:
            logger.debug(f"{sym}: 过滤后 {len(df)} 条 < min_bars={min_bars}")
            skipped.append(sym)
    if skipped:
        logger.warning(f"跳过数据不足的标的 ({len(skipped)}): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return data


# ---------------------------------------------------------------------------
# 结果展示
# ---------------------------------------------------------------------------

def load_benchmark(benchmark_code: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """加载大盘基准收盘价序列，优先读本地缓存"""
    from data.base import KlineData
    from data.providers.baostock import BaostockProvider
    from data.providers.akshare import AkShareProvider
    storage = ParquetStorage(str(PROJECT_ROOT / "data"))
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp('2020-01-01')
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    # 先查本地缓存
    cached = storage.load(benchmark_code, '1d', start=start.to_pydatetime(), end=end.to_pydatetime())
    if cached is not None and len(cached) > 20:
        return cached['close']
    # 缓存不存在，从数据源下载并保存
    bs_code = benchmark_code
    if '.SH' in bs_code.upper():
        bs_code = 'sh.' + bs_code.split('.')[0]
    elif '.SZ' in bs_code.upper():
        bs_code = 'sz.' + bs_code.split('.')[0]
    for provider_cls, code in [(BaostockProvider, bs_code), (AkShareProvider, benchmark_code)]:
        try:
            prov = provider_cls()
            kline = prov.get_index_data(code, start=start.to_pydatetime(), end=end.to_pydatetime())
            if kline and kline.df is not None and len(kline.df) > 0:
                # 保存到本地缓存
                kline_save = KlineData(symbol=benchmark_code, freq='1d',
                                       start=start.to_pydatetime(), end=end.to_pydatetime(),
                                       df=kline.df)
                storage.save(kline_save)
                return kline.df['close']
        except Exception:
            continue
    logger.warning(f"加载基准 {benchmark_code} 失败")
    return None


def print_result(result, symbols, strategy_name: str, benchmark_metrics: dict = None):
    print(f"\n{'='*50}")
    print(f"Portfolio Engine 回测结果: {strategy_name}")
    print(f"{'='*50}")
    print(f"标的数量:   {len(symbols)}")
    print(f"总收益率:   {result.total_return:.2%}")
    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"夏普比率:   {result.sharpe_ratio:.2f}")
    print(f"最大回撤:   {result.max_drawdown:.2%}")
    print(f"胜率:       {result.win_rate:.2%}")
    print(f"盈亏比:     {result.profit_loss_ratio:.2f}")
    print(f"交易次数:   {result.total_trades}")

    # 大盘基准对比
    if benchmark_metrics:
        bm = benchmark_metrics
        alpha = result.annual_return - bm['annual_return']
        print(f"\n--- 大盘基准对比 ({bm.get('name', '')}) ---")
        print(f"{'指标':<12} {'策略':>10} {'基准':>10} {'超额':>10}")
        print(f"{'-'*44}")
        print(f"{'总收益率':<12} {result.total_return:>10.2%} {bm['total_return']:>10.2%} {result.total_return - bm['total_return']:>+10.2%}")
        print(f"{'年化收益率':<12} {result.annual_return:>10.2%} {bm['annual_return']:>10.2%} {alpha:>+10.2%}")
        print(f"{'夏普比率':<12} {result.sharpe_ratio:>10.2f} {bm['sharpe_ratio']:>10.2f} {result.sharpe_ratio - bm['sharpe_ratio']:>+10.2f}")
        print(f"{'最大回撤':<12} {result.max_drawdown:>10.2%} {bm['max_drawdown']:>10.2%} {bm['max_drawdown'] - result.max_drawdown:>+10.2%}")
        print(f"  (回撤超额为正=策略回撤更小)")
        # 信息比率 (alpha / 跟踪误差)
        if bm.get('tracking_error', 0) > 0:
            ir = alpha / bm['tracking_error']
            print(f"  信息比率:   {ir:.2f}")

    print(f"\n--- 各标的期末市值 ---")
    for sym in symbols:
        eq = result.per_symbol_equity.get(sym)
        if eq is not None and len(eq) > 0:
            final_val = eq.iloc[-1]
            pct = final_val / result.initial_capital * 100
            print(f"  {sym}: {final_val:>12,.0f}  ({pct:.1f}%)")

    print(f"\n--- 各标的交易统计 ---")
    for sym in symbols:
        sym_trades = result.per_symbol_trades.get(sym, [])
        if sym_trades:
            wins = sum(1 for t in sym_trades if t.pnl and t.pnl > 0)
            total_pnl = sum(t.pnl for t in sym_trades if t.pnl)
            print(f"  {sym}: {len(sym_trades)} 笔, 胜率 {wins/len(sym_trades):.0%}, 盈亏 {total_pnl:>+,.0f}")
        else:
            print(f"  {sym}: 无交易")


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_portfolio(result, symbols, data, strategy_name, engine='matplotlib'):
    from visualization import plot_equity, plot_trade_pnl

    fig_equity = plot_equity(result, engine=engine)
    fig_allocation = _plot_symbol_allocation(result, symbols, engine)
    fig_pnl = plot_trade_pnl(result, engine=engine)

    if engine == 'plotly':
        fig_equity.show()
        fig_allocation.show()
        fig_pnl.show()
    else:
        import matplotlib.pyplot as plt
        plt.show()


def _plot_symbol_allocation(result, symbols, engine='matplotlib'):
    per_eq = result.per_symbol_equity
    if not per_eq:
        return None

    dfs = []
    for sym in symbols:
        s = per_eq.get(sym)
        if s is not None and len(s) > 0:
            dfs.append(s.rename(sym))
    if not dfs:
        return None

    combined = pd.concat(dfs, axis=1).fillna(0).sort_index()

    if engine == 'plotly':
        import plotly.graph_objects as go
        fig = go.Figure()
        for sym in combined.columns:
            fig.add_trace(go.Scatter(
                x=combined.index, y=combined[sym], name=sym,
                stackgroup='one', hovertemplate=f'{sym}: %{{y:,.0f}}<extra></extra>',
            ))
        fig.update_layout(
            title='各标的持仓市值', yaxis_title='市值',
            height=500, template='plotly_white',
        )
        return fig

    import matplotlib.pyplot as plt
    from visualization.candlestick import _setup_chinese_font
    _setup_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(combined.index, [combined[s].values for s in combined.columns],
                 labels=combined.columns.tolist(), alpha=0.8)
    ax.legend(loc='upper left', fontsize=8, ncol=min(len(symbols), 4))
    ax.set_ylabel('市值')
    ax.set_xlabel('日期')
    ax.set_title('各标的持仓市值')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_backtest(cfg: dict):
    # Stage 1: 获取标的列表
    symbols, industry_map = get_symbols(cfg)

    # Stage 2: 选股筛选 (可选)
    screen_cfg = cfg.get('screening', {})
    if screen_cfg.get('enabled'):
        logger.info("--- 选股筛选 ---")
        screened = run_screening(cfg)
        # per_industry: 筛选后按行业截取前N只
        per_industry = cfg.get('per_industry', 0)
        if per_industry > 0 and screened:
            from collections import defaultdict as _dd
            ind_groups = _dd(list)
            for s in screened:
                ind_groups[industry_map.get(s, 'unknown')].append(s)
            screened = []
            for ind, syms in ind_groups.items():
                screened.extend(syms[:per_industry])
        # 用筛选后的标的，但保留行业映射
        industry_map = {s: industry_map.get(s, 'unknown') for s in screened}
        symbols = screened
    else:
        # 无 screening 时，per_industry 直接限制候选池
        per_industry = cfg.get('per_industry', 0)
        if per_industry > 0:
            from collections import defaultdict as _dd
            ind_groups = _dd(list)
            for s in symbols:
                ind_groups[industry_map.get(s, 'unknown')].append(s)
            symbols = []
            for ind, syms in ind_groups.items():
                symbols.extend(syms[:per_industry])

    # Stage 3: 加载数据
    data = load_data(
        symbols,
        min_bars=cfg.get('min_bars', 60),
        start_date=cfg.get('start_date'),
        end_date=cfg.get('end_date'),
    )
    if not data:
        logger.error("无有效数据，退出")
        return

    # Stage 4: 策略搜索 (可选)
    search_cfg = cfg.get('strategy_search', {})
    if search_cfg.get('enabled'):
        logger.info("--- 策略搜索 ---")
        cfg = run_strategy_search(cfg, data)

    # Stage 5: 组合回测
    actual_symbols = list(data.keys())
    bt_config = BacktestConfig.from_yaml(str(PROJECT_ROOT / "config" / "settings.yaml"))

    # 分配策略
    alloc_type = cfg.get('allocation', 'auto')
    if alloc_type == 'custom' and cfg.get('weights'):
        weight_map = cfg['weights']
        allocation = CustomAllocation(weight_map)
        alloc_name = f"Custom({weight_map})"
    elif alloc_type == 'industry' or (alloc_type == 'auto' and len(set(industry_map.values())) > 1):
        allocation = IndustryBalancedAllocation(industry_map)
        n_ind = len(set(industry_map.values()))
        alloc_name = f"IndustryBalanced({n_ind}行业)"
    else:
        allocation = EqualWeightAllocation()
        alloc_name = "EqualWeight"

    # 再平衡配置
    rebalance = None
    rb_cfg = cfg.get('rebalance')
    if rb_cfg:
        rebalance = RebalanceConfig(
            frequency=rb_cfg.get('frequency'),
            drift_threshold=rb_cfg.get('drift'),
            record_trades=True,
        )

    # 策略
    strategy_name, strategy = build_strategy(cfg)

    logger.info("=" * 60)
    logger.info("Portfolio Engine: 多标的组合回测")
    logger.info(f"策略: {strategy_name} ({cfg.get('strategy')})")
    logger.info(f"标的: {actual_symbols}")
    logger.info(f"分配: {alloc_name}")
    logger.info(f"再平衡: {'无' if rebalance is None else rb_cfg}")
    logger.info(f"数据: {cfg.get('start_date', '最早')} ~ {cfg.get('end_date', '最新')}")
    logger.info(f"初始资金: {bt_config.initial_capital:,.0f}")
    logger.info("=" * 60)

    engine = PortfolioEngine(
        config=bt_config,
        allocation=allocation,
        rebalance=rebalance,
    )
    result = engine.run(strategy, data)

    # 计算大盘基准指标
    benchmark_code = bt_config.benchmark or '000300.SH'
    benchmark_name = {'000300.SH': '沪深300', '000016.SH': '上证50',
                      '000905.SH': '中证500', '000001.SH': '上证指数'}.get(benchmark_code, benchmark_code)
    benchmark_metrics = None
    bm_close = load_benchmark(benchmark_code, cfg.get('start_date'), cfg.get('end_date'))
    if bm_close is not None and len(bm_close) > 1:
        # 对齐日期：取策略权益曲线和基准的交集
        eq_curve = result.equity_curve
        if 'equity' in eq_curve.columns:
            eq_series = eq_curve['equity']
        else:
            eq_series = eq_curve.iloc[:, 0]
        bm_aligned = bm_close.reindex(eq_series.index).dropna()
        eq_aligned = eq_series.reindex(bm_aligned.index).dropna()
        common_idx = bm_aligned.index.intersection(eq_aligned.index)
        if len(common_idx) > 20:
            bm_metrics = calc_performance_metrics(bm_aligned.reindex(common_idx))
            # 计算跟踪误差和信息比率
            strat_returns = eq_aligned.reindex(common_idx).pct_change().dropna()
            bm_returns = bm_aligned.reindex(common_idx).pct_change().dropna()
            if len(strat_returns) > 0 and len(bm_returns) == len(strat_returns):
                tracking_error = (strat_returns.values - bm_returns.values).std() * np.sqrt(252)
                bm_metrics['tracking_error'] = tracking_error
            bm_metrics['name'] = benchmark_name
            benchmark_metrics = bm_metrics

    print_result(result, actual_symbols, strategy_name, benchmark_metrics=benchmark_metrics)

    if cfg.get('plot'):
        plot_portfolio(result, actual_symbols, data, strategy_name,
                       engine=cfg.get('plot_engine', 'matplotlib'))


# ---------------------------------------------------------------------------
# 选股 & 策略搜索
# ---------------------------------------------------------------------------

def run_screening(cfg: dict) -> list:
    """选股筛选, 返回 top_n 标的列表"""
    from strategies.screener import StockScreener, ScreeningConfig

    screen_cfg = cfg.get('screening', {})

    storage = ParquetStorage(str(PROJECT_ROOT / "data"))
    screener = StockScreener(
        config=ScreeningConfig(
            factors=screen_cfg.get('factors', {}),
            indicator_factors=screen_cfg.get('indicator_factors'),
            weights=screen_cfg.get('weights'),
            filters=screen_cfg.get('filters'),
            top_n=screen_cfg.get('top_n', 10),
            min_bars=cfg.get('min_bars', 60),
        ),
        storage=storage,
        data_dir=str(PROJECT_ROOT / "data"),
    )
    symbols, _ = get_symbols(cfg)
    top, all_scored = screener.screen_all(
        symbols=symbols,
        start_date=cfg.get('start_date'),
        end_date=cfg.get('end_date'),
    )
    print_screening_result(top, all_scored, len(symbols))
    return top['symbol'].tolist() if not top.empty else symbols


def run_strategy_search(cfg: dict, data: dict) -> dict:
    """策略搜索, 返回最优策略+参数, 并更新cfg"""
    from strategies.strategy_search import StrategySearcher, parse_search_config
    from strategies.engine import BacktestEngine

    search_cfg = cfg.get('strategy_search', {})
    specs = parse_search_config(search_cfg)
    bt_config = BacktestConfig.from_yaml(str(PROJECT_ROOT / "config" / "settings.yaml"))
    searcher = StrategySearcher(
        specs=specs,
        engine=BacktestEngine(config=bt_config),
        objective=search_cfg.get('objective', 'sharpe'),
        optimizer_type=search_cfg.get('optimizer', 'grid'),
        top_n=search_cfg.get('top_n', 3),
    )
    result = searcher.search_multi(data)
    print_search_result(result, searcher, data)

    # 选全局最优: 大多数标的上表现最好的策略
    best = result.loc[result['objective'].idxmax()]
    cfg['strategy'] = best['best_strategy']
    cfg['strategy_params'] = best['best_params']
    logger.info(f"策略搜索结果: {best['best_strategy']} {best['best_params']}")
    return cfg


def print_screening_result(top, all_scored, total_symbols: int = 0):
    print(f"\n{'='*60}")
    print("选股筛选结果 (因子)")
    print(f"{'='*60}")
    no_data = total_symbols - len(all_scored) if total_symbols else 0
    top_n = len(top) if not top.empty else 0
    passed = len(all_scored) if not all_scored.empty else 0
    print(f"候选池: {total_symbols} 只 → 有数据: {passed} 只 → 规则过滤后: {passed} 只 → 入选: {top_n} 只")
    if no_data > 0:
        print(f"  (数据不足跳过: {no_data} 只)")
    if all_scored.empty:
        print("无符合条件的标的")
        return

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)

    top_symbols = set(top['symbol'].tolist()) if not top.empty else set()

    # 按评分排序打印全部，标注入选
    all_sorted = all_scored.sort_values('composite_score', ascending=False).reset_index(drop=True)
    all_sorted.index = all_sorted.index + 1
    all_sorted.index.name = 'rank'

    print(f"\n--- 全部候选评分 (★=入选) ---")
    for idx, row in all_sorted.iterrows():
        marker = " ★" if row['symbol'] in top_symbols else ""
        score = row.get('composite_score', 0)
        print(f"  {idx:>3}. {row['symbol']}  评分={score:>7.2f}{marker}")


def print_search_result(result, searcher, data):
    print(f"\n{'='*60}")
    print("策略搜索结果")
    print(f"{'='*60}")
    if result.empty:
        print("无搜索结果")
        return

    # 各标的最优策略
    print("\n--- 各标的最优策略 ---")
    for _, row in result.iterrows():
        params_str = ', '.join(f'{k}={v}' for k, v in row['best_params'].items()) if row['best_params'] else ''
        print(f"  {row['symbol']}: {row['best_strategy']}({params_str}) "
              f"→ 夏普={row.get('sharpe_ratio', 0):.2f}, 收益={row.get('total_return', 0):.2%}")

    # 策略×标的矩阵
    try:
        matrix = searcher.search_matrix(data)
        print("\n--- 策略×标的 性能矩阵 ---")
        pd.set_option('display.float_format', '{:.2f}'.format)
        pd.set_option('display.width', 120)
        print(matrix.to_string())
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='Portfolio Engine: 多标的组合回测')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help='配置文件路径 (默认: config/portfolio.yaml)')
    # 以下参数可覆盖yaml配置
    parser.add_argument('--industry', type=str, default=None, nargs='+',
                        help='行业名称(可多选, 如 --industry C32... N78...)')
    parser.add_argument('--per-industry', type=int, default=None,
                        help='每个行业最多选多少只股票 (0=不限)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='逗号分隔 (如 601600.SH,000807.SZ)')
    parser.add_argument('--weights', type=str, default=None,
                        help='自定义权重 (如 601600.SH:0.4,000807.SZ:0.6)')
    parser.add_argument('--strategy', type=str, default=None, nargs='+',
                        choices=list(STRATEGY_MAP.keys()),
                        help='策略(可多选, 如 --strategy macd rsi kdj)')
    parser.add_argument('--vote-mode', type=str, default=None,
                        choices=['unanimous', 'any', 'majority'],
                        help='多策略投票模式 (默认: any)')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--rebalance', type=str, default=None, choices=['W', 'M', 'Q'])
    parser.add_argument('--drift', type=float, default=None)
    parser.add_argument('--min-bars', type=int, default=None)
    parser.add_argument('--plot', action='store_true', default=None)
    parser.add_argument('--plot-engine', type=str, default=None, choices=['matplotlib', 'plotly'])
    parser.add_argument('--screen', action='store_true', default=None,
                        help='启用选股筛选')
    parser.add_argument('--search', action='store_true', default=None,
                        help='启用策略搜索')
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    cfg = merge_config(yaml_cfg, args)
    run_backtest(cfg)


if __name__ == "__main__":
    main()
