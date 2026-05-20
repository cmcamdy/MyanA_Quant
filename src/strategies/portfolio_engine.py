"""多标的组合回测引擎"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

import pandas as pd
import numpy as np

from analysis.indicator_set import IndicatorSet
from .base import Strategy, Signal, SignalType, Context, Portfolio, Position
from .result import BacktestResult, TradeRecord, calc_performance_metrics
from .sizers import PositionSizer, FixedSizer
from .risk import RiskManager


@dataclass
class PortfolioContext:
    """多标的上下文"""
    bars: Dict[str, pd.Series]
    historical: Dict[str, pd.DataFrame]
    portfolio: Portfolio
    current_time: pd.Timestamp
    # 便捷属性：单标的模式下的第一个标的
    bar: Optional[pd.Series] = None
    symbol: Optional[str] = None


class MultiStrategy(Strategy):
    """多标的策略基类

    子类实现 on_init_multi 注册指标，on_bar_multi 返回多标的信号列表。
    继承 Strategy 保持兼容 — on_init/on_bar 有默认实现委托到多标的版本。
    """

    @abstractmethod
    def on_init_multi(self, context: PortfolioContext) -> None:
        pass

    @abstractmethod
    def on_bar_multi(self, context: PortfolioContext) -> List[Signal]:
        pass

    def on_init(self, context: Context) -> None:
        pctx = PortfolioContext(
            bars={context.symbol: context.bar},
            historical={context.symbol: context.bars},
            portfolio=context.portfolio,
            current_time=context.current_time,
            bar=context.bar,
            symbol=context.symbol,
        )
        self.on_init_multi(pctx)

    def on_bar(self, context: Context) -> Optional[Signal]:
        pctx = PortfolioContext(
            bars={context.symbol: context.bar},
            historical={context.symbol: context.bars},
            portfolio=context.portfolio,
            current_time=context.current_time,
            bar=context.bar,
            symbol=context.symbol,
        )
        signals = self.on_bar_multi(pctx)
        if signals:
            return signals[0]
        return None

    def on_finish_multi(self, context: PortfolioContext) -> None:
        pass

    def on_finish(self, context: Context) -> None:
        pctx = PortfolioContext(
            bars={context.symbol: context.bar} if context.bar is not None else {},
            historical={context.symbol: context.bars} if context.bars is not None else {},
            portfolio=context.portfolio,
            current_time=context.current_time,
            bar=context.bar,
            symbol=context.symbol,
        )
        self.on_finish_multi(pctx)


class EqualWeightAllocation:
    """等权分配"""

    def allocate(self, symbol: str, signal: Signal, portfolio: Portfolio,
                 current_price: float, target_symbols: List[str]) -> float:
        return 1.0 / len(target_symbols) if target_symbols else 0.0


class CustomAllocation:
    """自定义权重分配"""

    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def allocate(self, symbol: str, signal: Signal, portfolio: Portfolio,
                 current_price: float, target_symbols: List[str]) -> float:
        return self._weights.get(symbol, 0.0)


@dataclass
class RebalanceConfig:
    """再平衡配置"""
    # 时间再平衡: 'W'每周, 'M'每月, 'Q'每季, None不按时间再平衡
    frequency: Optional[str] = None
    # 漂移再平衡: 当权重偏离目标超过此阈值时触发
    drift_threshold: Optional[float] = None  # e.g. 0.05 = 5%
    # 再平衡时是否产生交易记录
    record_trades: bool = True


class PortfolioEngine:
    """多标的组合回测引擎"""

    def __init__(
        self,
        config: Optional['BacktestConfig'] = None,
        position_sizer: Optional[PositionSizer] = None,
        allocation: Optional[Union[EqualWeightAllocation, CustomAllocation]] = None,
        risk_manager: Optional[RiskManager] = None,
        rebalance: Optional[RebalanceConfig] = None,
    ):
        from .engine import BacktestConfig
        self._config = config or BacktestConfig()
        self._sizer = position_sizer or FixedSizer(percent=1.0)
        self._allocation = allocation or EqualWeightAllocation()
        self._risk = risk_manager
        self._rebalance = rebalance

    def run(
        self,
        strategy: Union[Strategy, MultiStrategy],
        data: Dict[str, pd.DataFrame],
        strategy_start: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """运行多标的回测

        Args:
            strategy: 回测策略
            data: {symbol: DataFrame} 多标的OHLCV数据
            strategy_start: 策略开始日期，之前的bar只跟踪持仓市值不生成信号
        """
        from .engine import BacktestConfig

        if strategy_start is not None:
            strategy_start = pd.Timestamp(strategy_start)

        if self._risk is not None:
            self._risk.reset()

        symbols = list(data.keys())
        is_multi = isinstance(strategy, MultiStrategy)

        # 1. 初始化策略
        portfolio = Portfolio(
            cash=self._config.initial_capital,
            equity=self._config.initial_capital,
        )

        # 设置初始持仓
        open_trades: Dict[str, Optional[TradeRecord]] = {sym: None for sym in symbols}
        if self._config.initial_positions:
            for sym, pos_spec in self._config.initial_positions.items():
                pos = portfolio.get_position(sym)
                pos.avg_cost = pos_spec['avg_cost']
                pos.quantity = pos_spec['quantity']
                portfolio.cash -= pos_spec['avg_cost'] * pos_spec['quantity']
                if sym in open_trades:
                    first_df = data[sym]
                    open_trades[sym] = TradeRecord(
                        symbol=sym,
                        entry_time=first_df.index[0],
                        exit_time=None,
                        entry_price=pos_spec['avg_cost'],
                        exit_price=None,
                        quantity=pos_spec['quantity'],
                        commission=0.0,
                    )

        first_symbol = symbols[0]
        first_df = data[first_symbol]
        init_pctx = PortfolioContext(
            bars={s: data[s].iloc[0] for s in symbols},
            historical={s: data[s].iloc[:1] for s in symbols},
            portfolio=portfolio,
            current_time=first_df.index[0],
            bar=first_df.iloc[0],
            symbol=first_symbol,
        )

        if is_multi:
            strategy.on_init_multi(init_pctx)
        else:
            init_ctx = Context(
                bar=first_df.iloc[0],
                bars=first_df.iloc[:1],
                portfolio=portfolio,
                current_time=first_df.index[0],
                symbol=first_symbol,
            )
            strategy.on_init(init_ctx)

        # 2. 预计算指标
        data_with_ind: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            iset = IndicatorSet()
            for spec in strategy.indicator_specs:
                iset.add(spec['name'], **{k: v for k, v in spec.items() if k != 'name'})
            data_with_ind[sym] = iset.compute(data[sym]) if iset.specs else data[sym].copy()

        # ATR 自动注册
        atr_col = None
        if self._risk is not None and self._risk._config.needs_atr:
            atr_period = self._risk._config.atr_period
            atr_col = f'atr_{atr_period}'
            for sym in symbols:
                if atr_col not in data_with_ind[sym].columns:
                    iset = IndicatorSet()
                    iset.add('atr', period=atr_period)
                    data_with_ind[sym] = iset.compute(data_with_ind[sym])

        # 3. 对齐时间轴
        all_indices = sorted(set().union(*(df.index for df in data_with_ind.values())))
        last_known: Dict[str, float] = {}

        # 4. 逐bar执行
        equity_records: List[dict] = []
        trades: List[TradeRecord] = []
        per_symbol_equity: Dict[str, List[float]] = {sym: [] for sym in symbols}
        _trailing_registered = False
        last_rebalance_date: Optional[pd.Timestamp] = None

        for t in all_indices:
            t = pd.Timestamp(t)
            # 确定本 bar 有哪些标的有数据
            bar_data: Dict[str, pd.Series] = {}
            for sym in symbols:
                df = data_with_ind[sym]
                if t in df.index:
                    row = df.loc[t]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    bar_data[sym] = row
                    last_known[sym] = row['close']

            # 更新持仓市值
            for sym in symbols:
                pos = portfolio.get_position(sym)
                if not pos.is_empty:
                    if sym in bar_data:
                        pos.market_value = pos.quantity * bar_data[sym]['close']
                    elif sym in last_known:
                        pos.market_value = pos.quantity * last_known[sym]

            portfolio.equity = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            # 策略开始前只跟踪持仓市值
            if strategy_start is not None and t < strategy_start:
                equity_records.append({
                    'time': t,
                    'equity': portfolio.equity,
                    'cash': portfolio.cash,
                    'market_value': sum(p.market_value for p in portfolio.positions.values()),
                })
                for sym in symbols:
                    pos = portfolio.get_position(sym)
                    per_symbol_equity[sym].append(pos.market_value)
                continue

            # 在策略开始时注册追踪止损
            if strategy_start is not None and not _trailing_registered and self._risk is not None:
                for sym, pos_in in portfolio.positions.items():
                    if not pos_in.is_empty and sym in bar_data:
                        atr_val = bar_data[sym].get(atr_col) if atr_col and atr_col in bar_data[sym].index else None
                        self._risk.register_trailing_stop(sym, bar_data[sym]['close'], atr_val)
                _trailing_registered = True

            # 风控: 止损/止盈检查
            if self._risk is not None:
                atr_values = {}
                if atr_col:
                    for sym in bar_data:
                        if atr_col in bar_data[sym].index:
                            atr_values[sym] = bar_data[sym][atr_col]
                exit_signals = self._risk.check_exits(portfolio, bar_data, atr_values)
                for exit_sig in exit_signals:
                    sym = exit_sig.symbol
                    if sym not in bar_data:
                        continue
                    pos = portfolio.get_position(sym)
                    if pos.is_empty:
                        continue
                    exec_price = bar_data[sym]['close'] * (1 - self._config.slippage)
                    quantity = pos.quantity
                    notional = exec_price * quantity
                    commission_fee = notional * self._config.commission
                    pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
                    portfolio.cash += (notional - commission_fee)
                    if open_trades.get(sym) is not None:
                        open_trades[sym].exit_time = t
                        open_trades[sym].exit_price = exec_price
                        open_trades[sym].pnl = pnl
                        open_trades[sym].commission += commission_fee
                        trades.append(open_trades[sym])
                        open_trades[sym] = None
                    if self._risk is not None:
                        self._risk.remove_trailing_stop(sym)
                    pos.quantity = 0.0
                    pos.avg_cost = 0.0
                    pos.market_value = 0.0

                portfolio.equity = portfolio.cash + sum(
                    p.market_value for p in portfolio.positions.values()
                )

            # 生成信号
            signals: List[Signal] = []
            if is_multi:
                hist = {}
                for sym in bar_data:
                    i_loc = data_with_ind[sym].index.get_loc(t) if t in data_with_ind[sym].index else None
                    if i_loc is not None:
                        hist[sym] = data_with_ind[sym].iloc[:i_loc + 1]
                    else:
                        hist[sym] = data_with_ind[sym]

                pctx = PortfolioContext(
                    bars=bar_data,
                    historical=hist,
                    portfolio=portfolio,
                    current_time=t,
                    bar=next(iter(bar_data.values())) if bar_data else None,
                    symbol=next(iter(bar_data.keys())) if bar_data else None,
                )
                signals = strategy.on_bar_multi(pctx)
            else:
                for sym in bar_data:
                    i_loc = data_with_ind[sym].index.get_loc(t) if t in data_with_ind[sym].index else None
                    if i_loc is None:
                        continue
                    ctx = Context(
                        bar=bar_data[sym],
                        bars=data_with_ind[sym].iloc[:i_loc + 1],
                        portfolio=portfolio,
                        current_time=t,
                        symbol=sym,
                    )
                    sig = strategy.on_bar(ctx)
                    if sig is not None:
                        signals.append(sig)

            # 风控: 过滤买入信号
            if self._risk is not None and signals:
                atr_values = {}
                if atr_col:
                    for sym in bar_data:
                        if atr_col in bar_data[sym].index:
                            atr_values[sym] = bar_data[sym][atr_col]
                signals = self._risk.check_signals(signals, portfolio, bar_data, atr_values, t)

            # 执行信号
            for sig in signals:
                sym = sig.symbol
                if sym not in bar_data:
                    continue
                pos = portfolio.get_position(sym)
                current_price = bar_data[sym]['close']

                if sig.type == SignalType.BUY and pos.is_empty:
                    exec_price = current_price * (1 + self._config.slippage)
                    alloc_pct = self._allocation.allocate(sym, sig, portfolio, exec_price, symbols)
                    raw_qty = self._sizer.compute_quantity(sig, portfolio, exec_price)
                    quantity = raw_qty * alloc_pct
                    if quantity <= 0:
                        continue

                    notional = exec_price * quantity
                    commission_fee = notional * self._config.commission

                    if notional + commission_fee > portfolio.cash:
                        quantity = portfolio.cash / (exec_price * (1 + self._config.commission))
                        if quantity <= 0:
                            continue
                        notional = exec_price * quantity
                        commission_fee = notional * self._config.commission

                    portfolio.cash -= (notional + commission_fee)
                    pos.quantity = quantity
                    pos.avg_cost = exec_price
                    pos.market_value = quantity * current_price

                    open_trades[sym] = TradeRecord(
                        symbol=sym,
                        entry_time=t,
                        exit_time=None,
                        entry_price=exec_price,
                        exit_price=None,
                        quantity=quantity,
                        commission=commission_fee,
                    )

                    if self._risk is not None:
                        atr_at_entry = bar_data[sym].get(atr_col) if atr_col and atr_col in bar_data[sym].index else None
                        self._risk.register_trailing_stop(sym, exec_price, atr_at_entry)

                elif sig.type == SignalType.SELL and not pos.is_empty:
                    exec_price = current_price * (1 - self._config.slippage)
                    quantity = pos.quantity
                    notional = exec_price * quantity
                    commission_fee = notional * self._config.commission
                    pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
                    portfolio.cash += (notional - commission_fee)

                    if open_trades.get(sym) is not None:
                        open_trades[sym].exit_time = t
                        open_trades[sym].exit_price = exec_price
                        open_trades[sym].pnl = pnl
                        open_trades[sym].commission += commission_fee
                        trades.append(open_trades[sym])
                        open_trades[sym] = None

                    if self._risk is not None:
                        self._risk.remove_trailing_stop(sym)
                    pos.quantity = 0.0
                    pos.avg_cost = 0.0
                    pos.market_value = 0.0

            # 再平衡
            if self._rebalance is not None and self._should_rebalance(t, portfolio, symbols, last_rebalance_date):
                self._execute_rebalance(portfolio, bar_data, symbols, t, trades, open_trades)
                last_rebalance_date = t

            # 更新权益
            portfolio.equity = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            equity_records.append({
                'time': t,
                'equity': portfolio.equity,
                'cash': portfolio.cash,
                'market_value': sum(p.market_value for p in portfolio.positions.values()),
            })

            for sym in symbols:
                pos = portfolio.get_position(sym)
                per_symbol_equity[sym].append(pos.market_value)

        # 5. on_finish
        if is_multi:
            pctx = PortfolioContext(
                bars=bar_data,
                historical={sym: data_with_ind[sym] for sym in symbols},
                portfolio=portfolio,
                current_time=all_indices[-1] if all_indices else pd.Timestamp.now(),
            )
            strategy.on_finish_multi(pctx)
        else:
            ctx = Context(
                bar=bar_data.get(first_symbol, data_with_ind[first_symbol].iloc[-1]),
                bars=data_with_ind[first_symbol],
                portfolio=portfolio,
                current_time=all_indices[-1] if all_indices else pd.Timestamp.now(),
                symbol=first_symbol,
            )
            strategy.on_finish(ctx)

        # 6. 计算绩效
        equity_df = pd.DataFrame(equity_records).set_index('time')
        equity_series = equity_df['equity']
        metrics = calc_performance_metrics(equity_series)

        closed_trades = [t for t in trades if t.exit_time is not None]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]
        total_trades = len(closed_trades)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 1.0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # 按标的分组交易
        per_sym_trades: Dict[str, List[TradeRecord]] = {sym: [] for sym in symbols}
        for tr in trades:
            if tr.symbol in per_sym_trades:
                per_sym_trades[tr.symbol].append(tr)

        per_sym_equity_series: Dict[str, pd.Series] = {}
        for sym in symbols:
            per_sym_equity_series[sym] = pd.Series(
                per_symbol_equity[sym],
                index=equity_df.index[:len(per_symbol_equity[sym])],
            )

        return BacktestResult(
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            max_drawdown_duration=metrics['max_drawdown_duration'],
            win_rate=win_rate,
            profit_loss_ratio=pl_ratio,
            total_trades=total_trades,
            equity_curve=equity_df,
            trades=trades,
            initial_capital=self._config.initial_capital,
            benchmark=self._config.benchmark,
            per_symbol_equity=per_sym_equity_series,
            per_symbol_trades=per_sym_trades,
        )

    def _should_rebalance(
        self,
        current_time: pd.Timestamp,
        portfolio: Portfolio,
        symbols: List[str],
        last_rebalance_date: Optional[pd.Timestamp],
    ) -> bool:
        """判断是否需要再平衡"""
        if self._rebalance is None:
            return False

        # 时间触发
        if self._rebalance.frequency is not None:
            if last_rebalance_date is None:
                return True
            freq_map = {'W': 'W', 'M': 'MS', 'Q': 'QS'}
            freq = freq_map.get(self._rebalance.frequency, self._rebalance.frequency)
            periods = pd.date_range(start=last_rebalance_date, end=current_time, freq=freq)
            if len(periods) > 1:
                return True

        # 漂移触发
        if self._rebalance.drift_threshold is not None and portfolio.equity > 0:
            for sym in symbols:
                pos = portfolio.get_position(sym)
                target_weight = self._allocation.allocate(sym, Signal(type=SignalType.BUY, symbol=sym), portfolio, 0, symbols)
                actual_weight = pos.market_value / portfolio.equity if portfolio.equity > 0 else 0
                if abs(actual_weight - target_weight) > self._rebalance.drift_threshold:
                    return True

        return False

    def _execute_rebalance(
        self,
        portfolio: Portfolio,
        bar_data: Dict[str, pd.Series],
        symbols: List[str],
        current_time: pd.Timestamp,
        trades: List[TradeRecord],
        open_trades: Dict[str, Optional[TradeRecord]],
    ) -> None:
        """执行再平衡：卖出所有持仓，按目标权重重新买入"""
        # 1. 卖出所有持仓
        for sym in symbols:
            pos = portfolio.get_position(sym)
            if pos.is_empty or sym not in bar_data:
                continue
            current_price = bar_data[sym]['close']
            exec_price = current_price * (1 - self._config.slippage)
            quantity = pos.quantity
            notional = exec_price * quantity
            commission_fee = notional * self._config.commission
            pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
            portfolio.cash += (notional - commission_fee)

            if self._rebalance.record_trades and open_trades.get(sym) is not None:
                open_trades[sym].exit_time = current_time
                open_trades[sym].exit_price = exec_price
                open_trades[sym].pnl = pnl
                open_trades[sym].commission += commission_fee
                trades.append(open_trades[sym])
                open_trades[sym] = None

            if self._risk is not None:
                self._risk.remove_trailing_stop(sym)
            pos.quantity = 0.0
            pos.avg_cost = 0.0
            pos.market_value = 0.0

        portfolio.equity = portfolio.cash

        # 2. 按目标权重重新买入
        for sym in symbols:
            if sym not in bar_data:
                continue
            current_price = bar_data[sym]['close']
            target_weight = self._allocation.allocate(sym, Signal(type=SignalType.BUY, symbol=sym), portfolio, current_price, symbols)
            if target_weight <= 0:
                continue

            target_amount = portfolio.equity * target_weight
            exec_price = current_price * (1 + self._config.slippage)
            quantity = target_amount / exec_price
            if quantity <= 0:
                continue

            notional = exec_price * quantity
            commission_fee = notional * self._config.commission

            if notional + commission_fee > portfolio.cash:
                quantity = portfolio.cash / (exec_price * (1 + self._config.commission))
                if quantity <= 0:
                    continue
                notional = exec_price * quantity
                commission_fee = notional * self._config.commission

            portfolio.cash -= (notional + commission_fee)
            pos = portfolio.get_position(sym)
            pos.quantity = quantity
            pos.avg_cost = exec_price
            pos.market_value = quantity * current_price

            if self._rebalance.record_trades:
                open_trades[sym] = TradeRecord(
                    symbol=sym,
                    entry_time=current_time,
                    exit_time=None,
                    entry_price=exec_price,
                    exit_price=None,
                    quantity=quantity,
                    commission=commission_fee,
                )

            if self._risk is not None:
                atr_val = bar_data[sym].get('atr_14') if 'atr_14' in bar_data[sym].index else None
                self._risk.register_trailing_stop(sym, exec_price, atr_val)

        portfolio.equity = portfolio.cash + sum(
            p.market_value for p in portfolio.positions.values()
        )
