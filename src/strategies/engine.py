"""向量化回测引擎"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from analysis.indicator_set import IndicatorSet
from .base import Strategy, Signal, SignalType, Context, Portfolio, Position
from .result import BacktestResult, TradeRecord, calc_performance_metrics
from .sizers import PositionSizer, FixedSizer


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission: float = 0.0003
    slippage: float = 0.0001
    benchmark: str = "000300.SH"
    initial_positions: Optional[Dict[str, Dict[str, float]]] = None  # {symbol: {"avg_cost": float, "quantity": float}}

    @classmethod
    def from_yaml(cls, path: str = "config/settings.yaml") -> 'BacktestConfig':
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            cfg = yaml.safe_load(f)
        bt = cfg.get('backtest', {})
        return cls(
            initial_capital=bt.get('initial_capital', 100000.0),
            commission=bt.get('commission', 0.0003),
            slippage=bt.get('slippage', 0.0001),
            benchmark=bt.get('benchmark', '000300.SH'),
        )


class BacktestEngine:
    """向量化回测引擎

    指标预计算(向量化) + 逐bar信号生成 + 组合状态跟踪
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        position_sizer: Optional[PositionSizer] = None,
        risk_manager: Optional['RiskManager'] = None,
    ):
        self._config = config or BacktestConfig()
        self._sizer = position_sizer or FixedSizer(percent=1.0)
        self._risk = risk_manager

    def run(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        strategy_start: Optional[pd.Timestamp] = None,
    ) -> BacktestResult:
        """运行回测

        Args:
            strategy: 回测策略
            df: OHLCV DataFrame (DatetimeIndex)
            symbol: 标的代码
            strategy_start: 策略开始日期，之前的bar只跟踪持仓市值不生成信号
        """
        from .risk import RiskManager

        if strategy_start is not None:
            strategy_start = pd.Timestamp(strategy_start)

        # 0. 重置风控状态
        if self._risk is not None:
            self._risk.reset()

        # 1. 初始化策略（先注册指标依赖）
        portfolio = Portfolio(
            cash=self._config.initial_capital,
            equity=self._config.initial_capital,
        )

        # 设置初始持仓
        open_trade: Optional[TradeRecord] = None
        if self._config.initial_positions:
            for sym, pos_spec in self._config.initial_positions.items():
                pos = portfolio.get_position(sym)
                pos.avg_cost = pos_spec['avg_cost']
                pos.quantity = pos_spec['quantity']
                portfolio.cash -= pos_spec['avg_cost'] * pos_spec['quantity']
                if sym == symbol:
                    open_trade = TradeRecord(
                        symbol=sym,
                        entry_time=df.index[0],
                        exit_time=None,
                        entry_price=pos_spec['avg_cost'],
                        exit_price=None,
                        quantity=pos_spec['quantity'],
                        commission=0.0,
                    )

        init_ctx = Context(
            bar=df.iloc[0],
            bars=df.iloc[:1],
            portfolio=portfolio,
            current_time=df.index[0],
            symbol=symbol,
        )
        strategy.on_init(init_ctx)

        # 2. 预计算指标（含风控所需ATR自动注册）
        iset = IndicatorSet()
        for spec in strategy.indicator_specs:
            iset.add(spec['name'], **{k: v for k, v in spec.items() if k != 'name'})

        if self._risk is not None and self._risk._config.needs_atr:
            atr_period = self._risk._config.atr_period
            atr_col = f'atr_{atr_period}'
            # 检查策略是否已注册 atr
            has_atr = any(
                s['name'] == 'atr' and s.get('period', 14) == atr_period
                for s in strategy.indicator_specs
            )
            if not has_atr:
                iset.add('atr', period=atr_period)

        df_with_ind = iset.compute(df) if iset.specs else df.copy()

        # 确定ATR列名
        atr_col = None
        if self._risk is not None and self._risk._config.needs_atr:
            atr_col = f'atr_{self._risk._config.atr_period}'

        # 3. 逐bar执行
        equity_records = []
        trades: List[TradeRecord] = []
        _trailing_registered = False

        for i in range(len(df_with_ind)):
            row = df_with_ind.iloc[i]
            current_time = pd.Timestamp(df_with_ind.index[i])

            # 更新持仓市值
            pos = portfolio.get_position(symbol)
            current_price = row['close']
            if not pos.is_empty:
                pos.market_value = pos.quantity * current_price

            portfolio.equity = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            # 策略开始前只跟踪持仓市值
            if strategy_start is not None and current_time < strategy_start:
                equity_records.append({
                    'time': current_time,
                    'equity': portfolio.equity,
                    'cash': portfolio.cash,
                    'market_value': pos.market_value,
                })
                continue

            # 在策略开始时注册追踪止损
            if strategy_start is not None and not _trailing_registered and self._risk is not None:
                for sym, pos_in in portfolio.positions.items():
                    if not pos_in.is_empty:
                        atr_at_entry = row[atr_col] if atr_col and atr_col in row.index else None
                        self._risk.register_trailing_stop(sym, current_price, atr_at_entry)
                _trailing_registered = True

            # 构建context
            ctx = Context(
                bar=row,
                bars=df_with_ind.iloc[:i + 1],
                portfolio=portfolio,
                current_time=current_time,
                symbol=symbol,
            )

            # 风控: 检查止损/止盈退出
            if self._risk is not None and not pos.is_empty:
                bar_data = {symbol: row}
                atr_val = {symbol: row[atr_col]} if atr_col and atr_col in row.index else {}
                exit_signals = self._risk.check_exits(portfolio, bar_data, atr_val)
                for exit_sig in exit_signals:
                    if exit_sig.type == SignalType.SELL and exit_sig.symbol == symbol:
                        # 执行风控卖出
                        exec_price = current_price * (1 - self._config.slippage)
                        quantity = pos.quantity
                        notional = exec_price * quantity
                        commission_fee = notional * self._config.commission
                        pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
                        portfolio.cash += (notional - commission_fee)
                        if open_trade is not None:
                            open_trade.exit_time = current_time
                            open_trade.exit_price = exec_price
                            open_trade.pnl = pnl
                            open_trade.commission += commission_fee
                            trades.append(open_trade)
                            open_trade = None
                        if self._risk is not None:
                            self._risk.remove_trailing_stop(symbol)
                        pos.quantity = 0.0
                        pos.avg_cost = 0.0
                        pos.market_value = 0.0
                        portfolio.equity = portfolio.cash

            # 重新获取 pos 状态（可能已被风控平仓）
            pos = portfolio.get_position(symbol)
            if not pos.is_empty:
                pos.market_value = pos.quantity * current_price
                portfolio.equity = portfolio.cash + sum(
                    p.market_value for p in portfolio.positions.values()
                )

            # 生成策略信号
            signal = strategy.on_bar(ctx)

            # 风控: 过滤买入信号
            if self._risk is not None and signal is not None and signal.type != SignalType.HOLD:
                bar_data = {symbol: row}
                atr_val = {symbol: row[atr_col]} if atr_col and atr_col in row.index else {}
                filtered = self._risk.check_signals([signal], portfolio, bar_data, atr_val, current_time)
                signal = filtered[0] if filtered else None

            if signal is None or signal.type == SignalType.HOLD:
                equity_records.append({
                    'time': current_time,
                    'equity': portfolio.equity,
                    'cash': portfolio.cash,
                    'market_value': pos.market_value,
                })
                continue

            # 执行交易
            if signal.type == SignalType.BUY and pos.is_empty:
                exec_price = current_price * (1 + self._config.slippage)
                quantity = self._sizer.compute_quantity(signal, portfolio, exec_price)
                if quantity <= 0:
                    equity_records.append({
                        'time': current_time,
                        'equity': portfolio.equity,
                        'cash': portfolio.cash,
                        'market_value': pos.market_value,
                    })
                    continue

                notional = exec_price * quantity
                commission_fee = notional * self._config.commission

                portfolio.cash -= (notional + commission_fee)
                pos.quantity = quantity
                pos.avg_cost = exec_price
                pos.market_value = quantity * current_price
                portfolio.equity = portfolio.cash + pos.market_value

                open_trade = TradeRecord(
                    symbol=symbol,
                    entry_time=current_time,
                    exit_time=None,
                    entry_price=exec_price,
                    exit_price=None,
                    quantity=quantity,
                    commission=commission_fee,
                )

                # 注册追踪止损
                if self._risk is not None:
                    atr_at_entry = row[atr_col] if atr_col and atr_col in row.index else None
                    self._risk.register_trailing_stop(symbol, exec_price, atr_at_entry)

            elif signal.type == SignalType.SELL and not pos.is_empty:
                exec_price = current_price * (1 - self._config.slippage)
                quantity = self._sizer.compute_quantity(signal, portfolio, exec_price)
                if quantity <= 0:
                    equity_records.append({
                        'time': current_time,
                        'equity': portfolio.equity,
                        'cash': portfolio.cash,
                        'market_value': pos.market_value,
                    })
                    continue

                quantity = min(quantity, pos.quantity)
                notional = exec_price * quantity
                commission_fee = notional * self._config.commission

                pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
                portfolio.cash += (notional - commission_fee)

                if open_trade is not None:
                    open_trade.exit_time = current_time
                    open_trade.exit_price = exec_price
                    open_trade.pnl = pnl
                    open_trade.commission += commission_fee
                    trades.append(open_trade)
                    open_trade = None

                if self._risk is not None:
                    self._risk.remove_trailing_stop(symbol)

                pos.quantity = 0.0
                pos.avg_cost = 0.0
                pos.market_value = 0.0
                portfolio.equity = portfolio.cash

            equity_records.append({
                'time': current_time,
                'equity': portfolio.equity,
                'cash': portfolio.cash,
                'market_value': pos.market_value,
            })

        # 4. on_finish
        ctx = init_ctx  # fallback if loop never entered strategy phase
        strategy.on_finish(ctx)

        # 5. 计算绩效
        equity_df = pd.DataFrame(equity_records).set_index('time')
        equity_series = equity_df['equity']
        metrics = calc_performance_metrics(equity_series)

        # 交易统计
        closed_trades = [t for t in trades if t.exit_time is not None]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]
        total_trades = len(closed_trades)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 1.0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

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
        )
