"""实时监控引擎"""

import logging
import os
import signal as sig_module
import time
import uuid
from typing import Dict, List, Optional, Union

import pandas as pd

from analysis.indicator_set import IndicatorSet
from data.manager import DataManager
from strategies.base import Strategy, Signal, SignalType, Context, Portfolio, Position
from strategies.portfolio_engine import (
    MultiStrategy, PortfolioContext, EqualWeightAllocation, CustomAllocation,
)
from strategies.result import TradeRecord, calc_performance_metrics
from strategies.risk import RiskConfig, RiskManager
from strategies.sizers import PositionSizer, FixedSizer

from .alerts import AlertManager, LoggingCallback
from .config import MonitorConfig
from .dashboard import ConsoleDashboard
from .poller import DataPoller
from .state import MonitorState

logger = logging.getLogger(__name__)


class LiveEngine:
    """实时策略监控引擎

    复用 Strategy/MultiStrategy ABC，逐 bar 处理实时数据，
    支持终端仪表盘、告警回调、状态持久化。
    """

    def __init__(
        self,
        config: MonitorConfig,
        strategy: Union[Strategy, MultiStrategy],
        position_sizer: Optional[PositionSizer] = None,
        risk_config: Optional[RiskConfig] = None,
        allocation: Optional[Union[EqualWeightAllocation, CustomAllocation]] = None,
    ):
        self._config = config
        self._strategy = strategy
        self._sizer = position_sizer or FixedSizer(percent=1.0)
        self._risk_config = risk_config
        self._risk_manager = RiskManager(risk_config) if risk_config else None
        self._allocation = allocation or EqualWeightAllocation()
        self._is_multi = isinstance(strategy, MultiStrategy)

        # 运行时状态
        self._portfolio: Optional[Portfolio] = None
        self._open_trades: Dict[str, Optional[TradeRecord]] = {}
        self._state: Optional[MonitorState] = None
        self._poller: Optional[DataPoller] = None
        self._dashboard: Optional[ConsoleDashboard] = None
        self._alert_manager: Optional[AlertManager] = None
        self._historical_data: Dict[str, pd.DataFrame] = {}
        self._indicator_specs = strategy.indicator_specs
        self._atr_col: Optional[str] = None
        self._running = False
        self._monitor_id = config.symbols[0] if config.symbols else str(uuid.uuid4())[:8]
        self._poll_count = 0
        self._last_poll_time: Optional[pd.Timestamp] = None
        self._start_time: float = 0.0
        self._recent_signals: List[Dict] = []
        self._last_known: Dict[str, float] = {}

        # ATR 自动注册
        if self._risk_manager is not None and self._risk_config is not None and self._risk_config.needs_atr:
            self._atr_col = f"atr_{self._risk_config.atr_period}"

    def start(self, resume: bool = True) -> None:
        """启动监控主循环"""
        self._initialize(resume)
        self._running = True
        self._start_time = time.time()

        original_sigint = sig_module.getsignal(sig_module.SIGINT)

        def _handle_shutdown(signum, frame):
            logger.info("Received shutdown signal")
            self._running = False

        sig_module.signal(sig_module.SIGINT, _handle_shutdown)

        try:
            while self._running:
                loop_start = time.time()
                try:
                    new_data = self._poller.poll()
                    self._poll_count += 1
                    self._last_poll_time = pd.Timestamp.now()

                    if new_data:
                        self._process_new_bars(new_data)

                    if self._dashboard.should_draw():
                        self._draw_dashboard()

                    self._save_state()
                except Exception as e:
                    logger.error("Poll cycle error: %s", e, exc_info=True)

                elapsed = time.time() - loop_start
                sleep_time = max(0, self._config.poll_interval - elapsed)
                time.sleep(sleep_time)
        finally:
            sig_module.signal(sig_module.SIGINT, original_sigint)
            self._save_state()
            if self._dashboard:
                self._dashboard.clear()
            print(f"\nMonitor stopped. State saved to {self._state_path()}")

    def stop(self) -> None:
        """停止监控"""
        self._running = False

    # ─── 初始化 ───

    def _initialize(self, resume: bool) -> None:
        """初始化所有组件"""
        data_manager = DataManager(data_dir=self._config.data_dir)
        symbols = self._config.symbols

        # 尝试恢复状态
        state_path = self._state_path()
        if resume and os.path.exists(state_path):
            try:
                self._state = MonitorState.load(state_path)
                self._portfolio = self._state.to_portfolio()
                if self._risk_manager is not None:
                    self._state.apply_risk_state(self._risk_manager)
                # 恢复 open_trades
                for sym, tr_dict in self._state.open_trades.items():
                    if tr_dict:
                        self._open_trades[sym] = TradeRecord(
                            symbol=tr_dict["symbol"],
                            entry_time=pd.Timestamp(tr_dict["entry_time"]),
                            exit_time=pd.Timestamp(tr_dict["exit_time"]) if tr_dict.get("exit_time") else None,
                            entry_price=tr_dict["entry_price"],
                            exit_price=tr_dict.get("exit_price"),
                            quantity=tr_dict["quantity"],
                            pnl=tr_dict.get("pnl", 0.0),
                            commission=tr_dict.get("commission", 0.0),
                        )
                # 恢复历史数据
                self._historical_data = {}
                for sym in symbols:
                    self._historical_data[sym] = pd.DataFrame()
                logger.info("Resumed monitor state from %s", state_path)
            except Exception as e:
                logger.warning("Failed to resume state: %s, starting fresh", e)
                self._state = None

        if self._state is None:
            # 新建状态
            self._portfolio = Portfolio(
                cash=self._config.initial_capital,
                equity=self._config.initial_capital,
            )
            # 设置初始持仓
            if self._config.initial_positions:
                for sym, pos_spec in self._config.initial_positions.items():
                    pos = self._portfolio.get_position(sym)
                    pos.avg_cost = pos_spec["avg_cost"]
                    pos.quantity = pos_spec["quantity"]
                    pos.market_value = pos_spec["avg_cost"] * pos_spec["quantity"]
                    self._portfolio.cash -= pos.market_value
                    self._open_trades[sym] = TradeRecord(
                        symbol=sym,
                        entry_time=pd.Timestamp.now(),
                        exit_time=None,
                        entry_price=pos_spec["avg_cost"],
                        exit_price=None,
                        quantity=pos_spec["quantity"],
                    )

            self._state = MonitorState.from_portfolio(self._portfolio, self._monitor_id)
            self._historical_data = {sym: pd.DataFrame() for sym in symbols}

        # 预热：加载历史数据
        self._poller = DataPoller(
            data_manager=data_manager,
            symbols=symbols,
            freq=self._config.freq,
            poll_interval=self._config.poll_interval,
            source=self._config.data_source,
            adjust=self._config.adjust,
        )
        for sym in symbols:
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            start = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            df = self._poller.warmup(sym, start, end)
            if df is not None:
                self._historical_data[sym] = self._compute_indicators(sym, df)
                # 更新 last_known
                if not df.empty:
                    self._last_known[sym] = df["close"].iloc[-1]
            # 恢复 last_processed
            if self._state and sym in self._state.last_processed_time:
                self._poller.set_last_processed(sym, pd.Timestamp(self._state.last_processed_time[sym]))

        # 策略初始化
        self._init_strategy()

        # 仪表盘和告警
        self._dashboard = ConsoleDashboard(refresh_interval=self._config.dashboard_refresh)
        self._alert_manager = AlertManager(
            on_signal=self._config.alert_on_signal,
            on_risk_trigger=self._config.alert_on_risk_trigger,
            equity_change_threshold=self._config.alert_on_equity_change_pct,
            callbacks=[LoggingCallback()],
        )

    def _init_strategy(self) -> None:
        """调用策略 on_init"""
        symbols = self._config.symbols
        first_sym = symbols[0] if symbols else "UNKNOWN"
        first_hist = self._historical_data.get(first_sym, pd.DataFrame())

        if self._is_multi:
            bars = {}
            for sym in symbols:
                hist = self._historical_data.get(sym, pd.DataFrame())
                if not hist.empty:
                    bars[sym] = hist.iloc[-1]
            pctx = PortfolioContext(
                bars=bars,
                historical={sym: self._historical_data.get(sym, pd.DataFrame()) for sym in symbols},
                portfolio=self._portfolio,
                current_time=pd.Timestamp.now(),
                bar=bars.get(first_sym),
                symbol=first_sym,
            )
            self._strategy.on_init_multi(pctx)
        else:
            if not first_hist.empty:
                ctx = Context(
                    bar=first_hist.iloc[-1],
                    bars=first_hist,
                    portfolio=self._portfolio,
                    current_time=first_hist.index[-1] if len(first_hist) > 0 else pd.Timestamp.now(),
                    symbol=first_sym,
                )
                self._strategy.on_init(ctx)

    def _compute_indicators(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标并追加到历史数据"""
        if not self._indicator_specs:
            return df.copy()

        iset = IndicatorSet()
        for spec in self._indicator_specs:
            iset.add(spec["name"], **{k: v for k, v in spec.items() if k != "name"})

        # ATR 自动注册
        if self._atr_col and self._atr_col not in df.columns:
            iset.add("atr", period=self._risk_config.atr_period)

        if iset.specs:
            return iset.compute(df)
        return df.copy()

    # ─── 核心处理 ───

    def _process_new_bars(self, new_data: Dict[str, pd.DataFrame]) -> None:
        """处理所有新增 bar（时间对齐）"""
        # 更新历史数据
        for sym, df in new_data.items():
            if sym in self._historical_data:
                df_with_ind = self._compute_indicators(sym, df)
                self._historical_data[sym] = pd.concat(
                    [self._historical_data[sym], df_with_ind]
                ).drop_duplicates()
            else:
                self._historical_data[sym] = self._compute_indicators(sym, df)

        # 时间对齐：取所有新 bar 的时间并集
        all_times = sorted(set().union(
            *(df.index for df in new_data.values())
        ))

        symbols = self._config.symbols
        for t in all_times:
            t = pd.Timestamp(t)
            self._process_single_bar(t, symbols, new_data)

    def _process_single_bar(
        self,
        current_time: pd.Timestamp,
        symbols: List[str],
        new_data: Dict[str, pd.DataFrame],
    ) -> None:
        """处理单个时间点的所有标的"""
        # 收集本 bar 有数据的标的
        bar_data: Dict[str, pd.Series] = {}
        for sym in symbols:
            hist = self._historical_data.get(sym, pd.DataFrame())
            if current_time in hist.index:
                row = hist.loc[current_time]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                bar_data[sym] = row
                self._last_known[sym] = row["close"]

        if not bar_data:
            return

        # 1. 更新持仓市值
        for sym in symbols:
            pos = self._portfolio.get_position(sym)
            if not pos.is_empty:
                if sym in bar_data:
                    pos.market_value = pos.quantity * bar_data[sym]["close"]
                elif sym in self._last_known:
                    pos.market_value = pos.quantity * self._last_known[sym]

        self._portfolio.equity = self._portfolio.cash + sum(
            p.market_value for p in self._portfolio.positions.values()
        )

        # 2. 风控止损/止盈检查
        if self._risk_manager is not None:
            atr_values = self._get_atr_values(bar_data)
            exit_signals = self._risk_manager.check_exits(self._portfolio, bar_data, atr_values)
            for exit_sig in exit_signals:
                self._execute_risk_exit(exit_sig, bar_data, current_time)
                self._alert_manager.on_risk_triggered(exit_sig, current_time)

            self._portfolio.equity = self._portfolio.cash + sum(
                p.market_value for p in self._portfolio.positions.values()
            )

        # 3. 策略信号生成
        signals: List[Signal] = []
        if self._is_multi:
            hist = {}
            for sym in bar_data:
                sym_hist = self._historical_data.get(sym, pd.DataFrame())
                if current_time in sym_hist.index:
                    i_loc = sym_hist.index.get_loc(current_time)
                    hist[sym] = sym_hist.iloc[: i_loc + 1]
                else:
                    hist[sym] = sym_hist

            pctx = PortfolioContext(
                bars=bar_data,
                historical=hist,
                portfolio=self._portfolio,
                current_time=current_time,
                bar=next(iter(bar_data.values())),
                symbol=next(iter(bar_data.keys())),
            )
            try:
                signals = self._strategy.on_bar_multi(pctx) or []
            except Exception as e:
                logger.error("Strategy on_bar_multi error: %s", e)
                signals = []
        else:
            for sym in bar_data:
                sym_hist = self._historical_data.get(sym, pd.DataFrame())
                if current_time not in sym_hist.index:
                    continue
                i_loc = sym_hist.index.get_loc(current_time)
                ctx = Context(
                    bar=bar_data[sym],
                    bars=sym_hist.iloc[: i_loc + 1],
                    portfolio=self._portfolio,
                    current_time=current_time,
                    symbol=sym,
                )
                try:
                    s = self._strategy.on_bar(ctx)
                    if s is not None:
                        signals.append(s)
                except Exception as e:
                    logger.error("Strategy on_bar error for %s: %s", sym, e)

        # 4. 风控信号过滤
        if self._risk_manager is not None and signals:
            atr_values = self._get_atr_values(bar_data)
            signals = self._risk_manager.check_signals(
                signals, self._portfolio, bar_data, atr_values, current_time
            )

        # 5. 执行信号
        for sig in signals:
            self._execute_signal(sig, bar_data, current_time)
            self._alert_manager.on_signal_fired(sig, current_time)
            self._recent_signals.append({
                "time": current_time.strftime("%H:%M"),
                "symbol": sig.symbol,
                "type": sig.type.value,
                "reason": sig.reason,
            })
        # 保留最近20条信号
        self._recent_signals = self._recent_signals[-20:]

        # 6. 记录权益
        self._portfolio.equity = self._portfolio.cash + sum(
            p.market_value for p in self._portfolio.positions.values()
        )
        self._state.equity_history.append({
            "time": current_time.isoformat(),
            "equity": self._portfolio.equity,
            "cash": self._portfolio.cash,
            "market_value": sum(p.market_value for p in self._portfolio.positions.values()),
        })
        self._alert_manager.on_equity_change(self._portfolio.equity, current_time)

        # 7. 更新状态
        self._update_state_from_runtime()

    def _execute_signal(
        self,
        signal: Signal,
        bar_data: Dict[str, pd.Series],
        current_time: pd.Timestamp,
    ) -> None:
        """执行交易信号"""
        sym = signal.symbol
        if sym not in bar_data:
            return
        pos = self._portfolio.get_position(sym)
        current_price = bar_data[sym]["close"]

        if signal.type == SignalType.BUY and pos.is_empty:
            exec_price = current_price * (1 + self._config.slippage)
            alloc_pct = self._allocation.allocate(
                sym, signal, self._portfolio, exec_price, self._config.symbols
            )
            raw_qty = self._sizer.compute_quantity(signal, self._portfolio, exec_price)
            quantity = raw_qty * alloc_pct
            if quantity <= 0:
                return

            notional = exec_price * quantity
            commission_fee = notional * self._config.commission

            if notional + commission_fee > self._portfolio.cash:
                quantity = self._portfolio.cash / (exec_price * (1 + self._config.commission))
                if quantity <= 0:
                    return
                notional = exec_price * quantity
                commission_fee = notional * self._config.commission

            self._portfolio.cash -= notional + commission_fee
            pos.quantity = quantity
            pos.avg_cost = exec_price
            pos.market_value = quantity * current_price

            self._open_trades[sym] = TradeRecord(
                symbol=sym,
                entry_time=current_time,
                exit_time=None,
                entry_price=exec_price,
                exit_price=None,
                quantity=quantity,
                commission=commission_fee,
            )

            if self._risk_manager is not None:
                atr_at_entry = bar_data[sym].get(self._atr_col) if self._atr_col and self._atr_col in bar_data[sym].index else None
                self._risk_manager.register_trailing_stop(sym, exec_price, atr_at_entry)

        elif signal.type == SignalType.SELL and not pos.is_empty:
            exec_price = current_price * (1 - self._config.slippage)
            quantity = pos.quantity
            notional = exec_price * quantity
            commission_fee = notional * self._config.commission
            pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
            self._portfolio.cash += notional - commission_fee

            if self._open_trades.get(sym) is not None:
                self._open_trades[sym].exit_time = current_time
                self._open_trades[sym].exit_price = exec_price
                self._open_trades[sym].pnl = pnl
                self._open_trades[sym].commission += commission_fee
                self._state.trade_history.append(self._trade_to_dict(self._open_trades[sym]))
                self._alert_manager.on_trade_executed(self._open_trades[sym], current_time)
                self._open_trades[sym] = None

            if self._risk_manager is not None:
                self._risk_manager.remove_trailing_stop(sym)
            pos.quantity = 0.0
            pos.avg_cost = 0.0
            pos.market_value = 0.0

    def _execute_risk_exit(
        self,
        signal: Signal,
        bar_data: Dict[str, pd.Series],
        current_time: pd.Timestamp,
    ) -> None:
        """执行风控卖出"""
        sym = signal.symbol
        if sym not in bar_data:
            return
        pos = self._portfolio.get_position(sym)
        if pos.is_empty:
            return

        current_price = bar_data[sym]["close"]
        exec_price = current_price * (1 - self._config.slippage)
        quantity = pos.quantity
        notional = exec_price * quantity
        commission_fee = notional * self._config.commission
        pnl = (exec_price - pos.avg_cost) * quantity - commission_fee
        self._portfolio.cash += notional - commission_fee

        if self._open_trades.get(sym) is not None:
            self._open_trades[sym].exit_time = current_time
            self._open_trades[sym].exit_price = exec_price
            self._open_trades[sym].pnl = pnl
            self._open_trades[sym].commission += commission_fee
            self._state.trade_history.append(self._trade_to_dict(self._open_trades[sym]))
            self._alert_manager.on_trade_executed(self._open_trades[sym], current_time)
            self._open_trades[sym] = None

        if self._risk_manager is not None:
            self._risk_manager.remove_trailing_stop(sym)
        pos.quantity = 0.0
        pos.avg_cost = 0.0
        pos.market_value = 0.0

    def _get_atr_values(self, bar_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """从 bar 数据中提取 ATR 值"""
        if not self._atr_col:
            return {}
        atr_values = {}
        for sym, bar in bar_data.items():
            if self._atr_col in bar.index:
                val = bar[self._atr_col]
                if pd.notna(val):
                    atr_values[sym] = float(val)
        return atr_values

    # ─── 状态管理 ───

    def _update_state_from_runtime(self) -> None:
        """从运行时对象更新状态"""
        self._state.cash = self._portfolio.cash
        self._state.positions = {}
        for sym, pos in self._portfolio.positions.items():
            if not pos.is_empty:
                self._state.positions[sym] = {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                }
        # open_trades
        self._state.open_trades = {}
        for sym, tr in self._open_trades.items():
            if tr is not None:
                self._state.open_trades[sym] = self._trade_to_dict(tr)
        # last_processed
        for sym in self._config.symbols:
            hist = self._historical_data.get(sym, pd.DataFrame())
            if not hist.empty:
                self._state.last_processed_time[sym] = hist.index[-1].isoformat()
        # RiskManager state
        if self._risk_manager is not None:
            self._state.extract_risk_state(self._risk_manager)

    def _save_state(self) -> None:
        """保存状态到文件"""
        self._update_state_from_runtime()
        self._state.save(self._state_path())

    def _state_path(self) -> str:
        return os.path.join(self._config.state_dir, f"{self._monitor_id}.json")

    @staticmethod
    def _trade_to_dict(tr: TradeRecord) -> dict:
        return {
            "symbol": tr.symbol,
            "entry_time": tr.entry_time.isoformat() if tr.entry_time else None,
            "exit_time": tr.exit_time.isoformat() if tr.exit_time else None,
            "entry_price": tr.entry_price,
            "exit_price": tr.exit_price,
            "quantity": tr.quantity,
            "pnl": tr.pnl,
            "commission": tr.commission,
        }

    # ─── 仪表盘 ───

    def _draw_dashboard(self) -> None:
        """绘制终端仪表盘"""
        # 构造 positions_detail
        positions_detail = {}
        for sym in self._config.symbols:
            pos = self._portfolio.get_position(sym)
            if not pos.is_empty:
                price = self._last_known.get(sym, pos.avg_cost)
                positions_detail[sym] = {"current_price": price}

        # 构造 recent_alerts
        recent_alerts = []
        if self._alert_manager:
            for evt in self._alert_manager.get_recent_events(3):
                recent_alerts.append({
                    "time": evt.timestamp.strftime("%H:%M"),
                    "symbol": evt.symbol,
                    "details": evt.details,
                })

        self._dashboard.draw(
            strategy_name=type(self._strategy).__name__,
            portfolio=self._portfolio,
            positions_detail=positions_detail,
            recent_signals=self._recent_signals,
            recent_alerts=recent_alerts,
            equity_history=self._state.equity_history,
            config=self._config,
            poll_count=self._poll_count,
            last_poll_time=self._last_poll_time,
            uptime_seconds=time.time() - self._start_time if self._start_time else 0,
        )
