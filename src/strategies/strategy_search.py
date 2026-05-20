"""策略搜索模块：跨策略类型 + 参数联合搜索"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .optimizer import (
    Optimizer, GeneticOptimizer, BayesianOptimizer,
    ParamRange, OptimizationResult, results_to_dataframe,
)
from .engine import BacktestEngine, BacktestConfig
from .result import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class StrategySpec:
    name: str
    strategy_class: type
    param_ranges: List[ParamRange] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySearchResult:
    symbol: str
    best_strategy_name: str
    best_params: Dict[str, Any]
    best_objective: float
    best_backtest_result: Optional[BacktestResult] = None
    all_results: Optional[pd.DataFrame] = None


class StrategySearcher:
    """跨策略类型搜索：对每只标的遍历所有策略+参数，找最优组合"""

    def __init__(
        self,
        specs: List[StrategySpec],
        engine: Optional[BacktestEngine] = None,
        objective: Union[str, Callable] = 'sharpe',
        optimizer_type: str = 'grid',
        optimizer_kwargs: Optional[Dict] = None,
        top_n: int = 3,
    ):
        self._specs = specs
        self._engine = engine or BacktestEngine(BacktestConfig())
        self._objective = objective
        self._optimizer_type = optimizer_type
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._top_n = top_n

    def search_single(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> StrategySearchResult:
        best_result = None
        all_rows = []

        for spec in self._specs:
            opt_results = self._run_optimizer(spec, data, symbol)
            if not opt_results:
                continue
            top = opt_results[0]
            row = {
                'strategy': spec.name,
                'objective': top.objective_value,
                'params': top.params,
            }
            if top.backtest_result is not None:
                row['total_return'] = top.backtest_result.total_return
                row['sharpe_ratio'] = top.backtest_result.sharpe_ratio
                row['max_drawdown'] = top.backtest_result.max_drawdown
                row['total_trades'] = top.backtest_result.total_trades
            all_rows.append(row)

            if best_result is None or top.objective_value > best_result.objective_value:
                best_result = top

        if best_result is None:
            return StrategySearchResult(
                symbol=symbol, best_strategy_name='none',
                best_params={}, best_objective=0.0,
            )

        # 找到 best_result 对应的 spec name
        best_name = 'unknown'
        for row in all_rows:
            if row['objective'] == best_result.objective_value:
                best_name = row['strategy']
                break

        all_df = pd.DataFrame(all_rows) if all_rows else None
        return StrategySearchResult(
            symbol=symbol,
            best_strategy_name=best_name,
            best_params=best_result.params,
            best_objective=best_result.objective_value,
            best_backtest_result=best_result.backtest_result,
            all_results=all_df,
        )

    def search_multi(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        rows = []
        for sym, df in data.items():
            logger.info(f"搜索标的: {sym}")
            result = self.search_single(df, symbol=sym)
            row = {
                'symbol': sym,
                'best_strategy': result.best_strategy_name,
                'best_params': result.best_params,
                'objective': result.best_objective,
            }
            if result.best_backtest_result is not None:
                row['total_return'] = result.best_backtest_result.total_return
                row['sharpe_ratio'] = result.best_backtest_result.sharpe_ratio
                row['max_drawdown'] = result.best_backtest_result.max_drawdown
                row['total_trades'] = result.best_backtest_result.total_trades
            rows.append(row)

        return pd.DataFrame(rows)

    def search_matrix(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        matrix = {}
        for spec in self._specs:
            row = {}
            for sym, df in data.items():
                opt_results = self._run_optimizer(spec, df, sym)
                row[sym] = opt_results[0].objective_value if opt_results else None
            matrix[spec.name] = row

        return pd.DataFrame(matrix).T

    def _make_factory(self, spec: StrategySpec) -> Callable:
        def factory(params: Dict[str, Any]) -> Any:
            merged = {**spec.default_params, **params}
            return spec.strategy_class(**merged)
        return factory

    def _run_optimizer(
        self,
        spec: StrategySpec,
        data: pd.DataFrame,
        symbol: str,
    ) -> List[OptimizationResult]:
        factory = self._make_factory(spec)

        # 无可搜索参数时，直接跑一次
        if not spec.param_ranges:
            strategy = factory({})
            result = self._engine.run(strategy, data, symbol=symbol)
            obj_val = self._eval_objective(result)
            return [OptimizationResult(params={}, objective_value=obj_val, backtest_result=result)]

        if self._optimizer_type == 'genetic':
            opt = GeneticOptimizer(
                engine=self._engine, strategy_factory=factory,
                param_ranges=spec.param_ranges, objective=self._objective,
                top_n=self._top_n, **self._optimizer_kwargs,
            )
        elif self._optimizer_type == 'bayesian':
            opt = BayesianOptimizer(
                engine=self._engine, strategy_factory=factory,
                param_ranges=spec.param_ranges, objective=self._objective,
                top_n=self._top_n, **self._optimizer_kwargs,
            )
        else:
            opt = Optimizer(
                engine=self._engine, strategy_factory=factory,
                param_ranges=spec.param_ranges, objective=self._objective,
                top_n=self._top_n,
            )

        return opt.run(data, symbol=symbol)

    def _eval_objective(self, result: BacktestResult) -> float:
        if callable(self._objective):
            return self._objective(result)
        if self._objective == 'return':
            return result.total_return
        return result.sharpe_ratio


def default_strategy_specs() -> List[StrategySpec]:
    """内置12种策略的搜索配置"""
    from .examples.ma_cross import MACrossStrategy
    from .examples.rsi import RSIStrategy
    from .examples.macd import MACDStrategy
    from .examples.bollinger import BollingerStrategy
    from .examples.kdj import KDJStrategy
    from .examples.sar import SARStrategy
    from .examples.wr import WRStrategy
    from .examples.cci import CCIStrategy
    from .examples.keltner import KeltnerStrategy
    from .examples.obv import OBVStrategy
    from .examples.mfi import MFIStrategy
    from .examples.vwap import VWAPStrategy

    return [
        StrategySpec('ma', MACrossStrategy, [
            ParamRange('fast_period', start=3, stop=10, step=1),
            ParamRange('slow_period', start=15, stop=40, step=5),
        ]),
        StrategySpec('rsi', RSIStrategy, [
            ParamRange('period', start=7, stop=21, step=7),
            ParamRange('oversold', values=[20.0, 25.0, 30.0]),
            ParamRange('overbought', values=[70.0, 75.0, 80.0]),
        ]),
        StrategySpec('macd', MACDStrategy, [
            ParamRange('fast_period', start=10, stop=14, step=2),
            ParamRange('slow_period', start=24, stop=28, step=2),
            ParamRange('signal_period', start=7, stop=11, step=2),
        ]),
        StrategySpec('bollinger', BollingerStrategy, [
            ParamRange('period', values=[15, 20, 25]),
            ParamRange('num_std', values=[1.5, 2.0, 2.5]),
        ]),
        StrategySpec('kdj', KDJStrategy, [
            ParamRange('n', values=[7, 9, 12]),
            ParamRange('oversold', values=[15.0, 20.0, 25.0]),
            ParamRange('overbought', values=[75.0, 80.0, 85.0]),
        ]),
        StrategySpec('sar', SARStrategy),
        StrategySpec('wr', WRStrategy, [
            ParamRange('period', values=[10, 14, 20]),
            ParamRange('oversold', values=[-70.0, -80.0, -90.0]),
            ParamRange('overbought', values=[-10.0, -20.0, -30.0]),
        ]),
        StrategySpec('cci', CCIStrategy, [
            ParamRange('period', values=[10, 14, 20]),
            ParamRange('oversold', values=[-80.0, -100.0, -120.0]),
            ParamRange('overbought', values=[80.0, 100.0, 120.0]),
        ]),
        StrategySpec('keltner', KeltnerStrategy, [
            ParamRange('ema_period', values=[15, 20, 25]),
            ParamRange('num_atr', values=[1.0, 1.5, 2.0]),
        ]),
        StrategySpec('obv', OBVStrategy, [
            ParamRange('ma_period', values=[10, 20, 30]),
        ]),
        StrategySpec('mfi', MFIStrategy, [
            ParamRange('period', values=[10, 14, 20]),
            ParamRange('oversold', values=[15.0, 20.0, 25.0]),
            ParamRange('overbought', values=[75.0, 80.0, 85.0]),
        ]),
        StrategySpec('vwap', VWAPStrategy),
    ]


def parse_search_config(search_cfg: dict) -> List[StrategySpec]:
    """从 yaml 配置解析策略搜索规格"""
    defaults = {s.name: s for s in default_strategy_specs()}
    requested = search_cfg.get('strategies', list(defaults.keys()))
    custom_ranges = search_cfg.get('param_ranges', {})

    specs = []
    for name in requested:
        spec = defaults.get(name)
        if spec is None:
            logger.warning(f"未知策略: {name}, 跳过")
            continue
        # 覆盖自定义参数范围
        if name in custom_ranges:
            ranges = []
            for pname, pdef in custom_ranges[name].items():
                if isinstance(pdef, list):
                    ranges.append(ParamRange(pname, values=pdef))
                elif isinstance(pdef, dict):
                    ranges.append(ParamRange(
                        pname,
                        start=pdef.get('start'),
                        stop=pdef.get('stop'),
                        step=pdef.get('step'),
                    ))
            spec = StrategySpec(
                name=spec.name,
                strategy_class=spec.strategy_class,
                param_ranges=ranges,
                default_params=spec.default_params,
            )
        specs.append(spec)

    return specs
