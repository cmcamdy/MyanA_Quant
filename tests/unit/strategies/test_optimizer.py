"""参数优化器测试"""

import pytest
import pandas as pd
import numpy as np

from strategies.base import Strategy, Signal, SignalType, Context
from strategies.engine import BacktestEngine, BacktestConfig
from strategies.optimizer import (
    Optimizer, ParamRange, OptimizationResult,
    sharpe_objective, total_return_objective, results_to_dataframe,
    GeneticOptimizer, BayesianOptimizer,
)


def _make_df(n=30, start_price=10.0, drift=0.1):
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = [start_price + i * drift + np.sin(i / 3) * 0.5 for i in range(n)]
    return pd.DataFrame({
        'open': close,
        'high': [c * 1.02 for c in close],
        'low': [c * 0.98 for c in close],
        'close': close,
        'volume': [10000 + i * 100 for i in range(n)],
    }, index=dates)


class PeriodCrossStrategy(Strategy):
    """可配置周期的均线交叉策略"""

    def __init__(self, fast_period=5, slow_period=20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_init(self, ctx):
        self.register_indicator('ma', period=self.fast_period)
        self.register_indicator('ma', period=self.slow_period)

    def on_bar(self, ctx):
        fast_col = f'ma_{self.fast_period}'
        slow_col = f'ma_{self.slow_period}'
        ma_fast = ctx.bar.get(fast_col)
        ma_slow = ctx.bar.get(slow_col)
        if ma_fast is None or ma_slow is None or pd.isna(ma_fast) or pd.isna(ma_slow):
            return None
        if len(ctx.bars) < 2:
            return None
        prev_fast = ctx.bars[fast_col].iloc[-2]
        prev_slow = ctx.bars[slow_col].iloc[-2]
        if pd.isna(prev_fast) or pd.isna(prev_slow):
            return None
        if prev_fast <= prev_slow and ma_fast > ma_slow:
            return Signal(type=SignalType.BUY, symbol=ctx.symbol)
        if prev_fast >= prev_slow and ma_fast < ma_slow:
            return Signal(type=SignalType.SELL, symbol=ctx.symbol)
        return None


class TestParamRange:

    def test_discrete_values(self):
        pr = ParamRange(name='x', values=[1, 3, 5])
        assert pr.iter_values() == [1, 3, 5]

    def test_range_with_step(self):
        pr = ParamRange(name='x', start=1.0, stop=3.0, step=1.0)
        vals = pr.iter_values()
        assert vals[0] == 1.0
        assert vals[-1] == 3.0

    def test_integer_range(self):
        pr = ParamRange(name='x', start=1, stop=5, step=2)
        assert pr.iter_values() == [1, 3, 5]

    def test_missing_spec_raises(self):
        pr = ParamRange(name='x')
        with pytest.raises(ValueError):
            pr.iter_values()


class TestOptimizationResult:

    def test_sorting(self):
        r1 = OptimizationResult(params={'a': 1}, objective_value=0.5)
        r2 = OptimizationResult(params={'a': 2}, objective_value=1.0)
        results = sorted([r1, r2], key=lambda r: r.objective_value, reverse=True)
        assert results[0].objective_value == 1.0


class TestOptimizer:

    def test_single_param_grid(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5, 7])],
            objective='sharpe',
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) == 3
        assert results[0].objective_value >= results[-1].objective_value

    def test_multi_param_cartesian_product(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(
            fast_period=p['fast_period'],
            slow_period=p['slow_period'],
        )
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[
                ParamRange(name='fast_period', values=[3, 5]),
                ParamRange(name='slow_period', values=[10, 20]),
            ],
            objective='return',
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) == 4  # 2x2 笛卡尔积

    def test_sharpe_objective(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[5])],
            objective='sharpe',
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) == 1

    def test_custom_objective(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[5])],
            objective=lambda r: -r.max_drawdown,  # 最小化回撤
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) == 1

    def test_top_n_retains_backtest_result(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5, 7])],
            objective='sharpe',
            top_n=1,
        )
        results = optimizer.run(df, symbol="TEST")
        assert results[0].backtest_result is not None
        assert results[1].backtest_result is None
        assert results[2].backtest_result is None

    def test_results_sorted_descending(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5, 7])],
            objective='sharpe',
        )
        results = optimizer.run(df, symbol="TEST")
        for i in range(len(results) - 1):
            assert results[i].objective_value >= results[i + 1].objective_value

    def test_results_to_dataframe(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = Optimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5])],
            objective='sharpe',
        )
        results = optimizer.run(df, symbol="TEST")
        rdf = results_to_dataframe(results)
        assert isinstance(rdf, pd.DataFrame)
        assert 'fast_period' in rdf.columns
        assert 'objective' in rdf.columns
        assert len(rdf) == 2


class TestGeneticOptimizer:

    def test_finds_good_params(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(
            fast_period=p['fast_period'], slow_period=p['slow_period'],
        )
        optimizer = GeneticOptimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[
                ParamRange(name='fast_period', values=[3, 5, 7]),
                ParamRange(name='slow_period', values=[10, 20]),
            ],
            objective='sharpe',
            population_size=10,
            generations=5,
            seed=42,
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) > 0
        assert results[0].objective_value >= results[-1].objective_value

    def test_custom_objective(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = GeneticOptimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5, 7])],
            objective='return',
            population_size=6,
            generations=3,
            seed=42,
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) > 0

    def test_deduplication(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = GeneticOptimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5])],
            objective='sharpe',
            population_size=6,
            generations=3,
            seed=42,
        )
        results = optimizer.run(df, symbol="TEST")
        # No duplicate parameter sets
        param_tuples = [tuple(sorted(r.params.items())) for r in results]
        assert len(param_tuples) == len(set(param_tuples))

    def test_top_n_retains_backtest_result(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(
            fast_period=p['fast_period'], slow_period=p['slow_period'],
        )
        optimizer = GeneticOptimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[
                ParamRange(name='fast_period', values=[3, 5, 7]),
                ParamRange(name='slow_period', values=[10, 20]),
            ],
            objective='sharpe',
            top_n=1,
            population_size=6,
            generations=3,
            seed=42,
        )
        results = optimizer.run(df, symbol="TEST")
        assert results[0].backtest_result is not None


class TestBayesianOptimizer:

    def test_finds_params(self):
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(
            fast_period=p['fast_period'], slow_period=p['slow_period'],
        )
        optimizer = BayesianOptimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[
                ParamRange(name='fast_period', values=[3, 5, 7]),
                ParamRange(name='slow_period', values=[10, 20]),
            ],
            objective='sharpe',
            n_initial=3,
            n_iterations=3,
            seed=42,
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) > 0
        assert results[0].objective_value >= results[-1].objective_value

    def test_random_fallback(self):
        """BayesianOptimizer should work even without sklearn"""
        df = _make_df(30)
        engine = BacktestEngine(BacktestConfig(commission=0, slippage=0))
        factory = lambda p: PeriodCrossStrategy(fast_period=p['fast_period'], slow_period=20)
        optimizer = BayesianOptimizer(
            engine=engine,
            strategy_factory=factory,
            param_ranges=[ParamRange(name='fast_period', values=[3, 5, 7])],
            objective='sharpe',
            n_initial=2,
            n_iterations=2,
            seed=42,
        )
        results = optimizer.run(df, symbol="TEST")
        assert len(results) > 0
